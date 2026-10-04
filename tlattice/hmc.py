"""HMC de gauge puro (Wilson) con fuerzas calculadas por autograd."""
import math

import torch

from . import gauge, config, su3


def validate_integrator_parameters(n_steps, eps):
    if not isinstance(n_steps, int) or isinstance(n_steps, bool) or n_steps < 1:
        raise ValueError("n_steps debe ser un entero >= 1")
    eps_value = float(eps)
    if not math.isfinite(eps_value) or eps_value <= 0.0:
        raise ValueError("eps debe ser finito y positivo")


def make_leaf(t):
    """Convierte un tensor en hoja (leaf) que requiere gradiente."""
    return t.detach().requires_grad_(True)


def project_tah(X):
    """Proyección al álgebra su(3): parte anti-hermítica sin traza."""
    ah = 0.5 * (X - X.mH)
    tr = torch.einsum("...ii->...", ah)
    eye = torch.eye(3, dtype=X.dtype, device=X.device)
    return ah - (tr / config.Nc)[..., None, None] * eye


def force(U, beta):
    """Fuerza molecular F_mu = P_TAH( U_mu g_mu^dag - g_mu U_mu^dag ).

    U debe ser una lista de tensores leaf con requires_grad=True.
    g_mu = dS/dU_mu (gradiente de autograd). torch incluye un factor 2 en su
    gradiente complejo (dS = Re Tr[g^dag dU]), por eso el 1/2 explícito aquí:
    F_mu = (1/2) P_TAH( U_mu g_mu^dag - g_mu U_mu^dag ).
    """
    S = gauge.wilson_action(U, beta)
    g = torch.autograd.grad(S, U, create_graph=False)
    F = []
    for g_mu, u_mu in zip(g, U):
        M = 0.5 * (u_mu @ g_mu.mH - g_mu @ u_mu.mH)
        F.append(project_tah(M))
    return F


def sample_momentum(shape, dtype=torch.complex64, dev=None, generator=None):
    """Momento canónico P_mu = i sum_a c_a T_a, c_a ~ N(0,2), en su(3).

    La varianza 2 corresponde a exp(-K) para K=-1/2 Tr(P^2) y T=lambda/2.
    El dtype real se deriva del dtype complejo para no mezclar precisiones.
    """
    if dtype not in (torch.complex64, torch.complex128):
        raise ValueError("dtype debe ser torch.complex64 o torch.complex128")
    dev = dev if dev is not None else config.device()
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    Ts = su3.generators(dtype=dtype, dev=dev)
    P = []
    for _ in range(config.Nd):
        p = torch.zeros(*shape, 3, 3, dtype=dtype, device=dev)
        for T in Ts:
            c = math.sqrt(2.0) * torch.randn(
                *shape, dtype=real_dtype, device=dev, generator=generator
            )
            p = p + (1j * c)[..., None, None] * T
        P.append(p)
    return P


def kinetic(P):
    """K = -(1/2) sum Tr(P^2), positivo para P anti-hermítico."""
    return -0.5 * sum(torch.einsum("...ij,...ji->...", p, p).real.sum() for p in P)


def hamiltonian(U, beta, P):
    return kinetic(P) + gauge.wilson_action(U, beta)


def leapfrog(U, P, beta, n_steps, eps):
    """Una trayectoria de dinámica molecular (leapfrog). Devuelve (U', P')."""
    validate_integrator_parameters(n_steps, eps)
    U = [u.clone() for u in U]
    P = [p.clone() for p in P]
    F = force(U, beta)
    P = [p + 0.5 * eps * f for p, f in zip(P, F)]  # medio paso inicial de momento
    for step in range(n_steps):
        U = [make_leaf(torch.matrix_exp(eps * p) @ u) for p, u in zip(P, U)]
        F = force(U, beta)
        if step < n_steps - 1:
            P = [p + eps * f for p, f in zip(P, F)]  # paso completo (salvo el último)
        else:
            P = [p + 0.5 * eps * f for p, f in zip(P, F)]  # medio paso final
    return U, P


def hmc_step(U, beta, n_steps, eps, generator=None):
    """Un paso completo de HMC: propone y acepta/rechaza por Metropolis.

    Devuelve (U_nuevo, aceptado, deltaH, plaqueto_actual).
    """
    validate_integrator_parameters(n_steps, eps)
    P = sample_momentum(
        U[0].shape[:4], dtype=U[0].dtype, dev=U[0].device, generator=generator
    )
    H0 = hamiltonian(U, beta, P).item()
    U_new, P_new = leapfrog(U, P, beta, n_steps, eps)
    H1 = hamiltonian(U_new, beta, P_new).item()
    deltaH = H1 - H0
    draw = torch.rand((), device=U[0].device, generator=generator).item()
    accept = deltaH < 0 or draw < math.exp(-deltaH)
    if accept:
        U = [make_leaf(u) for u in U_new]
    plaq = gauge.average_plaquette(U)["w_plaq"]
    return U, accept, deltaH, plaq
