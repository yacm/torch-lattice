"""HMC con fermiones dinámicos (2 sabores degenerados, Wilson) — HMC EXACTO.

La fuerza fermiónica sale de autograd SIN retropropagar por el solver:
si chi = (D^dag D)^{-1} phi, entonces

    dS_PF = d[ phi^dag (D^dag D)^{-1} phi ] = -chi^dag d(D^dag D) chi
          = d[ -||D chi||^2 ]            (con chi tratado como constante)

El grafo de autograd solo atraviesa la aplicación LOCAL y dispersa de D, no las
iteraciones del CG. Memoria O(V), no O(V * n_iter).

IMPORTANTE (HMC exacto vs phi-algorithm): chi se RE-RESUELVE en cada evaluación
de la fuerza usando el gauge ACTUAL. Mantener chi fijo durante la trayectoria es
el viejo phi-algorithm (fuerza inexacta -> violación de energía O(eps), no O(eps^2)).
Aquí `force_total` llama a `solve_chi` internamente en cada paso.

Todo en doble precisión (complex128). Retícula diminuta (512 sitios -> fermión
~98 KB), así que RAM no es problema SIEMPRE que se use el Dslash disperso y
nunca `dirac.dense_matrix` (6144x6144 ~ 600 MB).
"""
import math

import torch

from . import config, dirac, gauge, invert, su3
from .hmc import make_leaf, project_tah, kinetic, validate_integrator_parameters


class LinearSolveError(RuntimeError):
    """CG no produjo una solución finita dentro de la tolerancia solicitada."""

    def __init__(self, iterations, residual, tolerance):
        super().__init__(
            f"CG no convergió: iter={iterations}, resid={residual:.3e}, tol={tolerance:.3e}"
        )
        self.iterations = int(iterations)
        self.residual = float(residual)
        self.tolerance = float(tolerance)


def sample_momentum(shape, dtype=torch.complex128, dev=None, generator=None):
    """Momento canónico P_mu = i sum_a c_a T_a, c_a ~ N(0,2), en su(3).

    La varianza 2 corresponde a exp(-K) para K=-1/2 Tr(P^2) y T=lambda/2.
    """
    dev = dev if dev is not None else config.device()
    Ts = su3.generators(dtype, dev)
    P = []
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    for _ in range(config.Nd):
        p = torch.zeros(*shape, 3, 3, dtype=dtype, device=dev)
        for T in Ts:
            c = math.sqrt(2.0) * torch.randn(
                *shape, dtype=real_dtype, device=dev, generator=generator
            )
            p = p + (1j * c)[..., None, None] * T
        P.append(p)
    return P


def sample_phi(U, m0, generator=None):
    """Muestrea el campo de pseudofermiones phi = D^dag eta, eta ~ N(0,1) complejo.

    Así phi^dag (D^dag D)^{-1} phi = eta^dag eta = |eta|^2 (Gaussiana estándar),
    la medida correcta para 2 sabores degenerados.
    """
    shape = U[0].shape[:4]
    dev = U[0].device
    complex_dtype = U[0].dtype
    if complex_dtype not in (torch.complex64, torch.complex128):
        raise ValueError("U debe usar complex64 o complex128")
    real_dtype = torch.float32 if complex_dtype == torch.complex64 else torch.float64
    re = torch.randn(*shape, 4, 3, dtype=real_dtype, device=dev, generator=generator)
    im = torch.randn(*shape, 4, 3, dtype=real_dtype, device=dev, generator=generator)
    eta = ((re + 1j * im) / math.sqrt(2)).to(complex_dtype)
    U_d = [u.detach() for u in U]
    return invert._d_dagger(U_d, eta, m0)


def solve_chi(U, phi, m0, max_iter=5000, tol=1e-12, check=True):
    """Resuelve (D†D)χ=φ; por defecto falla cerrado si CG no converge."""
    U_d = [u.detach() for u in U]
    phi_d = phi.detach()

    def normal(x):
        return invert._d_dagger(U_d, invert._d(U_d, x, m0), m0)

    chi, it, resid = invert.cg(normal, phi_d, max_iter=max_iter, tol=tol)
    resid_value = float(resid)
    if check and (not math.isfinite(resid_value) or resid_value > tol):
        raise LinearSolveError(it, resid_value, tol)
    return chi.detach(), it, resid


def pseudofermion_action_value(phi, chi):
    """S_PF = phi^dag chi = ||D chi||^2 (valor real para el test de Metropolis)."""
    return (phi.conj() * chi).sum().real


def force_total(U, beta, phi, m0, max_iter=5000, tol=1e-12):
    """Fuerza molecular total (gauge + fermiónica), HMC exacto.

    Re-resuelve chi = (D^dag D)^{-1} phi en el gauge ACTUAL, y construye
    S = S_gauge - ||D chi||^2 cuyo gradiente es F_total = dS_gauge/dU + dS_PF/dU.
    """
    chi, _, _ = solve_chi(U, phi, m0, max_iter=max_iter, tol=tol)
    S = gauge.wilson_action(U, beta)
    Dchi = dirac.wilson_dslash(U, chi, m0)
    S = S - (Dchi.conj() * Dchi).sum().real
    g = torch.autograd.grad(S, U, create_graph=False)
    F = []
    for g_mu, u_mu in zip(g, U):
        M = 0.5 * (u_mu @ g_mu.mH - g_mu @ u_mu.mH)
        F.append(project_tah(M))
    return F


def hamiltonian_value(U, beta, P, phi, chi, m0):
    """Valor de H para Metropolis: K + S_gauge + S_PF (sin grafo)."""
    return (kinetic(P).detach()
            + gauge.wilson_action(U, beta).detach()
            + pseudofermion_action_value(phi, chi).detach())


def leapfrog_fermion(U, P, beta, phi, m0, n_steps, eps, force_fn=None,
                     max_iter=5000, tol=1e-12):
    """Trayectoria MD con fermiones (HMC exacto: chi se re-resuelve cada paso).

    force_fn(U, beta, phi, m0) -> lista de fuerzas F_μ proyectadas a su(3).
    Por defecto force_total (autograd); pasar force_total_analytic para la
    derivada a mano (bit-compatible, sin grafo de autograd).
    """
    validate_integrator_parameters(n_steps, eps)
    if force_fn is None:
        def active_force(field, b, pseudofermion, mass):
            return force_total(
                field, b, pseudofermion, mass, max_iter=max_iter, tol=tol
            )
    else:
        active_force = force_fn
    U = [u.clone() for u in U]
    P = [p.clone() for p in P]
    F = active_force(U, beta, phi, m0)
    P = [p + 0.5 * eps * f for p, f in zip(P, F)]
    for step in range(n_steps):
        U = [make_leaf(torch.matrix_exp(eps * p) @ u) for p, u in zip(P, U)]
        F = active_force(U, beta, phi, m0)
        if step < n_steps - 1:
            P = [p + eps * f for p, f in zip(P, F)]
        else:
            P = [p + 0.5 * eps * f for p, f in zip(P, F)]
    return U, P


def hmc_step_fermion(U, beta, m0, n_steps, eps, generator=None, force_fn=None,
                     max_iter=5000, tol=1e-12):
    """Un paso HMC exacto con 2 sabores de Wilson.

    Todo fallo/no convergencia del CG rechaza determinísticamente la propuesta.
    Devuelve (U, aceptado, deltaH, plaqueto, iter_CG_totales).
    """
    validate_integrator_parameters(n_steps, eps)
    shape = U[0].shape[:4]
    P = sample_momentum(
        shape, dtype=U[0].dtype, dev=U[0].device, generator=generator
    )
    phi = sample_phi(U, m0, generator=generator)
    cg_total = 0
    try:
        chi, n_solve, _ = solve_chi(
            U, phi, m0, max_iter=max_iter, tol=tol
        )
        cg_total += n_solve
        H0 = hamiltonian_value(U, beta, P, phi, chi, m0).item()
        U_new, P_new = leapfrog_fermion(
            U, P, beta, phi, m0, n_steps, eps, force_fn=force_fn,
            max_iter=max_iter, tol=tol,
        )
        chi_new, n_solve, _ = solve_chi(
            U_new, phi, m0, max_iter=max_iter, tol=tol
        )
        cg_total += n_solve
        H1 = hamiltonian_value(U_new, beta, P_new, phi, chi_new, m0).item()
    except LinearSolveError as failure:
        plaq = gauge.average_plaquette(U)["w_plaq"]
        return U, False, math.inf, plaq, failure.iterations

    deltaH = H1 - H0
    if not math.isfinite(deltaH):
        accept = False
        deltaH = math.inf
    else:
        accept = deltaH < 0
        if not accept:
            draw = torch.rand((), device=U[0].device, generator=generator).item()
            accept = draw < math.exp(-deltaH)
    if accept:
        U = [make_leaf(u) for u in U_new]
    plaq = gauge.average_plaquette(U)["w_plaq"]
    return U, accept, deltaH, plaq, cg_total
