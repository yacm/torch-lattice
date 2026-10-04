"""HMC clover con fermiones dinámicos (2 sabores degenerados, Wilson-clover).

Acción de gauge: Symanzik tree-level tadpole-improved (Lüscher-Weisz).
Fermiones: clover (Sheikholeslami-Wohlert) sobre enlaces stout-smeared (1 iter, rho).

La fuerza fluye por autograd a través de U -> stout(U) -> D_clover -> S
(el smearing es diferenciable vía matrix_exp de Q hermitiana sin traza).

HMC EXACTO: chi = (D_clover^dag D_clover)^{-1} phi se RE-RESUELVE en cada
evaluación de la fuerza. Mantener chi fijo durante la trayectoria es el viejo
phi-algorithm (fuerza inexacta, violación de energía O(eps), no O(eps^2)).

Parámetros del action HadStruc/NME: c_sw = 1/u0^3 (tree-level tadpole), rho=0.1.
Aquí c_sw y u0 se pasan explícitos; u0 = <plaqueta>^(1/4) se fija antes del run.
"""
import math

import torch

from . import clover, dirac, gauge, invert, stout
from .fermion_hmc import sample_momentum
from .hmc import kinetic, make_leaf, project_tah


def _d(U, x, m0, c_sw):
    return clover.clover_dslash(U, x, m0, c_sw)


def _d_dagger(U, x, m0, c_sw):
    # D^dag = gamma5 D gamma5 (gamma5-hermiticidad del Dslash clover)
    return dirac.apply_gamma5(_d(U, dirac.apply_gamma5(x), m0, c_sw))


def _smeared(U, rho):
    return stout.stout_smear(U, rho=rho, n_iter=1)


def sample_phi(U, m0, c_sw, rho, generator=None):
    """phi = D_clover^dag eta, eta ~ N(0,1): phi^dag (D^dag D)^{-1} phi = |eta|^2."""
    shape = U[0].shape[:4]
    dev = U[0].device
    re = torch.randn(*shape, 4, 3, dtype=torch.float64, device=dev, generator=generator)
    im = torch.randn(*shape, 4, 3, dtype=torch.float64, device=dev, generator=generator)
    eta = ((re + 1j * im) / math.sqrt(2)).to(torch.complex128)
    Us = _smeared([u.detach() for u in U], rho)
    return _d_dagger(Us, eta, m0, c_sw)


def solve_chi(U, phi, m0, c_sw, rho, max_iter=5000, tol=1e-12):
    """chi = (D_clover^dag D_clover)^{-1} phi, desacoplado (detached)."""
    Us = _smeared([u.detach() for u in U], rho)
    phi_d = phi.detach()

    def normal(x):
        return _d_dagger(Us, _d(Us, x, m0, c_sw), m0, c_sw)

    chi, it, resid = invert.cg(normal, phi_d, max_iter=max_iter, tol=tol)
    return chi.detach(), it, resid


def pseudofermion_action_value(phi, chi):
    """S_PF = phi^dag chi = ||D chi||^2 (valor real para Metropolis)."""
    return (phi.conj() * chi).sum().real


def force_total(U, beta, phi, m0, c_sw, rho, u0):
    """Fuerza molecular total (gauge Symanzik + fermiónica clover).

    Re-resuelve chi en el gauge actual y construye S = S_gauge - ||D chi||^2
    cuyo gradiente es F_total = dS_gauge/dU + dS_PF/dU (chi tratado constante).
    """
    chi, _, _ = solve_chi(U, phi, m0, c_sw, rho)
    Us = _smeared(U, rho)  # diferenciable (U son hojas)
    S = gauge.symanzik_action(U, beta, u0)
    Dchi = _d(Us, chi, m0, c_sw)
    S = S - (Dchi.conj() * Dchi).sum().real
    g = torch.autograd.grad(S, U, create_graph=False)
    F = []
    for g_mu, u_mu in zip(g, U):
        M = 0.5 * (u_mu @ g_mu.mH - g_mu @ u_mu.mH)
        F.append(project_tah(M))
    return F


def gauge_force(U, beta, u0):
    """Fuerza de gauge Symanzik pura (sin fermiones), proyectada a su(3)."""
    S = gauge.symanzik_action(U, beta, u0)
    g = torch.autograd.grad(S, U, create_graph=False)
    F = []
    for g_mu, u_mu in zip(g, U):
        M = 0.5 * (u_mu @ g_mu.mH - g_mu @ u_mu.mH)
        F.append(project_tah(M))
    return F


def hamiltonian_value(U, beta, P, phi, chi, m0, c_sw, rho, u0):
    return (kinetic(P).detach()
            + gauge.symanzik_action(U, beta, u0).detach()
            + pseudofermion_action_value(phi, chi).detach())


def leapfrog(U, P, beta, phi, m0, c_sw, rho, u0, n_steps, eps):
    """Trayectoria MD (HMC exacto: chi se re-resuelve en cada paso)."""
    U = [u.clone() for u in U]
    P = [p.clone() for p in P]
    F = force_total(U, beta, phi, m0, c_sw, rho, u0)
    P = [p + 0.5 * eps * f for p, f in zip(P, F)]
    for step in range(n_steps):
        U = [make_leaf(torch.matrix_exp(eps * p) @ u) for p, u in zip(P, U)]
        F = force_total(U, beta, phi, m0, c_sw, rho, u0)
        if step < n_steps - 1:
            P = [p + eps * f for p, f in zip(P, F)]
        else:
            P = [p + 0.5 * eps * f for p, f in zip(P, F)]
    return U, P


def hmc_step(U, beta, m0, c_sw, rho, u0, n_steps, eps, generator=None):
    """Un paso HMC exacto con 2 sabores clover. Devuelve (U, aceptado, deltaH, plaqueto, n_CG)."""
    shape = U[0].shape[:4]
    P = sample_momentum(shape, generator=generator)
    phi = sample_phi(U, m0, c_sw, rho, generator=generator)
    chi, n_cg0, _ = solve_chi(U, phi, m0, c_sw, rho)
    H0 = hamiltonian_value(U, beta, P, phi, chi, m0, c_sw, rho, u0).item()

    U_new, P_new = leapfrog(U, P, beta, phi, m0, c_sw, rho, u0, n_steps, eps)
    chi_new, n_cg1, _ = solve_chi(U_new, phi, m0, c_sw, rho)
    H1 = hamiltonian_value(U_new, beta, P_new, phi, chi_new, m0, c_sw, rho, u0).item()

    deltaH = H1 - H0
    accept = deltaH < 0 or torch.rand(1, generator=generator).item() < math.exp(-deltaH)
    if accept:
        U = [make_leaf(u) for u in U_new]
    plaq = gauge.average_plaquette(U)["w_plaq"]
    return U, accept, deltaH, plaq, n_cg0 + n_cg1 + n_steps


def thermalize(U0, beta, m0, c_sw, rho, u0, n_traj, n_steps, eps, generator=None):
    """Corre n_traj trayectorias HMC clover. Devuelve (U_final, historial)."""
    U = [make_leaf(u) for u in U0]
    hist = []
    for _ in range(n_traj):
        U, acc, dH, plaq, n_cg = hmc_step(U, beta, m0, c_sw, rho, u0, n_steps, eps,
                                          generator=generator)
        hist.append((acc, dH, plaq, n_cg))
    return U, hist
