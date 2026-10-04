"""Fuerzas analíticas (sin autograd) para el HMC: gauge (staple) + fermiónica (dsdu).

Convención idéntica a autograd: g = ∂S/∂U  con  δS = Re Tr[g† δU].
Fuerza molecular final (igual que el código):  F_μ = P_TA[ ½(U_μ g_μ† − g_μ U_μ†) ].

Fuerza fermiónica = derivada analítica exacta (ver derivación LaTeX):
    ψ = D χ,   χ = (D†D)⁻¹ φ
    g_PF_μ = Λ_f† + Λ_b
      Λ_f(x) = tr_spin[(1−γ_μ) χ(x+μ̂) ψ(x)†]
      Λ_b(x) = tr_spin[(1+γ_μ) χ(x) ψ(x+μ̂)†]

Fuerza gauge (Wilson):
    g_μ(x) = −(β/Nc) Σ_{ν≠μ} [ S_f(x)† + S_b(x) ]
      S_f(x) = U_ν(x+μ̂) U_μ†(x+ν̂) U_ν†(x)        (staple forward)
      S_b(x) = U_ν†(x−ν̂) U_μ(x−ν̂) U_ν(x+μ̂−ν̂)     (staple backward)
"""
import torch

from . import config, dirac, gauge, invert
from .fermion_hmc import solve_chi
from .hmc import project_tah


def gauge_force_analytic(U, beta):
    """g_gauge_μ = ∂S_wilson/∂U_μ (convención δS = Re Tr[g† δU])."""
    U = [u.detach() for u in U]
    Nc = config.Nc
    U_mH = [u.mH for u in U]
    F = []
    for mu in range(config.Nd):
        g = torch.zeros_like(U[mu])
        for nu in range(config.Nd):
            if nu == mu:
                continue
            U_nu_f = torch.roll(U[nu], -1, dims=mu)       # U_nu(x+mu)
            # staple forward: U_nu(x+mu) U_mu†(x+nu) U_nu†(x)
            S_f = U_nu_f @ torch.roll(U_mH[mu], -1, dims=nu) @ U_mH[nu]
            # staple backward: U_nu†(x-nu) U_mu(x-nu) U_nu(x+mu-nu)
            S_b = (torch.roll(U_mH[nu], +1, dims=nu)
                   @ torch.roll(U[mu], +1, dims=nu)
                   @ torch.roll(U_nu_f, +1, dims=nu))
            g = g - (beta / Nc) * (S_f.mH + S_b)
        F.append(g)
    return F


def fermion_force_analytic(U, chi, m0, boundary="antiperiodic"):
    """g_PF_μ = ∂S_PF/∂U_μ = Λ_f† + Λ_b  (ψ = D χ)."""
    U = [u.detach() for u in U]
    chi = chi.detach()
    dev = U[0].device
    dtype = U[0].dtype

    # APBC: horneamos el −1 en el enlace temporal de la frontera (U_eff = −U_t en x_t=Nt−1)
    U_eff = [u.clone() for u in U]
    if boundary == "antiperiodic":
        U_eff[config.t_dir][..., -1, :, :] *= -1

    # ψ = D[U_eff] χ  (periódico, el flip ya está en U_eff)
    psi = dirac.wilson_dslash(U_eff, chi, m0, boundary="periodic")
    gs = config.gammas(dtype=dtype, dev=dev)
    eye = torch.eye(4, dtype=dtype, device=dev)

    F = []
    for mu in range(config.Nd):
        om = eye - gs[mu]            # 1 − γ_μ
        op = eye + gs[mu]            # 1 + γ_μ
        chi_fwd = torch.roll(chi, -1, dims=mu)   # χ(x+μ̂)
        psi_fwd = torch.roll(psi, -1, dims=mu)   # ψ(x+μ̂)

        # Λ_f = tr_spin[(1−γ)χ(x+μ̂) ⊗ ψ(x)†]
        chi_til_f = torch.matmul(om, chi_fwd)
        Lambda_f = chi_til_f.transpose(-1, -2) @ psi.conj()
        # Λ_b = tr_spin[(1+γ)χ(x) ⊗ ψ(x+μ̂)†]
        chi_til_b = torch.matmul(op, chi)
        Lambda_b = chi_til_b.transpose(-1, -2) @ psi_fwd.conj()

        g = Lambda_f.mH + Lambda_b
        # APBC: la fuerza física sobre U_t(x_t=Nt−1) lleva −1 (U_eff = −U_t ahí)
        if boundary == "antiperiodic" and mu == config.t_dir:
            g = g.clone()
            g[..., -1, :, :] *= -1
        F.append(g)
    return F


def force_total_analytic(U, beta, phi, m0):
    """Fuerza molecular total analítica (gauge + fermiónica), HMC exacto.

    Equivalente a `force_total` (autograd) pero sin construir grafo de autograd.
    Re-resuelve χ = (D†D)⁻¹φ en el gauge actual (HMC exacto).
    Devuelve la lista F_μ ya proyectada a su(3).
    """
    chi, _, _ = solve_chi(U, phi, m0)
    g_gauge = gauge_force_analytic(U, beta)
    g_pf = fermion_force_analytic(U, chi, m0)
    F = []
    for gg, gp, u in zip(g_gauge, g_pf, U):
        u_d = u.detach()
        g = gg + gp
        M = 0.5 * (u_d @ g.mH - g @ u_d.mH)
        F.append(project_tah(M))
    return F
