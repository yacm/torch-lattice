"""Stout smearing (Morningstar-Peardon 2004) para suavizar enlaces de gauge.

U_mu -> exp(i Q_mu) U_mu, con Q_mu la parte hermitiana SIN TRAZA de
Omega = C U_mu^dag, donde C es la suma de los 6 staples que contienen a U_mu.

Diferenciable (autograd) para que la fuerza del HMC fluya a través del smearing
(dS/dU = dS/dU~ · dU~/dU, y dU~/dU sale de la exponencial de matriz).
"""
import torch

from . import config


def _stout_step(U, rho):
    bo = U[0].dim() - 6  # offset de batch (dims líderes)
    eye = torch.eye(3, dtype=U[0].dtype, device=U[0].device)
    U_new = []
    for mu in range(config.Nd):
        axis = bo + mu
        C = torch.zeros_like(U[mu])
        for nu in range(config.Nd):
            if nu == mu:
                continue
            ax = bo + nu
            U_nu = U[nu]
            # staple forward: U_nu(x) U_mu(x+nu) U_nu^dag(x+mu)
            U_mu_fwd = torch.roll(U[mu], -1, dims=ax)
            U_nu_fwd = torch.roll(U_nu, -1, dims=axis)
            s1 = U_nu @ U_mu_fwd @ U_nu_fwd.mH
            # staple backward: U_nu^dag(x-nu) U_mu(x-nu) U_nu(x-nu+mu)
            U_nu_bwd = torch.roll(U_nu, +1, dims=ax).mH
            U_mu_bwd = torch.roll(U[mu], +1, dims=ax)
            U_nu_bwd_fwd = torch.roll(torch.roll(U_nu, +1, dims=ax), -1, dims=axis)
            s2 = U_nu_bwd @ U_mu_bwd @ U_nu_bwd_fwd
            C = C + s1 + s2
        Omega = (rho * C) @ U[mu].mH
        A = 0.5j * (Omega.mH - Omega)  # hermitiana
        trA = torch.einsum("...ii->...", A).real
        Q = A - (trA / 3.0)[..., None, None] * eye  # hermitiana sin traza
        U_new.append(torch.linalg.matrix_exp(1j * Q) @ U[mu])
    return U_new


def stout_smear(U, rho=0.1, n_iter=1):
    """Aplica n_iter pasos de stout smearing con peso rho."""
    U = [u.clone() for u in U]
    for _ in range(n_iter):
        U = _stout_step(U, rho)
    return U
