"""Operador de Dirac de Wilson (Dslash) en JAX, con APBC en tiempo."""
import jax.numpy as jnp

from . import core


def _flip(u, axis, tslice):
    idx = [slice(None)] * u.ndim
    idx[axis] = tslice
    return u.at[tuple(idx)].multiply(-1.0)


def dslash(U, psi, m0, apbc=True):
    """D_w psi = (4+m0)psi - 0.5 Σ[(1-γ_μ)U_μ(x)psi(x+μ)+(1+γ_μ)U_μ†(x-μ)psi(x-μ)].

    U: lista de 4 tensores (..., Nx,Ny,Nz,Nt,3,3). psi: (..., Nx,Ny,Nz,Nt,4,3).
    Sin batch (dims líderes = retícula). apbc: condición anti-periódica en t.
    """
    fact = jnp.asarray(4.0 + m0, dtype=psi.dtype)
    out = fact * psi
    for mu in range(core.Nd):
        om = (core.EYE4 - core.GAMMAS[mu]).astype(psi.dtype)
        op = (core.EYE4 + core.GAMMAS[mu]).astype(psi.dtype)
        # forward: (1-γ_μ) U_μ(x) psi(x+μ)
        psi_fwd = jnp.roll(psi, -1, axis=mu)
        U_fwd = _flip(U[mu], mu, -1) if (apbc and mu == core.t_dir) else U[mu]
        tmp_fwd = jnp.matmul(psi_fwd, jnp.swapaxes(U_fwd, -1, -2))
        out = out - 0.5 * jnp.matmul(om, tmp_fwd)
        # backward: (1+γ_μ) U_μ†(x-μ) psi(x-μ)
        psi_bwd = jnp.roll(psi, 1, axis=mu)
        U_bwd = jnp.roll(core.mH(U[mu]), 1, axis=mu)
        U_bwd = _flip(U_bwd, mu, 0) if (apbc and mu == core.t_dir) else U_bwd
        tmp_bwd = jnp.matmul(psi_bwd, jnp.swapaxes(U_bwd, -1, -2))
        out = out - 0.5 * jnp.matmul(op, tmp_bwd)
    return out


def apply_gamma5(psi):
    """gamma5 = diag(1,1,-1,-1) en base DeGrand-Rossi → flip de signo, no matmul."""
    sign = jnp.array([1.0, 1.0, -1.0, -1.0], dtype=psi.dtype)
    return psi * sign[:, None]


def d_dagger(U, x, m0, apbc=True):
    """D† = (4+m0)x - 0.5 Σ[(1+γ_μ)U_μ(x)x(x+μ) + (1-γ_μ)U_μ†(x-μ)x(x-μ)].

    = γ5 D γ5 (γ5-hermiticidad), pero SIN envolver en γ5: solo se intercambian
    los proyectores (1±γ). Elimina 2 pases de memoria de flip de signo por aplicación.
    """
    fact = jnp.asarray(4.0 + m0, dtype=x.dtype)
    out = fact * x
    for mu in range(core.Nd):
        op = (core.EYE4 + core.GAMMAS[mu]).astype(x.dtype)  # forward: (1+γ_μ)
        om = (core.EYE4 - core.GAMMAS[mu]).astype(x.dtype)  # backward: (1-γ_μ)
        psi_fwd = jnp.roll(x, -1, axis=mu)
        U_fwd = _flip(U[mu], mu, -1) if (apbc and mu == core.t_dir) else U[mu]
        tmp_fwd = jnp.matmul(psi_fwd, jnp.swapaxes(U_fwd, -1, -2))
        out = out - 0.5 * jnp.matmul(op, tmp_fwd)
        psi_bwd = jnp.roll(x, 1, axis=mu)
        U_bwd = jnp.roll(core.mH(U[mu]), 1, axis=mu)
        U_bwd = _flip(U_bwd, mu, 0) if (apbc and mu == core.t_dir) else U_bwd
        tmp_bwd = jnp.matmul(psi_bwd, jnp.swapaxes(U_bwd, -1, -2))
        out = out - 0.5 * jnp.matmul(om, tmp_bwd)
    return out
