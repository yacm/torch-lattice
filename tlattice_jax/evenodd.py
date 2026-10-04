"""Precondicionamiento par-impar (even-odd) del operador de Wilson, en JAX.

Idea (complemento de Schur): pintando la retícula en sitios pares/impares, el
operador de Dirac D = (4+m0) - 0.5 H tiene bloques diagonales triviales (4+m0)·1
y hopping H sólo fuera de la diagonal (par<->impar). Eliminando los impares
analíticamente queda el operador reducido sobre pares:

    D̂ = (4+m0) - (0.25/(4+m0)) · H²        (γ5-hermítico: γ5 D̂ γ5 = D̂†)

Resolver en los pares da ~la mitad de sitios y ~la mitad de iteraciones de CG.
"""
import numpy as np

import jax.numpy as jnp

from . import core, dirac


def _validate_even_shape(shape):
    """La retícula periódica solo es bipartita si todas sus extensiones son pares."""
    if len(shape) != core.Nd or any(int(length) <= 0 for length in shape):
        raise ValueError(f"shape debe tener {core.Nd} extensiones positivas: {shape}")
    if any(int(length) % 2 for length in shape):
        raise ValueError(
            "El precondicionamiento par-impar periódico requiere que todas "
            f"las extensiones sean pares; recibido {tuple(shape)}"
        )


def checkerboard(shape):
    """Índices planos de sitios pares/impares para shape=(Nx,Ny,Nz,Nt).

    Paridad = (x+y+z+t) mod 2. Devuelve (even_idx, odd_idx), cada uno de largo
    V/2, como índices planos (row-major) sobre la retícula aplanada.

    Se computa en numpy (concreto) para que el indexado resultante sea válido
    dentro de jax.jit (los índices son constantes en tiempo de trazado).
    """
    _validate_even_shape(shape)
    Nx, Ny, Nz, Nt = shape
    x = np.arange(Nx)
    y = np.arange(Ny)
    z = np.arange(Nz)
    t = np.arange(Nt)
    X, Y, Z, T = np.meshgrid(x, y, z, t, indexing="ij")
    parity = (X + Y + Z + T) % 2
    flat = np.arange(Nx * Ny * Nz * Nt).reshape(Nx, Ny, Nz, Nt)
    return jnp.asarray(flat[parity == 0]), jnp.asarray(flat[parity == 1])


def split_eo(psi, even_idx, odd_idx):
    """psi (...,Nx,Ny,Nz,Nt,4,3) -> (psi_e, psi_o) compactos (V/2,4,3)."""
    V = np.prod(psi.shape[-5:-1])
    flat = psi.reshape(-1, 4, 3)
    return flat[even_idx], flat[odd_idx]


def recombine(psi_e, psi_o, even_idx, odd_idx, shape):
    """Inverso de split_eo: (psi_e, psi_o) -> (Nx,Ny,Nz,Nt,4,3)."""
    V = int(np.prod(shape))
    flat = jnp.zeros((V, 4, 3), dtype=psi_e.dtype)
    flat = flat.at[even_idx].set(psi_e).at[odd_idx].set(psi_o)
    return flat.reshape(*shape, 4, 3)


def hop(U, psi, apbc=True):
    """Hopping H ψ = Σ_μ [(1-γμ)U_μ(x)ψ(x+μ) + (1+γμ)U_μ†(x-μ)ψ(x-μ)].

    Es exactamente el dslash SIN la diagonal (4+m0) y SIN el factor -0.5:
    dslash(U, ψ, m0) = (4+m0)ψ - 0.5·hop(U, ψ).
    """
    out = jnp.zeros_like(psi)
    for mu in range(core.Nd):
        om = (core.EYE4 - core.GAMMAS[mu]).astype(psi.dtype)
        op = (core.EYE4 + core.GAMMAS[mu]).astype(psi.dtype)
        psi_fwd = jnp.roll(psi, -1, axis=mu)
        U_fwd = dirac._flip(U[mu], mu, -1) if (apbc and mu == core.t_dir) else U[mu]
        out = out + jnp.matmul(om, jnp.matmul(psi_fwd, jnp.swapaxes(U_fwd, -1, -2)))
        psi_bwd = jnp.roll(psi, 1, axis=mu)
        U_bwd = jnp.roll(core.mH(U[mu]), 1, axis=mu)
        U_bwd = dirac._flip(U_bwd, mu, 0) if (apbc and mu == core.t_dir) else U_bwd
        out = out + jnp.matmul(op, jnp.matmul(psi_bwd, jnp.swapaxes(U_bwd, -1, -2)))
    return out


def _H2_even(U, psi_e, even_idx, shape):
    """(H² ψ) restringido a sitios pares, con ψ_e compacto (V/2,4,3).

    H conecta par<->impar, luego H² = H·H manda par->par. Se hace scatter a la
    retícula completa, se aplica H dos veces, y se recoge el sector par.
    """
    V = int(np.prod(shape))
    full = jnp.zeros((V, 4, 3), dtype=psi_e.dtype).at[even_idx].set(psi_e)
    full = full.reshape(*shape, 4, 3)
    h = hop(U, full)
    h2 = hop(U, h)
    return h2.reshape(V, 4, 3)[even_idx]


def dhat(U, psi_e, even_idx, shape, m0):
    """D̂ ψ_e = (4+m0)ψ_e - (0.25/(4+m0))·(H² ψ)_e  (complemento de Schur)."""
    return (4.0 + m0) * psi_e - (0.25 / (4.0 + m0)) * _H2_even(U, psi_e, even_idx, shape)


def dhat_dagger(U, psi_e, even_idx, shape, m0):
    """D̂† ψ_e = (4+m0)ψ_e - (0.25/(4+m0))·γ5 (H² (γ5 ψ))_e.

    Usa la γ5-hermiticidad de D̂: D̂† = γ5 D̂ γ5, con (H²)† = γ5 H² γ5.
    """
    c = 0.25 / (4.0 + m0)
    h2 = _H2_even(U, dirac.apply_gamma5(psi_e), even_idx, shape)
    return (4.0 + m0) * psi_e - c * dirac.apply_gamma5(h2)


def dhat_dagger_dhat(U, psi_e, even_idx, shape, m0):
    """D̂†D̂ ψ_e  (hermítico definido positivo, el operador del CG par-impar)."""
    return dhat_dagger(U, dhat(U, psi_e, even_idx, shape, m0), even_idx, shape, m0)


# ---------------------------------------------------------------------------
# Versión COMPACTA (sin scatter/gather): hop directo sobre representación par/impar.
# ---------------------------------------------------------------------------

def build_geometry(shape):
    """Precomputa los mapas de vecinos pares<->impares para el hop compacto.

    Devuelve un dict con (todos jnp.asarray, concretos → válidos en jit):
      even_idx, odd_idx            : (V/2,) índices planos
      fwd_even[mu], bwd_even[mu]   : (V/2,) índice compacto PAR del vecino ±μ de
                                     cada sitio IMPAR
      fwd_odd[mu], bwd_odd[mu]     : (V/2,) índice compacto IMPAR del vecino ±μ de
                                     cada sitio PAR
    """
    _validate_even_shape(shape)
    Nx, Ny, Nz, Nt = shape
    V = Nx * Ny * Nz * Nt
    x = np.arange(Nx)
    y = np.arange(Ny)
    z = np.arange(Nz)
    t = np.arange(Nt)
    X, Y, Z, T = np.meshgrid(x, y, z, t, indexing="ij")
    parity = (X + Y + Z + T) % 2
    flat = np.arange(V).reshape(Nx, Ny, Nz, Nt)
    even_idx = flat[parity == 0].ravel()
    odd_idx = flat[parity == 1].ravel()
    even_inv = np.zeros(V, dtype=np.int32)
    even_inv[even_idx] = np.arange(V // 2)
    odd_inv = np.zeros(V, dtype=np.int32)
    odd_inv[odd_idx] = np.arange(V // 2)

    def neigh_flat(mu, sign):
        coords = [X, Y, Z, T]
        c = (coords[mu] + sign) % shape[mu]
        parts = [X, Y, Z, T]
        parts[mu] = c
        # índice plano en orden C (mismo convenio que flat=np.arange(V).reshape)
        return flat[parts[0], parts[1], parts[2], parts[3]].ravel()

    fwd_even = [even_inv[neigh_flat(mu, +1)[odd_idx]] for mu in range(4)]
    bwd_even = [even_inv[neigh_flat(mu, -1)[odd_idx]] for mu in range(4)]
    fwd_odd = [odd_inv[neigh_flat(mu, +1)[even_idx]] for mu in range(4)]
    bwd_odd = [odd_inv[neigh_flat(mu, -1)[even_idx]] for mu in range(4)]

    return {
        "even_idx": jnp.asarray(even_idx),
        "odd_idx": jnp.asarray(odd_idx),
        "fwd_even": [jnp.asarray(x) for x in fwd_even],
        "bwd_even": [jnp.asarray(x) for x in bwd_even],
        "fwd_odd": [jnp.asarray(x) for x in fwd_odd],
        "bwd_odd": [jnp.asarray(x) for x in bwd_odd],
    }


def gather_gauge(U, geo, shape, apbc=True):
    """Gauge en pares/impares, con el APBC horneado en U_t en el slice t=Nt-1.

    Horneando el −1 en U_t una sola vez, los hops compactos pueden ser PERIÓDICOS
    (idéntico al truco de fermion_force_analytic). Devuelve (U_even, U_odd):
    listas de 4 arrays (V/2, 3, 3) cada una.
    """
    U_eff = list(U)
    if apbc:
        U_eff[core.t_dir] = U_eff[core.t_dir].at[..., -1, :, :].multiply(-1.0)
    V = int(np.prod(shape))
    even_idx, odd_idx = geo["even_idx"], geo["odd_idx"]
    U_even = [u.reshape(V, 3, 3)[even_idx] for u in U_eff]
    U_odd = [u.reshape(V, 3, 3)[odd_idx] for u in U_eff]
    return U_even, U_odd


def hop_oe(U_odd, U_even, psi_e, geo):
    """H ψ en IMPARES desde ψ_e en pares (periódico; APBC ya horneado)."""
    out = jnp.zeros_like(psi_e)
    for mu in range(core.Nd):
        om = (core.EYE4 - core.GAMMAS[mu]).astype(psi_e.dtype)
        op = (core.EYE4 + core.GAMMAS[mu]).astype(psi_e.dtype)
        psi_fwd = psi_e[geo["fwd_even"][mu]]
        psi_bwd = psi_e[geo["bwd_even"][mu]]
        out = out + jnp.matmul(om, jnp.matmul(psi_fwd, jnp.swapaxes(U_odd[mu], -1, -2)))
        Ub = U_even[mu][geo["bwd_even"][mu]]
        out = out + jnp.matmul(op, jnp.matmul(psi_bwd, jnp.swapaxes(core.mH(Ub), -1, -2)))
    return out


def hop_eo(U_even, U_odd, psi_o, geo):
    """H ψ en PARES desde ψ_o en impares (periódico; APBC ya horneado)."""
    out = jnp.zeros_like(psi_o)
    for mu in range(core.Nd):
        om = (core.EYE4 - core.GAMMAS[mu]).astype(psi_o.dtype)
        op = (core.EYE4 + core.GAMMAS[mu]).astype(psi_o.dtype)
        psi_fwd = psi_o[geo["fwd_odd"][mu]]
        psi_bwd = psi_o[geo["bwd_odd"][mu]]
        out = out + jnp.matmul(om, jnp.matmul(psi_fwd, jnp.swapaxes(U_even[mu], -1, -2)))
        Ub = U_odd[mu][geo["bwd_odd"][mu]]
        out = out + jnp.matmul(op, jnp.matmul(psi_bwd, jnp.swapaxes(core.mH(Ub), -1, -2)))
    return out


def dhat_compact(U_odd, U_even, psi_e, geo, m0):
    """D̂ ψ_e = (4+m0)ψ_e − (0.25/(4+m0)) H²ψ_e, con hops compactos."""
    fact = jnp.asarray(4.0 + m0, dtype=psi_e.dtype)
    c = jnp.asarray(0.25 / (4.0 + m0), dtype=psi_e.dtype)
    h = hop_oe(U_odd, U_even, psi_e, geo)
    h2 = hop_eo(U_even, U_odd, h, geo)
    return fact * psi_e - c * h2


def dhat_dagger_compact(U_odd, U_even, psi_e, geo, m0):
    """D̂† ψ_e = (4+m0)ψ_e − (0.25/(4+m0)) γ5 H² γ5 ψ_e (hops compactos)."""
    fact = jnp.asarray(4.0 + m0, dtype=psi_e.dtype)
    c = jnp.asarray(0.25 / (4.0 + m0), dtype=psi_e.dtype)
    g5x = dirac.apply_gamma5(psi_e)
    h = hop_oe(U_odd, U_even, g5x, geo)
    h2 = hop_eo(U_even, U_odd, h, geo)
    return fact * psi_e - c * dirac.apply_gamma5(h2)


def dhat_dagger_dhat_compact(U_odd, U_even, psi_e, geo, m0):
    """D̂†D̂ ψ_e con hops compactos (el operador del CG par-impar rápido)."""
    return dhat_dagger_compact(U_odd, U_even, dhat_compact(U_odd, U_even, psi_e, geo, m0),
                               geo, m0)
