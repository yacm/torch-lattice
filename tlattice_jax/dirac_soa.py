"""Dslash de Wilson en JAX con layout "sitios al final" y proyección a half-spinor.

Layout SoA: psi (4,3,X,Y,Z,T), U (4,3,3,X,Y,Z,T) (dirección, color, color, sitios).
Los ejes pequeños (spin/color) van delante y el volumen es la dimensión contigua,
así XLA fusiona la aritmética elemento a elemento en pocos kernels, en lugar de
lanzar matmul batcheados de 3x3 / 4x4. Además (1±γ_μ) tiene rango 2: solo se
multiplican 2 vectores de color por SU(3) y el resto se reconstruye por fase.

Valida contra tlattice_jax.dirac (mismo resultado a ~1e-15 en complex128).
"""
import numpy as np
import jax.numpy as jnp

from . import core


def _perm_phase(g):
    """γ_μ = permutación con fases: γ[a, p[a]] = c[a] (una entrada por fila)."""
    g = np.asarray(g)
    p = [int(np.nonzero(g[a])[0][0]) for a in range(4)]
    return p, [complex(g[a, p[a]]) for a in range(4)]


_PP = [_perm_phase(g) for g in core.GAMMAS]
# La reconstrucción de half-spinor supone que las filas 0,1 se acoplan con 2,3.
assert all(set(p[:2]) == {2, 3} for p, _ in _PP)


def to_soa(U, psi=None):
    """U: lista de 4 (X,Y,Z,T,3,3) -> (4,3,3,X,Y,Z,T); psi (X,Y,Z,T,4,3) -> (4,3,X,Y,Z,T)."""
    Us = jnp.stack([jnp.moveaxis(u, (-2, -1), (0, 1)) for u in U])
    if psi is None:
        return Us
    return Us, jnp.moveaxis(psi, (-2, -1), (0, 1))


def psi_from_soa(ps):
    """Inverso del layout de psi: (4,3,X,Y,Z,T) -> (X,Y,Z,T,4,3)."""
    return jnp.moveaxis(ps, (0, 1), (-2, -1))


def _hop(Us, ps, m0, dagger, apbc):
    acc = [[(4.0 + m0) * ps[s, c] for c in range(3)] for s in range(4)]
    for mu in range(core.Nd):
        p, c = _PP[mu]
        # D:  forward (1-γ), backward (1+γ).  D†: proyectores intercambiados.
        for fwd in (True, False):
            sign = (+1 if dagger else -1) if fwd else (-1 if dagger else +1)
            shift = -1 if fwd else 1
            f = [[jnp.roll(ps[s, k], shift, axis=mu) for k in range(3)] for s in range(4)]
            h = [[f[a][k] + (sign * c[a]) * f[p[a]][k] for k in range(3)] for a in (0, 1)]
            if fwd:
                Um = Us[mu]
                if apbc and mu == core.t_dir:
                    Um = Um.at[..., -1].multiply(-1.0)
                mat = lambda i, j: Um[i, j]
            else:
                Um = jnp.roll(Us[mu], 1, axis=mu + 2)
                if apbc and mu == core.t_dir:
                    Um = Um.at[..., 0].multiply(-1.0)
                mat = lambda i, j: jnp.conj(Um[j, i])
            for a in (0, 1):
                uh = [sum(mat(i, j) * h[a][j] for j in range(3)) for i in range(3)]
                ap = p[a]
                r = sign * c[ap]
                for k in range(3):
                    acc[a][k] = acc[a][k] - 0.5 * uh[k]
                    acc[ap][k] = acc[ap][k] - 0.5 * r * uh[k]
    return jnp.stack([jnp.stack(row) for row in acc])


def dslash(Us, ps, m0, apbc=True):
    """D_w psi en layout SoA (equivale a dirac.dslash)."""
    return _hop(Us, ps, m0, False, apbc)


def d_dagger(Us, ps, m0, apbc=True):
    """D† psi en layout SoA (equivale a dirac.d_dagger)."""
    return _hop(Us, ps, m0, True, apbc)
