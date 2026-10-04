"""Dslash de Wilson como kernel Pallas (Triton) para GPU: un hilo por sitio.

Misma estructura que los kernels de QUDA: tablas de vecinos precomputadas,
parte real e imaginaria en planos separados (Triton no tiene complejos), proyección
a half-spinor (solo 2 de los 4 vectores de color se multiplican por SU(3)) y todo
el Dslash fusionado en un único kernel. En una RTX 5090, 24³×48: ~515 GFLOP/s en
fp64 y ~3800 en fp32, frente a ~26 del Dslash de dirac.py (XLA) en fp64.

Datos:  x (2, 12, V) real = (re/im, spin*3+color, sitio);  sitio = fila-mayor (x,y,z,t).
Requiere GPU con Triton; V debe ser múltiplo de `block` (por defecto 128).
Valida contra tlattice_jax.dirac_soa / dirac (tests/test_dirac_pallas.py).
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

from . import core
from .dirac_soa import _PP


def _tables(shape):
    idx = np.arange(int(np.prod(shape))).reshape(shape)
    nf = np.stack([np.roll(idx, -1, axis=m).reshape(-1) for m in range(core.Nd)])
    nb = np.stack([np.roll(idx, +1, axis=m).reshape(-1) for m in range(core.Nd)])
    t = np.broadcast_to(np.arange(shape[core.t_dir]), shape)
    last = shape[core.t_dir] - 1
    sf = np.where(t == last, -1.0, 1.0).reshape(-1)  # hop hacia delante cruza la frontera
    sb = np.where(t == 0, -1.0, 1.0).reshape(-1)     # hop hacia atrás cruza la frontera
    return nf.astype(np.int32), nb.astype(np.int32), sf, sb


def _make_kernel(m0, dagger, apbc, block):
    diag = 4.0 + m0

    def kern(pr, pi, ur, ui, nf_r, nb_r, sf_r, sb_r, o_r, o_i):
        own = pl.program_id(0) * block + jnp.arange(block)
        accr = [[diag * pr[s * 3 + k, own] for k in range(3)] for s in range(4)]
        acci = [[diag * pi[s * 3 + k, own] for k in range(3)] for s in range(4)]
        for mu in range(core.Nd):
            p, c = _PP[mu]
            for fwd in (True, False):
                # D: forward (1-γ), backward (1+γ). D†: proyectores intercambiados.
                sign = (+1 if dagger else -1) if fwd else (-1 if dagger else +1)
                nbr = (nf_r if fwd else nb_r)[mu, own]
                tsgn = None
                if apbc and mu == core.t_dir:
                    tsgn = (sf_r if fwd else sb_r)[own]
                fr = [[pr[s * 3 + k, nbr] for k in range(3)] for s in range(4)]
                fi = [[pi[s * 3 + k, nbr] for k in range(3)] for s in range(4)]
                hr = [[None] * 3 for _ in range(2)]
                hi = [[None] * 3 for _ in range(2)]
                for a in (0, 1):
                    w = sign * c[a]
                    for k in range(3):
                        zr, zi = fr[p[a]][k], fi[p[a]][k]
                        hr[a][k] = fr[a][k] + (w.real * zr - w.imag * zi)
                        hi[a][k] = fi[a][k] + (w.real * zi + w.imag * zr)
                if fwd:
                    ldr = lambda i, j: ur[mu, i * 3 + j, own]
                    ldi = lambda i, j: ui[mu, i * 3 + j, own]
                else:  # U†(x-μ)
                    ldr = lambda i, j: ur[mu, j * 3 + i, nbr]
                    ldi = lambda i, j: -ui[mu, j * 3 + i, nbr]
                Ur = [[ldr(i, j) for j in range(3)] for i in range(3)]
                Ui = [[ldi(i, j) for j in range(3)] for i in range(3)]
                for a in (0, 1):
                    ap = p[a]
                    r = sign * c[ap]
                    for i in range(3):
                        sr = sum(Ur[i][j] * hr[a][j] - Ui[i][j] * hi[a][j] for j in range(3))
                        si = sum(Ur[i][j] * hi[a][j] + Ui[i][j] * hr[a][j] for j in range(3))
                        if tsgn is not None:
                            sr = sr * tsgn
                            si = si * tsgn
                        accr[a][i] = accr[a][i] - 0.5 * sr
                        acci[a][i] = acci[a][i] - 0.5 * si
                        accr[ap][i] = accr[ap][i] - 0.5 * (r.real * sr - r.imag * si)
                        acci[ap][i] = acci[ap][i] - 0.5 * (r.real * si + r.imag * sr)
        for s in range(4):
            for k in range(3):
                o_r[s * 3 + k, own] = accr[s][k]
                o_i[s * 3 + k, own] = acci[s][k]

    return kern


class WilsonPallas:
    """Operador de Wilson (sin precondicionar) sobre arrays reales x (2,12,V)."""

    def __init__(self, shape, m0, dtype=jnp.float64, apbc=True, block=128, num_warps=4):
        self.shape, self.m0, self.dtype = tuple(shape), m0, dtype
        self.V = int(np.prod(shape))
        if self.V % block:
            raise ValueError(f"V={self.V} debe ser múltiplo de block={block}")
        nf, nb, sf, sb = _tables(self.shape)
        self._tab = (jnp.asarray(nf), jnp.asarray(nb),
                     jnp.asarray(sf, dtype), jnp.asarray(sb, dtype))
        full = lambda shp: pl.BlockSpec(shp, lambda i: (0,) * len(shp))
        V = self.V
        self._calls = {}
        for dagger in (False, True):
            self._calls[dagger] = pl.pallas_call(
                _make_kernel(m0, dagger, apbc, block),
                out_shape=(jax.ShapeDtypeStruct((12, V), dtype),
                           jax.ShapeDtypeStruct((12, V), dtype)),
                grid=(V // block,),
                in_specs=[full((12, V)), full((12, V)), full((4, 9, V)), full((4, 9, V)),
                          full((4, V)), full((4, V)), full((V,)), full((V,))],
                out_specs=(full((12, V)), full((12, V))),
                compiler_params=plt.CompilerParams(num_warps=num_warps),
            )

    # --- conversión de layouts -------------------------------------------
    def pack_gauge(self, U):
        """Lista de 4 (X,Y,Z,T,3,3) complejos -> (2,4,9,V) reales."""
        u = jnp.stack([jnp.moveaxis(x, (-2, -1), (0, 1)) for x in U]).reshape(4, 9, self.V)
        return jnp.stack([jnp.real(u), jnp.imag(u)]).astype(self.dtype)

    def pack(self, psi):
        """(X,Y,Z,T,4,3) complejo -> (2,12,V) real."""
        p = jnp.moveaxis(psi, (-2, -1), (0, 1)).reshape(12, self.V)
        return jnp.stack([jnp.real(p), jnp.imag(p)]).astype(self.dtype)

    def unpack(self, x):
        """(2,12,V) real -> (X,Y,Z,T,4,3) complejo."""
        p = (x[0] + 1j * x[1]).reshape(4, 3, *self.shape)
        return jnp.moveaxis(p, (0, 1), (-2, -1))

    # --- operadores (U = pack_gauge(...)) ----------------------------------
    def _apply(self, U, x, dagger):
        o_r, o_i = self._calls[dagger](x[0], x[1], U[0], U[1], *self._tab)
        return jnp.stack([o_r, o_i])

    def dslash(self, U, x):
        return self._apply(U, x, False)

    def d_dagger(self, U, x):
        return self._apply(U, x, True)

    def normal(self, U, x):
        """D†D x (hermítico definido positivo): matvec del CG."""
        return self._apply(U, self._apply(U, x, False), True)
