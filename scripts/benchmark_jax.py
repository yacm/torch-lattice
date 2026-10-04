"""Benchmark Dslash + CG: JAX (XLA JIT) vs PyTorch (eager).

Mismo campo gauge SU(3), misma precisión (complex128), misma retícula.
Compara:
  1. Corrección: JAX Dslash == PyTorch Dslash (componente a componente).
  2. Velocidad del Dslash crudo (JIT compilado vs eager).
  3. Velocidad del CG completo (D†D x = D†b, tol rel 1e-12).
"""
import time

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax

jax.config.update("jax_enable_x64", True)

import torch

from tlattice import dirac, invert, su3  # noqa: E402  (torch-lattice)

M0 = -0.5
TOL = 1e-12
MAX_ITER = 5000

# --- gammas DeGrand-Rossi (mismas que config.py) ---
_G = [
    np.array([[0, 0, 0, 1j], [0, 0, 1j, 0], [0, -1j, 0, 0], [-1j, 0, 0, 0]]),  # x
    np.array([[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]]),     # y
    np.array([[0, 0, 1j, 0], [0, 0, 0, -1j], [-1j, 0, 0, 0], [0, 1j, 0, 0]]), # z
    np.array([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]]),       # t
]
GS = [jnp.array(g, dtype=jnp.complex128) for g in _G]
EYE = jnp.eye(4, dtype=jnp.complex128)
G5 = GS[0] @ GS[1] @ GS[2] @ GS[3]
T_DIR = 3


def _flip(u, axis, tslice):
    idx = [slice(None)] * u.ndim
    idx[axis] = tslice
    return u.at[tuple(idx)].multiply(-1.0)


def dslash_jax(U, psi, m0):
    out = (4.0 + m0) * psi
    for mu in range(4):
        om = EYE - GS[mu]
        op = EYE + GS[mu]
        Umu = U[mu]
        # forward: (1-gamma_mu) U_mu(x) psi(x+mu)
        psi_fwd = jnp.roll(psi, -1, axis=mu)
        U_fwd = _flip(Umu, mu, -1) if mu == T_DIR else Umu
        tmp_fwd = jnp.matmul(psi_fwd, jnp.swapaxes(U_fwd, -1, -2))
        out = out - 0.5 * jnp.matmul(om, tmp_fwd)
        # backward: (1+gamma_mu) U_mu^dag(x-mu) psi(x-mu)
        psi_bwd = jnp.roll(psi, 1, axis=mu)
        U_bwd = jnp.roll(jnp.conj(Umu).swapaxes(-1, -2), 1, axis=mu)
        U_bwd = _flip(U_bwd, mu, 0) if mu == T_DIR else U_bwd
        tmp_bwd = jnp.matmul(psi_bwd, jnp.swapaxes(U_bwd, -1, -2))
        out = out - 0.5 * jnp.matmul(op, tmp_bwd)
    return out


dslash_jit = jax.jit(dslash_jax)


def apply_g5(x):
    return jnp.matmul(G5, x)


def d_dagger(U, x, m0):
    return apply_g5(dslash_jax(U, apply_g5(x), m0))


def solve_jax(U, b, m0):
    """D†D x = D†b vía CG con lax.while_loop (compilado)."""
    rhs = d_dagger(U, b, m0)

    def normal(x):
        return d_dagger(U, dslash_jax(U, x, m0), m0)

    x = jnp.zeros_like(b)
    r = rhs - normal(x)
    p = r
    rsold = jnp.real(jnp.sum(r.conj() * r))
    rhs_norm = jnp.sqrt(jnp.real(jnp.sum(rhs.conj() * rhs)))

    def cond(s):
        k, _, r, _, rs = s
        return (k < MAX_ITER) & (jnp.sqrt(rs) > TOL * rhs_norm)

    def body(s):
        k, x, r, p, rs = s
        Ap = normal(p)
        denom = jnp.real(jnp.sum(p.conj() * Ap))
        alpha = rs / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = jnp.real(jnp.sum(r.conj() * r))
        beta = rsnew / rs
        p = r + beta * p
        return (k + 1, x, r, p, rsnew)

    k, x, r, p, rs = lax.while_loop(cond, body, (0, x, r, p, rsold))
    resid = jnp.sqrt(rs) / rhs_norm
    return x, k, resid


solve_jit = jax.jit(solve_jax)


def bench(fn, n=30, warmup=5):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2] * 1000


def main():
    for SHAPE in [(4, 4, 4, 8), (8, 8, 8, 8)]:
        V = 1
        for s in SHAPE:
            V *= s
        print(f"\n=== retícula {SHAPE}  (V={V})  complex128 ===\n")

        # campo SU(3) (heat-kernel), el mismo para torch y jax
        U_t = [su3.random_su3(*SHAPE, dtype=torch.complex128) for _ in range(4)]
        U_np = [u.numpy() for u in U_t]
        U_j = [jnp.array(u) for u in U_np]
        psi_t = torch.randn(*SHAPE, 4, 3, dtype=torch.complex128)
        psi_j = jnp.array(psi_t.numpy())

        # 1. corrección
        out_t = dirac.wilson_dslash(U_t, psi_t, M0).numpy()
        out_j = np.asarray(dslash_jax(U_j, psi_j, M0))
        md = np.max(np.abs(out_t - out_j))
        print(f"  corrección JAX vs torch Dslash: max|Δ| = {md:.2e}")

        # 2. Dslash crudo (torch eager vs jax JIT)
        t_t = bench(lambda: dirac.wilson_dslash(U_t, psi_t, M0))
        t_j = bench(lambda: dslash_jit(U_j, psi_j, M0).block_until_ready())
        print(f"  Dslash  torch {t_t:7.2f} ms   JAX {t_j:7.2f} ms   "
              f"(JAX {t_t/t_j:5.2f}× más rápido)")

        # 3. CG completo (D†D x = D†b)
        x_t, it_t, r_t = invert.solve_dirac(U_t, psi_t, M0, max_iter=MAX_ITER, tol=TOL)
        x_j, it_j, r_j = solve_jit(U_j, psi_j, M0)
        x_j.block_until_ready()
        print(f"  CG: torch {it_t} iter (resid {r_t:.1e}) | JAX {int(it_j)} iter "
              f"(resid {float(r_j):.1e})")

        t_t_cg = bench(lambda: invert.solve_dirac(U_t, psi_t, M0, max_iter=MAX_ITER, tol=TOL),
                       n=5, warmup=2)
        t_j_cg = bench(lambda: solve_jit(U_j, psi_j, M0)[0].block_until_ready(),
                       n=5, warmup=2)
        print(f"  CG     torch {t_t_cg:7.2f} ms   JAX {t_j_cg:7.2f} ms   "
              f"(JAX {t_t_cg/t_j_cg:5.2f}× más rápido)")


if __name__ == "__main__":
    main()
