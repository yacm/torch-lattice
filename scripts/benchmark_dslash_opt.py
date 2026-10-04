"""Micro-benchmark del Dslash: compara variantes de implementación JAX.

Variantes:
  v1 = actual (matmul 4x4 + swapaxes)
  v2 = einsum para la contracción de color + proyección de espín ESPARSA (gamma como permutación)
  v3 = einsum, pero proyección de espín densa (matmul 4x4)
Mide ms/Dslash y verifica compatibilidad bit-a-bit vs v1.
"""
import time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from tlattice_jax import core
from tlattice_jax.dirac import dslash as dslash_v1

SHAPE = (4, 4, 4, 8)
M0 = -0.5


def gamma_apply(tmp, mu):
    """gamma_mu @ tmp, explotando que gamma_mu es dispersa (permutación + signo)."""
    if mu == 0:
        return 1j * jnp.stack([tmp[..., 3, :], tmp[..., 2, :], -tmp[..., 1, :], -tmp[..., 0, :]], axis=-2)
    if mu == 1:
        return jnp.stack([-tmp[..., 3, :], tmp[..., 2, :], tmp[..., 1, :], -tmp[..., 0, :]], axis=-2)
    if mu == 2:
        return 1j * jnp.stack([tmp[..., 2, :], -tmp[..., 3, :], -tmp[..., 0, :], tmp[..., 1, :]], axis=-2)
    return jnp.stack([tmp[..., 2, :], tmp[..., 3, :], tmp[..., 0, :], tmp[..., 1, :]], axis=-2)


def _flip(u, axis, tslice):
    idx = [slice(None)] * u.ndim
    idx[axis] = tslice
    return u.at[tuple(idx)].multiply(-1.0)


def dslash_v2(U, psi, m0, apbc=True):
    """einsum + proyección de espín dispersa."""
    out = (4.0 + m0) * psi
    for mu in range(core.Nd):
        psi_fwd = jnp.roll(psi, -1, axis=mu)
        U_fwd = _flip(U[mu], mu, -1) if (apbc and mu == core.t_dir) else U[mu]
        tmp = jnp.einsum("...sa,...ca->...sc", psi_fwd, U_fwd)          # color
        out = out - 0.5 * (tmp - gamma_apply(tmp, mu))                  # (1-gamma)
        psi_bwd = jnp.roll(psi, 1, axis=mu)
        U_bwd = jnp.roll(core.mH(U[mu]), 1, axis=mu)
        U_bwd = _flip(U_bwd, mu, 0) if (apbc and mu == core.t_dir) else U_bwd
        tmp = jnp.einsum("...sa,...ca->...sc", psi_bwd, U_bwd)          # color
        out = out - 0.5 * (tmp + gamma_apply(tmp, mu))                  # (1+gamma)
    return out


def dslash_v3(U, psi, m0, apbc=True):
    """einsum para color, pero proyección de espín densa (matmul 4x4)."""
    out = (4.0 + m0) * psi
    for mu in range(core.Nd):
        om = core.EYE4 - core.GAMMAS[mu]
        op = core.EYE4 + core.GAMMAS[mu]
        psi_fwd = jnp.roll(psi, -1, axis=mu)
        U_fwd = _flip(U[mu], mu, -1) if (apbc and mu == core.t_dir) else U[mu]
        tmp = jnp.einsum("...sa,...ca->...sc", psi_fwd, U_fwd)
        out = out - 0.5 * jnp.matmul(om, tmp)
        psi_bwd = jnp.roll(psi, 1, axis=mu)
        U_bwd = jnp.roll(core.mH(U[mu]), 1, axis=mu)
        U_bwd = _flip(U_bwd, mu, 0) if (apbc and mu == core.t_dir) else U_bwd
        tmp = jnp.einsum("...sa,...ca->...sc", psi_bwd, U_bwd)
        out = out - 0.5 * jnp.matmul(op, tmp)
    return out


def main():
    key = jax.random.PRNGKey(0)
    U = []
    for _ in range(4):
        key, sk = jax.random.split(key)
        U.append(jax.random.normal(sk, (*SHAPE, 3, 3), dtype=jnp.complex128))
    key, sk = jax.random.split(key)
    psi = jax.random.normal(sk, (*SHAPE, 4, 3), dtype=jnp.complex128)

    variants = {"v1 (actual)": dslash_v1, "v2 (einsum+gamma dispersa)": dslash_v2,
                "v3 (einsum+gamma densa)": dslash_v3}

    # correctitud vs v1
    ref = np.asarray(dslash_v1(U, psi, M0))
    for name, fn in variants.items():
        got = np.asarray(fn(U, psi, M0))
        print(f"{name:28s}  max|Δ| vs v1 = {np.max(np.abs(got - ref)):.2e}")

    print()
    for name, fn in variants.items():
        jf = jax.jit(fn)
        jf(U, psi, M0).block_until_ready()          # compilar
        for _ in range(3):
            jf(U, psi, M0).block_until_ready()      # warmup
        ts = []
        for _ in range(20):
            t0 = time.perf_counter()
            jf(U, psi, M0).block_until_ready()
            ts.append(time.perf_counter() - t0)
        ms = np.median(ts) * 1e3
        print(f"{name:28s}  {ms:8.3f} ms/Dslash   ({1e3/ms:6.1f} Dslash/s)")


if __name__ == "__main__":
    main()
