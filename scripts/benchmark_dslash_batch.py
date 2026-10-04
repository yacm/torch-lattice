"""Dslash: compara la versión actual (loop por dirección) vs la batcheada (un solo matmul)."""
import time
import numpy as np
import torch
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from tlattice import su3 as tsu3
from tlattice_jax import core
from tlattice_jax.dirac import dslash as dslash_v1

SHAPE = (4, 4, 4, 8)
M0 = -0.5


def _flip(u, axis, tslice):
    idx = [slice(None)] * u.ndim
    idx[axis] = tslice
    return u.at[tuple(idx)].multiply(-1.0)


def dslash_batched(U, psi, m0, apbc=True):
    """Los 8 hops (4 dirs × fwd/bwd) en UN solo matmul de color + UN matmul de espín."""
    psi_all, U_all = [], []
    for mu in range(core.Nd):
        # forward
        psi_all.append(jnp.roll(psi, -1, axis=mu))
        Uf = _flip(U[mu], mu, -1) if (apbc and mu == core.t_dir) else U[mu]
        U_all.append(Uf)
        # backward
        psi_all.append(jnp.roll(psi, 1, axis=mu))
        Ub = jnp.roll(core.mH(U[mu]), 1, axis=mu)
        Ub = _flip(Ub, mu, 0) if (apbc and mu == core.t_dir) else Ub
        U_all.append(Ub)

    psi_all = jnp.stack(psi_all)          # (8, Nx,Ny,Nz,Nt, 4,3)
    U_all = jnp.stack(U_all)              # (8, Nx,Ny,Nz,Nt, 3,3)
    tmp = jnp.matmul(psi_all, jnp.swapaxes(U_all, -1, -2))   # (8,...,4,3)

    # proyectores: par = forward (1-γ), impar = backward (1+γ)
    proj = []
    for mu in range(core.Nd):
        proj.append(core.EYE4 - core.GAMMAS[mu])   # forward
        proj.append(core.EYE4 + core.GAMMAS[mu])   # backward
    proj = jnp.stack(proj)                       # (8,4,4)
    contrib = jnp.einsum("doi,ds...ik->ds...ok", proj, tmp)   # (8,...,4,3)
    hop = jnp.sum(contrib, axis=0)               # suma los 8 hops
    return (4.0 + m0) * psi - 0.5 * hop


def main():
    torch.manual_seed(42)
    U_t = [tsu3.random_su3(*SHAPE, dtype=torch.complex128) for _ in range(4)]
    U = [jnp.asarray(u.numpy()) for u in U_t]
    key = jax.random.PRNGKey(0)
    key, sk = jax.random.split(key)
    psi = jax.random.normal(sk, (*SHAPE, 4, 3), dtype=jnp.complex128)

    ref = np.asarray(dslash_v1(U, psi, M0))
    got = np.asarray(dslash_batched(U, psi, M0))
    print(f"batched max|Δ| vs v1 = {np.max(np.abs(got - ref)):.2e}")

    for name, fn in [("v1 loop", dslash_v1), ("batched", dslash_batched)]:
        jf = jax.jit(fn)
        jf(U, psi, M0).block_until_ready()
        for _ in range(5):
            jf(U, psi, M0).block_until_ready()
        ts = []
        for _ in range(40):
            t0 = time.perf_counter()
            jf(U, psi, M0).block_until_ready()
            ts.append(time.perf_counter() - t0)
        print(f"{name:10s}  {np.median(ts)*1e3:7.3f} ms/Dslash")


if __name__ == "__main__":
    main()
