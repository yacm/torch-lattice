"""Perfil del CG + estructura de gamma5 + variante v4 (matmul color + gamma esparsa)."""
import time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from tlattice_jax import core, dirac, hmc, solve
from tlattice_jax.dirac import dslash as dslash_v1

SHAPE = (4, 4, 4, 8)
M0 = -0.5


def gamma_apply(tmp, mu):
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


def dslash_v4(U, psi, m0, apbc=True):
    """matmul para color + gamma esparsa para espín."""
    out = (4.0 + m0) * psi
    for mu in range(core.Nd):
        psi_fwd = jnp.roll(psi, -1, axis=mu)
        U_fwd = _flip(U[mu], mu, -1) if (apbc and mu == core.t_dir) else U[mu]
        tmp = jnp.matmul(psi_fwd, jnp.swapaxes(U_fwd, -1, -2))
        out = out - 0.5 * (tmp - gamma_apply(tmp, mu))
        psi_bwd = jnp.roll(psi, 1, axis=mu)
        U_bwd = jnp.roll(core.mH(U[mu]), 1, axis=mu)
        U_bwd = _flip(U_bwd, mu, 0) if (apbc and mu == core.t_dir) else U_bwd
        tmp = jnp.matmul(psi_bwd, jnp.swapaxes(U_bwd, -1, -2))
        out = out - 0.5 * (tmp + gamma_apply(tmp, mu))
    return out


def main():
    print("=== gamma5 (estructura) ===")
    print(np.asarray(core.GAMMA5))

    key = jax.random.PRNGKey(0)
    U = []
    for _ in range(4):
        key, sk = jax.random.split(key)
        U.append(jax.random.normal(sk, (*SHAPE, 3, 3), dtype=jnp.complex128))
    key, sk = jax.random.split(key)
    psi = jax.random.normal(sk, (*SHAPE, 4, 3), dtype=jnp.complex128)
    key, sk = jax.random.split(key)
    phi = jax.random.normal(sk, (*SHAPE, 4, 3), dtype=jnp.complex128)

    # correctitud v4
    ref = np.asarray(dslash_v1(U, psi, M0))
    got = np.asarray(dslash_v4(U, psi, M0))
    print(f"\nv4 max|Δ| vs v1 = {np.max(np.abs(got - ref)):.2e}")

    # benchmark dslash v1 vs v4
    for name, fn in [("v1 dslash", dslash_v1), ("v4 dslash (matmul+esparsa)", dslash_v4)]:
        jf = jax.jit(fn)
        jf(U, psi, M0).block_until_ready()
        for _ in range(3):
            jf(U, psi, M0).block_until_ready()
        ts = []
        for _ in range(20):
            t0 = time.perf_counter()
            jf(U, psi, M0).block_until_ready()
            ts.append(time.perf_counter() - t0)
        print(f"{name:28s}  {np.median(ts)*1e3:7.3f} ms/Dslash")

    # benchmark CG completo (solve_chi)
    jc = jax.jit(lambda u, p: hmc.solve_chi(u, p, M0))
    chi, iters, resid = jc(U, phi)
    iters = int(iters)
    jc(U, phi)[0].block_until_ready()
    for _ in range(3):
        jc(U, phi)[0].block_until_ready()
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        jc(U, phi)[0].block_until_ready()
        ts.append(time.perf_counter() - t0)
    ms = np.median(ts) * 1e3
    print(f"\nCG solve_chi: {ms:.1f} ms, {iters} iter → {ms/iters:.3f} ms/iter")
    print(f"  (2 dslash = {2*0.860:.3f} ms ⇒ overhead/iter ~ {ms/iters - 2*0.860:.3f} ms)")


if __name__ == "__main__":
    main()
