"""Benchmark par-impar vs sin precondicionar: solve y trayectoria completa."""
import time
import numpy as np
import jax
import jax.numpy as jnp

from tlattice_jax import core, dirac, evenodd, hmc, hmc_eo, gauge

jax.config.update("jax_enable_x64", True)

BETA, M0, SHAPE = 6.0, -0.5, (4, 4, 4, 8)


def random_su3(shape, key):
    k1, k2 = jax.random.split(key)
    A = jax.random.normal(k1, (*shape, 3, 3)) + 1j * jax.random.normal(k2, (*shape, 3, 3))
    return core.expm_batched(core.project_tah(A))


def timeit(fn, *args, n=5):
    out = fn(*args)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), out)  # warmup
    t0 = time.perf_counter()
    for _ in range(n):
        out = fn(*args)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), out)
    return (time.perf_counter() - t0) / n


def main():
    key = jax.random.PRNGKey(42)
    U = [random_su3(SHAPE, k) for k in jax.random.split(key, 4)]
    even_idx, odd_idx = evenodd.checkerboard(SHAPE)
    geo = evenodd.build_geometry(SHAPE)

    # phi (sin precond) y phi_e (par-impar)
    key, sk = jax.random.split(key)
    eta = jax.random.normal(sk, (*SHAPE, 4, 3)) + 1j * jax.random.normal(jax.random.split(sk)[1], (*SHAPE, 4, 3))
    eta = eta / jnp.sqrt(2.0)
    phi = dirac.d_dagger(U, eta, M0)
    phi_e, _ = evenodd.split_eo(phi, even_idx, odd_idx)

    solve_unp = jax.jit(lambda U: hmc.solve_chi(U, phi, M0))
    solve_eo = jax.jit(lambda U: hmc_eo.solve_chi_eo(U, phi_e, geo, SHAPE, M0))

    it_unp = int(solve_unp(U)[1])
    it_eo = int(solve_eo(U)[1])
    t_unp = timeit(solve_unp, U)
    t_eo = timeit(solve_eo, U)
    print(f"SOLVE:  sin precond {it_unp} iter, {t_unp*1e3:.1f} ms")
    print(f"        par-impar  {it_eo} iter, {t_eo*1e3:.1f} ms")
    print(f"        → aceleración en el solve: {t_unp/t_eo:.2f}×")

    step_unp = jax.jit(lambda U, k: hmc.hmc_step(U, BETA, M0, 10, 0.1, k))
    step_eo = jax.jit(lambda U, k: hmc_eo.hmc_step_eo(U, BETA, M0, 10, 0.1, k, geo, SHAPE))
    t_su = timeit(step_unp, U, jax.random.PRNGKey(0), n=3)
    t_se = timeit(step_eo, U, jax.random.PRNGKey(0), n=3)
    print(f"TRAJEC.: sin precond {t_su*1e3:.0f} ms/traj | par-impar {t_se*1e3:.0f} ms/traj")
    print(f"        → aceleración por trayectoria: {t_su/t_se:.2f}×")


if __name__ == "__main__":
    main()
