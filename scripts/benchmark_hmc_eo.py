"""Benchmark trayectoria completa: par-impar compacto vs sin precondicionar."""
import time
import numpy as np
import jax
import jax.numpy as jnp

from tlattice_jax import core, evenodd, hmc as hmc_unprec, hmc_eo

jax.config.update("jax_enable_x64", True)

BETA, M0, SHAPE = 6.0, -0.5, (4, 4, 4, 8)


def random_su3(shape, key):
    k1, k2 = jax.random.split(key)
    A = jax.random.normal(k1, (*shape, 3, 3)) + 1j * jax.random.normal(k2, (*shape, 3, 3))
    return core.expm_batched(core.project_tah(A))


def timeit(fn, *args, n=3):
    out = fn(*args)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), out)
    t0 = time.perf_counter()
    for _ in range(n):
        out = fn(*args)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), out)
    return (time.perf_counter() - t0) / n


def main():
    key = jax.random.PRNGKey(42)
    U = [random_su3(SHAPE, k) for k in jax.random.split(key, 4)]
    geo = evenodd.build_geometry(SHAPE)

    step_unp = jax.jit(lambda U, k: hmc_unprec.hmc_step(U, BETA, M0, 10, 0.1, k))
    step_eo = jax.jit(lambda U, k: hmc_eo.hmc_step_eo(U, BETA, M0, 10, 0.1, k, geo, SHAPE))

    # warmup
    _ = step_unp(U, jax.random.PRNGKey(0))
    _ = step_eo(U, jax.random.PRNGKey(0))

    t_unp = timeit(step_unp, U, jax.random.PRNGKey(1))
    t_eo = timeit(step_eo, U, jax.random.PRNGKey(1))
    print(f"TRAYECTORIA (10 pasos, eps=0.1):")
    print(f"  sin precondicionar:  {t_unp*1e3:.0f} ms/traj")
    print(f"  par-impar compacto:  {t_eo*1e3:.0f} ms/traj")
    print(f"  → aceleración: {t_unp/t_eo:.2f}×")


if __name__ == "__main__":
    main()
