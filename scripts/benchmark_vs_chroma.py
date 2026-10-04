"""Benchmark justo JAX vs Chroma: trayectoria HMC 2-sabores Wilson 4³×8.

Misma física que Chroma bench2flav: beta=6.0, m0=-0.5, tau=1.0 (10 pasos),
CG tol 1e-12, APBC temporal. Mide tiempo/trayectoria e iteraciones CG.

Control de hilos para comparar justo:
  - Chroma np=1  = 1 proceso, 1 hilo  → JAX con 1 hilo (XLA_FLAGS intra_op=1)
  - Chroma np=2  = 2 procesos        → JAX con 2 hilos (default)
"""
import os
import sys
import time
import numpy as np
import jax
import jax.numpy as jnp

from tlattice_jax import core, evenodd, hmc, hmc_eo

jax.config.update("jax_enable_x64", True)

BETA, M0, SHAPE = 6.0, -0.5, (4, 4, 4, 8)
N_STEPS, EPS = 10, 0.1  # tau = 1.0


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
    nthreads = int(os.environ.get("INTRA_OP", "0"))
    print(f"XLA intra_op_parallelism_threads = {nthreads if nthreads else 'default (auto)'}")
    print(f"os.cpu_count() = {os.cpu_count()}")

    key = jax.random.PRNGKey(42)
    U = [random_su3(SHAPE, k) for k in jax.random.split(key, 4)]
    geo = evenodd.build_geometry(SHAPE)

    step_unp = jax.jit(lambda U, k: hmc.hmc_step(U, BETA, M0, N_STEPS, EPS, k))
    step_eo = jax.jit(lambda U, k: hmc_eo.hmc_step_eo(U, BETA, M0, N_STEPS, EPS, k, geo, SHAPE))

    # warmup (compila)
    _ = step_unp(U, jax.random.PRNGKey(0))
    _ = step_eo(U, jax.random.PRNGKey(0))

    # iteraciones CG (una solve representativa)
    key, sk = jax.random.split(key)
    phi, _ = hmc.sample_phi(sk, U, M0)
    _, it_unp, _ = hmc.solve_chi(U, phi, M0)
    phi_e, _ = evenodd.split_eo(phi, *evenodd.checkerboard(SHAPE))
    _, it_eo, _ = hmc_eo.solve_chi_eo(U, phi_e, geo, SHAPE, M0)
    print(f"CG iteraciones:  sin precond {int(it_unp)}   par-impar {int(it_eo)}")

    t_unp = timeit(step_unp, U, jax.random.PRNGKey(1))
    t_eo = timeit(step_eo, U, jax.random.PRNGKey(1))
    print(f"\nTrayectoria HMC (4³×8, tau=1.0, 10 pasos):")
    print(f"  JAX sin precond:  {t_unp*1e3:.0f} ms/traj")
    print(f"  JAX par-impar:    {t_eo*1e3:.0f} ms/traj")


if __name__ == "__main__":
    main()
