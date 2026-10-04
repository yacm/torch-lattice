"""Benchmark JUSTO: misma config termalizada de Chroma, mismo CG tol 1e-12.

Compara tiempo/trayectoria HMC (4³×8, beta=6.0, m0=-0.5, tau=1.0, 10 pasos)
sobre el MISMO campo gauge (dummy_run_cfg_7, w_plaq=0.7257), con control de hilos.
"""
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from tlattice_jax import evenodd, hmc, hmc_eo

jax.config.update("jax_enable_x64", True)

BETA, M0, SHAPE = 6.0, -0.5, (4, 4, 4, 8)
ROOT = Path(__file__).resolve().parents[1]
FIELD = Path(os.environ.get("TORCH_LATTICE_FIELD", ROOT / "benchmarks/canonical/cfg_7.bin"))


def load_chroma_cfg(path):
    raw = np.fromfile(path, dtype=np.float64)
    cv = raw[0::2] + 1j * raw[1::2]
    cv = cv.reshape(4, 4, 4, 4, 8, 3, 3)
    return [jnp.asarray(cv[mu].copy()) for mu in range(4)]


def timeit(fn, *args, n=2):
    out = fn(*args)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), out)
    t0 = time.perf_counter()
    for _ in range(n):
        out = fn(*args)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), out)
    return (time.perf_counter() - t0) / n


def main():
    U = load_chroma_cfg(FIELD)
    geo = evenodd.build_geometry(SHAPE)

    step_unp = jax.jit(lambda U, k: hmc.hmc_step(U, BETA, M0, 10, 0.1, k))
    step_eo = jax.jit(lambda U, k: hmc_eo.hmc_step_eo(U, BETA, M0, 10, 0.1, k, geo, SHAPE))

    # warmup
    _ = step_unp(U, jax.random.PRNGKey(0))
    _ = step_eo(U, jax.random.PRNGKey(0))

    t_unp = timeit(step_unp, U, jax.random.PRNGKey(1))
    t_eo = timeit(step_eo, U, jax.random.PRNGKey(1))

    nthreads = os.environ.get("INTRA_OP", "default")
    print(f"JAX intra_op threads = {nthreads}   (campo termalizado w_plaq=0.7257)")
    print(f"  sin precond:  {t_unp*1e3:.0f} ms/traj")
    print(f"  par-impar:    {t_eo*1e3:.0f} ms/traj")


if __name__ == "__main__":
    main()
