"""complex64 vs complex128 en el solve par-impar (mismo campo termalizado)."""
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from tlattice_jax import evenodd, solve

jax.config.update("jax_enable_x64", True)

M0 = -0.5
SHAPE = (4, 4, 4, 8)
ROOT = Path(__file__).resolve().parents[1]
FIELD = Path(os.environ.get("TORCH_LATTICE_FIELD", ROOT / "benchmarks/canonical/cfg_7.bin"))


def load_chroma_cfg(path, dtype):
    raw = np.fromfile(path, dtype=np.float64)
    cv = raw[0::2] + 1j * raw[1::2]
    cv = cv.reshape(4, 4, 4, 4, 8, 3, 3)
    return [jnp.asarray(cv[mu].copy(), dtype=dtype) for mu in range(4)]


def timeit(fn, *args, n=3):
    out = fn(*args)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), out)
    t0 = time.perf_counter()
    for _ in range(n):
        out = fn(*args)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), out)
    return (time.perf_counter() - t0) / n


def bench(dtype):
    U = load_chroma_cfg(FIELD, dtype)
    geo = evenodd.build_geometry(SHAPE)
    V2 = int(np.prod(SHAPE)) // 2
    key = jax.random.PRNGKey(0)
    re = jax.random.normal(key, (V2, 4, 3))
    im = jax.random.normal(jax.random.split(key)[1], (V2, 4, 3))
    phi_e = (re + 1j * im).astype(dtype)

    def solve_fn(U):
        U_e, U_o = evenodd.gather_gauge(U, geo, SHAPE)
        matvec = lambda x: evenodd.dhat_dagger_dhat_compact(U_o, U_e, x, geo, M0)
        return solve.cg(matvec, phi_e, max_iter=5000, tol=1e-7)

    s = jax.jit(solve_fn)
    out = s(U)
    it = int(out[1])
    t = timeit(s, U)
    print(f"{dtype.__name__:11s}: {it:3d} iter, {t*1e3:6.1f} ms/solve")


if __name__ == "__main__":
    bench(jnp.complex128)
    bench(jnp.complex64)
