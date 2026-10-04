"""Perfil real del CG con campo SU(3) (no aleatorio): dónde se va el tiempo."""
import time
import numpy as np
import torch
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from tlattice import su3 as tsu3
from tlattice_jax import dirac, hmc, solve

SHAPE = (4, 4, 4, 8)
M0 = -0.5


def med(fn, n=30, warm=10):
    fn()  # compilar
    for _ in range(warm):
        fn().block_until_ready() if hasattr(fn(), "block_until_ready") else None
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        r = fn()
        if hasattr(r, "block_until_ready"):
            r.block_until_ready()
        ts.append(time.perf_counter() - t0)
    return np.median(ts) * 1e3


def main():
    torch.manual_seed(42)
    U_t = [tsu3.random_su3(*SHAPE, dtype=torch.complex128) for _ in range(4)]
    U = [jnp.asarray(u.numpy()) for u in U_t]
    key = jax.random.PRNGKey(0)
    key, sk = jax.random.split(key)
    psi = jax.random.normal(sk, (*SHAPE, 4, 3), dtype=jnp.complex128)
    key, sk = jax.random.split(key)
    phi = jax.random.normal(sk, (*SHAPE, 4, 3), dtype=jnp.complex128)

    dslash_j = jax.jit(lambda p: dirac.dslash(U, p, M0))
    matvec_j = jax.jit(lambda p: dirac.d_dagger(U, dirac.dslash(U, p, M0), M0))
    solve_j = jax.jit(lambda p: solve.cg(lambda x: dirac.d_dagger(U, dirac.dslash(U, x, M0), M0), p, max_iter=5000, tol=1e-12))

    # warmup separado
    dslash_j(psi).block_until_ready()
    matvec_j(psi).block_until_ready()
    chi, iters, resid = solve_j(phi)
    iters = int(iters)

    # medir
    d_ms = med(lambda: dslash_j(psi), n=40)
    mv_ms = med(lambda: matvec_j(psi), n=40)
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        solve_j(phi)[0].block_until_ready()
        ts.append(time.perf_counter() - t0)
    s_ms = np.median(ts) * 1e3

    print(f"dslash           : {d_ms:.3f} ms")
    print(f"matvec (D†D)     : {mv_ms:.3f} ms   (= 2 dslash + 2 γ5)")
    print(f"  → overhead matvec = {mv_ms - 2*d_ms:.3f} ms  (2 γ5)")
    print(f"solve_chi (CG)   : {s_ms:.1f} ms  en {iters} iter")
    print(f"  → {s_ms/iters:.3f} ms/iter")
    print(f"  → overhead/iter (no-dslash) = {s_ms/iters - mv_ms:.3f} ms  (reducciones + axpy)")
    print(f"\nConclusión: Dslash {d_ms:.2f} ms vs Chroma ~0.69 ms → gap Dslash {d_ms/0.69:.2f}×")
    print(f"             overhead CG {s_ms/iters - mv_ms:.3f} ms/iter × {iters} × 13 solves = {(s_ms/iters - mv_ms)*iters*13/1000:.1f} s/traj de overhead")


if __name__ == "__main__":
    main()
