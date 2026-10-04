"""Valida el hop/D̂ compacto contra la versión scatter/gather, y benchmarkea."""
import time
import numpy as np
import jax
import jax.numpy as jnp

from tlattice_jax import core, dirac, evenodd, hmc, hmc_eo, solve

jax.config.update("jax_enable_x64", True)

BETA, M0, SHAPE = 6.0, -0.5, (4, 4, 4, 8)


def random_su3(shape, key):
    k1, k2 = jax.random.split(key)
    A = jax.random.normal(k1, (*shape, 3, 3)) + 1j * jax.random.normal(k2, (*shape, 3, 3))
    return core.expm_batched(core.project_tah(A))


def main():
    key = jax.random.PRNGKey(123)
    U = [random_su3(SHAPE, k) for k in jax.random.split(key, 4)]
    even_idx, odd_idx = evenodd.checkerboard(SHAPE)
    geo = evenodd.build_geometry(SHAPE)
    V = int(np.prod(SHAPE))

    # psi_e aleatorio (compacto)
    k1, k2 = jax.random.split(key)
    psi_e = jax.random.normal(k1, (np.prod(SHAPE) // 2, 4, 3)) + 1j * jax.random.normal(k2, (np.prod(SHAPE) // 2, 4, 3))

    # --- 1) valida D̂†D̂ compacto vs scatter/gather ---
    U_even, U_odd = evenodd.gather_gauge(U, geo, SHAPE)
    ddd_scatter = evenodd.dhat_dagger_dhat(U, psi_e, even_idx, SHAPE, M0)
    ddd_compact = evenodd.dhat_dagger_dhat_compact(U_odd, U_even, psi_e, geo, M0)
    diff = float(jnp.max(jnp.abs(ddd_compact - ddd_scatter)))
    print(f"D̂†D̂ compacto vs scatter/gather: max|Δ| = {diff:.3e}")

    # --- 2) valida el hop_oe contra el hop completo ---
    psi_full = jnp.zeros((V, 4, 3), dtype=psi_e.dtype).at[even_idx].set(psi_e).reshape(*SHAPE, 4, 3)
    hop_full = evenodd.hop(U, psi_full)
    hop_full_odd = hop_full.reshape(-1, 4, 3)[odd_idx]
    hop_c = evenodd.hop_oe(U_odd, U_even, psi_e, geo)
    diff_hop = float(jnp.max(jnp.abs(hop_full_odd - hop_c)))
    print(f"hop_oe compacto vs hop completo:     max|Δ| = {diff_hop:.3e}")

    # --- 3) benchmark del solve ---
    key, sk = jax.random.split(key)
    eta = jax.random.normal(sk, (*SHAPE, 4, 3)) + 1j * jax.random.normal(jax.random.split(sk)[1], (*SHAPE, 4, 3))
    eta = eta / jnp.sqrt(2.0)
    phi = dirac.d_dagger(U, eta, M0)
    phi_e, _ = evenodd.split_eo(phi, even_idx, odd_idx)

    solve_unp = jax.jit(lambda U: hmc.solve_chi(U, phi, M0))
    solve_eo_old = jax.jit(lambda U: solve.cg(
        lambda x: evenodd.dhat_dagger_dhat(U, x, even_idx, SHAPE, M0),
        phi_e, max_iter=5000, tol=1e-12,
    ))

    def solve_eo_compact(U):
        U_e, U_o = evenodd.gather_gauge(U, geo, SHAPE)
        def matvec(x):
            return evenodd.dhat_dagger_dhat_compact(U_o, U_e, x, geo, M0)
        return solve.cg(matvec, phi_e, max_iter=5000, tol=1e-12)

    solve_eo_new = jax.jit(solve_eo_compact)

    def timeit(fn, *args, n=5):
        out = fn(*args)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), out)
        t0 = time.perf_counter()
        for _ in range(n):
            out = fn(*args)
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), out)
        return (time.perf_counter() - t0) / n

    it_unp = int(solve_unp(U)[1])
    it_eo_old = int(solve_eo_old(U)[1])
    it_eo_new = int(solve_eo_new(U)[1])
    t_unp = timeit(solve_unp, U)
    t_eo_old = timeit(solve_eo_old, U)
    t_eo_new = timeit(solve_eo_new, U)
    print(f"\nSOLVE (it, ms):  sin precond {it_unp} it, {t_unp*1e3:.1f} ms")
    print(f"                 par-impar viejo {it_eo_old} it, {t_eo_old*1e3:.1f} ms")
    print(f"                 par-impar compacto {it_eo_new} it, {t_eo_new*1e3:.1f} ms")
    print(f"→ aceleración compacto vs sin precond: {t_unp/t_eo_new:.2f}×")
    print(f"→ aceleración compacto vs par-impar viejo: {t_eo_old/t_eo_new:.2f}×")


if __name__ == "__main__":
    main()
