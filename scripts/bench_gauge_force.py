"""Optimización de la fuerza de gauge (staple): roll vs gather vs batched."""
import time
import numpy as np
import jax
import jax.numpy as jnp

from tlattice_jax import core, gauge

jax.config.update("jax_enable_x64", True)

SHAPE = (4, 4, 4, 8)
Nd = 4
V = int(np.prod(SHAPE))
BETA = 6.0


def build_neighbors(shape):
    flat = np.arange(int(np.prod(shape))).reshape(shape)
    fwd = [np.roll(flat, -1, axis=mu).ravel() for mu in range(len(shape))]  # x->x+mu
    bwd = [np.roll(flat, 1, axis=mu).ravel() for mu in range(len(shape))]   # x->x-mu
    # compuesta: x -> x+mu-nu  (primero -nu, luego +mu)
    comp = {(mu, nu): fwd[mu][bwd[nu]] for mu in range(len(shape)) for nu in range(len(shape))}
    return fwd, bwd, comp


fwd, bwd, comp = build_neighbors(SHAPE)


def gauge_force_compact(U, beta):
    """Gather directo con mapas de vecinos (sin roll)."""
    Uf = [u.reshape(V, 3, 3) for u in U]
    Uf_mH = [core.mH(u) for u in Uf]
    F = []
    for mu in range(Nd):
        g = jnp.zeros((V, 3, 3), dtype=U[0].dtype)
        for nu in range(Nd):
            if nu == mu:
                continue
            U_nu_f = Uf[nu][fwd[mu]]                 # U_nu(x+mu)
            U_muH_f = Uf_mH[mu][fwd[nu]]             # U_mu^H(x+nu)
            S_f = U_nu_f @ U_muH_f @ Uf_mH[nu]
            U_nuH_b = Uf_mH[nu][bwd[nu]]             # U_nu^H(x-nu)
            U_mu_b = Uf[mu][bwd[nu]]                 # U_mu(x-nu)
            U_nu_fb = Uf[nu][comp[mu, nu]]           # U_nu(x+mu-nu)
            S_b = U_nuH_b @ U_mu_b @ U_nu_fb
            g = g - (beta / core.Nc) * (core.mH(S_f) + S_b)
        F.append(g.reshape(*SHAPE, 3, 3))
    return F


def gauge_force_batched(U, beta):
    """Gather + matmuls por lotes sobre los 3 nu (reduce dispatch)."""
    Uf = [u.reshape(V, 3, 3) for u in U]
    Uf_mH = [core.mH(u) for u in Uf]
    F = []
    for mu in range(Nd):
        others = [nu for nu in range(Nd) if nu != mu]
        U_nu_f = jnp.stack([Uf[nu][fwd[mu]] for nu in others])      # (3,V,3,3)
        U_muH_f = jnp.stack([Uf_mH[mu][fwd[nu]] for nu in others])  # (3,V,3,3)
        U_nuH = jnp.stack([Uf_mH[nu] for nu in others])             # (3,V,3,3)
        S_f = (U_nu_f @ U_muH_f) @ U_nuH                            # (3,V,3,3)
        U_nuH_b = jnp.stack([Uf_mH[nu][bwd[nu]] for nu in others])
        U_mu_b = jnp.stack([Uf[mu][bwd[nu]] for nu in others])
        U_nu_fb = jnp.stack([Uf[nu][comp[mu, nu]] for nu in others])
        S_b = (U_nuH_b @ U_mu_b) @ U_nu_fb
        g = -(beta / core.Nc) * (core.mH(S_f).sum(0) + S_b.sum(0))
        F.append(g.reshape(*SHAPE, 3, 3))
    return F


def random_su3(shape, key):
    k1, k2 = jax.random.split(key)
    A = jax.random.normal(k1, (*shape, 3, 3)) + 1j * jax.random.normal(k2, (*shape, 3, 3))
    return core.expm_batched(core.project_tah(A))


def main():
    U = [random_su3(SHAPE, k) for k in jax.random.split(jax.random.PRNGKey(7), 4)]

    ref = gauge.gauge_force_analytic(U, BETA)
    f_c = jax.jit(gauge_force_compact)
    f_b = jax.jit(gauge_force_batched)
    c = f_c(U, BETA)
    b = f_b(U, BETA)

    for name, out in [("compact", c), ("batched", b)]:
        d = max(float(jnp.max(jnp.abs(a - r))) for a, r in zip(out, ref))
        print(f"  {name:8s} vs roll: max|Δ| = {d:.3e}")

    def timeit(fn, n=50):
        fn(U, BETA)[0].block_until_ready()
        t0 = time.perf_counter()
        for _ in range(n):
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), fn(U, BETA))
        return (time.perf_counter() - t0) / n

    t_ref = timeit(jax.jit(gauge.gauge_force_analytic))
    t_c = timeit(f_c)
    t_b = timeit(f_b)
    print(f"\n  roll:     {t_ref*1e3:.3f} ms/op")
    print(f"  compact:  {t_c*1e3:.3f} ms/op  ({t_ref/t_c:.2f}×)")
    print(f"  batched:  {t_b*1e3:.3f} ms/op  ({t_ref/t_b:.2f}×)")


if __name__ == "__main__":
    main()
