"""Benchmark HMC GAUGE-ONLY (Wilson gauge, sin fermiones): JAX vs Chroma.

Aísla el kernel de la fuerza de gauge (staple), sin CG. Misma física que Chroma
gauge_only.ini.xml: beta=6.0, tau=1.0 (10 pasos), doble precisión.
"""
import time
import numpy as np
import jax
import jax.numpy as jnp

from tlattice_jax import core, gauge, hmc

jax.config.update("jax_enable_x64", True)

BETA, SHAPE = 6.0, (4, 4, 4, 8)


def random_su3(shape, key):
    k1, k2 = jax.random.split(key)
    A = jax.random.normal(k1, (*shape, 3, 3)) + 1j * jax.random.normal(k2, (*shape, 3, 3))
    return core.expm_batched(core.project_tah(A))


def gauge_force(U, beta):
    """Fuerza de gauge proyectada a su(3): F = project_tah(0.5(U g† − g U†))."""
    g = gauge.gauge_force_analytic(U, beta)
    return [core.project_tah(0.5 * (u @ core.mH(gg) - gg @ core.mH(u)))
            for gg, u in zip(g, U)]


def hmc_gauge_step(U, beta, n_steps, eps, key):
    """Un paso HMC de gauge puro. Devuelve (U, aceptado, deltaH)."""
    shape = U[0].shape[:4]
    key, sk = jax.random.split(key)
    P, key = hmc.sample_momentum(sk, shape)
    U0 = U
    H0 = hmc.kinetic(P) + gauge.wilson_action(U, beta)

    F = gauge_force(U, beta)
    P = [p + 0.5 * eps * f for p, f in zip(P, F)]
    for step in range(n_steps):
        U = [core.expm_batched(eps * p) @ u for p, u in zip(P, U)]
        F = gauge_force(U, beta)
        if step < n_steps - 1:
            P = [p + eps * f for p, f in zip(P, F)]
        else:
            P = [p + 0.5 * eps * f for p, f in zip(P, F)]

    H1 = hmc.kinetic(P) + gauge.wilson_action(U, beta)
    dH = H1 - H0
    key, sk = jax.random.split(key)
    accept = jax.random.uniform(sk) < jnp.exp(-dH)
    U = [jnp.where(accept, a, b) for a, b in zip(U, U0)]
    return U, accept, dH


def timeit(fn, *args, n=30):
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
    step = jax.jit(lambda U, k: hmc_gauge_step(U, BETA, 10, 0.1, k))

    # validación: ΔH debe ser pequeño (integración de gauge puro)
    _, acc, dH = step(U, jax.random.PRNGKey(1))
    print(f"validación gauge-only JAX: accept={bool(acc)} dH={float(dH):.3e}")

    t = timeit(step, U, jax.random.PRNGKey(2))
    print(f"JAX gauge-only (1 hilo): {t*1e3:.2f} ms/traj  (n=30)")


if __name__ == "__main__":
    main()
