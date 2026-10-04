"""expm su(3) via eigendecomposición de iA (Hermitiana). Valida y benchmarkea."""
import time
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl

from tlattice_jax import core

jax.config.update("jax_enable_x64", True)

SHAPE = (4, 4, 4, 8)


def expm_su3(A):
    """exp(A) para A traceless anti-hermitiana (su(3)).

    iA es Hermitiana → eigh(iA) = (θ reales, V ortonormal), iA = V diag(θ) V†.
    Luego A = V diag(−iθ) V† y exp(A) = V diag(e^{−iθ}) V†.
    """
    theta, V = jnp.linalg.eigh(1j * A)  # theta: (...,3) real, V: (...,3,3)
    phase = jnp.exp(-1j * theta)[..., None, :]  # (...,1,3)
    return (V * phase) @ jnp.conj(V).swapaxes(-1, -2)


def main():
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    A = jax.random.normal(k1, (*SHAPE, 3, 3)) + 1j * jax.random.normal(k2, (*SHAPE, 3, 3))
    A = core.project_tah(A)  # anti-hermitiana traceless

    # --- validación ---
    ref = core.expm_batched(A)  # scipy expm (referencia)
    new = expm_su3(A)
    print(f"expm_su3 vs scipy.expm: max|Δ| = {float(jnp.max(jnp.abs(new - ref))):.3e}")

    # unitariedad
    U = new
    un = jnp.max(jnp.abs(U @ core.mH(U) - core.EYE3[None, None, None, None]))
    print(f"unitaridad U U†−I: {float(un):.3e}")

    # --- benchmark ---
    f_ref = jax.jit(core.expm_batched)
    f_new = jax.jit(expm_su3)

    def timeit(fn, x, n=50):
        fn(x).block_until_ready()
        t0 = time.perf_counter()
        for _ in range(n):
            fn(x).block_until_ready()
        return (time.perf_counter() - t0) / n

    t_ref = timeit(f_ref, A)
    t_new = timeit(f_new, A)
    print(f"\nscipy.expm (vmap): {t_ref*1e3:.3f} ms  (2048 matrices 3×3)")
    print(f"expm_su3 (eigh):   {t_new*1e3:.3f} ms")
    print(f"→ aceleración: {t_ref/t_new:.2f}×")


if __name__ == "__main__":
    main()
