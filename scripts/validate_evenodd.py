"""Valida el operador par-impar (D̂) contra el Dslash completo.

Chequeos:
  1. D̂ = complemento de Schur de D  (identidad exacta, para CUALQUIER U).
  2. D̂ es γ5-hermítico: γ5 D̂ γ5 = D̂†.
  3. El CG sobre D̂†D̂ converge (residual pequeño).
"""
import numpy as np
import jax
import jax.numpy as jnp

from tlattice_jax import core, dirac, evenodd, solve

jax.config.update("jax_enable_x64", True)

M0, SHAPE = -0.5, (4, 4, 4, 8)


def random_su3(shape, key):
    k1, k2 = jax.random.split(key)
    A = jax.random.normal(k1, (*shape, 3, 3)) + 1j * jax.random.normal(k2, (*shape, 3, 3))
    return core.expm_batched(core.project_tah(A))


def random_spinor(shape, key):
    k1, k2 = jax.random.split(key)
    return (jax.random.normal(k1, (*shape, 4, 3))
            + 1j * jax.random.normal(k2, (*shape, 4, 3)))


def main():
    key = jax.random.PRNGKey(0)
    U = [random_su3(SHAPE, k) for k in jax.random.split(key, 4)]

    even_idx, odd_idx = evenodd.checkerboard(SHAPE)
    V = int(np.prod(SHAPE))
    print(f"lattice {SHAPE}: V={V}, V/2={even_idx.shape[0]} pares, {odd_idx.shape[0]} impares")

    # --- 1. D̂ = complemento de Schur de D ---
    key, sk = jax.random.split(key)
    psi_e = random_spinor((even_idx.shape[0],), sk)  # (V/2,4,3)

    # reconstruir ψ_full con ψ_o = (0.5/(4+m0)) H_oe ψ_e  (ecuación impar con b_o=0)
    full_e = jnp.zeros((V, 4, 3), dtype=jnp.complex128).at[even_idx].set(psi_e)
    full_e = full_e.reshape(*SHAPE, 4, 3)
    H_oe_psi = evenodd.hop(U, full_e)                 # vive en impares
    psi_o = (0.5 / (4.0 + M0)) * H_oe_psi.reshape(V, 4, 3)[odd_idx]
    psi_full = evenodd.recombine(psi_e, psi_o, even_idx, odd_idx, SHAPE)

    Dpsi = dirac.dslash(U, psi_full, M0)
    Dpsi_e = Dpsi.reshape(V, 4, 3)[even_idx]           # parte par de D ψ_full
    dhat_psi = evenodd.dhat(U, psi_e, even_idx, SHAPE, M0)
    d1 = float(jnp.max(jnp.abs(Dpsi_e - dhat_psi)))
    print(f"1. D̂ = Schur(D):  max|Dψ_e − D̂ψ_e| = {d1:.2e}")

    # --- 2. γ5-hermiticidad: γ5 D̂ γ5 = D̂† ---
    g5 = dirac.apply_gamma5
    lhs = g5(evenodd.dhat(U, g5(psi_e), even_idx, SHAPE, M0))   # γ5 D̂ γ5 ψ
    rhs = evenodd.dhat_dagger(U, psi_e, even_idx, SHAPE, M0)    # D̂† ψ
    d2 = float(jnp.max(jnp.abs(lhs - rhs)))
    print(f"2. γ5-hermiticidad:  max|γ5D̂γ5ψ − D̂†ψ| = {d2:.2e}")

    # --- 3. CG sobre D̂†D̂ converge ---
    key, sk = jax.random.split(key)
    phi_e = random_spinor((even_idx.shape[0],), sk)
    matvec = lambda x: evenodd.dhat_dagger_dhat(U, x, even_idx, SHAPE, M0)
    chi_e, it, resid = solve.cg(matvec, phi_e, max_iter=3000, tol=1e-11)
    # residual explícito como verificación independiente
    r = phi_e - matvec(chi_e)
    rn = float(jnp.sqrt(jnp.real(jnp.sum(jnp.conj(r) * r)))
               / jnp.sqrt(jnp.real(jnp.sum(jnp.conj(phi_e) * phi_e))))
    print(f"3. CG(D̂†D̂):       {it} iter, resid(rel) reportado={resid:.2e}, "
          f"explícito={rn:.2e}")

    ok = d1 < 1e-11 and d2 < 1e-11 and rn < 1e-10
    print("\n" + ("=== PAR-IMPAR OK ===" if ok else "=== HAY FALLAS ==="))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
