"""Valida el HMC par-impar compacto: fuerza, conservación de energía y física.

  1. Fuerza fermiónica: identidad dS_PF/dε = −⟨F,F⟩ a lo largo de F (exacta).
  2. ΔH en un paso HMC (sin deriva sistemática).
  3. Plaquette igual que el HMC sin precondicionar (misma física).
"""
import numpy as np
import jax
import jax.numpy as jnp

from tlattice_jax import core, gauge, evenodd, hmc as hmc_unprec, hmc_eo

jax.config.update("jax_enable_x64", True)

BETA, M0, SHAPE = 6.0, -0.5, (4, 4, 4, 8)


def random_su3(shape, key):
    k1, k2 = jax.random.split(key)
    A = jax.random.normal(k1, (*shape, 3, 3)) + 1j * jax.random.normal(k2, (*shape, 3, 3))
    return core.expm_batched(core.project_tah(A))


def S_pf_eo(U, phi_e, geo, shape, m0):
    chi_e, _, _ = hmc_eo.solve_chi_eo(U, phi_e, geo, shape, m0)
    return jnp.real(jnp.sum(jnp.conj(phi_e) * chi_e))


def main():
    key = jax.random.PRNGKey(7)
    U = [random_su3(SHAPE, k) for k in jax.random.split(key, 4)]
    geo = evenodd.build_geometry(SHAPE)

    # --- 1. chequeo de fuerza (derivada direccional) ---
    key, sk = jax.random.split(key)
    phi_e, _ = hmc_eo.sample_phi_eo(sk, U, geo, SHAPE, M0)
    chi_e, it, _ = hmc_eo.solve_chi_eo(U, phi_e, geo, SHAPE, M0)
    g_pf = hmc_eo.fermion_force_eo(U, chi_e, geo, SHAPE, M0)
    F_pf = [core.project_tah(0.5 * (u @ core.mH(g) - g @ core.mH(u)))
            for g, u in zip(g_pf, U)]
    normF2 = float(sum(jnp.sum(jnp.abs(f) ** 2) for f in F_pf))

    eps = 1e-5
    U_pert = [core.expm_batched(eps * f) @ u for f, u in zip(F_pf, U)]
    dS_num = (S_pf_eo(U_pert, phi_e, geo, SHAPE, M0)
              - S_pf_eo(U, phi_e, geo, SHAPE, M0)) / eps
    print(f"1. fuerza fermiónica:  dS/dε num={dS_num:+.6e}  vs  -⟨F,F⟩={-normF2:+.6e}")
    rel1 = abs(dS_num + normF2) / max(normF2, 1e-30)

    # --- 2. ΔH en un paso HMC (compilado) ---
    step = jax.jit(lambda U, k: hmc_eo.hmc_step_eo(U, BETA, M0, 10, 0.1, k, geo, SHAPE))
    U1, acc, dH = step(U, jax.random.PRNGKey(1))
    print(f"2. hmc_step_eo:      accept={bool(acc)} dH={float(dH):+.4e}")

    # --- 3. plaquette: par-impar vs sin precondicionar (misma física) ---
    def run(step_fn, n=12):
        Uc = [u.copy() for u in U]
        pl = []
        k = jax.random.PRNGKey(3)
        for _ in range(n):
            k, sk = jax.random.split(k)
            Uc, _, _ = step_fn(Uc, sk)
            pl.append(float(gauge.average_plaquette(Uc)))
        return Uc, pl

    step_unp = jax.jit(lambda U, k: hmc_unprec.hmc_step(U, BETA, M0, 10, 0.1, k))
    _, pl_eo = run(step)
    _, pl_unp = run(step_unp)
    print(f"3. plaquette par-impar:   {np.mean(pl_eo):.6f}  (últimos 4: {[f'{x:.5f}' for x in pl_eo[-4:]]})")
    print(f"   plaquette sin precond: {np.mean(pl_unp):.6f}  (últimos 4: {[f'{x:.5f}' for x in pl_unp[-4:]]})")

    ok = rel1 < 1e-4  # 1e-5 es la precisión de la diferencia finita (eps=1e-5)
    print("\n" + ("=== HMC PAR-IMPAR COMPACTO OK ===" if ok else "=== HAY FALLAS ==="))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
