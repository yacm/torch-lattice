"""Valida el port JAX (tlattice_jax) contra el torch (tlattice) bit-compatible."""
import numpy as np
import torch
import jax
import jax.numpy as jnp

from tlattice import su3 as tsu3, force_analytic as tfa, fermion_hmc as tfh
from tlattice_jax import dirac as jdirac, gauge as jgauge, hmc as jhmc

BETA, M0, SHAPE = 6.0, -0.5, (4, 4, 4, 8)


def _to_np(x):
    return np.asarray(x.numpy() if hasattr(x, "numpy") else x)


def _max_diff(a, b):
    return max(float(np.max(np.abs(_to_np(x) - _to_np(y)))) for x, y in zip(a, b))


def main():
    torch.manual_seed(42)
    U_t = [tsu3.random_su3(*SHAPE, dtype=torch.complex128) for _ in range(4)]
    U_j = [jnp.asarray(u.numpy()) for u in U_t]
    phi_t = tfh.sample_phi(U_t, M0)
    chi_t, it, r = tfh.solve_chi(U_t, phi_t, M0)
    chi_j = jnp.asarray(chi_t.numpy())

    ok = True
    d = _max_diff(tfa.gauge_force_analytic(U_t, BETA),
                  jgauge.gauge_force_analytic(U_j, BETA))
    print(f"  gauge_force    max|Δ| = {d:.2e}"); ok &= d < 1e-12
    d = _max_diff(tfa.fermion_force_analytic(U_t, chi_t, M0),
                  jhmc.fermion_force_analytic(U_j, chi_j, M0))
    print(f"  fermion_force  max|Δ| = {d:.2e}"); ok &= d < 1e-12
    d = _max_diff(tfa.force_total_analytic(U_t, BETA, phi_t, M0),
                  jhmc.force_total(U_j, BETA, jnp.asarray(phi_t.numpy()), M0))
    print(f"  force_total    max|Δ| = {d:.2e}"); ok &= d < 1e-12

    # 1 paso HMC JAX corre y conserva su(3) en los enlaces
    U_j2, acc, dH = jhmc.hmc_step(U_j, BETA, M0, 10, 0.1, jax.random.PRNGKey(0))
    dets = [float(jnp.abs(jnp.linalg.det(u)).mean()) for u in U_j2]
    print(f"  hmc_step: accept={bool(acc)} dH={float(dH):+.3e} |det U|~{np.mean(dets):.6f}")
    ok &= all(abs(x - 1.0) < 1e-4 for x in dets)

    print("\n" + ("=== TODO OK ===" if ok else "=== HAY FALLAS ==="))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
