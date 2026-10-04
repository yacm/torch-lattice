"""Validación del HMC clover (2 sabores, Wilson-clover + gauge Symanzik).

1. Fuerza de gauge Symanzik vs derivada numérica.
2. Fuerza fermiónica clover (a través del stout smearing) vs derivada numérica.
3. Épsilon test: deltaH ~ O(eps^2) (HMC exacto).
"""
import torch

from tlattice import clover_hmc as ch
from tlattice import gauge, su3

M0 = -0.5
C_SW = 1.0
RHO = 0.1


def _random_config(shape, seed=0):
    torch.manual_seed(seed)
    return [ch.make_leaf(su3.random_su3(*shape, dtype=torch.complex128))
            for _ in range(4)]


def _s_pf(U, phi, m0, c_sw, rho):
    chi, _, _ = ch.solve_chi(U, phi, m0, c_sw, rho)
    return ch.pseudofermion_action_value(phi, chi).item()


def _s_gauge(U, beta, u0):
    return gauge.symanzik_action(U, beta, u0).item()


def test_gauge_force():
    print("== 1. fuerza de gauge Symanzik vs numérica ==")
    shape = (2, 2, 2, 4)
    U = _random_config(shape, seed=2)
    beta, u0 = 6.0, 0.86
    F = ch.gauge_force(U, beta, u0)
    P = ch.sample_momentum(shape)
    theo = sum(torch.einsum("...ij,...ji->...", f, p).real.sum().item()
               for f, p in zip(F, P))
    eps = 1e-4
    U_plus = [(torch.matrix_exp(eps * p) @ u).detach() for p, u in zip(P, U)]
    U_minus = [(torch.matrix_exp(-eps * p) @ u).detach() for p, u in zip(P, U)]
    num = (_s_gauge(U_plus, beta, u0) - _s_gauge(U_minus, beta, u0)) / (2 * eps)
    ratio = theo / num
    print(f"  teórico={theo:+.8f}  numérico={num:+.8f}  cociente={ratio:.5f}")
    return abs(ratio - 1.0) < 1e-5


def test_fermion_force():
    print("== 2. fuerza fermiónica clover (stout) vs numérica ==")
    shape = (2, 2, 2, 4)
    U = _random_config(shape, seed=3)
    phi = ch.sample_phi(U, M0, C_SW, RHO)
    chi, it, resid = ch.solve_chi(U, phi, M0, C_SW, RHO)
    print(f"  CG: {it} iter, residuo {resid:.2e}")

    F = ch.force_total(U, 0.0, phi, M0, C_SW, RHO, u0=0.86)  # beta=0 -> solo fermiónica
    P = ch.sample_momentum(shape)
    theo = sum(torch.einsum("...ij,...ji->...", f, p).real.sum().item()
               for f, p in zip(F, P))
    eps = 1e-4
    U_plus = [(torch.matrix_exp(eps * p) @ u).detach() for p, u in zip(P, U)]
    U_minus = [(torch.matrix_exp(-eps * p) @ u).detach() for p, u in zip(P, U)]
    num = (_s_pf(U_plus, phi, M0, C_SW, RHO) - _s_pf(U_minus, phi, M0, C_SW, RHO)) / (2 * eps)
    ratio = theo / num
    print(f"  teórico={theo:+.8f}  numérico={num:+.8f}  cociente={ratio:.5f}")
    return abs(ratio - 1.0) < 1e-5


def test_epsilon():
    print("== 3. épsilon test (HMC exacto) ==")
    shape = (2, 2, 2, 4)
    T = 1.0
    beta, u0 = 6.0, 0.86
    U0 = _random_config(shape, seed=4)
    P0 = ch.sample_momentum(shape)
    phi = ch.sample_phi(U0, M0, C_SW, RHO)

    print("  eps       n_steps    deltaH        deltaH/eps^2")
    ratios = []
    for eps in [0.1, 0.05, 0.025]:
        n = round(T / eps)
        chi0, _, _ = ch.solve_chi(U0, phi, M0, C_SW, RHO)
        H0 = ch.hamiltonian_value(U0, beta, P0, phi, chi0, M0, C_SW, RHO, u0).item()
        Ut, Pt = ch.leapfrog(U0, P0, beta, phi, M0, C_SW, RHO, u0, n, eps)
        chi1, _, _ = ch.solve_chi(Ut, phi, M0, C_SW, RHO)
        H1 = ch.hamiltonian_value(Ut, beta, Pt, phi, chi1, M0, C_SW, RHO, u0).item()
        dH = abs(H1 - H0)
        ratios.append(dH / eps ** 2)
        print(f"  {eps:.3f}    {n:3d}       {dH:+.4e}   {dH/eps**2:.4e}")
    print(f"  deltaH/eps^2 ~ constante (cociente max/min = "
          f"{max(ratios)/min(ratios):.2f})")
    return max(ratios) / min(ratios) < 5.0


def main():
    ok1 = test_gauge_force()
    ok2 = test_fermion_force()
    ok3 = test_epsilon()
    print("\n" + ("=== TODO OK ===" if (ok1 and ok2 and ok3) else "=== HAY FALLAS ==="))
    return 0 if (ok1 and ok2 and ok3) else 1


if __name__ == "__main__":
    raise SystemExit(main())
