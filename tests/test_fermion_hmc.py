"""Validación del HMC con fermiones dinámicos (2 sabores Wilson).

1. Fuerza fermiónica vs derivada numérica (signo y magnitud).
2. Épsilon test: deltaH ~ O(eps^2) (HMC exacto, chi re-resuelto cada paso).
"""
import torch

from tlattice import fermion_hmc as fh
from tlattice import gauge, su3

M0 = -0.5


def _random_config(shape, seed=0):
    torch.manual_seed(seed)
    return [fh.make_leaf(su3.random_su3(*shape, dtype=torch.complex128))
            for _ in range(4)]


def _s_pf(U, phi, m0):
    chi, _, _ = fh.solve_chi(U, phi, m0)
    return fh.pseudofermion_action_value(phi, chi).item()


def test_force():
    shape = (2, 2, 2, 4)
    U = _random_config(shape)
    phi = fh.sample_phi(U, M0)
    chi, it, resid = fh.solve_chi(U, phi, M0)
    print(f"  CG: {it} iter, residuo {resid:.2e}")

    F_PF = fh.force_total(U, 0.0, phi, M0)  # beta=0 -> solo fermiónica
    P = fh.sample_momentum(shape)
    theo = sum(torch.einsum("...ij,...ji->...", f, p).real.sum().item()
               for f, p in zip(F_PF, P))

    eps = 1e-4
    U_plus = [(torch.matrix_exp(eps * p) @ u).detach() for p, u in zip(P, U)]
    U_minus = [(torch.matrix_exp(-eps * p) @ u).detach() for p, u in zip(P, U)]
    num = (_s_pf(U_plus, phi, M0) - _s_pf(U_minus, phi, M0)) / (2 * eps)
    ratio = theo / num
    print(f"  teórico={theo:+.8f}  numérico={num:+.8f}  cociente={ratio:.5f}")
    return abs(ratio - 1.0) < 1e-5


def test_epsilon():
    shape = (2, 2, 2, 4)
    T = 1.0
    U0 = _random_config(shape, seed=1)
    P0 = fh.sample_momentum(shape)
    phi = fh.sample_phi(U0, M0)

    print("  eps       n_steps    deltaH        deltaH/eps^2")
    prev = None
    ratios = []
    for eps in [0.1, 0.05, 0.025]:
        n = round(T / eps)
        chi0, _, _ = fh.solve_chi(U0, phi, M0)
        H0 = fh.hamiltonian_value(U0, 6.0, P0, phi, chi0, M0).item()
        Ut, Pt = fh.leapfrog_fermion(U0, P0, 6.0, phi, M0, n, eps)
        chi1, _, _ = fh.solve_chi(Ut, phi, M0)
        H1 = fh.hamiltonian_value(Ut, 6.0, Pt, phi, chi1, M0).item()
        dH = abs(H1 - H0)
        ratios.append(dH / eps ** 2)
        print(f"  {eps:.3f}    {n:3d}       {dH:+.4e}   {dH/eps**2:.4e}")
        if prev is not None:
            r = (dH / eps ** 2) / prev
            assert 0.2 < r < 5.0, f"deltaH/eps^2 no constante: {r}"
        prev = dH / eps ** 2
    print(f"  deltaH/eps^2 ~ constante (cociente max/min = "
          f"{max(ratios)/min(ratios):.2f})")
    return max(ratios) / min(ratios) < 5.0


def main():
    print("== 1. fuerza fermiónica vs derivada numérica ==")
    ok1 = test_force()
    print("== 2. épsilon test (HMC exacto) ==")
    ok2 = test_epsilon()
    print("\n" + ("=== TODO OK ===" if (ok1 and ok2) else "=== HAY FALLAS ==="))
    return 0 if (ok1 and ok2) else 1


if __name__ == "__main__":
    raise SystemExit(main())
