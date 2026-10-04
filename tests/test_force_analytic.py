"""Validación de la fuerza analítica (sin autograd) contra autograd.

1. Fuerza gauge analítica vs autograd.
2. Fuerza fermiónica analítica vs autograd.
3. Fuerza total proyectada su(3) analítica vs autograd.
4. Épsilon test del HMC usando la fuerza analítica (deltaH ~ O(eps^2)).
"""
import torch

from tlattice import config, dirac, fermion_hmc as fh, force_analytic as fa
from tlattice import gauge, su3

BETA = 6.0
M0 = -0.5


def _config(shape, seed=0):
    torch.manual_seed(seed)
    return [fh.make_leaf(su3.random_su3(*shape, dtype=torch.complex128))
            for _ in range(4)]


def _max_diff(a, b):
    return max((x - y).abs().max().item() for x, y in zip(a, b))


def test_gauge_force():
    shape = (2, 2, 2, 4)
    U = _config(shape)
    S = gauge.wilson_action(U, BETA)
    g_ad = torch.autograd.grad(S, U, create_graph=False)
    g_an = fa.gauge_force_analytic(U, BETA)
    d = _max_diff(g_ad, g_an)
    print(f"  gauge: max|Δ| = {d:.2e}")
    return d < 1e-12


def test_fermion_force():
    shape = (2, 2, 2, 4)
    U = _config(shape)
    phi = fh.sample_phi(U, M0)
    chi, it, resid = fh.solve_chi(U, phi, M0)
    print(f"  CG: {it} iter, resid {resid:.2e}")
    Dchi = dirac.wilson_dslash(U, chi, M0)
    S = -(Dchi.conj() * Dchi).sum().real
    g_ad = torch.autograd.grad(S, U, create_graph=False)
    g_an = fa.fermion_force_analytic(U, chi, M0)
    d = _max_diff(g_ad, g_an)
    print(f"  fermión: max|Δ| = {d:.2e}")
    return d < 1e-12


def test_total_force():
    shape = (2, 2, 2, 4)
    U = _config(shape, seed=1)
    phi = fh.sample_phi(U, M0)
    F_ad = fh.force_total(U, BETA, phi, M0)
    F_an = fa.force_total_analytic(U, BETA, phi, M0)
    d = _max_diff(F_ad, F_an)
    print(f"  total su(3): max|Δ| = {d:.2e}")
    return d < 1e-12


def test_epsilon_analytic():
    shape = (2, 2, 2, 4)
    T = 1.0
    U0 = _config(shape, seed=2)
    P0 = fh.sample_momentum(shape)
    phi = fh.sample_phi(U0, M0)

    prev = None
    ratios = []
    for eps in [0.1, 0.05, 0.025]:
        n = round(T / eps)
        chi0, _, _ = fh.solve_chi(U0, phi, M0)
        H0 = fh.hamiltonian_value(U0, BETA, P0, phi, chi0, M0).item()
        Ut, Pt = fh.leapfrog_fermion(U0, P0, BETA, phi, M0, n, eps,
                                     force_fn=fa.force_total_analytic)
        chi1, _, _ = fh.solve_chi(Ut, phi, M0)
        H1 = fh.hamiltonian_value(Ut, BETA, Pt, phi, chi1, M0).item()
        dH = abs(H1 - H0)
        ratios.append(dH / eps ** 2)
        if prev is not None:
            r = (dH / eps ** 2) / prev
            assert 0.2 < r < 5.0, f"deltaH/eps^2 no constante: {r}"
        prev = dH / eps ** 2
    print(f"  deltaH/eps^2 ~ constante (max/min = {max(ratios)/min(ratios):.2f})")
    return max(ratios) / min(ratios) < 5.0


def main():
    print("== 1. fuerza gauge analítica vs autograd ==")
    ok1 = test_gauge_force()
    print("== 2. fuerza fermiónica analítica vs autograd ==")
    ok2 = test_fermion_force()
    print("== 3. fuerza total su(3) analítica vs autograd ==")
    ok3 = test_total_force()
    print("== 4. épsilon test con fuerza analítica ==")
    ok4 = test_epsilon_analytic()
    ok = ok1 and ok2 and ok3 and ok4
    print("\n" + ("=== TODO OK ===" if ok else "=== HAY FALLAS ==="))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
