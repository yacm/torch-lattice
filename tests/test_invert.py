"""Validación del inversor CG y del solve del operador de Dirac (M4)."""
import torch

from tlattice import config, gauge, dirac, invert


def main():
    torch.manual_seed(0)
    print(f"device: {config.device()}")
    ok = True

    shape = (4, 4, 4, 8)
    m0 = -0.5  # kappa = 1/(8+2m0) = 1/7, masa física positiva

    print("\n== CG sobre ecuaciones normales (gauge trivial, masa m0=-0.5) ==")
    U = gauge.unit_gauge(shape)
    b = torch.randn(*shape, 4, 3, dtype=torch.complex64)
    x, it, resid = invert.solve_dirac(U, b, m0, tol=1e-9, max_iter=3000)

    Dx = dirac.wilson_dslash(U, x, m0)
    err = (Dx - b).norm() / b.norm()
    print(f"  iteraciones: {it}")
    print(f"  residuo (norm. eqs): {resid:.2e}")
    print(f"  ||D x - b|| / ||b|| = {err:.2e}")
    ok &= err < 1e-5

    # gauge aleatorio (no trivial) también debe resolver
    print("\n== CG sobre gauge aleatorio ==")
    U_r = gauge.random_gauge(shape)
    b2 = torch.randn(*shape, 4, 3, dtype=torch.complex64)
    x2, it2, resid2 = invert.solve_dirac(U_r, b2, m0, tol=1e-7, max_iter=5000)
    Dx2 = dirac.wilson_dslash(U_r, x2, m0)
    err2 = (Dx2 - b2).norm() / b2.norm()
    print(f"  iteraciones: {it2}, residuo: {resid2:.2e}, ||Dx-b||/||b|| = {err2:.2e}")
    ok &= err2 < 1e-4

    print("\n" + ("=== TODO OK ===" if ok else "=== HAY FALLAS ==="))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
