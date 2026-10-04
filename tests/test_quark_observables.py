"""Validación de observables de quarks: condensado quiral y correlador del pión."""
import torch

from tlattice import gauge, su3, quark_observables as qo

M0 = -0.5


def main():
    torch.manual_seed(0)
    ok = True
    shape = (4, 4, 4, 8)

    print("== condensado quiral (gauge trivial, libre) ==")
    U0 = gauge.unit_gauge(shape, dtype=torch.complex128)
    c0 = qo.chiral_condensate(U0, M0, n_noise=8)
    print(f"  <psibar psi> libre = {c0:.6f}")
    ok &= (0 < c0 < 50)  # finito y positivo

    print("\n== condensado quiral (gauge aleatorio) ==")
    Ur = [su3.random_su3(*shape, dtype=torch.complex128) for _ in range(4)]
    c1 = qo.chiral_condensate(Ur, M0, n_noise=8)
    print(f"  <psibar psi> interactuante = {c1:.6f}")
    ok &= (0 < c1 < 50)

    print("\n== correlador del pión (gauge trivial) ==")
    c_t, it, resid = qo.pion_correlator(U0, M0)
    vals = c_t.tolist()
    print(f"  C(t) = {[f'{x:.5f}' for x in vals]}")
    print(f"  (CG: {it} iter, residuo {resid:.2e})")
    # C(0) debe ser el máximo y decaer al inicio
    ok &= vals[0] > vals[1] > 0
    ok &= all(v >= 0 for v in vals)

    print("\n" + ("=== TODO OK ===" if ok else "=== HAY FALLAS ==="))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
