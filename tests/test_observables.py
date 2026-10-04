"""Validación de observables: lazo de Polyakov y lazos de Wilson (M6)."""
import torch

from tlattice import config, gauge, observables


def main():
    torch.manual_seed(0)
    print(f"device: {config.device()}")
    ok = True

    shape = (4, 4, 4, 8)

    print("\n== lazo de Polyakov ==")
    U = gauge.unit_gauge(shape)
    p = observables.average_polyakov(U)
    print(f"  gauge trivial: <ReTr P>/Nc = {p:.6f} (esperado 1.0)")
    ok &= abs(p - 1.0) < 1e-6

    U_r = gauge.random_gauge(shape)
    p_r = observables.average_polyakov(U_r)
    print(f"  gauge aleatorio: {p_r:.6f} (esperado |.| < 0.3, desordenado)")
    ok &= abs(p_r) < 0.3

    print("\n== lazo de Wilson rectangular ==")
    w = observables.wilson_loop(U, R=1, T=1)
    print(f"  gauge trivial, R=T=1: <ReTr W>/Nc = {w:.6f} (esperado 1.0)")
    ok &= abs(w - 1.0) < 1e-6

    print("\n" + ("=== TODO OK ===" if ok else "=== HAY FALLAS ==="))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
