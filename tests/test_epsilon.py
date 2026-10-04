"""Épsilon test: verifica que el integrador (leapfrog) es de 2º orden.

Fija la longitud de trayectoria T = n_steps * eps y mide dH = H_final - H_inicial
para varios eps. Para un integrador simpléctico de 2º orden (leapfrog), el error
debe escalar como dH ~ C * eps^2, de modo que dH / eps^2 queda ~constante al
reducir eps. Si la fuerza o el integrador estuvieran mal, dH/eps^2 divergiría.
"""
import torch

from tlattice import gauge, hmc


def main():
    torch.manual_seed(0)
    shape = (4, 4, 4, 8)
    beta = 6.0
    T = 1.0  # longitud de trayectoria fija

    # misma configuración y mismo momento para todos los eps (comparación justa)
    U0 = [hmc.make_leaf(u) for u in gauge.random_gauge(shape)]
    P0 = hmc.sample_momentum(shape)

    print("eps         n_steps     deltaH          deltaH/eps^2")
    print("-" * 55)
    rows = []
    for eps in [0.1, 0.05, 0.025, 0.0125]:
        n = round(T / eps)
        H0 = hmc.hamiltonian(U0, beta, P0).item()
        Ut, Pt = hmc.leapfrog(U0, P0, beta, n, eps)
        H1 = hmc.hamiltonian(Ut, beta, Pt).item()
        dH = H1 - H0
        scaled = dH / eps**2
        rows.append((eps, n, dH, scaled))
        print(f"{eps:.4f}     {n:3d}        {dH:+14.6e}   {scaled:+14.6e}")

    # criterio: dH/eps^2 debe quedar del mismo orden de magnitud (factor < ~5)
    scaled_vals = [abs(r[3]) for r in rows]
    lo, hi = min(scaled_vals), max(scaled_vals)
    ok = (lo > 0) and (hi / lo < 5.0)
    print("-" * 55)
    print(f"cociente max/min de |dH/eps^2| = {hi/lo:.2f} (esperado < ~5)")
    print("OK: dH ~ O(eps^2) — integrador y fuerza correctos" if ok
          else "FALLA: dH no escala como eps^2")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
