"""Compara la fuerza analítica (sin autograd) contra autograd: correctitud + tiempo.

Compara:
  1. Gradiente gauge:  gauge_force_analytic  vs  autograd(dS_gauge/dU)
  2. Gradiente fermiónico: fermion_force_analytic  vs  autograd(d[-||Dchi||²]/dU)
  3. Fuerza total proyectada su(3): force_total_analytic vs force_total
  4. Tiempo por llamada (misma condición: re-resuelve chi con CG tol 1e-12)
"""
import time

import torch

from tlattice import config, dirac, gauge, fermion_hmc as fh, force_analytic as fa
from tlattice import su3

BETA = 6.0
M0 = -0.5
SHAPE = (4, 4, 4, 8)
SEED = 42


def report(name, a, b):
    """Compara dos listas de tensores componente a componente."""
    max_abs = 0.0
    max_rel = 0.0
    n_diff = 0
    total = 0
    for x, y in zip(a, b):
        d = (x - y).abs()
        max_abs = max(max_abs, d.max().item())
        # error relativo vs magnitud de y
        denom = y.abs()
        rel = (d / (denom + 1e-30)).max().item()
        max_rel = max(max_rel, rel)
        n_diff += int((d > 1e-13).sum().item())
        total += d.numel()
    print(f"  {name:28s} max|Δ|={max_abs:.3e}  max rel={max_rel:.3e}  "
          f"diffs>{1e-13}: {n_diff}/{total}")
    return max_abs


def main():
    dev = config.device()
    gen = torch.Generator(device=dev).manual_seed(SEED)
    torch.manual_seed(SEED)

    U = [fh.make_leaf(su3.random_su3(*SHAPE, dtype=torch.complex128, dev=dev))
         for _ in range(4)]
    phi = fh.sample_phi(U, M0, generator=gen).detach()

    print(f"=== Fuerza analítica vs autograd  (retícula {SHAPE}, beta={BETA}, m0={M0}) ===\n")

    chi, n_cg, resid = fh.solve_chi(U, phi, M0)
    print(f"  chi resuelto: {n_cg} iter CG, resid={resid:.2e}\n")

    # ---- 1. gradiente gauge ----
    S_g = gauge.wilson_action(U, BETA)
    g_gauge_ad = torch.autograd.grad(S_g, U, create_graph=False)
    g_gauge_an = fa.gauge_force_analytic(U, BETA)
    report("g_gauge (autograd vs analitico)", g_gauge_ad, g_gauge_an)

    # ---- 2. gradiente fermionico ----
    Dchi = dirac.wilson_dslash(U, chi, M0)
    S_pf = -(Dchi.conj() * Dchi).sum().real
    g_pf_ad = torch.autograd.grad(S_pf, U, create_graph=False)
    g_pf_an = fa.fermion_force_analytic(U, chi, M0)
    report("g_fermion (autograd vs analitico)", g_pf_ad, g_pf_an)

    # ---- 3. fuerza total proyectada su(3) ----
    F_ad = fh.force_total(U, BETA, phi, M0)
    F_an = fa.force_total_analytic(U, BETA, phi, M0)
    report("F_total su(3) (autograd vs analitico)", F_ad, F_an)

    # ---- 4. rendimiento ----
    print("\n=== Rendimiento (fuerza total, re-resuelve chi cada vez) ===")
    N = 7
    for name, fn in [("autograd", fh.force_total),
                     ("analitico", fa.force_total_analytic)]:
        # warm-up
        for _ in range(2):
            fn(U, BETA, phi, M0)
        ts = []
        for _ in range(N):
            t0 = time.perf_counter()
            fn(U, BETA, phi, M0)
            ts.append(time.perf_counter() - t0)
        ts.sort()
        med = ts[len(ts) // 2]
        print(f"  {name:10s} mediana {med*1000:7.2f} ms/llamada   (media {sum(ts)/N*1000:7.2f})")


if __name__ == "__main__":
    main()
