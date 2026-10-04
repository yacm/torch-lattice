"""Genera la gráfica del épsilon test: |ΔH| vs eps en escala log-log.

Verifica que el error del integrador leapfrog escale como eps^2 (2º orden).
Guarda runs/epsilon_test.png y también reporta la pendiente ajustada.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from tlattice import gauge, hmc


def main():
    torch.manual_seed(0)
    shape = (4, 4, 4, 8)
    beta = 6.0
    T = 1.0  # longitud de trayectoria fija

    U0 = [hmc.make_leaf(u) for u in gauge.random_gauge(shape)]
    P0 = hmc.sample_momentum(shape)

    eps_list = [0.1, 0.05, 0.025, 0.0125, 0.00625]
    dHs = []
    print("eps         n_steps     |dH|")
    for eps in eps_list:
        n = round(T / eps)
        H0 = hmc.hamiltonian(U0, beta, P0).item()
        Ut, Pt = hmc.leapfrog(U0, P0, beta, n, eps)
        H1 = hmc.hamiltonian(Ut, beta, Pt).item()
        dH = abs(H1 - H0)
        dHs.append(dH)
        print(f"{eps:.5f}   {n:3d}      {dH:.6e}")

    # ajuste lineal en log-log -> pendiente (debe ser ~2)
    loge = np.log(eps_list)
    logd = np.log(dHs)
    slope, intercept = np.polyfit(loge, logd, 1)
    print(f"\npendiente ajustada = {slope:.3f} (ideal: 2.0, integrador de 2º orden)")

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.loglog(eps_list, dHs, "o-", label="|ΔH| medido", color="#1f77b4",
              lw=2, ms=9, zorder=5)
    ref = [dHs[0] * (e / eps_list[0]) ** 2 for e in eps_list]
    ax.loglog(eps_list, ref, "--", label="∝ ε² (referencia)", color="#d62728", lw=2)
    ax.set_xlabel("ε  (paso de integración)", fontsize=12)
    ax.set_ylabel("|ΔH| = |H_final − H_inicial|", fontsize=12)
    ax.set_title(f"Épsilon test — integrador leapfrog (pendiente {slope:.2f})",
                 fontsize=13)
    ax.grid(True, which="both", alpha=0.3, ls=":")
    ax.legend(fontsize=11, framealpha=0.9)
    plt.tight_layout()
    out = "runs/epsilon_test.png"
    plt.savefig(out, dpi=150)
    print(f"\ngráfica guardada en: {out}")


if __name__ == "__main__":
    main()
