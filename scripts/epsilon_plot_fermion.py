"""Gráfica del épsilon test para HMC con fermiones dinámicos: |ΔH| vs eps.

Verifica que el error del integrador (HMC exacto) escale como eps^2.
Guarda runs/epsilon_test_fermion.png y reporta la pendiente ajustada.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from tlattice import fermion_hmc as fh
from tlattice import su3


def main():
    torch.manual_seed(0)
    shape = (2, 2, 2, 4)   # retícula chica para que el CG sea rápido
    beta, m0, T = 6.0, -0.5, 1.0

    U0 = [fh.make_leaf(su3.random_su3(*shape, dtype=torch.complex128))
          for _ in range(4)]
    P0 = fh.sample_momentum(shape)
    phi = fh.sample_phi(U0, m0)

    eps_list = [0.1, 0.05, 0.025, 0.0125, 0.00625]
    dHs = []
    print("eps         n_steps     |dH|")
    for eps in eps_list:
        n = round(T / eps)
        chi0, _, _ = fh.solve_chi(U0, phi, m0)
        H0 = fh.hamiltonian_value(U0, beta, P0, phi, chi0, m0).item()
        Ut, Pt = fh.leapfrog_fermion(U0, P0, beta, phi, m0, n, eps)
        chi1, _, _ = fh.solve_chi(Ut, phi, m0)
        H1 = fh.hamiltonian_value(Ut, beta, Pt, phi, chi1, m0).item()
        dH = abs(H1 - H0)
        dHs.append(dH)
        print(f"{eps:.5f}   {n:3d}      {dH:.6e}")

    loge = np.log(eps_list)
    logd = np.log(dHs)
    slope, intercept = np.polyfit(loge, logd, 1)
    print(f"\npendiente ajustada = {slope:.3f} (ideal: 2.0, HMC exacto de 2º orden)")

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.loglog(eps_list, dHs, "o-", label="|ΔH| medido (fermiones)", color="#2ca02c",
              lw=2, ms=9, zorder=5)
    ref = [dHs[0] * (e / eps_list[0]) ** 2 for e in eps_list]
    ax.loglog(eps_list, ref, "--", label="∝ ε² (referencia)", color="#d62728", lw=2)
    ax.set_xlabel("ε  (paso de integración)", fontsize=12)
    ax.set_ylabel("|ΔH| = |H_final − H_inicial|", fontsize=12)
    ax.set_title(f"Épsilon test — HMC con fermiones dinámicos (pendiente {slope:.2f})",
                 fontsize=13)
    ax.grid(True, which="both", alpha=0.3, ls=":")
    ax.legend(fontsize=11, framealpha=0.9)
    plt.tight_layout()
    out = "runs/epsilon_test_fermion.png"
    plt.savefig(out, dpi=150)
    print(f"\ngráfica guardada en: {out}")


if __name__ == "__main__":
    main()
