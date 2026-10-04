"""Medición en ensamble: condensado quiral y masa del pión sobre configs HMC.

Pipeline completo: termaliza con HMC de 2 sabores Wilson, produce configuraciones,
y mide el plaqueto, el condensado quiral (estimador estocástico) y el correlador
del pión (fuente puntual), extrayendo la masa efectiva m_eff(t) del plateau.
"""
import math
import time

import torch

from tlattice import fermion_hmc as fh
from tlattice import quark_observables as qo
from tlattice import su3
from tlattice import config

SHAPE = (4, 4, 4, 8)   # (Nx, Ny, Nz, Nt): tiempo = 8 cortes (eje 3 = t_dir)
BETA = 6.0
M0 = -0.5
N_THERM = 15
N_PROD = 20
MEAS_EVERY = 4
N_STEPS, EPS = 10, 0.05


def effective_mass(C, t):
    """m_eff(t) = arccosh((C(t-1) + C(t+1)) / (2 C(t))) para correlador periódico."""
    num = C[t - 1] + C[t + 1]
    den = 2.0 * C[t]
    if den <= 0 or num / den < 1.0:
        return float("nan")
    return math.acosh(num / den)


def main():
    torch.manual_seed(42)
    U = [fh.make_leaf(su3.random_su3(*SHAPE, dtype=torch.complex128))
         for _ in range(4)]

    t0 = time.time()
    print(f"termalizando {N_THERM} trayectorias...")
    n_acc = 0
    for i in range(N_THERM):
        U, acc, dH, plaq, ncg = fh.hmc_step_fermion(U, BETA, M0, N_STEPS, EPS)
        n_acc += acc

    print(f"producción {N_PROD} trayectorias + medición cada {MEAS_EVERY}...")
    cond_sum, n_cond = 0.0, 0
    pion_sum, n_pion = None, 0
    plaqs = []
    for i in range(N_PROD):
        U, acc, dH, plaq, ncg = fh.hmc_step_fermion(U, BETA, M0, N_STEPS, EPS)
        n_acc += acc
        plaqs.append(plaq)
        if i % MEAS_EVERY == 0:
            c = qo.chiral_condensate(U, M0, n_noise=4)
            cond_sum += c
            n_cond += 1
            ct, _, _ = qo.pion_correlator(U, M0)
            pion_sum = ct.clone() if pion_sum is None else pion_sum + ct
            n_pion += 1

    plaq_avg = sum(plaqs) / len(plaqs)
    cond_avg = cond_sum / n_cond
    pion_avg = (pion_sum / n_pion).tolist()
    T = SHAPE[config.t_dir]
    meffs = [effective_mass(pion_avg, t) for t in range(1, T - 1)]

    print("\n=== RESULTADOS DEL ENSAMBLE ===")
    print(f"tasa de aceptación : {n_acc/(N_THERM+N_PROD):.2f}")
    print(f"plaqueto <P>       : {plaq_avg:.4f}")
    print(f"condensado <ψ̄ψ>    : {cond_avg:.4f}")
    print(f"C(t) promedio      : {[f'{x:.5f}' for x in pion_avg]}")
    print(f"m_eff(t), t=1..{T-2}: {[f'{m:.3f}' for m in meffs]}")
    plateau = [m for m in meffs[2:] if not math.isnan(m)]  # t=3.. (plateau)
    if plateau:
        m_pi = sum(plateau) / len(plateau)
        print(f"masa del pión m_π (plateau) ≈ {m_pi:.3f}")
    print(f"tiempo total        : {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
