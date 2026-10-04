"""Validación del HMC (M5): fuerza, conservación de energía y termalización.

Nota: la verificación autoritativa de la fuerza es el epsilon test
(tests/test_epsilon.py). La comparación con derivada numérica aquí es solo
orientativa porque la diferencia finita sufre cancelación catastrófica en
float32.
"""
import torch

from tlattice import config, gauge, hmc


def main():
    torch.manual_seed(0)
    dev = config.device()
    print(f"device: {dev}")
    ok = True

    shape = (4, 4, 4, 8)
    beta = 6.0

    # ---- 1. Fuerza vs derivada numérica (orientativa) ----
    print("\n== fuerza molecular vs derivada numérica (orientativa) ==")
    U = [hmc.make_leaf(u) for u in gauge.random_gauge(shape)]
    P = hmc.sample_momentum(shape)
    eps = 1e-4
    U_plus = [(torch.matrix_exp(eps * p) @ u).detach() for p, u in zip(P, U)]
    U_minus = [(torch.matrix_exp(-eps * p) @ u).detach() for p, u in zip(P, U)]
    numerical = (gauge.wilson_action(U_plus, beta)
                 - gauge.wilson_action(U_minus, beta)).item() / (2 * eps)
    F = hmc.force(U, beta)
    theoretical = sum(torch.einsum("...ij,...ji->...", f, p).real.sum().item()
                      for f, p in zip(F, P))
    ratio = theoretical / numerical
    print(f"  cociente teórico/numérico = {ratio:.4f} (1.0 ideal; ~5% de ruido float32 ok)")
    ok &= abs(ratio - 1.0) < 0.1

    # ---- 2. Conservación de energía (leapfrog) ----
    print("\n== conservación de energía (leapfrog, eps=0.01) ==")
    U = [hmc.make_leaf(u) for u in gauge.unit_gauge(shape)]
    P = hmc.sample_momentum(shape)
    H0 = hmc.hamiltonian(U, beta, P).item()
    U_t, P_t = hmc.leapfrog(U, P, beta, n_steps=20, eps=0.01)
    H1 = hmc.hamiltonian(U_t, beta, P_t).item()
    dH = H1 - H0
    rel = abs(dH) / abs(H0)
    print(f"  H0 = {H0:.4f}, deltaH = {dH:.4e}, relativo = {rel:.2e}")
    ok &= rel < 1e-2

    # ---- 3. HMC: arranque frío debe termalizar (plaqueto baja de 1.0) ----
    print("\n== HMC: arranque frío, eps=0.008, n_steps=50 ==")
    U = [hmc.make_leaf(u) for u in gauge.unit_gauge(shape)]
    n_acc = 0
    n_traj = 20
    plaqs = []
    for i in range(n_traj):
        U, acc, dH, plaq = hmc.hmc_step(U, beta, n_steps=50, eps=0.008)
        n_acc += acc
        plaqs.append(plaq)
    rate = n_acc / n_traj
    print(f"  tasa de aceptación: {rate:.2f} (esperado > 0.6)")
    print(f"  plaqueto inicial 1.0000 -> final {plaqs[-1]:.4f}")
    print(f"  evolución (cada 4): {['%.3f' % p for p in plaqs[::4]]}")
    ok &= rate > 0.6
    ok &= plaqs[-1] < 0.999

    print("\n" + ("=== TODO OK ===" if ok else "=== HAY FALLAS ==="))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
