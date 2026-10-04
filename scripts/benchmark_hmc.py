"""Benchmark HMC 2 sabores Wilson (torch-lattice) — tiempo por trayectoria.

Flag --device {cpu,cuda} para correr en CPU o GPU (device-agnostic).

Condiciones fijas para comparar contra Chroma:
  - retícula 4³×8, doble precisión (complex128)
  - beta=6.0, m0=-0.5 (Wilson mass, ≡ Chroma Mass)
  - tau=1.0, n_steps=10 (eps=0.1), integrador leapfrog
  - CG tol=1e-12 (residuo relativo |r|/|b|)

Uso:
  python scripts/benchmark_hmc.py --device cpu --n 5
  TLATTICE_DEVICE=cuda python scripts/benchmark_hmc.py --n 5   # vía env
"""
import argparse
import time

import torch

from tlattice import config, fermion_hmc as fh
from tlattice import su3

BETA = 6.0
M0 = -0.5
N_STEPS = 10
EPS = 0.1          # tau = N_STEPS * EPS = 1.0
SHAPE = (4, 4, 4, 8)
N_WARMUP = 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "cuda"], default=None,
                    help="device: cpu o cuda (default: auto-detect)")
    ap.add_argument("--n", type=int, default=5, help="trayectorias a cronometrar")
    args = ap.parse_args()

    if args.device is not None:
        config.set_device(args.device)
    dev = config.device()
    print(f"device: {dev}  (torch {torch.__version__})")

    gen = torch.Generator(device=dev).manual_seed(42)
    torch.manual_seed(42)
    U = [fh.make_leaf(su3.random_su3(*SHAPE, dtype=torch.complex128, dev=dev))
         for _ in range(4)]

    # warm-up (no se cronometra; calienta caches/compilación de grafos)
    for _ in range(N_WARMUP):
        U, acc, dH, plaq, ncg = fh.hmc_step_fermion(
            U, BETA, M0, N_STEPS, EPS, generator=gen)

    times = []
    accs = 0
    for i in range(args.n):
        t0 = time.perf_counter()
        U, acc, dH, plaq, ncg = fh.hmc_step_fermion(
            U, BETA, M0, N_STEPS, EPS, generator=gen)
        dt = time.perf_counter() - t0
        times.append(dt)
        accs += int(acc)
        print(f"  traj {i}: {dt:6.2f} s  (plaq={plaq:.6f} dH={dH:+.3e} "
              f"CG-iter~{ncg})", flush=True)

    times = sorted(times)
    med = times[len(times) // 2]
    print(f"\n=== torch-lattice 2-flavor Wilson HMC [{dev}] ===")
    print(f"  retícula {SHAPE}, beta={BETA}, m0={M0}, tau=1.0, n_steps={N_STEPS}")
    print(f"  media {sum(times)/len(times):.2f} s/traj | mediana {med:.2f} s/traj")
    print(f"  aceptación {accs}/{args.n}")


if __name__ == "__main__":
    main()
