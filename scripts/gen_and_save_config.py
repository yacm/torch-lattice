"""Genera una configuración con HMC fermiónico, la guarda en LIME (formato Chroma)
y mide el plaqueto espacial para cross-check con read_plaq.cc."""
from pathlib import Path

import torch

from tlattice import fermion_hmc as fh
from tlattice import gauge, lime, su3

SHAPE = (4, 4, 4, 8)  # = dims LIME (L0,L1,L2,L3)
BETA, M0 = 6.0, -0.5
OUT = Path.cwd() / "torch_hmc_config.lime"


def spatial_plaq(U):
    """Mismo plaqueto espacial que read_plaq.cc: planos (1,0),(2,0),(2,1)."""
    planes = [(1, 0), (2, 0), (2, 1)]
    V = 1
    for s in SHAPE:
        V *= s
    return sum(gauge.plaquette(U, mu, nu).sum().item() for mu, nu in planes) / (3 * V)


def main():
    torch.manual_seed(123)
    U = [fh.make_leaf(su3.random_su3(*SHAPE, dtype=torch.complex128)) for _ in range(4)]

    print(f"generando {N_THERM} trayectorias HMC (2 sabores Wilson)...")
    for i in range(N_THERM):
        U, acc, _dH, plaq, _ncg = fh.hmc_step_fermion(
            U, BETA, M0, n_steps=10, eps=0.05
        )
        print(f"  traj {i}: plaqueto={plaq:.4f}  acc={acc}")

    U_d = [u.detach() for u in U]
    p = spatial_plaq(U_d)
    print(f"\nplaqueto espacial (torch) = {p:.15f}")

    suma, sumb = lime.write_lime(U_d, OUT, SHAPE)
    print(f"escrito {OUT}")
    print(f"checksum SciDAC: suma={suma:08x}  sumb={sumb:08x}")

    # round-trip con torch para confirmar
    U2, dims2 = lime.read_lime(OUT)
    p2 = spatial_plaq(U2)
    print(f"releído (torch)          = {p2:.15f}  (dims={dims2})")
    if abs(p - p2) >= 1e-14:
        raise RuntimeError("round-trip no idéntico")
    print("round-trip torch idéntico ✓")


N_THERM = 10

if __name__ == "__main__":
    main()
