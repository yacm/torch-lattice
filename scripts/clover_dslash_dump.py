"""Cross-check Dslash clover vs Chroma: aplica clover_dslash (BC periódica) a un
delta y vuelca chi en el mismo formato que progs/clover_dslash_test.cc."""
import sys
from pathlib import Path

import torch

from tlattice import clover, lime, su3

Mass = float(sys.argv[1])
c_sw = float(sys.argv[2])
boundary = sys.argv[3] if len(sys.argv) > 3 else "periodic"

shape = (4, 4, 4, 8)  # (Nx, Ny, Nz, Nt), tiempo en eje 3 (t_dir)
torch.manual_seed(42)
U = [su3.random_su3(*shape, dtype=torch.complex128) for _ in range(4)]

# Escribe la config (Chroma la lee con el mismo seed)
lime.write_lime(U, Path.cwd() / "torch_clover_config.lime", shape)

# Fuente delta en (x,y,z,t)=(0,0,0,0), spin 0, color 0
psi = torch.zeros(*shape, 4, 3, dtype=torch.complex128)
psi[0, 0, 0, 0, 0, 0] = 1.0

# Operador clover con la BC elegida (periodic o antiperiodic)
chi = clover.clover_dslash(U, psi, Mass, c_sw, boundary=boundary)

Nx, Ny, Nz, Nt = shape
for t in range(Nt):
    for z in range(Nz):
        for y in range(Ny):
            for x in range(Nx):
                vals = []
                for s in range(4):
                    for c in range(3):
                        v = chi[x, y, z, t, s, c]
                        vals.append(f"{v.real.item():.15e} {v.imag.item():.15e}")
                print(f"{x} {y} {z} {t} " + " ".join(vals))
