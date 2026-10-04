"""Validación del batch dimension: Dslash, CG y observables estocásticos.

Verifica que la versión con batch (dims líderes) coincide exactamente con
la versión sin batch, tanto con U batched como con una sola U expandida.
"""
import math

import torch

from tlattice import dirac, invert, quark_observables as qo
from tlattice import su3

B = 4
SHAPE = (2, 2, 2, 4)
M0 = -0.5


def main():
    torch.manual_seed(0)
    ok = True

    # configuraciones y campos por separado (sin batch)
    Us = [[su3.random_su3(*SHAPE, dtype=torch.complex128) for _ in range(4)]
          for _ in range(B)]
    psis = [torch.randn(*SHAPE, 4, 3, dtype=torch.complex128) for _ in range(B)]

    # --- 1. Dslash con U y psi batched == unbatched ---
    U_b = [torch.stack([Us[b][mu] for b in range(B)]) for mu in range(4)]
    psi_b = torch.stack(psis)
    out_b = dirac.wilson_dslash(U_b, psi_b, M0)
    err = max((out_b[b] - dirac.wilson_dslash(Us[b], psis[b], M0)).abs().max().item()
              for b in range(B))
    print(f"Dslash (U+psi batched)   err={err:.2e}")
    ok &= err < 1e-12

    # --- 2. Dslash con U única + psi batched (caso vectores de ruido) ---
    U = Us[0]
    out_b2 = dirac.wilson_dslash(U, psi_b, M0)  # U sin batch, psi con batch
    err2 = max((out_b2[b] - dirac.wilson_dslash(U, psis[b], M0)).abs().max().item()
               for b in range(B))
    print(f"Dslash (U-única + psi-batch) err={err2:.2e}")
    ok &= err2 < 1e-12

    # --- 3. CG batched == CG unbatched (misma U, mismas RHS) ---
    def normal(x):
        return invert._d_dagger(U, invert._d(U, x, M0), M0)

    rhs_b = invert._d_dagger(U, psi_b, M0)
    x_b, _, _ = invert.cg(normal, rhs_b, max_iter=2000, tol=1e-11)
    err3 = 0.0
    for b in range(B):
        rhs1 = invert._d_dagger(U, psis[b], M0)
        x1, _, _ = invert.cg(normal, rhs1, max_iter=2000, tol=1e-11)
        err3 = max(err3, (x_b[b] - x1).abs().max().item())
    print(f"CG batched vs unbatched   err={err3:.2e}")
    ok &= err3 < 1e-10

    # --- 4. condensado: resolver en batch == loop (mismo ruido) ---
    V = 1
    for s in SHAPE:
        V *= s
    n = 6
    re = torch.randn(n, *SHAPE, 4, 3, dtype=torch.float64)
    im = torch.randn(n, *SHAPE, 4, 3, dtype=torch.float64)
    eta_b = ((re + 1j * im) / math.sqrt(2)).to(torch.complex128)
    x_b, _, _ = qo._solve_D(U, eta_b, M0)  # batch
    c_batch = 2.0 * (eta_b.conj() * x_b).sum().real.item() / (n * V)
    tot = 0.0
    for b in range(n):
        x1, _, _ = qo._solve_D(U, eta_b[b], M0)  # uno a la vez
        tot += (eta_b[b].conj() * x1).sum().real.item()
    c_loop = 2.0 * tot / (n * V)
    print(f"condensado batch={c_batch:.8f}  loop={c_loop:.8f}  dif={abs(c_batch-c_loop):.2e}")
    ok &= abs(c_batch - c_loop) < 1e-8

    # --- 5. chiral_condensate() público (batch interno) da valor finito ---
    c_pub = qo.chiral_condensate(U, M0, n_noise=8)
    print(f"chiral_condensate() = {c_pub:.6f}")
    ok &= 0 < c_pub < 50

    print("\n" + ("=== TODO OK ===" if ok else "=== HAY FALLAS ==="))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
