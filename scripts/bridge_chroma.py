"""Carga una config de Chroma (dump binario) y compara plaquette + CG con Chroma.

Puente de configuraciones: dump_gauge_full escribe [mu][x][y][z][t][i][j]
float64 (re,im interleaved). La config es dummy_run_cfg_7.lime (termalizada a
beta=6.0). Validar el puente: la plaquette JAX debe dar el w_plaq de Chroma.
"""
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from tlattice_jax import evenodd, gauge, hmc, hmc_eo

jax.config.update("jax_enable_x64", True)

M0 = -0.5
SHAPE = (4, 4, 4, 8)
ROOT = Path(__file__).resolve().parents[1]
FIELD = Path(os.environ.get("TORCH_LATTICE_FIELD", ROOT / "benchmarks/canonical/cfg_7.bin"))


def load_chroma_cfg(path):
    raw = np.fromfile(path, dtype=np.float64)
    cv = raw[0::2] + 1j * raw[1::2]
    cv = cv.reshape(4, 4, 4, 4, 8, 3, 3)  # [mu][x][y][z][t][i][j]
    return [jnp.asarray(cv[mu].copy()) for mu in range(4)]


def main():
    U = load_chroma_cfg(FIELD)

    # --- validación del puente: plaquette ---
    w_plaq = float(gauge.average_plaquette(U))
    print(f"plaquette JAX  w_plaq = {w_plaq:.15f}")
    print("plaquette Chroma w_plaq = 0.725713584607721")

    # --- CG iteraciones sobre el MISMO campo ---
    key = jax.random.PRNGKey(0)
    phi, _ = hmc.sample_phi(key, U, M0)
    _chi, it_unp, resid = hmc.solve_chi(U, phi, M0)
    print(f"\nCG sin precond (mismo campo): {int(it_unp)} iter, resid={float(resid):.3e}")

    # iteración representativa de Chroma (último n_count)
    print("Chroma n_count (última config): ~406")

    geo = evenodd.build_geometry(SHAPE)
    phi_e, _ = evenodd.split_eo(phi, *evenodd.checkerboard(SHAPE))
    _, it_eo, _ = hmc_eo.solve_chi_eo(U, phi_e, geo, SHAPE, M0)
    print(f"CG par-impar (mismo campo):    {int(it_eo)} iter")


if __name__ == "__main__":
    main()
