"""Valida el port JAX/torch contra Chroma: carga el config LIME (dump binario)
y compara la plaquette plano por plano con los valores que reportó Chroma.

La config final (dummy_run_cfg_7.lime) corresponde a w_plaq=0.725713584607721
(los updates 3-7 fueron rechazados, así que la config no cambió desde update 2).
"""
import os
from pathlib import Path

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIELD = ROOT.parent / "lattice-qcd/runs/bench2flav/gauge_field.bin"
FIELD = Path(os.environ.get("TORCH_LATTICE_CHROMA_FIELD", DEFAULT_FIELD))

# Valores de Chroma (out.xml, update 7 = update 2 por rechazos 3-7)
CHROMA = {
    "w_plaq": 0.725713584607721,
    "s_plaq": 0.724155052493933,
    "t_plaq": 0.727272116721509,
    "plane_01": 0.721552198167977,
    "plane_02": 0.728403909283719,
    "plane_12": 0.722509050030103,
    "plane_03": 0.72017099957098,
    "plane_13": 0.732534404341137,
    "plane_23": 0.729110946252411,
}


def main():
    raw = np.fromfile(FIELD, dtype=np.float64)
    expected_size = 4 * 4 * 4 * 4 * 8 * 3 * 3 * 2
    if raw.size != expected_size:
        raise ValueError(f"tamaño inesperado: {raw.size}; esperado: {expected_size}")
    cv = raw[0::2] + 1j * raw[1::2]                    # re,im interleaved
    field = cv.reshape(4, 4, 4, 4, 8, 3, 3)            # [mu][x][y][z][t][i][j]
    U = [jnp.asarray(field[mu]) for mu in range(4)]    # cada uno (4,4,4,8,3,3)

    def mH(x):
        return jnp.conj(x).swapaxes(-1, -2)

    def plaq(mu, nu):
        Unu_f = jnp.roll(U[nu], -1, axis=mu)           # U_nu(x+mu)
        Umu_f = jnp.roll(U[mu], -1, axis=nu)           # U_mu(x+nu)
        pl = U[mu] @ Unu_f @ mH(Umu_f) @ mH(U[nu])
        return float((jnp.einsum("...ii->...", pl).real / 3.0).mean())

    mine = {
        "plane_01": plaq(0, 1), "plane_02": plaq(0, 2), "plane_12": plaq(1, 2),
        "plane_03": plaq(0, 3), "plane_13": plaq(1, 3), "plane_23": plaq(2, 3),
    }
    mine["s_plaq"] = (mine["plane_01"] + mine["plane_02"] + mine["plane_12"]) / 3
    mine["t_plaq"] = (mine["plane_03"] + mine["plane_13"] + mine["plane_23"]) / 3
    mine["w_plaq"] = (mine["s_plaq"] + mine["t_plaq"]) / 2

    print("      clave      |      JAX        |     Chroma      |     Δ")
    print("-----------------|-----------------|-----------------|------------")
    ok = True
    for k, reference in CHROMA.items():
        d = abs(mine[k] - reference)
        flag = "" if d < 1e-9 else "  <<< ¡MISMATCH!"
        print(f"{k:>16} | {mine[k]:.15f} | {reference:.15f} | {d:.2e}{flag}")
        ok &= d < 1e-9
    print("\n" + ("=== PLAQUETTE COINCIDE CON CHROMA ===" if ok
                  else "=== HAY DISCREPANCIA: revisar convención ==="))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
