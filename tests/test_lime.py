"""Cross-check bit-compatible con Chroma (M7): lector/escritor LIME.

1. Lee la config real de purgaug y compara el plaqueto espacial con el valor
   que reportó read_plaq.cc (0.660514080740162) — acuerdo bit-compatible.
2. Round-trip: escribe esa config y la relee — idéntico.
3. Checksum SciDAC del campo original == suma/sumb del archivo de Chroma.
"""
import tempfile
from pathlib import Path

from tlattice import gauge, lime

REF_SPATIAL_PLAQ = 0.660514080740162
REF_SUMA = 0x3F3C1349
REF_SUMB = 0x6D4D2B2C
ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT.parent / "lattice-qcd/runs/purgaug1/dummy_run.lime1"
DIMS = (4, 4, 4, 8)


def spatial_plaq(U):
    planes = [(1, 0), (2, 0), (2, 1)]
    return sum(gauge.plaquette(U, mu, nu).sum().item() for mu, nu in planes) / (3 * 512)


def main():
    print("== 1. lector: config de purgaug vs read_plaq.cc ==")
    U, dims = lime.read_lime(CFG)
    assert dims == list(DIMS), f"dims mal leídos: {dims}"
    p = spatial_plaq(U)
    print(f"  s_plaq torch  = {p:.15f}")
    print(f"  s_plaq chroma = {REF_SPATIAL_PLAQ:.15f}")
    print(f"  diferencia    = {abs(p - REF_SPATIAL_PLAQ):.3e}")
    assert abs(p - REF_SPATIAL_PLAQ) < 1e-14, "el lector NO es bit-compatible"

    print("\n== 2. round-trip: escribir y releer ==")
    with tempfile.TemporaryDirectory() as directory:
        roundtrip = Path(directory) / "torch_lattice_test.lime"
        lime.write_lime(U, roundtrip, DIMS)
        U2, dims2 = lime.read_lime(roundtrip)
    p2 = spatial_plaq(U2)
    assert dims2 == list(DIMS)
    assert abs(p2 - REF_SPATIAL_PLAQ) < 1e-14
    print(f"  s_plaq tras round-trip = {p2:.15f}  (idéntico ✓)")

    print("\n== 3. checksum SciDAC byte-exacto ==")
    binary = lime._field_to_binary(U)
    suma, sumb = lime.scidac_checksum(binary)
    print(f"  suma = {suma:08x} (Chroma {REF_SUMA:08x})")
    print(f"  sumb = {sumb:08x} (Chroma {REF_SUMB:08x})")
    assert suma == REF_SUMA and sumb == REF_SUMB, "checksum no coincide"
    # round-trip binario byte-exacto
    orig = CFG.read_bytes()
    # localizar el payload binario original
    recs = {t: p for t, p in lime._iter_records(orig)}
    assert binary == recs[lime.REC_BINARY], "round-trip binario no es byte-exacto"
    print("  round-trip binario byte-exacto ✓")

    print("\n=== TODO OK: bit-compatible con Chroma ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
