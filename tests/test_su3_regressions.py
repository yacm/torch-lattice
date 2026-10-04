"""Regresiones numéricas para la proyección SU(3)."""
from unittest import mock

import torch

from tlattice import su3


def test_complex64_projection_stops_at_dtype_precision_floor():
    """No repetir eigh sobre matrices casi identidad con espectro degenerado."""
    generator = torch.Generator().manual_seed(20261004)
    field = torch.randn(4, 4, 4, 8, 3, 3, dtype=torch.complex64, generator=generator)
    original = torch.linalg.eigh
    calls = 0

    def counted_eigh(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    with mock.patch.object(torch.linalg, "eigh", counted_eigh):
        projected = su3.project_su3(field)

    assert calls <= 3, f"eigh se repitió {calls} veces por debajo del piso float32"
    assert su3.is_su3(projected)


def main():
    test_complex64_projection_stops_at_dtype_precision_floor()
    print("=== REGRESIÓN SU3 OK ===")


if __name__ == "__main__":
    main()
