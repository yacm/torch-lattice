"""Constantes globales: grupo de color, gamma matrices euclídeas, device."""
import os

import torch

Nc = 3          # colores SU(3)
Nd = 4          # dimensiones (x, y, z, t)  —  t es el eje 3 (t_dir), como Chroma
t_dir = 3       # dirección temporal (convención de Chroma)

_device_override = None


def set_device(dev):
    """Fija el device explícitamente ('cpu' o 'cuda'), anulando el auto-detect."""
    global _device_override
    _device_override = dev


def device() -> torch.device:
    """Device-agnostic con flag, en orden de prioridad:

    1. `config.set_device(dev)` (override programático).
    2. variable de entorno `TLATTICE_DEVICE` (p.ej. `TLATTICE_DEVICE=cuda`).
    3. auto-detect: CUDA si hay GPU, si no CPU.
    """
    if _device_override is not None:
        return torch.device(_device_override)
    env = os.environ.get("TLATTICE_DEVICE")
    if env:
        return torch.device(env)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Matrices gamma EUCLÍDEAS (base DeGrand-Rossi, idénticas a Chroma/QDP++) ---
# Todas hermíticas y con gamma^2 = I. Orden: [x, y, z, t]  (mu = 3 es tiempo).
# Extraídas de los proyectores (1 +/- gamma_mu) en qdpxx sse_spin_proj.h.
_G0 = [[0, 0, 0, 1j], [0, 0, 1j, 0], [0, -1j, 0, 0], [-1j, 0, 0, 0]]   # x
_G1 = [[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]]      # y
_G2 = [[0, 0, 1j, 0], [0, 0, 0, -1j], [-1j, 0, 0, 0], [0, 1j, 0, 0]]  # z
_G3 = [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]]        # t


def gammas(dtype=torch.complex64, dev=None) -> list[torch.Tensor]:
    """Las 4 matrices gamma euclídeas como tensores 4x4 complejos."""
    dev = dev if dev is not None else device()
    raw = [_G0, _G1, _G2, _G3]
    return [torch.tensor(g, dtype=dtype, device=dev) for g in raw]


def gamma5(dtype=torch.complex64, dev=None):
    """gamma5 = gamma0 gamma1 gamma2 gamma3."""
    gs = gammas(dtype, dev)
    g5 = gs[0] @ gs[1] @ gs[2] @ gs[3]
    return g5


def eye_spin(dtype=torch.complex64, dev=None):
    dev = dev if dev is not None else device()
    return torch.eye(4, dtype=dtype, device=dev)
