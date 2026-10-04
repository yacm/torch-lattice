"""Utilidades de grupo SU(3): generadores de Gell-Mann, exp del álgebra, proyección."""
import math

import torch

from . import config


def gellmann(dtype=torch.complex64, dev=None):
    """Las 8 matrices de Gell-Mann (hermíticas, traza cero), 3x3."""
    dev = dev if dev is not None else config.device()
    lam = []
    lam.append([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    lam.append([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]])
    lam.append([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
    lam.append([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
    lam.append([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]])
    lam.append([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
    lam.append([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]])
    lam.append([[1, 0, 0], [0, 1, 0], [0, 0, -2]])
    lam[-1] = [[x / math.sqrt(3) for x in row] for row in lam[-1]]
    return [torch.tensor(l, dtype=dtype, device=dev) for l in lam]


def generators(dtype=torch.complex64, dev=None):
    """Generadores T_a = lambda_a / 2 (Tr(T_a T_b) = delta_ab / 2)."""
    return [l / 2.0 for l in gellmann(dtype, dev)]


def random_su3(*shape, dtype=torch.complex64, dev=None, generator=None):
    """Matriz(es) SU(3) aleatorias = exp(i sum_a c_a T_a), c_a ~ N(0,1).

    shape: dimensiones de lote (por ejemplo (Nt, Nx, Ny, Nz)).
    Devuelve tensor de forma (*shape, 3, 3) complejo, unitario y de det 1.
    """
    dev = dev if dev is not None else config.device()
    gen = generator if generator is not None else torch.Generator(device=dev)
    Ts = generators(dtype, dev)
    A = torch.zeros(*shape, 3, 3, dtype=dtype, device=dev)
    for T in Ts:
        c = torch.randn(*shape, dtype=torch.float32, device=dev, generator=gen)
        A = A + (1j * c)[..., None, None] * T
    return torch.matrix_exp(A)


def project_su3(U, tol=1e-10, max_iter=20):
    """Proyecta matrices 3x3 (generales) al grupo SU(3).

    Polar (unitarizar) + fijar determinante a 1. La convergencia respeta el piso
    del dtype: exigir 1e-10 a complex64 repetía ``eigh`` sobre un espectro casi
    degenerado hasta 20 veces y podía hacer fallar LAPACK de forma esporádica.
    """
    real_dtype = U.real.dtype
    effective_tol = max(float(tol), 10.0 * torch.finfo(real_dtype).eps)
    for _ in range(max_iter):
        # unitarizar: U -> U (U^dag U)^(-1/2)
        M = U.mH @ U  # hermítica positiva
        w, V = torch.linalg.eigh(M)
        M_inv_sqrt = (V * (1.0 / torch.sqrt(torch.clamp(w, min=1e-30)))[..., None, :]) @ V.mH
        U = U @ M_inv_sqrt
        # fijar det = 1 (quitar la fase)
        det = torch.det(U)
        phase = torch.angle(det)
        U = U * torch.exp(-1j * phase / 3.0)[..., None, None]
        # criterio de convergencia: U U^dag - I
        err = (U @ U.mH - torch.eye(3, dtype=U.dtype, device=U.device)).abs().max()
        if err <= effective_tol:
            break
    return U


def is_su3(U, tol=1e-5):
    """True si U es unitaria (U U^dag = I) y de determinante 1, a tolerancia."""
    I = torch.eye(3, dtype=U.dtype, device=U.device)
    unit = (U @ U.mH - I).abs().max().item() < tol
    det = torch.det(U)
    det_ok = (det - 1.0).abs().max().item() < tol
    return unit and det_ok
