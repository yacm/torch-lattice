"""Observables de quarks: condensado quiral y correlador del pión.

Ambos usan el CG sobre las ecuaciones normales (D^dag D), memoria O(V):
nada de inversas densas. El propagador D^{-1} b se obtiene como
(D^dag D)^{-1} (D^dag b), porque (D^dag D)^{-1} D^dag = D^{-1}.
"""
import math

import torch

from . import config, invert


def _solve_D(U, b, m0, max_iter=5000, tol=1e-12):
    """Resuelve D x = b via ecuaciones normales: x = (D^dag D)^{-1} (D^dag b)."""
    rhs = invert._d_dagger(U, b, m0)

    def normal(x):
        return invert._d_dagger(U, invert._d(U, x, m0), m0)

    x, it, resid = invert.cg(normal, rhs, max_iter=max_iter, tol=tol)
    return x, it, resid


def chiral_condensate(U, m0, n_noise=4):
    """Condensado quiral (2 sabores) por estimador estocástico, con batch.

    <psibar psi> = (2/V) Re <eta^dag D^{-1} eta>, con eta ruido gaussiano.
    Los n_noise vectores de ruido se resuelven TODOS a la vez (batch en dim 0),
    en lugar de un bucle de Python.
    Nota: el valor crudo de Wilson lleva renormalización aditiva (constante
    ~1/a), así que el número absoluto no es físico; importa la maquinaria.
    """
    shape = U[0].shape[-6:-2]  # (Nt, Nx, Ny, Nz), robusto a batch
    V = 1
    for s in shape:
        V *= s
    re = torch.randn(n_noise, *shape, 4, 3, dtype=torch.float64)
    im = torch.randn(n_noise, *shape, 4, 3, dtype=torch.float64)
    eta = ((re + 1j * im) / math.sqrt(2)).to(torch.complex128)
    x, _, _ = _solve_D(U, eta, m0)  # resuelve los n_noise sistemas a la vez
    total = (eta.conj() * x).sum().real.item()
    return 2.0 * total / (n_noise * V)


def pion_correlator(U, m0, t0=0):
    """Correlador del pión C(t) con fuente puntual en (t0,0,0,0).

    Fuente de un solo componente (espín=0, color=0). Con gamma5-hermiticidad,
    C(t) = sum_{x,y,z,s,c} |D^{-1}(x; fuente)|^2. Devuelve tensor (Nt,).
    """
    shape = U[0].shape[:4]           # (Nx, Ny, Nz, Nt)
    Nt = shape[config.t_dir]
    src = torch.zeros(*shape, 4, 3, dtype=torch.complex128)
    src[0, 0, 0, t0, 0, 0] = 1.0    # fuente puntual en (x,y,z)=(0,0,0), t=t0
    psi, it, resid = _solve_D(U, src, m0)
    # sumar |psi|^2 sobre x, y, z, espín y color -> queda el eje temporal (dim 3)
    c = (psi.conj() * psi).real.sum(dim=(0, 1, 2, 4, 5))
    return c, it, resid
