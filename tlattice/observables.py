"""Observables: lazo de Polyakov y lazos de Wilson."""
import torch
from . import config


def polyakov_loop(U):
    """Lazo de Polyakov (producto ordenado de enlaces temporales) por sitio espacial.

    Devuelve tensor de forma (Nx, Ny, Nz) con Re Tr(P)/Nc en cada sitio espacial.
    Para gauge trivial vale 1.0 en todos lados.
    """
    Ut = U[config.t_dir]  # enlaces temporales: (Nx, Ny, Nz, Nt, 3, 3)
    P = Ut[..., 0, :, :].clone()
    for t in range(1, Ut.shape[config.t_dir]):
        P = Ut[..., t, :, :] @ P
    return torch.einsum("...ii->...", P).real / config.Nc


def average_polyakov(U):
    """Valor medio (sobre sitios espaciales) de Re Tr(P)/Nc."""
    return polyakov_loop(U).mean().item()


def wilson_loop(U, R, T):
    """Lazo de Wilson rectangular R x T en el plano (0, mu) con mu espacial.

    R: extensión espacial (en la dirección mu), T: extensión temporal.
    Devuelve el promedio de Re Tr(W)/Nc sobre todas las posiciones.
    Devuelve NotImplemented si R o T no son 1 (v1: solo plaqueto rectangular
    se deja para una extensión posterior).
    """
    # Producto de enlaces a lo largo de un rectángulo R x T.
    mu = 1  # dirección espacial fija (x) para el v1
    td = config.t_dir
    shape = U[0].shape[:4]
    Nt = shape[td]
    # --- temporal side (t_dir) ---
    # bajamos T pasos en tiempo acumulando U_td, luego R pasos en mu acumulando U_mu,
    # luego subimos T pasos con U_td^dag, y volvemos R pasos con U_mu^dag.
    W = None
    # empezamos con identidad
    dev = U[0].device
    dt = U[0].dtype
    eye = torch.eye(3, dtype=dt, device=dev).expand(*shape, 3, 3).clone()
    W = eye
    pos = eye  # producto acumulado en el "camino"
    for t in range(T):
        # avanzar en tiempo (t_dir): W = U_td(x + t*td) @ W
        Ut = torch.roll(U[td], -t, dims=td)
        W = Ut @ W
    # ahora moverse R pasos en mu (dir 1)
    for r in range(R):
        Um = torch.roll(U[mu], -r, dims=mu)
        # ojo: U_mu evaluado en el sitio tras T pasos en tiempo
        Um = torch.roll(Um, -T, dims=td)
        W = Um @ W
    # subir de vuelta en tiempo con U_td^dag
    for t in range(T):
        # en el sitio x + T*td + R*mu, retroceder en tiempo
        shift = T - 1 - t
        U0d = torch.roll(U[td].mH, shift, dims=td)
        U0d = torch.roll(U0d, -R, dims=mu)
        W = U0d @ W
    # volver con U_mu^dag
    for r in range(R):
        shift = R - 1 - r
        Umd = torch.roll(U[mu].mH, shift, dims=mu)
        W = Umd @ W
    tr = torch.einsum("...ii->...", W).real / config.Nc
    return tr.mean().item()
