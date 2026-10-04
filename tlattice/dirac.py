"""Operador de Dirac de Wilson (Dslash) con condiciones de contorno anti-periódicas en tiempo."""
import torch
from . import config
from . import gauge as _gauge


def _expand_to(u, shape):
    """Expande u (posiblemente sin batch) a la forma objetivo, como vista sin copia."""
    shape = tuple(shape)
    if u.shape == shape:
        return u
    while u.dim() < len(shape):
        u = u.unsqueeze(0)
    return u.expand(*shape)


def _flip_boundary(u, axis, time_slice):
    """Multiplica por -1 el corte de frontera (time_slice = -1 o 0) del eje dado."""
    u = u.clone()
    idx = [slice(None)] * u.dim()
    idx[axis] = time_slice
    u[tuple(idx)] *= -1
    return u


def wilson_dslash(U, psi, m0, boundary="antiperiodic"):
    """Aplica el operador de Dirac de Wilson D_w a psi.

    U: lista de 4 tensores de gauge, cada uno (..., Nx, Ny, Nz, Nt, 3, 3) complex.
    psi: tensor fermiónico (..., Nx, Ny, Nz, Nt, 4, 3) complex (espín, color).
    m0: masa desnuda de Wilson.
    boundary: "antiperiodic" (default, APBC en t_dir) o "periodic".

    Soporta una dimensión de batch en las dims líderes (configuraciones o
    vectores de ruido). El offset de batch se infiere de psi.dim() - 6; U puede
    llevar el mismo batch o ser una sola configuración (se expande como vista).

    Convención (gamma5-hermítica, verificado):
      D_w psi = (4 + m0) psi
                - (1/2) sum_mu [ (1-gamma_mu) U_mu(x) psi(x+mu)
                                 + (1+gamma_mu) U_mu^dag(x-mu) psi(x-mu) ]
    """
    bo = psi.dim() - 6  # offset de batch: 0 sin batch, >=1 con batch
    dev = psi.device
    gs = config.gammas(dtype=psi.dtype, dev=dev)
    eye = torch.eye(4, dtype=psi.dtype, device=dev)

    out = (4.0 + m0) * psi
    apbc = (boundary == "antiperiodic")

    for mu in range(config.Nd):
        one_minus = eye - gs[mu]
        one_plus = eye + gs[mu]
        axis = bo + mu  # eje de la retícula para esta dirección

        U_mu = _expand_to(U[mu], psi.shape[:-2] + (3, 3))
        U_mu_dag = _expand_to(U[mu].mH, psi.shape[:-2] + (3, 3))

        # --- término forward: (1 - gamma_mu) U_mu(x) psi(x + mu_hat) ---
        psi_fwd = torch.roll(psi, -1, dims=axis)
        U_fwd = U_mu
        if apbc and mu == config.t_dir:
            # anti-periódico en tiempo: el enlace que cruza la frontera (x_t=Nt-1) lleva -1
            U_fwd = _flip_boundary(U_fwd, axis, -1)
        # color: tmp[..., s, b] = sum_a U[b,a] psi[s,a] == psi @ U^T
        tmp_fwd = torch.matmul(psi_fwd, U_fwd.transpose(-1, -2))
        # espín: out -= 0.5 * (1-gamma_mu) @ tmp
        out = out - 0.5 * torch.matmul(one_minus, tmp_fwd)

        # --- término backward: (1 + gamma_mu) U_mu^dag(x - mu_hat) psi(x - mu_hat) ---
        psi_bwd = torch.roll(psi, +1, dims=axis)
        U_bwd = torch.roll(U_mu_dag, +1, dims=axis)  # U_mu^dag(x - mu_hat)
        if apbc and mu == config.t_dir:
            U_bwd = _flip_boundary(U_bwd, axis, 0)
        tmp_bwd = torch.matmul(psi_bwd, U_bwd.transpose(-1, -2))
        out = out - 0.5 * torch.matmul(one_plus, tmp_bwd)

    return out


def apply_gamma5(psi):
    """Aplica gamma5 sobre el índice de espín."""
    g5 = config.gamma5(dtype=psi.dtype, dev=psi.device)
    return torch.einsum("ij,...ja->...ia", g5, psi)


def gamma5_hermiticity_error(U, m0, tol_sample=3):
    """Chequea gamma5 D gamma5 == D^dag aplicándolo a vectores aleatorios.

    Devuelve el error máximo relativo (debería ser ~1e-6 o menor en float32).
    """
    shape = U[0].shape[:4]
    dev = U[0].device
    worst = 0.0
    for _ in range(tol_sample):
        psi = torch.randn(*shape, 4, 3, dtype=torch.complex64, device=dev)
        phi = torch.randn(*shape, 4, 3, dtype=torch.complex64, device=dev)
        lhs = apply_gamma5(wilson_dslash(U, apply_gamma5(psi), m0))
        rhs = wilson_dslash(U, phi, m0)
        # <lhs, phi> == <psi, rhs>  es el test de gamma5-hermiticidad
        inner_lhs = (lhs.conj() * phi).sum().abs()
        inner_rhs = (psi.conj() * rhs).sum().abs()
        err = (inner_lhs - inner_rhs).abs() / (inner_lhs.abs() + 1e-30)
        worst = max(worst, err.item())
    return worst


def dense_matrix(U, m0):
    """Construye la matriz densa de D_w (solo para retículas chicas / tests).

    Índice lineal: (t,x,y,z,spin,color) -> un solo índice. Permite validar
    contra linalg.solve y contra las propiedades espectrales.
    """
    shape = U[0].shape[:4]
    n = 1
    for s in shape:
        n *= s
    n *= 4 * 3  # espín x color
    dev = U[0].device
    dtype = U[0].dtype
    D = torch.zeros(n, n, dtype=dtype, device=dev)
    eye = torch.eye(n, dtype=dtype, device=dev)
    # Aplica D_w a cada vector de la base canónica
    for i in range(n):
        e = torch.zeros(n, dtype=dtype, device=dev)
        e[i] = 1.0
        psi = e.reshape(*shape, 4, 3)
        Dcol = wilson_dslash(U, psi, m0)
        D[:, i] = Dcol.reshape(-1)
    return D
