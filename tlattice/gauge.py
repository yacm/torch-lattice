"""Campo de gauge SU(3), plaqueto y acción de Wilson (convenciones de Chroma)."""
import torch
from . import config
from . import su3


def unit_gauge(shape, dtype=torch.complex64, dev=None):
    """Campo de gauge trivial U_mu = I. shape = (Nt, Nx, Ny, Nz)."""
    dev = dev if dev is not None else config.device()
    I = torch.eye(3, dtype=dtype, device=dev)
    U = []
    for _ in range(config.Nd):
        U.append(I.expand(*shape, 3, 3).clone())
    return U


def random_gauge(shape, dtype=torch.complex64, dev=None, generator=None):
    """Campo de gauge con enlaces SU(3) aleatorios en cada sitio/dirección."""
    dev = dev if dev is not None else config.device()
    return [su3.random_su3(*shape, dtype=dtype, dev=dev, generator=generator)
            for _ in range(config.Nd)]


def shift_forward(t, direction):
    """t(x + mu_hat): desplaza el tensor una posición hacia 'adelante'."""
    return torch.roll(t, -1, dims=direction)


def shift_backward(t, direction):
    """t(x - mu_hat): desplaza el tensor una posición hacia 'atrás'."""
    return torch.roll(t, +1, dims=direction)


def plaquette(U, mu, nu):
    """Plaqueto por sitio (Re Tr / Nc) en el plano (mu, nu), forma (*lattice)."""
    U_mu = U[mu]
    U_nu = U[nu]
    U_nu_fwd = shift_forward(U_nu, mu)    # U_nu(x + mu_hat)
    U_mu_fwd = shift_forward(U_mu, nu)    # U_mu(x + nu_hat)
    pl = U_mu @ U_nu_fwd @ U_mu_fwd.mH @ U_nu.mH
    tr = torch.einsum("...ii->...", pl).real
    return tr / config.Nc


def average_plaquette(U):
    """Promedios de Chroma: s_plaq (planos espaciales), t_plaq (temporales), w_plaq."""
    s = 0.0
    n_s = 0
    t = 0.0
    n_t = 0
    for mu in range(1, config.Nd):
        for nu in range(mu):
            pl = plaquette(U, mu, nu).mean()
            if nu == 0:
                t += pl.item()
                n_t += 1
            else:
                s += pl.item()
                n_s += 1
    s_plaq = s / n_s
    t_plaq = t / n_t
    w_plaq = (s_plaq + t_plaq) / 2.0
    return {"s_plaq": s_plaq, "t_plaq": t_plaq, "w_plaq": w_plaq}


def wilson_action(U, beta):
    """Acción de gauge de Wilson S = beta * sum_{sitios,planos} (1 - ReTr/Nc)."""
    S = 0.0
    for mu in range(1, config.Nd):
        for nu in range(mu):
            S = S + (1.0 - plaquette(U, mu, nu)).sum()
    return beta * S


def rectangle(U, mu, nu):
    """Loop 1x2 por sitio (Re Tr / Nc): 2 enlaces en mu, 1 en nu."""
    U_mu = U[mu]
    U_nu = U[nu]
    U_mu_f1 = shift_forward(U_mu, mu)                          # U_mu(x+mu)
    U_nu_f2 = shift_forward(shift_forward(U_nu, mu), mu)       # U_nu(x+2mu)
    U_mu_f1_nu = shift_forward(U_mu_f1, nu)                    # U_mu(x+mu+nu)
    U_mu_f_nu = shift_forward(U_mu, nu)                        # U_mu(x+nu)
    R = U_mu @ U_mu_f1 @ U_nu_f2 @ U_mu_f1_nu.mH @ U_mu_f_nu.mH @ U_nu.mH
    tr = torch.einsum("...ii->...", R).real
    return tr / config.Nc


def symanzik_action(U, beta, u0=1.0):
    """Acción de gauge Symanzik tree-level tadpole-improved (Lüscher-Weisz).

    S_G = beta [ (5/3) sum_{mu<nu} (1 - P_mu_nu)
                 - (1/(12 u0^2)) sum_{mu!=nu} (1 - R_mu_nu) ]
    con P la plaqueta 1x1 y R el rectángulo 1x2 (2 enlaces en mu).
    """
    c0 = 5.0 / 3.0
    c1 = -1.0 / (12.0 * u0 ** 2)
    S = 0.0
    for mu in range(1, config.Nd):
        for nu in range(mu):
            S = S + c0 * (1.0 - plaquette(U, mu, nu)).sum()
            S = S + c1 * (1.0 - rectangle(U, mu, nu)).sum()
            S = S + c1 * (1.0 - rectangle(U, nu, mu)).sum()
    return beta * S


def tadpole_u0(U):
    """u0 = <ReTr P/3>^(1/4) del plaqueto promedio (factor tadpole)."""
    p = average_plaquette(U)["w_plaq"]
    return p ** 0.25
