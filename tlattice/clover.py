"""Fermiones clover (Sheikholeslami-Wohlert) y tensor de campo clover.

El operador de Dirac clover es:

    D_clover = D_wilson(U~)  -  (c_sw/4) sum_{mu<nu} [gamma_mu, gamma_nu] F_mu_nu

donde U~ son los enlaces stout-smeared, y F_mu_nu = (1/8)(Q_mu_nu - Q_mu_nu^dag)
es la "hoja de clover" (anti-hermitiana sin traza), con Q_mu_nu la suma de las
4 plaquetas orientadas que comparten la esquina x.

Convención SW estándar (sigma_mu_nu = (i/2)[gamma_mu, gamma_nu]):
    S_SW = (i/4) c_sw sum_{mu,nu} psibar sigma_mu_nu F_mu_nu psi
         = -(c_sw/4) sum_{mu<nu} psibar [gamma_mu, gamma_nu] F_mu_nu psi

El término clover es LOCAL en posición (no hay shift): acopla espín y color en
el mismo sitio. Es gamma5-hermitico con F anti-hermitiana.
"""
import torch

from . import config, dirac

_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def sigma_commutators(dtype, dev):
    """[gamma_mu, gamma_nu] para los 6 planos (mu<nu), cada uno (4,4)."""
    gs = config.gammas(dtype, dev)
    return [gs[mu] @ gs[nu] - gs[nu] @ gs[mu] for mu, nu in _PAIRS]


def clover_leaf(U, mu, nu):
    """Q_mu,nu(x) = suma de las 4 plaquetas orientadas (los 4 cuadrantes) con base x.

    Traducción exacta de mesField de Chroma. Las 4 plaquetas comparten la esquina x
    (gauge-covariante: bajo U_mu(x)->g(x)U_mu(x)g(x+mu)^dag, Q->g(x)Qg(x)^dag):

        U_mu,nu   =  u_mu(x)  u_nu(x+mu)        u_mu^dag(x+nu)   u_nu^dag(x)
        U_nu,-mu  =  u_nu(x)  u_mu^dag(x+nu-mu) u_nu^dag(x-mu)   u_mu(x-mu)
        U_-mu,-nu =  u_mu^dag(x-mu) u_nu^dag(x-mu-nu) u_mu(x-mu-nu) u_nu(x-nu)
        U_-nu,mu  =  u_nu^dag(x-nu) u_mu(x-nu) u_nu(x-nu+mu) u_mu^dag(x)

    y Q = U_mu,nu + U_nu,-mu + U_-mu,-nu + U_-nu,mu. (Ver mesfield.cc:
    tmp_0=u_nu*u_mu(x+nu), tmp_1=u_mu*u_nu(x+mu).)
    """
    A = torch.roll(U[nu], -1, mu)     # u_nu(x+mu)
    B = torch.roll(U[mu], -1, nu)     # u_mu(x+nu)
    C = U[nu] @ B                     # u_nu(x) u_mu(x+nu)
    D = U[mu] @ A                     # u_mu(x) u_nu(x+mu)

    Q = D @ C.mH                                      # U_mu,nu
    Q = Q + torch.roll(torch.roll(C.mH @ D, +1, nu), +1, mu)  # U_-mu,-nu

    tmp1 = B @ A.mH                                   # u_mu(x+nu) u_nu^dag(x+mu)
    tmp0 = U[nu].mH @ U[mu]                           # u_nu^dag(x) u_mu(x)
    Q = Q + torch.roll(tmp0 @ tmp1.mH, +1, nu)        # U_-nu,mu
    Q = Q + torch.roll(tmp1.mH @ tmp0, +1, mu)        # U_nu,-mu
    return Q


def clover_field_strength(U, traceless=False):
    """F_mu,nu = (1/8)(Q_mu,nu - Q_mu,nu^dag), anti-hermitiana, para mu<nu.

    Por defecto igual que mesField de Chroma (CON traza). Con traceless=True se
    proyecta a su(3) (parte sin traza), que es la convención SW estándar.
    """
    F = []
    for mu, nu in _PAIRS:
        Q = clover_leaf(U, mu, nu)
        Fmn = (Q - Q.mH) / 8.0  # anti-hermitiana
        if traceless:
            eye = torch.eye(3, dtype=U[0].dtype, device=U[0].device)
            tr = torch.einsum("...ii->...", Fmn)
            Fmn = Fmn - (tr / 3.0)[..., None, None] * eye
        F.append(Fmn)
    return F


def clover_term(U, psi, c_sw):
    """Aplica -(c_sw/4) sum_{mu<nu} [gamma_mu, gamma_nu] F_mu_nu psi (local)."""
    F = clover_field_strength(U)
    comms = sigma_commutators(psi.dtype, psi.device)
    acc = 0
    for Fmn, comm in zip(F, comms):
        Fpsi = torch.einsum("...ab,...sb->...sa", Fmn, psi)  # F actúa en color
        acc = acc + torch.einsum("ij,...ja->...ia", comm, Fpsi)  # [gamma,gamma] en espín
    return -(c_sw / 4.0) * acc


def clover_dslash(U, psi, m0, c_sw, boundary="antiperiodic"):
    """D_clover psi = D_wilson(U, psi) + término clover. U ya smeared si hace falta.

    boundary: "antiperiodic" (default, APBC en t_dir) o "periodic".
    """
    return dirac.wilson_dslash(U, psi, m0, boundary=boundary) + clover_term(U, psi, c_sw)
