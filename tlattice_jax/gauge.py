"""Acción de gauge de Wilson y su fuerza analítica (staple), en JAX."""
import jax.numpy as jnp

from . import core


def plaquette(U, mu, nu):
    U_nu_fwd = jnp.roll(U[nu], -1, axis=mu)   # U_nu(x+mu)
    U_mu_fwd = jnp.roll(U[mu], -1, axis=nu)   # U_mu(x+nu)
    pl = U[mu] @ U_nu_fwd @ core.mH(U_mu_fwd) @ core.mH(U[nu])
    tr = jnp.einsum("...ii->...", pl).real
    return tr / core.Nc


def wilson_action(U, beta):
    S = 0.0
    for mu in range(1, core.Nd):
        for nu in range(mu):
            S = S + (1.0 - plaquette(U, mu, nu)).sum()
    return beta * S


def plaquette_components(U):
    """Devuelve (s_plaq, t_plaq, w_plaq) con tiempo en ``core.t_dir``.

    ``s_plaq`` promedia los tres planos puramente espaciales y ``t_plaq`` los
    tres que contienen el eje temporal, igual que Chroma ``MesPlq``.
    """
    spatial = []
    temporal = []
    for mu in range(1, core.Nd):
        for nu in range(mu):
            value = plaquette(U, mu, nu).mean()
            if mu == core.t_dir or nu == core.t_dir:
                temporal.append(value)
            else:
                spatial.append(value)
    s_plaq = jnp.mean(jnp.stack(spatial))
    t_plaq = jnp.mean(jnp.stack(temporal))
    return s_plaq, t_plaq, 0.5 * (s_plaq + t_plaq)


def average_plaquette(U):
    return plaquette_components(U)[2]


def gauge_force_analytic(U, beta):
    """g_μ = ∂S_wilson/∂U_μ = -(β/Nc) Σ_{ν≠μ}(S_f† + S_b)."""
    U_mH = [core.mH(u) for u in U]
    F = []
    for mu in range(core.Nd):
        g = jnp.zeros_like(U[mu])
        for nu in range(core.Nd):
            if nu == mu:
                continue
            U_nu_f = jnp.roll(U[nu], -1, axis=mu)          # U_nu(x+mu)
            S_f = U_nu_f @ jnp.roll(U_mH[mu], -1, axis=nu) @ U_mH[nu]
            S_b = (jnp.roll(U_mH[nu], 1, axis=nu)
                   @ jnp.roll(U[mu], 1, axis=nu)
                   @ jnp.roll(U_nu_f, 1, axis=nu))
            g = g - (beta / core.Nc) * (core.mH(S_f) + S_b)
        F.append(g)
    return F
