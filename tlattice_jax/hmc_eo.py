"""HMC 2 sabores Wilson PAR-IMPAR (even-odd) en JAX, con operador COMPACTO.

Mismo HMC que hmc.py pero con el operador reducido D̂ sobre sitios pares,
implementado con el hop par<->impar DIRECTO (sin scatter/gather a retícula completa):

  · pseudofermión φ_e = D̂† η_e  (η_e ~ CN(0,1) en pares)
  · solve χ_e = (D̂†D̂)⁻¹ φ_e
  · acción S_PF = Re(φ_e† χ_e)
  · fuerza fermiónica vía jax.grad de un matvec (χ congelado) → exacta.

La física es idéntica a la del HMC sin precondicionar (det D se factoriza), y el
operador compacto está validado (0.000e+00) contra la versión scatter/gather.

`geo = evenodd.build_geometry(shape)` se calcula UNA VEZ (numpy, concreto) y se
cierra sobre él en las funciones jitted (los mapas de vecinos son índices que
deben ser constantes en tiempo de trazado).
"""
import jax
import jax.numpy as jnp

from . import core, gauge, evenodd, solve
from .hmc import sample_momentum, kinetic, validate_integrator_parameters


def sample_phi_eo(key, U, geo, shape, m0):
    """φ_e = D̂† η_e, con η_e ~ CN(0,1) sobre sitios pares (medida 2 sabores)."""
    V2 = int(geo["even_idx"].shape[0])
    complex_dtype = U[0].dtype
    real_dtype = jnp.float32 if complex_dtype == jnp.complex64 else jnp.float64
    key, k1, k2 = jax.random.split(key, 3)
    re = jax.random.normal(k1, (V2, 4, 3), dtype=real_dtype)
    im = jax.random.normal(k2, (V2, 4, 3), dtype=real_dtype)
    eta_e = (re + 1j * im).astype(complex_dtype) / jnp.sqrt(
        jnp.asarray(2.0, dtype=real_dtype)
    )
    U_even, U_odd = evenodd.gather_gauge(U, geo, shape)
    return evenodd.dhat_dagger_compact(U_odd, U_even, eta_e, geo, m0), key


def solve_chi_eo(U, phi_e, geo, shape, m0, max_iter=5000, tol=1e-12):
    """χ_e = (D̂†D̂)⁻¹ φ_e; API histórica sin bandera de convergencia."""
    x, k, resid, _ = solve_chi_eo_checked(
        U, phi_e, geo, shape, m0, max_iter=max_iter, tol=tol
    )
    return x, k, resid


def solve_chi_eo_checked(U, phi_e, geo, shape, m0, max_iter=5000, tol=1e-12):
    """Solve compacto con bandera de convergencia para HMC fail-closed."""
    U_even, U_odd = evenodd.gather_gauge(U, geo, shape)
    matvec = lambda x: evenodd.dhat_dagger_dhat_compact(U_odd, U_even, x, geo, m0)
    return solve.cg_checked(matvec, phi_e, max_iter=max_iter, tol=tol)


def fermion_force_eo(U, chi_e, geo, shape, m0):
    """∂S_PF/∂U = −χ† (∂(D̂†D̂)/∂U) χ, con χ congelado.

    Por el teorema de la función implícita, ∂S_PF/∂U = −∂[Re(χ†D̂†D̂χ)]/∂U
    (χ fijo). Se calcula con jax.grad sobre el escalar Re(χ† (D̂†D̂) χ), que es
    exacto y barato (~1 iteración de costo).

    Convención: jax.grad para complejos cumple df = Re(g·dz), pero el resto del
    código usa df = Re Tr(g† dz) = Re(conj(g)·dz). Además ∂S_PF/∂U = −∂energy/∂U.
    Luego g = −conj(jax.grad(energy)); el conj() arregla la parte imaginaria
    (verificado por diferencia finita contra la identidad dS/dε = −⟨F,F⟩).
    """
    def energy(U):
        U_even, U_odd = evenodd.gather_gauge(U, geo, shape)
        Mchi = evenodd.dhat_dagger_dhat_compact(U_odd, U_even, chi_e, geo, m0)
        return jnp.real(jnp.sum(jnp.conj(chi_e) * Mchi))

    return [-jnp.conj(g) for g in jax.grad(energy)(U)]


def force_total_eo(U, beta, phi_e, geo, shape, m0, max_iter=5000, tol=1e-12):
    """Fuerza molecular EO; API histórica sin bandera del CG."""
    F, _ = force_total_eo_checked(
        U, beta, phi_e, geo, shape, m0, max_iter=max_iter, tol=tol
    )
    return F


def force_total_eo_checked(U, beta, phi_e, geo, shape, m0,
                           max_iter=5000, tol=1e-12):
    """Fuerza EO y bandera que exige convergencia/valores finitos."""
    chi_e, _, _, converged = solve_chi_eo_checked(
        U, phi_e, geo, shape, m0, max_iter=max_iter, tol=tol
    )
    g_gauge = gauge.gauge_force_analytic(U, beta)
    g_pf = fermion_force_eo(U, chi_e, geo, shape, m0)
    F = []
    for gg, gp, u in zip(g_gauge, g_pf, U):
        g = gg + gp
        M = 0.5 * (u @ core.mH(g) - g @ core.mH(u))
        F.append(core.project_tah(M))
    finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(f)) for f in F]))
    return F, converged & finite


def hamiltonian_eo(U, beta, P, phi_e, chi_e):
    return (kinetic(P) + gauge.wilson_action(U, beta)
            + jnp.real(jnp.sum(jnp.conj(phi_e) * chi_e)))


def leapfrog_eo(U, P, beta, phi_e, geo, shape, m0, n_steps, eps,
                max_iter=5000, tol=1e-12):
    U, P, _ = leapfrog_eo_checked(
        U, P, beta, phi_e, geo, shape, m0, n_steps, eps, max_iter, tol
    )
    return U, P


def leapfrog_eo_checked(U, P, beta, phi_e, geo, shape, m0, n_steps, eps,
                        max_iter=5000, tol=1e-12):
    U = list(U)
    P = list(P)
    F, all_ok = force_total_eo_checked(
        U, beta, phi_e, geo, shape, m0, max_iter, tol
    )
    P = [p + 0.5 * eps * f for p, f in zip(P, F)]
    for step in range(n_steps):
        U = [core.expm_su3(eps * p) @ u for p, u in zip(P, U)]
        F, ok = force_total_eo_checked(
            U, beta, phi_e, geo, shape, m0, max_iter, tol
        )
        all_ok = all_ok & ok
        if step < n_steps - 1:
            P = [p + eps * f for p, f in zip(P, F)]
        else:
            P = [p + 0.5 * eps * f for p, f in zip(P, F)]
    return U, P, all_ok


def hmc_step_eo(U, beta, m0, n_steps, eps, key, geo, shape,
                max_iter=5000, tol=1e-12):
    """Un paso HMC EO exacto; cualquier solve fallido fuerza rechazo seguro."""
    validate_integrator_parameters(n_steps, eps)
    key, sk = jax.random.split(key)
    P, key = sample_momentum(sk, shape, dtype=U[0].dtype)
    phi_e, key = sample_phi_eo(key, U, geo, shape, m0)
    chi_e, _, _, ok0 = solve_chi_eo_checked(
        U, phi_e, geo, shape, m0, max_iter, tol
    )
    H0 = hamiltonian_eo(U, beta, P, phi_e, chi_e)
    U_new, P_new, ok_md = leapfrog_eo_checked(
        U, P, beta, phi_e, geo, shape, m0, n_steps, eps, max_iter, tol
    )
    chi_e_new, _, _, ok1 = solve_chi_eo_checked(
        U_new, phi_e, geo, shape, m0, max_iter, tol
    )
    H1 = hamiltonian_eo(U_new, beta, P_new, phi_e, chi_e_new)
    dH_raw = H1 - H0
    all_ok = ok0 & ok_md & ok1 & jnp.isfinite(dH_raw)
    dH = jnp.where(all_ok, dH_raw, jnp.asarray(jnp.inf, dtype=dH_raw.dtype))
    key, sk = jax.random.split(key)
    accept = all_ok & (jax.random.uniform(sk) < jnp.exp(-dH))
    U = [jnp.where(accept, a, b) for a, b in zip(U_new, U)]
    return U, accept, dH
