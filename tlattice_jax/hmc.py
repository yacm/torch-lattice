"""HMC 2 sabores Wilson en JAX (fuerza analítica, sin autograd)."""
import math

import jax
import jax.numpy as jnp

from . import core, dirac, gauge, solve


def validate_integrator_parameters(n_steps, eps):
    """Exige parámetros estáticos, finitos y físicamente válidos."""
    if not isinstance(n_steps, int) or isinstance(n_steps, bool) or n_steps < 1:
        raise ValueError("n_steps debe ser un entero >= 1")
    if isinstance(eps, jax.core.Tracer):
        raise TypeError("eps debe cerrarse como constante Python antes de jax.jit")
    eps_value = float(eps)
    if not math.isfinite(eps_value) or eps_value <= 0.0:
        raise ValueError("eps debe ser finito y positivo")


def sample_momentum(key, shape, dtype=jnp.complex128):
    """Momento canónico P_μ = i Σ_a c_a T_a, c_a ~ N(0,2), en su(3).

    Con Tr(T_a T_b)=δ_ab/2 y K=−½Tr(P²)=¼Σc_a², la varianza 2 produce
    la medida exp(−K). Usar N(0,1) produciría incorrectamente exp(−2K).
    """
    if dtype not in (jnp.complex64, jnp.complex128):
        raise ValueError("dtype debe ser complex64 o complex128")
    real_dtype = jnp.float32 if dtype == jnp.complex64 else jnp.float64
    generators = [g.astype(dtype) for g in core.GENERATORS]
    P = []
    for _ in range(core.Nd):
        p = jnp.zeros((*shape, 3, 3), dtype=dtype)
        for T in generators:
            key, sk = jax.random.split(key)
            c = jnp.sqrt(jnp.asarray(2.0, dtype=real_dtype)) * jax.random.normal(
                sk, shape, dtype=real_dtype
            )
            p = p + (1j * c)[..., None, None] * T
        P.append(p)
    return P, key


def sample_phi(key, U, m0):
    """phi = D† eta, eta ~ N(0,1) complejo (medida correcta para 2 sabores)."""
    shape = U[0].shape[:4]
    complex_dtype = U[0].dtype
    real_dtype = jnp.float32 if complex_dtype == jnp.complex64 else jnp.float64
    key, k1, k2 = jax.random.split(key, 3)
    re = jax.random.normal(k1, (*shape, 4, 3), dtype=real_dtype)
    im = jax.random.normal(k2, (*shape, 4, 3), dtype=real_dtype)
    eta = (re + 1j * im).astype(complex_dtype) / jnp.sqrt(
        jnp.asarray(2.0, dtype=real_dtype)
    )
    return dirac.d_dagger(U, eta, m0), key


def solve_chi(U, phi, m0, max_iter=5000, tol=1e-12):
    """chi = (D†D)⁻¹ phi vía CG en ecuaciones normales."""
    x, k, resid, _ = solve_chi_checked(U, phi, m0, max_iter=max_iter, tol=tol)
    return x, k, resid


def solve_chi_checked(U, phi, m0, max_iter=5000, tol=1e-12):
    """Versión con bandera de convergencia para que HMC pueda fallar cerrado."""
    def normal(x):
        return dirac.d_dagger(U, dirac.dslash(U, x, m0), m0)
    return solve.cg_checked(normal, phi, max_iter=max_iter, tol=tol)


def kinetic(P):
    """K = -(1/2) Σ Tr(P²) > 0 para P anti-hermítico."""
    return -0.5 * sum(jnp.einsum("...ij,...ji->...", p, p).real.sum() for p in P)


def fermion_force_analytic(U, chi, m0):
    """g_PF_μ = ∂S_PF/∂U_μ = Λ_f† + Λ_b  (ψ = Dχ, χ congelado)."""
    # APBC: hornear el −1 en el enlace temporal de la frontera (U_eff)
    U_eff = list(U)
    U_eff[core.t_dir] = U_eff[core.t_dir].at[..., -1, :, :].multiply(-1.0)
    psi = dirac.dslash(U_eff, chi, m0, apbc=False)  # periódico (flip ya en U_eff)

    F = []
    for mu in range(core.Nd):
        om = (core.EYE4 - core.GAMMAS[mu]).astype(chi.dtype)
        op = (core.EYE4 + core.GAMMAS[mu]).astype(chi.dtype)
        chi_fwd = jnp.roll(chi, -1, axis=mu)          # χ(x+μ)
        psi_fwd = jnp.roll(psi, -1, axis=mu)          # ψ(x+μ)
        # Λ_f = tr_spin[(1−γ)χ(x+μ) ⊗ ψ(x)†]
        chi_til_f = jnp.matmul(om, chi_fwd)
        Lambda_f = jnp.matmul(jnp.swapaxes(chi_til_f, -1, -2), jnp.conj(psi))
        # Λ_b = tr_spin[(1+γ)χ(x) ⊗ ψ(x+μ)†]
        chi_til_b = jnp.matmul(op, chi)
        Lambda_b = jnp.matmul(jnp.swapaxes(chi_til_b, -1, -2), jnp.conj(psi_fwd))
        g = core.mH(Lambda_f) + Lambda_b
        if mu == core.t_dir:  # revertir el −1 sobre la fuerza física
            g = g.at[..., -1, :, :].multiply(-1.0)
        F.append(g)
    return F


def force_total(U, beta, phi, m0, max_iter=5000, tol=1e-12):
    """Fuerza molecular total; API histórica sin exponer la bandera del CG."""
    F, _ = force_total_checked(U, beta, phi, m0, max_iter=max_iter, tol=tol)
    return F


def force_total_checked(U, beta, phi, m0, max_iter=5000, tol=1e-12):
    """Fuerza total y bandera que exige convergencia/valores finitos del CG."""
    chi, _, _, converged = solve_chi_checked(
        U, phi, m0, max_iter=max_iter, tol=tol
    )
    g_gauge = gauge.gauge_force_analytic(U, beta)
    g_pf = fermion_force_analytic(U, chi, m0)
    F = []
    for gg, gp, u in zip(g_gauge, g_pf, U):
        g = gg + gp
        M = 0.5 * (u @ core.mH(g) - g @ core.mH(u))
        F.append(core.project_tah(M))
    finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(f)) for f in F]))
    return F, converged & finite


def hamiltonian_value(U, beta, P, phi, chi, m0):
    return (kinetic(P) + gauge.wilson_action(U, beta)
            + jnp.real(jnp.sum(jnp.conj(phi) * chi)))


def leapfrog(U, P, beta, phi, m0, n_steps, eps, max_iter=5000, tol=1e-12):
    U, P, _ = leapfrog_checked(
        U, P, beta, phi, m0, n_steps, eps, max_iter=max_iter, tol=tol
    )
    return U, P


def leapfrog_checked(U, P, beta, phi, m0, n_steps, eps,
                     max_iter=5000, tol=1e-12):
    """Leapfrog que acumula el estado de todos los solves del trayecto."""
    U = list(U)
    P = list(P)
    F, all_ok = force_total_checked(U, beta, phi, m0, max_iter, tol)
    P = [p + 0.5 * eps * f for p, f in zip(P, F)]
    for step in range(n_steps):
        U = [core.expm_su3(eps * p) @ u for p, u in zip(P, U)]
        F, ok = force_total_checked(U, beta, phi, m0, max_iter, tol)
        all_ok = all_ok & ok
        if step < n_steps - 1:
            P = [p + eps * f for p, f in zip(P, F)]
        else:
            P = [p + 0.5 * eps * f for p, f in zip(P, F)]
    return U, P, all_ok


def hmc_step(U, beta, m0, n_steps, eps, key, max_iter=5000, tol=1e-12):
    """Un paso HMC exacto; cualquier solve fallido fuerza rechazo seguro."""
    validate_integrator_parameters(n_steps, eps)
    shape = U[0].shape[:4]
    key, sk = jax.random.split(key)
    P, key = sample_momentum(sk, shape, dtype=U[0].dtype)
    phi, key = sample_phi(key, U, m0)
    chi, _, _, ok0 = solve_chi_checked(U, phi, m0, max_iter, tol)
    H0 = hamiltonian_value(U, beta, P, phi, chi, m0)
    U_new, P_new, ok_md = leapfrog_checked(
        U, P, beta, phi, m0, n_steps, eps, max_iter, tol
    )
    chi_new, _, _, ok1 = solve_chi_checked(U_new, phi, m0, max_iter, tol)
    H1 = hamiltonian_value(U_new, beta, P_new, phi, chi_new, m0)
    dH_raw = H1 - H0
    all_ok = ok0 & ok_md & ok1 & jnp.isfinite(dH_raw)
    dH = jnp.where(all_ok, dH_raw, jnp.asarray(jnp.inf, dtype=dH_raw.dtype))
    key, sk = jax.random.split(key)
    accept = all_ok & (jax.random.uniform(sk) < jnp.exp(-dH))
    U = [jnp.where(accept, a, b) for a, b in zip(U_new, U)]
    return U, accept, dH
