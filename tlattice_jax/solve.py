"""Solvers lineales en JAX (gradiente conjugado sobre ecuaciones normales)."""
import jax
import jax.numpy as jnp

from . import dirac


def cg_checked(matvec, b, max_iter=5000, tol=1e-12):
    """CG con estado explícito: devuelve (x, k, residuo_relativo, convergió).

    Maneja RHS cero sin 0/0 y detiene limpiamente si aparece un denominador no
    positivo o no finito. La bandera es apta para propagarse dentro de ``jax.jit``.
    """
    x = jnp.zeros_like(b)
    r = b  # b - A·0 = b
    p = r
    rsold = jnp.real(jnp.sum(jnp.conj(r) * r))
    rhs_norm = jnp.sqrt(rsold)
    zero_rhs = rhs_norm == 0

    def cond(s):
        k, _, _, _, rs, healthy = s
        relative = jnp.sqrt(rs) / jnp.where(zero_rhs, 1.0, rhs_norm)
        return (k < max_iter) & (relative > tol) & healthy

    def body(s):
        k, x, r, p, rs = s[:5]
        Ap = matvec(p)
        denom = jnp.real(jnp.sum(jnp.conj(p) * Ap))
        good = jnp.isfinite(denom) & (denom > 0) & jnp.isfinite(rs)
        safe_denom = jnp.where(good, denom, 1.0)
        alpha = jnp.where(good, rs / safe_denom, 0.0)
        x_new = x + alpha * p
        r_new = r - alpha * Ap
        rsnew = jnp.real(jnp.sum(jnp.conj(r_new) * r_new))
        good = good & jnp.isfinite(rsnew) & (rsnew >= 0)
        safe_rs = jnp.where(rs > 0, rs, 1.0)
        beta = jnp.where(good, rsnew / safe_rs, 0.0)
        p_new = r_new + beta * p
        return (k + 1, x_new, r_new, p_new, rsnew, good)

    state = (jnp.asarray(0), x, r, p, rsold, jnp.asarray(True))
    k, x, _, _, rs, healthy = jax.lax.while_loop(cond, body, state)
    # El residuo recursivo puede derivar por redondeo; HMC decide con b-Ax real.
    true_r = b - matvec(x)
    true_rs = jnp.real(jnp.sum(jnp.conj(true_r) * true_r))
    true_healthy = jnp.isfinite(true_rs) & (true_rs >= 0)
    resid = jnp.where(zero_rhs, 0.0, jnp.sqrt(true_rs) / rhs_norm)
    converged = zero_rhs | (
        healthy & true_healthy & jnp.isfinite(resid) & (resid <= tol)
    )
    return x, k, resid, converged


def cg(matvec, b, max_iter=5000, tol=1e-12):
    """CG compatible con la API histórica: devuelve (x, k, residuo)."""
    x, k, resid, _ = cg_checked(matvec, b, max_iter=max_iter, tol=tol)
    return x, k, resid


def solve_dirac(U, b, m0, max_iter=5000, tol=1e-12):
    """Resuelve D x = b vía D†D x = D†b."""
    rhs = dirac.d_dagger(U, b, m0)

    def normal(x):
        return dirac.d_dagger(U, dirac.dslash(U, x, m0), m0)

    return cg(normal, rhs, max_iter=max_iter, tol=tol)
