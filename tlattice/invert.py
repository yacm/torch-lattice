"""Gradiente conjugado y solvers para el operador de Dirac."""
import torch
from . import dirac


def cg(matvec, b, x0=None, max_iter=2000, tol=1e-10):
    """CG para A x = b con A hermítica definida positiva (batch-aware).

    matvec: callable x -> A(x). Devuelve (x, num_iter, residuo_relativo).

    Si b lleva batch en las dims líderes, se resuelven todos los sistemas en
    paralelo (lockstep): cada uno conserva su propio residual y se detiene
    cuando todos convergen.
    """
    x = x0 if x0 is not None else torch.zeros_like(b)
    r = b - matvec(x)
    p = r.clone()

    # sumas sobre los ejes de campo (retícula + espín + color), no sobre batch
    field_dims = tuple(range(b.dim() - 6, b.dim()))

    def inner(v, w):
        return (v.conj() * w).real.sum(dim=field_dims)

    def bcast(a):
        # agrega 6 ejes singletón al final para multiplicar contra un campo
        return a.reshape(a.shape + (1,) * 6)

    rsold = inner(r, r)
    rhs_norm = inner(b, b).sqrt()

    def explicit_residual(candidate):
        true_r = b - matvec(candidate)
        numerator = inner(true_r, true_r).sqrt()
        safe_rhs = torch.where(rhs_norm == 0, torch.ones_like(rhs_norm), rhs_norm)
        relative = numerator / safe_rhs
        return torch.where(rhs_norm == 0, torch.zeros_like(relative), relative)

    iterations = 0
    for k in range(max_iter):
        Ap = matvec(p)
        denom = inner(p, Ap)
        if (denom.abs() < 1e-30).all():
            break
        alpha = rsold / denom
        x = x + bcast(alpha) * p
        r = r - bcast(alpha) * Ap
        rsnew = inner(r, r)
        iterations = k + 1
        resid = rsnew.sqrt() / torch.where(
            rhs_norm == 0, torch.ones_like(rhs_norm), rhs_norm
        )
        if (resid <= tol).all():
            true_resid = explicit_residual(x)
            if (true_resid <= tol).all():
                return x, iterations, true_resid.max().item()
        p = r + bcast(rsnew / rsold) * p
        rsold = rsnew
    resid = explicit_residual(x)
    return x, iterations, resid.max().item()


def _d(U, x, m0):
    return dirac.wilson_dslash(U, x, m0)


def _d_dagger(U, x, m0):
    # D^dag = gamma5 D gamma5 (gamma5-hermiticidad)
    return dirac.apply_gamma5(_d(U, dirac.apply_gamma5(x), m0))


def solve_dirac(U, b, m0, max_iter=2000, tol=1e-10):
    """Resuelve D x = b via ecuaciones normales D^dag D x = D^dag b."""
    rhs = _d_dagger(U, b, m0)

    def normal(x):
        return _d_dagger(U, _d(U, x, m0), m0)

    x, it, resid = cg(normal, rhs, max_iter=max_iter, tol=tol)
    return x, it, resid
