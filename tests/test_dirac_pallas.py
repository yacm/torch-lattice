"""Kernel Pallas (GPU) == dirac_soa / dirac, D y D†, fp64 y fp32. Se salta sin GPU."""
import jax
import jax.numpy as jnp

from tlattice_jax import core, dirac, dirac_pallas

SHAPE, M0 = (4, 4, 4, 8), -0.5


def _random_su3(shape, key):
    k1, k2 = jax.random.split(key)
    A = jax.random.normal(k1, (*shape, 3, 3)) + 1j * jax.random.normal(k2, (*shape, 3, 3))
    return core.expm_batched(core.project_tah(A))


def main():
    if jax.default_backend() != "gpu":
        print("SKIP: requiere GPU (Pallas/Triton)")
        return
    key = jax.random.PRNGKey(7)
    U = [_random_su3(SHAPE, k) for k in jax.random.split(key, 4)]
    kr, ki = jax.random.split(jax.random.PRNGKey(8))
    psi = jax.random.normal(kr, (*SHAPE, 4, 3)) + 1j * jax.random.normal(ki, (*SHAPE, 4, 3))

    ok = True
    for dtype, tol in ((jnp.float64, 1e-12), (jnp.float32, 1e-4)):
        for apbc in (True, False):
            op = dirac_pallas.WilsonPallas(SHAPE, M0, dtype, apbc=apbc)
            Up, x = op.pack_gauge(U), op.pack(psi)
            for name, ref_fn, new_fn in (("D ", dirac.dslash, op.dslash),
                                         ("D†", dirac.d_dagger, op.d_dagger)):
                ref = ref_fn(U, psi, M0, apbc=apbc)
                new = op.unpack(new_fn(Up, x))
                d = float(jnp.max(jnp.abs(new - ref)))
                print(f"  {name} {jnp.dtype(dtype).name} apbc={apbc!s:5}  max|Δ| = {d:.2e}")
                ok &= d < tol

    op = dirac_pallas.WilsonPallas(SHAPE, M0, jnp.float64)
    Up, x = op.pack_gauge(U), op.pack(psi)
    ref = dirac.d_dagger(U, dirac.dslash(U, psi, M0), M0)
    d = float(jnp.max(jnp.abs(op.unpack(op.normal(Up, x)) - ref)))
    print(f"  D†D fp64  max|Δ| = {d:.2e}"); ok &= d < 1e-11

    print("OK" if ok else "FALLO")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
