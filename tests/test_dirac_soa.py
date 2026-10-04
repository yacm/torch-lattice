"""dirac_soa (layout SoA + half-spinor) == dirac (referencia), D y D†, con/sin APBC."""
import jax
import jax.numpy as jnp

from tlattice_jax import core, dirac, dirac_soa

SHAPE, M0 = (4, 4, 4, 8), -0.5


def _random_su3(shape, key):
    k1, k2 = jax.random.split(key)
    A = jax.random.normal(k1, (*shape, 3, 3)) + 1j * jax.random.normal(k2, (*shape, 3, 3))
    return core.expm_batched(core.project_tah(A))


def main():
    key = jax.random.PRNGKey(7)
    U = [_random_su3(SHAPE, k) for k in jax.random.split(key, 4)]
    kr, ki = jax.random.split(jax.random.PRNGKey(8))
    psi = jax.random.normal(kr, (*SHAPE, 4, 3)) + 1j * jax.random.normal(ki, (*SHAPE, 4, 3))
    Us, ps = dirac_soa.to_soa(U, psi)

    ok = True
    for name, ref_fn, new_fn in (("D ", dirac.dslash, dirac_soa.dslash),
                                 ("D†", dirac.d_dagger, dirac_soa.d_dagger)):
        for apbc in (True, False):
            ref = ref_fn(U, psi, M0, apbc=apbc)
            new = dirac_soa.psi_from_soa(new_fn(Us, ps, M0, apbc=apbc))
            d = float(jnp.max(jnp.abs(new - ref)))
            print(f"  {name} apbc={apbc!s:5}  max|Δ| = {d:.2e}")
            ok &= d < 1e-12

    d32 = dirac_soa.psi_from_soa(
        dirac_soa.dslash(Us.astype(jnp.complex64), ps.astype(jnp.complex64), M0))
    d = float(jnp.max(jnp.abs(d32 - dirac.dslash(U, psi, M0))))
    print(f"  D  complex64 vs ref fp64  max|Δ| = {d:.2e}"); ok &= d < 1e-4

    # <χ, Dψ> = <D†χ, ψ>
    chi = psi[::-1] * (1 + 0.3j)
    _, cs = dirac_soa.to_soa(U, chi)
    lhs = jnp.vdot(cs, dirac_soa.dslash(Us, ps, M0))
    rhs = jnp.vdot(dirac_soa.d_dagger(Us, cs, M0), ps)
    d = float(jnp.abs(lhs - rhs) / jnp.abs(lhs))
    print(f"  adjunta <χ,Dψ>=<D†χ,ψ>  rel = {d:.2e}"); ok &= d < 1e-12

    print("OK" if ok else "FALLO")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
