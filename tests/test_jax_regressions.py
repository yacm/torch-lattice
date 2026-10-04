"""Regresiones de correctitud descubiertas en la auditoría JAX/Torch.

Se ejecuta directamente porque el proyecto no depende de pytest:
    PYTHONPATH=. .venv/bin/python tests/test_jax_regressions.py
"""
import math

import jax
import torch

from tlattice import fermion_hmc as torch_hmc
from tlattice_jax import hmc as jax_hmc


SHAPE = (8, 8, 4, 4)
N_LINKS = 4 * math.prod(SHAPE)


def test_momentum_matches_kinetic_measure():
    """Ocho grados gaussianos por enlace deben dar E[K/link] = 8/2 = 4."""
    p_jax, _ = jax_hmc.sample_momentum(jax.random.PRNGKey(123), SHAPE)
    k_jax = float(jax_hmc.kinetic(p_jax)) / N_LINKS

    generator = torch.Generator(device="cpu").manual_seed(123)
    p_torch = torch_hmc.sample_momentum(
        SHAPE, dtype=torch.complex128, dev=torch.device("cpu"), generator=generator
    )
    k_torch = float(jax_hmc.kinetic([jax.numpy.asarray(p.numpy()) for p in p_torch])) / N_LINKS

    expected = 4.0
    tolerance = 0.20
    assert abs(k_jax - expected) < tolerance, (k_jax, expected)
    assert abs(k_torch - expected) < tolerance, (k_torch, expected)


def test_cg_reports_zero_rhs_and_nonconvergence():
    import jax.numpy as jnp
    from tlattice_jax import solve

    zero = jnp.zeros((4, 3), dtype=jnp.complex128)
    x, iterations, residual, converged = solve.cg_checked(lambda z: z, zero)
    assert int(iterations) == 0
    assert float(residual) == 0.0
    assert bool(converged)
    assert bool(jnp.all(x == 0))

    rhs = jnp.ones((4, 3), dtype=jnp.complex128)
    # A=2I necesita una iteración; max_iter=0 debe informar que no convergió.
    _, iterations, residual, converged = solve.cg_checked(
        lambda z: 2.0 * z, rhs, max_iter=0, tol=1e-12
    )
    assert int(iterations) == 0
    assert float(residual) == 1.0
    assert not bool(converged)


def test_hmc_rejects_failed_linear_solve():
    import jax.numpy as jnp
    from tlattice_jax import hmc

    shape = (2, 2, 2, 2)
    identity = jnp.eye(3, dtype=jnp.complex128)
    gauge = [jnp.broadcast_to(identity, (*shape, 3, 3)) for _ in range(4)]
    returned, accepted, delta_h = hmc.hmc_step(
        gauge, 6.0, -0.5, 1, 0.1, jax.random.PRNGKey(9), max_iter=0
    )
    assert not bool(accepted)
    assert math.isinf(float(delta_h))
    for old, new in zip(gauge, returned):
        assert bool(jax.numpy.all(old == new))


def test_evenodd_hmc_rejects_failed_linear_solve():
    import jax.numpy as jnp
    from tlattice_jax import evenodd, hmc_eo

    shape = (2, 2, 2, 2)
    identity = jnp.eye(3, dtype=jnp.complex128)
    gauge = [jnp.broadcast_to(identity, (*shape, 3, 3)) for _ in range(4)]
    geo = evenodd.build_geometry(shape)
    returned, accepted, delta_h = hmc_eo.hmc_step_eo(
        gauge, 6.0, -0.5, 1, 0.1, jax.random.PRNGKey(10), geo, shape,
        max_iter=0,
    )
    assert not bool(accepted)
    assert math.isinf(float(delta_h))
    for old, new in zip(gauge, returned):
        assert bool(jax.numpy.all(old == new))


def test_evenodd_rejects_non_bipartite_periodic_lattice():
    from tlattice_jax import evenodd

    for shape in [(3, 4, 4, 4), (4, 3, 4, 4), (3, 3, 3, 3)]:
        try:
            evenodd.build_geometry(shape)
        except ValueError as exc:
            assert "even" in str(exc).lower() or "par" in str(exc).lower()
        else:
            raise AssertionError(f"build_geometry aceptó retícula no bipartita: {shape}")


def test_jax_kernels_preserve_complex64_dtype():
    import jax.numpy as jnp
    from tlattice_jax import core, dirac

    shape = (2, 2, 2, 2)
    identity = jnp.eye(3, dtype=jnp.complex64)
    gauge = [jnp.broadcast_to(identity, (*shape, 3, 3)) for _ in range(4)]
    psi = jnp.ones((*shape, 4, 3), dtype=jnp.complex64)
    assert dirac.dslash(gauge, psi, -0.5).dtype == jnp.complex64
    assert dirac.d_dagger(gauge, psi, -0.5).dtype == jnp.complex64
    assert core.project_tah(gauge[0]).dtype == jnp.complex64


def test_plaquette_components_classify_time_direction_correctly():
    import jax.numpy as jnp
    from tlattice_jax import core, gauge

    shape = (2, 2, 2, 4)
    keys = jax.random.split(jax.random.PRNGKey(77), 8)
    links = []
    for mu in range(4):
        raw = jax.random.normal(keys[2 * mu], (*shape, 3, 3))
        raw = raw + 1j * jax.random.normal(keys[2 * mu + 1], (*shape, 3, 3))
        links.append(core.expm_su3(0.2 * core.project_tah(raw)))

    spatial, temporal, total = gauge.plaquette_components(links)
    spatial_ref = jnp.mean(jnp.stack([
        gauge.plaquette(links, 1, 0).mean(),
        gauge.plaquette(links, 2, 0).mean(),
        gauge.plaquette(links, 2, 1).mean(),
    ]))
    temporal_ref = jnp.mean(jnp.stack([
        gauge.plaquette(links, 3, 0).mean(),
        gauge.plaquette(links, 3, 1).mean(),
        gauge.plaquette(links, 3, 2).mean(),
    ]))
    assert float(jnp.abs(spatial - spatial_ref)) < 1e-14
    assert float(jnp.abs(temporal - temporal_ref)) < 1e-14
    assert float(jnp.abs(total - 0.5 * (spatial_ref + temporal_ref))) < 1e-14


def test_torch_dtype_contracts_and_failed_cg_rejection():
    import torch
    from tlattice import fermion_hmc, gauge, hmc

    shape = (2, 2, 2, 2)
    p64 = hmc.sample_momentum(shape, dtype=torch.complex64)
    assert all(p.dtype == torch.complex64 for p in p64)
    p128 = hmc.sample_momentum(shape, dtype=torch.complex128)
    assert all(p.dtype == torch.complex128 for p in p128)
    pf64 = fermion_hmc.sample_momentum(shape, dtype=torch.complex64)
    assert all(p.dtype == torch.complex64 for p in pf64)

    U = [u.to(torch.complex128).requires_grad_(True) for u in gauge.unit_gauge(shape)]
    gen = torch.Generator().manual_seed(31)
    returned, accepted, delta_h, _, _ = fermion_hmc.hmc_step_fermion(
        U, 6.0, -0.5, 1, 0.1, generator=gen, max_iter=0
    )
    assert not accepted
    assert math.isinf(delta_h) and delta_h > 0
    assert all(torch.equal(a, b) for a, b in zip(returned, U))


def test_jax_fermion_force_preserves_complex64_dtype():
    import jax.numpy as jnp
    from tlattice_jax import hmc

    shape = (2, 2, 2, 2)
    identity = jnp.eye(3, dtype=jnp.complex64)
    gauge = [jnp.broadcast_to(identity, (*shape, 3, 3)) for _ in range(4)]
    chi = jnp.ones((*shape, 4, 3), dtype=jnp.complex64)
    forces = hmc.fermion_force_analytic(gauge, chi, -0.5)
    assert all(force.dtype == jnp.complex64 for force in forces)


def test_hmc_rejects_invalid_integrator_parameters():
    import jax.numpy as jnp
    from tlattice_jax import evenodd, hmc, hmc_eo

    shape = (2, 2, 2, 2)
    identity = jnp.eye(3, dtype=jnp.complex128)
    gauge = [jnp.broadcast_to(identity, (*shape, 3, 3)) for _ in range(4)]
    geo = evenodd.build_geometry(shape)
    calls = [
        lambda: hmc.hmc_step(gauge, 6.0, -0.5, 0, 0.1, jax.random.PRNGKey(1)),
        lambda: hmc.hmc_step(gauge, 6.0, -0.5, 1, 0.0, jax.random.PRNGKey(1)),
        lambda: hmc_eo.hmc_step_eo(
            gauge, 6.0, -0.5, 0, 0.1, jax.random.PRNGKey(1), geo, shape
        ),
        lambda: hmc_eo.hmc_step_eo(
            gauge, 6.0, -0.5, 1, -0.1, jax.random.PRNGKey(1), geo, shape
        ),
    ]
    for call in calls:
        try:
            call()
        except ValueError:
            pass
        else:
            raise AssertionError("HMC aceptó parámetros inválidos del integrador")


def test_cg_checks_true_residual_before_reporting_convergence():
    import jax.numpy as jnp
    from tlattice_jax import solve

    n = 128
    diagonal = jnp.logspace(0, 6, n, dtype=jnp.float32).astype(jnp.complex64)
    rhs = (jnp.arange(n) % 7 + 1).astype(jnp.complex64)
    _, _, residual, converged = solve.cg_checked(
        lambda value: diagonal * value, rhs, max_iter=5000, tol=1e-5
    )
    assert not bool(converged)
    assert float(residual) > 1e-5


def test_torch_cg_checks_true_residual_and_phi_preserves_dtype():
    import torch
    from tlattice import fermion_hmc, gauge, invert

    n = 128
    diagonal = torch.logspace(0, 6, n, dtype=torch.float32).to(torch.complex64)
    diagonal = diagonal.reshape(1, 1, 1, 1, 1, n)
    rhs = ((torch.arange(n) % 7) + 1).to(torch.complex64).reshape(1, 1, 1, 1, 1, n)
    _, iterations, residual = invert.cg(
        lambda value: diagonal * value, rhs, max_iter=5000, tol=1e-5
    )
    if residual > 1e-5:
        raise AssertionError("caso alcanzable debe converger con residuo verdadero")
    assert 0 < iterations < 5000

    shape = (2, 2, 2, 2)
    U = [u.to(torch.complex64) for u in gauge.unit_gauge(shape)]
    phi = fermion_hmc.sample_phi(U, -0.5, generator=torch.Generator().manual_seed(7))
    assert phi.dtype == torch.complex64


def test_jax_hmc_preserves_complex64_and_rejects_dynamic_eps():
    import jax.numpy as jnp
    from tlattice_jax import evenodd, hmc, hmc_eo

    shape = (2, 2, 2, 2)
    identity = jnp.eye(3, dtype=jnp.complex64)
    gauge = [jnp.broadcast_to(identity, (*shape, 3, 3)) for _ in range(4)]
    key = jax.random.PRNGKey(17)
    momentum, _ = hmc.sample_momentum(key, shape, dtype=jnp.complex64)
    phi, _ = hmc.sample_phi(key, gauge, -0.5)
    geo = evenodd.build_geometry(shape)
    phi_eo, _ = hmc_eo.sample_phi_eo(key, gauge, geo, shape, -0.5)
    assert all(value.dtype == jnp.complex64 for value in momentum)
    assert phi.dtype == phi_eo.dtype == jnp.complex64
    returned, _, _ = hmc.hmc_step(gauge, 6.0, -0.5, 1, 0.01, key)
    assert all(value.dtype == jnp.complex64 for value in returned)

    dynamic_step = jax.jit(
        lambda eps: hmc.hmc_step(gauge, 6.0, -0.5, 1, eps, key)[1]
    )
    try:
        dynamic_step(jnp.asarray(0.01))
    except (TypeError, ValueError):
        pass
    else:
        raise AssertionError("eps dinámico bajo jit debe rechazarse; decláralo estático")


def test_torch_hmc_rejects_invalid_integrator_parameters():
    import torch
    from tlattice import fermion_hmc, gauge, hmc

    shape = (2, 2, 2, 2)
    U = [u.to(torch.complex128).requires_grad_(True) for u in gauge.unit_gauge(shape)]
    calls = [
        lambda: hmc.hmc_step(U, 6.0, 0, 0.1),
        lambda: hmc.hmc_step(U, 6.0, 1, float("nan")),
        lambda: fermion_hmc.hmc_step_fermion(U, 6.0, -0.5, 0, 0.1),
        lambda: fermion_hmc.hmc_step_fermion(U, 6.0, -0.5, 1, 0.0),
    ]
    for call in calls:
        try:
            call()
        except ValueError:
            pass
        else:
            raise AssertionError("Torch HMC aceptó parámetros inválidos")


def test_torch_cg_true_residual_rejects_unreachable_tolerance():
    import torch
    from tlattice import invert

    n = 128
    diagonal = torch.logspace(0, 6, n, dtype=torch.float32).to(torch.complex64)
    diagonal = diagonal.reshape(1, 1, 1, 1, 1, n)
    rhs = ((torch.arange(n) % 7) + 1).to(torch.complex64).reshape(1, 1, 1, 1, 1, n)
    _, iterations, residual = invert.cg(
        lambda value: diagonal * value, rhs, max_iter=5000, tol=1e-6
    )
    assert residual > 1e-6
    assert iterations == 5000


if __name__ == "__main__":
    test_momentum_matches_kinetic_measure()
    test_cg_reports_zero_rhs_and_nonconvergence()
    test_hmc_rejects_failed_linear_solve()
    test_evenodd_hmc_rejects_failed_linear_solve()
    test_evenodd_rejects_non_bipartite_periodic_lattice()
    test_jax_kernels_preserve_complex64_dtype()
    test_plaquette_components_classify_time_direction_correctly()
    test_torch_dtype_contracts_and_failed_cg_rejection()
    test_jax_fermion_force_preserves_complex64_dtype()
    test_hmc_rejects_invalid_integrator_parameters()
    test_cg_checks_true_residual_before_reporting_convergence()
    test_torch_cg_checks_true_residual_and_phi_preserves_dtype()
    test_jax_hmc_preserves_complex64_and_rejects_dynamic_eps()
    test_torch_hmc_rejects_invalid_integrator_parameters()
    test_torch_cg_true_residual_rejects_unreachable_tolerance()
    print("=== REGRESIONES JAX/TORCH OK ===")
