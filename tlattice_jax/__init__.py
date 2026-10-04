"""Port completo del HMC 2 sabores Wilson a JAX. Coexiste con tlattice/ (torch)."""
import jax

jax.config.update("jax_enable_x64", True)

from . import core, dirac, gauge, solve, hmc  # noqa: E402,F401
from .hmc import (  # noqa: E402,F401
    force_total,
    solve_chi,
    hamiltonian_value,
    leapfrog,
    hmc_step,
    kinetic,
    sample_momentum,
    sample_phi,
)
