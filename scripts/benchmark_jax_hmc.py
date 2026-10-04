"""Benchmark HMC 2 sabores Wilson end-to-end: torch vs JAX (por niveles de JIT).

Misma config inicial SU(3), misma retícula, complex128, CG tol 1e-12.
Mide tiempo/trayectoria y muestra QUÉ hace falta en JAX para correr rápido:
  - eager (sin jit)      -> lento (overhead de tracing por op)
  - jit(force_total)     -> compila CG+fuerza en un ejecutable XLA
  - jit(hmc_step)        -> compila la trayectoria ENTERA en un ejecutable
"""
import time

import numpy as np
import torch
import jax
import jax.numpy as jnp

from tlattice import su3 as tsu3, force_analytic as tfa
from tlattice_jax import hmc as jhmc

BETA, M0, N_STEPS, EPS, SHAPE = 6.0, -0.5, 10, 0.1, (4, 4, 4, 8)
SEED = 42


def run_torch(U_t, n, warmup=2):
    gen = torch.Generator().manual_seed(SEED)
    for _ in range(warmup):
        U_t, *_ = tfa_hmc_step(U_t, gen)
    ts, accs = [], 0
    for _ in range(n):
        t0 = time.perf_counter()
        U_t, acc = tfa_hmc_step(U_t, gen)
        ts.append(time.perf_counter() - t0)
        accs += int(acc)
    return U_t, ts, accs


def tfa_hmc_step(U_t, gen):
    # reimplementar hmc_step con fuerza analítica (equiv a force_total_analytic)
    from tlattice import fermion_hmc as tfh
    return tfh.hmc_step_fermion(U_t, BETA, M0, N_STEPS, EPS,
                                generator=gen, force_fn=tfa.force_total_analytic)[:2]


def run_jax(hmc_fn, U_j, n, warmup=2):
    key = jax.random.PRNGKey(SEED)
    for _ in range(warmup):
        key, sk = jax.random.split(key)
        U_j, acc, dH = hmc_fn(U_j, BETA, M0, N_STEPS, EPS, sk)
    ts, accs = [], 0
    for _ in range(n):
        key, sk = jax.random.split(key)
        t0 = time.perf_counter()
        U_j, acc, dH = hmc_fn(U_j, BETA, M0, N_STEPS, EPS, sk)
        U_j = [x.block_until_ready() for x in U_j]
        ts.append(time.perf_counter() - t0)
        accs += int(acc)
    return U_j, ts, accs


def main():
    torch.manual_seed(SEED)
    U_t = [tsu3.random_su3(*SHAPE, dtype=torch.complex128) for _ in range(4)]
    U_j0 = [jnp.asarray(u.numpy()) for u in U_t]

    # compilar variantes JAX
    force_jit = jax.jit(jhmc.force_total)

    def hmc_force_jit(U, beta, m0, n_steps, eps, key):
        # leapfrog con force_total jiteado, loop en Python
        shape = U[0].shape[:4]
        key, sk = jax.random.split(key)
        P, key = jhmc.sample_momentum(sk, shape)
        phi, key = jhmc.sample_phi(key, U, m0)
        chi, _, _ = jhmc.solve_chi(U, phi, m0)
        H0 = jhmc.hamiltonian_value(U, beta, P, phi, chi, m0)
        U_new, P_new = _leapfrog_jit(U, P, beta, phi, m0, n_steps, eps, force_jit)
        chi_new, _, _ = jhmc.solve_chi(U_new, phi, m0)
        H1 = jhmc.hamiltonian_value(U_new, beta, P_new, phi, chi_new, m0)
        dH = H1 - H0
        key, sk = jax.random.split(key)
        accept = jax.random.uniform(sk) < jnp.exp(-dH)
        U = [jnp.where(accept, a, b) for a, b in zip(U_new, U)]
        return U, accept, dH

    hmc_step_jit = jax.jit(jhmc.hmc_step, static_argnames=("n_steps",))

    print("=== benchmark HMC 2-flavor Wilson  (4³×8, β=6.0, m0=-0.5, τ=1.0) ===\n")

    # torch (fuerza analítica)
    U_t2, ts_t, acc_t = run_torch(U_t, n=4)
    print(f"  torch (fuerza analítica) : mediana {np.median(ts_t):6.2f} s/traj  acc={acc_t}/4")

    # JAX eager
    U_j, ts, acc = run_jax(jhmc.hmc_step, U_j0, n=3)
    print(f"  JAX eager (sin jit)      : mediana {np.median(ts):6.2f} s/traj  acc={acc}/3")

    # JAX force jit
    U_j, ts, acc = run_jax(hmc_force_jit, U_j0, n=4)
    print(f"  JAX jit(force_total)     : mediana {np.median(ts):6.2f} s/traj  acc={acc}/4")

    # JAX full jit
    U_j, ts, acc = run_jax(hmc_step_jit, U_j0, n=4)
    print(f"  JAX jit(hmc_step)        : mediana {np.median(ts):6.2f} s/traj  acc={acc}/4")


def _leapfrog_jit(U, P, beta, phi, m0, n_steps, eps, force_jit):
    U = list(U)
    P = list(P)
    F = force_jit(U, beta, phi, m0)
    P = [p + 0.5 * eps * f for p, f in zip(P, F)]
    for step in range(n_steps):
        U = [jhmc.core.expm_batched(eps * p) @ u for p, u in zip(P, U)]
        F = force_jit(U, beta, phi, m0)
        if step < n_steps - 1:
            P = [p + eps * f for p, f in zip(P, F)]
        else:
            P = [p + 0.5 * eps * f for p, f in zip(P, F)]
    return U, P


if __name__ == "__main__":
    main()
