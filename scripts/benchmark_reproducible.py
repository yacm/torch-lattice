"""Reproducible fixed-field JAX HMC benchmark.

This measures independent trajectories restarted from the same canonical gauge
field. It is a microbenchmark, not an evolving Markov chain. Run in a fresh
process with CPU affinity fixed externally, for example::

    taskset -c 0 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      PYTHONPATH=. .venv/bin/python scripts/benchmark_reproducible.py \
      --method eo --samples 30 --output benchmarks/results/jax_eo_1c.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from tlattice_jax import evenodd, gauge, hmc, hmc_eo

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
FIELD = ROOT / "benchmarks/canonical/cfg_7.bin"
EXPECTED_SHA256 = "c022cac6a1d3d4857366077ab173a741df5a0eaf975fa34ecc7a127750ba286b"
SHAPE = (4, 4, 4, 8)
BETA = 6.0
M0 = -0.5
N_STEPS = 10
EPS = 0.1


def load_field(path: Path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError(f"SHA-256 inesperado para {path}: {digest}")
    raw = np.fromfile(path, dtype=np.float64)
    expected_values = 4 * int(np.prod(SHAPE)) * 3 * 3 * 2
    if raw.size != expected_values:
        raise ValueError(f"Tamaño incorrecto: {raw.size}, esperado {expected_values}")
    values = raw[0::2] + 1j * raw[1::2]
    values = values.reshape(4, *SHAPE, 3, 3)
    return [jnp.asarray(values[mu].copy()) for mu in range(4)], digest


def block_tree(tree):
    jax.tree_util.tree_map(lambda value: value.block_until_ready(), tree)


def robust_summary(values):
    if not values:
        raise ValueError("No hay muestras")
    median = statistics.median(values)
    if len(values) == 1:
        q1 = q3 = median
    else:
        q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    deviations = [abs(value - median) for value in values]
    return {
        "median": median,
        "mean": statistics.fmean(values),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "mad": statistics.median(deviations),
        "min": min(values),
        "max": max(values),
    }


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metadata(method: str, samples: int, warmups: int, digest: str, cores: int):
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    source_files = [
        ROOT / "tlattice_jax/core.py", ROOT / "tlattice_jax/dirac.py",
        ROOT / "tlattice_jax/evenodd.py", ROOT / "tlattice_jax/gauge.py",
        ROOT / "tlattice_jax/hmc.py", ROOT / "tlattice_jax/hmc_eo.py",
        ROOT / "tlattice_jax/solve.py", Path(__file__).resolve(),
    ]
    return {
        "kind": "jax_fixed_field_hmc_microbenchmark",
        "method": method,
        "cores": cores,
        "samples": samples,
        "warmups": warmups,
        "shape": SHAPE,
        "beta": BETA,
        "m0": M0,
        "n_steps": N_STEPS,
        "eps": EPS,
        "tau": N_STEPS * EPS,
        "cg_tolerance": 1e-12,
        "precision": "complex128",
        "field_sha256": digest,
        "field_path": str(FIELD),
        "jax": jax.__version__,
        "jaxlib": jax.lib.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "devices": [str(device) for device in jax.devices()],
        "cpu_affinity": affinity,
        "source_sha256": {str(path.relative_to(ROOT)): file_digest(path) for path in source_files},
        "environment": {
            key: os.environ.get(key)
            for key in ("XLA_FLAGS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("unprec", "eo"), required=True)
    parser.add_argument("--cores", type=int, choices=(1, 2), required=True)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 1 or args.warmups < 1:
        parser.error("samples y warmups deben ser positivos")
    args.output = args.output.resolve()
    progress_path = args.output.with_suffix(".jsonl")
    if args.output.exists() or progress_path.exists():
        raise FileExistsError(f"El resultado o progreso ya existe: {args.output}, {progress_path}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not hasattr(os, "sched_getaffinity"):
        raise RuntimeError("Esta plataforma no permite verificar afinidad CPU")
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) != args.cores:
        raise RuntimeError(
            f"Afinidad real {affinity}: se esperaban exactamente {args.cores} núcleos; usa taskset"
        )
    U, digest = load_field(FIELD)
    geo = evenodd.build_geometry(SHAPE)
    measured_s, measured_t, measured_plaq_jax = gauge.plaquette_components(U)
    measured_s = float(measured_s)
    measured_t = float(measured_t)
    measured_plaq = float(measured_plaq_jax)
    if abs(measured_plaq - 0.725713584607721) > 1e-14:
        raise RuntimeError(f"Plaqueta canónica inesperada: {measured_plaq}")

    if args.method == "unprec":
        step = jax.jit(lambda field, key: hmc.hmc_step(
            field, BETA, M0, N_STEPS, EPS, key
        ))

        def initial_iterations(field, key):
            key, momentum_key = jax.random.split(key)
            _, key = hmc.sample_momentum(momentum_key, SHAPE)
            phi, _ = hmc.sample_phi(key, field, M0)
            return hmc.solve_chi_checked(field, phi, M0)[1:]
    else:
        step = jax.jit(lambda field, key: hmc_eo.hmc_step_eo(
            field, BETA, M0, N_STEPS, EPS, key, geo, SHAPE
        ))

        def initial_iterations(field, key):
            key, momentum_key = jax.random.split(key)
            _, key = hmc.sample_momentum(momentum_key, SHAPE)
            phi, _ = hmc_eo.sample_phi_eo(key, field, geo, SHAPE, M0)
            return hmc_eo.solve_chi_eo_checked(field, phi, geo, SHAPE, M0)[1:]

    base_key = jax.random.PRNGKey(args.seed)
    keys = jax.random.split(base_key, args.warmups + args.samples)
    for key in keys[:args.warmups]:
        block_tree(step(U, key))

    rows = []
    for index, key in enumerate(keys[args.warmups:]):
        start = time.perf_counter_ns()
        _, accepted, delta_h = step(U, key)
        block_tree((accepted, delta_h))
        elapsed_ms = (time.perf_counter_ns() - start) / 1e6
        iterations, residual, converged = initial_iterations(U, key)
        block_tree((iterations, residual, converged))
        row = {
            "sample": index,
            "elapsed_ms": elapsed_ms,
            "accepted": bool(accepted),
            "delta_h": float(delta_h),
            "initial_cg_iterations": int(iterations),
            "initial_cg_residual": float(residual),
            "initial_cg_converged": bool(converged),
        }
        rows.append(row)
        with progress_path.open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    times = [row["elapsed_ms"] for row in rows]
    cg_iterations = [row["initial_cg_iterations"] for row in rows]
    result = {
        "metadata": metadata(
            args.method, args.samples, args.warmups, digest, args.cores
        ),
        "plaquette": {
            "s_plaq": measured_s,
            "t_plaq": measured_t,
            "w_plaq": measured_plaq,
        },
        "summary": {
            "trajectory_ms": robust_summary(times),
            "initial_cg_iterations": robust_summary(cg_iterations),
            "acceptance_rate": statistics.fmean(int(row["accepted"]) for row in rows),
        },
        "samples": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["metadata"], indent=2))
    print(json.dumps(result["summary"], indent=2))
    print(f"resultado: {args.output}")


if __name__ == "__main__":
    main()
