"""Constantes y utilidades (gammas, generadores SU(3), expm, proyección su(3))."""
import numpy as np

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl

Nc = 3
Nd = 4
t_dir = 3  # tiempo = eje 3 (convención Chroma)

# --- gammas euclídeas DeGrand-Rossi (idénticas a config.py de torch) ---
_G0 = np.array([[0, 0, 0, 1j], [0, 0, 1j, 0], [0, -1j, 0, 0], [-1j, 0, 0, 0]])  # x
_G1 = np.array([[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]])     # y
_G2 = np.array([[0, 0, 1j, 0], [0, 0, 0, -1j], [-1j, 0, 0, 0], [0, 1j, 0, 0]]) # z
_G3 = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]])       # t
GAMMAS = [jnp.asarray(g, dtype=jnp.complex128) for g in [_G0, _G1, _G2, _G3]]
GAMMA5 = GAMMAS[0] @ GAMMAS[1] @ GAMMAS[2] @ GAMMAS[3]
EYE4 = jnp.eye(4, dtype=jnp.complex128)
EYE3 = jnp.eye(3, dtype=jnp.complex128)

# --- generadores SU(3): T_a = lambda_a / 2 ---
_GELLMANN = [
    np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]),
    np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]]),
    np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]]),
    np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]]),
    np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]]),
    np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]]),
    np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]]),
    np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]]) / np.sqrt(3),
]
GENERATORS = [jnp.asarray(g, dtype=jnp.complex128) / 2.0 for g in _GELLMANN]


def mH(x):
    """Conjugada transpuesta sobre las últimas 2 dims (equiv a .mH de torch)."""
    return jnp.conj(x).swapaxes(-1, -2)


def expm_batched(A):
    """Exponencial de matriz batcheada: (..., 3, 3) -> (..., 3, 3)."""
    s = A.shape[:-2]
    return jax.vmap(jsl.expm)(A.reshape(-1, 3, 3)).reshape(*s, 3, 3)


def expm_su3(A):
    """exp(A) para A ANTI-HERMITIANA (su(3)), vía eigendecomposición de iA.

    iA es Hermitiana → eigh(iA) = (θ reales, V ortonormal), iA = V diag(θ) V†.
    Luego A = V diag(−iθ) V† y exp(A) = V diag(e^{−iθ}) V†. ~3× más rápida que
    scipy.expm (vmap) y bit-compatible (max|Δ| ~ 2e-15). Válida para CUALQUIER
    anti-hermitiana 3×3 (no exige traza nula).
    """
    theta, V = jnp.linalg.eigh(1j * A)  # theta: (...,3) real, V: (...,3,3)
    phase = jnp.exp(-1j * theta)[..., None, :]  # (...,1,3)
    return (V * phase) @ jnp.conj(V).swapaxes(-1, -2)


def project_tah(X):
    """Proyección al álgebra su(3): parte anti-hermítica sin traza."""
    ah = 0.5 * (X - mH(X))
    tr = jnp.einsum("...ii->...", ah)
    eye = EYE3.astype(X.dtype)
    return ah - (tr / Nc)[..., None, None] * eye
