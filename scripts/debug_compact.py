"""Debug del hop compacto: compara contra el hop completo en pasos aislados."""
import numpy as np
import jax
import jax.numpy as jnp

from tlattice_jax import core, evenodd

jax.config.update("jax_enable_x64", True)

SHAPE = (4, 4, 4, 8)
V = int(np.prod(SHAPE))


def random_su3(shape, key):
    k1, k2 = jax.random.split(key)
    A = jax.random.normal(k1, (*shape, 3, 3)) + 1j * jax.random.normal(k2, (*shape, 3, 3))
    return core.expm_batched(core.project_tah(A))


key = jax.random.PRNGKey(7)
U = [random_su3(SHAPE, k) for k in jax.random.split(key, 4)]
even_idx, odd_idx = evenodd.checkerboard(SHAPE)
geo = evenodd.build_geometry(SHAPE)

k1, k2 = jax.random.split(key)
psi = jax.random.normal(k1, (*SHAPE, 4, 3)) + 1j * jax.random.normal(k2, (*SHAPE, 4, 3))

# --- 1) equivalencia single-bake vs doble flip ---
U_eff = list(U)
U_eff[core.t_dir] = U_eff[core.t_dir].at[..., -1, :, :].multiply(-1.0)
h_full_apbc = evenodd.hop(U, psi, apbc=True)
h_full_baked = evenodd.hop(U_eff, psi, apbc=False)
print(f"1) hop completo: apbc=True vs baked+periódico  max|Δ| = {float(jnp.max(jnp.abs(h_full_apbc - h_full_baked))):.3e}")

# --- 2) hop_oe compacto vs hop completo (sector impar) ---
U_even, U_odd = evenodd.gather_gauge(U, geo, SHAPE)
psi_e = psi.reshape(V, 4, 3)[even_idx]
psi_full = jnp.zeros((V, 4, 3), dtype=psi.dtype).at[even_idx].set(psi_e).reshape(*SHAPE, 4, 3)
h_full = evenodd.hop(U, psi_full)               # H ψ, no-cero en impares
h_full_odd = h_full.reshape(V, 4, 3)[odd_idx]
h_c = evenodd.hop_oe(U_odd, U_even, psi_e, geo)
print(f"2) hop_oe compacto vs hop completo (impar)  max|Δ| = {float(jnp.max(jnp.abs(h_full_odd - h_c))):.3e}")

# --- 3) D̂†D̂ compacto vs scatter ---
ddd_scatter = evenodd.dhat_dagger_dhat(U, psi_e, even_idx, SHAPE, -0.5)
ddd_compact = evenodd.dhat_dagger_dhat_compact(U_odd, U_even, psi_e, geo, -0.5)
print(f"3) D̂†D̂ compacto vs scatter/gather           max|Δ| = {float(jnp.max(jnp.abs(ddd_compact - ddd_scatter))):.3e}")

# --- 4) si el hop_oe está bien, chequear hop_eo y D̂ ---
h_eo_c = evenodd.hop_eo(U_even, U_odd, h_c, geo)
# referencia: H(Hψ) en pares = H² ψ en pares
h2_full = evenodd.hop(U, h_full)
h2_full_even = h2_full.reshape(V, 4, 3)[even_idx]
print(f"4) hop_eo(hop_oe(ψ_e)) vs H²ψ en pares       max|Δ| = {float(jnp.max(jnp.abs(h2_full_even - h_eo_c))):.3e}")

# --- 5) verifica que even_idx de build_geometry == checkerboard ---
print(f"5) even_idx iguales: {np.array_equal(np.asarray(geo['even_idx']), np.asarray(even_idx))}")
print(f"   odd_idx iguales:  {np.array_equal(np.asarray(geo['odd_idx']), np.asarray(odd_idx))}")
