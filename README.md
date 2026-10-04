# Torch-Lattice

Un núcleo de **QCD de retícula (fermiones de Wilson + gauge SU(3))** escrito en
**PyTorch**, *device-agnostic*: corre en CPU hoy, y en **CUDA/GPU** sin cambiar
una línea (`cuda` se detecta automáticamente).

La ventaja decisiva frente a Chroma: **las fuerzas moleculares salen de
`autograd`** — escribes la acción y torch deriva el gradiente por ti, correcto al
orden de máquina (en Chroma se codifican a mano y es donde más errores se cuelan).

```
MPI/QMP/QDP++/Chroma (C++, décadas de HPC)   ← referencia y cross-check
vs
torch-lattice (PyTorch, autograd, GPU-ready) ← este proyecto
```

## Uso rápido

```bash
git clone https://github.com/yacm/torch-lattice.git
cd torch-lattice
# crea el entorno desde las versiones verificadas del lockfile
uv sync
source .venv/bin/activate

# correr toda la batería de validación
for t in tests/test_*.py; do PYTHONPATH=. python "$t" || exit 1; done

# ejemplo mínimo: plaqueto de un campo trivial
PYTHONPATH=. python -c "from tlattice import gauge; print(gauge.average_plaquette(gauge.unit_gauge((4,4,4,8))))"

# HMC corto
PYTHONPATH=. python - <<'EOF'
from tlattice import gauge, hmc
import torch
U = [hmc.make_leaf(u) for u in gauge.unit_gauge((4,4,4,8))]
for _ in range(20):
    U, acc, dH, plaq = hmc.hmc_step(U, 6.0, n_steps=20, eps=0.02)
print("plaqueto:", plaq)
EOF
```

## Estructura

```
tlattice/
  config.py       gamma matrices euclídeas, Nc, device
  su3.py          grupo SU(3): exp del álgebra, proyección, random_su3
  gauge.py        campo de gauge, plaqueto, acciones Wilson + Symanzik (tadpole)
  dirac.py        operador de Dirac de Wilson (Dslash), APBC en tiempo
  invert.py       gradiente conjugado + solve de Dirac (normales)
  hmc.py          HMC de gauge puro: fuerza por autograd, leapfrog, Metropolis
  observables.py  lazo de Polyakov, lazos de Wilson
  lime.py         lector/escritor LIME (bit-compatible con Chroma)
  fermion_hmc.py  HMC con fermiones dinámicos (2 sabores Wilson, exacto)
  stout.py        stout smearing (Morningstar-Peardon, 1 iter, rho)
  clover.py       fermiones clover (SW): tensor de campo F + Dslash clover
  clover_hmc.py   HMC clover (2 sabores) con stout + gauge Symanzik
  quark_observables.py  condensado quiral ⟨ψ̄ψ⟩ y correlador del pión
tests/            test_core, test_hmc, test_epsilon, test_invert, test_observables,
                  test_lime, test_fermion_hmc, test_quark_observables, test_batch,
                  test_clover, test_clover_hmc
```

## Qué está validado (y cómo)

| Qué | Prueba | Resultado |
|-----|--------|-----------|
| gamma matrices | hermíticas, γ²=I, anticommutan, γ5 | ✅ |
| SU(3) | unitariedad + det=1 | ✅ (~1e-6, piso float32) |
| plaqueto | gauge trivial = 1.0 exacto | ✅ |
| Dslash | **γ5-hermiticidad** < 1e-7 | ✅ |
| Dslash | matriz densa + `linalg.solve` | ✅ |
| CG | ‖Dx−b‖/‖b‖ ~ 3e-7, 45-66 iter | ✅ |
| **fuerza HMC** | **épsilon test: ΔH ∝ ε²** (cociente 1.10) | ✅ |
| HMC | conservación de energía (rel. 1.7e-4) | ✅ |
| HMC | aceptación 0.80-0.98, termaliza | ✅ |
| observables | Polyakov/Wilson loop = 1.0 (trivial) | ✅ |
| **LIME** | lector+escritor: plaqueto **idéntico a Chroma** (15 díg.) | ✅ |
| **fermiones** | fuerza fermiónica = derivada numérica (coc. 1.00000) | ✅ |
| **fermiones** | épsilon test ΔH ∝ ε² (HMC exacto) | ✅ |
| **fermiones** | termaliza (aceptación 1.00, plaqueto estable) | ✅ |
| **quarks** | condensado quiral ⟨ψ̄ψ⟩ finito y positivo | ✅ |
| **quarks** | correlador del pión C(t) decae | ✅ |
| **batch** | Dslash/CG/condensado batched == unbatched (err ~0) | ✅ |
| **stout** | unitario, suaviza (plaqueto ↑), punto fijo identidad | ✅ |
| **clover** | F_μν anti-hermitiana, hoja 4 cuadrantes (= mesField Chroma), F_12 = i sin(c)·λ₃ | ✅ |
| **clover** | Dslash γ5-hermítico, reduce a Wilson (c_sw=0) | ✅ |
| **clover** | fuerza (Symanzik + clover vía stout) = derivada numérica (coc. 1.00000) | ✅ |
| **clover** | épsilon test ΔH ∝ ε² (HMC exacto) | ✅ |
| **clover** | termaliza (plaqueto → 0.81, aceptación 1.00) | ✅ |

El **batch dimension** (dims líderes) es casi gratis: los `einsum` con `...`
ya hacen broadcast. Solo cambian el `torch.roll` (dims=offset+mu), el índice de
la frontera anti-periódica, y las normas del CG (suman sobre ejes de campo, no
batch). Ver `tlattice/dirac.py` (`_expand_to`), `invert.py` y `test_batch.py`.

El **épsilon test** es la prueba de oro: fija la longitud de trayectoria, baja ε
a la mitad, y verifica que ΔH se divide por 4 (integrador simpléctico de 2º
orden + fuerza correcta). Pasa con cociente max/min de 1.10 (ideal: 1.0).

## Action clover 2+1 (grupo HadStruc/NME) — núcleo hecho ✅

Implementado el action que usa el grupo **HadStruc** (JLab/W&M/LANL/MIT/Marseille)
para matrix elements, verificado contra su paper de los ensembles NME:

- **Gauge**: Symanzik tree-level tadpole-improved (Lüscher-Weisz), `c₀=5/3`,
  `c₁=−1/(12·u₀²)` (rectángulos 1×2). `tlattice/gauge.py::symanzik_action`.
- **Fermiones**: clover (Sheikholeslami-Wohlert), `c_sw = 1/u₀³` (tadpole
  tree-level), sobre enlaces **stout-smeared** (1 iter, ρ=0.1).
  `tlattice/stout.py` + `tlattice/clover.py`.
- **HMC clover de 2 sabores** (exacto): `tlattice/clover_hmc.py`.

La fuerza fluye por autograd a través de `U → stout(U) → D_clover → S`; el
smearing es diferenciable vía `matrix_exp`. Validado con fuerza = derivada
numérica (cociente 1.00000) y épsilon test ΔH ∝ ε² (cociente 1.02).

**Falta para el 2+1 completo**: RHMC (aproximación racional para el sabor
extraño), Hasenbusch (cocientes de determinantes para el ligero), e
integradores multi-escala de 4º orden. Es el siguiente paso natural.

## Cross-check contra Chroma (bit-compatible)

El núcleo clover está verificado contra una instalación vecina de Chroma en
`../lattice-qcd/` (programas de referencia en `progs/`):

| Cantidad | torch | Chroma | match |
|---|---|---|---|
| plaqueto (config 42) | 0.218134842687209 | 0.218134842687208 | 15 díg. |
| rectángulo 1×2 ⟨R⟩ | 0.046067140461826 | 0.046067140461826 | exacto |
| **S_Symanzik** (β=6, u₀=.86) | 20056.645409148048 | 20056.645409148048 | exacto |
| **Σ\|F_μν\|²** (clover) | 1212.5089991942629 | 1212.5089991942627 | 15 díg. |
| Σ\|Tr F_μν\|² | 168.5155474598206 | 168.5155474598206 | exacto |

La **hoja de clover** es la suma de las **4 plaquetas orientadas con base x**
(`U_μν + U_ν,−μ + U_−μ,−ν + U_−ν,μ`), igual que `mesField` de Chroma. Es
gauge-covariante (`Q → g(x)Q g(x)†`). `clover_field_strength(U)` devuelve el F
**crudo con traza** (como Chroma); `traceless=True` proyecta a su(3) si se
quiere la convención SW estándar.

### Dslash clover bit-compatible con Chroma

El operador completo `D_clover = D_wilson − (c_sw/4)Σ[γ_μ,γ_ν]F_μν` (con y sin
APBC) se verificó **componente a componente** contra `UnprecCloverLinOp` de
Chroma (`progs/clover_dslash_test.cc`): **max error relativo 1e-15, 0 de 12288
componentes difieren** (Mass=0.1, c_sw=1.0, 4³×8, seed 42; con c_sw=0 es exacto).
Para lograrlo hubo que alinear **dos convenciones** a Chroma:

1. **Base de gammas → DeGrand-Rossi** (γ₀ = `[[0,0,0,i],[0,0,i,0],[0,−i,0,0],[−i,0,0,0]]`,
   etc.), extraída de `qdpxx/sse_spin_proj.h`. Antes γ₀ era diag(1,1,−1,−1).
2. **t_dir = 3** (tiempo = eje 3, como Chroma; antes eje 0). El escritor LIME ya
   mapeaba eje 3→"lt", así que el código interno era el inconsistente.

Y se cazó un **bug real**: el `einsum` del Dslash aplicaba la matriz de color
**transpuesta** (`"...ab,...sa->...sb"` = Uᵀψ en vez de `"...ba,...sa->...sb"` = Uψ).
Era invisible a la γ5-hermiticidad y al épsilon test (ambos invariantes ante
U→Uᵀ); solo el cross-check componente a componente lo detectó (color0 coincidía,
color1/2 no). Los cross-checks de gauge (plaqueto/Symanzik/Σ|F|²) son invariantes
a traza/transposición y tampoco lo habrían pillado.

## Nota sobre GPU/CUDA

Esta máquina **no tiene GPU**, así que `torch` se instaló en su build CPU. El
código es 100% device-agnostic (`config.device()` decide). Para correr en un
clúster con GPU:

```bash
pip install --index-url https://download.pytorch.org/whl/cu124 torch   # build CUDA
# sin tocar una línea de tlattice/ — device() pasará a 'cuda' solo
```

## Límites honestos (qué NO es todavía)

- **No es un port completo de Chroma.** Falta el zoológico de fermiones
  (staggered, overlap, domain-wall), multigrid, y las optimizaciones HPC.
- **HMC tiene fermiones dinámicos** (2 sabores degenerados: Wilson y clover,
  ambos exactos). Falta: staggered/domain-wall, hasenbusch, multigrid,
  **RHMC para 1 sabor** (para el 2+1 completo del action HadStruc).
- **Sin MPI multi-nodo** (posible vía `torch.distributed`, no en esta caja).

## Cross-check bit-compatible con Chroma (hecho ✅)

`tlattice/lime.py` lee y escribe el formato LIME/SciDAC/ILDG de las
configuraciones de Chroma. Validado en **ambas direcciones** contra el binario
C++ `read_plaq.cc`:

- **Lector**: lee `dummy_run.lime1` de purgaug y da el plaqueto espacial
  `0.660514080740162`, idéntico al de Chroma (diferencia 0.0).
- **Escritor**: escribe una config generada en torch y Chroma la lee, dando el
  mismo plaqueto al último dígito.
- **Checksum SciDAC** byte-exacto (`suma`/`sumb` reproducidos).

Detalles del formato (big-endian, sitio lexicográfico, fila-mayor, CRC32 con
rotación por rango) en la skill `torch-lattice` → `references/lime-format.md`.

## Próximos pasos naturales

1. Medición física en ensamble: masa del pión y condensado sobre configs HMC.
2. Kernel de Dslash vectorizado / en GPU.
3. `torch.distributed` para multi-nodo.

## Física de referencia (para cross-check futuro)

El stack C++ instalado en `../lattice-qcd/` dejó resultados
verificados: purgaug con `w_plaq = 0.6979...` (β=6.0 anisotrópico ξ=2.464).
Nuestro plaqueto isotrópico a β=6 en 4⁴×8 da ~0.825 (volumen diminuto → plaqueto
cercano a 1, físicamente esperable).
