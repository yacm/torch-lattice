# Torch-Lattice — un "mini-Chroma" nativo en PyTorch

**Carpeta:** raíz del repositorio clonado
**Objetivo:** reconstruir el núcleo de física de retícula (QCD de Wilson) en PyTorch,
con código *device-agnostic* (corre en CPU hoy, en CUDA/GPU mañana sin tocar una línea).

---

## 1. Por qué torch (opinión)

La ganancia más grande no es la GPU en sí: es **autograd**.

En Chroma, la fuerza del gauge (la derivada de la acción respecto al campo de gauge)
y la fuerza fermiónica se codifican A MANO — es la parte más propensa a errores de todo
el paquete. En torch, escribes la acción `S[U]` y `torch.autograd.grad(S, U)` te da la
fuerza gratis, y correcta al orden de máquina. Esa es la ventaja decisiva.

**Límites honestos:**
- Esta máquina **no tiene GPU**. El código se escribe para CUDA, pero se valida en CPU
  aquí y se mueve a un clúster con GPU sin cambios (`device` se detecta solo).
- `torch.complex64` es maduro pero menos "de combate" que los reales; hay que probar cada
  operación (sobre todo `torch.linalg` con complejos).
- Esto NO es un port línea-por-línea de Chroma. Es un **núcleo torch-nativo** con la
  física esencial. Chroma trae décadas de optimización HPC y todo el zoológico de
  fermiones (overlap, domain-wall, staggered, multigrid) — eso queda fuera de alcance.

## 2. Alcance

**Dentro (el núcleo):**
- Grupo SU(3): exponencial del álgebra de Lie, proyección al grupo.
- Campo de gauge U (retícula 4D × 4 direcciones × matrices 3×3 complejas).
- Acción de gauge de Wilson + plaqueto (con convención de Chroma).
- Operador de Dirac de Wilson (Dslash) con condiciones de contorno anti-periódicas en tiempo.
- Inversor de gradiente conjugado (CG).
- HMC: trayectoria de dinámica molecular con fuerzas por autograd + Metropolis.
- Observables: plaqueto, lazo de Polyakov, lazos de Wilson.
- Harness de validación numérica.

**Fuera de alcance (v1):**
- Fermiones mejorados (clover, HISQ), overlap/domain-wall, staggered.
- Multigrid / solvers avanzados.
- MPI multi-nodo (torch.distributed es posible, pero no en esta caja).
- QUDA-style optimizaciones de kernel.

## 3. Decisiones de diseño

1. **Device-agnostic:** `DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")`.
   Todo se crea en `DEV`. En CPU corre igual (lento, pero correcto).
2. **Precisión:** `torch.complex64` para campos (igual que Chroma usa `double` en CPU,
   pero complex64 basta para validación y es lo natural en GPU).
3. **Layout del gauge:** `U[mu]` con forma `(Nt, Nz, Ny, Nx, 3, 3)`, `mu = 0..3`.
   Un diccionario/tupla `multi1d` de 4 tensores.
4. **Fermión:** `psi` con forma `(Nt, Nz, Ny, Nx, 4, 3)` (espín × color).
5. **Shifts:** `torch.roll` para el corrimiento periódico; multiplicar por −1 el enlace
   que cruza la frontera temporal (APBC), como hace `shift()` de QDP++.
6. **Convenciones = Chroma** (para poder validar contra el stack C++ ya instalado):
   - Plaqueto por sitio: `w_plaq = (s_plaq + t_plaq)/2`, promedio sobre los 6 planos.
   - `s_plaq` = planos espaciales {(1,2),(1,3),(2,3)}, `t_plaq` = {(0,1),(0,2),(0,3)}.
   - Acción de gauge: `S_g = β · Σ_plaq (1 − (1/Nc) Re Tr U_plaq)`, `β = 6/g²`.

## 4. Arquitectura de módulos

```
torch-lattice/
├── PLAN.md                 ← este documento
├── README.md               ← cómo correrlo (se escribe al final)
├── .venv/                  ← entorno aislado con torch (CPU)
├── tlattice/
│   ├── __init__.py
│   ├── config.py           ← device, constantes (Nc=3, γ matrices)
│   ├── su3.py              ← álgebra de Lie SU(3): exp, proyección, unitarización
│   ├── gauge.py            ← campo de gauge, plaqueto, acción de Wilson
│   ├── dirac.py            ← operador de Dirac de Wilson (Dslash) + γ5-hermiticidad
│   ├── invert.py           ← gradiente conjugado
│   ├── hmc.py              ← HMC (leapfrog + Metropolis), fuerzas por autograd
│   └── observables.py      ← plaqueto, Polyakov, lazos de Wilson
├── tests/
│   ├── test_su3.py
│   ├── test_gauge.py
│   ├── test_dirac.py
│   ├── test_invert.py
│   └── test_hmc.py
├── scripts/
│   └── validate_vs_chroma.py   ← cross-check contra el stack C++ (read_plaq)
└── runs/                   ← salidas de HMC
```

## 5. Convenciones físicas exactas

**Plaqueto** (en el punto x, plano μν):
```
U_plaq(x;μ,ν) = U_μ(x) · U_ν(x+μ̂) · U_μ†(x+ν̂) · U_ν†(x)
```
Plaqueto por sitio = `(1/Nc) Re Tr U_plaq`, promediado sobre los 6 planos y sobre sitios.

**Operador de Dirac de Wilson** (masa m₀, parámetro de salto κ = 1/(8+2m₀)):
```
D_w[U] = (4 + m₀)·δ_{x,x'} − (1/2)·Σ_μ [ (1−γ_μ)·U_μ(x)·δ_{x+μ̂,x'} + (1+γ_μ)·U_μ†(x−μ̂)·δ_{x−μ̂,x'} ]
```
Propiedad a verificar: `γ₅ D γ₅ = D†` (γ5-hermiticidad), y `D` tiene espectro positivo para
masas positivas.

## 6. Estrategia de validación (lo que prueba que NO está mal)

Orden de verdades que cada milestone debe cumplir:

1. **Unit gauge → plaqueto = 1.0 exacto.** (U=I ⇒ cada plaqueto = I ⇒ Re Tr = 3).
2. **Acción de Wilson** de un campo unitario = 0.
3. **Dslash:** `γ₅ D γ₅ == D†` (γ5-hermiticidad, a precisión float).
4. **CG:** `‖D x − b‖ → 0` para una fuente puntual; comparar contra `torch.linalg.solve` en retícula chica.
5. **HMC:** con β grande (g² pequeña) el plaqueto queda ≈1 y la tasa de aceptación es alta;
   el campo debe respetar unitariedad de SU(3) en cada paso.
6. **Cross-check vs C++ (oro):** generar una config SU(3) en torch, escribirla, leerla con
   `read_plaq.cc` (QDP++), y comparar el plaqueto. Si coinciden a ~1e-6, el gauge y el
   plaqueto son bit-compatibles con Chroma.

## 7. Milestones

| # | Qué | Criterio de éxito |
|---|-----|-------------------|
| M0 | Entorno: `.venv` + torch CPU + smoke test | `torch` importa, `cuda.is_available()` reporta estado |
| M1 | `su3.py` | `expm(antiherm)` da matrices unitarias (‖U U† − I‖ < 1e-6) |
| M2 | `gauge.py` (plaqueto + acción) | unit gauge ⇒ plaqueto 1.0 y acción 0 |
| M3 | `dirac.py` (Dslash) | γ5-hermiticidad a < 1e-5 |
| M4 | `invert.py` (CG) | residuo < 1e-10; coincide con `linalg.solve` |
| M5 | `hmc.py` | aceptación > 0.7 en β alta; plaqueto ≈ 1 y estable |
| M6 | `observables.py` | Polyakov/Wilson loops definidos y finitos |
| M7 | cross-check vs `read_plaq.cc` | plaqueto coincide a ~1e-6 |
| M8 | README + limpieza | documentación clara |

## 8. Entorno

- **Python:** venv dedicado `.venv/` (python 3.11).
- **torch:** build CPU (`pip install torch --index-url https://download.pytorch.org/whl/cpu`)
  porque no hay GPU aquí. El código es CUDA-ready; en una máquina con GPU basta reinstalar
  torch con CUDA y `DEV` pasa a `cuda` solo.
- **numpy** ya está; solo se usa para I/O de validación.
