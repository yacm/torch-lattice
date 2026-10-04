"""Validación del núcleo (M1-M3): gamma matrices, SU(3), plaqueto, Dslash."""
import torch

from tlattice import config, su3, gauge, dirac


def check(name, cond):
    print(f"  [{'OK' if cond else 'FALLA'}] {name}")
    return cond


def main():
    dev = config.device()
    print(f"device: {dev} | torch {torch.__version__}")
    ok = True

    # ---- M0: gamma matrices ----
    print("\n== gamma matrices (euclídeas, base Dirac) ==")
    gs = config.gammas()
    eye4 = torch.eye(4, dtype=torch.complex64)
    hermit = all(torch.allclose(g.mH, g, atol=1e-6) for g in gs)
    ok &= check("todas hermíticas", hermit)
    sq = all(torch.allclose(g @ g, eye4, atol=1e-6) for g in gs)
    ok &= check("gamma_mu^2 = I", sq)
    anti = True
    for mu in range(4):
        for nu in range(mu + 1, 4):
            anti = anti and torch.allclose(gs[mu] @ gs[nu] + gs[nu] @ gs[mu],
                                           torch.zeros(4, 4, dtype=torch.complex64), atol=1e-6)
    ok &= check("anticommutan (mu != nu)", anti)
    g5 = config.gamma5()
    ok &= check("gamma5 hermítica", torch.allclose(g5.mH, g5, atol=1e-6))
    ok &= check("gamma5^2 = I", torch.allclose(g5 @ g5, eye4, atol=1e-6))
    for mu in range(4):
        ok &= check(f"{{gamma5, gamma{mu}}} = 0",
                    torch.allclose(g5 @ gs[mu] + gs[mu] @ g5,
                                   torch.zeros(4, 4, dtype=torch.complex64), atol=1e-6))

    # ---- M1: SU(3) ----
    print("\n== SU(3) ==")
    R = su3.random_su3(4, 4, 4, 8)
    ok &= check("random_su3 unitaria", su3.is_su3(R))
    # proyección: tomamos una matriz general y la proyectamos
    G = torch.randn(4, 4, 4, 8, 3, 3, dtype=torch.complex64)
    P = su3.project_su3(G)
    ok &= check("project_su3 -> SU(3)", su3.is_su3(P))

    # ---- M2: plaqueto y acción ----
    print("\n== plaqueto y acción de Wilson ==")
    shape = (4, 4, 4, 8)
    U = gauge.unit_gauge(shape)
    pl = gauge.average_plaquette(U)
    ok &= check("unit gauge: w_plaq == 1.0", abs(pl["w_plaq"] - 1.0) < 1e-6)
    S = gauge.wilson_action(U, beta=6.0)
    ok &= check("unit gauge: acción == 0", abs(S.item()) < 1e-6)
    # gauge aleatorio -> plaqueto cerca de 0 (desordenado)
    U_r = gauge.random_gauge(shape)
    pl_r = gauge.average_plaquette(U_r)
    print(f"     plaqueto gauge aleatorio (heat-kernel, no uniforme): {pl_r['w_plaq']:.4f}")
    # random_su3 usa coeficientes gaussianos => distribución tipo heat-kernel
    # centrada en la identidad, por eso el plaqueto queda ~0.2 y no ~0 (Haar).
    ok &= check("gauge aleatorio: plaqueto claramente < 1 (desordenado)",
                0.05 < pl_r["w_plaq"] < 0.5)

    # ---- M3: Dslash ----
    print("\n== Dirac de Wilson (Dslash) ==")
    m0 = -0.5  # masa desnuda (positiva física para kappa ~ 0.14)
    err_g5 = dirac.gamma5_hermiticity_error(U, m0, tol_sample=2)
    print(f"     error de gamma5-hermiticidad: {err_g5:.3e}")
    ok &= check("gamma5-hermiticidad < 1e-4", err_g5 < 1e-4)

    # matriz densa (retícula chiquita) y comparación con linalg.solve
    small = (2, 2, 2, 4)
    Us = gauge.unit_gauge(small)
    D = dirac.dense_matrix(Us, m0)
    b = torch.randn(D.shape[0], dtype=torch.complex64)
    x_solve = torch.linalg.solve(D, b)
    resid = (D @ x_solve - b).norm() / b.norm()
    print(f"     residuo de linalg.solve sobre D densa: {resid.item():.3e}")
    ok &= check("D densa no singular, solve limpio", resid.item() < 1e-5)

    print("\n" + ("=== TODO OK ===" if ok else "=== HAY FALLAS ==="))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
