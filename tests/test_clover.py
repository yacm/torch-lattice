"""Validación del núcleo clover: stout, tensor de campo, Dslash clover, gauge Symanzik."""
import torch

from tlattice import clover, config, dirac, gauge, su3, stout

SHAPE = (2, 2, 2, 4)
DTYPE = torch.complex128
_DEVS = [torch.device("cpu")]


def _g5a(x):
    g5 = config.gamma5(x.dtype, x.device)
    return torch.einsum("ij,...ja->...ia", g5, x)


def main():
    torch.manual_seed(0)
    ok = True
    eye3 = torch.eye(3, dtype=DTYPE)

    # ---- 1. stout ----
    U_id = gauge.unit_gauge(SHAPE, dtype=DTYPE)
    Us = stout.stout_smear(U_id, rho=0.1)
    err = max((Us[mu] - U_id[mu]).abs().max().item() for mu in range(4))
    print(f"stout(identidad)=identidad: {err:.2e}")
    ok &= err < 1e-12

    U_rand = [su3.random_su3(*SHAPE, dtype=DTYPE) for _ in range(4)]
    Us = stout.stout_smear(U_rand, rho=0.1)
    uerr = max((Us[mu] @ Us[mu].mH - eye3).abs().max().item() for mu in range(4))
    print(f"stout unitario: {uerr:.2e}")
    ok &= uerr < 1e-10
    p0 = gauge.average_plaquette(U_rand)["w_plaq"]
    p1 = gauge.average_plaquette(Us)["w_plaq"]
    print(f"stout suaviza: plaq {p0:.4f} -> {p1:.4f}")
    ok &= p1 > p0

    # ---- 2. tensor clover ----
    F_id = clover.clover_field_strength(U_id)
    print(f"F(identidad)=0: {max(f.abs().max().item() for f in F_id):.2e}")
    ok &= max(f.abs().max().item() for f in F_id) < 1e-12
    F_r = clover.clover_field_strength(U_rand)  # raw, con traza (igual mesField)
    print(f"F anti-hermitiana: {max((f + f.mH).abs().max().item() for f in F_r):.2e}")
    ok &= max((f + f.mH).abs().max().item() for f in F_r) < 1e-10
    tr_raw = max(abs(torch.einsum("...ii->...", f)).max().item() for f in F_r)
    print(f"F raw con traza (tr != 0, como Chroma): {tr_raw:.2e}")
    ok &= tr_raw > 1e-3
    F_tl = clover.clover_field_strength(U_rand, traceless=True)
    print(f"F sin traza (traceless=True): {max(abs(torch.einsum('...ii->...', f)).max().item() for f in F_tl):.2e}")
    ok &= max(abs(torch.einsum("...ii->...", f)).max().item() for f in F_tl) < 1e-10

    # ---- 3. test de campo de fuerza no-abeliano (periódico): F_12 = i sin(c) lambda_3 ----
    shape_ab = (2, 6, 2, 2)
    c = 2.0 * torch.pi / shape_ab[1]  # periódico: e^{i c Nx} = 1
    U_ab = [eye3.expand(*shape_ab, 3, 3).clone() for _ in range(4)]
    x1 = torch.arange(shape_ab[1], dtype=torch.float64).reshape(1, shape_ab[1], 1, 1)
    # U_2(x) = diag(e^{i c x_1}, e^{-i c x_1}, 1): SU(3) NO abeliano, periódico en x_1
    diag = torch.stack([torch.exp(1j * c * x1), torch.exp(-1j * c * x1),
                        torch.ones_like(x1)], dim=-1)
    U_ab[2] = torch.diag_embed(diag)
    F_ab = clover.clover_field_strength(U_ab)
    F12 = F_ab[3]  # par (1,2)
    lam3 = torch.diag(torch.tensor([1.0, -1.0, 0.0], dtype=torch.float64))
    expected = 1j * torch.sin(torch.tensor(c, dtype=torch.float64)) * lam3
    err_ab = (F12 - expected).abs().max().item()  # constante en toda la retícula
    print(f"test no-abeliano F_12 = i sin(c) lambda_3: {err_ab:.2e}")
    ok &= err_ab < 1e-10
    other = max(F_ab[i].abs().max().item() for i in [0, 1, 2, 4, 5])
    print(f"otros F ~ 0: {other:.2e}")
    ok &= other < 1e-10

    # ---- 4. Dslash clover ----
    psi = torch.randn(*SHAPE, 4, 3, dtype=DTYPE)
    phi = torch.randn(*SHAPE, 4, 3, dtype=DTYPE)
    m0, c_sw = -0.5, 1.0
    red = (clover.clover_dslash(U_rand, psi, m0, 0.0)
           - dirac.wilson_dslash(U_rand, psi, m0)).abs().max().item()
    print(f"clover(c_sw=0)=Wilson: {red:.2e}")
    ok &= red < 1e-12
    lhs = (_g5a(clover.clover_dslash(U_rand, _g5a(psi), m0, c_sw)).conj() * phi).sum()
    rhs = (psi.conj() * clover.clover_dslash(U_rand, phi, m0, c_sw)).sum()
    g5err = (lhs - rhs).abs().item() / (rhs.abs().item() + 1e-30)
    print(f"clover gamma5-hermitico: err rel={g5err:.2e}")
    ok &= g5err < 1e-8

    # ---- 5. gauge Symanzik ----
    S_id = gauge.symanzik_action(U_id, 6.0, u0=0.86).item()
    print(f"Symanzik(identidad) = {S_id:.2e}")
    ok &= abs(S_id) < 1e-10
    S_rand = gauge.symanzik_action(U_rand, 6.0, u0=0.86).item()
    print(f"Symanzik(aleatorio, beta=6) = {S_rand:.3f} (debe ser > 0)")
    ok &= S_rand > 0

    print("\n" + ("=== TODO OK ===" if ok else "=== HAY FALLAS ==="))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
