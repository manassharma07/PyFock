"""
Analytical gradients with RI exact exchange (``xc='HF'`` and global hybrids, ``DF_algo=11``).

* The row-weight mode of the near-field derivative kernel reproduces the Coulomb mode for
  rank-one weights ``D_ij c_P`` (Cartesian and spherical fit space, strict Schwarz on and off).
* The two-center derivative contraction with a general weight matrix matches finite differences.
* The exchange term alone matches finite differences of the RI exchange energy at fixed density.
* Complete HF, PBE0 and B3LYP gradients of H2O match PySCF's density-fitted gradients, and the fused
  Coulomb + exchange pass equals the separate one.
"""
from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest
import scipy.linalg

from pyfock import Basis, Data, DFT, DFT_Grad, Integrals, Mol
from pyfock.DFT_Helper_Coulomb import _pseudo_cartesian_metric_diagonal
from pyfock.Integrals import df_algo11_exchange as algo11x
from pyfock.Integrals import df_algo12_grad as grad12
from pyfock.Integrals.df_algo10_helpers import STRICT_PAIR_CUTOFF
from pyfock.Integrals.schwarz_helpers import eri_4c2e_diag


BASIS, AUX = "def2-SVP", "def2-universal-jkfit"
WATER = [["O", 0.0, 0.05, 0.117], ["H", 0.02, 0.757, -0.467], ["H", 0.0, -0.787, -0.427]]


def _bases(atoms):
    mol = Mol(atoms=[list(a) for a in atoms])
    basis = Basis(mol, {"all": Basis.load(mol=mol, basis_name=BASIS)})
    aux = Basis(mol, {"all": Basis.load(mol=mol, basis_name=AUX)})
    return mol, basis, aux


def _displaced(atoms, iatom, d, step_bohr):
    out = [list(a) for a in atoms]
    out[iatom][1 + d] += step_bohr / Data.Angs2BohrFactor
    return out


@pytest.mark.parametrize("sao", [False, True])
@pytest.mark.parametrize("strict", [False, True])
def test_row_weights_reproduce_coulomb_kernel(sao, strict):
    atoms = WATER + [[s, x + 3.0, y + 0.5, z] for s, x, y, z in WATER]
    _, basis, aux = _bases(atoms)
    sqrt4 = np.sqrt(np.abs(eri_4c2e_diag(basis)))
    V = Integrals.rys_2c2e_symm(aux)
    S = Integrals.overlap_mat_symm(basis)
    D = 0.1 * (S + 0.3 * S @ S)
    nao = basis.bfs_nao
    rng = np.random.default_rng(1)
    if sao:
        metric_diag = _pseudo_cartesian_metric_diagonal(aux, aux.cart2sph_operator_blockwise(V), aux.sph2cart_basis()) + 1e-12
        T = aux.cart2sph_basis()
        c_fit = rng.standard_normal(T.shape[0])
        c_cart = T.T @ c_fit
        tables = algo11x._cart2sph_tables(aux)[:4]
    else:
        metric_diag = np.diag(V)
        c_fit = c_cart = rng.standard_normal(aux.bfs_nao)
        tables = None
    plan = grad12.build_grad_plan(basis, aux, sqrt4, np.sqrt(np.abs(metric_diag)), 1e-9, strict, sao=sao,
                                  far_field=False)
    g_ref = grad12.grad_contract(plan, D, c_cart, threshold_grad=1e-30)
    mu, nu = np.tril_indices(nao)
    if strict:
        keep = sqrt4[mu, nu] ** 2 >= STRICT_PAIR_CUTOFF
        assert not keep.all()
        mu, nu = mu[keep], nu[keep]
    row_of = np.full((nao, nao), -1, dtype=np.int64)
    row_of[mu, nu] = np.arange(mu.size)
    g = grad12.grad_contract_rows(plan, D[mu, nu][:, None] * c_fit[None, :], row_of, tables, threshold_grad=1e-30)
    assert np.abs(g - g_ref).max() < 1e-12 * np.abs(g_ref).max()


def test_two_center_general_weights_match_finite_differences():
    _, _, aux = _bases(WATER)
    rng = np.random.default_rng(3)
    c = rng.standard_normal(aux.bfs_nao)
    g_c = Integrals.rys_2c2e_grad_contract(aux, c)
    np.testing.assert_allclose(Integrals.rys_2c2e_grad_contract(aux, weights=np.outer(c, c)), g_c, atol=1e-12, rtol=0)
    A = rng.standard_normal((aux.bfs_nao, aux.bfs_nao))
    W = A + A.T
    g = Integrals.rys_2c2e_grad_contract(aux, weights=W)
    assert np.abs(g.sum(axis=0)).max() < 1e-10
    h = 1e-4
    for iatom, d in [(0, 2), (1, 0), (2, 1)]:
        e = [np.sum(W * Integrals.rys_2c2e_symm(_bases(_displaced(WATER, iatom, d, s * h))[2])) for s in (1, -1)]
        assert abs(g[iatom, d] - (e[0] - e[1]) / (2 * h)) < 1e-8 * np.abs(g).max()


def _converged(xc, sao, strict=False, conv=1e-11, pyscf_grids=False, atoms=WATER):
    mol, basis, aux = _bases(atoms)
    dft = DFT(mol, basis, aux, xc=xc, conv_crit=conv, ncores=2, gridsLevel=3, use_pyscf_grids=pyscf_grids)
    dft.max_itr = 80
    dft.sao = sao
    dft.strict_schwarz = strict
    dft.threshold_schwarz = 1e-12
    with contextlib.redirect_stdout(io.StringIO()):
        energy, _ = dft.scf()
    assert dft.converged
    return float(energy), dft


def _exchange_energy(atoms, D, sao):
    """-(1/4) sum D_ij D_kl (ik|P) [(P|Q)^-1] (Q|jl) from dense integrals."""
    _, basis, aux = _bases(atoms)
    V = Integrals.rys_2c2e_symm(aux)
    I3 = Integrals.rys_3c2e_symm(basis, aux)
    if sao:
        T = aux.cart2sph_basis()
        V = T @ V @ T.T
        I3 = np.einsum("ijP,QP->ijQ", I3, T)
    nao = basis.bfs_nao
    L = scipy.linalg.cholesky(V, lower=True)
    B = scipy.linalg.solve_triangular(L, I3.reshape(nao * nao, -1).T, lower=True).T.reshape(nao, nao, -1)
    return -0.25 * np.einsum("ikP,kl,jlP,ij->", B, D, B, D, optimize=True)


@pytest.mark.parametrize("sao", [False, True])
def test_exchange_term_matches_finite_differences_at_fixed_density(sao):
    _, dft = _converged("HF", sao, conv=1e-9)
    with contextlib.redirect_stdout(io.StringIO()):
        res = DFT_Grad(dft, threshold_schwarz_grad=1e-16, separate_exchange=True).calculate()
    gK = res["gradient_components"]["exchange_df"]
    assert "coulomb_exchange_df" not in res["gradient_components"]
    assert np.abs(gK.sum(axis=0)).max() < 1e-10
    h = 1e-4
    for iatom, d in [(0, 1), (1, 2), (2, 0)]:
        e = [_exchange_energy(_displaced(WATER, iatom, d, s * h), dft.dmat, sao) for s in (1, -1)]
        assert abs(gK[iatom, d] - (e[0] - e[1]) / (2 * h)) < 2e-8


def _pyscf_gradient(xc):
    pytest.importorskip("pyscf")
    from pyscf import dft as pyscf_dft, gto, scf
    mol = gto.Mole()
    mol.atom = [(s, (x, y, z)) for s, x, y, z in WATER]
    mol.basis = BASIS
    mol.verbose = 0
    mol.build()
    if xc == "HF":
        mf = scf.RHF(mol).density_fit(auxbasis=AUX)
    else:
        mf = pyscf_dft.RKS(mol).density_fit(auxbasis=AUX)
        mf.xc = xc
        mf.grids.level = 3
    mf.conv_tol = 1e-13
    mf.conv_tol_grad = 1e-8
    energy = mf.kernel()
    return energy, mf.nuc_grad_method().kernel()


@pytest.mark.parametrize("xc", ["HF", "PBE0", "B3LYP"])
def test_gradient_matches_pyscf(xc):
    e_ref, g_ref = _pyscf_gradient(xc)
    # the SCF criterion is the energy change; 1e-13 is at the round-off of these energies
    energy, dft = _converged(xc, sao=True, conv=1e-11 if xc == "HF" else 1e-12, pyscf_grids=True)
    assert abs(energy - e_ref) < 2e-9
    with contextlib.redirect_stdout(io.StringIO()):
        res = DFT_Grad(dft).calculate()
    assert "coulomb_exchange_df" in res["gradient_components"]
    assert np.abs(res["gradient"] - g_ref).max() < (5e-8 if xc == "HF" else 1e-7)
    if xc == "HF":
        assert not res["gradient_components"]["xc"].any()
        assert np.abs(res["gradient"].sum(axis=0)).max() < 1e-10


def test_fused_pass_equals_separate_passes_with_strict_screening():
    atoms = WATER + [[s, x + 4.0, y, z + 1.0] for s, x, y, z in WATER]
    _, dft = _converged("PBE0", sao=False, strict=True, conv=1e-9, atoms=atoms)
    with contextlib.redirect_stdout(io.StringIO()):
        fused = DFT_Grad(dft, threshold_schwarz_grad=1e-16).calculate()
        separate = DFT_Grad(dft, threshold_schwarz_grad=1e-16, separate_exchange=True).calculate()
    comp = separate["gradient_components"]
    np.testing.assert_allclose(fused["gradient_components"]["coulomb_exchange_df"],
                               comp["coulomb_df"] + comp["exchange_df"], atol=1e-11, rtol=0)
    np.testing.assert_allclose(fused["gradient"], separate["gradient"], atol=1e-11, rtol=0)


def test_exact_exchange_needs_df_algo_11():
    _, dft = _converged("HF", sao=True, conv=1e-7)
    with pytest.raises(ValueError, match="DF_algo=11"):
        DFT_Grad(dft, DF_algo=12)


def test_ase_calculator_uses_ri_exchange_with_analytical_forces(tmp_path):
    pytest.importorskip("ase")
    from ase import Atoms
    from pyfock import PyFockCalculator

    calc = PyFockCalculator(xc="HF", basis=BASIS, directory=str(tmp_path / "hf"), run_in_process=True,
                            conv_crit=1e-9, ncores=2)
    assert calc._prepare_runtime_options() == {"xc": "HF", "conv_crit": 1e-9, "ncores": 2}
    assert calc._auxbasis_name(calc._prepare_runtime_options()) == AUX
    assert PyFockCalculator(xc="PBE")._auxbasis_name({"xc": "PBE"}) == "def2-universal-jfit"
    assert PyFockCalculator(xc="HF", auxbasis="def2-universal-jfit")._auxbasis_name({"xc": "HF"}) == "def2-universal-jfit"
    atoms = Atoms("OH2", positions=[a[1:] for a in WATER])
    atoms.calc = calc
    forces = atoms.get_forces()
    assert calc.pyfock_results["force_method_used"] == "analytical"
    mol, basis, aux = _bases(WATER)
    dft = DFT(mol, basis, aux, xc="HF", conv_crit=1e-9, ncores=2)
    with contextlib.redirect_stdout(io.StringIO()):
        dft.scf()
        ref = DFT_Grad(dft).calculate()["forces"] * Data.au2eVFactor / Data.Bohr2AngsFactor
    np.testing.assert_allclose(forces, ref, atol=1e-5, rtol=0)
