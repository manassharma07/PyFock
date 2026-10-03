"""
Tests of the DF_algo=12 Coulomb gradient (:mod:`pyfock.Integrals.df_algo12_grad`) and of its use by
:class:`pyfock.DFT_Grad`.

The reference is the exact three-center derivative contraction
:func:`pyfock.Integrals.rys_3c2e_grad_contract` (itself validated against PySCF and finite
differences).  The near-field kernel must reproduce it when nothing is far field; with the far
field it must agree to the accuracy of the multipole expansions, on a chain of water molecules with
a substantial far field; and the far-field terms must be the derivative of what algorithm 12
actually computes, which finite differences of ``sum_P c_P gamma_P(R)`` check directly.
"""
from __future__ import annotations

import numpy as np
import pytest

from pyfock import Basis, DFT, DFT_Grad, DFT_NumGrad, Integrals, Mol
from pyfock.Integrals import df_algo12_grad as grad12
from pyfock.Integrals import df_algo12_helpers as algo12
from pyfock.Integrals.schwarz_helpers import eri_4c2e_diag

WATER = [["O", 0.0, 0.0, 0.117], ["H", 0.0, 0.757, -0.467], ["H", 0.0, -0.757, -0.467]]


def _build(atoms, sao, basis_name="def2-SVP", aux_name="def2-universal-jfit"):
    mol = Mol(atoms=atoms)
    basis = Basis(mol, {"all": Basis.load(mol=mol, basis_name=basis_name)})
    aux = Basis(mol, {"all": Basis.load(mol=mol, basis_name=aux_name)})
    metric = Integrals.rys_2c2e_symm(aux)
    if sao:
        proj = aux.sph2cart_basis() @ aux.cart2sph_basis()
        metric = proj @ metric @ proj.T + 1e-12 * np.eye(aux.bfs_nao)
    sqrt4 = np.sqrt(np.abs(eri_4c2e_diag(basis)))
    sqrt2 = np.sqrt(np.abs(np.diag(metric)))
    return basis, aux, sqrt4, sqrt2


def _chain(sao, n=4, spacing=5.0, **kwargs):
    atoms = [[s, x + spacing * k, y, z] for k in range(n) for s, x, y, z in WATER]
    basis, aux, sqrt4, sqrt2 = _build(atoms, sao, **kwargs)
    S = Integrals.overlap_mat_symm(basis)
    dmat = S + 0.3 * S @ S
    dmat *= 10.0 * len(atoms) / 3.0 / np.trace(dmat @ S)
    coeff = np.random.default_rng(3).standard_normal(aux.bfs_nao) * 0.05
    if sao:
        # the coefficients DFT_Grad contracts with in SAO mode, c_eff = T^T c_sph
        coeff = aux.cart2sph_basis().T @ (aux.cart2sph_basis() @ coeff)
    return atoms, basis, aux, sqrt4, sqrt2, dmat, coeff


@pytest.mark.parametrize("sao", [False, True])
def test_near_field_kernel_matches_exact_contraction(sao):
    """Without a far field and without screening the kernel is the exact derivative contraction."""
    _, basis, aux, sqrt4, sqrt2, dmat, coeff = _chain(sao)
    exact = Integrals.rys_3c2e_grad_contract(basis, aux, dmat, coeff, threshold_schwarz=1e-16)
    plan = grad12.build_grad_plan(basis, aux, sqrt4, sqrt2, 1e-16, False, sao=sao, far_field=False)
    grad = grad12.grad_contract(plan, dmat, coeff, threshold_grad=0.0)
    # the only difference left is the plan's primitive-pair cut-off (exp(-18.42), as the SCF's)
    np.testing.assert_allclose(grad, exact, atol=1e-9, rtol=0)
    assert np.abs(grad.sum(axis=0)).max() < 1e-12


@pytest.mark.parametrize("sao", [False, True])
def test_far_field_gradient_matches_exact_contraction(sao):
    _, basis, aux, sqrt4, sqrt2, dmat, coeff = _chain(sao)
    exact = Integrals.rys_3c2e_grad_contract(basis, aux, dmat, coeff, threshold_schwarz=1e-16)
    plan = grad12.build_grad_plan(basis, aux, sqrt4, sqrt2, 1e-9, True, sao=sao)
    assert plan.fraction_far_field > 0.15
    grad = grad12.grad_contract(plan, dmat, coeff)
    near = grad12.grad_contract(plan, dmat, coeff, far=False)
    # the far-field terms are a real part of the gradient ...
    assert np.abs(grad - near).max() > 1e-3
    # ... and together with the near field reproduce the exact one to the multipole accuracy
    np.testing.assert_allclose(grad, exact, atol=2e-9, rtol=0)
    # both halves are translationally invariant on their own
    assert np.abs((grad - near).sum(axis=0)).max() < 1e-12
    assert np.abs(grad.sum(axis=0)).max() < 1e-12


def test_far_field_gradient_is_the_derivative_of_algorithm_12():
    """Finite differences of sum_P c_P gamma_P(R), gamma from algorithm-12 plans rebuilt at each geometry."""
    atoms, basis, aux, sqrt4, sqrt2, dmat, coeff = _chain(False)
    plan = grad12.build_grad_plan(basis, aux, sqrt4, sqrt2, 1e-9, True)
    grad = grad12.grad_contract(plan, dmat, coeff)
    h_ang = 1e-4
    h = h_ang * 1.8897261245650618
    for atom in (0, 4):
        for d in range(3):
            values = []
            for sign in (1, -1):
                displaced = [list(a) for a in atoms]
                displaced[atom][1 + d] += sign * h_ang
                b2, a2, q4, q2 = _build(displaced, False)
                p2 = algo12.build_plan(b2, a2, q4, q2, 1e-9, True, max_memory_gb=0)
                values.append(coeff @ algo12.gamma_from_plan(p2, dmat))
            # the finite-difference error itself is a few 1e-9 at this step; the far-field terms
            # are ~1e-2, so a wrong one could not hide below the tolerance
            assert abs((values[0] - values[1]) / (2 * h) - grad[atom, d]) < 3e-8


def test_tzvpd_far_field_gradient():
    """f functions and g auxiliary shells (def2-TZVPD / def2-universal-jkfit), SAO."""
    _, basis, aux, sqrt4, sqrt2, dmat, coeff = _chain(True, basis_name="def2-TZVPD",
                                                      aux_name="def2-universal-jkfit")
    exact = Integrals.rys_3c2e_grad_contract(basis, aux, dmat, coeff, threshold_schwarz=1e-16)
    plan = grad12.build_grad_plan(basis, aux, sqrt4, sqrt2, 1e-9, True, sao=True)
    assert plan.fraction_far_field > 0.15
    grad = grad12.grad_contract(plan, dmat, coeff)
    np.testing.assert_allclose(grad, exact, atol=2e-8, rtol=0)
    assert np.abs(grad.sum(axis=0)).max() < 1e-12


@pytest.mark.regression
@pytest.mark.parametrize("sao", [False, True])
def test_dft_grad_default_is_the_derivative_of_the_scf_energy(sao):
    """DFT_Grad's default (DF_algo=12, the fit reused from the SCF, the SCF's screening) against finite
    differences of the SCF energy on a fixed grid, which is what the XC gradient without grid
    response differentiates, so every other term is compared exactly."""
    atoms = [[s, x + 4.0 * k, y, z] for k in range(3) for s, x, y, z in WATER]
    mol = Mol(atoms=atoms)
    basis = Basis(mol, {"all": Basis.load(mol=mol, basis_name="def2-SVP")})
    aux = Basis(mol, {"all": Basis.load(mol=mol, basis_name="def2-universal-jfit")})
    dft = DFT(mol, basis, aux, xc="PBE", ncores=2)
    dft.sao = sao
    dft.conv_crit = 1e-11
    dft.scf()
    assert dft.df_fit is not None and dft.df_fit["dmat"] is dft.dmat
    default = DFT_Grad(dft, verbose=False)
    assert default.DF_algo == 12 and default._scf_fit_gamma() is not None and default._strict_pairs()
    g12 = default.calculate()
    assert "df_plan" in g12["timings"]
    # the fit redone from the plan gives the same gradient
    fit, dft.df_fit = dft.df_fit, None
    g12_refit = DFT_Grad(dft, verbose=False).calculate()
    dft.df_fit = fit
    np.testing.assert_allclose(g12_refit["gradient"], g12["gradient"], atol=1e-10, rtol=0)
    # the previous implementation differentiates a slightly different energy (no strict cut-off),
    # which on three waters is invisible at this tolerance
    g10 = DFT_Grad(dft, DF_algo=10, verbose=False).calculate()
    np.testing.assert_allclose(g12["gradient"], g10["gradient"], atol=1e-6, rtol=0)
    fd = DFT_NumGrad(dft, step_size=1e-3, step_unit="bohr", use_fixed_grids=True,
                     verbose=False).calculate(atom_indices=[3])["gradient"]
    np.testing.assert_allclose(g12["gradient"][3], fd[3], atol=1e-6, rtol=0)
