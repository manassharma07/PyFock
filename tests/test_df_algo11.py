"""
Unit tests for the shell-blocked density-fitting Coulomb algorithm (``DF_algo=11``,
:mod:`pyfock.Integrals.df_algo11_helpers`).

Two water molecules 6 A apart give Schwarz-screened shell pairs; H2O/def2-TZVP
adds f functions in the orbital basis.  gamma and J from the shell-blocked plan
are compared with dense contractions over exactly the blocks the plan keeps
(tight), with the unscreened dense result (threshold-level), with algorithm 10,
and across memory budgets (bit-level agreement expected: same arithmetic).
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg

from pyfock import Basis, Integrals, Mol
from pyfock.Integrals import df_algo10_helpers as algo10
from pyfock.Integrals import df_algo11_helpers as algo11
from pyfock.Integrals.schwarz_helpers import eri_4c2e_diag


AUX_BASIS_NAME = "def2-universal-jfit"
WATER = [["O", 0.0, 0.0, 0.117], ["H", 0.0, 0.757, -0.467], ["H", 0.0, -0.757, -0.467]]


def _system(kind, basis_name, sao):
    if kind == "two_waters":
        atoms = list(WATER) + [[s, x + 6.0, y, z] for s, x, y, z in WATER]
    else:
        atoms = list(WATER)
    mol = Mol(atoms=atoms)
    basis = Basis(mol, {"all": Basis.load(mol=mol, basis_name=basis_name)})
    aux = Basis(mol, {"all": Basis.load(mol=mol, basis_name=AUX_BASIS_NAME)})
    metric = Integrals.rys_2c2e_symm(aux)
    proj = np.eye(aux.bfs_nao)
    if sao:
        proj = aux.sph2cart_basis() @ aux.cart2sph_basis()
        metric = proj @ metric @ proj.T + 1e-12 * np.eye(aux.bfs_nao)
    sqrt4 = np.sqrt(np.abs(eri_4c2e_diag(basis)))
    sqrt2 = np.sqrt(np.abs(np.diag(metric)))
    dense = np.einsum("rk,ijk->ijr", proj, Integrals.rys_3c2e_symm(basis, aux))
    S = Integrals.overlap_mat_symm(basis)
    dmat = 0.1 * (S + 0.3 * S @ S)  # smooth, physically scaled, symmetric
    return basis, aux, sqrt4, sqrt2, dense, metric, dmat


def _block_mask(plan, basis, aux):
    """(nao, nao, naux) mask of the (ij|P) the plan represents (both triangles)."""
    nao, naux = basis.bfs_nao, aux.bfs_nao
    mask = np.zeros((nao, nao, naux), dtype=bool)
    shell_of = np.asarray(basis.bfs_shell_index)
    aux_shell_of = np.asarray(aux.bfs_shell_index)
    sig_pairs = set(plan.work_iter.tolist())
    for p in range(plan.pair_I.shape[0]):
        if p not in sig_pairs:
            continue
        I, J = plan.pair_I[p], plan.pair_J[p]
        rows_i = np.nonzero(shell_of == I)[0]
        rows_j = np.nonzero(shell_of == J)[0]
        cols = np.zeros(naux, dtype=bool)
        for K in range(aux.nshells):
            if plan.Q_pair[p] * plan.Q_aux[K] > plan.threshold:
                cols |= aux_shell_of == K
        for i in rows_i:
            for j in rows_j:
                if plan.strict_schwarz and plan.sqrt_ints4c2e_diag[i, j] ** 2 < algo10.STRICT_PAIR_CUTOFF:
                    continue
                mask[i, j, cols] = True
                mask[j, i, cols] = True
    return mask


@pytest.mark.parametrize("sao", [False, True])
@pytest.mark.parametrize("strict", [False, True])
def test_algo11_matches_dense_two_waters(sao, strict):
    basis, aux, sqrt4, sqrt2, dense, metric, dmat = _system("two_waters", "def2-SVP", sao)
    nao, naux = basis.bfs_nao, aux.bfs_nao
    threshold = 1e-9
    plan = algo11.build_plan(basis, aux, sqrt4, sqrt2, threshold, strict, sao=sao)
    assert plan.n_pairs_significant < plan.n_pairs_total  # screening active
    assert plan.fraction_cached == 1.0

    mask = _block_mask(plan, basis, aux)
    kept = dense * mask
    gamma = algo11.gamma_from_plan(plan, dmat)
    np.testing.assert_allclose(gamma, np.einsum("ijP,ij->P", kept, dmat), atol=1e-10, rtol=1e-12)
    coeff = np.linspace(-1.0, 1.0, naux)
    J = algo11.J_from_plan(plan, coeff)
    assert np.allclose(J, J.T, atol=1e-14)
    np.testing.assert_allclose(J, np.einsum("ijP,P->ij", kept, coeff), atol=1e-10, rtol=1e-12)
    # against the unscreened dense contraction the error is bounded by the screening
    if not strict:
        assert np.abs(gamma - np.einsum("ijP,ij->P", dense, dmat)).max() < 1e-7
        assert np.abs(J - np.einsum("ijP,P->ij", dense, coeff)).max() < 1e-7


def test_algo11_f_functions_h2o_tzvp():
    basis, aux, sqrt4, sqrt2, dense, metric, dmat = _system("h2o", "def2-TZVP", False)
    naux = aux.bfs_nao
    plan = algo11.build_plan(basis, aux, sqrt4, sqrt2, 1e-9, False)
    gamma = algo11.gamma_from_plan(plan, dmat)
    np.testing.assert_allclose(gamma, np.einsum("ijP,ij->P", dense, dmat), atol=1e-10, rtol=1e-12)
    coeff = np.cos(np.arange(naux, dtype=float))
    np.testing.assert_allclose(algo11.J_from_plan(plan, coeff), np.einsum("ijP,P->ij", dense, coeff), atol=1e-10, rtol=1e-12)


@pytest.mark.parametrize("sao", [False, True])
def test_algo11_memory_budgets_agree(sao):
    basis, aux, sqrt4, sqrt2, dense, metric, dmat = _system("two_waters", "def2-SVP", sao)
    naux = aux.bfs_nao
    full = algo11.build_plan(basis, aux, sqrt4, sqrt2, 1e-9, True, sao=sao)
    half = algo11.build_plan(basis, aux, sqrt4, sqrt2, 1e-9, True, sao=sao, max_memory_gb=0.5 * full.memory_gb)
    none = algo11.build_plan(basis, aux, sqrt4, sqrt2, 1e-9, True, sao=sao, max_memory_gb=0.0)
    assert none.n_pairs_cached == 0 and none.values.size == 0
    assert 0 < half.n_pairs_cached < full.n_pairs_cached
    assert half.memory_gb <= 0.5 * full.memory_gb
    g_full = algo11.gamma_from_plan(full, dmat)
    coeff = np.sin(np.arange(naux, dtype=float))
    J_full = algo11.J_from_plan(full, coeff)
    for plan in (half, none):
        np.testing.assert_allclose(algo11.gamma_from_plan(plan, dmat), g_full, atol=1e-13, rtol=1e-13)
        np.testing.assert_allclose(algo11.J_from_plan(plan, coeff), J_full, atol=1e-13, rtol=1e-13)


@pytest.mark.parametrize("sao", [False, True])
def test_algo11_full_pipeline_agrees_with_algo10(sao):
    """gamma -> Cholesky solve -> J with both algorithms; they screen at different granularity."""
    basis, aux, sqrt4, sqrt2, dense, metric, dmat = _system("two_waters", "def2-SVP", sao)
    nao, naux = basis.bfs_nao, aux.bfs_nao
    threshold, strict = 1e-9, False
    cho = scipy.linalg.cho_factor(metric)
    # algorithm 11
    plan = algo11.build_plan(basis, aux, sqrt4, sqrt2, threshold, strict, sao=sao)
    g11 = algo11.gamma_from_plan(plan, dmat)
    c11 = scipy.linalg.cho_solve(cho, g11)
    J11 = algo11.J_from_plan(plan, c11)
    # algorithm 10 (shell-constant bounds in SAO mode)
    bounds = algo10.aux_shell_max_bounds(sqrt2, aux) if sao else sqrt2
    iA, iB = np.tril_indices(nao)
    offsets, nsig = algo10.calc_offsets_3c2e_schwarz(sqrt4, bounds, threshold, strict, iA, iB)
    vals = algo10.rys_3c2e_tri_schwarz_sparse_algo10(basis, aux, iA, iB, offsets, sqrt4, bounds, threshold, strict, nsig, sao=sao)
    dmat_tri = (2.0 * dmat - np.diag(np.diag(dmat)))[iA, iB]
    g10 = algo10.df_coeff_calculator_algo10(vals, dmat_tri, iA, iB, offsets, naux, sqrt4, bounds, threshold, strict, ncores=2)
    c10 = scipy.linalg.cho_solve(cho, g10)
    J_tri = algo10.J_tri_calculator_algo10(vals, c10, iA, iB, offsets, nao * (nao + 1) // 2, sqrt4, bounds, threshold, strict)
    J10 = np.zeros((nao, nao)); J10[iA, iB] = J_tri; J10 = J10 + J10.T - np.diag(np.diag(J10))
    # dense reference
    g_ref = np.einsum("ijP,ij->P", dense, dmat)
    c_ref = scipy.linalg.cho_solve(cho, g_ref)
    J_ref = np.einsum("ijP,P->ij", dense, c_ref)
    assert np.abs(J11 - J_ref).max() < 1e-7 and np.abs(J10 - J_ref).max() < 1e-7
    assert np.abs(J11 - J10).max() < 1e-7
    assert abs(c11 @ g11 - c_ref @ g_ref) < 1e-7
