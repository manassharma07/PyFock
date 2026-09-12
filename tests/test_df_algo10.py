"""
Unit tests for the default density-fitting Coulomb algorithm (``DF_algo=10``)
helpers in :mod:`pyfock.Integrals.df_algo10_helpers`.

Two water molecules 6 A apart are used so that many AO pairs are Schwarz
screened and, crucially, some pairs keep only *part* of an auxiliary shell.
Every stored value is compared with a dense reference computed by
:func:`pyfock.Integrals.rys_3c2e_symm` (projected onto the spherical auxiliary
subspace in the SAO case), and the gamma/J contractions are compared with dense
contractions over the same stored set.
"""
from __future__ import annotations

import numpy as np
import pytest

from pyfock import Basis, Integrals, Mol
from pyfock.Integrals import df_algo10_helpers as algo10
from pyfock.Integrals.schwarz_helpers import eri_4c2e_diag


AUX_BASIS_NAME = "def2-universal-jfit"
WATER = [["O", 0.0, 0.0, 0.117], ["H", 0.0, 0.757, -0.467], ["H", 0.0, -0.757, -0.467]]


def _two_waters(separation=6.0):
    atoms = list(WATER) + [[symbol, x + separation, y, z] for symbol, x, y, z in WATER]
    return Mol(atoms=atoms)


def _setup(basis_name, sao):
    mol = _two_waters()
    basis = Basis(mol, {"all": Basis.load(mol=mol, basis_name=basis_name)})
    aux = Basis(mol, {"all": Basis.load(mol=mol, basis_name=AUX_BASIS_NAME)})
    ints2c2e = Integrals.rys_2c2e_symm(aux)
    proj = np.eye(aux.bfs_nao)
    if sao:
        # pseudo-Cartesian representation of the spherical aux basis, as used by the DFT driver
        proj = aux.sph2cart_basis() @ aux.cart2sph_basis()
        ints2c2e = proj @ ints2c2e @ proj.T + 1e-12 * np.eye(aux.bfs_nao)
    sqrt4 = np.sqrt(np.abs(eri_4c2e_diag(basis)))
    sqrt2 = np.sqrt(np.abs(np.diag(ints2c2e)))
    if sao:
        # spherical aux shells are screened as a whole (see df_algo10_helpers docstring)
        sqrt2 = algo10.aux_shell_max_bounds(sqrt2, aux)
    dense = np.einsum("rk,ijk->ijr", proj, Integrals.rys_3c2e_symm(basis, aux))
    return basis, aux, sqrt4, sqrt2, dense


def _stored_mask(sqrt4, sqrt2, threshold, strict, indicesA, indicesB):
    """(npairs, naux) boolean mask of the values the sparse layout must contain."""
    mask = np.zeros((indicesA.shape[0], sqrt2.shape[0]), dtype=bool)
    for p, (i, j) in enumerate(zip(indicesA, indicesB)):
        s = sqrt4[i, j]
        if strict and s * s < algo10.STRICT_PAIR_CUTOFF:
            continue
        mask[p] = s * sqrt2 > threshold
    return mask


def _pairs_by_aux(dense, indicesA, indicesB):
    """dense (nao, nao, naux) -> (npairs, naux) in pair-list order."""
    return dense[indicesA, indicesB, :]


def test_sao_projectors_are_spherical_subspace_projectors():
    from pyfock.Basis import Basis as _Basis

    proj = algo10.sao_aux_projectors(4)
    for l in range(5):
        ncart = (l + 1) * (l + 2) // 2
        P = proj[l, :ncart, :ncart]
        assert np.allclose(P, P.T) and np.allclose(P @ P, P)
        assert np.linalg.matrix_rank(P) == 2 * l + 1
        C = _Basis.cart2sph(l)
        np.testing.assert_allclose(C @ P, C, atol=1e-13)  # identity on the spherical subspace
        assert not np.any(proj[l, ncart:, :]) and not np.any(proj[l, :, ncart:])
    # d shell: the explicit form that the kernels used historically
    v = np.zeros(6)
    v[[0, 3, 5]] = 1.0 / np.sqrt(3.0)
    np.testing.assert_allclose(proj[2, :6, :6], np.eye(6) - np.outer(v, v), atol=1e-14)


@pytest.mark.parametrize("sao", [False, True])
@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("threshold", [1e-9, 1e-12])
def test_sparse_3c2e_matches_dense(sao, strict, threshold):
    basis, aux, sqrt4, sqrt2, dense = _setup("def2-SVP", sao)
    nao, naux = basis.bfs_nao, aux.bfs_nao
    indicesA, indicesB = np.tril_indices(nao)

    offsets, nsignificant = algo10.calc_offsets_3c2e_schwarz(sqrt4, sqrt2, threshold, strict, indicesA, indicesB)
    mask = _stored_mask(sqrt4, sqrt2, threshold, strict, indicesA, indicesB)
    per_pair = mask.sum(axis=1)
    assert offsets.dtype == np.int64 and offsets.shape == (indicesA.shape[0] + 1,)
    assert nsignificant == mask.sum() == offsets[-1]
    assert np.array_equal(np.diff(offsets), per_pair)
    # the geometry must really exercise the screening
    if strict:
        # the pair-level cut-off removes whole pairs (and, for this aux basis, every
        # surviving pair keeps all of its aux functions)
        assert np.any(per_pair == 0)
    else:
        # some pairs keep only part of the aux basis
        assert np.any((per_pair > 0) & (per_pair < naux))

    values = algo10.rys_3c2e_tri_schwarz_sparse_algo10(
        basis, aux, indicesA, indicesB, offsets, sqrt4, sqrt2, threshold, strict, nsignificant, sao=sao)
    assert values.shape == (nsignificant,)
    expected = _pairs_by_aux(dense, indicesA, indicesB)[mask]  # pair-major, increasing aux index
    np.testing.assert_allclose(values, expected, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("sao", [False, True])
def test_partially_stored_pairs_are_exact(sao):
    """
    Pairs that keep only part of the aux basis must be stored exactly.  In CAO mode
    this includes pairs keeping only some functions of a d/f/g shell; in SAO mode
    shells are all-or-nothing, so the partial pairs keep some shells but not others.
    """
    basis, aux, sqrt4, sqrt2, dense = _setup("def2-SVP", sao)
    nao, naux = basis.bfs_nao, aux.bfs_nao
    indicesA, indicesB = np.tril_indices(nao)
    threshold, strict = 1e-9, False
    mask = _stored_mask(sqrt4, sqrt2, threshold, strict, indicesA, indicesB)
    shell_of = np.asarray(aux.bfs_shell_index)
    partial_shells = 0
    for s in range(aux.nshells):
        cols = np.nonzero(shell_of == s)[0]
        if cols.size < 6:
            continue
        kept = mask[:, cols].sum(axis=1)
        partial_shells += np.count_nonzero((kept > 0) & (kept < cols.size))
    per_pair = mask.sum(axis=1)
    assert np.any((per_pair > 0) & (per_pair < naux)), "test geometry does not produce partially stored pairs"
    if sao:
        assert partial_shells == 0  # whole spherical shells only
    else:
        assert partial_shells > 0, "test geometry does not produce partially stored shells"

    offsets, nsignificant = algo10.calc_offsets_3c2e_schwarz(sqrt4, sqrt2, threshold, strict, indicesA, indicesB)
    values = algo10.rys_3c2e_tri_schwarz_sparse_algo10(
        basis, aux, indicesA, indicesB, offsets, sqrt4, sqrt2, threshold, strict, nsignificant, sao=sao)
    np.testing.assert_allclose(values, _pairs_by_aux(dense, indicesA, indicesB)[mask], atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("sao", [False, True])
def test_gamma_and_J_match_dense_contractions(sao):
    basis, aux, sqrt4, sqrt2, dense = _setup("def2-SVP", sao)
    nao, naux = basis.bfs_nao, aux.bfs_nao
    indicesA, indicesB = np.tril_indices(nao)
    threshold, strict = 1e-9, False

    offsets, nsignificant = algo10.calc_offsets_3c2e_schwarz(sqrt4, sqrt2, threshold, strict, indicesA, indicesB)
    values = algo10.rys_3c2e_tri_schwarz_sparse_algo10(
        basis, aux, indicesA, indicesB, offsets, sqrt4, sqrt2, threshold, strict, nsignificant, sao=sao)
    mask = _stored_mask(sqrt4, sqrt2, threshold, strict, indicesA, indicesB)
    stored_dense = _pairs_by_aux(dense, indicesA, indicesB) * mask  # (npairs, naux), zero where not stored

    rng = np.random.default_rng(7)
    dmat = rng.normal(size=(nao, nao))
    dmat = 0.5 * (dmat + dmat.T)
    dmat_tri = (2.0 * dmat - np.diag(np.diag(dmat)))[indicesA, indicesB]  # off-diagonals doubled

    gamma = algo10.df_coeff_calculator_algo10(
        values, dmat_tri, indicesA, indicesB, offsets, naux, sqrt4, sqrt2, threshold, strict, ncores=2)
    np.testing.assert_allclose(gamma, stored_dense.T @ dmat_tri, atol=1e-10, rtol=1e-12)
    # against the unscreened contraction the difference is bounded by the screening threshold
    assert np.abs(gamma - np.einsum("ijP,ij->P", dense, dmat)).max() < 1e-6

    coeff = rng.normal(size=naux)
    J_tri = algo10.J_tri_calculator_algo10(
        values, coeff, indicesA, indicesB, offsets, nao * (nao + 1) // 2, sqrt4, sqrt2, threshold, strict)
    np.testing.assert_allclose(J_tri, stored_dense @ coeff, atol=1e-10, rtol=1e-12)
    assert np.abs(J_tri - np.einsum("ijP,P->ij", dense, coeff)[indicesA, indicesB]).max() < 1e-6


def test_gamma_is_deterministic_and_thread_count_independent():
    basis, aux, sqrt4, sqrt2, dense = _setup("def2-SVP", False)
    nao, naux = basis.bfs_nao, aux.bfs_nao
    indicesA, indicesB = np.tril_indices(nao)
    threshold, strict = 1e-9, True
    offsets, nsignificant = algo10.calc_offsets_3c2e_schwarz(sqrt4, sqrt2, threshold, strict, indicesA, indicesB)
    values = algo10.rys_3c2e_tri_schwarz_sparse_algo10(
        basis, aux, indicesA, indicesB, offsets, sqrt4, sqrt2, threshold, strict, nsignificant)
    dmat_tri = np.linspace(-1.0, 1.0, indicesA.shape[0])
    ref = algo10.df_coeff_calculator_algo10(values, dmat_tri, indicesA, indicesB, offsets, naux, sqrt4, sqrt2, threshold, strict, ncores=1)
    for ncores in (1, 3, 8):
        out = algo10.df_coeff_calculator_algo10(values, dmat_tri, indicesA, indicesB, offsets, naux, sqrt4, sqrt2, threshold, strict, ncores=ncores)
        np.testing.assert_allclose(out, ref, atol=1e-12, rtol=1e-12)


# ---------------------------------------------------------------------------
# Spherical auxiliary functions: shell-consistent screening and the full
# gamma -> metric solve -> J pipeline against a dense reference
# ---------------------------------------------------------------------------
def _dense_pipeline_reference(basis, aux, dense, metric, dmat):
    import scipy.linalg

    gamma = np.einsum("ijP,ij->P", dense, dmat)
    cho = scipy.linalg.cho_factor(metric)
    coeff = scipy.linalg.cho_solve(cho, gamma)
    return coeff, np.einsum("ijP,P->ij", dense, coeff), coeff @ gamma, cho


def test_sao_shell_max_bounds_store_whole_shells():
    basis, aux, sqrt4, _, dense = _setup("def2-SVP", True)
    nao, naux = basis.bfs_nao, aux.bfs_nao
    indicesA, indicesB = np.tril_indices(nao)
    proj = aux.sph2cart_basis() @ aux.cart2sph_basis()
    metric = proj @ Integrals.rys_2c2e_symm(aux) @ proj.T + 1e-12 * np.eye(naux)
    sqrt2 = np.sqrt(np.abs(np.diag(metric)))  # raw per-function bounds
    bounds = algo10.aux_shell_max_bounds(sqrt2, aux)
    assert bounds.shape == sqrt2.shape and np.all(bounds >= sqrt2)
    shell_of = np.asarray(aux.bfs_shell_index)
    for s in range(aux.nshells):
        cols = np.nonzero(shell_of == s)[0]
        assert np.all(bounds[cols] == sqrt2[cols].max())
    threshold, strict = 1e-9, False
    mask = _stored_mask(sqrt4, bounds, threshold, strict, indicesA, indicesB)
    for s in range(aux.nshells):
        cols = np.nonzero(shell_of == s)[0]
        kept = mask[:, cols].sum(axis=1)
        assert np.all((kept == 0) | (kept == cols.size))  # all-or-nothing per shell
    offsets, nsignificant = algo10.calc_offsets_3c2e_schwarz(sqrt4, bounds, threshold, strict, indicesA, indicesB)
    values = algo10.rys_3c2e_tri_schwarz_sparse_algo10(
        basis, aux, indicesA, indicesB, offsets, sqrt4, bounds, threshold, strict, nsignificant, sao=True)
    np.testing.assert_allclose(values, _pairs_by_aux(dense, indicesA, indicesB)[mask], atol=1e-12, rtol=0.0)
    # per-function bounds are rejected in SAO mode (they would split shells)
    offsets_pf, nsig_pf = algo10.calc_offsets_3c2e_schwarz(sqrt4, sqrt2, threshold, strict, indicesA, indicesB)
    with pytest.raises(ValueError):
        algo10.rys_3c2e_tri_schwarz_sparse_algo10(
            basis, aux, indicesA, indicesB, offsets_pf, sqrt4, sqrt2, threshold, strict, nsig_pf, sao=True)


@pytest.mark.parametrize("sao", [False, True])
@pytest.mark.parametrize("strict", [False, True])
def test_full_df_coulomb_pipeline_matches_dense(sao, strict):
    """offsets -> sparse (ij|P) -> gamma -> Cholesky solve -> J, compared with the dense route."""
    import scipy.linalg

    basis, aux, sqrt4, sqrt2, dense = _setup("def2-SVP", sao)
    nao, naux = basis.bfs_nao, aux.bfs_nao
    indicesA, indicesB = np.tril_indices(nao)
    threshold = 1e-9
    metric = Integrals.rys_2c2e_symm(aux)
    if sao:
        proj = aux.sph2cart_basis() @ aux.cart2sph_basis()
        metric = proj @ metric @ proj.T + 1e-12 * np.eye(naux)
    bounds = sqrt2  # per-function (CAO) or shell-constant (SAO) bounds from _setup
    # a smooth, physically scaled density-like matrix
    S = Integrals.overlap_mat_symm(basis)
    dmat = 0.1 * (S + 0.3 * S @ S)
    dmat_tri = (2.0 * dmat - np.diag(np.diag(dmat)))[indicesA, indicesB]

    offsets, nsignificant = algo10.calc_offsets_3c2e_schwarz(sqrt4, bounds, threshold, strict, indicesA, indicesB)
    values = algo10.rys_3c2e_tri_schwarz_sparse_algo10(
        basis, aux, indicesA, indicesB, offsets, sqrt4, bounds, threshold, strict, nsignificant, sao=sao)
    gamma = algo10.df_coeff_calculator_algo10(
        values, dmat_tri, indicesA, indicesB, offsets, naux, sqrt4, bounds, threshold, strict, ncores=2)
    coeff_ref, J_ref, E_ref, cho = _dense_pipeline_reference(basis, aux, dense, metric, dmat)
    coeff = scipy.linalg.cho_solve(cho, gamma)
    J_tri = algo10.J_tri_calculator_algo10(
        values, coeff, indicesA, indicesB, offsets, nao * (nao + 1) // 2, sqrt4, bounds, threshold, strict)
    J = np.zeros((nao, nao))
    J[indicesA, indicesB] = J_tri
    J = J + J.T - np.diag(np.diag(J))

    # strict screening deliberately drops whole pairs (compensated elsewhere in the DFT
    # energy), so only the non-strict runs must reproduce the dense result tightly
    tol = 1e-5 if strict else 1e-10
    assert np.abs(coeff).max() < 10.0 * max(1.0, np.abs(coeff_ref).max())  # no null-space blow-up
    np.testing.assert_allclose(J, J_ref, atol=tol, rtol=0.0)
    assert abs(coeff @ gamma - E_ref) < (1e-6 if strict else 1e-10)
