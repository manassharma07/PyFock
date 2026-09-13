"""
Unit tests for RI-HF exchange from the DF_algo=11 blocks
(:mod:`pyfock.Integrals.df_algo11_exchange`).

The orthonormalized rows are compared with dense references built from the
same screened integrals the plan represents: K against
``sum_kl D_kl sum_PQ (ik|P) [M^-1]_PQ (Q|jl)`` and J/gamma against the
algorithm-11 ``gamma -> Cholesky solve -> J`` path, for CAO (Cartesian fit
space) and SAO (spherical fit space), both Schwarz modes, f functions, several
auxiliary block sizes.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg

from pyfock import Basis, Integrals, Mol
from pyfock.Integrals import df_algo11_exchange as algo11x
from pyfock.Integrals import df_algo11_helpers as algo11
from pyfock.Integrals.schwarz_helpers import eri_4c2e_diag


AUX_BASIS_NAME = "def2-universal-jkfit"
WATER = [["O", 0.0, 0.0, 0.117], ["H", 0.0, 0.757, -0.467], ["H", 0.0, -0.757, -0.467]]


def _system(kind, basis_name, sao):
    atoms = list(WATER) + ([[s, x + 6.0, y, z] for s, x, y, z in WATER] if kind == "two_waters" else [])
    mol = Mol(atoms=atoms)
    basis = Basis(mol, {"all": Basis.load(mol=mol, basis_name=basis_name)})
    aux = Basis(mol, {"all": Basis.load(mol=mol, basis_name=AUX_BASIS_NAME)})
    metric_cart = Integrals.rys_2c2e_symm(aux)
    dense_cart = Integrals.rys_3c2e_symm(basis, aux)
    sqrt4 = np.sqrt(np.abs(eri_4c2e_diag(basis)))
    if sao:
        proj = aux.sph2cart_basis() @ aux.cart2sph_basis()
        metric_plan = proj @ metric_cart @ proj.T + 1e-12 * np.eye(aux.bfs_nao)
        dense = np.einsum("rk,ijk->ijr", proj, dense_cart)       # what the plan stores (projected)
        metric_fit = aux.cart2sph_operator_blockwise(metric_cart)  # spherical metric
        to_fit = aux.cart2sph_basis()                              # (nsph, ncart)
    else:
        metric_plan = metric_cart
        dense = dense_cart
        metric_fit = metric_cart
        to_fit = np.eye(aux.bfs_nao)
    sqrt2 = np.sqrt(np.abs(np.diag(metric_plan)))
    S = Integrals.overlap_mat_symm(basis)
    dmat = 0.1 * (S + 0.3 * S @ S)  # smooth, physically scaled, symmetric
    return basis, aux, sqrt4, sqrt2, dense, metric_plan, metric_fit, to_fit, dmat


def _block_mask(plan, basis, aux):
    """(nao, nao, naux) mask of the (ij|P) the plan represents (both triangles)."""
    nao, naux = basis.bfs_nao, aux.bfs_nao
    mask = np.zeros((nao, nao, naux), dtype=bool)
    shell_of = np.asarray(basis.bfs_shell_index)
    aux_shell_of = np.asarray(aux.bfs_shell_index)
    for p in plan.work_iter.tolist():
        I, J = plan.pair_I[p], plan.pair_J[p]
        cols = np.zeros(naux, dtype=bool)
        for K in range(aux.nshells):
            if plan.Q_pair[p] * plan.Q_aux[K] > plan.threshold:
                cols |= aux_shell_of == K
        for i in np.nonzero(shell_of == I)[0]:
            for j in np.nonzero(shell_of == J)[0]:
                if plan.strict_schwarz and plan.sqrt_ints4c2e_diag[i, j] ** 2 < algo11.STRICT_PAIR_CUTOFF:
                    continue
                mask[i, j, cols] = True
                mask[j, i, cols] = True
    return mask


def _dense_exchange(kept_fit, metric_fit, dmat):
    """K_ij = sum_kl D_kl sum_PQ (ik|P) [M^-1]_PQ (Q|jl) from a dense (nao, nao, naux) tensor."""
    nao, _, naux = kept_fit.shape
    T = kept_fit.reshape(nao * nao, naux)
    V = scipy.linalg.solve(metric_fit, T.T, assume_a="pos").reshape(naux, nao, nao)
    return np.einsum("mlP,Pns,ls->mn", kept_fit, V, dmat, optimize=True)


def _build(kind, basis_name, sao, strict, threshold=1e-9):
    basis, aux, sqrt4, sqrt2, dense, metric_plan, metric_fit, to_fit, dmat = _system(kind, basis_name, sao)
    plan = algo11.build_plan(basis, aux, sqrt4, sqrt2, threshold, strict, sao=sao)
    ex = algo11x.build_exchange(plan, basis, aux, metric_fit, sao=sao)
    kept = dense * _block_mask(plan, basis, aux)
    kept_fit = np.einsum("ijP,QP->ijQ", kept, to_fit)
    return basis, aux, plan, ex, kept_fit, metric_plan, metric_fit, dmat


@pytest.mark.parametrize("sao", [False, True])
@pytest.mark.parametrize("strict", [False, True])
def test_exchange_and_coulomb_match_dense_two_waters(sao, strict):
    basis, aux, plan, ex, kept_fit, metric_plan, metric_fit, dmat = _build("two_waters", "def2-SVP", sao, strict)
    nao = basis.bfs_nao
    assert ex.nrows < nao * (nao + 1) // 2  # pair screening active
    assert ex.naux == kept_fit.shape[2]
    # rows are orthonormalized: B B^T restricted to the fit space equals T M^-1 T^T
    # exchange against the dense reference on the same screened integrals
    factor = np.linalg.cholesky(dmat + 1e-14 * np.eye(nao))
    K = algo11x.K_from_exchange(ex, factor)
    K_ref = _dense_exchange(kept_fit, metric_fit, dmat)
    assert np.allclose(K, K.T, atol=1e-13)
    np.testing.assert_allclose(K, K_ref, atol=1e-10, rtol=1e-10)
    # Coulomb through the orthonormal rows equals the plan's gamma -> cho_solve -> J path
    gamma_ex = algo11x.gamma_from_exchange(ex, dmat)
    J_ex = algo11x.J_from_exchange(ex, gamma_ex)
    cho = scipy.linalg.cho_factor(metric_plan)
    gamma_plan = algo11.gamma_from_plan(plan, dmat)
    coeff = scipy.linalg.cho_solve(cho, gamma_plan)
    J_plan = algo11.J_from_plan(plan, coeff)
    np.testing.assert_allclose(J_ex, J_plan, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(gamma_ex @ gamma_ex, coeff @ gamma_plan, rtol=1e-10)
    # and the dense Coulomb matrix of the same screened integrals
    J_ref = np.einsum("ijQ,Q->ij", kept_fit, scipy.linalg.solve(metric_fit, np.einsum("ijQ,ij->Q", kept_fit, dmat), assume_a="pos"))
    np.testing.assert_allclose(J_ex, J_ref, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("sao", [False, True])
def test_exchange_f_functions_h2o_tzvp(sao):
    basis, aux, plan, ex, kept_fit, metric_plan, metric_fit, dmat = _build("h2o", "def2-TZVP", sao, False)
    factor = algo11x.np.linalg.cholesky(dmat + 1e-14 * np.eye(basis.bfs_nao))
    K = algo11x.K_from_exchange(ex, factor)
    np.testing.assert_allclose(K, _dense_exchange(kept_fit, metric_fit, dmat), atol=1e-10, rtol=1e-10)


def test_exchange_independent_of_blocking():
    basis, aux, plan, ex, kept_fit, metric_plan, metric_fit, dmat = _build("two_waters", "def2-SVP", True, False)
    rng = np.random.default_rng(5)
    factor = rng.standard_normal((basis.bfs_nao, 7))  # generic low-rank density factor
    K_ref = algo11x.K_from_exchange(ex, factor)
    for budget in (1, 4096, 1 << 20):  # one aux function per block ... several
        np.testing.assert_allclose(algo11x.K_from_exchange(ex, factor, block_memory_bytes=budget), K_ref, atol=1e-12, rtol=1e-12)
    # zero-rank density -> zero exchange
    assert np.all(algo11x.K_from_exchange(ex, np.zeros((basis.bfs_nao, 0))) == 0.0)


def test_exchange_rejects_partially_cached_plan():
    basis, aux, sqrt4, sqrt2, dense, metric_plan, metric_fit, to_fit, dmat = _system("two_waters", "def2-SVP", False)
    plan = algo11.build_plan(basis, aux, sqrt4, sqrt2, 1e-9, False, max_memory_gb=0.0)
    with pytest.raises(ValueError, match="max_memory_ints3c2e"):
        algo11x.build_exchange(plan, basis, aux, metric_fit, sao=False)


def test_pseudo_cartesian_metric_diagonal_matches_dense():
    from pyfock.DFT_Helper_Coulomb import _pseudo_cartesian_metric_diagonal
    mol = Mol(atoms=[list(a) for a in WATER])
    aux = Basis(mol, {"all": Basis.load(mol=mol, basis_name=AUX_BASIS_NAME)})  # has d, f (and g) shells
    metric_cart = Integrals.rys_2c2e_symm(aux)
    metric_sph = aux.cart2sph_operator_blockwise(metric_cart)
    S = aux.sph2cart_basis()
    dense = S @ metric_sph @ S.T
    np.testing.assert_allclose(_pseudo_cartesian_metric_diagonal(aux, metric_sph, S), np.diag(dense), rtol=1e-12, atol=1e-12)


def test_mo_factor_matches_density_factor_path():
    """K from an explicit (nao, nocc) factor equals K from the eigen-factorization of the same density."""
    from pyfock.DFT_Helper_Coulomb import _density_matrix_factor
    basis, aux, plan, ex, kept_fit, metric_plan, metric_fit, dmat = _build("two_waters", "def2-SVP", True, False)
    rng = np.random.default_rng(9)
    C = np.linalg.qr(rng.standard_normal((basis.bfs_nao, 10)))[0] * np.sqrt(2.0)
    K_direct = algo11x.K_from_exchange(ex, C)
    K_eig = algo11x.K_from_exchange(ex, _density_matrix_factor(C @ C.T))
    np.testing.assert_allclose(K_direct, K_eig, atol=1e-10, rtol=1e-10)
