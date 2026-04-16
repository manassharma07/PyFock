from __future__ import annotations

import numpy as np
import pytest

from pyfock import Integrals

from .conftest import AUX_BASIS_NAME, BASIS_NAMES, RYS_4C2E_SLICE, build_basis, build_h2o_mol


ATOL = 1e-10
RTOL = 1e-10
OS_4C2E_BASIS_NAMES = ("def2-SVP", "def2-TZVP")


@pytest.mark.slow
def test_rys_2c2e_and_diag_match_references(refs):
    mol = build_h2o_mol()
    auxbasis = build_basis(mol, AUX_BASIS_NAME)

    ints2c2e = Integrals.rys_2c2e_symm(auxbasis)
    ints2c2e_diag = Integrals.rys_2c2e_diag(auxbasis)

    np.testing.assert_allclose(ints2c2e, refs["rys_2c2e_aux"], atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(ints2c2e_diag, refs["rys_2c2e_diag_aux"], atol=ATOL, rtol=RTOL)


@pytest.mark.slow
@pytest.mark.parametrize("basis_name", BASIS_NAMES)
def test_rys_3c2e_family_matches_references(refs, basis_name):
    mol = build_h2o_mol()
    basis = build_basis(mol, basis_name)
    auxbasis = build_basis(mol, AUX_BASIS_NAME)
    tag = basis_name.lower()

    ints3c2e = Integrals.rys_3c2e_symm(basis, auxbasis)
    ints3c2e_tri = Integrals.rys_3c2e_tri(basis, auxbasis)

    np.testing.assert_allclose(ints3c2e, refs[f"rys_3c2e_{tag}"], atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(ints3c2e_tri, refs[f"rys_3c2e_tri_{tag}"], atol=ATOL, rtol=RTOL)


@pytest.mark.slow
@pytest.mark.parametrize("basis_name", BASIS_NAMES)
def test_rys_4c2e_slice_matches_references(refs, basis_name):
    mol = build_h2o_mol()
    basis = build_basis(mol, basis_name)
    tag = basis_name.lower()

    ints4c2e = Integrals.rys_4c2e_symm(basis, slice=list(RYS_4C2E_SLICE))
    np.testing.assert_allclose(ints4c2e, refs[f"rys_4c2e_slice_{tag}"], atol=ATOL, rtol=RTOL)


@pytest.mark.slow
@pytest.mark.parametrize("basis_name", OS_4C2E_BASIS_NAMES)
def test_os_4c2e_matches_reference_fingerprint(refs, basis_name):
    mol = build_h2o_mol()
    basis = build_basis(mol, basis_name)
    tag = basis_name.lower()

    ints4c2e = Integrals.os_4c2e_symm(basis)

    assert ints4c2e.shape == tuple(refs[f"os_4c2e_shape_{tag}"].tolist())
    np.testing.assert_allclose(np.array([ints4c2e.sum()]), refs[f"os_4c2e_sum_{tag}"], atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(
        np.array([np.linalg.norm(ints4c2e.ravel())]),
        refs[f"os_4c2e_norm_{tag}"],
        atol=ATOL,
        rtol=RTOL,
    )

    sample_indices = refs[f"os_4c2e_sample_indices_{tag}"].astype(int)
    sample_values = np.array([ints4c2e[tuple(index)] for index in sample_indices], dtype=np.float64)
    np.testing.assert_allclose(sample_values, refs[f"os_4c2e_sample_values_{tag}"], atol=ATOL, rtol=RTOL)
