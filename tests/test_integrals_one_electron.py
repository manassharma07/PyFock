from __future__ import annotations

import numpy as np
import pytest

from pyfock import Integrals

from .conftest import BASIS_NAMES, build_basis, build_h2o_mol


ATOL = 1e-10
RTOL = 1e-10


@pytest.mark.parametrize("basis_name", BASIS_NAMES)
def test_one_electron_integrals_match_references(refs, basis_name):
    mol = build_h2o_mol()
    basis = build_basis(mol, basis_name)
    tag = basis_name.lower()

    overlap = Integrals.overlap_mat_symm(basis)
    nuclear = Integrals.nuc_mat_symm(basis, mol)
    kinetic = Integrals.kin_mat_symm(basis)
    dipole = Integrals.dipole_moment_mat_symm(basis)

    np.testing.assert_allclose(overlap, refs[f"overlap_{tag}"], atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(nuclear, refs[f"nuclear_{tag}"], atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(kinetic, refs[f"kinetic_{tag}"], atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(dipole, refs[f"dipole_{tag}"], atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("basis_name", BASIS_NAMES)
def test_integral_gradients_match_references(refs, basis_name):
    basis = build_basis(build_h2o_mol(), basis_name)
    tag = basis_name.lower()

    overlap_grad = Integrals.overlap_mat_grad_symm(basis)
    kinetic_grad = Integrals.kin_mat_grad_symm(basis)

    np.testing.assert_allclose(overlap_grad, refs[f"overlap_grad_{tag}"], atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(kinetic_grad, refs[f"kinetic_grad_{tag}"], atol=ATOL, rtol=RTOL)

