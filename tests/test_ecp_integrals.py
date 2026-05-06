from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pyfock import Basis, Integrals, Mol


pyscf = pytest.importorskip("pyscf")
from pyscf import gto


def test_cd_def2_tzvp_ecp_matches_pyscf_spherical_matrix():
    atoms = [["Cd", 0.0, 0.0, 0.0]]
    loader_mol = SimpleNamespace(natoms=1, basisSpecies=["Cd"])
    basis_string = Basis.load(mol=loader_mol, basis_name="def2-TZVP")
    mol = Mol(atoms=atoms, basis={"all": basis_string})

    ecp_cart = Integrals.ecp_mat_symm(mol.basis, n_radial=64, n_theta=14, n_phi=28)
    ecp_sph = mol.basis.cart2sph_operator_blockwise(ecp_cart)
    ecp_cart_analytical = Integrals.ecp_mat_symm_analytical(mol.basis, series_order=0)
    ecp_sph_analytical = mol.basis.cart2sph_operator_blockwise(ecp_cart_analytical)

    pyscf_mol = gto.M(
        atom="Cd 0 0 0",
        basis="def2-tzvp",
        ecp="def2-tzvp",
        cart=False,
        verbose=0,
    )
    ecp_pyscf = pyscf_mol.intor("ECPscalar_sph")

    np.testing.assert_allclose(ecp_sph, ecp_pyscf, atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(ecp_sph_analytical, ecp_pyscf, atol=1e-9, rtol=1e-9)


def test_cd_dimer_def2_tzvp_analytical_ecp_matches_pyscf_spherical_matrix():
    atoms = [
        ["Cd", 0.0, 0.0, -1.49],
        ["Cd", 0.0, 0.0, 1.49],
    ]
    loader_mol = SimpleNamespace(natoms=2, basisSpecies=["Cd", "Cd"])
    basis_string = Basis.load(mol=loader_mol, basis_name="def2-TZVP")
    mol = Mol(atoms=atoms, basis={"all": basis_string})

    ecp_cart = Integrals.ecp_mat_symm_analytical(mol.basis, series_order=12)
    ecp_sph = mol.basis.cart2sph_operator_blockwise(ecp_cart)

    pyscf_mol = gto.M(
        atom="Cd 0 0 -1.49; Cd 0 0 1.49",
        basis="def2-tzvp",
        ecp="def2-tzvp",
        cart=False,
        verbose=0,
    )
    ecp_pyscf = pyscf_mol.intor("ECPscalar_sph")

    np.testing.assert_allclose(ecp_sph, ecp_pyscf, atol=2e-8, rtol=1e-8)
