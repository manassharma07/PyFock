from __future__ import annotations

import numpy as np

from pyfock import Integrals, Mol

from .conftest import build_basis, build_h2o_mol


def test_molecule_from_xyz_has_expected_basic_properties():
    mol = build_h2o_mol()

    assert mol.success is True
    assert mol.natoms == 3
    assert mol.charge == 0
    assert mol.atomicSpecies == ["o", "h", "h"]
    assert mol.Zcharges == [8, 1, 1]
    assert mol.coords.shape == (3, 3)
    assert mol.coordsBohrs.shape == (3, 3)


def test_molecule_from_atoms_matches_xyz_geometry():
    mol_from_file = build_h2o_mol()
    atoms = [[sym, *coord] for sym, coord in zip(mol_from_file.atomicSpecies, mol_from_file.coords.tolist())]
    mol_from_atoms = Mol(atoms=atoms)

    assert mol_from_atoms.natoms == mol_from_file.natoms
    assert mol_from_atoms.atomicSpecies == mol_from_file.atomicSpecies
    np.testing.assert_allclose(mol_from_atoms.coords, mol_from_file.coords, atol=1e-12, rtol=0.0)


def test_default_basis_falls_back_to_def2_svp_when_sto3g_is_missing():
    atoms = [
        ["Pb", 0.0, 0.0, 0.0],
        ["H", 0.999, 0.999, 0.999],
        ["H", 0.999, -0.999, -0.999],
        ["H", -0.999, 0.999, -0.999],
        ["H", -0.999, -0.999, 0.999],
    ]
    mol = Mol(atoms=atoms)

    assert mol.success is True
    assert mol.basis.has_ecp
    assert mol.basis.ecps[0]["symbol"] == "Pb"
    assert mol.basis.ecps[0]["ncore"] == 60
    assert mol.Zcharges[0] == 22
    assert mol.nelectrons == 26


def test_molecule_dipole_helpers_are_consistent():
    mol = build_h2o_mol()
    basis = build_basis(mol, "def2-SVP")
    dipole_matrix = Integrals.dipole_moment_mat_symm(basis)
    density = np.eye(basis.bfs_nao)

    nuc = mol.get_nuc_dip_moment()
    elec = mol.get_elec_dip_moment(dipole_matrix, density)
    total = mol.get_dipole_moment(dipole_matrix, density)

    np.testing.assert_allclose(total, nuc - elec, atol=1e-12, rtol=0.0)
