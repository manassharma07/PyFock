from __future__ import annotations

import numpy as np

from pyfock import Basis

from .conftest import H2O_XYZ, build_basis, build_h2o_mol


def test_basis_load_api_basic_properties():
    mol = build_h2o_mol()
    basis = build_basis(mol, "def2-SVP")

    assert basis.bfs_nao > 0
    assert basis.totalnprims == sum(basis.nprims)
    assert len(basis.bfs_label) == basis.bfs_nao
    assert len(basis.bfs_lmn) == basis.bfs_nao
    assert len(basis.bfs_atoms) == basis.bfs_nao
    assert len(basis.shell_bfs_offset) == basis.nshells


def test_basis_loadfromfile_matches_load_for_sto3g():
    mol = build_h2o_mol()
    basis_from_file = Basis(
        mol, {"all": Basis.loadfromfile(mol=mol, basis_name=str(H2O_XYZ.parent / "sto-3g.dat"))}
    )
    basis_from_library = Basis(mol, {"all": Basis.load(mol=mol, basis_name="sto-3g")})

    assert basis_from_file.bfs_nao == basis_from_library.bfs_nao
    np.testing.assert_allclose(
        np.asarray(basis_from_file.bfs_contr_prim_norms),
        np.asarray(basis_from_library.bfs_contr_prim_norms),
        atol=1e-12,
        rtol=1e-12,
    )


def test_basis_coordinates_follow_parent_molecule():
    mol = build_h2o_mol()
    basis = build_basis(mol, "def2-TZVP")

    atom_coords = np.asarray(mol.coordsBohrs)
    bfs_coords = np.asarray(basis.bfs_coords)
    bfs_atoms = np.asarray(basis.bfs_atoms, dtype=int)

    np.testing.assert_allclose(bfs_coords, atom_coords[bfs_atoms], atol=1e-12, rtol=0.0)
