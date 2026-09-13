"""
Tests of the SANO initial guess (superposition of atomic natural-orbital densities,
``pyfock.Guess`` and ``DFT(dmat_guess_method='sano')``).
"""
from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest

from pyfock import Basis, DFT, Grids, Guess, Integrals, Mol


WATER = [["O", 0.0, 0.0, 0.1173], ["H", 0.0, 0.7572, -0.4692], ["H", 0.0, -0.7572, -0.4692]]
HI = [["H", 0.0, 0.0, 0.0], ["I", 0.0, 0.0, 1.609]]


def _mol_basis(atoms, basis_name):
    mol = Mol(atoms=[list(a) for a in atoms])
    basis = Basis(mol, {"all": Basis.load(mol=mol, basis_name=basis_name)})
    return mol, basis


def _natural_occupations(D, S):
    w, V = np.linalg.eigh(S)
    S_half = (V * np.sqrt(w)) @ V.T
    return np.sort(np.linalg.eigvalsh(S_half @ D @ S_half))[::-1]


def _scf(dft):
    with contextlib.redirect_stdout(io.StringIO()):
        energy, dmat = dft.scf()
    assert dft.converged
    return float(energy), dft.niter, dmat


def test_shell_occupations():
    occ_c = Guess.shell_occupations(6)
    assert occ_c[0] == [2.0, 2.0]
    assert occ_c[1] == pytest.approx([2.0 / 3.0])
    assert occ_c[2] == [] and occ_c[3] == []
    # Fe in the spin-restricted HF ground configuration 3d8 (no 4s): 1s2s3s, 2p3p, 3d
    occ_fe = Guess.shell_occupations(26)
    assert occ_fe[0] == [2.0, 2.0, 2.0] and occ_fe[1] == [2.0, 2.0]
    assert occ_fe[2] == pytest.approx([1.6])
    # I with the 28-electron def2 ECP: 1s2s3s, 2p3p and 3d are removed
    occ_i = Guess.shell_occupations(53, ncore=28)
    assert occ_i[0] == [0.0, 0.0, 0.0, 2.0, 2.0]
    assert occ_i[1] == pytest.approx([0.0, 0.0, 2.0, 5.0 / 3.0])
    assert occ_i[2] == [0.0, 2.0]


def test_shell_occupations_rejects_unsupported_cases():
    with pytest.raises(ValueError):
        Guess.shell_occupations(97)
    with pytest.raises(ValueError):
        Guess.shell_occupations(53, ncore=27)


def test_sano_references_select_the_ano_rcc_papers_of_the_elements_present():
    refs_light = Guess.sano_references([1, 6, 8])
    refs_tm = Guess.sano_references([26])
    assert any("Main group atoms" in r for r in refs_light)
    assert not any("transition metal" in r for r in refs_light)
    assert any("transition metal" in r for r in refs_tm)
    assert "please cite" in Guess.sano_citation_text([1, 8])


@pytest.mark.parametrize(
    "basis_name, min_fraction",
    [("sto-3g", 0.985), ("def2-SVP", 0.998), ("def2-TZVP", 0.9995)],
)
def test_sano_water_electron_count_and_structure(basis_name, min_fraction):
    mol, basis = _mol_basis(WATER, basis_name)
    D, info = Guess.sano_dmat(mol, basis)
    S = Integrals.overlap_mat_symm(basis)
    nelec = float(np.einsum("ij,ji->", D, S))
    assert nelec == pytest.approx(info["nelectrons_guess"])
    # renormalized projected orbitals: exact electron count
    assert nelec == pytest.approx(mol.nelectrons, abs=1e-8)
    # what the calculation basis can represent of the atomic natural orbitals
    assert min_fraction * mol.nelectrons < info["nelectrons_projected"] <= mol.nelectrons + 1e-8
    assert D.shape == (basis.bfs_nao, basis.bfs_nao)
    assert np.allclose(D, D.T)
    assert np.linalg.eigvalsh(D).min() > -1e-10  # positive semidefinite
    assert info["species"] == ["O", "H"] and info["renormalized"]
    D_plain, info_plain = Guess.sano_dmat(mol, basis, renormalize=False)
    assert info_plain["nelectrons_guess"] == pytest.approx(info["nelectrons_projected"])
    assert not info_plain["renormalized"]


def test_sano_closed_shell_atom_in_its_own_minimal_basis_is_exact():
    # Zn (3d10 4s2) with ANO-RCC-MB itself as the calculation basis: the projection is the
    # identity and the guess is the exact atomic density (15 natural occupations of 2),
    # which also exercises the spherical -> Cartesian transform of the d shell.
    mol, basis = _mol_basis([["Zn", 0.0, 0.0, 0.0]], "ano-rcc-mb")
    D, info = Guess.sano_dmat(mol, basis)
    S = Integrals.overlap_mat_symm(basis)
    occ = _natural_occupations(D, S)
    assert info["nelectrons_guess"] == pytest.approx(30.0, abs=1e-8)
    np.testing.assert_allclose(occ[:15], 2.0, atol=1e-8)
    np.testing.assert_allclose(occ[15:], 0.0, atol=1e-8)
    # idempotent up to the precision of the tabulated ANO contraction coefficients
    assert np.abs(D @ S @ D - 2.0 * D).max() < 1e-6


def test_sano_is_rotationally_invariant():
    rng = np.random.default_rng(7)
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    rotated = [[a[0], *(Q @ np.array(a[1:])).tolist()] for a in WATER]
    spectra = []
    for atoms in (WATER, rotated):
        mol, basis = _mol_basis(atoms, "def2-SVP")
        D, _ = Guess.sano_dmat(mol, basis)
        spectra.append(_natural_occupations(D, Integrals.overlap_mat_symm(basis)))
    np.testing.assert_allclose(spectra[0], spectra[1], atol=1e-9)


def test_sano_matches_pyscf_minao():
    pytest.importorskip("pyscf")
    from pyscf import gto, scf

    for pyfock_name, pyscf_name, dm_tol in (("3-21G", "321g", 1e-5), ("def2-SVP", "def2-svp", None)):
        mol, basis = _mol_basis(WATER, pyfock_name)
        D, _ = Guess.sano_dmat(mol, basis, renormalize=False)  # plain projection = PySCF minao
        S = Integrals.overlap_mat_symm(basis)
        pmol = gto.M(atom=[[a[0], a[1:]] for a in WATER], basis=pyscf_name, cart=True, verbose=0)
        Dp = np.asarray(scf.hf.init_guess_by_minao(pmol))
        Sp = pmol.intor("int1e_ovlp")
        # The natural-occupation spectrum does not depend on AO ordering or normalization
        np.testing.assert_allclose(_natural_occupations(D, S), _natural_occupations(Dp, Sp), atol=1e-6)
        if dm_tol is not None:  # s,p-only basis: identical AO order and normalization
            np.testing.assert_allclose(D, Dp, atol=dm_tol)


def test_sano_ecp_atom_core_shells_removed_and_valence_renormalized():
    mol, basis = _mol_basis(HI, "def2-SVP")
    assert mol.nelectrons == 26  # 28-electron ECP on iodine
    D, info = Guess.sano_dmat(mol, basis)
    S = Integrals.overlap_mat_symm(basis)
    assert info["species"] == ["H", "I"]
    assert info["nelectrons_guess"] == pytest.approx(26.0, abs=1e-8)
    # the nodeless ECP valence functions represent only ~93% of the all-electron valence ANOs
    assert 0.90 * 26 < info["nelectrons_projected"] < 0.96 * 26
    # 13 doubly occupied valence orbitals of I (4s 4p 4d 5s) + 5p (5/3 each) + H 1s: the strongest
    # natural occupations are near 2 and the total is exactly N
    occ = _natural_occupations(D, S)
    assert occ.sum() == pytest.approx(26.0, abs=1e-8)
    assert occ[0] < 2.5
    D_plain, info_plain = Guess.sano_dmat(mol, basis, renormalize=False)
    assert info_plain["nelectrons_guess"] == pytest.approx(info["nelectrons_projected"])
    assert D.shape == D_plain.shape == (basis.bfs_nao, basis.bfs_nao)


def test_sano_ghost_atoms_carry_no_density():
    mol, basis = _mol_basis(WATER + [["X-Ne", 0.0, 0.0, 3.0]], "def2-SVP")
    assert mol.nelectrons == 10
    D, info = Guess.sano_dmat(mol, basis)
    assert info["nelectrons_guess"] == pytest.approx(10.0, abs=1e-8)
    assert info["species"] == ["O", "H"] and info["nuclear_charges"] == [8, 1, 1]
    S = Integrals.overlap_mat_symm(basis)
    population = np.einsum("ij,ji->i", D, S)
    on_ghost = np.asarray(basis.bfs_atoms) == 3
    assert abs(population[on_ghost].sum()) < 0.01


def test_project_dmat_onto_the_same_basis_is_the_identity():
    mol, basis = _mol_basis(WATER, "def2-SVP")
    D, _ = Guess.sano_dmat(mol, basis)
    np.testing.assert_allclose(Guess.project_dmat(D, basis, basis), D, atol=1e-10)


def test_sano_is_the_default_guess_method():
    mol, basis = _mol_basis(WATER, "sto-3g")
    assert DFT(mol, basis, xc=[1, 7], ncores=1).dmat_guess_method == "sano"
    assert DFT(mol, basis, xc=[1, 7], ncores=1, dmat_guess_method="core").dmat_guess_method == "core"


def test_dft_guess_dmat_dispatch_and_unknown_method():
    mol, basis = _mol_basis(WATER, "sto-3g")
    dft = DFT(mol, basis, xc=[1, 7], dmat_guess_method="sano", ncores=1)
    with contextlib.redirect_stdout(io.StringIO()):
        D = dft.guess_dmat()
    assert D.shape == (basis.bfs_nao, basis.bfs_nao)
    assert dft.guess_info["nelectrons"] == 10
    dft.dmat_guess_method = "huckel"
    with pytest.raises(ValueError):
        dft.guess_dmat()


def test_scf_from_sano_guess_reaches_the_core_guess_energy_in_fewer_iterations():
    mol, basis = _mol_basis(WATER, "def2-SVP")
    # one explicit (unpruned) grid for both runs, so that only the starting density differs
    grid_basis = Basis(mol, {"all": Basis.load(mol=mol, basis_name="def2-QZVP")})
    with contextlib.redirect_stdout(io.StringIO()):
        grids = Grids(mol, basis=grid_basis, level=3, ncores=2)
    results = {}
    for method in ("core", "sano"):
        dft = DFT(mol, basis, xc=[1, 7], dmat_guess_method=method, ncores=2, grids=grids)
        dft.conv_crit = 1e-9
        dft.max_itr = 60
        results[method] = _scf(dft)[:2]
    assert abs(results["sano"][0] - results["core"][0]) < 1e-7
    assert results["sano"][1] < results["core"][1]


def test_scf_output_lists_the_sano_references():
    mol, basis = _mol_basis(WATER, "sto-3g")
    dft = DFT(mol, basis, xc=[1, 7], dmat_guess_method="sano", ncores=1)
    dft.conv_crit = 1e-6
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        dft.scf()
    out = buf.getvalue()
    assert "Initial guess: SANO" in out
    assert "doi:10.1002/jcc.20393" in out       # Van Lenthe et al., SAD
    assert "doi:10.1021/jp031064+" in out       # Roos et al., ANO-RCC main group
    assert "doi:10.1021/acs.jcim.9b00725" in out  # Basis Set Exchange
