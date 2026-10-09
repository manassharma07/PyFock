"""
Coulomb and exchange matrices from four-center integrals by shell-quartet Rys quadrature
(``Integrals.rys_4c2e_jk``) and the SCF paths without density fitting that use them
(``isDF=False``, ``rys=True``: ``direct_scf``, ``coul_algo=1`` and ``coul_algo=2``).

The J and K matrices of all three modes (direct, Schwarz-screened store, complete store) are
compared with PySCF's in the Cartesian basis, the direct kernel is checked to be independent of
the number of threads, and H2O / def2-SVP HF and PBE0 energies are compared with PySCF.
"""
from __future__ import annotations

import io
import contextlib

import numba
import numpy as np
import pytest

from pyfock import Basis, DFT, Integrals, Mol
from pyfock.Integrals import rys_4c2e_jk


WATER = [["O", 0.0, 0.0, 0.117], ["H", 0.0, 0.757, -0.467], ["H", 0.0, -0.757, -0.467]]


def _basis(name):
    mol = Mol(atoms=[list(a) for a in WATER])
    return mol, Basis(mol, {"all": Basis.load(mol=mol, basis_name=name)})


def _pyscf_mol(name, cart):
    pytest.importorskip("pyscf")
    from pyscf import gto
    mol = gto.Mole()
    mol.atom = [(s, (x, y, z)) for s, x, y, z in WATER]
    mol.basis = name
    mol.cart = cart
    mol.verbose = 0
    mol.build()
    return mol


def _pyscf_jk(name, basis):
    """PySCF J and K of its MINAO density, expressed in PyFock's normalization."""
    from pyscf import scf
    mol = _pyscf_mol(name, cart=True)
    dm = scf.RHF(mol).init_guess_by_minao(mol)
    mf = scf.RHF(mol)
    mf._is_mem_enough = lambda: False
    mf.direct_scf_tol = 1e-14
    vj, vk = mf.get_jk(mol, dm)
    # PySCF's Cartesian functions are not individually unit-normalized (d_xx, ...).
    s_pyfock = Integrals.overlap_mat_symm(basis)
    s_pyscf = mol.intor("int1e_ovlp")
    n = np.sqrt(np.diag(s_pyscf) / np.diag(s_pyfock))
    nn = np.outer(n, n)
    assert np.abs(s_pyfock - s_pyscf / nn).max() < 1e-9
    return dm * nn, vj / nn, vk / nn


@pytest.mark.parametrize("mode", ["direct", "sparse", "dense"])
def test_jk_matches_pyscf(mode):
    _, basis = _basis("def2-SVP")
    dmat, J_ref, K_ref = _pyscf_jk("def2-SVP", basis)
    plan = rys_4c2e_jk.build_plan(basis, threshold=1e-13)
    if mode == "direct":
        J, K = rys_4c2e_jk.direct_jk(plan, dmat)
    else:
        rys_4c2e_jk.store_integrals(plan, dense=(mode == "dense"))
        assert plan.store_gb > 0
        J, K = rys_4c2e_jk.stored_jk(plan, dmat)
    assert np.abs(J - J_ref).max() < 1e-9
    assert np.abs(K - K_ref).max() < 1e-9
    assert np.allclose(J, J.T, atol=1e-14) and np.allclose(K, K.T, atol=1e-14)
    J_only = rys_4c2e_jk.direct_jk(plan, dmat, with_k=False)
    assert np.abs(J_only - J).max() < 1e-12


def test_jk_f_shells_threads_and_stores_agree():
    _, basis = _basis("def2-TZVP")
    plan = rys_4c2e_jk.build_plan(basis, threshold=1e-13)
    assert plan.lmax == 3
    rng = np.random.default_rng(7)
    c = rng.standard_normal((basis.bfs_nao, 5))
    dmat = c @ c.T
    nmax = numba.get_num_threads()
    numba.set_num_threads(1)
    try:
        J1, K1 = rys_4c2e_jk.direct_jk(plan, dmat)
    finally:
        numba.set_num_threads(nmax)
    J, K = rys_4c2e_jk.direct_jk(plan, dmat)
    assert np.abs(J - J1).max() < 1e-11 and np.abs(K - K1).max() < 1e-11
    for dense in (False, True):
        rys_4c2e_jk.store_integrals(plan, dense=dense)
        Js, Ks = rys_4c2e_jk.stored_jk(plan, dmat)
        assert np.abs(Js - J).max() < 1e-10 and np.abs(Ks - K).max() < 1e-10
    assert rys_4c2e_jk.store_size_gb(plan, dense=True) == pytest.approx(plan.store_gb)


def _scf(xc, sao, mode, pyscf_grids=False):
    mol, basis = _basis("def2-SVP")
    dft = DFT(mol, basis, xc=xc, conv_crit=1e-10, ncores=2, gridsLevel=3, use_pyscf_grids=pyscf_grids)
    dft.max_itr = 60
    dft.sao = sao
    dft.isDF = False
    dft.rys = True
    if mode == "direct":
        dft.direct_scf = True
    else:
        dft.coul_algo = 1 if mode == "dense" else 2
    with contextlib.redirect_stdout(io.StringIO()):
        energy, _ = dft.scf()
    assert dft.converged
    return float(energy), dft


@pytest.mark.parametrize("mode,sao", [("direct", False), ("sparse", True), ("dense", False), ("sparse", False)])
def test_hf_without_df_matches_pyscf(mode, sao):
    mol = _pyscf_mol("def2-SVP", cart=not sao)
    from pyscf import scf
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-11
    e_ref = mf.kernel()
    energy, dft = _scf("HF", sao, mode)
    assert abs(energy - e_ref) < 1e-8, (energy, e_ref)
    assert dft.Exx_energy < 0 and abs(dft.Total_energy - energy) < 1e-12


def test_pbe0_without_df_matches_pyscf():
    mol = _pyscf_mol("def2-SVP", cart=False)
    from pyscf import dft as pyscf_dft
    mf = pyscf_dft.RKS(mol)
    mf.xc = "PBE0"
    mf.grids.level = 3
    mf.conv_tol = 1e-11
    e_ref = mf.kernel()
    energy, dft = _scf("PBE0", True, "direct", pyscf_grids=True)
    assert dft.exx_coef == pytest.approx(0.25)
    assert abs(energy - e_ref) < 1e-8, (energy, e_ref)
