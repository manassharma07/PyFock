"""
Coulomb and exchange matrices from shell-quartet four-center integrals (``Integrals.jk_4c2e``,
Rys quadrature and the Obara-Saika scheme) and the SCF paths without density fitting that use
them (``isDF=False``: ``direct_scf``, ``coul_algo=1`` and ``coul_algo=2``).

The J and K matrices of all three modes (direct, Schwarz-screened store, complete store) and both
schemes are compared with PySCF's in the Cartesian basis, the direct kernel is checked to be
independent of the number of threads and of the scheme, and H2O / def2-SVP HF and PBE0 energies
are compared with PySCF.
"""
from __future__ import annotations

import io
import contextlib

import numba
import numpy as np
import pytest

from pyfock import Basis, DFT, Integrals, Mol
from pyfock.Integrals import jk_4c2e


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


@pytest.mark.parametrize("scheme", ["rys", "os"])
@pytest.mark.parametrize("mode", ["direct", "sparse", "dense"])
def test_jk_matches_pyscf(scheme, mode):
    _, basis = _basis("def2-SVP")
    dmat, J_ref, K_ref = _pyscf_jk("def2-SVP", basis)
    plan = jk_4c2e.build_plan(basis, threshold=1e-13, scheme=scheme)
    assert plan.scheme == scheme
    if mode == "direct":
        J, K = jk_4c2e.direct_jk(plan, dmat)
    else:
        jk_4c2e.store_integrals(plan, dense=(mode == "dense"))
        assert plan.store_gb > 0
        J, K = jk_4c2e.stored_jk(plan, dmat)
    assert np.abs(J - J_ref).max() < 1e-9
    assert np.abs(K - K_ref).max() < 1e-9
    assert np.allclose(J, J.T, atol=1e-14) and np.allclose(K, K.T, atol=1e-14)
    J_only = jk_4c2e.direct_jk(plan, dmat, with_k=False)
    assert np.abs(J_only - J).max() < 1e-12


@pytest.mark.parametrize("basis_name", ["def2-TZVP", "def2-QZVP"])
def test_schemes_threads_and_stores_agree(basis_name):
    """f (TZVP) and g (QZVP) shells: both schemes, 1 vs all threads, direct vs stored."""
    _, basis = _basis(basis_name)
    rng = np.random.default_rng(7)
    c = rng.standard_normal((basis.bfs_nao, 5))
    dmat = c @ c.T
    plan_rys = jk_4c2e.build_plan(basis, threshold=1e-13, scheme="rys")
    plan_os = jk_4c2e.build_plan(basis, threshold=1e-13, scheme="os")
    assert plan_rys.lmax == (3 if basis_name == "def2-TZVP" else 4)
    J, K = jk_4c2e.direct_jk(plan_rys, dmat)
    J_os, K_os = jk_4c2e.direct_jk(plan_os, dmat)
    assert np.abs(J_os - J).max() < 1e-10 and np.abs(K_os - K).max() < 1e-10
    nmax = numba.get_num_threads()
    numba.set_num_threads(1)
    try:
        J1, K1 = jk_4c2e.direct_jk(plan_os, dmat)
    finally:
        numba.set_num_threads(nmax)
    assert np.abs(J1 - J_os).max() < 1e-11 and np.abs(K1 - K_os).max() < 1e-11
    for dense in (False, True):
        jk_4c2e.store_integrals(plan_os, dense=dense)
        Js, Ks = jk_4c2e.stored_jk(plan_os, dmat)
        assert np.abs(Js - J).max() < 1e-10 and np.abs(Ks - K).max() < 1e-10
    assert jk_4c2e.store_size_gb(plan_os, dense=True) == pytest.approx(plan_os.store_gb)


def _scf(xc, sao, mode, rys=True, pyscf_grids=False):
    mol, basis = _basis("def2-SVP")
    dft = DFT(mol, basis, xc=xc, conv_crit=1e-10, ncores=2, gridsLevel=3, use_pyscf_grids=pyscf_grids)
    dft.max_itr = 60
    dft.sao = sao
    dft.isDF = False
    dft.rys = rys
    if mode == "direct":
        dft.direct_scf = True
    else:
        dft.coul_algo = 1 if mode == "dense" else 2
    with contextlib.redirect_stdout(io.StringIO()):
        energy, _ = dft.scf()
    assert dft.converged
    return float(energy), dft


@pytest.mark.parametrize("mode,sao,rys", [("direct", False, True), ("sparse", True, True), ("dense", False, True),
                                          ("sparse", False, True), ("direct", True, False), ("sparse", False, False),
                                          ("dense", True, False)])
def test_hf_without_df_matches_pyscf(mode, sao, rys):
    mol = _pyscf_mol("def2-SVP", cart=not sao)
    from pyscf import scf
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-11
    e_ref = mf.kernel()
    energy, dft = _scf("HF", sao, mode, rys=rys)
    assert abs(energy - e_ref) < 1e-8, (energy, e_ref)
    assert dft.Exx_energy < 0 and abs(dft.Total_energy - energy) < 1e-12


@pytest.mark.parametrize("rys", [True, False])
def test_pbe0_without_df_matches_pyscf(rys):
    mol = _pyscf_mol("def2-SVP", cart=False)
    from pyscf import dft as pyscf_dft
    mf = pyscf_dft.RKS(mol)
    mf.xc = "PBE0"
    mf.grids.level = 3
    mf.conv_tol = 1e-11
    e_ref = mf.kernel()
    energy, dft = _scf("PBE0", True, "direct", rys=rys, pyscf_grids=True)
    assert dft.exx_coef == pytest.approx(0.25)
    assert abs(energy - e_ref) < 1e-8, (energy, e_ref)


@pytest.mark.parametrize("coul_algo", [1, 2])
def test_store_over_budget_falls_back_to_direct(coul_algo):
    """A store larger than max_memory_4c2e is not built; the SCF runs direct and gives the same energy."""
    mol, basis = _basis("def2-SVP")
    energies = {}
    for budget in (None, 0.0):
        dft = DFT(mol, basis, xc="HF", conv_crit=1e-10, ncores=2)
        dft.max_itr = 60
        dft.isDF = False
        dft.coul_algo = coul_algo
        dft.max_memory_4c2e = budget
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            energy, _ = dft.scf()
        assert dft.converged and not dft.direct_scf
        log = out.getvalue()
        assert ("switching to direct SCF" in log) == (budget == 0.0)
        assert ("four-center integral block" in log) == (budget is None)
        energies[budget] = float(energy)
    assert abs(energies[None] - energies[0.0]) < 1e-9
