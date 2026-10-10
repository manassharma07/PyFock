"""
Analytical gradients without density fitting (``isDF=False``) from the four-center derivative
integrals of :mod:`pyfock.Integrals.jk_4c2e_grad`.

The two-electron term is compared with finite differences of ``1/2 Tr(D J) - a/4 Tr(D K)`` at a
fixed density (HF, PBE0 and pure-functional weights; d and f shells), and complete HF, PBE0 and PBE
gradients of H2O with PySCF's (spherical and Cartesian basis, direct and stored SCF).
"""
from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest

from pyfock import Basis, Data, DFT, DFT_Grad, Integrals, Mol
from pyfock.Integrals import jk_4c2e, jk_4c2e_grad


WATER = [["O", 0.0, 0.05, 0.117], ["H", 0.02, 0.757, -0.467], ["H", 0.0, -0.787, -0.427]]


def _basis(atoms, name):
    mol = Mol(atoms=[list(a) for a in atoms])
    return mol, Basis(mol, {"all": Basis.load(mol=mol, basis_name=name)})


def _displaced(atoms, iatom, d, step_bohr):
    out = [list(a) for a in atoms]
    out[iatom][1 + d] += step_bohr / Data.Angs2BohrFactor
    return out


def _two_electron_energy(atoms, name, D, exx):
    plan = jk_4c2e.build_plan(_basis(atoms, name)[1], threshold=1e-15)
    J, K = jk_4c2e.direct_jk(plan, D)
    return 0.5 * np.sum(D * J) - 0.25 * exx * np.sum(D * K)


@pytest.mark.parametrize("name,exx", [("def2-SVP", 1.0), ("def2-SVP", 0.25), ("def2-SVP", 0.0), ("def2-TZVP", 1.0)])
def test_two_electron_term_matches_finite_differences(name, exx):
    _, basis = _basis(WATER, name)
    rng = np.random.default_rng(2)
    C = 0.3 * rng.standard_normal((basis.bfs_nao, 5))
    D = C @ C.T + 0.05 * Integrals.overlap_mat_symm(basis)
    plan = jk_4c2e.build_plan(basis, threshold=1e-15)
    g = jk_4c2e_grad.grad_4c2e(plan, D, exx)
    assert np.abs(g.sum(axis=0)).max() < 1e-12 * np.abs(g).max()
    h = 1e-4
    for iatom, d in [(0, 1), (1, 2), (2, 0)]:
        e = [_two_electron_energy(_displaced(WATER, iatom, d, s * h), name, D, exx) for s in (1, -1)]
        assert abs(g[iatom, d] - (e[0] - e[1]) / (2 * h)) < 1e-8 * np.abs(g).max()
    # the plan's scheme does not matter (derivatives are always Rys); a loose threshold barely does
    g_os = jk_4c2e_grad.grad_4c2e(jk_4c2e.build_plan(basis, threshold=1e-15, scheme="os"), D, exx)
    np.testing.assert_allclose(g_os, g, atol=1e-12, rtol=0)
    assert np.abs(jk_4c2e_grad.grad_4c2e(plan, D, exx, threshold=1e-11) - g).max() < 1e-8


def _pyscf_gradient(xc, cart):
    pytest.importorskip("pyscf")
    from pyscf import dft as pyscf_dft, gto, scf
    mol = gto.Mole()
    mol.atom = [(s, (x, y, z)) for s, x, y, z in WATER]
    mol.basis = "def2-SVP"
    mol.cart = cart
    mol.verbose = 0
    mol.build()
    if xc == "HF":
        mf = scf.RHF(mol)
    else:
        mf = pyscf_dft.RKS(mol)
        mf.xc = xc
        mf.grids.level = 3
    mf.conv_tol = 1e-13
    mf.conv_tol_grad = 1e-8
    energy = mf.kernel()
    return energy, mf.nuc_grad_method().kernel()


@pytest.mark.parametrize("xc,sao,mode", [("HF", True, "direct"), ("HF", False, "sparse"),
                                         ("PBE0", True, "sparse"), ("PBE", True, "direct")])
def test_gradient_without_density_fitting_matches_pyscf(xc, sao, mode):
    e_ref, g_ref = _pyscf_gradient(xc, cart=not sao)
    mol, basis = _basis(WATER, "def2-SVP")
    # the forces of the functionals converge more slowly with the SCF (its criterion is the energy
    # change, and 1e-13 is at the round-off of these energies)
    dft = DFT(mol, basis, xc=xc, conv_crit=1e-11 if xc == "HF" else 1e-12, ncores=2, gridsLevel=3,
              use_pyscf_grids=True)
    dft.max_itr = 100
    dft.sao = sao
    dft.isDF = False
    if mode == "direct":
        dft.direct_scf = True
    else:
        dft.coul_algo = 2
    with contextlib.redirect_stdout(io.StringIO()):
        energy, _ = dft.scf()
        assert dft.converged
        res = DFT_Grad(dft).calculate()
    assert abs(energy - e_ref) < 2e-9
    key = "coulomb_exchange_4c2e" if xc != "PBE" else "coulomb_4c2e"
    assert key in res["gradient_components"]
    assert np.abs(res["gradient"] - g_ref).max() < (5e-8 if xc == "HF" else 1e-7)
