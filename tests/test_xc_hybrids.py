"""
Global hybrid functionals: native B3LYP / B3LYP5 / PBE0 (LibXC IDs 402, 475, 406)
and the exact-exchange machinery in the SCF.

* The native semilocal parts and the new LDA_C_VWN_RPA are compared with pylibxc
  point by point (energy per particle, vrho, vsigma).
* Full DF-RKS energies for H2O (PySCF grids, def2-SVP, def2-universal-jkfit) are
  compared between the native and the pylibxc path and with PySCF's density-fitted
  RKS for B3LYP and PBE0.
"""
from __future__ import annotations

import numpy as np
import pytest

from pyfock import XC


def _rho_sigma(n=300, seed=1):
    rng = np.random.default_rng(seed)
    rho = 10.0 ** rng.uniform(-6, 1, n)
    sigma = (10.0 ** rng.uniform(-8, 2, n)) * rho ** (8.0 / 3.0)
    return rho, sigma


def test_hybrid_metadata():
    assert XC.resolve_functional("B3LYP") == [402]
    assert XC.resolve_functional("B3LYP5") == [475]
    assert XC.resolve_functional("PBE0") == [406] == XC.resolve_functional("PBEH")
    assert XC.get_exx_coefficient(402) == 0.2 and XC.get_exx_coefficient(406) == 0.25 and XC.get_exx_coefficient(101) == 0.0
    assert XC.get_family(402) == 3 and XC.get_semilocal_family(402) == 2 and XC.get_semilocal_family(1) == 1
    assert XC.xc_semilocal_family([402]) == 2 and XC.xc_semilocal_family([1, 7]) == 1 and XC.xc_semilocal_family([1, 130]) == 2
    coefs = dict((fid, c) for c, fid in XC.get_hybrid_components(402))
    assert coefs == {1: 0.08, 106: 0.72, 8: 0.19, 131: 0.81}
    assert all(fid in XC.get_implemented_ids() for fid in (8, 402, 406, 475))


def test_vwn_rpa_matches_libxc():
    pylibxc = pytest.importorskip("pylibxc")
    rho, _ = _rho_sigma()
    ref = pylibxc.LibXCFunctional(8, "unpolarized").compute({"rho": rho})
    e, v = XC.func_compute(8, rho, use_gpu=False)
    np.testing.assert_allclose(e, ref["zk"].ravel(), rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(v, ref["vrho"].ravel(), rtol=1e-13, atol=1e-15)


@pytest.mark.parametrize("fid,tol", [(402, 1e-12), (475, 1e-12), (406, 5e-6)])
def test_native_hybrid_semilocal_part_matches_libxc(fid, tol):
    """The native composite equals LibXC's semilocal part (PBE0 to the accuracy of the native PBE constants)."""
    pylibxc = pytest.importorskip("pylibxc")
    rho, sigma = _rho_sigma()
    fn = pylibxc.LibXCFunctional(fid, "unpolarized")
    assert fn.get_hyb_exx_coef() == XC.get_exx_coefficient(fid)
    ref = fn.compute({"rho": rho, "sigma": sigma})
    e, vrho, vsigma = XC.func_compute(fid, rho, sigma=sigma, use_gpu=False)
    for mine, theirs in ((e, ref["zk"]), (vrho, ref["vrho"]), (vsigma, ref["vsigma"])):
        theirs = theirs.ravel()
        assert np.abs(mine - theirs).max() <= tol * np.abs(theirs).max()


# --------------------------------------------------------------------------- SCF level
WATER = [["O", 0.0, 0.0, 0.117], ["H", 0.0, 0.757, -0.467], ["H", 0.0, -0.757, -0.467]]
BASIS, AUX = "def2-SVP", "def2-universal-jkfit"


def _pyscf_rks(xc_name):
    pyscf = pytest.importorskip("pyscf")
    from pyscf import dft, gto
    mol = gto.Mole()
    mol.atom = [(s, (x, y, z)) for s, x, y, z in WATER]
    mol.basis = BASIS
    mol.cart = False
    mol.verbose = 0
    mol.build()
    mf = dft.RKS(mol).density_fit(auxbasis=AUX)
    mf.xc = xc_name
    mf.grids.level = 3
    mf.conv_tol = 1e-11
    dm0 = mf.init_guess_by_minao(mol)
    e = mf.kernel(dm0=dm0)
    return e, dm0, mf.grids


def _pyfock_rks(xc_name, use_libxc, dm0, grids):
    from pyfock import Basis, DFT, Mol
    mol = Mol(atoms=[list(a) for a in WATER])
    basis = Basis(mol, {"all": Basis.load(mol=mol, basis_name=BASIS)})
    aux = Basis(mol, {"all": Basis.load(mol=mol, basis_name=AUX)})
    dft = DFT(mol, basis, aux, xc=xc_name, conv_crit=1e-10, use_gpu=False, ncores=2, grids=grids, blocksize=5000, save_ao_values=True)
    dft.dmat = np.array(dm0)
    dft.max_itr = 60
    dft.XC_algo = 2
    dft.DF_algo = 11
    dft.sao = True
    dft.use_libxc = use_libxc
    dft.strict_schwarz = False
    energy, _ = dft.scf()
    assert dft.converged
    return float(energy), dft


@pytest.mark.parametrize("xc_name,tol_libxc,tol_pyscf", [("B3LYP", 1e-8, 3e-6), ("PBE0", 5e-7, 3e-6)])
def test_hybrid_scf_native_vs_libxc_vs_pyscf(xc_name, tol_libxc, tol_pyscf):
    e_pyscf, dm0, grids = _pyscf_rks(xc_name)
    e_native, dft = _pyfock_rks(xc_name, False, dm0, grids)
    assert dft.exx_coef == XC.get_exx_coefficient(XC.resolve_functional(xc_name)[0])
    assert dft.Exx_energy < 0 and dft.XC_energy < 0
    pytest.importorskip("pylibxc")
    e_libxc, _ = _pyfock_rks(xc_name, True, dm0, grids)
    assert abs(e_native - e_libxc) < tol_libxc, (e_native, e_libxc)
    assert abs(e_native - e_pyscf) < tol_pyscf, (e_native, e_pyscf)
