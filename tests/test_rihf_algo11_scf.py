"""
SCF-level checks of RI-HF with ``DF_algo=11``.

H2O / def2-SVP / def2-universal-jkfit RHF energies from the shell-blocked path
(``DF_algo=11``, orthonormalized rows) are compared with the dense reference
path (``DF_algo=3``, full three-center tensor) in CAO and SAO mode, and, when
PySCF is available, with PySCF's density-fitted RHF in the spherical basis.
"""
from __future__ import annotations

import numpy as np
import pytest

from pyfock import Basis, DFT, Mol


WATER = [["O", 0.0, 0.0, 0.117], ["H", 0.0, 0.757, -0.467], ["H", 0.0, -0.757, -0.467]]
BASIS, AUX = "def2-SVP", "def2-universal-jkfit"


def _pyscf_reference(sao):
    pyscf = pytest.importorskip("pyscf")
    from pyscf import gto, scf
    mol = gto.Mole()
    mol.atom = [(s, (x, y, z)) for s, x, y, z in WATER]
    mol.basis = BASIS
    mol.cart = not sao
    mol.verbose = 0
    mol.build()
    mf = scf.RHF(mol).density_fit(auxbasis=AUX)
    mf.conv_tol = 1e-11
    dm0 = mf.init_guess_by_minao(mol)
    return mf.kernel(dm0=dm0), dm0


def _run(df_algo, sao, dm0):
    mol = Mol(atoms=[list(a) for a in WATER])
    basis = Basis(mol, {"all": Basis.load(mol=mol, basis_name=BASIS)})
    aux = Basis(mol, {"all": Basis.load(mol=mol, basis_name=AUX)})
    dft = DFT(mol, basis, aux, xc="HF", conv_crit=1e-10, use_gpu=False, ncores=2)
    if dm0 is not None:
        dft.dmat = np.array(dm0, dtype=np.float64)
    dft.max_itr = 60
    dft.DF_algo = df_algo
    dft.sao = sao
    dft.strict_schwarz = False
    dft.threshold_schwarz = 1e-9
    energy, dmat = dft.scf()
    assert dft.converged
    return float(energy), dmat, dft


@pytest.mark.parametrize("sao", [False, True])
def test_rihf_algo11_matches_dense_algo3_and_pyscf(sao):
    try:
        e_pyscf, dm0 = _pyscf_reference(sao)
    except pytest.skip.Exception:
        e_pyscf, dm0 = None, None
    e11, d11, dft11 = _run(11, sao, dm0)
    e3, d3, _ = _run(3, sao, dm0)
    # same DF approximation, different storage/contraction path; only the 1e-9 Schwarz screening differs
    assert abs(e11 - e3) < 5e-8, (e11, e3)
    assert np.abs(d11 - d3).max() < 1e-6
    assert dft11.J_energy > 0 and abs(dft11.Total_energy - e11) < 1e-12
    if e_pyscf is not None:
        # PySCF fits in the spherical auxiliary basis, as PyFock does in SAO mode; the CAO
        # orbital basis (6 d functions) gives a slightly different, lower energy
        tol = 2e-6 if sao else 5e-3
        assert abs(e11 - e_pyscf) < tol, (e11, e_pyscf)
