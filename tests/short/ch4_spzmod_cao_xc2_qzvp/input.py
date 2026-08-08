#!/usr/bin/env python3
"""CH4 DFT calculation with the settings used by the PyFock benchmark."""

import os
import sys
from pathlib import Path


# Calculation settings
ncores = int(os.environ.get("OMP_NUM_THREADS", "4"))
basis_set_name = "def2-QZVP"
auxbasis_name = "def2-universal-jfit"
xc_ids = [1, 10]  # SPZMOD
ao_basis = "CAO"
xc_algo = 2

os.environ["OMP_NUM_THREADS"] = str(ncores)
os.environ["OPENBLAS_NUM_THREADS"] = str(ncores)
os.environ["MKL_NUM_THREADS"] = str(ncores)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(ncores)
os.environ["NUMEXPR_NUM_THREADS"] = str(ncores)
os.environ.setdefault("PYSCF_MAX_MEMORY", "25000")

from pyfock import Basis, DFT, Mol
from pyscf import dft as pyscf_dft
from pyscf import gto


xyz_file = "ch4.xyz"

# The benchmark uses a PySCF MINAO density as the initial PyFock density.
mol_pyscf = gto.Mole()
mol_pyscf.atom = str(xyz_file)
mol_pyscf.basis = basis_set_name
mol_pyscf.cart = True
mol_pyscf.verbose = 0
mol_pyscf.max_memory = 5000
mol_pyscf.build()
dmat_initial = pyscf_dft.RKS(mol_pyscf).init_guess_by_minao(mol_pyscf)

# PyFock calculation
mol = Mol(coordfile=str(xyz_file))
basis = Basis(mol, {"all": Basis.load(mol=mol, basis_name=basis_set_name)})
auxbasis = Basis(mol, {"all": Basis.load(mol=mol, basis_name=auxbasis_name)})

dft_obj = DFT(
    mol,
    basis,
    auxbasis,
    xc=xc_ids,
    conv_crit=1.0e-7,
    gridsLevel=3,
    use_pyscf_grids=True,
    blocksize=5000,
    save_ao_values=True,
    use_gpu=False,
    ncores=ncores,
)
dft_obj.dmat = dmat_initial
dft_obj.max_itr = 35
dft_obj.XC_algo = xc_algo
dft_obj.strict_schwarz = False
dft_obj.sao = False
dft_obj.use_libxc = False

energy, density_matrix = dft_obj.scf()

# Optional PySCF calculation with matching settings:
#     python3 input.py --with-pyscf
if "--with-pyscf" in sys.argv:
    mf = pyscf_dft.RKS(mol_pyscf).density_fit(auxbasis=auxbasis_name)
    mf.verbose = 4
    mf.xc = ",".join(str(functional_id) for functional_id in xc_ids)
    mf.direct_scf = False
    mf.max_cycle = 35
    mf.conv_tol = 1.0e-7
    mf.grids.level = 3
    energy_pyscf = mf.kernel(dm0=dmat_initial)

    print("\nPySCF comparison values")
    print("PySCF basis functions =", mol_pyscf.nao_nr())
    print("PySCF grid points =", mf.grids.weights.size)
    print("PySCF one-electron energy =", mf.scf_summary["e1"])
    print("PySCF nuclear repulsion energy =", mol_pyscf.energy_nuc())
    print("PySCF Coulomb energy =", mf.scf_summary["coul"])
    print("PySCF exchange-correlation energy =", mf.scf_summary["exc"])
    print("PySCF total energy =", energy_pyscf)

    print("delta E (PyFock - PySCF) =", energy - energy_pyscf)
