#!/usr/bin/env python3
"""CH4 RI-HF (density-fitted RHF) calculation with DF_algo=11, SAO orbitals."""

import os
import sys


# Calculation settings
ncores = int(os.environ.get("OMP_NUM_THREADS", "4"))
basis_set_name = "def2-SVP"
auxbasis_name = "def2-universal-jkfit"
ao_basis = "SAO"
df_algo = 11

os.environ["OMP_NUM_THREADS"] = str(ncores)
os.environ["OPENBLAS_NUM_THREADS"] = str(ncores)
os.environ["MKL_NUM_THREADS"] = str(ncores)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(ncores)
os.environ["NUMEXPR_NUM_THREADS"] = str(ncores)
os.environ.setdefault("PYSCF_MAX_MEMORY", "25000")

from pyfock import Basis, DFT, Mol
from pyscf import gto, scf


xyz_file = "ch4.xyz"

# A PySCF MINAO density is the initial PyFock density.
mol_pyscf = gto.Mole()
mol_pyscf.atom = xyz_file
mol_pyscf.basis = basis_set_name
mol_pyscf.cart = ao_basis == "CAO"
mol_pyscf.verbose = 0
mol_pyscf.max_memory = 5000
mol_pyscf.build()
dmat_initial = scf.RHF(mol_pyscf).init_guess_by_minao(mol_pyscf)

# PyFock calculation
mol = Mol(coordfile=xyz_file)
basis = Basis(mol, {"all": Basis.load(mol=mol, basis_name=basis_set_name)})
auxbasis = Basis(mol, {"all": Basis.load(mol=mol, basis_name=auxbasis_name)})

dft_obj = DFT(mol, basis, auxbasis, xc="HF", conv_crit=1.0e-7, use_gpu=False, ncores=ncores)
dft_obj.dmat = dmat_initial
dft_obj.max_itr = 35
dft_obj.rys = True
dft_obj.isDF = True
dft_obj.DF_algo = df_algo
dft_obj.threshold_schwarz = 1e-9
dft_obj.strict_schwarz = False
dft_obj.cholesky = True
dft_obj.orthogonalize = True
dft_obj.sao = ao_basis == "SAO"

energy, density_matrix = dft_obj.scf()

# Optional PySCF calculation with matching settings:
#     python3 input.py --with-pyscf
if "--with-pyscf" in sys.argv:
    mf = scf.RHF(mol_pyscf).density_fit(auxbasis=auxbasis_name)
    mf.verbose = 4
    mf.direct_scf = False
    mf.max_cycle = 35
    mf.conv_tol = 1.0e-7
    energy_pyscf = mf.kernel(dm0=dmat_initial)

    print("\nPySCF comparison values")
    print("PySCF basis functions =", mol_pyscf.nao_nr())
    print("PySCF one-electron energy =", mf.scf_summary["e1"])
    print("PySCF nuclear repulsion energy =", mol_pyscf.energy_nuc())
    print("PySCF two-electron energy =", mf.scf_summary["e2"])
    print("PySCF total energy =", energy_pyscf)

    print("delta E (PyFock - PySCF) =", energy - energy_pyscf)
