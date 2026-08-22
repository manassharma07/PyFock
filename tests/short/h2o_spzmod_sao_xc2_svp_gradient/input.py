#!/usr/bin/env python3
"""H2O analytical DFT gradient calculation using SAO orbitals."""

import os
import sys

import numpy as np


# Calculation settings
ncores = int(os.environ.get("OMP_NUM_THREADS", "4"))
basis_set_name = "def2-SVP"
auxbasis_name = "def2-universal-jfit"
xc_ids = [1, 10]  # SPZMOD
ao_basis = "SAO"
xc_algo = 2

os.environ["OMP_NUM_THREADS"] = str(ncores)
os.environ["OPENBLAS_NUM_THREADS"] = str(ncores)
os.environ["MKL_NUM_THREADS"] = str(ncores)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(ncores)
os.environ["NUMEXPR_NUM_THREADS"] = str(ncores)
os.environ.setdefault("PYSCF_MAX_MEMORY", "25000")

from pyfock import Basis, DFT, DFT_Grad, Mol
from pyscf import dft as pyscf_dft
from pyscf import gto


xyz_file = "h2o.xyz"

# Use a PySCF MINAO density as the initial PyFock density.
mol_pyscf = gto.Mole()
mol_pyscf.atom = xyz_file
mol_pyscf.basis = basis_set_name
mol_pyscf.cart = ao_basis == "CAO"
mol_pyscf.verbose = 0
mol_pyscf.max_memory = 5000
mol_pyscf.build()
dmat_initial = pyscf_dft.RKS(mol_pyscf).init_guess_by_minao(mol_pyscf)

# PyFock SCF calculation
mol = Mol(coordfile=xyz_file)
basis = Basis(mol, {"all": Basis.load(mol=mol, basis_name=basis_set_name)})
auxbasis = Basis(mol, {"all": Basis.load(mol=mol, basis_name=auxbasis_name)})

dft_obj = DFT(
    mol,
    basis,
    auxbasis,
    xc=xc_ids,
    conv_crit=1.0e-11,
    gridsLevel=3,
    use_pyscf_grids=True,
    blocksize=5000,
    save_ao_values=True,
    use_gpu=False,
    ncores=ncores,
)
dft_obj.dmat = dmat_initial
dft_obj.max_itr = 60
dft_obj.isDF = True
dft_obj.DF_algo = 10
dft_obj.XC_algo = xc_algo
dft_obj.sortGrids = False
dft_obj.xc_bf_screen = True
dft_obj.threshold_schwarz = 1.0e-9
dft_obj.strict_schwarz = False
dft_obj.cholesky = True
dft_obj.orthogonalize = True
dft_obj.sao = ao_basis == "SAO"
dft_obj.use_libxc = False

energy_pyfock, _ = dft_obj.scf()

# Analytical PyFock gradient in Hartree/Bohr.
gradient_result = DFT_Grad(dft_obj).calculate()
gradient_pyfock = gradient_result["gradient"]

print("\nPyFock analytical DFT gradient")
for atom_index, (symbol, components) in enumerate(
    zip(mol.atomicSpecies, gradient_pyfock), start=1
):
    for axis, value in zip("XYZ", components):
        print(
            f"PyFock gradient atom {atom_index} {symbol} {axis} (Ha/Bohr) = {value}"
        )
print("PyFock gradient norm (Ha/Bohr) =", np.linalg.norm(gradient_pyfock))
print("PyFock maximum gradient component (Ha/Bohr) =", np.abs(gradient_pyfock).max())
print("PyFock gradient sum X (Ha/Bohr) =", gradient_pyfock[:, 0].sum())
print("PyFock gradient sum Y (Ha/Bohr) =", gradient_pyfock[:, 1].sum())
print("PyFock gradient sum Z (Ha/Bohr) =", gradient_pyfock[:, 2].sum())

# Optional PySCF calculation with matching settings:
#     python3 input.py --with-pyscf
if "--with-pyscf" in sys.argv:
    mf = pyscf_dft.RKS(mol_pyscf).density_fit(auxbasis=auxbasis_name)
    mf.verbose = 4
    mf.xc = ",".join(str(functional_id) for functional_id in xc_ids)
    mf.direct_scf = False
    mf.max_cycle = 60
    mf.conv_tol = 1.0e-11
    mf.grids.level = 3
    energy_pyscf = mf.kernel(dm0=dmat_initial)
    gradient_pyscf = mf.nuc_grad_method().kernel()
    gradient_difference = np.abs(gradient_pyfock - gradient_pyscf)

    print("\nPySCF comparison values")
    print("PySCF basis functions =", mol_pyscf.nao_nr())
    print("PySCF grid points =", mf.grids.weights.size)
    print("PySCF total energy =", energy_pyscf)
    print("delta E (PyFock - PySCF) =", energy_pyfock - energy_pyscf)
    for atom_index, (symbol, components) in enumerate(
        zip(mol.atomicSpecies, gradient_pyscf), start=1
    ):
        for axis, value in zip("XYZ", components):
            print(
                f"PySCF gradient atom {atom_index} {symbol} {axis} (Ha/Bohr) = {value}"
            )
    print("PySCF gradient norm (Ha/Bohr) =", np.linalg.norm(gradient_pyscf))
    print("PySCF maximum gradient component (Ha/Bohr) =", np.abs(gradient_pyscf).max())
    print(
        "Max gradient difference (PyFock - PySCF) (Ha/Bohr) =",
        gradient_difference.max(),
    )
