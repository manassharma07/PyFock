# Check that PyFock analytical forces converge to PySCF forces as SCF
# convergence is tightened (isolates convergence noise from real errors).
import os
import sys
import numpy as np
from timeit import default_timer as timer

ncores = 4
os.environ["OMP_NUM_THREADS"] = str(ncores)
os.environ["OPENBLAS_NUM_THREADS"] = str(ncores)
os.environ["MKL_NUM_THREADS"] = str(ncores)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(ncores)
os.environ["NUMEXPR_NUM_THREADS"] = str(ncores)

from pyfock import Basis, DFT, DFT_Grad, Mol
from pyscf import dft as pyscf_dft
from pyscf import gto

basis_set_name = "def2-SVP"
auxbasis_name = "def2-universal-jfit"
xyz_filename = "H2O.xyz"
xc_pyfock = "PBE"
xc_pyscf = "101,130"

if len(sys.argv) > 1:
    xyz_filename = sys.argv[1]

mol_pyscf = gto.Mole()
mol_pyscf.atom = xyz_filename
mol_pyscf.basis = basis_set_name
mol_pyscf.cart = False
mol_pyscf.verbose = 0
mol_pyscf.build()

mf = pyscf_dft.rks.RKS(mol_pyscf).density_fit(auxbasis=auxbasis_name)
mf.xc = xc_pyscf
mf.init_guess = "atom"
dm0_pyscf = mf.init_guess_by_atom(mol_pyscf)
mf.conv_tol = 1e-12
mf.max_cycle = 60
mf.grids.level = 3
energy_pyscf = mf.kernel(dm0=dm0_pyscf)
forces_pyscf = -mf.nuc_grad_method().kernel()
pyscf_grids = mf.grids

mol_pyfock = Mol(coordfile=xyz_filename)
basis = Basis(mol_pyfock, {"all": Basis.load(mol=mol_pyfock, basis_name=basis_set_name)})
auxbasis = Basis(mol_pyfock, {"all": Basis.load(mol=mol_pyfock, basis_name=auxbasis_name)})

dft_obj = DFT(mol_pyfock, basis, auxbasis, xc=xc_pyfock, grids=pyscf_grids)
dft_obj.conv_crit = 1e-12
dft_obj.max_itr = 60
dft_obj.ncores = ncores
dft_obj.save_ao_values = True
dft_obj.isDF = True
dft_obj.DF_algo = 10
dft_obj.XC_algo = 2
dft_obj.sortGrids = False
dft_obj.xc_bf_screen = True
dft_obj.threshold_schwarz = 1e-9
dft_obj.strict_schwarz = False
dft_obj.cholesky = True
dft_obj.orthogonalize = True
dft_obj.sao = True
dft_obj.use_gpu = False
dft_obj.use_libxc = False
dft_obj.dmat = dm0_pyscf

energy_pyfock, _ = dft_obj.scf()

grad_obj = DFT_Grad(dft_obj)
res = grad_obj.calculate()
forces_ana = res["forces"]

print("\n\nPySCF energy:", energy_pyscf)
print("PyFock energy:", energy_pyfock)
print("Energy diff:", energy_pyfock - energy_pyscf)
print("\nPySCF forces:\n", np.array2string(forces_pyscf, precision=10))
print("PyFock analytical forces:\n", np.array2string(forces_ana, precision=10))
diff = forces_ana - forces_pyscf
print("\nForce diff:\n", np.array2string(diff, precision=3))
print("Max abs force diff (tight conv):", np.abs(diff).max())
print("\nGradient components translational sum check (should be ~0 for exact, "
      "small for fixed-grid XC):")
for key, val in res["gradient_components"].items():
    print(f"  {key:<22} sum over atoms: {np.array2string(val.sum(axis=0), precision=3)}")
print(f"  {'TOTAL':<22} sum over atoms: {np.array2string(res['gradient'].sum(axis=0), precision=3)}")
