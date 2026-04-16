import os
import numpy as np

from timeit import default_timer as timer

ncores = 4
os.environ["OMP_NUM_THREADS"] = str(ncores)
os.environ["OPENBLAS_NUM_THREADS"] = str(ncores)
os.environ["MKL_NUM_THREADS"] = str(ncores)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(ncores)
os.environ["NUMEXPR_NUM_THREADS"] = str(ncores)
os.environ["PYSCF_MAX_MEMORY"] = str(25000)

from pyfock import Basis
from pyfock import DFT
from pyfock import DFT_NumGrad
from pyfock import Mol
from pyfock import Utils

from pyscf import dft as pyscf_dft
from pyscf import gto


Utils.print_sys_info()

basis_set_name = "def2-SVP"
auxbasis_name = "def2-universal-jfit"
xyz_filename = "H2O.xyz"
xc_pyfock = "PBE"
xc_pyscf = "101,130"


print("\nPySCF numerical-gradient reference\n")
mol_pyscf = gto.Mole()
mol_pyscf.atom = xyz_filename
mol_pyscf.basis = basis_set_name
mol_pyscf.cart = False
mol_pyscf.verbose = 4
mol_pyscf.max_memory = 5000
mol_pyscf.build()

start = timer()
mf = pyscf_dft.rks.RKS(mol_pyscf).density_fit(auxbasis=auxbasis_name)
mf.xc = xc_pyscf
mf.direct_scf = False
mf.init_guess = "atom"
dm0_pyscf = mf.init_guess_by_atom(mol_pyscf)
mf.conv_tol = 1e-7
mf.max_cycle = 35
mf.grids.level = 3
energy_pyscf = mf.kernel(dm0=dm0_pyscf)
grad_pyscf = mf.nuc_grad_method().kernel()
forces_pyscf = -grad_pyscf
time_pyscf = timer() - start
pyscf_grids = mf.grids

print("PySCF energy (Ha):", energy_pyscf)
print("PySCF forces (Ha/Bohr):")
print(np.array2string(forces_pyscf, precision=8, suppress_small=False))
print("PySCF total time (s):", time_pyscf)


print("\nPyFock numerical gradients\n")
mol_pyfock = Mol(coordfile=xyz_filename)
basis = Basis(mol_pyfock, {"all": Basis.load(mol=mol_pyfock, basis_name=basis_set_name)})
auxbasis = Basis(
    mol_pyfock,
    {"all": Basis.load(mol=mol_pyfock, basis_name=auxbasis_name)},
)

dft_obj = DFT(mol_pyfock, basis, auxbasis, xc=xc_pyfock, grids=pyscf_grids)
dft_obj.conv_crit = 1e-7
dft_obj.max_itr = 35
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

start = timer()
dft_obj.scf()
# grad_obj = DFT_NumGrad(
#     dft_obj, step_size=1.0e-3, step_unit="bohr", method="forward", use_fixed_grids=True
# )
grad_obj = DFT_NumGrad(
    dft_obj, step_size=1.0e-3, step_unit="bohr", method="central", use_fixed_grids=True
)
results = grad_obj.calculate()
time_pyfock = timer() - start

forces_pyfock = results["forces"]
print("PyFock energy (Ha):", results["energy"])
print("PyFock forces (Ha/Bohr):")
print(np.array2string(forces_pyfock, precision=8, suppress_small=False))
print("PyFock total time (s):", time_pyfock)

force_diff = forces_pyfock - forces_pyscf
print("\nForce difference (PyFock - PySCF) (Ha/Bohr):")
print(np.array2string(force_diff, precision=8, suppress_small=False))
print("Max abs force diff (Ha/Bohr):", np.max(np.abs(force_diff)))
