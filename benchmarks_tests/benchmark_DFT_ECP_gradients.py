# Validate PyFock analytical DFT forces for systems with effective core
# potentials (ECPs) against PySCF analytical forces and PyFock numerical
# (finite-difference) forces.
#
# Default systems: AgCl and the Cd dimer with def2-SVP (which carries ECPs for
# Ag and Cd). Usage:
#     python3 benchmark_DFT_ECP_gradients.py [AgCl.xyz|Cd_dimer.xyz] [skip_numerical]
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
os.environ["PYSCF_MAX_MEMORY"] = str(25000)

from pyfock import Basis, DFT, DFT_Grad, DFT_NumGrad, Mol

from pyscf import dft as pyscf_dft
from pyscf import gto

basis_set_name = "def2-SVP"
auxbasis_name = "def2-universal-jfit"
xc_pyfock = "PBE"
xc_pyscf = "PBE"

xyz_filename = "AgCl.xyz"
run_numerical = True
if len(sys.argv) > 1:
    xyz_filename = sys.argv[1]
if len(sys.argv) > 2 and sys.argv[2] == "skip_numerical":
    run_numerical = False

print("=" * 70)
print(f"ECP gradient benchmark: {xyz_filename}  {basis_set_name}  {xc_pyfock}")
print("=" * 70)

# ---------------- PySCF (analytical forces, with ECP) ----------------
mol_pyscf = gto.Mole()
mol_pyscf.atom = xyz_filename
mol_pyscf.basis = basis_set_name
mol_pyscf.ecp = basis_set_name          # def2-SVP ECP for heavy atoms
mol_pyscf.cart = False
mol_pyscf.verbose = 0
mol_pyscf.max_memory = 5000
mol_pyscf.build()
print("Core electrons replaced by ECP:", mol_pyscf.nelec, "valence electrons total")

mf = pyscf_dft.rks.RKS(mol_pyscf).density_fit(auxbasis=auxbasis_name)
mf.xc = xc_pyscf
mf.init_guess = "minao"
dm0 = mf.init_guess_by_minao(mol_pyscf)
mf.conv_tol = 1e-10
mf.max_cycle = 50
mf.grids.level = 3
energy_pyscf = mf.kernel(dm0=dm0)
start = timer()
forces_pyscf = -mf.nuc_grad_method().kernel()
time_pyscf_grad = timer() - start
pyscf_grids = mf.grids

print("\nPySCF energy (Ha):", energy_pyscf)
print("PySCF forces (Ha/Bohr):\n", np.array2string(forces_pyscf, precision=8))
print("PySCF gradient time (s):", round(time_pyscf_grad, 3))

# ---------------- PyFock SCF (with ECP) ----------------
mol_pyfock = Mol(coordfile=xyz_filename)
basis = Basis(mol_pyfock, {"all": Basis.load(mol=mol_pyfock, basis_name=basis_set_name)})
auxbasis = Basis(mol_pyfock, {"all": Basis.load(mol=mol_pyfock, basis_name=auxbasis_name)})
print("\nPyFock has_ecp:", basis.has_ecp, " core electrons:", basis.ecp_total_core_electrons)

dft_obj = DFT(mol_pyfock, basis, auxbasis, xc=xc_pyfock, grids=pyscf_grids)
dft_obj.conv_crit = 1e-10
dft_obj.max_itr = 50
dft_obj.ncores = ncores
dft_obj.save_ao_values = True
dft_obj.isDF = True
dft_obj.DF_algo = 10
dft_obj.XC_algo = 2
dft_obj.xc_bf_screen = True
dft_obj.threshold_schwarz = 1e-9
dft_obj.strict_schwarz = False
dft_obj.cholesky = True
dft_obj.orthogonalize = True
dft_obj.sao = True
dft_obj.use_gpu = False
dft_obj.use_libxc = False
dft_obj.dmat = dm0
energy_pyfock, _ = dft_obj.scf()
print("PyFock energy (Ha):", energy_pyfock)
print("PyFock - PySCF energy diff (Ha):", energy_pyfock - energy_pyscf)

# ---------------- PyFock analytical forces (incl. ECP) ----------------
grad_obj = DFT_Grad(dft_obj, verbose=True)
results = grad_obj.calculate()   # first call (incl. any JIT)
start = timer()
results = grad_obj.calculate()   # warm
time_ana = timer() - start
forces_ana = results["forces"]

print("\nPyFock analytical forces (Ha/Bohr):\n", np.array2string(forces_ana, precision=8))
print("PyFock analytical gradient time, warm (s):", round(time_ana, 3))
print("ECP gradient component (Ha/Bohr):\n",
      np.array2string(results["gradient_components"]["ecp"], precision=8))

diff_ana_pyscf = forces_ana - forces_pyscf
print("\nMax |PyFock analytical - PySCF| (Ha/Bohr):", np.max(np.abs(diff_ana_pyscf)))

# ---------------- PyFock numerical forces ----------------
if run_numerical:
    numgrad = DFT_NumGrad(dft_obj, step_size=1.0e-3, step_unit="bohr",
                          method="central", use_fixed_grids=True, verbose=False)
    forces_num = numgrad.calculate()["forces"]
    print("\nPyFock numerical forces (Ha/Bohr):\n", np.array2string(forces_num, precision=8))
    print("Max |PyFock analytical - numerical| (Ha/Bohr):", np.max(np.abs(forces_ana - forces_num)))
    print("Max |PyFock numerical  - PySCF|     (Ha/Bohr):", np.max(np.abs(forces_num - forces_pyscf)))

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print(f"{'Max |dF| analytical vs PySCF (Ha/Bohr)':<45}{np.max(np.abs(diff_ana_pyscf)):.3e}")
if run_numerical:
    print(f"{'Max |dF| analytical vs numerical (Ha/Bohr)':<45}{np.max(np.abs(forces_ana - forces_num)):.3e}")
