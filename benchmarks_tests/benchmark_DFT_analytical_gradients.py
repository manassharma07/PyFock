# Benchmark of PyFock analytical DFT forces against PySCF analytical forces
# and PyFock numerical (finite-difference) forces.
#
# Based on benchmark_DFT_LDA_gradients.py.
#
# The SCF is run with the production settings (isDF=True, DF_algo=10,
# XC_algo=2, use_gpu=False) and the analytical gradients are obtained with
# pyfock.DFT_Grad which evaluates:
#   - one-electron gradients (overlap/kinetic/nuclear)
#   - density-fitted Coulomb gradients (contracted 3c2e + 2c2e derivatives)
#   - XC gradients (LDA/GGA) on the same fixed grid as the SCF
import os


from timeit import default_timer as timer

ncores = 4
os.environ["OMP_NUM_THREADS"] = str(ncores)
os.environ["OPENBLAS_NUM_THREADS"] = str(ncores)
os.environ["MKL_NUM_THREADS"] = str(ncores)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(ncores)
os.environ["NUMEXPR_NUM_THREADS"] = str(ncores)
os.environ["PYSCF_MAX_MEMORY"] = str(25000)

import numpy as np

from pyfock import Basis
from pyfock import DFT
from pyfock import DFT_NumGrad
from pyfock import DFT_Grad
from pyfock import Mol
from pyfock import Utils

from pyscf import dft as pyscf_dft
from pyscf import gto


Utils.print_sys_info()

import sys

# basis_set_name = 'sto-2g'
# basis_set_name = 'sto-3g'
# basis_set_name = 'sto-6g'
# basis_set_name = '6-31G'
basis_set_name = 'def2-SVP'
# basis_set_name = 'def2-SVPD'
# basis_set_name = 'def2-TZVP'
# basis_set_name = 'def2-QZVP'
# basis_set_name = 'def2-TZVPP'
# basis_set_name = 'def2-QZVPP'
# basis_set_name = 'def2-TZVPD'
# basis_set_name = 'def2-QZVPD'
# basis_set_name = 'def2-TZVPPD'
# basis_set_name = 'def2-QZVPPD'
# basis_set_name = 'cc-pVDZ'
# basis_set_name = 'ano-rcc'

auxbasis_name = 'def2-universal-jfit'
# auxbasis_name = 'def2-universal-jkfit'
# auxbasis_name = 'def2-TZVP'
# auxbasis_name = 'sto-3g'
# auxbasis_name = 'def2-SVP'
# auxbasis_name = '6-31G'

# xyzFilename = 'Benzene-Fulvene_Dimer.xyz'
# xyzFilename = 'Adenine-Thymine.xyz'
# xyzFilename = 'Zn.xyz'
# xyzFilename = 'Zn_dimer.xyz'
# xyzFilename = 'TPP.xyz'
# xyzFilename = 'Zn_TPP.xyz'
xyzFilename = 'H2O.xyz'

# xyzFilename = 'Caffeine.xyz'
# xyzFilename = 'Serotonin.xyz'
# xyzFilename = 'Cholesterol.xyz'
# xyzFilename = 'C60.xyz'
# xyzFilename = 'Taxol.xyz'
# xyzFilename = 'Valinomycin.xyz'
# xyzFilename = 'Olestra.xyz'
# xyzFilename = 'Ubiquitin.xyz'

### 1D Carbon Alkanes
# xyzFilename = 'Decane_C10H22.xyz'
# xyzFilename = 'Icosane_C20H42.xyz'
# xyzFilename = 'Tetracontane_C40H82.xyz'
# xyzFilename = 'Pentacontane_C50H102.xyz'
# xyzFilename = 'Octacontane_C80H162.xyz'
# xyzFilename = 'Hectane_C100H202.xyz'
# xyzFilename = 'Icosahectane_C120H242.xyz'

### 2D Carbon
# xyzFilename = 'Graphene_C16.xyz'
# xyzFilename = 'Graphene_C76.xyz'
# xyzFilename = 'Graphene_C102.xyz'
# xyzFilename = 'Graphene_C184.xyz'
# xyzFilename = 'Graphene_C210.xyz'
# xyzFilename = 'Graphene_C294.xyz'

### 3d Carbon Fullerenes
# xyzFilename = 'C60.xyz'
# xyzFilename = 'C70.xyz'
# xyzFilename = 'Graphene_C102.xyz'
# xyzFilename = 'Graphene_C184.xyz'
# xyzFilename = 'Graphene_C210.xyz'
# xyzFilename = 'Graphene_C294.xyz'

xc_pyfock = "PBE"
xc_pyscf = "PBE" # "101,130" is the PySCF code for PBE, "1,7" is LDA (VWN)
# xc_pyfock = "LDA"
# xc_pyscf = "1,7"

run_numerical = False  # Finite-difference forces need 6*natoms SCF runs

# Optional command line overrides:
#   python3 benchmark_DFT_analytical_gradients.py [xyz_file] [LDA|PBE] [skip_numerical]
if len(sys.argv) > 1:
    xyzFilename = sys.argv[1]
if len(sys.argv) > 2:
    if sys.argv[2].upper() == "LDA":
        xc_pyfock = "LDA"
        xc_pyscf = "1,7"
    elif sys.argv[2].upper() == "PBE":
        xc_pyfock = "PBE"
        xc_pyscf = "101,130"
if len(sys.argv) > 3 and sys.argv[3] == "skip_numerical":
    run_numerical = False


print("\n=================== PySCF (analytical forces) ===================\n")
mol_pyscf = gto.Mole()
mol_pyscf.atom = xyzFilename
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
mf.conv_tol = 1e-9
mf.max_cycle = 35
mf.grids.level = 3
energy_pyscf = mf.kernel(dm0=dm0_pyscf)
time_pyscf_scf = timer() - start

start = timer()
grad_pyscf = mf.nuc_grad_method().kernel()
forces_pyscf = -grad_pyscf
time_pyscf_grad = timer() - start
pyscf_grids = mf.grids

print("PySCF energy (Ha):", energy_pyscf)
print("PySCF forces (Ha/Bohr):")
print(np.array2string(forces_pyscf, precision=8, suppress_small=False))
print("PySCF SCF time (s):", time_pyscf_scf)
print("PySCF gradient time (s):", time_pyscf_grad)


print("\n=================== PyFock SCF ===================\n")
mol_pyfock = Mol(coordfile=xyzFilename)
basis = Basis(mol_pyfock, {"all": Basis.load(mol=mol_pyfock, basis_name=basis_set_name)})
auxbasis = Basis(
    mol_pyfock,
    {"all": Basis.load(mol=mol_pyfock, basis_name=auxbasis_name)},
)

dft_obj = DFT(mol_pyfock, basis, auxbasis, xc=xc_pyfock, grids=pyscf_grids)
dft_obj.conv_crit = 1e-9
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
energy_pyfock, _ = dft_obj.scf()
time_pyfock_scf = timer() - start
print("PyFock energy (Ha):", energy_pyfock)
print("PyFock SCF time (s):", time_pyfock_scf)
print("PyFock - PySCF energy difference (Ha):", energy_pyfock - energy_pyscf)


print("\n=================== PyFock analytical forces ===================\n")
# First call includes Numba JIT compilation of the gradient kernels
# (cached to disk, so subsequent runs/processes skip it).
start = timer()
grad_obj = DFT_Grad(dft_obj)
results_ana = grad_obj.calculate()
time_pyfock_ana = timer() - start

forces_pyfock_ana = results_ana["forces"]
print("PyFock energy (Ha):", results_ana["energy"])
print("PyFock analytical forces (Ha/Bohr):")
print(np.array2string(forces_pyfock_ana, precision=8, suppress_small=False))
print("PyFock analytical gradient time:", time_pyfock_ana)

diff_ana_pyscf = forces_pyfock_ana - forces_pyscf
print("\nForce difference (PyFock analytical - PySCF) (Ha/Bohr):")
print(np.array2string(diff_ana_pyscf, precision=8, suppress_small=False))
print("Max abs force diff (PyFock analytical vs PySCF) (Ha/Bohr):", np.max(np.abs(diff_ana_pyscf)))


if run_numerical:
    print("\n=================== PyFock numerical forces ===================\n")
    start = timer()
    numgrad_obj = DFT_NumGrad(
        dft_obj, step_size=1.0e-3, step_unit="bohr", method="central", use_fixed_grids=True
    )
    results_num = numgrad_obj.calculate()
    time_pyfock_num = timer() - start

    forces_pyfock_num = results_num["forces"]
    print("PyFock numerical forces (Ha/Bohr):")
    print(np.array2string(forces_pyfock_num, precision=8, suppress_small=False))
    print("PyFock numerical gradient time (s):", time_pyfock_num)

    diff_num_pyscf = forces_pyfock_num - forces_pyscf
    print("\nForce difference (PyFock numerical - PySCF) (Ha/Bohr):")
    print(np.array2string(diff_num_pyscf, precision=8, suppress_small=False))
    print("Max abs force diff (PyFock numerical vs PySCF) (Ha/Bohr):", np.max(np.abs(diff_num_pyscf)))

    diff_ana_num = forces_pyfock_ana - forces_pyfock_num
    print("\nForce difference (PyFock analytical - PyFock numerical) (Ha/Bohr):")
    print(np.array2string(diff_ana_num, precision=8, suppress_small=False))
    print("Max abs force diff (PyFock analytical vs numerical) (Ha/Bohr):", np.max(np.abs(diff_ana_num)))


print("\n=================== Summary ===================\n")
print(f"{'Method':<38}{'Max |dF| vs PySCF (Ha/Bohr)':>30}{'Time (s)':>12}")
print(f"{'PySCF analytical gradient':<38}{'-':>30}{time_pyscf_grad:>12.3f}")
print(f"{'PyFock analytical gradient':<38}{np.max(np.abs(diff_ana_pyscf)):>30.3e}{time_pyfock_ana:>12.3f}")
if run_numerical:
    print(f"{'PyFock numerical gradient':<38}{np.max(np.abs(diff_num_pyscf)):>30.3e}{time_pyfock_num:>12.3f}")
