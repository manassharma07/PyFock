####### NOTE: The scipy.linalg library appears to be using double the number of threads supplied for some reason.
####### To avoid such issues messing up the benchmarks, the benchmark should be run as 'taskset --cpu-list 0-3 python3 benchmark_DFT_LDA_DF.py'
####### This way one can set the number of CPUs seen by the python process and the benchmark would be much more reliable.
####### Furthermore, to confirm the CPU and memory usage throughout the whole process, one can profilie it using  
####### psrecord 13447 --interval 1 --duration 120 --plot 13447.png
#######
####### This may be required in some cases when using GPU on WSL
####### export NUMBA_CUDA_DRIVER="/usr/lib/wsl/lib/libcuda.so.1"

import os
import platform
import psutil
import numba
numba.config.THREADING_LAYER='omp'
# Set the number of threads/cores to be used by PyFock and PySCF
ncores = 4
os.environ['OMP_NUM_THREADS'] = str(ncores)
os.environ["OPENBLAS_NUM_THREADS"] = str(ncores) # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = str(ncores) # export MKL_NUM_THREADS=4
os.environ["VECLIB_MAXIMUM_THREADS"] = str(ncores) # export VECLIB_MAXIMUM_THREADS=4
# os.environ["NUMEXPR_NUM_THREADS"] = str(ncores) # export NUMEXPR_NUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = str(ncores) # export NUMEXPR_NUM_THREADS=1
# Set the max memory for PySCF
os.environ["PYSCF_MAX_MEMORY"] = str(225000) 

# Print system information 
from pyfock import Utils

Utils.print_sys_info()

# Check if the environment variables are properly set
print("Number of cores being actually used/requested for the benchmark:", ncores)
print('Confirming that the environment variables are properly set...')
print('OMP_NUM_THREADS =', os.environ.get('OMP_NUM_THREADS', None))
print('OPENBLAS_NUM_THREADS =', os.environ.get('OPENBLAS_NUM_THREADS', None))
print('MKL_NUM_THREADS =', os.environ.get('MKL_NUM_THREADS', None))
print('VECLIB_MAXIMUM_THREADS =', os.environ.get('VECLIB_MAXIMUM_THREADS', None))
print('NUMEXPR_NUM_THREADS =', os.environ.get('NUMEXPR_NUM_THREADS', None))
print('PYSCF_MAX_MEMORY =', os.environ.get('PYSCF_MAX_MEMORY', None))


# Run your tasks here
from pyfock import Basis
from pyfock import Mol
from pyfock import Integrals
from pyfock import DFT
from timeit import default_timer as timer
import numpy as np
import scipy

from pyscf import gto, dft, df, scf
from gpu4pyscf import dft as dft_gpu

#DFT SCF benchmark and comparison with PySCF
#Benchmarking and performance assessment and comparison using various techniques and different softwares

# LDA_X LDA_C_VWN 
# funcx = 1
# funcc = 7

# LDA_X LDA_C_PW 
# funcx = 1
# funcc = 12

# LDA_X LDA_C_PW_MOD 
# funcx = 1
# funcc = 13

# GGA_X_PBE, GGA_C_PBE (PBE)
funcx = 101
funcc = 130

# GGA_X_B88, GGA_C_LYP (BLYP)
# funcx = 106
# funcc = 131

funcidcrysx = [funcx, funcc]
funcidpyscf = str(funcx)+','+str(funcc)

# basis_set_name = 'sto-2g'
# basis_set_name = 'sto-3g'
# basis_set_name = 'sto-6g'
# basis_set_name = '6-31G'
# basis_set_name = 'def2-SVP'
# basis_set_name = 'def2-SVPD'
# basis_set_name = 'def2-TZVP'
# basis_set_name = 'def2-QZVP'
# basis_set_name = 'def2-TZVPP'
# basis_set_name = 'def2-QZVPP'
# basis_set_name = 'def2-TZVPD'
basis_set_name = 'def2-QZVPD'
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
# xyzFilename = 'H2O.xyz'

# xyzFilename = 'Caffeine.xyz'
# xyzFilename = 'Serotonin.xyz'
# xyzFilename = 'Cholesterol.xyz'
# xyzFilename = 'C60.xyz'
# xyzFilename = 'Taxol.xyz'
# xyzFilename = 'Valinomycin.xyz'
# xyzFilename = 'Olestra.xyz'
# xyzFilename = 'Ubiquitin.xyz'

### 1D Carbon Alkanes
# xyzFilename = 'Ethane.xyz'
xyzFilename = 'Decane_C10H22.xyz'
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


# ---------PySCF---------------
#Comparison with PySCF
molPySCF = gto.Mole()
molPySCF.atom = xyzFilename
molPySCF.basis = basis_set_name
molPySCF.cart = False
molPySCF.verbose = 4
molPySCF.max_memory=25000
# molPySCF.incore_anyway = True # Keeps the PySCF ERI integrals incore
molPySCF.build()
#print(molPySCF.cart_labels())

print('\n\nPySCF Results\n\n')
start=timer()
mf = dft_gpu.rks.RKS(molPySCF).density_fit(auxbasis=auxbasis_name)
# mf = scf.RHF(molPySCF).density_fit(auxbasis=auxbasis_name)
mf.xc = funcidpyscf
mf.verbose = 6
# dmat_init = mf.init_guess_by_1e(molPySCF)
# dmat_init = mf.init_guess_by_huckel(molPySCF)
mf.init_guess = 'minao'
dmat_init = mf.init_guess_by_minao(molPySCF)
# mf.init_guess = 'atom'
# dmat_init = mf.init_guess_by_atom(molPySCF)
mf.max_cycle = 1
mf.conv_tol = 1e-7
mf.grids.level = 5
energyPyscf = mf.kernel(dm0=dmat_init)
print('Nuc-Nuc PySCF= ', molPySCF.energy_nuc())
print('One electron integrals energy',mf.scf_summary['e1'])
print('Coulomb energy ',mf.scf_summary['coul'])
print('EXC ',mf.scf_summary['exc'])
duration = timer()-start
print('PySCF time: ', duration)
pyscfGrids = mf.grids
print('PySCF Grid Size: ', pyscfGrids.weights.shape)
mf = 0#None
import cupy as cp
cp._default_memory_pool.free_all_blocks()

# Get an initial guess for dmat from PySCF using CAO basis (Need to use CPU pyscf for this as GPU version oesnt support SAO yet)
molPySCF.cart = True
molPySCF.build()
mf = dft_gpu.rks.RKS(molPySCF).density_fit(auxbasis=auxbasis_name)
mf.init_guess = 'minao'
dmat_init = mf.init_guess_by_minao(molPySCF)

import psutil

# Get memory information
memory_info = psutil.virtual_memory()

# Convert bytes to human-readable format
used_memory = psutil._common.bytes2human(memory_info.used)


# If you want to print in a more human-readable format, you can use psutil's utility function
print(f"Currently Used memory: {used_memory}")
#--------------------CrysX --------------------------

#Initialize a Mol object with somewhat large geometry
molCrysX = Mol(coordfile=xyzFilename)
print('\n\nNatoms :',molCrysX.natoms)
# print(molCrysX.coordsBohrs)

#Initialize a Basis object with a very large basis set
basis = Basis(molCrysX, {'all':Basis.load(mol=molCrysX, basis_name=basis_set_name)})
print('\n\nNAO :',basis.bfs_nao)

auxbasis = Basis(molCrysX, {'all':Basis.load(mol=molCrysX, basis_name=auxbasis_name)})
print('\n\naux NAO :',auxbasis.bfs_nao)

dftObj = DFT(molCrysX, basis, xc=funcidcrysx)
# GPU acceleration
dftObj.use_gpu = True
dftObj.keep_ao_in_gpu = False
dftObj.use_libxc = False
dftObj.n_streams = 1 # Changing this to anything other than 1 won't make any difference 
dftObj.n_gpus = 1 # Specify the number of GPUs
dftObj.free_gpu_mem = True
dftObj.threads_x = 32
dftObj.threads_y = 32
dftObj.dynamic_precision = False
dftObj.keep_ints3c2e_in_gpu = True
# SAO or CAO basis
dftObj.sao = False
# print(dmat_init)
# Using PySCF grids to compare the energies
energyCrysX, dmat = dftObj.scf(max_itr=35, ncores=ncores, dmat=dmat_init, conv_crit=1.0E-7, grids=pyscfGrids, \
                               isDF=True, auxbasis=auxbasis, rys=True, DF_algo=10, blocksize=20480, XC_algo=3, debug=False, \
                                sortGrids=False, save_ao_values=False, xc_bf_screen=True, threshold_schwarz=1e-9, \
                                strict_schwarz=False, cholesky=True, orthogonalize=True)
# print(dmat)

# Using CrysX grids 
# To get the same energies as PySCF (level=5) upto 1e-7 au, use the following settings
# radial_precision=1.0e-13
# level=3
# pruning by density with threshold = 1e-011
# alpha_min and alpha_max corresponding to QZVP
# energyCrysX, dmat = dftObj.scf(max_itr=30, ncores=ncores, dmat=dmat_init, grids=None, gridsLevel=3, isDF=True, auxbasis=auxbasis,
#                             rys=True, DF_algo=6, blocksize=5000, XC_algo=2, debug=False, sortGrids=False, save_ao_values=True,
#                             xc_bf_screen=True,threshold_schwarz=1e-9)


print('Energy diff (PySCF-CrysX)', abs(energyCrysX-energyPyscf[0]))

#Print package versions
import joblib
import scipy
import numba
import threadpoolctl
import opt_einsum
import pylibxc
import llvmlite 
import cupy
import numexpr
import pyscf
print('\n\n\n Package versions')
print('pyscf version', pyscf.__version__)
# print('psi4 version', psi4.__version__)
print('np version', np.__version__)
print('joblib version', joblib.__version__)
print('numba version', numba.__version__)
print('threadpoolctl version', threadpoolctl.__version__)
print('opt_einsum version', opt_einsum.__version__)
# print('pylibxc version', pylibxc.__version__)
print('llvmlite version', llvmlite.__version__)
print('cupy version', cupy.__version__)
print('numexpr version', numexpr.__version__)
print('scipy version', scipy.__version__)
