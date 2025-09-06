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
#numba.config.THREADING_LAYER='tbb'
# Set the number of threads/cores to be used by PyFock and PySCF
ncores = 4
os.environ['OMP_NUM_THREADS'] = str(ncores)
os.environ["OPENBLAS_NUM_THREADS"] = str(ncores) # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = str(ncores) # export MKL_NUM_THREADS=4
os.environ["VECLIB_MAXIMUM_THREADS"] = str(ncores) # export VECLIB_MAXIMUM_THREADS=4
# os.environ["NUMEXPR_NUM_THREADS"] = str(ncores) # export NUMEXPR_NUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = str(ncores) # export NUMEXPR_NUM_THREADS=1
# Set the max memory for PySCF
os.environ["PYSCF_MAX_MEMORY"] = str(1500000) 

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


basis_set_name = 'def2-SVP'
# basis_set_name = 'def2-TZVP'
# basis_set_name = 'cc-pVDZ'


auxbasis_name = 'def2-universal-jfit'

# List of all molecules to calculate
xyz_files = [
    '../../Structures/water_cluster_1.xyz',
    '../../Structures/water_cluster_5.xyz',
    '../../Structures/water_cluster_10.xyz',
    '../../Structures/water_cluster_20.xyz',
    '../../Structures/water_cluster_32.xyz',
    '../../Structures/water_cluster_47.xyz',
    '../../Structures/water_cluster_76.xyz',
    '../../Structures/water_cluster_100.xyz',
    '../../Structures/water_cluster_139.xyz',
]


# Loop through all molecules
for xyzFilename in xyz_files:
    print(f"\n{'='*60}")
    print(f"Processing: {xyzFilename}")
    print(f"{'='*60}")

    # ---------PySCF---------------
    #Comparison with PySCF
    molPySCF = gto.Mole()
    molPySCF.atom = xyzFilename
    molPySCF.basis = basis_set_name
    molPySCF.cart = True
    molPySCF.verbose = 5
    molPySCF.max_memory=1500000
    molPySCF.build()

    print('\n\nPySCF Results\n\n')
    mf = dft.rks.RKS(molPySCF).density_fit(auxbasis=auxbasis_name)
    mf.xc = funcidpyscf
    mf.verbose = 5
    mf.direct_scf = False
    mf.init_guess = 'minao'
    dmat_init = mf.init_guess_by_minao(molPySCF)
    mf.max_cycle = 50
    mf.conv_tol = 1e-7
    mf.grids.level = 3
    start=timer()
    energyPyscf = mf.kernel(dm0=dmat_init)
    print('Nuc-Nuc PySCF= ', molPySCF.energy_nuc())
    print('One electron integrals energy',mf.scf_summary['e1'])
    print('Coulomb energy ',mf.scf_summary['coul'])
    print('EXC ',mf.scf_summary['exc'])
    duration = timer()-start
    print('PySCF time: ', duration)
