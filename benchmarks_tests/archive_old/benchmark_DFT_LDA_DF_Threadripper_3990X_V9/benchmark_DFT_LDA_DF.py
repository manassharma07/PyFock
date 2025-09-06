import os
import platform
import psutil
# import numba
# numba.config.THREADING_LAYER='omp'
# Set the number of threads/cores to be used by PyFock and PySCF
ncores = 16
os.environ['OMP_NUM_THREADS'] = str(ncores)
os.environ["OPENBLAS_NUM_THREADS"] = str(ncores) # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = str(ncores) # export MKL_NUM_THREADS=4
os.environ["VECLIB_MAXIMUM_THREADS"] = str(ncores) # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = str(ncores) # export NUMEXPR_NUM_THREADS=4
# Set the max memory for PySCF
os.environ["PYSCF_MAX_MEMORY"] = str(125000) 

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

from pyscf import gto, dft, df

#DFT SCF benchmark and comparison with PySCF
#Benchmarking and performance assessment and comparison using various techniques and different softwares

#LDA
funcx = 1
funcc = 7
#PBE
#funcx = 101
#funcc = 130
funcidcrysx = [funcx, funcc]
funcidpyscf = str(funcx)+','+str(funcc)

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
# auxbasis_name = 'sto-3g'
# auxbasis_name = 'def2-SVP'
# auxbasis_name = '6-31G'

# xyzFilename = 'Benzene-Fulvene_Dimer.xyz'
# xyzFilename = 'Adenine-Thymine.xyz'
# xyzFilename = 'Zn.xyz'
# xyzFilename = 'Zn_dimer.xyz'
# xyzFilename = 'TPP.xyz'
# xyzFilename = 'Zn_TPP.xyz'
# xyzFilename = 'h2o.xyz'
# xyzFilename = 'Ethane.xyz'
# xyzFilename = 'Cholesterol.xyz'
# xyzFilename = 'Serotonin.xyz'
# xyzFilename = 'Decane_C10H22.xyz'
# xyzFilename = 'Icosane_C20H42.xyz'
# xyzFilename = 'Tetracontane_C40H82.xyz'
# xyzFilename = 'Pentacontane_C50H102.xyz'
# xyzFilename = 'Octacontane_C80H162.xyz'
# xyzFilename = 'Hectane_C100H202.xyz'
xyzFilename = 'Icosahectane_C120H242.xyz'

# ---------PySCF---------------
#Comparison with PySCF
molPySCF = gto.Mole()
molPySCF.atom = xyzFilename
molPySCF.basis = basis_set_name
molPySCF.cart = True
molPySCF.verbose = 4
molPySCF.max_memory=25000
# molPySCF.incore_anyway = True # Keeps the PySCF ERI integrals incore
molPySCF.build()
#print(molPySCF.cart_labels())

print('\n\nPySCF Results\n\n')
start=timer()
mf = dft.rks.RKS(molPySCF).density_fit(auxbasis=auxbasis_name)
mf.xc = funcidpyscf
# mf.verbose = 4
mf.direct_scf = False
# mf.with_df.max_memory = 25000
# dmat_init = mf.init_guess_by_1e(molPySCF)
# dmat_init = mf.init_guess_by_huckel(molPySCF)
mf.init_guess = 'minao'
dmat_init = mf.init_guess_by_minao(molPySCF)
# mf.init_guess = 'atom'
# dmat_init = mf.init_guess_by_atom(molPySCF)
mf.max_cycle = 30
mf.conv_tol = 1e-7
mf.grids.level = 5
print('begin df build')
start_df_pyscf=timer()
# mf.with_df.build()
duration_df_pyscf = timer()- start_df_pyscf
print('PySCF df time: ', duration_df_pyscf)
print('end df build')
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

#--------------------CrysX --------------------------

#Initialize a Mol object with somewhat large geometry
molCrysX = Mol(coordfile=xyzFilename)
print('\n\nNatoms :',molCrysX.natoms)

#Initialize a Basis object with a very large basis set
basis = Basis(molCrysX, {'all':Basis.load(mol=molCrysX, basis_name=basis_set_name)})
print('\n\nNAO :',basis.bfs_nao)

auxbasis = Basis(molCrysX, {'all':Basis.load(mol=molCrysX, basis_name=auxbasis_name)})
print('\n\naux NAO :',auxbasis.bfs_nao)

dftObj = DFT(molCrysX, basis,xc=funcidcrysx)
# GPU acceleration
# dftObj.use_gpu = True
# Using PySCF grids to compare the energies
energyCrysX, dmat = dftObj.scf(max_itr=30, ncores=ncores, dmat=dmat_init, conv_crit=1.0E-7, grids=pyscfGrids, isDF=True, auxbasis=auxbasis, rys=True, DF_algo=6, blocksize=5000, XC_algo=2, debug=False, sortGrids=False, save_ao_values=True, xc_bf_screen=True,threshold_schwarz=1e-9)
# Using CrysX grids 
# To get the same energies as PySCF (level=5) upto 1e-7 au, use the following settings
# radial_precision=1.0e-13
# level=3
# pruning by density with threshold = 1e-011
# alpha_min and alpha_max corresponding to QZVP
# energyCrysX, dmat = dftObj.scf(max_itr=30, ncores=ncores, dmat=dmat_init, grids=None, gridsLevel=3, isDF=True, auxbasis=auxbasis, rys=True, DF_algo=6, blocksize=5000, XC_algo=2, debug=False, sortGrids=False, save_ao_values=True, xc_bf_screen=True,threshold_schwarz=1e-9)


print('Energy diff (PySCF-CrysX)', abs(energyCrysX-energyPyscf))
