from pyfock import Basis
from pyfock import Mol
from pyfock import Integrals
from timeit import default_timer as timer

import numpy as np
import numba 
import os

ncores = 4
bench_GPU = True

numba.set_num_threads(ncores)
os.environ['OMP_NUM_THREADS'] = str(ncores)
os.environ["OPENBLAS_NUM_THREADS"] = str(ncores) # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = str(ncores) # export MKL_NUM_THREADS=4
os.environ["VECLIB_MAXIMUM_THREADS"] = str(ncores) # export VECLIB_MAXIMUM_THREADS=4
# os.environ["NUMEXPR_NUM_THREADS"] = str(ncores) # export NUMEXPR_NUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = str(ncores) # export NUMEXPR_NUM_THREADS=1


# 3c2e ERI

# basis_set_name = '6-31G'
# basis_set_name = 'sto-2g'
# basis_set_name = 'sto-3g'
# basis_set_name = 'sto-6g'
basis_set_name = 'def2-SVP'
# basis_set_name = 'def2-DZVP'
# basis_set_name = 'def2-TZVP'
# basis_set_name = 'def2-QZVP'
# basis_set_name = 'def2-TZVPPD'
# basis_set_name = 'def2-QZVPPD'
# basis_set_name = 'ano-rcc'

# xyzFilename = 'Benzene-Fulvene_Dimer.xyz'
# xyzFilename = 'H2O.xyz'
# xyzFilename = 'Zn_dimer.xyz'
xyzFilename = 'Ethane.xyz'
# xyzFilename = 'Cholesterol.xyz'
# xyzFilename = 'Serotonin.xyz'
# xyzFilename = 'Decane_C10H22.xyz'
# xyzFilename = 'Icosane_C20H42.xyz'
# xyzFilename = 'Tetracontane_C40H82.xyz'
# xyzFilename = 'Pentacontane_C50H102.xyz'
# xyzFilename = 'Octacontane_C80H162.xyz'
# xyzFilename = 'Hectane_C100H202.xyz'
# xyzFilename = 'Icosahectane_C120H242.xyz'

# auxbasisName = 'sto-3g'
# auxbasisName = '6-31G'
auxbasisName = 'def2-universal-jfit'

#First of all we need a mol object with some geometry
mol = Mol(coordfile = xyzFilename)

# Next we need to specify some basis
# The basis set can then be used to calculate things like Overlap, KE, integrals/matrices.
basis = Basis(mol, {'all':Basis.load(mol=mol, basis_name=basis_set_name)})

# We also need an auxiliary basis for density fitting
auxbasis = Basis(mol, {'all':Basis.load(mol=mol, basis_name=auxbasisName)})

#Now we can calculate integrals using different algorithms

#Let's calculate the complete ERI array using the explicit conventional formula (Slow)
print('\n\n\n')
print('Integrals')
print('3c2e ERI array (Conventional Algorithm)\n')
print('NAO: ', basis.bfs_nao)
print('NAO (aux): ', auxbasis.bfs_nao)
start=timer()
#NOTE: The matrices are calculated in CAO basis and not the SAO basis
#You should refer to the example that shows the transformation between the two if you need matrices in SAO basis.
ERI_conv = Integrals.conv_3c2e_symm(basis, auxbasis)
print(ERI_conv[0:7,0:7,0]) 
duration = timer() - start
print('Duration for 3c2e ERI using PyFock Conventional algorithm: ',duration)


#Let's calculate the complete 3c2e ERI array using the Rys algorithm
print('\n\n\n')
print('Integrals')
print('3c2e ERI array (Rys)\n')
print('NAO: ', basis.bfs_nao)
print('NAO (aux): ', auxbasis.bfs_nao)
start=timer()
#NOTE: The matrices are calculated in CAO basis and not the SAO basis
#You should refer to the example that shows the transformation between the two if you need matrices in SAO basis.
ERI_rys = Integrals.rys_3c2e_symm(basis, auxbasis, schwarz=False)
print(ERI_rys[0:7,0:7,0]) 
duration = timer() - start
print(abs(ERI_conv - ERI_rys).max())
print('Duration for 3c2e ERI using PyFock Rys algorithm: ',duration)

if bench_GPU:
    print('\n\n\n')
    print('CrysX-PyFock (GPU)')
    print('NAO: ', basis.bfs_nao)
    print('NAO (aux): ', auxbasis.bfs_nao)
    #NOTE: The matrices are calculated in CAO basis and not the SAO basis
    #You should refer to the example that shows the transformation between the two if you need matrices in SAO basis.
    start=timer()
    ERI_rys_gpu = Integrals.rys_3c2e_symm_cupy(basis, auxbasis, schwarz=False)
    print(ERI_rys_gpu[0:7,0:7,0]) 
    duration = timer() - start
    print('Duration for 3c2e ERI using PyFock (GPU): ', duration)
    import cupy as cp
    print('Difference b/w CPU and GPU version: ', abs(ERI_rys - cp.asnumpy(ERI_rys_gpu)).max())

    print('\n\n\n')
    print('CrysX-PyFock (GPU 32-bit)')
    print('NAO: ', basis.bfs_nao)
    print('NAO (aux): ', auxbasis.bfs_nao)
    #NOTE: The matrices are calculated in CAO basis and not the SAO basis
    #You should refer to the example that shows the transformation between the two if you need matrices in SAO basis.
    start=timer()
    ERI_rys_gpu_fp32 = Integrals.rys_3c2e_symm_cupy_fp32(basis, auxbasis, schwarz=False)
    print(ERI_rys_gpu_fp32[0:7,0:7,0]) 
    duration = timer() - start
    print('Duration for 3c2e ERI using PyFock (GPU 32-bit): ', duration)
    import cupy as cp
    print('Difference b/w CPU and GPU (32 bit) version: ', abs(ERI_rys - cp.asnumpy(ERI_rys_gpu_fp32)).max())


#Comparison with PySCF
from pyscf import gto, dft, df
from timeit import default_timer as timer
molPySCF = gto.Mole()
molPySCF.atom = xyzFilename
molPySCF.basis = basis_set_name
molPySCF.cart = True
molPySCF.build()
#print(molPySCF.cart_labels())

auxmol = df.addons.make_auxmol(molPySCF, auxbasisName)

#ERI array/tensor
start=timer()
ERI_pyscf = df.incore.aux_e2(molPySCF, auxmol, intor='int3c2e')
duration = timer() - start
print('\n\nPySCF')
print(ERI_pyscf[0:7,0:7,0])
print('Array dimensions: ', ERI_pyscf.shape)
print('Duration for 3c2e ERI using PySCF: ',duration)
print(abs(ERI_pyscf - ERI_conv).max())  #There will sometimes be a difference b/w PySCF and CrysX values because PySCF doesn't normalize d,f,g orbitals.
# print(abs(ERI_pyscf - ERI_mmd).max())  #There will sometimes be a difference b/w PySCF and CrysX values because PySCF doesn't normalize d,f,g orbitals.
