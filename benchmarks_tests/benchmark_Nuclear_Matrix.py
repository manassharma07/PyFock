from pyfock import Basis
from pyfock import Mol
from pyfock import Integrals
from timeit import default_timer as timer
import numpy as np
import os
from pyscf import gto, dft
import numba 

ncores = 4
bench_GPU = False

numba.set_num_threads(ncores)
os.environ['OMP_NUM_THREADS'] = str(ncores)
os.environ["OPENBLAS_NUM_THREADS"] = str(ncores) # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = str(ncores) # export MKL_NUM_THREADS=4
os.environ["VECLIB_MAXIMUM_THREADS"] = str(ncores) # export VECLIB_MAXIMUM_THREADS=4
# os.environ["NUMEXPR_NUM_THREADS"] = str(ncores) # export NUMEXPR_NUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = str(ncores) # export NUMEXPR_NUM_THREADS=1


#IMPORTANT
#Since, it seems that in order for the Numba implementation to work efficiently we must remove the compile time from the calculation.
#So, a very simple calculation on a very small system must be run before, to have the numba functions compiled.
mol_temp = Mol(coordfile='H2O.xyz')
basis_temp = Basis(mol_temp, {'all':Basis.load(mol=mol_temp, basis_name='sto-2g')})
Vmat_temp = Integrals.nuc_mat_symm(basis_temp, mol_temp)
Vmat_temp = Integrals.rys_nuc_mat_symm(basis_temp, mol_temp)
Vmat_temp = Integrals.mmd_nuc_mat_symm(basis_temp, mol_temp)

#NUCLEAR POTENTIAL MATRIX BENCHMARK and comparison with PySCF
#Benchmarking and performance assessment and comparison using various techniques and different softwares

# basis_set_name = 'sto-2g'
# basis_set_name = 'sto-3g'
# basis_set_name = 'sto-6g'
# basis_set_name = 'def2-SVP'
# basis_set_name = 'def2-DZVP'
basis_set_name = 'def2-TZVP'
# basis_set_name = 'def2-TZVPPD'
# basis_set_name = 'def2-QZVP'
# basis_set_name = 'def2-QZVPPD'
# basis_set_name = 'ano-rcc'

# xyzFilename = 'Benzene-Fulvene_Dimer.xyz'
# xyzFilename = 'H2O.xyz'
# xyzFilename = 'Ethane.xyz'
# xyzFilename = 'Zn.xyz'
# xyzFilename = 'Zn_dimer.xyz'
# xyzFilename = 'TPP.xyz'
# xyzFilename = 'Zn_TPP.xyz'
# xyzFilename = 'Cholesterol.xyz'
# xyzFilename = 'Serotonin.xyz'
# xyzFilename = 'Decane_C10H22.xyz'
# xyzFilename = 'Icosane_C20H42.xyz'
# xyzFilename = 'Tetracontane_C40H82.xyz'
# xyzFilename = 'Pentacontane_C50H102.xyz'
# xyzFilename = 'Octacontane_C80H162.xyz'
# xyzFilename = 'Hectane_C100H202.xyz'
xyzFilename = 'Icosahectane_C120H242.xyz'

#First of all we need a mol object with some geometry
mol = Mol(coordfile = xyzFilename)

# Next we need to specify some basis
# The basis set can then be used to calculate things like Overlap, KE, integrals/matrices.
basis = Basis(mol, {'all':Basis.load(mol=mol, basis_name=basis_set_name)})
#basis = Basis(mol, {'all':Basis.load(mol=mol, basis_name='def2-svp')})


print('\n\n\n')
print('CrysX-PyFock')
print('NAO: ', basis.bfs_nao)
#NOTE: The matrices are calculated in CAO basis and not the SAO basis
#You should refer to the example that shows the transformation between the two if you need matrices in SAO basis.
start=timer()
Vnuc_conv = Integrals.nuc_mat_symm(basis, mol)
print(Vnuc_conv) 
duration = timer() - start
print('Matrix dimensions: ', Vnuc_conv.shape)
print('Duration for Vnuc using PyFock: ',duration)

# print('\n\n\n')
# print('CrysX-PyFock Rys (3c2e Integral scheme with sharp gaussians at nuclear centers)')
# print('NAO: ', basis.bfs_nao)
# #NOTE: The matrices are calculated in CAO basis and not the SAO basis
# #You should refer to the example that shows the transformation between the two if you need matrices in SAO basis.
# start=timer()
# Vnuc_rys = Integrals.rys_nuc_mat_symm(basis, mol)
# print(Vnuc_rys) 
# duration = timer() - start
# print('Matrix dimensions: ', Vnuc_rys.shape)
# print(abs(Vnuc_rys-Vnuc_conv).max())
# print('Duration for Vnuc using PyFock (Rys 3c2e): ',duration)

# print('\n\n\n')
# print('CrysX-PyFock MMD (Unstable as gives segmentation fault sometimes)')
# print('NAO: ', basis.bfs_nao)
# #NOTE: The matrices are calculated in CAO basis and not the SAO basis
# #You should refer to the example that shows the transformation between the two if you need matrices in SAO basis.
# start=timer()
# Vnuc_mmd = Integrals.mmd_nuc_mat_symm(basis, mol)
# print(Vnuc_mmd) 
# duration = timer() - start
# print('Matrix dimensions: ', Vnuc_mmd.shape)
# print(abs(Vnuc_mmd-Vnuc_conv).max())
# print('Duration for Vnuc using PyFock (MMD): ',duration)

if bench_GPU:
    print('\n\n\n')
    print('CrysX-PyFock (GPU)')
    print('NAO: ', basis.bfs_nao)
    #NOTE: The matrices are calculated in CAO basis and not the SAO basis
    #You should refer to the example that shows the transformation between the two if you need matrices in SAO basis.
    start=timer()
    Vnuc_gpu = Integrals.nuc_mat_symm_cupy(basis, mol)
    print(Vnuc_gpu) 
    duration = timer() - start
    print('Matrix dimensions: ', Vnuc_gpu.shape)
    print('Duration for Vnuc using PyFock (GPU): ',duration)
    import cupy as cp
    print('Difference b/w CPU and GPU version: ', abs(Vnuc_conv - cp.asnumpy(Vnuc_gpu)).max())


#Comparison with PySCF
molPySCF = gto.Mole()
molPySCF.atom = xyzFilename
molPySCF.basis = basis_set_name
molPySCF.cart = True
molPySCF.build()
#print(molPySCF.cart_labels())


#Nuclear mat
start=timer()
# V = molPySCF.intor('int1e_nuc')
V = molPySCF.intor_symmetric('int1e_nuc')
duration = timer() - start
print('\n\nPySCF')
print(V)
print('Matrix dimensions: ', V.shape)
print('Duration for Vnuc using PySCF: ',duration)
print('Difference b/w PyFock (default) and PySCF: ',abs(V - Vnuc_conv).max())  #There will sometimes be a difference b/w PySCF and CrysX values because PySCF doesn't normalize d,f,g orbitals.
# print('Difference b/w PyFock (Rys) and PySCF: ',abs(V - Vnuc_rys).max())  #There will sometimes be a difference b/w PySCF and CrysX values because PySCF doesn't normalize d,f,g orbitals.
# print('Difference b/w PyFock (MMD) and PySCF: ',abs(V - Vnuc_mmd).max())  #There will sometimes be a difference b/w PySCF and CrysX values because PySCF doesn't normalize d,f,g orbitals.
if bench_GPU:
    print('Difference b/w PyFock (GPU) and PySCF: ',abs(V - cp.asnumpy(Vnuc_gpu)).max())  #There will sometimes be a difference b/w PySCF and CrysX values because PySCF doesn't normalize d,f,g orbitals.
    