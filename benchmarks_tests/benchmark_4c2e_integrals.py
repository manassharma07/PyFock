from pyfock import Basis
from pyfock import Mol
from pyfock import Integrals
from timeit import default_timer as timer

import numpy as np
import numba 

ncores = 4
numba.set_num_threads(ncores)


#IMPORTANT
#Since, it seems that in order for the Numba implementation to work efficiently we must remove the compile time from the calculation.
#So, a very simple calculation on a very small system must be run before, to have the numba functions compiled.
mol_temp = Mol(coordfile='H2O.xyz')
basis_temp = Basis(mol_temp, {'all':Basis.load(mol=mol_temp, basis_name='sto-2g')})
# Vmat_temp = Integrals.mmd_4c2e_symm(basis_temp)
Vmat_temp = Integrals.rys_4c2e_symm(basis_temp)
Vmat_temp = Integrals.rys_4c2e_symm_old(basis_temp)
Vmat_temp = Integrals.conv_4c2e_symm(basis_temp)


# 4c2e ERI via explicit conventional integrals (quite slow)

# basis_set_name = '6-31G'
# basis_set_name = 'sto-2g'
# basis_set_name = 'sto-3g'
# basis_set_name = 'sto-6g'
# basis_set_name = 'def2-SVP'
# basis_set_name = 'def2-DZVP'
# basis_set_name = 'def2-TZVP'
basis_set_name = 'def2-QZVP'
# basis_set_name = 'def2-TZVPPD'
# basis_set_name = 'def2-QZVPPD'
# basis_set_name = 'ano-rcc'
# basis_set_name = 'cc-pVDZ'

# xyzFilename = 'Benzene-Fulvene_Dimer.xyz'
xyzFilename = 'H2O.xyz'
# xyzFilename = 'Zn.xyz'
# xyzFilename = 'Zn_dimer.xyz'
# xyzFilename = 'Ethane.xyz'
# xyzFilename = 'Cholesterol.xyz'
# xyzFilename = 'Serotonin.xyz'
# xyzFilename = 'Decane_C10H22.xyz'
# xyzFilename = 'Icosane_C20H42.xyz'
# xyzFilename = 'Tetracontane_C40H82.xyz'
# xyzFilename = 'Pentacontane_C50H102.xyz'
# xyzFilename = 'Octacontane_C80H162.xyz'
# xyzFilename = 'Hectane_C100H202.xyz'
# xyzFilename = 'Icosahectane_C120H242.xyz'

#First of all we need a mol object with some geometry
mol = Mol(coordfile = xyzFilename)

# Next we need to specify some basis
# The basis set can then be used to calculate things like Overlap, KE, integrals/matrices.
basis = Basis(mol, {'all':Basis.load(mol=mol, basis_name=basis_set_name)})

#Now we can calculate integrals using different algorithms

#Let's calculate the complete ERI array using the explicit conventional formula (Slow)
print('\n\n\n')
print('Integrals')
print('4c2e ERI array (Conventional Algorithm)\n')
print('NAO: ', basis.bfs_nao)
start=timer()
#NOTE: The matrices are calculated in CAO basis and not the SAO basis
#You should refer to the example that shows the transformation between the two if you need matrices in SAO basis.
ERI_conv = Integrals.conv_4c2e_symm(basis)
print(ERI_conv[0:7,0:7,0,0]) 
duration = timer() - start
print('Duration for ERI using PyFock Conventional algorithm: ',duration)


#Let's calculate the complete ERI array using the MMD algorithm
# print('\n\n\n')
# print('Integrals')
# print('4c2e ERI array (MMD) (Unstable as gives segmentation fault on the second run)\n')
# print('NAO: ', basis.bfs_nao)
# start=timer()
# #NOTE: The matrices are calculated in CAO basis and not the SAO basis
# #You should refer to the example that shows the transformation between the two if you need matrices in SAO basis.
# ERI_mmd = Integrals.mmd_4c2e_symm(basis)
# print(ERI_mmd[0:7,0:7,0,0]) 
# duration = timer() - start
# print(abs(ERI_conv - ERI_mmd).max())
# print('Duration for ERI using PyFock MMD algorithm: ',duration)

#Let's calculate the complete ERI array using the Rys algorithm
print('\n\n\n')
print('Integrals')
print('4c2e ERI array (Rys Old)\n')
print('NAO: ', basis.bfs_nao)
start=timer()
#NOTE: The matrices are calculated in CAO basis and not the SAO basis
#You should refer to the example that shows the transformation between the two if you need matrices in SAO basis.
ERI_rys = Integrals.rys_4c2e_symm_old(basis)
print(ERI_rys[0:7,0:7,0,0]) 
duration = timer() - start
print(abs(ERI_conv - ERI_rys).max())
print('Duration for ERI using PyFock Rys algorithm: ',duration)

#Let's calculate the complete ERI array using the Rys algorithm
print('\n\n\n')
print('Integrals')
print('4c2e ERI array (Rys New)\n')
print('NAO: ', basis.bfs_nao)
start=timer()
#NOTE: The matrices are calculated in CAO basis and not the SAO basis
#You should refer to the example that shows the transformation between the two if you need matrices in SAO basis.
ERI_rys = Integrals.rys_4c2e_symm(basis)
print(ERI_rys[0:7,0:7,0,0]) 
duration = timer() - start
print(abs(ERI_conv - ERI_rys).max())
print('Duration for ERI using PyFock Rys algorithm: ',duration)


#Comparison with PySCF
from pyscf import gto, dft
from timeit import default_timer as timer
molPySCF = gto.Mole()
molPySCF.atom = xyzFilename
molPySCF.basis = basis_set_name
molPySCF.cart = True
molPySCF.build()
#print(molPySCF.cart_labels())


#ERI array/tensor
start=timer()
# ERI_pyscf = molPySCF.intor('int2e',aosym='s8')
ERI_pyscf = molPySCF.intor('int2e')
duration = timer() - start
print('\n\nPySCF')
# print(ERI_pyscf[0:7,0:7,0,0])
print('Array dimensions: ', ERI_pyscf.shape)
print('Duration for ERI using PySCF: ',duration)
print(abs(ERI_pyscf - ERI_conv).max())  #There will sometimes be a difference b/w PySCF and CrysX values because PySCF doesn't normalize d,f,g orbitals.
# print(abs(ERI_pyscf - ERI_mmd).max())  #There will sometimes be a difference b/w PySCF and CrysX values because PySCF doesn't normalize d,f,g orbitals.
