from pyfock import Basis
from pyfock import Mol
from pyfock import Integrals

import numpy as np

# 2c2e Integral via Rys Quadrature (Faster than the conventional implementation)
# This is basically the Coulomb metric used for density fitting. 

# xyzFilename = 'Benzene-Fulvene_Dimer.xyz'
xyzFilename = 'h2o.xyz'
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

# basisName = 'sto-3g'
# basisName = 'sto-6g'
basisName = '6-31G'
# basisName = 'def2-SVP'

auxbasisName = '6-31G'
# auxbasisName = 'def2-universal-jfit'

#First of all we need a mol object with some geometry
mol = Mol(coordfile = xyzFilename)

# Next we need to specify some basis
# The basis set can then be used to calculate things like Overlap, KE, integrals/matrices.
basis = Basis(mol, {'all':Basis.load(mol=mol, basis_name=basisName)})
#basis = Basis(mol, {'all':Basis.load(mol=mol, basis_name='def2-svp')})

#Now we can calculate integrals.
# This example shows how to calculate the 4c2e ERI array using the basis set object created.
# One can specify exactly which elements of the ERI array they want to calculate.
# So, one can either calculate a single element or a continuous block of the matrix using the slice.

#Let's calculate the complete ERI array
print('\n\n\n')
print('Integrals')
print('2c2e array\n')
print('NAO: ', basis.bfs_nao)
#NOTE: The matrices are calculated in CAO basis and not the SAO basis
#You should refer to the example that shows the transformation between the two if you need matrices in SAO basis.
ERI = Integrals.conv_2c2e_symm(basis)
print(ERI) 

#Instead of calculating the complete ERI array you can also just calculate a specific element or a continuous block of the ERI array
#This is done by specifying an additional argument that contains the start and indices of the block of the ERI array
#that you want to calculate.
print('\n\n\n')
print('Integrals')
indx_startA = 1
indx_endA = 4
indx_startB = 1
indx_endB = 4
print('\nSubset of 2c2e array 2c2e[indx_startA:indx_endA, indx_startB:indx_endB]\n')
print('Slice',[indx_startA, indx_endA, indx_startB, indx_endB])
ERI_subset = Integrals.conv_2c2e_symm(basis, slice=[indx_startA, indx_endA, indx_startB, indx_endB])
print(ERI_subset[:, :]) 
print('\nSlice from the original array\n')
# print(ERI[indx_startA:indx_endA, indx_startB:indx_endB, 0, 0]) 
print(ERI[indx_startA:indx_endA, indx_startB:indx_endB]) 

# Compare with the slice of the original full matrix
print('\n\nCompare with the slice of the original full matrix')
print('abs(2c2e[indx_startA:indx_endA, indx_startB:indx_endB] - 2c2e_subset).max()')
print(abs(ERI_subset - ERI[indx_startA:indx_endA, indx_startB:indx_endB]).max())  

#Let's try for a larger basis set like def2-svp
basisBig = Basis(mol, {'all':Basis.load(mol=mol, basis_name='def2-svp')})
print('\n\n\n')
print('Integrals')
print('\n2c2e ERI array in def2-SVP basis\n')
print('NAO: ', basisBig.bfs_nao)
print(Integrals.conv_2c2e_symm(basisBig))


#Comparison with PySCF
from pyscf import gto, dft, df
from timeit import default_timer as timer
molPySCF = gto.Mole()
molPySCF.atom = 'h2o.xyz'
molPySCF.basis = basisName
molPySCF.cart = True
molPySCF.build()
#print(molPySCF.cart_labels())

# auxmol = df.addons.make_auxmol(molPySCF, auxbasis='weigend')
auxmol = df.addons.make_auxmol(molPySCF, auxbasis='6-31G')


#Nuclear mat
start=timer()
# V = molPySCF.intor('int1e_nuc')
ERI_pyscf = df.incore.aux_e2(molPySCF, auxmol, intor='int2c2e')
duration = timer() - start
print('\n\nPySCF')
print(ERI_pyscf)
print('Array dimensions: ', ERI_pyscf.shape)
print(abs(ERI_pyscf - ERI).max())  #There will sometimes be a difference b/w PySCF and CrysX values because PySCF doesn't normalize d,f,g orbitals.
print('Duration for ERI using PySCF: ',duration)