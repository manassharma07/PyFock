from pyfock import Basis
from pyfock import Mol
from pyfock import Integrals

import numpy as np

# 3c2e Integral via Rys Quadrature (Faster than the conventional implementation)
# Here we use a memory efficient version of rys_3c2e_symm 
# that returns a 2D array instead of a 3D array.
# Only the symmetrically unique values are returned. 

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


# We also need an auxiliary basis for density fitting
auxbasis = Basis(mol, {'all':Basis.load(mol=mol, basis_name=auxbasisName)})

#Now we can calculate integrals.
# This example shows how to calculate the 4c2e ERI array using the basis set object created.
# One can specify exactly which elements of the ERI array they want to calculate.
# So, one can either calculate a single element or a continuous block of the matrix using the slice.

#Let's calculate the complete ERI array
print('\n\n\n')
print('Integrals')
print('3c2e array (triangular)\n')
print('NAO: ', basis.bfs_nao)
#NOTE: The matrices are calculated in CAO basis and not the SAO basis
#You should refer to the example that shows the transformation between the two if you need matrices in SAO basis.
ERI = Integrals.rys_3c2e_tri(basis, auxbasis)
print(ERI) 

#Let's try for a larger basis set like def2-svp
basisBig = Basis(mol, {'all':Basis.load(mol=mol, basis_name='def2-svp')})
print('\n\n\n')
print('Integrals')
print('\n2c2e ERI array in def2-SVP basis\n')
print('NAO: ', basisBig.bfs_nao)
print(Integrals.rys_3c2e_tri(basisBig, auxbasis))


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
ERI_pyscf = df.incore.aux_e2(molPySCF, auxmol, intor='int3c2e', aosym='s2ij')
duration = timer() - start
print('\n\nPySCF')
print(ERI_pyscf)
print('Array dimensions: ', ERI_pyscf.shape)
print(abs(ERI_pyscf - ERI).max())  #There will sometimes be a difference b/w PySCF and CrysX values because PySCF doesn't normalize d,f,g orbitals.
print('Duration for ERI using PySCF: ',duration)