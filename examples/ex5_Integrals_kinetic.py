from pyfock import Basis
from pyfock import Mol
from pyfock import Integrals

import numpy as np

# KINETIC POTENTIAL MATRIX

# xyzFilename = 'Benzene-Fulvene_Dimer.xyz'
# xyzFilename = 'h2o.xyz'
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

#First of all we need a mol object with some geometry
mol = Mol(coordfile = xyzFilename)

# Next we need to specify some basis
# The basis set can then be used to calculate things like Overlap, KE, integrals/matrices.
basis = Basis(mol, {'all':Basis.load(mol=mol, basis_name='sto-3g')})
#basis = Basis(mol, {'all':Basis.load(mol=mol, basis_name='def2-svp')})

#Now we can calculate integrals.
# This example shows how to calculate the kinetic matrix using the basis set object created.
# One can specify exactly which elements of the kinetic matrix they want to calculate.
# So, one can either calculate a single element or a continuous block of the matrix using the slice.

#Let's calculate the complete kinetic potential matrix
print('\n\n\n')
print('Integrals')
print('Kinetic energy matrix\n')
#NOTE: The matrices are calculated in CAO basis and not the SAO basis
#You should refer to the example that shows the transformation between the two if you need matrices in SAO basis.
Vkin = Integrals.kin_mat_symm(basis)
print(Vkin) 
print(Vkin.shape)

#Instead of calculating the complete KE matrix you can also just calculate a specific element or a continuous block of the kinetic matrix
#This is done by specifying an additional argument that contains the start and indices of the block of the kinetic matrix 
#that you want to calculate.
print('\n\n\n')
print('Integrals')
start_row = 1
end_row = 7
start_col = 1
end_col = 7
print('\nSubset of kinetic energy matrix Vkin[start_row:end_row, start_col:end_col]\n')
print([start_row, end_row, start_col, end_col])
Vkin_subset = Integrals.kin_mat_symm(basis, slice=[start_row, end_row, start_col, end_col])
print(Vkin_subset) 

# Compare with the slice of the original full matrix
print('\n\nCompare with the slice of the original full matrix')
print('abs(Vkin[start_row:end_row, start_col:end_col] - Vkin_subset).max()')
print(abs(Vkin_subset - Vkin[start_row:end_row, start_col:end_col]).max())  

#Let's try for a larger basis set like def2-tzvp
basisBig = Basis(mol, {'all':Basis.load(mol=mol, basis_name='def2-tzvp')})
print('\n\n\n')
print('Integrals')
print('\nKinetic energy matrix in def2-TZVP basis\n')
print(Integrals.kin_mat_symm(basisBig))
