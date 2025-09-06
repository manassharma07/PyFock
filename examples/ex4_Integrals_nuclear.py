from pyfock import Basis
from pyfock import Mol
from pyfock import Integrals
# from timeit import default_timer as timer
import numpy as np

#NUCLEAR POTENTIAL MATRIX BENCHMARK and comparison with PySCF
#Benchmarking and performance assessment and comparison using various techniques and different softwares

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

#First of all we need a mol object with some geometry
mol = Mol(coordfile = xyzFilename)

# Next we need to specify some basis
# The basis set can then be used to calculate things like Overlap, KE, integrals/matrices.
basis = Basis(mol, {'all':Basis.load(mol=mol, basis_name='sto-3g')})
#basis = Basis(mol, {'all':Basis.load(mol=mol, basis_name='def2-svp')})

#Now we can calculate integrals.
# This example shows how to calculate the Nuclear matrix using the basis set object created.
# Notice, how we don't need a mol object for things like overlap/KE integrals but we do need one for Nuclear matrix.
# However, the mol object and basis object do not have to be of the same molecule.
# This is because the basis functions in basis set object contain their own coordinates and don't rely on mol object.
# Therefore, you can calculate the nuclear matrix in the basis of one molecule due to the nuclei of another molecule.
# I feel this can be a useful feature and offers reasonable flexibility and modularity.
# Furthermore, one can specify exactly which elements of the nuclear matrix they want to calculate.
# So, one can either calculate a single element or a continuous block of the matrix using the slice.

#Let's calculate the complete nuclear potential matrix
print('\n\n\n')
print('Integrals')
print('Nuclear matrix\n')
#NOTE: The matrices are calculated in CAO basis and not the SAO basis
#You should refer to the example that shows the transformation between the two if you need matrices in SAO basis.
Vnuc = Integrals.nuc_mat_symm(basis, mol)
print(Vnuc) 

#Instead of calculating the complete KE matrix you can also just calculate a specific element or a continuous block of the kinetic matrix
#This is done by specifying an additional argument that contains the start and indices of the block of the kinetic matrix 
#that you want to calculate.
print('\n\n\n')
print('Integrals')
start_row = 5
end_row = 7
start_col = 5
end_col = 7
print('\nSubset of Nuclear matrix Vnuc[start_row:end_row, start_col:end_col]\n')
print([start_row, end_row, start_col, end_col])
Vnuc_subset = Integrals.nuc_mat_symm(basis, mol, slice=[start_row, end_row, start_col, end_col])
print(Vnuc_subset) 

# Compare with the slice of the original full matrix
print('\n\nCompare with the slice of the original full matrix')
print('abs(Vnuc[start_row:end_row, start_col:end_col] - Vnuc_subset).max()')
print(abs(Vnuc_subset - Vnuc[start_row:end_row, start_col:end_col]).max())  

#Let's try for a larger basis set like def2-tzvp
basisBig = Basis(mol, {'all':Basis.load(mol=mol, basis_name='def2-tzvp')})
print('\n\n\n')
print('Integrals')
print('\nNuclear matrix in def2-TZVP basis\n')
print(Integrals.nuc_mat_symm(basisBig, mol))