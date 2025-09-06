from pyfock import Basis
from pyfock import Mol
#Example on how to specify basis


#First of all we need a mol object with some geometry
mol = Mol(coordfile='h2o.xyz')

# Define some basis as strings in a variable
# This can later be used in Basis.load() function
sto3g = '''
*
*
o STO-3G
*
3   s
0.1307093214E+03       0.1543289673E+00
0.2380886605E+02       0.5353281423E+00
0.6443608313E+01       0.4446345422E+00
3   s
0.5033151319E+01      -0.9996722919E-01
0.1169596125E+01       0.3995128261E+00
0.3803889600E+00       0.7001154689E+00
3   p
0.5033151319E+01       0.1559162750E+00
0.1169596125E+01       0.6076837186E+00
0.3803889600E+00       0.3919573931E+00
*
*
h STO-3G
*
3   s
0.3425250914E+01       0.1543289673E+00
0.6239137298E+00       0.5353281423E+00
0.1688554040E+00       0.4446345422E+00
*
'''
# To create a basis set object we need an already instantiated mol object.
# The second argument is supposed to be a dictionary that specifies the basis set to be used for 
# a given atomic species.
# The keyname= 'all' means that all the atoms would be specified the same basis set.
# after the keyname one should provide the basis set to be used (The complete basis set, not just the name) 
basis = Basis(mol, {'all':sto3g}) # Load from string
print(basis.nprims)
print(basis.prim_coeffs)
print('\nNormalization factor of the contraction of primitives that make up a basis function')
print(basis.bfs_contr_prim_norms)


#Example 2
# Alternatively one can load the basis set from a file in the same working directory
# NOTE: The basis set should be specifiec in the Turbomole format
# Basis.loadfromfile is a function that would read the basis sets corresponding to the atoms in the given mol object
# from a file(should be in Turbomole format).
basis = Basis(mol, {'all':Basis.loadfromfile(mol=mol, basis_name='sto-3g.dat')}) # Load from file
print(basis.nprims)
print(basis.prim_coeffs)
print('\nNormalization factor of the contraction of primitives that make up a basis function')
print(basis.bfs_contr_prim_norms)

#Example 3
# Alternatively one can load the basis set from crysxdft library by specifying the name as shown below
# using the Basis.load() function. The function takes the mol and basis set name as arguments.
basis = Basis(mol, {'all':Basis.load(mol=mol, basis_name='sto-3g')}) # Load from crysxdft library
print(basis.nprims)
print(basis.prim_coeffs)
print('\nNormalization factor of the contraction of primitives that make up a basis function')
print(basis.bfs_contr_prim_norms)
#print(basis.basisSet)