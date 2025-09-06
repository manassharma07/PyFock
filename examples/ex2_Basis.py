from pyfock import Basis
from pyfock import Mol

#Example on how to access various properties of Basis object such info on primitives,
#shells and basis functions

# xyzFilename = 'Benzene-Fulvene_Dimer.xyz'
xyzFilename = 'H2O.xyz'
# xyzFilename = 'H2.xyz'
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
# xyzFilename = 'Graphene_C76.xyz'

#First of all we need a mol object with some geometry
mol = Mol(coordfile = xyzFilename)

# This example shows how to load the basis set from the CrysX library,
# and the properties of the basis set object.
# The basis set can then be used to calculate things like Overlap, KE, integrals/matrices.
# basis = Basis(mol, {'all':Basis.load(mol=mol, basis_name='def2-tzvp')})
basis = Basis(mol, {'all':Basis.load(mol=mol, basis_name='def2-SVP')})
# basis = Basis(mol, {'all':Basis.load(mol=mol, basis_name='def2-svp')})
#print(basis.basisSet)
print('\n\n\nTotal number of basis functions\n')
print(basis.bfs_nao)
print('\n\n\nTotal number of primitives\n')
print(basis.totalnprims)
print('\nNumber of primitives in each shell\n')
print(basis.nprims, len(basis.nprims))
print('\nIndices of the atoms to which each primitive function corresponds/belongs to\n')
print(basis.prim_atoms)
print('\nNo. of primitives corresponding to each atomic index\n')
print(basis.nprims_atoms)
print('\nIndices of the shells to which each primitive function corresponds/belongs to\n')
print(basis.prim_shells)
print('\nNo. of primitives corresponding to each shell index\n')
print(basis.nprims_shells)
print('\nA list of tuples that shows the angular momentum of shell in the first index and the no. of corresponding primitives in the second index\n')
print(basis.nprims_shell_l_list)
print('\nLargest exponent values corresponding to each atom\n')
print(basis.alpha_max)
print('\nSmallest exponent values corresponding to each shell of each atom\n')
print(basis.alpha_min)
print('\nCoefficients and exponents of primitives\n')
print(basis.prim_coeffs)
print(basis.prim_expnts)
print('\nConfirming if the number of primitives and the number of exponents/coefficients read is the same\n')
print(len(basis.prim_coeffs))
print('\nShell labels\n')
print(basis.shellsLabel)
print('\nShells\n')
print(basis.shells)
print('\nBasis function information\n')
print('\nBF labels\n')
print(basis.bfs_label)
print(len(basis.bfs_label))
print('\nBF lmn triplets\n')
print(basis.bfs_lmn)
print(len(basis.bfs_label))
print('\nBF angular momentum\n')
print(basis.bfs_lm)
print(len(basis.bfs_lm))
print('\nNumber of basis functions in each shell\n')
print(basis.bfs_nbfshell)
print('\nOffset index of the basis function corresponding to a given shell index')
print(basis.shell_bfs_offset)
print('\nShell indices of basis functions\n')
print(basis.bfs_shell_index)
print('\nNumber of primitives for a basis function\n')
print(basis.bfs_nprim)
print(len(basis.bfs_label))
print('\nExponents of primitives for each basis functions\n')
print(basis.bfs_expnts)
print(len(basis.bfs_expnts))
print('\nContraction coefficients of primitives for each basis functions\n')
print(basis.bfs_coeffs)
print(len(basis.bfs_coeffs))
print('\nNormalization factors of primitives for each basis functions\n')
print(basis.bfs_prim_norms)
print(len(basis.bfs_prim_norms))
print('\nNormalization factor of the contraction of primitives that make up a basis function\n')
print(basis.bfs_contr_prim_norms)
print(len(basis.bfs_contr_prim_norms))
print('\nCoordinates of basis functions\n')
print(basis.bfs_coords)
print(len(basis.bfs_coords))
print('Indices of atoms to which basis functions correspond to\n')
print(basis.bfs_atoms)
print(len(basis.bfs_atoms))
print('\n\n\n')
