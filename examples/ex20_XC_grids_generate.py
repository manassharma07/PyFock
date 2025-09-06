from pyfock import Basis
from pyfock import Mol
from pyfock import Grids
import numpy as np

ncores = 4

# Define the molecule using an XYZ file
mol = Mol(coordfile = 'h2o.xyz')

# Generate grids for XC term
gridsLevel = 3
basis = Basis(mol, {'all':Basis.load(mol=mol, basis_name='def2-QZVP')})
grids = Grids(mol, basis=basis, level = gridsLevel, ncores=ncores)

print('\n\nWeights:')
print(grids.weights)
print('\n\nCoords:')
print(grids.coords)
print('\nNo. of generated grid points: ', grids.coords.shape[0])