from pyfock import Basis
from pyfock import Mol
from pyfock import Grids
from pyfock import Integrals
from pyfock import DFT
import numpy as np
from opt_einsum import contract

ncores = 2
import numba
numba.set_num_threads(ncores)


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

ao_values, ao_grad_values = Integrals.bf_val_helpers.eval_bfs_and_grad(basis, grids.coords, parallel=True)

print('\n\nAO Values:')
print(ao_values)
print('\n\nAO gradient Values:')
print(ao_grad_values)