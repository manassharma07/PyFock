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

ao_values = Integrals.bf_val_helpers.eval_bfs(basis, grids.coords, parallel=True)

print('\n\nAO Values:')
print(ao_values)

# basis = Basis(mol, {'all':Basis.load(mol=mol, basis_name='def2-SVP')})
dft = DFT(mol, basis)
dmat = dft.guessCoreH()

rho = contract('ij,mi,mj->m', dmat, ao_values, ao_values)
print('\n\nRho Values:')
print(rho)


rho_alternative = Integrals.bf_val_helpers.eval_rho(ao_values, dmat) # This is by-far the fastest now (when not using non_zero_indices) <-----
print('\n\nRho Values calculated in another way:')
print(rho_alternative)