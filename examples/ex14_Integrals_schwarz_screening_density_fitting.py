from pyfock import Basis
from pyfock import Mol
from pyfock import Integrals

from timeit import default_timer as timer

import numpy as np

import warnings
warnings.filterwarnings('ignore')

import numba
numba.set_num_threads(4)

# Get rid of compilation time for getting accurate timings
print('Compiling...')
mol = Mol(coordfile = 'h2o.xyz')
basis_temp = Basis(mol, {'all':Basis.load(mol=mol, basis_name='sto-2g')})
auxbasis_temp = Basis(mol, {'all':Basis.load(mol=mol, basis_name='sto-2g')})
ints2c2e_temp = Integrals.rys_2c2e_symm(auxbasis_temp)
ints3c2e_temp = Integrals.rys_3c2e_symm(basis_temp, auxbasis_temp)
ints3c2e_temp = Integrals.rys_3c2e_tri(basis_temp, auxbasis_temp)
ints3c2e_temp = Integrals.rys_3c2e_tri_schwarz(basis_temp, auxbasis_temp, np.array([0,1,2,3], dtype=np.uint16), np.array([0,1,2,3], dtype=np.uint16), np.array([0,1,2,3], dtype=np.uint16))
ints4c2e_diag_temp = Integrals.schwarz_helpers.eri_4c2e_diag(basis_temp)
print('Compilation done!\n\n')

# Example on how to perform Schwarz screening for density fitting calculations

# xyzFilename = 'Benzene-Fulvene_Dimer.xyz'
# xyzFilename = 'h2o.xyz'
# xyzFilename = 'Ethane.xyz'
# xyzFilename = 'Cholesterol.xyz'
# xyzFilename = 'Serotonin.xyz'
xyzFilename = 'Decane_C10H22.xyz'
# xyzFilename = 'Icosane_C20H42.xyz'
# xyzFilename = 'Tetracontane_C40H82.xyz'
# xyzFilename = 'Pentacontane_C50H102.xyz'
# xyzFilename = 'Octacontane_C80H162.xyz'
# xyzFilename = 'Hectane_C100H202.xyz'
# xyzFilename = 'Icosahectane_C120H242.xyz'

# basisName = 'sto-3g'
# basisName = 'sto-6g'
# basisName = '6-31G'
basisName = 'def2-SVP'

# auxbasisName = '6-31G'
auxbasisName = 'def2-universal-jfit'

#First of all we need a mol object with some geometry
mol = Mol(coordfile = xyzFilename)

# Next we need to specify some basis
# The basis set can then be used to calculate things like Overlap, KE, integrals/matrices.
basis = Basis(mol, {'all':Basis.load(mol=mol, basis_name=basisName)})
#basis = Basis(mol, {'all':Basis.load(mol=mol, basis_name='def2-svp')})

# We also need an auxiliary basis for density fitting
auxbasis = Basis(mol, {'all':Basis.load(mol=mol, basis_name=auxbasisName)})

# Calculate two-centered two electron integrals (Coulomb metric) for the density fitting procedure
ints2c2e = Integrals.rys_2c2e_symm(auxbasis)

start = timer()
print('\n\nPerforming Schwarz screening...')
threshold_schwarz = 1e-11
print('Threshold ', threshold_schwarz)

# Size of the 3c2e array naively
nints3c2e = basis.bfs_nao*basis.bfs_nao*auxbasis.bfs_nao
# Size of the 3c2e array considering the ij symmetry of the first two indices
nints3c2e_tri = int(basis.bfs_nao*(basis.bfs_nao+1)/2.0)*auxbasis.bfs_nao

# This is based on Schwarz inequality screening
# "Diagonal" elements of ERI 4c2e array (uv|uv)
ints4c2e_diag = Integrals.schwarz_helpers.eri_4c2e_diag(basis)

# Calculate the indices of the ints3c2e array with significant contributions based on Schwarz inequality
# The following returns an array with 0s and 1s. 1=significant; 0=non-significant
indices_temp = Integrals.schwarz_helpers.calc_indices_3c2e_schwarz(ints4c2e_diag, ints2c2e, basis.bfs_nao, auxbasis.bfs_nao, threshold_schwarz)
# Find the corresponding indices
if basis.bfs_nao<65500 and auxbasis.bfs_nao<65500:
    indices = [a.astype(np.uint16) for a in indices_temp.nonzero()] # Will work as long as the no. of max(Bfs,auxbfs) is less than 65535
else:
    indices = [a.astype(np.uint32) for a in indices_temp.nonzero()] # Will work as long as the no. of Bfs/auxbfs is less than 4294967295

# Get rid of temp variables
indices_temp=0

print('Size of array storing the significant indices of 3c2e ERI in GB ', indices[0].nbytes/1e9, flush=True)

nsignificant = len(indices[0])
print('No. of elements in the standard three-centered two electron ERI tensor: ', nints3c2e, flush=True)
print('No. of elements in the triangular three-centered two electron ERI tensor: ', nints3c2e_tri, flush=True)
print('No. of significant triplets based on Schwarz inequality and triangularity: ' + str(nsignificant) + ' or '+str(np.round(nsignificant/nints3c2e*100,1)) + '% of original', flush=True)
print('Schwarz screening done!')
duration = timer() - start
print('Time taken ', duration)

# Calculate the 3c2e integrals for only the significant pairs
start = timer()
ints3c2e = Integrals.rys_3c2e_tri_schwarz(basis, auxbasis, indices[0], indices[1], indices[2])
duration = timer() - start
print('\n3c2e integral with Schwarz screening')
print(ints3c2e)
print('Time taken ', duration)
size = ints3c2e.nbytes/1e9
print('Size (GB) ', size)

# Calculate the unique (triangular) 3c2e integrals without Schwarz screening
start = timer()
ints3c2e = Integrals.rys_3c2e_tri(basis, auxbasis)
duration = timer() - start
print('\nUnique (triangular) 3c2e integral without Schwarz screening')
print(ints3c2e)
print('Time taken ', duration)
size = ints3c2e.nbytes/1e9
print('Size (GB) ', size)

# Calculate the full 3c2e integrals without Schwarz screening
start = timer()
ints3c2e = Integrals.rys_3c2e_symm(basis, auxbasis)
duration = timer() - start
print('\nFull 3c2e integral without Schwarz screening')
print(ints3c2e)
print('Time taken ', duration)
size = ints3c2e.nbytes/1e9
print('Size (GB) ', size)