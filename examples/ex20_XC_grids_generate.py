# XC integration grids in PyFock: schemes, levels and options
#
# The exchange-correlation energy is integrated numerically on atom-centered grids.
# PyFock offers two grid schemes:
#
#   'treutler' (default): Treutler-Ahlrichs radial grids + Lebedev angular grids,
#                         angular pruning by radial regions, Becke partitioning
#   'numgrid'           : grids from the numgrid library (LMG radial grids)
#
# The grid level goes from 0 (coarsest) to 9 (finest). Level 3 is the default.

from pyfock import Basis
from pyfock import Mol
from pyfock import DFT
from pyfock import Grids

ncores = 4

mol = Mol(coordfile='h2o.xyz')
basis = Basis(mol, {'all': Basis.load(mol=mol, basis_name='def2-SVP')})
auxbasis = Basis(mol, {'all': Basis.load(mol=mol, basis_name='def2-universal-jfit')})


# ---------------------------------------------------------------
# 1. Standalone grids with the Grids class
# ---------------------------------------------------------------

# Default scheme, level 3
grids = Grids(mol, level=3, ncores=ncores)
print('Number of grid points:', grids.size)
print('Coordinates (Bohr):', grids.coords.shape)      # (N, 3) array
print('Weights:', grids.weights.shape)                # (N,) array
print('Atom index of each point:', grids.atom_idx)    # which atom a point belongs to
print('Radial and angular points per element:', grids.element_points)

# A coarser and a finer grid
grids_coarse = Grids(mol, level=1, ncores=ncores)
grids_fine = Grids(mol, level=5, ncores=ncores)

# Options of the default scheme
# pruning: 'regions' (default) uses fewer angular points close to the nucleus and far away,
#          None keeps the full angular grid on every radial shell
grids_unpruned = Grids(mol, level=3, ncores=ncores, pruning=None)

# size_adjustment: how the Becke partitioning shifts the cell boundary between atoms of
#                  different size: 'treutler' (default), 'becke' or None
grids_becke = Grids(mol, level=3, ncores=ncores, size_adjustment='becke')

# points_per_element: set the number of radial points and the largest angular grid yourself
grids_custom = Grids(mol, level=3, ncores=ncores, points_per_element={'O': (75, 302), 'H': (50, 194)})

# numgrid scheme
# preset 'compact' (default) gives grids of a similar size to the default scheme,
# preset 'dense' gives the previous (denser) PyFock grids
grids_numgrid = Grids(mol, level=3, ncores=ncores, scheme='numgrid')
grids_numgrid_dense = Grids(mol, level=3, ncores=ncores, scheme='numgrid', preset='dense')

# numgrid parameters can also be set directly: radial precision and (min, max) angular points
grids_numgrid_custom = Grids(mol, level=3, ncores=ncores, scheme='numgrid', radial_precision=1e-10, angular_points=(86, 434))


# ---------------------------------------------------------------
# 2. DFT calculations with the different grids
# ---------------------------------------------------------------

xc = [101, 130]  # PBE

# Default: 'treutler' scheme at gridsLevel=3
dftObj = DFT(mol, basis, auxbasis, xc=xc, gridsLevel=3, ncores=ncores)
energy, dmat = dftObj.scf()
print('Energy with the default grid (level 3):', energy)

# Finer grid
dftObj = DFT(mol, basis, auxbasis, xc=xc, gridsLevel=5, ncores=ncores)
energy, dmat = dftObj.scf()
print('Energy with level 5:', energy)

# Options of the default scheme are passed through grids_options
dftObj = DFT(mol, basis, auxbasis, xc=xc, gridsLevel=3, ncores=ncores)
dftObj.grids_options = {'pruning': None, 'size_adjustment': 'becke'}
energy, dmat = dftObj.scf()
print('Energy without pruning and with Becke size adjustment:', energy)

# numgrid scheme
dftObj = DFT(mol, basis, auxbasis, xc=xc, gridsLevel=3, ncores=ncores)
dftObj.grids_scheme = 'numgrid'
dftObj.grids_preset = 'dense'     # or 'compact'
energy, dmat = dftObj.scf()
print('Energy with the numgrid dense grid:', energy)

# A grid generated beforehand can be passed directly
dftObj = DFT(mol, basis, auxbasis, xc=xc, ncores=ncores, grids=grids_fine)
energy, dmat = dftObj.scf()
print('Energy with a precomputed level-5 grid:', energy)

# Number of grid points actually used in the last calculation (after pruning points with negligible density)
print('Grid points used:', dftObj.grids.coords.shape[0])
