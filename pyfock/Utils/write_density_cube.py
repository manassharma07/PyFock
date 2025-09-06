import numba
import numpy as np
from opt_einsum import contract
from pyfock import Integrals
from .Cube import generate_cube_coords, write_cube_file
import os

def write_density_cube(mol, basis, dmat, cube_file, nx=100, ny=100, nz=100, ncores=1):
    """Write the electron density of a molecule as a cube file."""
    numba.set_num_threads(ncores)
    os.environ['OMP_NUM_THREADS'] = str(ncores)
    os.environ["OPENBLAS_NUM_THREADS"] = str(ncores) # export OPENBLAS_NUM_THREADS=4 
    os.environ["MKL_NUM_THREADS"] = str(ncores) # export MKL_NUM_THREADS=4
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(ncores) # export VECLIB_MAXIMUM_THREADS=4
    os.environ["NUMEXPR_NUM_THREADS"] = str(ncores) # export NUMEXPR_NUM_THREADS=4


    # Generate a grid of coordinates for the cube file
    coords, origin, spacing = generate_cube_coords(mol, nx=nx, ny=ny, nz=nz, padding=5.0)  # shape: (ncoord, 3)
    # Calculate ao and density values at these coordinates
    bf_values = Integrals.bf_val_helpers.eval_bfs(basis, coords, parallel=True, non_zero_indices=None) # shape: (ncoord, nao)
    dens_values = Integrals.bf_val_helpers.eval_rho(bf_values, dmat) # shape: (ncoord, 1)
    # dens_values = contract('mj,mj->m', bf_values @ dmat, bf_values)

    # Write the cube file
    write_cube_file(cube_file, mol, dens_values, origin, spacing, nx, ny, nz)


    