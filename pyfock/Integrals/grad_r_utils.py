import numpy as np
from numba import njit, prange


def basis_atom_gradient_from_r(grad_r, basis, slice=None):
    if slice is None:
        slice = [0, basis.bfs_nao, 0, basis.bfs_nao]

    start_row = int(slice[0])
    end_row = int(slice[1])
    start_col = int(slice[2])
    end_col = int(slice[3])

    bfs_atoms = np.asarray(basis.bfs_atoms, dtype=np.int64)
    natoms = int(np.max(bfs_atoms)) + 1

    return basis_atom_gradient_from_r_internal(
        grad_r,
        bfs_atoms,
        natoms,
        start_row,
        end_row,
        start_col,
        end_col,
    )


@njit(parallel=True, cache=True)
def basis_atom_gradient_from_r_internal(
    grad_r,
    bfs_atoms,
    natoms,
    start_row,
    end_row,
    start_col,
    end_col,
):
    num_rows = end_row - start_row
    num_cols = end_col - start_col
    grad_atoms = np.zeros((natoms, 3, num_rows, num_cols))

    for i in prange(num_rows):
        atom_i = bfs_atoms[start_row + i]
        for j in range(num_cols):
            atom_j = bfs_atoms[start_col + j]
            if atom_i != atom_j:
                for direction in range(3):
                    value = grad_r[direction, i, j]
                    grad_atoms[atom_i, direction, i, j] += value
                    grad_atoms[atom_j, direction, i, j] -= value

    return grad_atoms
