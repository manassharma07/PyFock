import numpy as np
from numba import njit, prange

from .grad_r_utils import basis_atom_gradient_from_r
from .integral_helpers import calcS


def overlap_mat_grad_r_symm(basis, slice=None, wrt_atoms=False):
    bfs_coords = np.array([basis.bfs_coords])
    bfs_contr_prim_norms = np.array([basis.bfs_contr_prim_norms])
    bfs_lmn = np.array([basis.bfs_lmn])
    bfs_nprim = np.array([basis.bfs_nprim])
    bfs_atoms = np.array([basis.bfs_atoms])

    maxnprim = max(basis.bfs_nprim)
    bfs_coeffs = np.zeros([basis.bfs_nao, maxnprim])
    bfs_expnts = np.zeros([basis.bfs_nao, maxnprim])
    bfs_prim_norms = np.zeros([basis.bfs_nao, maxnprim])
    for i in range(basis.bfs_nao):
        for j in range(basis.bfs_nprim[i]):
            bfs_coeffs[i, j] = basis.bfs_coeffs[i][j]
            bfs_expnts[i, j] = basis.bfs_expnts[i][j]
            bfs_prim_norms[i, j] = basis.bfs_prim_norms[i][j]

    if slice is None:
        slice = [0, basis.bfs_nao, 0, basis.bfs_nao]

    a = int(slice[0])
    b = int(slice[1])
    c = int(slice[2])
    d = int(slice[3])

    dS_r = overlap_mat_grad_r_symm_internal(
        bfs_atoms[0],
        bfs_coords[0],
        bfs_contr_prim_norms[0],
        bfs_lmn[0],
        bfs_nprim[0],
        bfs_coeffs,
        bfs_prim_norms,
        bfs_expnts,
        a,
        b,
        c,
        d,
    )

    if wrt_atoms:
        return basis_atom_gradient_from_r(dS_r, basis, slice)
    return dS_r


@njit(parallel=True, fastmath=True, cache=True)
def overlap_mat_grad_r_symm_internal(
    bfs_atoms,
    bfs_coords,
    bfs_contr_prim_norms,
    bfs_lmn,
    bfs_nprim,
    bfs_coeffs,
    bfs_prim_norms,
    bfs_expnts,
    start_row,
    end_row,
    start_col,
    end_col,
):
    num_rows = end_row - start_row
    num_cols = end_col - start_col
    dS_r = np.zeros((3, num_rows, num_cols))

    upper_tri = False
    lower_tri = False
    both_tri_symm = False
    both_tri_nonsymm = False
    if end_row <= start_col:
        upper_tri = True
    elif start_row >= end_col:
        lower_tri = True
    elif start_row == start_col and end_row == end_col:
        both_tri_symm = True
    else:
        both_tri_nonsymm = True

    for i in prange(start_row, end_row):
        I = bfs_coords[i]
        lmni = bfs_lmn[i]
        Ni = bfs_contr_prim_norms[i]
        for j in range(start_col, end_col):
            if lower_tri or upper_tri or (both_tri_symm and j <= i) or both_tri_nonsymm:
                result = np.zeros(3)
                J = bfs_coords[j]
                IJ = I - J
                tempfac = np.sum(IJ**2)
                Nj = bfs_contr_prim_norms[j]
                lmnj = bfs_lmn[j]

                for ik in range(bfs_nprim[i]):
                    alphaik = bfs_expnts[i][ik]
                    dik = bfs_coeffs[i][ik]
                    Nik = bfs_prim_norms[i][ik]
                    for jk in range(bfs_nprim[j]):
                        alphajk = bfs_expnts[j][jk]
                        gamma = alphaik + alphajk
                        screenfactor = np.exp(-alphaik * alphajk / gamma * tempfac)
                        if abs(screenfactor) < 1.0e-12:
                            continue

                        djk = bfs_coeffs[j][jk]
                        Njk = bfs_prim_norms[j][jk]

                        P = (alphaik * I + alphajk * J) / gamma
                        PI = P - I
                        PJ = P - J

                        temp = dik * djk
                        temp = temp * Nik * Njk
                        temp = temp * Ni * Nj

                        for direction in range(3):
                            if direction == 0:
                                lfactor = lmni[0]
                                Sx = calcS(lmni[0] - 1, lmnj[0], gamma, PI[0], PJ[0])
                                Sy = calcS(lmni[1], lmnj[1], gamma, PI[1], PJ[1])
                                Sz = calcS(lmni[2], lmnj[2], gamma, PI[2], PJ[2])
                            if direction == 1:
                                lfactor = lmni[1]
                                Sx = calcS(lmni[0], lmnj[0], gamma, PI[0], PJ[0])
                                Sy = calcS(lmni[1] - 1, lmnj[1], gamma, PI[1], PJ[1])
                                Sz = calcS(lmni[2], lmnj[2], gamma, PI[2], PJ[2])
                            if direction == 2:
                                lfactor = lmni[2]
                                Sx = calcS(lmni[0], lmnj[0], gamma, PI[0], PJ[0])
                                Sy = calcS(lmni[1], lmnj[1], gamma, PI[1], PJ[1])
                                Sz = calcS(lmni[2] - 1, lmnj[2], gamma, PI[2], PJ[2])

                            tempA = -lfactor * temp * screenfactor * Sx * Sy * Sz

                            if direction == 0:
                                Sx = calcS(lmni[0] + 1, lmnj[0], gamma, PI[0], PJ[0])
                            if direction == 1:
                                Sy = calcS(lmni[1] + 1, lmnj[1], gamma, PI[1], PJ[1])
                            if direction == 2:
                                Sz = calcS(lmni[2] + 1, lmnj[2], gamma, PI[2], PJ[2])

                            tempB = 2 * alphaik * temp * screenfactor * Sx * Sy * Sz
                            result[direction] += tempA + tempB

                for direction in range(3):
                    dS_r[direction, i - start_row, j - start_col] = result[direction]
                    if both_tri_symm:
                        dS_r[direction, j - start_col, i - start_row] = -result[direction]

    return dS_r
