import numpy as np
from numba import njit, prange

from .nuc_mat_grad_symm import primitive_nuc_center_grad


def nuc_mat_grad_r_symm(basis, mol, slice=None, sqrt_ints4c2e_diag=None, wrt_atoms=False):
    bfs_coords = np.array([basis.bfs_coords])
    bfs_contr_prim_norms = np.array([basis.bfs_contr_prim_norms])
    bfs_lmn = np.array([basis.bfs_lmn])
    bfs_nprim = np.array([basis.bfs_nprim])
    bfs_atoms = np.array([basis.bfs_atoms])
    coordsBohrs = np.array([mol.coordsBohrs])
    Z = np.array([mol.Zcharges])
    natoms = mol.natoms

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

    if sqrt_ints4c2e_diag is None:
        sqrt_ints4c2e_diag = np.zeros((1, 1), dtype=np.float64)
        isSchwarz = False
    else:
        isSchwarz = True

    if wrt_atoms:
        return nuc_mat_grad_r_to_atoms_symm_internal(
            natoms,
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
            Z[0],
            coordsBohrs[0],
            sqrt_ints4c2e_diag,
            isSchwarz,
        )

    return nuc_mat_grad_r_symm_internal(
        natoms,
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
        Z[0],
        coordsBohrs[0],
        sqrt_ints4c2e_diag,
        isSchwarz,
    )


@njit(parallel=True, cache=True, fastmath=True, error_model="numpy")
def nuc_mat_grad_r_symm_internal(
    natoms,
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
    Z,
    coordsMol,
    sqrt_ints4c2e_diag,
    isSchwarz=False,
):
    num_rows = end_row - start_row
    num_cols = end_col - start_col
    V_r = np.zeros((3, num_rows, num_cols))

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
        Ni = bfs_contr_prim_norms[i]
        lmni = bfs_lmn[i]
        for j in range(start_col, end_col):
            if not (lower_tri or upper_tri or (both_tri_symm and j <= i) or both_tri_nonsymm):
                continue
            if isSchwarz:
                sqrt_ij = sqrt_ints4c2e_diag[i, j]
                if sqrt_ij * sqrt_ij < 1e-13:
                    continue

            J = bfs_coords[j]
            Nj = bfs_contr_prim_norms[j]
            lmnj = bfs_lmn[j]
            IJsq = np.sum((I - J) ** 2)
            bra_sum = np.zeros(3)
            ket_sum = np.zeros(3)

            for ik in range(bfs_nprim[i]):
                alphaik = bfs_expnts[i, ik]
                dik = bfs_coeffs[i, ik]
                Nik = bfs_prim_norms[i, ik]
                temp_alphaikIjsq = alphaik * IJsq
                temp_NiNjNikdik = Ni * Nj * Nik * dik
                for jk in range(bfs_nprim[j]):
                    alphajk = bfs_expnts[j, jk]
                    gamma_inv = 1.0 / (alphaik + alphajk)
                    screenfactor = np.exp(-temp_alphaikIjsq * alphajk * gamma_inv)
                    if screenfactor < 1.0e-8:
                        continue

                    prefactor = temp_NiNjNikdik * bfs_coeffs[j, jk] * bfs_prim_norms[j, jk]
                    if np.abs(prefactor) < 1.0e-8:
                        continue

                    for iatom in range(natoms):
                        Rc = coordsMol[iatom]
                        Zc = Z[iatom]
                        bra_grad = primitive_nuc_center_grad(
                            lmni, lmnj, alphaik, alphajk, I, J, Rc, Zc
                        )
                        for direction in range(3):
                            bra_sum[direction] += prefactor * bra_grad[direction]

                        if both_tri_symm:
                            ket_grad = primitive_nuc_center_grad(
                                lmnj, lmni, alphajk, alphaik, J, I, Rc, Zc
                            )
                            for direction in range(3):
                                ket_sum[direction] += prefactor * ket_grad[direction]

            for direction in range(3):
                V_r[direction, i - start_row, j - start_col] = bra_sum[direction]
                if both_tri_symm:
                    V_r[direction, j - start_col, i - start_row] = ket_sum[direction]

    return V_r


@njit(parallel=True, cache=True, fastmath=True, error_model="numpy")
def nuc_mat_grad_r_to_atoms_symm_internal(
    natoms,
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
    Z,
    coordsMol,
    sqrt_ints4c2e_diag,
    isSchwarz=False,
):
    num_rows = end_row - start_row
    num_cols = end_col - start_col
    V = np.zeros((natoms, 3, num_rows, num_cols))

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
        Ni = bfs_contr_prim_norms[i]
        lmni = bfs_lmn[i]
        atom_i = bfs_atoms[i]
        for j in range(start_col, end_col):
            if not (lower_tri or upper_tri or (both_tri_symm and j <= i) or both_tri_nonsymm):
                continue
            if isSchwarz:
                sqrt_ij = sqrt_ints4c2e_diag[i, j]
                if sqrt_ij * sqrt_ij < 1e-13:
                    continue

            J = bfs_coords[j]
            Nj = bfs_contr_prim_norms[j]
            lmnj = bfs_lmn[j]
            atom_j = bfs_atoms[j]
            IJsq = np.sum((I - J) ** 2)

            for ik in range(bfs_nprim[i]):
                alphaik = bfs_expnts[i, ik]
                dik = bfs_coeffs[i, ik]
                Nik = bfs_prim_norms[i, ik]
                temp_alphaikIjsq = alphaik * IJsq
                temp_NiNjNikdik = Ni * Nj * Nik * dik
                for jk in range(bfs_nprim[j]):
                    alphajk = bfs_expnts[j, jk]
                    gamma_inv = 1.0 / (alphaik + alphajk)
                    screenfactor = np.exp(-temp_alphaikIjsq * alphajk * gamma_inv)
                    if screenfactor < 1.0e-8:
                        continue

                    prefactor = temp_NiNjNikdik * bfs_coeffs[j, jk] * bfs_prim_norms[j, jk]
                    if np.abs(prefactor) < 1.0e-8:
                        continue

                    for iatom in range(natoms):
                        Rc = coordsMol[iatom]
                        Zc = Z[iatom]
                        bra_grad = primitive_nuc_center_grad(
                            lmni, lmnj, alphaik, alphajk, I, J, Rc, Zc
                        )
                        ket_grad = primitive_nuc_center_grad(
                            lmnj, lmni, alphajk, alphaik, J, I, Rc, Zc
                        )

                        for direction in range(3):
                            bra_val = prefactor * bra_grad[direction]
                            ket_val = prefactor * ket_grad[direction]
                            V[atom_i, direction, i - start_row, j - start_col] += bra_val
                            V[atom_j, direction, i - start_row, j - start_col] += ket_val
                            V[iatom, direction, i - start_row, j - start_col] -= bra_val + ket_val

    if both_tri_symm:
        for i in prange(start_row, end_row):
            for j in range(start_col, end_col):
                if j > i:
                    V[:, :, i - start_row, j - start_col] = V[:, :, j - start_col, i - start_row]

    return V
