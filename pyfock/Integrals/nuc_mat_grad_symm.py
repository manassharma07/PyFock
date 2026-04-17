import numpy as np
from numba import njit, prange

from .integral_helpers import c2k, vlriPartial, Fboys


def nuc_mat_grad_symm(basis, mol, slice=None, sqrt_ints4c2e_diag=None):
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

    return nuc_mat_grad_symm_internal(
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
def nuc_mat_grad_symm_internal(
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
                    gamma = alphaik + alphajk
                    gamma_inv = 1.0 / gamma
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

                        for dir in range(3):
                            bra_val = prefactor * bra_grad[dir]
                            ket_val = prefactor * ket_grad[dir]
                            V[atom_i, dir, i - start_row, j - start_col] += bra_val
                            V[atom_j, dir, i - start_row, j - start_col] += ket_val
                            V[iatom, dir, i - start_row, j - start_col] -= (bra_val + ket_val)

    if both_tri_symm:
        for i in prange(start_row, end_row):
            for j in range(start_col, end_col):
                if j > i:
                    V[:, :, i - start_row, j - start_col] = V[:, :, j - start_col, i - start_row]

    return V


@njit(cache=True, fastmath=True, error_model="numpy")
def primitive_nuc_center_grad(lmn_left, lmn_right, alpha_left, alpha_right, left_center, right_center, nuc_center, nuc_charge):
    grad = np.zeros(3)
    for dir in range(3):
        lfactor = lmn_left[dir]
        minus_lmn = np.array([lmn_left[0], lmn_left[1], lmn_left[2]])
        plus_lmn = np.array([lmn_left[0], lmn_left[1], lmn_left[2]])
        minus_lmn[dir] -= 1
        plus_lmn[dir] += 1

        tempA = 0.0
        if lfactor > 0:
            tempA = -lfactor * primitive_nuc_single_center(
                minus_lmn[0], minus_lmn[1], minus_lmn[2],
                lmn_right[0], lmn_right[1], lmn_right[2],
                alpha_left, alpha_right, left_center, right_center, nuc_center, nuc_charge,
            )
        tempB = 2.0 * alpha_left * primitive_nuc_single_center(
            plus_lmn[0], plus_lmn[1], plus_lmn[2],
            lmn_right[0], lmn_right[1], lmn_right[2],
            alpha_left, alpha_right, left_center, right_center, nuc_center, nuc_charge,
        )
        grad[dir] = tempA + tempB
    return grad


@njit(cache=True, fastmath=True, error_model="numpy")
def primitive_nuc_single_center(la, ma, na, lb, mb, nb, alphaik, alphajk, I, J, Rc, Zc):
    PIx2 = 6.283185307179586
    IJ = I - J
    IJsq = np.sum(IJ ** 2)
    gamma = alphaik + alphajk
    gamma_inv = 1.0 / gamma
    screenfactor = np.exp(-alphaik * alphajk * gamma_inv * IJsq)
    if screenfactor < 1.0e-12:
        return 0.0

    epsilon = 0.25 * gamma_inv
    P = (alphaik * I + alphajk * J) * gamma_inv
    PI = P - I
    PJ = P - J
    PC = P - Rc
    tempfac = -Zc * (PIx2 * gamma_inv) * screenfactor

    max_l = la + lb
    max_m = ma + mb
    max_n = na + nb
    max_boys = max_l + max_m + max_n + 1

    facl = np.zeros(max_l + 1)
    facm = np.zeros(max_m + 1)
    facn = np.zeros(max_n + 1)
    F_ = np.zeros(max_boys)
    vmsj = np.zeros((max_m + 1, max_m // 2 + 1, max_m // 2 + 1))
    vntk = np.zeros((max_n + 1, max_n // 2 + 1, max_n // 2 + 1))

    for l in range(max_l + 1):
        facl[l] = c2k(l, la, lb, PI[0], PJ[0])
    for m in range(max_m + 1):
        facm[m] = c2k(m, ma, mb, PI[1], PJ[1])
    for n in range(max_n + 1):
        facn[n] = c2k(n, na, nb, PI[2], PJ[2])

    temp_gamma_sum_PCsq = gamma * np.sum(PC ** 2)
    for li in range(max_boys):
        F_[li] = Fboys(li, temp_gamma_sum_PCsq)

    for m in range(max_m + 1):
        for s in range(m // 2 + 1):
            for j1 in range((m - 2 * s) // 2 + 1):
                vmsj[m, s, j1] = vlriPartial(PC[1], m, s, j1) * epsilon ** (s + j1) * facm[m]
    for n in range(max_n + 1):
        for t in range(n // 2 + 1):
            for k in range((n - 2 * t) // 2 + 1):
                vntk[n, t, k] = vlriPartial(PC[2], n, t, k) * epsilon ** (t + k) * facn[n]

    sum_Vl = 0.0
    for l in range(max_l + 1):
        for r in range(l // 2 + 1):
            for i1 in range((l - 2 * r) // 2 + 1):
                v_lri = vlriPartial(PC[0], l, r, i1) * epsilon ** (r + i1) * facl[l]
                sum_Vm = 0.0
                for m in range(max_m + 1):
                    for s in range(m // 2 + 1):
                        for j1 in range((m - 2 * s) // 2 + 1):
                            v_msj = vmsj[m, s, j1]
                            sum_Vn = 0.0
                            for n in range(max_n + 1):
                                for t in range(n // 2 + 1):
                                    for k in range((n - 2 * t) // 2 + 1):
                                        v_ntk = vntk[n, t, k]
                                        F = F_[l + m + n - 2 * (r + s + t) - (i1 + j1 + k)]
                                        sum_Vn += v_ntk * F
                            sum_Vm += v_msj * sum_Vn
                sum_Vl += v_lri * sum_Vm
    return tempfac * sum_Vl