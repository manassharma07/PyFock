import numpy as np
from numba import njit, prange
from .integral_helpers import Fboys

def os_4c2e_symm(basis):
    """
    Compute four-center two-electron (4c2e) electron repulsion integrals (ERIs) 
    using the Obara-Saika scheme with exploitation of 8-fold permutational 
    symmetry.

    This function evaluates integrals of the form (A B | C D), where A, B, C, D 
    are basis functions from the same primary basis set. It uses Numba-accelerated 
    backends and shell-based computation for efficiency.

    Parameters
    ----------
    basis : object
        Primary basis set object containing shell and basis function information

    Returns
    -------
    ints4c2e : ndarray
        The computed 4-center 2-electron integrals, shape (Nbf, Nbf, Nbf, Nbf)
    """
    
    #Convert basis data to numpy arrays for Numba
    nbf = basis.bfs_nao
    nshells = len(basis.shells)
    
    #Shell data
    shell_L = np.array([basis.bfs_lm[i] for i in basis.shell_bfs_offset], dtype=np.int32)
    shell_centers = np.array([basis.bfs_coords[i] for i in basis.shell_bfs_offset], dtype=np.float64)
    shell_bfs_offset = np.array(basis.shell_bfs_offset, dtype=np.int32)
    bfs_nbfshell = np.array(basis.bfs_nbfshell, dtype=np.int32)
    
    #Basis function data
    bfs_coords = np.array(basis.bfs_coords, dtype=np.float64)
    bfs_contr_prim_norms = np.array(basis.bfs_contr_prim_norms, dtype=np.float64)
    bfs_lmn = np.array(basis.bfs_lmn, dtype=np.int32)
    bfs_nprim = np.array(basis.bfs_nprim, dtype=np.int32)
    bfs_shell_index = np.array(basis.bfs_shell_index, dtype=np.int32)
    
    #Primitive data
    maxnprim = max(basis.bfs_nprim)
    bfs_coeffs = np.zeros((nbf, maxnprim), dtype=np.float64)
    bfs_expnts = np.zeros((nbf, maxnprim), dtype=np.float64)
    bfs_prim_norms = np.zeros((nbf, maxnprim), dtype=np.float64)
    
    for i in range(nbf):
        for j in range(basis.bfs_nprim[i]):
            bfs_coeffs[i, j] = basis.bfs_coeffs[i][j]
            bfs_expnts[i, j] = basis.bfs_expnts[i][j]
            bfs_prim_norms[i, j] = basis.bfs_prim_norms[i][j]
    
    ints4c2e = os_4c2e_symm_internal(
        nbf, nshells, shell_L, shell_centers, shell_bfs_offset, bfs_nbfshell,
        bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_shell_index,
        bfs_coeffs, bfs_expnts, bfs_prim_norms
    )
    
    return ints4c2e

@njit(parallel=True, cache=True, fastmath=True, nogil=True, error_model="numpy")
def os_4c2e_symm_internal(nbf, nshells, shell_L, shell_centers, shell_bfs_offset,
                          bfs_nbfshell, bfs_coords, bfs_contr_prim_norms, bfs_lmn,
                          bfs_nprim, bfs_shell_index, bfs_coeffs, bfs_expnts, bfs_prim_norms):
    
    fourC2E = np.zeros((nbf, nbf, nbf, nbf), dtype=np.float64)


    pi = np.pi
    two_pi_52 = 2.0 * pi ** 2.5

    n_ab = nshells * (nshells + 1) // 2

    #Let store all the imp things like shell pair data beforehand
    AB_vecs = np.zeros((n_ab, 3), dtype=np.float64)
    AB_sqs = np.zeros(n_ab, dtype=np.float64)
    ab_shell_a = np.zeros(n_ab, dtype=np.int64)
    ab_shell_b = np.zeros(n_ab, dtype=np.int64)

    idx = 0
    for a in range(nshells):
        for b in range(a + 1):
            ab_shell_a[idx] = a
            ab_shell_b[idx] = b
            dx = shell_centers[a, 0] - shell_centers[b, 0]
            dy = shell_centers[a, 1] - shell_centers[b, 1]
            dz = shell_centers[a, 2] - shell_centers[b, 2]
            AB_vecs[idx, 0] = dx
            AB_vecs[idx, 1] = dy
            AB_vecs[idx, 2] = dz
            AB_sqs[idx] = dx * dx + dy * dy + dz * dz
            idx += 1

    #find out some max sizes of things
    max_L = 0
    max_nprim = 0
    max_nbf_shell = 0
    for i in range(nshells):
        if shell_L[i] > max_L:
            max_L = shell_L[i]
        ibf0 = shell_bfs_offset[i]
        if bfs_nprim[ibf0] > max_nprim:
            max_nprim = bfs_nprim[ibf0]
        if bfs_nbfshell[i] > max_nbf_shell:
            max_nbf_shell = bfs_nbfshell[i]

    max_L_all = 4 * max_L
    max_dim = 2 * max_L + 1
    max_dim_m = max_L_all + 1
    max_prim_pairs = max_nprim * max_nprim
    max_hrr = (max_L + 1) ** 3

    for ab_idx in prange(n_ab):
        ish_a = ab_shell_a[ab_idx]
        ish_b = ab_shell_b[ab_idx]

        L_a = shell_L[ish_a]
        L_b = shell_L[ish_b]
        L_bra = L_a + L_b
        bf_a_start = shell_bfs_offset[ish_a]
        bf_b_start = shell_bfs_offset[ish_b]
        nbf_a = bfs_nbfshell[ish_a]
        nbf_b = bfs_nbfshell[ish_b]

        X_AB = AB_vecs[ab_idx]
        AB_sq = AB_sqs[ab_idx]

        center_a = shell_centers[ish_a]
        center_b = shell_centers[ish_b]

        ibf_a0 = bf_a_start
        ibf_b0 = bf_b_start
        nprimi = bfs_nprim[ibf_a0]
        nprimj = bfs_nprim[ibf_b0]

        # lots of allocations for scratch data (need to see if there is a better way to do this?)
        bra_p = np.empty(max_prim_pairs, dtype=np.float64)
        bra_K = np.empty(max_prim_pairs, dtype=np.float64)
        bra_Px = np.empty(max_prim_pairs, dtype=np.float64)
        bra_Py = np.empty(max_prim_pairs, dtype=np.float64)
        bra_Pz = np.empty(max_prim_pairs, dtype=np.float64)
        bra_PAx = np.empty(max_prim_pairs, dtype=np.float64)
        bra_PAy = np.empty(max_prim_pairs,dtype=np.float64)
        bra_PAz = np.empty(max_prim_pairs,dtype=np.float64)
        bra_alpha = np.empty(max_prim_pairs,dtype=np.float64)
        bra_beta = np.empty(max_prim_pairs,dtype=np.float64)
        bra_ipa = np.empty(max_prim_pairs,dtype=np.int64)
        bra_ipb = np.empty(max_prim_pairs, dtype=np.int64)

        ket_q =np.empty(max_prim_pairs, dtype=np.float64)
        ket_K =np.empty(max_prim_pairs, dtype=np.float64)
        ket_Qx =np.empty(max_prim_pairs, dtype=np.float64)
        ket_Qy =np.empty(max_prim_pairs, dtype=np.float64)
        ket_Qz =np.empty(max_prim_pairs, dtype=np.float64)
        ket_QCx =np.empty(max_prim_pairs, dtype=np.float64)
        ket_QCy = np.empty(max_prim_pairs, dtype=np.float64)
        ket_QCz = np.empty(max_prim_pairs, dtype=np.float64)
        ket_gamma = np.empty(max_prim_pairs, dtype=np.float64)
        ket_delta = np.empty(max_prim_pairs, dtype=np.float64)
        ket_ipc = np.empty(max_prim_pairs, dtype=np.int64)
        ket_ipd = np.empty(max_prim_pairs, dtype=np.int64)

        V_vrr = np.zeros((max_dim_m, max_dim, max_dim, max_dim,
                          max_dim, max_dim, max_dim), dtype=np.float64)

        ERI_shell = np.zeros((max_nbf_shell, max_nbf_shell,
                              max_nbf_shell, max_nbf_shell), dtype=np.float64)

        bra_hrr_coeffs = np.empty((max_nbf_shell, max_nbf_shell, max_hrr), dtype=np.float64)
        bra_hrr_ax = np.empty((max_nbf_shell, max_nbf_shell, max_hrr), dtype=np.int64)
        bra_hrr_ay = np.empty((max_nbf_shell, max_nbf_shell, max_hrr), dtype=np.int64)
        bra_hrr_az = np.empty((max_nbf_shell, max_nbf_shell, max_hrr), dtype=np.int64)
        bra_hrr_n = np.empty((max_nbf_shell, max_nbf_shell), dtype=np.int64)

        ket_hrr_coeffs = np.empty((max_nbf_shell, max_nbf_shell, max_hrr), dtype=np.float64)
        ket_hrr_cx = np.empty((max_nbf_shell, max_nbf_shell, max_hrr), dtype=np.int64)
        ket_hrr_cy = np.empty((max_nbf_shell, max_nbf_shell, max_hrr), dtype=np.int64)
        ket_hrr_cz = np.empty((max_nbf_shell, max_nbf_shell, max_hrr), dtype=np.int64)
        ket_hrr_n = np.empty((max_nbf_shell, max_nbf_shell), dtype=np.int64)

        X_PA = np.empty(3, dtype=np.float64)
        X_QC = np.empty(3, dtype=np.float64)
        X_WP = np.empty(3, dtype=np.float64)
        X_WQ = np.empty(3, dtype=np.float64)
        F = np.empty(max_dim_m, dtype=np.float64)

        
        n_bra_pairs = 0
        for ipa in range(nprimi):
            alpha = bfs_expnts[ibf_a0, ipa]
            for ipb in range(nprimj):
                beta = bfs_expnts[ibf_b0, ipb]
                p = alpha + beta
                mu = alpha * beta / p
                K_AB_val = np.exp(-mu * AB_sq)

                if abs(K_AB_val) < 1.0e-8:
                    continue

                ii = n_bra_pairs
                bra_p[ii] = p
                bra_K[ii] = K_AB_val
                bra_alpha[ii] = alpha
                bra_beta[ii] = beta
                bra_ipa[ii] = ipa
                bra_ipb[ii] = ipb

                inv_p = 1.0 / p
                Px = (alpha * center_a[0] + beta * center_b[0]) * inv_p
                Py = (alpha * center_a[1] + beta * center_b[1]) * inv_p
                Pz = (alpha * center_a[2] + beta * center_b[2]) * inv_p
                bra_Px[ii] = Px
                bra_Py[ii] = Py
                bra_Pz[ii] = Pz
                bra_PAx[ii] = Px - center_a[0]
                bra_PAy[ii] = Py - center_a[1]
                bra_PAz[ii] = Pz - center_a[2]

                n_bra_pairs += 1

        if n_bra_pairs == 0:
            continue

        
        for ic_a in range(nbf_a):
            ibf_a = bf_a_start + ic_a
            ax, ay, az = bfs_lmn[ibf_a, 0], bfs_lmn[ibf_a, 1], bfs_lmn[ibf_a, 2]
            for ic_b in range(nbf_b):
                ibf_b = bf_b_start + ic_b
                bx, by, bz = bfs_lmn[ibf_b, 0], bfs_lmn[ibf_b, 1], bfs_lmn[ibf_b, 2]
                count = 0
                for px in range(bx + 1):
                    binom_x = _binom(bx, px)
                    pow_x = X_AB[0] ** (bx - px)
                    ax_f = ax + px
                    for py in range(by + 1):
                        binom_y = _binom(by, py)
                        pow_y = X_AB[1] ** (by - py)
                        ay_f = ay + py
                        for pz in range(bz + 1):
                            binom_z = _binom(bz, pz)
                            pow_z = X_AB[2] ** (bz - pz)
                            az_f = az + pz
                            coeff = binom_x * pow_x * binom_y * pow_y * binom_z * pow_z
                            bra_hrr_coeffs[ic_a, ic_b, count] = coeff
                            bra_hrr_ax[ic_a, ic_b, count] = ax_f
                            bra_hrr_ay[ic_a, ic_b, count] = ay_f
                            bra_hrr_az[ic_a, ic_b, count] = az_f
                            count += 1
                bra_hrr_n[ic_a, ic_b] = count

        
        for cd_idx in range(ab_idx + 1):
            ish_c = ab_shell_a[cd_idx]
            ish_d = ab_shell_b[cd_idx]

            L_c = shell_L[ish_c]
            L_d = shell_L[ish_d]
            L_ket = L_c + L_d
            bf_c_start = shell_bfs_offset[ish_c]
            bf_d_start = shell_bfs_offset[ish_d]
            nbf_c = bfs_nbfshell[ish_c]
            nbf_d = bfs_nbfshell[ish_d]

            center_c = shell_centers[ish_c]
            center_d = shell_centers[ish_d]

            X_CD = AB_vecs[cd_idx]
            CD_sq = AB_sqs[cd_idx]

            L_all = L_bra + L_ket
            dim_bra = L_bra + 1
            dim_ket = L_ket + 1
            dim_m = L_all + 1

            ibf_c0 = bf_c_start
            ibf_d0 = bf_d_start
            nprimk = bfs_nprim[ibf_c0]
            npriml = bfs_nprim[ibf_d0]

            
            n_ket_pairs = 0
            for ipc in range(nprimk):
                gamma = bfs_expnts[ibf_c0, ipc]
                for ipd in range(npriml):
                    delta = bfs_expnts[ibf_d0, ipd]
                    q = gamma + delta
                    mu_cd = gamma * delta / q
                    K_CD_val = np.exp(-mu_cd * CD_sq)

                    if abs(K_CD_val) < 1.0e-8:
                        continue

                    ii = n_ket_pairs
                    ket_q[ii] = q
                    ket_K[ii] = K_CD_val
                    ket_gamma[ii] = gamma
                    ket_delta[ii] = delta
                    ket_ipc[ii] = ipc
                    ket_ipd[ii] = ipd

                    inv_q = 1.0 / q
                    Qx = (gamma * center_c[0] + delta * center_d[0]) * inv_q
                    Qy = (gamma * center_c[1] + delta * center_d[1]) * inv_q
                    Qz = (gamma * center_c[2] + delta * center_d[2]) * inv_q
                    ket_Qx[ii] = Qx
                    ket_Qy[ii] = Qy
                    ket_Qz[ii] = Qz
                    ket_QCx[ii] = Qx - center_c[0]
                    ket_QCy[ii] = Qy - center_c[1]
                    ket_QCz[ii] = Qz - center_c[2]

                    n_ket_pairs += 1

            if n_ket_pairs == 0:
                continue

            #Zero ERI_shell (only the portion we need)
            for ic_a in range(nbf_a):
                for ic_b in range(nbf_b):
                    for ic_c in range(nbf_c):
                        for ic_d in range(nbf_d):
                            ERI_shell[ic_a, ic_b, ic_c, ic_d] = 0.0

            
            for ic_c in range(nbf_c):
                ibf_c = bf_c_start + ic_c
                cx, cy, cz = bfs_lmn[ibf_c, 0], bfs_lmn[ibf_c, 1], bfs_lmn[ibf_c, 2]
                for ic_d in range(nbf_d):
                    ibf_d = bf_d_start + ic_d
                    ddx, ddy, ddz = bfs_lmn[ibf_d, 0], bfs_lmn[ibf_d, 1], bfs_lmn[ibf_d, 2]
                    count = 0
                    for ix in range(ddx + 1):
                        binom_x = _binom(ddx, ix)
                        pow_x = X_CD[0] ** (ddx - ix)
                        cx_s = cx + ix
                        for iy in range(ddy + 1):
                            binom_y = _binom(ddy, iy)
                            pow_y = X_CD[1] ** (ddy - iy)
                            cy_s = cy + iy
                            for iz in range(ddz + 1):
                                binom_z = _binom(ddz, iz)
                                pow_z = X_CD[2] ** (ddz - iz)
                                cz_s = cz + iz
                                coeff = binom_x * pow_x * binom_y * pow_y * binom_z * pow_z
                                ket_hrr_coeffs[ic_c, ic_d, count] = coeff
                                ket_hrr_cx[ic_c, ic_d, count] = cx_s
                                ket_hrr_cy[ic_c, ic_d, count] = cy_s
                                ket_hrr_cz[ic_c, ic_d, count] = cz_s
                                count += 1
                    ket_hrr_n[ic_c, ic_d] = count

            
            for ibra in range(n_bra_pairs):
                p = bra_p[ibra]
                K_AB_val = bra_K[ibra]
                ipa = bra_ipa[ibra]
                ipb = bra_ipb[ibra]

                Px = bra_Px[ibra]
                Py = bra_Py[ibra]
                Pz = bra_Pz[ibra]
                X_PA[0] = bra_PAx[ibra]
                X_PA[1] = bra_PAy[ibra]
                X_PA[2] = bra_PAz[ibra]

                for iket in range(n_ket_pairs):
                    q = ket_q[iket]
                    K_CD_val = ket_K[iket]
                    ipc = ket_ipc[iket]
                    ipd = ket_ipd[iket]

                    K_prod = K_AB_val * K_CD_val
                    if abs(K_prod) < 1.0e-10:
                        continue

                    Qx = ket_Qx[iket]
                    Qy = ket_Qy[iket]
                    Qz = ket_Qz[iket]

                    X_QC[0] = ket_QCx[iket]
                    X_QC[1] = ket_QCy[iket]
                    X_QC[2] = ket_QCz[iket]

                    pq = p + q
                    inv_pq = 1.0 / pq
                    eta = p * q * inv_pq

                    PQx = Px - Qx
                    PQy = Py - Qy
                    PQz = Pz - Qz
                    PQ_sq = PQx * PQx + PQy * PQy + PQz * PQz
                    T_arg = eta * PQ_sq

                    Wx = (p * Px + q * Qx) * inv_pq
                    Wy = (p * Py + q * Qy) * inv_pq
                    Wz = (p * Pz + q * Qz) * inv_pq
                    X_WP[0] = Wx - Px
                    X_WP[1] = Wy - Py
                    X_WP[2] = Wz - Pz
                    X_WQ[0] = Wx - Qx
                    X_WQ[1] = Wy - Qy
                    X_WQ[2] = Wz - Qz

                    prefactor = two_pi_52 / (p * q * np.sqrt(pq)) * K_prod

                    for m in range(dim_m):
                        F[m] = Fboys(m, T_arg)

                    
                    # for m in range(dim_m):
                    #     for i1 in range(dim_bra):
                    #         for i2 in range(dim_bra):
                    #             for i3 in range(dim_bra):
                    #                 for i4 in range(dim_ket):
                    #                     for i5 in range(dim_ket):
                    #                         for i6 in range(dim_ket):
                    #                             V_vrr[m, i1, i2, i3, i4, i5, i6] = 0.0

                    for m in range(dim_m):
                        V_vrr[m, 0, 0, 0, 0, 0, 0] = prefactor * F[m]

                    if L_bra > 0:
                        os_bra_vrr_opt(V_vrr, X_PA, X_WP, p, eta, L_bra, L_all)
                    if L_ket > 0:
                        os_ket_vrr_opt(V_vrr, X_QC, X_WQ, q, eta, L_bra, L_ket, L_all)

                    #Contract over basis functions
                    for ic_a in range(nbf_a):
                        ibf_a = bf_a_start + ic_a
                        da = bfs_coeffs[ibf_a, ipa]
                        Nik = bfs_prim_norms[ibf_a, ipa]
                        Ni = bfs_contr_prim_norms[ibf_a]
                        c_a = Ni * da * Nik

                        for ic_b in range(nbf_b):
                            ibf_b = bf_b_start + ic_b
                            db = bfs_coeffs[ibf_b, ipb]
                            Njk = bfs_prim_norms[ibf_b, ipb]
                            Nj = bfs_contr_prim_norms[ibf_b]
                            c_ab = c_a * Nj * db * Njk

                            n_bra = bra_hrr_n[ic_a, ic_b]

                            for ic_c in range(nbf_c):
                                ibf_c = bf_c_start + ic_c
                                dc = bfs_coeffs[ibf_c, ipc]
                                Nkk = bfs_prim_norms[ibf_c, ipc]
                                Nk = bfs_contr_prim_norms[ibf_c]
                                c_abc = c_ab * Nk * dc * Nkk

                                for ic_d in range(nbf_d):
                                    ibf_d = bf_d_start + ic_d
                                    dd = bfs_coeffs[ibf_d, ipd]
                                    Nlk = bfs_prim_norms[ibf_d, ipd]
                                    Nl = bfs_contr_prim_norms[ibf_d]
                                    c_abcd = c_abc * Nl * dd * Nlk

                                    n_ket_h = ket_hrr_n[ic_c, ic_d]

                                    hrr_val = 0.0
                                    for ib in range(n_bra):
                                        bc = bra_hrr_coeffs[ic_a, ic_b, ib]
                                        ax_f = bra_hrr_ax[ic_a, ic_b, ib]
                                        ay_f = bra_hrr_ay[ic_a, ic_b, ib]
                                        az_f = bra_hrr_az[ic_a, ic_b, ib]
                                        for ik in range(n_ket_h):
                                            kc = ket_hrr_coeffs[ic_c, ic_d, ik]
                                            cx_f = ket_hrr_cx[ic_c, ic_d, ik]
                                            cy_f = ket_hrr_cy[ic_c, ic_d, ik]
                                            cz_f = ket_hrr_cz[ic_c, ic_d, ik]
                                            hrr_val += bc * kc * V_vrr[0, ax_f, ay_f, az_f, cx_f, cy_f, cz_f]

                                    ERI_shell[ic_a, ic_b, ic_c, ic_d] += c_abcd * hrr_val

            #Store results with 8-fold symmetry
            for ic_a in range(nbf_a):
                ibf_a = bf_a_start + ic_a
                for ic_b in range(nbf_b):
                    ibf_b = bf_b_start + ic_b
                    for ic_c in range(nbf_c):
                        ibf_c = bf_c_start + ic_c
                        for ic_d in range(nbf_d):
                            ibf_d = bf_d_start + ic_d

                            val = ERI_shell[ic_a, ic_b, ic_c, ic_d]
                            if abs(val) < 1e-12:
                                continue

                            fourC2E[ibf_a, ibf_b, ibf_c, ibf_d] = val
                            fourC2E[ibf_b, ibf_a, ibf_c, ibf_d] = val
                            fourC2E[ibf_a, ibf_b, ibf_d, ibf_c] = val
                            fourC2E[ibf_b, ibf_a, ibf_d, ibf_c] = val
                            fourC2E[ibf_c, ibf_d, ibf_a, ibf_b] = val
                            fourC2E[ibf_c, ibf_d, ibf_b, ibf_a] = val
                            fourC2E[ibf_d, ibf_c, ibf_a, ibf_b] = val
                            fourC2E[ibf_d, ibf_c, ibf_b, ibf_a] = val

    return fourC2E


@njit(cache=True, fastmath=True, nogil=True, error_model="numpy", inline='always')
def os_bra_vrr_opt(V, X_PA, X_WP, p, eta, L_bra, m_total):
    inv_2p = 0.5 / p
    eta_over_p = eta / p

    # Build angular momentum shell by shell
    for L_e in range(L_bra):
        m_top = m_total - L_e - 1

        for ax in range(L_e + 1):
            for ay in range(L_e + 1 - ax):
                az = L_e - ax - ay

                # Increment x
                ax1 = ax + 1
                for m in range(m_top + 1):
                    v = X_PA[0] * V[m, ax, ay, az, 0, 0, 0] + X_WP[0] * V[m+1, ax, ay, az, 0, 0, 0]
                    if ax > 0:
                        v += ax * inv_2p * (V[m, ax-1, ay, az, 0, 0, 0] - eta_over_p * V[m+1, ax-1, ay, az, 0, 0, 0])
                    V[m, ax1, ay, az, 0, 0, 0] = v

                # Increment y
                ay1 = ay + 1
                for m in range(m_top + 1):
                    v = X_PA[1] * V[m, ax, ay, az, 0, 0, 0] + X_WP[1] * V[m+1, ax, ay, az, 0, 0, 0]
                    if ay > 0:
                        v += ay * inv_2p * (V[m, ax, ay-1, az, 0, 0, 0] - eta_over_p * V[m+1, ax, ay-1, az, 0, 0, 0])
                    V[m, ax, ay1, az, 0, 0, 0] = v

                # Increment z
                az1 = az + 1
                for m in range(m_top + 1):
                    v = X_PA[2] * V[m, ax, ay, az, 0, 0, 0] + X_WP[2] * V[m+1, ax, ay, az, 0, 0, 0]
                    if az > 0:
                        v += az * inv_2p * (V[m, ax, ay, az-1, 0, 0, 0] - eta_over_p * V[m+1, ax, ay, az-1, 0, 0, 0])
                    V[m, ax, ay, az1, 0, 0, 0] = v


@njit(cache=True, fastmath=True, nogil=True, error_model="numpy", inline='always')
def os_ket_vrr_opt(V, X_QC, X_WQ, q, eta, L_bra, L_ket, m_total):
    inv_2q = 0.5 / q
    eta_over_q = eta / q
    inv_2pq = eta / (2.0 * q)  
    p = eta * q / (q - eta)
    inv_2pq = 0.5 / (p + q)

    for L_f in range(L_ket):
        for L_e in range(L_bra + 1):
            m_top = m_total - L_e - L_f - 1

            for ex in range(L_e + 1):
                for ey in range(L_e + 1 - ex):
                    ez = L_e - ex - ey

                    for fx in range(L_f + 1):
                        for fy in range(L_f + 1 - fx):
                            fz = L_f - fx - fy

                            # Increment fx
                            fx1 = fx + 1
                            for m in range(m_top + 1):
                                v = X_QC[0] * V[m, ex, ey, ez, fx, fy, fz] + X_WQ[0] * V[m+1, ex, ey, ez, fx, fy, fz]
                                if fx > 0:
                                    v += fx * inv_2q * (V[m, ex, ey, ez, fx-1, fy, fz] - eta_over_q * V[m+1, ex, ey, ez, fx-1, fy, fz])
                                if ex > 0:
                                    v += ex * inv_2pq * V[m+1, ex-1, ey, ez, fx, fy, fz]
                                V[m, ex, ey, ez, fx1, fy, fz] = v

                            # Increment fy
                            fy1 = fy + 1
                            for m in range(m_top + 1):
                                v = X_QC[1] * V[m, ex, ey, ez, fx, fy, fz] + X_WQ[1] * V[m+1, ex, ey, ez, fx, fy, fz]
                                if fy > 0:
                                    v += fy * inv_2q * (V[m, ex, ey, ez, fx, fy-1, fz] - eta_over_q * V[m+1, ex, ey, ez, fx, fy-1, fz])
                                if ey > 0:
                                    v += ey * inv_2pq * V[m+1, ex, ey-1, ez, fx, fy, fz]
                                V[m, ex, ey, ez, fx, fy1, fz] = v

                            # Increment fz
                            fz1 = fz + 1
                            for m in range(m_top + 1):
                                v = X_QC[2] * V[m, ex, ey, ez, fx, fy, fz] + X_WQ[2] * V[m+1, ex, ey, ez, fx, fy, fz]
                                if fz > 0:
                                    v += fz * inv_2q * (V[m, ex, ey, ez, fx, fy, fz-1] - eta_over_q * V[m+1, ex, ey, ez, fx, fy, fz-1])
                                if ez > 0:
                                    v += ez * inv_2pq * V[m+1, ex, ey, ez-1, fx, fy, fz]
                                V[m, ex, ey, ez, fx, fy, fz1] = v


@njit(cache=True, fastmath=True, nogil=True, error_model="numpy", inline='always')
def _binom(n, k):
    if k < 0 or k > n:
        return 0
    if k > n - k:
        k = n - k
    res = 1
    for i in range(1, k + 1):
        res = res * (n - i + 1) // i
    return res


@njit(cache=True, fastmath=True, nogil=True, error_model="numpy", inline='always')
def os_hrr_element(V_vrr, ax, ay, az, bx, by, bz,
                   cx, cy, cz, dx, dy, dz, X_AB, X_CD):
    result = 0.0

    for ix in range(dx + 1):
        binom_x_ket = _binom(dx, ix)
        pow_x_ket = X_CD[0] ** (dx - ix)
        cx_shift = cx + ix

        for iy in range(dy + 1):
            binom_y_ket = _binom(dy, iy)
            pow_y_ket = X_CD[1] ** (dy - iy)
            cy_shift = cy + iy

            for iz in range(dz + 1):
                binom_z_ket = _binom(dz, iz)
                pow_z_ket = X_CD[2] ** (dz - iz)
                cz_shift = cz + iz

                ket_coeff = (binom_x_ket * pow_x_ket *
                             binom_y_ket * pow_y_ket *
                             binom_z_ket * pow_z_ket)

                for px in range(bx + 1):
                    binom_x_bra = _binom(bx, px)
                    pow_x_bra = X_AB[0] ** (bx - px)
                    ax_final = ax + px

                    for py in range(by + 1):
                        binom_y_bra = _binom(by, py)
                        pow_y_bra = X_AB[1] ** (by - py)
                        ay_final = ay + py

                        for pz in range(bz + 1):
                            binom_z_bra = _binom(bz, pz)
                            pow_z_bra = X_AB[2] ** (bz - pz)
                            az_final = az + pz

                            bra_coeff = (binom_x_bra * pow_x_bra *
                                        binom_y_bra * pow_y_bra *
                                        binom_z_bra * pow_z_bra)

                            result += ket_coeff * bra_coeff * V_vrr[0, ax_final, ay_final, az_final,
                                                                     cx_shift, cy_shift, cz_shift]

    return result


