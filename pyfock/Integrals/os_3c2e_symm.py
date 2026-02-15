import numpy as np
from numba import njit, prange
from .integral_helpers import Fboys
from .integral_helpers import comb


def os_3c2e_symm(basis, auxbasis):
    """
    Compute three-center two-electron (3c2e) integrals (A B | C) using the 
    Obara-Saika scheme with exploitation of permutational symmetry (AB swap).

    Parameters
    ----------
    basis : object
        Primary basis set object
    auxbasis : object
        Auxiliary basis set object

    Returns
    -------
    ints3c2e : ndarray
        The computed 3-center 2-electron integrals, shape (Nbf, Nbf, Naux)
    """
    
    nbf = basis.bfs_nao
    nshells = len(basis.shells)
    
    shell_L = np.array([basis.bfs_lm[i] for i in basis.shell_bfs_offset], dtype=np.int32)
    shell_centers = np.array([basis.bfs_coords[i] for i in basis.shell_bfs_offset], dtype=np.float64)
    shell_bfs_offset = np.array(basis.shell_bfs_offset, dtype=np.int32)
    bfs_nbfshell = np.array(basis.bfs_nbfshell, dtype=np.int32)
    
    bfs_coords = np.array(basis.bfs_coords, dtype=np.float64)
    bfs_contr_prim_norms = np.array(basis.bfs_contr_prim_norms, dtype=np.float64)
    bfs_lmn = np.array(basis.bfs_lmn, dtype=np.int32)
    bfs_nprim = np.array(basis.bfs_nprim, dtype=np.int32)
    bfs_shell_index = np.array(basis.bfs_shell_index, dtype=np.int32)
    
    maxnprim = max(basis.bfs_nprim)
    bfs_coeffs = np.zeros((nbf, maxnprim), dtype=np.float64)
    bfs_expnts = np.zeros((nbf, maxnprim), dtype=np.float64)
    bfs_prim_norms = np.zeros((nbf, maxnprim), dtype=np.float64)
    
    for i in range(nbf):
        for j in range(basis.bfs_nprim[i]):
            bfs_coeffs[i, j] = basis.bfs_coeffs[i][j]
            bfs_expnts[i, j] = basis.bfs_expnts[i][j]
            bfs_prim_norms[i, j] = basis.bfs_prim_norms[i][j]
    
    # Auxiliary basis data
    naux = auxbasis.bfs_nao
    nshells_aux = len(auxbasis.shells)
    
    aux_shell_L = np.array([auxbasis.bfs_lm[i] for i in auxbasis.shell_bfs_offset], dtype=np.int32)
    aux_shell_centers = np.array([auxbasis.bfs_coords[i] for i in auxbasis.shell_bfs_offset], dtype=np.float64)
    aux_shell_bfs_offset = np.array(auxbasis.shell_bfs_offset, dtype=np.int32)
    aux_bfs_nbfshell = np.array(auxbasis.bfs_nbfshell, dtype=np.int32)
    
    aux_bfs_contr_prim_norms = np.array(auxbasis.bfs_contr_prim_norms, dtype=np.float64)
    aux_bfs_lmn = np.array(auxbasis.bfs_lmn, dtype=np.int32)
    aux_bfs_nprim = np.array(auxbasis.bfs_nprim, dtype=np.int32)
    
    aux_maxnprim = max(auxbasis.bfs_nprim)
    aux_bfs_coeffs = np.zeros((naux, aux_maxnprim), dtype=np.float64)
    aux_bfs_expnts = np.zeros((naux, aux_maxnprim), dtype=np.float64)
    aux_bfs_prim_norms = np.zeros((naux, aux_maxnprim), dtype=np.float64)
    
    for i in range(naux):
        for j in range(auxbasis.bfs_nprim[i]):
            aux_bfs_coeffs[i, j] = auxbasis.bfs_coeffs[i][j]
            aux_bfs_expnts[i, j] = auxbasis.bfs_expnts[i][j]
            aux_bfs_prim_norms[i, j] = auxbasis.bfs_prim_norms[i][j]
    
    ints3c2e = os_3c2e_symm_internal(
        nbf, nshells, shell_L, shell_centers, shell_bfs_offset, bfs_nbfshell,
        bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_shell_index,
        bfs_coeffs, bfs_expnts, bfs_prim_norms,
        naux, nshells_aux, aux_shell_L, aux_shell_centers, aux_shell_bfs_offset,
        aux_bfs_nbfshell, aux_bfs_contr_prim_norms, aux_bfs_lmn, aux_bfs_nprim,
        aux_bfs_coeffs, aux_bfs_expnts, aux_bfs_prim_norms
    )
    
    return ints3c2e


@njit(cache=True, fastmath=True, nogil=True, error_model="numpy", inline='always')
def _flat_idx_3c(ax, ay, az, cx, cy, cz, d1, d2):
    return ((((ax * d1 + ay) * d1 + az) * d2 + cx) * d2 + cy) * d2 + cz


@njit(cache=True, fastmath=True, nogil=True, error_model="numpy", inline='always')
def os_bra_vrr_flat_3c(V, X_PA, X_WP, p, eta, L_bra, m_total, d1, d2):
    inv_2p = 0.5 / p
    eta_over_p = eta / p

    for L_e in range(L_bra):
        m_top = m_total - L_e - 1

        for ax in range(L_e + 1):
            for ay in range(L_e + 1 - ax):
                az = L_e - ax - ay

                idx_src = _flat_idx_3c(ax, ay, az, 0, 0, 0, d1, d2)

                # Increment x
                ax1 = ax + 1
                idx_dst = _flat_idx_3c(ax1, ay, az, 0, 0, 0, d1, d2)
                if ax > 0:
                    idx_am1 = _flat_idx_3c(ax - 1, ay, az, 0, 0, 0, d1, d2)
                    for m in range(m_top + 1):
                        v = X_PA[0] * V[m, idx_src] + X_WP[0] * V[m + 1, idx_src]
                        v += ax * inv_2p * (V[m, idx_am1] - eta_over_p * V[m + 1, idx_am1])
                        V[m, idx_dst] = v
                else:
                    for m in range(m_top + 1):
                        V[m, idx_dst] = X_PA[0] * V[m, idx_src] + X_WP[0] * V[m + 1, idx_src]

                # Increment y
                ay1 = ay + 1
                idx_dst = _flat_idx_3c(ax, ay1, az, 0, 0, 0, d1, d2)
                if ay > 0:
                    idx_am1 = _flat_idx_3c(ax, ay - 1, az, 0, 0, 0, d1, d2)
                    for m in range(m_top + 1):
                        v = X_PA[1] * V[m, idx_src] + X_WP[1] * V[m + 1, idx_src]
                        v += ay * inv_2p * (V[m, idx_am1] - eta_over_p * V[m + 1, idx_am1])
                        V[m, idx_dst] = v
                else:
                    for m in range(m_top + 1):
                        V[m, idx_dst] = X_PA[1] * V[m, idx_src] + X_WP[1] * V[m + 1, idx_src]

                # Increment z
                az1 = az + 1
                idx_dst = _flat_idx_3c(ax, ay, az1, 0, 0, 0, d1, d2)
                if az > 0:
                    idx_am1 = _flat_idx_3c(ax, ay, az - 1, 0, 0, 0, d1, d2)
                    for m in range(m_top + 1):
                        v = X_PA[2] * V[m, idx_src] + X_WP[2] * V[m + 1, idx_src]
                        v += az * inv_2p * (V[m, idx_am1] - eta_over_p * V[m + 1, idx_am1])
                        V[m, idx_dst] = v
                else:
                    for m in range(m_top + 1):
                        V[m, idx_dst] = X_PA[2] * V[m, idx_src] + X_WP[2] * V[m + 1, idx_src]


@njit(cache=True, fastmath=True, nogil=True, error_model="numpy", inline='always')
def os_ket_vrr_flat_3c(V, X_QC, X_WQ, q, eta, L_bra, L_ket, m_total, d1, d2):
    inv_2q = 0.5 / q
    eta_over_q = eta / q
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

                            idx_src = _flat_idx_3c(ex, ey, ez, fx, fy, fz, d1, d2)

                            # Increment fx
                            fx1 = fx + 1
                            idx_dst = _flat_idx_3c(ex, ey, ez, fx1, fy, fz, d1, d2)
                            has_fx = fx > 0
                            has_ex = ex > 0
                            if has_fx:
                                idx_fm1 = _flat_idx_3c(ex, ey, ez, fx - 1, fy, fz, d1, d2)
                            if has_ex:
                                idx_em1 = _flat_idx_3c(ex - 1, ey, ez, fx, fy, fz, d1, d2)

                            for m in range(m_top + 1):
                                v = X_QC[0] * V[m, idx_src] + X_WQ[0] * V[m + 1, idx_src]
                                if has_fx:
                                    v += fx * inv_2q * (V[m, idx_fm1] - eta_over_q * V[m + 1, idx_fm1])
                                if has_ex:
                                    v += ex * inv_2pq * V[m + 1, idx_em1]
                                V[m, idx_dst] = v

                            # Increment fy
                            fy1 = fy + 1
                            idx_dst = _flat_idx_3c(ex, ey, ez, fx, fy1, fz, d1, d2)
                            has_fy = fy > 0
                            has_ey = ey > 0
                            if has_fy:
                                idx_fm1 = _flat_idx_3c(ex, ey, ez, fx, fy - 1, fz, d1, d2)
                            if has_ey:
                                idx_em1 = _flat_idx_3c(ex, ey - 1, ez, fx, fy, fz, d1, d2)

                            for m in range(m_top + 1):
                                v = X_QC[1] * V[m, idx_src] + X_WQ[1] * V[m + 1, idx_src]
                                if has_fy:
                                    v += fy * inv_2q * (V[m, idx_fm1] - eta_over_q * V[m + 1, idx_fm1])
                                if has_ey:
                                    v += ey * inv_2pq * V[m + 1, idx_em1]
                                V[m, idx_dst] = v

                            # Increment fz
                            fz1 = fz + 1
                            idx_dst = _flat_idx_3c(ex, ey, ez, fx, fy, fz1, d1, d2)
                            has_fz = fz > 0
                            has_ez = ez > 0
                            if has_fz:
                                idx_fm1 = _flat_idx_3c(ex, ey, ez, fx, fy, fz - 1, d1, d2)
                            if has_ez:
                                idx_em1 = _flat_idx_3c(ex, ey, ez - 1, fx, fy, fz, d1, d2)

                            for m in range(m_top + 1):
                                v = X_QC[2] * V[m, idx_src] + X_WQ[2] * V[m + 1, idx_src]
                                if has_fz:
                                    v += fz * inv_2q * (V[m, idx_fm1] - eta_over_q * V[m + 1, idx_fm1])
                                if has_ez:
                                    v += ez * inv_2pq * V[m + 1, idx_em1]
                                V[m, idx_dst] = v


@njit(parallel=True, cache=True, fastmath=True, nogil=True, error_model="numpy")
def os_3c2e_symm_internal(
    nbf, nshells, shell_L, shell_centers, shell_bfs_offset, bfs_nbfshell,
    bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_shell_index,
    bfs_coeffs, bfs_expnts, bfs_prim_norms,
    naux, nshells_aux, aux_shell_L, aux_shell_centers, aux_shell_bfs_offset,
    aux_bfs_nbfshell, aux_bfs_contr_prim_norms, aux_bfs_lmn, aux_bfs_nprim,
    aux_bfs_coeffs, aux_bfs_expnts, aux_bfs_prim_norms
):

    threeC2E = np.zeros((nbf, nbf, naux), dtype=np.float64)

    pi = np.pi
    two_pi_52 = 2.0 * pi ** 2.5

    # Build unique bra shell pairs (a >= b) for primary basis
    n_ab = nshells * (nshells + 1) // 2

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

    # Compute max dimensions
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

    max_L_aux = 0
    max_nprim_aux = 0
    max_nbf_shell_aux = 0
    for i in range(nshells_aux):
        if aux_shell_L[i] > max_L_aux:
            max_L_aux = aux_shell_L[i]
        ibf0 = aux_shell_bfs_offset[i]
        if aux_bfs_nprim[ibf0] > max_nprim_aux:
            max_nprim_aux = aux_bfs_nprim[ibf0]
        if aux_bfs_nbfshell[i] > max_nbf_shell_aux:
            max_nbf_shell_aux = aux_bfs_nbfshell[i]

    # For 3c2e: bra has L_a + L_b, ket has L_c (single aux function, no D)
    max_L_bra = 2 * max_L
    max_L_ket = max_L_aux
    max_L_all = max_L_bra + max_L_ket

    max_dim_bra = max_L_bra + 1
    max_dim_ket = max_L_ket + 1
    max_dim_m = max_L_all + 1
    max_prim_pairs = max_nprim * max_nprim
    max_hrr = (max_L + 1) ** 3

    max_flat = max_dim_bra ** 3 * max_dim_ket ** 3

    # Total shell triplets: n_ab * nshells_aux
    n_tasks = n_ab * nshells_aux

    for task_idx in prange(n_tasks):
        ab_idx = task_idx // nshells_aux
        ish_c_idx = task_idx % nshells_aux

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

        # Auxiliary shell C
        ish_c = ish_c_idx
        L_c = aux_shell_L[ish_c]
        L_ket = L_c  # No D function, ket angular momentum is just L_c
        bf_c_start = aux_shell_bfs_offset[ish_c]
        nbf_c = aux_bfs_nbfshell[ish_c]
        center_c = aux_shell_centers[ish_c]
        ibf_c0 = bf_c_start
        nprimk = aux_bfs_nprim[ibf_c0]

        L_all = L_bra + L_ket
        dim_bra = L_bra + 1
        dim_ket = L_ket + 1
        dim_m = L_all + 1

        flat_size = dim_bra ** 3 * dim_ket ** 3

        # Allocate thread-local arrays
        bra_p = np.empty(max_prim_pairs, dtype=np.float64)
        bra_K = np.empty(max_prim_pairs, dtype=np.float64)
        bra_Px = np.empty(max_prim_pairs, dtype=np.float64)
        bra_Py = np.empty(max_prim_pairs, dtype=np.float64)
        bra_Pz = np.empty(max_prim_pairs, dtype=np.float64)
        bra_PAx = np.empty(max_prim_pairs, dtype=np.float64)
        bra_PAy = np.empty(max_prim_pairs, dtype=np.float64)
        bra_PAz = np.empty(max_prim_pairs, dtype=np.float64)
        bra_ipa = np.empty(max_prim_pairs, dtype=np.int64)
        bra_ipb = np.empty(max_prim_pairs, dtype=np.int64)

        V_vrr = np.zeros((max_dim_m, max_flat), dtype=np.float64)

        ERI_shell = np.zeros((max_nbf_shell, max_nbf_shell, max_nbf_shell_aux), dtype=np.float64)

        bra_hrr_coeffs = np.empty((max_nbf_shell, max_nbf_shell, max_hrr), dtype=np.float64)
        bra_hrr_ax = np.empty((max_nbf_shell, max_nbf_shell, max_hrr), dtype=np.int64)
        bra_hrr_ay = np.empty((max_nbf_shell, max_nbf_shell, max_hrr), dtype=np.int64)
        bra_hrr_az = np.empty((max_nbf_shell, max_nbf_shell, max_hrr), dtype=np.int64)
        bra_hrr_n = np.empty((max_nbf_shell, max_nbf_shell), dtype=np.int64)

        X_PA = np.empty(3, dtype=np.float64)
        X_QC = np.empty(3, dtype=np.float64)
        X_WP = np.empty(3, dtype=np.float64)
        X_WQ = np.empty(3, dtype=np.float64)
        F = np.empty(max_dim_m, dtype=np.float64)

        # Build bra primitive pairs
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

        # Build bra HRR expansion coefficients
        for ic_a in range(nbf_a):
            ibf_a = bf_a_start + ic_a
            ax, ay, az = bfs_lmn[ibf_a, 0], bfs_lmn[ibf_a, 1], bfs_lmn[ibf_a, 2]
            for ic_b in range(nbf_b):
                ibf_b = bf_b_start + ic_b
                bx, by, bz = bfs_lmn[ibf_b, 0], bfs_lmn[ibf_b, 1], bfs_lmn[ibf_b, 2]
                count = 0
                for px in range(bx + 1):
                    binom_x = comb(bx, px)
                    pow_x = X_AB[0] ** (bx - px)
                    ax_f = ax + px
                    for py in range(by + 1):
                        binom_y = comb(by, py)
                        pow_y = X_AB[1] ** (by - py)
                        ay_f = ay + py
                        for pz in range(bz + 1):
                            binom_z = comb(bz, pz)
                            pow_z = X_AB[2] ** (bz - pz)
                            az_f = az + pz
                            coeff = binom_x * pow_x * binom_y * pow_y * binom_z * pow_z
                            bra_hrr_coeffs[ic_a, ic_b, count] = coeff
                            bra_hrr_ax[ic_a, ic_b, count] = ax_f
                            bra_hrr_ay[ic_a, ic_b, count] = ay_f
                            bra_hrr_az[ic_a, ic_b, count] = az_f
                            count += 1
                bra_hrr_n[ic_a, ic_b] = count

        # Zero ERI shell block
        for ic_a in range(nbf_a):
            for ic_b in range(nbf_b):
                for ic_c in range(nbf_c):
                    ERI_shell[ic_a, ic_b, ic_c] = 0.0

        # Loop over bra primitive pairs and ket primitives
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

            for ipc in range(nprimk):
                gamma = aux_bfs_expnts[ibf_c0, ipc]

                # For 3c2e: ket is a single function at center C with exponent gamma
                # q = gamma (delta = 0 limit doesn't work; instead, for (ab|c) the 
                # "ket" Gaussian is just a single primitive at C)
                q = gamma
                # No CD separation, so K_CD = 1, Q = C
                K_CD_val = 1.0

                K_prod = K_AB_val * K_CD_val
                if abs(K_prod) < 1.0e-10:
                    continue

                Qx = center_c[0]
                Qy = center_c[1]
                Qz = center_c[2]

                # X_QC = Q - C = 0 (since Q = C for single aux function)
                X_QC[0] = 0.0
                X_QC[1] = 0.0
                X_QC[2] = 0.0

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

                # Zero V_vrr
                # for m in range(dim_m):
                #     for fi in range(flat_size):
                #         V_vrr[m, fi] = 0.0

                idx_000000 = _flat_idx_3c(0, 0, 0, 0, 0, 0, dim_bra, dim_ket)
                for m in range(dim_m):
                    V_vrr[m, idx_000000] = prefactor * F[m]

                if L_bra > 0:
                    os_bra_vrr_flat_3c(V_vrr, X_PA, X_WP, p, eta, L_bra, L_all, dim_bra, dim_ket)
                if L_ket > 0:
                    os_ket_vrr_flat_3c(V_vrr, X_QC, X_WQ, q, eta, L_bra, L_ket, L_all, dim_bra, dim_ket)

                # Contract over basis functions
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

                        n_bra_h = bra_hrr_n[ic_a, ic_b]

                        for ic_c in range(nbf_c):
                            ibf_c = bf_c_start + ic_c
                            dc = aux_bfs_coeffs[ibf_c, ipc]
                            Nkk = aux_bfs_prim_norms[ibf_c, ipc]
                            Nk = aux_bfs_contr_prim_norms[ibf_c]
                            c_abc = c_ab * Nk * dc * Nkk

                            # For 3c2e, ket has no HRR (no D function)
                            # Just read V directly at the aux angular momentum
                            cx = aux_bfs_lmn[ibf_c, 0]
                            cy = aux_bfs_lmn[ibf_c, 1]
                            cz = aux_bfs_lmn[ibf_c, 2]

                            hrr_val = 0.0
                            for ib in range(n_bra_h):
                                bc = bra_hrr_coeffs[ic_a, ic_b, ib]
                                ax_f = bra_hrr_ax[ic_a, ic_b, ib]
                                ay_f = bra_hrr_ay[ic_a, ic_b, ib]
                                az_f = bra_hrr_az[ic_a, ic_b, ib]
                                fidx = _flat_idx_3c(ax_f, ay_f, az_f, cx, cy, cz, dim_bra, dim_ket)
                                hrr_val += bc * V_vrr[0, fidx]

                            ERI_shell[ic_a, ic_b, ic_c] += c_abc * hrr_val

        # Store results with AB symmetry (swap a <-> b)
        for ic_a in prange(nbf_a):
            ibf_a = bf_a_start + ic_a
            for ic_b in range(nbf_b):
                ibf_b = bf_b_start + ic_b
                for ic_c in range(nbf_c):
                    ibf_c = bf_c_start + ic_c

                    val = ERI_shell[ic_a, ic_b, ic_c]
                    if abs(val) < 1e-15:
                        continue

                    threeC2E[ibf_a, ibf_b, ibf_c] = val
                    threeC2E[ibf_b, ibf_a, ibf_c] = val

    return threeC2E