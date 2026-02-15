import numpy as np
from numba import njit, prange, get_num_threads, get_thread_id
from .integral_helpers import Fboys
from .integral_helpers import comb
from .schwarz_helpers import eri_4c2e_diag


def os_coulomb_matrix(basis, density_matrix, schwarz_shell_pair=None,
                      threshold_schwarz=1e-9, threshold_density=1e-10):
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

    maxnprim = max(basis.bfs_nprim)
    bfs_coeffs = np.zeros((nbf, maxnprim), dtype=np.float64)
    bfs_expnts = np.zeros((nbf, maxnprim), dtype=np.float64)
    bfs_prim_norms = np.zeros((nbf, maxnprim), dtype=np.float64)

    for i in range(nbf):
        for j in range(basis.bfs_nprim[i]):
            bfs_coeffs[i, j] = basis.bfs_coeffs[i][j]
            bfs_expnts[i, j] = basis.bfs_expnts[i][j]
            bfs_prim_norms[i, j] = basis.bfs_prim_norms[i][j]

    if schwarz_shell_pair is None:
        ints4c2e_diag = eri_4c2e_diag(basis)
        sqrt_ints4c2e_diag = np.sqrt(np.abs(ints4c2e_diag))
        schwarz_shell_pair = np.zeros((nshells, nshells), dtype=np.float64)
        for ish in range(nshells):
            bf_i_start = shell_bfs_offset[ish]
            nbf_i = bfs_nbfshell[ish]
            for jsh in range(ish + 1):
                bf_j_start = shell_bfs_offset[jsh]
                nbf_j = bfs_nbfshell[jsh]
                max_val = 0.0
                for ia in range(nbf_i):
                    for jb in range(nbf_j):
                        val = sqrt_ints4c2e_diag[bf_i_start + ia, bf_j_start + jb]
                        if val > max_val:
                            max_val = val
                schwarz_shell_pair[ish, jsh] = max_val
                schwarz_shell_pair[jsh, ish] = max_val

    density_matrix = np.ascontiguousarray(density_matrix, dtype=np.float64)
    nthreads = get_num_threads()

    # Precompute combined contraction coefficients per shell pair per primitive pair
    # coeff_ab[ibf_a, ibf_b, ipa, ipb] = Ni*da*Nik * Nj*db*Njk
    max_nbf_shell = 0
    max_nprim = 0
    max_L = 0
    for i in range(nshells):
        if bfs_nbfshell[i] > max_nbf_shell:
            max_nbf_shell = bfs_nbfshell[i]
        if shell_L[i] > max_L:
            max_L = shell_L[i]
        ibf0 = shell_bfs_offset[i]
        if bfs_nprim[ibf0] > max_nprim:
            max_nprim = bfs_nprim[ibf0]

    # Precompute the angular momentum triples for each shell
    # For shell with L, there are (L+1)(L+2)/2 basis functions
    # Store lmn indices per shell compactly
    shell_lmn = np.zeros((nshells, max_nbf_shell, 3), dtype=np.int32)
    for ish in range(nshells):
        bf0 = shell_bfs_offset[ish]
        for ibf in range(bfs_nbfshell[ish]):
            shell_lmn[ish, ibf, 0] = bfs_lmn[bf0 + ibf, 0]
            shell_lmn[ish, ibf, 1] = bfs_lmn[bf0 + ibf, 1]
            shell_lmn[ish, ibf, 2] = bfs_lmn[bf0 + ibf, 2]

    # Precompute contraction coefficients: c[bf, prim] = contr_norm * coeff * prim_norm
    cont_coeffs = np.zeros((nbf, maxnprim), dtype=np.float64)
    for i in range(nbf):
        Ni = bfs_contr_prim_norms[i]
        for j in range(bfs_nprim[i]):
            cont_coeffs[i, j] = Ni * bfs_coeffs[i, j] * bfs_prim_norms[i, j]

    J_matrix = _os_coulomb_internal(
        nbf, nshells, nthreads, max_L, max_nprim, max_nbf_shell,
        shell_L, shell_centers, shell_bfs_offset, bfs_nbfshell,
        shell_lmn, bfs_nprim, bfs_expnts, cont_coeffs,
        density_matrix, schwarz_shell_pair,
        threshold_schwarz, threshold_density
    )

    return J_matrix


@njit(cache=True, fastmath=True, nogil=True, error_model="numpy", inline='always')
def _compute_vrr_ssss(V_flat, dim_bra, dim_ket, dim_m,
                      X_PA, X_WP, X_QC, X_WQ,
                      p, q, eta, prefactor, F,
                      L_bra, L_ket, L_all):
    # V_flat layout: [m, ax, ay, az, cx, cy, cz]
    # Dimensions: dim_m x dim_bra x dim_bra x dim_bra x dim_ket x dim_ket x dim_ket
    # Strides
    s6 = 1
    s5 = dim_ket
    s4 = dim_ket * dim_ket
    s3 = dim_ket * dim_ket * dim_ket
    s2 = dim_bra * s3
    s1 = dim_bra * s2
    s0 = dim_bra * s1

    # Initialize [m,0,0,0,0,0,0]
    for m in range(dim_m):
        V_flat[m * s0] = prefactor * F[m]

    inv_2p = 0.5 / p
    eta_over_p = eta / p
    inv_2q = 0.5 / q
    eta_over_q = eta / q
    inv_2pq = 0.5 / (p + q)

    # Bra VRR - build up angular momentum on bra center A
    # Only need ket = (0,0,0) during bra VRR
    for L_e in range(L_bra):
        m_top = L_all - L_e - 1
        for ax in range(L_e + 1):
            for ay in range(L_e + 1 - ax):
                az = L_e - ax - ay
                base = ax * s1 + ay * s2 + az * s3

                # Increment x
                ax1_base = (ax + 1) * s1 + ay * s2 + az * s3
                if ax > 0:
                    axm1_base = (ax - 1) * s1 + ay * s2 + az * s3
                    fac = ax * inv_2p
                    for m in range(m_top + 1):
                        V_flat[m * s0 + ax1_base] = (
                            X_PA[0] * V_flat[m * s0 + base] +
                            X_WP[0] * V_flat[(m + 1) * s0 + base] +
                            fac * (V_flat[m * s0 + axm1_base] -
                                   eta_over_p * V_flat[(m + 1) * s0 + axm1_base]))
                else:
                    for m in range(m_top + 1):
                        V_flat[m * s0 + ax1_base] = (
                            X_PA[0] * V_flat[m * s0 + base] +
                            X_WP[0] * V_flat[(m + 1) * s0 + base])

                # Increment y
                ay1_base = ax * s1 + (ay + 1) * s2 + az * s3
                if ay > 0:
                    aym1_base = ax * s1 + (ay - 1) * s2 + az * s3
                    fac = ay * inv_2p
                    for m in range(m_top + 1):
                        V_flat[m * s0 + ay1_base] = (
                            X_PA[1] * V_flat[m * s0 + base] +
                            X_WP[1] * V_flat[(m + 1) * s0 + base] +
                            fac * (V_flat[m * s0 + aym1_base] -
                                   eta_over_p * V_flat[(m + 1) * s0 + aym1_base]))
                else:
                    for m in range(m_top + 1):
                        V_flat[m * s0 + ay1_base] = (
                            X_PA[1] * V_flat[m * s0 + base] +
                            X_WP[1] * V_flat[(m + 1) * s0 + base])

                # Increment z
                az1_base = ax * s1 + ay * s2 + (az + 1) * s3
                if az > 0:
                    azm1_base = ax * s1 + ay * s2 + (az - 1) * s3
                    fac = az * inv_2p
                    for m in range(m_top + 1):
                        V_flat[m * s0 + az1_base] = (
                            X_PA[2] * V_flat[m * s0 + base] +
                            X_WP[2] * V_flat[(m + 1) * s0 + base] +
                            fac * (V_flat[m * s0 + azm1_base] -
                                   eta_over_p * V_flat[(m + 1) * s0 + azm1_base]))
                else:
                    for m in range(m_top + 1):
                        V_flat[m * s0 + az1_base] = (
                            X_PA[2] * V_flat[m * s0 + base] +
                            X_WP[2] * V_flat[(m + 1) * s0 + base])

    # Ket VRR
    if L_ket > 0:
        for L_f in range(L_ket):
            for L_e in range(L_bra + 1):
                m_top = L_all - L_e - L_f - 1
                for ex in range(L_e + 1):
                    for ey in range(L_e + 1 - ex):
                        ez = L_e - ex - ey
                        for fx in range(L_f + 1):
                            for fy in range(L_f + 1 - fx):
                                fz = L_f - fx - fy
                                base = ex * s1 + ey * s2 + ez * s3 + fx * s4 + fy * s5 + fz * s6

                                # Increment cx
                                fx1_base = ex * s1 + ey * s2 + ez * s3 + (fx + 1) * s4 + fy * s5 + fz * s6
                                has_fx = fx > 0
                                has_ex = ex > 0
                                if has_fx:
                                    fxm1_base = ex * s1 + ey * s2 + ez * s3 + (fx - 1) * s4 + fy * s5 + fz * s6
                                    fac_fx = fx * inv_2q
                                if has_ex:
                                    exm1_base = (ex - 1) * s1 + ey * s2 + ez * s3 + fx * s4 + fy * s5 + fz * s6
                                    fac_ex = ex * inv_2pq

                                for m in range(m_top + 1):
                                    v = X_QC[0] * V_flat[m * s0 + base] + X_WQ[0] * V_flat[(m + 1) * s0 + base]
                                    if has_fx:
                                        v += fac_fx * (V_flat[m * s0 + fxm1_base] - eta_over_q * V_flat[(m + 1) * s0 + fxm1_base])
                                    if has_ex:
                                        v += fac_ex * V_flat[(m + 1) * s0 + exm1_base]
                                    V_flat[m * s0 + fx1_base] = v

                                # Increment cy
                                fy1_base = ex * s1 + ey * s2 + ez * s3 + fx * s4 + (fy + 1) * s5 + fz * s6
                                has_fy = fy > 0
                                has_ey = ey > 0
                                if has_fy:
                                    fym1_base = ex * s1 + ey * s2 + ez * s3 + fx * s4 + (fy - 1) * s5 + fz * s6
                                    fac_fy = fy * inv_2q
                                if has_ey:
                                    eym1_base = ex * s1 + (ey - 1) * s2 + ez * s3 + fx * s4 + fy * s5 + fz * s6
                                    fac_ey = ey * inv_2pq

                                for m in range(m_top + 1):
                                    v = X_QC[1] * V_flat[m * s0 + base] + X_WQ[1] * V_flat[(m + 1) * s0 + base]
                                    if has_fy:
                                        v += fac_fy * (V_flat[m * s0 + fym1_base] - eta_over_q * V_flat[(m + 1) * s0 + fym1_base])
                                    if has_ey:
                                        v += fac_ey * V_flat[(m + 1) * s0 + eym1_base]
                                    V_flat[m * s0 + fy1_base] = v

                                # Increment cz
                                fz1_base = ex * s1 + ey * s2 + ez * s3 + fx * s4 + fy * s5 + (fz + 1) * s6
                                has_fz = fz > 0
                                has_ez = ez > 0
                                if has_fz:
                                    fzm1_base = ex * s1 + ey * s2 + ez * s3 + fx * s4 + fy * s5 + (fz - 1) * s6
                                    fac_fz = fz * inv_2q
                                if has_ez:
                                    ezm1_base = ex * s1 + ey * s2 + (ez - 1) * s3 + fx * s4 + fy * s5 + fz * s6
                                    fac_ez = ez * inv_2pq

                                for m in range(m_top + 1):
                                    v = X_QC[2] * V_flat[m * s0 + base] + X_WQ[2] * V_flat[(m + 1) * s0 + base]
                                    if has_fz:
                                        v += fac_fz * (V_flat[m * s0 + fzm1_base] - eta_over_q * V_flat[(m + 1) * s0 + fzm1_base])
                                    if has_ez:
                                        v += fac_ez * V_flat[(m + 1) * s0 + ezm1_base]
                                    V_flat[m * s0 + fz1_base] = v


@njit(parallel=True, cache=False, fastmath=True, nogil=True, error_model="numpy")
def _os_coulomb_internal(nbf, nshells, nthreads, max_L, max_nprim, max_nbf_shell,
                         shell_L, shell_centers, shell_bfs_offset, bfs_nbfshell,
                         shell_lmn, bfs_nprim, bfs_expnts, cont_coeffs,
                         density_matrix, schwarz_shell_pair,
                         threshold_schwarz, threshold_density):

    pi = np.pi
    two_pi_52 = 2.0 * pi ** 2.5

    # Precompute shell-pair index data
    n_ab = nshells * (nshells + 1) // 2

    AB_vecs = np.empty((n_ab, 3), dtype=np.float64)
    AB_sqs = np.empty(n_ab, dtype=np.float64)
    ab_shell_a = np.empty(n_ab, dtype=np.int32)
    ab_shell_b = np.empty(n_ab, dtype=np.int32)
    ab_schwarz_bound = np.empty(n_ab, dtype=np.float64)

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
            ab_schwarz_bound[idx] = schwarz_shell_pair[a, b]
            idx += 1

    # Precompute max density per shell pair
    D_shell_max = np.zeros(n_ab, dtype=np.float64)
    for ab_idx in range(n_ab):
        ish_a = ab_shell_a[ab_idx]
        ish_b = ab_shell_b[ab_idx]
        bf_a_start = shell_bfs_offset[ish_a]
        bf_b_start = shell_bfs_offset[ish_b]
        nbf_a = bfs_nbfshell[ish_a]
        nbf_b = bfs_nbfshell[ish_b]
        dmax = 0.0
        for ia in range(nbf_a):
            for jb in range(nbf_b):
                val = abs(density_matrix[bf_a_start + ia, bf_b_start + jb])
                if val > dmax:
                    dmax = val
        D_shell_max[ab_idx] = dmax

    max_L_all = 4 * max_L
    max_dim_bra = 2 * max_L + 1
    max_dim_ket = 2 * max_L + 1
    max_dim_m = max_L_all + 1
    max_prim_pairs = max_nprim * max_nprim
    max_hrr = (max_L + 1) ** 3

    
    # Per-thread J buffers
    J_threads = np.zeros((nthreads, nbf, nbf), dtype=np.float64)

    # Precompute V_flat size
    V_flat_size = max_dim_m * max_dim_bra * max_dim_bra * max_dim_bra * max_dim_ket * max_dim_ket * max_dim_ket

    for ab_idx in prange(n_ab):
        schwarz_ab = ab_schwarz_bound[ab_idx]
        if schwarz_ab < threshold_schwarz:
            continue

        tid = get_thread_id()
        J_local = J_threads[tid]

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

        # Allocate thread-local scratch (outside cd loop)
        bra_p = np.empty(max_prim_pairs, dtype=np.float64)
        bra_K = np.empty(max_prim_pairs, dtype=np.float64)
        bra_Px = np.empty(max_prim_pairs, dtype=np.float64)
        bra_Py = np.empty(max_prim_pairs, dtype=np.float64)
        bra_Pz = np.empty(max_prim_pairs, dtype=np.float64)
        bra_PAx = np.empty(max_prim_pairs, dtype=np.float64)
        bra_PAy = np.empty(max_prim_pairs, dtype=np.float64)
        bra_PAz = np.empty(max_prim_pairs, dtype=np.float64)
        bra_ipa = np.empty(max_prim_pairs, dtype=np.int32)
        bra_ipb = np.empty(max_prim_pairs, dtype=np.int32)

        V_flat = np.empty(V_flat_size, dtype=np.float64)
        F = np.empty(max_dim_m, dtype=np.float64)
        X_PA = np.empty(3, dtype=np.float64)
        X_QC = np.empty(3, dtype=np.float64)
        X_WP = np.empty(3, dtype=np.float64)
        X_WQ = np.empty(3, dtype=np.float64)

        ERI_shell = np.empty((max_nbf_shell, max_nbf_shell,
                              max_nbf_shell, max_nbf_shell), dtype=np.float64)

        # Build bra primitive pairs (once per ab_idx)
        n_bra_pairs = 0
        for ipa in range(nprimi):
            alpha = bfs_expnts[ibf_a0, ipa]
            for ipb in range(nprimj):
                beta = bfs_expnts[ibf_b0, ipb]
                p = alpha + beta
                mu = alpha * beta / p
                K_AB_val = np.exp(-mu * AB_sq)
                if K_AB_val < 1.0e-8:
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

        # Precompute bra HRR coefficients (once per ab_idx)
        bra_hrr_coeffs = np.empty((nbf_a, nbf_b, max_hrr), dtype=np.float64)
        bra_hrr_ax = np.empty((nbf_a, nbf_b, max_hrr), dtype=np.int32)
        bra_hrr_ay = np.empty((nbf_a, nbf_b, max_hrr), dtype=np.int32)
        bra_hrr_az = np.empty((nbf_a, nbf_b, max_hrr), dtype=np.int32)
        bra_hrr_n = np.empty((nbf_a, nbf_b), dtype=np.int32)

        for ic_a in range(nbf_a):
            ax = shell_lmn[ish_a, ic_a, 0]
            ay = shell_lmn[ish_a, ic_a, 1]
            az = shell_lmn[ish_a, ic_a, 2]
            for ic_b in range(nbf_b):
                bx = shell_lmn[ish_b, ic_b, 0]
                by = shell_lmn[ish_b, ic_b, 1]
                bz = shell_lmn[ish_b, ic_b, 2]
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

        # Loop over ket shell pairs
        for cd_idx in range(ab_idx + 1):
            ish_c = ab_shell_a[cd_idx]
            ish_d = ab_shell_b[cd_idx]

            schwarz_cd = ab_schwarz_bound[cd_idx]
            if schwarz_cd < threshold_schwarz:
                continue
            if schwarz_ab * schwarz_cd < threshold_schwarz:
                continue

            if D_shell_max[cd_idx] < threshold_density:
                if D_shell_max[ab_idx] < threshold_density:
                    continue
            if D_shell_max[cd_idx] * schwarz_ab * schwarz_cd < 0.001 * threshold_schwarz:
                continue

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

            is_diag = (ab_idx == cd_idx)

            # Zero ERI_shell
            for ic_a in range(nbf_a):
                for ic_b in range(nbf_b):
                    for ic_c in range(nbf_c):
                        for ic_d in range(nbf_d):
                            ERI_shell[ic_a, ic_b, ic_c, ic_d] = 0.0

            # Build ket HRR coefficients
            ket_hrr_coeffs = np.empty((nbf_c, nbf_d, max_hrr), dtype=np.float64)
            ket_hrr_cx = np.empty((nbf_c, nbf_d, max_hrr), dtype=np.int32)
            ket_hrr_cy = np.empty((nbf_c, nbf_d, max_hrr), dtype=np.int32)
            ket_hrr_cz = np.empty((nbf_c, nbf_d, max_hrr), dtype=np.int32)
            ket_hrr_n = np.empty((nbf_c, nbf_d), dtype=np.int32)

            for ic_c in range(nbf_c):
                cx = shell_lmn[ish_c, ic_c, 0]
                cy = shell_lmn[ish_c, ic_c, 1]
                cz = shell_lmn[ish_c, ic_c, 2]
                for ic_d in range(nbf_d):
                    ddx = shell_lmn[ish_d, ic_d, 0]
                    ddy = shell_lmn[ish_d, ic_d, 1]
                    ddz = shell_lmn[ish_d, ic_d, 2]
                    count = 0
                    for ix in range(ddx + 1):
                        binom_x = comb(ddx, ix)
                        pow_x = X_CD[0] ** (ddx - ix)
                        cx_s = cx + ix
                        for iy in range(ddy + 1):
                            binom_y = comb(ddy, iy)
                            pow_y = X_CD[1] ** (ddy - iy)
                            cy_s = cy + iy
                            for iz in range(ddz + 1):
                                binom_z = comb(ddz, iz)
                                pow_z = X_CD[2] ** (ddz - iz)
                                cz_s = cz + iz
                                coeff = binom_x * pow_x * binom_y * pow_y * binom_z * pow_z
                                ket_hrr_coeffs[ic_c, ic_d, count] = coeff
                                ket_hrr_cx[ic_c, ic_d, count] = cx_s
                                ket_hrr_cy[ic_c, ic_d, count] = cy_s
                                ket_hrr_cz[ic_c, ic_d, count] = cz_s
                                count += 1
                    ket_hrr_n[ic_c, ic_d] = count

            # Precompute VRR index mapping for contraction
            # For each (ic_a, ic_b, ic_c, ic_d) pair, precompute the linear indices into V_flat
            # This avoids recomputing strides in the innermost loop
            s6 = 1
            s5 = dim_ket
            s4 = dim_ket * dim_ket
            s3 = dim_ket * dim_ket * dim_ket
            s2 = dim_bra * s3
            s1 = dim_bra * s2

            # Primitive pair loops
            for ibra in range(n_bra_pairs):
                p = bra_p[ibra]
                K_AB_val = bra_K[ibra]
                ipa = bra_ipa[ibra]
                ipb = bra_ipb[ibra]

                X_PA[0] = bra_PAx[ibra]
                X_PA[1] = bra_PAy[ibra]
                X_PA[2] = bra_PAz[ibra]
                Px = bra_Px[ibra]
                Py = bra_Py[ibra]
                Pz = bra_Pz[ibra]

                for ipc in range(nprimk):
                    gamma = bfs_expnts[ibf_c0, ipc]
                    for ipd in range(npriml):
                        delta = bfs_expnts[ibf_d0, ipd]
                        q = gamma + delta
                        mu_cd = gamma * delta / q
                        K_CD_val = np.exp(-mu_cd * CD_sq)

                        K_prod = K_AB_val * K_CD_val
                        if K_prod < 1.0e-10:
                            continue

                        inv_q = 1.0 / q
                        Qx = (gamma * center_c[0] + delta * center_d[0]) * inv_q
                        Qy = (gamma * center_c[1] + delta * center_d[1]) * inv_q
                        Qz = (gamma * center_c[2] + delta * center_d[2]) * inv_q

                        X_QC[0] = Qx - center_c[0]
                        X_QC[1] = Qy - center_c[1]
                        X_QC[2] = Qz - center_c[2]

                        pq = p + q
                        inv_pq = 1.0 / pq
                        eta = p * q * inv_pq

                        PQx = Px - Qx
                        PQy = Py - Qy
                        PQz = Pz - Qz
                        T_arg = eta * (PQx * PQx + PQy * PQy + PQz * PQz)

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

                        _compute_vrr_ssss(V_flat, dim_bra, dim_ket, dim_m,
                                          X_PA, X_WP, X_QC, X_WQ,
                                          p, q, eta, prefactor, F,
                                          L_bra, L_ket, L_all)

                        # Contract over basis functions
                        for ic_a in range(nbf_a):
                            ibf_a = bf_a_start + ic_a
                            c_a = cont_coeffs[ibf_a, ipa]

                            for ic_b in range(nbf_b):
                                ibf_b = bf_b_start + ic_b
                                c_ab = c_a * cont_coeffs[ibf_b, ipb]

                                # Precompute bra HRR contribution indices
                                n_bra = bra_hrr_n[ic_a, ic_b]

                                for ic_c in range(nbf_c):
                                    ibf_c = bf_c_start + ic_c
                                    c_abc = c_ab * cont_coeffs[ibf_c, ipc]

                                    for ic_d in range(nbf_d):
                                        ibf_d = bf_d_start + ic_d
                                        c_abcd = c_abc * cont_coeffs[ibf_d, ipd]

                                        n_ket_h = ket_hrr_n[ic_c, ic_d]

                                        hrr_val = 0.0
                                        for ib in range(n_bra):
                                            bc = bra_hrr_coeffs[ic_a, ic_b, ib]
                                            ax_f = bra_hrr_ax[ic_a, ic_b, ib]
                                            ay_f = bra_hrr_ay[ic_a, ic_b, ib]
                                            az_f = bra_hrr_az[ic_a, ic_b, ib]
                                            bra_offset = ax_f * s1 + ay_f * s2 + az_f * s3
                                            for ik in range(n_ket_h):
                                                kc = ket_hrr_coeffs[ic_c, ic_d, ik]
                                                cx_f = ket_hrr_cx[ic_c, ic_d, ik]
                                                cy_f = ket_hrr_cy[ic_c, ic_d, ik]
                                                cz_f = ket_hrr_cz[ic_c, ic_d, ik]
                                                hrr_val += bc * kc * V_flat[bra_offset + cx_f * s4 + cy_f * s5 + cz_f]

                                        ERI_shell[ic_a, ic_b, ic_c, ic_d] += c_abcd * hrr_val

            # Accumulate into J matrix
            if not is_diag:
                # Contribution from (ab|cd) -> J[a,b] += D_cd_eff * val
                for ic_a in range(nbf_a):
                    ibf_a_g = bf_a_start + ic_a
                    for ic_b in range(nbf_b):
                        ibf_b_g = bf_b_start + ic_b
                        j_ab_val = 0.0
                        for ic_c in range(nbf_c):
                            ibf_c_g = bf_c_start + ic_c
                            for ic_d in range(nbf_d):
                                ibf_d_g = bf_d_start + ic_d
                                val = ERI_shell[ic_a, ic_b, ic_c, ic_d]
                                if abs(val) < 1e-15:
                                    continue
                                if ish_c != ish_d:
                                    D_cd = density_matrix[ibf_c_g, ibf_d_g] + density_matrix[ibf_d_g, ibf_c_g]
                                else:
                                    D_cd = density_matrix[ibf_c_g, ibf_d_g]
                                j_ab_val += D_cd * val
                        if abs(j_ab_val) > 1e-15:
                            J_local[ibf_a_g, ibf_b_g] += j_ab_val
                            if ish_a != ish_b:
                                J_local[ibf_b_g, ibf_a_g] += j_ab_val

                # Contribution from (cd|ab) -> J[c,d] += D_ab_eff * val
                for ic_c in range(nbf_c):
                    ibf_c_g = bf_c_start + ic_c
                    for ic_d in range(nbf_d):
                        ibf_d_g = bf_d_start + ic_d
                        j_cd_val = 0.0
                        for ic_a in range(nbf_a):
                            ibf_a_g = bf_a_start + ic_a
                            for ic_b in range(nbf_b):
                                ibf_b_g = bf_b_start + ic_b
                                val = ERI_shell[ic_a, ic_b, ic_c, ic_d]
                                if abs(val) < 1e-15:
                                    continue
                                if ish_a != ish_b:
                                    D_ab = density_matrix[ibf_a_g, ibf_b_g] + density_matrix[ibf_b_g, ibf_a_g]
                                else:
                                    D_ab = density_matrix[ibf_a_g, ibf_b_g]
                                j_cd_val += D_ab * val
                        if abs(j_cd_val) > 1e-15:
                            J_local[ibf_c_g, ibf_d_g] += j_cd_val
                            if ish_c != ish_d:
                                J_local[ibf_d_g, ibf_c_g] += j_cd_val
            else:
                for ic_a in range(nbf_a):
                    ibf_a_g = bf_a_start + ic_a
                    for ic_b in range(nbf_b):
                        ibf_b_g = bf_b_start + ic_b
                        j_ab_val = 0.0
                        for ic_c in range(nbf_c):
                            ibf_c_g = bf_c_start + ic_c
                            for ic_d in range(nbf_d):
                                ibf_d_g = bf_d_start + ic_d
                                val = ERI_shell[ic_a, ic_b, ic_c, ic_d]
                                if abs(val) < 1e-15:
                                    continue
                                if ish_c != ish_d:
                                    D_cd = density_matrix[ibf_c_g, ibf_d_g] + density_matrix[ibf_d_g, ibf_c_g]
                                else:
                                    D_cd = density_matrix[ibf_c_g, ibf_d_g]
                                j_ab_val += D_cd * val
                        if abs(j_ab_val) > 1e-15:
                            J_local[ibf_a_g, ibf_b_g] += j_ab_val
                            if ish_a != ish_b:
                                J_local[ibf_b_g, ibf_a_g] += j_ab_val

    # Reduce per-thread J buffers
    J_matrix = np.zeros((nbf, nbf), dtype=np.float64)
    for t in range(nthreads):
        for i in prange(nbf):
            for j in range(nbf):
                J_matrix[i, j] += J_threads[t, i, j]

    return J_matrix
# import numpy as np
# from numba import njit, prange, get_num_threads, get_thread_id
# from .integral_helpers import Fboys
# from .integral_helpers import comb
# from .schwarz_helpers import eri_4c2e_diag

# def os_coulomb_matrix(basis, density_matrix, schwarz_shell_pair=None, 
#                       threshold_schwarz=1e-9, threshold_density=1e-10):
#     """
#     Compute the Coulomb matrix (J matrix) directly from the density matrix
#     using the Obara-Saika recurrence scheme with exploitation of 8-fold 
#     permutational symmetry.

#     The Coulomb matrix is defined as:
#         J[i,j] = Σ_kl D[k,l] * (ij|kl)
    
#     where D is the density matrix and (ij|kl) are the electron repulsion integrals.

#     This function computes J directly without storing the full 4-center 2-electron
#     integral tensor, significantly reducing memory requirements.

#     Symmetries exploited
#     --------------------
#     The 4c2e integrals obey 8-fold permutational symmetry:
#         (i j | k l) = (j i | k l) = (i j | l k) = (j i | l k)
#                     = (k l | i j) = (l k | i j) = (k l | j i) = (l k | j i)

#     The density matrix is symmetric: D[k,l] = D[l,k]

#     For the Coulomb matrix J[i,j] = Σ_kl D[k,l] * (ij|kl), we exploit:
#     - Shell-quartet symmetry (ab|cd) = (cd|ab) to process each unique 
#       shell quartet only once
#     - Bra symmetry (ab|cd) = (ba|cd) to accumulate into both J[a,b] and J[b,a]
#     - Ket symmetry (ab|cd) = (ab|dc) to combine D[c,d] + D[d,c] contributions
#     - Diagonal shell quartet special handling to avoid double-counting

#     Screening
#     ---------
#     - Schwarz screening: skip shell quartets where 
#       schwarz_shell_pair[ab] * schwarz_shell_pair[cd] < threshold_schwarz
#     - Density screening: skip shell quartets where the maximum density element
#       in the ket shell pair is below threshold_density

#     Parameters
#     ----------
#     basis : object
#         Primary basis set object containing shell and basis function information.

#     density_matrix : ndarray, shape (N, N)
#         The density matrix in the AO basis. Must be symmetric.

#     schwarz_shell_pair : ndarray, shape (nshells, nshells), optional
#         Shell-pair Schwarz upper bounds. schwarz_shell_pair[i,j] is the maximum
#         of sqrt(|(ab|ab)|) over all basis functions a in shell i, b in shell j.
#         If None, no Schwarz screening is performed.

#     threshold_schwarz : float, optional
#         Threshold for Schwarz screening. Shell quartets with estimated integral
#         bound below this value are skipped. Default is 1e-9.

#     threshold_density : float, optional
#         Threshold for density-based screening. Shell pairs with maximum density
#         element below this value are skipped. Default is 1e-10.

#     Returns
#     -------
#     J_matrix : ndarray, shape (N, N)
#         The Coulomb matrix in the AO basis. Symmetric.

#     Notes
#     -----
#     - Memory efficient: Does not store the full O(N^4) ERI tensor.
#     - Uses combined Schwarz and density screening for efficient integral 
#       selection.
#     - Parallelized using Numba's prange over bra shell pairs.
#     - Thread-safe accumulation using per-thread J matrix buffers.

#     Examples
#     --------
#     >>> J = os_coulomb_matrix(basis, density_matrix)
#     >>> # With screening:
#     >>> J = os_coulomb_matrix(basis, density_matrix, schwarz_shell_pair, 
#     ...                       threshold_schwarz=1e-10, threshold_density=1e-10)
#     """
#     # Convert basis data to numpy arrays for Numba
#     nbf = basis.bfs_nao
#     nshells = len(basis.shells)
    
#     # Shell data
#     shell_L = np.array([basis.bfs_lm[i] for i in basis.shell_bfs_offset], dtype=np.int32)
#     shell_centers = np.array([basis.bfs_coords[i] for i in basis.shell_bfs_offset], dtype=np.float64)
#     shell_bfs_offset = np.array(basis.shell_bfs_offset, dtype=np.int32)
#     bfs_nbfshell = np.array(basis.bfs_nbfshell, dtype=np.int32)
    
#     # Basis function data
#     bfs_coords = np.array(basis.bfs_coords, dtype=np.float64)
#     bfs_contr_prim_norms = np.array(basis.bfs_contr_prim_norms, dtype=np.float64)
#     bfs_lmn = np.array(basis.bfs_lmn, dtype=np.int32)
#     bfs_nprim = np.array(basis.bfs_nprim, dtype=np.int32)
#     bfs_shell_index = np.array(basis.bfs_shell_index, dtype=np.int32)
    
#     # Primitive data (padded arrays)
#     maxnprim = max(basis.bfs_nprim)
#     bfs_coeffs = np.zeros((nbf, maxnprim), dtype=np.float64)
#     bfs_expnts = np.zeros((nbf, maxnprim), dtype=np.float64)
#     bfs_prim_norms = np.zeros((nbf, maxnprim), dtype=np.float64)
    
#     for i in range(nbf):
#         for j in range(basis.bfs_nprim[i]):
#             bfs_coeffs[i, j] = basis.bfs_coeffs[i][j]
#             bfs_expnts[i, j] = basis.bfs_expnts[i][j]
#             bfs_prim_norms[i, j] = basis.bfs_prim_norms[i][j]
    
#     if schwarz_shell_pair is None:
#         # Setup Schwarz screening
#         ints4c2e_diag = eri_4c2e_diag(basis)    
#         sqrt_ints4c2e_diag = np.sqrt(np.abs(ints4c2e_diag))

#     if schwarz_shell_pair is None:
#         # Build shell-pair Schwarz upper bounds: max over bf pairs in the shell pair
#         # schwarz_shell_pair[i_shell, j_shell] = max_{a in i, b in j} sqrt(|(ab|ab)|)
#         schwarz_shell_pair = np.zeros((nshells, nshells), dtype=np.float64)
#         for ish in range(nshells):
#             bf_i_start = shell_bfs_offset[ish]
#             nbf_i = bfs_nbfshell[ish]
#             for jsh in range(ish + 1):
#                 bf_j_start = shell_bfs_offset[jsh]
#                 nbf_j = bfs_nbfshell[jsh]
#                 max_val = 0.0
#                 for ia in range(nbf_i):
#                     for jb in range(nbf_j):
#                         val = sqrt_ints4c2e_diag[bf_i_start + ia, bf_j_start + jb]
#                         if val > max_val:
#                             max_val = val
#                 schwarz_shell_pair[ish, jsh] = max_val
#                 schwarz_shell_pair[jsh, ish] = max_val
    
#     # Ensure density matrix is contiguous
#     density_matrix = np.ascontiguousarray(density_matrix, dtype=np.float64)
    
#     nthreads = get_num_threads()
    
#     J_matrix = os_coulomb_matrix_internal(
#         nbf, nshells, nthreads, shell_L, shell_centers, shell_bfs_offset,
#         bfs_nbfshell, bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim,
#         bfs_shell_index, bfs_coeffs, bfs_expnts, bfs_prim_norms,
#         density_matrix, schwarz_shell_pair,
#         threshold_schwarz, threshold_density
#     )
    
#     return J_matrix


# @njit(parallel=True, cache=False, fastmath=True, nogil=True, error_model="numpy")
# def os_coulomb_matrix_internal(nbf, nshells, nthreads, shell_L, shell_centers,
#                                 shell_bfs_offset, bfs_nbfshell, bfs_coords,
#                                 bfs_contr_prim_norms, bfs_lmn, bfs_nprim,
#                                 bfs_shell_index, bfs_coeffs, bfs_expnts,
#                                 bfs_prim_norms, density_matrix,
#                                 schwarz_shell_pair,
#                                 threshold_schwarz, threshold_density):
    
    
        
#     pi = np.pi
#     two_pi_52 = 2.0 * pi ** 2.5

#     # Precompute shell-pair index data
#     n_ab = nshells * (nshells + 1) // 2

#     AB_vecs = np.zeros((n_ab, 3), dtype=np.float64)
#     AB_sqs = np.zeros(n_ab, dtype=np.float64)
#     ab_shell_a = np.zeros(n_ab, dtype=np.int64)
#     ab_shell_b = np.zeros(n_ab, dtype=np.int64)
#     ab_schwarz_bound = np.zeros(n_ab, dtype=np.float64)

#     idx = 0
#     for a in range(nshells):
#         for b in range(a + 1):
#             ab_shell_a[idx] = a
#             ab_shell_b[idx] = b
#             dx = shell_centers[a, 0] - shell_centers[b, 0]
#             dy = shell_centers[a, 1] - shell_centers[b, 1]
#             dz = shell_centers[a, 2] - shell_centers[b, 2]
#             AB_vecs[idx, 0] = dx
#             AB_vecs[idx, 1] = dy
#             AB_vecs[idx, 2] = dz
#             AB_sqs[idx] = dx * dx + dy * dy + dz * dz

#             ab_schwarz_bound[idx] = schwarz_shell_pair[a, b]
#             idx += 1

#     # Precompute shell-pair max density for screening
#     # D_shell_max[ab_idx] = max |D[a,b]| over all bf a in shell A, bf b in shell B
#     D_shell_max = np.zeros(n_ab, dtype=np.float64)
#     for ab_idx in prange(n_ab):
#         ish_a = ab_shell_a[ab_idx]
#         ish_b = ab_shell_b[ab_idx]
#         bf_a_start = shell_bfs_offset[ish_a]
#         bf_b_start = shell_bfs_offset[ish_b]
#         nbf_a = bfs_nbfshell[ish_a]
#         nbf_b = bfs_nbfshell[ish_b]
#         dmax = 0.0
#         for ia in range(nbf_a):
#             for jb in range(nbf_b):
#                 val = abs(density_matrix[bf_a_start + ia, bf_b_start + jb])
#                 if val > dmax:
#                     dmax = val
#         D_shell_max[ab_idx] = dmax

#     # Find max sizes for scratch allocation
#     max_L = 0
#     max_nprim = 0
#     max_nbf_shell = 0
#     for i in range(nshells):
#         if shell_L[i] > max_L:
#             max_L = shell_L[i]
#         ibf0 = shell_bfs_offset[i]
#         if bfs_nprim[ibf0] > max_nprim:
#             max_nprim = bfs_nprim[ibf0]
#         if bfs_nbfshell[i] > max_nbf_shell:
#             max_nbf_shell = bfs_nbfshell[i]

#     max_L_all = 4 * max_L
#     max_dim = 2 * max_L + 1
#     max_dim_m = max_L_all + 1
#     max_prim_pairs = max_nprim * max_nprim
#     max_hrr = (max_L + 1) ** 3

#     # Per-thread J buffers for thread-safe accumulation
#     J_threads = np.zeros((nthreads, nbf, nbf), dtype=np.float64)

#     for ab_idx in prange(n_ab):
#         schwarz_ab = ab_schwarz_bound[ab_idx]
#         if schwarz_ab < threshold_schwarz:
#             continue
#         # if D_shell_max[ab_idx] < threshold_density:
#         #     continue
#         # if D_shell_max[ab_idx]*schwarz_ab < 0.1*threshold_density:
#         #     continue
#         # Get thread id for thread-local accumulation
#         tid = get_thread_id()
#         J_local = J_threads[tid]

#         ish_a = ab_shell_a[ab_idx]
#         ish_b = ab_shell_b[ab_idx]

#         L_a = shell_L[ish_a]
#         L_b = shell_L[ish_b]
#         L_bra = L_a + L_b
#         bf_a_start = shell_bfs_offset[ish_a]
#         bf_b_start = shell_bfs_offset[ish_b]
#         nbf_a = bfs_nbfshell[ish_a]
#         nbf_b = bfs_nbfshell[ish_b]

#         X_AB = AB_vecs[ab_idx]
#         AB_sq = AB_sqs[ab_idx]

#         center_a = shell_centers[ish_a]
#         center_b = shell_centers[ish_b]

#         ibf_a0 = bf_a_start
#         ibf_b0 = bf_b_start
#         nprimi = bfs_nprim[ibf_a0]
#         nprimj = bfs_nprim[ibf_b0]

        

#         # Scratch allocations for this thread
#         bra_p = np.empty(max_prim_pairs, dtype=np.float64)
#         bra_K = np.empty(max_prim_pairs, dtype=np.float64)
#         bra_Px = np.empty(max_prim_pairs, dtype=np.float64)
#         bra_Py = np.empty(max_prim_pairs, dtype=np.float64)
#         bra_Pz = np.empty(max_prim_pairs, dtype=np.float64)
#         bra_PAx = np.empty(max_prim_pairs, dtype=np.float64)
#         bra_PAy = np.empty(max_prim_pairs, dtype=np.float64)
#         bra_PAz = np.empty(max_prim_pairs, dtype=np.float64)
#         bra_alpha = np.empty(max_prim_pairs, dtype=np.float64)
#         bra_beta = np.empty(max_prim_pairs, dtype=np.float64)
#         bra_ipa = np.empty(max_prim_pairs, dtype=np.int64)
#         bra_ipb = np.empty(max_prim_pairs, dtype=np.int64)

#         ket_q = np.empty(max_prim_pairs, dtype=np.float64)
#         ket_K = np.empty(max_prim_pairs, dtype=np.float64)
#         ket_Qx = np.empty(max_prim_pairs, dtype=np.float64)
#         ket_Qy = np.empty(max_prim_pairs, dtype=np.float64)
#         ket_Qz = np.empty(max_prim_pairs, dtype=np.float64)
#         ket_QCx = np.empty(max_prim_pairs, dtype=np.float64)
#         ket_QCy = np.empty(max_prim_pairs, dtype=np.float64)
#         ket_QCz = np.empty(max_prim_pairs, dtype=np.float64)
#         ket_gamma = np.empty(max_prim_pairs, dtype=np.float64)
#         ket_delta = np.empty(max_prim_pairs, dtype=np.float64)
#         ket_ipc = np.empty(max_prim_pairs, dtype=np.int64)
#         ket_ipd = np.empty(max_prim_pairs, dtype=np.int64)

#         V_vrr = np.zeros((max_dim_m, max_dim, max_dim, max_dim,
#                           max_dim, max_dim, max_dim), dtype=np.float64)

#         ERI_shell = np.zeros((max_nbf_shell, max_nbf_shell,
#                               max_nbf_shell, max_nbf_shell), dtype=np.float64)

#         bra_hrr_coeffs = np.empty((max_nbf_shell, max_nbf_shell, max_hrr), dtype=np.float64)
#         bra_hrr_ax = np.empty((max_nbf_shell, max_nbf_shell, max_hrr), dtype=np.int64)
#         bra_hrr_ay = np.empty((max_nbf_shell, max_nbf_shell, max_hrr), dtype=np.int64)
#         bra_hrr_az = np.empty((max_nbf_shell, max_nbf_shell, max_hrr), dtype=np.int64)
#         bra_hrr_n = np.empty((max_nbf_shell, max_nbf_shell), dtype=np.int64)

#         ket_hrr_coeffs = np.empty((max_nbf_shell, max_nbf_shell, max_hrr), dtype=np.float64)
#         ket_hrr_cx = np.empty((max_nbf_shell, max_nbf_shell, max_hrr), dtype=np.int64)
#         ket_hrr_cy = np.empty((max_nbf_shell, max_nbf_shell, max_hrr), dtype=np.int64)
#         ket_hrr_cz = np.empty((max_nbf_shell, max_nbf_shell, max_hrr), dtype=np.int64)
#         ket_hrr_n = np.empty((max_nbf_shell, max_nbf_shell), dtype=np.int64)

#         X_PA = np.empty(3, dtype=np.float64)
#         X_QC = np.empty(3, dtype=np.float64)
#         X_WP = np.empty(3, dtype=np.float64)
#         X_WQ = np.empty(3, dtype=np.float64)
#         F = np.empty(max_dim_m, dtype=np.float64)

#         # Build bra primitive pairs
#         n_bra_pairs = 0
#         for ipa in range(nprimi):
#             alpha = bfs_expnts[ibf_a0, ipa]
#             for ipb in range(nprimj):
#                 beta = bfs_expnts[ibf_b0, ipb]
#                 p = alpha + beta
#                 mu = alpha * beta / p
#                 K_AB_val = np.exp(-mu * AB_sq)

#                 if abs(K_AB_val) < 1.0e-8:
#                     continue

#                 ii = n_bra_pairs
#                 bra_p[ii] = p
#                 bra_K[ii] = K_AB_val
#                 bra_alpha[ii] = alpha
#                 bra_beta[ii] = beta
#                 bra_ipa[ii] = ipa
#                 bra_ipb[ii] = ipb

#                 inv_p = 1.0 / p
#                 Px = (alpha * center_a[0] + beta * center_b[0]) * inv_p
#                 Py = (alpha * center_a[1] + beta * center_b[1]) * inv_p
#                 Pz = (alpha * center_a[2] + beta * center_b[2]) * inv_p
#                 bra_Px[ii] = Px
#                 bra_Py[ii] = Py
#                 bra_Pz[ii] = Pz
#                 bra_PAx[ii] = Px - center_a[0]
#                 bra_PAy[ii] = Py - center_a[1]
#                 bra_PAz[ii] = Pz - center_a[2]

#                 n_bra_pairs += 1

#         if n_bra_pairs == 0:
#             continue

#         # Build bra HRR coefficients
#         for ic_a in range(nbf_a):
#             ibf_a = bf_a_start + ic_a
#             ax, ay, az = bfs_lmn[ibf_a, 0], bfs_lmn[ibf_a, 1], bfs_lmn[ibf_a, 2]
#             for ic_b in range(nbf_b):
#                 ibf_b = bf_b_start + ic_b
#                 bx, by, bz = bfs_lmn[ibf_b, 0], bfs_lmn[ibf_b, 1], bfs_lmn[ibf_b, 2]
#                 count = 0
#                 for px in range(bx + 1):
#                     binom_x = comb(bx, px)
#                     pow_x = X_AB[0] ** (bx - px)
#                     ax_f = ax + px
#                     for py in range(by + 1):
#                         binom_y = comb(by, py)
#                         pow_y = X_AB[1] ** (by - py)
#                         ay_f = ay + py
#                         for pz in range(bz + 1):
#                             binom_z = comb(bz, pz)
#                             pow_z = X_AB[2] ** (bz - pz)
#                             az_f = az + pz
#                             coeff = binom_x * pow_x * binom_y * pow_y * binom_z * pow_z
#                             bra_hrr_coeffs[ic_a, ic_b, count] = coeff
#                             bra_hrr_ax[ic_a, ic_b, count] = ax_f
#                             bra_hrr_ay[ic_a, ic_b, count] = ay_f
#                             bra_hrr_az[ic_a, ic_b, count] = az_f
#                             count += 1
#                 bra_hrr_n[ic_a, ic_b] = count

#         # Loop over ket shell pairs
#         for cd_idx in range(ab_idx + 1):
#             ish_c = ab_shell_a[cd_idx]
#             ish_d = ab_shell_b[cd_idx]

#             # Schwarz screening at shell-quartet level
#             schwarz_cd = ab_schwarz_bound[cd_idx]
#             if schwarz_cd < threshold_schwarz:
#                 continue
            
#             if schwarz_ab * schwarz_cd < threshold_schwarz:
#                 continue

#             # Density screening: check if max |D| in ket shell pair is significant
#             # if D_shell_max[cd_idx] < threshold_density:
#             #     continue
#             # Also check bra as ket (for bra-ket exchange contribution)
#             if D_shell_max[cd_idx] < threshold_density:
#                 if D_shell_max[ab_idx] < threshold_density:
#                     continue
#             if D_shell_max[cd_idx] * schwarz_ab * schwarz_cd < 0.001*threshold_schwarz:
#                 continue

#             L_c = shell_L[ish_c]
#             L_d = shell_L[ish_d]
#             L_ket = L_c + L_d
#             bf_c_start = shell_bfs_offset[ish_c]
#             bf_d_start = shell_bfs_offset[ish_d]
#             nbf_c = bfs_nbfshell[ish_c]
#             nbf_d = bfs_nbfshell[ish_d]

#             center_c = shell_centers[ish_c]
#             center_d = shell_centers[ish_d]

#             X_CD = AB_vecs[cd_idx]
#             CD_sq = AB_sqs[cd_idx]

#             L_all = L_bra + L_ket
#             dim_bra = L_bra + 1
#             dim_ket = L_ket + 1
#             dim_m = L_all + 1

#             ibf_c0 = bf_c_start
#             ibf_d0 = bf_d_start
#             nprimk = bfs_nprim[ibf_c0]
#             npriml = bfs_nprim[ibf_d0]

#             # Determine if this is a diagonal quartet (ab_idx == cd_idx)
#             is_diag = (ab_idx == cd_idx)

#             # Build ket primitive pairs
#             n_ket_pairs = 0
#             for ipc in range(nprimk):
#                 gamma = bfs_expnts[ibf_c0, ipc]
#                 for ipd in range(npriml):
#                     delta = bfs_expnts[ibf_d0, ipd]
#                     q = gamma + delta
#                     mu_cd = gamma * delta / q
#                     K_CD_val = np.exp(-mu_cd * CD_sq)

#                     if abs(K_CD_val) < 1.0e-8:
#                         continue

#                     ii = n_ket_pairs
#                     ket_q[ii] = q
#                     ket_K[ii] = K_CD_val
#                     ket_gamma[ii] = gamma
#                     ket_delta[ii] = delta
#                     ket_ipc[ii] = ipc
#                     ket_ipd[ii] = ipd

#                     inv_q = 1.0 / q
#                     Qx = (gamma * center_c[0] + delta * center_d[0]) * inv_q
#                     Qy = (gamma * center_c[1] + delta * center_d[1]) * inv_q
#                     Qz = (gamma * center_c[2] + delta * center_d[2]) * inv_q
#                     ket_Qx[ii] = Qx
#                     ket_Qy[ii] = Qy
#                     ket_Qz[ii] = Qz
#                     ket_QCx[ii] = Qx - center_c[0]
#                     ket_QCy[ii] = Qy - center_c[1]
#                     ket_QCz[ii] = Qz - center_c[2]

#                     n_ket_pairs += 1

#             if n_ket_pairs == 0:
#                 continue

#             # Zero ERI_shell
#             for ic_a in range(nbf_a):
#                 for ic_b in range(nbf_b):
#                     for ic_c in range(nbf_c):
#                         for ic_d in range(nbf_d):
#                             ERI_shell[ic_a, ic_b, ic_c, ic_d] = 0.0
#             # ERI_shell[:, :, :, :] = 0.0

#             # Build ket HRR coefficients
#             for ic_c in range(nbf_c):
#                 ibf_c = bf_c_start + ic_c
#                 cx, cy, cz = bfs_lmn[ibf_c, 0], bfs_lmn[ibf_c, 1], bfs_lmn[ibf_c, 2]
#                 for ic_d in range(nbf_d):
#                     ibf_d = bf_d_start + ic_d
#                     ddx, ddy, ddz = bfs_lmn[ibf_d, 0], bfs_lmn[ibf_d, 1], bfs_lmn[ibf_d, 2]
#                     count = 0
#                     for ix in range(ddx + 1):
#                         binom_x = comb(ddx, ix)
#                         pow_x = X_CD[0] ** (ddx - ix)
#                         cx_s = cx + ix
#                         for iy in range(ddy + 1):
#                             binom_y = comb(ddy, iy)
#                             pow_y = X_CD[1] ** (ddy - iy)
#                             cy_s = cy + iy
#                             for iz in range(ddz + 1):
#                                 binom_z = comb(ddz, iz)
#                                 pow_z = X_CD[2] ** (ddz - iz)
#                                 cz_s = cz + iz
#                                 coeff = binom_x * pow_x * binom_y * pow_y * binom_z * pow_z
#                                 ket_hrr_coeffs[ic_c, ic_d, count] = coeff
#                                 ket_hrr_cx[ic_c, ic_d, count] = cx_s
#                                 ket_hrr_cy[ic_c, ic_d, count] = cy_s
#                                 ket_hrr_cz[ic_c, ic_d, count] = cz_s
#                                 count += 1
#                     ket_hrr_n[ic_c, ic_d] = count

#             # Primitive pair loops - compute ERI shell block
#             for ibra in range(n_bra_pairs):
#                 p = bra_p[ibra]
#                 K_AB_val = bra_K[ibra]
#                 ipa = bra_ipa[ibra]
#                 ipb = bra_ipb[ibra]

#                 Px = bra_Px[ibra]
#                 Py = bra_Py[ibra]
#                 Pz = bra_Pz[ibra]
#                 X_PA[0] = bra_PAx[ibra]
#                 X_PA[1] = bra_PAy[ibra]
#                 X_PA[2] = bra_PAz[ibra]

#                 for iket in range(n_ket_pairs):
#                     q = ket_q[iket]
#                     K_CD_val = ket_K[iket]
#                     ipc = ket_ipc[iket]
#                     ipd = ket_ipd[iket]

#                     K_prod = K_AB_val * K_CD_val
#                     if abs(K_prod) < 1.0e-10:
#                         continue

#                     Qx = ket_Qx[iket]
#                     Qy = ket_Qy[iket]
#                     Qz = ket_Qz[iket]

#                     X_QC[0] = ket_QCx[iket]
#                     X_QC[1] = ket_QCy[iket]
#                     X_QC[2] = ket_QCz[iket]

#                     pq = p + q
#                     inv_pq = 1.0 / pq
#                     eta = p * q * inv_pq

#                     PQx = Px - Qx
#                     PQy = Py - Qy
#                     PQz = Pz - Qz
#                     PQ_sq = PQx * PQx + PQy * PQy + PQz * PQz
#                     T_arg = eta * PQ_sq

#                     Wx = (p * Px + q * Qx) * inv_pq
#                     Wy = (p * Py + q * Qy) * inv_pq
#                     Wz = (p * Pz + q * Qz) * inv_pq
#                     X_WP[0] = Wx - Px
#                     X_WP[1] = Wy - Py
#                     X_WP[2] = Wz - Pz
#                     X_WQ[0] = Wx - Qx
#                     X_WQ[1] = Wy - Qy
#                     X_WQ[2] = Wz - Qz

#                     prefactor = two_pi_52 / (p * q * np.sqrt(pq)) * K_prod

#                     for m in range(dim_m):
#                         F[m] = Fboys(m, T_arg)

#                     # Zero V_vrr
#                     # for m in range(dim_m):
#                     #     for i1 in range(dim_bra):
#                     #         for i2 in range(dim_bra):
#                     #             for i3 in range(dim_bra):
#                     #                 for i4 in range(dim_ket):
#                     #                     for i5 in range(dim_ket):
#                     #                         for i6 in range(dim_ket):
#                     #                             V_vrr[m, i1, i2, i3, i4, i5, i6] = 0.0

#                     for m in range(dim_m):
#                         V_vrr[m, 0, 0, 0, 0, 0, 0] = prefactor * F[m]

#                     if L_bra > 0:
#                         os_bra_vrr_opt(V_vrr, X_PA, X_WP, p, eta, L_bra, L_all)
#                     if L_ket > 0:
#                         os_ket_vrr_opt(V_vrr, X_QC, X_WQ, q, eta, L_bra, L_ket, L_all)

#                     # Contract over basis functions and accumulate into ERI_shell
#                     for ic_a in range(nbf_a):
#                         ibf_a = bf_a_start + ic_a
#                         da = bfs_coeffs[ibf_a, ipa]
#                         Nik = bfs_prim_norms[ibf_a, ipa]
#                         Ni = bfs_contr_prim_norms[ibf_a]
#                         c_a = Ni * da * Nik

#                         for ic_b in range(nbf_b):
#                             ibf_b = bf_b_start + ic_b
#                             db = bfs_coeffs[ibf_b, ipb]
#                             Njk = bfs_prim_norms[ibf_b, ipb]
#                             Nj = bfs_contr_prim_norms[ibf_b]
#                             c_ab = c_a * Nj * db * Njk

#                             n_bra = bra_hrr_n[ic_a, ic_b]

#                             for ic_c in range(nbf_c):
#                                 ibf_c = bf_c_start + ic_c
#                                 dc = bfs_coeffs[ibf_c, ipc]
#                                 Nkk = bfs_prim_norms[ibf_c, ipc]
#                                 Nk = bfs_contr_prim_norms[ibf_c]
#                                 c_abc = c_ab * Nk * dc * Nkk

#                                 for ic_d in range(nbf_d):
#                                     ibf_d = bf_d_start + ic_d
#                                     dd = bfs_coeffs[ibf_d, ipd]
#                                     Nlk = bfs_prim_norms[ibf_d, ipd]
#                                     Nl = bfs_contr_prim_norms[ibf_d]
#                                     c_abcd = c_abc * Nl * dd * Nlk

#                                     n_ket_h = ket_hrr_n[ic_c, ic_d]

#                                     hrr_val = 0.0
#                                     for ib in range(n_bra):
#                                         bc = bra_hrr_coeffs[ic_a, ic_b, ib]
#                                         ax_f = bra_hrr_ax[ic_a, ic_b, ib]
#                                         ay_f = bra_hrr_ay[ic_a, ic_b, ib]
#                                         az_f = bra_hrr_az[ic_a, ic_b, ib]
#                                         for ik in range(n_ket_h):
#                                             kc = ket_hrr_coeffs[ic_c, ic_d, ik]
#                                             cx_f = ket_hrr_cx[ic_c, ic_d, ik]
#                                             cy_f = ket_hrr_cy[ic_c, ic_d, ik]
#                                             cz_f = ket_hrr_cz[ic_c, ic_d, ik]
#                                             hrr_val += bc * kc * V_vrr[0, ax_f, ay_f, az_f, cx_f, cy_f, cz_f]

#                                     ERI_shell[ic_a, ic_b, ic_c, ic_d] += c_abcd * hrr_val

#             # ================================================================
#             # Accumulate ERI_shell into J matrix exploiting symmetry
#             # ================================================================
#             # We have computed (AB|CD) for unique shell quartet ab_idx >= cd_idx.
#             #
#             # The integral (ab|cd) has 8-fold symmetry:
#             #   (ab|cd) = (ba|cd) = (ab|dc) = (ba|dc)
#             #           = (cd|ab) = (dc|ab) = (cd|ba) = (dc|ba)
#             #
#             # For J[i,j] = Σ_kl D[k,l] * (ij|kl), each unique integral
#             # contributes to multiple J elements.
#             #
#             # Case 1: ab_idx != cd_idx (off-diagonal shell quartets)
#             #   From (ab|cd): J[a,b] += D_cd * val, J[b,a] += D_cd * val
#             #   From (cd|ab): J[c,d] += D_ab * val, J[d,c] += D_ab * val
#             #   where D_cd accounts for ket symmetry: 
#             #     if c != d: D_cd = D[c,d] + D[d,c] = 2*D[c,d]
#             #     if c == d: D_cd = D[c,c]
#             #   Similarly for D_ab.
#             #
#             # Case 2: ab_idx == cd_idx (diagonal shell quartets)
#             #   Only accumulate once (no bra-ket exchange doubling).
#             #   But still handle within-shell symmetries carefully.
#             # ================================================================

#             if not is_diag:
#                 for ic_a in range(nbf_a):
#                     ibf_a_g = bf_a_start + ic_a
#                     for ic_b in range(nbf_b):
#                         ibf_b_g = bf_b_start + ic_b

#                         j_ab_val = 0.0
#                         for ic_c in range(nbf_c):
#                             ibf_c_g = bf_c_start + ic_c
#                             for ic_d in range(nbf_d):
#                                 ibf_d_g = bf_d_start + ic_d
#                                 val = ERI_shell[ic_a, ic_b, ic_c, ic_d]
#                                 if abs(val) < 1e-12:
#                                     continue

#                                 # If ish_c == ish_d, both (c,d) and (d,c) appear
#                                 # in the loop, so don't double the density
#                                 if ish_c != ish_d and ibf_c_g != ibf_d_g:
#                                     D_cd = density_matrix[ibf_c_g, ibf_d_g] + density_matrix[ibf_d_g, ibf_c_g]
#                                 else:
#                                     D_cd = density_matrix[ibf_c_g, ibf_d_g]
#                                 j_ab_val += D_cd * val

#                         if abs(j_ab_val) > 1e-12:
#                             J_local[ibf_a_g, ibf_b_g] += j_ab_val
#                             if ish_a != ish_b and ibf_a_g != ibf_b_g:
#                                 J_local[ibf_b_g, ibf_a_g] += j_ab_val

#                 for ic_c in range(nbf_c):
#                     ibf_c_g = bf_c_start + ic_c
#                     for ic_d in range(nbf_d):
#                         ibf_d_g = bf_d_start + ic_d

#                         j_cd_val = 0.0
#                         for ic_a in range(nbf_a):
#                             ibf_a_g = bf_a_start + ic_a
#                             for ic_b in range(nbf_b):
#                                 ibf_b_g = bf_b_start + ic_b
#                                 val = ERI_shell[ic_a, ic_b, ic_c, ic_d]
#                                 if abs(val) < 1e-12:
#                                     continue

#                                 if ish_a != ish_b and ibf_a_g != ibf_b_g:
#                                     D_ab = density_matrix[ibf_a_g, ibf_b_g] + density_matrix[ibf_b_g, ibf_a_g]
#                                 else:
#                                     D_ab = density_matrix[ibf_a_g, ibf_b_g]
#                                 j_cd_val += D_ab * val

#                         if abs(j_cd_val) > 1e-12:
#                             J_local[ibf_c_g, ibf_d_g] += j_cd_val
#                             if ish_c != ish_d and ibf_c_g != ibf_d_g:
#                                 J_local[ibf_d_g, ibf_c_g] += j_cd_val

#             else:
#                 for ic_a in range(nbf_a):
#                     ibf_a_g = bf_a_start + ic_a
#                     for ic_b in range(nbf_b):
#                         ibf_b_g = bf_b_start + ic_b

#                         j_ab_val = 0.0
#                         for ic_c in range(nbf_c):
#                             ibf_c_g = bf_c_start + ic_c
#                             for ic_d in range(nbf_d):
#                                 ibf_d_g = bf_d_start + ic_d
#                                 val = ERI_shell[ic_a, ic_b, ic_c, ic_d]
#                                 if abs(val) < 1e-12:
#                                     continue

#                                 if ish_c != ish_d and ibf_c_g != ibf_d_g:
#                                     D_cd = density_matrix[ibf_c_g, ibf_d_g] + density_matrix[ibf_d_g, ibf_c_g]
#                                 else:
#                                     D_cd = density_matrix[ibf_c_g, ibf_d_g]
#                                 j_ab_val += D_cd * val

#                         if abs(j_ab_val) > 1e-12:
#                             J_local[ibf_a_g, ibf_b_g] += j_ab_val
#                             if ish_a != ish_b and ibf_a_g != ibf_b_g:
#                                 J_local[ibf_b_g, ibf_a_g] += j_ab_val

#     # Reduce per-thread J buffers
#     J_matrix = np.zeros((nbf, nbf), dtype=np.float64)
#     for t in range(nthreads):
#         for i in prange(nbf):
#             for j in range(nbf):
#                 J_matrix[i, j] += J_threads[t, i, j]

#     return J_matrix

# @njit(cache=True, fastmath=True, nogil=True, error_model="numpy", inline='always')
# def os_bra_vrr_opt(V, X_PA, X_WP, p, eta, L_bra, m_total):
#     inv_2p = 0.5 / p
#     eta_over_p = eta / p

#     # Build angular momentum shell by shell
#     for L_e in range(L_bra):
#         m_top = m_total - L_e - 1

#         for ax in range(L_e + 1):
#             for ay in range(L_e + 1 - ax):
#                 az = L_e - ax - ay

#                 # Increment x
#                 ax1 = ax + 1
#                 for m in range(m_top + 1):
#                     v = X_PA[0] * V[m, ax, ay, az, 0, 0, 0] + X_WP[0] * V[m+1, ax, ay, az, 0, 0, 0]
#                     if ax > 0:
#                         v += ax * inv_2p * (V[m, ax-1, ay, az, 0, 0, 0] - eta_over_p * V[m+1, ax-1, ay, az, 0, 0, 0])
#                     V[m, ax1, ay, az, 0, 0, 0] = v

#                 # Increment y
#                 ay1 = ay + 1
#                 for m in range(m_top + 1):
#                     v = X_PA[1] * V[m, ax, ay, az, 0, 0, 0] + X_WP[1] * V[m+1, ax, ay, az, 0, 0, 0]
#                     if ay > 0:
#                         v += ay * inv_2p * (V[m, ax, ay-1, az, 0, 0, 0] - eta_over_p * V[m+1, ax, ay-1, az, 0, 0, 0])
#                     V[m, ax, ay1, az, 0, 0, 0] = v

#                 # Increment z
#                 az1 = az + 1
#                 for m in range(m_top + 1):
#                     v = X_PA[2] * V[m, ax, ay, az, 0, 0, 0] + X_WP[2] * V[m+1, ax, ay, az, 0, 0, 0]
#                     if az > 0:
#                         v += az * inv_2p * (V[m, ax, ay, az-1, 0, 0, 0] - eta_over_p * V[m+1, ax, ay, az-1, 0, 0, 0])
#                     V[m, ax, ay, az1, 0, 0, 0] = v


# @njit(cache=True, fastmath=True, nogil=True, error_model="numpy", inline='always')
# def os_ket_vrr_opt(V, X_QC, X_WQ, q, eta, L_bra, L_ket, m_total):
#     inv_2q = 0.5 / q
#     eta_over_q = eta / q
#     inv_2pq = eta / (2.0 * q)  
#     p = eta * q / (q - eta)
#     inv_2pq = 0.5 / (p + q)

#     for L_f in range(L_ket):
#         for L_e in range(L_bra + 1):
#             m_top = m_total - L_e - L_f - 1

#             for ex in range(L_e + 1):
#                 for ey in range(L_e + 1 - ex):
#                     ez = L_e - ex - ey

#                     for fx in range(L_f + 1):
#                         for fy in range(L_f + 1 - fx):
#                             fz = L_f - fx - fy

#                             # Increment fx
#                             fx1 = fx + 1
#                             for m in range(m_top + 1):
#                                 v = X_QC[0] * V[m, ex, ey, ez, fx, fy, fz] + X_WQ[0] * V[m+1, ex, ey, ez, fx, fy, fz]
#                                 if fx > 0:
#                                     v += fx * inv_2q * (V[m, ex, ey, ez, fx-1, fy, fz] - eta_over_q * V[m+1, ex, ey, ez, fx-1, fy, fz])
#                                 if ex > 0:
#                                     v += ex * inv_2pq * V[m+1, ex-1, ey, ez, fx, fy, fz]
#                                 V[m, ex, ey, ez, fx1, fy, fz] = v

#                             # Increment fy
#                             fy1 = fy + 1
#                             for m in range(m_top + 1):
#                                 v = X_QC[1] * V[m, ex, ey, ez, fx, fy, fz] + X_WQ[1] * V[m+1, ex, ey, ez, fx, fy, fz]
#                                 if fy > 0:
#                                     v += fy * inv_2q * (V[m, ex, ey, ez, fx, fy-1, fz] - eta_over_q * V[m+1, ex, ey, ez, fx, fy-1, fz])
#                                 if ey > 0:
#                                     v += ey * inv_2pq * V[m+1, ex, ey-1, ez, fx, fy, fz]
#                                 V[m, ex, ey, ez, fx, fy1, fz] = v

#                             # Increment fz
#                             fz1 = fz + 1
#                             for m in range(m_top + 1):
#                                 v = X_QC[2] * V[m, ex, ey, ez, fx, fy, fz] + X_WQ[2] * V[m+1, ex, ey, ez, fx, fy, fz]
#                                 if fz > 0:
#                                     v += fz * inv_2q * (V[m, ex, ey, ez, fx, fy, fz-1] - eta_over_q * V[m+1, ex, ey, ez, fx, fy, fz-1])
#                                 if ez > 0:
#                                     v += ez * inv_2pq * V[m+1, ex, ey, ez-1, fx, fy, fz]
#                                 V[m, ex, ey, ez, fx, fy, fz1] = v



