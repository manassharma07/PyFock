import numpy as np
from numba import njit, prange, get_num_threads, get_thread_id
from .integral_helpers import Fboys
from .integral_helpers import comb
from .schwarz_helpers import eri_4c2e_diag


def os_coulomb_matrix(basis, density_matrix, schwarz_shell_pair=None,
                      threshold_schwarz=1e-9, threshold_density=1e-10,
                      fock_exchange=False):
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

    shell_lmn = np.zeros((nshells, max_nbf_shell, 3), dtype=np.int32)
    for ish in range(nshells):
        bf0 = shell_bfs_offset[ish]
        for ibf in range(bfs_nbfshell[ish]):
            shell_lmn[ish, ibf, 0] = bfs_lmn[bf0 + ibf, 0]
            shell_lmn[ish, ibf, 1] = bfs_lmn[bf0 + ibf, 1]
            shell_lmn[ish, ibf, 2] = bfs_lmn[bf0 + ibf, 2]

    cont_coeffs = np.zeros((nbf, maxnprim), dtype=np.float64)
    for i in range(nbf):
        Ni = bfs_contr_prim_norms[i]
        for j in range(bfs_nprim[i]):
            cont_coeffs[i, j] = Ni * bfs_coeffs[i, j] * bfs_prim_norms[i, j]

    # Precompute shell-level contraction coefficients
    shell_cont_coeffs = np.zeros((nshells, max_nbf_shell, maxnprim), dtype=np.float64)
    shell_nprim = np.zeros(nshells, dtype=np.int32)
    for ish in range(nshells):
        bf0 = shell_bfs_offset[ish]
        shell_nprim[ish] = bfs_nprim[bf0]
        for ibf in range(bfs_nbfshell[ish]):
            for ip in range(bfs_nprim[bf0]):
                shell_cont_coeffs[ish, ibf, ip] = cont_coeffs[bf0 + ibf, ip]

    # Precompute shell exponents
    shell_expnts = np.zeros((nshells, maxnprim), dtype=np.float64)
    for ish in range(nshells):
        bf0 = shell_bfs_offset[ish]
        for ip in range(bfs_nprim[bf0]):
            shell_expnts[ish, ip] = bfs_expnts[bf0, ip]

    # Precompute all HRR coefficients for all shell pairs
    n_ab = nshells * (nshells + 1) // 2
    max_hrr = (max_L + 1) ** 3
    
    all_hrr_coeffs, all_hrr_ax, all_hrr_ay, all_hrr_az, all_hrr_n = _precompute_all_hrr(
        nshells, n_ab, max_nbf_shell, max_hrr, shell_L, shell_centers,
        shell_bfs_offset, bfs_nbfshell, shell_lmn
    )

    result = _os_coulomb_internal(
        nbf, nshells, nthreads, max_L, max_nprim, max_nbf_shell,
        shell_L, shell_centers, shell_bfs_offset, bfs_nbfshell,
        shell_lmn, shell_nprim, shell_expnts, shell_cont_coeffs,
        density_matrix, schwarz_shell_pair,
        threshold_schwarz, threshold_density,
        fock_exchange,
        all_hrr_coeffs, all_hrr_ax, all_hrr_ay, all_hrr_az, all_hrr_n
    )

    if fock_exchange:
        return result[0], result[1]
    else:
        return result[0]


@njit(cache=True, fastmath=True, nogil=True)
def _precompute_all_hrr(nshells, n_ab, max_nbf_shell, max_hrr, shell_L, shell_centers,
                        shell_bfs_offset, bfs_nbfshell, shell_lmn):
    """Precompute HRR coefficients for all shell pairs."""
    all_hrr_coeffs = np.zeros((n_ab, max_nbf_shell, max_nbf_shell, max_hrr), dtype=np.float64)
    all_hrr_ax = np.zeros((n_ab, max_nbf_shell, max_nbf_shell, max_hrr), dtype=np.int32)
    all_hrr_ay = np.zeros((n_ab, max_nbf_shell, max_nbf_shell, max_hrr), dtype=np.int32)
    all_hrr_az = np.zeros((n_ab, max_nbf_shell, max_nbf_shell, max_hrr), dtype=np.int32)
    all_hrr_n = np.zeros((n_ab, max_nbf_shell, max_nbf_shell), dtype=np.int32)
    
    idx = 0
    for ish_a in range(nshells):
        for ish_b in range(ish_a + 1):
            # Compute AB vector
            X_AB_0 = shell_centers[ish_a, 0] - shell_centers[ish_b, 0]
            X_AB_1 = shell_centers[ish_a, 1] - shell_centers[ish_b, 1]
            X_AB_2 = shell_centers[ish_a, 2] - shell_centers[ish_b, 2]
            
            nbf_a = bfs_nbfshell[ish_a]
            nbf_b = bfs_nbfshell[ish_b]
            
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
                        pow_x = X_AB_0 ** (bx - px)
                        ax_f = ax + px
                        for py in range(by + 1):
                            binom_y = comb(by, py)
                            pow_y = X_AB_1 ** (by - py)
                            ay_f = ay + py
                            for pz in range(bz + 1):
                                binom_z = comb(bz, pz)
                                pow_z = X_AB_2 ** (bz - pz)
                                az_f = az + pz
                                coeff = binom_x * pow_x * binom_y * pow_y * binom_z * pow_z
                                all_hrr_coeffs[idx, ic_a, ic_b, count] = coeff
                                all_hrr_ax[idx, ic_a, ic_b, count] = ax_f
                                all_hrr_ay[idx, ic_a, ic_b, count] = ay_f
                                all_hrr_az[idx, ic_a, ic_b, count] = az_f
                                count += 1
                    all_hrr_n[idx, ic_a, ic_b] = count
            idx += 1
    
    return all_hrr_coeffs, all_hrr_ax, all_hrr_ay, all_hrr_az, all_hrr_n


@njit(cache=True, fastmath=True, nogil=True, error_model="numpy", inline='always')
def _compute_vrr_ssss(V_flat, s0, s1, s2, s3, s4, s5,
                      X_PA, X_WP, X_QC, X_WQ,
                      inv_2p, eta_over_p, inv_2q, eta_over_q, inv_2pq,
                      prefactor, F, L_bra, L_ket, L_all):
    """Optimized VRR computation with precomputed strides and factors."""
    
    dim_m = L_all + 1
    
    for m in range(dim_m):
        V_flat[m * s0] = prefactor * F[m]

    # Build bra (e) indices
    for L_e in range(L_bra):
        m_top = L_all - L_e - 1
        for ax in range(L_e + 1):
            for ay in range(L_e + 1 - ax):
                az = L_e - ax - ay
                base = ax * s1 + ay * s2 + az * s3

                # x direction
                ax1_base = base + s1
                if ax > 0:
                    axm1_base = base - s1
                    fac = ax * inv_2p
                    for m in range(m_top + 1):
                        m_s0 = m * s0
                        m1_s0 = (m + 1) * s0
                        V_flat[m_s0 + ax1_base] = (
                            X_PA[0] * V_flat[m_s0 + base] +
                            X_WP[0] * V_flat[m1_s0 + base] +
                            fac * (V_flat[m_s0 + axm1_base] -
                                   eta_over_p * V_flat[m1_s0 + axm1_base]))
                else:
                    for m in range(m_top + 1):
                        m_s0 = m * s0
                        V_flat[m_s0 + ax1_base] = (
                            X_PA[0] * V_flat[m_s0 + base] +
                            X_WP[0] * V_flat[(m + 1) * s0 + base])

                # y direction
                ay1_base = base + s2
                if ay > 0:
                    aym1_base = base - s2
                    fac = ay * inv_2p
                    for m in range(m_top + 1):
                        m_s0 = m * s0
                        m1_s0 = (m + 1) * s0
                        V_flat[m_s0 + ay1_base] = (
                            X_PA[1] * V_flat[m_s0 + base] +
                            X_WP[1] * V_flat[m1_s0 + base] +
                            fac * (V_flat[m_s0 + aym1_base] -
                                   eta_over_p * V_flat[m1_s0 + aym1_base]))
                else:
                    for m in range(m_top + 1):
                        m_s0 = m * s0
                        V_flat[m_s0 + ay1_base] = (
                            X_PA[1] * V_flat[m_s0 + base] +
                            X_WP[1] * V_flat[(m + 1) * s0 + base])

                # z direction
                az1_base = base + s3
                if az > 0:
                    azm1_base = base - s3
                    fac = az * inv_2p
                    for m in range(m_top + 1):
                        m_s0 = m * s0
                        m1_s0 = (m + 1) * s0
                        V_flat[m_s0 + az1_base] = (
                            X_PA[2] * V_flat[m_s0 + base] +
                            X_WP[2] * V_flat[m1_s0 + base] +
                            fac * (V_flat[m_s0 + azm1_base] -
                                   eta_over_p * V_flat[m1_s0 + azm1_base]))
                else:
                    for m in range(m_top + 1):
                        m_s0 = m * s0
                        V_flat[m_s0 + az1_base] = (
                            X_PA[2] * V_flat[m_s0 + base] +
                            X_WP[2] * V_flat[(m + 1) * s0 + base])

    # Build ket (f) indices
    if L_ket > 0:
        for L_f in range(L_ket):
            for L_e in range(L_bra + 1):
                m_top = L_all - L_e - L_f - 1
                for ex in range(L_e + 1):
                    for ey in range(L_e + 1 - ex):
                        ez = L_e - ex - ey
                        e_base = ex * s1 + ey * s2 + ez * s3
                        for fx in range(L_f + 1):
                            for fy in range(L_f + 1 - fx):
                                fz = L_f - fx - fy
                                f_base = fx * s4 + fy * s5 + fz
                                base = e_base + f_base

                                # x direction
                                fx1_base = base + s4
                                has_fx = fx > 0
                                has_ex = ex > 0
                                
                                for m in range(m_top + 1):
                                    m_s0 = m * s0
                                    m1_s0 = (m + 1) * s0
                                    v = X_QC[0] * V_flat[m_s0 + base] + X_WQ[0] * V_flat[m1_s0 + base]
                                    if has_fx:
                                        fxm1_base = base - s4
                                        v += (fx * inv_2q) * (V_flat[m_s0 + fxm1_base] - eta_over_q * V_flat[m1_s0 + fxm1_base])
                                    if has_ex:
                                        exm1_base = base - s1
                                        v += (ex * inv_2pq) * V_flat[m1_s0 + exm1_base]
                                    V_flat[m_s0 + fx1_base] = v

                                # y direction
                                fy1_base = base + s5
                                has_fy = fy > 0
                                has_ey = ey > 0
                                
                                for m in range(m_top + 1):
                                    m_s0 = m * s0
                                    m1_s0 = (m + 1) * s0
                                    v = X_QC[1] * V_flat[m_s0 + base] + X_WQ[1] * V_flat[m1_s0 + base]
                                    if has_fy:
                                        fym1_base = base - s5
                                        v += (fy * inv_2q) * (V_flat[m_s0 + fym1_base] - eta_over_q * V_flat[m1_s0 + fym1_base])
                                    if has_ey:
                                        eym1_base = base - s2
                                        v += (ey * inv_2pq) * V_flat[m1_s0 + eym1_base]
                                    V_flat[m_s0 + fy1_base] = v

                                # z direction
                                fz1_base = base + 1
                                has_fz = fz > 0
                                has_ez = ez > 0
                                
                                for m in range(m_top + 1):
                                    m_s0 = m * s0
                                    m1_s0 = (m + 1) * s0
                                    v = X_QC[2] * V_flat[m_s0 + base] + X_WQ[2] * V_flat[m1_s0 + base]
                                    if has_fz:
                                        fzm1_base = base - 1
                                        v += (fz * inv_2q) * (V_flat[m_s0 + fzm1_base] - eta_over_q * V_flat[m1_s0 + fzm1_base])
                                    if has_ez:
                                        ezm1_base = base - s3
                                        v += (ez * inv_2pq) * V_flat[m1_s0 + ezm1_base]
                                    V_flat[m_s0 + fz1_base] = v


@njit(cache=True, fastmath=True, nogil=True, inline='always')
def _contract_eri_shell(ERI_shell, V_flat, s1, s2, s3, s4, s5,
                        nbf_a, nbf_b, nbf_c, nbf_d,
                        bra_hrr_coeffs, bra_hrr_ax, bra_hrr_ay, bra_hrr_az, bra_hrr_n,
                        ket_hrr_coeffs, ket_hrr_ax, ket_hrr_ay, ket_hrr_az, ket_hrr_n,
                        c_a_arr, c_b_arr, c_c_arr, c_d_arr):
    """Contract primitives into shell ERIs."""
    for ic_a in range(nbf_a):
        c_a = c_a_arr[ic_a]
        for ic_b in range(nbf_b):
            c_ab = c_a * c_b_arr[ic_b]
            n_bra = bra_hrr_n[ic_a, ic_b]
            
            for ic_c in range(nbf_c):
                c_abc = c_ab * c_c_arr[ic_c]
                
                for ic_d in range(nbf_d):
                    c_abcd = c_abc * c_d_arr[ic_d]
                    n_ket = ket_hrr_n[ic_c, ic_d]
                    
                    hrr_val = 0.0
                    for ib in range(n_bra):
                        bc = bra_hrr_coeffs[ic_a, ic_b, ib]
                        bra_offset = (bra_hrr_ax[ic_a, ic_b, ib] * s1 + 
                                     bra_hrr_ay[ic_a, ic_b, ib] * s2 + 
                                     bra_hrr_az[ic_a, ic_b, ib] * s3)
                        for ik in range(n_ket):
                            kc = ket_hrr_coeffs[ic_c, ic_d, ik]
                            ket_offset = (ket_hrr_cx[ic_c, ic_d, ik] * s4 + 
                                         ket_hrr_cy[ic_c, ic_d, ik] * s5 + 
                                         ket_hrr_cz[ic_c, ic_d, ik])
                            hrr_val += bc * kc * V_flat[bra_offset + ket_offset]
                    
                    ERI_shell[ic_a, ic_b, ic_c, ic_d] += c_abcd * hrr_val


@njit(cache=True, fastmath=True, nogil=True, inline='always')
def _accumulate_J_nondiag(J_local, ERI_shell, density_matrix,
                          bf_a_start, bf_b_start, bf_c_start, bf_d_start,
                          nbf_a, nbf_b, nbf_c, nbf_d,
                          same_ab, same_cd):
    """Accumulate Coulomb contributions for non-diagonal shell quartets."""
    # J[ab] += D[cd] * (ab|cd)
    for ic_a in range(nbf_a):
        ibf_a = bf_a_start + ic_a
        for ic_b in range(nbf_b):
            ibf_b = bf_b_start + ic_b
            j_ab_val = 0.0
            for ic_c in range(nbf_c):
                ibf_c = bf_c_start + ic_c
                for ic_d in range(nbf_d):
                    ibf_d = bf_d_start + ic_d
                    val = ERI_shell[ic_a, ic_b, ic_c, ic_d]
                    if abs(val) > 1e-15:
                        if same_cd:
                            D_cd = density_matrix[ibf_c, ibf_d]
                        else:
                            D_cd = density_matrix[ibf_c, ibf_d] + density_matrix[ibf_d, ibf_c]
                        j_ab_val += D_cd * val
            if abs(j_ab_val) > 1e-15:
                J_local[ibf_a, ibf_b] += j_ab_val
                if not same_ab:
                    J_local[ibf_b, ibf_a] += j_ab_val

    # J[cd] += D[ab] * (ab|cd) (bra-ket swap)
    for ic_c in range(nbf_c):
        ibf_c = bf_c_start + ic_c
        for ic_d in range(nbf_d):
            ibf_d = bf_d_start + ic_d
            j_cd_val = 0.0
            for ic_a in range(nbf_a):
                ibf_a = bf_a_start + ic_a
                for ic_b in range(nbf_b):
                    ibf_b = bf_b_start + ic_b
                    val = ERI_shell[ic_a, ic_b, ic_c, ic_d]
                    if abs(val) > 1e-15:
                        if same_ab:
                            D_ab = density_matrix[ibf_a, ibf_b]
                        else:
                            D_ab = density_matrix[ibf_a, ibf_b] + density_matrix[ibf_b, ibf_a]
                        j_cd_val += D_ab * val
            if abs(j_cd_val) > 1e-15:
                J_local[ibf_c, ibf_d] += j_cd_val
                if not same_cd:
                    J_local[ibf_d, ibf_c] += j_cd_val


@njit(cache=True, fastmath=True, nogil=True, inline='always')
def _accumulate_K_nondiag(K_local, ERI_shell, density_matrix,
                          bf_a_start, bf_b_start, bf_c_start, bf_d_start,
                          nbf_a, nbf_b, nbf_c, nbf_d,
                          same_ab, same_cd):
    """Accumulate exchange contributions for non-diagonal shell quartets."""
    for ic_a in range(nbf_a):
        ibf_a = bf_a_start + ic_a
        for ic_b in range(nbf_b):
            ibf_b = bf_b_start + ic_b
            for ic_c in range(nbf_c):
                ibf_c = bf_c_start + ic_c
                for ic_d in range(nbf_d):
                    ibf_d = bf_d_start + ic_d
                    val = ERI_shell[ic_a, ic_b, ic_c, ic_d]
                    if abs(val) < 1e-15:
                        continue

                    # K[a,c] += D[b,d] * val and K[c,a] += D[d,b] * val
                    K_local[ibf_a, ibf_c] += density_matrix[ibf_b, ibf_d] * val
                    K_local[ibf_c, ibf_a] += density_matrix[ibf_d, ibf_b] * val

                    if not same_cd:
                        # K[a,d] += D[b,c] * val and K[d,a] += D[c,b] * val
                        K_local[ibf_a, ibf_d] += density_matrix[ibf_b, ibf_c] * val
                        K_local[ibf_d, ibf_a] += density_matrix[ibf_c, ibf_b] * val

                    if not same_ab:
                        # K[b,c] += D[a,d] * val and K[c,b] += D[d,a] * val
                        K_local[ibf_b, ibf_c] += density_matrix[ibf_a, ibf_d] * val
                        K_local[ibf_c, ibf_b] += density_matrix[ibf_d, ibf_a] * val

                    if not same_ab and not same_cd:
                        # K[b,d] += D[a,c] * val and K[d,b] += D[c,a] * val
                        K_local[ibf_b, ibf_d] += density_matrix[ibf_a, ibf_c] * val
                        K_local[ibf_d, ibf_b] += density_matrix[ibf_c, ibf_a] * val


@njit(parallel=True, cache=False, fastmath=True, nogil=True, error_model="numpy")
def _os_coulomb_internal(nbf, nshells, nthreads, max_L, max_nprim, max_nbf_shell,
                         shell_L, shell_centers, shell_bfs_offset, bfs_nbfshell,
                         shell_lmn, shell_nprim, shell_expnts, shell_cont_coeffs,
                         density_matrix, schwarz_shell_pair,
                         threshold_schwarz, threshold_density,
                         do_exchange,
                         all_hrr_coeffs, all_hrr_ax, all_hrr_ay, all_hrr_az, all_hrr_n):

    pi = np.pi
    two_pi_52 = 2.0 * pi ** 2.5

    n_ab = nshells * (nshells + 1) // 2

    # Precompute shell pair data
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

    # Precompute max density for shell pairs
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

    J_threads = np.zeros((nthreads, nbf, nbf), dtype=np.float64)

    if do_exchange:
        K_threads = np.zeros((nthreads, nbf, nbf), dtype=np.float64)
    else:
        K_threads = np.zeros((1, 1, 1), dtype=np.float64)

    V_flat_size = max_dim_m * max_dim_bra * max_dim_bra * max_dim_bra * max_dim_ket * max_dim_ket * max_dim_ket

    for ab_idx in prange(n_ab):
        schwarz_ab = ab_schwarz_bound[ab_idx]
        if schwarz_ab < threshold_schwarz:
            continue

        tid = get_thread_id()
        J_local = J_threads[tid]
        if do_exchange:
            K_local = K_threads[tid]

        ish_a = ab_shell_a[ab_idx]
        ish_b = ab_shell_b[ab_idx]
        same_ab = (ish_a == ish_b)

        L_a = shell_L[ish_a]
        L_b = shell_L[ish_b]
        L_bra = L_a + L_b
        bf_a_start = shell_bfs_offset[ish_a]
        bf_b_start = shell_bfs_offset[ish_b]
        nbf_a = bfs_nbfshell[ish_a]
        nbf_b = bfs_nbfshell[ish_b]

        AB_sq = AB_sqs[ab_idx]
        center_a = shell_centers[ish_a]
        center_b = shell_centers[ish_b]

        nprimi = shell_nprim[ish_a]
        nprimj = shell_nprim[ish_b]

        # Thread-local workspace arrays
        bra_p = np.empty(max_prim_pairs, dtype=np.float64)
        bra_K = np.empty(max_prim_pairs, dtype=np.float64)
        bra_Px = np.empty(max_prim_pairs, dtype=np.float64)
        bra_Py = np.empty(max_prim_pairs, dtype=np.float64)
        bra_Pz = np.empty(max_prim_pairs, dtype=np.float64)
        bra_PAx = np.empty(max_prim_pairs, dtype=np.float64)
        bra_PAy = np.empty(max_prim_pairs, dtype=np.float64)
        bra_PAz = np.empty(max_prim_pairs, dtype=np.float64)
        bra_inv_2p = np.empty(max_prim_pairs, dtype=np.float64)
        bra_ipa = np.empty(max_prim_pairs, dtype=np.int32)
        bra_ipb = np.empty(max_prim_pairs, dtype=np.int32)

        V_flat = np.zeros(V_flat_size, dtype=np.float64)
        F = np.empty(max_dim_m, dtype=np.float64)
        X_PA = np.empty(3, dtype=np.float64)
        X_QC = np.empty(3, dtype=np.float64)
        X_WP = np.empty(3, dtype=np.float64)
        X_WQ = np.empty(3, dtype=np.float64)

        ERI_shell = np.zeros((max_nbf_shell, max_nbf_shell,
                              max_nbf_shell, max_nbf_shell), dtype=np.float64)

        # Precompute bra primitive pairs
        n_bra_pairs = 0
        for ipa in range(nprimi):
            alpha = shell_expnts[ish_a, ipa]
            for ipb in range(nprimj):
                beta = shell_expnts[ish_b, ipb]
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
                bra_inv_2p[ii] = 0.5 / p

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

        # Get precomputed bra HRR coefficients
        bra_hrr_coeffs = all_hrr_coeffs[ab_idx]
        bra_hrr_ax = all_hrr_ax[ab_idx]
        bra_hrr_ay = all_hrr_ay[ab_idx]
        bra_hrr_az = all_hrr_az[ab_idx]
        bra_hrr_n = all_hrr_n[ab_idx]

        for cd_idx in range(ab_idx + 1):
            ish_c = ab_shell_a[cd_idx]
            ish_d = ab_shell_b[cd_idx]
            same_cd = (ish_c == ish_d)

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

            CD_sq = AB_sqs[cd_idx]

            L_all = L_bra + L_ket
            dim_bra = L_bra + 1
            dim_ket = L_ket + 1
            dim_m = L_all + 1

            nprimk = shell_nprim[ish_c]
            npriml = shell_nprim[ish_d]

            is_diag = (ab_idx == cd_idx)

            # Reset ERI_shell
            for ic_a in range(nbf_a):
                for ic_b in range(nbf_b):
                    for ic_c in range(nbf_c):
                        for ic_d in range(nbf_d):
                            ERI_shell[ic_a, ic_b, ic_c, ic_d] = 0.0

            # Get precomputed ket HRR coefficients
            ket_hrr_coeffs = all_hrr_coeffs[cd_idx]
            ket_hrr_cx = all_hrr_ax[cd_idx]
            ket_hrr_cy = all_hrr_ay[cd_idx]
            ket_hrr_cz = all_hrr_az[cd_idx]
            ket_hrr_n = all_hrr_n[cd_idx]

            # Precompute strides
            s6 = 1
            s5 = dim_ket
            s4 = dim_ket * dim_ket
            s3 = dim_ket * dim_ket * dim_ket
            s2 = dim_bra * s3
            s1 = dim_bra * s2
            s0 = dim_bra * s1

            for ibra in range(n_bra_pairs):
                p = bra_p[ibra]
                K_AB_val = bra_K[ibra]
                ipa = bra_ipa[ibra]
                ipb = bra_ipb[ibra]
                inv_2p = bra_inv_2p[ibra]

                X_PA[0] = bra_PAx[ibra]
                X_PA[1] = bra_PAy[ibra]
                X_PA[2] = bra_PAz[ibra]
                Px = bra_Px[ibra]
                Py = bra_Py[ibra]
                Pz = bra_Pz[ibra]

                for ipc in range(nprimk):
                    gamma = shell_expnts[ish_c, ipc]
                    for ipd in range(npriml):
                        delta = shell_expnts[ish_d, ipd]
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

                        # Compute Boys function values
                        for m in range(dim_m):
                            F[m] = Fboys(m, T_arg)

                        # Precompute VRR factors
                        inv_2q = 0.5 / q
                        eta_over_p = eta / p
                        eta_over_q = eta / q
                        inv_2pq = 0.5 / pq

                        _compute_vrr_ssss(V_flat, s0, s1, s2, s3, s4, s5,
                                          X_PA, X_WP, X_QC, X_WQ,
                                          inv_2p, eta_over_p, inv_2q, eta_over_q, inv_2pq,
                                          prefactor, F, L_bra, L_ket, L_all)

                        # Contract into ERI_shell
                        for ic_a in range(nbf_a):
                            c_a = shell_cont_coeffs[ish_a, ic_a, ipa]

                            for ic_b in range(nbf_b):
                                c_ab = c_a * shell_cont_coeffs[ish_b, ic_b, ipb]
                                n_bra_h = bra_hrr_n[ic_a, ic_b]

                                for ic_c in range(nbf_c):
                                    c_abc = c_ab * shell_cont_coeffs[ish_c, ic_c, ipc]

                                    for ic_d in range(nbf_d):
                                        c_abcd = c_abc * shell_cont_coeffs[ish_d, ic_d, ipd]
                                        n_ket_h = ket_hrr_n[ic_c, ic_d]

                                        hrr_val = 0.0
                                        for ib in range(n_bra_h):
                                            bc = bra_hrr_coeffs[ic_a, ic_b, ib]
                                            bra_offset = (bra_hrr_ax[ic_a, ic_b, ib] * s1 + 
                                                         bra_hrr_ay[ic_a, ic_b, ib] * s2 + 
                                                         bra_hrr_az[ic_a, ic_b, ib] * s3)
                                            for ik in range(n_ket_h):
                                                kc = ket_hrr_coeffs[ic_c, ic_d, ik]
                                                ket_offset = (ket_hrr_cx[ic_c, ic_d, ik] * s4 + 
                                                             ket_hrr_cy[ic_c, ic_d, ik] * s5 + 
                                                             ket_hrr_cz[ic_c, ic_d, ik])
                                                hrr_val += bc * kc * V_flat[bra_offset + ket_offset]

                                        ERI_shell[ic_a, ic_b, ic_c, ic_d] += c_abcd * hrr_val

            # Accumulate into J (and optionally K)
            if not is_diag:
                _accumulate_J_nondiag(J_local, ERI_shell, density_matrix,
                                      bf_a_start, bf_b_start, bf_c_start, bf_d_start,
                                      nbf_a, nbf_b, nbf_c, nbf_d,
                                      same_ab, same_cd)
                if do_exchange:
                    _accumulate_K_nondiag(K_local, ERI_shell, density_matrix,
                                          bf_a_start, bf_b_start, bf_c_start, bf_d_start,
                                          nbf_a, nbf_b, nbf_c, nbf_d,
                                          same_ab, same_cd)
            else:
                # Diagonal case
                for ic_a in range(nbf_a):
                    ibf_a = bf_a_start + ic_a
                    for ic_b in range(nbf_b):
                        ibf_b = bf_b_start + ic_b
                        j_ab_val = 0.0
                        for ic_c in range(nbf_c):
                            ibf_c = bf_c_start + ic_c
                            for ic_d in range(nbf_d):
                                ibf_d = bf_d_start + ic_d
                                val = ERI_shell[ic_a, ic_b, ic_c, ic_d]
                                if abs(val) > 1e-15:
                                    if same_cd:
                                        D_cd = density_matrix[ibf_c, ibf_d]
                                    else:
                                        D_cd = density_matrix[ibf_c, ibf_d] + density_matrix[ibf_d, ibf_c]
                                    j_ab_val += D_cd * val
                        if abs(j_ab_val) > 1e-15:
                            J_local[ibf_a, ibf_b] += j_ab_val
                            if not same_ab:
                                J_local[ibf_b, ibf_a] += j_ab_val

                if do_exchange:
                    for ic_a in range(nbf_a):
                        ibf_a = bf_a_start + ic_a
                        for ic_b in range(nbf_b):
                            ibf_b = bf_b_start + ic_b
                            for ic_c in range(nbf_c):
                                ibf_c = bf_c_start + ic_c
                                for ic_d in range(nbf_d):
                                    ibf_d = bf_d_start + ic_d
                                    val = ERI_shell[ic_a, ic_b, ic_c, ic_d]
                                    if abs(val) < 1e-15:
                                        continue

                                    K_local[ibf_a, ibf_c] += density_matrix[ibf_b, ibf_d] * val

                                    if not same_ab:
                                        K_local[ibf_a, ibf_d] += density_matrix[ibf_b, ibf_c] * val
                                        K_local[ibf_b, ibf_c] += density_matrix[ibf_a, ibf_d] * val
                                        K_local[ibf_b, ibf_d] += density_matrix[ibf_a, ibf_c] * val

    # Reduce per-thread J buffers
    J_matrix = np.zeros((nbf, nbf), dtype=np.float64)
    for t in range(nthreads):
        for i in prange(nbf):
            for j in range(nbf):
                J_matrix[i, j] += J_threads[t, i, j]

    if do_exchange:
        K_matrix = np.zeros((nbf, nbf), dtype=np.float64)
        for t in range(nthreads):
            for i in prange(nbf):
                for j in range(nbf):
                    K_matrix[i, j] += K_threads[t, i, j]
        return J_matrix, K_matrix
    else:
        K_matrix = np.zeros((1, 1), dtype=np.float64)
        return J_matrix, K_matrix