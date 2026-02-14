import numpy as np
from numba import njit, prange
from .integral_helpers import Fboys

def os_4c2e_symm_old(basis):
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
    
    # Convert basis data to numpy arrays for Numba
    nbf = basis.bfs_nao
    nshells = len(basis.shells)
    
    # Shell properties
    shell_L = np.array([basis.bfs_lm[i] for i in basis.shell_bfs_offset], dtype=np.int32)
    shell_centers = np.array([basis.bfs_coords[i] for i in basis.shell_bfs_offset], dtype=np.float64)
    shell_bfs_offset = np.array(basis.shell_bfs_offset, dtype=np.int32)
    bfs_nbfshell = np.array(basis.bfs_nbfshell, dtype=np.int32)
    
    # Basis function properties
    bfs_coords = np.array(basis.bfs_coords, dtype=np.float64)
    bfs_contr_prim_norms = np.array(basis.bfs_contr_prim_norms, dtype=np.float64)
    bfs_lmn = np.array(basis.bfs_lmn, dtype=np.int32)
    bfs_nprim = np.array(basis.bfs_nprim, dtype=np.int32)
    bfs_shell_index = np.array(basis.bfs_shell_index, dtype=np.int32)
    
    # Primitive data - convert to 2D arrays
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
    
    # Maximum angular momentum for workspace allocation
    max_L = np.max(shell_L)
    L_max_pair = 2 * max_L
    L_max_total = 4 * max_L
    m_max_total = L_max_total
    
    # Shell quartet loops with symmetry
    for ish_a in prange(nshells):
        shell_a_center = shell_centers[ish_a]
        L_a = shell_L[ish_a]
        bf_a_start = shell_bfs_offset[ish_a]
        nbf_a = bfs_nbfshell[ish_a]
        
        for ish_b in range(ish_a + 1):  # Symmetry: a >= b
            shell_b_center = shell_centers[ish_b]
            L_b = shell_L[ish_b]
            bf_b_start = shell_bfs_offset[ish_b]
            nbf_b = bfs_nbfshell[ish_b]
            L_bra = L_a + L_b
            
            X_AB = shell_a_center - shell_b_center
            
            # Compute triangular index for (a,b)
            if ish_a < ish_b:
                triangle2_ab = ish_b * (ish_b + 1) // 2 + ish_a
            else:
                triangle2_ab = ish_a * (ish_a + 1) // 2 + ish_b
            
            for ish_c in range(nshells):
                shell_c_center = shell_centers[ish_c]
                L_c = shell_L[ish_c]
                bf_c_start = shell_bfs_offset[ish_c]
                nbf_c = bfs_nbfshell[ish_c]
                
                for ish_d in range(ish_c + 1):  # Symmetry: c >= d
                    shell_d_center = shell_centers[ish_d]
                    L_d = shell_L[ish_d]
                    bf_d_start = shell_bfs_offset[ish_d]
                    nbf_d = bfs_nbfshell[ish_d]
                    L_ket = L_c + L_d
                    
                    # Compute triangular index for (c,d)
                    if ish_c < ish_d:
                        triangle2_cd = ish_d * (ish_d + 1) // 2 + ish_c
                    else:
                        triangle2_cd = ish_c * (ish_c + 1) // 2 + ish_d
                    
                    # Skip if (ab) > (cd) by symmetry
                    if triangle2_ab > triangle2_cd:
                        continue
                    
                    X_CD = shell_c_center - shell_d_center
                    L_all = L_bra + L_ket
                    
                    # Allocate workspace for this shell quartet
                    V_vrr = np.zeros((m_max_total + 1, L_max_pair + 1, L_max_pair + 1, 
                                     L_max_pair + 1, L_max_pair + 1, L_max_pair + 1, 
                                     L_max_pair + 1), dtype=np.float64)
                    
                    # Temporary array for this shell quartet
                    ERI_shell = np.zeros((nbf_a, nbf_b, nbf_c, nbf_d), dtype=np.float64)
                    
                    # Loop over basis functions in shell A
                    for ic_a in range(nbf_a):
                        ibf_a = bf_a_start + ic_a
                        ax, ay, az = bfs_lmn[ibf_a]
                        Ni = bfs_contr_prim_norms[ibf_a]
                        nprimi = bfs_nprim[ibf_a]
                        
                        for ic_b in range(nbf_b):
                            ibf_b = bf_b_start + ic_b
                            bx, by, bz = bfs_lmn[ibf_b]
                            Nj = bfs_contr_prim_norms[ibf_b]
                            nprimj = bfs_nprim[ibf_b]
                            tempcoeff1 = Ni * Nj
                            
                            for ic_c in range(nbf_c):
                                ibf_c = bf_c_start + ic_c
                                cx, cy, cz = bfs_lmn[ibf_c]
                                Nk = bfs_contr_prim_norms[ibf_c]
                                nprimk = bfs_nprim[ibf_c]
                                tempcoeff2 = tempcoeff1 * Nk
                                
                                for ic_d in range(nbf_d):
                                    ibf_d = bf_d_start + ic_d
                                    dx, dy, dz = bfs_lmn[ibf_d]
                                    Nl = bfs_contr_prim_norms[ibf_d]
                                    npriml = bfs_nprim[ibf_d]
                                    tempcoeff3 = tempcoeff2 * Nl
                                    
                                    val = 0.0
                                    
                                    # Primitive loops
                                    for ipa in range(nprimi):
                                        alpha = bfs_expnts[ibf_a, ipa]
                                        da = bfs_coeffs[ibf_a, ipa]
                                        Nik = bfs_prim_norms[ibf_a, ipa]
                                        tempcoeff4 = tempcoeff3 * da * Nik
                                        
                                        for ipb in range(nprimj):
                                            beta = bfs_expnts[ibf_b, ipb]
                                            db = bfs_coeffs[ibf_b, ipb]
                                            Njk = bfs_prim_norms[ibf_b, ipb]
                                            
                                            # Gaussian product theorem for bra
                                            p = alpha + beta
                                            mu_ab = alpha * beta / p
                                            AB_sq = np.sum(X_AB**2)
                                            K_AB = np.exp(-mu_ab * AB_sq)
                                            
                                            # Screening
                                            if abs(K_AB) < 1.0e-8:
                                                continue
                                            
                                            P_center = (alpha * shell_a_center + beta * shell_b_center) / p
                                            X_PA = P_center - shell_a_center
                                            
                                            tempcoeff5 = tempcoeff4 * db * Njk
                                            
                                            for ipc in range(nprimk):
                                                gamma = bfs_expnts[ibf_c, ipc]
                                                dc = bfs_coeffs[ibf_c, ipc]
                                                Nkk = bfs_prim_norms[ibf_c, ipc]
                                                tempcoeff6 = tempcoeff5 * dc * Nkk
                                                
                                                for ipd in range(npriml):
                                                    delta = bfs_expnts[ibf_d, ipd]
                                                    dd = bfs_coeffs[ibf_d, ipd]
                                                    Nlk = bfs_prim_norms[ibf_d, ipd]
                                                    
                                                    # Gaussian product theorem for ket
                                                    q = gamma + delta
                                                    mu_cd = gamma * delta / q
                                                    CD_sq = np.sum(X_CD**2)
                                                    K_CD = np.exp(-mu_cd * CD_sq)
                                                    
                                                    # Screening
                                                    if abs(K_CD) < 1.0e-8:
                                                        continue
                                                    if abs(K_AB * K_CD) < 1.0e-10:
                                                        continue
                                                    
                                                    Q_center = (gamma * shell_c_center + delta * shell_d_center) / q
                                                    X_QC = Q_center - shell_c_center
                                                    
                                                    # Combined quantities
                                                    eta = p * q / (p + q)
                                                    PQ = P_center - Q_center
                                                    PQ_sq = np.sum(PQ**2)
                                                    T_arg = eta * PQ_sq
                                                    W = (p * P_center + q * Q_center) / (p + q)
                                                    X_WP = W - P_center
                                                    X_WQ = W - Q_center
                                                    
                                                    tempcoeff7 = tempcoeff6 * dd * Nlk
                                                    
                                                    # Base case prefactor
                                                    prefactor = 2.0 * pi**2.5 / (p * q * np.sqrt(p + q)) * K_AB * K_CD
                                                    
                                                    # Boys function values
                                                    F = np.zeros(L_all + 1, dtype=np.float64)
                                                    for m in range(L_all + 1):
                                                        F[m] = Fboys(m, T_arg)
                                                    
                                                    # Reset V_vrr for this primitive quartet
                                                    V_vrr[:, :, :, :, :, :, :] = 0.0
                                                    
                                                    # Base case: [ss|ss]^(m)
                                                    for m in range(L_all + 1):
                                                        V_vrr[m, 0, 0, 0, 0, 0, 0] = prefactor * F[m]
                                                    
                                                    # Bra VRR
                                                    os_bra_vrr(V_vrr, X_PA, X_WP, p, eta, L_bra, L_all)
                                                    
                                                    # Ket VRR
                                                    os_ket_vrr(V_vrr, X_QC, X_WQ, q, eta, L_bra, L_ket, L_all)
                                                    
                                                    # HRR to get final integral
                                                    hrr_val = os_hrr_element(V_vrr, ax, ay, az, bx, by, bz,
                                                                            cx, cy, cz, dx, dy, dz, X_AB, X_CD)
                                                    
                                                    val += tempcoeff7 * hrr_val
                                    
                                    ERI_shell[ic_a, ic_b, ic_c, ic_d] = val
                    
                    # Store results with 8-fold symmetry
                    for ic_a in range(nbf_a):
                        ibf_a = bf_a_start + ic_a
                        for ic_b in range(nbf_b):
                            ibf_b = bf_b_start + ic_b
                            for ic_c in range(nbf_c):
                                ibf_c = bf_c_start + ic_c
                                for ic_d in range(nbf_d):
                                    ibf_d = bf_d_start + ic_d
                                    
                                    val = ERI_shell[ic_a, ic_b, ic_c, ic_d]
                                    
                                    # Apply 8-fold symmetry
                                    fourC2E[ibf_a, ibf_b, ibf_c, ibf_d] = val
                                    fourC2E[ibf_b, ibf_a, ibf_c, ibf_d] = val
                                    fourC2E[ibf_a, ibf_b, ibf_d, ibf_c] = val
                                    fourC2E[ibf_b, ibf_a, ibf_d, ibf_c] = val
                                    fourC2E[ibf_c, ibf_d, ibf_a, ibf_b] = val
                                    fourC2E[ibf_c, ibf_d, ibf_b, ibf_a] = val
                                    fourC2E[ibf_d, ibf_c, ibf_a, ibf_b] = val
                                    fourC2E[ibf_d, ibf_c, ibf_b, ibf_a] = val
    
    return fourC2E


@njit(cache=True, fastmath=True, nogil=True, error_model="numpy")
def os_bra_vrr(V, X_PA, X_WP, p, eta, L_bra, m_total):
    inv_2p = 0.5 / p
    eta_over_p = eta / p
    
    for L_e in range(L_bra):
        m_top = m_total - L_e - 1
        
        for ax in range(L_e, -1, -1):
            for ay in range(L_e - ax, -1, -1):
                az = L_e - ax - ay
                
                # Increment x
                for m in range(m_top + 1):
                    V[m, ax+1, ay, az, 0, 0, 0] = (
                        X_PA[0] * V[m, ax, ay, az, 0, 0, 0] +
                        X_WP[0] * V[m+1, ax, ay, az, 0, 0, 0]
                    )
                    if ax > 0:
                        V[m, ax+1, ay, az, 0, 0, 0] += (
                            ax * inv_2p * (V[m, ax-1, ay, az, 0, 0, 0] -
                                          eta_over_p * V[m+1, ax-1, ay, az, 0, 0, 0])
                        )
                
                # Increment y
                for m in range(m_top + 1):
                    V[m, ax, ay+1, az, 0, 0, 0] = (
                        X_PA[1] * V[m, ax, ay, az, 0, 0, 0] +
                        X_WP[1] * V[m+1, ax, ay, az, 0, 0, 0]
                    )
                    if ay > 0:
                        V[m, ax, ay+1, az, 0, 0, 0] += (
                            ay * inv_2p * (V[m, ax, ay-1, az, 0, 0, 0] -
                                          eta_over_p * V[m+1, ax, ay-1, az, 0, 0, 0])
                        )
                
                # Increment z
                for m in range(m_top + 1):
                    V[m, ax, ay, az+1, 0, 0, 0] = (
                        X_PA[2] * V[m, ax, ay, az, 0, 0, 0] +
                        X_WP[2] * V[m+1, ax, ay, az, 0, 0, 0]
                    )
                    if az > 0:
                        V[m, ax, ay, az+1, 0, 0, 0] += (
                            az * inv_2p * (V[m, ax, ay, az-1, 0, 0, 0] -
                                          eta_over_p * V[m+1, ax, ay, az-1, 0, 0, 0])
                        )


@njit(cache=True, fastmath=True, nogil=True, error_model="numpy")
def os_ket_vrr(V, X_QC, X_WQ, q, eta, L_bra, L_ket, m_total):
    inv_2q = 0.5 / q
    eta_over_q = eta / q
    
    # Compute p from eta and q: eta = p*q/(p+q), so p = eta*q/(q-eta)
    p = eta * q / (q - eta)
    inv_2pq = 0.5 / (p + q)
    
    for L_f in range(L_ket):
        for L_e in range(L_bra + 1):
            m_top = m_total - L_e - L_f - 1
            
            for ex in range(L_e, -1, -1):
                for ey in range(L_e - ex, -1, -1):
                    ez = L_e - ex - ey
                    
                    for fx in range(L_f, -1, -1):
                        for fy in range(L_f - fx, -1, -1):
                            fz = L_f - fx - fy
                            
                            # Increment fx
                            for m in range(m_top + 1):
                                V[m, ex, ey, ez, fx+1, fy, fz] = (
                                    X_QC[0] * V[m, ex, ey, ez, fx, fy, fz] +
                                    X_WQ[0] * V[m+1, ex, ey, ez, fx, fy, fz]
                                )
                                if fx > 0:
                                    V[m, ex, ey, ez, fx+1, fy, fz] += (
                                        fx * inv_2q * (V[m, ex, ey, ez, fx-1, fy, fz] -
                                                      eta_over_q * V[m+1, ex, ey, ez, fx-1, fy, fz])
                                    )
                                if ex > 0:
                                    V[m, ex, ey, ez, fx+1, fy, fz] += (
                                        ex * inv_2pq * V[m+1, ex-1, ey, ez, fx, fy, fz]
                                    )
                            
                            # Increment fy
                            for m in range(m_top + 1):
                                V[m, ex, ey, ez, fx, fy+1, fz] = (
                                    X_QC[1] * V[m, ex, ey, ez, fx, fy, fz] +
                                    X_WQ[1] * V[m+1, ex, ey, ez, fx, fy, fz]
                                )
                                if fy > 0:
                                    V[m, ex, ey, ez, fx, fy+1, fz] += (
                                        fy * inv_2q * (V[m, ex, ey, ez, fx, fy-1, fz] -
                                                      eta_over_q * V[m+1, ex, ey, ez, fx, fy-1, fz])
                                    )
                                if ey > 0:
                                    V[m, ex, ey, ez, fx, fy+1, fz] += (
                                        ey * inv_2pq * V[m+1, ex, ey-1, ez, fx, fy, fz]
                                    )
                            
                            # Increment fz
                            for m in range(m_top + 1):
                                V[m, ex, ey, ez, fx, fy, fz+1] = (
                                    X_QC[2] * V[m, ex, ey, ez, fx, fy, fz] +
                                    X_WQ[2] * V[m+1, ex, ey, ez, fx, fy, fz]
                                )
                                if fz > 0:
                                    V[m, ex, ey, ez, fx, fy, fz+1] += (
                                        fz * inv_2q * (V[m, ex, ey, ez, fx, fy, fz-1] -
                                                      eta_over_q * V[m+1, ex, ey, ez, fx, fy, fz-1])
                                    )
                                if ez > 0:
                                    V[m, ex, ey, ez, fx, fy, fz+1] += (
                                        ez * inv_2pq * V[m+1, ex, ey, ez-1, fx, fy, fz]
                                    )




@njit(cache=True, fastmath=True, nogil=True, error_model="numpy")
def _binom(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    if k > n - k:
        k = n - k
    res = 1
    for i in range(1, k + 1):
        res = res * (n - i + 1) // i
    return res

@njit(cache=True, fastmath=True, nogil=True, error_model="numpy")
def os_hrr_element(V_vrr: np.ndarray,
                   ax: int, ay: int, az: int,
                   bx: int, by: int, bz: int,
                   cx: int, cy: int, cz: int,
                   dx: int, dy: int, dz: int,
                   X_AB: np.ndarray,   # shape (3,), float64
                   X_CD: np.ndarray) -> float:   # shape (3,), float64

    result = 0.0

    # Ket HRR loops (shifts on C)
    for ix in range(dx + 1):
        binom_x_ket = _binom(dx, ix)
        pow_x_ket   = X_CD[0] ** (dx - ix)
        cx_shift    = cx + ix

        for iy in range(dy + 1):   # using iy instead of jy to avoid j-i confusion
            binom_y_ket = _binom(dy, iy)
            pow_y_ket   = X_CD[1] ** (dy - iy)
            cy_shift    = cy + iy

            for iz in range(dz + 1):
                binom_z_ket = _binom(dz, iz)
                pow_z_ket   = X_CD[2] ** (dz - iz)
                cz_shift    = cz + iz

                ket_coeff = (binom_x_ket * pow_x_ket *
                             binom_y_ket * pow_y_ket *
                             binom_z_ket * pow_z_ket)

                # Bra HRR loops (shifts on A)
                for px in range(bx + 1):
                    binom_x_bra = _binom(bx, px)
                    pow_x_bra   = X_AB[0] ** (bx - px)
                    ax_final    = ax + px

                    for py in range(by + 1):
                        binom_y_bra = _binom(by, py)
                        pow_y_bra   = X_AB[1] ** (by - py)
                        ay_final    = ay + py

                        for pz in range(bz + 1):
                            binom_z_bra = _binom(bz, pz)
                            pow_z_bra   = X_AB[2] ** (bz - pz)
                            az_final    = az + pz

                            bra_coeff = (binom_x_bra * pow_x_bra *
                                        binom_y_bra * pow_y_bra *
                                        binom_z_bra * pow_z_bra)

                            total_coeff = ket_coeff * bra_coeff
                            result += total_coeff * V_vrr[0, ax_final, ay_final, az_final,
                                                         cx_shift, cy_shift, cz_shift]

    return result


# @njit(cache=False, nogil=True)
# def os_hrr_element(V_vrr, ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz, X_AB, X_CD):
#     # Ket HRR: reduce d to zero
#     if dx > 0:
#         return (os_hrr_element(V_vrr, ax, ay, az, bx, by, bz, cx+1, cy, cz, dx-1, dy, dz, X_AB, X_CD) +
#                 X_CD[0] * os_hrr_element(V_vrr, ax, ay, az, bx, by, bz, cx, cy, cz, dx-1, dy, dz, X_AB, X_CD))
#     elif dy > 0:
#         return (os_hrr_element(V_vrr, ax, ay, az, bx, by, bz, cx, cy+1, cz, dx, dy-1, dz, X_AB, X_CD) +
#                 X_CD[1] * os_hrr_element(V_vrr, ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy-1, dz, X_AB, X_CD))
#     elif dz > 0:
#         return (os_hrr_element(V_vrr, ax, ay, az, bx, by, bz, cx, cy, cz+1, dx, dy, dz-1, X_AB, X_CD) +
#                 X_CD[2] * os_hrr_element(V_vrr, ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz-1, X_AB, X_CD))
    
#     # d is now zero, apply bra HRR: reduce b to zero
#     if bx > 0:
#         return (os_hrr_element(V_vrr, ax+1, ay, az, bx-1, by, bz, cx, cy, cz, 0, 0, 0, X_AB, X_CD) +
#                 X_AB[0] * os_hrr_element(V_vrr, ax, ay, az, bx-1, by, bz, cx, cy, cz, 0, 0, 0, X_AB, X_CD))
#     elif by > 0:
#         return (os_hrr_element(V_vrr, ax, ay+1, az, bx, by-1, bz, cx, cy, cz, 0, 0, 0, X_AB, X_CD) +
#                 X_AB[1] * os_hrr_element(V_vrr, ax, ay, az, bx, by-1, bz, cx, cy, cz, 0, 0, 0, X_AB, X_CD))
#     elif bz > 0:
#         return (os_hrr_element(V_vrr, ax, ay, az+1, bx, by, bz-1, cx, cy, cz, 0, 0, 0, X_AB, X_CD) +
#                 X_AB[2] * os_hrr_element(V_vrr, ax, ay, az, bx, by, bz-1, cx, cy, cz, 0, 0, 0, X_AB, X_CD))
    
#     # Base case: b = d = 0, return from VRR at m=0
#     return V_vrr[0, ax, ay, az, cx, cy, cz]
