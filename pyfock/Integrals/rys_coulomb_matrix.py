import numpy as np
from numba import njit, prange, get_num_threads
from numba import get_thread_id
# from .rys_helpers_4c2e import coulomb_rys
from .rys_helpers import coulomb_rys, coulomb_rys_new
from .integral_helpers import Fboys

def rys_coulomb_matrix(basis, density_matrix, sqrt_ints4c2e_diag=None, threshold=1e-9):
    """
    Compute the Coulomb matrix (J matrix) directly from the density matrix
    using the Rys quadrature method with exploitation of 8-fold permutational 
    symmetry.

    The Coulomb matrix is defined as:
        J[i,j] = Σ_kl D[k,l] * (ij|kl)
    
    where D is the density matrix and (ij|kl) are the electron repulsion integrals.

    This function computes J directly without storing the full 4-center 2-electron
    integral tensor, significantly reducing memory requirements.

    Symmetries exploited
    --------------------
    The 4c2e integrals obey 8-fold permutational symmetry:
        (i j | k l) = (j i | k l) = (i j | l k) = (j i | l k)
                    = (k l | i j) = (l k | i j) = (k l | j i) = (l k | j i)

    The density matrix is symmetric: D[k,l] = D[l,k]

    These symmetries are exploited to minimize redundant integral evaluations.

    Parameters
    ----------
    basis : object
        Primary basis set object containing:
        - bfs_coords : Cartesian coordinates of basis function centers.
        - bfs_coeffs : Contraction coefficients.
        - bfs_expnts : Gaussian exponents.
        - bfs_prim_norms : Primitive normalization constants.
        - bfs_contr_prim_norms : Contraction normalization factors.
        - bfs_lmn : Angular momentum quantum numbers (ℓ, m, n).
        - bfs_nprim : Number of primitives per basis function.
        - bfs_nao : Total number of atomic orbitals.

    density_matrix : ndarray, shape (N, N)
        The density matrix in the AO basis. Must be symmetric.

    sqrt_ints4c2e_diag : ndarray, shape (N, N), optional
        Square root of diagonal elements (ij|ij) for Schwarz screening.
        If provided, enables efficient screening of negligible integrals.

    threshold : float, optional
        Threshold for Schwarz screening. Default is 1e-9.

    Returns
    -------
    J_matrix : ndarray, shape (N, N)
        The Coulomb matrix in the AO basis.

    Notes
    -----
    - Memory efficient: Does not store the full O(N^4) ERI tensor.
    - Uses Schwarz screening when sqrt_ints4c2e_diag is provided to skip
      negligible integral evaluations.
    - Parallelized using Numba's prange for efficient computation.

    Examples
    --------
    >>> J = rys_coulomb_matrix(basis, density_matrix)
    >>> # With Schwarz screening:
    >>> J = rys_coulomb_matrix(basis, density_matrix, sqrt_ints4c2e_diag=schwarz)
    """
    # Convert basis properties to numpy arrays for Numba efficiency
    bfs_coords = np.array([basis.bfs_coords])
    bfs_contr_prim_norms = np.array([basis.bfs_contr_prim_norms])
    bfs_lmn = np.array([basis.bfs_lmn])
    bfs_nprim = np.array([basis.bfs_nprim])
    
    # Convert irregular lists to 2D arrays with padding
    maxnprim = max(basis.bfs_nprim)
    bfs_coeffs = np.zeros([basis.bfs_nao, maxnprim])
    bfs_expnts = np.zeros([basis.bfs_nao, maxnprim])
    shell_indices = np.array([basis.bfs_shell_index], dtype=np.uint16)[0]
    bfs_prim_norms = np.zeros([basis.bfs_nao, maxnprim])
    
    for i in range(basis.bfs_nao):
        for j in range(basis.bfs_nprim[i]):
            bfs_coeffs[i,j] = basis.bfs_coeffs[i][j]
            bfs_expnts[i,j] = basis.bfs_expnts[i][j]
            bfs_prim_norms[i,j] = basis.bfs_prim_norms[i][j]
    
    # Setup Schwarz screening
    if sqrt_ints4c2e_diag is None:
        sqrt_ints4c2e_diag = np.zeros((1,1), dtype=np.float64)
        isSchwarz = False
    else:
        isSchwarz = True
    
    # Ensure density matrix is contiguous for Numba
    # density_matrix = np.ascontiguousarray(density_matrix, dtype=np.float64)
    ncore = get_num_threads()
    J_matrix = rys_coulomb_matrix_internal(
        bfs_coords[0], bfs_contr_prim_norms[0], bfs_lmn[0], bfs_nprim[0], 
        bfs_coeffs, bfs_prim_norms, bfs_expnts, shell_indices, density_matrix,
        sqrt_ints4c2e_diag, isSchwarz, threshold
    )
    
    return J_matrix


@njit(parallel=True, fastmath=True, error_model="numpy")
def rys_coulomb_matrix_internal(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, 
                                     bfs_coeffs, bfs_prim_norms, bfs_expnts, shell_indices,
                                     density_matrix, 
                                     sqrt_ints4c2e_diag, isSchwarz=False, threshold=1e-9):
    """
    Calculates the Coulomb (J) matrix in-place using 8-fold symmetry and Rys Quadrature.
    
    OPTIMIZATIONS:
    - Direct SCF: Contracts integrals with Density immediately (No 4D tensor I/O).
    - 8-Fold Symmetry: Computes unique (ij|kl) and updates both J[i,j] and J[k,l].
    - Thread Local Storage: Uses private buffers to avoid race conditions during scatter operations.
    """

    # Determine total basis size from coordinates array
    nbf_total = len(bfs_coords)
    
    # Initialize Thread Local Storage for J Matrix
    # Shape: (Num_Threads, N, N). This eats memory but guarantees thread safety + max speed.
    n_threads = get_num_threads()
    J_local = np.zeros((n_threads, nbf_total, nbf_total), dtype=np.float64)

    # Check symmetries based on index ranges (Block symmetry logic)
    all_symm = True

    pi = np.pi
    twopisq = 2.0 * pi * pi 
    n_negligible = 0
    n_significant = 0
    DENS_THRESH = 1e-11
    COMBINED_THRESH = threshold * 0.1
    PRIMITIVE_THRESH = 1e-8
    # --- PARALLEL LOOP OVER SHELL 'i' ---
    for i in prange(0, nbf_total): 
        tid = get_thread_id() # ID for local buffer access
        roots = np.zeros((10))
        weights = np.zeros((10))
        G = np.zeros((20, 20))
        I = bfs_coords[i]
        Ni = bfs_contr_prim_norms[i]
        la, ma, na = bfs_lmn[i]
        nprimi = bfs_nprim[i]
        jshell_previous = -12 # Some random number
        kshell_previous = -12 # Some random number
        lshell_previous = -122 # Some random number

        for j in range(0, i+1): 
            # Symmetry Check 1: Pair Permutation (ij vs ji)
            if (all_symm and j<=i):
                
                # Check Triangular index for 8-fold (Pair Exchange)
                if all_symm:
                    if i < j:
                        triangle2ij = (j)*(j+1)/2 + i
                    else:
                        triangle2ij = (i)*(i+1)/2 + j
                
                # Schwarz Screening (Outer)
                sqrt_ij = 0.0
                if isSchwarz:
                    sqrt_ij = sqrt_ints4c2e_diag[i,j]
                    if sqrt_ij < threshold: # Loose check
                        continue

                lb, mb, nb = bfs_lmn[j]
                jshell = shell_indices[j]
                if jshell!=jshell_previous:
                    
                    J_coord = bfs_coords[j]
                    IJ = I - J_coord
                    IJsq = np.sum(IJ**2)
                    Nj = bfs_contr_prim_norms[j]
                    
                    tempcoeff1 = Ni * Nj
                    nprimj = bfs_nprim[j]
                # Pre-fetch Density ij
                d_val_ij = density_matrix[i, j]
                if abs(d_val_ij)<DENS_THRESH:
                    continue
                
                for k in range(0, nbf_total): 
                    lc, mc, nc = bfs_lmn[k]
                    kshell = shell_indices[k]
                    if kshell!=kshell_previous:
                        K = bfs_coords[k]
                        nprimk = bfs_nprim[k]
                    Nk = bfs_contr_prim_norms[k]
                    
                    if jshell!=jshell_previous or kshell!=kshell_previous:
                        tempcoeff2 = tempcoeff1 * Nk
                        
                    
                    for l in range(0, k+1): 
                        # Symmetry Check 2: Pair Permutation (kl vs lk) and Pair Exchange (ij vs kl)
                        if (all_symm and l<=k):
                            
                            # 8-Fold Check: Ensure pair(ij) >= pair(kl)
                            if all_symm:
                                if k < l:
                                    triangle2kl = (l)*(l+1)/2 + k
                                else:
                                    triangle2kl = (k)*(k+1)/2 + l
                                if triangle2ij > triangle2kl:
                                    continue
                            
                            # Schwarz Screening (Inner)
                            if isSchwarz:
                                sqrt_kl = sqrt_ints4c2e_diag[k,l]
                                if sqrt_kl < threshold:
                                    continue
                                schwarz_prod = sqrt_ij * sqrt_kl
                                if schwarz_prod < threshold:
                                    continue
                                
                                # Density Screening (Standard Direct SCF)
                                # Skip if both density contributions are negligible
                                d_val_kl = density_matrix[k, l]
                                if abs(d_val_kl)<DENS_THRESH:
                                    continue
                                D_max = max(abs(d_val_ij), abs(d_val_kl), 
                                        abs(density_matrix[i,k]), abs(density_matrix[i,l]), 
                                        abs(density_matrix[j,k]), abs(density_matrix[j,l]))
                                if abs(D_max) < DENS_THRESH:
                                    continue
                                if abs(D_max)*schwarz_prod < COMBINED_THRESH:
                                    continue
                                if (abs(d_val_ij) * schwarz_prod < COMBINED_THRESH) and (abs(d_val_kl) * schwarz_prod < COMBINED_THRESH):
                                    continue
                                
                            else:
                                d_val_kl = density_matrix[k, l]

                            
                            ld, md, nd = bfs_lmn[l]
                            lshell = shell_indices[l]
                            # if lshell!=lshell_previous:
                            L = bfs_coords[l]
                            npriml = bfs_nprim[l]
                            Nl = bfs_contr_prim_norms[l]
                                     
                            # if kshell!=kshell_previous or lshell!=lshell_previous:
                            KL = K - L  
                            KLsq = np.sum(KL**2)

                            # if jshell!=jshell_previous or kshell!=kshell_previous or lshell!=lshell_previous:
                            tempcoeff3 = tempcoeff2*Nl
                            
                            
                            # Rys Roots Setup
                            norder = int((la+ma+na+lb+mb+nb+lc+mc+nc+ld+md+nd)/2 + 1) 
                            n = int(max(la+lb, ma+mb, na+nb))
                            m = int(max(lc+ld, mc+md, nc+nd))
                            

                            val = 0.0
                            
                            # --- PRIMITIVE LOOPS ---
                            for ik in range(nprimi):   
                                dik = bfs_coeffs[i][ik]
                                Nik = bfs_prim_norms[i][ik]
                                alphaik = bfs_expnts[i][ik]
                                tempcoeff4 = tempcoeff3 * dik * Nik
                                
                                for jk in range(nprimj):
                                    alphajk = bfs_expnts[j][jk]
                                    gammaP = alphaik + alphajk
                                    screenfactorAB = np.exp(-alphaik * alphajk / gammaP * IJsq)
                                    
                                    if screenfactorAB < PRIMITIVE_THRESH: continue

                                    djk = bfs_coeffs[j][jk] 
                                    Njk = bfs_prim_norms[j][jk]      
                                    tempcoeff5 = tempcoeff4 * djk * Njk  
                                    
                                    for kk in range(nprimk):
                                        dkk = bfs_coeffs[k][kk]
                                        Nkk = bfs_prim_norms[k][kk]
                                        alphakk = bfs_expnts[k][kk]
                                        tempcoeff6 = tempcoeff5 * dkk * Nkk 
                                        
                                        for lk in range(npriml): 
                                            alphalk = bfs_expnts[l][lk]
                                            gammaQ = alphakk + alphalk
                                            screenfactorKL = np.exp(-alphakk * alphalk / gammaQ * KLsq)
                                            if screenfactorKL < PRIMITIVE_THRESH: continue
                                            # Combined Screening
                                            if screenfactorKL * screenfactorAB < PRIMITIVE_THRESH: continue

                                            dlk = bfs_coeffs[l][lk] 
                                            Nlk = bfs_prim_norms[l][lk]     
                                            
                                            rho = gammaP * gammaQ / (gammaP + gammaQ)
                                            
                                            # Geometry for Rys
                                            P = (alphaik*I + alphajk*J_coord)/gammaP
                                            Q = (alphakk*K + alphalk*L)/gammaQ        
                                            PQ = P - Q
                                            PQsq = np.sum(PQ**2)

                                            tempcoeff7 = tempcoeff6 * dlk * Nlk
                                            # (ss|ss) case
                                            if (la+ma+na+lb+mb+nb+lc+mc+nc+ld+md+nd)==0: 
                                                val += tempcoeff7*twopisq/(gammaP*gammaQ)*np.sqrt(pi/(gammaP+gammaQ))\
                                                    *screenfactorAB*screenfactorKL*Fboys(0,PQsq/(1/gammaP+1/gammaQ))
                                            else:
                                                if norder <= 10:
                                                    val += tempcoeff7 * coulomb_rys(roots, weights, G, PQsq, rho, norder, n, m, 
                                                                                la, lb, lc, ld, ma, mb, mc, md, na, nb, nc, nd, 
                                                                                alphaik, alphajk, alphakk, alphalk, I, J_coord, K, L)
                            if abs(val)<1e-13:
                                n_negligible+=1
                            else:
                                n_significant+=1
                            
                            # --- IN-PLACE J-MATRIX UPDATE (8-FOLD SCATTER) ---
                            # Logic:
                            # We computed integral (ij|kl). This single value represents multiple terms in the full sum
                            # due to permutational symmetry.
                            # Contribution to J[i,j] comes from Density[k,l]
                            # Contribution to J[k,l] comes from Density[i,j]
                            
                            # Factor for k,l pair: if k!=l, D[k,l] and D[l,k] both contribute.
                            fac_kl = 2.0 if k != l else 1.0
                            term_ij = val * d_val_kl * fac_kl
                            J_local[tid, i, j] += term_ij
                            
                            # 8-fold Symmetry update: Update J[k,l] ONLY if pair (kl) is distinct from (ij)
                            # "Distinct" means we aren't on the diagonal of the pair-matrix.
                            is_same_pair = (i == k and j == l)
                            
                            if not is_same_pair:
                                fac_ij = 2.0 if i != j else 1.0
                                term_kl = val * d_val_ij * fac_ij
                                J_local[tid, k, l] += term_kl
                        lshell_previous = lshell
                    kshell_previous = kshell    
                jshell_previous = jshell
    # --- REDUCTION STEP ---
    # Sum all thread-local matrices into the master J matrix
    # Summing axis 0 (threads) -> (N, N)
    J_final = np.sum(J_local, axis=0)
    print("N_negligible integrals: ", n_negligible)
    print("N_significant integrals: ", n_significant)
    # --- SYMMETRIZATION ---
    # We only filled J[i,j] where j<=i (and similarly for k,l). 
    # Now we copy lower triangle to upper triangle.
    for r in prange(nbf_total):
        for c in range(r):
            val = J_final[r, c]
            J_final[c, r] = val

    return J_final
