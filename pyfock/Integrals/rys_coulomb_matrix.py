import numpy as np
from numba import njit, prange, get_num_threads

# from .rys_helpers_4c2e import coulomb_rys
from .rys_helpers import coulomb_rys

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
        bfs_coeffs, bfs_prim_norms, bfs_expnts, density_matrix,
        sqrt_ints4c2e_diag, isSchwarz, threshold
    )
    
    return J_matrix

# @njit(parallel=True, cache=True, fastmath=True, error_model="numpy")
# def rys_coulomb_matrix_internal(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, 
#                                      bfs_coeffs, bfs_prim_norms, bfs_expnts, 
#                                      density_matrix, 
#                                      sqrt_ints4c2e_diag, ncore=1, isSchwarz=False, threshold=1e-9):
#     """
#     Calculates the Coulomb (J) matrix in-place using 8-fold symmetry and Rys Quadrature.
    
#     OPTIMIZATIONS:
#     - Direct SCF: Contracts integrals with Density immediately (No 4D tensor I/O).
#     - 8-Fold Symmetry: Computes unique (ij|kl) and updates both J[i,j] and J[k,l].
#     - Thread Local Storage: Uses private buffers with explicit index ranges per thread.
#     - Cache-friendly: No get_thread_id() calls, allowing disk caching.
    
#     Strategy: Divide basis functions into ncore chunks. Each thread processes a specific
#     range of i indices, ensuring no conflicts.
#     """

#     # Determine total basis size from coordinates array
#     nbf_total = len(bfs_coords)
    
#     # Create index ranges for each thread
#     # Divide [0, nbf_total) into ncore roughly equal chunks
#     chunk_size = (nbf_total + ncore - 1) // ncore  # Ceiling division
    
#     # Build array of (start, end) pairs for each thread
#     thread_ranges = np.zeros((ncore, 2), dtype=np.int64)
#     for t in range(ncore):
#         start = t * chunk_size
#         end = min(start + chunk_size, nbf_total)
#         thread_ranges[t, 0] = start
#         thread_ranges[t, 1] = end
    
#     # Initialize Thread Local Storage for J Matrix
#     # Shape: (ncore, N, N). Each thread gets its own slice.
#     J_local = np.zeros((ncore, nbf_total, nbf_total), dtype=np.float64)

#     # Check symmetries based on index ranges (Block symmetry logic)
#     all_symm = False
#     left_side_symm = False
#     right_side_symm = False
#     both_left_right_symm = False
#     no_symm = False
#     indx_startA = 0
#     indx_endA = nbf_total
#     indx_startB = 0
#     indx_endB = nbf_total   
#     indx_startC = 0
#     indx_endC = nbf_total
#     indx_startD = 0
#     indx_endD = nbf_total
#     if indx_startA==indx_startB==indx_startC==indx_startD and indx_endA==indx_endB==indx_endC==indx_endD:
#         all_symm = True
#     elif (indx_startA==indx_startB and indx_endA==indx_endB) and (indx_startC==indx_startD and indx_endC==indx_endD):
#         both_left_right_symm = True
#     elif indx_startA==indx_startB and indx_endA==indx_endB:
#         left_side_symm = True
#     elif indx_startC==indx_startD and indx_endC==indx_endD:
#         right_side_symm = True
#     else:
#         no_symm = True

#     pi = np.pi
#     twopisq = 2.0 * pi * pi 

#     # --- PARALLEL LOOP OVER THREAD CHUNKS ---
#     # Each iteration processes one thread's assigned range of i indices
#     for thread_idx in prange(ncore):
#         # Get this thread's assigned range
#         i_start = thread_ranges[thread_idx, 0]
#         i_end = thread_ranges[thread_idx, 1]
        
#         # Process all i values in this thread's range
#         for i in range(i_start, i_end):
        
#             I = bfs_coords[i]
#             Ni = bfs_contr_prim_norms[i]
#             la, ma, na = bfs_lmn[i]
#             nprimi = bfs_nprim[i]
            
#             for j in range(indx_startB, indx_endB): 
#                 # Symmetry Check 1: Pair Permutation (ij vs ji)
#                 if (all_symm and j<=i) or (left_side_symm and j<=i) or right_side_symm or no_symm or (both_left_right_symm and j<=i):
                    
#                     # Check Triangular index for 8-fold (Pair Exchange)
#                     if all_symm:
#                         if i < j:
#                             triangle2ij = (j)*(j+1)/2 + i
#                         else:
#                             triangle2ij = (i)*(i+1)/2 + j
                    
#                     # Schwarz Screening (Outer)
#                     sqrt_ij = 0.0
#                     if isSchwarz:
#                         sqrt_ij = sqrt_ints4c2e_diag[i,j]
#                         if sqrt_ij < threshold: # Loose check
#                             continue

#                     J_coord = bfs_coords[j]
#                     IJ = I - J_coord
#                     IJsq = np.sum(IJ**2)
#                     Nj = bfs_contr_prim_norms[j]
#                     lb, mb, nb = bfs_lmn[j]
#                     tempcoeff1 = Ni * Nj
#                     nprimj = bfs_nprim[j]
                    
#                     # Pre-fetch Density ij
#                     d_val_ij = density_matrix[i, j]
                    
#                     for k in range(indx_startC, indx_endC): 
#                         K = bfs_coords[k]
#                         Nk = bfs_contr_prim_norms[k]
#                         lc, mc, nc = bfs_lmn[k]
#                         tempcoeff2 = tempcoeff1 * Nk
#                         nprimk = bfs_nprim[k]
                        
#                         for l in range(indx_startD, indx_endD): 
#                             # Symmetry Check 2: Pair Permutation (kl vs lk) and Pair Exchange (ij vs kl)
#                             if (all_symm and l<=k) or (right_side_symm and l<=k) or (left_side_symm and j<=i) or no_symm or (both_left_right_symm and l<=k):
                                
#                                 # 8-Fold Check: Ensure pair(ij) >= pair(kl)
#                                 if all_symm:
#                                     if k < l:
#                                         triangle2kl = (l)*(l+1)/2 + k
#                                     else:
#                                         triangle2kl = (k)*(k+1)/2 + l
#                                     if triangle2ij > triangle2kl:
#                                         continue
                                
#                                 # Schwarz Screening (Inner)
#                                 sqrt_kl = 0.0
#                                 if isSchwarz:
#                                     sqrt_kl = sqrt_ints4c2e_diag[k,l]
#                                     schwarz_prod = sqrt_ij * sqrt_kl
#                                     if schwarz_prod < threshold:
#                                         continue
                                    
#                                     # Density Screening (Standard Direct SCF)
#                                     # Skip if both density contributions are negligible
#                                     d_val_kl = density_matrix[k, l]
#                                     if (abs(d_val_ij) * schwarz_prod < threshold*0.1) and (abs(d_val_kl) * schwarz_prod < threshold*0.1):
#                                         continue
#                                 else:
#                                     d_val_kl = density_matrix[k, l]

#                                 L = bfs_coords[l]
#                                 KL = K - L  
#                                 KLsq = np.sum(KL**2)
#                                 Nl = bfs_contr_prim_norms[l]
#                                 ld, md, nd = bfs_lmn[l]
#                                 tempcoeff3 = tempcoeff2 * Nl
#                                 npriml = bfs_nprim[l]

#                                 # Rys Roots Setup
#                                 norder = int((la+ma+na+lb+mb+nb+lc+mc+nc+ld+md+nd)/2 + 1) 
#                                 n = int(max(la+lb, ma+mb, na+nb))
#                                 m = int(max(lc+ld, mc+md, nc+nd))
                                
#                                 # Stack/Heap allocation for Rys (Small enough for stack usually)
#                                 roots = np.zeros((norder))
#                                 weights = np.zeros((norder))
#                                 G = np.zeros((n+1, m+1))

#                                 val = 0.0
                                
#                                 # --- PRIMITIVE LOOPS ---
#                                 for ik in range(nprimi):   
#                                     dik = bfs_coeffs[i][ik]
#                                     Nik = bfs_prim_norms[i][ik]
#                                     alphaik = bfs_expnts[i][ik]
#                                     tempcoeff4 = tempcoeff3 * dik * Nik
                                    
#                                     for jk in range(nprimj):
#                                         alphajk = bfs_expnts[j][jk]
#                                         gammaP = alphaik + alphajk
#                                         screenfactorAB = np.exp(-alphaik * alphajk / gammaP * IJsq)
                                        
#                                         if screenfactorAB < 1.0e-8: continue

#                                         djk = bfs_coeffs[j][jk] 
#                                         Njk = bfs_prim_norms[j][jk]      
#                                         tempcoeff5 = tempcoeff4 * djk * Njk  
                                        
#                                         for kk in range(nprimk):
#                                             dkk = bfs_coeffs[k][kk]
#                                             Nkk = bfs_prim_norms[k][kk]
#                                             alphakk = bfs_expnts[k][kk]
#                                             tempcoeff6 = tempcoeff5 * dkk * Nkk 
                                            
#                                             for lk in range(npriml): 
#                                                 alphalk = bfs_expnts[l][lk]
#                                                 gammaQ = alphakk + alphalk
#                                                 screenfactorKL = np.exp(-alphakk * alphalk / gammaQ * KLsq)
#                                                 if screenfactorKL < 1.0e-8: continue
#                                                 # Combined Screening
#                                                 if screenfactorKL * screenfactorAB < 1.0e-8: continue

#                                                 dlk = bfs_coeffs[l][lk] 
#                                                 Nlk = bfs_prim_norms[l][lk]     
                                                
#                                                 rho = gammaP * gammaQ / (gammaP + gammaQ)
                                                
#                                                 # Geometry for Rys
#                                                 P = (alphaik*I + alphajk*J_coord)/gammaP
#                                                 Q = (alphakk*K + alphalk*L)/gammaQ        
#                                                 PQ = P - Q
#                                                 PQsq = np.sum(PQ**2)

#                                                 tempcoeff7 = tempcoeff6 * dlk * Nlk
                                                
#                                                 if norder <= 10:
#                                                     val += tempcoeff7 * coulomb_rys(roots, weights, G, PQsq, rho, norder, n, m, 
#                                                                                 la, lb, lc, ld, ma, mb, mc, md, na, nb, nc, nd, 
#                                                                                 alphaik, alphajk, alphakk, alphalk, I, J_coord, K, L)
                                
#                                 # --- IN-PLACE J-MATRIX UPDATE (8-FOLD SCATTER) ---
#                                 # Write to this thread's local buffer (thread_idx)
#                                 # This is guaranteed safe because only this thread processes this i value
                                
#                                 # Factor for k,l pair: if k!=l, D[k,l] and D[l,k] both contribute.
#                                 fac_kl = 2.0 if k != l else 1.0
#                                 term_ij = val * d_val_kl * fac_kl
#                                 J_local[thread_idx, i, j] += term_ij
                                
#                                 # 8-fold Symmetry update: Update J[k,l] ONLY if pair (kl) is distinct from (ij)
#                                 is_same_pair = (i == k and j == l)
                                
#                                 if not is_same_pair:
#                                     fac_ij = 2.0 if i != j else 1.0
#                                     term_kl = val * d_val_ij * fac_ij
#                                     # Find which thread owns index k
#                                     tid_k = k // chunk_size
#                                     if tid_k >= ncore:  # Handle edge case
#                                         tid_k = ncore - 1
#                                     J_local[tid_k, k, l] += term_kl

#     # --- REDUCTION STEP ---
#     # Sum all thread-local matrices into the master J matrix
#     J_final = np.sum(J_local, axis=0)

#     # --- SYMMETRIZATION ---
#     # Copy lower triangle to upper triangle
#     for r in range(nbf_total):
#         for c in range(r):
#             val = J_final[r, c]
#             J_final[c, r] = val

#     return J_final
# @njit(parallel=True, fastmath=True, cache=True, error_model="numpy")
# def rys_coulomb_matrix_internal(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, 
#                                      bfs_coeffs, bfs_prim_norms, bfs_expnts, 
#                                      density_matrix, 
#                                      sqrt_ints4c2e_diag, ncore=1, isSchwarz=False, threshold=1e-9):
#     """
#     Calculates the Coulomb (J) matrix in-place using 8-fold symmetry and Rys Quadrature.
    
#     OPTIMIZATIONS:
#     - Direct SCF: Contracts integrals with Density immediately (No 4D tensor I/O).
#     - 8-Fold Symmetry: Computes unique (ij|kl) and updates both J[i,j] and J[k,l].
#     - Thread Local Storage: Uses private buffers indexed by shell pair to avoid race conditions.
#     - Cache-friendly: No get_thread_id() calls, allowing disk caching.
    
#     Strategy: Each thread works on a subset of shell pairs (i,j). Since we control which
#     thread writes to which (i,j) pairs, we don't need thread IDs.
#     """

#     # Determine total basis size from coordinates array
#     nbf_total = len(bfs_coords)
    
#     # Initialize Thread Local Storage for J Matrix
#     # Shape: (ncore, N, N). Each thread gets its own slice.
#     J_local = np.zeros((ncore, nbf_total, nbf_total), dtype=np.float64)

#     # Check symmetries based on index ranges (Block symmetry logic)
#     all_symm = False
#     left_side_symm = False
#     right_side_symm = False
#     both_left_right_symm = False
#     no_symm = False
#     indx_startA = 0
#     indx_endA = nbf_total
#     indx_startB = 0
#     indx_endB = nbf_total   
#     indx_startC = 0
#     indx_endC = nbf_total
#     indx_startD = 0
#     indx_endD = nbf_total
#     if indx_startA==indx_startB==indx_startC==indx_startD and indx_endA==indx_endB==indx_endC==indx_endD:
#         all_symm = True
#     elif (indx_startA==indx_startB and indx_endA==indx_endB) and (indx_startC==indx_startD and indx_endC==indx_endD):
#         both_left_right_symm = True
#     elif indx_startA==indx_startB and indx_endA==indx_endB:
#         left_side_symm = True
#     elif indx_startC==indx_startD and indx_endC==indx_endD:
#         right_side_symm = True
#     else:
#         no_symm = True

#     pi = np.pi
#     twopisq = 2.0 * pi * pi 

#     # --- PARALLEL LOOP OVER SHELL 'i' ---
#     # Strategy: prange automatically distributes iterations across threads.
#     # Each iteration (i) will be assigned to exactly one thread by Numba's scheduler.
#     # Since we process all j for a given i in sequence (not parallel), there's no
#     # race condition on J_local for the (i,j) pairs.
    
#     for i in prange(indx_startA, indx_endA): 
#         # Determine which thread-local buffer to use based on modulo
#         # This ensures each thread writes to its own slice
#         tid = i % ncore
        
#         I = bfs_coords[i]
#         Ni = bfs_contr_prim_norms[i]
#         la, ma, na = bfs_lmn[i]
#         nprimi = bfs_nprim[i]
        
#         for j in range(indx_startB, indx_endB): 
#             # Symmetry Check 1: Pair Permutation (ij vs ji)
#             if (all_symm and j<=i) or (left_side_symm and j<=i) or right_side_symm or no_symm or (both_left_right_symm and j<=i):
                
#                 # Check Triangular index for 8-fold (Pair Exchange)
#                 if all_symm:
#                     if i < j:
#                         triangle2ij = (j)*(j+1)/2 + i
#                     else:
#                         triangle2ij = (i)*(i+1)/2 + j
                
#                 # Schwarz Screening (Outer)
#                 sqrt_ij = 0.0
#                 if isSchwarz:
#                     sqrt_ij = sqrt_ints4c2e_diag[i,j]
#                     if sqrt_ij < threshold: # Loose check
#                         continue

#                 J_coord = bfs_coords[j]
#                 IJ = I - J_coord
#                 IJsq = np.sum(IJ**2)
#                 Nj = bfs_contr_prim_norms[j]
#                 lb, mb, nb = bfs_lmn[j]
#                 tempcoeff1 = Ni * Nj
#                 nprimj = bfs_nprim[j]
                
#                 # Pre-fetch Density ij
#                 d_val_ij = density_matrix[i, j]
                
#                 for k in range(indx_startC, indx_endC): 
#                     K = bfs_coords[k]
#                     Nk = bfs_contr_prim_norms[k]
#                     lc, mc, nc = bfs_lmn[k]
#                     tempcoeff2 = tempcoeff1 * Nk
#                     nprimk = bfs_nprim[k]
                    
#                     for l in range(indx_startD, indx_endD): 
#                         # Symmetry Check 2: Pair Permutation (kl vs lk) and Pair Exchange (ij vs kl)
#                         if (all_symm and l<=k) or (right_side_symm and l<=k) or (left_side_symm and j<=i) or no_symm or (both_left_right_symm and l<=k):
                            
#                             # 8-Fold Check: Ensure pair(ij) >= pair(kl)
#                             if all_symm:
#                                 if k < l:
#                                     triangle2kl = (l)*(l+1)/2 + k
#                                 else:
#                                     triangle2kl = (k)*(k+1)/2 + l
#                                 if triangle2ij > triangle2kl:
#                                     continue
                            
#                             # Schwarz Screening (Inner)
#                             sqrt_kl = 0.0
#                             if isSchwarz:
#                                 sqrt_kl = sqrt_ints4c2e_diag[k,l]
#                                 schwarz_prod = sqrt_ij * sqrt_kl
#                                 if schwarz_prod < threshold:
#                                     continue
                                
#                                 # Density Screening (Standard Direct SCF)
#                                 # Skip if both density contributions are negligible
#                                 d_val_kl = density_matrix[k, l]
#                                 if (abs(d_val_ij) * schwarz_prod < threshold*0.1) and (abs(d_val_kl) * schwarz_prod < threshold*0.1):
#                                     continue
#                             else:
#                                 d_val_kl = density_matrix[k, l]

#                             L = bfs_coords[l]
#                             KL = K - L  
#                             KLsq = np.sum(KL**2)
#                             Nl = bfs_contr_prim_norms[l]
#                             ld, md, nd = bfs_lmn[l]
#                             tempcoeff3 = tempcoeff2 * Nl
#                             npriml = bfs_nprim[l]

#                             # Rys Roots Setup
#                             norder = int((la+ma+na+lb+mb+nb+lc+mc+nc+ld+md+nd)/2 + 1) 
#                             n = int(max(la+lb, ma+mb, na+nb))
#                             m = int(max(lc+ld, mc+md, nc+nd))
                            
#                             # Stack/Heap allocation for Rys (Small enough for stack usually)
#                             roots = np.zeros((norder))
#                             weights = np.zeros((norder))
#                             G = np.zeros((n+1, m+1))

#                             val = 0.0
                            
#                             # --- PRIMITIVE LOOPS ---
#                             for ik in range(nprimi):   
#                                 dik = bfs_coeffs[i][ik]
#                                 Nik = bfs_prim_norms[i][ik]
#                                 alphaik = bfs_expnts[i][ik]
#                                 tempcoeff4 = tempcoeff3 * dik * Nik
                                
#                                 for jk in range(nprimj):
#                                     alphajk = bfs_expnts[j][jk]
#                                     gammaP = alphaik + alphajk
#                                     screenfactorAB = np.exp(-alphaik * alphajk / gammaP * IJsq)
                                    
#                                     if screenfactorAB < 1.0e-8: continue

#                                     djk = bfs_coeffs[j][jk] 
#                                     Njk = bfs_prim_norms[j][jk]      
#                                     tempcoeff5 = tempcoeff4 * djk * Njk  
                                    
#                                     for kk in range(nprimk):
#                                         dkk = bfs_coeffs[k][kk]
#                                         Nkk = bfs_prim_norms[k][kk]
#                                         alphakk = bfs_expnts[k][kk]
#                                         tempcoeff6 = tempcoeff5 * dkk * Nkk 
                                        
#                                         for lk in range(npriml): 
#                                             alphalk = bfs_expnts[l][lk]
#                                             gammaQ = alphakk + alphalk
#                                             screenfactorKL = np.exp(-alphakk * alphalk / gammaQ * KLsq)
#                                             if screenfactorKL < 1.0e-8: continue
#                                             # Combined Screening
#                                             if screenfactorKL * screenfactorAB < 1.0e-8: continue

#                                             dlk = bfs_coeffs[l][lk] 
#                                             Nlk = bfs_prim_norms[l][lk]     
                                            
#                                             rho = gammaP * gammaQ / (gammaP + gammaQ)
                                            
#                                             # Geometry for Rys
#                                             P = (alphaik*I + alphajk*J_coord)/gammaP
#                                             Q = (alphakk*K + alphalk*L)/gammaQ        
#                                             PQ = P - Q
#                                             PQsq = np.sum(PQ**2)

#                                             tempcoeff7 = tempcoeff6 * dlk * Nlk
                                            
#                                             if norder <= 10:
#                                                 val += tempcoeff7 * coulomb_rys(roots, weights, G, PQsq, rho, norder, n, m, 
#                                                                                la, lb, lc, ld, ma, mb, mc, md, na, nb, nc, nd, 
#                                                                                alphaik, alphajk, alphakk, alphalk, I, J_coord, K, L)
                            
#                             # --- IN-PLACE J-MATRIX UPDATE (8-FOLD SCATTER) ---
#                             # Write to thread-local buffer determined by tid = i % ncore
#                             # This ensures no race conditions since each 'i' is processed by one thread
                            
#                             # Factor for k,l pair: if k!=l, D[k,l] and D[l,k] both contribute.
#                             fac_kl = 2.0 if k != l else 1.0
#                             term_ij = val * d_val_kl * fac_kl
#                             J_local[tid, i, j] += term_ij
                            
#                             # 8-fold Symmetry update: Update J[k,l] ONLY if pair (kl) is distinct from (ij)
#                             is_same_pair = (i == k and j == l)
                            
#                             if not is_same_pair:
#                                 fac_ij = 2.0 if i != j else 1.0
#                                 term_kl = val * d_val_ij * fac_ij
#                                 # Determine which buffer to write to for (k,l)
#                                 tid_kl = k % ncore
#                                 J_local[tid_kl, k, l] += term_kl

#     # --- REDUCTION STEP ---
#     # Sum all thread-local matrices into the master J matrix
#     J_final = np.sum(J_local, axis=0)

#     # --- SYMMETRIZATION ---
#     # Copy lower triangle to upper triangle
#     for r in range(nbf_total):
#         for c in range(r):
#             val = J_final[r, c]
#             J_final[c, r] = val

#     return J_final
from numba import get_thread_id
# from numba.np.ufunc.parallel import _get_thread_id

@njit(parallel=True, fastmath=True, error_model="numpy")
def rys_coulomb_matrix_internal(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, 
                                     bfs_coeffs, bfs_prim_norms, bfs_expnts, 
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
    all_symm = False
    left_side_symm = False
    right_side_symm = False
    both_left_right_symm = False
    no_symm = False
    indx_startA = 0
    indx_endA = nbf_total
    indx_startB = 0
    indx_endB = nbf_total   
    indx_startC = 0
    indx_endC = nbf_total
    indx_startD = 0
    indx_endD = nbf_total
    if indx_startA==indx_startB==indx_startC==indx_startD and indx_endA==indx_endB==indx_endC==indx_endD:
        all_symm = True
    elif (indx_startA==indx_startB and indx_endA==indx_endB) and (indx_startC==indx_startD and indx_endC==indx_endD):
        both_left_right_symm = True
    elif indx_startA==indx_startB and indx_endA==indx_endB:
        left_side_symm = True
    elif indx_startC==indx_startD and indx_endC==indx_endD:
        right_side_symm = True
    else:
        no_symm = True

    pi = np.pi
    twopisq = 2.0 * pi * pi 

    # --- PARALLEL LOOP OVER SHELL 'i' ---
    for i in prange(indx_startA, indx_endA): 
        tid = get_thread_id() # ID for local buffer access
        
        I = bfs_coords[i]
        Ni = bfs_contr_prim_norms[i]
        la, ma, na = bfs_lmn[i]
        nprimi = bfs_nprim[i]
        
        for j in range(indx_startB, indx_endB): 
            # Symmetry Check 1: Pair Permutation (ij vs ji)
            if (all_symm and j<=i) or (left_side_symm and j<=i) or right_side_symm or no_symm or (both_left_right_symm and j<=i):
                
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

                J_coord = bfs_coords[j]
                IJ = I - J_coord
                IJsq = np.sum(IJ**2)
                Nj = bfs_contr_prim_norms[j]
                lb, mb, nb = bfs_lmn[j]
                tempcoeff1 = Ni * Nj
                nprimj = bfs_nprim[j]
                
                # Pre-fetch Density ij
                d_val_ij = density_matrix[i, j]
                
                for k in range(indx_startC, indx_endC): 
                    K = bfs_coords[k]
                    Nk = bfs_contr_prim_norms[k]
                    lc, mc, nc = bfs_lmn[k]
                    tempcoeff2 = tempcoeff1 * Nk
                    nprimk = bfs_nprim[k]
                    
                    for l in range(indx_startD, indx_endD): 
                        # Symmetry Check 2: Pair Permutation (kl vs lk) and Pair Exchange (ij vs kl)
                        if (all_symm and l<=k) or (right_side_symm and l<=k) or (left_side_symm and j<=i) or no_symm or (both_left_right_symm and l<=k):
                            
                            # 8-Fold Check: Ensure pair(ij) >= pair(kl)
                            if all_symm:
                                if k < l:
                                    triangle2kl = (l)*(l+1)/2 + k
                                else:
                                    triangle2kl = (k)*(k+1)/2 + l
                                if triangle2ij > triangle2kl:
                                    continue
                            
                            # Schwarz Screening (Inner)
                            sqrt_kl = 0.0
                            if isSchwarz:
                                sqrt_kl = sqrt_ints4c2e_diag[k,l]
                                schwarz_prod = sqrt_ij * sqrt_kl
                                if schwarz_prod < threshold:
                                    continue
                                
                                # Density Screening (Standard Direct SCF)
                                # Skip if both density contributions are negligible
                                d_val_kl = density_matrix[k, l]
                                if (abs(d_val_ij) * schwarz_prod < threshold*0.1) and (abs(d_val_kl) * schwarz_prod < threshold*0.1):
                                    continue
                            else:
                                d_val_kl = density_matrix[k, l]

                            L = bfs_coords[l]
                            KL = K - L  
                            KLsq = np.sum(KL**2)
                            Nl = bfs_contr_prim_norms[l]
                            ld, md, nd = bfs_lmn[l]
                            tempcoeff3 = tempcoeff2 * Nl
                            npriml = bfs_nprim[l]

                            # Rys Roots Setup
                            norder = int((la+ma+na+lb+mb+nb+lc+mc+nc+ld+md+nd)/2 + 1) 
                            n = int(max(la+lb, ma+mb, na+nb))
                            m = int(max(lc+ld, mc+md, nc+nd))
                            
                            # Stack/Heap allocation for Rys (Small enough for stack usually)
                            roots = np.zeros((norder))
                            weights = np.zeros((norder))
                            G = np.zeros((n+1, m+1))

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
                                    
                                    if screenfactorAB < 1.0e-8: continue

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
                                            if screenfactorKL < 1.0e-8: continue
                                            # Combined Screening
                                            if screenfactorKL * screenfactorAB < 1.0e-8: continue

                                            dlk = bfs_coeffs[l][lk] 
                                            Nlk = bfs_prim_norms[l][lk]     
                                            
                                            rho = gammaP * gammaQ / (gammaP + gammaQ)
                                            
                                            # Geometry for Rys
                                            P = (alphaik*I + alphajk*J_coord)/gammaP
                                            Q = (alphakk*K + alphalk*L)/gammaQ        
                                            PQ = P - Q
                                            PQsq = np.sum(PQ**2)

                                            tempcoeff7 = tempcoeff6 * dlk * Nlk
                                            
                                            if norder <= 10:
                                                val += tempcoeff7 * coulomb_rys(roots, weights, G, PQsq, rho, norder, n, m, 
                                                                               la, lb, lc, ld, ma, mb, mc, md, na, nb, nc, nd, 
                                                                               alphaik, alphajk, alphakk, alphalk, I, J_coord, K, L)
                            
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

    # --- REDUCTION STEP ---
    # Sum all thread-local matrices into the master J matrix
    # Summing axis 0 (threads) -> (N, N)
    J_final = np.sum(J_local, axis=0)

    # --- SYMMETRIZATION ---
    # We only filled J[i,j] where j<=i (and similarly for k,l). 
    # Now we copy lower triangle to upper triangle.
    for r in prange(nbf_total):
        for c in range(r):
            val = J_final[r, c]
            J_final[c, r] = val

    return J_final

# @njit(parallel=True, cache=True, fastmath=True, error_model="numpy")
# def rys_coulomb_matrix_internal(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, 
#                                  bfs_coeffs, bfs_prim_norms, bfs_expnts, density_matrix,
#                                  sqrt_ints4c2e_diag, isSchwarz=False, threshold=1e-9):
#     """
#     J matrix using parallelized 4-fold symmetry with aggressive screening.
#     """
#     nbf = len(bfs_coords)
#     J_matrix = np.zeros((nbf, nbf), dtype=np.float64)
    
#     # Constants
#     PI = np.pi
#     IJ_SQ_THRESH = 1.0e-8
#     DENS_THRESH = 1e-10
#     THRESH_TIGHT = threshold * 0.1
#     INV_TWO = 0.5
    
#     # Precompute max angular momentum for G buffer sizing
#     max_l_total = 0
#     for i in range(nbf):
#         l_sum = bfs_lmn[i][0] + bfs_lmn[i][1] + bfs_lmn[i][2]
#         if l_sum > max_l_total:
#             max_l_total = l_sum
    
#     max_dim = int(2 * max_l_total + 1)
    
#     # PARALLEL LOOP: Each thread owns J_matrix[i, :]
#     for i in prange(nbf):
#         I = bfs_coords[i]
#         Ni = bfs_contr_prim_norms[i]
#         la, ma, na = bfs_lmn[i]
#         nprimi = bfs_nprim[i]
#         la_tot = la + ma + na
        
#         # Thread-local G buffer (reused across all integrals)
#         G = np.zeros((max_dim, max_dim))
#         roots = np.zeros(10)  # Max order 10
#         weights = np.zeros(10)
        
#         # Precompute i-shell data
#         coeffs_i = bfs_coeffs[i][:nprimi]
#         norms_i = bfs_prim_norms[i][:nprimi]
#         expnts_i = bfs_expnts[i][:nprimi]
#         coeff_norm_i = coeffs_i * norms_i
        
#         for j in range(i + 1):
#             d_val = density_matrix[i, j]
#             abs_d = abs(d_val)
            
#             if abs_d < DENS_THRESH:
#                 # print('here1')
#                 continue
#             J_coord = bfs_coords[j]
#             IJ = I - J_coord
#             IJsq = IJ[0]*IJ[0] + IJ[1]*IJ[1] + IJ[2]*IJ[2]
            
#             Nj = bfs_contr_prim_norms[j]
#             lb, mb, nb = bfs_lmn[j]
#             nprimj = bfs_nprim[j]
#             lb_tot = lb + mb + nb
            
#             # Early screening for ij pair
#             sqrt_ij = 0.0
#             if isSchwarz:
#                 sqrt_ij = sqrt_ints4c2e_diag[i, j]
#                 if sqrt_ij < THRESH_TIGHT:
#                     # print('here0')
#                     continue
            
#             tempcoeff1 = Ni * Nj
#             n_rys = int(max(la + lb, ma + mb, na + nb))
            
#             # Precompute j-shell data
#             coeffs_j = bfs_coeffs[j][:nprimj]
#             norms_j = bfs_prim_norms[j][:nprimj]
#             expnts_j = bfs_expnts[j][:nprimj]
#             coeff_norm_j = coeffs_j * norms_j
            
#             val_ij = 0.0
            
#             # Loop over k shells
#             for k in range(nbf):
#                 K = bfs_coords[k]
#                 Nk = bfs_contr_prim_norms[k]
#                 lc, mc, nc = bfs_lmn[k]
#                 nprimk = bfs_nprim[k]
#                 lc_tot = lc + mc + nc
                
#                 tempcoeff2 = tempcoeff1 * Nk
#                 m_rys = int(max(lc + mc, nc))
                
#                 # Precompute k-shell data
#                 coeffs_k = bfs_coeffs[k][:nprimk]
#                 norms_k = bfs_prim_norms[k][:nprimk]
#                 expnts_k = bfs_expnts[k][:nprimk]
#                 coeff_norm_k = coeffs_k * norms_k
                
#                 for l in range(k + 1):
#                     if isSchwarz:
#                         sqrt_kl = sqrt_ints4c2e_diag[k, l]
#                         schwarz_prod = sqrt_ij * sqrt_kl
                        
#                         if schwarz_prod < threshold:
#                             # print('here1')
#                             continue
                        
#                         d_val = density_matrix[k, l]
#                         abs_d = abs(d_val)
                        
#                         if abs_d < DENS_THRESH:
#                             # print('here1')
#                             continue
#                         if abs_d * schwarz_prod < THRESH_TIGHT:
#                             # print('here2')
#                             continue
#                     else:
#                         d_val = density_matrix[k, l]
                        
#                     L = bfs_coords[l]
#                     KL = K - L
#                     KLsq = KL[0]*KL[0] + KL[1]*KL[1] + KL[2]*KL[2]
                    
#                     Nl = bfs_contr_prim_norms[l]
#                     ld, md, nd = bfs_lmn[l]
#                     npriml = bfs_nprim[l]
#                     ld_tot = ld + md + nd
                    
#                     # Rys order calculation
#                     norder = int((la_tot + lb_tot + lc_tot + ld_tot) * INV_TWO + 1)
#                     if norder > 10:
#                         print("Rys order exceeds maximum of 10.")
#                         # return J_matrix
                    
#                     m_rys_final = int(max(m_rys, lc + ld, mc + md, nc + nd))
                    
#                     # Precompute l-shell data
#                     coeffs_l = bfs_coeffs[l][:npriml]
#                     norms_l = bfs_prim_norms[l][:npriml]
#                     expnts_l = bfs_expnts[l][:npriml]
#                     coeff_norm_l = coeffs_l * norms_l
                    
#                     tempcoeff3 = tempcoeff2 * Nl
#                     integral_val = 0.0
                    
#                     # --- Optimized Primitive Loops ---
#                     for ik in range(nprimi):
#                         alphaik = expnts_i[ik]
#                         p_coeff_ik = tempcoeff3 * coeff_norm_i[ik]
                        
#                         for jk in range(nprimj):
#                             alphajk = expnts_j[jk]
#                             gammaP = alphaik + alphajk
#                             inv_gammaP = 1.0 / gammaP
                            
#                             # Primitive screening with strength reduction
#                             screenfactorAB = np.exp(-alphaik * alphajk * inv_gammaP * IJsq)
#                             if screenfactorAB < IJ_SQ_THRESH:
#                                 continue
                            
#                             # Precompute P center
#                             Px = (alphaik * I[0] + alphajk * J_coord[0]) * inv_gammaP
#                             Py = (alphaik * I[1] + alphajk * J_coord[1]) * inv_gammaP
#                             Pz = (alphaik * I[2] + alphajk * J_coord[2]) * inv_gammaP
                            
#                             p_coeff_jk = p_coeff_ik * coeff_norm_j[jk]
                            
#                             for kk in range(nprimk):
#                                 alphakk = expnts_k[kk]
#                                 p_coeff_kk = p_coeff_jk * coeff_norm_k[kk]
                                
#                                 for lk in range(npriml):
#                                     alphalk = expnts_l[lk]
#                                     gammaQ = alphakk + alphalk
#                                     inv_gammaQ = 1.0 / gammaQ
                                    
#                                     screenfactorKL = np.exp(-alphakk * alphalk * inv_gammaQ * KLsq)
#                                     if screenfactorKL < IJ_SQ_THRESH:
#                                         continue
                                    
#                                     # Combined screening
#                                     combined_screen = screenfactorAB * screenfactorKL
#                                     if combined_screen < IJ_SQ_THRESH:
#                                         continue
                                    
#                                     # Precompute Q center
#                                     Qx = (alphakk * K[0] + alphalk * L[0]) * inv_gammaQ
#                                     Qy = (alphakk * K[1] + alphalk * L[1]) * inv_gammaQ
#                                     Qz = (alphakk * K[2] + alphalk * L[2]) * inv_gammaQ
                                    
#                                     # PQ distance squared
#                                     PQx = Px - Qx
#                                     PQy = Py - Qy
#                                     PQz = Pz - Qz
#                                     PQsq = PQx*PQx + PQy*PQy + PQz*PQz
                                    
#                                     rho = gammaP * gammaQ / (gammaP + gammaQ)
#                                     final_coeff = p_coeff_kk * coeff_norm_l[lk]
                                    
#                                     # Rys quadrature kernel
#                                     integral_val += final_coeff * coulomb_rys(
#                                         roots, weights, G, PQsq, rho, norder, 
#                                         n_rys, m_rys_final,
#                                         la, lb, lc, ld, ma, mb, mc, md, na, nb, nc, nd,
#                                         alphaik, alphajk, alphakk, alphalk, 
#                                         I, J_coord, K, L
#                                     )
                    
#                     # Accumulate with symmetry factor
#                     d_factor = d_val * (2.0 if k != l else 1.0)
#                     val_ij += d_factor * integral_val
            
#             J_matrix[i, j] = val_ij
    
#     # Symmetrize (copy lower to upper triangle)
#     for i in prange(nbf):
#         for j in range(i):
#             J_matrix[j, i] = J_matrix[i, j]
    
#     return J_matrix

# @njit(parallel=True, cache=True, fastmath=True, error_model="numpy")
# def rys_coulomb_matrix_internal(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, 
#                                  bfs_coeffs, bfs_prim_norms, bfs_expnts, density_matrix,
#                                  sqrt_ints4c2e_diag, isSchwarz=False, threshold=1e-9):
#     """
#     Computes J matrix using parallelized 4-fold symmetry.
    
#     Optimization Strategy:
#     - Parallelizes over shell 'i' to ensure thread ownership of J[i,:] (No race conditions).
#     - Iterates over all k, l for each i, j to complete the integral sum locally.
#     """
#     nbf = len(bfs_coords)
#     J_matrix = np.zeros((nbf, nbf), dtype=np.float64)
    
#     pi = np.pi
#     IJ_sq_thresh = 1.0e-8
#     dens_thresh = 1e-12
    
#     # PARALLEL LOOP: Thread 'i' exclusively owns writes to J_matrix[i, :]
#     for i in prange(nbf):
#         I = bfs_coords[i]
#         Ni = bfs_contr_prim_norms[i]
#         la, ma, na = bfs_lmn[i]
#         nprimi = bfs_nprim[i]
        
#         for j in range(i + 1):
#             J_coord = bfs_coords[j]
#             IJ = I - J_coord
#             IJsq = np.sum(IJ**2)
#             Nj = bfs_contr_prim_norms[j]
#             lb, mb, nb = bfs_lmn[j]
#             nprimj = bfs_nprim[j]
            
#             # Precompute ij screening value
#             sqrt_ij = 0.0
#             if isSchwarz:
#                 sqrt_ij = sqrt_ints4c2e_diag[i, j]

#             # Precompute coefficients for ij pair
#             tempcoeff1 = Ni * Nj
            
#             # Local accumulator for J[i, j]
#             val_ij = 0.0
            
#             # INNER LOOPS: Run over ALL k, l to sum contributions for J[i, j]
#             # (4-fold symmetry approach: safer and more cache-efficient for parallel CPU)
#             for k in range(nbf):
#                 K = bfs_coords[k]
#                 Nk = bfs_contr_prim_norms[k]
#                 lc, mc, nc = bfs_lmn[k]
#                 nprimk = bfs_nprim[k]
                
#                 # Precompute coefficient partial
#                 tempcoeff2 = tempcoeff1 * Nk
                
#                 for l in range(k + 1):
#                     # --- Screening Checks ---
#                     if isSchwarz:
#                         sqrt_kl = sqrt_ints4c2e_diag[k, l]
#                         # 1. Integral magnitude screen
#                         if sqrt_ij * sqrt_kl < threshold:
#                             continue
#                         # 2. Density weighted screen
#                         d_val = density_matrix[k, l]
#                         if abs(d_val) * sqrt_ij * sqrt_kl < threshold * 0.1:
#                             continue
#                         # 3. Pure density screen
#                         if abs(d_val) < dens_thresh:
#                             continue

#                     L = bfs_coords[l]
#                     KL = K - L
#                     KLsq = np.sum(KL**2)
#                     Nl = bfs_contr_prim_norms[l]
#                     ld, md, nd = bfs_lmn[l]
#                     npriml = bfs_nprim[l]
                    
#                     # Rys Quadrature Order Setup
#                     # norder = (L_total / 2) + 1
#                     norder = int((la + ma + na + lb + mb + nb + lc + mc + nc + ld + md + nd) * 0.5 + 1)
                    
#                     # Pre-allocate Rys scratch arrays (small stack allocation)
#                     roots = np.zeros(norder)
#                     weights = np.zeros(norder)
#                     # Dimensions for G matrix
#                     n_rys = int(max(la + lb, ma + mb, na + nb))
#                     m_rys = int(max(lc + ld, mc + md, nc + nd))
#                     G = np.zeros((n_rys + 1, m_rys + 1))

#                     tempcoeff3 = tempcoeff2 * Nl
#                     integral_val = 0.0
                    
#                     # --- Primitive Loops ---
#                     for ik in range(nprimi):
#                         alphaik = bfs_expnts[i][ik]
#                         # Precompute partial primitive coeff
#                         p_coeff_ik = tempcoeff3 * bfs_coeffs[i][ik] * bfs_prim_norms[i][ik]
                        
#                         for jk in range(nprimj):
#                             alphajk = bfs_expnts[j][jk]
#                             gammaP = alphaik + alphajk
#                             inv_gammaP = 1.0 / gammaP
                            
#                             # Primitive Screening AB
#                             screenfactorAB = np.exp(-alphaik * alphajk * inv_gammaP * IJsq)
#                             if abs(screenfactorAB) < IJ_sq_thresh:
#                                 continue
                                
#                             # Precompute P coordinate
#                             P = (alphaik * I + alphajk * J_coord) * inv_gammaP
#                             p_coeff_jk = p_coeff_ik * bfs_coeffs[j][jk] * bfs_prim_norms[j][jk]
                            
#                             for kk in range(nprimk):
#                                 alphakk = bfs_expnts[k][kk]
#                                 p_coeff_kk = p_coeff_jk * bfs_coeffs[k][kk] * bfs_prim_norms[k][kk]
                                
#                                 for lk in range(npriml):
#                                     alphalk = bfs_expnts[l][lk]
#                                     gammaQ = alphakk + alphalk
#                                     inv_gammaQ = 1.0 / gammaQ
                                    
#                                     # Primitive Screening KL
#                                     screenfactorKL = np.exp(-alphakk * alphalk * inv_gammaQ * KLsq)
#                                     if abs(screenfactorKL) < IJ_sq_thresh:
#                                         continue
                                        
#                                     # Combined Screening
#                                     if abs(screenfactorAB * screenfactorKL) < IJ_sq_thresh:
#                                         continue
                                    
#                                     Q = (alphakk * K + alphalk * L) * inv_gammaQ
#                                     PQ = P - Q
#                                     PQsq = np.sum(PQ**2)
#                                     rho = gammaP * gammaQ / (gammaP + gammaQ)
                                    
#                                     final_coeff = p_coeff_kk * bfs_coeffs[l][lk] * bfs_prim_norms[l][lk]
                                    
#                                     # Call Rys Quadrature Kernel
#                                     # Note: Assumes coulomb_rys is available in the scope
#                                     if norder <= 10: 
#                                         integral_val += final_coeff * coulomb_rys(
#                                             roots, weights, G, PQsq, rho, norder, 
#                                             n_rys, m_rys,
#                                             la, lb, lc, ld, ma, mb, mc, md, na, nb, nc, nd,
#                                             alphaik, alphajk, alphakk, alphalk, 
#                                             I, J_coord, K, L
#                                         )

#                     # --- Accumulate to J[i,j] ---
#                     # Account for Density Symmetry: D[k,l] == D[l,k]
#                     # If k != l, we add (ij|kl)*D[kl] + (ij|lk)*D[lk] = 2 * (ij|kl)*D[kl]
#                     d_factor = density_matrix[k, l]
#                     if k != l:
#                         d_factor *= 2.0
                        
#                     val_ij += d_factor * integral_val
            
#             # Write result to J matrix (Safe: i is unique to thread)
#             J_matrix[i, j] = val_ij

#     # Symmetrize result (copy lower triangle to upper)
#     for i in prange(nbf):
#         for j in range(i):
#             J_matrix[j, i] = J_matrix[i, j]
            
#     return J_matrix

# import numpy as np
# from numba import njit, prange, get_num_threads, get_thread_id

# @njit(parallel=True, cache=True, fastmath=True, error_model="numpy")
# def rys_coulomb_matrix_internal(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, 
#                                 bfs_coeffs, bfs_prim_norms, bfs_expnts, density_matrix,
#                                 sqrt_ints4c2e_diag, isSchwarz=False, threshold=1e-9):
#     """
#     Computes J matrix exploiting full 8-fold symmetry using thread-local accumulation.
#     """
#     nbf = len(bfs_coords)
    
#     # 1. Allocate thread-local buffers to avoid race conditions when writing J[k, l]
#     # Numba requires knowing the number of threads to allocate the buffer.
#     n_threads = get_num_threads()
#     J_buffers = np.zeros((n_threads, nbf, nbf), dtype=np.float64)
    
#     pi = np.pi
#     IJ_sq_thresh = 1.0e-8 # Pre-define small constants
    
#     # Loop over basis functions
#     for i in prange(nbf):
#         tid = get_thread_id() # Get current thread ID for local buffer access
        
#         I = bfs_coords[i]
#         Ni = bfs_contr_prim_norms[i]
#         lmni = bfs_lmn[i]
#         la, ma, na = lmni
#         nprimi = bfs_nprim[i]
        
#         for j in range(i + 1):
#             if isSchwarz:
#                 sqrt_ij = sqrt_ints4c2e_diag[i, j]
            
#             J = bfs_coords[j]
#             IJ = I - J
#             IJsq = np.sum(IJ**2)
#             Nj = bfs_contr_prim_norms[j]
#             lmnj = bfs_lmn[j]
#             lb, mb, nb = lmnj
#             tempcoeff1 = Ni * Nj
#             nprimj = bfs_nprim[j]
            
#             # 8-FOLD SYMMETRY FIX:
#             # Restrict k-loop to run only up to i.
#             # If k == i, restrict l-loop to run only up to j.
#             # This ensures CompositeIndex(ij) >= CompositeIndex(kl)
            
#             for k in range(i + 1):
#                 # Determine the upper limit for l based on 8-fold logic
#                 l_limit = k + 1
#                 if k == i:
#                     l_limit = j + 1
                
#                 K = bfs_coords[k]
#                 Nk = bfs_contr_prim_norms[k]
#                 lmnk = bfs_lmn[k]
#                 lc, mc, nc = lmnk
#                 tempcoeff2 = tempcoeff1 * Nk
#                 nprimk = bfs_nprim[k]
                
#                 for l in range(l_limit):
#                     if isSchwarz:
#                         sqrt_kl = sqrt_ints4c2e_diag[k, l]
#                         # Screen using max density element or simply integral bounds
#                         # Note: Screening logic might need adjustment depending on Density input properties
#                         if sqrt_ij * sqrt_kl < threshold:
#                             continue
#                         if abs(density_matrix[k,l]) * sqrt_ij * sqrt_kl < threshold*0.1:
#                             continue
#                         if abs(density_matrix[k,l]) < 1e-12:
#                             continue

#                     L = bfs_coords[l]
#                     KL = K - L
#                     KLsq = np.sum(KL**2)
#                     Nl = bfs_contr_prim_norms[l]
#                     lmnl = bfs_lmn[l]
#                     ld, md, nd = lmnl
#                     tempcoeff3 = tempcoeff2 * Nl
#                     npriml = bfs_nprim[l]
                    
#                     # --- Rys Quadrature Setup ---
#                     norder = int((la + ma + na + lb + mb + nb + lc + mc + nc + ld + md + nd) / 2 + 1)
#                     n = int(max(la + lb, ma + mb, na + nb))
#                     m = int(max(lc + ld, mc + md, nc + nd))
#                     roots = np.zeros(norder)
#                     weights = np.zeros(norder)
#                     G = np.zeros((n + 1, m + 1))
                    
#                     val = 0.0
                    
#                     # Loop over primitives
#                     for ik in range(nprimi):
#                         dik = bfs_coeffs[i][ik]
#                         Nik = bfs_prim_norms[i][ik]
#                         alphaik = bfs_expnts[i][ik]
#                         tempcoeff4 = tempcoeff3 * dik * Nik
                        
#                         for jk in range(nprimj):
#                             alphajk = bfs_expnts[j][jk]
#                             gammaP = alphaik + alphajk
#                             screenfactorAB = np.exp(-alphaik * alphajk / gammaP * IJsq)
#                             if abs(screenfactorAB) < IJ_sq_thresh:
#                                 continue
                            
#                             djk = bfs_coeffs[j][jk]
#                             Njk = bfs_prim_norms[j][jk]
#                             P = (alphaik * I + alphajk * J) / gammaP
#                             tempcoeff5 = tempcoeff4 * djk * Njk
                            
#                             for kk in range(nprimk):
#                                 dkk = bfs_coeffs[k][kk]
#                                 Nkk = bfs_prim_norms[k][kk]
#                                 alphakk = bfs_expnts[k][kk]
#                                 tempcoeff6 = tempcoeff5 * dkk * Nkk
                                
#                                 for lk in range(npriml):
#                                     alphalk = bfs_expnts[l][lk]
#                                     gammaQ = alphakk + alphalk
#                                     screenfactorKL = np.exp(-alphakk * alphalk / gammaQ * KLsq)
#                                     if abs(screenfactorKL) < IJ_sq_thresh:
#                                         continue
#                                     if abs(screenfactorAB * screenfactorKL) < IJ_sq_thresh:
#                                         continue
                                    
#                                     dlk = bfs_coeffs[l][lk]
#                                     Nlk = bfs_prim_norms[l][lk]
#                                     Q = (alphakk * K + alphalk * L) / gammaQ
#                                     PQ = P - Q
#                                     PQsq = np.sum(PQ**2)
#                                     rho = gammaP * gammaQ / (gammaP + gammaQ)
#                                     tempcoeff7 = tempcoeff6 * dlk * Nlk
                                    
#                                     if norder <= 10:
#                                         val += tempcoeff7 * coulomb_rys(
#                                             roots, weights, G, PQsq, rho, norder, n, m,
#                                             la, lb, lc, ld, ma, mb, mc, md, na, nb, nc, nd,
#                                             alphaik, alphajk, alphakk, alphalk, I, J, K, L
#                                         )
                    
#                     # --- 8-Fold Update Logic ---
                    
#                     # 1. Update J[i, j] using Density[k, l]
#                     # Account for off-diagonal density symmetry D[k,l] == D[l,k]
#                     d_factor_kl = density_matrix[k, l] if k == l else 2.0 * density_matrix[k, l]
#                     J_buffers[tid, i, j] += d_factor_kl * val
                    
#                     # 2. Update J[k, l] using Density[i, j] IF (ij) != (kl)
#                     # This is the step that requires thread safety (hence J_buffers)
#                     ij_idx = i * (i + 1) // 2 + j
#                     kl_idx = k * (k + 1) // 2 + l
                    
#                     if ij_idx != kl_idx:
#                         d_factor_ij = density_matrix[i, j] if i == j else 2.0 * density_matrix[i, j]
#                         J_buffers[tid, k, l] += d_factor_ij * val

#     # Reduction: Sum all thread-local matrices into the final result
#     J_matrix = np.sum(J_buffers, axis=0)
    
#     # Symmetrize the result (since we only filled J[i,j] where j<=i)
#     # Copy lower triangle to upper triangle
#     for i in prange(nbf):
#         for j in range(i):
#             val = J_matrix[i, j]
#             J_matrix[j, i] = val
            
#     return J_matrix
# @njit(parallel=True, cache=True, fastmath=True, error_model="numpy")
# def rys_coulomb_matrix_internal(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, 
#                                 bfs_coeffs, bfs_prim_norms, bfs_expnts, density_matrix,
#                                 sqrt_ints4c2e_diag, isSchwarz=False, threshold=1e-9):
#     """
#     Internal Numba-accelerated function to compute the Coulomb matrix directly.
    
#     Computes J[i,j] = Σ_kl D[k,l] * (ij|kl) without storing the full ERI tensor.
#     """
#     nbf = len(bfs_coords)
#     J_matrix = np.zeros((nbf, nbf), dtype=np.float64)
    
#     pi = np.pi
#     pisq = pi * pi
#     twopisq = 2.0 * pi * pi
    
#     # Loop over basis functions with symmetry
#     for i in prange(nbf):
#         I = bfs_coords[i]
#         Ni = bfs_contr_prim_norms[i]
#         lmni = bfs_lmn[i]
#         la, ma, na = lmni
#         nprimi = bfs_nprim[i]
        
#         for j in range(i + 1):  # j <= i (use symmetry)
#             if isSchwarz:
#                 sqrt_ij = sqrt_ints4c2e_diag[i, j]
            
#             J = bfs_coords[j]
#             IJ = I - J
#             IJsq = np.sum(IJ**2)
#             Nj = bfs_contr_prim_norms[j]
#             lmnj = bfs_lmn[j]
#             lb, mb, nb = lmnj
#             tempcoeff1 = Ni * Nj
#             nprimj = bfs_nprim[j]
            
#             J_ij = 0.0
            
#             for k in range(nbf):
#                 K = bfs_coords[k]
#                 Nk = bfs_contr_prim_norms[k]
#                 lmnk = bfs_lmn[k]
#                 lc, mc, nc = lmnk
#                 tempcoeff2 = tempcoeff1 * Nk
#                 nprimk = bfs_nprim[k]
                
#                 for l in range(k + 1):  # l <= k (use symmetry)
#                     if isSchwarz:
#                         sqrt_kl = sqrt_ints4c2e_diag[k, l]
#                         if abs(density_matrix[k,l]) * sqrt_ij * sqrt_kl < threshold:
#                         # if sqrt_ij * sqrt_kl < threshold:
#                             continue
                    
#                     L = bfs_coords[l]
#                     KL = K - L
#                     KLsq = np.sum(KL**2)
#                     Nl = bfs_contr_prim_norms[l]
#                     lmnl = bfs_lmn[l]
#                     ld, md, nd = lmnl
#                     tempcoeff3 = tempcoeff2 * Nl
#                     npriml = bfs_nprim[l]
                    
#                     norder = int((la + ma + na + lb + mb + nb + lc + mc + nc + ld + md + nd) / 2 + 1)
#                     n = int(max(la + lb, ma + mb, na + nb))
#                     m = int(max(lc + ld, mc + md, nc + nd))
#                     roots = np.zeros(norder)
#                     weights = np.zeros(norder)
#                     G = np.zeros((n + 1, m + 1))
                    
#                     val = 0.0
                    
#                     # Loop over primitives
#                     for ik in range(nprimi):
#                         dik = bfs_coeffs[i][ik]
#                         Nik = bfs_prim_norms[i][ik]
#                         alphaik = bfs_expnts[i][ik]
#                         tempcoeff4 = tempcoeff3 * dik * Nik
                        
#                         for jk in range(nprimj):
#                             alphajk = bfs_expnts[j][jk]
#                             gammaP = alphaik + alphajk
#                             screenfactorAB = np.exp(-alphaik * alphajk / gammaP * IJsq)
#                             if abs(screenfactorAB) < 1.0e-8:
#                                 continue
                            
#                             djk = bfs_coeffs[j][jk]
#                             Njk = bfs_prim_norms[j][jk]
#                             P = (alphaik * I + alphajk * J) / gammaP
#                             tempcoeff5 = tempcoeff4 * djk * Njk
                            
#                             for kk in range(nprimk):
#                                 dkk = bfs_coeffs[k][kk]
#                                 Nkk = bfs_prim_norms[k][kk]
#                                 alphakk = bfs_expnts[k][kk]
#                                 tempcoeff6 = tempcoeff5 * dkk * Nkk
                                
#                                 for lk in range(npriml):
#                                     alphalk = bfs_expnts[l][lk]
#                                     gammaQ = alphakk + alphalk
#                                     screenfactorKL = np.exp(-alphakk * alphalk / gammaQ * KLsq)
#                                     if abs(screenfactorKL) < 1.0e-8:
#                                         continue
#                                     if abs(screenfactorAB * screenfactorKL) < 1.0e-10:
#                                         continue
                                    
#                                     dlk = bfs_coeffs[l][lk]
#                                     Nlk = bfs_prim_norms[l][lk]
#                                     Q = (alphakk * K + alphalk * L) / gammaQ
#                                     PQ = P - Q
#                                     PQsq = np.sum(PQ**2)
#                                     rho = gammaP * gammaQ / (gammaP + gammaQ)
#                                     tempcoeff7 = tempcoeff6 * dlk * Nlk
                                    
#                                     if norder <= 10:
#                                         val += tempcoeff7 * coulomb_rys(
#                                             roots, weights, G, PQsq, rho, norder, n, m,
#                                             la, lb, lc, ld, ma, mb, mc, md, na, nb, nc, nd,
#                                             alphaik, alphajk, alphakk, alphalk, I, J, K, L
#                                         )
#                                     else:
#                                         print("Error: Rys quadrature order greater than 10 not implemented.")
                    
#                     # Contract with density matrix
#                     # Account for symmetry: D[k,l] = D[l,k]
#                     if k == l:
#                         J_ij += density_matrix[k, l] * val
#                     else:
#                         J_ij += 2.0 * density_matrix[k, l] * val
            
#             # Store result with symmetry
#             J_matrix[i, j] = J_ij
#             if i != j:
#                 J_matrix[j, i] = J_ij
    
#     return J_matrix