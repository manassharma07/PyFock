import numpy as np
from numba import njit, prange
from .integral_helpers import calcS

def kin_mat_symm(basis, slice=None):
    """
    Compute the kinetic energy matrix for a given basis set.

    This function evaluates the one-electron kinetic energy integrals 
    ⟨χ_i | -½∇² | χ_j⟩ for a set of contracted Gaussian basis functions defined in `basis`. 
    It uses an optimized backend with improved memory layout, early termination, 
    and partial vectorization for enhanced performance.

    Block-wise computation is supported through the optional `slice` parameter.

    Parameters
    ----------
    basis : object
        Basis set object containing properties such as:
        - bfs_coords: Cartesian coordinates of basis function centers.
        - bfs_coeffs: Contraction coefficients.
        - bfs_expnts: Gaussian exponents.
        - bfs_prim_norms: Normalization constants for primitives.
        - bfs_contr_prim_norms: Normalization constants for contracted functions.
        - bfs_lmn: Angular momentum quantum numbers (ℓ, m, n).
        - bfs_nprim: Number of primitives per basis function.
        - bfs_nao: Number of atomic orbitals (contracted basis functions).

    slice : list of int, optional
        A 4-element list `[start_row, end_row, start_col, end_col]` specifying the 
        matrix block to compute. Rows and columns refer to AOs. If `None` (default), 
        the entire kinetic energy matrix is calculated.

    Returns
    -------
    T : ndarray of shape (end_row - start_row, end_col - start_col)
        The computed (sub)matrix of kinetic energy integrals.

    Notes
    -----
    Internally, this function:
    - Pads coefficient/exponent arrays for uniform shape and Numba compatibility.
    - Reduces redundant operations and loops.
    - Utilizes an optimized `kin_mat_symm_internal_optimized` backend.

    Examples
    --------
    >>> T = kin_mat_symm(basis)
    >>> T_block = kin_mat_symm(basis, slice=[0, 10, 0, 10])
    """
    
    # Convert to numpy arrays (same as before)
    bfs_coords = np.array(basis.bfs_coords)  # Remove unnecessary list wrapping
    bfs_contr_prim_norms = np.array(basis.bfs_contr_prim_norms)
    bfs_lmn = np.array(basis.bfs_lmn)
    bfs_nprim = np.array(basis.bfs_nprim)
    
    # Optimize the coefficient/exponent arrays
    maxnprim = max(basis.bfs_nprim)
    bfs_coeffs = np.zeros((basis.bfs_nao, maxnprim))
    bfs_expnts = np.zeros((basis.bfs_nao, maxnprim))
    bfs_prim_norms = np.zeros((basis.bfs_nao, maxnprim))
    
    # Vectorized assignment (faster than nested loops)
    for i in range(basis.bfs_nao):
        n = basis.bfs_nprim[i]
        bfs_coeffs[i, :n] = basis.bfs_coeffs[i]
        bfs_expnts[i, :n] = basis.bfs_expnts[i]
        bfs_prim_norms[i, :n] = basis.bfs_prim_norms[i]
    
    if slice is None:
        slice = [0, basis.bfs_nao, 0, basis.bfs_nao]
    
    a, b, c, d = map(int, slice)
    
    T = kin_mat_symm_internal_optimized(
        bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, 
        bfs_coeffs, bfs_prim_norms, bfs_expnts, a, b, c, d
    )
    
    return T

@njit(parallel=True, cache=True, fastmath=True)  # Added fastmath for better performance
def kin_mat_symm_internal_optimized(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, 
                                   bfs_coeffs, bfs_prim_norms, bfs_expnts, 
                                   start_row, end_row, start_col, end_col):
    """
    Optimized internal function with several improvements:
    1. Pre-compute commonly used values
    2. Better loop structure
    3. Early termination conditions
    4. Reduced memory allocations
    """
    
    num_rows = end_row - start_row 
    num_cols = end_col - start_col 
    
    # Determine triangle type (same logic as before)
    upper_tri = end_row <= start_col
    lower_tri = start_row >= end_col
    both_tri_symm = start_row == start_col and end_row == end_col
    both_tri_nonsymm = not (upper_tri or lower_tri or both_tri_symm)
    
    T = np.zeros((num_rows, num_cols))
    
    # Pre-compute some constants
    CUTOFF = 1.0e-9  # Threshold for early termination
    
    for i in prange(start_row, end_row):
        I = bfs_coords[i]
        Ni = bfs_contr_prim_norms[i]
        lmni = bfs_lmn[i]
        nprim_i = bfs_nprim[i]
        
        # Pre-compute for this row
        lmni_0, lmni_1, lmni_2 = lmni[0], lmni[1], lmni[2]
        
        for j in range(start_col, end_col):
            if not (lower_tri or upper_tri or (both_tri_symm and j <= i) or both_tri_nonsymm):
                continue
                
            J = bfs_coords[j]
            IJ = I - J
            Nj = bfs_contr_prim_norms[j]
            lmnj = bfs_lmn[j]
            nprim_j = bfs_nprim[j]
            
            # Pre-compute commonly used values
            lmnj_0, lmnj_1, lmnj_2 = lmnj[0], lmnj[1], lmnj[2]
            fac1 = np.sum(IJ * IJ)  # Slightly faster than IJ**2
            fac2 = Ni * Nj
            fac3 = 2 * (lmnj_0 + lmnj_1 + lmnj_2) + 3
            fac4 = lmnj_0 * (lmnj_0 - 1)
            fac5 = lmnj_1 * (lmnj_1 - 1)
            fac6 = lmnj_2 * (lmnj_2 - 1)
            
            result_sum = 0.0
            
            for ik in range(nprim_i):  # Use actual number of primitives
                alphaik = bfs_expnts[i, ik]
                dik = bfs_coeffs[i, ik]
                Nik = bfs_prim_norms[i, ik]
                
                for jk in range(nprim_j):  # Use actual number of primitives
                    alphajk = bfs_expnts[j, jk]
                    gamma = alphaik + alphajk
                    
                    # Early termination check
                    temp_1 = np.exp(-alphaik * alphajk / gamma * fac1)
                    if temp_1 < CUTOFF:
                        continue
                    
                    djk = bfs_coeffs[j, jk]
                    Njk = bfs_prim_norms[j, jk]
                    
                    # Pre-compute P, PI, PJ once
                    gamma_inv = 1.0 / gamma
                    P = (alphaik * I + alphajk * J) * gamma_inv
                    PI = P - I
                    PJ = P - J
                    
                    # Pre-compute coefficient term
                    temp_coeff = (dik * djk * Nik * Njk * fac2 * temp_1)
                    
                    # Calculate overlaps more efficiently
                    # Group similar calculations together
                    PI_0, PI_1, PI_2 = PI[0], PI[1], PI[2]
                    PJ_0, PJ_1, PJ_2 = PJ[0], PJ[1], PJ[2]
                    
                    # Base overlap
                    Sx_base = calcS(lmni_0, lmnj_0, gamma, PI_0, PJ_0)
                    Sy_base = calcS(lmni_1, lmnj_1, gamma, PI_1, PJ_1)
                    Sz_base = calcS(lmni_2, lmnj_2, gamma, PI_2, PJ_2)
                    overlap1 = Sx_base * Sy_base * Sz_base
                    
                    # +2 overlaps
                    Sx_p2 = calcS(lmni_0, lmnj_0 + 2, gamma, PI_0, PJ_0)
                    Sy_p2 = calcS(lmni_1, lmnj_1 + 2, gamma, PI_1, PJ_1)
                    Sz_p2 = calcS(lmni_2, lmnj_2 + 2, gamma, PI_2, PJ_2)
                    
                    overlap2 = Sx_p2 * Sy_base * Sz_base
                    overlap3 = Sx_base * Sy_p2 * Sz_base
                    overlap4 = Sx_base * Sy_base * Sz_p2
                    
                    # -2 overlaps (only if needed)
                    overlap5 = overlap6 = overlap7 = 0.0
                    if fac4 != 0.0:
                        Sx_m2 = calcS(lmni_0, lmnj_0 - 2, gamma, PI_0, PJ_0)
                        overlap5 = Sx_m2 * Sy_base * Sz_base
                    if fac5 != 0.0:
                        Sy_m2 = calcS(lmni_1, lmnj_1 - 2, gamma, PI_1, PJ_1)
                        overlap6 = Sx_base * Sy_m2 * Sz_base
                    if fac6 != 0.0:
                        Sz_m2 = calcS(lmni_2, lmnj_2 - 2, gamma, PI_2, PJ_2)
                        overlap7 = Sx_base * Sy_base * Sz_m2
                    
                    # Combine terms
                    part1 = overlap1 * alphajk * fac3
                    part2 = 2 * alphajk * alphajk * (overlap2 + overlap3 + overlap4)
                    part3 = 0.5 * (fac4 * overlap5 + fac5 * overlap6 + fac6 * overlap7)
                    
                    result = temp_coeff * (part1 - part2 - part3)
                    result_sum += result
            
            T[i - start_row, j - start_col] = result_sum
    
    # Handle symmetry
    if both_tri_symm:
        for i in prange(start_row, end_row):
            for j in range(start_col, min(i, end_col)):  # Only fill upper triangle
                T[j - start_col, i - start_row] = T[i - start_row, j - start_col]
    
    return T

# Alternative: Consider using a block-based approach for very large matrices
@njit(parallel=True, cache=True, fastmath=True)
def kin_mat_symm_blocked(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim,
                        bfs_coeffs, bfs_prim_norms, bfs_expnts,
                        start_row, end_row, start_col, end_col, block_size=64):
    """
    Block-based computation for better cache locality with large matrices.
    This can be particularly beneficial for very large basis sets.
    """
    num_rows = end_row - start_row
    num_cols = end_col - start_col
    T = np.zeros((num_rows, num_cols))
    
    # Process in blocks for better cache locality
    for i_block in range(0, num_rows, block_size):
        for j_block in range(0, num_cols, block_size):
            i_end = min(i_block + block_size, num_rows)
            j_end = min(j_block + block_size, num_cols)
            
            # Call the optimized function for this block
            block_result = kin_mat_symm_internal_optimized(
                bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim,
                bfs_coeffs, bfs_prim_norms, bfs_expnts,
                start_row + i_block, start_row + i_end,
                start_col + j_block, start_col + j_end
            )
            
            T[i_block:i_end, j_block:j_end] = block_result
    
    return T








######### SHELL BASED IMPLEMENTATION (very slow for ano-rcc for some reason https://claude.ai/chat/b28b3466-9530-4c68-9f3c-758a2c8b1a37)






# import numpy as np
# from numba import njit, prange
# from .integral_helpers import calcS

# def kin_mat_symm(basis, slice=None):
#     """
#     Shell-based kinetic energy matrix calculation.
#     This approach groups basis functions by shells (same center, same angular momentum)
#     to avoid recomputing primitive integrals for functions that differ only in orientation.
#     """
    
#     # First, we need to organize basis functions into shells
#     shells_info = organize_basis_into_shells(basis)
    
#     if slice is None:
#         slice = [0, basis.bfs_nao, 0, basis.bfs_nao]
    
#     a, b, c, d = map(int, slice)
    
#     T = kin_mat_shell_internal(*shells_info, a, b, c, d)
    
#     return T

# def organize_basis_into_shells(basis):
#     """
#     Organize basis functions into shells and return Numba-compatible arrays.
#     """
#     shells = []
#     shell_map = {}
    
#     for ibf in range(basis.bfs_nao):
#         # Create a shell identifier
#         center = tuple(basis.bfs_coords[ibf])
#         L = sum(basis.bfs_lmn[ibf])  # Total angular momentum
        
#         # Primitive data (exponents and coefficients should be identical for same shell)
#         prim_key = (tuple(basis.bfs_expnts[ibf]), tuple(basis.bfs_coeffs[ibf]))
        
#         shell_key = (center, L, prim_key)
        
#         if shell_key not in shell_map:
#             # Create new shell
#             shell_idx = len(shells)
#             shell_info = {
#                 'center': np.array(center),
#                 'L': L,
#                 'exponents': np.array(basis.bfs_expnts[ibf]),
#                 'coefficients': np.array(basis.bfs_coeffs[ibf]),
#                 'prim_norms': np.array(basis.bfs_prim_norms[ibf]),
#                 'contr_norm': basis.bfs_contr_prim_norms[ibf],
#                 'nprim': basis.bfs_nprim[ibf],
#                 'bf_indices': [],
#                 'lmn_list': []
#             }
#             shells.append(shell_info)
#             shell_map[shell_key] = shell_idx
        
#         shell_idx = shell_map[shell_key]
#         shells[shell_idx]['bf_indices'].append(ibf)
#         shells[shell_idx]['lmn_list'].append(basis.bfs_lmn[ibf])
    
#     # Convert to Numba-compatible arrays
#     nshells = len(shells)
#     nbf_total = basis.bfs_nao
    
#     # Find maximum dimensions
#     max_nprim = max(shell['nprim'] for shell in shells)
#     max_nbf_in_shell = max(len(shell['bf_indices']) for shell in shells)
    
#     # Create arrays that Numba can handle
#     shell_centers = np.zeros((nshells, 3))
#     shell_L = np.zeros(nshells, dtype=np.int32)
#     shell_nprim = np.zeros(nshells, dtype=np.int32)
#     shell_nbf = np.zeros(nshells, dtype=np.int32)
#     shell_exponents = np.zeros((nshells, max_nprim))
#     shell_coefficients = np.zeros((nshells, max_nprim))
#     shell_prim_norms = np.zeros((nshells, max_nprim))
#     shell_contr_norms = np.zeros(nshells)
#     shell_lmn = np.zeros((nshells, max_nbf_in_shell, 3), dtype=np.int32)
#     shell_bf_indices = np.zeros((nshells, max_nbf_in_shell), dtype=np.int32)
    
#     # Mapping arrays
#     bf_to_shell = np.zeros(nbf_total, dtype=np.int32)
#     bf_to_shell_func = np.zeros(nbf_total, dtype=np.int32)
    
#     for ishell, shell in enumerate(shells):
#         shell_centers[ishell] = shell['center']
#         shell_L[ishell] = shell['L']
#         shell_nprim[ishell] = shell['nprim']
#         shell_nbf[ishell] = len(shell['bf_indices'])
#         shell_contr_norms[ishell] = shell['contr_norm']
        
#         nprim = shell['nprim']
#         shell_exponents[ishell, :nprim] = shell['exponents']
#         shell_coefficients[ishell, :nprim] = shell['coefficients']
#         shell_prim_norms[ishell, :nprim] = shell['prim_norms']
        
#         nbf = len(shell['bf_indices'])
#         for i in range(nbf):
#             shell_lmn[ishell, i] = shell['lmn_list'][i]
#             shell_bf_indices[ishell, i] = shell['bf_indices'][i]
            
#             # Fill mapping arrays
#             bf_idx = shell['bf_indices'][i]
#             bf_to_shell[bf_idx] = ishell
#             bf_to_shell_func[bf_idx] = i
    
#     return (nshells, shell_centers, shell_L, shell_nprim, shell_nbf,
#             shell_exponents, shell_coefficients, shell_prim_norms, shell_contr_norms,
#             shell_lmn, shell_bf_indices, bf_to_shell, bf_to_shell_func,
#             max_nprim, max_nbf_in_shell)

# @njit(parallel=True, cache=True, fastmath=True)
# def kin_mat_shell_internal(nshells, shell_centers, shell_L, shell_nprim, shell_nbf,
#                           shell_exponents, shell_coefficients, shell_prim_norms, shell_contr_norms,
#                           shell_lmn, shell_bf_indices, bf_to_shell, bf_to_shell_func,
#                           max_nprim, max_nbf_in_shell, start_row, end_row, start_col, end_col):
#     """
#     Shell-based kinetic energy matrix calculation - Numba optimized version.
#     """
    
#     num_rows = end_row - start_row
#     num_cols = end_col - start_col
#     T = np.zeros((num_rows, num_cols))
    
#     CUTOFF = 1.0e-9
    
#     # Determine which shells we need to process
#     shell_row_needed = np.zeros(nshells, dtype=np.bool_)
#     shell_col_needed = np.zeros(nshells, dtype=np.bool_)
    
#     for ishell in range(nshells):
#         nbf_i = shell_nbf[ishell]
#         for i_in_shell in range(nbf_i):
#             bf_i = shell_bf_indices[ishell, i_in_shell]
#             if start_row <= bf_i < end_row:
#                 shell_row_needed[ishell] = True
#             if start_col <= bf_i < end_col:
#                 shell_col_needed[ishell] = True
    
#     # Loop over shell pairs
#     for ishell in prange(nshells):
#         if not shell_row_needed[ishell]:
#             continue
            
#         I_center = shell_centers[ishell]
#         nprim_i = shell_nprim[ishell]
#         nbf_i = shell_nbf[ishell]
#         Ni = shell_contr_norms[ishell]
        
#         for jshell in range(nshells):
#             if not shell_col_needed[jshell]:
#                 continue
                
#             J_center = shell_centers[jshell]
#             nprim_j = shell_nprim[jshell]
#             nbf_j = shell_nbf[jshell]
#             Nj = shell_contr_norms[jshell]
            
#             # Pre-compute shell-pair quantities
#             IJ = I_center - J_center
#             fac1 = np.sum(IJ * IJ)
#             fac2 = Ni * Nj
            
#             # Process all basis function pairs in this shell pair
#             for i_in_shell in range(nbf_i):
#                 bf_i = shell_bf_indices[ishell, i_in_shell]
#                 if not (start_row <= bf_i < end_row):
#                     continue
                    
#                 lmni = shell_lmn[ishell, i_in_shell]
#                 lmni_0, lmni_1, lmni_2 = lmni[0], lmni[1], lmni[2]
                
#                 for j_in_shell in range(nbf_j):
#                     bf_j = shell_bf_indices[jshell, j_in_shell]
#                     if not (start_col <= bf_j < end_col):
#                         continue
                    
#                     # Symmetry check for triangular matrices
#                     if ishell == jshell and j_in_shell > i_in_shell:
#                         continue
                    
#                     lmnj = shell_lmn[jshell, j_in_shell]
#                     lmnj_0, lmnj_1, lmnj_2 = lmnj[0], lmnj[1], lmnj[2]
                    
#                     # Pre-compute factors that depend only on lmnj
#                     fac3 = 2 * (lmnj_0 + lmnj_1 + lmnj_2) + 3
#                     fac4 = lmnj_0 * (lmnj_0 - 1)
#                     fac5 = lmnj_1 * (lmnj_1 - 1)
#                     fac6 = lmnj_2 * (lmnj_2 - 1)
                    
#                     result_sum = 0.0
                    
#                     # Loop over primitive pairs
#                     for ik in range(nprim_i):
#                         alphaik = shell_exponents[ishell, ik]
#                         dik = shell_coefficients[ishell, ik]
#                         Nik = shell_prim_norms[ishell, ik]
                        
#                         for jk in range(nprim_j):
#                             alphajk = shell_exponents[jshell, jk]
#                             djk = shell_coefficients[jshell, jk]
#                             Njk = shell_prim_norms[jshell, jk]
                            
#                             gamma = alphaik + alphajk
                            
#                             # Early termination check
#                             temp_1 = np.exp(-alphaik * alphajk / gamma * fac1)
#                             if temp_1 < CUTOFF:
#                                 continue
                            
#                             # Pre-compute P, PI, PJ
#                             gamma_inv = 1.0 / gamma
#                             P = (alphaik * I_center + alphajk * J_center) * gamma_inv
#                             PI = P - I_center
#                             PJ = P - J_center
                            
#                             PI_0, PI_1, PI_2 = PI[0], PI[1], PI[2]
#                             PJ_0, PJ_1, PJ_2 = PJ[0], PJ[1], PJ[2]
                            
#                             # Pre-compute coefficient term
#                             coeff_term = dik * djk * Nik * Njk * fac2 * temp_1
                            
#                             # Calculate the required overlap integrals
#                             # Base overlap
#                             Sx_base = calcS(lmni_0, lmnj_0, gamma, PI_0, PJ_0)
#                             Sy_base = calcS(lmni_1, lmnj_1, gamma, PI_1, PJ_1)
#                             Sz_base = calcS(lmni_2, lmnj_2, gamma, PI_2, PJ_2)
#                             overlap1 = Sx_base * Sy_base * Sz_base
                            
#                             # +2 overlaps
#                             Sx_p2 = calcS(lmni_0, lmnj_0 + 2, gamma, PI_0, PJ_0)
#                             Sy_p2 = calcS(lmni_1, lmnj_1 + 2, gamma, PI_1, PJ_1)
#                             Sz_p2 = calcS(lmni_2, lmnj_2 + 2, gamma, PI_2, PJ_2)
                            
#                             overlap2 = Sx_p2 * Sy_base * Sz_base
#                             overlap3 = Sx_base * Sy_p2 * Sz_base
#                             overlap4 = Sx_base * Sy_base * Sz_p2
                            
#                             # -2 overlaps (only compute if needed)
#                             overlap5 = overlap6 = overlap7 = 0.0
#                             if fac4 != 0.0:
#                                 Sx_m2 = calcS(lmni_0, lmnj_0 - 2, gamma, PI_0, PJ_0)
#                                 overlap5 = Sx_m2 * Sy_base * Sz_base
#                             if fac5 != 0.0:
#                                 Sy_m2 = calcS(lmni_1, lmnj_1 - 2, gamma, PI_1, PJ_1)
#                                 overlap6 = Sx_base * Sy_m2 * Sz_base
#                             if fac6 != 0.0:
#                                 Sz_m2 = calcS(lmni_2, lmnj_2 - 2, gamma, PI_2, PJ_2)
#                                 overlap7 = Sx_base * Sy_base * Sz_m2
                            
#                             # Combine terms
#                             part1 = overlap1 * alphajk * fac3
#                             part2 = 2 * alphajk * alphajk * (overlap2 + overlap3 + overlap4)
#                             part3 = 0.5 * (fac4 * overlap5 + fac5 * overlap6 + fac6 * overlap7)
                            
#                             result = coeff_term * (part1 - part2 - part3)
#                             result_sum += result
                    
#                     T[bf_i - start_row, bf_j - start_col] = result_sum
                    
#                     # Apply symmetry if needed
#                     if ishell == jshell and i_in_shell != j_in_shell:
#                         T[bf_j - start_row, bf_i - start_col] = result_sum
    
#     return T

# # Simplified version that focuses on the core optimization
# @njit(parallel=True, cache=True, fastmath=True)
# def kin_mat_shell_simple(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, 
#                         bfs_coeffs, bfs_prim_norms, bfs_expnts, 
#                         start_row, end_row, start_col, end_col):
#     """
#     Simplified shell-based approach that groups calculations by identical primitive sets.
#     This is easier to implement and still provides significant speedup.
#     """
    
#     num_rows = end_row - start_row
#     num_cols = end_col - start_col
#     T = np.zeros((num_rows, num_cols))
    
#     CUTOFF = 1.0e-9
    
#     # Create a simple grouping based on primitive data
#     # We'll identify groups of basis functions with identical exponents/coefficients
#     nbf = len(bfs_coords)
#     processed = np.zeros(nbf, dtype=np.bool_)
    
#     for i in prange(start_row, end_row):
#         if processed[i]:
#             continue
            
#         I = bfs_coords[i]
#         Ni = bfs_contr_prim_norms[i]
#         lmni = bfs_lmn[i]
#         nprim_i = bfs_nprim[i]
        
#         # Find all basis functions with same center and same primitive data
#         same_shell_i = [i]
#         for k in range(i + 1, nbf):
#             if (np.allclose(bfs_coords[k], I, atol=1e-10) and 
#                 bfs_nprim[k] == nprim_i and
#                 np.allclose(bfs_expnts[k, :nprim_i], bfs_expnts[i, :nprim_i], atol=1e-10) and
#                 np.allclose(bfs_coeffs[k, :nprim_i], bfs_coeffs[i, :nprim_i], atol=1e-10)):
#                 same_shell_i.append(k)
#                 processed[k] = True
        
#         processed[i] = True
        
#         for j in range(start_col, end_col):
#             J = bfs_coords[j]
#             Nj = bfs_contr_prim_norms[j]
#             lmnj = bfs_lmn[j]
#             nprim_j = bfs_nprim[j]
            
#             # Find all basis functions with same center and same primitive data as j
#             same_shell_j = [j]
#             for k in range(j + 1, nbf):
#                 if (np.allclose(bfs_coords[k], J, atol=1e-10) and 
#                     bfs_nprim[k] == nprim_j and
#                     np.allclose(bfs_expnts[k, :nprim_j], bfs_expnts[j, :nprim_j], atol=1e-10) and
#                     np.allclose(bfs_coeffs[k, :nprim_j], bfs_coeffs[j, :nprim_j], atol=1e-10)):
#                     same_shell_j.append(k)
            
#             # Now we have shell-like groups, compute primitive integrals once
#             IJ = I - J
#             fac1 = np.sum(IJ * IJ)
#             fac2 = Ni * Nj
            
#             # Process all combinations within these shell groups
#             for ii in same_shell_i:
#                 if not (start_row <= ii < end_row):
#                     continue
                    
#                 lmni_ii = bfs_lmn[ii]
#                 lmni_0, lmni_1, lmni_2 = lmni_ii[0], lmni_ii[1], lmni_ii[2]
                
#                 for jj in same_shell_j:
#                     if not (start_col <= jj < end_col):
#                         continue
                    
#                     if ii == i and jj == j:  # Already processed or will be processed
#                         continue
                        
#                     lmnj_jj = bfs_lmn[jj]
#                     lmnj_0, lmnj_1, lmnj_2 = lmnj_jj[0], lmnj_jj[1], lmnj_jj[2]
                    
#                     # Compute kinetic integral (same primitive loop as before)
#                     fac3 = 2 * (lmnj_0 + lmnj_1 + lmnj_2) + 3
#                     fac4 = lmnj_0 * (lmnj_0 - 1)
#                     fac5 = lmnj_1 * (lmnj_1 - 1)
#                     fac6 = lmnj_2 * (lmnj_2 - 1)
                    
#                     result_sum = 0.0
                    
#                     for ik in range(nprim_i):
#                         alphaik = bfs_expnts[ii, ik]
#                         dik = bfs_coeffs[ii, ik]
#                         Nik = bfs_prim_norms[ii, ik]
                        
#                         for jk in range(nprim_j):
#                             alphajk = bfs_expnts[jj, jk]
#                             djk = bfs_coeffs[jj, jk]
#                             Njk = bfs_prim_norms[jj, jk]
                            
#                             gamma = alphaik + alphajk
#                             temp_1 = np.exp(-alphaik * alphajk / gamma * fac1)
#                             if temp_1 < CUTOFF:
#                                 continue
                            
#                             gamma_inv = 1.0 / gamma
#                             P = (alphaik * I + alphajk * J) * gamma_inv
#                             PI = P - I
#                             PJ = P - J
                            
#                             coeff_term = dik * djk * Nik * Njk * fac2 * temp_1
                            
#                             # Calculate overlaps (same as before)
#                             Sx_base = calcS(lmni_0, lmnj_0, gamma, PI[0], PJ[0])
#                             Sy_base = calcS(lmni_1, lmnj_1, gamma, PI[1], PJ[1])
#                             Sz_base = calcS(lmni_2, lmnj_2, gamma, PI[2], PJ[2])
#                             overlap1 = Sx_base * Sy_base * Sz_base
                            
#                             Sx_p2 = calcS(lmni_0, lmnj_0 + 2, gamma, PI[0], PJ[0])
#                             Sy_p2 = calcS(lmni_1, lmnj_1 + 2, gamma, PI[1], PJ[1])
#                             Sz_p2 = calcS(lmni_2, lmnj_2 + 2, gamma, PI[2], PJ[2])
                            
#                             overlap2 = Sx_p2 * Sy_base * Sz_base
#                             overlap3 = Sx_base * Sy_p2 * Sz_base
#                             overlap4 = Sx_base * Sy_base * Sz_p2
                            
#                             overlap5 = overlap6 = overlap7 = 0.0
#                             if fac4 != 0.0:
#                                 Sx_m2 = calcS(lmni_0, lmnj_0 - 2, gamma, PI[0], PJ[0])
#                                 overlap5 = Sx_m2 * Sy_base * Sz_base
#                             if fac5 != 0.0:
#                                 Sy_m2 = calcS(lmni_1, lmnj_1 - 2, gamma, PI[1], PJ[1])
#                                 overlap6 = Sx_base * Sy_m2 * Sz_base
#                             if fac6 != 0.0:
#                                 Sz_m2 = calcS(lmni_2, lmnj_2 - 2, gamma, PI[2], PJ[2])
#                                 overlap7 = Sx_base * Sy_base * Sz_m2
                            
#                             part1 = overlap1 * alphajk * fac3
#                             part2 = 2 * alphajk * alphajk * (overlap2 + overlap3 + overlap4)
#                             part3 = 0.5 * (fac4 * overlap5 + fac5 * overlap6 + fac6 * overlap7)
                            
#                             result = coeff_term * (part1 - part2 - part3)
#                             result_sum += result
                    
#                     T[ii - start_row, jj - start_col] = result_sum
    
#     return T