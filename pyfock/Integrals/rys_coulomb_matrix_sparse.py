import numpy as np
from numba import njit, prange, get_num_threads, get_thread_id


def rys_coulomb_matrix_sparse(ints4c2e_values, ints4c2e_indices, density_matrix, threshold, sqrt_ints4c2e_diag):
    """
    Compute the Coulomb matrix (J matrix) from sparse 4c2e integrals.
    
    The Coulomb matrix is defined as:
        J[i,j] = Σ_kl D[k,l] * (ij|kl)
    
    This function takes pre-computed sparse 4c2e integrals and contracts them
    with the density matrix to produce the Coulomb matrix efficiently.
    
    Parameters
    ----------
    ints4c2e_values : list of ndarrays
        List where ints4c2e_values[i] contains all significant integral values
        for basis function i. Each element is a 1D array of floats.
        
    ints4c2e_indices : list of ndarrays
        List where ints4c2e_indices[i] contains the (j,k,l) indices corresponding
        to the integrals in ints4c2e_values[i]. Each element is a 2D array of 
        shape (n_integrals, 3) with dtype int32.
        
    density_matrix : ndarray, shape (N, N)
        The density matrix in the AO basis. Must be symmetric.
    
    Returns
    -------
    J_matrix : ndarray, shape (N, N)
        The Coulomb matrix in the AO basis.
    
    Notes
    -----
    The sparse format stores integrals where the first index 'i' is implicit
    (it's the list index), and j,k,l are stored explicitly.
    
    The integrals are stored with the following symmetry constraints:
    - j <= i (enforced during sparse generation)
    - l <= k (enforced during sparse generation)
    - triangle(i,j) >= triangle(k,l) (8-fold symmetry)
    
    This function properly accounts for all 8-fold permutational symmetries:
        (i j | k l) = (j i | k l) = (i j | l k) = (j i | l k)
                    = (k l | i j) = (l k | i j) = (k l | j i) = (l k | j i)
    
    Examples
    --------
    >>> # Compute sparse integrals first
    >>> ints_vals, ints_idxs = rys_4c2e_schwarz_sparse_symm(basis, schwarz, threshold=1e-9)
    >>> # Then compute J matrix
    >>> J = coulomb_matrix_from_sparse(ints_vals, ints_idxs, density_matrix)
    """
    nbf = len(ints4c2e_values)
    
    # Ensure density matrix is the right type
    density_matrix = np.ascontiguousarray(density_matrix, dtype=np.float64)
    
    # Call the numba-compiled internal function
    J_matrix = _coulomb_from_sparse_internal(
        ints4c2e_values, ints4c2e_indices, density_matrix, nbf, threshold, sqrt_ints4c2e_diag
    )
    
    return J_matrix


# @njit(parallel=False, cache=True, fastmath=True, error_model="numpy")
# def _coulomb_from_sparse_internal(ints4c2e_values, ints4c2e_indices, density_matrix, nbf, threshold):
#     """
#     Internal numba-compiled function to compute Coulomb matrix from sparse integrals.
    
#     This function exploits 8-fold permutational symmetry to update all relevant
#     J matrix elements from each stored integral.
#     """
#     # Initialize J matrix
#     J_matrix = np.zeros((nbf, nbf), dtype=np.float64)
#     DENS_THRESH = 1e-11
#     COMBINED_THRESH = threshold * 0.1
    
    
#     # Iterate over all stored integrals
#     # Each iteration processes all integrals with first index = i
#     for i in prange(nbf):
#         n_integrals = len(ints4c2e_values[i])
        
#         for idx in range(n_integrals):
#             # Get the integral value and indices
#             val = ints4c2e_values[i][idx]
#             j = ints4c2e_indices[i][idx, 0]
#             k = ints4c2e_indices[i][idx, 1]
#             l = ints4c2e_indices[i][idx, 2]
            
#             # Get density matrix elements
#             d_ij = density_matrix[i, j]
#             if abs(d_ij)<DENS_THRESH:
#                 continue
#             d_kl = density_matrix[k, l]
#             if abs(d_kl)<DENS_THRESH:
#                 continue
#             d_ik = density_matrix[i, k]
#             if abs(d_ik)<DENS_THRESH:
#                 continue
#             d_il = density_matrix[i, l]
#             if abs(d_il)<DENS_THRESH:
#                 continue
#             d_jk = density_matrix[j, k]
#             if abs(d_jk)<DENS_THRESH:
#                 continue
#             d_jl = density_matrix[j, l]
#             if abs(d_jl)<DENS_THRESH:
#                 continue
#             D_max = max(abs(d_ij), abs(d_kl), 
#                     abs(d_ik), abs(d_il), 
#                     abs(d_jk), abs(d_jl))
#             if abs(D_max) < DENS_THRESH:
#                 continue
#             # if abs(D_max)*schwarz_prod < COMBINED_THRESH:
#             #     continue
#             # if (abs(d_val_ij) * schwarz_prod < COMBINED_THRESH) and (abs(d_val_kl) * schwarz_prod < COMBINED_THRESH):
#             #     continue
            
#             # We have the integral (ij|kl)
#             # Due to 8-fold symmetry, this represents 8 potentially distinct integrals:
#             # (ij|kl), (ji|kl), (ij|lk), (ji|lk), (kl|ij), (lk|ij), (kl|ji), (lk|ji)
            
#             # However, we need to be careful not to double-count
#             # The storage scheme already ensures j<=i, l<=k, and triangle(ij)>=triangle(kl)
            
#             # Contribution to J[i,j] from D[k,l]
#             # J[i,j] += (ij|kl) * D[k,l]
#             # If k != l, we also have (ij|lk) * D[l,k] = (ij|kl) * D[k,l]
#             fac_kl = 2.0 if k != l else 1.0
#             J_matrix[i, j] += val * d_kl * fac_kl
            
#             # Due to symmetry J[j,i] = J[i,j], but we update it explicitly
#             if i != j:
#                 J_matrix[j, i] += val * d_kl * fac_kl
            
#             # Contribution to J[k,l] from D[i,j]
#             # J[k,l] += (kl|ij) * D[i,j] = (ij|kl) * D[i,j]
#             # If i != j, we also have (kl|ji) * D[j,i] = (ij|kl) * D[i,j]
#             # But we need to avoid double-counting when (i,j) == (k,l)
#             if not (i == k and j == l):
#                 fac_ij = 2.0 if i != j else 1.0
#                 J_matrix[k, l] += val * d_ij * fac_ij
                
#                 # Symmetry update
#                 if k != l:
#                     J_matrix[l, k] += val * d_ij * fac_ij
    
#     return J_matrix
@njit(parallel=True, cache=True, fastmath=True, error_model="numpy")
def _coulomb_from_sparse_internal(ints4c2e_values, ints4c2e_indices, density_matrix, nbf, threshold, sqrt_ints4c2e_diag):
    """
    Internal numba-compiled function to compute Coulomb matrix from sparse integrals.
    Uses thread-local matrices to avoid race conditions.
    """
    DENS_THRESH = 1e-11
    COMBINED_THRESH = threshold * 0.1
    
    # Each thread gets its own J matrix
    num_threads = get_num_threads()
    J_matrices = np.zeros((num_threads, nbf, nbf), dtype=np.float64)
    
    for i in prange(nbf):
        thread_id = get_thread_id()
        J_local = J_matrices[thread_id]
        
        n_integrals = len(ints4c2e_values[i])
        for idx in range(n_integrals):
            val = ints4c2e_values[i][idx]
            j = ints4c2e_indices[i][idx, 0]
            k = ints4c2e_indices[i][idx, 1]
            l = ints4c2e_indices[i][idx, 2]
            
            d_ij = density_matrix[i, j]
            if abs(d_ij) < DENS_THRESH:
                continue
            d_kl = density_matrix[k, l]
            if abs(d_kl) < DENS_THRESH:
                continue
            d_ik = density_matrix[i, k]
            if abs(d_ik) < DENS_THRESH:
                continue
            d_il = density_matrix[i, l]
            if abs(d_il) < DENS_THRESH:
                continue
            d_jk = density_matrix[j, k]
            if abs(d_jk) < DENS_THRESH:
                continue
            d_jl = density_matrix[j, l]
            if abs(d_jl) < DENS_THRESH:
                continue
            
            D_max = max(abs(d_ij), abs(d_kl), 
                       abs(d_ik), abs(d_il), 
                       abs(d_jk), abs(d_jl))
            if abs(D_max) < DENS_THRESH:
                continue
            sqrt_ij = sqrt_ints4c2e_diag[i,j]
            sqrt_kl = sqrt_ints4c2e_diag[k,l]
            schwarz_prod = sqrt_ij * sqrt_kl
                                
            if abs(D_max)*schwarz_prod < COMBINED_THRESH:
                continue
            # if (abs(d_ij) * schwarz_prod < COMBINED_THRESH) and (abs(d_kl) * schwarz_prod < COMBINED_THRESH):
            #     continue
            
            # All updates go to thread-local matrix
            fac_kl = 2.0 if k != l else 1.0
            J_local[i, j] += val * d_kl * fac_kl
            if i != j:
                J_local[j, i] += val * d_kl * fac_kl
            
            if not (i == k and j == l):
                fac_ij = 2.0 if i != j else 1.0
                J_local[k, l] += val * d_ij * fac_ij
                if k != l:
                    J_local[l, k] += val * d_ij * fac_ij
    
    # Reduce all thread-local matrices
    J_matrix = np.sum(J_matrices, axis=0)
    return J_matrix

@njit(parallel=False, cache=True, fastmath=True)
def _coulomb_from_sparse_internal_v2(ints4c2e_values, ints4c2e_indices, density_matrix, nbf):
    """
    Alternative implementation with explicit 8-fold symmetry handling.
    
    This version explicitly considers all 8 permutations and updates J matrix
    elements accordingly. May be clearer but potentially slower due to more
    conditional checks.
    """
    # Initialize J matrix
    J_matrix = np.zeros((nbf, nbf), dtype=np.float64)
    
    # Iterate over all stored integrals
    for i in prange(nbf):
        n_integrals = len(ints4c2e_values[i])
        
        for idx in range(n_integrals):
            # Get the integral value (ij|kl) and indices
            val = ints4c2e_values[i][idx]
            j = ints4c2e_indices[i][idx, 0]
            k = ints4c2e_indices[i][idx, 1]
            l = ints4c2e_indices[i][idx, 2]
            
            # 8-fold symmetry contributions to J matrix
            # J[μ,ν] = Σ_λσ D[λ,σ] * (μν|λσ)
            
            # (ij|kl) * D[k,l] -> J[i,j]
            J_matrix[i, j] += val * density_matrix[k, l]
            
            # (ij|lk) * D[l,k] -> J[i,j] (only if k != l)
            if k != l:
                J_matrix[i, j] += val * density_matrix[l, k]
            
            # (ji|kl) * D[k,l] -> J[j,i] (only if i != j)
            if i != j:
                J_matrix[j, i] += val * density_matrix[k, l]
                
                # (ji|lk) * D[l,k] -> J[j,i] (only if i != j and k != l)
                if k != l:
                    J_matrix[j, i] += val * density_matrix[l, k]
            
            # (kl|ij) * D[i,j] -> J[k,l] (only if (k,l) != (i,j))
            if not (k == i and l == j):
                J_matrix[k, l] += val * density_matrix[i, j]
                
                # (kl|ji) * D[j,i] -> J[k,l] (only if i != j)
                if i != j:
                    J_matrix[k, l] += val * density_matrix[j, i]
                
                # (lk|ij) * D[i,j] -> J[l,k] (only if k != l)
                if k != l:
                    J_matrix[l, k] += val * density_matrix[i, j]
                    
                    # (lk|ji) * D[j,i] -> J[l,k] (only if i != j and k != l)
                    if i != j:
                        J_matrix[l, k] += val * density_matrix[j, i]
    
    return J_matrix


def rys_coulomb_matrix_sparse_v2(ints4c2e_values, ints4c2e_indices, density_matrix, threshold):
    """
    Compute the Coulomb matrix using the alternative (more explicit) implementation.
    
    See coulomb_matrix_from_sparse() for parameter documentation.
    """
    nbf = len(ints4c2e_values)
    density_matrix = np.ascontiguousarray(density_matrix, dtype=np.float64)
    
    J_matrix = _coulomb_from_sparse_internal_v2(
        ints4c2e_values, ints4c2e_indices, density_matrix, nbf
    )
    
    return J_matrix