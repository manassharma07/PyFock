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


@njit(parallel=False, cache=True, fastmath=True, error_model="numpy")
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
    
    for i in range(nbf):
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
            
            # D_max = max(abs(d_ij), abs(d_kl), 
            #            abs(d_ik), abs(d_il), 
            #            abs(d_jk), abs(d_jl))
            # if abs(D_max) < DENS_THRESH:
            #     continue
            sqrt_ij = sqrt_ints4c2e_diag[i,j]
            if sqrt_ij < threshold:
                continue
            sqrt_kl = sqrt_ints4c2e_diag[k,l]
            if sqrt_kl < threshold:
                continue
            schwarz_prod = sqrt_ij * sqrt_kl
            if schwarz_prod < threshold:
                continue
                                
            # if abs(D_max)*schwarz_prod < COMBINED_THRESH:
            #     continue
            if (abs(d_ij) * schwarz_prod < COMBINED_THRESH) and (abs(d_kl) * schwarz_prod < COMBINED_THRESH):
                continue
            
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
