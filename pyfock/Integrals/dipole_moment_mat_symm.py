import numpy as np
from numba import njit , prange

from .integral_helpers import hermite_gauss_coeff

def dipole_moment_mat_symm(basis, slice=None, origin=np.zeros((3))):
    """
    Compute the dipole moment integral matrix in the atomic orbital (AO) basis
    using symmetry and the McMurchie–Davidson algorithm.

    This routine evaluates the matrix elements:

        μ_ij^(k) = ⟨ χ_i | r_k | χ_j ⟩

    where:
        - χ_i, χ_j are Gaussian-type atomic orbitals (AOs)
        - r_k is the k-th Cartesian coordinate (x, y, or z)
        - origin is the coordinate origin for the dipole operator

    The implementation follows the McMurchie–Davidson scheme for Gaussian
    integrals, adapted from:

        https://github.com/jjgoings/McMurchie-Davidson

    Symmetries exploited
    --------------------
    The dipole moment matrix is symmetric for real-valued AOs:

        μ_ij^(k) = μ_ji^(k)

    This halves the number of unique integrals from N_bf^2 to
    N_bf*(N_bf+1)/2 for each Cartesian component.

    Parameters
    ----------
    basis : object
        Basis set object containing:
        - bfs_coords : Cartesian coordinates of AO centers
        - bfs_coeffs : Contraction coefficients
        - bfs_expnts : Gaussian exponents
        - bfs_prim_norms : Primitive normalization constants
        - bfs_contr_prim_norms : Contraction normalization factors
        - bfs_lmn : Angular momentum quantum numbers (ℓ, m, n)
        - bfs_nprim : Number of primitives per basis function
        - bfs_nao : Number of atomic orbitals

    slice : list of int, optional
        A 4-element list [row_start, row_end, col_start, col_end]
        specifying the sub-block of the matrix to compute.
        If None (default), computes the full symmetric matrix.

    origin : ndarray of shape (3,), optional
        Origin of the dipole operator in Cartesian coordinates.
        Default is (0.0, 0.0, 0.0).

    Returns
    -------
    M : ndarray of shape (3, n_rows, n_cols)
        The computed dipole moment integrals for the requested block,
        with the first dimension indexing the x, y, z components.

    Notes
    -----
    - Basis set data are pre-packed into NumPy arrays for compatibility
      with Numba JIT compilation.
    - The algorithm avoids redundant evaluations by exploiting matrix
      symmetry when `slice` covers the full range.
    - The dipole moment integrals are typically used in computing
      molecular dipole moments from density matrices.

    Examples
    --------
    >>> mu_full = dipole_moment_mat_symm(basis)
    >>> mu_block = dipole_moment_mat_symm(basis, slice=[0, 5, 0, 5])
    >>> mu_shifted = dipole_moment_mat_symm(basis, origin=np.array([1.0, 0.0, 0.0]))
    """
    # Here the lists are converted to numpy arrays for better use with Numba.
    # Once these conversions are done we pass these to a Numba decorated
    # function that uses prange, etc. to calculate the matrix efficiently.

    # This function calculates the kinetic energy matrix for a given basis object.
    # The basis object holds the information of basis functions like: exponents, coeffs, etc.
    # It is possible to only calculate a slice (block) of the complete matrix.
    # slice is a 4 element list whose first and second elements give the range of the rows to be calculated.
    # the third and fourth element give the range of columns to be calculated.
    # slice = [start_row, end_row, start_col, end_col]
    # The integrals are performed using the formulas

    #We convert the required properties to numpy arrays as this is what Numba likes.
    bfs_coords = np.array([basis.bfs_coords])
    bfs_contr_prim_norms = np.array([basis.bfs_contr_prim_norms])
    bfs_lmn = np.array([basis.bfs_lmn])
    bfs_nprim = np.array([basis.bfs_nprim])
        

    #The remaining properties like bfs_coeffs are a list of lists of unequal sizes.
    #Numba won't be able to work with these efficiently.
    #So, we convert them to a numpy 2d array by applying a trick,
    #that the second dimension is that of the largest list. So that
    #it can accomadate all the lists.
    maxnprim = max(basis.bfs_nprim)
    bfs_coeffs = np.zeros([basis.bfs_nao, maxnprim])
    bfs_expnts = np.zeros([basis.bfs_nao, maxnprim])
    bfs_prim_norms = np.zeros([basis.bfs_nao, maxnprim])
    for i in range(basis.bfs_nao):
        for j in range(basis.bfs_nprim[i]):
            bfs_coeffs[i,j] = basis.bfs_coeffs[i][j]
            bfs_expnts[i,j] = basis.bfs_expnts[i][j]
            bfs_prim_norms[i,j] = basis.bfs_prim_norms[i][j]
        


    if slice is None:
        slice = [0, basis.bfs_nao, 0, basis.bfs_nao]
        
    #Limits for the calculation of overlap integrals
    a = int(slice[0]) #row start index
    b = int(slice[1]) #row end index
    c = int(slice[2]) #column start index
    d = int(slice[3]) #column end index

        
    M = dipole_moment_mat_symm_internal(bfs_coords[0], bfs_contr_prim_norms[0], bfs_lmn[0], bfs_nprim[0], bfs_coeffs, bfs_prim_norms, bfs_expnts, a, b, c, d, origin)
    
    return M

@njit(parallel=True, cache=False)
def dipole_moment_mat_symm_internal(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts, start_row, end_row, start_col, end_col, origin):
    # This function calculates the dipole moment matrix and uses the symmetry property to only calculate half-ish the elements
    # and get the remaining half by symmetry.
    # The reason we need this extra function is because we want the callable function to be simple and not require so many 
    # arguments. But when using Numba to optimize, we can't have too many custom objects and stuff. Numba likes numpy arrays
    # so passing those is okay. But lists and custom objects are not okay.
    # This function calculates the dipole moment matrix for a given basis object.
    # The basis object holds the information of basis functions like: exponents, coeffs, etc.
    # It is possible to only calculate a slice (block) of the complete matrix.
    # slice is a 4 element list whose first and second elements give the range of the rows to be calculated.
    # the third and fourth element give the range of columns to be calculated.
    # The integrals are performed using the formulas here https://github.com/jjgoings/McMurchie-Davidson/blob/8c9d176204498655a358edf41698e59cf970a548/mmd/backup/reference-integrals.py#L29

    # Infer the matrix shape from the start and end indices
    num_rows = end_row - start_row 
    num_cols = end_col - start_col 
    matrix_shape = (3, num_rows, num_cols) # Default: (3, nao, nao)

    
    # Check if the slice of the matrix requested falls in the lower/upper triangle or in both the triangles
    upper_tri = False
    lower_tri = False
    both_tri_symm = False
    both_tri_nonsymm = False
    if end_row <= start_col:
        upper_tri = True
    elif start_row >= end_col:
        lower_tri = True
    elif start_row==start_col and end_row==end_col:
        both_tri_symm = True
    else:
        both_tri_nonsymm = True


    # Initialize the matrix with zeros
    M = np.zeros(matrix_shape) 

    PI = 3.141592653589793


    #Loop over BFs
    for i in prange(start_row, end_row): 
        I = bfs_coords[i]
        Ni = bfs_contr_prim_norms[i]
        lmni = bfs_lmn[i]
        la, ma, na = lmni
        for j in prange(start_col, end_col):
            
            
            if lower_tri or upper_tri or (both_tri_symm and j<=i) or both_tri_nonsymm:
                result_x = 0.0
                result_y = 0.0
                result_z = 0.0
                J = bfs_coords[j]
                IJ = I - J  
                IJsq = np.sum(IJ**2)
                
                Nj = bfs_contr_prim_norms[j]
                
                lmnj = bfs_lmn[j]
                
                lb, mb, nb = lmnj
                #Loop over primitives
                for ik in range(bfs_nprim[i]):     #Parallelising over primitives doesn't seem to make a difference
                    alphaik = bfs_expnts[i,ik]
                    dik = bfs_coeffs[i,ik]
                    Nik = bfs_prim_norms[i,ik]
                    for jk in range(bfs_nprim[j]):
                        
                        alphajk = bfs_expnts[j,jk]
                        gamma = alphaik + alphajk
                        gamma_inv = 1/gamma
                        temp_gamma = alphaik*alphajk*gamma_inv
                        screenfactor = np.exp(-temp_gamma*IJsq)
                        if abs(screenfactor)<1.0e-8:   
                        # Going lower than E-8 doesn't change the max erroor. There could still be some effects from error compounding but the max error doesnt budge.
                        #TODO: This is quite low. But since this is the slowest part.
                        #But I had to do this because this is a very slow part of the program.
                        #Will have to check how the accuracy is affected and if the screening factor
                        #can be reduced further.
                            continue

                        
                        djk = bfs_coeffs[j,jk] 
                        
                        Njk = bfs_prim_norms[j,jk]
                        
                        # epsilon = 0.25*gamma_inv
                        P = (alphaik*I + alphajk*J)*gamma_inv
                        # PI = P - I
                        # PJ = P - J
                        PC = P - origin
                        # tempfac = (PIx2*gamma_inv)

                        norm_contr_factor = dik*djk*Nik*Njk*Ni*Nj

                        D_x  = hermite_gauss_coeff(la,lb,1,IJ[0],alphaik,alphajk,gamma,temp_gamma) + PC[0]*hermite_gauss_coeff(la,lb,0,IJ[0],alphaik,alphajk,gamma,temp_gamma)
                        #    D  = E(l1,l2,0,A[0]-B[0],a,b,1+n[0],A[0]-gOrigin[0])
                        S2_x = hermite_gauss_coeff(ma,mb,0,IJ[1],alphaik,alphajk)
                        S3_x = hermite_gauss_coeff(na,nb,0,IJ[2],alphaik,alphajk)
                        result_x += D_x*S2_x*S3_x*np.power(PI/gamma,1.5)*norm_contr_factor
                    
                        
                        S1_y = hermite_gauss_coeff(la,lb,0,IJ[0],alphaik,alphajk)
                        D_y  = hermite_gauss_coeff(ma,mb,1,IJ[1],alphaik,alphajk,gamma,temp_gamma) + PC[1]*hermite_gauss_coeff(ma,mb,0,IJ[1],alphaik,alphajk,gamma,temp_gamma)
                        #    D  = E(m1,m2,0,A[1]-B[1],a,b,1+n[1],A[1]-gOrigin[1])
                        S3_y = hermite_gauss_coeff(na,nb,0,IJ[2],alphaik,alphajk)
                        result_y += S1_y*D_y*S3_y*np.power(PI/(gamma),1.5)*norm_contr_factor
                
                        
                        S1_z = hermite_gauss_coeff(la,lb,0,IJ[0],alphaik,alphajk)
                        S2_z = hermite_gauss_coeff(ma,mb,0,IJ[1],alphaik,alphajk)
                        D_z  = hermite_gauss_coeff(na,nb,1,IJ[2],alphaik,alphajk,gamma,temp_gamma) + PC[2]*hermite_gauss_coeff(na,nb,0,IJ[2],alphaik,alphajk,gamma,temp_gamma)
                        #    D  = E(n1,n2,0,A[2]-B[2],a,b,1+n[2],A[2]-gOrigin[2]) 
                        result_z += S1_z*S2_z*D_z*np.power(PI/(gamma),1.5)*norm_contr_factor
                # print(result_y)

                M[0, i - start_row, j - start_col] = result_x
                M[1, i - start_row, j - start_col] = result_y
                M[2, i - start_row, j - start_col] = result_z
                
            
    if both_tri_symm:
        #We save time by evaluating only the lower diagonal elements and then use symmetry Mi,j=Mj,i 
        for i in prange(start_row, end_row):
            for j in prange(start_col, end_col):
                if j>i:
                    M[0, i-start_row, j-start_col] = M[0, j-start_col, i-start_row]
                    M[1, i-start_row, j-start_col] = M[1, j-start_col, i-start_row]
                    M[2, i-start_row, j-start_col] = M[2, j-start_col, i-start_row]
                
    return M