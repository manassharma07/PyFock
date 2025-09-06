import numpy as np
from numba import njit , prange

from .integral_helpers import calcS

def cross_overlap_mat_symm(basisA, basisB, slice=None):
    """
    Compute the overlap matrix between two distinct basis sets.

    This function calculates the overlap integrals ⟨χ_i^A | χ_j^B⟩ between basis functions 
    from two different basis sets, `basisA` and `basisB`. It supports block-wise computation 
    of the matrix to save time and memory during large-scale calculations.

    S_{μν}^{AB} = ⟨ χ_μ^A | χ_ν^B ⟩

    All integrals are evaluated using a Numba-accelerated backend that converts 
    required basis set properties into NumPy arrays for efficient performance.

    Parameters
    ----------
    basisA : object
        The first basis set object containing properties such as:
        - bfs_coords: Cartesian coordinates of centers
        - bfs_coeffs: Contraction coefficients
        - bfs_expnts: Gaussian exponents
        - bfs_prim_norms: Primitive normalization constants
        - bfs_contr_prim_norms: Contraction normalization constants
        - bfs_lmn: Angular momentum quantum numbers
        - bfs_nprim: Number of primitives per AO
        - bfs_nao: Total number of atomic orbitals

    basisB : object
        The second basis set object, structured identically to `basisA`.

    slice : list of int, optional
        A 4-element list `[start_row, end_row, start_col, end_col]` that defines
        a block of the full matrix to compute. Rows correspond to functions in `basisA`,
        and columns to those in `basisB`. If `None` (default), the full matrix is computed.

    Returns
    -------
    S : ndarray of shape (end_row - start_row, end_col - start_col)
        The computed cross overlap (sub)matrix.

    Notes
    -----
    The function handles contraction and normalization internally. Since Numba does not
    support jagged lists, the function reshapes data into padded 2D arrays based on
    the maximum number of primitives in either basis set.

    Examples
    --------
    >>> S = cross_overlap_mat_symm(basisA, basisB)
    >>> S_block = cross_overlap_mat_symm(basisA, basisB, slice=[0, 5, 0, 10])
    """
    # Here the lists are converted to numpy arrays for better use with Numba.
    # Once these conversions are done we pass these to a Numba decorated
    # function that uses prange, etc. to calculate the matrix efficiently.

    # This function calculates the overlap matrix between two basis objects: basisA and basisB.
    # The basis objects hold the information of basis functions like: exponents, coeffs, etc.
    # It is possible to only calculate a slice (block) of the complete matrix.
    # slice is a 4 element list whose first and second elements give the range of the rows (basisA) to be calculated.
    # the third and fourth element give the range of columns (basisB) to be calculated.
    # slice = [start_row, end_row, start_col, end_col]
    # The integrals are performed using the formulas

    #We convert the required properties to numpy arrays as this is what Numba likes.
    bfsA_coords = np.array([basisA.bfs_coords])
    bfsA_contr_prim_norms = np.array([basisA.bfs_contr_prim_norms])
    bfsA_lmn = np.array([basisA.bfs_lmn])
    bfsA_nprim = np.array([basisA.bfs_nprim])
    
    bfsB_coords = np.array([basisB.bfs_coords])
    bfsB_contr_prim_norms = np.array([basisB.bfs_contr_prim_norms])
    bfsB_lmn = np.array([basisB.bfs_lmn])
    bfsB_nprim = np.array([basisB.bfs_nprim])
        

    #The remaining properties like bfs_coeffs are a list of lists of unequal sizes.
    #Numba won't be able to work with these efficiently.
    #So, we convert them to a numpy 2d array by applying a trick,
    #that the second dimension is that of the largest list. So that
    #it can accomadate all the lists.
    maxnprimA = max(basisA.bfs_nprim)
    bfsA_coeffs = np.zeros([basisA.bfs_nao, maxnprimA])
    bfsA_expnts = np.zeros([basisA.bfs_nao, maxnprimA])
    bfsA_prim_norms = np.zeros([basisA.bfs_nao, maxnprimA])
    for i in range(basisA.bfs_nao):
        for j in range(basisA.bfs_nprim[i]):
            bfsA_coeffs[i,j] = basisA.bfs_coeffs[i][j]
            bfsA_expnts[i,j] = basisA.bfs_expnts[i][j]
            bfsA_prim_norms[i,j] = basisA.bfs_prim_norms[i][j]
    
    maxnprimB = max(basisB.bfs_nprim)
    bfsB_coeffs = np.zeros([basisB.bfs_nao, maxnprimB])
    bfsB_expnts = np.zeros([basisB.bfs_nao, maxnprimB])
    bfsB_prim_norms = np.zeros([basisB.bfs_nao, maxnprimB])
    for i in range(basisB.bfs_nao):
        for j in range(basisB.bfs_nprim[i]):
            bfsB_coeffs[i,j] = basisB.bfs_coeffs[i][j]
            bfsB_expnts[i,j] = basisB.bfs_expnts[i][j]
            bfsB_prim_norms[i,j] = basisB.bfs_prim_norms[i][j]
        

    if slice is None:
        slice = [0, basisA.bfs_nao, 0, basisB.bfs_nao]
        
    #Limits for the calculation of overlap integrals
    a = int(slice[0]) #row start index (basisA)
    b = int(slice[1]) #row end index (basisA)
    c = int(slice[2]) #column start index (basisB)
    d = int(slice[3]) #column end index (basisB)
        
    S = cross_overlap_mat_internal(bfsA_coords[0], bfsA_contr_prim_norms[0], bfsA_lmn[0], bfsA_nprim[0], bfsA_coeffs, bfsA_prim_norms, bfsA_expnts,
                                       bfsB_coords[0], bfsB_contr_prim_norms[0], bfsB_lmn[0], bfsB_nprim[0], bfsB_coeffs, bfsB_prim_norms, bfsB_expnts,
                                       a, b, c, d)
    return S

@njit(parallel=True, cache=True)
def cross_overlap_mat_internal(bfsA_coords, bfsA_contr_prim_norms, bfsA_lmn, bfsA_nprim, bfsA_coeffs, bfsA_prim_norms, bfsA_expnts,
                                   bfsB_coords, bfsB_contr_prim_norms, bfsB_lmn, bfsB_nprim, bfsB_coeffs, bfsB_prim_norms, bfsB_expnts,
                                   start_row, end_row, start_col, end_col):
    # This function calculates the overlap matrix between two different basis objects.
    # Since the two basis objects can be different, we don't use symmetry properties.
    # The reason we need this extra function is because we want the callable function to be simple and not require so many 
    # arguments. But when using Numba to optimize, we can't have too many custom objects and stuff. Numba likes numpy arrays
    # so passing those is okay. But lists and custom objects are not okay.
    # This function calculates the overlap matrix between basisA (rows) and basisB (columns).
    # It is possible to only calculate a slice (block) of the complete matrix.
    # slice is a 4 element list whose first and second elements give the range of the rows to be calculated.
    # the third and fourth element give the range of columns to be calculated.
    # The integrals are performed using the formulas here https://pubs.acs.org/doi/full/10.1021/acs.jchemed.8b00255

    # Infer the matrix shape from the start and end indices
    num_rows = end_row - start_row 
    num_cols = end_col - start_col 
    matrix_shape = (num_rows, num_cols)

    # Initialize the matrix with zeros
    S = np.zeros(matrix_shape) 

    for i in prange(start_row, end_row):
        I = bfsA_coords[i]
        lmni = bfsA_lmn[i]
        Ni = bfsA_contr_prim_norms[i]
        for j in prange(start_col, end_col):
            result = 0.0
            
            J = bfsB_coords[j]
            IJ = I - J  
            tempfac = np.sum(IJ**2)
            
            Nj = bfsB_contr_prim_norms[j]
            
            lmnj = bfsB_lmn[j]
            for ik in prange(bfsA_nprim[i]):
                alphaik = bfsA_expnts[i][ik]
                dik = bfsA_coeffs[i][ik]
                Nik = bfsA_prim_norms[i][ik]
                for jk in prange(bfsB_nprim[j]):
                    
                    alphajk = bfsB_expnts[j][jk]
                    gamma = alphaik + alphajk
                    screenfactor = np.exp(-alphaik*alphajk/gamma*tempfac)
                    if (abs(screenfactor)<1.0e-12):
                        continue
                    
                    djk = bfsB_coeffs[j][jk] 
                    Njk = bfsB_prim_norms[j][jk]
                    
                    P = (alphaik*I + alphajk*J)/gamma
                    PI = P - I
                    PJ = P - J
                    Sx = calcS(lmni[0],lmnj[0],gamma,PI[0],PJ[0])
                    Sy = calcS(lmni[1],lmnj[1],gamma,PI[1],PJ[1])
                    Sz = calcS(lmni[2],lmnj[2],gamma,PI[2],PJ[2])
                    temp = dik*djk
                    temp = temp*Nik*Njk
                    temp = temp*Ni*Nj
                    temp = temp*screenfactor*Sx*Sy*Sz
                    result += temp
            S[i - start_row, j - start_col] = result

    return S