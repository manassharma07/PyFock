import numpy as np
from numba import njit, prange

from .integral_helpers import innerLoop4c2e
from .rys_helpers import coulomb_rys

def rys_2c2e_symm(basis, slice=None):
    """
    Compute symmetric two-center two-electron integrals using Rys quadrature.

    This function evaluates the electron repulsion integrals (ERIs)
    involving only two centers (2c2e), using the Rys quadrature method. 
    The integrals are computed efficiently using Numba, and the data is prepared 
    accordingly for compatibility with JIT compilation. It assumes symmetry, 
    and computes only a specified block if requested.

    Parameters
    ----------
    basis : object
        A basis set object containing the basis function data. It must have attributes:
        - bfs_coords : list of (x, y, z) coordinates of basis function centers
        - bfs_contr_prim_norms : contraction-normalization factors
        - bfs_lmn : angular momentum tuples (l, m, n)
        - bfs_nprim : number of primitives for each basis function
        - bfs_coeffs : contraction coefficients
        - bfs_expnts : primitive exponents
        - bfs_prim_norms : primitive normalization constants
        - bfs_nao : total number of atomic orbitals (basis functions)

    slice : list of int, optional
        A four-element list `[start_row, end_row, start_col, end_col]` specifying the 
        row and column ranges of the matrix to be computed. If None, the full matrix 
        is computed.

    Returns
    -------
    ints2c2e : ndarray
        A 2D numpy array of shape `(end_row - start_row, end_col - start_col)` 
        containing the computed symmetric two-center two-electron integrals.

    Notes
    -----
    This function prepares the basis set data in a Numba-friendly format by 
    converting ragged lists into padded 2D arrays, where the second dimension 
    corresponds to the maximum number of primitives. The core integral computation 
    is offloaded to a Numba-accelerated function `rys_2c2e_symm_internal`.
    """
    # Here the lists are converted to numpy arrays for better use with Numba.
    # Once these conversions are done we pass these to a Numba decorated
    # function that uses prange, etc. to calculate the 3c2e integrals efficiently.
    # This function calculates the 3c2e electron-electron ERIs for a given basis object and auxbasis object.
    # The basis object holds the information of basis functions like: exponents, coeffs, etc.
    
    # It is possible to only calculate a slice (block/subset) of the complete set of integrals.
    # slice is a 4 element list whose first and second elements give the range of the A functions to be calculated.
    # and so on.
    # slice = [start_row, end_row, start_col, end_col]

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
    start_row = int(slice[0]) #row start index
    end_row = int(slice[1]) #row end index
    start_col = int(slice[2]) #column start index
    end_col = int(slice[3]) #column end index

    ints2c2e = rys_2c2e_symm_internal(bfs_coords[0], bfs_contr_prim_norms[0], bfs_lmn[0], bfs_nprim[0], bfs_coeffs, bfs_prim_norms, bfs_expnts, start_row, end_row, start_col, end_col)
    return ints2c2e

@njit(parallel=True, cache=True, fastmath=True, error_model="numpy")
def rys_2c2e_symm_internal(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts, start_row, end_row, start_col, end_col):
    # Two centered two electron integrals by hacking the 4c2e routines based on rys quadrature.
    # Based on Rys Quadrature from https://github.com/rpmuller/MolecularIntegrals.jl
    # This function calculates the electron-electron Coulomb potential matrix for a given basis object.
    # The basis object holds the information of basis functions like: exponents, coeffs, etc.
        
    # returns (A|C) 
    
    # Infer the matrix shape from the start and end indices
    num_rows = end_row - start_row 
    num_cols = end_col - start_col 
    matrix_shape = (num_rows, num_cols)

    
    # Check if the slice of the matrix requested falls in the lower/upper triangle or in both the triangles
    tri_symm = False
    no_symm = False
    if start_row==start_col and end_row==end_col:
        tri_symm = True
    else:
        no_symm = True


    # Initialize the matrix with zeros
    twoC2E = np.zeros(matrix_shape) 
    
       
        
    # pi = 3.141592653589793
    # pisq = 9.869604401089358  #PI^2
    # twopisq = 19.739208802178716  #2*PI^2
    J = L = np.zeros((3))
    Nj = Nl = 1
    lnmj = lmnl = np.zeros((3),dtype=np.int32)
    lb, mb, nb = int(0), int(0), int(0)
    ld, md, nd = int(0), int(0), int(0)
    alphajk = alphalk = 0.0
    djk, dlk = 1.0, 1.0
    Njk, Nlk = 1.0, 1.0
    #Loop pver BFs
    for i in prange(start_row, end_row): #A
        I = bfs_coords[i]
        # J = I
        # IJ = I #I - J
        P = I
        # IJsq = np.sum(IJ**2)
        Ni = bfs_contr_prim_norms[i]
        lmni = bfs_lmn[i]
        la, ma, na = lmni
        nprimi = bfs_nprim[i]    
        
        for k in range(start_col, end_col): #C
            if (tri_symm and k<=i) or no_symm:

                val = 0.0


                K = bfs_coords[k]
                # L = K
                # KL = K
                Q = K
                # KLsq = np.sum(KL**2)
                Nk = bfs_contr_prim_norms[k]
                lmnk = bfs_lmn[k]
                lc, mc, nc = lmnk
                tempcoeff1 = Ni*Nk
                nprimk = bfs_nprim[k]
                

                norder = int((la+ma+na+lc+mc+nc)/2+1 ) 
                n = int(max(la,ma,na))
                m = int(max(lc,mc,nc))
                roots = np.zeros((norder))
                weights = np.zeros((norder))
                G = np.zeros((n+1,m+1))
                        
                PQ = P - Q
                PQsq = np.sum(PQ**2)
                
                        
                #Loop over primitives
                for ik in range(nprimi):   
                    dik = bfs_coeffs[i][ik]
                    Nik = bfs_prim_norms[i][ik]
                    alphaik = bfs_expnts[i][ik]
                    alphajk = 0.0
                    tempcoeff2 = tempcoeff1*dik*Nik
                    gammaP = alphaik
                            
                    for kk in range(nprimk):
                        dkk = bfs_coeffs[k][kk]
                        Nkk = bfs_prim_norms[k][kk]
                        alphakk = bfs_expnts[k][kk]
                        alphalk = 0.0
                        tempcoeff3 = tempcoeff2*dkk*Nkk 
                        gammaQ = alphakk
                                    
                                        
                                        
                        rho = gammaP*gammaQ/(gammaP+gammaQ)
                        
                                        
                                        
                        val += tempcoeff3*coulomb_rys(roots,weights,G,PQsq, rho, norder,n,m,la,lb,lc,ld,ma,mb,mc,md,na,nb,nc,nd,alphaik, alphajk, alphakk, alphalk,I,J,K,L)
                                    

                twoC2E[i-start_row, k-start_col] = val        
                    
    if tri_symm:
        #We save time by evaluating only the lower diagonal elements and then use symmetry Si,j=Sj,i 
        for i in prange(start_row, end_row):
            for j in prange(start_col, end_col):
                if j>i:
                    twoC2E[i-start_row, j-start_col] = twoC2E[j-start_col, i-start_row]                        
                                   
                            
        
    return twoC2E