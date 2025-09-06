import numpy as np
from numba import njit, prange

from .integral_helpers import innerLoop4c2e

def conv_3c2e_symm(basis, auxbasis, slice=None):
    """
    Compute three-center two-electron (3c2e) electron repulsion integrals (ERIs)
    using the conventional and slow (analytical) formula-based method with symmetry exploitation.

    This function evaluates integrals of the form (A B | C), where:
        - A and B are primary basis functions from `basis`
        - C is an auxiliary basis function from `auxbasis`

    The "conv" variant uses explicit analytical integral formulas and nested
    loops over primitive Gaussians, following the derivations in:

        J. Chem. Educ. 2018, 95, 9, 1572–1578
        https://pubs.acs.org/doi/full/10.1021/acs.jchemed.8b00255

    Compared to the Rys quadrature method, this conventional approach is
    significantly more computationally expensive, but can be useful for
    validation or when Rys is not applicable.

    Symmetries exploited
    --------------------
    For 3c2e integrals, the following bra symmetry is used:

        (A B | C) = (B A | C)

    This reduces the number of computed integrals from N_bf² * N_aux
    to N_bf*(N_bf+1)/2 * N_aux when the full tensor is computed.

    Parameters
    ----------
    basis : object
        Primary basis set object containing:
        - bfs_coords : Cartesian coordinates of basis function centers
        - bfs_coeffs : Contraction coefficients
        - bfs_expnts : Gaussian exponents
        - bfs_prim_norms : Primitive normalization constants
        - bfs_contr_prim_norms : Contraction normalization factors
        - bfs_lmn : Angular momentum quantum numbers (ℓ, m, n)
        - bfs_nprim : Number of primitives per basis function
        - bfs_nao : Total number of atomic orbitals

    auxbasis : object
        Auxiliary basis set object with the same attributes as `basis`.

    slice : list of int, optional
        A 6-element list specifying a sub-block of integrals to compute:
        [start_A, end_A, start_B, end_B, start_C, end_C]
        If None (default), computes the full (Nbf, Nbf, Naux) tensor.

        **Note:** When slices are used, AB symmetry exploitation is limited
        to permutations that lie entirely within the specified slice.

    Returns
    -------
    ints3c2e : ndarray
        The computed 3-center 2-electron integrals for the requested range.
        Shape: (Nbf, Nbf, Nauxbf) or 
        (end_A - start_A, end_B - start_B, end_C - start_C) 
        if slice is given.

    Notes
    -----
    - All basis set data are pre-packed into NumPy arrays for Numba acceleration.
    - Uses explicit primitive Gaussian formula evaluation without numerical quadrature.
    - Symmetry exploitation is maximal only for full-range computations.
    - Conventional evaluation scales poorly compared to Rys quadrature, but is
      exact for the given formula set.

    Examples
    --------
    >>> eri_full = conv_3c2e_symm(basis, auxbasis)
    >>> eri_block = conv_3c2e_symm(basis, auxbasis, slice=[0,5, 0,5, 0,10])
    """
    # Here the lists are converted to numpy arrays for better use with Numba.
    # Once these conversions are done we pass these to a Numba decorated
    # function that uses prange, etc. to calculate the 3c2e integrals efficiently.
    # This function calculates the 3c2e electron-electron ERIs for a given basis object and auxbasis object.
    # The basis object holds the information of basis functions like: exponents, coeffs, etc.
    
    # The integrals are performed using the formulas https://pubs.acs.org/doi/full/10.1021/acs.jchemed.8b00255

    # It is possible to only calculate a slice (block/subset) of the complete set of integrals.
    # slice is an 6 element list whose first and second elements give the range of the A functions to be calculated.
    # and so on.
    # slice = [indx_startA, indx_endA, indx_startB, indx_endB, indx_startC, indx_endC]

    #We convert the required properties to numpy arrays as this is what Numba likes.
    bfs_coords = np.array([basis.bfs_coords])
    bfs_contr_prim_norms = np.array([basis.bfs_contr_prim_norms])
    bfs_lmn = np.array([basis.bfs_lmn])
    bfs_nprim = np.array([basis.bfs_nprim])

    #We convert the required properties to numpy arrays as this is what Numba likes.
    aux_bfs_coords = np.array([auxbasis.bfs_coords])
    aux_bfs_contr_prim_norms = np.array([auxbasis.bfs_contr_prim_norms])
    aux_bfs_lmn = np.array([auxbasis.bfs_lmn])
    aux_bfs_nprim = np.array([auxbasis.bfs_nprim])
        

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

    maxnprimaux = max(auxbasis.bfs_nprim)
    aux_bfs_coeffs = np.zeros([auxbasis.bfs_nao, maxnprimaux])
    aux_bfs_expnts = np.zeros([auxbasis.bfs_nao, maxnprimaux])
    aux_bfs_prim_norms = np.zeros([auxbasis.bfs_nao, maxnprimaux])
    for i in range(auxbasis.bfs_nao):
        for j in range(auxbasis.bfs_nprim[i]):
            aux_bfs_coeffs[i,j] = auxbasis.bfs_coeffs[i][j]
            aux_bfs_expnts[i,j] = auxbasis.bfs_expnts[i][j]
            aux_bfs_prim_norms[i,j] = auxbasis.bfs_prim_norms[i][j]
            


        
    if slice is None:
        slice = [0, basis.bfs_nao, 0, basis.bfs_nao, 0, auxbasis.bfs_nao]
        
    #Limits for the calculation of 4c2e integrals
    indx_startA = int(slice[0])
    indx_endA = int(slice[1])
    indx_startB = int(slice[2])
    indx_endB = int(slice[3])
    indx_startC = int(slice[4])
    indx_endC = int(slice[5])

    ints3c2e = conv_3c2e_symm_internal(bfs_coords[0], bfs_contr_prim_norms[0], bfs_lmn[0], bfs_nprim[0], bfs_coeffs, bfs_prim_norms, bfs_expnts,aux_bfs_coords[0], aux_bfs_contr_prim_norms[0], aux_bfs_lmn[0], aux_bfs_nprim[0], aux_bfs_coeffs, aux_bfs_prim_norms, aux_bfs_expnts,indx_startA, indx_endA, indx_startB, indx_endB, indx_startC, indx_endC)
    return ints3c2e

@njit(parallel=True, cache=True, fastmath=True, error_model="numpy")
def conv_3c2e_symm_internal(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts,aux_bfs_coords, aux_bfs_contr_prim_norms, aux_bfs_lmn, aux_bfs_nprim, aux_bfs_coeffs, aux_bfs_prim_norms, aux_bfs_expnts, indx_startA, indx_endA, indx_startB, indx_endB, indx_startC, indx_endC):
    # This function calculates the three-centered two electron integrals for density fitting
    # The basis object holds the information of basis functions like: exponents, coeffs, etc.
    
    # The integrals are performed using the formulas https://pubs.acs.org/doi/full/10.1021/acs.jchemed.8b00255
    # returns (AB|P) 
    
    # Infer the matrix shape from the start and end indices
    num_A = indx_endA - indx_startA 
    num_B = indx_endB - indx_startB 
    num_C = indx_endC - indx_startC
    matrix_shape = (num_A, num_B, num_C)

    
    # Check if the slice of the matrix requested falls in the lower/upper triangle or in both the triangles
    tri_symm = False
    no_symm = False
    if indx_startA==indx_startB and indx_endA==indx_endB:
        tri_symm = True
    else:
        no_symm = True
    

    # Initialize the matrix with zeros
    threeC2E = np.zeros(matrix_shape, dtype=np.float64) 
        
    pi = 3.141592653589793
    pisq = 9.869604401089358  #PI^2
    twopisq = 19.739208802178716  #2*PI^2

    L = np.zeros((3))
    ld, md, nd = int(0), int(0), int(0)
    alphalk = 0.0

    #Loop pver BFs
    for i in prange(indx_startA, indx_endA): #A
        I = bfs_coords[i]
        Ni = bfs_contr_prim_norms[i]
        lmni = bfs_lmn[i]
        la, ma, na = lmni
        nprimi = bfs_nprim[i]
        
        for j in prange(indx_startB, indx_endB): #B
            if (tri_symm and j<=i) or no_symm:
                J = bfs_coords[j]
                IJ = I - J
                IJsq = np.sum(IJ**2)
                Nj = bfs_contr_prim_norms[j]
                lmnj = bfs_lmn[j]
                lb, mb, nb = lmnj
                tempcoeff1 = Ni*Nj
                nprimj = bfs_nprim[j]
                
                for k in prange(indx_startC, indx_endC): #C
                    K = aux_bfs_coords[k]
                    Nk = aux_bfs_contr_prim_norms[k]
                    lmnk = aux_bfs_lmn[k]
                    lc, mc, nc = lmnk
                    tempcoeff2 = tempcoeff1*Nk
                    nprimk = aux_bfs_nprim[k]
                    
                    
                        
                    KL = K #- L  
                    KLsq = np.sum(KL**2)
                    
                    val = 0.0
                        
                    #Loop over primitives
                    for ik in range(nprimi):   
                        dik = bfs_coeffs[i][ik]
                        Nik = bfs_prim_norms[i][ik]
                        alphaik = bfs_expnts[i][ik]
                        tempcoeff3 = tempcoeff2*dik*Nik
                            
                        for jk in range(nprimj):
                            alphajk = bfs_expnts[j][jk]
                            gammaP = alphaik + alphajk
                            screenfactorAB = np.exp(-alphaik*alphajk/gammaP*IJsq)
                            if abs(screenfactorAB)<1.0e-8:   
                                #TODO: Check for optimal value for screening
                                continue
                            djk = bfs_coeffs[j][jk] 
                            Njk = bfs_prim_norms[j][jk]      
                            P = (alphaik*I + alphajk*J)/gammaP
                            PI = P - I
                            PJ = P - J  
                            fac1 = twopisq/gammaP*screenfactorAB   
                            onefourthgammaPinv = 0.25/gammaP  
                            tempcoeff4 = tempcoeff3*djk*Njk  
                                
                            for kk in range(nprimk):
                                dkk = aux_bfs_coeffs[k][kk]
                                Nkk = aux_bfs_prim_norms[k][kk]
                                alphakk = aux_bfs_expnts[k][kk]
                                tempcoeff5 = tempcoeff4*dkk*Nkk 
                                    
                                
                                gammaQ = alphakk #+ alphalk
                                screenfactorKL = np.exp(-alphakk*alphalk/gammaQ*KLsq)
                                if abs(screenfactorKL)<1.0e-8:   
                                    #TODO: Check for optimal value for screening
                                    continue
                                if abs(screenfactorAB*screenfactorKL)<1.0e-10:   
                                    #TODO: Check for optimal value for screening
                                    continue
                                
                                Q = K#(alphakk*K + alphalk*L)/gammaQ        
                                PQ = P - Q
                                        
                                QK = Q - K
                                QL = Q #- L
                                
                                        
                                fac2 = fac1/gammaQ
                                        

                                omega = (fac2)*np.sqrt(pi/(gammaP + gammaQ))#*screenfactorKL
                                delta = 0.25*(1/gammaQ) + onefourthgammaPinv          
                                PQsqBy4delta = np.sum(PQ**2)/(4*delta)         
                                            
                                sum1 = innerLoop4c2e(la,lb,lc,ld,ma,mb,mc,md,na,nb,nc,nd,gammaP,gammaQ,PI,PJ,QK,QL,PQ,PQsqBy4delta,delta)
                                            
                                val += omega*sum1*tempcoeff5

                        threeC2E[i-indx_startA, j-indx_startB, k-indx_startC] = val
                    
                                    
    if tri_symm:
        #We save time by evaluating only the lower diagonal elements and then use symmetry Si,j=Sj,i 
        for i in prange(indx_startA, indx_endA):
            for j in prange(indx_startB, indx_endB):
                if j<=i:
                    threeC2E[j-indx_startB, i-indx_startA, :] = threeC2E[i-indx_startA, j-indx_startB, :]       
                            
        
    return threeC2E