import numpy as np
from numba import njit, prange

from .integral_helpers import Fboys
from .rys_helpers import coulomb_rys, ChebGausInt

def rys_4c2e_symm(basis, slice=None):
    """
    Compute four-center two-electron (4c2e) electron repulsion integrals (ERIs) 
    using the Rys quadrature method with exploitation of 8-fold permutational 
    symmetry.

    This function evaluates integrals of the form (A B | C D), where A, B, C, D 
    are basis functions from the same primary basis set. It uses Numba-accelerated 
    backends and symmetry-aware optimizations to reduce computational cost.

    Symmetries exploited
    --------------------
    The 4c2e integrals obey the following permutational symmetry relations:

        (A B | C D) = (B A | C D)  # exchange within bra
                    = (A B | D C)  # exchange within ket
                    = (B A | D C)  
                    = (C D | A B)  # bra ↔ ket exchange
                    = (D C | A B)
                    = (C D | B A)
                    = (D C | B A)

    These 8 equivalent permutations mean that only a subset of integrals 
    needs to be explicitly computed; the rest can be obtained by symmetry.
    
    This reduces the number of independent integrals from:
        N_bf^4   →   N_bf*(N_bf+1)/2 * N_bf*(N_bf+1)/2
    when computing the full tensor.

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
        - bfs_shell_index : Index of the shell each basis function belongs to.
        - bfs_nao : Total number of atomic orbitals.

    slice : list of int, optional
        An 8-element list specifying a sub-block of integrals to compute:
        [start_A, end_A, start_B, end_B, start_C, end_C, start_D, end_D]
        If None (default), computes the full (Nbf, Nbf, Nbf, Nbf) block.

        **Note:** When slices are used, not all 8-fold symmetries may be 
        available because the requested block may not contain all 
        symmetric permutations. In such cases, only the applicable 
        symmetries are used.

    Returns
    -------
    ints4c2e : ndarray
        The computed 4-center 2-electron integrals for the requested range.

    Notes
    -----
    - Precomputes and stores basis set data in NumPy arrays for Numba efficiency.
    - Exploits all possible symmetry permutations when the full tensor is 
      computed (no slice) to reduce redundant work.
    - If a slice is specified, symmetry exploitation is limited to the 
      permutations that fall within the slice's index ranges.

    Examples
    --------
    >>> eri_full = rys_4c2e_symm(basis)
    >>> eri_block = rys_4c2e_symm(basis, slice=[0,5, 0,5, 0,5, 0,5])
    """
    # Here the lists are converted to numpy arrays for better use with Numba.
    # Once these conversions are done we pass these to a Numba decorated
    # function that uses prange, etc. to calculate the 4c2e integrals efficiently.
    # This function calculates the 4c2e electron-electron ERIs for a given basis object.
    # The basis object holds the information of basis functions like: exponents, coeffs, etc.

    # It is possible to only calculate a slice (block/subset) of the complete set of integrals.
    # slice is an 8 element list whose first and second elements give the range of the A functions to be calculated.
    # and so on.

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
        slice = [0, basis.bfs_nao, 0, basis.bfs_nao, 0, basis.bfs_nao, 0, basis.bfs_nao]
        
    #Limits for the calculation of 4c2e integrals
    indx_startA = int(slice[0])
    indx_endA = int(slice[1])
    indx_startB = int(slice[2])
    indx_endB = int(slice[3])
    indx_startC = int(slice[4])
    indx_endC = int(slice[5])
    indx_startD = int(slice[6])
    indx_endD = int(slice[7])

    ints4c2e = rys_4c2e_symm_internal(bfs_coords[0], bfs_contr_prim_norms[0], bfs_lmn[0], bfs_nprim[0], bfs_coeffs, bfs_prim_norms, bfs_expnts,\
        indx_startA,indx_endA, indx_startB, indx_endB, indx_startC, indx_endC, indx_startD,indx_endD)

    return ints4c2e

@njit(parallel=True, cache=True, fastmath=True, error_model="numpy")
def rys_4c2e_symm_internal(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts, indx_startA, indx_endA, indx_startB, indx_endB, indx_startC, indx_endC, indx_startD, indx_endD):
    #  Based on Rys Quadrature from https://github.com/rpmuller/MolecularIntegrals.jl
    # This function calculates the 4D electron-electron repulsion integrals (ERIs) array for a given basis object and mol object.
    # This uses 8 fold symmetry to only calculate the unique elements and assign the rest via symmetry
    # The basis object holds the information of basis functions like: exponents, coeffs, etc.
    
    # returns (AB|CD) 

    # Infer the matrix shape from the start and end indices
    num_A = indx_endA - indx_startA 
    num_B = indx_endB - indx_startB 
    num_C = indx_endC - indx_startC 
    num_D = indx_endD - indx_startD 
    array_shape = (num_A, num_B, num_C, num_D)

    
    # Check if the slice of the matrix requested has some symmetries that can be used
    all_symm = False
    left_side_symm = False
    right_side_symm = False
    both_left_right_symm = False
    no_symm = False
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
    

    # Initialize the 4c2e array with zeros
    fourC2E = np.zeros(array_shape, dtype=np.float64) 
        
    sRys = np.ones((12,12))
        
    pi = 3.141592653589793
    pisq = 9.869604401089358  #PI^2
    twopisq = 19.739208802178716  #2*PI^2

    #Loop pver BFs
    for i in prange(indx_startA, indx_endA): #A
        I = bfs_coords[i]
        Ni = bfs_contr_prim_norms[i]
        lmni = bfs_lmn[i]
        la, ma, na = lmni
        nprimi = bfs_nprim[i]
        
        for j in range(indx_startB, indx_endB): #B
            if (all_symm and j<=i) or (left_side_symm and j<=i) or right_side_symm or no_symm or (both_left_right_symm and j<=i):
                # Take care of further symmetries
                if all_symm:
                    if i<j:
                        triangle2ij = (j)*(j+1)/2+i
                    else:
                        triangle2ij = (i)*(i+1)/2+j
                J = bfs_coords[j]
                IJ = I - J
                IJsq = np.sum(IJ**2)
                Nj = bfs_contr_prim_norms[j]
                lmnj = bfs_lmn[j]
                lb, mb, nb = lmnj
                tempcoeff1 = Ni*Nj
                nprimj = bfs_nprim[j]
                
                for k in range(indx_startC, indx_endC): #C
                    K = bfs_coords[k]
                    Nk = bfs_contr_prim_norms[k]
                    lmnk = bfs_lmn[k]
                    lc, mc, nc = lmnk
                    tempcoeff2 = tempcoeff1*Nk
                    nprimk = bfs_nprim[k]
                    
                    for l in range(indx_startD, indx_endD): #D
                        if (all_symm and l<=k) or (right_side_symm and l<=k) or (left_side_symm and j<=i) or no_symm or (both_left_right_symm and l<=k):
                            # Take care of further symmetries
                            if all_symm:
                                if k<l:
                                    triangle2kl = (l)*(l+1)/2+k
                                else:
                                    triangle2kl = (k)*(k+1)/2+l
                                if triangle2ij>triangle2kl:
                                    continue
                            L = bfs_coords[l]
                            KL = K - L  
                            KLsq = np.sum(KL**2)
                            Nl = bfs_contr_prim_norms[l]
                            lmnl = bfs_lmn[l]  
                            ld, md, nd = lmnl
                            tempcoeff3 = tempcoeff2*Nl
                            npriml = bfs_nprim[l]

                            norder = int((la+ma+na+lb+mb+nb+lc+mc+nc+ld+md+nd)/2 + 1 ) 
                            n = int(max(la+lb,ma+mb,na+nb))
                            m = int(max(lc+ld,mc+md,nc+nd))
                            roots = np.zeros((norder))
                            weights = np.zeros((norder))
                            G = np.zeros((n+1,m+1))

                            val = 0.0
                            
                            #Loop over primitives
                            for ik in range(nprimi):   
                                dik = bfs_coeffs[i][ik]
                                Nik = bfs_prim_norms[i][ik]
                                alphaik = bfs_expnts[i][ik]
                                tempcoeff4 = tempcoeff3*dik*Nik
                                
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
                                    tempcoeff5 = tempcoeff4*djk*Njk  
                                    
                                    for kk in range(nprimk):
                                        dkk = bfs_coeffs[k][kk]
                                        Nkk = bfs_prim_norms[k][kk]
                                        alphakk = bfs_expnts[k][kk]
                                        tempcoeff6 = tempcoeff5*dkk*Nkk 
                                        
                                        for lk in range(npriml): 
                                            alphalk = bfs_expnts[l][lk]
                                            gammaQ = alphakk + alphalk
                                            screenfactorKL = np.exp(-alphakk*alphalk/gammaQ*KLsq)
                                            if abs(screenfactorKL)<1.0e-8:   
                                                #TODO: Check for optimal value for screening
                                                continue
                                            if abs(screenfactorAB*screenfactorKL)<1.0e-10:   
                                                #TODO: Check for optimal value for screening
                                                continue
                                            dlk = bfs_coeffs[l][lk] 
                                            Nlk = bfs_prim_norms[l][lk]     
                                            Q = (alphakk*K + alphalk*L)/gammaQ        
                                            PQ = P - Q

                                            PQsq = np.sum(PQ**2)
                                            rho = gammaP*gammaQ/(gammaP+gammaQ)
                                            
                                            tempcoeff7 = tempcoeff6*dlk*Nlk
                                            
                                            if norder<=10:
                                                val += tempcoeff7*coulomb_rys(roots,weights,G,PQsq, rho, norder,n,m,la,lb,lc,ld,ma,mb,mc,md,na,nb,nc,nd,alphaik, alphajk, alphakk, alphalk,I,J,K,L)
                                            else:
                                                T = rho*PQsq
                                                sERI = 34.9868366552497256925 * (screenfactorAB+screenfactorKL) / ((gammaP*gammaQ) * np.sqrt(gammaP + gammaQ))
                                                val += tempcoeff7*sERI*ChebGausInt(1E-8,50000, gammaP, gammaQ, la, lb, lc, ld,ma, mb, mc,  md, na, nb, nc,nd, I[0],
                                                                                J[0],  K[0], L[0],  I[1],  J[1],  K[1], L[1], I[2],  J[2],  K[2],
                                                                                L[2],  P[0], P[1], P[2], Q[0], Q[1], Q[2], T, sRys)
                                            
                            
                            fourC2E[i-indx_startA, j-indx_startB, k-indx_startC, l-indx_startD] = val
                        


    # Symmetries that can be used (for reference)
    # fourC2E[j,i,k,l] = fourC2E[i,j,k,l]
    # fourC2E[i,j,l,k] = fourC2E[i,j,k,l]
    # fourC2E[j,i,l,k] = fourC2E[i,j,k,l]
    # fourC2E[k,l,i,j] = fourC2E[i,j,k,l]
    # fourC2E[k,l,j,i] = fourC2E[i,j,k,l]
    # fourC2E[l,k,i,j] = fourC2E[i,j,k,l]
    # fourC2E[l,k,j,i] = fourC2E[i,j,k,l]
    
    
    if all_symm:
        # Fill the remaining values of the array using symmetries 
        for i in range(indx_startA, indx_endA):
            for j in range(indx_startB, indx_endB):
                if j<=i:
                    for k in prange(indx_startC, indx_endC):
                        for l in prange(indx_startD, indx_endD):
                            val = fourC2E[i-indx_startA, j-indx_startB, k-indx_startC, l-indx_startD]
                            if l<=k:
                                fourC2E[j-indx_startB, i-indx_startA, k-indx_startC, l-indx_startD] = val
                                fourC2E[i-indx_startA, j-indx_startB, l-indx_startD, k-indx_startC] = val
                                fourC2E[j-indx_startB, i-indx_startA, l-indx_startD, k-indx_startC] = val
                                fourC2E[k-indx_startC, l-indx_startD, i-indx_startA, j-indx_startB] = val
                                fourC2E[k-indx_startC, l-indx_startD, j-indx_startB, i-indx_startA] = val
                                fourC2E[l-indx_startD, k-indx_startC, i-indx_startA, j-indx_startB] = val
                                fourC2E[l-indx_startD, k-indx_startC, j-indx_startB, i-indx_startA] = val  

    if left_side_symm:
        # Fill the remaining values of the array using symmetries 
        for i in range(indx_startA, indx_endA):
            for j in range(indx_startB, indx_endB):
                if j<=i:
                    fourC2E[j-indx_startB, i-indx_startA, :, :] = fourC2E[i-indx_startA, j-indx_startB, :, :] 
    
    if right_side_symm:
        # Fill the remaining values of the array using symmetries 
        for k in range(indx_startC, indx_endC):
            for l in range(indx_startD, indx_endD):
                if l<=k:
                    fourC2E[:, :, l-indx_startD, k-indx_startC] = fourC2E[:, :, k-indx_startC, l-indx_startD] 
                                   
    if both_left_right_symm:
        # Fill the remaining values of the array using symmetries 
        for i in range(indx_startA, indx_endA):
            for j in range(indx_startB, indx_endB):
                if j<=i:
                    for k in prange(indx_startC, indx_endC):
                        for l in prange(indx_startD, indx_endD):
                            val = fourC2E[i-indx_startA, j-indx_startB, k-indx_startC, l-indx_startD]
                            if l<=k:
                                fourC2E[j-indx_startB, i-indx_startA, k-indx_startC, l-indx_startD] = val
                                fourC2E[i-indx_startA, j-indx_startB, l-indx_startD, k-indx_startC] = val
                                fourC2E[j-indx_startB, i-indx_startA, l-indx_startD, k-indx_startC] = val 
                            
        
    return fourC2E

def rys_4c2e_symm_old(basis, slice=None):
    """
    Compute four-center two-electron (4c2e) electron repulsion integrals (ERIs) 
    using the Rys quadrature method with exploitation of 8-fold permutational 
    symmetry.

    This function evaluates integrals of the form (A B | C D), where A, B, C, D 
    are basis functions from the same primary basis set. It uses Numba-accelerated 
    backends and symmetry-aware optimizations to reduce computational cost.

    Symmetries exploited
    --------------------
    The 4c2e integrals obey the following permutational symmetry relations:

        (A B | C D) = (B A | C D)   # exchange within bra
                    = (A B | D C)  # exchange within ket
                    = (B A | D C)  
                    = (C D | A B)  # bra ↔ ket exchange
                    = (D C | A B)
                    = (C D | B A)
                    = (D C | B A)

    These 8 equivalent permutations mean that only a subset of integrals 
    needs to be explicitly computed; the rest can be obtained by symmetry.
    
    This reduces the number of independent integrals from:
        N_bf^4   →   N_bf*(N_bf+1)/2 * N_bf*(N_bf+1)/2
    when computing the full tensor.

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
        - bfs_shell_index : Index of the shell each basis function belongs to.
        - bfs_nao : Total number of atomic orbitals.

    slice : list of int, optional
        An 8-element list specifying a sub-block of integrals to compute:
        [start_A, end_A, start_B, end_B, start_C, end_C, start_D, end_D]
        If None (default), computes the full (Nbf, Nbf, Nbf, Nbf) block.

        **Note:** When slices are used, not all 8-fold symmetries may be 
        available because the requested block may not contain all 
        symmetric permutations. In such cases, only the applicable 
        symmetries are used.

    Returns
    -------
    ints4c2e : ndarray
        The computed 4-center 2-electron integrals for the requested range.

    Notes
    -----
    - Precomputes and stores basis set data in NumPy arrays for Numba efficiency.
    - Exploits all possible symmetry permutations when the full tensor is 
      computed (no slice) to reduce redundant work.
    - If a slice is specified, symmetry exploitation is limited to the 
      permutations that fall within the slice's index ranges.

    Examples
    --------
    >>> eri_full = rys_4c2e_symm(basis)
    >>> eri_block = rys_4c2e_symm(basis, slice=[0,5, 0,5, 0,5, 0,5])
    """
    # Here the lists are converted to numpy arrays for better use with Numba.
    # Once these conversions are done we pass these to a Numba decorated
    # function that uses prange, etc. to calculate the 4c2e integrals efficiently.
    # This function calculates the 4c2e electron-electron ERIs for a given basis object.
    # The basis object holds the information of basis functions like: exponents, coeffs, etc.

    # It is possible to only calculate a slice (block/subset) of the complete set of integrals.
    # slice is an 8 element list whose first and second elements give the range of the A functions to be calculated.
    # and so on.

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
    shell_indices = np.array([basis.bfs_shell_index], dtype=np.uint16)[0]
    for i in range(basis.bfs_nao):
        for j in range(basis.bfs_nprim[i]):
            bfs_coeffs[i,j] = basis.bfs_coeffs[i][j]
            bfs_expnts[i,j] = basis.bfs_expnts[i][j]
            bfs_prim_norms[i,j] = basis.bfs_prim_norms[i][j]
        


    
    if slice is None:
        slice = [0, basis.bfs_nao, 0, basis.bfs_nao, 0, basis.bfs_nao, 0, basis.bfs_nao]
        
    #Limits for the calculation of 4c2e integrals
    indx_startA = int(slice[0])
    indx_endA = int(slice[1])
    indx_startB = int(slice[2])
    indx_endB = int(slice[3])
    indx_startC = int(slice[4])
    indx_endC = int(slice[5])
    indx_startD = int(slice[6])
    indx_endD = int(slice[7])


    ints4c2e = rys_4c2e_symm_internal(bfs_coords[0], bfs_contr_prim_norms[0], bfs_lmn[0], bfs_nprim[0], bfs_coeffs, bfs_prim_norms, bfs_expnts, shell_indices,\
        indx_startA,indx_endA, indx_startB, indx_endB, indx_startC, indx_endC, indx_startD,indx_endD)

    return ints4c2e

@njit(parallel=False, cache=True, fastmath=True, error_model="numpy")
def rys_4c2e_symm_internal_old(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts, shell_indices, indx_startA, indx_endA, indx_startB, indx_endB, indx_startC, indx_endC, indx_startD, indx_endD):
    #  Based on Rys Quadrature from https://github.com/rpmuller/MolecularIntegrals.jl
    # This function calculates the 4D electron-electron repulsion integrals (ERIs) array for a given basis object and mol object.
    # This uses 8 fold symmetry to only calculate the unique elements and assign the rest via symmetry
    # The basis object holds the information of basis functions like: exponents, coeffs, etc.
    
    # returns (AB|CD) 

    # Infer the matrix shape from the start and end indices
    num_A = indx_endA - indx_startA 
    num_B = indx_endB - indx_startB 
    num_C = indx_endC - indx_startC 
    num_D = indx_endD - indx_startD 
    array_shape = (num_A, num_B, num_C, num_D)

    
    # Check if the slice of the matrix requested has some symmetries that can be used
    all_symm = False
    left_side_symm = False
    right_side_symm = False
    both_left_right_symm = False
    no_symm = False
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
    

    # Initialize the 4c2e array with zeros
    fourC2E = np.zeros(array_shape, dtype=np.float64) 
        
    sRys = np.ones((12,12))
        
    pi = 3.141592653589793
    pisq = 9.869604401089358  #PI^2
    twopisq = 19.739208802178716  #2*PI^2

    maxprims = bfs_coeffs.shape[1]

    #Loop pver BFs
    for i in range(indx_startA, indx_endA): #A
        I = bfs_coords[i]
        Ni = bfs_contr_prim_norms[i]
        lmni = bfs_lmn[i]
        la, ma, na = lmni
        nprimi = bfs_nprim[i]

        jshell_previous = -12 # Some random number
        kshell_previous = -10 # Some random number
        lshell_previous = -122 # Some random number
        gammaP = np.zeros((maxprims, maxprims), dtype=np.float64) # Should be Hoisted out
        screenfactorAB = np.zeros((maxprims, maxprims), dtype=np.float64) # Should be Hoisted out
        gammaQ = np.zeros((maxprims, maxprims), dtype=np.float64) # Should be Hoisted out
        screenfactorKL = np.zeros((maxprims, maxprims), dtype=np.float64) # Should be Hoisted out
        
        for j in range(indx_startB, indx_endB): #B
            if (all_symm and j<=i) or (left_side_symm and j<=i) or right_side_symm or no_symm or (both_left_right_symm and j<=i):
                # Take care of further symmetries
                if all_symm:
                    if i<j:
                        triangle2ij = (j)*(j+1)/2+i
                    else:
                        triangle2ij = (i)*(i+1)/2+j

                lmnj = bfs_lmn[j]
                lb, mb, nb = lmnj
                jshell = shell_indices[j]
                if jshell!=jshell_previous:
                    J = bfs_coords[j]
                    nprimj = bfs_nprim[j]
                    Nj = bfs_contr_prim_norms[j]
                    
                    IJ = I - J
                    IJsq = np.sum(IJ**2)
                    tempcoeff1 = Ni*Nj

                
                
                for k in range(indx_startC, indx_endC): #C
                    lmnk = bfs_lmn[k]
                    lc, mc, nc = lmnk
                    kshell = shell_indices[k]
                    if kshell!=kshell_previous:
                        K = bfs_coords[k]
                        Nk = bfs_contr_prim_norms[k]
                        nprimk = bfs_nprim[k]
                    
                    if jshell!=jshell_previous or kshell!=kshell_previous:
                        tempcoeff2 = tempcoeff1*Nk
                        
                        
                    for l in range(indx_startD, indx_endD): #D
                        if (all_symm and l<=k) or (right_side_symm and l<=k) or (left_side_symm and j<=i) or no_symm or (both_left_right_symm and l<=k):
                            # Take care of further symmetries
                            if all_symm:
                                if k<l:
                                    triangle2kl = (l)*(l+1)/2+k
                                else:
                                    triangle2kl = (k)*(k+1)/2+l
                                if triangle2ij>triangle2kl:
                                    continue

                            lmnl = bfs_lmn[l]  
                            ld, md, nd = lmnl
                            lshell = shell_indices[l]
                            if lshell!=lshell_previous:
                                L = bfs_coords[l]
                                npriml = bfs_nprim[l]
                                Nl = bfs_contr_prim_norms[l]
                                     
                            if kshell!=kshell_previous or lshell!=lshell_previous:
                                KL = K - L  
                                KLsq = np.sum(KL**2)

                            if jshell!=jshell_previous or kshell!=kshell_previous or lshell!=lshell_previous:
                                tempcoeff3 = tempcoeff2*Nl


                            norder = int((la+ma+na+lb+mb+nb+lc+mc+nc+ld+md+nd)/2 + 1 ) 
                            n = int(max(la+lb,ma+mb,na+nb))
                            m = int(max(lc+ld,mc+md,nc+nd))
                            # roots = np.zeros((norder))
                            # weights = np.zeros((norder))
                            roots = np.zeros((10)) 
                            weights = np.zeros((10))
                            G = np.zeros((n+1,m+1))

                            val = 0.0
                            
                            #Loop over primitives
                            for ik in range(nprimi):   
                                dik = bfs_coeffs[i][ik]
                                Nik = bfs_prim_norms[i][ik]
                                alphaik = bfs_expnts[i][ik]
                                tempcoeff4 = tempcoeff3*dik*Nik
                                
                                for jk in range(nprimj):
                                    alphajk = bfs_expnts[j][jk]
                                    # gammaP = alphaik + alphajk
                                    # screenfactorAB = np.exp(-alphaik*alphajk/gammaP*IJsq)
                                    if jshell!=jshell_previous:
                                        gammaP[ik,jk] = alphaik + alphajk
                                        screenfactorAB[ik,jk] = np.exp(-alphaik*alphajk/gammaP[ik,jk]*IJsq)
                                    # if abs(screenfactorAB)<1.0e-8:
                                    if abs(screenfactorAB[ik,jk])<1.0e-8:     
                                        #TODO: Check for optimal value for screening
                                        continue
                                    djk = bfs_coeffs[j][jk] 
                                    Njk = bfs_prim_norms[j][jk]      
                                    # P = (alphaik*I + alphajk*J)/gammaP
                                    P = (alphaik*I + alphajk*J)/gammaP[ik,jk]
                                    # PI = P - I
                                    # PJ = P - J  
                                    # fac1 = twopisq/gammaP*screenfactorAB 
                                    # fac1 = twopisq/gammaP[ik,jk]*screenfactorAB[ik,jk]   
                                    # onefourthgammaPinv = 0.25/gammaP  
                                    # onefourthgammaPinv = 0.25/gammaP[ik,jk]
                                    tempcoeff5 = tempcoeff4*djk*Njk  
                                    
                                    for kk in range(nprimk):
                                        dkk = bfs_coeffs[k][kk]
                                        Nkk = bfs_prim_norms[k][kk]
                                        alphakk = bfs_expnts[k][kk]
                                        tempcoeff6 = tempcoeff5*dkk*Nkk 
                                        
                                        for lk in range(npriml): 
                                            alphalk = bfs_expnts[l][lk]
                                            # gammaQ = alphakk + alphalk
                                            # screenfactorKL = np.exp(-alphakk*alphalk/gammaQ*KLsq)
                                            if kshell!=kshell_previous or lshell!=lshell_previous:
                                                gammaQ[kk,lk] = alphakk + alphalk
                                                screenfactorKL[kk,lk] = np.exp(-alphakk*alphalk/gammaQ[kk,lk]*KLsq)
                                            # if abs(screenfactorKL)<1.0e-8: 
                                            if abs(screenfactorKL[kk,lk])<1.0e-8:   
                                                #TODO: Check for optimal value for screening
                                                continue
                                            # if abs(screenfactorAB*screenfactorKL)<1.0e-10:  
                                            if abs(screenfactorAB[ik,jk]*screenfactorKL[kk,lk])<1.0e-10:  
                                                #TODO: Check for optimal value for screening
                                                continue
                                            dlk = bfs_coeffs[l][lk] 
                                            Nlk = bfs_prim_norms[l][lk]     
                                            # Q = (alphakk*K + alphalk*L)/gammaQ     
                                            Q = (alphakk*K + alphalk*L)/gammaQ[kk,lk]     
                                            PQ = P - Q

                                            PQsq = np.sum(PQ**2)
                                            tempcoeff7 = tempcoeff6*dlk*Nlk

                                            # (ss|ss) case
                                            if (la+ma+na+lb+mb+nb+lc+mc+nc+ld+md+nd)==0: 
                                                val += tempcoeff7*twopisq/(gammaP[ik,jk]*gammaQ[kk,lk])*np.sqrt(pi/(gammaP[ik,jk]+gammaQ[kk,lk]))\
                                                    *screenfactorAB[ik,jk]*screenfactorKL[kk,lk]*Fboys(0,PQsq/(1/gammaP[ik,jk]+1/gammaQ[kk,lk]))
                                            else:
                                                # rho = gammaP*gammaQ/(gammaP+gammaQ)
                                                rho = gammaP[ik,jk]*gammaQ[kk,lk]/(gammaP[ik,jk]+gammaQ[kk,lk])
                                                if norder<=10:
                                                    val += tempcoeff7*coulomb_rys(roots,weights,G,PQsq, rho, norder,n,m,la,lb,lc,ld,ma,mb,mc,md,na,nb,nc,nd,alphaik, alphajk, alphakk, alphalk,I,J,K,L)
                                                    
                                                else:
                                                    T = rho*PQsq
                                                    # sERI = 34.9868366552497256925 * (screenfactorAB+screenfactorKL) / ((gammaP*gammaQ) * np.sqrt(gammaP + gammaQ))
                                                    # val += tempcoeff7*sERI*ChebGausInt(1E-8,50000, gammaP, gammaQ, la, lb, lc, ld,ma, mb, mc,  md, na, nb, nc,nd, I[0],
                                                    #                                 J[0],  K[0], L[0],  I[1],  J[1],  K[1], L[1], I[2],  J[2],  K[2],
                                                    #                                 L[2],  P[0], P[1], P[2], Q[0], Q[1], Q[2], T, sRys)
                                                    sERI = 34.9868366552497256925 * (screenfactorAB[ik,jk]+screenfactorKL[kk,lk]) / ((gammaP[ik,jk]*gammaQ[kk,lk]) * np.sqrt(gammaP[ik,jk] + gammaQ[kk,lk]))
                                                    val += tempcoeff7*sERI*ChebGausInt(1E-8,50000, gammaP[ik,jk], gammaQ[kk,lk], la, lb, lc, ld,ma, mb, mc,  md, na, nb, nc,nd, I[0],
                                                                                    J[0],  K[0], L[0],  I[1],  J[1],  K[1], L[1], I[2],  J[2],  K[2],
                                                                                    L[2],  P[0], P[1], P[2], Q[0], Q[1], Q[2], T, sRys)
                                                
                                            
                            
                            lshell_previous = lshell
                            fourC2E[i-indx_startA, j-indx_startB, k-indx_startC, l-indx_startD] = val

                    kshell_previous = kshell    
                jshell_previous = jshell

    # Symmetries that can be used (for reference)
    # fourC2E[j,i,k,l] = fourC2E[i,j,k,l]
    # fourC2E[i,j,l,k] = fourC2E[i,j,k,l]
    # fourC2E[j,i,l,k] = fourC2E[i,j,k,l]
    # fourC2E[k,l,i,j] = fourC2E[i,j,k,l]
    # fourC2E[k,l,j,i] = fourC2E[i,j,k,l]
    # fourC2E[l,k,i,j] = fourC2E[i,j,k,l]
    # fourC2E[l,k,j,i] = fourC2E[i,j,k,l]
    
    
    if all_symm:
        # Fill the remaining values of the array using symmetries 
        for i in range(indx_startA, indx_endA):
            for j in range(indx_startB, indx_endB):
                if j<=i:
                    for k in range(indx_startC, indx_endC):
                        for l in range(indx_startD, indx_endD):
                            val = fourC2E[i-indx_startA, j-indx_startB, k-indx_startC, l-indx_startD]
                            if l<=k:
                                fourC2E[j-indx_startB, i-indx_startA, k-indx_startC, l-indx_startD] = val
                                fourC2E[i-indx_startA, j-indx_startB, l-indx_startD, k-indx_startC] = val
                                fourC2E[j-indx_startB, i-indx_startA, l-indx_startD, k-indx_startC] = val
                                fourC2E[k-indx_startC, l-indx_startD, i-indx_startA, j-indx_startB] = val
                                fourC2E[k-indx_startC, l-indx_startD, j-indx_startB, i-indx_startA] = val
                                fourC2E[l-indx_startD, k-indx_startC, i-indx_startA, j-indx_startB] = val
                                fourC2E[l-indx_startD, k-indx_startC, j-indx_startB, i-indx_startA] = val  

    if left_side_symm:
        # Fill the remaining values of the array using symmetries 
        for i in range(indx_startA, indx_endA):
            for j in range(indx_startB, indx_endB):
                if j<=i:
                    fourC2E[j-indx_startB, i-indx_startA, :, :] = fourC2E[i-indx_startA, j-indx_startB, :, :] 
    
    if right_side_symm:
        # Fill the remaining values of the array using symmetries 
        for k in range(indx_startC, indx_endC):
            for l in range(indx_startD, indx_endD):
                if l<=k:
                    fourC2E[:, :, l-indx_startD, k-indx_startC] = fourC2E[:, :, k-indx_startC, l-indx_startD] 
                                
    if both_left_right_symm:
        # Fill the remaining values of the array using symmetries 
        for i in range(indx_startA, indx_endA):
            for j in range(indx_startB, indx_endB):
                if j<=i:
                    for k in range(indx_startC, indx_endC):
                        for l in range(indx_startD, indx_endD):
                            val = fourC2E[i-indx_startA, j-indx_startB, k-indx_startC, l-indx_startD]
                            if l<=k:
                                fourC2E[j-indx_startB, i-indx_startA, k-indx_startC, l-indx_startD] = val
                                fourC2E[i-indx_startA, j-indx_startB, l-indx_startD, k-indx_startC] = val
                                fourC2E[j-indx_startB, i-indx_startA, l-indx_startD, k-indx_startC] = val 
                            
        
    return fourC2E

