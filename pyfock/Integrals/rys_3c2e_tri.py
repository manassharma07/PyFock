import numpy as np
from numba import njit, prange

from .integral_helpers import innerLoop4c2e
from .rys_helpers import coulomb_rys


def rys_3c2e_tri(basis, auxbasis):
    """
    Compute three-center two-electron (3c2e) electron repulsion integrals using
    the Rys quadrature method with symmetry considerations, stored in packed
    triangular form for memory efficiency.

    This function evaluates integrals of the form (A B | C), where A and B are
    basis functions from a primary basis set and C is from an auxiliary basis set.
    Symmetry in the first two indices is exploited ((A B | C) = (B A | C)),
    so only the upper-triangular part of the (A,B) block is computed and stored.
    The resulting array has shape (N_{bf}*(N_{bf}+1)/2, N_{auxbf}), where the first
    index corresponds to a flattened 1D triangular index over basis function pairs.

    Unlike `rys_3c2e_symm`, this routine does not support computing arbitrary slices
    — the full triangular set is always evaluated.

    Parameters
    ----------
    basis : object
        Primary basis set object containing information about atomic orbitals, such as:
        - bfs_coords : Cartesian coordinates of basis function centers.
        - bfs_coeffs : Contraction coefficients.
        - bfs_expnts : Gaussian exponents.
        - bfs_prim_norms : Primitive normalization constants.
        - bfs_contr_prim_norms : Contraction normalization factors.
        - bfs_lmn : Angular momentum quantum numbers (ℓ, m, n).
        - bfs_nprim : Number of primitives per basis function.
        - bfs_nao : Total number of atomic orbitals.

    auxbasis : object
        Auxiliary basis set object with the same attributes as `basis`,
        typically used for resolution-of-the-identity (RI) expansions.

    Returns
    -------
    ints3c2e : ndarray of shape (Nbf*(Nbf+1)//2, Nauxbf)
        The computed 3-center 2-electron integrals in packed triangular form.
        The mapping from a pair (i, j) with i ≤ j to the first dimension index
        follows standard upper-triangular packing order.

    Notes
    -----
    - This is a memory-efficient variant of `rys_3c2e_symm` that avoids storing
      the full (Nbf, Nbf, Nauxbf) array.
    - Uses preallocated NumPy arrays for primitive data to ensure efficient Numba processing.
    - Handles irregular contraction patterns by padding primitive arrays to the size
      of the largest contraction in the set.
    - No Schwarz screening or partial computation is available in this function.

    Examples
    --------
    >>> ints_tri = rys_3c2e_tri(basis, auxbasis)
    >>> # Retrieve value for pair (i, j) and auxiliary k:
    >>> def packed_index(i, j, Nbf):
    ...     if i > j:
    ...         i, j = j, i
    ...     return i * Nbf - i*(i-1)//2 + (j - i)
    >>> val = ints_tri[packed_index(i, j, basis.bfs_nao), k]
    """
    # This is a memory efficient version of the rys_3c2e_symm().
    # Instead of returning an array of shape (Nbf, Nbf, Nauxbf) it returns
    # an array of shape (Nbf*(Nbf+1)/2, Nauxbf), where the first index corresponds to the
    # 1D triangular matrix.

    # You cannot provide a slice here to calculate only a subset.

    # Here the lists are converted to numpy arrays for better use with Numba.
    # Once these conversions are done we pass these to a Numba decorated
    # function that uses prange, etc. to calculate the 3c2e integrals efficiently.
    # This function calculates the 3c2e electron-electron ERIs for a given basis object and auxbasis object.
    # The basis object holds the information of basis functions like: exponents, coeffs, etc.

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
            

    ints3c2e = rys_3c2e_tri_internal(bfs_coords[0], bfs_contr_prim_norms[0], bfs_lmn[0], bfs_nprim[0], bfs_coeffs, bfs_prim_norms, bfs_expnts,aux_bfs_coords[0], aux_bfs_contr_prim_norms[0], aux_bfs_lmn[0], aux_bfs_nprim[0], aux_bfs_coeffs, aux_bfs_prim_norms, aux_bfs_expnts, basis.bfs_nao, auxbasis.bfs_nao)
    return ints3c2e

@njit(parallel=True, cache=True, fastmath=True, error_model="numpy")
def rys_3c2e_tri_internal(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts,aux_bfs_coords, aux_bfs_contr_prim_norms, aux_bfs_lmn, aux_bfs_nprim, aux_bfs_coeffs, aux_bfs_prim_norms, aux_bfs_expnts, nbf, naux):
    # This function is a memory efficient version of rys_3c2e_symm_internal.
    # This does not support slicing and only returns a 2D array instead of a 3D array.
    # This function calculates the three-centered two electron integrals for density fitting
    # The basis object holds the information of basis functions like: exponents, coeffs, etc.
    
    # The integrals are performed using the formulas https://pubs.acs.org/doi/full/10.1021/acs.jchemed.8b00255
    # returns (AB|P) 
    
    # Infer the matrix shape from the start and end indices
    matrix_shape = (int(nbf*(nbf+1)/2.0), naux)

    # Initialize the matrix with zeros
    threeC2E = np.zeros(matrix_shape, dtype=np.float64) 
        
    pi = 3.141592653589793
    pisq = 9.869604401089358  #PI^2
    twopisq = 19.739208802178716  #2*PI^2

    L = np.zeros((3))
    ld, md, nd = int(0), int(0), int(0)
    alphalk = 0.0

    #Loop pver BFs
    for i in prange(0, nbf): #A
        offset = int(i*(i+1)/2)
        I = bfs_coords[i]
        Ni = bfs_contr_prim_norms[i]
        lmni = bfs_lmn[i]
        la, ma, na = lmni
        nprimi = bfs_nprim[i]
        
        for j in prange(0, i+1): #B
            J = bfs_coords[j]
            IJ = I - J
            IJsq = np.sum(IJ**2)
            Nj = bfs_contr_prim_norms[j]
            lmnj = bfs_lmn[j]
            lb, mb, nb = lmnj
            tempcoeff1 = Ni*Nj
            nprimj = bfs_nprim[j]
            
            for k in prange(0,  naux): #C
                K = aux_bfs_coords[k]
                Nk = aux_bfs_contr_prim_norms[k]
                lmnk = aux_bfs_lmn[k]
                lc, mc, nc = lmnk
                tempcoeff2 = tempcoeff1*Nk
                nprimk = aux_bfs_nprim[k]
                
                
                    
                KL = K #- L  
                KLsq = np.sum(KL**2)
                
                norder = int((la+ma+na+lb+mb+nb+lc+mc+nc+ld+md+nd)/2 + 1 ) 
                val = 0.0

                if norder<=10: # Use rys quadrature
                    n = int(max(la+lb,ma+mb,na+nb))
                    m = int(max(lc+ld,mc+md,nc+nd))
                    roots = np.zeros((norder))
                    weights = np.zeros((norder))
                    G = np.zeros((n+1,m+1))
                    
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
                                
                                Q = K        
                                PQ = P - Q
                                PQsq = np.sum(PQ**2)
                                rho = gammaP*gammaQ/(gammaP+gammaQ)
                                        
                                        
                                        
                                        
                                val += tempcoeff5*coulomb_rys(roots,weights,G,PQsq, rho, norder,n,m,la,lb,lc,ld,ma,mb,mc,md,na,nb,nc,nd,alphaik, alphajk, alphakk, alphalk,I,J,K,L)
                                    

                else: # Analytical (Conventional)
                    #Loop over primitives
                    for ik in range(bfs_nprim[i]):   
                        dik = bfs_coeffs[i][ik]
                        Nik = bfs_prim_norms[i][ik]
                        alphaik = bfs_expnts[i][ik]
                        tempcoeff3 = tempcoeff2*dik*Nik
                            
                        for jk in range(bfs_nprim[j]):
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
                                
                            for kk in range(aux_bfs_nprim[k]):
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

                threeC2E[j+offset, k] = val
                    
                                          
                            
        
    return threeC2E