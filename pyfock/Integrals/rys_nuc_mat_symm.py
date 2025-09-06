import numpy as np
from numba import njit, prange

from .integral_helpers import innerLoop4c2e
from .rys_helpers import coulomb_rys


def rys_nuc_mat_symm(basis, mol, slice=None):
    # Uses the rys 3c2e algorithm to calculate the nuclear attraction integrals by making one of the basis function as a steep/sharp s-primitive gaussian
    # Ref: https://arxiv.org/pdf/2302.11307.pdf
    # Here the lists are converted to numpy arrays for better use with Numba.
    # Once these conversions are done we pass these to a Numba decorated
    # function that uses prange, etc. to calculate the 3c2e integrals efficiently.
    # This function calculates the 3c2e electron-electron ERIs for a given basis object and auxbasis object.
    # The basis object holds the information of basis functions like: exponents, coeffs, etc.
    

    # It is possible to only calculate a slice (block/subset) of the complete set of integrals.
    # slice is a 6 element list whose first and second elements give the range of the A functions to be calculated.
    # and so on.
    # slice = [indx_startA, indx_endA, indx_startB, indx_endB, indx_startC, indx_endC]

    #We convert the required properties to numpy arrays as this is what Numba likes.
    bfs_coords = np.array([basis.bfs_coords])
    bfs_contr_prim_norms = np.array([basis.bfs_contr_prim_norms])
    bfs_lmn = np.array([basis.bfs_lmn])
    bfs_nprim = np.array([basis.bfs_nprim])

    coordsBohrs = np.array([mol.coordsBohrs])
    Z = np.array([mol.Zcharges])
    natoms = mol.natoms
        

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
        
    #Limits for the calculation of 4c2e integrals
    indx_startA = int(slice[0])
    indx_endA = int(slice[1])
    indx_startB = int(slice[2])
    indx_endB = int(slice[3])

    ints3c2e = rys_nuc_symm_internal(bfs_coords[0], bfs_contr_prim_norms[0], bfs_lmn[0], bfs_nprim[0], bfs_coeffs, bfs_prim_norms, bfs_expnts, indx_startA, indx_endA, indx_startB, indx_endB, Z[0], coordsBohrs[0], natoms)
    return ints3c2e

@njit(parallel=True, cache=True, fastmath=True, error_model="numpy")
def rys_nuc_symm_internal(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts, indx_startA, indx_endA, indx_startB, indx_endB, Z, coordsBohrs, natoms):
    # This function calculates the three-centered two electron integrals for density fitting
    # The basis object holds the information of basis functions like: exponents, coeffs, etc.
    
    # The integrals are performed using the formulas https://pubs.acs.org/doi/full/10.1021/acs.jchemed.8b00255
    # returns (AB|P) 
    
    # Infer the matrix shape from the start and end indices
    num_A = indx_endA - indx_startA 
    num_B = indx_endB - indx_startB 
    matrix_shape = (num_A, num_B)

    
    # Check if the slice of the matrix requested falls in the lower/upper triangle or in both the triangles
    upper_tri = False
    lower_tri = False
    both_tri_symm = False
    both_tri_nonsymm = False
    if indx_endA <= indx_startB:
        upper_tri = True
    elif indx_startA >= indx_endB:
        lower_tri = True
    elif indx_startA==indx_startB and indx_endA==indx_endB:
        both_tri_symm = True
    else:
        both_tri_nonsymm = True
    

    # Initialize the matrix with zeros
    Vnuc = np.zeros(matrix_shape, dtype=np.float64) 
        
    pi = 3.141592653589793
    pisq = 9.869604401089358  #PI^2
    twopisq = 19.739208802178716  #2*PI^2

    L = np.zeros((3))
    ld, md, nd = int(0), int(0), int(0)
    alphalk = 0.0

    zeta = 1e12
    zeta_2pi_32 = np.sqrt((2*zeta/(pi))**(3/2))
    # zeta_2pi_32 = (pi/zeta)**(3/2)
    # zeta_2pi_32 = np.sqrt((2*zeta/(pi))**(3/2))
    zeta_2pi_32 = (zeta/(pi))**(3/2)

    #Loop pver BFs
    for i in prange(indx_startA, indx_endA): #A
        I = bfs_coords[i]
        Ni = bfs_contr_prim_norms[i]
        lmni = bfs_lmn[i]
        la, ma, na = lmni
        nprimi = bfs_nprim[i]
        
        for j in prange(indx_startB, indx_endB): #B
            if lower_tri or upper_tri or (both_tri_symm and j<=i) or both_tri_nonsymm:
                J = bfs_coords[j]
                IJ = I - J
                IJsq = np.sum(IJ**2)
                Nj = bfs_contr_prim_norms[j]
                lmnj = bfs_lmn[j]
                lb, mb, nb = lmnj
                tempcoeff1 = Ni*Nj
                nprimj = bfs_nprim[j]
                
                val = 0.0
                for k in prange(natoms): #C # These would be our nuclei
                    K = coordsBohrs[k]
                    Nk = -Z[k]*zeta_2pi_32
                    lmnk = [0, 0, 0]
                    lc, mc, nc = lmnk
                    tempcoeff2 = tempcoeff1*Nk
                    nprimk = 1
                    
                    
                        
                    KL = K #- L  
                    KLsq = np.sum(KL**2)
                    
                    norder = int((la+ma+na+lb+mb+nb+lc+mc+nc+ld+md+nd)/2 + 1 ) 
                    # val = 0.0

                    if norder<=7: # Use rys quadrature
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
                                    dkk = 1.0
                                    Nkk = 1.0
                                    alphakk = zeta
                                    tempcoeff5 = tempcoeff4#*dkk*Nkk 
                                        
                                        
                                    gammaQ = alphakk #+ alphalk
                                    # screenfactorKL = np.exp(-alphakk*alphalk/gammaQ*KLsq)
                                    # if abs(screenfactorKL)<1.0e-8:   
                                    #     #TODO: Check for optimal value for screening
                                    #     continue
                                    # if abs(screenfactorAB*screenfactorKL)<1.0e-10:   
                                    #     #TODO: Check for optimal value for screening
                                    #     continue
                                    
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
                                    
                                for kk in range(1):
                                    dkk = 1.0
                                    Nkk = 1.0
                                    alphakk = zeta
                                    tempcoeff5 = tempcoeff4#*dkk*Nkk 
                                        
                                    
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

                Vnuc[i-indx_startA, j-indx_startB] = val
                    
                                    
    if both_tri_symm:
        #We save time by evaluating only the lower diagonal elements and then use symmetry Vi,j=Vj,i 
        for i in prange(indx_startA, indx_endA):
            for j in prange(indx_startB, indx_endB):
                if j>i:
                    Vnuc[i-indx_startA, j-indx_startB] = Vnuc[j-indx_startB, i-indx_startA]
                            
        
    return Vnuc