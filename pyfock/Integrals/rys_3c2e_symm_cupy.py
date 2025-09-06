try:
    import cupy as cp
    from cupy import fuse
except Exception as e:
    # Handle the case when Cupy is not installed
    cp = None
    # Define a dummy fuse decorator for CPU version
    def fuse(kernel_name):
        def decorator(func):
            return func 
        return decorator
from numba import cuda
import math
from numba import njit , prange
import numpy as np
import numba
from .rys_helpers_cuda import coulomb_rys, Roots, DATA_X, DATA_W
from .schwarz_helpers import eri_4c2e_diag
from .rys_2c2e_symm_cupy import rys_2c2e_symm_cupy


def rys_3c2e_symm_cupy(basis, auxbasis, slice=None, schwarz=False, schwarz_threshold=1e-9, cp_stream=None):
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
    bfs_coords = cp.array([basis.bfs_coords])
    bfs_contr_prim_norms = cp.array([basis.bfs_contr_prim_norms])
    bfs_lmn = cp.array([basis.bfs_lmn])
    bfs_nprim = cp.array([basis.bfs_nprim])

    aux_bfs_coords = cp.array([auxbasis.bfs_coords])
    aux_bfs_contr_prim_norms = cp.array([auxbasis.bfs_contr_prim_norms])
    aux_bfs_lmn = cp.array([auxbasis.bfs_lmn])
    aux_bfs_nprim = cp.array([auxbasis.bfs_nprim])

    #The remaining properties like bfs_coeffs are a list of lists of unequal sizes.
    #Numba won't be able to work with these efficiently.
    #So, we convert them to a numpy 2d array by applying a trick,
    #that the second dimension is that of the largest list. So that
    #it can accomadate all the lists.
    maxnprim = max(basis.bfs_nprim)
    bfs_coeffs = cp.zeros([basis.bfs_nao, maxnprim])
    bfs_expnts = cp.zeros([basis.bfs_nao, maxnprim])
    bfs_prim_norms = cp.zeros([basis.bfs_nao, maxnprim])
    for i in range(basis.bfs_nao):
        for j in range(basis.bfs_nprim[i]):
            bfs_coeffs[i,j] = basis.bfs_coeffs[i][j]
            bfs_expnts[i,j] = basis.bfs_expnts[i][j]
            bfs_prim_norms[i,j] = basis.bfs_prim_norms[i][j]
    maxnprimaux = max(auxbasis.bfs_nprim)
    aux_bfs_coeffs = cp.zeros([auxbasis.bfs_nao, maxnprimaux])
    aux_bfs_expnts = cp.zeros([auxbasis.bfs_nao, maxnprimaux])
    aux_bfs_prim_norms = cp.zeros([auxbasis.bfs_nao, maxnprimaux])
    for i in range(auxbasis.bfs_nao):
        for j in range(auxbasis.bfs_nprim[i]):
            aux_bfs_coeffs[i,j] = auxbasis.bfs_coeffs[i][j]
            aux_bfs_expnts[i,j] = auxbasis.bfs_expnts[i][j]
            aux_bfs_prim_norms[i,j] = auxbasis.bfs_prim_norms[i][j]
        

    DATA_X_cuda = cp.asarray(DATA_X)
    DATA_W_cuda = cp.asarray(DATA_W)

    if slice is None:
        slice = [0, basis.bfs_nao, 0, basis.bfs_nao, 0, auxbasis.bfs_nao]

    if schwarz:
        ints4c2e_diag = cp.asarray(eri_4c2e_diag(basis))
        ints2c2e = rys_2c2e_symm_cupy(auxbasis)
        sqrt_ints4c2e_diag = cp.sqrt(cp.abs(ints4c2e_diag))
        sqrt_diag_ints2c2e = cp.sqrt(cp.abs(cp.diag(ints2c2e)))
        print('Prelims calc done for Schwarz screening!')
    else:
        #Create dummy array
        sqrt_ints4c2e_diag = cp.zeros((1,1), dtype=cp.float64)
        sqrt_diag_ints2c2e = cp.zeros((1), dtype=cp.float64)
        
    #Limits for the calculation of 4c2e integrals
    indx_startA = int(slice[0])
    indx_endA = int(slice[1])
    indx_startB = int(slice[2])
    indx_endB = int(slice[3])
    indx_startC = int(slice[4])
    indx_endC = int(slice[5])
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
    threeC2E = cp.zeros(matrix_shape, dtype=cp.float64)

    if cp_stream is None:
        device = 0
        cp.cuda.Device(device).use()
        cp_stream = cp.cuda.Stream(non_blocking = True)
        nb_stream = cuda.external_stream(cp_stream.ptr)
        cp_stream.use()
    else:
        nb_stream = cuda.external_stream(cp_stream.ptr)
        cp_stream.use()

    thread_x = 32
    thread_y = 32
    blocks_per_grid = ((num_A + (thread_x - 1))//thread_x, (num_B + (thread_y - 1))//thread_y) 
    rys_3c2e_symm_internal_cuda_new[blocks_per_grid, (thread_x, thread_y), nb_stream](bfs_coords[0], bfs_contr_prim_norms[0], bfs_lmn[0], bfs_nprim[0], bfs_coeffs, bfs_prim_norms, bfs_expnts, aux_bfs_coords[0], aux_bfs_contr_prim_norms[0], aux_bfs_lmn[0], aux_bfs_nprim[0], aux_bfs_coeffs, aux_bfs_prim_norms, aux_bfs_expnts, indx_startA, indx_endA, indx_startB, indx_endB, indx_startC, indx_endC, tri_symm, no_symm, DATA_X_cuda, DATA_W_cuda, schwarz, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, schwarz_threshold, threeC2E)
    if tri_symm:
        symmetrize[blocks_per_grid, (thread_x, thread_y), nb_stream](indx_startA, indx_endA, indx_startB, indx_endB, indx_startC, indx_endC,threeC2E)
    if cp_stream is None:
        cuda.synchronize()
    else:
        cp_stream.synchronize()
        cp.cuda.Stream.null.synchronize()
        # cp._default_memory_pool.free_all_blocks()
    return threeC2E





@cuda.jit(fastmath=True, cache=True, max_registers=50)#(device=True)
def rys_3c2e_symm_internal_cuda(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts, aux_bfs_coords, aux_bfs_contr_prim_norms, aux_bfs_lmn, aux_bfs_nprim, aux_bfs_coeffs, aux_bfs_prim_norms, aux_bfs_expnts, indx_startA, indx_endA, indx_startB, indx_endB, indx_startC, indx_endC, tri_symm, no_symm, DATA_X, DATA_W, schwarz, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, schwarz_threshold, out):

    
    i, j = cuda.grid(2)
    pi = 3.141592653589793
    pisq = 9.869604401089358  #PI^2
    twopisq = 19.739208802178716  #2*PI^2

    L = cuda.local.array((3), numba.float64)
    L[0] = 0.0
    L[1] = 0.0
    L[2] = 0.0
    ld, md, nd = int(0), int(0), int(0)
    alphalk = 0.0

    if i>=indx_startA and i<indx_endA and j>=indx_startB and j<indx_endB:  
        if (tri_symm and j<=i) or no_symm:
            if schwarz:
                sqrt_ints4c2e_diag_ij = sqrt_ints4c2e_diag[i,j]
            IJ = cuda.local.array((3), numba.float64)
            P = cuda.local.array((3), numba.float64)
            PQ = cuda.local.array((3), numba.float64)
            I = bfs_coords[i]
            Ni = bfs_contr_prim_norms[i]
            lmni = bfs_lmn[i]
            la, ma, na = lmni
            nprimi = bfs_nprim[i]

            J = bfs_coords[j]
            IJ[0] = I[0] - J[0]
            IJ[1] = I[1] - J[1]
            IJ[2] = I[2] - J[2]
            IJsq = IJ[0]**2 + IJ[1]**2 + IJ[2]**2
            Nj = bfs_contr_prim_norms[j]
            lmnj = bfs_lmn[j]
            lb, mb, nb = lmnj
            tempcoeff1 = Ni*Nj
            nprimj = bfs_nprim[j]
            
            roots = cuda.local.array((10), numba.float64) # Good for upto i shells; i orbitals have an angular momentum of 6;
            weights = cuda.local.array((10), numba.float64) # Good for upto i shells; i orbitals have an angular momentum of 6;
            G = cuda.local.array((13, 13), numba.float64) # Good for upto i shells; i orbitals have an angular momentum of 6;

            for k in range(indx_startC, indx_endC): #C
                if schwarz:
                    if sqrt_ints4c2e_diag_ij*sqrt_diag_ints2c2e[k]<schwarz_threshold:
                        return
                K = aux_bfs_coords[k]
                Nk = aux_bfs_contr_prim_norms[k]
                lmnk = aux_bfs_lmn[k]
                lc, mc, nc = lmnk
                tempcoeff2 = tempcoeff1*Nk
                nprimk = aux_bfs_nprim[k]
                
                
                    
                KL = K #- L  
                # KLsq = KL[0]**2 + KL[1]**2 + KL[2]**2
                
                norder = int((la+ma+na+lb+mb+nb+lc+mc+nc+ld+md+nd)/2 + 1 ) 
                val = 0.0

                if norder<=10: # Use rys quadrature # Good for upto i orbitals
                    n = int(max(la+lb,ma+mb,na+nb))
                    m = int(max(lc+ld,mc+md,nc+nd))
                    
                    
                    #Loop over primitives
                    for ik in range(nprimi):   
                        dik = bfs_coeffs[i][ik]
                        Nik = bfs_prim_norms[i][ik]
                        alphaik = bfs_expnts[i][ik]
                        tempcoeff3 = tempcoeff2*dik*Nik
                            
                        for jk in range(nprimj):
                            alphajk = bfs_expnts[j][jk]
                            gammaP = alphaik + alphajk
                            screenfactorAB = math.exp(-alphaik*alphajk/gammaP*IJsq)
                            if abs(screenfactorAB)<1.0e-8:   
                                #TODO: Check for optimal value for screening
                                continue
                            djk = bfs_coeffs[j][jk] 
                            Njk = bfs_prim_norms[j][jk]      
                            P[0] = (alphaik*I[0] + alphajk*J[0])/gammaP
                            P[1] = (alphaik*I[1] + alphajk*J[1])/gammaP
                            P[2] = (alphaik*I[2] + alphajk*J[2])/gammaP
                            tempcoeff4 = tempcoeff3*djk*Njk  
                                
                            for kk in range(nprimk):
                                dkk = aux_bfs_coeffs[k][kk]
                                Nkk = aux_bfs_prim_norms[k][kk]
                                alphakk = aux_bfs_expnts[k][kk]
                                tempcoeff5 = tempcoeff4*dkk*Nkk 
                                    
                                    
                                gammaQ = alphakk #+ alphalk
                                # screenfactorKL = np.exp(-alphakk*alphalk/gammaQ*KLsq)
                                # if abs(screenfactorKL)<1.0e-8:   
                                #     #TODO: Check for optimal value for screening
                                #     continue
                                # if abs(screenfactorAB*screenfactorKL)<1.0e-10:   
                                #     #TODO: Check for optimal value for screening
                                #     continue
                                
                                Q = K        
                                PQ[0] = P[0] - Q[0]
                                PQ[1] = P[1] - Q[1]
                                PQ[2] = P[2] - Q[2]
                                PQsq = PQ[0]**2 + PQ[1]**2 + PQ[2]**2
                                rho = gammaP*gammaQ/(gammaP+gammaQ)     
                                        
                                X = PQsq*rho               
                                roots, weights = Roots(norder,X,DATA_X,DATA_W,roots,weights)
                                        
                                val += tempcoeff5*coulomb_rys(roots,weights,G,PQsq, rho, norder,n,m,la,lb,lc,ld,ma,mb,mc,md,na,nb,nc,nd,alphaik, alphajk, alphakk, alphalk,I,J,K,L)
                                        


                out[i-indx_startA, j-indx_startB, k-indx_startC] = val  


@cuda.jit(fastmath=True, cache=True, max_registers=50)#(device=True)
def rys_3c2e_symm_internal_cuda_new(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts, aux_bfs_coords, aux_bfs_contr_prim_norms, aux_bfs_lmn, aux_bfs_nprim, aux_bfs_coeffs, aux_bfs_prim_norms, aux_bfs_expnts, indx_startA, indx_endA, indx_startB, indx_endB, indx_startC, indx_endC, tri_symm, no_symm, DATA_X, DATA_W, schwarz, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, schwarz_threshold, out):

    
    i, j = cuda.grid(2)
    pi = 3.141592653589793
    pisq = 9.869604401089358  #PI^2
    twopisq = 19.739208802178716  #2*PI^2

    L = cuda.local.array((3), numba.float64)
    L[0] = 0.0
    L[1] = 0.0
    L[2] = 0.0
    ld, md, nd = int(0), int(0), int(0)
    alphalk = 0.0

    if i>=indx_startA and i<indx_endA and j>=indx_startB and j<indx_endB:  
        if (tri_symm and j<=i) or no_symm:
            if schwarz:
                sqrt_ints4c2e_diag_ij = sqrt_ints4c2e_diag[i,j]
            IJ = cuda.local.array((3), numba.float64)
            P = cuda.local.array((3), numba.float64)
            PQ = cuda.local.array((3), numba.float64)
            I = bfs_coords[i]
            Ni = bfs_contr_prim_norms[i]
            lmni = bfs_lmn[i]
            la, ma, na = lmni
            nprimi = bfs_nprim[i]

            J = bfs_coords[j]
            IJ[0] = I[0] - J[0]
            IJ[1] = I[1] - J[1]
            IJ[2] = I[2] - J[2]
            IJsq = IJ[0]**2 + IJ[1]**2 + IJ[2]**2
            Nj = bfs_contr_prim_norms[j]
            lmnj = bfs_lmn[j]
            lb, mb, nb = lmnj
            tempcoeff1 = Ni*Nj
            nprimj = bfs_nprim[j]
            
            roots = cuda.local.array((10), numba.float64) # Good for upto i shells; i orbitals have an angular momentum of 6;
            weights = cuda.local.array((10), numba.float64) # Good for upto i shells; i orbitals have an angular momentum of 6;
            G = cuda.local.array((13, 13), numba.float64) # Good for upto i shells; i orbitals have an angular momentum of 6;

            #Loop over primitives
            for ik in range(nprimi):   
                dik = bfs_coeffs[i][ik]
                Nik = bfs_prim_norms[i][ik]
                alphaik = bfs_expnts[i][ik]
                tempcoeff2 = tempcoeff1*dik*Nik
                    
                for jk in range(nprimj):
                    alphajk = bfs_expnts[j][jk]
                    gammaP = alphaik + alphajk
                    screenfactorAB = math.exp(-alphaik*alphajk/gammaP*IJsq)
                    if abs(screenfactorAB)<1.0e-8:   
                        #TODO: Check for optimal value for screening
                        continue
                    djk = bfs_coeffs[j][jk] 
                    Njk = bfs_prim_norms[j][jk]      
                    P[0] = (alphaik*I[0] + alphajk*J[0])/gammaP
                    P[1] = (alphaik*I[1] + alphajk*J[1])/gammaP
                    P[2] = (alphaik*I[2] + alphajk*J[2])/gammaP
                    tempcoeff3 = tempcoeff2*djk*Njk  

                    for k in range(indx_startC, indx_endC): #C
                        if schwarz:
                            if sqrt_ints4c2e_diag_ij*sqrt_diag_ints2c2e[k]<schwarz_threshold:
                                return
                        K = aux_bfs_coords[k]
                        Nk = aux_bfs_contr_prim_norms[k]
                        lmnk = aux_bfs_lmn[k]
                        lc, mc, nc = lmnk
                        tempcoeff4 = tempcoeff3*Nk
                        nprimk = aux_bfs_nprim[k]
                        
                        
                            
                        KL = K #- L  
                        # KLsq = KL[0]**2 + KL[1]**2 + KL[2]**2
                        
                        norder = int((la+ma+na+lb+mb+nb+lc+mc+nc+ld+md+nd)/2 + 1 ) 
                        val = 0.0

                        if norder<=10: # Use rys quadrature # Good for upto i orbitals
                            n = int(max(la+lb,ma+mb,na+nb))
                            m = int(max(lc+ld,mc+md,nc+nd))
                            
                            for kk in range(nprimk):
                                dkk = aux_bfs_coeffs[k][kk]
                                Nkk = aux_bfs_prim_norms[k][kk]
                                alphakk = aux_bfs_expnts[k][kk]
                                tempcoeff5 = tempcoeff4*dkk*Nkk 
                                    
                                    
                                gammaQ = alphakk #+ alphalk
                                # screenfactorKL = np.exp(-alphakk*alphalk/gammaQ*KLsq)
                                # if abs(screenfactorKL)<1.0e-8:   
                                #     #TODO: Check for optimal value for screening
                                #     continue
                                # if abs(screenfactorAB*screenfactorKL)<1.0e-10:   
                                #     #TODO: Check for optimal value for screening
                                #     continue
                                
                                Q = K        
                                PQ[0] = P[0] - Q[0]
                                PQ[1] = P[1] - Q[1]
                                PQ[2] = P[2] - Q[2]
                                PQsq = PQ[0]**2 + PQ[1]**2 + PQ[2]**2
                                rho = gammaP*gammaQ/(gammaP+gammaQ)     
                                        
                                X = PQsq*rho               
                                roots, weights = Roots(norder,X,DATA_X,DATA_W,roots,weights)
                                        
                                val += tempcoeff5*coulomb_rys(roots,weights,G,PQsq, rho, norder,n,m,la,lb,lc,ld,ma,mb,mc,md,na,nb,nc,nd,alphaik, alphajk, alphakk, alphalk,I,J,K,L)
                                            


                        out[i-indx_startA, j-indx_startB, k-indx_startC] += val   
                    
                

@cuda.jit(fastmath=True, cache=True)#(device=True)           
def symmetrize(indx_startA, indx_endA, indx_startB, indx_endB, indx_startC, indx_endC, out):
    #T + T.T - cp.diag(cp.diag(T))
    i, j = cuda.grid(2)
    if i>=indx_startA and i<indx_endA and j>=indx_startB and j<indx_endB:
        if j>i:
            for k in range(indx_startC, indx_endC):
                out[i-indx_startA, j-indx_startB, k] = out[j-indx_startB, i-indx_startA, k]