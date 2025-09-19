import numpy as np
from numba import cuda
import numba
import math
try:
    import cupy as cp
except Exception as e:
    # Handle the case when Cupy is not installed
    cp = None
    # Define a dummy fuse decorator for CPU version
    def fuse(kernel_name):
        def decorator(func):
            return func 
        return decorator
    pass
from .rys_helpers_cuda import coulomb_rys, coulomb_rys_3c2e, Roots, DATA_X, DATA_W

def eri_4c2e_diag_cupy(basis, cp_stream=None):
    # Used for Schwarz inequality test

    #We convert the required properties to numpy arrays as this is what Numba likes.
    bfs_coords = cp.array([basis.bfs_coords])
    bfs_contr_prim_norms = cp.array([basis.bfs_contr_prim_norms])
    bfs_lmn = cp.array([basis.bfs_lmn])
    bfs_nprim = cp.array([basis.bfs_nprim])
        

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

    DATA_X_cuda = cp.asarray(DATA_X)
    DATA_W_cuda = cp.asarray(DATA_W)

    # Initialize the matrix with zeros
    fourC2E_diag = cp.zeros((basis.bfs_nao, basis.bfs_nao), dtype=cp.float64)

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
    blocks_per_grid = ((basis.bfs_nao + (thread_x - 1))//thread_x, (basis.bfs_nao + (thread_y - 1))//thread_y) 
    rys_eri_4c2e_diag_internal_cuda[blocks_per_grid, (thread_x, thread_y), nb_stream](bfs_coords[0], bfs_contr_prim_norms[0], bfs_lmn[0], bfs_nprim[0], bfs_coeffs, bfs_prim_norms, bfs_expnts, DATA_X_cuda, DATA_W_cuda, fourC2E_diag)
    
    cp_stream.synchronize()
    cp.cuda.Stream.null.synchronize()
    # cp._default_memory_pool.free_all_blocks()
    return fourC2E_diag



@cuda.jit(fastmath=True, cache=True, max_registers=50)#(device=True)
def rys_eri_4c2e_diag_internal_cuda(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts, DATA_X, DATA_W, out):
    # This function calculates the "diagonal" elements of the 4c2e ERI array
    # Used to implement Schwarz screening
    # http://vergil.chemistry.gatech.edu/notes/df.pdf
    # returns a 2D array whose elements are given as A[i,j] = (ij|ij) 
    nao = bfs_coords.shape[0]
    
    pi = 3.141592653589793
    pisq = 9.869604401089358  #PI^2
    twopisq = 19.739208802178716  #2*PI^2

    i, j = cuda.grid(2)

    if i<nao and j<=i:
        IJ = cuda.local.array((3), numba.float64)
        P = cuda.local.array((3), numba.float64)
        PQ = cuda.local.array((3), numba.float64)
        I = bfs_coords[i]
        Ni = bfs_contr_prim_norms[i]
        lmni = bfs_lmn[i]
        la, ma, na = lmni
        nprimi = bfs_nprim[i]

        K = I
        lc, mc, nc = lmni
        
        
        J = bfs_coords[j]
        IJ[0] = I[0] - J[0]
        IJ[1] = I[1] - J[1]
        IJ[2] = I[2] - J[2]
        IJsq = IJ[0]**2 + IJ[1]**2 + IJ[2]**2
        L = J
        Nj = bfs_contr_prim_norms[j]
        lmnj = bfs_lmn[j]
        lb, mb, nb = lmnj
        tempcoeff3 = (Ni*Nj)**2
        nprimj = bfs_nprim[j]


        ld, md, nd = lmnj
        
        
        norder = int((la+ma+na+lb+mb+nb+lc+mc+nc+ld+md+nd)/2 + 1 ) 
        n = int(max(la+lb,ma+mb,na+nb))
        m = int(max(lc+ld,mc+md,nc+nd))
        # G = np.zeros((n+1,m+1))
        roots = cuda.local.array((10), numba.float64) # Good for upto g shells; g orbitals have an angular momentum of 5;
        weights = cuda.local.array((10), numba.float64) # Good for upto g shells; g orbitals have an angular momentum of 5;
        G = cuda.local.array((13, 13), numba.float64) # Good for upto g shells; g orbitals have an angular momentum of 5;
        val = 0.0
        
        #Loop over primitives
        for ik in range(nprimi):   
            dik = bfs_coeffs[i][ik]
            Nik = bfs_prim_norms[i][ik]
            alphaik = bfs_expnts[i][ik]
            tempcoeff4 = tempcoeff3*(dik*Nik)**2
            
            for jk in range(nprimj):
                alphajk = bfs_expnts[j][jk]
                gammaP = alphaik + alphajk
                djk = bfs_coeffs[j][jk] 
                Njk = bfs_prim_norms[j][jk]      
                P[0] = (alphaik*I[0] + alphajk*J[0])/gammaP
                P[1] = (alphaik*I[1] + alphajk*J[1])/gammaP
                P[2] = (alphaik*I[2] + alphajk*J[2])/gammaP
                
                    
                tempcoeff5 = tempcoeff4*(djk*Njk)**2


                gammaQ = gammaP
                dlk = djk
                Nlk = Njk     
                Q = P       
                PQ[0] = P[0] - Q[0]
                PQ[1] = P[1] - Q[1]
                PQ[2] = P[2] - Q[2]
                PQsq = PQ[0]**2 + PQ[1]**2 + PQ[2]**2
                
                tempcoeff6 = tempcoeff5*dlk*Nlk
                
                if norder<=10:
                    rho = gammaP*gammaQ/(gammaP+gammaQ)
                    
                                        
                    X = PQsq*rho               
                    roots, weights = Roots(norder,X,DATA_X,DATA_W,roots,weights)
                    val += tempcoeff6*coulomb_rys(roots,weights,G,PQsq, rho, norder,n,m,la,lb,lc,ld,ma,mb,mc,md,na,nb,nc,nd,alphaik, alphajk, alphaik, alphajk,I,J,K,L)
                        
        out[i,j] = val
        out[j,i] = val


def rys_3c2e_tri_schwarz_sparse_algo10_cupy(basis, auxbasis, indicesA, indicesB, offsets, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold, strict_schwarz, nsignificant, cp_stream=None):
    # Wrapper for hybrid Rys+conv. 3c2e integral calculator
    # using a list of significant contributions obtained via Schwarz screening.
    # It returns the 3c2e integrals in triangular form.

    # print('preprocessing starts', flush=True)

    #We convert the required properties to numpy arrays as this is what Numba likes.
    bfs_coords = cp.array([basis.bfs_coords])
    bfs_contr_prim_norms = cp.array([basis.bfs_contr_prim_norms])
    bfs_lmn =cp.array([basis.bfs_lmn])
    bfs_nprim = cp.array([basis.bfs_nprim])

    #We convert the required properties to numpy arrays as this is what Numba likes.
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
    shell_indices = cp.array([basis.bfs_shell_index], dtype=cp.uint16)[0]
    aux_shell_indices = cp.array([auxbasis.bfs_shell_index], dtype=cp.uint16)[0]
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

    sqrt_ints4c2e_diag = cp.asarray(sqrt_ints4c2e_diag) 
    sqrt_diag_ints2c2e = cp.asarray(sqrt_diag_ints2c2e) 
    offsets = cp.asarray(offsets) 
    

    # Initialize the matrix with zeros
    threeC2E = cp.zeros(int(nsignificant), dtype=cp.float64)

    if cp_stream is None:
        device = 0
        cp.cuda.Device(device).use()
        cp_stream = cp.cuda.Stream(non_blocking = True)
        nb_stream = cuda.external_stream(cp_stream.ptr)
        cp_stream.use()
    else:
        nb_stream = cuda.external_stream(cp_stream.ptr)
        cp_stream.use()

    thread_x = 8
    thread_y = 8
    blocks_per_grid = ((basis.bfs_nao + (thread_x - 1))//thread_x, (basis.bfs_nao + (thread_y - 1))//thread_y) 
    #rys_3c2e_tri_schwarz_sparse_algo10_internal_cuda[blocks_per_grid, (thread_x, thread_y), nb_stream](bfs_coords[0], bfs_contr_prim_norms[0], bfs_lmn[0], \
    #             bfs_nprim[0], bfs_coeffs, bfs_prim_norms, bfs_expnts,aux_bfs_coords[0], aux_bfs_contr_prim_norms[0], aux_bfs_lmn[0], aux_bfs_nprim[0], \
    #             aux_bfs_coeffs, aux_bfs_prim_norms, aux_bfs_expnts, 0, basis.bfs_nao, 0, basis.bfs_nao, 0, auxbasis.bfs_nao, DATA_X_cuda, \
    #             DATA_W_cuda, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold, offsets, strict_schwarz, threeC2E)
    rys_3c2e_tri_schwarz_sparse_algo10_internal_cuda_new[blocks_per_grid, (thread_x, thread_y), nb_stream](bfs_coords[0], bfs_contr_prim_norms[0], bfs_lmn[0], \
                bfs_nprim[0], bfs_coeffs, bfs_prim_norms, bfs_expnts,aux_bfs_coords[0], aux_bfs_contr_prim_norms[0], aux_bfs_lmn[0], aux_bfs_nprim[0], \
                aux_bfs_coeffs, aux_bfs_prim_norms, aux_bfs_expnts, 0, basis.bfs_nao, 0, basis.bfs_nao, 0, auxbasis.bfs_nao, DATA_X_cuda, \
                DATA_W_cuda, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold, offsets, strict_schwarz, aux_shell_indices, threeC2E)
    
    cp_stream.synchronize()
    cp.cuda.Stream.null.synchronize()
    cp._default_memory_pool.free_all_blocks()
    return threeC2E

@cuda.jit(fastmath=True, cache=True, max_registers=50)#(device=True)
def rys_3c2e_tri_schwarz_sparse_algo10_internal_cuda(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts, aux_bfs_coords, aux_bfs_contr_prim_norms, aux_bfs_lmn, aux_bfs_nprim, aux_bfs_coeffs, aux_bfs_prim_norms, aux_bfs_expnts, indx_startA, indx_endA, indx_startB, indx_endB, indx_startC, indx_endC, DATA_X, DATA_W, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, schwarz_threshold, offsets, strict_schwarz, out):
    
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

    if i>=indx_startA and i<indx_endA and j>=indx_startB and j<indx_endB and (j<=i):  
        sqrt_ints4c2e_diag_ij = sqrt_ints4c2e_diag[i,j]
        if strict_schwarz:
            if sqrt_ints4c2e_diag_ij*sqrt_ints4c2e_diag_ij<1e-13:
                return
        linear_index = j + i*(i+1)//2
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

        index_k = 0
        for k in range(indx_startC, indx_endC): #C
            if sqrt_ints4c2e_diag_ij*sqrt_diag_ints2c2e[k]<schwarz_threshold:
                continue
            K = aux_bfs_coords[k]
            Nk = aux_bfs_contr_prim_norms[k]
            lmnk = aux_bfs_lmn[k]
            lc, mc, nc = lmnk
            tempcoeff2 = tempcoeff1*Nk
            nprimk = aux_bfs_nprim[k]
            
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
                            
                            Q = K        
                            PQ[0] = P[0] - Q[0]
                            PQ[1] = P[1] - Q[1]
                            PQ[2] = P[2] - Q[2]
                            PQsq = PQ[0]**2 + PQ[1]**2 + PQ[2]**2
                            rho = gammaP*gammaQ/(gammaP+gammaQ)     
                                    
                            X = PQsq*rho               
                            roots, weights = Roots(norder,X,DATA_X,DATA_W,roots,weights)
                                    
                            # val += tempcoeff5*coulomb_rys(roots,weights,G,PQsq, rho, norder,n,m,la,lb,lc,ld,ma,mb,mc,md,na,nb,nc,nd,alphaik, alphajk, alphakk, alphalk,I,J,K,L)
                            val += tempcoeff5*coulomb_rys_3c2e(roots,weights,G,PQsq, rho, norder,n,m,la,lb,lc,ld,ma,mb,mc,md,na,nb,nc,nd,alphaik, alphajk,alphakk,alphalk,I,J,K,L,IJ,P)   

   
            out[offsets[linear_index]+index_k] = val
            index_k += 1

@cuda.jit(fastmath=True, cache=True, max_registers=800)#(device=True)
def rys_3c2e_tri_schwarz_sparse_algo10_internal_cuda_new(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts, aux_bfs_coords, aux_bfs_contr_prim_norms, aux_bfs_lmn, aux_bfs_nprim, aux_bfs_coeffs, aux_bfs_prim_norms, aux_bfs_expnts, indx_startA, indx_endA, indx_startB, indx_endB, indx_startC, indx_endC, DATA_X, DATA_W, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, schwarz_threshold, offsets, strict_schwarz, aux_shell_indices, out):
    
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

    if i>=indx_startA and i<indx_endA and j>=indx_startB and j<indx_endB and (j<=i):  
        sqrt_ints4c2e_diag_ij = sqrt_ints4c2e_diag[i,j]
        if strict_schwarz:
            if sqrt_ints4c2e_diag_ij*sqrt_ints4c2e_diag_ij<1e-13:
                return
        linear_index = j + i*(i+1)//2
        offset_ = offsets[linear_index]
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
        
        # roots = cuda.local.array((10), numba.float64) # Good for upto i shells; i orbitals have an angular momentum of 6;
        # weights = cuda.local.array((10), numba.float64) # Good for upto i shells; i orbitals have an angular momentum of 6;
        G = cuda.local.array((13, 13), numba.float64) # Good for upto i shells; i orbitals have an angular momentum of 6;
        roots = cuda.local.array((20,10), numba.float64) # Good for upto i shells; i orbitals have an angular momentum of 6;
        weights = cuda.local.array((20,10), numba.float64) # Good for upto i shells; i orbitals have an angular momentum of 6;

        #Loop over primitives
        for ik in range(nprimi):   
            dik = bfs_coeffs[i][ik]
            Nik = bfs_prim_norms[i][ik]
            alphaik = bfs_expnts[i][ik]
            tempcoeff2 = tempcoeff1*dik*Nik
                
            for jk in range(nprimj):
                alphajk = bfs_expnts[j][jk]
                gammaP = alphaik + alphajk
                prod_alphaikjk = alphaik*alphajk
                screenfactorAB = math.exp(-prod_alphaikjk/gammaP*IJsq)
                if abs(screenfactorAB)<1.0e-8:   
                    #TODO: Check for optimal value for screening
                    continue
                djk = bfs_coeffs[j][jk] 
                Njk = bfs_prim_norms[j][jk]      
                P[0] = (alphaik*I[0] + alphajk*J[0])/gammaP
                P[1] = (alphaik*I[1] + alphajk*J[1])/gammaP
                P[2] = (alphaik*I[2] + alphajk*J[2])/gammaP
                tempcoeff3 = tempcoeff2*djk*Njk 

                index_k = 0
                shell_index_previous = -1
                for k in range(indx_startC, indx_endC): #C
                    if sqrt_ints4c2e_diag_ij*sqrt_diag_ints2c2e[k]<schwarz_threshold:
                        continue
                    shell_index = aux_shell_indices[k]

                    K = aux_bfs_coords[k]
                    Nk = aux_bfs_contr_prim_norms[k]
                    lmnk = aux_bfs_lmn[k]
                    lc, mc, nc = lmnk
                    tempcoeff4 = tempcoeff3*Nk
                    nprimk = aux_bfs_nprim[k]

                    Q = K        
                    PQ[0] = P[0] - Q[0]
                    PQ[1] = P[1] - Q[1]
                    PQ[2] = P[2] - Q[2]
                    PQsq = PQ[0]**2 + PQ[1]**2 + PQ[2]**2
                    
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
                            ABsrt = math.sqrt(gammaP*alphakk)
                                
                                
                            gammaQ = alphakk #+ alphalk
                            
                            
                            rho = gammaP*gammaQ/(gammaP+gammaQ)     
                                    
                            X = PQsq*rho  
                            roots_kk = roots[kk,:]
                            weights_kk = weights[kk,:]  
                            if shell_index != shell_index_previous:           
                                roots_kk, weights_kk = Roots(norder,X,DATA_X,DATA_W,roots_kk,weights_kk)
                            # else:
                            #     print(shell_index, shell_index_previous)
                                    
                            # val += tempcoeff5*coulomb_rys(roots,weights,G,PQsq, rho, norder,n,m,la,lb,lc,ld,ma,mb,mc,md,na,nb,nc,nd,alphaik, alphajk, alphakk, alphalk,I,J,K,L)
                            val += tempcoeff5*coulomb_rys_3c2e(roots_kk,weights_kk,G,PQsq, rho, norder,n,m,la,lb,lc,ld,ma,mb,mc,md,na,nb,nc,nd,alphaik, alphajk,alphakk,alphalk,I,J,K,L,IJ,P, prod_alphaikjk, gammaP,ABsrt)   

                    shell_index_previous = shell_index
                    out[offset_+index_k] += val
                    index_k += 1

def J_tri_calculator_algo10_cupy(ints3c2e_1d, df_coeff, size_J_tri, nao, offsets_3c2e, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold, naux, strict_schwarz, auxbfs_lm, cp_stream=None):
    # This can also be simply calculated using:
    # J = contract('ijk,k', ints3c2e, df_coeff) # For general 3d and 2d arrays
    # or
    # J_tri = contract('pP,P', ints3c2e, df_coeff) # For triangular versions of the above arrays
    # However, we need to create a custom function to
    # do this with a sparse 1d array holding the values of significant
    # elements of ints3c2e array.
    # print(indicesA)
    
    if cp_stream is None:
        device = 0
        cp.cuda.Device(device).use()
        cp_stream = cp.cuda.Stream(non_blocking = True)
        nb_stream = cuda.external_stream(cp_stream.ptr)
        # cp_stream.use()
    else:
        nb_stream = cuda.external_stream(cp_stream.ptr)
        cp_stream.use()

    J_tri = cp.zeros(size_J_tri, dtype=cp.float64)
    
    cp_stream.synchronize()
    cp.cuda.Stream.null.synchronize()

    thread_x = 32
    thread_y = 32
    blocks_per_grid = ((nao + (thread_x - 1))//thread_x, (nao + (thread_y - 1))//thread_y) 
    J_tri_calculator_algo10_internal_cuda[blocks_per_grid, (thread_x, thread_y), nb_stream](ints3c2e_1d, df_coeff, nao, offsets_3c2e, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold, naux, strict_schwarz, J_tri)
    
    
    cp_stream.synchronize()
    cp.cuda.Stream.null.synchronize()
    # cp._default_memory_pool.free_all_blocks()
    return J_tri

@cuda.jit(fastmath=True, cache=True)#(device=True)
def J_tri_calculator_algo10_internal_cuda(ints3c2e_1d, df_coeff, nao, offsets_3c2e, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold, naux, strict_schwarz, J_tri):
    # This can also be simply calculated using:
    # J = contract('ijk,k', ints3c2e, df_coeff) # For general 3d and 2d arrays
    # or
    # J_tri = contract('pP,P', ints3c2e, df_coeff) # For triangular versions of the above arrays
    # However, we need to create a custom function to
    # do this with a sparse 1d array holding the values of significant
    # elements of ints3c2e array.
    # print(indicesA)

    i, j = cuda.grid(2)
    if i>=0 and i<nao and j>=0 and j<nao and (j<=i):
        offset = int(i*(i+1)/2) # Offset for the J_tri array
        sqrt_ij = sqrt_ints4c2e_diag[i,j]
        if strict_schwarz:
            if sqrt_ij*sqrt_ij<1e-13:
                return  
        index_k = 0
        val = 0.0
        linear_index = j + i*(i+1)//2
        offset_3c2e = offsets_3c2e[linear_index] # Offset for the 3c2e array
        for k in range(naux):
            # if strict_schwarz:
            #     max_val = sqrt_ij*sqrt_diag_ints2c2e[k]
            #     if max_val>threshold:
            #         if max_val<1e-8:
            #             if (auxbfs_lm[k])>=1: # s aux functions
            #                 continue
            #         elif max_val<1e-7:
            #             if (auxbfs_lm[k])>=2: # s, p aux functions
            #                 continue
            #         elif max_val<1e-6:
            #             if (auxbfs_lm[k])>=3: # s, p, d aux functions
            #                 continue
            #         J_tri[j+offset] += ints3c2e_1d[offset_3c2e+index_k]*df_coeff[k] 
            #         index_k += 1
            #     else:
            #         continue
            # else:
            #     if sqrt_ij*sqrt_diag_ints2c2e[k]>threshold:
            #         J_tri[j+offset] += ints3c2e_1d[offset_3c2e+index_k]*df_coeff[k] 
            #         index_k += 1
            if sqrt_ij*sqrt_diag_ints2c2e[k]>threshold:
                # J_tri[j+offset] += ints3c2e_1d[offset_3c2e+index_k]*df_coeff[k] 
                val += ints3c2e_1d[offset_3c2e+index_k]*df_coeff[k] 
                index_k += 1
        J_tri[j+offset] = val

def df_coeff_calculator_algo10_cupy(ints3c2e_1d, dmat_1d, nao, offsets_3c2e, naux, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold, strict_schwarz, cp_stream=None):
    if cp_stream is None:
        device = 0
        cp.cuda.Device(device).use()
        cp_stream = cp.cuda.Stream(non_blocking = True)
        nb_stream = cuda.external_stream(cp_stream.ptr)
        # cp_stream.use()
    else:
        nb_stream = cuda.external_stream(cp_stream.ptr)
        cp_stream.use()

    
    df_coeff = cp.zeros(naux, dtype=cp.float64)

    thread_x = 32
    thread_y = 32
    blocks_per_grid = ((nao + (thread_x - 1))//thread_x, (nao + (thread_y - 1))//thread_y) 
    size_dmat_1d = dmat_1d.shape[0]
    df_coeff_calculator_algo10_cuda_internal[blocks_per_grid, (thread_x, thread_y), nb_stream](ints3c2e_1d, dmat_1d, nao, size_dmat_1d, offsets_3c2e, naux, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold, strict_schwarz, df_coeff)
    
    cp_stream.synchronize()
    cp.cuda.Stream.null.synchronize()
    # cp._default_memory_pool.free_all_blocks()
    return df_coeff

@cuda.jit(fastmath=True, cache=True)#(device=True)
def df_coeff_calculator_algo10_cuda_internal(ints3c2e_1d, dmat_1d, nao, size_dmat_1d, offsets_3c2e, naux, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold, strict_schwarz, df_coeff):
    # This function calculates the coefficients of the auxiliary basis for
    # density fitting. 
    # This can also be simply calculated using:
    # df_coeff = contract('pqP,pq->P', ints3c2e, dmat) # For general 3d and 2d arrays
    # or
    # df_coeff = contract('pP,p->P', ints3c2e, dmat_tri) # For triangular versions of the above arrays
    # However, we need to create a custom function to
    # do this with a sparse 1d array holding the values of significant
    # elements of ints3c2e array.
    # nelements = ints3c2e_1d.shape[0]
    i, j = cuda.grid(2)
    if i>=0 and i<nao and j>=0 and j<=i:
        offset = int(j + i*(i+1)/2)
        if offset>=size_dmat_1d:
            return
        sqrt_ij = sqrt_ints4c2e_diag[i,j]
        if strict_schwarz:
            if sqrt_ij*sqrt_ij<1e-13:
                return  
        index_k = 0
        # linear_index = j + i*(i+1)//2
        offset_3c2e = offsets_3c2e[offset]
        dmat_val = dmat_1d[offset]
        for k in range(0, naux):
            if sqrt_ij*sqrt_diag_ints2c2e[k]>threshold:
                # df_coeff[k] += ints3c2e_1d[offset_3c2e+index_k]*dmat_val  # This leads to race condition
                temp = ints3c2e_1d[offset_3c2e+index_k]*dmat_val
                cuda.atomic.add(df_coeff, k, temp)  # Arguments are array, array index, value to add
                index_k += 1 

