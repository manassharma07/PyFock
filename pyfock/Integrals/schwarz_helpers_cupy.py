"""
Legacy GPU Schwarz-screened 3c2e kernels kept for reference only.

The active GPU implementation of the default density-fitting algorithm
(DF_algo=10) lives in :mod:`pyfock.Integrals.df_algo10_helpers_cupy`.
Nothing in PyFock calls the kernels below.
"""
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
from .rys_helpers_cuda import coulomb_rys, coulomb_rys_3c2e, Roots, Roots_5, DATA_X, DATA_W

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

# Define global numpy arrays so Numba can load them into __constant__ memory
a = 10.0/19.0
b = 14.0/19.0
c = 5.0/19.0
d = 3.0*math.sqrt(5.0)/19.0

M_f_arr = np.array([
    [ a,  0,  0, -d,  0, -d,  0,  0,  0,  0],
    [ 0,  b,  0,  0,  0,  0, -d,  0, -c,  0],
    [ 0,  0,  b,  0,  0,  0,  0, -c,  0, -d],
    [-d,  0,  0,  b,  0, -c,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  1,  0,  0,  0,  0,  0],
    [-d,  0,  0, -c,  0,  b,  0,  0,  0,  0],
    [ 0, -d,  0,  0,  0,  0,  a,  0, -d,  0],
    [ 0,  0, -c,  0,  0,  0,  0,  b,  0, -d],
    [ 0, -c,  0,  0,  0,  0, -d,  0,  b,  0],
    [ 0,  0, -d,  0,  0,  0,  0, -d,  0,  a]
], dtype=np.float64)

M_g_arr = np.array([
    [ 0.3513422061809158,  0.0,  0.0, -0.3085873961776689,  0.0, -0.3085873961776688,  0.0,  0.0,  0.0,  0.0,  0.1065869614256711,  0.0,  0.1213545940024541,  0.0,  0.1065869614256711],
    [ 0.0,  0.6086956521739131,  0.0,  0.0,  0.0,  0.0, -0.3913043478260871,  0.0, -0.2916610405434507,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
    [ 0.0,  0.0,  0.6086956521739130,  0.0,  0.0,  0.0,  0.0, -0.2916610405434508,  0.0, -0.3913043478260869,  0.0,  0.0,  0.0,  0.0,  0.0],
    [-0.3085873961776690,  0.0,  0.0,  0.6486577938190843,  0.0, -0.1065869614256711,  0.0,  0.0,  0.0,  0.0, -0.3085873961776689,  0.0, -0.1065869614256713,  0.0,  0.1213545940024542],
    [ 0.0,  0.0,  0.0,  0.0,  0.7826086956521738,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.2916610405434508,  0.0, -0.2916610405434507,  0.0],
    [-0.3085873961776688,  0.0,  0.0, -0.1065869614256711,  0.0,  0.6486577938190838,  0.0,  0.0,  0.0,  0.0,  0.1213545940024540,  0.0, -0.1065869614256710,  0.0, -0.3085873961776688],
    [ 0.0, -0.3913043478260871,  0.0,  0.0,  0.0,  0.0,  0.6086956521739131,  0.0, -0.2916610405434508,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
    [ 0.0,  0.0, -0.2916610405434507,  0.0,  0.0,  0.0,  0.0,  0.7826086956521741,  0.0, -0.2916610405434510,  0.0,  0.0,  0.0,  0.0,  0.0],
    [ 0.0, -0.2916610405434509,  0.0,  0.0,  0.0,  0.0, -0.2916610405434509,  0.0,  0.7826086956521741,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
    [ 0.0,  0.0, -0.3913043478260869,  0.0,  0.0,  0.0,  0.0, -0.2916610405434509,  0.0,  0.6086956521739130,  0.0,  0.0,  0.0,  0.0,  0.0],
    [ 0.1065869614256711,  0.0,  0.0, -0.3085873961776689,  0.0,  0.1213545940024540,  0.0,  0.0,  0.0,  0.0,  0.3513422061809157,  0.0, -0.3085873961776687,  0.0,  0.1065869614256710],
    [ 0.0,  0.0,  0.0,  0.0, -0.2916610405434508,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.6086956521739130,  0.0, -0.3913043478260869,  0.0],
    [ 0.1213545940024541,  0.0,  0.0, -0.1065869614256714,  0.0, -0.1065869614256709,  0.0,  0.0,  0.0,  0.0, -0.3085873961776687,  0.0,  0.6486577938190840,  0.0, -0.3085873961776689],
    [ 0.0,  0.0,  0.0,  0.0, -0.2916610405434508,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.3913043478260868,  0.0,  0.6086956521739129,  0.0],
    [ 0.1065869614256710,  0.0,  0.0,  0.1213545940024542,  0.0, -0.3085873961776688,  0.0,  0.0,  0.0,  0.0,  0.1065869614256710,  0.0, -0.3085873961776688,  0.0,  0.3513422061809157]
], dtype=np.float64)

@cuda.jit(fastmath=True, cache=True, max_registers=128)
def rys_3c2e_tri_schwarz_sparse_algo10_sao_internal_cuda_new(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts, aux_bfs_coords, aux_bfs_contr_prim_norms, aux_bfs_lmn, aux_bfs_nprim, aux_bfs_coeffs, aux_bfs_prim_norms, aux_bfs_expnts, indx_startA, indx_endA, indx_startB, indx_endB, indx_startC, indx_endC, DATA_X, DATA_W, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, schwarz_threshold, offsets, strict_schwarz, aux_shell_indices, out):
    
    i, j = cuda.grid(2)
    pi = 3.141592653589793
    pisq = 9.869604401089358
    twopisq = 19.739208802178716

    # Map transformation matrices to fast read-only constant memory
    M_f = cuda.const.array_like(M_f_arr)
    M_g = cuda.const.array_like(M_g_arr)

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
        
        # norder = (2L + Laux)/2 + 1 = L + Laux/2 + 1
        # G = cuda.local.array((13, 13), numba.float64)
        # roots = cuda.local.array((20,10), numba.float64) # Good for upto i shells and i auxshells; i orbitals have an angular momentum of 6;
        # weights = cuda.local.array((20,10), numba.float64) # Good for upto i shells and i auxshells; i orbitals have an angular momentum of 6;
        G = cuda.local.array((5, 5), numba.float64)  # Good for upto g auxshells and d shells;
        roots = cuda.local.array((7,5), numba.float64) # Good for upto g auxshells and d shells; and 7 primitives per bf
        weights = cuda.local.array((7,5), numba.float64) # Good for upto g auxshells and d shells; and 7 primitives per bf
        

        # Local SAO Buffers
        d_buffer = cuda.local.array((6), numba.float64)
        f_buffer = cuda.local.array((10), numba.float64)
        g_buffer = cuda.local.array((15), numba.float64)

        # Loop over primitives
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
                    continue
                djk = bfs_coeffs[j][jk] 
                Njk = bfs_prim_norms[j][jk]      
                P[0] = (alphaik*I[0] + alphajk*J[0])/gammaP
                P[1] = (alphaik*I[1] + alphajk*J[1])/gammaP
                P[2] = (alphaik*I[2] + alphajk*J[2])/gammaP
                tempcoeff3 = tempcoeff2*djk*Njk 

                roots_shell_previous = -1
                index_k = 0
                k = indx_startC
                
                d_idx = 0; f_idx = 0; g_idx = 0
                
                # Switch to a while loop to support skipping entire shell blocks natively
                while k < indx_endC:
                    shell_index = aux_shell_indices[k]
                    lmnk = aux_bfs_lmn[k]
                    lc, mc, nc = lmnk
                    tot_ang = lc + mc + nc
                    
                    is_new_shell = (shell_index != roots_shell_previous)
                    
                    if is_new_shell:
                        if sqrt_ints4c2e_diag_ij*sqrt_diag_ints2c2e[k] < schwarz_threshold:
                            if tot_ang == 0: k += 1
                            elif tot_ang == 1: k += 3
                            elif tot_ang == 2: k += 6
                            elif tot_ang == 3: k += 10
                            elif tot_ang == 4: k += 15
                            continue
                            
                        # Reset tracking pointers when processing a valid new shell
                        if tot_ang == 2: d_idx = 0
                        elif tot_ang == 3: f_idx = 0
                        elif tot_ang == 4: g_idx = 0

                    K = aux_bfs_coords[k]
                    Nk = aux_bfs_contr_prim_norms[k]
                    tempcoeff4 = tempcoeff3*Nk
                    nprimk = aux_bfs_nprim[k]

                    Q = K        
                    PQ[0] = P[0] - Q[0]
                    PQ[1] = P[1] - Q[1]
                    PQ[2] = P[2] - Q[2]
                    PQsq = PQ[0]**2 + PQ[1]**2 + PQ[2]**2
                    
                    norder = int((la+ma+na+lb+mb+nb+lc+mc+nc+ld+md+nd)/2 + 1) 
                    val = 0.0

                    if norder <= 10:
                        n = int(max(la+lb,ma+mb,na+nb))
                        m = int(max(lc+ld,mc+md,nc+nd))
                        
                        for kk in range(nprimk):
                            dkk = aux_bfs_coeffs[k][kk]
                            Nkk = aux_bfs_prim_norms[k][kk]
                            alphakk = aux_bfs_expnts[k][kk]
                            tempcoeff5 = tempcoeff4*dkk*Nkk 
                            ABsrt = math.sqrt(gammaP*alphakk)
                            
                            gammaQ = alphakk
                            rho = gammaP*gammaQ/(gammaP+gammaQ)       
                            X = PQsq*rho  
                            roots_kk = roots[kk,:]
                            weights_kk = weights[kk,:]  
                            
                            # Only recalculate roots dynamically if transitioning between distinct shells
                            if is_new_shell:            
                                roots_kk, weights_kk = Roots(norder,X,DATA_X,DATA_W,roots_kk,weights_kk)
                            
                            val += tempcoeff5*coulomb_rys_3c2e(roots_kk,weights_kk,G,PQsq, rho, norder,n,m,la,lb,lc,ld,ma,mb,mc,md,na,nb,nc,nd,alphaik, alphajk,alphakk,alphalk,I,J,K,L,IJ,P, prod_alphaikjk, gammaP,ABsrt)   

                    roots_shell_previous = shell_index

                    # --- SAO Transformation Block ---
                    if tot_ang == 0 or tot_ang == 1:
                        out[offset_ + index_k] += val
                        index_k += 1
                    
                    elif tot_ang == 2:
                        d_buffer[d_idx] = val
                        d_idx += 1
                        if d_idx == 6:
                            out[offset_ + index_k]     += 2.0/3.0 * d_buffer[0] - 1.0/3.0 * (d_buffer[3] + d_buffer[5])
                            out[offset_ + index_k + 1] += d_buffer[1]
                            out[offset_ + index_k + 2] += d_buffer[2]
                            out[offset_ + index_k + 3] += 2.0/3.0 * d_buffer[3] - 1.0/3.0 * (d_buffer[0] + d_buffer[5])
                            out[offset_ + index_k + 4] += d_buffer[4]
                            out[offset_ + index_k + 5] += 2.0/3.0 * d_buffer[5] - 1.0/3.0 * (d_buffer[0] + d_buffer[3])
                            index_k += 6
                            
                    elif tot_ang == 3:
                        f_buffer[f_idx] = val
                        f_idx += 1
                        if f_idx == 10:
                            for r in range(10):
                                temp_val = 0.0
                                for c in range(10):
                                    temp_val += M_f[r, c] * f_buffer[c]
                                out[offset_ + index_k + r] += temp_val
                            index_k += 10
                            
                    elif tot_ang == 4:
                        g_buffer[g_idx] = val
                        g_idx += 1
                        if g_idx == 15:
                            for r in range(15):
                                temp_val = 0.0
                                for c in range(15):
                                    temp_val += M_g[r, c] * g_buffer[c]
                                out[offset_ + index_k + r] += temp_val
                            index_k += 15

                    k += 1
