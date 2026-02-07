import numpy as np
from numba import njit, prange, guvectorize, float64, int16, int32, config, threading_layer, get_thread_id, get_num_threads, get_parallel_chunksize
from numba import cuda

try:
    import cupy as cp
except Exception as e:
    pass

from .integral_helpers import innerLoop4c2e
from .rys_helpers import coulomb_rys, coulomb_rys_fast, coulomb_rys_new, coulomb_rys_3c2e, coulomb_rys_3c2e_new
from .integral_helpers import Fboys

from joblib import Parallel, delayed

def eri_4c2e_diag(basis):
    # Used for Schwarz inequality test

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
        



    # ints4c2e_diag = eri_4c2e_diag_internal(bfs_coords[0], bfs_contr_prim_norms[0], bfs_lmn[0], bfs_nprim[0], bfs_coeffs, bfs_prim_norms, bfs_expnts)
    ints4c2e_diag = rys_eri_4c2e_diag_internal(bfs_coords[0], bfs_contr_prim_norms[0], bfs_lmn[0], bfs_nprim[0], bfs_coeffs, bfs_prim_norms, bfs_expnts)

    return ints4c2e_diag

# Reference: https://github.com/numba/numba/issues/8007#issuecomment-1113187684
parallel_options = {
    'comprehension': False,  # parallel comprehension
    'prange':        True,  # parallel for-loop
    'numpy':         True,  # parallel numpy calls
    'reduction':     False,  # parallel reduce calls
    'setitem':       True,  # parallel setitem
    'stencil':       True,  # parallel stencils
    'fusion':        True,  # enable fusion or not
}
@njit(parallel=parallel_options, cache=True, fastmath=True, error_model="numpy")
def eri_4c2e_diag_internal(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts):
    # This function calculates the "diagonal" elements of the 4c2e ERI array
    # Used to implement Schwarz screening
    # http://vergil.chemistry.gatech.edu/notes/df.pdf
    # returns a 2D array whose elements are given as A[i,j] = (ij|ij) 
    nao = bfs_coords.shape[0]
    fourC2E_diag = np.zeros((nao, nao),dtype=np.float64) 
        
    pi = 3.141592653589793
    pisq = 9.869604401089358  #PI^2
    twopisq = 19.739208802178716  #2*PI^2

    #Loop pver BFs
    for i in prange(0, nao): #A
        I = bfs_coords[i]
        Ni = bfs_contr_prim_norms[i]
        lmni = bfs_lmn[i]
        la, ma, na = lmni
        nprimi = bfs_nprim[i]

        K = I
        lc, mc, nc = lmni
        
        for j in prange(0, i + 1): #B
            J = bfs_coords[j]
            IJ = I - J
            IJsq = np.sum(IJ**2)
            Nj = bfs_contr_prim_norms[j]
            lmnj = bfs_lmn[j]
            lb, mb, nb = lmnj
            tempcoeff3 = (Ni*Nj)**2
            nprimj = bfs_nprim[j]


            ld, md, nd = lmnj
            
            

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
                    screenfactorAB = np.exp(-alphaik*alphajk/gammaP*IJsq)
                    djk = bfs_coeffs[j][jk] 
                    Njk = bfs_prim_norms[j][jk]      
                    P = (alphaik*I + alphajk*J)/gammaP
                    PI = P - I
                    PJ = P - J  
                    fac1 = twopisq/gammaP*screenfactorAB   
                    onefourthgammaPinv = 0.25/gammaP  
                    tempcoeff5 = tempcoeff4*(djk*Njk)**2


                    gammaQ = gammaP
                    screenfactorKL = screenfactorAB
                    dlk = djk
                    Nlk = Njk     
                    Q = P       
                    PQ = P - Q
                    
                    QK = PI
                    QL = PJ
                    tempcoeff6 = tempcoeff5*dlk*Nlk
                    
                  
                              
                    fac2 = fac1/gammaQ
                    

                    omega = (fac2)*np.sqrt(pi/(gammaP + gammaQ))*screenfactorKL
                    delta = 0.25*(1/gammaQ) + onefourthgammaPinv          
                    PQsqBy4delta = np.sum(PQ**2)/(4*delta)         
                        
                    sum1 = innerLoop4c2e(la,lb,lc,ld,ma,mb,mc,md,na,nb,nc,nd,gammaP,gammaQ,PI,PJ,QK,QL,PQ,PQsqBy4delta,delta)
                        
                    val += omega*sum1*tempcoeff6
                            
            fourC2E_diag[i,j] = val
            fourC2E_diag[j,i] = val
        
    return fourC2E_diag

@njit(parallel=parallel_options, cache=True, fastmath=True, error_model="numpy")
def rys_eri_4c2e_diag_internal(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts):
    # This function calculates the "diagonal" elements of the 4c2e ERI array
    # Used to implement Schwarz screening
    # http://vergil.chemistry.gatech.edu/notes/df.pdf
    # returns a 2D array whose elements are given as A[i,j] = (ij|ij) 
    nao = bfs_coords.shape[0]
    fourC2E_diag = np.zeros((nao, nao),dtype=np.float64) 
        
    pi = 3.141592653589793
    pisq = 9.869604401089358  #PI^2
    twopisq = 19.739208802178716  #2*PI^2

    #Loop pver BFs
    for i in prange(0, nao): #A
        I = bfs_coords[i]
        Ni = bfs_contr_prim_norms[i]
        lmni = bfs_lmn[i]
        la, ma, na = lmni
        nprimi = bfs_nprim[i]

        K = I
        lc, mc, nc = lmni
        
        for j in prange(0, i + 1): #B
            J = bfs_coords[j]
            IJ = I - J
            L = J
            IJsq = np.sum(IJ**2)
            Nj = bfs_contr_prim_norms[j]
            lmnj = bfs_lmn[j]
            lb, mb, nb = lmnj
            tempcoeff3 = (Ni*Nj)**2
            nprimj = bfs_nprim[j]


            ld, md, nd = lmnj
            
            
            norder = int((la+ma+na+lb+mb+nb+lc+mc+nc+ld+md+nd)/2 + 1 ) 
            roots = np.zeros((10))
            weights = np.zeros((10))
            n = int(max(la+lb,ma+mb,na+nb))
            m = int(max(lc+ld,mc+md,nc+nd))
            G = np.zeros((n+1,m+1))
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
                    screenfactorAB = np.exp(-alphaik*alphajk/gammaP*IJsq)
                    djk = bfs_coeffs[j][jk] 
                    Njk = bfs_prim_norms[j][jk]      
                    P = (alphaik*I + alphajk*J)/gammaP
                    PI = P - I
                    PJ = P - J  
                    
                      
                    tempcoeff5 = tempcoeff4*(djk*Njk)**2


                    gammaQ = gammaP
                    screenfactorKL = screenfactorAB
                    dlk = djk
                    Nlk = Njk     
                    Q = P       
                    PQ = P - Q
                    PQsq = np.sum(PQ**2)
                    
                    QK = PI
                    QL = PJ
                    tempcoeff6 = tempcoeff5*dlk*Nlk
                    
                  
                              
                    

                    # (ss|ss) case
                    if (la+ma+na+lb+mb+nb+lc+mc+nc+ld+md+nd)==0: 
                        val += tempcoeff6*twopisq/(gammaP*gammaQ)*np.sqrt(pi/(gammaP+gammaQ))\
                            *screenfactorAB*screenfactorKL*Fboys(0,PQsq/(1/gammaP+1/gammaQ))
                    elif norder<=10:
                        rho = gammaP*gammaQ/(gammaP+gammaQ)
                        val += tempcoeff6*coulomb_rys(roots,weights,G,PQsq, rho, norder,n,m,la,lb,lc,ld,ma,mb,mc,md,na,nb,nc,nd,alphaik, alphajk, alphaik, alphajk,I,J,K,L)
                    else:                                
                        onefourthgammaPinv = 0.25/gammaP
                        fac1 = twopisq/gammaP*screenfactorAB   
                        fac2 = fac1/gammaQ
                        omega = (fac2)*np.sqrt(pi/(gammaP + gammaQ))*screenfactorKL
                        delta = 0.25*(1/gammaQ) + onefourthgammaPinv          
                        PQsqBy4delta = np.sum(PQ**2)/(4*delta)         
                            
                        sum1 = innerLoop4c2e(la,lb,lc,ld,ma,mb,mc,md,na,nb,nc,nd,gammaP,gammaQ,PI,PJ,QK,QL,PQ,PQsqBy4delta,delta)
                            
                        val += omega*sum1*tempcoeff6
                            
            fourC2E_diag[i,j] = val
            fourC2E_diag[j,i] = val
        
    return fourC2E_diag

@njit(parallel=True, cache=True, fastmath=True, error_model="numpy")
def calc_indices_3c2e_schwarz(eri_4c2e_diag, ints2c2e, nao, naux, threshold):
    # This function will return a numpy array of the same size as ints3c2e array (nao*nao*naux)
    # The array will have a value of 1 where there is a significant contribution and 0 otherwise.
    # Later on this array can be used to find the indices of non-zero arrays
    indices = np.zeros((nao, nao, naux), dtype=np.uint8)
    # Loop over the lower-triangular ints3c2e array
    for i in range(nao):
        for j in range(i+1):
            for k in prange(naux):
                if np.sqrt(eri_4c2e_diag[i,j])*np.sqrt(ints2c2e[k,k])>threshold:
                    indices[i,j,k] = 1
    return indices

@njit(parallel=True, cache=True, fastmath=True, error_model="numpy")
def calc_indices_3c2e_schwarz2(sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, chunk_size, nao, naux, istart, jstart, kstart, threshold, strict_schwarz):
    # This is a variant of the previous function 'calc_indices_3c2e_schwarz'. Instead of returning arrays of 1s and 0s
    # which are then needed to be processed to get the indices of the contributing triplets.
    # Instead, here we directly calculate the indices, however, it is not straightforwardly parallelizable,
    # so it can be a bit slow for larger systems (>2000 bfs)
    indicesA = np.zeros((chunk_size), dtype=np.uint16)
    indicesB = np.zeros((chunk_size), dtype=np.uint16)
    indicesC = np.zeros((chunk_size), dtype=np.uint16)
    # Loop over the lower-triangular ints3c2e array
    count = 0
    for i in prange(istart, nao):
        for j in prange(jstart, i+1):
            sqrt_ij = sqrt_ints4c2e_diag[i,j]
            if j==i:
                jstart = 0
            if strict_schwarz:
                if sqrt_ij*sqrt_ij<1e-13:
                    continue
            for k in prange(kstart, naux):
                if k==naux-1:
                    # jstart = 0
                    kstart = 0
                if count<chunk_size:
                    if sqrt_ij*sqrt_diag_ints2c2e[k]>threshold:
                        indicesA[count] = i
                        indicesB[count] = j
                        indicesC[count] = k
                        count += 1
                else:
                    return indicesA, indicesB, indicesC, [i,j,k], count
                
    return indicesA, indicesB, indicesC, [i,j,k], count

@njit(parallel=True, cache=True, fastmath=True, error_model="numpy")
def calc_indices_3c2e_schwarz_fine(sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, chunk_size, nao, naux, istart, jstart, kstart, threshold, strict_schwarz, auxbfs_lm):
    # This is a variant of the previous function 'calc_indices_3c2e_schwarz'. Instead of returning arrays of 1s and 0s
    # which are then needed to be processed to get the indices of the contributing triplets.
    # Instead, here we directly calculate the indices, however, it is not straightforwardly parallelizable,
    # so it can be a bit slow for larger systems (>2000 bfs)
    indicesA = np.zeros((chunk_size), dtype=np.uint16)
    indicesB = np.zeros((chunk_size), dtype=np.uint16)
    indicesC = np.zeros((chunk_size), dtype=np.uint16)
    # Loop over the lower-triangular ints3c2e array
    count = 0
    for i in prange(istart, nao):
        # coord_i = bfs_coords[i]
        # cutoff_i = bfs_radius_cutoff[i]
        for j in prange(jstart, i+1):
            sqrt_ij = sqrt_ints4c2e_diag[i,j]
            if j==i:
                jstart = 0
            # coord_j = bfs_coords[j]
            if strict_schwarz:
                if sqrt_ij*sqrt_ij<1e-13:
                    continue    
            for k in prange(kstart, naux):
                if k==naux-1:
                    # jstart = 0
                    kstart = 0
                # coord_k = auxbfs_coords[k]
                # print(coord_k - coord_i) # Gives segmentation fault
                # print(coord_k)
                # print(coord_i)
                # print(np.linalg.norm(coord_k-coord_i))
                # if np.sqrt(np.sum((coord_k - coord_i)**2))>18:# and abs(np.linalg.norm(coord_k-coord_j))<bfs_radius_cutoff[j]:
                #     continue
                if count<chunk_size:
                    if strict_schwarz:
                        max_val = sqrt_ij*sqrt_diag_ints2c2e[k]
                        if max_val>threshold:
                            if max_val<1e-8:
                                if auxbfs_lm[k]<1: # s aux functions
                                    indicesA[count] = i
                                    indicesB[count] = j
                                    indicesC[count] = k
                                    count += 1
                            elif max_val<1e-7:
                                if auxbfs_lm[k]<2: # s, p aux functions
                                    indicesA[count] = i
                                    indicesB[count] = j
                                    indicesC[count] = k
                                    count += 1
                            elif max_val<1e-6:
                                if auxbfs_lm[k]<3: # s, p, d aux functions
                                    indicesA[count] = i
                                    indicesB[count] = j
                                    indicesC[count] = k
                                    count += 1
                            else:
                                indicesA[count] = i
                                indicesB[count] = j
                                indicesC[count] = k
                                count += 1
                    else:
                        if sqrt_ij*sqrt_diag_ints2c2e[k]>threshold:
                            indicesA[count] = i
                            indicesB[count] = j
                            indicesC[count] = k
                            count += 1
                else:
                    return indicesA, indicesB, indicesC, [i,j,k], count
    return indicesA, indicesB, indicesC, [i,j,k], count

@njit(parallel=True, cache=True, fastmath=True, error_model="numpy")
def calc_offsets_3c2e_schwarz(sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold, strict_schwarz, auxbfs_lm, aux_bfs_lmn, ntri, naux, tril_indicesA, tril_indicesB):
    # Calculate the offsets for 3c2e integral evluations
    offsets = np.zeros((ntri+1), dtype=np.uint16)
    offsets[0] = 0
    # Loop over the lower-triangular ints3c2e array
    for ij in prange(ntri):
        i = tril_indicesA[ij]
        j = tril_indicesB[ij]
        sqrt_ij = sqrt_ints4c2e_diag[i,j] 
        if strict_schwarz:
            if sqrt_ij*sqrt_ij<1e-13:
                continue    
        count = 0
        for k in range(naux):    
            # if strict_schwarz:
            #     max_val = sqrt_ij*sqrt_diag_ints2c2e[k]
            #     if max_val>threshold:
            #         if max_val<1e-8:
            #             if auxbfs_lm[k]<1: # s aux functions
            #                 count += 1
            #         elif max_val<1e-7:
            #             if auxbfs_lm[k]<2: # s, p aux functions
            #                 count += 1
            #         elif max_val<1e-6:
            #             if auxbfs_lm[k]<3: # s, p, d aux functions
            #                 count += 1
            #         else:
            #             count += 1
            # else:  
            #     if sqrt_ij*sqrt_diag_ints2c2e[k]>threshold:
            #         count += 1
            if sqrt_ij*sqrt_diag_ints2c2e[k]>threshold:
                count += 1
            # else:
            #     print("yes")
        offsets[ij+1] = count 
    return offsets
@njit(parallel=True, cache=True, fastmath=True, error_model="numpy")
def calc_offsets_3c2e_schwarz_n(sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold, strict_schwarz, auxbfs_lm, aux_bfs_lmn, ntri, naux, tril_indicesA, tril_indicesB):
    # Calculate the offsets for 3c2e integral evluations
    offsets = np.zeros((ntri+1), dtype=np.uint16)
    offsets[0] = 0
    # Loop over the lower-triangular ints3c2e array
    for ij in prange(ntri):
        i = tril_indicesA[ij]
        j = tril_indicesB[ij]
        sqrt_ij = sqrt_ints4c2e_diag[i,j] 
        if strict_schwarz:
            if sqrt_ij*sqrt_ij<1e-13:
                continue    
        count = 0
        k = 0
        while k < naux:    
            # if strict_schwarz:
            #     max_val = sqrt_ij*sqrt_diag_ints2c2e[k]
            #     if max_val>threshold:
            #         if max_val<1e-8:
            #             if auxbfs_lm[k]<1: # s aux functions
            #                 count += 1
            #         elif max_val<1e-7:
            #             if auxbfs_lm[k]<2: # s, p aux functions
            #                 count += 1
            #         elif max_val<1e-6:
            #             if auxbfs_lm[k]<3: # s, p, d aux functions
            #                 count += 1
            #         else:
            #             count += 1
            # else:  
            #     if sqrt_ij*sqrt_diag_ints2c2e[k]>threshold:
            #         count += 1
            lmnk = aux_bfs_lmn[k]
            tot_ang = lmnk.sum()
            is_first = lmnk[0]==tot_ang
            # print(lmnk, is_first)
            if is_first:
                if sqrt_ij*sqrt_diag_ints2c2e[k]>threshold:
                    # Determine shell size: (L+1)(L+2)/2
                    shell_size = (tot_ang + 1) * (tot_ang + 2) // 2
                    k += shell_size
                    count += shell_size
                    # continue
                else:
                    k += 1
            else:
                k += 1
                # k += ((tot_ang + 1)*(tot_ang + 2)//2)
            # if sqrt_ij*sqrt_diag_ints2c2e[k]>threshold:
            #     count += 1
            # else:
            #     print("yes")
        offsets[ij+1] = count 
    return offsets

@njit(parallel=True, cache=True, fastmath=True, error_model="numpy")
def calc_indices_3c2e_schwarz3(sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, chunk_size, nao, naux, istart, jstart, kstart, threshold):
    # This is a variant of the previous function 'calc_indices_3c2e_schwarz'. Instead of returning arrays of 1s and 0s
    # which are then needed to be processed to get the indices of the contributing triplets.
    # Instead, here we directly calculate the indices, however, it is not straightforwardly parallelizable,
    # so it can be a bit slow for larger systems (>2000 bfs)
    offset = np.zeros((nao), dtype=np.uint32)
    indicesB = np.zeros((chunk_size), dtype=np.uint16)
    indicesC = np.zeros((chunk_size), dtype=np.uint16)
    # Loop over the lower-triangular ints3c2e array
    count = 0
    indx_offset = 0
    for i in prange(istart, nao):
        for j in prange(jstart, i+1):
            sqrt_ij = sqrt_ints4c2e_diag[i,j]
            if j==i:
                jstart = 0
            for k in prange(kstart, naux):
                if k==naux-1:
                    # jstart = 0
                    kstart = 0
                if count<chunk_size:
                    if sqrt_ij*sqrt_diag_ints2c2e[k]>threshold:
                        indicesB[count] = j
                        indicesC[count] = k
                        count += 1
                        offset[i] = count
                else:
                    return offset, indicesB, indicesC, [i,j,k], count
    return offset, indicesB, indicesC, [i,j,k], count

@njit(parallel=True, cache=True, fastmath=True, error_model="numpy")
def calc_count_3c2e_schwarz(sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, nao, naux, threshold):
    # Calculates the total no. of significant 3c2e triplets after Schwarz screening
    # Loop over the lower-triangular ints3c2e array
    count = 0
    for i in prange(0, nao):
        for j in prange(0, i+1):
            sqrt_ij = sqrt_ints4c2e_diag[i,j]
            for k in prange(0, naux):
                if sqrt_ij*sqrt_diag_ints2c2e[k]>threshold:
                    count += 1
    return count

@njit(parallel=True, cache=True, fastmath=True, error_model="numpy")
def calc_indices_4c2e_schwarz(sqrt_ints4c2e_diag, chunk_size, nao, istart, jstart, kstart, lstart, threshold):
    # This is a variant of the previous function 'calc_indices_3c2e_schwarz'. Instead of returning arrays of 1s and 0s
    # which are then needed to be processed to get the indices of the contributing triplets.
    # Instead, here we directly calculate the indices, however, it is not straightforwardly parallelizable,
    # so it can be a bit slow for larger systems (>2000 bfs)
    indicesA = np.zeros((chunk_size), dtype=np.uint16)
    indicesB = np.zeros((chunk_size), dtype=np.uint16)
    indicesC = np.zeros((chunk_size), dtype=np.uint16)
    indicesD = np.zeros((chunk_size), dtype=np.uint16)
    # Loop over the lower-triangular ints3c2e array
    count = 0
    for i in prange(istart, nao):
        for j in range(jstart, i+1):
            if i<j:
                triangle2ij = (j)*(j+1)/2+i
            else:
                triangle2ij = (i)*(i+1)/2+j
            if j==i:
                jstart = 0
            sqrt_ij = sqrt_ints4c2e_diag[i,j]
            for k in range(kstart, nao):
                if k==nao:
                    kstart = 0
                for l in range(lstart, k+1):
                    if k<l:
                        triangle2kl = (l)*(l+1)/2+k
                    else:
                        triangle2kl = (k)*(k+1)/2+l
                    if triangle2ij>triangle2kl:
                        continue
                    if l==k:
                        lstart = 0
                    sqrt_kl = sqrt_ints4c2e_diag[k,l]
                    if count<chunk_size:
                        if sqrt_ij*sqrt_kl>threshold:
                            indicesA[count] = i
                            indicesB[count] = j
                            indicesC[count] = k
                            indicesD[count] = l
                            count += 1
                    else:
                        return indicesA, indicesB, indicesC, indicesD, [i,j,k,l], count
    return indicesA, indicesB, indicesC, indicesD, [i,j,k], count

@njit(parallel=True, cache=True, fastmath=True, error_model="numpy")
def calc_indices_3c2e_schwarz2_test(eri_4c2e_diag, ints2c2e, chunk_size, nao, naux, istart, jstart, kstart, threshold):
    # This is a variant of the previous function 'calc_indices_3c2e_schwarz'. Instead of returning arrays of 1s and 0s
    # which are then needed to be processed to get the indices of the contributing triplets.
    # Instead, here we directly calculate the indices, however, it is not straightforwardly parallelizable,
    # so it can be a bit slow for larger systems (>2000 bfs)
    indicesA = np.zeros((chunk_size), dtype=np.uint16)
    indicesB = np.zeros((chunk_size), dtype=np.uint16)
    indicesC = np.zeros((chunk_size), dtype=np.uint16)
    # Loop over k first
    count = 0
    for k in prange(kstart, naux):
        for i in range(istart, nao):
            jstart = i if i == istart else 0
            for j in range(jstart, i+1):
                if np.sqrt(eri_4c2e_diag[i, j]) * np.sqrt(ints2c2e[k, k]) > threshold:
                    if count < chunk_size:
                        indicesA[count] = i
                        indicesB[count] = j
                        indicesC[count] = k
                        count += 1
                    else:
                        return indicesA, indicesB, indicesC, [i,j,k], count
            kstart = 0 # reset kstart for the next j loop
    return indicesA, indicesB, indicesC, [i,j,k], count

@njit(parallel=True, cache=True, fastmath=True, error_model="numpy")
def calc_indices_3c2e_schwarz_shells(eri_4c2e_diag, ints2c2e, chunk_size, nao, naux, istart, jstart, kstart, shell_indices, aux_shell_indices, threshold):
    # This is a variant of the previous function 'calc_indices_3c2e_schwarz2'. 
    # Instead of calculating the indices which yield non zero contributions, 
    # it calculates the shell indices.
    # This will probably take the same amount of time as 
    # 'calc_indices_3c2e_schwarz2' but will require much less memory, as instead of having three indices for 
    # px,py and pz we will just have an index corresponding to p.
    indicesA_shells = np.zeros((chunk_size), dtype=np.uint16)
    indicesB_shells = np.zeros((chunk_size), dtype=np.uint16)
    indicesC_shells = np.zeros((chunk_size), dtype=np.uint16)
    # Loop over the lower-triangular ints3c2e array
    count = 0
    ishell_previous = 0
    jshell_previous = 0
    kshell_previous = 0
    for ibf in range(istart, nao):
        ishell = shell_indices[ibf]
        for jbf in range(jstart, ibf+1):
            jshell = shell_indices[jbf]
            if jbf==ibf:
                jstart = 0
            for kbf in prange(kstart, naux):
                kshell = aux_shell_indices[kbf]
                if count>0 and (ishell==ishell_previous and jshell==jshell_previous and kshell==kshell_previous):
                    # print(ishell, ishell_previous)
                    continue
                if kbf==naux-1:
                    # jstart = 0
                    kstart = 0
                if count<chunk_size:
                    if np.sqrt(eri_4c2e_diag[ibf, jbf])*np.sqrt(ints2c2e[kbf, kbf])>threshold:
                        indicesA_shells[count] = ishell
                        indicesB_shells[count] = jshell
                        indicesC_shells[count] = kshell
                        count += 1
                else:
                    return indicesA_shells, indicesB_shells, indicesC_shells, [ibf,jbf,kbf], count
                ishell_previous = ishell
                jshell_previous = jshell
                kshell_previous = kshell
    return indicesA_shells, indicesB_shells, indicesC_shells, [ibf,jbf,kbf], count

def rys_3c2e_tri_schwarz(basis, auxbasis, indicesA, indicesB, indicesC):
    # Wrapper for hybrid Rys+conv. 3c2e integral calculator
    # using a list of significant contributions obtained via Schwarz screening.
    # It returns the 3c2e integrals in triangular form.

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
    

    ints3c2e = rys_3c2e_tri_schwarz_internal(bfs_coords[0], bfs_contr_prim_norms[0], bfs_lmn[0], \
                bfs_nprim[0], bfs_coeffs, bfs_prim_norms, bfs_expnts,aux_bfs_coords[0], aux_bfs_contr_prim_norms[0], aux_bfs_lmn[0], aux_bfs_nprim[0], \
                aux_bfs_coeffs, aux_bfs_prim_norms, aux_bfs_expnts, indicesA, indicesB, indicesC, basis.bfs_nao, auxbasis.bfs_nao)
    return ints3c2e

def rys_3c2e_tri_schwarz_sparse(basis, auxbasis, indicesA, indicesB, indicesC):
    # Wrapper for hybrid Rys+conv. 3c2e integral calculator
    # using a list of significant contributions obtained via Schwarz screening.
    # It returns the 3c2e integrals in triangular form.

    # print('preprocessing starts', flush=True)

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
    shell_indices = np.array([basis.bfs_shell_index], dtype=np.uint16)[0]
    aux_shell_indices = np.array([auxbasis.bfs_shell_index], dtype=np.uint16)[0]
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

    

    diff = bfs_coords[0][:, np.newaxis, :] - bfs_coords[0][np.newaxis, :, :]
    IJsq_arr = np.sum(diff**2, axis=2)
    # IJsq_arr = 0
    # print(IJsq_arr.nbytes/1e9)

    # print('preprocessing done', flush=True)
    # exit()
    

    ints3c2e = rys_3c2e_tri_schwarz_sparse_internal(bfs_coords[0], bfs_contr_prim_norms[0], bfs_lmn[0], \
                bfs_nprim[0], bfs_coeffs, bfs_prim_norms, bfs_expnts,aux_bfs_coords[0], aux_bfs_contr_prim_norms[0], aux_bfs_lmn[0], aux_bfs_nprim[0], \
                aux_bfs_coeffs, aux_bfs_prim_norms, aux_bfs_expnts, indicesA, indicesB, indicesC, basis.bfs_nao, auxbasis.bfs_nao, IJsq_arr, shell_indices, aux_shell_indices)
    # rys_3c2e_tri_schwarz_sparse_internal.parallel_diagnostics(level=1)
    return ints3c2e

def rys_3c2e_tri_schwarz_sparse_algo10_sao(basis, auxbasis, indicesA, indicesB, offsets, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold, strict_schwarz, nsignificant):
    # Wrapper for hybrid Rys+conv. 3c2e integral calculator
    # using a list of significant contributions obtained via Schwarz screening.
    # It returns the 3c2e integrals in triangular form.

    # print('preprocessing starts', flush=True)

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
    shell_indices = np.array([basis.bfs_shell_index], dtype=np.uint16)[0]
    aux_shell_indices = np.array([auxbasis.bfs_shell_index], dtype=np.uint16)[0]
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

    

    diff = bfs_coords[0][:, np.newaxis, :] - bfs_coords[0][np.newaxis, :, :]
    IJsq_arr = np.sum(diff**2, axis=2)
    # IJsq_arr = 0
    # print(IJsq_arr.nbytes/1e9)

    # print('preprocessing done', flush=True)
    # exit()

    
    ints3c2e = rys_3c2e_tri_schwarz_sparse_algo10_sao_internal(bfs_coords[0], bfs_contr_prim_norms[0], bfs_lmn[0], \
                bfs_nprim[0], bfs_coeffs, bfs_prim_norms, bfs_expnts,aux_bfs_coords[0], aux_bfs_contr_prim_norms[0], aux_bfs_lmn[0], aux_bfs_nprim[0], \
                aux_bfs_coeffs, aux_bfs_prim_norms, aux_bfs_expnts, indicesA, indicesB, offsets, basis.bfs_nao, auxbasis.bfs_nao, IJsq_arr, shell_indices, aux_shell_indices, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold, strict_schwarz, nsignificant)
    # rys_3c2e_tri_schwarz_sparse_internal.parallel_diagnostics(level=1)
    return ints3c2e

def rys_3c2e_tri_schwarz_sparse_algo10(basis, auxbasis, indicesA, indicesB, offsets, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold, strict_schwarz, nsignificant):
    # Wrapper for hybrid Rys+conv. 3c2e integral calculator
    # using a list of significant contributions obtained via Schwarz screening.
    # It returns the 3c2e integrals in triangular form.

    # print('preprocessing starts', flush=True)

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
    shell_indices = np.array([basis.bfs_shell_index], dtype=np.uint16)[0]
    aux_shell_indices = np.array([auxbasis.bfs_shell_index], dtype=np.uint16)[0]
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

    

    diff = bfs_coords[0][:, np.newaxis, :] - bfs_coords[0][np.newaxis, :, :]
    IJsq_arr = np.sum(diff**2, axis=2)
    # IJsq_arr = 0
    # print(IJsq_arr.nbytes/1e9)

    # print('preprocessing done', flush=True)
    # exit()
    

    ints3c2e = rys_3c2e_tri_schwarz_sparse_algo10_internal(bfs_coords[0], bfs_contr_prim_norms[0], bfs_lmn[0], \
                bfs_nprim[0], bfs_coeffs, bfs_prim_norms, bfs_expnts,aux_bfs_coords[0], aux_bfs_contr_prim_norms[0], aux_bfs_lmn[0], aux_bfs_nprim[0], \
                aux_bfs_coeffs, aux_bfs_prim_norms, aux_bfs_expnts, indicesA, indicesB, offsets, basis.bfs_nao, auxbasis.bfs_nao, IJsq_arr, shell_indices, aux_shell_indices, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold, strict_schwarz, nsignificant)
    # rys_3c2e_tri_schwarz_sparse_internal.parallel_diagnostics(level=1)
    return ints3c2e

def rys_3c2e_tri_schwarz_sparse2(basis, auxbasis, indicesA, indicesB, indicesC, ncores):
    # Wrapper for hybrid Rys+conv. 3c2e integral calculator
    # using a list of significant contributions obtained via Schwarz screening.
    # It returns the 3c2e integrals in triangular form.

    # print('preprocessing starts', flush=True)

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

    

    diff = bfs_coords[0][:, np.newaxis, :] - bfs_coords[0][np.newaxis, :, :]
    IJsq_arr = np.sum(diff**2, axis=2)
    # IJsq_arr = 0

    # print('preprocessing done', flush=True)
    # exit()
    ntriplets = indicesA.shape[0] # No. of significant triplets
    ints3c2e = np.zeros((ntriplets), dtype=np.float64) 

    # This works but not the best way to calculate
    # chunk_size = min(ntriplets, 100000)
    # nchunks = ntriplets // chunk_size #+ 1

    if ntriplets>100000: # No parallelization upto 100000 values
        nchunks = ncores*2
        chunk_size = ntriplets//nchunks


    print(nchunks)
    print(chunk_size)

    # ints3c2e = rys_3c2e_tri_schwarz_sparse_internal2(bfs_coords[0], bfs_contr_prim_norms[0], bfs_lmn[0], \
    #             bfs_nprim[0], bfs_coeffs, bfs_prim_norms, bfs_expnts,aux_bfs_coords[0], aux_bfs_contr_prim_norms[0], aux_bfs_lmn[0], aux_bfs_nprim[0], \
    #             aux_bfs_coeffs, aux_bfs_prim_norms, aux_bfs_expnts, indicesA, indicesB, indicesC, basis.bfs_nao, auxbasis.bfs_nao, IJsq_arr, chunk_size, threeC2E)
    # output = Parallel(n_jobs=ncores, backend='threading', require='sharedmem')(delayed(rys_3c2e_tri_schwarz_sparse_internal2)(iblock, blocksize, weights[iblock*blocksize : min(iblock*blocksize+blocksize,ngrids)], coords[iblock*blocksize : min(iblock*blocksize+blocksize,ngrids)], ngrids, basis, dmat[np.ix_(list_nonzero_indices[iblock][0:count_nonzero_indices[iblock]], list_nonzero_indices[iblock][0:count_nonzero_indices[iblock]])], funcid, bfs_data_as_np_arrays, list_nonzero_indices[iblock][0:count_nonzero_indices[iblock]], list_ao_values[iblock], funcx=funcx, funcc=funcc, x_family_code=x_family_code, c_family_code=c_family_code, xc_family_dict=xc_family_dict) for ichunk in range(nchunks))
    output = Parallel(n_jobs=ncores, backend='threading', require='sharedmem')(delayed(rys_3c2e_tri_schwarz_sparse_internal2)(bfs_coords[0], bfs_contr_prim_norms[0], bfs_lmn[0], \
                bfs_nprim[0], bfs_coeffs, bfs_prim_norms, bfs_expnts,aux_bfs_coords[0], aux_bfs_contr_prim_norms[0], aux_bfs_lmn[0], aux_bfs_nprim[0], \
                aux_bfs_coeffs, aux_bfs_prim_norms, aux_bfs_expnts, indicesA[ichunk*chunk_size : min(ichunk*chunk_size+chunk_size,ntriplets)], indicesB[ichunk*chunk_size : min(ichunk*chunk_size+chunk_size,ntriplets)], indicesC[ichunk*chunk_size : min(ichunk*chunk_size+chunk_size,ntriplets)], basis.bfs_nao, auxbasis.bfs_nao, IJsq_arr, chunk_size, ints3c2e[ichunk*chunk_size : min(ichunk*chunk_size+chunk_size,ntriplets)]) for ichunk in range(nchunks+1))
    # print(ints3c2e)
    return ints3c2e

def rys_3c2e_tri_schwarz_sparse_algo8(basis, auxbasis, ntriplets, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold):
    # Wrapper for hybrid Rys+conv. 3c2e integral calculator
    # using a list of significant contributions obtained via Schwarz screening.
    # It returns the 3c2e integrals in triangular form.

    # print('preprocessing starts', flush=True)

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
    shell_indices = np.array([basis.bfs_shell_index], dtype=np.uint16)[0]
    aux_shell_indices = np.array([auxbasis.bfs_shell_index], dtype=np.uint16)[0]
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
    

    ints3c2e = rys_3c2e_tri_schwarz_sparse_algo8_internal(bfs_coords[0], bfs_contr_prim_norms[0], bfs_lmn[0], \
                bfs_nprim[0], bfs_coeffs, bfs_prim_norms, bfs_expnts,aux_bfs_coords[0], aux_bfs_contr_prim_norms[0], aux_bfs_lmn[0], aux_bfs_nprim[0], \
                aux_bfs_coeffs, aux_bfs_prim_norms, aux_bfs_expnts, ntriplets, basis.bfs_nao, auxbasis.bfs_nao, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold)
    # rys_3c2e_tri_schwarz_sparse_internal.parallel_diagnostics(level=1)
    return ints3c2e

def rys_3c2e_tri_schwarz_sparse_algo9(basis, auxbasis, offset, indicesB, indicesC):
    # Wrapper for hybrid Rys+conv. 3c2e integral calculator
    # using a list of significant contributions obtained via Schwarz screening.
    # It returns the 3c2e integrals in triangular form.

    # print('preprocessing starts', flush=True)

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
    shell_indices = np.array([basis.bfs_shell_index], dtype=np.uint16)[0]
    aux_shell_indices = np.array([auxbasis.bfs_shell_index], dtype=np.uint16)[0]
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

    

    diff = bfs_coords[0][:, np.newaxis, :] - bfs_coords[0][np.newaxis, :, :]
    IJsq_arr = np.sum(diff**2, axis=2)
    

    ints3c2e = rys_3c2e_tri_schwarz_sparse_algo9_internal(bfs_coords[0], bfs_contr_prim_norms[0], bfs_lmn[0], \
                bfs_nprim[0], bfs_coeffs, bfs_prim_norms, bfs_expnts,aux_bfs_coords[0], aux_bfs_contr_prim_norms[0], aux_bfs_lmn[0], aux_bfs_nprim[0], \
                aux_bfs_coeffs, aux_bfs_prim_norms, aux_bfs_expnts, offset, indicesB, indicesC, basis.bfs_nao, auxbasis.bfs_nao, IJsq_arr, shell_indices, aux_shell_indices)
    return ints3c2e


@njit(parallel=True, cache=True, fastmath=True, error_model="numpy")
def rys_3c2e_tri_schwarz_internal(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts,aux_bfs_coords, aux_bfs_contr_prim_norms, aux_bfs_lmn, aux_bfs_nprim, aux_bfs_coeffs, aux_bfs_prim_norms, aux_bfs_expnts, indicesA, indicesB, indicesC, nao, naux):
    # Calculates 3c2e integrals based on a provided list of significant triplets determined using Schwarz inequality
    # It is assumed that the provided list was made with triangular int3c2e in mind

    threeC2E = np.zeros((int(nao*(nao+1)/2.0),naux), dtype=np.float64) 

    ntriplets = indicesA.shape[0] # No. of significant triplets
   
        
    pi = 3.141592653589793
    # pisq = 9.869604401089358  #PI^2
    twopisq = 19.739208802178716  #2*PI^2
    L = np.zeros((3))
    alphalk = 0.0
    ld, md, nd = int(0), int(0), int(0)

    # Create arrays to avoid contention of allocator 
    # (https://stackoverflow.com/questions/70339388/using-numba-with-np-concatenate-is-not-efficient-in-parallel/70342014#70342014)
    
    #Loop over BFs
    for itemp in prange(ntriplets): 
        i = indicesA[itemp]
        j = indicesB[itemp]
        k = indicesC[itemp]
        offset = int(i*(i+1)/2)
        I = bfs_coords[i]
        Ni = bfs_contr_prim_norms[i]
        lmni = bfs_lmn[i]
        la, ma, na = lmni

        J = bfs_coords[j]
        IJ = I - J
        IJsq = np.sum(IJ**2)
        Nj = bfs_contr_prim_norms[j]
        lmnj = bfs_lmn[j]
        lb, mb, nb = lmnj
        tempcoeff1 = Ni*Nj
        
        K = aux_bfs_coords[k]
        Nk = aux_bfs_contr_prim_norms[k]
        lmnk = aux_bfs_lmn[k]
        lc, mc, nc = lmnk
        tempcoeff2 = tempcoeff1*Nk
                
               
                  
        KL = K - L
        KLsq = np.sum(KL**2)
                
                # npriml = bfs_nprim[l]

        norder = int((la+ma+na+lb+mb+nb+lc+mc+nc+ld+md+nd)/2 + 1 ) 
        val = 0.0
        if norder<=10: # Use rys quadrature
            n = int(max(la+lb,ma+mb,na+nb))
            m = int(max(lc+ld,mc+md,nc+nd))
            roots = np.zeros((norder))
            weights = np.zeros((norder))
            G = np.zeros((n+1,m+1))
                

            
                
            #Loop over primitives
            for ik in range(bfs_nprim[i]):   
                dik = bfs_coeffs[i,ik]
                Nik = bfs_prim_norms[i,ik]
                alphaik = bfs_expnts[i,ik]
                tempcoeff3 = tempcoeff2*dik*Nik
                    
                for jk in range(bfs_nprim[j]):
                    alphajk = bfs_expnts[j,jk]
                    gammaP = alphaik + alphajk
                    screenfactorAB = np.exp(-alphaik*alphajk/gammaP*IJsq)
                    if abs(screenfactorAB)<1.0e-8:   
                        #Although this value of screening threshold seems very large
                        # it actually gives the best consistency with PySCF. Reducing it to 1e-15,
                        # actually worsened the agreement.
                        # I suspect that this is caused due to an error cancellation
                        # that happens with the nucmat calculation, as the same screening is 
                        # used there as well
                        continue
                    djk = bfs_coeffs[j,jk] 
                    Njk = bfs_prim_norms[j,jk]      
                    P = (alphaik*I + alphajk*J)/gammaP
                    tempcoeff4 = tempcoeff3*djk*Njk  
                        
                    for kk in range(aux_bfs_nprim[k]):
                        dkk = aux_bfs_coeffs[k,kk]
                        Nkk = aux_bfs_prim_norms[k,kk]
                        alphakk = aux_bfs_expnts[k,kk]
                        tempcoeff5 = tempcoeff4*dkk*Nkk 
                              
                            
                        gammaQ = alphakk #+ alphalk
                        screenfactorKL = np.exp(-alphakk*alphalk/gammaQ*KLsq)
                        if abs(screenfactorKL)<1.0e-8:   
                            #Although this value of screening threshold seems very large
                            # it actually gives the best consistency with PySCF. Reducing it to 1e-15,
                            # actually worsened the agreement.
                            # I suspect that this is caused due to an error cancellation
                            # that happens with the nucmat calculation, as the same screening is 
                            # used there as well
                            continue
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
                dik = bfs_coeffs[i,ik]
                Nik = bfs_prim_norms[i,ik]
                alphaik = bfs_expnts[i,ik]
                tempcoeff3 = tempcoeff2*dik*Nik
                    
                for jk in range(bfs_nprim[j]):
                    alphajk = bfs_expnts[j,jk]
                    gammaP = alphaik + alphajk
                    screenfactorAB = np.exp(-alphaik*alphajk/gammaP*IJsq)
                    if abs(screenfactorAB)<1.0e-8:   
                        #Although this value of screening threshold seems very large
                        # it actually gives the best consistency with PySCF. Reducing it to 1e-15,
                        # actually worsened the agreement.
                        # I suspect that this is caused due to an error cancellation
                        # that happens with the nucmat calculation, as the same screening is 
                        # used there as well
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
                        dkk = aux_bfs_coeffs[k,kk]
                        Nkk = aux_bfs_prim_norms[k,kk]
                        alphakk = aux_bfs_expnts[k,kk]
                        tempcoeff5 = tempcoeff4*dkk*Nkk 
                              
                        
                        gammaQ = alphakk #+ alphalk
                        screenfactorKL = np.exp(-alphakk*alphalk/gammaQ*KLsq)
                        if abs(screenfactorKL)<1.0e-8:   
                            #Although this value of screening threshold seems very large
                            # it actually gives the best consistency with PySCF. Reducing it to 1e-15,
                            # actually worsened the agreement.
                            # I suspect that this is caused due to an error cancellation
                            # that happens with the nucmat calculation, as the same screening is 
                            # used there as well
                            continue
                        
                        Q = K     
                        PQ = P - Q
                                
                        QK = Q - K
                        QL = Q #- L
                        
                                  
                        fac2 = fac1/gammaQ
                                

                        omega = (fac2)*np.sqrt(pi/(gammaP + gammaQ))
                        delta = 0.25*(1/gammaQ) + onefourthgammaPinv          
                        PQsqBy4delta = np.sum(PQ**2)/(4*delta)         
                                    
                        sum1 = innerLoop4c2e(la,lb,lc,ld,ma,mb,mc,md,na,nb,nc,nd,gammaP,gammaQ,PI,PJ,QK,QL,PQ,PQsqBy4delta,delta)
                                    
                        val += omega*sum1*tempcoeff5

        threeC2E[j+offset,k] = val
                                    
                                   
                            
        
    return threeC2E

@njit(parallel=True, cache=True, fastmath=True, error_model="numpy", nogil=True, boundscheck=False)
def rys_3c2e_tri_schwarz_sparse_internal(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts,aux_bfs_coords, aux_bfs_contr_prim_norms, aux_bfs_lmn, aux_bfs_nprim, aux_bfs_coeffs, aux_bfs_prim_norms, aux_bfs_expnts, indicesA, indicesB, indicesC, nao, naux, IJsq_arr, shell_indices, aux_shell_indices):
    # Calculates 3c2e integrals based on a provided list of significant triplets determined using Schwarz inequality
    # It is assumed that the provided list was made with triangular int3c2e in mind
    # TODO: Check out these screening methods as well 
    # https://kops.uni-konstanz.de/server/api/core/bitstreams/79ada61a-fd29-43fd-a298-79c1696a0601/content
    # https://aip.scitation.org/doi/10.1063/1.4917519

    ntriplets = indicesA.shape[0] # No. of significant triplets
    threeC2E = np.zeros((ntriplets), dtype=np.float64) 

    #Debug:
    # print(config.THREADING_LAYER)
    # print(threading_layer())
    # print(get_parallel_chunksize())
    # print(ntriplets)
    # set_parallel_chunksize(ntriplets//128)
    # print(get_parallel_chunksize())
        
    pi = 3.141592653589793
    # pisq = 9.869604401089358  #PI^2
    twopisq = 19.739208802178716  #2*PI^2
    L = np.zeros((3))
    alphalk = 0.0
    ld, md, nd = int(0), int(0), int(0)

    # Initialize before hand to avoid contention of memory allocator
    # roots = np.zeros((5)) # 5 is the maximum possible value currently in the code
    # weights = np.zeros((5))
    # G = np.zeros((21,21)) # 11 should be enough

    
    maxprims = bfs_coeffs.shape[1]
    maxprims_aux = aux_bfs_coeffs.shape[1]

    # Create arrays to avoid contention of allocator 
    # (https://stackoverflow.com/questions/70339388/using-numba-with-np-concatenate-is-not-efficient-in-parallel/70342014#70342014)
    

    #Loop over BFs
    for itemp in prange(ntriplets):
        # id_thrd = get_thread_id()

        

        i = indicesA[itemp]
        j = indicesB[itemp]
        k = indicesC[itemp]
        
        

        # ishell = shell_indices[i]
        
        Ni = bfs_contr_prim_norms[i]
        nprimi = bfs_nprim[i]
        alphaik = np.zeros(maxprims, dtype=np.float64) # Should be Hoisted out
        alphaik[:] = bfs_expnts[i,:]
        lmni = bfs_lmn[i]
        la, ma, na = lmni
        I = bfs_coords[i]

        # jshell = shell_indices[j]
        
        Nj = bfs_contr_prim_norms[j]
        nprimj = bfs_nprim[j]
        alphajk = np.zeros(maxprims, dtype=np.float64) # Should be Hoisted out
        alphajk[:] = bfs_expnts[j,:]
        lmnj = bfs_lmn[j]
        lb, mb, nb = lmnj
        J = bfs_coords[j]

        
        IJsq = IJsq_arr[i,j]
        tempcoeff1 = Ni*Nj
        gammaP = np.zeros((maxprims, maxprims), dtype=np.float64) # Should be Hoisted out
        screenfactorAB = np.zeros((maxprims, maxprims), dtype=np.float64) # Should be Hoisted out
        # Ap = np.zeros((bfs_coeffs.shape[1], bfs_coeffs.shape[1]), dtype=np.float64) # Should be Hoisted out
        
        # kshell = aux_shell_indices[k]
        Nk = aux_bfs_contr_prim_norms[k]
        nprimk = aux_bfs_nprim[k]
        alphakk = np.zeros(maxprims_aux, dtype=np.float64) # Should be Hoisted out
        alphakk[:] = aux_bfs_expnts[k,:]
        lmnk = aux_bfs_lmn[k]
        lc, mc, nc = lmnk
        K = aux_bfs_coords[k]
        Q = K   

        
        tempcoeff2 = tempcoeff1*Nk
        norder = int((la+ma+na+lb+mb+nb+lc+mc+nc+ld+md+nd)/2 + 1 ) 
        tempcoeff3 = np.zeros(maxprims, dtype=np.float64) # Should be Hoisted out 
        tempcoeff3[:] = tempcoeff2*bfs_coeffs[i,:]*bfs_prim_norms[i,:]
        # PQsq = np.zeros((bfs_coeffs.shape[1], bfs_coeffs.shape[1], aux_bfs_coeffs.shape[1]), dtype=np.float64)
        # rho = np.zeros((bfs_coeffs.shape[1], bfs_coeffs.shape[1], aux_bfs_coeffs.shape[1]), dtype=np.float64)
            
            
            
        val = 0.0
        if norder<=10: # Use rys quadrature
            
            n = int(max(la+lb,ma+mb,na+nb))
            m = int(max(lc+ld,mc+md,nc+nd))
            # roots = np.zeros((norder)) 
            # weights = np.zeros((norder)) 
            roots = np.zeros((10)) 
            weights = np.zeros((10)) 
            # G = np.zeros((n+1,m+1)) 
            G = np.zeros((n+1,m+1)) 
            
                
            #Loop over primitives
            for ik in range(nprimi): 
                alphaik_ = alphaik[ik]
                tempcoeff3_ = tempcoeff3[ik]
                for jk in range(nprimj):
                    alphajk_ = alphajk[jk]
                    # gammaP[ik,jk] = alphaik_ + alphajk_
                    gammaP_ = alphaik_ + alphajk_
                    gamma_inv = 1/gammaP_
                    # Ap[ik,jk] = alphaik[ik]*alphajk[jk]
                    # screenfactorAB[ik,jk] = np.exp(-Ap[ik,jk]/gammaP[ik,jk]*IJsq)
                    screenfactorAB = np.exp(-alphaik_*alphajk_/gammaP_*IJsq)
                        
                    if abs(screenfactorAB)<1.0e-8:   
                        #Although this value of screening threshold seems very large
                        # it actually gives the best consistency with PySCF. Reducing it to 1e-15,
                        # actually worsened the agreement.
                        # I suspect that this is caused due to an error cancellation
                        # that happens with the nucmat calculation, as the same screening is 
                        # used there as well
                        continue

                    djk = bfs_coeffs[j,jk] 
                    Njk = bfs_prim_norms[j,jk]     
                    P = (alphaik_*I + alphajk_*J)/gammaP_
                    PQ = P - Q
                    PQsq = np.sum(PQ**2)
                    tempcoeff4 = tempcoeff3_*djk*Njk  

                    # screenfactor_2 = tempcoeff4/Nk*screenfactorAB*gamma_inv*np.sqrt(gamma_inv)*15.5031383401
                    # if abs(screenfactor_2)<1.0e-9: # The threshold used here should be the same as Schwarz screening threshold
                    #     continue 
                    
                    # gammaP_ = gammaP[ik,jk]
                        
                    for kk in range(nprimk):

                        dkk = aux_bfs_coeffs[k,kk]
                        Nkk = aux_bfs_prim_norms[k,kk]
                        tempcoeff5 = tempcoeff4*dkk*Nkk 
                              
                            
                        gammaQ = alphakk[kk]
                        rho = gammaP_*gammaQ/(gammaP_ + gammaQ)
                                
                        # ABsrt = np.sqrt(gammaP[ik,jk]*alphakk[kk])
                        # X = PQsq*rho
                        # factor = 2*np.sqrt(rho/pi)
                        # print('s')
                        val += tempcoeff5*coulomb_rys_3c2e(roots,weights,G,PQsq, rho, norder,n,m,la,lb,lc,ld,ma,mb,mc,md,na,nb,nc,nd,alphaik_, alphajk_,alphakk[kk],alphalk,I,J,K,L,P)
                        # The following should have been faster but isnt somehow
                        # val += tempcoeff5*coulomb_rys_new(roots,weights,G,PQsq, rho, norder,n,m,la,lb,lc,ld,ma,mb,mc,md,na,nb,nc,nd,alphaik_, alphajk_,alphakk[kk],alphalk,I,J,K,L)
                        # The following should have been faster but isnt
                        # val += tempcoeff5*coulomb_rys_fast(roots,weights,G,norder,la,lb,lc,ld,ma,mb,mc,md,na,nb,nc,nd,alphaik[ik], alphajk[jk], alphakk[kk], alphalk,I,J,K,L,X,gammaP[ik,jk],alphakk[kk],Ap[ik,jk],0.0,ABsrt,factor,P,Q)
                            
        else: # Analytical (Conventional)
            #Loop over primitives
            for ik in range(nprimi):   
                for jk in range(nprimj):
                    
                    gammaP[ik,jk] = alphaik[ik] + alphajk[jk]
                    # Ap[ik,jk] = alphaik[ik]*alphajk[jk]
                    # screenfactorAB[ik,jk] = np.exp(-Ap[ik,jk]/gammaP[ik,jk]*IJsq)
                    screenfactorAB[ik,jk] = np.exp(-alphaik[ik]*alphajk[jk]/gammaP[ik,jk]*IJsq)
                        
                    if abs(screenfactorAB[ik,jk])<1.0e-8:   
                        #Although this value of screening threshold seems very large
                        # it actually gives the best consistency with PySCF. Reducing it to 1e-15,
                        # actually worsened the agreement.
                        # I suspect that this is caused due to an error cancellation
                        # that happens with the nucmat calculation, as the same screening is 
                        # used there as well
                        continue
                    djk = bfs_coeffs[j,jk] 
                    Njk = bfs_prim_norms[j,jk]      
                    P = (alphaik[ik]*I + alphajk[jk]*J)/gammaP[ik,jk]
                    PI = P - I
                    PJ = P - J  
                    PQ = P - Q
                    fac1 = twopisq/gammaP[ik,jk]*screenfactorAB[ik,jk]  
                    onefourthgammaPinv = 0.25/gammaP[ik,jk]
                    tempcoeff4 = tempcoeff3[ik]*djk*Njk  
                        
                    for kk in range(nprimk):
                        dkk = aux_bfs_coeffs[k,kk]
                        Nkk = aux_bfs_prim_norms[k,kk]
                        tempcoeff5 = tempcoeff4*dkk*Nkk 
                              
                        
                        gammaQ = alphakk[kk] #+ alphalk
                        
                        
                                
                        QK = Q - K
                        QL = Q #- L
                        
                                  
                        fac2 = fac1/gammaQ
                                

                        omega = (fac2)*np.sqrt(pi/(gammaP[ik,jk] + gammaQ))
                        delta = 0.25*(1/gammaQ) + onefourthgammaPinv          
                        PQsqBy4delta = np.sum(PQ**2)/(4*delta)         
                                    
                        sum1 = innerLoop4c2e(la,lb,lc,ld,ma,mb,mc,md,na,nb,nc,nd,gammaP[ik,jk],gammaQ,PI,PJ,QK,QL,PQ,PQsqBy4delta,delta)
                                    
                        val += omega*sum1*tempcoeff5

        threeC2E[itemp] = val
                                    
                                 

    return threeC2E

# Compile-time constants (point 7)
PI = 3.141592653589793
TWOPISQ = 19.739208802178716
EXP_THRESHOLD = 18.42
SCREEN_THRESHOLD = 1e-8

# Pre-compute transformation matrices as module-level constants (point 6)
M_F = np.array((
    [10/19,  0,  0, -3*np.sqrt(5)/19,  0, -3*np.sqrt(5)/19,  0,  0,  0,  0],
    [ 0, 14/19,  0,  0,  0,  0, -3*np.sqrt(5)/19,  0, -5/19,  0],
    [ 0,  0, 14/19,  0,  0,  0,  0, -5/19,  0, -3*np.sqrt(5)/19],
    [-3*np.sqrt(5)/19,  0,  0, 14/19,  0, -5/19,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  1,  0,  0,  0,  0,  0],
    [-3*np.sqrt(5)/19,  0,  0, -5/19,  0, 14/19,  0,  0,  0,  0],
    [ 0, -3*np.sqrt(5)/19,  0,  0,  0,  0, 10/19,  0, -3*np.sqrt(5)/19,  0],
    [ 0,  0, -5/19,  0,  0,  0,  0, 14/19,  0, -3*np.sqrt(5)/19],
    [ 0, -5/19,  0,  0,  0,  0, -3*np.sqrt(5)/19,  0, 14/19,  0],
    [ 0,  0, -3*np.sqrt(5)/19,  0,  0,  0,  0, -3*np.sqrt(5)/19,  0, 10/19]
), dtype=np.float64)

M_G = np.array((
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
), dtype=np.float64)

# Optimized transformation functions (point 6)
@njit(cache=True, fastmath=True, error_model="numpy", nogil=True, inline='always')
def fast_transform_f(output, f_buffer, offset):
    """In-place F-orbital transformation"""
    for i in range(10):
        acc = 0.0
        for j in range(10):
            acc += M_F[i, j] * f_buffer[j]
        output[offset + i] = acc

@njit(cache=True, fastmath=True, error_model="numpy", nogil=True, inline='always')
def fast_transform_g(output, g_buffer, offset):
    """In-place G-orbital transformation"""
    for i in range(15):
        acc = 0.0
        for j in range(15):
            acc += M_G[i, j] * g_buffer[j]
        output[offset + i] = acc

@njit(parallel=True, cache=True, fastmath=True, error_model="numpy", nogil=True, boundscheck=False)
def rys_3c2e_tri_schwarz_sparse_algo10_sao_internal_new(
    bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, 
    bfs_prim_norms, bfs_expnts, aux_bfs_coords, aux_bfs_contr_prim_norms, 
    aux_bfs_lmn, aux_bfs_nprim, aux_bfs_coeffs, aux_bfs_prim_norms, 
    aux_bfs_expnts, indicesA, indicesB, offsets, nao, naux, IJsq_arr, 
    shell_indices, aux_shell_indices, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, 
    threshold, strict_schwarz, nsignificant):
    
    threeC2E = np.zeros((nsignificant), dtype=np.float64) 
    
    L = np.zeros((3), dtype=np.float64)
    alphalk = 0.0
    
    maxprims = bfs_coeffs.shape[1]
    maxprims_aux = aux_bfs_coeffs.shape[1]
    
    # Loop over BFs
    for itemp in prange(indicesA.shape[0]):
        i = indicesA[itemp]
        j = indicesB[itemp]
        
        sqrt_ij = sqrt_ints4c2e_diag[i, j] 

        if strict_schwarz:
            if sqrt_ij * sqrt_ij < 1e-13:
                continue  

        # Point 2: Allocate thread-local buffers once per (i,j) pair
        alphaik = np.empty(maxprims, dtype=np.float64)
        alphajk = np.empty(maxprims, dtype=np.float64)
        alphakk = np.empty(maxprims_aux, dtype=np.float64)
        tempcoeff3 = np.empty(maxprims, dtype=np.float64)
        d_buffer = np.empty(6, dtype=np.float64)
        f_buffer = np.empty(10, dtype=np.float64)
        g_buffer = np.empty(15, dtype=np.float64)
        roots = np.empty(10, dtype=np.float64)
        weights = np.empty(10, dtype=np.float64)
        G = np.empty((20, 20), dtype=np.float64)
        
        Ni = bfs_contr_prim_norms[i]
        nprimi = bfs_nprim[i]
        alphaik[:nprimi] = bfs_expnts[i, :nprimi]
        lmni = bfs_lmn[i]
        la, ma, na = lmni[0], lmni[1], lmni[2]
        I = bfs_coords[i]
        
        # Point 4: Cache coordinates
        I_0, I_1, I_2 = I[0], I[1], I[2]

        Nj = bfs_contr_prim_norms[j]
        nprimj = bfs_nprim[j]
        alphajk[:nprimj] = bfs_expnts[j, :nprimj]
        lmnj = bfs_lmn[j]
        lb, mb, nb = lmnj[0], lmnj[1], lmnj[2]
        J = bfs_coords[j]
        
        # Point 4: Cache coordinates
        J_0, J_1, J_2 = J[0], J[1], J[2]
        
        IJsq = IJsq_arr[i, j]
        tempcoeff1 = Ni * Nj

        # Screening for diffuse basis sets
        alphaik_min = alphaik[0]
        for ik in range(1, nprimi):
            if alphaik[ik] < alphaik_min:
                alphaik_min = alphaik[ik]
        
        alphajk_min = alphajk[0]
        for jk in range(1, nprimj):
            if alphajk[jk] < alphajk_min:
                alphajk_min = alphajk[jk]
        
        gammaP_min = alphaik_min + alphajk_min
        rho_min = alphaik_min * alphajk_min / gammaP_min
        arg = rho_min * IJsq
        if arg > EXP_THRESHOLD:
            continue
        
        index_k = 0
        k = 0
        
        while k < naux:
            lmnk = aux_bfs_lmn[k]
            lc, mc, nc = lmnk[0], lmnk[1], lmnk[2]
            tot_ang = lc + mc + nc
            is_first = (lmnk[0] != 0) and (lmnk[0] + lmnk[1] + lmnk[2] == lmnk[0])
            
            if is_first:
                if sqrt_ij * sqrt_diag_ints2c2e[k] < threshold:
                    if tot_ang == 0:
                        k += 1
                    elif tot_ang == 1:
                        k += 3
                    elif tot_ang == 2:
                        k += 6
                    elif tot_ang == 3:
                        k += 10
                    elif tot_ang == 4:
                        k += 15
                    continue
            
            if is_first:
                if tot_ang == 2:
                    d_idx = 0
                elif tot_ang == 3:
                    f_idx = 0
                elif tot_ang == 4:
                    g_idx = 0
            
            Nk = aux_bfs_contr_prim_norms[k]
            nprimk = aux_bfs_nprim[k]
            alphakk[:nprimk] = aux_bfs_expnts[k, :nprimk]
            
            K = aux_bfs_coords[k]
            Q = K
            
            # Point 4: Cache coordinates
            Q_0, Q_1, Q_2 = Q[0], Q[1], Q[2]
            
            tempcoeff2 = tempcoeff1 * Nk
            norder = int((la + ma + na + lb + mb + nb + lc + mc + nc) / 2 + 1) 
            
            # Vectorize coefficient multiplication
            for ik in range(nprimi):
                tempcoeff3[ik] = tempcoeff2 * bfs_coeffs[i, ik] * bfs_prim_norms[i, ik]
            
            val = 0.0
            if norder <= 10:
                n = int(max(la + lb, ma + mb, na + nb))
                m = int(max(lc, mc, nc))
                
                # Keep original nested loop structure (faster for this use case)
                for ik in range(nprimi): 
                    alphaik_ = alphaik[ik]
                    tempcoeff3_ = tempcoeff3[ik]
                    
                    for jk in range(nprimj):
                        alphajk_ = alphajk[jk]
                        gammaP_ = alphaik_ + alphajk_
                        gamma_inv = 1.0 / gammaP_
                        
                        arg = alphaik_ * alphajk_ * gamma_inv * IJsq
                        if arg > EXP_THRESHOLD:
                            continue
                        
                        # Point 4: Use cached coordinates
                        P_0 = (alphaik_ * I_0 + alphajk_ * J_0) * gamma_inv
                        P_1 = (alphaik_ * I_1 + alphajk_ * J_1) * gamma_inv
                        P_2 = (alphaik_ * I_2 + alphajk_ * J_2) * gamma_inv
                        
                        PQ_0 = P_0 - Q_0
                        PQ_1 = P_1 - Q_1
                        PQ_2 = P_2 - Q_2
                        PQsq = PQ_0 * PQ_0 + PQ_1 * PQ_1 + PQ_2 * PQ_2
                        
                        djk = bfs_coeffs[j, jk] 
                        Njk = bfs_prim_norms[j, jk]     
                        tempcoeff4 = tempcoeff3_ * djk * Njk  
                        
                        for kk in range(nprimk):
                            dkk = aux_bfs_coeffs[k, kk]
                            Nkk = aux_bfs_prim_norms[k, kk]
                            tempcoeff5 = tempcoeff4 * dkk * Nkk 
                            
                            alphakk_ = alphakk[kk]
                            gammaQ = alphakk_
                            rho = gammaP_ * gammaQ / (gammaP_ + gammaQ)
                            
                            # Reconstruct P for coulomb function
                            P = np.array([P_0, P_1, P_2], dtype=np.float64)
                            
                            val += tempcoeff5 * coulomb_rys_3c2e(
                                roots, weights, G, PQsq, rho, norder, n, m,
                                la, lb, lc, 0, ma, mb, mc, 0, na, nb, nc, 0,
                                alphaik_, alphajk_, alphakk_, alphalk,
                                I, J, K, L, P
                            )
            
            # Store results
            if tot_ang <= 1:
                threeC2E[offsets[itemp] + index_k] = val
                index_k += 1
            elif tot_ang == 2:
                d_buffer[d_idx] = val
                d_idx += 1
                if d_idx == 6:
                    offset_base = offsets[itemp] + index_k
                    threeC2E[offset_base] = 2.0/3.0 * d_buffer[0] - 1.0/3.0 * (d_buffer[3] + d_buffer[5])
                    threeC2E[offset_base + 1] = d_buffer[1]
                    threeC2E[offset_base + 2] = d_buffer[2]
                    threeC2E[offset_base + 3] = 2.0/3.0 * d_buffer[3] - 1.0/3.0 * (d_buffer[0] + d_buffer[5])
                    threeC2E[offset_base + 4] = d_buffer[4]
                    threeC2E[offset_base + 5] = 2.0/3.0 * d_buffer[5] - 1.0/3.0 * (d_buffer[0] + d_buffer[3])
                    index_k += 6
            elif tot_ang == 3:
                f_buffer[f_idx] = val
                f_idx += 1
                if f_idx == 10:
                    fast_transform_f(threeC2E, f_buffer, offsets[itemp] + index_k)
                    index_k += 10
            elif tot_ang == 4:
                g_buffer[g_idx] = val
                g_idx += 1
                if g_idx == 15:
                    fast_transform_g(threeC2E, g_buffer, offsets[itemp] + index_k)
                    index_k += 15
            
            k += 1
    return threeC2E


@njit(parallel=True, cache=True, fastmath=True, error_model="numpy", nogil=True, boundscheck=False)
def rys_3c2e_tri_schwarz_sparse_algo10_sao_internal(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts,aux_bfs_coords, aux_bfs_contr_prim_norms, aux_bfs_lmn, aux_bfs_nprim, aux_bfs_coeffs, aux_bfs_prim_norms, aux_bfs_expnts, indicesA, indicesB, offsets, nao, naux, IJsq_arr, shell_indices, aux_shell_indices, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold, strict_schwarz, nsignificant):
    # Calculates 3c2e integrals based on a provided list of significant triplets determined using Schwarz inequality
    # It is assumed that the provided list was made with triangular int3c2e in mind
    # TODO: Check out these screening methods as well 
    # https://kops.uni-konstanz.de/server/api/core/bitstreams/79ada61a-fd29-43fd-a298-79c1696a0601/content
    # https://aip.scitation.org/doi/10.1063/1.4917519


    threeC2E = np.zeros((nsignificant), dtype=np.float64) 
    
    a = 10/19
    b = 14/19
    c = 5/19
    d = 3*np.sqrt(5)/19
    M_f = np.array((
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
        ))

    M_g = np.array((
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
        ))

    #Debug:
    # print(config.THREADING_LAYER)
    # print(threading_layer())
    # print(get_parallel_chunksize())
    # print(ntriplets)
    # set_parallel_chunksize(ntriplets//128)
    # print(get_parallel_chunksize())
        
    pi = 3.141592653589793
    # pisq = 9.869604401089358  #PI^2
    twopisq = 19.739208802178716  #2*PI^2
    L = np.zeros((3))
    alphalk = 0.0
    ld, md, nd = int(0), int(0), int(0)

    # Initialize before hand to avoid contention of memory allocator
    # roots = np.zeros((5)) # 5 is the maximum possible value currently in the code
    # weights = np.zeros((5))
    # G = np.zeros((21,21)) # 11 should be enough

    
    maxprims = bfs_coeffs.shape[1]
    maxprims_aux = aux_bfs_coeffs.shape[1]

    # Create arrays to avoid contention of allocator 
    # (https://stackoverflow.com/questions/70339388/using-numba-with-np-concatenate-is-not-efficient-in-parallel/70342014#70342014)
    # chunk_size = 20 
    # for start in range(0, len(indicesA), chunk_size):
    #     end = min(start + chunk_size, len(indicesA))

    #     idxA = indicesA[start:end]
    #     idxB = indicesB[start:end]
    #     offs = offsets[start:end] - offsets[start]

        # sub_nsig = offs[-1] + (offsets[start+1] - offsets[start])

    #Loop over BFs
    for itemp in prange(indicesA.shape[0]):
        # id_thrd = get_thread_id()
        
        i = indicesA[itemp]
        j = indicesB[itemp]
        
        sqrt_ij = sqrt_ints4c2e_diag[i,j] 

        if strict_schwarz:
            if sqrt_ij*sqrt_ij<1e-13:
                continue  

        # ishell = shell_indices[i]
        
        Ni = bfs_contr_prim_norms[i]
        nprimi = bfs_nprim[i]
        alphaik = np.zeros(maxprims, dtype=np.float64) # Should be Hoisted out
        alphaik[:] = bfs_expnts[i,:]
        lmni = bfs_lmn[i]
        la, ma, na = lmni
        I = bfs_coords[i]

        # jshell = shell_indices[j]
        
        Nj = bfs_contr_prim_norms[j]
        nprimj = bfs_nprim[j]
        alphajk = np.zeros(maxprims, dtype=np.float64) # Should be Hoisted out
        alphajk[:] = bfs_expnts[j,:]
        lmnj = bfs_lmn[j]
        lb, mb, nb = lmnj
        J = bfs_coords[j]

        
        IJsq = IJsq_arr[i,j]
        tempcoeff1 = Ni*Nj

        ## This screening is not useful for def2-SVP kind of basis sets
        ## But useful for diffuse basis sets like def2-TZVP
        alphaik_min = np.min(alphaik[:nprimi])
        alphajk_min = np.min(alphajk[:nprimj])
        gammaP_min = alphaik_min + alphajk_min
        rho_min = alphaik_min * alphajk_min / gammaP_min
        arg = rho_min * IJsq
        if arg > 18.42:  # exp(-18.42) ≈ 1e-8
            continue
        # screening_AB_min = np.exp(-rho_min * IJsq)
        # if abs(screening_AB_min)<1.0e-8:
        #     continue
        # P_min = (alphaik_min*I + alphajk_min*J)/gammaP_min
        index_k = 0
        k = 0
        d_buffer = np.zeros(6, dtype=np.float64)
        f_buffer = np.zeros(10, dtype=np.float64)
        g_buffer = np.zeros(15, dtype=np.float64)
        roots = np.zeros((10)) 
        weights = np.zeros((10)) 
        G = np.zeros((20, 20)) 
        while k < naux:
            lmnk = aux_bfs_lmn[k]
            lc, mc, nc = lmnk
            tot_ang = lc + mc + nc
            is_first = (lmnk[0] != 0) and (lmnk.sum() == lmnk[0])
            if is_first:
                if sqrt_ij*sqrt_diag_ints2c2e[k]<threshold:
                    if tot_ang == 0:
                        k += 1
                    elif tot_ang == 1:
                        k += 3
                    elif tot_ang == 2:
                        k += 6
                    elif tot_ang == 3:
                        k += 10
                    elif tot_ang == 4:
                        k += 15
                    continue
            # if tot_ang==2 and is_first:
            #     d_buffer = np.zeros(6, dtype=np.float64)
            #     d_idx = 0
            if is_first:
                if tot_ang == 2: # d functions
                    # d_buffer = np.zeros(6, dtype=np.float64)
                    d_buffer[:] = 0.0
                    d_idx = 0
                if tot_ang == 3: # f functions
                    # f_buffer = np.zeros(10, dtype=np.float64)
                    f_buffer[:] = 0.0
                    f_idx = 0
                if tot_ang == 4: # g functions
                    # g_buffer = np.zeros(15, dtype=np.float64)
                    g_buffer[:] = 0.0
                    g_idx = 0
            
            # lmnk = aux_bfs_lmn[k]
            # lc, mc, nc = lmnk
            # if strict_schwarz:
            #     max_val = sqrt_ij*sqrt_diag_ints2c2e[k]
            #     if max_val>threshold:
            #         if max_val<1e-8:
            #             if (lc+mc+nc)>=1: # s aux function
            #                 continue
            #         elif max_val<1e-7:
            #             if (lc+mc+nc)>=2: # s, p aux functions
            #                 continue
            #         elif max_val<1e-6:
            #             if (lc+mc+nc)>=3: # s, p, d aux functions
            #                 continue
            #     else:
            #         continue
                    
            # else:  
            #     if sqrt_ij*sqrt_diag_ints2c2e[k]<threshold:
            #         continue
            
            Nk = aux_bfs_contr_prim_norms[k]
            nprimk = aux_bfs_nprim[k]
            alphakk = np.zeros(maxprims_aux, dtype=np.float64) # Should be Hoisted out
            alphakk[:] = aux_bfs_expnts[k,:]
            
            
            K = aux_bfs_coords[k]
            Q = K   

            # NEW SCREENING
            # PQ_min = P_min - Q
            # PQsq_min = np.sum(PQ_min**2)

            # alphakk_min = np.min(alphakk[:nprimk])
            # rho_abP_min = gammaP_min * alphakk_min / (gammaP_min + alphakk_min)
            # screening_ABP_min = np.exp(-rho_abP_min * PQsq_min)
            # print('screening_ABP_min:', screening_ABP_min)
            # if rho_abP_min*PQsq_min > 1300.0:  # Try even lower threshold
            #     index_k += 1
            #     continue
            # if abs(screening_ABP_min)<1.0e-8:
            #     index_k += 1
            #     continue
            # geometric prescreen ONLY
            # if rho_abP_min * PQsq_min > np.log(1.0 / 1e-6):
            #     continue
                
            tempcoeff2 = tempcoeff1*Nk
            norder = int((la+ma+na+lb+mb+nb+lc+mc+nc+ld+md+nd)/2 + 1 ) 
            tempcoeff3 = np.zeros(maxprims, dtype=np.float64) # Should be Hoisted out 
            tempcoeff3[:] = tempcoeff2*bfs_coeffs[i,:]*bfs_prim_norms[i,:]
            # PQsq = np.zeros((bfs_coeffs.shape[1], bfs_coeffs.shape[1], aux_bfs_coeffs.shape[1]), dtype=np.float64)
            # rho = np.zeros((bfs_coeffs.shape[1], bfs_coeffs.shape[1], aux_bfs_coeffs.shape[1]), dtype=np.float64)
                
            val = 0.0
            if norder<=10: # Use rys quadrature
                
                n = int(max(la+lb,ma+mb,na+nb))
                m = int(max(lc+ld,mc+md,nc+nd))
                # roots = np.zeros((norder)) 
                # weights = np.zeros((norder)) 
                # G = np.zeros((n+1,m+1))
                
                    
                #Loop over primitives
                for ik in range(nprimi): 
                    alphaik_ = alphaik[ik]
                    tempcoeff3_ = tempcoeff3[ik]
                    for jk in range(nprimj):
                        alphajk_ = alphajk[jk]
                        # gammaP[ik,jk] = alphaik_ + alphajk_
                        gammaP_ = alphaik_ + alphajk_
                        # gamma_inv = 1/gammaP_
                        arg = alphaik_*alphajk_/gammaP_*IJsq
                        if arg > 18.42:  # exp(-18.42) ≈ 1e-8
                            continue
                        # screenfactorAB = np.exp(-alphaik_*alphajk_/gammaP_*IJsq)
                            
                        # if abs(screenfactorAB)<1.0e-8:   
                            #Although this value of screening threshold seems very large
                            # it actually gives the best consistency with PySCF. Reducing it to 1e-15,
                            # actually worsened the agreement.
                            # I suspect that this is caused due to an error cancellation
                            # that happens with the nucmat calculation, as the same screening is 
                            # used there as well
                            # continue
                        P = (alphaik_*I + alphajk_*J)/gammaP_
                        PQ = P - Q
                        PQsq = np.sum(PQ**2)
                        djk = bfs_coeffs[j,jk] 
                        Njk = bfs_prim_norms[j,jk]     

                        tempcoeff4 = tempcoeff3_*djk*Njk  

                        # screenfactor_2 = tempcoeff4/Nk*screenfactorAB*gamma_inv*np.sqrt(gamma_inv)*15.5031383401
                        # if abs(screenfactor_2)<1.0e-9: # The threshold used here should be the same as Schwarz screening threshold
                        #     continue 
                        
                        # gammaP_ = gammaP[ik,jk]
                            
                        for kk in range(nprimk):

                            dkk = aux_bfs_coeffs[k,kk]
                            Nkk = aux_bfs_prim_norms[k,kk]
                            tempcoeff5 = tempcoeff4*dkk*Nkk 
                                
                                
                            gammaQ = alphakk[kk]
                            rho = gammaP_*gammaQ/(gammaP_ + gammaQ)

                            # NEW SCREENING
                            # More aggressive early exit
                            # if rho*PQsq > 15.0:  # Try even lower threshold
                            #     continue

                            # screenfactorAB_P = np.exp(-rho*PQsq)
                            # if abs(screenfactorAB_P) < 1.0e-8: 
                            #     continue

                            # # Additional check: if the combined screening is too small
                            # combined_screening = screenfactorAB * screenfactorAB_P
                            # if abs(combined_screening) < 1.0e-10:
                            #     continue
                            
                            # ABsrt = np.sqrt(gammaP[ik,jk]*alphakk[kk])
                            # X = PQsq*rho
                            # factor = 2*np.sqrt(rho/pi)
                            # print('s')
                            val += tempcoeff5*coulomb_rys_3c2e(roots,weights,G,PQsq, rho, norder,n,m,la,lb,lc,ld,ma,mb,mc,md,na,nb,nc,nd,alphaik_, alphajk_,alphakk[kk],alphalk,I,J,K,L,P)
                            # val += tempcoeff5*coulomb_rys_3c2e_new(roots,weights,G,PQsq, rho, norder,n,m,la,lb,lc,ld,ma,mb,mc,md,na,nb,nc,nd,alphaik_, alphajk_,alphakk[kk],alphalk,I,J,K,L,P)
                            
            # threeC2E[offsets[itemp] + index_k] = val
            if tot_ang == 0 or tot_ang == 1: # s or p functions
                threeC2E[offsets[itemp] + index_k] = val
                index_k += 1
            if tot_ang == 2: # d functions
                d_buffer[d_idx] = val
                d_idx += 1
                if d_idx==6:
                    threeC2E[offsets[itemp] + index_k] = 2/3 * d_buffer[0] - 1/3 * (d_buffer[3] + d_buffer[5])
                    threeC2E[offsets[itemp] + index_k + 1] = d_buffer[1]
                    threeC2E[offsets[itemp] + index_k + 2] = d_buffer[2]
                    threeC2E[offsets[itemp] + index_k + 3] = 2/3 * d_buffer[3] - 1/3 * (d_buffer[0] + d_buffer[5])
                    threeC2E[offsets[itemp] + index_k + 4] = d_buffer[4]
                    threeC2E[offsets[itemp] + index_k + 5] = 2/3 * d_buffer[5] - 1/3 * (d_buffer[0] + d_buffer[3])
                    index_k += 6
            if tot_ang == 3: # f functions
                f_buffer[f_idx] = val
                f_idx += 1
                if f_idx==10:

                    
                    
                    threeC2E[offsets[itemp] + index_k : offsets[itemp] + index_k + 10] = M_f @ f_buffer[:]
                    index_k += 10
            if tot_ang == 4: # g functions
                g_buffer[g_idx] = val
                g_idx += 1
                if g_idx==15:
                    
                    
                    threeC2E[offsets[itemp] + index_k : offsets[itemp] + index_k + 15] = M_g @ g_buffer[:]
                    index_k += 15
            k += 1
            
            
                                 

    return threeC2E


@njit(parallel=True, cache=True, fastmath=True, error_model="numpy", nogil=True, boundscheck=False)
def rys_3c2e_tri_schwarz_sparse_algo10_internal(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts,aux_bfs_coords, aux_bfs_contr_prim_norms, aux_bfs_lmn, aux_bfs_nprim, aux_bfs_coeffs, aux_bfs_prim_norms, aux_bfs_expnts, indicesA, indicesB, offsets, nao, naux, IJsq_arr, shell_indices, aux_shell_indices, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold, strict_schwarz, nsignificant):
    # Calculates 3c2e integrals based on a provided list of significant triplets determined using Schwarz inequality
    # It is assumed that the provided list was made with triangular int3c2e in mind
    # TODO: Check out these screening methods as well 
    # https://kops.uni-konstanz.de/server/api/core/bitstreams/79ada61a-fd29-43fd-a298-79c1696a0601/content
    # https://aip.scitation.org/doi/10.1063/1.4917519

    threeC2E = np.zeros((nsignificant), dtype=np.float64) 

    #Debug:
    # print(config.THREADING_LAYER)
    # print(threading_layer())
    # print(get_parallel_chunksize())
    # print(ntriplets)
    # set_parallel_chunksize(ntriplets//128)
    # print(get_parallel_chunksize())
        
    pi = 3.141592653589793
    # pisq = 9.869604401089358  #PI^2
    twopisq = 19.739208802178716  #2*PI^2
    L = np.zeros((3))
    alphalk = 0.0
    ld, md, nd = int(0), int(0), int(0)

    # Initialize before hand to avoid contention of memory allocator
    # roots = np.zeros((5)) # 5 is the maximum possible value currently in the code
    # weights = np.zeros((5))
    # G = np.zeros((21,21)) # 11 should be enough

    
    maxprims = bfs_coeffs.shape[1]
    maxprims_aux = aux_bfs_coeffs.shape[1]

    # Create arrays to avoid contention of allocator 
    # (https://stackoverflow.com/questions/70339388/using-numba-with-np-concatenate-is-not-efficient-in-parallel/70342014#70342014)
    

    #Loop over BFs
    for itemp in prange(indicesA.shape[0]):
        # id_thrd = get_thread_id()
        
        i = indicesA[itemp]
        j = indicesB[itemp]
        
        sqrt_ij = sqrt_ints4c2e_diag[i,j] 

        if strict_schwarz:
            if sqrt_ij*sqrt_ij<1e-13:
                continue  
        # roots = np.zeros((10)) 
        # weights = np.zeros((10))  
        # G = np.zeros((20, 20)) 
        # ishell = shell_indices[i]
        
        Ni = bfs_contr_prim_norms[i]
        nprimi = bfs_nprim[i]
        alphaik = np.zeros(maxprims, dtype=np.float64) # Should be Hoisted out
        alphaik[:] = bfs_expnts[i,:]
        lmni = bfs_lmn[i]
        la, ma, na = lmni
        I = bfs_coords[i]

        # jshell = shell_indices[j]

        
        Nj = bfs_contr_prim_norms[j]
        nprimj = bfs_nprim[j]
        alphajk = np.zeros(maxprims, dtype=np.float64) # Should be Hoisted out
        alphajk[:] = bfs_expnts[j,:]
        lmnj = bfs_lmn[j]
        lb, mb, nb = lmnj
        J = bfs_coords[j]

        
        IJsq = IJsq_arr[i,j]
        tempcoeff1 = Ni*Nj

        ## This screening is not useful for def2-SVP kind of basis sets
        ## But useful for diffuse basis sets like def2-TZVP
        alphaik_min = np.min(alphaik[:nprimi])
        alphajk_min = np.min(alphajk[:nprimj])
        gammaP_min = alphaik_min + alphajk_min
        rho_min = alphaik_min * alphajk_min / gammaP_min
        arg = rho_min * IJsq
        if arg > 18.42:  # exp(-18.42) ≈ 1e-8
            continue
        # screening_AB_min = np.exp(-rho_min * IJsq)
        # if abs(screening_AB_min)<1.0e-8:
        #     continue
        # P_min = (alphaik_min*I + alphajk_min*J)/gammaP_min
        index_k = 0
        for k in range(naux):
            if sqrt_ij*sqrt_diag_ints2c2e[k]<threshold:
                continue
            lmnk = aux_bfs_lmn[k]
            lc, mc, nc = lmnk
            # if strict_schwarz:
            #     max_val = sqrt_ij*sqrt_diag_ints2c2e[k]
            #     if max_val>threshold:
            #         if max_val<1e-8:
            #             if (lc+mc+nc)>=1: # s aux function
            #                 continue
            #         elif max_val<1e-7:
            #             if (lc+mc+nc)>=2: # s, p aux functions
            #                 continue
            #         elif max_val<1e-6:
            #             if (lc+mc+nc)>=3: # s, p, d aux functions
            #                 continue
            #     else:
            #         continue
                    
            # else:  
            #     if sqrt_ij*sqrt_diag_ints2c2e[k]<threshold:
            #         continue
            
            Nk = aux_bfs_contr_prim_norms[k]
            nprimk = aux_bfs_nprim[k]
            alphakk = np.zeros(maxprims_aux, dtype=np.float64) # Should be Hoisted out
            alphakk[:] = aux_bfs_expnts[k,:]
            
            
            K = aux_bfs_coords[k]
            Q = K   

            # NEW SCREENING
            # PQ_min = P_min - Q
            # PQsq_min = np.sum(PQ_min**2)

            # alphakk_min = np.min(alphakk[:nprimk])
            # rho_abP_min = gammaP_min * alphakk_min / (gammaP_min + alphakk_min)
            # screening_ABP_min = np.exp(-rho_abP_min * PQsq_min)
            # print('screening_ABP_min:', screening_ABP_min)
            # if rho_abP_min*PQsq_min > 1300.0:  # Try even lower threshold
            #     index_k += 1
            #     continue
            # if abs(screening_ABP_min)<1.0e-8:
            #     index_k += 1
            #     continue
            # geometric prescreen ONLY
            # if rho_abP_min * PQsq_min > np.log(1.0 / 1e-6):
            #     continue
                
            tempcoeff2 = tempcoeff1*Nk
            norder = int((la+ma+na+lb+mb+nb+lc+mc+nc+ld+md+nd)/2 + 1 ) 
            tempcoeff3 = np.zeros(maxprims, dtype=np.float64) # Should be Hoisted out 
            tempcoeff3[:] = tempcoeff2*bfs_coeffs[i,:]*bfs_prim_norms[i,:]
            # PQsq = np.zeros((bfs_coeffs.shape[1], bfs_coeffs.shape[1], aux_bfs_coeffs.shape[1]), dtype=np.float64)
            # rho = np.zeros((bfs_coeffs.shape[1], bfs_coeffs.shape[1], aux_bfs_coeffs.shape[1]), dtype=np.float64)
                
            val = 0.0
            if norder<=10: # Use rys quadrature
                
                n = int(max(la+lb,ma+mb,na+nb))
                m = int(max(lc+ld,mc+md,nc+nd))
                roots = np.zeros((norder)) 
                weights = np.zeros((norder)) 
                G = np.zeros((n+1,m+1))
                
                
                    
                #Loop over primitives
                for ik in range(nprimi): 
                    alphaik_ = alphaik[ik]
                    tempcoeff3_ = tempcoeff3[ik]
                    for jk in range(nprimj):
                        alphajk_ = alphajk[jk]
                        # gammaP[ik,jk] = alphaik_ + alphajk_
                        gammaP_ = alphaik_ + alphajk_
                        # gamma_inv = 1/gammaP_
                        arg = alphaik_*alphajk_/gammaP_*IJsq
                        if arg > 18.42:  # exp(-18.42) ≈ 1e-8
                            continue
                        # screenfactorAB = np.exp(-alphaik_*alphajk_/gammaP_*IJsq)
                            
                        # if abs(screenfactorAB)<1.0e-8:   
                            #Although this value of screening threshold seems very large
                            # it actually gives the best consistency with PySCF. Reducing it to 1e-15,
                            # actually worsened the agreement.
                            # I suspect that this is caused due to an error cancellation
                            # that happens with the nucmat calculation, as the same screening is 
                            # used there as well
                            # continue
                        P = (alphaik_*I + alphajk_*J)/gammaP_
                        PQ = P - Q
                        PQsq = np.sum(PQ**2)
                        djk = bfs_coeffs[j,jk] 
                        Njk = bfs_prim_norms[j,jk]     

                        tempcoeff4 = tempcoeff3_*djk*Njk  

                        # screenfactor_2 = tempcoeff4/Nk*screenfactorAB*gamma_inv*np.sqrt(gamma_inv)*15.5031383401
                        # if abs(screenfactor_2)<1.0e-9: # The threshold used here should be the same as Schwarz screening threshold
                        #     continue 
                        
                        # gammaP_ = gammaP[ik,jk]
                            
                        for kk in range(nprimk):

                            dkk = aux_bfs_coeffs[k,kk]
                            Nkk = aux_bfs_prim_norms[k,kk]
                            tempcoeff5 = tempcoeff4*dkk*Nkk 
                                
                                
                            gammaQ = alphakk[kk]
                            rho = gammaP_*gammaQ/(gammaP_ + gammaQ)

                            # NEW SCREENING
                            # More aggressive early exit
                            # if rho*PQsq > 15.0:  # Try even lower threshold
                            #     continue

                            # screenfactorAB_P = np.exp(-rho*PQsq)
                            # if abs(screenfactorAB_P) < 1.0e-8: 
                            #     continue

                            # # Additional check: if the combined screening is too small
                            # combined_screening = screenfactorAB * screenfactorAB_P
                            # if abs(combined_screening) < 1.0e-10:
                            #     continue
                            
                            # ABsrt = np.sqrt(gammaP[ik,jk]*alphakk[kk])
                            # X = PQsq*rho
                            # factor = 2*np.sqrt(rho/pi)
                            val += tempcoeff5*coulomb_rys_3c2e(roots,weights,G,PQsq, rho, norder,n,m,la,lb,lc,ld,ma,mb,mc,md,na,nb,nc,nd,alphaik_, alphajk_,alphakk[kk],alphalk,I,J,K,L,P)
                            # val += tempcoeff5*coulomb_rys_3c2e_new(roots,weights,G,PQsq, rho, norder,n,m,la,lb,lc,ld,ma,mb,mc,md,na,nb,nc,nd,alphaik_, alphajk_,alphakk[kk],alphalk,I,J,K,L,P)
                            
                            
            
            threeC2E[offsets[itemp] + index_k] = val
            index_k += 1
                                 

    return threeC2E

@njit(parallel=True, cache=True, fastmath=True, error_model="numpy", nogil=True, boundscheck=False)
def rys_3c2e_tri_schwarz_sparse_internal_old(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts,aux_bfs_coords, aux_bfs_contr_prim_norms, aux_bfs_lmn, aux_bfs_nprim, aux_bfs_coeffs, aux_bfs_prim_norms, aux_bfs_expnts, indicesA, indicesB, indicesC, nao, naux, IJsq_arr, shell_indices, aux_shell_indices):
    # Calculates 3c2e integrals based on a provided list of significant triplets determined using Schwarz inequality
    # It is assumed that the provided list was made with triangular int3c2e in mind
    # TODO: Check out these screening methods as well 
    # https://kops.uni-konstanz.de/server/api/core/bitstreams/79ada61a-fd29-43fd-a298-79c1696a0601/content
    # https://aip.scitation.org/doi/10.1063/1.4917519

    ntriplets = indicesA.shape[0] # No. of significant triplets
    threeC2E = np.zeros((ntriplets), dtype=np.float64) 

    #Debug:
    # print(config.THREADING_LAYER)
    # print(threading_layer())
    # print(get_parallel_chunksize())
    # print(ntriplets)
    # set_parallel_chunksize(ntriplets//128)
    # print(get_parallel_chunksize())
        
    pi = 3.141592653589793
    # pisq = 9.869604401089358  #PI^2
    twopisq = 19.739208802178716  #2*PI^2
    L = np.zeros((3))
    alphalk = 0.0
    ld, md, nd = int(0), int(0), int(0)

    # Initialize before hand to avoid contention of memory allocator
    # roots = np.zeros((5)) # 5 is the maximum possible value currently in the code
    # weights = np.zeros((5))
    # G = np.zeros((21,21)) # 11 should be enough

    
    maxprims = bfs_coeffs.shape[1]
    maxprims_aux = aux_bfs_coeffs.shape[1]

    # Create arrays to avoid contention of allocator 
    # (https://stackoverflow.com/questions/70339388/using-numba-with-np-concatenate-is-not-efficient-in-parallel/70342014#70342014)

    # ncores = get_num_threads()
    # ishell_previous = np.zeros((ncores), dtype=np.uint16)
    # jshell_previous = np.zeros((ncores), dtype=np.uint16)
    # kshell_previous = np.zeros((ncores), dtype=np.uint16)
    # # The values don't have any significance. I just hope that the first triplet doesn't coincidentally have these values
    # ishell_previous[:] = 999
    # jshell_previous[:] = 918
    # kshell_previous[:] = 202
    
    ishell_previous = -1
    jshell_previous = -1
    kshell_previous = -1

    #Loop over BFs
    for itemp in prange(ntriplets):
        # id_thrd = get_thread_id()

        

        i = indicesA[itemp]
        j = indicesB[itemp]
        k = indicesC[itemp]

        
        # i_previous = indicesA[itemp-1]
        # j_previous = indicesB[itemp-1]
        # k_previous = indicesC[itemp-1]

        # ishell_previous = shell_indices[i_previous]
        # jshell_previous = shell_indices[j_previous]
        # kshell_previous = shell_indices[k_previous]
        # if itemp!=0:
        #     i_previous = indicesA[itemp-1]
        #     j_previous = indicesB[itemp-1]
        #     k_previous = indicesC[itemp-1]

        #     ishell_previous = shell_indices[i_previous]
        #     jshell_previous = shell_indices[j_previous]
        #     kshell_previous = shell_indices[k_previous]
        # else:
        #     ishell_previous = -1
        #     jshell_previous = -1
        #     kshell_previous = -1
        
        

        ishell = shell_indices[i]
        if ishell!=ishell_previous:
            Ni = bfs_contr_prim_norms[i]
            nprimi = bfs_nprim[i]
            alphaik = np.zeros(bfs_coeffs.shape[1], dtype=np.float64) # Should be Hoisted out
            alphaik[:] = bfs_expnts[i,:]
        lmni = bfs_lmn[i]
        la, ma, na = lmni
        I = bfs_coords[i]

        jshell = shell_indices[j]
        if jshell!=jshell_previous:
            Nj = bfs_contr_prim_norms[j]
            nprimj = bfs_nprim[j]
            alphajk = np.zeros(bfs_coeffs.shape[1], dtype=np.float64) # Should be Hoisted out
            alphajk[:] = bfs_expnts[j,:]
        lmnj = bfs_lmn[j]
        lb, mb, nb = lmnj
        J = bfs_coords[j]

        if ishell!=ishell_previous or jshell!=jshell_previous:
            IJsq = IJsq_arr[i,j]
            tempcoeff1 = Ni*Nj
            gammaP = np.zeros((bfs_coeffs.shape[1], bfs_coeffs.shape[1]), dtype=np.float64) # Should be Hoisted out
            screenfactorAB = np.zeros((bfs_coeffs.shape[1], bfs_coeffs.shape[1]), dtype=np.float64) # Should be Hoisted out
            # Ap = np.zeros((bfs_coeffs.shape[1], bfs_coeffs.shape[1]), dtype=np.float64) # Should be Hoisted out
        
        kshell = aux_shell_indices[k]
        if kshell!=kshell_previous:
            Nk = aux_bfs_contr_prim_norms[k]
            nprimk = aux_bfs_nprim[k]
            alphakk = np.zeros(aux_bfs_coeffs.shape[1], dtype=np.float64) # Should be Hoisted out
            alphakk[:] = aux_bfs_expnts[k,:]
        lmnk = aux_bfs_lmn[k]
        lc, mc, nc = lmnk
        K = aux_bfs_coords[k]
        Q = K   
                
        if ishell!=ishell_previous or jshell!=jshell_previous or kshell!=kshell_previous:
            tempcoeff2 = tempcoeff1*Nk
            norder = int((la+ma+na+lb+mb+nb+lc+mc+nc+ld+md+nd)/2 + 1 ) 
            tempcoeff3 = np.zeros(bfs_coeffs.shape[1], dtype=np.float64) # Should be Hoisted out 
            tempcoeff3[:] = tempcoeff2*bfs_coeffs[i,:]*bfs_prim_norms[i,:]
            # PQsq = np.zeros((bfs_coeffs.shape[1], bfs_coeffs.shape[1], aux_bfs_coeffs.shape[1]), dtype=np.float64)
            # rho = np.zeros((bfs_coeffs.shape[1], bfs_coeffs.shape[1], aux_bfs_coeffs.shape[1]), dtype=np.float64)
            
            
            
        val = 0.0
        if norder<=10: # Use rys quadrature
            if ishell!=ishell_previous or jshell!=jshell_previous or kshell!=kshell_previous:
                n = int(max(la+lb,ma+mb,na+nb))
                m = int(max(lc+ld,mc+md,nc+nd))
                roots = np.zeros((norder)) 
                weights = np.zeros((norder)) 
                # roots = np.zeros((5)) 
                # weights = np.zeros((5)) 
                G = np.zeros((n+1,m+1)) 
            
                
            #Loop over primitives
            for ik in range(nprimi): 
                for jk in range(nprimj):
                    if ishell!=ishell_previous or jshell!=jshell_previous:
                        gammaP[ik,jk] = alphaik[ik] + alphajk[jk]
                        # Ap[ik,jk] = alphaik[ik]*alphajk[jk]
                        # screenfactorAB[ik,jk] = np.exp(-Ap[ik,jk]/gammaP[ik,jk]*IJsq)
                        screenfactorAB[ik,jk] = np.exp(-alphaik[ik]*alphajk[jk]/gammaP[ik,jk]*IJsq)
                        
                    if abs(screenfactorAB[ik,jk])<1.0e-8:   
                        #Although this value of screening threshold seems very large
                        # it actually gives the best consistency with PySCF. Reducing it to 1e-15,
                        # actually worsened the agreement.
                        # I suspect that this is caused due to an error cancellation
                        # that happens with the nucmat calculation, as the same screening is 
                        # used there as well
                        continue
                    djk = bfs_coeffs[j,jk] 
                    Njk = bfs_prim_norms[j,jk]     
                    P = (alphaik[ik]*I + alphajk[jk]*J)/gammaP[ik,jk]
                    PQ = P - Q
                    PQsq = np.sum(PQ**2)
                    tempcoeff4 = tempcoeff3[ik]*djk*Njk  
                    
                        
                    for kk in range(nprimk):
                        dkk = aux_bfs_coeffs[k,kk]
                        Nkk = aux_bfs_prim_norms[k,kk]
                        tempcoeff5 = tempcoeff4*dkk*Nkk 
                              
                            
                        gammaQ = alphakk[kk]
                        rho = gammaP[ik,jk]*gammaQ/(gammaP[ik,jk]+gammaQ)
                                
                        # ABsrt = np.sqrt(gammaP[ik,jk]*alphakk[kk])
                        # X = PQsq*rho
                        # factor = 2*np.sqrt(rho/pi)
                                
                                  
                        val += tempcoeff5*coulomb_rys(roots,weights,G,PQsq, rho, norder,n,m,la,lb,lc,ld,ma,mb,mc,md,na,nb,nc,nd,alphaik[ik], alphajk[jk],alphakk[kk],alphalk,I,J,K,L)
                        # The following should have been faster but isnt
                        # val += tempcoeff5*coulomb_rys_fast(roots,weights,G,norder,la,lb,lc,ld,ma,mb,mc,md,na,nb,nc,nd,alphaik[ik], alphajk[jk], alphakk[kk], alphalk,I,J,K,L,X,gammaP[ik,jk],alphakk[kk],Ap[ik,jk],0.0,ABsrt,factor,P,Q)
                            
        else: # Analytical (Conventional)
            #Loop over primitives
            for ik in range(nprimi):   
                for jk in range(nprimj):
                    if ishell!=ishell_previous or jshell!=jshell_previous:
                        gammaP[ik,jk] = alphaik[ik] + alphajk[jk]
                        # Ap[ik,jk] = alphaik[ik]*alphajk[jk]
                        # screenfactorAB[ik,jk] = np.exp(-Ap[ik,jk]/gammaP[ik,jk]*IJsq)
                        screenfactorAB[ik,jk] = np.exp(-alphaik[ik]*alphajk[jk]/gammaP[ik,jk]*IJsq)
                        
                    if abs(screenfactorAB[ik,jk])<1.0e-8:   
                        #Although this value of screening threshold seems very large
                        # it actually gives the best consistency with PySCF. Reducing it to 1e-15,
                        # actually worsened the agreement.
                        # I suspect that this is caused due to an error cancellation
                        # that happens with the nucmat calculation, as the same screening is 
                        # used there as well
                        continue
                    djk = bfs_coeffs[j,jk] 
                    Njk = bfs_prim_norms[j,jk]      
                    P = (alphaik[ik]*I + alphajk[jk]*J)/gammaP[ik,jk]
                    PI = P - I
                    PJ = P - J  
                    PQ = P - Q
                    fac1 = twopisq/gammaP[ik,jk]*screenfactorAB[ik,jk]  
                    onefourthgammaPinv = 0.25/gammaP[ik,jk]
                    tempcoeff4 = tempcoeff3[ik]*djk*Njk  
                        
                    for kk in range(nprimk):
                        dkk = aux_bfs_coeffs[k,kk]
                        Nkk = aux_bfs_prim_norms[k,kk]
                        tempcoeff5 = tempcoeff4*dkk*Nkk 
                              
                        
                        gammaQ = alphakk[kk] #+ alphalk
                        
                        
                                
                        QK = Q - K
                        QL = Q #- L
                        
                                  
                        fac2 = fac1/gammaQ
                                

                        omega = (fac2)*np.sqrt(pi/(gammaP[ik,jk] + gammaQ))
                        delta = 0.25*(1/gammaQ) + onefourthgammaPinv          
                        PQsqBy4delta = np.sum(PQ**2)/(4*delta)         
                                    
                        sum1 = innerLoop4c2e(la,lb,lc,ld,ma,mb,mc,md,na,nb,nc,nd,gammaP[ik,jk],gammaQ,PI,PJ,QK,QL,PQ,PQsqBy4delta,delta)
                                    
                        val += omega*sum1*tempcoeff5

        threeC2E[itemp] = val
        # ishell_previous = ishell
        # jshell_previous = jshell
        # kshell_previous = kshell 
                                    
                                 

    return threeC2E

@njit(parallel=False, cache=True, fastmath=True, error_model="numpy",nogil=True)
def rys_3c2e_tri_schwarz_sparse_internal2(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts,aux_bfs_coords, aux_bfs_contr_prim_norms, aux_bfs_lmn, aux_bfs_nprim, aux_bfs_coeffs, aux_bfs_prim_norms, aux_bfs_expnts, indicesA, indicesB, indicesC, nao, naux, IJsq_arr, ntriplets, threeC2E):
    # Calculates 3c2e integrals based on a provided list of significant triplets determined using Schwarz inequality
    # It is assumed that the provided list was made with triangular int3c2e in mind
    # TODO: Check out these screening methods as well 
    # https://kops.uni-konstanz.de/server/api/core/bitstreams/79ada61a-fd29-43fd-a298-79c1696a0601/content
    # https://aip.scitation.org/doi/10.1063/1.4917519

    ntriplets = indicesA.shape[0] # No. of significant triplets
    

    #Debug:
    # print(config.THREADING_LAYER)
    # print(threading_layer())
    # print(get_parallel_chunksize())
    # print(ntriplets)
    # set_parallel_chunksize(ntriplets//128)
    # print(get_parallel_chunksize())
        
    pi = 3.141592653589793
    # pisq = 9.869604401089358  #PI^2
    twopisq = 19.739208802178716  #2*PI^2
    L = np.zeros((3))
    alphalk = 0.0
    ld, md, nd = int(0), int(0), int(0)

    # Initialize before hand to avoid contention of memory allocator
    # roots = np.zeros((5)) # 5 is the maximum possible value currently in the code
    # weights = np.zeros((5))
    # G = np.zeros((21,21)) # 11 should be enough

    

    # Create arrays to avoid contention of allocator 
    # (https://stackoverflow.com/questions/70339388/using-numba-with-np-concatenate-is-not-efficient-in-parallel/70342014#70342014)

     
    
    #Loop over BFs
    for itemp in range(0,ntriplets): 
        i = indicesA[itemp]
        j = indicesB[itemp]
        k = indicesC[itemp]
        I = bfs_coords[i]
        Ni = bfs_contr_prim_norms[i]
        lmni = bfs_lmn[i]
        la, ma, na = lmni
        nprimi = bfs_nprim[i]

        J = bfs_coords[j]
        # IJ = I - J
        # IJsq = np.sum(IJ**2)
        # IJsq = np.dot(IJ,IJ)
        IJsq = IJsq_arr[i,j]
        Nj = bfs_contr_prim_norms[j]
        lmnj = bfs_lmn[j]
        lb, mb, nb = lmnj
        nprimj = bfs_nprim[j]
        tempcoeff1 = Ni*Nj
        
        K = aux_bfs_coords[k]
        Nk = aux_bfs_contr_prim_norms[k]
        lmnk = aux_bfs_lmn[k]
        lc, mc, nc = lmnk
        nprimk = aux_bfs_nprim[k]
        tempcoeff2 = tempcoeff1*Nk
                
               
                  
        # KL = K - L
        # KLsq = np.sum(K**2)
        # KLsq = np.dot(K,K)
                
                # npriml = bfs_nprim[l]

        norder = int((la+ma+na+lb+mb+nb+lc+mc+nc+ld+md+nd)/2 + 1 ) 
        val = 0.0
        if norder<=10: # Use rys quadrature
            n = int(max(la+lb,ma+mb,na+nb))
            m = int(max(lc+ld,mc+md,nc+nd))
            roots = np.zeros((norder))
            weights = np.zeros((norder))
            G = np.zeros((n+1,m+1))
            # G[:,:] = 0.0
            # roots[:] = 0.0
            # weights[:] = 0.0 

            
                
            #Loop over primitives
            for ik in range(nprimi):   
                dik = bfs_coeffs[i,ik]
                Nik = bfs_prim_norms[i,ik]
                alphaik = bfs_expnts[i,ik]
                tempcoeff3 = tempcoeff2*dik*Nik
                    
                for jk in range(nprimj):
                    alphajk = bfs_expnts[j,jk]
                    gammaP = alphaik + alphajk
                    screenfactorAB = np.exp(-alphaik*alphajk/gammaP*IJsq)
                    if abs(screenfactorAB)<1.0e-8:   
                        #Although this value of screening threshold seems very large
                        # it actually gives the best consistency with PySCF. Reducing it to 1e-15,
                        # actually worsened the agreement.
                        # I suspect that this is caused due to an error cancellation
                        # that happens with the nucmat calculation, as the same screening is 
                        # used there as well
                        continue
                    djk = bfs_coeffs[j,jk] 
                    Njk = bfs_prim_norms[j,jk]      
                    P = (alphaik*I + alphajk*J)/gammaP
                    tempcoeff4 = tempcoeff3*djk*Njk  
                        
                    for kk in range(nprimk):
                        dkk = aux_bfs_coeffs[k,kk]
                        Nkk = aux_bfs_prim_norms[k,kk]
                        alphakk = aux_bfs_expnts[k,kk]
                        tempcoeff5 = tempcoeff4*dkk*Nkk 
                              
                            
                        gammaQ = alphakk #+ alphalk
                        # screenfactorKL = np.exp(-alphakk*alphalk/gammaQ*KLsq)
                        # if abs(screenfactorKL)<1.0e-8:   
                        #     #Although this value of screening threshold seems very large
                        #     # it actually gives the best consistency with PySCF. Reducing it to 1e-15,
                        #     # actually worsened the agreement.
                        #     # I suspect that this is caused due to an error cancellation
                        #     # that happens with the nucmat calculation, as the same screening is 
                        #     # used there as well
                        #     continue
                        # # if abs(screenfactorAB*screenfactorKL)<1.0e-10:   
                        # #     #TODO: Check for optimal value for screening
                        # #     continue
                          
                        Q = K      
                        PQ = P - Q
                        PQsq = np.sum(PQ**2)
                        # PQsq = np.dot(PQ,PQ)
                        rho = gammaP*gammaQ/(gammaP+gammaQ)
                                
                                
                                
                                  
                        val += tempcoeff5*coulomb_rys(roots,weights,G,PQsq, rho, norder,n,m,la,lb,lc,ld,ma,mb,mc,md,na,nb,nc,nd,alphaik, alphajk, alphakk, alphalk,I,J,K,L)
                            
        else: # Analytical (Conventional)
            #Loop over primitives
            for ik in range(bfs_nprim[i]):   
                dik = bfs_coeffs[i,ik]
                Nik = bfs_prim_norms[i,ik]
                alphaik = bfs_expnts[i,ik]
                tempcoeff3 = tempcoeff2*dik*Nik
                    
                for jk in range(bfs_nprim[j]):
                    alphajk = bfs_expnts[j,jk]
                    gammaP = alphaik + alphajk
                    screenfactorAB = np.exp(-alphaik*alphajk/gammaP*IJsq)
                    if abs(screenfactorAB)<1.0e-8:   
                        #Although this value of screening threshold seems very large
                        # it actually gives the best consistency with PySCF. Reducing it to 1e-15,
                        # actually worsened the agreement.
                        # I suspect that this is caused due to an error cancellation
                        # that happens with the nucmat calculation, as the same screening is 
                        # used there as well
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
                        dkk = aux_bfs_coeffs[k,kk]
                        Nkk = aux_bfs_prim_norms[k,kk]
                        alphakk = aux_bfs_expnts[k,kk]
                        tempcoeff5 = tempcoeff4*dkk*Nkk 
                              
                        
                        gammaQ = alphakk #+ alphalk
                        # screenfactorKL = np.exp(-alphakk*alphalk/gammaQ*KLsq)
                        # if abs(screenfactorKL)<1.0e-8:   
                        #     #Although this value of screening threshold seems very large
                        #     # it actually gives the best consistency with PySCF. Reducing it to 1e-15,
                        #     # actually worsened the agreement.
                        #     # I suspect that this is caused due to an error cancellation
                        #     # that happens with the nucmat calculation, as the same screening is 
                        #     # used there as well
                        #     continue
                        
                        Q = K     
                        PQ = P - Q
                                
                        QK = Q - K
                        QL = Q #- L
                        
                                  
                        fac2 = fac1/gammaQ
                                

                        omega = (fac2)*np.sqrt(pi/(gammaP + gammaQ))
                        delta = 0.25*(1/gammaQ) + onefourthgammaPinv          
                        PQsqBy4delta = np.sum(PQ**2)/(4*delta)         
                                    
                        sum1 = innerLoop4c2e(la,lb,lc,ld,ma,mb,mc,md,na,nb,nc,nd,gammaP,gammaQ,PI,PJ,QK,QL,PQ,PQsqBy4delta,delta)
                                    
                        val += omega*sum1*tempcoeff5

        threeC2E[itemp] = val
                                    
                                   
                            
    #Debug:
    # print(config.THREADING_LAYER)
    # print(threading_layer())

    return None

@njit(parallel=True, cache=True, fastmath=True, error_model="numpy",nogil=True)
def rys_3c2e_tri_schwarz_sparse_algo9_internal(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts,aux_bfs_coords, aux_bfs_contr_prim_norms, aux_bfs_lmn, aux_bfs_nprim, aux_bfs_coeffs, aux_bfs_prim_norms, aux_bfs_expnts, offset, indicesB, indicesC, nao, naux, IJsq_arr, shell_indices, aux_shell_indices):
    # Calculates 3c2e integrals based on a provided list of significant triplets determined using Schwarz inequality
    # It is assumed that the provided list was made with triangular int3c2e in mind
    # TODO: Check out these screening methods as well 
    # https://kops.uni-konstanz.de/server/api/core/bitstreams/79ada61a-fd29-43fd-a298-79c1696a0601/content
    # https://aip.scitation.org/doi/10.1063/1.4917519

    ntriplets = indicesB.shape[0] # No. of significant triplets
    threeC2E = np.zeros((ntriplets), dtype=np.float64) 

    #Debug:
    # print(config.THREADING_LAYER)
    # print(threading_layer())
    # print(get_parallel_chunksize())
    # print(ntriplets)
    # set_parallel_chunksize(ntriplets//128)
    # print(get_parallel_chunksize())
        
    pi = 3.141592653589793
    twopisq = 19.739208802178716  #2*PI^2
    L = np.zeros((3))
    alphalk = 0.0
    ld, md, nd = int(0), int(0), int(0)

    
    maxprims = bfs_coeffs.shape[1]
    maxprims_aux = aux_bfs_coeffs.shape[1]

    # Create arrays to avoid contention of allocator 
    # (https://stackoverflow.com/questions/70339388/using-numba-with-np-concatenate-is-not-efficient-in-parallel/70342014#70342014)

    ibatch_threads = np.zeros(get_num_threads(), dtype=np.uint8)
    i_threads = np.zeros(get_num_threads(), dtype=np.uint8)

    #Loop over BFs
    for itemp in prange(ntriplets):
        id_thrd = get_thread_id()

        if ibatch_threads[id_thrd]<offset.shape[0]:
            if itemp>=offset[ibatch_threads[id_thrd]]:
                ibatch_threads[id_thrd] +=1
                i_threads[id_thrd] += 1 
                
                # print(itemp)
                # print(ibatch_threads)
                # print(i_threads)
        

        i = i_threads[id_thrd]
        j = indicesB[itemp]
        k = indicesC[itemp]
        
        
        Ni = bfs_contr_prim_norms[i]
        nprimi = bfs_nprim[i]
        alphaik = np.zeros(maxprims, dtype=np.float64) # Should be Hoisted out
        alphaik[:] = bfs_expnts[i,:]
        lmni = bfs_lmn[i]
        la, ma, na = lmni
        I = bfs_coords[i]

        Nj = bfs_contr_prim_norms[j]
        nprimj = bfs_nprim[j]
        alphajk = np.zeros(maxprims, dtype=np.float64) # Should be Hoisted out
        alphajk[:] = bfs_expnts[j,:]
        lmnj = bfs_lmn[j]
        lb, mb, nb = lmnj
        J = bfs_coords[j]

        
        IJsq = IJsq_arr[i,j]
        tempcoeff1 = Ni*Nj
        gammaP = np.zeros((maxprims, maxprims), dtype=np.float64) # Should be Hoisted out
        screenfactorAB = np.zeros((maxprims, maxprims), dtype=np.float64) # Should be Hoisted out
        # Ap = np.zeros((bfs_coeffs.shape[1], bfs_coeffs.shape[1]), dtype=np.float64) # Should be Hoisted out
        
        # kshell = aux_shell_indices[k]
        Nk = aux_bfs_contr_prim_norms[k]
        nprimk = aux_bfs_nprim[k]
        alphakk = np.zeros(maxprims_aux, dtype=np.float64) # Should be Hoisted out
        alphakk[:] = aux_bfs_expnts[k,:]
        lmnk = aux_bfs_lmn[k]
        lc, mc, nc = lmnk
        K = aux_bfs_coords[k]
        Q = K   

        
        tempcoeff2 = tempcoeff1*Nk
        norder = int((la+ma+na+lb+mb+nb+lc+mc+nc+ld+md+nd)/2 + 1 ) 
        tempcoeff3 = np.zeros(maxprims, dtype=np.float64) # Should be Hoisted out 
        tempcoeff3[:] = tempcoeff2*bfs_coeffs[i,:]*bfs_prim_norms[i,:]
        # PQsq = np.zeros((bfs_coeffs.shape[1], bfs_coeffs.shape[1], aux_bfs_coeffs.shape[1]), dtype=np.float64)
        # rho = np.zeros((bfs_coeffs.shape[1], bfs_coeffs.shape[1], aux_bfs_coeffs.shape[1]), dtype=np.float64)
            
            
            
        val = 0.0
        if norder<=10: # Use rys quadrature
            
            n = int(max(la+lb,ma+mb,na+nb))
            m = int(max(lc+ld,mc+md,nc+nd))
            # roots = np.zeros((norder)) 
            # weights = np.zeros((norder)) 
            roots = np.zeros((10)) 
            weights = np.zeros((10)) 
            # G = np.zeros((n+1,m+1)) 
            G = np.zeros((n+1,m+1)) 
            
                
            #Loop over primitives
            for ik in range(nprimi): 
                alphaik_ = alphaik[ik]
                tempcoeff3_ = tempcoeff3[ik]
                for jk in range(nprimj):
                    alphajk_ = alphajk[jk]
                    # gammaP[ik,jk] = alphaik_ + alphajk_
                    gammaP_ = alphaik_ + alphajk_
                    # Ap[ik,jk] = alphaik[ik]*alphajk[jk]
                    # screenfactorAB[ik,jk] = np.exp(-Ap[ik,jk]/gammaP[ik,jk]*IJsq)
                    screenfactorAB = np.exp(-alphaik_*alphajk_/gammaP_*IJsq)
                        
                    if abs(screenfactorAB)<1.0e-8:   
                        #Although this value of screening threshold seems very large
                        # it actually gives the best consistency with PySCF. Reducing it to 1e-15,
                        # actually worsened the agreement.
                        # I suspect that this is caused due to an error cancellation
                        # that happens with the nucmat calculation, as the same screening is 
                        # used there as well
                        continue
                    djk = bfs_coeffs[j,jk] 
                    Njk = bfs_prim_norms[j,jk]     
                    P = (alphaik_*I + alphajk_*J)/gammaP_
                    PQ = P - Q
                    PQsq = np.sum(PQ**2)
                    tempcoeff4 = tempcoeff3_*djk*Njk  
                    
                    # gammaP_ = gammaP[ik,jk]
                        
                    for kk in range(nprimk):

                        dkk = aux_bfs_coeffs[k,kk]
                        Nkk = aux_bfs_prim_norms[k,kk]
                        tempcoeff5 = tempcoeff4*dkk*Nkk 
                              
                            
                        gammaQ = alphakk[kk]
                        rho = gammaP_*gammaQ/(gammaP_ + gammaQ)
                                
                        # ABsrt = np.sqrt(gammaP[ik,jk]*alphakk[kk])
                        # X = PQsq*rho
                        # factor = 2*np.sqrt(rho/pi)
                        # print('s')
                        val += tempcoeff5*coulomb_rys_3c2e(roots,weights,G,PQsq, rho, norder,n,m,la,lb,lc,ld,ma,mb,mc,md,na,nb,nc,nd,alphaik_, alphajk_,alphakk[kk],alphalk,I,J,K,L)
                        # The following should have been faster but isnt somehow
                        # val += tempcoeff5*coulomb_rys_new(roots,weights,G,PQsq, rho, norder,n,m,la,lb,lc,ld,ma,mb,mc,md,na,nb,nc,nd,alphaik_, alphajk_,alphakk[kk],alphalk,I,J,K,L)
                        # The following should have been faster but isnt
                        # val += tempcoeff5*coulomb_rys_fast(roots,weights,G,norder,la,lb,lc,ld,ma,mb,mc,md,na,nb,nc,nd,alphaik[ik], alphajk[jk], alphakk[kk], alphalk,I,J,K,L,X,gammaP[ik,jk],alphakk[kk],Ap[ik,jk],0.0,ABsrt,factor,P,Q)
                            
        else: # Analytical (Conventional)
            #Loop over primitives
            for ik in range(nprimi):   
                for jk in range(nprimj):
                    
                    gammaP[ik,jk] = alphaik[ik] + alphajk[jk]
                    # Ap[ik,jk] = alphaik[ik]*alphajk[jk]
                    # screenfactorAB[ik,jk] = np.exp(-Ap[ik,jk]/gammaP[ik,jk]*IJsq)
                    screenfactorAB[ik,jk] = np.exp(-alphaik[ik]*alphajk[jk]/gammaP[ik,jk]*IJsq)
                        
                    if abs(screenfactorAB[ik,jk])<1.0e-8:   
                        #Although this value of screening threshold seems very large
                        # it actually gives the best consistency with PySCF. Reducing it to 1e-15,
                        # actually worsened the agreement.
                        # I suspect that this is caused due to an error cancellation
                        # that happens with the nucmat calculation, as the same screening is 
                        # used there as well
                        continue
                    djk = bfs_coeffs[j,jk] 
                    Njk = bfs_prim_norms[j,jk]      
                    P = (alphaik[ik]*I + alphajk[jk]*J)/gammaP[ik,jk]
                    PI = P - I
                    PJ = P - J  
                    PQ = P - Q
                    fac1 = twopisq/gammaP[ik,jk]*screenfactorAB[ik,jk]  
                    onefourthgammaPinv = 0.25/gammaP[ik,jk]
                    tempcoeff4 = tempcoeff3[ik]*djk*Njk  
                        
                    for kk in range(nprimk):
                        dkk = aux_bfs_coeffs[k,kk]
                        Nkk = aux_bfs_prim_norms[k,kk]
                        tempcoeff5 = tempcoeff4*dkk*Nkk 
                              
                        
                        gammaQ = alphakk[kk] #+ alphalk
                        
                        
                                
                        QK = Q - K
                        QL = Q #- L
                        
                                  
                        fac2 = fac1/gammaQ
                                

                        omega = (fac2)*np.sqrt(pi/(gammaP[ik,jk] + gammaQ))
                        delta = 0.25*(1/gammaQ) + onefourthgammaPinv          
                        PQsqBy4delta = np.sum(PQ**2)/(4*delta)         
                                    
                        sum1 = innerLoop4c2e(la,lb,lc,ld,ma,mb,mc,md,na,nb,nc,nd,gammaP[ik,jk],gammaQ,PI,PJ,QK,QL,PQ,PQsqBy4delta,delta)
                                    
                        val += omega*sum1*tempcoeff5

        threeC2E[itemp] = val
                                    
                                 

    return threeC2E

# Reference: https://github.com/numba/numba/issues/8007#issuecomment-1113187684
parallel_options_algo8 = {
    'comprehension': False,  # parallel comprehension
    'prange':        False,  # parallel for-loop
    'numpy':         False,  # parallel numpy calls
    'reduction':     False,  # parallel reduce calls
    'setitem':       False,  # parallel setitem
    'stencil':       False,  # parallel stencils
    'fusion':        False,  # enable fusion or not
}
@njit(parallel=False, cache=True, fastmath=True, error_model="numpy", nogil=True, boundscheck=False, debug=False)
def rys_3c2e_tri_schwarz_sparse_algo8_internal(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts,aux_bfs_coords, aux_bfs_contr_prim_norms, aux_bfs_lmn, aux_bfs_nprim, aux_bfs_coeffs, aux_bfs_prim_norms, aux_bfs_expnts, ntriplets, nao, naux, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold):
    # Calculates 3c2e integrals based on a provided list of significant triplets determined using Schwarz inequality
    # It is assumed that the provided list was made with triangular int3c2e in mind
    # TODO: Check out these screening methods as well 
    # https://kops.uni-konstanz.de/server/api/core/bitstreams/79ada61a-fd29-43fd-a298-79c1696a0601/content
    # https://aip.scitation.org/doi/10.1063/1.4917519

    #Debug:
    # print(config.THREADING_LAYER)
    # print(threading_layer())
    # print(get_parallel_chunksize())
    # print(ntriplets)
    # set_parallel_chunksize(ntriplets//128)
    # print(get_parallel_chunksize())

    threeC2E = np.zeros((ntriplets), dtype=np.float64) 

    pi = 3.141592653589793
    pisq = 9.869604401089358  #PI^2
    twopisq = 19.739208802178716  #2*PI^2

    L = np.zeros((3))
    ld, md, nd = int(0), int(0), int(0)
    alphalk = 0.0

    
    index = 0
    #Loop pver BFs
    for i in prange(0, nao): #A
        
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

            sqrt_ij = sqrt_ints4c2e_diag[i,j]
            
            
            for k in prange(0, naux): #C
                if sqrt_ij*sqrt_diag_ints2c2e[k]<=threshold:
                    continue
                K = aux_bfs_coords[k]
                Nk = aux_bfs_contr_prim_norms[k]
                lmnk = aux_bfs_lmn[k]
                lc, mc, nc = lmnk
                tempcoeff2 = tempcoeff1*Nk
                nprimk = aux_bfs_nprim[k]
                
                
                norder = int((la+ma+na+lb+mb+nb+lc+mc+nc+ld+md+nd)/2 + 1 ) 
                

                if norder<=10: # Use rys quadrature
                    n = int(max(la+lb,ma+mb,na+nb))
                    m = int(max(lc+ld,mc+md,nc+nd))
                    roots = np.zeros((norder), dtype=np.float64)
                    weights = np.zeros((norder), dtype=np.float64)
                    G = np.zeros((n+1,m+1), dtype=np.float64)
                    
                    val = 0.0
                    #Loop over primitives
                    for ik in range(nprimi):   
                        dik = bfs_coeffs[i,ik]
                        Nik = bfs_prim_norms[i,ik]
                        alphaik = bfs_expnts[i,ik]
                        tempcoeff3 = tempcoeff2*dik*Nik
                            
                        for jk in range(nprimj):
                            alphajk = bfs_expnts[j,jk]
                            gammaP = alphaik + alphajk
                            screenfactorAB = np.exp(-alphaik*alphajk/gammaP*IJsq)
                            if abs(screenfactorAB)<1.0e-8:   
                                #TODO: Check for optimal value for screening
                                continue
                            djk = bfs_coeffs[j,jk] 
                            Njk = bfs_prim_norms[j,jk]      
                            P = (alphaik*I + alphajk*J)/gammaP
                            tempcoeff4 = tempcoeff3*djk*Njk  
                                
                            for kk in range(nprimk):
                                dkk = aux_bfs_coeffs[k,kk]
                                Nkk = aux_bfs_prim_norms[k,kk]
                                alphakk = aux_bfs_expnts[k,kk]
                                tempcoeff5 = tempcoeff4*dkk*Nkk 
                                    
                                    
                                gammaQ = alphakk 
                                
                                Q = K        
                                PQ = P - Q
                                PQsq = np.sum(PQ**2)
                                rho = gammaP*gammaQ/(gammaP+gammaQ) 
                                
                                val += tempcoeff5*coulomb_rys_3c2e(roots,weights,G,PQsq, rho, norder,n,m,la,lb,lc,ld,ma,mb,mc,md,na,nb,nc,nd,alphaik, alphajk, alphakk, alphalk,I,J,K,L)
                                

                else: # Analytical (Conventional)
                    val = 0.0
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
                                    
                                
                                gammaQ = alphakk 
                                
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

                threeC2E[index] = val
                index += 1
        
    return threeC2E


@njit(parallel=True, cache=True, fastmath=True, error_model="numpy", nogil=True)
def df_coeff_calculator_test(ints3c2e_1d, dmat_1d, indicesA, indicesB, indicesC, naux):
    # This function calculates the coefficients of the auxiliary basis for
    # density fitting. 
    # This can also be simply calculated using:
    # df_coeff = contract('pqP,pq->P', ints3c2e, dmat) # For general 3d and 2d arrays
    # or
    # df_coeff = contract('pP,p->P', ints3c2e, dmat_tri) # For triangular versions of the above arrays
    # However, we need to create a custom function to
    # do this with a sparse 1d array holding the values of significant
    # elements of ints3c2e array.
    nelements = ints3c2e_1d.shape[0]
    df_coeff = np.zeros((naux), dtype=np.float64)
    for itemp in prange(nelements):
        # This is extremely slow even though the following is hoisted out automatically
        df_coeff_temp = np.zeros((naux), dtype=np.float64) # Temp arrary to avoid race condition 
        i = indicesA[itemp]
        offset = int(i*(i+1)/2)
        j = indicesB[itemp] 
        k = indicesC[itemp]
        df_coeff_temp[k] = ints3c2e_1d[itemp]*dmat_1d[j+offset] # This avoids the race condition
        df_coeff += df_coeff_temp
    return df_coeff

def df_coeff_calculator(ints3c2e_1d, dmat_1d, indicesA, indicesB, indicesC, naux, ncores):
    ntriplets = len(indicesA)
    batch_size = min(ntriplets, int(1e6))
    nbatches = ntriplets//batch_size
    output = Parallel(n_jobs=ncores, backend='threading', require='sharedmem', batch_size='auto')(delayed(df_coeff_calculator_internal)(ints3c2e_1d[ibatch*batch_size : min(ibatch*batch_size+batch_size,ntriplets)], dmat_1d, indicesA[ibatch*batch_size : min(ibatch*batch_size+batch_size,ntriplets)], indicesB[ibatch*batch_size : min(ibatch*batch_size+batch_size,ntriplets)], indicesC[ibatch*batch_size : min(ibatch*batch_size+batch_size,ntriplets)], naux) for ibatch in range(nbatches+1))
    df_coeff = np.zeros((naux), dtype=np.float64)
    for ibatch in range(0,len(output)):
        df_coeff += output[ibatch]
    # Free memory
    output = 0
    del output
    return df_coeff


@njit(parallel=False, cache=True, fastmath=True, error_model="numpy", nogil=True)
def df_coeff_calculator_internal(ints3c2e_1d, dmat_1d, indicesA, indicesB, indicesC, naux):
    # This function calculates the coefficients of the auxiliary basis for
    # density fitting. 
    # This can also be simply calculated using:
    # df_coeff = contract('pqP,pq->P', ints3c2e, dmat) # For general 3d and 2d arrays
    # or
    # df_coeff = contract('pP,p->P', ints3c2e, dmat_tri) # For triangular versions of the above arrays
    # However, we need to create a custom function to
    # do this with a sparse 1d array holding the values of significant
    # elements of ints3c2e array.
    nelements = ints3c2e_1d.shape[0]
    df_coeff = np.zeros((naux), dtype=np.float64)
    for itemp in range(nelements):
        i = indicesA[itemp]
        offset = int(i*(i+1)/2)
        j = indicesB[itemp] 
        k = indicesC[itemp]
        df_coeff[k] += ints3c2e_1d[itemp]*dmat_1d[j+offset] # This leads to race condition
    return df_coeff

@njit(parallel=True, cache=True, fastmath=True, error_model="numpy", nogil=True)
def df_coeff_calculator_old(ints3c2e_1d, dmat_1d, indicesA, indicesB, indicesC, naux):
    # This function calculates the coefficients of the auxiliary basis for
    # density fitting. 
    # This can also be simply calculated using:
    # df_coeff = contract('pqP,pq->P', ints3c2e, dmat) # For general 3d and 2d arrays
    # or
    # df_coeff = contract('pP,p->P', ints3c2e, dmat_tri) # For triangular versions of the above arrays
    # However, we need to create a custom function to
    # do this with a sparse 1d array holding the values of significant
    # elements of ints3c2e array.
    nelements = ints3c2e_1d.shape[0]
    df_coeff = np.zeros((naux), dtype=np.float64)
    for itemp in range(nelements):
        i = indicesA[itemp]
        offset = int(i*(i+1)/2)
        j = indicesB[itemp] 
        k = indicesC[itemp]
        df_coeff[k] += ints3c2e_1d[itemp]*dmat_1d[j+offset] # This leads to race condition
    return df_coeff

@njit(parallel=True, cache=True, fastmath=True, error_model="numpy", nogil=True)
def df_coeff_calculator_algo8(ints3c2e_1d, dmat_1d, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold, nao, naux):
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
    df_coeff = np.zeros((naux), dtype=np.float64)
    itemp = 0
    for i in range(0, nao):
        offset = int(i*(i+1)/2)
        for j in range(0, i+1):
            sqrt_ij = sqrt_ints4c2e_diag[i,j]
            for k in range(0, naux):
                if sqrt_ij*sqrt_diag_ints2c2e[k]>threshold:
                    df_coeff[k] += ints3c2e_1d[itemp]*dmat_1d[j+offset] # This leads to race condition
                    itemp += 1
    return df_coeff

@njit(parallel=True, cache=True, fastmath=True, error_model="numpy", nogil=True)
def df_coeff_calculator_algo10_serial(ints3c2e_1d, dmat_1d, indicesA, indicesB, offsets_3c2e, naux, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold, strict_schwarz):
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
    df_coeff = np.zeros((naux), dtype=np.float64)
    for ij in range(indicesA.shape[0]):
        i = indicesA[ij]
        j = indicesB[ij]
        offset = int(i*(i+1)/2)
        sqrt_ij = sqrt_ints4c2e_diag[i,j]
        if strict_schwarz:
            if sqrt_ij*sqrt_ij<1e-13:
                continue  
        index_k = 0
        for k in range(0, naux):
            if sqrt_ij*sqrt_diag_ints2c2e[k]>threshold:
                df_coeff[k] += ints3c2e_1d[offsets_3c2e[ij]+index_k]*dmat_1d[j+offset]  # This leads to race condition
                index_k += 1 
    return df_coeff


def df_coeff_calculator_algo10_parallel(ints3c2e_1d, dmat_1d, indicesA, indicesB, offsets_3c2e, naux, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold, ncores, strict_schwarz, auxbfs_lm, auxbfs_lmn):
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
    batch_size = min(indicesA.shape[0], int(500))
    nbatches = indicesA.shape[0]//batch_size
    output = Parallel(n_jobs=ncores, backend='threading', require='sharedmem', batch_size='auto')(delayed(df_coeff_calculator_algo10_parallel_internal)(ints3c2e_1d[offsets_3c2e[ibatch*batch_size] : offsets_3c2e[min(ibatch*batch_size+batch_size, indicesA.shape[0])]], dmat_1d, indicesA[ibatch*batch_size : min(ibatch*batch_size+batch_size,indicesA.shape[0])], indicesB[ibatch*batch_size : min(ibatch*batch_size+batch_size,indicesA.shape[0])], offsets_3c2e[ibatch*batch_size : min(ibatch*batch_size+batch_size,indicesA.shape[0])], naux, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold, strict_schwarz, auxbfs_lm, auxbfs_lmn) for ibatch in range(nbatches+1))
    df_coeff = np.zeros((naux), dtype=np.float64)
    for ibatch in range(0,len(output)):
        df_coeff += output[ibatch]
    # Free memory
    output = 0
    del output
    return df_coeff

@njit(parallel=False, cache=True, fastmath=True, error_model="numpy", nogil=True)
def df_coeff_calculator_algo10_parallel_internal(ints3c2e_1d, dmat_1d, indicesA, indicesB, offsets_3c2e, naux, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold, strict_schwarz, auxbfs_lm, auxbfs_lmn):
    # This function calculates the coefficients of the auxiliary basis for
    # density fitting. 
    # This can also be simply calculated using:
    # df_coeff = contract('pqP,pq->P', ints3c2e, dmat) # For general 3d and 2d arrays
    # or
    # df_coeff = contract('pP,p->P', ints3c2e, dmat_tri) # For triangular versions of the above arrays
    # However, we need to create a custom function to
    # do this with a sparse 1d array holding the values of significant
    # elements of ints3c2e array.
    df_coeff = np.zeros((naux), dtype=np.float64)
    for ij in range(indicesA.shape[0]):
        i = indicesA[ij]
        j = indicesB[ij]
        offset = int(i*(i+1)/2)
        sqrt_ij = sqrt_ints4c2e_diag[i,j]
        if strict_schwarz:
            if sqrt_ij*sqrt_ij<1e-13:
                continue  
        index_k = 0
        for k in range(0, naux):
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
            #         df_coeff[k] += ints3c2e_1d[offsets_3c2e[ij]-offsets_3c2e[0]+index_k]*dmat_1d[j+offset]  # This leads to race condition
            #         index_k += 1 
            #     else:
            #         continue
            # else:  
            #     if sqrt_ij*sqrt_diag_ints2c2e[k]>threshold:
            #         df_coeff[k] += ints3c2e_1d[offsets_3c2e[ij]-offsets_3c2e[0]+index_k]*dmat_1d[j+offset]  # This leads to race condition
            #         index_k += 1 
            if sqrt_ij*sqrt_diag_ints2c2e[k]>threshold:
                df_coeff[k] += ints3c2e_1d[offsets_3c2e[ij]-offsets_3c2e[0]+index_k]*dmat_1d[j+offset]  # This leads to race condition
                index_k += 1 
        # k = 0
        # index_k = 0
        # while k < naux:    
        #     lmnk = auxbfs_lmn[k]
        #     tot_ang = lmnk.sum()
        #     is_first = lmnk[0]==tot_ang
        #     if is_first:
        #         if sqrt_ij*sqrt_diag_ints2c2e[k]>threshold:
        #             # Determine shell size: (L+1)(L+2)/2
        #             shell_size = (tot_ang + 1) * (tot_ang + 2) // 2
        #             df_coeff[k:k+shell_size] += ints3c2e_1d[offsets_3c2e[ij]-offsets_3c2e[0]+index_k:offsets_3c2e[ij]-offsets_3c2e[0]+index_k+shell_size]*dmat_1d[j+offset]
        #             k += shell_size
        #             index_k += shell_size
        #             # continue
                
    return df_coeff



# def df_coeff_calculator(ints3c2e_1d, dmat_1d, indicesA, indicesB, indicesC, naux):
#     # This function calculates the coefficients of the auxiliary basis for
#     # density fitting.
#     i = indicesA
#     j = indicesB
#     k = indicesC
#     offset = (i*(i+1))//2
#     df_coeff = np.zeros((naux))
#     df_coeff[k] += ints3c2e_1d * dmat_1d[j+offset]
#     return df_coeff

# @guvectorize([(float64[:], float64[:], int16[:], int16[:], int16[:], int16[:], float64[:])], '(n),(m),(n)->(m)', target='parallel')
# def df_coeff_calculator(ints3c2e_1d, dmat_1d, indicesA, indicesB, indicesC, naux, df_coeff_out):
#     # This function calculates the coefficients of the auxiliary basis for
#     # density fitting. 
#     # This can also be simply calculated using:
#     # df_coeff = contract('pqP,pq->P', ints3c2e, dmat) # For general 3d and 2d arrays
#     # or
#     # df_coeff = contract('pP,p->P', ints3c2e, dmat_tri) # For triangular versions of the above arrays
#     # However, we need to create a custom function to
#     # do this with a sparse 1d array holding the values of significant
#     # elements of ints3c2e array.
#     for i in range(ints3c2e_1d.shape[0]):
#         offset = int(indicesA[i]*(indicesA[i]+1)/2)
#         df_coeff_out[indicesC[i]] += ints3c2e_1d[i]*dmat_1d[indicesB[i]+offset]

@njit(parallel=True, cache=True, fastmath=True, error_model="numpy")
def J_tri_calculator(ints3c2e_1d, df_coeff, indicesA, indicesB, indicesC, size_J_tri):
    # This can also be simply calculated using:
    # J = contract('ijk,k', ints3c2e, df_coeff) # For general 3d and 2d arrays
    # or
    # J_tri = contract('pP,P', ints3c2e, df_coeff) # For triangular versions of the above arrays
    # However, we need to create a custom function to
    # do this with a sparse 1d array holding the values of significant
    # elements of ints3c2e array.
    nelements = ints3c2e_1d.shape[0]
    J_tri = np.zeros((size_J_tri), dtype=np.float64) 
    for itemp in prange(nelements): 
        i = indicesA[itemp]
        offset = int(i*(i+1)/2)
        j = indicesB[itemp] 
        k = indicesC[itemp]
        J_tri[j+offset] += ints3c2e_1d[itemp]*df_coeff[k] 
    return J_tri

@njit(parallel=True, cache=True, fastmath=True, error_model="numpy")
def J_tri_calculator_algo10(ints3c2e_1d, df_coeff, indicesA, indicesB, offsets_3c2e, size_J_tri, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold, naux, strict_schwarz, auxbfs_lm):
    # This can also be simply calculated using:
    # J = contract('ijk,k', ints3c2e, df_coeff) # For general 3d and 2d arrays
    # or
    # J_tri = contract('pP,P', ints3c2e, df_coeff) # For triangular versions of the above arrays
    # However, we need to create a custom function to
    # do this with a sparse 1d array holding the values of significant
    # elements of ints3c2e array.
    # print(indicesA)
    npairs = indicesA.shape[0]
    J_tri = np.zeros((size_J_tri), dtype=np.float64) 
    for itemp in prange(npairs): 
        i = indicesA[itemp]
        offset = int(i*(i+1)/2)
        j = indicesB[itemp] 
        sqrt_ij = sqrt_ints4c2e_diag[i,j]
        if strict_schwarz:
            if sqrt_ij*sqrt_ij<1e-13:
                continue  
        index_k = 0
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
            #         J_tri[j+offset] += ints3c2e_1d[offsets_3c2e[itemp]+index_k]*df_coeff[k] 
            #         index_k += 1
            #     else:
            #         continue
            # else:
            #     if sqrt_ij*sqrt_diag_ints2c2e[k]>threshold:
            #         J_tri[j+offset] += ints3c2e_1d[offsets_3c2e[itemp]+index_k]*df_coeff[k] 
            #         index_k += 1
            if sqrt_ij*sqrt_diag_ints2c2e[k]>threshold:
                J_tri[j+offset] += ints3c2e_1d[offsets_3c2e[itemp]+index_k]*df_coeff[k] 
                index_k += 1
    return J_tri

@njit(parallel=True, cache=True, fastmath=True, error_model="numpy")
def J_tri_calculator_algo8(ints3c2e_1d, df_coeff, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold, nao, naux, size_J_tri):
    # This can also be simply calculated using:
    # J = contract('ijk,k', ints3c2e, df_coeff) # For general 3d and 2d arrays
    # or
    # J_tri = contract('pP,P', ints3c2e, df_coeff) # For triangular versions of the above arrays
    # However, we need to create a custom function to
    # do this with a sparse 1d array holding the values of significant
    # elements of ints3c2e array.
    J_tri = np.zeros((size_J_tri), dtype=np.float64) 
    itemp = 0
    for i in range(0, nao):
        offset = int(i*(i+1)/2)
        for j in range(0, i+1):
            sqrt_ij = sqrt_ints4c2e_diag[i,j]
            for k in range(0, naux):
                if sqrt_ij*sqrt_diag_ints2c2e[k]>threshold:
                    J_tri[j+offset] += ints3c2e_1d[itemp]*df_coeff[k] # This should lead to a race condition but doesn't somehow
                    itemp += 1
    return J_tri

@njit(parallel=True, cache=True, fastmath=True, error_model="numpy")
def J_tri_calculator_from_4c2e(ints4c2e_1d, dmat_1d, indicesA, indicesB, indicesC, indicesD, size_J_tri):
    # This can also be simply calculated using:
    # J = contract('ijkl,lk', ints4c2e, ddmat) # For general 3d and 2d arrays
    # However, we need to create a custom function to
    # do this with a sparse 1d array holding the values of significant
    # elements of ints4c2e array.
    nelements = ints4c2e_1d.shape[0]
    J_tri = np.zeros((size_J_tri), dtype=np.float64) 
    for itemp in prange(nelements): 
        i = indicesA[itemp]
        offset = int(i*(i+1)/2)
        j = indicesB[itemp] 
        k = indicesC[itemp]
        l = indicesD[itemp]
        J_tri[j+offset] += ints4c2e_1d[itemp]*dmat_1d[j+offset] # This should lead to a race condition but doesn't somehow
    return J_tri

import numpy as np
from numba import cuda, njit

@cuda.jit
def J_tri_calculator_kernel(ints3c2e_1d, df_coeff, indicesA, indicesB, indicesC, J_tri):
    # This can also be simply calculated using:
    # J = contract('ijk,k', ints3c2e, df_coeff) # For general 3d and 2d arrays
    # or
    # J_tri = contract('pP,P', ints3c2e, df_coeff) # For triangular versions of the above arrays
    # However, we need to create a custom function to
    # do this with a sparse 1d array holding the values of significant
    # elements of ints3c2e array.
    tid = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    stride = cuda.gridDim.x * cuda.blockDim.x

    nelements = ints3c2e_1d.shape[0]
    offset = 0
    for itemp in range(tid, nelements, stride):
        i = indicesA[itemp]
        offset = int(i*(i+1)/2)
        j = indicesB[itemp] 
        k = indicesC[itemp]
        cuda.atomic.add(J_tri, j+offset, ints3c2e_1d[itemp]*df_coeff[k])


def J_tri_calculator_cupy(ints3c2e_1d, df_coeff, indicesA, indicesB, indicesC, size_J_tri):
    nelements = ints3c2e_1d.shape[0]
    J_tri = np.zeros((size_J_tri), dtype=cp.float64)
    blockdim = 512
    # griddim = (cuda.device_array(1, dtype=cp.int32), 1)

    # Set the grid dimensions such that there are enough blocks to cover all threads
    griddim = (nelements + blockdim - 1) // blockdim

    # Launch the kernel
    J_tri_calculator_kernel[griddim, blockdim](ints3c2e_1d, df_coeff, indicesA, indicesB, indicesC, J_tri)

    return J_tri
