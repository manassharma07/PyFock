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


def kin_mat_symm_shell_cupy(basis, slice=None, cp_stream=None):
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
    bfs_offset = cp.array([basis.shell_bfs_offset])
    bfs_nbfshell = cp.array([basis.bfs_nbfshell])
        

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
        


    
    matrix_shape = (basis.bfs_nao, basis.bfs_nao)

    # Initialize the matrix with zeros
    T = cp.zeros(matrix_shape) 

    thread_x = 32
    thread_y = 32

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
    blocks_per_grid = ((basis.nshells + (thread_x - 1))//thread_x, (basis.nshells + (thread_y - 1))//thread_y) 
    kin_mat_symm_shell_internal_cuda[blocks_per_grid, (32, 32), nb_stream](bfs_offset[0], bfs_nbfshell[0], bfs_coords[0], bfs_contr_prim_norms[0], bfs_lmn[0], bfs_nprim[0], bfs_coeffs, bfs_prim_norms, bfs_expnts, basis.nshells, T)
    blocks_per_grid = ((basis.bfs_nao + (thread_x - 1))//thread_x, (basis.bfs_nao + (thread_y - 1))//thread_y) 
    symmetrize[blocks_per_grid, (thread_x, thread_y), nb_stream](0,basis.bfs_nao,0,basis.bfs_nao,T)
    if cp_stream is None:
        cuda.synchronize()
    else:
        cp_stream.synchronize()
        cp.cuda.Stream.null.synchronize()
    return T #+ T.T - cp.diag(cp.diag(T))

LOOKUP_TABLE = np.array([
    1, 1, 2, 6, 24, 120, 720, 5040, 40320,
    362880, 3628800, 39916800, 479001600,
    6227020800, 87178291200, 1307674368000,
    20922789888000, 355687428096000, 6402373705728000,
    121645100408832000, 2432902008176640000], dtype='int64')

@cuda.jit(fastmath=True, cache=True, device=True)
def fastFactorial(n):
    # This is the way to access global constant arrays (which need to be on host, i.e. created using numpy for some reason)
    # See https://stackoverflow.com/questions/63311574/in-numba-how-to-copy-an-array-into-constant-memory-when-targeting-cuda
    LOOKUP_TABLE_ = cuda.const.array_like(LOOKUP_TABLE) 
    # 2-3x faster than the fastFactorial_old for values less than 21
    if n<= 1:
        return 1
    elif n<=20:
        return LOOKUP_TABLE_[n]
    else:
        factorial = 1
        for i in range(2, n+1):
            factorial *= i
        return factorial
    # factorial = 1
    # for i in range(2, n+1):
    #     factorial *= i
    # return factorial

LOOKUP_TABLE_COMB = np.array([[ 1,  0,  0,  0,  0,  0],
                        [ 1,  1,  0,  0,  0,  0],
                        [ 1,  2,  1,  0,  0,  0],
                        [ 1,  3,  3,  1,  0,  0],
                        [ 1,  4,  6,  4,  1,  0],
                        [ 1,  5, 10, 10,  5,  1]])

@cuda.jit(fastmath=True, cache=True, device=True)
def comb(x, y):
    table = cuda.const.array_like(LOOKUP_TABLE_COMB) 
    if y == 0: 
        return 1
    if x == y: 
        return 1
    if x<=5 and y<=5:
        return table[x,y]
    binom = fastFactorial(x) // fastFactorial(y) // fastFactorial(x - y)
    return binom

@cuda.jit(fastmath=True, cache=True, device=True)
def doublefactorial(n):
    if n <= 0:
        return 1
    else:
        result = 1
        for i in range(n, 0, -2):
            result *= i
        return result
        

@cuda.jit(fastmath=True, cache=True, device=True)   
def c2k(k,la,lb,PA,PB):
    temp = 0.0
    for i in range(la+1):
        if i>k:
            continue
        factor1 = comb(la,i)
        factor2 = PA**(la-i)
        for j in range(lb+1):
            # if j>k:
            #     continue
            if (i+j)==k :
                temp +=  factor1*comb(lb,j)*factor2*PB**(lb-j)
    return temp

@cuda.jit(fastmath=True, cache=True, device=True)
def calcS(la,lb,fac1,fac2,PA,PB):
    temp = 0.0
    # fac1 = math.sqrt(math.pi/gamma)
    # fac2 = 2*gamma
    for k in range(0, int((la+lb)/2)+1):
        temp +=  c2k(2*k,la,lb,PA,PB)*fac1*doublefactorial(2*k-1)/(fac2)**k
    return temp

@cuda.jit(fastmath=True, cache=True)#(device=True)           
def symmetrize(start_row, end_row, start_col, end_col, out):
    #T + T.T - cp.diag(cp.diag(T))
    i, j = cuda.grid(2)
    if i>=start_row and i<end_row and j>=start_col and j<end_col:
        if j>i:
            out[i-start_row, j-start_col] = out[j-start_col, i-start_row]


@cuda.jit(fastmath=True, cache=True, max_registers=60)#(device=True)
def kin_mat_symm_shell_internal_cuda(bfs_offset, bfs_nbfshell, bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts, nshells, out):
    # This function calculates the kinetic energy matrix and uses the symmetry property to only calculate half-ish the elements
    # and get the remaining half by symmetry.
    # The reason we need this extra function is because we want the callable function to be simple and not require so many 
    # arguments. But when using Numba to optimize, we can't have too many custom objects and stuff. Numba likes numpy arrays
    # so passing those is okay. But lists and custom objects are not okay.
    # This function calculates the kinetic matrix for a given basis object.
    # The basis object holds the information of basis functions like: exponents, coeffs, etc.
    # It is possible to only calculate a slice (block) of the complete matrix.
    # slice is a 4 element list whose first and second elements give the range of the rows to be calculated.
    # the third and fourth element give the range of columns to be calculated.
    # The integrals are performed using the formulas here https://pubs.acs.org/doi/full/10.1021/acs.jchemed.8b00255

    ishell, jshell = cuda.grid(2)
    if ishell>=0 and ishell<nshells and jshell>=0 and jshell<=ishell:
        # result = cuda.local.array((20,20), numba.float64) 
        # result[:,:] = 0.0
        # print(ishell, jshell)
        i = bfs_offset[ishell]
        j = bfs_offset[jshell]
        I = bfs_coords[i]
        # Ni = bfs_contr_prim_norms[i]
        # lmni = bfs_lmn[i] 
        result_sum = 0.0
        J = bfs_coords[j]
        # IJ = I - J  
        IJ = cuda.local.array((3), numba.float64) 
        P = cuda.local.array((3), numba.float64) 
        PI = cuda.local.array((3), numba.float64) 
        PJ = cuda.local.array((3), numba.float64) 
        IJ[0] = I[0] - J[0]
        IJ[1] = I[1] - J[1]
        IJ[2] = I[2] - J[2]
        # Nj = bfs_contr_prim_norms[j]
        # lmnj = bfs_lmn[j]
        #Some factors to save FLOPS
        fac1 = IJ[0]**2 + IJ[1]**2 + IJ[2]**2
        # fac2 = Ni*Nj
        # fac3 = (2*(lmnj[0]+lmnj[1]+lmnj[2])+3)
        # fac4 = (lmnj[0]*(lmnj[0]-1))
        # fac5 = (lmnj[1]*(lmnj[1]-1))
        # fac6 = (lmnj[2]*(lmnj[2]-1))
        for ik in range(bfs_nprim[i]):
            for jk in range(bfs_nprim[j]):
                alphaik = bfs_expnts[i,ik]
                alphajk = bfs_expnts[j,jk]
                alphajksq = alphajk*alphajk
                gamma = alphaik + alphajk
                fac_1 = math.sqrt(math.pi/gamma)
                fac_2 = 2*gamma
                temp_1 = math.exp(-alphaik*alphajk/gamma*fac1)
                if (abs(temp_1)<1.0e-9):
                    continue

                dik = bfs_coeffs[i,ik]
                djk = bfs_coeffs[j,jk] 
                
                # P = (alphaik*I + alphajk*J)/gamma
                P[0] = (alphaik*I[0] + alphajk*J[0])/gamma
                P[1] = (alphaik*I[1] + alphajk*J[1])/gamma
                P[2] = (alphaik*I[2] + alphajk*J[2])/gamma
                
                # PI = P - I
                PI[0] = P[0] - I[0]
                PI[1] = P[1] - I[1]
                PI[2] = P[2] - I[2]
                # PJ = P - J
                PJ[0] = P[0] - J[0]
                PJ[1] = P[1] - J[1]
                PJ[2] = P[2] - J[2]

                temp1 = dik*djk  #coeff of primitives as read from basis set
                
                

                for ibf in range(i, i + bfs_nbfshell[ishell]):
                    Ni = bfs_contr_prim_norms[ibf]
                    lmni = bfs_lmn[ibf] 
                    Nik = bfs_prim_norms[ibf,ik]
                    temp2 = temp1*Ni*Nik
                
                    for jbf in range(j, j + min(bfs_nbfshell[jshell], ibf+1)):
                        Nj = bfs_contr_prim_norms[jbf]
                        lmnj = bfs_lmn[jbf]
                        Njk = bfs_prim_norms[jbf,jk]
                        temp = temp2*Nj*Njk 
                        fac3 = (2*(lmnj[0]+lmnj[1]+lmnj[2])+3)
                        fac4 = (lmnj[0]*(lmnj[0]-1))
                        fac5 = (lmnj[1]*(lmnj[1]-1))
                        fac6 = (lmnj[2]*(lmnj[2]-1))
                        

                        Sx = calcS(lmni[0],lmnj[0],fac_1,fac_2,PI[0],PJ[0])
                        Sy = calcS(lmni[1],lmnj[1],fac_1,fac_2,PI[1],PJ[1])
                        Sz = calcS(lmni[2],lmnj[2],fac_1,fac_2,PI[2],PJ[2])
                        overlap1 = Sx*Sy*Sz

                        Sx = calcS(lmni[0],lmnj[0]+2,fac_1,fac_2,PI[0],PJ[0])
                        overlap2 = Sx*Sy*Sz

                        Sx = calcS(lmni[0],lmnj[0],fac_1,fac_2,PI[0],PJ[0])
                        Sy = calcS(lmni[1],lmnj[1]+2,fac_1,fac_2,PI[1],PJ[1])
                        overlap3 = Sx*Sy*Sz

                        Sy = calcS(lmni[1],lmnj[1],fac_1,fac_2,PI[1],PJ[1])
                        Sz = calcS(lmni[2],lmnj[2]+2,fac_1,fac_2,PI[2],PJ[2])
                        overlap4 = Sx*Sy*Sz
                        
                        Sx = calcS(lmni[0],lmnj[0]-2,fac_1,fac_2,PI[0],PJ[0])
                        Sz = calcS(lmni[2],lmnj[2],fac_1,fac_2,PI[2],PJ[2])
                        overlap5 = Sx*Sy*Sz

                        Sx = calcS(lmni[0],lmnj[0],fac_1,fac_2,PI[0],PJ[0])
                        Sy = calcS(lmni[1],lmnj[1]-2,fac_1,fac_2,PI[1],PJ[1])
                        overlap6 = Sx*Sy*Sz

                        if lmnj[2]==0:
                            part5 = 0
                        else:
                            Sy = calcS(lmni[1],lmnj[1],fac_1,fac_2,PI[1],PJ[1])
                            Sz = calcS(lmni[2],lmnj[2]-2,fac_1,fac_2,PI[2],PJ[2])
                            overlap7 = Sx*Sy*Sz
                            part5 = fac6*overlap7

                        part1 = overlap1*alphajk*fac3
                        part2 = 2*alphajksq*(overlap2+overlap3+overlap4)
                        part3 = fac4*overlap5
                        part4 = fac5*overlap6
                        

                        # result[ibf-i, jbf-j] += temp*(part1 - part2 - 0.5*(part3+part4+part5))*temp_1
                        out[ibf, jbf] += temp*(part1 - part2 - 0.5*(part3+part4+part5))*temp_1
                        
        # out[ibf:ibf+bfs_nbfshell[ishell], jbf:jbf+bfs_nbfshell[jshell]] = result[0:bfs_nbfshell[ishell], 0:bfs_nbfshell[jshell]]
        # for ibf in range(i, i + bfs_nbfshell[ishell]):     
        #     for jbf in range(j, j + min(bfs_nbfshell[jshell], ibf+1)):       
        #         out[ibf, jbf] = result[ibf-i, jbf-j]
