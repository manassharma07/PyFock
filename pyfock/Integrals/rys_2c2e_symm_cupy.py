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


def rys_2c2e_symm_cupy(basis, slice=None, cp_stream=None):
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

    if slice is None:
        slice = [0, basis.bfs_nao, 0, basis.bfs_nao]
        
    #Limits for the calculation of overlap integrals
    a = int(slice[0]) #row start index
    b = int(slice[1]) #row end index
    c = int(slice[2]) #column start index
    d = int(slice[3]) #column end index

    # Infer the matrix shape from the start and end indices
    num_rows = b - a 
    num_cols = d - c 
    start_row = a
    end_row = b
    start_col = c
    end_col = d
    matrix_shape = (num_rows, num_cols)

    # Check if the slice of the matrix requested falls in the lower/upper triangle or in both the triangles
    # Check if the slice of the matrix requested falls in the lower/upper triangle or in both the triangles
    tri_symm = False
    no_symm = False
    if start_row==start_col and end_row==end_col:
        tri_symm = True
    else:
        no_symm = True

    # Initialize the matrix with zeros
    V = cp.zeros(matrix_shape) 

    

    

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
    blocks_per_grid = ((num_rows + (thread_x - 1))//thread_x, (num_cols + (thread_y - 1))//thread_y) 
    rys_2c2e_symm_internal_cuda[blocks_per_grid, (thread_x, thread_y), nb_stream](bfs_coords[0], bfs_contr_prim_norms[0], bfs_lmn[0], bfs_nprim[0], bfs_coeffs, bfs_prim_norms, bfs_expnts, a, b, c, d, tri_symm, no_symm, DATA_X_cuda, DATA_W_cuda, V)
    if tri_symm:
        thread_x = 32
        thread_y = 32
        blocks_per_grid = ((num_rows + (thread_x - 1))//thread_x, (num_cols + (thread_y - 1))//thread_y) 
        symmetrize[blocks_per_grid, (thread_x, thread_y), nb_stream](a,b,c,d,V)
    if cp_stream is None:
        cuda.synchronize()
    else:
        cp_stream.synchronize()
        cp.cuda.Stream.null.synchronize()
        # cp._default_memory_pool.free_all_blocks()
    return V





@cuda.jit(fastmath=True, cache=True, max_registers=50)#(device=True)
def rys_2c2e_symm_internal_cuda(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts, start_row, end_row, start_col, end_col, tri_symm, no_symm, DATA_X, DATA_W, out):
    # Two centered two electron integrals by hacking the 4c2e routines based on rys quadrature.
    # Based on Rys Quadrature from https://github.com/rpmuller/MolecularIntegrals.jl
    # This function calculates the electron-electron Coulomb potential matrix for a given basis object.
    # The basis object holds the information of basis functions like: exponents, coeffs, etc.
        
    # returns (A|C) 
    
    i, k = cuda.grid(2)
    J = L = cuda.local.array((3), numba.float64)
    lb, mb, nb = int(0), int(0), int(0)
    ld, md, nd = int(0), int(0), int(0)
    alphajk = alphalk = 0.0

    if i>=start_row and i<end_row and k>=start_col and k<end_col:
        I = bfs_coords[i]
        P = I
        Ni = bfs_contr_prim_norms[i]
        lmni = bfs_lmn[i]
        la, ma, na = lmni
        nprimi = bfs_nprim[i]    
        
        if (tri_symm and k<=i) or no_symm:
            PQ = cuda.local.array((3), numba.float64)

            val = 0.0


            K = bfs_coords[k]
            Q = K
            Nk = bfs_contr_prim_norms[k]
            lmnk = bfs_lmn[k]
            lc, mc, nc = lmnk
            tempcoeff1 = Ni*Nk
            nprimk = bfs_nprim[k]
            

            norder = int((la+ma+na+lc+mc+nc)/2+1 ) 
            n = int(max(la,ma,na))
            m = int(max(lc,mc,nc))
            roots = cuda.local.array((8), numba.float64) # Good for upto j shells; j orbitals have an angular momentum of 7;
            weights = cuda.local.array((8), numba.float64) # Good for upto j shells; j orbitals have an angular momentum of 7;
            G = cuda.local.array((8, 8), numba.float64) # Good for upto j shells; j orbitals have an angular momentum of 7;
                    
            PQ[0] = P[0] - Q[0]
            PQ[1] = P[1] - Q[1]
            PQ[2] = P[2] - Q[2]
            PQsq = PQ[0]**2 + PQ[1]**2 + PQ[2]**2
            
                    
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
                    
                    X = PQsq*rho               
                    roots, weights = Roots(norder,X,DATA_X,DATA_W,roots,weights)
                    val += tempcoeff3*coulomb_rys(roots,weights,G,PQsq, rho, norder,n,m,la,lb,lc,ld,ma,mb,mc,md,na,nb,nc,nd,alphaik, alphajk, alphakk, alphalk,I,J,K,L)
                                

            out[i-start_row, k-start_col] = val        
                    
                

@cuda.jit(fastmath=True, cache=True)#(device=True)           
def symmetrize(start_row, end_row, start_col, end_col, out):
    #T + T.T - cp.diag(cp.diag(T))
    i, j = cuda.grid(2)
    if i>=start_row and i<end_row and j>=start_col and j<end_col:
        if j>i:
            out[i-start_row, j-start_col] = out[j-start_col, i-start_row]