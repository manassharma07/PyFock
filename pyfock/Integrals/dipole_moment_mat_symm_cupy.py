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


def dipole_moment_mat_symm_cupy(basis, slice=None, origin=None, stream=None):
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

    if origin is None:
        origin = cp.zeros((3))
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
    matrix_shape = (3, num_rows, num_cols)

    # Check if the slice of the matrix requested falls in the lower/upper triangle or in both the triangles
    upper_tri = False
    lower_tri = False
    both_tri_symm = False
    both_tri_nonsymm = False
    if end_row <= start_col:
        upper_tri = True
    elif start_row >= end_col:
        lower_tri = True
    elif start_row==start_col and end_row==end_col:
        both_tri_symm = True
    else:
        both_tri_nonsymm = True

    # Initialize the matrix with zeros
    M = cp.zeros(matrix_shape) 

    thread_x = 32
    thread_y = 16

    if stream is None:
        device = 0
        cp.cuda.Device(device).use()
        cp_stream = cp.cuda.Stream(non_blocking = True)
        nb_stream = cuda.external_stream(cp_stream.ptr)
    else:
        nb_stream = stream

    blocks_per_grid = ((num_rows + (thread_x - 1))//thread_x, (num_cols + (thread_y - 1))//thread_y) 
    dipole_moment_mat_symm_internal_cuda[blocks_per_grid, (thread_x, thread_y), nb_stream](bfs_coords[0], bfs_contr_prim_norms[0], bfs_lmn[0], bfs_nprim[0], bfs_coeffs, bfs_prim_norms, bfs_expnts, a, b, c, d, lower_tri, upper_tri, both_tri_symm, both_tri_nonsymm, origin, M)
    if both_tri_symm:
        symmetrize[blocks_per_grid, (thread_x, thread_y), nb_stream](a,b,c,d,M)
    cuda.synchronize()
    return M 



# @cuda.jit(device=True )
# def hermite_gauss_coeff(i,j,t,Qx,a,b,p,q): 
#     ''' Recursive definition of Hermite Gaussian coefficients.
#         Returns a float.
#         Source: https://github.com/jjgoings/McMurchie-Davidson/blob/master/mmd/integrals/reference.py
#         a: orbital exponent on Gaussian 'a' (e.g. alpha in the text)
#         b: orbital exponent on Gaussian 'b' (e.g. beta in the text)
#         i,j: orbital angular momentum number on Gaussian 'a' and 'b'
#         t: number nodes in Hermite (depends on type of integral, 
#            e.g. always zero for overlap integrals)
#         Qx: distance between origins of Gaussian 'a' and 'b'
#     '''
#     if p is None:
#         p = a + b
#     if q is None:
#         q = a*b/p
#     if (t < 0) or (t > (i + j)):
#         # out of bounds for t  
#         return 0.0
#     elif i == j == t == 0:
#         # base case
#         return math.exp(-q*Qx*Qx) # K_AB
#     elif j == 0:
#         # decrement index i
#         return (1/(2*p))*hermite_gauss_coeff(i-1,j,t-1,Qx,a,b,p,q) - (q*Qx/a)*hermite_gauss_coeff(i-1,j,t,Qx,a,b,p,q) + (t+1)*hermite_gauss_coeff(i-1,j,t+1,Qx,a,b,p,q)
#     else:
#         # decrement index j
#         return (1/(2*p))*hermite_gauss_coeff(i,j-1,t-1,Qx,a,b,p,q) + (q*Qx/b)*hermite_gauss_coeff(i,j-1,t,Qx,a,b,p,q) + (t+1)*hermite_gauss_coeff(i,j-1,t+1,Qx,a,b,p,q)

#@cuda.jit(['float64(int16, int16, int16, float64, float64, float64)'], device=True)
@cuda.jit(device=True)
def hermite_gauss_coeff2(i, j, t, Qx, a, b):

    p = a + b  
    q = a*b/p
    
    if (t < 0) or (t > (i + j)):
        # out of bounds for t  
        return 0.0
    elif i == j == t == 0:
        # base case
        return math.exp(-q*Qx*Qx) # K_AB
    else:    
        coeff = 0.0
        coeffs = cuda.local.array((25, 25, 50), numba.float64)
        coeffs[:,:,:] = 0.0
        coeffs[0, 0, 0]  = math.exp(-q*Qx*Qx)
        # for jj in range(j, -1, -1):
        #     for ii in range(i, -1, -1): 
        tt = t
        for ii in range(i, -1, -1):
            for jj in range(j, -1, -1):
        # for ii in range(0, i+1, 1):
        #     for jj in range(0, j+1, 1):
                # for tt in range(max(0, t-i-j), min(i+j, t)+1):
                # if (tt < 0) or (tt > (ii-1 + jj)):
                #     coeffs[ii-1, jj, tt] = 0.0
                # if (tt < 0) or (tt > (ii + jj-1)):
                #     coeffs[ii, jj-1, tt] = 0.0
                # if ((tt-1) < 0) or ((tt-1) > (ii-1 + jj)):
                #     coeffs[ii-1, jj, tt-1] = 0.0
                # if ((tt-1) < 0) or ((tt-1) > (ii + jj-1)):
                #     coeffs[ii, jj-1, tt-1] = 0.0
                # if ((tt+1) < 0) or ((tt+1) > (ii-1 + jj)):
                #     coeffs[ii-1, jj, tt+1] = 0.0
                # if ((tt+1) < 0) or ((tt+1) > (ii + jj-1)):
                #     coeffs[ii, jj-1, tt+1] = 0.0
                # if ii == 0 and jj == 0 and tt == 0:
                if ii == jj == tt == 0:
                    coeff = coeffs[ii, jj, tt]
                elif jj != 0:
                    coeff += (1/(2*p)) * coeffs[ii, jj-1, tt-1]
                    coeff += (q*Qx/b) * coeffs[ii, jj-1, tt]
                    coeff += (tt+1) * coeffs[ii, jj-1, tt+1]
                elif jj == 0:
                    coeff += (1/(2*p)) * coeffs[ii-1, jj, tt-1]
                    coeff -= (q*Qx/a) * coeffs[ii-1, jj, tt]  
                    coeff += (tt+1) * coeffs[ii-1, jj, tt+1]
                # elif ii==0:
                #     coeff += (1/(2*p)) * coeffs[ii, jj-1, tt-1]
                #     coeff += (q*Qx/b) * coeffs[ii, jj-1, tt]
                #     coeff += (tt+1) * coeffs[ii, jj-1, tt+1]
                # elif jj == 0:
                #     coeff += (1/(2*p)) * coeffs[ii-1, jj, tt-1]
                #     coeff -= (q*Qx/a) * coeffs[ii-1, jj, tt]  
                #     coeff += (tt+1) * coeffs[ii-1, jj, tt+1]
                    
                coeffs[ii, jj, tt] = coeff
                coeff = 0.0
                            
        return coeffs[i, j, t]
# @cuda.jit(device=True)
# def hermite_gauss_coeff2(i, j, t, Qx, a, b):
    
#     p = a + b
#     q = a * b / p
#     if (t < 0) or (t > (i + j)):
#         return 0.0
#     elif i == j == t == 0:
#         return math.exp(-q * Qx * Qx)  # K_AB
#     else:
#         result = 0.0
#         if j == 0:
#             for k in range(t + 1):
#                 term = (1 / (2 * p)) if k == 0 else (k + 1)
#                 coeff = hermite_gauss_coeff2(i - 1, j, k, Qx, a, b)
#                 result += term * coeff
#                 result -= (q * Qx / a) if k > 0 else 0.0
#                 result *= coeff
#         else:
#             for k in range(t + 1):
#                 term = (1 / (2 * p)) if k == 0 else (k + 1)
#                 coeff = hermite_gauss_coeff2(i, j - 1, k, Qx, a, b)
#                 result += term * coeff
#                 result += (q * Qx / b) if k > 0 else 0.0
#                 result *= coeff
#         return result
# def hermite_gauss_coeff2(i,j,t,Qx,a,b): 
#     ''' Recursive definition of Hermite Gaussian coefficients.
#         Returns a float.
#         Source: https://github.com/jjgoings/McMurchie-Davidson/blob/master/mmd/integrals/reference.py
#         a: orbital exponent on Gaussian 'a' (e.g. alpha in the text)
#         b: orbital exponent on Gaussian 'b' (e.g. beta in the text)
#         i,j: orbital angular momentum number on Gaussian 'a' and 'b'
#         t: number nodes in Hermite (depends on type of integral, 
#            e.g. always zero for overlap integrals)
#         Qx: distance between origins of Gaussian 'a' and 'b'
#     '''
#     p = a + b
#     q = a*b/p
#     if (t < 0) or (t > (i + j)):
#         # out of bounds for t  
#         return 0.0
#     elif i == j == t == 0:
#         # base case
#         return math.exp(-q*Qx*Qx) # K_AB
    
#     elif j == 0:
#         # decrement index i
#         temp1 = (1/(2*p))*hermite_gauss_coeff2(i-1,j,t-1,Qx,a,b)
#         temp2 = (q*Qx/a)*hermite_gauss_coeff2(i-1,j,t,Qx,a,b)
#         temp3 = (t+1)*hermite_gauss_coeff2(i-1,j,t+1,Qx,a,b)
#         return temp1 - temp2 + temp3
#     else:
#         # decrement index j
#         return (1/(2*p))*hermite_gauss_coeff2(i,j-1,t-1,Qx,a,b) + (q*Qx/b)*hermite_gauss_coeff2(i,j-1,t,Qx,a,b) + (t+1)*hermite_gauss_coeff2(i,j-1,t+1,Qx,a,b)
#     return 0.0

@cuda.jit(fastmath=True, cache=True, max_registers=8)#(device=True)
def dipole_moment_mat_symm_internal_cuda(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts, start_row, end_row, start_col, end_col, lower_tri, upper_tri, both_tri_symm, both_tri_nonsymm, origin, out):
    # This function calculates the dipole moment matrix and uses the symmetry property to only calculate half-ish the elements
    # and get the remaining half by symmetry.
    # The reason we need this extra function is because we want the callable function to be simple and not require so many 
    # arguments. But when using Numba to optimize, we can't have too many custom objects and stuff. Numba likes numpy arrays
    # so passing those is okay. But lists and custom objects are not okay.
    # This function calculates the dipole moment matrix for a given basis object.
    # The basis object holds the information of basis functions like: exponents, coeffs, etc.
    # It is possible to only calculate a slice (block) of the complete matrix.
    # slice is a 4 element list whose first and second elements give the range of the rows to be calculated.
    # the third and fourth element give the range of columns to be calculated.
    # The integrals are performed using the formulas here https://github.com/jjgoings/McMurchie-Davidson/blob/master/mmd/integrals/reference.py
    PI = 3.141592653589793

    i, j = cuda.grid(2)
    if i>=start_row and i<end_row and j>=start_col and j<end_col:
    # if i<end_row and j<end_col:
        I = bfs_coords[i]
        Ni = bfs_contr_prim_norms[i]
        lmni = bfs_lmn[i]
        la, ma, na = lmni
        if lower_tri or upper_tri or (both_tri_symm and j<=i) or both_tri_nonsymm:
            J = bfs_coords[j]
            # IJ = I - J  
            IJ = cuda.local.array((3), numba.float64) 
            P = cuda.local.array((3), numba.float64) 
            PC = cuda.local.array((3), numba.float64)  
            IJ[0] = I[0] - J[0]
            IJ[1] = I[1] - J[1]
            IJ[2] = I[2] - J[2]
            result_x = 0.0
            result_y = 0.0
            result_z = 0.0
                
            J = bfs_coords[j]
            IJsq = IJ[0]**2 + IJ[1]**2 + IJ[2]**2
            
            Nj = bfs_contr_prim_norms[j]
            
            lmnj = bfs_lmn[j]
                
            lb, mb, nb = lmnj
            #Loop over primitives
            for ik in range(bfs_nprim[i]):     #Parallelising over primitives doesn't seem to make a difference
                alphaik = bfs_expnts[i,ik]
                dik = bfs_coeffs[i,ik]
                Nik = bfs_prim_norms[i,ik]
                for jk in range(bfs_nprim[j]):
                    
                    alphajk = bfs_expnts[j,jk]
                    gamma = alphaik + alphajk
                    gamma_inv = 1/gamma
                    temp_gamma = alphaik*alphajk*gamma_inv
                    screenfactor = math.exp(-temp_gamma*IJsq)
                    if abs(screenfactor)<1.0e-8:   
                    # Going lower than E-8 doesn't change the max erroor. There could still be some effects from error compounding but the max error doesnt budge.
                    #TODO: This is quite low. But since this is the slowest part.
                    #But I had to do this because this is a very slow part of the program.
                    #Will have to check how the accuracy is affected and if the screening factor
                    #can be reduced further.
                        continue

                    
                    djk = bfs_coeffs[j,jk] 
                    
                    Njk = bfs_prim_norms[j,jk]
                    
                    # epsilon = 0.25*gamma_inv
                    # P = (alphaik*I + alphajk*J)*gamma_inv
                    P[0] = (alphaik*I[0] + alphajk*J[0])/gamma
                    P[1] = (alphaik*I[1] + alphajk*J[1])/gamma
                    P[2] = (alphaik*I[2] + alphajk*J[2])/gamma
                    
                    # PC = P - origin
                    PC[0] = P[0] - origin[0]
                    PC[1] = P[1] - origin[1]
                    PC[2] = P[2] - origin[2]
                    
                    # tempfac = (PIx2*gamma_inv)

                    norm_contr_factor = dik*djk*Nik*Njk*Ni*Nj

                    D_x  = hermite_gauss_coeff2(la,lb,1,IJ[0],alphaik,alphajk) + PC[0]*hermite_gauss_coeff2(la,lb,0,IJ[0],alphaik,alphajk)
                    #    D  = E(l1,l2,0,A[0]-B[0],a,b,1+n[0],A[0]-gOrigin[0])
                    S2_x = hermite_gauss_coeff2(ma,mb,0,IJ[1],alphaik,alphajk)
                    S3_x = hermite_gauss_coeff2(na,nb,0,IJ[2],alphaik,alphajk)
                    result_x += D_x*S2_x*S3_x*math.pow(PI/gamma,1.5)*norm_contr_factor
                
                    
                    S1_y = hermite_gauss_coeff2(la,lb,0,IJ[0],alphaik,alphajk)
                    D_y  = hermite_gauss_coeff2(ma,mb,1,IJ[1],alphaik,alphajk) + PC[1]*hermite_gauss_coeff2(ma,mb,0,IJ[1],alphaik,alphajk)
                    #    D  = E(m1,m2,0,A[1]-B[1],a,b,1+n[1],A[1]-gOrigin[1])
                    S3_y = hermite_gauss_coeff2(na,nb,0,IJ[2],alphaik,alphajk)
                    result_y += S1_y*D_y*S3_y*math.pow(PI/(gamma),1.5)*norm_contr_factor
            
                    
                    S1_z = hermite_gauss_coeff2(la,lb,0,IJ[0],alphaik,alphajk)
                    S2_z = hermite_gauss_coeff2(ma,mb,0,IJ[1],alphaik,alphajk)
                    D_z  = hermite_gauss_coeff2(na,nb,1,IJ[2],alphaik,alphajk) + PC[2]*hermite_gauss_coeff2(na,nb,0,IJ[2],alphaik,alphajk)
                    #    D  = E(n1,n2,0,A[2]-B[2],a,b,1+n[2],A[2]-gOrigin[2]) 
                    result_z += S1_z*S2_z*D_z*math.pow(PI/(gamma),1.5)*norm_contr_factor
            

            out[0, i - start_row, j - start_col] = result_x
            out[1, i - start_row, j - start_col] = result_y
            out[2, i - start_row, j - start_col] = result_z
            
                

@cuda.jit(fastmath=True, cache=True)#(device=True)           
def symmetrize(start_row, end_row, start_col, end_col, out):
    #T + T.T - cp.diag(cp.diag(T))
    i, j = cuda.grid(2)
    if i>=start_row and i<end_row and j>=start_col and j<end_col:
        if j>i:
            out[0, i-start_row, j-start_col] = out[0, j-start_col, i-start_row]
            out[1, i-start_row, j-start_col] = out[1, j-start_col, i-start_row]
            out[2, i-start_row, j-start_col] = out[2, j-start_col, i-start_row]
