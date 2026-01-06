import numpy as np
from numba import njit, prange, jit
import numba
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

def eval_bfs_and_grad(basis, coord, deriv=1, parallel=True, non_zero_indices=None):

        #We convert the required properties to numpy arrays as this is what Numba likes.
        bfs_coords = np.array([basis.bfs_coords])
        bfs_contr_prim_norms = np.array([basis.bfs_contr_prim_norms])
        bfs_lmn = np.array([basis.bfs_lmn])
        bfs_nprim = np.array([basis.bfs_nprim])

           

        #The remaining properties like bfs_coeffs are a list of lists of unequal sizes.
        #Numba won't be able to work with these efficiently.
        #So, we convert them to a numpy 2d array by applying a trick,
        #that the second dimension is that of the largest list. So that
        #it can accomodate all the lists.
        maxnprim = max(basis.bfs_nprim)
        bfs_coeffs = np.zeros([basis.bfs_nao, maxnprim])
        bfs_expnts = np.zeros([basis.bfs_nao, maxnprim])
        bfs_prim_norms = np.zeros([basis.bfs_nao, maxnprim])
        bfs_radius_cutoff = np.zeros([basis.bfs_nao])
        for i in range(basis.bfs_nao):
            for j in range(basis.bfs_nprim[i]):
                bfs_coeffs[i,j] = basis.bfs_coeffs[i][j]
                bfs_expnts[i,j] = basis.bfs_expnts[i][j]
                bfs_prim_norms[i,j] = basis.bfs_prim_norms[i][j]
                bfs_radius_cutoff[i] = basis.bfs_radius_cutoff[i]

        if parallel:
            # Uses the same number of threads as the defined by numba.set_num_threads before calling this function
            if non_zero_indices is not None:
                bf_values, bf_grad_values = eval_bfs_and_grad_sparse_internal(bfs_coords[0], bfs_contr_prim_norms[0], bfs_nprim[0], bfs_lmn[0], bfs_coeffs, bfs_prim_norms, bfs_expnts, coord, non_zero_indices)
            else:
                bf_values, bf_grad_values = eval_bfs_and_grad_internal(bfs_coords[0], bfs_contr_prim_norms[0], bfs_nprim[0], bfs_lmn[0], bfs_coeffs, bfs_prim_norms, bfs_expnts, bfs_radius_cutoff, coord)
        else:
            numba.set_num_threads(1) # Set number of threads to 1
            if non_zero_indices is not None:
                bf_values, bf_grad_values = eval_bfs_and_grad_sparse_internal(bfs_coords[0], bfs_contr_prim_norms[0], bfs_nprim[0], bfs_lmn[0], bfs_coeffs, bfs_prim_norms, bfs_expnts, coord, non_zero_indices)
            else:
                bf_values, bf_grad_values = eval_bfs_and_grad_internal(bfs_coords[0], bfs_contr_prim_norms[0], bfs_nprim[0], bfs_lmn[0], bfs_coeffs, bfs_prim_norms, bfs_expnts, bfs_radius_cutoff, coord)
        
            
        # return bf_grad_values
        return bf_values, bf_grad_values


def eval_bfs(basis, coord, parallel=True, non_zero_indices=None):

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
    bfs_radius_cutoff = np.zeros([basis.bfs_nao])
    for i in range(basis.bfs_nao):
        for j in range(basis.bfs_nprim[i]):
            bfs_coeffs[i,j] = basis.bfs_coeffs[i][j]
            bfs_expnts[i,j] = basis.bfs_expnts[i][j]
            bfs_prim_norms[i,j] = basis.bfs_prim_norms[i][j]
            bfs_radius_cutoff[i] = basis.bfs_radius_cutoff[i]

    if parallel:
        # Uses the same number of threads as the defined by numba.set_num_threads before calling this function
        if non_zero_indices is not None:
            bf_values = eval_bfs_sparse_internal(bfs_coords[0], bfs_contr_prim_norms[0], bfs_nprim[0], bfs_lmn[0], bfs_coeffs, bfs_prim_norms, bfs_expnts, coord, non_zero_indices)
        else:
            bf_values = eval_bfs_internal(bfs_coords[0], bfs_contr_prim_norms[0], bfs_nprim[0], bfs_lmn[0], bfs_coeffs, bfs_prim_norms, bfs_expnts, bfs_radius_cutoff, coord)
    else:
        numba.set_num_threads(1) # Set number of threads to 1
        if non_zero_indices is not None:
            bf_values = eval_bfs_sparse_internal(bfs_coords[0], bfs_contr_prim_norms[0], bfs_nprim[0], bfs_lmn[0], bfs_coeffs, bfs_prim_norms, bfs_expnts, coord, non_zero_indices)
        else:
            bf_values = eval_bfs_internal(bfs_coords[0], bfs_contr_prim_norms[0], bfs_nprim[0], bfs_lmn[0], bfs_coeffs, bfs_prim_norms, bfs_expnts, bfs_radius_cutoff, coord)
    
    
    return bf_values

@njit(parallel=True, cache=True, fastmath=True, error_model="numpy", inline='always', nogil=True)
def eval_rho(bf_values, densmat):
    # Evaluates the value of density on a grid using the values of the basis functions 
    # at those grid points and the density matrix.
    # rho at the grid point m is given as:
    # \rho^m = \sum_{\mu}\sum_{\nu} D_{\mu \nu} \mu^m \nu^m
    rho = np.zeros((bf_values.shape[0]))
    ncoords = bf_values.shape[0] 
    nbfs = bf_values.shape[1]

    # The following is the fastest because here we utilize symmetry information and also skip calculation for small values
    #Loop over BFs
    for m in prange(ncoords):
        rho_temp = 0.0
        for i in prange(nbfs):
            mu = bf_values[m,i]
            if abs(mu)<1.0e-8: #A value of 8 is good for an accuracy of 7-8 decimal places.
                    continue
            for j in prange(i+1):
                dens = densmat[i,j]
                if abs(dens)<1.0e-8: #A value of 9 is good for an accuracy of 7-8 decimal places.
                    continue
                if i==j: # Diagonal terms
                    nu = mu
                    rho_temp += dens*mu*nu 
                else: # Non-diagonal terms
                    nu = bf_values[m,j]
                    if abs(nu)<1.0e-8: #A value of 8 is good for an accuracy of 7-8 decimal places.
                        continue
                    rho_temp += 2*dens*mu*nu 
        rho[m] = rho_temp
    return rho 

@njit(parallel=False, cache=True, fastmath=True, error_model="numpy", nogil=True, inline='always')
def eval_gto(alpha, coeff, lmn, x, y, z, exponent_dist_sq):
    # This function evaluates the value of a given Gaussian primitive 
    # with given values of alpha (exponent), coefficient, and angular momentum.
    # x,y,z contain the information of both grid point and bf center
    # x = coord_grid[0] - coord_bf_center[0]
    # y = coord_grid[1] - coord_bf_center[1]
    # z = coord_grid[2] - coord_bf_center[2]
    
    xl = x**lmn[0]
    ym = y**lmn[1]
    zn = z**lmn[2]
    exp = math.exp(-alpha*exponent_dist_sq)
    value = coeff*xl*ym*zn*exp
    
    return value


@njit(parallel=True, cache=True, nogil=True, fastmath=True, error_model="numpy")
def eval_bfs_internal(bfs_coords, bfs_contr_prim_norms, bfs_nprim, bfs_lmn, bfs_coeffs, bfs_prim_norms, bfs_expnts, bfs_radius_cutoff, coord):
    #This function evaluates the value of all the given Basis functions on the grid (coord).
    # 'coord' should be a nx3 array
        
    nao = bfs_coords.shape[0]
    ncoord = coord.shape[0]
    result = np.zeros((ncoord, nao))

    #Loop over grid points
    for k in prange(ncoord):
        coord_grid = coord[k]
        #Loop over BFs
        for i in range(nao):
            value = 0.0
            coord_bf = bfs_coords[i]
            x = coord_grid[0]-coord_bf[0]
            y = coord_grid[1]-coord_bf[1]
            z = coord_grid[2]-coord_bf[2]
            if (np.sqrt(x**2+y**2+z**2)>bfs_radius_cutoff[i]):
                continue
            Ni = bfs_contr_prim_norms[i]
            lmni = bfs_lmn[i]
            exponent_dist_sq = x**2 + y**2 + z**2
            for ik in range(bfs_nprim[i]):
                dik = bfs_coeffs[i][ik] 
                Nik = bfs_prim_norms[i][ik]
                alphaik = bfs_expnts[i][ik]
                value += eval_gto(alphaik, Ni*Nik*dik, lmni, x, y, z, exponent_dist_sq)
            result[k,i] = value

    return result

@njit(parallel=True, cache=True, nogil=True, fastmath=True, error_model="numpy")
def eval_bfs_sparse_internal(bfs_coords, bfs_contr_prim_norms, bfs_nprim, bfs_lmn, bfs_coeffs, bfs_prim_norms, bfs_expnts, coord, bf_indices):
    # This function evaluates the values of the specific (significant) basis functions (bf_indices) on a set of grid points (coord).
    # 'coord' should be a nx3 array

    # This is essentially a "sparse" version of eval_bfs function. Therefore, instead of calculating the values of all basis functions
    # it only calculates the values of basis functions that are provided as list of indices (bf_indices)
    # The bf_indices list is generated by considering the extents of the basis functions
        
    nao = bf_indices.shape[0]
    ncoord = coord.shape[0]
    result = np.zeros((ncoord, nao))

    #Loop over grid points
    for k in prange(ncoord):
        coord_grid = coord[k]
        #Loop over BFs
        for i in prange(nao):
            # Actual index in original basis set
            ibf = bf_indices[i]
            value = 0.0
            coord_bf = bfs_coords[ibf]
            x = coord_grid[0]-coord_bf[0]
            y = coord_grid[1]-coord_bf[1]
            z = coord_grid[2]-coord_bf[2]
            Ni = bfs_contr_prim_norms[ibf]
            lmni = bfs_lmn[ibf]
            exponent_dist_sq = x**2 + y**2 + z**2
            for ik in range(bfs_nprim[ibf]):
                dik = bfs_coeffs[ibf][ik] 
                Nik = bfs_prim_norms[ibf][ik]
                alphaik = bfs_expnts[ibf][ik]
                value += eval_gto(alphaik, Ni*Nik*dik, lmni, x, y, z, exponent_dist_sq)
            result[k,i] = value

    return result


@njit(parallel=True, cache=True, nogil=True, fastmath=True, error_model="numpy")
def eval_bfs_and_grad_internal(bfs_coords, bfs_contr_prim_norms, bfs_nprim, bfs_lmn, bfs_coeffs, bfs_prim_norms, bfs_expnts, bfs_radius_cutoff, coord):
    #This function evaluates the value of all the given Basis functions on the grid (coord).
    # 'coord' should be a nx3 array
        
    nao = bfs_coords.shape[0]
    ncoord = coord.shape[0]
    # result = np.zeros((4, ncoord, nao))
    result1 = np.zeros((ncoord,nao))
    result2 = np.zeros((3,ncoord,nao))

    #Loop over grid points
    for k in prange(ncoord):
        coord_grid = coord[k]
        #Loop over BFs
        for i in range(nao):
            value_ao = 0.0
            valuex = 0.0
            valuey = 0.0
            valuez = 0.0
            # values[0] = 0.0
            # values[1] = 0.0
            # values[2] = 0.0
            # values[3] = 0.0
            coord_bf = bfs_coords[i]
            x = coord_grid[0]-coord_bf[0]
            y = coord_grid[1]-coord_bf[1]
            z = coord_grid[2]-coord_bf[2]
            if (np.sqrt(x**2+y**2+z**2)>bfs_radius_cutoff[i]):
                continue
            Ni = bfs_contr_prim_norms[i]
            lmni = bfs_lmn[i]
            exponent_dist_sq = x**2 + y**2 + z**2
            #cutoff_radius = 0 #Cutoff radius for this basis function
            for ik in range(bfs_nprim[i]):
                dik = bfs_coeffs[i, ik] 
                Nik = bfs_prim_norms[i, ik]
                alphaik = bfs_expnts[i, ik]
                a,b,c,d = eval_gto_and_grad(alphaik, Ni*Nik*dik, lmni, x, y, z, exponent_dist_sq)
                value_ao = value_ao + a
                valuex += b
                valuey += c
                valuez += d
            result1[k,i] = value_ao
            result2[0,k,i] = valuex
            result2[1,k,i] = valuey
            result2[2,k,i] = valuez
            
    return result1, result2

@njit(parallel=True, cache=True, nogil=True, fastmath=True, error_model="numpy")
def eval_bfs_and_grad_sparse_internal(bfs_coords, bfs_contr_prim_norms, bfs_nprim, bfs_lmn, bfs_coeffs, bfs_prim_norms, bfs_expnts, coord, bf_indices):
    #This function evaluates the value of all the given Basis functions on the grid (coord).
    # 'coord' should be a nx3 array
        
    nao = bf_indices.shape[0]
    ncoord = coord.shape[0]
    # result = np.zeros((4, ncoord, nao))
    result1 = np.zeros((ncoord,nao))
    result2 = np.zeros((3,ncoord,nao))

    #Loop over grid points
    for k in prange(ncoord):
        coord_grid = coord[k]
        #Loop over BFs
        for i in range(nao):
            value_ao = 0.0
            valuex = 0.0
            valuey = 0.0
            valuez = 0.0
            ibf = bf_indices[i]
            coord_bf = bfs_coords[ibf]
            x = coord_grid[0]-coord_bf[0]
            y = coord_grid[1]-coord_bf[1]
            z = coord_grid[2]-coord_bf[2]
            Ni = bfs_contr_prim_norms[ibf]
            lmni = bfs_lmn[ibf]
            exponent_dist_sq = x**2 + y**2 + z**2
            for ik in range(bfs_nprim[ibf]): # Loop over primitives
                dik = bfs_coeffs[ibf, ik] 
                Nik = bfs_prim_norms[ibf, ik]
                alphaik = bfs_expnts[ibf, ik]
                a,b,c,d = eval_gto_and_grad(alphaik, Ni*Nik*dik, lmni, x, y, z, exponent_dist_sq)
                value_ao += a
                valuex += b
                valuey += c
                valuez += d
            result1[k,i] = value_ao
            result2[0,k,i] = valuex
            result2[1,k,i] = valuey
            result2[2,k,i] = valuez
            
    return result1, result2

@njit(parallel=False, cache=True, nogil=True, fastmath=True, error_model="numpy")
def eval_bfs_and_grad_sparse_internal_serial(bfs_coords, bfs_contr_prim_norms, bfs_nprim, bfs_lmn, bfs_coeffs, bfs_prim_norms, bfs_expnts, coord, bf_indices):
    #This function evaluates the value of all the given Basis functions on the grid (coord).
    # 'coord' should be a nx3 array
        
    nao = bf_indices.shape[0]
    ncoord = coord.shape[0]
    # result = np.zeros((4, ncoord, nao))
    result1 = np.zeros((ncoord,nao))
    result2 = np.zeros((3,ncoord,nao))

    #Loop over grid points
    for k in prange(ncoord):
        coord_grid = coord[k]
        #Loop over BFs
        for i in range(nao):
            value_ao = 0.0
            valuex = 0.0
            valuey = 0.0
            valuez = 0.0
            ibf = bf_indices[i]
            coord_bf = bfs_coords[ibf]
            x = coord_grid[0]-coord_bf[0]
            y = coord_grid[1]-coord_bf[1]
            z = coord_grid[2]-coord_bf[2]
            Ni = bfs_contr_prim_norms[ibf]
            lmni = bfs_lmn[ibf]
            exponent_dist_sq = x**2 + y**2 + z**2
            for ik in range(bfs_nprim[ibf]): # Loop over primitives
                dik = bfs_coeffs[ibf, ik] 
                Nik = bfs_prim_norms[ibf, ik]
                alphaik = bfs_expnts[ibf, ik]
                a,b,c,d = eval_gto_and_grad(alphaik, Ni*Nik*dik, lmni, x, y, z, exponent_dist_sq)
                value_ao += a
                valuex += b
                valuey += c
                valuez += d
            result1[k,i] = value_ao
            result2[0,k,i] = valuex
            result2[1,k,i] = valuey
            result2[2,k,i] = valuez
            
    return result1, result2


@njit(parallel=False, cache=True, fastmath=True, error_model="numpy", nogil=True, inline='always')
def eval_gto_and_grad(alpha, coeff, lmn, x, y, z, exponent_dist_sq):
    # https://www.wolframalpha.com/input?i=dy%2FdA+for+y%3D%28x-A%29%5E%28l%29*%28y-B%29%5E%28m%29*%28z-C%29%5E%28n%29*exp%28-alpha*%28%28x-A%29%5E%282%29%2B%28y-B%29%5E%282%29%2B%28z-C%29%5E%282%29%29+
    # https://www.wolframalpha.com/input?i=derivative+of+%28x-A%29%5E%28l%29*%28y-B%29%5E%28m%29*%28z-C%29%5E%28n%29*exp%28-alpha*%28%28x-A%29%5E%282%29%2B%28y-B%29%5E%282%29%2B%28z-C%29%5E%282%29%29
    # A very low-level way to calculate the ao values as well as their gradients simultaneously, without 
    # running similar calls again and again.
    # value = np.zeros((4))
    # Prelims
    # x = coord[0]-coordCenter[0]
    # y = coord[1]-coordCenter[1]
    # z = coord[2]-coordCenter[2]
    xl = x**lmn[0]
    ym = y**lmn[1]
    zn = z**lmn[2]
    exp = math.exp(-alpha*(exponent_dist_sq))
    factor2 = coeff*exp

    # AO Value
    value0 = factor2*xl*ym*zn
    # Grad x
    if np.abs(x-0)<1e-14:
        value1 = 0.0
    else:
        xl = x**(lmn[0]-1)
        factor = (lmn[0]-2*alpha*x**2)
        value1 = factor2*xl*ym*zn*factor
    # Grad y
    if np.abs(y-0)<1e-14:
        value2 = 0.0
    else:
        xl = x**lmn[0]
        ym = y**(lmn[1]-1)
        factor = (lmn[1]-2*alpha*y**2)
        value2 = factor2*xl*ym*zn*factor 
    # Grad z 
    if np.abs(z-0)<1e-14:
        value3 = 0.0
    else:
        zn = z**(lmn[2]-1)
        xl = x**lmn[0]
        ym = y**lmn[1]  
        factor = (lmn[2]-2*alpha*z**2)
        value3 = factor2*xl*ym*zn*factor
        
    return value0, value1, value2, value3

@njit(parallel=True, cache=True, nogil=True, fastmath=True, error_model="numpy")
def eval_bfs_sparse_vectorized_internal(bfs_coords, bfs_contr_prim_norms, bfs_nprim, bfs_lmn, bfs_coeffs, bfs_prim_norms, bfs_expnts, coord, bf_indices):
    # This function evaluates the values of the specific (significant) basis functions (bf_indices) on a set of grid points (coord).
    # 'coord' should be a nx3 array

    # This is essentially a "sparse" version of eval_bfs function. Therefore, instead of calculating the values of all basis functions
    # it only calculates the values of basis functions that are provided as list of indices (bf_indices)
    # The bf_indices list is generated by considering the extents of the basis functions
        
    nao = bf_indices.shape[0]
    ncoord = coord.shape[0]
    result = np.zeros((ncoord, nao))

    for i in prange(nao):
        # Actual index in original basis set
        ibf = bf_indices[i]
        coord_bf = bfs_coords[ibf]
        # x, y, z = coord - coord_bf
        x = coord[:,0] - coord_bf[0]
        y = coord[:,1] - coord_bf[1]
        z = coord[:,2] - coord_bf[2]
        Ni = bfs_contr_prim_norms[ibf]
        lmni = bfs_lmn[ibf]
        exp_sq_term = x**2 + y**2 + z**2
        xl = x**lmni[0]
        ym = y**lmni[1]
        zn = z**lmni[2]
        value = np.zeros(coord.shape[0], dtype=np.float64)
        for ik in range(bfs_nprim[ibf]):
            dik = bfs_coeffs[ibf][ik] 
            Nik = bfs_prim_norms[ibf][ik]
            alphaik = bfs_expnts[ibf][ik]
            value += eval_gto_vectorize(alphaik, Ni*Nik*dik, exp_sq_term, xl, ym, zn)
        result[:,i] = value

    return result

@njit(parallel=False, cache=True, fastmath=True, error_model="numpy", nogil=True, inline='always')
def eval_gto_vectorize(alpha, coeff, exp_sq_term, xl, ym, zn):
    # This function evaluates the value of a given Gaussian primitive 
    # with given values of alpha (exponent), coefficient, and angular momentum.
    exp = np.exp(-alpha*(exp_sq_term))
    value = coeff*xl*ym*zn*exp
    
    return value
# import cupyx
# @jit(nopython=False, parallel=True, forceobj=True, cache=True)
def eval_bfs_sparse_vectorized_internal_cupy(bfs_coords, bfs_contr_prim_norms, bfs_nprim, bfs_lmn, bfs_coeffs, bfs_prim_norms, \
                                             bfs_expnts, coord, bf_indices, scratches):
    # This function evaluates the values of the specific (significant) basis functions (bf_indices) on a set of grid points (coord).
    # 'coord' should be a nx3 array

    # This is essentially a "sparse" version of eval_bfs function. Therefore, instead of calculating the values of all basis functions
    # it only calculates the values of basis functions that are provided as list of indices (bf_indices)
    # The bf_indices list is generated by considering the extents of the basis functions
    # with cupyx.profiler.profile():
    nao = bf_indices.shape[0]
    ncoord = coord.shape[0]
    result = cp.zeros((ncoord, nao))

    value = scratches[0][0:ncoord]
    x = scratches[1][0:ncoord]
    y = scratches[2][0:ncoord]
    z = scratches[3][0:ncoord]
    exp_sq_term = scratches[4][0:ncoord]
    xlymzn_ = scratches[5][0:ncoord]

    for i in range(nao):
        # Actual index in original basis set
        ibf = bf_indices[i]
        coord_bf = bfs_coords[ibf]
        Ni = bfs_contr_prim_norms[ibf]
        lmni = bfs_lmn[ibf]
        # x, y, z = coord - coord_bf
        cp.subtract(coord[:,0], coord_bf[0], out=x)
        cp.subtract(coord[:,1], coord_bf[1], out=y)
        cp.subtract(coord[:,2], coord_bf[2], out=z)
        exp_sq_term[:] = calc_norm(x, y, z)
        # xl = x**lmni[0]
        # ym = y**lmni[1]
        # zn = z**lmni[2]
        xlymzn_[:] = xlymzn(x, y, z, lmni[0], lmni[1], lmni[2])
        value[:] = 0.0
        for ik in range(bfs_nprim[ibf]):
            dik = bfs_coeffs[ibf][ik] 
            Nik = bfs_prim_norms[ibf][ik]
            alphaik = bfs_expnts[ibf][ik]
            cp.add(value, eval_gto_vectorize_cupy(alphaik, Ni*Nik*dik, exp_sq_term, xlymzn_), out=value)
            # result[:,i] += eval_gto_vectorize_cupy(alphaik, Ni*Nik*dik, exp_sq_term, xl, ym, zn)
        result[:,i] = value

    return result

# # IMPORTANT:
# # export NUMBA_CUDA_DRIVER="/usr/lib/wsl/lib/libcuda.so.1"
@cuda.jit(fastmath=True, cache=True)#(device=True)
def eval_bfs_sparse_internal_cuda(bfs_coords, bfs_contr_prim_norms, bfs_nprim, bfs_lmn, bfs_coeffs, 
              bfs_prim_norms, bfs_expnts, coord, bf_indices, out):
    nao = bf_indices.shape[0]
    ncoord = coord.shape[0]
    indx, igrd = cuda.grid(2)
    if indx<nao and igrd<ncoord:
        ibf = bf_indices[indx]
        coord_bf = bfs_coords[ibf]
        coord_grid = coord[igrd]
        Ni = bfs_contr_prim_norms[ibf]
        lmni = bfs_lmn[ibf]
        x = coord_grid[0] - coord_bf[0]
        y = coord_grid[1] - coord_bf[1]
        z = coord_grid[2] - coord_bf[2]
        # x, y, z = coord_grid[:] - coord_bf[:]
        # exp_sq_term = x**2 + y**2 + z**2
        exp_sq_term = x*x + y*y + z*z
        # if exp_sq_term<40:
        xl = x**lmni[0]
        ym = y**lmni[1]
        zn = z**lmni[2]
        xlymzn = xl*ym*zn
        value = 0.0
        for iprim in range(bfs_nprim[ibf]):
            dik = bfs_coeffs[ibf, iprim] 
            Nik = bfs_prim_norms[ibf, iprim]
            alphaik = bfs_expnts[ibf, iprim]
            value += Nik*dik*math.exp(-alphaik*(exp_sq_term))

        out[igrd,indx] = Ni*xlymzn*value

@cuda.jit(fastmath=True, cache=True, max_registers=50)#(device=True)
def eval_bfs_and_grad_sparse_internal_cuda(bfs_coords, bfs_contr_prim_norms, bfs_nprim, bfs_lmn, bfs_coeffs, 
              bfs_prim_norms, bfs_expnts, coord, bf_indices, out_ao, out_ao_grad):
    nao = bf_indices.shape[0]
    ncoord = coord.shape[0]
    indx, igrd = cuda.grid(2)
    if indx<nao and igrd<ncoord:
        ibf = bf_indices[indx]
        coord_bf = bfs_coords[ibf]
        coord_grid = coord[igrd]
        Ni = bfs_contr_prim_norms[ibf]
        lmni = bfs_lmn[ibf]
        x = coord_grid[0] - coord_bf[0]
        y = coord_grid[1] - coord_bf[1]
        z = coord_grid[2] - coord_bf[2]
        # x, y, z = coord_grid[:] - coord_bf[:]
        # exp_sq_term = x**2 + y**2 + z**2
        x2 = x*x
        y2 = y*y
        z2 = z*z
        # exp_sq_term = x*x + y*y + z*z
        exp_sq_term = x2 + y2 + z2
        xl = x**lmni[0]
        ym = y**lmni[1]
        zn = z**lmni[2]
        if abs(x-0.0)<1e-14:
            xl_1 = 0.0
        else:
            xl_1 = x**(lmni[0]-1)
        if abs(y-0.0)<1e-14:
            ym_1 = 0.0
        else:
            ym_1 = y**(lmni[1]-1)
        if abs(z-0.0)<1e-14:
            zn_1 = 0.0
        else:
            zn_1 = z**(lmni[2]-1)
        # xl_1 = x**(lmni[0]-1)
        # ym_1 = y**(lmni[1]-1)
        # zn_1 = z**(lmni[2]-1)
        xlymzn = xl*ym*zn
        xl_1ymzn = xl_1*ym*zn
        xlym_1zn = xl*ym_1*zn
        xlymzn_1 = xl*ym*zn_1
        value0 = 0.0
        value1 = 0.0
        value2 = 0.0
        value3 = 0.0
        for iprim in range(bfs_nprim[ibf]):
            dik = bfs_coeffs[ibf, iprim] 
            Nik = bfs_prim_norms[ibf, iprim]
            alphaik = bfs_expnts[ibf, iprim]
            exp = math.exp(-alphaik*(exp_sq_term))
            factor2 = Ni*Nik*dik*exp

            # AO Value
            value0 += factor2*xlymzn
            # Grad x
            factor = (lmni[0]-2*alphaik*x2)
            value1 += factor2*xl_1ymzn*factor
            # Grad y
            factor = (lmni[1]-2*alphaik*y2)
            value2 += factor2*xlym_1zn*factor 
            # Grad z 
            factor = (lmni[2]-2*alphaik*z2)
            value3 += factor2*xlymzn_1*factor
            # # Grad x
            # if abs(x-0)<1e-14:
            #     value1 = 0.0
            # else:
            #     factor = (lmni[0]-2*alphaik*x2)
            #     value1 += factor2*xl_1ymzn*factor
            # # # Grad y
            # if abs(y-0)<1e-14:
            #     value2 = 0.0
            # else:
            #     factor = (lmni[1]-2*alphaik*y2)
            #     value2 += factor2*xlym_1zn*factor 
            # # Grad z 
            # if abs(z-0)<1e-14:
            #     value3 = 0.0
            # else:
            #     factor = (lmni[2]-2*alphaik*z2)
            #     value3 += factor2*xlymzn_1*factor

        out_ao[igrd,indx] = value0
        out_ao_grad[0,igrd,indx] = value1
        out_ao_grad[1,igrd,indx] = value2
        out_ao_grad[2,igrd,indx] = value3




# @guvectorize([(int64[:], int64, int64[:])], '(n),()->(n)')
# def g(x, y, res):
#     for i in range(x.shape[0]):
#         res[i] = x[i] + y
# from numba import guvectorize, vectorize
# from numba.types import float64, int64
# import cmath

# # IMPORTANT:
# # export NUMBA_CUDA_DRIVER="/usr/lib/wsl/lib/libcuda.so.1"
# @guvectorize(['void(int64, float64, int64, float64[:,:], float64[:,:], float64[:,:], float64[:], float64[:], float64[:])'],'(),(),(m,o),(m,o),(m,o),(n),(n) -> (n)', nopython=True, target='cuda')
# def g(ibf, Ni, nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts, exp_sq_term, xlymzn_, res):
#     for ik in range(nprim):
#         dik = bfs_coeffs[ibf, ik] 
#         Nik = bfs_prim_norms[ibf, ik]
#         alphaik = bfs_expnts[ibf, ik]
        
#         # res[:] += Ni*Nik*dik*xlymzn_*cp.exp(-alphaik*(exp_sq_term))
#         res[:] = cmath.exp((exp_sq_term))

@fuse(kernel_name='calc_norm')
def calc_norm(x,y,z):
    return x**2 + y**2 + z**2

@fuse(kernel_name='xlymzn')
def xlymzn(x, y, z, l, m, n):
    return (x**l) * (y**m) * (z**n)

# @fuse
# def temp(coord, coord_bf, lmni):
#     x = coord[:,0] - coord_bf[0]
#     y = coord[:,1] - coord_bf[1]
#     z = coord[:,2] - coord_bf[2]
#     exp_sq_term = x**2 + y**2 + z**2
#     xl = x**lmni[0]
#     ym = y**lmni[1]
#     zn = z**lmni[2]
#     return xl,ym,zn, exp_sq_term

@fuse(kernel_name='eval_gto_vectorize_cupy')
def eval_gto_vectorize_cupy(alpha, coeff, exp_sq_term, xlymzn):
    # This function evaluates the value of a given Gaussian primitive 
    # with given values of alpha (exponent), coefficient, and angular momentum.
    value = coeff*xlymzn*cp.exp(-alpha*(exp_sq_term))
    # value = (coeff*(xl*ym*zn)[:,cp.newaxis])*cp.exp(-alpha*exp_sq_term[:,cp.newaxis])
    # value = exp
    
    return value

# Define parallelserial versions of the above functions for use in XC term evaluation, where the parallelization will be done over batches of grid points instead
# Seems like these don't work and the number of threads has to be manually set using numba.set_num_threads
# eval_bfs_sparse_internal = njit(parallel=True, cache=True, nogil=True, fastmath=True, error_model="numpy")(eval_bfs_sparse_internal_)
# eval_bfs_internal = njit(parallel=True, cache=True, nogil=True, fastmath=True, error_model="numpy")(eval_bfs_internal_)
# eval_bfs_sparse_internal_serial = njit(parallel=False, cache=True, nogil=True, fastmath=True, error_model="numpy")(eval_bfs_sparse_internal_)
# eval_bfs_internal_serial = njit(parallel=False, cache=True, nogil=True, fastmath=True, error_model="numpy")(eval_bfs_internal_)
# eval_bfs_and_grad_sparse_internal = njit(parallel=True, cache=True, nogil=True, fastmath=True, error_model="numpy")(eval_bfs_and_grad_sparse_internal_)
# eval_bfs_and_grad_internal = njit(parallel=True, cache=True, nogil=True, fastmath=True, error_model="numpy")(eval_bfs_and_grad_internal_)
# eval_bfs_and_grad_sparse_internal_serial = njit(parallel=False, cache=True, nogil=True, fastmath=True, error_model="numpy")(eval_bfs_and_grad_sparse_internal_)
# eval_bfs_and_grad_internal_serial = njit(parallel=False, cache=True, nogil=True, fastmath=True, error_model="numpy")(eval_bfs_and_grad_internal_)


@njit(parallel=True, cache=True, fastmath=True, error_model="numpy", nogil=True, inline='always')
def nonzero_ao_indices_batch(coords, bfs_coords, bfs_radius_cutoff):
    nbfs = bfs_coords.shape[0]
    ncoords = coords.shape[0]
    count = 0
    indices = np.zeros((nbfs), dtype='uint16')
    # Loop over the basis functions 
    for ibf in range(nbfs):
        coord_bf = bfs_coords[ibf]
        cutoff = bfs_radius_cutoff[ibf]
        # Loop over the grid points and check if the value of the basis function is greater than the threshold
        for igrd in range(ncoords):
            coord_grid = coords[igrd]
            x = coord_grid[0]-coord_bf[0]
            y = coord_grid[1]-coord_bf[1]
            z = coord_grid[2]-coord_bf[2]
            if (np.sqrt(x**2+y**2+z**2)<cutoff):
                indices[count] = ibf
                count = count + 1
                break

    # Return the indices array and the number of non-zero bfs
    return indices, count

def nonzero_ao_indices(basis, coords, blocksize, nblocks, ngrids):
    #TODO: Parallelize this using joblib
    # For a given set of grids and the batch/block size
    # it calculates the list of indices for each block
    # that corresponds to the basis functions which 
    # have non-zero contributions to those batches/blocks.
    # It also returns the number of such significant bfs
    # for each block/batch.
    bfs_coords = np.array([basis.bfs_coords])
    bfs_radius_cutoff = np.zeros([basis.bfs_nao])
    for i in range(basis.bfs_nao):
        bfs_radius_cutoff[i] = basis.bfs_radius_cutoff[i]
    # Calculate the value of basis functions for all grid points in batches
    # and find the indices of basis functions that have a significant contribution to those batches for each batch
    list_nonzero_indices = []
    count_nonzero_indices = []
    # Loop over batches
    for iblock in range(nblocks+1):
        offset = iblock*blocksize
        coords_block = coords[offset : min(offset+blocksize,ngrids)]   
        nonzero_indices, count = nonzero_ao_indices_batch(coords_block, bfs_coords[0], bfs_radius_cutoff)
        list_nonzero_indices.append(nonzero_indices)
        count_nonzero_indices.append(count)
    return list_nonzero_indices, count_nonzero_indices

@cuda.jit(fastmath=True, cache=True)
def nonzero_ao_indices_batch_cuda(coords, bfs_coords, bfs_radius_cutoff, nblocks, blocksize, ngrids, nbfs, nonzero_indices_mask):
    iblock, ibf = cuda.grid(2)
    if iblock<(nblocks+1) and ibf<nbfs:
        coord_bf = bfs_coords[ibf]
        offset = iblock*blocksize
        # Loop over the grid points and check if the value of the basis function is greater than the threshold
        for igrd in range(offset,  min(offset+blocksize, ngrids)):
            coord_grid = coords[igrd]
            x = coord_grid[0]-coord_bf[0]
            y = coord_grid[1]-coord_bf[1]
            z = coord_grid[2]-coord_bf[2]
            if (math.sqrt(x**2+y**2+z**2)<bfs_radius_cutoff[ibf]):
                nonzero_indices_mask[iblock, ibf] = 1
                break


def nonzero_ao_indices_cupy(basis, coords, blocksize, nblocks, ngrids, cp_stream=None):
    #TODO: Parallelize for multiple GPU case using joblib
    # For a given set of grids and the batch/block size
    # it calculates the list of indices for each block
    # that corresponds to the basis functions which 
    # have non-zero contributions to those batches/blocks.
    # It also returns the number of such significant bfs
    # for each block/batch.
    # NOTE: ONLY CUPY STREAMS ARE EXPECTED
    if cp_stream is None:
        device = 0
        cp.cuda.Device(device).use()
        cp_stream = cp.cuda.Stream(non_blocking = True)
        nb_stream = cuda.external_stream(cp_stream.ptr)
        cp_stream.use()
    else:
        nb_stream = cuda.external_stream(cp_stream.ptr)
        cp_stream.use()
    bfs_coords = cp.array([basis.bfs_coords])[0]
    bfs_radius_cutoff = cp.asarray(basis.bfs_radius_cutoff)
    # Calculate the value of basis functions for all grid points in batches
    # and find the indices of basis functions that have a significant contribution to those batches for each batch
    nonzero_indices_mask = cp.zeros((nblocks+1, basis.bfs_nao), dtype='uint8')

    coords = cp.asarray(coords, dtype=cp.float64)
    thread_x = min(nblocks, 4) #iblock
    thread_y = min(basis.bfs_nao, 128) #ibf
    blocks_per_grid = ((nblocks+1 + (thread_x - 1))//thread_x, (basis.bfs_nao + (thread_y - 1))//thread_y)
    # print(blocks_per_grid) 
    nonzero_ao_indices_batch_cuda[blocks_per_grid, (thread_x, thread_y), nb_stream](coords, bfs_coords, bfs_radius_cutoff, nblocks, blocksize, ngrids, basis.bfs_nao, nonzero_indices_mask)
    # cuda.synchronize()
    list_nonzero_indices = []
    count_nonzero_indices = []
    for iblock in range(nblocks+1):
        list_ = [a.astype(cp.uint16) for a in nonzero_indices_mask[iblock].nonzero()][0]
        list_nonzero_indices.append(list_)
        count_nonzero_indices.append(list_.shape[0])

    # cuda.synchronize()
    cp_stream.synchronize()
    cp.cuda.Stream.null.synchronize()
        # cp._default_memory_pool.free_all_blocks()
    
    return list_nonzero_indices, count_nonzero_indices