import numpy as np
import numexpr
import pylibxc
from timeit import default_timer as timer
# from time import process_time
from pyfock import Integrals
from opt_einsum import contract
from joblib import Parallel, delayed
from threadpoolctl import ThreadpoolController, threadpool_info, threadpool_limits
try:
    import cupy as cp
except Exception as e:
    # Handle the case when Cupy is not installed
    cp = np
import numba

def eval_xc_2_cupy(basis, dmat, weights, coords, funcid=[1,7], spin=0, ncores=2, blocksize=5000, list_nonzero_indices=None, count_nonzero_indices=None, list_ao_values=None, list_ao_grad_values=None, debug=False):
    print('Calculating XC term using GPU and algo 2', flush=True)
    print('Algo 2 uses a hybrid CPU+GPU approach for XC evaluation.', flush=True)
    print('CPU threads specified by ncores are used to evaluate functional values (via LibXC) and AO/AO grad values.', flush=True)
    # This performs parallelization at the blocks/batches level.
    # Therefore, joblib is perfect for such embarrasingly parallel task
    # In order to evaluate a density functional we will use the 
    # libxc library with Python bindings.
    # However, some sort of simple functionals like LDA, GGA, etc would
    # need to be implemented in CrysX also, so that the it doesn't depend
    # on external libraries so heavily that it becomes unusable without those.

    #Useful links:
    # LibXC manual: https://www.tddft.org/programs/libxc/manual/
    # LibXC gitlab: https://gitlab.com/libxc/libxc/-/tree/master
    # LibXC python interface code: https://gitlab.com/libxc/libxc/-/blob/master/pylibxc/functional.py
    # LibXC python version installation and example: https://www.tddft.org/programs/libxc/installation/
    # Formulae for XC energy and potential calculation: https://pubs.acs.org/doi/full/10.1021/ct200412r
    # LibXC code list: https://github.com/pyscf/pyscf/blob/master/pyscf/dft/libxc.py
    # PySCF nr_rks code: https://github.com/pyscf/pyscf/blob/master/pyscf/dft/numint.py
    # https://www.osti.gov/pages/servlets/purl/1650078

    #OUTPUT
    #Functional energy
    efunc = 0.0
    #Functional potential V_{\mu \nu} = \mu|\partial{f}/\partial{\rho}|\nu
    v = cp.zeros((basis.bfs_nao, basis.bfs_nao))
    nelec = 0

    # TODO it only works for LDA functionals for now.
    # Need to make it work for GGA, Hybrid, range-separated Hybrid and MetaGGA functionals as well.
    

    ngrids = coords.shape[0]
    nblocks = ngrids//blocksize

    # print('Number of blocks: ', nblocks)

    # Some stuff to note timings
    timings = None
    if debug:
        durationLibxc = 0.0
        durationE = 0.0
        durationF = 0.0
        durationZ = 0.0
        durationV = 0.0
        durationRho = 0.0
        durationAO = 0.0
        timings = {'durationLibxc':durationLibxc, 'durationE':durationE, 'durationF':durationF, 'durationZ':durationZ, 'durationV':durationV, 'durationRho':durationRho, 'durationAO':durationAO}

    

    ### Calculate stuff necessary for bf/ao evaluation on grid points
    ### Doesn't make any difference for 510 bfs but might be significant for >1000 bfs
    # This will help to make the call to evalbfnumba1 faster by skipping the mediator evalbfnumbawrap function 
    # that prepares the following stuff at every iteration
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
    # Now bf/ao values can be evaluated by calling the following
    # bf_values = Integrals.bf_val_helpers.eval_bfs(bfs_coords[0], bfs_contr_prim_norms[0], bfs_nprim[0], bfs_lmn[0], bfs_coeffs, bfs_prim_norms, bfs_expnts, bfs_radius_cutoff, coord)
    bfs_data_as_np_arrays = [bfs_coords[0], bfs_contr_prim_norms[0], bfs_nprim[0], bfs_lmn[0], bfs_coeffs, bfs_prim_norms, bfs_expnts, bfs_radius_cutoff]

    xc_family_dict = {1:'LDA',2:'GGA',4:'MGGA'} 
    # Create a LibXC object  
    funcx = pylibxc.LibXCFunctional(funcid[0], "unpolarized")
    funcc = pylibxc.LibXCFunctional(funcid[1], "unpolarized")
    x_family_code = funcx.get_family()
    c_family_code = funcc.get_family()

    weights_cp = cp.asarray(weights)
    dmat_cp = cp.asarray(dmat)
    # Create streams for asynchronous execution
    streams = [cp.cuda.Stream(non_blocking=True) for i in range(ncores)]

    #### Set number of cores for numba related evaluations within 'block_dens_func()' for example bf_value evaluations 
    #### Apparently, it was still using as many cores as possible and creating a serial version of thise functions as I have 
    #### currently done doesn't work  
    # numba.set_num_threads(1)
    if list_nonzero_indices is not None:
        if list_ao_values is not None:
            if xc_family_dict[x_family_code]=='LDA' and xc_family_dict[c_family_code]=='LDA':
                output = Parallel(n_jobs=ncores, backend='threading', require='sharedmem', batch_size=nblocks//ncores)(delayed(block_dens_func)(weights_cp[iblock*blocksize : min(iblock*blocksize+blocksize,ngrids)], coords[iblock*blocksize : min(iblock*blocksize+blocksize,ngrids)], dmat_cp[cp.ix_(list_nonzero_indices[iblock][0:count_nonzero_indices[iblock]], list_nonzero_indices[iblock][0:count_nonzero_indices[iblock]])], funcid, bfs_data_as_np_arrays, list_nonzero_indices[iblock][0:count_nonzero_indices[iblock]], list_ao_values[iblock], funcx=funcx, funcc=funcc, x_family_code=x_family_code, c_family_code=c_family_code, xc_family_dict=xc_family_dict, debug=debug, stream=streams[iblock%len(streams)]) for iblock in range(nblocks+1))
            else: #GGA
                output = Parallel(n_jobs=ncores, backend='threading', require='sharedmem')(delayed(block_dens_func)(weights_cp[iblock*blocksize : min(iblock*blocksize+blocksize,ngrids)], coords[iblock*blocksize : min(iblock*blocksize+blocksize,ngrids)], dmat_cp[np.ix_(list_nonzero_indices[iblock][0:count_nonzero_indices[iblock]], list_nonzero_indices[iblock][0:count_nonzero_indices[iblock]])], funcid, bfs_data_as_np_arrays, list_nonzero_indices[iblock][0:count_nonzero_indices[iblock]], list_ao_values[iblock], list_ao_grad_values[iblock], funcx=funcx, funcc=funcc, x_family_code=x_family_code, c_family_code=c_family_code, xc_family_dict=xc_family_dict, debug=debug) for iblock in range(nblocks+1))
        else:
            output = Parallel(n_jobs=ncores, backend='threading', require='sharedmem')(delayed(block_dens_func)(weights_cp[iblock*blocksize : min(iblock*blocksize+blocksize,ngrids)], coords[iblock*blocksize : min(iblock*blocksize+blocksize,ngrids)], dmat_cp[np.ix_(list_nonzero_indices[iblock][0:count_nonzero_indices[iblock]], list_nonzero_indices[iblock][0:count_nonzero_indices[iblock]])], funcid, bfs_data_as_np_arrays, list_nonzero_indices[iblock][0:count_nonzero_indices[iblock]], funcx=funcx, funcc=funcc, x_family_code=x_family_code, c_family_code=c_family_code, xc_family_dict=xc_family_dict, debug=debug) for iblock in range(nblocks+1))
    else:
        output = Parallel(n_jobs=ncores, backend='threading', require='sharedmem')(delayed(block_dens_func)(weights_cp[iblock*blocksize : min(iblock*blocksize+blocksize,ngrids)], coords[iblock*blocksize : min(iblock*blocksize+blocksize,ngrids)], dmat_cp, funcid, bfs_data_as_np_arrays, non_zero_indices=None, ao_values=None, funcx=funcx, funcc=funcc, x_family_code=x_family_code, c_family_code=c_family_code, xc_family_dict=xc_family_dict, debug=debug) for iblock in range(nblocks+1))
        
    # print(len(output))
    for iblock in range(0,len(output)):
        efunc += output[iblock][0]
        if list_nonzero_indices is not None:
            non_zero_indices = list_nonzero_indices[iblock][0:count_nonzero_indices[iblock]]
            v[cp.ix_(non_zero_indices, non_zero_indices)] += output[iblock][1]
        else:
            v += output[iblock][1]
        nelec += output[iblock][2]
        if debug:
            timings['durationLibxc'] += output[iblock][3]['durationLibxc']
            timings['durationE'] += output[iblock][3]['durationE']
            timings['durationF'] += output[iblock][3]['durationF']
            timings['durationZ'] += output[iblock][3]['durationZ']
            timings['durationV'] += output[iblock][3]['durationV']
            timings['durationRho'] += output[iblock][3]['durationRho']
            timings['durationAO'] += output[iblock][3]['durationAO']
        # v = numexpr.evaluate('(v + output[iblock][1])')


    

    numba.set_num_threads(ncores)    
    print('Number of electrons: ', nelec)
    if debug:
        print('Timings:', timings)

    ####### Free memory
    ## The following is very important to prevent memory leaks and also to make sure that the number of 
    # threads used by the program is same as that specified by the user 
    # gc.collect()  # Avoiding using it for now, as it is usually quite slow, although in this case it might not make much difference
    # Anyway, the following also works
    output = 0
    non_zero_indices = 0
    coords = 0
    cp.cuda.Stream.null.synchronize()
    return efunc, v


def block_dens_func(weights_block, coords_block, dmat, funcid, bfs_data_as_np_arrays, non_zero_indices=None, ao_values=None, ao_grad_values=None, funcx=None, funcc=None, x_family_code=None, c_family_code=None, xc_family_dict=None, debug=False, stream=None):
    ### Use threadpoolctl https://github.com/numpy/numpy/issues/11826
    # to set the number of threads to 1
    # https://github.com/joblib/threadpoolctl
    # https://stackoverflow.com/questions/29559338/set-max-number-of-threads-at-runtime-on-numpy-openblas
    
    
    durationLibxc = 0.0
    durationE = 0.0
    durationF = 0.0
    durationZ = 0.0
    durationV = 0.0
    durationRho = 0.0
    durationAO = 0.0
    numba.set_num_threads(1)

    if funcx is None:
        xc_family_dict = {1:'LDA',2:'GGA',4:'MGGA'}
        funcx = pylibxc.LibXCFunctional(funcid[0], "unpolarized")
        funcc = pylibxc.LibXCFunctional(funcid[1], "unpolarized")
        x_family_code = funcx.get_family()
        c_family_code = funcc.get_family()

    
    bfs_coords = bfs_data_as_np_arrays[0]
    bfs_contr_prim_norms = bfs_data_as_np_arrays[1]
    bfs_nprim = bfs_data_as_np_arrays[2]
    bfs_lmn = bfs_data_as_np_arrays[3]
    bfs_coeffs = bfs_data_as_np_arrays[4]
    bfs_prim_norms = bfs_data_as_np_arrays[5]
    bfs_expnts = bfs_data_as_np_arrays[6]
    bfs_radius_cutoff = bfs_data_as_np_arrays[7]

    # with stream:
    if debug:
        startAO = timer()
    # AO and Grad values
    # LDA
    if xc_family_dict[x_family_code]=='LDA' and xc_family_dict[c_family_code]=='LDA':
        if ao_values is not None: # If ao_values are calculated once and saved, then they can be provided to avoid recalculation
            ao_value_block = ao_values
        else:
            # ao_value_block = Integrals.evalBFsNumbawrap(basis, coords_block, parallel=False)
            if non_zero_indices is not None:
                # ao_value_block = Integrals.bf_val_helpers.eval_bfs_sparse_internal(bfs_coords, bfs_contr_prim_norms, bfs_nprim, bfs_lmn, bfs_coeffs, bfs_prim_norms, bfs_expnts, coords_block, non_zero_indices)
                ao_value_block = Integrals.bf_val_helpers.eval_bfs_sparse_vectorized_internal(bfs_coords, bfs_contr_prim_norms, bfs_nprim, bfs_lmn, bfs_coeffs, bfs_prim_norms, bfs_expnts, coords_block, non_zero_indices)
            else:
                ao_value_block = Integrals.bf_val_helpers.eval_bfs_internal(bfs_coords, bfs_contr_prim_norms, bfs_nprim, bfs_lmn, bfs_coeffs, bfs_prim_norms, bfs_expnts, bfs_radius_cutoff, coords_block)
    # GGA/MGGA (# If either x or c functional is of GGA/MGGA type we need ao_grad_values)
    # If either x or c functional is of GGA/MGGA type we need ao_grad_values
    if xc_family_dict[x_family_code]!='LDA' or xc_family_dict[c_family_code]!='LDA':
        if ao_values is not None: # If ao_values are calculated once and saved, then they can be provided to avoid recalculation
            ao_value_block, ao_values_grad_block = ao_values, ao_grad_values
        else:
            # ao_value_block, ao_values_grad_block = Integrals.bf_val_helpers.eval_bfs_and_grad(basis, coords_block, deriv=1, parallel=True, non_zero_indices=non_zero_indices)
            if non_zero_indices is not None:
                # Calculating ao values and gradients together, didn't really do much improvement in computational speed
                ao_value_block, ao_values_grad_block = Integrals.bf_val_helpers.eval_bfs_and_grad_sparse_internal(bfs_coords, bfs_contr_prim_norms, bfs_nprim, bfs_lmn, bfs_coeffs, bfs_prim_norms, bfs_expnts, coords_block, non_zero_indices)
            else:
                ao_value_block, ao_values_grad_block = Integrals.bf_val_helpers.eval_bfs_and_grad_internal(bfs_coords, bfs_contr_prim_norms, bfs_nprim, bfs_lmn, bfs_coeffs, bfs_prim_norms, bfs_expnts, bfs_radius_cutoff, coords_block) 
    if debug:
        durationAO = durationAO + timer() - startAO
    # print('Duration for AO values: ', durationAO)
    #  
    # Convert to cupy array
    ao_value_block = cp.asarray(ao_value_block)
    if xc_family_dict[x_family_code]!='LDA' or xc_family_dict[c_family_code]!='LDA':
        ao_values_grad_block = cp.asarray(ao_values_grad_block)

    if debug:
        startRho = timer()
    rho_block = contract('ij,mi,mj->m', dmat, ao_value_block, ao_value_block, backend='cupy') # Original (pretty fast)
    # If either x or c functional is of GGA/MGGA type we need rho_grad_values too
    if xc_family_dict[x_family_code]!='LDA' or xc_family_dict[c_family_code]!='LDA':
        # rho_grad_block_x = contract('ij,mi,mj->m',dmat,ao_values_grad_block[0],ao_value_block)+\
        #                         contract('ij,mi,mj->m',dmat,ao_value_block,ao_values_grad_block[0])
        # rho_grad_block_y = contract('ij,mi,mj->m',dmat,ao_values_grad_block[1],ao_value_block)+\
        #                         contract('ij,mi,mj->m',dmat,ao_value_block,ao_values_grad_block[1])
        # rho_grad_block_z = contract('ij,mi,mj->m',dmat,ao_values_grad_block[2],ao_value_block)+\
        #                         contract('ij,mi,mj->m',dmat,ao_value_block,ao_values_grad_block[2])
        rho_grad_block_x, rho_grad_block_y, rho_grad_block_z  = ( contract('ij,kmi,mj->km',dmat,ao_values_grad_block,ao_value_block)+\
                            contract('ij,mi,kmj->km',dmat,ao_value_block,ao_values_grad_block) )[:]
        sigma_block = rho_grad_block_x**2 + rho_grad_block_y**2 + rho_grad_block_z**2
        sigma_block = cp.asnumpy(sigma_block)
    if debug:
        durationRho = timer() - startRho
    # print('Duration for Rho at grid points: ',durationRho)


    
    #LibXC stuff
    # Exchange
    if debug:
        startLibxc = timer()
    
    # Input dictionary for libxc
    inp = {}
    # Input dictionary needs density values at grid points
    inp['rho'] = cp.asnumpy(rho_block)
    if xc_family_dict[x_family_code]!='LDA':
        # Input dictionary needs sigma (\nabla \rho \cdot \nabla \rho) values at grid points
        inp['sigma'] = sigma_block
    # Calculate the necessary quantities using LibXC
    retx = funcx.compute(inp)
    # durationLibxc = durationLibxc + timer() - startLibxc
    # print('Duration for LibXC computations at grid points: ',durationLibxc)

    # Correlation
    # startLibxc = timer()
    
    # Input dictionary for libxc
    # inp = {}
    # Input dictionary needs density values at grid points
    # inp['rho'] = rho_block
    if xc_family_dict[c_family_code]!='LDA':
        # Input dictionary needs sigma (\nabla \rho \cdot \nabla \rho) values at grid points
        inp['sigma'] = sigma_block
    # Calculate the necessary quantities using LibXC
    retc = funcc.compute(inp)
    if debug:
        durationLibxc = durationLibxc + timer() - startLibxc
    # print('Duration for LibXC computations at grid points: ',durationLibxc)

    if debug:
        startE = timer()
    #ENERGY-----------
    e = retx['zk'] + retc['zk'] # Functional values at grid points
    # Testing CrysX's own implmentation
    #e = densfuncs.lda_x(rho)

    # Calculate the total energy 
    # Multiply the density at grid points by weights
    den = rho_block*weights_block #elementwise multiply
    efunc = cp.dot(den, cp.asarray(e)) #Multiply with functional values at grid points and sum
    nelec = cp.sum(den)
    if debug:
        durationE = durationE + timer() - startE
    # print('Duration for calculation of total density functional energy: ',durationE)

    #POTENTIAL----------
    # The derivative of functional wrt density is vrho
    vrho = retx['vrho'] + retc['vrho']
    vsigma = 0
    # If either x or c functional is of GGA/MGGA type we need rho_grad_values
    if xc_family_dict[x_family_code]!='LDA':
        # The derivative of functional wrt grad \rho square.
        vsigma = retx['vsigma']
    if xc_family_dict[c_family_code]!='LDA':
        # The derivative of functional wrt grad \rho square.
        vsigma += retc['vsigma']
    retx = 0
    retc = 0
    func = 0
    
    if debug:
        startF = timer()
    v_rho_temp = cp.asarray(vrho[:,0])
    F = weights_block*v_rho_temp
    # If either x or c functional is of GGA/MGGA type we need rho_grad_values
    if xc_family_dict[x_family_code]!='LDA' or xc_family_dict[c_family_code]!='LDA':
        vsigma_temp = cp.asarray(vsigma[:,0])
        Ftemp = 2*weights_block*vsigma_temp
        # Ftemp = 2*weights_block*vsigma[:,0]
        Fx = Ftemp*rho_grad_block_x
        Fy = Ftemp*rho_grad_block_y
        Fz = Ftemp*rho_grad_block_z
    if debug:
        durationF = durationF + timer() - startF
    # print('Duration for calculation of F: ',durationF)
    
    if debug:
        startZ = timer()
    z = 0.5*F*ao_value_block.T
    # If either x or c functional is of GGA/MGGA type we need rho_grad_values
    if xc_family_dict[x_family_code]!='LDA' or xc_family_dict[c_family_code]!='LDA':
        z += Fx*ao_values_grad_block[0].T + Fy*ao_values_grad_block[1].T + Fz*ao_values_grad_block[2].T
    if debug:
        durationZ = durationZ + timer() - startZ
    # Free memory
    F = 0
    v_rho_temp = 0
    Fx = 0
    Fy = 0
    Fz = 0
    Ftemp = 0
    vsigma_temp = 0
    if debug:
        startV = timer()
    # with threadpool_limits(limits=1, user_api='blas'):
    # v_temp = z @ ao_value_block  # The fastest uptil now
    v_temp = cp.matmul(z, ao_value_block)  
    v = v_temp + v_temp.T
    if debug:
        durationV = durationV + timer() - startV
    z = 0
    ao_value_block = 0
    rho_block = 0
    temp = 0
    vrho = 0 
    weights_block=0
    coords_block=0
    func = 0
    dmat = 0
    v_temp_T = 0
    v_temp = 0
    del ao_value_block
    # del ao_values_grad_block
    # del rho_grad_block_x
    # del rho_grad_block_y
    del rho_block
    del z
    del temp
    del F
    del weights_block
    del coords_block
    del dmat
    del vrho
    
    
    profiling_timings = {'durationLibxc':durationLibxc, 'durationE':durationE, 'durationF':durationF, 'durationZ':durationZ, 'durationV':durationV, 'durationRho':durationRho, 'durationAO':durationAO}

    # print(durationRho)
    return efunc[0], v, nelec, profiling_timings
