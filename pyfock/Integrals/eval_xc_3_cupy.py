import numpy as np
import numexpr
import pylibxc
from timeit import default_timer as timer
# from time import process_time
from pyfock import Integrals
from pyfock import XC
from opt_einsum import contract, contract_expression
from joblib import Parallel, delayed
from threadpoolctl import ThreadpoolController, threadpool_info, threadpool_limits
try:
    import cupy as cp
    from cupy import fuse
except Exception as e:
    # Handle the case when Cupy is not installed
    cp = np
    # Define a dummy fuse decorator for CPU version
    def fuse(kernel_name):
        def decorator(func):
            return func 
        return decorator
import numba
from numba import cuda
def eval_xc_3_cupy(basis, dmat, weights, coords, funcid=[1,7], spin=0, blocksize=10240, debug=False, list_nonzero_indices=None, \
                   count_nonzero_indices=None, list_ao_values=None, list_ao_grad_values=None, use_libxc=True, nstreams=1, ngpus=1,\
                    freemem=True, threads_per_block=None, type=cp.float64, streams=None, nb_streams=None, bfs_data_as_np_arrays=None):
    print('Calculating XC term using GPU and algo 3', flush=True)
    if not use_libxc:
        print('Not using LibXC for XC evaluations', flush=True)
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
    # Start from the main GPU
    cp.cuda.Device(0).use()
    #Functional potential V_{\mu \nu} = \mu|\partial{f}/\partial{\rho}|\nu
    v = cp.zeros((basis.bfs_nao, basis.bfs_nao), dtype=type)
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
    if bfs_data_as_np_arrays is None:
        bfs_coords = cp.asarray([basis.bfs_coords], dtype=type)
        bfs_contr_prim_norms = cp.asarray([basis.bfs_contr_prim_norms], dtype=type)
        bfs_lmn = cp.asarray([basis.bfs_lmn])
        bfs_nprim = cp.asarray([basis.bfs_nprim])
        #The remaining properties like bfs_coeffs are a list of lists of unequal sizes.
        #Numba won't be able to work with these efficiently.
        #So, we convert them to a numpy 2d array by applying a trick,
        #that the second dimension is that of the largest list. So that
        #it can accomadate all the lists.
        maxnprim = max(basis.bfs_nprim)
        bfs_coeffs = cp.zeros([basis.bfs_nao, maxnprim], dtype=type)
        bfs_expnts = cp.zeros([basis.bfs_nao, maxnprim], dtype=type)
        bfs_prim_norms = cp.zeros([basis.bfs_nao, maxnprim], dtype=type)
        bfs_radius_cutoff = cp.zeros([basis.bfs_nao], dtype=type)
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

    my_expr = contract_expression('ij,mi,mj->m', (150, 150), (blocksize, 150), (blocksize, 150))
    my_expr_grad1 = None
    my_expr_grad2 = None
    if xc_family_dict[x_family_code]!='LDA' or xc_family_dict[c_family_code]!='LDA':
        my_expr_grad1 = contract_expression('ij,kmi,mj->km', (150, 150), (3, blocksize, 150), (blocksize, 150))
        my_expr_grad2 = contract_expression('ij,mi,kmj->km', (150, 150), (blocksize, 150), (3, blocksize, 150))
    contr_expr = [my_expr, my_expr_grad1, my_expr_grad2]

    if threads_per_block is None:
        # Determine the optimal number of blocks per grid
        max_threads_per_block = cuda.get_current_device().MAX_THREADS_PER_BLOCK
        thread_x = max_threads_per_block/16
        thread_y = max_threads_per_block/64
        threads_per_block = (thread_x, thread_y)
    else:
        thread_x = threads_per_block[0]
        thread_y = threads_per_block[1]
    
    for iblock in range(len(list_nonzero_indices)):
        list_nonzero_indices[iblock] = cp.asarray(list_nonzero_indices[iblock])

    
    weights_cp = cp.asarray(weights, dtype=type)
    coords_cp = cp.asarray(coords, dtype=type)

    dmat_cp = cp.asarray(dmat, dtype=type)
    # Create dmat lists
    dmats_list = []
    for iblock in range(nblocks + 1):
        dmats_list.append(cp.copy(dmat_cp[cp.ix_(list_nonzero_indices[iblock][0:count_nonzero_indices[iblock]], list_nonzero_indices[iblock][0:count_nonzero_indices[iblock]])]))
        
    # Create streams for asynchronous execution
    # on different GPUs 
    if streams is None and nb_streams is None:
        streams = []
        nb_streams = []
        for i in range(ngpus):
            cp.cuda.Device(i).use()
            cp_stream = cp.cuda.Stream(non_blocking = True)
            nb_stream = cuda.external_stream(cp_stream.ptr)
            streams.append(cp_stream)
            nb_streams.append(nb_stream)

    if list_nonzero_indices is not None:
        if list_ao_values is not None:
            if xc_family_dict[x_family_code]=='LDA' and xc_family_dict[c_family_code]=='LDA':
                output = Parallel(n_jobs=ngpus, backend='threading', require='sharedmem')(delayed(block_dens_func)(weights_cp[iblock*blocksize : min(iblock*blocksize+blocksize,ngrids)], coords_cp[iblock*blocksize : min(iblock*blocksize+blocksize,ngrids)], dmats_list[iblock], funcid, bfs_data_as_np_arrays, list_nonzero_indices[iblock][0:count_nonzero_indices[iblock]], list_ao_values[iblock], funcx=funcx, funcc=funcc, x_family_code=x_family_code, c_family_code=c_family_code, xc_family_dict=xc_family_dict, debug=debug, stream=streams[iblock%ngpus], nb_stream=nb_streams[iblock%ngpus], thread_x=thread_x, 
                    thread_y=thread_y, threads_per_block=threads_per_block, type=type, use_libxc=use_libxc, expressions=contr_expr, device=iblock%ngpus) for iblock in range(nblocks+1))
            else: #GGA
                output = Parallel(n_jobs=ngpus, backend='threading', require='sharedmem')(delayed(block_dens_func)(weights_cp[iblock*blocksize : min(iblock*blocksize+blocksize,ngrids)], coords_cp[iblock*blocksize : min(iblock*blocksize+blocksize,ngrids)], dmats_list[iblock], funcid, bfs_data_as_np_arrays, list_nonzero_indices[iblock][0:count_nonzero_indices[iblock]], list_ao_values[iblock], list_ao_grad_values[iblock], funcx=funcx, funcc=funcc, x_family_code=x_family_code, c_family_code=c_family_code, xc_family_dict=xc_family_dict, debug=debug, stream=streams[iblock%ngpus], nb_stream=nb_streams[iblock%ngpus], thread_x=thread_x, 
                    thread_y=thread_y, threads_per_block=threads_per_block, type=type, use_libxc=use_libxc, expressions=contr_expr, device=iblock%ngpus) for iblock in range(nblocks+1))
        else:
            output = Parallel(n_jobs=ngpus, backend='threading', require='sharedmem')(delayed(block_dens_func)(weights_cp[iblock*blocksize : min(iblock*blocksize+blocksize,ngrids)], coords_cp[iblock*blocksize : min(iblock*blocksize+blocksize,ngrids)], dmats_list[iblock], funcid, bfs_data_as_np_arrays, list_nonzero_indices[iblock][0:count_nonzero_indices[iblock]], funcx=funcx, funcc=funcc, x_family_code=x_family_code, c_family_code=c_family_code, xc_family_dict=xc_family_dict, debug=debug, stream=streams[iblock%ngpus], nb_stream=nb_streams[iblock%ngpus], thread_x=thread_x, 
                    thread_y=thread_y, threads_per_block=threads_per_block, type=type, use_libxc=use_libxc, expressions=contr_expr, device=iblock%ngpus) for iblock in range(nblocks+1))
    else:
        output = Parallel(n_jobs=ngpus, backend='threading', require='sharedmem')(delayed(block_dens_func)(weights_cp[iblock*blocksize : min(iblock*blocksize+blocksize,ngrids)], coords_cp[iblock*blocksize : min(iblock*blocksize+blocksize,ngrids)], dmat_cp, funcid, bfs_data_as_np_arrays, non_zero_indices=None, ao_values=None, funcx=funcx, funcc=funcc, x_family_code=x_family_code, c_family_code=c_family_code, xc_family_dict=xc_family_dict, debug=debug, stream=streams[iblock%ngpus], nb_stream=nb_streams[iblock%ngpus], thread_x=thread_x, 
                    thread_y=thread_y, threads_per_block=threads_per_block, type=type, use_libxc=use_libxc, expressions=contr_expr, device=iblock%ngpus) for iblock in range(nblocks+1))
        
    for istream in range(ngpus):
        streams[istream].synchronize()
    # Switch back to main GPU
    cp.cuda.Device(0).use()
    for iblock in range(0,len(output)):
        efunc += cp.asarray(output[iblock][0])
        if list_nonzero_indices is not None:
            non_zero_indices = cp.asarray(list_nonzero_indices[iblock][0:count_nonzero_indices[iblock]])
            # v[cp.ix_(non_zero_indices, non_zero_indices)] += cp.asarray(output[iblock][1])
            v[cp.ix_(non_zero_indices, non_zero_indices)] = cp.asarray(output[iblock][1]) + cp.asarray(v)[cp.ix_(non_zero_indices, non_zero_indices)]
        else:
            # v += cp.asarray(output[iblock][1])
            v = cp.asarray(output[iblock][1]) + cp.asarray(v)
        nelec += cp.asarray(output[iblock][2]) 
        if debug:
            timings['durationLibxc'] += output[iblock][3]['durationLibxc']
            timings['durationE'] += output[iblock][3]['durationE']
            timings['durationF'] += output[iblock][3]['durationF']
            timings['durationZ'] += output[iblock][3]['durationZ']
            timings['durationV'] += output[iblock][3]['durationV']
            timings['durationRho'] += output[iblock][3]['durationRho']
            timings['durationAO'] += output[iblock][3]['durationAO']
        # v = numexpr.evaluate('(v + output[iblock][1])')


    
   
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

    if use_libxc:
        efunc = efunc[0]

    if freemem:
        for istream in range(ngpus):
            cp.cuda.Device(istream).use()
            cp._default_memory_pool.free_all_blocks()
            # cuda.current_context().memory_manager.deallocations.clear()
        # Switch back to main GPU
        cp.cuda.Device(0).use()

    return efunc, v


def block_dens_func(weights_block, coords_block, dmat, funcid, bfs_data_as_np_arrays, non_zero_indices=None, ao_values=None, ao_grad_values=None, funcx=None, 
                    funcc=None, x_family_code=None, c_family_code=None, xc_family_dict=None, debug=False, stream=None, nb_stream=None, thread_x=None, 
                    thread_y=None, threads_per_block=None, type=None, use_libxc=True, expressions=None, device=None):
    ### Use threadpoolctl https://github.com/numpy/numpy/issues/11826
    # to set the number of threads to 1
    # https://github.com/joblib/threadpoolctl
    # https://stackoverflow.com/questions/29559338/set-max-number-of-threads-at-runtime-on-numpy-openblas

    # exit()
    cp.cuda.Device(device).use()
    stream.use()

    weights_block = cp.asarray(weights_block)
    coords_block = cp.asarray(coords_block)

    dmat = cp.asarray(dmat)
    
    
    durationLibxc = 0.0
    durationE = 0.0
    durationF = 0.0
    durationZ = 0.0
    durationV = 0.0
    durationRho = 0.0
    durationAO = 0.0

    if funcx is None:
        xc_family_dict = {1:'LDA',2:'GGA',4:'MGGA'}
        funcx = pylibxc.LibXCFunctional(funcid[0], "unpolarized")
        funcc = pylibxc.LibXCFunctional(funcid[1], "unpolarized")
        x_family_code = funcx.get_family()
        c_family_code = funcc.get_family()

    
    bfs_coords = cp.asarray(bfs_data_as_np_arrays[0])
    bfs_contr_prim_norms = cp.asarray(bfs_data_as_np_arrays[1])
    bfs_nprim = cp.asarray(bfs_data_as_np_arrays[2])
    bfs_lmn = cp.asarray(bfs_data_as_np_arrays[3])
    bfs_coeffs = cp.asarray(bfs_data_as_np_arrays[4])
    bfs_prim_norms = cp.asarray(bfs_data_as_np_arrays[5])
    bfs_expnts = cp.asarray(bfs_data_as_np_arrays[6])
    bfs_radius_cutoff = cp.asarray(bfs_data_as_np_arrays[7])

    # bfs_coords = bfs_data_as_np_arrays[0]
    # bfs_contr_prim_norms = bfs_data_as_np_arrays[1]
    # bfs_nprim = bfs_data_as_np_arrays[2]
    # bfs_lmn = bfs_data_as_np_arrays[3]
    # bfs_coeffs = bfs_data_as_np_arrays[4]
    # bfs_prim_norms = bfs_data_as_np_arrays[5]
    # bfs_expnts = bfs_data_as_np_arrays[6]
    # bfs_radius_cutoff = bfs_data_as_np_arrays[7]

    non_zero_indices = cp.asarray(non_zero_indices)

    
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
                ao_value_block = cp.zeros((coords_block.shape[0], non_zero_indices.shape[0]), dtype=type) 
                blocks_per_grid = ((non_zero_indices.shape[0] + (thread_x - 1))//thread_x, (coords_block.shape[0] + (thread_y - 1))//thread_y) 
                Integrals.bf_val_helpers.eval_bfs_sparse_internal_cuda[blocks_per_grid, threads_per_block, nb_stream](bfs_coords, bfs_contr_prim_norms, bfs_nprim, bfs_lmn, bfs_coeffs, bfs_prim_norms, bfs_expnts, coords_block, non_zero_indices, ao_value_block)
                           
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
                ao_value_block = cp.zeros((coords_block.shape[0], non_zero_indices.shape[0]), dtype=type) 
                ao_values_grad_block = cp.zeros((3, coords_block.shape[0], non_zero_indices.shape[0]), dtype=type) 
                blocks_per_grid = ((non_zero_indices.shape[0] + (thread_x - 1))//thread_x, (coords_block.shape[0] + (thread_y - 1))//thread_y) 
                Integrals.bf_val_helpers.eval_bfs_and_grad_sparse_internal_cuda[blocks_per_grid, threads_per_block, nb_stream](bfs_coords, bfs_contr_prim_norms, bfs_nprim, 
                                                                                                                bfs_lmn, bfs_coeffs, bfs_prim_norms, bfs_expnts, 
                                                                                                                coords_block, non_zero_indices, ao_value_block, 
                                                                                                                ao_values_grad_block)
            else:
                ao_value_block, ao_values_grad_block = Integrals.bf_val_helpers.eval_bfs_and_grad_internal(bfs_coords, bfs_contr_prim_norms, bfs_nprim, bfs_lmn, bfs_coeffs, bfs_prim_norms, bfs_expnts, bfs_radius_cutoff, coords_block) 
    if debug:
        stream.synchronize()
        durationAO = durationAO + timer() - startAO
    # print('Duration for AO values: ', durationAO)
    #  
    # Convert to cupy array
    ao_value_block = cp.asarray(ao_value_block, dtype=type)
    if xc_family_dict[x_family_code]!='LDA' or xc_family_dict[c_family_code]!='LDA':
        ao_values_grad_block = cp.asarray(ao_values_grad_block, dtype=type)

    if debug:
        startRho = timer()
    # rho_block = contract('ij,mi,mj->m', dmat, ao_value_block, ao_value_block, backend='cupy') # Original (pretty fast)
    # rho_block = expressions[0](dmat, ao_value_block, ao_value_block, backend='cupy') # Original (pretty fast)

    # New approach based on this: https://pubs.acs.org/doi/10.1021/acs.jctc.0c01252 
    # Fjm = contract('ij,mi->jm', dmat, ao_value_block) # This intermediate helps in accelerating grad rho
    # rho_block = contract('jm,mj->m', Fjm, ao_value_block, backend='cupy')

    Fmj = ao_value_block @ dmat # This intermediate helps in accelerating grad rho
    # rho_block = contract('mj,mj->m', Fmj, ao_value_block, backend='cupy')
    rho_block = cp.sum(Fmj * ao_value_block, axis=1)


    # If either x or c functional is of GGA/MGGA type we need rho_grad_values too
    if xc_family_dict[x_family_code]!='LDA' or xc_family_dict[c_family_code]!='LDA':
        # rho_grad_block_x, rho_grad_block_y, rho_grad_block_z  = ( contract('ij,kmi,mj->km',dmat,ao_values_grad_block,ao_value_block, backend='cupy')+\
        #                     contract('ij,mi,kmj->km',dmat,ao_value_block,ao_values_grad_block, backend='cupy') )[:]
        # rho_grad_block_x, rho_grad_block_y, rho_grad_block_z  = expressions[1](dmat,ao_values_grad_block,ao_value_block, backend='cupy') \
        #                                                                 + expressions[2](dmat, ao_value_block, ao_values_grad_block, backend='cupy') 

        # New approach based on this: https://pubs.acs.org/doi/10.1021/acs.jctc.0c01252 
        # rho_grad_block_x, rho_grad_block_y, rho_grad_block_z  = 2*contract('jm,kmj->km', Fjm, ao_values_grad_block, backend='cupy')[:] 
        rho_grad_block_x, rho_grad_block_y, rho_grad_block_z  = 2*contract('mj,kmj->km', Fmj, ao_values_grad_block, backend='cupy')[:] 
        sigma_block = calc_sigma(rho_grad_block_x, rho_grad_block_y, rho_grad_block_z)
        if use_libxc:
            sigma_block = cp.asnumpy(sigma_block)
    if debug:
        stream.synchronize()
        durationRho = timer() - startRho
        # print('Duration for Rho at grid points: ',durationRho)


    
    #LibXC stuff
    if use_libxc:
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
        #inp = {}
        # Input dictionary needs density values at grid points
        #inp['rho'] = rho_block
        if xc_family_dict[c_family_code]!='LDA':
            # Input dictionary needs sigma (\nabla \rho \cdot \nabla \rho) values at grid points
            inp['sigma'] = sigma_block
        # Calculate the necessary quantities using LibXC
        retc = funcc.compute(inp)
        if debug:
            cp.cuda.Stream.null.synchronize()
            durationLibxc = durationLibxc + timer() - startLibxc
        # print('Duration for LibXC computations at grid points: ',durationLibxc)
    else:
        # Exchange
        if debug:
            startLibxc = timer()
        # Calculate the necessary quantities using own implementation
        # retx = lda_x(rho_block)
        # retx = gga_x_b88(rho_block, sigma_block)
        if xc_family_dict[x_family_code]=='LDA':
            retx = XC.func_compute(funcid[0], rho_block, use_gpu=True)
        else:
            retx = XC.func_compute(funcid[0], rho_block, sigma_block, use_gpu=True)
            # print(retx[0])
            # print(retx[1])
            # print(retx[2])

        # Correlation
        # Calculate the necessary quantities using own implementation
        # retc = lda_c_vwn(rho_block)
        # retc = gga_c_lyp(rho_block, sigma_block)
        if xc_family_dict[c_family_code]=='LDA':
            retc = XC.func_compute(funcid[1], rho_block, use_gpu=True)
        else:
            retc = XC.func_compute(funcid[1], rho_block, sigma_block, use_gpu=True)
        if debug:
            stream.synchronize()
            cp.cuda.Stream.null.synchronize()
            durationLibxc = durationLibxc + timer() - startLibxc

    if debug:
        startE = timer()
    #ENERGY-----------
    if use_libxc:
        e = retx['zk'] + retc['zk'] # Functional values at grid points
    else:
        e = retx[0] + retc[0]
    # Testing CrysX's own implmentation
    #e = densfuncs.lda_x(rho)

    # Calculate the total energy 
    # Multiply the density at grid points by weights
    den = rho_block*weights_block #elementwise multiply
    if use_libxc:
        efunc = cp.dot(den, cp.asarray(e)) #Multiply with functional values at grid points and sum
    else:
        efunc = cp.dot(den, e) #Multiply with functional values at grid points and sum
    nelec = cp.sum(den)
    if debug:
        stream.synchronize()
        durationE = durationE + timer() - startE
    # print('Duration for calculation of total density functional energy: ',durationE)

    #POTENTIAL----------
    # The derivative of functional wrt density is vrho
    if use_libxc:
        vrho = retx['vrho'] + retc['vrho']
    else:
        vrho = retx[1] + retc[1]
    vsigma = 0
    # If either x or c functional is of GGA/MGGA type we need rho_grad_values
    if xc_family_dict[x_family_code]!='LDA':
        # The derivative of functional wrt grad \rho square.
        if use_libxc:
            vsigma += retx['vsigma']
        else:
            vsigma += retx[2]
        
    if xc_family_dict[c_family_code]!='LDA':
        # The derivative of functional wrt grad \rho square.
        if use_libxc:
            vsigma += retc['vsigma']
        else:
            vsigma += retc[2]
    
    # F = np.multiply(weights_block,vrho[:,0]) #This is fast enough.
    if use_libxc:
        v_rho_temp = cp.asarray(vrho[:,0])
    else:
        v_rho_temp = vrho

    if debug:
        startF = timer()
    # F = weights_block*v_rho_temp
    # F = calc_F(weights_block, v_rho_temp)
    # If either x or c functional is of GGA/MGGA type we need rho_grad_values
    if xc_family_dict[x_family_code]!='LDA' or xc_family_dict[c_family_code]!='LDA':
        if use_libxc:
            Ftemp = 2*weights_block*cp.asarray(vsigma[:,0])
        else:
            Ftemp = 2*weights_block*vsigma
            # Ftemp = 2*cp.multiply(weights_block, vsigma)
            # Ftemp = 2*elementwise_multiply(weights_block, vsigma)
        # Fx = Ftemp*rho_grad_block_x
        # Fy = Ftemp*rho_grad_block_y
        # Fz = Ftemp*rho_grad_block_z
    if debug:
        stream.synchronize()
        durationF = durationF + timer() - startF
    # print('Duration for calculation of F: ',durationF)
    
    if debug:
        startZ = timer()
    z = calc_z(weights_block, v_rho_temp, ao_value_block.T)
    # z = 0.5*np.einsum('m,mi->mi',F,ao_value_block)
    # If either x or c functional is of GGA/MGGA type we need rho_grad_values
    if xc_family_dict[x_family_code]!='LDA' or xc_family_dict[c_family_code]!='LDA':
        # z += calc_z_gga(Fx, Fy, Fz, ao_values_grad_block[0].T, ao_values_grad_block[1].T, ao_values_grad_block[2].T)
        z += Ftemp*(rho_grad_block_x*ao_values_grad_block[0].T + rho_grad_block_y*ao_values_grad_block[1].T + rho_grad_block_z*ao_values_grad_block[2].T)
    if debug:
        stream.synchronize()
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
    
    
    profiling_timings = {'durationLibxc':durationLibxc, 'durationE':durationE, 'durationF':durationF, 'durationZ':durationZ, 'durationV':durationV, 'durationRho':durationRho, 'durationAO':durationAO}

    # cp.cuda.Device(0).use()
    # v = cp.array(v)
    # stream.synchronize()
    return efunc, v, nelec, profiling_timings

@fuse(kernel_name='calc_F')
def calc_F(weights_block, v_rho_temp):
    return weights_block*v_rho_temp

# @fuse(kernel_name='calc_z')
# def calc_z(F, ao_value_block):
#     # ao_value_block should be supplied after transposing it
#     return 0.5*F*ao_value_block

@fuse(kernel_name='calc_z')
def calc_z(weights_block, v_rho_temp, ao_value_block):
    # ao_value_block should be supplied after transposing it
    F = weights_block*v_rho_temp
    return 0.5*F*ao_value_block

@fuse(kernel_name='calc_sigma')
def calc_sigma(rho_grad_block_x, rho_grad_block_y, rho_grad_block_z):
    return rho_grad_block_x**2 + rho_grad_block_y**2 + rho_grad_block_z**2

@fuse(kernel_name='calc_z_gga')
def calc_z_gga(Fx, Fy, Fz, ao_values_grad_block_x, ao_values_grad_block_y, ao_values_grad_block_z):
    return Fx*ao_values_grad_block_x + Fy*ao_values_grad_block_y + Fz*ao_values_grad_block_z

@fuse(kernel_name='elementwise_multiply')
def elementwise_multiply(a, b):
    return a*b