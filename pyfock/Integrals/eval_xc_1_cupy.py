import numpy as np
import pylibxc
from timeit import default_timer as timer
from pyfock import Integrals
from opt_einsum import contract, contract_expression
from pyfock import XC
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
from numba import cuda
import math

def eval_xc_1_cupy(basis, dmat, weights, coords, funcid=[1,7], spin=0, blocksize=50000, debug=False, list_nonzero_indices=None, \
                   count_nonzero_indices=None, list_ao_values=None, list_ao_grad_values=None, use_libxc=True, nstreams=1, ngpus=1,\
                    freemem=True, threads_per_block=None, type=cp.float64):
    print('Calculating XC term using GPU and algo 1', flush=True)
    # Evaluate the XC term using GPU
    # Here instead of parallelizing over batches, the operations within
    # a batch are parallelized. Therefore, this requires a larger batchsize~20480.
    # While the CPU version of this algorithm is quite slow, 
    # the GPU version performs great!

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

    startTotal = timer()

    # For debugging and benchmarking purposes
    durationLibxc = 0.0
    durationE = 0.0
    durationF = 0.0
    durationZ = 0.0
    durationV = 0.0
    durationRho = 0.0
    durationAO = 0.0
    durationPrelims = 0.0
    durationTotal = 0.0
    n_gpu= 1
    # nstreams = 1
    # Create streams for asynchronous execution
    # streams = [cp.cuda.Stream(non_blocking=False) for i in range(nstreams)]
    # nb_streams = [cuda.external_stream(streams[i].ptr) for i in range(nstreams)]
    # nb_streams = [cuda.stream() for i in range(nstreams)]
    # cp_streams = [cp.cuda.ExternalStream(nb_streams[i].ptr) for i in range(nstreams)]
    # TODO: Try using this for multiGPU support: https://github.com/cupy/cupy/issues/5692
    # streams = []
    # for i in range(n_gpu):
    #     cp.cuda.Device(i).use()
    #     streams.append(cp.cuda.Stream(non_blocking = True))

    if not use_libxc:
        print('Not using LibXC for XC evaluations', flush=True)

    if debug:
        startPrelims = timer()
    #OUTPUT
    #Functional energy
    efunc = 0.0
    #Functional potential V_{\mu \nu} = \mu|\partial{f}/\partial{\rho}|\nu
    v = cp.zeros((basis.bfs_nao, basis.bfs_nao), dtype=type)
    #print(v.dtype)
    #TODO mGGA, Hybrid
    
    #Calculate number of blocks/batches
    ngrids = coords.shape[0]
    nblocks = ngrids//blocksize
    nelec = 0.0

    # Convert the arrays needed for CUDA computations to cupy arrays

    # If a list of significant basis functions for each block of grid points is provided
    if list_nonzero_indices is not None:
        dmat_orig = dmat
        dmat_orig_cp = cp.asarray(dmat_orig, dtype=type)
    else:
        dmat = cp.asarray(dmat, dtype=type)

    ### Calculate stuff necessary for bf/ao evaluation on grid points
    ### Doesn't make any difference for 510 bfs but might be significant for >1000 bfs
    # This will help to make the call to eval_bfs faster by skipping the mediator eval_bfs function 
    # that prepares the following stuff at every iteration
    #We convert the required properties to numpy arrays as this is what Numba likes.
    bfs_coords = cp.asarray([basis.bfs_coords], dtype=type)
    bfs_contr_prim_norms = cp.array([basis.bfs_contr_prim_norms], dtype=type)
    bfs_lmn = cp.array([basis.bfs_lmn])
    bfs_nprim = cp.array([basis.bfs_nprim])
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
    for iblock in range(len(list_nonzero_indices)):
        list_nonzero_indices[iblock] = cp.asarray(list_nonzero_indices[iblock])

    
    weights_cp = cp.asarray(weights, dtype=type)
    coords_cp = cp.asarray(coords, dtype=type)
    

    xc_family_dict = {1:'LDA',2:'GGA',4:'MGGA'} 

    # Create a LibXC object  
    funcx = pylibxc.LibXCFunctional(funcid[0], "unpolarized")
    funcc = pylibxc.LibXCFunctional(funcid[1], "unpolarized")
    x_family_code = funcx.get_family()
    c_family_code = funcc.get_family()


    my_expr = contract_expression('ij,mi,mj->m', (150, 150), (blocksize, 150), (blocksize, 150))
    if xc_family_dict[x_family_code]!='LDA' or xc_family_dict[c_family_code]!='LDA':
        my_expr_grad1 = contract_expression('ij,kmi,mj->km', (150, 150), (3, blocksize, 150), (blocksize, 150))
        my_expr_grad2 = contract_expression('ij,mi,kmj->km', (150, 150), (blocksize, 150), (3, blocksize, 150))

    if threads_per_block is None:
        # Determine the optimal number of blocks per grid
        max_threads_per_block = cuda.get_current_device().MAX_THREADS_PER_BLOCK
        thread_x = max_threads_per_block/16
        thread_y = max_threads_per_block/64
        threads_per_block = (thread_x, thread_y)
    else:
        thread_x = threads_per_block[0]
        thread_y = threads_per_block[1]


    if debug:
        cp.cuda.Stream.null.synchronize()
        durationPrelims += timer() - startPrelims

    # A large scratch for evaluating batch local ao values
    ao_value_block_ = cp.zeros((blocksize, max(count_nonzero_indices)), dtype=type)
    # If either x or c functional is of GGA/MGGA type we need ao_grad_values
    if xc_family_dict[x_family_code]!='LDA' or xc_family_dict[c_family_code]!='LDA':
        ao_value_grad_block_ = cp.zeros((3, blocksize, max(count_nonzero_indices)), dtype=type)

    # Just a precautionary synchronize here so that all the cupy arrays created above should finish getting initialized.
    # cp.cuda.Stream.null.synchronize()
    # cuda.synchronize()
    # with cupyx.profiler.profile():
    # with streams[0]:
    # Loop over blocks/batches of grid points
    for iblock in range(nblocks+1):
        # stream_id = iblock % nstreams
        # with streams[stream_id]:
        # with nb_streams[stream_id]:
            # cp.cuda.Device(stream_id).use()
            # cp.cuda.Stream(cp_streams[stream_id]).use()
        offset = iblock*blocksize

        # Get weights and coordinates of grid points for this block/batch
        weights_block = weights_cp[offset : min(offset+blocksize,ngrids)]
        coords_block = coords_cp[offset : min(offset+blocksize,ngrids)] 
        #coords_block_cp = cp.asarray(coords[offset : min(offset+blocksize,ngrids)])

        # Get the list of basis functions with significant contributions to this block
        if list_nonzero_indices is not None:
            non_zero_indices = list_nonzero_indices[iblock][0:count_nonzero_indices[iblock]]
            # Get the subset of density matrix corresponding to the siginificant basis functions
            dmat = dmat_orig_cp[cp.ix_(non_zero_indices, non_zero_indices)]

        if debug:
            startAO = timer()
        if xc_family_dict[x_family_code]=='LDA' and xc_family_dict[c_family_code]=='LDA':
            if list_ao_values is not None: # If ao_values are calculated once and saved, then they can be provided to avoid recalculation
                ao_value_block = list_ao_values[iblock]
            else:
                # ao_value_block = Integrals.eval_bfs(basis, coords_block)       
                if list_nonzero_indices is not None:
                    # ao_value_block = Integrals.bf_val_helpers.eval_bfs_sparse_vectorized_internal_cupy(bfs_coords[0], bfs_contr_prim_norms[0], bfs_nprim[0], bfs_lmn[0], bfs_coeffs, bfs_prim_norms, bfs_expnts, coords_block, non_zero_indices, scratches)
                    # Get a subset of the scratch for evaluating batch local ao values
                    ao_value_block = cp.asarray(ao_value_block_[0:coords_block.shape[0], 0:non_zero_indices.shape[0]]) 
                    # cp.cuda.Stream.null.synchronize() # Precautionary synchronize here
                    # blocks_per_grid = ((non_zero_indices.shape[0]//thread_x) + 1, (coords_block.shape[0]//thread_y) + 1) # "lazy" round-up
                    blocks_per_grid = ((non_zero_indices.shape[0] + (thread_x - 1))//thread_x, (coords_block.shape[0] + (thread_y - 1))//thread_y) 
                    # nb_stream = stream_cupy_to_numba(streams[stream_id])
                    # get the pointer to actual CUDA stream
                    # raw_str = streams[stream_id].ptr
                    # nb_stream = numba.cuda.external_stream(raw_str)
                    Integrals.bf_val_helpers.eval_bfs_sparse_internal_cuda[blocks_per_grid, threads_per_block](bfs_coords[0], bfs_contr_prim_norms[0], bfs_nprim[0], bfs_lmn[0], bfs_coeffs, bfs_prim_norms, bfs_expnts, coords_block, non_zero_indices, ao_value_block)
                    
                    # cuda.synchronize() # This is needed when using this in a cupy stream
                    # cp.cuda.Stream.null.synchronize() 
                else:
                    ao_value_block = Integrals.bf_val_helpers.eval_bfs_internal(bfs_coords[0], bfs_contr_prim_norms[0], bfs_nprim[0], bfs_lmn[0], bfs_coeffs, bfs_prim_norms, bfs_expnts, coords_block)
        # If either x or c functional is of GGA/MGGA type we need ao_grad_values
        if xc_family_dict[x_family_code]!='LDA' or xc_family_dict[c_family_code]!='LDA':
            if list_ao_values is not None: # If ao_values are calculated once and saved, then they can be provided to avoid recalculation
                ao_value_block, ao_values_grad_block = list_ao_values[iblock], list_ao_grad_values[iblock]
            else:
                # ao_value_block, ao_values_grad_block = Integrals.bf_val_helpers.eval_bfs_and_grad(basis, coords_block, deriv=1, parallel=True, non_zero_indices=non_zero_indices)
                if list_nonzero_indices is not None:
                    # Calculating ao values and gradients together, didn't really do much improvement in computational speed
                    # ao_value_block, ao_values_grad_block = Integrals.bf_val_helpers.eval_bfs_and_grad_sparse_internal(bfs_coords[0], bfs_contr_prim_norms[0], bfs_nprim[0], bfs_lmn[0], bfs_coeffs, bfs_prim_norms, bfs_expnts, coords_block, non_zero_indices)
                    # Get a subset of the scratch for evaluating batch local ao values
                    ao_value_block = cp.asarray(ao_value_block_[0:coords_block.shape[0], 0:non_zero_indices.shape[0]]) 
                    ao_values_grad_block = cp.asarray(ao_value_grad_block_[0:3, 0:coords_block.shape[0], 0:non_zero_indices.shape[0]]) 
                    blocks_per_grid = ((non_zero_indices.shape[0] + (thread_x - 1))//thread_x, (coords_block.shape[0] + (thread_y - 1))//thread_y) 
                    # nb_stream = stream_cupy_to_numba(streams[stream_id])
                    Integrals.bf_val_helpers.eval_bfs_and_grad_sparse_internal_cuda[blocks_per_grid, threads_per_block](bfs_coords[0], bfs_contr_prim_norms[0], bfs_nprim[0], 
                                                                                                            bfs_lmn[0], bfs_coeffs, bfs_prim_norms, bfs_expnts, 
                                                                                                            coords_block, non_zero_indices, ao_value_block, 
                                                                                                            ao_values_grad_block)
                    
                    # streams[stream_id].synchronize()
                    # nb_streams[stream_id].synchronize()
                    # cp.cuda.Stream.null.synchronize() 
                    # streams[stream_id].synchronize()
                    
                else:
                    ao_value_block, ao_values_grad_block = Integrals.bf_val_helpers.eval_bfs_and_grad_internal(bfs_coords[0], bfs_contr_prim_norms[0], bfs_nprim[0], bfs_lmn[0], bfs_coeffs, bfs_prim_norms, bfs_expnts, coords_block)
        

        # Convert to cupy array
        ao_value_block = cp.asarray(ao_value_block, dtype=type)
        if xc_family_dict[x_family_code]!='LDA' or xc_family_dict[c_family_code]!='LDA':
            ao_values_grad_block = cp.asarray(ao_values_grad_block, dtype=type)

        if debug:
            cp.cuda.Stream.null.synchronize()
            durationAO = durationAO + timer() - startAO

        if debug:
            startRho = timer()
            # print('calculating rho...')
        # rho_block = contract('ij,mi,mj->m', dmat, ao_value_block, ao_value_block, backend='cupy') # Original (pretty fast)
        rho_block = my_expr(dmat, ao_value_block, ao_value_block, backend='cupy')
        # X = ao_value_block @ dmat.T 
        # rho_block = cp.sum(ao_value_block*X,axis=1)
        # rho_block = contract('mi,im->m', ao_value_block, X)
        # If either x or c functional is of GGA/MGGA type we need rho_grad_values too
        if xc_family_dict[x_family_code]!='LDA' or xc_family_dict[c_family_code]!='LDA':
            # rho_grad_block_x, rho_grad_block_y, rho_grad_block_z  = ( contract('ij,kmi,mj->km',dmat,ao_values_grad_block,ao_value_block, backend='cupy', optimize='optimal')+\
            #                 contract('ij,mi,kmj->km',dmat,ao_value_block,ao_values_grad_block, backend='cupy', optimize='optimal') )[:]
            rho_grad_block_x, rho_grad_block_y, rho_grad_block_z  = my_expr_grad1(dmat,ao_values_grad_block,ao_value_block, backend='cupy') \
                                                                    + my_expr_grad2(dmat, ao_value_block, ao_values_grad_block, backend='cupy') 
            sigma_block = calc_sigma(rho_grad_block_x, rho_grad_block_y, rho_grad_block_z)
            if use_libxc:
                sigma_block = cp.asnumpy(sigma_block)

        if debug:
            cp.cuda.Stream.null.synchronize()
            durationRho = durationRho + timer() - startRho
            #print('Duration for Rho at grid points: ',durationRho)
            # print('done calculating rho')
    
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

            # Correlation
            # Calculate the necessary quantities using own implementation
            # retc = lda_c_vwn(rho_block)
            # retc = gga_c_lyp(rho_block, sigma_block)
            if xc_family_dict[c_family_code]=='LDA':
                retc = XC.func_compute(funcid[1], rho_block, use_gpu=True)
            else:
                retc = XC.func_compute(funcid[1], rho_block, sigma_block, use_gpu=True)
            if debug:
                cp.cuda.Stream.null.synchronize()
                durationLibxc = durationLibxc + timer() - startLibxc
            # print('Duration for LibXC computations at grid points: ',durationLibxc)


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
        den = rho_block*weights_block
        # den = cp.multiply(rho_block, weights_block)
        # den = elementwise_multiply(rho_block, weights_block)
        if use_libxc:
            efunc = efunc + cp.dot(den, cp.asarray(e)) #Multiply with functional values at grid points and sum
        else:
            efunc = efunc + cp.dot(den, e) #Multiply with functional values at grid points and sum
            # efunc = efunc + cp.sum(elementwise_multiply(den, e)) #Multiply with functional values at grid points and sum
        nelec += cp.sum(den)
        if debug:
            cp.cuda.Stream.null.synchronize()
            durationE = durationE + timer() - startE
        # print('Duration for calculation of total density functional energy: ',durationE)

        #POTENTIAL----------
        # The derivative of functional wrt density is vrho
        if debug:
            startF = timer()
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
            Fx = Ftemp*rho_grad_block_x
            Fy = Ftemp*rho_grad_block_y
            Fz = Ftemp*rho_grad_block_z
        if debug:
            cp.cuda.Stream.null.synchronize()
            durationF = durationF + timer() - startF
        # print('Duration for calculation of F: ',durationF)
        if debug:
            startZ = timer()
        # z = 0.5*F*ao_value_block.T
        # z = calc_z(F, ao_value_block.T)
        z = calc_z(weights_block, v_rho_temp, ao_value_block.T)
        # z = 0.5*np.einsum('m,mi->mi',F,ao_value_block)
        # If either x or c functional is of GGA/MGGA type we need rho_grad_values
        if xc_family_dict[x_family_code]!='LDA' or xc_family_dict[c_family_code]!='LDA':
            z += calc_z_gga(Fx, Fy, Fz, ao_values_grad_block[0].T, ao_values_grad_block[1].T, ao_values_grad_block[2].T)
            # z += Fx*ao_values_grad_block[0].T + Fy*ao_values_grad_block[1].T + Fz*ao_values_grad_block[2].T
        
            
        if debug:
            cp.cuda.Stream.null.synchronize()
            durationZ = durationZ + timer() - startZ
        # print('Duration for calculation of z : ',durationZ)
        # Free memory
        F = 0
        ao_value_block_T = 0
        vrho = 0

        if debug:
            startV = timer()
        
        #v_temp = z @ ao_value_block
        #v_temp = cp.dot(z, ao_value_block)
        v_temp = cp.matmul(z, ao_value_block)
        if list_nonzero_indices is not None:
            v[cp.ix_(non_zero_indices, non_zero_indices)] += v_temp + v_temp.T
        else:
            v += v_temp + v_temp.T

            
        if debug:
            cp.cuda.Stream.null.synchronize()
            durationV = durationV + timer() - startV

    # for stream in streams:
    #     stream.synchronize()


    print('Number of electrons: ', nelec)
    #print(v_temp.dtype)
    if debug:
        print('Duration for Prelims: ', durationPrelims)
        print('Duration for AO values: ', durationAO)
        print('Duration for V: ',durationV)
        print('Duration for Rho at grid points: ',durationRho)
        print('Duration for F: ',durationF)
        print('Duration for Z: ',durationZ)
        print('Duration for E: ',durationE)
        print('Duration for LibXC: ',durationLibxc)
        

        

    if debug:
        startV_tonumpy = timer()
    if debug:
        cp.cuda.Stream.null.synchronize()
        durationV_tonumpy = timer() - startV_tonumpy
        print('Duration for V to numpy', durationV_tonumpy)
        print('Total time: ', durationPrelims+durationAO+durationRho+durationLibxc+durationE+durationF+durationV+durationZ+durationV_tonumpy)


    if use_libxc:
        efunc = efunc[0]
    if debug:
        cp.cuda.Stream.null.synchronize()
        durationTotal += timer() - startTotal
        print('Total time2: ', durationTotal)

    if freemem or nstreams!=1:
        cp._default_memory_pool.free_all_blocks()


    # Final synchronize before return
    cp.cuda.Stream.null.synchronize()
    return efunc, v

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