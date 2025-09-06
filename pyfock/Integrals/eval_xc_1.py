import numpy as np
import numexpr
import pylibxc
from timeit import default_timer as timer
from pyfock import Integrals
from opt_einsum import contract

def eval_xc_1(basis, dmat, weights, coords, funcid=[1,7], spin=0, blocksize=50000, debug=False, list_nonzero_indices=None, count_nonzero_indices=None, list_ao_values=None, list_ao_grad_values=None):
    """
    Evaluate exchange-correlation (XC) energy and potential matrix for DFT
    using algorithm 1, which is a baseline method for grid-based DFT.

    In algorithm 1, the XC term is evaluated by looping over blocks of grid 
    points, and within each block, the operations are parallelized. This 
    approach is functional but generally slower than algorithm 2 for CPU-based 
    execution.

    This function evaluates the XC energy and potential matrix elements for a given
    density matrix using numerical integration over a 3D real-space grid. It supports 
    LDA and GGA functionals via LibXC and optional use of precomputed AO values and 
    gradients for performance gains. Sparse AO matrix techniques are also supported.

    Parameters
    ----------
    basis : Basis
        A basis object containing basis function data: exponents, coefficients, 
        angular momentum, normalization, and other AO metadata.

    dmat : np.ndarray
        The one-electron density matrix in the AO basis.

    weights : np.ndarray
        Integration weights associated with each grid point.

    coords : np.ndarray
        Grid point coordinates as an (N, 3) array.

    funcid : list of int, optional
        LibXC functional IDs. Default is [1, 7] for Slater (X) and VWN (C) (LDA).

    spin : int, optional
        Spin multiplicity: 0 for unpolarized. Spin-polarized (1) not currently supported.

    blocksize : int, optional
        Number of grid points to process per block. Default is 50000.

    debug : bool, optional
        If True, enables verbose timing and diagnostic output.

    list_nonzero_indices : list of np.ndarray, optional
        List of AO indices with non-negligible contributions in each grid block
        for sparse matrix optimizations.

    count_nonzero_indices : list of int, optional
        Number of significant AO indices per block; matches entries in `list_nonzero_indices`.

    list_ao_values : list of np.ndarray, optional
        Precomputed AO values at grid points for each block.

    list_ao_grad_values : list of tuple of np.ndarray, optional
        Precomputed AO gradient values (x, y, z) at grid points for each block.

    Returns
    -------
    efunc : float
        Total exchange-correlation energy.

    v : np.ndarray
        Exchange-correlation potential matrix in the AO basis.

    Notes
    -----
    - This algorithm prioritizes code clarity and correctness, not maximum speed.
    - Only LDA and GGA functionals are currently supported. meta-GGA and hybrid functionals
      are planned for future implementation.
    - Uses LibXC for exchange-correlation energy and potential evaluation.
    - AO values and gradients can be reused via precomputation to improve speed.
    - The number of electrons (via integrated density) is printed for validation.

    References
    ----------
    - LibXC Functional Codes: https://libxc.gitlab.io/functionals/
    - Functional energy and potential formulation: https://pubs.acs.org/doi/full/10.1021/ct200412r
    - LibXC Python interface: https://www.tddft.org/programs/libxc/manual/
    """
    # Evaluate the XC term
    # This is a slightly slower algorithm.
    # Here the parallelization is done when evaluating bfs or the density

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
    v = np.zeros((basis.bfs_nao, basis.bfs_nao))
    #TODO mGGA, Hybrid
    
    #Calculate number of blocks/batches
    ngrids = coords.shape[0]
    nblocks = ngrids//blocksize
    nelec = 0.0

    # If a list of significant basis functions for each block of grid points is provided
    if list_nonzero_indices is not None:
        dmat_orig = dmat

    ### Calculate stuff necessary for bf/ao evaluation on grid points
    ### Doesn't make any difference for 510 bfs but might be significant for >1000 bfs
    # This will help to make the call to eval_bfs faster by skipping the mediator eval_bfs function 
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

    # For debugging and benchmarking purposes
    durationLibxc = 0.0
    durationE = 0.0
    durationF = 0.0
    durationZ = 0.0
    durationV = 0.0
    durationRho = 0.0
    durationAO = 0.0

    xc_family_dict = {1:'LDA',2:'GGA',4:'MGGA'} 

    # Create a LibXC object  
    funcx = pylibxc.LibXCFunctional(funcid[0], "unpolarized")
    funcc = pylibxc.LibXCFunctional(funcid[1], "unpolarized")
    x_family_code = funcx.get_family()
    c_family_code = funcc.get_family()

    # Loop over blocks/batches of grid points
    for iblock in range(nblocks+1):
        offset = iblock*blocksize

        # Get weights and coordinates of grid points for this block/batch
        weights_block = weights[offset : min(offset+blocksize,ngrids)]
        coords_block = coords[offset : min(offset+blocksize,ngrids)] 

        # Get the list of basis functions with significant contributions to this block
        if list_nonzero_indices is not None:
            non_zero_indices = list_nonzero_indices[iblock][0:count_nonzero_indices[iblock]]
            # Get the subset of density matrix corresponding to the siginificant basis functions
            dmat = dmat_orig[np.ix_(non_zero_indices, non_zero_indices)]

        if debug:
            startAO = timer()
        if xc_family_dict[x_family_code]=='LDA' and xc_family_dict[c_family_code]=='LDA':
            if list_ao_values is not None: # If ao_values are calculated once and saved, then they can be provided to avoid recalculation
                ao_value_block = list_ao_values[iblock]
            else:
                # ao_value_block = Integrals.eval_bfs(basis, coords_block)       
                if list_nonzero_indices is not None:
                    # ao_value_block = Integrals.bf_val_helpers.eval_bfs_sparse_internal(bfs_coords[0], bfs_contr_prim_norms[0], bfs_nprim[0], bfs_lmn[0], bfs_coeffs, bfs_prim_norms, bfs_expnts, coords_block, non_zero_indices)
                    ao_value_block = Integrals.bf_val_helpers.eval_bfs_sparse_vectorized_internal(bfs_coords[0], bfs_contr_prim_norms[0], bfs_nprim[0], bfs_lmn[0], bfs_coeffs, bfs_prim_norms, bfs_expnts, coords_block, non_zero_indices)
                else:
                    ao_value_block = Integrals.bf_val_helpers.eval_bfs_internal(bfs_coords[0], bfs_contr_prim_norms[0], bfs_nprim[0], bfs_lmn[0], bfs_coeffs, bfs_prim_norms, bfs_expnts, coords_block)
        # If either x or c functional is of GGA/MGGA type we need ao_grad_values
        if xc_family_dict[x_family_code]!='LDA' or xc_family_dict[c_family_code]!='LDA':
            # ao_value_block, ao_values_grad_block = Integrals.bf_val_helpers.eval_bfs_and_grad(basis, coords_block, deriv=1, parallel=True, non_zero_indices=non_zero_indices)
            if list_nonzero_indices is not None:
                # Calculating ao values and gradients together, didn't really do much improvement in computational speed
                ao_value_block, ao_values_grad_block = Integrals.bf_val_helpers.eval_bfs_and_grad_sparse_internal(bfs_coords[0], bfs_contr_prim_norms[0], bfs_nprim[0], bfs_lmn[0], bfs_coeffs, bfs_prim_norms, bfs_expnts, coords_block, non_zero_indices)
            else:
                ao_value_block, ao_values_grad_block = Integrals.bf_val_helpers.eval_bfs_and_grad_internal(bfs_coords[0], bfs_contr_prim_norms[0], bfs_nprim[0], bfs_lmn[0], bfs_coeffs, bfs_prim_norms, bfs_expnts, coords_block)
        if debug:
            durationAO = durationAO + timer() - startAO
            

        if debug:
            startRho = timer()
        
        if list_nonzero_indices is not None:
            rho_block = contract('ij,mi,mj->m', dmat, ao_value_block, ao_value_block) # Original (pretty fast)
        else:
            rho_block = Integrals.bf_val_helpers.bf_val_helpers.eval_rho(ao_value_block, dmat) # This is by-far the fastest now <-----
        
        # If either x or c functional is of GGA/MGGA type we need rho_grad_values too
        if xc_family_dict[x_family_code]!='LDA' or xc_family_dict[c_family_code]!='LDA':
            rho_grad_block_x = contract('ij,mi,mj->m',dmat,ao_values_grad_block[0],ao_value_block)+\
                                    contract('ij,mi,mj->m',dmat,ao_value_block,ao_values_grad_block[0])
            rho_grad_block_y = contract('ij,mi,mj->m',dmat,ao_values_grad_block[1],ao_value_block)+\
                                    contract('ij,mi,mj->m',dmat,ao_value_block,ao_values_grad_block[1])
            rho_grad_block_z = contract('ij,mi,mj->m',dmat,ao_values_grad_block[2],ao_value_block)+\
                                    contract('ij,mi,mj->m',dmat,ao_value_block,ao_values_grad_block[2])
            sigma_block = np.zeros((3,weights_block.shape[0]))
            sigma_block[1] = rho_grad_block_x**2 + rho_grad_block_y**2 + rho_grad_block_z**2

        if debug:
            durationRho = durationRho + timer() - startRho
    
        #LibXC stuff
        # Exchange
        if debug:
            startLibxc = timer()
        # Input dictionary for libxc
        inp = {}
        # Input dictionary needs density values at grid points
        inp['rho'] = rho_block
        if xc_family_dict[x_family_code]!='LDA':
            # Input dictionary needs sigma (\nabla \rho \cdot \nabla \rho) values at grid points
            inp['sigma'] = sigma_block[1]
        # Calculate the necessary quantities using LibXC
        retx = funcx.compute(inp)
        # print('Duration for LibXC computations at grid points: ',durationLibxc)

        # Correlation
        # Input dictionary for libxc
        inp = {}
        # Input dictionary needs density values at grid points
        inp['rho'] = rho_block
        if xc_family_dict[c_family_code]!='LDA':
            # Input dictionary needs sigma (\nabla \rho \cdot \nabla \rho) values at grid points
            inp['sigma'] = sigma_block[1]
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
        efunc = efunc + np.dot(den, e) #Multiply with functional values at grid points and sum
        nelec = nelec + np.sum(den)

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
        
        if debug:
            startF = timer()
        # F = np.multiply(weights_block,vrho[:,0]) #This is fast enough.
        v_rho_temp = vrho[:,0]
        F = numexpr.evaluate('(weights_block*v_rho_temp)')
        # If either x or c functional is of GGA/MGGA type we need rho_grad_values
        if xc_family_dict[x_family_code]!='LDA' or xc_family_dict[c_family_code]!='LDA':
            Ftemp = 2*weights_block*vsigma[:,0]
            Fx = Ftemp*rho_grad_block_x
            Fy = Ftemp*rho_grad_block_y
            Fz = Ftemp*rho_grad_block_z
            # Ftemp = 2*np.multiply(weights_block,vsigma.T)
            # Fx = Ftemp*rho_grad_block_x
            # Fy = Ftemp*rho_grad_block_y
            # Fz = Ftemp*rho_grad_block_z
        if debug:
            durationF = durationF + timer() - startF
        # print('Duration for calculation of F: ',durationF)

        if debug:
            startZ = timer()
        ao_value_block_T = ao_value_block.T
        z = numexpr.evaluate('(0.5*F*ao_value_block_T)')
        # z = 0.5*np.einsum('m,mi->mi',F,ao_value_block)
        # If either x or c functional is of GGA/MGGA type we need rho_grad_values
        if xc_family_dict[x_family_code]!='LDA' or xc_family_dict[c_family_code]!='LDA':
            z = z + Fx*ao_values_grad_block[0].T + Fy*ao_values_grad_block[1].T + Fz*ao_values_grad_block[2].T
            
        if debug:
            durationZ = durationZ + timer() - startZ
        # print('Duration for calculation of z : ',durationZ)
        # Free memory
        F = 0
        ao_value_block_T = 0
        vrho = 0

        if debug:
            startV = timer()
        # Numexpr
        v_temp = z @ ao_value_block  
        v_temp_T = v_temp.T
        if list_nonzero_indices is not None:
            v[np.ix_(non_zero_indices, non_zero_indices)] += numexpr.evaluate('(v_temp + v_temp_T)')
        else:
            v = numexpr.evaluate('(v + v_temp + v_temp_T)')

        
        if debug:
            durationV = durationV + timer() - startV

    print('Number of electrons: ', nelec)

    if debug:
        print('Duration for AO values: ', durationAO)
        print('Duration for V: ',durationV)
        print('Duration for Rho at grid points: ',durationRho)
        print('Duration for F: ',durationF)
        print('Duration for Z: ',durationZ)
        print('Duration for E: ',durationE)
        print('Duration for LibXC: ',durationLibxc)

    

    return efunc[0], v