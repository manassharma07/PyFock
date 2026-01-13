import pyfock.Mol as Mol
import pyfock.Basis as Basis
import pyfock.Integrals as Integrals
from timeit import default_timer as timer
import numba
import numpy as np
from numpy.linalg import eig, multi_dot as dot
import scipy 
try:
    import cupy as cp
    from cupy import fuse
    import cupyx
    CUPY_AVAILABLE = True
except Exception as e:
    CUPY_AVAILABLE = False
    pass
from threadpoolctl import ThreadpoolController, threadpool_info, threadpool_limits
from opt_einsum import contract

def density_fitting_prelims_for_DFT_development(mol, basis, auxbasis, T, dmat, use_gpu=False, keep_ints3c2e_in_gpu=True, threshold_schwarz=1e-9, strict_schwarz=False, rys=True, DF_algo=6, cholesky=True):
    startCoulomb = timer()
    if not rys:
        if DF_algo>4:
            print('ERROR: Density fitting algorithms 4 and higher are only supported if rys quadrature is enabled! Please pass rys=True to the scf function.')
            exit()
    
    # List of stuff that would be returned
    H = None # In case of strict schwarz and algorithm 6
    V = None
    ints3c2e = None
    ints2c2e = None
    nsignificant = None
    indicesA = None
    indicesB = None
    indicesC = None
    offsets_3c2e = None
    indices = None
    ints4c2e_diag = None
    sqrt_ints4c2e_diag = None
    sqrt_diag_ints2c2e = None
    indices_dmat_tri = None
    indices_dmat_tri_2 = None
    df_coeff0 = None
    Qpq = None
    cho_decomp_ints2c2e = None
    durationDF_cholesky = 0


    print('Stricter version of Schwarz screening: ', strict_schwarz, flush=True)
    print('\nCalculating three centered two electron and two-centered two-electron integrals...\n\n', flush=True)
    if rys:
        start2c2e = timer()
        if not use_gpu:
            ints2c2e = Integrals.rys_2c2e_symm(auxbasis)
        else:
            ints2c2e = Integrals.rys_2c2e_symm_cupy(auxbasis)
        duration2c2e = timer() - start2c2e
        print('Time taken for two-centered two-electron integrals '+str(round(duration2c2e, 2))+' seconds.\n', flush=True)
        if DF_algo==4: #Triangular version
            ints3c2e = Integrals.rys_3c2e_tri(basis, auxbasis)
            # ints3c2e_pyscf = Integrals.rys_3c2e_tri(basis, auxbasis)
            # print(ints3c2e_pyscf.shape)
        elif DF_algo==5:
            print('\n\nPerforming Schwarz screening...')
            # threshold_schwarz = 1e-11
            print('Threshold ', threshold_schwarz)
            startSchwarz = timer()
            nints3c2e_tri = int(basis.bfs_nao*(basis.bfs_nao+1)/2.0)*auxbasis.bfs_nao
            nints3c2e = basis.bfs_nao*basis.bfs_nao*auxbasis.bfs_nao
            # This is based on Schwarz inequality screening
            # Diagonal elements of ERI 4c2e array
            ints4c2e_diag = Integrals.schwarz_helpers.eri_4c2e_diag(basis)
            # Calculate the indices of the ints3c2e array based on Schwarz inequality
            indices_temp = Integrals.schwarz_helpers.calc_indices_3c2e_schwarz(ints4c2e_diag, ints2c2e, basis.bfs_nao, auxbasis.bfs_nao, threshold_schwarz)
            print('Size of temporary array used to calculate the significant indices of 3c2e ERI in GB ', indices_temp.nbytes/1e9, flush=True)
            if basis.bfs_nao<65500 and auxbasis.bfs_nao<65500:
                indices = [a.astype(np.uint16) for a in indices_temp.nonzero()] # Will work as long as the no. of max(Bfs,auxbfs) is less than 65535
            else:
                indices = [a.astype(np.uint32) for a in indices_temp.nonzero()] # Will work as long as the no. of Bfs/auxbfs is less than 4294967295
            
            # Get rid of temp variables
            indices_temp=0
            
            print('Size of permanent array storing the significant indices of 3c2e ERI in GB ', indices[0].nbytes/1e9+indices[1].nbytes/1e9+indices[2].nbytes/1e9, flush=True)

            nsignificant = len(indices[0])
            print('No. of elements in the standard three-centered two electron ERI tensor: ', nints3c2e, flush=True)
            print('No. of elements in the triangular three-centered two electron ERI tensor: ', nints3c2e_tri, flush=True)
            print('No. of significant triplets based on Schwarz inequality and triangularity: ' + str(nsignificant) + ' or '+str(np.round(nsignificant/nints3c2e*100,1)) + '% of original', flush=True)
            print('Schwarz screening done!')
            durationSchwarz = timer() - startSchwarz
            print('Time taken '+str(round(durationSchwarz, 2))+' seconds.\n', flush=True)
            
            ints3c2e = Integrals.schwarz_helpers.rys_3c2e_tri_schwarz(basis, auxbasis, indices[0], indices[1], indices[2])
            # ints3c2e = Integrals.schwarz_helpers.rys_3c2e_tri_schwarz_sparse(basis, auxbasis, indices[0], indices[1], indices[2])
            # print('here')
            # indices_ssss = []
            # for itemp in range(indices[0].shape[0]):
            #     i = indices[0][itemp]
            #     j = indices[1][itemp] 
            #     offset = int(i*(i+1)/2)
            #     indices_ssss.append(j + offset)
            # coords = np.array([np.array(indices_ssss), indices[2]])
            # print(coords)
            # print(indices)
            # ints3c2e = sparse.COO(coords, ints3c2e, shape=(int(basis.bfs_nao*(basis.bfs_nao+1)/2.0), auxbasis.bfs_nao))
            # print('here')
            # Get rid of variables no longer needed
            indices=0
        elif DF_algo==6:
            # TODO: The Schwarz can also be made more efficient (work on this)
            # Takes a lot of time and most of the stuff is only done serially
            # Also gets slower with more number of cores
            print('\n\nPerforming Schwarz screening...')
            # threshold_schwarz = 1e-09
            print('Threshold ', threshold_schwarz)
            startSchwarz = timer()
            nints3c2e_tri = int(basis.bfs_nao*(basis.bfs_nao+1)/2.0)*auxbasis.bfs_nao
            nints3c2e = basis.bfs_nao*basis.bfs_nao*auxbasis.bfs_nao
            # This is based on Schwarz inequality screening
            # Diagonal elements of ERI 4c2e array
            duration_4c2e_diag = 0.0
            start_4c2e_diag = timer()
            ints4c2e_diag = Integrals.schwarz_helpers.eri_4c2e_diag(basis)
            duration_4c2e_diag = timer() - start_4c2e_diag
            print('Time taken to evaluate the "diagonal" of 4c2e ERI tensor: ', round(duration_4c2e_diag, 2))
            
            # Calculate the square roots required for 
            duration_square_roots = 0.0
            start_square_roots = timer()
            sqrt_ints4c2e_diag = np.sqrt(np.abs(ints4c2e_diag))
            sqrt_diag_ints2c2e = np.sqrt(np.abs(np.diag(ints2c2e)))
            duration_square_roots = timer() - start_square_roots
            print('Time taken to evaluate the square roots needed: ', round(duration_square_roots, 2))
            # print(ints4c2e_diag.max())
            # print(ints4c2e_diag.min())
            # print(ints2c2e.max())
            # print(np.diag(ints2c2e).min())

            # Create a mask for dmat based on sqrt_ints4c2e_diag
            # Create a boolean mask for elements satisfying the condition
            # mask = sqrt_ints4c2e_diag**2 < 1e-17

            # bfs_coords = np.array([basis.bfs_coords])[0]
            # bfs_radius_cutoff = np.array([basis.bfs_radius_cutoff])[0]
            auxbfs_lm = np.array(auxbasis.bfs_lm)

            # Calculate the indices of the ints3c2e array based on Schwarz inequality
            # indices_temp = Integrals.schwarz_helpers.calc_indices_3c2e_schwarz(ints4c2e_diag, ints2c2e, basis.bfs_nao, auxbasis.bfs_nao, threshold_schwarz)
            # print('Size of temporary array used to calculate the significant indices of 3c2e ERI in GB ', indices_temp.nbytes/1e9, flush=True)
            # print('here2')
            # if basis.bfs_nao<65500 and auxbasis.bfs_nao<65500:
            #     indices = [a.astype(np.uint16) for a in indices_temp.nonzero()] # Will work as long as the no. of max(Bfs,auxbfs) is less than 65535
            # else:
            #     indices = [a.astype(np.uint32) for a in indices_temp.nonzero()] # Will work as long as the no. of Bfs/auxbfs is less than 4294967295
            chunksize = int(1e9) # Results in 2 GB chunks
            duration_indices_calc = 0.0
            duration_concatenation = 0.0
            start_indices_calc = timer()
            # indices_temp = []
            ijk = [0, 0, 0]
            if chunksize<nints3c2e_tri:
                nchunks = nints3c2e_tri//chunksize + 1
            else:
                nchunks=1
            indicesA = None
            for ichunk in range(nchunks):
                # indices_temp = Integrals.schwarz_helpers.calc_indices_3c2e_schwarz2(sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, min(chunksize, nints3c2e_tri), basis.bfs_nao, auxbasis.bfs_nao, ijk[0], ijk[1], ijk[2], threshold_schwarz, strict_schwarz)
                # The following leads to slightly more savings in terms of number of integrals calculated and stored as well as the number of indices to be stored
                indices_temp = Integrals.schwarz_helpers.calc_indices_3c2e_schwarz_fine(sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, min(chunksize, nints3c2e_tri), basis.bfs_nao, auxbasis.bfs_nao, ijk[0], ijk[1], ijk[2], threshold_schwarz, strict_schwarz, auxbfs_lm)
                
                ijk = indices_temp[3]
                count = indices_temp[4]
                # start_concatenation = timer()
                if indicesA is not None and count>0:
                    indicesA = np.concatenate([indicesA, indices_temp[0][0:count]])
                    indicesB = np.concatenate([indicesB, indices_temp[1][0:count]])
                    indicesC = np.concatenate([indicesC, indices_temp[2][0:count]])
                else:
                    
                    indicesA = indices_temp[0][0:count]
                    indicesB = indices_temp[1][0:count]
                    indicesC = indices_temp[2][0:count]
                # duration_concatenation += timer() - start_concatenation
                # Get rid of temp variables
                    del indices_temp
                    
                # Break out of the for loop if the nol. of significant triplets found is less than the chunksize 
                # This is because, it means that there are no more significant triplets to be found from all possible configurations. 
                if count<chunksize: 
                    break
            
            duration_indices_calc += timer() - start_indices_calc
            print('Time for significant indices evaluation: ', round(duration_indices_calc, 2))
            # print('Time for array concatenation: ', duration_concatenation)

            # Get rid of temp variables
            count = None
            indices_temp = None
            ijk = None
            del indices_temp
            del ijk
            del count
            
            print('Size of permanent array storing the significant indices of 3c2e ERI in GB ', indicesA.nbytes/1e9+indicesB.nbytes/1e9+indicesC.nbytes/1e9, flush=True)

            nsignificant = len(indicesA)
            print('No. of elements in the standard three-centered two electron ERI tensor: ', nints3c2e, flush=True)
            print('No. of elements in the triangular three-centered two electron ERI tensor: ', nints3c2e_tri, flush=True)
            print('No. of significant triplets based on Schwarz inequality and triangularity: ' + str(nsignificant) + ' or '+str(np.round(nsignificant/nints3c2e*100,1)) + '% of original', flush=True)
            print('Schwarz screening done!')
            durationSchwarz = timer() - startSchwarz
            print('Total time taken for Schwarz screening '+str(round(durationSchwarz, 2))+' seconds.\n', flush=True)
            
            ints3c2e = Integrals.schwarz_helpers.rys_3c2e_tri_schwarz_sparse(basis, auxbasis, indicesA, indicesB, indicesC)

            if strict_schwarz:
                start_strict_schwarz_nuc_mat = timer()
                if not use_gpu:
                    V = Integrals.nuc_mat_symm(basis, mol, None, sqrt_ints4c2e_diag)
                else:
                    V = Integrals.nuc_mat_symm_cupy(basis, mol, None, None, sqrt_ints4c2e_diag) 
                    # V = cp.asarray(Integrals.nuc_mat_symm(basis, mol, None, sqrt_ints4c2e_diag))
                H = T + V
                duration_strict_schwarz_nuc_mat = timer() - start_strict_schwarz_nuc_mat
                print('Time taken to evaluate the nuclear potential matrix with strict Schwarz screening: ', round(duration_strict_schwarz_nuc_mat, 2), flush=True)
            # The following uses joblib to parallelize instead
            # ints3c2e = Integrals.schwarz_helpers.rys_3c2e_tri_schwarz_sparse2(basis, auxbasis, indicesA, indicesB, indicesC, ncores)
            # print(ints3c2e)

            # Some tests to see how many useless integrals are still being evaluated after Schwarz screening
            # mask = np.abs(ints3c2e) < 1e-8
            # count_below_threshold = np.count_nonzero(mask)

            # print("Number of elements below 1e-9:", count_below_threshold)
            # print('Percentage of total 3c2e integrals: ', np.round(count_below_threshold/nints3c2e*100,1))
            # ints3c2e[mask] = 0.0
            
            
        elif DF_algo==7:
            print('\n\nPerforming Schwarz screening...')
            # threshold_schwarz = 1e-11
            print('Threshold ', threshold_schwarz)
            startSchwarz = timer()
            nints3c2e_tri = int(basis.bfs_nao*(basis.bfs_nao+1)/2.0)*auxbasis.bfs_nao
            nints3c2e = basis.bfs_nao*basis.bfs_nao*auxbasis.bfs_nao
            # This is based on Schwarz inequality screening
            # Diagonal elements of ERI 4c2e array
            ints4c2e_diag = Integrals.schwarz_helpers.eri_4c2e_diag(basis)
            bfs_shell_index = np.array([basis.bfs_shell_index], dtype=np.uint16)[0]
            print(bfs_shell_index)
            auxbfs_shell_index = np.array([auxbasis.bfs_shell_index], dtype=np.uint16)[0]
            bfs_nbfshell = np.array([basis.bfs_nbfshell], dtype=np.uint8)[0]
            auxbfs_nbfshell = np.array([auxbasis.bfs_nbfshell], dtype=np.uint8)[0]
            # Calculate significant shell indices
            chunksize = int(1e9)
            # indices_temp = []
            ijk = [0, 0, 0]
            if chunksize<nints3c2e_tri:
                nchunks = nints3c2e_tri//chunksize + 1
            else:
                nchunks=1
            indicesA = None
            for ichunk in range(nchunks):
                indices_temp = Integrals.schwarz_helpers.calc_indices_3c2e_schwarz_shells(ints4c2e_diag, ints2c2e, min(chunksize, nints3c2e_tri), basis.bfs_nao, auxbasis.bfs_nao, ijk[0], ijk[1], ijk[2], bfs_shell_index, auxbfs_shell_index, threshold_schwarz)
                ijk = indices_temp[3]
                count = indices_temp[4]
                if indicesA is not None and count>0:
                    indicesA_shell = np.concatenate([indicesA, indices_temp[0][0:count]])
                    indicesB_shell = np.concatenate([indicesB, indices_temp[1][0:count]])
                    indicesC_shell = np.concatenate([indicesC, indices_temp[2][0:count]])
                else:
                    
                    indicesA_shell = indices_temp[0][0:count]
                    indicesB_shell = indices_temp[1][0:count]
                    indicesC_shell = indices_temp[2][0:count]
                # Break out of the for loop if the nol. of significant triplets found is less than the chunksize 
                # This is because, it means that there are no more significant triplets to be found from all possible configurations. 
                if count<chunksize: 
                    break

            # Get rid of temp variables
            indices_temp=0
            ijk = 0
            
            print('Size of permanent array storing the significant shell indices of 3c2e ERI in GB ', indicesA_shell.nbytes/1e9+indicesB_shell.nbytes/1e9+indicesC_shell.nbytes/1e9, flush=True)
            print(indicesA_shell)
            print(len(indicesA_shell))
            print(indicesB_shell)
            print(len(indicesB_shell))
            print(indicesC_shell)
            print(len(indicesC_shell))
            nsignificant = np.sum(np.multiply(np.multiply(bfs_nbfshell[indicesA_shell], bfs_nbfshell[indicesB_shell]), auxbfs_nbfshell[indicesC_shell]))
            print(bfs_nbfshell[indicesA_shell]) 
            print(np.sum(basis.shells))
            print(np.sum(auxbasis.shells))
            print('No. of elements in the standard three-centered two electron ERI tensor: ', nints3c2e, flush=True)
            print('No. of elements in the triangular three-centered two electron ERI tensor: ', nints3c2e_tri, flush=True)
            print('No. of significant triplets based on Schwarz inequality and triangularity: ' + str(nsignificant) + ' or '+str(np.round(nsignificant/nints3c2e*100,1)) + '% of original', flush=True)
            print('Schwarz screening done!')
            durationSchwarz = timer() - startSchwarz
            print('Time taken '+str(round(durationSchwarz, 2))+' seconds.\n', flush=True)
            

            ints3c2e = Integrals.schwarz_helpers.rys_3c2e_tri_schwarz_sparse(basis, auxbasis, indicesA, indicesB, indicesC)
        elif DF_algo==8:
            print('\n\nCalculating total no. of non-negligible three-centered two electron integrals based on Schwarz inequality...')
            print('Schwarz threshold ', threshold_schwarz)
            startSchwarz = timer()
            nints3c2e_tri = int(basis.bfs_nao*(basis.bfs_nao+1)/2.0)*auxbasis.bfs_nao
            nints3c2e = basis.bfs_nao*basis.bfs_nao*auxbasis.bfs_nao
            # This is based on Schwarz inequality screening
            # Diagonal elements of ERI 4c2e array
            duration_4c2e_diag = 0.0
            start_4c2e_diag = timer()
            ints4c2e_diag = Integrals.schwarz_helpers.eri_4c2e_diag(basis)
            duration_4c2e_diag = timer() - start_4c2e_diag
            print('Time taken to evaluate the "diagonal" of 4c2e ERI tensor: ', round(duration_4c2e_diag, 2))
            
            # Calculate the square roots required for 
            duration_square_roots = 0.0
            start_square_roots = timer()
            sqrt_ints4c2e_diag = np.sqrt(np.abs(ints4c2e_diag))
            sqrt_diag_ints2c2e = np.sqrt(np.abs(np.diag(ints2c2e)))
            duration_square_roots = timer() - start_square_roots
            print('Time taken to evaluate the square roots needed: ', round(duration_square_roots, 2))

            start_indices_calc = timer()
            count_significant = Integrals.schwarz_helpers.calc_count_3c2e_schwarz(sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, basis.bfs_nao, auxbasis.bfs_nao, threshold_schwarz)
            
            duration_indices_calc = timer() - start_indices_calc
            print('Time for total significant integral count calculation: ', round(duration_indices_calc, 2))
            # print('Time for array concatenation: ', duration_concatenation)

            
            print('No. of elements in the standard three-centered two electron ERI tensor: ', nints3c2e, flush=True)
            print('No. of elements in the triangular three-centered two electron ERI tensor: ', nints3c2e_tri, flush=True)
            print('No. of significant triplets based on Schwarz inequality and triangularity: ' + str(count_significant) + ' or '+str(np.round(count_significant/nints3c2e*100,1)) + '% of original', flush=True)
            print('Schwarz screening partially done!')
            durationSchwarz = timer() - startSchwarz
            print('Total time taken for Schwarz screening (partial) '+str(round(durationSchwarz, 2))+' seconds.\n', flush=True)
            
            # The following works alright, except it is not very parallel efficient (this uses prange)
            ints3c2e = Integrals.schwarz_helpers.rys_3c2e_tri_schwarz_sparse_algo8(basis, auxbasis, count_significant, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold_schwarz)
        elif DF_algo==9:
            print('\n\nPerforming Schwarz screening...')
            print('Threshold ', threshold_schwarz)
            startSchwarz = timer()
            nints3c2e_tri = int(basis.bfs_nao*(basis.bfs_nao+1)/2.0)*auxbasis.bfs_nao
            nints3c2e = basis.bfs_nao*basis.bfs_nao*auxbasis.bfs_nao
            # This is based on Schwarz inequality screening
            # Diagonal elements of ERI 4c2e array
            duration_4c2e_diag = 0.0
            start_4c2e_diag = timer()
            ints4c2e_diag = Integrals.schwarz_helpers.eri_4c2e_diag(basis)
            duration_4c2e_diag = timer() - start_4c2e_diag
            print('Time taken to evaluate the "diagonal" of 4c2e ERI tensor: ', round(duration_4c2e_diag, 2))
            
            # Calculate the square roots required for 
            duration_square_roots = 0.0
            start_square_roots = timer()
            sqrt_ints4c2e_diag = np.sqrt(np.abs(ints4c2e_diag))
            sqrt_diag_ints2c2e = np.sqrt(np.abs(np.diag(ints2c2e)))
            duration_square_roots = timer() - start_square_roots
            print('Time taken to evaluate the square roots needed: ', round(duration_square_roots, 2))

            chunksize = int(1e9) # Results in 2 GB chunks
            duration_indices_calc = 0.0
            duration_concatenation = 0.0
            start_indices_calc = timer()
            # indices_temp = []
            ijk = [0, 0, 0]
            if chunksize<nints3c2e_tri:
                nchunks = nints3c2e_tri//chunksize + 1
            else:
                nchunks=1
            indicesB = None
            for ichunk in range(nchunks):
                indices_temp = Integrals.schwarz_helpers.calc_indices_3c2e_schwarz3(sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, min(chunksize, nints3c2e_tri), basis.bfs_nao, auxbasis.bfs_nao, ijk[0], ijk[1], ijk[2], threshold_schwarz)
                ijk = indices_temp[3]
                count = indices_temp[4]
                print(count)
                # start_concatenation = timer()
                if indicesB is not None and count>0:
                    offset = np.concatenate([offset, indices_temp[0][0:np.argmax(indices_temp[0])]])
                    indicesB = np.concatenate([indicesB, indices_temp[1][0:count]])
                    indicesC = np.concatenate([indicesC, indices_temp[2][0:count]])
                else:
                    
                    offset = indices_temp[0][0:np.argmax(indices_temp[0])]
                    indicesB = indices_temp[1][0:count]
                    indicesC = indices_temp[2][0:count]
                # duration_concatenation += timer() - start_concatenation
                # Break out of the for loop if the nol. of significant triplets found is less than the chunksize 
                # This is because, it means that there are no more significant triplets to be found from all possible configurations. 
                if count<chunksize: 
                    break
            
            duration_indices_calc += timer() - start_indices_calc
            print('Time for significant indices evaluation: ', duration_indices_calc)
            # print('Time for array concatenation: ', duration_concatenation)

            # Get rid of temp variables
            indices_temp=0
            ijk = 0
            
            print('Size of permanent array storing the significant indices of 3c2e ERI in GB ', offset.nbytes/1e9+indicesB.nbytes/1e9+indicesC.nbytes/1e9, flush=True)

            nsignificant = len(indicesB)
            print('No. of elements in the standard three-centered two electron ERI tensor: ', nints3c2e, flush=True)
            print('No. of elements in the triangular three-centered two electron ERI tensor: ', nints3c2e_tri, flush=True)
            print('No. of significant triplets based on Schwarz inequality and triangularity: ' + str(nsignificant) + ' or '+str(np.round(nsignificant/nints3c2e*100,1)) + '% of original', flush=True)
            print('Schwarz screening done!')
            durationSchwarz = timer() - startSchwarz
            print('Total time taken for Schwarz screening '+str(round(durationSchwarz, 2))+' seconds.\n', flush=True)
            
            # print(offset)
            ints3c2e = Integrals.schwarz_helpers.rys_3c2e_tri_schwarz_sparse_algo9(basis, auxbasis, offset, indicesB, indicesC)
            # The following uses joblib to parallelize instead
            # ints3c2e = Integrals.schwarz_helpers.rys_3c2e_tri_schwarz_sparse2(basis, auxbasis, indicesA, indicesB, indicesC, ncores)
            
            # print(ints3c2e)

            # Some tests to see how many useless integrals are still being evaluated after Schwarz screening
            # mask = np.abs(ints3c2e) < 1e-9
            # count_below_threshold = np.count_nonzero(mask)

            # print("Number of elements below 1e-9:", count_below_threshold)
            # print('Percentage of total calculated: ', count_below_threshold/nsignificant*100)
            # ints3c2e[mask] = 0.0
        elif DF_algo==10:
            # TODO: The Schwarz can also be made more efficient (work on this)
            # Takes a lot of time and most of the stuff is only done serially
            # Also gets slower with more number of cores
            print('\n\nPerforming Schwarz screening...')
            # threshold_schwarz = 1e-09
            print('Threshold ', threshold_schwarz)
            startSchwarz = timer()
            nints3c2e_tri = int(basis.bfs_nao*(basis.bfs_nao+1)/2.0)*auxbasis.bfs_nao
            nints3c2e = basis.bfs_nao*basis.bfs_nao*auxbasis.bfs_nao
            # This is based on Schwarz inequality screening
            # Diagonal elements of ERI 4c2e array
            duration_4c2e_diag = 0.0
            start_4c2e_diag = timer()
            if not use_gpu:
                ints4c2e_diag = Integrals.schwarz_helpers.eri_4c2e_diag(basis)
            else:
                ints4c2e_diag = Integrals.schwarz_helpers_cupy.eri_4c2e_diag_cupy(basis)
            # ints4c2e_diag = Integrals.schwarz_helpers.eri_4c2e_diag(basis)
            duration_4c2e_diag = timer() - start_4c2e_diag
            print('Time taken to evaluate the "diagonal" of 4c2e ERI tensor: ', round(duration_4c2e_diag, 2))
            
            # Calculate the square roots required for 
            duration_square_roots = 0.0
            start_square_roots = timer()
            if use_gpu:
                sqrt_ints4c2e_diag = cp.sqrt(np.abs(ints4c2e_diag))
                sqrt_diag_ints2c2e = cp.sqrt(np.abs(np.diag(ints2c2e)))
            else:
                sqrt_ints4c2e_diag = np.sqrt(np.abs(ints4c2e_diag))
                sqrt_diag_ints2c2e = np.sqrt(np.abs(np.diag(ints2c2e)))
            duration_square_roots = timer() - start_square_roots
            print('Time taken to evaluate the square roots needed: ', round(duration_square_roots, 2))
            
            auxbfs_lm = np.array(auxbasis.bfs_lm)



            # Calculate the indices of the ints3c2e array based on Schwarz inequality
            duration_indices_calc = 0.0
            start_indices_calc = timer()
            indicesA, indicesB = np.tril_indices_from(dmat)
            if use_gpu:
                offsets_3c2e = Integrals.schwarz_helpers.calc_offsets_3c2e_schwarz(cp.asnumpy(sqrt_ints4c2e_diag), cp.asnumpy(sqrt_diag_ints2c2e), threshold_schwarz, strict_schwarz, auxbfs_lm,  indicesA.shape[0] , auxbasis.bfs_nao, indicesA, indicesB)
            else:
                offsets_3c2e = Integrals.schwarz_helpers.calc_offsets_3c2e_schwarz(sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold_schwarz, strict_schwarz, auxbfs_lm,  indicesA.shape[0] , auxbasis.bfs_nao, indicesA, indicesB)
            nsignificant = np.sum(offsets_3c2e)
            offsets_3c2e = np.cumsum(offsets_3c2e)
            duration_indices_calc += timer() - start_indices_calc
            print('Time for significant indices evaluation: ', duration_indices_calc)
            # print('Time for array concatenation: ', duration_concatenation)
            
            print('Size of permanent array storing the significant indices of 3c2e ERI in GB ', indicesA.nbytes/1e9+indicesB.nbytes/1e9+offsets_3c2e.nbytes/1e9, flush=True)

            
            print('No. of elements in the standard three-centered two electron ERI tensor: ', nints3c2e, flush=True)
            print('No. of elements in the triangular three-centered two electron ERI tensor: ', nints3c2e_tri, flush=True)
            print('No. of significant triplets based on Schwarz inequality and triangularity: ' + str(nsignificant) + ' or '+str(np.round(nsignificant/nints3c2e*100,1)) + '% of original', flush=True)
            print('Schwarz screening done!')
            durationSchwarz = timer() - startSchwarz
            print('Total time taken for Schwarz screening '+str(round(durationSchwarz, 2))+' seconds.\n', flush=True)

            if use_gpu:
                ints3c2e = Integrals.schwarz_helpers_cupy.rys_3c2e_tri_schwarz_sparse_algo10_cupy(basis, auxbasis, indicesA, indicesB, offsets_3c2e, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold_schwarz, strict_schwarz, nsignificant)
                if not keep_ints3c2e_in_gpu:
                    ints3c2e = cp.asnumpy(ints3c2e)
            else:
                ints3c2e = Integrals.schwarz_helpers.rys_3c2e_tri_schwarz_sparse_algo10(basis, auxbasis, indicesA, indicesB, offsets_3c2e, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold_schwarz, strict_schwarz, nsignificant)
            
            if strict_schwarz:
                start_strict_schwarz_nuc_mat = timer()
                if not use_gpu:
                    V = Integrals.nuc_mat_symm(basis, mol, None, sqrt_ints4c2e_diag)
                else:
                    V = Integrals.nuc_mat_symm_cupy(basis, mol, None, None, sqrt_ints4c2e_diag) 
                    # V = cp.asarray(Integrals.nuc_mat_symm(basis, mol, None, sqrt_ints4c2e_diag))
                H = T + V
                duration_strict_schwarz_nuc_mat = timer() - start_strict_schwarz_nuc_mat
                print('Time taken to evaluate the nuclear potential matrix with strict Schwarz screening: ', round(duration_strict_schwarz_nuc_mat, 2), flush=True)
            
        else:
            ints3c2e = Integrals.rys_3c2e_symm(basis, auxbasis)
            
        
    else:
        start2c2e = timer()
        ints2c2e = Integrals.conv_2c2e_symm(auxbasis) 
        # if sao:
            # print('here\n\n\n\n\n\n')
            # c2sph_mat = auxbasis.cart2sph_basis() # CAO --> SAO
            # sph2c_mat_pseudo = auxbasis.sph2cart_basis() # SAO --> CAO
            # ints2c2e = np.dot(c2sph_mat, np.dot(ints2c2e, c2sph_mat.T)) # CAO --> SAO
            # ints2c2e = np.dot(sph2c_mat_pseudo, np.dot(ints2c2e, sph2c_mat_pseudo.T)) #SAO --> CAO
        duration2c2e = timer() - start2c2e
        print('Time taken for two-centered two-electron integrals '+str(round(duration2c2e, 2))+' seconds.\n', flush=True)
        ints3c2e = Integrals.conv_3c2e_symm(basis, auxbasis)
        


    #### Dask array stuff (Doesn't work)
    # ints3c2e_dask = da.from_array(ints3c2e, chunks=(1000, 1000, 1500)) # Around 11 Gigs
    # ints3c2e_dask = da.from_array(ints3c2e, chunks=(400, 400, 400))
    # ints2c2e_dask = da.from_array(ints2c2e, chunks=(400, 400))
    # # ints3c2e = 0
    # temp_dask = da.from_array(ints3c2e.reshape(basis.bfs_nao*basis.bfs_nao, auxbasis.bfs_nao).T, chunks=(400, 400))
    # df_coeff0 = da.linalg.solve(ints2c2e_dask, temp_dask)


    # Compute the intermediate DF coefficients (df_coeff0) 
    if DF_algo==1:
        #The following solve step is very sloww and makes the Coulomb time much longer. 
        df_coeff0 = scipy.linalg.solve(ints2c2e, ints3c2e.reshape(basis.bfs_nao*basis.bfs_nao, auxbasis.bfs_nao).T)
        df_coeff0 = df_coeff0.reshape(auxbasis.bfs_nao, basis.bfs_nao, basis.bfs_nao)
        print('Three Center Two electron ERI size in GB ',ints3c2e.nbytes/1e9, flush=True)
        print('Two Center Two electron ERI size in GB ',ints2c2e.nbytes/1e9, flush=True)
        print('Intermediate Auxiliary Density fitting coefficients size in GB ',df_coeff0.nbytes/1e9, flush=True)

    ## Alternative based on Psi4numpy tutorial
    # https://github.com/psi4/psi4numpy/blob/master/Tutorials/03_Hartree-Fock/density-fitting.ipynb
    # metric_inverse_sqrt = scipy.linalg.inv(scipy.linalg.sqrtm(ints2c2e))
    # metric_inverse_sqrt = scipy.linalg.sqrtm(scipy.linalg.inv(ints2c2e))
    if DF_algo==2:
        ##### This version requires double the memory of a 3c2e array, as the result of the scipy solve (Qpq) is also of the same size.
        ##### We later don't require the ints3c2e array and get rid of it to free memory but still we did require it at some point so 
        ##### the memory requirement of the program as a whole is still 2x int3c2e array. 
        # # metric_sqrt = scipy.linalg.fractional_matrix_power(ints2c2e, 0.5)
        # metric_sqrt = scipy.linalg.sqrtm(ints2c2e)
        # print('Sqrt done!')
        # #TODO: The following solve step is very sloww and makes the Coulomb time much longer. Try to make it faster.
        # Qpq = scipy.linalg.solve(metric_sqrt, ints3c2e.reshape(basis.bfs_nao*basis.bfs_nao, auxbasis.bfs_nao).T, \
        #                         assume_a='pos', overwrite_a=False, overwrite_b=True)
        # print('Solve done!')
        # Qpq =  Qpq.reshape(auxbasis.bfs_nao, basis.bfs_nao, basis.bfs_nao)
        # print('Reshape done!')
        # # metric_inverse_sqrt = scipy.linalg.fractional_matrix_power(ints2c2e, -0.5) #Unstable
        # # print('Sqrt inverse done!')
        # # # Build the Qpq object
        # # Qpq = contract('QP,pqP->Qpq', metric_inverse_sqrt, ints3c2e)
        # # print('Contraction done!')
        # ints3c2e = 0
        # print('Two Center Two electron ERI size in GB ',ints2c2e.nbytes/1e9, flush=True)
        # print('Intermediate Auxiliary Density fitting coefficients size in GB ',Qpq.nbytes/1e9, flush=True)

        ##### New version, that as far as I can understand doesn't involve any creation of new arrays of the size of int3c2e array
        ##### This is done by not storing the result of scipy solve in an array but rather overwriting the original array.
        ##### A lot of reshaping is involved, but it is checked that it does not result in copies of arrays by using 
        ##### np.shares_memory(a,b) https://stackoverflow.com/questions/69447431/numpy-reshape-copying-data-or-not
        metric_sqrt = scipy.linalg.sqrtm(ints2c2e)
        print('Sqrt done!')
        ints3c2e_reshape = ints3c2e.reshape(basis.bfs_nao*basis.bfs_nao, auxbasis.bfs_nao).T
        print('Reshape done!')
        print(np.shares_memory(ints3c2e_reshape, ints3c2e))
        #TODO: The following solve step is very sloww and makes the Coulomb time much longer. Try to make it faster.
        scipy.linalg.solve(metric_sqrt, ints3c2e_reshape, \
                                assume_a='pos', overwrite_a=False, overwrite_b=True)
        print('Solve done!')
        Qpq =  ints3c2e_reshape.reshape(auxbasis.bfs_nao, basis.bfs_nao, basis.bfs_nao)
        print(np.shares_memory(Qpq, ints3c2e_reshape))
        print(np.shares_memory(Qpq, ints3c2e))
        print('Reshape done!')
        print('Two Center Two electron ERI size in GB ',ints2c2e.nbytes/1e9, flush=True)
        print('Intermediate Auxiliary Density fitting coefficients size in GB ',Qpq.nbytes/1e9, flush=True)

        ##### Sparse version of above (The solve is very slow and the reshape after solve on sparse matrix doesn't work)
        # metric_sqrt = scipy.linalg.sqrtm(ints2c2e)
        # print('Sqrt done!')
        # ints3c2e_reshape = ints3c2e.reshape(basis.bfs_nao*basis.bfs_nao, auxbasis.bfs_nao).T
        # print('Reshape done!')
        # ints3c2e_reshape[np.abs(ints3c2e_reshape) < 1E-09] = 0# fill most of the array with zeros (1E-09 is optimal i guess)
        # ints3c2e_reshape_sparse = csc_matrix(ints3c2e_reshape)
        # metric_sqrt[np.abs(metric_sqrt) < 1E-09] = 0# fill most of the array with zeros (1E-09 is optimal i guess)
        # metric_sqrt_sparse = csc_matrix(metric_sqrt)
        # print('Sparsification done!')
        # print('ints3c2e Sparse size in GB ',(ints3c2e_reshape_sparse.data.nbytes + ints3c2e_reshape_sparse.indptr.nbytes + ints3c2e_reshape_sparse.indices.nbytes)/1e9, flush=True)
        # # Sparse solve
        # Qpq_sparse = scipy.sparse.linalg.spsolve(metric_sqrt_sparse, ints3c2e_reshape_sparse)
        # print('Sparse Solve done!')
        # Qpq_sparse =  Qpq_sparse.reshape(auxbasis.bfs_nao, basis.bfs_nao, basis.bfs_nao)
        # print('Reshape done!')
        # print('Qpq Sparse size in GB ',(Qpq_sparse.data.nbytes + Qpq_sparse.indptr.nbytes + Qpq_sparse.indices.nbytes)/1e9, flush=True)
        # Qpq = sparse.COO.from_scipy_sparse(Qpq_sparse)

    if DF_algo==3: # Best algorithm (Memory efficient and fast without any prefactor linalg.solve)
        #https://aip.scitation.org/doi/pdf/10.1063/1.1567253
        print('Two Center Two electron ERI size in GB ',ints2c2e.nbytes/1e9, flush=True)
        print('Three Center Two electron ERI size in GB ',ints3c2e.nbytes/1e9, flush=True)
    if DF_algo==4 or DF_algo==5 or DF_algo==6 or DF_algo==8 or DF_algo==10:
        indices_dmat_tri = np.tril_indices_from(dmat) # Lower triangular including diagonal
        indices_dmat_tri_2 = np.tril_indices_from(dmat, k=-1) # lower tri, without the diagonal
        print('Two Center Two electron ERI size in GB ',ints2c2e.nbytes/1e9, flush=True)
        print('Three Center Two electron ERI size in GB ',ints3c2e.nbytes/1e9, flush=True)




    print('Three-centered two electron evaluation done!', flush=True)

    ### TESTING SOME SPARSE STUFF
    # Unfortunately, the current settings only provide small memory savings and the error is also quite large (0.00001 Ha)
    # Anyway this is not very useful as we are making a sparse array from an already calculated array which may not fit into memory
    # So I don't see much use for this right now.
    # df_coeff0[np.abs(df_coeff0) < 1E-09] = 0# fill most of the array with zeros (1E-09 is optimal i guess)
    # df_coeff0_sp = sparse.COO(df_coeff0)  # convert to sparse array
    # ints3c2e[np.abs(ints3c2e) < 1E-07] = 0# fill most of the array with zeros (1E-07 is optimal i guess) 
    # ints3c2e_sp = sparse.COO(ints3c2e)  # convert to sparse array
    # print('Sparse Three Center Two electron ERI size in GB ',ints3c2e_sp.nbytes/1e9, flush=True)
    # print('Intermediate Sparse Auxiliary Density fitting coefficients size in GB ',df_coeff0_sp.nbytes/1e9, flush=True)

    if cholesky:
        startDF_cholesky = timer()
        if use_gpu:
            cho_decomp_ints2c2e = scipy.linalg.cho_factor(cp.asnumpy(ints2c2e))
        else:
            cho_decomp_ints2c2e = scipy.linalg.cho_factor(ints2c2e)
        durationDF_cholesky = timer() - startDF_cholesky
        print('Time taken for Cholesky factorization of two-centered two-electron integrals '+str(round(durationDF_cholesky, 2))+' seconds.\n', flush=True)


    durationCoulomb = timer() - startCoulomb
    print('Time taken for Coulomb term related calculations (integrals, screening, prelims..) with the density fitting approximation  '+str(round(durationCoulomb, 2))+' seconds.\n', flush=True)
    result = [
        H,
        V,
        ints3c2e,
        ints2c2e,
        nsignificant,
        indicesA,
        indicesB,
        indicesC,
        offsets_3c2e,
        indices,
        ints4c2e_diag,
        sqrt_ints4c2e_diag,
        sqrt_diag_ints2c2e,
        indices_dmat_tri,
        indices_dmat_tri_2,
        df_coeff0,
        Qpq,
        cho_decomp_ints2c2e,
        durationDF_cholesky,
        durationCoulomb
    ]

    return result

def Jmat_from_density_fitting(dmat, DF_algo, cholesky, cho_decomp_ints2c2e, df_coeff0, Qpq, ints3c2e, ints2c2e, indices_dmat_tri, indices_dmat_tri_2, indicesA, indicesB, indicesC, offsets_3c2e, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold, strict_schwarz, basis, auxbasis, use_gpu, keep_ints3c2e_in_gpu, durationDF_gamma, ncores, durationDF_coeff, durationDF_Jtri, durationDF):
    Ecoul_temp = 0
    startDF = timer()
    # Using DF calculate the density fitting coefficients of the auxiliary basis
    if DF_algo==1:
        df_coeff = contract('ijk,jk', df_coeff0, dmat)
        J = contract('ijk,k',ints3c2e,df_coeff)
    if DF_algo==2:
        df_coeff = contract('ijk,jk', Qpq, dmat) # Not exactly the same thing as above (not the auxiliary density coeffs)
        J = contract('ijk,i',Qpq,df_coeff)
    if DF_algo==3: # Fastest      
        df_coeff = contract('pqP,pq->P', ints3c2e, dmat) # This is actually the gamma_alpha (and not df_coeff (c_alpha)) in this paper (https://aip.scitation.org/doi/pdf/10.1063/1.1567253)
        scipy.linalg.solve(ints2c2e, df_coeff, assume_a='pos', overwrite_a=False, overwrite_b=True) # This gives the actual df coeff
        J = contract('ijk,k', ints3c2e, df_coeff)
    if DF_algo==4 or DF_algo==5: # Fastest and triangular (half the memory of algo 3)
        dmat_temp = dmat.copy()
        dmat_temp[indices_dmat_tri_2] = 2*dmat[indices_dmat_tri_2] # Double the non-diagonal elements of the triangular density matrix
        dmat_tri = dmat_temp[indices_dmat_tri]
        df_coeff = contract('pP,p->P', ints3c2e, dmat_tri) # This is actually the gamma_alpha (and not df_coeff (c_alpha)) in this paper (https://aip.scitation.org/doi/pdf/10.1063/1.1567253)
        scipy.linalg.solve(ints2c2e, df_coeff, assume_a='pos', overwrite_a=False, overwrite_b=True)
        J_tri = contract('pP,P', ints3c2e, df_coeff)
        # https://stackoverflow.com/questions/17527693/transform-the-upper-lower-triangular-part-of-a-symmetric-matrix-2d-array-into
        J = np.zeros((basis.bfs_nao, basis.bfs_nao))
        J[indices_dmat_tri] = J_tri
        J += J.T - np.diag(np.diag(J))
    if DF_algo==6: 
        dmat_temp = dmat.copy()
        dmat_temp[indices_dmat_tri_2] = 2*dmat[indices_dmat_tri_2] # Double the non-diagonal elements of the triangular density matrix
        dmat_tri = dmat_temp[indices_dmat_tri]
        startDF_gamma = timer()
        # df_coeff_1 = contract('pP,p->P', ints3c2e, dmat_tri) # This is actually the gamma_alpha (and not df_coeff (c_alpha)) in this paper (https://aip.scitation.org/doi/pdf/10.1063/1.1567253)
        gamma_alpha = Integrals.schwarz_helpers.df_coeff_calculator(ints3c2e, dmat_tri, indicesA, indicesB, indicesC, auxbasis.bfs_nao, ncores) # This is actually the gamma_alpha (and not df_coeff (c_alpha)) in this paper (https://aip.scitation.org/doi/pdf/10.1063/1.1567253)
        durationDF_gamma += timer() - startDF_gamma
        startDF_coeff = timer()
        with threadpool_limits(limits=ncores, user_api='blas'):
            # print('Density fitting', controller.info())
            if not cholesky:
                df_coeff = scipy.linalg.solve(ints2c2e, gamma_alpha, assume_a='pos', overwrite_a=False, overwrite_b=False)
            else:
                df_coeff = scipy.linalg.cho_solve(cho_decomp_ints2c2e, gamma_alpha, overwrite_b=False, check_finite=True)
        durationDF_coeff += timer() - startDF_coeff
        with threadpool_limits(limits=ncores, user_api='blas'):
            Ecoul_temp = np.dot(df_coeff, gamma_alpha) # (rho^~|rho^~) Coulomb energy due to interactions b/w auxiliary density
        startDF_Jtri = timer()
        #J_tri = contract('pP,P', ints3c2e, df_coeff)
        J_tri = Integrals.schwarz_helpers.J_tri_calculator(ints3c2e, df_coeff, indicesA, indicesB, indicesC, int(basis.bfs_nao*(basis.bfs_nao+1)/2))
        # https://stackoverflow.com/questions/17527693/transform-the-upper-lower-triangular-part-of-a-symmetric-matrix-2d-array-into
        with threadpool_limits(limits=ncores, user_api='blas'):
            J = np.zeros((basis.bfs_nao, basis.bfs_nao))
            J[indices_dmat_tri] = J_tri
            J += J.T - np.diag(np.diag(J))
        durationDF_Jtri += timer() - startDF_Jtri
    if DF_algo==8: 
        dmat_temp = dmat.copy()
        dmat_temp[indices_dmat_tri_2] = 2*dmat[indices_dmat_tri_2] # Double the non-diagonal elements of the triangular density matrix
        dmat_tri = dmat_temp[indices_dmat_tri]
        startDF_gamma = timer()
        # df_coeff_1 = contract('pP,p->P', ints3c2e, dmat_tri) # This is actually the gamma_alpha (and not df_coeff (c_alpha)) in this paper (https://aip.scitation.org/doi/pdf/10.1063/1.1567253)
        gamma_alpha = Integrals.schwarz_helpers.df_coeff_calculator_algo8(ints3c2e, dmat_tri, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold_schwarz, basis.bfs_nao, auxbasis.bfs_nao) # This is actually the gamma_alpha (and not df_coeff (c_alpha)) in this paper (https://aip.scitation.org/doi/pdf/10.1063/1.1567253)
        durationDF_gamma += timer() - startDF_gamma
        startDF_coeff = timer()
        with threadpool_limits(limits=ncores, user_api='blas'):
            # print('Density fitting', controller.info())
            df_coeff = scipy.linalg.solve(ints2c2e, gamma_alpha, assume_a='pos', overwrite_a=False, overwrite_b=False)
        durationDF_coeff += timer() - startDF_coeff
        with threadpool_limits(limits=ncores, user_api='blas'):
            Ecoul_temp = np.dot(df_coeff, gamma_alpha) # (rho^~|rho^~) Coulomb energy due to interactions b/w auxiliary density
        startDF_Jtri = timer()
        #J_tri = contract('pP,P', ints3c2e, df_coeff)
        J_tri = Integrals.schwarz_helpers.J_tri_calculator_algo8(ints3c2e, df_coeff, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold_schwarz, basis.bfs_nao, auxbasis.bfs_nao, int(basis.bfs_nao*(basis.bfs_nao+1)/2))
        # https://stackoverflow.com/questions/17527693/transform-the-upper-lower-triangular-part-of-a-symmetric-matrix-2d-array-into
        with threadpool_limits(limits=ncores, user_api='blas'):
            J = np.zeros((basis.bfs_nao, basis.bfs_nao))
            J[indices_dmat_tri] = J_tri
            J += J.T - np.diag(np.diag(J))
        durationDF_Jtri += timer() - startDF_Jtri
    if DF_algo==10: 
        dmat_temp = dmat.copy()
        dmat_temp[indices_dmat_tri_2] = 2*dmat[indices_dmat_tri_2] # Double the non-diagonal elements of the triangular density matrix
        dmat_tri = dmat_temp[indices_dmat_tri]
        startDF_gamma = timer()
        auxbfs_lm = np.array(auxbasis.bfs_lm)
        if use_gpu:
            ints3c2e_cp = cp.asarray(ints3c2e)
            dmat_tri_cp = cp.array(dmat_tri)
            offsets_3c2e_cp = cp.array(offsets_3c2e)
            sqrt_ints4c2e_diag_cp = cp.array(sqrt_ints4c2e_diag)
            sqrt_diag_ints2c2e_cp = cp.array(sqrt_diag_ints2c2e)
            gamma_alpha = Integrals.schwarz_helpers_cupy.df_coeff_calculator_algo10_cupy(ints3c2e, dmat_tri_cp, basis.bfs_nao, offsets_3c2e_cp, auxbasis.bfs_nao, sqrt_ints4c2e_diag_cp, sqrt_diag_ints2c2e_cp, threshold, strict_schwarz) # This is actually the gamma_alpha (and not df_coeff (c_alpha)) in this paper (https://aip.scitation.org/doi/pdf/10.1063/1.1567253)
            # gamma_alpha = cp.asnumpy(gamma_alpha)
        else:
            # df_coeff_1 = contract('pP,p->P', ints3c2e, dmat_tri) # This is actually the gamma_alpha (and not df_coeff (c_alpha)) in this paper (https://aip.scitation.org/doi/pdf/10.1063/1.1567253)
            # gamma_alpha = Integrals.schwarz_helpers.df_coeff_calculator_algo10_serial(ints3c2e, dmat_tri, indicesA, indicesB, offsets_3c2e, auxbasis.bfs_nao, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold) # This is actually the gamma_alpha (and not df_coeff (c_alpha)) in this paper (https://aip.scitation.org/doi/pdf/10.1063/1.1567253)
            gamma_alpha = Integrals.schwarz_helpers.df_coeff_calculator_algo10_parallel(ints3c2e, dmat_tri, indicesA, indicesB, offsets_3c2e, auxbasis.bfs_nao, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold, ncores, strict_schwarz, auxbfs_lm) # This is actually the gamma_alpha (and not df_coeff (c_alpha)) in this paper (https://aip.scitation.org/doi/pdf/10.1063/1.1567253)
        # gamma_alpha = Integrals.schwarz_helpers.df_coeff_calculator_algo10_parallel(ints3c2e, dmat_tri, indicesA, indicesB, offsets_3c2e, auxbasis.bfs_nao, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold, ncores, strict_schwarz, auxbfs_lm) # This is actually the gamma_alpha (and not df_coeff (c_alpha)) in this paper (https://aip.scitation.org/doi/pdf/10.1063/1.1567253)
        durationDF_gamma += timer() - startDF_gamma
        startDF_coeff = timer()
        with threadpool_limits(limits=ncores, user_api='blas'):
            # print('Density fitting', controller.info())
            if not cholesky:
                if not use_gpu:
                    df_coeff = scipy.linalg.solve(ints2c2e, gamma_alpha, assume_a='pos', overwrite_a=False, overwrite_b=False)
                else:
                    df_coeff = cp.linalg.solve(ints2c2e, gamma_alpha)
                    # print(df_coeff[0:10])
            else:
                df_coeff = scipy.linalg.cho_solve(cho_decomp_ints2c2e, gamma_alpha, overwrite_b=False, check_finite=True)
        durationDF_coeff += timer() - startDF_coeff
        if use_gpu:
            Ecoul_temp = cp.dot(df_coeff, gamma_alpha)
        else:
            with threadpool_limits(limits=ncores, user_api='blas'):
                Ecoul_temp = np.dot(df_coeff, gamma_alpha) # (rho^~|rho^~) Coulomb energy due to interactions b/w auxiliary density
        startDF_Jtri = timer()
        #J_tri = contract('pP,P', ints3c2e, df_coeff)
        if use_gpu:
            df_coeff_cp = cp.asarray(df_coeff)
            J_tri = Integrals.schwarz_helpers_cupy.J_tri_calculator_algo10_cupy(ints3c2e_cp, df_coeff_cp, int(basis.bfs_nao*(basis.bfs_nao+1)/2), basis.bfs_nao, offsets_3c2e_cp, sqrt_ints4c2e_diag_cp, sqrt_diag_ints2c2e_cp, threshold, auxbasis.bfs_nao, strict_schwarz, auxbfs_lm)
            # J_tri = Integrals.schwarz_helpers_cupy.J_tri_calculator_algo10_cupy(ints3c2e_cp, df_coeff, int(basis.bfs_nao*(basis.bfs_nao+1)/2), basis.bfs_nao, offsets_3c2e, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold, auxbasis.bfs_nao, strict_schwarz, auxbfs_lm)
            J = cp.zeros((basis.bfs_nao, basis.bfs_nao))
            J[indices_dmat_tri] = J_tri
            J += J.T - cp.diag(cp.diag(J))
        else:
            J_tri = Integrals.schwarz_helpers.J_tri_calculator_algo10(ints3c2e, df_coeff, indicesA, indicesB, offsets_3c2e, int(basis.bfs_nao*(basis.bfs_nao+1)/2), sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold, auxbasis.bfs_nao, strict_schwarz, auxbfs_lm)
            # https://stackoverflow.com/questions/17527693/transform-the-upper-lower-triangular-part-of-a-symmetric-matrix-2d-array-into
            with threadpool_limits(limits=ncores, user_api='blas'):
                J = np.zeros((basis.bfs_nao, basis.bfs_nao))
                J[indices_dmat_tri] = J_tri
                J += J.T - np.diag(np.diag(J))
        durationDF_Jtri += timer() - startDF_Jtri
    durationDF = durationDF + timer() - startDF

    # Free memory
    if use_gpu:
        if not keep_ints3c2e_in_gpu:
            ints3c2e_cp = None
        dmat_tri_cp = None
        offsets_3c2e_cp = None
        sqrt_ints4c2e_diag_cp = None
        sqrt_diag_ints2c2e_cp = None
        df_coeff_cp = None
        cp.cuda.Device(0).use()
        cp._default_memory_pool.free_all_blocks()
    return J, durationDF, durationDF_coeff, durationDF_gamma, durationDF_Jtri, Ecoul_temp
