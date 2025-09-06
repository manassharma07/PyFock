# UHF_atoms.py
# Author: Manas Sharma (manassharma07@live.com)
# This is a part of CrysX (https://bragitoff.com/crysx)
#
#
#  .d8888b.                            Y88b   d88P       8888888b.           8888888888                888      
# d88P  Y88b                            Y88b d88P        888   Y88b          888                       888      
# 888    888                             Y88o88P         888    888          888                       888      
# 888        888d888 888  888 .d8888b     Y888P          888   d88P 888  888 8888888  .d88b.   .d8888b 888  888 
# 888        888P"   888  888 88K         d888b          8888888P"  888  888 888     d88""88b d88P"    888 .88P 
# 888    888 888     888  888 "Y8888b.   d88888b  888888 888        888  888 888     888  888 888      888888K  
# Y88b  d88P 888     Y88b 888      X88  d88P Y88b        888        Y88b 888 888     Y88..88P Y88b.    888 "88b 
#  "Y8888P"  888      "Y88888  88888P' d88P   Y88b       888         "Y88888 888      "Y88P"   "Y8888P 888  888 
#                         888                                            888                                    
#                    Y8b d88P                                       Y8b d88P                                    
#                     "Y88P"                                         "Y88P"                                       
from re import T
import pyfock.Mol as Mol
import pyfock.Basis as Basis
# import pyfock.Integrals as Integrals
import pyfock.Integrals as Integrals
import pyfock.Grids as Grids
from threadpoolctl import ThreadpoolController, threadpool_info, threadpool_limits
controller = ThreadpoolController()
import numpy as np
from numpy.linalg import eig, multi_dot as dot
import scipy 
    
from timeit import default_timer as timer
import numba
from opt_einsum import contract
import pylibxc
# import sparse
# import dask.array as da
from scipy.sparse import csr_matrix, csc_matrix
# from memory_profiler import profile
import os
from numba import njit, prange, cuda
import numexpr
try:
    import cupy as cp
    from cupy import fuse
    import cupyx
    CUPY_AVAILABLE = True
except Exception as e:
    print('Cupy is not installed. GPU acceleration is not availble.')
    CUPY_AVAILABLE = False
    pass


@njit(parallel=True, cache=True, fastmath=True, error_model="numpy")
def compute_B(errVecs):
    nKS = errVecs.shape[0]
    B = np.zeros((nKS + 1, nKS + 1))
    B[-1, :] = B[:, -1] = -1.0
    B[-1, -1] = 0.0
    for i in prange(nKS):
        # errVec_i_conj_T = errVecs[i].conj().T
        errVec_i_conj_T = errVecs[i].T
        for j in range(i + 1):
            # B[i, j] = B[j, i] = np.real(np.trace(np.dot(errVec_i_conj_T, errVecs[j])))
            B[i, j] = B[j, i] = np.trace(np.dot(errVec_i_conj_T, errVecs[j]))
    return B

class DFT:
    def __init__(self, mol, basis, dmat_guess_method=None, xc=None, gridsLevel=3): 
        if mol is None:
            print('ERROR: A Mol object is required to initialize a DFT object.')
        else: 
            self.mol = mol
        if basis is None:
            print('ERROR: A Basis object is required to initialize a DFT object.')
        else: 
            self.basis = basis
        if dmat_guess_method is None:
            self.dmat_guess_method = 'core'
        else: 
            self.dmat_guess_method = dmat_guess_method
        if xc is None: 
            self.xc = [1, 7] #LDA
        else:
            self.xc = xc
        # DIIS
        self.KSmats = []
        self.errVecs = []
        self.diisSpace = 6

        # Grids
        self.gridsLevel = gridsLevel

        # GPU acceleration
        self.use_gpu = False
        self.keep_ao_in_gpu = True
        self.use_libxc = True
        self.n_streams = 1
        self.n_gpus = 1
        self.free_gpu_mem = False
        try:
            self.max_threads_per_block = cuda.get_current_device().MAX_THREADS_PER_BLOCK
        except:
            self.max_threads_per_block = 1024
        
        self.threads_x = int(self.max_threads_per_block/16)
        self.threads_y = int(self.max_threads_per_block/64)
        
        # self.max_threads_per_block = 1024
        # self.threads_x = 64
        # self.threads_y = 16

        # if self.use_gpu:
        #     try:
        #         global cp
        #         global cupy_scipy
        #         import cupy as cp
        #         import cupyx.scipy as cupy_scipy
        #         # from cupy.linalg import eig, multi_dot as dot
        #     except ModuleNotFoundError:
        #         print('Cupy was not found!')

    def removeLinearDep(self, H, S):
        return 1 
    
    def nuclear_rep_energy(self, mol=None):
        # Nuclear-nuclear energy
        if mol is None: 
            mol = self.mol
    
        e = 0
        for i in range(mol.natoms):
            for j in range(i, mol.natoms):
                if (i!=j):
                    dist = mol.coordsBohrs[i]-mol.coordsBohrs[j]
                    e = e + mol.Zcharges[i]*mol.Zcharges[j]/np.sqrt(np.sum(dist**2))

        return e
        
    # full density matrix for RHF
    def gen_dm(self, mo_coeff, mo_occ):
   
        mocc = mo_coeff[:,mo_occ>0]
   
        return np.dot(mocc*mo_occ[mo_occ>0], mocc.conj().T)
    
    def getOcc(self, mol=None, energy_mo=None, coeff_mo=None):
        e_idx = np.argsort(energy_mo)
        e_sort = energy_mo[e_idx]
        nmo = energy_mo.size
        occ_mo = np.zeros(nmo)
        nocc = mol.nelectrons // 2
        occ_mo[e_idx[:nocc]] = 2
        return occ_mo
    
    def gen_dm_cupy(self, mo_coeff, mo_occ):
        mocc = mo_coeff[:,mo_occ>0]
        return cp.dot(mocc*mo_occ[mo_occ>0], mocc.conj().T)
    
    def getOcc_cupy(self, mol=None, energy_mo=None, coeff_mo=None):
        e_idx = cp.argsort(energy_mo)
        e_sort = energy_mo[e_idx]
        nmo = energy_mo.size
        occ_mo = cp.zeros(nmo)
        nocc = mol.nelectrons // 2
        occ_mo[e_idx[:nocc]] = 2
        return occ_mo
    
    def solve(self, H, S, orthogonalize=False):
        if not orthogonalize:
            #Solve the generalized eigenvalue equation HC = SCE
            eigvalues, eigvectors = scipy.linalg.eigh(H, S)
            
        else:
            eig_val_s, eig_vec_s = scipy.linalg.eigh(S)
            # Removing the eigenvectors assoicated to the smallest eigenvalue.
            x = eig_vec_s[:,eig_val_s>1e-7] / np.sqrt(eig_val_s[eig_val_s>1e-7])
            xhx = x.T @ H @ x
            #Solve the canonical eigenvalue equation HC = SCE
            eigvalues, eigvectors = scipy.linalg.eigh(xhx)
            eigvectors = np.dot(x, eigvectors)

        idx = np.argmax(np.abs(eigvectors.real), axis=0)
        eigvectors[:,eigvectors[idx,np.arange(len(eigvalues))].real<0] *= -1
        return eigvalues, eigvectors # E, C
    
    def solve_cupy(self, H, S, orthogonalize=True):
        eig_val_s, eig_vec_s = cp.linalg.eigh(S)
        # Removing the eigenvectors assoicated to the smallest eigenvalue.
        x = eig_vec_s[:,eig_val_s>1e-7] / cp.sqrt(eig_val_s[eig_val_s>1e-7])
        xhx = x.T @ H @ x
        #Solve the canonical eigenvalue equation HC = SCE
        eigvalues, eigvectors = cp.linalg.eigh(xhx)
        eigvectors = cp.dot(x, eigvectors)

        idx = cp.argmax(cp.abs(eigvectors.real), axis=0)
        eigvectors[:,eigvectors[idx,cp.arange(len(eigvalues))].real<0] *= -1
        return eigvalues, eigvectors # E, C
    

    def getCoreH(self, mol=None, basis=None):
        #Get the core Hamiltonian
        if mol is None:
            mol = self.mol
        if basis is None:
            basis = self.basis
        nao = basis.bfs_nao 
        H = np.empty((nao,nao))
        Vmat = Integrals.nucMatSymmNumbawrap(basis, mol)
        Tmat = Integrals.kinMatSymmNumbawrap(basis)
        H = Vmat + Tmat

        return H


    def guessCoreH(self, mol=None, basis=None, Hcore=None, S=None):
        #Get a guess for the density matrix using the core Hamiltonian
        if mol is None:
            mol = self.mol
        if basis is None:
            basis = self.basis
        if Hcore is None:
            Hcore = self.getCoreH(mol, basis)
        if S is None:
            S = Integrals.overlapMatSymmNumbawrap(basis)

        eigvalues, eigvectors = scipy.linalg.eigh(Hcore, S)
        # print(eigvalues)
        idx = np.argmax(abs(eigvectors.real), axis=0)
        eigvectors[:,eigvectors[idx,np.arange(len(eigvalues))].real<0] *= -1
        mo_occ = self.getOcc(mol, eigvalues, eigvectors)
        # print(mo_occ)
        return self.gen_dm(eigvectors, mo_occ)
    
    

    # def DIIS(self, S, D, F):
    #     FDS = np.dot(np.dot(F, D), S)
    #     errVec = FDS - FDS.T.conj()
    #     self.KSmats.append(F)
    #     self.errVecs.append(errVec)
    #     nKS = len(self.KSmats)
    #     if nKS > self.diisSpace:
    #         self.KSmats.pop(0)
    #         self.errVecs.pop(0)
    #         nKS = nKS - 1
    #     B = compute_B(self.errVecs)
    #     residual = np.zeros(nKS + 1)
    #     residual[-1] = -1.0
    #     weights = scipy.linalg.solve(B, residual)
    #     assert np.isclose(np.sum(weights[:-1]), 1.0)
    #     F = np.zeros(F.shape)
    #     for i, KS in enumerate(self.KSmats):
    #         F += weights[i] * KS
    #     return F

    def DIIS(self,S,D,F):
        FDS =   dot([F,D,S])
        # SDF =   np.conjugate(FDS).T 
        errVec = FDS - np.conjugate(FDS).T 
        self.KSmats.append(F)
        self.errVecs.append(errVec) 
        nKS = len(self.KSmats)
        if nKS > self.diisSpace:
            self.KSmats.pop(0) 
            self.errVecs.pop(0)
            nKS = nKS - 1
        B = np.zeros((nKS + 1,nKS + 1)) 
        B[-1,:] = B[:,-1] = -1.0
        B[-1,-1] = 0.0
        # B is symmetric
        for i in range(nKS):
            for j in range(i+1):
                B[i,j] = B[j,i] = \
                    np.real(np.trace(np.dot(np.conjugate(self.errVecs[i]).T, self.errVecs[j])))
        # for i in range(nKS):
        #     for j in range(i+1):
        #         print(self.errVecs[i].shape)
        #         B[i,j] = np.real(np.dot(np.conjugate(self.errVecs[i]).T, self.errVecs[j]))
        #         B[j,i] = B[i,j]
                                                    
        residual = np.zeros((nKS + 1, 1))
        residual[-1] = -1.0
        weights = scipy.linalg.solve(B,residual)

        # weights is 1 x numFock + 1, but first numFock values
        # should sum to one if we are doing DIIS correctly
        assert np.isclose(sum(weights[:-1]),1.0)

        F = np.zeros(F.shape)
        for i, KS in enumerate(self.KSmats):
            weight = weights[i]
            F += numexpr.evaluate('(weight * KS)')

        return F 
    
    def DIIS_cupy(self,S,D,F):
        FDS =   F @ D @ S
        errVec = FDS - cp.conjugate(FDS).T 
        self.KSmats.append(F)
        self.errVecs.append(errVec) 
        nKS = len(self.KSmats)
        if nKS > self.diisSpace:
            self.KSmats.pop(0) 
            self.errVecs.pop(0)
            nKS = nKS - 1
        B = cp.zeros((nKS + 1,nKS + 1)) 
        B[-1,:] = B[:,-1] = -1.0
        B[-1,-1] = 0.0
        # B is symmetric
        for i in range(nKS):
            for j in range(i+1):
                B[i,j] = B[j,i] = \
                    cp.real(cp.trace(cp.dot(cp.conjugate(self.errVecs[i]).T, self.errVecs[j])))
                                                    
        residual = cp.zeros((nKS + 1, 1))
        residual[-1] = -1.0
        weights = cp.linalg.solve(B,residual)

        # weights is 1 x numFock + 1, but first numFock values
        # should sum to one if we are doing DIIS correctly
        assert cp.isclose(sum(weights[:-1]),1.0)

        F = cp.zeros(F.shape)
        for i, KS in enumerate(self.KSmats):
            weight = weights[i]
            F += weight * KS

        return F 

    # @profile
    def scf(self, mol=None, basis=None, dmat=None, xc=None, conv_crit=1.0E-7, max_itr=50, ncores=2, grids=None, gridsLevel=3, isDF=True, 
            auxbasis=None, rys=True, DF_algo=6, blocksize=None, XC_algo=None, debug=False, sortGrids=False, save_ao_values=False,xc_bf_screen=True,
            threshold_schwarz = 1e-09):
        #### Timings
        duration1e = 0
        durationDIIS = 0
        durationAO_values = 0
        durationCoulomb = 0
        durationgrids = 0
        durationItr = 0
        durationxc = 0
        durationXCpreprocessing = 0
        durationDF = 0
        durationKS = 0
        durationSCF = 0
        durationgrids_prune_rho = 0
        durationSchwarz = 0
        duration2c2e = 0
        durationDF_coeff = 0
        durationDF_gamma = 0
        durationDF_Jtri = 0
        startSCF = timer()      

        

        #### Set number of cores
        numba.set_num_threads(ncores)
        os.environ['RAYON_NUM_THREADS'] = str(ncores)
        
        print('Running DFT using '+str(numba.get_num_threads())+' threads for Numba.\n\n', flush=True)
        if basis is None:
            basis = self.basis
        if mol is None:
            mol = self.mol
        if xc is None:
            xc = self.xc
        if gridsLevel is None:
            gridsLevel = self.gridsLevel
        if auxbasis is None:
            auxbasis = Basis(mol, {'all':Basis.load(mol=mol, basis_name='def2-universal-jfit')})
        if grids is None:
            sortGrids = True
        if xc_bf_screen==False:
            if save_ao_values==True:
                print('Warning! AO screening is set to False, but AO values are requested to be saved. \
                      AO values can only be saved when XC_BF_SCREEN=TRUE. AO values will not be saved.', flush=True)
                save_ao_values = False
        if CUPY_AVAILABLE:
            if self.use_gpu:
                print('GPU acceleration is enabled. Currently this only accelerates AO values and XC term evaluation.', flush=True)
                print('GPU(s) information:')
                # print(cp.cuda.Device.mem_info())
                print(cuda.detect())
                print('Max threads per block supported by the GPU: ', cuda.get_current_device().MAX_THREADS_PER_BLOCK, flush=True)
                print('The user has specified to use '+str(self.n_gpus)+' GPU(s).')
            
                # For CUDA computations
                threads_per_block = (self.threads_x, self.threads_y)
                print('Threads per block configuration: ', threads_per_block, flush=True)

                print('\n\nWill use dynamic precision. ')
                print('This means that the XC term will be evaluated in single precision until the ')
                print('relative energy difference b/w successive iterations is less than 5.0E-7.')
                precision_XC = cp.float32

                if XC_algo is None:
                    XC_algo = 3
                if blocksize is None:
                    blocksize = 51200
        else:
            if self.use_gpu:
                print('GPU acceleration requested but cannot be enabled as Cupy is not installed.', flush=True)
                self.use_gpu = False

                if XC_algo is None:
                    XC_algo = 2
                if blocksize is None:
                    blocksize = 5000

        isSchwarz = True


        eigvectors = None
        # DF_algo = 1 # Worst algorithm for more than 500 bfs/auxbfs (requires 2x mem of 3c2e integrals and a large prefactor)
        # DF_algo = 2 # Sligthly better (2x memory efficient) algorithm than above (requires 1x mem of 3c2e integrals and a large prefactor)
        # DF_algo = 3 # Memory effcient without any prefactor. (Can easily be converted into a sparse version, unlike the others) (https://aip.scitation.org/doi/pdf/10.1063/1.1567253)
        # DF_algo = 4 # Same as 3, except now we use triangular version of ints3c2e to save on memory
        # DF_algo = 5 # Same as 4 in terms of memory requirements, however faster in performance due to the use of Schwarz screening.
        # DF_algo = 6 # Much cheaper than 4 and 5 in terms of memory requirements because the indices of significant (ij|P) are efficiently calculated without duplicates/temporary arrays. 
        #               The speed maybe same or just slightly slower. 
        # DF_algo = 7 # The significant indices (ij|P) are stored even more efficiently by using shell indices instead of bf indices.
        # DF_algo = 8 # Similar to 6, except that here the significant indices are not stored resulting in 50% memory savings
        # DF_algo = 9 # Same as 4, however, here we use pyscf (libcint) for integral evaluation

        start1e = timer()
        print('\nCalculating one electron integrals...\n\n', flush=True)
        # One electron integrals
        S = Integrals.overlap_mat_symm(basis)
        V = Integrals.nuc_mat_symm(basis, mol)
        T = Integrals.kin_mat_symm(basis)
        # Core hamiltonian
        H = V + T
        if self.use_gpu:
            S = cp.asarray(S, dtype=cp.float64)
            V = cp.asarray(V, dtype=cp.float64)
            T = cp.asarray(T, dtype=cp.float64)
            H = cp.asarray(H, dtype=cp.float64)
            cp.cuda.Stream.null.synchronize()

        print('Core H size in GB ',H.nbytes/1e9, flush=True)
        print('done!', flush=True)
        duration1e = timer() - start1e
        print('Time taken '+str(duration1e)+' seconds.\n', flush=True)

        if dmat is None:
            if self.dmat_guess_method=='core':
                dmat = self.guessCoreH(mol, basis, Hcore=H, S=S)

        if self.use_gpu:
            dmat_cp = cp.asarray(dmat, dtype=cp.float64)
            cp.cuda.Stream.null.synchronize()

        startCoulomb = timer()
        if not isDF: # 4c2e ERI case
            
            print('\nCalculating four centered two electron integrals (ERIs)...\n\n', flush=True)
            if not isSchwarz:
                # Four centered two electron integrals (ERIs)
                if rys:
                    ints4c2e = Integrals.rys_4c2e_symm(basis)
                else:
                    ints4c2e = Integrals.conv_4c2e_symm(basis)
                print('Four Center Two electron ERI size in GB ',ints4c2e.nbytes/1e9, flush=True)
                print('done!')
            else:
                print('\n\nPerforming Schwarz screening...')
                # threshold_schwarz = 1e-09
                print('Threshold ', threshold_schwarz)
                startSchwarz = timer()
                nints4c2e_sym8 = int(basis.bfs_nao**4/8)
                nints4c2e = int(basis.bfs_nao**4)
                duration_4c2e_diag = 0.0
                start_4c2e_diag = timer()
                # Diagonal elements of ERI 4c2e array
                ints4c2e_diag = Integrals.schwarz_helpers.eri_4c2e_diag(basis)
                duration_4c2e_diag = timer() - start_4c2e_diag
                print('Time taken to evaluate the "diagonal" of 4c2e ERI tensor: ', duration_4c2e_diag)
                # Calculate the square roots required for 
                duration_square_roots = 0.0
                start_square_roots = timer()
                sqrt_ints4c2e_diag = np.sqrt(np.abs(ints4c2e_diag))
                duration_square_roots = timer() - start_square_roots
                print('Time taken to evaluate the square roots needed: ', duration_square_roots)
                chunksize = int(1e9) # Results in 2 GB chunks
                duration_indices_calc = 0.0
                duration_concatenation = 0.0
                start_indices_calc = timer()
                # indices_temp = []
                ijkl = [0, 0, 0, 0]
                if chunksize<nints4c2e_sym8:
                    nchunks = nints4c2e_sym8//chunksize + 1
                else:
                    nchunks=1
                indicesA = None
                for ichunk in range(nchunks):
                    indices_temp = Integrals.schwarz_helpers.calc_indices_4c2e_schwarz(sqrt_ints4c2e_diag, min(chunksize, nints4c2e_sym8), basis.bfs_nao, ijkl[0], ijkl[1], ijkl[2], ijkl[3], threshold_schwarz)
                    
                    ijkl = indices_temp[4]
                    count = indices_temp[5]
                    # start_concatenation = timer()
                    if indicesA is not None and count>0:
                        indicesA = np.concatenate([indicesA, indices_temp[0][0:count]])
                        indicesB = np.concatenate([indicesB, indices_temp[1][0:count]])
                        indicesC = np.concatenate([indicesC, indices_temp[2][0:count]])
                        indicesD = np.concatenate([indicesD, indices_temp[3][0:count]])
                    else:
                        
                        indicesA = indices_temp[0][0:count]
                        indicesB = indices_temp[1][0:count]
                        indicesC = indices_temp[2][0:count]
                        indicesD = indices_temp[3][0:count]
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
                
                print('Size of permanent array storing the significant indices of 4c2e ERI in GB ', indicesA.nbytes/1e9+indicesB.nbytes/1e9+indicesC.nbytes/1e9+indicesD.nbytes/1e9, flush=True)

                nsignificant = len(indicesA)
                print('No. of elements in the standard four-centered two electron ERI tensor: ', nints4c2e, flush=True)
                print('No. of elements after factoring in the 8-fold symmetry in the four-centered two electron ERI tensor: ', nints4c2e_sym8, flush=True)
                print('No. of significant quadruplets based on Schwarz inequality and 8-fold symmetry: ' + str(nsignificant) + ' or '+str(np.round(nsignificant/nints4c2e*100,1)) + '% of original', flush=True)
                print('Schwarz screening done!')
                durationSchwarz = timer() - startSchwarz
                print('Total time taken for Schwarz screening '+str(durationSchwarz)+' seconds.\n', flush=True)

                ints4c2e = Integrals.schwarz_helpers.rys_4c2e_tri_schwarz_sparse(basis, auxbasis, indicesA, indicesB, indicesC, indicesD)

        else: # Density fitting case (3c2e, and 2c2e will be calculated)

            print('\nCalculating three centered two electron and two-centered two-electron integrals...\n\n', flush=True)
            if rys:
                start2c2e = timer()
                ints2c2e = Integrals.rys_2c2e_symm(auxbasis)
                duration2c2e = timer() - start2c2e
                print('Time taken for two-centered two-electron integrals '+str(duration2c2e)+' seconds.\n', flush=True)
                if DF_algo==4: #Triangular version
                    ints3c2e = Integrals.rys_3c2e_tri(basis, auxbasis)
                    print(ints3c2e.shape)
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
                    print('Time taken '+str(durationSchwarz)+' seconds.\n', flush=True)
                    
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
                    print('Time taken to evaluate the "diagonal" of 4c2e ERI tensor: ', duration_4c2e_diag)
                    
                    # Calculate the square roots required for 
                    duration_square_roots = 0.0
                    start_square_roots = timer()
                    sqrt_ints4c2e_diag = np.sqrt(np.abs(ints4c2e_diag))
                    sqrt_diag_ints2c2e = np.sqrt(np.abs(np.diag(ints2c2e)))
                    duration_square_roots = timer() - start_square_roots
                    print('Time taken to evaluate the square roots needed: ', duration_square_roots)

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
                        indices_temp = Integrals.schwarz_helpers.calc_indices_3c2e_schwarz2(sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, min(chunksize, nints3c2e_tri), basis.bfs_nao, auxbasis.bfs_nao, ijk[0], ijk[1], ijk[2], threshold_schwarz)
                        
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
                    
                    print('Size of permanent array storing the significant indices of 3c2e ERI in GB ', indicesA.nbytes/1e9+indicesB.nbytes/1e9+indicesC.nbytes/1e9, flush=True)

                    nsignificant = len(indicesA)
                    print('No. of elements in the standard three-centered two electron ERI tensor: ', nints3c2e, flush=True)
                    print('No. of elements in the triangular three-centered two electron ERI tensor: ', nints3c2e_tri, flush=True)
                    print('No. of significant triplets based on Schwarz inequality and triangularity: ' + str(nsignificant) + ' or '+str(np.round(nsignificant/nints3c2e*100,1)) + '% of original', flush=True)
                    print('Schwarz screening done!')
                    durationSchwarz = timer() - startSchwarz
                    print('Total time taken for Schwarz screening '+str(durationSchwarz)+' seconds.\n', flush=True)
                    
                    ints3c2e = Integrals.schwarz_helpers.rys_3c2e_tri_schwarz_sparse(basis, auxbasis, indicesA, indicesB, indicesC)
                    # The following uses joblib to parallelize instead
                    # ints3c2e = Integrals.schwarz_helpers.rys_3c2e_tri_schwarz_sparse2(basis, auxbasis, indicesA, indicesB, indicesC, ncores)
                    # print(ints3c2e)

                    # Some tests to see how many useless integrals are still being evaluated after Schwarz screening
                    # mask = np.abs(ints3c2e) < 1e-9
                    # count_below_threshold = np.count_nonzero(mask)

                    # print("Number of elements below 1e-9:", count_below_threshold)
                    # print('Percentage of significant calculated: ', count_below_threshold/nsignificant*100)
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
                    print('Time taken '+str(durationSchwarz)+' seconds.\n', flush=True)
                    

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
                    print('Time taken to evaluate the "diagonal" of 4c2e ERI tensor: ', duration_4c2e_diag)
                    
                    # Calculate the square roots required for 
                    duration_square_roots = 0.0
                    start_square_roots = timer()
                    sqrt_ints4c2e_diag = np.sqrt(np.abs(ints4c2e_diag))
                    sqrt_diag_ints2c2e = np.sqrt(np.abs(np.diag(ints2c2e)))
                    duration_square_roots = timer() - start_square_roots
                    print('Time taken to evaluate the square roots needed: ', duration_square_roots)

                    start_indices_calc = timer()
                    count_significant = Integrals.schwarz_helpers.calc_count_3c2e_schwarz(sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, basis.bfs_nao, auxbasis.bfs_nao, threshold_schwarz)
                    
                    duration_indices_calc = timer() - start_indices_calc
                    print('Time for total significant integral count calculation: ', duration_indices_calc)
                    # print('Time for array concatenation: ', duration_concatenation)

                    
                    print('No. of elements in the standard three-centered two electron ERI tensor: ', nints3c2e, flush=True)
                    print('No. of elements in the triangular three-centered two electron ERI tensor: ', nints3c2e_tri, flush=True)
                    print('No. of significant triplets based on Schwarz inequality and triangularity: ' + str(count_significant) + ' or '+str(np.round(count_significant/nints3c2e*100,1)) + '% of original', flush=True)
                    print('Schwarz screening partially done!')
                    durationSchwarz = timer() - startSchwarz
                    print('Total time taken for Schwarz screening (partial) '+str(durationSchwarz)+' seconds.\n', flush=True)
                    
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
                    print('Time taken to evaluate the "diagonal" of 4c2e ERI tensor: ', duration_4c2e_diag)
                    
                    # Calculate the square roots required for 
                    duration_square_roots = 0.0
                    start_square_roots = timer()
                    sqrt_ints4c2e_diag = np.sqrt(np.abs(ints4c2e_diag))
                    sqrt_diag_ints2c2e = np.sqrt(np.abs(np.diag(ints2c2e)))
                    duration_square_roots = timer() - start_square_roots
                    print('Time taken to evaluate the square roots needed: ', duration_square_roots)

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
                    print('Total time taken for Schwarz screening '+str(durationSchwarz)+' seconds.\n', flush=True)
                    
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
                else:
                    ints3c2e = Integrals.rys_3c2e_symm(basis, auxbasis)
                    
                
            else:
                start2c2e = timer()
                ints2c2e = Integrals.conv_2c2e_symm(auxbasis) #TODO Isn't implemented yet
                duration2c2e = timer() - start2c2e
                print('Time taken for two-centered two-electron integrals '+str(duration2c2e)+' seconds.\n', flush=True)
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
                #TODO: The following solve step is very sloww and makes the Coulomb time much longer. Try to make it faster.
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
            if DF_algo==4 or DF_algo==5 or DF_algo==6 or DF_algo==8:
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
            

        durationCoulomb = timer() - startCoulomb
        print('Time taken for Coulomb term related calculations (integrals, screening, prelims..) '+str(durationCoulomb)+' seconds.\n', flush=True)
        
        
        scf_converged = False
        durationgrids = 0

        if grids is None:
            startGrids = timer()
            print('\nGenerating grids...\n\n', flush=True)
            # To get the same energies as PySCF (level=5) upto 1e-7 au, use the following settings
            # radial_precision = 1.0e-13
            # level=3
            # pruning by density with threshold = 1e-011
            # alpha_min and alpha_max corresponding to QZVP
            print('Grids level: ', gridsLevel)
            # Generate grids for XC term
            basisGrids = Basis(mol, {'all':Basis.load(mol=mol, basis_name='def2-QZVP')})
            grids = Grids(mol, basis=basisGrids, level = gridsLevel, ncores=ncores)

            print('done!', flush=True)
            durationgrids = timer() - startGrids
            print('Time taken '+str(durationgrids)+' seconds.\n', flush=True)

            # Begin pruning the grids based on density (rho)
            # Evaluate ao_values to calculate rho
            print('\nPruning generated grids by rho...\n\n', flush=True)
            startGrids_prune_rho = timer()
            threshold_rho = 1e-011
            ngrids_temp = grids.coords.shape[0]
            ndeleted = 0
            blocksize_temp = 50000
            nblocks_temp = ngrids_temp//blocksize_temp
            weightsNew = None
            coordsNew = None
            for iblock in range(nblocks_temp+1):
                offset = iblock*blocksize_temp
                weights_block = grids.weights[offset : min(offset+blocksize_temp,ngrids_temp)]
                coords_block = grids.coords[offset : min(offset+blocksize_temp,ngrids_temp)] 
                ao_value_block = Integrals.bf_val_helpers.eval_bfs(basis, coords_block)  
                rho_block = contract('ij,mi,mj->m', dmat, ao_value_block, ao_value_block)
                zero_indices = np.where(np.abs(rho_block*weights_block) < threshold_rho)[0]
                ndeleted += len(zero_indices)
                weightsNew_block = np.delete(weights_block, zero_indices)
                coordsNew_block = np.delete(coords_block, zero_indices, 0)
                if weightsNew_block.shape[0]>0:
                    if weightsNew is None:
                        weightsNew = weightsNew_block
                        coordsNew = coordsNew_block
                    else:
                        weightsNew = np.concatenate((weightsNew, weightsNew_block))
                        coordsNew = np.concatenate([coordsNew, coordsNew_block], axis=0)

            grids.coords = coordsNew
            grids.weights = weightsNew
            print('done!', flush=True)
            durationgrids_prune_rho = timer() - startGrids_prune_rho
            print('Time taken '+str(durationgrids_prune_rho)+' seconds.\n', flush=True)
            print('\nDeleted '+ str(ndeleted) + ' grid points.', flush=True)
            
        else:
            print('\nUsing the user supplied grids!\n\n', flush=True)
        

        # Grid information initial
        print('\nNo. of supplied/generated grid points: ', grids.coords.shape[0], flush=True)

        # Prune grids based on weights
        # start_pruning_weights = timer()
        # print('\nPruning grids based on weights....', flush=True)
        # zero_indices = np.where(np.logical_and(grids.weights>=-1.0e-12, grids.weights<=1.e-12))
        # grids.weights = np.delete(grids.weights, zero_indices)
        # grids.coords = np.delete(grids.coords, zero_indices, 0)
        # print('done!', flush=True)
        # duration_pruning_weights = timer() - start_pruning_weights
        # print('\nTime taken '+str(duration_pruning_weights)+' seconds.\n', flush=True)
        # print('\nNo. of grid points after screening by weights: ', grids.coords.shape[0], flush=True)

        print('Size (in GB) for storing the coordinates of grid:      ', grids.coords.nbytes/1e9, flush=True)
        print('Size (in GB) for storing the weights of grid:          ', grids.weights.nbytes/1e9, flush=True)
        print('Size (in GB) for storing the density at gridpoints:    ', grids.weights.nbytes/1e9, flush=True)

        # Sort the grids for slightly better performance with batching (doesn't seem to make much difference)
        if sortGrids:
            print('\nSorting grids ....', flush=True)
            # Function to sort grids
            def get_ordered_list(points, x, y, z):
                points.sort(key = lambda p: (p[0] - x)**2 + (p[1] - y)**2 + (p[2] - z)**2)
                # print(points[0:10])
                return points
            # Make a single array of coords and weights
            coords_weights = np.c_[grids.coords, grids.weights]
            coords_weights = np.array(get_ordered_list(coords_weights.tolist(), min(grids.coords[:,0]), min(grids.coords[:,1]), min(grids.coords[:,2])))
            # Now go back to two arrays for coords and weights
            grids.weights = coords_weights[:,3]
            grids.coords = coords_weights[:,0:3]
            coords_weights = 0#None
            print('done!', flush=True)

        # blocksize = 10000
        ngrids = grids.coords.shape[0]
        nblocks = ngrids//blocksize
        print('\nWill use batching to evaluate the XC term for memory efficiency.', flush=True)
        print('Batch size: ', blocksize, flush=True)
        print('No. of batches: ', nblocks+1, flush=True)


        #### Some preliminary stuff for XC evaluation
        durationXCpreprocessing = 0
        list_nonzero_indices = None
        count_nonzero_indices = None
        list_ao_values = None
        list_ao_grad_values = None
        if xc_bf_screen:
            xc_family_dict = {1:'LDA',2:'GGA',4:'MGGA'} 
            # Create a LibXC object  
            funcx = pylibxc.LibXCFunctional(xc[0], "unpolarized")
            funcc = pylibxc.LibXCFunctional(xc[1], "unpolarized")
            x_family_code = funcx.get_family()
            c_family_code = funcc.get_family()
            ### Find the list of significanlty contributing bfs for xc evaluations
            startXCpreprocessing = timer()
            print('\nPreliminary processing for XC term evaluations...', flush=True)
            print('Calculating the value of basis functions (atomic orbitals) and get the indices of siginificantly contributing functions...', flush=True)
            # Calculate the value of basis functions for all grid points in batches
            # and find the indices of basis functions that have a significant contribution to those batches for each batch
            list_nonzero_indices, count_nonzero_indices = Integrals.bf_val_helpers.nonzero_ao_indices(basis, grids.coords, blocksize, nblocks, ngrids)
            print('done!', flush=True)
            durationXCpreprocessing = timer() - startXCpreprocessing
            print('Time taken '+str(durationXCpreprocessing)+' seconds.\n', flush=True)
            print('Maximum no. of basis functions contributing to a batch of grid points:   ', max(count_nonzero_indices))
            print('Average no. of basis functions contributing to a batch of grid points:   ', int(np.mean(count_nonzero_indices)))

            
            durationAO_values = 0
            if save_ao_values:
                startAO_values = timer()
                if xc_family_dict[x_family_code]=='LDA' and xc_family_dict[c_family_code]=='LDA':
                    print('\nYou have asked to save the values of significant basis functions on grid points so as to avoid recalculation for each SCF cycle.', flush=True)
                    memory_required = sum(count_nonzero_indices*blocksize)*8/1024/1024/1024
                    print('Please note: This will require addtional memory that is approximately :'+ str(np.round(memory_required,1))+ ' GB', flush=True)
                    print('Calculating the value of significantly contributing basis functions (atomic orbitals)...', flush=True)
                    list_ao_values = []
                    # Loop over batches
                    for iblock in range(nblocks+1):
                        offset = iblock*blocksize
                        coords_block = grids.coords[offset : min(offset+blocksize,ngrids)]   
                        ao_values_block = Integrals.bf_val_helpers.eval_bfs(basis, coords_block, parallel=True, non_zero_indices=list_nonzero_indices[iblock][0:count_nonzero_indices[iblock]])
                        if self.use_gpu and self.keep_ao_in_gpu:
                            list_ao_values.append(cp.asarray(ao_values_block))
                        else:
                            list_ao_values.append(ao_values_block)
                    #Free memory 
                    ao_values_block = 0 
                if xc_family_dict[x_family_code]!='LDA' or xc_family_dict[c_family_code]!='LDA':
                    print('\nYou have asked to save the values of significant basis functions and their gradients on grid points so as to avoid recalculation for each SCF cycle.', flush=True)
                    memory_required = 4*sum(count_nonzero_indices*blocksize)*8/1024/1024/1024
                    print('Please note: This will require addtional memory that is approximately :'+ str(np.round(memory_required,1))+ ' GB', flush=True)
                    print('Calculating the value of significantly contributing basis functions (atomic orbitals)...', flush=True)
                    list_ao_values = []
                    list_ao_grad_values = []
                    # Loop over batches
                    for iblock in range(nblocks+1):
                        offset = iblock*blocksize
                        coords_block = grids.coords[offset : min(offset+blocksize,ngrids)]   
                        # ao_values_block = Integrals.bf_val_helpers.eval_bfs(basis, coords_block, parallel=True, non_zero_indices=list_nonzero_indices[iblock][0:count_nonzero_indices[iblock]])
                        ao_values_block, ao_grad_values_block = Integrals.bf_val_helpers.eval_bfs_and_grad(basis, coords_block, parallel=True, non_zero_indices=list_nonzero_indices[iblock][0:count_nonzero_indices[iblock]])
                        if self.use_gpu and self.keep_ao_in_gpu:
                            list_ao_values.append(cp.asarray(ao_values_block))
                            list_ao_grad_values.append(cp.asarray(ao_grad_values_block))
                        else:
                            list_ao_values.append(ao_values_block)
                            list_ao_grad_values.append(ao_grad_values_block)
                    #Free memory 
                    ao_values_block = 0 
                    ao_grad_values_block =0
                print('done!', flush=True)
                durationAO_values = timer() - startAO_values
                print('Time taken '+str(durationAO_values)+' seconds.\n', flush=True)
        
        if self.use_gpu:
            grids.coords = cp.asarray(grids.coords, dtype=cp.float64)
            grids.weights = cp.asarray(grids.weights, dtype=cp.float64)
        
        #-------XC Stuff start----------------------

        funcid = self.xc

        xc_family_dict = {1:'LDA',2:'GGA',4:'MGGA'} 

        # Create a LibXC object  
        funcx = pylibxc.LibXCFunctional(funcid[0], "unpolarized")
        funcc = pylibxc.LibXCFunctional(funcid[1], "unpolarized")
        x_family_code = funcx.get_family()
        c_family_code = funcc.get_family()


        print('\n\n------------------------------------------------------', flush=True)
        print('Exchange-Correlation Functional')
        print('------------------------------------------------------\n', flush=True)
        print('XC Functional IDs supplied: ', funcid, flush=True)
        print('\n\nDescription of exchange functional: \n')
        print('The Exchange function belongs to the family:', xc_family_dict[x_family_code], flush=True)
        print(funcx.describe())
        print('\n\nDescription of correlation functional: \n', flush=True)
        print(' The Correlation function belongs to the family:', xc_family_dict[c_family_code], flush=True)
        print(funcc.describe())
        print('------------------------------------------------------\n', flush=True)
        print('\n\n', flush=True)
        #-------XC Stuff end----------------------

        Etot = 0

        itr = 1
        Enn = self.nuclear_rep_energy(mol)

        #diis = scf.CDIIS()
        dmat_old = 0
        J_diff = 0
        Ecoul = 0.0
        Ecoul_temp = 0.0

        durationxc = 0
        

        while not (scf_converged or itr==max_itr+1):
            startIter = timer()
            if itr==1:
                dmat_diff = dmat
                # Coulomb (Hartree) matrix
                if not isDF:
                    # J = np.einsum('ijkl,lk',ints4c2e,dmat)
                    J = contract('ijkl,ij',ints4c2e,dmat)
                else:
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
                        scipy.linalg.solve(ints2c2e, df_coeff, assume_a='pos', overwrite_a=False, overwrite_b=True)
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
                            df_coeff = scipy.linalg.solve(ints2c2e, gamma_alpha, assume_a='pos', overwrite_a=False, overwrite_b=False)
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
                    durationDF = durationDF + timer() - startDF


                    
                
            else:
                dmat_diff = dmat-dmat_old
                if not isDF:
                    # J_diff = np.einsum('ijkl,lk',ints4c2e,dmat_diff)
                    J_diff = contract('ijkl,ij',ints4c2e,dmat_diff)
                    J += J_diff
                
                    
                else:
                    startDF = timer()
                    # Using DF calculate the density fitting coefficients of the auxiliary basis
                    if DF_algo==1:
                        df_coeff = contract('ijk,jk', df_coeff0, dmat_diff)
                        J_diff = contract('ijk,k',ints3c2e,df_coeff)
                        J += J_diff
                    if DF_algo==2:
                        df_coeff = contract('ijk,jk', Qpq, dmat) # Not exactly the same thing as above (not the auxiliary density coeffs)
                        J = contract('ijk,i',Qpq,df_coeff)
                    if DF_algo==3:
                        df_coeff = contract('pqP,pq->P', ints3c2e, dmat) # This is actually the gamma_alpha in this paper ()
                        scipy.linalg.solve(ints2c2e, df_coeff, assume_a='pos', overwrite_a=False, overwrite_b=True) # This gives the actual df coeff
                        J = contract('ijk,k', ints3c2e, df_coeff)
                    if DF_algo==4 or DF_algo==5: # Fastest and triangular (half the memory of algo 3)
                        dmat_temp = dmat.copy()
                        dmat_temp[indices_dmat_tri_2] = 2*dmat[indices_dmat_tri_2] # Double the non-diagonal elements of the triangular density matrix
                        dmat_tri = dmat_temp[indices_dmat_tri]
                        df_coeff = contract('pP,p->P', ints3c2e, dmat_tri) # This is actually the gamma_alpha in this paper (https://aip.scitation.org/doi/pdf/10.1063/1.1567253)
                        scipy.linalg.solve(ints2c2e, df_coeff, assume_a='pos', overwrite_a=False, overwrite_b=True)
                        J_tri = contract('pP,P', ints3c2e, df_coeff)
                        J = np.zeros((basis.bfs_nao, basis.bfs_nao))
                        J[indices_dmat_tri] = J_tri
                        J += J.T - np.diag(np.diag(J))
                    if DF_algo==6: 
                        dmat_temp = dmat.copy()
                        dmat_temp[indices_dmat_tri_2] = 2*dmat[indices_dmat_tri_2] # Double the non-diagonal elements of the triangular density matrix
                        dmat_tri = dmat_temp[indices_dmat_tri]
                        startDF_gamma = timer()
                        # df_coeff_1 = contract('pP,p->P', ints3c2e, dmat_tri) # This is actually the gamma_alpha in this paper (https://aip.scitation.org/doi/pdf/10.1063/1.1567253)
                        gamma_alpha = Integrals.schwarz_helpers.df_coeff_calculator(ints3c2e, dmat_tri, indicesA, indicesB, indicesC, auxbasis.bfs_nao, ncores)
                        # Integrals.schwarz_helpers.df_coeff_calculator.parallel_diagnostics(level=4)
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
                        J_tri = Integrals.schwarz_helpers.J_tri_calculator(ints3c2e, df_coeff, indicesA, indicesB, indicesC, int(basis.bfs_nao*(basis.bfs_nao+1)/2))
                        # if self.use_gpu:
                        #     J_tri = Integrals.schwarz_helpers.J_tri_calculator_cupy(ints3c2e, df_coeff, indicesA, indicesB, indicesC, int(basis.bfs_nao*(basis.bfs_nao+1)/2))
                        # else:
                        #     J_tri = Integrals.schwarz_helpers.J_tri_calculator(ints3c2e, df_coeff, indicesA, indicesB, indicesC, int(basis.bfs_nao*(basis.bfs_nao+1)/2))
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
                    durationDF = durationDF + timer() - startDF
                # J += J_diff
            if self.use_gpu:
                J = cp.asarray(J, dtype=cp.float64)
                cp.cuda.Stream.null.synchronize()
            
                
            # XC energy and potential
            startxc = timer()

            
            if not self.use_gpu:
                if XC_algo==1:
                    # Much slower than JOBLIB version
                    # Still keeping it because, it can be useful when using GPUs
                    Exc, Vxc = Integrals.eval_xc_1(basis, dmat, grids.weights, grids.coords, funcid, blocksize=blocksize, debug=debug, \
                                                list_nonzero_indices=list_nonzero_indices, count_nonzero_indices=count_nonzero_indices, \
                                                    list_ao_values=list_ao_values, list_ao_grad_values=list_ao_grad_values)
                if XC_algo==2:
                    # Much faster than above and stable too, therefore this should be default now.
                    # Used to unstable and had memory leaks,
                    # But now all that is fixed by using threadpoolctl, garbage collection or freeing up memory after XC evaluation at each iteration
                    # print(list_ao_values)
                    Exc, Vxc = Integrals.eval_xc_2(basis, dmat, grids.weights, grids.coords, funcid, ncores=ncores, blocksize=blocksize, \
                                                list_nonzero_indices=list_nonzero_indices, count_nonzero_indices=count_nonzero_indices, \
                                                    list_ao_values=list_ao_values, list_ao_grad_values=list_ao_grad_values, debug=debug)
                    
                if XC_algo==3:
                    # Much faster than above and stable too, therefore this should be default now.
                    # Used to unstable and had memory leaks,
                    # But now all that is fixed by using threadpoolctl, garbage collection or freeing up memory after XC evaluation at each iteration
                    # print(list_ao_values)
                    with threadpool_limits(limits=1, user_api='blas'):
                        Exc, Vxc = Integrals.eval_xc_3(basis, dmat, grids.weights, grids.coords, funcid, ncores=ncores, blocksize=blocksize, \
                                                    list_nonzero_indices=list_nonzero_indices, count_nonzero_indices=count_nonzero_indices, \
                                                        list_ao_values=list_ao_values, list_ao_grad_values=list_ao_grad_values, debug=debug)
            else: # GPU
                if XC_algo==1:
                    Exc, Vxc = Integrals.eval_xc_1_cupy(basis, dmat_cp, grids.weights, grids.coords, funcid, blocksize=blocksize, debug=debug, \
                                                list_nonzero_indices=list_nonzero_indices, count_nonzero_indices=count_nonzero_indices, \
                                                list_ao_values=list_ao_values, list_ao_grad_values=list_ao_grad_values, use_libxc=self.use_libxc,\
                                                nstreams=self.n_streams, ngpus=self.n_gpus, freemem=self.free_gpu_mem, threads_per_block=threads_per_block,
                                                type=precision_XC)
                if XC_algo==2:
                    Exc, Vxc = Integrals.eval_xc_2_cupy(basis, dmat_cp, grids.weights, cp.asnumpy(grids.coords), funcid, ncores=ncores, blocksize=blocksize, \
                                                list_nonzero_indices=list_nonzero_indices, count_nonzero_indices=count_nonzero_indices, \
                                                    list_ao_values=list_ao_values, list_ao_grad_values=list_ao_grad_values, debug=debug)
                if XC_algo==3:
                    # Default for GPUs
                    Exc, Vxc = Integrals.eval_xc_3_cupy(basis, dmat_cp, grids.weights, grids.coords, funcid, blocksize=blocksize, debug=debug, \
                                                list_nonzero_indices=list_nonzero_indices, count_nonzero_indices=count_nonzero_indices, \
                                                list_ao_values=list_ao_values, list_ao_grad_values=list_ao_grad_values, use_libxc=self.use_libxc,\
                                                nstreams=self.n_streams, ngpus=self.n_gpus, freemem=self.free_gpu_mem, threads_per_block=threads_per_block,
                                                type=precision_XC)

                Vxc = cp.asarray(Vxc, dtype=cp.float64)

            durationxc = durationxc + timer() - startxc
            

            dmat_old = dmat

            if self.use_gpu:
                Enuc = contract('ij,ji->', dmat_cp, V)
                Ekin = contract('ij,ji->', dmat_cp, T)
                Ecoul = contract('ij,ji->', dmat_cp, J)*0.5
            else:
                with threadpool_limits(limits=ncores, user_api='blas'):
                    # print('Energy contractions', controller.info())
                    Enuc = contract('ij,ji->', dmat, V)
                    Ekin = contract('ij,ji->', dmat, T)
                    Ecoul = contract('ij,ji->', dmat, J)*0.5
            if isDF and DF_algo==6:
                Ecoul = Ecoul*2 - 0.5*Ecoul_temp # This is the correct formula for Coulomb energy with DF

            Etot_new = Exc + Enuc + Ekin + Enn + Ecoul

            print('\n\n\n------Iteration '+str(itr)+'--------\n\n', flush=True)
            print('Energies')
            print('Electron-Nuclear Energy      ', Enuc, flush=True)
            print('Nuclear repulsion Energy     ', Enn, flush=True)
            print('Kinetic Energy               ', Ekin, flush=True)
            print('Coulomb Energy               ', Ecoul, flush=True)
            print('Exchange-Correlation Energy  ', Exc, flush=True)
            print('-------------------------')
            print('Total Energy ',Etot_new, flush=True)
            print('-------------------------\n\n\n', flush=True)

            print('Energy difference : ',abs(Etot_new-Etot), flush=True)


            KS = H + J + Vxc 

            #### DIIS
            startDIIS = timer()
            diis_start_itr = 1
            if itr >= diis_start_itr:
                if not self.use_gpu:
                    with threadpool_limits(limits=ncores, user_api='blas'):
                        # print('DIIS ', controller.info())
                        KS = self.DIIS(S, dmat, KS)
                else:
                    KS = self.DIIS_cupy(S, dmat_cp, KS)
                    cp.cuda.Stream.null.synchronize()
            durationDIIS = durationDIIS + timer() - startDIIS

            #### Solve KS equation (Diagonalize KS matrix)
            startKS = timer()
            if self.use_gpu:
                eigvalues, eigvectors = self.solve_cupy(KS, S, orthogonalize=True) # Orthogonalization is necessary with CUDA
                mo_occ = self.getOcc_cupy(mol, eigvalues, eigvectors)
                dmat = self.gen_dm_cupy(eigvectors, mo_occ)
                dmat_cp = dmat
                dmat = cp.asnumpy(dmat)
                cp.cuda.Stream.null.synchronize()
            else:
                with threadpool_limits(limits=ncores, user_api='blas'):
                    # print('KS eigh', controller.info())
                    eigvalues, eigvectors = self.solve(KS, S, orthogonalize=True)
                mo_occ = self.getOcc(mol, eigvalues, eigvectors)
                dmat = self.gen_dm(eigvectors, mo_occ)
            durationKS = durationKS + timer() - startKS

            # Check when to switch to double precision for XC
            if self.use_gpu:
                if precision_XC is cp.float32:
                    if abs(Etot_new-Etot)/abs(Etot_new)<5e-7:
                        precision_XC = cp.float64
                        print('\nSwitching to double precision for XC evaluation after '+str(itr) +' iterations!', flush=True)

            durationItr = timer() - startIter
            print('\n\nTime taken for the previous iteration: '+str(durationItr)+'\n\n', flush=True)

            # Check convergence criteria
            if abs(Etot_new-Etot)<conv_crit:
                scf_converged = True
                print('\nSCF Converged after '+str(itr) +' iterations!', flush=True)
                Etot = Etot_new
                print('\n-------------------------------------', flush=True)
                print('Total Energy = ', Etot, flush=True)
                print('-------------------------------------\n\n', flush=True)
                break

            Etot = Etot_new
            itr = itr + 1
            

        if itr>=max_itr and not scf_converged:
            print('\nSCF NOT Converged after '+str(itr-1) +' iterations!', flush=True)


        durationSCF = timer() - startSCF
        # print(dmat)
        print('\nTime taken : '+str(durationSCF) +' seconds.', flush=True)
        print('\n\n', flush=True)
        print('-------------------------------------', flush=True)
        print('Profiling', flush=True)
        print('-------------------------------------', flush=True)
        print('Preprocessing                          ', durationXCpreprocessing + durationAO_values + durationgrids_prune_rho + durationSchwarz)
        if isDF:
            print('Density Fitting                        ', durationDF, flush=True)
            if DF_algo==6:
                print('DF (gamma)                             ', durationDF_gamma, flush=True)
                print('DF (coeff)                             ', durationDF_coeff, flush=True)
                print('DF (Jtri)                              ', durationDF_Jtri, flush=True)
        print('DIIS                                   ', durationDIIS, flush=True)
        print('KS matrix diagonalization              ', durationKS, flush=True)
        print('One electron Integrals (S, T, Vnuc)    ', duration1e, flush=True)
        if isDF:
            print('Coulomb Integrals (2c2e + 3c2e)        ', durationCoulomb-durationSchwarz, flush=True)
        if not isDF:
            print('Coulomb Integrals (4c2e)               ', durationCoulomb-durationSchwarz, flush=True)
        print('Grids construction                     ', durationgrids, flush=True)
        print('Exchange-Correlation Term              ', durationxc, flush=True)
        totalTime = durationXCpreprocessing + durationAO_values + duration1e + durationCoulomb + \
            durationgrids + durationxc + durationDF + durationKS + durationDIIS + durationgrids_prune_rho 
        print('Misc.                                  ', durationSCF - totalTime, flush=True)
        print('Complete SCF                           ', durationSCF, flush=True)

        if self.use_gpu:
            for igpu in range(self.n_gpus):
                cp.cuda.Device(igpu).use()
                cp._default_memory_pool.free_all_blocks()

            # Switch back to main GPU
            cp.cuda.Device(0).use()
        return Etot, dmat
        

    

    

        

        

