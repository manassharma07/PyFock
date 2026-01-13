# DFT.py
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
from pyfock.Utils import print_pyfock_logo
from pyfock.Utils import print_scientist
# Print system information 
from pyfock.Utils import print_sys_info
import pyfock.Mol as Mol
import pyfock.Basis as Basis
# import pyfock.Integrals as Integrals
import pyfock.Integrals as Integrals
import pyfock.Grids as Grids
from threadpoolctl import ThreadpoolController, threadpool_info, threadpool_limits
# controller = ThreadpoolController()
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
from pyfock.DFT_Helper_Coulomb import density_fitting_prelims_for_DFT_development
from pyfock.DFT_Helper_Coulomb import Jmat_from_density_fitting

# @njit(parallel=True, cache=True, fastmath=True, error_model="numpy")
# def compute_B(errVecs):
#     nKS = errVecs.shape[0]
#     B = np.zeros((nKS + 1, nKS + 1))
#     B[-1, :] = B[:, -1] = -1.0
#     B[-1, -1] = 0.0
#     for i in prange(nKS):
#         # errVec_i_conj_T = errVecs[i].conj().T
#         errVec_i_conj_T = errVecs[i].T
#         for j in range(i + 1):
#             # B[i, j] = B[j, i] = np.real(np.trace(np.dot(errVec_i_conj_T, errVecs[j])))
#             B[i, j] = B[j, i] = np.trace(np.dot(errVec_i_conj_T, errVecs[j]))
#     return B

class DFT:
    """
    A class for performing Density Functional Theory (DFT) calculations 
    with optional support for density fitting (DF), GPU acceleration, and LibXC.

    Parameters
    ----------
    mol : Molecule
        Molecular object on which the DFT calculation is to be performed.
    
    basis : Basis
        Orbital basis set used for the SCF calculation.

    auxbasis : Basis, optional
        Auxiliary basis set for density fitting (DF). If None, a default will be assigned.

    conv_crit : float, optional
        Convergence criterion for the SCF cycle in Hartrees (default is 1e-7).

    dmat_guess_method : str, optional
        Method for the initial density matrix guess (e.g., 'core', 'huckel').

    xc : list or str, optional
        Exchange-correlation functional specification. If None, defaults to LDA (`[1, 7]`).

    grids : object, optional
        Precomputed numerical integration grids. If None, they will be generated automatically.

    gridsLevel : int, optional
        Level of numerical integration grid refinement (default is 3).

    blocksize : int, optional
        Block size for XC grid evaluations. Defaults depend on whether GPU is used.

    save_ao_values : bool, optional
        If True, saves AO values to reuse during XC evaluation. Increases speed but uses more memory.

    use_gpu : bool, optional
        Whether to use GPU acceleration.

    ncores : int, optional
        Number of CPU cores to use (default is 2).

    Attributes
    ----------
    dmat : ndarray
        Initial guess for the density matrix, will be computed during setup.

    KSmats : list
        List of Kohn–Sham matrices used in DIIS extrapolation.

    errVecs : list
        List of error vectors for DIIS.

    max_itr : int
        Maximum number of SCF iterations (default is 50).

    isDF : bool
        Whether to use density fitting for Coulomb integrals.

    rys : bool
        Whether to use Rys quadrature for evaluating electron repulsion integrals.

    DF_algo : int
        Algorithm selector for DF (reserved for developer use).

    XC_algo : int
        Algorithm selector for XC evaluation (2 for CPU, 3 for GPU).

    sortGrids : bool
        Whether to sort DFT integration grids (not recommended).

    xc_bf_screen : bool
        Enable basis function screening for XC term evaluation.

    threshold_schwarz : float
        Threshold for Schwarz screening (default is 1e-9).

    strict_schwarz : bool
        If True, applies stricter Schwarz screening.

    cholesky : bool
        If True, uses Cholesky decomposition for DF.

    orthogonalize : bool
        If True, orthogonalizes AO basis functions.

    sao : bool
        If True, uses SAO basis instead of CAO basis.

    keep_ao_in_gpu : bool
        Whether to retain AO values in GPU memory during SCF (if `save_ao_values` is True).

    use_libxc : bool
        Whether to use LibXC for XC functional evaluation (recommended off for GPU).

    n_streams : int
        Number of CUDA streams to use (if applicable).

    n_gpus : int
        Number of GPUs to use.

    free_gpu_mem : bool
        Whether to forcibly free GPU memory after use.

    max_threads_per_block : int
        Maximum threads per CUDA block supported by the device.

    threads_x : int
        CUDA thread configuration (X dimension).

    threads_y : int
        CUDA thread configuration (Y dimension).

    dynamic_precision : bool
        Whether to use precision switching during XC evaluation for performance gains.

    keep_ints3c2e_in_gpu : bool
        Whether to keep 3-center 2-electron integrals in GPU memory to avoid transfers.

    debug : bool
        If True, prints debugging output during DFT calculations.

    Notes
    -----
    - This class supports SCF DFT calculations with density fitting (DF).
    - GPU support is optional and provides significant speed-up for large systems.
    - The class is tightly integrated with PyFock and LibXC libraries.

    Examples
    --------
    >>> mol = Molecule(...)
    >>> basis = Basis(mol, ...)
    >>> dft = DFT(mol, basis, xc='PBE', use_gpu=True)
    >>> dft.run_scf()
    """
    def __init__(self, mol, basis, auxbasis=None, conv_crit=1e-7, dmat_guess_method=None, 
                xc=None, grids=None, gridsLevel=3, blocksize=None, 
                save_ao_values=False, use_gpu=False, ncores=1): 
         
        self.mol = mol
        """ Molecular object for which the DFT calculation will be performed """
        if self.mol is None:
            print('ERROR: A Mol object is required to initialize a DFT object.')
        
        self.basis = basis
        """ Basis object for corresponding to the molecule for the DFT calculation """
        if self.basis is None:
            print('ERROR: A Basis object is required to initialize a DFT object.')

        
        self.dmat_guess_method = dmat_guess_method
        """ Initial guess for the density matrix """
        if self.dmat_guess_method is None:
            self.dmat_guess_method = 'core'

        self.dmat = None
        """ Initial density matrix guess for SCF. Will get updated at each SCF iteration. """
        
        self.xc = xc
        """ Exchange-Correlation Functional """
        if self.xc is None: 
            self.xc = [1, 7] #LDA
            print("XC not specified, defaulting to LDA functional (LibXC codes: 1, 7)")

        # DIIS
        self.KSmats = []
        self.errVecs = []
        self.diisSpace = 6

        self.conv_crit = conv_crit
        """ Convergence criterion for SCF (in Hartrees) """
        # if self.conv_crit is None:
        #     self.conv_crit = 1.0E-7

        self.max_itr = 50 
        """ Maximum number of iterations for SCF """
        # if self.max_itr is None:
        #     self.max_itr = 50

        self.ncores = ncores
        """ Number of cores to be used for DFT calculation """
        if self.ncores is None:
            self.ncores = 2

        self.grids = grids 
        """ Atomic grids for DFT calculation (If None or not supplied, will be generated using NumGrid) """
        self.gridsLevel = gridsLevel
        """ Atomic grids for DFT calculation """

        self.isDF = True
        """ Use density fitting (DF) for two-electron Coulomb integrals. This is only for developers. Users should not change it. """

        self.auxbasis = auxbasis
        """ Basis object to be used as the auxiliary basis for DF """
        if self.auxbasis is None:
            auxbasis = Basis(mol, {'all':Basis.load(mol=mol, basis_name='def2-universal-jfit')})

        self.rys = True
        """ Use rys quadrature for the evaluation of two electron integrals (with and without DF)"""

        self.DF_algo = 10
        """ This is only for developers. Users should not change it. """

        self.blocksize = blocksize
        """ Block size for the evaulation of XC term on grids. For CPUs a value of ~5000 is recommended. For GPUs, a value >20480 is recommended. """
        if blocksize is None:
            if use_gpu:
                self.blocksize = 20480
            else:
                self.blocksize = 5000
        
        self.XC_algo = None
        """ This is only for developers. Users should not change it. The algorithm for XC evaluation should be 2 for CPU and 3 for GPU."""
        if use_gpu:
            self.XC_algo = 3
        else:
            self.XC_algo = 2 

        self.debug = False
        """ Turn on printing debug statements """
        self.sortGrids = False
        """ Enable/Disable sorting of DFT grids. Doesn't seem to offer any signficant advantage."""
        self.save_ao_values = save_ao_values
        """ Whether to save atomic orbital (AO) values for reuse during XC evaluation. Improves performance but requires more memory. """
        self.xc_bf_screen = True
        """ Enable screening of basis functions for XC term evaluation to reduce computation time drastically. """
        self.threshold_schwarz = 1e-09
        """ Threshold for Schwarz screening of two-electron integrals. Smaller values increase accuracy but reduce sparsity. """
        self.strict_schwarz = True
        """ If True, enforce stricter Schwarz screening to aggressively eliminate small two-electron integrals. """
        self.cholesky = True
        """ Whether to use Cholesky decomposition for DF. Slightly speeds up calculations. """
        self.orthogonalize = True
        """ Apply orthogonalization to the AO basis. Should be True for most standard calculations. """

        self.mo_occupations = None
        """ Molecular orbital occupations. Will be computed during SCF. """
        self.mo_coefficients = None
        """ Molecular orbital coefficients. Will be computed during SCF. """
        self.mo_energies = None
        """ Molecular orbital energies. Will be computed during SCF. """
        self.Total_energy = None
        """ Total energy of the system. Will be computed during SCF. """
        self.J_energy = None
        """ Coulomb energy contribution. Will be computed during SCF. """
        self.XC_energy = None
        """ Exchange-correlation energy contribution. Will be computed during SCF. """
        self.Nuclear_rep_energy = None
        """ Nuclear repulsion energy. Will be computed during SCF. """
        self.Kinetic_energy = None
        """ Kinetic energy contribution. Will be computed during SCF. """
        self.Nuc_energy = None
        """ Nuclear potential energy contribution. Will be computed during SCF. """

        self.converged = False
        """ Whether the SCF has converged or not. Will be updated during SCF. """
        self.niter = 0
        """ Number of SCF iterations performed. Will be updated during SCF. """
        self.scf_energies = []
        """ List of SCF energies at each iteration. """

        # CAO or SAO
        self.sao = False
        """ Whether to use SAO basis or CAO basis. Default is CAO basis. """


        # GPU acceleration
        self.use_gpu = use_gpu
        """ Whether to use GPU acceleration or not """
        self.keep_ao_in_gpu = True
        """ Whether to keep the atomic orbitals for XC evaluation in GPU memory or CPU memory. Only relevant if save_ao_values = True. """
        self.use_libxc = True
        """ Whether to use LibXC's version of XC functionals or PyFock implementations. 
        Only relevant when GPU is used. For GPU calculations it is recommended to use PyFock 
        implementation as it avoids CPU-GPU transfers."""
        self.n_streams = 1
        self.n_gpus = 1
        """ Number of GPUs to be used """
        self.free_gpu_mem = False
        """ Whether the GPU memory should be freed by force or not"""
        try:
            self.max_threads_per_block = cuda.get_current_device().MAX_THREADS_PER_BLOCK
        except:
            self.max_threads_per_block = 1024
        
        self.threads_x = int(self.max_threads_per_block/16)
        self.threads_y = int(self.max_threads_per_block/64)
        self.dynamic_precision = False # Only for the XC term
        """ Whether to use dynamic precision switching for XC term or not """
        self.keep_ints3c2e_in_gpu = True
        """ Whether to keep the 3c2e integrals in GPU memory or not. 
        Recommended to keep in GPU memory to avoid CPU-GPU transfers at each iteration."""
        
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

    # def removeLinearDep(self, H, S):
    #     return 1 
    
    def nuclear_rep_energy(self, mol=None):
        """
        Compute the nuclear-nuclear repulsion energy.

        Parameters
        ----------
        mol : Molecule, optional
            Molecule object containing nuclear coordinates and charges. 
            If None, uses `self.mol`.

        Returns
        -------
        float
            The nuclear repulsion energy in Hartrees.
        """
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
        
    # full density matrix for RKS/RHF
    def gen_dm(self, mo_coeff, mo_occ):
        """
        Generate the density matrix from molecular orbital coefficients and occupations.

        Parameters
        ----------
        mo_coeff : ndarray
            Molecular orbital coefficient matrix.

        mo_occ : ndarray
            Array of MO occupation numbers.

        Returns
        -------
        ndarray
            Density matrix (RHF/RKS type).
        """
        mocc = mo_coeff[:,mo_occ>0]
   
        return np.dot(mocc*mo_occ[mo_occ>0], mocc.conj().T)
    
    def getOcc(self, mol=None, energy_mo=None, coeff_mo=None):
        """
        Assign occupation numbers to molecular orbitals based on their energies.

        Parameters
        ----------
        mol : Molecule
            Molecule object to extract the number of electrons.

        energy_mo : ndarray
            Array of MO energies.

        coeff_mo : ndarray
            Array of MO coefficients (not used but kept for compatibility).

        Returns
        -------
        ndarray
            Array of MO occupations (0 or 2 for RHF).
        """
        e_idx = np.argsort(energy_mo)
        e_sort = energy_mo[e_idx]
        nmo = energy_mo.size
        occ_mo = np.zeros(nmo)
        nocc = mol.nelectrons // 2
        occ_mo[e_idx[:nocc]] = 2
        return occ_mo
    
    def gen_dm_cupy(self, mo_coeff, mo_occ):
        """
        Generate the density matrix using CuPy for GPU acceleration.

        Parameters
        ----------
        mo_coeff : cp.ndarray
            Molecular orbital coefficient matrix (on GPU).

        mo_occ : cp.ndarray
            Array of MO occupation numbers (on GPU).

        Returns
        -------
        cp.ndarray
            Density matrix (RHF/RKS type) computed on the GPU.
        """
        mocc = mo_coeff[:,mo_occ>0]
        return cp.dot(mocc*mo_occ[mo_occ>0], mocc.conj().T)
    
    def getOcc_cupy(self, mol=None, energy_mo=None, coeff_mo=None):
        """
        Assign occupation numbers to MOs using CuPy (for GPU-based calculations).

        Parameters
        ----------
        mol : Molecule
            Molecule object to determine number of electrons.

        energy_mo : cp.ndarray
            MO energy array on the GPU.

        coeff_mo : cp.ndarray
            MO coefficients array on the GPU (not used).

        Returns
        -------
        cp.ndarray
            Array of MO occupations (on GPU).
        """
        e_idx = cp.argsort(energy_mo)
        e_sort = energy_mo[e_idx]
        nmo = energy_mo.size
        occ_mo = cp.zeros(nmo)
        nocc = mol.nelectrons // 2
        occ_mo[e_idx[:nocc]] = 2
        return occ_mo
    
    def solve(self, H, S, orthogonalize=False, x=None):
        """
        Solve the generalized or canonical eigenvalue equation.

        Parameters
        ----------
        H : ndarray
            Hamiltonian matrix.

        S : ndarray
            Overlap matrix.

        orthogonalize : bool, optional
            If True, solve using orthogonalized basis.

        x : ndarray, optional
            Transformation matrix (if already computed).

        Returns
        -------
        tuple of (ndarray, ndarray)
            Eigenvalues and eigenvectors of the system.
        """
        if not orthogonalize:
            #Solve the generalized eigenvalue equation HC = SCE
            eigvalues, eigvectors = scipy.linalg.eigh(H, S)
            
        else:
            if x is None:
                eig_val_s, eig_vec_s = scipy.linalg.eigh(S)
                # Removing the eigenvectors assoicated to the smallest eigenvalue.
                x = eig_vec_s[:,eig_val_s>1e-7] / np.sqrt(eig_val_s[eig_val_s>1e-7])
            xHx = x.T @ H @ x
            #Solve the canonical eigenvalue equation HC = CE
            eigvalues, eigvectors = scipy.linalg.eigh(xHx)
            eigvectors = np.dot(x, eigvectors)

        idx = np.argmax(np.abs(eigvectors.real), axis=0)
        eigvectors[:,eigvectors[idx,np.arange(len(eigvalues))].real<0] *= -1
        return eigvalues, eigvectors # E, C
    
    def solve_cupy(self, H, S, orthogonalize=True):
        """
        Solve the generalized eigenvalue problem using CuPy (GPU).

        Parameters
        ----------
        H : cp.ndarray
            Hamiltonian matrix (on GPU).

        S : cp.ndarray
            Overlap matrix (on GPU).

        orthogonalize : bool, optional
            If True, solve using orthogonalized basis.

        Returns
        -------
        tuple of (cp.ndarray, cp.ndarray)
            Eigenvalues and eigenvectors (on GPU).
        """
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
        """
        Compute the core Hamiltonian matrix (T + V).

        Parameters
        ----------
        mol : Molecule, optional
            Molecule object.

        basis : Basis, optional
            Basis set object.

        Returns
        -------
        ndarray
            Core Hamiltonian matrix.
        """
        #Get the core Hamiltonian
        if mol is None:
            mol = self.mol
        if basis is None:
            basis = self.basis
        nao = basis.bfs_nao 
        H = np.empty((nao,nao))
        Vmat = Integrals.nuc_mat_symm(basis, mol)
        Tmat = Integrals.kin_mat_symm(basis)
        H = Vmat + Tmat

        return H


    def guessCoreH(self, mol=None, basis=None, Hcore=None, S=None):
        """
        Generate a guess density matrix using the core Hamiltonian.

        Parameters
        ----------
        mol : Molecule, optional
            Molecule object.

        basis : Basis, optional
            Basis set.

        Hcore : ndarray, optional
            Core Hamiltonian. If None, it will be computed.

        S : ndarray, optional
            Overlap matrix. If None, it will be computed.

        Returns
        -------
        ndarray
            Initial guess density matrix.
        """
        #Get a guess for the density matrix using the core Hamiltonian
        if mol is None:
            mol = self.mol
        if basis is None:
            basis = self.basis
        if Hcore is None:
            Hcore = self.getCoreH(mol, basis)
        if S is None:
            S = Integrals.overlap_mat_symm(basis)

        eigvalues, eigvectors = scipy.linalg.eigh(Hcore, S)
        # print(eigvalues)
        idx = np.argmax(abs(eigvectors.real), axis=0)
        eigvectors[:,eigvectors[idx,np.arange(len(eigvalues))].real<0] *= -1
        mo_occ = self.getOcc(mol, eigvalues, eigvectors)
        # print(mo_occ)
        dmat = self.gen_dm(eigvectors, mo_occ)

        return dmat
    

    def DIIS(self,S,D,F):
        """
        Perform Direct Inversion in the Iterative Subspace (DIIS) to improve SCF convergence.

        Adapted from
        ----------
        McMurchie-Davidson project:
        https://github.com/jjgoings/McMurchie-Davidson
        Licensed under the BSD-3-Clause license

        Parameters
        ----------
        S : ndarray
            Overlap matrix.

        D : ndarray
            Density matrix.

        F : ndarray
            Fock matrix.

        Returns
        -------
        ndarray
            DIIS-extrapolated Fock matrix.
        """
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
        """
        Perform DIIS on GPU using CuPy to accelerate SCF convergence.

        Adapted from
        ----------
        McMurchie-Davidson project:
        https://github.com/jjgoings/McMurchie-Davidson
        Licensed under the BSD-3-Clause license

        Parameters
        ----------
        S : cp.ndarray
            Overlap matrix (on GPU).

        D : cp.ndarray
            Density matrix (on GPU).

        F : cp.ndarray
            Fock matrix (on GPU).

        Returns
        -------
        cp.ndarray
            DIIS-extrapolated Fock matrix (on GPU).
        """
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
    def scf(self):
        """
        Perform Self-Consistent Field (SCF) calculation for Density Functional Theory (DFT).
        
        This method implements a complete DFT SCF procedure including:
        - One-electron integral calculation (overlap, kinetic, nuclear attraction)
        - Two-electron integral calculation (4-center ERIs or density fitting)
        - Grid generation and pruning for exchange-correlation evaluation
        - Iterative SCF cycles with DIIS convergence acceleration
        - Exchange-correlation energy and potential evaluation using LibXC
        - GPU acceleration support for performance-critical operations
        
        The implementation supports multiple algorithmic variants:
        - Density fitting (DF)
        - Schwarz screening for integral sparsity
        - Multiple XC evaluation algorithms (CPU/GPU optimized)
        - Dynamic precision switching for GPU calculations
        
        SCF Procedure:
        1. Initialize one-electron integrals (S, T, V_nuc)
        2. Calculate/prepare two-electron integrals with optional screening
        3. Generate and prune integration grids for XC evaluation
        4. Iterative SCF loop:
        - Build Coulomb matrix J from density matrix
        - Evaluate exchange-correlation energy/potential on grids
        - Form Kohn-Sham matrix: H_KS = H_core + J + V_xc
        - Apply DIIS convergence acceleration
        - Diagonalize KS matrix to get new orbitals
        - Generate new density matrix from occupied orbitals
        - Check energy convergence
        5. Return converged total energy and density matrix
        
        Computational Features:
        - Multi-threading support via Numba and configurable core count
        - GPU acceleration using CuPy for grid-based operations
        - Memory-efficient batched evaluation of XC terms
        - Multiple density fitting algorithms (DF_algo 1-10)
        - Basis function screening for XC evaluation efficiency
        - Optional Cholesky decomposition for 2-center integrals
        
        Returns:
        --------
        tuple[float, numpy.ndarray]
            Etot : float
                Converged total electronic energy in atomic units
            dmat : numpy.ndarray, shape (nbf, nbf)
                Converged density matrix in atomic orbital basis
                
        Raises:
        -------
        ConvergenceError
            If SCF fails to converge within max_itr iterations
            
        Notes:
        ------
        - Uses class attributes for all computational parameters (basis, xc, conv_crit, etc.)
        - Extensive timing and profiling information printed during execution
        - Memory usage information displayed for large arrays
        - GPU memory automatically freed after completion
        - Supports both Cartesian (CAO) and Spherical (SAO) atomic orbital bases
        
        The function performs comprehensive error checking and provides detailed
        timing breakdowns for performance analysis. GPU acceleration is automatically
        enabled when CuPy is available and use_gpu=True.
        
        Example Energy Components:
        - Electron-nuclear attraction energy
        - Nuclear repulsion energy  
        - Kinetic energy
        - Coulomb (electron-electron repulsion) energy
        - Exchange-correlation energy
        """
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
        durationDF_cholesky = 0
        startSCF = timer()    

        mol = self.mol
        basis = self.basis
        dmat = self.dmat
        xc = self.xc
        conv_crit = self.conv_crit
        max_itr = self.max_itr
        ncores = self.ncores
        grids = self.grids
        gridsLevel = self.gridsLevel
        isDF = self.isDF
        auxbasis = self.auxbasis
        rys = self.rys
        DF_algo = self.DF_algo
        blocksize = self.blocksize
        XC_algo = self.XC_algo
        debug = self.debug
        sortGrids = self.sortGrids
        save_ao_values = self.save_ao_values
        xc_bf_screen = self.xc_bf_screen
        threshold_schwarz = self.threshold_schwarz
        strict_schwarz = self.strict_schwarz
        cholesky = self.cholesky
        orthogonalize = self.orthogonalize

        print_pyfock_logo()
        print_scientist()
        print('\n\nNumber of atoms:', mol.natoms)
        print('\n\nNumber of basis functions (atomic orbitals):', basis.bfs_nao)
        print('\n\nNumber of auxiliary basis functions:', auxbasis.bfs_nao)
        print("\n" + "="*70 + "\n")

        print_sys_info()
        print("\n" + "="*70 + "\n")
        #### Set number of cores
        numba.set_num_threads(ncores)
        os.environ['RAYON_NUM_THREADS'] = str(ncores)
        
        print('Running DFT using '+str(numba.get_num_threads())+' threads for Numba.\n\n', flush=True)
        # if basis is None:
        #     basis = self.basis
        # if mol is None:
        #     mol = self.mol
        # if xc is None:
        #     xc = self.xc
        # if gridsLevel is None:
        #     gridsLevel = self.gridsLevel
        # if auxbasis is None:
        #     auxbasis = Basis(mol, {'all':Basis.load(mol=mol, basis_name='def2-universal-jfit')})
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
                print('Threads per block configuration for the XC term: ', threads_per_block, flush=True)
                print('Threads per block configuration for the all other calculations: ', (32, 32), flush=True)
                if self.dynamic_precision:
                    print('\n\nWill use dynamic precision. ')
                    print('This means that the XC term will be evaluated in single precision until the ')
                    print('relative energy difference b/w successive iterations is less than 5.0E-7.')
                    precision_XC = cp.float32
                else:
                    precision_XC = cp.float64

                if XC_algo is None:
                    XC_algo = 3
                if blocksize is None:
                    blocksize = 20480
                streams = []
                nb_streams = []
                for i in range(self.n_gpus):
                    cp.cuda.Device(i).use()
                    cp_stream = cp.cuda.Stream(non_blocking = True)
                    nb_stream = cuda.external_stream(cp_stream.ptr)
                    streams.append(cp_stream)
                    nb_streams.append(nb_stream)
                # Switch back to main GPU
                cp.cuda.Device(0).use()
                streams[0].use()
                # # Set some basis function data as cupy arrays to avoid redoing it during XC term evaluation at every SCF iteration
                # bfs_coords = cp.asarray([basis.bfs_coords], dtype=precision_XC)
                # bfs_contr_prim_norms = cp.asarray([basis.bfs_contr_prim_norms], dtype=precision_XC)
                # bfs_lmn = cp.asarray([basis.bfs_lmn])
                # bfs_nprim = cp.asarray([basis.bfs_nprim])
                # #The remaining properties like bfs_coeffs are a list of lists of unequal sizes.
                # #Numba won't be able to work with these efficiently.
                # #So, we convert them to a numpy 2d array by applying a trick,
                # #that the second dimension is that of the largest list. So that
                # #it can accomadate all the lists.
                # maxnprim = max(basis.bfs_nprim)
                # bfs_coeffs = cp.zeros([basis.bfs_nao, maxnprim], dtype=precision_XC)
                # bfs_expnts = cp.zeros([basis.bfs_nao, maxnprim], dtype=precision_XC)
                # bfs_prim_norms = cp.zeros([basis.bfs_nao, maxnprim], dtype=precision_XC)
                # bfs_radius_cutoff = cp.zeros([basis.bfs_nao], dtype=precision_XC)
                # for i in range(basis.bfs_nao):
                #     for j in range(basis.bfs_nprim[i]):
                #         bfs_coeffs[i,j] = basis.bfs_coeffs[i][j]
                #         bfs_expnts[i,j] = basis.bfs_expnts[i][j]
                #         bfs_prim_norms[i,j] = basis.bfs_prim_norms[i][j]
                #         bfs_radius_cutoff[i] = basis.bfs_radius_cutoff[i]
                # # Now bf/ao values can be evaluated by calling the following
                # # bf_values = Integrals.bf_val_helpers.eval_bfs(bfs_coords[0], bfs_contr_prim_norms[0], bfs_nprim[0], bfs_lmn[0], bfs_coeffs, bfs_prim_norms, bfs_expnts, bfs_radius_cutoff, coord)
                # bfs_data_as_np_arrays = [bfs_coords[0], bfs_contr_prim_norms[0], bfs_nprim[0], bfs_lmn[0], bfs_coeffs, bfs_prim_norms, bfs_expnts, bfs_radius_cutoff]
                

        else:
            if self.use_gpu:
                print('GPU acceleration requested but cannot be enabled as Cupy is not installed.', flush=True)
                self.use_gpu = False

                if XC_algo is None:
                    XC_algo = 2
                if blocksize is None:
                    blocksize = 5000
        if isDF:
            isSchwarz = True
        else:
            isSchwarz = False # Schwarz screening is not yet implemented for 4c2e integrals
        if strict_schwarz:
            if not (DF_algo==6 or DF_algo==10):
                print('Warning: The stricter variation of Schwarz screening is only compatible with DF algo #6 or #10 so turning it off.')
                strict_schwarz = False
        if cholesky:
            if not (DF_algo==6 or DF_algo==10):
                print('Warning: The Cholesky decomposition of 2c2e integrls is only compatible with DF algo #6 or #10 so turning it off.')
                cholesky = False
        
        if self.sao:
            print('\n\nSpherical Atomic Orbitals are being used!\n\n')
            # Get the CAO to SAO transformation matrix
            c2sph_mat = basis.cart2sph_basis() # CAO --> SAO
            # Calculate the pseudoinverse transformation matrix (for back transformation of SAO dmat to CAO dmat)
            sph2c_mat_pseudo = basis.sph2cart_basis() # SAO --> CAO
                

        eigvectors = None
        # DF_algo = 1 # Worst algorithm for more than 500 bfs/auxbfs (requires 2x mem of 3c2e integrals and a large prefactor)
        # DF_algo = 2 # Sligthly better (2x memory efficient) algorithm than above (requires 1x mem of 3c2e integrals and a large prefactor)
        # DF_algo = 3 # Memory effcient without any prefactor. (Can easily be converted into a sparse version, unlike the others) (https://aip.scitation.org/doi/pdf/10.1063/1.1567253)
        # DF_algo = 4 # Same as 3, except now we use triangular version of ints3c2e to save on memory
        # DF_algo = 5 # Same as 4 in terms of memory requirements, however faster in performance due to the use of Schwarz screening.
        # DF_algo = 6 # Much cheaper than 4 and 5 in terms of memory requirements because the indices of significant (ij|P) are efficiently calculated without duplicates/temporary arrays. 
        #               The speed maybe same or just slightly slower. 
        # DF_algo = 7 # The significant indices (ij|P) are stored even more efficiently by using shell indices instead of bf indices.
        # DF_algo = 8 # Similar to 6, except that here the significant indices are not stored resulting in 50% memory savings. The drawback is that it only works in serial which is useful for Google colab or Kaggle perhaps.
        # DF_algo = 9 # 

        if not strict_schwarz: # If a stricter variant of Schwarz screening is not requested
            start1e = timer()
            print('\nCalculating one electron integrals...\n\n', flush=True)
            # One electron integrals
            if not self.use_gpu:
                S = Integrals.overlap_mat_symm(basis)
                V = Integrals.nuc_mat_symm(basis, mol)
                T = Integrals.kin_mat_symm(basis)
                # Core hamiltonian
                H = T + V
            else:
                S = Integrals.overlap_mat_symm_cupy(basis, cp_stream=streams[0])
                V = Integrals.nuc_mat_symm_cupy(basis, mol, cp_stream = streams[0])
                T = Integrals.kin_mat_symm_cupy(basis, cp_stream = streams[0])
                # Core hamiltonian
                H = T + V

            print('Core H size in GB ',H.nbytes/1e9, flush=True)
            print('done!', flush=True)
            duration1e = timer() - start1e
            print('Time taken '+str(round(duration1e, 2))+' seconds.\n', flush=True)
        else:
            start1e = timer()
            print('\nCalculating overlap and kinetic integrals...\n\n', flush=True)
            # One electron integrals
            if not self.use_gpu:
                S = Integrals.overlap_mat_symm(basis)
                T = Integrals.kin_mat_symm(basis)
                # Core hamiltonian
                H = T 
            else:
                S = Integrals.overlap_mat_symm_cupy(basis, cp_stream = streams[0])
                T = Integrals.kin_mat_symm_cupy(basis, cp_stream = streams[0])
                # Core hamiltonian
                H = T 

            print('Core H size in GB ',(H.nbytes/1e9)*2, flush=True) # Factor of 2 because nuclear matrix will also be included here later
            print('done!', flush=True)
            duration1e = timer() - start1e
            print('Time taken '+str(round(duration1e, 2))+' seconds.\n', flush=True)


        if dmat is None:
            if self.dmat_guess_method=='core':
                dmat = self.guessCoreH(mol, basis, Hcore=H, S=S)

        if self.use_gpu:
            dmat_cp = cp.asarray(dmat, dtype=cp.float64)
            streams[0].synchronize()
            cp.cuda.Stream.null.synchronize()

        
        if self.sao:
            # It is possible that the supplied density matrix to the SCF was in SAO format already.
            # In such a case we need to transform this density matrix to CAO basis so that the J and XC term evaluations can be done properly 
            if not dmat.shape==S.shape:
                dmat = np.dot(sph2c_mat_pseudo, np.dot(dmat, sph2c_mat_pseudo.T)) # Convert to CAO from SAO (SAO --> CAO)
                # Later the dmat will be converted back to SAO after J and XC term evaluations
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
                print('Time taken to evaluate the "diagonal" of 4c2e ERI tensor: ', round(duration_4c2e_diag, 2))
                # Calculate the square roots required for 
                duration_square_roots = 0.0
                start_square_roots = timer()
                sqrt_ints4c2e_diag = np.sqrt(np.abs(ints4c2e_diag))
                duration_square_roots = timer() - start_square_roots
                print('Time taken to evaluate the square roots needed: ', round(duration_square_roots, 2))
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
                print('Time for significant indices evaluation: ', round(duration_indices_calc, 2))
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
                print('Total time taken for Schwarz screening '+str(round(durationSchwarz, 2))+' seconds.\n', flush=True)

                ints4c2e = Integrals.schwarz_helpers.rys_4c2e_tri_schwarz_sparse(basis, auxbasis, indicesA, indicesB, indicesC, indicesD)

        else: # Density fitting case (3c2e, and 2c2e will be calculated)
            H_temp, V_temp, ints3c2e, ints2c2e, nsignificant, indicesA, indicesB, indicesC, offsets_3c2e, indices, ints4c2e_diag, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, indices_dmat_tri, indices_dmat_tri_2, df_coeff0, Qpq, cho_decomp_ints2c2e, durationDF_cholesky, durationCoulomb = density_fitting_prelims_for_DFT_development(mol, basis, auxbasis, T, dmat, self.use_gpu, self.keep_ints3c2e_in_gpu, threshold_schwarz, strict_schwarz, rys, DF_algo, cholesky)
            if not H_temp is None:
                H = H_temp
            if not V_temp is None:
                V = V_temp
            if cholesky:
                durationDF += durationDF_cholesky

            
            
        
        
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
            print('Time taken '+str(round(durationgrids, 2))+' seconds.\n', flush=True)

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
            print('Time taken '+str(round(durationgrids_prune_rho, 2))+' seconds.\n', flush=True)
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
            if not self.use_gpu:
                list_nonzero_indices, count_nonzero_indices = Integrals.bf_val_helpers.nonzero_ao_indices(basis, grids.coords, blocksize, nblocks, ngrids)
            else:
                list_nonzero_indices, count_nonzero_indices = Integrals.bf_val_helpers.nonzero_ao_indices_cupy(basis, grids.coords, blocksize, nblocks, ngrids, streams[0])
            print('done!', flush=True)
            durationXCpreprocessing = timer() - startXCpreprocessing
            print('Time taken '+str(round(durationXCpreprocessing, 2))+' seconds.\n', flush=True)
            print('Maximum no. of basis functions contributing to a batch of grid points:   ', max(count_nonzero_indices))
            print('Average no. of basis functions contributing to a batch of grid points:   ', int(np.mean(count_nonzero_indices)))

            
            durationAO_values = 0
            if save_ao_values:
                startAO_values = timer()
                if xc_family_dict[x_family_code]=='LDA' and xc_family_dict[c_family_code]=='LDA':
                    print('\nYou have asked to save the values of significant basis functions on grid points so as to avoid recalculation for each SCF cycle.', flush=True)
                    memory_required = sum(count_nonzero_indices*blocksize)*8/1024/1024/1024
                    print('Please note: This will require addtional memory that is approximately: '+ str(np.round(memory_required,1))+ ' GB', flush=True)
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
                    print('Please note: This will require addtional memory that is approximately: '+ str(np.round(memory_required,1))+ ' GB', flush=True)
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
                print('Time taken '+str(round(durationAO_values, 2))+' seconds.\n', flush=True)
        
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

        startKS = timer()
        if self.sao:
            #########
            # I use a trick to calculate the DFT energy in SAO basis. 
            # The trick is to get rid of the extra information in the CAO basis matrices.
            # The following does just that. 
            # Going from CAO --> SAO we lose the extra information then we go back to from SAO --> CAO
            # This way all the integrals (potential matrices and density matrices) can still be calculated
            # in the CAO basis. In fact even the KS matrix is diagonalized in the CAO basis with the extra information 
            # removed by forward and backward transformation: CAO --> SAO followed by SAO --> CAO.
            # This also helps in reducing the number of transformations needed for calculation of various energy contributions.
            #########
            # Convert the overlap matrix from CAO to SAO basis
            S = np.dot(c2sph_mat, np.dot(S, c2sph_mat.T)) # CAO --> SAO
            # Convert back to SAO so that now we lose the extra information that the CAO basis had
            S = np.dot(sph2c_mat_pseudo, np.dot(S, sph2c_mat_pseudo.T))
        if orthogonalize:
            if not self.use_gpu:
                eig_val_s, eig_vec_s = scipy.linalg.eigh(S)
                # Removing the eigenvectors assoicated to the smallest eigenvalue.
                x = eig_vec_s[:,eig_val_s>1e-7] / np.sqrt(eig_val_s[eig_val_s>1e-7])
        else:
            x = None
        durationKS = durationKS + timer() - startKS
        

        while not (scf_converged or itr==max_itr+1):
            startIter = timer()
            if itr==1:
                dmat_diff = dmat # This is in CAO basis
                # Coulomb (Hartree) matrix
                if not isDF:
                    J = contract('ijkl,ij',ints4c2e,dmat) # This is in CAO basis
                else:
                    J, durationDF, durationDF_coeff, durationDF_gamma, durationDF_Jtri, Ecoul_temp = Jmat_from_density_fitting(dmat, DF_algo, cholesky, cho_decomp_ints2c2e, df_coeff0, Qpq, ints3c2e, ints2c2e, indices_dmat_tri, indices_dmat_tri_2, indicesA, indicesB, indicesC, offsets_3c2e, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold_schwarz, strict_schwarz, basis, auxbasis, self.use_gpu, self.keep_ints3c2e_in_gpu, durationDF_gamma, ncores, durationDF_coeff, durationDF_Jtri, durationDF)


                    
                
            else:
                dmat_diff = dmat-dmat_old
                if not isDF:
                    # J_diff = contract('ijkl,ij',ints4c2e,dmat_diff)
                    # J += J_diff
                    J = contract('ijkl,ij', ints4c2e, dmat)
                else:
                    J, durationDF, durationDF_coeff, durationDF_gamma, durationDF_Jtri, Ecoul_temp = Jmat_from_density_fitting(dmat, DF_algo, cholesky, cho_decomp_ints2c2e, df_coeff0, Qpq, ints3c2e, ints2c2e, indices_dmat_tri, indices_dmat_tri_2, indicesA, indicesB, indicesC, offsets_3c2e, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold_schwarz, strict_schwarz, basis, auxbasis, self.use_gpu, self.keep_ints3c2e_in_gpu, durationDF_gamma, ncores, durationDF_coeff, durationDF_Jtri, durationDF)
                # J += J_diff
            if self.use_gpu:
                J = cp.asarray(J, dtype=cp.float64)
                streams[0].synchronize()
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
                    
                    Exc, Vxc = Integrals.eval_xc_2(basis, dmat, grids.weights, grids.coords, funcid, ncores=ncores, blocksize=blocksize, \
                                                list_nonzero_indices=list_nonzero_indices, count_nonzero_indices=count_nonzero_indices, \
                                                    list_ao_values=list_ao_values, list_ao_grad_values=list_ao_grad_values, debug=debug)
                    
                if XC_algo==3:
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
                                                type=precision_XC, streams=streams, nb_streams=nb_streams)

                Vxc = cp.asarray(Vxc, dtype=cp.float64)

            durationxc = durationxc + timer() - startxc

            
            dmat_old = dmat # This is in CAO basis


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
            if isDF and (DF_algo==6 or DF_algo==10):
                Ecoul = Ecoul*2 - 0.5*Ecoul_temp # This is the correct formula for Coulomb energy with DF

            Etot_new = Exc + Enuc + Ekin + Enn + Ecoul
            self.scf_energies.append(Etot_new)
            self.Total_energy = Etot_new
            self.XC_energy = Exc
            self.Kinetic_energy = Ekin
            self.Nuclear_repulsion_energy = Enn
            self.J_energy = Ecoul
            self.Nuc_energy = Enuc

            # Set label width and numeric format
            label_w = 30
            num_fmt = "{:>20.13f}"  # 20-wide, 10 decimal places

            print(f"\n\n\n------Iteration {itr}--------\n\n", flush=True)
            print("Energies (in Hartrees)\n")
            print(f"{'Electron-Nuclear Energy':<{label_w}}{num_fmt.format(Enuc)}")
            print(f"{'Nuclear repulsion Energy':<{label_w}}{num_fmt.format(Enn)}")
            print(f"{'Kinetic Energy':<{label_w}}{num_fmt.format(Ekin)}")
            print(f"{'Coulomb Energy':<{label_w}}{num_fmt.format(Ecoul)}")
            print(f"{'Exchange-Correlation Energy':<{label_w}}{num_fmt.format(Exc)}")
            print('-' * (label_w + 20))
            print(f"{'Total Energy':<{label_w}}{num_fmt.format(Etot_new)}")
            print('-' * (label_w + 20) + "\n\n\n", flush=True)



            print('Energy difference : ',abs(Etot_new-Etot), flush=True)

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

            

            if not scf_converged:
                KS = H + J + Vxc 
                if self.sao:
                    # The following gets rid of the extra information in the CAO basis KS matrix by going to SAO and then back to CAO.
                    # This way even though the matrix dimensions would be that of CAO but the information would be the same as SAO
                    # leading to the same energy as SAO basis PySCF or TURBOMOLE calculations
                    KS = np.dot(c2sph_mat, np.dot(KS, c2sph_mat.T)) # CAO --> SAO
                    KS = np.dot(sph2c_mat_pseudo, np.dot(KS, sph2c_mat_pseudo.T)) #SAO --> CAO
                    

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
                        streams[0].synchronize()
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
                    streams[0].synchronize()
                    cp.cuda.Stream.null.synchronize()
                    # HOMO-LUMO gap
                    occupied = cp.where(mo_occ > 1e-8)[0]
                    if len(occupied) < len(eigvalues):
                        homo_idx = occupied[-1]
                        lumo_idx = homo_idx + 1
                        gap = (eigvalues[lumo_idx] - eigvalues[homo_idx])
                        print(f"\n\nHOMO-LUMO gap: {gap} au", flush=True)
                        print(f"HOMO-LUMO gap: {gap*27.211324570273:.3f} eV", flush=True)
                else:
                    with threadpool_limits(limits=ncores, user_api='blas'):
                        # print('KS eigh', controller.info())
                        eigvalues, eigvectors = self.solve(KS, S, orthogonalize=orthogonalize, x=x)
                    mo_occ = self.getOcc(mol, eigvalues, eigvectors)
                    dmat = self.gen_dm(eigvectors, mo_occ)
                    # HOMO-LUMO gap
                    occupied = np.where(mo_occ > 1e-8)[0]
                    if len(occupied) < len(eigvalues):
                        homo_idx = occupied[-1]
                        lumo_idx = homo_idx + 1
                        gap = (eigvalues[lumo_idx] - eigvalues[homo_idx]) 
                        print(f"\n\nHOMO-LUMO gap: {gap} au", flush=True)
                        print(f"HOMO-LUMO gap: {gap*27.211324570273:.3f} eV", flush=True)
                durationKS = durationKS + timer() - startKS

                self.dmat = dmat
                self.mo_coefficients = eigvectors
                self.mo_energies = eigvalues
                self.mo_occupations = mo_occ

                # Check when to switch to double precision for XC
                if self.use_gpu:
                    if precision_XC is cp.float32:
                        if abs(Etot_new-Etot)/abs(Etot_new)<5e-7:
                            precision_XC = cp.float64
                            print('\nSwitching to double precision for XC evaluation after '+str(itr) +' iterations!', flush=True)
                            


            durationItr = timer() - startIter
            print('\n\nTime taken for the previous iteration: '+str(round(durationItr, 2))+' seconds \n\n', flush=True)

        
        self.converged = scf_converged
        self.niter = itr-1
        

        durationSCF = timer() - startSCF
        # print(dmat)
        print('\nTime taken : '+str(round(durationSCF, 2)) +' seconds.', flush=True)
        print('\n\n', flush=True)
        print('-------------------------------------------------------------', flush=True)
        print('Profiling (Wall times in seconds)', flush=True)
        print('-------------------------------------------------------------', flush=True)
        print('Preprocessing                          ', round(durationXCpreprocessing + durationAO_values + durationgrids_prune_rho + durationSchwarz, 2), flush=True)
        if isDF:
            print('Density Fitting                        ', round(durationDF, 2), flush=True)
            if DF_algo==6 or DF_algo==10:
                print('    DF (gamma)                         ', round(durationDF_gamma, 2), flush=True)
                print('    DF (coeff)                         ', round(durationDF_coeff, 2), flush=True)
                print('    DF (Jtri)                          ', round(durationDF_Jtri, 2), flush=True)
                if cholesky:
                    print('    DF (Cholesky)                      ', round(durationDF_cholesky, 2), flush=True)
        print('DIIS                                   ', round(durationDIIS, 2), flush=True)
        print('KS matrix diagonalization              ', round(durationKS, 2), flush=True)
        print('One electron Integrals (S, T, Vnuc)    ', round(duration1e, 2), flush=True)
        if isDF:
            print('Coulomb Integrals (2c2e + 3c2e)        ', round(durationCoulomb-durationSchwarz-durationDF_cholesky, 2), flush=True)
        if not isDF:
            print('Coulomb Integrals (4c2e)               ', round(durationCoulomb-durationSchwarz, 2), flush=True)
        print('Grids construction                     ', round(durationgrids, 2), flush=True)
        print('Exchange-Correlation Term              ', round(durationxc, 2), flush=True)
        totalTime = round(durationXCpreprocessing + durationAO_values + duration1e + durationCoulomb - durationDF_cholesky + \
            durationgrids + durationxc + durationDF + durationKS + durationDIIS + durationgrids_prune_rho, 2)
        print('Misc.                                  ', round(durationSCF - totalTime, 2), flush=True)
        print('Complete SCF                           ', round(durationSCF, 2), flush=True)

        if self.use_gpu:
            # Free memory of all GPUs
            for igpu in range(self.n_gpus):
                cp.cuda.Device(igpu).use()
                cp._default_memory_pool.free_all_blocks()

            # Switch back to main GPU
            cp.cuda.Device(0).use()
        return Etot, dmat
        

    

    

        

        

