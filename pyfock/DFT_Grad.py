import copy

import numpy as np
import scipy
import numba
from timeit import default_timer as timer

from opt_einsum import contract
from threadpoolctl import threadpool_limits

import pyfock.Integrals as Integrals
from pyfock import XC
from pyfock import Data
from pyfock.Basis import Basis
from pyfock.Mol import Mol

try:
    import cupy as cp
except Exception:                                  # pragma: no cover - CPU-only install
    cp = None


def _to_host(array):
    """NumPy view of an array that may be on the device after a ``use_gpu`` SCF."""
    if array is None or isinstance(array, np.ndarray):
        return array
    getter = getattr(array, 'get', None)   # CuPy's own transfer; np.asarray refuses a device array
    return np.asarray(getter() if getter is not None else array)


def _contract_grad_r_gpu(dX_r, mat):
    """contract('dij,ij->id', dX_r, mat) for a device ``(3, nbf, nbf)`` bra-center r-gradient.

    Done one Cartesian direction at a time so the elementwise temporary is (nbf, nbf) rather than
    another copy of the whole tensor. The result is (nbf, 3), so it comes back to the host for the
    per-atom scatter; the device tensor is freed here rather than at the end of the caller's scope.
    """
    mat_d = cp.asarray(mat)
    out = cp.empty((dX_r.shape[1], 3))
    for d in range(3):
        out[:, d] = cp.sum(dX_r[d] * mat_d, axis=1)
    out = cp.asnumpy(out)
    del dX_r
    return out


class DFT_Grad:
    """
    Analytical nuclear gradients (and forces) for converged PyFock DFT
    calculations with density fitting.

    The total gradient is assembled as

        dE/dR = dE_nn/dR                              (nuclear repulsion)
              + sum_ij D_ij dT_ij/dR                  (kinetic)
              + sum_ij D_ij dV_ij/dR                  (nuclear attraction,
                                                       incl. Hellmann-Feynman)
              - sum_ij W_ij dS_ij/dR                  (Pulay / overlap)
              + sum_ijP D_ij c_P d(ij|P)/dR
              - 0.5 sum_PQ c_P c_Q d(P|Q)/dR          (DF Coulomb)
              + sum_ij D_ij dV_ecp_ij/dR              (ECP, if present)
              + dExc/dR                               (XC, fixed grid)

    where W is the energy-weighted density matrix and c_P are the density
    fitting coefficients of the converged density. The grid-weight response
    of the XC term is neglected (same approximation as PySCF's default).

    When effective core potentials (ECPs) are present, ``mol.Zcharges`` already
    holds the reduced (Z - n_core) charges, so the nuclear-repulsion and
    nuclear-attraction gradient terms above are automatically consistent. The
    extra ECP energy term ``Tr(D V_ecp)`` contributes ``sum_ij D_ij
    dV_ecp_ij/dR``, evaluated analytically (``ecp_grad_mode='analytical'``,
    default) by differentiating the series ECP integrals via angular-momentum
    shifts (local part) and derivative moment arrays (projector part), with the
    ECP-center term obtained from translational invariance. A finite-difference
    fallback (``ecp_grad_mode='fd'``) differentiates the ECP integral matrix
    directly. Both give the exact derivative of the ECP energy at fixed D; the
    orbital response is already captured by the energy-weighted W term.

    Currently supported: restricted KS-DFT with density fitting (the DF
    gradient corresponds to the robust-fit Coulomb energy used by all DF
    algorithms), LDA, GGA and meta-GGA (tau-dependent) functionals via either
    the native PyFock functionals or pylibxc, Skala, and ECPs.
    Laplacian-dependent meta-GGAs are not yet supported.

    With ``use_gpu=True`` every term above except the ECP one is evaluated on
    the GPU by a device port of the corresponding CPU routine; the results
    agree to round-off (see ``benchmarks_tests/benchmark_DFT_gradients_gpu.py``).
    The default follows the SCF, so a GPU SCF is followed by a GPU gradient.

    Parameters
    ----------
    dft_obj : DFT
        A converged PyFock DFT object (after ``dft_obj.scf()``).
    threshold_schwarz_grad : float, optional
        Screening threshold used for the contracted 3c2e derivative
        integrals (includes density/coefficient weighting).
    use_gpu : bool, optional
        Evaluate the gradient on the GPU. ``None`` (default) inherits
        ``dft_obj.use_gpu``. The ECP term has no device implementation and
        stays on the CPU.
    ecp_grad_mode : {'analytical', 'fd'}, optional
        How to evaluate the ECP gradient term. 'analytical' (default)
        differentiates the series ECP integrals; 'fd' differentiates the ECP
        integral matrix by finite differences. Both are consistent with the
        SCF energy.
    ecp_series_order : int, optional
        Power-series order for the analytical ECP gradient. Should match the
        order used by the SCF energy (``ecp_mat_symm`` default = 12) so the
        force is consistent with the energy. Default 12.
    ecp_fd_step : float, optional
        Finite-difference step (in Bohr) for the 'fd' ECP gradient mode.
        Default 1e-3.
    verbose : bool, optional
        Print timing information.
    """

    def __init__(self, dft_obj, threshold_schwarz_grad=1e-11, ecp_grad_mode='analytical',
                 ecp_series_order=12, ecp_fd_step=1e-3, verbose=True, grid_response=None,
                 use_gpu=None):
        if dft_obj is None:
            raise ValueError('ERROR: A PyFock DFT object is required.')
        if not getattr(dft_obj, 'converged', False):
            raise ValueError('ERROR: The supplied DFT object must already be converged.')
        # The converged quantities the gradient reads -- the density matrix, the MOs and the grid --
        # are brought back to the host whatever the SCF ran on. The device routines want them packed
        # their own way anyway, and these are small next to the work done with them.
        self.grids = dft_obj.grids
        if dft_obj.use_gpu:
            self.grids = copy.copy(dft_obj.grids)   # shallow: only the arrays below are replaced
            for name in ('coords', 'weights', 'atomic_weights', 'atom_idx'):
                setattr(self.grids, name, _to_host(getattr(dft_obj.grids, name, None)))

        if use_gpu is None:
            use_gpu = bool(getattr(dft_obj, 'use_gpu', False))
        if use_gpu and cp is None:
            raise RuntimeError('use_gpu=True was requested but CuPy is not available.')
        self.use_gpu = bool(use_gpu)
        self._cp_stream = None      # set by calculate() for the duration of a device gradient
        if not dft_obj.isDF:
            raise NotImplementedError('Analytical gradients are currently implemented for density-fitted (isDF=True) calculations only.')
        if dft_obj.xc == 'HF' or getattr(dft_obj, 'exx_coef', 0.0) > 0:
            raise NotImplementedError('Analytical gradients are currently implemented for pure DFT functionals only (no exact exchange).')
        if ecp_grad_mode not in ('analytical', 'fd'):
            raise ValueError("ecp_grad_mode must be 'analytical' or 'fd'.")

        self.dft_obj = dft_obj
        self.threshold_schwarz_grad = threshold_schwarz_grad
        self.ecp_grad_mode = ecp_grad_mode
        self.ecp_series_order = ecp_series_order
        self.ecp_fd_step = ecp_fd_step
        self.verbose = verbose

        # Grid response: differentiate the quadrature grid's own dependence on the nuclei (the points
        # of an atom move with it, and the Becke weights depend on every nuclear position). Off by
        # default for semilocal functionals, matching PySCF and PyFock's published references, but
        # mandatory for Skala, whose features are integrals over each atomic grid.
        self.grid_response = grid_response

        # Resolve the functional specification the same way DFT.scf does. Skala is not a LibXC
        # functional: the converged DFT object already holds the loaded model, and the gradient goes
        # through its own driver.
        self.skala = getattr(dft_obj, 'skala', None)
        # the model's peak memory scales with the points per call; use the SCF's setting
        self.skala_max_points_per_chunk = int(getattr(dft_obj, 'skala_max_points_per_chunk', 250000))
        if self.grid_response is None:
            self.grid_response = self.skala is not None
        if self.skala is not None and not self.grid_response:
            raise ValueError(
                'grid_response=False is not usable with Skala: on a frozen grid its gradient is wrong '
                'by ~1e-2 Ha/Bohr, the size of the forces themselves. Pass grid_response=True (the '
                'default for Skala) or use DFT_NumGrad.')
        if self.grid_response and getattr(self.grids, 'atomic_weights', None) is None:
            raise ValueError(
                "grid_response=True needs the unpartitioned single-atom quadrature weights, which the "
                "'numgrid' grid scheme does not expose. Build the grid with the native scheme, e.g. "
                "Grids(mol, level=3).")
        xc = dft_obj.xc
        if self.skala is not None:
            self.funcid = xc
        else:
            if isinstance(xc, list):
                if all(isinstance(v, str) for v in xc):
                    xc = [XC.get_functional_id(name) for name in xc]
            elif isinstance(xc, str):
                xc = XC.resolve_functional(xc)
            self.funcid = xc

    def _energy_weighted_dmat(self):
        """Energy-weighted density matrix W in the CAO basis."""
        dft_obj = self.dft_obj
        mo_coeff = _to_host(dft_obj.mo_coefficients)
        mo_energy = _to_host(dft_obj.mo_energies)
        mo_occ = _to_host(dft_obj.mo_occupations)
        if mo_coeff is None or mo_energy is None or mo_occ is None:
            raise ValueError('The converged DFT object must contain MO coefficients, energies and occupations.')
        mo_occ = np.asarray(mo_occ)
        mo_energy = np.asarray(mo_energy)
        occupied = mo_occ > 0
        mocc = mo_coeff[:, occupied]
        W = (mocc * (mo_occ[occupied] * mo_energy[occupied])) @ mocc.T
        if dft_obj.sao:
            # The stored MOs are in the SAO basis; transform W to CAO the same
            # way the density matrix is transformed (W_cao = T^T W_sao T).
            W = dft_obj.basis.sph2cart_dmat_blockwise(W)
        return W

    def _nuclear_repulsion_grad(self):
        mol = self.dft_obj.mol
        coords = np.asarray(mol.coordsBohrs, dtype=np.float64)
        Z = np.asarray(mol.Zcharges, dtype=np.float64)
        grad = np.zeros((mol.natoms, 3))
        for i in range(mol.natoms):
            for j in range(mol.natoms):
                if i == j:
                    continue
                rij = coords[i] - coords[j]
                dist = np.sqrt(np.sum(rij**2))
                grad[i] -= Z[i] * Z[j] * rij / dist**3
        return grad

    def _ecp_matrix_at(self, coords_angstrom):
        """Build the ECP integral matrix (CAO basis) at a displaced geometry."""
        dft_obj = self.dft_obj
        atoms = []
        for iatom, symbol in enumerate(dft_obj.mol.atomicSpecies):
            x, y, z = coords_angstrom[iatom]
            atoms.append([symbol, float(x), float(y), float(z)])
        mol = Mol(atoms=atoms, charge=dft_obj.mol.charge)
        basis = Basis(mol, copy.deepcopy(dft_obj.basis.basis))
        return Integrals.ecp_mat_symm(basis)

    def _ecp_grad(self, dmat):
        """
        ECP contribution to the gradient: G[A,d] = sum_ij D_ij dV_ecp_ij/dR_{A,d}.

        Evaluated by central finite differences of the ECP integral matrix
        (cheap, one-electron) at the fixed converged density. Displacing an
        atom moves both its basis functions and its ECP operator center, so the
        recomputed ECP matrix captures every contribution to the ECP energy
        derivative.
        """
        mol = self.dft_obj.mol
        natoms = mol.natoms
        step_bohr = self.ecp_fd_step
        step_ang = step_bohr / Data.Angs2BohrFactor  # displace coordinates (Angstrom)
        coords0 = np.asarray(mol.coords, dtype=np.float64)

        grad = np.zeros((natoms, 3))
        for iatom in range(natoms):
            for icart in range(3):
                coords_plus = coords0.copy()
                coords_plus[iatom, icart] += step_ang
                coords_minus = coords0.copy()
                coords_minus[iatom, icart] -= step_ang
                Vp = self._ecp_matrix_at(coords_plus)
                Vm = self._ecp_matrix_at(coords_minus)
                # derivative w.r.t. Bohr -> forces in Ha/Bohr
                grad[iatom, icart] = np.sum(dmat * (Vp - Vm)) / (2.0 * step_bohr)
        return grad

    def calculate(self):
        """
        Calculate analytical gradients and forces.

        Returns
        -------
        dict
            Dictionary with `energy`, `gradient` (natoms, 3) in Ha/Bohr,
            `forces` (= -gradient) and per-term `timings`.
        """
        if not self.use_gpu:
            self._cp_stream = None
            return self._calculate()
        # One stream for the whole gradient, made current for its duration. Several of the CuPy
        # integral routines pack their basis arrays with the stream that is current on entry and
        # only then create the one they launch on, which is a race unless the two are the same
        # stream; see Integrals.cuda_stream for the details.
        cp_stream, _ = Integrals.cuda_stream.gradient_stream()
        self._cp_stream = cp_stream
        with cp_stream:
            result = self._calculate()
        cp_stream.synchronize()
        return result

    def _calculate(self):
        dft_obj = self.dft_obj
        mol = dft_obj.mol
        basis = dft_obj.basis
        auxbasis = dft_obj.auxbasis
        ncores = dft_obj.ncores
        natoms = mol.natoms

        numba.set_num_threads(ncores)

        dmat = np.ascontiguousarray(_to_host(dft_obj.dmat), dtype=np.float64)
        bfs_atoms = np.asarray(basis.bfs_atoms, dtype=np.int64)
        use_gpu = self.use_gpu

        timings = {}

        # Schwarz diagonals, shared by the fitting-coefficient, 3c2e-derivative and
        # nuclear-attraction steps. Each device routine would otherwise rebuild them.
        sqrt_ints4c2e_diag = sqrt_diag_ints2c2e = None
        if use_gpu:
            start = timer()
            sqrt_ints4c2e_diag = np.sqrt(np.abs(Integrals.schwarz_helpers.eri_4c2e_diag(basis)))
            sqrt_diag_ints2c2e = np.sqrt(np.abs(Integrals.rys_2c2e_diag(auxbasis)))
            timings['schwarz_diagonals'] = timer() - start

        # ---------------- Nuclear repulsion ----------------
        start = timer()
        grad_nn = self._nuclear_repulsion_grad()
        timings['nuclear_repulsion'] = timer() - start

        # ---------------- Kinetic (Pulay-type) ----------------
        # dT_r[d, i, j] = dT_ij / d(center of bf i)
        start = timer()
        if use_gpu:
            tmpT = _contract_grad_r_gpu(
                Integrals.kin_mat_grad_r_symm_cupy(basis, cp_stream=self._cp_stream), dmat)
        else:
            dT_r = Integrals.kin_mat_grad_r_symm(basis)
            tmpT = contract('dij,ij->id', dT_r, dmat)
            dT_r = None
        grad_T = np.zeros((natoms, 3))
        np.add.at(grad_T, bfs_atoms, 2.0 * tmpT)
        timings['kinetic'] = timer() - start

        # ---------------- Nuclear attraction ----------------
        # Full derivative incl. operator (Hellmann-Feynman) contributions,
        # contracted with the density matrix on the fly.
        start = timer()
        if use_gpu:
            grad_V = Integrals.rys_nuc_grad_contract_cupy(
                basis, mol, dmat, sqrt_ints4c2e_diag=sqrt_ints4c2e_diag,
                cp_stream=self._cp_stream)
        else:
            grad_V = Integrals.rys_nuc_grad_contract(basis, mol, dmat, ncores=ncores)
        timings['nuclear_attraction'] = timer() - start

        # ---------------- Overlap (Pulay) ----------------
        start = timer()
        W = self._energy_weighted_dmat()
        if use_gpu:
            tmpS = _contract_grad_r_gpu(
                Integrals.overlap_mat_grad_r_symm_cupy(basis, cp_stream=self._cp_stream), W)
        else:
            dS_r = Integrals.overlap_mat_grad_r_symm(basis)
            tmpS = contract('dij,ij->id', dS_r, W)
            dS_r = None
        grad_S = np.zeros((natoms, 3))
        np.add.at(grad_S, bfs_atoms, -2.0 * tmpS)
        timings['overlap'] = timer() - start

        # ---------------- DF Coulomb ----------------
        # gamma_P = sum_ij D_ij (ij|P);  c = (P|Q)^-1 gamma
        # The 3c2e tensor is only needed transiently for gamma, so on the CPU it is evaluated in
        # chunks over the auxiliary dimension to bound memory; the device kernel contracts it as it
        # goes and never forms it.
        start = timer()
        nbf = basis.bfs_nao
        naux = auxbasis.bfs_nao
        if use_gpu:
            ints2c2e = Integrals.rys_2c2e_symm_cupy(auxbasis, cp_stream=self._cp_stream)
            if dft_obj.sao:
                # The spherical transform below is host code, and the metric is only naux x naux.
                ints2c2e = cp.asnumpy(ints2c2e)
            gamma = Integrals.rys_3c2e_gamma_contract_cupy(
                basis, auxbasis, dmat,
                threshold_schwarz=min(dft_obj.threshold_schwarz, 1e-9),
                sqrt_ints4c2e_diag=sqrt_ints4c2e_diag,
                sqrt_diag_ints2c2e=sqrt_diag_ints2c2e, cp_stream=self._cp_stream)
        else:
            ints2c2e = Integrals.rys_2c2e_symm(auxbasis)
            max_chunk_bytes = 1e9
            chunk_naux = max(1, min(naux, int(max_chunk_bytes / (nbf * nbf * 8))))
            gamma = np.zeros(naux)
            with threadpool_limits(limits=ncores, user_api='blas'):
                for c0 in range(0, naux, chunk_naux):
                    c1 = min(c0 + chunk_naux, naux)
                    ints3c2e_chunk = Integrals.rys_3c2e_symm(
                        basis, auxbasis, slice=[0, nbf, 0, nbf, c0, c1],
                        schwarz=True,
                        threshold_schwarz=min(dft_obj.threshold_schwarz, 1e-9),
                    )
                    gamma[c0:c1] = contract('ijP,ij->P', ints3c2e_chunk, dmat)
                    ints3c2e_chunk = None
        with threadpool_limits(limits=ncores, user_api='blas'):
            if dft_obj.sao:
                # With SAOs the SCF performs the density fitting in the
                # spherical auxiliary space. The effective Cartesian
                # coefficients are c_eff = T^T (T C T^T)^-1 T gamma, and the
                # usual gradient formula holds since T is geometry
                # independent.
                c2sph_aux = auxbasis.cart2sph_basis()
                ints2c2e_sph = auxbasis.cart2sph_operator_blockwise(ints2c2e)
                gamma_sph = c2sph_aux @ gamma
                c_sph = scipy.linalg.solve(ints2c2e_sph, gamma_sph, assume_a='pos')
                df_coeff = c2sph_aux.T @ c_sph
            elif use_gpu:
                # cuSOLVER's Cholesky solve, so the naux x naux metric never leaves the device.
                df_coeff = cp.asnumpy(cp.linalg.solve(ints2c2e, cp.asarray(gamma)))
            else:
                df_coeff = scipy.linalg.solve(ints2c2e, gamma, assume_a='pos')
        ints2c2e = None
        timings['df_coefficients'] = timer() - start

        start = timer()
        if use_gpu:
            grad_J3c = Integrals.rys_3c2e_grad_contract_cupy(
                basis, auxbasis, dmat, df_coeff,
                schwarz=True, threshold_schwarz=self.threshold_schwarz_grad,
                sqrt_ints4c2e_diag=sqrt_ints4c2e_diag,
                sqrt_diag_ints2c2e=sqrt_diag_ints2c2e, cp_stream=self._cp_stream,
            )
        else:
            grad_J3c = Integrals.rys_3c2e_grad_contract(
                basis, auxbasis, dmat, df_coeff,
                schwarz=True, threshold_schwarz=self.threshold_schwarz_grad,
                ncores=ncores,
            )
        timings['coulomb_3c2e_grad'] = timer() - start

        start = timer()
        if use_gpu:
            grad_J2c = Integrals.rys_2c2e_grad_contract_cupy(auxbasis, df_coeff,
                                                             cp_stream=self._cp_stream)
        else:
            grad_J2c = Integrals.rys_2c2e_grad_contract(auxbasis, df_coeff, ncores=ncores)
        grad_J = grad_J3c - 0.5 * grad_J2c
        timings['coulomb_2c2e_grad'] = timer() - start

        # ---------------- XC ----------------
        start = timer()
        grids = self.grids
        coords_grid = np.asarray(grids.coords)
        weights_grid = np.asarray(grids.weights)
        ngrids = coords_grid.shape[0]
        blocksize = dft_obj.blocksize if dft_obj.blocksize is not None else 5000
        nblocks = ngrids // blocksize

        list_nonzero_indices = None
        count_nonzero_indices = None
        if dft_obj.xc_bf_screen:
            list_nonzero_indices, count_nonzero_indices = Integrals.bf_val_helpers.nonzero_ao_indices(
                basis, coords_grid, blocksize, nblocks, ngrids)

        explicit_xc_grad = None
        if self.skala is not None:
            # Skala also depends on the nuclear positions explicitly, through the grid geometry it
            # reads; that part comes back already resolved per atom (see eval_xc_grad_skala).
            skala_grad = (Integrals.eval_xc_grad_skala_cupy if use_gpu
                          else Integrals.eval_xc_grad_skala)
            skala_kwargs = {} if not use_gpu else {'cp_stream': self._cp_stream}
            dexc_dbf, explicit_xc_grad = skala_grad(
                basis, dmat, self.grids, self.skala, ncores=ncores, blocksize=blocksize,
                list_nonzero_indices=list_nonzero_indices,
                count_nonzero_indices=count_nonzero_indices,
                max_points_per_chunk=self.skala_max_points_per_chunk, **skala_kwargs,
            )
        elif use_gpu:
            xc_result = Integrals.eval_xc_grad_2_cupy(
                basis, dmat, weights_grid, coords_grid, funcid=self.funcid,
                use_libxc=dft_obj.use_libxc, blocksize=blocksize,
                list_nonzero_indices=list_nonzero_indices,
                count_nonzero_indices=count_nonzero_indices,
                grids=self.grids, grid_response=self.grid_response,
                cp_stream=self._cp_stream,
            )
            dexc_dbf, explicit_xc_grad = xc_result if self.grid_response else (xc_result, None)
        else:
            xc_result = Integrals.eval_xc_grad_2(
                basis, dmat, weights_grid, coords_grid, funcid=self.funcid,
                use_libxc=dft_obj.use_libxc, ncores=ncores, blocksize=blocksize,
                list_nonzero_indices=list_nonzero_indices,
                count_nonzero_indices=count_nonzero_indices,
                grids=self.grids, grid_response=self.grid_response,
            )
            # eval_xc_grad_2 only returns the per-atom grid-response term when it was asked for.
            if self.grid_response:
                dexc_dbf, explicit_xc_grad = xc_result
            else:
                dexc_dbf = xc_result
        grad_xc = np.zeros((natoms, 3))
        np.add.at(grad_xc, bfs_atoms, -2.0 * dexc_dbf.T)
        if explicit_xc_grad is not None:
            grad_xc += explicit_xc_grad
        timings['xc'] = timer() - start

        # ---------------- ECP (if present) ----------------
        grad_ecp = np.zeros((natoms, 3))
        if getattr(basis, 'has_ecp', False):
            start = timer()
            if self.ecp_grad_mode == 'analytical':
                grad_ecp = Integrals.ecp_grad_contract(
                    basis, mol, dmat, series_order=self.ecp_series_order)
            else:
                grad_ecp = self._ecp_grad(dmat)
            timings['ecp'] = timer() - start

        gradient = grad_nn + grad_T + grad_V + grad_S + grad_J + grad_xc + grad_ecp
        forces = -gradient

        if self.verbose:
            label_w = 28
            print('\n---------------------------------------------------------')
            print('Analytical gradient timings (seconds)')
            print('---------------------------------------------------------')
            for key, value in timings.items():
                print(f'{key:<{label_w}}{value:>12.3f}')
            print(f'{"total":<{label_w}}{sum(timings.values()):>12.3f}')
            print('---------------------------------------------------------\n')

        return {
            'energy': float(dft_obj.Total_energy),
            'gradient': gradient,
            'forces': forces,
            'gradient_components': {
                'nuclear_repulsion': grad_nn,
                'kinetic': grad_T,
                'nuclear_attraction': grad_V,
                'overlap_pulay': grad_S,
                'coulomb_df': grad_J,
                'xc': grad_xc,
                'ecp': grad_ecp,
            },
            'timings': timings,
        }
