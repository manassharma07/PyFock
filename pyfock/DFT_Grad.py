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
from pyfock import Dispersion
from pyfock.Basis import Basis
from pyfock.Mol import Mol
from pyfock.DFT_Helper_Coulomb import _pseudo_cartesian_metric_diagonal
from pyfock.Integrals.df_algo10_helpers import STRICT_PAIR_CUTOFF

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
    Analytical nuclear gradients (and forces) for converged PyFock DFT and HF
    calculations, with or without density fitting.

    The total gradient is assembled as

        dE/dR = dE_nn/dR                              (nuclear repulsion)
              + sum_ij D_ij dT_ij/dR                  (kinetic)
              + sum_ij D_ij dV_ij/dR                  (nuclear attraction,
                                                       incl. Hellmann-Feynman)
              - sum_ij W_ij dS_ij/dR                  (Pulay / overlap)
              + sum_ijP D_ij c_P d(ij|P)/dR
              - 0.5 sum_PQ c_P c_Q d(P|Q)/dR          (DF Coulomb)
              - (a/2) sum_ijP Gamma^P_ij d(ij|P)/dR
              + (a/4) sum_PQ W_PQ d(P|Q)/dR           (RI exact exchange, HF and hybrids)
              + sum_ij D_ij dV_ecp_ij/dR              (ECP, if present)
              + dExc/dR                               (XC, fixed grid)
              + dE_disp/dR                            (DFT-D3, if the SCF applied it)

    where W is the energy-weighted density matrix and c_P are the density
    fitting coefficients of the converged density. The grid-weight response
    of the XC term is neglected (same approximation as PySCF's default).

    The exchange terms are those of the RI exchange energy the SCF computes,
    ``E_K = -(a/4) sum D_ij D_kl (ik|P) [(P|Q)^-1] (Q|jl)`` with ``a`` the
    fraction of exact exchange (1 for HF). With the occupied factor ``X`` of the
    density (``D = X X^T``) and the fitting coefficients ``c^P_ij`` of the pair
    densities, ``Y_P = X^T c^P X``, ``Gamma^P = X Y_P X^T = D c^P D`` and
    ``W_PQ = <Y_P, Y_Q>``. The three-center integrals are evaluated once more,
    with the SCF's DF_algo=11 screening, and contracted to ``Y``; the
    derivative integrals are never stored. By default one derivative pass
    serves the Coulomb and the exchange terms (weights ``D_ij c_P - (a/2)
    Gamma^P_ij``), reported together as ``coulomb_exchange_df``.

    The D3 term is present whenever the SCF added a dispersion correction
    (``DFT(..., dispersion=...)``, e.g. ``dispersion=True`` for Skala), with
    the parametrisation it used, so the gradient is that of the corrected
    energy ``scf()`` returns. D3 depends on the nuclear positions alone, so
    the term is exact; simple-dftd3 evaluates it on the CPU in either mode.

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

    Without density fitting (``isDF=False``) the Coulomb and exchange terms are
    replaced by ``sum Gamma_abcd d(ab|cd)/dR`` of the two-electron energy
    ``1/2 sum D_ab D_cd (ab|cd) - (a/4) sum D_ac D_bd (ab|cd)``, from the
    four-center derivative integrals contracted on the fly
    (:mod:`pyfock.Integrals.jk_4c2e_grad`, screened with
    ``threshold_schwarz_grad``), whichever of the direct, screened-store or
    complete-store modes the SCF used; reported as ``coulomb_exchange_4c2e``
    (``coulomb_4c2e`` for a pure functional).

    Currently supported: restricted KS-DFT and Hartree-Fock with or without
    density fitting (the DF gradient corresponds to the robust-fit Coulomb
    energy used by all DF algorithms), LDA, GGA and meta-GGA (tau-dependent)
    functionals via either the native PyFock functionals or pylibxc, global
    hybrids (exact exchange on the CPU), Skala, and ECPs. Laplacian-dependent
    meta-GGAs are not yet supported.

    The DF Coulomb terms follow the SCF's density-fitting algorithm (``DF_algo``).
    With the default, 12, the three-center derivative integrals are split exactly
    as the SCF split the integrals: the near field by a shell-pair-blocked Rys
    derivative kernel, the far field by differentiating the multipole expansions
    (:mod:`pyfock.Integrals.df_algo12_grad`), so the gradient is that of the
    DF_algo=12 energy. The fitting coefficients come from the SCF's last
    iteration, which fitted the converged density itself, instead of a second
    three-center pass. The SCF's screening is part of that energy: with
    ``strict_schwarz`` the function pairs it leaves out of the nuclear
    attraction and Coulomb matrices are left out of both derivatives.
    ``DF_algo=11`` is the same without the far field, and ``DF_algo=10`` the
    previous implementation (every significant derivative integral, including
    those of the pairs the strict cut-off drops, and the fitting redone from a
    full three-center pass).

    With ``use_gpu=True`` every term above except the ECP one is evaluated on
    the GPU by a device port of the corresponding CPU routine; the results
    agree to round-off (see ``benchmarks_tests/benchmark_DFT_gradients_gpu.py``).
    The default follows the SCF, so a GPU SCF is followed by a GPU gradient,
    except with exact exchange or without density fitting, whose gradients are
    evaluated on the CPU.

    Parameters
    ----------
    dft_obj : DFT
        A converged PyFock DFT object (after ``dft_obj.scf()``).
    threshold_schwarz_grad : float, optional
        Screening threshold used for the contracted 3c2e derivative
        integrals (includes density/coefficient weighting), and for the 4c2e
        ones without density fitting.
    DF_algo : {12, 11, 10} or None, optional
        Algorithm for the DF Coulomb terms (see above). ``None`` (default) gives
        11 when the SCF ran with ``DF_algo=11`` and 12 otherwise. 12 and 11 screen
        with the SCF's ``threshold_schwarz``, ``strict_schwarz`` and
        ``multipole_options``. With exact exchange only 11 is available (the
        SCF itself falls back to it from 12), and ``None`` gives 11.
    use_gpu : bool, optional
        Evaluate the gradient on the GPU. ``None`` (default) inherits
        ``dft_obj.use_gpu`` (``False`` with exact exchange or without density
        fitting, which have no device implementation yet). The ECP term has no
        device implementation and stays on the CPU.
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
    separate_exchange : bool, optional
        With exact exchange, evaluate the Coulomb and exchange derivatives in
        separate passes and report them as ``coulomb_df`` and ``exchange_df``
        (default ``False``: one pass, about a third faster, reported together as
        ``coulomb_exchange_df``).
    """

    def __init__(self, dft_obj, threshold_schwarz_grad=1e-11, ecp_grad_mode='analytical',
                 ecp_series_order=12, ecp_fd_step=1e-3, verbose=True, grid_response=None,
                 use_gpu=None, DF_algo=None, separate_exchange=False):
        if dft_obj is None:
            raise ValueError('ERROR: A PyFock DFT object is required.')
        if not getattr(dft_obj, 'converged', False):
            raise ValueError('ERROR: The supplied DFT object must already be converged.')
        # The converged quantities the gradient reads -- the density matrix, the MOs and the grid --
        # are brought back to the host whatever the SCF ran on. The device routines want them packed
        # their own way anyway, and these are small next to the work done with them.
        self.grids = dft_obj.grids
        if dft_obj.use_gpu and dft_obj.grids is not None:
            self.grids = copy.copy(dft_obj.grids)   # shallow: only the arrays below are replaced
            for name in ('coords', 'weights', 'atomic_weights', 'atom_idx'):
                setattr(self.grids, name, _to_host(getattr(dft_obj.grids, name, None)))

        # Fraction of exact exchange the SCF used (1 for HF, the hybrid fraction for a global hybrid).
        is_hf = isinstance(dft_obj.xc, str) and dft_obj.xc == 'HF'
        self.exx_coef = 1.0 if is_hf else float(getattr(dft_obj, 'exx_coef', 0.0) or 0.0)
        if use_gpu is None:
            # the exchange and four-center gradients have no device implementation: such a GPU SCF
            # is differentiated on the CPU
            use_gpu = bool(getattr(dft_obj, 'use_gpu', False)) and self.exx_coef == 0 and bool(dft_obj.isDF)
        if use_gpu and cp is None:
            raise RuntimeError('use_gpu=True was requested but CuPy is not available.')
        self.use_gpu = bool(use_gpu)
        self._cp_stream = None      # set by calculate() for the duration of a device gradient
        if ecp_grad_mode not in ('analytical', 'fd'):
            raise ValueError("ecp_grad_mode must be 'analytical' or 'fd'.")
        if not dft_obj.isDF:
            if self.use_gpu:
                raise NotImplementedError('The gradient without density fitting runs on the CPU '
                                          '(use_gpu=False, the default for it).')
        elif self.exx_coef > 0:
            # RI exact exchange contracts the three-center blocks themselves, so the SCF evaluates
            # them with DF_algo=11 (the default 12 falls back to it); its gradient does the same.
            if DF_algo is None:
                DF_algo = 11
            if DF_algo != 11:
                raise ValueError('The RI exchange gradient (HF and hybrid functionals) is implemented for '
                                 'DF_algo=11, the algorithm the SCF uses for exact exchange.')
            if self.use_gpu:
                raise NotImplementedError('The RI exchange gradient (HF and hybrid functionals) runs on the CPU '
                                          '(use_gpu=False, the default for them; a GPU SCF is differentiated '
                                          'on the CPU).')
        if DF_algo is None:
            DF_algo = 11 if getattr(dft_obj, 'DF_algo', 12) == 11 else 12
        if DF_algo not in (10, 11, 12):
            raise ValueError('DF_algo must be 12 (multipole-accelerated, default), 11 or 10 for the gradient.')
        self.DF_algo = int(DF_algo)

        self.dft_obj = dft_obj
        self.threshold_schwarz_grad = threshold_schwarz_grad
        self.separate_exchange = bool(separate_exchange)
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
        elif is_hf:
            self.funcid = None      # no XC term
        else:
            if isinstance(xc, list):
                if all(isinstance(v, str) for v in xc):
                    xc = [XC.get_functional_id(name) for name in xc]
            elif isinstance(xc, str):
                xc = XC.resolve_functional(xc)
            self.funcid = xc

    def _strict_pairs(self):
        """Whether the SCF's strict pair cut-off is part of the energy this gradient differentiates.

        It is when the SCF applied it (``strict_schwarz`` with DF algorithm 6, 10, 11 or 12, which
        screen V and J alike) and the gradient uses its plan (DF_algo 11 or 12)."""
        dft_obj = self.dft_obj
        scf_algo = getattr(dft_obj, 'DF_algo', 12)
        if self.exx_coef > 0 and scf_algo == 12:
            scf_algo = 11           # what the SCF ran for exact exchange
        return (dft_obj.isDF and self.DF_algo in (11, 12) and bool(getattr(dft_obj, 'strict_schwarz', False))
                and scf_algo in (6, 10, 11, 12))

    def _scf_fit_gamma(self):
        """``gamma`` of the SCF's last iteration, when it fitted this density with the plan used here."""
        dft_obj = self.dft_obj
        fit = getattr(dft_obj, 'df_fit', None)
        if not fit or fit.get('DF_algo') != self.DF_algo or fit.get('dmat') is not dft_obj.dmat:
            return None
        same_plan = (fit['threshold_schwarz'] == dft_obj.threshold_schwarz
                     and fit['strict_schwarz'] == dft_obj.strict_schwarz
                     and fit['sao'] == bool(dft_obj.sao)
                     and fit['multipole_options'] == dict(getattr(dft_obj, 'multipole_options', None) or {}))
        return np.asarray(fit['gamma'], dtype=np.float64) if same_plan else None

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

    def _occupied_factor(self, dmat):
        """``X`` (nao_cart, nocc) with ``X X^T = D``: the occupied MOs scaled by sqrt(occupation)."""
        dft_obj = self.dft_obj
        mo_coeff = _to_host(dft_obj.mo_coefficients)
        mo_occ = np.asarray(_to_host(dft_obj.mo_occupations))
        occupied = mo_occ > 0
        X = mo_coeff[:, occupied] * np.sqrt(mo_occ[occupied])
        if dft_obj.sao:
            X = dft_obj.basis.cart2sph_basis().T @ X       # SAO --> CAO, as the SCF transforms D
        X = np.ascontiguousarray(X, dtype=np.float64)
        if np.abs(X @ X.T - dmat).max() > 1e-8 * max(1.0, np.abs(dmat).max()):
            # not the factor of this density (e.g. an SCF that ended on a damped or mixed matrix)
            from pyfock.DFT_Helper_Coulomb import _density_matrix_factor
            X = np.ascontiguousarray(_density_matrix_factor(dmat), dtype=np.float64)
        return X

    def _exchange_rows(self, grad_plan, sqrt_ints4c2e_diag, metric_diag, strict, ncores):
        """
        The three-center integrals of the SCF's RI exchange as raw rows ``B[(ij), P] = (ij|P)``
        (fit space: Cartesian, or spherical in SAO mode): the DF_algo=11 blocks with the SCF's
        screening, every significant block evaluated once.  The significant shell pairs must be
        those of the gradient plan, whose kernel differentiates them.
        """
        dft_obj = self.dft_obj
        plan11 = Integrals.df_algo11_helpers.build_plan(
            dft_obj.basis, dft_obj.auxbasis, sqrt_ints4c2e_diag, np.sqrt(np.abs(metric_diag)),
            dft_obj.threshold_schwarz, strict, sao=dft_obj.sao, max_memory_gb=None, ncores=ncores)
        pairs11 = set(zip(plan11.pair_I[plan11.work_iter].tolist(), plan11.pair_J[plan11.work_iter].tolist()))
        pairs12 = set(zip(grad_plan.pair_I[grad_plan.sig].tolist(), grad_plan.pair_J[grad_plan.sig].tolist()))
        if pairs11 != pairs12:
            raise RuntimeError('RI exchange gradient: the integral and derivative plans screen different shell pairs.')
        # Raw rows: the gradient applies the inverse metric to the much smaller occupied blocks
        # (Integrals.df_algo11_exchange.occupied_fit_blocks) instead of orthonormalizing them.
        return Integrals.df_algo11_exchange.build_exchange(
            plan11, dft_obj.basis, dft_obj.auxbasis, None, sao=dft_obj.sao,
            release_plan_values=True, orthonormalize=False)

    def _exchange_weights(self, exchange, dmat, metric_fit, scale=1.0, coulomb_coeff=None):
        """
        Weights of the RI exchange gradient for the occupied factor ``X`` of the density:
        ``Y_P = X^T c^P X`` with ``c^P = sum_Q [(P|Q)^-1] (Q|ij)``, the metric weights
        ``W_PQ = <Y_P, Y_Q>`` (Cartesian auxiliary functions; returned) and, overwriting the rows
        of ``exchange``, the three-center weights ``scale * Gamma^P_ij (+ D_ij c_P)`` with
        ``Gamma^P = X Y_P X^T`` and, when ``coulomb_coeff`` (fit space) is given, the Coulomb
        weights of the same rows.
        """
        dft_obj = self.dft_obj
        X = self._occupied_factor(dmat)
        ncores = dft_obj.ncores
        Y = Integrals.df_algo11_exchange.occupied_fit_blocks(exchange, X, metric_fit)
        naux = Y.shape[0]
        with threadpool_limits(limits=ncores, user_api='blas'):
            Ymat = Y.reshape(naux, -1)
            W = Ymat @ Ymat.T
            if dft_obj.sao:
                c2sph_aux = dft_obj.auxbasis.cart2sph_basis()
                W = c2sph_aux.T @ W @ c2sph_aux
        Integrals.df_algo11_exchange.gradient_rows(exchange, X, Y, scale=scale, dmat=dmat, coeff=coulomb_coeff)
        return np.ascontiguousarray(W)

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

    def _xc_grad(self, dmat, bfs_atoms, natoms, ncores, use_gpu):
        """XC term of the gradient (fixed grid, plus the grid response when requested)."""
        dft_obj = self.dft_obj
        basis = dft_obj.basis
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
        return grad_xc

    def _df_two_electron_grad(self, dmat, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, strict, timings):
        """
        Density-fitted Coulomb terms and, with exact exchange, the RI exchange terms. Returns
        ``(grad_J, grad_K, fused)``: with ``fused`` both are in ``grad_J`` and ``grad_K`` is None.
        """
        dft_obj = self.dft_obj
        basis = dft_obj.basis
        auxbasis = dft_obj.auxbasis
        ncores = dft_obj.ncores
        use_gpu = self.use_gpu

        # ---------------- DF Coulomb ----------------
        # gamma_P = sum_ij D_ij (ij|P);  c = (P|Q)^-1 gamma
        start = timer()
        nbf = basis.bfs_nao
        naux = auxbasis.bfs_nao
        if use_gpu:
            ints2c2e = Integrals.rys_2c2e_symm_cupy(auxbasis, cp_stream=self._cp_stream)
            if dft_obj.sao:
                # The spherical transform below is host code, and the metric is only naux x naux.
                ints2c2e = cp.asnumpy(ints2c2e)
        else:
            ints2c2e = Integrals.rys_2c2e_symm(auxbasis)
        ints2c2e_sph = auxbasis.cart2sph_operator_blockwise(ints2c2e) if dft_obj.sao else None
        timings['df_metric'] = timer() - start

        plan = None
        if self.DF_algo in (11, 12):
            # The SCF's plan without its integral values: the same screening (from the Schwarz bounds
            # of the metric it fitted with, which is the projected one in SAO mode), branches,
            # near/far classification and group moments.
            start = timer()
            if dft_obj.sao:
                metric_diag = _pseudo_cartesian_metric_diagonal(
                    auxbasis, ints2c2e_sph, auxbasis.sph2cart_basis()) + 1e-12
            else:
                metric_diag = _to_host(ints2c2e.diagonal())
            plan = Integrals.df_algo12_grad.build_grad_plan(
                basis, auxbasis, sqrt_ints4c2e_diag, np.sqrt(np.abs(metric_diag)),
                dft_obj.threshold_schwarz, strict, sao=dft_obj.sao,
                options=getattr(dft_obj, 'multipole_options', None),
                far_field=self.DF_algo == 12, ncores=ncores)
            timings['df_plan'] = timer() - start

        exchange = None
        if self.exx_coef > 0:
            # The three-center integrals of the SCF's exchange, as raw rows (ij|P) in its fit space
            # (spherical in SAO mode), screened exactly as the SCF screened them.
            start = timer()
            exchange = self._exchange_rows(plan, sqrt_ints4c2e_diag, metric_diag, strict, ncores)
            timings['exchange_integrals'] = timer() - start

        start = timer()
        gamma = self._scf_fit_gamma()
        gamma_fit = None
        if gamma is not None:
            pass        # the SCF's last iteration fitted this very density with the same plan
        elif exchange is not None:
            # the rows hold the same screened integrals in the fit space
            gamma_fit = Integrals.df_algo11_exchange.gamma_from_exchange(exchange, dmat)
        elif plan is not None and not use_gpu:
            gamma = Integrals.df_algo12_helpers.gamma_from_plan(plan, dmat)
        elif use_gpu:
            # the device kernel contracts the 3c2e integrals as it goes and never forms them
            gamma = Integrals.rys_3c2e_gamma_contract_cupy(
                basis, auxbasis, dmat,
                threshold_schwarz=min(dft_obj.threshold_schwarz, 1e-9),
                sqrt_ints4c2e_diag=sqrt_ints4c2e_diag,
                sqrt_diag_ints2c2e=sqrt_diag_ints2c2e, cp_stream=self._cp_stream)
        else:
            # The 3c2e tensor is only needed transiently, so it is evaluated in chunks over the
            # auxiliary dimension to bound memory.
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
                # independent. (A projected gamma, as algorithms 11 and 12
                # produce it in SAO mode, has the same T gamma.)
                c2sph_aux = auxbasis.cart2sph_basis()
                gamma_sph = gamma_fit if gamma_fit is not None else c2sph_aux @ _to_host(gamma)
                c_sph = scipy.linalg.solve(ints2c2e_sph, gamma_sph, assume_a='pos')
                df_coeff = c2sph_aux.T @ c_sph
            elif use_gpu:
                # cuSOLVER's Cholesky solve, so the naux x naux metric never leaves the device.
                df_coeff = cp.asnumpy(cp.linalg.solve(ints2c2e, cp.asarray(gamma)))
            else:
                df_coeff = scipy.linalg.solve(ints2c2e, gamma if gamma_fit is None else gamma_fit,
                                              assume_a='pos')
        timings['df_coefficients'] = timer() - start

        # With exact exchange the three-center derivatives of the Coulomb and exchange terms share
        # one pass over the rows (weights D_ij c_P - exx/2 Gamma^P_ij), unless separate_exchange.
        fused = exchange is not None and not self.separate_exchange
        exchange_metric_weights = None
        if exchange is not None:
            # Y_P = X^T c^P X for the occupied factor X of the density, then the metric weights
            # W_PQ = <Y_P, Y_Q> and, in place of the integrals, the three-center weights
            # Gamma^P = X Y_P X^T = D c^P D.
            start = timer()
            exchange_metric_weights = self._exchange_weights(
                exchange, dmat, ints2c2e_sph if dft_obj.sao else ints2c2e, scale=-0.5 * self.exx_coef,
                coulomb_coeff=((c_sph if dft_obj.sao else df_coeff) if fused else None))
            timings['exchange_weights'] = timer() - start
        ints2c2e = ints2c2e_sph = None

        grad_K = None
        if exchange is not None:
            # E_K = -(exx/4) sum D_ij D_kl (ik|P) [(P|Q)^-1] (Q|jl):
            # dE_K = -(exx/2) sum (ik|P)' Gamma^P_ik + (exx/4) sum (P|Q)' W_PQ
            label = 'coulomb_exchange' if fused else 'exchange'
            start = timer()
            row_of = np.full((nbf, nbf), -1, dtype=np.int64)
            row_of[exchange.row_mu, exchange.row_nu] = np.arange(exchange.nrows, dtype=np.int64)
            fit_tables = (Integrals.df_algo11_exchange._cart2sph_tables(auxbasis)[:4]
                          if dft_obj.sao else None)
            grad_K = Integrals.df_algo12_grad.grad_contract_rows(
                plan, exchange.B, row_of, fit_tables, threshold_grad=self.threshold_schwarz_grad)
            exchange = row_of = None
            timings[label + '_3c2e_grad'] = timer() - start
            start = timer()
            weights_2c = 0.25 * self.exx_coef * exchange_metric_weights
            exchange_metric_weights = None
            if fused:
                weights_2c -= 0.5 * np.outer(df_coeff, df_coeff)
            grad_K = grad_K + Integrals.rys_2c2e_grad_contract(auxbasis, weights=weights_2c, ncores=ncores)
            weights_2c = None
            timings[label + '_2c2e_grad'] = timer() - start

        if fused:
            grad_J = grad_K         # Coulomb and exchange together
            grad_K = None
        else:
            start = timer()
            if plan is not None and use_gpu:
                # near field on the device, with the plan's far-field mask; the far field is a few
                # translations per group and stays on the host
                grad_J3c = Integrals.rys_3c2e_grad_contract_cupy(
                    basis, auxbasis, dmat, df_coeff,
                    schwarz=True, threshold_schwarz=self.threshold_schwarz_grad,
                    sqrt_ints4c2e_diag=sqrt_ints4c2e_diag,
                    sqrt_diag_ints2c2e=sqrt_diag_ints2c2e, cp_stream=self._cp_stream,
                    df12_plan=plan)
                grad_J3c = grad_J3c + Integrals.df_algo12_grad.grad_contract(
                    plan, dmat, df_coeff, threshold_grad=self.threshold_schwarz_grad, near=False)
            elif plan is not None:
                grad_J3c = Integrals.df_algo12_grad.grad_contract(
                    plan, dmat, df_coeff, threshold_grad=self.threshold_schwarz_grad)
            elif use_gpu:
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
        plan = None
        return grad_J, grad_K, fused

    def calculate(self):
        """
        Calculate analytical gradients and forces.

        Returns
        -------
        dict
            Dictionary with `energy` (the energy ``scf()`` returned, D3
            included when applied), `gradient` (natoms, 3) in Ha/Bohr,
            `forces` (= -gradient), the per-term `gradient_components` and
            per-term `timings`.
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
        if dft_obj.isDF and (use_gpu or self.DF_algo in (11, 12)):
            start = timer()
            sqrt_ints4c2e_diag = np.sqrt(np.abs(Integrals.schwarz_helpers.eri_4c2e_diag(basis)))
            if use_gpu:
                sqrt_diag_ints2c2e = np.sqrt(np.abs(Integrals.rys_2c2e_diag(auxbasis)))
            timings['schwarz_diagonals'] = timer() - start

        # With strict Schwarz screening the SCF leaves the function pairs with (ij|ij) below the
        # strict cut-off out of the nuclear attraction matrix as well as out of the Coulomb term, so
        # the gradient of its energy leaves them out of both derivatives. (DF_algo=10 keeps the
        # previous behaviour, which differentiates them in both.)
        strict = self._strict_pairs()
        dmat_V = dmat
        if strict:
            dmat_V = np.where(sqrt_ints4c2e_diag ** 2 < STRICT_PAIR_CUTOFF, 0.0, dmat)

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
                basis, mol, dmat_V, sqrt_ints4c2e_diag=sqrt_ints4c2e_diag,
                cp_stream=self._cp_stream)
        else:
            grad_V = Integrals.rys_nuc_grad_contract(basis, mol, dmat_V, ncores=ncores)
        dmat_V = None
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

        # ---------------- Two-electron terms ----------------
        if dft_obj.isDF:
            grad_J, grad_K, fused = self._df_two_electron_grad(dmat, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e,
                                                               strict, timings)
        else:
            # Coulomb and exact exchange from the derivatives of the four-center integrals
            start = timer()
            plan4 = Integrals.jk_4c2e.build_plan(basis, threshold=self.threshold_schwarz_grad, scheme='rys')
            grad_J = Integrals.jk_4c2e_grad.grad_4c2e(plan4, dmat, self.exx_coef)
            grad_K = fused = plan4 = None
            timings['coulomb_exchange_4c2e_grad' if self.exx_coef > 0 else 'coulomb_4c2e_grad'] = timer() - start

        # ---------------- XC ----------------
        start = timer()
        grad_xc = np.zeros((natoms, 3))
        if self.funcid is None and self.skala is None:
            pass                    # Hartree-Fock: no XC term
        else:
            grad_xc = self._xc_grad(dmat, bfs_atoms, natoms, ncores, use_gpu)
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

        # ---------------- DFT-D3 dispersion (if the SCF applied it) ----------------
        # scf() adds E_disp to the energy it returns, so the gradient must carry dE_disp/dR as well.
        # Its net force vanishes on its own, so translational invariance would not notice it missing.
        grad_disp = np.zeros((natoms, 3))
        dispersion_method = getattr(dft_obj, 'dispersion_method', None)
        if dispersion_method is not None:
            start = timer()
            _, grad_disp = Dispersion.d3_energy_and_gradient(
                mol, dispersion_method, version=dft_obj.dispersion_version,
                atm=dft_obj.dispersion_atm)
            timings['dispersion'] = timer() - start

        gradient = grad_nn + grad_T + grad_V + grad_S + grad_J + grad_xc + grad_ecp + grad_disp
        if grad_K is not None:
            gradient = gradient + grad_K
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

        components = {
            'nuclear_repulsion': grad_nn,
            'kinetic': grad_T,
            'nuclear_attraction': grad_V,
            'overlap_pulay': grad_S,
        }
        if not dft_obj.isDF:
            components['coulomb_exchange_4c2e' if self.exx_coef > 0 else 'coulomb_4c2e'] = grad_J
        elif fused:
            components['coulomb_exchange_df'] = grad_J
        else:
            components['coulomb_df'] = grad_J
            if grad_K is not None:
                components['exchange_df'] = grad_K
        components.update({'xc': grad_xc, 'ecp': grad_ecp, 'dispersion': grad_disp})
        return {
            'energy': float(dft_obj.Total_energy),
            'gradient': gradient,
            'forces': forces,
            'gradient_components': components,
            'timings': timings,
        }
