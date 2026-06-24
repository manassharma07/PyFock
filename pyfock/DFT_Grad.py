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
    the native PyFock functionals or pylibxc, and ECPs, CPU only.
    Laplacian-dependent meta-GGAs are not yet supported.

    Parameters
    ----------
    dft_obj : DFT
        A converged PyFock DFT object (after ``dft_obj.scf()``).
    threshold_schwarz_grad : float, optional
        Screening threshold used for the contracted 3c2e derivative
        integrals (includes density/coefficient weighting).
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
                 ecp_series_order=12, ecp_fd_step=1e-3, verbose=True):
        if dft_obj is None:
            raise ValueError('ERROR: A PyFock DFT object is required.')
        if not getattr(dft_obj, 'converged', False):
            raise ValueError('ERROR: The supplied DFT object must already be converged.')
        if dft_obj.use_gpu:
            raise NotImplementedError('Analytical gradients are currently implemented for CPU only.')
        if not dft_obj.isDF:
            raise NotImplementedError('Analytical gradients are currently implemented for density-fitted (isDF=True) calculations only.')
        if dft_obj.xc == 'HF':
            raise NotImplementedError('Analytical gradients are currently implemented for pure DFT functionals only.')
        if ecp_grad_mode not in ('analytical', 'fd'):
            raise ValueError("ecp_grad_mode must be 'analytical' or 'fd'.")

        self.dft_obj = dft_obj
        self.threshold_schwarz_grad = threshold_schwarz_grad
        self.ecp_grad_mode = ecp_grad_mode
        self.ecp_series_order = ecp_series_order
        self.ecp_fd_step = ecp_fd_step
        self.verbose = verbose

        # Resolve the functional specification the same way DFT.scf does
        xc = dft_obj.xc
        if isinstance(xc, list):
            if all(isinstance(v, str) for v in xc):
                xc = [XC.get_functional_id(name) for name in xc]
        elif isinstance(xc, str):
            xc = XC.resolve_functional(xc)
        self.funcid = xc

    def _energy_weighted_dmat(self):
        """Energy-weighted density matrix W in the CAO basis."""
        dft_obj = self.dft_obj
        mo_coeff = dft_obj.mo_coefficients
        mo_energy = dft_obj.mo_energies
        mo_occ = dft_obj.mo_occupations
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
        dft_obj = self.dft_obj
        mol = dft_obj.mol
        basis = dft_obj.basis
        auxbasis = dft_obj.auxbasis
        ncores = dft_obj.ncores
        natoms = mol.natoms

        numba.set_num_threads(ncores)

        dmat = np.ascontiguousarray(dft_obj.dmat, dtype=np.float64)
        bfs_atoms = np.asarray(basis.bfs_atoms, dtype=np.int64)

        timings = {}

        # ---------------- Nuclear repulsion ----------------
        start = timer()
        grad_nn = self._nuclear_repulsion_grad()
        timings['nuclear_repulsion'] = timer() - start

        # ---------------- Kinetic (Pulay-type) ----------------
        # dT_r[d, i, j] = dT_ij / d(center of bf i)
        start = timer()
        dT_r = Integrals.kin_mat_grad_r_symm(basis)
        tmpT = contract('dij,ij->id', dT_r, dmat)
        grad_T = np.zeros((natoms, 3))
        np.add.at(grad_T, bfs_atoms, 2.0 * tmpT)
        dT_r = None
        timings['kinetic'] = timer() - start

        # ---------------- Nuclear attraction ----------------
        # Full derivative incl. operator (Hellmann-Feynman) contributions,
        # contracted with the density matrix on the fly.
        start = timer()
        grad_V = Integrals.rys_nuc_grad_contract(basis, mol, dmat, ncores=ncores)
        timings['nuclear_attraction'] = timer() - start

        # ---------------- Overlap (Pulay) ----------------
        start = timer()
        W = self._energy_weighted_dmat()
        dS_r = Integrals.overlap_mat_grad_r_symm(basis)
        tmpS = contract('dij,ij->id', dS_r, W)
        grad_S = np.zeros((natoms, 3))
        np.add.at(grad_S, bfs_atoms, -2.0 * tmpS)
        dS_r = None
        timings['overlap'] = timer() - start

        # ---------------- DF Coulomb ----------------
        # gamma_P = sum_ij D_ij (ij|P);  c = (P|Q)^-1 gamma
        # The 3c2e tensor is only needed transiently for gamma, so it is
        # evaluated in chunks over the auxiliary dimension to bound memory.
        start = timer()
        ints2c2e = Integrals.rys_2c2e_symm(auxbasis)
        nbf = basis.bfs_nao
        naux = auxbasis.bfs_nao
        max_chunk_bytes = 1e9
        chunk_naux = max(1, min(naux, int(max_chunk_bytes / (nbf * nbf * 8))))
        gamma = np.zeros(naux)
        with threadpool_limits(limits=ncores, user_api='blas'):
            for c0 in range(0, naux, chunk_naux):
                c1 = min(c0 + chunk_naux, naux)
                ints3c2e_chunk = Integrals.rys_3c2e_symm_test(
                    basis, auxbasis, slice=[0, nbf, 0, nbf, c0, c1],
                    schwarz=True,
                    threshold_schwarz=min(dft_obj.threshold_schwarz, 1e-9),
                )
                gamma[c0:c1] = contract('ijP,ij->P', ints3c2e_chunk, dmat)
                ints3c2e_chunk = None
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
            else:
                df_coeff = scipy.linalg.solve(ints2c2e, gamma, assume_a='pos')
        ints2c2e = None
        timings['df_coefficients'] = timer() - start

        start = timer()
        grad_J3c = Integrals.rys_3c2e_grad_contract(
            basis, auxbasis, dmat, df_coeff,
            schwarz=True, threshold_schwarz=self.threshold_schwarz_grad,
            ncores=ncores,
        )
        timings['coulomb_3c2e_grad'] = timer() - start

        start = timer()
        grad_J2c = Integrals.rys_2c2e_grad_contract(auxbasis, df_coeff, ncores=ncores)
        grad_J = grad_J3c - 0.5 * grad_J2c
        timings['coulomb_2c2e_grad'] = timer() - start

        # ---------------- XC ----------------
        start = timer()
        grids = dft_obj.grids
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

        dexc_dbf = Integrals.eval_xc_grad_2(
            basis, dmat, weights_grid, coords_grid, funcid=self.funcid,
            use_libxc=dft_obj.use_libxc, ncores=ncores, blocksize=blocksize,
            list_nonzero_indices=list_nonzero_indices,
            count_nonzero_indices=count_nonzero_indices,
        )
        grad_xc = np.zeros((natoms, 3))
        np.add.at(grad_xc, bfs_atoms, -2.0 * dexc_dbf.T)
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
            'energy': dft_obj.Total_energy,
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
