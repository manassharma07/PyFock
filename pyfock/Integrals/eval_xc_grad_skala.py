"""Nuclear gradient of the Skala neural exchange-correlation functional (CPU).

Writing the XC energy as ``E(rho_p, grad rho_p, tau_p, r_p, w_p, R)``, the nuclear gradient collects four
kinds of term. The first is shared with every semilocal functional; the other three exist because Skala
reads the grid geometry itself and because its features are integrals over each atomic grid.

1. **Density-mediated (Pulay).** The basis functions centred on an atom move with it, changing the
   density features at fixed grid points. Once the per-point derivatives ``dE/drho`` etc. are known, this
   contraction is functional-agnostic, so it is exactly the meta-GGA contraction of
   :func:`pyfock.Integrals.eval_xc_grad_2` with the model's cotangents in place of
   ``vrho``/``vsigma``/``vtau``. Because the cotangents already carry the quadrature weights, the
   ``weights_block`` factors of the semilocal path are absent here.
2. **Grid translation.** The grid points of an atom translate rigidly with it, so the density features
   evaluated *at those points* change even with the density field held fixed. This contributes
   ``sum_{p in A} [c_rho grad rho + c_grad . grad grad rho + c_tau grad tau]`` and needs only the AO
   Hessians that term 1 already evaluates.
3. **Grid response.** The Becke partitioning weights depend on every nuclear position, and the points
   they are evaluated at move too. :func:`pyfock.Grids.becke_weight_gradient` supplies ``dw/dR``.
4. **Explicit.** Skala consumes the grid coordinates and the nuclear coordinates directly, so the model
   depends on the geometry beyond the density. This falls out of the same reverse pass as the
   cotangents.

Terms 2 and 3 are large and of opposite sign -- they cancel to a large extent -- so neither may be
included without the other. Omitting both (the fixed-grid approximation PyFock makes for semilocal
functionals) costs about 1e-4 Ha/Bohr for a meta-GGA but about 1e-2 Ha/Bohr for Skala, which is the size
of the forces themselves. That is why this driver evaluates them.

Cost
----
Three passes over the grid: AO values and gradients to build the density, one model call, then AO values,
gradients *and* Hessians for the contraction. The extra terms ride along on the third pass and on one
reverse pass through the partitioning, so they add little over a meta-GGA gradient.
"""

import numpy as np
import random
from timeit import default_timer as timer

import numba
from joblib import Parallel, delayed
from opt_einsum import contract
from threadpoolctl import threadpool_limits

from pyfock import Integrals
from .eval_xc_skala import _bfs_arrays, _block_aos

__all__ = ['eval_xc_grad_skala']

# ao_hess components are stored as 0:xx 1:xy 2:xz 3:yy 4:yz 5:zz.
_HESS = ((0, 1, 2), (1, 3, 4), (2, 4, 5))


def eval_xc_grad_skala(basis, dmat, grids, skala, ncores=2, blocksize=5000,
                       list_nonzero_indices=None, count_nonzero_indices=None,
                       max_points_per_chunk=250000, grid_response=True, debug=False):
    """XC gradient contributions of a Skala functional.

    Parameters
    ----------
    basis : Basis
        Basis set of the calculation.
    dmat : (nao, nao) ndarray
        Converged total density matrix in the AO basis. Closed shell only.
    grids : Grids
        Molecular grid; must carry :attr:`~pyfock.Grids.Grids.atomic_weights`.
    skala : SkalaFunctional
        Loaded functional, from :func:`pyfock.XC.load_skala`.
    ncores : int
        Threads for the Numba AO kernels and the joblib block loop.
    blocksize : int
        Grid points per block.
    list_nonzero_indices, count_nonzero_indices : list, optional
        Per-block AO screening data, as built by the SCF driver.
    max_points_per_chunk : int
        Upper bound on the grid points handed to the model at once.
    grid_response : bool
        Include the grid-translation and Becke weight-response terms. Leaving them out reproduces the
        fixed-grid approximation used for semilocal functionals, which is *not* accurate enough for
        Skala; the switch exists to measure that.
    debug : bool
        Print a breakdown of the time spent in each pass.

    Returns
    -------
    dexc_dbf : (3, nao) ndarray
        Per-basis-function contributions, in the same convention as
        :func:`pyfock.Integrals.eval_xc_grad_2`: the caller maps them onto atoms with
        ``np.add.at(grad, basis.bfs_atoms, -2.0 * dexc_dbf.T)``.
    atom_grad : (natm, 3) ndarray
        Everything that is already resolved per atom -- the explicit, grid-translation and
        grid-response terms. Added to the total gradient as it is.
    """
    if getattr(grids, 'atomic_weights', None) is None:
        raise ValueError(
            "Skala needs the unpartitioned single-atom quadrature weights, which the 'numgrid' grid "
            "scheme does not expose. Build the grid with the native scheme instead.")

    coords, weights = grids.coords, grids.weights
    ngrids = coords.shape[0]
    nblocks = ngrids // blocksize
    nao = basis.bfs_nao
    atom_idx = np.ascontiguousarray(grids.atom_idx, dtype=np.int64)
    atom_coords = np.asarray(grids.mol.coordsBohrs, dtype=np.float64).reshape(-1, 3)
    natm = atom_coords.shape[0]
    bfs = _bfs_arrays(basis)
    durations = {}

    def bounds(iblock):
        lo = iblock * blocksize
        return lo, min(lo + blocksize, ngrids)

    def indices(iblock):
        if list_nonzero_indices is None:
            return None
        return list_nonzero_indices[iblock][0:count_nonzero_indices[iblock]]

    # ------------------------------------------------------------------ pass 1: density on the grid
    start = timer()
    numba.set_num_threads(ncores)
    rho = np.zeros(ngrids)
    rho_grad = np.zeros((3, ngrids))
    tau = np.zeros(ngrids)
    for iblock in range(nblocks + 1):
        lo, hi = bounds(iblock)
        if hi <= lo:
            continue
        idx = indices(iblock)
        ao, ao_grad = _block_aos(bfs, coords[lo:hi], idx, None, None)
        dmat_block = dmat if idx is None else dmat[np.ix_(idx, idx)]
        Fmj = ao @ dmat_block
        rho[lo:hi] = contract('mj,mj->m', Fmj, ao)
        rho_grad[:, lo:hi] = 2 * contract('mj,kmj->km', Fmj, ao_grad)
        tau[lo:hi] = 0.5 * contract('ij,kmi,kmj->m', dmat_block, ao_grad, ao_grad)
    durations['rho'] = timer() - start

    # ------------------------------------------------------------------ pass 2: one model call
    start = timer()
    _, vrho, vgrad, vtau, atom_grad, dweights = skala.exc_and_potential(
        rho, rho_grad, tau, coords, weights, grids.atomic_weights, atom_idx, atom_coords,
        max_points_per_chunk=max_points_per_chunk, nuclear_terms=True)
    durations['model'] = timer() - start
    del rho, tau
    if not grid_response:
        atom_grad = np.zeros_like(atom_grad)

    # ------------------------------------------------------------------ grid response: dE/dw . dw/dR
    start = timer()
    if grid_response:
        from pyfock.Grids import becke_weight_gradient, size_adjustment_table
        # the charges the grid was built with (the element, also for ghost and ECP atoms)
        a_table = size_adjustment_table(grids.charges, getattr(grids, 'size_adjustment', 'treutler'))
        # dE/dw_p * dw_p/dR, with w_p = vol_p * P_A(r_p): the volume element rides along in the
        # cotangent because only the partitioning factor depends on the nuclei.
        atom_grad = atom_grad + becke_weight_gradient(coords, atom_idx, atom_coords, a_table,
                                                      dweights * grids.atomic_weights)
    durations['weights'] = timer() - start

    # ------------------------------------------------------------------ pass 3: AO contraction
    start = timer()
    numba.set_num_threads(1)
    order = [i for i in range(nblocks + 1) if bounds(i)[1] > bounds(i)[0]]
    random.shuffle(order)                       # load balancing, as in eval_xc_grad_2
    batch_size = 'auto' if 2 * ncores > nblocks else max(1, nblocks // (ncores * 2))
    full = np.arange(nao)

    with threadpool_limits(limits=1, user_api='blas'):
        output = Parallel(n_jobs=ncores, backend='threading', require='sharedmem',
                          batch_size=batch_size)(
            delayed(_block_grad)(
                coords[bounds(i)[0]:bounds(i)[1]],
                dmat if indices(i) is None else dmat[np.ix_(indices(i), indices(i))],
                vrho[bounds(i)[0]:bounds(i)[1]],
                vgrad[:, bounds(i)[0]:bounds(i)[1]],
                vtau[bounds(i)[0]:bounds(i)[1]],
                rho_grad[:, bounds(i)[0]:bounds(i)[1]],
                atom_idx[bounds(i)[0]:bounds(i)[1]],
                natm, bfs, full if indices(i) is None else indices(i), grid_response)
            for i in order)

    dexc_dbf = np.zeros((3, nao))
    for iblock, (block, translation) in zip(order, output):
        idx = indices(iblock)
        if idx is None:
            dexc_dbf += block
        else:
            dexc_dbf[:, idx] += block
        atom_grad = atom_grad + translation
    numba.set_num_threads(ncores)
    durations['grad'] = timer() - start

    if debug:
        print('Skala gradient timings (s): density %.3f, model %.3f, weights %.3f, contraction %.3f'
              % (durations['rho'], durations['model'], durations['weights'], durations['grad']),
              flush=True)
    return dexc_dbf, atom_grad


def _block_grad(coords_block, dmat, vrho_block, vgrad_block, vtau_block, rho_grad_block,
                atom_idx_block, natm, bfs, non_zero_indices, grid_response):
    """Per-basis-function gradient contributions of one grid block, and its grid-translation term."""
    numba.set_num_threads(1)
    ao, ao_grad, ao_hess = Integrals.bf_val_helpers.eval_bfs_grad_and_hess_sparse_internal_serial(
        bfs[0], bfs[1], bfs[2], bfs[3], bfs[4], bfs[5], bfs[6], coords_block, non_zero_indices)

    Fmj = ao @ dmat                                     # (chi D)[g, mu]
    Hgrad = [ao_grad[0] @ dmat, ao_grad[1] @ dmat, ao_grad[2] @ dmat]
    Fx, Fy, Fz = vgrad_block[0], vgrad_block[1], vgrad_block[2]

    # ---- 1. Pulay: the meta-GGA contraction of eval_xc_grad_2, cotangents already weighted.
    aow = vrho_block[:, None] * ao
    aow += Fx[:, None] * ao_grad[0]
    aow += Fy[:, None] * ao_grad[1]
    aow += Fz[:, None] * ao_grad[2]
    aowD = aow @ dmat

    res = np.empty((3, ao.shape[1]))
    res[0] = np.einsum('mj,mj->j', ao_grad[0], aowD)
    res[1] = np.einsum('mj,mj->j', ao_grad[1], aowD)
    res[2] = np.einsum('mj,mj->j', ao_grad[2], aowD)

    hessF = Fx[:, None] * ao_hess[0] + Fy[:, None] * ao_hess[1] + Fz[:, None] * ao_hess[2]
    res[0] += np.einsum('mj,mj->j', hessF, Fmj)
    hessF = Fx[:, None] * ao_hess[1] + Fy[:, None] * ao_hess[3] + Fz[:, None] * ao_hess[4]
    res[1] += np.einsum('mj,mj->j', hessF, Fmj)
    hessF = Fx[:, None] * ao_hess[2] + Fy[:, None] * ao_hess[4] + Fz[:, None] * ao_hess[5]
    res[2] += np.einsum('mj,mj->j', hessF, Fmj)

    Gtau = 0.5 * vtau_block
    GH = [Gtau[:, None] * Hgrad[0], Gtau[:, None] * Hgrad[1], Gtau[:, None] * Hgrad[2]]
    for d in range(3):
        res[d] += sum(np.einsum('mj,mj->j', ao_hess[_HESS[d][k]], GH[k]) for k in range(3))

    translation = np.zeros((natm, 3))
    if not grid_response:
        return res, translation

    # ---- 2. Grid translation: the points of an atom move with it, so the density features evaluated
    #         there change even at fixed density. d(rho)/dR_b = grad_b rho, and likewise for grad rho
    #         (its Hessian) and tau (its gradient) -- all available from the AO Hessians above.
    for b in range(3):
        term = vrho_block * rho_grad_block[b]
        for a in range(3):
            # d(grad_a rho)/d b = 2 [ (d_a d_b chi) . (chi D) + (d_a chi) . (d_b chi D) ]
            hessian_ab = 2.0 * (np.einsum('mj,mj->m', ao_hess[_HESS[a][b]], Fmj)
                                + np.einsum('mj,mj->m', ao_grad[a], Hgrad[b]))
            term += vgrad_block[a] * hessian_ab
        # d(tau)/d b = sum_k (d_b d_k chi) . (d_k chi D)
        grad_tau = sum(np.einsum('mj,mj->m', ao_hess[_HESS[b][k]], Hgrad[k]) for k in range(3))
        term += vtau_block * grad_tau
        translation[:, b] = np.bincount(atom_idx_block, weights=term, minlength=natm)
    return res, translation
