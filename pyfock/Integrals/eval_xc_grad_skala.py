"""Nuclear gradient of the Skala neural exchange-correlation functional (CPU).

The XC gradient splits into two pieces, and only the second is specific to a neural functional.

**The density-mediated part** is the ordinary meta-GGA gradient. Once the per-grid-point derivatives
``dE/drho``, ``dE/d(grad rho)`` and ``dE/dtau`` are known, contracting them with the AO derivatives is
functional-agnostic: the chain rule from grid quantities through the density matrix to the nuclei does
not care where those derivatives came from. This module therefore runs exactly the contraction of
:func:`pyfock.Integrals.eval_xc_grad_2`, with the ``F``/``Fx,Fy,Fz``/``G_tau`` intermediates taken from
the model's cotangents instead of from ``vrho``/``vsigma``/``vtau``. Because the cotangents already
carry the quadrature weights, the ``weights_block`` factors of the semilocal path are absent here.

**The explicit part** has no semilocal analogue. Skala consumes the grid geometry itself -- the point
coordinates and the nuclear coordinates are model inputs -- so the energy depends on the nuclear
positions beyond the density. Both contributions fall out of the same reverse pass that produces the
cotangents (see :meth:`pyfock.XC.SkalaFunctional.exc_and_potential` with ``nuclear_terms=True``).

**What is left out.** The energy also depends on the nuclei through the Becke partitioning weights,
``dE/dw . dw/dR``. PyFock does not evaluate ``dw/dR`` anywhere, and its semilocal analytical gradients
make the same fixed-grid approximation (see :func:`pyfock.Integrals.eval_xc_grad_2`, which follows
PySCF's default). Skala's own implementation notes that this weight-response term largely cancels
against the grid-translation contribution, so the two are best thought of as a pair; the size of what is
neglected here is measured against numerical gradients in
``benchmarks_tests/benchmark_skala_gradients.py``.

Cost
----
Three passes over the grid, the same shape as the energy driver: AO values and gradients to build the
density, one model call, then AO values, gradients *and* Hessians for the contraction. Only the last is
appreciably more expensive than an energy evaluation, and it is the same pass a meta-GGA gradient needs.
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


def eval_xc_grad_skala(basis, dmat, grids, skala, ncores=2, blocksize=5000,
                       list_nonzero_indices=None, count_nonzero_indices=None,
                       max_points_per_chunk=250000, debug=False):
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
    debug : bool
        Print a breakdown of the time spent in the three passes.

    Returns
    -------
    dexc_dbf : (3, nao) ndarray
        Per-basis-function contributions, in the same convention as
        :func:`pyfock.Integrals.eval_xc_grad_2`: the caller maps them onto atoms with
        ``np.add.at(grad, basis.bfs_atoms, -2.0 * dexc_dbf.T)``.
    explicit_grad : (natm, 3) ndarray
        The explicit nuclear gradient, already per atom. Added to the total gradient as it is.
    """
    if getattr(grids, 'atomic_weights', None) is None:
        raise ValueError(
            "Skala needs the unpartitioned single-atom quadrature weights, which the 'numgrid' grid "
            "scheme does not expose. Build the grid with the native scheme instead.")

    coords = grids.coords
    weights = grids.weights
    ngrids = coords.shape[0]
    nblocks = ngrids // blocksize
    nao = basis.bfs_nao
    bfs = _bfs_arrays(basis)
    durations = {}

    def block_bounds(iblock):
        lo = iblock * blocksize
        return lo, min(lo + blocksize, ngrids)

    def block_indices(iblock):
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
        lo, hi = block_bounds(iblock)
        if hi <= lo:
            continue
        idx = block_indices(iblock)
        ao, ao_grad = _block_aos(bfs, coords[lo:hi], idx, None, None)
        dmat_block = dmat if idx is None else dmat[np.ix_(idx, idx)]
        Fmj = ao @ dmat_block
        rho[lo:hi] = contract('mj,mj->m', Fmj, ao)
        rho_grad[:, lo:hi] = 2 * contract('mj,kmj->km', Fmj, ao_grad)
        tau[lo:hi] = 0.5 * contract('ij,kmi,kmj->m', dmat_block, ao_grad, ao_grad)
    durations['rho'] = timer() - start

    # ------------------------------------------------------------------ pass 2: one model call
    start = timer()
    _, vrho, vgrad, vtau, explicit_grad = skala.exc_and_potential(
        rho, rho_grad, tau, coords, weights, grids.atomic_weights, grids.atom_idx,
        np.asarray(grids.mol.coordsBohrs, dtype=np.float64).reshape(-1, 3),
        max_points_per_chunk=max_points_per_chunk, nuclear_terms=True)
    durations['model'] = timer() - start
    del rho, rho_grad, tau

    # ------------------------------------------------------------------ pass 3: AO contraction
    start = timer()
    numba.set_num_threads(1)
    order = list(range(nblocks + 1))
    random.shuffle(order)                       # load balancing, as in eval_xc_grad_2
    batch_size = 'auto' if 2 * ncores > nblocks else max(1, nblocks // (ncores * 2))
    full_indices = np.arange(nao)

    # One BLAS thread inside the workers: they already parallelize over grid blocks.
    with threadpool_limits(limits=1, user_api='blas'):
        output = Parallel(n_jobs=ncores, backend='threading', require='sharedmem',
                          batch_size=batch_size)(
            delayed(_block_grad)(
                coords[block_bounds(iblock)[0]:block_bounds(iblock)[1]],
                dmat if block_indices(iblock) is None
                else dmat[np.ix_(block_indices(iblock), block_indices(iblock))],
                vrho[block_bounds(iblock)[0]:block_bounds(iblock)[1]],
                vgrad[:, block_bounds(iblock)[0]:block_bounds(iblock)[1]],
                vtau[block_bounds(iblock)[0]:block_bounds(iblock)[1]],
                bfs,
                full_indices if block_indices(iblock) is None else block_indices(iblock))
            for iblock in order if block_bounds(iblock)[1] > block_bounds(iblock)[0])

    dexc_dbf = np.zeros((3, nao))
    produced = [iblock for iblock in order if block_bounds(iblock)[1] > block_bounds(iblock)[0]]
    for iblock, block in zip(produced, output):
        idx = block_indices(iblock)
        if idx is None:
            dexc_dbf += block
        else:
            dexc_dbf[:, idx] += block
    numba.set_num_threads(ncores)
    durations['grad'] = timer() - start

    if debug:
        print('Skala gradient timings (s): density %.3f, model %.3f, contraction %.3f'
              % (durations['rho'], durations['model'], durations['grad']), flush=True)

    return dexc_dbf, explicit_grad


def _block_grad(coords_block, dmat, vrho_block, vgrad_block, vtau_block, bfs, non_zero_indices):
    """Per-basis-function gradient contributions of one grid block.

    This is the meta-GGA branch of :func:`pyfock.Integrals.eval_xc_grad_2.block_xc_grad_func` with the
    functional evaluation replaced by the model's cotangents, which already include the quadrature
    weights: ``F = dE/drho``, ``(Fx, Fy, Fz) = dE/d(grad rho)`` and ``G_tau = 0.5 * dE/dtau``.
    """
    numba.set_num_threads(1)
    ao, ao_grad, ao_hess = Integrals.bf_val_helpers.eval_bfs_grad_and_hess_sparse_internal_serial(
        bfs[0], bfs[1], bfs[2], bfs[3], bfs[4], bfs[5], bfs[6], coords_block, non_zero_indices)

    Fmj = ao @ dmat                                     # (chi D)[g, mu]
    Hgrad = [ao_grad[0] @ dmat, ao_grad[1] @ dmat, ao_grad[2] @ dmat]

    Fx, Fy, Fz = vgrad_block[0], vgrad_block[1], vgrad_block[2]

    # aow[g, nu] = F chi_nu + sum_k Fk d_k chi_nu
    aow = vrho_block[:, None] * ao
    aow += Fx[:, None] * ao_grad[0]
    aow += Fy[:, None] * ao_grad[1]
    aow += Fz[:, None] * ao_grad[2]
    aowD = aow @ dmat

    res = np.empty((3, ao.shape[1]))
    res[0] = np.einsum('mj,mj->j', ao_grad[0], aowD)
    res[1] = np.einsum('mj,mj->j', ao_grad[1], aowD)
    res[2] = np.einsum('mj,mj->j', ao_grad[2], aowD)

    # Hessian terms: sum_k Fk * d_k d_d chi_mu contracted with (chi D).
    # ao_hess components are ordered 0:xx 1:xy 2:xz 3:yy 4:yz 5:zz.
    hessF = Fx[:, None] * ao_hess[0] + Fy[:, None] * ao_hess[1] + Fz[:, None] * ao_hess[2]
    res[0] += np.einsum('mj,mj->j', hessF, Fmj)
    hessF = Fx[:, None] * ao_hess[1] + Fy[:, None] * ao_hess[3] + Fz[:, None] * ao_hess[4]
    res[1] += np.einsum('mj,mj->j', hessF, Fmj)
    hessF = Fx[:, None] * ao_hess[2] + Fy[:, None] * ao_hess[4] + Fz[:, None] * ao_hess[5]
    res[2] += np.einsum('mj,mj->j', hessF, Fmj)

    # tau term: sum_k G_tau (d_d d_k chi_mu) Hk[g, mu]
    Gtau = 0.5 * vtau_block
    GHx = Gtau[:, None] * Hgrad[0]
    GHy = Gtau[:, None] * Hgrad[1]
    GHz = Gtau[:, None] * Hgrad[2]
    res[0] += (np.einsum('mj,mj->j', ao_hess[0], GHx) + np.einsum('mj,mj->j', ao_hess[1], GHy)
               + np.einsum('mj,mj->j', ao_hess[2], GHz))
    res[1] += (np.einsum('mj,mj->j', ao_hess[1], GHx) + np.einsum('mj,mj->j', ao_hess[3], GHy)
               + np.einsum('mj,mj->j', ao_hess[4], GHz))
    res[2] += (np.einsum('mj,mj->j', ao_hess[2], GHx) + np.einsum('mj,mj->j', ao_hess[4], GHy)
               + np.einsum('mj,mj->j', ao_hess[5], GHz))
    return res
