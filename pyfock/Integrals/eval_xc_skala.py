"""XC energy and potential from the Skala neural functional (CPU).

Skala cannot be folded into the blocked loop of :func:`pyfock.Integrals.eval_xc_2`, because it is not a
pointwise functional: its non-local layers aggregate over the points of a whole atomic grid, so the
density everywhere must be known before the energy anywhere can be evaluated. This module therefore runs
three passes over the grid instead of one:

1. **density** -- loop the grid blocks and accumulate ``rho``, ``grad rho`` and ``tau`` over the whole grid;
2. **functional** -- one call into the model (:meth:`pyfock.XC.SkalaFunctional.exc_and_potential`), which
   returns the total XC energy and ``dE/drho``, ``dE/d(grad rho)`` and ``dE/dtau`` at every grid point;
3. **potential** -- loop the same blocks again and contract those derivatives with the AO values.

The third pass is the meta-GGA contraction of :func:`pyfock.Integrals.eval_xc_2` unchanged; the
derivatives Skala returns already carry the quadrature weights, so the ``weights_block`` factors of the
semilocal path are simply absent here.

The AO values are evaluated twice unless the caller supplies cached ones through ``list_ao_values`` and
``list_ao_grad_values``, which makes the XC step roughly 1.3-1.4x more expensive than a meta-GGA of the
same grid on top of the cost of the model itself. That is the price of non-locality and the reason the
implementation stays this simple.
"""

import numpy as np
from opt_einsum import contract
from timeit import default_timer as timer

import numba

from pyfock import Integrals

__all__ = ['eval_xc_skala']


def _bfs_arrays(basis):
    """Basis data as the flat NumPy arrays the Numba AO kernels take (as in eval_xc_2)."""
    maxnprim = max(basis.bfs_nprim)
    bfs_coeffs = np.zeros([basis.bfs_nao, maxnprim])
    bfs_expnts = np.zeros([basis.bfs_nao, maxnprim])
    bfs_prim_norms = np.zeros([basis.bfs_nao, maxnprim])
    bfs_radius_cutoff = np.zeros([basis.bfs_nao])
    for i in range(basis.bfs_nao):
        for j in range(basis.bfs_nprim[i]):
            bfs_coeffs[i, j] = basis.bfs_coeffs[i][j]
            bfs_expnts[i, j] = basis.bfs_expnts[i][j]
            bfs_prim_norms[i, j] = basis.bfs_prim_norms[i][j]
            bfs_radius_cutoff[i] = basis.bfs_radius_cutoff[i]
    return [np.array(basis.bfs_coords), np.array(basis.bfs_contr_prim_norms), np.array(basis.bfs_nprim),
            np.array(basis.bfs_lmn), bfs_coeffs, bfs_prim_norms, bfs_expnts, bfs_radius_cutoff]


def _block_aos(bfs, coords_block, non_zero_indices, ao_values, ao_grad_values):
    """AO values and Cartesian gradients on one block, from the cache when the caller provided one."""
    if ao_values is not None:
        return ao_values, ao_grad_values
    if non_zero_indices is not None:
        return Integrals.bf_val_helpers.eval_bfs_and_grad_sparse_internal(
            bfs[0], bfs[1], bfs[2], bfs[3], bfs[4], bfs[5], bfs[6], coords_block, non_zero_indices)
    return Integrals.bf_val_helpers.eval_bfs_and_grad_internal(
        bfs[0], bfs[1], bfs[2], bfs[3], bfs[4], bfs[5], bfs[6], bfs[7], coords_block)


def eval_xc_skala(basis, dmat, grids, skala, ncores=2, blocksize=5000,
                  list_nonzero_indices=None, count_nonzero_indices=None,
                  list_ao_values=None, list_ao_grad_values=None,
                  max_points_per_chunk=250000, debug=False, print_nelec=True):
    """Exchange-correlation energy and potential matrix from a Skala functional.

    Parameters
    ----------
    basis : Basis
        Basis set of the calculation.
    dmat : (nao, nao) ndarray
        Total (spin-summed) density matrix in the AO basis. Closed shell only.
    grids : Grids
        Molecular grid. Must carry :attr:`~pyfock.Grids.Grids.atomic_weights`, i.e. it must have been
        built with the native ``scheme='treutler'``; Skala needs both the Becke-partitioned and the
        unpartitioned quadrature weights.
    skala : SkalaFunctional
        Loaded functional, from :func:`pyfock.XC.load_skala`.
    ncores : int
        Threads for the Numba AO kernels.
    blocksize : int
        Grid points per block in the two AO passes.
    list_nonzero_indices, count_nonzero_indices : list, optional
        Per-block indices of the AOs that are non-negligible, as built by the SCF driver.
    list_ao_values, list_ao_grad_values : list, optional
        Cached per-block AO values and gradients; supplying them removes the second AO evaluation.
    max_points_per_chunk : int
        Upper bound on the grid points handed to the model at once.
    debug : bool
        Print a breakdown of the time spent in the three passes.
    print_nelec : bool
        Print the number of electrons recovered by integrating the density over the grid.

    Returns
    -------
    efunc : float
        Exchange-correlation energy in Hartree.
    v : (nao, nao) ndarray
        Exchange-correlation potential matrix.
    """
    if getattr(grids, 'atomic_weights', None) is None:
        raise ValueError(
            "Skala needs the unpartitioned single-atom quadrature weights, which the 'numgrid' grid "
            "scheme does not expose. Build the grid with the native scheme instead, e.g. "
            "Grids(mol, level=3) or Grids(mol, scheme='treutler', level=3).")

    coords = grids.coords
    weights = grids.weights
    ngrids = coords.shape[0]
    nblocks = ngrids // blocksize
    nao = basis.bfs_nao
    bfs = _bfs_arrays(basis)
    numba.set_num_threads(ncores)

    durations = {'rho': 0.0, 'model': 0.0, 'potential': 0.0}

    # ------------------------------------------------------------------ pass 1: density over the grid
    start = timer()
    rho = np.zeros(ngrids)
    rho_grad = np.zeros((3, ngrids))
    tau = np.zeros(ngrids)

    for iblock in range(nblocks + 1):
        lo = iblock * blocksize
        hi = min(lo + blocksize, ngrids)
        if hi <= lo:
            continue
        idx = None
        if list_nonzero_indices is not None:
            idx = list_nonzero_indices[iblock][0:count_nonzero_indices[iblock]]
        ao, ao_grad = _block_aos(bfs, coords[lo:hi], idx,
                                 None if list_ao_values is None else list_ao_values[iblock],
                                 None if list_ao_grad_values is None else list_ao_grad_values[iblock])
        dmat_block = dmat if idx is None else dmat[np.ix_(idx, idx)]

        Fmj = ao @ dmat_block
        rho[lo:hi] = contract('mj,mj->m', Fmj, ao)
        rho_grad[:, lo:hi] = 2 * contract('mj,kmj->km', Fmj, ao_grad)
        tau[lo:hi] = 0.5 * contract('ij,kmi,kmj->m', dmat_block, ao_grad, ao_grad)
    durations['rho'] = timer() - start

    nelec = float(np.dot(rho, weights))
    if print_nelec:
        print('Number of electrons: ', nelec)

    # ------------------------------------------------------------------ pass 2: the model
    start = timer()
    efunc, vrho, vgrad, vtau = skala.exc_and_potential(
        rho, rho_grad, tau, coords, weights, grids.atomic_weights, grids.atom_idx,
        np.asarray(grids.mol.coordsBohrs, dtype=np.float64).reshape(-1, 3),
        max_points_per_chunk=max_points_per_chunk)
    durations['model'] = timer() - start

    # ------------------------------------------------------------------ pass 3: the potential matrix
    start = timer()
    v = np.zeros((nao, nao))
    for iblock in range(nblocks + 1):
        lo = iblock * blocksize
        hi = min(lo + blocksize, ngrids)
        if hi <= lo:
            continue
        idx = None
        if list_nonzero_indices is not None:
            idx = list_nonzero_indices[iblock][0:count_nonzero_indices[iblock]]
        ao, ao_grad = _block_aos(bfs, coords[lo:hi], idx,
                                 None if list_ao_values is None else list_ao_values[iblock],
                                 None if list_ao_grad_values is None else list_ao_grad_values[iblock])

        # The quadrature weights are already inside the derivatives the model returned, so this is the
        # meta-GGA contraction of eval_xc_2 with the weights_block factors dropped.
        z = 0.5 * vrho[lo:hi] * ao.T
        z = z + (vgrad[0, lo:hi] * ao_grad[0].T + vgrad[1, lo:hi] * ao_grad[1].T
                 + vgrad[2, lo:hi] * ao_grad[2].T)
        v_temp = z @ ao
        v_block = v_temp + v_temp.T
        v_block += contract('m,kmi,kmj->ij', 0.5 * vtau[lo:hi], ao_grad, ao_grad)

        if idx is None:
            v += v_block
        else:
            v[np.ix_(idx, idx)] += v_block
    durations['potential'] = timer() - start

    if debug:
        print('Skala timings (s): density %.3f, model %.3f, potential %.3f'
              % (durations['rho'], durations['model'], durations['potential']), flush=True)

    return efunc, v
