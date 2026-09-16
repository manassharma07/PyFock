"""XC energy and potential from the Skala neural functional (GPU).

The same three passes as :mod:`pyfock.Integrals.eval_xc_skala`, with the AO evaluation, the density
contractions and the potential assembly on the device:

1. **density** -- loop the grid blocks, evaluate the AOs and their gradients with the CUDA kernel and
   accumulate ``rho``, ``grad rho`` and ``tau`` over the whole grid;
2. **functional** -- one call into the model, which already runs on the GPU when the SCF does;
3. **potential** -- loop the same blocks again and contract the returned derivatives with the AOs.

Skala is non-local over each atomic grid, so it cannot be folded into the per-block loop of
:func:`pyfock.Integrals.eval_xc_3_cupy`, where every block calls the functional on its own points and
returns a partial energy. Everything else is that function's code: the AO kernel, the ``Fmj`` density
contraction and the meta-GGA potential assembly are the same, with the ``weights_block`` factors absent
because the derivatives Skala returns already carry the quadrature weights.

Pass 2 hands the density to the model through the host. That looks wasteful next to a DLPack
hand-off, but the arrays are five per grid point -- 12 MB for a 295k-point grid, a few milliseconds --
against several seconds of model evaluation, and going through NumPy keeps one code path in
:meth:`pyfock.XC.SkalaFunctional.exc_and_potential` for both devices.
"""

import numpy as np
from opt_einsum import contract
from timeit import default_timer as timer

try:
    import cupy as cp
    from numba import cuda
except ImportError:
    cp = None

from pyfock import Integrals

__all__ = ['eval_xc_skala_cupy']


def _bfs_arrays_cupy(basis, dtype):
    """Basis data as the device arrays the CUDA AO kernels take (as in eval_xc_3_cupy)."""
    maxnprim = max(basis.bfs_nprim)
    bfs_coeffs = cp.zeros([basis.bfs_nao, maxnprim], dtype=dtype)
    bfs_expnts = cp.zeros([basis.bfs_nao, maxnprim], dtype=dtype)
    bfs_prim_norms = cp.zeros([basis.bfs_nao, maxnprim], dtype=dtype)
    bfs_radius_cutoff = cp.zeros([basis.bfs_nao], dtype=dtype)
    for i in range(basis.bfs_nao):
        for j in range(basis.bfs_nprim[i]):
            bfs_coeffs[i, j] = basis.bfs_coeffs[i][j]
            bfs_expnts[i, j] = basis.bfs_expnts[i][j]
            bfs_prim_norms[i, j] = basis.bfs_prim_norms[i][j]
            bfs_radius_cutoff[i] = basis.bfs_radius_cutoff[i]
    return [cp.asarray(basis.bfs_coords, dtype=dtype), cp.asarray(basis.bfs_contr_prim_norms, dtype=dtype),
            cp.asarray(basis.bfs_nprim), cp.asarray(basis.bfs_lmn),
            bfs_coeffs, bfs_prim_norms, bfs_expnts, bfs_radius_cutoff]


def _block_aos_cupy(bfs, coords_block, non_zero_indices, ao_values, ao_grad_values,
                    threads_per_block, nb_stream, dtype):
    """AO values and Cartesian gradients on one block, from the cache when the caller provided one."""
    if ao_values is not None:
        return cp.asarray(ao_values, dtype=dtype), cp.asarray(ao_grad_values, dtype=dtype)

    npoints = coords_block.shape[0]
    thread_x, thread_y = threads_per_block
    if non_zero_indices is not None:
        nbfs = non_zero_indices.shape[0]
        ao = cp.zeros((npoints, nbfs), dtype=dtype)
        ao_grad = cp.zeros((3, npoints, nbfs), dtype=dtype)
        blocks_per_grid = ((nbfs + (thread_x - 1)) // thread_x, (npoints + (thread_y - 1)) // thread_y)
        Integrals.bf_val_helpers.eval_bfs_and_grad_sparse_internal_cuda[
            blocks_per_grid, threads_per_block, nb_stream](
                bfs[0], bfs[1], bfs[2], bfs[3], bfs[4], bfs[5], bfs[6],
                coords_block, non_zero_indices, ao, ao_grad)
        return ao, ao_grad
    # Only the sparse AO kernel has a CUDA version, so the device path needs the screening indices.
    raise ValueError('eval_xc_skala_cupy needs the AO screening indices: only the sparse AO kernel '
                     'has a CUDA version. Leave xc_bf_screen at its default (True).')


def eval_xc_skala_cupy(basis, dmat, grids, skala, blocksize=20480, list_nonzero_indices=None,
                       count_nonzero_indices=None, list_ao_values=None, list_ao_grad_values=None,
                       max_points_per_chunk=250000, print_nelec=False, debug=False,
                       threads_per_block=None, dtype=None):
    """Skala's XC energy and potential matrix on the GPU.

    Parameters
    ----------
    basis : Basis
        Basis set of the calculation.
    dmat : (nao, nao) cupy.ndarray
        Density matrix in the AO basis, on the device.
    grids : Grids
        Integration grid. Must be a native ``'treutler'`` grid: Skala needs ``atom_idx`` and the
        unpartitioned ``atomic_weights`` alongside the Becke-partitioned ``weights``.
    skala : SkalaFunctional
        The loaded model, ideally on the same device (``XC.load_skala(name, use_gpu=True)``).
    blocksize : int
        Grid points per block in passes 1 and 3.
    list_nonzero_indices, count_nonzero_indices : list, optional
        Per-block indices of the significantly contributing basis functions (AO screening).
    list_ao_values, list_ao_grad_values : list, optional
        Cached AO values and gradients per block, which skip the evaluation in both passes.
    max_points_per_chunk : int
        Upper bound on the grid points the model evaluates at once.

    Returns
    -------
    efunc : float
        Total XC energy in Hartree.
    v : (nao, nao) cupy.ndarray
        XC potential matrix, on the device.
    """
    if cp is None:
        raise ImportError('eval_xc_skala_cupy needs CuPy and numba.cuda; use eval_xc_skala on the CPU.')
    if grids.atomic_weights is None or grids.atom_idx is None:
        raise ValueError("Skala needs a native 'treutler' grid: it reads both the Becke-partitioned "
                         'weights and the raw single-atom weights, which the other schemes do not keep.')

    dtype = cp.float64 if dtype is None else dtype
    if dtype != cp.float64:
        raise ValueError('Skala is a float64 model; the XC term cannot run in reduced precision. '
                         'Set dynamic_precision=False.')

    if threads_per_block is None:
        max_threads = cuda.get_current_device().MAX_THREADS_PER_BLOCK
        threads_per_block = (max_threads // 16, max_threads // 64)

    # Launch the AO kernel on whatever stream CuPy is using, so the contractions below are ordered
    # after it. Handing the kernel a stream of its own instead leaves the reads racing the writes, and
    # the SCF then wanders off by ~2e-2 Ha to a slightly different place on every run.
    stream_ptr = cp.cuda.get_current_stream().ptr
    nb_stream = cuda.default_stream() if stream_ptr == 0 else cuda.external_stream(stream_ptr)

    coords = cp.asarray(grids.coords, dtype=dtype)
    weights = cp.asarray(grids.weights, dtype=dtype)
    ngrids = coords.shape[0]
    nblocks = ngrids // blocksize
    nao = basis.bfs_nao
    bfs = _bfs_arrays_cupy(basis, dtype)
    dmat = cp.asarray(dmat, dtype=dtype)

    # The screening indices arrive as host arrays the first time; the CUDA kernel wants them on device.
    if list_nonzero_indices is not None:
        list_nonzero_indices = [cp.asarray(block) for block in list_nonzero_indices]

    durations = {'rho': 0.0, 'model': 0.0, 'potential': 0.0}

    # ------------------------------------------------------------------ pass 1: density over the grid
    start = timer()
    rho = cp.zeros(ngrids, dtype=dtype)
    rho_grad = cp.zeros((3, ngrids), dtype=dtype)
    tau = cp.zeros(ngrids, dtype=dtype)

    for iblock in range(nblocks + 1):
        lo = iblock * blocksize
        hi = min(lo + blocksize, ngrids)
        if hi <= lo:
            continue
        idx = None
        if list_nonzero_indices is not None:
            idx = list_nonzero_indices[iblock][0:count_nonzero_indices[iblock]]
        ao, ao_grad = _block_aos_cupy(bfs, coords[lo:hi], idx,
                                      None if list_ao_values is None else list_ao_values[iblock],
                                      None if list_ao_grad_values is None else list_ao_grad_values[iblock],
                                      threads_per_block, nb_stream, dtype)
        dmat_block = dmat if idx is None else dmat[cp.ix_(idx, idx)]

        Fmj = ao @ dmat_block
        rho[lo:hi] = cp.sum(Fmj * ao, axis=1)
        rho_grad[:, lo:hi] = 2 * contract('mj,kmj->km', Fmj, ao_grad, backend='cupy')
        tau[lo:hi] = 0.5 * contract('ij,kmi,kmj->m', dmat_block, ao_grad, ao_grad, backend='cupy')
    durations['rho'] = timer() - start

    if print_nelec:
        print('Number of electrons: ', float(cp.dot(rho, weights)))

    # ------------------------------------------------------------------ pass 2: the model
    # Through the host: five arrays per grid point, a few milliseconds against seconds of model time,
    # and exc_and_potential then needs no separate device code path.
    start = timer()
    efunc, vrho, vgrad, vtau = skala.exc_and_potential(
        cp.asnumpy(rho), cp.asnumpy(rho_grad), cp.asnumpy(tau), cp.asnumpy(coords),
        cp.asnumpy(weights), grids.atomic_weights, grids.atom_idx,
        np.asarray(grids.mol.coordsBohrs, dtype=np.float64).reshape(-1, 3),
        max_points_per_chunk=max_points_per_chunk)
    vrho = cp.asarray(vrho, dtype=dtype)
    vgrad = cp.asarray(vgrad, dtype=dtype)
    vtau = cp.asarray(vtau, dtype=dtype)
    durations['model'] = timer() - start

    # ------------------------------------------------------------------ pass 3: the potential matrix
    start = timer()
    v = cp.zeros((nao, nao), dtype=dtype)
    for iblock in range(nblocks + 1):
        lo = iblock * blocksize
        hi = min(lo + blocksize, ngrids)
        if hi <= lo:
            continue
        idx = None
        if list_nonzero_indices is not None:
            idx = list_nonzero_indices[iblock][0:count_nonzero_indices[iblock]]
        ao, ao_grad = _block_aos_cupy(bfs, coords[lo:hi], idx,
                                      None if list_ao_values is None else list_ao_values[iblock],
                                      None if list_ao_grad_values is None else list_ao_grad_values[iblock],
                                      threads_per_block, nb_stream, dtype)

        # The quadrature weights are already inside the derivatives the model returned, so this is the
        # meta-GGA contraction of eval_xc_3_cupy with the weights_block factors dropped.
        z = 0.5 * vrho[lo:hi] * ao.T
        z = z + (vgrad[0, lo:hi] * ao_grad[0].T + vgrad[1, lo:hi] * ao_grad[1].T
                 + vgrad[2, lo:hi] * ao_grad[2].T)
        v_temp = z @ ao
        v_block = v_temp + v_temp.T
        v_block += contract('m,kmi,kmj->ij', 0.5 * vtau[lo:hi], ao_grad, ao_grad, backend='cupy')

        if idx is None:
            v += v_block
        else:
            v[cp.ix_(idx, idx)] = v_block + v[cp.ix_(idx, idx)]
    durations['potential'] = timer() - start

    if debug:
        print('Skala timings (s): density %.3f, model %.3f, potential %.3f'
              % (durations['rho'], durations['model'], durations['potential']), flush=True)

    return efunc, v
