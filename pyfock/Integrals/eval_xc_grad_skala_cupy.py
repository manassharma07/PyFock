"""Nuclear gradient of the Skala neural exchange-correlation functional (GPU).

Device counterpart of :func:`pyfock.Integrals.eval_xc_grad_skala`; that module documents the four
terms and why the grid-response ones cannot be dropped. The structure is unchanged -- density pass,
one model call, contraction pass -- but the two AO passes run on the GPU.

The model call itself is left alone: it is a PyTorch forward/backward that already runs on whatever
device the functional was loaded on (``XC.load_skala(..., use_gpu=True)``), and it wants its inputs
in its own atom-major layout, so the per-point quantities make one round trip through the host
between the passes. They are a handful of length-G vectors, negligible next to the passes.
"""
import numpy as np
from timeit import default_timer as timer

try:
    import cupy as cp
except Exception:                                  # pragma: no cover - CPU-only install
    cp = None
from numba import cuda

from . import bf_val_helpers
from .cuda_stream import gradient_stream

__all__ = ['eval_xc_grad_skala_cupy']

# ao_hess components are stored as 0:xx 1:xy 2:xz 3:yy 4:yz 5:zz.
_HESS = ((0, 1, 2), (1, 3, 4), (2, 4, 5))


def _pack_basis_arrays(basis):
    maxnprim = max(basis.bfs_nprim)
    coeffs = np.zeros((basis.bfs_nao, maxnprim))
    expnts = np.zeros((basis.bfs_nao, maxnprim))
    prim_norms = np.zeros((basis.bfs_nao, maxnprim))
    for i in range(basis.bfs_nao):
        for j in range(basis.bfs_nprim[i]):
            coeffs[i, j] = basis.bfs_coeffs[i][j]
            expnts[i, j] = basis.bfs_expnts[i][j]
            prim_norms[i, j] = basis.bfs_prim_norms[i][j]
    return (cp.asarray(np.array(basis.bfs_coords)),
            cp.asarray(np.array(basis.bfs_contr_prim_norms)),
            cp.asarray(np.array(basis.bfs_nprim)),
            cp.asarray(np.array(basis.bfs_lmn)),
            cp.asarray(coeffs), cp.asarray(prim_norms), cp.asarray(expnts))


def _sum_prod(a, b):
    return cp.sum(a * b, axis=1)


def _col_sum_prod(a, b):
    return cp.sum(a * b, axis=0)


def eval_xc_grad_skala_cupy(basis, dmat, grids, skala, ncores=2, blocksize=20480,
                            list_nonzero_indices=None, count_nonzero_indices=None,
                            max_points_per_chunk=250000, grid_response=True, debug=False,
                            cp_stream=None):
    """GPU counterpart of :func:`eval_xc_grad_skala`; same arguments and return convention."""
    if cp is None:
        raise RuntimeError('CuPy is required for eval_xc_grad_skala_cupy.')
    if getattr(grids, 'atomic_weights', None) is None:
        raise ValueError(
            "Skala needs the unpartitioned single-atom quadrature weights, which the 'numgrid' grid "
            "scheme does not expose. Build the grid with the native scheme instead.")

    if cp_stream is None:
        cp_stream, nb_stream = gradient_stream()
    else:
        nb_stream = cuda.external_stream(cp_stream.ptr)

    coords_host = np.asarray(grids.coords)
    weights_host = np.asarray(grids.weights)
    ngrids = coords_host.shape[0]
    nblocks = ngrids // blocksize
    nao = basis.bfs_nao
    atom_idx = np.ascontiguousarray(grids.atom_idx, dtype=np.int64)
    atom_coords = np.asarray(grids.mol.coordsBohrs, dtype=np.float64).reshape(-1, 3)
    natm = atom_coords.shape[0]
    durations = {}
    thread_x, thread_y = 8, 32

    def block_range(iblock):
        lo = iblock * blocksize
        return lo, min(lo + blocksize, ngrids)

    with cp_stream:
        bfs = _pack_basis_arrays(basis)
        coords_d = cp.asarray(coords_host)
        dmat_d = cp.asarray(np.ascontiguousarray(dmat, dtype=np.float64))
        atom_idx_d = cp.asarray(atom_idx)

        block_info = []
        for iblock in range(nblocks + 1):
            lo, hi = block_range(iblock)
            if hi <= lo:
                continue
            if list_nonzero_indices is None:
                nz = cp.arange(nao)
                dmat_block = dmat_d
            else:
                nz = cp.asarray(list_nonzero_indices[iblock][0:count_nonzero_indices[iblock]])
                if nz.shape[0] == 0:
                    continue
                dmat_block = cp.ascontiguousarray(dmat_d[cp.ix_(nz, nz)])
            block_info.append((lo, hi, nz, dmat_block))

        # ------------------------------------------------------ pass 1: density on the grid
        start = timer()
        rho = cp.zeros(ngrids)
        rho_grad = cp.zeros((3, ngrids))
        tau = cp.zeros(ngrids)
        for lo, hi, nz, dmat_block in block_info:
            npts = hi - lo
            nbf_block = nz.shape[0]
            ao = cp.empty((npts, nbf_block))
            ao_grad = cp.empty((3, npts, nbf_block))
            grid_dim = ((nbf_block + thread_x - 1) // thread_x, (npts + thread_y - 1) // thread_y)
            bf_val_helpers.eval_bfs_and_grad_sparse_internal_cuda[
                grid_dim, (thread_x, thread_y), nb_stream](
                bfs[0], bfs[1], bfs[2], bfs[3], bfs[4], bfs[5], bfs[6],
                coords_d[lo:hi], nz, ao, ao_grad)
            Fmj = ao @ dmat_block
            rho[lo:hi] = _sum_prod(Fmj, ao)
            for k in range(3):
                rho_grad[k, lo:hi] = 2.0 * _sum_prod(Fmj, ao_grad[k])
            tau[lo:hi] = 0.5 * sum(_sum_prod(ao_grad[k], ao_grad[k] @ dmat_block)
                                   for k in range(3))
        rho_host = cp.asnumpy(rho)
        rho_grad_host = cp.asnumpy(rho_grad)
        tau_host = cp.asnumpy(tau)
        del rho, tau
    cp_stream.synchronize()
    durations['rho'] = timer() - start

    # ------------------------------------------------------ pass 2: one model call
    start = timer()
    _, vrho, vgrad, vtau, atom_grad, dweights = skala.exc_and_potential(
        rho_host, rho_grad_host, tau_host, coords_host, weights_host, grids.atomic_weights,
        atom_idx, atom_coords, max_points_per_chunk=max_points_per_chunk, nuclear_terms=True)
    durations['model'] = timer() - start
    if not grid_response:
        atom_grad = np.zeros_like(atom_grad)

    # ------------------------------------------------------ grid response: dE/dw . dw/dR
    start = timer()
    if grid_response:
        from pyfock.Grids import becke_weight_gradient, size_adjustment_table
        # the charges the grid was built with (the element, also for ghost and ECP atoms)
        a_table = size_adjustment_table(grids.charges, getattr(grids, 'size_adjustment', 'treutler'))
        atom_grad = atom_grad + becke_weight_gradient(coords_host, atom_idx, atom_coords, a_table,
                                                      dweights * grids.atomic_weights)
    durations['weights'] = timer() - start

    # ------------------------------------------------------ pass 3: AO contraction
    start = timer()
    with cp_stream:
        vrho_d = cp.asarray(vrho)
        vgrad_d = cp.asarray(vgrad)
        vtau_d = cp.asarray(vtau)
        dexc_dbf = cp.zeros((3, nao))
        translation = cp.zeros((natm, 3))

        for lo, hi, nz, dmat_block in block_info:
            npts = hi - lo
            nbf_block = nz.shape[0]
            # empty, not zeros: the AO kernels write every element (see eval_xc_grad_2_cupy).
            ao = cp.empty((npts, nbf_block))
            ao_grad = cp.empty((3, npts, nbf_block))
            ao_hess = cp.empty((6, npts, nbf_block))
            grid_dim = ((nbf_block + thread_x - 1) // thread_x, (npts + thread_y - 1) // thread_y)
            bf_val_helpers.eval_bfs_grad_and_hess_sparse_internal_cuda[
                grid_dim, (thread_x, thread_y), nb_stream](
                bfs[0], bfs[1], bfs[2], bfs[3], bfs[4], bfs[5], bfs[6],
                coords_d[lo:hi], nz, ao, ao_grad, ao_hess)

            Fmj = ao @ dmat_block
            Hgrad = [ao_grad[k] @ dmat_block for k in range(3)]
            vrho_b = vrho_d[lo:hi]
            vgrad_b = vgrad_d[:, lo:hi]
            vtau_b = vtau_d[lo:hi]

            # ---- 1. Pulay. The cotangents already carry the quadrature weights, so unlike the
            #         semilocal path there is no weights_block factor here.
            aow = vrho_b[:, None] * ao
            for k in range(3):
                aow += vgrad_b[k][:, None] * ao_grad[k]
            aowD = aow @ dmat_block

            res = cp.zeros((3, nbf_block))
            Gtau = 0.5 * vtau_b
            GH = [Gtau[:, None] * Hgrad[k] for k in range(3)]
            for d in range(3):
                res[d] = _col_sum_prod(ao_grad[d], aowD)
                hessF = (vgrad_b[0][:, None] * ao_hess[_HESS[d][0]]
                         + vgrad_b[1][:, None] * ao_hess[_HESS[d][1]]
                         + vgrad_b[2][:, None] * ao_hess[_HESS[d][2]])
                res[d] += _col_sum_prod(hessF, Fmj)
                res[d] += (_col_sum_prod(ao_hess[_HESS[d][0]], GH[0])
                           + _col_sum_prod(ao_hess[_HESS[d][1]], GH[1])
                           + _col_sum_prod(ao_hess[_HESS[d][2]], GH[2]))

            if list_nonzero_indices is None:
                dexc_dbf += res
            else:
                dexc_dbf[:, nz] += res

            if not grid_response:
                continue

            # ---- 2. Grid translation: the points of an atom move with it, so the density features
            #         evaluated there change even at fixed density.
            rho_grad_b = cp.asarray(rho_grad_host[:, lo:hi])
            atom_idx_b = atom_idx_d[lo:hi]
            for b in range(3):
                term = vrho_b * rho_grad_b[b]
                for a in range(3):
                    hessian_ab = 2.0 * (_sum_prod(ao_hess[_HESS[a][b]], Fmj)
                                        + _sum_prod(ao_grad[a], Hgrad[b]))
                    term = term + vgrad_b[a] * hessian_ab
                grad_tau = (_sum_prod(ao_hess[_HESS[b][0]], Hgrad[0])
                            + _sum_prod(ao_hess[_HESS[b][1]], Hgrad[1])
                            + _sum_prod(ao_hess[_HESS[b][2]], Hgrad[2]))
                term = term + vtau_b * grad_tau
                translation[:, b] += cp.bincount(atom_idx_b, weights=term, minlength=natm)

        dexc_dbf_host = cp.asnumpy(dexc_dbf)
        atom_grad = atom_grad + cp.asnumpy(translation)
    cp_stream.synchronize()
    durations['grad'] = timer() - start

    if debug:
        print('Skala gradient timings (s): density %.3f, model %.3f, weights %.3f, contraction %.3f'
              % (durations['rho'], durations['model'], durations['weights'], durations['grad']),
              flush=True)
    return dexc_dbf_host, atom_grad
