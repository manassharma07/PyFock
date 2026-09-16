"""GPU (Numba-CUDA) contracted 2c2e nuclear gradient -- the metric part of the DF Coulomb force.

Device port of :func:`pyfock.Integrals.rys_2c2e_grad_contract`:

    grad[iatom, xyz] = sum_{PQ} c_P c_Q d(P|Q)/dR_{iatom, xyz}

One thread per auxiliary function pair (P, Q) with Q < P; the diagonal carries no net force and the
derivative with respect to the second center follows from translational invariance.
"""
import math

import numpy as np
import numba
from numba import cuda

try:
    import cupy as cp
except Exception:                                  # pragma: no cover - CPU-only install
    cp = None

from .rys_helpers_cuda import Recur_3c2e, Roots, DATA_X, DATA_W
from .cuda_stream import gradient_stream

__all__ = ['rys_2c2e_grad_contract_cupy']

_NBINS = 64
_KERNEL_CACHE = {}


def _get_kernel(g_rows, g_cols, nroots_max):
    key = (g_rows, g_cols, nroots_max)
    kernel = _KERNEL_CACHE.get(key)
    if kernel is not None:
        return kernel

    G_ROWS = g_rows
    G_COLS = g_cols
    NROOTS = nroots_max

    @cuda.jit(fastmath=True, cache=False)
    def _kernel(pair_i, pair_k, bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim,
                bfs_coeffs, bfs_prim_norms, bfs_expnts, bfs_atoms, df_coeff,
                threshold, DATA_X_, DATA_W_, grad_partial):
        idx = cuda.grid(1)
        if idx >= pair_i.shape[0]:
            return

        i = pair_i[idx]
        k = pair_k[idx]
        atom_i = bfs_atoms[i]
        atom_k = bfs_atoms[k]

        cik = df_coeff[i] * df_coeff[k]
        if abs(cik) < threshold:
            return

        la = bfs_lmn[i, 0]
        ma = bfs_lmn[i, 1]
        na = bfs_lmn[i, 2]
        lc = bfs_lmn[k, 0]
        mc = bfs_lmn[k, 1]
        nc = bfs_lmn[k, 2]

        norder = (la + ma + na + 1 + lc + mc + nc) // 2 + 1

        ix = bfs_coords[i, 0]
        iy = bfs_coords[i, 1]
        iz = bfs_coords[i, 2]
        kx = bfs_coords[k, 0]
        ky = bfs_coords[k, 1]
        kz = bfs_coords[k, 2]
        pqx = ix - kx
        pqy = iy - ky
        pqz = iz - kz
        pqsq = pqx * pqx + pqy * pqy + pqz * pqz

        roots = cuda.local.array(NROOTS, numba.float64)
        weights = cuda.local.array(NROOTS, numba.float64)
        G = cuda.local.array((G_ROWS, G_COLS), numba.float64)

        gax = 0.0
        gay = 0.0
        gaz = 0.0
        pi = 3.141592653589793
        tempcoeff1 = bfs_contr_prim_norms[i] * bfs_contr_prim_norms[k]

        for ik in range(bfs_nprim[i]):
            alpha = bfs_expnts[i, ik]
            two_alpha = 2.0 * alpha
            tempcoeff2 = tempcoeff1 * bfs_coeffs[i, ik] * bfs_prim_norms[i, ik]
            gamma_p = alpha

            for kk in range(bfs_nprim[k]):
                gamma_q = bfs_expnts[k, kk]
                tempcoeff3 = tempcoeff2 * bfs_coeffs[k, kk] * bfs_prim_norms[k, kk]

                rho = gamma_p * gamma_q / (gamma_p + gamma_q)
                x = rho * pqsq
                gamma_pq_sqrt = math.sqrt(gamma_p * gamma_q)

                Roots(norder, x, DATA_X_, DATA_W_, roots, weights)
                rys_prefactor = 2.0 * math.sqrt(rho / pi) * tempcoeff3

                for iroot in range(norder):
                    root = roots[iroot]
                    root_weight = rys_prefactor * weights[iroot]

                    # (P|Q) is a 3c2e integral with a dummy s function at the bra: alpha_j = 0, so
                    # the second bra center is irrelevant and I is passed for it (xij = 0).
                    Recur_3c2e(G, root, la + 1, 0, lc, 0, ix, ix, kx, 0.0,
                               alpha, 0.0, gamma_q, 0.0, gamma_p, gamma_q, 0.0, gamma_pq_sqrt)
                    sx = G[la, lc]
                    dax = two_alpha * G[la + 1, lc]
                    if la > 0:
                        dax -= la * G[la - 1, lc]

                    Recur_3c2e(G, root, ma + 1, 0, mc, 0, iy, iy, ky, 0.0,
                               alpha, 0.0, gamma_q, 0.0, gamma_p, gamma_q, 0.0, gamma_pq_sqrt)
                    sy = G[ma, mc]
                    day = two_alpha * G[ma + 1, mc]
                    if ma > 0:
                        day -= ma * G[ma - 1, mc]

                    Recur_3c2e(G, root, na + 1, 0, nc, 0, iz, iz, kz, 0.0,
                               alpha, 0.0, gamma_q, 0.0, gamma_p, gamma_q, 0.0, gamma_pq_sqrt)
                    sz = G[na, nc]
                    daz = two_alpha * G[na + 1, nc]
                    if na > 0:
                        daz -= na * G[na - 1, nc]

                    gax += root_weight * dax * sy * sz
                    gay += root_weight * sx * day * sz
                    gaz += root_weight * sx * sy * daz

        # Both (P|Q) and (Q|P) appear in the double sum, so weight by 2.
        gx = 2.0 * cik * gax
        gy = 2.0 * cik * gay
        gz = 2.0 * cik * gaz

        ibin = cuda.blockIdx.x % _NBINS
        cuda.atomic.add(grad_partial, (ibin, atom_i, 0), gx)
        cuda.atomic.add(grad_partial, (ibin, atom_i, 1), gy)
        cuda.atomic.add(grad_partial, (ibin, atom_i, 2), gz)
        # Translational invariance: d/dQ = -d/dP
        cuda.atomic.add(grad_partial, (ibin, atom_k, 0), -gx)
        cuda.atomic.add(grad_partial, (ibin, atom_k, 1), -gy)
        cuda.atomic.add(grad_partial, (ibin, atom_k, 2), -gz)

    _KERNEL_CACHE[key] = _kernel
    return _kernel


def rys_2c2e_grad_contract_cupy(auxbasis, df_coeff, threshold=1e-14, cp_stream=None):
    """GPU counterpart of :func:`rys_2c2e_grad_contract`. Returns a NumPy ``(natoms, 3)`` array."""
    if cp is None:
        raise RuntimeError('CuPy is required for rys_2c2e_grad_contract_cupy.')

    naux = auxbasis.bfs_nao
    bfs_coords = np.array(auxbasis.bfs_coords, dtype=np.float64)
    bfs_contr_prim_norms = np.array(auxbasis.bfs_contr_prim_norms, dtype=np.float64)
    bfs_lmn = np.array(auxbasis.bfs_lmn, dtype=np.int32)
    bfs_nprim = np.array(auxbasis.bfs_nprim, dtype=np.int32)
    bfs_atoms = np.array(auxbasis.bfs_atoms, dtype=np.int32)
    natoms = int(bfs_atoms.max()) + 1

    maxnprim = int(bfs_nprim.max())
    bfs_coeffs = np.zeros((naux, maxnprim))
    bfs_expnts = np.zeros((naux, maxnprim))
    bfs_prim_norms = np.zeros((naux, maxnprim))
    for i in range(naux):
        for j in range(auxbasis.bfs_nprim[i]):
            bfs_coeffs[i, j] = auxbasis.bfs_coeffs[i][j]
            bfs_expnts[i, j] = auxbasis.bfs_expnts[i][j]
            bfs_prim_norms[i, j] = auxbasis.bfs_prim_norms[i][j]

    l_tot = bfs_lmn.sum(axis=1)
    max_l = int(l_tot.max())
    nroots_max = (2 * max_l + 1) // 2 + 1
    if nroots_max > 10:
        raise NotImplementedError('rys_2c2e_grad_contract_cupy supports Rys orders up to 10 only.')
    g_rows = max_l + 2       # the bra recursion runs to la+1
    g_cols = max_l + 1

    df_coeff = np.ascontiguousarray(df_coeff, dtype=np.float64)

    # Strictly lower triangle, minus the pairs on a common atom (no net force), sorted by the
    # (l_bra, l_ket) pair so that a warp stays on one recursion shape.
    tri_i, tri_k = np.tril_indices(naux, k=-1)
    keep = bfs_atoms[tri_i] != bfs_atoms[tri_k]
    tri_i = tri_i[keep].astype(np.int32)
    tri_k = tri_k[keep].astype(np.int32)
    order = np.argsort(l_tot[tri_i] * (max_l + 1) + l_tot[tri_k], kind='stable')
    tri_i = np.ascontiguousarray(tri_i[order])
    tri_k = np.ascontiguousarray(tri_k[order])

    if cp_stream is None:
        cp_stream, nb_stream = gradient_stream()
    else:
        nb_stream = cuda.external_stream(cp_stream.ptr)

    with cp_stream:
        args = [cp.asarray(a) for a in (tri_i, tri_k, bfs_coords, bfs_contr_prim_norms, bfs_lmn,
                                        bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts,
                                        bfs_atoms, df_coeff)]
        grad_partial = cp.zeros((_NBINS, natoms, 3), dtype=cp.float64)
        kernel = _get_kernel(g_rows, g_cols, nroots_max)
        threads = 64
        blocks = (tri_i.shape[0] + threads - 1) // threads
        if blocks > 0:
            kernel[blocks, threads, nb_stream](*args, float(threshold),
                                               cp.asarray(DATA_X), cp.asarray(DATA_W), grad_partial)
        grad = cp.asnumpy(grad_partial.sum(axis=0))
    cp_stream.synchronize()
    return grad
