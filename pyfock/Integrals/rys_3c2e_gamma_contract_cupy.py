"""GPU (Numba-CUDA) contraction of the 3c2e integrals with the density matrix.

    gamma_P = sum_ij D_ij (ij|P)

This is the first half of the density-fitting step the analytical gradient needs: the fitting
coefficients are c = (P|Q)^-1 gamma. The CPU gradient gets gamma by building the (nbf, nbf, naux)
integral tensor in chunks and contracting it; here the contraction happens inside the integral
kernel, so nothing of that size is ever allocated.

The screening matches :func:`rys_3c2e_symm` with ``schwarz=True`` -- a pure integral-magnitude
bound, no density weighting -- so the GPU and CPU fitting coefficients agree to round-off.
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
from .rys_3c2e_symm import _pack_basis
from .cuda_stream import gradient_stream

__all__ = ['rys_3c2e_gamma_contract_cupy']

_COMB = np.array([[float(math.comb(n, k)) if k <= n else 0.0 for k in range(16)]
                  for n in range(16)], dtype=np.float64)

_KERNEL_CACHE = {}


def _get_kernel(g_rows, g_cols, nroots_max, max_nbf_c):
    key = (g_rows, g_cols, nroots_max, max_nbf_c)
    kernel = _KERNEL_CACHE.get(key)
    if kernel is not None:
        return kernel

    G_ROWS = g_rows
    G_COLS = g_cols
    NROOTS = nroots_max
    MAX_NBF_C = max_nbf_c

    @cuda.jit(fastmath=True, cache=False)
    def _kernel(task_list, nshells_aux, ab_shell_a, ab_shell_b,
                bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim,
                bfs_coeffs, bfs_prim_norms, bfs_expnts,
                shell_l, shell_bfs_offset, bfs_nbfshell,
                aux_bfs_coords, aux_bfs_contr_prim_norms, aux_bfs_lmn, aux_bfs_nprim,
                aux_bfs_coeffs, aux_bfs_prim_norms, aux_bfs_expnts,
                aux_shell_l, aux_shell_bfs_offset, aux_bfs_nbfshell,
                dmat, DATA_X_, DATA_W_, gamma):
        comb = cuda.const.array_like(_COMB)

        itask = cuda.grid(1)
        if itask >= task_list.shape[0]:
            return

        task = task_list[itask]
        ab_idx = task // nshells_aux
        ksh = task - ab_idx * nshells_aux
        ish = ab_shell_a[ab_idx]
        jsh = ab_shell_b[ab_idx]

        bf_a_start = shell_bfs_offset[ish]
        bf_b_start = shell_bfs_offset[jsh]
        bf_c_start = aux_shell_bfs_offset[ksh]
        nbf_a = bfs_nbfshell[ish]
        nbf_b = bfs_nbfshell[jsh]
        nbf_c = aux_bfs_nbfshell[ksh]

        la_shell = shell_l[ish]
        lb_shell = shell_l[jsh]
        lc_shell = aux_shell_l[ksh]
        bra_order = la_shell + lb_shell
        aux_order = lc_shell
        nroots = (la_shell + lb_shell + lc_shell) // 2 + 1

        nprim_a = bfs_nprim[bf_a_start]
        nprim_b = bfs_nprim[bf_b_start]
        nprim_c = aux_bfs_nprim[bf_c_start]

        ax0 = bfs_coords[bf_a_start, 0]
        ay0 = bfs_coords[bf_a_start, 1]
        az0 = bfs_coords[bf_a_start, 2]
        bx0 = bfs_coords[bf_b_start, 0]
        by0 = bfs_coords[bf_b_start, 1]
        bz0 = bfs_coords[bf_b_start, 2]
        cx0 = aux_bfs_coords[bf_c_start, 0]
        cy0 = aux_bfs_coords[bf_c_start, 1]
        cz0 = aux_bfs_coords[bf_c_start, 2]

        xij0 = ax0 - bx0
        xij1 = ay0 - by0
        xij2 = az0 - bz0
        ijsq = xij0 * xij0 + xij1 * xij1 + xij2 * xij2

        roots = cuda.local.array(NROOTS, numba.float64)
        weights = cuda.local.array(NROOTS, numba.float64)
        gx = cuda.local.array((G_ROWS, G_COLS), numba.float64)
        gy = cuda.local.array((G_ROWS, G_COLS), numba.float64)
        gz = cuda.local.array((G_ROWS, G_COLS), numba.float64)
        acc = cuda.local.array(MAX_NBF_C, numba.float64)
        for ic in range(nbf_c):
            acc[ic] = 0.0

        pair_factor = 2.0 if ish != jsh else 1.0
        pi = 3.141592653589793

        for iprim_a in range(nprim_a):
            alpha = bfs_expnts[bf_a_start, iprim_a]
            for iprim_b in range(nprim_b):
                beta = bfs_expnts[bf_b_start, iprim_b]
                gamma_p = alpha + beta
                inv_gamma_p = 1.0 / gamma_p
                # 1e-8, not the 1e-10 the derivative kernels use: this reproduces the primitive
                # screen of rys_3c2e_symm, which is how the CPU gradient builds the same gamma.
                if math.exp(-alpha * beta * inv_gamma_p * ijsq) < 1.0e-8:
                    continue

                px = (alpha * ax0 + beta * bx0) * inv_gamma_p
                py = (alpha * ay0 + beta * by0) * inv_gamma_p
                pz = (alpha * az0 + beta * bz0) * inv_gamma_p
                pqx = px - cx0
                pqy = py - cy0
                pqz = pz - cz0
                pqsq = pqx * pqx + pqy * pqy + pqz * pqz

                for iprim_c in range(nprim_c):
                    gamma_q = aux_bfs_expnts[bf_c_start, iprim_c]
                    rho = gamma_p * gamma_q / (gamma_p + gamma_q)
                    x = rho * pqsq
                    gamma_pq_sqrt = math.sqrt(gamma_p * gamma_q)

                    Roots(nroots, x, DATA_X_, DATA_W_, roots, weights)
                    rys_prefactor = 2.0 * math.sqrt(rho / pi)

                    for iroot in range(nroots):
                        root = roots[iroot]
                        Recur_3c2e(gx, root, bra_order, 0, aux_order, 0,
                                   ax0, bx0, cx0, 0.0, alpha, beta, gamma_q, 0.0,
                                   gamma_p, gamma_q, alpha * beta, gamma_pq_sqrt)
                        Recur_3c2e(gy, root, bra_order, 0, aux_order, 0,
                                   ay0, by0, cy0, 0.0, alpha, beta, gamma_q, 0.0,
                                   gamma_p, gamma_q, alpha * beta, gamma_pq_sqrt)
                        Recur_3c2e(gz, root, bra_order, 0, aux_order, 0,
                                   az0, bz0, cz0, 0.0, alpha, beta, gamma_q, 0.0,
                                   gamma_p, gamma_q, alpha * beta, gamma_pq_sqrt)

                        root_weight = rys_prefactor * weights[iroot]

                        for ia in range(nbf_a):
                            ibf_a = bf_a_start + ia
                            axl = bfs_lmn[ibf_a, 0]
                            ayl = bfs_lmn[ibf_a, 1]
                            azl = bfs_lmn[ibf_a, 2]
                            ca = (bfs_contr_prim_norms[ibf_a] * bfs_coeffs[ibf_a, iprim_a]
                                  * bfs_prim_norms[ibf_a, iprim_a])
                            for ib in range(nbf_b):
                                ibf_b = bf_b_start + ib
                                bxl = bfs_lmn[ibf_b, 0]
                                byl = bfs_lmn[ibf_b, 1]
                                bzl = bfs_lmn[ibf_b, 2]
                                cb = (ca * bfs_contr_prim_norms[ibf_b]
                                      * bfs_coeffs[ibf_b, iprim_b] * bfs_prim_norms[ibf_b, iprim_b])
                                dm_ab = pair_factor * dmat[ibf_a, ibf_b] * cb * root_weight
                                for ic in range(nbf_c):
                                    ibf_c = bf_c_start + ic
                                    cxl = aux_bfs_lmn[ibf_c, 0]
                                    cyl = aux_bfs_lmn[ibf_c, 1]
                                    czl = aux_bfs_lmn[ibf_c, 2]
                                    w = (dm_ab * aux_bfs_contr_prim_norms[ibf_c]
                                         * aux_bfs_coeffs[ibf_c, iprim_c]
                                         * aux_bfs_prim_norms[ibf_c, iprim_c])

                                    sx = 0.0
                                    for n in range(bxl + 1):
                                        sx += comb[bxl, n] * xij0 ** (bxl - n) * gx[n + axl, cxl]
                                    sy = 0.0
                                    for n in range(byl + 1):
                                        sy += comb[byl, n] * xij1 ** (byl - n) * gy[n + ayl, cyl]
                                    sz = 0.0
                                    for n in range(bzl + 1):
                                        sz += comb[bzl, n] * xij2 ** (bzl - n) * gz[n + azl, czl]

                                    acc[ic] += w * sx * sy * sz

        for ic in range(nbf_c):
            cuda.atomic.add(gamma, bf_c_start + ic, acc[ic])

    _KERNEL_CACHE[key] = _kernel
    return _kernel


_TASK_CHUNK = 32 * 1024 * 1024


def rys_3c2e_gamma_contract_cupy(basis, auxbasis, dmat, threshold_schwarz=1e-9,
                                 sqrt_ints4c2e_diag=None, sqrt_diag_ints2c2e=None,
                                 cp_stream=None):
    """Return ``gamma_P = sum_ij D_ij (ij|P)`` as a NumPy ``(naux,)`` array."""
    if cp is None:
        raise RuntimeError('CuPy is required for rys_3c2e_gamma_contract_cupy.')

    (bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms,
     bfs_expnts, shell_l, shell_bfs_offset, bfs_nbfshell) = _pack_basis(basis)
    (aux_bfs_coords, aux_bfs_contr_prim_norms, aux_bfs_lmn, aux_bfs_nprim, aux_bfs_coeffs,
     aux_bfs_prim_norms, aux_bfs_expnts, aux_shell_l, aux_shell_bfs_offset,
     aux_bfs_nbfshell) = _pack_basis(auxbasis)

    nshells = len(basis.shells)
    nshells_aux = len(auxbasis.shells)
    naux = auxbasis.bfs_nao

    max_l_bra = int(shell_l.max())
    max_l_aux = int(aux_shell_l.max())
    nroots_max = (2 * max_l_bra + max_l_aux) // 2 + 1
    if nroots_max > 10:
        raise NotImplementedError('rys_3c2e_gamma_contract_cupy supports Rys orders up to 10 only.')
    g_rows = 2 * max_l_bra + 1
    g_cols = max_l_aux + 1
    max_nbf_c = int(aux_bfs_nbfshell.max())

    dmat = np.ascontiguousarray(dmat, dtype=np.float64)

    if sqrt_ints4c2e_diag is None:
        from .schwarz_helpers import eri_4c2e_diag
        sqrt_ints4c2e_diag = np.sqrt(np.abs(eri_4c2e_diag(basis)))
    if sqrt_diag_ints2c2e is None:
        from .rys_2c2e_diag import rys_2c2e_diag
        sqrt_diag_ints2c2e = np.sqrt(np.abs(rys_2c2e_diag(auxbasis)))

    ab_shell_a = np.empty(nshells * (nshells + 1) // 2, dtype=np.int32)
    ab_shell_b = np.empty_like(ab_shell_a)
    idx = 0
    for ish in range(nshells):
        for jsh in range(ish + 1):
            ab_shell_a[idx] = ish
            ab_shell_b[idx] = jsh
            idx += 1
    n_ab = ab_shell_a.shape[0]

    from .rys_3c2e_grad_contract import _shell_bounds
    shell_pair_bound, aux_shell_bound, _, _ = _shell_bounds(
        nshells, nshells_aux, shell_bfs_offset, bfs_nbfshell,
        aux_shell_bfs_offset, aux_bfs_nbfshell,
        np.asarray(sqrt_ints4c2e_diag), np.asarray(sqrt_diag_ints2c2e),
        np.abs(dmat), np.ones(naux))
    pair_bound = shell_pair_bound[ab_shell_a, ab_shell_b]

    if cp_stream is None:
        cp_stream, nb_stream = gradient_stream()
    else:
        nb_stream = cuda.external_stream(cp_stream.ptr)

    with cp_stream:
        args = [cp.asarray(a) for a in (ab_shell_a, ab_shell_b, bfs_coords, bfs_contr_prim_norms,
                                        bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts,
                                        shell_l, shell_bfs_offset, bfs_nbfshell,
                                        aux_bfs_coords, aux_bfs_contr_prim_norms, aux_bfs_lmn,
                                        aux_bfs_nprim, aux_bfs_coeffs, aux_bfs_prim_norms,
                                        aux_bfs_expnts, aux_shell_l, aux_shell_bfs_offset,
                                        aux_bfs_nbfshell, dmat)]
        data_x, data_w = cp.asarray(DATA_X), cp.asarray(DATA_W)
        gamma = cp.zeros(naux)
        kernel = _get_kernel(g_rows, g_cols, nroots_max, max_nbf_c)

        pair_bound_d = cp.asarray(pair_bound)
        aux_bound_d = cp.asarray(aux_shell_bound)
        lab_key = cp.asarray((shell_l[ab_shell_a].astype(np.int64) * 8
                              + shell_l[ab_shell_b].astype(np.int64)) * 8)
        lc_key = cp.asarray(aux_shell_l.astype(np.int64))

        threads = 64
        chunk = max(1, min(n_ab, _TASK_CHUNK // max(1, nshells_aux)))
        for lo in range(0, n_ab, chunk):
            hi = min(lo + chunk, n_ab)
            keep = pair_bound_d[lo:hi, None] * aux_bound_d[None, :] >= threshold_schwarz
            flat = cp.flatnonzero(keep.ravel())
            if flat.size == 0:
                continue
            key = lab_key[lo + flat // nshells_aux] + lc_key[flat % nshells_aux]
            task_list = flat[cp.argsort(key, kind='stable')] + lo * nshells_aux
            blocks = (task_list.shape[0] + threads - 1) // threads
            kernel[blocks, threads, nb_stream](
                task_list, nshells_aux, args[0], args[1], args[2], args[3], args[4], args[5],
                args[6], args[7], args[8], args[9], args[10], args[11],
                args[12], args[13], args[14], args[15], args[16], args[17], args[18],
                args[19], args[20], args[21], args[22], data_x, data_w, gamma)
        gamma_host = cp.asnumpy(gamma)
    cp_stream.synchronize()
    return gamma_host
