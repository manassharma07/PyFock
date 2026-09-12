"""
GPU (CuPy / Numba-CUDA) counterpart of :mod:`pyfock.Integrals.df_algo10_helpers`
for the default density-fitted Coulomb algorithm (``DF_algo=10``).

The four steps mirror the CPU module and share its sparse storage layout and
screening rules, so ``offsets`` computed with the CPU
:func:`~pyfock.Integrals.df_algo10_helpers.calc_offsets_3c2e_schwarz` are used
directly here:

1. :func:`eri_4c2e_diag_cupy` - Schwarz diagonal ``(ij|ij)`` on the device.
2. :func:`rys_3c2e_tri_schwarz_sparse_algo10_cupy` - screened ``(ij|P)`` (CAO or SAO).
3. :func:`df_coeff_calculator_algo10_cupy` - ``gamma_P = sum_ij D_ij (ij|P)``.
4. :func:`J_tri_calculator_algo10_cupy` - ``J_ij = sum_P (ij|P) c_P``.

Kernel layout: one CUDA thread per AO pair ``(i, j)`` with ``j <= i``
(``linear_index = i*(i+1)//2 + j`` addresses ``offsets``).  The integral kernels
put the primitive loops outermost and accumulate into the output, so Rys roots
are computed once per (primitive pair, aux shell) and reused for all Cartesian
components of that shell.

SAO: the auxiliary shells with ``l >= 2`` are projected onto their spherical
subspace with ``proj = pinv(C) C`` (see the CPU module).  As on the CPU, the
Schwarz bounds passed in must be shell-constant (``aux_shell_max_bounds``) so that
a shell is skipped, evaluated, projected and stored as a whole - the same set that
:func:`calc_offsets_3c2e_schwarz` counted.  The projector tables are placed in
CUDA constant memory; shells up to ``l = 4`` (g) are supported here.
"""

import math
import numpy as np
import numba
from numba import cuda

try:
    import cupy as cp
except Exception:  # CuPy is optional; the module must still import on CPU-only machines
    cp = None

from .rys_helpers_cuda import coulomb_rys, coulomb_rys_3c2e, Roots, Roots_5, DATA_X, DATA_W
from .df_algo10_helpers import (
    STRICT_PAIR_CUTOFF, pack_basis_arrays, aux_shell_arrays, sao_aux_projectors,
    check_sao_bounds_are_shell_constant)

__all__ = [
    'eri_4c2e_diag_cupy',
    'rys_3c2e_tri_schwarz_sparse_algo10_cupy',
    'df_coeff_calculator_algo10_cupy',
    'J_tri_calculator_algo10_cupy',
]

#: Largest auxiliary angular momentum handled by the SAO CUDA kernel.
MAX_L_AUX_SAO_CUDA = 4

# Spherical-subspace projectors for d, f and g aux shells (constant memory in the kernel).
_PROJ = sao_aux_projectors(MAX_L_AUX_SAO_CUDA)
PROJ_D = np.ascontiguousarray(_PROJ[2, :6, :6])
PROJ_F = np.ascontiguousarray(_PROJ[3, :10, :10])
PROJ_G = np.ascontiguousarray(_PROJ[4, :15, :15])


def _stream_pair(cp_stream):
    """Return (cp_stream, numba external stream), creating a non-blocking stream if needed."""
    if cp_stream is None:
        cp.cuda.Device(0).use()
        cp_stream = cp.cuda.Stream(non_blocking=True)
    nb_stream = cuda.external_stream(cp_stream.ptr)
    cp_stream.use()
    return cp_stream, nb_stream


def _to_device(arrays):
    return tuple(cp.asarray(a) for a in arrays)


# ----------------------------------------------------------------------------
# 1. Schwarz diagonal (ij|ij)
# ----------------------------------------------------------------------------
def eri_4c2e_diag_cupy(basis, cp_stream=None):
    """``(ij|ij)`` for all AO pairs, as a (nao, nao) CuPy array (used for Schwarz screening)."""
    (bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim,
     bfs_coeffs, bfs_prim_norms, bfs_expnts) = _to_device(pack_basis_arrays(basis))
    DATA_X_cuda = cp.asarray(DATA_X)
    DATA_W_cuda = cp.asarray(DATA_W)
    nao = basis.bfs_nao
    fourC2E_diag = cp.zeros((nao, nao), dtype=cp.float64)

    cp_stream, nb_stream = _stream_pair(cp_stream)
    thread_x = 32
    thread_y = 32
    blocks_per_grid = ((nao + (thread_x - 1)) // thread_x, (nao + (thread_y - 1)) // thread_y)
    rys_eri_4c2e_diag_internal_cuda[blocks_per_grid, (thread_x, thread_y), nb_stream](
        bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts,
        DATA_X_cuda, DATA_W_cuda, fourC2E_diag)
    cp_stream.synchronize()
    cp.cuda.Stream.null.synchronize()
    return fourC2E_diag


@cuda.jit(fastmath=True, cache=True, max_registers=50)
def rys_eri_4c2e_diag_internal_cuda(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts, DATA_X, DATA_W, out):
    # "Diagonal" elements of the 4c2e ERI tensor, out[i,j] = (ij|ij), by Rys quadrature.
    nao = bfs_coords.shape[0]
    i, j = cuda.grid(2)

    if i < nao and j <= i:
        IJ = cuda.local.array((3), numba.float64)
        P = cuda.local.array((3), numba.float64)
        PQ = cuda.local.array((3), numba.float64)
        I = bfs_coords[i]
        Ni = bfs_contr_prim_norms[i]
        lmni = bfs_lmn[i]
        la, ma, na = lmni
        nprimi = bfs_nprim[i]

        K = I
        lc, mc, nc = lmni

        J = bfs_coords[j]
        IJ[0] = I[0] - J[0]
        IJ[1] = I[1] - J[1]
        IJ[2] = I[2] - J[2]
        L = J
        Nj = bfs_contr_prim_norms[j]
        lmnj = bfs_lmn[j]
        lb, mb, nb = lmnj
        tempcoeff3 = (Ni * Nj) ** 2
        nprimj = bfs_nprim[j]

        ld, md, nd = lmnj

        norder = int((la + ma + na + lb + mb + nb + lc + mc + nc + ld + md + nd) / 2 + 1)
        n = int(max(la + lb, ma + mb, na + nb))
        m = int(max(lc + ld, mc + md, nc + nd))
        roots = cuda.local.array((10), numba.float64)
        weights = cuda.local.array((10), numba.float64)
        G = cuda.local.array((13, 13), numba.float64)
        val = 0.0

        for ik in range(nprimi):
            dik = bfs_coeffs[i][ik]
            Nik = bfs_prim_norms[i][ik]
            alphaik = bfs_expnts[i][ik]
            tempcoeff4 = tempcoeff3 * (dik * Nik) ** 2

            for jk in range(nprimj):
                alphajk = bfs_expnts[j][jk]
                gammaP = alphaik + alphajk
                djk = bfs_coeffs[j][jk]
                Njk = bfs_prim_norms[j][jk]
                P[0] = (alphaik * I[0] + alphajk * J[0]) / gammaP
                P[1] = (alphaik * I[1] + alphajk * J[1]) / gammaP
                P[2] = (alphaik * I[2] + alphajk * J[2]) / gammaP

                tempcoeff5 = tempcoeff4 * (djk * Njk) ** 2

                gammaQ = gammaP
                dlk = djk
                Nlk = Njk
                Q = P
                PQ[0] = P[0] - Q[0]
                PQ[1] = P[1] - Q[1]
                PQ[2] = P[2] - Q[2]
                PQsq = PQ[0] ** 2 + PQ[1] ** 2 + PQ[2] ** 2

                tempcoeff6 = tempcoeff5 * dlk * Nlk

                if norder <= 10:
                    rho = gammaP * gammaQ / (gammaP + gammaQ)
                    X = PQsq * rho
                    roots, weights = Roots(norder, X, DATA_X, DATA_W, roots, weights)
                    val += tempcoeff6 * coulomb_rys(roots, weights, G, PQsq, rho, norder, n, m, la, lb, lc, ld, ma, mb, mc, md, na, nb, nc, nd, alphaik, alphajk, alphaik, alphajk, I, J, K, L)

        out[i, j] = val
        out[j, i] = val


# ----------------------------------------------------------------------------
# 2. Screened three-center integrals in sparse triangular storage
# ----------------------------------------------------------------------------
def rys_3c2e_tri_schwarz_sparse_algo10_cupy(basis, auxbasis, indicesA, indicesB, offsets,
                                            sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold,
                                            strict_schwarz, nsignificant, sao=False, cp_stream=None):
    """
    GPU evaluation of the Schwarz-screened ``(ij|P)`` in the sparse layout of the
    CPU module.  ``indicesA``/``indicesB`` must be the full lower triangle (the
    kernel addresses ``offsets`` by ``i*(i+1)//2 + j``).  Returns a CuPy array.

    With ``sao=True`` the auxiliary d/f/g shells are projected onto their
    spherical subspaces; shells with ``l > 4`` are not supported on the GPU.
    """
    (bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim,
     bfs_coeffs, bfs_prim_norms, bfs_expnts) = _to_device(pack_basis_arrays(basis))
    (aux_bfs_coords, aux_bfs_contr_prim_norms, aux_bfs_lmn, aux_bfs_nprim,
     aux_bfs_coeffs, aux_bfs_prim_norms, aux_bfs_expnts) = _to_device(pack_basis_arrays(auxbasis))
    aux_shell_indices = cp.asarray(np.asarray(auxbasis.bfs_shell_index, dtype=np.int64))

    sqrt_ints4c2e_diag = cp.asarray(sqrt_ints4c2e_diag)
    sqrt_diag_ints2c2e = cp.asarray(sqrt_diag_ints2c2e)
    offsets = cp.asarray(offsets)
    threeC2E = cp.zeros(int(nsignificant), dtype=cp.float64)

    cp_stream, nb_stream = _stream_pair(cp_stream)
    nao = basis.bfs_nao
    naux = auxbasis.bfs_nao
    thread_x = 8
    thread_y = 8
    blocks_per_grid = ((nao + (thread_x - 1)) // thread_x, (nao + (thread_y - 1)) // thread_y)

    if sao:
        aux_shell_bfs_offset, aux_shell_nbf, aux_shell_l = aux_shell_arrays(auxbasis)
        if int(aux_shell_l.max()) > MAX_L_AUX_SAO_CUDA:
            raise ValueError('The GPU SAO 3c2e kernel supports auxiliary shells up to l=4 (g).')
        check_sao_bounds_are_shell_constant(cp.asnumpy(sqrt_diag_ints2c2e), aux_shell_bfs_offset, aux_shell_nbf)
        rys_3c2e_tri_schwarz_sparse_algo10_sao_internal_cuda[blocks_per_grid, (thread_x, thread_y), nb_stream](
            bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts,
            aux_bfs_coords, aux_bfs_contr_prim_norms, aux_bfs_lmn, aux_bfs_nprim,
            aux_bfs_coeffs, aux_bfs_prim_norms, aux_bfs_expnts,
            0, nao, 0, nao, 0, naux,
            sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold, offsets,
            strict_schwarz, aux_shell_indices, threeC2E)
    else:
        DATA_X_cuda = cp.asarray(DATA_X)
        DATA_W_cuda = cp.asarray(DATA_W)
        rys_3c2e_tri_schwarz_sparse_algo10_internal_cuda[blocks_per_grid, (thread_x, thread_y), nb_stream](
            bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts,
            aux_bfs_coords, aux_bfs_contr_prim_norms, aux_bfs_lmn, aux_bfs_nprim,
            aux_bfs_coeffs, aux_bfs_prim_norms, aux_bfs_expnts,
            0, nao, 0, nao, 0, naux, DATA_X_cuda, DATA_W_cuda,
            sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold, offsets, strict_schwarz,
            aux_shell_indices, threeC2E)

    cp_stream.synchronize()
    cp.cuda.Stream.null.synchronize()
    cp._default_memory_pool.free_all_blocks()
    return threeC2E


@cuda.jit(fastmath=True, cache=True, max_registers=128)
def rys_3c2e_tri_schwarz_sparse_algo10_internal_cuda(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts, aux_bfs_coords, aux_bfs_contr_prim_norms, aux_bfs_lmn, aux_bfs_nprim, aux_bfs_coeffs, aux_bfs_prim_norms, aux_bfs_expnts, indx_startA, indx_endA, indx_startB, indx_endB, indx_startC, indx_endC, DATA_X, DATA_W, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, schwarz_threshold, offsets, strict_schwarz, aux_shell_indices, out):
    # Cartesian auxiliary functions: one thread per AO pair (i, j), j <= i.
    # Only aux functions passing the function-level Schwarz test are evaluated
    # and stored, in increasing k, matching calc_offsets_3c2e_schwarz.
    i, j = cuda.grid(2)

    L = cuda.local.array((3), numba.float64)
    L[0] = 0.0
    L[1] = 0.0
    L[2] = 0.0
    ld, md, nd = int(0), int(0), int(0)
    alphalk = 0.0

    if i >= indx_startA and i < indx_endA and j >= indx_startB and j < indx_endB and (j <= i):
        sqrt_ints4c2e_diag_ij = sqrt_ints4c2e_diag[i, j]
        if strict_schwarz:
            if sqrt_ints4c2e_diag_ij * sqrt_ints4c2e_diag_ij < STRICT_PAIR_CUTOFF:
                return
        linear_index = j + i * (i + 1) // 2
        offset_ = offsets[linear_index]
        IJ = cuda.local.array((3), numba.float64)
        P = cuda.local.array((3), numba.float64)
        PQ = cuda.local.array((3), numba.float64)
        I = bfs_coords[i]
        Ni = bfs_contr_prim_norms[i]
        lmni = bfs_lmn[i]
        la, ma, na = lmni
        nprimi = bfs_nprim[i]

        J = bfs_coords[j]
        IJ[0] = I[0] - J[0]
        IJ[1] = I[1] - J[1]
        IJ[2] = I[2] - J[2]
        IJsq = IJ[0] ** 2 + IJ[1] ** 2 + IJ[2] ** 2
        Nj = bfs_contr_prim_norms[j]
        lmnj = bfs_lmn[j]
        lb, mb, nb = lmnj
        tempcoeff1 = Ni * Nj
        nprimj = bfs_nprim[j]

        # norder = (2L + Laux)/2 + 1 = L + Laux/2 + 1
        G = cuda.local.array((5, 5), numba.float64)  # Good for upto g auxshells and d shells;
        roots = cuda.local.array((7, 5), numba.float64)  # Good for upto g auxshells and d shells; and 7 primitives per bf
        weights = cuda.local.array((7, 5), numba.float64)  # Good for upto g auxshells and d shells; and 7 primitives per bf

        # Loop over primitives
        for ik in range(nprimi):
            dik = bfs_coeffs[i][ik]
            Nik = bfs_prim_norms[i][ik]
            alphaik = bfs_expnts[i][ik]
            tempcoeff2 = tempcoeff1 * dik * Nik

            for jk in range(nprimj):
                alphajk = bfs_expnts[j][jk]
                gammaP = alphaik + alphajk
                prod_alphaikjk = alphaik * alphajk
                screenfactorAB = math.exp(-prod_alphaikjk / gammaP * IJsq)
                if abs(screenfactorAB) < 1.0e-8:
                    continue
                djk = bfs_coeffs[j][jk]
                Njk = bfs_prim_norms[j][jk]
                P[0] = (alphaik * I[0] + alphajk * J[0]) / gammaP
                P[1] = (alphaik * I[1] + alphajk * J[1]) / gammaP
                P[2] = (alphaik * I[2] + alphajk * J[2]) / gammaP
                tempcoeff3 = tempcoeff2 * djk * Njk

                index_k = 0
                shell_index_previous = -1
                for k in range(indx_startC, indx_endC):
                    if sqrt_ints4c2e_diag_ij * sqrt_diag_ints2c2e[k] <= schwarz_threshold:
                        continue
                    shell_index = aux_shell_indices[k]

                    K = aux_bfs_coords[k]
                    Nk = aux_bfs_contr_prim_norms[k]
                    lmnk = aux_bfs_lmn[k]
                    lc, mc, nc = lmnk
                    tempcoeff4 = tempcoeff3 * Nk
                    nprimk = aux_bfs_nprim[k]

                    Q = K
                    PQ[0] = P[0] - Q[0]
                    PQ[1] = P[1] - Q[1]
                    PQ[2] = P[2] - Q[2]
                    PQsq = PQ[0] ** 2 + PQ[1] ** 2 + PQ[2] ** 2

                    norder = int((la + ma + na + lb + mb + nb + lc + mc + nc + ld + md + nd) / 2 + 1)
                    val = 0.0

                    if norder <= 10:
                        n = int(max(la + lb, ma + mb, na + nb))
                        m = int(max(lc + ld, mc + md, nc + nd))

                        for kk in range(nprimk):
                            dkk = aux_bfs_coeffs[k][kk]
                            Nkk = aux_bfs_prim_norms[k][kk]
                            alphakk = aux_bfs_expnts[k][kk]
                            tempcoeff5 = tempcoeff4 * dkk * Nkk
                            ABsrt = math.sqrt(gammaP * alphakk)

                            gammaQ = alphakk
                            rho = gammaP * gammaQ / (gammaP + gammaQ)

                            X = PQsq * rho
                            roots_kk = roots[kk, :]
                            weights_kk = weights[kk, :]
                            # Rys roots depend only on the aux shell (same center/exponents),
                            # so they are computed once per shell and reused for its components.
                            if shell_index != shell_index_previous:
                                roots_kk, weights_kk = Roots(norder, X, DATA_X, DATA_W, roots_kk, weights_kk)

                            val += tempcoeff5 * coulomb_rys_3c2e(roots_kk, weights_kk, G, PQsq, rho, norder, n, m, la, lb, lc, ld, ma, mb, mc, md, na, nb, nc, nd, alphaik, alphajk, alphakk, alphalk, I, J, K, L, IJ, P, prod_alphaikjk, gammaP, ABsrt)

                    shell_index_previous = shell_index
                    out[offset_ + index_k] += val
                    index_k += 1


@cuda.jit(fastmath=True, cache=True, max_registers=128)
def rys_3c2e_tri_schwarz_sparse_algo10_sao_internal_cuda(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts, aux_bfs_coords, aux_bfs_contr_prim_norms, aux_bfs_lmn, aux_bfs_nprim, aux_bfs_coeffs, aux_bfs_prim_norms, aux_bfs_expnts, indx_startA, indx_endA, indx_startB, indx_endB, indx_startC, indx_endC, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, schwarz_threshold, offsets, strict_schwarz, aux_shell_indices, out):
    # Spherical auxiliary functions (pseudo-Cartesian storage): one thread per AO pair.
    # sqrt_diag_ints2c2e must be shell-constant (aux_shell_max_bounds), so the Schwarz
    # test decides per shell: a d/f/g shell is either skipped completely or evaluated
    # completely, projected with proj_d/f/g and stored completely - the same set that
    # calc_offsets_3c2e_schwarz counted, and every stored shell block stays inside the
    # spherical subspace as the eps-regularized metric requires.
    # Uses the fixed 5-root Rys tables (Roots_5): orbital shells up to d, aux up to g.
    i, j = cuda.grid(2)

    proj_d = cuda.const.array_like(PROJ_D)
    proj_f = cuda.const.array_like(PROJ_F)
    proj_g = cuda.const.array_like(PROJ_G)

    L = cuda.local.array((3), numba.float64)
    L[0] = 0.0
    L[1] = 0.0
    L[2] = 0.0
    ld, md, nd = int(0), int(0), int(0)
    alphalk = 0.0

    if i >= indx_startA and i < indx_endA and j >= indx_startB and j < indx_endB and (j <= i):
        sqrt_ints4c2e_diag_ij = sqrt_ints4c2e_diag[i, j]
        if strict_schwarz:
            if sqrt_ints4c2e_diag_ij * sqrt_ints4c2e_diag_ij < STRICT_PAIR_CUTOFF:
                return
        linear_index = j + i * (i + 1) // 2
        offset_ = offsets[linear_index]
        IJ = cuda.local.array((3), numba.float64)
        P = cuda.local.array((3), numba.float64)
        PQ = cuda.local.array((3), numba.float64)
        I = bfs_coords[i]
        Ni = bfs_contr_prim_norms[i]
        lmni = bfs_lmn[i]
        la, ma, na = lmni
        nprimi = bfs_nprim[i]

        J = bfs_coords[j]
        IJ[0] = I[0] - J[0]
        IJ[1] = I[1] - J[1]
        IJ[2] = I[2] - J[2]
        IJsq = IJ[0] ** 2 + IJ[1] ** 2 + IJ[2] ** 2
        Nj = bfs_contr_prim_norms[j]
        lmnj = bfs_lmn[j]
        lb, mb, nb = lmnj
        tempcoeff1 = Ni * Nj
        nprimj = bfs_nprim[j]

        G = cuda.local.array((5, 5), numba.float64)  # Good for upto g auxshells and d shells;
        roots = cuda.local.array((7, 5), numba.float64)  # Good for upto g auxshells and d shells; and 7 primitives per bf
        weights = cuda.local.array((7, 5), numba.float64)  # Good for upto g auxshells and d shells; and 7 primitives per bf

        # Cartesian components of the current d/f/g shell
        shell_buffer = cuda.local.array((15), numba.float64)

        # Loop over primitives
        for ik in range(nprimi):
            dik = bfs_coeffs[i][ik]
            Nik = bfs_prim_norms[i][ik]
            alphaik = bfs_expnts[i][ik]
            tempcoeff2 = tempcoeff1 * dik * Nik

            for jk in range(nprimj):
                alphajk = bfs_expnts[j][jk]
                gammaP = alphaik + alphajk
                prod_alphaikjk = alphaik * alphajk
                screenfactorAB = math.exp(-prod_alphaikjk / gammaP * IJsq)
                if abs(screenfactorAB) < 1.0e-8:
                    continue
                djk = bfs_coeffs[j][jk]
                Njk = bfs_prim_norms[j][jk]
                P[0] = (alphaik * I[0] + alphajk * J[0]) / gammaP
                P[1] = (alphaik * I[1] + alphajk * J[1]) / gammaP
                P[2] = (alphaik * I[2] + alphajk * J[2]) / gammaP
                tempcoeff3 = tempcoeff2 * djk * Njk

                roots_shell_previous = -1
                index_k = 0
                k = indx_startC
                buf_idx = 0

                while k < indx_endC:
                    shell_index = aux_shell_indices[k]
                    lmnk = aux_bfs_lmn[k]
                    lc, mc, nc = lmnk
                    tot_ang = lc + mc + nc
                    if tot_ang == 0:
                        shell_size = 1
                    elif tot_ang == 1:
                        shell_size = 3
                    elif tot_ang == 2:
                        shell_size = 6
                    elif tot_ang == 3:
                        shell_size = 10
                    else:
                        shell_size = 15

                    is_new_shell = (shell_index != roots_shell_previous)

                    if is_new_shell:
                        # Whole shell insignificant -> skip all of its components.
                        if sqrt_ints4c2e_diag_ij * sqrt_diag_ints2c2e[k] <= schwarz_threshold:
                            k += shell_size
                            continue
                        buf_idx = 0

                    # s and p functions need no projection: treat them individually.
                    if tot_ang <= 1 and sqrt_ints4c2e_diag_ij * sqrt_diag_ints2c2e[k] <= schwarz_threshold:
                        k += 1
                        continue

                    K = aux_bfs_coords[k]
                    Nk = aux_bfs_contr_prim_norms[k]
                    tempcoeff4 = tempcoeff3 * Nk
                    nprimk = aux_bfs_nprim[k]

                    Q = K
                    PQ[0] = P[0] - Q[0]
                    PQ[1] = P[1] - Q[1]
                    PQ[2] = P[2] - Q[2]
                    PQsq = PQ[0] ** 2 + PQ[1] ** 2 + PQ[2] ** 2

                    norder = int((la + ma + na + lb + mb + nb + lc + mc + nc + ld + md + nd) / 2 + 1)
                    val = 0.0

                    if norder <= 10:
                        n = int(max(la + lb, ma + mb, na + nb))
                        m = int(max(lc + ld, mc + md, nc + nd))

                        for kk in range(nprimk):
                            dkk = aux_bfs_coeffs[k][kk]
                            Nkk = aux_bfs_prim_norms[k][kk]
                            alphakk = aux_bfs_expnts[k][kk]
                            tempcoeff5 = tempcoeff4 * dkk * Nkk
                            ABsrt = math.sqrt(gammaP * alphakk)

                            gammaQ = alphakk
                            rho = gammaP * gammaQ / (gammaP + gammaQ)
                            X = PQsq * rho
                            roots_kk = roots[kk, :]
                            weights_kk = weights[kk, :]

                            # Rys roots are shared by all components of an aux shell.
                            if is_new_shell:
                                roots_kk, weights_kk = Roots_5(norder, X, roots_kk, weights_kk)

                            val += tempcoeff5 * coulomb_rys_3c2e(roots_kk, weights_kk, G, PQsq, rho, norder, n, m, la, lb, lc, ld, ma, mb, mc, md, na, nb, nc, nd, alphaik, alphajk, alphakk, alphalk, I, J, K, L, IJ, P, prod_alphaikjk, gammaP, ABsrt)

                    roots_shell_previous = shell_index
                    is_new_shell = False

                    if tot_ang <= 1:
                        out[offset_ + index_k] += val
                        index_k += 1
                    else:
                        shell_buffer[buf_idx] = val
                        buf_idx += 1
                        if buf_idx == shell_size:
                            # Shell complete: project onto the spherical subspace and
                            # store the components that pass the function-level test.
                            k_first = k - shell_size + 1
                            for r in range(shell_size):
                                if sqrt_ints4c2e_diag_ij * sqrt_diag_ints2c2e[k_first + r] > schwarz_threshold:
                                    temp_val = 0.0
                                    if tot_ang == 2:
                                        for c in range(6):
                                            temp_val += proj_d[r, c] * shell_buffer[c]
                                    elif tot_ang == 3:
                                        for c in range(10):
                                            temp_val += proj_f[r, c] * shell_buffer[c]
                                    else:
                                        for c in range(15):
                                            temp_val += proj_g[r, c] * shell_buffer[c]
                                    out[offset_ + index_k] += temp_val
                                    index_k += 1

                    k += 1


# ----------------------------------------------------------------------------
# 3. gamma_P = sum_ij D_ij (ij|P)
# ----------------------------------------------------------------------------
def df_coeff_calculator_algo10_cupy(ints3c2e_1d, dmat_1d, nao, offsets_3c2e, naux, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold, strict_schwarz, cp_stream=None):
    """``gamma_P = sum_{i>=j} D~_ij (ij|P)`` on the device (``dmat_1d`` is the triangular, off-diagonal-doubled density)."""
    cp_stream, nb_stream = _stream_pair(cp_stream)
    df_coeff = cp.zeros(naux, dtype=cp.float64)

    thread_x = 32
    thread_y = 32
    blocks_per_grid = ((nao + (thread_x - 1)) // thread_x, (nao + (thread_y - 1)) // thread_y)
    size_dmat_1d = dmat_1d.shape[0]
    df_coeff_calculator_algo10_cuda_internal[blocks_per_grid, (thread_x, thread_y), nb_stream](ints3c2e_1d, dmat_1d, nao, size_dmat_1d, offsets_3c2e, naux, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold, strict_schwarz, df_coeff)

    cp_stream.synchronize()
    cp.cuda.Stream.null.synchronize()
    return df_coeff


@cuda.jit(fastmath=True, cache=True)
def df_coeff_calculator_algo10_cuda_internal(ints3c2e_1d, dmat_1d, nao, size_dmat_1d, offsets_3c2e, naux, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold, strict_schwarz, df_coeff):
    i, j = cuda.grid(2)
    if i >= 0 and i < nao and j >= 0 and j <= i:
        offset = int(j + i * (i + 1) / 2)
        if offset >= size_dmat_1d:
            return
        sqrt_ij = sqrt_ints4c2e_diag[i, j]
        if strict_schwarz:
            if sqrt_ij * sqrt_ij < STRICT_PAIR_CUTOFF:
                return
        index_k = 0
        offset_3c2e = offsets_3c2e[offset]
        dmat_val = dmat_1d[offset]
        for k in range(0, naux):
            if sqrt_ij * sqrt_diag_ints2c2e[k] > threshold:
                temp = ints3c2e_1d[offset_3c2e + index_k] * dmat_val
                cuda.atomic.add(df_coeff, k, temp)
                index_k += 1


# ----------------------------------------------------------------------------
# 4. J_ij = sum_P (ij|P) c_P
# ----------------------------------------------------------------------------
def J_tri_calculator_algo10_cupy(ints3c2e_1d, df_coeff, size_J_tri, nao, offsets_3c2e, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold, naux, strict_schwarz, cp_stream=None):
    """``J_tri[i*(i+1)//2 + j] = sum_P (ij|P) c_P`` on the device."""
    cp_stream, nb_stream = _stream_pair(cp_stream)
    J_tri = cp.zeros(size_J_tri, dtype=cp.float64)

    cp_stream.synchronize()
    cp.cuda.Stream.null.synchronize()

    thread_x = 32
    thread_y = 32
    blocks_per_grid = ((nao + (thread_x - 1)) // thread_x, (nao + (thread_y - 1)) // thread_y)
    J_tri_calculator_algo10_internal_cuda[blocks_per_grid, (thread_x, thread_y), nb_stream](ints3c2e_1d, df_coeff, nao, offsets_3c2e, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold, naux, strict_schwarz, J_tri)

    cp_stream.synchronize()
    cp.cuda.Stream.null.synchronize()
    return J_tri


@cuda.jit(fastmath=True, cache=True)
def J_tri_calculator_algo10_internal_cuda(ints3c2e_1d, df_coeff, nao, offsets_3c2e, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold, naux, strict_schwarz, J_tri):
    i, j = cuda.grid(2)
    if i >= 0 and i < nao and j >= 0 and j < nao and (j <= i):
        offset = int(i * (i + 1) / 2)  # Offset for the J_tri array
        sqrt_ij = sqrt_ints4c2e_diag[i, j]
        if strict_schwarz:
            if sqrt_ij * sqrt_ij < STRICT_PAIR_CUTOFF:
                return
        index_k = 0
        val = 0.0
        linear_index = j + i * (i + 1) // 2
        offset_3c2e = offsets_3c2e[linear_index]  # Offset for the 3c2e array
        for k in range(naux):
            if sqrt_ij * sqrt_diag_ints2c2e[k] > threshold:
                val += ints3c2e_1d[offset_3c2e + index_k] * df_coeff[k]
                index_k += 1
        J_tri[j + offset] = val
