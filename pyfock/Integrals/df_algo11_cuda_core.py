"""CUDA implementation shared by the generated angular-momentum kernels.

Small items use one thread; large items use one block, shared Rys/shift tables,
and strided component accumulators. No integral is evaluated per AO function.
The build writes Cartesian values; a separate row-owned pass projects whole
auxiliary shells, including h/i shells, without in-place read/write races.

``evaluate_item`` serves both density-fitting algorithms. With ``masked`` false
(algorithm 11) every surviving primitive pair of the shell pair contributes and
``mask`` is ignored; with ``masked`` true (algorithm 12) a primitive pair is
skipped for auxiliary shell ``K`` when its branch is far field for that shell,
so the block holds the near-field part only and the rest comes from the
multipole expansions. ``mask`` is ``(pp_off, pp_branch, ff)``: the branch of
primitive pair ``(ip, jp)`` of pair ``p`` is ``pp_branch[pp_off[p] + ip * nprimB
+ jp]`` and ``ff[branch, K]`` is its far-field flag. The index is dense in
``(ip, jp)`` rather than a count of surviving pairs on purpose: the device would
otherwise have to reproduce the host's Gaussian-product pre-screen decision
exactly, and a one-ulp disagreement would shift every later branch of that shell
pair. Pairs the host dropped carry a branch whose row is far field everywhere, so
they are skipped here too even if this pre-screen keeps them. Both callers pass
arrays of the same dtypes, so one specialization is compiled.
"""
import math

from numba import cuda, float64

from .df_algo10_helpers import EXP_ARG_CUTOFF, STRICT_PAIR_CUTOFF
from .rys_helpers_cuda import Roots, Recur_3c2e, Shift_3c2e


@cuda.jit(device=True, cache=True)
def evaluate_item(item, lane, stride, cooperative, orbital, auxiliary, shells,
                  aux_shells, pairs, items, offsets, values, data_x, data_w, masked, mask,
                  roots, weights, gx, gy, gz, sx, sy, sz, accum, tmp):
    coords, lmn, nprim, expnts, coef = orbital
    pp_off, pp_branch, ff = mask
    acoords, almn, anprim, aexpnts, acoef = auxiliary
    off, nbf, ls = shells
    aoff, anbf, als = aux_shells
    pair_i, pair_j, nrows, ncols = pairs
    p, K, col = items[item, 0], items[item, 1], items[item, 2]
    I, J = pair_i[p], pair_j[p]
    a0, b0, c0 = off[I], off[J], aoff[K]
    nA, nB, nC = nbf[I], nbf[J], anbf[K]
    lA, lB, lC = ls[I], ls[J], als[K]
    nr = (lA + lB + lC) // 2 + 1
    nelem = nA * nB * nC
    dx = coords[a0, 0] - coords[b0, 0]
    dy = coords[a0, 1] - coords[b0, 1]
    dz = coords[a0, 2] - coords[b0, 2]
    r2 = dx * dx + dy * dy + dz * dz
    for e in range(lane, nelem, stride):
        accum[e // stride] = 0.0
    for ip in range(nprim[a0]):
        alpha = expnts[a0, ip]
        for jp in range(nprim[b0]):
            beta = expnts[b0, jp]
            gp = alpha + beta
            ab = alpha * beta
            if ab / gp * r2 > EXP_ARG_CUTOFF:
                continue
            if masked and ff[pp_branch[pp_off[p] + ip * nprim[b0] + jp], K]:
                continue
            px = (alpha * coords[a0, 0] + beta * coords[b0, 0]) / gp
            py = (alpha * coords[a0, 1] + beta * coords[b0, 1]) / gp
            pz = (alpha * coords[a0, 2] + beta * coords[b0, 2]) / gp
            pq2 = ((px - acoords[c0, 0]) ** 2 + (py - acoords[c0, 1]) ** 2
                   + (pz - acoords[c0, 2]) ** 2)
            for kp in range(anprim[c0]):
                gq = aexpnts[c0, kp]
                rho = gp * gq / (gp + gq)
                pref = 2.0 * math.sqrt(rho / math.pi)
                gpq = math.sqrt(gp * gq)
                if lane == 0:
                    Roots(nr, rho * pq2, data_x, data_w, roots, weights)
                if cooperative:
                    cuda.syncthreads()
                for e in range(lane, nelem, stride):
                    tmp[e // stride] = 0.0
                for ir in range(nr):
                    if lane == 0:
                        t = roots[ir]
                        Recur_3c2e(gx, t, lA + lB, 0, lC, 0, coords[a0, 0], coords[b0, 0],
                                    acoords[c0, 0], 0.0, alpha, beta, gq, 0.0, gp, gq, ab, gpq)
                        Recur_3c2e(gy, t, lA + lB, 0, lC, 0, coords[a0, 1], coords[b0, 1],
                                    acoords[c0, 1], 0.0, alpha, beta, gq, 0.0, gp, gq, ab, gpq)
                        Recur_3c2e(gz, t, lA + lB, 0, lC, 0, coords[a0, 2], coords[b0, 2],
                                    acoords[c0, 2], 0.0, alpha, beta, gq, 0.0, gp, gq, ab, gpq)
                        for a in range(lA + 1):
                            for b in range(lB + 1):
                                for c in range(lC + 1):
                                    sx[a, b, c] = Shift_3c2e(gx, a, b, c, 0, dx)
                                    sy[a, b, c] = Shift_3c2e(gy, a, b, c, 0, dy)
                                    sz[a, b, c] = Shift_3c2e(gz, a, b, c, 0, dz)
                    if cooperative:
                        cuda.syncthreads()
                    w = pref * weights[ir]
                    for e in range(lane, nelem, stride):
                        ia = e // (nB * nC)
                        ib = e // nC % nB
                        ic = e % nC
                        tmp[e // stride] += (w * sx[lmn[a0 + ia, 0], lmn[b0 + ib, 0], almn[c0 + ic, 0]]
                                            * sy[lmn[a0 + ia, 1], lmn[b0 + ib, 1], almn[c0 + ic, 1]]
                                            * sz[lmn[a0 + ia, 2], lmn[b0 + ib, 2], almn[c0 + ic, 2]])
                    if cooperative:
                        cuda.syncthreads()
                for e in range(lane, nelem, stride):
                    ia = e // (nB * nC)
                    ib = e // nC % nB
                    ic = e % nC
                    cab = coef[a0 + ia, ip] * coef[b0 + ib, jp]
                    accum[e // stride] += cab * acoef[c0 + ic, kp] * tmp[e // stride]
                # Protect weights until every lane has consumed the final root.
                if cooperative:
                    cuda.syncthreads()
    for e in range(lane, nelem, stride):
        ia = e // (nB * nC)
        ib = e // nC % nB
        ic = e % nC
        if I == J:
            if ib > ia:
                continue
            row = ia * (ia + 1) // 2 + ib
        else:
            row = ia * nB + ib
        values[offsets[p] + row * ncols[p] + col + ic] = accum[e // stride]


@cuda.jit(cache=True)
def project_items(items, pairs, aux_shells, offsets, values, projectors):
    item = cuda.blockIdx.x
    p, K, col = items[item, 0], items[item, 1], items[item, 2]
    _, nbf, ls = aux_shells
    lC, nC = ls[K], nbf[K]
    if lC < 2:
        return
    projected = cuda.local.array(28, float64)
    for row in range(cuda.threadIdx.x, pairs[2][p], cuda.blockDim.x):
        start = offsets[p] + row * pairs[3][p] + col
        for r in range(nC):
            acc = 0.0
            for c in range(nC):
                acc += projectors[lC, r, c] * values[start + c]
            projected[r] = acc
        for r in range(nC):
            values[start + r] = projected[r]


@cuda.jit(cache=True)
def gamma_items(items, pairs, shells, aux_shells, offsets, values, sqrt4, strict, dmat, gamma):
    item = cuda.grid(1) // 8
    lane = cuda.threadIdx.x % 8
    if item >= items.shape[0]:
        return
    p, K, col = items[item, 0], items[item, 1], items[item, 2]
    I, J = pairs[0][p], pairs[1][p]
    a0, b0 = shells[0][I], shells[0][J]
    nA, nB = shells[1][I], shells[1][J]
    for c in range(lane, aux_shells[1][K], 8):
        acc = 0.0
        for ia in range(nA):
            for ib in range(ia + 1 if I == J else nB):
                i, j = a0 + ia, b0 + ib
                if strict and sqrt4[i, j] * sqrt4[i, j] < STRICT_PAIR_CUTOFF:
                    continue
                row = ia * (ia + 1) // 2 + ib if I == J else ia * nB + ib
                weight = 1.0 if i == j else 2.0
                acc += weight * dmat[i, j] * values[offsets[p] + row * pairs[3][p] + col + c]
        cuda.atomic.add(gamma, aux_shells[0][K] + c, acc)


@cuda.jit(cache=True)
def make_column_map(items, aux_shells, column_offsets, column_map):
    item = cuda.grid(1)
    if item < items.shape[0]:
        p, K, col = items[item, 0], items[item, 1], items[item, 2]
        for c in range(aux_shells[1][K]):
            column_map[column_offsets[p] + col + c] = aux_shells[0][K] + c


@cuda.jit(cache=True)
def j_pairs(work, pairs, shells, aux_shells, offsets, values, sqrt4, strict,
            column_offsets, column_map, coeff, out):
    idx = cuda.grid(1) // 32
    lane = cuda.threadIdx.x % 32
    if idx >= work.shape[0]:
        return
    p, row = work[idx, 0], work[idx, 1]
    I, J = pairs[0][p], pairs[1][p]
    nB = shells[1][J]
    if I == J:
        ia = int((math.sqrt(8.0 * row + 1.0) - 1.0) / 2.0)
        ib = row - ia * (ia + 1) // 2
    else:
        ia, ib = row // nB, row % nB
    i, j = shells[0][I] + ia, shells[0][J] + ib
    if strict and sqrt4[i, j] * sqrt4[i, j] < STRICT_PAIR_CUTOFF:
        return
    acc = 0.0
    start = offsets[p] + row * pairs[3][p]
    map_start = column_offsets[p]
    for col in range(lane, pairs[3][p], 32):
        acc += values[start + col] * coeff[column_map[map_start + col]]
    for shift in (16, 8, 4, 2, 1):
        acc += cuda.shfl_down_sync(0xffffffff, acc, shift)
    if lane == 0:
        out[i, j] = acc
        out[j, i] = acc


@cuda.jit(cache=True)
def count_columns(work, items, pairs, aux_shells, qpair, qaux, threshold, counts, errors):
    """Independent device replay for debug validation of work lists and sizing."""
    idx = cuda.grid(1)
    if idx < items.shape[0]:
        p, K, col = items[idx, 0], items[idx, 1], items[idx, 2]
        expected_col = 0
        for k in range(K):
            if qpair[p] * qaux[k] > threshold:
                expected_col += aux_shells[1][k]
        if (qpair[p] * qaux[K] <= threshold or col != expected_col
                or col + aux_shells[1][K] > pairs[3][p]):
            errors[idx] = 1
        cuda.atomic.add(counts, p, aux_shells[1][K])
