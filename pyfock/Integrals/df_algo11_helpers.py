"""
Density-fitting Coulomb term, algorithm 11 (``DF_algo=11``, the CPU default): shell-blocked
Rys quadrature, block-sparse storage and an optional memory budget.

Compared with algorithm 10 (:mod:`~pyfock.Integrals.df_algo10_helpers`, the former default)
this module keeps the same physics - Schwarz screening, the strict pair cut-off,
the pseudo-Cartesian treatment of spherical auxiliary functions - but organizes
the work around *shells* instead of individual basis functions:

* **Kernel.**  For a shell pair ``(I, J)`` and an auxiliary shell ``K`` the whole
  Cartesian block ``(ij|P)``, ``i in I, j in J, P in K``, is produced from one set
  of Rys roots per primitive triple.  The two-dimensional Rys recurrences and the
  horizontal (A-B) shifts are evaluated once per unique exponent triple
  ``(a_x, b_x, c_x)`` and looked up for all components, so the cost per integral
  drops by a factor of 2-3 relative to the per-function kernel.  Primitive pairs
  that fail the Gaussian-product pre-screen are dropped once per shell pair.

* **Storage.**  Only shell pairs with ``I >= J`` are kept.  Each pair owns one
  contiguous *row block*: rows are the function pairs ``(i, j)`` of the shell pair
  (all ``n_I n_J`` pairs, or the ``n(n+1)/2`` pairs with ``i >= j`` for ``I == J``),
  columns are the functions of the significant auxiliary shells in increasing
  order.  An auxiliary shell ``K`` is significant for pair ``p`` when
  ``Q_pair[p] * Q_aux[K] > threshold`` with ``Q_pair = max sqrt((ij|ij))`` over the
  pair and ``Q_aux = max sqrt((P|P))`` over the shell.  No index arrays are stored:
  the contraction passes replay this shell-level test, which costs one comparison
  per (shell pair, aux shell) instead of one per (function pair, aux function).

* **Screening.**  A shell pair is dropped entirely when every primitive pair fails
  the Gaussian-product pre-screen or (``strict_schwarz``) when
  ``max (ij|ij) < STRICT_PAIR_CUTOFF``.  Inside a surviving shell pair the strict
  pair cut-off is still applied to every function pair in the contractions, so
  the result is the same set of pairs that algorithm 10 keeps and that the nuclear
  attraction matrix drops.

* **Memory budget.**  ``max_memory_gb`` caps the storage of the row blocks.  Shell
  pairs are ranked by *compute cost per stored element* (many primitives, high
  angular momentum first) and cached greedily until the budget is exhausted; the
  remaining pairs are re-evaluated in every SCF iteration - once for
  ``gamma_P = sum_ij D_ij (ij|P)`` and once for ``J_ij = sum_P (ij|P) c_P``.
  ``max_memory_gb=0`` therefore gives a fully direct DF-J; ``None`` stores all
  significant blocks.

* **Spherical auxiliary functions.**  With ``sao=True`` every auxiliary d, f, g,
  ... block is projected onto its spherical subspace, ``Proj = pinv(C) C``, before
  storage.  Whole shells are always stored, so the stored vectors stay inside the
  subspace of the projected, ``eps``-regularized metric (see algorithm 10).

* **Parallelism.**  Shell pairs are assigned to one bin per thread by a
  longest-processing-time-first heuristic on the cost model (Numba's workqueue
  layer only schedules ``prange`` statically); every pass is a ``prange`` over
  the bins.  ``gamma`` accumulates into per-bin vectors, ``J`` writes disjoint
  matrix elements, so there are no races and no atomics.
"""

import numpy as np
import numba
from numba import njit, prange

from .rys_helpers import Roots, Recur_3c2e_new, Shift_3c2e
from .df_algo10_helpers import (
    STRICT_PAIR_CUTOFF, EXP_ARG_CUTOFF, MAX_RYS_ROOTS,
    pack_basis_arrays, aux_shell_arrays, sao_aux_projectors)

__all__ = ['DFAlgo11Plan', 'build_plan', 'gamma_from_plan', 'J_from_plan']

PI = 3.141592653589793


# ----------------------------------------------------------------------------
# Plan object
# ----------------------------------------------------------------------------
class DFAlgo11Plan:
    """
    Everything the per-iteration passes need: packed basis data, the shell-pair
    work list with its screening data, and the cached row blocks.  Created by
    :func:`build_plan`; consumed by :func:`gamma_from_plan` and :func:`J_from_plan`.

    Attributes of general interest
    ------------------------------
    nao, naux : int
    n_pairs_total, n_pairs_significant, n_pairs_cached : int
    n_elements_significant, n_elements_cached : int
        Row-block elements (doubles) that are significant / kept in memory.
    memory_gb : float
        Size of the cached row blocks in GB.
    fraction_cached : float
        ``n_elements_cached / n_elements_significant``.
    """

    def __init__(self):
        self.values = None

    @property
    def memory_gb(self):
        return 0.0 if self.values is None else self.values.nbytes / 1e9

    @property
    def fraction_cached(self):
        if self.n_elements_significant == 0:
            return 1.0
        return self.n_elements_cached / self.n_elements_significant

    def summary(self):
        return (
            f"DF_algo=11: {self.n_pairs_significant} of {self.n_pairs_total} shell pairs significant, "
            f"{self.n_elements_significant} block elements ({8e-9 * self.n_elements_significant:.3f} GB); "
            f"cached {self.n_pairs_cached} pairs = {100 * self.fraction_cached:.1f}% of the elements "
            f"({self.memory_gb:.3f} GB), the rest is recomputed in every SCF iteration")


# ----------------------------------------------------------------------------
# Shell-blocked kernel: row block of one shell pair
# ----------------------------------------------------------------------------
@njit(nogil=True, cache=True, fastmath=True, error_model="numpy", boundscheck=False)
def _compute_pair_rows(I, J, rows,
                       bfs_coords, bfs_lmn, bfs_nprim, bfs_expnts, bf_coef, shell_off, shell_nbf, shell_l,
                       aux_coords, aux_lmn, aux_nprim, aux_expnts, aux_coef, aux_off, aux_nbf, aux_l,
                       Qp, Q_aux, threshold, sao, projectors,
                       roots, weights, gx, gy, gz, Sx, Sy, Sz, blk, tmp, blk2,
                       pp_alpha, pp_beta, pp_gamma, pp_px, pp_py, pp_pz, pp_ip, pp_jp,
                       ax_, ay_, az_, bx_, by_, bz_, cx_, cy_, cz_):
    """
    Fill ``rows[r, :ncols]`` with ``(ij|P)`` for shell pair ``(I, J)``: row ``r``
    is the function pair ``(ia, ib)`` (``ia*nB + ib``, or ``ia*(ia+1)/2 + ib`` with
    ``ib <= ia`` when ``I == J``), columns run over the functions of the
    significant auxiliary shells in increasing order.  Returns the number of
    columns written.
    """
    nA = shell_nbf[I]
    nB = shell_nbf[J]
    a0 = shell_off[I]
    b0 = shell_off[J]
    lA = shell_l[I]
    lB = shell_l[J]
    bra_order = lA + lB
    nprimA = bfs_nprim[a0]
    nprimB = bfs_nprim[b0]
    Ax = bfs_coords[a0, 0]
    Ay = bfs_coords[a0, 1]
    Az = bfs_coords[a0, 2]
    Bx = bfs_coords[b0, 0]
    By = bfs_coords[b0, 1]
    Bz = bfs_coords[b0, 2]
    xij0 = Ax - Bx
    xij1 = Ay - By
    xij2 = Az - Bz
    ijsq = xij0 * xij0 + xij1 * xij1 + xij2 * xij2
    diag_pair = (I == J)

    for ia in range(nA):
        ax_[ia] = bfs_lmn[a0 + ia, 0]
        ay_[ia] = bfs_lmn[a0 + ia, 1]
        az_[ia] = bfs_lmn[a0 + ia, 2]
    for ib in range(nB):
        bx_[ib] = bfs_lmn[b0 + ib, 0]
        by_[ib] = bfs_lmn[b0 + ib, 1]
        bz_[ib] = bfs_lmn[b0 + ib, 2]

    # Primitive pairs that survive the Gaussian-product pre-screen (once per shell pair).
    npp = 0
    for ip in range(nprimA):
        alpha = bfs_expnts[a0, ip]
        for jp in range(nprimB):
            beta = bfs_expnts[b0, jp]
            gamma_p = alpha + beta
            if alpha * beta / gamma_p * ijsq > EXP_ARG_CUTOFF:
                continue
            pp_alpha[npp] = alpha
            pp_beta[npp] = beta
            pp_gamma[npp] = gamma_p
            pp_px[npp] = (alpha * Ax + beta * Bx) / gamma_p
            pp_py[npp] = (alpha * Ay + beta * By) / gamma_p
            pp_pz[npp] = (alpha * Az + beta * Bz) / gamma_p
            pp_ip[npp] = ip
            pp_jp[npp] = jp
            npp += 1

    col = 0
    for K in range(aux_off.shape[0]):
        if Qp * Q_aux[K] <= threshold:
            continue
        nC = aux_nbf[K]
        c0 = aux_off[K]
        lC = aux_l[K]
        nprimC = aux_nprim[c0]
        Cx = aux_coords[c0, 0]
        Cy = aux_coords[c0, 1]
        Cz = aux_coords[c0, 2]
        for ic in range(nC):
            cx_[ic] = aux_lmn[c0 + ic, 0]
            cy_[ic] = aux_lmn[c0 + ic, 1]
            cz_[ic] = aux_lmn[c0 + ic, 2]
        nroots = (bra_order + lC) // 2 + 1

        for ia in range(nA):
            for ib in range(nB):
                for ic in range(nC):
                    blk[ia, ib, ic] = 0.0

        for q in range(npp):
            alpha = pp_alpha[q]
            beta = pp_beta[q]
            gamma_p = pp_gamma[q]
            px = pp_px[q]
            py = pp_py[q]
            pz = pp_pz[q]
            ip = pp_ip[q]
            jp = pp_jp[q]
            ab = alpha * beta
            pqsq = (px - Cx) ** 2 + (py - Cy) ** 2 + (pz - Cz) ** 2
            for kp in range(nprimC):
                gamma_q = aux_expnts[c0, kp]
                rho = gamma_p * gamma_q / (gamma_p + gamma_q)
                x = rho * pqsq
                gpq_sqrt = np.sqrt(gamma_p * gamma_q)
                Roots(nroots, x, roots, weights)
                pref = 2.0 * np.sqrt(rho / PI)

                for ia in range(nA):
                    for ib in range(nB):
                        for ic in range(nC):
                            tmp[ia, ib, ic] = 0.0

                for ir in range(nroots):
                    t = roots[ir]
                    Recur_3c2e_new(gx, t, bra_order, 0, lC, 0, Ax, Bx, Cx, 0.0,
                                   alpha, beta, gamma_q, 0.0, gamma_p, gamma_q, ab, gpq_sqrt)
                    Recur_3c2e_new(gy, t, bra_order, 0, lC, 0, Ay, By, Cy, 0.0,
                                   alpha, beta, gamma_q, 0.0, gamma_p, gamma_q, ab, gpq_sqrt)
                    Recur_3c2e_new(gz, t, bra_order, 0, lC, 0, Az, Bz, Cz, 0.0,
                                   alpha, beta, gamma_q, 0.0, gamma_p, gamma_q, ab, gpq_sqrt)
                    # Horizontal shifts once per unique (a, b, c) exponent triple per Cartesian direction.
                    for a in range(lA + 1):
                        for b in range(lB + 1):
                            for c in range(lC + 1):
                                Sx[a, b, c] = Shift_3c2e(gx, a, b, c, 0, xij0)
                                Sy[a, b, c] = Shift_3c2e(gy, a, b, c, 0, xij1)
                                Sz[a, b, c] = Shift_3c2e(gz, a, b, c, 0, xij2)
                    w = pref * weights[ir]
                    for ia in range(nA):
                        axa = ax_[ia]
                        aya = ay_[ia]
                        aza = az_[ia]
                        for ib in range(nB):
                            bxb = bx_[ib]
                            byb = by_[ib]
                            bzb = bz_[ib]
                            for ic in range(nC):
                                tmp[ia, ib, ic] += w * Sx[axa, bxb, cx_[ic]] * Sy[aya, byb, cy_[ic]] * Sz[aza, bzb, cz_[ic]]

                # Contraction coefficients and normalization of this primitive triple.
                for ia in range(nA):
                    ca = bf_coef[a0 + ia, ip]
                    for ib in range(nB):
                        cab = ca * bf_coef[b0 + ib, jp]
                        for ic in range(nC):
                            blk[ia, ib, ic] += cab * aux_coef[c0 + ic, kp] * tmp[ia, ib, ic]

        # Spherical auxiliary functions: project the block onto the spherical subspace.
        if sao and lC >= 2:
            for ia in range(nA):
                for ib in range(nB):
                    for r in range(nC):
                        acc = 0.0
                        for c in range(nC):
                            acc += projectors[lC, r, c] * blk[ia, ib, c]
                        blk2[ia, ib, r] = acc
            src = blk2
        else:
            src = blk

        # Scatter into the row block (lower triangle of function pairs for I == J).
        for ia in range(nA):
            ibmax = ia + 1 if diag_pair else nB
            for ib in range(ibmax):
                r = (ia * (ia + 1)) // 2 + ib if diag_pair else ia * nB + ib
                for ic in range(nC):
                    rows[r, col + ic] = src[ia, ib, ic]
        col += nC
    return col


# ----------------------------------------------------------------------------
# Screening data and cost model for all shell pairs
# ----------------------------------------------------------------------------
@njit(parallel=True, cache=True, fastmath=True, error_model="numpy", nogil=True)
def _pair_screening_info(pair_I, pair_J, shell_off, shell_nbf, shell_l, bfs_nprim, bfs_expnts, bfs_coords,
                         sqrt_ints4c2e_diag, Q_aux, aux_off, aux_nbf, aux_l, aux_nprim,
                         threshold, strict_schwarz):
    npairs = pair_I.shape[0]
    nsh_aux = aux_off.shape[0]
    Q_pair = np.zeros(npairs, dtype=np.float64)
    alive = np.zeros(npairs, dtype=np.bool_)
    nrows = np.zeros(npairs, dtype=np.int64)
    ncols = np.zeros(npairs, dtype=np.int64)
    cost = np.zeros(npairs, dtype=np.float64)
    max_roots = np.zeros(npairs, dtype=np.int64)
    for p in prange(npairs):
        I = pair_I[p]
        J = pair_J[p]
        a0 = shell_off[I]
        b0 = shell_off[J]
        nA = shell_nbf[I]
        nB = shell_nbf[J]
        lA = shell_l[I]
        lB = shell_l[J]
        Q = 0.0
        for ia in range(nA):
            for ib in range(nB):
                v = sqrt_ints4c2e_diag[a0 + ia, b0 + ib]
                if v > Q:
                    Q = v
        Q_pair[p] = Q
        nrows[p] = nA * nB if I != J else nA * (nA + 1) // 2
        ok = True
        if strict_schwarz and Q * Q < STRICT_PAIR_CUTOFF:
            ok = False
        # Gaussian-product pre-screen with the most diffuse primitives: if it fails,
        # every primitive pair fails and the whole block is zero.
        nprimA = bfs_nprim[a0]
        nprimB = bfs_nprim[b0]
        amin = bfs_expnts[a0, 0]
        for ip in range(1, nprimA):
            if bfs_expnts[a0, ip] < amin:
                amin = bfs_expnts[a0, ip]
        bmin = bfs_expnts[b0, 0]
        for jp in range(1, nprimB):
            if bfs_expnts[b0, jp] < bmin:
                bmin = bfs_expnts[b0, jp]
        ijsq = ((bfs_coords[a0, 0] - bfs_coords[b0, 0]) ** 2 + (bfs_coords[a0, 1] - bfs_coords[b0, 1]) ** 2
                + (bfs_coords[a0, 2] - bfs_coords[b0, 2]) ** 2)
        if amin * bmin / (amin + bmin) * ijsq > EXP_ARG_CUTOFF:
            ok = False
        nc = 0
        cst = 0.0
        mr = 0
        if ok:
            for K in range(nsh_aux):
                if Q * Q_aux[K] > threshold:
                    nC = aux_nbf[K]
                    lC = aux_l[K]
                    nprimC = aux_nprim[aux_off[K]]
                    nroots = (lA + lB + lC) // 2 + 1
                    if nroots > mr:
                        mr = nroots
                    nc += nC
                    # per primitive triple: roots + recurrences + shift tables + accumulation
                    cst += nprimA * nprimB * nprimC * (
                        6.0 * nroots * (lA + lB + 1) * (lC + 1)
                        + 3.0 * nroots * (lA + 1) * (lB + 1) * (lC + 1) * (lB + 1)
                        + 3.0 * nroots * nA * nB * nC
                        + 2.0 * nA * nB * nC)
        if nc == 0:
            ok = False
        alive[p] = ok
        ncols[p] = nc
        cost[p] = cst
        max_roots[p] = mr
    return Q_pair, alive, nrows, ncols, cost, max_roots


# ----------------------------------------------------------------------------
# Parallel passes
# ----------------------------------------------------------------------------
@njit(parallel=True, cache=True, fastmath=True, error_model="numpy", nogil=True, boundscheck=False)
def _build_cached(bin_off, bin_items, pair_I, pair_J, pair_offset, pair_nrows, pair_ncols, values,
                  bfs_coords, bfs_lmn, bfs_nprim, bfs_expnts, bf_coef, shell_off, shell_nbf, shell_l,
                  aux_coords, aux_lmn, aux_nprim, aux_expnts, aux_coef, aux_off, aux_nbf, aux_l,
                  Q_pair, Q_aux, threshold, sao, projectors, dims):
    nthreads = bin_off.shape[0] - 1
    maxA, maxC, lmaxA, lmaxC, maxpp = dims[0], dims[1], dims[2], dims[3], dims[4]
    roots = np.zeros((nthreads, MAX_RYS_ROOTS))
    weights = np.zeros((nthreads, MAX_RYS_ROOTS))
    gx = np.zeros((nthreads, 2 * lmaxA + 1, lmaxC + 1))
    gy = np.zeros((nthreads, 2 * lmaxA + 1, lmaxC + 1))
    gz = np.zeros((nthreads, 2 * lmaxA + 1, lmaxC + 1))
    Sx = np.zeros((nthreads, lmaxA + 1, lmaxA + 1, lmaxC + 1))
    Sy = np.zeros((nthreads, lmaxA + 1, lmaxA + 1, lmaxC + 1))
    Sz = np.zeros((nthreads, lmaxA + 1, lmaxA + 1, lmaxC + 1))
    blk = np.zeros((nthreads, maxA, maxA, maxC))
    tmp = np.zeros((nthreads, maxA, maxA, maxC))
    blk2 = np.zeros((nthreads, maxA, maxA, maxC))
    ppf = np.zeros((nthreads, 6, maxpp))
    ppi = np.zeros((nthreads, 2, maxpp), dtype=np.int64)
    comp = np.zeros((nthreads, 9, max(maxA, maxC)), dtype=np.int64)
    for tid in prange(nthreads):
      for idx in range(bin_off[tid], bin_off[tid + 1]):
        p = bin_items[idx]
        I = pair_I[p]
        J = pair_J[p]
        nr = pair_nrows[p]
        nc = pair_ncols[p]
        o = pair_offset[p]
        rows = values[o:o + nr * nc].reshape((nr, nc))
        _compute_pair_rows(I, J, rows,
                           bfs_coords, bfs_lmn, bfs_nprim, bfs_expnts, bf_coef, shell_off, shell_nbf, shell_l,
                           aux_coords, aux_lmn, aux_nprim, aux_expnts, aux_coef, aux_off, aux_nbf, aux_l,
                           Q_pair[p], Q_aux, threshold, sao, projectors,
                           roots[tid], weights[tid], gx[tid], gy[tid], gz[tid], Sx[tid], Sy[tid], Sz[tid],
                           blk[tid], tmp[tid], blk2[tid],
                           ppf[tid, 0], ppf[tid, 1], ppf[tid, 2], ppf[tid, 3], ppf[tid, 4], ppf[tid, 5],
                           ppi[tid, 0], ppi[tid, 1],
                           comp[tid, 0], comp[tid, 1], comp[tid, 2], comp[tid, 3], comp[tid, 4], comp[tid, 5],
                           comp[tid, 6], comp[tid, 7], comp[tid, 8])


@njit(parallel=True, cache=True, fastmath=True, error_model="numpy", nogil=True, boundscheck=False)
def _gamma_pass(bin_off, bin_items, pair_I, pair_J, pair_offset, pair_nrows, pair_ncols, values, dmat, sqrt_ints4c2e_diag,
                strict_schwarz, naux,
                bfs_coords, bfs_lmn, bfs_nprim, bfs_expnts, bf_coef, shell_off, shell_nbf, shell_l,
                aux_coords, aux_lmn, aux_nprim, aux_expnts, aux_coef, aux_off, aux_nbf, aux_l,
                Q_pair, Q_aux, threshold, sao, projectors, dims, max_nrows):
    """gamma_P = sum_{i>=j} w_ij D_ij (ij|P); uncached pairs (offset < 0) are recomputed on the fly."""
    nthreads = bin_off.shape[0] - 1
    maxA, maxC, lmaxA, lmaxC, maxpp = dims[0], dims[1], dims[2], dims[3], dims[4]
    nsh_aux = aux_off.shape[0]
    partial = np.zeros((nthreads, naux))
    roots = np.zeros((nthreads, MAX_RYS_ROOTS))
    weights = np.zeros((nthreads, MAX_RYS_ROOTS))
    gx = np.zeros((nthreads, 2 * lmaxA + 1, lmaxC + 1))
    gy = np.zeros((nthreads, 2 * lmaxA + 1, lmaxC + 1))
    gz = np.zeros((nthreads, 2 * lmaxA + 1, lmaxC + 1))
    Sx = np.zeros((nthreads, lmaxA + 1, lmaxA + 1, lmaxC + 1))
    Sy = np.zeros((nthreads, lmaxA + 1, lmaxA + 1, lmaxC + 1))
    Sz = np.zeros((nthreads, lmaxA + 1, lmaxA + 1, lmaxC + 1))
    blk = np.zeros((nthreads, maxA, maxA, maxC))
    tmp = np.zeros((nthreads, maxA, maxA, maxC))
    blk2 = np.zeros((nthreads, maxA, maxA, maxC))
    ppf = np.zeros((nthreads, 6, maxpp))
    ppi = np.zeros((nthreads, 2, maxpp), dtype=np.int64)
    comp = np.zeros((nthreads, 9, max(maxA, maxC)), dtype=np.int64)
    rowbuf = np.zeros((nthreads, max_nrows, naux))
    klist = np.zeros((nthreads, 2, nsh_aux), dtype=np.int64)
    for tid in prange(nthreads):
      for idx in range(bin_off[tid], bin_off[tid + 1]):
        p = bin_items[idx]
        I = pair_I[p]
        J = pair_J[p]
        nr = pair_nrows[p]
        nc = pair_ncols[p]
        o = pair_offset[p]
        if o >= 0:
            rows = values[o:o + nr * nc].reshape((nr, nc))
        else:
            rows = rowbuf[tid]
            _compute_pair_rows(I, J, rows,
                               bfs_coords, bfs_lmn, bfs_nprim, bfs_expnts, bf_coef, shell_off, shell_nbf, shell_l,
                               aux_coords, aux_lmn, aux_nprim, aux_expnts, aux_coef, aux_off, aux_nbf, aux_l,
                               Q_pair[p], Q_aux, threshold, sao, projectors,
                               roots[tid], weights[tid], gx[tid], gy[tid], gz[tid], Sx[tid], Sy[tid], Sz[tid],
                               blk[tid], tmp[tid], blk2[tid],
                               ppf[tid, 0], ppf[tid, 1], ppf[tid, 2], ppf[tid, 3], ppf[tid, 4], ppf[tid, 5],
                               ppi[tid, 0], ppi[tid, 1],
                               comp[tid, 0], comp[tid, 1], comp[tid, 2], comp[tid, 3], comp[tid, 4], comp[tid, 5],
                               comp[tid, 6], comp[tid, 7], comp[tid, 8])
        # significant aux shells of this pair (replayed shell-level test)
        nk = 0
        Qp = Q_pair[p]
        for K in range(nsh_aux):
            if Qp * Q_aux[K] > threshold:
                klist[tid, 0, nk] = aux_off[K]
                klist[tid, 1, nk] = aux_nbf[K]
                nk += 1
        a0 = shell_off[I]
        b0 = shell_off[J]
        nA = shell_nbf[I]
        nB = shell_nbf[J]
        diag_pair = (I == J)
        acc = partial[tid]
        for ia in range(nA):
            i = a0 + ia
            ibmax = ia + 1 if diag_pair else nB
            for ib in range(ibmax):
                j = b0 + ib
                sij = sqrt_ints4c2e_diag[i, j]
                if strict_schwarz and sij * sij < STRICT_PAIR_CUTOFF:
                    continue
                coef = dmat[i, j] if i == j else dmat[i, j] + dmat[j, i]
                r = (ia * (ia + 1)) // 2 + ib if diag_pair else ia * nB + ib
                col = 0
                for kk in range(nk):
                    k0 = klist[tid, 0, kk]
                    nC = klist[tid, 1, kk]
                    for c in range(nC):
                        acc[k0 + c] += coef * rows[r, col + c]
                    col += nC
    gamma = np.zeros(naux)
    for t in range(nthreads):
        for k in range(naux):
            gamma[k] += partial[t, k]
    return gamma


@njit(parallel=True, cache=True, fastmath=True, error_model="numpy", nogil=True, boundscheck=False)
def _J_pass(bin_off, bin_items, pair_I, pair_J, pair_offset, pair_nrows, pair_ncols, values, coeff, sqrt_ints4c2e_diag,
            strict_schwarz, nao,
            bfs_coords, bfs_lmn, bfs_nprim, bfs_expnts, bf_coef, shell_off, shell_nbf, shell_l,
            aux_coords, aux_lmn, aux_nprim, aux_expnts, aux_coef, aux_off, aux_nbf, aux_l,
            Q_pair, Q_aux, threshold, sao, projectors, dims, max_nrows):
    """J_ij = sum_P (ij|P) c_P for all stored pairs (and J_ji by symmetry)."""
    nthreads = bin_off.shape[0] - 1
    maxA, maxC, lmaxA, lmaxC, maxpp = dims[0], dims[1], dims[2], dims[3], dims[4]
    nsh_aux = aux_off.shape[0]
    naux = coeff.shape[0]
    J = np.zeros((nao, nao))
    roots = np.zeros((nthreads, MAX_RYS_ROOTS))
    weights = np.zeros((nthreads, MAX_RYS_ROOTS))
    gx = np.zeros((nthreads, 2 * lmaxA + 1, lmaxC + 1))
    gy = np.zeros((nthreads, 2 * lmaxA + 1, lmaxC + 1))
    gz = np.zeros((nthreads, 2 * lmaxA + 1, lmaxC + 1))
    Sx = np.zeros((nthreads, lmaxA + 1, lmaxA + 1, lmaxC + 1))
    Sy = np.zeros((nthreads, lmaxA + 1, lmaxA + 1, lmaxC + 1))
    Sz = np.zeros((nthreads, lmaxA + 1, lmaxA + 1, lmaxC + 1))
    blk = np.zeros((nthreads, maxA, maxA, maxC))
    tmp = np.zeros((nthreads, maxA, maxA, maxC))
    blk2 = np.zeros((nthreads, maxA, maxA, maxC))
    ppf = np.zeros((nthreads, 6, maxpp))
    ppi = np.zeros((nthreads, 2, maxpp), dtype=np.int64)
    comp = np.zeros((nthreads, 9, max(maxA, maxC)), dtype=np.int64)
    rowbuf = np.zeros((nthreads, max_nrows, naux))
    klist = np.zeros((nthreads, 2, nsh_aux), dtype=np.int64)
    for tid in prange(nthreads):
      for idx in range(bin_off[tid], bin_off[tid + 1]):
        p = bin_items[idx]
        I = pair_I[p]
        J_ = pair_J[p]
        nr = pair_nrows[p]
        nc = pair_ncols[p]
        o = pair_offset[p]
        if o >= 0:
            rows = values[o:o + nr * nc].reshape((nr, nc))
        else:
            rows = rowbuf[tid]
            _compute_pair_rows(I, J_, rows,
                               bfs_coords, bfs_lmn, bfs_nprim, bfs_expnts, bf_coef, shell_off, shell_nbf, shell_l,
                               aux_coords, aux_lmn, aux_nprim, aux_expnts, aux_coef, aux_off, aux_nbf, aux_l,
                               Q_pair[p], Q_aux, threshold, sao, projectors,
                               roots[tid], weights[tid], gx[tid], gy[tid], gz[tid], Sx[tid], Sy[tid], Sz[tid],
                               blk[tid], tmp[tid], blk2[tid],
                               ppf[tid, 0], ppf[tid, 1], ppf[tid, 2], ppf[tid, 3], ppf[tid, 4], ppf[tid, 5],
                               ppi[tid, 0], ppi[tid, 1],
                               comp[tid, 0], comp[tid, 1], comp[tid, 2], comp[tid, 3], comp[tid, 4], comp[tid, 5],
                               comp[tid, 6], comp[tid, 7], comp[tid, 8])
        nk = 0
        Qp = Q_pair[p]
        for K in range(nsh_aux):
            if Qp * Q_aux[K] > threshold:
                klist[tid, 0, nk] = aux_off[K]
                klist[tid, 1, nk] = aux_nbf[K]
                nk += 1
        a0 = shell_off[I]
        b0 = shell_off[J_]
        nA = shell_nbf[I]
        nB = shell_nbf[J_]
        diag_pair = (I == J_)
        for ia in range(nA):
            i = a0 + ia
            ibmax = ia + 1 if diag_pair else nB
            for ib in range(ibmax):
                j = b0 + ib
                sij = sqrt_ints4c2e_diag[i, j]
                if strict_schwarz and sij * sij < STRICT_PAIR_CUTOFF:
                    continue
                r = (ia * (ia + 1)) // 2 + ib if diag_pair else ia * nB + ib
                val = 0.0
                col = 0
                for kk in range(nk):
                    k0 = klist[tid, 0, kk]
                    nC = klist[tid, 1, kk]
                    for c in range(nC):
                        val += rows[r, col + c] * coeff[k0 + c]
                    col += nC
                J[i, j] = val
                J[j, i] = val
    return J


# ----------------------------------------------------------------------------
# Python drivers
# ----------------------------------------------------------------------------
def _bf_coef_table(basis, packed):
    """contraction coefficient x primitive norm x contraction norm, (nbf, maxnprim)."""
    _, contr_norms, _, _, coeffs, prim_norms, _ = packed
    return np.ascontiguousarray(contr_norms[:, None] * coeffs * prim_norms)


def build_plan(basis, auxbasis, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold, strict_schwarz,
               sao=False, max_memory_gb=None, ncores=None):
    """
    Screen, rank and (within the memory budget) evaluate the shell-pair row blocks.

    Parameters
    ----------
    basis, auxbasis : Basis
    sqrt_ints4c2e_diag : (nao, nao) ndarray   sqrt(|(ij|ij)|)
    sqrt_diag_ints2c2e : (naux,) ndarray      sqrt(|(P|P)|) of the (SAO: projected) metric
    threshold : float                          Schwarz threshold (block level)
    strict_schwarz : bool                      pair cut-off ``(ij|ij) < STRICT_PAIR_CUTOFF``
    sao : bool                                 project auxiliary shells with l >= 2
    max_memory_gb : float or None              budget for the cached row blocks; None = all, 0 = none
    ncores : int or None                       threads (defaults to numba.get_num_threads())

    Returns
    -------
    DFAlgo11Plan
    """
    plan = _plan_metadata(basis, auxbasis, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e,
                          threshold, strict_schwarz, sao, max_memory_gb, ncores)
    plan.values = np.zeros(plan.n_elements_cached, dtype=np.float64)
    if plan.work_build.size:
        bin_off, bin_items = _lpt_bins(plan.work_build, plan.build_cost, plan.nthreads)
        _build_cached(bin_off, bin_items, plan.pair_I, plan.pair_J, plan.pair_offset,
                      plan.pair_nrows, plan.pair_ncols, plan.values, *_kernel_args(plan), plan.dims)
    return plan


def _plan_metadata(basis, auxbasis, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold,
                   strict_schwarz, sao=False, max_memory_gb=None, ncores=None):
    """Shared CPU/GPU screening and cache selection; allocate no integral values."""
    plan = DFAlgo11Plan()
    nthreads = int(numba.get_num_threads() if ncores is None else max(1, ncores))
    plan.nthreads = nthreads
    plan.threshold = float(threshold)
    plan.strict_schwarz = bool(strict_schwarz)
    plan.sao = bool(sao)
    plan.nao = basis.bfs_nao
    plan.naux = auxbasis.bfs_nao

    packed = pack_basis_arrays(basis)
    apacked = pack_basis_arrays(auxbasis)
    (plan.bfs_coords, _, plan.bfs_lmn, plan.bfs_nprim, _, _, plan.bfs_expnts) = packed
    (plan.aux_coords, _, plan.aux_lmn, plan.aux_nprim, _, _, plan.aux_expnts) = apacked
    plan.bf_coef = _bf_coef_table(basis, packed)
    plan.aux_coef = _bf_coef_table(auxbasis, apacked)
    plan.shell_off, plan.shell_nbf, plan.shell_l = aux_shell_arrays(basis)
    plan.aux_off, plan.aux_nbf, plan.aux_l = aux_shell_arrays(auxbasis)
    lmaxA = int(plan.shell_l.max())
    lmaxC = int(plan.aux_l.max())
    plan.projectors = sao_aux_projectors(lmaxC) if sao else sao_aux_projectors(1)
    plan.sqrt_ints4c2e_diag = np.ascontiguousarray(sqrt_ints4c2e_diag, dtype=np.float64)

    sq2 = np.asarray(sqrt_diag_ints2c2e, dtype=np.float64)
    plan.Q_aux = np.array([sq2[k0:k0 + nk].max() for k0, nk in zip(plan.aux_off, plan.aux_nbf)], dtype=np.float64)

    nsh = plan.shell_off.shape[0]
    pair_I, pair_J = np.tril_indices(nsh)
    plan.pair_I = np.ascontiguousarray(pair_I, dtype=np.int64)
    plan.pair_J = np.ascontiguousarray(pair_J, dtype=np.int64)
    (plan.Q_pair, alive, plan.pair_nrows, plan.pair_ncols, cost, max_roots) = _pair_screening_info(
        plan.pair_I, plan.pair_J, plan.shell_off, plan.shell_nbf, plan.shell_l,
        plan.bfs_nprim, plan.bfs_expnts, plan.bfs_coords, plan.sqrt_ints4c2e_diag,
        plan.Q_aux, plan.aux_off, plan.aux_nbf, plan.aux_l, plan.aux_nprim,
        plan.threshold, plan.strict_schwarz)
    if alive.any() and int(max_roots[alive].max()) > MAX_RYS_ROOTS:
        raise ValueError('DF_algo=11 supports at most %d Rys roots (total angular momentum <= %d).'
                         % (MAX_RYS_ROOTS, 2 * MAX_RYS_ROOTS - 1))

    sig = np.nonzero(alive)[0]
    elems = plan.pair_nrows[sig] * plan.pair_ncols[sig]
    plan.n_pairs_total = int(nsh * (nsh + 1) // 2)
    plan.n_pairs_significant = int(sig.size)
    plan.n_elements_significant = int(elems.sum())

    # Caching: most expensive per stored element first, greedily within the budget.
    if max_memory_gb is None:
        budget = int(elems.sum())
    else:
        budget = int(max(0.0, float(max_memory_gb)) * 1e9 // 8)
    ratio = cost[sig] / np.maximum(elems, 1)
    order_ratio = sig[np.argsort(-ratio, kind='stable')]
    cached = np.zeros(plan.pair_I.shape[0], dtype=np.bool_)
    used = 0
    for p in order_ratio:
        n = int(plan.pair_nrows[p] * plan.pair_ncols[p])
        if used + n <= budget:
            cached[p] = True
            used += n
    plan.pair_offset = np.full(plan.pair_I.shape[0], -1, dtype=np.int64)
    cached_idx = np.nonzero(cached)[0]
    sizes = plan.pair_nrows[cached_idx] * plan.pair_ncols[cached_idx]
    plan.pair_offset[cached_idx] = np.concatenate(([0], np.cumsum(sizes)[:-1])).astype(np.int64) if cached_idx.size else np.zeros(0, dtype=np.int64)
    plan.n_pairs_cached = int(cached_idx.size)
    plan.n_elements_cached = int(sizes.sum())

    # Load balancing: explicit LPT assignment of shell pairs to one bin per thread.
    # Per-iteration cost: full evaluation for uncached pairs, contraction only for cached ones.
    uncached_idx = sig[~cached[sig]]
    plan.work_iter = sig.astype(np.int64)
    iter_cost = np.where(cached[sig], 2.0 * elems, cost[sig])
    plan.iter_cost = iter_cost
    plan.cost_uncached = float(cost[uncached_idx].sum())
    plan.cost_total = float(cost[sig].sum())
    plan.work_build = cached_idx.astype(np.int64)
    plan.build_cost = cost[cached_idx]
    plan._iter_bins = {}

    maxA = int(plan.shell_nbf.max())
    maxC = int(plan.aux_nbf.max())
    maxpp = int(plan.bfs_nprim.max()) ** 2
    plan.dims = np.array([maxA, maxC, lmaxA, lmaxC, maxpp], dtype=np.int64)
    # Row buffer for on-the-fly evaluation is only needed for uncached pairs.
    plan.max_nrows = int(plan.pair_nrows[uncached_idx].max()) if uncached_idx.size else 1

    return plan


def _iteration_bins(plan):
    """LPT bins of the per-iteration work for the current Numba thread count (cached per count)."""
    nthreads = int(numba.get_num_threads())
    if nthreads not in plan._iter_bins:
        plan._iter_bins[nthreads] = _lpt_bins(plan.work_iter, plan.iter_cost, nthreads)
    return plan._iter_bins[nthreads]


def _lpt_bins(items, costs, nbins):
    """
    Longest-processing-time-first assignment of ``items`` (with estimated ``costs``)
    to ``nbins`` bins.  Returns ``(bin_off, bin_items)`` in CSR form: bin ``b`` owns
    ``bin_items[bin_off[b]:bin_off[b+1]]``.  The Numba workqueue threading layer
    only schedules ``prange`` statically, so the balance is done here explicitly.
    """
    import heapq
    items = np.asarray(items, dtype=np.int64)
    nbins = max(1, int(nbins))
    order = np.argsort(-np.asarray(costs, dtype=np.float64), kind='stable')
    heap = [(0.0, b) for b in range(nbins)]
    heapq.heapify(heap)
    lists = [[] for _ in range(nbins)]
    for idx in order:
        load, b = heapq.heappop(heap)
        lists[b].append(int(items[idx]))
        heapq.heappush(heap, (load + float(costs[idx]), b))
    bin_off = np.zeros(nbins + 1, dtype=np.int64)
    bin_off[1:] = np.cumsum([len(l) for l in lists])
    bin_items = np.array([i for l in lists for i in l], dtype=np.int64)
    return bin_off, bin_items


def _kernel_args(plan):
    return (plan.bfs_coords, plan.bfs_lmn, plan.bfs_nprim, plan.bfs_expnts, plan.bf_coef,
            plan.shell_off, plan.shell_nbf, plan.shell_l,
            plan.aux_coords, plan.aux_lmn, plan.aux_nprim, plan.aux_expnts, plan.aux_coef,
            plan.aux_off, plan.aux_nbf, plan.aux_l,
            plan.Q_pair, plan.Q_aux, plan.threshold, plan.sao, plan.projectors)


def gamma_from_plan(plan, dmat):
    """``gamma_P = sum_ij D_ij (ij|P)`` (full double sum) from the plan; uncached blocks are recomputed."""
    dmat = np.ascontiguousarray(dmat, dtype=np.float64)
    bin_off, bin_items = _iteration_bins(plan)
    return _gamma_pass(bin_off, bin_items, plan.pair_I, plan.pair_J, plan.pair_offset, plan.pair_nrows, plan.pair_ncols,
                       plan.values, dmat, plan.sqrt_ints4c2e_diag, plan.strict_schwarz, plan.naux,
                       *_kernel_args(plan), plan.dims, plan.max_nrows)


def J_from_plan(plan, coeff):
    """Full symmetric Coulomb matrix ``J_ij = sum_P (ij|P) c_P`` from the plan; uncached blocks are recomputed."""
    coeff = np.ascontiguousarray(coeff, dtype=np.float64)
    bin_off, bin_items = _iteration_bins(plan)
    return _J_pass(bin_off, bin_items, plan.pair_I, plan.pair_J, plan.pair_offset, plan.pair_nrows, plan.pair_ncols,
                   plan.values, coeff, plan.sqrt_ints4c2e_diag, plan.strict_schwarz, plan.nao,
                   *_kernel_args(plan), plan.dims, plan.max_nrows)
