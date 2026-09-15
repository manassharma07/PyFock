"""
Density-fitting Coulomb term, algorithm 12 (``DF_algo=12``): the shell-blocked
three-center integrals of algorithm 11 for the near field, multipole expansions for the
far field (multipole-accelerated density fitting in the spirit of the continuous fast
multipole method).

The density-fitted Coulomb term needs, in every SCF iteration,

    gamma_P = sum_ij D_ij (ij|P)          and          J_ij = sum_P (ij|P) c_P ,

i.e. the Coulomb interaction of the *distributions* ``phi_i phi_j`` with the auxiliary
functions ``P``.  Whenever a distribution and an auxiliary function do not overlap, their
interaction is that of two classical charge distributions and is given exactly (up to an
``erfc``-type overlap error) by a finite double multipole expansion.  Algorithm 12 splits
the ``(ij|P)`` tensor accordingly:

* **Distributions.**  A significant shell pair is a sum of primitive pair products
  ``g_a g_b``, each a Gaussian ``exp(-p|r - P_ab|^2)`` times a polynomial of degree
  ``l_a + l_b`` centred at ``P_ab``.  The primitive pairs of a shell pair are grouped by
  centre ("groups"; the primitive pairs of two shells on the same atom share one centre).
  Every group has *exact, finite* multipole moments about its centre (orders
  ``l <= l_a + l_b``, :func:`~pyfock.Integrals.multipole_helpers.gaussian_product_moments`),
  computed once and stored per function pair: ``(l_a + l_b + 1)^2`` doubles per function
  pair and group, typically two orders of magnitude less than the far-field integrals they
  replace.  Groups are sorted into cubic boxes by their centre and, within a box, into
  extent classes; each (box, class) is a *branch* whose expansion centre is the midpoint of
  its members' bounding box.

* **Near field.**  For a shell pair and an auxiliary shell ``K`` the Rys kernel of
  :mod:`~pyfock.Integrals.df_algo11_helpers` evaluates the primitive pairs whose branch is
  not well separated from ``K``; the block is stored in the block-sparse row layout of
  algorithm 11 (auxiliary shells for which every primitive pair is far field are left out
  of the columns).  Schwarz screening, strict pair cut-off, spherical projection and the
  memory budget of algorithm 11 are unchanged.  Tight (core) primitive pairs, which
  dominate the Rys cost, are compact and become far field for almost every auxiliary
  shell even when their shell pair also contains diffuse primitives.

* **Far field, density side.**  Every function pair carries the moments of its groups
  translated to their branch centre (multipole-to-multipole, truncated at ``lmax``; see
  *Pre-translated moments* below).  The ``gamma`` pass accumulates them weighted by the
  density into one moment vector per branch, and the ``J`` pass contracts them with the
  branch's local (Taylor) expansion.

* **Far field, auxiliary side.**  An auxiliary Gaussian ``x^a y^b z^c exp(-q r^2)`` on
  atom ``A`` has exact moments about ``A`` only up to ``l = a + b + c`` (in SAO mode only
  ``l = l_shell``).  The far-field potential of the fitted density is therefore generated
  by one exact point multipole of order ``<= l_aux`` per (atom, auxiliary shell), and the
  far-field ``gamma_P`` is the contraction of the local expansion of the density's
  potential at ``A`` (orders ``<= l_aux``) with the moments of ``P``.  No box hierarchy is
  needed on this side; the branch <-> atom coupling costs ``(l_aux + 1)^2 (lmax + l_aux + 1)^2``
  per pair (one outer product with the irregular harmonics of the branch-atom vector) plus
  one sparse coupling contraction per branch (:mod:`~pyfock.Integrals.multipole_helpers`).

* **Well-separatedness.**  A branch ``b`` (centre ``B``, radius ``r_b`` = largest distance
  of a member centre from ``B``, most diffuse member exponent ``p_b``, largest member
  "charge" ``Q_b``) and an auxiliary shell ``K`` on atom ``A`` (smallest exponent ``q_K``,
  charge ``Q_K``) interact through multipoles when

      |A - B| - r_b >= erfc^-1(precision / (Q_b Q_K)) sqrt(1/p_b + 1/q_K)     (overlap)
      |A - B|       >= separation * r_b                                       (truncation).

  The first line keeps the ``erfc``-type deviation of every member's interaction from its
  multipole form below ``precision`` (times ``Q_b Q_K / d``, i.e. relative to the size of a
  unit-charge interaction); the "charge" of a Gaussian is its coefficient times the
  Gaussian-product prefactor and normalization volume.  The second line bounds the
  truncation error of the branch expansions, which converge like ``(r_b / |A - B|)^(lmax+1)``.

* **Profitability.**  The multipole path costs ``(lmax + 1)^2`` operations per function pair
  and SCF iteration, the direct contraction one per far-field auxiliary function.  A group is
  therefore only expanded when it has at least ``break_even * (lmax + 1)^2`` far-field
  auxiliary functions; below that its auxiliary shells stay near field.  Small molecules, where
  the far field is a thin shell of distant functions, degrade gracefully to algorithm 11 instead
  of paying for expansions that buy nothing.

* **Pre-translated moments.**  The moments of every function pair are translated to their
  branch centre *once*, at build time (:func:`_build_translated_moments`), instead of once per
  SCF iteration.  Each SCF iteration then only needs a dense length-``(lmax + 1)^2`` axpy per
  row for ``gamma`` and the matching dot product for ``J``, which is several times cheaper than
  the sparse translation it replaces and vectorizes.  This costs ``(lmax + 1)^2`` doubles per
  function pair and branch, which the break-even rule keeps below the far-field integrals it
  replaces but which is not free; ``low_memory=True`` drops it and translates every group again
  in each pass, trading iteration time for that storage.  Both give the same numbers to
  rounding: local-to-local is the exact adjoint of multipole-to-multipole.

* **Consistency.**  The same moments, translations and coupling tensors are used for
  ``gamma`` and for ``J`` (local-to-local is the exact adjoint of multipole-to-multipole
  under the same truncation), so both derive from one approximate ``(ij|P)`` tensor and
  the SCF energy stays a proper functional of the density.

Parameters (``DFT.multipole_options``): ``precision`` (default 1e-10), ``lmax`` (12),
``box_size`` in bohr (2.5), ``separation`` (4.0), ``class_factor`` (4, geometric factor of the
extent classes within a box; larger values give fewer branches), ``break_even`` (1, see the
profitability rule above) and ``low_memory`` (False; see *Pre-translated moments* above).
:func:`build_plan` reports the far-field fractions and the storage; :func:`approx_block` and
:func:`exact_block` give one ``(ij|P)`` block as the passes see it and exactly, for validation.
"""

import numpy as np
import numba
from numba import njit, prange
from scipy.special import erfcinv
from timeit import default_timer as timer

from .rys_helpers import Roots, Recur_3c2e_new, Shift_3c2e
from .df_algo10_helpers import (
    STRICT_PAIR_CUTOFF, EXP_ARG_CUTOFF, MAX_RYS_ROOTS,
    pack_basis_arrays, aux_shell_arrays, sao_aux_projectors)
from .df_algo11_helpers import _lpt_bins, _bf_coef_table
from . import multipole_helpers as mp

__all__ = ['DFAlgo12Plan', 'DEFAULT_OPTIONS', 'build_plan', 'gamma_from_plan', 'J_from_plan',
           'approx_block', 'exact_block']

PI = 3.141592653589793

DEFAULT_OPTIONS = dict(precision=1e-10, lmax=12, box_size=2.5, separation=4.0, class_factor=4.0,
                       break_even=1.0, low_memory=False)

_TABLE_CACHE = {}


def _tables(l_small, l_big):
    key = (int(l_small), int(l_big))
    if key not in _TABLE_CACHE:
        _TABLE_CACHE[key] = mp.translation_table(l_small, l_big)
    return _TABLE_CACHE[key]


def _polynomials(lmax):
    key = ('poly', int(lmax))
    if key not in _TABLE_CACHE:
        _TABLE_CACHE[key] = mp.regular_harmonic_polynomials(lmax)
    return _TABLE_CACHE[key]


# ----------------------------------------------------------------------------
# Plan object
# ----------------------------------------------------------------------------
class DFAlgo12Plan:
    """
    Everything the per-iteration passes need: packed basis data, the shell-pair work
    list with its screening data, the cached near-field row blocks, the far-field
    moments and the branch/atom interaction lists.  Created by :func:`build_plan`;
    consumed by :func:`gamma_from_plan` and :func:`J_from_plan`.
    """

    def __init__(self):
        self.values = None

    @property
    def memory_gb(self):
        return 0.0 if self.values is None else self.values.nbytes / 1e9

    @property
    def moments_gb(self):
        """Far-field storage: the branch-centred row moments plus the group moments they come from."""
        return (self.Mtil.nbytes + self.moments.nbytes) / 1e9

    @property
    def fraction_cached(self):
        if self.n_elements_nf == 0:
            return 1.0
        return self.n_elements_cached / self.n_elements_nf

    @property
    def fraction_far_field(self):
        """Far-field share of the significant ``(ij|P)`` elements (those algorithm 11 would store)."""
        if self.n_elements_significant == 0:
            return 0.0
        return 1.0 - self.n_elements_nf / self.n_elements_significant

    @property
    def fraction_far_field_work(self):
        """Far-field share of the Rys work (primitive triples) of the significant integrals."""
        if self.cost_all == 0:
            return 0.0
        return 1.0 - self.cost_nf / self.cost_all

    def summary(self):
        return (
            f"DF_algo=12: {self.n_pairs_significant} of {self.n_pairs_total} shell pairs significant; "
            f"far field: {100 * self.fraction_far_field:.1f}% of the {self.n_elements_significant} significant (ij|P) "
            f"and {100 * self.fraction_far_field_work:.1f}% of their Rys work "
            f"({self.n_branches} branches in {self.n_boxes} boxes of {self.box_size:.2f} bohr, lmax={self.lmax}, "
            f"{self.n_groups} moment centres, {self.moments_gb:.3f} GB of branch-centred moments); "
            + ('low memory: the branch-centred moments are re-translated every iteration; '
               if self.low_memory else '')
            + f"near field: {self.n_elements_nf} elements ({8e-9 * self.n_elements_nf:.3f} GB), cached "
            f"{100 * self.fraction_cached:.1f}% ({self.memory_gb:.3f} GB)")


# ----------------------------------------------------------------------------
# Near-field kernel: row block of one shell pair (algorithm 11 kernel with far-field masks)
# ----------------------------------------------------------------------------
@njit(nogil=True, cache=True, fastmath=True, error_model="numpy", boundscheck=False)
def _compute_pair_rows(I, J, rows,
                       bfs_coords, bfs_lmn, bfs_nprim, bfs_expnts, bf_coef, shell_off, shell_nbf, shell_l,
                       aux_coords, aux_lmn, aux_nprim, aux_expnts, aux_coef, aux_off, aux_nbf, aux_l,
                       Qp, Q_aux, threshold, sao, projectors, pp_group_p, grp_branch_p, ff,
                       roots, weights, gx, gy, gz, Sx, Sy, Sz, blk, tmp, blk2,
                       pp_alpha, pp_beta, pp_gamma, pp_px, pp_py, pp_pz, pp_ip, pp_jp, pp_br,
                       ax_, ay_, az_, bx_, by_, bz_, cx_, cy_, cz_):
    """
    Fill ``rows[r, :ncols]`` with the near-field ``(ij|P)`` of shell pair ``(I, J)``:
    columns run over the Schwarz-significant auxiliary shells with at least one near-field
    primitive pair, in increasing order; within a column only the primitive pairs whose
    branch is not far field for that shell (``ff[branch, K]``) are evaluated.  Returns the
    number of columns written.  Row convention as in algorithm 11.
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
            pp_br[npp] = grp_branch_p[pp_group_p[npp]]
            npp += 1

    col = 0
    for K in range(aux_off.shape[0]):
        if Qp * Q_aux[K] <= threshold:
            continue
        nf_any = False
        for q in range(npp):
            if not ff[pp_br[q], K]:
                nf_any = True
                break
        if not nf_any:
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
            if ff[pp_br[q], K]:
                continue
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

                for ia in range(nA):
                    ca = bf_coef[a0 + ia, ip]
                    for ib in range(nB):
                        cab = ca * bf_coef[b0 + ib, jp]
                        for ic in range(nC):
                            blk[ia, ib, ic] += cab * aux_coef[c0 + ic, kp] * tmp[ia, ib, ic]

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

        for ia in range(nA):
            ibmax = ia + 1 if diag_pair else nB
            for ib in range(ibmax):
                r = (ia * (ia + 1)) // 2 + ib if diag_pair else ia * nB + ib
                for ic in range(nC):
                    rows[r, col + ic] = src[ia, ib, ic]
        col += nC
    return col


@njit(parallel=True, cache=True, fastmath=True, error_model="numpy", nogil=True, boundscheck=False)
def _near_field_column_map(work, Q_pair, Q_aux, threshold, aux_off, aux_nbf, ngroups, grp_branch, ff,
                           col_off, cols):
    """
    Auxiliary-function index of every near-field column of every pair, in the order the row
    blocks store them (CSR over pairs).  Deciding this once at build time rather than in every
    pass matters: the test runs over all auxiliary shells and, in this algorithm, over the
    groups of the pair as well, which otherwise costs more per iteration than the contraction.
    """
    for w in prange(work.shape[0]):
        p = work[w]
        Qp = Q_pair[p]
        o = col_off[p]
        n = 0
        for K in range(aux_off.shape[0]):
            if Qp * Q_aux[K] <= threshold:
                continue
            nf_any = False
            for g in range(ngroups[p]):
                if not ff[grp_branch[p, g], K]:
                    nf_any = True
                    break
            if not nf_any:
                continue
            k0 = aux_off[K]
            for c in range(aux_nbf[K]):
                cols[o + n + c] = k0 + c
            n += aux_nbf[K]


# ----------------------------------------------------------------------------
# Geometry of the distributions: primitive-pair centres, exponents, charges, extents
# ----------------------------------------------------------------------------
@njit(parallel=True, cache=True, fastmath=True, error_model="numpy", nogil=True)
def _pair_geometry(pair_I, pair_J, shell_off, shell_nbf, shell_l, bfs_nprim, bfs_expnts, bfs_coords, bf_coef,
                   inv_precision, tcap, maxpp):
    """
    Per shell pair: number of groups (primitive-pair centres) and of surviving primitive
    pairs (kernel order); per group: centre, smallest exponent, largest "charge" and
    Gaussian extent ``erfc^-1(precision/Q)/sqrt(p)``; per primitive pair: its group.
    """
    npairs = pair_I.shape[0]
    ngroups = np.zeros(npairs, dtype=np.int64)
    npp_arr = np.zeros(npairs, dtype=np.int64)
    grp_center = np.zeros((npairs, maxpp, 3))
    grp_pmin = np.zeros((npairs, maxpp))
    grp_qmax = np.zeros((npairs, maxpp))
    grp_extent = np.zeros((npairs, maxpp))
    pp_group = np.zeros((npairs, maxpp), dtype=np.int64)
    for p in prange(npairs):
        I = pair_I[p]
        J = pair_J[p]
        a0 = shell_off[I]
        b0 = shell_off[J]
        nA = shell_nbf[I]
        nB = shell_nbf[J]
        lA = shell_l[I]
        lB = shell_l[J]
        nprimA = bfs_nprim[a0]
        nprimB = bfs_nprim[b0]
        Ax = bfs_coords[a0, 0]
        Ay = bfs_coords[a0, 1]
        Az = bfs_coords[a0, 2]
        Bx = bfs_coords[b0, 0]
        By = bfs_coords[b0, 1]
        Bz = bfs_coords[b0, 2]
        ijsq = (Ax - Bx) ** 2 + (Ay - By) ** 2 + (Az - Bz) ** 2
        same_center = ijsq < 1e-20
        npp = 0
        ng = 0
        for ip in range(nprimA):
            alpha = bfs_expnts[a0, ip]
            camax = 0.0
            for ia in range(nA):
                v = abs(bf_coef[a0 + ia, ip])
                if v > camax:
                    camax = v
            for jp in range(nprimB):
                beta = bfs_expnts[b0, jp]
                gamma_p = alpha + beta
                arg = alpha * beta / gamma_p * ijsq
                if arg > EXP_ARG_CUTOFF:
                    continue
                cbmax = 0.0
                for ib in range(nB):
                    v = abs(bf_coef[b0 + ib, jp])
                    if v > cbmax:
                        cbmax = v
                # "charge": coefficients x product prefactor x Gaussian volume x polynomial scale
                Q = camax * cbmax * np.exp(-arg) * (PI / gamma_p) ** 1.5
                if lA + lB > 0:
                    Q *= (0.5 * (lA + lB) / gamma_p) ** (0.5 * (lA + lB))
                x = Q * inv_precision
                r_ext = 0.0
                if x > 1.0:
                    t = _erfcinv_approx(1.0 / x)
                    if t > tcap:
                        t = tcap
                    r_ext = t / np.sqrt(gamma_p)
                if same_center:
                    g = 0
                    if ng == 0:
                        grp_center[p, 0, 0] = Ax
                        grp_center[p, 0, 1] = Ay
                        grp_center[p, 0, 2] = Az
                        grp_pmin[p, 0] = gamma_p
                        grp_qmax[p, 0] = Q
                        grp_extent[p, 0] = r_ext
                        ng = 1
                    else:
                        if gamma_p < grp_pmin[p, 0]:
                            grp_pmin[p, 0] = gamma_p
                        if Q > grp_qmax[p, 0]:
                            grp_qmax[p, 0] = Q
                        if r_ext > grp_extent[p, 0]:
                            grp_extent[p, 0] = r_ext
                else:
                    g = ng
                    grp_center[p, g, 0] = (alpha * Ax + beta * Bx) / gamma_p
                    grp_center[p, g, 1] = (alpha * Ay + beta * By) / gamma_p
                    grp_center[p, g, 2] = (alpha * Az + beta * Bz) / gamma_p
                    grp_pmin[p, g] = gamma_p
                    grp_qmax[p, g] = Q
                    grp_extent[p, g] = r_ext
                    ng += 1
                pp_group[p, npp] = g
                npp += 1
        ngroups[p] = ng
        npp_arr[p] = npp
    return ngroups, npp_arr, grp_center, grp_pmin, grp_qmax, grp_extent, pp_group


@njit(cache=True, fastmath=True, error_model="numpy", nogil=True)
def _erfc(x):
    """Complementary error function (rational approximation; refined by the caller's Newton steps)."""
    z = abs(x)
    t = 1.0 / (1.0 + 0.5 * z)
    r = t * np.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (
        -0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
    if x >= 0.0:
        return r
    return 2.0 - r


@njit(cache=True, fastmath=True, error_model="numpy", nogil=True)
def _erfcinv_approx(y):
    """Inverse complementary error function for 0 < y < 1 (rational start, Newton refinement)."""
    if y >= 1.0:
        return 0.0
    w = -np.log(y * (2.0 - y))
    if w < 5.0:
        w = w - 2.5
        pp = 2.81022636e-08
        pp = 3.43273939e-07 + pp * w
        pp = -3.5233877e-06 + pp * w
        pp = -4.39150654e-06 + pp * w
        pp = 0.00021858087 + pp * w
        pp = -0.00125372503 + pp * w
        pp = -0.00417768164 + pp * w
        pp = 0.246640727 + pp * w
        pp = 1.50140941 + pp * w
    else:
        w = np.sqrt(w) - 3.0
        pp = -0.000200214257
        pp = 0.000100950558 + pp * w
        pp = 0.00134934322 + pp * w
        pp = -0.00367342844 + pp * w
        pp = 0.00573950773 + pp * w
        pp = -0.0076224613 + pp * w
        pp = 0.00943887047 + pp * w
        pp = 1.00167406 + pp * w
        pp = 2.83297682 + pp * w
    x = pp * (1.0 - y)
    for _ in range(3):
        f = _erfc(x) - y
        df = -2.0 / np.sqrt(PI) * np.exp(-x * x)
        if df == 0.0:
            break
        x = x - f / df
    return x


# ----------------------------------------------------------------------------
# Screening: which pairs are alive, and their near-field columns / cost
# ----------------------------------------------------------------------------
@njit(parallel=True, cache=True, fastmath=True, error_model="numpy", nogil=True)
def _pair_alive(pair_I, pair_J, shell_off, shell_nbf, bfs_nprim, bfs_expnts, bfs_coords,
                sqrt_ints4c2e_diag, Q_aux, threshold, strict_schwarz):
    npairs = pair_I.shape[0]
    nsh_aux = Q_aux.shape[0]
    Q_pair = np.zeros(npairs)
    alive = np.zeros(npairs, dtype=np.bool_)
    nrows = np.zeros(npairs, dtype=np.int64)
    Q_aux_max = 0.0
    for K in range(nsh_aux):
        if Q_aux[K] > Q_aux_max:
            Q_aux_max = Q_aux[K]
    for p in prange(npairs):
        I = pair_I[p]
        J = pair_J[p]
        a0 = shell_off[I]
        b0 = shell_off[J]
        nA = shell_nbf[I]
        nB = shell_nbf[J]
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
        if Q * Q_aux_max <= threshold:
            ok = False
        alive[p] = ok
    return Q_pair, alive, nrows


@njit(parallel=True, cache=True, fastmath=True, error_model="numpy", nogil=True)
def _pair_columns(work, pair_I, pair_J, shell_off, shell_nbf, shell_l, bfs_nprim, bfs_expnts, bfs_coords,
                  Q_pair, Q_aux, aux_off, aux_nbf, aux_l, aux_nprim, threshold, npp_arr, pp_group, grp_branch, ff):
    """Near-field columns, Rys cost (near field and total) and root count of every alive pair."""
    npairs = pair_I.shape[0]
    nsh_aux = aux_off.shape[0]
    ncols = np.zeros(npairs, dtype=np.int64)
    ncols_all = np.zeros(npairs, dtype=np.int64)
    cost = np.zeros(npairs)
    cost_all = np.zeros(npairs)
    max_roots = np.zeros(npairs, dtype=np.int64)
    maxpp = pp_group.shape[1]
    for w in prange(work.shape[0]):
        p = work[w]
        I = pair_I[p]
        J = pair_J[p]
        nA = shell_nbf[I]
        nB = shell_nbf[J]
        lA = shell_l[I]
        lB = shell_l[J]
        npp = npp_arr[p]
        Q = Q_pair[p]
        pp_br = np.zeros(maxpp, dtype=np.int64)
        for q in range(npp):
            pp_br[q] = grp_branch[p, pp_group[p, q]]
        nc = 0
        nc_all = 0
        cst = 0.0
        cst_all = 0.0
        mr = 0
        for K in range(nsh_aux):
            if Q * Q_aux[K] <= threshold:
                continue
            nC = aux_nbf[K]
            lC = aux_l[K]
            nprimC = aux_nprim[aux_off[K]]
            nroots = (lA + lB + lC) // 2 + 1
            unit = nprimC * (6.0 * nroots * (lA + lB + 1) * (lC + 1)
                             + 3.0 * nroots * (lA + 1) * (lB + 1) * (lC + 1) * (lB + 1)
                             + 3.0 * nroots * nA * nB * nC
                             + 2.0 * nA * nB * nC)
            nc_all += nC
            cst_all += npp * unit
            n_nf = 0
            for q in range(npp):
                if not ff[pp_br[q], K]:
                    n_nf += 1
            if n_nf == 0:
                continue
            if nroots > mr:
                mr = nroots
            nc += nC
            cst += n_nf * unit
        ncols[p] = nc
        ncols_all[p] = nc_all
        cost[p] = cst
        cost_all[p] = cst_all
        max_roots[p] = mr
    return ncols, ncols_all, cost, cost_all, max_roots


# ----------------------------------------------------------------------------
# Far-field moments of the groups
# ----------------------------------------------------------------------------
@njit(parallel=True, cache=True, fastmath=True, error_model="numpy", nogil=True, boundscheck=False)
def _build_moments(bin_off, bin_items, pair_I, pair_J, pair_nrows, mom_off, ngroups, pp_group, grp_center,
                   bfs_coords, bfs_lmn, bfs_nprim, bfs_expnts, bf_coef, shell_off, shell_nbf, shell_l,
                   mono_off, mono_abc, mono_coef, moments, maxA, lpair_max):
    """moments[mom_off[p] + (g*nrows + r)*nlm + lm] = exact moments about the group centre (pairs by LPT bins)."""
    nthreads = bin_off.shape[0] - 1
    nlm_max = (lpair_max + 1) * (lpair_max + 1)
    Xt = np.zeros((nthreads, maxA, maxA, lpair_max + 1))
    Yt = np.zeros((nthreads, maxA, maxA, lpair_max + 1))
    Zt = np.zeros((nthreads, maxA, maxA, lpair_max + 1))
    G = np.zeros((nthreads, 3 * lpair_max + 2))
    out = np.zeros((nthreads, maxA * maxA, nlm_max))
    comp = np.zeros((nthreads, 6, maxA), dtype=np.int64)
    for tid in prange(nthreads):
      for idx in range(bin_off[tid], bin_off[tid + 1]):
        p = bin_items[idx]
        I = pair_I[p]
        J = pair_J[p]
        a0 = shell_off[I]
        b0 = shell_off[J]
        nA = shell_nbf[I]
        nB = shell_nbf[J]
        lA = shell_l[I]
        lB = shell_l[J]
        ltot = lA + lB
        nlm = (ltot + 1) * (ltot + 1)
        nrows = pair_nrows[p]
        diag_pair = (I == J)
        ax_ = comp[tid, 0]
        ay_ = comp[tid, 1]
        az_ = comp[tid, 2]
        bx_ = comp[tid, 3]
        by_ = comp[tid, 4]
        bz_ = comp[tid, 5]
        for ia in range(nA):
            ax_[ia] = bfs_lmn[a0 + ia, 0]
            ay_[ia] = bfs_lmn[a0 + ia, 1]
            az_[ia] = bfs_lmn[a0 + ia, 2]
        for ib in range(nB):
            bx_[ib] = bfs_lmn[b0 + ib, 0]
            by_[ib] = bfs_lmn[b0 + ib, 1]
            bz_[ib] = bfs_lmn[b0 + ib, 2]
        Ax = bfs_coords[a0, 0]
        Ay = bfs_coords[a0, 1]
        Az = bfs_coords[a0, 2]
        Bx = bfs_coords[b0, 0]
        By = bfs_coords[b0, 1]
        Bz = bfs_coords[b0, 2]
        ijsq = (Ax - Bx) ** 2 + (Ay - By) ** 2 + (Az - Bz) ** 2
        base = mom_off[p]
        for k in range(ngroups[p] * nrows * nlm):
            moments[base + k] = 0.0
        q = 0
        for ip in range(bfs_nprim[a0]):
            alpha = bfs_expnts[a0, ip]
            for jp in range(bfs_nprim[b0]):
                beta = bfs_expnts[b0, jp]
                gamma_p = alpha + beta
                arg = alpha * beta / gamma_p * ijsq
                if arg > EXP_ARG_CUTOFF:
                    continue
                g = pp_group[p, q]
                q += 1
                px = grp_center[p, g, 0]
                py = grp_center[p, g, 1]
                pz = grp_center[p, g, 2]
                pref = np.exp(-arg)
                mp.gaussian_1d_moments(gamma_p, px - Ax, px - Bx, lA, lB, ltot, G[tid], Xt[tid])
                mp.gaussian_1d_moments(gamma_p, py - Ay, py - By, lA, lB, ltot, G[tid], Yt[tid])
                mp.gaussian_1d_moments(gamma_p, pz - Az, pz - Bz, lA, lB, ltot, G[tid], Zt[tid])
                mp.gaussian_product_moments(nA, nB, ax_, ay_, az_, bx_, by_, bz_, lA, lB,
                                            Xt[tid], Yt[tid], Zt[tid], mono_off, mono_abc, mono_coef,
                                            diag_pair, out[tid])
                for ia in range(nA):
                    ca = bf_coef[a0 + ia, ip] * pref
                    ibmax = ia + 1 if diag_pair else nB
                    for ib in range(ibmax):
                        cab = ca * bf_coef[b0 + ib, jp]
                        r = (ia * (ia + 1)) // 2 + ib if diag_pair else ia * nB + ib
                        o = base + (g * nrows + r) * nlm
                        for lm in range(nlm):
                            moments[o + lm] += cab * out[tid, r, lm]


def _aux_moments(auxbasis, apacked, aux_coef, aux_off, aux_nbf, aux_l, sao, projectors):
    """Exact moments of every auxiliary function about its own centre, ``(naux, (l_aux_max+1)^2)``."""
    aux_coords, _, aux_lmn, aux_nprim, _, _, aux_expnts = apacked
    lmax_aux = int(aux_l.max())
    nlm = (lmax_aux + 1) ** 2
    mono_off, mono_abc, mono_coef = _polynomials(lmax_aux)
    mom = np.zeros((auxbasis.bfs_nao, nlm))
    for K in range(aux_off.shape[0]):
        c0 = int(aux_off[K])
        nC = int(aux_nbf[K])
        lC = int(aux_l[K])
        blk = np.zeros((nC, nlm))
        for ic in range(nC):
            a, b, c = (int(v) for v in aux_lmn[c0 + ic])
            for kp in range(int(aux_nprim[c0])):
                q = aux_expnts[c0 + ic, kp]
                coef = aux_coef[c0 + ic, kp]
                nmax = 2 * lC + 1
                G = np.zeros(nmax + 1)
                G[0] = np.sqrt(np.pi / q)
                for n in range(2, nmax + 1, 2):
                    G[n] = (n - 1) * G[n - 2] / (2.0 * q)
                for lm in range((lC + 1) ** 2):
                    s = 0.0
                    for n in range(mono_off[lm], mono_off[lm + 1]):
                        s += mono_coef[n] * G[a + mono_abc[n, 0]] * G[b + mono_abc[n, 1]] * G[c + mono_abc[n, 2]]
                    blk[ic, lm] += coef * s
        if sao and lC >= 2:
            blk = projectors[lC, :nC, :nC] @ blk
        mom[c0:c0 + nC] = blk
    return mom


def _aux_charges(apacked, aux_coef, aux_off, aux_nbf, aux_l):
    """Smallest exponent and largest "charge" (coefficient x Gaussian volume x polynomial scale) of every auxiliary shell."""
    _, _, _, aux_nprim, _, _, aux_expnts = apacked
    nsh = aux_off.shape[0]
    qmin = np.zeros(nsh)
    Qmax = np.zeros(nsh)
    for K in range(nsh):
        c0 = int(aux_off[K])
        nC = int(aux_nbf[K])
        lC = int(aux_l[K])
        qm = np.inf
        Qm = 0.0
        for kp in range(int(aux_nprim[c0])):
            q = aux_expnts[c0, kp]
            cmax = np.abs(aux_coef[c0:c0 + nC, kp]).max()
            Q = cmax * (np.pi / q) ** 1.5
            if lC > 0:
                Q *= (0.5 * lC / q) ** (0.5 * lC)
            qm = min(qm, q)
            Qm = max(Qm, Q)
        qmin[K] = qm
        Qmax[K] = Qm
    return qmin, Qmax


def _aux_atoms(aux_coords, aux_off):
    """Atom index of every auxiliary shell (shells sharing a centre) and the atom coordinates."""
    nsh = aux_off.shape[0]
    centers = np.array([aux_coords[aux_off[K]] for K in range(nsh)])
    keys = np.round(centers, 8)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    natoms = int(inv.max()) + 1
    coords = np.zeros((natoms, 3))
    for K in range(nsh):
        coords[inv[K]] = centers[K]
    return inv.astype(np.int64), coords


# ----------------------------------------------------------------------------
# Profitability of the multipole path, and the branch-centred row moments
# ----------------------------------------------------------------------------
@njit(parallel=True, cache=True, fastmath=True, error_model="numpy", nogil=True)
def _group_far_field(work, Q_pair, Q_aux, aux_off, aux_nbf, threshold, ngroups, grp_branch, ff,
                     n_big, break_even, maxpp):
    """
    Whether expanding each group's far-field auxiliary shells is worth it.  Per function pair
    and SCF iteration the multipole path costs ``n_big = (lmax + 1)^2`` operations while the
    direct contraction costs one per far-field auxiliary function, so the expansion only pays
    off from ``break_even * n_big`` far-field auxiliary functions on.  Groups below that stay
    near field, which is what makes the algorithm degrade to DF_algo=11 on small molecules
    instead of paying for expansions that replace almost nothing.
    """
    npairs = Q_pair.shape[0]
    nsh_aux = aux_off.shape[0]
    out = np.zeros((npairs, maxpp), dtype=np.bool_)
    need = break_even * n_big
    for w in prange(work.shape[0]):
        p = work[w]
        Q = Q_pair[p]
        for g in range(ngroups[p]):
            b = grp_branch[p, g]
            cnt = 0
            for K in range(nsh_aux):
                if Q * Q_aux[K] > threshold and ff[b, K]:
                    cnt += aux_nbf[K]
            out[p, g] = cnt >= need
    return out


@njit(parallel=True, cache=True, fastmath=True, error_model="numpy", nogil=True, boundscheck=False)
def _build_translated_moments(bin_off, bin_items, entry_pair, entry_branch, entry_moff,
                              pair_I, pair_J, pair_nrows, mom_off, ngroups, grp_branch, grp_ff,
                              grp_center, branch_center, shell_l, moments, Mtil, lmax, n_big,
                              pair_off_tab, ent_LM, ent_coef):
    """
    ``Mtil[entry][r, :]`` are the moments of function pair ``r`` about the branch centre of
    ``entry``: the multipole-to-multipole translation of the exact group moments truncated at
    ``lmax``, summed over the groups of that pair that belong to that branch.  Translating once
    here rather than in every SCF iteration leaves the per-iteration far field as one dense
    length-``n_big`` axpy (``gamma``) and dot product (``J``) per function pair.
    """
    nthreads = bin_off.shape[0] - 1
    Rd = np.zeros((nthreads, n_big))
    for tid in prange(nthreads):
      for idx in range(bin_off[tid], bin_off[tid + 1]):
        e = bin_items[idx]
        p = entry_pair[e]
        b = entry_branch[e]
        o = entry_moff[e]
        nr = pair_nrows[p]
        lp = shell_l[pair_I[p]] + shell_l[pair_J[p]]
        nlm = (lp + 1) * (lp + 1)
        for k in range(nr * n_big):
            Mtil[o + k] = 0.0
        for g in range(ngroups[p]):
            if grp_branch[p, g] != b or not grp_ff[p, g]:
                continue
            mp.regular_harmonics(grp_center[p, g, 0] - branch_center[b, 0],
                                 grp_center[p, g, 1] - branch_center[b, 1],
                                 grp_center[p, g, 2] - branch_center[b, 2], lmax, Rd[tid])
            mbase = mom_off[p] + g * nr * nlm
            for r in range(nr):
                mp.translate_moments(moments[mbase + r * nlm: mbase + (r + 1) * nlm], lp, Rd[tid],
                                     lmax, n_big, pair_off_tab, ent_LM, ent_coef,
                                     Mtil[o + r * n_big: o + (r + 1) * n_big])


# ----------------------------------------------------------------------------
# Near-field build (cached blocks)
# ----------------------------------------------------------------------------
@njit(parallel=True, cache=True, fastmath=True, error_model="numpy", nogil=True, boundscheck=False)
def _build_cached(bin_off, bin_items, pair_I, pair_J, pair_offset, pair_nrows, pair_ncols, values,
                  bfs_coords, bfs_lmn, bfs_nprim, bfs_expnts, bf_coef, shell_off, shell_nbf, shell_l,
                  aux_coords, aux_lmn, aux_nprim, aux_expnts, aux_coef, aux_off, aux_nbf, aux_l,
                  Q_pair, Q_aux, threshold, sao, projectors, pp_group, grp_branch, ff, dims):
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
    ppi = np.zeros((nthreads, 3, maxpp), dtype=np.int64)
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
                           Q_pair[p], Q_aux, threshold, sao, projectors, pp_group[p], grp_branch[p], ff,
                           roots[tid], weights[tid], gx[tid], gy[tid], gz[tid], Sx[tid], Sy[tid], Sz[tid],
                           blk[tid], tmp[tid], blk2[tid],
                           ppf[tid, 0], ppf[tid, 1], ppf[tid, 2], ppf[tid, 3], ppf[tid, 4], ppf[tid, 5],
                           ppi[tid, 0], ppi[tid, 1], ppi[tid, 2],
                           comp[tid, 0], comp[tid, 1], comp[tid, 2], comp[tid, 3], comp[tid, 4], comp[tid, 5],
                           comp[tid, 6], comp[tid, 7], comp[tid, 8])


# ----------------------------------------------------------------------------
# Per-iteration passes over the shell pairs
# ----------------------------------------------------------------------------
@njit(parallel=True, cache=True, fastmath=True, error_model="numpy", nogil=True, boundscheck=False)
def _gamma_pass(bin_off, bin_items, pair_I, pair_J, pair_offset, pair_nrows, pair_ncols, values, dmat,
                sqrt_ints4c2e_diag, strict_schwarz, naux,
                bfs_coords, bfs_lmn, bfs_nprim, bfs_expnts, bf_coef, shell_off, shell_nbf, shell_l,
                aux_coords, aux_lmn, aux_nprim, aux_expnts, aux_coef, aux_off, aux_nbf, aux_l,
                Q_pair, Q_aux, threshold, sao, projectors, pp_group, grp_branch, ff, dims, max_nrows,
                ngroups, col_off, cols, entry_off, entry_branch, entry_moff, Mtil, n_branches, n_big,
                max_nrows_all, store_translated, mom_off, moments, grp_center, branch_center, lmax,
                pair_off_tab, ent_LM, ent_coef, lpair_max, ff_mode):
    """
    Near field: ``gamma_P += sum_{i>=j} w_ij D_ij (ij|P)`` over the stored/recomputed
    blocks.  Far field: the density-weighted branch-centred row moments accumulate into
    ``branch_mom[b, LM]``.  Returns ``(gamma_nf, branch_mom)``.
    """
    nthreads = bin_off.shape[0] - 1
    maxA, maxC, lmaxA, lmaxC, maxpp = dims[0], dims[1], dims[2], dims[3], dims[4]
    partial = np.zeros((nthreads, naux))
    partial_mom = np.zeros((nthreads, n_branches, n_big))
    local = np.zeros((nthreads, naux))
    rowcoef = np.zeros((nthreads, max_nrows_all))
    # only used when the branch-centred moments are not stored (low_memory)
    nacc = 1 if store_translated else maxpp
    nlm_acc = 1 if store_translated else (lpair_max + 1) * (lpair_max + 1)
    accg = np.zeros((nthreads, nacc, nlm_acc))
    Rd = np.zeros((nthreads, n_big))
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
    ppi = np.zeros((nthreads, 3, maxpp), dtype=np.int64)
    comp = np.zeros((nthreads, 9, max(maxA, maxC)), dtype=np.int64)
    rowbuf = np.zeros((nthreads, max_nrows, naux))
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
                               Q_pair[p], Q_aux, threshold, sao, projectors, pp_group[p], grp_branch[p], ff,
                               roots[tid], weights[tid], gx[tid], gy[tid], gz[tid], Sx[tid], Sy[tid], Sz[tid],
                               blk[tid], tmp[tid], blk2[tid],
                               ppf[tid, 0], ppf[tid, 1], ppf[tid, 2], ppf[tid, 3], ppf[tid, 4], ppf[tid, 5],
                               ppi[tid, 0], ppi[tid, 1], ppi[tid, 2],
                               comp[tid, 0], comp[tid, 1], comp[tid, 2], comp[tid, 3], comp[tid, 4], comp[tid, 5],
                               comp[tid, 6], comp[tid, 7], comp[tid, 8])
        oc = col_off[p]
        ncol = col_off[p + 1] - oc
        loc = local[tid]
        first = True
        rc = rowcoef[tid]
        for r in range(nr):
            rc[r] = 0.0
        a0 = shell_off[I]
        b0 = shell_off[J]
        nA = shell_nbf[I]
        nB = shell_nbf[J]
        diag_pair = (I == J)
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
                row = rows[r]
                if ff_mode != 2:
                    # the first contributing row writes the staging buffer, the rest accumulate:
                    # one pass over the columns fewer than zeroing it first
                    if first:
                        for c in range(ncol):
                            loc[c] = coef * row[c]
                        first = False
                    else:
                        for c in range(ncol):
                            loc[c] += coef * row[c]
                rc[r] = coef
        # staging keeps the inner loop a contiguous axpy, which vectorizes; accumulating straight
        # into the scattered gamma vector measured ~25% slower
        if not first:
            part = partial[tid]
            for c in range(ncol):
                part[cols[oc + c]] += loc[c]
        # far field: accumulate the density-weighted branch-centred row moments
        if ff_mode != 1:
            if store_translated:
                for e in range(entry_off[p], entry_off[p + 1]):
                    acc = partial_mom[tid, entry_branch[e]]
                    o = entry_moff[e]
                    for r in range(nr):
                        c = rc[r]
                        if c == 0.0:
                            continue
                        base = o + r * n_big
                        for LM in range(n_big):
                            acc[LM] += c * Mtil[base + LM]
            else:
                # low_memory: contract the group moments with the density first, then translate
                # each group once per pass instead of keeping the branch-centred row moments
                lp = shell_l[I] + shell_l[J]
                nlm = (lp + 1) * (lp + 1)
                ag = accg[tid]
                mbase = mom_off[p]
                for g in range(ngroups[p]):
                    b = grp_branch[p, g]
                    if b >= n_branches:
                        continue
                    for lm in range(nlm):
                        ag[g, lm] = 0.0
                    for r in range(nr):
                        c = rc[r]
                        if c == 0.0:
                            continue
                        mo = mbase + (g * nr + r) * nlm
                        for lm in range(nlm):
                            ag[g, lm] += c * moments[mo + lm]
                    mp.regular_harmonics(grp_center[p, g, 0] - branch_center[b, 0],
                                         grp_center[p, g, 1] - branch_center[b, 1],
                                         grp_center[p, g, 2] - branch_center[b, 2], lmax, Rd[tid])
                    mp.translate_moments(ag[g], lp, Rd[tid], lmax, n_big, pair_off_tab, ent_LM, ent_coef,
                                         partial_mom[tid, b])
    gamma = np.zeros(naux)
    for t in range(nthreads):
        for k in range(naux):
            gamma[k] += partial[t, k]
    branch_mom = np.zeros((n_branches, n_big))
    for t in range(nthreads):
        for b in range(n_branches):
            for k in range(n_big):
                branch_mom[b, k] += partial_mom[t, b, k]
    return gamma, branch_mom


@njit(parallel=True, cache=True, fastmath=True, error_model="numpy", nogil=True, boundscheck=False)
def _J_pass(bin_off, bin_items, pair_I, pair_J, pair_offset, pair_nrows, pair_ncols, values, coeff,
            sqrt_ints4c2e_diag, strict_schwarz, nao,
            bfs_coords, bfs_lmn, bfs_nprim, bfs_expnts, bf_coef, shell_off, shell_nbf, shell_l,
            aux_coords, aux_lmn, aux_nprim, aux_expnts, aux_coef, aux_off, aux_nbf, aux_l,
            Q_pair, Q_aux, threshold, sao, projectors, pp_group, grp_branch, ff, dims, max_nrows,
            ngroups, col_off, cols, entry_off, entry_branch, entry_moff, Mtil, branch_local, n_branches, n_big,
            max_nrows_all, store_translated, mom_off, moments, grp_center, branch_center, lmax,
            pair_off_tab, ent_LM, ent_coef, lpair_max, ff_mode):
    """``J_ij = sum_P (ij|P) c_P`` (near field from the blocks, far field from the branch local expansions)."""
    nthreads = bin_off.shape[0] - 1
    maxA, maxC, lmaxA, lmaxC, maxpp = dims[0], dims[1], dims[2], dims[3], dims[4]
    naux = coeff.shape[0]
    J = np.zeros((nao, nao))
    cloc = np.zeros((nthreads, naux))
    rowval = np.zeros((nthreads, max_nrows_all))
    # only used when the branch-centred moments are not stored (low_memory)
    nacc = 1 if store_translated else maxpp
    nlm_acc = 1 if store_translated else (lpair_max + 1) * (lpair_max + 1)
    Lg = np.zeros((nthreads, nacc, nlm_acc))
    Rd = np.zeros((nthreads, n_big))
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
    ppi = np.zeros((nthreads, 3, maxpp), dtype=np.int64)
    comp = np.zeros((nthreads, 9, max(maxA, maxC)), dtype=np.int64)
    rowbuf = np.zeros((nthreads, max_nrows, naux))
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
                               Q_pair[p], Q_aux, threshold, sao, projectors, pp_group[p], grp_branch[p], ff,
                               roots[tid], weights[tid], gx[tid], gy[tid], gz[tid], Sx[tid], Sy[tid], Sz[tid],
                               blk[tid], tmp[tid], blk2[tid],
                               ppf[tid, 0], ppf[tid, 1], ppf[tid, 2], ppf[tid, 3], ppf[tid, 4], ppf[tid, 5],
                               ppi[tid, 0], ppi[tid, 1], ppi[tid, 2],
                               comp[tid, 0], comp[tid, 1], comp[tid, 2], comp[tid, 3], comp[tid, 4], comp[tid, 5],
                               comp[tid, 6], comp[tid, 7], comp[tid, 8])
        oc = col_off[p]
        ncol = col_off[p + 1] - oc
        cl = cloc[tid]
        for c in range(ncol):
            cl[c] = coeff[cols[oc + c]]
        # far field: contract the branch local expansions with the branch-centred row moments
        rv = rowval[tid]
        for r in range(nr):
            rv[r] = 0.0
        if ff_mode != 1:
            if store_translated:
                for e in range(entry_off[p], entry_off[p + 1]):
                    bl = branch_local[entry_branch[e]]
                    o = entry_moff[e]
                    for r in range(nr):
                        acc = 0.0
                        base = o + r * n_big
                        for LM in range(n_big):
                            acc += Mtil[base + LM] * bl[LM]
                        rv[r] += acc
            else:
                # low_memory: re-expand the branch potential about each group centre per pass
                lp = shell_l[I] + shell_l[J_]
                nlm = (lp + 1) * (lp + 1)
                mbase = mom_off[p]
                for g in range(ngroups[p]):
                    b = grp_branch[p, g]
                    if b >= n_branches:
                        continue
                    mp.regular_harmonics(grp_center[p, g, 0] - branch_center[b, 0],
                                         grp_center[p, g, 1] - branch_center[b, 1],
                                         grp_center[p, g, 2] - branch_center[b, 2], lmax, Rd[tid])
                    mp.translate_local(branch_local[b], lmax, Rd[tid], lp, n_big, pair_off_tab,
                                       ent_LM, ent_coef, Lg[tid, g])
                    for r in range(nr):
                        acc = 0.0
                        mo = mbase + (g * nr + r) * nlm
                        for lm in range(nlm):
                            acc += moments[mo + lm] * Lg[tid, g, lm]
                        rv[r] += acc
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
                row = rows[r]
                val = rv[r]
                if ff_mode != 2:
                    for c in range(ncol):
                        val += row[c] * cl[c]
                J[i, j] = val
                J[j, i] = val
    return J


# ----------------------------------------------------------------------------
# Branch <-> atom coupling
# ----------------------------------------------------------------------------
@njit(parallel=True, cache=True, fastmath=True, error_model="numpy", nogil=True, boundscheck=False)
def _branches_to_atoms(branch_mom, branch_center, atom_coords, atom_shell_off, atom_shells, atom_lmax,
                       aux_l, ff, any_ff, lmax, n_big, l_aux_max, pair_off_tab, ent_LM, ent_coef, sign_big):
    """
    gamma direction: local expansions ``L_K[lm]`` (``l <= l_K``) at every auxiliary shell's
    atom from the moments of all its far-field branches.  The branch moments are first
    contracted with the coupling table, ``Y[b, lm, LM] = sum_jk (-1)^j A^{LM}_{lm,jk} M_b[jk]``,
    so that every branch/atom pair only costs one contraction with the irregular harmonics
    of the branch-atom vector.  Returns ``(nsh_aux, (l_aux_max+1)^2)``.
    """
    natoms = atom_coords.shape[0]
    n_branches = branch_mom.shape[0]
    nsh_aux = aux_l.shape[0]
    n_small = (l_aux_max + 1) * (l_aux_max + 1)
    n_irr = (lmax + l_aux_max + 1) * (lmax + l_aux_max + 1)
    Y = np.zeros((n_branches, n_small, n_irr))
    for b in prange(n_branches):
        for lm in range(n_small):
            for jk in range(n_big):
                v = sign_big[jk] * branch_mom[b, jk]
                if v == 0.0:
                    continue
                pidx = lm * n_big + jk
                for e in range(pair_off_tab[pidx], pair_off_tab[pidx + 1]):
                    Y[b, lm, ent_LM[e]] += ent_coef[e] * v
    L_K = np.zeros((nsh_aux, n_small))
    for A in prange(natoms):
        IR = np.zeros(n_irr)
        V = np.zeros(n_small)
        lA = atom_lmax[A]
        nsA = (lA + 1) * (lA + 1)
        nirrA = (lmax + lA + 1) * (lmax + lA + 1)
        for b in range(n_branches):
            if not any_ff[b, A]:
                continue
            mp.irregular_harmonics(branch_center[b, 0] - atom_coords[A, 0], branch_center[b, 1] - atom_coords[A, 1],
                                   branch_center[b, 2] - atom_coords[A, 2], lmax + lA, IR)
            for lm in range(nsA):
                s = 0.0
                for LM in range(nirrA):
                    s += IR[LM] * Y[b, lm, LM]
                V[lm] = s
            for s_ in range(atom_shell_off[A], atom_shell_off[A + 1]):
                K = atom_shells[s_]
                if ff[b, K]:
                    nK = (aux_l[K] + 1) * (aux_l[K] + 1)
                    for lm in range(nK):
                        L_K[K, lm] += V[lm]
    return L_K


@njit(parallel=True, cache=True, fastmath=True, error_model="numpy", nogil=True, boundscheck=False)
def _atoms_to_branches(shell_mom, branch_center, atom_coords, atom_shell_off, atom_shells, atom_lmax,
                       aux_l, ff, any_ff, lmax, n_big, l_aux_max, pair_off_tab, ent_LM, ent_coef, sign_big):
    """
    J direction: local expansion ``branch_local[b, jk]`` (``j <= lmax``) of the far-field
    fitted density (exact point multipoles ``shell_mom[K]`` of the auxiliary shells) at
    every branch centre: ``W[lm, LM] = sum_A S_A[lm] I_LM(B - A)`` over the far-field atoms
    (``S_A`` = moments of the far-field shells of ``A``), then
    ``L_b[jk] = (-1)^j sum_lm sum_M A^{l+j,M}_{lm,jk} W[lm, (l+j)M]``.
    """
    natoms = atom_coords.shape[0]
    n_branches = branch_center.shape[0]
    n_small = (l_aux_max + 1) * (l_aux_max + 1)
    n_irr = (lmax + l_aux_max + 1) * (lmax + l_aux_max + 1)
    branch_local = np.zeros((n_branches, n_big))
    for b in prange(n_branches):
        IR = np.zeros(n_irr)
        W = np.zeros((n_small, n_irr))
        S = np.zeros(n_small)
        for A in range(natoms):
            if not any_ff[b, A]:
                continue
            lA = atom_lmax[A]
            nsA = (lA + 1) * (lA + 1)
            nirrA = (lmax + lA + 1) * (lmax + lA + 1)
            for lm in range(nsA):
                S[lm] = 0.0
            for s_ in range(atom_shell_off[A], atom_shell_off[A + 1]):
                K = atom_shells[s_]
                if ff[b, K]:
                    nK = (aux_l[K] + 1) * (aux_l[K] + 1)
                    for lm in range(nK):
                        S[lm] += shell_mom[K, lm]
            mp.irregular_harmonics(branch_center[b, 0] - atom_coords[A, 0], branch_center[b, 1] - atom_coords[A, 1],
                                   branch_center[b, 2] - atom_coords[A, 2], lmax + lA, IR)
            for lm in range(nsA):
                s = S[lm]
                if s == 0.0:
                    continue
                for LM in range(nirrA):
                    W[lm, LM] += s * IR[LM]
        for lm in range(n_small):
            for jk in range(n_big):
                pidx = lm * n_big + jk
                acc = 0.0
                for e in range(pair_off_tab[pidx], pair_off_tab[pidx + 1]):
                    acc += ent_coef[e] * W[lm, ent_LM[e]]
                branch_local[b, jk] += sign_big[jk] * acc
    return branch_local


@njit(cache=True, fastmath=True, error_model="numpy", nogil=True, boundscheck=False)
def _shell_moments(coeff, aux_mom, aux_off, aux_nbf, aux_l, n_small):
    """``shell_mom[K, lm] = sum_{P in K} c_P M^P_lm``."""
    nsh = aux_off.shape[0]
    out = np.zeros((nsh, n_small))
    for K in range(nsh):
        k0 = aux_off[K]
        nK = (aux_l[K] + 1) * (aux_l[K] + 1)
        for c in range(aux_nbf[K]):
            cP = coeff[k0 + c]
            for lm in range(nK):
                out[K, lm] += cP * aux_mom[k0 + c, lm]
    return out


@njit(cache=True, fastmath=True, error_model="numpy", nogil=True, boundscheck=False)
def _gamma_far_field(L_K, aux_mom, aux_off, aux_nbf, aux_l, naux):
    """``gamma_P (far field) = sum_lm M^P_lm L_K[lm]``."""
    out = np.zeros(naux)
    for K in range(aux_off.shape[0]):
        k0 = aux_off[K]
        nK = (aux_l[K] + 1) * (aux_l[K] + 1)
        for c in range(aux_nbf[K]):
            s = 0.0
            for lm in range(nK):
                s += aux_mom[k0 + c, lm] * L_K[K, lm]
            out[k0 + c] = s
    return out


# ----------------------------------------------------------------------------
# Python drivers
# ----------------------------------------------------------------------------
def build_plan(basis, auxbasis, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold, strict_schwarz,
               sao=False, max_memory_gb=None, ncores=None, options=None):
    """
    Screen, classify (near/far field), evaluate the near-field row blocks (within the
    memory budget) and the far-field moments.

    Parameters
    ----------
    basis, auxbasis : Basis
    sqrt_ints4c2e_diag : (nao, nao) ndarray   sqrt(|(ij|ij)|)
    sqrt_diag_ints2c2e : (naux,) ndarray      sqrt(|(P|P)|) of the (SAO: projected) metric
    threshold : float                          Schwarz threshold (block level)
    strict_schwarz : bool                      pair cut-off ``(ij|ij) < STRICT_PAIR_CUTOFF``
    sao : bool                                 project auxiliary shells with l >= 2
    max_memory_gb : float or None              budget for the cached near-field blocks; None = all
    ncores : int or None                       threads (defaults to numba.get_num_threads())
    options : dict or None                     overrides of :data:`DEFAULT_OPTIONS`

    Returns
    -------
    DFAlgo12Plan
    """
    t_start = timer()
    timings = {}
    opts = dict(DEFAULT_OPTIONS)
    if options:
        unknown = set(options) - set(DEFAULT_OPTIONS)
        if unknown:
            raise ValueError('unknown multipole options %s; valid keys: %s' % (sorted(unknown), sorted(DEFAULT_OPTIONS)))
        opts.update(options)
    plan = DFAlgo12Plan()
    nthreads = int(numba.get_num_threads() if ncores is None else max(1, ncores))
    plan.nthreads = nthreads
    plan.threshold = float(threshold)
    plan.strict_schwarz = bool(strict_schwarz)
    plan.sao = bool(sao)
    plan.nao = basis.bfs_nao
    plan.naux = auxbasis.bfs_nao
    plan.precision = float(opts['precision'])
    plan.lmax = int(opts['lmax'])
    plan.box_size = float(opts['box_size'])
    plan.separation = float(opts['separation'])
    plan.class_factor = float(opts['class_factor'])
    plan.break_even = float(opts['break_even'])
    plan.low_memory = bool(opts['low_memory'])
    if (plan.lmax < 0 or plan.precision <= 0 or plan.box_size <= 0 or plan.separation < 1.0
            or plan.class_factor <= 1.0 or plan.break_even < 0.0):  # low_memory is a plain flag
        raise ValueError('multipole options: precision > 0, lmax >= 0, box_size > 0, separation >= 1, '
                         'class_factor > 1 and break_even >= 0 are required')

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
    npairs = plan.pair_I.shape[0]
    maxpp = int(plan.bfs_nprim.max()) ** 2
    plan.maxpp = maxpp
    tcap = float(erfcinv(1e-16))   # extents never exceed those of a 1e-16 point-charge error
    maxA = int(plan.shell_nbf.max())
    maxC = int(plan.aux_nbf.max())
    plan.dims = np.array([maxA, maxC, lmaxA, lmaxC, maxpp], dtype=np.int64)
    timings['setup'] = timer() - t_start

    # --- alive pairs (Schwarz, strict cut-off, product pre-screen) ------------------
    t0 = timer()
    plan.Q_pair, alive, plan.pair_nrows = _pair_alive(
        plan.pair_I, plan.pair_J, plan.shell_off, plan.shell_nbf, plan.bfs_nprim, plan.bfs_expnts, plan.bfs_coords,
        plan.sqrt_ints4c2e_diag, plan.Q_aux, plan.threshold, plan.strict_schwarz)
    sig = np.nonzero(alive)[0].astype(np.int64)
    plan.n_pairs_total = int(npairs)
    plan.n_pairs_significant = int(sig.size)

    # --- geometry of the distributions and branches ---------------------------------
    (ngroups, npp_arr, grp_center, grp_pmin, grp_qmax, grp_extent, pp_group) = _pair_geometry(
        plan.pair_I, plan.pair_J, plan.shell_off, plan.shell_nbf, plan.shell_l, plan.bfs_nprim, plan.bfs_expnts,
        plan.bfs_coords, plan.bf_coef, 1.0 / plan.precision, tcap, maxpp)
    ngroups = np.where(alive, ngroups, 0).astype(np.int64)
    npp_arr = np.where(alive, npp_arr, 0).astype(np.int64)
    # flatten the groups of the alive pairs
    gp = np.repeat(sig, ngroups[sig])                                  # pair of every group
    gg = np.concatenate([np.arange(n) for n in ngroups[sig]]) if sig.size else np.zeros(0, dtype=np.int64)
    gcen = grp_center[gp, gg]                                          # (n_groups, 3)
    gext = grp_extent[gp, gg]
    n_groups = int(gp.shape[0])
    h = plan.box_size
    if n_groups:
        origin = gcen.min(axis=0) - 0.5 * h
        box_ijk = np.floor((gcen - origin) / h).astype(np.int64)
        nbox_dim = box_ijk.max(axis=0) + 1
        box_lin = (box_ijk[:, 0] * nbox_dim[1] + box_ijk[:, 1]) * nbox_dim[2] + box_ijk[:, 2]
    else:
        box_lin = np.zeros(0, dtype=np.int64)
    # extent classes: geometric bins (class_factor) of the Gaussian extent relative to the box size
    ext_bin = np.zeros(n_groups, dtype=np.int64)
    pos = gext > 0.25 * h
    ext_bin[pos] = np.ceil(np.log(gext[pos] / (0.25 * h)) / np.log(plan.class_factor)).astype(np.int64)
    ext_bin = np.clip(ext_bin, 0, 63)
    keys = box_lin * 64 + ext_bin
    uniq, g_branch = np.unique(keys, return_inverse=True)
    g_branch = np.asarray(g_branch, dtype=np.int64).reshape(-1)
    n_branches = int(uniq.shape[0])
    branch_center = np.zeros((n_branches, 3))
    branch_radius = np.zeros(n_branches)
    branch_pmin = np.full(n_branches, np.inf)
    branch_qmax = np.zeros(n_branches)
    if n_groups:
        order = np.argsort(g_branch, kind='stable')
        bounds = np.searchsorted(g_branch[order], np.arange(n_branches + 1))
        gpm = grp_pmin[gp, gg]
        gqm = grp_qmax[gp, gg]
        for b in range(n_branches):
            members = order[bounds[b]:bounds[b + 1]]
            pts = gcen[members]
            lo = pts.min(axis=0)
            hi = pts.max(axis=0)
            branch_center[b] = 0.5 * (lo + hi)
            branch_radius[b] = np.sqrt(((pts - branch_center[b]) ** 2).sum(axis=1)).max()
            branch_pmin[b] = gpm[members].min()
            branch_qmax[b] = gqm[members].max()
    grp_branch = np.zeros((npairs, maxpp), dtype=np.int64)
    grp_branch[gp, gg] = g_branch
    plan.n_boxes = int(np.unique(box_lin).shape[0])
    plan.n_branches = n_branches
    plan.n_groups = n_groups
    plan.branch_center = branch_center
    plan.branch_radius = branch_radius
    plan.branch_pmin = branch_pmin
    plan.branch_qmax = branch_qmax
    plan.grp_branch = grp_branch
    plan.ngroups = ngroups
    plan.npp = npp_arr
    plan.grp_center = grp_center
    plan.pp_group = pp_group
    timings['geometry'] = timer() - t0

    # --- auxiliary side ---------------------------------------------------------------
    t0 = timer()
    aux_atom, atom_coords = _aux_atoms(plan.aux_coords, plan.aux_off)
    natoms = atom_coords.shape[0]
    aux_qmin, aux_Qmax = _aux_charges(apacked, plan.aux_coef, plan.aux_off, plan.aux_nbf, plan.aux_l)
    order = np.argsort(aux_atom, kind='stable')
    atom_shells = np.ascontiguousarray(order, dtype=np.int64)
    atom_shell_off = np.zeros(natoms + 1, dtype=np.int64)
    atom_shell_off[1:] = np.cumsum(np.bincount(aux_atom, minlength=natoms))
    atom_lmax = np.zeros(natoms, dtype=np.int64)
    for A in range(natoms):
        atom_lmax[A] = plan.aux_l[atom_shells[atom_shell_off[A]:atom_shell_off[A + 1]]].max()
    plan.aux_atom = aux_atom
    plan.atom_coords = atom_coords
    plan.atom_shells = atom_shells
    plan.atom_shell_off = atom_shell_off
    plan.atom_lmax = atom_lmax
    plan.aux_qmin = aux_qmin
    plan.aux_Qmax = aux_Qmax
    plan.l_aux_max = lmaxC
    plan.aux_mom = _aux_moments(auxbasis, apacked, plan.aux_coef, plan.aux_off, plan.aux_nbf, plan.aux_l,
                                sao, plan.projectors)

    # --- far-field classification: branch x auxiliary shell ----------------------------
    dist = np.linalg.norm(branch_center[:, None, :] - atom_coords[None, :, :], axis=2)   # (nb, natoms)
    R = dist[:, aux_atom]                                                                 # (nb, nsh_aux)
    with np.errstate(divide='ignore', invalid='ignore'):
        x = plan.precision / np.maximum(branch_qmax[:, None] * aux_Qmax[None, :], 1e-300)
        t = np.where(x < 1.0, erfcinv(np.clip(x, 1e-300, 1.0)), 0.0)
        t = np.minimum(t, tcap)
        width = np.sqrt(1.0 / branch_pmin[:, None] + 1.0 / aux_qmin[None, :])
    plan.R_overlap = branch_radius[:, None] + t * width
    ff = (R >= plan.R_overlap) & (R >= plan.separation * branch_radius[:, None])
    plan.ff = np.ascontiguousarray(ff)
    plan.any_ff = np.ascontiguousarray(np.zeros((n_branches, natoms), dtype=np.bool_))
    for A in range(natoms):
        plan.any_ff[:, A] = ff[:, atom_shells[atom_shell_off[A]:atom_shell_off[A + 1]]].any(axis=1)
    timings['classification'] = timer() - t0

    # --- profitability: which groups actually use the multipole path -------------------
    t0 = timer()
    plan.lpair_max = int(2 * lmaxA)
    plan.n_big = (plan.lmax + 1) ** 2
    plan.table = _tables(max(plan.lpair_max, lmaxC), plan.lmax)
    plan.mono = _polynomials(plan.lpair_max)
    plan.sign_big = np.array([(-1.0) ** j for j in range(plan.lmax + 1) for _ in range(2 * j + 1)], dtype=np.float64)
    grp_ff = _group_far_field(sig, plan.Q_pair, plan.Q_aux, plan.aux_off, plan.aux_nbf, plan.threshold,
                              ngroups, grp_branch, plan.ff, plan.n_big, plan.break_even, maxpp)
    # Groups that do not use multipoles are pointed at an all-near-field row of the mask, so
    # every kernel keeps a single branch-indexed test and needs no extra argument.
    plan.ff_eff = np.ascontiguousarray(np.vstack([plan.ff, np.zeros((1, plan.ff.shape[1]), dtype=np.bool_)]))
    plan.grp_branch_eff = np.ascontiguousarray(np.where(grp_ff, grp_branch, n_branches).astype(np.int64))
    plan.grp_ff = grp_ff
    # one entry = one (shell pair, branch) block of branch-centred row moments
    if n_groups:
        keep = grp_ff[gp, gg]
        uniq_e = np.unique(gp[keep].astype(np.int64) * (n_branches + 1) + g_branch[keep])
    else:
        uniq_e = np.zeros(0, dtype=np.int64)
    entry_pair = (uniq_e // (n_branches + 1)).astype(np.int64)
    plan.entry_branch = np.ascontiguousarray((uniq_e % (n_branches + 1)).astype(np.int64))
    plan.entry_pair = entry_pair
    plan.entry_off = np.zeros(npairs + 1, dtype=np.int64)
    plan.entry_off[1:] = np.cumsum(np.bincount(entry_pair, minlength=npairs))
    entry_sizes = plan.pair_nrows[entry_pair] * plan.n_big
    plan.entry_moff = (np.concatenate(([0], np.cumsum(entry_sizes)[:-1])).astype(np.int64)
                       if entry_pair.size else np.zeros(0, dtype=np.int64))
    plan.n_entries = int(entry_pair.size)
    plan.max_nrows_all = int(plan.pair_nrows[sig].max()) if sig.size else 1
    timings['profitability'] = timer() - t0

    # --- near-field columns and cost model -------------------------------------------
    t0 = timer()
    (plan.pair_ncols, ncols_all, cost, cost_all, max_roots) = _pair_columns(
        sig, plan.pair_I, plan.pair_J, plan.shell_off, plan.shell_nbf, plan.shell_l, plan.bfs_nprim,
        plan.bfs_expnts, plan.bfs_coords, plan.Q_pair, plan.Q_aux, plan.aux_off, plan.aux_nbf, plan.aux_l,
        plan.aux_nprim, plan.threshold, npp_arr, pp_group, plan.grp_branch_eff, plan.ff_eff)
    if sig.size and int(max_roots[sig].max()) > MAX_RYS_ROOTS:
        raise ValueError('DF_algo=12 supports at most %d Rys roots (total angular momentum <= %d).'
                         % (MAX_RYS_ROOTS, 2 * MAX_RYS_ROOTS - 1))
    elems_nf = plan.pair_nrows[sig] * plan.pair_ncols[sig]
    plan.n_elements_significant = int((plan.pair_nrows[sig] * ncols_all[sig]).sum())
    plan.n_elements_nf = int(elems_nf.sum())
    plan.cost_nf = float(cost[sig].sum())
    plan.cost_all = float(cost_all[sig].sum())
    plan.col_off = np.zeros(npairs + 1, dtype=np.int64)
    plan.col_off[1:] = np.cumsum(plan.pair_ncols)
    plan.cols = np.zeros(int(plan.col_off[-1]), dtype=np.int32)
    if sig.size:
        _near_field_column_map(sig, plan.Q_pair, plan.Q_aux, plan.threshold, plan.aux_off, plan.aux_nbf,
                               ngroups, plan.grp_branch_eff, plan.ff_eff, plan.col_off, plan.cols)
    timings['columns'] = timer() - t0

    # --- far-field moments --------------------------------------------------------------
    t0 = timer()
    lpair = plan.shell_l[plan.pair_I] + plan.shell_l[plan.pair_J]
    nlm_pair = (lpair + 1) ** 2
    mom_sizes = np.zeros(npairs, dtype=np.int64)
    mom_sizes[sig] = ngroups[sig] * plan.pair_nrows[sig] * nlm_pair[sig]
    plan.mom_off = np.zeros(npairs, dtype=np.int64)
    if sig.size:
        plan.mom_off[sig] = np.concatenate(([0], np.cumsum(mom_sizes[sig])[:-1])).astype(np.int64)
    plan.moments = np.zeros(int(mom_sizes.sum()), dtype=np.float64)
    if sig.size:
        mom_cost = (ngroups[sig] * plan.pair_nrows[sig] * nlm_pair[sig]).astype(np.float64)
        bin_off, bin_items = _lpt_bins(sig, mom_cost, nthreads)
        _build_moments(bin_off, bin_items, plan.pair_I, plan.pair_J, plan.pair_nrows, plan.mom_off, ngroups,
                       pp_group, grp_center, plan.bfs_coords, plan.bfs_lmn, plan.bfs_nprim, plan.bfs_expnts,
                       plan.bf_coef, plan.shell_off, plan.shell_nbf, plan.shell_l,
                       plan.mono[0], plan.mono[1], plan.mono[2], plan.moments, maxA, plan.lpair_max)
    timings['moments'] = timer() - t0

    # --- branch-centred row moments (translated once, used in every iteration) ----------
    t0 = timer()
    plan.Mtil = np.zeros(0 if plan.low_memory else int(entry_sizes.sum()), dtype=np.float64)
    if entry_pair.size and not plan.low_memory:
        bin_off, bin_items = _lpt_bins(np.arange(entry_pair.size, dtype=np.int64),
                                       entry_sizes.astype(np.float64), nthreads)
        _build_translated_moments(bin_off, bin_items, entry_pair, plan.entry_branch, plan.entry_moff,
                                  plan.pair_I, plan.pair_J, plan.pair_nrows, plan.mom_off, ngroups,
                                  grp_branch, grp_ff, grp_center, plan.branch_center, plan.shell_l,
                                  plan.moments, plan.Mtil, plan.lmax, plan.n_big,
                                  plan.table[0], plan.table[1], plan.table[2])
    timings['translated_moments'] = timer() - t0

    # --- caching within the budget (near-field blocks) ----------------------------------
    t0 = timer()
    if max_memory_gb is None:
        budget = int(elems_nf.sum())
    else:
        budget = int(max(0.0, float(max_memory_gb)) * 1e9 // 8)
    ratio = cost[sig] / np.maximum(elems_nf, 1)
    order_ratio = sig[np.argsort(-ratio, kind='stable')]
    cached = np.zeros(npairs, dtype=np.bool_)
    used = 0
    for p in order_ratio:
        n = int(plan.pair_nrows[p] * plan.pair_ncols[p])
        if used + n <= budget:
            cached[p] = True
            used += n
    plan.pair_offset = np.full(npairs, -1, dtype=np.int64)
    cached_idx = np.nonzero(cached)[0]
    sizes = plan.pair_nrows[cached_idx] * plan.pair_ncols[cached_idx]
    plan.pair_offset[cached_idx] = (np.concatenate(([0], np.cumsum(sizes)[:-1])).astype(np.int64)
                                    if cached_idx.size else np.zeros(0, dtype=np.int64))
    plan.n_pairs_cached = int(cached_idx.size)
    plan.n_elements_cached = int(sizes.sum())
    n_entries_pair = plan.entry_off[1:] - plan.entry_off[:-1]
    if plan.low_memory:
        n_ff_groups = np.array([int((plan.grp_branch_eff[p, :ngroups[p]] < n_branches).sum()) for p in sig],
                               dtype=np.float64)
        ff_cost = n_ff_groups * (2.0 * plan.pair_nrows[sig] * nlm_pair[sig] + 2.0 * nlm_pair[sig] * plan.n_big)
    else:
        ff_cost = 4.0 * n_entries_pair[sig] * plan.pair_nrows[sig] * plan.n_big
    uncached_idx = sig[~cached[sig]]
    plan.work_iter = sig
    plan.iter_cost = np.where(cached[sig], 2.0 * elems_nf, cost[sig]) + ff_cost
    plan.work_build = cached_idx.astype(np.int64)
    plan.build_cost = cost[cached_idx]
    plan._iter_bins = {}
    plan.max_nrows = int(plan.pair_nrows[uncached_idx].max()) if uncached_idx.size else 1

    plan.values = np.zeros(plan.n_elements_cached, dtype=np.float64)
    if plan.work_build.size:
        bin_off, bin_items = _lpt_bins(plan.work_build, plan.build_cost, nthreads)
        _build_cached(bin_off, bin_items, plan.pair_I, plan.pair_J, plan.pair_offset,
                      plan.pair_nrows, plan.pair_ncols, plan.values, *_kernel_args(plan), plan.pp_group,
                      plan.grp_branch_eff, plan.ff_eff, plan.dims)
    timings['near_field'] = timer() - t0
    timings['total'] = timer() - t_start
    plan.timings = timings
    return plan


def _iteration_bins(plan):
    nthreads = int(numba.get_num_threads())
    if nthreads not in plan._iter_bins:
        plan._iter_bins[nthreads] = _lpt_bins(plan.work_iter, plan.iter_cost, nthreads)
    return plan._iter_bins[nthreads]


def _kernel_args(plan):
    return (plan.bfs_coords, plan.bfs_lmn, plan.bfs_nprim, plan.bfs_expnts, plan.bf_coef,
            plan.shell_off, plan.shell_nbf, plan.shell_l,
            plan.aux_coords, plan.aux_lmn, plan.aux_nprim, plan.aux_expnts, plan.aux_coef,
            plan.aux_off, plan.aux_nbf, plan.aux_l,
            plan.Q_pair, plan.Q_aux, plan.threshold, plan.sao, plan.projectors)


def gamma_from_plan(plan, dmat, timings=None, ff_mode=0):
    """``gamma_P = sum_ij D_ij (ij|P)`` (near field from the blocks, far field through the multipole expansions)."""
    dmat = np.ascontiguousarray(dmat, dtype=np.float64)
    bin_off, bin_items = _iteration_bins(plan)
    pair_off_tab, ent_LM, ent_coef = plan.table
    t0 = timer()
    gamma, branch_mom = _gamma_pass(
        bin_off, bin_items, plan.pair_I, plan.pair_J, plan.pair_offset, plan.pair_nrows, plan.pair_ncols,
        plan.values, dmat, plan.sqrt_ints4c2e_diag, plan.strict_schwarz, plan.naux,
        *_kernel_args(plan), plan.pp_group, plan.grp_branch_eff, plan.ff_eff, plan.dims, plan.max_nrows,
        plan.ngroups, plan.col_off, plan.cols, plan.entry_off, plan.entry_branch, plan.entry_moff, plan.Mtil,
        plan.n_branches, plan.n_big, plan.max_nrows_all, not plan.low_memory, plan.mom_off, plan.moments,
        plan.grp_center, plan.branch_center, plan.lmax, pair_off_tab, ent_LM, ent_coef, plan.lpair_max, ff_mode)
    t1 = timer()
    L_K = _branches_to_atoms(branch_mom, plan.branch_center, plan.atom_coords, plan.atom_shell_off, plan.atom_shells,
                             plan.atom_lmax, plan.aux_l, plan.ff, plan.any_ff, plan.lmax, plan.n_big, plan.l_aux_max,
                             pair_off_tab, ent_LM, ent_coef, plan.sign_big)
    gamma += _gamma_far_field(L_K, plan.aux_mom, plan.aux_off, plan.aux_nbf, plan.aux_l, plan.naux)
    if timings is not None:
        timings['gamma_pairs'] = timings.get('gamma_pairs', 0.0) + (t1 - t0)
        timings['gamma_coupling'] = timings.get('gamma_coupling', 0.0) + (timer() - t1)
    return gamma


def J_from_plan(plan, coeff, timings=None, ff_mode=0):
    """Full symmetric Coulomb matrix ``J_ij = sum_P (ij|P) c_P`` (near field + far field)."""
    coeff = np.ascontiguousarray(coeff, dtype=np.float64)
    bin_off, bin_items = _iteration_bins(plan)
    pair_off_tab, ent_LM, ent_coef = plan.table
    n_small = (plan.l_aux_max + 1) ** 2
    t0 = timer()
    shell_mom = _shell_moments(coeff, plan.aux_mom, plan.aux_off, plan.aux_nbf, plan.aux_l, n_small)
    branch_local = _atoms_to_branches(shell_mom, plan.branch_center, plan.atom_coords, plan.atom_shell_off,
                                      plan.atom_shells, plan.atom_lmax, plan.aux_l, plan.ff, plan.any_ff,
                                      plan.lmax, plan.n_big, plan.l_aux_max, pair_off_tab, ent_LM, ent_coef,
                                      plan.sign_big)
    t1 = timer()
    J = _J_pass(
        bin_off, bin_items, plan.pair_I, plan.pair_J, plan.pair_offset, plan.pair_nrows, plan.pair_ncols,
        plan.values, coeff, plan.sqrt_ints4c2e_diag, plan.strict_schwarz, plan.nao,
        *_kernel_args(plan), plan.pp_group, plan.grp_branch_eff, plan.ff_eff, plan.dims, plan.max_nrows,
        plan.ngroups, plan.col_off, plan.cols, plan.entry_off, plan.entry_branch, plan.entry_moff, plan.Mtil,
        branch_local, plan.n_branches, plan.n_big, plan.max_nrows_all, not plan.low_memory, plan.mom_off,
        plan.moments, plan.grp_center, plan.branch_center, plan.lmax, pair_off_tab, ent_LM, ent_coef,
        plan.lpair_max, ff_mode)
    if timings is not None:
        timings['J_coupling'] = timings.get('J_coupling', 0.0) + (t1 - t0)
        timings['J_pairs'] = timings.get('J_pairs', 0.0) + (timer() - t1)
    return J


# ----------------------------------------------------------------------------
# Validation helpers
# ----------------------------------------------------------------------------
def _block_kernel(plan, p, ff):
    """Rows of pair ``p`` for every Schwarz-significant auxiliary shell with the given far-field mask."""
    I, J = int(plan.pair_I[p]), int(plan.pair_J[p])
    nr = int(plan.pair_nrows[p])
    maxA, maxC, lmaxA, lmaxC, maxpp = (int(v) for v in plan.dims)
    rows = np.zeros((nr, plan.naux))
    scratch = (np.zeros(MAX_RYS_ROOTS), np.zeros(MAX_RYS_ROOTS),
               np.zeros((2 * lmaxA + 1, lmaxC + 1)), np.zeros((2 * lmaxA + 1, lmaxC + 1)), np.zeros((2 * lmaxA + 1, lmaxC + 1)),
               np.zeros((lmaxA + 1, lmaxA + 1, lmaxC + 1)), np.zeros((lmaxA + 1, lmaxA + 1, lmaxC + 1)),
               np.zeros((lmaxA + 1, lmaxA + 1, lmaxC + 1)),
               np.zeros((maxA, maxA, maxC)), np.zeros((maxA, maxA, maxC)), np.zeros((maxA, maxA, maxC)))
    ppf = np.zeros((6, maxpp))
    ppi = np.zeros((3, maxpp), dtype=np.int64)
    comp = np.zeros((9, max(maxA, maxC)), dtype=np.int64)
    nc = _compute_pair_rows(I, J, rows, *_kernel_args(plan)[:16], plan.Q_pair[p], plan.Q_aux, plan.threshold,
                            plan.sao, plan.projectors, plan.pp_group[p], plan.grp_branch_eff[p], ff, *scratch,
                            ppf[0], ppf[1], ppf[2], ppf[3], ppf[4], ppf[5], ppi[0], ppi[1], ppi[2],
                            comp[0], comp[1], comp[2], comp[3], comp[4], comp[5], comp[6], comp[7], comp[8])
    return rows[:, :nc]


def _columns_of(plan, p, ff):
    """(K, first column) of the shells present in the block returned by :func:`_block_kernel`."""
    cols = {}
    c = 0
    for K in range(plan.aux_off.shape[0]):
        if plan.Q_pair[p] * plan.Q_aux[K] <= plan.threshold:
            continue
        if all(ff[plan.grp_branch_eff[p, g], K] for g in range(int(plan.ngroups[p]))):
            continue
        cols[K] = c
        c += int(plan.aux_nbf[K])
    return cols


def exact_block(plan, p, K):
    """Exact Rys ``(ij|P)`` block of shell pair ``p`` and auxiliary shell ``K``, shape ``(nrows, nC)``."""
    ff = np.zeros_like(plan.ff_eff)
    rows = _block_kernel(plan, p, ff)
    c = _columns_of(plan, p, ff)[K]
    return rows[:, c:c + int(plan.aux_nbf[K])]


def approx_block(plan, p, K):
    """
    ``(ij|P)`` block of shell pair ``p`` and auxiliary shell ``K`` as algorithm 12 uses it:
    near-field primitive pairs from the Rys kernel plus far-field primitive pairs from the
    multipole expansions (group moments -> branch centre -> atom of ``K``, truncated at ``lmax``).
    """
    pair_off_tab, ent_LM, ent_coef = plan.table
    I, J = int(plan.pair_I[p]), int(plan.pair_J[p])
    nr = int(plan.pair_nrows[p])
    lp = int(plan.shell_l[I] + plan.shell_l[J])
    nlm = (lp + 1) ** 2
    k0, nC, lC = int(plan.aux_off[K]), int(plan.aux_nbf[K]), int(plan.aux_l[K])
    nK = (lC + 1) ** 2
    A = plan.atom_coords[plan.aux_atom[K]]
    lA = int(plan.atom_lmax[plan.aux_atom[K]])
    n_big = plan.n_big
    out = np.zeros((nr, nC))
    cols = _columns_of(plan, p, plan.ff_eff)
    if K in cols:
        rows = _block_kernel(plan, p, plan.ff_eff)
        out += rows[:, cols[K]:cols[K] + nC]
    Rd = np.zeros(n_big)
    for g in range(int(plan.ngroups[p])):
        b = int(plan.grp_branch_eff[p, g])
        if b >= plan.n_branches or not plan.ff[b, K]:
            continue
        B = plan.branch_center[b]
        IR = np.zeros((plan.lmax + lA + 1) ** 2)
        mp.irregular_harmonics(*(B - A), plan.lmax + lA, IR)
        T = np.zeros(((lA + 1) ** 2, n_big))
        mp.interaction_tensor(IR, lA, plan.lmax, n_big, pair_off_tab, ent_LM, ent_coef, T)
        mp.regular_harmonics(*(plan.grp_center[p, g] - B), plan.lmax, Rd)
        for r in range(nr):
            o = plan.mom_off[p] + (g * nr + r) * nlm
            Mb = np.zeros(n_big)
            mp.translate_moments(plan.moments[o:o + nlm], lp, Rd, plan.lmax, n_big, pair_off_tab, ent_LM, ent_coef, Mb)
            L_A = T[:nK] @ Mb
            out[r] += plan.aux_mom[k0:k0 + nC, :nK] @ L_A
    return out
