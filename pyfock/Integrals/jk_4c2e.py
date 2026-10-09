"""
Coulomb (J) and exchange (K) matrices from four-center two-electron integrals evaluated over
shell quartets, for calculations without density fitting.

The integrals come from one of two schemes, chosen when the plan is built (:func:`build_plan`):

* ``'rys'``: Rys quadrature;
* ``'os'``: the Obara-Saika vertical recurrences.

Everything else is shared: the shell pairs and their screening, the horizontal transfers, the
storage of the integrals and their contraction with the density matrix. A plan is used in one of
three ways:

* **direct** (:func:`direct_jk`): the integrals are evaluated in every SCF iteration and
  contracted on the fly with the density matrix (or with its change since the previous
  iteration);
* **stored, screened** (:func:`store_integrals` with ``dense=False``): the shell-quartet
  blocks that pass the Schwarz test are evaluated once and kept in memory;
* **stored, complete** (``dense=True``): every unique shell-quartet block is kept.

Shell pairs and quartets
------------------------
Shell pairs are formed once, the shell with the higher angular momentum first, together
with the primitive pairs that survive the Gaussian-product cut-off, and they are sorted by
their Schwarz factor ``Q = sqrt(max |(ab|ab)|)`` in decreasing order. A shell quartet
``(XY|ZW)`` is a bra pair ``i`` and a ket pair ``j >= i`` (8-fold permutational symmetry).
Both schemes accumulate the contracted integrals ``(e0|f0)`` (all Cartesian ``e`` with
``l_X <= |e| <= l_X + l_Y`` on ``X``, all ``f`` with ``l_Z <= |f| <= l_Z + l_W`` on ``Z``) over
the primitive quartets; the horizontal transfers to ``Y`` and ``W`` are then applied once per
shell quartet in closed (binomial) form. The primitive quartets are processed in batches of
*lanes*, and the recurrences run as loops over lanes, which the compiler vectorizes; the
contraction coefficients enter the starting values of the recurrences.

Within a shell pair the primitive pairs are sorted by their own Schwarz factor, and a
primitive quartet is skipped when the product of the two factors (times the density factor
of the screening below) is below ``PRIM_FACTOR`` times the threshold.

Rys quadrature
--------------
For every primitive quartet the Rys roots and weights are computed once and the recurrence
coefficients of every (primitive quartet, root) lane are stored; batches of up to ``MAXK``
lanes run the two-dimensional recurrences per Cartesian direction (built on ``X`` for the bra
and on ``Z`` for the ket), and ``(e0|f0)`` is the sum over lanes of products of the three
directions.

Obara-Saika
-----------
One lane per primitive quartet. The Boys function values ``F_m(T)``, ``m <= L`` (``L`` the total
angular momentum of the quartet), come from a table on a grid of spacing ``BOYS_DT``: a Taylor
expansion around the nearest grid point for ``F_L`` and downward recursion for the lower orders
(the asymptotic form beyond ``BOYS_TMAX``). The vertical recurrences then build ``[e0|00]^(m)`` on
``X`` and ``[e0|f0]^(m)`` on ``Z``, including the term that couples bra and ket, and
``(e0|f0) = [e0|f0]^(0)``; at every ket level only the orders ``m`` and bra triples that the final
integrals depend on are built, and every triple is built along its smallest nonzero exponent. The
number of lanes per batch is limited by the size of the intermediate array (``OS_WORK`` doubles
per thread).

Contraction
-----------
With the degeneracy factor ``s`` (a factor 1/2 each for ``X == Y``, ``Z == W`` and identical
bra and ket pairs) every element ``v`` of a block adds::

    H_xy += s v D_zw     H_zw += s v D_xy
    G_xz += s v D_yw     G_xw += s v D_yz     G_yz += s v D_xw     G_yw += s v D_xz

and ``J = 2 (H + H^T)``, ``K = G + G^T``.

Screening
---------
A quartet is skipped when ``Q_XY Q_ZW max(4 D_XY, 4 D_ZW, D_XZ, D_XW, D_YZ, D_YW)`` is below
the threshold (``D_AB`` is the largest ``|D|`` element of the shell block; the exchange terms
only when K is built). As the pairs are sorted by ``Q``, the ket loop stops at the first pair
with ``Q_XY Q_ZW 4 max|D|`` below the threshold. The stored modes apply the same test in
every iteration, which skips the blocks that a small density change does not need.

Storage
-------
The stored modes keep, for every bra pair ``i``, the blocks of its ket pairs ``i <= j < jend[i]``
contiguously (``jend[i] = npairs`` for the complete store, the first pair below the Schwarz
threshold otherwise, as the pairs are sorted by ``Q``), pre-multiplied by their degeneracy
factor. No index arrays are needed: 8 bytes per stored integral.

Parallelism
-----------
Bra pairs are assigned to one bin per thread by a longest-processing-time-first heuristic on
their estimated cost (the workqueue threading layer schedules ``prange`` statically). Every
bin accumulates into its own ``H`` and ``G``, so the kernels need neither thread ids nor
atomics, and all of them are cached on disk by Numba.
"""

import math

import numba
import numpy as np
from numba import njit, prange

from .df_algo10_helpers import pack_basis_arrays
from .df_algo11_helpers import _lpt_bins
from .rys_helpers import Roots

__all__ = ['Plan4c2e', 'SCHEMES', 'build_plan', 'direct_jk', 'store_size_gb', 'store_integrals', 'stored_jk']

TWO_PI_52 = 2.0 * np.pi ** 2.5
SQRT_PI_HALF = 0.5 * math.sqrt(math.pi)
# Primitive pairs with mu |AB|^2 above this are dropped (exp(-40) = 4e-18).
EXP_CUTOFF = 40.0
# Largest number of Rys roots provided by rys_helpers.Roots.
MAX_ROOTS = 10
# Lanes (primitive quartets x roots) evaluated together by the Rys recurrences.
MAXK = 256
# Lanes (primitive quartets) evaluated together by the Obara-Saika recurrences, and the size in
# doubles of their intermediate array per thread (at least one lane always fits).
MAXK_OS = 128
OS_WORK = 1 << 19
# Boys function table of the Obara-Saika scheme: F_m on the grid T = 0, BOYS_DT, ..., BOYS_TMAX, evaluated by a
# Taylor expansion of BOYS_TAYLOR + 1 terms around the nearest grid point (asymptotic form beyond BOYS_TMAX).
BOYS_DT = 0.05
BOYS_TMAX = 120.0
BOYS_TAYLOR = 6
# Primitive quartets whose primitive Schwarz bound is below PRIM_FACTOR x the block threshold are skipped.
PRIM_FACTOR = 1e-2
# Integral schemes.
SCHEMES = {'rys': 0, 'os': 1}


# ----------------------------------------------------------------------------
# Plan
# ----------------------------------------------------------------------------
class Plan4c2e:
    """
    Shell-pair data shared by all passes; created by :func:`build_plan`.

    Attributes of general interest
    ------------------------------
    nao : int
    scheme : str
        ``'rys'`` or ``'os'``.
    npairs : int
        Shell pairs kept (sorted by decreasing Schwarz factor ``Q``).
    threshold : float
        Default screening threshold of :func:`direct_jk` and :func:`stored_jk`.
    store_dense : bool or None
        Kind of stored integrals (None: nothing stored).
    store_gb : float
        Size of the stored integrals in GB.
    """

    def __init__(self):
        self.values = None
        self.store_dense = None
        self._bins = {}

    @property
    def store_gb(self):
        return 0.0 if self.values is None else self.values.nbytes / 1e9


@njit(cache=True, nogil=True)
def _shell_pairs(sh_l, sh_nprim, sh_exp, sh_coef, sh_cen, exp_cutoff):
    """All shell pairs (higher angular momentum first) and their surviving primitive pairs."""
    nsh = sh_l.shape[0]
    npair = nsh * (nsh + 1) // 2
    maxpp = 0
    for A in range(nsh):
        for B in range(A + 1):
            maxpp += sh_nprim[A] * sh_nprim[B]
    pair_sh = np.empty((npair, 2), dtype=np.int64)
    pair_pp0 = np.empty(npair, dtype=np.int64)
    pair_ppn = np.empty(npair, dtype=np.int64)
    pair_ab = np.empty((npair, 3), dtype=np.float64)
    pp_p = np.empty(maxpp, dtype=np.float64)
    pp_x = np.empty(maxpp, dtype=np.float64)
    pp_y = np.empty(maxpp, dtype=np.float64)
    pp_z = np.empty(maxpp, dtype=np.float64)
    pp_c = np.empty(maxpp, dtype=np.float64)
    ip = 0
    n = 0
    for A in range(nsh):
        for B in range(A + 1):
            if sh_l[A] >= sh_l[B]:
                X = A
                Y = B
            else:
                X = B
                Y = A
            pair_sh[n, 0] = X
            pair_sh[n, 1] = Y
            abx = sh_cen[X, 0] - sh_cen[Y, 0]
            aby = sh_cen[X, 1] - sh_cen[Y, 1]
            abz = sh_cen[X, 2] - sh_cen[Y, 2]
            pair_ab[n, 0] = abx
            pair_ab[n, 1] = aby
            pair_ab[n, 2] = abz
            r2 = abx * abx + aby * aby + abz * abz
            pair_pp0[n] = ip
            for i in range(sh_nprim[X]):
                ai = sh_exp[X, i]
                for j in range(sh_nprim[Y]):
                    aj = sh_exp[Y, j]
                    p = ai + aj
                    mur2 = ai * aj / p * r2
                    if mur2 > exp_cutoff:
                        continue
                    pp_p[ip] = p
                    pp_x[ip] = (ai * sh_cen[X, 0] + aj * sh_cen[Y, 0]) / p
                    pp_y[ip] = (ai * sh_cen[X, 1] + aj * sh_cen[Y, 1]) / p
                    pp_z[ip] = (ai * sh_cen[X, 2] + aj * sh_cen[Y, 2]) / p
                    pp_c[ip] = sh_coef[X, i] * sh_coef[Y, j] * math.exp(-mur2)
                    ip += 1
            pair_ppn[n] = ip - pair_pp0[n]
            n += 1
    return (pair_sh, pair_pp0, pair_ppn, pair_ab,
            pp_p[:ip].copy(), pp_x[:ip].copy(), pp_y[:ip].copy(), pp_z[:ip].copy(), pp_c[:ip].copy())


# ----------------------------------------------------------------------------
# Tables and scratch arrays
# ----------------------------------------------------------------------------
@njit(cache=True, nogil=True)
def _cart_tables(lt):
    """
    Cartesian triples for L = 0..lt (internal order), the offset of every L, a lookup table, the
    index of every triple lowered by one in each direction (-1 if impossible) and the direction
    in which the Obara-Saika recurrences build every triple.
    """
    n = (lt + 1) * (lt + 2) * (lt + 3) // 6
    xyz = np.empty((n, 3), dtype=np.int64)
    off = np.zeros(lt + 2, dtype=np.int64)
    idx3 = -np.ones((lt + 1, lt + 1, lt + 1), dtype=np.int64)
    g = 0
    for L in range(lt + 1):
        off[L] = g
        for x in range(L, -1, -1):
            for y in range(L - x, -1, -1):
                z = L - x - y
                xyz[g, 0] = x
                xyz[g, 1] = y
                xyz[g, 2] = z
                idx3[x, y, z] = g
                g += 1
    off[lt + 1] = g
    dec = -np.ones((n, 3), dtype=np.int64)
    dirn = np.zeros(n, dtype=np.int64)
    for g in range(n):
        x = xyz[g, 0]
        y = xyz[g, 1]
        z = xyz[g, 2]
        if x > 0:
            dec[g, 0] = idx3[x - 1, y, z]
        if y > 0:
            dec[g, 1] = idx3[x, y - 1, z]
        if z > 0:
            dec[g, 2] = idx3[x, y, z - 1]
        # Build every triple along its smallest nonzero exponent: the recurrence term of the
        # triple lowered twice in that direction is then absent whenever that exponent is 1.
        best = 0
        for d in range(3):
            if xyz[g, d] > 0 and (xyz[g, best] == 0 or xyz[g, d] < xyz[g, best]):
                best = d
        dirn[g] = best
    return xyz, off, idx3, dec, dirn


@njit(cache=True, nogil=True)
def _binomials(n):
    b = np.zeros((n + 1, n + 1), dtype=np.float64)
    for i in range(n + 1):
        b[i, 0] = 1.0
        for k in range(1, i + 1):
            b[i, k] = b[i - 1, k - 1] + (b[i - 1, k] if k < i else 0.0)
    return b


def _boys_table(mmax):
    """
    ``F_m(T_i)`` for ``m = 0..mmax + BOYS_TAYLOR`` on the grid ``T_i = i BOYS_DT``: the highest order
    from its series ``e^{-T} sum_k (2T)^k / ((2m+1)(2m+3)...(2m+2k+1))``, the others by downward recursion.
    """
    T = np.arange(int(round(BOYS_TMAX / BOYS_DT)) + 1) * BOYS_DT
    mtop = mmax + BOYS_TAYLOR
    tab = np.empty((mtop + 1, T.size))
    term = np.full(T.size, 1.0 / (2 * mtop + 1))
    total = term.copy()
    for k in range(1, int(2 * BOYS_TMAX) + 200):
        term = term * (2.0 * T) / (2 * mtop + 2 * k + 1)
        total += term
        if term.max() < 1e-17 * total.min():
            break
    eT = np.exp(-T)
    tab[mtop] = eT * total
    for m in range(mtop, 0, -1):
        tab[m - 1] = (2.0 * T * tab[m] + eT) / (2 * m - 1)
    return tab


@njit(cache=True, fastmath=True, nogil=True, error_model='numpy', boundscheck=False, inline='always')
def _boys_values(nm, T, pref, tab, fm, col):
    """``fm[m, col] = pref F_m(T)`` for ``m = 0..nm-1``."""
    mt = nm - 1
    if T < BOYS_TMAX - BOYS_DT:
        i = int(T / BOYS_DT + 0.5)
        d = i * BOYS_DT - T
        f = tab[mt + BOYS_TAYLOR, i]
        for k in range(BOYS_TAYLOR - 1, -1, -1):
            f = tab[mt + k, i] + d * f / (k + 1)
        e = math.exp(-T)
        fm[mt, col] = pref * f
        for m in range(mt, 0, -1):
            f = (2.0 * T * f + e) / (2 * m - 1)
            fm[m - 1, col] = pref * f
    else:
        f = 0.5 * math.sqrt(math.pi / T)
        fm[0, col] = pref * f
        for m in range(1, nm):
            f *= (2 * m - 1) / (2.0 * T)
            fm[m, col] = pref * f


@njit(cache=True, nogil=True)
def _work_arrays(lmax, scheme):
    """
    Scratch arrays of one thread for shells up to angular momentum ``lmax``. Every entry is written
    before it is read, so they are not initialized.
    """
    lt = 2 * lmax
    ncart = (lmax + 1) * (lmax + 2) // 2
    ne = (lt + 1) * (lt + 2) * (lt + 3) // 6
    roots = np.zeros(MAX_ROOTS, dtype=np.float64)
    weights = np.zeros(MAX_ROOTS, dtype=np.float64)
    if scheme == 0:
        ln = np.empty((10, MAXK), dtype=np.float64)
        g2 = np.empty((3, (lt + 1) * (lt + 1) * MAXK), dtype=np.float64)
        lo = np.empty((17, 1), dtype=np.float64)
        fm = np.empty((1, 1), dtype=np.float64)
        V = np.empty(1, dtype=np.float64)
    else:
        ln = np.empty((10, 1), dtype=np.float64)
        g2 = np.empty((3, 1), dtype=np.float64)
        lo = np.empty((17, MAXK_OS), dtype=np.float64)
        fm = np.empty((2 * lt + 1, MAXK_OS), dtype=np.float64)
        V = np.empty(max(OS_WORK, ne * ne * (2 * lt + 1)), dtype=np.float64)
    E = np.empty(ne * ne, dtype=np.float64)
    F1 = np.empty(ne * ncart * ncart, dtype=np.float64)
    blk = np.empty(ncart ** 4, dtype=np.float64)
    pw = np.ones((2, 3, lmax + 1), dtype=np.float64)
    return roots, weights, ln, g2, lo, fm, V, E, F1, blk, pw


@njit(cache=True, fastmath=True, nogil=True, error_model='numpy', boundscheck=False, inline='always')
def _boys0(x):
    if x < 1e-12:
        return 1.0 - x / 3.0
    sx = math.sqrt(x)
    return SQRT_PI_HALF * math.erf(sx) / sx


# ----------------------------------------------------------------------------
# Rys quadrature
# ----------------------------------------------------------------------------
@njit(cache=True, fastmath=True, nogil=True, error_model='numpy', boundscheck=False, inline='always')
def _chunk_rys(K, lbra, lket, ln, g2, E, e0, ne, f0, nf, cart_xyz):
    """Rys recurrences of ``K`` (primitive quartet, root) lanes; adds their (e0|f0) to ``E``."""
    b00 = ln[0]
    b10 = ln[1]
    b01 = ln[2]
    c0x = ln[3]
    c0y = ln[4]
    c0z = ln[5]
    d0x = ln[6]
    d0y = ln[7]
    d0z = ln[8]
    wz = ln[9]
    gx = g2[0]
    gy = g2[1]
    gz = g2[2]
    lk1 = lket + 1
    S = MAXK
    for k in range(K):
        gx[k] = 1.0
        gy[k] = 1.0
        gz[k] = wz[k]
    if lbra > 0:
        o1 = lk1 * S
        for k in range(K):
            gx[o1 + k] = c0x[k]
            gy[o1 + k] = c0y[k]
            gz[o1 + k] = c0z[k] * wz[k]
        for n in range(1, lbra):
            o1 = (n + 1) * lk1 * S
            o0 = n * lk1 * S
            om = (n - 1) * lk1 * S
            for k in range(K):
                nb = n * b10[k]
                gx[o1 + k] = c0x[k] * gx[o0 + k] + nb * gx[om + k]
                gy[o1 + k] = c0y[k] * gy[o0 + k] + nb * gy[om + k]
                gz[o1 + k] = c0z[k] * gz[o0 + k] + nb * gz[om + k]
    if lket > 0:
        for k in range(K):
            gx[S + k] = d0x[k]
            gy[S + k] = d0y[k]
            gz[S + k] = d0z[k] * wz[k]
        for n in range(1, lbra + 1):
            o1 = (n * lk1 + 1) * S
            o0 = n * lk1 * S
            om = (n - 1) * lk1 * S
            for k in range(K):
                nb = n * b00[k]
                gx[o1 + k] = d0x[k] * gx[o0 + k] + nb * gx[om + k]
                gy[o1 + k] = d0y[k] * gy[o0 + k] + nb * gy[om + k]
                gz[o1 + k] = d0z[k] * gz[o0 + k] + nb * gz[om + k]
        for m in range(1, lket):
            o1 = (m + 1) * S
            o0 = m * S
            om = (m - 1) * S
            for k in range(K):
                mb = m * b01[k]
                gx[o1 + k] = d0x[k] * gx[o0 + k] + mb * gx[om + k]
                gy[o1 + k] = d0y[k] * gy[o0 + k] + mb * gy[om + k]
                gz[o1 + k] = d0z[k] * gz[o0 + k] + mb * gz[om + k]
            for n in range(1, lbra + 1):
                o1 = (n * lk1 + m + 1) * S
                o0 = (n * lk1 + m) * S
                om = (n * lk1 + m - 1) * S
                on = ((n - 1) * lk1 + m) * S
                for k in range(K):
                    mb = m * b01[k]
                    nb = n * b00[k]
                    gx[o1 + k] = d0x[k] * gx[o0 + k] + mb * gx[om + k] + nb * gx[on + k]
                    gy[o1 + k] = d0y[k] * gy[o0 + k] + mb * gy[om + k] + nb * gy[on + k]
                    gz[o1 + k] = d0z[k] * gz[o0 + k] + mb * gz[om + k] + nb * gz[on + k]
    for ie in range(ne):
        ex = cart_xyz[e0 + ie, 0] * lk1
        ey = cart_xyz[e0 + ie, 1] * lk1
        ez = cart_xyz[e0 + ie, 2] * lk1
        for jf in range(nf):
            ox = (ex + cart_xyz[f0 + jf, 0]) * S
            oy = (ey + cart_xyz[f0 + jf, 1]) * S
            oz = (ez + cart_xyz[f0 + jf, 2]) * S
            acc = 0.0
            for k in range(K):
                acc += gx[ox + k] * gy[oy + k] * gz[oz + k]
            E[ie * nf + jf] += acc


@njit(cache=True, fastmath=True, nogil=True, error_model='numpy', boundscheck=False, inline='always')
def _e0f0_rys(a0, a1, b0, b1, thr_prim, pp_p, pp_x, pp_y, pp_z, pp_c, pp_q, xx, xy, xz, zx, zy, zz,
              lbra, lket, e0, ne, f0, nf, cart_xyz, roots, weights, ln, g2, E):
    """Adds (e0|f0) of all screened primitive quartets to ``E`` (Rys quadrature)."""
    nroots = (lbra + lket) // 2 + 1
    kmax = MAXK // nroots
    nk = 0
    for a in range(a0, a1):
        qa = pp_q[a]
        if qa * pp_q[b0] < thr_prim:
            break
        p = pp_p[a]
        px = pp_x[a]
        py = pp_y[a]
        pz = pp_z[a]
        ca = pp_c[a]
        pax = px - xx
        pay = py - xy
        paz = pz - xz
        inv_2p = 0.5 / p
        for b in range(b0, b1):
            if qa * pp_q[b] < thr_prim:
                break
            q = pp_p[b]
            qx = pp_x[b]
            qy = pp_y[b]
            qz = pp_z[b]
            pq = p + q
            inv_pq = 1.0 / pq
            pqx = px - qx
            pqy = py - qy
            pqz = pz - qz
            Roots(nroots, p * q * inv_pq * (pqx * pqx + pqy * pqy + pqz * pqz), roots, weights)
            pref = TWO_PI_52 / (p * q * math.sqrt(pq)) * ca * pp_c[b]
            qcx = qx - zx
            qcy = qy - zy
            qcz = qz - zz
            inv_2q = 0.5 / q
            lane = nk * nroots
            for r in range(nroots):
                u = roots[r]
                t2 = u / (1.0 + u)
                ln[9, lane] = weights[r] * pref
                if lbra > 0:
                    fb = q * t2 * inv_pq
                    ln[3, lane] = pax - fb * pqx
                    ln[4, lane] = pay - fb * pqy
                    ln[5, lane] = paz - fb * pqz
                    if lbra > 1:
                        ln[1, lane] = (1.0 - fb) * inv_2p
                if lket > 0:
                    fk = p * t2 * inv_pq
                    ln[6, lane] = qcx + fk * pqx
                    ln[7, lane] = qcy + fk * pqy
                    ln[8, lane] = qcz + fk * pqz
                    if lket > 1:
                        ln[2, lane] = (1.0 - fk) * inv_2q
                    if lbra > 0:
                        ln[0, lane] = 0.5 * t2 * inv_pq
                lane += 1
            nk += 1
            if nk == kmax:
                _chunk_rys(nk * nroots, lbra, lket, ln, g2, E, e0, ne, f0, nf, cart_xyz)
                nk = 0
    if nk > 0:
        _chunk_rys(nk * nroots, lbra, lket, ln, g2, E, e0, ne, f0, nf, cart_xyz)


# ----------------------------------------------------------------------------
# Obara-Saika
# ----------------------------------------------------------------------------
@njit(cache=True, fastmath=True, nogil=True, error_model='numpy', boundscheck=False, inline='always')
def _chunk_os(K, lx, lket, ltot, ea, fa, lo, fm, V, E, e0, ne, f0, nf, cart_xyz, cart_off, dec, dirn):
    """
    Obara-Saika recurrences of ``K`` primitive-quartet lanes; adds their (e0|f0) to ``E``.
    ``V[((e fa + f) (ltot + 1) + m) K + k]`` holds ``[e0|f0]^(m)`` of lane ``k`` for the triples
    ``e < ea`` (|e| <= l_X + l_Y) and ``f < fa`` (|f| <= l_Z + l_W). At ket level ``|f|`` only the
    entries the final ``(e0|f0)``, ``|e| >= l_X``, depend on are built: ``m <= l_ket - |f|`` and
    ``|e| >= l_X - (l_ket - |f|)`` (every level down the ket recurrence raises ``m`` and, through the
    bra-ket coupling term, lowers ``|e|`` by at most one).
    """
    nm = ltot + 1
    f2p = lo[12]
    rp = lo[13]
    f2q = lo[14]
    rq = lo[15]
    f2pq = lo[16]
    for m in range(nm):
        o = m * K
        for k in range(K):
            V[o + k] = fm[m, k]
    # [e0|00]^(m), built on X
    for g in range(1, ea):
        i = dirn[g]
        e = dec[g, i]
        lg = cart_xyz[g, 0] + cart_xyz[g, 1] + cart_xyz[g, 2]
        ei = cart_xyz[e, i]
        pa = lo[i]
        wp = lo[3 + i]
        if ei > 0:
            e2 = dec[e, i]
            for m in range(nm - lg):
                o = (g * fa * nm + m) * K
                oa = (e * fa * nm + m) * K
                ob = (e2 * fa * nm + m) * K
                for k in range(K):
                    V[o + k] = (pa[k] * V[oa + k] + wp[k] * V[oa + K + k]
                                + ei * f2p[k] * (V[ob + k] - rp[k] * V[ob + K + k]))
        else:
            for m in range(nm - lg):
                o = (g * fa * nm + m) * K
                oa = (e * fa * nm + m) * K
                for k in range(K):
                    V[o + k] = pa[k] * V[oa + k] + wp[k] * V[oa + K + k]
    # [e0|f0]^(m), built on Z
    for h in range(1, fa):
        i = dirn[h]
        f = dec[h, i]
        lh = cart_xyz[h, 0] + cart_xyz[h, 1] + cart_xyz[h, 2]
        fi = cart_xyz[f, i]
        f2 = dec[f, i] if fi > 0 else 0
        qc = lo[6 + i]
        wq = lo[9 + i]
        mmax = lket - lh
        for g in range(cart_off[max(0, lx - mmax)], ea):
            gi = cart_xyz[g, i]
            eg = dec[g, i] if gi > 0 else 0
            for m in range(mmax + 1):
                o = ((g * fa + h) * nm + m) * K
                oa = ((g * fa + f) * nm + m) * K
                ob = ((g * fa + f2) * nm + m) * K
                oc = ((eg * fa + f) * nm + m + 1) * K
                if fi > 0 and gi > 0:
                    for k in range(K):
                        V[o + k] = (qc[k] * V[oa + k] + wq[k] * V[oa + K + k]
                                    + fi * f2q[k] * (V[ob + k] - rq[k] * V[ob + K + k])
                                    + gi * f2pq[k] * V[oc + k])
                elif fi > 0:
                    for k in range(K):
                        V[o + k] = (qc[k] * V[oa + k] + wq[k] * V[oa + K + k]
                                    + fi * f2q[k] * (V[ob + k] - rq[k] * V[ob + K + k]))
                elif gi > 0:
                    for k in range(K):
                        V[o + k] = qc[k] * V[oa + k] + wq[k] * V[oa + K + k] + gi * f2pq[k] * V[oc + k]
                else:
                    for k in range(K):
                        V[o + k] = qc[k] * V[oa + k] + wq[k] * V[oa + K + k]
    for ie in range(ne):
        g = e0 + ie
        for jf in range(nf):
            o = ((g * fa + f0 + jf) * nm) * K
            acc = 0.0
            for k in range(K):
                acc += V[o + k]
            E[ie * nf + jf] += acc


@njit(cache=True, fastmath=True, nogil=True, error_model='numpy', boundscheck=False, inline='always')
def _e0f0_os(a0, a1, b0, b1, thr_prim, pp_p, pp_x, pp_y, pp_z, pp_c, pp_q, xx, xy, xz, zx, zy, zz,
             lx, lbra, lket, e0, ne, f0, nf, cart_xyz, cart_off, dec, dirn, boys, lo, fm, V, E):
    """Adds (e0|f0) of all screened primitive quartets to ``E`` (Obara-Saika)."""
    ltot = lbra + lket
    nm = ltot + 1
    ea = cart_off[lbra + 1]
    fa = cart_off[lket + 1]
    kmax = V.shape[0] // (ea * fa * nm)
    if kmax > MAXK_OS:
        kmax = MAXK_OS
    nk = 0
    for a in range(a0, a1):
        qa = pp_q[a]
        if qa * pp_q[b0] < thr_prim:
            break
        p = pp_p[a]
        px = pp_x[a]
        py = pp_y[a]
        pz = pp_z[a]
        ca = pp_c[a]
        for b in range(b0, b1):
            if qa * pp_q[b] < thr_prim:
                break
            q = pp_p[b]
            qx = pp_x[b]
            qy = pp_y[b]
            qz = pp_z[b]
            pq = p + q
            inv_pq = 1.0 / pq
            pqx = px - qx
            pqy = py - qy
            pqz = pz - qz
            _boys_values(nm, p * q * inv_pq * (pqx * pqx + pqy * pqy + pqz * pqz),
                         TWO_PI_52 / (p * q * math.sqrt(pq)) * ca * pp_c[b], boys, fm, nk)
            wx = (p * px + q * qx) * inv_pq
            wy = (p * py + q * qy) * inv_pq
            wz = (p * pz + q * qz) * inv_pq
            lo[0, nk] = px - xx
            lo[1, nk] = py - xy
            lo[2, nk] = pz - xz
            lo[3, nk] = wx - px
            lo[4, nk] = wy - py
            lo[5, nk] = wz - pz
            lo[6, nk] = qx - zx
            lo[7, nk] = qy - zy
            lo[8, nk] = qz - zz
            lo[9, nk] = wx - qx
            lo[10, nk] = wy - qy
            lo[11, nk] = wz - qz
            lo[12, nk] = 0.5 / p
            lo[13, nk] = q * inv_pq
            lo[14, nk] = 0.5 / q
            lo[15, nk] = p * inv_pq
            lo[16, nk] = 0.5 * inv_pq
            nk += 1
            if nk == kmax:
                _chunk_os(nk, lx, lket, ltot, ea, fa, lo, fm, V, E, e0, ne, f0, nf, cart_xyz, cart_off, dec, dirn)
                nk = 0
    if nk > 0:
        _chunk_os(nk, lx, lket, ltot, ea, fa, lo, fm, V, E, e0, ne, f0, nf, cart_xyz, cart_off, dec, dirn)


# ----------------------------------------------------------------------------
# Shell quartet: (e0|f0) by either scheme, then the horizontal transfers
# ----------------------------------------------------------------------------
@njit(cache=True, fastmath=True, nogil=True, error_model='numpy', boundscheck=False, inline='always')
def _quartet(scheme, i, j, thr_prim, pair_sh, pair_pp0, pair_ppn, pair_ab, pp_p, pp_x, pp_y, pp_z, pp_c, pp_q,
             sh_l, sh_off, sh_nbf, sh_cen, bfs_lmn, comp_scale, cart_xyz, cart_off, idx3, binom, dec, dirn, boys,
             roots, weights, ln, g2, lo, fm, V, E, F1, blk, pw):
    """
    Normalized block ``(XY|ZW)`` of bra pair ``i`` and ket pair ``j`` written to ``blk``
    (index ``((x nY + y) nZ + z) nW + w``), by Rys quadrature (``scheme`` 0) or Obara-Saika
    (``scheme`` 1). Primitive quartets with ``q_a q_b < thr_prim`` (primitive-pair Schwarz
    factors, sorted in decreasing order within every pair) are skipped. Returns the number of
    elements.
    """
    X = pair_sh[i, 0]
    Y = pair_sh[i, 1]
    Z = pair_sh[j, 0]
    W = pair_sh[j, 1]
    lx = sh_l[X]
    ly = sh_l[Y]
    lz = sh_l[Z]
    lw = sh_l[W]
    nX = sh_nbf[X]
    nY = sh_nbf[Y]
    nZ = sh_nbf[Z]
    nW = sh_nbf[W]
    ncomp = nX * nY * nZ * nW
    lbra = lx + ly
    lket = lz + lw
    a0 = pair_pp0[i]
    a1 = a0 + pair_ppn[i]
    b0 = pair_pp0[j]
    b1 = b0 + pair_ppn[j]
    x0 = sh_off[X]
    y0 = sh_off[Y]
    z0 = sh_off[Z]
    w0 = sh_off[W]

    if lbra + lket == 0:
        acc = 0.0
        for a in range(a0, a1):
            qa = pp_q[a]
            if qa * pp_q[b0] < thr_prim:
                break
            p = pp_p[a]
            px = pp_x[a]
            py = pp_y[a]
            pz = pp_z[a]
            ca = pp_c[a]
            for b in range(b0, b1):
                if qa * pp_q[b] < thr_prim:
                    break
                q = pp_p[b]
                pq = p + q
                dx = px - pp_x[b]
                dy = py - pp_y[b]
                dz = pz - pp_z[b]
                acc += ca * pp_c[b] / (p * q * math.sqrt(pq)) * _boys0(p * q / pq * (dx * dx + dy * dy + dz * dz))
        blk[0] = TWO_PI_52 * acc * comp_scale[x0] * comp_scale[y0] * comp_scale[z0] * comp_scale[w0]
        return 1

    e0 = cart_off[lx]
    ne = cart_off[lbra + 1] - e0
    f0 = cart_off[lz]
    nf = cart_off[lket + 1] - f0
    for c in range(ne * nf):
        E[c] = 0.0
    if scheme == 0:
        _e0f0_rys(a0, a1, b0, b1, thr_prim, pp_p, pp_x, pp_y, pp_z, pp_c, pp_q,
                  sh_cen[X, 0], sh_cen[X, 1], sh_cen[X, 2], sh_cen[Z, 0], sh_cen[Z, 1], sh_cen[Z, 2],
                  lbra, lket, e0, ne, f0, nf, cart_xyz, roots, weights, ln, g2, E)
    else:
        _e0f0_os(a0, a1, b0, b1, thr_prim, pp_p, pp_x, pp_y, pp_z, pp_c, pp_q,
                 sh_cen[X, 0], sh_cen[X, 1], sh_cen[X, 2], sh_cen[Z, 0], sh_cen[Z, 1], sh_cen[Z, 2],
                 lx, lbra, lket, e0, ne, f0, nf, cart_xyz, cart_off, dec, dirn, boys, lo, fm, V, E)

    # Horizontal transfers (once per shell quartet): (e0|f0) -> (e0|zw) -> (xy|zw).
    if ly == 0 and lw == 0:
        for x in range(nX):
            ie = (idx3[bfs_lmn[x0 + x, 0], bfs_lmn[x0 + x, 1], bfs_lmn[x0 + x, 2]] - e0) * nf
            sx = comp_scale[x0 + x] * comp_scale[y0] * comp_scale[w0]
            for z in range(nZ):
                jf = idx3[bfs_lmn[z0 + z, 0], bfs_lmn[z0 + z, 1], bfs_lmn[z0 + z, 2]] - f0
                blk[x * nZ + z] = E[ie + jf] * sx * comp_scale[z0 + z]
        return ncomp
    zw = nZ * nW
    if lw > 0:
        for d in range(3):
            pw[1, d, 0] = 1.0
            for k in range(1, lw + 1):
                pw[1, d, k] = pw[1, d, k - 1] * pair_ab[j, d]
    if ly > 0:
        for d in range(3):
            pw[0, d, 0] = 1.0
            for k in range(1, ly + 1):
                pw[0, d, k] = pw[0, d, k - 1] * pair_ab[i, d]
    for z in range(nZ):
        cx = bfs_lmn[z0 + z, 0]
        cy = bfs_lmn[z0 + z, 1]
        cz = bfs_lmn[z0 + z, 2]
        for w in range(nW):
            col = z * nW + w
            if lw == 0:
                jf = idx3[cx, cy, cz] - f0
                for ie in range(ne):
                    F1[ie * zw + col] = E[ie * nf + jf]
            else:
                dx = bfs_lmn[w0 + w, 0]
                dy = bfs_lmn[w0 + w, 1]
                dz = bfs_lmn[w0 + w, 2]
                for ie in range(ne):
                    F1[ie * zw + col] = 0.0
                for qx in range(dx + 1):
                    fx = binom[dx, qx] * pw[1, 0, dx - qx]
                    for qy in range(dy + 1):
                        fxy = fx * binom[dy, qy] * pw[1, 1, dy - qy]
                        for qz in range(dz + 1):
                            coef = fxy * binom[dz, qz] * pw[1, 2, dz - qz]
                            jf = idx3[cx + qx, cy + qy, cz + qz] - f0
                            for ie in range(ne):
                                F1[ie * zw + col] += coef * E[ie * nf + jf]
    for x in range(nX):
        ax = bfs_lmn[x0 + x, 0]
        ay = bfs_lmn[x0 + x, 1]
        az = bfs_lmn[x0 + x, 2]
        sx = comp_scale[x0 + x]
        for y in range(nY):
            o = (x * nY + y) * zw
            sxy = sx * comp_scale[y0 + y]
            if ly == 0:
                ie = idx3[ax, ay, az] - e0
                for col in range(zw):
                    blk[o + col] = F1[ie * zw + col]
            else:
                bx = bfs_lmn[y0 + y, 0]
                by = bfs_lmn[y0 + y, 1]
                bz = bfs_lmn[y0 + y, 2]
                for col in range(zw):
                    blk[o + col] = 0.0
                for qx in range(bx + 1):
                    fx = binom[bx, qx] * pw[0, 0, bx - qx]
                    for qy in range(by + 1):
                        fxy = fx * binom[by, qy] * pw[0, 1, by - qy]
                        for qz in range(bz + 1):
                            coef = fxy * binom[bz, qz] * pw[0, 2, bz - qz]
                            ie = idx3[ax + qx, ay + qy, az + qz] - e0
                            for col in range(zw):
                                blk[o + col] += coef * F1[ie * zw + col]
            for z in range(nZ):
                sxyz = sxy * comp_scale[z0 + z]
                for w in range(nW):
                    blk[o + z * nW + w] *= sxyz * comp_scale[w0 + w]
    return ncomp


@njit(cache=True, fastmath=True, nogil=True, error_model='numpy', boundscheck=False, inline='always')
def _contract(blk, s, X, Y, Z, W, sh_off, sh_nbf, dmat, H, G, with_k):
    """Add one block (scaled by ``s``) to the half Coulomb (H) and half exchange (G) matrices."""
    x0 = sh_off[X]
    y0 = sh_off[Y]
    z0 = sh_off[Z]
    w0 = sh_off[W]
    nX = sh_nbf[X]
    nY = sh_nbf[Y]
    nZ = sh_nbf[Z]
    nW = sh_nbf[W]
    c = 0
    for x in range(nX):
        ix = x0 + x
        for y in range(nY):
            iy = y0 + y
            dxy = s * dmat[ix, iy]
            hxy = 0.0
            for z in range(nZ):
                iz = z0 + z
                if with_k:
                    dxz = s * dmat[ix, iz]
                    dyz = s * dmat[iy, iz]
                    gxz = 0.0
                    gyz = 0.0
                    for w in range(nW):
                        iw = w0 + w
                        v = blk[c]
                        c += 1
                        hxy += v * dmat[iz, iw]
                        H[iz, iw] += v * dxy
                        gxz += v * dmat[iy, iw]
                        gyz += v * dmat[ix, iw]
                        G[ix, iw] += v * dyz
                        G[iy, iw] += v * dxz
                    G[ix, iz] += s * gxz
                    G[iy, iz] += s * gyz
                else:
                    for w in range(nW):
                        iw = w0 + w
                        v = blk[c]
                        c += 1
                        hxy += v * dmat[iz, iw]
                        H[iz, iw] += v * dxy
            H[ix, iy] += s * hxy


# ----------------------------------------------------------------------------
# Passes
# ----------------------------------------------------------------------------
@njit(parallel=True, cache=True, fastmath=True, nogil=True, error_model='numpy', boundscheck=False)
def _pair_schwarz(pair_sh, pair_pp0, pair_ppn, pair_ab, pp_p, pp_x, pp_y, pp_z, pp_c, pp_q,
                  sh_l, sh_off, sh_nbf, sh_cen, bfs_lmn, comp_scale, cart_xyz, cart_off, idx3, binom, dec, dirn, boys,
                  lmax, scheme, nchunk):
    """Q_i = sqrt(max |(xy|xy)|) of every shell pair (``nchunk`` contiguous chunks of pairs)."""
    npair = pair_sh.shape[0]
    Q = np.zeros(npair, dtype=np.float64)
    for c in prange(nchunk):
        roots, weights, ln, g2, lo, fm, V, E, F1, blk, pw = _work_arrays(lmax, scheme)
        for i in range(c * npair // nchunk, (c + 1) * npair // nchunk):
            _quartet(scheme, i, i, 0.0, pair_sh, pair_pp0, pair_ppn, pair_ab, pp_p, pp_x, pp_y, pp_z, pp_c, pp_q,
                     sh_l, sh_off, sh_nbf, sh_cen, bfs_lmn, comp_scale, cart_xyz, cart_off, idx3, binom, dec, dirn,
                     boys, roots, weights, ln, g2, lo, fm, V, E, F1, blk, pw)
            nX = sh_nbf[pair_sh[i, 0]]
            nY = sh_nbf[pair_sh[i, 1]]
            m = 0.0
            for x in range(nX):
                for y in range(nY):
                    v = abs(blk[((x * nY + y) * nX + x) * nY + y])
                    if v > m:
                        m = v
            Q[i] = math.sqrt(m)
    return Q


@njit(cache=True, nogil=True)
def _bra_costs(Q, threshold_q, pair_sh, pair_ppn, sh_l, sh_nbf):
    """Estimated work of every bra pair (Schwarz-significant ket pairs only) and the ket ends."""
    npair = Q.shape[0]
    cost = np.zeros(npair, dtype=np.float64)
    jend = np.empty(npair, dtype=np.int64)
    for i in range(npair):
        X = pair_sh[i, 0]
        Y = pair_sh[i, 1]
        nci = sh_nbf[X] * sh_nbf[Y]
        li = sh_l[X] + sh_l[Y]
        ppi = pair_ppn[i]
        acc = 0.0
        j = i
        while j < npair and Q[i] * Q[j] >= threshold_q:
            Z = pair_sh[j, 0]
            W = pair_sh[j, 1]
            ncj = sh_nbf[Z] * sh_nbf[W]
            nr = (li + sh_l[Z] + sh_l[W]) // 2 + 1
            acc += ppi * pair_ppn[j] * (nr * (nci * ncj + 12.0) + 25.0) + 2.0 * nci * ncj
            j += 1
        cost[i] = acc
        jend[i] = j
    return cost, jend


@njit(cache=True, nogil=True)
def _shell_dmax(dmat, sh_off, sh_nbf):
    nsh = sh_off.shape[0]
    dsh = np.zeros((nsh, nsh), dtype=np.float64)
    for A in range(nsh):
        for B in range(A + 1):
            m = 0.0
            for a in range(sh_off[A], sh_off[A] + sh_nbf[A]):
                for b in range(sh_off[B], sh_off[B] + sh_nbf[B]):
                    v = abs(dmat[a, b])
                    if v > m:
                        m = v
            dsh[A, B] = m
            dsh[B, A] = m
    return dsh


@njit(parallel=True, cache=True, fastmath=True, nogil=True, error_model='numpy', boundscheck=False)
def _direct_pass(bin_off, bin_items, Q, pair_sh, pair_pp0, pair_ppn, pair_ab, pp_p, pp_x, pp_y, pp_z, pp_c, pp_q,
                 sh_l, sh_off, sh_nbf, sh_cen, bfs_lmn, comp_scale, cart_xyz, cart_off, idx3, binom, dec, dirn, boys,
                 lmax, scheme, dmat, dsh, dmax, threshold, with_k):
    nbins = bin_off.shape[0] - 1
    nbf = dmat.shape[0]
    npair = Q.shape[0]
    H = np.zeros((nbins, nbf, nbf), dtype=np.float64)
    if with_k:
        G = np.zeros((nbins, nbf, nbf), dtype=np.float64)
    else:
        G = np.zeros((nbins, 1, 1), dtype=np.float64)
    dlim = 4.0 * dmax
    for bn in prange(nbins):
        roots, weights, ln, g2, lo, fm, V, E, F1, blk, pw = _work_arrays(lmax, scheme)
        Hb = H[bn]
        Gb = G[bn]
        for t in range(bin_off[bn], bin_off[bn + 1]):
            i = bin_items[t]
            X = pair_sh[i, 0]
            Y = pair_sh[i, 1]
            qi = Q[i]
            dxy4 = 4.0 * dsh[X, Y]
            for j in range(i, npair):
                qq = qi * Q[j]
                if qq * dlim < threshold:
                    break
                Z = pair_sh[j, 0]
                W = pair_sh[j, 1]
                dm = max(dxy4, 4.0 * dsh[Z, W])
                if with_k:
                    dm = max(dm, dsh[X, Z], dsh[X, W], dsh[Y, Z], dsh[Y, W])
                if qq * dm < threshold:
                    continue
                _quartet(scheme, i, j, PRIM_FACTOR * threshold / dm, pair_sh, pair_pp0, pair_ppn, pair_ab,
                         pp_p, pp_x, pp_y, pp_z, pp_c, pp_q, sh_l, sh_off, sh_nbf, sh_cen, bfs_lmn, comp_scale,
                         cart_xyz, cart_off, idx3, binom, dec, dirn, boys,
                         roots, weights, ln, g2, lo, fm, V, E, F1, blk, pw)
                s = 1.0
                if X == Y:
                    s *= 0.5
                if Z == W:
                    s *= 0.5
                if i == j:
                    s *= 0.5
                _contract(blk, s, X, Y, Z, W, sh_off, sh_nbf, dmat, Hb, Gb, with_k)
    return H, G


@njit(parallel=True, cache=True, fastmath=True, nogil=True, error_model='numpy', boundscheck=False)
def _store_pass(bin_off, bin_items, jend, row_off, thr_prim, pair_sh, pair_pp0, pair_ppn, pair_ab,
                pp_p, pp_x, pp_y, pp_z, pp_c, pp_q,
                sh_l, sh_off, sh_nbf, sh_cen, bfs_lmn, comp_scale, cart_xyz, cart_off, idx3, binom, dec, dirn, boys,
                lmax, scheme, values):
    """Evaluate and store the blocks ``(i, j)``, ``i <= j < jend[i]``, scaled by their degeneracy factor."""
    nbins = bin_off.shape[0] - 1
    for bn in prange(nbins):
        roots, weights, ln, g2, lo, fm, V, E, F1, blk, pw = _work_arrays(lmax, scheme)
        for t in range(bin_off[bn], bin_off[bn + 1]):
            i = bin_items[t]
            X = pair_sh[i, 0]
            Y = pair_sh[i, 1]
            off = row_off[i]
            for j in range(i, jend[i]):
                nc = _quartet(scheme, i, j, thr_prim, pair_sh, pair_pp0, pair_ppn, pair_ab,
                              pp_p, pp_x, pp_y, pp_z, pp_c, pp_q, sh_l, sh_off, sh_nbf, sh_cen, bfs_lmn, comp_scale,
                              cart_xyz, cart_off, idx3, binom, dec, dirn, boys,
                              roots, weights, ln, g2, lo, fm, V, E, F1, blk, pw)
                s = 1.0
                if X == Y:
                    s *= 0.5
                if pair_sh[j, 0] == pair_sh[j, 1]:
                    s *= 0.5
                if i == j:
                    s *= 0.5
                for c in range(nc):
                    values[off + c] = s * blk[c]
                off += nc


@njit(parallel=True, cache=True, fastmath=True, nogil=True, error_model='numpy', boundscheck=False)
def _stored_pass(bin_off, bin_items, jend, row_off, Q, pair_sh, sh_off, sh_nbf, values,
                 dmat, dsh, dmax, threshold, with_k):
    nbins = bin_off.shape[0] - 1
    nbf = dmat.shape[0]
    H = np.zeros((nbins, nbf, nbf), dtype=np.float64)
    if with_k:
        G = np.zeros((nbins, nbf, nbf), dtype=np.float64)
    else:
        G = np.zeros((nbins, 1, 1), dtype=np.float64)
    dlim = 4.0 * dmax
    for bn in prange(nbins):
        Hb = H[bn]
        Gb = G[bn]
        for t in range(bin_off[bn], bin_off[bn + 1]):
            i = bin_items[t]
            X = pair_sh[i, 0]
            Y = pair_sh[i, 1]
            nci = sh_nbf[X] * sh_nbf[Y]
            qi = Q[i]
            dxy4 = 4.0 * dsh[X, Y]
            off = row_off[i]
            for j in range(i, jend[i]):
                Z = pair_sh[j, 0]
                W = pair_sh[j, 1]
                nc = nci * sh_nbf[Z] * sh_nbf[W]
                qq = qi * Q[j]
                if qq * dlim < threshold:
                    break
                dm = max(dxy4, 4.0 * dsh[Z, W])
                if with_k:
                    dm = max(dm, dsh[X, Z], dsh[X, W], dsh[Y, Z], dsh[Y, W])
                if qq * dm >= threshold:
                    _contract(values[off:off + nc], 1.0, X, Y, Z, W, sh_off, sh_nbf, dmat, Hb, Gb, with_k)
                off += nc
    return H, G


@njit(cache=True, nogil=True)
def _row_sizes(jend, pair_sh, sh_nbf):
    npair = jend.shape[0]
    ncomp = np.empty(npair, dtype=np.int64)
    for i in range(npair):
        ncomp[i] = sh_nbf[pair_sh[i, 0]] * sh_nbf[pair_sh[i, 1]]
    cum = np.zeros(npair + 1, dtype=np.int64)
    for i in range(npair):
        cum[i + 1] = cum[i] + ncomp[i]
    sizes = np.empty(npair, dtype=np.int64)
    for i in range(npair):
        sizes[i] = ncomp[i] * (cum[jend[i]] - cum[i])
    return sizes


@njit(cache=True, nogil=True)
def _reduce(H, G, with_k):
    nbins = H.shape[0]
    nbf = H.shape[1]
    J = np.zeros((nbf, nbf), dtype=np.float64)
    for bn in range(nbins):
        for a in range(nbf):
            for b in range(nbf):
                J[a, b] += H[bn, a, b]
    for a in range(nbf):
        for b in range(a + 1):
            v = 2.0 * (J[a, b] + J[b, a])
            J[a, b] = v
            J[b, a] = v
    if not with_k:
        return J, np.zeros((1, 1), dtype=np.float64)
    K = np.zeros((nbf, nbf), dtype=np.float64)
    for bn in range(nbins):
        for a in range(nbf):
            for b in range(nbf):
                K[a, b] += G[bn, a, b]
    for a in range(nbf):
        for b in range(a + 1):
            v = K[a, b] + K[b, a]
            K[a, b] = v
            K[b, a] = v
    return J, K


# ----------------------------------------------------------------------------
# Python drivers
# ----------------------------------------------------------------------------
def _nchunks(n):
    """Chunks of a parallel loop over ``n`` items whose scratch arrays are allocated once per chunk."""
    return max(1, min(n, 16 * int(numba.get_num_threads())))


def build_plan(basis, threshold=1e-12, scheme='rys'):
    """
    Shell pairs with their primitive pairs and Schwarz factors, sorted by decreasing ``Q``.

    Parameters
    ----------
    basis : Basis
    threshold : float
        Screening threshold of the density-weighted Schwarz bound (default of
        :func:`direct_jk` and :func:`stored_jk`). Pairs with ``Q_i max(Q) < threshold`` are
        dropped.
    scheme : {'rys', 'os'}
        Integral scheme: Rys quadrature or the Obara-Saika recurrences.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"Unknown integral scheme {scheme!r}; use one of {sorted(SCHEMES)}.")
    packed = pack_basis_arrays(basis)
    bfs_coords, contr_norms, bfs_lmn, bfs_nprim, coeffs, prim_norms, expnts = packed
    bf_coef = contr_norms[:, None] * coeffs * prim_norms
    sh_off = np.ascontiguousarray(basis.shell_bfs_offset, dtype=np.int64)
    sh_nbf = np.ascontiguousarray(basis.bfs_nbfshell, dtype=np.int64)
    sh_l = np.ascontiguousarray(np.asarray(basis.shells, dtype=np.int64) - 1)
    nsh = sh_off.shape[0]
    lmax = int(sh_l.max())
    if 2 * lmax + 1 > MAX_ROOTS:
        raise ValueError(f'Shells up to l = {(MAX_ROOTS - 1) // 2} are supported (got l = {lmax}).')

    sh_nprim = np.ascontiguousarray(bfs_nprim[sh_off])
    sh_exp = np.ascontiguousarray(expnts[sh_off])
    sh_coef = np.ascontiguousarray(bf_coef[sh_off])
    sh_cen = np.ascontiguousarray(bfs_coords[sh_off])
    # The primitive coefficients of the Cartesian components of a shell differ by one factor.
    comp_scale = np.empty(basis.bfs_nao, dtype=np.float64)
    for s in range(nsh):
        n = sh_nprim[s]
        k = int(np.argmax(np.abs(sh_coef[s, :n])))
        for f in range(sh_off[s], sh_off[s] + sh_nbf[s]):
            comp_scale[f] = bf_coef[f, k] / sh_coef[s, k]
            if not np.allclose(bf_coef[f, :n], comp_scale[f] * sh_coef[s, :n], rtol=1e-12, atol=0.0):
                raise ValueError('Basis function normalization does not factorize over the primitives of its shell.')

    (pair_sh, pair_pp0, pair_ppn, pair_ab,
     pp_p, pp_x, pp_y, pp_z, pp_c) = _shell_pairs(sh_l, sh_nprim, sh_exp, sh_coef, sh_cen, EXP_CUTOFF)
    cart_xyz, cart_off, idx3, dec, dirn = _cart_tables(2 * lmax)
    bfs_lmn = np.ascontiguousarray(bfs_lmn, dtype=np.int64)
    boys = _boys_table(4 * lmax) if scheme == 'os' else np.zeros((1, 1))
    tables = (sh_l, sh_off, sh_nbf, sh_cen, bfs_lmn, comp_scale, cart_xyz, cart_off, idx3, _binomials(lmax),
              dec, dirn, boys, lmax, SCHEMES[scheme])

    # Schwarz factor of every primitive pair (each evaluated as a pair of its own), then the
    # primitive pairs of every shell pair in decreasing order of it, so that the kernel can stop
    # the primitive loops at the first negligible product.
    npp = pp_p.shape[0]
    owner = np.repeat(np.arange(pair_sh.shape[0]), pair_ppn)
    pp_q = _pair_schwarz(np.ascontiguousarray(pair_sh[owner]), np.arange(npp, dtype=np.int64),
                         np.ones(npp, dtype=np.int64), np.ascontiguousarray(pair_ab[owner]),
                         pp_p, pp_x, pp_y, pp_z, pp_c, np.ones(npp), *tables, _nchunks(npp))
    perm = np.lexsort((-pp_q, owner))
    pp_p, pp_x, pp_y, pp_z, pp_c, pp_q = (np.ascontiguousarray(a[perm]) for a in (pp_p, pp_x, pp_y, pp_z, pp_c, pp_q))
    Q = _pair_schwarz(pair_sh, pair_pp0, pair_ppn, pair_ab, pp_p, pp_x, pp_y, pp_z, pp_c, pp_q, *tables,
                      _nchunks(pair_sh.shape[0]))

    order = np.argsort(-Q, kind='stable')
    keep = order[Q[order] * Q.max() >= threshold * 1e-3] if Q.size else order

    plan = Plan4c2e()
    plan.nao = int(basis.bfs_nao)
    plan.nshells = int(nsh)
    plan.lmax = lmax
    plan.scheme = scheme
    plan.threshold = float(threshold)
    plan.npairs_total = int(pair_sh.shape[0])
    plan.Q = np.ascontiguousarray(Q[keep])
    plan.pair_sh = np.ascontiguousarray(pair_sh[keep])
    plan.pair_pp0 = np.ascontiguousarray(pair_pp0[keep])
    plan.pair_ppn = np.ascontiguousarray(pair_ppn[keep])
    plan.pair_ab = np.ascontiguousarray(pair_ab[keep])
    plan.npairs = int(plan.Q.shape[0])
    plan.pp = (pp_p, pp_x, pp_y, pp_z, pp_c, pp_q)
    plan.tables = tables
    plan.sh_l = sh_l
    plan.sh_off = sh_off
    plan.sh_nbf = sh_nbf
    return plan


def _kernel_args(plan):
    return (plan.pair_sh, plan.pair_pp0, plan.pair_ppn, plan.pair_ab, *plan.pp, *plan.tables)


def _bins(plan, key, costs):
    nthreads = int(numba.get_num_threads())
    if (key, nthreads) not in plan._bins:
        plan._bins[(key, nthreads)] = _lpt_bins(np.arange(plan.npairs), costs, nthreads)
    return plan._bins[(key, nthreads)]


def _direct_bins(plan):
    if 'direct_cost' not in plan.__dict__:
        # Density-independent estimate: the ket range of a unit density.
        plan.direct_cost, _ = _bra_costs(plan.Q, plan.threshold / 8.0, plan.pair_sh, plan.pair_ppn,
                                         plan.sh_l, plan.sh_nbf)
    return _bins(plan, 'direct', plan.direct_cost)


def _density_bounds(plan, dmat):
    dsh = _shell_dmax(dmat, plan.sh_off, plan.sh_nbf)
    return dsh, float(dsh.max()) if dsh.size else 0.0


def direct_jk(plan, dmat, with_k=True, threshold=None):
    """
    Coulomb and (``with_k``) exchange matrices of a symmetric density matrix ``dmat``
    (Cartesian AO basis), ``J_ab = sum_cd (ab|cd) D_cd`` and ``K_ac = sum_bd (ab|cd) D_bd``,
    with integrals evaluated on the fly. Returns ``(J, K)``, or ``J`` when ``with_k`` is False.
    """
    dmat = np.ascontiguousarray(dmat, dtype=np.float64)
    thr = plan.threshold if threshold is None else float(threshold)
    bin_off, bin_items = _direct_bins(plan)
    dsh, dmax = _density_bounds(plan, dmat)
    H, G = _direct_pass(bin_off, bin_items, plan.Q, *_kernel_args(plan), dmat, dsh, dmax, thr, bool(with_k))
    J, K = _reduce(H, G, bool(with_k))
    return (J, K) if with_k else J


def store_size_gb(plan, dense=False, threshold_store=None):
    """Memory in GB that :func:`store_integrals` needs for the same arguments."""
    thr_store = plan.threshold if threshold_store is None else float(threshold_store)
    _, jend = _bra_costs(plan.Q, 0.0 if dense else thr_store, plan.pair_sh, plan.pair_ppn, plan.sh_l, plan.sh_nbf)
    return 8e-9 * float(_row_sizes(jend, plan.pair_sh, plan.sh_nbf).sum())


def store_integrals(plan, dense=False, threshold_store=None):
    """
    Evaluate and keep the shell-quartet blocks: every unique block, evaluated without primitive
    screening (``dense=True``), or the blocks with ``Q_XY Q_ZW >= threshold_store`` (default: the
    plan threshold). Returns the size in GB.
    """
    npair = plan.npairs
    thr_store = plan.threshold if threshold_store is None else float(threshold_store)
    build_cost, jend = _bra_costs(plan.Q, 0.0 if dense else thr_store, plan.pair_sh, plan.pair_ppn,
                                  plan.sh_l, plan.sh_nbf)
    sizes = _row_sizes(jend, plan.pair_sh, plan.sh_nbf)
    row_off = np.zeros(npair, dtype=np.int64)
    if npair > 1:
        row_off[1:] = np.cumsum(sizes[:-1])
    plan.values = None
    values = np.empty(int(sizes.sum()), dtype=np.float64)
    bin_off, bin_items = _lpt_bins(np.arange(npair), build_cost, int(numba.get_num_threads()))
    # The complete store skips no primitive quartets, so that it holds every integral in full.
    _store_pass(bin_off, bin_items, jend, row_off, 0.0 if dense else PRIM_FACTOR * thr_store, *_kernel_args(plan), values)
    plan.values = values
    plan.jend = jend
    plan.row_off = row_off
    plan.row_sizes = sizes
    plan.store_dense = bool(dense)
    for key in [k for k in plan._bins if k[0] == 'stored']:
        del plan._bins[key]
    return plan.store_gb


def stored_jk(plan, dmat, with_k=True, threshold=None):
    """Like :func:`direct_jk`, from the blocks kept by :func:`store_integrals`."""
    if plan.values is None:
        raise RuntimeError('store_integrals(plan) has to be called first.')
    dmat = np.ascontiguousarray(dmat, dtype=np.float64)
    thr = plan.threshold if threshold is None else float(threshold)
    bin_off, bin_items = _bins(plan, 'stored', plan.row_sizes.astype(np.float64) + 1.0)
    dsh, dmax = _density_bounds(plan, dmat)
    H, G = _stored_pass(bin_off, bin_items, plan.jend, plan.row_off, plan.Q, plan.pair_sh, plan.sh_off, plan.sh_nbf,
                        plan.values, dmat, dsh, dmax, thr, bool(with_k))
    J, K = _reduce(H, G, bool(with_k))
    return (J, K) if with_k else J
