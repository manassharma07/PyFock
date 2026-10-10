"""
Nuclear gradient of the two-electron energy without density fitting, from derivatives of the
shell-quartet integrals of :mod:`~pyfock.Integrals.jk_4c2e` (Rys quadrature).

For a closed-shell density ``D`` with a fraction ``a`` of exact exchange (1 for HF, 0 for a pure
functional) the two-electron energy is

    E_2 = 1/2 sum D_ab D_cd (ab|cd) - a/4 sum D_ac D_bd (ab|cd)

and its explicit nuclear derivative (the orbital response is the energy-weighted density term of
the gradient) is ``sum_abcd Gamma_abcd d(ab|cd)/dR``.  Over the unique shell quartets ``(XY|ZW)`` of
:mod:`~pyfock.Integrals.jk_4c2e` (bra pair ``i``, ket pair ``j >= i``) every element enters with the
weight

    w_abcd = s (4 D_ab D_cd - a (D_ac D_bd + D_ad D_bc)),

``s`` the degeneracy factor (1/2 each for ``X == Y``, ``Z == W`` and ``i == j``).  The derivative with
respect to the centre of a function raises or lowers its angular momentum,

    d(ab|cd)/dX_k = 2 alpha (a + 1_k, b|cd) - a_k (a - 1_k, b|cd),

with ``alpha`` the exponent of the primitive on ``X``; the centres ``X``, ``Y`` and ``Z`` are
differentiated this way and ``W`` follows from translational invariance.

Per shell quartet the contracted ``(e0|f0)`` (``l_X - 1 <= |e| <= l_X + l_Y + 1``,
``l_Z - 1 <= |f| <= l_Z + l_W + 1``) is accumulated over the primitive quartets four times: plain and
weighted by ``2 alpha``, ``2 beta`` and ``2 gamma`` (the exponents of the primitives on ``X``, ``Y``
and ``Z``).  The weights ``w`` are folded into the ket transfer (for the ``X`` and ``Y`` derivatives)
or the bra transfer (for ``Z``) of these arrays, so that the shifted horizontal transfers reduce to
sums of a few products per element; no derivative integral is stored.

Screening is that of the energy: a quartet is skipped when
``Q_XY Q_ZW max(4 D_XY D_ZW, a (D_XZ D_YW + D_XW D_YZ))`` (shell-block maxima of ``|D|``) is below the
threshold, and its primitive quartets below ``PRIM_FACTOR`` times the threshold over that density
factor.  Quartets on a single atom have no net force and are skipped, and of a pair on one atom only
the sum of its two derivatives is needed, which translational invariance gives: with ``Z`` and ``W``
on one atom the ``Z`` derivative is not evaluated, and with ``X`` and ``Y`` on one atom (and ``Z``,
``W`` on two) the quartet is differentiated as ``(ZW|XY)``.  The bra pairs are distributed
over the threads in balanced bins, each accumulating its own forces (no thread ids: every kernel is
cached on disk).
"""

import math

import numba
import numpy as np
from numba import njit, prange

from .df_algo11_helpers import _lpt_bins
from .jk_4c2e import (MAX_ROOTS, MAXK, PRIM_FACTOR, TWO_PI_52, _bra_costs, _binomials, _cart_tables,
                      _shell_dmax)
from .rys_helpers import Roots

__all__ = ['grad_4c2e']


@njit(cache=True, nogil=True)
def _grad_work_arrays(lmax):
    """Scratch arrays of one thread (every entry is written before it is read)."""
    lt = 2 * lmax + 1
    ncart = (lmax + 1) * (lmax + 2) // 2
    ne = (lt + 1) * (lt + 2) * (lt + 3) // 6
    roots = np.zeros(MAX_ROOTS, dtype=np.float64)
    weights = np.zeros(MAX_ROOTS, dtype=np.float64)
    ln = np.empty((13, MAXK), dtype=np.float64)
    g2 = np.empty((3, (lt + 1) * (lt + 1) * MAXK), dtype=np.float64)
    E4 = np.empty((4, ne * ne), dtype=np.float64)
    F3 = np.empty((3, ne * ncart * ncart), dtype=np.float64)
    B2 = np.empty((2, ncart * ncart * ne), dtype=np.float64)
    dots = np.empty((3, ne), dtype=np.float64)
    gam = np.empty(ncart ** 4, dtype=np.float64)
    pw = np.ones((2, 3, lmax + 2), dtype=np.float64)
    g9 = np.zeros((3, 3), dtype=np.float64)
    tk = np.empty(MAXK, dtype=np.float64)
    return roots, weights, ln, g2, E4, F3, B2, dots, gam, pw, g9, tk


@njit(cache=True, fastmath=True, nogil=True, error_model='numpy', boundscheck=False, inline='always')
def _chunk_rys_grad(K, lbra, lket, ln, g2, E4, e0, ne, f0, nf, cart_xyz, lx, lz, needz, tk):
    """
    Rys recurrences of ``K`` lanes up to bra order ``lbra`` and ket order ``lket`` (one more than
    the quartet's ``l_X + l_Y`` and, with ``needz``, ``l_Z + l_W``); adds their (e0|f0) to ``E4[0]``
    and, weighted by the lane weights ``ln[10:13]``, to ``E4[1:4]``, each for the ``|e|, |f|`` that
    the derivatives read: ``E_alpha`` for ``|e| > l_X``, ``E_beta`` for ``|e| >= l_X`` (both with the
    quartet's ``|f|``), ``E_gamma`` for ``|f| > l_Z`` with the quartet's ``|e|``, and the plain
    integrals one below on either side.
    """
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
    wa = ln[10]
    wb = ln[11]
    wc = ln[12]
    gx = g2[0]
    gy = g2[1]
    gz = g2[2]
    E0 = E4[0]
    Ea = E4[1]
    Eb = E4[2]
    Ec = E4[3]
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
    lbra_q = lbra - 1                  # l_X + l_Y and l_Z + l_W of the quartet
    lket_q = lket - 1 if needz else lket
    for ie in range(ne):
        ex = cart_xyz[e0 + ie, 0]
        ey = cart_xyz[e0 + ie, 1]
        ez = cart_xyz[e0 + ie, 2]
        le = ex + ey + ez
        for jf in range(nf):
            fx = cart_xyz[f0 + jf, 0]
            fy = cart_xyz[f0 + jf, 1]
            fz = cart_xyz[f0 + jf, 2]
            lf = fx + fy + fz
            ket_q = lf >= lz and lf <= lket_q
            na = ket_q and le > lx
            nb = ket_q and le >= lx
            nc = needz and lf > lz and le >= lx and le <= lbra_q
            n0 = ((ket_q and le < lbra_q) or (needz and lf < lket_q and le >= lx and le <= lbra_q))
            if not (nb or nc or n0):
                continue
            ox = (ex * lk1 + fx) * S
            oy = (ey * lk1 + fy) * S
            oz = (ez * lk1 + fz) * S
            for k in range(K):
                tk[k] = gx[ox + k] * gy[oy + k] * gz[oz + k]
            c = ie * nf + jf
            if n0:
                acc = 0.0
                for k in range(K):
                    acc += tk[k]
                E0[c] += acc
            if na:
                acc = 0.0
                for k in range(K):
                    acc += wa[k] * tk[k]
                Ea[c] += acc
            if nb:
                acc = 0.0
                for k in range(K):
                    acc += wb[k] * tk[k]
                Eb[c] += acc
            if nc:
                acc = 0.0
                for k in range(K):
                    acc += wc[k] * tk[k]
                Ec[c] += acc


@njit(cache=True, fastmath=True, nogil=True, error_model='numpy', boundscheck=False, inline='always')
def _e0f0_rys_grad(a0, a1, b0, b1, thr_prim, pp_p, pp_x, pp_y, pp_z, pp_c, pp_q, pp_a, centres,
                   lbra, lket, e0, ne, f0, nf, cart_xyz, roots, weights, ln, g2, E4, lx, lz, needz, tk):
    """
    Adds the plain and exponent-weighted (e0|f0), ``|e| <= lbra``, ``|f| <= lket`` (one more than
    the quartet on the bra side and, with ``needz``, on the ket side), of all screened primitive
    quartets to ``E4``.  No element with both sides raised is needed, so the quadrature is that of
    ``lbra + lket - 1`` (``lbra + lket`` without ``needz``).  ``centres`` is the tuple
    ``(X_x, X_y, X_z, Z_x, Z_y, Z_z)``: an inlined call with more than 30 arguments is a ``*args``
    call on Python 3.10, which Numba cannot inline.
    """
    xx, xy, xz, zx, zy, zz = centres
    nroots = (lbra + lket - (1 if needz else 0)) // 2 + 1
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
        two_alpha = 2.0 * pp_a[a]
        two_beta = 2.0 * p - two_alpha
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
            two_gamma = 2.0 * pp_a[b]
            qcx = qx - zx
            qcy = qy - zy
            qcz = qz - zz
            inv_2q = 0.5 / q
            lane = nk * nroots
            for r in range(nroots):
                u = roots[r]
                t2 = u / (1.0 + u)
                fb = q * t2 * inv_pq
                fk = p * t2 * inv_pq
                ln[0, lane] = 0.5 * t2 * inv_pq
                ln[1, lane] = (1.0 - fb) * inv_2p
                ln[2, lane] = (1.0 - fk) * inv_2q
                ln[3, lane] = pax - fb * pqx
                ln[4, lane] = pay - fb * pqy
                ln[5, lane] = paz - fb * pqz
                ln[6, lane] = qcx + fk * pqx
                ln[7, lane] = qcy + fk * pqy
                ln[8, lane] = qcz + fk * pqz
                ln[9, lane] = weights[r] * pref
                ln[10, lane] = two_alpha
                ln[11, lane] = two_beta
                ln[12, lane] = two_gamma
                lane += 1
            nk += 1
            if nk == kmax:
                _chunk_rys_grad(nk * nroots, lbra, lket, ln, g2, E4, e0, ne, f0, nf, cart_xyz, lx, lz, needz, tk)
                nk = 0
    if nk > 0:
        _chunk_rys_grad(nk * nroots, lbra, lket, ln, g2, E4, e0, ne, f0, nf, cart_xyz, lx, lz, needz, tk)


@njit(cache=True, fastmath=True, nogil=True, error_model='numpy', boundscheck=False, inline='always')
def _transfer_sum(dots, e0, idx3, binom, pw, ax, ay, az, bx, by, bz):
    """``sum_q C(b, q) AB^(b-q) dots[(a + q) - e0]``: one element of a horizontal transfer."""
    acc = 0.0
    for qx in range(bx + 1):
        fx = binom[bx, qx] * pw[0, bx - qx]
        for qy in range(by + 1):
            fxy = fx * binom[by, qy] * pw[1, by - qy]
            for qz in range(bz + 1):
                acc += fxy * binom[bz, qz] * pw[2, bz - qz] * dots[idx3[ax + qx, ay + qy, az + qz] - e0]
    return acc


@njit(cache=True, fastmath=True, nogil=True, error_model='numpy', boundscheck=False)
def _quartet_grad(i, j, thr_prim, s, exx, needz, dmat, pair_sh, pair_pp0, pair_ppn, pair_ab,
                  pp_p, pp_x, pp_y, pp_z, pp_c, pp_q, pp_a, sh_l, sh_off, sh_nbf, sh_cen, bfs_lmn, comp_scale,
                  cart_xyz, cart_off, idx3, binom, roots, weights, ln, g2, E4, F3, B2, dots, gam, pw, g9, tk):
    """
    ``g9[c, k] = sum_abcd w_abcd d(ab|cd)/dC_k`` for the centres ``C = X, Y, Z`` of the quartet of
    bra pair ``i`` and ket pair ``j`` (weights ``w`` of the module docstring, degeneracy ``s``).
    Without ``needz`` (``Z`` and ``W`` on one atom, which then only needs ``-(g_X + g_Y)``) the ``Z``
    row is left zero.
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
    x0 = sh_off[X]
    y0 = sh_off[Y]
    z0 = sh_off[Z]
    w0 = sh_off[W]
    lbra = lx + ly
    lket = lz + lw
    zw = nZ * nW
    nxy = nX * nY

    # weights, with the component normalization of the four functions
    c = 0
    for x in range(nX):
        ix = x0 + x
        for y in range(nY):
            iy = y0 + y
            sxy = s * comp_scale[ix] * comp_scale[iy]
            dxy4 = 4.0 * dmat[ix, iy]
            for z in range(nZ):
                iz = z0 + z
                sxyz = sxy * comp_scale[iz]
                dxz = exx * dmat[ix, iz]
                dyz = exx * dmat[iy, iz]
                for w in range(nW):
                    iw = w0 + w
                    gam[c] = sxyz * comp_scale[iw] * (dxy4 * dmat[iz, iw] - dxz * dmat[iy, iw] - dyz * dmat[ix, iw])
                    c += 1

    # (e0|f0), plain and weighted by 2 alpha, 2 beta, 2 gamma
    lket_r = lket + 1 if needz else lket
    e0 = cart_off[max(lx - 1, 0)]
    ne = cart_off[lbra + 2] - e0
    f0 = cart_off[max(lz - 1, 0)] if needz else cart_off[lz]
    nf = cart_off[lket_r + 1] - f0
    for k in range(4):
        for c in range(ne * nf):
            E4[k, c] = 0.0
    _e0f0_rys_grad(pair_pp0[i], pair_pp0[i] + pair_ppn[i], pair_pp0[j], pair_pp0[j] + pair_ppn[j], thr_prim,
                   pp_p, pp_x, pp_y, pp_z, pp_c, pp_q, pp_a,
                   (sh_cen[X, 0], sh_cen[X, 1], sh_cen[X, 2], sh_cen[Z, 0], sh_cen[Z, 1], sh_cen[Z, 2]),
                   lbra + 1, lket_r, e0, ne, f0, nf, cart_xyz, roots, weights, ln, g2, E4, lx, lz, needz, tk)

    # powers of AB = X - Y (pw[0]) and CD = Z - W (pw[1])
    for d in range(3):
        pw[0, d, 0] = 1.0
        pw[1, d, 0] = 1.0
        for k in range(1, ly + 2):
            pw[0, d, k] = pw[0, d, k - 1] * pair_ab[i, d]
        for k in range(1, lw + 2):
            pw[1, d, k] = pw[1, d, k - 1] * pair_ab[j, d]
    for c in range(3):
        for k in range(3):
            g9[c, k] = 0.0

    # ---- X and Y: ket transfer of E0, E_alpha, E_beta to (cd), contracted with the weights ----
    # F3[t, ie * zw + col] = (e0|cd), col = (c, d); then per (a, b) dots[t, ie] = sum_cd w (e0|cd)
    for t in range(3):
        Et = E4[t]
        Ft = F3[t]
        for z in range(nZ):
            cx = bfs_lmn[z0 + z, 0]
            cy = bfs_lmn[z0 + z, 1]
            cz = bfs_lmn[z0 + z, 2]
            for w in range(nW):
                col = z * nW + w
                dx = bfs_lmn[w0 + w, 0]
                dy = bfs_lmn[w0 + w, 1]
                dz = bfs_lmn[w0 + w, 2]
                for ie in range(ne):
                    Ft[ie * zw + col] = 0.0
                for qx in range(dx + 1):
                    fx = binom[dx, qx] * pw[1, 0, dx - qx]
                    for qy in range(dy + 1):
                        fxy = fx * binom[dy, qy] * pw[1, 1, dy - qy]
                        for qz in range(dz + 1):
                            coef = fxy * binom[dz, qz] * pw[1, 2, dz - qz]
                            jf = idx3[cx + qx, cy + qy, cz + qz] - f0
                            for ie in range(ne):
                                Ft[ie * zw + col] += coef * Et[ie * nf + jf]
    for x in range(nX):
        ax = bfs_lmn[x0 + x, 0]
        ay = bfs_lmn[x0 + x, 1]
        az = bfs_lmn[x0 + x, 2]
        for y in range(nY):
            bx = bfs_lmn[y0 + y, 0]
            by = bfs_lmn[y0 + y, 1]
            bz = bfs_lmn[y0 + y, 2]
            o = (x * nY + y) * zw
            for t in range(3):
                Ft = F3[t]
                for ie in range(ne):
                    acc = 0.0
                    for col in range(zw):
                        acc += gam[o + col] * Ft[ie * zw + col]
                    dots[t, ie] = acc
            # d/dX_k: 2 alpha (a + 1_k, b| - a_k (a - 1_k, b|
            g9[0, 0] += _transfer_sum(dots[1], e0, idx3, binom, pw[0], ax + 1, ay, az, bx, by, bz)
            g9[0, 1] += _transfer_sum(dots[1], e0, idx3, binom, pw[0], ax, ay + 1, az, bx, by, bz)
            g9[0, 2] += _transfer_sum(dots[1], e0, idx3, binom, pw[0], ax, ay, az + 1, bx, by, bz)
            if ax > 0:
                g9[0, 0] -= ax * _transfer_sum(dots[0], e0, idx3, binom, pw[0], ax - 1, ay, az, bx, by, bz)
            if ay > 0:
                g9[0, 1] -= ay * _transfer_sum(dots[0], e0, idx3, binom, pw[0], ax, ay - 1, az, bx, by, bz)
            if az > 0:
                g9[0, 2] -= az * _transfer_sum(dots[0], e0, idx3, binom, pw[0], ax, ay, az - 1, bx, by, bz)
            # d/dY_k: 2 beta (a, b + 1_k| - b_k (a, b - 1_k|
            g9[1, 0] += _transfer_sum(dots[2], e0, idx3, binom, pw[0], ax, ay, az, bx + 1, by, bz)
            g9[1, 1] += _transfer_sum(dots[2], e0, idx3, binom, pw[0], ax, ay, az, bx, by + 1, bz)
            g9[1, 2] += _transfer_sum(dots[2], e0, idx3, binom, pw[0], ax, ay, az, bx, by, bz + 1)
            if bx > 0:
                g9[1, 0] -= bx * _transfer_sum(dots[0], e0, idx3, binom, pw[0], ax, ay, az, bx - 1, by, bz)
            if by > 0:
                g9[1, 1] -= by * _transfer_sum(dots[0], e0, idx3, binom, pw[0], ax, ay, az, bx, by - 1, bz)
            if bz > 0:
                g9[1, 2] -= bz * _transfer_sum(dots[0], e0, idx3, binom, pw[0], ax, ay, az, bx, by, bz - 1)

    if not needz:
        return
    # ---- Z: bra transfer of E0 and E_gamma to (ab), contracted with the weights ----
    # B2[t, xy * nf + jf] = (ab|f0); then per (c, d) dots[t, jf] = sum_ab w (ab|f0)
    for t in range(2):
        Et = E4[3 * t]          # E0 and E_gamma
        Bt = B2[t]
        for x in range(nX):
            ax = bfs_lmn[x0 + x, 0]
            ay = bfs_lmn[x0 + x, 1]
            az = bfs_lmn[x0 + x, 2]
            for y in range(nY):
                bx = bfs_lmn[y0 + y, 0]
                by = bfs_lmn[y0 + y, 1]
                bz = bfs_lmn[y0 + y, 2]
                o = (x * nY + y) * nf
                for jf in range(nf):
                    Bt[o + jf] = 0.0
                for qx in range(bx + 1):
                    fx = binom[bx, qx] * pw[0, 0, bx - qx]
                    for qy in range(by + 1):
                        fxy = fx * binom[by, qy] * pw[0, 1, by - qy]
                        for qz in range(bz + 1):
                            coef = fxy * binom[bz, qz] * pw[0, 2, bz - qz]
                            ie = idx3[ax + qx, ay + qy, az + qz] - e0
                            for jf in range(nf):
                                Bt[o + jf] += coef * Et[ie * nf + jf]
    for z in range(nZ):
        cx = bfs_lmn[z0 + z, 0]
        cy = bfs_lmn[z0 + z, 1]
        cz = bfs_lmn[z0 + z, 2]
        for w in range(nW):
            dx = bfs_lmn[w0 + w, 0]
            dy = bfs_lmn[w0 + w, 1]
            dz = bfs_lmn[w0 + w, 2]
            col = z * nW + w
            for t in range(2):
                Bt = B2[t]
                for jf in range(nf):
                    acc = 0.0
                    for xy in range(nxy):
                        acc += gam[xy * zw + col] * Bt[xy * nf + jf]
                    dots[t, jf] = acc
            # d/dZ_k: 2 gamma (ab|c + 1_k, d) - c_k (ab|c - 1_k, d)
            g9[2, 0] += _transfer_sum(dots[1], f0, idx3, binom, pw[1], cx + 1, cy, cz, dx, dy, dz)
            g9[2, 1] += _transfer_sum(dots[1], f0, idx3, binom, pw[1], cx, cy + 1, cz, dx, dy, dz)
            g9[2, 2] += _transfer_sum(dots[1], f0, idx3, binom, pw[1], cx, cy, cz + 1, dx, dy, dz)
            if cx > 0:
                g9[2, 0] -= cx * _transfer_sum(dots[0], f0, idx3, binom, pw[1], cx - 1, cy, cz, dx, dy, dz)
            if cy > 0:
                g9[2, 1] -= cy * _transfer_sum(dots[0], f0, idx3, binom, pw[1], cx, cy - 1, cz, dx, dy, dz)
            if cz > 0:
                g9[2, 2] -= cz * _transfer_sum(dots[0], f0, idx3, binom, pw[1], cx, cy, cz - 1, dx, dy, dz)


@njit(parallel=True, cache=True, fastmath=True, nogil=True, error_model='numpy', boundscheck=False)
def _grad_pass(bin_off, bin_items, Q, pair_sh, pair_pp0, pair_ppn, pair_ab, pp_p, pp_x, pp_y, pp_z, pp_c, pp_q,
               pp_a, sh_l, sh_off, sh_nbf, sh_cen, bfs_lmn, comp_scale, sh_atom, natoms,
               cart_xyz, cart_off, idx3, binom, lmax, dmat, dsh, dmax, exx, threshold):
    nbins = bin_off.shape[0] - 1
    npair = Q.shape[0]
    gbin = np.zeros((nbins, natoms, 3), dtype=np.float64)
    dlim = 4.0 * dmax * dmax
    for bn in prange(nbins):
        roots, weights, ln, g2, E4, F3, B2, dots, gam, pw, g9, tk = _grad_work_arrays(lmax)
        gb = gbin[bn]
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
                dm = max(dxy4 * dsh[Z, W], exx * (dsh[X, Z] * dsh[Y, W] + dsh[X, W] * dsh[Y, Z]))
                if qq * dm < threshold:
                    continue
                aX = sh_atom[X]
                aY = sh_atom[Y]
                aZ = sh_atom[Z]
                aW = sh_atom[W]
                if aX == aY and aX == aZ and aX == aW:
                    continue                    # one-centre quartet: no net force
                s = 1.0
                if X == Y:
                    s *= 0.5
                if Z == W:
                    s *= 0.5
                if i == j:
                    s *= 0.5
                if aX == aY and aZ != aW:
                    # the bra pair on one atom only needs -(g_Z + g_W): differentiate the ket centres,
                    # i.e. the quartet (ZW|XY) without its third centre
                    _quartet_grad(j, i, PRIM_FACTOR * threshold / dm, s, exx, False, dmat, pair_sh, pair_pp0,
                                  pair_ppn, pair_ab, pp_p, pp_x, pp_y, pp_z, pp_c, pp_q, pp_a, sh_l, sh_off,
                                  sh_nbf, sh_cen, bfs_lmn, comp_scale, cart_xyz, cart_off, idx3, binom,
                                  roots, weights, ln, g2, E4, F3, B2, dots, gam, pw, g9, tk)
                    for k in range(3):
                        gb[aZ, k] += g9[0, k]
                        gb[aW, k] += g9[1, k]
                        gb[aX, k] -= g9[0, k] + g9[1, k]
                else:
                    _quartet_grad(i, j, PRIM_FACTOR * threshold / dm, s, exx, aZ != aW, dmat, pair_sh, pair_pp0,
                                  pair_ppn, pair_ab, pp_p, pp_x, pp_y, pp_z, pp_c, pp_q, pp_a, sh_l, sh_off,
                                  sh_nbf, sh_cen, bfs_lmn, comp_scale, cart_xyz, cart_off, idx3, binom,
                                  roots, weights, ln, g2, E4, F3, B2, dots, gam, pw, g9, tk)
                    for k in range(3):
                        gb[aX, k] += g9[0, k]
                        gb[aY, k] += g9[1, k]
                        gb[aZ, k] += g9[2, k]
                        gb[aW, k] -= g9[0, k] + g9[1, k] + g9[2, k]
    grad = np.zeros((natoms, 3), dtype=np.float64)
    for bn in range(nbins):
        for a in range(natoms):
            for k in range(3):
                grad[a, k] += gbin[bn, a, k]
    return grad


def grad_4c2e(plan, dmat, exx_coef=0.0, threshold=None):
    """
    ``sum_abcd Gamma_abcd d(ab|cd)/dR`` for the two-electron energy
    ``1/2 sum D_ab D_cd (ab|cd) - exx_coef/4 sum D_ac D_bd (ab|cd)`` of a symmetric density matrix
    ``dmat`` (Cartesian AO basis), with the shell pairs and screening of a
    :func:`~pyfock.Integrals.jk_4c2e.build_plan` plan (either scheme: the derivatives always use
    Rys quadrature).  ``threshold`` defaults to the plan's.

    Returns the ``(natoms, 3)`` gradient term in Hartree/Bohr.
    """
    dmat = np.ascontiguousarray(dmat, dtype=np.float64)
    thr = plan.threshold if threshold is None else float(threshold)
    lmax = int(plan.lmax)
    if 2 * lmax + 1 > MAX_ROOTS:
        raise ValueError(f'Four-center derivative integrals support shells up to l = {(MAX_ROOTS - 1) // 2}.')
    sh_l, sh_off, sh_nbf, sh_cen, bfs_lmn, comp_scale = plan.tables[:6]
    cart_xyz, cart_off, idx3, _, _ = _cart_tables(2 * lmax + 1)
    binom = _binomials(lmax + 1)
    sh_atom = np.ascontiguousarray(plan.bfs_atoms[sh_off])
    natoms = int(plan.bfs_atoms.max()) + 1
    dsh = _shell_dmax(dmat, sh_off, sh_nbf)
    dmax = float(dsh.max()) if dsh.size else 0.0
    nthreads = int(numba.get_num_threads())
    key = ('grad', nthreads)
    if key not in plan._bins:
        cost, _ = _bra_costs(plan.Q, thr / 16.0, plan.pair_sh, plan.pair_ppn, sh_l, sh_nbf)
        plan._bins[key] = _lpt_bins(np.arange(plan.npairs), 4.0 * cost + 1.0, nthreads)
    bin_off, bin_items = plan._bins[key]
    return _grad_pass(bin_off, bin_items, plan.Q, plan.pair_sh, plan.pair_pp0, plan.pair_ppn, plan.pair_ab,
                      *plan.pp, plan.pp_a, sh_l, sh_off, sh_nbf, sh_cen, bfs_lmn, comp_scale, sh_atom, natoms,
                      cart_xyz, cart_off, idx3, binom, lmax, dmat, dsh, dmax, float(exx_coef), thr)
