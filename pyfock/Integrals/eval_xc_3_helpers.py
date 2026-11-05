"""
Numba kernels of the CPU exchange-correlation algorithm 3 (:mod:`pyfock.Integrals.eval_xc_3`,
which documents the algorithm): basis functions on a leaf, grid partitioning and screening,
the AO cache builder, the one-region XC kernel and the assembly of the potential matrix.

They live in their own module so that Numba's on-disk cache of the compiled kernels survives
edits of the driver.
"""
import math

import numpy as np
from numba import njit, prange

from pyfock.XC import (lda_x, lda_c_vwn, lda_c_vwn_rpa, lda_c_pz, lda_c_pz_mod, lda_c_pw, lda_c_pw_mod,
                       gga_x_pbe, gga_c_pbe, gga_x_pbe_sol, gga_c_pbe_sol, gga_x_rpbe, gga_x_pw91,
                       gga_c_p86, gga_x_b88, gga_c_lyp)
from pyfock.XC.pointwise import eval_pointwise_mgga, eval_pointwise_gga, POINTWISE_IDS

_NB = dict(cache=True, nogil=True, fastmath=True, error_model='numpy')

# Functionals evaluated inside the Numba kernel (one pass over the grid): every natively
# implemented functional (LDA, GGA, meta-GGA; hybrids through their components). LibXC
# functionals go through the two-pass path.
INKERNEL_FUNCTIONALS = frozenset((1, 7, 8, 9, 10, 12, 13, 101, 106, 109, 116, 117, 130, 131, 132, 133)) | POINTWISE_IDS

LMAX = 8          # highest angular momentum of the power tables (per-shell polynomial x^l y^m z^n)
EXP_CUTOFF = 46.0  # exp(-46) ~ 1e-20: a primitive beyond this is dropped at that point


# ---------------------------------------------------------------------------------------
# Functional evaluation inside the kernel
# ---------------------------------------------------------------------------------------
# The native functionals are compiled with inline='always'; these thin wrappers keep each one
# as a separately compiled (and cached) unit so that the dispatcher below stays small.
@njit(**_NB)
def _f_1(rho):
    return lda_x(rho)


@njit(**_NB)
def _f_7(rho):
    return lda_c_vwn(rho)


@njit(**_NB)
def _f_8(rho):
    return lda_c_vwn_rpa(rho)


@njit(**_NB)
def _f_9(rho):
    return lda_c_pz(rho)


@njit(**_NB)
def _f_10(rho):
    return lda_c_pz_mod(rho)


@njit(**_NB)
def _f_12(rho):
    return lda_c_pw(rho)


@njit(**_NB)
def _f_13(rho):
    return lda_c_pw_mod(rho)


@njit(**_NB)
def _f_101(rho, sigma):
    return gga_x_pbe(rho, sigma)


@njit(**_NB)
def _f_106(rho, sigma):
    return gga_x_b88(rho, sigma)


@njit(**_NB)
def _f_109(rho, sigma):
    return gga_x_pw91(rho, sigma)


@njit(**_NB)
def _f_116(rho, sigma):
    return gga_x_pbe_sol(rho, sigma)


@njit(**_NB)
def _f_117(rho, sigma):
    return gga_x_rpbe(rho, sigma)


@njit(**_NB)
def _f_130(rho, sigma):
    return gga_c_pbe(rho, sigma)


@njit(**_NB)
def _f_131(rho, sigma):
    return gga_c_lyp(rho, sigma)


@njit(**_NB)
def _f_132(rho, sigma):
    return gga_c_p86(rho, sigma)


@njit(**_NB)
def _f_133(rho, sigma):
    return gga_c_pbe_sol(rho, sigma)


@njit(**_NB)
def _acc2(e, vrho, a, b, c):
    for i in range(e.shape[0]):
        e[i] += c * a[i]
        vrho[i] += c * b[i]


@njit(**_NB)
def _acc3(e, vrho, vsigma, a, b, d, c):
    for i in range(e.shape[0]):
        e[i] += c * a[i]
        vrho[i] += c * b[i]
        vsigma[i] += c * d[i]


@njit(**_NB)
def _xc_native_eval(fids, coefs, rho, sigma, tau, e, vrho, vsigma, vtau):
    """Energy per particle and the derivatives of the coefficient-weighted sum of the
    functionals ``fids`` at the points of one leaf."""
    e[:] = 0.0
    vrho[:] = 0.0
    vsigma[:] = 0.0
    vtau[:] = 0.0
    for f in range(fids.shape[0]):
        fid = fids[f]
        c = coefs[f]
        if fid >= 200:
            eval_pointwise_mgga(fid, c, rho, sigma, tau, e, vrho, vsigma, vtau)
        elif fid == 134:
            eval_pointwise_gga(fid, c, rho, sigma, e, vrho, vsigma)
        elif fid == 1:
            a, b = _f_1(rho)
            _acc2(e, vrho, a, b, c)
        elif fid == 7:
            a, b = _f_7(rho)
            _acc2(e, vrho, a, b, c)
        elif fid == 8:
            a, b = _f_8(rho)
            _acc2(e, vrho, a, b, c)
        elif fid == 9:
            a, b = _f_9(rho)
            _acc2(e, vrho, a, b, c)
        elif fid == 10:
            a, b = _f_10(rho)
            _acc2(e, vrho, a, b, c)
        elif fid == 12:
            a, b = _f_12(rho)
            _acc2(e, vrho, a, b, c)
        elif fid == 13:
            a, b = _f_13(rho)
            _acc2(e, vrho, a, b, c)
        elif fid == 101:
            a, b, d = _f_101(rho, sigma)
            _acc3(e, vrho, vsigma, a, b, d, c)
        elif fid == 106:
            a, b, d = _f_106(rho, sigma)
            _acc3(e, vrho, vsigma, a, b, d, c)
        elif fid == 109:
            a, b, d = _f_109(rho, sigma)
            _acc3(e, vrho, vsigma, a, b, d, c)
        elif fid == 116:
            a, b, d = _f_116(rho, sigma)
            _acc3(e, vrho, vsigma, a, b, d, c)
        elif fid == 117:
            a, b, d = _f_117(rho, sigma)
            _acc3(e, vrho, vsigma, a, b, d, c)
        elif fid == 130:
            a, b, d = _f_130(rho, sigma)
            _acc3(e, vrho, vsigma, a, b, d, c)
        elif fid == 131:
            a, b, d = _f_131(rho, sigma)
            _acc3(e, vrho, vsigma, a, b, d, c)
        elif fid == 132:
            a, b, d = _f_132(rho, sigma)
            _acc3(e, vrho, vsigma, a, b, d, c)
        elif fid == 133:
            a, b, d = _f_133(rho, sigma)
            _acc3(e, vrho, vsigma, a, b, d, c)


# ---------------------------------------------------------------------------------------
# Basis functions on the points of one leaf
# ---------------------------------------------------------------------------------------
@njit(**_NB)
def _eval_ao_block(coords, g0, k, leaf_sh, q0, nshb, sh_coords, sh_rcut2, sh_nprim, sh_expnts,
                   sh_bf0, sh_nbf, sh_L, bf_lmn, bf_coef, nd, out):
    """Values (``nd`` = 1), values and gradients (4) or values, gradients and second
    derivatives (10) of the basis functions of the shells ``leaf_sh[q0:q0+nshb]`` at the
    points ``coords[g0:g0+k]``, into ``out[nd, k, n]`` (columns in shell-list order).

    The exponentials of a shell are shared by its Cartesian components; a primitive whose
    exponent times the squared distance exceeds ``EXP_CUTOFF`` is dropped at that point, and
    a shell beyond its radial cutoff is zero. Derivative components follow the order
    x, y, z and xx, xy, xz, yy, yz, zz.
    """
    maxprim = sh_expnts.shape[1]
    ev = np.empty(maxprim)
    px = np.empty(LMAX + 1)
    py = np.empty(LMAX + 1)
    pz = np.empty(LMAX + 1)
    for kk in range(k):
        g = g0 + kk
        x0 = coords[g, 0]
        y0 = coords[g, 1]
        z0 = coords[g, 2]
        col = 0
        for q in range(q0, q0 + nshb):
            s = leaf_sh[q]
            nb = sh_nbf[s]
            dx = x0 - sh_coords[s, 0]
            dy = y0 - sh_coords[s, 1]
            dz = z0 - sh_coords[s, 2]
            r2 = dx * dx + dy * dy + dz * dz
            if r2 > sh_rcut2[s]:
                for d in range(nd):
                    for c in range(nb):
                        out[d, kk, col + c] = 0.0
                col += nb
                continue
            npr = sh_nprim[s]
            for p in range(npr):
                t = sh_expnts[s, p] * r2
                if t < EXP_CUTOFF:
                    ev[p] = math.exp(-t)
                else:
                    ev[p] = 0.0
            L = sh_L[s]
            px[0] = 1.0
            py[0] = 1.0
            pz[0] = 1.0
            for l in range(1, L + 1):
                px[l] = px[l - 1] * dx
                py[l] = py[l - 1] * dy
                pz[l] = pz[l - 1] * dz
            bf0 = sh_bf0[s]
            for c in range(nb):
                i = bf0 + c
                l = bf_lmn[i, 0]
                m = bf_lmn[i, 1]
                n = bf_lmn[i, 2]
                R = 0.0
                R2 = 0.0
                R3 = 0.0
                for p in range(npr):
                    cp = bf_coef[i, p] * ev[p]
                    a = sh_expnts[s, p]
                    R += cp
                    R2 += a * cp
                    R3 += a * a * cp
                P = px[l] * py[m] * pz[n]
                out[0, kk, col + c] = P * R
                if nd >= 4:
                    if l > 0:
                        Px = l * px[l - 1] * py[m] * pz[n]
                    else:
                        Px = 0.0
                    if m > 0:
                        Py = m * px[l] * py[m - 1] * pz[n]
                    else:
                        Py = 0.0
                    if n > 0:
                        Pz = n * px[l] * py[m] * pz[n - 1]
                    else:
                        Pz = 0.0
                    # d/dx [P(x,y,z) g(r^2)] = P_x g + 2 x P g',  g' = -R2 (derivative wrt r^2)
                    out[1, kk, col + c] = Px * R - 2.0 * dx * P * R2
                    out[2, kk, col + c] = Py * R - 2.0 * dy * P * R2
                    out[3, kk, col + c] = Pz * R - 2.0 * dz * P * R2
                    if nd == 10:
                        if l > 1:
                            Pxx = l * (l - 1) * px[l - 2] * py[m] * pz[n]
                        else:
                            Pxx = 0.0
                        if m > 1:
                            Pyy = m * (m - 1) * px[l] * py[m - 2] * pz[n]
                        else:
                            Pyy = 0.0
                        if n > 1:
                            Pzz = n * (n - 1) * px[l] * py[m] * pz[n - 2]
                        else:
                            Pzz = 0.0
                        if l > 0 and m > 0:
                            Pxy = l * m * px[l - 1] * py[m - 1] * pz[n]
                        else:
                            Pxy = 0.0
                        if l > 0 and n > 0:
                            Pxz = l * n * px[l - 1] * py[m] * pz[n - 1]
                        else:
                            Pxz = 0.0
                        if m > 0 and n > 0:
                            Pyz = m * n * px[l] * py[m - 1] * pz[n - 1]
                        else:
                            Pyz = 0.0
                        # d2/dx2 = P_xx g - 4 x P_x R2 + P (-2 R2 + 4 x^2 R3), g'' = R3
                        out[4, kk, col + c] = Pxx * R - 4.0 * dx * Px * R2 + P * (-2.0 * R2 + 4.0 * dx * dx * R3)
                        out[5, kk, col + c] = Pxy * R - 2.0 * (dy * Px + dx * Py) * R2 + 4.0 * dx * dy * P * R3
                        out[6, kk, col + c] = Pxz * R - 2.0 * (dz * Px + dx * Pz) * R2 + 4.0 * dx * dz * P * R3
                        out[7, kk, col + c] = Pyy * R - 4.0 * dy * Py * R2 + P * (-2.0 * R2 + 4.0 * dy * dy * R3)
                        out[8, kk, col + c] = Pyz * R - 2.0 * (dz * Py + dy * Pz) * R2 + 4.0 * dy * dz * P * R3
                        out[9, kk, col + c] = Pzz * R - 4.0 * dz * Pz * R2 + P * (-2.0 * R2 + 4.0 * dz * dz * R3)
            col += nb


# ---------------------------------------------------------------------------------------
# Grid partitioning and screening
# ---------------------------------------------------------------------------------------
@njit(**_NB)
def _kd_order(coords, leaf_max):
    """Permutation that orders the points by recursive median bisection (longest extent
    first) into leaves of at most ``leaf_max`` points, and the leaf ranges ``[lo, hi)`` in
    that order (consecutive leaves are neighbours in space)."""
    n = coords.shape[0]
    perm = np.arange(n)
    max_leaves = n // max(leaf_max // 2, 1) + 64
    leaf_lo = np.empty(max_leaves, np.int64)
    leaf_hi = np.empty(max_leaves, np.int64)
    nleaf = 0
    stack_lo = np.empty(512, np.int64)
    stack_hi = np.empty(512, np.int64)
    stack_lo[0] = 0
    stack_hi[0] = n
    sp = 1
    vals = np.empty(n)
    tmp = np.empty(n, np.int64)
    while sp > 0:
        sp -= 1
        lo = stack_lo[sp]
        hi = stack_hi[sp]
        cnt = hi - lo
        if cnt <= leaf_max:
            leaf_lo[nleaf] = lo
            leaf_hi[nleaf] = hi
            nleaf += 1
            continue
        xmin = np.inf
        xmax = -np.inf
        ymin = np.inf
        ymax = -np.inf
        zmin = np.inf
        zmax = -np.inf
        for i in range(lo, hi):
            p = perm[i]
            x = coords[p, 0]
            y = coords[p, 1]
            z = coords[p, 2]
            if x < xmin:
                xmin = x
            if x > xmax:
                xmax = x
            if y < ymin:
                ymin = y
            if y > ymax:
                ymax = y
            if z < zmin:
                zmin = z
            if z > zmax:
                zmax = z
        axis = 0
        ext = xmax - xmin
        if ymax - ymin > ext:
            axis = 1
            ext = ymax - ymin
        if zmax - zmin > ext:
            axis = 2
        for i in range(cnt):
            vals[i] = coords[perm[lo + i], axis]
        m = cnt // 2
        med = np.partition(vals[:cnt], m)[m]
        # three-way partition: smaller values left, larger right, equal ones fill the gap
        il = 0
        ir = cnt - 1
        for i in range(cnt):
            v = vals[i]
            p = perm[lo + i]
            if v < med:
                tmp[il] = p
                il += 1
            elif v > med:
                tmp[ir] = p
                ir -= 1
        for i in range(cnt):
            if vals[i] == med:
                tmp[il] = perm[lo + i]
                il += 1
        for i in range(cnt):
            perm[lo + i] = tmp[i]
        stack_lo[sp] = lo + m
        stack_hi[sp] = hi
        sp += 1
        stack_lo[sp] = lo
        stack_hi[sp] = lo + m
        sp += 1
    return perm, leaf_lo[:nleaf].copy(), leaf_hi[:nleaf].copy()


@njit(parallel=True, **_NB)
def _screen_leaves_mask(coords, leaf_lo, leaf_hi, sh_coords, sh_rcut, mask):
    """``mask[leaf, shell] = 1`` when a point of the leaf lies within the radial cutoff of
    the shell. A bounding sphere of the leaf settles most pairs; the rest are checked point
    by point (the same criterion as the per-batch screening of algorithm 2)."""
    nleaf = leaf_lo.shape[0]
    nsh = sh_coords.shape[0]
    margin = 1.0e-10
    for il in prange(nleaf):
        lo = leaf_lo[il]
        hi = leaf_hi[il]
        npts = hi - lo
        if npts <= 0:
            continue
        cx = 0.0
        cy = 0.0
        cz = 0.0
        for g in range(lo, hi):
            cx += coords[g, 0]
            cy += coords[g, 1]
            cz += coords[g, 2]
        cx /= npts
        cy /= npts
        cz /= npts
        rad2 = 0.0
        for g in range(lo, hi):
            x = coords[g, 0] - cx
            y = coords[g, 1] - cy
            z = coords[g, 2] - cz
            d2 = x * x + y * y + z * z
            if d2 > rad2:
                rad2 = d2
        rad = math.sqrt(rad2)
        for s in range(nsh):
            bx = sh_coords[s, 0]
            by = sh_coords[s, 1]
            bz = sh_coords[s, 2]
            rc = sh_rcut[s]
            dist = math.sqrt((cx - bx) ** 2 + (cy - by) ** 2 + (cz - bz) ** 2)
            if dist - rad > rc + margin:
                continue
            if dist + rad < rc - margin:
                mask[il, s] = 1
                continue
            for g in range(lo, hi):
                x = coords[g, 0] - bx
                y = coords[g, 1] - by
                z = coords[g, 2] - bz
                if math.sqrt(x * x + y * y + z * z) < rc:
                    mask[il, s] = 1
                    break


@njit(**_NB)
def _mask_to_csr(mask):
    nrow = mask.shape[0]
    ncol = mask.shape[1]
    ptr = np.zeros(nrow + 1, np.int64)
    for r in range(nrow):
        c = 0
        for s in range(ncol):
            if mask[r, s] != 0:
                c += 1
        ptr[r + 1] = ptr[r] + c
    idx = np.empty(ptr[nrow], np.int64)
    for r in range(nrow):
        q = ptr[r]
        for s in range(ncol):
            if mask[r, s] != 0:
                idx[q] = s
                q += 1
    return ptr, idx


@njit(**_NB)
def _expand_shells_to_bf(ptr, idx, sh_bf0, sh_nbf):
    nrow = ptr.shape[0] - 1
    bptr = np.zeros(nrow + 1, np.int64)
    for r in range(nrow):
        c = 0
        for q in range(ptr[r], ptr[r + 1]):
            c += sh_nbf[idx[q]]
        bptr[r + 1] = bptr[r] + c
    bidx = np.empty(bptr[nrow], np.int64)
    for r in range(nrow):
        p = bptr[r]
        for q in range(ptr[r], ptr[r + 1]):
            s = idx[q]
            for c in range(sh_nbf[s]):
                bidx[p] = sh_bf0[s] + c
                p += 1
    return bptr, bidx


@njit(parallel=True, **_NB)
def _build_ao_cache_kernel(coords, leaf_lo, leaf_hi, leaf_sh_ptr, leaf_sh, leaf_nbf, sh_coords, sh_rcut2,
                           sh_nprim, sh_expnts, sh_bf0, sh_nbf, sh_L, bf_lmn, bf_coef, nd, cache_ptr, cache_data):
    nleaf = leaf_lo.shape[0]
    for il in prange(nleaf):
        lo = leaf_lo[il]
        k = leaf_hi[il] - lo
        q0 = leaf_sh_ptr[il]
        nshb = leaf_sh_ptr[il + 1] - q0
        n = leaf_nbf[il]
        if k == 0 or n == 0:
            continue
        c0 = cache_ptr[il]
        out = cache_data[c0:c0 + nd * k * n].reshape(nd, k, n)
        _eval_ao_block(coords, lo, k, leaf_sh, q0, nshb, sh_coords, sh_rcut2, sh_nprim, sh_expnts,
                       sh_bf0, sh_nbf, sh_L, bf_lmn, bf_coef, nd, out)


# ---------------------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------------------
@njit(parallel=True, **_NB)
def _xc3_kernel(mode, nd, fam, coords, weights, dmat,
                sh_coords, sh_rcut2, sh_nprim, sh_expnts, sh_bf0, sh_nbf, sh_L, bf_lmn, bf_coef,
                leaf_lo, leaf_hi, leaf_sh_ptr, leaf_sh,
                macro_leaf_lo, macro_leaf_hi, macro_sh_ptr, macro_sh, macro_bf_ptr, macro_bf, store_ptr,
                bin_ptr, bin_blocks, nthreads,
                fids, coefs,
                cache_data, cache_ptr, use_cache,
                rho_g, grad_g, tau_g, vrho_g, vsigma_g, vtau_g,
                V_store, exc_M, nel_M, max_k, max_nb, max_nbM, ws, iws):
    """One parallel region over the threads; thread ``t`` processes the macro blocks
    ``bin_blocks[bin_ptr[t]:bin_ptr[t+1]]``. ``ws[t]`` and ``iws[t]`` are the preallocated
    float and integer workspaces of the thread (see :func:`xc_workspace_size`).

    mode 0: density, functional (``fids``/``coefs``, Numba) and potential in one pass;
    mode 1: density (and gradient, tau) only, written to ``rho_g``, ``grad_g``, ``tau_g``;
    mode 2: potential only, from ``vrho_g``, ``vsigma_g``, ``vtau_g`` and ``grad_g``.
    """
    nsh = sh_coords.shape[0]
    kn = max_k * max_nb
    for t in prange(nthreads):
        w = ws[t]
        o = 0
        ao_buf = w[o:o + 4 * kn]
        o += 4 * kn
        B_buf = w[o:o + 4 * kn]
        o += 4 * kn
        F_buf = w[o:o + 3 * kn]
        o += 3 * kn
        Ds_buf = w[o:o + max_nb * max_nb]
        o += max_nb * max_nb
        M_buf = w[o:o + max_nb * max_nb]
        o += max_nb * max_nb
        D_M = w[o:o + max_nbM * max_nbM].reshape(max_nbM, max_nbM)
        o += max_nbM * max_nbM
        V_M = w[o:o + max_nbM * max_nbM].reshape(max_nbM, max_nbM)
        o += max_nbM * max_nbM
        rho = w[o:o + max_k]
        o += max_k
        sigma = w[o:o + max_k]
        o += max_k
        tau = w[o:o + max_k]
        o += max_k
        gx = w[o:o + max_k]
        o += max_k
        gy = w[o:o + max_k]
        o += max_k
        gz = w[o:o + max_k]
        o += max_k
        e = w[o:o + max_k]
        o += max_k
        vrho = w[o:o + max_k]
        o += max_k
        vsigma = w[o:o + max_k]
        o += max_k
        vtau = w[o:o + max_k]
        pos_in_macro = iws[t, :nsh]
        ridx = iws[t, nsh:nsh + max_nb]
        for jb in range(bin_ptr[t], bin_ptr[t + 1]):
            ib = bin_blocks[jb]
            p0 = macro_bf_ptr[ib]
            nbM = macro_bf_ptr[ib + 1] - p0
            off = 0
            for q in range(macro_sh_ptr[ib], macro_sh_ptr[ib + 1]):
                s = macro_sh[q]
                pos_in_macro[s] = off
                off += sh_nbf[s]
            for i in range(nbM):
                gi = macro_bf[p0 + i]
                for j in range(nbM):
                    D_M[i, j] = dmat[gi, macro_bf[p0 + j]]
                    V_M[i, j] = 0.0
            exc_acc = 0.0
            nel_acc = 0.0
            for il in range(macro_leaf_lo[ib], macro_leaf_hi[ib]):
                lo = leaf_lo[il]
                k = leaf_hi[il] - lo
                q0 = leaf_sh_ptr[il]
                nshb = leaf_sh_ptr[il + 1] - q0
                n = 0
                for q in range(q0, q0 + nshb):
                    s = leaf_sh[q]
                    pp = pos_in_macro[s]
                    for c in range(sh_nbf[s]):
                        ridx[n] = pp + c
                        n += 1
                if n == 0 or k == 0:
                    if mode == 1:
                        for kk in range(k):
                            rho_g[lo + kk] = 0.0
                            if fam >= 2:
                                grad_g[0, lo + kk] = 0.0
                                grad_g[1, lo + kk] = 0.0
                                grad_g[2, lo + kk] = 0.0
                            if fam == 4:
                                tau_g[lo + kk] = 0.0
                    continue
                # ---- basis functions on the leaf
                if use_cache == 1:
                    c0 = cache_ptr[il]
                    ao = cache_data[c0:c0 + nd * k * n].reshape(nd, k, n)
                else:
                    ao = ao_buf[:nd * k * n].reshape(nd, k, n)
                    _eval_ao_block(coords, lo, k, leaf_sh, q0, nshb, sh_coords, sh_rcut2, sh_nprim, sh_expnts,
                                   sh_bf0, sh_nbf, sh_L, bf_lmn, bf_coef, nd, ao)
                # ---- density matrix of the leaf and the density
                Ds = Ds_buf[:n * n].reshape(n, n)
                for i in range(n):
                    ri = ridx[i]
                    for j in range(n):
                        Ds[i, j] = D_M[ri, ridx[j]]
                F = F_buf[:k * n].reshape(k, n)
                np.dot(ao[0], Ds, F)
                for kk in range(k):
                    s0 = 0.0
                    for j in range(n):
                        s0 += F[kk, j] * ao[0, kk, j]
                    rho[kk] = s0
                if fam >= 2:
                    for kk in range(k):
                        sx = 0.0
                        sy = 0.0
                        sz = 0.0
                        for j in range(n):
                            f = F[kk, j]
                            sx += f * ao[1, kk, j]
                            sy += f * ao[2, kk, j]
                            sz += f * ao[3, kk, j]
                        gx[kk] = 2.0 * sx
                        gy[kk] = 2.0 * sy
                        gz[kk] = 2.0 * sz
                        sigma[kk] = gx[kk] * gx[kk] + gy[kk] * gy[kk] + gz[kk] * gz[kk]
                if fam == 4:
                    H = F_buf[:3 * k * n].reshape(3 * k, n)
                    A3 = ao[1:4].reshape(3 * k, n)
                    np.dot(A3, Ds, H)
                    for kk in range(k):
                        s0 = 0.0
                        for c in range(3):
                            for j in range(n):
                                s0 += ao[1 + c, kk, j] * H[c * k + kk, j]
                        tau[kk] = 0.5 * s0
                if mode == 1:
                    for kk in range(k):
                        rho_g[lo + kk] = rho[kk]
                        if fam >= 2:
                            grad_g[0, lo + kk] = gx[kk]
                            grad_g[1, lo + kk] = gy[kk]
                            grad_g[2, lo + kk] = gz[kk]
                        if fam == 4:
                            tau_g[lo + kk] = tau[kk]
                    continue
                # ---- functional
                if mode == 0:
                    _xc_native_eval(fids, coefs, rho[:k], sigma[:k], tau[:k], e[:k], vrho[:k], vsigma[:k], vtau[:k])
                    for kk in range(k):
                        wr = weights[lo + kk] * rho[kk]
                        exc_acc += wr * e[kk]
                        nel_acc += wr
                else:
                    for kk in range(k):
                        vrho[kk] = vrho_g[lo + kk]
                        if fam >= 2:
                            vsigma[kk] = vsigma_g[lo + kk]
                            gx[kk] = grad_g[0, lo + kk]
                            gy[kk] = grad_g[1, lo + kk]
                            gz[kk] = grad_g[2, lo + kk]
                        if fam == 4:
                            vtau[kk] = vtau_g[lo + kk]
                # ---- potential: V_leaf = M + M^T with M = A^T B
                if fam == 4:
                    ndB = 4
                else:
                    ndB = 1
                B = B_buf[:ndB * k * n].reshape(ndB, k, n)
                for kk in range(k):
                    w = weights[lo + kk]
                    f0 = 0.5 * w * vrho[kk]
                    if fam == 1:
                        for j in range(n):
                            B[0, kk, j] = f0 * ao[0, kk, j]
                    else:
                        fs = 2.0 * w * vsigma[kk]
                        fx = fs * gx[kk]
                        fy = fs * gy[kk]
                        fz = fs * gz[kk]
                        for j in range(n):
                            B[0, kk, j] = f0 * ao[0, kk, j] + fx * ao[1, kk, j] + fy * ao[2, kk, j] + fz * ao[3, kk, j]
                        if fam == 4:
                            gt = 0.25 * w * vtau[kk]
                            for c in range(3):
                                for j in range(n):
                                    B[1 + c, kk, j] = gt * ao[1 + c, kk, j]
                A = ao[:ndB].reshape(ndB * k, n)
                B2 = B.reshape(ndB * k, n)
                M = M_buf[:n * n].reshape(n, n)
                np.dot(A.T, B2, M)
                for i in range(n):
                    ri = ridx[i]
                    for j in range(n):
                        V_M[ri, ridx[j]] += M[i, j]
            if mode != 1:
                base = store_ptr[ib]
                for i in range(nbM):
                    for j in range(nbM):
                        V_store[base + i * nbM + j] = V_M[i, j]
            exc_M[ib] = exc_acc
            nel_M[ib] = nel_acc


def xc_workspace_size(max_k, max_nb, max_nbM):
    """Doubles per thread the XC kernel needs (see the slicing at the top of its thread loop)."""
    return 11 * max_k * max_nb + 2 * max_nb * max_nb + 2 * max_nbM * max_nbM + 10 * max_k


@njit(parallel=True, **_NB)
def _assemble_v(V_store, store_ptr, macro_bf_ptr, macro_bf, nbf, nbands):
    """Sum the macro-block slots into the full matrix by row bands (each band is written by
    one thread, in a fixed order), then symmetrize: ``V = S + S^T``."""
    nmacro = macro_bf_ptr.shape[0] - 1
    S = np.zeros((nbf, nbf))
    band = (nbf + nbands - 1) // nbands
    for b in prange(nbands):
        r0 = b * band
        r1 = min(nbf, r0 + band)
        if r0 >= r1:
            continue
        for ib in range(nmacro):
            p0 = macro_bf_ptr[ib]
            nbM = macro_bf_ptr[ib + 1] - p0
            seg = macro_bf[p0:p0 + nbM]
            i_lo = np.searchsorted(seg, r0)
            i_hi = np.searchsorted(seg, r1)
            base = store_ptr[ib]
            for i in range(i_lo, i_hi):
                gi = seg[i]
                row = base + i * nbM
                for j in range(nbM):
                    S[gi, seg[j]] += V_store[row + j]
    V = np.empty((nbf, nbf))
    for i in prange(nbf):
        for j in range(nbf):
            V[i, j] = S[i, j] + S[j, i]
    return V


