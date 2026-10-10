"""
Nuclear gradient of the two-electron energy without density fitting on the GPU: the device port of
:mod:`pyfock.Integrals.jk_4c2e_grad`.

For a closed-shell density ``D`` and a fraction ``a`` of exact exchange,

    E_2 = 1/2 sum D_ab D_cd (ab|cd) - a/4 sum D_ac D_bd (ab|cd),

the gradient term is ``sum_abcd Gamma_abcd d(ab|cd)/dR`` over the unique shell quartets ``(XY|ZW)`` of
a :func:`~pyfock.Integrals.jk_4c2e.build_plan` plan (bra pair ``i``, ket pair ``j >= i``), every
element weighted by ``s (4 D_ab D_cd - a (D_ac D_bd + D_ad D_bc))`` (``s`` the degeneracy factor),
with the screening of the CPU code: a quartet is skipped when
``Q_XY Q_ZW max(4 D_XY D_ZW, a (D_XZ D_YW + D_XW D_YZ))`` (shell-block maxima of ``|D|``) is below
the threshold, and its primitive quartets below ``PRIM_FACTOR`` times the threshold over that
density factor.

One thread per surviving shell quartet.  The CPU kernel accumulates the contracted ``(e0|f0)``
over the primitive quartets and folds the weights into one horizontal transfer per quartet; its
scratch arrays (several ``(l+1)^2 (l+2)^2``-sized tables per thread) do not fit a GPU thread, so
the device kernel works per primitive quartet and root instead: the three two-dimensional Rys
tables up to one order above the quartet on the bra side (and, when the ``Z`` derivative is needed,
on the ket side), then for every component quartet the horizontal transfers of the plain integral
and of the shifted ones that the derivatives

    d(ab|cd)/dX_k = 2 alpha (a + 1_k, b|cd) - a_k (a - 1_k, b|cd)

(and likewise for ``Y`` and ``Z``; ``W`` by translational invariance) need, contracted at once with
the weight.  The ``X`` and ``Y`` primitives are those of the bra primitive pair, ``Z`` that of the
ket pair (``pp_a``), so the derivatives can be taken inside the primitive loops.  As on the CPU, a
quartet with all four centres on one atom is skipped, the ``Z`` derivative is not evaluated when
``Z`` and ``W`` share an atom, and a quartet whose bra pair is on one atom (and ket pair on two) is
differentiated as ``(ZW|XY)``.  The tasks are generated on the device in chunks of bra pairs and
sorted by their angular-momentum class, so that a warp follows one recursion shape.
"""
import math

import numpy as np
from numba import cuda
import numba

try:
    import cupy as cp
except Exception:                                  # pragma: no cover - CPU-only install
    cp = None

from .jk_4c2e import MAX_ROOTS, PRIM_FACTOR, TWO_PI_52, _shell_dmax
from .rys_helpers_cuda import Roots, DATA_X, DATA_W
from .cuda_stream import gradient_stream

__all__ = ['grad_4c2e_cupy']

# Binomial coefficients for the horizontal transfers (up to l = 15).
_COMB = np.array([[float(math.comb(n, k)) if k <= n else 0.0 for k in range(16)]
                  for n in range(16)], dtype=np.float64)

# Spread the per-atom atomics over several copies of the gradient (summed at the end).
_NBINS = 64

# Upper bound on the (bra pair, ket pair) candidates materialised at once.
_TASK_CHUNK = 16 * 1024 * 1024

_KERNEL_CACHE = {}


def _nb(array):
    """Numba view of a CuPy array without Numba's implicit synchronization of the current stream."""
    return cuda.as_cuda_array(array, sync=False)


def _get_kernel(lmax):
    """Compile (once per maximum angular momentum) the quartet kernel with local tables sized for it."""
    kernel = _KERNEL_CACHE.get(lmax)
    if kernel is not None:
        return kernel
    NG = 2 * lmax + 2               # bra/ket orders up to l_X + l_Y + 1
    NP = lmax + 2                   # powers of AB and CD
    NR = MAX_ROOTS

    @cuda.jit(device=True, inline=True)
    def _hrr(g, a, b, c, d, pab, pcd, comb):
        """sum_n C(b,n) AB^(b-n) sum_m C(d,m) CD^(d-m) g[a+n, c+m]: one direction of (ab|cd)."""
        acc = 0.0
        for m in range(d + 1):
            inner = 0.0
            for n in range(b + 1):
                inner += comb[b, n] * pab[b - n] * g[a + n, c + m]
            acc += comb[d, m] * pcd[d - m] * inner
        return acc

    @cuda.jit(fastmath=True, cache=False)
    def _kernel(tasks, npair, exx, threshold, pair_sh, pair_pp0, pair_ppn, pair_ab,
                pp_p, pp_x, pp_y, pp_z, pp_c, pp_q, pp_a,
                sh_l, sh_off, sh_nbf, sh_cen, sh_atom, bfs_lmn, comp_scale, dmat, dsh,
                DATA_X_, DATA_W_, grad_partial):
        comb = cuda.const.array_like(_COMB)
        it = cuda.grid(1)
        if it >= tasks.shape[0]:
            return
        t = tasks[it]
        i = t // npair                  # bra pair (possibly the original ket pair, see the module docstring)
        j = t - i * npair
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
        aX = sh_atom[X]
        aY = sh_atom[Y]
        aZ = sh_atom[Z]
        aW = sh_atom[W]
        needz = aZ != aW
        s = 1.0
        if X == Y:
            s *= 0.5
        if Z == W:
            s *= 0.5
        if i == j:
            s *= 0.5
        dm = max(4.0 * dsh[X, Y] * dsh[Z, W], exx * (dsh[X, Z] * dsh[Y, W] + dsh[X, W] * dsh[Y, Z]))
        thr_prim = PRIM_FACTOR * threshold / dm
        lbra = lx + ly + 1              # bra order, one above the quartet for the X and Y derivatives
        lket = lz + lw + 1 if needz else lz + lw
        nroots = (lx + ly + lz + lw + 1) // 2 + 1

        roots = cuda.local.array(NR, numba.float64)
        weights = cuda.local.array(NR, numba.float64)
        gx = cuda.local.array((NG, NG), numba.float64)
        gy = cuda.local.array((NG, NG), numba.float64)
        gz = cuda.local.array((NG, NG), numba.float64)
        pabx = cuda.local.array(NP, numba.float64)
        paby = cuda.local.array(NP, numba.float64)
        pabz = cuda.local.array(NP, numba.float64)
        pcdx = cuda.local.array(NP, numba.float64)
        pcdy = cuda.local.array(NP, numba.float64)
        pcdz = cuda.local.array(NP, numba.float64)
        pabx[0] = 1.0
        paby[0] = 1.0
        pabz[0] = 1.0
        pcdx[0] = 1.0
        pcdy[0] = 1.0
        pcdz[0] = 1.0
        for k in range(1, ly + 2):
            pabx[k] = pabx[k - 1] * pair_ab[i, 0]
            paby[k] = paby[k - 1] * pair_ab[i, 1]
            pabz[k] = pabz[k - 1] * pair_ab[i, 2]
        for k in range(1, lw + 2):
            pcdx[k] = pcdx[k - 1] * pair_ab[j, 0]
            pcdy[k] = pcdy[k - 1] * pair_ab[j, 1]
            pcdz[k] = pcdz[k - 1] * pair_ab[j, 2]
        Xx = sh_cen[X, 0]
        Xy = sh_cen[X, 1]
        Xz = sh_cen[X, 2]
        Zx = sh_cen[Z, 0]
        Zy = sh_cen[Z, 1]
        Zz = sh_cen[Z, 2]

        gXx = 0.0
        gXy = 0.0
        gXz = 0.0
        gYx = 0.0
        gYy = 0.0
        gYz = 0.0
        gZx = 0.0
        gZy = 0.0
        gZz = 0.0

        a0 = pair_pp0[i]
        a1 = a0 + pair_ppn[i]
        b0 = pair_pp0[j]
        b1 = b0 + pair_ppn[j]
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
            pax = px - Xx
            pay = py - Xy
            paz = pz - Xz
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
                Roots(nroots, p * q * inv_pq * (pqx * pqx + pqy * pqy + pqz * pqz), DATA_X_, DATA_W_,
                      roots, weights)
                pref = TWO_PI_52 / (p * q * math.sqrt(pq)) * ca * pp_c[b]
                two_gamma = 2.0 * pp_a[b]
                qcx = qx - Zx
                qcy = qy - Zy
                qcz = qz - Zz
                inv_2q = 0.5 / q
                for r in range(nroots):
                    u = roots[r]
                    t2 = u / (1.0 + u)
                    fb = q * t2 * inv_pq
                    fk = p * t2 * inv_pq
                    b00 = 0.5 * t2 * inv_pq
                    b10 = (1.0 - fb) * inv_2p
                    b01 = (1.0 - fk) * inv_2q
                    c0x = pax - fb * pqx
                    c0y = pay - fb * pqy
                    c0z = paz - fb * pqz
                    d0x = qcx + fk * pqx
                    d0y = qcy + fk * pqy
                    d0z = qcz + fk * pqz
                    wz = weights[r] * pref
                    # two-dimensional Rys tables g[n, m] = I(n, 0, m, 0), n <= lbra, m <= lket
                    gx[0, 0] = 1.0
                    gy[0, 0] = 1.0
                    gz[0, 0] = wz
                    gx[1, 0] = c0x
                    gy[1, 0] = c0y
                    gz[1, 0] = c0z * wz
                    for n in range(1, lbra):
                        nb10 = n * b10
                        gx[n + 1, 0] = c0x * gx[n, 0] + nb10 * gx[n - 1, 0]
                        gy[n + 1, 0] = c0y * gy[n, 0] + nb10 * gy[n - 1, 0]
                        gz[n + 1, 0] = c0z * gz[n, 0] + nb10 * gz[n - 1, 0]
                    if lket > 0:
                        gx[0, 1] = d0x
                        gy[0, 1] = d0y
                        gz[0, 1] = d0z * wz
                        for n in range(1, lbra + 1):
                            nb00 = n * b00
                            gx[n, 1] = d0x * gx[n, 0] + nb00 * gx[n - 1, 0]
                            gy[n, 1] = d0y * gy[n, 0] + nb00 * gy[n - 1, 0]
                            gz[n, 1] = d0z * gz[n, 0] + nb00 * gz[n - 1, 0]
                        for m in range(1, lket):
                            mb01 = m * b01
                            gx[0, m + 1] = d0x * gx[0, m] + mb01 * gx[0, m - 1]
                            gy[0, m + 1] = d0y * gy[0, m] + mb01 * gy[0, m - 1]
                            gz[0, m + 1] = d0z * gz[0, m] + mb01 * gz[0, m - 1]
                            for n in range(1, lbra + 1):
                                nb00 = n * b00
                                gx[n, m + 1] = d0x * gx[n, m] + mb01 * gx[n, m - 1] + nb00 * gx[n - 1, m]
                                gy[n, m + 1] = d0y * gy[n, m] + mb01 * gy[n, m - 1] + nb00 * gy[n - 1, m]
                                gz[n, m + 1] = d0z * gz[n, m] + mb01 * gz[n, m - 1] + nb00 * gz[n - 1, m]

                    for x in range(nX):
                        ix = x0 + x
                        ax_ = bfs_lmn[ix, 0]
                        ay_ = bfs_lmn[ix, 1]
                        az_ = bfs_lmn[ix, 2]
                        for y in range(nY):
                            iy = y0 + y
                            bx_ = bfs_lmn[iy, 0]
                            by_ = bfs_lmn[iy, 1]
                            bz_ = bfs_lmn[iy, 2]
                            sxy = s * comp_scale[ix] * comp_scale[iy]
                            dxy4 = 4.0 * dmat[ix, iy]
                            for z in range(nZ):
                                iz = z0 + z
                                cx_ = bfs_lmn[iz, 0]
                                cy_ = bfs_lmn[iz, 1]
                                cz_ = bfs_lmn[iz, 2]
                                sxyz = sxy * comp_scale[iz]
                                dxz = exx * dmat[ix, iz]
                                dyz = exx * dmat[iy, iz]
                                for w in range(nW):
                                    iw = w0 + w
                                    gam = sxyz * comp_scale[iw] * (dxy4 * dmat[iz, iw] - dxz * dmat[iy, iw]
                                                                   - dyz * dmat[ix, iw])
                                    if gam == 0.0:
                                        continue
                                    dx_ = bfs_lmn[iw, 0]
                                    dy_ = bfs_lmn[iw, 1]
                                    dz_ = bfs_lmn[iw, 2]
                                    # ---- x ----
                                    ix0 = _hrr(gx, ax_, bx_, cx_, dx_, pabx, pcdx, comb)
                                    dXx = two_alpha * _hrr(gx, ax_ + 1, bx_, cx_, dx_, pabx, pcdx, comb)
                                    if ax_ > 0:
                                        dXx -= ax_ * _hrr(gx, ax_ - 1, bx_, cx_, dx_, pabx, pcdx, comb)
                                    dYx = two_beta * _hrr(gx, ax_, bx_ + 1, cx_, dx_, pabx, pcdx, comb)
                                    if bx_ > 0:
                                        dYx -= bx_ * _hrr(gx, ax_, bx_ - 1, cx_, dx_, pabx, pcdx, comb)
                                    # ---- y ----
                                    iy0 = _hrr(gy, ay_, by_, cy_, dy_, paby, pcdy, comb)
                                    dXy = two_alpha * _hrr(gy, ay_ + 1, by_, cy_, dy_, paby, pcdy, comb)
                                    if ay_ > 0:
                                        dXy -= ay_ * _hrr(gy, ay_ - 1, by_, cy_, dy_, paby, pcdy, comb)
                                    dYy = two_beta * _hrr(gy, ay_, by_ + 1, cy_, dy_, paby, pcdy, comb)
                                    if by_ > 0:
                                        dYy -= by_ * _hrr(gy, ay_, by_ - 1, cy_, dy_, paby, pcdy, comb)
                                    # ---- z ----
                                    iz0 = _hrr(gz, az_, bz_, cz_, dz_, pabz, pcdz, comb)
                                    dXz = two_alpha * _hrr(gz, az_ + 1, bz_, cz_, dz_, pabz, pcdz, comb)
                                    if az_ > 0:
                                        dXz -= az_ * _hrr(gz, az_ - 1, bz_, cz_, dz_, pabz, pcdz, comb)
                                    dYz = two_beta * _hrr(gz, az_, bz_ + 1, cz_, dz_, pabz, pcdz, comb)
                                    if bz_ > 0:
                                        dYz -= bz_ * _hrr(gz, az_, bz_ - 1, cz_, dz_, pabz, pcdz, comb)
                                    yz = gam * iy0 * iz0
                                    xz = gam * ix0 * iz0
                                    xy = gam * ix0 * iy0
                                    gXx += dXx * yz
                                    gXy += dXy * xz
                                    gXz += dXz * xy
                                    gYx += dYx * yz
                                    gYy += dYy * xz
                                    gYz += dYz * xy
                                    if needz:
                                        dZx = two_gamma * _hrr(gx, ax_, bx_, cx_ + 1, dx_, pabx, pcdx, comb)
                                        if cx_ > 0:
                                            dZx -= cx_ * _hrr(gx, ax_, bx_, cx_ - 1, dx_, pabx, pcdx, comb)
                                        dZy = two_gamma * _hrr(gy, ay_, by_, cy_ + 1, dy_, paby, pcdy, comb)
                                        if cy_ > 0:
                                            dZy -= cy_ * _hrr(gy, ay_, by_, cy_ - 1, dy_, paby, pcdy, comb)
                                        dZz = two_gamma * _hrr(gz, az_, bz_, cz_ + 1, dz_, pabz, pcdz, comb)
                                        if cz_ > 0:
                                            dZz -= cz_ * _hrr(gz, az_, bz_, cz_ - 1, dz_, pabz, pcdz, comb)
                                        gZx += dZx * yz
                                        gZy += dZy * xz
                                        gZz += dZz * xy

        ibin = cuda.blockIdx.x % _NBINS
        cuda.atomic.add(grad_partial, (ibin, aX, 0), gXx)
        cuda.atomic.add(grad_partial, (ibin, aX, 1), gXy)
        cuda.atomic.add(grad_partial, (ibin, aX, 2), gXz)
        cuda.atomic.add(grad_partial, (ibin, aY, 0), gYx)
        cuda.atomic.add(grad_partial, (ibin, aY, 1), gYy)
        cuda.atomic.add(grad_partial, (ibin, aY, 2), gYz)
        if needz:
            cuda.atomic.add(grad_partial, (ibin, aZ, 0), gZx)
            cuda.atomic.add(grad_partial, (ibin, aZ, 1), gZy)
            cuda.atomic.add(grad_partial, (ibin, aZ, 2), gZz)
        # translational invariance: d/dW = -(d/dX + d/dY + d/dZ); without needz Z and W share the atom
        cuda.atomic.add(grad_partial, (ibin, aW, 0), -(gXx + gYx + gZx))
        cuda.atomic.add(grad_partial, (ibin, aW, 1), -(gXy + gYy + gZy))
        cuda.atomic.add(grad_partial, (ibin, aW, 2), -(gXz + gYz + gZz))

    _KERNEL_CACHE[lmax] = _kernel
    return _kernel


@cuda.jit(cache=True)
def _quartet_tasks(offsets, lo, npair, Q, pair_sh, pair_ppn, sh_l, sh_atom, dsh, exx, threshold, tasks, keys):
    """
    Candidate ``k`` of a chunk of bra pairs ``lo, lo + 1, ...`` (``offsets``: running count of their
    ket ranges ``j = i, i + 1, ...``) is the quartet ``(i, j)``; screen it.  ``tasks[k]`` is ``-1``
    for a quartet the energy's screening or the one-centre rule drops, else the task id
    ``first * npair + second`` with the pair that is differentiated first (the ket pair when the
    bra pair is on one atom and the ket pair on two); ``keys[k]`` is its class (angular momenta and
    whether ``Z`` is needed) and, within the class, its number of primitive quartets, so that the
    lanes of a warp run loops of similar length.
    """
    k = cuda.grid(1)
    if k >= tasks.shape[0]:
        return
    # the bra pair whose ket range holds candidate k: offsets[t] <= k < offsets[t + 1]
    left = 0
    right = offsets.shape[0] - 1
    while right - left > 1:
        mid = (left + right) // 2
        if offsets[mid] <= k:
            left = mid
        else:
            right = mid
    i = lo + left
    j = i + (k - offsets[left])
    tasks[k] = -1
    X = pair_sh[i, 0]
    Y = pair_sh[i, 1]
    Z = pair_sh[j, 0]
    W = pair_sh[j, 1]
    qq = Q[i] * Q[j]
    dm = max(4.0 * dsh[X, Y] * dsh[Z, W], exx * (dsh[X, Z] * dsh[Y, W] + dsh[X, W] * dsh[Y, Z]))
    if qq * dm < threshold:
        return
    aX = sh_atom[X]
    aY = sh_atom[Y]
    aZ = sh_atom[Z]
    aW = sh_atom[W]
    if aX == aY and aX == aZ and aX == aW:
        return
    nprim = min(pair_ppn[i] * pair_ppn[j], 1048575)
    if aX == aY and aZ != aW:
        tasks[k] = j * npair + i
        keys[k] = ((((sh_l[Z] * 8 + sh_l[W]) * 8 + sh_l[X]) * 8 + sh_l[Y]) * 2) * 1048576 + nprim
    else:
        tasks[k] = i * npair + j
        keys[k] = (((((sh_l[X] * 8 + sh_l[Y]) * 8 + sh_l[Z]) * 8 + sh_l[W]) * 2 + (1 if aZ != aW else 0))
                   * 1048576 + nprim)


def grad_4c2e_cupy(plan, dmat, exx_coef=0.0, threshold=None, cp_stream=None):
    """
    GPU counterpart of :func:`pyfock.Integrals.jk_4c2e_grad.grad_4c2e`: ``sum_abcd Gamma_abcd
    d(ab|cd)/dR`` for the two-electron energy ``1/2 sum D_ab D_cd (ab|cd) - exx_coef/4 sum D_ac D_bd
    (ab|cd)`` of a symmetric density matrix ``dmat`` (Cartesian AO basis, NumPy or CuPy), with the
    shell pairs and screening of a :func:`~pyfock.Integrals.jk_4c2e.build_plan` plan (either scheme:
    the derivatives always use Rys quadrature).  ``threshold`` defaults to the plan's.

    Returns a NumPy ``(natoms, 3)`` array in Hartree/Bohr.
    """
    if cp is None:
        raise RuntimeError('CuPy is required for grad_4c2e_cupy.')
    dmat_h = np.ascontiguousarray(cp.asnumpy(dmat) if isinstance(dmat, cp.ndarray) else dmat, dtype=np.float64)
    thr = plan.threshold if threshold is None else float(threshold)
    lmax = int(plan.lmax)
    if 2 * lmax + 1 > MAX_ROOTS:
        raise ValueError(f'Four-center derivative integrals support shells up to l = {(MAX_ROOTS - 1) // 2}.')
    sh_l, sh_off, sh_nbf, sh_cen, bfs_lmn, comp_scale = plan.tables[:6]
    sh_atom = np.ascontiguousarray(plan.bfs_atoms[sh_off])
    natoms = int(plan.bfs_atoms.max()) + 1
    dsh = _shell_dmax(dmat_h, sh_off, sh_nbf)
    dmax = float(dsh.max()) if dsh.size else 0.0
    npair = int(plan.npairs)
    Q = np.ascontiguousarray(plan.Q, dtype=np.float64)
    # ket ranges: as the pairs are sorted by decreasing Q, the ket loop of bra pair i ends at the first
    # pair with Q_i Q_j 4 max|D|^2 below the threshold (the CPU kernel's break)
    dlim = 4.0 * dmax * dmax
    with np.errstate(divide='ignore'):
        qmin = np.where(Q * dlim > 0.0, thr / (Q * dlim), np.inf)
    jend = np.searchsorted(-Q, -qmin, side='right')          # number of j with Q_j >= qmin_i
    first = np.arange(npair)
    counts = np.maximum(jend - first, 0).astype(np.int64)

    if cp_stream is None:
        cp_stream, nb_stream = gradient_stream()
    else:
        nb_stream = cuda.external_stream(cp_stream.ptr)
    kernel = _get_kernel(lmax)
    exx = float(exx_coef)
    upload = cp.cuda.get_current_stream()
    with cp_stream:
        if upload.ptr != cp_stream.ptr:
            cp_stream.wait_event(upload.record())
        pp_p, pp_x, pp_y, pp_z, pp_c, pp_q = plan.pp
        d = {name: cp.asarray(np.ascontiguousarray(value)) for name, value in (
            ('Q', Q), ('pair_sh', plan.pair_sh), ('pair_pp0', plan.pair_pp0), ('pair_ppn', plan.pair_ppn),
            ('pair_ab', plan.pair_ab), ('pp_p', pp_p), ('pp_x', pp_x), ('pp_y', pp_y), ('pp_z', pp_z),
            ('pp_c', pp_c), ('pp_q', pp_q), ('pp_a', plan.pp_a), ('sh_l', sh_l), ('sh_off', sh_off),
            ('sh_nbf', sh_nbf), ('sh_cen', sh_cen), ('sh_atom', sh_atom), ('bfs_lmn', bfs_lmn),
            ('comp_scale', comp_scale), ('dmat', dmat_h), ('dsh', dsh), ('DATA_X', DATA_X),
            ('DATA_W', DATA_W))}
        nb = {name: _nb(value) for name, value in d.items()}
        grad_partial = cp.zeros((_NBINS, natoms, 3), dtype=cp.float64)
        nb_grad = _nb(grad_partial)
        threads = 64
        lo = 0
        while lo < npair:
            # bra pairs [lo, hi) with at most _TASK_CHUNK candidates (at least one bra pair)
            hi = lo + 1
            total = int(counts[lo])
            while hi < npair and total + counts[hi] <= _TASK_CHUNK:
                total += int(counts[hi])
                hi += 1
            if total:
                offsets = np.zeros(hi - lo + 1, dtype=np.int64)
                offsets[1:] = np.cumsum(counts[lo:hi])
                offsets = cp.asarray(offsets)
                tasks = cp.empty(total, dtype=cp.int64)
                keys = cp.empty(total, dtype=cp.int64)
                _quartet_tasks[(total + 127) // 128, 128, nb_stream](
                    _nb(offsets), lo, npair, nb['Q'], nb['pair_sh'], nb['pair_ppn'], nb['sh_l'], nb['sh_atom'],
                    nb['dsh'], exx, thr, _nb(tasks), _nb(keys))
                keep = tasks >= 0
                tasks = tasks[keep]
                if tasks.size:
                    tasks = cp.ascontiguousarray(tasks[cp.argsort(keys[keep])])
                    kernel[(tasks.size + threads - 1) // threads, threads, nb_stream](
                        _nb(tasks), npair, exx, thr, nb['pair_sh'], nb['pair_pp0'], nb['pair_ppn'],
                        nb['pair_ab'], nb['pp_p'], nb['pp_x'], nb['pp_y'], nb['pp_z'], nb['pp_c'],
                        nb['pp_q'], nb['pp_a'], nb['sh_l'], nb['sh_off'], nb['sh_nbf'], nb['sh_cen'],
                        nb['sh_atom'], nb['bfs_lmn'], nb['comp_scale'], nb['dmat'], nb['dsh'],
                        nb['DATA_X'], nb['DATA_W'], nb_grad)
                del offsets, keys, keep, tasks
            lo = hi
        grad = cp.asnumpy(grad_partial.sum(axis=0))
    cp_stream.synchronize()
    return grad
