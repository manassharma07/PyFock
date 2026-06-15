try:
    import cupy as cp
    from cupy import fuse
except Exception as e:
    # Handle the case when Cupy is not installed
    cp = None

    def fuse(kernel_name):
        def decorator(func):
            return func
        return decorator
from numba import cuda
import math
from numba import njit, prange
import numpy as np
import numba

# Import only the (large) lookup-table DATA arrays from the nuclear-matrix
# CuPy kernel; the device helper functions are defined locally below so that
# every ``cuda.const.array_like`` call lives in this module. This keeps the
# module self-contained (no cross-module device-function calls, which are
# fragile under the CUDA simulator and add compile-time coupling) while
# avoiding duplicating the big Boys Taylor table.
from .nuc_mat_symm_cupy import TABLE, LOOKUP_TABLE, LOOKUP_TABLE_COMB


@cuda.jit(fastmath=True, cache=True, device=True)
def fastFactorial(n):
    LOOKUP_TABLE_ = cuda.const.array_like(LOOKUP_TABLE)
    if n <= 1:
        return 1
    elif n <= 20:
        return LOOKUP_TABLE_[n]
    else:
        factorial = 1
        for i in range(2, n + 1):
            factorial *= i
        return factorial


@cuda.jit(fastmath=True, cache=True, device=True)
def comb(x, y):
    table = cuda.const.array_like(LOOKUP_TABLE_COMB)
    if y == 0:
        return 1
    if x == y:
        return 1
    if x <= 5 and y <= 5:
        return table[x, y]
    binom = fastFactorial(x) // fastFactorial(y) // fastFactorial(x - y)
    return binom


@cuda.jit(fastmath=True, cache=True, device=True)
def c2k(k, la, lb, PA, PB):
    temp = 0.0
    for i in range(la + 1):
        if i > k:
            continue
        factor1 = comb(la, i)
        factor2 = PA**(la - i)
        for j in range(lb + 1):
            if (i + j) == k:
                temp += factor1 * comb(lb, j) * factor2 * PB**(lb - j)
    return temp


@cuda.jit(fastmath=True, cache=True, device=True)
def vlriPartial(Ci, l, r, i):
    return (-1)**l * ((-1)**i * fastFactorial(l) * Ci**(l - 2 * r - 2 * i) / (fastFactorial(r) * fastFactorial(i) * fastFactorial(l - 2 * r - 2 * i)))


@cuda.jit(fastmath=True, cache=True, device=True)
def taylor_cuda(a, z):
    table = cuda.const.array_like(TABLE)
    z0 = int(round(z))
    zi = z0 + 50
    ai = a
    return (a + 0.5)*(z - z0)**15*table[ai + 15, zi]/(1307674368000.0*a + 20268952704000.0) + (a + 0.5)*(z - z0)**14*table[ai + 14, zi]/(87178291200.0*a + 1264085222400.0) + (a + 0.5)*(z - z0)**13*table[ai + 13, zi]/(6227020800.0*a + 84064780800.0) + (a + 0.5)*(z - z0)**12*table[ai + 12, zi]/(479001600.0*a + 5987520000.0) + (a + 0.5)*(z - z0)**11*table[ai + 11, zi]/(39916800.0*a + 459043200.0) + (a + 0.5)*(z - z0)**10*table[ai + 10, zi]/(3628800.0*a + 38102400.0) + (a + 0.5)*(z - z0)**9*table[ai + 9, zi]/(362880.0*a + 3447360.0) + (a + 0.5)*(z - z0)**8*table[ai + 8, zi]/(40320.0*a + 342720.0) + (a + 0.5)*(z - z0)**7*table[ai + 7, zi]/(5040.0*a + 37800.0) + (a + 0.5)*(z - z0)**6*table[ai + 6, zi]/(720.0*a + 4680.0) + (a + 0.5)*(z - z0)**5*table[ai + 5, zi]/(120.0*a + 660.0) + (a + 0.5)*(z - z0)**4*table[ai + 4, zi]/(24.0*a + 108.0) + (a + 0.5)*(z - z0)**3*table[ai + 3, zi]/(6.0*a + 21.0) + (a + 0.5)*(z - z0)**2*table[ai + 2, zi]/(2.0*a + 5.0) + (a + 0.5)*(z - z0)*table[ai + 1, zi]/(a + 1.5) + table[ai, zi]


@cuda.jit(fastmath=True, cache=True, device=True)
def hyp0minus(x):
    z = math.sqrt(-x)
    return 0.5 * math.erf(z) * math.sqrt(math.pi) / z


@cuda.jit(fastmath=True, cache=True, device=True)
def hyp1f1_new(m, z, hyp0minus_):
    TAYLOR_THRESHOLD = -25.0
    if z < TAYLOR_THRESHOLD:
        if m == 0:
            return hyp0minus_
        else:
            result = hyp0minus_
            for k in range(1, m + 1):
                result = ((2 * k + 1) * result - math.exp(z)) / (-2 * z)
            return result
    else:
        return taylor_cuda(m, z)


@cuda.jit(fastmath=True, cache=True, device=True)
def Fboys(m, T, hyp0minus_):
    return hyp1f1_new(m, -T, hyp0minus_) / (2 * m + 1)


def nuc_mat_grad_r_symm_cupy(basis, mol, slice=None, sqrt_ints4c2e_diag=None, cp_stream=None):
    """
    GPU (CuPy/Numba-CUDA) counterpart of :func:`nuc_mat_grad_r_symm`
    (``wrt_atoms=False``).

    Returns the derivative of the nuclear-attraction matrix elements with
    respect to the *bra* basis-function center,

        dV_r[d, i, j] = d <chi_i | V_nuc | chi_j> / dR_{center(i), d}

    as a CuPy array of shape ``(3, num_rows, num_cols)``, with the same layout
    and sign convention as the CPU routine. Note this is ONLY the
    basis-function (bra-center) derivative; the operator (Hellmann-Feynman)
    contribution from moving the nuclei is handled separately (the CPU
    ``wrt_atoms=True`` path / ``rys_nuc_grad_contract`` adds it). For the
    symmetric block the (j, i) entry is filled with the ket-center derivative.

    NOTE: Not yet executed on a CUDA device (developed on a CPU-only machine).
    It mirrors the verified ``nuc_mat_symm_cupy`` machinery and the verified
    ``nuc_mat_grad_r_symm`` math; validate against the CPU routine before
    production use.
    """
    bfs_coords = cp.array([basis.bfs_coords])
    bfs_contr_prim_norms = cp.array([basis.bfs_contr_prim_norms])
    bfs_lmn = cp.array([basis.bfs_lmn])
    bfs_nprim = cp.array([basis.bfs_nprim])
    coordsBohrs = cp.array([mol.coordsBohrs])
    Z = cp.array([mol.Zcharges])
    natoms = mol.natoms

    maxnprim = max(basis.bfs_nprim)
    bfs_coeffs = cp.zeros([basis.bfs_nao, maxnprim])
    bfs_expnts = cp.zeros([basis.bfs_nao, maxnprim])
    bfs_prim_norms = cp.zeros([basis.bfs_nao, maxnprim])
    for i in range(basis.bfs_nao):
        for j in range(basis.bfs_nprim[i]):
            bfs_coeffs[i, j] = basis.bfs_coeffs[i][j]
            bfs_expnts[i, j] = basis.bfs_expnts[i][j]
            bfs_prim_norms[i, j] = basis.bfs_prim_norms[i][j]

    if slice is None:
        slice = [0, basis.bfs_nao, 0, basis.bfs_nao]

    a = int(slice[0])
    b = int(slice[1])
    c = int(slice[2])
    d = int(slice[3])

    num_rows = b - a
    num_cols = d - c
    start_row, end_row, start_col, end_col = a, b, c, d

    upper_tri = False
    lower_tri = False
    both_tri_symm = False
    both_tri_nonsymm = False
    if end_row <= start_col:
        upper_tri = True
    elif start_row >= end_col:
        lower_tri = True
    elif start_row == start_col and end_row == end_col:
        both_tri_symm = True
    else:
        both_tri_nonsymm = True

    dV_r = cp.zeros((3, num_rows, num_cols))

    if sqrt_ints4c2e_diag is None:
        sqrt_ints4c2e_diag = cp.zeros((1, 1), dtype=cp.float64)
        isSchwarz = False
    else:
        isSchwarz = True
        sqrt_ints4c2e_diag = cp.asarray(sqrt_ints4c2e_diag)

    if cp_stream is None:
        device = 0
        cp.cuda.Device(device).use()
        cp_stream = cp.cuda.Stream(non_blocking=True)
        nb_stream = cuda.external_stream(cp_stream.ptr)
        cp_stream.use()
    else:
        nb_stream = cuda.external_stream(cp_stream.ptr)
        cp_stream.use()

    thread_x = 16
    thread_y = 16
    blocks_per_grid = ((num_rows + (thread_x - 1)) // thread_x, (num_cols + (thread_y - 1)) // thread_y)
    nuc_mat_grad_r_symm_internal_cuda[blocks_per_grid, (thread_x, thread_y), nb_stream](
        bfs_coords[0], bfs_contr_prim_norms[0], bfs_lmn[0], bfs_nprim[0],
        bfs_coeffs, bfs_prim_norms, bfs_expnts, a, b, c, d, Z[0], coordsBohrs[0], natoms,
        lower_tri, upper_tri, both_tri_symm, both_tri_nonsymm,
        sqrt_ints4c2e_diag, isSchwarz, dV_r)

    cp_stream.synchronize()
    cp.cuda.Stream.null.synchronize()
    return dV_r


@cuda.jit(fastmath=True, cache=True, device=True)
def primitive_nuc_single_center_cuda(la, ma, na, lb, mb, nb, alphaik, alphajk,
                                     Ix, Iy, Iz, Jx, Jy, Jz, Rcx, Rcy, Rcz, Zc):
    # Single primitive, single-nucleus nuclear-attraction integral, matching the
    # CPU primitive_nuc_single_center (returns tempfac * sum_Vl, WITHOUT the
    # contraction/normalization prefactors). Local arrays are sized for an
    # angular momentum shifted by +1 above j shells (8 + 7 + 1 = 16 -> 18).
    if la < 0 or ma < 0 or na < 0 or lb < 0 or mb < 0 or nb < 0:
        return 0.0

    PIx2 = 6.283185307179586
    TAYLOR_THRESHOLD = -25.0

    IJx = Ix - Jx
    IJy = Iy - Jy
    IJz = Iz - Jz
    IJsq = IJx * IJx + IJy * IJy + IJz * IJz
    gamma = alphaik + alphajk
    gamma_inv = 1.0 / gamma
    screenfactor = math.exp(-alphaik * alphajk * gamma_inv * IJsq)
    if screenfactor < 1.0e-12:
        return 0.0

    epsilon = 0.25 * gamma_inv
    Px = (alphaik * Ix + alphajk * Jx) * gamma_inv
    Py = (alphaik * Iy + alphajk * Jy) * gamma_inv
    Pz = (alphaik * Iz + alphajk * Jz) * gamma_inv
    PIx_ = Px - Ix
    PIy_ = Py - Iy
    PIz_ = Pz - Iz
    PJx_ = Px - Jx
    PJy_ = Py - Jy
    PJz_ = Pz - Jz
    PCx = Px - Rcx
    PCy = Py - Rcy
    PCz = Pz - Rcz
    tempfac = -Zc * (PIx2 * gamma_inv) * screenfactor

    facl = cuda.local.array((18), numba.float64)
    facm = cuda.local.array((18), numba.float64)
    facn = cuda.local.array((18), numba.float64)
    F_ = cuda.local.array((50), numba.float64)
    epsilonl = cuda.local.array((10), numba.float64)
    vmsj = cuda.local.array((18, 10, 10), numba.float64)
    vntk = cuda.local.array((18, 10, 10), numba.float64)

    max_l = la + lb
    max_m = ma + mb
    max_n = na + nb

    for l in range(max_l + 1):
        facl[l] = c2k(l, la, lb, PIx_, PJx_)
    for m in range(max_m + 1):
        facm[m] = c2k(m, ma, mb, PIy_, PJy_)
    for n in range(max_n + 1):
        facn[n] = c2k(n, na, nb, PIz_, PJz_)

    maxlmn = max(max_l, max_m, max_n)
    for li in range(maxlmn // 2 + 1):
        epsilonl[li] = epsilon**li

    temp_gamma_sum_PCsq = gamma * (PCx * PCx + PCy * PCy + PCz * PCz)
    if -temp_gamma_sum_PCsq < TAYLOR_THRESHOLD:
        hyp0minus_ = hyp0minus(-temp_gamma_sum_PCsq)
    else:
        hyp0minus_ = 0.0

    for li in range(max_l + max_m + max_n + 1):
        F_[li] = Fboys(li, temp_gamma_sum_PCsq, hyp0minus_)

    for m in range(max_m + 1):
        for s in range(m // 2 + 1):
            for j1 in range((m - 2 * s) // 2 + 1):
                vmsj[m, s, j1] = vlriPartial(PCy, m, s, j1) * epsilonl[s + j1] * facm[m]
    for n in range(max_n + 1):
        for t in range(n // 2 + 1):
            for k in range((n - 2 * t) // 2 + 1):
                vntk[n, t, k] = vlriPartial(PCz, n, t, k) * epsilonl[t + k] * facn[n]

    sum_Vl = 0.0
    for l in range(max_l + 1):
        for r in range(l // 2 + 1):
            for i1 in range((l - 2 * r) // 2 + 1):
                v_lri = vlriPartial(PCx, l, r, i1) * epsilonl[r + i1] * facl[l]
                sum_Vm = 0.0
                for m in range(max_m + 1):
                    for s in range(m // 2 + 1):
                        for j1 in range((m - 2 * s) // 2 + 1):
                            v_msj = vmsj[m, s, j1]
                            sum_Vn = 0.0
                            for n in range(max_n + 1):
                                for t in range(n // 2 + 1):
                                    for k in range((n - 2 * t) // 2 + 1):
                                        v_ntk = vntk[n, t, k]
                                        F = F_[l + m + n - 2 * (r + s + t) - (i1 + j1 + k)]
                                        sum_Vn += v_ntk * F
                            sum_Vm += v_msj * sum_Vn
                sum_Vl += v_lri * sum_Vm

    return tempfac * sum_Vl


@cuda.jit(fastmath=True, cache=True, device=True)
def primitive_nuc_center_grad_cuda(lax, lay, laz, lbx, lby, lbz, alpha_left, alpha_right,
                                   Lx, Ly, Lz, Rx, Ry, Rz, Rcx, Rcy, Rcz, Zc):
    # Gradient w.r.t. the LEFT center (the one with angular momenta lax,lay,laz
    # and exponent alpha_left, located at L). Mirrors the CPU
    # primitive_nuc_center_grad. Returns the 3-tuple (gx, gy, gz).
    # x
    tA = 0.0
    if lax > 0:
        tA = -lax * primitive_nuc_single_center_cuda(lax - 1, lay, laz, lbx, lby, lbz,
                                                     alpha_left, alpha_right, Lx, Ly, Lz, Rx, Ry, Rz, Rcx, Rcy, Rcz, Zc)
    tB = 2.0 * alpha_left * primitive_nuc_single_center_cuda(lax + 1, lay, laz, lbx, lby, lbz,
                                                            alpha_left, alpha_right, Lx, Ly, Lz, Rx, Ry, Rz, Rcx, Rcy, Rcz, Zc)
    gx = tA + tB
    # y
    tA = 0.0
    if lay > 0:
        tA = -lay * primitive_nuc_single_center_cuda(lax, lay - 1, laz, lbx, lby, lbz,
                                                     alpha_left, alpha_right, Lx, Ly, Lz, Rx, Ry, Rz, Rcx, Rcy, Rcz, Zc)
    tB = 2.0 * alpha_left * primitive_nuc_single_center_cuda(lax, lay + 1, laz, lbx, lby, lbz,
                                                            alpha_left, alpha_right, Lx, Ly, Lz, Rx, Ry, Rz, Rcx, Rcy, Rcz, Zc)
    gy = tA + tB
    # z
    tA = 0.0
    if laz > 0:
        tA = -laz * primitive_nuc_single_center_cuda(lax, lay, laz - 1, lbx, lby, lbz,
                                                     alpha_left, alpha_right, Lx, Ly, Lz, Rx, Ry, Rz, Rcx, Rcy, Rcz, Zc)
    tB = 2.0 * alpha_left * primitive_nuc_single_center_cuda(lax, lay, laz + 1, lbx, lby, lbz,
                                                            alpha_left, alpha_right, Lx, Ly, Lz, Rx, Ry, Rz, Rcx, Rcy, Rcz, Zc)
    gz = tA + tB
    return gx, gy, gz


@cuda.jit(fastmath=True, cache=True, max_registers=64)
def nuc_mat_grad_r_symm_internal_cuda(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts, start_row, end_row, start_col, end_col, Z, coordsMol, natoms, lower_tri, upper_tri, both_tri_symm, both_tri_nonsymm, sqrt_ints4c2e_diag, isSchwarz, out):
    i, j = cuda.grid(2)
    if i >= start_row and i < end_row and j >= start_col and j < end_col:
        if not (lower_tri or upper_tri or (both_tri_symm and j <= i) or both_tri_nonsymm):
            return
        if isSchwarz:
            sqrt_ij = sqrt_ints4c2e_diag[i, j]
            if sqrt_ij * sqrt_ij < 1e-13:
                return

        I = bfs_coords[i]
        Ni = bfs_contr_prim_norms[i]
        lmni = bfs_lmn[i]
        J = bfs_coords[j]
        Nj = bfs_contr_prim_norms[j]
        lmnj = bfs_lmn[j]

        la = lmni[0]
        ma = lmni[1]
        na = lmni[2]
        lb = lmnj[0]
        mb = lmnj[1]
        nb = lmnj[2]

        bra0 = 0.0
        bra1 = 0.0
        bra2 = 0.0
        ket0 = 0.0
        ket1 = 0.0
        ket2 = 0.0

        IJsq = (I[0] - J[0])**2 + (I[1] - J[1])**2 + (I[2] - J[2])**2

        for ik in range(bfs_nprim[i]):
            alphaik = bfs_expnts[i][ik]
            dik = bfs_coeffs[i][ik]
            Nik = bfs_prim_norms[i][ik]
            temp_NiNjNikdik = Ni * Nj * Nik * dik
            for jk in range(bfs_nprim[j]):
                alphajk = bfs_expnts[j][jk]
                gamma_inv = 1.0 / (alphaik + alphajk)
                screenfactor = math.exp(-alphaik * IJsq * alphajk * gamma_inv)
                if screenfactor < 1.0e-8:
                    continue
                prefactor = temp_NiNjNikdik * bfs_coeffs[j][jk] * bfs_prim_norms[j][jk]
                if abs(prefactor) < 1.0e-8:
                    continue

                for iatom in range(natoms):
                    Rc = coordsMol[iatom]
                    Zc = Z[iatom]

                    gx, gy, gz = primitive_nuc_center_grad_cuda(
                        la, ma, na, lb, mb, nb, alphaik, alphajk,
                        I[0], I[1], I[2], J[0], J[1], J[2], Rc[0], Rc[1], Rc[2], Zc)
                    bra0 += prefactor * gx
                    bra1 += prefactor * gy
                    bra2 += prefactor * gz

                    if both_tri_symm:
                        # Derivative w.r.t. the center of bf j (roles swapped)
                        kx, ky, kz = primitive_nuc_center_grad_cuda(
                            lb, mb, nb, la, ma, na, alphajk, alphaik,
                            J[0], J[1], J[2], I[0], I[1], I[2], Rc[0], Rc[1], Rc[2], Zc)
                        ket0 += prefactor * kx
                        ket1 += prefactor * ky
                        ket2 += prefactor * kz

        out[0, i - start_row, j - start_col] = bra0
        out[1, i - start_row, j - start_col] = bra1
        out[2, i - start_row, j - start_col] = bra2
        if both_tri_symm:
            out[0, j - start_col, i - start_row] = ket0
            out[1, j - start_col, i - start_row] = ket1
            out[2, j - start_col, i - start_row] = ket2
