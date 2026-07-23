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


def kin_mat_grad_r_symm_cupy(basis, slice=None, cp_stream=None):
    """
    GPU (CuPy/Numba-CUDA) counterpart of :func:`kin_mat_grad_r_symm`.

    Returns the derivative of the kinetic-energy matrix with respect to the
    *bra* basis-function center,

        dT_r[d, i, j] = d <chi_i | -1/2 nabla^2 | chi_j> / dR_{center(i), d}

    as a CuPy array of shape ``(3, num_rows, num_cols)`` — identical layout and
    sign convention to the CPU routine. The atomic gradient is obtained from it
    exactly as on the CPU (``basis_atom_gradient_from_r``).

    NOTE: Not yet executed on a CUDA device (developed on a CPU-only machine).
    It mirrors the verified ``kin_mat_symm_cupy`` launch pattern and the
    verified ``kin_mat_grad_r_symm`` math; validate against the CPU routine
    before production use.
    """
    bfs_coords = cp.array([basis.bfs_coords])
    bfs_contr_prim_norms = cp.array([basis.bfs_contr_prim_norms])
    bfs_lmn = cp.array([basis.bfs_lmn])
    bfs_nprim = cp.array([basis.bfs_nprim])

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

    dT_r = cp.zeros((3, num_rows, num_cols))

    thread_x = 24
    thread_y = 16

    if cp_stream is None:
        device = 0
        cp.cuda.Device(device).use()
        cp_stream = cp.cuda.Stream(non_blocking=True)
        nb_stream = cuda.external_stream(cp_stream.ptr)
        cp_stream.use()
    else:
        nb_stream = cuda.external_stream(cp_stream.ptr)
        cp_stream.use()

    blocks_per_grid = ((num_rows + (thread_x - 1)) // thread_x, (num_cols + (thread_y - 1)) // thread_y)
    kin_mat_grad_r_symm_internal_cuda[blocks_per_grid, (thread_x, thread_y), nb_stream](
        bfs_coords[0], bfs_contr_prim_norms[0], bfs_lmn[0], bfs_nprim[0],
        bfs_coeffs, bfs_prim_norms, bfs_expnts, a, b, c, d,
        lower_tri, upper_tri, both_tri_symm, both_tri_nonsymm, dT_r)

    cp_stream.synchronize()
    cp.cuda.Stream.null.synchronize()
    return dT_r


LOOKUP_TABLE = np.array([
    1, 1, 2, 6, 24, 120, 720, 5040, 40320,
    362880, 3628800, 39916800, 479001600,
    6227020800, 87178291200, 1307674368000,
    20922789888000, 355687428096000, 6402373705728000,
    121645100408832000, 2432902008176640000], dtype='int64')


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
    if y == 0:
        return 1
    if x == y:
        return 1
    binom = fastFactorial(x) // fastFactorial(y) // fastFactorial(x - y)
    return binom


@cuda.jit(fastmath=True, cache=True, device=True)
def doublefactorial(n):
    if n <= 0:
        return 1
    else:
        result = 1
        for i in range(n, 0, -2):
            result *= i
        return result


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
def calcS(la, lb, gamma, PA, PB):
    # Zero for negative angular momentum (used by the +/-1 and -2 shifts)
    if la < 0 or lb < 0:
        return 0.0
    temp = 0.0
    fac1 = math.sqrt(math.pi / gamma)
    fac2 = 2 * gamma
    for k in range(0, int((la + lb) / 2) + 1):
        temp += c2k(2 * k, la, lb, PA, PB) * fac1 * doublefactorial(2 * k - 1) / (fac2)**k
    return temp


@cuda.jit(fastmath=True, cache=True, device=True)
def calcKE(lix, liy, liz, ljx, ljy, ljz, gamma, alphajk, PIx, PIy, PIz, PJx, PJy, PJz):
    # Primitive kinetic-energy integral  <chi_i | -1/2 nabla_j^2 | chi_j>
    # (without contraction/normalization prefactors and screen factor), with
    # the Laplacian acting on the ket. Same formula as the kinetic matrix; the
    # gradient simply calls this with the bra angular momenta shifted by +/-1.
    fac3 = 2 * (ljx + ljy + ljz) + 3
    fac4 = ljx * (ljx - 1)
    fac5 = ljy * (ljy - 1)
    fac6 = ljz * (ljz - 1)

    Sx1 = calcS(lix, ljx, gamma, PIx, PJx)
    Sy1 = calcS(liy, ljy, gamma, PIy, PJy)
    Sz1 = calcS(liz, ljz, gamma, PIz, PJz)
    overlap1 = Sx1 * Sy1 * Sz1

    overlap2 = calcS(lix, ljx + 2, gamma, PIx, PJx) * Sy1 * Sz1
    overlap3 = Sx1 * calcS(liy, ljy + 2, gamma, PIy, PJy) * Sz1
    overlap4 = Sx1 * Sy1 * calcS(liz, ljz + 2, gamma, PIz, PJz)

    overlap5 = calcS(lix, ljx - 2, gamma, PIx, PJx) * Sy1 * Sz1
    overlap6 = Sx1 * calcS(liy, ljy - 2, gamma, PIy, PJy) * Sz1
    overlap7 = Sx1 * Sy1 * calcS(liz, ljz - 2, gamma, PIz, PJz)

    part1 = overlap1 * alphajk * fac3
    part2 = 2 * alphajk * alphajk * (overlap2 + overlap3 + overlap4)
    part3 = fac4 * overlap5
    part4 = fac5 * overlap6
    part5 = fac6 * overlap7

    return part1 - part2 - 0.5 * (part3 + part4 + part5)


@cuda.jit(fastmath=True, cache=True, max_registers=60)
def kin_mat_grad_r_symm_internal_cuda(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts, start_row, end_row, start_col, end_col, lower_tri, upper_tri, both_tri_symm, both_tri_nonsymm, out):
    i, j = cuda.grid(2)
    if i >= start_row and i < end_row and j >= start_col and j < end_col:
        if lower_tri or upper_tri or (both_tri_symm and j <= i) or both_tri_nonsymm:
            I = bfs_coords[i]
            Ni = bfs_contr_prim_norms[i]
            lmni = bfs_lmn[i]
            J = bfs_coords[j]
            Nj = bfs_contr_prim_norms[j]
            lmnj = bfs_lmn[j]

            IJ = cuda.local.array((3), numba.float64)
            P = cuda.local.array((3), numba.float64)
            PI = cuda.local.array((3), numba.float64)
            PJ = cuda.local.array((3), numba.float64)
            result = cuda.local.array((3), numba.float64)
            IJ[0] = I[0] - J[0]
            IJ[1] = I[1] - J[1]
            IJ[2] = I[2] - J[2]
            fac1 = IJ[0]**2 + IJ[1]**2 + IJ[2]**2
            fac2 = Ni * Nj
            result[0] = 0.0
            result[1] = 0.0
            result[2] = 0.0

            lax = lmni[0]
            lay = lmni[1]
            laz = lmni[2]
            lbx = lmnj[0]
            lby = lmnj[1]
            lbz = lmnj[2]

            for ik in range(bfs_nprim[i]):
                alphaik = bfs_expnts[i][ik]
                dik = bfs_coeffs[i][ik]
                Nik = bfs_prim_norms[i][ik]
                for jk in range(bfs_nprim[j]):
                    alphajk = bfs_expnts[j][jk]
                    gamma = alphaik + alphajk
                    temp_1 = math.exp(-alphaik * alphajk / gamma * fac1)
                    if abs(temp_1) < 1.0e-9:
                        continue

                    djk = bfs_coeffs[j][jk]
                    Njk = bfs_prim_norms[j][jk]

                    P[0] = (alphaik * I[0] + alphajk * J[0]) / gamma
                    P[1] = (alphaik * I[1] + alphajk * J[1]) / gamma
                    P[2] = (alphaik * I[2] + alphajk * J[2]) / gamma
                    PI[0] = P[0] - I[0]
                    PI[1] = P[1] - I[1]
                    PI[2] = P[2] - I[2]
                    PJ[0] = P[0] - J[0]
                    PJ[1] = P[1] - J[1]
                    PJ[2] = P[2] - J[2]

                    temp = dik * djk * Nik * Njk * fac2 * temp_1

                    # x derivative of bra center
                    keA = calcKE(lax - 1, lay, laz, lbx, lby, lbz, gamma, alphajk, PI[0], PI[1], PI[2], PJ[0], PJ[1], PJ[2])
                    keB = calcKE(lax + 1, lay, laz, lbx, lby, lbz, gamma, alphajk, PI[0], PI[1], PI[2], PJ[0], PJ[1], PJ[2])
                    result[0] += -lax * temp * keA + 2 * alphaik * temp * keB

                    # y derivative of bra center
                    keA = calcKE(lax, lay - 1, laz, lbx, lby, lbz, gamma, alphajk, PI[0], PI[1], PI[2], PJ[0], PJ[1], PJ[2])
                    keB = calcKE(lax, lay + 1, laz, lbx, lby, lbz, gamma, alphajk, PI[0], PI[1], PI[2], PJ[0], PJ[1], PJ[2])
                    result[1] += -lay * temp * keA + 2 * alphaik * temp * keB

                    # z derivative of bra center
                    keA = calcKE(lax, lay, laz - 1, lbx, lby, lbz, gamma, alphajk, PI[0], PI[1], PI[2], PJ[0], PJ[1], PJ[2])
                    keB = calcKE(lax, lay, laz + 1, lbx, lby, lbz, gamma, alphajk, PI[0], PI[1], PI[2], PJ[0], PJ[1], PJ[2])
                    result[2] += -laz * temp * keA + 2 * alphaik * temp * keB

            for direction in range(3):
                out[direction, i - start_row, j - start_col] = result[direction]
                if both_tri_symm:
                    out[direction, j - start_col, i - start_row] = -result[direction]
