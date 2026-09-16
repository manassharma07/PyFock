"""GPU (Numba-CUDA) contracted 3c2e nuclear gradient of the density-fitted Coulomb term.

Device port of :func:`pyfock.Integrals.rys_3c2e_grad_contract`. It computes the same quantity,

    grad[iatom, xyz] = sum_{ij, P} D_ij c_P d(ij|P)/dR_{iatom, xyz}

with one thread per (shell pair AB, auxiliary shell C) triplet.

The CPU kernel keeps two shell-block scratch tensors -- the five shifted 1D integrals and the six
derivative components it calls dblock -- which together are far too large for per-thread storage on
the device. Neither is actually needed: the shift tables are rebuilt for every root anyway, so
forming them inside the accumulation loop costs nothing, and the density/coefficient weight
D_ij c_P is constant over primitives and roots, so it can be folded into the root weight and the six
components accumulated as scalars. What is left per thread is the three 1D Rys tables plus six
accumulators.
"""
import math

import numpy as np
import numba
from numba import cuda

try:
    import cupy as cp
except Exception:                                  # pragma: no cover - CPU-only install
    cp = None

from .rys_helpers_cuda import Recur_3c2e, Roots, DATA_X, DATA_W
from .rys_3c2e_symm import _pack_basis
from .cuda_stream import gradient_stream
from .rys_3c2e_grad_contract import _shell_bounds

__all__ = ['rys_3c2e_grad_contract_cupy']

# Binomial coefficients; the CUDA copy in rys_helpers_cuda only goes up to l = 5.
_COMB = np.array([[float(math.comb(n, k)) if k <= n else 0.0 for k in range(16)]
                  for n in range(16)], dtype=np.float64)

# Every thread atomically accumulates into natoms*3 doubles. Spreading the blocks over a handful of
# copies keeps same-address contention off the critical path; the copies are summed afterwards.
_NBINS = 64

_KERNEL_CACHE = {}


def _get_kernel(g_rows, g_cols, nroots_max):
    """Compile (once per shape) a kernel whose scratch arrays are sized for this basis pair.

    The Rys tables are indexed with runtime angular momenta but must be declared with compile-time
    extents, so the kernel is specialised on the largest shell triplet the basis can produce rather
    than on a worst case that would triple the per-thread local memory traffic.
    """
    key = (g_rows, g_cols, nroots_max)
    kernel = _KERNEL_CACHE.get(key)
    if kernel is not None:
        return kernel

    G_ROWS = g_rows
    G_COLS = g_cols
    NROOTS = nroots_max

    @cuda.jit(fastmath=True, cache=False)
    def _kernel(task_list, nshells_aux, ab_shell_a, ab_shell_b,
                bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim,
                bfs_coeffs, bfs_prim_norms, bfs_expnts,
                shell_l, shell_bfs_offset, bfs_nbfshell, bfs_atoms,
                aux_bfs_coords, aux_bfs_contr_prim_norms, aux_bfs_lmn, aux_bfs_nprim,
                aux_bfs_coeffs, aux_bfs_prim_norms, aux_bfs_expnts,
                aux_shell_l, aux_shell_bfs_offset, aux_bfs_nbfshell, aux_bfs_atoms,
                dmat, df_coeff, DATA_X_, DATA_W_, grad_partial):
        comb = cuda.const.array_like(_COMB)

        itask = cuda.grid(1)
        if itask >= task_list.shape[0]:
            return

        task = task_list[itask]
        ab_idx = task // nshells_aux
        ksh = task - ab_idx * nshells_aux
        ish = ab_shell_a[ab_idx]
        jsh = ab_shell_b[ab_idx]

        bf_a_start = shell_bfs_offset[ish]
        bf_b_start = shell_bfs_offset[jsh]
        bf_c_start = aux_shell_bfs_offset[ksh]

        nbf_a = bfs_nbfshell[ish]
        nbf_b = bfs_nbfshell[jsh]
        nbf_c = aux_bfs_nbfshell[ksh]

        atom_a = bfs_atoms[bf_a_start]
        atom_b = bfs_atoms[bf_b_start]
        atom_c = aux_bfs_atoms[bf_c_start]

        la_shell = shell_l[ish]
        lb_shell = shell_l[jsh]
        lc_shell = aux_shell_l[ksh]
        bra_order = la_shell + lb_shell + 1        # +1 for the derivative
        aux_order = lc_shell + 1
        nroots = (la_shell + lb_shell + lc_shell + 1) // 2 + 1

        nprim_a = bfs_nprim[bf_a_start]
        nprim_b = bfs_nprim[bf_b_start]
        nprim_c = aux_bfs_nprim[bf_c_start]

        ax0 = bfs_coords[bf_a_start, 0]
        ay0 = bfs_coords[bf_a_start, 1]
        az0 = bfs_coords[bf_a_start, 2]
        bx0 = bfs_coords[bf_b_start, 0]
        by0 = bfs_coords[bf_b_start, 1]
        bz0 = bfs_coords[bf_b_start, 2]
        cx0 = aux_bfs_coords[bf_c_start, 0]
        cy0 = aux_bfs_coords[bf_c_start, 1]
        cz0 = aux_bfs_coords[bf_c_start, 2]

        xij0 = ax0 - bx0
        xij1 = ay0 - by0
        xij2 = az0 - bz0
        ijsq = xij0 * xij0 + xij1 * xij1 + xij2 * xij2

        roots = cuda.local.array(NROOTS, numba.float64)
        weights = cuda.local.array(NROOTS, numba.float64)
        gx = cuda.local.array((G_ROWS, G_COLS), numba.float64)
        gy = cuda.local.array((G_ROWS, G_COLS), numba.float64)
        gz = cuda.local.array((G_ROWS, G_COLS), numba.float64)

        ga_x = 0.0
        ga_y = 0.0
        ga_z = 0.0
        gc_x = 0.0
        gc_y = 0.0
        gc_z = 0.0

        pair_factor = 2.0 if ish != jsh else 1.0
        pi = 3.141592653589793

        for iprim_a in range(nprim_a):
            alpha = bfs_expnts[bf_a_start, iprim_a]
            two_alpha = 2.0 * alpha
            for iprim_b in range(nprim_b):
                beta = bfs_expnts[bf_b_start, iprim_b]
                gamma_p = alpha + beta
                inv_gamma_p = 1.0 / gamma_p
                if math.exp(-alpha * beta * inv_gamma_p * ijsq) < 1.0e-10:
                    continue

                px = (alpha * ax0 + beta * bx0) * inv_gamma_p
                py = (alpha * ay0 + beta * by0) * inv_gamma_p
                pz = (alpha * az0 + beta * bz0) * inv_gamma_p

                pqx = px - cx0
                pqy = py - cy0
                pqz = pz - cz0
                pqsq = pqx * pqx + pqy * pqy + pqz * pqz

                for iprim_c in range(nprim_c):
                    gamma_q = aux_bfs_expnts[bf_c_start, iprim_c]
                    two_gamma_q = 2.0 * gamma_q
                    rho = gamma_p * gamma_q / (gamma_p + gamma_q)
                    x = rho * pqsq
                    gamma_pq_sqrt = math.sqrt(gamma_p * gamma_q)

                    Roots(nroots, x, DATA_X_, DATA_W_, roots, weights)

                    rys_prefactor = 2.0 * math.sqrt(rho / pi)
                    for iroot in range(nroots):
                        root = roots[iroot]
                        Recur_3c2e(gx, root, bra_order, 0, aux_order, 0,
                                   ax0, bx0, cx0, 0.0, alpha, beta, gamma_q, 0.0,
                                   gamma_p, gamma_q, alpha * beta, gamma_pq_sqrt)
                        Recur_3c2e(gy, root, bra_order, 0, aux_order, 0,
                                   ay0, by0, cy0, 0.0, alpha, beta, gamma_q, 0.0,
                                   gamma_p, gamma_q, alpha * beta, gamma_pq_sqrt)
                        Recur_3c2e(gz, root, bra_order, 0, aux_order, 0,
                                   az0, bz0, cz0, 0.0, alpha, beta, gamma_q, 0.0,
                                   gamma_p, gamma_q, alpha * beta, gamma_pq_sqrt)

                        root_weight = rys_prefactor * weights[iroot]

                        for ia in range(nbf_a):
                            ibf_a = bf_a_start + ia
                            axl = bfs_lmn[ibf_a, 0]
                            ayl = bfs_lmn[ibf_a, 1]
                            azl = bfs_lmn[ibf_a, 2]
                            ca = (bfs_contr_prim_norms[ibf_a] * bfs_coeffs[ibf_a, iprim_a]
                                  * bfs_prim_norms[ibf_a, iprim_a])
                            for ib in range(nbf_b):
                                ibf_b = bf_b_start + ib
                                bxl = bfs_lmn[ibf_b, 0]
                                byl = bfs_lmn[ibf_b, 1]
                                bzl = bfs_lmn[ibf_b, 2]
                                cb = (ca * bfs_contr_prim_norms[ibf_b]
                                      * bfs_coeffs[ibf_b, iprim_b] * bfs_prim_norms[ibf_b, iprim_b])
                                dm_ab = pair_factor * dmat[ibf_a, ibf_b] * cb * root_weight
                                for ic in range(nbf_c):
                                    ibf_c = bf_c_start + ic
                                    cxl = aux_bfs_lmn[ibf_c, 0]
                                    cyl = aux_bfs_lmn[ibf_c, 1]
                                    czl = aux_bfs_lmn[ibf_c, 2]
                                    w = (dm_ab * df_coeff[ibf_c] * aux_bfs_contr_prim_norms[ibf_c]
                                         * aux_bfs_coeffs[ibf_c, iprim_c]
                                         * aux_bfs_prim_norms[ibf_c, iprim_c])

                                    # ---- x ----
                                    s = 0.0
                                    sap = 0.0
                                    sam = 0.0
                                    scp = 0.0
                                    scm = 0.0
                                    for n in range(bxl + 1):
                                        t = comb[bxl, n] * xij0 ** (bxl - n)
                                        row = n + axl
                                        s += t * gx[row, cxl]
                                        sap += t * gx[row + 1, cxl]
                                        scp += t * gx[row, cxl + 1]
                                        if axl > 0:
                                            sam += t * gx[row - 1, cxl]
                                        if cxl > 0:
                                            scm += t * gx[row, cxl - 1]
                                    sx = s
                                    dax = two_alpha * sap - axl * sam
                                    dcx = two_gamma_q * scp - cxl * scm

                                    # ---- y ----
                                    s = 0.0
                                    sap = 0.0
                                    sam = 0.0
                                    scp = 0.0
                                    scm = 0.0
                                    for n in range(byl + 1):
                                        t = comb[byl, n] * xij1 ** (byl - n)
                                        row = n + ayl
                                        s += t * gy[row, cyl]
                                        sap += t * gy[row + 1, cyl]
                                        scp += t * gy[row, cyl + 1]
                                        if ayl > 0:
                                            sam += t * gy[row - 1, cyl]
                                        if cyl > 0:
                                            scm += t * gy[row, cyl - 1]
                                    sy = s
                                    day = two_alpha * sap - ayl * sam
                                    dcy = two_gamma_q * scp - cyl * scm

                                    # ---- z ----
                                    s = 0.0
                                    sap = 0.0
                                    sam = 0.0
                                    scp = 0.0
                                    scm = 0.0
                                    for n in range(bzl + 1):
                                        t = comb[bzl, n] * xij2 ** (bzl - n)
                                        row = n + azl
                                        s += t * gz[row, czl]
                                        sap += t * gz[row + 1, czl]
                                        scp += t * gz[row, czl + 1]
                                        if azl > 0:
                                            sam += t * gz[row - 1, czl]
                                        if czl > 0:
                                            scm += t * gz[row, czl - 1]
                                    sz = s
                                    daz = two_alpha * sap - azl * sam
                                    dcz = two_gamma_q * scp - czl * scm

                                    syz = sy * sz
                                    sxz = sx * sz
                                    sxy = sx * sy
                                    ga_x += w * dax * syz
                                    ga_y += w * day * sxz
                                    ga_z += w * daz * sxy
                                    gc_x += w * dcx * syz
                                    gc_y += w * dcy * sxz
                                    gc_z += w * dcz * sxy

        ibin = cuda.blockIdx.x % _NBINS
        cuda.atomic.add(grad_partial, (ibin, atom_a, 0), ga_x)
        cuda.atomic.add(grad_partial, (ibin, atom_a, 1), ga_y)
        cuda.atomic.add(grad_partial, (ibin, atom_a, 2), ga_z)
        cuda.atomic.add(grad_partial, (ibin, atom_c, 0), gc_x)
        cuda.atomic.add(grad_partial, (ibin, atom_c, 1), gc_y)
        cuda.atomic.add(grad_partial, (ibin, atom_c, 2), gc_z)
        # Translational invariance: d/dB = -(d/dA + d/dC)
        cuda.atomic.add(grad_partial, (ibin, atom_b, 0), -(ga_x + gc_x))
        cuda.atomic.add(grad_partial, (ibin, atom_b, 1), -(ga_y + gc_y))
        cuda.atomic.add(grad_partial, (ibin, atom_b, 2), -(ga_z + gc_z))

    _KERNEL_CACHE[key] = _kernel
    return _kernel


def rys_3c2e_grad_contract_cupy(basis, auxbasis, dmat, df_coeff, schwarz=True,
                                threshold_schwarz=1e-11, sqrt_ints4c2e_diag=None,
                                sqrt_diag_ints2c2e=None, cp_stream=None):
    """GPU counterpart of :func:`rys_3c2e_grad_contract`; see it for the definition.

    ``sqrt_ints4c2e_diag`` / ``sqrt_diag_ints2c2e`` let the caller pass in Schwarz diagonals it has
    already built, since the same two arrays are needed by the fitting-coefficient step.

    Returns a NumPy ``(natoms, 3)`` array -- the gradient is natoms*3 numbers, so there is nothing to
    gain by leaving it on the device.
    """
    if cp is None:
        raise RuntimeError('CuPy is required for rys_3c2e_grad_contract_cupy.')

    (bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms,
     bfs_expnts, shell_l, shell_bfs_offset, bfs_nbfshell) = _pack_basis(basis)
    (aux_bfs_coords, aux_bfs_contr_prim_norms, aux_bfs_lmn, aux_bfs_nprim, aux_bfs_coeffs,
     aux_bfs_prim_norms, aux_bfs_expnts, aux_shell_l, aux_shell_bfs_offset,
     aux_bfs_nbfshell) = _pack_basis(auxbasis)

    bfs_atoms = np.array(basis.bfs_atoms, dtype=np.int32)
    aux_bfs_atoms = np.array(auxbasis.bfs_atoms, dtype=np.int32)
    natoms = int(max(bfs_atoms.max(), aux_bfs_atoms.max())) + 1

    nshells = len(basis.shells)
    nshells_aux = len(auxbasis.shells)

    max_l_bra = int(shell_l.max())
    max_l_aux = int(aux_shell_l.max())
    nroots_max = (2 * max_l_bra + max_l_aux + 1) // 2 + 1
    if nroots_max > 10:
        raise NotImplementedError('rys_3c2e_grad_contract_cupy supports Rys orders up to 10 only.')
    g_rows = 2 * max_l_bra + 2      # bra order (la+lb+1) plus one for the [0, order] range
    g_cols = max_l_aux + 2          # aux order (lc+1) plus one

    dmat = np.ascontiguousarray(dmat, dtype=np.float64)
    df_coeff = np.ascontiguousarray(df_coeff, dtype=np.float64)

    ab_shell_a = np.empty(nshells * (nshells + 1) // 2, dtype=np.int32)
    ab_shell_b = np.empty_like(ab_shell_a)
    idx = 0
    for ish in range(nshells):
        for jsh in range(ish + 1):
            ab_shell_a[idx] = ish
            ab_shell_b[idx] = jsh
            idx += 1
    n_ab = ab_shell_a.shape[0]

    pair_bound = np.ones(n_ab)
    aux_bound = np.ones(nshells_aux)
    if schwarz:
        if sqrt_ints4c2e_diag is None:
            from .schwarz_helpers import eri_4c2e_diag
            sqrt_ints4c2e_diag = np.sqrt(np.abs(eri_4c2e_diag(basis)))
        if sqrt_diag_ints2c2e is None:
            from .rys_2c2e_diag import rys_2c2e_diag
            sqrt_diag_ints2c2e = np.sqrt(np.abs(rys_2c2e_diag(auxbasis)))
        (shell_pair_bound, aux_shell_bound, dm_shell_pair_max,
         aux_shell_coeff_max) = _shell_bounds(
            nshells, nshells_aux, shell_bfs_offset, bfs_nbfshell,
            aux_shell_bfs_offset, aux_bfs_nbfshell,
            np.asarray(sqrt_ints4c2e_diag), np.asarray(sqrt_diag_ints2c2e),
            np.abs(dmat), np.abs(df_coeff))
        pair_bound = (shell_pair_bound * dm_shell_pair_max)[ab_shell_a, ab_shell_b]
        aux_bound = aux_shell_bound * aux_shell_coeff_max

    if cp_stream is None:
        cp_stream, nb_stream = gradient_stream()
    else:
        nb_stream = cuda.external_stream(cp_stream.ptr)

    with cp_stream:
        d = {name: cp.asarray(value) for name, value in (
            ('ab_shell_a', ab_shell_a), ('ab_shell_b', ab_shell_b),
            ('bfs_coords', bfs_coords), ('bfs_contr_prim_norms', bfs_contr_prim_norms),
            ('bfs_lmn', bfs_lmn), ('bfs_nprim', bfs_nprim), ('bfs_coeffs', bfs_coeffs),
            ('bfs_prim_norms', bfs_prim_norms), ('bfs_expnts', bfs_expnts),
            ('shell_l', shell_l), ('shell_bfs_offset', shell_bfs_offset),
            ('bfs_nbfshell', bfs_nbfshell), ('bfs_atoms', bfs_atoms),
            ('aux_bfs_coords', aux_bfs_coords),
            ('aux_bfs_contr_prim_norms', aux_bfs_contr_prim_norms),
            ('aux_bfs_lmn', aux_bfs_lmn), ('aux_bfs_nprim', aux_bfs_nprim),
            ('aux_bfs_coeffs', aux_bfs_coeffs), ('aux_bfs_prim_norms', aux_bfs_prim_norms),
            ('aux_bfs_expnts', aux_bfs_expnts), ('aux_shell_l', aux_shell_l),
            ('aux_shell_bfs_offset', aux_shell_bfs_offset),
            ('aux_bfs_nbfshell', aux_bfs_nbfshell), ('aux_bfs_atoms', aux_bfs_atoms),
            ('dmat', dmat), ('df_coeff', df_coeff),
            ('DATA_X', DATA_X), ('DATA_W', DATA_W))}
        grad_partial = cp.zeros((_NBINS, natoms, 3), dtype=cp.float64)

        kernel = _get_kernel(g_rows, g_cols, nroots_max)
        threads = 64
        for task_list in _task_batches(pair_bound, aux_bound, threshold_schwarz,
                                       ab_shell_a, ab_shell_b, bfs_atoms, aux_bfs_atoms,
                                       shell_l, aux_shell_l, shell_bfs_offset,
                                       aux_shell_bfs_offset, nshells_aux):
            blocks = (task_list.shape[0] + threads - 1) // threads
            kernel[blocks, threads, nb_stream](
                task_list, nshells_aux, d['ab_shell_a'], d['ab_shell_b'],
                d['bfs_coords'], d['bfs_contr_prim_norms'], d['bfs_lmn'], d['bfs_nprim'],
                d['bfs_coeffs'], d['bfs_prim_norms'], d['bfs_expnts'],
                d['shell_l'], d['shell_bfs_offset'], d['bfs_nbfshell'], d['bfs_atoms'],
                d['aux_bfs_coords'], d['aux_bfs_contr_prim_norms'], d['aux_bfs_lmn'],
                d['aux_bfs_nprim'], d['aux_bfs_coeffs'], d['aux_bfs_prim_norms'],
                d['aux_bfs_expnts'], d['aux_shell_l'], d['aux_shell_bfs_offset'],
                d['aux_bfs_nbfshell'], d['aux_bfs_atoms'],
                d['dmat'], d['df_coeff'], d['DATA_X'], d['DATA_W'], grad_partial)
        grad = cp.asnumpy(grad_partial.sum(axis=0))
    cp_stream.synchronize()
    return grad


# Upper bound on the flattened (shell pair, aux shell) task table materialised at once, in tasks.
# 32 M tasks is 256 MB of int64 indices plus the same again for the sort key.
_TASK_CHUNK = 32 * 1024 * 1024


def _task_batches(pair_bound, aux_bound, threshold, ab_shell_a, ab_shell_b,
                  bfs_atoms, aux_bfs_atoms, shell_l, aux_shell_l,
                  shell_bfs_offset, aux_shell_bfs_offset, nshells_aux):
    """Yield device arrays of surviving (ab_idx * nshells_aux + ksh) task ids, grouped by shape.

    Dropping the screened-out and identically-zero triplets up front means the kernel launches no
    thread that returns immediately, and sorting what is left by the angular momentum triple keeps
    each warp on shells of the same size: threads in a warp then agree on the root count, the
    recursion orders and the shell-block loop bounds, which is where the divergence would otherwise
    be. The table is chunked over shell pairs so its size never depends on the molecule.
    """
    n_ab = pair_bound.shape[0]
    aux_bound_d = cp.asarray(aux_bound)
    # Class key: (la, lb, lc) fully determines the loop bounds and the local array extents used.
    lc_key = cp.asarray(aux_shell_l.astype(np.int64))
    lab_key = cp.asarray((shell_l[ab_shell_a].astype(np.int64) * 8
                          + shell_l[ab_shell_b].astype(np.int64)) * 8)
    # A triplet with all three centers on one atom has zero total derivative.
    atom_ab = cp.asarray(np.where(bfs_atoms[shell_bfs_offset[ab_shell_a]]
                                  == bfs_atoms[shell_bfs_offset[ab_shell_b]],
                                  bfs_atoms[shell_bfs_offset[ab_shell_a]], -1).astype(np.int64))
    atom_c = cp.asarray(aux_bfs_atoms[aux_shell_bfs_offset].astype(np.int64))

    chunk = max(1, min(n_ab, _TASK_CHUNK // max(1, nshells_aux)))
    pair_bound_d = cp.asarray(pair_bound)
    for lo in range(0, n_ab, chunk):
        hi = min(lo + chunk, n_ab)
        keep = (pair_bound_d[lo:hi, None] * aux_bound_d[None, :] >= threshold)
        keep &= atom_ab[lo:hi, None] != atom_c[None, :]
        flat = cp.flatnonzero(keep.ravel())
        if flat.size == 0:
            continue
        key = lab_key[lo + flat // nshells_aux] + lc_key[flat % nshells_aux]
        order = cp.argsort(key, kind='stable')
        yield (flat[order] + lo * nshells_aux)
