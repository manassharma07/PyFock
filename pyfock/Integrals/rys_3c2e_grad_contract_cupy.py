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

The same kernel with a general weight per function pair and auxiliary function,
``sum_{ij, P} W^P_ij d(ij|P)/dR`` (:class:`RowsGradContext`), is the three-center part of the RI
exchange gradient, the device counterpart of
:func:`pyfock.Integrals.df_algo12_grad.grad_contract_rows`: the weights are the rows of a
``(nrows, ncols)`` device array, one row per function pair, its columns a range of whole Cartesian
auxiliary shells, so that the caller can produce them in column chunks of bounded size.
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
from .df_algo10_helpers import STRICT_PAIR_CUTOFF
from .df_algo12_helpers_cupy import _pp_branch_dense

__all__ = ['rys_3c2e_grad_contract_cupy', 'RowsGradContext', 'rys_3c2e_grad_contract_rows_cupy']

# Binomial coefficients; the CUDA copy in rys_helpers_cuda only goes up to l = 5.
_COMB = np.array([[float(math.comb(n, k)) if k <= n else 0.0 for k in range(16)]
                  for n in range(16)], dtype=np.float64)

# Every thread atomically accumulates into natoms*3 doubles. Spreading the blocks over a handful of
# copies keeps same-address contention off the critical path; the copies are summed afterwards.
_NBINS = 64

_KERNEL_CACHE = {}


def _nb(array):
    """Numba view of a CuPy array without Numba's implicit synchronization of the current stream."""
    return cuda.as_cuda_array(array, sync=False)


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
                dmat, df_coeff, DATA_X_, DATA_W_, use_mask, pp_off, pp_branch, ff, grad_partial):
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
        pp_base = pp_off[ab_idx] if use_mask else 0

        for iprim_a in range(nprim_a):
            alpha = bfs_expnts[bf_a_start, iprim_a]
            two_alpha = 2.0 * alpha
            for iprim_b in range(nprim_b):
                # DF_algo=12: a primitive pair whose branch is far field for this auxiliary shell is
                # handled by the multipole expansions (and one the plan drops is in neither)
                if use_mask and ff[pp_branch[pp_base + iprim_a * nprim_b + iprim_b], ksh]:
                    continue
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
                                sqrt_diag_ints2c2e=None, cp_stream=None, df12_plan=None):
    """GPU counterpart of :func:`rys_3c2e_grad_contract`; see it for the definition.

    ``sqrt_ints4c2e_diag`` / ``sqrt_diag_ints2c2e`` let the caller pass in Schwarz diagonals it has
    already built, since the same two arrays are needed by the fitting-coefficient step.

    ``df12_plan`` (from :func:`pyfock.Integrals.df_algo12_grad.build_grad_plan`) restricts the
    contraction to the near field of algorithm 12: the plan's significant (shell pair, auxiliary
    shell) blocks, without the primitive pairs that are far field for the auxiliary shell, and with
    the plan's strict pair cut-off. The far field is then
    ``df_algo12_grad.grad_contract(plan, ..., near=False)``.

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
    if df12_plan is not None and df12_plan.strict_schwarz:
        # the function pairs the plan's strict cut-off drops from every contraction
        dmat = np.where(df12_plan.sqrt_ints4c2e_diag ** 2 < STRICT_PAIR_CUTOFF, 0.0, dmat)

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

    mask = None
    if df12_plan is not None:
        plan = df12_plan
        if plan.pair_I.shape[0] != ab_shell_a.shape[0]:
            raise ValueError('df12_plan was built for a different basis')
        # Branch of every primitive pair at a dense index, with the plan's dropped pairs on a row of
        # the mask that is far field everywhere: the layout of the DF_algo=12 CUDA driver.
        nprim_pairs = np.zeros(plan.pair_I.shape[0], dtype=np.int64)
        nprim_pairs[plan.sig] = (plan.bfs_nprim[plan.shell_off[plan.pair_I[plan.sig]]]
                                 * plan.bfs_nprim[plan.shell_off[plan.pair_J[plan.sig]]])
        pp_off = np.zeros(plan.pair_I.shape[0] + 1, dtype=np.int64)
        pp_off[1:] = np.cumsum(nprim_pairs)
        pp_branch = _pp_branch_dense(plan.sig, plan.pair_I, plan.pair_J, plan.shell_off,
                                     plan.bfs_nprim, plan.bfs_expnts, plan.bfs_coords, plan.pp_group,
                                     plan.grp_branch_eff, pp_off, plan.ff_eff.shape[0])
        if pp_branch.size == 0:
            pp_branch = np.zeros(1, dtype=np.int32)
        ff_mask = np.ascontiguousarray(
            np.vstack([plan.ff_eff, np.ones((1, plan.ff_eff.shape[1]), dtype=np.bool_)]))
        mask = (pp_off, pp_branch, ff_mask, plan.Q_pair, plan.Q_aux, plan.threshold)

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
        if mask is not None:
            mask_d = tuple(cp.asarray(v) for v in mask[:5]) + (float(mask[5]),)
        else:
            mask_d = (cp.zeros(1, dtype=cp.int64), cp.zeros(1, dtype=cp.int32),
                      cp.zeros((1, 1), dtype=cp.bool_))

        kernel = _get_kernel(g_rows, g_cols, nroots_max)
        threads = 64
        for task_list in _task_batches(pair_bound, aux_bound, threshold_schwarz,
                                       ab_shell_a, ab_shell_b, bfs_atoms, aux_bfs_atoms,
                                       shell_l, aux_shell_l, shell_bfs_offset,
                                       aux_shell_bfs_offset, nshells_aux,
                                       mask=mask_d if mask is not None else None,
                                       nb_stream=nb_stream):
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
                d['dmat'], d['df_coeff'], d['DATA_X'], d['DATA_W'],
                mask is not None, mask_d[0], mask_d[1], mask_d[2], grad_partial)
        grad = cp.asnumpy(grad_partial.sum(axis=0))
    cp_stream.synchronize()
    return grad


@cuda.jit(cache=False)
def _near_field_tasks(flat, lo, nshells_aux, pp_off, pp_branch, ff, q_pair, q_aux, threshold, near):
    """``near[i] = 1`` when chunk task ``flat[i]`` is a significant DF_algo=12 block with a near-field primitive pair."""
    i = cuda.grid(1)
    if i >= flat.shape[0]:
        return
    t = flat[i]
    ab = lo + t // nshells_aux
    k = t - (t // nshells_aux) * nshells_aux
    near[i] = 0
    if q_pair[ab] * q_aux[k] <= threshold:
        return
    for q in range(pp_off[ab], pp_off[ab + 1]):
        if not ff[pp_branch[q], k]:
            near[i] = 1
            return


# Upper bound on the flattened (shell pair, aux shell) task table materialised at once, in tasks.
# 32 M tasks is 256 MB of int64 indices plus the same again for the sort key.
_TASK_CHUNK = 32 * 1024 * 1024


def _task_batches(pair_bound, aux_bound, threshold, ab_shell_a, ab_shell_b,
                  bfs_atoms, aux_bfs_atoms, shell_l, aux_shell_l,
                  shell_bfs_offset, aux_shell_bfs_offset, nshells_aux, mask=None, nb_stream=None):
    """Yield device arrays of surviving (ab_idx * nshells_aux + ksh) task ids, grouped by shape.

    Dropping the screened-out and identically-zero triplets up front means the kernel launches no
    thread that returns immediately, and sorting what is left by the angular momentum triple keeps
    each warp on shells of the same size: threads in a warp then agree on the root count, the
    recursion orders and the shell-block loop bounds, which is where the divergence would otherwise
    be. The table is chunked over shell pairs so its size never depends on the molecule. With a
    DF_algo=12 ``mask`` (device ``pp_off, pp_branch, ff, Q_pair, Q_aux`` and the plan threshold)
    only the plan's significant blocks with at least one near-field primitive pair are kept.
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
        if flat.size and mask is not None:
            near = cp.zeros(flat.size, dtype=cp.int8)
            _near_field_tasks[(flat.size + 127) // 128, 128, nb_stream](
                flat, lo, nshells_aux, mask[0], mask[1], mask[2], mask[3], mask[4], mask[5], near)
            flat = flat[near.astype(cp.bool_)]
        if flat.size == 0:
            continue
        key = lab_key[lo + flat // nshells_aux] + lc_key[flat % nshells_aux]
        order = cp.argsort(key, kind='stable')
        yield (flat[order] + lo * nshells_aux)


# ----------------------------------------------------------------------------
# General weights per function pair and auxiliary function (RI exchange)
# ----------------------------------------------------------------------------
_ROWS_KERNEL_CACHE = {}


def _get_rows_kernel(g_rows, g_cols, nroots_max):
    """:func:`_get_kernel` with the weight ``fac * W[row_of[i, j], P - col0]`` in place of ``D_ij c_P``.

    ``row_of`` is symmetric (``-1`` for a pair the energy leaves out), so the loop over every
    component pair of a diagonal shell pair visits ``(i, j)`` and ``(j, i)`` with one weight each,
    which is the factor 2 of an off-diagonal pair; off-diagonal shell pairs carry it explicitly.
    The plan's primitive-pair mask is always applied.
    """
    key = (g_rows, g_cols, nroots_max)
    kernel = _ROWS_KERNEL_CACHE.get(key)
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
                weights, row_of, col0, DATA_X_, DATA_W_, pp_off, pp_branch, ff, grad_partial):
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
        weights_r = cuda.local.array(NROOTS, numba.float64)
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
        pp_base = pp_off[ab_idx]
        c_col = bf_c_start - col0

        for iprim_a in range(nprim_a):
            alpha = bfs_expnts[bf_a_start, iprim_a]
            two_alpha = 2.0 * alpha
            for iprim_b in range(nprim_b):
                # a primitive pair the plan drops (or that is far field for this shell) is skipped
                if ff[pp_branch[pp_base + iprim_a * nprim_b + iprim_b], ksh]:
                    continue
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

                    Roots(nroots, x, DATA_X_, DATA_W_, roots, weights_r)

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

                        root_weight = rys_prefactor * weights_r[iroot]

                        for ia in range(nbf_a):
                            ibf_a = bf_a_start + ia
                            axl = bfs_lmn[ibf_a, 0]
                            ayl = bfs_lmn[ibf_a, 1]
                            azl = bfs_lmn[ibf_a, 2]
                            ca = (bfs_contr_prim_norms[ibf_a] * bfs_coeffs[ibf_a, iprim_a]
                                  * bfs_prim_norms[ibf_a, iprim_a])
                            for ib in range(nbf_b):
                                ibf_b = bf_b_start + ib
                                r = row_of[ibf_a, ibf_b]
                                if r < 0:
                                    continue
                                bxl = bfs_lmn[ibf_b, 0]
                                byl = bfs_lmn[ibf_b, 1]
                                bzl = bfs_lmn[ibf_b, 2]
                                cb = (ca * bfs_contr_prim_norms[ibf_b]
                                      * bfs_coeffs[ibf_b, iprim_b] * bfs_prim_norms[ibf_b, iprim_b])
                                dm_ab = pair_factor * cb * root_weight
                                for ic in range(nbf_c):
                                    ibf_c = bf_c_start + ic
                                    cxl = aux_bfs_lmn[ibf_c, 0]
                                    cyl = aux_bfs_lmn[ibf_c, 1]
                                    czl = aux_bfs_lmn[ibf_c, 2]
                                    w = (dm_ab * weights[r, c_col + ic] * aux_bfs_contr_prim_norms[ibf_c]
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

    _ROWS_KERNEL_CACHE[key] = _kernel
    return _kernel


@cuda.jit(cache=False)
def _rows_tasks(tasks, nshells_aux, pp_off, pp_branch, ff, q_pair, q_aux, threshold,
                ab_shell_a, ab_shell_b, shell_bfs_offset, bfs_nbfshell, aux_shell_bfs_offset,
                aux_bfs_nbfshell, weights, row_of, col0, thr_grad, keep):
    """
    ``keep[i] = 1`` when task ``tasks[i]`` (``pair * nshells_aux + aux shell``) is a significant block of
    the plan with a near-field primitive pair whose weights pass the gradient screening of
    :func:`pyfock.Integrals.df_algo12_grad.grad_contract_rows`:
    ``Q_pair Q_aux max|fac W| >= thr_grad`` over the block (``fac`` = 2 off the diagonal).
    """
    i = cuda.grid(1)
    if i >= tasks.shape[0]:
        return
    keep[i] = 0
    t = tasks[i]
    ab = t // nshells_aux
    k = t - ab * nshells_aux
    qq = q_pair[ab] * q_aux[k]
    if qq <= threshold:
        return
    near = False
    for q in range(pp_off[ab], pp_off[ab + 1]):
        if not ff[pp_branch[q], k]:
            near = True
            break
    if not near:
        return
    a0 = shell_bfs_offset[ab_shell_a[ab]]
    b0 = shell_bfs_offset[ab_shell_b[ab]]
    na = bfs_nbfshell[ab_shell_a[ab]]
    nb = bfs_nbfshell[ab_shell_b[ab]]
    c0 = aux_shell_bfs_offset[k] - col0
    nc = aux_bfs_nbfshell[k]
    wmax = 0.0
    for ia in range(na):
        for ib in range(nb):
            r = row_of[a0 + ia, b0 + ib]
            if r < 0:
                continue
            f = 1.0 if a0 + ia == b0 + ib else 2.0
            for ic in range(nc):
                v = f * abs(weights[r, c0 + ic])
                if v > wmax:
                    wmax = v
    if qq * wmax >= thr_grad:
        keep[i] = 1


class RowsGradContext:
    """
    ``sum_{ij, P} W^P_ij d(ij|P)/dR`` on the device for weights stored per function pair: the
    three-center part of the RI exchange gradient
    (:func:`pyfock.Integrals.df_algo12_grad.grad_contract_rows` on the CPU).

    The basis arrays, the plan's screening and primitive-pair mask and the gradient accumulator
    are uploaded once; :meth:`contract` then adds the contribution of one block of weight columns
    (whole Cartesian auxiliary shells ``K0 <= K < K1``) and may be called for several blocks, so
    that the weights never have to exist in full.

    Parameters
    ----------
    basis, auxbasis : Basis
    plan : DFAlgo12Plan
        Gradient plan of :func:`pyfock.Integrals.df_algo12_grad.build_grad_plan`, built with
        ``far_field=False`` (exchange contracts the integrals themselves).
    cp_stream : cupy.cuda.Stream or None
        Stream of all device work (default: the shared gradient stream).
    """

    def __init__(self, basis, auxbasis, plan, cp_stream=None):
        if cp is None:
            raise RuntimeError('CuPy is required for RowsGradContext.')
        if plan.far_field and plan.n_entries:
            raise ValueError('RowsGradContext needs a plan without far field (far_field=False).')
        (bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms,
         bfs_expnts, shell_l, shell_bfs_offset, bfs_nbfshell) = _pack_basis(basis)
        (aux_bfs_coords, aux_bfs_contr_prim_norms, aux_bfs_lmn, aux_bfs_nprim, aux_bfs_coeffs,
         aux_bfs_prim_norms, aux_bfs_expnts, aux_shell_l, aux_shell_bfs_offset,
         aux_bfs_nbfshell) = _pack_basis(auxbasis)
        bfs_atoms = np.array(basis.bfs_atoms, dtype=np.int32)
        aux_bfs_atoms = np.array(auxbasis.bfs_atoms, dtype=np.int32)
        self.natoms = int(max(bfs_atoms.max(), aux_bfs_atoms.max())) + 1
        nshells = len(basis.shells)
        self.nshells_aux = len(auxbasis.shells)
        self.aux_off = np.asarray(aux_shell_bfs_offset, dtype=np.int64)
        self.aux_nbf = np.asarray(aux_bfs_nbfshell, dtype=np.int64)
        self.nao = int(basis.bfs_nao)
        self.naux = int(auxbasis.bfs_nao)

        max_l_bra = int(shell_l.max())
        max_l_aux = int(aux_shell_l.max())
        nroots_max = (2 * max_l_bra + max_l_aux + 1) // 2 + 1
        if nroots_max > 10:
            raise NotImplementedError('RowsGradContext supports Rys orders up to 10 only.')
        self.kernel = _get_rows_kernel(2 * max_l_bra + 2, max_l_aux + 2, nroots_max)

        ab_shell_a, ab_shell_b = (a.astype(np.int32) for a in np.tril_indices(nshells))
        if plan.pair_I.shape[0] != ab_shell_a.shape[0]:
            raise ValueError('plan was built for a different basis')
        self.n_ab = int(ab_shell_a.shape[0])
        # the plan's dense primitive-pair branches; dropped pairs point at the all-far row of the mask
        nprim_pairs = np.zeros(self.n_ab, dtype=np.int64)
        nprim_pairs[plan.sig] = (plan.bfs_nprim[plan.shell_off[plan.pair_I[plan.sig]]]
                                 * plan.bfs_nprim[plan.shell_off[plan.pair_J[plan.sig]]])
        pp_off = np.zeros(self.n_ab + 1, dtype=np.int64)
        pp_off[1:] = np.cumsum(nprim_pairs)
        pp_branch = _pp_branch_dense(plan.sig, plan.pair_I, plan.pair_J, plan.shell_off,
                                     plan.bfs_nprim, plan.bfs_expnts, plan.bfs_coords, plan.pp_group,
                                     plan.grp_branch_eff, pp_off, plan.ff_eff.shape[0])
        if pp_branch.size == 0:
            pp_branch = np.zeros(1, dtype=np.int32)
        ff_mask = np.ascontiguousarray(
            np.vstack([plan.ff_eff, np.ones((1, plan.ff_eff.shape[1]), dtype=np.bool_)]))
        self.threshold = float(plan.threshold)
        sig_mask = np.zeros(self.n_ab, dtype=np.bool_)
        sig_mask[plan.sig] = True
        # A triplet with all three centers on one atom has zero total derivative.
        atom_ab = np.where(bfs_atoms[shell_bfs_offset[ab_shell_a]] == bfs_atoms[shell_bfs_offset[ab_shell_b]],
                           bfs_atoms[shell_bfs_offset[ab_shell_a]], -1).astype(np.int64)
        atom_c = aux_bfs_atoms[aux_shell_bfs_offset].astype(np.int64)
        # Class key: (la, lb, lc) fully determines the loop bounds and the local array extents used;
        # within a class the number of primitive triples sets the trip counts of the loops.
        lab_key = ((shell_l[ab_shell_a].astype(np.int64) * 8 + shell_l[ab_shell_b].astype(np.int64)) * 8)
        aux_nprim_sh = aux_bfs_nprim[aux_shell_bfs_offset].astype(np.int64)

        if cp_stream is None:
            cp_stream, nb_stream = gradient_stream()
        else:
            nb_stream = cuda.external_stream(cp_stream.ptr)
        self.stream = cp_stream
        self.nb_stream = nb_stream
        upload = cp.cuda.get_current_stream()
        with cp_stream:
            if upload.ptr != cp_stream.ptr:
                cp_stream.wait_event(upload.record())
            self.d = {name: cp.asarray(value) for name, value in (
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
                ('DATA_X', DATA_X), ('DATA_W', DATA_W),
                ('pp_off', pp_off), ('pp_branch', pp_branch), ('ff', ff_mask),
                ('Q_pair', np.ascontiguousarray(plan.Q_pair, dtype=np.float64)),
                ('Q_aux', np.ascontiguousarray(plan.Q_aux, dtype=np.float64)),
                ('sig', sig_mask), ('atom_ab', atom_ab), ('atom_c', atom_c), ('lab_key', lab_key),
                ('lc_key', aux_shell_l.astype(np.int64)), ('npp', np.diff(pp_off)),
                ('aux_nprim_sh', aux_nprim_sh))}
            self.grad_partial = cp.zeros((_NBINS, self.natoms, 3), dtype=cp.float64)
        # Numba views without the implicit stream synchronization of a CuPy argument: everything
        # here runs in order on one stream.
        self.nb = {name: _nb(value) for name, value in self.d.items()}
        self.nb_grad_partial = _nb(self.grad_partial)

    def column_range(self, K0, K1):
        """Cartesian auxiliary columns ``[c0, c1)`` of the shells ``K0 <= K < K1``."""
        c0 = int(self.aux_off[K0])
        c1 = int(self.aux_off[K1 - 1] + self.aux_nbf[K1 - 1]) if K1 > K0 else c0
        return c0, c1

    def _tasks(self, K0, K1, weights_nb, row_of_nb, col0, threshold_grad):
        """Device task lists of the shells ``K0 <= K < K1`` that pass the screening, grouped by shape."""
        d = self.d
        nb = self.nb
        nK = K1 - K0
        chunk = max(1, min(self.n_ab, _TASK_CHUNK // max(1, nK)))
        for lo in range(0, self.n_ab, chunk):
            hi = min(lo + chunk, self.n_ab)
            keep = d['sig'][lo:hi, None] & (d['atom_ab'][lo:hi, None] != d['atom_c'][None, K0:K1])
            flat = cp.flatnonzero(keep.ravel())
            if flat.size == 0:
                continue
            tasks = (lo + flat // nK) * self.nshells_aux + (K0 + flat % nK)
            ok = cp.zeros(tasks.size, dtype=cp.int8)
            _rows_tasks[(tasks.size + 127) // 128, 128, self.nb_stream](
                _nb(tasks), self.nshells_aux, nb['pp_off'], nb['pp_branch'], nb['ff'], nb['Q_pair'],
                nb['Q_aux'], self.threshold, nb['ab_shell_a'], nb['ab_shell_b'], nb['shell_bfs_offset'],
                nb['bfs_nbfshell'], nb['aux_shell_bfs_offset'], nb['aux_bfs_nbfshell'], weights_nb,
                row_of_nb, col0, float(threshold_grad), _nb(ok))
            tasks = tasks[ok.astype(cp.bool_)]
            if tasks.size == 0:
                continue
            ab = tasks // self.nshells_aux
            k = tasks % self.nshells_aux
            key = ((d['lab_key'][ab] + d['lc_key'][k]) * 4096
                   + cp.minimum(d['npp'][ab] * d['aux_nprim_sh'][k], 4095))
            yield cp.ascontiguousarray(tasks[cp.argsort(key)])

    def contract(self, weights, row_of, K0, K1, threshold_grad=1e-11):
        """
        Add ``sum_{ij, P} fac W[row_of[i, j], P - c0] d(ij|P)/dR`` over the shells ``K0 <= K < K1``.

        ``weights`` is a C-ordered float64 device array ``(nrows, c1 - c0)`` whose columns are the
        Cartesian auxiliary functions of those shells (:meth:`column_range`); ``row_of`` a symmetric
        ``(nao, nao)`` int32 device array of the row of every function pair (``-1``: left out). Both
        must have been produced on (or before) this context's stream.
        """
        c0, c1 = self.column_range(K0, K1)
        if weights.shape[1] != c1 - c0:
            raise ValueError('weights has %d columns, the shells %d..%d have %d' % (weights.shape[1], K0, K1, c1 - c0))
        if not weights.flags.c_contiguous or weights.dtype != cp.float64:
            raise ValueError('weights must be a C-ordered float64 array')
        nb = self.nb
        weights_nb = _nb(weights)
        row_of_nb = _nb(row_of)
        threads = 64
        with self.stream:
            for task_list in self._tasks(K0, K1, weights_nb, row_of_nb, c0, threshold_grad):
                blocks = (task_list.shape[0] + threads - 1) // threads
                self.kernel[blocks, threads, self.nb_stream](
                    _nb(task_list), self.nshells_aux, nb['ab_shell_a'], nb['ab_shell_b'],
                    nb['bfs_coords'], nb['bfs_contr_prim_norms'], nb['bfs_lmn'], nb['bfs_nprim'],
                    nb['bfs_coeffs'], nb['bfs_prim_norms'], nb['bfs_expnts'],
                    nb['shell_l'], nb['shell_bfs_offset'], nb['bfs_nbfshell'], nb['bfs_atoms'],
                    nb['aux_bfs_coords'], nb['aux_bfs_contr_prim_norms'], nb['aux_bfs_lmn'],
                    nb['aux_bfs_nprim'], nb['aux_bfs_coeffs'], nb['aux_bfs_prim_norms'],
                    nb['aux_bfs_expnts'], nb['aux_shell_l'], nb['aux_shell_bfs_offset'],
                    nb['aux_bfs_nbfshell'], nb['aux_bfs_atoms'],
                    weights_nb, row_of_nb, c0, nb['DATA_X'], nb['DATA_W'],
                    nb['pp_off'], nb['pp_branch'], nb['ff'], self.nb_grad_partial)

    def gradient(self):
        """The accumulated ``(natoms, 3)`` term as a NumPy array."""
        with self.stream:
            grad = cp.asnumpy(self.grad_partial.sum(axis=0))
        self.stream.synchronize()
        return grad


def symmetric_row_of(row_mu, row_nu, nao):
    """Device ``(nao, nao)`` int32 table of the row of every function pair, both triangles (``-1``: none)."""
    row_of = cp.full((nao, nao), -1, dtype=cp.int32)
    mu = cp.asarray(row_mu)
    nu = cp.asarray(row_nu)
    rows = cp.arange(mu.shape[0], dtype=cp.int32)
    row_of[mu, nu] = rows
    row_of[nu, mu] = rows
    return row_of


def rys_3c2e_grad_contract_rows_cupy(basis, auxbasis, plan, Grows, row_mu, row_nu, fit_tables=None,
                                     threshold_grad=1e-11, cp_stream=None):
    """
    GPU counterpart of :func:`pyfock.Integrals.df_algo12_grad.grad_contract_rows`:
    ``grad[A, d] = sum_{ij, P} Gamma^P_ij d(ij|P)/dR_{A,d}`` over both triangles, with
    ``Gamma^P_ij`` in row ``r`` of ``Grows`` for the function pair ``(row_mu[r], row_nu[r])``.

    ``Grows`` (NumPy or CuPy) holds the fit-space columns: the Cartesian auxiliary functions
    (``fit_tables=None``) or, in SAO mode, the spherical ones, with ``fit_tables`` =
    ``(c2s_flat, c2s_off, sph_off, aux_nsph)`` of
    :func:`pyfock.Integrals.df_algo11_exchange._cart2sph_tables`; they are mapped onto the Cartesian
    functions before the contraction. Returns a NumPy ``(natoms, 3)`` array.
    """
    if cp is None:
        raise RuntimeError('CuPy is required for rys_3c2e_grad_contract_rows_cupy.')
    ctx = RowsGradContext(basis, auxbasis, plan, cp_stream=cp_stream)
    with ctx.stream:
        G = cp.asarray(Grows, dtype=cp.float64)
        if fit_tables is not None:
            c2s_flat, c2s_off, sph_off, aux_nsph = (np.asarray(t) for t in fit_tables)
            T = np.zeros((int(aux_nsph.sum()), ctx.naux))
            for K in range(ctx.nshells_aux):
                nC, nS = int(ctx.aux_nbf[K]), int(aux_nsph[K])
                T[sph_off[K]:sph_off[K] + nS, ctx.aux_off[K]:ctx.aux_off[K] + nC] = \
                    c2s_flat[c2s_off[K]:c2s_off[K] + nS * nC].reshape(nS, nC)
            G = G @ cp.asarray(T)
        G = cp.ascontiguousarray(G)
        row_of = symmetric_row_of(row_mu, row_nu, ctx.nao)
        ctx.contract(G, row_of, 0, ctx.nshells_aux, threshold_grad)
    return ctx.gradient()
