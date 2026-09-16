"""GPU (Numba-CUDA) contracted nuclear-attraction gradient.

Device port of :func:`pyfock.Integrals.rys_nuc_grad_contract`:

    grad[iatom, xyz] = sum_ij D_ij dV_ij/dR_{iatom, xyz}

including both the basis-function (Pulay-type) and operator (Hellmann-Feynman) contributions. Each
nucleus is treated as a very sharp s-type Gaussian, so the integral is a 3c2e one and the same Rys
machinery applies; the operator derivative follows from translational invariance,
d/dR_C = -(d/dA + d/dB).

One thread handles one (shell pair AB, nucleus C) task. As in the 3c2e gradient the shell-block
scratch of the CPU kernel is not materialised: the shift tables are re-derived inside the
accumulation loop and the density weight is folded into the root weight.
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
from .rys_nuc_grad_contract import _shell_pair_bounds_nuc

__all__ = ['rys_nuc_grad_contract_cupy']

_COMB = np.array([[float(math.comb(n, k)) if k <= n else 0.0 for k in range(16)]
                  for n in range(16)], dtype=np.float64)

_NBINS = 64
_KERNEL_CACHE = {}


def _get_kernel(g_rows, nroots_max):
    key = (g_rows, nroots_max)
    kernel = _KERNEL_CACHE.get(key)
    if kernel is not None:
        return kernel

    G_ROWS = g_rows
    NROOTS = nroots_max

    @cuda.jit(fastmath=True, cache=False)
    def _kernel(task_ab, task_k, ab_shell_a, ab_shell_b,
                bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim,
                bfs_coeffs, bfs_prim_norms, bfs_expnts,
                shell_l, shell_bfs_offset, bfs_nbfshell, bfs_atoms,
                coords_nuc, Z, dmat, DATA_X_, DATA_W_, grad_partial):
        comb = cuda.const.array_like(_COMB)

        itask = cuda.grid(1)
        if itask >= task_ab.shape[0]:
            return

        ab_idx = task_ab[itask]
        k = task_k[itask]
        ish = ab_shell_a[ab_idx]
        jsh = ab_shell_b[ab_idx]

        bf_a_start = shell_bfs_offset[ish]
        bf_b_start = shell_bfs_offset[jsh]
        nbf_a = bfs_nbfshell[ish]
        nbf_b = bfs_nbfshell[jsh]

        atom_a = bfs_atoms[bf_a_start]
        atom_b = bfs_atoms[bf_b_start]

        la_shell = shell_l[ish]
        lb_shell = shell_l[jsh]
        bra_order = la_shell + lb_shell + 1        # +1 for the derivative
        nroots = bra_order // 2 + 1

        nprim_a = bfs_nprim[bf_a_start]
        nprim_b = bfs_nprim[bf_b_start]

        ax0 = bfs_coords[bf_a_start, 0]
        ay0 = bfs_coords[bf_a_start, 1]
        az0 = bfs_coords[bf_a_start, 2]
        bx0 = bfs_coords[bf_b_start, 0]
        by0 = bfs_coords[bf_b_start, 1]
        bz0 = bfs_coords[bf_b_start, 2]

        xij0 = ax0 - bx0
        xij1 = ay0 - by0
        xij2 = az0 - bz0
        ijsq = xij0 * xij0 + xij1 * xij1 + xij2 * xij2

        pi = 3.141592653589793
        zeta = 1e12
        zeta_pi_32 = (zeta / pi) ** 1.5

        kx = coords_nuc[k, 0]
        ky = coords_nuc[k, 1]
        kz = coords_nuc[k, 2]
        ck = -Z[k] * zeta_pi_32
        pair_factor = 2.0 if ish != jsh else 1.0

        roots = cuda.local.array(NROOTS, numba.float64)
        weights = cuda.local.array(NROOTS, numba.float64)
        gx = cuda.local.array((G_ROWS, 1), numba.float64)
        gy = cuda.local.array((G_ROWS, 1), numba.float64)
        gz = cuda.local.array((G_ROWS, 1), numba.float64)

        ga_x = 0.0
        ga_y = 0.0
        ga_z = 0.0
        gb_x = 0.0
        gb_y = 0.0
        gb_z = 0.0

        for iprim_a in range(nprim_a):
            alpha = bfs_expnts[bf_a_start, iprim_a]
            two_alpha = 2.0 * alpha
            for iprim_b in range(nprim_b):
                beta = bfs_expnts[bf_b_start, iprim_b]
                two_beta = 2.0 * beta
                gamma_p = alpha + beta
                inv_gamma_p = 1.0 / gamma_p
                if math.exp(-alpha * beta * inv_gamma_p * ijsq) < 1.0e-10:
                    continue

                px = (alpha * ax0 + beta * bx0) * inv_gamma_p
                py = (alpha * ay0 + beta * by0) * inv_gamma_p
                pz = (alpha * az0 + beta * bz0) * inv_gamma_p

                pqx = px - kx
                pqy = py - ky
                pqz = pz - kz
                pqsq = pqx * pqx + pqy * pqy + pqz * pqz

                rho = gamma_p * zeta / (gamma_p + zeta)
                x = rho * pqsq
                gamma_pq_sqrt = math.sqrt(gamma_p * zeta)

                Roots(nroots, x, DATA_X_, DATA_W_, roots, weights)

                rys_prefactor = 2.0 * math.sqrt(rho / pi)
                for iroot in range(nroots):
                    root = roots[iroot]
                    Recur_3c2e(gx, root, bra_order, 0, 0, 0, ax0, bx0, kx, 0.0,
                               alpha, beta, zeta, 0.0, gamma_p, zeta,
                               alpha * beta, gamma_pq_sqrt)
                    Recur_3c2e(gy, root, bra_order, 0, 0, 0, ay0, by0, ky, 0.0,
                               alpha, beta, zeta, 0.0, gamma_p, zeta,
                               alpha * beta, gamma_pq_sqrt)
                    Recur_3c2e(gz, root, bra_order, 0, 0, 0, az0, bz0, kz, 0.0,
                               alpha, beta, zeta, 0.0, gamma_p, zeta,
                               alpha * beta, gamma_pq_sqrt)

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
                            cb = (ca * bfs_contr_prim_norms[ibf_b] * bfs_coeffs[ibf_b, iprim_b]
                                  * bfs_prim_norms[ibf_b, iprim_b])
                            w = pair_factor * dmat[ibf_a, ibf_b] * ck * cb * root_weight

                            # ---- x ----
                            s = 0.0
                            sap = 0.0
                            sam = 0.0
                            for n in range(bxl + 1):
                                t = comb[bxl, n] * xij0 ** (bxl - n)
                                row = n + axl
                                s += t * gx[row, 0]
                                sap += t * gx[row + 1, 0]
                                if axl > 0:
                                    sam += t * gx[row - 1, 0]
                            sbp = 0.0
                            for n in range(bxl + 2):
                                sbp += comb[bxl + 1, n] * xij0 ** (bxl + 1 - n) * gx[n + axl, 0]
                            sbm = 0.0
                            if bxl > 0:
                                for n in range(bxl):
                                    sbm += comb[bxl - 1, n] * xij0 ** (bxl - 1 - n) * gx[n + axl, 0]
                            sx = s
                            dax = two_alpha * sap - axl * sam
                            dbx = two_beta * sbp - bxl * sbm

                            # ---- y ----
                            s = 0.0
                            sap = 0.0
                            sam = 0.0
                            for n in range(byl + 1):
                                t = comb[byl, n] * xij1 ** (byl - n)
                                row = n + ayl
                                s += t * gy[row, 0]
                                sap += t * gy[row + 1, 0]
                                if ayl > 0:
                                    sam += t * gy[row - 1, 0]
                            sbp = 0.0
                            for n in range(byl + 2):
                                sbp += comb[byl + 1, n] * xij1 ** (byl + 1 - n) * gy[n + ayl, 0]
                            sbm = 0.0
                            if byl > 0:
                                for n in range(byl):
                                    sbm += comb[byl - 1, n] * xij1 ** (byl - 1 - n) * gy[n + ayl, 0]
                            sy = s
                            day = two_alpha * sap - ayl * sam
                            dby = two_beta * sbp - byl * sbm

                            # ---- z ----
                            s = 0.0
                            sap = 0.0
                            sam = 0.0
                            for n in range(bzl + 1):
                                t = comb[bzl, n] * xij2 ** (bzl - n)
                                row = n + azl
                                s += t * gz[row, 0]
                                sap += t * gz[row + 1, 0]
                                if azl > 0:
                                    sam += t * gz[row - 1, 0]
                            sbp = 0.0
                            for n in range(bzl + 2):
                                sbp += comb[bzl + 1, n] * xij2 ** (bzl + 1 - n) * gz[n + azl, 0]
                            sbm = 0.0
                            if bzl > 0:
                                for n in range(bzl):
                                    sbm += comb[bzl - 1, n] * xij2 ** (bzl - 1 - n) * gz[n + azl, 0]
                            sz = s
                            daz = two_alpha * sap - azl * sam
                            dbz = two_beta * sbp - bzl * sbm

                            syz = sy * sz
                            sxz = sx * sz
                            sxy = sx * sy
                            ga_x += w * dax * syz
                            ga_y += w * day * sxz
                            ga_z += w * daz * sxy
                            gb_x += w * dbx * syz
                            gb_y += w * dby * sxz
                            gb_z += w * dbz * sxy

        ibin = cuda.blockIdx.x % _NBINS
        cuda.atomic.add(grad_partial, (ibin, atom_a, 0), ga_x)
        cuda.atomic.add(grad_partial, (ibin, atom_a, 1), ga_y)
        cuda.atomic.add(grad_partial, (ibin, atom_a, 2), ga_z)
        cuda.atomic.add(grad_partial, (ibin, atom_b, 0), gb_x)
        cuda.atomic.add(grad_partial, (ibin, atom_b, 1), gb_y)
        cuda.atomic.add(grad_partial, (ibin, atom_b, 2), gb_z)
        # Operator (Hellmann-Feynman) term by translational invariance
        cuda.atomic.add(grad_partial, (ibin, k, 0), -(ga_x + gb_x))
        cuda.atomic.add(grad_partial, (ibin, k, 1), -(ga_y + gb_y))
        cuda.atomic.add(grad_partial, (ibin, k, 2), -(ga_z + gb_z))

    _KERNEL_CACHE[key] = _kernel
    return _kernel


def rys_nuc_grad_contract_cupy(basis, mol, dmat, schwarz=True, threshold=1e-13,
                               sqrt_ints4c2e_diag=None, cp_stream=None):
    """GPU counterpart of :func:`rys_nuc_grad_contract`. Returns a NumPy ``(natoms, 3)`` array."""
    if cp is None:
        raise RuntimeError('CuPy is required for rys_nuc_grad_contract_cupy.')

    (bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms,
     bfs_expnts, shell_l, shell_bfs_offset, bfs_nbfshell) = _pack_basis(basis)

    bfs_atoms = np.array(basis.bfs_atoms, dtype=np.int32)
    natoms = mol.natoms
    coords_nuc = np.array(mol.coordsBohrs, dtype=np.float64)
    Z = np.array(mol.Zcharges, dtype=np.float64)
    nshells = len(basis.shells)
    dmat = np.ascontiguousarray(dmat, dtype=np.float64)

    max_l = int(shell_l.max())
    nroots_max = (2 * max_l + 1) // 2 + 1
    if nroots_max > 10:
        raise NotImplementedError('rys_nuc_grad_contract_cupy supports Rys orders up to 10 only.')
    g_rows = 2 * max_l + 2

    ab_shell_a = np.empty(nshells * (nshells + 1) // 2, dtype=np.int32)
    ab_shell_b = np.empty_like(ab_shell_a)
    idx = 0
    for ish in range(nshells):
        for jsh in range(ish + 1):
            ab_shell_a[idx] = ish
            ab_shell_b[idx] = jsh
            idx += 1

    pair_keep = np.ones(ab_shell_a.shape[0], dtype=bool)
    if schwarz:
        if sqrt_ints4c2e_diag is None:
            from .schwarz_helpers import eri_4c2e_diag
            sqrt_ints4c2e_diag = np.sqrt(np.abs(eri_4c2e_diag(basis)))
        shell_pair_bound, dm_shell_pair_max = _shell_pair_bounds_nuc(
            nshells, shell_bfs_offset, bfs_nbfshell,
            np.asarray(sqrt_ints4c2e_diag), np.abs(dmat))
        pair_keep = (shell_pair_bound * dm_shell_pair_max)[ab_shell_a, ab_shell_b] >= threshold

    # One task per surviving (shell pair, nucleus); a pair with both shells and the nucleus on the
    # same atom has zero total derivative. Sorted by (la, lb) so each warp shares a recursion shape.
    ab_idx = np.flatnonzero(pair_keep)
    task_ab = np.repeat(ab_idx, natoms).astype(np.int32)
    task_k = np.tile(np.arange(natoms, dtype=np.int32), ab_idx.shape[0])
    atom_a = bfs_atoms[shell_bfs_offset[ab_shell_a[task_ab]]]
    atom_b = bfs_atoms[shell_bfs_offset[ab_shell_b[task_ab]]]
    keep = ~((atom_a == atom_b) & (atom_b == task_k))
    task_ab = task_ab[keep]
    task_k = task_k[keep]
    order = np.argsort(shell_l[ab_shell_a[task_ab]].astype(np.int64) * 8
                       + shell_l[ab_shell_b[task_ab]], kind='stable')
    task_ab = np.ascontiguousarray(task_ab[order])
    task_k = np.ascontiguousarray(task_k[order])

    if cp_stream is None:
        cp_stream, nb_stream = gradient_stream()
    else:
        nb_stream = cuda.external_stream(cp_stream.ptr)

    with cp_stream:
        args = [cp.asarray(a) for a in (task_ab, task_k, ab_shell_a, ab_shell_b, bfs_coords,
                                        bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs,
                                        bfs_prim_norms, bfs_expnts, shell_l, shell_bfs_offset,
                                        bfs_nbfshell, bfs_atoms, coords_nuc, Z, dmat)]
        grad_partial = cp.zeros((_NBINS, natoms, 3), dtype=cp.float64)
        kernel = _get_kernel(g_rows, nroots_max)
        threads = 64
        blocks = (task_ab.shape[0] + threads - 1) // threads
        if blocks > 0:
            kernel[blocks, threads, nb_stream](*args, cp.asarray(DATA_X), cp.asarray(DATA_W),
                                               grad_partial)
        grad = cp.asnumpy(grad_partial.sum(axis=0))
    cp_stream.synchronize()
    return grad
