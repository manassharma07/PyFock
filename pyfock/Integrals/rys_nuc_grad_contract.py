import numpy as np
import numba
from numba import njit, prange

from .rys_helpers import Roots, Recur_3c2e_new, LOOKUP_TABLE_COMB
from .rys_3c2e_symm_test import _pack_basis


def rys_nuc_grad_contract(basis, mol, dmat, schwarz=True, threshold=1e-13, ncores=None):
    """
    Contracted nuclear gradient of the nuclear-attraction matrix.

    Computes

        grad[iatom, xyz] = sum_ij D_ij * dV_ij/dR_{iatom, xyz}

    including both the basis-function (Pulay-type) and operator
    (Hellmann-Feynman) contributions. The integrals are evaluated
    shell-pair-blocked with Rys quadrature by treating each nucleus as a
    very sharp s-type Gaussian (zeta = 1e12, same trick as
    :func:`rys_nuc_mat_symm`, see https://arxiv.org/pdf/2302.11307.pdf).

    Only the bra (A) and ket (B) derivatives are evaluated; the operator
    derivative follows from translational invariance:
    d/dR_C = -(d/dA + d/dB).

    Parameters
    ----------
    basis : Basis
        Basis set object.
    mol : Mol
        Molecule object (nuclear charges and coordinates).
    dmat : ndarray (nbf, nbf)
        Symmetric density matrix in the (Cartesian) AO basis.
    schwarz : bool, optional
        Screen shell pairs by max|D| * sqrt((ab|ab)).
    threshold : float, optional
        Screening threshold.
    ncores : int, optional
        Number of threads for Numba.

    Returns
    -------
    grad : ndarray (natoms, 3)
    """
    if ncores is not None:
        numba.set_num_threads(ncores)

    (
        bfs_coords,
        bfs_contr_prim_norms,
        bfs_lmn,
        bfs_nprim,
        bfs_coeffs,
        bfs_prim_norms,
        bfs_expnts,
        shell_l,
        shell_bfs_offset,
        bfs_nbfshell,
    ) = _pack_basis(basis)

    bfs_atoms = np.array(basis.bfs_atoms, dtype=np.int32)
    natoms = mol.natoms
    coords_nuc = np.array(mol.coordsBohrs, dtype=np.float64)
    Z = np.array(mol.Zcharges, dtype=np.float64)

    nshells = len(basis.shells)
    dmat = np.ascontiguousarray(dmat, dtype=np.float64)

    shell_pair_bound = np.ones((1, 1), dtype=np.float64)
    dm_shell_pair_max = np.ones((1, 1), dtype=np.float64)
    if schwarz:
        from .schwarz_helpers import eri_4c2e_diag

        sqrt_ints4c2e_diag = np.sqrt(np.abs(eri_4c2e_diag(basis)))
        shell_pair_bound, dm_shell_pair_max = _shell_pair_bounds_nuc(
            nshells,
            shell_bfs_offset,
            bfs_nbfshell,
            sqrt_ints4c2e_diag,
            np.abs(dmat),
        )

    nthreads = numba.get_num_threads()

    return rys_nuc_grad_contract_internal(
        natoms,
        nthreads,
        nshells,
        bfs_coords,
        bfs_contr_prim_norms,
        bfs_lmn,
        bfs_nprim,
        bfs_coeffs,
        bfs_prim_norms,
        bfs_expnts,
        shell_l,
        shell_bfs_offset,
        bfs_nbfshell,
        bfs_atoms,
        coords_nuc,
        Z,
        dmat,
        schwarz,
        shell_pair_bound,
        dm_shell_pair_max,
        threshold,
    )


@njit(cache=True, fastmath=True, nogil=True, error_model="numpy")
def _shell_pair_bounds_nuc(
    nshells,
    shell_bfs_offset,
    bfs_nbfshell,
    sqrt_ints4c2e_diag,
    dmat_abs,
):
    shell_pair_bound = np.zeros((nshells, nshells), dtype=np.float64)
    dm_shell_pair_max = np.zeros((nshells, nshells), dtype=np.float64)
    for ish in range(nshells):
        ia0 = shell_bfs_offset[ish]
        nia = bfs_nbfshell[ish]
        for jsh in range(ish + 1):
            ib0 = shell_bfs_offset[jsh]
            nib = bfs_nbfshell[jsh]
            bound = 0.0
            dm_max = 0.0
            for ia in range(ia0, ia0 + nia):
                for ib in range(ib0, ib0 + nib):
                    if sqrt_ints4c2e_diag[ia, ib] > bound:
                        bound = sqrt_ints4c2e_diag[ia, ib]
                    if dmat_abs[ia, ib] > dm_max:
                        dm_max = dmat_abs[ia, ib]
            shell_pair_bound[ish, jsh] = bound
            shell_pair_bound[jsh, ish] = bound
            dm_shell_pair_max[ish, jsh] = dm_max
            dm_shell_pair_max[jsh, ish] = dm_max
    return shell_pair_bound, dm_shell_pair_max


@njit(parallel=True, cache=True, fastmath=True, nogil=True, error_model="numpy")
def rys_nuc_grad_contract_internal(
    natoms,
    nthreads,
    nshells,
    bfs_coords,
    bfs_contr_prim_norms,
    bfs_lmn,
    bfs_nprim,
    bfs_coeffs,
    bfs_prim_norms,
    bfs_expnts,
    shell_l,
    shell_bfs_offset,
    bfs_nbfshell,
    bfs_atoms,
    coords_nuc,
    Z,
    dmat,
    schwarz,
    shell_pair_bound,
    dm_shell_pair_max,
    threshold,
):
    n_ab = nshells * (nshells + 1) // 2
    pi = 3.141592653589793
    zeta = 1e12
    zeta_pi_32 = (zeta / pi) ** 1.5

    ab_shell_a = np.empty(n_ab, dtype=np.int32)
    ab_shell_b = np.empty(n_ab, dtype=np.int32)
    idx = 0
    for ish in range(nshells):
        for jsh in range(ish + 1):
            ab_shell_a[idx] = ish
            ab_shell_b[idx] = jsh
            idx += 1

    max_l_bra = 0
    max_nbf_shell = 0
    for ish in range(nshells):
        if shell_l[ish] > max_l_bra:
            max_l_bra = shell_l[ish]
        if bfs_nbfshell[ish] > max_nbf_shell:
            max_nbf_shell = bfs_nbfshell[ish]

    max_bra_order = 2 * max_l_bra + 1
    max_nbf_pair = max_nbf_shell * max_nbf_shell

    grad_threads = np.zeros((nthreads, natoms, 3), dtype=np.float64)

    for ab_idx in prange(n_ab):
        tid = numba.get_thread_id()

        ish = ab_shell_a[ab_idx]
        jsh = ab_shell_b[ab_idx]

        if schwarz:
            if dm_shell_pair_max[ish, jsh] * shell_pair_bound[ish, jsh] < threshold:
                continue

        bf_a_start = shell_bfs_offset[ish]
        bf_b_start = shell_bfs_offset[jsh]
        nbf_a = bfs_nbfshell[ish]
        nbf_b = bfs_nbfshell[jsh]

        atom_a = bfs_atoms[bf_a_start]
        atom_b = bfs_atoms[bf_b_start]

        la_shell = shell_l[ish]
        lb_shell = shell_l[jsh]
        bra_order = la_shell + lb_shell + 1  # +1 for the derivative
        nroots = bra_order // 2 + 1

        ibf_a0 = bf_a_start
        ibf_b0 = bf_b_start
        nprim_a = bfs_nprim[ibf_a0]
        nprim_b = bfs_nprim[ibf_b0]

        ax0 = bfs_coords[ibf_a0, 0]
        ay0 = bfs_coords[ibf_a0, 1]
        az0 = bfs_coords[ibf_a0, 2]
        bx0 = bfs_coords[ibf_b0, 0]
        by0 = bfs_coords[ibf_b0, 1]
        bz0 = bfs_coords[ibf_b0, 2]

        xij0 = ax0 - bx0
        xij1 = ay0 - by0
        xij2 = az0 - bz0
        ijsq = xij0 * xij0 + xij1 * xij1 + xij2 * xij2

        roots = np.zeros(10, dtype=np.float64)
        weights = np.zeros(10, dtype=np.float64)
        gx = np.zeros((max_bra_order + 1, 1), dtype=np.float64)
        gy = np.zeros((max_bra_order + 1, 1), dtype=np.float64)
        gz = np.zeros((max_bra_order + 1, 1), dtype=np.float64)
        shift_x = np.zeros(max_nbf_pair, dtype=np.float64)
        shift_y = np.zeros(max_nbf_pair, dtype=np.float64)
        shift_z = np.zeros(max_nbf_pair, dtype=np.float64)
        dshift_ax = np.zeros(max_nbf_pair, dtype=np.float64)
        dshift_ay = np.zeros(max_nbf_pair, dtype=np.float64)
        dshift_az = np.zeros(max_nbf_pair, dtype=np.float64)
        dshift_bx = np.zeros(max_nbf_pair, dtype=np.float64)
        dshift_by = np.zeros(max_nbf_pair, dtype=np.float64)
        dshift_bz = np.zeros(max_nbf_pair, dtype=np.float64)
        dblock = np.zeros((6, max_nbf_shell, max_nbf_shell), dtype=np.float64)

        pair_factor = 2.0 if ish != jsh else 1.0

        for k in range(natoms):
            atom_k = k
            if atom_a == atom_b and atom_b == atom_k:
                continue  # vanishes by translational invariance

            kx = coords_nuc[k, 0]
            ky = coords_nuc[k, 1]
            kz = coords_nuc[k, 2]
            # Nuclear charge as a sharp s-Gaussian
            ck = -Z[k] * zeta_pi_32

            dblock[:, :, :] = 0.0

            for iprim_a in range(nprim_a):
                alpha = bfs_expnts[ibf_a0, iprim_a]
                two_alpha = 2.0 * alpha
                for iprim_b in range(nprim_b):
                    beta = bfs_expnts[ibf_b0, iprim_b]
                    two_beta = 2.0 * beta
                    gamma_p = alpha + beta
                    inv_gamma_p = 1.0 / gamma_p
                    screen_ab = np.exp(-alpha * beta * inv_gamma_p * ijsq)
                    if screen_ab < 1.0e-10:
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
                    gamma_pq_sqrt = np.sqrt(gamma_p * zeta)

                    Roots(nroots, x, roots, weights)

                    rys_prefactor = 2.0 * np.sqrt(rho / pi)
                    for iroot in range(nroots):
                        root = roots[iroot]
                        Recur_3c2e_new(
                            gx, root, bra_order, 0, 0, 0,
                            ax0, bx0, kx, 0.0,
                            alpha, beta, zeta, 0.0,
                            gamma_p, zeta, alpha * beta, gamma_pq_sqrt,
                        )
                        Recur_3c2e_new(
                            gy, root, bra_order, 0, 0, 0,
                            ay0, by0, ky, 0.0,
                            alpha, beta, zeta, 0.0,
                            gamma_p, zeta, alpha * beta, gamma_pq_sqrt,
                        )
                        Recur_3c2e_new(
                            gz, root, bra_order, 0, 0, 0,
                            az0, bz0, kz, 0.0,
                            alpha, beta, zeta, 0.0,
                            gamma_p, zeta, alpha * beta, gamma_pq_sqrt,
                        )

                        # Fused shift tables for the shell pair
                        for ia in range(nbf_a):
                            ibf_a = bf_a_start + ia
                            axl = bfs_lmn[ibf_a, 0]
                            ayl = bfs_lmn[ibf_a, 1]
                            azl = bfs_lmn[ibf_a, 2]
                            for ib in range(nbf_b):
                                ibf_b = bf_b_start + ib
                                bxl = bfs_lmn[ibf_b, 0]
                                byl = bfs_lmn[ibf_b, 1]
                                bzl = bfs_lmn[ibf_b, 2]
                                iab = ia * nbf_b + ib

                                # ---- x ----
                                s = 0.0
                                sap = 0.0
                                sam = 0.0
                                for n in range(bxl + 1):
                                    t = LOOKUP_TABLE_COMB[bxl, n] * xij0 ** (bxl - n)
                                    row = n + axl
                                    s += t * gx[row, 0]
                                    sap += t * gx[row + 1, 0]
                                    if axl > 0:
                                        sam += t * gx[row - 1, 0]
                                sbp = 0.0
                                for n in range(bxl + 2):
                                    sbp += LOOKUP_TABLE_COMB[bxl + 1, n] * xij0 ** (bxl + 1 - n) * gx[n + axl, 0]
                                sbm = 0.0
                                if bxl > 0:
                                    for n in range(bxl):
                                        sbm += LOOKUP_TABLE_COMB[bxl - 1, n] * xij0 ** (bxl - 1 - n) * gx[n + axl, 0]
                                shift_x[iab] = s
                                dshift_ax[iab] = two_alpha * sap - axl * sam
                                dshift_bx[iab] = two_beta * sbp - bxl * sbm

                                # ---- y ----
                                s = 0.0
                                sap = 0.0
                                sam = 0.0
                                for n in range(byl + 1):
                                    t = LOOKUP_TABLE_COMB[byl, n] * xij1 ** (byl - n)
                                    row = n + ayl
                                    s += t * gy[row, 0]
                                    sap += t * gy[row + 1, 0]
                                    if ayl > 0:
                                        sam += t * gy[row - 1, 0]
                                sbp = 0.0
                                for n in range(byl + 2):
                                    sbp += LOOKUP_TABLE_COMB[byl + 1, n] * xij1 ** (byl + 1 - n) * gy[n + ayl, 0]
                                sbm = 0.0
                                if byl > 0:
                                    for n in range(byl):
                                        sbm += LOOKUP_TABLE_COMB[byl - 1, n] * xij1 ** (byl - 1 - n) * gy[n + ayl, 0]
                                shift_y[iab] = s
                                dshift_ay[iab] = two_alpha * sap - ayl * sam
                                dshift_by[iab] = two_beta * sbp - byl * sbm

                                # ---- z ----
                                s = 0.0
                                sap = 0.0
                                sam = 0.0
                                for n in range(bzl + 1):
                                    t = LOOKUP_TABLE_COMB[bzl, n] * xij2 ** (bzl - n)
                                    row = n + azl
                                    s += t * gz[row, 0]
                                    sap += t * gz[row + 1, 0]
                                    if azl > 0:
                                        sam += t * gz[row - 1, 0]
                                sbp = 0.0
                                for n in range(bzl + 2):
                                    sbp += LOOKUP_TABLE_COMB[bzl + 1, n] * xij2 ** (bzl + 1 - n) * gz[n + azl, 0]
                                sbm = 0.0
                                if bzl > 0:
                                    for n in range(bzl):
                                        sbm += LOOKUP_TABLE_COMB[bzl - 1, n] * xij2 ** (bzl - 1 - n) * gz[n + azl, 0]
                                shift_z[iab] = s
                                dshift_az[iab] = two_alpha * sap - azl * sam
                                dshift_bz[iab] = two_beta * sbp - bzl * sbm

                        root_weight = rys_prefactor * weights[iroot]
                        for ia in range(nbf_a):
                            ibf_a = bf_a_start + ia
                            ca = (
                                bfs_contr_prim_norms[ibf_a]
                                * bfs_coeffs[ibf_a, iprim_a]
                                * bfs_prim_norms[ibf_a, iprim_a]
                            )
                            for ib in range(nbf_b):
                                ibf_b = bf_b_start + ib
                                cb = (
                                    ca
                                    * bfs_contr_prim_norms[ibf_b]
                                    * bfs_coeffs[ibf_b, iprim_b]
                                    * bfs_prim_norms[ibf_b, iprim_b]
                                )
                                iab = ia * nbf_b + ib
                                w = cb * root_weight
                                sx = shift_x[iab]
                                sy = shift_y[iab]
                                sz = shift_z[iab]
                                syz = sy * sz
                                sxz = sx * sz
                                sxy = sx * sy
                                dblock[0, ia, ib] += w * dshift_ax[iab] * syz
                                dblock[1, ia, ib] += w * dshift_ay[iab] * sxz
                                dblock[2, ia, ib] += w * dshift_az[iab] * sxy
                                dblock[3, ia, ib] += w * dshift_bx[iab] * syz
                                dblock[4, ia, ib] += w * dshift_by[iab] * sxz
                                dblock[5, ia, ib] += w * dshift_bz[iab] * sxy

            # Contract with the density matrix for this nucleus
            ga_x = 0.0
            ga_y = 0.0
            ga_z = 0.0
            gb_x = 0.0
            gb_y = 0.0
            gb_z = 0.0
            for ia in range(nbf_a):
                ibf_a = bf_a_start + ia
                for ib in range(nbf_b):
                    ibf_b = bf_b_start + ib
                    dm_ab = pair_factor * dmat[ibf_a, ibf_b] * ck
                    ga_x += dm_ab * dblock[0, ia, ib]
                    ga_y += dm_ab * dblock[1, ia, ib]
                    ga_z += dm_ab * dblock[2, ia, ib]
                    gb_x += dm_ab * dblock[3, ia, ib]
                    gb_y += dm_ab * dblock[4, ia, ib]
                    gb_z += dm_ab * dblock[5, ia, ib]

            grad_threads[tid, atom_a, 0] += ga_x
            grad_threads[tid, atom_a, 1] += ga_y
            grad_threads[tid, atom_a, 2] += ga_z
            grad_threads[tid, atom_b, 0] += gb_x
            grad_threads[tid, atom_b, 1] += gb_y
            grad_threads[tid, atom_b, 2] += gb_z
            # Operator (Hellmann-Feynman) term by translational invariance
            grad_threads[tid, atom_k, 0] -= ga_x + gb_x
            grad_threads[tid, atom_k, 1] -= ga_y + gb_y
            grad_threads[tid, atom_k, 2] -= ga_z + gb_z

    grad = np.zeros((natoms, 3), dtype=np.float64)
    for t in range(nthreads):
        for iatom in range(natoms):
            for direction in range(3):
                grad[iatom, direction] += grad_threads[t, iatom, direction]

    return grad
