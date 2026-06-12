import numpy as np
import numba
from numba import njit, prange

from .rys_helpers import Roots, Recur_3c2e_new, LOOKUP_TABLE_COMB
from .rys_3c2e_symm_test import _pack_basis


def rys_3c2e_grad_contract(
    basis,
    auxbasis,
    dmat,
    df_coeff,
    schwarz=True,
    threshold_schwarz=1e-11,
    ncores=None,
):
    """
    Contracted nuclear gradient of three-center two-electron (3c2e) integrals.

    Computes the density-fitted Coulomb 3c2e gradient contribution

        grad[iatom, xyz] = sum_{ij, P} D_ij * c_P * d(ij|P)/dR_{iatom, xyz}

    without ever storing the full derivative tensor d(ij|P)/dR. The derivative
    integrals are evaluated shell-blocked with Rys quadrature (following
    :func:`rys_3c2e_symm_test`) and contracted on the fly with the density
    matrix ``dmat`` and the density-fitting coefficients ``df_coeff``.

    Only the bra (A) and auxiliary (C) derivatives are evaluated explicitly;
    they share the same horizontal-shift binomial loop, so all five required
    shifted integrals are formed in a single fused pass. The remaining
    derivative follows from translational invariance: d/dB = -(d/dA + d/dC).

    Parameters
    ----------
    basis : Basis
        Orbital basis set object.
    auxbasis : Basis
        Auxiliary basis set object used for density fitting.
    dmat : ndarray (nbf, nbf)
        Symmetric density matrix in the (Cartesian) AO basis.
    df_coeff : ndarray (naux,)
        Density-fitting coefficients c_P = (P|Q)^-1 (Q|kl) D_kl.
    schwarz : bool, optional
        Apply Schwarz screening combined with density/coefficient weighting.
    threshold_schwarz : float, optional
        Screening threshold for |D_blk|*|c_blk|*sqrt((ab|ab))*sqrt((c|c)).
    ncores : int, optional
        Number of threads for Numba. If None the current setting is used.

    Returns
    -------
    grad : ndarray (natoms, 3)
        The contracted 3c2e gradient contribution.
    """
    if ncores is not None:
        numba.set_num_threads(ncores)

    max_l_total = (
        2 * int(max(np.array(basis.shells, dtype=np.int32)) - 1)
        + int(max(np.array(auxbasis.shells, dtype=np.int32)) - 1)
    )
    # +1 for the derivative
    if (max_l_total + 1) // 2 + 1 > 10:
        raise NotImplementedError(
            'rys_3c2e_grad_contract currently supports Rys orders up to 10 only.'
        )

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

    (
        aux_bfs_coords,
        aux_bfs_contr_prim_norms,
        aux_bfs_lmn,
        aux_bfs_nprim,
        aux_bfs_coeffs,
        aux_bfs_prim_norms,
        aux_bfs_expnts,
        aux_shell_l,
        aux_shell_bfs_offset,
        aux_bfs_nbfshell,
    ) = _pack_basis(auxbasis)

    bfs_atoms = np.array(basis.bfs_atoms, dtype=np.int32)
    aux_bfs_atoms = np.array(auxbasis.bfs_atoms, dtype=np.int32)
    natoms = int(max(bfs_atoms.max(), aux_bfs_atoms.max())) + 1

    nshells = len(basis.shells)
    nshells_aux = len(auxbasis.shells)

    dmat = np.ascontiguousarray(dmat, dtype=np.float64)
    df_coeff = np.ascontiguousarray(df_coeff, dtype=np.float64)

    shell_pair_bound = np.ones((1, 1), dtype=np.float64)
    aux_shell_bound = np.ones(1, dtype=np.float64)
    dm_shell_pair_max = np.ones((1, 1), dtype=np.float64)
    aux_shell_coeff_max = np.ones(1, dtype=np.float64)
    if schwarz:
        from .rys_2c2e_diag import rys_2c2e_diag
        from .schwarz_helpers import eri_4c2e_diag

        sqrt_ints4c2e_diag = np.sqrt(np.abs(eri_4c2e_diag(basis)))
        sqrt_diag_ints2c2e = np.sqrt(np.abs(rys_2c2e_diag(auxbasis)))

        shell_pair_bound, aux_shell_bound, dm_shell_pair_max, aux_shell_coeff_max = _shell_bounds(
            nshells,
            nshells_aux,
            shell_bfs_offset,
            bfs_nbfshell,
            aux_shell_bfs_offset,
            aux_bfs_nbfshell,
            sqrt_ints4c2e_diag,
            sqrt_diag_ints2c2e,
            np.abs(dmat),
            np.abs(df_coeff),
        )

    nthreads = numba.get_num_threads()

    grad = rys_3c2e_grad_contract_internal(
        natoms,
        nthreads,
        nshells,
        nshells_aux,
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
        aux_bfs_coords,
        aux_bfs_contr_prim_norms,
        aux_bfs_lmn,
        aux_bfs_nprim,
        aux_bfs_coeffs,
        aux_bfs_prim_norms,
        aux_bfs_expnts,
        aux_shell_l,
        aux_shell_bfs_offset,
        aux_bfs_nbfshell,
        aux_bfs_atoms,
        dmat,
        df_coeff,
        schwarz,
        shell_pair_bound,
        aux_shell_bound,
        dm_shell_pair_max,
        aux_shell_coeff_max,
        threshold_schwarz,
    )
    return grad


@njit(cache=True, fastmath=True, nogil=True, error_model="numpy")
def _shell_bounds(
    nshells,
    nshells_aux,
    shell_bfs_offset,
    bfs_nbfshell,
    aux_shell_bfs_offset,
    aux_bfs_nbfshell,
    sqrt_ints4c2e_diag,
    sqrt_diag_ints2c2e,
    dmat_abs,
    df_coeff_abs,
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

    aux_shell_bound = np.zeros(nshells_aux, dtype=np.float64)
    aux_shell_coeff_max = np.zeros(nshells_aux, dtype=np.float64)
    for ksh in range(nshells_aux):
        ic0 = aux_shell_bfs_offset[ksh]
        nic = aux_bfs_nbfshell[ksh]
        bound = 0.0
        c_max = 0.0
        for ic in range(ic0, ic0 + nic):
            if sqrt_diag_ints2c2e[ic] > bound:
                bound = sqrt_diag_ints2c2e[ic]
            if df_coeff_abs[ic] > c_max:
                c_max = df_coeff_abs[ic]
        aux_shell_bound[ksh] = bound
        aux_shell_coeff_max[ksh] = c_max

    return shell_pair_bound, aux_shell_bound, dm_shell_pair_max, aux_shell_coeff_max


@njit(cache=True, fastmath=True, nogil=True, error_model="numpy", inline="always")
def _build_shift_tables_grad(
    shift_x,
    shift_y,
    shift_z,
    dshift_ax,
    dshift_ay,
    dshift_az,
    dshift_cx,
    dshift_cy,
    dshift_cz,
    nbf_a,
    nbf_b,
    nbf_c,
    bf_a_start,
    bf_b_start,
    bf_c_start,
    bfs_lmn,
    aux_bfs_lmn,
    gx,
    gy,
    gz,
    xij0,
    xij1,
    xij2,
    two_alpha,
    two_gamma_q,
):
    # For each (ab|c) entry, all five shifted 1D integrals needed for the A-
    # and C-derivatives share the same binomial expansion over the ket angular
    # momentum, so they are accumulated in a single fused loop:
    #   S(a, b, c), S(a+1, b, c), S(a-1, b, c), S(a, b, c+1), S(a, b, c-1)
    for ia in range(nbf_a):
        ibf_a = bf_a_start + ia
        ax = bfs_lmn[ibf_a, 0]
        ay = bfs_lmn[ibf_a, 1]
        az = bfs_lmn[ibf_a, 2]
        for ib in range(nbf_b):
            ibf_b = bf_b_start + ib
            bx = bfs_lmn[ibf_b, 0]
            by = bfs_lmn[ibf_b, 1]
            bz = bfs_lmn[ibf_b, 2]
            iab = ia * nbf_b + ib
            for ic in range(nbf_c):
                ibf_c = bf_c_start + ic
                cx = aux_bfs_lmn[ibf_c, 0]
                cy = aux_bfs_lmn[ibf_c, 1]
                cz = aux_bfs_lmn[ibf_c, 2]

                # ---- x ----
                s = 0.0
                sap = 0.0
                sam = 0.0
                scp = 0.0
                scm = 0.0
                for n in range(bx + 1):
                    t = LOOKUP_TABLE_COMB[bx, n] * xij0 ** (bx - n)
                    row = n + ax
                    s += t * gx[row, cx]
                    sap += t * gx[row + 1, cx]
                    scp += t * gx[row, cx + 1]
                    if ax > 0:
                        sam += t * gx[row - 1, cx]
                    if cx > 0:
                        scm += t * gx[row, cx - 1]
                shift_x[iab, ic] = s
                dshift_ax[iab, ic] = two_alpha * sap - ax * sam
                dshift_cx[iab, ic] = two_gamma_q * scp - cx * scm

                # ---- y ----
                s = 0.0
                sap = 0.0
                sam = 0.0
                scp = 0.0
                scm = 0.0
                for n in range(by + 1):
                    t = LOOKUP_TABLE_COMB[by, n] * xij1 ** (by - n)
                    row = n + ay
                    s += t * gy[row, cy]
                    sap += t * gy[row + 1, cy]
                    scp += t * gy[row, cy + 1]
                    if ay > 0:
                        sam += t * gy[row - 1, cy]
                    if cy > 0:
                        scm += t * gy[row, cy - 1]
                shift_y[iab, ic] = s
                dshift_ay[iab, ic] = two_alpha * sap - ay * sam
                dshift_cy[iab, ic] = two_gamma_q * scp - cy * scm

                # ---- z ----
                s = 0.0
                sap = 0.0
                sam = 0.0
                scp = 0.0
                scm = 0.0
                for n in range(bz + 1):
                    t = LOOKUP_TABLE_COMB[bz, n] * xij2 ** (bz - n)
                    row = n + az
                    s += t * gz[row, cz]
                    sap += t * gz[row + 1, cz]
                    scp += t * gz[row, cz + 1]
                    if az > 0:
                        sam += t * gz[row - 1, cz]
                    if cz > 0:
                        scm += t * gz[row, cz - 1]
                shift_z[iab, ic] = s
                dshift_az[iab, ic] = two_alpha * sap - az * sam
                dshift_cz[iab, ic] = two_gamma_q * scp - cz * scm


@njit(parallel=True, cache=True, fastmath=True, nogil=True, error_model="numpy")
def rys_3c2e_grad_contract_internal(
    natoms,
    nthreads,
    nshells,
    nshells_aux,
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
    aux_bfs_coords,
    aux_bfs_contr_prim_norms,
    aux_bfs_lmn,
    aux_bfs_nprim,
    aux_bfs_coeffs,
    aux_bfs_prim_norms,
    aux_bfs_expnts,
    aux_shell_l,
    aux_shell_bfs_offset,
    aux_bfs_nbfshell,
    aux_bfs_atoms,
    dmat,
    df_coeff,
    schwarz,
    shell_pair_bound,
    aux_shell_bound,
    dm_shell_pair_max,
    aux_shell_coeff_max,
    threshold_schwarz,
):
    n_ab = nshells * (nshells + 1) // 2
    n_tasks = n_ab * nshells_aux
    pi = 3.141592653589793

    ab_shell_a = np.empty(n_ab, dtype=np.int32)
    ab_shell_b = np.empty(n_ab, dtype=np.int32)
    idx = 0
    for ish in range(nshells):
        for jsh in range(ish + 1):
            ab_shell_a[idx] = ish
            ab_shell_b[idx] = jsh
            idx += 1

    max_l_bra = 0
    max_l_aux = 0
    max_nbf_shell = 0
    max_nbf_aux_shell = 0
    for ish in range(nshells):
        if shell_l[ish] > max_l_bra:
            max_l_bra = shell_l[ish]
        if bfs_nbfshell[ish] > max_nbf_shell:
            max_nbf_shell = bfs_nbfshell[ish]
    for ksh in range(nshells_aux):
        if aux_shell_l[ksh] > max_l_aux:
            max_l_aux = aux_shell_l[ksh]
        if aux_bfs_nbfshell[ksh] > max_nbf_aux_shell:
            max_nbf_aux_shell = aux_bfs_nbfshell[ksh]

    # +1 in the bra and aux orders for the derivative shifts
    max_bra_order = 2 * max_l_bra + 1
    max_aux_order = max_l_aux + 1
    max_nbf_pair = max_nbf_shell * max_nbf_shell

    # Thread-local gradient accumulators to avoid races
    grad_threads = np.zeros((nthreads, natoms, 3), dtype=np.float64)

    for task_idx in prange(n_tasks):
        tid = numba.get_thread_id()

        ab_idx = task_idx // nshells_aux
        ksh = task_idx - ab_idx * nshells_aux

        ish = ab_shell_a[ab_idx]
        jsh = ab_shell_b[ab_idx]

        if schwarz:
            if (
                shell_pair_bound[ish, jsh]
                * aux_shell_bound[ksh]
                * dm_shell_pair_max[ish, jsh]
                * aux_shell_coeff_max[ksh]
                < threshold_schwarz
            ):
                continue

        bf_a_start = shell_bfs_offset[ish]
        bf_b_start = shell_bfs_offset[jsh]
        bf_c_start = aux_shell_bfs_offset[ksh]

        nbf_a = bfs_nbfshell[ish]
        nbf_b = bfs_nbfshell[jsh]
        nbf_c = aux_bfs_nbfshell[ksh]

        atom_a = bfs_atoms[bf_a_start]
        atom_b = bfs_atoms[bf_b_start]
        atom_c = aux_bfs_atoms[bf_c_start]

        # If all three centers sit on the same atom the total derivative
        # vanishes by translational invariance.
        if atom_a == atom_b and atom_b == atom_c:
            continue

        la_shell = shell_l[ish]
        lb_shell = shell_l[jsh]
        lc_shell = aux_shell_l[ksh]
        bra_order = la_shell + lb_shell + 1  # +1 for the derivative
        aux_order = lc_shell + 1  # +1 for the derivative
        nroots = (la_shell + lb_shell + lc_shell + 1) // 2 + 1

        ibf_a0 = bf_a_start
        ibf_b0 = bf_b_start
        ibf_c0 = bf_c_start
        nprim_a = bfs_nprim[ibf_a0]
        nprim_b = bfs_nprim[ibf_b0]
        nprim_c = aux_bfs_nprim[ibf_c0]

        ax0 = bfs_coords[ibf_a0, 0]
        ay0 = bfs_coords[ibf_a0, 1]
        az0 = bfs_coords[ibf_a0, 2]
        bx0 = bfs_coords[ibf_b0, 0]
        by0 = bfs_coords[ibf_b0, 1]
        bz0 = bfs_coords[ibf_b0, 2]
        cx0 = aux_bfs_coords[ibf_c0, 0]
        cy0 = aux_bfs_coords[ibf_c0, 1]
        cz0 = aux_bfs_coords[ibf_c0, 2]

        xij0 = ax0 - bx0
        xij1 = ay0 - by0
        xij2 = az0 - bz0
        ijsq = xij0 * xij0 + xij1 * xij1 + xij2 * xij2

        roots = np.zeros(10, dtype=np.float64)
        weights = np.zeros(10, dtype=np.float64)
        gx = np.zeros((max_bra_order + 1, max_aux_order + 1), dtype=np.float64)
        gy = np.zeros((max_bra_order + 1, max_aux_order + 1), dtype=np.float64)
        gz = np.zeros((max_bra_order + 1, max_aux_order + 1), dtype=np.float64)
        shift_x = np.zeros((max_nbf_pair, max_nbf_aux_shell), dtype=np.float64)
        shift_y = np.zeros((max_nbf_pair, max_nbf_aux_shell), dtype=np.float64)
        shift_z = np.zeros((max_nbf_pair, max_nbf_aux_shell), dtype=np.float64)
        dshift_ax = np.zeros((max_nbf_pair, max_nbf_aux_shell), dtype=np.float64)
        dshift_ay = np.zeros((max_nbf_pair, max_nbf_aux_shell), dtype=np.float64)
        dshift_az = np.zeros((max_nbf_pair, max_nbf_aux_shell), dtype=np.float64)
        dshift_cx = np.zeros((max_nbf_pair, max_nbf_aux_shell), dtype=np.float64)
        dshift_cy = np.zeros((max_nbf_pair, max_nbf_aux_shell), dtype=np.float64)
        dshift_cz = np.zeros((max_nbf_pair, max_nbf_aux_shell), dtype=np.float64)
        dblock = np.zeros(
            (6, max_nbf_shell, max_nbf_shell, max_nbf_aux_shell),
            dtype=np.float64,
        )

        for iprim_a in range(nprim_a):
            alpha = bfs_expnts[ibf_a0, iprim_a]
            two_alpha = 2.0 * alpha
            for iprim_b in range(nprim_b):
                beta = bfs_expnts[ibf_b0, iprim_b]
                gamma_p = alpha + beta
                inv_gamma_p = 1.0 / gamma_p
                screen_ab = np.exp(-alpha * beta * inv_gamma_p * ijsq)
                if screen_ab < 1.0e-10:
                    continue

                px = (alpha * ax0 + beta * bx0) * inv_gamma_p
                py = (alpha * ay0 + beta * by0) * inv_gamma_p
                pz = (alpha * az0 + beta * bz0) * inv_gamma_p

                pqx = px - cx0
                pqy = py - cy0
                pqz = pz - cz0
                pqsq = pqx * pqx + pqy * pqy + pqz * pqz

                for iprim_c in range(nprim_c):
                    gamma_q = aux_bfs_expnts[ibf_c0, iprim_c]
                    two_gamma_q = 2.0 * gamma_q
                    rho = gamma_p * gamma_q / (gamma_p + gamma_q)
                    x = rho * pqsq
                    gamma_pq_sqrt = np.sqrt(gamma_p * gamma_q)

                    Roots(nroots, x, roots, weights)

                    rys_prefactor = 2.0 * np.sqrt(rho / pi)
                    for iroot in range(nroots):
                        root = roots[iroot]
                        Recur_3c2e_new(
                            gx,
                            root,
                            bra_order,
                            0,
                            aux_order,
                            0,
                            ax0,
                            bx0,
                            cx0,
                            0.0,
                            alpha,
                            beta,
                            gamma_q,
                            0.0,
                            gamma_p,
                            gamma_q,
                            alpha * beta,
                            gamma_pq_sqrt,
                        )
                        Recur_3c2e_new(
                            gy,
                            root,
                            bra_order,
                            0,
                            aux_order,
                            0,
                            ay0,
                            by0,
                            cy0,
                            0.0,
                            alpha,
                            beta,
                            gamma_q,
                            0.0,
                            gamma_p,
                            gamma_q,
                            alpha * beta,
                            gamma_pq_sqrt,
                        )
                        Recur_3c2e_new(
                            gz,
                            root,
                            bra_order,
                            0,
                            aux_order,
                            0,
                            az0,
                            bz0,
                            cz0,
                            0.0,
                            alpha,
                            beta,
                            gamma_q,
                            0.0,
                            gamma_p,
                            gamma_q,
                            alpha * beta,
                            gamma_pq_sqrt,
                        )

                        _build_shift_tables_grad(
                            shift_x,
                            shift_y,
                            shift_z,
                            dshift_ax,
                            dshift_ay,
                            dshift_az,
                            dshift_cx,
                            dshift_cy,
                            dshift_cz,
                            nbf_a,
                            nbf_b,
                            nbf_c,
                            bf_a_start,
                            bf_b_start,
                            bf_c_start,
                            bfs_lmn,
                            aux_bfs_lmn,
                            gx,
                            gy,
                            gz,
                            xij0,
                            xij1,
                            xij2,
                            two_alpha,
                            two_gamma_q,
                        )

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
                                for ic in range(nbf_c):
                                    ibf_c = bf_c_start + ic
                                    cc = (
                                        cb
                                        * aux_bfs_contr_prim_norms[ibf_c]
                                        * aux_bfs_coeffs[ibf_c, iprim_c]
                                        * aux_bfs_prim_norms[ibf_c, iprim_c]
                                    )
                                    w = cc * root_weight
                                    sx = shift_x[iab, ic]
                                    sy = shift_y[iab, ic]
                                    sz = shift_z[iab, ic]
                                    syz = sy * sz
                                    sxz = sx * sz
                                    sxy = sx * sy
                                    dblock[0, ia, ib, ic] += w * dshift_ax[iab, ic] * syz
                                    dblock[1, ia, ib, ic] += w * dshift_ay[iab, ic] * sxz
                                    dblock[2, ia, ib, ic] += w * dshift_az[iab, ic] * sxy
                                    dblock[3, ia, ib, ic] += w * dshift_cx[iab, ic] * syz
                                    dblock[4, ia, ib, ic] += w * dshift_cy[iab, ic] * sxz
                                    dblock[5, ia, ib, ic] += w * dshift_cz[iab, ic] * sxy

        # Contract the derivative shell block with D_ij * c_P
        ga_x = 0.0
        ga_y = 0.0
        ga_z = 0.0
        gc_x = 0.0
        gc_y = 0.0
        gc_z = 0.0
        pair_factor = 2.0 if ish != jsh else 1.0
        for ia in range(nbf_a):
            ibf_a = bf_a_start + ia
            for ib in range(nbf_b):
                ibf_b = bf_b_start + ib
                dm_ab = pair_factor * dmat[ibf_a, ibf_b]
                for ic in range(nbf_c):
                    ibf_c = bf_c_start + ic
                    dc = dm_ab * df_coeff[ibf_c]
                    ga_x += dc * dblock[0, ia, ib, ic]
                    ga_y += dc * dblock[1, ia, ib, ic]
                    ga_z += dc * dblock[2, ia, ib, ic]
                    gc_x += dc * dblock[3, ia, ib, ic]
                    gc_y += dc * dblock[4, ia, ib, ic]
                    gc_z += dc * dblock[5, ia, ib, ic]

        grad_threads[tid, atom_a, 0] += ga_x
        grad_threads[tid, atom_a, 1] += ga_y
        grad_threads[tid, atom_a, 2] += ga_z
        grad_threads[tid, atom_c, 0] += gc_x
        grad_threads[tid, atom_c, 1] += gc_y
        grad_threads[tid, atom_c, 2] += gc_z
        # Translational invariance: d/dB = -(d/dA + d/dC)
        grad_threads[tid, atom_b, 0] -= ga_x + gc_x
        grad_threads[tid, atom_b, 1] -= ga_y + gc_y
        grad_threads[tid, atom_b, 2] -= ga_z + gc_z

    grad = np.zeros((natoms, 3), dtype=np.float64)
    for t in range(nthreads):
        for iatom in range(natoms):
            for direction in range(3):
                grad[iatom, direction] += grad_threads[t, iatom, direction]

    return grad
