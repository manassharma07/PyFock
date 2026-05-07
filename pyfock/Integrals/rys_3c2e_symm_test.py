import numpy as np
from numba import njit, prange

from .rys_helpers import Roots, Recur_3c2e_new, Shift_3c2e


def _pack_basis(basis):
    nbf = basis.bfs_nao
    maxnprim = max(basis.bfs_nprim)

    coeffs = np.zeros((nbf, maxnprim), dtype=np.float64)
    expnts = np.zeros((nbf, maxnprim), dtype=np.float64)
    prim_norms = np.zeros((nbf, maxnprim), dtype=np.float64)

    for ibf in range(nbf):
        for iprim in range(basis.bfs_nprim[ibf]):
            coeffs[ibf, iprim] = basis.bfs_coeffs[ibf][iprim]
            expnts[ibf, iprim] = basis.bfs_expnts[ibf][iprim]
            prim_norms[ibf, iprim] = basis.bfs_prim_norms[ibf][iprim]

    shell_l = np.array(
        [basis.bfs_lm[bf0] for bf0 in basis.shell_bfs_offset],
        dtype=np.int32,
    )

    return (
        np.array(basis.bfs_coords, dtype=np.float64),
        np.array(basis.bfs_contr_prim_norms, dtype=np.float64),
        np.array(basis.bfs_lmn, dtype=np.int32),
        np.array(basis.bfs_nprim, dtype=np.int32),
        coeffs,
        prim_norms,
        expnts,
        shell_l,
        np.array(basis.shell_bfs_offset, dtype=np.int32),
        np.array(basis.bfs_nbfshell, dtype=np.int32),
    )


def rys_3c2e_symm_test(
    basis,
    auxbasis,
    slice=None,
    schwarz=False,
    threshold_schwarz=1e-9,
):
    """
    Shell-blocked Rys 3c2e integrals .

    This is an additive implementation of ``(AB|C)`` integrals.  It keeps the
    public shape and slicing behavior of :func:`rys_3c2e_symm`, but evaluates
    shell triplets instead of basis-function triples.  For each shell pair it
    builds primitive-pair data once, then reuses each Rys recurrence over the 
    whole Cartesian shell block.
    """
    if slice is None:
        slice = [0, basis.bfs_nao, 0, basis.bfs_nao, 0, auxbasis.bfs_nao]

    indx_start_a = int(slice[0])
    indx_end_a = int(slice[1])
    indx_start_b = int(slice[2])
    indx_end_b = int(slice[3])
    indx_start_c = int(slice[4])
    indx_end_c = int(slice[5])

    max_l_total = (
        2 * int(max(np.array(basis.shells, dtype=np.int32)) - 1)
        + int(max(np.array(auxbasis.shells, dtype=np.int32)) - 1)
    )
    if max_l_total // 2 + 1 > 10:
        from .rys_3c2e_symm import rys_3c2e_symm

        return rys_3c2e_symm(
            basis,
            auxbasis,
            slice=slice,
            schwarz=schwarz,
            threshold_schwarz=threshold_schwarz,
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

    nshells = len(basis.shells)
    nshells_aux = len(auxbasis.shells)

    shell_pair_bound = np.ones((1, 1), dtype=np.float64)
    aux_shell_bound = np.ones(1, dtype=np.float64)
    if schwarz:
        from .rys_2c2e_diag import rys_2c2e_diag
        from .schwarz_helpers import eri_4c2e_diag

        sqrt_ints4c2e_diag = np.sqrt(np.abs(eri_4c2e_diag(basis)))
        sqrt_diag_ints2c2e = np.sqrt(np.abs(rys_2c2e_diag(auxbasis)))

        shell_pair_bound = np.zeros((nshells, nshells), dtype=np.float64)
        for ish in range(nshells):
            ia0 = shell_bfs_offset[ish]
            nia = bfs_nbfshell[ish]
            for jsh in range(ish + 1):
                ib0 = shell_bfs_offset[jsh]
                nib = bfs_nbfshell[jsh]
                bound = 0.0
                for ia in range(ia0, ia0 + nia):
                    for ib in range(ib0, ib0 + nib):
                        if sqrt_ints4c2e_diag[ia, ib] > bound:
                            bound = sqrt_ints4c2e_diag[ia, ib]
                shell_pair_bound[ish, jsh] = bound
                shell_pair_bound[jsh, ish] = bound

        aux_shell_bound = np.zeros(nshells_aux, dtype=np.float64)
        for ksh in range(nshells_aux):
            ic0 = aux_shell_bfs_offset[ksh]
            nic = aux_bfs_nbfshell[ksh]
            bound = 0.0
            for ic in range(ic0, ic0 + nic):
                if sqrt_diag_ints2c2e[ic] > bound:
                    bound = sqrt_diag_ints2c2e[ic]
            aux_shell_bound[ksh] = bound

    return rys_3c2e_symm_test_internal(
        basis.bfs_nao,
        auxbasis.bfs_nao,
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
        indx_start_a,
        indx_end_a,
        indx_start_b,
        indx_end_b,
        indx_start_c,
        indx_end_c,
        schwarz,
        shell_pair_bound,
        aux_shell_bound,
        threshold_schwarz,
    )


@njit(cache=True, fastmath=True, nogil=True, error_model="numpy", inline="always")
def _shell_intersects(start, count, lo, hi):
    return start < hi and start + count > lo


@njit(cache=True, fastmath=True, nogil=True, error_model="numpy", inline="always")
def _build_shift_tables(
    shift_x,
    shift_y,
    shift_z,
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
):
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
                shift_x[iab, ic] = Shift_3c2e(gx, ax, bx, cx, 0, xij0)
                shift_y[iab, ic] = Shift_3c2e(gy, ay, by, cy, 0, xij1)
                shift_z[iab, ic] = Shift_3c2e(gz, az, bz, cz, 0, xij2)


@njit(parallel=True, cache=True, fastmath=True, nogil=True, error_model="numpy")
def rys_3c2e_symm_test_internal(
    nbf,
    naux,
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
    indx_start_a,
    indx_end_a,
    indx_start_b,
    indx_end_b,
    indx_start_c,
    indx_end_c,
    schwarz,
    shell_pair_bound,
    aux_shell_bound,
    threshold_schwarz,
):
    three_c2e = np.zeros(
        (
            indx_end_a - indx_start_a,
            indx_end_b - indx_start_b,
            indx_end_c - indx_start_c,
        ),
        dtype=np.float64,
    )

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

    max_bra_order = 2 * max_l_bra
    max_aux_order = max_l_aux
    max_nbf_pair = max_nbf_shell * max_nbf_shell

    for task_idx in prange(n_tasks):
        ab_idx = task_idx // nshells_aux
        ksh = task_idx - ab_idx * nshells_aux

        ish = ab_shell_a[ab_idx]
        jsh = ab_shell_b[ab_idx]

        bf_a_start = shell_bfs_offset[ish]
        bf_b_start = shell_bfs_offset[jsh]
        bf_c_start = aux_shell_bfs_offset[ksh]

        nbf_a = bfs_nbfshell[ish]
        nbf_b = bfs_nbfshell[jsh]
        nbf_c = aux_bfs_nbfshell[ksh]

        if not _shell_intersects(bf_c_start, nbf_c, indx_start_c, indx_end_c):
            continue

        ab_in_requested_order = (
            _shell_intersects(bf_a_start, nbf_a, indx_start_a, indx_end_a)
            and _shell_intersects(bf_b_start, nbf_b, indx_start_b, indx_end_b)
        )
        ba_in_requested_order = (
            ish != jsh
            and _shell_intersects(bf_b_start, nbf_b, indx_start_a, indx_end_a)
            and _shell_intersects(bf_a_start, nbf_a, indx_start_b, indx_end_b)
        )
        if not (ab_in_requested_order or ba_in_requested_order):
            continue

        if schwarz:
            if shell_pair_bound[ish, jsh] * aux_shell_bound[ksh] < threshold_schwarz:
                continue

        la_shell = shell_l[ish]
        lb_shell = shell_l[jsh]
        lc_shell = aux_shell_l[ksh]
        bra_order = la_shell + lb_shell
        aux_order = lc_shell
        nroots = (bra_order + aux_order) // 2 + 1

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
        shell_block = np.zeros(
            (max_nbf_shell, max_nbf_shell, max_nbf_aux_shell),
            dtype=np.float64,
        )

        for iprim_a in range(nprim_a):
            alpha = bfs_expnts[ibf_a0, iprim_a]
            for iprim_b in range(nprim_b):
                beta = bfs_expnts[ibf_b0, iprim_b]
                gamma_p = alpha + beta
                inv_gamma_p = 1.0 / gamma_p
                screen_ab = np.exp(-alpha * beta * inv_gamma_p * ijsq)
                if screen_ab < 1.0e-8:
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

                        _build_shift_tables(
                            shift_x,
                            shift_y,
                            shift_z,
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
                                    shell_block[ia, ib, ic] += (
                                        cc
                                        * root_weight
                                        * shift_x[iab, ic]
                                        * shift_y[iab, ic]
                                        * shift_z[iab, ic]
                                    )

        for ia in range(nbf_a):
            ibf_a = bf_a_start + ia
            a_in_a = indx_start_a <= ibf_a < indx_end_a
            a_in_b = indx_start_b <= ibf_a < indx_end_b
            for ib in range(nbf_b):
                ibf_b = bf_b_start + ib
                b_in_b = indx_start_b <= ibf_b < indx_end_b
                b_in_a = indx_start_a <= ibf_b < indx_end_a
                for ic in range(nbf_c):
                    ibf_c = bf_c_start + ic
                    if not (indx_start_c <= ibf_c < indx_end_c):
                        continue

                    val = shell_block[ia, ib, ic]

                    if a_in_a and b_in_b:
                        three_c2e[
                            ibf_a - indx_start_a,
                            ibf_b - indx_start_b,
                            ibf_c - indx_start_c,
                        ] = val

                    if ish != jsh and b_in_a and a_in_b:
                        three_c2e[
                            ibf_b - indx_start_a,
                            ibf_a - indx_start_b,
                            ibf_c - indx_start_c,
                        ] = val

    return three_c2e
