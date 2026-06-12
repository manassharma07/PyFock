import numpy as np
import numba
from numba import njit, prange

from .rys_helpers import Roots, Recur_3c2e_new, Shift_3c2e


def rys_2c2e_grad_contract(auxbasis, df_coeff, ncores=None, threshold=1e-14):
    """
    Contracted nuclear gradient of two-center two-electron (2c2e) integrals.

    Computes

        grad[iatom, xyz] = sum_{PQ} c_P * c_Q * d(P|Q)/dR_{iatom, xyz}

    without storing the derivative tensor. This is the metric-derivative part
    of the density-fitted Coulomb gradient (the caller multiplies by -0.5).

    The derivative with respect to the second center is obtained from
    translational invariance: d/dQ = -d/dP. Pairs with both functions on the
    same atom therefore do not contribute.

    Parameters
    ----------
    auxbasis : Basis
        Auxiliary basis set object.
    df_coeff : ndarray (naux,)
        Density fitting coefficients c_P.
    ncores : int, optional
        Number of threads for Numba. If None the current setting is used.
    threshold : float, optional
        Skip pairs with |c_P * c_Q| below this threshold.

    Returns
    -------
    grad : ndarray (natoms, 3)
        The contracted 2c2e gradient contribution.
    """
    if ncores is not None:
        numba.set_num_threads(ncores)

    bfs_coords = np.array(auxbasis.bfs_coords, dtype=np.float64)
    bfs_contr_prim_norms = np.array(auxbasis.bfs_contr_prim_norms, dtype=np.float64)
    bfs_lmn = np.array(auxbasis.bfs_lmn, dtype=np.int32)
    bfs_nprim = np.array(auxbasis.bfs_nprim, dtype=np.int32)
    bfs_atoms = np.array(auxbasis.bfs_atoms, dtype=np.int32)
    natoms = int(bfs_atoms.max()) + 1

    maxnprim = max(auxbasis.bfs_nprim)
    bfs_coeffs = np.zeros((auxbasis.bfs_nao, maxnprim), dtype=np.float64)
    bfs_expnts = np.zeros((auxbasis.bfs_nao, maxnprim), dtype=np.float64)
    bfs_prim_norms = np.zeros((auxbasis.bfs_nao, maxnprim), dtype=np.float64)
    for i in range(auxbasis.bfs_nao):
        for j in range(auxbasis.bfs_nprim[i]):
            bfs_coeffs[i, j] = auxbasis.bfs_coeffs[i][j]
            bfs_expnts[i, j] = auxbasis.bfs_expnts[i][j]
            bfs_prim_norms[i, j] = auxbasis.bfs_prim_norms[i][j]

    df_coeff = np.ascontiguousarray(df_coeff, dtype=np.float64)
    nthreads = numba.get_num_threads()

    return rys_2c2e_grad_contract_internal(
        natoms,
        nthreads,
        auxbasis.bfs_nao,
        bfs_coords,
        bfs_contr_prim_norms,
        bfs_lmn,
        bfs_nprim,
        bfs_coeffs,
        bfs_prim_norms,
        bfs_expnts,
        bfs_atoms,
        df_coeff,
        threshold,
    )


@njit(parallel=True, cache=True, fastmath=True, nogil=True, error_model="numpy")
def rys_2c2e_grad_contract_internal(
    natoms,
    nthreads,
    naux,
    bfs_coords,
    bfs_contr_prim_norms,
    bfs_lmn,
    bfs_nprim,
    bfs_coeffs,
    bfs_prim_norms,
    bfs_expnts,
    bfs_atoms,
    df_coeff,
    threshold,
):
    pi = 3.141592653589793

    max_l = 0
    for i in range(naux):
        l_tot = bfs_lmn[i, 0] + bfs_lmn[i, 1] + bfs_lmn[i, 2]
        if l_tot > max_l:
            max_l = l_tot

    grad_threads = np.zeros((nthreads, natoms, 3), dtype=np.float64)

    for i in prange(naux):
        tid = numba.get_thread_id()

        I = bfs_coords[i]
        Ni = bfs_contr_prim_norms[i]
        lmni = bfs_lmn[i]
        la, ma, na = lmni[0], lmni[1], lmni[2]
        nprimi = bfs_nprim[i]
        atom_i = bfs_atoms[i]
        ci = df_coeff[i]

        roots = np.zeros(10, dtype=np.float64)
        weights = np.zeros(10, dtype=np.float64)
        # bra order on axis 0 (+1 for derivative), ket order on axis 1
        G = np.zeros((2 * max_l + 2, max_l + 1), dtype=np.float64)

        gx = 0.0
        gy = 0.0
        gz = 0.0

        for k in range(i):  # strictly lower triangle; diagonal has no net force
            atom_k = bfs_atoms[k]
            if atom_i == atom_k:
                continue

            cik = ci * df_coeff[k]
            if abs(cik) < threshold:
                continue

            K = bfs_coords[k]
            Nk = bfs_contr_prim_norms[k]
            lmnk = bfs_lmn[k]
            lc, mc, nc = lmnk[0], lmnk[1], lmnk[2]
            nprimk = bfs_nprim[k]

            norder = (la + ma + na + 1 + lc + mc + nc) // 2 + 1

            PQx = I[0] - K[0]
            PQy = I[1] - K[1]
            PQz = I[2] - K[2]
            pqsq = PQx * PQx + PQy * PQy + PQz * PQz

            gax = 0.0
            gay = 0.0
            gaz = 0.0

            tempcoeff1 = Ni * Nk

            for ik in range(nprimi):
                alpha = bfs_expnts[i, ik]
                two_alpha = 2.0 * alpha
                dik = bfs_coeffs[i, ik]
                Nik = bfs_prim_norms[i, ik]
                tempcoeff2 = tempcoeff1 * dik * Nik
                gamma_p = alpha

                for kk in range(nprimk):
                    gamma_q = bfs_expnts[k, kk]
                    dkk = bfs_coeffs[k, kk]
                    Nkk = bfs_prim_norms[k, kk]
                    tempcoeff3 = tempcoeff2 * dkk * Nkk

                    rho = gamma_p * gamma_q / (gamma_p + gamma_q)
                    x = rho * pqsq
                    gamma_pq_sqrt = np.sqrt(gamma_p * gamma_q)

                    Roots(norder, x, roots, weights)
                    rys_prefactor = 2.0 * np.sqrt(rho / pi) * tempcoeff3

                    for iroot in range(norder):
                        root = roots[iroot]
                        root_weight = rys_prefactor * weights[iroot]

                        # The 2c2e integral (P|Q) is a 3c2e integral with a
                        # dummy s-function at the bra: alpha_j = 0, so xj is
                        # irrelevant (we pass I so that xij = 0).
                        Recur_3c2e_new(
                            G, root, la + 1, 0, lc, 0,
                            I[0], I[0], K[0], 0.0,
                            alpha, 0.0, gamma_q, 0.0,
                            gamma_p, gamma_q, 0.0, gamma_pq_sqrt,
                        )
                        sx = G[la, lc]
                        dax = two_alpha * G[la + 1, lc]
                        if la > 0:
                            dax -= la * G[la - 1, lc]

                        Recur_3c2e_new(
                            G, root, ma + 1, 0, mc, 0,
                            I[1], I[1], K[1], 0.0,
                            alpha, 0.0, gamma_q, 0.0,
                            gamma_p, gamma_q, 0.0, gamma_pq_sqrt,
                        )
                        sy = G[ma, mc]
                        day = two_alpha * G[ma + 1, mc]
                        if ma > 0:
                            day -= ma * G[ma - 1, mc]

                        Recur_3c2e_new(
                            G, root, na + 1, 0, nc, 0,
                            I[2], I[2], K[2], 0.0,
                            alpha, 0.0, gamma_q, 0.0,
                            gamma_p, gamma_q, 0.0, gamma_pq_sqrt,
                        )
                        sz = G[na, nc]
                        daz = two_alpha * G[na + 1, nc]
                        if na > 0:
                            daz -= na * G[na - 1, nc]

                        gax += root_weight * dax * sy * sz
                        gay += root_weight * sx * day * sz
                        gaz += root_weight * sx * sy * daz

            # Both (P|Q) and (Q|P) appear in the double sum, so weight by 2.
            gx = 2.0 * cik * gax
            gy = 2.0 * cik * gay
            gz = 2.0 * cik * gaz

            grad_threads[tid, atom_i, 0] += gx
            grad_threads[tid, atom_i, 1] += gy
            grad_threads[tid, atom_i, 2] += gz
            # Translational invariance: d/dQ = -d/dP
            grad_threads[tid, atom_k, 0] -= gx
            grad_threads[tid, atom_k, 1] -= gy
            grad_threads[tid, atom_k, 2] -= gz

    grad = np.zeros((natoms, 3), dtype=np.float64)
    for t in range(nthreads):
        for iatom in range(natoms):
            for direction in range(3):
                grad[iatom, direction] += grad_threads[t, iatom, direction]

    return grad
