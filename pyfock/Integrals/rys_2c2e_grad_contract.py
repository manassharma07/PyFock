import numpy as np
import numba
from numba import njit, prange

from .rys_helpers import Roots, Recur_3c2e_new
from .df_algo10_helpers import pack_basis_arrays, aux_shell_arrays
from .df_algo11_helpers import _lpt_bins, _bf_coef_table


def rys_2c2e_grad_contract(auxbasis, df_coeff=None, ncores=None, threshold=1e-14, weights=None):
    """
    Contracted nuclear gradient of two-center two-electron (2c2e) integrals.

    Computes

        grad[iatom, xyz] = sum_{PQ} W_PQ * d(P|Q)/dR_{iatom, xyz}

    without storing the derivative tensor, either for ``W_PQ = c_P c_Q`` (``df_coeff``: the
    metric-derivative part of the density-fitted Coulomb gradient, which the caller multiplies by
    -0.5) or for a general symmetric matrix ``weights`` (the metric part of the RI exchange
    gradient).

    The derivative with respect to the second center is obtained from
    translational invariance: d/dQ = -d/dP. Pairs with both functions on the
    same atom therefore do not contribute. The recursions are built once per
    shell pair, primitive pair and root for all components of the two shells,
    and the shell pairs are distributed over the threads in balanced bins.

    Parameters
    ----------
    auxbasis : Basis
        Auxiliary basis set object.
    df_coeff : ndarray (naux,), optional
        Density fitting coefficients c_P (used when ``weights`` is not given).
    ncores : int, optional
        Number of threads for Numba. If None the current setting is used.
    threshold : float, optional
        Skip shell pairs whose largest ``|W_PQ|`` is below this threshold.
    weights : ndarray (naux, naux), optional
        Symmetric weight matrix ``W`` (Cartesian auxiliary functions).

    Returns
    -------
    grad : ndarray (natoms, 3)
        The contracted 2c2e gradient contribution.
    """
    if ncores is not None:
        numba.set_num_threads(ncores)
    naux = auxbasis.bfs_nao
    if weights is not None:
        W = np.ascontiguousarray(weights, dtype=np.float64)
        if W.shape != (naux, naux):
            raise ValueError('weights must have shape (%d, %d)' % (naux, naux))
        coeff = np.zeros(1)
        use_matrix = True
    else:
        if df_coeff is None:
            raise ValueError('Pass either df_coeff or weights.')
        coeff = np.ascontiguousarray(df_coeff, dtype=np.float64)
        W = np.zeros((1, 1))
        use_matrix = False

    packed = pack_basis_arrays(auxbasis)
    coords, _, lmn, nprim, _, _, expnts = packed
    coef = _bf_coef_table(auxbasis, packed)
    shell_off, shell_nbf, shell_l = aux_shell_arrays(auxbasis)
    bfs_atoms = np.asarray(auxbasis.bfs_atoms, dtype=np.int64)
    shell_atom = np.ascontiguousarray(bfs_atoms[shell_off])
    natoms = int(bfs_atoms.max()) + 1

    # shell pairs K > L on different atoms (pairs on one atom have no net force)
    pair_K, pair_L = np.tril_indices(shell_off.shape[0], -1)
    keep = shell_atom[pair_K] != shell_atom[pair_L]
    pair_K = np.ascontiguousarray(pair_K[keep], dtype=np.int64)
    pair_L = np.ascontiguousarray(pair_L[keep], dtype=np.int64)
    nrt = (shell_l[pair_K] + shell_l[pair_L] + 1) // 2 + 1
    cost = (nprim[shell_off[pair_K]] * nprim[shell_off[pair_L]] * nrt
            * (12 + shell_nbf[pair_K] * shell_nbf[pair_L])).astype(np.float64)
    nthreads = numba.get_num_threads()
    bin_off, bin_items = _lpt_bins(np.arange(pair_K.shape[0]), cost, nthreads)
    return _grad_2c2e_pass(bin_off, bin_items, pair_K, pair_L, coeff, W, use_matrix, float(threshold),
                           coords, lmn, expnts, coef, nprim, shell_off, shell_nbf, shell_l, shell_atom,
                           natoms, int(shell_l.max()), int(shell_nbf.max()))


@njit(parallel=True, cache=True, fastmath=True, nogil=True, error_model="numpy", boundscheck=False)
def _grad_2c2e_pass(bin_off, bin_items, pair_K, pair_L, coeff, W, use_matrix, threshold,
                    coords, lmn, expnts, coef, nprim, shell_off, shell_nbf, shell_l, shell_atom,
                    natoms, lmax, maxnbf):
    """Bins of shell pairs ``(K, L)`` -> ``sum_PQ W_PQ d(P|Q)/dR`` as ``(natoms, 3)``."""
    pi = 3.141592653589793
    nbins = bin_off.shape[0] - 1
    gbin = np.zeros((nbins, natoms, 3))
    for b in prange(nbins):
        roots = np.zeros(12)
        rweights = np.zeros(12)
        G = np.zeros((lmax + 2, lmax + 1))
        Gx = np.zeros((lmax + 2, lmax + 1))
        Gy = np.zeros((lmax + 2, lmax + 1))
        Gz = np.zeros((lmax + 2, lmax + 1))
        Wb = np.zeros((maxnbf, maxnbf))
        for idx in range(bin_off[b], bin_off[b + 1]):
            p = bin_items[idx]
            K = pair_K[p]
            L = pair_L[p]
            k0 = shell_off[K]
            l0 = shell_off[L]
            nK = shell_nbf[K]
            nL = shell_nbf[L]
            wmax = 0.0
            for i in range(nK):
                for k in range(nL):
                    if use_matrix:
                        w = W[k0 + i, l0 + k]
                    else:
                        w = coeff[k0 + i] * coeff[l0 + k]
                    Wb[i, k] = w
                    if abs(w) > wmax:
                        wmax = abs(w)
            if wmax < threshold:
                continue
            lK = shell_l[K]
            lL = shell_l[L]
            norder = (lK + 1 + lL) // 2 + 1
            Px = coords[k0, 0]
            Py = coords[k0, 1]
            Pz = coords[k0, 2]
            Qx = coords[l0, 0]
            Qy = coords[l0, 1]
            Qz = coords[l0, 2]
            pqsq = (Px - Qx) ** 2 + (Py - Qy) ** 2 + (Pz - Qz) ** 2
            gx = 0.0
            gy = 0.0
            gz = 0.0
            for ik in range(nprim[k0]):
                alpha = expnts[k0, ik]
                two_alpha = 2.0 * alpha
                for il in range(nprim[l0]):
                    gamma_q = expnts[l0, il]
                    rho = alpha * gamma_q / (alpha + gamma_q)
                    gpq_sqrt = np.sqrt(alpha * gamma_q)
                    Roots(norder, rho * pqsq, roots, rweights)
                    pref = 2.0 * np.sqrt(rho / pi)
                    for ir in range(norder):
                        t = roots[ir]
                        # (P|Q) as a three-center integral with a dummy s function (exponent 0) at
                        # the bra, one order higher on P for its derivative
                        Recur_3c2e_new(G, t, lK + 1, 0, lL, 0, Px, Px, Qx, 0.0,
                                       alpha, 0.0, gamma_q, 0.0, alpha, gamma_q, 0.0, gpq_sqrt)
                        for a in range(lK + 2):
                            for c in range(lL + 1):
                                Gx[a, c] = G[a, c]
                        Recur_3c2e_new(G, t, lK + 1, 0, lL, 0, Py, Py, Qy, 0.0,
                                       alpha, 0.0, gamma_q, 0.0, alpha, gamma_q, 0.0, gpq_sqrt)
                        for a in range(lK + 2):
                            for c in range(lL + 1):
                                Gy[a, c] = G[a, c]
                        Recur_3c2e_new(G, t, lK + 1, 0, lL, 0, Pz, Pz, Qz, 0.0,
                                       alpha, 0.0, gamma_q, 0.0, alpha, gamma_q, 0.0, gpq_sqrt)
                        for a in range(lK + 2):
                            for c in range(lL + 1):
                                Gz[a, c] = G[a, c]
                        wr = pref * rweights[ir]
                        for i in range(nK):
                            la = lmn[k0 + i, 0]
                            ma = lmn[k0 + i, 1]
                            na = lmn[k0 + i, 2]
                            ci = wr * coef[k0 + i, ik]
                            for k in range(nL):
                                w = Wb[i, k]
                                if w == 0.0:
                                    continue
                                lc = lmn[l0 + k, 0]
                                mc = lmn[l0 + k, 1]
                                nc = lmn[l0 + k, 2]
                                sx = Gx[la, lc]
                                sy = Gy[ma, mc]
                                sz = Gz[na, nc]
                                dax = two_alpha * Gx[la + 1, lc]
                                day = two_alpha * Gy[ma + 1, mc]
                                daz = two_alpha * Gz[na + 1, nc]
                                if la > 0:
                                    dax -= la * Gx[la - 1, lc]
                                if ma > 0:
                                    day -= ma * Gy[ma - 1, mc]
                                if na > 0:
                                    daz -= na * Gz[na - 1, nc]
                                f = ci * coef[l0 + k, il] * w
                                gx += f * dax * sy * sz
                                gy += f * sx * day * sz
                                gz += f * sx * sy * daz
            # both (P|Q) and (Q|P) appear in the double sum; d/dQ = -d/dP
            atom_k = shell_atom[K]
            atom_l = shell_atom[L]
            gbin[b, atom_k, 0] += 2.0 * gx
            gbin[b, atom_k, 1] += 2.0 * gy
            gbin[b, atom_k, 2] += 2.0 * gz
            gbin[b, atom_l, 0] -= 2.0 * gx
            gbin[b, atom_l, 1] -= 2.0 * gy
            gbin[b, atom_l, 2] -= 2.0 * gz
    grad = np.zeros((natoms, 3))
    for b in range(nbins):
        for a in range(natoms):
            for d in range(3):
                grad[a, d] += gbin[b, a, d]
    return grad
