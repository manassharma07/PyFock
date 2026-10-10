"""
Nuclear gradient of the density-fitted Coulomb term with the near/far split of algorithm 12
(``DF_algo=12``, :mod:`~pyfock.Integrals.df_algo12_helpers`).

The DF Coulomb energy of a converged density is

    E_J = sum_ijP D_ij c_P (ij|P) - 1/2 sum_PQ c_P (P|Q) c_Q ,

so its explicit nuclear derivative needs ``sum_ijP D_ij c_P d(ij|P)/dR`` (this module) besides the
two-center term (:func:`~pyfock.Integrals.rys_2c2e_grad_contract`).  Algorithm 12 evaluates
``(ij|P)`` as a near field, the Rys integrals of the primitive pairs whose branch is not well
separated from the auxiliary shell of ``P``, plus a far field of multipole expansions.  This module
differentiates exactly that split, with the plan metadata of algorithm 12 (screening, branches,
near/far classification, profitability, group moments), so the forces are the gradient of the
energy the SCF computed.  The only dependence left out is that of the branch expansion centres on
the geometry, which enters through the truncation error of the expansions alone.

* **Near field.**  A shell-pair-blocked derivative kernel in the style of the algorithm-11/12
  integral kernel: the primitive pairs of a shell pair are screened once, the Rys recursions and the
  horizontal shifts are built once per primitive triple and root for all components of the
  shells, and the derivatives with respect to the bra centre ``A`` and the auxiliary centre ``C``
  are contracted on the fly with ``D_ij c_P``; the ``B`` derivative follows from translational
  invariance.  Primitive pairs whose branch is far field for the auxiliary shell are skipped,
  exactly as in the integral kernel, and shell pairs on one atom only need the ``C`` derivative.
  Besides the Schwarz test of the plan, a (pair, auxiliary shell) block is skipped when
  ``Q_pair Q_aux max|D| max|c|`` is below ``threshold_grad``.

* **Far field, auxiliary centres.**  Moving an auxiliary shell rigidly changes its moments about its
  atom through the first-order term of the multipole translation only, so its force is the
  contraction of those moments, one order higher, with the local expansion at the atom of the
  potential of the density-weighted branch moments (built one order higher than the SCF needs).

* **Far field, density centres.**  The derivative with respect to the centre ``A`` of the bra
  function is the interaction of the derivative distribution ``(d phi_i / dA) phi_j`` with the
  far-field potential: its exact moments about the primitive-pair centre (orders up to
  ``l_a + l_b + 1``, from 1D Gaussian moments with ``a`` raised by one) are contracted with the
  local expansion of the potential at that centre.  The ``B`` derivative of every (shell pair,
  branch) entry then follows from translational invariance with the branch centre moving with the
  distribution, which needs the branch local expansion one order above ``lmax``.  This makes the
  far-field forces sum to zero to rounding, like the near-field ones.

The far field does not need the branch-centred row moments ``Mtil`` of the SCF passes: every group
is translated once, here, so the plan is built with ``low_memory=True`` and no near-field values.

The near-field kernel also contracts a general weight per function pair and auxiliary function,
``sum_{ij,P} Gamma^P_ij d(ij|P)/dR`` (:func:`grad_contract_rows`), which the RI exchange gradient
needs (algorithm 11 only: exchange contracts the integrals themselves, not their multipoles).
"""

import numpy as np
import numba
from numba import njit, prange
from timeit import default_timer as timer

from .rys_helpers import Roots, Recur_3c2e_new, Shift_3c2e
from .df_algo10_helpers import STRICT_PAIR_CUTOFF, EXP_ARG_CUTOFF
from .df_algo11_helpers import _lpt_bins
from . import df_algo12_helpers as a12
from . import multipole_helpers as mp

__all__ = ['build_grad_plan', 'grad_contract', 'grad_contract_rows']

PI = 3.141592653589793

# The one l = 1 real harmonic whose derivative along x, y and z does not vanish, and its value:
# R_{1,1} = -x/sqrt(2), R_{1,-1} = -y/sqrt(2), R_{1,0} = z (multipole_helpers.regular_harmonics).
_D1_INDEX = np.array([3, 1, 2], dtype=np.int64)
_D1_VALUE = np.array([-0.7071067811865476, -0.7071067811865476, 1.0])


@njit(cache=True, fastmath=True, nogil=True, error_model='numpy', boundscheck=False)
def _first_order_dot(M, l_src, L, pair_off, ent_LM, ent_coef, n_big, out):
    """
    ``out[d] = sum_LM dM_d[LM] L[LM]``, where ``dM_d`` is the first-order change of the moments ``M``
    (orders ``<= l_src`` about a fixed centre) when their distribution moves along ``d``: the
    ``R_1k`` term of :func:`~pyfock.Integrals.multipole_helpers.translate_moments`, read from the
    table with ``(1, k)`` as the small index (the coefficients are symmetric in the two).  ``L`` must
    hold orders up to ``l_src + 1`` and the table's big index must reach ``l_src``.
    """
    nsrc = (l_src + 1) * (l_src + 1)
    for d in range(3):
        k1 = _D1_INDEX[d]
        acc = 0.0
        for lm in range(nsrc):
            v = M[lm]
            if v == 0.0:
                continue
            p = k1 * n_big + lm
            s = 0.0
            for e in range(pair_off[p], pair_off[p + 1]):
                s += ent_coef[e] * L[ent_LM[e]]
            acc += v * s
        out[d] = _D1_VALUE[d] * acc


@njit(cache=True, fastmath=True, nogil=True, error_model='numpy', boundscheck=False)
def _row_weights(I, J, dmat, sqrt_ints4c2e_diag, strict_schwarz, shell_off, shell_nbf, rc):
    """
    ``rc[r]`` = weight of function pair ``r`` of shell pair ``(I, J)`` in ``sum_ij D_ij (ij|P)``
    (``D_ij + D_ji``, or ``D_ii`` on the diagonal), zero for pairs the strict cut-off drops; rows in
    the algorithm-11 convention.  Returns the largest absolute weight.
    """
    a0 = shell_off[I]
    b0 = shell_off[J]
    nA = shell_nbf[I]
    nB = shell_nbf[J]
    diag_pair = I == J
    wmax = 0.0
    for ia in range(nA):
        i = a0 + ia
        ibmax = ia + 1 if diag_pair else nB
        for ib in range(ibmax):
            j = b0 + ib
            r = (ia * (ia + 1)) // 2 + ib if diag_pair else ia * nB + ib
            sij = sqrt_ints4c2e_diag[i, j]
            if strict_schwarz and sij * sij < STRICT_PAIR_CUTOFF:
                rc[r] = 0.0
                continue
            w = dmat[i, j] if i == j else dmat[i, j] + dmat[j, i]
            rc[r] = w
            if abs(w) > wmax:
                wmax = abs(w)
    return wmax


# ----------------------------------------------------------------------------
# Near field
# ----------------------------------------------------------------------------
@njit(nogil=True, cache=True, fastmath=True, error_model="numpy", boundscheck=False)
def _pair_grad_near(I, J, rc, wmax, coeff, cmax_aux, thr_grad, atom_a, atom_b, aux_atom, gacc,
                    bfs_coords, bfs_lmn, bfs_nprim, bfs_expnts, bf_coef, shell_off, shell_nbf, shell_l,
                    aux_coords, aux_lmn, aux_nprim, aux_expnts, aux_coef, aux_off, aux_nbf, aux_l,
                    Qp, Q_aux, threshold, pp_group_p, grp_branch_p, ff,
                    roots, weights, gx, gy, gz, Sx, Sy, Sz, dAx, dAy, dAz, dCx, dCy, dCz, Wk, Wp,
                    pp_alpha, pp_beta, pp_gamma, pp_px, pp_py, pp_pz, pp_ip, pp_jp, pp_br,
                    ax_, ay_, az_, bx_, by_, bz_, cx_, cy_, cz_,
                    mode, Grows, row_of, sao_fit, c2s_flat, c2s_off, sph_off, aux_nsph):
    """
    Add to ``gacc[atom, :]`` the near-field part of ``sum_{ij in (I,J), P} w_ij D_ij c_P d(ij|P)/dR``
    with the row weights ``rc`` of :func:`_row_weights`: the same primitive pairs, auxiliary shells and
    Rys quadrature as :func:`~pyfock.Integrals.df_algo12_helpers._compute_pair_rows`, differentiated
    with respect to ``A`` and ``C`` (``B`` by translational invariance).

    ``mode = 1`` contracts with a general weight per function pair and auxiliary function instead,
    ``sum_{ij, P} Gamma^P_ij d(ij|P)/dR`` over both triangles (the RI exchange gradient): row
    ``row_of[i, j]`` of ``Grows`` holds ``Gamma^P_ij`` in the fit space (spherical when ``sao_fit``,
    mapped onto the Cartesian functions with the per-shell tables ``c2s_*``), ``-1`` marks a pair the
    energy leaves out, and a block is skipped when ``Q_pair Q_aux max|Gamma|`` is below ``thr_grad``.
    """
    nA = shell_nbf[I]
    nB = shell_nbf[J]
    a0 = shell_off[I]
    b0 = shell_off[J]
    lA = shell_l[I]
    lB = shell_l[J]
    nprimA = bfs_nprim[a0]
    nprimB = bfs_nprim[b0]
    Ax = bfs_coords[a0, 0]
    Ay = bfs_coords[a0, 1]
    Az = bfs_coords[a0, 2]
    Bx = bfs_coords[b0, 0]
    By = bfs_coords[b0, 1]
    Bz = bfs_coords[b0, 2]
    xij0 = Ax - Bx
    xij1 = Ay - By
    xij2 = Az - Bz
    ijsq = xij0 * xij0 + xij1 * xij1 + xij2 * xij2
    diag_pair = I == J
    # with both functions on one atom only the auxiliary derivative is needed: d/dA + d/dB = -d/dC
    need_A = atom_a != atom_b
    la_tab = lA + 1 if need_A else lA

    for ia in range(nA):
        ax_[ia] = bfs_lmn[a0 + ia, 0]
        ay_[ia] = bfs_lmn[a0 + ia, 1]
        az_[ia] = bfs_lmn[a0 + ia, 2]
    for ib in range(nB):
        bx_[ib] = bfs_lmn[b0 + ib, 0]
        by_[ib] = bfs_lmn[b0 + ib, 1]
        bz_[ib] = bfs_lmn[b0 + ib, 2]

    # surviving primitive pairs, in the order the plan's pp_group indexes them
    npp = 0
    for ip in range(nprimA):
        alpha = bfs_expnts[a0, ip]
        for jp in range(nprimB):
            beta = bfs_expnts[b0, jp]
            gamma_p = alpha + beta
            if alpha * beta / gamma_p * ijsq > EXP_ARG_CUTOFF:
                continue
            pp_alpha[npp] = alpha
            pp_beta[npp] = beta
            pp_gamma[npp] = gamma_p
            pp_px[npp] = (alpha * Ax + beta * Bx) / gamma_p
            pp_py[npp] = (alpha * Ay + beta * By) / gamma_p
            pp_pz[npp] = (alpha * Az + beta * Bz) / gamma_p
            pp_ip[npp] = ip
            pp_jp[npp] = jp
            pp_br[npp] = grp_branch_p[pp_group_p[npp]]
            npp += 1

    for K in range(aux_off.shape[0]):
        QQ = Qp * Q_aux[K]
        if QQ <= threshold:
            continue
        if mode == 0 and QQ * wmax * cmax_aux[K] < thr_grad:
            continue
        atom_c = aux_atom[K]
        if not need_A and atom_c == atom_a:
            continue                       # one-centre triple: its total derivative vanishes
        nf_any = False
        for q in range(npp):
            if not ff[pp_br[q], K]:
                nf_any = True
                break
        if not nf_any:
            continue
        nC = aux_nbf[K]
        c0 = aux_off[K]
        lC = aux_l[K]
        nprimC = aux_nprim[c0]
        Cx = aux_coords[c0, 0]
        Cy = aux_coords[c0, 1]
        Cz = aux_coords[c0, 2]
        for ic in range(nC):
            cx_[ic] = aux_lmn[c0 + ic, 0]
            cy_[ic] = aux_lmn[c0 + ic, 1]
            cz_[ic] = aux_lmn[c0 + ic, 2]
        nroots = (lA + lB + lC + 1) // 2 + 1
        bra_order = lA + lB + 1 if need_A else lA + lB
        aux_order = lC + 1

        if mode == 0:
            for ia in range(nA):
                ibmax = ia + 1 if diag_pair else nB
                for ib in range(ibmax):
                    w0 = rc[(ia * (ia + 1)) // 2 + ib if diag_pair else ia * nB + ib]
                    for ic in range(nC):
                        Wk[ia, ib, ic] = w0 * coeff[c0 + ic]
        else:
            wkmax = 0.0
            for ia in range(nA):
                ibmax = ia + 1 if diag_pair else nB
                for ib in range(ibmax):
                    r = row_of[a0 + ia, b0 + ib]
                    if r < 0:
                        for ic in range(nC):
                            Wk[ia, ib, ic] = 0.0
                        continue
                    fac = 1.0 if a0 + ia == b0 + ib else 2.0     # (ij) and (ji)
                    if sao_fit:
                        nS = aux_nsph[K]
                        t0 = c2s_off[K]
                        s0 = sph_off[K]
                        for ic in range(nC):
                            acc = 0.0
                            for js in range(nS):
                                acc += c2s_flat[t0 + js * nC + ic] * Grows[r, s0 + js]
                            Wk[ia, ib, ic] = fac * acc
                    else:
                        for ic in range(nC):
                            Wk[ia, ib, ic] = fac * Grows[r, c0 + ic]
                    for ic in range(nC):
                        if abs(Wk[ia, ib, ic]) > wkmax:
                            wkmax = abs(Wk[ia, ib, ic])
            if QQ * wkmax < thr_grad:
                continue

        gax = 0.0
        gay = 0.0
        gaz = 0.0
        gcx = 0.0
        gcy = 0.0
        gcz = 0.0
        for q in range(npp):
            if ff[pp_br[q], K]:
                continue
            alpha = pp_alpha[q]
            beta = pp_beta[q]
            gamma_p = pp_gamma[q]
            ip = pp_ip[q]
            jp = pp_jp[q]
            ab = alpha * beta
            two_alpha = 2.0 * alpha
            pqsq = (pp_px[q] - Cx) ** 2 + (pp_py[q] - Cy) ** 2 + (pp_pz[q] - Cz) ** 2
            for kp in range(nprimC):
                gamma_q = aux_expnts[c0, kp]
                two_gq = 2.0 * gamma_q
                rho = gamma_p * gamma_q / (gamma_p + gamma_q)
                gpq_sqrt = np.sqrt(gamma_p * gamma_q)
                Roots(nroots, rho * pqsq, roots, weights)
                pref = 2.0 * np.sqrt(rho / PI)
                for ia in range(nA):
                    ca = pref * bf_coef[a0 + ia, ip]
                    ibmax = ia + 1 if diag_pair else nB
                    for ib in range(ibmax):
                        cab = ca * bf_coef[b0 + ib, jp]
                        for ic in range(nC):
                            Wp[ia, ib, ic] = cab * aux_coef[c0 + ic, kp] * Wk[ia, ib, ic]

                for ir in range(nroots):
                    t = roots[ir]
                    Recur_3c2e_new(gx, t, bra_order, 0, aux_order, 0, Ax, Bx, Cx, 0.0,
                                   alpha, beta, gamma_q, 0.0, gamma_p, gamma_q, ab, gpq_sqrt)
                    Recur_3c2e_new(gy, t, bra_order, 0, aux_order, 0, Ay, By, Cy, 0.0,
                                   alpha, beta, gamma_q, 0.0, gamma_p, gamma_q, ab, gpq_sqrt)
                    Recur_3c2e_new(gz, t, bra_order, 0, aux_order, 0, Az, Bz, Cz, 0.0,
                                   alpha, beta, gamma_q, 0.0, gamma_p, gamma_q, ab, gpq_sqrt)
                    for a in range(la_tab + 1):
                        for b in range(lB + 1):
                            for c in range(aux_order + 1):
                                Sx[a, b, c] = Shift_3c2e(gx, a, b, c, 0, xij0)
                                Sy[a, b, c] = Shift_3c2e(gy, a, b, c, 0, xij1)
                                Sz[a, b, c] = Shift_3c2e(gz, a, b, c, 0, xij2)
                    # derivative of the auxiliary Gaussian (c) and, unless one-centre, of the bra one (a)
                    for a in range(lA + 1):
                        for b in range(lB + 1):
                            for c in range(lC + 1):
                                vx = two_gq * Sx[a, b, c + 1]
                                vy = two_gq * Sy[a, b, c + 1]
                                vz = two_gq * Sz[a, b, c + 1]
                                if c > 0:
                                    vx -= c * Sx[a, b, c - 1]
                                    vy -= c * Sy[a, b, c - 1]
                                    vz -= c * Sz[a, b, c - 1]
                                dCx[a, b, c] = vx
                                dCy[a, b, c] = vy
                                dCz[a, b, c] = vz
                    if need_A:
                        for a in range(lA + 1):
                            for b in range(lB + 1):
                                for c in range(lC + 1):
                                    vx = two_alpha * Sx[a + 1, b, c]
                                    vy = two_alpha * Sy[a + 1, b, c]
                                    vz = two_alpha * Sz[a + 1, b, c]
                                    if a > 0:
                                        vx -= a * Sx[a - 1, b, c]
                                        vy -= a * Sy[a - 1, b, c]
                                        vz -= a * Sz[a - 1, b, c]
                                    dAx[a, b, c] = vx
                                    dAy[a, b, c] = vy
                                    dAz[a, b, c] = vz
                    w = weights[ir]
                    for ia in range(nA):
                        axa = ax_[ia]
                        aya = ay_[ia]
                        aza = az_[ia]
                        ibmax = ia + 1 if diag_pair else nB
                        for ib in range(ibmax):
                            bxb = bx_[ib]
                            byb = by_[ib]
                            bzb = bz_[ib]
                            if need_A:
                                for ic in range(nC):
                                    cxc = cx_[ic]
                                    cyc = cy_[ic]
                                    czc = cz_[ic]
                                    wt = w * Wp[ia, ib, ic]
                                    sx = Sx[axa, bxb, cxc]
                                    sy = Sy[aya, byb, cyc]
                                    sz = Sz[aza, bzb, czc]
                                    syz = wt * sy * sz
                                    sxz = wt * sx * sz
                                    sxy = wt * sx * sy
                                    gcx += dCx[axa, bxb, cxc] * syz
                                    gcy += dCy[aya, byb, cyc] * sxz
                                    gcz += dCz[aza, bzb, czc] * sxy
                                    gax += dAx[axa, bxb, cxc] * syz
                                    gay += dAy[aya, byb, cyc] * sxz
                                    gaz += dAz[aza, bzb, czc] * sxy
                            else:
                                for ic in range(nC):
                                    cxc = cx_[ic]
                                    cyc = cy_[ic]
                                    czc = cz_[ic]
                                    wt = w * Wp[ia, ib, ic]
                                    sx = Sx[axa, bxb, cxc]
                                    sy = Sy[aya, byb, cyc]
                                    sz = Sz[aza, bzb, czc]
                                    gcx += dCx[axa, bxb, cxc] * (wt * sy * sz)
                                    gcy += dCy[aya, byb, cyc] * (wt * sx * sz)
                                    gcz += dCz[aza, bzb, czc] * (wt * sx * sy)
        gacc[atom_a, 0] += gax
        gacc[atom_a, 1] += gay
        gacc[atom_a, 2] += gaz
        gacc[atom_c, 0] += gcx
        gacc[atom_c, 1] += gcy
        gacc[atom_c, 2] += gcz
        gacc[atom_b, 0] -= gax + gcx
        gacc[atom_b, 1] -= gay + gcy
        gacc[atom_b, 2] -= gaz + gcz


@njit(parallel=True, cache=True, fastmath=True, error_model="numpy", nogil=True, boundscheck=False)
def _grad_near_pass(bin_off, bin_items, pair_I, pair_J, dmat, coeff, cmax_aux, thr_grad,
                    sqrt_ints4c2e_diag, strict_schwarz, shell_atom, aux_atom, natoms,
                    bfs_coords, bfs_lmn, bfs_nprim, bfs_expnts, bf_coef, shell_off, shell_nbf, shell_l,
                    aux_coords, aux_lmn, aux_nprim, aux_expnts, aux_coef, aux_off, aux_nbf, aux_l,
                    Q_pair, Q_aux, threshold, pp_group, grp_branch, ff, dims):
    """Near-field 3c2e gradient over the shell pairs of the LPT bins; returns ``(natoms, 3)``."""
    nthreads = bin_off.shape[0] - 1
    maxA, maxC, lmaxA, lmaxC, maxpp = dims[0], dims[1], dims[2], dims[3], dims[4]
    gacc = np.zeros((nthreads, natoms, 3))
    rcv = np.zeros((nthreads, maxA * maxA))
    roots = np.zeros((nthreads, 12))
    weights = np.zeros((nthreads, 12))
    gx = np.zeros((nthreads, 2 * lmaxA + 2, lmaxC + 2))
    gy = np.zeros((nthreads, 2 * lmaxA + 2, lmaxC + 2))
    gz = np.zeros((nthreads, 2 * lmaxA + 2, lmaxC + 2))
    Sx = np.zeros((nthreads, lmaxA + 2, lmaxA + 1, lmaxC + 2))
    Sy = np.zeros((nthreads, lmaxA + 2, lmaxA + 1, lmaxC + 2))
    Sz = np.zeros((nthreads, lmaxA + 2, lmaxA + 1, lmaxC + 2))
    dA = np.zeros((nthreads, 3, lmaxA + 1, lmaxA + 1, lmaxC + 1))
    dC = np.zeros((nthreads, 3, lmaxA + 1, lmaxA + 1, lmaxC + 1))
    Wk = np.zeros((nthreads, maxA, maxA, maxC))
    Wp = np.zeros((nthreads, maxA, maxA, maxC))
    ppf = np.zeros((nthreads, 6, maxpp))
    ppi = np.zeros((nthreads, 3, maxpp), dtype=np.int64)
    comp = np.zeros((nthreads, 9, max(maxA, maxC)), dtype=np.int64)
    no_rows = np.zeros((1, 1))
    no_row_of = np.zeros((1, 1), dtype=np.int64)
    no_tab = np.zeros(1)
    no_idx = np.zeros(1, dtype=np.int64)
    for tid in prange(nthreads):
        for idx in range(bin_off[tid], bin_off[tid + 1]):
            p = bin_items[idx]
            I = pair_I[p]
            J = pair_J[p]
            wmax = _row_weights(I, J, dmat, sqrt_ints4c2e_diag, strict_schwarz, shell_off, shell_nbf,
                                rcv[tid])
            if wmax == 0.0:
                continue
            _pair_grad_near(I, J, rcv[tid], wmax, coeff, cmax_aux, thr_grad,
                            shell_atom[I], shell_atom[J], aux_atom, gacc[tid],
                            bfs_coords, bfs_lmn, bfs_nprim, bfs_expnts, bf_coef, shell_off, shell_nbf, shell_l,
                            aux_coords, aux_lmn, aux_nprim, aux_expnts, aux_coef, aux_off, aux_nbf, aux_l,
                            Q_pair[p], Q_aux, threshold, pp_group[p], grp_branch[p], ff,
                            roots[tid], weights[tid], gx[tid], gy[tid], gz[tid], Sx[tid], Sy[tid], Sz[tid],
                            dA[tid, 0], dA[tid, 1], dA[tid, 2], dC[tid, 0], dC[tid, 1], dC[tid, 2],
                            Wk[tid], Wp[tid],
                            ppf[tid, 0], ppf[tid, 1], ppf[tid, 2], ppf[tid, 3], ppf[tid, 4], ppf[tid, 5],
                            ppi[tid, 0], ppi[tid, 1], ppi[tid, 2],
                            comp[tid, 0], comp[tid, 1], comp[tid, 2], comp[tid, 3], comp[tid, 4],
                            comp[tid, 5], comp[tid, 6], comp[tid, 7], comp[tid, 8],
                            0, no_rows, no_row_of, False, no_tab, no_idx, no_idx, no_idx)
    grad = np.zeros((natoms, 3))
    for t in range(nthreads):
        for a in range(natoms):
            for d in range(3):
                grad[a, d] += gacc[t, a, d]
    return grad


@njit(parallel=True, cache=True, fastmath=True, error_model="numpy", nogil=True, boundscheck=False)
def _grad_near_pass_rows(bin_off, bin_items, pair_I, pair_J, Grows, row_of, sao_fit, c2s_flat, c2s_off,
                         sph_off, aux_nsph, thr_grad, shell_atom, aux_atom, natoms,
                         bfs_coords, bfs_lmn, bfs_nprim, bfs_expnts, bf_coef, shell_off, shell_nbf, shell_l,
                         aux_coords, aux_lmn, aux_nprim, aux_expnts, aux_coef, aux_off, aux_nbf, aux_l,
                         Q_pair, Q_aux, threshold, pp_group, grp_branch, ff, dims):
    """Near-field ``sum_{ij,P} Gamma^P_ij d(ij|P)/dR`` (``mode = 1`` of :func:`_pair_grad_near`); ``(natoms, 3)``."""
    nthreads = bin_off.shape[0] - 1
    maxA, maxC, lmaxA, lmaxC, maxpp = dims[0], dims[1], dims[2], dims[3], dims[4]
    gacc = np.zeros((nthreads, natoms, 3))
    rcv = np.zeros((nthreads, maxA * maxA))
    roots = np.zeros((nthreads, 12))
    weights = np.zeros((nthreads, 12))
    gx = np.zeros((nthreads, 2 * lmaxA + 2, lmaxC + 2))
    gy = np.zeros((nthreads, 2 * lmaxA + 2, lmaxC + 2))
    gz = np.zeros((nthreads, 2 * lmaxA + 2, lmaxC + 2))
    Sx = np.zeros((nthreads, lmaxA + 2, lmaxA + 1, lmaxC + 2))
    Sy = np.zeros((nthreads, lmaxA + 2, lmaxA + 1, lmaxC + 2))
    Sz = np.zeros((nthreads, lmaxA + 2, lmaxA + 1, lmaxC + 2))
    dA = np.zeros((nthreads, 3, lmaxA + 1, lmaxA + 1, lmaxC + 1))
    dC = np.zeros((nthreads, 3, lmaxA + 1, lmaxA + 1, lmaxC + 1))
    Wk = np.zeros((nthreads, maxA, maxA, maxC))
    Wp = np.zeros((nthreads, maxA, maxA, maxC))
    ppf = np.zeros((nthreads, 6, maxpp))
    ppi = np.zeros((nthreads, 3, maxpp), dtype=np.int64)
    comp = np.zeros((nthreads, 9, max(maxA, maxC)), dtype=np.int64)
    no_coeff = np.zeros(1)
    for tid in prange(nthreads):
        for idx in range(bin_off[tid], bin_off[tid + 1]):
            p = bin_items[idx]
            I = pair_I[p]
            J = pair_J[p]
            _pair_grad_near(I, J, rcv[tid], 1.0, no_coeff, no_coeff, thr_grad,
                            shell_atom[I], shell_atom[J], aux_atom, gacc[tid],
                            bfs_coords, bfs_lmn, bfs_nprim, bfs_expnts, bf_coef, shell_off, shell_nbf, shell_l,
                            aux_coords, aux_lmn, aux_nprim, aux_expnts, aux_coef, aux_off, aux_nbf, aux_l,
                            Q_pair[p], Q_aux, threshold, pp_group[p], grp_branch[p], ff,
                            roots[tid], weights[tid], gx[tid], gy[tid], gz[tid], Sx[tid], Sy[tid], Sz[tid],
                            dA[tid, 0], dA[tid, 1], dA[tid, 2], dC[tid, 0], dC[tid, 1], dC[tid, 2],
                            Wk[tid], Wp[tid],
                            ppf[tid, 0], ppf[tid, 1], ppf[tid, 2], ppf[tid, 3], ppf[tid, 4], ppf[tid, 5],
                            ppi[tid, 0], ppi[tid, 1], ppi[tid, 2],
                            comp[tid, 0], comp[tid, 1], comp[tid, 2], comp[tid, 3], comp[tid, 4],
                            comp[tid, 5], comp[tid, 6], comp[tid, 7], comp[tid, 8],
                            1, Grows, row_of, sao_fit, c2s_flat, c2s_off, sph_off, aux_nsph)
    grad = np.zeros((natoms, 3))
    for t in range(nthreads):
        for a in range(natoms):
            for d in range(3):
                grad[a, d] += gacc[t, a, d]
    return grad


# ----------------------------------------------------------------------------
# Far field
# ----------------------------------------------------------------------------
@njit(parallel=True, cache=True, fastmath=True, error_model="numpy", nogil=True, boundscheck=False)
def _grad_far_pass(bin_off, bin_items, pair_I, pair_J, pair_nrows, dmat, sqrt_ints4c2e_diag,
                   strict_schwarz, shell_atom, natoms, bfs_coords, bfs_lmn, bfs_nprim, bfs_expnts, bf_coef,
                   shell_off, shell_nbf, shell_l, ngroups, pp_group, grp_branch, grp_center,
                   branch_center, n_branches, mom_off, moments, entry_off, entry_branch, BL1, lmax, n_big,
                   tab_off, tab_LM, tab_coef, mono_off, mono_abc, mono_coef, lpair_max, maxA, lmaxA):
    """
    Density side of the far-field gradient.  For every (shell pair, branch) entry: the
    density-weighted moments of its groups about the branch centre (accumulated into the branch
    moments of the auxiliary-side pass), the derivative with respect to the bra centre ``A`` from the
    derivative distributions, and the ``B`` derivative from translational invariance with the branch
    centre moving with the distribution (``BL1`` = branch local expansions to ``lmax + 1``).
    Returns ``(grad (natoms, 3), branch_mom (n_branches, n_big))``.
    """
    nthreads = bin_off.shape[0] - 1
    L1max = lpair_max + 1
    nlm_max = (lpair_max + 1) * (lpair_max + 1)
    nlm1_max = (L1max + 1) * (L1max + 1)
    gacc = np.zeros((nthreads, natoms, 3))
    partial_mom = np.zeros((nthreads, n_branches, n_big))
    rcv = np.zeros((nthreads, maxA * maxA))
    agv = np.zeros((nthreads, nlm_max))
    emv = np.zeros((nthreads, n_big))
    Rdv = np.zeros((nthreads, n_big))
    Lgv = np.zeros((nthreads, nlm1_max))
    F3v = np.zeros((nthreads, L1max + 1, L1max + 1, L1max + 1))
    Gsv = np.zeros((nthreads, 2 * lmaxA + L1max + 3))
    Xtv = np.zeros((nthreads, 3, lmaxA + 2, lmaxA + 1, L1max + 1))
    dXv = np.zeros((nthreads, 3, lmaxA + 1, lmaxA + 1, L1max + 1))
    fev = np.zeros((nthreads, 3))
    comp = np.zeros((nthreads, 6, maxA), dtype=np.int64)
    for tid in prange(nthreads):
      for idx in range(bin_off[tid], bin_off[tid + 1]):
        p = bin_items[idx]
        if entry_off[p + 1] == entry_off[p]:
            continue
        I = pair_I[p]
        J = pair_J[p]
        rc = rcv[tid]
        if _row_weights(I, J, dmat, sqrt_ints4c2e_diag, strict_schwarz, shell_off, shell_nbf, rc) == 0.0:
            continue
        nA = shell_nbf[I]
        nB = shell_nbf[J]
        a0 = shell_off[I]
        b0 = shell_off[J]
        lA = shell_l[I]
        lB = shell_l[J]
        lp = lA + lB
        nlm = (lp + 1) * (lp + 1)
        nlm1 = (lp + 2) * (lp + 2)
        nr = pair_nrows[p]
        diag_pair = I == J
        atom_a = shell_atom[I]
        atom_b = shell_atom[J]
        need_A = atom_a != atom_b
        ax_ = comp[tid, 0]
        ay_ = comp[tid, 1]
        az_ = comp[tid, 2]
        bx_ = comp[tid, 3]
        by_ = comp[tid, 4]
        bz_ = comp[tid, 5]
        for ia in range(nA):
            ax_[ia] = bfs_lmn[a0 + ia, 0]
            ay_[ia] = bfs_lmn[a0 + ia, 1]
            az_[ia] = bfs_lmn[a0 + ia, 2]
        for ib in range(nB):
            bx_[ib] = bfs_lmn[b0 + ib, 0]
            by_[ib] = bfs_lmn[b0 + ib, 1]
            bz_[ib] = bfs_lmn[b0 + ib, 2]
        Ax = bfs_coords[a0, 0]
        Ay = bfs_coords[a0, 1]
        Az = bfs_coords[a0, 2]
        Bx = bfs_coords[b0, 0]
        By = bfs_coords[b0, 1]
        Bz = bfs_coords[b0, 2]
        ijsq = (Ax - Bx) ** 2 + (Ay - By) ** 2 + (Az - Bz) ** 2
        mbase = mom_off[p]
        ag = agv[tid]
        em = emv[tid]
        Rd = Rdv[tid]
        Lg = Lgv[tid]
        F3 = F3v[tid]
        Gs = Gsv[tid]
        Xt = Xtv[tid, 0]
        Yt = Xtv[tid, 1]
        Zt = Xtv[tid, 2]
        dX = dXv[tid, 0]
        dY = dXv[tid, 1]
        dZ = dXv[tid, 2]
        fe = fev[tid]
        for e in range(entry_off[p], entry_off[p + 1]):
            b = entry_branch[e]
            for k in range(n_big):
                em[k] = 0.0
            dix = 0.0
            diy = 0.0
            diz = 0.0
            for g in range(ngroups[p]):
                if grp_branch[p, g] != b:
                    continue
                # density-weighted moments of the group, translated to the branch centre
                for lm in range(nlm):
                    ag[lm] = 0.0
                for r in range(nr):
                    c = rc[r]
                    if c == 0.0:
                        continue
                    mo = mbase + (g * nr + r) * nlm
                    for lm in range(nlm):
                        ag[lm] += c * moments[mo + lm]
                gcx = grp_center[p, g, 0]
                gcy = grp_center[p, g, 1]
                gcz = grp_center[p, g, 2]
                mp.regular_harmonics(gcx - branch_center[b, 0], gcy - branch_center[b, 1],
                                     gcz - branch_center[b, 2], lmax, Rd)
                mp.translate_moments(ag, lp, Rd, lmax, n_big, tab_off, tab_LM, tab_coef, em)
                if not need_A:
                    continue
                # the far-field potential about the group centre as a polynomial of degree lp + 1
                mp.translate_local(BL1[b], lmax, Rd, lp + 1, n_big, tab_off, tab_LM, tab_coef, Lg)
                for nx in range(lp + 2):
                    for ny in range(lp + 2 - nx):
                        for nz in range(lp + 2 - nx - ny):
                            F3[nx, ny, nz] = 0.0
                for lm in range(nlm1):
                    v = Lg[lm]
                    if v == 0.0:
                        continue
                    for n in range(mono_off[lm], mono_off[lm + 1]):
                        F3[mono_abc[n, 0], mono_abc[n, 1], mono_abc[n, 2]] += v * mono_coef[n]
                # derivative distributions (d phi_i / dA) phi_j of the primitive pairs of the group
                q = 0
                for ip in range(bfs_nprim[a0]):
                    alpha = bfs_expnts[a0, ip]
                    for jp in range(bfs_nprim[b0]):
                        beta = bfs_expnts[b0, jp]
                        gamma_p = alpha + beta
                        arg = alpha * beta / gamma_p * ijsq
                        if arg > EXP_ARG_CUTOFF:
                            continue
                        gq = pp_group[p, q]
                        q += 1
                        if gq != g:
                            continue
                        pref = np.exp(-arg)
                        mp.gaussian_1d_moments(gamma_p, gcx - Ax, gcx - Bx, lA + 1, lB, lp + 1, Gs, Xt)
                        mp.gaussian_1d_moments(gamma_p, gcy - Ay, gcy - By, lA + 1, lB, lp + 1, Gs, Yt)
                        mp.gaussian_1d_moments(gamma_p, gcz - Az, gcz - Bz, lA + 1, lB, lp + 1, Gs, Zt)
                        two_alpha = 2.0 * alpha
                        for a in range(lA + 1):
                            for bb in range(lB + 1):
                                for n in range(lp + 2):
                                    vx = two_alpha * Xt[a + 1, bb, n]
                                    vy = two_alpha * Yt[a + 1, bb, n]
                                    vz = two_alpha * Zt[a + 1, bb, n]
                                    if a > 0:
                                        vx -= a * Xt[a - 1, bb, n]
                                        vy -= a * Yt[a - 1, bb, n]
                                        vz -= a * Zt[a - 1, bb, n]
                                    dX[a, bb, n] = vx
                                    dY[a, bb, n] = vy
                                    dZ[a, bb, n] = vz
                        for ia in range(nA):
                            ca = pref * bf_coef[a0 + ia, ip]
                            axa = ax_[ia]
                            aya = ay_[ia]
                            aza = az_[ia]
                            ibmax = ia + 1 if diag_pair else nB
                            for ib in range(ibmax):
                                r = (ia * (ia + 1)) // 2 + ib if diag_pair else ia * nB + ib
                                w = rc[r]
                                if w == 0.0:
                                    continue
                                w *= ca * bf_coef[b0 + ib, jp]
                                bxb = bx_[ib]
                                byb = by_[ib]
                                bzb = bz_[ib]
                                sx = 0.0
                                sy = 0.0
                                sz = 0.0
                                for nx in range(lp + 2):
                                    x0 = Xt[axa, bxb, nx]
                                    x1 = dX[axa, bxb, nx]
                                    for ny in range(lp + 2 - nx):
                                        y0 = Yt[aya, byb, ny]
                                        y1 = dY[aya, byb, ny]
                                        tx = x1 * y0
                                        ty = x0 * y1
                                        tz = x0 * y0
                                        for nz in range(lp + 2 - nx - ny):
                                            f = F3[nx, ny, nz]
                                            z0 = Zt[aza, bzb, nz] * f
                                            sx += tx * z0
                                            sy += ty * z0
                                            sz += tz * dZ[aza, bzb, nz] * f
                                dix += w * sx
                                diy += w * sy
                                diz += w * sz
            # the entry's derivative when its distribution and the branch centre move together
            _first_order_dot(em, lmax, BL1[b], tab_off, tab_LM, tab_coef, n_big, fe)
            pm = partial_mom[tid, b]
            for k in range(n_big):
                pm[k] += em[k]
            gacc[tid, atom_a, 0] += dix
            gacc[tid, atom_a, 1] += diy
            gacc[tid, atom_a, 2] += diz
            gacc[tid, atom_b, 0] += fe[0] - dix
            gacc[tid, atom_b, 1] += fe[1] - diy
            gacc[tid, atom_b, 2] += fe[2] - diz
    grad = np.zeros((natoms, 3))
    for t in range(nthreads):
        for a in range(natoms):
            for d in range(3):
                grad[a, d] += gacc[t, a, d]
    branch_mom = np.zeros((n_branches, n_big))
    for t in range(nthreads):
        for b in range(n_branches):
            for k in range(n_big):
                branch_mom[b, k] += partial_mom[t, b, k]
    return grad, branch_mom


@njit(cache=True, fastmath=True, error_model="numpy", nogil=True, boundscheck=False)
def _aux_far_grad(shell_mom, L_K1, aux_l, aux_atom, natoms, tab_off, tab_LM, tab_coef, n_big):
    """Auxiliary side of the far-field gradient: every shell moved rigidly in its local field."""
    grad = np.zeros((natoms, 3))
    out = np.zeros(3)
    for K in range(aux_l.shape[0]):
        _first_order_dot(shell_mom[K], aux_l[K], L_K1[K], tab_off, tab_LM, tab_coef, n_big, out)
        for d in range(3):
            grad[aux_atom[K], d] += out[d]
    return grad


# ----------------------------------------------------------------------------
# Python drivers
# ----------------------------------------------------------------------------
def build_grad_plan(basis, auxbasis, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold, strict_schwarz,
                    sao=False, options=None, far_field=True, ncores=None):
    """
    Algorithm-12 plan metadata for the gradient: the screening, branches, near/far classification
    and group moments of :func:`~pyfock.Integrals.df_algo12_helpers._plan_metadata` with the
    arguments the SCF used, but no near-field values (``max_memory_gb=0``, every block is evaluated
    directly) and no branch-centred row moments (``low_memory=True``, every group is translated
    once).  ``far_field=False`` gives algorithm 11: every group stays in the near field.  The plan
    also serves :func:`~pyfock.Integrals.df_algo12_helpers.gamma_from_plan`.
    """
    opts = dict(options or {})
    opts['low_memory'] = True
    if not far_field:
        opts['break_even'] = np.inf      # no group is profitable: the far field is empty
    plan = a12._plan_metadata(basis, auxbasis, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold,
                              strict_schwarz, sao, max_memory_gb=0, ncores=ncores, options=opts)
    # the derivative raises the total angular momentum by one (the same limit as rys_3c2e_grad_contract)
    if (2 * int(plan.shell_l.max()) + int(plan.aux_l.max()) + 1) // 2 + 1 > 10:
        raise NotImplementedError('3c2e gradients support Rys orders up to 10 only '
                                  '(2 l_orbital + l_auxiliary <= 18).')
    plan.values = np.zeros(0, dtype=np.float64)
    plan.far_field = bool(far_field)
    # The gradient contracts with Cartesian coefficients, so its far field needs the moments of the
    # Cartesian auxiliary functions; a SAO plan holds those of the projected ones (the two agree for
    # coefficients c = T^T c_sph, but not for a general vector).
    plan.aux_mom_cart = plan.aux_mom
    if sao:
        plan.aux_mom_cart = a12._aux_moments(auxbasis, a12.pack_basis_arrays(auxbasis), plan.aux_coef,
                                             plan.aux_off, plan.aux_nbf, plan.aux_l, False, plan.projectors)
    plan.shell_atom = np.asarray(basis.bfs_atoms, dtype=np.int64)[plan.shell_off]
    plan.aux_atom_mol = np.asarray(auxbasis.bfs_atoms, dtype=np.int64)[plan.aux_off]
    plan.natoms_mol = int(max(plan.shell_atom.max(), plan.aux_atom_mol.max())) + 1
    # the derivatives need one order more than the energy on both sides of the expansions
    plan.grad_table = a12._tables(max(plan.lpair_max, plan.l_aux_max) + 1, plan.lmax)
    plan.grad_table_big = a12._tables(plan.l_aux_max, plan.lmax + 1)
    plan.grad_mono = a12._polynomials(plan.lpair_max + 1)
    plan.sign_big1 = np.array([(-1.0) ** j for j in range(plan.lmax + 2) for _ in range(2 * j + 1)],
                              dtype=np.float64)
    return plan


def _near_args(plan):
    return (plan.bfs_coords, plan.bfs_lmn, plan.bfs_nprim, plan.bfs_expnts, plan.bf_coef,
            plan.shell_off, plan.shell_nbf, plan.shell_l,
            plan.aux_coords, plan.aux_lmn, plan.aux_nprim, plan.aux_expnts, plan.aux_coef,
            plan.aux_off, plan.aux_nbf, plan.aux_l,
            plan.Q_pair, plan.Q_aux, plan.threshold)


def grad_contract_rows(plan, Grows, row_of, fit_tables=None, threshold_grad=1e-11):
    """
    ``grad[A, d] = sum_{ij, P} Gamma^P_ij d(ij|P)/dR_{A,d}`` over both triangles, for a weight that is
    symmetric in ``(i, j)`` and stored per function pair: ``Gamma^P_ij`` is row ``row_of[i, j]``
    (``i >= j``, ``-1`` for pairs left out) of ``Grows``.  This is the three-center part of the RI
    exchange gradient (:func:`~pyfock.Integrals.df_algo11_exchange.gradient_rows`).  ``fit_tables`` =
    ``(c2s_flat, c2s_off, sph_off, aux_nsph)`` when the columns are spherical fit functions (SAO),
    ``None`` when they are the Cartesian auxiliary functions.  The plan must be built with
    ``far_field=False`` (algorithm 11), since the exchange contracts the integrals themselves.

    Returns the ``(natoms, 3)`` gradient term.
    """
    if plan.far_field and plan.n_entries:
        raise ValueError('grad_contract_rows needs a plan without far field (far_field=False).')
    Grows = np.ascontiguousarray(Grows, dtype=np.float64)
    row_of = np.ascontiguousarray(row_of, dtype=np.int64)
    if fit_tables is None:
        sao_fit = False
        c2s_flat = np.zeros(1)
        c2s_off = sph_off = aux_nsph = np.zeros(1, dtype=np.int64)
    else:
        sao_fit = True
        c2s_flat, c2s_off, sph_off, aux_nsph = (np.ascontiguousarray(t) for t in fit_tables)
    nthreads = int(numba.get_num_threads())
    work = plan.sig
    cost = plan.cost_nf_pair[work] + 1.0
    bin_off, bin_items = _lpt_bins(work, cost, nthreads)
    return _grad_near_pass_rows(bin_off, bin_items, plan.pair_I, plan.pair_J, Grows, row_of, sao_fit,
                                c2s_flat, c2s_off, sph_off, aux_nsph, float(threshold_grad),
                                plan.shell_atom, plan.aux_atom_mol, plan.natoms_mol, *_near_args(plan),
                                plan.pp_group, plan.grp_branch_eff, plan.ff_eff, plan.dims)


def grad_contract(plan, dmat, df_coeff, threshold_grad=1e-11, timings=None, near=True, far=True):
    """
    ``grad[A, d] = sum_ijP D_ij c_P d(ij|P)/dR_{A,d}`` with ``(ij|P)`` as algorithm 12 evaluates it
    (near-field Rys integrals plus far-field multipoles; algorithm 11 for a plan built with
    ``far_field=False``).  ``df_coeff`` are the *Cartesian* fitting coefficients the gradient
    contracts with (in SAO mode ``c_eff = T^T c_sph``), for which the plan keeps the moments of the
    Cartesian auxiliary functions.  ``near`` and ``far`` select the two parts (the GPU driver
    evaluates the near field on the device).

    Returns the ``(natoms, 3)`` gradient term.
    """
    dmat = np.ascontiguousarray(dmat, dtype=np.float64)
    coeff = np.ascontiguousarray(df_coeff, dtype=np.float64)
    nthreads = int(numba.get_num_threads())
    natoms = plan.natoms_mol
    grad = np.zeros((natoms, 3))

    # near field: the near-field Rys cost of every significant pair balances the bins
    t0 = timer()
    if near:
        cmax_aux = np.array([np.abs(coeff[k0:k0 + nk]).max()
                             for k0, nk in zip(plan.aux_off, plan.aux_nbf)])
        work = plan.sig
        cost = plan.cost_nf_pair[work] + 1.0
        bin_off, bin_items = _lpt_bins(work, cost, nthreads)
        grad += _grad_near_pass(bin_off, bin_items, plan.pair_I, plan.pair_J, dmat, coeff, cmax_aux,
                                float(threshold_grad), plan.sqrt_ints4c2e_diag, plan.strict_schwarz,
                                plan.shell_atom, plan.aux_atom_mol, natoms, *_near_args(plan),
                                plan.pp_group, plan.grp_branch_eff, plan.ff_eff, plan.dims)
    t1 = timer()
    if timings is not None:
        timings['near_field'] = timings.get('near_field', 0.0) + (t1 - t0)
    if not far or not plan.far_field or plan.n_entries == 0:
        return grad

    # far field, density side: needs the local expansions of the far-field fitted density at the
    # branch centres, one order above lmax
    pair_off_tab, ent_LM, ent_coef = plan.grad_table
    n_small = (plan.l_aux_max + 1) ** 2
    shell_mom = a12._shell_moments(coeff, plan.aux_mom_cart, plan.aux_off, plan.aux_nbf, plan.aux_l, n_small)
    big_off, big_LM, big_coef = plan.grad_table_big
    BL1 = a12._atoms_to_branches(shell_mom, plan.branch_center, plan.atom_coords, plan.atom_shell_off,
                                 plan.atom_shells, plan.atom_lmax, plan.aux_l, plan.ff, plan.any_ff,
                                 plan.lmax + 1, (plan.lmax + 2) ** 2, plan.l_aux_max,
                                 big_off, big_LM, big_coef, plan.sign_big1)
    BL1 = np.ascontiguousarray(BL1)
    entries = plan.entry_off[1:] - plan.entry_off[:-1]
    far_work = np.nonzero(entries > 0)[0].astype(np.int64)
    lpair = plan.shell_l[plan.pair_I[far_work]] + plan.shell_l[plan.pair_J[far_work]]
    far_cost = (plan.ngroups[far_work] * plan.pair_nrows[far_work] * (lpair + 2) ** 3
                + entries[far_work] * plan.n_big * 4).astype(np.float64)
    bin_off, bin_items = _lpt_bins(far_work, far_cost, nthreads)
    mono_off, mono_abc, mono_coef = plan.grad_mono
    grad_far, branch_mom = _grad_far_pass(
        bin_off, bin_items, plan.pair_I, plan.pair_J, plan.pair_nrows, dmat, plan.sqrt_ints4c2e_diag,
        plan.strict_schwarz, plan.shell_atom, natoms, plan.bfs_coords, plan.bfs_lmn, plan.bfs_nprim,
        plan.bfs_expnts, plan.bf_coef, plan.shell_off, plan.shell_nbf, plan.shell_l, plan.ngroups,
        plan.pp_group, plan.grp_branch_eff, plan.grp_center, plan.branch_center, plan.n_branches,
        plan.mom_off, plan.moments, plan.entry_off, plan.entry_branch, BL1, plan.lmax, plan.n_big,
        pair_off_tab, ent_LM, ent_coef, mono_off, mono_abc, mono_coef, plan.lpair_max,
        int(plan.dims[0]), int(plan.dims[2]))
    t2 = timer()

    # far field, auxiliary side: local expansions at the atoms one order above every shell
    L_K1 = a12._branches_to_atoms(branch_mom, plan.branch_center, plan.atom_coords, plan.atom_shell_off,
                                  plan.atom_shells, plan.atom_lmax + 1, plan.aux_l + 1, plan.ff,
                                  plan.any_ff, plan.lmax, plan.n_big, plan.l_aux_max + 1,
                                  pair_off_tab, ent_LM, ent_coef, plan.sign_big)
    grad_aux = _aux_far_grad(shell_mom, np.ascontiguousarray(L_K1), plan.aux_l, plan.aux_atom_mol, natoms,
                             pair_off_tab, ent_LM, ent_coef, plan.n_big)
    if timings is not None:
        timings['far_field'] = timings.get('far_field', 0.0) + (timer() - t1)
        timings['far_field_density'] = timings.get('far_field_density', 0.0) + (t2 - t1)
    return grad + grad_far + grad_aux
