"""
Density-fitting Coulomb term, algorithm 10 (``DF_algo=10``; the former default, still used on the GPU).

This module collects every routine that the default density-fitted Coulomb
evaluation needs, in the order in which they are used during an SCF run:

1. :func:`calc_offsets_3c2e_schwarz`  - Schwarz screening; sizes the sparse storage.
2. :func:`rys_3c2e_tri_schwarz_sparse_algo10` - evaluates the screened
   three-center integrals ``(ij|P)`` with Rys quadrature (CAO or SAO aux basis).
3. :func:`df_coeff_calculator_algo10` - every SCF cycle: ``gamma_P = sum_ij D_ij (ij|P)``.
4. :func:`J_tri_calculator_algo10` - every SCF cycle: ``J_ij = sum_P (ij|P) c_P``.

The ``(ij|ij)`` Schwarz diagonal itself comes from
:func:`pyfock.Integrals.schwarz_helpers.eri_4c2e_diag`, which is shared with the
four-center integral code.  The GPU counterparts live in
:mod:`pyfock.Integrals.df_algo10_helpers_cupy` and use the same storage layout,
so ``offsets`` computed here can be handed straight to the CuPy routines.

Sparse storage layout
---------------------
Only the lower triangle of the AO pair index is stored, ``ij`` running over
``indicesA, indicesB = np.tril_indices(nao)`` (row-major, so the position of the
pair ``(i, j)`` is ``i*(i+1)//2 + j``).  For each pair the significant auxiliary
functions are stored contiguously, in increasing ``P``, starting at
``offsets[ij]``; ``offsets`` has ``npairs + 1`` entries and ``offsets[-1]`` is
the total number of stored values.  No auxiliary index array is stored: every
routine that reads the values re-derives which ``P`` are present by replaying the
screening test below, which is why all four steps must use *exactly* the same
rule.

Screening rules (identical in every routine of this module and its GPU twin)
----------------------------------------------------------------------------
* Pair level ("strict Schwarz", ``strict_schwarz=True``): the pair ``ij`` is
  dropped completely when ``(ij|ij) < STRICT_PAIR_CUTOFF``.  The same pair
  criterion is applied to the nuclear attraction matrix elsewhere, which is what
  makes this aggressive cut-off usable.
* Function level: the value ``(ij|P)`` is stored when
  ``sqrt((ij|ij)) * sqrt((P|P)) > threshold``.

Spherical auxiliary functions (``sao=True``)
--------------------------------------------
The auxiliary basis stays Cartesian in memory, but every d, f, g, ... shell is
projected onto its spherical-harmonic subspace, ``(ij|P~) = sum_Q Proj_PQ (ij|Q)``
with ``Proj = pinv(C) C`` built from :meth:`pyfock.Basis.Basis.cart2sph`.  This
is the same "pseudo-Cartesian" representation that is used for the two-center
metric ``Proj V Proj^T + eps*1``.  Two consequences:

* the projection mixes the Cartesian components of a shell, so the integral
  kernel evaluates *all* components of a shell whenever the shell is
  significant and projects them together;
* the projected metric is singular on the complement of the spherical subspace
  and only regularized by ``eps = 1e-12``; any stored pair vector that is not a
  complete projected shell block would leave a component in that null space and
  be amplified by ``1/eps`` in the fitting solve.

Therefore, with ``sao=True`` the Schwarz decision is made per *shell*: the
caller passes shell-constant bounds from :func:`aux_shell_max_bounds` (every
function carries the maximum ``sqrt((P|P))`` of its shell) to all four routines,
so the function-level rule above stores or skips whole shells, and the storage
layout, offsets, gamma and J stay exactly as in the Cartesian case.  The
integral driver refuses per-function bounds in SAO mode.  (Earlier versions
decided per shell inside the kernel while sizing the storage per function,
which mis-aligned the storage whenever the components of one shell straddled
the threshold, and never screened s shells at all.)
"""

import numpy as np
import numba
from numba import njit, prange

from .rys_helpers import coulomb_rys_3c2e

__all__ = [
    'STRICT_PAIR_CUTOFF',
    'EXP_ARG_CUTOFF',
    'pack_basis_arrays',
    'aux_shell_arrays',
    'sao_aux_projectors',
    'aux_shell_max_bounds',
    'check_sao_bounds_are_shell_constant',
    'calc_offsets_3c2e_schwarz',
    'rys_3c2e_tri_schwarz_sparse_algo10',
    'df_coeff_calculator_algo10',
    'J_tri_calculator_algo10',
]

#: A pair ``ij`` is discarded by the strict (pair-level) Schwarz screening when
#: ``(ij|ij)`` is below this value.  Must match the value used for the nuclear
#: attraction matrix in ``nuc_mat_symm``.
STRICT_PAIR_CUTOFF = 1e-13

#: A primitive pair (or a whole contracted pair, judged by its most diffuse
#: primitives) is skipped when ``alpha*beta/(alpha+beta) * |A-B|^2`` exceeds this
#: value, i.e. when the Gaussian product prefactor ``exp(-...)`` is below 1e-8.
EXP_ARG_CUTOFF = 18.42

#: Largest number of Rys roots supported by the quadrature helpers.
MAX_RYS_ROOTS = 10


# ----------------------------------------------------------------------------
# Basis-set packing helpers
# ----------------------------------------------------------------------------
def pack_basis_arrays(basis):
    """
    Pack the per-basis-function data of a :class:`~pyfock.Basis.Basis` into
    contiguous NumPy arrays that Numba/CUDA kernels can consume.

    The ragged per-function primitive lists are padded to the largest
    contraction length (unused entries are zero and are never visited because
    the kernels loop up to ``bfs_nprim[i]``).

    Returns
    -------
    bfs_coords : (nbf, 3) float64
    bfs_contr_prim_norms : (nbf,) float64
    bfs_lmn : (nbf, 3) int64
    bfs_nprim : (nbf,) int64
    bfs_coeffs, bfs_prim_norms, bfs_expnts : (nbf, maxnprim) float64
    """
    nbf = basis.bfs_nao
    maxnprim = max(basis.bfs_nprim)
    bfs_coeffs = np.zeros((nbf, maxnprim), dtype=np.float64)
    bfs_prim_norms = np.zeros((nbf, maxnprim), dtype=np.float64)
    bfs_expnts = np.zeros((nbf, maxnprim), dtype=np.float64)
    for ibf in range(nbf):
        nprim = basis.bfs_nprim[ibf]
        bfs_coeffs[ibf, :nprim] = basis.bfs_coeffs[ibf]
        bfs_prim_norms[ibf, :nprim] = basis.bfs_prim_norms[ibf]
        bfs_expnts[ibf, :nprim] = basis.bfs_expnts[ibf]
    return (
        np.ascontiguousarray(basis.bfs_coords, dtype=np.float64),
        np.ascontiguousarray(basis.bfs_contr_prim_norms, dtype=np.float64),
        np.ascontiguousarray(basis.bfs_lmn, dtype=np.int64),
        np.ascontiguousarray(basis.bfs_nprim, dtype=np.int64),
        bfs_coeffs,
        bfs_prim_norms,
        bfs_expnts,
    )


def aux_shell_arrays(auxbasis):
    """
    Shell structure of the auxiliary basis as three int64 arrays of length
    ``nshells``: index of the first function of each shell, number of
    (Cartesian) functions in the shell, and angular momentum of the shell.
    """
    shell_bfs_offset = np.ascontiguousarray(auxbasis.shell_bfs_offset, dtype=np.int64)
    shell_nbf = np.ascontiguousarray(auxbasis.bfs_nbfshell, dtype=np.int64)
    shell_l = np.asarray(auxbasis.shells, dtype=np.int64) - 1
    return shell_bfs_offset, shell_nbf, shell_l


def sao_aux_projectors(max_l):
    """
    Stacked Cartesian -> spherical-subspace projectors for ``l = 0 .. max_l``.

    ``proj[l, :ncart(l), :ncart(l)]`` is ``pinv(C_l) @ C_l`` where ``C_l`` is the
    Cartesian-to-real-spherical transformation of :meth:`Basis.cart2sph`; it is
    the identity for s and p shells and projects out the ``(l-2)``-like
    contaminants (e.g. ``x^2+y^2+z^2``) of higher shells.  Entries outside the
    ``ncart(l)`` block are zero.
    """
    from ..Basis import Basis  # local import: Basis imports Numba kernels of its own

    ncart_max = (max_l + 1) * (max_l + 2) // 2
    proj = np.zeros((max_l + 1, ncart_max, ncart_max), dtype=np.float64)
    for l in range(max_l + 1):
        ncart = (l + 1) * (l + 2) // 2
        if l < 2:
            proj[l, :ncart, :ncart] = np.eye(ncart)
        else:
            c2s = np.asarray(Basis.cart2sph(l), dtype=np.float64)
            proj[l, :ncart, :ncart] = np.linalg.pinv(c2s) @ c2s
    return proj


def aux_shell_max_bounds(sqrt_diag_ints2c2e, auxbasis):
    """
    Shell-constant Schwarz bounds for spherical auxiliary functions.

    Returns a copy of ``sqrt_diag_ints2c2e`` in which every function carries the
    largest ``sqrt((P|P))`` of its shell.  Feeding this array (instead of the
    per-function values) to all DF_algo=10 routines makes the function-level
    Schwarz test decide identically for all components of a shell, so a d/f/g
    shell is skipped or stored as a whole.  This is required with ``sao=True``:
    the stored ``(ij|P~)`` of a shell are the projection of the Cartesian block
    onto the spherical subspace, and only a complete block stays inside that
    subspace.  Partial blocks would leave a component in the null space of the
    projected two-center metric, which is regularized with ``eps = 1e-12`` and
    would amplify it by ``1/eps`` in the fitting solve.  The bound stays valid
    (it can only grow), so the screening remains rigorous.
    """
    bounds = np.array(sqrt_diag_ints2c2e, dtype=np.float64, copy=True)
    shell_bfs_offset, shell_nbf, _ = aux_shell_arrays(auxbasis)
    for k0, nk in zip(shell_bfs_offset, shell_nbf):
        bounds[k0:k0 + nk] = bounds[k0:k0 + nk].max()
    return bounds


def check_sao_bounds_are_shell_constant(sqrt_diag_ints2c2e, aux_shell_bfs_offset, aux_shell_nbf):
    """Raise ``ValueError`` unless the bounds are constant within every aux shell (see :func:`aux_shell_max_bounds`)."""
    sqrt_diag_ints2c2e = np.asarray(sqrt_diag_ints2c2e)
    for k0, nk in zip(aux_shell_bfs_offset, aux_shell_nbf):
        block = sqrt_diag_ints2c2e[k0:k0 + nk]
        if np.any(block != block[0]):
            raise ValueError(
                'sao=True needs shell-constant Schwarz bounds for the auxiliary basis: pass '
                'aux_shell_max_bounds(sqrt_diag_ints2c2e, auxbasis) to every DF_algo=10 routine '
                '(offsets, integrals, gamma and J) so that whole spherical shells are stored.')


# ----------------------------------------------------------------------------
# 1. Schwarz screening: number of significant aux functions per AO pair
# ----------------------------------------------------------------------------
def calc_offsets_3c2e_schwarz(sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold,
                              strict_schwarz, indicesA, indicesB):
    """
    Size the sparse ``(ij|P)`` storage for the AO pairs ``(indicesA, indicesB)``.

    Parameters
    ----------
    sqrt_ints4c2e_diag : (nao, nao) ndarray
        ``sqrt(|(ij|ij)|)``.
    sqrt_diag_ints2c2e : (naux,) ndarray
        ``sqrt(|(P|P)|)`` (of the pseudo-Cartesian metric when ``sao`` is used).
    threshold : float
        Function-level Schwarz threshold.
    strict_schwarz : bool
        Also drop whole pairs with ``(ij|ij) < STRICT_PAIR_CUTOFF``.
    indicesA, indicesB : (npairs,) int ndarrays
        The lower-triangular pair list, normally ``np.tril_indices(nao)``.

    Returns
    -------
    offsets : (npairs + 1,) int64 ndarray
        ``offsets[ij]`` is the position of the first stored value of pair ``ij``.
    nsignificant : int
        Total number of stored values (``offsets[-1]``).
    """
    offsets = _count_significant_aux_per_pair(
        np.asarray(sqrt_ints4c2e_diag), np.asarray(sqrt_diag_ints2c2e),
        float(threshold), bool(strict_schwarz),
        np.asarray(indicesA), np.asarray(indicesB))
    np.cumsum(offsets, out=offsets)
    return offsets, int(offsets[-1])


@njit(parallel=True, cache=True, fastmath=True, error_model="numpy", nogil=True)
def _count_significant_aux_per_pair(sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold,
                                    strict_schwarz, indicesA, indicesB):
    npairs = indicesA.shape[0]
    naux = sqrt_diag_ints2c2e.shape[0]
    counts = np.zeros(npairs + 1, dtype=np.int64)  # counts[ij + 1] = number stored for pair ij
    for ij in prange(npairs):
        i = indicesA[ij]
        j = indicesB[ij]
        sqrt_ij = sqrt_ints4c2e_diag[i, j]
        if strict_schwarz and sqrt_ij * sqrt_ij < STRICT_PAIR_CUTOFF:
            continue
        count = 0
        for k in range(naux):
            if sqrt_ij * sqrt_diag_ints2c2e[k] > threshold:
                count += 1
        counts[ij + 1] = count
    return counts


# ----------------------------------------------------------------------------
# 2. Screened three-center integrals in sparse triangular storage
# ----------------------------------------------------------------------------
def rys_3c2e_tri_schwarz_sparse_algo10(basis, auxbasis, indicesA, indicesB, offsets,
                                       sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold,
                                       strict_schwarz, nsignificant, sao=False):
    """
    Evaluate the Schwarz-screened three-center integrals ``(ij|P)`` with Rys
    quadrature and return them as a flat array in the sparse layout described in
    the module docstring.

    Parameters
    ----------
    basis, auxbasis : Basis
        Orbital and auxiliary basis (both Cartesian internally).
    indicesA, indicesB, offsets, nsignificant :
        Pair list and storage layout from :func:`calc_offsets_3c2e_schwarz`.
    sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold, strict_schwarz :
        The screening data; must be the same objects that produced ``offsets``.
    sao : bool, optional
        Project every auxiliary shell with ``l >= 2`` onto its spherical
        subspace (see module docstring).  The orbital basis is unaffected.

    Returns
    -------
    ints3c2e : (nsignificant,) float64 ndarray
    """
    (bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim,
     bfs_coeffs, bfs_prim_norms, bfs_expnts) = pack_basis_arrays(basis)
    (aux_bfs_coords, aux_bfs_contr_prim_norms, aux_bfs_lmn, aux_bfs_nprim,
     aux_bfs_coeffs, aux_bfs_prim_norms, aux_bfs_expnts) = pack_basis_arrays(auxbasis)
    aux_shell_bfs_offset, aux_shell_nbf, aux_shell_l = aux_shell_arrays(auxbasis)

    sqrt_diag_ints2c2e = np.asarray(sqrt_diag_ints2c2e, dtype=np.float64)
    # Largest sqrt((P|P)) within each aux shell: a shell can be skipped as a
    # whole exactly when none of its functions passes the function-level test.
    aux_shell_sqrt_max = np.array(
        [sqrt_diag_ints2c2e[k0:k0 + nk].max()
         for k0, nk in zip(aux_shell_bfs_offset, aux_shell_nbf)], dtype=np.float64)

    max_l_aux = int(aux_shell_l.max())
    if sao:
        # Whole spherical shells must be stored (see aux_shell_max_bounds).
        check_sao_bounds_are_shell_constant(sqrt_diag_ints2c2e, aux_shell_bfs_offset, aux_shell_nbf)
    projectors = sao_aux_projectors(max_l_aux) if sao else sao_aux_projectors(1)

    return rys_3c2e_tri_schwarz_sparse_algo10_internal(
        bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim,
        bfs_coeffs, bfs_prim_norms, bfs_expnts,
        aux_bfs_coords, aux_bfs_contr_prim_norms, aux_bfs_lmn, aux_bfs_nprim,
        aux_bfs_coeffs, aux_bfs_prim_norms, aux_bfs_expnts,
        aux_shell_bfs_offset, aux_shell_nbf, aux_shell_l, aux_shell_sqrt_max,
        bool(sao), projectors,
        np.asarray(indicesA), np.asarray(indicesB), np.asarray(offsets),
        np.asarray(sqrt_ints4c2e_diag, dtype=np.float64), sqrt_diag_ints2c2e,
        float(threshold), bool(strict_schwarz), int(nsignificant))


@njit(cache=True, fastmath=True, error_model="numpy", nogil=True, inline='always')
def _int3c2e_single(i, j, k, la, ma, na, lb, mb, nb, I, J, IJsq, nprimi, nprimj, tempcoeff1,
                    bfs_coeffs, bfs_prim_norms, bfs_expnts,
                    aux_bfs_coords, aux_bfs_contr_prim_norms, aux_bfs_lmn, aux_bfs_nprim,
                    aux_bfs_coeffs, aux_bfs_prim_norms, aux_bfs_expnts,
                    roots, weights, G, P, L):
    """One contracted Cartesian integral ``(ij|k)`` by Rys quadrature."""
    lmnk = aux_bfs_lmn[k]
    lc = lmnk[0]
    mc = lmnk[1]
    nc = lmnk[2]
    K = aux_bfs_coords[k]
    nprimk = aux_bfs_nprim[k]
    tempcoeff2 = tempcoeff1 * aux_bfs_contr_prim_norms[k]
    norder = (la + ma + na + lb + mb + nb + lc + mc + nc) // 2 + 1
    val = 0.0
    if norder > MAX_RYS_ROOTS:  # beyond the tabulated Rys roots (needs L_total >= 20)
        return val
    n = max(la + lb, ma + mb, na + nb)
    m = max(lc, mc, nc)
    for ik in range(nprimi):
        alphaik = bfs_expnts[i, ik]
        tempcoeff3 = tempcoeff2 * bfs_coeffs[i, ik] * bfs_prim_norms[i, ik]
        for jk in range(nprimj):
            alphajk = bfs_expnts[j, jk]
            gammaP = alphaik + alphajk
            if alphaik * alphajk / gammaP * IJsq > EXP_ARG_CUTOFF:
                continue
            P[0] = (alphaik * I[0] + alphajk * J[0]) / gammaP
            P[1] = (alphaik * I[1] + alphajk * J[1]) / gammaP
            P[2] = (alphaik * I[2] + alphajk * J[2]) / gammaP
            PQsq = (P[0] - K[0]) ** 2 + (P[1] - K[1]) ** 2 + (P[2] - K[2]) ** 2
            tempcoeff4 = tempcoeff3 * bfs_coeffs[j, jk] * bfs_prim_norms[j, jk]
            for kk in range(nprimk):
                alphakk = aux_bfs_expnts[k, kk]
                rho = gammaP * alphakk / (gammaP + alphakk)
                tempcoeff5 = tempcoeff4 * aux_bfs_coeffs[k, kk] * aux_bfs_prim_norms[k, kk]
                val += tempcoeff5 * coulomb_rys_3c2e(
                    roots, weights, G, PQsq, rho, norder, n, m,
                    la, lb, lc, 0, ma, mb, mc, 0, na, nb, nc, 0,
                    alphaik, alphajk, alphakk, 0.0, I, J, K, L, P)
    return val


@njit(parallel=True, cache=True, fastmath=True, error_model="numpy", nogil=True, boundscheck=False)
def rys_3c2e_tri_schwarz_sparse_algo10_internal(
        bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim,
        bfs_coeffs, bfs_prim_norms, bfs_expnts,
        aux_bfs_coords, aux_bfs_contr_prim_norms, aux_bfs_lmn, aux_bfs_nprim,
        aux_bfs_coeffs, aux_bfs_prim_norms, aux_bfs_expnts,
        aux_shell_bfs_offset, aux_shell_nbf, aux_shell_l, aux_shell_sqrt_max,
        sao, projectors,
        indicesA, indicesB, offsets, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e,
        threshold, strict_schwarz, nsignificant):
    """Numba kernel behind :func:`rys_3c2e_tri_schwarz_sparse_algo10`."""
    threeC2E = np.zeros(nsignificant, dtype=np.float64)
    nshells_aux = aux_shell_bfs_offset.shape[0]
    ncart_max = projectors.shape[1]
    L = np.zeros(3, dtype=np.float64)  # dummy fourth center (read-only)

    for itemp in prange(indicesA.shape[0]):
        i = indicesA[itemp]
        j = indicesB[itemp]
        sqrt_ij = sqrt_ints4c2e_diag[i, j]
        if strict_schwarz and sqrt_ij * sqrt_ij < STRICT_PAIR_CUTOFF:
            continue

        I = bfs_coords[i]
        J = bfs_coords[j]
        lmni = bfs_lmn[i]
        la = lmni[0]
        ma = lmni[1]
        na = lmni[2]
        lmnj = bfs_lmn[j]
        lb = lmnj[0]
        mb = lmnj[1]
        nb = lmnj[2]
        nprimi = bfs_nprim[i]
        nprimj = bfs_nprim[j]
        IJsq = (I[0] - J[0]) ** 2 + (I[1] - J[1]) ** 2 + (I[2] - J[2]) ** 2
        tempcoeff1 = bfs_contr_prim_norms[i] * bfs_contr_prim_norms[j]

        # Whole-pair pre-screening on the most diffuse primitives: if even the
        # broadest Gaussian product is negligible, nothing of this pair survives.
        # (Storage for the pair stays allocated and zero-filled.)
        alpha_i_min = bfs_expnts[i, 0]
        for ik in range(1, nprimi):
            if bfs_expnts[i, ik] < alpha_i_min:
                alpha_i_min = bfs_expnts[i, ik]
        alpha_j_min = bfs_expnts[j, 0]
        for jk in range(1, nprimj):
            if bfs_expnts[j, jk] < alpha_j_min:
                alpha_j_min = bfs_expnts[j, jk]
        if alpha_i_min * alpha_j_min / (alpha_i_min + alpha_j_min) * IJsq > EXP_ARG_CUTOFF:
            continue

        # Thread-private work buffers, allocated once per pair.
        roots = np.zeros(MAX_RYS_ROOTS, dtype=np.float64)
        weights = np.zeros(MAX_RYS_ROOTS, dtype=np.float64)
        G = np.zeros((20, 20), dtype=np.float64)
        P = np.zeros(3, dtype=np.float64)
        cart_buf = np.zeros(ncart_max, dtype=np.float64)
        sph_buf = np.zeros(ncart_max, dtype=np.float64)

        base = offsets[itemp]
        index_k = 0
        for s in range(nshells_aux):
            # Skip the shell only if none of its functions is significant;
            # this reproduces the function-level count of calc_offsets exactly.
            if sqrt_ij * aux_shell_sqrt_max[s] <= threshold:
                continue
            k0 = aux_shell_bfs_offset[s]
            nk = aux_shell_nbf[s]
            l = aux_shell_l[s]

            if sao and l >= 2:
                # The spherical projection couples all Cartesian components of
                # the shell: evaluate all of them, project, store the significant ones.
                for c in range(nk):
                    cart_buf[c] = _int3c2e_single(
                        i, j, k0 + c, la, ma, na, lb, mb, nb, I, J, IJsq, nprimi, nprimj, tempcoeff1,
                        bfs_coeffs, bfs_prim_norms, bfs_expnts,
                        aux_bfs_coords, aux_bfs_contr_prim_norms, aux_bfs_lmn, aux_bfs_nprim,
                        aux_bfs_coeffs, aux_bfs_prim_norms, aux_bfs_expnts,
                        roots, weights, G, P, L)
                for r in range(nk):
                    acc = 0.0
                    for c in range(nk):
                        acc += projectors[l, r, c] * cart_buf[c]
                    sph_buf[r] = acc
                for c in range(nk):
                    if sqrt_ij * sqrt_diag_ints2c2e[k0 + c] > threshold:
                        threeC2E[base + index_k] = sph_buf[c]
                        index_k += 1
            else:
                # Cartesian (or s/p) shell: functions are independent, so only
                # the significant ones are evaluated.
                for c in range(nk):
                    k = k0 + c
                    if sqrt_ij * sqrt_diag_ints2c2e[k] > threshold:
                        threeC2E[base + index_k] = _int3c2e_single(
                            i, j, k, la, ma, na, lb, mb, nb, I, J, IJsq, nprimi, nprimj, tempcoeff1,
                            bfs_coeffs, bfs_prim_norms, bfs_expnts,
                            aux_bfs_coords, aux_bfs_contr_prim_norms, aux_bfs_lmn, aux_bfs_nprim,
                            aux_bfs_coeffs, aux_bfs_prim_norms, aux_bfs_expnts,
                            roots, weights, G, P, L)
                        index_k += 1
    return threeC2E


# ----------------------------------------------------------------------------
# 3. gamma_P = sum_ij D_ij (ij|P)   (every SCF cycle)
# ----------------------------------------------------------------------------
def df_coeff_calculator_algo10(ints3c2e_1d, dmat_tri, indicesA, indicesB, offsets, naux,
                               sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold,
                               strict_schwarz, ncores=None):
    """
    Contract the sparse ``(ij|P)`` with the (triangular, off-diagonal-doubled)
    density matrix: ``gamma_P = sum_{i>=j} D~_ij (ij|P)``.

    ``dmat_tri`` is indexed like the pair list (position ``i*(i+1)//2 + j``) and
    must already contain ``2*D_ij`` for ``i != j``.  The work is split into
    contiguous pair chunks with a private accumulator each and reduced at the
    end, so the result is deterministic for a fixed thread count.
    """
    nthreads = numba.get_num_threads() if ncores is None else max(1, int(ncores))
    npairs = int(np.asarray(indicesA).shape[0])
    nchunks = max(1, min(npairs, 8 * nthreads))
    return _gamma_from_sparse_3c2e(
        np.asarray(ints3c2e_1d), np.asarray(dmat_tri, dtype=np.float64),
        np.asarray(indicesA), np.asarray(indicesB), np.asarray(offsets), int(naux),
        np.asarray(sqrt_ints4c2e_diag, dtype=np.float64),
        np.asarray(sqrt_diag_ints2c2e, dtype=np.float64),
        float(threshold), bool(strict_schwarz), nchunks)


@njit(parallel=True, cache=True, fastmath=True, error_model="numpy", nogil=True)
def _gamma_from_sparse_3c2e(ints3c2e_1d, dmat_tri, indicesA, indicesB, offsets, naux,
                            sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold,
                            strict_schwarz, nchunks):
    npairs = indicesA.shape[0]
    chunk = (npairs + nchunks - 1) // nchunks
    partial = np.zeros((nchunks, naux), dtype=np.float64)
    for ic in prange(nchunks):
        start = ic * chunk
        stop = min(start + chunk, npairs)
        for ij in range(start, stop):
            i = indicesA[ij]
            j = indicesB[ij]
            sqrt_ij = sqrt_ints4c2e_diag[i, j]
            if strict_schwarz and sqrt_ij * sqrt_ij < STRICT_PAIR_CUTOFF:
                continue
            d_ij = dmat_tri[i * (i + 1) // 2 + j]
            base = offsets[ij]
            index_k = 0
            for k in range(naux):
                if sqrt_ij * sqrt_diag_ints2c2e[k] > threshold:
                    partial[ic, k] += ints3c2e_1d[base + index_k] * d_ij
                    index_k += 1
    gamma = np.zeros(naux, dtype=np.float64)
    for ic in range(nchunks):
        for k in range(naux):
            gamma[k] += partial[ic, k]
    return gamma


# ----------------------------------------------------------------------------
# 4. J_ij = sum_P (ij|P) c_P   (every SCF cycle, lower triangle)
# ----------------------------------------------------------------------------
def J_tri_calculator_algo10(ints3c2e_1d, df_coeff, indicesA, indicesB, offsets, size_J_tri,
                            sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold, strict_schwarz):
    """
    Contract the sparse ``(ij|P)`` with the fitting coefficients ``c_P``:
    ``J_tri[i*(i+1)//2 + j] = sum_P (ij|P) c_P`` for ``i >= j``.
    Pairs removed by the screening keep ``J_ij = 0``.
    """
    return _J_tri_from_sparse_3c2e(
        np.asarray(ints3c2e_1d), np.asarray(df_coeff, dtype=np.float64),
        np.asarray(indicesA), np.asarray(indicesB), np.asarray(offsets), int(size_J_tri),
        np.asarray(sqrt_ints4c2e_diag, dtype=np.float64),
        np.asarray(sqrt_diag_ints2c2e, dtype=np.float64),
        float(threshold), bool(strict_schwarz))


@njit(parallel=True, cache=True, fastmath=True, error_model="numpy", nogil=True)
def _J_tri_from_sparse_3c2e(ints3c2e_1d, df_coeff, indicesA, indicesB, offsets, size_J_tri,
                            sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold, strict_schwarz):
    naux = df_coeff.shape[0]
    J_tri = np.zeros(size_J_tri, dtype=np.float64)
    for ij in prange(indicesA.shape[0]):
        i = indicesA[ij]
        j = indicesB[ij]
        sqrt_ij = sqrt_ints4c2e_diag[i, j]
        if strict_schwarz and sqrt_ij * sqrt_ij < STRICT_PAIR_CUTOFF:
            continue
        base = offsets[ij]
        index_k = 0
        val = 0.0
        for k in range(naux):
            if sqrt_ij * sqrt_diag_ints2c2e[k] > threshold:
                val += ints3c2e_1d[base + index_k] * df_coeff[k]
                index_k += 1
        J_tri[i * (i + 1) // 2 + j] = val
    return J_tri
