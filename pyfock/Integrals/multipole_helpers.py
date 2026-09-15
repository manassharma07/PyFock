"""
Real scaled solid harmonics, multipole translation tables and the multipole moments
of Gaussian product distributions.  This is the far-field machinery of the
multipole-accelerated density-fitting Coulomb term (``DF_algo=12``,
:mod:`~pyfock.Integrals.df_algo12_helpers`).

Conventions
-----------
Complex *scaled* solid harmonics (the scaling of the fast multipole method)::

    R_lm(r) = r^l P_l^m(cos theta) e^{i m phi} / (l+m)!      (regular)
    I_lm(r) = (l-m)! P_l^m(cos theta) e^{i m phi} / r^{l+1}  (irregular)

with the Condon-Shortley phase of ``P_l^m`` and ``R_{l,-m} = (-1)^m conj(R_lm)``.  In this
scaling the two classical identities carry no coefficients::

    1/|r - a|      = sum_lm R_lm(a) conj(I_lm(r))                        |r| > |a|
    R_LM(a + b)    = sum_{l<=L} sum_m R_lm(a) R_{L-l,M-m}(b)

The code works with the **real** combinations (``m > 0``: ``sqrt(2) Re``, ``m < 0``:
``sqrt(2) Im``, ``m = 0``: the real value), stored at ``harmonic_index(l, m) = l*l + l + m``.
For them ``1/|r - a| = sum_lm R_lm(a) I_lm(r)`` (no conjugation) and the addition
theorem becomes ``R_LM(a + b) = sum_{lm,jk} A^{LM}_{lm,jk} R_lm(a) R_jk(b)`` with the
real, sparse coefficients ``A`` of :func:`translation_table` (``L = l + j``).  The same
table drives the three translations used by the far field:

* moments of a distribution centred at ``P`` about a box centre ``B``
  (``M_B[LM] = sum A^{LM}_{lm,jk} M_P[lm] R_jk(P - B)``, :func:`translate_moments`),
* a local (Taylor) expansion about ``B`` re-expanded about ``P``
  (``L_P[jk] = sum A^{LM}_{jk,lm} R_lm(P - B) L_B[LM]``, :func:`translate_local`),
* the interaction tensor between an expansion of low order (``l <= l_small``, e.g. the
  exact moments of one atom's auxiliary functions) about ``O_small`` and one of high
  order about ``O_big`` (a box), ``T_{lm,jk}(R) = (-1)^j sum_M A^{l+j,M}_{lm,jk}
  I_{l+j,M}(R)`` with ``R = O_big - O_small`` (:func:`interaction_tensor`); it maps
  moments on either side to the local expansion on the other.

The multipole moments of a distribution ``rho`` about ``O`` are
``M_lm = int rho(r) R_lm(r - O) d3r``; the potential of ``rho`` at ``r`` outside its
extent is ``sum_lm M_lm I_lm(r - O)``, and the Coulomb interaction of two
non-overlapping distributions is ``sum_{lm,jk} M^S_lm T_{lm,jk}(R) M^T_jk``.

A primitive Cartesian Gaussian product ``x^a y^b z^c exp(-p |r - P|^2)`` has non-zero
moments about its own centre ``P`` only up to ``l = a + b + c``
(:func:`gaussian_product_moments`), so the moments of the density-side distributions
are exact and finite; only the box-level re-expansions are truncated at ``lmax``.
"""

import numpy as np
from numba import njit

__all__ = [
    'harmonic_index', 'regular_harmonics', 'irregular_harmonics',
    'regular_harmonic_polynomials', 'translation_table',
    'translate_moments', 'translate_local', 'interaction_tensor',
    'gaussian_1d_moments', 'gaussian_product_moments',
]

SQRT2 = 1.4142135623730951
PI = 3.141592653589793


def harmonic_index(l, m):
    """Position of the real component ``(l, m)`` in a harmonics vector (``l*l + l + m``)."""
    return l * l + l + m


# ----------------------------------------------------------------------------
# Real scaled solid harmonics by recurrence
# ----------------------------------------------------------------------------
@njit(cache=True, fastmath=True, nogil=True, error_model='numpy', boundscheck=False)
def regular_harmonics(x, y, z, lmax, out):
    """
    Real scaled regular solid harmonics ``R_lm(x, y, z)`` for ``l <= lmax`` into
    ``out[:(lmax+1)**2]`` (layout :func:`harmonic_index`).
    """
    rr = x * x + y * y + z * z
    out[0] = 1.0
    if lmax == 0:
        return
    # l = 1 from R_00 (no sqrt(2) on the seed)
    out[1] = -y * SQRT2 * 0.5   # (1,-1): sqrt2 * Im(-(x+iy)/2)
    out[2] = z                   # (1, 0)
    out[3] = -x * SQRT2 * 0.5   # (1, 1): sqrt2 * Re(-(x+iy)/2)
    for l in range(2, lmax + 1):
        base = l * l + l
        pbase = (l - 1) * (l - 1) + (l - 1)
        ppbase = (l - 2) * (l - 2) + (l - 2)
        # m = l from m = l-1 of the previous degree: R_ll = -(x+iy)/(2l) R_{l-1,l-1}
        re_p = out[pbase + (l - 1)]
        im_p = out[pbase - (l - 1)]
        out[base + l] = -(x * re_p - y * im_p) / (2.0 * l)
        out[base - l] = -(x * im_p + y * re_p) / (2.0 * l)
        # m = 0 .. l-1: ((2l-1) z R_{l-1,m} - r^2 R_{l-2,m}) / ((l-m)(l+m)), same for Re and Im parts
        for m in range(0, l):
            denom = 1.0 / ((l - m) * (l + m))
            t = (2.0 * l - 1.0) * z * out[pbase + m]
            if l - 2 >= m:
                t -= rr * out[ppbase + m]
            out[base + m] = t * denom
            if m > 0:
                t = (2.0 * l - 1.0) * z * out[pbase - m]
                if l - 2 >= m:
                    t -= rr * out[ppbase - m]
                out[base - m] = t * denom


@njit(cache=True, fastmath=True, nogil=True, error_model='numpy', boundscheck=False)
def irregular_harmonics(x, y, z, lmax, out):
    """
    Real scaled irregular solid harmonics ``I_lm(x, y, z)`` for ``l <= lmax`` into
    ``out[:(lmax+1)**2]`` (layout :func:`harmonic_index`).  ``r`` must be non-zero.
    """
    rr = x * x + y * y + z * z
    inv_rr = 1.0 / rr
    out[0] = np.sqrt(inv_rr)
    if lmax == 0:
        return
    # l = 1: I_11 = -(x+iy)/r^2 I_00 ; I_10 = z/r^2 I_00
    out[1] = -y * inv_rr * out[0] * SQRT2
    out[2] = z * inv_rr * out[0]
    out[3] = -x * inv_rr * out[0] * SQRT2
    for l in range(2, lmax + 1):
        base = l * l + l
        pbase = (l - 1) * (l - 1) + (l - 1)
        ppbase = (l - 2) * (l - 2) + (l - 2)
        re_p = out[pbase + (l - 1)]
        im_p = out[pbase - (l - 1)]
        f = (2.0 * l - 1.0) * inv_rr
        out[base + l] = -f * (x * re_p - y * im_p)
        out[base - l] = -f * (x * im_p + y * re_p)
        for m in range(0, l):
            t = (2.0 * l - 1.0) * z * out[pbase + m]
            if l - 2 >= m:
                t -= ((l - 1) * (l - 1) - m * m) * out[ppbase + m]
            out[base + m] = t * inv_rr
            if m > 0:
                t = (2.0 * l - 1.0) * z * out[pbase - m]
                if l - 2 >= m:
                    t -= ((l - 1) * (l - 1) - m * m) * out[ppbase - m]
                out[base - m] = t * inv_rr


# ----------------------------------------------------------------------------
# Cartesian monomial expansion of the real regular harmonics
# ----------------------------------------------------------------------------
def _complex_regular_polynomials(lmax):
    """Complex R_lm as dicts {(a, b, c): coefficient} of x^a y^b z^c, built by recurrence."""
    def add(dst, src, factor):
        for k, v in src.items():
            dst[k] = dst.get(k, 0.0) + factor * v

    def mul_xyz(src, dx, dy, dz, factor):
        out = {}
        for (a, b, c), v in src.items():
            out[(a + dx, b + dy, c + dz)] = out.get((a + dx, b + dy, c + dz), 0.0) + factor * v
        return out

    polys = {(0, 0): {(0, 0, 0): 1.0 + 0.0j}}
    for l in range(1, lmax + 1):
        prev = polys[(l - 1, l - 1)]
        top = {}
        add(top, mul_xyz(prev, 1, 0, 0, 1.0), -1.0 / (2 * l))
        add(top, mul_xyz(prev, 0, 1, 0, 1.0), -1j / (2 * l))
        polys[(l, l)] = top
        for m in range(0, l):
            cur = {}
            add(cur, mul_xyz(polys[(l - 1, m)], 0, 0, 1, 1.0), (2 * l - 1))
            if l - 2 >= m:
                p2 = polys[(l - 2, m)]
                add(cur, mul_xyz(p2, 2, 0, 0, 1.0), -1.0)
                add(cur, mul_xyz(p2, 0, 2, 0, 1.0), -1.0)
                add(cur, mul_xyz(p2, 0, 0, 2, 1.0), -1.0)
            polys[(l, m)] = {k: v / ((l - m) * (l + m)) for k, v in cur.items()}
        for m in range(1, l + 1):
            polys[(l, -m)] = {k: (-1) ** m * np.conj(v) for k, v in polys[(l, m)].items()}
    return polys


def regular_harmonic_polynomials(lmax, tol=1e-14):
    """
    Monomial expansion of the real scaled regular harmonics.

    Returns ``(mono_off, mono_abc, mono_coef)``: component ``harmonic_index(l, m)``
    is ``sum_n mono_coef[n] x^a y^b z^c`` with ``(a, b, c) = mono_abc[n]`` for ``n`` in
    ``mono_off[lm]:mono_off[lm+1]``; ``a + b + c = l`` for every monomial.
    """
    cpolys = _complex_regular_polynomials(lmax)
    mono_off = np.zeros((lmax + 1) ** 2 + 1, dtype=np.int64)
    abc = []
    coef = []
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            src = cpolys[(l, abs(m))]
            if m == 0:
                terms = {k: v.real for k, v in src.items()}
            elif m > 0:
                terms = {k: SQRT2 * v.real for k, v in src.items()}
            else:
                terms = {k: SQRT2 * v.imag for k, v in src.items()}
            for k in sorted(terms):
                if abs(terms[k]) > tol:
                    abc.append(k)
                    coef.append(terms[k])
            mono_off[harmonic_index(l, m) + 1] = len(coef)
    return (mono_off, np.array(abc, dtype=np.int64).reshape(-1, 3), np.array(coef, dtype=np.float64))


# ----------------------------------------------------------------------------
# Real addition (translation) coefficients
# ----------------------------------------------------------------------------
def _complex_to_real_matrix(l):
    """``U`` with ``R^real_{l,m1} = sum_m U[m1+l, m+l] R^complex_{l,m}`` (unitary)."""
    n = 2 * l + 1
    U = np.zeros((n, n), dtype=complex)
    U[l, l] = 1.0
    for m in range(1, l + 1):
        U[l + m, l + m] = 1.0 / SQRT2
        U[l + m, l - m] = (-1) ** m / SQRT2
        U[l - m, l + m] = -1j / SQRT2
        U[l - m, l - m] = 1j * (-1) ** m / SQRT2
    return U


def translation_table(l_small, l_big, tol=1e-13):
    """
    Sparse real addition coefficients ``A^{LM}_{lm,jk}`` (``L = l + j``) for
    ``l <= l_small`` and ``j <= l_big``.

    Returns ``(pair_off, ent_LM, ent_coef)`` in CSR form over the pair index
    ``p = harmonic_index(l, m) * (l_big + 1)**2 + harmonic_index(j, k)``: the entries
    ``ent_LM[e], ent_coef[e]`` for ``e`` in ``pair_off[p]:pair_off[p+1]`` are the
    non-zero ``(harmonic_index(L, M), A)`` of that pair.  ``A`` is symmetric under the
    exchange of ``(lm)`` and ``(jk)``.
    """
    n_big = (l_big + 1) ** 2
    n_small = (l_small + 1) ** 2
    Us = {l: _complex_to_real_matrix(l) for l in range(l_small + l_big + 1)}
    entries = [[] for _ in range(n_small * n_big)]
    for l in range(l_small + 1):
        Ul = Us[l]
        for j in range(l_big + 1):
            L = l + j
            UL = Us[L]
            Uj = Us[j]
            # W[M, m1, k1] = sum_m conj(Ul[m1, m]) conj(Uj[k1, M - m])
            W = np.zeros((2 * L + 1, 2 * l + 1, 2 * j + 1), dtype=complex)
            for M in range(-L, L + 1):
                for m in range(max(-l, M - j), min(l, M + j) + 1):
                    k = M - m
                    W[M + L] += np.conj(Ul[:, m + l])[:, None] * np.conj(Uj[:, k + j])[None, :]
            # A[M1, m1, k1] = sum_M UL[M1, M] W[M, m1, k1]
            A = np.einsum('ab,bcd->acd', UL, W)
            if np.abs(A.imag).max() > 1e-10:
                raise RuntimeError('translation coefficients are not real (l=%d, j=%d)' % (l, j))
            A = A.real
            for m1 in range(-l, l + 1):
                lm = harmonic_index(l, m1)
                for k1 in range(-j, j + 1):
                    jk = harmonic_index(j, k1)
                    p = lm * n_big + jk
                    for M1 in range(-L, L + 1):
                        v = A[M1 + L, m1 + l, k1 + j]
                        if abs(v) > tol:
                            entries[p].append((harmonic_index(L, M1), v))
    pair_off = np.zeros(n_small * n_big + 1, dtype=np.int64)
    ent_LM = []
    ent_coef = []
    for p, ent in enumerate(entries):
        for LM, v in ent:
            ent_LM.append(LM)
            ent_coef.append(v)
        pair_off[p + 1] = len(ent_LM)
    return pair_off, np.array(ent_LM, dtype=np.int64), np.array(ent_coef, dtype=np.float64)


# ----------------------------------------------------------------------------
# Translations
# ----------------------------------------------------------------------------
@njit(cache=True, fastmath=True, nogil=True, error_model='numpy', boundscheck=False)
def translate_moments(M_src, l_src, Rd, lmax, n_big, pair_off, ent_LM, ent_coef, M_out):
    """
    Add to ``M_out`` (about ``B``, ``l <= lmax``) the moments ``M_src`` (about ``P``,
    ``l <= l_src``) of the same distribution: ``M_out[LM] += sum A^{LM}_{lm,jk}
    M_src[lm] R_jk(d)`` with ``Rd`` the regular harmonics of ``d = P - B`` (``j <= lmax``).
    ``n_big`` is the table's ``(l_big + 1)**2`` and must satisfy ``l_big >= lmax``.
    """
    n_out = (lmax + 1) * (lmax + 1)
    for l in range(l_src + 1):
        for m in range(-l, l + 1):
            lm = l * l + l + m
            v = M_src[lm]
            if v == 0.0:
                continue
            jmax = lmax - l
            for j in range(jmax + 1):
                for k in range(-j, j + 1):
                    jk = j * j + j + k
                    w = v * Rd[jk]
                    p = lm * n_big + jk
                    for e in range(pair_off[p], pair_off[p + 1]):
                        LM = ent_LM[e]
                        if LM < n_out:
                            M_out[LM] += ent_coef[e] * w


@njit(cache=True, fastmath=True, nogil=True, error_model='numpy', boundscheck=False)
def translate_local(L_big, lmax, Rd, l_dst, n_big, pair_off, ent_LM, ent_coef, L_out):
    """
    Local expansion ``L_big`` about ``B`` (``l <= lmax``) re-expanded about ``P``:
    ``L_out[jk] = sum_{LM} sum_{lm} A^{LM}_{jk,lm} R_lm(d) L_big[LM]`` for ``j <= l_dst``,
    with ``Rd`` the regular harmonics of ``d = P - B``.  Overwrites ``L_out``.
    """
    n_dst = (l_dst + 1) * (l_dst + 1)
    for i in range(n_dst):
        L_out[i] = 0.0
    for j in range(l_dst + 1):
        for k in range(-j, j + 1):
            jk = j * j + j + k
            acc = 0.0
            lmax_l = lmax - j
            for l in range(lmax_l + 1):
                for m in range(-l, l + 1):
                    lm = l * l + l + m
                    r = Rd[lm]
                    if r == 0.0:
                        continue
                    p = jk * n_big + lm
                    s = 0.0
                    for e in range(pair_off[p], pair_off[p + 1]):
                        s += ent_coef[e] * L_big[ent_LM[e]]
                    acc += r * s
            L_out[jk] = acc


@njit(cache=True, fastmath=True, nogil=True, error_model='numpy', boundscheck=False)
def interaction_tensor(I_R, l_small, l_big, n_big, pair_off, ent_LM, ent_coef, T):
    """
    ``T[lm, jk] = (-1)^j sum_M A^{l+j,M}_{lm,jk} I_{l+j,M}(R)`` for ``l <= l_small``,
    ``j <= l_big``, from the irregular harmonics ``I_R`` of ``R = O_big - O_small``
    (``L <= l_small + l_big``), the vector from the centre of the ``l_small`` expansion
    to the centre of the ``l_big`` expansion.  One tensor serves both directions:
    ``L_big[jk] = sum_lm T[lm, jk] M_small[lm]`` and ``L_small[lm] = sum_jk T[lm, jk] M_big[jk]``.
    ``T`` has shape ``((l_small+1)**2, (l_big+1)**2)``.
    """
    n_small = (l_small + 1) * (l_small + 1)
    for lm in range(n_small):
        for j in range(l_big + 1):
            sign = 1.0 if (j % 2 == 0) else -1.0
            for k in range(-j, j + 1):
                jk = j * j + j + k
                p = lm * n_big + jk
                s = 0.0
                for e in range(pair_off[p], pair_off[p + 1]):
                    s += ent_coef[e] * I_R[ent_LM[e]]
                T[lm, jk] = sign * s


# ----------------------------------------------------------------------------
# Multipole moments of Gaussian product distributions
# ----------------------------------------------------------------------------
@njit(cache=True, fastmath=True, nogil=True, error_model='numpy', boundscheck=False)
def gaussian_1d_moments(p, PA, PB, la, lb, nmax, G, tab):
    """
    ``tab[a, b, n] = int (x-A)^a (x-B)^b (x-P)^n exp(-p (x-P)^2) dx`` for ``a <= la``,
    ``b <= lb``, ``n <= nmax``; ``PA = P - A``, ``PB = P - B``.  ``G`` is scratch of
    length ``>= la + lb + nmax + 1`` for the Gaussian moments ``int u^k exp(-p u^2)``.
    """
    kmax = la + lb + nmax
    G[0] = np.sqrt(PI / p)
    if kmax >= 1:
        G[1] = 0.0
    for k in range(2, kmax + 1):
        G[k] = (k - 1) * G[k - 2] / (2.0 * p) if (k % 2 == 0) else 0.0
    # binomial expansion of (u + PA)^a (u + PB)^b
    for a in range(la + 1):
        for b in range(lb + 1):
            for n in range(nmax + 1):
                s = 0.0
                # coefficient of u^k in (u+PA)^a (u+PB)^b, times G[k + n]
                for i in range(a + 1):
                    ca = 1.0
                    for t in range(i):
                        ca = ca * (a - t) / (t + 1)
                    ca *= PA ** (a - i)
                    for jb in range(b + 1):
                        cb = 1.0
                        for t in range(jb):
                            cb = cb * (b - t) / (t + 1)
                        cb *= PB ** (b - jb)
                        s += ca * cb * G[i + jb + n]
                tab[a, b, n] = s


@njit(cache=True, fastmath=True, nogil=True, error_model='numpy', boundscheck=False)
def gaussian_product_moments(nA, nB, ax_, ay_, az_, bx_, by_, bz_, lA, lB,
                             Xt, Yt, Zt, mono_off, mono_abc, mono_coef, diag_pair, out):
    """
    Real moments about the product centre ``P`` of all function pairs of a primitive
    shell pair: ``out[r, lm] = sum_n c_n X[ax, bx, a] Y[ay, by, b] Z[az, bz, c]`` over
    the monomials of ``R_lm`` for ``l <= lA + lB``; ``Xt, Yt, Zt`` come from
    :func:`gaussian_1d_moments` (without the common prefactor, which the caller applies).
    Rows follow the DF_algo=11 convention (``ia*nB + ib``, or the lower triangle for a
    diagonal shell pair).
    """
    ltot = lA + lB
    nlm = (ltot + 1) * (ltot + 1)
    for ia in range(nA):
        ibmax = ia + 1 if diag_pair else nB
        for ib in range(ibmax):
            r = (ia * (ia + 1)) // 2 + ib if diag_pair else ia * nB + ib
            axa = ax_[ia]
            aya = ay_[ia]
            aza = az_[ia]
            bxb = bx_[ib]
            byb = by_[ib]
            bzb = bz_[ib]
            for lm in range(nlm):
                s = 0.0
                for n in range(mono_off[lm], mono_off[lm + 1]):
                    s += mono_coef[n] * (Xt[axa, bxb, mono_abc[n, 0]] * Yt[aya, byb, mono_abc[n, 1]]
                                         * Zt[aza, bzb, mono_abc[n, 2]])
                out[r, lm] = s
