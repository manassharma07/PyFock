"""
Unit tests for the multipole-accelerated density-fitting Coulomb algorithm (``DF_algo=12``,
:mod:`pyfock.Integrals.df_algo12_helpers`) and its solid-harmonic toolkit
(:mod:`pyfock.Integrals.multipole_helpers`).

The toolkit is checked against brute force (explicit Legendre functions, point-charge
sums, numerical quadrature).  The algorithm is checked block by block (near-field Rys
plus far-field multipoles against the exact Rys block) and as a whole (gamma, J and the
DF Coulomb energy against DF_algo=11) on a chain of four water molecules, which has a
substantial far field, and on a single water molecule, which has none.
"""
from __future__ import annotations

from math import factorial

import numpy as np
import pytest
import scipy.linalg
from scipy.special import lpmv

from pyfock import Basis, DFT, Integrals, Mol
from pyfock.Integrals import df_algo11_helpers as algo11
from pyfock.Integrals import df_algo12_helpers as algo12
from pyfock.Integrals import multipole_helpers as mp
from pyfock.Integrals.df_algo10_helpers import aux_shell_max_bounds
from pyfock.Integrals.schwarz_helpers import eri_4c2e_diag


AUX_BASIS_NAME = "def2-universal-jfit"
WATER = [["O", 0.0, 0.0, 0.117], ["H", 0.0, 0.757, -0.467], ["H", 0.0, -0.757, -0.467]]


# ----------------------------------------------------------------------------
# toolkit
# ----------------------------------------------------------------------------
def _explicit_regular(r, l, m):
    x, y, z = r
    rr = np.sqrt(x * x + y * y + z * z)
    P = lpmv(abs(m), l, z / rr)
    if m < 0:
        P = (-1) ** abs(m) * factorial(l - abs(m)) / factorial(l + abs(m)) * P
    return rr ** l * P * np.exp(1j * m * np.arctan2(y, x)) / factorial(l + m)


def _explicit_irregular(r, l, m):
    x, y, z = r
    rr = np.sqrt(x * x + y * y + z * z)
    P = lpmv(abs(m), l, z / rr)
    if m < 0:
        P = (-1) ** abs(m) * factorial(l - abs(m)) / factorial(l + abs(m)) * P
    return factorial(l - m) * P * np.exp(1j * m * np.arctan2(y, x)) / rr ** (l + 1)


def _real_from_complex(fn, r, lmax):
    out = np.zeros((lmax + 1) ** 2)
    for l in range(lmax + 1):
        out[mp.harmonic_index(l, 0)] = fn(r, l, 0).real
        for m in range(1, l + 1):
            c = fn(r, l, m)
            out[mp.harmonic_index(l, m)] = np.sqrt(2.0) * c.real
            out[mp.harmonic_index(l, -m)] = np.sqrt(2.0) * c.imag
    return out


def test_harmonics_match_explicit_legendre_functions():
    rng = np.random.default_rng(3)
    lmax = 9
    for _ in range(3):
        r = rng.normal(size=3)
        R = np.zeros((lmax + 1) ** 2)
        I = np.zeros((lmax + 1) ** 2)
        mp.regular_harmonics(r[0], r[1], r[2], lmax, R)
        mp.irregular_harmonics(r[0], r[1], r[2], lmax, I)
        Rref = _real_from_complex(_explicit_regular, r, lmax)
        Iref = _real_from_complex(_explicit_irregular, r, lmax)
        np.testing.assert_allclose(R, Rref, atol=1e-13, rtol=1e-12)
        np.testing.assert_allclose(I, Iref, atol=0, rtol=1e-11)
    # monomial tables reproduce the recurrence
    mono_off, mono_abc, mono_coef = mp.regular_harmonic_polynomials(lmax)
    r = rng.normal(size=3)
    Rpoly = np.zeros((lmax + 1) ** 2)
    for lm in range((lmax + 1) ** 2):
        for n in range(mono_off[lm], mono_off[lm + 1]):
            a, b, c = mono_abc[n]
            assert a + b + c == int(np.sqrt(lm))
            Rpoly[lm] += mono_coef[n] * r[0] ** a * r[1] ** b * r[2] ** c
    R = np.zeros((lmax + 1) ** 2)
    mp.regular_harmonics(r[0], r[1], r[2], lmax, R)
    np.testing.assert_allclose(Rpoly, R, atol=1e-13, rtol=1e-12)


def test_expansion_identity_and_addition_theorem():
    rng = np.random.default_rng(5)
    a = rng.normal(size=3) * 0.3
    r = rng.normal(size=3)
    r *= 3.0 / np.linalg.norm(r)
    Ra = np.zeros(31 ** 2)
    Ir = np.zeros(31 ** 2)
    mp.regular_harmonics(*a, 30, Ra)
    mp.irregular_harmonics(*r, 30, Ir)
    assert abs(Ra @ Ir - 1.0 / np.linalg.norm(r - a)) < 1e-13
    l_small, l_big = 4, 8
    pair_off, ent_LM, ent_coef = mp.translation_table(l_small, l_big)
    b = rng.normal(size=3)
    n_big = (l_big + 1) ** 2
    Rab = np.zeros((l_small + l_big + 1) ** 2)
    Ra = np.zeros((l_small + 1) ** 2)
    Rb = np.zeros(n_big)
    mp.regular_harmonics(*(a + b), l_small + l_big, Rab)
    mp.regular_harmonics(*a, l_small, Ra)
    mp.regular_harmonics(*b, l_big, Rb)
    recon = np.zeros_like(Rab)
    for lm in range((l_small + 1) ** 2):
        for jk in range(n_big):
            p = lm * n_big + jk
            for e in range(pair_off[p], pair_off[p + 1]):
                recon[ent_LM[e]] += ent_coef[e] * Ra[lm] * Rb[jk]
    # complete for L <= l_small (every (l, j) split is covered)
    n = (l_small + 1) ** 2
    np.testing.assert_allclose(recon[:n], Rab[:n], atol=1e-13, rtol=1e-12)


def test_translations_against_point_charges():
    rng = np.random.default_rng(11)
    lmax = 12
    n_big = (lmax + 1) ** 2
    pair_off, ent_LM, ent_coef = mp.translation_table(4, lmax)
    B = np.zeros(3)
    charges = [(rng.normal(), rng.normal(size=3) * 0.4) for _ in range(6)]
    # moments about B by translating point-charge monopoles (M2M)
    M_B = np.zeros(n_big)
    Rd = np.zeros(n_big)
    for q, pos in charges:
        mp.regular_harmonics(*(pos - B), lmax, Rd)
        mp.translate_moments(np.array([q]), 0, Rd, lmax, n_big, pair_off, ent_LM, ent_coef, M_B)
    x = np.array([4.0, 2.5, -3.0])
    Ix = np.zeros(n_big)
    mp.irregular_harmonics(*(x - B), lmax, Ix)
    exact = sum(q / np.linalg.norm(x - pos) for q, pos in charges)
    assert abs(M_B @ Ix - exact) < 1e-9 * abs(exact)
    # interaction tensor, both directions, and the local-to-local translation
    C = np.array([5.0, 1.0, -2.0])
    IR = np.zeros((4 + lmax + 1) ** 2)
    mp.irregular_harmonics(*(B - C), 4 + lmax, IR)   # R = O_big - O_small
    T = np.zeros((25, n_big))
    mp.interaction_tensor(IR, 4, lmax, n_big, pair_off, ent_LM, ent_coef, T)
    # (i) box moments -> local expansion at C: potential at C itself is the l = 0 coefficient
    L_C = T @ M_B
    assert abs(L_C[0] - sum(q / np.linalg.norm(C - pos) for q, pos in charges)) < 1e-9
    # (ii) point multipoles at C -> local expansion at B, evaluated near B and re-expanded by L2L
    Ms = rng.normal(size=25) * np.repeat([1.0, 0.3, 0.1, 0.03, 0.01], [1, 3, 5, 7, 9])
    L_B = T.T @ Ms
    for _ in range(3):
        y = B + rng.normal(size=3) * 0.4
        Iy = np.zeros(25)
        mp.irregular_harmonics(*(y - C), 4, Iy)
        Ry = np.zeros(n_big)
        mp.regular_harmonics(*(y - B), lmax, Ry)
        assert abs(L_B @ Ry - Ms @ Iy) < 1e-9
        Rd = np.zeros(n_big)
        mp.regular_harmonics(*(y - B), lmax, Rd)
        L_y = np.zeros(25)
        mp.translate_local(L_B, lmax, Rd, 4, n_big, pair_off, ent_LM, ent_coef, L_y)
        assert abs(L_y[0] - Ms @ Iy) < 1e-9


def test_gaussian_product_moments_against_quadrature():
    la, lb = 1, 1
    A = np.array([0.1, -0.2, 0.3])
    B = np.array([1.0, 0.4, -0.5])
    alpha, beta = 0.9, 1.4
    p = alpha + beta
    P = (alpha * A + beta * B) / p
    Xt = np.zeros((la + 1, lb + 1, la + lb + 1))
    Yt = np.zeros_like(Xt)
    Zt = np.zeros_like(Xt)
    G = np.zeros(3 * (la + lb) + 2)
    mp.gaussian_1d_moments(p, P[0] - A[0], P[0] - B[0], la, lb, la + lb, G, Xt)
    mp.gaussian_1d_moments(p, P[1] - A[1], P[1] - B[1], la, lb, la + lb, G, Yt)
    mp.gaussian_1d_moments(p, P[2] - A[2], P[2] - B[2], la, lb, la + lb, G, Zt)
    mono_off, mono_abc, mono_coef = mp.regular_harmonic_polynomials(la + lb)
    comp = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    out = np.zeros((9, (la + lb + 1) ** 2))
    mp.gaussian_product_moments(3, 3, comp[:, 0], comp[:, 1], comp[:, 2], comp[:, 0], comp[:, 1], comp[:, 2],
                                la, lb, Xt, Yt, Zt, mono_off, mono_abc, mono_coef, False, out)
    pref = np.exp(-alpha * beta / p * np.sum((A - B) ** 2))
    g = np.linspace(-4.5, 4.5, 109)
    dx = g[1] - g[0]
    X, Y, Z = np.meshgrid(g + P[0], g + P[1], g + P[2], indexing="ij")
    gauss = np.exp(-alpha * ((X - A[0]) ** 2 + (Y - A[1]) ** 2 + (Z - A[2]) ** 2)
                   - beta * ((X - B[0]) ** 2 + (Y - B[1]) ** 2 + (Z - B[2]) ** 2))
    Rgrid = np.zeros(((la + lb + 1) ** 2,) + X.shape)
    for lm in range((la + lb + 1) ** 2):
        for n in range(mono_off[lm], mono_off[lm + 1]):
            a, b, c = mono_abc[n]
            Rgrid[lm] += mono_coef[n] * (X - P[0]) ** a * (Y - P[1]) ** b * (Z - P[2]) ** c
    xa = [X - A[0], Y - A[1], Z - A[2]]
    xb = [X - B[0], Y - B[1], Z - B[2]]
    for ia in range(3):
        for ib in range(3):
            f = gauss * xa[ia] * xb[ib]
            num = np.array([(f * Rgrid[lm]).sum() * dx ** 3 for lm in range((la + lb + 1) ** 2)])
            np.testing.assert_allclose(out[ia * 3 + ib] * pref, num, atol=1e-9)


# ----------------------------------------------------------------------------
# the algorithm
# ----------------------------------------------------------------------------
def _system(kind, sao):
    if kind == "water_chain":
        atoms = [[s, x + 5.0 * k, y, z] for k in range(4) for s, x, y, z in WATER]
    else:
        atoms = list(WATER)
    mol = Mol(atoms=atoms)
    basis = Basis(mol, {"all": Basis.load(mol=mol, basis_name="def2-SVP")})
    aux = Basis(mol, {"all": Basis.load(mol=mol, basis_name=AUX_BASIS_NAME)})
    metric = Integrals.rys_2c2e_symm(aux)
    if sao:
        proj = aux.sph2cart_basis() @ aux.cart2sph_basis()
        metric = proj @ metric @ proj.T + 1e-12 * np.eye(aux.bfs_nao)
    sqrt4 = np.sqrt(np.abs(eri_4c2e_diag(basis)))
    sqrt2 = np.sqrt(np.abs(np.diag(metric)))
    if sao:
        sqrt2 = aux_shell_max_bounds(sqrt2, aux)
    S = Integrals.overlap_mat_symm(basis)
    # a smooth, physically scaled symmetric "density" (trace ~ number of electrons)
    dmat = S + 0.3 * S @ S
    dmat *= 10.0 * len(atoms) / 3.0 / np.trace(dmat @ S)
    return basis, aux, sqrt4, sqrt2, metric, dmat


@pytest.mark.parametrize("sao", [False, True])
def test_algo12_far_field_blocks_match_exact_integrals(sao):
    basis, aux, sqrt4, sqrt2, metric, dmat = _system("water_chain", sao)
    plan = algo12.build_plan(basis, aux, sqrt4, sqrt2, 1e-9, True, sao=sao)
    assert plan.fraction_far_field > 0.15          # the chain has a real far field
    assert plan.fraction_far_field_work > plan.fraction_far_field
    checked = 0
    worst = 0.0
    for p in plan.work_iter[::7]:
        for K in range(aux.nshells):
            if plan.Q_pair[p] * plan.Q_aux[K] <= plan.threshold:
                continue
            if not any(plan.ff_eff[plan.grp_branch_eff[p, g], K] for g in range(int(plan.ngroups[p]))):
                continue
            exact = algo12.exact_block(plan, p, K)
            approx = algo12.approx_block(plan, p, K)
            worst = max(worst, np.abs(exact - approx).max())
            checked += 1
    assert checked > 100
    assert worst < 2e-9


@pytest.mark.parametrize("sao", [False, True])
def test_algo12_matches_algo11_water_chain(sao):
    basis, aux, sqrt4, sqrt2, metric, dmat = _system("water_chain", sao)
    naux = aux.bfs_nao
    cho = scipy.linalg.cho_factor(metric)
    p11 = algo11.build_plan(basis, aux, sqrt4, sqrt2, 1e-9, True, sao=sao)
    p12 = algo12.build_plan(basis, aux, sqrt4, sqrt2, 1e-9, True, sao=sao)
    assert p12.n_pairs_significant == p11.n_pairs_significant
    assert p12.memory_gb < p11.memory_gb
    g11 = algo11.gamma_from_plan(p11, dmat)
    g12 = algo12.gamma_from_plan(p12, dmat)
    np.testing.assert_allclose(g12, g11, atol=2e-8, rtol=0)
    c11 = scipy.linalg.cho_solve(cho, g11)
    c12 = scipy.linalg.cho_solve(cho, g12)
    J11 = algo11.J_from_plan(p11, c11)
    J12 = algo12.J_from_plan(p12, c11)
    assert np.allclose(J12, J12.T, atol=1e-14)
    np.testing.assert_allclose(J12, J11, atol=2e-8, rtol=0)
    # DF Coulomb energy of this density
    assert abs(0.5 * c12 @ g12 - 0.5 * c11 @ g11) < 2e-8
    # a generic coefficient vector, not only the fitted one
    coeff = np.cos(np.arange(naux, dtype=float))
    np.testing.assert_allclose(algo12.J_from_plan(p12, coeff), algo11.J_from_plan(p11, coeff), atol=2e-8, rtol=0)


def test_algo12_without_far_field_equals_algo11():
    basis, aux, sqrt4, sqrt2, metric, dmat = _system("h2o", True)
    p11 = algo11.build_plan(basis, aux, sqrt4, sqrt2, 1e-9, True, sao=True)
    p12 = algo12.build_plan(basis, aux, sqrt4, sqrt2, 1e-9, True, sao=True)
    assert p12.n_elements_nf == p12.n_elements_significant == p11.n_elements_significant
    # same integrals, different summation order of the contraction (rounding only)
    g11 = algo11.gamma_from_plan(p11, dmat)
    g12 = algo12.gamma_from_plan(p12, dmat)
    np.testing.assert_allclose(g12, g11, atol=1e-10, rtol=0)
    coeff = np.sin(np.arange(aux.bfs_nao, dtype=float))
    np.testing.assert_allclose(algo12.J_from_plan(p12, coeff), algo11.J_from_plan(p11, coeff), atol=1e-10, rtol=0)


def test_algo12_memory_budgets_agree():
    basis, aux, sqrt4, sqrt2, metric, dmat = _system("water_chain", True)
    full = algo12.build_plan(basis, aux, sqrt4, sqrt2, 1e-9, True, sao=True)
    half = algo12.build_plan(basis, aux, sqrt4, sqrt2, 1e-9, True, sao=True, max_memory_gb=0.5 * full.memory_gb)
    none = algo12.build_plan(basis, aux, sqrt4, sqrt2, 1e-9, True, sao=True, max_memory_gb=0.0)
    assert none.n_pairs_cached == 0 and none.values.size == 0
    assert 0 < half.n_pairs_cached < full.n_pairs_cached
    g_full = algo12.gamma_from_plan(full, dmat)
    coeff = np.sin(np.arange(aux.bfs_nao, dtype=float))
    J_full = algo12.J_from_plan(full, coeff)
    for plan in (half, none):
        np.testing.assert_allclose(algo12.gamma_from_plan(plan, dmat), g_full, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(algo12.J_from_plan(plan, coeff), J_full, atol=1e-12, rtol=1e-12)


def test_algo12_low_memory_matches_stored_moments():
    """low_memory re-translates the moments every pass instead of storing them: same numbers."""
    basis, aux, sqrt4, sqrt2, metric, dmat = _system("water_chain", True)
    stored = algo12.build_plan(basis, aux, sqrt4, sqrt2, 1e-9, True, sao=True)
    lean = algo12.build_plan(basis, aux, sqrt4, sqrt2, 1e-9, True, sao=True, options={"low_memory": True})
    assert stored.fraction_far_field > 0.15
    assert lean.fraction_far_field == stored.fraction_far_field      # same classification
    assert lean.Mtil.size == 0 and stored.Mtil.size > 0
    assert lean.moments_gb < stored.moments_gb
    g_stored = algo12.gamma_from_plan(stored, dmat)
    np.testing.assert_allclose(algo12.gamma_from_plan(lean, dmat), g_stored, atol=1e-11, rtol=0)
    coeff = np.cos(np.arange(aux.bfs_nao, dtype=float))
    np.testing.assert_allclose(algo12.J_from_plan(lean, coeff), algo12.J_from_plan(stored, coeff),
                               atol=1e-11, rtol=0)


def test_algo12_options_are_validated():
    basis, aux, sqrt4, sqrt2, metric, dmat = _system("h2o", False)
    with pytest.raises(ValueError):
        algo12.build_plan(basis, aux, sqrt4, sqrt2, 1e-9, True, options={"boxsize": 2.0})
    with pytest.raises(ValueError):
        algo12.build_plan(basis, aux, sqrt4, sqrt2, 1e-9, True, options={"separation": 0.5})


@pytest.mark.regression
def test_algo12_scf_energy_matches_algo11():
    atoms = [[s, x + 5.0 * k, y, z] for k in range(3) for s, x, y, z in WATER]
    energies = {}
    for algo in (11, 12):
        mol = Mol(atoms=atoms)
        basis = Basis(mol, {"all": Basis.load(mol=mol, basis_name="def2-SVP")})
        aux = Basis(mol, {"all": Basis.load(mol=mol, basis_name=AUX_BASIS_NAME)})
        dft = DFT(mol, basis, aux, xc=[1, 7], save_ao_values=True, ncores=2)
        dft.DF_algo = algo
        dft.sao = True
        dft.conv_crit = 1e-8
        energies[algo], _ = dft.scf()
        assert dft.converged
    assert abs(energies[12] - energies[11]) < 5e-8
