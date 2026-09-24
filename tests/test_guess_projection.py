"""
Tests of ``pyfock.guess_projection``: projection, transfer and extrapolation of AO densities
between geometries (original code by Prof. Vincenzo Barone, adapted for PyFock).
"""
from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest
import scipy.linalg

from pyfock import Basis, DFT, Integrals, Mol
from pyfock import guess_projection as gp


WATER = [["O", 0.0, 0.0, 0.1173], ["H", 0.0, 0.7572, -0.4692], ["H", 0.0, -0.7572, -0.4692]]


def _mol_basis(atoms, basis_name="def2-SVP"):
    mol = Mol(atoms=[list(a) for a in atoms])
    basis = Basis(mol, {"all": Basis.load(mol=mol, basis_name=basis_name)})
    return mol, basis


def _displaced(atoms, shifts):
    return [[a[0]] + [a[i + 1] + shifts[k][i] for i in range(3)] for k, a in enumerate(atoms)]


def _idempotent_density(mol, basis, nocc, seed=0):
    """A closed-shell density D = 2 C C^T with C^T S C = 1: occupied eigenvectors of the core
    Hamiltonian plus a small random symmetric perturbation (no SCF needed)."""
    S = Integrals.overlap_mat_symm(basis)
    rng = np.random.default_rng(seed)
    X = rng.normal(size=S.shape)
    H = Integrals.kin_mat_symm(basis) + Integrals.nuc_mat_symm(basis, mol) + 0.05 * (X + X.T)
    _, C = scipy.linalg.eigh(H, S)
    return 2.0 * C[:, :nocc] @ C[:, :nocc].T, S


_setup = _mol_basis


def _natural_occupations(D, S):
    L = np.linalg.cholesky(S)
    return np.sort(np.linalg.eigvalsh(L.T @ D @ L))[::-1]


# ---------------------------------------------------------------------------------------
# Reference implementation of the original algorithm (pseudo-inverses and symmetric powers)
# ---------------------------------------------------------------------------------------
def _power(M, p, rcond):
    w, V = np.linalg.eigh(0.5 * (M + M.T))
    w = np.maximum(w, max(float(np.max(np.abs(w))), 1.0) * rcond)
    return (V * w ** p) @ V.T


def _reference_projection(D_old, S_no, S_nn, S_oo=None, electron_count=None, rcond=1e-12):
    T = np.linalg.pinv(S_nn, rcond=rcond) @ S_no
    if S_oo is None:
        P = T @ D_old @ T.T
    else:
        occ, V = np.linalg.eigh(_power(S_oo, 0.5, rcond) @ D_old @ _power(S_oo, 0.5, rcond))
        keep = occ > max(10 * rcond, 1e-8)
        C = T @ (_power(S_oo, -0.5, rcond) @ V[:, keep])
        C = C @ _power(C.T @ S_nn @ C, -0.5, rcond)
        P = (C * np.clip(occ[keep], 0, 2)) @ C.T
    P = 0.5 * (P + P.T)
    if electron_count is not None:
        P *= electron_count / np.trace(P @ S_nn)
    return P


def _reference_extrapolation(densities, electron_count=None, overlap=None, damping=1.0):
    k = len(densities)
    degree = min(k - 1, 2)
    design = np.vander(np.arange(k, dtype=float), N=degree + 1, increasing=True)
    c = np.linalg.lstsq(design.T, np.array([float(k) ** p for p in range(degree + 1)]), rcond=None)[0]
    latest = densities[-1]
    P = latest + damping * (sum(ci * Di for ci, Di in zip(c, densities)) - latest)
    P = 0.5 * (P + P.T)
    if overlap is not None:
        occ, V = np.linalg.eigh(_power(overlap, 0.5, 1e-12) @ P @ _power(overlap, 0.5, 1e-12))
        P = _power(overlap, -0.5, 1e-12) @ ((V * np.clip(occ, 0, 2)) @ V.T) @ _power(overlap, -0.5, 1e-12)
        P = 0.5 * (P + P.T)
    if electron_count is not None:
        metric = np.eye(P.shape[0]) if overlap is None else overlap
        P *= electron_count / np.trace(P @ metric)
    return P


@pytest.fixture(scope="module")
def water_pair():
    mol_a, basis_a = _setup(WATER)
    shifts = [[0.0, 0.0, 0.03], [0.0, 0.04, -0.02], [0.0, -0.01, -0.03]]
    mol_b, basis_b = _setup(_displaced(WATER, shifts))
    D_a, S_a = _idempotent_density(mol_a, basis_a, mol_a.nelectrons // 2)
    S_b = Integrals.overlap_mat_symm(basis_b)
    S_ba = Integrals.cross_overlap_mat_symm(basis_b, basis_a)
    return dict(mol_a=mol_a, basis_a=basis_a, mol_b=mol_b, basis_b=basis_b, D_a=D_a, S_a=S_a, S_b=S_b, S_ba=S_ba)


# ---------------------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("natural_orbitals", [True, False])
def test_projection_reproduces_the_original_algorithm(water_pair, natural_orbitals):
    w = water_pair
    S_oo = w["S_a"] if natural_orbitals else None
    ref = _reference_projection(w["D_a"], w["S_ba"], w["S_b"], S_oo, electron_count=10)
    P, diag = gp.project_density(w["D_a"], w["S_ba"], w["S_b"], overlap_old_old=S_oo,
                                 electron_count=10, return_diagnostics=True)
    np.testing.assert_allclose(P, ref, atol=1e-10)
    assert diag.electron_count_after == pytest.approx(10.0, abs=1e-10)
    assert diag.condition_number == pytest.approx(np.linalg.cond(w["S_b"]), rel=1e-6)


def test_projection_between_bases_is_idempotent_and_keeps_the_electrons(water_pair):
    w = water_pair
    P = gp.project_density_between_bases(w["D_a"], w["basis_a"], w["basis_b"])
    assert np.trace(P @ w["S_b"]) == pytest.approx(10.0, abs=1e-10)
    np.testing.assert_allclose(P @ w["S_b"] @ P, 2.0 * P, atol=1e-10)


def test_projection_onto_the_same_basis_is_the_identity(water_pair):
    w = water_pair
    P = gp.project_density_between_bases(w["D_a"], w["basis_a"], w["basis_a"])
    np.testing.assert_allclose(P, w["D_a"], atol=1e-10)


def test_projected_operator_onto_the_same_basis_is_unchanged(water_pair):
    w = water_pair
    F = Integrals.kin_mat_symm(w["basis_a"])
    np.testing.assert_allclose(gp.project_operator_between_bases(F, w["basis_a"], w["basis_a"]), F, atol=1e-9)


# ---------------------------------------------------------------------------------------
# Transfer with the atoms
# ---------------------------------------------------------------------------------------
def test_transfer_is_exact_for_a_rigid_translation(water_pair):
    """The density moves with the atoms; a projection would leave it behind."""
    w = water_pair
    _, basis_t = _setup(_displaced(WATER, [[0.3, -0.2, 0.1]] * 3))
    S_t = Integrals.overlap_mat_symm(basis_t)
    np.testing.assert_allclose(S_t, w["S_a"], atol=1e-12)
    D_t = gp.transfer_density(w["D_a"], w["S_a"], S_t, electron_count=10)
    np.testing.assert_allclose(D_t, w["D_a"], atol=1e-10)
    D_p = gp.project_density_between_bases(w["D_a"], w["basis_a"], basis_t)
    assert np.abs(D_p - w["D_a"]).max() > 1e-2


def test_transfer_keeps_the_electrons_and_idempotency(water_pair):
    w = water_pair
    D = gp.transfer_density(w["D_a"], w["S_a"], w["S_b"])
    assert np.trace(D @ w["S_b"]) == pytest.approx(10.0, abs=1e-10)
    np.testing.assert_allclose(D @ w["S_b"] @ D, 2.0 * D, atol=1e-10)
    # the raw matrix has neither property in the new metric
    assert abs(np.trace(w["D_a"] @ w["S_b"]) - 10.0) > 1e-4


def test_natural_orbitals_reconstruct_the_density(water_pair):
    w = water_pair
    C, occ = gp.natural_orbitals(w["D_a"], w["S_a"])
    np.testing.assert_allclose((C * occ) @ C.T, w["D_a"], atol=1e-10)
    np.testing.assert_allclose(C.T @ w["S_a"] @ C, np.eye(len(occ)), atol=1e-10)
    np.testing.assert_allclose(occ, 2.0, atol=1e-10)


def test_linearly_dependent_overlap_uses_the_pseudo_inverse():
    """Duplicated basis functions: Cholesky is not usable, the eigenvalue cutoff takes over."""
    rng = np.random.default_rng(3)
    A = rng.normal(size=(6, 5))
    B = np.hstack([A, A[:, :1]])           # 6 functions, rank 5
    S = B.T @ B
    metric = gp._Metric(S)
    assert metric.cholesky is None
    x = rng.normal(size=6)
    b = S @ x
    np.testing.assert_allclose(S @ metric.solve(b), b, atol=1e-9)


# ---------------------------------------------------------------------------------------
# Extrapolation
# ---------------------------------------------------------------------------------------
def test_uniform_weights():
    np.testing.assert_allclose(gp.extrapolation_weights(2), [-1.0, 2.0], atol=1e-12)
    np.testing.assert_allclose(gp.extrapolation_weights(3), [1.0, -3.0, 3.0], atol=1e-12)
    for k in (2, 3, 4, 6):
        assert gp.extrapolation_weights(k).sum() == pytest.approx(1.0, abs=1e-12)
    # least squares for longer histories: still exact for quadratics
    t = np.arange(5.0)
    c = gp.extrapolation_weights(5)
    assert c @ (1.0 + 2.0 * t - 0.5 * t ** 2) == pytest.approx(1.0 + 2.0 * 5 - 0.5 * 25, abs=1e-10)


def test_extrapolation_is_exact_for_quadratic_histories():
    rng = np.random.default_rng(1)
    A, B, C = (0.5 * (M + M.T) for M in rng.normal(size=(3, 7, 7)))
    history = [A + t * B + t * t * C for t in range(4)]
    np.testing.assert_allclose(gp.extrapolate_density(history[:3]), history[3], atol=1e-10)
    np.testing.assert_allclose(gp.extrapolate_density(history[1:3]), 2 * history[2] - history[1], atol=1e-12)


def test_extrapolation_reproduces_the_original_algorithm(water_pair):
    w = water_pair
    D_b = gp.transfer_density(w["D_a"], w["S_a"], w["S_b"])
    D_c, _ = _idempotent_density(w["mol_b"], w["basis_b"], 5, seed=4)
    history = [w["D_a"], D_b, D_c]
    for damping in (1.0, 0.5):
        ref = _reference_extrapolation(history, electron_count=10, overlap=w["S_b"], damping=damping)
        new = gp.extrapolate_density(history, electron_count=10, overlap=w["S_b"], damping=damping)
        np.testing.assert_allclose(new, ref, atol=1e-10)
        occ = _natural_occupations(new, w["S_b"])
        assert occ.min() > -1e-10 and np.trace(new @ w["S_b"]) == pytest.approx(10.0, abs=1e-10)


def test_geometry_weights():
    R0 = np.zeros((3, 3))
    d = np.array([[0.1, 0.0, 0.0], [0.0, 0.05, 0.0], [0.0, 0.0, -0.02]])
    e = np.array([[0.0, 0.1, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])   # orthogonal to d
    assert np.sum(d * e) == 0.0
    # equal collinear steps: linear extrapolation
    np.testing.assert_allclose(gp.geometry_extrapolation_weights([R0, R0 + d], R0 + 2 * d), [-1.0, 2.0])
    # half a step further
    np.testing.assert_allclose(gp.geometry_extrapolation_weights([R0, R0 + d], R0 + 1.5 * d), [-0.5, 1.5])
    # a new direction: nothing to extrapolate from, the latest density is kept
    np.testing.assert_allclose(gp.geometry_extrapolation_weights([R0, R0 + d], R0 + d + e), [0.0, 1.0], atol=1e-12)
    # stepping back to an earlier geometry recovers its density
    np.testing.assert_allclose(gp.geometry_extrapolation_weights([R0, R0 + d, R0 + d + e], R0 + d),
                               [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(gp.geometry_extrapolation_weights([R0, R0 + d, R0 + d + e], R0),
                               [1.0, 0.0, 0.0], atol=1e-12)
    # nearly collinear history: the ill-determined direction is dropped, no huge weights
    c = gp.geometry_extrapolation_weights([R0, R0 + d, R0 + 2 * d + 1e-4 * e], R0 + 2 * d + e)
    assert np.abs(c).max() < 5.0 and c.sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------------------
# History driver with real SCF densities
# ---------------------------------------------------------------------------------------
def _scf(mol, basis, dmat=None):
    dft = DFT(mol, basis, xc="PBE", ncores=2, conv_crit=1e-8)
    dft.grid_pruning_use_core_guess = True
    if dmat is not None:
        dft.dmat = dmat
    with contextlib.redirect_stdout(io.StringIO()):
        energy, D = dft.scf()
    assert dft.converged
    return energy, D, dft.niter


@pytest.fixture(scope="module")
def water_path():
    """Three converged water densities along a straight path and the geometry that follows."""
    step = np.array([[0.0, 0.0, 0.02], [0.0, 0.02, -0.01], [0.0, -0.02, -0.01]])
    points = [_setup(_displaced(WATER, k * step)) for k in range(4)]
    dens = [_scf(mol, basis) for mol, basis in points]
    return points, dens


@pytest.mark.parametrize("method", gp.GUESS_METHODS)
def test_history_guess_has_the_right_electron_count(water_path, method):
    points, dens = water_path
    history = gp.DensityHistory(method=method)
    assert history.guess(points[0][1], 10) is None
    for (mol, basis), (_, D, _) in zip(points[:3], dens[:3]):
        history.append(basis, D)
    D = history.guess(points[3][1], points[3][0].nelectrons)
    S = Integrals.overlap_mat_symm(points[3][1])
    if method == "previous":
        np.testing.assert_allclose(D, dens[2][1])
    else:
        assert np.trace(D @ S) == pytest.approx(10.0, abs=1e-8)
        assert _natural_occupations(D, S).min() > -1e-8
    assert history.last_info["method"] == method


def test_history_extrapolation_beats_reusing_the_previous_density(water_path):
    """Along a straight path the geometric extrapolation is second-order accurate."""
    points, dens = water_path
    target = dens[3][1]
    L = np.linalg.cholesky(Integrals.overlap_mat_symm(points[3][1]))
    errors = {}
    for method in ("previous", "transfer", "extrapolate"):
        history = gp.DensityHistory(method=method)
        for (mol, basis), (_, D, _) in zip(points[:3], dens[:3]):
            history.append(basis, D)
        errors[method] = np.linalg.norm(L.T @ (history.guess(points[3][1], 10) - target) @ L)
    assert errors["extrapolate"] < 0.3 * errors["previous"]
    assert errors["transfer"] < errors["previous"]
    # the weights express the new geometry through the three earlier ones (exactly, on a line)
    c = np.asarray(history.last_info["weights"])
    R = np.stack([np.asarray(mol.coordsBohrs) for mol, _ in points])
    np.testing.assert_allclose(np.tensordot(c, R[:3], axes=1), R[3], atol=1e-8)


def test_history_rejects_a_different_basis(water_path):
    points, dens = water_path
    history = gp.DensityHistory()
    history.append(points[0][1], dens[0][1])
    _, sto = _setup(WATER, "sto-3g")
    with pytest.raises(ValueError):
        history.guess(sto, 10)


def test_transferred_sao_density_stays_spherical():
    """With sao=True the SCF density lives in the spherical subspace of the Cartesian basis; carrying
    and extrapolating it must not leak into the Cartesian-only components."""
    step = np.array([[0.0, 0.0, 0.03], [0.0, 0.02, -0.02], [0.0, -0.03, -0.01]])
    points = [_setup(_displaced(WATER, k * step)) for k in range(3)]
    history = gp.DensityHistory(method="extrapolate")
    for mol, basis in points[:2]:
        dft = DFT(mol, basis, xc="PBE", ncores=2, conv_crit=1e-8)
        dft.sao = True
        with contextlib.redirect_stdout(io.StringIO()):
            _, D = dft.scf()
        history.append(basis, D)
    basis = points[2][1]
    D = history.guess(basis, 10)
    T = basis.cart2sph_basis()                      # CAO -> SAO
    P = T.T @ np.linalg.pinv(T @ T.T) @ T           # projector onto the spherical rows' span
    np.testing.assert_allclose(P.T @ D @ P, D, atol=1e-10)
