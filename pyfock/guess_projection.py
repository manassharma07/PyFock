"""
Starting densities for SCF calculations along a geometry path: AO density projection,
transfer and extrapolation.

When the nuclei move, the atom-centred basis functions move with them, so the converged density
matrix of one geometry is not expressed in the basis of the next one. This module turns converged
densities of earlier geometries (or of another basis set) into a starting density for a new SCF:

Projection (:func:`project_density`, :func:`project_density_between_bases`)
    The occupied natural orbitals of the old density are mapped into the new basis with
    ``S_new^-1 S_new,old`` and re-orthonormalized in the new overlap metric. The projected density
    represents the old density, at its old position in space, as well as the new basis can. This is
    the right tool when the basis set changes at a fixed geometry.

Transfer (:func:`transfer_density`)
    The occupied natural orbitals keep their AO coefficients and are re-orthonormalized in the new
    overlap metric, so the density moves together with the atoms. Along a geometry path this is the
    better starting point: a rigid translation of the molecule, for instance, is reproduced exactly,
    while the projected density would stay behind.

Extrapolation (:func:`extrapolate_density`, :class:`DensityHistory`)
    Densities of several earlier geometries, all brought into the current basis, are combined into a
    prediction for the new geometry, which is then made N-representable in the new metric (natural
    occupations clipped to [0, 2]) and normalized to the electron count. The weights either follow
    Barone's scheme of equally spaced points (linear extrapolation from two points, least-squares
    quadratic from three or more, :func:`extrapolation_weights`) or are fitted to the geometries
    themselves (:func:`geometry_extrapolation_weights`), which suits optimizers whose steps change in
    length and direction.

:func:`project_operator_between_bases` projects a covariant AO operator such as a Kohn-Sham matrix.

The result is only a starting density: every integral is rebuilt at the new geometry and a complete
SCF is run from it.

Credits
-------
Original implementation by Prof. Vincenzo Barone, who wrote it while interfacing PyFock with a
geometry optimizer and contributed it to PyFock: the natural-orbital projection with
re-orthonormalization in the new metric, the covariant operator projection, the extrapolation of a
density history with the N-representability and electron-count safeguards, and the projection
diagnostics. Added to PyFock and adapted by Manas Sharma with Prof. Barone's permission (September
2026): Python 3.9 support, Cholesky-based linear algebra restricted to the occupied subspace, the
transfer of densities with the atoms, geometry-fitted extrapolation weights and the history driver
used by the ASE calculator.
"""

# Postponed evaluation keeps the ``X | Y`` annotations below valid on Python 3.9.
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import scipy.linalg

__all__ = [
    "DensityProjectionDiagnostics",
    "DensityHistory",
    "extrapolate_density",
    "extrapolation_weights",
    "geometry_extrapolation_weights",
    "natural_orbitals",
    "project_density",
    "project_density_between_bases",
    "project_operator_between_bases",
    "transfer_density",
    "displaced_basis",
    "guess_from_checkpoints",
]

GUESS_METHODS = ("previous", "transfer", "project", "extrapolate")
"""Starting-density methods of :class:`DensityHistory` (and of the ASE calculator's ``density_guess``)."""

EXTRAPOLATION_WEIGHTS = ("geometry", "uniform")
"""Weighting schemes of an extrapolated history, see :meth:`DensityHistory.guess`."""

EXTRAPOLATION_REPRESENTATIONS = ("transfer", "project")
"""How older densities are brought into the current basis before they are combined."""

CREDITS = ("Density projection/extrapolation code originally written by Prof. Vincenzo Barone; "
           "added to PyFock and adapted by Manas Sharma with his permission.")


@dataclass(frozen=True)
class DensityProjectionDiagnostics:
    """Checks recorded while projecting a density into a new AO basis."""

    electron_count_before: float
    electron_count_after: float
    density_trace: float
    condition_number: float
    scale_factor: float


# ---------------------------------------------------------------------------------------
# Linear algebra in a (possibly ill-conditioned) AO overlap metric
# ---------------------------------------------------------------------------------------
class _Metric:
    """
    Factorization ``S = X^-T X^-1`` of an AO overlap matrix used for every metric operation here.

    A Cholesky factor ``S = L L^T`` (``X = L^-T``) is used whenever the overlap is safely positive
    definite, which is the normal case and several times cheaper than an eigendecomposition. For
    (nearly) linearly dependent basis sets, eigenvalues below ``rcond * max(1, lambda_max)`` are
    dropped instead (canonical orthogonalization), which reproduces the pseudo-inverse semantics of
    the original implementation.
    """

    def __init__(self, overlap, rcond=1.0e-12):
        S = np.asarray(overlap, dtype=np.float64)
        if S.ndim != 2 or S.shape[0] != S.shape[1]:
            raise ValueError("overlap must be a square matrix")
        if not np.all(np.isfinite(S)):
            raise ValueError("overlap must be finite")
        if rcond <= 0.0:
            raise ValueError("rcond must be positive")
        self.S = 0.5 * (S + S.T)
        self.n = S.shape[0]
        self.cholesky = None
        self.vectors = None
        self.values = None
        try:
            L = scipy.linalg.cholesky(self.S, lower=True, check_finite=False)
            # LAPACK's O(n^2) estimate of the reciprocal 1-norm condition number from the factor:
            # Cholesky is kept unless the overlap is within a safety factor of numerical singularity
            anorm = float(np.max(np.sum(np.abs(self.S), axis=0)))
            rcond_estimate, info = scipy.linalg.lapack.dpocon(L, anorm, uplo='L')
            if info == 0 and rcond_estimate > 1.0e3 * rcond:
                self.cholesky = L
        except scipy.linalg.LinAlgError:
            pass
        if self.cholesky is None:
            w, V = np.linalg.eigh(self.S)
            cutoff = max(float(np.max(np.abs(w))), 1.0) * rcond
            if np.any(w < -cutoff):
                raise ValueError("overlap metric is not positive semidefinite")
            keep = w > cutoff
            self.values, self.vectors = w[keep], V[:, keep]

    def solve(self, B):
        """``S^-1 B`` (pseudo-inverse for linearly dependent bases)."""
        if self.cholesky is not None:
            return scipy.linalg.cho_solve((self.cholesky, True), B, check_finite=False)
        B = np.asarray(B, dtype=np.float64)
        scale = self.values.reshape((-1,) + (1,) * (B.ndim - 1))
        return self.vectors @ ((self.vectors.T @ B) / scale)

    def to_orthonormal(self, D):
        """``X^T S D S X``: a density in an orthonormal basis, whose eigenvalues are the natural occupations."""
        if self.cholesky is not None:
            L = self.cholesky
            return L.T @ D @ L
        Y = self.vectors * np.sqrt(self.values)
        return Y.T @ D @ Y

    def from_orthonormal(self, V):
        """AO coefficients ``X V`` of orthonormal-basis vectors ``V``."""
        if self.cholesky is not None:
            return scipy.linalg.solve_triangular(self.cholesky.T, V, lower=False, check_finite=False)
        return (self.vectors / np.sqrt(self.values)) @ V

    def trace(self, D):
        """Electron count ``tr(D S)`` in O(n^2)."""
        return float(np.einsum("ij,ji->", D, self.S))

    def condition_number(self):
        w = np.linalg.eigvalsh(self.S)
        return float(w[-1] / w[0]) if w[0] > 0.0 else float("inf")


def _as_metric(overlap, rcond=1.0e-12):
    return overlap if isinstance(overlap, _Metric) else _Metric(overlap, rcond)


def _symmetrize(A):
    return 0.5 * (A + A.T)


def _lowdin_orthonormalize(C, metric):
    """``C (C^T S C)^-1/2``: symmetric orthonormalization of the columns of ``C`` in the metric."""
    M = _symmetrize(C.T @ (metric.S @ C))
    w, U = np.linalg.eigh(M)
    if w[0] <= 1.0e-10 * max(1.0, w[-1]):
        raise ValueError("the transferred occupied orbitals are linearly dependent in the new basis")
    return C @ ((U / np.sqrt(w)) @ U.T)


def natural_orbitals(density, overlap, *, threshold=1.0e-8, rcond=1.0e-12):
    """
    Occupied natural orbitals of an AO density matrix.

    Parameters
    ----------
    density : ndarray (n, n)
        AO density matrix.
    overlap : ndarray (n, n)
        AO overlap matrix of the basis the density is expressed in.
    threshold : float
        Natural orbitals with smaller occupation are dropped.

    Returns
    -------
    coefficients : ndarray (n, k)
        AO coefficients, orthonormal in the overlap metric.
    occupations : ndarray (k,)
        Natural occupations, clipped to [0, 2]; ``density = C diag(occ) C^T``.
    """
    metric = _as_metric(overlap, rcond)
    D = np.asarray(density, dtype=np.float64)
    if D.shape != (metric.n, metric.n):
        raise ValueError("density and overlap have incompatible dimensions")
    occ, V = np.linalg.eigh(_symmetrize(metric.to_orthonormal(_symmetrize(D))))
    keep = occ > threshold
    return metric.from_orthonormal(V[:, keep]), np.clip(occ[keep], 0.0, 2.0)


def _density_from_orbitals(C, occ):
    return _symmetrize((C * occ) @ C.T)


def _normalize(D, metric, electron_count, what="density"):
    """Rescale ``D`` to ``tr(D S) = electron_count``; returns (D, trace before, scale)."""
    raw = metric.trace(D)
    if electron_count is None:
        return D, raw, 1.0
    target = float(electron_count)
    if not np.isfinite(target) or target <= 0.0:
        raise ValueError("electron_count must be positive and finite")
    if raw <= np.finfo(np.float64).tiny:
        raise ValueError(what + " has zero electron metric trace")
    scale = target / raw
    return D * scale, raw, scale


def _n_representable(D, metric):
    """Clip the natural occupations of ``D`` (in the metric) to [0, 2]."""
    occ, V = np.linalg.eigh(_symmetrize(metric.to_orthonormal(_symmetrize(D))))
    occ = np.clip(occ, 0.0, 2.0)
    keep = occ > 0.0
    return _density_from_orbitals(metric.from_orthonormal(V[:, keep]), occ[keep])


# ---------------------------------------------------------------------------------------
# Projection, transfer and extrapolation
# ---------------------------------------------------------------------------------------
def project_density(
    density_old: np.ndarray,
    overlap_new_old: np.ndarray,
    overlap_new_new: np.ndarray,
    *,
    overlap_old_old: np.ndarray | None = None,
    electron_count: float | None = None,
    rcond: float = 1.0e-12,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, DensityProjectionDiagnostics]:
    """Project an AO density from an old geometry (or basis) into a new AO basis.

    Parameters
    ----------
    density_old
        Old AO density, in the old basis.
    overlap_new_old
        Cross overlap ``<chi_new|chi_old>``.
    overlap_new_new
        New-basis AO overlap ``<chi_new|chi_new>``.
    overlap_old_old
        Old-basis AO overlap. When given (recommended), the occupied natural orbitals of the old
        density are projected and re-orthonormalized in the new metric, so the result is
        N-representable and keeps the natural occupations; without it the density is simply
        mapped, ``P D_old P^T`` with ``P = S_new^-1 <chi_new|chi_old>``.
    electron_count
        If supplied, rescale the projected density so that
        ``trace(P_new @ S_new)`` equals this value.
    rcond
        Relative eigenvalue cutoff for (nearly) singular overlap matrices.
    return_diagnostics
        Also return electron-count and conditioning diagnostics.

    Notes
    -----
    This is a conservative density projection for SCF restarting.  It does
    not claim that the projected density is an idempotent ground-state
    density; PyFock must run a fresh SCF at the new geometry. For a starting density along a
    geometry path, :func:`transfer_density` is usually the better choice, see the module notes.
    """
    old = np.asarray(density_old, dtype=np.float64)
    cross = np.asarray(overlap_new_old, dtype=np.float64)
    new_overlap = np.asarray(overlap_new_new, dtype=np.float64)
    if old.ndim != 2 or old.shape[0] != old.shape[1]:
        raise ValueError("density_old must be a square matrix")
    if cross.ndim != 2 or cross.shape[1] != old.shape[0]:
        raise ValueError("overlap_new_old has incompatible dimensions")
    if new_overlap.ndim != 2 or new_overlap.shape != (cross.shape[0], cross.shape[0]):
        raise ValueError("overlap_new_new has incompatible dimensions")
    if rcond <= 0.0:
        raise ValueError("rcond must be positive")
    if not all(np.all(np.isfinite(item)) for item in (old, cross, new_overlap)):
        raise ValueError("density and overlap matrices must be finite")

    metric = _Metric(new_overlap, rcond)
    if overlap_old_old is None:
        transform = metric.solve(cross)
        projected = _symmetrize(transform @ old @ transform.T)
    else:
        old_overlap = np.asarray(overlap_old_old, dtype=np.float64)
        if old_overlap.shape != old.shape:
            raise ValueError("overlap_old_old has incompatible dimensions")
        # Recover the occupied natural-orbital subspace in the old metric, project the orbitals
        # (only these k columns are mapped), then re-orthonormalize them in the new metric.
        coefficients_old, occupations = natural_orbitals(old, old_overlap,
                                                         threshold=max(10.0 * rcond, 1.0e-8), rcond=rcond)
        coefficients_new = _lowdin_orthonormalize(metric.solve(cross @ coefficients_old), metric)
        projected = _density_from_orbitals(coefficients_new, occupations)
    # The old electron count is intentionally not inferred: callers should provide the molecule's
    # authoritative electron count when restarting SCF.
    projected, electron_before, scale = _normalize(projected, metric, electron_count, "projected density")
    if not return_diagnostics:
        return projected
    diagnostics = DensityProjectionDiagnostics(
        electron_count_before=electron_before,
        electron_count_after=metric.trace(projected),
        density_trace=float(np.trace(projected)),
        condition_number=metric.condition_number(),
        scale_factor=scale,
    )
    return projected, diagnostics


def project_density_between_bases(
    density_old: np.ndarray,
    basis_old: object,
    basis_new: object,
    *,
    electron_count: float | None = None,
    rcond: float = 1.0e-12,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, DensityProjectionDiagnostics]:
    """Project a density using two PyFock ``Basis`` objects directly (see :func:`project_density`)."""
    from .Integrals import cross_overlap_mat_symm, overlap_mat_symm

    return project_density(
        density_old,
        cross_overlap_mat_symm(basis_new, basis_old),
        overlap_mat_symm(basis_new),
        overlap_old_old=overlap_mat_symm(basis_old),
        electron_count=electron_count,
        rcond=rcond,
        return_diagnostics=return_diagnostics,
    )


def transfer_density(
    density_old: np.ndarray,
    overlap_old: np.ndarray,
    overlap_new: np.ndarray,
    *,
    electron_count: float | None = None,
    rcond: float = 1.0e-12,
) -> np.ndarray:
    """Carry an AO density to a displaced geometry together with its basis functions.

    The occupied natural orbitals of the old density (in the old metric) keep their AO coefficients
    and are re-orthonormalized in the metric of the new geometry. Both bases must be the same basis
    set on the same atoms in the same order. The result is N-representable with the natural
    occupations of the old density; for an idempotent old density it is idempotent in the new metric.
    Unlike the plain reuse of the old matrix it has the right electron count, and unlike a projection
    it moves with the atoms, which is what the electrons mostly do.
    """
    old = np.asarray(density_old, dtype=np.float64)
    metric = _as_metric(overlap_new, rcond)
    if old.shape != (metric.n, metric.n):
        raise ValueError("density_old and overlap_new have incompatible dimensions")
    coefficients, occupations = natural_orbitals(old, overlap_old, rcond=rcond)
    density = _density_from_orbitals(_lowdin_orthonormalize(coefficients, metric), occupations)
    return _normalize(density, metric, electron_count)[0]


def project_operator_between_bases(
    operator_old: np.ndarray,
    basis_old: object,
    basis_new: object,
    *,
    rcond: float = 1.0e-12,
) -> np.ndarray:
    """Project a covariant AO operator (e.g. a Fock/Kohn–Sham matrix) into a new basis.

    Unlike a density, the operator is not occupation-filtered or electron-count normalized.
    """
    from .Integrals import cross_overlap_mat_symm, overlap_mat_symm

    old = np.asarray(operator_old, dtype=np.float64)
    if old.ndim != 2 or old.shape[0] != old.shape[1]:
        raise ValueError("operator_old must be a square matrix")
    metric = _Metric(overlap_mat_symm(basis_new), rcond)
    cross = cross_overlap_mat_symm(basis_new, basis_old)
    # For a covariant operator matrix, the old basis is approximately
    # ``chi_old = chi_new @ T``. Therefore ``F_old = T.T @ F_new @ T`` and
    # the least-squares back transformation is ``T+ .T @ F_old @ T+``.
    coefficient_map = metric.solve(cross)
    inverse_map = np.linalg.pinv(coefficient_map, rcond=rcond)
    return _symmetrize(inverse_map.T @ old @ inverse_map)


def extrapolation_weights(n_points: int) -> np.ndarray:
    """Barone's weights for equally spaced points: the prediction one step beyond the latest point.

    Two points give linear extrapolation (-1, 2); three or more a least-squares quadratic fit,
    which uses the whole history without amplifying SCF noise the way a high-order interpolating
    polynomial would. The weights sum to one and are ordered oldest first.
    """
    k = int(n_points)
    if k < 2:
        raise ValueError("at least two densities are required")
    times = np.arange(k, dtype=np.float64)
    degree = min(k - 1, 2)
    design = np.vander(times, N=degree + 1, increasing=True)
    target_vector = np.array([float(k) ** power for power in range(degree + 1)])
    return np.linalg.lstsq(design.T, target_vector, rcond=None)[0]


def geometry_extrapolation_weights(previous_coordinates, new_coordinates, *, rcond=1.0e-2):
    """Affine weights that express a new geometry through earlier ones.

    Finds ``c`` with ``sum(c) = 1`` such that ``sum_j c_j R_j`` is the point of the affine span of
    the earlier geometries closest to the new one. Applied to their densities, this is a linear model
    of the density along the directions the path has explored: it gives linear extrapolation for
    equal collinear steps, keeps the latest density when the new step is orthogonal to all earlier
    ones and returns to an earlier density when the optimizer steps back. Directions whose
    singular value is below ``rcond`` times the largest one are left out, so nearly collinear
    histories cannot produce large weights.

    Parameters
    ----------
    previous_coordinates : array (k, natoms, 3) or (k, 3 natoms)
        Earlier geometries, oldest first (any consistent length unit).
    new_coordinates : array (natoms, 3) or (3 natoms,)

    Returns
    -------
    ndarray (k,)
    """
    R = np.asarray(previous_coordinates, dtype=np.float64)
    R = R.reshape(R.shape[0], -1)
    target = np.asarray(new_coordinates, dtype=np.float64).ravel()
    if R.shape[0] < 1 or R.shape[1] != target.size:
        raise ValueError("geometries have incompatible shapes")
    weights = np.zeros(R.shape[0])
    weights[-1] = 1.0
    if R.shape[0] == 1:
        return weights
    A = (R[:-1] - R[-1]).T
    step = target - R[-1]
    if not np.any(A):
        return weights
    a = np.linalg.lstsq(A, step, rcond=rcond)[0]
    weights[:-1] = a
    weights[-1] = 1.0 - a.sum()
    return weights


def extrapolate_density(
    densities: tuple[np.ndarray, ...] | list[np.ndarray],
    *,
    electron_count: float | None = None,
    overlap: np.ndarray | None = None,
    damping: float = 1.0,
    weights: np.ndarray | None = None,
    rcond: float = 1.0e-12,
) -> np.ndarray:
    """Extrapolate several densities that are expressed in a common AO basis.

    The input densities (oldest first) must already have been brought into the same current AO
    basis (:func:`transfer_density` or :func:`project_density`). Without ``weights``, Barone's
    weights for equally spaced points are used (:func:`extrapolation_weights`): linear
    extrapolation from two points, a least-squares quadratic fit evaluated one step beyond the
    latest point from three or more. ``weights`` (summing to one, e.g. from
    :func:`geometry_extrapolation_weights`) replace them. ``damping`` in (0, 1] mixes the prediction
    with the latest density, ``latest + damping (prediction - latest)``; use it for histories longer
    than two with the equally spaced weights.

    With ``overlap`` the prediction is made N-representable in that metric (natural occupations
    clipped to [0, 2], which multi-point extrapolation could otherwise push negative); with
    ``electron_count`` it is normalized to ``tr(D S) = electron_count``.
    """
    if len(densities) < 2 and weights is None:
        raise ValueError("at least two densities are required")
    if not 0.0 < damping <= 1.0:
        raise ValueError("damping must be in (0, 1]")
    history = [np.asarray(item, dtype=np.float64) for item in densities]
    if not history:
        raise ValueError("at least one density is required")
    latest = history[-1]
    if latest.ndim != 2 or latest.shape[0] != latest.shape[1]:
        raise ValueError("densities must be square matrices of equal dimensions")
    if any(item.shape != latest.shape for item in history):
        raise ValueError("densities must be square matrices of equal dimensions")
    coefficients = extrapolation_weights(len(history)) if weights is None else np.asarray(weights, dtype=np.float64)
    if coefficients.shape != (len(history),):
        raise ValueError("extrapolation weights do not match the density history")
    if abs(coefficients.sum() - 1.0) > 1.0e-8:
        raise ValueError("extrapolation weights must sum to one")
    # latest + damping * (sum_j c_j D_j - latest), accumulated without temporaries per term
    scaled = damping * coefficients
    scaled[-1] += 1.0 - damping
    predicted = np.zeros_like(latest)
    for coefficient, item in zip(scaled, history):
        if coefficient != 0.0:
            predicted += coefficient * item
    predicted = _symmetrize(predicted)
    if overlap is None and electron_count is None:
        return predicted
    metric = _as_metric(np.eye(latest.shape[0]) if overlap is None else overlap, rcond)
    if metric.n != latest.shape[0]:
        raise ValueError("overlap has incompatible dimensions")
    if overlap is not None:
        predicted = _n_representable(predicted, metric)
    return _normalize(predicted, metric, electron_count, "extrapolated density")[0]


# ---------------------------------------------------------------------------------------
# History of converged densities along a geometry path
# ---------------------------------------------------------------------------------------
IMPLEMENTATIONS = ("pyfock", "original")
"""'pyfock': this module; 'original': Prof. Barone's code as contributed (:mod:`pyfock.guess_projection_original`)."""


class DensityHistory:
    """
    Converged densities of earlier geometries, turned into the starting density of the next SCF.

    Keep one object for a geometry optimization (or any sequence of related geometries with the
    same basis set and atoms), ``append`` every converged density and ask for a ``guess`` before
    each new SCF::

        history = DensityHistory(method='extrapolate')
        for coords in path:
            mol = Mol(atoms=...); basis = Basis(mol, ...)
            dft = DFT(mol, basis, auxbasis, xc='PBE')
            dft.dmat = history.guess(basis, mol.nelectrons)   # None for the first geometry
            energy, dmat = dft.scf()
            if dft.converged:
                history.append(basis, dmat)

    Parameters
    ----------
    method : str
        'previous': the latest converged density matrix unchanged (no metric correction).
        'transfer': the latest density carried with the atoms (:func:`transfer_density`).
        'project': the latest density projected into the new basis (:func:`project_density`).
        'extrapolate' (default): the last ``max_points`` densities, each brought into the new basis
        (``representation``), combined with ``weights`` and made N-representable
        (:func:`extrapolate_density`).
    max_points : int, optional
        Number of densities kept and extrapolated. Default: 5 with the geometry-fitted weights, 3 with
        the equally-spaced-point weights (and the original implementation).
    weights : str
        'geometry' (default): :func:`geometry_extrapolation_weights`; 'uniform': Barone's weights for
        equally spaced points (:func:`extrapolation_weights`).
    representation : str
        'transfer' (default) or 'project': how the older densities are brought into the new basis.
    damping : float
        See :func:`extrapolate_density`.
    rcond : float
        Cutoff of the geometric fit, see :func:`geometry_extrapolation_weights`.
    implementation : str
        'pyfock' (default) or 'original'. The latter runs Prof. Barone's code exactly as contributed
        (:mod:`pyfock.guess_projection_original`): 'project' is its natural-orbital projection and
        'extrapolate' projects every stored density with it and extrapolates them with its
        equally-spaced-point weights, damping and safeguards. It implies ``weights='uniform'`` and
        ``representation='project'`` and has no 'transfer' method; it serves as the reference for
        comparisons of results and timings.
    """

    def __init__(self, method="extrapolate", max_points=None, weights=None, representation=None,
                 damping=1.0, rcond=1.0e-2, implementation="pyfock"):
        if implementation not in IMPLEMENTATIONS:
            raise ValueError("Unknown implementation '" + str(implementation) + "'. Available: "
                             + ", ".join(IMPLEMENTATIONS) + ".")
        if method not in GUESS_METHODS:
            raise ValueError("Unknown starting-density method '" + str(method) + "'. Available: "
                             + ", ".join(GUESS_METHODS) + ".")
        if implementation == "original":
            if method == "transfer":
                raise ValueError("The original implementation has no 'transfer' method.")
            if weights not in (None, "uniform") or representation not in (None, "project"):
                raise ValueError("The original implementation extrapolates projected densities with "
                                 "equally-spaced-point weights (weights='uniform', representation='project').")
            weights, representation = "uniform", "project"
        weights = "geometry" if weights is None else weights
        representation = "transfer" if representation is None else representation
        if weights not in EXTRAPOLATION_WEIGHTS:
            raise ValueError("Unknown extrapolation weights '" + str(weights) + "'. Available: "
                             + ", ".join(EXTRAPOLATION_WEIGHTS) + ".")
        if representation not in EXTRAPOLATION_REPRESENTATIONS:
            raise ValueError("Unknown extrapolation representation '" + str(representation)
                             + "'. Available: " + ", ".join(EXTRAPOLATION_REPRESENTATIONS) + ".")
        if max_points is None:
            max_points = 5 if weights == "geometry" else 3
        if int(max_points) < 1:
            raise ValueError("max_points must be at least 1")
        if not 0.0 < damping <= 1.0:
            raise ValueError("damping must be in (0, 1]")
        self.method = method
        self.max_points = int(max_points) if method == "extrapolate" else 1
        self.weights = weights
        self.representation = representation
        self.damping = float(damping)
        self.rcond = float(rcond)
        self.implementation = implementation
        self._entries = []
        self.last_info = None
        """Summary of the last :meth:`guess` (method, number of points, weights, time in seconds)."""

    def __len__(self):
        return len(self._entries)

    def clear(self):
        self._entries = []

    def append(self, basis, density, overlap=None):
        """Store the converged ``density`` (CAO) of the geometry of ``basis``."""
        D = np.asarray(density, dtype=np.float64)
        if D.shape != (basis.bfs_nao, basis.bfs_nao):
            raise ValueError("density does not match the basis")
        if self._entries and self._entries[-1]["basis"].bfs_nao != basis.bfs_nao:
            self.clear()
        self._entries.append({"basis": basis, "density": D, "overlap": overlap, "orbitals": None,
                              "coords": _basis_centres(basis)})
        del self._entries[:-self.max_points]

    def _orbitals(self, entry):
        """Occupied natural orbitals of a stored density, computed once per entry."""
        from .Integrals import overlap_mat_symm

        if entry["orbitals"] is None:
            S = entry["overlap"] if entry["overlap"] is not None else overlap_mat_symm(entry["basis"])
            entry["orbitals"] = natural_orbitals(entry["density"], S)
        return entry["orbitals"]

    def guess(self, basis, electron_count, overlap=None, ncores=None):
        """Starting density (CAO) at the geometry of ``basis``, or None without a stored density.

        ``ncores`` caps the BLAS threads for the duration of the guess, as the SCF does for its own
        linear algebra. It matters when numpy was imported before the thread count was set in the
        environment: an uncapped multithreaded BLAS can make these small dense operations 10-100
        times slower (2.4 s instead of 0.02 s for ethanol/def2-SVP on a 10-core laptop).
        """
        if ncores is None:
            return self._guess(basis, electron_count, overlap)
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=max(1, int(ncores)), user_api="blas"):
            return self._guess(basis, electron_count, overlap)

    def _guess(self, basis, electron_count, overlap):
        from .Integrals import cross_overlap_mat_symm, overlap_mat_symm

        if not self._entries:
            self.last_info = None
            return None
        if self._entries[-1]["basis"].bfs_nao != basis.bfs_nao:
            raise ValueError("the stored densities belong to a different basis")
        t0 = time.perf_counter()
        entries = self._entries[-self.max_points:]
        weights = np.ones(1)
        if self.method == "previous":
            D = np.array(entries[-1]["density"], copy=True)
        elif self.implementation == "original":
            from . import guess_projection_original as original

            projected = [original.project_density_between_bases(e["density"], e["basis"], basis,
                                                                electron_count=electron_count)
                         for e in entries]
            if self.method == "project" or len(projected) == 1:
                D = projected[-1]
            else:
                weights = extrapolation_weights(len(projected))
                D = original.extrapolate_density(projected, electron_count=electron_count,
                                                 overlap=overlap_mat_symm(basis) if overlap is None else overlap,
                                                 damping=self.damping)
        else:
            metric = _Metric(overlap_mat_symm(basis) if overlap is None else overlap)

            def bring(entry):
                C, occ = self._orbitals(entry)
                if self.method == "project" or (self.method == "extrapolate" and self.representation == "project"):
                    C = metric.solve(cross_overlap_mat_symm(basis, entry["basis"]) @ C)
                return _density_from_orbitals(_lowdin_orthonormalize(C, metric), occ)

            if self.method in ("transfer", "project") or len(entries) == 1:
                D = bring(entries[-1])
            else:
                if self.weights == "geometry":
                    weights = geometry_extrapolation_weights(np.stack([e["coords"] for e in entries]),
                                                             _basis_centres(basis), rcond=self.rcond)
                else:
                    weights = extrapolation_weights(len(entries))
                # points with zero weight need not be brought into the new basis at all
                used = [i for i, w in enumerate(weights) if w != 0.0 or i == len(entries) - 1]
                D = extrapolate_density([bring(entries[i]) for i in used], overlap=metric,
                                        damping=self.damping, weights=weights[used])
            D = _normalize(D, metric, electron_count)[0]
        self.last_info = {"method": self.method, "implementation": self.implementation,
                          "npoints": len(entries), "weights": np.asarray(weights).tolist(),
                          "time": time.perf_counter() - t0}
        return D


def displaced_basis(basis, centres):
    """
    Copy of ``basis`` with its functions centred on ``centres`` (natoms x 3, bohr).

    The copy shares every geometry-independent array with ``basis`` and is meant for the
    overlap-type integrals needed here (``overlap_mat_symm``, ``cross_overlap_mat_symm``); building
    it takes microseconds, while a new ``Basis`` re-reads the basis-set files.
    """
    import copy

    centres = np.asarray(centres, dtype=np.float64).reshape(-1, 3)
    moved = copy.copy(basis)
    moved.bfs_coords = [centres[a] for a in basis.bfs_atoms]
    moved.prim_coords = [centres[a] for a in basis.prim_atoms]
    moved.shell_coords = [centres[basis.bfs_atoms[o]] for o in basis.shell_bfs_offset]
    return moved


def read_xyz_coordinates(path):
    """Cartesian coordinates (natoms x 3, bohr) of an .xyz file, converted like :class:`pyfock.Mol`."""
    from .Data import Data

    with open(path, "r", encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    natoms = int(lines[0].split()[0])
    coords = np.array([[float(x) for x in line.split()[1:4]] for line in lines[2:2 + natoms]])
    return coords * Data.Angs2BohrFactor


def guess_from_checkpoints(mol, basis, checkpoints, ncores=None, **history_options):
    """
    Starting density for ``(mol, basis)`` from converged checkpoints of earlier geometries.

    This is what the ASE calculator uses between optimization steps, where each step may run in its
    own process and only files are passed on. The bases of the earlier geometries are the basis of
    the new one moved to the old atomic positions (:func:`displaced_basis`), so the atoms, their
    order and the basis set must be the same.

    Parameters
    ----------
    mol, basis : Mol, Basis
        The new geometry.
    checkpoints : sequence of (xyz_path, dmat_path)
        Structure and converged CAO density matrix (``.npy``) of earlier geometries, oldest first.
    ncores : int, optional
        BLAS thread cap for the guess, see :meth:`DensityHistory.guess`.
    **history_options
        Passed to :class:`DensityHistory` (``method``, ``max_points``, ``weights``, ...).

    Returns
    -------
    dmat : ndarray or None
    history : DensityHistory
        Its ``last_info`` describes the guess.
    """
    history = DensityHistory(**history_options)
    for xyz_path, dmat_path in list(checkpoints)[-history.max_points:]:
        centres = read_xyz_coordinates(xyz_path)
        if centres.shape[0] != mol.natoms:
            raise ValueError("checkpoint '" + str(xyz_path) + "' has a different number of atoms")
        density = np.load(dmat_path, allow_pickle=False)
        if density.shape != (basis.bfs_nao, basis.bfs_nao) or not np.all(np.isfinite(density)):
            raise ValueError("checkpoint '" + str(dmat_path) + "' does not match the basis or is not finite")
        history.append(displaced_basis(basis, centres), density)
    return history.guess(basis, mol.nelectrons, ncores=ncores), history


def describe_guess(info):
    """One-line description of :attr:`DensityHistory.last_info` for the output."""
    if not info:
        return "no earlier density available"
    labels = {"previous": "converged density of the previous geometry (unchanged matrix)",
              "transfer": "converged density of the previous geometry, carried with the atoms",
              "project": "converged density of the previous geometry, projected into the new basis",
              "extrapolate": "extrapolated from the converged densities of the previous %d geometries"
                             % info["npoints"]}
    text = labels[info["method"]]
    if info["method"] == "extrapolate" and info["npoints"] > 1:
        text += " (weights " + ", ".join("%.3f" % w for w in info["weights"]) + ")"
    if info.get("implementation") == "original" and info["method"] != "previous":
        text += ", original implementation"
    return text


def _basis_centres(basis):
    """Atomic positions (bohr) as seen by the basis functions, one row per atom."""
    atoms = np.asarray(basis.bfs_atoms)
    coords = np.asarray(basis.bfs_coords, dtype=np.float64)
    natoms = int(atoms.max()) + 1
    centres = np.zeros((natoms, 3))
    centres[atoms] = coords
    return centres
