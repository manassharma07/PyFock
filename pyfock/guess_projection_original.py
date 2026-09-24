"""AO density projection for geometry-following SCF calculations.

The projection is intended as an SCF initial guess only.  The electronic
integrals must still be rebuilt for the new geometry.

Original implementation by Prof. Vincenzo Barone, contributed to PyFock and added by Manas Sharma
with his permission (September 2026). This module keeps the code as contributed, so that results
and timings can always be compared with it; the only changes make it run on Python 3.9 (postponed
evaluation of annotations, and an explicit length check instead of ``zip(..., strict=True)``).
The optimized and extended version used by PyFock is :mod:`pyfock.guess_projection`; select this
reference implementation there with ``implementation='original'``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DensityProjectionDiagnostics:
    """Checks recorded while projecting a density into a new AO basis."""

    electron_count_before: float
    electron_count_after: float
    density_trace: float
    condition_number: float
    scale_factor: float


def extrapolate_density(
    densities: tuple[np.ndarray, ...] | list[np.ndarray],
    *,
    electron_count: float | None = None,
    overlap: np.ndarray | None = None,
    damping: float = 1.0,
) -> np.ndarray:
    """Extrapolate up to several densities in a common AO basis.

    The input densities must already have been projected into the same current
    AO basis. With three or more points, a least-squares quadratic fit is
    evaluated one step beyond the latest point; two points use linear
    extrapolation. This uses the full history without amplifying SCF noise as
    a high-order interpolating polynomial would. Callers should use damping
    for histories longer than two.
    """
    if len(densities) < 2:
        raise ValueError("at least two densities are required")
    if not 0.0 < damping <= 1.0:
        raise ValueError("damping must be in (0, 1]")
    latest = np.asarray(densities[-1], dtype=np.float64)
    if latest.ndim != 2 or latest.shape[0] != latest.shape[1]:
        raise ValueError("densities must be square matrices of equal dimensions")
    history = [np.asarray(item, dtype=np.float64) for item in densities]
    if any(item.shape != latest.shape for item in history):
        raise ValueError("densities must be square matrices of equal dimensions")
    # Fit a line for two points and a quadratic for longer histories.  The
    # coefficients map the matrices directly to the predicted next point.
    k = len(history)
    times = np.arange(k, dtype=np.float64)
    target_time = float(k)
    degree = min(k - 1, 2)
    design = np.vander(times, N=degree + 1, increasing=True)
    target_vector = np.array([target_time**power for power in range(degree + 1)])
    coefficients = np.linalg.lstsq(design.T, target_vector, rcond=None)[0]
    if len(coefficients) != len(history):
        raise ValueError("zip() arguments have different lengths")
    extrapolated = sum(coefficient * item for coefficient, item in zip(coefficients, history))
    predicted = latest + damping * (extrapolated - latest)
    predicted = 0.5 * (predicted + predicted.T)
    metric = None if overlap is None else np.asarray(overlap, dtype=np.float64)
    if metric is not None:
        if metric.shape != predicted.shape or not np.all(np.isfinite(metric)):
            raise ValueError("overlap has incompatible dimensions")
        # Keep the guess N-representable in the new AO metric. This is
        # especially important after multi-point extrapolation, which can
        # otherwise create negative natural occupations.
        metric = 0.5 * (metric + metric.T)
        metric_sqrt = _symmetric_matrix_power(metric, 0.5, 1.0e-12)
        metric_inv_sqrt = _symmetric_matrix_power(metric, -0.5, 1.0e-12)
        occupations, orbitals = np.linalg.eigh(metric_sqrt @ predicted @ metric_sqrt)
        occupations = np.clip(occupations, 0.0, 2.0)
        predicted = metric_inv_sqrt @ ((orbitals * occupations) @ orbitals.T) @ metric_inv_sqrt
        predicted = 0.5 * (predicted + predicted.T)
    if electron_count is not None:
        target = float(electron_count)
        if not np.isfinite(target) or target <= 0.0:
            raise ValueError("electron_count must be positive and finite")
        metric = np.eye(predicted.shape[0]) if metric is None else metric
        metric_trace = float(np.trace(predicted @ metric))
        if metric_trace <= np.finfo(np.float64).tiny:
            raise ValueError("extrapolated density has zero trace")
        predicted *= target / metric_trace
    return predicted


def project_density_between_bases(
    density_old: np.ndarray,
    basis_old: object,
    basis_new: object,
    *,
    electron_count: float | None = None,
    rcond: float = 1.0e-12,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, DensityProjectionDiagnostics]:
    """Project a density using two PyFock ``Basis`` objects directly."""
    from .Integrals import cross_overlap_mat_symm, overlap_mat_symm

    overlap_old_old = overlap_mat_symm(basis_old)
    overlap_new_old = cross_overlap_mat_symm(basis_new, basis_old)
    overlap_new_new = overlap_mat_symm(basis_new)
    return project_density(
        density_old,
        overlap_new_old,
        overlap_new_new,
        overlap_old_old=overlap_old_old,
        electron_count=electron_count,
        rcond=rcond,
        return_diagnostics=return_diagnostics,
    )


def project_operator_between_bases(
    operator_old: np.ndarray,
    basis_old: object,
    basis_new: object,
    *,
    rcond: float = 1.0e-12,
) -> np.ndarray:
    """Project a covariant AO operator into a new basis.

    This is intended for Fock/Kohn–Sham matrices used to seed PyFock's DIIS.
    Unlike a density, the operator is not occupation-filtered or electron-count
    normalized.
    """
    from .Integrals import cross_overlap_mat_symm, overlap_mat_symm

    old = np.asarray(operator_old, dtype=np.float64)
    overlap_new = overlap_mat_symm(basis_new)
    cross = cross_overlap_mat_symm(basis_new, basis_old)
    if old.ndim != 2 or old.shape[0] != old.shape[1]:
        raise ValueError("operator_old must be a square matrix")
    # For a covariant operator matrix, the old basis is approximately
    # ``chi_old = chi_new @ T``. Therefore ``F_old = T.T @ F_new @ T`` and
    # the least-squares back transformation is ``T+ .T @ F_old @ T+``.
    coefficient_map = np.linalg.pinv(overlap_new, rcond=rcond) @ cross
    inverse_map = np.linalg.pinv(coefficient_map, rcond=rcond)
    projected = inverse_map.T @ old @ inverse_map
    return 0.5 * (projected + projected.T)


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
    """Project an AO density from an old geometry into a new AO basis.

    Parameters
    ----------
    density_old
        Old AO density, in the old basis.
    overlap_new_old
        Cross overlap ``<chi_new|chi_old>``.
    overlap_new_new
        New-basis AO overlap ``<chi_new|chi_new>``.
    electron_count
        If supplied, rescale the projected density so that
        ``trace(P_new @ S_new)`` equals this value.
    rcond
        Relative cutoff used for the overlap pseudoinverse.
    return_diagnostics
        Also return electron-count and conditioning diagnostics.

    Notes
    -----
    This is a conservative density projection for SCF restarting.  It does
    not claim that the projected density is an idempotent ground-state
    density; PyFock must run a fresh SCF at the new geometry.
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

    new_overlap = 0.5 * (new_overlap + new_overlap.T)
    transform = np.linalg.pinv(new_overlap, rcond=rcond) @ cross
    if overlap_old_old is None:
        projected = transform @ old @ transform.T
    else:
        old_overlap = np.asarray(overlap_old_old, dtype=np.float64)
        if old_overlap.shape != old.shape:
            raise ValueError("overlap_old_old has incompatible dimensions")
        old_overlap = 0.5 * (old_overlap + old_overlap.T)
        # Recover the occupied natural-orbital subspace in the old metric,
        # project the orbitals, then re-orthonormalize in the new metric.
        old_sqrt = _symmetric_matrix_power(old_overlap, 0.5, rcond)
        old_inv_sqrt = _symmetric_matrix_power(old_overlap, -0.5, rcond)
        occupations, natural_orbitals = np.linalg.eigh(old_sqrt @ old @ old_sqrt)
        keep = occupations > max(10.0 * rcond, 1.0e-8)
        occupations = np.clip(occupations[keep], 0.0, 2.0)
        coefficients_old = old_inv_sqrt @ natural_orbitals[:, keep]
        coefficients_new = transform @ coefficients_old
        metric = coefficients_new.T @ new_overlap @ coefficients_new
        metric_inv_sqrt = _symmetric_matrix_power(metric, -0.5, rcond)
        coefficients_new = coefficients_new @ metric_inv_sqrt
        projected = (coefficients_new * occupations) @ coefficients_new.T
    projected = 0.5 * (projected + projected.T)
    # This is the metric trace of the unscaled projected density.  The old
    # electron count is intentionally not inferred: callers should provide
    # the molecule's authoritative electron count when restarting SCF.
    raw_count = float(np.trace(projected @ new_overlap))
    electron_before = raw_count
    scale = 1.0
    if electron_count is not None:
        target = float(electron_count)
        if not np.isfinite(target) or target <= 0.0:
            raise ValueError("electron_count must be positive and finite")
        if raw_count <= np.finfo(np.float64).tiny:
            raise ValueError("projected density has zero electron metric trace")
        scale = target / raw_count
        projected *= scale
    projected_count = float(np.trace(projected @ new_overlap))
    diagnostics = DensityProjectionDiagnostics(
        electron_count_before=electron_before,
        electron_count_after=projected_count,
        density_trace=float(np.trace(projected)),
        condition_number=float(np.linalg.cond(new_overlap)),
        scale_factor=scale,
    )
    return (projected, diagnostics) if return_diagnostics else projected


def _symmetric_matrix_power(matrix: np.ndarray, power: float, rcond: float) -> np.ndarray:
    """Apply a real power to a symmetric positive-semidefinite matrix."""
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    cutoff = max(float(np.max(np.abs(values))), 1.0) * rcond
    if np.any(values < -cutoff):
        raise ValueError("overlap metric is not positive semidefinite")
    values = np.maximum(values, cutoff)
    return (vectors * values**power) @ vectors.T
