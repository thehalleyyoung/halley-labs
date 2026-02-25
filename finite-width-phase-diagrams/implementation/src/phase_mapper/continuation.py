"""Pseudo-arclength continuation for tracking phase boundary curves.

Provides:
  - ContinuationConfig: tuning knobs for the continuation algorithm
  - ContinuationPoint: state at a single point along a branch
  - BranchInfo: an entire traced branch
  - ContinuationResult: all branches and bifurcation points
  - PseudoArclengthContinuation: predictor-corrector continuation engine
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import linalg as sp_linalg


# ======================================================================
# Data structures
# ======================================================================


@dataclass
class ContinuationConfig:
    """Parameters governing the continuation algorithm.

    Parameters
    ----------
    initial_step_size : float
        Starting arclength step size.
    min_step : float
        Minimum allowed step size before giving up.
    max_step : float
        Maximum allowed step size.
    max_steps : int
        Maximum number of continuation steps per branch.
    tolerance : float
        Newton corrector convergence tolerance.
    max_corrections : int
        Maximum Newton iterations per corrector step.
    eigenvalue_gap_threshold : float
        Threshold on eigenvalue gap for bifurcation detection.
    """

    initial_step_size: float = 0.01
    min_step: float = 1e-6
    max_step: float = 0.5
    max_steps: int = 2000
    tolerance: float = 1e-8
    max_corrections: int = 20
    eigenvalue_gap_threshold: float = 1e-4


@dataclass
class ContinuationPoint:
    """State at a single point along a continuation branch.

    Parameters
    ----------
    parameter : np.ndarray
        Parameter vector (2-D point on the boundary).
    tangent_vector : np.ndarray
        Unit tangent to the branch at this point.
    eigenvalue_gap : float
        Gap between the two smallest singular values of the Jacobian
        (small values suggest proximity to a bifurcation).
    step_size : float
        Arclength step size used to reach this point.
    arc_length : float
        Cumulative arclength along the branch.
    converged : bool
        Whether the Newton corrector converged at this point.
    """

    parameter: np.ndarray = field(default_factory=lambda: np.zeros(2))
    tangent_vector: np.ndarray = field(default_factory=lambda: np.zeros(2))
    eigenvalue_gap: float = 0.0
    step_size: float = 0.0
    arc_length: float = 0.0
    converged: bool = True


@dataclass
class BranchInfo:
    """A fully traced branch of the phase boundary.

    Parameters
    ----------
    points : list of ContinuationPoint
        Ordered sequence of points along the branch.
    branch_id : str
        Unique identifier for this branch.
    parent_branch : str or None
        ID of the branch from which this one bifurcated, if any.
    """

    points: List[ContinuationPoint] = field(default_factory=list)
    branch_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    parent_branch: Optional[str] = None

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def parameters_array(self) -> np.ndarray:
        """Return parameter vectors as a 2-D array.

        Returns
        -------
        np.ndarray
            Shape ``(n_points, dim)``.
        """
        return np.array([p.parameter for p in self.points])

    def arc_lengths(self) -> np.ndarray:
        """Return cumulative arclengths.

        Returns
        -------
        np.ndarray
            Shape ``(n_points,)``.
        """
        return np.array([p.arc_length for p in self.points])


@dataclass
class ContinuationResult:
    """Complete result of a continuation run.

    Parameters
    ----------
    branches : list of BranchInfo
        All traced branches.
    bifurcation_points : list of ContinuationPoint
        Detected bifurcation points.
    total_arc_length : float
        Sum of arclengths across all branches.
    """

    branches: List[BranchInfo] = field(default_factory=list)
    bifurcation_points: List[ContinuationPoint] = field(default_factory=list)
    total_arc_length: float = 0.0


# ======================================================================
# Pseudo-arclength continuation
# ======================================================================


class PseudoArclengthContinuation:
    """Predictor-corrector continuation engine for phase boundaries.

    The algorithm traces curves defined implicitly by
    ``boundary_fn(params) = 0`` using pseudo-arclength parameterisation,
    automatically adapting step size and detecting bifurcation / fold
    points.

    Parameters
    ----------
    boundary_fn : callable
        ``(params: np.ndarray) -> np.ndarray`` returning the residual
        vector.  Must be zero on the boundary.  ``params`` has shape
        ``(n,)`` and the output has shape ``(n - 1,)`` for a 1-D
        manifold.
    config : ContinuationConfig
        Algorithm tuning parameters.
    """

    def __init__(
        self,
        boundary_fn: Callable[[np.ndarray], np.ndarray],
        config: Optional[ContinuationConfig] = None,
    ) -> None:
        self.boundary_fn = boundary_fn
        self.config = config or ContinuationConfig()
        self._jac_eps = 1e-7

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, start_point: np.ndarray) -> ContinuationResult:
        """Run the full continuation from *start_point*.

        Traces the primary branch, detects bifurcations, and optionally
        switches to secondary branches.

        Parameters
        ----------
        start_point : np.ndarray
            Initial parameter vector on the boundary.

        Returns
        -------
        ContinuationResult
        """
        start_point = np.asarray(start_point, dtype=np.float64)
        tangent = self._compute_tangent(start_point)

        primary = self.trace_branch(start_point, tangent)
        result = ContinuationResult(
            branches=[primary],
            total_arc_length=primary.points[-1].arc_length if primary.points else 0.0,
        )

        # Scan for bifurcations and attempt branch switching
        for i in range(len(primary.points) - 1):
            p1, p2 = primary.points[i], primary.points[i + 1]
            bif = self.detect_bifurcation(p1, p2)
            if bif is not None:
                result.bifurcation_points.append(bif)
                new_tangent = self.switch_branch(bif)
                if new_tangent is not None:
                    secondary = self.trace_branch(
                        bif.parameter, new_tangent
                    )
                    secondary.parent_branch = primary.branch_id
                    result.branches.append(secondary)
                    result.total_arc_length += (
                        secondary.points[-1].arc_length
                        if secondary.points
                        else 0.0
                    )

        return result

    def trace_branch(
        self,
        start_point: np.ndarray,
        initial_tangent: np.ndarray,
    ) -> BranchInfo:
        """Trace a single branch starting from *start_point*.

        Parameters
        ----------
        start_point : np.ndarray
            Parameter vector on the boundary.
        initial_tangent : np.ndarray
            Initial unit tangent direction.

        Returns
        -------
        BranchInfo
        """
        cfg = self.config
        start_point = np.asarray(start_point, dtype=np.float64)
        tangent = np.asarray(initial_tangent, dtype=np.float64)
        tangent = tangent / (np.linalg.norm(tangent) + 1e-30)

        branch = BranchInfo()
        current = ContinuationPoint(
            parameter=start_point.copy(),
            tangent_vector=tangent.copy(),
            eigenvalue_gap=self._eigenvalue_gap(start_point),
            step_size=cfg.initial_step_size,
            arc_length=0.0,
            converged=True,
        )
        branch.points.append(current)

        step_size = cfg.initial_step_size
        for _ in range(cfg.max_steps):
            predicted = self._predictor_step(
                current.parameter, tangent, step_size
            )
            corrected = self._corrector_step(predicted, tangent, current)

            if not corrected.converged:
                step_size *= 0.5
                if step_size < cfg.min_step:
                    break
                continue

            corrected.arc_length = current.arc_length + step_size
            corrected.step_size = step_size

            new_tangent = self._compute_tangent(corrected.parameter)
            # Keep consistent orientation
            if np.dot(new_tangent, tangent) < 0:
                new_tangent = -new_tangent
            corrected.tangent_vector = new_tangent
            corrected.eigenvalue_gap = self._eigenvalue_gap(
                corrected.parameter
            )

            branch.points.append(corrected)
            tangent = new_tangent
            step_size = self._adapt_step_size(corrected, current)
            current = corrected

        return branch

    # ------------------------------------------------------------------
    # Bifurcation / fold detection
    # ------------------------------------------------------------------

    def detect_bifurcation(
        self,
        p1: ContinuationPoint,
        p2: ContinuationPoint,
    ) -> Optional[ContinuationPoint]:
        """Detect a bifurcation between two consecutive continuation points.

        A bifurcation is indicated by a sign change in the determinant of
        the Jacobian.

        Parameters
        ----------
        p1 : ContinuationPoint
        p2 : ContinuationPoint

        Returns
        -------
        ContinuationPoint or None
            Approximate bifurcation point, or ``None`` if none detected.
        """
        J1 = self._jacobian(p1.parameter)
        J2 = self._jacobian(p2.parameter)
        det1 = np.linalg.det(J1 @ J1.T)
        det2 = np.linalg.det(J2 @ J2.T)

        if det1 * det2 >= 0:
            return None

        # Linear interpolation to approximate the crossing
        alpha = abs(det1) / (abs(det1) + abs(det2) + 1e-30)
        bif_param = (1 - alpha) * p1.parameter + alpha * p2.parameter
        tangent = self._compute_tangent(bif_param)
        return ContinuationPoint(
            parameter=bif_param,
            tangent_vector=tangent,
            eigenvalue_gap=self._eigenvalue_gap(bif_param),
            step_size=0.0,
            arc_length=0.5 * (p1.arc_length + p2.arc_length),
            converged=True,
        )

    def detect_fold(
        self,
        p1: ContinuationPoint,
        p2: ContinuationPoint,
    ) -> Optional[ContinuationPoint]:
        """Detect a fold (limit) point between two consecutive points.

        A fold is indicated by a flip in the tangent vector direction
        along any coordinate axis.

        Parameters
        ----------
        p1 : ContinuationPoint
        p2 : ContinuationPoint

        Returns
        -------
        ContinuationPoint or None
        """
        # Check for sign change in the first component of the tangent
        for dim in range(len(p1.tangent_vector)):
            if p1.tangent_vector[dim] * p2.tangent_vector[dim] < 0:
                alpha = abs(p1.tangent_vector[dim]) / (
                    abs(p1.tangent_vector[dim])
                    + abs(p2.tangent_vector[dim])
                    + 1e-30
                )
                fold_param = (1 - alpha) * p1.parameter + alpha * p2.parameter
                tangent = self._compute_tangent(fold_param)
                return ContinuationPoint(
                    parameter=fold_param,
                    tangent_vector=tangent,
                    eigenvalue_gap=self._eigenvalue_gap(fold_param),
                    step_size=0.0,
                    arc_length=0.5 * (p1.arc_length + p2.arc_length),
                    converged=True,
                )
        return None

    def switch_branch(
        self,
        bifurcation_point: ContinuationPoint,
    ) -> Optional[np.ndarray]:
        """Compute an initial tangent for branch switching at a bifurcation.

        The new tangent is chosen from the null space of the Jacobian,
        orthogonal to the incoming tangent.

        Parameters
        ----------
        bifurcation_point : ContinuationPoint

        Returns
        -------
        np.ndarray or None
            New initial tangent, or ``None`` if branch switching is not
            possible.
        """
        J = self._jacobian(bifurcation_point.parameter)
        U, s, Vt = np.linalg.svd(J, full_matrices=True)
        n = len(bifurcation_point.parameter)
        null_dim = max(1, n - len(s[s > self.config.eigenvalue_gap_threshold]))

        if null_dim < 2:
            return None

        # The last rows of Vt span the null space
        null_vectors = Vt[-null_dim:]
        incoming = bifurcation_point.tangent_vector

        # Find the null-space vector most orthogonal to the incoming tangent
        best_idx = 0
        min_overlap = np.inf
        for i, v in enumerate(null_vectors):
            overlap = abs(np.dot(v, incoming))
            if overlap < min_overlap:
                min_overlap = overlap
                best_idx = i

        new_tangent = null_vectors[best_idx].copy()
        # Remove component along incoming tangent
        new_tangent -= np.dot(new_tangent, incoming) * incoming
        norm = np.linalg.norm(new_tangent)
        if norm < 1e-12:
            return None
        return new_tangent / norm

    # ------------------------------------------------------------------
    # Predictor / corrector
    # ------------------------------------------------------------------

    def _predictor_step(
        self,
        current: np.ndarray,
        tangent: np.ndarray,
        step_size: float,
    ) -> np.ndarray:
        """Euler predictor along the tangent direction.

        Parameters
        ----------
        current : np.ndarray
            Current parameter vector.
        tangent : np.ndarray
            Unit tangent vector.
        step_size : float
            Arclength step size.

        Returns
        -------
        np.ndarray
            Predicted parameter vector.
        """
        return current + step_size * tangent

    def _corrector_step(
        self,
        predicted: np.ndarray,
        tangent: np.ndarray,
        prev_point: ContinuationPoint,
    ) -> ContinuationPoint:
        """Newton corrector with pseudo-arclength constraint.

        The augmented system appends the arclength constraint
        ``tangent^T (u - prev) - ds = 0`` to the original residual.

        Parameters
        ----------
        predicted : np.ndarray
            Predicted parameter vector from the predictor step.
        tangent : np.ndarray
            Tangent vector used in the prediction.
        prev_point : ContinuationPoint
            Previous converged point.

        Returns
        -------
        ContinuationPoint
            Corrected point (check ``converged`` flag).
        """
        cfg = self.config
        u = predicted.copy()
        ds = np.linalg.norm(predicted - prev_point.parameter)

        for _ in range(cfg.max_corrections):
            F = np.atleast_1d(self.boundary_fn(u))
            arc_constraint = np.dot(tangent, u - prev_point.parameter) - ds
            residual = np.concatenate([F, [arc_constraint]])

            if np.linalg.norm(residual) < cfg.tolerance:
                return ContinuationPoint(
                    parameter=u.copy(), converged=True,
                )

            J = self._jacobian(u)
            # Augmented Jacobian: stack tangent^T row below J
            J_aug = np.vstack([J, tangent.reshape(1, -1)])
            try:
                delta = np.linalg.lstsq(J_aug, -residual, rcond=None)[0]
            except np.linalg.LinAlgError:
                return ContinuationPoint(parameter=u.copy(), converged=False)
            u += delta

        return ContinuationPoint(parameter=u.copy(), converged=False)

    # ------------------------------------------------------------------
    # Tangent computation
    # ------------------------------------------------------------------

    def _compute_tangent(self, point: np.ndarray) -> np.ndarray:
        """Compute the unit tangent via the null space of the Jacobian.

        Parameters
        ----------
        point : np.ndarray
            Parameter vector.

        Returns
        -------
        np.ndarray
            Unit tangent vector.
        """
        J = self._jacobian(point)
        _, s, Vt = np.linalg.svd(J, full_matrices=True)
        # Tangent is the last row of Vt (smallest singular value direction)
        tangent = Vt[-1]
        norm = np.linalg.norm(tangent)
        if norm < 1e-30:
            tangent = np.zeros_like(point)
            tangent[0] = 1.0
            return tangent
        return tangent / norm

    # ------------------------------------------------------------------
    # Step-size adaptation
    # ------------------------------------------------------------------

    def _adapt_step_size(
        self,
        current: ContinuationPoint,
        prev: ContinuationPoint,
    ) -> float:
        """Adapt the step size based on eigenvalue gap and angle change.

        Parameters
        ----------
        current : ContinuationPoint
        prev : ContinuationPoint

        Returns
        -------
        float
            New step size clamped to ``[min_step, max_step]``.
        """
        cfg = self.config

        # Angle change between tangents
        cos_angle = np.clip(
            np.dot(current.tangent_vector, prev.tangent_vector), -1.0, 1.0
        )
        angle = np.arccos(abs(cos_angle))

        # Target: keep angle change around 0.1 rad
        if angle > 1e-12:
            scale = 0.1 / angle
        else:
            scale = 2.0
        scale = np.clip(scale, 0.25, 4.0)

        # Shrink near bifurcations
        if current.eigenvalue_gap < cfg.eigenvalue_gap_threshold:
            scale *= 0.5

        new_step = current.step_size * scale
        return float(np.clip(new_step, cfg.min_step, cfg.max_step))

    # ------------------------------------------------------------------
    # Jacobian helpers
    # ------------------------------------------------------------------

    def _jacobian(self, point: np.ndarray) -> np.ndarray:
        """Finite-difference Jacobian of boundary_fn at *point*.

        Parameters
        ----------
        point : np.ndarray
            Parameter vector.

        Returns
        -------
        np.ndarray
            Jacobian matrix of shape ``(m, n)`` where ``m`` is the
            residual dimension and ``n = len(point)``.
        """
        eps = self._jac_eps
        f0 = np.atleast_1d(self.boundary_fn(point))
        n = len(point)
        m = len(f0)
        J = np.empty((m, n), dtype=np.float64)
        for i in range(n):
            p_plus = point.copy()
            p_plus[i] += eps
            J[:, i] = (np.atleast_1d(self.boundary_fn(p_plus)) - f0) / eps
        return J

    def _eigenvalue_gap(self, point: np.ndarray) -> float:
        """Compute eigenvalue gap at *point*.

        The gap is the difference between the two smallest singular
        values of the Jacobian.

        Parameters
        ----------
        point : np.ndarray

        Returns
        -------
        float
        """
        J = self._jacobian(point)
        s = np.linalg.svd(J, compute_uv=False)
        if len(s) < 2:
            return float(s[-1]) if len(s) > 0 else 0.0
        return float(s[-2] - s[-1])
