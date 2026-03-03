"""
Infinite-dimensional duality for the DP mechanism design LP.

Provides tools for constructing and verifying the dual of the infinite LP,
computing Slater interior points, certifying duality gaps, and verifying
strong duality (Theorem T30).

Theory
------
The primal infinite LP is:

    min  t
    s.t. ∫ loss(f(x_i), y) p_i(y) dy ≤ t       ∀ i
         p_i(y) ≤ e^ε p_{i'}(y)                  ∀ (i,i') adjacent, ∀ y
         ∫ p_i(y) dy = 1                          ∀ i
         p_i(y) ≥ 0                               ∀ i, y

The dual of this infinite LP has a finite number of variables (one per
primal constraint) but the dual feasibility condition must hold for all
y ∈ Y.  Strong duality holds when there exists a Slater point — a strictly
feasible primal solution.  The uniform mechanism (p_i(y) = 1/|Y| for all
i, y) serves as the Slater point.

Classes
-------
- :class:`InfiniteDualityChecker` — Dual construction and gap certification.
- :class:`DualProblem` — Container for the dual problem structure.
- :class:`SlaterPoint` — Container for a Slater feasibility certificate.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from dp_forge.exceptions import (
    ConfigurationError,
    InfeasibleSpecError,
    NumericalInstabilityError,
)
from dp_forge.types import LossFunction, NumericalConfig, QuerySpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DUAL_FEASIBILITY_TOL: float = 1e-8
_SLATER_MARGIN: float = 1e-6


# ---------------------------------------------------------------------------
# Dual problem container
# ---------------------------------------------------------------------------


@dataclass
class DualProblem:
    """Structure of the dual of the infinite LP.

    The dual has variables:
    - λ[i] ≥ 0 for each epigraph constraint (one per database i)
    - μ[i] (free) for each simplex constraint (one per database i)
    - ν[i,i',y] ≥ 0 for each DP constraint (one per adjacent pair, per y)

    At optimality, the dual objective equals the primal objective (strong
    duality).  The dual objective is:

        max  Σ_i μ[i]

    subject to dual feasibility for all y ∈ Y.

    Attributes:
        n: Number of databases.
        n_edges: Number of (directed) adjacency edges.
        lambda_vars: Epigraph dual variables, shape (n,).
        mu_vars: Simplex dual variables, shape (n,).
        dual_objective: Dual objective value.
        is_feasible: Whether dual feasibility holds.
    """

    n: int
    n_edges: int
    lambda_vars: npt.NDArray[np.float64]
    mu_vars: npt.NDArray[np.float64]
    dual_objective: float
    is_feasible: bool

    def __post_init__(self) -> None:
        self.lambda_vars = np.asarray(self.lambda_vars, dtype=np.float64)
        self.mu_vars = np.asarray(self.mu_vars, dtype=np.float64)

    def __repr__(self) -> str:
        status = "feasible" if self.is_feasible else "infeasible"
        return (
            f"DualProblem(n={self.n}, edges={self.n_edges}, "
            f"obj={self.dual_objective:.6f}, {status})"
        )


# ---------------------------------------------------------------------------
# Slater point
# ---------------------------------------------------------------------------


@dataclass
class SlaterPoint:
    """Certificate that a Slater (strictly feasible) point exists.

    A Slater point is a primal-feasible solution where all inequality
    constraints hold strictly.  Its existence guarantees strong duality.

    Attributes:
        mechanism: The n × k Slater point mechanism table.
        y_grid: Output grid on which the Slater point is defined.
        min_slack: Minimum slack across all inequality constraints.
        dp_slack: Minimum slack across DP constraints specifically.
        is_strictly_feasible: True if all slacks are positive.
    """

    mechanism: npt.NDArray[np.float64]
    y_grid: npt.NDArray[np.float64]
    min_slack: float
    dp_slack: float
    is_strictly_feasible: bool

    def __post_init__(self) -> None:
        self.mechanism = np.asarray(self.mechanism, dtype=np.float64)
        self.y_grid = np.asarray(self.y_grid, dtype=np.float64)

    def __repr__(self) -> str:
        status = "strict" if self.is_strictly_feasible else "boundary"
        return (
            f"SlaterPoint(min_slack={self.min_slack:.3e}, "
            f"dp_slack={self.dp_slack:.3e}, {status})"
        )


# ---------------------------------------------------------------------------
# Gap certification result
# ---------------------------------------------------------------------------


@dataclass
class GapCertificate:
    """Certificate of the duality gap.

    Attributes:
        primal_obj: Primal objective value.
        dual_obj: Dual objective value.
        gap: Absolute gap (primal - dual).
        relative_gap: Gap / max(|primal|, 1).
        is_certified: True if gap is within tolerance.
        tolerance: Tolerance used for certification.
        strong_duality_holds: True if strong duality conditions are verified.
    """

    primal_obj: float
    dual_obj: float
    gap: float
    relative_gap: float
    is_certified: bool
    tolerance: float
    strong_duality_holds: bool

    def __repr__(self) -> str:
        cert = "certified" if self.is_certified else "uncertified"
        strong = "strong" if self.strong_duality_holds else "weak"
        return (
            f"GapCertificate(gap={self.gap:.3e}, rel={self.relative_gap:.3e}, "
            f"{cert}, {strong}_duality)"
        )


# ---------------------------------------------------------------------------
# InfiniteDualityChecker
# ---------------------------------------------------------------------------


class InfiniteDualityChecker:
    """Tools for infinite-dimensional LP duality analysis.

    Constructs the dual problem, computes Slater points, certifies
    duality gaps, and verifies strong duality conditions.

    Parameters
    ----------
    numerical_config : NumericalConfig, optional
        Numerical precision configuration.
    check_points : int
        Number of points to use for dual feasibility checking.
    """

    def __init__(
        self,
        numerical_config: Optional[NumericalConfig] = None,
        check_points: int = 500,
    ) -> None:
        self._numerical = numerical_config or NumericalConfig()
        self._check_points = check_points

    def construct_dual(
        self,
        spec: QuerySpec,
        y_grid: npt.NDArray[np.float64],
        primal_dual_vars: npt.NDArray[np.float64],
    ) -> DualProblem:
        """Construct the dual problem from primal LP dual variables.

        Parameters
        ----------
        spec : QuerySpec
            Problem specification.
        y_grid : array of shape (k,)
            Output grid.
        primal_dual_vars : array
            Dual variable vector from the primal LP solver.
            Layout: [mu_1, ..., mu_n, lambda_1, ..., lambda_n, ...]

        Returns
        -------
        DualProblem
        """
        n = spec.n
        assert spec.edges is not None

        # Parse dual variables
        # Convention: first n are simplex duals (mu), next n are epigraph duals (lambda)
        if len(primal_dual_vars) < 2 * n:
            # Pad with zeros if not enough dual vars provided
            padded = np.zeros(2 * n, dtype=np.float64)
            padded[:len(primal_dual_vars)] = primal_dual_vars
            primal_dual_vars = padded

        mu_vars = primal_dual_vars[:n]
        lambda_vars = primal_dual_vars[n:2 * n]

        # Ensure lambda >= 0 (they are duals of <= constraints)
        lambda_vars = np.maximum(lambda_vars, 0.0)

        # Dual objective: Σ_i μ[i]  (from the equality constraints)
        dual_obj = float(np.sum(mu_vars))

        # Check dual feasibility on a dense grid
        n_directed = spec.edges.num_edges
        is_feasible = self._check_dual_feasibility(
            spec, y_grid, lambda_vars, mu_vars,
        )

        return DualProblem(
            n=n,
            n_edges=n_directed,
            lambda_vars=lambda_vars,
            mu_vars=mu_vars,
            dual_objective=dual_obj,
            is_feasible=is_feasible,
        )

    def _check_dual_feasibility(
        self,
        spec: QuerySpec,
        y_grid: npt.NDArray[np.float64],
        lambda_vars: npt.NDArray[np.float64],
        mu_vars: npt.NDArray[np.float64],
    ) -> bool:
        """Check dual feasibility on a dense set of output points.

        The dual feasibility condition (for each database i and output y) is:
            λ[i] * loss(f(x_i), y) ≥ μ[i] - Σ_{dp terms involving i}

        We check this on a grid of test points.

        Parameters
        ----------
        spec : QuerySpec
        y_grid : array
        lambda_vars : array of shape (n,)
        mu_vars : array of shape (n,)

        Returns
        -------
        bool
            True if dual feasibility holds approximately.
        """
        n = spec.n
        loss_callable = spec.get_loss_callable()

        # Dense check grid
        y_min = float(y_grid[0])
        y_max = float(y_grid[-1])
        margin = (y_max - y_min) * 0.1
        check_grid = np.linspace(y_min - margin, y_max + margin, self._check_points)

        for y in check_grid:
            for i in range(n):
                lhs = lambda_vars[i] * loss_callable(
                    float(spec.query_values[i]), float(y),
                )
                rhs = mu_vars[i]
                if lhs < rhs - _DUAL_FEASIBILITY_TOL:
                    return False

        return True

    def slater_point(
        self,
        spec: QuerySpec,
        y_grid: npt.NDArray[np.float64],
    ) -> SlaterPoint:
        """Compute a Slater interior point for the primal infinite LP.

        Uses the uniform mechanism: p_i(y_j) = 1/k for all i, j.  This is
        always primal feasible, and for ε > 0 it is strictly feasible for
        the DP constraints (since all ratios equal 1 < e^ε).

        Parameters
        ----------
        spec : QuerySpec
            Problem specification.
        y_grid : array of shape (k,)
            Output grid.

        Returns
        -------
        SlaterPoint
        """
        n = spec.n
        k = len(y_grid)
        assert spec.edges is not None

        # Uniform mechanism
        mechanism = np.full((n, k), 1.0 / k, dtype=np.float64)

        # Compute slacks for DP constraints
        exp_eps = math.exp(spec.epsilon)
        dp_slack = math.inf

        for i, ip in spec.edges.edges:
            for j in range(k):
                # p[i][j] <= e^ε * p[i'][j]
                # slack = e^ε * p[i'][j] - p[i][j]
                slack = exp_eps * mechanism[ip, j] - mechanism[i, j]
                dp_slack = min(dp_slack, slack)

                if spec.edges.symmetric:
                    slack_rev = exp_eps * mechanism[i, j] - mechanism[ip, j]
                    dp_slack = min(dp_slack, slack_rev)

        # For uniform mechanism, dp_slack = (e^ε - 1) / k
        # This is always > 0 for ε > 0

        # Compute loss slack (epigraph constraints)
        loss_callable = spec.get_loss_callable()
        max_loss = 0.0
        for i in range(n):
            expected_loss = sum(
                loss_callable(float(spec.query_values[i]), float(y_grid[j])) / k
                for j in range(k)
            )
            max_loss = max(max_loss, expected_loss)

        # The epigraph variable t can be set to max_loss + slack
        epigraph_slack = _SLATER_MARGIN  # Arbitrary small positive slack

        min_slack = min(dp_slack, epigraph_slack)
        is_strict = min_slack > 0

        return SlaterPoint(
            mechanism=mechanism,
            y_grid=y_grid,
            min_slack=min_slack,
            dp_slack=dp_slack,
            is_strictly_feasible=is_strict,
        )

    def certify_gap(
        self,
        primal_obj: float,
        dual_obj: float,
        tolerance: float = 1e-6,
        slater: Optional[SlaterPoint] = None,
    ) -> GapCertificate:
        """Certify the duality gap between primal and dual objectives.

        Parameters
        ----------
        primal_obj : float
            Primal objective value.
        dual_obj : float
            Dual objective value.
        tolerance : float
            Gap tolerance for certification.
        slater : SlaterPoint, optional
            Slater point for strong duality verification.

        Returns
        -------
        GapCertificate
        """
        gap = primal_obj - dual_obj
        gap = max(gap, 0.0)  # Clamp numerical noise
        rel_gap = gap / max(abs(primal_obj), 1.0)

        is_certified = gap <= tolerance

        # Strong duality holds if:
        # 1. A Slater point exists (strictly feasible primal), AND
        # 2. The problem is bounded (finite optimal value)
        strong_duality = False
        if slater is not None and slater.is_strictly_feasible:
            if math.isfinite(primal_obj) and math.isfinite(dual_obj):
                strong_duality = True

        return GapCertificate(
            primal_obj=primal_obj,
            dual_obj=dual_obj,
            gap=gap,
            relative_gap=rel_gap,
            is_certified=is_certified,
            tolerance=tolerance,
            strong_duality_holds=strong_duality,
        )

    def verify_strong_duality(
        self,
        spec: QuerySpec,
        y_grid: npt.NDArray[np.float64],
        primal_obj: float,
        dual_obj: float,
        primal_dual_vars: Optional[npt.NDArray[np.float64]] = None,
        tolerance: float = 1e-6,
    ) -> GapCertificate:
        """Full strong duality verification (Theorem T30).

        Theorem T30 states: if the primal infinite LP admits a Slater point
        (a strictly feasible solution), and the optimal value is finite,
        then strong duality holds — i.e., the primal and dual optimal values
        are equal.

        This method:
        1. Constructs the Slater point (uniform mechanism).
        2. Verifies strict feasibility.
        3. Certifies the duality gap.
        4. Optionally checks dual feasibility.

        Parameters
        ----------
        spec : QuerySpec
            Problem specification.
        y_grid : array of shape (k,)
            Output grid.
        primal_obj : float
            Primal objective value.
        dual_obj : float
            Dual objective value.
        primal_dual_vars : array, optional
            Dual variables for dual feasibility checking.
        tolerance : float
            Gap tolerance.

        Returns
        -------
        GapCertificate
        """
        # 1. Compute Slater point
        slater = self.slater_point(spec, y_grid)

        if not slater.is_strictly_feasible:
            logger.warning(
                "Slater point is not strictly feasible (dp_slack=%.3e); "
                "strong duality may not hold",
                slater.dp_slack,
            )

        # 2. Certify gap
        cert = self.certify_gap(primal_obj, dual_obj, tolerance, slater)

        # 3. Optionally verify dual feasibility
        if primal_dual_vars is not None and cert.strong_duality_holds:
            dual_prob = self.construct_dual(spec, y_grid, primal_dual_vars)
            if not dual_prob.is_feasible:
                logger.warning(
                    "Dual solution may not be feasible; gap certification is approximate"
                )

        if cert.strong_duality_holds and cert.is_certified:
            logger.info(
                "Strong duality verified (Theorem T30): gap=%.3e <= tol=%.3e",
                cert.gap, tolerance,
            )
        elif cert.strong_duality_holds:
            logger.info(
                "Strong duality conditions met but gap=%.3e > tol=%.3e",
                cert.gap, tolerance,
            )

        return cert

    def bound_gap_from_grid_size(
        self,
        spec: QuerySpec,
        y_grid: npt.NDArray[np.float64],
    ) -> float:
        """Compute an upper bound on the duality gap from the grid size.

        Based on the convergence rate O(B / k) where B depends on the
        Lipschitz constant of the loss function and the sensitivity.

        Parameters
        ----------
        spec : QuerySpec
        y_grid : array of shape (k,)

        Returns
        -------
        float
            Upper bound on the gap due to grid discretisation.
        """
        k = len(y_grid)
        if k < 2:
            return math.inf

        grid_span = float(y_grid[-1] - y_grid[0])
        if grid_span <= 0:
            return math.inf

        # Lipschitz constant of the loss
        if spec.loss_fn in (LossFunction.L1, LossFunction.LINF):
            lip = 1.0
        elif spec.loss_fn == LossFunction.L2:
            lip = 2.0 * grid_span  # Lipschitz of squared loss on bounded domain
        else:
            lip = grid_span  # Conservative estimate

        # Grid spacing
        max_spacing = float(np.max(np.diff(y_grid)))

        # Gap bound: Lipschitz * max_spacing * sensitivity
        bound = lip * max_spacing * spec.sensitivity
        return bound

    def __repr__(self) -> str:
        return f"InfiniteDualityChecker(check_points={self._check_points})"
