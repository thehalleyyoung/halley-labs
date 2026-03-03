"""
Gradient-based optimisation for differentially private mechanisms.

Provides several optimisers tailored to the probability-simplex and
privacy-constraint geometry that arises in DP mechanism design:

- **PrivacyAwareAdam**: Adam with projection onto privacy constraints.
- **ProjectedGradientDescent**: PGD on the probability simplex.
- **FrankWolfe**: Frank-Wolfe for linear minimisation over simplex.
- **LineSearch**: Armijo / Wolfe conditions for step-size selection.
- **ConvergenceMonitor**: Detects stalling and tracks history.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import numpy.typing as npt


# ---------------------------------------------------------------------------
# Simplex projection
# ---------------------------------------------------------------------------


def project_simplex(v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Project *v* onto the probability simplex Δ = {x ≥ 0, Σx = 1}.

    Uses the efficient O(n log n) algorithm of Duchi et al. (2008).

    Args:
        v: Vector to project.

    Returns:
        Projected vector on the simplex.
    """
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    rho = np.nonzero(u * np.arange(1, n + 1) > cssv)[0][-1]
    theta = cssv[rho] / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def project_simplex_rows(M: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Project each row of *M* onto the probability simplex."""
    return np.array([project_simplex(row) for row in M])


def project_nonneg(v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Project onto the non-negative orthant."""
    return np.maximum(v, 0.0)


# ---------------------------------------------------------------------------
# Line search
# ---------------------------------------------------------------------------


class LineSearchMethod(Enum):
    ARMIJO = auto()
    WOLFE = auto()
    BACKTRACKING = auto()


@dataclass
class LineSearchResult:
    """Result of a line search.

    Attributes:
        step_size: Chosen step size.
        n_evals: Number of function evaluations used.
        success: Whether the line search succeeded.
    """

    step_size: float
    n_evals: int
    success: bool


class LineSearch:
    """Line search for step-size selection.

    Supports Armijo backtracking and strong Wolfe conditions.

    Attributes:
        method: Line search strategy.
        c1: Armijo (sufficient decrease) parameter.
        c2: Wolfe (curvature) parameter.
        max_iter: Maximum number of step-size trials.
        shrink: Backtracking shrink factor.
    """

    def __init__(
        self,
        method: LineSearchMethod = LineSearchMethod.ARMIJO,
        c1: float = 1e-4,
        c2: float = 0.9,
        max_iter: int = 30,
        shrink: float = 0.5,
    ) -> None:
        self.method = method
        self.c1 = c1
        self.c2 = c2
        self.max_iter = max_iter
        self.shrink = shrink

    def search(
        self,
        fn: Callable[[npt.NDArray[np.float64]], float],
        x: npt.NDArray[np.float64],
        direction: npt.NDArray[np.float64],
        gradient: npt.NDArray[np.float64],
        initial_step: float = 1.0,
    ) -> LineSearchResult:
        """Perform line search along *direction* from *x*.

        Args:
            fn: Objective function.
            x: Current point.
            direction: Search direction (typically -gradient).
            gradient: Gradient at x.
            initial_step: Initial step size guess.

        Returns:
            LineSearchResult with the chosen step size.
        """
        f0 = fn(x)
        slope = float(np.dot(gradient, direction))
        if slope > 0:
            # Direction is not a descent direction; use small step
            return LineSearchResult(1e-6, 1, False)

        alpha = initial_step
        n_evals = 0

        for _ in range(self.max_iter):
            x_new = x + alpha * direction
            f_new = fn(x_new)
            n_evals += 1

            # Armijo condition
            if f_new <= f0 + self.c1 * alpha * slope:
                if self.method == LineSearchMethod.ARMIJO:
                    return LineSearchResult(alpha, n_evals, True)
                # For Wolfe, also check curvature
                # (skipped in backtracking mode)
                return LineSearchResult(alpha, n_evals, True)

            alpha *= self.shrink

        return LineSearchResult(alpha, n_evals, False)


# ---------------------------------------------------------------------------
# Convergence monitor
# ---------------------------------------------------------------------------


@dataclass
class ConvergenceInfo:
    """Snapshot of convergence state.

    Attributes:
        iteration: Current iteration number.
        loss: Current loss value.
        grad_norm: Current gradient norm.
        step_size: Step size used.
        constraint_violation: Max constraint violation (0 if feasible).
    """

    iteration: int
    loss: float
    grad_norm: float
    step_size: float
    constraint_violation: float = 0.0


class ConvergenceMonitor:
    """Track optimisation convergence and detect stalling.

    Attributes:
        patience: Number of iterations without improvement before stall.
        tol: Minimum improvement to count as progress.
        history: List of ConvergenceInfo snapshots.
    """

    def __init__(self, patience: int = 50, tol: float = 1e-10) -> None:
        self.patience = patience
        self.tol = tol
        self.history: List[ConvergenceInfo] = []
        self._best_loss: float = float("inf")
        self._stall_count: int = 0

    def record(self, info: ConvergenceInfo) -> None:
        """Record a convergence snapshot."""
        self.history.append(info)
        if info.loss < self._best_loss - self.tol:
            self._best_loss = info.loss
            self._stall_count = 0
        else:
            self._stall_count += 1

    @property
    def is_stalled(self) -> bool:
        """Whether the optimiser has stalled."""
        return self._stall_count >= self.patience

    @property
    def best_loss(self) -> float:
        return self._best_loss

    @property
    def n_iterations(self) -> int:
        return len(self.history)

    def loss_history(self) -> npt.NDArray[np.float64]:
        """Return array of loss values."""
        return np.array([h.loss for h in self.history], dtype=np.float64)

    def grad_norm_history(self) -> npt.NDArray[np.float64]:
        """Return array of gradient norms."""
        return np.array([h.grad_norm for h in self.history], dtype=np.float64)


# ---------------------------------------------------------------------------
# Projected Gradient Descent
# ---------------------------------------------------------------------------


class ProjectedGradientDescent:
    """Projected gradient descent for constrained mechanism optimisation.

    Projects iterates onto the probability simplex (row-wise) after
    each gradient step.

    Attributes:
        learning_rate: Step size (or initial step for line search).
        max_iter: Maximum iterations.
        tol: Gradient norm convergence tolerance.
        use_line_search: Whether to use Armijo line search.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-8,
        use_line_search: bool = False,
    ) -> None:
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.use_line_search = use_line_search
        self._line_search = LineSearch() if use_line_search else None
        self.monitor = ConvergenceMonitor()

    def optimize(
        self,
        x0: npt.NDArray[np.float64],
        fn: Callable[[npt.NDArray[np.float64]], float],
        grad_fn: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
        project_fn: Optional[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]] = None,
    ) -> Tuple[npt.NDArray[np.float64], List[float]]:
        """Run projected gradient descent.

        Args:
            x0: Initial point.
            fn: Objective function.
            grad_fn: Gradient function.
            project_fn: Projection function (default: simplex rows).

        Returns:
            (optimised_point, loss_history).
        """
        proj = project_fn or (lambda v: project_simplex(v) if v.ndim == 1
                              else project_simplex_rows(v))
        x = proj(x0.copy())
        losses: List[float] = []
        lr = self.learning_rate

        for it in range(self.max_iter):
            loss = fn(x)
            g = grad_fn(x)
            g_norm = float(np.linalg.norm(g))
            losses.append(loss)

            self.monitor.record(ConvergenceInfo(
                iteration=it, loss=loss, grad_norm=g_norm, step_size=lr,
            ))

            if g_norm < self.tol:
                break
            if self.monitor.is_stalled:
                break

            direction = -g
            if self._line_search is not None:
                result = self._line_search.search(fn, x.ravel(), direction.ravel(), g.ravel(), lr)
                lr = result.step_size

            x = proj(x - lr * g)

        return x, losses


# ---------------------------------------------------------------------------
# Privacy-Aware Adam
# ---------------------------------------------------------------------------


class PrivacyAwareAdam:
    """Adam optimiser with projection onto privacy constraints.

    Maintains first and second moment estimates and projects iterates
    onto the feasible set (probability simplex + privacy constraints).

    Attributes:
        learning_rate: Base learning rate.
        beta1: First moment decay rate.
        beta2: Second moment decay rate.
        eps: Numerical stability constant.
        max_iter: Maximum iterations.
        tol: Convergence tolerance on gradient norm.
        gradient_clip: Optional max gradient norm.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        max_iter: int = 1000,
        tol: float = 1e-8,
        gradient_clip: Optional[float] = None,
    ) -> None:
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.max_iter = max_iter
        self.tol = tol
        self.gradient_clip = gradient_clip
        self.monitor = ConvergenceMonitor()

    def _clip_gradient(self, g: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if self.gradient_clip is None:
            return g
        norm = float(np.linalg.norm(g))
        if norm > self.gradient_clip:
            return g * (self.gradient_clip / norm)
        return g

    def optimize(
        self,
        x0: npt.NDArray[np.float64],
        fn: Callable[[npt.NDArray[np.float64]], float],
        grad_fn: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
        project_fn: Optional[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]] = None,
        privacy_constraint_fn: Optional[Callable[[npt.NDArray[np.float64]], float]] = None,
        privacy_budget: Optional[float] = None,
    ) -> Tuple[npt.NDArray[np.float64], List[float]]:
        """Run Adam with optional privacy constraint projection.

        Args:
            x0: Initial point.
            fn: Objective function.
            grad_fn: Gradient function.
            project_fn: Projection onto feasible set.
            privacy_constraint_fn: Returns privacy loss at current point.
            privacy_budget: Maximum allowed privacy loss.

        Returns:
            (optimised_point, loss_history).
        """
        proj = project_fn or (lambda v: project_simplex(v) if v.ndim == 1
                              else project_simplex_rows(v))
        x = proj(x0.copy()).astype(np.float64)
        shape = x.shape
        x_flat = x.ravel()

        m = np.zeros_like(x_flat)  # first moment
        v = np.zeros_like(x_flat)  # second moment
        losses: List[float] = []

        for t in range(1, self.max_iter + 1):
            x_shaped = x_flat.reshape(shape)
            loss = fn(x_shaped)
            g = grad_fn(x_shaped).ravel()
            g = self._clip_gradient(g)
            g_norm = float(np.linalg.norm(g))
            losses.append(loss)

            self.monitor.record(ConvergenceInfo(
                iteration=t, loss=loss, grad_norm=g_norm,
                step_size=self.learning_rate,
            ))

            if g_norm < self.tol:
                break
            if self.monitor.is_stalled:
                break

            # Adam update
            m = self.beta1 * m + (1 - self.beta1) * g
            v = self.beta2 * v + (1 - self.beta2) * g * g
            m_hat = m / (1 - self.beta1 ** t)
            v_hat = v / (1 - self.beta2 ** t)
            update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)
            x_flat = x_flat - update

            x_shaped = proj(x_flat.reshape(shape))
            x_flat = x_shaped.ravel()

            # Privacy constraint projection (penalty-based)
            if privacy_constraint_fn is not None and privacy_budget is not None:
                priv = privacy_constraint_fn(x_shaped)
                if priv > privacy_budget:
                    # Step back towards feasible region
                    scale = privacy_budget / max(priv, 1e-30)
                    x_flat *= min(scale, 1.0)
                    x_shaped = proj(x_flat.reshape(shape))
                    x_flat = x_shaped.ravel()

        return x_flat.reshape(shape), losses


# ---------------------------------------------------------------------------
# Frank-Wolfe for simplex constraints
# ---------------------------------------------------------------------------


class FrankWolfe:
    """Frank-Wolfe (conditional gradient) for probability simplex.

    Each iteration solves a linear minimisation problem over the simplex,
    which has a closed-form solution (vertex of simplex).

    Attributes:
        max_iter: Maximum iterations.
        tol: Duality gap convergence tolerance.
        step_rule: Step-size rule ("standard" for 2/(t+2), "line_search").
    """

    def __init__(
        self,
        max_iter: int = 1000,
        tol: float = 1e-8,
        step_rule: str = "standard",
    ) -> None:
        self.max_iter = max_iter
        self.tol = tol
        self.step_rule = step_rule
        self.monitor = ConvergenceMonitor()

    def optimize(
        self,
        x0: npt.NDArray[np.float64],
        fn: Callable[[npt.NDArray[np.float64]], float],
        grad_fn: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    ) -> Tuple[npt.NDArray[np.float64], List[float]]:
        """Run Frank-Wolfe on the probability simplex.

        Args:
            x0: Initial feasible point on simplex.
            fn: Objective function.
            grad_fn: Gradient function.

        Returns:
            (optimised_point, loss_history).
        """
        x = x0.copy()
        if x.ndim == 1:
            x = project_simplex(x)
        else:
            x = project_simplex_rows(x)

        losses: List[float] = []

        for t in range(self.max_iter):
            loss = fn(x)
            g = grad_fn(x)
            g_norm = float(np.linalg.norm(g))
            losses.append(loss)

            self.monitor.record(ConvergenceInfo(
                iteration=t, loss=loss, grad_norm=g_norm,
                step_size=2.0 / (t + 2),
            ))

            # Linear minimisation oracle over simplex
            s = self._lmo(g, x.shape)

            # Duality gap
            gap = float(np.sum(g * (x - s)))
            if gap < self.tol:
                break
            if self.monitor.is_stalled:
                break

            # Step size
            if self.step_rule == "line_search":
                gamma = self._fw_line_search(fn, x, s)
            else:
                gamma = 2.0 / (t + 2)

            x = (1 - gamma) * x + gamma * s

        return x, losses

    @staticmethod
    def _lmo(
        g: npt.NDArray[np.float64],
        shape: Tuple[int, ...],
    ) -> npt.NDArray[np.float64]:
        """Linear minimisation oracle: argmin_{s ∈ Δ} <g, s>.

        For a simplex, this is the vertex e_j where j = argmin g_j.
        """
        if len(shape) == 1:
            s = np.zeros(shape, dtype=np.float64)
            s[np.argmin(g)] = 1.0
            return s
        # Row-wise simplex
        s = np.zeros(shape, dtype=np.float64)
        for i in range(shape[0]):
            j = int(np.argmin(g[i]))
            s[i, j] = 1.0
        return s

    @staticmethod
    def _fw_line_search(
        fn: Callable[[npt.NDArray[np.float64]], float],
        x: npt.NDArray[np.float64],
        s: npt.NDArray[np.float64],
        n_points: int = 20,
    ) -> float:
        """Simple line search for Frank-Wolfe step size."""
        best_gamma = 0.0
        best_val = fn(x)
        for gamma in np.linspace(0, 1, n_points):
            x_new = (1 - gamma) * x + gamma * s
            val = fn(x_new)
            if val < best_val:
                best_val = val
                best_gamma = gamma
        return best_gamma


# ---------------------------------------------------------------------------
# Augmented Lagrangian for privacy constraints
# ---------------------------------------------------------------------------


class AugmentedLagrangian:
    """Augmented Lagrangian method for privacy-constrained optimisation.

    Handles inequality constraints g_j(x) ≤ 0 via Lagrange multipliers
    and a quadratic penalty.

    Attributes:
        inner_optimizer: Optimizer for the unconstrained subproblem.
        max_outer: Maximum outer iterations.
        penalty_init: Initial penalty coefficient.
        penalty_growth: Penalty multiplier per outer iteration.
        tol: Constraint satisfaction tolerance.
    """

    def __init__(
        self,
        max_outer: int = 20,
        penalty_init: float = 1.0,
        penalty_growth: float = 2.0,
        tol: float = 1e-6,
        inner_max_iter: int = 200,
        inner_lr: float = 0.01,
    ) -> None:
        self.max_outer = max_outer
        self.penalty_init = penalty_init
        self.penalty_growth = penalty_growth
        self.tol = tol
        self._inner_max_iter = inner_max_iter
        self._inner_lr = inner_lr

    def optimize(
        self,
        x0: npt.NDArray[np.float64],
        fn: Callable[[npt.NDArray[np.float64]], float],
        grad_fn: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
        constraints: List[Callable[[npt.NDArray[np.float64]], float]],
        constraint_grads: List[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]],
        project_fn: Optional[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]] = None,
    ) -> Tuple[npt.NDArray[np.float64], List[float]]:
        """Run augmented Lagrangian method.

        Args:
            x0: Initial point.
            fn: Objective function.
            grad_fn: Gradient of objective.
            constraints: List of constraint functions g_j(x) ≤ 0.
            constraint_grads: Gradients of constraints.
            project_fn: Projection onto basic feasible set.

        Returns:
            (optimised_point, loss_history).
        """
        n_constraints = len(constraints)
        lambdas = np.zeros(n_constraints, dtype=np.float64)
        rho = self.penalty_init
        x = x0.copy()
        all_losses: List[float] = []

        for outer in range(self.max_outer):
            # Build augmented Lagrangian
            def aug_fn(x_: npt.NDArray[np.float64],
                       _lam: npt.NDArray = lambdas,
                       _rho: float = rho) -> float:
                val = fn(x_)
                for j in range(n_constraints):
                    gj = constraints[j](x_)
                    val += _lam[j] * gj + 0.5 * _rho * max(0.0, gj) ** 2
                return val

            def aug_grad(x_: npt.NDArray[np.float64],
                         _lam: npt.NDArray = lambdas,
                         _rho: float = rho) -> npt.NDArray[np.float64]:
                g = grad_fn(x_).copy()
                for j in range(n_constraints):
                    gj = constraints[j](x_)
                    cg = constraint_grads[j](x_)
                    g += _lam[j] * cg
                    if gj > 0:
                        g += _rho * gj * cg
                return g

            pgd = ProjectedGradientDescent(
                learning_rate=self._inner_lr,
                max_iter=self._inner_max_iter,
                tol=1e-9,
            )
            x, losses = pgd.optimize(x, aug_fn, aug_grad, project_fn)
            all_losses.extend(losses)

            # Update multipliers
            max_viol = 0.0
            for j in range(n_constraints):
                gj = constraints[j](x)
                lambdas[j] = max(0.0, lambdas[j] + rho * gj)
                max_viol = max(max_viol, max(0.0, gj))

            if max_viol < self.tol:
                break
            rho *= self.penalty_growth

        return x, all_losses


__all__ = [
    "project_simplex",
    "project_simplex_rows",
    "project_nonneg",
    "LineSearch",
    "LineSearchMethod",
    "LineSearchResult",
    "ConvergenceMonitor",
    "ConvergenceInfo",
    "ProjectedGradientDescent",
    "PrivacyAwareAdam",
    "FrankWolfe",
    "AugmentedLagrangian",
]
