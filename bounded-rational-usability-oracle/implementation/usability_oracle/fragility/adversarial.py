"""
usability_oracle.fragility.adversarial — Adversarial β analysis.

Implements :class:`AdversarialAnalyzer`, which finds the *worst-case*
and *best-case* rationality parameter β for a given MDP, and performs
minimax regret comparison between two UI versions.

The adversarial perspective answers: "How bad can this interface be
for the least-favourable user population?"

Key functions
-------------
- **find_worst_beta**: β that maximizes task cost (adversarial).
- **find_best_beta**: β that minimizes task cost (optimistic).
- **minimax_regret**: Compares two designs under worst-case regret.
- **adversarial_comparison**: Regression verdict under adversarial β.

References
----------
- Wald, A. (1945). Statistical decision functions which minimize the
  maximum risk. *Annals of Mathematics*, 46(2).
- Savage, L. J. (1951). The theory of statistical decision. *J. American
  Statistical Association*, 46(253).
"""

from __future__ import annotations

import math
import logging
from typing import Any, Optional, Tuple

import numpy as np
from scipy import optimize as sp_optimize

from usability_oracle.core.enums import RegressionVerdict
from usability_oracle.mdp.models import MDP

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _evaluate_cost(mdp: MDP, beta: float, n_trajectories: int = 100) -> float:
    """Compute expected task cost for *mdp* at rationality *beta*."""
    from usability_oracle.comparison.paired import (
        _solve_softmax_policy,
        _sample_trajectory_costs,
    )
    policy = _solve_softmax_policy(mdp, beta)
    samples = _sample_trajectory_costs(mdp, policy, n_trajectories)
    return float(np.mean(samples))


# ---------------------------------------------------------------------------
# AdversarialAnalyzer
# ---------------------------------------------------------------------------


class AdversarialAnalyzer:
    """Adversarial analysis of usability under worst-case β.

    Finds the rationality parameter that maximizes (or minimizes) task
    cost, and performs minimax regret comparison between UI versions.

    Parameters
    ----------
    n_trajectories : int
        Monte Carlo trajectories per β evaluation.
    optimizer_maxiter : int
        Maximum iterations for scipy.optimize.
    grid_resolution : int
        Grid resolution for initial coarse search.
    """

    def __init__(
        self,
        n_trajectories: int = 100,
        optimizer_maxiter: int = 100,
        grid_resolution: int = 50,
    ) -> None:
        self.n_trajectories = n_trajectories
        self.optimizer_maxiter = optimizer_maxiter
        self.grid_resolution = grid_resolution

    def find_worst_beta(
        self,
        mdp: MDP,
        beta_range: tuple[float, float],
    ) -> tuple[float, float]:
        """Find the β that maximizes task cost (adversarial analysis).

        Solves:

        .. math::
            \\beta^* = \\arg\\max_{\\beta \\in [\\beta_l, \\beta_h]} C(\\beta)

        Parameters
        ----------
        mdp : MDP
        beta_range : tuple[float, float]

        Returns
        -------
        tuple[float, float]
            ``(β*, C(β*))`` — worst-case β and corresponding cost.
        """
        return self._optimize_beta(mdp, "maximize", beta_range)

    def find_best_beta(
        self,
        mdp: MDP,
        beta_range: tuple[float, float],
    ) -> tuple[float, float]:
        """Find the β that minimizes task cost (optimistic analysis).

        Solves:

        .. math::
            \\beta^* = \\arg\\min_{\\beta \\in [\\beta_l, \\beta_h]} C(\\beta)

        Parameters
        ----------
        mdp : MDP
        beta_range : tuple[float, float]

        Returns
        -------
        tuple[float, float]
            ``(β*, C(β*))`` — best-case β and corresponding cost.
        """
        return self._optimize_beta(mdp, "minimize", beta_range)

    def minimax_regret(
        self,
        mdp_a: MDP,
        mdp_b: MDP,
        beta_range: tuple[float, float],
    ) -> float:
        """Compute the minimax regret between two designs.

        The minimax regret is:

        .. math::
            R_{mm} = \\max_{\\beta} \\min\\bigl(C_A(\\beta),\\, C_B(\\beta)\\bigr)

        This measures the worst-case cost achievable by the *better* of
        the two designs at any given β.  A design with lower minimax
        regret is more robust.

        Parameters
        ----------
        mdp_a : MDP
            *Before* version.
        mdp_b : MDP
            *After* version.
        beta_range : tuple[float, float]

        Returns
        -------
        float
            Minimax regret value.
        """
        beta_lo, beta_hi = beta_range

        def neg_min_cost(beta: float) -> float:
            """Negative of min(C_A, C_B) for maximization."""
            beta = float(np.clip(beta, beta_lo, beta_hi))
            cost_a = _evaluate_cost(mdp_a, beta, self.n_trajectories)
            cost_b = _evaluate_cost(mdp_b, beta, self.n_trajectories)
            return -min(cost_a, cost_b)

        # Coarse grid search first
        betas_grid = np.linspace(beta_lo, beta_hi, self.grid_resolution)
        grid_vals = np.array([neg_min_cost(b) for b in betas_grid])
        best_idx = np.argmin(grid_vals)  # most negative = highest min cost
        beta_init = float(betas_grid[best_idx])

        # Refine with bounded minimization (we minimize the negative)
        result = sp_optimize.minimize_scalar(
            neg_min_cost,
            bounds=(beta_lo, beta_hi),
            method="bounded",
            options={"maxiter": self.optimizer_maxiter, "xatol": 0.01},
        )

        optimal_beta = float(result.x)
        minimax_value = -float(result.fun)

        logger.info(
            "Minimax regret: β*=%.3f, R_mm=%.4f", optimal_beta, minimax_value
        )
        return minimax_value

    def _optimize_beta(
        self,
        mdp: MDP,
        objective: str,
        beta_range: tuple[float, float],
    ) -> tuple[float, float]:
        """Find the β that maximizes or minimizes task cost.

        Uses ``scipy.optimize.minimize_scalar`` with bounded method.
        For maximization, we negate the objective.

        Parameters
        ----------
        mdp : MDP
        objective : str
            ``"maximize"`` or ``"minimize"``.
        beta_range : tuple[float, float]

        Returns
        -------
        tuple[float, float]
            ``(β*, C(β*))``.
        """
        beta_lo, beta_hi = beta_range
        sign = -1.0 if objective == "maximize" else 1.0

        # Cache to avoid redundant evaluations
        cache: dict[float, float] = {}

        def obj_fn(beta: float) -> float:
            beta = float(np.clip(beta, beta_lo, beta_hi))
            beta_rounded = round(beta, 6)
            if beta_rounded not in cache:
                cache[beta_rounded] = _evaluate_cost(
                    mdp, beta_rounded, self.n_trajectories
                )
            return sign * cache[beta_rounded]

        # Coarse grid search for initialization
        betas_grid = np.linspace(beta_lo, beta_hi, self.grid_resolution)
        grid_vals = np.array([obj_fn(b) for b in betas_grid])
        best_grid_idx = int(np.argmin(grid_vals))
        beta_init = float(betas_grid[best_grid_idx])

        # Fine-grained optimization
        result = sp_optimize.minimize_scalar(
            obj_fn,
            bounds=(beta_lo, beta_hi),
            method="bounded",
            options={"maxiter": self.optimizer_maxiter, "xatol": 0.005},
        )

        optimal_beta = float(result.x)
        optimal_cost = _evaluate_cost(mdp, optimal_beta, self.n_trajectories)

        logger.info(
            "%s β: β*=%.4f, C(β*)=%.4f",
            objective.capitalize(), optimal_beta, optimal_cost,
        )
        return (optimal_beta, optimal_cost)

    def adversarial_comparison(
        self,
        mdp_a: MDP,
        mdp_b: MDP,
        beta_range: tuple[float, float],
    ) -> RegressionVerdict:
        """Determine the regression verdict under adversarial β.

        Finds the worst-case β for each design and compares.  A
        regression is declared if the *after* version is worse than
        the *before* version at the adversarial β.

        The adversarial β is chosen to maximize the cost difference:

        .. math::
            \\beta^* = \\arg\\max_{\\beta} \\bigl[C_B(\\beta) - C_A(\\beta)\\bigr]

        Parameters
        ----------
        mdp_a : MDP
            *Before* version.
        mdp_b : MDP
            *After* version.
        beta_range : tuple[float, float]

        Returns
        -------
        RegressionVerdict
        """
        beta_lo, beta_hi = beta_range

        def neg_delta(beta: float) -> float:
            """Negative of (C_B - C_A) for maximization of the difference."""
            beta = float(np.clip(beta, beta_lo, beta_hi))
            cost_a = _evaluate_cost(mdp_a, beta, self.n_trajectories)
            cost_b = _evaluate_cost(mdp_b, beta, self.n_trajectories)
            return -(cost_b - cost_a)

        # Coarse grid
        betas_grid = np.linspace(beta_lo, beta_hi, self.grid_resolution)
        grid_vals = np.array([neg_delta(b) for b in betas_grid])
        best_idx = int(np.argmin(grid_vals))

        # Fine optimization
        result = sp_optimize.minimize_scalar(
            neg_delta,
            bounds=(beta_lo, beta_hi),
            method="bounded",
            options={"maxiter": self.optimizer_maxiter, "xatol": 0.01},
        )

        adversarial_beta = float(result.x)
        cost_a = _evaluate_cost(mdp_a, adversarial_beta, self.n_trajectories)
        cost_b = _evaluate_cost(mdp_b, adversarial_beta, self.n_trajectories)
        max_delta = cost_b - cost_a

        logger.info(
            "Adversarial comparison: β*=%.3f, C_A=%.4f, C_B=%.4f, Δ=%.4f",
            adversarial_beta, cost_a, cost_b, max_delta,
        )

        # Also check the best case for B (minimizes delta)
        def pos_delta(beta: float) -> float:
            beta = float(np.clip(beta, beta_lo, beta_hi))
            cost_a = _evaluate_cost(mdp_a, beta, self.n_trajectories)
            cost_b = _evaluate_cost(mdp_b, beta, self.n_trajectories)
            return cost_b - cost_a

        result_min = sp_optimize.minimize_scalar(
            pos_delta,
            bounds=(beta_lo, beta_hi),
            method="bounded",
            options={"maxiter": self.optimizer_maxiter, "xatol": 0.01},
        )
        min_delta = float(result_min.fun)

        # Decision logic:
        # - If even the adversarial β shows no regression → no regression
        # - If even the best-case β shows regression → definite regression
        # - Otherwise → inconclusive
        if max_delta <= 0:
            return RegressionVerdict.IMPROVEMENT
        if min_delta > 0:
            return RegressionVerdict.REGRESSION
        if max_delta > 0.1:
            return RegressionVerdict.REGRESSION
        return RegressionVerdict.INCONCLUSIVE
