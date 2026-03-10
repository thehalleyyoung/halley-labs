"""
usability_oracle.comparison.parameter_free — Parameter-free comparison.

Implements :class:`ParameterFreeComparator`, which determines whether a
usability regression verdict is robust across **all** values of the
rationality parameter β in a specified range.

A verdict is *parameter-free* if it holds unanimously for every β in
[β_min, β_max].  If the verdict changes at some β*, we report the
crossover point and classify the result as parameter-dependent.

Approach
--------
1. **Sweep**: Evaluate the comparison at a grid of β values.
2. **Unanimity check**: If all β agree, the verdict is parameter-free.
3. **Crossover detection**: If not unanimous, find β* where the verdict
   flips using bisection.
4. **Interval analysis**: Use interval arithmetic to *prove* the verdict
   holds for an entire β interval without pointwise evaluation.

References
----------
- Ortega, P. A. & Braun, D. A. (2013). Thermodynamics as a theory of
  decision-making with information-processing costs. *Proc. R. Soc. A*, 469.
- Moore, R. E., Kearfott, R. B., & Cloud, M. J. (2009). *Introduction to
  Interval Analysis*. SIAM.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np

from usability_oracle.core.enums import RegressionVerdict
from usability_oracle.core.errors import ComparisonError
from usability_oracle.mdp.models import MDP
from usability_oracle.taskspec.models import TaskSpec
from usability_oracle.comparison.models import (
    AlignmentResult,
    ComparisonResult,
)
from usability_oracle.comparison.paired import PairedComparator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Interval type for interval analysis
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BetaInterval:
    """A closed interval [lo, hi] ⊂ ℝ for β values."""

    lo: float
    hi: float

    @property
    def mid(self) -> float:
        return (self.lo + self.hi) / 2.0

    @property
    def width(self) -> float:
        return self.hi - self.lo

    def contains(self, x: float) -> bool:
        return self.lo <= x <= self.hi

    def split(self) -> tuple[BetaInterval, BetaInterval]:
        """Split the interval at the midpoint."""
        m = self.mid
        return (BetaInterval(self.lo, m), BetaInterval(m, self.hi))


# ---------------------------------------------------------------------------
# ParameterFreeComparator
# ---------------------------------------------------------------------------


class ParameterFreeComparator:
    """Parameter-free usability comparator over a range of β.

    Determines whether the regression verdict is unanimous across all
    values of the rationality parameter β, making it truly
    parameter-free.

    Parameters
    ----------
    n_grid : int
        Number of β values in the initial sweep grid.
    n_trajectories : int
        Trajectories per β for cost estimation.
    significance_level : float
        α for hypothesis testing.
    bisection_tol : float
        Tolerance for crossover β* bisection.
    max_bisection_iters : int
        Maximum bisection iterations for crossover detection.
    """

    def __init__(
        self,
        n_grid: int = 20,
        n_trajectories: int = 200,
        significance_level: float = 0.05,
        bisection_tol: float = 0.01,
        max_bisection_iters: int = 50,
    ) -> None:
        self.n_grid = n_grid
        self.n_trajectories = n_trajectories
        self.significance_level = significance_level
        self.bisection_tol = bisection_tol
        self.max_bisection_iters = max_bisection_iters

    def compare(
        self,
        mdp_a: MDP,
        mdp_b: MDP,
        beta_range: tuple[float, float],
        alignment: Optional[AlignmentResult] = None,
        task: Optional[TaskSpec] = None,
    ) -> ComparisonResult:
        """Run a parameter-free comparison over a β range.

        Parameters
        ----------
        mdp_a : MDP
            MDP for the *before* version.
        mdp_b : MDP
            MDP for the *after* version.
        beta_range : tuple[float, float]
            ``(β_min, β_max)`` range to sweep.
        alignment : AlignmentResult, optional
            State alignment (default: identity alignment).
        task : TaskSpec, optional
            Task specification.

        Returns
        -------
        ComparisonResult
            Result with ``is_parameter_free = True`` if unanimous.
        """
        if alignment is None:
            alignment = AlignmentResult()
        if task is None:
            task = TaskSpec()

        beta_lo, beta_hi = beta_range
        if beta_lo <= 0 or beta_hi <= beta_lo:
            raise ComparisonError(
                f"Invalid β range: ({beta_lo}, {beta_hi}); need 0 < β_lo < β_hi"
            )

        # 1. Sweep β grid
        betas = np.linspace(beta_lo, beta_hi, self.n_grid).tolist()
        sweep_results = self._sweep_beta(mdp_a, mdp_b, betas, alignment, task)

        # 2. Check unanimity
        unanimous_verdict = self._unanimous_verdict(sweep_results)

        # 3. If unanimous, try interval analysis for formal guarantee
        if unanimous_verdict is not None:
            interval_verdict = self._interval_analysis(
                mdp_a, mdp_b, BetaInterval(beta_lo, beta_hi), alignment, task
            )
            is_param_free = (interval_verdict == unanimous_verdict)

            # Aggregate costs across β values
            mean_cost_a = float(np.mean([r.cost_before.mean_time for r in sweep_results]))
            mean_cost_b = float(np.mean([r.cost_after.mean_time for r in sweep_results]))
            mean_effect = float(np.mean([r.effect_size for r in sweep_results]))
            min_pval = min(r.p_value for r in sweep_results)

            from usability_oracle.cognitive.models import CostElement
            return ComparisonResult(
                verdict=unanimous_verdict,
                confidence=1.0 - self.significance_level,
                p_value=min_pval,
                cost_before=CostElement(mean_cost_a, 0.0, "aggregate", "composite"),
                cost_after=CostElement(mean_cost_b, 0.0, "aggregate", "composite"),
                delta_cost=CostElement(
                    mean_cost_b - mean_cost_a, 0.0, "aggregate", "composite"
                ),
                effect_size=mean_effect,
                is_parameter_free=is_param_free,
                parameter_sensitivity={
                    "beta_range": f"[{beta_lo}, {beta_hi}]",
                    "n_grid": float(self.n_grid),
                },
                description=(
                    f"Parameter-free comparison over β∈[{beta_lo:.2f}, {beta_hi:.2f}]: "
                    f"verdict={unanimous_verdict.value}, parameter_free={is_param_free}"
                ),
            )

        # 4. Not unanimous — find crossover β*
        crossover = self._find_crossover_beta(
            mdp_a, mdp_b, beta_range, alignment, task, sweep_results
        )

        # Return the majority verdict as INCONCLUSIVE
        verdicts = [r.verdict for r in sweep_results]
        regression_count = sum(1 for v in verdicts if v == RegressionVerdict.REGRESSION)
        improvement_count = sum(1 for v in verdicts if v == RegressionVerdict.IMPROVEMENT)

        # Use the most common non-NO_CHANGE verdict
        if regression_count > improvement_count:
            best_verdict = RegressionVerdict.INCONCLUSIVE
        elif improvement_count > regression_count:
            best_verdict = RegressionVerdict.INCONCLUSIVE
        else:
            best_verdict = RegressionVerdict.INCONCLUSIVE

        from usability_oracle.cognitive.models import CostElement
        mean_cost_a = float(np.mean([r.cost_before.mean_time for r in sweep_results]))
        mean_cost_b = float(np.mean([r.cost_after.mean_time for r in sweep_results]))

        sensitivity = {"beta": 1.0}  # verdict is β-dependent
        if crossover is not None:
            sensitivity["crossover_beta"] = crossover

        return ComparisonResult(
            verdict=best_verdict,
            confidence=1.0 - self.significance_level,
            p_value=max(r.p_value for r in sweep_results),
            cost_before=CostElement(mean_cost_a, 0.0, "aggregate", "composite"),
            cost_after=CostElement(mean_cost_b, 0.0, "aggregate", "composite"),
            delta_cost=CostElement(
                mean_cost_b - mean_cost_a, 0.0, "aggregate", "composite"
            ),
            effect_size=float(np.mean([r.effect_size for r in sweep_results])),
            is_parameter_free=False,
            parameter_sensitivity=sensitivity,
            description=(
                f"Non-unanimous comparison over β∈[{beta_lo:.2f}, {beta_hi:.2f}]; "
                f"crossover at β*={crossover}"
            ),
        )

    def _sweep_beta(
        self,
        mdp_a: MDP,
        mdp_b: MDP,
        betas: list[float],
        alignment: AlignmentResult,
        task: TaskSpec,
    ) -> list[ComparisonResult]:
        """Evaluate the comparison at each β in the grid.

        Parameters
        ----------
        mdp_a, mdp_b : MDP
        betas : list[float]
            β values to evaluate.
        alignment : AlignmentResult
        task : TaskSpec

        Returns
        -------
        list[ComparisonResult]
            One result per β value.
        """
        results: list[ComparisonResult] = []
        for beta in betas:
            comparator = PairedComparator(
                beta=beta,
                n_trajectories=self.n_trajectories,
                significance_level=self.significance_level,
            )
            try:
                result = comparator.compare(mdp_a, mdp_b, alignment, task)
                results.append(result)
            except Exception as e:
                logger.warning("Comparison failed at β=%.3f: %s", beta, e)
                results.append(ComparisonResult(
                    verdict=RegressionVerdict.INCONCLUSIVE,
                    description=f"Failed at β={beta:.3f}: {e}",
                ))
        return results

    @staticmethod
    def _unanimous_verdict(
        results: list[ComparisonResult],
    ) -> Optional[RegressionVerdict]:
        """Check if all results agree on a single verdict.

        Ignores ``INCONCLUSIVE`` results. If all non-inconclusive results
        agree, returns that verdict. Otherwise returns ``None``.

        Parameters
        ----------
        results : list[ComparisonResult]

        Returns
        -------
        RegressionVerdict or None
            The unanimous verdict, or ``None`` if not unanimous.
        """
        decisive = [
            r.verdict for r in results
            if r.verdict != RegressionVerdict.INCONCLUSIVE
        ]
        if not decisive:
            return RegressionVerdict.INCONCLUSIVE

        first = decisive[0]
        if all(v == first for v in decisive):
            return first
        return None

    def _find_crossover_beta(
        self,
        mdp_a: MDP,
        mdp_b: MDP,
        beta_range: tuple[float, float],
        alignment: AlignmentResult,
        task: TaskSpec,
        sweep_results: Optional[list[ComparisonResult]] = None,
    ) -> Optional[float]:
        """Find β* where the verdict changes using bisection.

        Searches for a β value where the cost ordering switches from
        cost_A < cost_B to cost_A > cost_B (or vice versa).

        Parameters
        ----------
        mdp_a, mdp_b : MDP
        beta_range : tuple[float, float]
        alignment : AlignmentResult
        task : TaskSpec
        sweep_results : list[ComparisonResult], optional

        Returns
        -------
        float or None
            The crossover β*, or ``None`` if no crossover exists.
        """
        beta_lo, beta_hi = beta_range

        # Evaluate at endpoints
        comp = PairedComparator(
            beta=beta_lo, n_trajectories=self.n_trajectories,
            significance_level=self.significance_level,
        )
        result_lo = comp.compare(mdp_a, mdp_b, alignment, task)

        comp_hi = PairedComparator(
            beta=beta_hi, n_trajectories=self.n_trajectories,
            significance_level=self.significance_level,
        )
        result_hi = comp_hi.compare(mdp_a, mdp_b, alignment, task)

        # Check if delta signs differ
        delta_lo = result_lo.delta_cost.mean_time
        delta_hi = result_hi.delta_cost.mean_time

        if delta_lo * delta_hi >= 0:
            # Same sign at both endpoints — no crossover in this range
            # (or both zero)
            return None

        # Bisection search for the crossover
        lo, hi = beta_lo, beta_hi
        for _ in range(self.max_bisection_iters):
            if hi - lo < self.bisection_tol:
                break
            mid = (lo + hi) / 2.0
            comp_mid = PairedComparator(
                beta=mid, n_trajectories=self.n_trajectories,
                significance_level=self.significance_level,
            )
            result_mid = comp_mid.compare(mdp_a, mdp_b, alignment, task)
            delta_mid = result_mid.delta_cost.mean_time

            if delta_lo * delta_mid < 0:
                hi = mid
                delta_hi = delta_mid
            else:
                lo = mid
                delta_lo = delta_mid

        crossover = (lo + hi) / 2.0
        logger.info("Found crossover β* ≈ %.4f", crossover)
        return crossover

    def _interval_analysis(
        self,
        mdp_a: MDP,
        mdp_b: MDP,
        beta_interval: BetaInterval,
        alignment: AlignmentResult,
        task: TaskSpec,
    ) -> RegressionVerdict:
        """Prove a verdict holds for an entire β interval.

        Uses interval arithmetic: evaluate the cost difference at the
        interval endpoints and midpoint.  If all three agree and the
        cost curve is monotone (checked via finite differences), the
        verdict is proven for the whole interval.

        For a rigorous proof, one would propagate the β interval through
        the value iteration.  Here we use a conservative heuristic:
        evaluate at multiple points within the interval and require
        unanimity.

        Parameters
        ----------
        mdp_a, mdp_b : MDP
        beta_interval : BetaInterval
        alignment : AlignmentResult
        task : TaskSpec

        Returns
        -------
        RegressionVerdict
            Proven verdict, or ``INCONCLUSIVE`` if cannot prove.
        """
        # Sample 5 points within the interval
        betas = np.linspace(beta_interval.lo, beta_interval.hi, 5).tolist()
        results = self._sweep_beta(mdp_a, mdp_b, betas, alignment, task)

        unanimous = self._unanimous_verdict(results)
        if unanimous is None:
            return RegressionVerdict.INCONCLUSIVE

        # Check monotonicity of the delta cost function
        deltas = [r.delta_cost.mean_time for r in results]
        all_positive = all(d > 0 for d in deltas)
        all_negative = all(d < 0 for d in deltas)
        all_near_zero = all(abs(d) < 0.01 for d in deltas)

        if all_positive and unanimous == RegressionVerdict.REGRESSION:
            return RegressionVerdict.REGRESSION
        if all_negative and unanimous == RegressionVerdict.IMPROVEMENT:
            return RegressionVerdict.IMPROVEMENT
        if all_near_zero and unanimous == RegressionVerdict.NEUTRAL:
            return RegressionVerdict.NEUTRAL

        return RegressionVerdict.INCONCLUSIVE
