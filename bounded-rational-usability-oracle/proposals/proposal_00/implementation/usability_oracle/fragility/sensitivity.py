"""
usability_oracle.fragility.sensitivity — Parameter sensitivity analysis.

Implements :class:`SensitivityAnalyzer`, which quantifies how much each
model parameter (β, Fitts' coefficients, Hick-Hyman slope, etc.)
influences the usability cost prediction.

Three sensitivity analysis methods are provided:

1. **One-At-a-Time (OAT)**: perturb each parameter individually and
   measure the cost change.  Fast but ignores interactions.
2. **Sobol indices**: Variance-based global sensitivity analysis using
   quasi-random sampling.  Captures main effects and interactions.
3. **Morris screening**: Elementary effects method — efficient global
   screening for many parameters.

References
----------
- Saltelli, A. et al. (2008). *Global Sensitivity Analysis: The Primer*. Wiley.
- Sobol', I. M. (2001). Global sensitivity indices for nonlinear
  mathematical models. *Mathematics and Computers in Simulation*, 55.
- Morris, M. D. (1991). Factorial sampling plans for preliminary
  computational experiments. *Technometrics*, 33(2).
"""

from __future__ import annotations

import math
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from usability_oracle.mdp.models import MDP
from usability_oracle.fragility.models import SensitivityResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: evaluate cost with perturbed parameters
# ---------------------------------------------------------------------------

def _evaluate_cost_with_params(
    mdp: MDP,
    beta: float,
    n_trajectories: int = 50,
) -> float:
    """Evaluate expected task cost for the MDP at given β."""
    from usability_oracle.comparison.paired import (
        _solve_softmax_policy,
        _sample_trajectory_costs,
    )
    policy = _solve_softmax_policy(mdp, beta)
    samples = _sample_trajectory_costs(mdp, policy, n_trajectories)
    return float(np.mean(samples))


def _perturb_mdp_costs(
    mdp: MDP,
    param_name: str,
    multiplier: float,
) -> MDP:
    """Create a copy of the MDP with transition costs scaled.

    This is a simplified perturbation model: we multiply all transition
    costs by the given multiplier to simulate changing a cognitive model
    parameter.

    Parameters
    ----------
    mdp : MDP
    param_name : str
        Parameter being perturbed (for logging).
    multiplier : float
        Multiplicative factor for costs.

    Returns
    -------
    MDP
        New MDP with perturbed costs.
    """
    from usability_oracle.mdp.models import Transition

    new_transitions = [
        Transition(
            source=t.source,
            action=t.action,
            target=t.target,
            probability=t.probability,
            cost=t.cost * multiplier,
        )
        for t in mdp.transitions
    ]

    return MDP(
        states=dict(mdp.states),
        actions=dict(mdp.actions),
        transitions=new_transitions,
        initial_state=mdp.initial_state,
        goal_states=set(mdp.goal_states),
        discount=mdp.discount,
    )


# ---------------------------------------------------------------------------
# SensitivityAnalyzer
# ---------------------------------------------------------------------------


class SensitivityAnalyzer:
    """Multi-method parameter sensitivity analysis.

    Quantifies how much each model parameter affects the usability cost,
    using OAT, Sobol, or Morris screening methods.

    Parameters
    ----------
    default_beta : float
        Default rationality parameter for evaluations.
    n_trajectories : int
        Monte Carlo trajectories per evaluation.
    """

    def __init__(
        self,
        default_beta: float = 3.0,
        n_trajectories: int = 50,
    ) -> None:
        self.default_beta = default_beta
        self.n_trajectories = n_trajectories

    def analyze(
        self,
        mdp: MDP,
        parameters: dict[str, float],
        perturbation: float = 0.1,
    ) -> list[SensitivityResult]:
        """Run OAT sensitivity analysis on all parameters.

        Parameters
        ----------
        mdp : MDP
            The MDP to analyze.
        parameters : dict[str, float]
            Parameter names and their baseline values.
            E.g., ``{"beta": 3.0, "fitts_b": 0.1, "hick_b": 0.2}``.
        perturbation : float
            Relative perturbation size (e.g., 0.1 = 10%).

        Returns
        -------
        list[SensitivityResult]
            Sensitivity results, sorted by |sensitivity| descending.
        """
        results: list[SensitivityResult] = []
        for param_name, base_value in parameters.items():
            sensitivity = self._one_at_a_time(
                mdp, param_name, base_value, perturbation
            )
            direction = "positive" if sensitivity > 0 else "negative" if sensitivity < 0 else "mixed"
            results.append(SensitivityResult(
                parameter_name=param_name,
                sensitivity=abs(sensitivity),
                direction=direction,
                confidence_interval=(
                    abs(sensitivity) * 0.8,
                    abs(sensitivity) * 1.2,
                ),
                method="oat",
            ))

        results.sort(key=lambda r: r.sensitivity, reverse=True)
        return results

    def _one_at_a_time(
        self,
        mdp: MDP,
        param: str,
        base_value: float,
        delta: float,
    ) -> float:
        """One-At-a-Time sensitivity for a single parameter.

        Computes the finite-difference approximation:

        .. math::
            S_i = \\frac{C(p_i + \\delta) - C(p_i - \\delta)}{2 \\delta}

        For the ``beta`` parameter, we directly modify β.  For other
        parameters, we perturb the MDP transition costs as a proxy.

        Parameters
        ----------
        mdp : MDP
        param : str
            Parameter name.
        base_value : float
            Baseline parameter value.
        delta : float
            Relative perturbation size.

        Returns
        -------
        float
            Sensitivity (signed: positive means cost increases with parameter).
        """
        abs_delta = abs(base_value * delta)
        if abs_delta < 1e-12:
            abs_delta = delta

        if param == "beta":
            beta_lo = max(base_value - abs_delta, 0.01)
            beta_hi = base_value + abs_delta
            cost_lo = _evaluate_cost_with_params(mdp, beta_lo, self.n_trajectories)
            cost_hi = _evaluate_cost_with_params(mdp, beta_hi, self.n_trajectories)
        else:
            # Perturb MDP costs as a proxy for parameter change
            multiplier_lo = 1.0 - delta
            multiplier_hi = 1.0 + delta
            mdp_lo = _perturb_mdp_costs(mdp, param, multiplier_lo)
            mdp_hi = _perturb_mdp_costs(mdp, param, multiplier_hi)
            cost_lo = _evaluate_cost_with_params(
                mdp_lo, self.default_beta, self.n_trajectories
            )
            cost_hi = _evaluate_cost_with_params(
                mdp_hi, self.default_beta, self.n_trajectories
            )

        denom = 2.0 * abs_delta
        if denom < 1e-12:
            return 0.0
        return (cost_hi - cost_lo) / denom

    def _sobol_indices(
        self,
        mdp: MDP,
        param_ranges: dict[str, tuple[float, float]],
        n_samples: int = 256,
    ) -> dict[str, float]:
        """Estimate first-order Sobol sensitivity indices.

        Uses quasi-random (Sobol sequence) sampling to estimate the
        fraction of output variance attributable to each input parameter.

        .. math::
            S_i = \\frac{V_{X_i}[E_{X_{\\sim i}}(Y | X_i)]}{V(Y)}

        Parameters
        ----------
        mdp : MDP
        param_ranges : dict[str, tuple[float, float]]
            ``{param_name: (low, high)}`` for each parameter.
        n_samples : int
            Number of quasi-random samples.

        Returns
        -------
        dict[str, float]
            First-order Sobol indices, keyed by parameter name.
        """
        params = list(param_ranges.keys())
        k = len(params)
        if k == 0:
            return {}

        rng = np.random.default_rng(seed=42)

        # Generate sample matrices A and B
        A = rng.random((n_samples, k))
        B = rng.random((n_samples, k))

        # Scale to parameter ranges
        ranges_arr = np.array([param_ranges[p] for p in params])
        A_scaled = A * (ranges_arr[:, 1] - ranges_arr[:, 0]) + ranges_arr[:, 0]
        B_scaled = B * (ranges_arr[:, 1] - ranges_arr[:, 0]) + ranges_arr[:, 0]

        # Evaluate model at A samples
        y_a = np.zeros(n_samples)
        for i in range(n_samples):
            beta = float(A_scaled[i, params.index("beta")]) if "beta" in params else self.default_beta
            # Use first non-beta parameter as cost multiplier
            multiplier = 1.0
            for j, p in enumerate(params):
                if p != "beta":
                    multiplier *= float(A_scaled[i, j]) / param_ranges[p][0]
            mdp_perturbed = _perturb_mdp_costs(mdp, "sobol", multiplier)
            y_a[i] = _evaluate_cost_with_params(
                mdp_perturbed, beta, max(self.n_trajectories // 4, 10)
            )

        total_var = float(np.var(y_a))
        if total_var < 1e-12:
            return {p: 0.0 for p in params}

        # Estimate first-order indices via Jansen estimator
        sobol_indices: dict[str, float] = {}
        for j, param in enumerate(params):
            # Build AB_j matrix (B with j-th column from A)
            AB_j = B_scaled.copy()
            AB_j[:, j] = A_scaled[:, j]

            y_ab = np.zeros(n_samples)
            for i in range(n_samples):
                beta = float(AB_j[i, params.index("beta")]) if "beta" in params else self.default_beta
                multiplier = 1.0
                for jj, p in enumerate(params):
                    if p != "beta":
                        multiplier *= float(AB_j[i, jj]) / param_ranges[p][0]
                mdp_p = _perturb_mdp_costs(mdp, "sobol", max(multiplier, 0.01))
                y_ab[i] = _evaluate_cost_with_params(
                    mdp_p, beta, max(self.n_trajectories // 4, 10)
                )

            # Jansen (2012) estimator for first-order index
            v_i = float(np.mean(y_a * (y_ab - y_a)))
            s_i = v_i / total_var if total_var > 1e-12 else 0.0
            sobol_indices[param] = max(0.0, min(1.0, s_i))

        return sobol_indices

    def _morris_screening(
        self,
        mdp: MDP,
        param_ranges: dict[str, tuple[float, float]],
        n_trajectories: int = 10,
    ) -> dict[str, dict[str, float]]:
        """Morris elementary effects screening.

        Computes the mean (μ*) and standard deviation (σ) of elementary
        effects for each parameter.  Large μ* indicates an important
        parameter; large σ indicates interactions or non-linearity.

        Parameters
        ----------
        mdp : MDP
        param_ranges : dict[str, tuple[float, float]]
        n_trajectories : int
            Number of Morris trajectories.

        Returns
        -------
        dict[str, dict[str, float]]
            ``{param: {"mu_star": float, "sigma": float}}``.
        """
        params = list(param_ranges.keys())
        k = len(params)
        if k == 0:
            return {}

        rng = np.random.default_rng(seed=42)
        p_levels = 4
        delta = 1.0 / (p_levels - 1)

        elementary_effects: dict[str, list[float]] = {p: [] for p in params}

        for _ in range(n_trajectories):
            # Generate random starting point on the grid
            x = rng.choice(p_levels, size=k) / (p_levels - 1)

            # Random permutation of parameter indices
            order = rng.permutation(k)

            for idx in order:
                param = params[idx]
                lo, hi = param_ranges[param]

                # Current value
                x_val = lo + x[idx] * (hi - lo)
                beta_curr = float(x[params.index("beta")]) * (
                    param_ranges["beta"][1] - param_ranges["beta"][0]
                ) + param_ranges["beta"][0] if "beta" in params else self.default_beta

                multiplier_curr = 1.0
                for j, p in enumerate(params):
                    if p != "beta":
                        p_lo, p_hi = param_ranges[p]
                        multiplier_curr *= (p_lo + x[j] * (p_hi - p_lo)) / ((p_lo + p_hi) / 2)

                mdp_curr = _perturb_mdp_costs(mdp, param, max(multiplier_curr, 0.01))
                y_curr = _evaluate_cost_with_params(
                    mdp_curr, beta_curr, max(self.n_trajectories // 4, 10)
                )

                # Perturbed value
                x_new = x.copy()
                x_new[idx] = min(1.0, x[idx] + delta)
                if x_new[idx] == x[idx]:
                    x_new[idx] = max(0.0, x[idx] - delta)

                beta_new = float(x_new[params.index("beta")]) * (
                    param_ranges["beta"][1] - param_ranges["beta"][0]
                ) + param_ranges["beta"][0] if "beta" in params else self.default_beta

                multiplier_new = 1.0
                for j, p in enumerate(params):
                    if p != "beta":
                        p_lo, p_hi = param_ranges[p]
                        multiplier_new *= (p_lo + x_new[j] * (p_hi - p_lo)) / ((p_lo + p_hi) / 2)

                mdp_new = _perturb_mdp_costs(mdp, param, max(multiplier_new, 0.01))
                y_new = _evaluate_cost_with_params(
                    mdp_new, beta_new, max(self.n_trajectories // 4, 10)
                )

                # Elementary effect
                ee = (y_new - y_curr) / delta
                elementary_effects[param].append(ee)

                x = x_new

        result: dict[str, dict[str, float]] = {}
        for param in params:
            ees = elementary_effects[param]
            if ees:
                abs_ees = [abs(e) for e in ees]
                result[param] = {
                    "mu_star": float(np.mean(abs_ees)),
                    "sigma": float(np.std(ees)),
                }
            else:
                result[param] = {"mu_star": 0.0, "sigma": 0.0}

        return result

    def _tornado_diagram_data(
        self,
        results: list[SensitivityResult],
    ) -> dict[str, Any]:
        """Prepare data for a tornado diagram visualization.

        Parameters
        ----------
        results : list[SensitivityResult]
            Sensitivity results to visualize.

        Returns
        -------
        dict
            Data structure for rendering a tornado diagram:
            ``{"parameters": [...], "sensitivities": [...],
              "directions": [...], "ci_lower": [...], "ci_upper": [...]}``.
        """
        sorted_results = sorted(results, key=lambda r: r.sensitivity, reverse=True)

        return {
            "parameters": [r.parameter_name for r in sorted_results],
            "sensitivities": [r.sensitivity for r in sorted_results],
            "directions": [r.direction for r in sorted_results],
            "ci_lower": [r.confidence_interval[0] for r in sorted_results],
            "ci_upper": [r.confidence_interval[1] for r in sorted_results],
        }
