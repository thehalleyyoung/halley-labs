"""
usability_oracle.fragility.inclusive — Inclusive design analysis.

Implements :class:`InclusiveDesignAnalyzer`, which evaluates how well
a UI design serves diverse user populations by computing per-profile
task costs for different cognitive/motor capability profiles.

Population profiles represent distinct user groups:
- Motor-impaired users (low β, high Fitts' costs)
- Cognitively-impaired users (very low β)
- Elderly users (moderate β, slower motor execution)
- Novice users (low β, unfamiliar with interface)
- Expert users (high β, familiar with interface)

An *equitable* design has similar costs across all profiles.  A large
equity gap indicates that the design is exclusionary.

References
----------
- Wobbrock, J. O. et al. (2011). Ability-based design: Concept,
  principles, and examples. *ACM TOCHI*, 18(3).
- Stephanidis, C. (2001). *User Interfaces for All*. Lawrence Erlbaum.
- W3C (2018). *Web Content Accessibility Guidelines (WCAG) 2.1*.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from usability_oracle.core.enums import BottleneckType
from usability_oracle.cognitive.models import CostElement
from usability_oracle.mdp.models import MDP
from usability_oracle.fragility.models import InclusiveDesignResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default population profiles
# ---------------------------------------------------------------------------

DEFAULT_POPULATION_PROFILES: dict[str, dict[str, Any]] = {
    "motor_impaired": {
        "description": "Users with motor impairments (e.g., tremor, limited range)",
        "beta": 2.0,
        "motor_cost_multiplier": 3.0,
        "perceptual_cost_multiplier": 1.0,
        "cognitive_cost_multiplier": 1.0,
        "population_fraction": 0.07,
    },
    "cognitively_impaired": {
        "description": "Users with cognitive impairments (e.g., learning disability)",
        "beta": 0.5,
        "motor_cost_multiplier": 1.0,
        "perceptual_cost_multiplier": 1.0,
        "cognitive_cost_multiplier": 2.5,
        "population_fraction": 0.05,
    },
    "elderly": {
        "description": "Elderly users (65+, slower motor/cognitive processing)",
        "beta": 1.5,
        "motor_cost_multiplier": 1.8,
        "perceptual_cost_multiplier": 1.5,
        "cognitive_cost_multiplier": 1.5,
        "population_fraction": 0.16,
    },
    "novice": {
        "description": "Novice users (unfamiliar with interface conventions)",
        "beta": 1.0,
        "motor_cost_multiplier": 1.2,
        "perceptual_cost_multiplier": 1.3,
        "cognitive_cost_multiplier": 1.5,
        "population_fraction": 0.20,
    },
    "average": {
        "description": "Average adult user",
        "beta": 3.0,
        "motor_cost_multiplier": 1.0,
        "perceptual_cost_multiplier": 1.0,
        "cognitive_cost_multiplier": 1.0,
        "population_fraction": 0.40,
    },
    "expert": {
        "description": "Expert users (highly familiar with interface)",
        "beta": 10.0,
        "motor_cost_multiplier": 0.8,
        "perceptual_cost_multiplier": 0.8,
        "cognitive_cost_multiplier": 0.7,
        "population_fraction": 0.12,
    },
}


# ---------------------------------------------------------------------------
# Helper: create a cost-scaled MDP
# ---------------------------------------------------------------------------

def _scale_mdp_costs(
    mdp: MDP,
    multiplier: float,
) -> MDP:
    """Create a new MDP with all transition costs scaled by *multiplier*.

    Parameters
    ----------
    mdp : MDP
    multiplier : float

    Returns
    -------
    MDP
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


def _evaluate_profile_cost(
    mdp: MDP,
    profile: dict[str, Any],
    n_trajectories: int = 100,
) -> float:
    """Evaluate expected task cost for a specific user profile.

    Scales the MDP costs according to the profile's multipliers and
    solves at the profile's β.

    Parameters
    ----------
    mdp : MDP
    profile : dict
    n_trajectories : int

    Returns
    -------
    float
        Mean task completion cost.
    """
    from usability_oracle.comparison.paired import (
        _solve_softmax_policy,
        _sample_trajectory_costs,
    )

    # Compute an aggregate cost multiplier from the profile
    motor_m = profile.get("motor_cost_multiplier", 1.0)
    perceptual_m = profile.get("perceptual_cost_multiplier", 1.0)
    cognitive_m = profile.get("cognitive_cost_multiplier", 1.0)
    avg_multiplier = (motor_m + perceptual_m + cognitive_m) / 3.0

    scaled_mdp = _scale_mdp_costs(mdp, avg_multiplier)
    beta = profile.get("beta", 3.0)

    policy = _solve_softmax_policy(scaled_mdp, beta)
    samples = _sample_trajectory_costs(scaled_mdp, policy, n_trajectories)
    return float(np.mean(samples))


# ---------------------------------------------------------------------------
# InclusiveDesignAnalyzer
# ---------------------------------------------------------------------------


class InclusiveDesignAnalyzer:
    """Analyzes how inclusively a UI serves diverse user populations.

    For each population profile, evaluates the expected task cost
    under that profile's cognitive/motor constraints, then computes
    equity metrics and identifies exclusionary elements.

    Parameters
    ----------
    n_trajectories : int
        Monte Carlo trajectories per profile evaluation.
    cost_threshold : float
        Maximum acceptable task cost (seconds).  Profiles exceeding
        this are considered "excluded".
    population_profiles : dict, optional
        Custom population profiles.  Defaults to built-in profiles.
    """

    def __init__(
        self,
        n_trajectories: int = 100,
        cost_threshold: float = 30.0,
        population_profiles: Optional[dict[str, dict[str, Any]]] = None,
    ) -> None:
        self.n_trajectories = n_trajectories
        self.cost_threshold = cost_threshold
        self.profiles = population_profiles or DEFAULT_POPULATION_PROFILES

    def analyze(
        self,
        mdp: MDP,
        population_profiles: Optional[dict[str, dict[str, Any]]] = None,
    ) -> InclusiveDesignResult:
        """Run a full inclusive design analysis.

        Parameters
        ----------
        mdp : MDP
            The MDP to analyze.
        population_profiles : dict, optional
            Override population profiles.

        Returns
        -------
        InclusiveDesignResult
        """
        profiles = population_profiles or self.profiles

        logger.info(
            "Starting inclusive design analysis with %d profiles", len(profiles)
        )

        # 1. Compute per-profile costs
        per_profile: dict[str, float] = {}
        per_profile_elements: dict[str, CostElement] = {}
        for name, profile in profiles.items():
            cost = self._compute_per_profile_cost(mdp, profile)
            per_profile[name] = cost.mean_time
            per_profile_elements[name] = cost

        # 2. Equity gap
        gap = self._equity_gap(per_profile_elements)

        # 3. Identify most/least affected
        if per_profile:
            most_affected = max(per_profile, key=per_profile.get)
            least_affected = min(per_profile, key=per_profile.get)
        else:
            most_affected = ""
            least_affected = ""

        # 4. Population coverage (fraction who can complete within threshold)
        coverage = self._population_coverage(per_profile, profiles)

        # 5. Identify exclusive elements per profile
        exclusive: dict[str, list[str]] = {}
        for name, profile in profiles.items():
            elements = self._identify_exclusive_elements(mdp, profile)
            if elements:
                exclusive[name] = elements

        # 6. Accessibility scores
        accessibility_scores: dict[str, float] = {}
        for name, profile in profiles.items():
            accessibility_scores[name] = self._accessibility_score(mdp, profile)

        # 7. Population percentile costs
        percentile_costs = self._population_percentile_costs(
            mdp, [5, 25, 50, 75, 95], profiles
        )

        # 8. Generate recommendations
        recommendations = self._generate_recommendations(
            per_profile, gap, exclusive, coverage, profiles
        )

        return InclusiveDesignResult(
            per_profile_costs=per_profile,
            equity_gap=gap,
            most_affected_profile=most_affected,
            least_affected_profile=least_affected,
            recommendations=recommendations,
            population_coverage=coverage,
            exclusive_elements=exclusive,
        )

    def _compute_per_profile_cost(
        self,
        mdp: MDP,
        profile: dict[str, Any],
    ) -> CostElement:
        """Compute the cost for a specific user profile.

        Parameters
        ----------
        mdp : MDP
        profile : dict

        Returns
        -------
        CostElement
        """
        from usability_oracle.comparison.paired import (
            _solve_softmax_policy,
            _sample_trajectory_costs,
        )

        motor_m = profile.get("motor_cost_multiplier", 1.0)
        perceptual_m = profile.get("perceptual_cost_multiplier", 1.0)
        cognitive_m = profile.get("cognitive_cost_multiplier", 1.0)
        avg_multiplier = (motor_m + perceptual_m + cognitive_m) / 3.0

        scaled_mdp = _scale_mdp_costs(mdp, avg_multiplier)
        beta = profile.get("beta", 3.0)

        policy = _solve_softmax_policy(scaled_mdp, beta)
        samples = _sample_trajectory_costs(
            scaled_mdp, policy, self.n_trajectories
        )

        mean_cost = float(np.mean(samples))
        var_cost = float(np.var(samples, ddof=1)) if len(samples) > 1 else 0.0

        return CostElement(
            mean_time=mean_cost,
            variance=var_cost,
            channel="aggregate",
            law="composite",
        )

    def _equity_gap(self, costs: dict[str, CostElement]) -> float:
        """Compute the equity gap: maximum cost difference between profiles.

        .. math::
            G = \\max_{i,j} |C_i - C_j|

        A gap of 0 means perfect equity; larger values indicate
        that some user groups are significantly disadvantaged.

        Parameters
        ----------
        costs : dict[str, CostElement]

        Returns
        -------
        float
        """
        if len(costs) < 2:
            return 0.0

        values = [c.mean_time for c in costs.values()]
        return float(max(values) - min(values))

    def _accessibility_score(
        self,
        mdp: MDP,
        profile: dict[str, Any],
    ) -> float:
        """Compute an accessibility score for the MDP given a profile.

        The score is in [0, 1] where 1 = fully accessible.  Based on:
        - Fraction of states reachable under the profile's policy
        - Whether the goal is reachable within the cost threshold

        Parameters
        ----------
        mdp : MDP
        profile : dict

        Returns
        -------
        float
        """
        from usability_oracle.comparison.paired import (
            _solve_softmax_policy,
            _sample_trajectory_costs,
        )

        beta = profile.get("beta", 3.0)
        motor_m = profile.get("motor_cost_multiplier", 1.0)
        perceptual_m = profile.get("perceptual_cost_multiplier", 1.0)
        cognitive_m = profile.get("cognitive_cost_multiplier", 1.0)
        avg_multiplier = (motor_m + perceptual_m + cognitive_m) / 3.0

        scaled_mdp = _scale_mdp_costs(mdp, avg_multiplier)
        policy = _solve_softmax_policy(scaled_mdp, beta)

        # Check how many states the policy can reach goal from
        reachable = scaled_mdp.reachable_states()
        n_total = max(len(scaled_mdp.states), 1)
        reachability_score = len(reachable) / n_total

        # Check if average cost is within threshold
        samples = _sample_trajectory_costs(scaled_mdp, policy, 50)
        mean_cost = float(np.mean(samples))
        cost_score = 1.0 - min(mean_cost / self.cost_threshold, 1.0)

        return (reachability_score + cost_score) / 2.0

    def _population_percentile_costs(
        self,
        mdp: MDP,
        percentiles: list[int],
        profiles: dict[str, dict[str, Any]],
    ) -> dict[str, float]:
        """Compute cost at population percentiles.

        Maps each percentile to a β value based on the profiles'
        population fractions, then evaluates the cost.

        Parameters
        ----------
        mdp : MDP
        percentiles : list[int]
        profiles : dict

        Returns
        -------
        dict[str, float]
            ``{"p5": cost, "p25": cost, ...}``.
        """
        # Sort profiles by beta (ascending = least capable first)
        sorted_profiles = sorted(
            profiles.items(),
            key=lambda x: x[1].get("beta", 3.0),
        )

        # Build CDF of population fractions
        cum_fractions: list[float] = []
        betas: list[float] = []
        cumsum = 0.0
        for name, profile in sorted_profiles:
            cumsum += profile.get("population_fraction", 0.1)
            cum_fractions.append(cumsum)
            betas.append(profile.get("beta", 3.0))

        # Normalize
        if cumsum > 0:
            cum_fractions = [f / cumsum for f in cum_fractions]

        result: dict[str, float] = {}
        for pct in percentiles:
            target = pct / 100.0
            # Find the beta for this percentile
            beta_pct = betas[0]  # default
            for i, cf in enumerate(cum_fractions):
                if cf >= target:
                    beta_pct = betas[i]
                    break
            else:
                beta_pct = betas[-1]

            cost = _evaluate_profile_cost(
                mdp, {"beta": beta_pct}, max(self.n_trajectories // 2, 20)
            )
            result[f"p{pct}"] = cost

        return result

    def _identify_exclusive_elements(
        self,
        mdp: MDP,
        profile: dict[str, Any],
    ) -> list[str]:
        """Identify MDP states that are exclusionary for this profile.

        A state is *exclusionary* if the expected cost from that state
        under the profile's policy exceeds the threshold, effectively
        blocking the user from completing the task.

        Parameters
        ----------
        mdp : MDP
        profile : dict

        Returns
        -------
        list[str]
            State IDs that are exclusionary for this profile.
        """
        from usability_oracle.comparison.paired import _solve_softmax_policy

        beta = profile.get("beta", 3.0)
        motor_m = profile.get("motor_cost_multiplier", 1.0)
        perceptual_m = profile.get("perceptual_cost_multiplier", 1.0)
        cognitive_m = profile.get("cognitive_cost_multiplier", 1.0)
        avg_multiplier = (motor_m + perceptual_m + cognitive_m) / 3.0

        scaled_mdp = _scale_mdp_costs(mdp, avg_multiplier)
        policy = _solve_softmax_policy(scaled_mdp, beta)

        exclusive_states: list[str] = []
        for sid in scaled_mdp.states:
            if sid in scaled_mdp.goal_states:
                continue
            s_obj = scaled_mdp.states[sid]
            if s_obj.is_terminal:
                continue

            # Check if this state has viable actions
            probs = policy.action_probs(sid)
            if not probs:
                exclusive_states.append(sid)
                continue

            # Check if the policy is too entropic (confused)
            entropy = policy.entropy(sid)
            max_ent = policy.max_entropy(sid)
            if max_ent > 0 and entropy / max_ent > 0.95:
                exclusive_states.append(sid)
                continue

            # Check if all transitions lead to high-cost paths
            mean_cost = 0.0
            for a, p in probs.items():
                for t_state, t_prob, t_cost in scaled_mdp.get_transitions(sid, a):
                    mean_cost += p * t_prob * t_cost

            if mean_cost > self.cost_threshold * 0.5:
                exclusive_states.append(sid)

        return exclusive_states

    def _population_coverage(
        self,
        per_profile_costs: dict[str, float],
        profiles: dict[str, dict[str, Any]],
    ) -> float:
        """Compute the population fraction that can complete the task.

        A profile is "covered" if its mean cost is within the threshold.

        Parameters
        ----------
        per_profile_costs : dict[str, float]
        profiles : dict

        Returns
        -------
        float
            Population coverage fraction in [0, 1].
        """
        total_pop = 0.0
        covered_pop = 0.0

        for name, cost in per_profile_costs.items():
            pop_frac = profiles.get(name, {}).get("population_fraction", 0.1)
            total_pop += pop_frac
            if cost <= self.cost_threshold:
                covered_pop += pop_frac

        if total_pop < 1e-12:
            return 1.0
        return covered_pop / total_pop

    def _generate_recommendations(
        self,
        per_profile: dict[str, float],
        equity_gap: float,
        exclusive: dict[str, list[str]],
        coverage: float,
        profiles: dict[str, dict[str, Any]],
    ) -> list[str]:
        """Generate actionable inclusive design recommendations.

        Parameters
        ----------
        per_profile : dict[str, float]
        equity_gap : float
        exclusive : dict[str, list[str]]
        coverage : float
        profiles : dict

        Returns
        -------
        list[str]
        """
        recs: list[str] = []

        if coverage < 0.9:
            recs.append(
                f"Population coverage is {coverage * 100:.0f}% — "
                f"below the 90% target.  Simplify critical interaction paths."
            )

        if equity_gap > 5.0:
            worst = max(per_profile, key=per_profile.get) if per_profile else ""
            best = min(per_profile, key=per_profile.get) if per_profile else ""
            recs.append(
                f"Large equity gap ({equity_gap:.1f}s) between "
                f"'{worst}' and '{best}' profiles.  "
                "Consider adding alternative interaction paths."
            )

        for name, elements in exclusive.items():
            profile_desc = profiles.get(name, {}).get("description", name)
            recs.append(
                f"Profile '{name}' ({profile_desc}) encounters "
                f"{len(elements)} exclusionary state(s): {elements[:3]}. "
                "Consider providing accessible alternatives."
            )

        # Check for motor-impaired-specific issues
        motor_cost = per_profile.get("motor_impaired", 0.0)
        avg_cost = per_profile.get("average", 0.0)
        if motor_cost > avg_cost * 2:
            recs.append(
                "Motor-impaired users face >2× cost.  "
                "Increase target sizes and reduce required pointing actions."
            )

        # Check for cognitive-impaired-specific issues
        cog_cost = per_profile.get("cognitively_impaired", 0.0)
        if cog_cost > avg_cost * 2.5:
            recs.append(
                "Cognitively-impaired users face >2.5× cost.  "
                "Reduce the number of choices per screen and provide "
                "progressive disclosure."
            )

        if not recs:
            recs.append(
                "Good: the design appears inclusive across all tested profiles."
            )

        return recs
