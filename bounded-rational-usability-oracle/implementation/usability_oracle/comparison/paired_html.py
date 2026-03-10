"""Paired HTML comparison for usability regression detection.

Provides a high-level API that takes two HTML strings (before/after),
builds accessibility trees and MDPs for both, aligns them, and runs
the full paired comparison pipeline to detect cognitive cost regressions.

This is the tool's core use case: given two versions of the same component
(e.g., GovUK accordion v3 vs v4), compute the MDP value function difference
and determine whether usability has regressed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from usability_oracle.accessibility.html_to_tree import RealHTMLParser
from usability_oracle.accessibility.models import AccessibilityTree
from usability_oracle.mdp.builder import MDPBuilder, MDPBuilderConfig
from usability_oracle.mdp.models import MDP
from usability_oracle.mdp.solver import ValueIterationSolver
from usability_oracle.comparison.models import (
    AlignmentResult,
    StateMapping,
    ComparisonResult,
)
from usability_oracle.comparison.paired import _solve_softmax_policy, _sample_trajectory_costs
from usability_oracle.cognitive.models import CostElement
from usability_oracle.core.enums import RegressionVerdict


@dataclass
class PairedHTMLResult:
    """Result of comparing two HTML versions.

    Attributes
    ----------
    verdict : str
        "regression", "improvement", "no_change", or "inconclusive"
    cost_before : float
        Mean cognitive cost for version A (value function at initial state)
    cost_after : float
        Mean cognitive cost for version B
    delta : float
        cost_after - cost_before (positive = regression)
    effect_size : float
        Cohen's d effect size
    p_value : float
        Statistical significance
    details : dict
        Full breakdown including per-state comparisons
    """
    verdict: str = "inconclusive"
    cost_before: float = 0.0
    cost_after: float = 0.0
    delta: float = 0.0
    effect_size: float = 0.0
    p_value: float = 1.0
    details: Dict[str, Any] = field(default_factory=dict)


def compare_html_versions(
    html_a: str,
    html_b: str,
    task_spec: Any = None,
    beta: float = 3.0,
    n_trajectories: int = 200,
) -> PairedHTMLResult:
    """Compare two HTML versions for usability regression.

    Parameters
    ----------
    html_a : str
        HTML of the *before* version.
    html_b : str
        HTML of the *after* version.
    task_spec : duck-typed task spec, optional
        If None, a generic browsing task is synthesised.
    beta : float
        Rationality parameter.
    n_trajectories : int
        Monte Carlo trajectories for cost estimation.

    Returns
    -------
    PairedHTMLResult
    """
    parser = RealHTMLParser()
    tree_a = parser.parse(html_a)
    parser_b = RealHTMLParser()
    tree_b = parser_b.parse(html_b)

    # Synthesise a simple task spec if none given
    if task_spec is None:
        task_spec = _make_generic_task(tree_a)

    # Build MDPs
    config = MDPBuilderConfig(
        max_states=5000,
        max_task_progress_bits=4,
        include_read_actions=False,
        include_scroll_actions=False,
    )
    builder = MDPBuilder(config)
    mdp_a = builder.build(tree_a, task_spec)
    mdp_b = builder.build(tree_b, task_spec)

    if not mdp_a.states or not mdp_b.states:
        return PairedHTMLResult(verdict="inconclusive",
                                details={"error": "Empty MDP(s)"})

    # Solve both with bounded-rational policies
    policy_a = _solve_softmax_policy(mdp_a, beta)
    policy_b = _solve_softmax_policy(mdp_b, beta)

    # Value at initial state = expected cognitive cost
    v_a = policy_a.values.get(mdp_a.initial_state, 0.0)
    v_b = policy_b.values.get(mdp_b.initial_state, 0.0)

    # Cost = negative value (higher cost = worse)
    cost_a = -v_a if v_a != 0 else _mean_trajectory_cost(mdp_a, policy_a, n_trajectories)
    cost_b = -v_b if v_b != 0 else _mean_trajectory_cost(mdp_b, policy_b, n_trajectories)

    delta = cost_b - cost_a

    # Sample trajectory costs for statistical test
    samples_a = _sample_trajectory_costs(mdp_a, policy_a, n_trajectories)
    samples_b = _sample_trajectory_costs(mdp_b, policy_b, n_trajectories)

    # Effect size (Cohen's d)
    pooled_std = math.sqrt((np.var(samples_a) + np.var(samples_b)) / 2)
    if pooled_std > 0:
        effect_size = (np.mean(samples_b) - np.mean(samples_a)) / pooled_std
    else:
        effect_size = 0.0

    # Simple t-test approximation for p-value
    from scipy import stats
    try:
        t_stat, p_value = stats.ttest_ind(samples_b, samples_a, equal_var=False)
    except Exception:
        p_value = 1.0

    # Determine verdict
    if p_value < 0.05 and effect_size > 0.2:
        verdict = "regression"
    elif p_value < 0.05 and effect_size < -0.2:
        verdict = "improvement"
    elif abs(effect_size) < 0.2:
        verdict = "no_change"
    else:
        verdict = "inconclusive"

    # Value function difference by state
    vf_diff = {}
    common_states = set(policy_a.values.keys()) & set(policy_b.values.keys())
    for s in list(common_states)[:20]:
        vf_diff[s] = policy_b.values[s] - policy_a.values[s]

    return PairedHTMLResult(
        verdict=verdict,
        cost_before=float(np.mean(samples_a)),
        cost_after=float(np.mean(samples_b)),
        delta=float(np.mean(samples_b) - np.mean(samples_a)),
        effect_size=float(effect_size),
        p_value=float(p_value),
        details={
            "mdp_a_states": len(mdp_a.states),
            "mdp_b_states": len(mdp_b.states),
            "mdp_a_transitions": len(mdp_a.transitions),
            "mdp_b_transitions": len(mdp_b.transitions),
            "value_a_initial": float(v_a),
            "value_b_initial": float(v_b),
            "trajectory_mean_a": float(np.mean(samples_a)),
            "trajectory_mean_b": float(np.mean(samples_b)),
            "trajectory_std_a": float(np.std(samples_a)),
            "trajectory_std_b": float(np.std(samples_b)),
            "value_function_diff_sample": vf_diff,
        },
    )


def _mean_trajectory_cost(
    mdp: MDP, policy: Any, n: int = 100
) -> float:
    """Compute mean trajectory cost."""
    samples = _sample_trajectory_costs(mdp, policy, n)
    return float(np.mean(samples))


class _SimpleSubGoal:
    def __init__(self, target_node_id: str):
        self.target_node_id = target_node_id


class _SimpleTaskSpec:
    """Minimal duck-typed task spec for the MDP builder."""
    def __init__(self, task_id: str, sub_goals: list, target_node_ids: list,
                 description: str = ""):
        self.task_id = task_id
        self.sub_goals = sub_goals
        self.target_node_ids = target_node_ids
        self.description = description


def _make_generic_task(tree: AccessibilityTree) -> _SimpleTaskSpec:
    """Create a generic task that visits all interactive nodes."""
    interactive = tree.get_interactive_nodes()[:3]
    sub_goals = [_SimpleSubGoal(n.id) for n in interactive]
    target_ids = [n.id for n in interactive]
    return _SimpleTaskSpec(
        task_id="generic_browse",
        sub_goals=sub_goals,
        target_node_ids=target_ids,
        description="Visit interactive elements",
    )
