"""
usability_oracle.goms.analyzer — GOMS model analyzer.

Builds GOMS models from task specifications and accessibility trees,
decomposes tasks into Goals → Methods → Operators hierarchy, and
produces execution traces with predictions for execution time, learning
time, and error probability using bounded-rational method selection.

References
----------
Card, S. K., Moran, T. P., & Newell, A. (1983). *The Psychology of
Human-Computer Interaction*. Lawrence Erlbaum Associates.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from usability_oracle.core.types import BoundingBox, Point2D
from usability_oracle.core.constants import (
    HICK_A_RANGE,
    HICK_B_RANGE,
    WORKING_MEMORY_CAPACITY,
)
from usability_oracle.goms.types import (
    GomsGoal,
    GomsMethod,
    GomsModel,
    GomsOperator,
    GomsTrace,
    KLMSequence,
    OperatorType,
)
from usability_oracle.goms.klm import (
    KLMConfig,
    SkillLevel,
    apply_placement_rules,
    build_klm_sequence,
    compute_sequence_time,
    make_mental,
)
from usability_oracle.goms.decomposition import (
    decompose_task,
    detect_goal_conflicts,
)


# ═══════════════════════════════════════════════════════════════════════════
# AnalyzerConfig
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AnalyzerConfig:
    """Configuration for the GOMS analyzer."""

    klm_config: KLMConfig = field(default_factory=KLMConfig)
    rationality_beta: float = 5.0
    """Bounded rationality parameter. Higher → more optimal selection."""
    learning_rate_factor: float = 0.4
    """Power-law-of-practice exponent for learning time estimation."""
    base_error_rate: float = 0.02
    """Base per-operator error probability."""
    expert_overlap: float = 0.3
    """Fraction of cognitive–motor overlap for expert models."""


# ═══════════════════════════════════════════════════════════════════════════
# Method selection (bounded rational)
# ═══════════════════════════════════════════════════════════════════════════

def _softmax_select(
    methods: Sequence[GomsMethod],
    beta: float,
    rng: Optional[np.random.Generator] = None,
) -> GomsMethod:
    """Select a method using softmax (Luce choice) with bounded rationality.

    P(m_i) ∝ exp(-β * cost(m_i))

    A perfectly rational agent (β → ∞) always picks the fastest method.
    A bounded-rational agent (finite β) sometimes picks sub-optimal ones.

    Parameters
    ----------
    methods : Sequence[GomsMethod]
        Available methods.
    beta : float
        Rationality parameter.
    rng : Optional[np.random.Generator]
        Random generator (deterministic default).

    Returns
    -------
    GomsMethod
        Selected method.
    """
    if not methods:
        raise ValueError("No methods available for selection")
    if len(methods) == 1:
        return methods[0]

    costs = np.array([m.total_duration_s for m in methods])
    # Shift for numerical stability
    shifted = -beta * (costs - costs.min())
    exp_vals = np.exp(shifted)
    probs = exp_vals / exp_vals.sum()

    gen = rng or np.random.default_rng(42)
    idx = int(gen.choice(len(methods), p=probs))
    return methods[idx]


def _deterministic_select(
    methods: Sequence[GomsMethod],
    policy: Optional[Mapping[str, str]],
    goal_id: str,
) -> Optional[GomsMethod]:
    """Select a method using an explicit policy mapping."""
    if policy and goal_id in policy:
        preferred = policy[goal_id]
        for m in methods:
            if m.method_id == preferred:
                return m
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Execution time prediction
# ═══════════════════════════════════════════════════════════════════════════

def predict_execution_time(
    methods: Sequence[GomsMethod],
    *,
    config: AnalyzerConfig = AnalyzerConfig(),
) -> float:
    """Predict total execution time from selected methods.

    For expert users, applies cognitive–motor overlap.
    """
    total = 0.0
    for method in methods:
        serial_time = method.total_duration_s
        if config.klm_config.skill_level == SkillLevel.EXPERT:
            # Apply overlap between M and motor operators
            overlap_time = 0.0
            ops = method.operators
            for i, op in enumerate(ops):
                if op.op_type == OperatorType.M and i > 0:
                    prev = ops[i - 1]
                    if prev.is_motor:
                        overlap_time += config.expert_overlap * min(
                            op.duration_s, prev.duration_s
                        )
            total += serial_time - overlap_time
        else:
            total += serial_time
    return total


# ═══════════════════════════════════════════════════════════════════════════
# Learning time estimation
# ═══════════════════════════════════════════════════════════════════════════

def estimate_learning_time(
    model: GomsModel,
    *,
    config: AnalyzerConfig = AnalyzerConfig(),
) -> float:
    """Estimate time to learn all methods in a GOMS model.

    Uses a power-law-of-practice model:
        T_learn = Σ_methods (n_operators * base_op_learn_time * method_complexity)

    Method complexity accounts for the number of unique operator types,
    decision points (selection rules), and goal depth.

    Returns
    -------
    float
        Estimated learning time in seconds.
    """
    base_learn_per_op = 2.0  # seconds to learn one operator in context
    total = 0.0

    for method in model.methods:
        n_ops = method.operator_count
        unique_types = len(set(op.op_type for op in method.operators))
        complexity_factor = 1.0 + 0.2 * (unique_types - 1)

        # If method has a selection rule, add decision learning cost
        if method.selection_rule:
            complexity_factor += 0.3

        method_learn = n_ops * base_learn_per_op * complexity_factor
        total += method_learn

    # Apply power-law: first session takes full time, subsequent decay
    return total * (1.0 + config.learning_rate_factor)


# ═══════════════════════════════════════════════════════════════════════════
# Error probability estimation
# ═══════════════════════════════════════════════════════════════════════════

def estimate_error_probability(
    methods: Sequence[GomsMethod],
    *,
    config: AnalyzerConfig = AnalyzerConfig(),
) -> float:
    """Estimate overall error probability for a sequence of methods.

    Model: P(error) = 1 - Π(1 - p_i) where p_i depends on operator
    type and working memory load.

    Motor errors are base rate; cognitive errors scale with WM load.
    """
    p_no_error = 1.0
    wm_chunks = 0.0
    wm_capacity = WORKING_MEMORY_CAPACITY.midpoint

    for method in methods:
        for op in method.operators:
            # Base error rate per operator
            p_op_error = config.base_error_rate

            if op.op_type == OperatorType.M:
                # Mental operators: error scales with WM load
                wm_chunks += 1.0
                wm_load_factor = min(wm_chunks / wm_capacity, 2.0)
                p_op_error *= (1.0 + wm_load_factor)
            elif op.op_type == OperatorType.K:
                # Keystroke errors depend on skill
                if config.klm_config.skill_level == SkillLevel.NOVICE:
                    p_op_error *= 2.0
                elif config.klm_config.skill_level == SkillLevel.EXPERT:
                    p_op_error *= 0.5
            elif op.op_type == OperatorType.P:
                # Pointing errors: smaller targets → more errors
                fitts_id = op.parameters.get("fitts_id", 3.0)
                p_op_error *= (1.0 + 0.1 * fitts_id)
            elif op.op_type == OperatorType.R:
                # System response: user may lose context
                if op.duration_s > 2.0:
                    wm_chunks = max(0, wm_chunks - 1.0)
                    p_op_error *= 1.5

            p_op_error = min(p_op_error, 1.0)
            p_no_error *= (1.0 - p_op_error)

    return 1.0 - p_no_error


# ═══════════════════════════════════════════════════════════════════════════
# Critical path through operator DAG
# ═══════════════════════════════════════════════════════════════════════════

def _build_operator_dag(
    methods: Sequence[GomsMethod],
) -> Tuple[List[GomsOperator], Dict[int, List[int]]]:
    """Build a DAG of operators across all methods.

    Operators within a method are sequential. Methods that share
    no target resources can execute in parallel (for CPM-GOMS).

    Returns
    -------
    tuple[list[GomsOperator], dict[int, list[int]]]
        Flat operator list and adjacency list (successors).
    """
    all_ops: List[GomsOperator] = []
    adj: Dict[int, List[int]] = {}
    method_ranges: List[Tuple[int, int]] = []

    offset = 0
    for method in methods:
        start = offset
        for i, op in enumerate(method.operators):
            idx = offset + i
            all_ops.append(op)
            adj[idx] = []
            if i > 0:
                adj[offset + i - 1].append(idx)
        end = offset + len(method.operators)
        method_ranges.append((start, end))
        offset = end

    # Add cross-method dependencies for shared targets
    target_to_methods: Dict[str, List[int]] = {}
    for mi, method in enumerate(methods):
        for op in method.operators:
            if op.target_id:
                target_to_methods.setdefault(op.target_id, []).append(mi)

    for _target, method_indices in target_to_methods.items():
        unique = sorted(set(method_indices))
        for k in range(len(unique) - 1):
            mi_a = unique[k]
            mi_b = unique[k + 1]
            end_a = method_ranges[mi_a][1] - 1
            start_b = method_ranges[mi_b][0]
            if end_a >= 0 and start_b < len(all_ops):
                adj.setdefault(end_a, []).append(start_b)

    return all_ops, adj


def compute_critical_path_time(
    methods: Sequence[GomsMethod],
) -> float:
    """Compute the critical path time through the operator DAG.

    Uses longest-path computation on the DAG (topological order).

    Returns
    -------
    float
        Critical path time in seconds.
    """
    if not methods:
        return 0.0

    ops, adj = _build_operator_dag(methods)
    n = len(ops)
    if n == 0:
        return 0.0

    # Compute in-degree
    in_degree = [0] * n
    for u, successors in adj.items():
        for v in successors:
            in_degree[v] += 1

    # Topological sort (Kahn's algorithm)
    from collections import deque
    queue: deque[int] = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)

    dist = [0.0] * n
    for i in range(n):
        dist[i] = ops[i].duration_s

    topo_order: List[int] = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in adj.get(u, []):
            candidate = dist[u] + ops[v].duration_s
            if candidate > dist[v]:
                dist[v] = candidate
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    return max(dist) if dist else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Operator merging for expert models
# ═══════════════════════════════════════════════════════════════════════════

def merge_operators_expert(
    methods: Sequence[GomsMethod],
) -> Tuple[GomsMethod, ...]:
    """Merge operators for expert performance model.

    Experts chunk consecutive motor operators and reduce mental
    preparation for familiar sequences.

    - Consecutive K operators are merged into a single K with summed duration.
    - M operators before chunked sequences are halved.

    Returns
    -------
    tuple[GomsMethod, ...]
        Methods with merged operators.
    """
    merged_methods: List[GomsMethod] = []
    for method in methods:
        merged_ops: List[GomsOperator] = []
        i = 0
        ops = method.operators
        while i < len(ops):
            # Check for a run of K operators
            if ops[i].op_type == OperatorType.K:
                run_start = i
                total_dur = 0.0
                while i < len(ops) and ops[i].op_type == OperatorType.K:
                    total_dur += ops[i].duration_s
                    i += 1
                run_len = i - run_start
                if run_len > 1:
                    # Halve the preceding M if it exists
                    if (
                        merged_ops
                        and merged_ops[-1].op_type == OperatorType.M
                    ):
                        old_m = merged_ops[-1]
                        merged_ops[-1] = GomsOperator(
                            op_type=OperatorType.M,
                            duration_s=old_m.duration_s * 0.5,
                            description=f"{old_m.description} (expert chunk)",
                        )
                    # Single merged K
                    merged_ops.append(GomsOperator(
                        op_type=OperatorType.K,
                        duration_s=total_dur,
                        description=f"Chunked {run_len} keystrokes",
                        parameters={"chunk_size": run_len},
                    ))
                else:
                    merged_ops.append(ops[run_start])
            else:
                merged_ops.append(ops[i])
                i += 1

        merged_methods.append(GomsMethod(
            method_id=method.method_id,
            goal_id=method.goal_id,
            name=method.name,
            operators=tuple(merged_ops),
            selection_rule=method.selection_rule,
        ))
    return tuple(merged_methods)


# ═══════════════════════════════════════════════════════════════════════════
# GomsAnalyzerImpl — concrete analyzer implementation
# ═══════════════════════════════════════════════════════════════════════════

class GomsAnalyzerImpl:
    """Concrete implementation of the GomsAnalyzer protocol.

    Builds GOMS models from accessibility trees, produces execution
    traces, and compares traces for regression detection.
    """

    def __init__(self, config: Optional[AnalyzerConfig] = None) -> None:
        self._config = config or AnalyzerConfig()
        self._rng = np.random.default_rng(42)

    @property
    def config(self) -> AnalyzerConfig:
        return self._config

    def build_model(
        self,
        tree: Any,
        task_description: str,
        *,
        top_level_goal: Optional[str] = None,
    ) -> GomsModel:
        """Build a GOMS model from an accessibility tree and task description."""
        model = decompose_task(
            tree,
            task_description,
            config=self._config.klm_config,
            top_level_goal=top_level_goal,
        )

        # For expert users, apply operator merging
        if self._config.klm_config.skill_level == SkillLevel.EXPERT:
            merged = merge_operators_expert(model.methods)
            model = GomsModel(
                model_id=model.model_id,
                name=model.name,
                goals=model.goals,
                methods=merged,
                top_level_goal_id=model.top_level_goal_id,
            )
        return model

    def trace(
        self,
        model: GomsModel,
        *,
        selection_policy: Optional[Mapping[str, str]] = None,
    ) -> GomsTrace:
        """Execute a GOMS model trace using bounded-rational method selection."""
        selected: List[GomsMethod] = []
        goals_used: List[GomsGoal] = []

        # Process each leaf goal
        for goal in model.goals:
            if goal.is_leaf and goal.goal_id != model.top_level_goal_id:
                available = model.methods_for_goal(goal.goal_id)
                if not available:
                    continue

                # Try explicit policy first, then bounded-rational selection
                chosen = _deterministic_select(
                    available, selection_policy, goal.goal_id,
                )
                if chosen is None:
                    chosen = _softmax_select(
                        available,
                        self._config.rationality_beta,
                        self._rng,
                    )
                selected.append(chosen)
                goals_used.append(goal)

        exec_time = predict_execution_time(selected, config=self._config)
        crit_time = compute_critical_path_time(selected)
        error_prob = estimate_error_probability(selected, config=self._config)
        learn_time = estimate_learning_time(model, config=self._config)

        return GomsTrace(
            trace_id=f"trace-{uuid.uuid4().hex[:8]}",
            task_name=model.name,
            goals=tuple(goals_used),
            methods_selected=tuple(selected),
            total_time_s=exec_time,
            critical_path_time_s=crit_time,
            metadata={
                "error_probability": error_prob,
                "learning_time_s": learn_time,
                "skill_level": self._config.klm_config.skill_level.value,
                "rationality_beta": self._config.rationality_beta,
                "model_id": model.model_id,
            },
        )

    def compare_traces(
        self,
        trace_old: GomsTrace,
        trace_new: GomsTrace,
    ) -> Mapping[str, Any]:
        """Compare two traces for regression detection."""
        time_delta = trace_new.total_time_s - trace_old.total_time_s
        op_delta = trace_new.total_operator_count - trace_old.total_operator_count
        crit_delta = (
            trace_new.critical_path_time_s - trace_old.critical_path_time_s
        )

        # Compute error probability delta
        err_old = trace_old.metadata.get("error_probability", 0.0)
        err_new = trace_new.metadata.get("error_probability", 0.0)
        err_delta = err_new - err_old

        # A regression is any meaningful increase in time or error
        regression = time_delta > 0.5 or err_delta > 0.05

        return {
            "time_delta_s": time_delta,
            "time_delta_pct": (
                (time_delta / trace_old.total_time_s * 100.0)
                if trace_old.total_time_s > 0 else 0.0
            ),
            "operator_count_delta": op_delta,
            "critical_path_delta_s": crit_delta,
            "error_probability_delta": err_delta,
            "regression": regression,
            "old_trace_id": trace_old.trace_id,
            "new_trace_id": trace_new.trace_id,
        }


__all__ = [
    "AnalyzerConfig",
    "GomsAnalyzerImpl",
    "compute_critical_path_time",
    "estimate_error_probability",
    "estimate_learning_time",
    "merge_operators_expert",
    "predict_execution_time",
]
