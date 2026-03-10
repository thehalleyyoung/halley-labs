"""
usability_oracle.goms.klm — Keystroke-Level Model implementation.

Full KLM operator set with Fitts' law integration, Hick-Hyman mental
preparation adjustment, Card/Moran/Newell M-operator placement rules,
and skill-level calibration.

References
----------
Card, S. K., Moran, T. P., & Newell, A. (1980). The keystroke-level model
for user performance time with interactive systems. *Communications of the
ACM*, 23(7), 396-410.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from usability_oracle.core.types import BoundingBox, Point2D
from usability_oracle.core.constants import (
    FITTS_A_RANGE,
    FITTS_B_RANGE,
    HICK_A_RANGE,
    HICK_B_RANGE,
    PIXEL_TO_MM,
)
from usability_oracle.goms.types import (
    GomsOperator,
    KLMSequence,
    OperatorType,
)


# ═══════════════════════════════════════════════════════════════════════════
# Skill level
# ═══════════════════════════════════════════════════════════════════════════

@unique
class SkillLevel(Enum):
    """User skill level affecting KLM operator durations."""

    NOVICE = "novice"
    INTERMEDIATE = "intermediate"
    EXPERT = "expert"


# Per-skill keystroke durations (Card et al. 1980, Table 2)
_K_DURATIONS: Dict[SkillLevel, float] = {
    SkillLevel.NOVICE: 0.40,       # hunt-and-peck
    SkillLevel.INTERMEDIATE: 0.28,  # average typist
    SkillLevel.EXPERT: 0.12,        # expert typist (90+ WPM)
}

_H_DURATIONS: Dict[SkillLevel, float] = {
    SkillLevel.NOVICE: 0.50,
    SkillLevel.INTERMEDIATE: 0.40,
    SkillLevel.EXPERT: 0.30,
}

_M_DURATIONS: Dict[SkillLevel, float] = {
    SkillLevel.NOVICE: 1.50,
    SkillLevel.INTERMEDIATE: 1.35,
    SkillLevel.EXPERT: 1.10,
}

_B_DURATIONS: Dict[SkillLevel, float] = {
    SkillLevel.NOVICE: 0.12,
    SkillLevel.INTERMEDIATE: 0.10,
    SkillLevel.EXPERT: 0.08,
}


# ═══════════════════════════════════════════════════════════════════════════
# KLMConfig — calibration parameters
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class KLMConfig:
    """Configuration for KLM operator timing and placement rules."""

    skill_level: SkillLevel = SkillLevel.INTERMEDIATE
    fitts_a: float = FITTS_A_RANGE.midpoint
    fitts_b: float = FITTS_B_RANGE.midpoint
    hick_a: float = HICK_A_RANGE.midpoint
    hick_b: float = HICK_B_RANGE.midpoint
    system_response_s: float = 0.30
    overlap_fraction: float = 0.0
    """Fraction [0,1] of cognitive–motor overlap for expert models."""
    pixel_to_mm: float = PIXEL_TO_MM

    @property
    def k_duration(self) -> float:
        return _K_DURATIONS[self.skill_level]

    @property
    def h_duration(self) -> float:
        return _H_DURATIONS[self.skill_level]

    @property
    def m_duration(self) -> float:
        return _M_DURATIONS[self.skill_level]

    @property
    def b_duration(self) -> float:
        return _B_DURATIONS[self.skill_level]


# ═══════════════════════════════════════════════════════════════════════════
# Fitts' law helper
# ═══════════════════════════════════════════════════════════════════════════

def fitts_time(
    distance_px: float,
    target_width_px: float,
    config: KLMConfig,
) -> float:
    """Compute Fitts' law movement time (Shannon formulation).

    MT = a + b * log2(D / W + 1)

    Parameters
    ----------
    distance_px : float
        Distance from starting position to target centre (pixels).
    target_width_px : float
        Target width along movement axis (pixels).
    config : KLMConfig
        Fitts' law coefficients.

    Returns
    -------
    float
        Movement time in seconds.
    """
    if target_width_px <= 0:
        target_width_px = 1.0
    d_mm = distance_px * config.pixel_to_mm
    w_mm = target_width_px * config.pixel_to_mm
    index_of_difficulty = math.log2(d_mm / w_mm + 1.0)
    return config.fitts_a + config.fitts_b * index_of_difficulty


def hick_hyman_time(n_choices: int, config: KLMConfig) -> float:
    """Compute Hick-Hyman choice reaction time.

    RT = a + b * log2(n + 1)

    Parameters
    ----------
    n_choices : int
        Number of equiprobable alternatives.
    config : KLMConfig
        Hick-Hyman coefficients.

    Returns
    -------
    float
        Decision time in seconds.
    """
    if n_choices < 1:
        return config.hick_a
    return config.hick_a + config.hick_b * math.log2(n_choices + 1)


# ═══════════════════════════════════════════════════════════════════════════
# Operator constructors
# ═══════════════════════════════════════════════════════════════════════════

def make_keystroke(
    key: str = "",
    *,
    config: KLMConfig = KLMConfig(),
    target_id: str = "",
) -> GomsOperator:
    """Create a K (keystroke) operator."""
    return GomsOperator(
        op_type=OperatorType.K,
        duration_s=config.k_duration,
        target_id=target_id,
        description=f"Type '{key}'" if key else "Keystroke",
        parameters={"key": key},
    )


def make_pointing(
    source: Point2D,
    target_center: Point2D,
    target_width_px: float,
    *,
    config: KLMConfig = KLMConfig(),
    target_id: str = "",
    target_bounds: Optional[BoundingBox] = None,
) -> GomsOperator:
    """Create a P (pointing) operator using Fitts' law."""
    distance = source.distance(target_center)
    duration = fitts_time(distance, target_width_px, config)
    return GomsOperator(
        op_type=OperatorType.P,
        duration_s=duration,
        target_id=target_id,
        target_bounds=target_bounds,
        description=f"Point to target ({distance:.0f}px, w={target_width_px:.0f}px)",
        parameters={
            "distance_px": distance,
            "target_width_px": target_width_px,
            "fitts_id": math.log2(distance / max(target_width_px, 1.0) + 1.0),
        },
    )


def make_homing(
    *,
    config: KLMConfig = KLMConfig(),
    description: str = "Home hand to mouse",
) -> GomsOperator:
    """Create an H (homing) operator."""
    return GomsOperator(
        op_type=OperatorType.H,
        duration_s=config.h_duration,
        description=description,
    )


def make_mental(
    *,
    config: KLMConfig = KLMConfig(),
    n_choices: int = 0,
    description: str = "Mental preparation",
) -> GomsOperator:
    """Create an M (mental preparation) operator.

    If *n_choices* > 0, duration is adjusted via Hick-Hyman law.
    """
    if n_choices > 0:
        duration = hick_hyman_time(n_choices, config)
        desc = f"Mental preparation (Hick-Hyman, n={n_choices})"
    else:
        duration = config.m_duration
        desc = description
    return GomsOperator(
        op_type=OperatorType.M,
        duration_s=duration,
        description=desc,
        parameters={"n_choices": n_choices} if n_choices > 0 else {},
    )


def make_system_response(
    *,
    latency_s: float = 0.0,
    config: KLMConfig = KLMConfig(),
    description: str = "System response",
) -> GomsOperator:
    """Create an R (system response) operator."""
    duration = latency_s if latency_s > 0 else config.system_response_s
    return GomsOperator(
        op_type=OperatorType.R,
        duration_s=duration,
        description=description,
        parameters={"latency_s": duration},
    )


def make_button_press(
    *,
    config: KLMConfig = KLMConfig(),
    target_id: str = "",
    description: str = "Mouse button press",
) -> GomsOperator:
    """Create a B (button press) operator."""
    return GomsOperator(
        op_type=OperatorType.B,
        duration_s=config.b_duration,
        target_id=target_id,
        description=description,
    )


def make_button_release(
    *,
    config: KLMConfig = KLMConfig(),
    target_id: str = "",
    description: str = "Mouse button release",
) -> GomsOperator:
    """Create a BB (button release) operator — same type B, tagged."""
    return GomsOperator(
        op_type=OperatorType.B,
        duration_s=config.b_duration,
        target_id=target_id,
        description=description,
        parameters={"action": "release"},
    )


# ═══════════════════════════════════════════════════════════════════════════
# M-operator placement rules (Card, Moran & Newell 1980; Raskin 2000)
# ═══════════════════════════════════════════════════════════════════════════

def apply_placement_rules(
    operators: Sequence[GomsOperator],
    *,
    config: KLMConfig = KLMConfig(),
) -> Tuple[GomsOperator, ...]:
    """Apply Card/Moran/Newell M-operator placement heuristics.

    **Rule 0** — Insert M before every K and P.
    **Rule 1** — Delete M before a K that is part of a continuous string.
    **Rule 2** — Delete M before a K following a P that selects
                  a command argument (redundant pointing).
    **Rule 3** — Delete M before the second P in a double-point
                  (drag) sequence.
    **Rule 4** — Delete M that directly follows R (system has user's attention).
    **Rule 5** — Delete overlapping Ms (adjacent Ms collapse to one).

    Returns
    -------
    tuple[GomsOperator, ...]
        Operator sequence with Ms properly placed.
    """
    # Rule 0: insert M before every K and P
    expanded: List[GomsOperator] = []
    m_op = make_mental(config=config)
    for op in operators:
        if op.op_type in (OperatorType.K, OperatorType.P):
            expanded.append(m_op)
        expanded.append(op)

    # Rule 1: delete M before K that continues a keystroke string
    cleaned: List[GomsOperator] = []
    i = 0
    while i < len(expanded):
        if (
            i + 1 < len(expanded)
            and expanded[i].op_type == OperatorType.M
            and expanded[i + 1].op_type == OperatorType.K
        ):
            # Check if previous non-M operator was also K (string continuation)
            prev_real = None
            for j in range(len(cleaned) - 1, -1, -1):
                if cleaned[j].op_type != OperatorType.M:
                    prev_real = cleaned[j]
                    break
            if prev_real is not None and prev_real.op_type == OperatorType.K:
                # Delete this M — keystroke string continues
                i += 1
                continue
        cleaned.append(expanded[i])
        i += 1

    # Rule 2: delete M before K following a P (selecting a command arg)
    result: List[GomsOperator] = []
    for idx, op in enumerate(cleaned):
        if (
            op.op_type == OperatorType.M
            and idx + 1 < len(cleaned)
            and cleaned[idx + 1].op_type == OperatorType.K
        ):
            # Look back for a P operator
            prev_real = None
            for j in range(idx - 1, -1, -1):
                if cleaned[j].op_type != OperatorType.M:
                    prev_real = cleaned[j]
                    break
            if prev_real is not None and prev_real.op_type == OperatorType.P:
                continue  # skip this M
        result.append(op)

    # Rule 3: delete M before second P in a double-point
    filtered: List[GomsOperator] = []
    for idx, op in enumerate(result):
        if op.op_type == OperatorType.M and idx + 1 < len(result):
            next_op = result[idx + 1]
            if next_op.op_type == OperatorType.P:
                prev_real = None
                for j in range(idx - 1, -1, -1):
                    if result[j].op_type != OperatorType.M:
                        prev_real = result[j]
                        break
                if prev_real is not None and prev_real.op_type == OperatorType.P:
                    continue  # skip M before second P
        filtered.append(op)

    # Rule 4: delete M that directly follows R
    final: List[GomsOperator] = []
    for idx, op in enumerate(filtered):
        if op.op_type == OperatorType.M and idx > 0:
            if filtered[idx - 1].op_type == OperatorType.R:
                continue  # skip M after R
        final.append(op)

    # Rule 5: collapse consecutive Ms to one
    collapsed: List[GomsOperator] = []
    for op in final:
        if op.op_type == OperatorType.M:
            if collapsed and collapsed[-1].op_type == OperatorType.M:
                continue
        collapsed.append(op)

    return tuple(collapsed)


# ═══════════════════════════════════════════════════════════════════════════
# Operator overlap estimation
# ═══════════════════════════════════════════════════════════════════════════

def estimate_overlap(
    operators: Sequence[GomsOperator],
    *,
    config: KLMConfig = KLMConfig(),
) -> float:
    """Estimate time saved from cognitive–motor overlap.

    Expert users overlap mental preparation with motor execution.
    We model this as a fraction of M-operator time that occurs in
    parallel with the preceding motor operator.

    Returns
    -------
    float
        Time savings in seconds from overlap.
    """
    if config.overlap_fraction <= 0.0:
        return 0.0

    saved = 0.0
    for i, op in enumerate(operators):
        if op.op_type == OperatorType.M and i > 0:
            prev = operators[i - 1]
            if prev.is_motor:
                overlap_duration = min(op.duration_s, prev.duration_s)
                saved += config.overlap_fraction * overlap_duration
    return saved


def compute_sequence_time(
    operators: Sequence[GomsOperator],
    *,
    config: KLMConfig = KLMConfig(),
) -> float:
    """Compute total execution time for a KLM sequence.

    Sums all operator durations then subtracts overlap savings.

    Returns
    -------
    float
        Predicted total time in seconds.
    """
    serial_time = sum(op.duration_s for op in operators)
    overlap_savings = estimate_overlap(operators, config=config)
    return max(0.0, serial_time - overlap_savings)


# ═══════════════════════════════════════════════════════════════════════════
# KLM sequence construction
# ═══════════════════════════════════════════════════════════════════════════

def build_klm_sequence(
    task_name: str,
    raw_operators: Sequence[GomsOperator],
    *,
    config: KLMConfig = KLMConfig(),
    apply_m_placement: bool = True,
) -> KLMSequence:
    """Build a complete KLM sequence from raw operators.

    Parameters
    ----------
    task_name : str
        Name of the task.
    raw_operators : Sequence[GomsOperator]
        Operators *without* M-placement applied.
    config : KLMConfig
        Timing and calibration configuration.
    apply_m_placement : bool
        Whether to apply M-operator placement heuristics.

    Returns
    -------
    KLMSequence
        Complete sequence with timing.
    """
    if apply_m_placement:
        ops = apply_placement_rules(raw_operators, config=config)
    else:
        ops = tuple(raw_operators)
    return KLMSequence(
        task_name=task_name,
        operators=ops,
        mental_prep_placed=apply_m_placement,
    )


# ═══════════════════════════════════════════════════════════════════════════
# KLM sequence optimization
# ═══════════════════════════════════════════════════════════════════════════

def optimize_klm_sequence(
    sequence: KLMSequence,
    *,
    config: KLMConfig = KLMConfig(),
) -> Tuple[KLMSequence, Mapping[str, Any]]:
    """Optimize a KLM sequence by removing redundant operators.

    Applies the following optimizations:
    1. Merge consecutive homing ops (only one H needed per device switch).
    2. Remove unnecessary button releases between clicks on the same target.
    3. Reduce pointing when targets are co-located (merge P ops).

    Returns
    -------
    tuple[KLMSequence, Mapping[str, Any]]
        Optimized sequence and a report of changes made.
    """
    ops = list(sequence.operators)
    changes: List[str] = []
    original_time = compute_sequence_time(ops, config=config)

    # 1. Merge consecutive H operators
    merged: List[GomsOperator] = []
    for op in ops:
        if op.op_type == OperatorType.H and merged and merged[-1].op_type == OperatorType.H:
            changes.append("Merged consecutive homing operators")
            continue
        merged.append(op)

    # 2. Remove redundant B (release) followed by B (press) on same target
    filtered: List[GomsOperator] = []
    skip_next = False
    for i, op in enumerate(merged):
        if skip_next:
            skip_next = False
            continue
        if (
            i + 1 < len(merged)
            and op.op_type == OperatorType.B
            and merged[i + 1].op_type == OperatorType.B
            and op.parameters.get("action") == "release"
            and op.target_id
            and op.target_id == merged[i + 1].target_id
        ):
            changes.append(f"Removed redundant button release/press on {op.target_id}")
            skip_next = True
            continue
        filtered.append(op)

    # 3. Merge P operators to co-located targets (distance < 5px)
    final: List[GomsOperator] = []
    for i, op in enumerate(filtered):
        if (
            op.op_type == OperatorType.P
            and final
            and final[-1].op_type == OperatorType.P
            and op.parameters.get("distance_px", float("inf")) < 5.0
        ):
            changes.append("Merged pointing to co-located targets")
            continue
        final.append(op)

    optimized_time = compute_sequence_time(final, config=config)
    report: Dict[str, Any] = {
        "original_time_s": original_time,
        "optimized_time_s": optimized_time,
        "time_saved_s": original_time - optimized_time,
        "operators_removed": len(ops) - len(final),
        "changes": changes,
    }
    optimized_seq = KLMSequence(
        task_name=sequence.task_name,
        operators=tuple(final),
        mental_prep_placed=sequence.mental_prep_placed,
    )
    return optimized_seq, report


# ═══════════════════════════════════════════════════════════════════════════
# KLMPredictor implementation
# ═══════════════════════════════════════════════════════════════════════════

class KLMPredictorImpl:
    """Concrete implementation of the KLMPredictor protocol.

    Generates KLM operator sequences from accessibility trees and
    predicts task completion times.
    """

    def __init__(self, config: Optional[KLMConfig] = None) -> None:
        self._config = config or KLMConfig()

    @property
    def config(self) -> KLMConfig:
        return self._config

    def predict(
        self,
        tree: Any,
        task_description: str,
    ) -> KLMSequence:
        """Generate a KLM operator sequence from an accessibility tree.

        Walks interactive nodes and produces operators for typical
        interaction patterns (click, type, select, navigate).
        """
        raw_ops = self._generate_raw_operators(tree)
        return build_klm_sequence(
            task_name=task_description,
            raw_operators=raw_ops,
            config=self._config,
            apply_m_placement=True,
        )

    def apply_mental_prep_heuristics(
        self,
        sequence: KLMSequence,
    ) -> KLMSequence:
        """Apply M-operator placement heuristics to a raw sequence."""
        if sequence.mental_prep_placed:
            return sequence
        ops = apply_placement_rules(sequence.operators, config=self._config)
        return KLMSequence(
            task_name=sequence.task_name,
            operators=ops,
            mental_prep_placed=True,
        )

    def _generate_raw_operators(self, tree: Any) -> List[GomsOperator]:
        """Generate raw operator sequence from accessibility tree nodes.

        Processes interactive elements based on their role.
        """
        ops: List[GomsOperator] = []
        cursor = Point2D(0.0, 0.0)

        nodes = list(tree.traverse_preorder())
        for node in nodes:
            role = getattr(node, "role", "")
            if not _is_interactive_role(role):
                continue

            node_id = getattr(node, "node_id", "")
            bounds = getattr(node, "bounds", None)
            if bounds is None:
                # Use default target size if bounds unknown
                center = Point2D(cursor.x + 100, cursor.y + 50)
                width = 80.0
            else:
                center = bounds.center
                width = bounds.width

            # Point to element
            ops.append(make_pointing(
                source=cursor,
                target_center=center,
                target_width_px=width,
                config=self._config,
                target_id=node_id,
                target_bounds=bounds,
            ))
            cursor = center

            # Role-specific operators
            if role in ("button", "link", "menuitem", "tab", "treeitem"):
                ops.append(make_button_press(
                    config=self._config, target_id=node_id,
                ))
            elif role == "textfield":
                # Homing + type a representative string
                ops.append(make_homing(config=self._config))
                for _ in range(5):  # average 5 chars
                    ops.append(make_keystroke(config=self._config, target_id=node_id))
                ops.append(make_homing(
                    config=self._config,
                    description="Home hand to mouse",
                ))
            elif role in ("checkbox", "radio"):
                ops.append(make_button_press(
                    config=self._config, target_id=node_id,
                ))
            elif role == "combobox":
                ops.append(make_button_press(
                    config=self._config, target_id=node_id,
                    description="Open combobox",
                ))
                ops.append(make_system_response(config=self._config))
            elif role == "slider":
                ops.append(make_button_press(
                    config=self._config, target_id=node_id,
                    description="Grab slider",
                ))
                # Drag
                ops.append(GomsOperator(
                    op_type=OperatorType.D,
                    duration_s=0.90,
                    target_id=node_id,
                    description="Drag slider",
                ))
                ops.append(make_button_release(
                    config=self._config, target_id=node_id,
                ))
        return ops


def _is_interactive_role(role: str) -> bool:
    """Check if a role string represents an interactive element."""
    return role.lower() in {
        "button", "link", "textfield", "checkbox", "radio",
        "combobox", "slider", "tab", "menuitem", "treeitem",
        "listitem", "cell", "row",
    }


__all__ = [
    "KLMConfig",
    "KLMPredictorImpl",
    "SkillLevel",
    "apply_placement_rules",
    "build_klm_sequence",
    "compute_sequence_time",
    "estimate_overlap",
    "fitts_time",
    "hick_hyman_time",
    "make_button_press",
    "make_button_release",
    "make_homing",
    "make_keystroke",
    "make_mental",
    "make_pointing",
    "make_system_response",
    "optimize_klm_sequence",
]
