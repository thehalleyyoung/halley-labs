"""
usability_oracle.simulation.klm — Keystroke-Level Model implementation.

The KLM is the simplest GOMS-family model, predicting expert error-free
task completion time as the sum of elementary operator durations.  This
module implements the original KLM from Card, Moran & Newell (1983) with
the mental-operator insertion rules (Rules 0–4) and extensions for
skill-level adjustment and design comparison.

References:
    Card, S. K., Moran, T. P., & Newell, A. (1983). *The Psychology
        of Human-Computer Interaction*. Erlbaum. (Ch. 8, The Keystroke-
        Level Model, pp. 259-297.)
    Kieras, D. (2001). Using the Keystroke-Level Model to Estimate
        Execution Times. Technical report, Univ. of Michigan.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# KLM operator definitions
# ═══════════════════════════════════════════════════════════════════════════

@unique
class KLMOperator(Enum):
    """Keystroke-Level Model primitive operators.

    Standard durations from Card, Moran & Newell (1983), Table 8.1, p. 264.

    K — Keystroke or button press
    P — Pointing to a target on the display
    H — Homing — moving hand between keyboard and mouse
    M — Mental preparation
    R — System response time (user waits)
    B — Mouse button press (distinct from keyboard K in some formulations)
    W — Wait / watch system feedback
    """

    K = "K"   # Keystroke
    P = "P"   # Pointing
    H = "H"   # Homing
    M = "M"   # Mental preparation
    R = "R"   # System response
    B = "B"   # Mouse button press
    W = "W"   # Wait / watch

    @property
    def is_physical(self) -> bool:
        return self in {KLMOperator.K, KLMOperator.P, KLMOperator.H, KLMOperator.B}

    @property
    def is_cognitive(self) -> bool:
        return self in {KLMOperator.M, KLMOperator.W}

    @property
    def is_system(self) -> bool:
        return self == KLMOperator.R


# ═══════════════════════════════════════════════════════════════════════════
# Operator timing — published parameter values
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class KLMTimings:
    """Operator durations for the Keystroke-Level Model.

    All times in seconds.  Default values from Card, Moran & Newell (1983),
    Table 8.1, pp. 263-265.

    Attributes:
        t_k_average: Average typist keystroke (0.280 s; 55 wpm).
        t_k_expert: Expert typist keystroke (0.120 s; 135 wpm).
        t_k_worst: Worst-case typist (0.500 s; 40 wpm non-typist).
        t_k_complex: Complex keystroke (Shift+key) (0.350 s).
        t_p: Pointing (mouse) to a target (1.100 s, Fitts' law avg).
        t_h: Homing between keyboard and mouse (0.400 s).
        t_m: Mental preparation operator (1.350 s).
        t_r: System response time (context-dependent, default 0.0 s).
        t_b: Mouse button click (0.100 s).
        t_w: Watch / wait for feedback (0.0 s; context-dependent).
    """
    t_k_average: float = 0.280
    t_k_expert: float = 0.120
    t_k_worst: float = 0.500
    t_k_complex: float = 0.350
    t_p: float = 1.100
    t_h: float = 0.400
    t_m: float = 1.350
    t_r: float = 0.0
    t_b: float = 0.100
    t_w: float = 0.0

    def get_time(self, op: KLMOperator, skill_level: str = "average") -> float:
        """Get operator time for a given skill level.

        Skill levels: 'expert', 'average', 'novice'
        Reference: Card et al. (1983), Table 8.1.
        """
        if op == KLMOperator.K:
            if skill_level == "expert":
                return self.t_k_expert
            elif skill_level == "novice":
                return self.t_k_worst
            return self.t_k_average
        mapping = {
            KLMOperator.P: self.t_p,
            KLMOperator.H: self.t_h,
            KLMOperator.M: self.t_m,
            KLMOperator.R: self.t_r,
            KLMOperator.B: self.t_b,
            KLMOperator.W: self.t_w,
        }
        return mapping.get(op, 0.0)


# ═══════════════════════════════════════════════════════════════════════════
# KLM Operator sequence item
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class KLMStep:
    """A single step in a KLM operator sequence.

    Attributes:
        operator: The KLM operator type.
        description: Human-readable description of this step.
        duration: Time for this step (seconds). Auto-populated if not set.
        element_id: Associated UI element (for pointing/clicking).
        metadata: Extra context.
    """
    operator: KLMOperator = KLMOperator.K
    description: str = ""
    duration: float = 0.0
    element_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def with_duration(self, timings: KLMTimings, skill: str = "average") -> KLMStep:
        """Return a copy with duration set from timings."""
        return KLMStep(
            operator=self.operator,
            description=self.description,
            duration=timings.get_time(self.operator, skill),
            element_id=self.element_id,
            metadata=self.metadata,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Mental operator insertion rules (Card et al. 1983, pp. 265-268)
# ═══════════════════════════════════════════════════════════════════════════

def apply_heuristic_rules(
    operators: List[KLMStep],
    timings: Optional[KLMTimings] = None,
) -> List[KLMStep]:
    """Apply the mental-operator insertion heuristic rules (Rules 0-4).

    Card et al. (1983, pp. 265-268) specify five rules for inserting M
    operators to account for mental preparation time:

    Rule 0: Insert M before all K and P operators initially.
    Rule 1: Delete M within a cognitive unit (e.g., typing a filename).
    Rule 2: Delete M before a K that is part of a redundant string
             (e.g., repeated characters, closing delimiters).
    Rule 3: Delete M before the first K after a P (if P targets the
             same cognitive unit).
    Rule 4: Delete M before K if the preceding operator is also K
             and both are part of the same argument string.

    Parameters:
        operators: Raw KLM operator sequence (without M operators).
        timings: Optional timing configuration.

    Returns:
        New operator sequence with M operators inserted per the rules.
    """
    t = timings or KLMTimings()

    # Rule 0: Insert M before every K and P
    expanded: List[KLMStep] = []
    for step in operators:
        if step.operator in (KLMOperator.K, KLMOperator.P, KLMOperator.B):
            expanded.append(KLMStep(
                operator=KLMOperator.M,
                description=f"Mental prep before {step.operator.value}",
                duration=t.t_m,
            ))
        expanded.append(step)

    # Rule 1: Delete M within a cognitive unit (consecutive Ks for same element)
    result: List[KLMStep] = []
    i = 0
    while i < len(expanded):
        step = expanded[i]
        if (step.operator == KLMOperator.M
                and i + 1 < len(expanded)
                and expanded[i + 1].operator == KLMOperator.K):
            # Check if this K is in same cognitive unit as previous K
            if result and result[-1].operator == KLMOperator.K:
                prev_elem = result[-1].element_id
                next_elem = expanded[i + 1].element_id
                if prev_elem and prev_elem == next_elem:
                    i += 1  # Skip this M
                    continue
        result.append(step)
        i += 1

    # Rule 2: Delete M before redundant terminators
    cleaned: List[KLMStep] = []
    for idx, step in enumerate(result):
        if step.operator == KLMOperator.M and idx + 1 < len(result):
            next_step = result[idx + 1]
            desc_lower = next_step.description.lower()
            if any(term in desc_lower for term in ("enter", "return", "tab", "delimiter", "closing")):
                continue  # Skip M before terminator
        cleaned.append(step)

    # Rule 3: Delete M before first K after P if same cognitive unit
    final: List[KLMStep] = []
    saw_p = False
    for step in cleaned:
        if step.operator == KLMOperator.P:
            saw_p = True
            final.append(step)
        elif step.operator == KLMOperator.M and saw_p:
            saw_p = False
            continue  # Delete M after P (same unit)
        else:
            saw_p = False
            final.append(step)

    # Rule 4: Delete M within argument strings (consecutive Ks, same group)
    output: List[KLMStep] = []
    in_arg_string = False
    for idx, step in enumerate(final):
        if step.operator == KLMOperator.K:
            if in_arg_string:
                # Remove preceding M if it was just added
                if output and output[-1].operator == KLMOperator.M:
                    m_desc = output[-1].description.lower()
                    if "mental prep" in m_desc:
                        output.pop()
            in_arg_string = True
            output.append(step)
        elif step.operator == KLMOperator.M:
            in_arg_string = False
            output.append(step)
        else:
            in_arg_string = False
            output.append(step)

    return output


# ═══════════════════════════════════════════════════════════════════════════
# KLMModel
# ═══════════════════════════════════════════════════════════════════════════

class KLMModel:
    """Keystroke-Level Model for predicting error-free expert task time.

    The KLM predicts execution time T as:

        T = Σ t_op  for each operator op in the task sequence

    where t_op is the standard time for operator op.

    Usage::

        model = KLMModel()
        ops = model.encode_task(task_spec, ui_structure)
        time = model.predict_task_time(ops)

    Reference: Card, Moran & Newell (1983), Chapter 8.
    """

    def __init__(
        self,
        timings: Optional[KLMTimings] = None,
        skill_level: str = "average",
        auto_insert_mental: bool = True,
    ) -> None:
        self._timings = timings or KLMTimings()
        self._skill = skill_level
        self._auto_mental = auto_insert_mental

    @property
    def timings(self) -> KLMTimings:
        return self._timings

    @property
    def skill_level(self) -> str:
        return self._skill

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_task_time(self, operator_sequence: List[KLMStep]) -> float:
        """Predict total task execution time.

        T_execute = Σ t_K + Σ t_P + Σ t_H + Σ t_M + Σ t_R

        Reference: Card et al. (1983), Eq. 8.1, p. 260.
        """
        total = 0.0
        for step in operator_sequence:
            if step.duration > 0:
                total += step.duration
            else:
                total += self._timings.get_time(step.operator, self._skill)
        return total

    def predict_with_breakdown(self, operator_sequence: List[KLMStep]) -> Dict[str, float]:
        """Predict time with per-operator-type breakdown."""
        breakdown: Dict[str, float] = {}
        for step in operator_sequence:
            op_name = step.operator.value
            t = step.duration if step.duration > 0 else self._timings.get_time(step.operator, self._skill)
            breakdown[op_name] = breakdown.get(op_name, 0.0) + t
        breakdown["total"] = sum(breakdown.values())
        return breakdown

    # ------------------------------------------------------------------
    # Task encoding
    # ------------------------------------------------------------------

    def encode_task(
        self,
        task_spec: Dict[str, Any],
        ui_structure: Optional[Dict[str, Any]] = None,
    ) -> List[KLMStep]:
        """Automatically generate a KLM operator sequence from a task spec.

        The task_spec should contain a list of 'steps', each with:
        - 'type': 'click', 'type_text', 'select', 'navigate', etc.
        - 'target': UI element identifier
        - 'text': (for type_text) the text to type
        - 'n_items': (for select) number of items to scan

        Parameters:
            task_spec: Task description.
            ui_structure: Optional UI layout for computing pointing times.

        Returns:
            KLM operator sequence (with M operators if auto_insert_mental).
        """
        steps = task_spec.get("steps", [])
        operators: List[KLMStep] = []

        hand_on_mouse = False

        for step in steps:
            step_type = step.get("type", "click")
            target = step.get("target", "")

            if step_type == "click":
                if not hand_on_mouse:
                    operators.append(KLMStep(
                        operator=KLMOperator.H,
                        description=f"Home hand to mouse",
                        element_id=target,
                    ))
                    hand_on_mouse = True
                operators.append(KLMStep(
                    operator=KLMOperator.P,
                    description=f"Point to {target}",
                    element_id=target,
                ))
                operators.append(KLMStep(
                    operator=KLMOperator.B,
                    description=f"Click {target}",
                    element_id=target,
                ))

            elif step_type == "type_text":
                text = step.get("text", "")
                if hand_on_mouse:
                    operators.append(KLMStep(
                        operator=KLMOperator.H,
                        description="Home hand to keyboard",
                    ))
                    hand_on_mouse = False
                for ch in text:
                    operators.append(KLMStep(
                        operator=KLMOperator.K,
                        description=f"Type '{ch}'",
                        element_id=target,
                    ))

            elif step_type == "select":
                if not hand_on_mouse:
                    operators.append(KLMStep(
                        operator=KLMOperator.H,
                        description="Home hand to mouse",
                    ))
                    hand_on_mouse = True
                # Open dropdown
                operators.append(KLMStep(
                    operator=KLMOperator.P,
                    description=f"Point to dropdown {target}",
                    element_id=target,
                ))
                operators.append(KLMStep(
                    operator=KLMOperator.B,
                    description=f"Click to open {target}",
                    element_id=target,
                ))
                # Point to desired option
                operators.append(KLMStep(
                    operator=KLMOperator.P,
                    description=f"Point to option in {target}",
                    element_id=target,
                ))
                operators.append(KLMStep(
                    operator=KLMOperator.B,
                    description=f"Select option in {target}",
                    element_id=target,
                ))

            elif step_type == "navigate":
                if not hand_on_mouse:
                    operators.append(KLMStep(
                        operator=KLMOperator.H,
                        description="Home hand to mouse",
                    ))
                    hand_on_mouse = True
                operators.append(KLMStep(
                    operator=KLMOperator.P,
                    description=f"Point to nav element {target}",
                    element_id=target,
                ))
                operators.append(KLMStep(
                    operator=KLMOperator.B,
                    description=f"Click nav {target}",
                    element_id=target,
                ))
                # System response
                response_time = step.get("response_time", 1.0)
                operators.append(KLMStep(
                    operator=KLMOperator.R,
                    description="Wait for page load",
                    duration=response_time,
                ))

            elif step_type == "wait":
                wait_time = step.get("duration", 1.0)
                operators.append(KLMStep(
                    operator=KLMOperator.R,
                    description=f"System response ({wait_time}s)",
                    duration=wait_time,
                ))

            elif step_type == "keyboard_shortcut":
                if hand_on_mouse:
                    operators.append(KLMStep(
                        operator=KLMOperator.H,
                        description="Home hand to keyboard",
                    ))
                    hand_on_mouse = False
                keys = step.get("keys", "")
                for key in keys.split("+"):
                    operators.append(KLMStep(
                        operator=KLMOperator.K,
                        description=f"Press {key.strip()}",
                        element_id=target,
                    ))

        # Optionally insert mental operators
        if self._auto_mental:
            operators = apply_heuristic_rules(operators, self._timings)

        # Set durations
        operators = [
            op.with_duration(self._timings, self._skill) if op.duration == 0 else op
            for op in operators
        ]

        return operators

    # ------------------------------------------------------------------
    # Skill adjustment
    # ------------------------------------------------------------------

    def expert_vs_novice(
        self,
        operators: List[KLMStep],
        skill_level: str = "average",
    ) -> Dict[str, float]:
        """Compute task time adjusted for skill level.

        Skill affects primarily keystroke time (K) and mental preparation (M):
        - Expert: K=0.120s, M reduced by 33% (Kieras 2001)
        - Novice: K=0.500s, M increased by 25%

        Reference: Card et al. (1983), Table 8.1; Kieras (2001).
        """
        # Skill multipliers for M operator
        m_multipliers = {
            "expert": 0.67,    # Experts need less mental preparation
            "average": 1.0,
            "novice": 1.25,    # Novices need more mental preparation
        }
        m_mult = m_multipliers.get(skill_level, 1.0)

        total = 0.0
        for step in operators:
            if step.operator == KLMOperator.K:
                total += self._timings.get_time(KLMOperator.K, skill_level)
            elif step.operator == KLMOperator.M:
                total += self._timings.t_m * m_mult
            elif step.duration > 0:
                total += step.duration
            else:
                total += self._timings.get_time(step.operator, skill_level)

        return {
            "skill_level": skill_level,
            "total_time": total,
            "k_time": sum(
                self._timings.get_time(KLMOperator.K, skill_level)
                for s in operators if s.operator == KLMOperator.K
            ),
            "m_time": sum(
                self._timings.t_m * m_mult
                for s in operators if s.operator == KLMOperator.M
            ),
        }

    # ------------------------------------------------------------------
    # Design comparison
    # ------------------------------------------------------------------

    def compare_designs(
        self,
        design_a_ops: List[KLMStep],
        design_b_ops: List[KLMStep],
    ) -> Dict[str, Any]:
        """Compare two designs for regression detection via KLM.

        Returns:
            Dict with time difference, operator counts, and regression flag.
        """
        time_a = self.predict_task_time(design_a_ops)
        time_b = self.predict_task_time(design_b_ops)

        breakdown_a = self.predict_with_breakdown(design_a_ops)
        breakdown_b = self.predict_with_breakdown(design_b_ops)

        count_a = {op.value: sum(1 for s in design_a_ops if s.operator == op) for op in KLMOperator}
        count_b = {op.value: sum(1 for s in design_b_ops if s.operator == op) for op in KLMOperator}

        delta = time_b - time_a
        pct_change = (delta / time_a * 100.0) if time_a > 0 else 0.0

        # Regression if design B is >10% slower
        regression_threshold = 0.10
        is_regression = delta > 0 and (delta / max(time_a, 0.001)) > regression_threshold

        return {
            "time_a": time_a,
            "time_b": time_b,
            "delta_seconds": delta,
            "pct_change": pct_change,
            "is_regression": is_regression,
            "breakdown_a": breakdown_a,
            "breakdown_b": breakdown_b,
            "operator_counts_a": count_a,
            "operator_counts_b": count_b,
            "n_operators_a": len(design_a_ops),
            "n_operators_b": len(design_b_ops),
            "dominant_cost_a": max(breakdown_a, key=breakdown_a.get) if breakdown_a else "",
            "dominant_cost_b": max(breakdown_b, key=breakdown_b.get) if breakdown_b else "",
        }

    # ------------------------------------------------------------------
    # Parsing from text
    # ------------------------------------------------------------------

    @staticmethod
    def parse_klm_string(klm_string: str) -> List[KLMStep]:
        """Parse a KLM operator string like 'M P B K K K M P B'.

        Each character/token maps to a KLM operator. Useful for quick
        task specification following Card et al. notation.

        Reference: Card et al. (1983), notation on p. 262.
        """
        tokens = klm_string.strip().split()
        steps: List[KLMStep] = []
        op_map = {op.value: op for op in KLMOperator}
        for tok in tokens:
            tok_upper = tok.upper()
            if tok_upper in op_map:
                steps.append(KLMStep(operator=op_map[tok_upper], description=tok_upper))
            else:
                # Try to parse as repeated K: e.g. "3K" -> K K K
                match = re.match(r"(\d+)([A-Z])", tok_upper)
                if match:
                    count = int(match.group(1))
                    op_char = match.group(2)
                    if op_char in op_map:
                        for _ in range(count):
                            steps.append(KLMStep(operator=op_map[op_char], description=op_char))
        return steps

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def summary(self, operators: List[KLMStep]) -> str:
        """Human-readable summary of a KLM analysis."""
        breakdown = self.predict_with_breakdown(operators)
        total = breakdown.pop("total", 0.0)
        lines = [f"KLM Analysis (skill={self._skill})"]
        lines.append(f"  Total predicted time: {total:.3f}s")
        lines.append(f"  Number of operators: {len(operators)}")
        for op_name, t in sorted(breakdown.items()):
            count = sum(1 for s in operators if s.operator.value == op_name)
            lines.append(f"  {op_name}: {count} ops, {t:.3f}s")
        return "\n".join(lines)
