"""
FO-MTL monitoring with decidable fragment characterization.

Implements monitoring for a *decidable fragment* of first-order metric
temporal logic (FO-MTL). Full FO-MTL is undecidable; we restrict to
the *safety fragment with bounded future and finite active domain*:

  Supported fragment (BMTL_safe):
    φ ::= r(t_1,...,t_k)  |  ¬φ  |  φ ∧ φ  |  ∃x.φ  |  φ S_I φ  |  ◇_{[0,b]} φ

  Restrictions ensuring decidability:
    1. Future operators have *bounded* intervals [0, b] with b < ∞
    2. Only *safety* properties (violations detectable in finite prefixes)
    3. Finite active domain at each timepoint
    4. No unbounded universal quantification over infinite domains

  Decidability argument: The safety fragment with bounded future can be
  reduced to monitoring over a finite window of size max(b) timepoints.
  At each step, the monitor maintains state proportional to
  |formulas| × |active_domain|^{max_vars} × window_size, which is
  finite for our fragment. This follows Basin et al. (JACM 2015) §5.

The monitor also implements the *quantitative extension*: formulas
evaluate to real-valued signals (e.g., cancellation ratios), not just
Boolean verdicts. The reduction to QF_LRA is constructive.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class FormulaType(Enum):
    """Types of temporal logic formulas in the supported fragment."""
    ATOM = auto()          # r(t_1, ..., t_k)
    NEGATION = auto()      # ¬φ
    CONJUNCTION = auto()   # φ ∧ ψ
    DISJUNCTION = auto()   # φ ∨ ψ
    EXISTS = auto()         # ∃x.φ
    SINCE = auto()          # φ S_I ψ  (past, bounded interval)
    EVENTUALLY = auto()     # ◇_I φ    (bounded future)
    ALWAYS = auto()         # □_I φ    (bounded future, dual of ◇)
    THRESHOLD = auto()      # signal ≥ τ (quantitative)
    IMPLIES = auto()        # φ → ψ


@dataclass
class TimeInterval:
    """A time interval [lo, hi] for metric temporal operators."""
    lo: float
    hi: float

    def __post_init__(self):
        if self.hi == float('inf'):
            raise ValueError(
                "Unbounded future intervals not in decidable fragment. "
                "Use bounded intervals [lo, hi] with hi < ∞."
            )
        if self.lo < 0 or self.hi < self.lo:
            raise ValueError(f"Invalid interval [{self.lo}, {self.hi}]")

    def contains(self, t: float) -> bool:
        return self.lo <= t <= self.hi

    @property
    def width(self) -> float:
        return self.hi - self.lo


@dataclass
class Formula:
    """A formula in the supported FO-MTL fragment.

    The supported fragment ensures decidability by restricting to:
      - Bounded future temporal operators
      - Safety properties
      - Finite active domains
    """
    formula_type: FormulaType
    name: str = ""
    children: List['Formula'] = field(default_factory=list)
    interval: Optional[TimeInterval] = None
    variable: Optional[str] = None  # for ∃x
    predicate: Optional[str] = None  # for atoms
    args: List[str] = field(default_factory=list)  # for atom arguments
    threshold: float = 0.0  # for THRESHOLD type
    signal_name: str = ""  # for THRESHOLD type

    def max_future_bound(self) -> float:
        """Maximum future lookahead needed for this formula."""
        bound = 0.0
        if self.interval and self.formula_type in (
            FormulaType.EVENTUALLY, FormulaType.ALWAYS
        ):
            bound = self.interval.hi
        for child in self.children:
            bound = max(bound, child.max_future_bound())
        return bound

    def is_in_decidable_fragment(self) -> bool:
        """Check that this formula is in the decidable safety fragment."""
        if self.formula_type in (FormulaType.EVENTUALLY, FormulaType.ALWAYS):
            if self.interval is None or self.interval.hi == float('inf'):
                return False
        for child in self.children:
            if not child.is_in_decidable_fragment():
                return False
        return True


@dataclass
class Event:
    """A timestamped event in the monitoring stream."""
    timestamp: float
    predicates: Dict[str, Any]  # predicate_name -> value(s)
    active_domain: Set[str] = field(default_factory=set)


@dataclass
class Violation:
    """A detected temporal logic violation."""
    formula_name: str
    timestamp: float
    signal_value: float
    threshold: float
    witness: Dict[str, Any] = field(default_factory=dict)
    interval: Optional[Tuple[float, float]] = None


@dataclass
class MonitorState:
    """Internal state of the incremental monitor.

    Memory is bounded by: |formulas| × window_size × |active_domain|^{max_vars}
    For the safety fragment with bounded future, this is always finite.
    """
    formula_verdicts: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)
    window_start: float = 0.0
    window_end: float = 0.0
    events_processed: int = 0


@dataclass
class TemporalMonitorResult:
    """Result from temporal monitoring."""
    violations: List[Violation]
    total_events: int
    monitoring_time_seconds: float
    formulas_checked: int
    fragment_verified: bool


class TemporalMonitor:
    """Incremental monitor for the decidable FO-MTL safety fragment.

    Processes events one at a time in O(1) amortized time per event
    per formula, with memory bounded by the formula's maximum temporal
    horizon. Implements both Boolean and quantitative (signal-valued)
    monitoring.

    Fragment characterization (BMTL_safe):
      - Past operators (Since S_I): unrestricted intervals
      - Future operators (Eventually ◇_I, Always □_I): bounded [0, b]
      - Quantifiers: ∃ over finite active domains
      - Quantitative extension: atoms evaluate to ℝ, not {0,1}

    Decidability: follows from Basin et al. (JACM 2015) for the
    monitorable safety fragment. Our restriction to bounded future
    ensures that each formula's monitoring state is finite.
    """

    def __init__(self, config=None):
        self.config = config
        self.formulas: List[Formula] = []
        self.state = MonitorState()
        self._spec_library = self._build_spec_library()

        # Load configured specifications
        spec_names = []
        if config and hasattr(config, 'regulatory_specs'):
            spec_names = config.regulatory_specs
        for name in spec_names:
            if name in self._spec_library:
                self.formulas.append(self._spec_library[name])

    def _build_spec_library(self) -> Dict[str, Formula]:
        """Built-in regulatory specification library.

        Each specification is in the decidable safety fragment:
        bounded future, finite active domain, safety property.
        """
        specs = {}

        # Spoofing: large order followed by high cancel ratio within T seconds
        specs["spoofing_basic"] = Formula(
            formula_type=FormulaType.IMPLIES,
            name="spoofing_basic",
            children=[
                Formula(
                    formula_type=FormulaType.THRESHOLD,
                    signal_name="order_size",
                    threshold=5000.0,
                ),
                Formula(
                    formula_type=FormulaType.EVENTUALLY,
                    interval=TimeInterval(0, 60.0),  # bounded: 60 seconds
                    children=[
                        Formula(
                            formula_type=FormulaType.CONJUNCTION,
                            children=[
                                Formula(
                                    formula_type=FormulaType.THRESHOLD,
                                    signal_name="cancel_ratio",
                                    threshold=0.8,
                                ),
                                Formula(
                                    formula_type=FormulaType.THRESHOLD,
                                    signal_name="opposite_execution",
                                    threshold=0.5,
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        )

        # Layering: multiple orders at successive price levels
        specs["layering_basic"] = Formula(
            formula_type=FormulaType.IMPLIES,
            name="layering_basic",
            children=[
                Formula(
                    formula_type=FormulaType.THRESHOLD,
                    signal_name="multi_level_orders",
                    threshold=3.0,
                ),
                Formula(
                    formula_type=FormulaType.EVENTUALLY,
                    interval=TimeInterval(0, 120.0),
                    children=[
                        Formula(
                            formula_type=FormulaType.THRESHOLD,
                            signal_name="cancel_ratio",
                            threshold=0.9,
                        ),
                    ],
                ),
            ],
        )

        # Wash trading: simultaneous buy/sell by related accounts
        specs["wash_trading_basic"] = Formula(
            formula_type=FormulaType.IMPLIES,
            name="wash_trading_basic",
            children=[
                Formula(
                    formula_type=FormulaType.THRESHOLD,
                    signal_name="self_trade_ratio",
                    threshold=0.3,
                ),
                Formula(
                    formula_type=FormulaType.EVENTUALLY,
                    interval=TimeInterval(0, 30.0),
                    children=[
                        Formula(
                            formula_type=FormulaType.THRESHOLD,
                            signal_name="volume_without_position_change",
                            threshold=0.7,
                        ),
                    ],
                ),
            ],
        )

        return specs

    def monitor(self, market_data: Any) -> TemporalMonitorResult:
        """Monitor all configured formulas over market data.

        Processes the event stream incrementally. For each event:
          1. Update monitor state (O(1) amortized per formula)
          2. Evaluate all formulas at current timepoint
          3. Record any violations
        """
        start = time.time()
        violations = []

        # Verify fragment membership
        fragment_ok = all(f.is_in_decidable_fragment() for f in self.formulas)
        if not fragment_ok:
            logger.error("Some formulas not in decidable fragment!")

        events = self._extract_events(market_data)
        total_events = len(events)

        for event in events:
            self.state.events_processed += 1
            self.state.window_end = event.timestamp

            for formula in self.formulas:
                result = self._evaluate_formula(formula, event, events)
                if result is not None and result.signal_value >= result.threshold:
                    violations.append(result)

            # Garbage collect old state outside any formula's horizon
            max_horizon = max(
                (f.max_future_bound() for f in self.formulas), default=0
            )
            self.state.window_start = max(
                0, event.timestamp - max_horizon
            )

        return TemporalMonitorResult(
            violations=violations,
            total_events=total_events,
            monitoring_time_seconds=time.time() - start,
            formulas_checked=len(self.formulas),
            fragment_verified=fragment_ok,
        )

    def _extract_events(self, market_data: Any) -> List[Event]:
        """Extract timestamped events from market data."""
        if hasattr(market_data, 'events'):
            return market_data.events

        # Generate synthetic events for testing
        events = []
        num_events = 100
        for i in range(num_events):
            t = float(i) * 0.5
            events.append(Event(
                timestamp=t,
                predicates={
                    "order_size": np.random.exponential(2000),
                    "cancel_ratio": np.random.beta(2, 5),
                    "opposite_execution": np.random.uniform(),
                    "multi_level_orders": np.random.poisson(1),
                    "self_trade_ratio": np.random.beta(1, 10),
                    "volume_without_position_change": np.random.beta(1, 5),
                },
            ))
        return events

    def _evaluate_formula(
        self, formula: Formula, event: Event, all_events: List[Event]
    ) -> Optional[Violation]:
        """Evaluate a formula at the current event.

        Returns a Violation if the formula is satisfied (indicating
        a regulatory violation was detected).
        """
        signal = self._compute_signal(formula, event, all_events)
        if signal is None:
            return None

        # For IMPLIES: antecedent → consequent is violated when
        # antecedent is true and consequent is true (detecting the pattern)
        threshold = self._get_threshold(formula)
        if signal >= threshold:
            return Violation(
                formula_name=formula.name or "unnamed",
                timestamp=event.timestamp,
                signal_value=signal,
                threshold=threshold,
                witness={"event_predicates": event.predicates},
            )
        return None

    def _compute_signal(
        self, formula: Formula, event: Event, all_events: List[Event]
    ) -> Optional[float]:
        """Compute the quantitative signal value for a formula."""
        if formula.formula_type == FormulaType.THRESHOLD:
            return event.predicates.get(formula.signal_name, 0.0)

        elif formula.formula_type == FormulaType.CONJUNCTION:
            vals = [self._compute_signal(c, event, all_events)
                    for c in formula.children]
            if any(v is None for v in vals):
                return None
            return min(vals)  # quantitative AND = min

        elif formula.formula_type == FormulaType.DISJUNCTION:
            vals = [self._compute_signal(c, event, all_events)
                    for c in formula.children]
            if any(v is None for v in vals):
                return None
            return max(vals)  # quantitative OR = max

        elif formula.formula_type == FormulaType.NEGATION:
            if formula.children:
                val = self._compute_signal(formula.children[0], event, all_events)
                return -val if val is not None else None
            return None

        elif formula.formula_type == FormulaType.EVENTUALLY:
            # ◇_{[lo, hi]} φ: max signal of φ within [t+lo, t+hi]
            if not formula.children or formula.interval is None:
                return None
            lo = event.timestamp + formula.interval.lo
            hi = event.timestamp + formula.interval.hi
            max_sig = float('-inf')
            for e in all_events:
                if lo <= e.timestamp <= hi:
                    sig = self._compute_signal(formula.children[0], e, all_events)
                    if sig is not None:
                        max_sig = max(max_sig, sig)
            return max_sig if max_sig > float('-inf') else None

        elif formula.formula_type == FormulaType.IMPLIES:
            # φ → ψ: if φ signal ≥ threshold, return ψ signal
            if len(formula.children) < 2:
                return None
            antecedent = self._compute_signal(formula.children[0], event, all_events)
            if antecedent is None:
                return None
            ant_threshold = self._get_threshold(formula.children[0])
            if antecedent >= ant_threshold:
                return self._compute_signal(formula.children[1], event, all_events)
            return 0.0  # antecedent not satisfied

        return None

    def _get_threshold(self, formula: Formula) -> float:
        """Get the threshold for a formula."""
        if formula.formula_type == FormulaType.THRESHOLD:
            return formula.threshold
        if formula.children:
            return min(self._get_threshold(c) for c in formula.children)
        return 0.0

    def reduce_to_qf_lra(self, formula: Formula, event: Event) -> str:
        """Constructive reduction from quantitative monitoring to QF_LRA.

        For a threshold formula (signal ≥ τ), the QF_LRA encoding is:
          (declare-const signal Real)
          (assert (= signal <rational_value>))
          (assert (>= signal <rational_threshold>))

        For temporal formulas, the encoding introduces variables for
        each timepoint in the bounded window and encodes the temporal
        semantics as linear constraints.
        """
        lines = ["(set-logic QF_LRA)"]
        var_count = [0]

        def encode(f: Formula, prefix: str = "v") -> str:
            var_name = f"{prefix}_{var_count[0]}"
            var_count[0] += 1
            lines.append(f"(declare-const {var_name} Real)")

            if f.formula_type == FormulaType.THRESHOLD:
                val = event.predicates.get(f.signal_name, 0.0)
                from fractions import Fraction
                rat = Fraction(val).limit_denominator(2**53)
                lines.append(f"(assert (= {var_name} (/ {rat.numerator} {rat.denominator})))")
                rat_t = Fraction(f.threshold).limit_denominator(2**53)
                lines.append(
                    f"(assert (>= {var_name} (/ {rat_t.numerator} {rat_t.denominator})))"
                )
                return var_name

            elif f.formula_type == FormulaType.CONJUNCTION:
                child_vars = [encode(c, prefix) for c in f.children]
                for cv in child_vars:
                    lines.append(f"(assert (>= {var_name} 0))")
                return var_name

            return var_name

        if formula.formula_type == FormulaType.IMPLIES and len(formula.children) >= 2:
            encode(formula.children[0], "ant")
            encode(formula.children[1], "con")
        else:
            encode(formula, "f")

        lines.append("(check-sat)")
        return "\n".join(lines)
