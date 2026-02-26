"""
Conformance gap analysis for k-bounded equivalence testing.

Characterizes the gap between k-bounded conformance testing (the practical
equivalence oracle) and true F-bisimulation equivalence. Provides:

1. A formal error bound: if the hypothesis passes k-bounded conformance,
   any missed counterexample has length > k, so the number of potentially
   misclassified states is bounded by |S| · |Act|^{-(k - diam(H))}.

2. Empirical gap measurement: for systems where the true bisimulation
   quotient is known (e.g., via partition refinement), compare the
   learner's output at each depth k to the true quotient and measure
   how quickly the gap closes.

3. Depth sufficiency criterion: a k is sufficient when k >= diam(H) + n,
   where diam(H) is the diameter of the hypothesis and n = |H| is its
   state count. Beyond this, the W-method guarantees no missed states
   (assuming the concrete system has at most 2n states).

Reference: Chow (1978) W-method, Vasilevskii (1973) for the original
conformance testing bound.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Tuple,
)

logger = logging.getLogger(__name__)


@dataclass
class GapMeasurement:
    """A single measurement of the conformance gap at depth k."""
    depth_k: int
    hypothesis_states: int
    true_quotient_states: int
    misclassified_states: int
    extra_classes: int  # classes in hypothesis not in true quotient
    missing_classes: int  # true quotient classes not captured
    gap_ratio: float  # misclassified / true_quotient_states
    error_bound: float  # theoretical upper bound on miss probability

    def summary(self) -> str:
        return (
            f"k={self.depth_k}: hyp={self.hypothesis_states}, "
            f"true={self.true_quotient_states}, "
            f"misclassified={self.misclassified_states}, "
            f"gap={self.gap_ratio:.4f}, bound={self.error_bound:.6f}"
        )


@dataclass
class GapAnalysisResult:
    """Full result of a conformance gap analysis across multiple depths."""
    measurements: List[GapMeasurement] = field(default_factory=list)
    sufficient_k: Optional[int] = None
    convergence_rate: Optional[float] = None
    hypothesis_diameter: Optional[int] = None

    def summary(self) -> str:
        lines = ["Conformance Gap Analysis"]
        lines.append(f"  Sufficient k: {self.sufficient_k}")
        lines.append(f"  Convergence rate: {self.convergence_rate}")
        lines.append(f"  Hypothesis diameter: {self.hypothesis_diameter}")
        for m in self.measurements:
            lines.append(f"  {m.summary()}")
        return "\n".join(lines)

    def gap_at(self, k: int) -> Optional[GapMeasurement]:
        """Return the measurement at depth k, if available."""
        for m in self.measurements:
            if m.depth_k == k:
                return m
        return None

    def convergence_data(self) -> Dict[str, List[Any]]:
        """Return data suitable for plotting gap convergence."""
        return {
            "k": [m.depth_k for m in self.measurements],
            "gap_ratio": [m.gap_ratio for m in self.measurements],
            "error_bound": [m.error_bound for m in self.measurements],
            "hypothesis_states": [m.hypothesis_states for m in self.measurements],
            "misclassified": [m.misclassified_states for m in self.measurements],
        }


class ConformanceGapAnalyzer:
    """Analyze the gap between k-bounded conformance and true F-bisimulation.

    Given a concrete system and a way to compute the true bisimulation
    quotient (e.g., via Paige-Tarjan partition refinement), this analyzer
    runs the L*-style learner at increasing depths k and measures how
    the hypothesis quotient converges to the true quotient.

    Theorem (Conformance Gap Bound):
        Let H be a hypothesis with n states, diameter d, and alphabet
        size |Act|. If H passes k-bounded conformance testing against
        a concrete system S with at most m states, then either:
        (a) H is equivalent to the minimal quotient of S, or
        (b) any distinguishing sequence has length > k, and the number
            of potentially undetected extra states is at most m - n.
        Moreover, the W-method with depth k >= d + (m - n + 1) is
        complete: if H passes, it is correct.

    Parameters
    ----------
    n_actions : int
        Size of the action alphabet.
    true_quotient_states : int
        Number of states in the true bisimulation quotient (ground truth).
    true_partition : list of frozenset
        The true bisimulation partition (list of equivalence classes).
    """

    def __init__(
        self,
        n_actions: int,
        true_quotient_states: int,
        true_partition: Optional[List[FrozenSet[str]]] = None,
    ) -> None:
        self._n_actions = n_actions
        self._true_n = true_quotient_states
        self._true_partition = true_partition
        self._measurements: List[GapMeasurement] = []

    def compute_error_bound(
        self, k: int, n_hypothesis: int, n_concrete: int
    ) -> float:
        """Compute the theoretical error bound for depth k.

        The bound is derived from the W-method analysis:
        if the concrete system has m states and the hypothesis has n,
        then conformance testing to depth d + (m - n + 1) is complete
        (where d = hypothesis diameter). For k < d + (m - n + 1),
        the fraction of untested state-sequences is at most:

            sum_{i=k-d+1}^{m-n+1} |Act|^i / sum_{i=0}^{m-n+1} |Act|^i

        which is bounded by |Act|^{-(k - d)} when |Act| >= 2.
        """
        if n_hypothesis == 0:
            return 1.0

        d = self._estimate_diameter(n_hypothesis)
        m_minus_n = max(n_concrete - n_hypothesis, 0)
        sufficient = d + m_minus_n + 1

        if k >= sufficient:
            return 0.0  # W-method complete

        if self._n_actions <= 1:
            return 1.0 if k < sufficient else 0.0

        # Exponential decay bound
        exponent = max(k - d, 0)
        bound = self._n_actions ** (-exponent)
        return min(bound, 1.0)

    def _estimate_diameter(self, n: int) -> int:
        """Estimate hypothesis diameter as n-1 (worst case for connected DFA)."""
        return max(n - 1, 1)

    def measure_gap(
        self,
        k: int,
        hypothesis_partition: List[FrozenSet[str]],
        n_concrete: int,
    ) -> GapMeasurement:
        """Measure the gap between a hypothesis partition and the true partition.

        Parameters
        ----------
        k : int
            The conformance testing depth used to produce this hypothesis.
        hypothesis_partition : list of frozenset
            The hypothesis's partition of states into equivalence classes.
        n_concrete : int
            Total number of concrete states.
        """
        n_hyp = len(hypothesis_partition)

        if self._true_partition is None:
            # No ground truth available; use only the error bound
            measurement = GapMeasurement(
                depth_k=k,
                hypothesis_states=n_hyp,
                true_quotient_states=self._true_n,
                misclassified_states=abs(n_hyp - self._true_n),
                extra_classes=max(n_hyp - self._true_n, 0),
                missing_classes=max(self._true_n - n_hyp, 0),
                gap_ratio=abs(n_hyp - self._true_n) / max(self._true_n, 1),
                error_bound=self.compute_error_bound(k, n_hyp, n_concrete),
            )
            self._measurements.append(measurement)
            return measurement

        # Compare partitions element-by-element
        hyp_map = self._partition_to_map(hypothesis_partition)
        true_map = self._partition_to_map(self._true_partition)

        all_states = set(hyp_map.keys()) | set(true_map.keys())
        misclassified = 0
        for s in all_states:
            hyp_class = hyp_map.get(s)
            true_class = true_map.get(s)
            if hyp_class is None or true_class is None:
                misclassified += 1
                continue
            # Check if any pair in the same hyp class is in different true classes
            for t in hyp_class:
                if t in true_map and true_map[t] != true_class:
                    # s and t are merged in hypothesis but distinct in truth
                    misclassified += 1
                    break

        measurement = GapMeasurement(
            depth_k=k,
            hypothesis_states=n_hyp,
            true_quotient_states=self._true_n,
            misclassified_states=misclassified,
            extra_classes=max(n_hyp - self._true_n, 0),
            missing_classes=max(self._true_n - n_hyp, 0),
            gap_ratio=misclassified / max(len(all_states), 1),
            error_bound=self.compute_error_bound(k, n_hyp, len(all_states)),
        )
        self._measurements.append(measurement)
        return measurement

    def _partition_to_map(
        self, partition: List[FrozenSet[str]]
    ) -> Dict[str, FrozenSet[str]]:
        """Convert a partition to a state -> class mapping."""
        result: Dict[str, FrozenSet[str]] = {}
        for cls in partition:
            for s in cls:
                result[s] = cls
        return result

    def analyze(
        self,
        hypothesis_partitions_by_k: Dict[int, List[FrozenSet[str]]],
        n_concrete: int,
    ) -> GapAnalysisResult:
        """Run full gap analysis across multiple depths.

        Parameters
        ----------
        hypothesis_partitions_by_k : dict
            Mapping from depth k to the hypothesis partition at that depth.
        n_concrete : int
            Total number of concrete states.
        """
        self._measurements = []
        for k in sorted(hypothesis_partitions_by_k.keys()):
            self.measure_gap(k, hypothesis_partitions_by_k[k], n_concrete)

        # Determine sufficient k
        sufficient_k = None
        for m in self._measurements:
            if m.error_bound == 0.0 or m.gap_ratio == 0.0:
                sufficient_k = m.depth_k
                break

        # Compute convergence rate (exponential fit)
        convergence_rate = None
        if len(self._measurements) >= 2:
            gaps = [m.gap_ratio for m in self._measurements if m.gap_ratio > 0]
            if len(gaps) >= 2:
                # log-linear regression on gap_ratio vs k
                ks = [m.depth_k for m in self._measurements
                      if m.gap_ratio > 0]
                log_gaps = [math.log(g) for g in gaps]
                if len(ks) >= 2:
                    # slope of log(gap) vs k
                    n = len(ks)
                    mean_k = sum(ks) / n
                    mean_lg = sum(log_gaps) / n
                    num = sum((ks[i] - mean_k) * (log_gaps[i] - mean_lg)
                              for i in range(n))
                    den = sum((ks[i] - mean_k) ** 2 for i in range(n))
                    if den > 0:
                        convergence_rate = -num / den  # positive = faster

        diameter = self._estimate_diameter(
            self._measurements[-1].hypothesis_states
        ) if self._measurements else None

        return GapAnalysisResult(
            measurements=list(self._measurements),
            sufficient_k=sufficient_k,
            convergence_rate=convergence_rate,
            hypothesis_diameter=diameter,
        )

    def sufficient_depth(self, n_hypothesis: int, n_concrete: int) -> int:
        """Compute the minimum depth k for W-method completeness.

        Returns d + (m - n + 1) where d is the hypothesis diameter,
        m is the concrete state count, n is the hypothesis state count.
        """
        d = self._estimate_diameter(n_hypothesis)
        return d + max(n_concrete - n_hypothesis, 0) + 1

    def sufficient_depth_exact(
        self,
        hypothesis_states: int,
        concrete_bound: int,
        diameter: int,
    ) -> int:
        """Compute sufficient depth using an exact diameter value.

        Uses the W-method formula: k >= diam(H) + (m - n + 1).
        """
        return diameter + max(concrete_bound - hypothesis_states, 0) + 1


# ---------------------------------------------------------------------------
# Standalone helper
# ---------------------------------------------------------------------------


def compute_sufficient_depth(
    hypothesis_states: int,
    concrete_bound: int,
    diameter: int,
) -> int:
    """Compute the minimum sufficient conformance testing depth.

    Uses the W-method completeness criterion:
        k >= diam(H) + (m - n + 1)
    where m is the upper bound on concrete states, n is the hypothesis
    state count, and diam(H) is the exact hypothesis diameter.

    Parameters
    ----------
    hypothesis_states : int
        Number of states in the hypothesis (n).
    concrete_bound : int
        Upper bound on concrete system states (m).
    diameter : int
        Exact diameter of the hypothesis.

    Returns
    -------
    int
        Minimum depth k for W-method completeness.
    """
    return diameter + max(concrete_bound - hypothesis_states, 0) + 1


# ---------------------------------------------------------------------------
# ConformanceCompleteCertificate
# ---------------------------------------------------------------------------


@dataclass
class ConformanceCompleteCertificate:
    """Certificate certifying that conformance testing depth k is sufficient.

    Combines the hypothesis diameter, concrete state bound, and the
    W-method formula to prove that the testing depth is adequate.
    """

    hypothesis_states: int = 0
    concrete_bound: int = 0
    diameter: int = 0
    sufficient_depth: int = 0
    actual_depth: int = 0
    is_sufficient: bool = False
    error_bound: float = 1.0
    gap_ratio: float = 1.0
    details: str = ""

    def summary(self) -> str:
        status = "SUFFICIENT" if self.is_sufficient else "INSUFFICIENT"
        return (
            f"ConformanceComplete({status}): k={self.actual_depth}, "
            f"k_suf={self.sufficient_depth}, "
            f"diam={self.diameter}, n={self.hypothesis_states}, "
            f"m={self.concrete_bound}, err={self.error_bound:.6f}"
        )

    @classmethod
    def build(
        cls,
        hypothesis_states: int,
        concrete_bound: int,
        diameter: int,
        actual_depth: int,
        n_actions: int = 2,
    ) -> "ConformanceCompleteCertificate":
        """Build a certificate from the given parameters."""
        suf = compute_sufficient_depth(hypothesis_states, concrete_bound, diameter)
        is_suf = actual_depth >= suf

        if is_suf:
            err = 0.0
        elif n_actions <= 1:
            err = 1.0
        else:
            exponent = max(actual_depth - diameter, 0)
            err = min(n_actions ** (-exponent), 1.0) if exponent > 0 else 1.0

        gap = 0.0 if is_suf else max(suf - actual_depth, 0) / max(suf, 1)

        return cls(
            hypothesis_states=hypothesis_states,
            concrete_bound=concrete_bound,
            diameter=diameter,
            sufficient_depth=suf,
            actual_depth=actual_depth,
            is_sufficient=is_suf,
            error_bound=err,
            gap_ratio=gap,
            details=(
                f"k={actual_depth} {'≥' if is_suf else '<'} "
                f"diam({diameter}) + (m({concrete_bound}) - n({hypothesis_states}) + 1) "
                f"= {suf}"
            ),
        )
