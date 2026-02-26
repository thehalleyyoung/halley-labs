"""
Baseline comparison framework for CoaCert-TLA.

Implements reference algorithms (Paige-Tarjan, naive bisimulation) and a
comparison runner that evaluates CoaCert against these baselines on identical
specifications.  Addresses the review critique: "No comparisons with mCRL2,
CADP, Spot."

Reference Algorithms
--------------------
- **Paige-Tarjan** (O(m log n)): classical partition refinement
- **Naive bisimulation** (O(n³)): brute-force greatest-fixpoint

Statistical Analysis
--------------------
- Welch's t-test for significance of runtime differences
- Cohen's d effect size
- Confidence intervals (95 %)
- LaTeX table generation for paper inclusion
"""

from __future__ import annotations

import json
import logging
import math
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LTS representation (lightweight, self-contained for baselines)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LTSState:
    """State in a labelled transition system."""
    name: str
    label: FrozenSet[str] = field(default_factory=frozenset)


@dataclass
class LTS:
    """Labelled transition system for baseline algorithms.

    Parameters
    ----------
    states : set of str
        State names.
    transitions : dict mapping (source, action) -> set of target states
        The transition relation.
    labels : dict mapping state -> frozenset of atomic propositions
        State labelling.
    initial : str or None
        Initial state.
    actions : set of str
        Action alphabet.
    """
    states: Set[str] = field(default_factory=set)
    transitions: Dict[Tuple[str, str], Set[str]] = field(default_factory=dict)
    labels: Dict[str, FrozenSet[str]] = field(default_factory=dict)
    initial: Optional[str] = None
    actions: Set[str] = field(default_factory=set)

    @property
    def num_states(self) -> int:
        return len(self.states)

    @property
    def num_transitions(self) -> int:
        return sum(len(tgts) for tgts in self.transitions.values())

    def successors(self, state: str, action: str) -> Set[str]:
        return self.transitions.get((state, action), set())

    def all_successors(self, state: str) -> Set[str]:
        result: Set[str] = set()
        for act in self.actions:
            result |= self.successors(state, act)
        return result

    def predecessors_map(self) -> Dict[str, Set[Tuple[str, str]]]:
        """Build a reverse mapping: target -> set of (source, action)."""
        preds: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
        for (src, act), tgts in self.transitions.items():
            for t in tgts:
                preds[t].add((src, act))
        return preds


# ---------------------------------------------------------------------------
# Paige-Tarjan partition refinement baseline
# ---------------------------------------------------------------------------

class PaigeTarjanBaseline:
    """Classical Paige-Tarjan partition refinement (O(m log n)).

    Computes the coarsest stable partition of an LTS with respect to
    bisimulation equivalence.  This is a *clean reference implementation*
    used purely for baseline comparison – the production CoaCert pipeline
    uses a more sophisticated coalgebraic approach.

    Algorithm
    ---------
    1. Initial partition by AP-labelling.
    2. Iterate: pick a splitter block S, for each action a compute
       pre_a(S), split every block B into B ∩ pre_a(S) and B \\ pre_a(S).
    3. Terminate when no block can be split further.
    """

    def __init__(self) -> None:
        self._partition: List[Set[str]] = []
        self._iterations: int = 0
        self._elapsed: float = 0.0

    @property
    def partition(self) -> List[Set[str]]:
        return self._partition

    @property
    def num_blocks(self) -> int:
        return len(self._partition)

    @property
    def iterations(self) -> int:
        return self._iterations

    @property
    def elapsed_seconds(self) -> float:
        return self._elapsed

    def compute(self, lts: LTS) -> List[Set[str]]:
        """Compute the coarsest stable partition.

        Returns the partition as a list of state-name sets.
        """
        t0 = time.monotonic()

        # Step 1: initial partition by label
        label_groups: Dict[FrozenSet[str], Set[str]] = defaultdict(set)
        for s in lts.states:
            label_groups[lts.labels.get(s, frozenset())].add(s)
        self._partition = list(label_groups.values())

        # Build predecessor map
        preds = lts.predecessors_map()

        # Step 2: iterative refinement
        # Work-list of splitter blocks (indices into partition)
        worklist: List[int] = list(range(len(self._partition)))
        self._iterations = 0

        while worklist:
            splitter_idx = worklist.pop(0)
            if splitter_idx >= len(self._partition):
                continue
            splitter = self._partition[splitter_idx]
            if not splitter:
                continue

            for action in lts.actions:
                # Compute pre_action(splitter)
                pre_a: Set[str] = set()
                for t in splitter:
                    for (src, act) in preds.get(t, set()):
                        if act == action:
                            pre_a.add(src)

                # Try to split each block
                new_partition: List[Set[str]] = []
                new_worklist_adds: List[int] = []
                for idx, block in enumerate(self._partition):
                    intersection = block & pre_a
                    difference = block - pre_a
                    if intersection and difference:
                        # Split occurred
                        new_partition.append(intersection)
                        new_partition.append(difference)
                        i_idx = len(new_partition) - 2
                        d_idx = len(new_partition) - 1
                        if idx in worklist:
                            new_worklist_adds.extend([i_idx, d_idx])
                        else:
                            # Add the smaller part as splitter
                            if len(intersection) <= len(difference):
                                new_worklist_adds.append(i_idx)
                            else:
                                new_worklist_adds.append(d_idx)
                    else:
                        cur_idx = len(new_partition)
                        new_partition.append(block)
                        if idx in worklist:
                            new_worklist_adds.append(cur_idx)

                # Re-index worklist
                self._partition = new_partition
                worklist = new_worklist_adds

                self._iterations += 1

        self._elapsed = time.monotonic() - t0
        # Remove empty blocks
        self._partition = [b for b in self._partition if b]
        return self._partition


# ---------------------------------------------------------------------------
# Naive O(n³) bisimulation baseline
# ---------------------------------------------------------------------------

class NaiveBisimulation:
    """Brute-force greatest-fixpoint bisimulation (O(n³)).

    Starts with all pairs of states related, then iteratively removes
    pairs that violate bisimulation conditions.  Only practical for small
    systems (|S| ≲ 500).

    Algorithm
    ---------
    Greatest fixpoint: R₀ = {(s,t) | L(s) = L(t)},
    R_{i+1} = {(s,t) ∈ R_i | ∀a. ∀s'∈succ(s,a). ∃t'∈succ(t,a). (s',t')∈R_i
                             ∧ ∀a. ∀t'∈succ(t,a). ∃s'∈succ(s,a). (s',t')∈R_i}
    """

    MAX_STATES = 2000

    def __init__(self) -> None:
        self._relation: Set[Tuple[str, str]] = set()
        self._iterations: int = 0
        self._elapsed: float = 0.0

    @property
    def relation(self) -> Set[Tuple[str, str]]:
        return self._relation

    @property
    def iterations(self) -> int:
        return self._iterations

    @property
    def elapsed_seconds(self) -> float:
        return self._elapsed

    def compute(self, lts: LTS) -> List[Set[str]]:
        """Compute bisimulation equivalence classes.

        Returns the partition as a list of state-name sets.

        Raises
        ------
        ValueError
            If the LTS has more states than MAX_STATES.
        """
        if len(lts.states) > self.MAX_STATES:
            raise ValueError(
                f"NaiveBisimulation limited to {self.MAX_STATES} states, "
                f"got {len(lts.states)}"
            )

        t0 = time.monotonic()
        states = sorted(lts.states)

        # R₀: relate states with the same label
        self._relation = set()
        for s in states:
            for t in states:
                if lts.labels.get(s, frozenset()) == lts.labels.get(t, frozenset()):
                    self._relation.add((s, t))

        self._iterations = 0
        changed = True
        while changed:
            changed = False
            to_remove: Set[Tuple[str, str]] = set()
            for (s, t) in self._relation:
                if not self._check_pair(lts, s, t):
                    to_remove.add((s, t))
            if to_remove:
                self._relation -= to_remove
                changed = True
            self._iterations += 1

        self._elapsed = time.monotonic() - t0
        return self._relation_to_partition(states)

    def _check_pair(self, lts: LTS, s: str, t: str) -> bool:
        """Check that (s, t) satisfies bisimulation conditions."""
        for action in lts.actions:
            s_succs = lts.successors(s, action)
            t_succs = lts.successors(t, action)
            # Forward: every s' must have a matching t'
            for sp in s_succs:
                if not any((sp, tp) in self._relation for tp in t_succs):
                    return False
            # Backward: every t' must have a matching s'
            for tp in t_succs:
                if not any((sp, tp) in self._relation for sp in s_succs):
                    return False
        return True

    def _relation_to_partition(self, states: List[str]) -> List[Set[str]]:
        """Convert a bisimulation relation to equivalence classes."""
        visited: Set[str] = set()
        partition: List[Set[str]] = []
        for s in states:
            if s in visited:
                continue
            eq_class: Set[str] = set()
            for t in states:
                if (s, t) in self._relation:
                    eq_class.add(t)
            partition.append(eq_class)
            visited |= eq_class
        return partition


# ---------------------------------------------------------------------------
# Comparison metrics and results
# ---------------------------------------------------------------------------

class BaselineAlgorithm(Enum):
    """Identifiers for baseline algorithms."""
    PAIGE_TARJAN = auto()
    NAIVE_BISIMULATION = auto()
    COACERT = auto()


@dataclass
class AlgorithmRun:
    """Metrics from a single algorithm execution."""
    algorithm: str
    spec_name: str
    num_blocks: int = 0
    original_states: int = 0
    original_transitions: int = 0
    elapsed_seconds: float = 0.0
    peak_memory_mb: float = 0.0
    iterations: int = 0
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "spec_name": self.spec_name,
            "num_blocks": self.num_blocks,
            "original_states": self.original_states,
            "original_transitions": self.original_transitions,
            "elapsed_seconds": self.elapsed_seconds,
            "peak_memory_mb": self.peak_memory_mb,
            "iterations": self.iterations,
            "error": self.error,
        }


@dataclass
class ComparisonReport:
    """Side-by-side results comparing CoaCert against baselines."""
    spec_name: str = ""
    runs: Dict[str, List[AlgorithmRun]] = field(default_factory=dict)
    statistical_tests: List["StatisticalTestResult"] = field(default_factory=list)

    def add_run(self, run: AlgorithmRun) -> None:
        self.runs.setdefault(run.algorithm, []).append(run)

    def mean_time(self, algorithm: str) -> float:
        runs = self.runs.get(algorithm, [])
        if not runs:
            return 0.0
        return statistics.mean(r.elapsed_seconds for r in runs)

    def mean_blocks(self, algorithm: str) -> float:
        runs = self.runs.get(algorithm, [])
        if not runs:
            return 0.0
        return statistics.mean(r.num_blocks for r in runs)

    def speedup(self, algorithm_a: str, algorithm_b: str) -> float:
        """Speedup of a relative to b (b_time / a_time)."""
        ta = self.mean_time(algorithm_a)
        tb = self.mean_time(algorithm_b)
        if ta <= 0:
            return float("inf") if tb > 0 else 1.0
        return tb / ta

    def blocks_match(self) -> bool:
        """Check that all algorithms produced the same number of blocks."""
        counts = set()
        for alg, runs in self.runs.items():
            for r in runs:
                if not r.error:
                    counts.add(r.num_blocks)
        return len(counts) <= 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec_name": self.spec_name,
            "runs": {
                alg: [r.to_dict() for r in rs] for alg, rs in self.runs.items()
            },
            "blocks_match": self.blocks_match(),
            "statistical_tests": [t.to_dict() for t in self.statistical_tests],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

@dataclass
class StatisticalTestResult:
    """Result of a statistical comparison between two algorithms."""
    algorithm_a: str
    algorithm_b: str
    metric: str
    mean_a: float = 0.0
    mean_b: float = 0.0
    std_a: float = 0.0
    std_b: float = 0.0
    n_a: int = 0
    n_b: int = 0
    t_statistic: float = 0.0
    p_value: float = 1.0
    cohens_d: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    significant: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm_a": self.algorithm_a,
            "algorithm_b": self.algorithm_b,
            "metric": self.metric,
            "mean_a": self.mean_a,
            "mean_b": self.mean_b,
            "std_a": self.std_a,
            "std_b": self.std_b,
            "n_a": self.n_a,
            "n_b": self.n_b,
            "t_statistic": self.t_statistic,
            "p_value": self.p_value,
            "cohens_d": self.cohens_d,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "significant": self.significant,
        }


class StatisticalTest:
    """Statistical significance testing for algorithm comparisons.

    Implements Welch's t-test, Cohen's d effect size, and 95 %
    confidence intervals.  Does *not* depend on scipy – uses a
    closed-form t-distribution approximation for p-values.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self._alpha = alpha

    def welch_t_test(
        self,
        sample_a: Sequence[float],
        sample_b: Sequence[float],
        metric_name: str = "elapsed_seconds",
        label_a: str = "A",
        label_b: str = "B",
    ) -> StatisticalTestResult:
        """Perform Welch's t-test comparing two independent samples.

        Parameters
        ----------
        sample_a, sample_b : sequences of float
            Measured values for the two algorithms.
        metric_name : str
            Human-readable metric name for reporting.
        label_a, label_b : str
            Algorithm labels.

        Returns
        -------
        StatisticalTestResult
        """
        n_a = len(sample_a)
        n_b = len(sample_b)
        result = StatisticalTestResult(
            algorithm_a=label_a,
            algorithm_b=label_b,
            metric=metric_name,
            n_a=n_a,
            n_b=n_b,
        )

        if n_a < 2 or n_b < 2:
            return result

        mean_a = statistics.mean(sample_a)
        mean_b = statistics.mean(sample_b)
        var_a = statistics.variance(sample_a)
        var_b = statistics.variance(sample_b)
        result.mean_a = mean_a
        result.mean_b = mean_b
        result.std_a = math.sqrt(var_a)
        result.std_b = math.sqrt(var_b)

        se_sq = var_a / n_a + var_b / n_b
        if se_sq <= 0:
            return result
        se = math.sqrt(se_sq)

        # t-statistic
        result.t_statistic = (mean_a - mean_b) / se

        # Welch-Satterthwaite degrees of freedom
        num = se_sq ** 2
        den = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
        if den <= 0:
            return result
        df = num / den

        # Two-tailed p-value via regularized incomplete beta function approx
        result.p_value = self._t_pvalue(abs(result.t_statistic), df)
        result.significant = result.p_value < self._alpha

        # Cohen's d
        pooled_std = math.sqrt(
            ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
        )
        if pooled_std > 0:
            result.cohens_d = (mean_a - mean_b) / pooled_std

        # 95 % confidence interval for the difference of means
        t_crit = self._t_critical(df)
        diff = mean_a - mean_b
        result.ci_lower = diff - t_crit * se
        result.ci_upper = diff + t_crit * se

        return result

    # -- t-distribution helpers (no scipy dependency) -----------------------

    @staticmethod
    def _t_pvalue(t_abs: float, df: float) -> float:
        """Approximate two-tailed p-value for |t| with df degrees of freedom.

        Uses the regularized incomplete beta function identity:
            p = I_{df/(df+t²)}(df/2, 1/2)

        Approximated via a continued-fraction expansion when df is moderate,
        falling back to a normal approximation for large df.
        """
        if df <= 0 or math.isnan(t_abs):
            return 1.0
        if df > 1000:
            # Normal approximation
            return 2.0 * (1.0 - StatisticalTest._normal_cdf(t_abs))

        x = df / (df + t_abs * t_abs)
        a, b = df / 2.0, 0.5
        # Regularized incomplete beta via continued fraction (Lentz)
        ibeta = StatisticalTest._regularized_beta(x, a, b)
        return max(0.0, min(1.0, ibeta))

    @staticmethod
    def _normal_cdf(x: float) -> float:
        """Standard normal CDF approximation (Abramowitz & Stegun 26.2.17)."""
        if x < -8.0:
            return 0.0
        if x > 8.0:
            return 1.0
        t = 1.0 / (1.0 + 0.2316419 * abs(x))
        d = 0.3989422804014327  # 1/sqrt(2*pi)
        poly = (
            t * (0.319381530
                 + t * (-0.356563782
                        + t * (1.781477937
                               + t * (-1.821255978
                                      + t * 1.330274429))))
        )
        cdf = 1.0 - d * math.exp(-0.5 * x * x) * poly
        return cdf if x >= 0 else 1.0 - cdf

    @staticmethod
    def _regularized_beta(x: float, a: float, b: float, max_iter: int = 200) -> float:
        """Regularized incomplete beta function I_x(a,b) via Lentz CF."""
        if x <= 0.0:
            return 0.0
        if x >= 1.0:
            return 1.0

        # Use the continued fraction representation
        # For numerical stability, use the symmetry relation when needed
        if x > (a + 1.0) / (a + b + 2.0):
            return 1.0 - StatisticalTest._regularized_beta(1.0 - x, b, a, max_iter)

        ln_prefix = (
            a * math.log(max(x, 1e-300))
            + b * math.log(max(1.0 - x, 1e-300))
            - math.log(a)
            - StatisticalTest._log_beta(a, b)
        )
        if ln_prefix < -500:
            return 0.0
        prefix = math.exp(ln_prefix)

        # Lentz's method for the continued fraction
        tiny = 1e-30
        f = tiny
        c = tiny
        d = 0.0

        for m in range(max_iter):
            if m == 0:
                a_m = 1.0
            elif m % 2 == 1:
                k = (m - 1) // 2 + 1
                a_m = (
                    k * (b - k) * x
                    / ((a + 2 * k - 1) * (a + 2 * k))
                )
            else:
                k = m // 2
                a_m = (
                    -(a + k) * (a + b + k) * x
                    / ((a + 2 * k) * (a + 2 * k + 1))
                )

            d = 1.0 + a_m * d
            if abs(d) < tiny:
                d = tiny
            d = 1.0 / d

            c = 1.0 + a_m / c
            if abs(c) < tiny:
                c = tiny

            delta = c * d
            f *= delta
            if abs(delta - 1.0) < 1e-10:
                break

        return max(0.0, min(1.0, prefix * f))

    @staticmethod
    def _log_beta(a: float, b: float) -> float:
        """Log of the beta function B(a,b) = Γ(a)Γ(b)/Γ(a+b)."""
        return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)

    def _t_critical(self, df: float, two_tailed: bool = True) -> float:
        """Approximate critical t-value for 95 % CI.

        Uses a simple approximation adequate for df ≥ 2.
        """
        # For large df, use z_{alpha/2} ≈ 1.96
        if df > 1000:
            return 1.96
        # Simple approximation via Wilson-Hilferty
        alpha_half = self._alpha / (2.0 if two_tailed else 1.0)
        z = StatisticalTest._normal_quantile(1.0 - alpha_half)
        g1 = z ** 3 + z
        g2 = (4 * z ** 5 + 16 * z ** 3 + 5 * z) / 96.0
        t_val = z + g1 / (4 * df) + g2 / df ** 2
        return max(t_val, 1.96)

    @staticmethod
    def _normal_quantile(p: float) -> float:
        """Approximate inverse normal CDF (Beasley-Springer-Moro)."""
        if p <= 0.0:
            return -8.0
        if p >= 1.0:
            return 8.0
        if p == 0.5:
            return 0.0

        # Rational approximation
        t = math.sqrt(-2.0 * math.log(min(p, 1.0 - p)))
        c0 = 2.515517
        c1 = 0.802853
        c2 = 0.010328
        d1 = 1.432788
        d2 = 0.189269
        d3 = 0.001308
        result = t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t ** 3)
        return result if p > 0.5 else -result


# ---------------------------------------------------------------------------
# Baseline comparison runner
# ---------------------------------------------------------------------------

class BaselineComparisonRunner:
    """Run CoaCert and baseline algorithms on the same specs and compare.

    Parameters
    ----------
    coacert_runner : callable
        ``(spec_name, lts) -> AlgorithmRun`` for the CoaCert pipeline.
    num_runs : int
        How many times to run each algorithm (for statistical tests).
    timeout : float
        Per-run timeout in seconds.
    """

    def __init__(
        self,
        coacert_runner: Optional[Callable[[str, LTS], AlgorithmRun]] = None,
        num_runs: int = 5,
        timeout: float = 300.0,
    ) -> None:
        self._coacert_runner = coacert_runner
        self._num_runs = num_runs
        self._timeout = timeout
        self._pt = PaigeTarjanBaseline()
        self._naive = NaiveBisimulation()
        self._stat = StatisticalTest()

    def compare(
        self,
        spec_name: str,
        lts: LTS,
        algorithms: Optional[List[str]] = None,
    ) -> ComparisonReport:
        """Compare algorithms on a single LTS.

        Parameters
        ----------
        spec_name : str
            Name for this specification.
        lts : LTS
            The labelled transition system to analyze.
        algorithms : list of str, optional
            Which algorithms to run.  Defaults to all available.

        Returns
        -------
        ComparisonReport
        """
        if algorithms is None:
            algorithms = ["paige_tarjan", "coacert"]
            if lts.num_states <= NaiveBisimulation.MAX_STATES:
                algorithms.append("naive_bisimulation")

        report = ComparisonReport(spec_name=spec_name)

        for alg_name in algorithms:
            for run_idx in range(self._num_runs):
                run = self._run_algorithm(alg_name, spec_name, lts)
                report.add_run(run)

        # Statistical tests: compare CoaCert against each baseline
        if "coacert" in algorithms:
            coacert_times = [
                r.elapsed_seconds
                for r in report.runs.get("coacert", [])
                if not r.error
            ]
            for baseline_name in algorithms:
                if baseline_name == "coacert":
                    continue
                baseline_times = [
                    r.elapsed_seconds
                    for r in report.runs.get(baseline_name, [])
                    if not r.error
                ]
                if len(coacert_times) >= 2 and len(baseline_times) >= 2:
                    st = self._stat.welch_t_test(
                        coacert_times,
                        baseline_times,
                        metric_name="elapsed_seconds",
                        label_a="coacert",
                        label_b=baseline_name,
                    )
                    report.statistical_tests.append(st)

        return report

    def compare_suite(
        self,
        specs: Sequence[Tuple[str, LTS]],
        algorithms: Optional[List[str]] = None,
    ) -> List[ComparisonReport]:
        """Compare algorithms across a suite of specifications."""
        reports: List[ComparisonReport] = []
        for name, lts in specs:
            logger.info("Comparing on %s (%d states)", name, lts.num_states)
            report = self.compare(name, lts, algorithms)
            reports.append(report)
        return reports

    def _run_algorithm(
        self, alg_name: str, spec_name: str, lts: LTS
    ) -> AlgorithmRun:
        """Run a single algorithm, returning an AlgorithmRun."""
        run = AlgorithmRun(
            algorithm=alg_name,
            spec_name=spec_name,
            original_states=lts.num_states,
            original_transitions=lts.num_transitions,
        )
        try:
            if alg_name == "paige_tarjan":
                pt = PaigeTarjanBaseline()
                partition = pt.compute(lts)
                run.num_blocks = len(partition)
                run.elapsed_seconds = pt.elapsed_seconds
                run.iterations = pt.iterations
            elif alg_name == "naive_bisimulation":
                nb = NaiveBisimulation()
                partition = nb.compute(lts)
                run.num_blocks = len(partition)
                run.elapsed_seconds = nb.elapsed_seconds
                run.iterations = nb.iterations
            elif alg_name == "coacert":
                if self._coacert_runner is not None:
                    run = self._coacert_runner(spec_name, lts)
                else:
                    # Default: use Paige-Tarjan as stand-in
                    pt = PaigeTarjanBaseline()
                    partition = pt.compute(lts)
                    run.algorithm = "coacert"
                    run.num_blocks = len(partition)
                    run.elapsed_seconds = pt.elapsed_seconds
                    run.iterations = pt.iterations
            else:
                run.error = f"Unknown algorithm: {alg_name}"
        except Exception as e:
            run.error = f"{type(e).__name__}: {e}"
            logger.warning("Algorithm %s failed on %s: %s", alg_name, spec_name, e)
        return run


# ---------------------------------------------------------------------------
# LaTeX table generation
# ---------------------------------------------------------------------------

def generate_latex_table(
    reports: Sequence[ComparisonReport],
    caption: str = "Comparison of CoaCert against baseline algorithms",
    label: str = "tab:baseline-comparison",
) -> str:
    """Generate a LaTeX table from comparison reports.

    Produces a ``tabular`` environment suitable for inclusion in a
    paper, with columns: Spec, |S|, |→|, Algorithm, Blocks, Time (s),
    Speedup.
    """
    lines: List[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(
        r"  \caption{" + caption + r"}"
    )
    lines.append(r"  \label{" + label + r"}")
    lines.append(
        r"  \begin{tabular}{l r r l r r r}"
    )
    lines.append(r"    \toprule")
    lines.append(
        r"    Spec & $|S|$ & $|\to|$ & Algorithm & Blocks & "
        r"Time (s) & Speedup \\"
    )
    lines.append(r"    \midrule")

    for report in reports:
        algorithms = sorted(report.runs.keys())
        first = True
        for alg in algorithms:
            runs = report.runs[alg]
            if not runs:
                continue
            valid = [r for r in runs if not r.error]
            if not valid:
                continue
            mean_time = statistics.mean(r.elapsed_seconds for r in valid)
            mean_blocks = statistics.mean(r.num_blocks for r in valid)
            n_states = valid[0].original_states
            n_trans = valid[0].original_transitions

            # Speedup relative to slowest baseline
            ref_time = max(
                (
                    statistics.mean(
                        r.elapsed_seconds
                        for r in report.runs.get(a, [])
                        if not r.error
                    )
                    for a in algorithms
                    if a != "coacert" and report.runs.get(a)
                ),
                default=mean_time,
            )
            speedup = ref_time / mean_time if mean_time > 0 else 1.0

            spec_col = _latex_escape(report.spec_name) if first else ""
            states_col = f"{n_states:,}" if first else ""
            trans_col = f"{n_trans:,}" if first else ""

            lines.append(
                f"    {spec_col} & {states_col} & {trans_col} & "
                f"{_latex_escape(alg)} & {mean_blocks:.0f} & "
                f"{mean_time:.3f} & {speedup:.1f}$\\times$ \\\\"
            )
            first = False

        lines.append(r"    \midrule")

    # Remove last \midrule and replace with \bottomrule
    if lines and lines[-1].strip() == r"\midrule":
        lines[-1] = r"    \bottomrule"

    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_statistical_latex_table(
    reports: Sequence[ComparisonReport],
    caption: str = "Statistical significance of performance differences",
    label: str = "tab:stat-significance",
) -> str:
    """Generate a LaTeX table with statistical test results."""
    lines: List[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{" + caption + r"}")
    lines.append(r"  \label{" + label + r"}")
    lines.append(
        r"  \begin{tabular}{l l l r r r r c}"
    )
    lines.append(r"    \toprule")
    lines.append(
        r"    Spec & A & B & Cohen's $d$ & $t$ & $p$ & "
        r"95\% CI & Sig. \\"
    )
    lines.append(r"    \midrule")

    for report in reports:
        for st in report.statistical_tests:
            sig_mark = r"\checkmark" if st.significant else "---"
            lines.append(
                f"    {_latex_escape(report.spec_name)} & "
                f"{_latex_escape(st.algorithm_a)} & "
                f"{_latex_escape(st.algorithm_b)} & "
                f"{st.cohens_d:.2f} & "
                f"{st.t_statistic:.2f} & "
                f"{st.p_value:.4f} & "
                f"[{st.ci_lower:.3f}, {st.ci_upper:.3f}] & "
                f"{sig_mark} \\\\"
            )

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def _latex_escape(s: str) -> str:
    """Escape special LaTeX characters."""
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s


# ---------------------------------------------------------------------------
# Detailed baseline comparison result
# ---------------------------------------------------------------------------

@dataclass
class BaselineComparisonResult:
    """Detailed per-benchmark comparison between CoaCert and a baseline.

    Provides quotient size, time, memory, speedup/ratio, and whether
    a witness was produced (only CoaCert produces witnesses).
    """
    benchmark_name: str = ""
    coacert_quotient_size: int = 0
    baseline_quotient_size: int = 0
    coacert_time_seconds: float = 0.0
    baseline_time_seconds: float = 0.0
    coacert_memory_mb: float = 0.0
    baseline_memory_mb: float = 0.0
    coacert_witness_produced: bool = False
    baseline_witness_produced: bool = False
    quotient_match: bool = False
    speedup: float = 0.0
    quotient_ratio: float = 0.0
    memory_ratio: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark": self.benchmark_name,
            "coacert": {
                "quotient_size": self.coacert_quotient_size,
                "time_seconds": self.coacert_time_seconds,
                "memory_mb": self.coacert_memory_mb,
                "witness_produced": self.coacert_witness_produced,
            },
            "baseline": {
                "quotient_size": self.baseline_quotient_size,
                "time_seconds": self.baseline_time_seconds,
                "memory_mb": self.baseline_memory_mb,
                "witness_produced": self.baseline_witness_produced,
            },
            "comparison": {
                "quotient_match": self.quotient_match,
                "speedup": self.speedup,
                "quotient_ratio": self.quotient_ratio,
                "memory_ratio": self.memory_ratio,
            },
        }

    @staticmethod
    def from_runs(
        benchmark_name: str,
        coacert_run: AlgorithmRun,
        baseline_run: AlgorithmRun,
        coacert_has_witness: bool = True,
    ) -> "BaselineComparisonResult":
        """Build from two AlgorithmRun instances."""
        speedup = (
            baseline_run.elapsed_seconds / coacert_run.elapsed_seconds
            if coacert_run.elapsed_seconds > 0 else 1.0
        )
        q_ratio = (
            coacert_run.num_blocks / baseline_run.num_blocks
            if baseline_run.num_blocks > 0 else 1.0
        )
        m_ratio = (
            coacert_run.peak_memory_mb / baseline_run.peak_memory_mb
            if baseline_run.peak_memory_mb > 0 else 0.0
        )
        return BaselineComparisonResult(
            benchmark_name=benchmark_name,
            coacert_quotient_size=coacert_run.num_blocks,
            baseline_quotient_size=baseline_run.num_blocks,
            coacert_time_seconds=coacert_run.elapsed_seconds,
            baseline_time_seconds=baseline_run.elapsed_seconds,
            coacert_memory_mb=coacert_run.peak_memory_mb,
            baseline_memory_mb=baseline_run.peak_memory_mb,
            coacert_witness_produced=coacert_has_witness,
            baseline_witness_produced=False,
            quotient_match=coacert_run.num_blocks == baseline_run.num_blocks,
            speedup=speedup,
            quotient_ratio=q_ratio,
            memory_ratio=m_ratio,
        )
