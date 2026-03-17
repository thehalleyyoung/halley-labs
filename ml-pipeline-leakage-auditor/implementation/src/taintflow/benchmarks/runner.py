"""
taintflow.benchmarks.runner – Benchmark execution and result analysis.

This module provides the machinery to *run* benchmark scenarios against
TaintFlow's analysis pipeline, collect quantitative results, and
compare them to ground-truth records.  It also includes profiling and
regression-detection utilities for CI integration.

Public API
----------
* :class:`BenchmarkResult`     – single-scenario result with precision /
                                  recall / F1 metrics.
* :class:`BenchmarkReport`     – aggregate report across many scenarios.
* :class:`BenchmarkSuite`      – named collection of scenarios + runner.
* :class:`BenchmarkRunner`     – execute a scenario and score the output.
* :class:`PerformanceProfiler` – wall-clock / memory profiling wrapper.
* :class:`RegressionDetector`  – detect regressions relative to a saved
                                  baseline.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from taintflow.core.types import OpType, Origin, Severity

if TYPE_CHECKING:
    from taintflow.dag.pidag import PIDAG
    from taintflow.benchmarks.synthetic import (
        DatasetConfig,
        GroundTruth,
    )
    from taintflow.benchmarks.scenarios import BenchmarkScenario

# ===================================================================
#  Constants
# ===================================================================

_DEFAULT_BIT_TOLERANCE: float = 0.5
_DEFAULT_DETECTION_THRESHOLD: float = 0.01
_METRIC_PRECISION: int = 6
_NS_PER_MS: float = 1_000_000.0


# ===================================================================
#  BenchmarkResult
# ===================================================================


@dataclass
class BenchmarkResult:
    """Quantitative result for a single benchmark scenario.

    The result records both the *detected* leakage (from TaintFlow) and
    the *ground-truth* leakage so that standard information-retrieval
    metrics (precision, recall, F1) can be computed.

    Attributes
    ----------
    scenario_name:
        Human-readable name of the scenario.
    detected_leakage_bits:
        ``{feature_name: bits}`` as reported by the analysis.
    ground_truth_bits:
        ``{feature_name: bits}`` from the :class:`GroundTruth`.
    precision:
        Proportion of detected features that are truly leaking.
    recall:
        Proportion of truly leaking features that are detected.
    f1_score:
        Harmonic mean of precision and recall.
    false_positives:
        Feature names detected as leaking but clean in ground truth.
    false_negatives:
        Feature names that leak but were not detected.
    analysis_time_ms:
        Wall-clock time for the analysis in milliseconds.
    memory_peak_mb:
        Peak RSS memory during analysis, in megabytes.
    bit_tolerance:
        Tolerance (in bits) used when comparing detected ↔ truth.
    detection_threshold:
        Minimum bits for a feature to be considered "detected".
    metadata:
        Arbitrary extra fields carried through serialisation.
    """

    scenario_name: str = ""
    detected_leakage_bits: Dict[str, float] = field(
        default_factory=dict,
    )
    ground_truth_bits: Dict[str, float] = field(
        default_factory=dict,
    )
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    false_positives: List[str] = field(default_factory=list)
    false_negatives: List[str] = field(default_factory=list)
    analysis_time_ms: float = 0.0
    memory_peak_mb: float = 0.0
    bit_tolerance: float = _DEFAULT_BIT_TOLERANCE
    detection_threshold: float = _DEFAULT_DETECTION_THRESHOLD
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- derived properties ----------------------------------------

    @property
    def n_detected(self) -> int:
        """Number of features reported as leaking."""
        return sum(
            1
            for v in self.detected_leakage_bits.values()
            if v >= self.detection_threshold
        )

    @property
    def n_ground_truth_leaking(self) -> int:
        """Number of features that truly leak."""
        return sum(
            1
            for v in self.ground_truth_bits.values()
            if v > 0.0
        )

    @property
    def n_false_positives(self) -> int:
        """Count of false positives."""
        return len(self.false_positives)

    @property
    def n_false_negatives(self) -> int:
        """Count of false negatives."""
        return len(self.false_negatives)

    @property
    def total_detected_bits(self) -> float:
        """Sum of all detected leakage bits."""
        return sum(self.detected_leakage_bits.values())

    @property
    def total_ground_truth_bits(self) -> float:
        """Sum of all ground-truth leakage bits."""
        return sum(self.ground_truth_bits.values())

    @property
    def bit_error(self) -> float:
        """Absolute error between total detected and truth bits."""
        return abs(self.total_detected_bits - self.total_ground_truth_bits)

    @property
    def relative_bit_error(self) -> float:
        """Relative error normalised by ground-truth total."""
        gt = self.total_ground_truth_bits
        if gt == 0.0:
            return 0.0 if self.total_detected_bits == 0.0 else float("inf")
        return self.bit_error / gt

    @property
    def is_sound(self) -> bool:
        """``True`` when every leaking feature is detected (recall = 1)."""
        return self.recall >= 1.0 - 1e-9

    @property
    def is_tight(self) -> bool:
        """``True`` when no clean feature is falsely flagged."""
        return len(self.false_positives) == 0

    @property
    def severity(self) -> Severity:
        """Worst-case :class:`Severity` of the detected leakage."""
        if not self.detected_leakage_bits:
            return Severity.NEGLIGIBLE
        max_bits = max(self.detected_leakage_bits.values())
        return Severity.from_bits(max_bits)

    @property
    def passed(self) -> bool:
        """``True`` when precision ≥ 0.9 and recall ≥ 0.9."""
        return self.precision >= 0.9 and self.recall >= 0.9

    # -- validation ------------------------------------------------

    def validate(self) -> list[str]:
        """Return a list of validation error strings (empty ⇒ valid)."""
        errors: list[str] = []
        if not self.scenario_name:
            errors.append("scenario_name must not be empty")
        if not (0.0 <= self.precision <= 1.0):
            errors.append(
                f"precision out of [0, 1]: {self.precision}"
            )
        if not (0.0 <= self.recall <= 1.0):
            errors.append(f"recall out of [0, 1]: {self.recall}")
        if not (0.0 <= self.f1_score <= 1.0):
            errors.append(f"f1_score out of [0, 1]: {self.f1_score}")
        if self.analysis_time_ms < 0.0:
            errors.append(
                f"analysis_time_ms must be ≥ 0, "
                f"got {self.analysis_time_ms}"
            )
        if self.memory_peak_mb < 0.0:
            errors.append(
                f"memory_peak_mb must be ≥ 0, "
                f"got {self.memory_peak_mb}"
            )
        return errors

    # -- serialisation ---------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "scenario_name": self.scenario_name,
            "detected_leakage_bits": dict(self.detected_leakage_bits),
            "ground_truth_bits": dict(self.ground_truth_bits),
            "precision": round(self.precision, _METRIC_PRECISION),
            "recall": round(self.recall, _METRIC_PRECISION),
            "f1_score": round(self.f1_score, _METRIC_PRECISION),
            "false_positives": list(self.false_positives),
            "false_negatives": list(self.false_negatives),
            "analysis_time_ms": round(
                self.analysis_time_ms, _METRIC_PRECISION
            ),
            "memory_peak_mb": round(
                self.memory_peak_mb, _METRIC_PRECISION
            ),
            "bit_tolerance": self.bit_tolerance,
            "detection_threshold": self.detection_threshold,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BenchmarkResult:
        """Deserialise from a dictionary."""
        return cls(
            scenario_name=str(data.get("scenario_name", "")),
            detected_leakage_bits=dict(
                data.get("detected_leakage_bits", {})
            ),
            ground_truth_bits=dict(
                data.get("ground_truth_bits", {})
            ),
            precision=float(data.get("precision", 0.0)),
            recall=float(data.get("recall", 0.0)),
            f1_score=float(data.get("f1_score", 0.0)),
            false_positives=list(data.get("false_positives", [])),
            false_negatives=list(data.get("false_negatives", [])),
            analysis_time_ms=float(
                data.get("analysis_time_ms", 0.0)
            ),
            memory_peak_mb=float(data.get("memory_peak_mb", 0.0)),
            bit_tolerance=float(
                data.get("bit_tolerance", _DEFAULT_BIT_TOLERANCE)
            ),
            detection_threshold=float(
                data.get(
                    "detection_threshold", _DEFAULT_DETECTION_THRESHOLD
                )
            ),
            metadata=dict(data.get("metadata", {})),
        )

    # -- summary ---------------------------------------------------

    def summary_line(self) -> str:
        """One-line human-readable summary."""
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.scenario_name}: "
            f"P={self.precision:.3f} R={self.recall:.3f} "
            f"F1={self.f1_score:.3f} "
            f"FP={self.n_false_positives} FN={self.n_false_negatives} "
            f"({self.analysis_time_ms:.1f}ms)"
        )


# ===================================================================
#  BenchmarkReport
# ===================================================================


@dataclass
class BenchmarkReport:
    """Aggregate report across one or more benchmark scenarios.

    Attributes
    ----------
    results:
        Individual :class:`BenchmarkResult` entries.
    suite_name:
        Name of the benchmark suite that produced this report.
    timestamp:
        ISO-8601 timestamp when the report was created.
    total_time_ms:
        Wall-clock time for the entire suite, in milliseconds.
    metadata:
        Arbitrary metadata dict.
    """

    results: List[BenchmarkResult] = field(default_factory=list)
    suite_name: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    total_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- aggregate properties --------------------------------------

    @property
    def n_scenarios(self) -> int:
        """Total number of scenarios in this report."""
        return len(self.results)

    @property
    def n_passed(self) -> int:
        """Number of passing scenarios (P ≥ 0.9, R ≥ 0.9)."""
        return sum(1 for r in self.results if r.passed)

    @property
    def n_failed(self) -> int:
        """Number of failing scenarios."""
        return self.n_scenarios - self.n_passed

    @property
    def pass_rate(self) -> float:
        """Fraction of scenarios that pass."""
        if self.n_scenarios == 0:
            return 0.0
        return self.n_passed / self.n_scenarios

    @property
    def mean_precision(self) -> float:
        """Mean precision across all scenarios."""
        if not self.results:
            return 0.0
        return statistics.mean(r.precision for r in self.results)

    @property
    def mean_recall(self) -> float:
        """Mean recall across all scenarios."""
        if not self.results:
            return 0.0
        return statistics.mean(r.recall for r in self.results)

    @property
    def mean_f1(self) -> float:
        """Mean F1 score across all scenarios."""
        if not self.results:
            return 0.0
        return statistics.mean(r.f1_score for r in self.results)

    @property
    def median_time_ms(self) -> float:
        """Median per-scenario analysis time in milliseconds."""
        if not self.results:
            return 0.0
        return statistics.median(
            r.analysis_time_ms for r in self.results
        )

    @property
    def max_memory_mb(self) -> float:
        """Peak memory across all scenarios."""
        if not self.results:
            return 0.0
        return max(r.memory_peak_mb for r in self.results)

    @property
    def total_false_positives(self) -> int:
        """Total false positives across all scenarios."""
        return sum(r.n_false_positives for r in self.results)

    @property
    def total_false_negatives(self) -> int:
        """Total false negatives across all scenarios."""
        return sum(r.n_false_negatives for r in self.results)

    @property
    def all_passed(self) -> bool:
        """``True`` when every scenario passes."""
        return self.n_failed == 0 and self.n_scenarios > 0

    @property
    def summary(self) -> str:
        """Multi-line human-readable summary string."""
        lines: list[str] = [
            f"Benchmark Report: {self.suite_name}",
            f"  Timestamp : {self.timestamp}",
            f"  Scenarios : {self.n_passed}/{self.n_scenarios} passed "
            f"({self.pass_rate:.1%})",
            f"  Mean P/R/F1: {self.mean_precision:.3f} / "
            f"{self.mean_recall:.3f} / {self.mean_f1:.3f}",
            f"  Median time: {self.median_time_ms:.1f} ms",
            f"  Peak memory: {self.max_memory_mb:.1f} MB",
            f"  Total FP/FN: {self.total_false_positives} / "
            f"{self.total_false_negatives}",
            "",
        ]
        for r in self.results:
            lines.append(f"  {r.summary_line()}")
        return "\n".join(lines)

    # -- filtering -------------------------------------------------

    def failed_results(self) -> list[BenchmarkResult]:
        """Return only the failing scenario results."""
        return [r for r in self.results if not r.passed]

    def passed_results(self) -> list[BenchmarkResult]:
        """Return only the passing scenario results."""
        return [r for r in self.results if r.passed]

    def results_by_severity(
        self,
        severity: Severity,
    ) -> list[BenchmarkResult]:
        """Return results whose severity matches *severity*."""
        return [r for r in self.results if r.severity == severity]

    # -- validation ------------------------------------------------

    def validate(self) -> list[str]:
        """Return a list of validation error strings (empty ⇒ valid)."""
        errors: list[str] = []
        if not self.results:
            errors.append("report contains no results")
        for idx, r in enumerate(self.results):
            for err in r.validate():
                errors.append(f"results[{idx}]: {err}")
        if self.total_time_ms < 0.0:
            errors.append(
                f"total_time_ms must be ≥ 0, got {self.total_time_ms}"
            )
        return errors

    # -- serialisation ---------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "results": [r.to_dict() for r in self.results],
            "suite_name": self.suite_name,
            "timestamp": self.timestamp,
            "total_time_ms": round(
                self.total_time_ms, _METRIC_PRECISION
            ),
            "metadata": dict(self.metadata),
            "aggregate": {
                "n_scenarios": self.n_scenarios,
                "n_passed": self.n_passed,
                "n_failed": self.n_failed,
                "pass_rate": round(self.pass_rate, _METRIC_PRECISION),
                "mean_precision": round(
                    self.mean_precision, _METRIC_PRECISION
                ),
                "mean_recall": round(
                    self.mean_recall, _METRIC_PRECISION
                ),
                "mean_f1": round(self.mean_f1, _METRIC_PRECISION),
                "median_time_ms": round(
                    self.median_time_ms, _METRIC_PRECISION
                ),
                "max_memory_mb": round(
                    self.max_memory_mb, _METRIC_PRECISION
                ),
                "total_false_positives": self.total_false_positives,
                "total_false_negatives": self.total_false_negatives,
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BenchmarkReport:
        """Deserialise from a dictionary."""
        return cls(
            results=[
                BenchmarkResult.from_dict(r)
                for r in data.get("results", [])
            ],
            suite_name=str(data.get("suite_name", "")),
            timestamp=str(
                data.get(
                    "timestamp",
                    datetime.now(timezone.utc).isoformat(),
                )
            ),
            total_time_ms=float(data.get("total_time_ms", 0.0)),
            metadata=dict(data.get("metadata", {})),
        )

    # -- export ----------------------------------------------------

    def to_json(self, indent: int = 2) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> BenchmarkReport:
        """Deserialise from a JSON string."""
        return cls.from_dict(json.loads(text))


# ===================================================================
#  BenchmarkRunner
# ===================================================================


class BenchmarkRunner:
    """Execute a single :class:`BenchmarkScenario` and score the output.

    The runner invokes a caller-supplied *analysis function* on the
    scenario's dataset and compares detected leakage against the
    scenario's :class:`GroundTruth`.

    Parameters
    ----------
    bit_tolerance:
        Two bit-bounds match if ``|detected − truth| ≤ bit_tolerance``.
    detection_threshold:
        Minimum bits for a feature to count as "detected".
    """

    def __init__(
        self,
        bit_tolerance: float = _DEFAULT_BIT_TOLERANCE,
        detection_threshold: float = _DEFAULT_DETECTION_THRESHOLD,
    ) -> None:
        self._bit_tolerance = bit_tolerance
        self._detection_threshold = detection_threshold

    # -- properties ------------------------------------------------

    @property
    def bit_tolerance(self) -> float:
        """Tolerance used when comparing detected ↔ truth bits."""
        return self._bit_tolerance

    @property
    def detection_threshold(self) -> float:
        """Minimum bits for a feature to be considered detected."""
        return self._detection_threshold

    # -- public API ------------------------------------------------

    def run(
        self,
        scenario: BenchmarkScenario,
        analysis_fn: Callable[
            [list[list[float]], list[list[float]], list[float], list[float]],
            Dict[str, float],
        ],
    ) -> BenchmarkResult:
        """Run *analysis_fn* on the scenario and return a scored result.

        Parameters
        ----------
        scenario:
            A :class:`BenchmarkScenario` with dataset config and ground
            truth.
        analysis_fn:
            Callable ``(X_train, X_test, y_train, y_test) → {feat: bits}``.

        Returns
        -------
        BenchmarkResult
        """
        from taintflow.benchmarks.synthetic import SyntheticDataGenerator

        gen = SyntheticDataGenerator(scenario.dataset_config)
        X_tr, X_te, y_tr, y_te, gt = gen.generate_with_known_leakage()

        profiler = PerformanceProfiler()
        profiler.start()
        detected = analysis_fn(X_tr, X_te, y_tr, y_te)
        profiler.stop()

        result = self.compare(
            scenario_name=scenario.name,
            detected=detected,
            ground_truth_bits=gt.per_feature_bits,
        )
        result.analysis_time_ms = profiler.elapsed_ms
        result.memory_peak_mb = profiler.peak_memory_mb

        return result

    def compare(
        self,
        scenario_name: str,
        detected: Dict[str, float],
        ground_truth_bits: Dict[str, float],
    ) -> BenchmarkResult:
        """Compare detected leakage against ground truth.

        Computes precision, recall, F1, false positives and false
        negatives using the configured thresholds.

        Parameters
        ----------
        scenario_name:
            Name for reporting.
        detected:
            ``{feature_name: bits}`` from the analysis.
        ground_truth_bits:
            ``{feature_name: bits}`` from the ground truth.

        Returns
        -------
        BenchmarkResult
        """
        all_features = sorted(
            set(detected) | set(ground_truth_bits)
        )

        truly_leaking: set[str] = set()
        detected_leaking: set[str] = set()

        for f in all_features:
            gt_bits = ground_truth_bits.get(f, 0.0)
            det_bits = detected.get(f, 0.0)
            if gt_bits > 0.0:
                truly_leaking.add(f)
            if det_bits >= self._detection_threshold:
                detected_leaking.add(f)

        tp = truly_leaking & detected_leaking
        fp = detected_leaking - truly_leaking
        fn = truly_leaking - detected_leaking

        precision = len(tp) / len(detected_leaking) if detected_leaking else 1.0
        recall = len(tp) / len(truly_leaking) if truly_leaking else 1.0
        f1 = self._f1(precision, recall)

        return BenchmarkResult(
            scenario_name=scenario_name,
            detected_leakage_bits=dict(detected),
            ground_truth_bits=dict(ground_truth_bits),
            precision=precision,
            recall=recall,
            f1_score=f1,
            false_positives=sorted(fp),
            false_negatives=sorted(fn),
            bit_tolerance=self._bit_tolerance,
            detection_threshold=self._detection_threshold,
        )

    def compare_bit_accuracy(
        self,
        detected: Dict[str, float],
        ground_truth_bits: Dict[str, float],
    ) -> Dict[str, Dict[str, float]]:
        """Per-feature bit-level accuracy breakdown.

        Returns a dict ``{feature: {detected, truth, error, rel_error}}``.
        """
        all_features = sorted(
            set(detected) | set(ground_truth_bits)
        )
        result: dict[str, dict[str, float]] = {}
        for f in all_features:
            det = detected.get(f, 0.0)
            gt = ground_truth_bits.get(f, 0.0)
            err = abs(det - gt)
            rel = err / gt if gt > 0.0 else (0.0 if det == 0.0 else float("inf"))
            result[f] = {
                "detected": det,
                "truth": gt,
                "error": err,
                "relative_error": rel,
            }
        return result

    # -- internal --------------------------------------------------

    @staticmethod
    def _f1(precision: float, recall: float) -> float:
        """Compute F1 from precision and recall."""
        if precision + recall == 0.0:
            return 0.0
        return 2.0 * precision * recall / (precision + recall)


# ===================================================================
#  BenchmarkSuite
# ===================================================================


class BenchmarkSuite:
    """Named collection of :class:`BenchmarkScenario` instances.

    Register scenarios, then call :meth:`run_all` with an analysis
    function to produce a :class:`BenchmarkReport`.

    Parameters
    ----------
    name:
        Human-readable suite name.
    runner:
        Optional :class:`BenchmarkRunner`; a default is created if
        ``None``.
    """

    def __init__(
        self,
        name: str = "default",
        runner: BenchmarkRunner | None = None,
    ) -> None:
        self._name = name
        self._runner = runner or BenchmarkRunner()
        self._scenarios: list[BenchmarkScenario] = []
        self._tags_index: dict[str, list[int]] = {}

    # -- properties ------------------------------------------------

    @property
    def name(self) -> str:
        """Suite name."""
        return self._name

    @property
    def n_scenarios(self) -> int:
        """Number of registered scenarios."""
        return len(self._scenarios)

    @property
    def scenario_names(self) -> list[str]:
        """Sorted list of registered scenario names."""
        return sorted(s.name for s in self._scenarios)

    @property
    def all_tags(self) -> list[str]:
        """Sorted list of unique tags across all scenarios."""
        tags: set[str] = set()
        for s in self._scenarios:
            tags.update(s.tags)
        return sorted(tags)

    # -- registration ----------------------------------------------

    def register(self, scenario: BenchmarkScenario) -> None:
        """Add a scenario to this suite.

        Parameters
        ----------
        scenario:
            A fully-configured :class:`BenchmarkScenario`.

        Raises
        ------
        ValueError
            If a scenario with the same name is already registered.
        """
        for existing in self._scenarios:
            if existing.name == scenario.name:
                raise ValueError(
                    f"Scenario {scenario.name!r} already registered"
                )
        idx = len(self._scenarios)
        self._scenarios.append(scenario)
        for tag in scenario.tags:
            self._tags_index.setdefault(tag, []).append(idx)

    def register_many(
        self,
        scenarios: Sequence[BenchmarkScenario],
    ) -> None:
        """Register multiple scenarios at once."""
        for s in scenarios:
            self.register(s)

    def get_scenario(self, name: str) -> BenchmarkScenario | None:
        """Look up a scenario by name (``None`` if not found)."""
        for s in self._scenarios:
            if s.name == name:
                return s
        return None

    def scenarios_by_tag(self, tag: str) -> list[BenchmarkScenario]:
        """Return all scenarios carrying *tag*."""
        indices = self._tags_index.get(tag, [])
        return [self._scenarios[i] for i in indices]

    # -- execution -------------------------------------------------

    def run_all(
        self,
        analysis_fn: Callable[
            [list[list[float]], list[list[float]], list[float], list[float]],
            Dict[str, float],
        ],
    ) -> BenchmarkReport:
        """Run every registered scenario and return an aggregate report.

        Parameters
        ----------
        analysis_fn:
            Callable ``(X_train, X_test, y_train, y_test) → {feat: bits}``.

        Returns
        -------
        BenchmarkReport
        """
        suite_start = time.monotonic_ns()
        results: list[BenchmarkResult] = []
        for scenario in self._scenarios:
            result = self._runner.run(scenario, analysis_fn)
            results.append(result)
        suite_end = time.monotonic_ns()

        return BenchmarkReport(
            results=results,
            suite_name=self._name,
            total_time_ms=(suite_end - suite_start) / _NS_PER_MS,
        )

    def run_tagged(
        self,
        tag: str,
        analysis_fn: Callable[
            [list[list[float]], list[list[float]], list[float], list[float]],
            Dict[str, float],
        ],
    ) -> BenchmarkReport:
        """Run only scenarios matching *tag*."""
        suite_start = time.monotonic_ns()
        results: list[BenchmarkResult] = []
        for scenario in self.scenarios_by_tag(tag):
            result = self._runner.run(scenario, analysis_fn)
            results.append(result)
        suite_end = time.monotonic_ns()

        return BenchmarkReport(
            results=results,
            suite_name=f"{self._name}[tag={tag}]",
            total_time_ms=(suite_end - suite_start) / _NS_PER_MS,
        )

    def run_single(
        self,
        name: str,
        analysis_fn: Callable[
            [list[list[float]], list[list[float]], list[float], list[float]],
            Dict[str, float],
        ],
    ) -> BenchmarkResult:
        """Run a single scenario by name.

        Raises
        ------
        KeyError
            If no scenario with *name* is registered.
        """
        scenario = self.get_scenario(name)
        if scenario is None:
            raise KeyError(f"Unknown scenario {name!r}")
        return self._runner.run(scenario, analysis_fn)

    # -- serialisation ---------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the suite metadata (not scenario bodies)."""
        return {
            "name": self._name,
            "n_scenarios": self.n_scenarios,
            "scenario_names": self.scenario_names,
            "tags": self.all_tags,
        }


# ===================================================================
#  PerformanceProfiler
# ===================================================================


class PerformanceProfiler:
    """Lightweight wall-clock and memory profiler.

    Usage::

        profiler = PerformanceProfiler()
        profiler.start()
        # … do work …
        profiler.stop()
        print(profiler.elapsed_ms, profiler.peak_memory_mb)
    """

    def __init__(self) -> None:
        self._start_ns: int = 0
        self._stop_ns: int = 0
        self._peak_memory_bytes: int = 0
        self._running: bool = False
        self._snapshots: list[dict[str, Any]] = []

    # -- properties ------------------------------------------------

    @property
    def elapsed_ns(self) -> int:
        """Elapsed time in nanoseconds."""
        if self._running:
            return time.monotonic_ns() - self._start_ns
        return self._stop_ns - self._start_ns

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        return self.elapsed_ns / _NS_PER_MS

    @property
    def elapsed_s(self) -> float:
        """Elapsed time in seconds."""
        return self.elapsed_ns / 1_000_000_000.0

    @property
    def peak_memory_mb(self) -> float:
        """Peak memory in megabytes."""
        return self._peak_memory_bytes / (1024.0 * 1024.0)

    @property
    def peak_memory_bytes(self) -> int:
        """Peak memory in bytes."""
        return self._peak_memory_bytes

    @property
    def is_running(self) -> bool:
        """``True`` when the profiler is active."""
        return self._running

    @property
    def n_snapshots(self) -> int:
        """Number of captured snapshots."""
        return len(self._snapshots)

    # -- control ---------------------------------------------------

    def start(self) -> None:
        """Begin profiling."""
        self._running = True
        self._peak_memory_bytes = self._current_memory_bytes()
        self._start_ns = time.monotonic_ns()
        self._snapshots.clear()

    def stop(self) -> None:
        """Stop profiling and record peak memory."""
        self._stop_ns = time.monotonic_ns()
        self._running = False
        mem = self._current_memory_bytes()
        if mem > self._peak_memory_bytes:
            self._peak_memory_bytes = mem

    def snapshot(self, label: str = "") -> dict[str, Any]:
        """Capture a named intermediate snapshot.

        Returns
        -------
        dict
            ``{label, elapsed_ms, memory_mb, timestamp}``.
        """
        snap: dict[str, Any] = {
            "label": label,
            "elapsed_ms": self.elapsed_ms,
            "memory_mb": self._current_memory_bytes() / (1024 * 1024),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._snapshots.append(snap)
        mem = self._current_memory_bytes()
        if mem > self._peak_memory_bytes:
            self._peak_memory_bytes = mem
        return snap

    def reset(self) -> None:
        """Reset all counters."""
        self._start_ns = 0
        self._stop_ns = 0
        self._peak_memory_bytes = 0
        self._running = False
        self._snapshots.clear()

    # -- export ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise profiling results."""
        return {
            "elapsed_ms": round(self.elapsed_ms, _METRIC_PRECISION),
            "elapsed_s": round(self.elapsed_s, _METRIC_PRECISION),
            "peak_memory_mb": round(
                self.peak_memory_mb, _METRIC_PRECISION
            ),
            "peak_memory_bytes": self._peak_memory_bytes,
            "n_snapshots": self.n_snapshots,
            "snapshots": list(self._snapshots),
        }

    # -- internal --------------------------------------------------

    @staticmethod
    def _current_memory_bytes() -> int:
        """Best-effort current RSS via :mod:`resource` (Unix) or 0."""
        try:
            import resource  # noqa: PLC0415

            usage = resource.getrusage(resource.RUSAGE_SELF)
            # macOS returns bytes; Linux returns kilobytes.
            import sys as _sys

            if _sys.platform == "darwin":
                return usage.ru_maxrss
            return usage.ru_maxrss * 1024
        except Exception:  # noqa: BLE001
            return 0


# ===================================================================
#  RegressionDetector
# ===================================================================


class RegressionDetector:
    """Compare a new :class:`BenchmarkReport` against a saved baseline.

    A *regression* is flagged when any of the following occur:

    * A previously-passing scenario now fails.
    * Mean precision or recall drops by more than a configurable delta.
    * Any scenario's analysis time increases beyond a threshold.

    Parameters
    ----------
    precision_delta:
        Maximum allowed drop in mean precision before flagging.
    recall_delta:
        Maximum allowed drop in mean recall before flagging.
    time_factor:
        A scenario's time may grow by at most this multiplicative
        factor before being flagged (e.g. ``2.0`` allows doubling).
    """

    def __init__(
        self,
        precision_delta: float = 0.05,
        recall_delta: float = 0.05,
        time_factor: float = 2.0,
    ) -> None:
        self._precision_delta = precision_delta
        self._recall_delta = recall_delta
        self._time_factor = time_factor

    # -- properties ------------------------------------------------

    @property
    def precision_delta(self) -> float:
        """Allowed precision drop."""
        return self._precision_delta

    @property
    def recall_delta(self) -> float:
        """Allowed recall drop."""
        return self._recall_delta

    @property
    def time_factor(self) -> float:
        """Allowed time multiplicative increase."""
        return self._time_factor

    # -- public API ------------------------------------------------

    def detect(
        self,
        current: BenchmarkReport,
        baseline: BenchmarkReport,
    ) -> list[str]:
        """Return a list of human-readable regression descriptions.

        An empty list means no regressions were detected.

        Parameters
        ----------
        current:
            The newly-produced report.
        baseline:
            The saved baseline to compare against.

        Returns
        -------
        list[str]
        """
        regressions: list[str] = []

        # Build lookup for baseline results by scenario name.
        baseline_by_name: dict[str, BenchmarkResult] = {
            r.scenario_name: r for r in baseline.results
        }

        # 1. Per-scenario pass/fail regressions.
        for r in current.results:
            base = baseline_by_name.get(r.scenario_name)
            if base is None:
                continue  # new scenario – no baseline to regress against
            if base.passed and not r.passed:
                regressions.append(
                    f"REGRESSION: {r.scenario_name!r} was PASS, now FAIL "
                    f"(P={r.precision:.3f} R={r.recall:.3f})"
                )

        # 2. Per-scenario precision / recall drops.
        for r in current.results:
            base = baseline_by_name.get(r.scenario_name)
            if base is None:
                continue
            p_drop = base.precision - r.precision
            r_drop = base.recall - r.recall
            if p_drop > self._precision_delta:
                regressions.append(
                    f"PRECISION DROP: {r.scenario_name!r} "
                    f"precision {base.precision:.3f} → {r.precision:.3f} "
                    f"(Δ = {p_drop:+.3f})"
                )
            if r_drop > self._recall_delta:
                regressions.append(
                    f"RECALL DROP: {r.scenario_name!r} "
                    f"recall {base.recall:.3f} → {r.recall:.3f} "
                    f"(Δ = {r_drop:+.3f})"
                )

        # 3. Per-scenario timing regressions.
        for r in current.results:
            base = baseline_by_name.get(r.scenario_name)
            if base is None or base.analysis_time_ms <= 0:
                continue
            ratio = r.analysis_time_ms / base.analysis_time_ms
            if ratio > self._time_factor:
                regressions.append(
                    f"SLOWDOWN: {r.scenario_name!r} "
                    f"{base.analysis_time_ms:.1f}ms → "
                    f"{r.analysis_time_ms:.1f}ms "
                    f"({ratio:.1f}× baseline)"
                )

        # 4. Aggregate metric drops.
        p_agg_drop = baseline.mean_precision - current.mean_precision
        r_agg_drop = baseline.mean_recall - current.mean_recall
        if p_agg_drop > self._precision_delta:
            regressions.append(
                f"AGGREGATE PRECISION DROP: "
                f"{baseline.mean_precision:.3f} → "
                f"{current.mean_precision:.3f} "
                f"(Δ = {p_agg_drop:+.3f})"
            )
        if r_agg_drop > self._recall_delta:
            regressions.append(
                f"AGGREGATE RECALL DROP: "
                f"{baseline.mean_recall:.3f} → "
                f"{current.mean_recall:.3f} "
                f"(Δ = {r_agg_drop:+.3f})"
            )

        # 5. New false negatives that didn't exist before.
        for r in current.results:
            base = baseline_by_name.get(r.scenario_name)
            if base is None:
                continue
            new_fn = set(r.false_negatives) - set(base.false_negatives)
            if new_fn:
                regressions.append(
                    f"NEW FALSE NEGATIVES: {r.scenario_name!r} "
                    f"features {sorted(new_fn)} now missed"
                )

        return regressions

    def has_regressions(
        self,
        current: BenchmarkReport,
        baseline: BenchmarkReport,
    ) -> bool:
        """``True`` when at least one regression is detected."""
        return len(self.detect(current, baseline)) > 0

    def summary(
        self,
        current: BenchmarkReport,
        baseline: BenchmarkReport,
    ) -> str:
        """Human-readable regression summary."""
        issues = self.detect(current, baseline)
        if not issues:
            return "No regressions detected."
        header = f"{len(issues)} regression(s) detected:\n"
        return header + "\n".join(f"  • {i}" for i in issues)

    # -- serialisation ---------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise detector configuration."""
        return {
            "precision_delta": self._precision_delta,
            "recall_delta": self._recall_delta,
            "time_factor": self._time_factor,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RegressionDetector:
        """Deserialise from a dictionary."""
        return cls(
            precision_delta=float(data.get("precision_delta", 0.05)),
            recall_delta=float(data.get("recall_delta", 0.05)),
            time_factor=float(data.get("time_factor", 2.0)),
        )
