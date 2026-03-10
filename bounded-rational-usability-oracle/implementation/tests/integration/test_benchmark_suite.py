"""Integration tests for the benchmark suite.

These tests exercise the ``BenchmarkSuite``, ``MutationGenerator``, and
``BenchmarkMetrics`` (aliased as ``MetricComputer``) classes with small
synthetic inputs.  They verify that the suite can be constructed, run,
and produce valid metric reports.
"""

from __future__ import annotations

import math
from typing import Any, List

import numpy as np
import pytest

from usability_oracle.benchmarks.suite import (
    BenchmarkSuite,
    BenchmarkCase,
    BenchmarkResult,
    BenchmarkReport,
)
from usability_oracle.benchmarks.mutations import MutationGenerator
from usability_oracle.benchmarks.metrics import BenchmarkMetrics
from usability_oracle.accessibility.models import (
    AccessibilityNode,
    AccessibilityTree,
    BoundingBox,
    AccessibilityState,
)
from usability_oracle.accessibility.html_parser import HTMLAccessibilityParser
from usability_oracle.accessibility.normalizer import AccessibilityNormalizer
from usability_oracle.core.enums import RegressionVerdict
from usability_oracle.taskspec.models import TaskStep, TaskFlow, TaskSpec

from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
SAMPLE_HTML_DIR = FIXTURES_DIR / "sample_html"


def _load_html(name: str) -> str:
    return (SAMPLE_HTML_DIR / f"{name}.html").read_text()


def _parse(html: str) -> AccessibilityTree:
    tree = HTMLAccessibilityParser().parse(html)
    return AccessibilityNormalizer().normalize(tree)


def _make_task() -> TaskSpec:
    steps = [
        TaskStep(step_id="s1", action_type="click", target_role="button",
                 target_name="Submit", description="Submit"),
    ]
    flow = TaskFlow(flow_id="f1", name="Simple", steps=steps)
    return TaskSpec(spec_id="t1", name="Simple Task", flows=[flow])


def _make_small_tree() -> AccessibilityTree:
    """Build a minimal accessibility tree programmatically."""
    child1 = AccessibilityNode(
        id="n1", role="button", name="Submit",
        bounding_box=BoundingBox(x=10.0, y=10.0, width=80.0, height=30.0),
        state=AccessibilityState(),
        children=[], depth=1,
    )
    child2 = AccessibilityNode(
        id="n2", role="textfield", name="Username",
        bounding_box=BoundingBox(x=10.0, y=50.0, width=200.0, height=30.0),
        state=AccessibilityState(),
        children=[], depth=1,
    )
    child3 = AccessibilityNode(
        id="n3", role="textfield", name="Password",
        bounding_box=BoundingBox(x=10.0, y=90.0, width=200.0, height=30.0),
        state=AccessibilityState(),
        children=[], depth=1,
    )
    root = AccessibilityNode(
        id="root", role="form", name="Login Form",
        bounding_box=BoundingBox(x=0.0, y=0.0, width=300.0, height=200.0),
        state=AccessibilityState(),
        children=[child1, child2, child3], depth=0,
    )
    return AccessibilityTree(root=root)


def _make_dummy_results(n: int = 10) -> List[BenchmarkResult]:
    """Create synthetic benchmark results for metric testing."""
    results = []
    for i in range(n):
        # Alternate between correct and incorrect predictions
        expected = RegressionVerdict.REGRESSION if i % 3 == 0 else RegressionVerdict.NEUTRAL
        actual = expected if i % 4 != 0 else RegressionVerdict.IMPROVEMENT
        results.append(BenchmarkResult(
            case_name=f"case_{i}",
            actual_verdict=actual,
            expected_verdict=expected,
            timing=0.1 * (i + 1),
            correct=(actual == expected),
            error=None,
            metadata={},
        ))
    return results


def _make_benchmark_cases() -> List[BenchmarkCase]:
    """Create small benchmark cases for suite testing."""
    task = _make_task()
    tree = _make_small_tree()
    cases = [
        BenchmarkCase(
            name="identical",
            source_a=tree,
            source_b=tree,
            task_spec=task,
            expected_verdict=RegressionVerdict.NEUTRAL,
            category="identity",
            metadata={},
        ),
    ]
    return cases


# ===================================================================
# Tests – BenchmarkSuite construction
# ===================================================================


class TestBenchmarkSuiteConstruction:
    """Verify BenchmarkSuite can be instantiated and configured."""

    def test_default_construction(self) -> None:
        """BenchmarkSuite should construct with default arguments."""
        suite = BenchmarkSuite()
        assert suite is not None

    def test_construction_with_positive_class(self) -> None:
        """BenchmarkSuite accepts a ``positive_class`` argument."""
        suite = BenchmarkSuite(
            positive_class=RegressionVerdict.REGRESSION,
        )
        assert suite is not None

    def test_construction_with_verbose(self) -> None:
        """BenchmarkSuite accepts a ``verbose`` flag."""
        suite = BenchmarkSuite(verbose=True)
        assert suite is not None

    def test_construction_with_pipeline_fn(self) -> None:
        """BenchmarkSuite accepts a custom pipeline function."""
        def dummy_fn(source_a, source_b, task_spec, **kw):
            return RegressionVerdict.NEUTRAL
        suite = BenchmarkSuite(pipeline_fn=dummy_fn)
        assert suite is not None


# ===================================================================
# Tests – BenchmarkSuite execution
# ===================================================================


class TestBenchmarkSuiteExecution:
    """Run the benchmark suite with synthetic data."""

    def test_run_with_cases(self) -> None:
        """Running the suite with explicit cases should produce a report."""
        def dummy_fn(source_a, source_b, task_spec, **kw):
            return RegressionVerdict.NEUTRAL
        suite = BenchmarkSuite(pipeline_fn=dummy_fn)
        cases = _make_benchmark_cases()
        report = suite.run(cases=cases)
        assert isinstance(report, BenchmarkReport)
        assert len(report.results) == len(cases)

    def test_report_has_accuracy(self) -> None:
        """The benchmark report should include an accuracy field."""
        def dummy_fn(source_a, source_b, task_spec, **kw):
            return RegressionVerdict.NEUTRAL
        suite = BenchmarkSuite(pipeline_fn=dummy_fn)
        cases = _make_benchmark_cases()
        report = suite.run(cases=cases)
        assert 0.0 <= report.accuracy <= 1.0

    def test_report_timing_stats(self) -> None:
        """The report should include timing statistics."""
        def dummy_fn(source_a, source_b, task_spec, **kw):
            return RegressionVerdict.NEUTRAL
        suite = BenchmarkSuite(pipeline_fn=dummy_fn)
        cases = _make_benchmark_cases()
        report = suite.run(cases=cases)
        assert isinstance(report.timing_stats, dict)

    def test_report_summary_string(self) -> None:
        """``summary()`` should return a non-empty string."""
        def dummy_fn(source_a, source_b, task_spec, **kw):
            return RegressionVerdict.NEUTRAL
        suite = BenchmarkSuite(pipeline_fn=dummy_fn)
        cases = _make_benchmark_cases()
        report = suite.run(cases=cases)
        s = report.summary()
        assert isinstance(s, str)
        assert len(s) > 0


# ===================================================================
# Tests – MutationGenerator
# ===================================================================


class TestMutationGenerator:
    """Verify MutationGenerator creates valid mutations."""

    def test_construction(self) -> None:
        """MutationGenerator should construct with a seed."""
        gen = MutationGenerator(seed=42)
        assert gen is not None

    def test_apply_motor_difficulty(self) -> None:
        """``apply_motor_difficulty`` should shrink bounding boxes."""
        tree = _make_small_tree()
        gen = MutationGenerator(seed=42)
        mutated = gen.apply_motor_difficulty(tree, severity=0.5)
        assert mutated.size() == tree.size()

    def test_apply_choice_paralysis(self) -> None:
        """``apply_choice_paralysis`` should add extra nodes."""
        tree = _make_small_tree()
        gen = MutationGenerator(seed=42)
        mutated = gen.apply_choice_paralysis(tree, severity=0.5)
        assert mutated.size() >= tree.size()

    def test_apply_perceptual_overload(self) -> None:
        """``apply_perceptual_overload`` should add visual clutter."""
        tree = _make_small_tree()
        gen = MutationGenerator(seed=42)
        mutated = gen.apply_perceptual_overload(tree, severity=0.5)
        assert mutated.size() >= tree.size()

    def test_apply_memory_decay(self) -> None:
        """``apply_memory_decay`` should increase tree depth."""
        tree = _make_small_tree()
        gen = MutationGenerator(seed=42)
        mutated = gen.apply_memory_decay(tree, severity=0.5)
        assert mutated.size() >= 1

    def test_apply_interference(self) -> None:
        """``apply_interference`` should shuffle or modify nodes."""
        tree = _make_small_tree()
        gen = MutationGenerator(seed=42)
        mutated = gen.apply_interference(tree, severity=0.5)
        assert mutated.size() >= 1

    def test_apply_random_mutation(self) -> None:
        """``apply_random_mutation`` returns a tree and mutation label."""
        tree = _make_small_tree()
        gen = MutationGenerator(seed=42)
        mutated, label = gen.apply_random_mutation(tree)
        assert isinstance(mutated, AccessibilityTree)
        assert isinstance(label, str)
        assert len(label) > 0

    def test_mutated_tree_validates(self) -> None:
        """A mutated tree should pass structural validation."""
        tree = _make_small_tree()
        gen = MutationGenerator(seed=42)
        mutated = gen.apply_motor_difficulty(tree, severity=0.3)
        errors = mutated.validate()
        assert isinstance(errors, list)


# ===================================================================
# Tests – BenchmarkMetrics (MetricComputer)
# ===================================================================


class TestBenchmarkMetrics:
    """Verify metric computation on synthetic results."""

    def test_accuracy(self) -> None:
        """Accuracy should be between 0 and 1."""
        results = _make_dummy_results(10)
        acc = BenchmarkMetrics.accuracy(results)
        assert 0.0 <= acc <= 1.0

    def test_precision(self) -> None:
        """Precision should be between 0 and 1."""
        results = _make_dummy_results(10)
        prec = BenchmarkMetrics.precision(results)
        assert 0.0 <= prec <= 1.0

    def test_recall(self) -> None:
        """Recall should be between 0 and 1."""
        results = _make_dummy_results(10)
        rec = BenchmarkMetrics.recall(results)
        assert 0.0 <= rec <= 1.0

    def test_f1_score(self) -> None:
        """F1 score should be between 0 and 1."""
        results = _make_dummy_results(10)
        f1 = BenchmarkMetrics.f1_score(results)
        assert 0.0 <= f1 <= 1.0

    def test_confusion_matrix_shape(self) -> None:
        """Confusion matrix should be a square numpy array."""
        results = _make_dummy_results(10)
        cm = BenchmarkMetrics.confusion_matrix(results)
        assert isinstance(cm, np.ndarray)
        assert cm.shape[0] == cm.shape[1]

    def test_timing_statistics(self) -> None:
        """Timing statistics should include mean and std."""
        results = _make_dummy_results(10)
        stats = BenchmarkMetrics.timing_statistics(results)
        assert isinstance(stats, dict)

    def test_per_category_accuracy(self) -> None:
        """Per-category accuracy should return a dict."""
        results = _make_dummy_results(10)
        pca = BenchmarkMetrics.per_category_accuracy(results)
        assert isinstance(pca, dict)

    def test_scalability_curve(self) -> None:
        """Scalability curve should return a list of (size, time) tuples."""
        results = _make_dummy_results(10)
        curve = BenchmarkMetrics.scalability_curve(results)
        assert isinstance(curve, list)
