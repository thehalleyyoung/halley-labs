"""
Unit tests for ``arc.quality`` — quality monitoring, distribution
analysis, constraint evaluation, and data profiling.

Uses numpy arrays as test data. Imports are wrapped in try/except.
"""

from __future__ import annotations

from typing import Any

import pytest

try:
    import numpy as np
except ImportError:
    pytest.skip("numpy not installed", allow_module_level=True)

from arc.quality.monitor import QualityMonitor
from arc.quality.distribution import DistributionAnalyzer
from arc.quality.constraints import ConstraintEngine
from arc.quality.profiler import DataProfiler
from arc.types.base import (
    CheckResult,
    ColumnProfile,
    ConstraintResult,
    ConstraintSuggestion,
    TableProfile,
    Violation,
)


# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def monitor() -> QualityMonitor:
    return QualityMonitor()


@pytest.fixture
def analyzer() -> DistributionAnalyzer:
    return DistributionAnalyzer()


@pytest.fixture
def constraint_engine() -> ConstraintEngine:
    return ConstraintEngine()


@pytest.fixture
def profiler() -> DataProfiler:
    return DataProfiler()


# =====================================================================
# Helper data builders
# =====================================================================


def _dict_data(**columns: Any) -> dict[str, np.ndarray]:
    """Build a dict-of-arrays dataset from keyword args."""
    return {k: np.asarray(v) for k, v in columns.items()}


# =====================================================================
# QualityMonitor — null_rate_check
# =====================================================================


class TestQualityMonitorNullRate:
    """null_rate_check: passes when no nulls, fails when above threshold."""

    def test_no_nulls_passes(self, monitor: QualityMonitor):
        data = _dict_data(col=np.array([1, 2, 3, 4, 5]))
        result = monitor.null_rate_check(data, "col", threshold=0.0)
        assert result.passed is True
        assert result.metric_value == 0.0

    def test_all_valid_threshold_nonzero(self, monitor: QualityMonitor):
        data = _dict_data(col=np.array([10.0, 20.0, 30.0]))
        result = monitor.null_rate_check(data, "col", threshold=0.5)
        assert result.passed is True

    def test_nulls_exceeding_threshold_fails(self, monitor: QualityMonitor):
        data = _dict_data(
            col=np.array([1.0, float("nan"), float("nan"), 4.0])
        )
        result = monitor.null_rate_check(data, "col", threshold=0.1)
        assert result.passed is False
        assert result.metric_value > 0.1

    def test_nulls_within_threshold_passes(self, monitor: QualityMonitor):
        data = _dict_data(
            col=np.array([1.0, float("nan"), 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        )
        # 1 out of 10 = 0.1
        result = monitor.null_rate_check(data, "col", threshold=0.15)
        assert result.passed is True

    def test_empty_dataset_passes(self, monitor: QualityMonitor):
        data = _dict_data(col=np.array([]))
        result = monitor.null_rate_check(data, "col", threshold=0.0)
        assert result.passed is True


# =====================================================================
# QualityMonitor — range_check
# =====================================================================


class TestQualityMonitorRangeCheck:
    """range_check: passes when all values in bounds."""

    def test_within_bounds_passes(self, monitor: QualityMonitor):
        data = _dict_data(val=np.array([5, 10, 15, 20]))
        result = monitor.range_check(data, "val", min_val=0, max_val=25)
        assert result.passed is True

    def test_below_min_fails(self, monitor: QualityMonitor):
        data = _dict_data(val=np.array([1, 2, 3, -5]))
        result = monitor.range_check(data, "val", min_val=0, max_val=100)
        assert result.passed is False

    def test_above_max_fails(self, monitor: QualityMonitor):
        data = _dict_data(val=np.array([1, 2, 3, 200]))
        result = monitor.range_check(data, "val", min_val=0, max_val=100)
        assert result.passed is False

    def test_unbounded_min(self, monitor: QualityMonitor):
        data = _dict_data(val=np.array([-1000, 0, 50]))
        result = monitor.range_check(data, "val", min_val=None, max_val=100)
        assert result.passed is True

    def test_unbounded_max(self, monitor: QualityMonitor):
        data = _dict_data(val=np.array([1, 100, 9999]))
        result = monitor.range_check(data, "val", min_val=0, max_val=None)
        assert result.passed is True


# =====================================================================
# QualityMonitor — uniqueness_check
# =====================================================================


class TestQualityMonitorUniqueness:
    """uniqueness_check: passes when values are unique."""

    def test_unique_values_pass(self, monitor: QualityMonitor):
        data = _dict_data(id=np.array([1, 2, 3, 4, 5]))
        result = monitor.uniqueness_check(data, ["id"])
        assert result.passed is True
        assert result.metric_value == 0.0

    def test_duplicate_values_fail(self, monitor: QualityMonitor):
        data = _dict_data(id=np.array([1, 2, 2, 3, 3]))
        result = monitor.uniqueness_check(data, ["id"])
        assert result.passed is False
        assert result.metric_value > 0

    def test_multi_column_uniqueness(self, monitor: QualityMonitor):
        data = _dict_data(
            a=np.array([1, 1, 2]),
            b=np.array([10, 20, 10]),
        )
        result = monitor.uniqueness_check(data, ["a", "b"])
        assert result.passed is True

    def test_multi_column_duplicates(self, monitor: QualityMonitor):
        data = _dict_data(
            a=np.array([1, 1, 2]),
            b=np.array([10, 10, 20]),
        )
        result = monitor.uniqueness_check(data, ["a", "b"])
        assert result.passed is False

    def test_empty_columns_list(self, monitor: QualityMonitor):
        data = _dict_data(x=np.array([1, 2, 3]))
        result = monitor.uniqueness_check(data, [])
        assert result.passed is True


# =====================================================================
# QualityMonitor — check_constraints (batch)
# =====================================================================


class TestQualityMonitorBatch:
    """check_constraints: batch evaluation of multiple constraints."""

    def test_all_pass_returns_empty(self, monitor: QualityMonitor):
        data = _dict_data(
            id=np.array([1, 2, 3]),
            val=np.array([10.0, 20.0, 30.0]),
        )
        constraints = [
            {"name": "id_not_null", "type": "not_null", "column": "id"},
            {"name": "val_range", "type": "range", "column": "val",
             "min": 0, "max": 100},
            {"name": "id_unique", "type": "unique", "columns": ["id"]},
        ]
        violations = monitor.check_constraints(data, constraints)
        assert violations == []

    def test_mixed_pass_fail(self, monitor: QualityMonitor):
        data = _dict_data(
            id=np.array([1, 2, 2]),
            val=np.array([10.0, float("nan"), 30.0]),
        )
        constraints = [
            {"name": "id_unique", "type": "unique", "columns": ["id"]},
            {"name": "val_not_null", "type": "not_null", "column": "val",
             "threshold": 0.0},
        ]
        violations = monitor.check_constraints(data, constraints)
        assert len(violations) >= 2
        names = [v.constraint_name for v in violations]
        assert "id_unique" in names
        assert "val_not_null" in names

    def test_empty_constraints_returns_empty(self, monitor: QualityMonitor):
        data = _dict_data(x=np.array([1]))
        violations = monitor.check_constraints(data, [])
        assert violations == []

    def test_custom_constraint(self, monitor: QualityMonitor):
        data = _dict_data(val=np.array([1, 2, 3, 4, 5]))
        constraints = [
            {
                "name": "mean_check",
                "type": "custom",
                "predicate": "mean(val) > 0",
            }
        ]
        violations = monitor.check_constraints(data, constraints)
        assert violations == []


# =====================================================================
# DistributionAnalyzer — KS test
# =====================================================================


class TestDistributionAnalyzerKS:
    """ks_test: detects identical and different distributions."""

    def test_same_distribution_high_pvalue(self):
        np.random.seed(42)
        s1 = np.random.normal(0, 1, 500)
        s2 = np.random.normal(0, 1, 500)
        result = DistributionAnalyzer.ks_test(s1, s2)
        assert result.p_value > 0.05
        assert result.sample_size_1 == 500
        assert result.sample_size_2 == 500

    def test_different_distributions_low_pvalue(self):
        np.random.seed(42)
        s1 = np.random.normal(0, 1, 500)
        s2 = np.random.normal(5, 1, 500)
        result = DistributionAnalyzer.ks_test(s1, s2)
        assert result.p_value < 0.05
        assert result.statistic > 0.1

    def test_identical_arrays(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = DistributionAnalyzer.ks_test(arr, arr)
        assert result.statistic == 0.0
        assert result.p_value == 1.0

    def test_empty_samples(self):
        result = DistributionAnalyzer.ks_test(np.array([]), np.array([1.0]))
        assert result.p_value == 1.0


# =====================================================================
# DistributionAnalyzer — PSI score
# =====================================================================


class TestDistributionAnalyzerPSI:
    """psi_score: measures distribution shift magnitude."""

    def test_identical_distribution_near_zero(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        psi = DistributionAnalyzer.psi_score(data, data)
        assert psi < 0.01

    def test_shifted_distribution_positive(self):
        np.random.seed(42)
        expected = np.random.normal(0, 1, 1000)
        actual = np.random.normal(3, 1, 1000)
        psi = DistributionAnalyzer.psi_score(expected, actual)
        assert psi > 0.1

    def test_empty_returns_zero(self):
        psi = DistributionAnalyzer.psi_score(np.array([]), np.array([1.0]))
        assert psi == 0.0

    def test_same_data_copies(self):
        np.random.seed(42)
        data = np.random.uniform(0, 10, 500)
        psi = DistributionAnalyzer.psi_score(data, data.copy())
        assert psi < 0.01


# =====================================================================
# DistributionAnalyzer — detect_shift
# =====================================================================


class TestDistributionAnalyzerShift:
    """detect_shift: multi-column shift detection."""

    def test_no_shift(self, analyzer: DistributionAnalyzer):
        np.random.seed(42)
        old = _dict_data(x=np.random.normal(0, 1, 500))
        new = _dict_data(x=np.random.normal(0, 1, 500))
        results = analyzer.detect_shift(old, new, ["x"], threshold=0.25)
        assert len(results) == 1
        assert results[0].shifted is False

    def test_clear_shift(self, analyzer: DistributionAnalyzer):
        np.random.seed(42)
        old = _dict_data(x=np.random.normal(0, 1, 500))
        new = _dict_data(x=np.random.normal(10, 1, 500))
        results = analyzer.detect_shift(old, new, ["x"], threshold=0.1)
        assert len(results) == 1
        assert results[0].shifted is True

    def test_multiple_columns(self, analyzer: DistributionAnalyzer):
        np.random.seed(42)
        old = _dict_data(
            a=np.random.normal(0, 1, 300),
            b=np.random.normal(0, 1, 300),
        )
        new = _dict_data(
            a=np.random.normal(0, 1, 300),
            b=np.random.normal(5, 1, 300),
        )
        results = analyzer.detect_shift(old, new, ["a", "b"], threshold=0.1)
        assert len(results) == 2
        col_shifted = {r.column_name: r.shifted for r in results}
        assert col_shifted["b"] is True


# =====================================================================
# DistributionAnalyzer — chi_squared and JSD
# =====================================================================


class TestDistributionAnalyzerStatTests:
    """chi_squared_test and jensen_shannon_divergence."""

    def test_chi_squared_equal_frequencies(self):
        obs = np.array([50, 50, 50, 50])
        exp = np.array([50, 50, 50, 50])
        result = DistributionAnalyzer.chi_squared_test(obs, exp)
        assert result.statistic < 0.01
        assert result.p_value > 0.9

    def test_chi_squared_different_frequencies(self):
        obs = np.array([100, 10, 10, 10])
        exp = np.array([25, 25, 25, 25])
        result = DistributionAnalyzer.chi_squared_test(obs, exp)
        assert result.statistic > 10.0
        assert result.p_value < 0.05

    def test_jsd_identical_zero(self):
        p = np.array([0.25, 0.25, 0.25, 0.25])
        jsd = DistributionAnalyzer.jensen_shannon_divergence(p, p)
        assert jsd < 1e-10

    def test_jsd_different_positive(self):
        p = np.array([0.9, 0.1])
        q = np.array([0.1, 0.9])
        jsd = DistributionAnalyzer.jensen_shannon_divergence(p, q)
        assert jsd > 0.1

    def test_jsd_empty_zero(self):
        jsd = DistributionAnalyzer.jensen_shannon_divergence(
            np.array([]), np.array([])
        )
        assert jsd == 0.0


# =====================================================================
# ConstraintEngine — evaluate
# =====================================================================


class TestConstraintEngineEvaluate:
    """Single constraint evaluation."""

    def test_not_null_passes(self, constraint_engine: ConstraintEngine):
        data = _dict_data(name=np.array(["alice", "bob", "charlie"]))
        result = constraint_engine.evaluate(
            data, {"type": "not_null", "column": "name"}
        )
        assert isinstance(result, ConstraintResult)
        assert result.passed is True

    def test_not_null_fails(self, constraint_engine: ConstraintEngine):
        data = _dict_data(
            name=np.array(["alice", None, "charlie"], dtype=object)
        )
        result = constraint_engine.evaluate(
            data, {"type": "not_null", "column": "name"}
        )
        assert result.passed is False

    def test_unique_passes(self, constraint_engine: ConstraintEngine):
        data = _dict_data(id=np.array([1, 2, 3, 4, 5]))
        result = constraint_engine.evaluate(
            data, {"type": "unique", "column": "id"}
        )
        assert result.passed is True

    def test_unique_fails(self, constraint_engine: ConstraintEngine):
        data = _dict_data(id=np.array([1, 2, 2, 3]))
        result = constraint_engine.evaluate(
            data, {"type": "unique", "column": "id"}
        )
        assert result.passed is False

    def test_range_passes(self, constraint_engine: ConstraintEngine):
        data = _dict_data(score=np.array([10.0, 50.0, 90.0]))
        result = constraint_engine.evaluate(
            data,
            {"type": "range", "column": "score", "min": 0, "max": 100},
        )
        assert result.passed is True

    def test_range_fails(self, constraint_engine: ConstraintEngine):
        data = _dict_data(score=np.array([10.0, 50.0, 150.0]))
        result = constraint_engine.evaluate(
            data,
            {"type": "range", "column": "score", "min": 0, "max": 100},
        )
        assert result.passed is False

    def test_unknown_type_passes(self, constraint_engine: ConstraintEngine):
        data = _dict_data(x=np.array([1]))
        result = constraint_engine.evaluate(
            data, {"type": "imaginary_type"}
        )
        assert result.passed is True


# =====================================================================
# ConstraintEngine — evaluate_batch
# =====================================================================


class TestConstraintEngineBatch:
    """Batch constraint evaluation."""

    def test_evaluate_batch_returns_list(
        self, constraint_engine: ConstraintEngine
    ):
        data = _dict_data(
            id=np.array([1, 2, 3]),
            val=np.array([10.0, 20.0, 30.0]),
        )
        constraints = [
            {"type": "not_null", "column": "id"},
            {"type": "range", "column": "val", "min": 0, "max": 50},
        ]
        results = constraint_engine.evaluate_batch(data, constraints)
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, ConstraintResult) for r in results)
        assert all(r.passed for r in results)

    def test_evaluate_batch_mixed(self, constraint_engine: ConstraintEngine):
        data = _dict_data(
            id=np.array([1, 1, 2]),
            val=np.array([10.0, 200.0, 30.0]),
        )
        constraints = [
            {"type": "unique", "column": "id"},
            {"type": "range", "column": "val", "min": 0, "max": 100},
        ]
        results = constraint_engine.evaluate_batch(data, constraints)
        passed = [r.passed for r in results]
        assert False in passed

    def test_evaluate_batch_empty(self, constraint_engine: ConstraintEngine):
        data = _dict_data(x=np.array([1]))
        results = constraint_engine.evaluate_batch(data, [])
        assert results == []


# =====================================================================
# ConstraintEngine — infer_constraints
# =====================================================================


class TestConstraintEngineInfer:
    """Automatic constraint inference."""

    def test_infer_suggests_not_null(
        self, constraint_engine: ConstraintEngine
    ):
        data = _dict_data(col=np.array([1, 2, 3, 4, 5]))
        suggestions = constraint_engine.infer_constraints(data, confidence=0.95)
        assert isinstance(suggestions, list)
        preds = [s.predicate for s in suggestions]
        has_not_null = any("NOT NULL" in p.upper() for p in preds)
        assert has_not_null

    def test_infer_suggests_unique(self, constraint_engine: ConstraintEngine):
        data = _dict_data(id=np.array([10, 20, 30, 40, 50]))
        suggestions = constraint_engine.infer_constraints(data, confidence=0.95)
        preds = [s.predicate for s in suggestions]
        has_unique = any("UNIQUE" in p.upper() for p in preds)
        assert has_unique

    def test_infer_suggests_range(self, constraint_engine: ConstraintEngine):
        data = _dict_data(val=np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        suggestions = constraint_engine.infer_constraints(data, confidence=0.95)
        preds = [s.predicate for s in suggestions]
        has_range = any("BETWEEN" in p.upper() for p in preds)
        assert has_range

    def test_infer_confidence_metadata(
        self, constraint_engine: ConstraintEngine
    ):
        data = _dict_data(x=np.array([1, 2, 3]))
        suggestions = constraint_engine.infer_constraints(data)
        for s in suggestions:
            assert isinstance(s, ConstraintSuggestion)
            assert 0.0 <= s.confidence <= 1.0
            assert s.sample_support > 0


# =====================================================================
# DataProfiler — profile_table
# =====================================================================


class TestDataProfilerTable:
    """Table-level profiling."""

    def test_profile_table_basic(self, profiler: DataProfiler):
        data = _dict_data(
            id=np.array([1, 2, 3, 4, 5]),
            score=np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
        )
        profile = profiler.profile_table(data, table_name="test")
        assert isinstance(profile, TableProfile)
        assert profile.table_name == "test"
        assert profile.row_count == 5
        assert profile.column_count == 2
        assert "id" in profile.column_profiles
        assert "score" in profile.column_profiles

    def test_profile_table_empty(self, profiler: DataProfiler):
        data: dict[str, np.ndarray] = {}
        profile = profiler.profile_table(data, table_name="empty")
        assert profile.row_count == 0
        assert profile.column_count == 0

    def test_profile_table_single_column(self, profiler: DataProfiler):
        data = _dict_data(x=np.array([100, 200, 300]))
        profile = profiler.profile_table(data)
        assert profile.column_count == 1
        assert profile.row_count == 3


# =====================================================================
# DataProfiler — profile_column
# =====================================================================


class TestDataProfilerColumn:
    """Column-level profiling for numeric and string data."""

    def test_profile_numeric_column(self, profiler: DataProfiler):
        data = _dict_data(
            val=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        )
        cp = profiler.profile_column(data, "val")
        assert isinstance(cp, ColumnProfile)
        assert cp.column_name == "val"
        assert cp.count == 10
        assert cp.null_count == 0
        assert cp.mean is not None
        assert cp.mean == pytest.approx(5.5)
        assert cp.std is not None
        assert cp.min_val == pytest.approx(1.0)
        assert cp.max_val == pytest.approx(10.0)

    def test_profile_string_column(self, profiler: DataProfiler):
        data = _dict_data(
            name=np.array(["alice", "bob", "charlie", "alice"], dtype=object)
        )
        cp = profiler.profile_column(data, "name")
        assert cp.count == 4
        assert cp.unique_count == 3
        # For non-numeric columns, mean should be None
        assert cp.mean is None

    def test_profile_column_with_nulls(self, profiler: DataProfiler):
        data = _dict_data(
            val=np.array([1.0, float("nan"), 3.0, float("nan"), 5.0])
        )
        cp = profiler.profile_column(data, "val")
        assert cp.null_count >= 2
        assert cp.null_rate > 0

    def test_profile_column_all_same_value(self, profiler: DataProfiler):
        data = _dict_data(val=np.array([42.0, 42.0, 42.0]))
        cp = profiler.profile_column(data, "val")
        assert cp.unique_count == 1
        assert cp.std is not None
        assert cp.std == pytest.approx(0.0)


# =====================================================================
# DataProfiler — compute_data_quality_score
# =====================================================================


class TestDataProfilerQualityScore:
    """Composite data quality score."""

    def test_perfect_data_high_score(self, profiler: DataProfiler):
        data = _dict_data(
            id=np.array([1, 2, 3, 4, 5]),
            val=np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
        )
        profile = profiler.profile_table(data, table_name="perfect")
        score = profiler.compute_data_quality_score(profile)
        assert 0.0 <= score <= 1.0
        assert score > 0.5

    def test_empty_profile_returns_one(self, profiler: DataProfiler):
        profile = TableProfile(table_name="empty")
        score = profiler.compute_data_quality_score(profile)
        assert score == 1.0

    def test_score_within_bounds(self, profiler: DataProfiler):
        data = _dict_data(
            x=np.array([1.0, float("nan"), 3.0]),
            y=np.array([10.0, 20.0, 30.0]),
        )
        profile = profiler.profile_table(data)
        score = profiler.compute_data_quality_score(profile)
        assert 0.0 <= score <= 1.0


# =====================================================================
# DataProfiler — detect_anomalies
# =====================================================================


class TestDataProfilerAnomalies:
    """Anomaly detection between profiles."""

    def test_identical_profiles_no_anomalies(self, profiler: DataProfiler):
        data = _dict_data(val=np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        p1 = profiler.profile_table(data, "t")
        p2 = profiler.profile_table(data, "t")
        anomalies = profiler.detect_anomalies(p2, p1)
        assert isinstance(anomalies, list)
        # Minor numerical differences may not trigger anomalies
        severe = [a for a in anomalies if a.severity == "error"]
        assert len(severe) == 0

    def test_row_count_change_detected(self, profiler: DataProfiler):
        small = _dict_data(val=np.array([1.0, 2.0]))
        big = _dict_data(val=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]))
        p_baseline = profiler.profile_table(small, "t")
        p_new = profiler.profile_table(big, "t")
        anomalies = profiler.detect_anomalies(p_new, p_baseline)
        types = [a.anomaly_type for a in anomalies]
        assert "row_count_change" in types


# =====================================================================
# Edge cases
# =====================================================================


class TestEdgeCases:
    """Boundary and degenerate input tests."""

    def test_single_row_data(self, monitor: QualityMonitor):
        data = _dict_data(x=np.array([42]))
        result = monitor.null_rate_check(data, "x", threshold=0.0)
        assert result.passed is True
        result = monitor.range_check(data, "x", min_val=0, max_val=100)
        assert result.passed is True
        result = monitor.uniqueness_check(data, ["x"])
        assert result.passed is True

    def test_all_nulls_fail_null_check(self, monitor: QualityMonitor):
        data = _dict_data(
            x=np.array([None, None, None], dtype=object)
        )
        result = monitor.null_rate_check(data, "x", threshold=0.5)
        assert result.passed is False
        assert result.metric_value == pytest.approx(2.0)

    def test_all_same_value_uniqueness(self, monitor: QualityMonitor):
        data = _dict_data(x=np.array([7, 7, 7, 7]))
        result = monitor.uniqueness_check(data, ["x"])
        assert result.passed is False

    def test_ks_test_single_element(self):
        s1 = np.array([1.0])
        s2 = np.array([2.0])
        result = DistributionAnalyzer.ks_test(s1, s2)
        assert result.statistic >= 0

    def test_psi_single_bin(self):
        data = np.array([1.0, 1.0, 1.0])
        psi = DistributionAnalyzer.psi_score(data, data, bins=1)
        assert psi >= 0.0

    def test_profiler_single_row(self, profiler: DataProfiler):
        data = _dict_data(val=np.array([99.0]))
        profile = profiler.profile_table(data, "single")
        assert profile.row_count == 1
        cp = profile.column_profiles.get("val")
        assert cp is not None
        assert cp.count == 1

    def test_constraint_engine_empty_data(
        self, constraint_engine: ConstraintEngine
    ):
        data = _dict_data(x=np.array([]))
        result = constraint_engine.evaluate(
            data, {"type": "not_null", "column": "x"}
        )
        # Empty data typically passes not-null (no violations possible)
        assert isinstance(result, ConstraintResult)

    def test_quality_monitor_generate_quality_delta(
        self, monitor: QualityMonitor
    ):
        old = [
            CheckResult(check_name="null_check", passed=True,
                        metric_value=0.0, threshold=0.1),
        ]
        new = [
            CheckResult(check_name="null_check", passed=False,
                        metric_value=0.5, threshold=0.1),
        ]
        delta = monitor.generate_quality_delta(old, new)
        assert len(delta.metric_changes) == 1
        assert delta.metric_changes[0].old_value == 0.0
        assert delta.metric_changes[0].new_value == 0.5
        assert "null_check" in delta.constraint_violations

    def test_profiler_report_generation(self, profiler: DataProfiler):
        data = _dict_data(
            val=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            name=np.array(["a", "b", "c", "d", "e"]),
        )
        profile = profiler.profile_table(data, "report_test")
        report = profiler.generate_profile_report(profile)
        assert isinstance(report, str)
        assert "report_test" in report
        assert "Rows" in report
