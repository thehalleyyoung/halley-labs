"""Tests for B1 improvements: conformance gap, T-Fair counterexample, asymmetric benchmark."""

import pytest
from coacert.learner.conformance_gap import (
    ConformanceGapAnalyzer,
    GapMeasurement,
    GapAnalysisResult,
)
from coacert.functor.coherence_counterexample import (
    build_coherence_counterexample,
    verify_coherence_fails,
    verify_liveness_violation,
    full_counterexample_analysis,
)
from coacert.specs.asymmetric_token_ring import AsymmetricTokenRingSpec
from coacert.specs.spec_registry import SpecRegistry


# ── Conformance Gap Analysis ──────────────────────────────────────────────

class TestConformanceGapAnalyzer:

    def test_error_bound_zero_at_sufficient_depth(self):
        """W-method completeness: bound is 0 when k >= d + (m-n+1)."""
        analyzer = ConformanceGapAnalyzer(
            n_actions=2,
            true_quotient_states=5,
        )
        # n_hypothesis=5, n_concrete=10 => d=4, sufficient = 4+5+1 = 10
        bound = analyzer.compute_error_bound(k=10, n_hypothesis=5, n_concrete=10)
        assert bound == 0.0

    def test_error_bound_positive_below_sufficient(self):
        analyzer = ConformanceGapAnalyzer(n_actions=2, true_quotient_states=5)
        bound = analyzer.compute_error_bound(k=3, n_hypothesis=5, n_concrete=10)
        assert bound > 0.0

    def test_error_bound_decreases_with_k(self):
        analyzer = ConformanceGapAnalyzer(n_actions=3, true_quotient_states=5)
        bounds = [
            analyzer.compute_error_bound(k=k, n_hypothesis=5, n_concrete=10)
            for k in range(1, 15)
        ]
        # Bound should be non-increasing
        for i in range(len(bounds) - 1):
            assert bounds[i] >= bounds[i + 1]

    def test_sufficient_depth_formula(self):
        analyzer = ConformanceGapAnalyzer(n_actions=2, true_quotient_states=5)
        sd = analyzer.sufficient_depth(n_hypothesis=5, n_concrete=20)
        # d = 4, m-n = 15, so sufficient = 4 + 15 + 1 = 20
        assert sd == 20

    def test_measure_gap_without_ground_truth(self):
        analyzer = ConformanceGapAnalyzer(
            n_actions=2, true_quotient_states=10
        )
        partition = [frozenset({f"s{i}"}) for i in range(8)]
        m = analyzer.measure_gap(k=5, hypothesis_partition=partition, n_concrete=20)
        assert m.hypothesis_states == 8
        assert m.true_quotient_states == 10
        assert m.missing_classes == 2

    def test_measure_gap_with_ground_truth(self):
        true_partition = [
            frozenset({"s0", "s1"}),
            frozenset({"s2"}),
            frozenset({"s3", "s4"}),
        ]
        analyzer = ConformanceGapAnalyzer(
            n_actions=2,
            true_quotient_states=3,
            true_partition=true_partition,
        )
        # Hypothesis that incorrectly splits {s0, s1}
        hyp_partition = [
            frozenset({"s0"}),
            frozenset({"s1"}),
            frozenset({"s2"}),
            frozenset({"s3", "s4"}),
        ]
        m = analyzer.measure_gap(k=3, hypothesis_partition=hyp_partition, n_concrete=5)
        assert m.hypothesis_states == 4
        assert m.extra_classes == 1

    def test_full_analysis(self):
        true_partition = [
            frozenset({"s0", "s1"}),
            frozenset({"s2", "s3"}),
        ]
        analyzer = ConformanceGapAnalyzer(
            n_actions=2,
            true_quotient_states=2,
            true_partition=true_partition,
        )
        partitions_by_k = {
            1: [frozenset({"s0"}), frozenset({"s1"}),
                frozenset({"s2"}), frozenset({"s3"})],
            3: [frozenset({"s0", "s1"}), frozenset({"s2"}), frozenset({"s3"})],
            5: [frozenset({"s0", "s1"}), frozenset({"s2", "s3"})],
        }
        result = analyzer.analyze(partitions_by_k, n_concrete=4)
        assert len(result.measurements) == 3
        # At k=5, should match ground truth
        m5 = result.gap_at(5)
        assert m5 is not None
        assert m5.hypothesis_states == 2

    def test_convergence_data(self):
        analyzer = ConformanceGapAnalyzer(n_actions=2, true_quotient_states=3)
        partitions = {
            1: [frozenset({"s0"}), frozenset({"s1"}), frozenset({"s2"}),
                frozenset({"s3"}), frozenset({"s4"})],
            3: [frozenset({"s0", "s1"}), frozenset({"s2"}),
                frozenset({"s3", "s4"})],
        }
        result = analyzer.analyze(partitions, n_concrete=5)
        data = result.convergence_data()
        assert "k" in data
        assert "gap_ratio" in data
        assert len(data["k"]) == 2


# ── T-Fair Coherence Counterexample ───────────────────────────────────────

class TestCoherenceCounterexample:

    def test_counterexample_system_constructed(self):
        sys = build_coherence_counterexample()
        assert len(sys.states) == 4
        assert "s0" in sys.initial_states
        assert len(sys.fairness_pairs) == 1
        assert len(sys.stutter_classes) == 3

    def test_coherence_fails(self):
        sys = build_coherence_counterexample()
        assert verify_coherence_fails(sys) is True

    def test_stutter_class_split_by_B(self):
        sys = build_coherence_counterexample()
        b_set = sys.fairness_pairs[0][0]
        merged_class = frozenset({"s1", "s2"})
        assert merged_class in sys.stutter_classes
        # s1 is in B but s2 is not
        assert "s1" in b_set
        assert "s2" not in b_set

    def test_liveness_violation_detected(self):
        sys = build_coherence_counterexample()
        result = verify_liveness_violation(sys)
        assert result["coherence_fails"] is True
        assert result["liveness_spurious"] is True
        # Original: path via s2 doesn't visit B
        assert result["original_system"]["path_via_s2_visits_B"] is False
        # Quotient: merged path appears to visit B (spurious)
        assert result["quotient_system"]["merged_path_visits_B"] is True

    def test_full_analysis(self):
        analysis = full_counterexample_analysis()
        assert analysis["coherence_fails"] is True
        assert analysis["liveness_analysis"]["liveness_spurious"] is True
        assert len(analysis["system"]["states"]) == 4

    def test_same_ap_labels(self):
        """s1 and s2 must have identical AP labels for stutter equivalence."""
        sys = build_coherence_counterexample()
        assert sys.labels["s1"] == sys.labels["s2"]

    def test_same_successor_structure(self):
        """s1 and s2 must have identical successor structure."""
        sys = build_coherence_counterexample()
        assert sys.transitions["s1"] == sys.transitions["s2"]


# ── Asymmetric Token Ring ─────────────────────────────────────────────────

class TestAsymmetricTokenRing:

    def test_construction(self):
        spec = AsymmetricTokenRingSpec(n_nodes=4)
        module = spec.get_spec()
        assert module.name == "AsymmetricTokenRing"

    def test_validation_passes(self):
        spec = AsymmetricTokenRingSpec(n_nodes=3)
        errors = spec.validate()
        assert errors == [], f"Validation errors: {errors}"

    def test_properties(self):
        spec = AsymmetricTokenRingSpec(n_nodes=4)
        props = spec.get_properties()
        assert len(props) == 3
        names = [p.name for p in props]
        assert "Mutex" in names
        assert "AllServed" in names
        assert "PriorityServed" in names

    def test_config_no_symmetry(self):
        """Asymmetric protocol should have no symmetry sets."""
        spec = AsymmetricTokenRingSpec(n_nodes=4)
        config = spec.get_config()
        assert config["symmetry_sets"] == []
        assert config["spec_name"] == "AsymmetricTokenRing"

    def test_min_nodes(self):
        with pytest.raises(ValueError):
            AsymmetricTokenRingSpec(n_nodes=2)

    def test_supported_configurations(self):
        configs = AsymmetricTokenRingSpec.supported_configurations()
        assert len(configs) >= 3

    def test_registry_includes_asymmetric(self):
        registry = SpecRegistry.default()
        assert "AsymmetricTokenRing" in registry
        presets = registry.get_presets("AsymmetricTokenRing")
        assert len(presets) >= 2

    def test_state_estimate(self):
        spec = AsymmetricTokenRingSpec(n_nodes=3)
        config = spec.get_config()
        assert config["expected_states"] > 0

    def test_multiple_sizes(self):
        for n in [3, 4, 5]:
            spec = AsymmetricTokenRingSpec(n_nodes=n)
            errors = spec.validate()
            assert errors == [], f"n={n}: {errors}"
