"""
Tests for Phase B1 improvements: formal proofs, symbolic analysis,
ablation studies, and scalable benchmarks.
"""

import math
import pytest
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple


# ===================================================================
# Mock types needed by formal proof modules
# ===================================================================

@dataclass
class MockStutterClass:
    representative: str
    members: FrozenSet[str]

    def size(self):
        return len(self.members)


@dataclass
class MockStutterMonad:
    _classes: List[MockStutterClass]
    _unit: Dict[str, str]

    def compute_stutter_equivalence_classes(self):
        return self._classes

    def unit_map(self):
        return self._unit


@dataclass
class MockFairnessConstraint:
    b_states: FrozenSet[str]
    g_states: FrozenSet[str]
    index: int = 0


@dataclass
class MockCoalgebra:
    states: Set[str] = field(default_factory=set)
    initial_states: Set[str] = field(default_factory=set)
    fairness_constraints: List[MockFairnessConstraint] = field(default_factory=list)
    structure_map: Dict[str, Any] = field(default_factory=dict)


# ===================================================================
# Tests for T-Fair Theorem
# ===================================================================

class TestTFairTheorem:
    """Test the formal T-Fair coherence theorem prover."""

    def test_trivial_coherence_no_pairs(self):
        """With no fairness pairs, coherence holds vacuously."""
        from coacert.formal_proofs.tfair_theorem import TFairTheorem
        prover = TFairTheorem()
        classes = [
            MockStutterClass("s0", frozenset({"s0", "s1"})),
            MockStutterClass("s2", frozenset({"s2"})),
        ]
        holds, witnesses = prover.prove(classes, [])
        assert holds is True
        assert len(witnesses) == 0

    def test_coherence_holds_saturated(self):
        """When B_i and G_i are unions of stutter classes, coherence holds."""
        from coacert.formal_proofs.tfair_theorem import TFairTheorem
        prover = TFairTheorem()
        classes = [
            MockStutterClass("s0", frozenset({"s0", "s1"})),
            MockStutterClass("s2", frozenset({"s2", "s3"})),
            MockStutterClass("s4", frozenset({"s4"})),
        ]
        # B = {s0, s1} (= class [s0]_T), G = {s2, s3, s4} (= class [s2] ∪ [s4])
        pairs = [
            (frozenset({"s0", "s1"}), frozenset({"s2", "s3", "s4"})),
        ]
        holds, witnesses = prover.prove(classes, pairs)
        assert holds is True
        assert len(witnesses) == 1
        assert witnesses[0].is_valid
        assert witnesses[0].all_discharged

    def test_coherence_fails_split_class(self):
        """When a stutter class is split by B_i, coherence fails."""
        from coacert.formal_proofs.tfair_theorem import TFairTheorem
        prover = TFairTheorem()
        classes = [
            MockStutterClass("s0", frozenset({"s0", "s1"})),
            MockStutterClass("s2", frozenset({"s2"})),
        ]
        # B = {s0} splits class [s0]_T = {s0, s1}
        pairs = [
            (frozenset({"s0"}), frozenset({"s2"})),
        ]
        holds, witnesses = prover.prove(classes, pairs)
        assert holds is False
        assert witnesses[0].failed_count > 0

    def test_proof_hash_deterministic(self):
        """Proof hashes are deterministic."""
        from coacert.formal_proofs.tfair_theorem import TFairTheorem
        classes = [MockStutterClass("s0", frozenset({"s0", "s1"}))]
        pairs = [(frozenset({"s0", "s1"}), frozenset())]
        p1 = TFairTheorem()
        _, w1 = p1.prove(classes, pairs)
        p2 = TFairTheorem()
        _, w2 = p2.prove(classes, pairs)
        assert w1[0].proof_hash == w2[0].proof_hash

    def test_proof_summary(self):
        """Proof summary includes all fields."""
        from coacert.formal_proofs.tfair_theorem import TFairTheorem
        prover = TFairTheorem()
        classes = [MockStutterClass("s0", frozenset({"s0"}))]
        pairs = [(frozenset({"s0"}), frozenset())]
        prover.prove(classes, pairs)
        summary = prover.proof_summary()
        assert summary["theorem"] == "T-Fair Coherence"
        assert summary["holds"] is True
        assert summary["total_obligations"] > 0

    def test_singleton_classes_always_coherent(self):
        """Singleton stutter classes are always coherent."""
        from coacert.formal_proofs.tfair_theorem import TFairTheorem
        prover = TFairTheorem()
        classes = [
            MockStutterClass("s0", frozenset({"s0"})),
            MockStutterClass("s1", frozenset({"s1"})),
        ]
        pairs = [(frozenset({"s0"}), frozenset({"s1"}))]
        holds, _ = prover.prove(classes, pairs)
        assert holds is True

    def test_multiple_pairs(self):
        """Multiple fairness pairs are all checked."""
        from coacert.formal_proofs.tfair_theorem import TFairTheorem
        prover = TFairTheorem()
        classes = [
            MockStutterClass("s0", frozenset({"s0", "s1"})),
            MockStutterClass("s2", frozenset({"s2"})),
        ]
        pairs = [
            (frozenset({"s0", "s1"}), frozenset({"s2"})),
            (frozenset({"s2"}), frozenset({"s0", "s1"})),
        ]
        holds, witnesses = prover.prove(classes, pairs)
        assert holds is True
        assert len(witnesses) == 2


# ===================================================================
# Tests for Coherence Certificate
# ===================================================================

class TestCoherenceCertificate:
    """Test coherence certificate generation."""

    def test_build_certificate_coherent(self):
        """Build certificate for a coherent system."""
        from coacert.formal_proofs.coherence_certificate import CoherenceCertificateBuilder
        builder = CoherenceCertificateBuilder()
        classes = [
            MockStutterClass("s0", frozenset({"s0", "s1"})),
            MockStutterClass("s2", frozenset({"s2"})),
        ]
        # Unit map η is identity: each state maps to itself
        monad = MockStutterMonad(
            classes, {"s0": "s0", "s1": "s1", "s2": "s2"}
        )
        coalgebra = MockCoalgebra(
            states={"s0", "s1", "s2"},
            fairness_constraints=[
                MockFairnessConstraint(
                    frozenset({"s0", "s1"}), frozenset({"s2"}), index=0
                ),
            ],
        )
        cert = builder.build(coalgebra, monad, system_id="test")
        assert cert.all_coherent is True
        assert cert.certificate_hash
        assert cert.num_stutter_classes == 2
        assert cert.verification_time_seconds >= 0

    def test_certificate_json(self):
        """Certificate can be serialized to JSON."""
        from coacert.formal_proofs.coherence_certificate import CoherenceCertificateBuilder
        builder = CoherenceCertificateBuilder()
        classes = [MockStutterClass("s0", frozenset({"s0"}))]
        monad = MockStutterMonad(classes, {"s0": "s0"})
        coalgebra = MockCoalgebra(states={"s0"}, fairness_constraints=[])
        cert = builder.build(coalgebra, monad)
        j = cert.to_json()
        assert '"all_coherent": true' in j


# ===================================================================
# Tests for Conformance Certificate
# ===================================================================

class TestConformanceCertificate:
    """Test conformance certificate with automatic depth computation."""

    def test_depth_sufficiency(self):
        """Test automatic depth sufficiency computation."""
        from coacert.formal_proofs.conformance_certificate import ConformanceCertificateBuilder

        @dataclass
        class MockHypothesis:
            _states: Set[str]
            _actions: Set[str]

            def states(self): return self._states
            def actions(self): return self._actions
            def transition(self, s, a):
                idx = sorted(self._states).index(s) if s in self._states else 0
                next_idx = (idx + 1) % len(self._states)
                return sorted(self._states)[next_idx]

        hyp = MockHypothesis({"q0", "q1", "q2"}, {"a", "b"})
        builder = ConformanceCertificateBuilder()
        cert = builder.build(
            hypothesis=hyp,
            actual_depth=10,
            total_tests=500,
            concrete_state_count=6,
            system_id="test",
        )
        assert cert.depth_proof.hypothesis_states == 3
        assert cert.depth_proof.hypothesis_diameter >= 0
        assert cert.depth_proof.sufficient_depth > 0
        assert cert.certificate_hash

    def test_suggest_depth(self):
        """Test depth suggestion."""
        from coacert.formal_proofs.conformance_certificate import ConformanceCertificateBuilder

        @dataclass
        class MockHyp:
            _states: Set[str]
            _actions: Set[str]
            def states(self): return self._states
            def actions(self): return self._actions
            def transition(self, s, a): return sorted(self._states)[0]

        hyp = MockHyp({"q0", "q1"}, {"a"})
        builder = ConformanceCertificateBuilder()
        depth = builder.suggest_depth(hyp, concrete_state_count=10)
        assert depth >= 2  # At least diam + 1

    def test_state_bound_from_spec_params(self):
        """Test state bound derivation from spec parameters."""
        from coacert.formal_proofs.conformance_certificate import ConformanceCertificateBuilder
        builder = ConformanceCertificateBuilder()
        bound, derivation = builder.compute_state_bound({
            "x": {"type": "boolean"},
            "y": {"type": "bounded_int", "lo": 0, "hi": 3},
            "z": {"type": "enum", "values": ["a", "b", "c"]},
        })
        assert bound == 2 * 4 * 3  # 24
        assert "bool" in derivation

    def test_convergence_detection(self):
        """Test convergence detection from history."""
        from coacert.formal_proofs.conformance_certificate import ConformanceCertificateBuilder

        @dataclass
        class MockHyp2:
            _states: Set[str] = field(default_factory=lambda: {"q0"})
            _actions: Set[str] = field(default_factory=lambda: {"a"})
            def states(self): return self._states
            def actions(self): return self._actions
            def transition(self, s, a): return "q0"

        hyp = MockHyp2()
        builder = ConformanceCertificateBuilder()
        cert = builder.build(
            hyp, actual_depth=5, total_tests=100,
            convergence_history=[3, 2, 2, 2],
        )
        assert cert.convergence_detected is True


# ===================================================================
# Tests for Minimality Proof
# ===================================================================

class TestMinimalityProof:
    """Test minimality proof construction."""

    def test_basic_minimality(self):
        """Test basic minimality proof."""
        from coacert.formal_proofs.minimality_proof import MinimalityProof
        partition = [
            frozenset({"s0", "s1"}),
            frozenset({"s2"}),
        ]
        coalgebra = MockCoalgebra(states={"s0", "s1", "s2"})
        morphism = {"s0": "q0", "s1": "q0", "s2": "q1"}
        prover = MinimalityProof()
        witness = prover.prove(partition, coalgebra, morphism)
        assert witness.myhill_nerode.quotient_size == 2
        assert witness.myhill_nerode.original_size == 3
        assert witness.morphism_is_valid is True
        assert witness.certificate_hash


# ===================================================================
# Tests for Bloom Filter Analysis
# ===================================================================

class TestBloomFilterAnalysis:
    """Test Bloom filter false-positive rate analysis."""

    def test_optimal_parameters(self):
        """Test optimal Bloom filter parameter computation."""
        from coacert.symbolic.bloom_analysis import optimal_bloom_parameters
        config = optimal_bloom_parameters(1000, target_fpr=0.01)
        assert config.num_bits > 0
        assert config.num_hash_functions > 0
        assert config.expected_elements == 1000

    def test_fpr_bound(self):
        """Test FPR computation."""
        from coacert.symbolic.bloom_analysis import false_positive_bound
        # Large filter with few elements: very low FPR
        fpr = false_positive_bound(m=100000, k=7, n=100)
        assert fpr < 0.001

        # Small filter with many elements: high FPR
        fpr = false_positive_bound(m=64, k=1, n=1000)
        assert fpr > 0.5

    def test_soundness_analysis(self):
        """Test complete soundness analysis."""
        from coacert.symbolic.bloom_analysis import BloomFilterAnalysis
        analyzer = BloomFilterAnalysis(target_soundness=0.999)
        result = analyzer.analyze(
            witness_entries=100,
            total_states=1000,
            equivalence_classes=50,
        )
        assert result.bloom_fpr >= 0
        assert result.sha256_collision_prob >= 0
        assert result.combined_unsoundness_bound >= 0
        assert len(result.details) > 0
        assert result.recommended_config is not None

    def test_sensitivity_analysis(self):
        """Test sensitivity analysis."""
        from coacert.symbolic.bloom_analysis import BloomFilterAnalysis
        analyzer = BloomFilterAnalysis()
        table = analyzer.sensitivity_analysis(
            witness_entries=100,
            n_elements=50,
        )
        assert len(table) > 0
        # More bits per element → lower FPR
        for i in range(1, len(table)):
            assert table[i]["fpr"] <= table[i-1]["fpr"]

    def test_sha256_collision_negligible(self):
        """SHA-256 collision probability should be negligible for reasonable state counts."""
        from coacert.symbolic.bloom_analysis import BloomFilterAnalysis
        analyzer = BloomFilterAnalysis()
        result = analyzer.analyze(total_states=1000000, equivalence_classes=1000)
        assert result.sha256_collision_prob < 1e-50


# ===================================================================
# Tests for State Space Bounds
# ===================================================================

class TestStateSpaceBounds:
    """Test state-space bound computation."""

    def test_boolean_bound(self):
        """Boolean variables contribute factor 2."""
        from coacert.symbolic.state_space_bounds import StateSpaceBounds
        bounds = StateSpaceBounds.from_spec_params({
            "x": {"type": "boolean"},
            "y": {"type": "boolean"},
        })
        assert bounds.total_bound == 4

    def test_mixed_bounds(self):
        """Mixed variable types compute correct product."""
        from coacert.symbolic.state_space_bounds import StateSpaceBounds
        bounds = StateSpaceBounds.from_spec_params({
            "flag": {"type": "boolean"},
            "counter": {"type": "bounded_int", "lo": 0, "hi": 9},
            "status": {"type": "enum", "values": ["idle", "active", "done"]},
        })
        assert bounds.total_bound == 2 * 10 * 3  # 60

    def test_process_array_bound(self):
        """Process arrays compute exponential bounds."""
        from coacert.symbolic.state_space_bounds import StateSpaceBounds
        bounds = StateSpaceBounds.from_spec_params({
            "procs": {"type": "process_array", "count": 3, "states_per_process": 4},
        })
        assert bounds.total_bound == 4 ** 3  # 64

    def test_domain_analyzer(self):
        """Test individual domain analyzer methods."""
        from coacert.symbolic.state_space_bounds import DomainAnalyzer
        da = DomainAnalyzer()
        assert da.analyze_boolean("x").cardinality == 2
        assert da.analyze_bounded_int("y", 1, 10).cardinality == 10
        assert da.analyze_finite_set("z", 3).cardinality == 8
        assert da.analyze_function("f", 2, 3).cardinality == 9
        assert da.analyze_sequence("s", 2, 3).cardinality == 15  # 1+2+4+8


# ===================================================================
# Tests for Scalable Benchmarks
# ===================================================================

class TestScalableBenchmarks:
    """Test scalable benchmark generation."""

    def test_dining_philosophers_3(self):
        """Dining philosophers with 3 philosophers."""
        from coacert.specs.scalable_benchmarks import make_dining_philosophers
        spec = make_dining_philosophers(3)
        assert spec.state_count > 0
        assert len(spec.initial_states) > 0
        assert len(spec.fairness_pairs) == 3
        assert len(spec.actions) > 0

    def test_dining_philosophers_scaling(self):
        """State count scales with n."""
        from coacert.specs.scalable_benchmarks import make_dining_philosophers
        s3 = make_dining_philosophers(3).state_count
        s4 = make_dining_philosophers(4).state_count
        assert s4 > s3

    def test_token_ring_4(self):
        """Token ring with 4 processes."""
        from coacert.specs.scalable_benchmarks import make_token_ring
        spec = make_token_ring(4)
        assert spec.state_count == 64  # 4 × 2^4
        assert len(spec.initial_states) > 0
        assert len(spec.fairness_pairs) == 4

    def test_mutex_3(self):
        """Mutual exclusion with 3 processes."""
        from coacert.specs.scalable_benchmarks import make_mutex_n
        spec = make_mutex_n(3)
        assert spec.state_count > 0
        assert len(spec.initial_states) > 0
        # Check mutual exclusion: no state has >1 process in critical
        for s in spec.states:
            critical = sum(1 for c in s[:3] if c == "C")
            assert critical <= 1

    def test_list_benchmarks(self):
        """List available benchmarks."""
        from coacert.specs.scalable_benchmarks import list_scalable_benchmarks
        benchmarks = list_scalable_benchmarks()
        assert len(benchmarks) >= 3
        names = [b["name"] for b in benchmarks]
        assert "dining_philosophers" in names
        assert "token_ring" in names
        assert "mutex" in names


# ===================================================================
# Tests for Ablation Framework
# ===================================================================

class TestAblationFramework:
    """Test ablation study framework."""

    def test_ablation_result_compression(self):
        """Test AblationResult compression improvement calculation."""
        from coacert.evaluation.ablation import AblationResult, AblationComponent
        result = AblationResult(
            component=AblationComponent.STUTTERING,
            enabled=True,
            original_states=100,
            quotient_states=25,
        )
        assert abs(result.compression_improvement - 0.75) < 1e-6

    def test_ablation_study_result_table(self):
        """Test ablation study summary table generation."""
        from coacert.evaluation.ablation import (
            AblationResult, AblationStudyResult, AblationComponent
        )
        baseline = AblationResult(
            component=AblationComponent.STUTTERING,
            enabled=True,
            original_states=100,
            quotient_states=20,
            total_time_seconds=1.5,
            membership_queries=500,
        )
        ablation = AblationResult(
            component=AblationComponent.STUTTERING,
            enabled=False,
            original_states=100,
            quotient_states=40,
            total_time_seconds=1.0,
            membership_queries=300,
        )
        study = AblationStudyResult(
            benchmark_name="test",
            baseline=baseline,
            ablations=[ablation],
        )
        table = study.summary_table()
        assert "test" in table
        assert "STUTTERING" in table

    def test_scalability_result_complexity(self):
        """Test scalability result complexity estimation."""
        from coacert.evaluation.ablation import ScalabilityResult, ScalabilityDataPoint
        result = ScalabilityResult(
            benchmark_name="test",
            parameter_name="n",
            data_points=[
                ScalabilityDataPoint(parameter_value=10, original_states=100, total_time_seconds=0.1),
                ScalabilityDataPoint(parameter_value=20, original_states=400, total_time_seconds=0.4),
                ScalabilityDataPoint(parameter_value=40, original_states=1600, total_time_seconds=1.6),
                ScalabilityDataPoint(parameter_value=80, original_states=6400, total_time_seconds=6.4),
            ],
        )
        result.estimate_complexity()
        assert result.time_complexity_estimate != "insufficient data"
        assert "O(n^" in result.time_complexity_estimate

    def test_comparison_suite(self):
        """Test baseline comparison suite."""
        from coacert.evaluation.ablation import BaselineComparison, ComparisonSuiteResult
        suite = ComparisonSuiteResult(comparisons=[
            BaselineComparison(
                benchmark_name="2PC",
                coacert_states=10,
                coacert_time=0.5,
                baseline_name="Paige-Tarjan",
                baseline_states=10,
                baseline_time=1.0,
                states_match=True,
                speedup=2.0,
            ),
        ])
        d = suite.to_dict()
        assert d["all_match"] is True
        assert d["avg_speedup"] == 2.0
        table = suite.summary_table()
        assert "2PC" in table


# ===================================================================
# Tests for Preservation Theorem
# ===================================================================

class TestPreservationTheorem:
    """Test the preservation theorem prover."""

    def test_preservation_with_coherence(self):
        """Preservation holds when T-Fair coherence holds."""
        from coacert.formal_proofs.tfair_theorem import (
            TFairTheorem, PreservationTheorem
        )
        prover = TFairTheorem()
        classes = [
            MockStutterClass("s0", frozenset({"s0", "s1"})),
            MockStutterClass("s2", frozenset({"s2"})),
        ]
        pairs = [(frozenset({"s0", "s1"}), frozenset({"s2"}))]
        _, tfair_witnesses = prover.prove(classes, pairs)

        coalgebra = MockCoalgebra(states={"s0", "s1", "s2"})
        quotient = MockCoalgebra(states={"q0", "q1"})
        morphism = {"s0": "q0", "s1": "q0", "s2": "q1"}

        pres = PreservationTheorem()
        witness = pres.prove(coalgebra, quotient, morphism, tfair_witnesses)
        assert witness.coherence_holds is True
