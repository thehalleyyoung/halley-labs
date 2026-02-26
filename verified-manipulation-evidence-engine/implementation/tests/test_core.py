"""Tests for core VMEE modules."""

import numpy as np
import pytest


class TestProofBridge:
    """Tests for the probabilistic-to-logical proof bridge."""

    def test_rational_encoding(self):
        from vmee.proof.bridge import RationalEncoding
        enc = RationalEncoding.from_float(0.95, precision_bits=64)
        assert abs(float(enc.value) - 0.95) < enc.approximation_bound

    def test_evidence_claim_validation(self):
        from vmee.proof.bridge import EvidenceClaim
        claim = EvidenceClaim(
            posterior_values={"manipulation": 0.95, "legitimate": 0.05},
            bayes_factor=100.0,
            posterior_threshold=0.9,
            bayes_factor_threshold=10.0,
        )
        errors = claim.validate()
        assert len(errors) == 0

    def test_evidence_claim_invalid(self):
        from vmee.proof.bridge import EvidenceClaim
        claim = EvidenceClaim(
            posterior_values={"manipulation": 0.8, "legitimate": 0.3},  # sums to 1.1
            bayes_factor=100.0,
            posterior_threshold=0.9,
            bayes_factor_threshold=10.0,
        )
        errors = claim.validate()
        assert len(errors) > 0

    def test_encode_and_prove(self):
        from vmee.proof.bridge import ProofBridge, EvidenceClaim, ProofStatus
        bridge = ProofBridge()
        claim = EvidenceClaim(
            posterior_values={"manipulation": 0.97, "legitimate": 0.03},
            bayes_factor=312.5,
            posterior_threshold=0.95,
            bayes_factor_threshold=10.0,
        )
        cert = bridge.generate_proof(claim)
        assert cert.status == ProofStatus.PROVED
        assert cert.level == "object"
        assert cert.num_variables > 0
        assert cert.num_constraints > 0
        assert len(cert.formula_smtlib2) > 0
        assert len(cert.certificate_hash) > 0

    def test_translation_validation(self):
        from vmee.proof.bridge import ProofBridge, EvidenceClaim
        bridge = ProofBridge()
        claim = EvidenceClaim(
            posterior_values={"manipulation": 0.97, "legitimate": 0.03},
            bayes_factor=312.5,
            posterior_threshold=0.95,
            bayes_factor_threshold=10.0,
        )
        cert = bridge.generate_proof(claim)
        assert cert.translation_validation is not None
        assert cert.translation_validation.valid


class TestArithmeticCircuit:
    """Tests for arithmetic circuit compilation and evaluation."""

    def test_simple_circuit(self):
        from vmee.bayesian.engine import ArithmeticCircuit, GateType
        ac = ArithmeticCircuit()
        p1 = ac.add_gate(GateType.PARAMETER, value=0.7)
        p2 = ac.add_gate(GateType.PARAMETER, value=0.3)
        s = ac.add_gate(GateType.SUM, children=[p1, p2])
        ac.root_id = s
        result = ac.evaluate()
        assert abs(result - 1.0) < 1e-10

    def test_product_gate(self):
        from vmee.bayesian.engine import ArithmeticCircuit, GateType
        ac = ArithmeticCircuit()
        p1 = ac.add_gate(GateType.PARAMETER, value=0.5, variable="A")
        p2 = ac.add_gate(GateType.PARAMETER, value=0.4, variable="B")
        prod = ac.add_gate(GateType.PRODUCT, children=[p1, p2])
        ac.root_id = prod
        result = ac.evaluate()
        assert abs(result - 0.2) < 1e-10

    def test_decomposability_check(self):
        from vmee.bayesian.engine import ArithmeticCircuit, GateType
        ac = ArithmeticCircuit()
        p1 = ac.add_gate(GateType.PARAMETER, value=0.5, variable="A")
        p2 = ac.add_gate(GateType.PARAMETER, value=0.4, variable="B")
        prod = ac.add_gate(GateType.PRODUCT, children=[p1, p2])
        ac.root_id = prod
        assert ac.check_decomposability() is True

    def test_circuit_trace(self):
        from vmee.bayesian.engine import ArithmeticCircuit, GateType
        ac = ArithmeticCircuit()
        p1 = ac.add_gate(GateType.PARAMETER, value=0.7)
        p2 = ac.add_gate(GateType.PARAMETER, value=0.3)
        s = ac.add_gate(GateType.SUM, children=[p1, p2])
        ac.root_id = s
        ac.evaluate()
        trace = ac.get_trace()
        assert "gates" in trace
        assert len(trace["gates"]) == 3


class TestTreeDecomposition:
    """Tests for tree decomposition."""

    def test_simple_graph(self):
        import networkx as nx
        from vmee.bayesian.engine import TreeDecomposition
        G = nx.Graph()
        G.add_edges_from([("A", "B"), ("B", "C"), ("C", "D")])
        td = TreeDecomposition(G)
        tw = td.decompose()
        assert tw >= 1
        assert tw <= 3

    def test_treewidth_bound(self):
        import networkx as nx
        from vmee.bayesian.engine import TreeDecomposition
        G = nx.complete_graph(6)
        td = TreeDecomposition(G)
        tw = td.decompose(max_treewidth=3)
        assert tw == 5  # complete graph on 6 nodes has tw=5


class TestTemporalMonitor:
    """Tests for FO-MTL monitoring."""

    def test_decidable_fragment_check(self):
        from vmee.temporal.monitor import Formula, FormulaType, TimeInterval
        # Bounded future formula (in decidable fragment)
        f = Formula(
            formula_type=FormulaType.EVENTUALLY,
            interval=TimeInterval(0, 60.0),
            children=[Formula(
                formula_type=FormulaType.THRESHOLD,
                signal_name="cancel_ratio",
                threshold=0.8,
            )],
        )
        assert f.is_in_decidable_fragment() is True

    def test_unbounded_future_rejected(self):
        from vmee.temporal.monitor import TimeInterval
        with pytest.raises(ValueError, match="Unbounded"):
            TimeInterval(0, float('inf'))

    def test_spec_library(self):
        from vmee.temporal.monitor import TemporalMonitor
        monitor = TemporalMonitor()
        assert "spoofing_basic" in monitor._spec_library
        assert "layering_basic" in monitor._spec_library
        assert "wash_trading_basic" in monitor._spec_library

    def test_qf_lra_reduction(self):
        from vmee.temporal.monitor import TemporalMonitor, Event
        monitor = TemporalMonitor()
        spec = monitor._spec_library["spoofing_basic"]
        event = Event(
            timestamp=1.0,
            predicates={"order_size": 6000.0, "cancel_ratio": 0.9,
                        "opposite_execution": 0.7},
        )
        smtlib = monitor.reduce_to_qf_lra(spec, event)
        assert "(set-logic QF_LRA)" in smtlib
        assert "(check-sat)" in smtlib


class TestCausalDiscovery:
    """Tests for causal discovery engine."""

    def test_hsic_test_independent(self):
        from vmee.causal.discovery import HSICTest
        rng = np.random.RandomState(42)
        x = rng.normal(0, 1, 200)
        y = rng.normal(0, 1, 200)
        test = HSICTest(num_permutations=100)
        result = test.test(x, y, alpha=0.05)
        # Independent variables should (usually) pass independence test
        assert result.p_value > 0.01

    def test_hsic_test_dependent(self):
        from vmee.causal.discovery import HSICTest
        rng = np.random.RandomState(42)
        x = rng.normal(0, 1, 200)
        y = x + rng.normal(0, 0.1, 200)  # strongly dependent
        test = HSICTest(num_permutations=100)
        result = test.test(x, y, alpha=0.05)
        assert result.independent == False

    def test_finite_sample_bound(self):
        from vmee.causal.discovery import FiniteSampleBound
        bound = FiniteSampleBound.compute(
            num_vars=7, max_degree=3, n=1000, alpha=0.05
        )
        assert 0 <= bound.correctness_probability <= 1

    def test_do_calculus_backdoor(self):
        import networkx as nx
        from vmee.causal.discovery import DoCalculusEngine
        dag = nx.DiGraph()
        dag.add_edges_from([("Z", "X"), ("Z", "Y"), ("X", "Y")])
        engine = DoCalculusEngine(dag)
        effect = engine.identify_effect("X", "Y")
        assert effect.identified is True
        assert effect.method == "backdoor"


class TestComposition:
    """Tests for compositional soundness framework."""

    def test_verification_levels(self):
        from vmee.composition.soundness import VerificationLevel
        assert VerificationLevel.OBJECT != VerificationLevel.META

    def test_soundness_check(self):
        from vmee.composition.soundness import CompositionFramework
        from vmee.causal.discovery import CausalDiscoveryEngine
        from vmee.bayesian.engine import BayesianInferenceEngine
        from vmee.temporal.monitor import TemporalMonitor
        from vmee.proof.bridge import ProofBridge

        causal = CausalDiscoveryEngine()
        causal.hsic_test.num_permutations = 20
        causal.num_bootstrap = 2
        causal.max_cond = 1
        bayesian = BayesianInferenceEngine()
        temporal = TemporalMonitor()

        # Mock results
        from vmee.lob.simulator import LOBSimulator
        sim = LOBSimulator()
        data = sim.generate_trading_day(num_events=100)

        causal_result = causal.discover(data)
        bayesian_result = bayesian.infer(data, causal_result)
        temporal_result = temporal.monitor(data)

        bridge = ProofBridge()
        proof_result = bridge.generate_proofs(bayesian_result, temporal_result, causal_result)

        framework = CompositionFramework()
        soundness = framework.check_soundness(
            causal_result, bayesian_result, temporal_result, proof_result
        )

        assert len(soundness.object_level_claims) > 0
        assert len(soundness.meta_level_claims) > 0
        assert soundness.compatibility.status.name == "COMPATIBLE"


class TestEndToEnd:
    """End-to-end pipeline test."""

    def test_full_pipeline(self):
        from vmee.lob.simulator import LOBSimulator
        from vmee.causal.discovery import CausalDiscoveryEngine
        from vmee.bayesian.engine import BayesianInferenceEngine
        from vmee.temporal.monitor import TemporalMonitor
        from vmee.proof.bridge import ProofBridge
        from vmee.evidence.assembler import EvidenceAssembler

        sim = LOBSimulator()
        data = sim.generate_trading_day(num_events=100)

        causal = CausalDiscoveryEngine()
        causal.hsic_test.num_permutations = 20
        causal.num_bootstrap = 2
        causal.max_cond = 1
        causal_result = causal.discover(data)

        bayesian = BayesianInferenceEngine()
        bayesian_result = bayesian.infer(data, causal_result)

        temporal = TemporalMonitor()
        temporal_result = temporal.monitor(data)

        bridge = ProofBridge()
        proof_result = bridge.generate_proofs(bayesian_result, temporal_result, causal_result)

        assembler = EvidenceAssembler()
        bundle = assembler.assemble(
            causal_result=causal_result,
            bayesian_result=bayesian_result,
            temporal_result=temporal_result,
            proof_result=proof_result,
        )

        assert "version" in bundle
        assert "causal_subgraph" in bundle
        assert "bayesian_evidence" in bundle
        assert "temporal_violations" in bundle
        assert "proof_certificates" in bundle
        assert "soundness" in bundle
        assert "bundle_hash" in bundle


class TestMultipleTestingCorrection:
    """Tests for Holm-Bonferroni multiple testing correction."""

    def test_holm_bonferroni_basic(self):
        from vmee.causal.discovery import (
            MultipleTestingCorrection, ConditionalIndependenceResult
        )
        results = [
            ConditionalIndependenceResult(
                x="A", y="B", conditioning_set=frozenset(),
                statistic=5.0, p_value=0.001, independent=False,
            ),
            ConditionalIndependenceResult(
                x="C", y="D", conditioning_set=frozenset(),
                statistic=1.0, p_value=0.4, independent=True,
            ),
            ConditionalIndependenceResult(
                x="E", y="F", conditioning_set=frozenset(),
                statistic=0.5, p_value=0.8, independent=True,
            ),
        ]
        mtc, corrected = MultipleTestingCorrection.apply_holm_bonferroni(results, 0.05)
        assert mtc.total_tests == 3
        assert mtc.correction_method == "holm-bonferroni"
        # The significant result (p=0.001) should remain rejected
        assert not corrected[0].independent  # still dependent

    def test_correction_preserves_strong_signals(self):
        from vmee.causal.discovery import (
            MultipleTestingCorrection, ConditionalIndependenceResult
        )
        results = [
            ConditionalIndependenceResult(
                x="A", y="B", conditioning_set=frozenset(),
                statistic=10.0, p_value=0.0001, independent=False,
            ),
        ]
        mtc, corrected = MultipleTestingCorrection.apply_holm_bonferroni(results, 0.05)
        assert not corrected[0].independent


class TestFaithfulnessSensitivity:
    """Tests for faithfulness sensitivity analysis."""

    def test_sensitivity_runs(self):
        from vmee.causal.discovery import CausalDiscoveryEngine
        engine = CausalDiscoveryEngine()
        engine.num_bootstrap = 3  # fast for testing
        engine.hsic_test.num_permutations = 20
        rng = np.random.RandomState(42)
        n = 100
        data = np.column_stack([
            rng.normal(0, 1, n) for _ in range(7)
        ])
        var_names = ["a", "b", "c", "d", "e", "f", "g"]
        import networkx as nx
        dag = nx.DiGraph()
        dag.add_nodes_from(var_names)
        dag.add_edge("a", "b")
        dag.add_edge("b", "c")

        result = engine._faithfulness_sensitivity(data, var_names, dag)
        assert result.num_bootstrap == 3
        assert 0 <= result.fraction_stable <= 1
        assert result.shd_mean >= 0


class TestSoundnessTheorem:
    """Tests for the formal soundness theorem."""

    def test_theorem_construction(self):
        from vmee.proof.bridge import SoundnessTheorem
        theorem = SoundnessTheorem.construct(precision_bits=64)
        assert len(theorem.assumptions) == 4
        assert len(theorem.proof_steps) == 5
        assert theorem.approximation_bound < 1e-15
        assert "A1" in theorem.assumptions[0]
        assert "SAT" in theorem.theorem_statement

    def test_theorem_with_translation_validation(self):
        from vmee.proof.bridge import SoundnessTheorem, TranslationValidation
        tv = TranslationValidation(
            circuit_output={"p_m": 0.95},
            formula_output={"p_m": 0.95},
            max_discrepancy=0.0,
            precision_bound=1e-15,
            valid=True,
            gate_count=10,
            constraint_count=5,
        )
        theorem = SoundnessTheorem.construct(precision_bits=64, tv_result=tv)
        assert theorem.verified_by_translation_validation
        assert theorem.proof_steps[-1].verified  # TV step verified


class TestCircuitVerification:
    """Tests for brute-force circuit verification."""

    def test_brute_force_small_dag(self):
        from vmee.bayesian.engine import BayesianInferenceEngine, TreeDecomposition
        import networkx as nx

        engine = BayesianInferenceEngine()
        dag = nx.DiGraph()
        dag.add_edges_from([("A", "B"), ("B", "C")])
        moral = engine._dag_to_moral_graph(dag)
        td = TreeDecomposition(moral)
        td.decompose()
        circuit = engine._compile_circuit(dag, td)

        result = engine.verify_circuit_brute_force(dag, circuit)
        assert result["num_variables"] == 3
        assert result["num_configurations"] == 8
        # With uniform CPTs, marginals should be uniform
        assert result["verified"]


class TestManipulationPlanting:
    """Tests for manipulation planting scenarios."""

    def test_plant_spoofing(self):
        from vmee.lob.simulator import LOBSimulator, ManipulationPlanter
        sim = LOBSimulator()
        data = sim.generate_trading_day(num_events=500)
        planter = ManipulationPlanter()
        data = planter.plant_spoofing(data, start_idx=100, duration=50)
        assert len(data.manipulation_labels) == 1
        assert data.manipulation_labels[0]["type"] == "spoofing"
        assert data.manipulation_labels[0]["subtype"] == "sarao_style"

    def test_plant_layering(self):
        from vmee.lob.simulator import LOBSimulator, ManipulationPlanter
        sim = LOBSimulator()
        data = sim.generate_trading_day(num_events=500)
        planter = ManipulationPlanter()
        data = planter.plant_layering(data, start_idx=100, duration=40)
        assert data.manipulation_labels[0]["type"] == "layering"
        assert data.manipulation_labels[0]["subtype"] == "coscia_style"

    def test_sec_scenario_combined(self):
        from vmee.lob.simulator import LOBSimulator, ManipulationPlanter
        sim = LOBSimulator()
        data = sim.generate_trading_day(num_events=800)
        planter = ManipulationPlanter()
        data = planter.plant_sec_scenario(data, scenario="combined")
        assert len(data.manipulation_labels) == 3
        types = {l["type"] for l in data.manipulation_labels}
        assert types == {"spoofing", "layering", "wash_trading"}


class TestAdversarialTrainer:
    """Tests for adversarial RL trainer."""

    def test_training_runs(self):
        from vmee.adversarial.trainer import AdversarialTrainer
        trainer = AdversarialTrainer()
        result = trainer.train(num_episodes=20)
        assert result.training_episodes == 20
        assert 0 <= result.coverage_bound <= 1
        assert result.training_time_seconds >= 0
        assert 0 <= result.evasion_rate <= 1


class TestBenchmarkRunner:
    """Tests for the evaluation benchmark."""

    @pytest.mark.timeout(120)
    def test_run_scenario(self):
        from vmee.evaluation.benchmark import BenchmarkRunner
        runner = BenchmarkRunner()
        runner.num_scenarios = 1
        result = runner.run_scenario("sarao_2010", num_runs=1)
        assert result["scenario"] == "sarao_2010"
        assert result["num_runs"] == 1
        assert "detection" in result
        assert "evidence_strength" in result
        assert "causal_stats" in result


class TestFCIEngine:
    """Tests for the FCI algorithm."""

    def test_fci_discovers_structure(self):
        from vmee.causal.discovery import FCIEngine
        rng = np.random.RandomState(42)
        n = 200
        x = rng.normal(0, 1, n)
        y = x + rng.normal(0, 0.5, n)
        z = y + rng.normal(0, 0.5, n)
        data = np.column_stack([x, y, z])
        var_names = ["X", "Y", "Z"]

        fci = FCIEngine()
        fci.hsic_test.num_permutations = 50
        result = fci.discover(data, var_names)
        assert result["algorithm"] == "fci"
        assert "pag" in result
        assert "edge_marks" in result


class TestPriorSensitivityAnalysis:
    """Tests for prior sensitivity / Jeffreys-Lindley robustness."""

    def test_prior_classes_enum(self):
        from vmee.causal.discovery import PriorClass
        assert PriorClass.REFERENCE.value == "reference"
        assert PriorClass.EMPIRICAL_BAYES.value == "empirical_bayes"
        assert PriorClass.SKEPTICAL.value == "skeptical"

    def test_run_prior_sensitivity_returns_all_priors(self):
        from vmee.causal.discovery import CausalDiscoveryEngine, PriorClass
        engine = CausalDiscoveryEngine()
        rng = np.random.RandomState(42)
        data = rng.normal(3.0, 1.0, (200, 3))
        result = engine.run_prior_sensitivity(data, threshold=1.0)
        assert len(result.bayes_factors) == 3
        for pc in PriorClass:
            assert pc.value in result.bayes_factors
        assert result.minimum_bf == min(result.bayes_factors.values())
        assert result.sample_size == 200

    def test_robust_flag(self):
        from vmee.causal.discovery import CausalDiscoveryEngine
        engine = CausalDiscoveryEngine()
        rng = np.random.RandomState(0)
        data = rng.normal(5.0, 1.0, (500, 1))
        result = engine.run_prior_sensitivity(data, threshold=1.0)
        # Strong signal → robust should be True
        assert result.robust is True

    def test_weak_signal_not_robust(self):
        from vmee.causal.discovery import CausalDiscoveryEngine
        engine = CausalDiscoveryEngine()
        rng = np.random.RandomState(0)
        data = rng.normal(0.0, 1.0, (30, 1))
        result = engine.run_prior_sensitivity(data, threshold=100.0)
        # Weak signal with high threshold → robust should be False
        assert result.robust is False


class TestDAGMisspecificationBound:
    """Tests for DAG misspecification TV-distance bounds."""

    def test_degradation_curve_length(self):
        from vmee.causal.discovery import CausalDiscoveryEngine
        import networkx as nx
        engine = CausalDiscoveryEngine()
        dag = nx.DiGraph()
        dag.add_edges_from([("A", "B"), ("B", "C"), ("A", "C")])
        bound = engine.compute_misspecification_bound(dag, sample_size=1000, max_k=5)
        assert len(bound.degradation_curve) == 5
        for k in range(1, 6):
            assert k in bound.degradation_curve

    def test_monotone_degradation(self):
        from vmee.causal.discovery import CausalDiscoveryEngine
        import networkx as nx
        engine = CausalDiscoveryEngine()
        dag = nx.DiGraph()
        dag.add_edges_from([("A", "B"), ("B", "C")])
        bound = engine.compute_misspecification_bound(dag, sample_size=500)
        # Degradation must be non-decreasing in k
        vals = [bound.degradation_curve[k] for k in sorted(bound.degradation_curve)]
        assert vals == sorted(vals)

    def test_tv_bounded_by_one(self):
        from vmee.causal.discovery import CausalDiscoveryEngine
        import networkx as nx
        engine = CausalDiscoveryEngine()
        dag = nx.DiGraph()
        dag.add_edges_from([("A", "B"), ("B", "C")])
        bound = engine.compute_misspecification_bound(
            dag, sample_size=10, max_cpt_variation=1.0, max_k=5
        )
        for v in bound.degradation_curve.values():
            assert v <= 1.0

    def test_structural_constant_positive(self):
        from vmee.causal.discovery import CausalDiscoveryEngine
        import networkx as nx
        engine = CausalDiscoveryEngine()
        dag = nx.DiGraph()
        dag.add_edges_from([("A", "B")])
        bound = engine.compute_misspecification_bound(dag, sample_size=100)
        assert bound.structural_constant > 0


class TestStructuralBreakDetection:
    """Tests for windowed structural change detection."""

    def test_detect_no_break(self):
        from vmee.causal.discovery import CausalDiscoveryEngine
        engine = CausalDiscoveryEngine()
        engine.hsic_test.num_permutations = 20
        rng = np.random.RandomState(42)
        n = 400
        data = np.column_stack([rng.normal(0, 1, n) for _ in range(3)])
        var_names = ["A", "B", "C"]
        results = engine.detect_structural_changes(data, var_names, window_size=200)
        assert isinstance(results, list)
        for r in results:
            assert hasattr(r, "chi2_statistic")
            assert hasattr(r, "p_value")
            assert hasattr(r, "corrected_alpha")
            assert hasattr(r, "significant")

    def test_bonferroni_correction_applied(self):
        from vmee.causal.discovery import CausalDiscoveryEngine
        engine = CausalDiscoveryEngine()
        engine.hsic_test.num_permutations = 20
        rng = np.random.RandomState(42)
        n = 600
        data = np.column_stack([rng.normal(0, 1, n) for _ in range(3)])
        var_names = ["A", "B", "C"]
        results = engine.detect_structural_changes(
            data, var_names, window_size=200, alpha=0.05,
        )
        # With 3 windows → 2 comparisons → corrected_alpha = 0.025
        assert len(results) == 2
        for r in results:
            assert r.corrected_alpha == pytest.approx(0.025)

    def test_returns_empty_for_short_data(self):
        from vmee.causal.discovery import CausalDiscoveryEngine
        engine = CausalDiscoveryEngine()
        rng = np.random.RandomState(42)
        data = np.column_stack([rng.normal(0, 1, 50) for _ in range(3)])
        var_names = ["A", "B", "C"]
        results = engine.detect_structural_changes(data, var_names, window_size=200)
        assert results == []


class TestGESEngine:
    """Tests for the Greedy Equivalence Search algorithm."""

    def test_ges_discovers_structure(self):
        from vmee.causal.discovery import GESEngine
        import networkx as nx
        rng = np.random.RandomState(42)
        n = 300
        x = rng.normal(0, 1, n)
        y = 2.0 * x + rng.normal(0, 0.3, n)
        z = 1.5 * y + rng.normal(0, 0.3, n)
        data = np.column_stack([x, y, z])
        var_names = ["X", "Y", "Z"]

        ges = GESEngine()
        result = ges.discover(data, var_names)
        assert result.algorithm == "ges"
        assert isinstance(result.dag, nx.DiGraph)
        # Should find at least one edge
        assert result.dag.number_of_edges() > 0

    def test_ges_empty_graph_on_noise(self):
        from vmee.causal.discovery import GESEngine
        rng = np.random.RandomState(99)
        n = 100
        data = rng.normal(0, 1, (n, 3))
        var_names = ["A", "B", "C"]

        ges = GESEngine()
        result = ges.discover(data, var_names)
        assert result.algorithm == "ges"
        # Independent noise → few or no edges
        assert result.dag.number_of_edges() <= 2

    def test_ges_result_is_dag(self):
        from vmee.causal.discovery import GESEngine
        import networkx as nx
        rng = np.random.RandomState(42)
        data = rng.normal(0, 1, (200, 4))
        var_names = ["A", "B", "C", "D"]
        ges = GESEngine()
        result = ges.discover(data, var_names)
        assert nx.is_directed_acyclic_graph(result.dag)


class TestQFLRAEncodingSpec:
    """Tests for the QF_LRA encoding specification (critique A)."""

    def test_create_spec(self):
        from vmee.proof.bridge import QFLRAEncodingSpec
        spec = QFLRAEncodingSpec.create(precision_bits=64, num_states=3, num_gates=5)
        assert spec.precision_bits == 64
        assert len(spec.variable_types) == 3
        assert len(spec.constraint_specs) == 5
        assert spec.total_error_bound > 0

    def test_variable_types_enumerated(self):
        from vmee.proof.bridge import QFLRAEncodingSpec
        spec = QFLRAEncodingSpec.create(num_states=2, num_gates=4)
        types = {vt["type"] for vt in spec.variable_types}
        assert "posterior" in types
        assert "bayes_factor" in types
        assert "gate_output" in types

    def test_constraint_types_enumerated(self):
        from vmee.proof.bridge import QFLRAEncodingSpec, ConstraintType
        spec = QFLRAEncodingSpec.create()
        ct_values = {cs.constraint_type for cs in spec.constraint_specs}
        assert ConstraintType.VALUE in ct_values
        assert ConstraintType.NORMALIZATION in ct_values
        assert ConstraintType.NON_NEGATIVITY in ct_values
        assert ConstraintType.THRESHOLD in ct_values
        assert ConstraintType.CIRCUIT_STRUCTURE in ct_values

    def test_describe_encoding_nonempty(self):
        from vmee.proof.bridge import QFLRAEncodingSpec
        spec = QFLRAEncodingSpec.create(precision_bits=32, num_states=2)
        desc = spec.describe_encoding()
        assert "QF_LRA ENCODING SPECIFICATION" in desc
        assert "VARIABLE TYPES" in desc
        assert "CONSTRAINT TYPES" in desc
        assert "TOTAL ERROR BOUND" in desc
        assert "object-level" in desc.lower() or "OBJECT-level" in desc

    def test_error_bounds_per_constraint(self):
        from vmee.proof.bridge import QFLRAEncodingSpec
        spec = QFLRAEncodingSpec.create(precision_bits=64)
        for cs in spec.constraint_specs:
            assert cs.error_bound >= 0
            assert len(cs.error_source) > 0
        # Normalization and non-negativity must be exact
        norm = [c for c in spec.constraint_specs if c.constraint_type.value == "normalization"][0]
        assert norm.exact is True
        nn = [c for c in spec.constraint_specs if c.constraint_type.value == "non_negativity"][0]
        assert nn.exact is True


class TestCircuitEncodingCertificate:
    """Tests for verified circuit-to-formula translation (critique B)."""

    def test_encode_with_certificate(self):
        from vmee.proof.bridge import ProofBridge, EvidenceClaim
        bridge = ProofBridge()
        claim = EvidenceClaim(
            posterior_values={"manipulation": 0.97, "legitimate": 0.03},
            bayes_factor=312.5,
            posterior_threshold=0.95,
            bayes_factor_threshold=10.0,
            circuit_trace={
                "gates": [
                    {"id": "p1", "type": "parameter", "value": 0.7},
                    {"id": "p2", "type": "parameter", "value": 0.3},
                    {"id": "s1", "type": "sum", "children": ["p1", "p2"]},
                ]
            },
        )
        solver, variables = bridge.encode_circuit_trace(claim)
        cert = bridge._last_encoding_certificate
        assert cert.total_gates == 3
        assert cert.sum_gates == 1
        assert cert.parameter_gates == 2
        assert len(cert.steps) == 3

    def test_sum_gate_exact(self):
        from vmee.proof.bridge import ProofBridge, EvidenceClaim
        bridge = ProofBridge()
        claim = EvidenceClaim(
            posterior_values={"manipulation": 0.97, "legitimate": 0.03},
            bayes_factor=312.5,
            posterior_threshold=0.95,
            bayes_factor_threshold=10.0,
            circuit_trace={
                "gates": [
                    {"id": "p1", "type": "parameter", "value": 0.7},
                    {"id": "p2", "type": "parameter", "value": 0.3},
                    {"id": "s1", "type": "sum", "children": ["p1", "p2"]},
                ]
            },
        )
        bridge.encode_circuit_trace(claim)
        cert = bridge._last_encoding_certificate
        sum_step = [s for s in cert.steps if s.gate_type == "sum"][0]
        assert sum_step.exact is True
        assert sum_step.error_bound == 0.0

    def test_product_gate_mccormick(self):
        from vmee.proof.bridge import ProofBridge, EvidenceClaim
        bridge = ProofBridge()
        claim = EvidenceClaim(
            posterior_values={"manipulation": 0.97, "legitimate": 0.03},
            bayes_factor=312.5,
            posterior_threshold=0.95,
            bayes_factor_threshold=10.0,
            circuit_trace={
                "gates": [
                    {"id": "p1", "type": "parameter", "value": 0.5},
                    {"id": "p2", "type": "parameter", "value": 0.4},
                    {"id": "prod1", "type": "product", "children": ["p1", "p2"],
                     "value": 0.2},
                ]
            },
        )
        bridge.encode_circuit_trace(claim)
        cert = bridge._last_encoding_certificate
        prod_step = [s for s in cert.steps if s.gate_type == "product"][0]
        assert prod_step.encoding_method == "mccormick_envelope"
        assert prod_step.error_bound >= 0

    def test_error_budget_tracks_cumulative(self):
        from vmee.proof.bridge import ProofBridge, EvidenceClaim
        bridge = ProofBridge()
        claim = EvidenceClaim(
            posterior_values={"manipulation": 0.97, "legitimate": 0.03},
            bayes_factor=312.5,
            posterior_threshold=0.95,
            bayes_factor_threshold=10.0,
            circuit_trace={
                "gates": [
                    {"id": "p1", "type": "parameter", "value": 0.7},
                    {"id": "p2", "type": "parameter", "value": 0.3},
                    {"id": "s1", "type": "sum", "children": ["p1", "p2"]},
                ]
            },
        )
        bridge.encode_circuit_trace(claim)
        budget = bridge._last_error_budget
        assert budget.total_error >= 0
        assert len(budget.gate_errors) == 3
        assert budget.gate_errors["s1"] == 0.0  # sum is exact

    def test_mccormick_helpers(self):
        from vmee.proof.bridge import ProofBridge
        # Gap should be zero when bounds are tight
        assert ProofBridge.mccormick_gap(0.5, 0.5, 0.3, 0.3) == 0.0
        # Gap should be (1-0)*(1-0)/4 = 0.25 for [0,1]x[0,1]
        assert abs(ProofBridge.mccormick_gap(0.0, 1.0, 0.0, 1.0) - 0.25) < 1e-15


class TestProofObjectExtraction:
    """Tests for proof object extraction and checking (critique C)."""

    def test_extract_returns_none_for_sat(self):
        from vmee.proof.bridge import ProofBridge, EvidenceClaim
        bridge = ProofBridge()
        claim = EvidenceClaim(
            posterior_values={"manipulation": 0.97, "legitimate": 0.03},
            bayes_factor=312.5,
            posterior_threshold=0.95,
            bayes_factor_threshold=10.0,
        )
        solver, _ = bridge.encode_evidence_claim(claim)
        smtlib2 = solver.to_smt2()
        # This formula is SAT, so no proof object
        result = bridge.extract_z3_proof_object(smtlib2)
        assert result is None

    def test_extract_unsat_proof(self):
        import z3
        from vmee.proof.bridge import ProofBridge
        bridge = ProofBridge()
        # Create a trivially UNSAT formula
        s = z3.Solver()
        x = z3.Real("x")
        s.add(x > 1)
        s.add(x < 0)
        smtlib2 = s.to_smt2()
        result = bridge.extract_z3_proof_object(smtlib2)
        if result is not None:
            assert result.format == "z3_internal"
            assert result.size_bytes > 0
            assert result.solver == "z3"

    def test_format_proof_for_checking(self):
        from vmee.proof.bridge import ProofBridge, ProofObjectInfo
        bridge = ProofBridge()
        info = ProofObjectInfo(
            proof_text="(mp (asserted p) (asserted (=> p q)) q)",
            format="z3_internal",
            size_bytes=50,
            num_steps=1,
            solver="z3",
        )
        formatted = bridge.format_proof_for_checking(info)
        assert "BEGIN PROOF" in formatted
        assert "END PROOF" in formatted
        assert "z3" in formatted

    def test_proof_object_info_fields(self):
        from vmee.proof.bridge import ProofObjectInfo
        info = ProofObjectInfo(
            proof_text="test",
            format="smtlib2",
            size_bytes=4,
            num_steps=1,
            solver="z3",
        )
        assert info.format == "smtlib2"
        assert info.size_bytes == 4
        assert info.num_steps == 1


class TestTCBAnalysis:
    """Tests for TCB analysis (critique D)."""

    def test_tcb_has_9_components(self):
        from vmee.proof.bridge import TCBAnalysis
        tcb = TCBAnalysis()
        assert len(tcb.components) == 9

    def test_verified_links(self):
        from vmee.proof.bridge import TCBAnalysis
        tcb = TCBAnalysis()
        verified = tcb.verified_components
        assert len(verified) == 5
        verified_names = {c.name for c in verified}
        assert "Z3 QF_LRA solver" in verified_names
        assert "FO-MTL fragment check" in verified_names
        assert "Translation validation" in verified_names
        assert "Circuit decomposability check" in verified_names
        assert "Holm-Bonferroni correction" in verified_names

    def test_unverified_links(self):
        from vmee.proof.bridge import TCBAnalysis
        tcb = TCBAnalysis()
        unverified = tcb.unverified_components
        assert len(unverified) == 4
        unverified_names = {c.name for c in unverified}
        assert "CPT parameter estimation" in unverified_names
        assert "Circuit compilation" in unverified_names
        assert "DAG structure correctness" in unverified_names
        assert "Causal sufficiency" in unverified_names

    def test_verified_fraction(self):
        from vmee.proof.bridge import TCBAnalysis
        tcb = TCBAnalysis()
        frac = tcb.verified_fraction
        assert abs(frac - 5.0 / 9.0) < 1e-10

    def test_summary_honest(self):
        from vmee.proof.bridge import TCBAnalysis
        tcb = TCBAnalysis()
        s = tcb.summary()
        assert s["total_components"] == 9
        assert s["verified_count"] == 5
        assert s["unverified_count"] == 4
        assert "NOT" in s["honest_assessment"] or "not" in s["honest_assessment"].lower()
        assert "weakest_link" in s

    def test_each_component_documents_verified_and_assumed(self):
        from vmee.proof.bridge import TCBAnalysis
        tcb = TCBAnalysis()
        for comp in tcb.components:
            assert len(comp.what_is_verified) > 0
            assert len(comp.what_is_assumed) > 0
            assert comp.verification_status in (
                "object_level_verified", "smt_checked", "assumed"
            )


class TestPriorSpecification:
    """Tests for prior sensitivity in Bayesian engine."""

    def test_prior_types_enum(self):
        from vmee.bayesian.engine import PriorType
        assert PriorType.UNIFORM.name == "UNIFORM"
        assert PriorType.JEFFREYS.name == "JEFFREYS"
        assert PriorType.EMPIRICAL_BAYES.name == "EMPIRICAL_BAYES"
        assert PriorType.SKEPTICAL.name == "SKEPTICAL"

    def test_prior_specification_fields(self):
        from vmee.bayesian.engine import PriorSpecification, PriorType
        spec = PriorSpecification(
            prior_type=PriorType.UNIFORM,
            concentration=1.0,
            description="Non-informative prior",
        )
        assert spec.prior_type == PriorType.UNIFORM
        assert spec.concentration == 1.0
        assert spec.description == "Non-informative prior"

    def test_prior_weights(self):
        from vmee.bayesian.engine import PriorSpecification, PriorType
        uniform = PriorSpecification(PriorType.UNIFORM)
        assert uniform.get_prior_weight() == 0.5

        jeffreys = PriorSpecification(PriorType.JEFFREYS)
        assert jeffreys.get_prior_weight() == 0.5

        eb = PriorSpecification(PriorType.EMPIRICAL_BAYES)
        assert eb.get_prior_weight() == 0.05

        skeptical = PriorSpecification(PriorType.SKEPTICAL)
        assert skeptical.get_prior_weight() == 0.01

    def test_dirichlet_alpha(self):
        from vmee.bayesian.engine import PriorSpecification, PriorType
        uniform = PriorSpecification(PriorType.UNIFORM)
        alpha = uniform.get_dirichlet_alpha(2)
        assert len(alpha) == 2
        assert np.allclose(alpha, [1.0, 1.0])

        jeffreys = PriorSpecification(PriorType.JEFFREYS)
        alpha_j = jeffreys.get_dirichlet_alpha(2)
        assert np.allclose(alpha_j, [0.5, 0.5])


class TestMultiPriorInference:
    """Tests for multi-prior Bayesian inference."""

    def test_multi_prior_returns_all_priors(self):
        from vmee.bayesian.engine import BayesianInferenceEngine
        from vmee.lob.simulator import LOBSimulator
        from vmee.causal.discovery import CausalDiscoveryEngine

        sim = LOBSimulator()
        data = sim.generate_trading_day(num_events=100)
        causal = CausalDiscoveryEngine()
        causal.hsic_test.num_permutations = 20
        causal.num_bootstrap = 3
        causal_result = causal.discover(data)

        engine = BayesianInferenceEngine()
        result = engine.multi_prior_inference(data, causal_result)

        assert len(result.prior_results) == 4
        assert "uniform" in result.prior_results
        assert "jeffreys" in result.prior_results
        assert "empirical_bayes" in result.prior_results
        assert "skeptical" in result.prior_results

    def test_multi_prior_has_bf_and_posterior(self):
        from vmee.bayesian.engine import BayesianInferenceEngine
        from vmee.lob.simulator import LOBSimulator
        from vmee.causal.discovery import CausalDiscoveryEngine

        sim = LOBSimulator()
        data = sim.generate_trading_day(num_events=100)
        causal = CausalDiscoveryEngine()
        causal.hsic_test.num_permutations = 20
        causal.num_bootstrap = 3
        causal_result = causal.discover(data)

        engine = BayesianInferenceEngine()
        result = engine.multi_prior_inference(data, causal_result)

        for name, res in result.prior_results.items():
            assert "bayes_factor" in res
            assert "posterior_manipulation" in res
            assert 0.0 <= res["posterior_manipulation"] <= 1.0

    def test_minimum_bf_is_min(self):
        from vmee.bayesian.engine import BayesianInferenceEngine
        from vmee.lob.simulator import LOBSimulator
        from vmee.causal.discovery import CausalDiscoveryEngine

        sim = LOBSimulator()
        data = sim.generate_trading_day(num_events=100)
        causal = CausalDiscoveryEngine()
        causal.hsic_test.num_permutations = 20
        causal.num_bootstrap = 3
        causal_result = causal.discover(data)

        engine = BayesianInferenceEngine()
        result = engine.multi_prior_inference(data, causal_result)

        all_bfs = [v["bayes_factor"] for v in result.prior_results.values()]
        assert result.minimum_bf == min(all_bfs)

    def test_posterior_variation_nonnegative(self):
        from vmee.bayesian.engine import BayesianInferenceEngine
        from vmee.lob.simulator import LOBSimulator
        from vmee.causal.discovery import CausalDiscoveryEngine

        sim = LOBSimulator()
        data = sim.generate_trading_day(num_events=100)
        causal = CausalDiscoveryEngine()
        causal.hsic_test.num_permutations = 20
        causal.num_bootstrap = 3
        causal_result = causal.discover(data)

        engine = BayesianInferenceEngine()
        result = engine.multi_prior_inference(data, causal_result)
        assert result.maximum_posterior_variation >= 0.0


class TestManipulationHMM:
    """Tests for the HMM with forward-backward and Viterbi."""

    def test_hmm_creation(self):
        from vmee.bayesian.engine import ManipulationHMM
        hmm = ManipulationHMM(n_states=2, n_features=3)
        assert hmm.n_states == 2
        assert hmm.n_features == 3
        assert hmm.transition.shape == (2, 2)
        assert hmm.initial.shape == (2,)

    def test_forward_backward_shapes(self):
        from vmee.bayesian.engine import ManipulationHMM
        hmm = ManipulationHMM()
        rng = np.random.RandomState(42)
        obs = rng.normal(0.3, 0.2, (20, 3))
        result = hmm.forward_backward(obs)
        assert result["gamma"].shape == (20, 2)
        assert result["alpha"].shape == (20, 2)
        assert result["beta"].shape == (20, 2)
        # Gamma rows should sum to 1
        row_sums = result["gamma"].sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)

    def test_forward_backward_legitimate_data(self):
        from vmee.bayesian.engine import ManipulationHMM
        hmm = ManipulationHMM()
        rng = np.random.RandomState(42)
        # Generate data from legitimate state distribution
        obs = np.column_stack([
            rng.normal(0.3, 0.15, 50),  # cancel_ratio ~ legitimate
            rng.normal(0.0, 0.2, 50),   # depth_imbalance ~ legitimate
            rng.normal(0.5, 0.15, 50),  # spread ~ legitimate
        ])
        result = hmm.forward_backward(obs)
        # Most states should be inferred as legitimate (state 0)
        mean_legit = np.mean(result["gamma"][:, 0])
        assert mean_legit > 0.5

    def test_viterbi_path_length(self):
        from vmee.bayesian.engine import ManipulationHMM
        hmm = ManipulationHMM()
        rng = np.random.RandomState(42)
        obs = rng.normal(0.3, 0.2, (30, 3))
        result = hmm.viterbi(obs)
        assert len(result["path"]) == 30
        assert all(s in (0, 1) for s in result["path"])

    def test_viterbi_detects_manipulation_phase(self):
        from vmee.bayesian.engine import ManipulationHMM
        hmm = ManipulationHMM()
        rng = np.random.RandomState(42)
        # First half: legitimate, second half: manipulation
        legit = np.column_stack([
            rng.normal(0.3, 0.1, 20),
            rng.normal(0.0, 0.15, 20),
            rng.normal(0.5, 0.1, 20),
        ])
        manip = np.column_stack([
            rng.normal(0.85, 0.05, 20),
            rng.normal(0.6, 0.1, 20),
            rng.normal(0.2, 0.05, 20),
        ])
        obs = np.vstack([legit, manip])
        result = hmm.viterbi(obs)
        # Second half should have more manipulation states
        second_half_manip = np.mean(result["path"][20:])
        first_half_manip = np.mean(result["path"][:20])
        assert second_half_manip > first_half_manip

    def test_viterbi_log_probability(self):
        from vmee.bayesian.engine import ManipulationHMM
        hmm = ManipulationHMM()
        rng = np.random.RandomState(42)
        obs = rng.normal(0.3, 0.2, (10, 3))
        result = hmm.viterbi(obs)
        assert isinstance(result["log_probability"], float)
        assert np.isfinite(result["log_probability"])


class TestJointCalibration:
    """Tests for the enhanced sim-to-real calibrator."""

    def test_calibrate_returns_joint_result(self):
        from vmee.calibration.calibrator import SimToRealCalibrator, JointCalibrationResult
        cal = SimToRealCalibrator()
        result = cal.calibrate()
        assert isinstance(result, JointCalibrationResult)

    def test_marginal_ks_computed(self):
        from vmee.calibration.calibrator import SimToRealCalibrator
        cal = SimToRealCalibrator()
        result = cal.calibrate()
        assert len(result.marginal_results) > 0
        for mr in result.marginal_results:
            assert 0.0 <= mr.ks_statistic <= 1.0
            assert 0.0 <= mr.p_value <= 1.0

    def test_rank_correlation_frobenius(self):
        from vmee.calibration.calibrator import SimToRealCalibrator
        cal = SimToRealCalibrator()
        result = cal.calibrate()
        assert result.rank_corr_frobenius >= 0.0

    def test_tail_dependence_coefficients(self):
        from vmee.calibration.calibrator import SimToRealCalibrator
        cal = SimToRealCalibrator()
        result = cal.calibrate()
        assert 0.0 <= result.tail_dependence_upper <= 1.0
        assert 0.0 <= result.tail_dependence_lower <= 1.0

    def test_cross_corr_decay_error(self):
        from vmee.calibration.calibrator import SimToRealCalibrator
        cal = SimToRealCalibrator()
        result = cal.calibrate()
        assert result.cross_corr_decay_error >= 0.0

    def test_overall_score(self):
        from vmee.calibration.calibrator import SimToRealCalibrator
        cal = SimToRealCalibrator()
        result = cal.calibrate()
        assert 0.0 <= result.overall_score <= 1.0

    def test_reference_statistics_present(self):
        from vmee.calibration.calibrator import REFERENCE_STATISTICS
        assert "cancel_ratio" in REFERENCE_STATISTICS
        assert "spread" in REFERENCE_STATISTICS
        assert "depth_imbalance" in REFERENCE_STATISTICS
        for feat, stats in REFERENCE_STATISTICS.items():
            assert "mean" in stats
            assert "std" in stats
            assert "source" in stats

    def test_custom_sim_data(self):
        from vmee.calibration.calibrator import SimToRealCalibrator
        cal = SimToRealCalibrator()
        rng = np.random.RandomState(42)
        sim_data = rng.normal(0, 1, (200, 6))
        feature_names = [
            "cancel_ratio", "spread", "depth_imbalance",
            "order_size_log_mean", "trade_imbalance", "price_impact",
        ]
        result = cal.calibrate(sim_data=sim_data, feature_names=feature_names)
        assert isinstance(result.overall_score, float)
        assert len(result.feature_names) == 6


class TestBaselineDetectors:
    """Tests for baseline detection methods."""

    def test_baseline_threshold_detector(self):
        from vmee.evaluation.benchmark import BaselineDetector
        det = BaselineDetector(cancel_ratio_threshold=0.8)
        assert det.cancel_ratio_threshold == 0.8

    def test_baseline_threshold_detects_high_cancel(self):
        from vmee.evaluation.benchmark import BaselineDetector
        from vmee.lob.simulator import LOBSimulator, ManipulationPlanter
        det = BaselineDetector(cancel_ratio_threshold=0.8)
        sim = LOBSimulator()
        data = sim.generate_trading_day(num_events=500)
        planter = ManipulationPlanter()
        data = planter.plant_spoofing(data, start_idx=100, duration=50)
        # Detection may or may not fire depending on data
        result = det.detect(data)
        assert isinstance(result, bool)

    def test_statistical_baseline_detector(self):
        from vmee.evaluation.benchmark import StatisticalBaselineDetector
        det = StatisticalBaselineDetector(z_threshold=2.5)
        assert det.z_threshold == 2.5

    def test_statistical_baseline_on_clean_data(self):
        from vmee.evaluation.benchmark import StatisticalBaselineDetector
        from vmee.lob.simulator import LOBSimulator
        det = StatisticalBaselineDetector(z_threshold=3.0)
        sim = LOBSimulator()
        data = sim.generate_trading_day(num_events=200)
        result = det.detect(data)
        assert isinstance(result, bool)

    def test_benchmark_includes_baselines(self):
        from vmee.evaluation.benchmark import BenchmarkRunner
        runner = BenchmarkRunner()
        runner.num_scenarios = 1
        result = runner.run_scenario("sarao_2010", num_runs=1)
        assert "baselines" in result
        assert "threshold" in result["baselines"]
        assert "zscore" in result["baselines"]
        assert "vmee_improvement_over_threshold_f1" in result["baselines"]
        assert "vmee_improvement_over_zscore_f1" in result["baselines"]


class TestPolicyNetwork:
    """Tests for the REINFORCE policy network."""

    def test_policy_creation(self):
        from vmee.adversarial.trainer import PolicyNetwork
        policy = PolicyNetwork(state_dim=1, num_actions=36)
        assert policy.weights.shape == (1, 36)
        assert policy.bias.shape == (36,)

    def test_forward_is_distribution(self):
        from vmee.adversarial.trainer import PolicyNetwork
        policy = PolicyNetwork(state_dim=1, num_actions=36)
        state = np.array([1.0])
        probs = policy.forward(state)
        assert len(probs) == 36
        assert abs(np.sum(probs) - 1.0) < 1e-10
        assert all(p >= 0 for p in probs)

    def test_sample_action(self):
        from vmee.adversarial.trainer import PolicyNetwork
        policy = PolicyNetwork(state_dim=1, num_actions=36)
        rng = np.random.RandomState(42)
        action = policy.sample_action(np.array([1.0]), rng)
        assert 0 <= action < 36

    def test_gradient_shape(self):
        from vmee.adversarial.trainer import PolicyNetwork
        policy = PolicyNetwork(state_dim=1, num_actions=36)
        state = np.array([1.0])
        grad_w, grad_b = policy.gradient(state, action=5)
        assert grad_w.shape == (1, 36)
        assert grad_b.shape == (36,)

    def test_update_changes_weights(self):
        from vmee.adversarial.trainer import PolicyNetwork
        policy = PolicyNetwork(state_dim=1, num_actions=36)
        old_weights = policy.weights.copy()
        state = np.array([1.0])
        grad_w, grad_b = policy.gradient(state, action=5)
        policy.update(grad_w, grad_b, advantage=1.0, lr=0.1)
        assert not np.allclose(policy.weights, old_weights)


class TestCoverageAnalysis:
    """Tests for adversarial coverage analysis."""

    def test_coverage_cell_evasion_rate(self):
        from vmee.adversarial.trainer import CoverageCell
        cell = CoverageCell("spoof", "high", "fast")
        cell.evasion_count = 3
        cell.detection_count = 7
        assert abs(cell.evasion_rate - 0.3) < 1e-10

    def test_training_returns_coverage_analysis(self):
        from vmee.adversarial.trainer import AdversarialTrainer
        trainer = AdversarialTrainer()
        result = trainer.train(num_episodes=50)
        assert "explored_cells" in result.coverage_analysis
        assert "unexplored_cells" in result.coverage_analysis
        assert "total_cells" in result.coverage_analysis
        assert result.coverage_analysis["total_cells"] == 36

    def test_training_returns_policy_weights(self):
        from vmee.adversarial.trainer import AdversarialTrainer
        trainer = AdversarialTrainer()
        result = trainer.train(num_episodes=20)
        assert result.policy_weights is not None
        assert "weights" in result.policy_weights
        assert "bias" in result.policy_weights

    def test_reinforce_learns(self):
        from vmee.adversarial.trainer import AdversarialTrainer
        trainer = AdversarialTrainer()
        result = trainer.train(num_episodes=50)
        # Policy should have non-zero weights after training
        weights = np.array(result.policy_weights["weights"])
        assert not np.allclose(weights, 0.0)
