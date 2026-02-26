"""
Phase B2 Math Rigor Pass: comprehensive tests verifying all mathematical
claims in the formal_proofs and evaluation modules, ensuring consistency
across theorem statements.
"""

import math
import pytest
from types import SimpleNamespace
from typing import FrozenSet, List, Set, Tuple

# ── Imports from formal_proofs ────────────────────────────────────────────
from coacert.formal_proofs.tfair_theorem import (
    ObligationStatus,
    TFairProofObligation,
    TFairProofWitness,
    TFairTheorem,
    PreservationTheorem,
    PreservationProofWitness,
)
from coacert.formal_proofs.categorical_diagram import (
    CategoricalDiagramVerifier,
    DiagramStatus,
    NaturalityWitness,
    DiagramVerificationResult,
)
from coacert.formal_proofs.ctl_star_preservation import (
    FormulaKind,
    FormulaNode,
    InductionCaseStatus,
    CTLStarPreservationProof,
    StreettAcceptancePreservation,
)
from coacert.formal_proofs.conformance_certificate import (
    ConformanceCertificateBuilder,
    DepthSufficiencyProof,
    ConformanceCertificate,
)
from coacert.formal_proofs.minimality_proof import (
    MinimalityProof,
    MinimalityWitness,
    MyHillNerodeWitness,
)
from coacert.formal_proofs.proof_obligation_tracker import (
    ProofObligationTracker,
    ObligationCategory,
    DischargeStatus,
    ProofObligation,
)

# ── Imports from evaluation ───────────────────────────────────────────────
from coacert.evaluation.bloom_soundness import (
    BloomSoundnessAnalyzer,
    AdaptiveBloomConfig,
    SoundnessExperiment,
)
from coacert.evaluation.baseline_comparison import (
    LTS,
    PaigeTarjanBaseline,
    NaiveBisimulation,
    StatisticalTest,
    ComparisonReport,
    AlgorithmRun,
)


# ── Helpers ───────────────────────────────────────────────────────────────

def _make_stutter_class(representative: str, members: FrozenSet[str]):
    """Create a SimpleNamespace mimicking a stutter equivalence class."""
    return SimpleNamespace(representative=representative, members=members)


def _make_4state_coherent_system():
    """4-state system where coherence holds.

    States: s0, s1, s2, s3
    Stutter classes: {s0, s1}, {s2, s3}
    Fairness pair: B = {s0, s1}, G = {s2, s3}
    Both B and G are exact unions of stutter classes.
    """
    classes = [
        _make_stutter_class("s0", frozenset({"s0", "s1"})),
        _make_stutter_class("s2", frozenset({"s2", "s3"})),
    ]
    fairness_pairs = [
        (frozenset({"s0", "s1"}), frozenset({"s2", "s3"})),
    ]
    return classes, fairness_pairs


def _make_incoherent_system():
    """System where coherence fails: acceptance set splits a stutter class.

    States: s0, s1, s2, s3
    Stutter classes: {s0, s1}, {s2, s3}
    Fairness pair: B = {s0, s2}, G = {s1, s3}
    B splits class {s0, s1} (contains s0 but not s1).
    """
    classes = [
        _make_stutter_class("s0", frozenset({"s0", "s1"})),
        _make_stutter_class("s2", frozenset({"s2", "s3"})),
    ]
    fairness_pairs = [
        (frozenset({"s0", "s2"}), frozenset({"s1", "s3"})),
    ]
    return classes, fairness_pairs


def _make_small_lts():
    """Small LTS for baseline comparison tests.

    States a, b, c, d with labels distinguishing a|b from c|d.
    a --x--> b, b --x--> a (cycle), c --x--> d, d --x--> c (cycle)
    a and b are bisimilar; c and d are bisimilar.
    """
    lts = LTS(
        states={"a", "b", "c", "d"},
        transitions={
            ("a", "x"): {"b"},
            ("b", "x"): {"a"},
            ("c", "x"): {"d"},
            ("d", "x"): {"c"},
        },
        labels={
            "a": frozenset({"p"}),
            "b": frozenset({"p"}),
            "c": frozenset({"q"}),
            "d": frozenset({"q"}),
        },
        initial="a",
        actions={"x"},
    )
    return lts


# ══════════════════════════════════════════════════════════════════════════
# 1. TestTFairCoherenceTheorem
# ══════════════════════════════════════════════════════════════════════════

class TestTFairCoherenceTheorem:
    """Verify formal T-Fair coherence theorem on constructed examples."""

    def test_coherence_holds_4state(self):
        """Coherence holds when acceptance sets are unions of stutter classes."""
        classes, pairs = _make_4state_coherent_system()
        prover = TFairTheorem()
        holds, witnesses = prover.prove(classes, pairs)
        assert holds is True
        assert len(witnesses) == 1
        w = witnesses[0]
        assert w.is_valid is True
        assert w.all_discharged is True
        assert w.failed_count == 0

    def test_coherence_fails_split_class(self):
        """Coherence fails when an acceptance set splits a stutter class."""
        classes, pairs = _make_incoherent_system()
        prover = TFairTheorem()
        holds, witnesses = prover.prove(classes, pairs)
        assert holds is False
        w = witnesses[0]
        assert w.is_valid is False
        assert w.failed_count > 0
        # At least one obligation should record a split
        failed = [o for o in w.obligations if o.status == ObligationStatus.FAILED]
        assert len(failed) > 0
        assert "split" in failed[0].error_detail.lower()

    def test_proof_witness_obligation_counts(self):
        """Proof witness contains the correct number of obligations.

        For k stutter classes and 1 fairness pair, we expect 2*k obligations
        (one B-check and one G-check per class).
        """
        classes, pairs = _make_4state_coherent_system()
        prover = TFairTheorem()
        _, witnesses = prover.prove(classes, pairs)
        w = witnesses[0]
        # 2 classes × 2 components (B, G) = 4 obligations
        assert len(w.obligations) == 4
        assert w.discharged_count == 4

    def test_proof_hash_deterministic(self):
        """Proof hash should be deterministic across invocations."""
        classes, pairs = _make_4state_coherent_system()
        prover1 = TFairTheorem()
        _, w1 = prover1.prove(classes, pairs)
        prover2 = TFairTheorem()
        _, w2 = prover2.prove(classes, pairs)
        assert w1[0].proof_hash == w2[0].proof_hash
        assert len(w1[0].proof_hash) == 64  # SHA-256 hex digest

    def test_preservation_theorem_coherent(self):
        """Preservation theorem reports correctly when coherence holds."""
        classes, pairs = _make_4state_coherent_system()
        prover = TFairTheorem()
        _, witnesses = prover.prove(classes, pairs)
        # Build mock coalgebra/quotient
        coalgebra = SimpleNamespace(
            states=["s0", "s1", "s2", "s3"],
            initial_states={"s0"},
        )
        quotient = SimpleNamespace(
            states=["q0", "q1"],
            initial_states={"q0"},
        )
        morphism = {"s0": "q0", "s1": "q0", "s2": "q1", "s3": "q1"}
        pres = PreservationTheorem()
        w = pres.prove(coalgebra, quotient, morphism, witnesses)
        assert w.coherence_holds is True
        assert w.morphism_is_surjective is True
        assert w.morphism_respects_initial is True
        assert w.all_preserved is True
        assert len(w.proof_hash) == 64

    def test_preservation_theorem_incoherent(self):
        """Preservation theorem reports failure when coherence fails."""
        classes, pairs = _make_incoherent_system()
        prover = TFairTheorem()
        _, witnesses = prover.prove(classes, pairs)
        coalgebra = SimpleNamespace(states=["s0", "s1", "s2", "s3"])
        quotient = SimpleNamespace(states=["q0", "q1"])
        morphism = {"s0": "q0", "s1": "q0", "s2": "q1", "s3": "q1"}
        pres = PreservationTheorem()
        w = pres.prove(coalgebra, quotient, morphism, witnesses)
        assert w.coherence_holds is False
        assert w.all_preserved is False

    def test_multiple_fairness_pairs(self):
        """Coherence checked independently for each fairness pair."""
        classes = [
            _make_stutter_class("s0", frozenset({"s0", "s1"})),
            _make_stutter_class("s2", frozenset({"s2", "s3"})),
        ]
        pairs = [
            (frozenset({"s0", "s1"}), frozenset({"s2", "s3"})),  # coherent
            (frozenset({"s0"}), frozenset({"s2", "s3"})),          # splits {s0,s1}
        ]
        prover = TFairTheorem()
        holds, witnesses = prover.prove(classes, pairs)
        assert holds is False
        assert witnesses[0].is_valid is True
        assert witnesses[1].is_valid is False


# ══════════════════════════════════════════════════════════════════════════
# 2. TestCategoricalDiagram
# ══════════════════════════════════════════════════════════════════════════

class TestCategoricalDiagram:
    """Verify categorical diagram (naturality, unit, multiplication)."""

    def _setup_coherent(self):
        """Build a coherent setup with singleton stutter classes.

        Using singleton classes ensures η is the identity and η(B)
        is trivially a union of stutter classes, so all diagrams commute.
        """
        classes = [
            _make_stutter_class("s0", frozenset({"s0"})),
            _make_stutter_class("s1", frozenset({"s1"})),
            _make_stutter_class("s2", frozenset({"s2"})),
            _make_stutter_class("s3", frozenset({"s3"})),
        ]
        pairs = [
            (frozenset({"s0", "s1"}), frozenset({"s2", "s3"})),
        ]
        # Identity eta (each state is its own representative)
        eta = {"s0": "s0", "s1": "s1", "s2": "s2", "s3": "s3"}
        # mu is idempotent identity
        mu = {"s0": "s0", "s1": "s1", "s2": "s2", "s3": "s3"}
        # Morphism that respects stutter classes
        morphism = {"s0": "q0", "s1": "q0", "s2": "q1", "s3": "q1"}
        return classes, pairs, eta, mu, morphism

    def test_naturality_square_commutes(self):
        """Naturality square commutes for a well-formed morphism."""
        classes, pairs, eta, mu, h = self._setup_coherent()
        verifier = CategoricalDiagramVerifier()
        result = verifier.verify_naturality(
            classes, pairs, eta, [("h1", h)]
        )
        assert result.aggregate_commutes is True
        assert result.status == DiagramStatus.COMMUTES
        assert len(result.naturality_witnesses) == 1
        assert result.naturality_witnesses[0].commutes is True

    def test_unit_compatibility(self):
        """Unit compatibility: δ ∘ η^Fair = Fair(η) with singleton classes."""
        classes, pairs, eta, mu, _ = self._setup_coherent()
        verifier = CategoricalDiagramVerifier()
        result = verifier.verify_unit_compatibility(classes, pairs, eta)
        assert result.aggregate_commutes is True
        assert result.status == DiagramStatus.COMMUTES

    def test_unit_compatibility_fails_unsaturated(self):
        """Unit compatibility fails when η(B) is not a union of stutter classes.

        With non-singleton classes {s0,s1}, {s2,s3} and eta collapsing
        to representatives, η(B) = {s0} which is NOT a union of {s0,s1}.
        """
        classes = [
            _make_stutter_class("s0", frozenset({"s0", "s1"})),
            _make_stutter_class("s2", frozenset({"s2", "s3"})),
        ]
        pairs = [(frozenset({"s0", "s1"}), frozenset({"s2", "s3"}))]
        eta = {"s0": "s0", "s1": "s0", "s2": "s2", "s3": "s2"}
        verifier = CategoricalDiagramVerifier()
        result = verifier.verify_unit_compatibility(classes, pairs, eta)
        assert result.aggregate_commutes is False
        assert result.status == DiagramStatus.FAILS

    def test_verify_all_diagrams(self):
        """Full verify_all succeeds on coherent singleton-class setup."""
        classes, pairs, eta, mu, h = self._setup_coherent()
        verifier = CategoricalDiagramVerifier()
        all_ok, results = verifier.verify_all(
            classes, pairs, eta, mu, [("h1", h)]
        )
        assert all_ok is True
        assert len(results) == 3  # naturality, unit, multiplication

    def test_aggregate_result(self):
        """Aggregate result contains correct counts."""
        classes, pairs, eta, mu, h = self._setup_coherent()
        verifier = CategoricalDiagramVerifier()
        verifier.verify_all(classes, pairs, eta, mu, [("h1", h)])
        agg = verifier.aggregate_result()
        assert agg["total_diagrams"] == 3
        assert agg["commuting"] == 3
        assert agg["all_commute"] is True


# ══════════════════════════════════════════════════════════════════════════
# 3. TestCTLStarPreservation
# ══════════════════════════════════════════════════════════════════════════

class TestCTLStarPreservation:
    """Test CTL*\\X preservation structural induction proof."""

    def test_simple_formula_tree(self):
        """Structural induction on E(p U q): produces correct step count.

        Formula tree:  EXISTS_PATH
                          |
                        UNTIL
                       /     \\
                   ATOMIC(p) ATOMIC(q)

        Expected steps: 4 (p, q, until, exists_path) — depth-first post-order.
        """
        p = FormulaNode(kind=FormulaKind.ATOMIC, label="p", formula_id="f1")
        q = FormulaNode(kind=FormulaKind.ATOMIC, label="q", formula_id="f2")
        until = FormulaNode(kind=FormulaKind.UNTIL, label="p U q",
                            children=[p, q], formula_id="f3")
        epsi = FormulaNode(kind=FormulaKind.EXISTS_PATH, label="E(p U q)",
                           children=[until], formula_id="f4")

        proof = CTLStarPreservationProof()
        preserved, steps = proof.check_preservation(epsi)
        assert preserved is True
        assert len(steps) == 4
        assert all(s.status == InductionCaseStatus.VERIFIED for s in steps)

    def test_ag_formula(self):
        """AG(p) = A(G(p)) produces 3 steps, all verified."""
        p = FormulaNode(kind=FormulaKind.ATOMIC, label="p")
        g = FormulaNode(kind=FormulaKind.GLOBALLY, label="G(p)", children=[p])
        ag = FormulaNode(kind=FormulaKind.FORALL_PATH, label="A(G(p))",
                         children=[g])

        proof = CTLStarPreservationProof()
        preserved, steps = proof.check_preservation(ag)
        assert preserved is True
        assert len(steps) == 3

    def test_no_coherence_fails_exists_path(self):
        """Without coherence, E-path preservation fails."""
        p = FormulaNode(kind=FormulaKind.ATOMIC, label="p")
        ep = FormulaNode(kind=FormulaKind.EXISTS_PATH, label="E(p)",
                         children=[p])
        proof = CTLStarPreservationProof()
        preserved, steps = proof.check_preservation(ep, coherence_holds=False)
        assert preserved is False
        # The E-path step should fail
        exists_step = [s for s in steps if s.formula_kind == FormulaKind.EXISTS_PATH]
        assert len(exists_step) == 1
        assert exists_step[0].status == InductionCaseStatus.FAILED

    def test_streett_acceptance_preservation(self):
        """Streett acceptance preserved on mock accepting runs."""
        classes = [
            _make_stutter_class("s0", frozenset({"s0", "s1"})),
            _make_stutter_class("s2", frozenset({"s2", "s3"})),
        ]
        pairs = [
            (frozenset({"s0", "s1"}), frozenset({"s2", "s3"})),
        ]
        morphism = {"s0": "q0", "s1": "q0", "s2": "q1", "s3": "q1"}

        checker = StreettAcceptancePreservation()
        all_ok, results = checker.verify(pairs, morphism, classes,
                                         coherence_holds=True)
        assert all_ok is True
        assert len(results) == 1
        assert results[0].b_preserved is True
        assert results[0].g_preserved is True
        assert results[0].coherence_used is True

    def test_streett_no_coherence(self):
        """Without coherence, Streett acceptance cannot be guaranteed."""
        classes = [
            _make_stutter_class("s0", frozenset({"s0", "s1"})),
        ]
        pairs = [
            (frozenset({"s0", "s1"}), frozenset({"s0", "s1"})),
        ]
        morphism = {"s0": "q0", "s1": "q0"}

        checker = StreettAcceptancePreservation()
        all_ok, results = checker.verify(pairs, morphism, classes,
                                         coherence_holds=False)
        assert all_ok is False
        assert results[0].b_preserved is False

    def test_boolean_connective_preservation(self):
        """Conjunction and disjunction are preserved by induction."""
        p = FormulaNode(kind=FormulaKind.ATOMIC, label="p")
        q = FormulaNode(kind=FormulaKind.ATOMIC, label="q")
        conj = FormulaNode(kind=FormulaKind.CONJUNCTION, label="p ∧ q",
                           children=[p, q])
        proof = CTLStarPreservationProof()
        preserved, steps = proof.check_preservation(conj)
        assert preserved is True
        assert len(steps) == 3  # p, q, conjunction


# ══════════════════════════════════════════════════════════════════════════
# 4. TestConformanceSoundness
# ══════════════════════════════════════════════════════════════════════════

class TestConformanceSoundness:
    """Test conformance certificates and depth sufficiency."""

    def _make_chain_hypothesis(self, n: int):
        """Chain of n nodes: s0 -> s1 -> ... -> s_{n-1} with action 'a'.

        Diameter = n - 1.
        """
        states = [f"s{i}" for i in range(n)]
        transitions = {}
        for i in range(n - 1):
            transitions[(f"s{i}", "a")] = f"s{i+1}"
        # SimpleNamespace with callable states/actions/transition
        hyp = SimpleNamespace(
            states=lambda: states,
            actions=lambda: ["a"],
            transition=lambda s, a: transitions.get((s, a)),
        )
        return hyp

    def test_diameter_chain_5_nodes(self):
        """Chain of 5 nodes has diameter 4."""
        hyp = self._make_chain_hypothesis(5)
        builder = ConformanceCertificateBuilder()
        d = builder.compute_exact_diameter(hyp)
        assert d == 4

    def test_depth_sufficiency_formula(self):
        """Sufficient depth k = d + (m - n + 1).

        For n=5 hypothesis states, m=10 concrete states, d=4:
        k = 4 + (10 - 5 + 1) = 10.
        """
        hyp = self._make_chain_hypothesis(5)
        builder = ConformanceCertificateBuilder()
        cert = builder.build(
            hypothesis=hyp,
            actual_depth=10,
            total_tests=100,
            concrete_state_count=10,
            system_id="test-chain",
        )
        dp = cert.depth_proof
        assert dp.hypothesis_states == 5
        assert dp.hypothesis_diameter == 4
        assert dp.concrete_state_bound == 10
        assert dp.sufficient_depth == 10  # 4 + (10-5) + 1
        assert dp.is_sufficient is True
        assert cert.w_method_complete is True

    def test_depth_insufficient(self):
        """When actual_depth < sufficient_depth, w_method_complete is False."""
        hyp = self._make_chain_hypothesis(5)
        builder = ConformanceCertificateBuilder()
        cert = builder.build(
            hypothesis=hyp,
            actual_depth=7,
            total_tests=50,
            concrete_state_count=10,
            system_id="test-short",
        )
        assert cert.depth_proof.is_sufficient is False
        assert cert.w_method_complete is False

    def test_convergence_detection_stable_sizes(self):
        """Convergence detected when hypothesis sizes are stable for 3 depths."""
        hyp = self._make_chain_hypothesis(3)
        builder = ConformanceCertificateBuilder()
        cert = builder.build(
            hypothesis=hyp,
            actual_depth=10,
            total_tests=50,
            concrete_state_count=5,
            system_id="conv-test",
            convergence_history=[3, 5, 5, 5],
        )
        assert cert.convergence_detected is True
        assert cert.convergence_at_depth == 8  # actual_depth - 2

    def test_no_convergence_varying_sizes(self):
        """No convergence if hypothesis sizes keep changing."""
        hyp = self._make_chain_hypothesis(3)
        builder = ConformanceCertificateBuilder()
        cert = builder.build(
            hypothesis=hyp,
            actual_depth=10,
            total_tests=50,
            concrete_state_count=5,
            system_id="no-conv",
            convergence_history=[3, 4, 5, 6],
        )
        assert cert.convergence_detected is False

    def test_w_method_coverage_small_automaton(self):
        """Certificate generated for a small automaton with known parameters."""
        hyp = self._make_chain_hypothesis(3)
        builder = ConformanceCertificateBuilder()
        k = builder.suggest_depth(hyp, concrete_state_count=6)
        # d=2, m=6, n=3 => k = 2 + (6-3) + 1 = 6
        assert k == 6
        cert = builder.build(
            hypothesis=hyp,
            actual_depth=k,
            total_tests=100,
            concrete_state_count=6,
        )
        assert cert.w_method_complete is True

    def test_state_bound_derivation(self):
        """State bound derived from spec params."""
        builder = ConformanceCertificateBuilder()
        bound, derivation = builder.compute_state_bound({
            "flag": {"type": "boolean"},
            "counter": {"type": "bounded_int", "lo": 0, "hi": 3},
        })
        # 2 × 4 = 8
        assert bound == 8
        assert "flag" in derivation
        assert "counter" in derivation


# ══════════════════════════════════════════════════════════════════════════
# 5. TestBloomSoundness
# ══════════════════════════════════════════════════════════════════════════

class TestBloomSoundness:
    """Test Bloom filter soundness analysis (FPR formula, optimal params)."""

    def test_fpr_formula(self):
        """Verify FPR = (1 - e^{-kn/m})^k."""
        m, k, n = 10000, 7, 100
        expected = (1.0 - math.exp(-k * n / m)) ** k
        actual = BloomSoundnessAnalyzer.per_query_fpr(m, k, n)
        assert abs(actual - expected) < 1e-12

    def test_fpr_zero_elements(self):
        """FPR is 0 when no elements are inserted."""
        assert BloomSoundnessAnalyzer.per_query_fpr(1000, 7, 0) == 0.0

    def test_optimal_k(self):
        """Optimal k ≈ (m/n) · ln 2."""
        m, n = 10000, 1000
        expected = round((m / n) * math.log(2))
        actual = BloomSoundnessAnalyzer.optimal_k(m, n)
        assert actual == expected

    def test_soundness_monotonically_decreasing(self):
        """Soundness bound (false acceptance) decreases with increasing m/n."""
        analyzer = BloomSoundnessAnalyzer(target_soundness=0.999)
        n = 1000
        V = 100
        prev_fa = 2.0  # start above 1.0 for first comparison
        for bpe in [4, 8, 12, 16, 20, 24, 32]:
            m = bpe * n
            k = BloomSoundnessAnalyzer.optimal_k(m, n)
            bound = analyzer.analyze(m, k, n, V)
            assert bound.false_acceptance_exact < prev_fa, (
                f"FP acceptance not decreasing at bpe={bpe}: "
                f"{bound.false_acceptance_exact} >= {prev_fa}"
            )
            prev_fa = bound.false_acceptance_exact

    def test_adaptive_config_meets_target(self):
        """AdaptiveBloomConfig computes parameters meeting target soundness."""
        config = AdaptiveBloomConfig(target_soundness=0.9999)
        config.compute(n_elements=5000, verification_checks=200)
        # Verify the computed FPR with computed parameters meets the target
        fpr = BloomSoundnessAnalyzer.per_query_fpr(
            config.computed_bits, config.computed_hash_functions, 5000
        )
        fa_exact = 1.0 - (1.0 - fpr) ** 200
        soundness = 1.0 - fa_exact
        assert soundness >= 0.9999 - 1e-9

    def test_minimum_bits(self):
        """minimum_bits returns enough bits for the target FPR."""
        n = 1000
        target_fpr = 0.001
        m = BloomSoundnessAnalyzer.minimum_bits(n, target_fpr)
        k = BloomSoundnessAnalyzer.optimal_k(m, n)
        actual_fpr = BloomSoundnessAnalyzer.per_query_fpr(m, k, n)
        assert actual_fpr <= target_fpr

    def test_analyze_proof_sketch(self):
        """Analyze produces a non-empty proof sketch."""
        analyzer = BloomSoundnessAnalyzer(target_soundness=0.999)
        bound = analyzer.analyze(10000, 7, 1000, 50)
        assert len(bound.proof_sketch) > 100
        assert "THEOREM" in bound.proof_sketch

    def test_union_bound_vs_exact(self):
        """Union bound ≥ exact false acceptance probability."""
        analyzer = BloomSoundnessAnalyzer()
        bound = analyzer.analyze(10000, 7, 1000, 100)
        assert bound.false_acceptance_union_bound >= bound.false_acceptance_exact


# ══════════════════════════════════════════════════════════════════════════
# 6. TestProofObligationTracker
# ══════════════════════════════════════════════════════════════════════════

class TestProofObligationTracker:
    """Test unified proof obligation tracking."""

    def test_register_and_discharge(self):
        """Register obligations from multiple sources, discharge, verify."""
        tracker = ProofObligationTracker()
        tracker.register("coh-1", ObligationCategory.COHERENCE,
                         "T-Fair coherence pair 0",
                         source_module="tfair_theorem")
        tracker.register("cat-nat", ObligationCategory.CATEGORICAL_DIAGRAM,
                         "Naturality square",
                         depends_on=["coh-1"],
                         source_module="categorical_diagram")
        tracker.register("ctl-1", ObligationCategory.CTL_STAR,
                         "CTL* preservation",
                         depends_on=["coh-1"],
                         source_module="ctl_star_preservation")

        assert len(tracker.obligations) == 3
        assert len(tracker.pending()) == 3
        assert tracker.all_discharged() is False

        # Discharge coherence first
        assert tracker.discharge("coh-1", "Exhaustive saturation check") is True
        assert tracker.get("coh-1").status == DischargeStatus.DISCHARGED

        # Now dependents can be discharged
        assert tracker.discharge("cat-nat", "Diagram commutes") is True
        assert tracker.discharge("ctl-1", "Structural induction") is True
        assert tracker.all_discharged() is True

    def test_dependency_blocks_discharge(self):
        """Cannot discharge an obligation with unsatisfied dependencies."""
        tracker = ProofObligationTracker()
        tracker.register("dep", ObligationCategory.COHERENCE, "Base")
        tracker.register("child", ObligationCategory.PRESERVATION,
                         "Needs base", depends_on=["dep"])
        result = tracker.discharge("child", "attempt")
        assert result is False
        assert tracker.get("child").status == DischargeStatus.FAILED

    def test_aggregate_status(self):
        """Aggregate summary reflects correct counts."""
        tracker = ProofObligationTracker()
        tracker.register("a", ObligationCategory.COHERENCE, "A")
        tracker.register("b", ObligationCategory.MINIMALITY, "B")
        tracker.register("c", ObligationCategory.CONFORMANCE, "C")
        tracker.discharge("a", "ok")
        tracker.discharge("b", "ok")
        tracker.fail("c", "not enough depth")

        summary = tracker.summary()
        assert summary["total_obligations"] == 3
        assert summary["by_status"]["DISCHARGED"] == 2
        assert summary["by_status"]["FAILED"] == 1
        assert summary["all_discharged"] is False

    def test_dependency_chain_detection(self):
        """Dependency chain is computed as layers."""
        tracker = ProofObligationTracker()
        tracker.register("L0-a", ObligationCategory.COHERENCE, "Base A")
        tracker.register("L0-b", ObligationCategory.COHERENCE, "Base B")
        tracker.register("L1", ObligationCategory.PRESERVATION, "Layer 1",
                         depends_on=["L0-a", "L0-b"])
        tracker.register("L2", ObligationCategory.CTL_STAR, "Layer 2",
                         depends_on=["L1"])

        chain = tracker.dependency_chain()
        assert len(chain) == 3
        assert set(chain[0]) == {"L0-a", "L0-b"}
        assert chain[1] == ["L1"]
        assert chain[2] == ["L2"]

    def test_proof_hash_deterministic(self):
        """Proof hash is deterministic for same obligations and states."""
        def build():
            t = ProofObligationTracker()
            t.register("x", ObligationCategory.COHERENCE, "X")
            t.register("y", ObligationCategory.MINIMALITY, "Y")
            t.discharge("x", "witness-x")
            t.discharge("y", "witness-y")
            return t.compute_proof_hash()
        assert build() == build()
        assert len(build()) == 64

    def test_by_category(self):
        """Filter obligations by category."""
        tracker = ProofObligationTracker()
        tracker.register("c1", ObligationCategory.COHERENCE, "C1")
        tracker.register("c2", ObligationCategory.COHERENCE, "C2")
        tracker.register("m1", ObligationCategory.MINIMALITY, "M1")
        coh = tracker.by_category(ObligationCategory.COHERENCE)
        assert len(coh) == 2
        mini = tracker.by_category(ObligationCategory.MINIMALITY)
        assert len(mini) == 1


# ══════════════════════════════════════════════════════════════════════════
# 7. TestBaselineComparison
# ══════════════════════════════════════════════════════════════════════════

class TestBaselineComparison:
    """Test baseline comparison framework (Paige-Tarjan, Naive)."""

    def test_paige_tarjan_small_lts(self):
        """PaigeTarjan on a small LTS produces correct partition."""
        lts = _make_small_lts()
        pt = PaigeTarjanBaseline()
        partition = pt.compute(lts)
        # Should produce 2 blocks: {a,b} and {c,d}
        assert pt.num_blocks == 2
        blocks_as_frozensets = {frozenset(b) for b in partition}
        assert frozenset({"a", "b"}) in blocks_as_frozensets
        assert frozenset({"c", "d"}) in blocks_as_frozensets

    def test_naive_bisim_small_lts(self):
        """NaiveBisimulation on same LTS produces same partition."""
        lts = _make_small_lts()
        naive = NaiveBisimulation()
        partition = naive.compute(lts)
        assert len(partition) == 2
        blocks_as_frozensets = {frozenset(b) for b in partition}
        assert frozenset({"a", "b"}) in blocks_as_frozensets
        assert frozenset({"c", "d"}) in blocks_as_frozensets

    def test_both_algorithms_agree(self):
        """Both algorithms produce the same number of equivalence classes."""
        lts = _make_small_lts()
        pt = PaigeTarjanBaseline()
        pt.compute(lts)
        naive = NaiveBisimulation()
        naive.compute(lts)
        assert pt.num_blocks == len(naive._relation_to_partition(sorted(lts.states)))

    def test_singleton_lts(self):
        """Single-state LTS yields one equivalence class."""
        lts = LTS(
            states={"s"},
            transitions={},
            labels={"s": frozenset({"p"})},
            initial="s",
            actions=set(),
        )
        pt = PaigeTarjanBaseline()
        pt.compute(lts)
        assert pt.num_blocks == 1

    def test_all_distinct_labels(self):
        """States with distinct labels are never merged."""
        lts = LTS(
            states={"a", "b", "c"},
            transitions={
                ("a", "x"): {"b"},
                ("b", "x"): {"c"},
                ("c", "x"): {"a"},
            },
            labels={
                "a": frozenset({"p"}),
                "b": frozenset({"q"}),
                "c": frozenset({"r"}),
            },
            initial="a",
            actions={"x"},
        )
        pt = PaigeTarjanBaseline()
        pt.compute(lts)
        assert pt.num_blocks == 3

    def test_welch_t_test_computation(self):
        """Welch's t-test produces reasonable p-value and effect size."""
        st = StatisticalTest(alpha=0.05)
        sample_a = [1.0, 1.1, 0.9, 1.05, 0.95]
        sample_b = [2.0, 2.1, 1.9, 2.05, 1.95]
        result = st.welch_t_test(sample_a, sample_b, "time", "A", "B")
        # Means are clearly different
        assert result.p_value < 0.01
        assert result.significant is True
        assert abs(result.cohens_d) > 1.0  # large effect
        assert result.ci_lower < 0  # A < B => diff < 0
        assert result.n_a == 5
        assert result.n_b == 5

    def test_welch_t_test_same_samples(self):
        """T-test on identical samples should not be significant."""
        st = StatisticalTest(alpha=0.05)
        sample = [1.0, 1.1, 0.9, 1.05, 0.95]
        result = st.welch_t_test(sample, sample, "time", "A", "A")
        # When the samples are identical, variance may be identical
        # and p-value should be high (not significant)
        assert result.significant is False or result.p_value >= 0.05

    def test_comparison_report_blocks_match(self):
        """ComparisonReport.blocks_match works correctly."""
        report = ComparisonReport(spec_name="test")
        report.add_run(AlgorithmRun(algorithm="PT", spec_name="test", num_blocks=3))
        report.add_run(AlgorithmRun(algorithm="NB", spec_name="test", num_blocks=3))
        assert report.blocks_match() is True
        report.add_run(AlgorithmRun(algorithm="CC", spec_name="test", num_blocks=4))
        assert report.blocks_match() is False


# ══════════════════════════════════════════════════════════════════════════
# 8. TestMinimalityProof
# ══════════════════════════════════════════════════════════════════════════

class TestMinimalityProof:
    """Test minimality proofs for coalgebraic quotients."""

    def _make_minimal_coalgebra(self):
        """Coalgebra where the partition is already minimal.

        Two states with different AP labels and self-loops.
        Partition {{s0}, {s1}} is minimal: they are distinguished by AP.
        """
        structure = {
            "s0": SimpleNamespace(
                propositions=frozenset({"p"}),
                successors={"a": frozenset({"s0"})},
            ),
            "s1": SimpleNamespace(
                propositions=frozenset({"q"}),
                successors={"a": frozenset({"s1"})},
            ),
        }
        coalgebra = SimpleNamespace(
            states=["s0", "s1"],
            structure_map=structure,
        )
        partition = [frozenset({"s0"}), frozenset({"s1"})]
        morphism = {"s0": "q0", "s1": "q1"}
        return coalgebra, partition, morphism

    def _make_nonminimal_coalgebra(self):
        """Coalgebra with a non-minimal partition (mergeable classes).

        Three states: s0, s1, s2.  s0 and s1 have same AP and same
        successor class, so they could be merged.
        """
        structure = {
            "s0": SimpleNamespace(
                propositions=frozenset({"p"}),
                successors={"a": frozenset({"s2"})},
            ),
            "s1": SimpleNamespace(
                propositions=frozenset({"p"}),
                successors={"a": frozenset({"s2"})},
            ),
            "s2": SimpleNamespace(
                propositions=frozenset({"q"}),
                successors={"a": frozenset({"s2"})},
            ),
        }
        coalgebra = SimpleNamespace(
            states=["s0", "s1", "s2"],
            structure_map=structure,
        )
        # Non-minimal: s0 and s1 are in separate classes but can be merged
        partition = [frozenset({"s0"}), frozenset({"s1"}), frozenset({"s2"})]
        morphism = {"s0": "q0", "s1": "q1", "s2": "q2"}
        return coalgebra, partition, morphism

    def test_minimal_partition_succeeds(self):
        """Minimality proof succeeds for a truly minimal partition."""
        coalgebra, partition, morphism = self._make_minimal_coalgebra()
        prover = MinimalityProof()
        witness = prover.prove(partition, coalgebra, morphism)
        assert witness.partition_is_coarsest is True
        assert witness.morphism_is_valid is True
        assert witness.myhill_nerode.quotient_size == 2
        assert witness.myhill_nerode.original_size == 2
        assert witness.myhill_nerode.is_minimal is True
        assert len(witness.myhill_nerode.distinguishing_families) == 1

    def test_nonminimal_partition_detected(self):
        """Non-minimal partition (mergeable classes) is detected."""
        coalgebra, partition, morphism = self._make_nonminimal_coalgebra()
        prover = MinimalityProof()
        witness = prover.prove(partition, coalgebra, morphism)
        # s0 and s1 can be merged, so partition_is_coarsest should be False
        assert witness.partition_is_coarsest is False

    def test_morphism_consistency(self):
        """Morphism consistency: all states in a class map to same target."""
        coalgebra, _, _ = self._make_minimal_coalgebra()
        partition = [frozenset({"s0"}), frozenset({"s1"})]
        # Consistent morphism
        good_morph = {"s0": "q0", "s1": "q1"}
        prover1 = MinimalityProof()
        w1 = prover1.prove(partition, coalgebra, good_morph)
        assert w1.morphism_is_valid is True

        # Inconsistent morphism: same class maps to different targets
        partition2 = [frozenset({"s0", "s1"})]
        bad_morph = {"s0": "q0", "s1": "q1"}  # splits the class
        prover2 = MinimalityProof()
        w2 = prover2.prove(partition2, coalgebra, bad_morph)
        assert w2.morphism_is_valid is False

    def test_compression_ratio(self):
        """Compression ratio = quotient_size / original_size."""
        coalgebra = SimpleNamespace(
            states=["s0", "s1", "s2", "s3"],
            structure_map={},
        )
        partition = [frozenset({"s0", "s1"}), frozenset({"s2", "s3"})]
        morphism = {"s0": "q0", "s1": "q0", "s2": "q1", "s3": "q1"}
        prover = MinimalityProof()
        w = prover.prove(partition, coalgebra, morphism)
        assert w.compression_ratio == pytest.approx(0.5)
        assert w.myhill_nerode.quotient_size == 2
        assert w.myhill_nerode.original_size == 4

    def test_certificate_hash(self):
        """Certificate hash is non-empty and deterministic."""
        coalgebra, partition, morphism = self._make_minimal_coalgebra()
        prover1 = MinimalityProof()
        w1 = prover1.prove(partition, coalgebra, morphism)
        prover2 = MinimalityProof()
        w2 = prover2.prove(partition, coalgebra, morphism)
        assert len(w1.certificate_hash) == 64
        assert w1.certificate_hash == w2.certificate_hash
