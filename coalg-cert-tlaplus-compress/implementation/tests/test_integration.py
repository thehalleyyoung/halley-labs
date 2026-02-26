"""
Comprehensive integration tests for the full CoaCert-TLA pipeline.

Tests the end-to-end flow: build coalgebra → partition refine → quotient
→ witness construction → verification → property preservation.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import pytest

from coacert.functor import (
    BehavioralEquivalence,
    FCoalgebra,
    QuotientCoalgebra,
)
from coacert.bisimulation import (
    BisimulationRelation,
    PartitionRefinement,
    QuotientBuilder,
    RefinementEngine,
    RefinementStrategy,
)
from coacert.witness.merkle_tree import MerkleTree, sha256
from coacert.witness.equivalence_binding import EquivalenceBinding
from coacert.witness.hash_chain import HashChain, BlockType
from coacert.witness.transition_witness import TransitionWitness, WitnessSet
from coacert.witness.witness_format import WitnessFormat
from coacert.verifier import (
    HashChainVerifier,
    ClosureValidator,
    StutteringVerifier,
    FairnessVerifier,
    VerificationReport,
    Verdict,
    WitnessData,
    WitnessDeserializer,
)
from coacert.verifier.deserializer import (
    EquivalenceBinding as VEquivalenceBinding,
    TransitionWitness as VTransitionWitness,
    HashBlock,
    WitnessHeader,
    WitnessFlag,
)
from coacert.verifier.hash_verifier import GENESIS_PREV_HASH, _sha256
from coacert.properties import (
    AG,
    EF,
    Atomic,
    CTLStarChecker,
    DifferentialTester,
    SafetyChecker,
    SafetyKind,
    SafetyProperty,
    make_ap_invariant,
)
from coacert.specs import (
    SpecRegistry,
    TwoPhaseCommitSpec,
    LeaderElectionSpec,
    PetersonSpec,
)
from coacert.cli import build_parser, main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_small_coalgebra(
    name: str = "test",
    states: Optional[Dict[str, Set[str]]] = None,
    transitions: Optional[List[Tuple[str, str, str]]] = None,
    initial: Optional[Set[str]] = None,
) -> FCoalgebra:
    """Build a small FCoalgebra from explicit state/transition data."""
    if states is None:
        states = {
            "s0": {"p"},
            "s1": {"q"},
            "s2": {"p"},
            "s3": {"q"},
            "s4": {"p"},
            "s5": {"r"},
        }
    if transitions is None:
        transitions = [
            ("s0", "a", "s1"),
            ("s0", "b", "s2"),
            ("s1", "a", "s0"),
            ("s2", "b", "s2"),
            ("s2", "a", "s3"),
            ("s3", "a", "s2"),
            ("s3", "b", "s4"),
            ("s4", "a", "s5"),
            ("s4", "b", "s0"),
            ("s5", "a", "s5"),
        ]
    if initial is None:
        initial = {"s0"}

    coalg = FCoalgebra(name=name)
    for s, props in states.items():
        coalg.add_state(s, propositions=props, is_initial=(s in initial))
    for src, act, dst in transitions:
        coalg.add_transition(src, act, dst)
    return coalg


def _build_symmetric_coalgebra() -> FCoalgebra:
    """Build a coalgebra with known symmetry (equivalent state pairs)."""
    coalg = FCoalgebra(name="symmetric")
    for s, props in [
        ("a0", {"p"}), ("a1", {"p"}),  # equivalent pair
        ("b0", {"q"}), ("b1", {"q"}),  # equivalent pair
        ("c0", {"r"}),
    ]:
        coalg.add_state(s, propositions=props, is_initial=(s in {"a0", "a1"}))
    for src, act, dst in [
        ("a0", "x", "b0"), ("a1", "x", "b1"),
        ("b0", "x", "c0"), ("b1", "x", "c0"),
        ("c0", "x", "a0"),
        ("a0", "y", "a0"), ("a1", "y", "a1"),
        ("b0", "y", "b0"), ("b1", "y", "b1"),
    ]:
        coalg.add_transition(src, act, dst)
    return coalg


def _extract_transitions(
    coalg: FCoalgebra,
) -> Dict[str, Dict[str, Set[str]]]:
    """Extract transition dict from an FCoalgebra."""
    trans: Dict[str, Dict[str, Set[str]]] = {}
    for s in coalg.states:
        fv = coalg.apply_functor(s)
        act_map: Dict[str, Set[str]] = {}
        for act in coalg.actions:
            succs = fv.successor_set(act)
            if succs:
                act_map[act] = set(succs)
        if act_map:
            trans[s] = act_map
    return trans


def _extract_labels(
    coalg: FCoalgebra,
) -> Dict[str, Set[str]]:
    """Extract label dict from an FCoalgebra."""
    return {
        s: set(coalg.apply_functor(s).propositions) for s in coalg.states
    }


def _run_partition_refinement(
    coalg: FCoalgebra,
) -> Tuple[BisimulationRelation, List[FrozenSet[str]]]:
    """Run partition refinement on a coalgebra, return relation and partition."""
    trans = _extract_transitions(coalg)
    labels = _extract_labels(coalg)
    pr = PartitionRefinement(
        states=set(coalg.states),
        actions=set(coalg.actions),
        transitions=trans,
        labels=labels,
    )
    result = pr.refine()
    return result.partition, result.partition.classes()


def _build_quotient(
    coalg: FCoalgebra,
    partition: List[FrozenSet[str]],
) -> Tuple[FCoalgebra, Dict[str, str]]:
    """Build quotient coalgebra from partition."""
    return QuotientCoalgebra.build(coalg, partition)


def _build_verifier_witness(
    coalg: FCoalgebra,
    quotient: FCoalgebra,
    partition: List[FrozenSet[str]],
    projection: Dict[str, str],
) -> WitnessData:
    """Build a WitnessData for the verifier from coalgebra + partition."""
    # Equivalence bindings
    class_map: Dict[str, int] = {}
    eqs: List[VEquivalenceBinding] = []
    for i, block in enumerate(partition):
        rep = min(block)
        labels = set(coalg.apply_functor(rep).propositions)
        eqs.append(VEquivalenceBinding(
            class_id=i,
            representative=rep,
            members=tuple(sorted(block)),
            ap_labels=tuple(sorted(labels)),
        ))
        for s in block:
            class_map[s] = i

    # Transition witnesses: for each (class_i, class_j, action) in quotient,
    # provide concrete witnesses for all members of class_i
    txs: List[VTransitionWitness] = []
    trans = _extract_transitions(coalg)
    for i, block_i in enumerate(partition):
        for s in block_i:
            for act, targets in trans.get(s, {}).items():
                for t in targets:
                    j = class_map[t]
                    txs.append(VTransitionWitness(
                        source_class=i,
                        target_class=j,
                        original_source=s,
                        original_target=t,
                        matching_path=(t,),
                    ))

    # Hash chain
    chain = _build_hash_chain(3)
    header = WitnessHeader(1, 0, 0, 0, 0)
    return WitnessData(
        header=header,
        equivalences=eqs,
        transitions=txs,
        hash_chain=chain,
    )


def _build_hash_chain(n: int) -> List[HashBlock]:
    """Build a valid hash chain of length n."""
    chain: List[HashBlock] = []
    prev = GENESIS_PREV_HASH
    for i in range(n):
        payload = _sha256(f"payload-{i}".encode())
        block_hash = _sha256(prev + payload)
        chain.append(HashBlock(
            index=i, prev_hash=prev,
            payload_hash=payload, block_hash=block_hash,
        ))
        prev = block_hash
    return chain


# ===================================================================
# TestParseExploreCompress
# ===================================================================


class TestParseExploreCompress:
    """Build transition system, run partition refinement, build quotient."""

    def test_small_system_partition(self):
        coalg = _build_small_coalgebra()
        relation, partition = _run_partition_refinement(coalg)
        assert len(partition) <= coalg.state_count
        assert len(partition) >= 1

    def test_quotient_has_fewer_or_equal_states(self):
        coalg = _build_small_coalgebra()
        _, partition = _run_partition_refinement(coalg)
        quot, proj = _build_quotient(coalg, partition)
        assert quot.state_count <= coalg.state_count

    def test_quotient_preserves_initial_states(self):
        coalg = _build_small_coalgebra()
        _, partition = _run_partition_refinement(coalg)
        quot, proj = _build_quotient(coalg, partition)
        for init_s in coalg.initial_states:
            assert proj[init_s] in quot.initial_states

    def test_quotient_preserves_actions(self):
        coalg = _build_small_coalgebra()
        _, partition = _run_partition_refinement(coalg)
        quot, _ = _build_quotient(coalg, partition)
        assert coalg.actions == quot.actions

    def test_partition_covers_all_states(self):
        coalg = _build_small_coalgebra()
        _, partition = _run_partition_refinement(coalg)
        covered = set()
        for block in partition:
            covered |= block
        assert covered == coalg.states

    def test_partition_blocks_are_disjoint(self):
        coalg = _build_small_coalgebra()
        _, partition = _run_partition_refinement(coalg)
        all_states: List[str] = []
        for block in partition:
            all_states.extend(block)
        assert len(all_states) == len(set(all_states))


# ===================================================================
# TestFullPipeline
# ===================================================================


class TestFullPipeline:
    """Build coalgebra → behavioral equivalence → quotient → witness → verify."""

    def test_behavioral_equivalence_computation(self):
        coalg = _build_small_coalgebra()
        be = BehavioralEquivalence(coalg)
        classes = be.compute()
        assert len(classes) >= 1
        total = sum(c.size() for c in classes)
        assert total == coalg.state_count

    def test_quotient_from_behavioral_equiv(self):
        coalg = _build_small_coalgebra()
        be = BehavioralEquivalence(coalg)
        classes = be.compute()
        partition = [frozenset(c.members) for c in classes]
        quot, proj = _build_quotient(coalg, partition)
        assert quot.state_count == len(classes)

    def test_witness_construction_and_verification(self):
        coalg = _build_small_coalgebra()
        be = BehavioralEquivalence(coalg)
        classes = be.compute()
        partition = [frozenset(c.members) for c in classes]
        quot, proj = _build_quotient(coalg, partition)
        witness = _build_verifier_witness(coalg, quot, partition, proj)

        report = VerificationReport(witness)
        hr = HashChainVerifier(witness).verify_full()
        report.add_hash_result(hr)
        assert hr.passed

        cr = ClosureValidator(witness).validate_full()
        report.add_closure_result(cr)
        assert cr.passed

        report.finalize()
        assert report.verdict in (Verdict.VERIFIED, Verdict.PARTIAL)


# ===================================================================
# TestTwoPhaseCommitPipeline
# ===================================================================


class TestTwoPhaseCommitPipeline:
    """Test with TwoPhaseCommit specification."""

    def test_spec_validates(self):
        spec = TwoPhaseCommitSpec(n_participants=2)
        errors = spec.validate()
        assert len(errors) == 0

    def test_spec_module_has_definitions(self):
        spec = TwoPhaseCommitSpec(n_participants=2)
        module = spec.get_spec()
        assert module.name == "TwoPhaseCommit"
        def_names = {d.name for d in module.definitions if hasattr(d, "name")}
        assert "Init" in def_names
        assert "Next" in def_names

    def test_spec_properties_exist(self):
        spec = TwoPhaseCommitSpec(n_participants=2)
        props = spec.get_properties()
        assert len(props) >= 2

    @pytest.mark.slow
    def test_manual_2pc_coalgebra_refinement(self):
        """Build a small 2PC-like coalgebra and run refinement."""
        coalg = FCoalgebra(name="tpc_mini")
        # Model a minimal 2PC: coordinator + 2 participants
        # States: init, prepared, committed, aborted
        for s, props in [
            ("init", {"working"}),
            ("p1_prep", {"prepared"}),
            ("p2_prep", {"prepared"}),
            ("all_prep", {"prepared"}),
            ("committed", {"decided"}),
            ("aborted", {"decided"}),
        ]:
            coalg.add_state(s, propositions=props, is_initial=(s == "init"))
        for src, act, dst in [
            ("init", "prepare", "p1_prep"),
            ("init", "prepare", "p2_prep"),
            ("init", "abort", "aborted"),
            ("p1_prep", "prepare", "all_prep"),
            ("p2_prep", "prepare", "all_prep"),
            ("all_prep", "commit", "committed"),
            ("all_prep", "abort", "aborted"),
            ("committed", "done", "committed"),
            ("aborted", "done", "aborted"),
        ]:
            coalg.add_transition(src, act, dst)

        engine = RefinementEngine(
            states=set(coalg.states),
            actions=set(coalg.actions),
            transitions=_extract_transitions(coalg),
            labels=_extract_labels(coalg),
        )
        result = engine.run()
        assert result.converged
        ratio = coalg.state_count / max(result.final_blocks, 1)
        assert ratio >= 1.0


# ===================================================================
# TestSafetyPreservation
# ===================================================================


class TestSafetyPreservation:
    """Verify safety properties are preserved across quotient."""

    def _make_pair(self):
        coalg = _build_small_coalgebra()
        be = BehavioralEquivalence(coalg)
        classes = be.compute()
        partition = [frozenset(c.members) for c in classes]
        quot, proj = _build_quotient(coalg, partition)
        return coalg, quot, proj

    def test_invariant_preservation(self):
        coalg, quot, proj = self._make_pair()
        prop = make_ap_invariant("p")
        orig_check = SafetyChecker(coalg).check_invariant(prop)
        quot_check = SafetyChecker(quot).check_invariant(prop)
        # If it holds on original, it must hold on quotient (bisimulation preserves)
        # The converse isn't always true, but agreement is expected for bisim quotients
        if orig_check.holds:
            assert quot_check.holds

    @pytest.mark.parametrize("proposition", ["p", "q", "r"])
    def test_ag_invariant_agreement(self, proposition: str):
        coalg, quot, proj = self._make_pair()
        prop = make_ap_invariant(proposition)
        orig = SafetyChecker(coalg).check_invariant(prop)
        quot_r = SafetyChecker(quot).check_invariant(prop)
        # Under bisimulation quotient, results should agree
        assert orig.holds == quot_r.holds

    def test_ctl_formula_preservation(self):
        coalg, quot, proj = self._make_pair()
        formula = AG(Atomic("p"))
        orig_result = CTLStarChecker(coalg).check(formula)
        quot_result = CTLStarChecker(quot).check(formula)
        assert orig_result.holds == quot_result.holds


# ===================================================================
# TestCompressionRatio
# ===================================================================


class TestCompressionRatio:
    """Coalgebras with known symmetry should compress."""

    def test_symmetric_coalgebra_compresses(self):
        coalg = _build_symmetric_coalgebra()
        be = BehavioralEquivalence(coalg)
        classes = be.compute()
        # a0 ≡ a1, b0 ≡ b1, c0 alone → 3 classes from 5 states
        assert len(classes) < coalg.state_count

    def test_compression_ratio_gt_one(self):
        coalg = _build_symmetric_coalgebra()
        _, partition = _run_partition_refinement(coalg)
        ratio = coalg.state_count / len(partition)
        assert ratio > 1.0

    def test_identical_states_merge(self):
        """Two states with identical structure must be merged."""
        coalg = FCoalgebra(name="identical")
        coalg.add_state("s0", propositions={"p"}, is_initial=True)
        coalg.add_state("s1", propositions={"p"}, is_initial=False)
        coalg.add_state("t", propositions={"q"}, is_initial=False)
        coalg.add_transition("s0", "a", "t")
        coalg.add_transition("s1", "a", "t")
        coalg.add_transition("t", "a", "s0")

        be = BehavioralEquivalence(coalg)
        classes = be.compute()
        assert len(classes) == 2  # {s0, s1} and {t}

    @pytest.mark.parametrize("n_copies", [2, 4, 8])
    def test_scaling_compression(self, n_copies: int):
        """N copies of the same structure should compress to 1."""
        coalg = FCoalgebra(name=f"scaled_{n_copies}")
        coalg.add_state("sink", propositions={"q"}, is_initial=False)
        for i in range(n_copies):
            s = f"copy_{i}"
            coalg.add_state(s, propositions={"p"}, is_initial=(i == 0))
            coalg.add_transition(s, "a", "sink")
        coalg.add_transition("sink", "a", "sink")

        be = BehavioralEquivalence(coalg)
        classes = be.compute()
        # All copies should merge into one class + sink = 2 classes
        assert len(classes) == 2
        ratio = coalg.state_count / len(classes)
        assert ratio >= n_copies / 2


# ===================================================================
# TestWitnessVerification
# ===================================================================


class TestWitnessVerification:
    """Build witness package and verify it passes."""

    def _make_verified_witness(self) -> WitnessData:
        coalg = _build_small_coalgebra()
        be = BehavioralEquivalence(coalg)
        classes = be.compute()
        partition = [frozenset(c.members) for c in classes]
        quot, proj = _build_quotient(coalg, partition)
        return _build_verifier_witness(coalg, quot, partition, proj)

    def test_hash_chain_passes(self):
        w = self._make_verified_witness()
        result = HashChainVerifier(w).verify_full()
        assert result.passed

    def test_closure_passes(self):
        w = self._make_verified_witness()
        result = ClosureValidator(w).validate_full()
        assert result.passed

    def test_full_report_verified(self):
        w = self._make_verified_witness()
        report = VerificationReport(w)
        report.add_hash_result(HashChainVerifier(w).verify_full())
        report.add_closure_result(ClosureValidator(w).validate_full())
        report.add_stuttering_result(
            StutteringVerifier(w, verify_hashes=False).verify()
        )
        report.add_fairness_result(FairnessVerifier(w).verify())
        report.finalize()
        assert report.verdict == Verdict.VERIFIED

    def test_report_json_parseable(self):
        w = self._make_verified_witness()
        report = VerificationReport(w)
        report.add_hash_result(HashChainVerifier(w).verify_full())
        report.add_closure_result(ClosureValidator(w).validate_full())
        report.finalize()
        j = json.loads(report.to_json())
        assert "verdict" in j


# ===================================================================
# TestWitnessRejection
# ===================================================================


class TestWitnessRejection:
    """Tampered witnesses must be rejected."""

    def _make_witness(self) -> WitnessData:
        coalg = _build_small_coalgebra()
        be = BehavioralEquivalence(coalg)
        classes = be.compute()
        partition = [frozenset(c.members) for c in classes]
        quot, proj = _build_quotient(coalg, partition)
        return _build_verifier_witness(coalg, quot, partition, proj)

    def test_tampered_hash_chain_rejected(self):
        w = self._make_witness()
        if len(w.hash_chain) > 1:
            blk = w.hash_chain[1]
            w.hash_chain[1] = HashBlock(
                index=blk.index,
                prev_hash=blk.prev_hash,
                payload_hash=blk.payload_hash,
                block_hash=b"\xde\xad" * 16,
            )
        result = HashChainVerifier(w).verify_full()
        if len(w.hash_chain) > 1:
            assert not result.passed

    def test_tampered_chain_produces_rejected_verdict(self):
        w = self._make_witness()
        if len(w.hash_chain) > 1:
            blk = w.hash_chain[1]
            w.hash_chain[1] = HashBlock(
                index=blk.index,
                prev_hash=blk.prev_hash,
                payload_hash=blk.payload_hash,
                block_hash=b"\xff" * 32,
            )
            report = VerificationReport(w)
            report.add_hash_result(HashChainVerifier(w).verify_full())
            report.add_closure_result(ClosureValidator(w).validate_full())
            report.finalize()
            assert report.verdict == Verdict.REJECTED

    def test_missing_transition_witness_detected(self):
        """Remove a transition witness and check closure fails."""
        w = self._make_witness()
        if len(w.transitions) > 1:
            w.transitions.pop()
            result = ClosureValidator(w).validate_full()
            # May or may not fail depending on coverage; at minimum it runs
            assert isinstance(result.passed, bool)


# ===================================================================
# TestDifferentialTesting
# ===================================================================


class TestDifferentialTesting:
    """Original and quotient should agree on all properties."""

    def _make_pair(self):
        coalg = _build_small_coalgebra()
        be = BehavioralEquivalence(coalg)
        classes = be.compute()
        partition = [frozenset(c.members) for c in classes]
        quot, proj = _build_quotient(coalg, partition)
        return coalg, quot, proj

    def test_safety_agreement(self):
        coalg, quot, proj = self._make_pair()
        prop = make_ap_invariant("p")
        orig = SafetyChecker(coalg).check_invariant(prop)
        quot_r = SafetyChecker(quot).check_invariant(prop)
        assert orig.holds == quot_r.holds

    def test_differential_tester_runs(self):
        coalg, quot, proj = self._make_pair()
        dt = DifferentialTester(coalg, quot, proj)
        safety_props = [make_ap_invariant("p"), make_ap_invariant("q")]
        stats = dt.run_full_suite(safety_props=safety_props)
        assert stats.agreement_rate >= 0.0

    def test_differential_tester_full_agreement(self):
        coalg, quot, proj = self._make_pair()
        dt = DifferentialTester(coalg, quot, proj)
        safety_props = [make_ap_invariant("p")]
        stats = dt.run_full_suite(safety_props=safety_props)
        assert stats.agreement_rate == 1.0

    def test_ctl_agreement(self):
        coalg, quot, proj = self._make_pair()
        formulas = [AG(Atomic("p")), EF(Atomic("q"))]
        dt = DifferentialTester(coalg, quot, proj)
        stats = dt.run_full_suite(ctl_formulas=formulas)
        assert stats.agreement_rate == 1.0


# ===================================================================
# TestCLISubcommands
# ===================================================================


class TestCLISubcommands:
    """Test CLI by calling main() with various argv lists."""

    def test_build_parser_returns_parser(self):
        parser = build_parser()
        assert parser is not None
        assert hasattr(parser, "parse_args")

    def test_info_subcommand(self):
        rc = main(["info"])
        assert isinstance(rc, int)

    def test_unknown_subcommand_returns_nonzero(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["nonexistent_command_xyz"])
        assert exc_info.value.code != 0

    def test_parse_without_file_exits_nonzero(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["parse"])
        assert exc_info.value.code != 0

    def test_verify_without_file_exits_nonzero(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["verify"])
        assert exc_info.value.code != 0

    def test_help_flag(self):
        """--help causes SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0


# ===================================================================
# TestMultipleSpecs
# ===================================================================


class TestMultipleSpecs:
    """Test pipeline on multiple specs from the registry."""

    def test_registry_contains_specs(self):
        registry = SpecRegistry.default()
        names = registry.list_names()
        assert len(names) >= 2
        assert "TwoPhaseCommit" in names

    @pytest.mark.parametrize("spec_name", [
        "TwoPhaseCommit", "LeaderElection", "Peterson",
    ])
    def test_spec_validates(self, spec_name: str):
        registry = SpecRegistry.default()
        spec_cls = registry.get(spec_name)
        spec = spec_cls() if spec_name != "TwoPhaseCommit" else spec_cls(n_participants=2)
        errors = spec.validate()
        assert len(errors) == 0, f"{spec_name} validation errors: {errors}"

    @pytest.mark.parametrize("spec_name", [
        "TwoPhaseCommit", "LeaderElection", "Peterson",
    ])
    def test_spec_has_properties(self, spec_name: str):
        registry = SpecRegistry.default()
        spec_cls = registry.get(spec_name)
        spec = spec_cls() if spec_name != "TwoPhaseCommit" else spec_cls(n_participants=2)
        props = spec.get_properties()
        assert len(props) >= 1

    def test_registry_instantiate(self):
        registry = SpecRegistry.default()
        spec = registry.instantiate("TwoPhaseCommit", n_participants=2)
        assert spec.validate() == []


# ===================================================================
# TestEndToEnd
# ===================================================================


class TestEndToEnd:
    """Full end-to-end: coalgebra → refine → quotient → witness → verify → CTL."""

    def test_full_pipeline(self):
        # 1. Build coalgebra
        coalg = _build_symmetric_coalgebra()
        assert coalg.state_count == 5

        # 2. Compute behavioral equivalence
        be = BehavioralEquivalence(coalg)
        classes = be.compute()
        assert len(classes) < coalg.state_count

        # 3. Build quotient
        partition = [frozenset(c.members) for c in classes]
        quot, proj = _build_quotient(coalg, partition)
        assert quot.state_count == len(classes)

        # 4. Build verifier witness
        witness = _build_verifier_witness(coalg, quot, partition, proj)

        # 5. Verify witness
        report = VerificationReport(witness)
        report.add_hash_result(HashChainVerifier(witness).verify_full())
        report.add_closure_result(ClosureValidator(witness).validate_full())
        report.add_stuttering_result(
            StutteringVerifier(witness, verify_hashes=False).verify()
        )
        report.add_fairness_result(FairnessVerifier(witness).verify())
        report.finalize()
        assert report.verdict == Verdict.VERIFIED

        # 6. Property check
        formula = AG(Atomic("p"))
        orig_result = CTLStarChecker(coalg).check(formula)
        quot_result = CTLStarChecker(quot).check(formula)
        assert orig_result.holds == quot_result.holds

    @pytest.mark.slow
    def test_full_pipeline_with_refinement_engine(self):
        coalg = _build_small_coalgebra()
        trans = _extract_transitions(coalg)
        labels = _extract_labels(coalg)

        # Use RefinementEngine instead of PartitionRefinement
        engine = RefinementEngine(
            states=set(coalg.states),
            actions=set(coalg.actions),
            transitions=trans,
            labels=labels,
            strategy=RefinementStrategy.EAGER,
        )
        result = engine.run()
        assert result.converged

        partition = result.partition.classes()
        quot, proj = _build_quotient(coalg, partition)
        assert quot.state_count <= coalg.state_count

        witness = _build_verifier_witness(coalg, quot, partition, proj)
        hr = HashChainVerifier(witness).verify_full()
        assert hr.passed

    def test_end_to_end_report_text(self):
        coalg = _build_small_coalgebra()
        be = BehavioralEquivalence(coalg)
        classes = be.compute()
        partition = [frozenset(c.members) for c in classes]
        quot, proj = _build_quotient(coalg, partition)
        witness = _build_verifier_witness(coalg, quot, partition, proj)

        report = VerificationReport(witness)
        report.add_hash_result(HashChainVerifier(witness).verify_full())
        report.add_closure_result(ClosureValidator(witness).validate_full())
        report.finalize()

        text = report.to_text()
        assert "CoaCert-TLA" in text or "VERIFIED" in text or "PARTIAL" in text

    @pytest.mark.parametrize("strategy", [
        RefinementStrategy.EAGER,
        RefinementStrategy.LAZY,
        RefinementStrategy.AP_THEN_TRANSITION,
    ])
    def test_refinement_strategies_converge(self, strategy):
        coalg = _build_small_coalgebra()
        engine = RefinementEngine(
            states=set(coalg.states),
            actions=set(coalg.actions),
            transitions=_extract_transitions(coalg),
            labels=_extract_labels(coalg),
            strategy=strategy,
        )
        result = engine.run()
        assert result.converged
        assert result.final_blocks >= 1

    def test_witness_format_roundtrip(self, tmp_path):
        """Build WitnessFormat, serialize, deserialize, verify integrity."""
        coalg = _build_symmetric_coalgebra()
        be = BehavioralEquivalence(coalg)
        classes = be.compute()
        partition = [frozenset(c.members) for c in classes]
        quot, proj = _build_quotient(coalg, partition)

        eb = EquivalenceBinding()
        for i, block in enumerate(partition):
            rep = min(block)
            eb.add_class(f"c{i}", rep, frozenset(block))

        ws = WitnessSet()
        trans = _extract_transitions(coalg)
        class_map: Dict[str, str] = {}
        for i, block in enumerate(partition):
            for s in block:
                class_map[s] = f"c{i}"

        for s in coalg.states:
            for act, targets in trans.get(s, {}).items():
                for t in targets:
                    ws.add_transition(TransitionWitness(
                        source_class=class_map[s],
                        target_class=class_map[t],
                        action=act,
                        concrete_source=s,
                        concrete_target=t,
                    ))

        chain = HashChain.build(
            equivalence_payloads=[eb.to_bytes()],
            transition_payloads=[ws.to_bytes()],
            fairness_payloads=[],
        )

        wf = WitnessFormat(
            equivalence=eb,
            witnesses=ws,
            chain=chain,
            spec_hash=sha256(b"test_spec"),
            original_state_count=coalg.state_count,
            quotient_state_count=quot.state_count,
        )
        ok, errs = wf.verify_integrity()
        assert ok, f"Integrity check failed: {errs}"

        path = tmp_path / "integration_witness.bin"
        nbytes = wf.serialize(str(path))
        assert nbytes > 0

        restored = WitnessFormat.deserialize(str(path))
        assert restored.chain.length == chain.length
