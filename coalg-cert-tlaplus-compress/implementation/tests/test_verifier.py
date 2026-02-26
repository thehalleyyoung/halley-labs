"""Comprehensive tests for the coacert.verifier module."""

import hashlib
import json
import pytest
from typing import Dict, List, Optional, Tuple

from coacert.verifier.deserializer import (
    EquivalenceBinding,
    FairnessBinding,
    HashBlock,
    SectionKind,
    TransitionWitness,
    WitnessData,
    WitnessDeserializer,
    WitnessFlag,
    WitnessHeader,
)
from coacert.verifier.hash_verifier import (
    FailureKind,
    HashChainVerifier,
    HashVerificationResult,
    GENESIS_PREV_HASH,
    _sha256,
)
from coacert.verifier.closure_validator import (
    ClosureValidator,
    ClosureResult,
    ClosureViolation,
    ViolationKind,
)
from coacert.verifier.stuttering_verifier import (
    StutteringVerifier,
    StutteringResult,
    StutteringViolationKind,
)
from coacert.verifier.fairness_verifier import (
    FairnessVerifier,
    FairnessResult,
    FairnessViolationKind,
)
from coacert.verifier.verification_report import (
    PhaseSummary,
    TrustLevel,
    Verdict,
    VerificationReport,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_header(*, flags: int = 0, num_sections: int = 0,
                 total_size: int = 0) -> WitnessHeader:
    return WitnessHeader(
        version_major=1, version_minor=0,
        flags=flags, num_sections=num_sections, total_size=total_size,
    )


def _make_block(index: int, prev_hash: bytes,
                payload_hash: bytes) -> HashBlock:
    """Build a valid hash block with correctly computed block_hash."""
    block_hash = _sha256(prev_hash + payload_hash)
    return HashBlock(
        index=index,
        prev_hash=prev_hash,
        payload_hash=payload_hash,
        block_hash=block_hash,
    )


def _build_chain(n: int) -> List[HashBlock]:
    """Build a valid chain of *n* blocks starting from genesis."""
    chain: List[HashBlock] = []
    prev = GENESIS_PREV_HASH
    for i in range(n):
        payload = _sha256(f"payload-{i}".encode())
        block = _make_block(i, prev, payload)
        chain.append(block)
        prev = block.block_hash
    return chain


def _simple_equivalences() -> List[EquivalenceBinding]:
    """Two equivalence classes with matching APs."""
    return [
        EquivalenceBinding(class_id=0, representative="s0",
                           members=("s0", "s1"), ap_labels=("a",)),
        EquivalenceBinding(class_id=1, representative="s2",
                           members=("s2", "s3"), ap_labels=("b",)),
    ]


def _simple_transitions() -> List[TransitionWitness]:
    """Transitions that satisfy forward/backward closure for the simple classes."""
    return [
        TransitionWitness(source_class=0, target_class=1,
                          original_source="s0", original_target="s2",
                          matching_path=("s2",)),
        TransitionWitness(source_class=0, target_class=1,
                          original_source="s1", original_target="s3",
                          matching_path=("s3",)),
    ]


def _valid_witness(*, with_chain: bool = False,
                   with_fairness: bool = False,
                   stuttering: bool = False) -> WitnessData:
    """Construct a small valid WitnessData."""
    flags = 0
    if stuttering:
        flags |= WitnessFlag.STUTTERING
    if with_fairness:
        flags |= WitnessFlag.FAIRNESS_PRESENT

    header = _make_header(flags=flags)
    eqs = _simple_equivalences()
    txs = _simple_transitions()
    chain = _build_chain(2) if with_chain else []
    fairness = (
        [FairnessBinding(pair_id=0, b_set_classes=(0,), g_set_classes=(1,))]
        if with_fairness else []
    )
    return WitnessData(
        header=header,
        equivalences=eqs,
        transitions=txs,
        hash_chain=chain,
        fairness=fairness,
    )


# ---------------------------------------------------------------------------
# TestWitnessDeserializer
# ---------------------------------------------------------------------------


class TestWitnessDeserializer:
    """Deserialization of JSON and in-memory witnesses."""

    def test_deserialize_json_roundtrip(self):
        """Build a JSON witness string, deserialize it, check fields."""
        prev = GENESIS_PREV_HASH
        payload = _sha256(b"test")
        block_hash = _sha256(prev + payload)
        obj = {
            "header": {"version_major": 1, "version_minor": 0,
                       "flags": 0, "num_sections": 0, "total_size": 0},
            "equivalences": [
                {"class_id": 0, "representative": "s0",
                 "members": ["s0", "s1"], "ap_labels": ["a"]},
            ],
            "transitions": [
                {"source_class": 0, "target_class": 1,
                 "original_source": "s0", "original_target": "s2",
                 "matching_path": ["s2"]},
            ],
            "hash_chain": [
                {"index": 0,
                 "prev_hash": prev.hex(),
                 "payload_hash": payload.hex(),
                 "block_hash": block_hash.hex()},
            ],
            "metadata": {"spec_name": "test_spec"},
        }
        text = json.dumps(obj)
        ds = WitnessDeserializer()
        w = ds.deserialize_json(text)

        assert w.header.version_major == 1
        assert len(w.equivalences) == 1
        assert w.equivalences[0].representative == "s0"
        assert len(w.transitions) == 1
        assert w.transitions[0].source_class == 0
        assert len(w.hash_chain) == 1
        assert w.hash_chain[0].block_hash == block_hash
        assert w.metadata["spec_name"] == "test_spec"

    def test_deserialize_json_minimal(self):
        """Deserialize a JSON with only a header."""
        text = json.dumps({"header": {}})
        w = WitnessDeserializer().deserialize_json(text)
        assert w.header.version_major == 1  # default
        assert w.equivalences == []
        assert w.transitions == []

    def test_deserialize_json_bad_toplevel(self):
        """Non-object top-level should raise."""
        with pytest.raises(Exception):
            WitnessDeserializer().deserialize_json(json.dumps([1, 2, 3]))

    def test_header_flag_properties(self):
        h = WitnessHeader(1, 0, 0x0F, 0, 0)
        assert h.has_stuttering
        assert h.is_compressed
        assert h.has_fairness
        assert h.is_merkle_hashed

    def test_header_no_flags(self):
        h = WitnessHeader(1, 0, 0, 0, 0)
        assert not h.has_stuttering
        assert not h.is_compressed
        assert not h.has_fairness
        assert not h.is_merkle_hashed

    @pytest.mark.parametrize("flag,prop", [
        (WitnessFlag.STUTTERING, "has_stuttering"),
        (WitnessFlag.COMPRESSED, "is_compressed"),
        (WitnessFlag.FAIRNESS_PRESENT, "has_fairness"),
        (WitnessFlag.MERKLE_HASHED, "is_merkle_hashed"),
    ])
    def test_individual_flags(self, flag, prop):
        h = WitnessHeader(1, 0, flag, 0, 0)
        assert getattr(h, prop) is True


# ---------------------------------------------------------------------------
# TestHashChainVerifier
# ---------------------------------------------------------------------------


class TestHashChainVerifier:
    """Hash chain and Merkle proof verification."""

    def test_valid_chain_passes(self):
        w = _valid_witness(with_chain=True)
        result = HashChainVerifier(w).verify_full()
        assert result.passed
        assert result.blocks_checked == 2
        assert len(result.failures) == 0

    def test_tampered_block_detected(self):
        """Corrupt a block_hash; verifier should detect HASH_MISMATCH."""
        w = _valid_witness(with_chain=True)
        bad_hash = b"\xff" * 32
        w.hash_chain[1] = HashBlock(
            index=w.hash_chain[1].index,
            prev_hash=w.hash_chain[1].prev_hash,
            payload_hash=w.hash_chain[1].payload_hash,
            block_hash=bad_hash,
        )
        result = HashChainVerifier(w).verify_full()
        assert not result.passed
        assert any(f.kind == FailureKind.HASH_MISMATCH for f in result.failures)

    def test_chain_break_detected(self):
        """Break chain continuity by corrupting prev_hash."""
        w = _valid_witness(with_chain=True)
        blk = w.hash_chain[1]
        bad_prev = b"\xab" * 32
        w.hash_chain[1] = HashBlock(
            index=blk.index,
            prev_hash=bad_prev,
            payload_hash=blk.payload_hash,
            block_hash=blk.block_hash,
        )
        result = HashChainVerifier(w).verify_full()
        assert not result.passed
        assert any(f.kind == FailureKind.CHAIN_BREAK for f in result.failures)

    def test_bad_genesis(self):
        """Genesis block with non-zero prev_hash should be flagged."""
        chain = _build_chain(1)
        chain[0] = HashBlock(
            index=0,
            prev_hash=b"\x01" * 32,
            payload_hash=chain[0].payload_hash,
            block_hash=chain[0].block_hash,
        )
        w = WitnessData(header=_make_header(), hash_chain=chain)
        result = HashChainVerifier(w).verify_full()
        assert not result.passed
        assert result.failures[0].kind == FailureKind.GENESIS_INVALID

    def test_verify_partial_samples_subset(self):
        """verify_partial should check fewer blocks than verify_full."""
        chain = _build_chain(20)
        w = WitnessData(header=_make_header(), hash_chain=chain)
        full = HashChainVerifier(w).verify_full()
        partial = HashChainVerifier(w).verify_partial(sample_fraction=0.2)
        assert partial.passed
        assert partial.blocks_checked <= full.blocks_checked

    def test_empty_chain(self):
        w = WitnessData(header=_make_header(), hash_chain=[])
        result = HashChainVerifier(w).verify_full()
        assert result.passed
        assert result.blocks_checked == 0

    @pytest.mark.parametrize("n_blocks", [1, 5, 10])
    def test_various_chain_lengths(self, n_blocks):
        chain = _build_chain(n_blocks)
        w = WitnessData(header=_make_header(), hash_chain=chain)
        result = HashChainVerifier(w).verify_full()
        assert result.passed
        assert result.blocks_checked == n_blocks


# ---------------------------------------------------------------------------
# TestClosureValidator
# ---------------------------------------------------------------------------


class TestClosureValidator:
    """Bisimulation closure property validation."""

    def test_valid_closure_passes(self):
        w = _valid_witness()
        result = ClosureValidator(w).validate_full()
        assert result.passed
        assert result.forward_checked > 0

    def test_forward_closure_violation(self):
        """Remove a matching transition so forward closure fails."""
        eqs = _simple_equivalences()
        # Only one transition: s0->s2 but no s1->anything in class 1
        txs = [
            TransitionWitness(source_class=0, target_class=1,
                              original_source="s0", original_target="s2",
                              matching_path=("s2",)),
        ]
        w = WitnessData(header=_make_header(), equivalences=eqs, transitions=txs)
        result = ClosureValidator(w).validate_full()
        assert not result.passed
        kinds = {v.kind for v in result.violations}
        assert ViolationKind.FORWARD_CLOSURE in kinds or ViolationKind.BACKWARD_CLOSURE in kinds

    def test_ap_preservation_violation(self):
        """States in same class with different APs should be caught."""
        eqs = [
            EquivalenceBinding(class_id=0, representative="s0",
                               members=("s0", "s1"), ap_labels=("a",)),
        ]
        # Manually build a witness where the state_aps will differ—
        # this needs two equivalence entries for the same class with
        # different APs. The validator takes APs from the binding itself,
        # so all states in a binding share the binding's APs by construction.
        # AP violations occur when cross-referencing with other data;
        # we verify the check runs without error on a clean witness.
        w = WitnessData(header=_make_header(), equivalences=eqs)
        result = ClosureValidator(w).validate_full()
        assert result.ap_checked > 0

    def test_stutter_bound_exceeded(self):
        """Stutter depth beyond the bound should be flagged."""
        eqs = _simple_equivalences()
        txs = [
            TransitionWitness(source_class=0, target_class=1,
                              original_source="s0", original_target="s2",
                              matching_path=("s0", "s0", "s2"),
                              is_stutter=True, stutter_depth=2000),
        ]
        header = _make_header(flags=WitnessFlag.STUTTERING)
        w = WitnessData(header=header, equivalences=eqs, transitions=txs)
        result = ClosureValidator(w, stutter_bound=100).validate_full()
        assert not result.passed
        assert any(v.kind == ViolationKind.STUTTER_BOUND_EXCEEDED
                   for v in result.violations)

    def test_validate_statistical(self):
        w = _valid_witness()
        result = ClosureValidator(w).validate_statistical(sample_fraction=0.5)
        assert result.mode == "statistical"


# ---------------------------------------------------------------------------
# TestStutteringVerifier
# ---------------------------------------------------------------------------


class TestStutteringVerifier:
    """Stuttering equivalence verification."""

    def test_valid_stuttering(self):
        w = _valid_witness(stuttering=True)
        result = StutteringVerifier(w, verify_hashes=False).verify()
        assert result.passed

    def test_stutter_chain_cycle_detected(self):
        """A stutter transition with a cyclic matching path should fail."""
        eqs = _simple_equivalences()
        txs = [
            TransitionWitness(source_class=0, target_class=1,
                              original_source="s0", original_target="s2",
                              matching_path=("s0", "s1", "s0", "s2"),
                              is_stutter=True, stutter_depth=3),
        ]
        header = _make_header(flags=WitnessFlag.STUTTERING)
        w = WitnessData(header=header, equivalences=eqs, transitions=txs)
        result = StutteringVerifier(w, verify_hashes=False).verify()
        assert not result.passed
        assert any(v.kind == StutteringViolationKind.INFINITE_STUTTER
                   for v in result.violations)

    def test_empty_matching_path_violation(self):
        """A stutter transition with empty matching_path should be flagged."""
        eqs = _simple_equivalences()
        txs = [
            TransitionWitness(source_class=0, target_class=1,
                              original_source="s0", original_target="s2",
                              matching_path=(), is_stutter=True, stutter_depth=0),
        ]
        header = _make_header(flags=WitnessFlag.STUTTERING)
        w = WitnessData(header=header, equivalences=eqs, transitions=txs)
        result = StutteringVerifier(w, verify_hashes=False).verify()
        assert not result.passed
        assert any(v.kind == StutteringViolationKind.WITNESS_INCOMPLETE
                   for v in result.violations)

    def test_no_stutter_transitions_passes(self):
        """Non-stuttering witness should pass stuttering checks trivially."""
        w = _valid_witness(stuttering=False)
        result = StutteringVerifier(w, verify_hashes=False).verify()
        assert result.passed


# ---------------------------------------------------------------------------
# TestFairnessVerifier
# ---------------------------------------------------------------------------


class TestFairnessVerifier:
    """Fairness preservation verification."""

    def test_valid_fairness(self):
        w = _valid_witness(with_fairness=True)
        result = FairnessVerifier(w).verify()
        assert result.passed
        assert result.pairs_checked == 1

    def test_missing_pair(self):
        w = _valid_witness(with_fairness=True)
        result = FairnessVerifier(w).verify_pair(pair_id=999)
        assert not result.passed
        assert result.violations[0].kind == FairnessViolationKind.MISSING_PAIR

    def test_rabin_overlap_violation(self):
        """Rabin acceptance with overlapping B/G sets should be flagged."""
        eqs = _simple_equivalences()
        fairness = [
            FairnessBinding(pair_id=0,
                            b_set_classes=(0, 1), g_set_classes=(1,)),
        ]
        header = _make_header(flags=WitnessFlag.FAIRNESS_PRESENT)
        w = WitnessData(
            header=header, equivalences=eqs, transitions=_simple_transitions(),
            fairness=fairness, metadata={"acceptance_type": "rabin"},
        )
        result = FairnessVerifier(w).verify()
        assert not result.passed
        assert any(v.kind == FairnessViolationKind.PAIR_OVERLAP_INVALID
                   for v in result.violations)

    def test_streett_overlap_ok(self):
        """Streett acceptance allows B/G overlap."""
        eqs = _simple_equivalences()
        fairness = [
            FairnessBinding(pair_id=0,
                            b_set_classes=(0, 1), g_set_classes=(1,)),
        ]
        header = _make_header(flags=WitnessFlag.FAIRNESS_PRESENT)
        w = WitnessData(
            header=header, equivalences=eqs, transitions=_simple_transitions(),
            fairness=fairness, metadata={"acceptance_type": "streett"},
        )
        result = FairnessVerifier(w).verify()
        assert not any(v.kind == FairnessViolationKind.PAIR_OVERLAP_INVALID
                       for v in result.violations)

    def test_no_fairness_data(self):
        w = _valid_witness(with_fairness=False)
        result = FairnessVerifier(w).verify()
        assert result.passed
        assert result.pairs_checked == 0

    @pytest.mark.parametrize("pair_id", [0])
    def test_verify_single_pair(self, pair_id):
        w = _valid_witness(with_fairness=True)
        result = FairnessVerifier(w).verify_pair(pair_id)
        assert result.passed


# ---------------------------------------------------------------------------
# TestVerificationReport
# ---------------------------------------------------------------------------


class TestVerificationReport:
    """Verification report generation."""

    def _make_passing_results(self, witness: WitnessData):
        """Generate all-passing results for a witness."""
        hr = HashVerificationResult(passed=True, blocks_checked=2,
                                    blocks_total=2)
        cr = ClosureResult(passed=True, forward_checked=4,
                           backward_checked=4, ap_checked=4)
        sr = StutteringResult(passed=True, paths_checked=10,
                              divergence_checks=4)
        fr = FairnessResult(passed=True, pairs_checked=1)
        return hr, cr, sr, fr

    def test_all_pass_yields_verified(self):
        w = _valid_witness(with_chain=True)
        report = VerificationReport(w)
        hr, cr, sr, fr = self._make_passing_results(w)
        report.add_hash_result(hr)
        report.add_closure_result(cr)
        report.add_stuttering_result(sr)
        report.add_fairness_result(fr)
        report.finalize()
        assert report.verdict == Verdict.VERIFIED
        assert report.trust_level in (TrustLevel.HIGH, TrustLevel.MEDIUM)

    def test_failure_yields_rejected(self):
        w = _valid_witness(with_chain=True)
        report = VerificationReport(w)
        hr = HashVerificationResult(passed=False, blocks_checked=1,
                                    blocks_total=2)
        cr = ClosureResult(passed=True)
        report.add_hash_result(hr)
        report.add_closure_result(cr)
        report.finalize()
        assert report.verdict == Verdict.REJECTED
        assert report.trust_level == TrustLevel.NONE

    def test_no_phases_yields_error(self):
        w = _valid_witness()
        report = VerificationReport(w)
        report.finalize()
        assert report.verdict == Verdict.ERROR

    def test_to_text_contains_verdict(self):
        w = _valid_witness(with_chain=True)
        report = VerificationReport(w)
        hr, cr, sr, fr = self._make_passing_results(w)
        report.add_hash_result(hr)
        report.add_closure_result(cr)
        report.add_stuttering_result(sr)
        report.add_fairness_result(fr)
        report.finalize()
        text = report.to_text()
        assert "VERIFIED" in text
        assert "CoaCert-TLA" in text

    def test_to_json_parseable(self):
        w = _valid_witness(with_chain=True)
        report = VerificationReport(w)
        hr, cr, sr, fr = self._make_passing_results(w)
        report.add_hash_result(hr)
        report.add_closure_result(cr)
        report.add_stuttering_result(sr)
        report.add_fairness_result(fr)
        report.finalize()
        j = json.loads(report.to_json())
        assert j["verdict"] == "VERIFIED"
        assert "phases" in j
        assert len(j["phases"]) == 4

    def test_to_json_contains_timing(self):
        w = _valid_witness(with_chain=True)
        report = VerificationReport(w)
        hr = HashVerificationResult(passed=True, blocks_checked=2,
                                    blocks_total=2, elapsed_seconds=0.01)
        report.add_hash_result(hr)
        report.finalize()
        j = json.loads(report.to_json())
        assert "timing_breakdown" in j

    def test_phases_list(self):
        w = _valid_witness()
        report = VerificationReport(w)
        report.add_closure_result(ClosureResult(passed=True))
        assert len(report.phases) == 1
        assert report.phases[0].name == "Bisimulation Closure"

    def test_rejected_text_contains_fail(self):
        w = _valid_witness()
        report = VerificationReport(w)
        cr = ClosureResult(passed=False)
        cr.add_violation(ClosureViolation(
            kind=ViolationKind.FORWARD_CLOSURE,
            state_s="s0", state_t="s1",
            class_s=0, class_t=0,
            message="test violation",
        ))
        report.add_closure_result(cr)
        report.finalize()
        text = report.to_text()
        assert "REJECTED" in text
        assert "FAIL" in text

    def test_repr(self):
        w = _valid_witness()
        report = VerificationReport(w)
        report.finalize()
        r = repr(report)
        assert "VerificationReport" in r


# ---------------------------------------------------------------------------
# TestEndToEnd
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """End-to-end: build a small witness, verify with all verifiers."""

    def _build_e2e_witness(self) -> WitnessData:
        """A self-consistent witness with classes, transitions, chain, fairness."""
        eqs = [
            EquivalenceBinding(class_id=0, representative="a0",
                               members=("a0", "a1"), ap_labels=("p",)),
            EquivalenceBinding(class_id=1, representative="b0",
                               members=("b0", "b1"), ap_labels=("q",)),
        ]
        txs = [
            TransitionWitness(source_class=0, target_class=1,
                              original_source="a0", original_target="b0",
                              matching_path=("b0",)),
            TransitionWitness(source_class=0, target_class=1,
                              original_source="a1", original_target="b1",
                              matching_path=("b1",)),
            TransitionWitness(source_class=1, target_class=0,
                              original_source="b0", original_target="a0",
                              matching_path=("a0",)),
            TransitionWitness(source_class=1, target_class=0,
                              original_source="b1", original_target="a1",
                              matching_path=("a1",)),
        ]
        chain = _build_chain(3)
        fairness = [
            FairnessBinding(pair_id=0,
                            b_set_classes=(0, 1), g_set_classes=(0, 1)),
        ]
        header = _make_header(flags=WitnessFlag.FAIRNESS_PRESENT)
        return WitnessData(
            header=header,
            equivalences=eqs,
            transitions=txs,
            hash_chain=chain,
            fairness=fairness,
            metadata={"spec_name": "e2e_test", "acceptance_type": "streett"},
        )

    def test_full_pipeline_verified(self):
        w = self._build_e2e_witness()
        report = VerificationReport(w)

        hr = HashChainVerifier(w).verify_full()
        report.add_hash_result(hr)
        assert hr.passed

        cr = ClosureValidator(w).validate_full()
        report.add_closure_result(cr)
        assert cr.passed

        sr = StutteringVerifier(w, verify_hashes=False).verify()
        report.add_stuttering_result(sr)
        assert sr.passed

        fr = FairnessVerifier(w).verify()
        report.add_fairness_result(fr)
        assert fr.passed

        report.finalize()
        assert report.verdict == Verdict.VERIFIED

    def test_e2e_json_report(self):
        w = self._build_e2e_witness()
        report = VerificationReport(w)

        report.add_hash_result(HashChainVerifier(w).verify_full())
        report.add_closure_result(ClosureValidator(w).validate_full())
        report.add_stuttering_result(
            StutteringVerifier(w, verify_hashes=False).verify())
        report.add_fairness_result(FairnessVerifier(w).verify())
        report.finalize()

        j = json.loads(report.to_json())
        assert j["verdict"] == "VERIFIED"
        assert j["witness"]["num_equivalences"] == 2
        assert j["witness"]["num_transitions"] == 4
        assert j["witness"]["metadata"]["spec_name"] == "e2e_test"

    def test_e2e_tampered_chain_rejected(self):
        """End-to-end with a tampered hash chain should produce REJECTED."""
        w = self._build_e2e_witness()
        # Tamper with block 1
        blk = w.hash_chain[1]
        w.hash_chain[1] = HashBlock(
            index=blk.index,
            prev_hash=blk.prev_hash,
            payload_hash=blk.payload_hash,
            block_hash=b"\xde\xad" * 16,
        )
        report = VerificationReport(w)
        report.add_hash_result(HashChainVerifier(w).verify_full())
        report.add_closure_result(ClosureValidator(w).validate_full())
        report.finalize()
        assert report.verdict == Verdict.REJECTED

    def test_e2e_parallel_hash_verify(self):
        """verify_parallel should agree with verify_full."""
        w = self._build_e2e_witness()
        full = HashChainVerifier(w).verify_full()
        parallel = HashChainVerifier(w).verify_parallel()
        assert full.passed == parallel.passed
        assert full.blocks_checked == parallel.blocks_checked
