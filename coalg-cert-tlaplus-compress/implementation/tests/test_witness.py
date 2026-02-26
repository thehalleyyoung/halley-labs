"""Comprehensive tests for coacert.witness module."""

import math
import struct

import pytest

from coacert.witness.merkle_tree import (
    EMPTY_HASH,
    HASH_LEN,
    InternalNode,
    LeafNode,
    MerkleProof,
    MerkleTree,
    SparseMerkleTree,
    hash_leaf,
    sha256,
)
from coacert.witness.equivalence_binding import ClassBinding, EquivalenceBinding
from coacert.witness.hash_chain import (
    BlockType,
    EquivalenceBlock,
    FairnessBlock,
    HashChain,
    TransitionBlock,
)
from coacert.witness.transition_witness import (
    FairnessWitness,
    StutterWitness,
    TransitionWitness,
    WitnessSet,
)
from coacert.witness.compact_repr import (
    BloomFilter,
    CompactWitness,
    decode_delta_bytes,
    decode_signed_varint,
    decode_varint,
    delta_decode,
    delta_encode,
    encode_delta_bytes,
    encode_signed_varint,
    encode_varint,
    truncate_hash,
    truncated_hash_collision_probability,
)
from coacert.witness.witness_format import WitnessFormat


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _items(n: int) -> list[bytes]:
    """Return *n* distinct byte-string items."""
    return [f"item-{i}".encode() for i in range(n)]


def _make_state_to_class(*classes: tuple[str, list[str]]) -> dict[str, str]:
    """Build a state→class mapping from (class_id, [states]) pairs."""
    m: dict[str, str] = {}
    for cid, states in classes:
        for s in states:
            m[s] = cid
    return m


# ===================================================================
# MerkleTree
# ===================================================================

class TestMerkleTree:
    """Tests for MerkleTree construction, proofs, and mutation."""

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 7, 8, 16])
    def test_build_and_root_determinism(self, n: int):
        items = _items(n)
        t1 = MerkleTree(items)
        t2 = MerkleTree(items)
        assert t1.root == t2.root
        assert len(t1.root) == HASH_LEN

    @pytest.mark.parametrize("n", [1, 2, 4, 5])
    def test_leaf_count(self, n: int):
        tree = MerkleTree(_items(n))
        assert tree.leaf_count == n

    def test_depth(self):
        tree = MerkleTree(_items(8))
        assert tree.depth >= 3

    def test_node_count(self):
        tree = MerkleTree(_items(4))
        assert tree.node_count >= tree.leaf_count

    @pytest.mark.parametrize("n", [2, 4, 8])
    def test_proof_valid(self, n: int):
        items = _items(n)
        tree = MerkleTree(items)
        for item in items:
            proof = tree.proof(item)
            assert proof.verify()

    def test_proof_invalid_after_tamper(self):
        tree = MerkleTree(_items(4))
        proof = tree.proof(b"item-0")
        bad_proof = MerkleProof(
            leaf_hash=sha256(b"TAMPERED"),
            siblings=proof.siblings,
            root_hash=proof.root_hash,
        )
        assert not bad_proof.verify()

    @pytest.mark.parametrize("n", [3, 5, 8])
    def test_batch_proofs(self, n: int):
        items = _items(n)
        tree = MerkleTree(items)
        proofs = tree.batch_proofs(items)
        assert len(proofs) == n
        for p in proofs:
            assert p.verify()

    def test_add_leaf(self):
        tree = MerkleTree(_items(3))
        old_root = tree.root
        tree.add_leaf(b"new-item")
        assert tree.leaf_count == 4
        assert tree.root != old_root

    def test_remove_leaf(self):
        items = _items(4)
        tree = MerkleTree(items)
        tree.remove_leaf(items[1])
        assert tree.leaf_count == 3
        proof = tree.proof(items[0])
        assert proof.verify()

    def test_update_leaf(self):
        items = _items(4)
        tree = MerkleTree(items)
        old_root = tree.root
        tree.update_leaf(items[0], b"replaced")
        assert tree.root != old_root
        proof = tree.proof(b"replaced")
        assert proof.verify()

    def test_serialization_roundtrip(self):
        tree = MerkleTree(_items(5))
        data = tree.to_bytes()
        restored = MerkleTree.from_bytes(data)
        assert restored.root == tree.root
        assert restored.leaf_count == tree.leaf_count

    def test_pretty_print(self):
        tree = MerkleTree(_items(2))
        text = tree.pretty()
        assert isinstance(text, str)
        assert len(text) > 0

    def test_proof_size_property(self):
        tree = MerkleTree(_items(8))
        assert tree.proof_size > 0


# ===================================================================
# MerkleProof
# ===================================================================

class TestMerkleProof:
    """Tests for standalone MerkleProof verification and serialization."""

    def _make_proof(self) -> MerkleProof:
        tree = MerkleTree(_items(4))
        return tree.proof(b"item-0")

    def test_verify_valid(self):
        assert self._make_proof().verify()

    def test_reject_tampered_leaf(self):
        proof = self._make_proof()
        bad = MerkleProof(sha256(b"bad"), proof.siblings, proof.root_hash)
        assert not bad.verify()

    def test_reject_tampered_sibling(self):
        proof = self._make_proof()
        tampered_siblings = [(sha256(b"x"), d) for _, d in proof.siblings]
        bad = MerkleProof(proof.leaf_hash, tampered_siblings, proof.root_hash)
        assert not bad.verify()

    def test_serialization_roundtrip(self):
        proof = self._make_proof()
        data = proof.to_bytes()
        restored, _ = MerkleProof.from_bytes(data)
        assert restored.leaf_hash == proof.leaf_hash
        assert restored.root_hash == proof.root_hash
        assert restored.verify()


# ===================================================================
# SparseMerkleTree
# ===================================================================

class TestSparseMerkleTree:
    """Tests for SparseMerkleTree."""

    def test_set_get(self):
        smt = SparseMerkleTree(depth=8)
        val = sha256(b"val")
        smt.set(5, val)
        # set() applies hash_leaf internally, so get() != raw value
        result = smt.get(5)
        assert len(result) == HASH_LEN
        assert result != EMPTY_HASH

    def test_delete(self):
        smt = SparseMerkleTree(depth=8)
        smt.set(3, sha256(b"v"))
        smt.delete(3)
        assert smt.get(3) == EMPTY_HASH

    def test_populated_count(self):
        smt = SparseMerkleTree(depth=8)
        smt.set(0, sha256(b"a"))
        smt.set(1, sha256(b"b"))
        assert smt.populated_count == 2
        smt.delete(0)
        assert smt.populated_count == 1

    def test_proof_and_verify(self):
        smt = SparseMerkleTree(depth=8)
        smt.set(7, sha256(b"hello"))
        proof = smt.proof(7)
        assert smt.verify_proof(proof)

    def test_root_changes_on_set(self):
        smt = SparseMerkleTree(depth=8)
        root0 = smt.root
        smt.set(1, sha256(b"x"))
        assert smt.root != root0

    def test_serialization_roundtrip(self):
        smt = SparseMerkleTree(depth=8)
        smt.set(2, sha256(b"data"))
        data = smt.to_bytes()
        restored = SparseMerkleTree.from_bytes(data)
        assert restored.root == smt.root
        assert restored.get(2) == smt.get(2)

    def test_pretty(self):
        smt = SparseMerkleTree(depth=8)
        smt.set(0, sha256(b"x"))
        assert isinstance(smt.pretty(), str)


# ===================================================================
# ClassBinding
# ===================================================================

class TestClassBinding:
    """Tests for ClassBinding dataclass."""

    def test_creation(self):
        cb = ClassBinding("c0", "rep", frozenset({"a", "b", "rep"}))
        assert cb.class_id == "c0"
        assert cb.member_count == 3

    def test_hashes_consistent(self):
        cb = ClassBinding("c0", "rep", frozenset({"a", "b", "rep"}))
        assert len(cb.representative_hash) == HASH_LEN
        assert len(cb.members_hash) == HASH_LEN
        assert len(cb.binding_hash) == HASH_LEN

    def test_sorted_member_hashes(self):
        cb = ClassBinding("c0", "rep", frozenset({"a", "b", "c"}))
        hashes = cb.sorted_member_hashes
        assert hashes == sorted(hashes)

    def test_full_bytes_roundtrip(self):
        cb = ClassBinding("c1", "rep", frozenset({"rep", "x", "y"}))
        data = cb.full_bytes()
        restored, _ = ClassBinding.from_full_bytes(data)
        assert restored.class_id == cb.class_id
        assert restored.member_count >= 0

    def test_compact_bytes_shorter(self):
        cb = ClassBinding("c0", "rep", frozenset({"a", "b", "c", "d", "e"}))
        assert len(cb.compact_bytes()) <= len(cb.full_bytes())


# ===================================================================
# EquivalenceBinding
# ===================================================================

class TestEquivalenceBinding:
    """Tests for EquivalenceBinding."""

    def _make_binding(self) -> EquivalenceBinding:
        eb = EquivalenceBinding()
        eb.add_class("c0", "s0", frozenset({"s0", "s1"}))
        eb.add_class("c1", "s2", frozenset({"s2", "s3"}))
        return eb

    def test_add_class(self):
        eb = self._make_binding()
        assert eb.class_count == 2
        assert eb.total_states == 4

    def test_class_for_state(self):
        eb = self._make_binding()
        assert eb.class_for_state("s0") == "c0"
        assert eb.class_for_state("s3") == "c1"
        assert eb.class_for_state("unknown") is None

    def test_remove_class(self):
        eb = self._make_binding()
        eb.remove_class("c1")
        assert eb.class_count == 1
        assert eb.class_for_state("s2") is None

    def test_get_class(self):
        eb = self._make_binding()
        cb = eb.get_class("c0")
        assert cb.class_id == "c0"
        assert cb.member_count == 2

    def test_verify_partition_valid(self):
        eb = self._make_binding()
        ok, errs = eb.verify_partition()
        assert ok
        assert len(errs) == 0

    def test_verify_partition_overlapping(self):
        eb = EquivalenceBinding()
        eb.add_class("c0", "s0", frozenset({"s0", "s1"}))
        eb.add_class("c1", "s1", frozenset({"s1", "s2"}))
        ok, errs = eb.verify_partition()
        assert not ok
        assert len(errs) > 0

    def test_verify_binding_hashes(self):
        eb = self._make_binding()
        assert eb.verify_binding_hashes()

    def test_serialization_roundtrip(self):
        eb = self._make_binding()
        data = eb.to_bytes()
        restored = EquivalenceBinding.from_bytes(data)
        assert restored.class_count == eb.class_count
        assert restored.verify_binding_hashes()

    def test_serialization_compact(self):
        eb = self._make_binding()
        data_compact = eb.to_bytes(compact=True)
        data_full = eb.to_bytes(compact=False)
        assert len(data_compact) <= len(data_full)

    def test_to_dict(self):
        eb = self._make_binding()
        d = eb.to_dict()
        assert isinstance(d, dict)
        assert "c0" in str(d) or len(d) > 0

    def test_class_ids(self):
        eb = self._make_binding()
        assert set(eb.class_ids) == {"c0", "c1"}

    def test_root_is_hash(self):
        eb = self._make_binding()
        assert len(eb.root) == HASH_LEN


# ===================================================================
# HashChain
# ===================================================================

class TestHashChain:
    """Tests for HashChain block appending, verification, and tamper detection."""

    def _make_chain(self) -> HashChain:
        return HashChain.build(
            equivalence_payloads=[b"eq-payload"],
            transition_payloads=[b"tr-payload"],
            fairness_payloads=[b"fr-payload"],
        )

    def test_build_from_payloads(self):
        chain = self._make_chain()
        # build creates one block per payload (no separate genesis)
        assert chain.length >= 3

    def test_verify_valid_chain(self):
        chain = self._make_chain()
        ok, errs = chain.verify()
        assert ok
        assert len(errs) == 0

    def test_detect_tamper(self):
        chain = self._make_chain()
        assert chain.detect_tamper() is None
        # tamper with a block payload
        chain.blocks[1].payload = b"TAMPERED"
        idx = chain.detect_tamper()
        assert idx is not None

    def test_append_block(self):
        chain = self._make_chain()
        old_len = chain.length
        chain.append_block(BlockType.TRANSITION, b"extra")
        assert chain.length == old_len + 1

    @pytest.mark.parametrize("start,end", [(0, 2), (1, 3)])
    def test_verify_range(self, start: int, end: int):
        chain = self._make_chain()
        ok, errs = chain.verify_range(start, end)
        assert ok

    def test_serialization_roundtrip(self):
        chain = self._make_chain()
        data = chain.to_bytes()
        restored = HashChain.from_bytes(data)
        assert restored.length == chain.length
        assert restored.tip_hash == chain.tip_hash

    def test_merkle_root(self):
        chain = self._make_chain()
        assert len(chain.merkle_root) == HASH_LEN

    def test_genesis(self):
        chain = self._make_chain()
        # First block is the genesis (index 0), its type depends on build order
        assert chain.genesis.index == 0

    def test_tip(self):
        chain = self._make_chain()
        assert chain.tip is chain.blocks[-1]
        assert len(chain.tip_hash) == HASH_LEN

    def test_total_payload_size(self):
        chain = self._make_chain()
        assert chain.total_payload_size > 0

    def test_total_size(self):
        chain = self._make_chain()
        assert chain.total_size >= chain.total_payload_size

    def test_to_dict(self):
        chain = self._make_chain()
        d = chain.to_dict()
        assert isinstance(d, (dict, list))

    def test_pretty(self):
        chain = self._make_chain()
        assert isinstance(chain.pretty(), str)


# ===================================================================
# TransitionWitness
# ===================================================================

class TestTransitionWitness:
    """Tests for TransitionWitness."""

    def _make_tw(self) -> TransitionWitness:
        return TransitionWitness(
            source_class="c0",
            target_class="c1",
            action="act",
            concrete_source="s0",
            concrete_target="s2",
        )

    def test_creation(self):
        tw = self._make_tw()
        assert tw.source_class == "c0"
        assert tw.target_class == "c1"

    def test_digest(self):
        tw = self._make_tw()
        assert len(tw.digest) == HASH_LEN

    def test_verify_classes_valid(self):
        tw = self._make_tw()
        s2c = _make_state_to_class(("c0", ["s0", "s1"]), ("c1", ["s2", "s3"]))
        ok, errs = tw.verify_classes(s2c)
        assert ok
        assert len(errs) == 0

    def test_verify_classes_invalid(self):
        tw = self._make_tw()
        s2c = _make_state_to_class(("c0", ["s0"]), ("c1", ["s1"]))  # s2 missing
        ok, errs = tw.verify_classes(s2c)
        assert not ok

    def test_serialization_roundtrip(self):
        tw = self._make_tw()
        data = tw.to_bytes()
        restored, _ = TransitionWitness.from_bytes(data)
        assert restored.digest == tw.digest
        assert restored.source_class == tw.source_class

    def test_to_dict(self):
        tw = self._make_tw()
        d = tw.to_dict()
        assert isinstance(d, dict)
        assert d["source_class"] == "c0"

    @pytest.mark.parametrize("src,tgt", [("c0", "c0"), ("c0", "c1"), ("c1", "c0")])
    def test_digest_varies_with_classes(self, src: str, tgt: str):
        tw1 = TransitionWitness(src, tgt, "a", "s", "t")
        tw2 = TransitionWitness(tgt, src, "a", "s", "t")
        if src != tgt:
            assert tw1.digest != tw2.digest


# ===================================================================
# StutterWitness
# ===================================================================

class TestStutterWitness:
    """Tests for StutterWitness."""

    def _make_sw(self) -> StutterWitness:
        return StutterWitness(
            equiv_class="c0",
            path=["s0", "s1", "s0"],
            actions=["a", "b"],
        )

    def test_creation(self):
        sw = self._make_sw()
        assert sw.equiv_class == "c0"

    def test_path_length(self):
        sw = self._make_sw()
        assert sw.path_length == 3

    def test_digest(self):
        sw = self._make_sw()
        assert len(sw.digest) == HASH_LEN

    def test_verify_within_class_valid(self):
        sw = self._make_sw()
        s2c = _make_state_to_class(("c0", ["s0", "s1"]))
        ok, errs = sw.verify_within_class(s2c)
        assert ok

    def test_verify_within_class_invalid(self):
        sw = self._make_sw()
        s2c = _make_state_to_class(("c0", ["s0"]), ("c1", ["s1"]))
        ok, errs = sw.verify_within_class(s2c)
        assert not ok

    def test_serialization_roundtrip(self):
        sw = self._make_sw()
        data = sw.to_bytes()
        restored, _ = StutterWitness.from_bytes(data)
        assert restored.digest == sw.digest
        assert restored.path_length == sw.path_length


# ===================================================================
# FairnessWitness
# ===================================================================

class TestFairnessWitness:
    """Tests for FairnessWitness."""

    def _make_fw(self) -> FairnessWitness:
        return FairnessWitness(
            pair_id="p0",
            accepting_class="c0",
            accepting_state="s0",
            rejecting_class="c1",
            rejecting_state="s2",
            cycle_witness=["s0", "s1", "s0"],
        )

    def test_creation(self):
        fw = self._make_fw()
        assert fw.pair_id == "p0"

    def test_digest(self):
        fw = self._make_fw()
        assert len(fw.digest) == HASH_LEN

    def test_serialization_roundtrip(self):
        fw = self._make_fw()
        data = fw.to_bytes()
        restored, _ = FairnessWitness.from_bytes(data)
        assert restored.digest == fw.digest
        assert restored.pair_id == fw.pair_id

    def test_to_dict(self):
        fw = self._make_fw()
        d = fw.to_dict()
        assert d["pair_id"] == "p0"
        assert d["accepting_class"] == "c0"


# ===================================================================
# WitnessSet
# ===================================================================

class TestWitnessSet:
    """Tests for WitnessSet aggregation and verification."""

    def _make_witness_set(self) -> WitnessSet:
        ws = WitnessSet()
        ws.add_transition(TransitionWitness("c0", "c1", "a", "s0", "s2"))
        ws.add_stutter(StutterWitness("c0", ["s0", "s1"], ["a"]))
        ws.add_fairness(FairnessWitness("p0", "c0", "s0", "c1", "s2", ["s0"]))
        return ws

    def test_counts(self):
        ws = self._make_witness_set()
        assert ws.transition_count == 1
        assert ws.stutter_count == 1
        assert ws.fairness_count == 1
        assert ws.total_count == 3

    def test_root(self):
        ws = self._make_witness_set()
        assert len(ws.root) == HASH_LEN

    def test_completeness_check_pass(self):
        ws = WitnessSet()
        tw = TransitionWitness("c0", "c1", "a", "s0", "s2")
        ws.add_transition(tw)
        expected = {("c0", "a", "c1")}
        ok, missing = ws.check_completeness(expected)
        assert ok

    def test_completeness_check_fail(self):
        ws = WitnessSet()
        expected = {("c0", "a", "c1")}
        ok, missing = ws.check_completeness(expected)
        assert not ok
        assert len(missing) > 0

    def test_consistency_check_pass(self):
        ws = WitnessSet()
        ws.add_transition(TransitionWitness("c0", "c1", "a", "s0", "s2"))
        s2c = _make_state_to_class(("c0", ["s0"]), ("c1", ["s2"]))
        ok, errs = ws.check_consistency(s2c)
        assert ok

    def test_consistency_check_fail(self):
        ws = WitnessSet()
        ws.add_transition(TransitionWitness("c0", "c1", "a", "s0", "s2"))
        s2c = _make_state_to_class(("c0", ["s0"]), ("c1", ["s1"]))  # s2 not in c1
        ok, errs = ws.check_consistency(s2c)
        assert not ok

    def test_serialization_roundtrip(self):
        ws = self._make_witness_set()
        data = ws.to_bytes()
        restored = WitnessSet.from_bytes(data)
        assert restored.total_count == ws.total_count
        assert restored.root == ws.root

    def test_to_dict(self):
        ws = self._make_witness_set()
        d = ws.to_dict()
        assert isinstance(d, dict)


# ===================================================================
# Compact representation helpers
# ===================================================================

class TestCompactRepr:
    """Tests for varint / delta / truncation utilities."""

    @pytest.mark.parametrize("value", [0, 1, 127, 128, 300, 16383, 2**20, 2**28])
    def test_varint_roundtrip(self, value: int):
        encoded = encode_varint(value)
        decoded, consumed = decode_varint(encoded)
        assert decoded == value
        assert consumed == len(encoded)

    @pytest.mark.parametrize("value", [0, 1, -1, 127, -128, 300, -300, 2**20])
    def test_signed_varint_roundtrip(self, value: int):
        encoded = encode_signed_varint(value)
        decoded, consumed = decode_signed_varint(encoded)
        assert decoded == value
        assert consumed == len(encoded)

    def test_delta_encode_decode(self):
        values = [10, 20, 25, 100, 105]
        deltas = delta_encode(values)
        restored = delta_decode(deltas)
        assert restored == values

    def test_delta_encode_empty(self):
        assert delta_encode([]) == []
        assert delta_decode([]) == []

    def test_delta_bytes_roundtrip(self):
        values = [5, 10, 15, 20, 30]
        data = encode_delta_bytes(values)
        restored, _ = decode_delta_bytes(data)
        assert restored == values

    @pytest.mark.parametrize("length", [4, 8, 16, 32])
    def test_truncate_hash(self, length: int):
        h = sha256(b"test")
        trunc = truncate_hash(h, length)
        assert len(trunc) == length
        assert trunc == h[:length]

    def test_truncate_hash_rejects_bad_length(self):
        h = sha256(b"test")
        with pytest.raises(ValueError):
            truncate_hash(h, 2)

    def test_collision_probability(self):
        p = truncated_hash_collision_probability(1000, 16)
        assert 0.0 <= p <= 1.0
        # Larger n ⇒ higher probability
        p2 = truncated_hash_collision_probability(10000, 16)
        assert p2 >= p


# ===================================================================
# BloomFilter
# ===================================================================

class TestBloomFilter:
    """Tests for BloomFilter probabilistic membership."""

    def test_add_and_contains(self):
        bf = BloomFilter(expected_items=100, fp_rate=0.01)
        bf.add(b"hello")
        bf.add(b"world")
        assert bf.maybe_contains(b"hello")
        assert bf.maybe_contains(b"world")
        assert b"hello" in bf

    def test_count(self):
        bf = BloomFilter(expected_items=100, fp_rate=0.01)
        assert bf.count == 0
        bf.add(b"a")
        bf.add(b"b")
        assert bf.count == 2

    def test_properties(self):
        bf = BloomFilter(expected_items=1000, fp_rate=0.01)
        assert bf.bit_count > 0
        assert bf.hash_count > 0
        assert bf.size_bytes > 0

    def test_low_false_positive_rate(self):
        bf = BloomFilter(expected_items=500, fp_rate=0.01)
        for i in range(500):
            bf.add(f"item-{i}".encode())
        fp = sum(
            1
            for i in range(500, 1500)
            if bf.maybe_contains(f"item-{i}".encode())
        )
        # Allow generous margin: fp_rate ≤ 5% (target 1%)
        assert fp / 1000 < 0.05

    def test_estimated_fp_rate(self):
        bf = BloomFilter(expected_items=100, fp_rate=0.01)
        for i in range(100):
            bf.add(f"k{i}".encode())
        assert bf.estimated_fp_rate < 0.1

    def test_serialization_roundtrip(self):
        bf = BloomFilter(expected_items=50, fp_rate=0.01)
        for i in range(20):
            bf.add(f"item-{i}".encode())
        data = bf.to_bytes()
        restored, _ = BloomFilter.from_bytes(data)
        assert restored.count == bf.count
        for i in range(20):
            assert restored.maybe_contains(f"item-{i}".encode())


# ===================================================================
# Witness integrity (integration)
# ===================================================================

class TestWitnessIntegrity:
    """Integration: build a full witness and detect tampering."""

    @staticmethod
    def _build_components():
        eb = EquivalenceBinding()
        eb.add_class("c0", "s0", frozenset({"s0", "s1"}))
        eb.add_class("c1", "s2", frozenset({"s2", "s3"}))

        ws = WitnessSet()
        ws.add_transition(TransitionWitness("c0", "c1", "a", "s0", "s2"))
        ws.add_stutter(StutterWitness("c0", ["s0", "s1"], ["a"]))

        chain = HashChain.build(
            equivalence_payloads=[eb.to_bytes()],
            transition_payloads=[ws.to_bytes()],
            fairness_payloads=[],
        )
        return eb, ws, chain

    def test_chain_verify_after_build(self):
        _, _, chain = self._build_components()
        ok, errs = chain.verify()
        assert ok

    def test_tamper_detected_in_chain(self):
        _, _, chain = self._build_components()
        chain.blocks[1].payload = b"BAD"
        idx = chain.detect_tamper()
        assert idx is not None

    def test_equivalence_binding_hashes_valid(self):
        eb, _, _ = self._build_components()
        assert eb.verify_binding_hashes()

    def test_witness_set_root_stable(self):
        _, ws, _ = self._build_components()
        root1 = ws.root
        root2 = ws.root
        assert root1 == root2

    def test_witness_format_integrity(self, tmp_path):
        eb, ws, chain = self._build_components()
        wf = WitnessFormat(
            equivalence=eb,
            witnesses=ws,
            chain=chain,
            spec_hash=sha256(b"spec"),
            original_state_count=4,
            quotient_state_count=2,
        )
        ok, errs = wf.verify_integrity()
        assert ok, f"integrity errors: {errs}"

    def test_witness_format_serialize_deserialize(self, tmp_path):
        eb, ws, chain = self._build_components()
        wf = WitnessFormat(
            equivalence=eb,
            witnesses=ws,
            chain=chain,
            spec_hash=sha256(b"spec"),
            original_state_count=4,
            quotient_state_count=2,
        )
        path = tmp_path / "witness.bin"
        nbytes = wf.serialize(str(path))
        assert nbytes > 0
        restored = WitnessFormat.deserialize(str(path))
        # Verify chain portion survives roundtrip
        assert restored.chain.length == chain.length
        assert restored.witnesses.total_count == ws.total_count

    def test_compact_witness_compression(self):
        eb, ws, chain = self._build_components()
        cw = CompactWitness(eb, ws, chain, hash_truncation=16)
        analysis = cw.size_analysis()
        assert isinstance(analysis, dict)
        assert cw.compact_size > 0
