"""Tests for I1 improvements: causal inference enhancements."""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from marace.hb.causal_inference import (
    AgentEvent,
    CausalEdge,
    CausalChainType,
    CausalEvidence,
    SoundnessClassification,
    TransitiveCausalityClosure,
    CommunicationCausalityExtractor,
    CommunicationEvent,
    EnvironmentMediatedCausalChain,
    CausalInferenceEngine,
    classify_edge_soundness,
)


# ======================================================================
# CausalChainType and SoundnessClassification
# ======================================================================

class TestCausalChainType:
    def test_enum_values(self):
        assert CausalChainType.OBSERVATION_DEPENDENCY.value == "observation_dependency"
        assert CausalChainType.PHYSICS_MEDIATED.value == "physics_mediated"
        assert CausalChainType.COMMUNICATION.value == "communication"
        assert CausalChainType.ENVIRONMENT_MEDIATED.value == "environment_mediated"
        assert CausalChainType.TRANSITIVE.value == "transitive"
        assert CausalChainType.USER_ANNOTATED.value == "user_annotated"

    def test_soundness_values(self):
        assert SoundnessClassification.EXACT.value == "exact"
        assert SoundnessClassification.OVER_APPROXIMATE.value == "over-approximate"
        assert SoundnessClassification.USER_ANNOTATED.value == "user-annotated"


# ======================================================================
# CausalEvidence
# ======================================================================

class TestCausalEvidence:
    def test_create(self):
        ev = CausalEvidence(
            chain_type=CausalChainType.COMMUNICATION,
            soundness=SoundnessClassification.EXACT,
            confidence=1.0,
        )
        assert ev.chain_type == CausalChainType.COMMUNICATION
        assert ev.soundness == SoundnessClassification.EXACT

    def test_round_trip_dict(self):
        ev = CausalEvidence(
            chain_type=CausalChainType.PHYSICS_MEDIATED,
            soundness=SoundnessClassification.OVER_APPROXIMATE,
            supporting_edges=[("e0", "e1")],
            state_variable="pos_x",
            delay=2,
            confidence=0.8,
        )
        d = ev.to_dict()
        ev2 = CausalEvidence.from_dict(d)
        assert ev2.chain_type == CausalChainType.PHYSICS_MEDIATED
        assert ev2.soundness == SoundnessClassification.OVER_APPROXIMATE
        assert ev2.state_variable == "pos_x"
        assert ev2.delay == 2

    def test_confidence_preserved(self):
        ev = CausalEvidence(
            chain_type=CausalChainType.TRANSITIVE,
            soundness=SoundnessClassification.OVER_APPROXIMATE,
            confidence=0.42,
        )
        d = ev.to_dict()
        ev2 = CausalEvidence.from_dict(d)
        assert abs(ev2.confidence - 0.42) < 1e-10


# ======================================================================
# classify_edge_soundness
# ======================================================================

class TestClassifyEdgeSoundness:
    def test_communication_is_exact(self):
        edge = CausalEdge(source="e0", target="e1",
                          mechanism="communication", confidence=1.0)
        assert classify_edge_soundness(edge) == SoundnessClassification.EXACT

    def test_physics_is_over_approximate(self):
        edge = CausalEdge(source="e0", target="e1",
                          mechanism="physics_mediated", confidence=0.8)
        assert classify_edge_soundness(edge) == SoundnessClassification.OVER_APPROXIMATE

    def test_observation_is_over_approximate(self):
        edge = CausalEdge(source="e0", target="e1",
                          mechanism="observation_dependency", confidence=0.5)
        assert classify_edge_soundness(edge) == SoundnessClassification.OVER_APPROXIMATE

    def test_user_annotated(self):
        edge = CausalEdge(source="e0", target="e1",
                          mechanism="user_annotated", confidence=1.0)
        assert classify_edge_soundness(edge) == SoundnessClassification.USER_ANNOTATED

    def test_transitive_is_over_approximate(self):
        edge = CausalEdge(source="e0", target="e2",
                          mechanism="transitive", confidence=0.5)
        assert classify_edge_soundness(edge) == SoundnessClassification.OVER_APPROXIMATE


# ======================================================================
# TransitiveCausalityClosure
# ======================================================================

class TestTransitiveCausalityClosure:
    def test_simple_chain(self):
        edges = [
            CausalEdge(source="e0", target="e1",
                       mechanism="communication", confidence=1.0),
            CausalEdge(source="e1", target="e2",
                       mechanism="communication", confidence=1.0),
        ]
        tc = TransitiveCausalityClosure(max_depth=5)
        closed = tc.close(edges)
        # Should derive e0 -> e2 transitively
        targets = {(e.source, e.target) for e in closed}
        assert ("e0", "e2") in targets

    def test_transitive_confidence_is_product(self):
        edges = [
            CausalEdge(source="e0", target="e1",
                       mechanism="physics_mediated", confidence=0.5),
            CausalEdge(source="e1", target="e2",
                       mechanism="physics_mediated", confidence=0.5),
        ]
        tc = TransitiveCausalityClosure(max_depth=5)
        closed = tc.close(edges)
        transitive = [e for e in closed if e.mechanism == "transitive"]
        assert len(transitive) == 1
        assert abs(transitive[0].confidence - 0.25) < 1e-10

    def test_bounded_depth(self):
        edges = [
            CausalEdge(source=f"e{i}", target=f"e{i+1}",
                       mechanism="communication", confidence=1.0)
            for i in range(10)
        ]
        tc = TransitiveCausalityClosure(max_depth=3)
        closed = tc.close(edges)
        # Should not derive e0 -> e5 (depth > 3)
        targets = {(e.source, e.target) for e in closed}
        assert ("e0", "e5") not in targets
        # But should derive e0 -> e3 (depth 3)
        assert ("e0", "e3") in targets

    def test_min_confidence_filter(self):
        edges = [
            CausalEdge(source="e0", target="e1",
                       mechanism="physics_mediated", confidence=0.1),
            CausalEdge(source="e1", target="e2",
                       mechanism="physics_mediated", confidence=0.1),
        ]
        tc = TransitiveCausalityClosure(max_depth=5, min_confidence=0.05)
        closed = tc.close(edges)
        # 0.1 * 0.1 = 0.01 < 0.05, so no transitive edge
        transitive = [e for e in closed if e.mechanism == "transitive"]
        assert len(transitive) == 0

    def test_no_self_loops(self):
        edges = [
            CausalEdge(source="e0", target="e1",
                       mechanism="communication", confidence=1.0),
            CausalEdge(source="e1", target="e0",
                       mechanism="communication", confidence=1.0),
        ]
        tc = TransitiveCausalityClosure(max_depth=5)
        closed = tc.close(edges)
        self_loops = [e for e in closed if e.source == e.target]
        assert len(self_loops) == 0

    def test_close_with_evidence(self):
        edges = [
            CausalEdge(source="e0", target="e1",
                       mechanism="communication", confidence=1.0),
            CausalEdge(source="e1", target="e2",
                       mechanism="physics_mediated", confidence=0.8),
        ]
        tc = TransitiveCausalityClosure(max_depth=5)
        closed, evidence = tc.close_with_evidence(edges)
        assert len(evidence) == len(closed)
        # Find evidence for transitive edge
        transitive_ev = [ev for ev in evidence
                         if ev.chain_type == CausalChainType.TRANSITIVE]
        assert len(transitive_ev) >= 1
        assert transitive_ev[0].soundness == SoundnessClassification.OVER_APPROXIMATE

    def test_evidence_for_direct_edges(self):
        edges = [
            CausalEdge(source="e0", target="e1",
                       mechanism="communication", confidence=1.0),
        ]
        tc = TransitiveCausalityClosure(max_depth=5)
        closed, evidence = tc.close_with_evidence(edges)
        assert len(evidence) == 1
        assert evidence[0].chain_type == CausalChainType.COMMUNICATION
        assert evidence[0].soundness == SoundnessClassification.EXACT

    def test_empty_edges(self):
        tc = TransitiveCausalityClosure()
        closed = tc.close([])
        assert closed == []


# ======================================================================
# Environment-mediated causal chain with evidence
# ======================================================================

class TestEnvironmentMediatedWithEvidence:
    def test_basic_write_read_chain(self):
        chain = EnvironmentMediatedCausalChain(
            state_dimensions=["pos_x", "pos_y"],
            staleness_limit=5,
        )
        chain.register_write("w1", "agent_0", 0, {"pos_x"})
        edges = chain.register_read("r1", "agent_1", 2, {"pos_x"})
        assert len(edges) == 1
        assert edges[0].mechanism == "environment_mediated"
        assert edges[0].metadata["dimension"] == "pos_x"

    def test_soundness_classification(self):
        chain = EnvironmentMediatedCausalChain(staleness_limit=5)
        chain.register_write("w1", "agent_0", 0, {"x"})
        edges = chain.register_read("r1", "agent_1", 1, {"x"})
        for e in edges:
            sc = classify_edge_soundness(e)
            assert sc == SoundnessClassification.OVER_APPROXIMATE
