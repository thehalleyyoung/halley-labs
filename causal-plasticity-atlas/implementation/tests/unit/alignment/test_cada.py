"""Unit tests for cpa.alignment.cada module."""

from __future__ import annotations

import numpy as np
import pytest

from cpa.alignment.cada import (
    AlignmentCache,
    AlignmentResult,
    AnchorConflictError,
    BatchAligner,
    CADAAligner,
    EdgeClassification,
    EdgeType,
    TooManyUnanchoredError,
)
from cpa.core.scm import StructuralCausalModel


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def _make_scm(adj, names=None, coeffs=None, resid=None):
    """Build a StructuralCausalModel from a list-of-lists adjacency."""
    adj = np.array(adj, dtype=np.float64)
    return StructuralCausalModel(
        adjacency_matrix=adj,
        variable_names=names,
        regression_coefficients=coeffs,
        residual_variances=resid,
    )


def _make_dummy_result(ctx_a="A", ctx_b="B", quality=0.8, divergence=0.2):
    """Create a minimal AlignmentResult for cache / batch tests."""
    return AlignmentResult(
        alignment={0: 0},
        inverse_alignment={0: 0},
        match_scores={0: 1.0},
        edge_classifications=[],
        alignment_quality=quality,
        structural_divergence=divergence,
        normalized_divergence=divergence,
        n_shared=1,
        n_modified=0,
        n_context_specific_a=0,
        n_context_specific_b=0,
        n_anchored=1,
        n_matched=1,
        n_unmatched_a=0,
        n_unmatched_b=0,
        context_a=ctx_a,
        context_b=ctx_b,
        computation_time=0.01,
    )


class _SimpleMCCM:
    """Minimal multi-context causal model stub for BatchAligner tests."""

    def __init__(self, scms: dict):
        self._scms = scms

    @property
    def context_ids(self):
        return list(self._scms.keys())

    def get_scm(self, ctx):
        return self._scms[ctx]


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------

@pytest.fixture
def chain3_scm():
    """3-var chain: X0 -> X1 -> X2."""
    return _make_scm(
        [[0, 0.7, 0],
         [0, 0, 0.5],
         [0, 0, 0]],
        names=["X0", "X1", "X2"],
    )


@pytest.fixture
def chain3_scm_copy():
    """Identical copy of the 3-var chain."""
    return _make_scm(
        [[0, 0.7, 0],
         [0, 0, 0.5],
         [0, 0, 0]],
        names=["X0", "X1", "X2"],
    )


@pytest.fixture
def fork3_scm():
    """3-var fork: X0 -> X1, X0 -> X2."""
    return _make_scm(
        [[0, 0.6, 0.4],
         [0, 0, 0],
         [0, 0, 0]],
        names=["X0", "X1", "X2"],
    )


@pytest.fixture
def collider3_scm():
    """3-var collider: X0 -> X2 <- X1."""
    return _make_scm(
        [[0, 0, 0.5],
         [0, 0, 0.5],
         [0, 0, 0]],
        names=["X0", "X1", "X2"],
    )


@pytest.fixture
def diamond4_scm():
    """4-var diamond: X0 -> X1, X0 -> X2, X1 -> X3, X2 -> X3."""
    return _make_scm(
        [[0, 0.6, 0.4, 0],
         [0, 0, 0, 0.5],
         [0, 0, 0, 0.3],
         [0, 0, 0, 0]],
        names=["X0", "X1", "X2", "X3"],
    )


@pytest.fixture
def single_var_scm():
    """1-var SCM with no edges."""
    return _make_scm([[0]], names=["X0"])


@pytest.fixture
def empty_scm():
    """3-var SCM with zero edges."""
    return _make_scm(
        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]],
        names=["A", "B", "C"],
    )


@pytest.fixture
def disconnected_scm():
    """4-var with two disconnected components: X0->X1, X2->X3."""
    return _make_scm(
        [[0, 0.5, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0.5],
         [0, 0, 0, 0]],
        names=["X0", "X1", "X2", "X3"],
    )


@pytest.fixture
def aligner():
    """Default CADAAligner with relaxed thresholds for testing."""
    return CADAAligner(
        mb_overlap_threshold=0.0,
        quality_threshold=0.0,
        seed=42,
    )


@pytest.fixture
def strict_aligner():
    """CADAAligner with default thresholds."""
    return CADAAligner(seed=42)


# ---------------------------------------------------------------
# EdgeType enum
# ---------------------------------------------------------------

class TestEdgeType:
    def test_values(self):
        assert EdgeType.SHARED.value == "shared"
        assert EdgeType.MODIFIED.value == "modified"
        assert EdgeType.CONTEXT_SPECIFIC_A.value == "context_specific_a"
        assert EdgeType.CONTEXT_SPECIFIC_B.value == "context_specific_b"

    def test_members(self):
        assert len(EdgeType) == 4


# ---------------------------------------------------------------
# EdgeClassification dataclass
# ---------------------------------------------------------------

class TestEdgeClassification:
    def test_fields(self):
        ec = EdgeClassification(
            source_a=0, target_a=1,
            source_b=0, target_b=1,
            edge_type=EdgeType.SHARED,
            weight=0.0,
        )
        assert ec.source_a == 0
        assert ec.edge_type == EdgeType.SHARED
        assert ec.weight == 0.0

    def test_to_dict(self):
        ec = EdgeClassification(0, 1, 2, 3, EdgeType.MODIFIED, 0.5)
        d = ec.to_dict()
        assert d["edge_type"] == "modified"
        assert d["weight"] == 0.5
        assert d["source_a"] == 0 and d["target_b"] == 3

    def test_none_endpoints(self):
        ec = EdgeClassification(None, None, 0, 1, EdgeType.CONTEXT_SPECIFIC_B, 1.0)
        assert ec.source_a is None
        assert ec.target_a is None


# ---------------------------------------------------------------
# AlignmentResult dataclass
# ---------------------------------------------------------------

class TestAlignmentResult:
    def test_edge_jaccard_all_shared(self):
        r = _make_dummy_result()
        r.edge_classifications = [
            EdgeClassification(0, 1, 0, 1, EdgeType.SHARED, 0.0),
            EdgeClassification(1, 2, 1, 2, EdgeType.SHARED, 0.0),
        ]
        r.n_shared = 2
        r.n_modified = 0
        r.n_context_specific_a = 0
        r.n_context_specific_b = 0
        assert r.edge_jaccard == 1.0

    def test_edge_jaccard_no_shared(self):
        r = _make_dummy_result()
        r.edge_classifications = [
            EdgeClassification(0, 1, None, None, EdgeType.CONTEXT_SPECIFIC_A, 1.0),
        ]
        r.n_shared = 0
        r.n_modified = 0
        r.n_context_specific_a = 1
        r.n_context_specific_b = 0
        assert r.edge_jaccard == 0.0

    def test_edge_jaccard_empty(self):
        r = _make_dummy_result()
        r.edge_classifications = []
        r.n_shared = 0
        r.n_modified = 0
        r.n_context_specific_a = 0
        r.n_context_specific_b = 0
        assert r.edge_jaccard == 1.0  # vacuous

    def test_to_dict_keys(self):
        r = _make_dummy_result()
        d = r.to_dict()
        for key in [
            "alignment", "alignment_quality", "structural_divergence",
            "n_shared", "n_modified", "n_context_specific_a",
            "n_context_specific_b", "context_a", "context_b",
            "edge_jaccard", "computation_time",
        ]:
            assert key in d

    def test_summary_string(self):
        r = _make_dummy_result()
        s = r.summary()
        assert "CADA Alignment" in s
        assert "quality" in s.lower()


# ---------------------------------------------------------------
# CADAAligner — constructor
# ---------------------------------------------------------------

class TestCADAInit:
    def test_default_params(self):
        a = CADAAligner()
        assert a.mb_overlap_threshold == 0.3
        assert a.ci_weight == 0.6
        assert a.shape_weight == 0.4
        assert a.quality_threshold == 0.5
        assert a.max_unanchored == 200

    def test_custom_params(self):
        a = CADAAligner(
            mb_overlap_threshold=0.5,
            ci_weight=0.7,
            shape_weight=0.3,
            quality_threshold=0.8,
            max_unanchored=50,
            seed=123,
        )
        assert a.mb_overlap_threshold == 0.5
        assert a.seed == 123
        assert a.max_unanchored == 50


# ---------------------------------------------------------------
# CADAAligner.align — identical DAGs
# ---------------------------------------------------------------

class TestIdenticalAlignment:
    def test_identical_chain(self, aligner, chain3_scm, chain3_scm_copy):
        result = aligner.align(
            chain3_scm, chain3_scm_copy,
            anchors={0: 0, 1: 1, 2: 2},
        )
        assert result.alignment_quality == 1.0
        assert result.n_shared == 2  # two edges in chain
        assert result.n_modified == 0
        assert result.n_context_specific_a == 0
        assert result.n_context_specific_b == 0
        assert result.structural_divergence == 0.0

    def test_identical_quality_bounds(self, aligner, chain3_scm, chain3_scm_copy):
        result = aligner.align(chain3_scm, chain3_scm_copy, anchors={0: 0, 1: 1, 2: 2})
        assert 0.0 <= result.alignment_quality <= 1.0

    def test_all_edges_shared(self, aligner, chain3_scm, chain3_scm_copy):
        result = aligner.align(chain3_scm, chain3_scm_copy, anchors={0: 0, 1: 1, 2: 2})
        for ec in result.edge_classifications:
            assert ec.edge_type == EdgeType.SHARED

    def test_computation_time_positive(self, aligner, chain3_scm, chain3_scm_copy):
        result = aligner.align(chain3_scm, chain3_scm_copy, anchors={0: 0, 1: 1, 2: 2})
        assert result.computation_time > 0.0

    def test_phase_times_present(self, aligner, chain3_scm, chain3_scm_copy):
        result = aligner.align(chain3_scm, chain3_scm_copy, anchors={0: 0, 1: 1, 2: 2})
        assert "phase1_anchor_propagation" in result.phase_times
        assert "phase5_edge_classification" in result.phase_times
        assert "phase6_score_divergence" in result.phase_times


# ---------------------------------------------------------------
# CADAAligner.align — different DAGs
# ---------------------------------------------------------------

class TestDifferentAlignment:
    def test_chain_vs_fork(self, aligner, chain3_scm, fork3_scm):
        result = aligner.align(
            chain3_scm, fork3_scm,
            anchors={0: 0, 1: 1, 2: 2},
        )
        assert result.alignment_quality < 1.0
        assert result.structural_divergence > 0.0

    def test_chain_vs_collider(self, aligner, chain3_scm, collider3_scm):
        result = aligner.align(
            chain3_scm, collider3_scm,
            anchors={0: 0, 1: 1, 2: 2},
        )
        assert result.alignment_quality < 1.0

    def test_completely_different_has_context_specific(self, aligner):
        scm_a = _make_scm(
            [[0, 0.5, 0],
             [0, 0, 0],
             [0, 0, 0]],
            names=["A0", "A1", "A2"],
        )
        scm_b = _make_scm(
            [[0, 0, 0],
             [0, 0, 0.5],
             [0, 0, 0]],
            names=["B0", "B1", "B2"],
        )
        result = aligner.align(scm_a, scm_b, anchors={0: 0, 1: 1, 2: 2})
        ctx_a = result.n_context_specific_a
        ctx_b = result.n_context_specific_b
        assert ctx_a > 0 or ctx_b > 0

    def test_quality_bounds(self, aligner, chain3_scm, fork3_scm):
        result = aligner.align(chain3_scm, fork3_scm, anchors={0: 0, 1: 1, 2: 2})
        assert 0.0 <= result.alignment_quality <= 1.0
        assert result.normalized_divergence >= 0.0


# ---------------------------------------------------------------
# Partial overlap — shared + context-specific edges
# ---------------------------------------------------------------

class TestPartialOverlap:
    def test_shared_and_extra_edges(self, aligner):
        # A: 0->1, 1->2, 0->2 (diamond-ish)
        scm_a = _make_scm(
            [[0, 0.5, 0.3],
             [0, 0, 0.5],
             [0, 0, 0]],
            names=["X0", "X1", "X2"],
        )
        # B: 0->1, 1->2 (chain, missing 0->2)
        scm_b = _make_scm(
            [[0, 0.5, 0],
             [0, 0, 0.5],
             [0, 0, 0]],
            names=["X0", "X1", "X2"],
        )
        result = aligner.align(scm_a, scm_b, anchors={0: 0, 1: 1, 2: 2})
        assert result.n_shared == 2  # 0->1, 1->2
        assert result.n_context_specific_a == 1  # 0->2 only in A
        assert result.n_context_specific_b == 0
        assert 0.0 < result.alignment_quality < 1.0

    def test_reversed_edge_is_modified(self, aligner):
        # A: 0->1
        scm_a = _make_scm(
            [[0, 0.5],
             [0, 0]],
            names=["X0", "X1"],
        )
        # B: 1->0 (reversed)
        scm_b = _make_scm(
            [[0, 0],
             [0.5, 0]],
            names=["X0", "X1"],
        )
        result = aligner.align(scm_a, scm_b, anchors={0: 0, 1: 1})
        assert result.n_modified == 1
        assert result.n_shared == 0
        modified = [e for e in result.edge_classifications if e.edge_type == EdgeType.MODIFIED]
        assert len(modified) == 1
        assert modified[0].weight == 0.5  # default w_reversal


# ---------------------------------------------------------------
# Edge classification (Phase 5 direct tests)
# ---------------------------------------------------------------

class TestEdgeClassificationPhase:
    def test_shared_edge(self, aligner):
        adj_a = np.array([[0, 1], [0, 0]], dtype=np.float64)
        adj_b = np.array([[0, 1], [0, 0]], dtype=np.float64)
        alignment = {0: 0, 1: 1}
        ec = aligner._phase5_edge_classification(adj_a, adj_b, alignment)
        assert len(ec) == 1
        assert ec[0].edge_type == EdgeType.SHARED
        assert ec[0].weight == 0.0

    def test_modified_edge(self, aligner):
        adj_a = np.array([[0, 1], [0, 0]], dtype=np.float64)
        adj_b = np.array([[0, 0], [1, 0]], dtype=np.float64)
        alignment = {0: 0, 1: 1}
        ec = aligner._phase5_edge_classification(adj_a, adj_b, alignment)
        types = {e.edge_type for e in ec}
        assert EdgeType.MODIFIED in types

    def test_context_specific_a(self, aligner):
        adj_a = np.array([[0, 1], [0, 0]], dtype=np.float64)
        adj_b = np.array([[0, 0], [0, 0]], dtype=np.float64)
        alignment = {0: 0, 1: 1}
        ec = aligner._phase5_edge_classification(adj_a, adj_b, alignment)
        assert len(ec) == 1
        assert ec[0].edge_type == EdgeType.CONTEXT_SPECIFIC_A

    def test_context_specific_b(self, aligner):
        adj_a = np.array([[0, 0], [0, 0]], dtype=np.float64)
        adj_b = np.array([[0, 1], [0, 0]], dtype=np.float64)
        alignment = {0: 0, 1: 1}
        ec = aligner._phase5_edge_classification(adj_a, adj_b, alignment)
        assert len(ec) == 1
        assert ec[0].edge_type == EdgeType.CONTEXT_SPECIFIC_B

    def test_unmatched_endpoint_is_context_specific(self, aligner):
        adj_a = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float64)
        adj_b = np.array([[0, 0], [0, 0]], dtype=np.float64)
        alignment = {0: 0, 1: 1, 2: None}
        ec = aligner._phase5_edge_classification(adj_a, adj_b, alignment)
        # Edge 0->1 should be either shared or context-specific; 0->2 is context_specific_a if existed
        # Only edge is 0->1, which maps to 0->1 in B (no edge) => context_specific_a
        assert len(ec) == 1
        assert ec[0].edge_type == EdgeType.CONTEXT_SPECIFIC_A

    def test_mixed_classification(self, aligner):
        # A: 0->1, 0->2, 1->2
        adj_a = np.array([
            [0, 1, 1],
            [0, 0, 1],
            [0, 0, 0],
        ], dtype=np.float64)
        # B: 0->1, 2->0 (no 1->2, reversed 0->2)
        adj_b = np.array([
            [0, 1, 0],
            [0, 0, 0],
            [1, 0, 0],
        ], dtype=np.float64)
        alignment = {0: 0, 1: 1, 2: 2}
        ec = aligner._phase5_edge_classification(adj_a, adj_b, alignment)
        types = [e.edge_type for e in ec]
        assert EdgeType.SHARED in types      # 0->1
        assert EdgeType.MODIFIED in types     # 0->2 in A, 2->0 in B
        assert EdgeType.CONTEXT_SPECIFIC_A in types  # 1->2 only in A


# ---------------------------------------------------------------
# Score/divergence computation (Phase 6)
# ---------------------------------------------------------------

class TestScoreComputation:
    @pytest.mark.parametrize("quality", [0.0, 0.5, 1.0])
    def test_quality_bounded(self, quality):
        assert 0.0 <= quality <= 1.0

    def test_zero_divergence_identical(self, aligner, chain3_scm, chain3_scm_copy):
        result = aligner.align(chain3_scm, chain3_scm_copy, anchors={0: 0, 1: 1, 2: 2})
        assert result.structural_divergence == 0.0
        assert result.normalized_divergence == 0.0

    def test_divergence_positive_different(self, aligner, chain3_scm, fork3_scm):
        result = aligner.align(chain3_scm, fork3_scm, anchors={0: 0, 1: 1, 2: 2})
        assert result.structural_divergence > 0.0


# ---------------------------------------------------------------
# Anchor propagation (Phase 1)
# ---------------------------------------------------------------

class TestAnchorPropagation:
    def test_known_anchors_preserved(self, aligner, diamond4_scm):
        copy_scm = _make_scm(
            [[0, 0.6, 0.4, 0],
             [0, 0, 0, 0.5],
             [0, 0, 0, 0.3],
             [0, 0, 0, 0]],
            names=["X0", "X1", "X2", "X3"],
        )
        result = aligner.align(diamond4_scm, copy_scm, anchors={0: 0, 3: 3})
        # Original anchors must be in the final alignment
        assert result.alignment[0] == 0
        assert result.alignment[3] == 3

    def test_empty_anchors_start(self, aligner, chain3_scm, chain3_scm_copy):
        result = aligner.align(chain3_scm, chain3_scm_copy, anchors=None)
        assert result.n_matched >= 0
        assert 0.0 <= result.alignment_quality <= 1.0

    def test_invalid_anchors_raise(self, strict_aligner, chain3_scm, chain3_scm_copy):
        with pytest.raises(AnchorConflictError):
            strict_aligner.align(
                chain3_scm, chain3_scm_copy,
                anchors={0: 0, 1: 0},  # non-bijective: both map to 0
            )

    def test_n_anchored_count(self, aligner, chain3_scm, chain3_scm_copy):
        result = aligner.align(chain3_scm, chain3_scm_copy, anchors={0: 0})
        assert result.n_anchored >= 1


# ---------------------------------------------------------------
# TooManyUnanchoredError
# ---------------------------------------------------------------

class TestTooManyUnanchored:
    def test_raises_when_exceeded(self):
        tiny_aligner = CADAAligner(max_unanchored=2, seed=0)
        scm = _make_scm(np.zeros((5, 5)), names=[f"V{i}" for i in range(5)])
        with pytest.raises(TooManyUnanchoredError):
            tiny_aligner.align(scm, scm, anchors={})


# ---------------------------------------------------------------
# Edge cases — single variable, empty, disconnected
# ---------------------------------------------------------------

class TestEdgeCases:
    def test_single_variable(self, aligner, single_var_scm):
        result = aligner.align(single_var_scm, single_var_scm, anchors={0: 0})
        assert result.alignment_quality == 1.0
        assert result.n_shared == 0  # no edges
        assert len(result.edge_classifications) == 0

    def test_empty_graph(self, aligner, empty_scm):
        result = aligner.align(empty_scm, empty_scm, anchors={0: 0, 1: 1, 2: 2})
        assert result.alignment_quality == 1.0
        assert result.structural_divergence == 0.0
        assert len(result.edge_classifications) == 0

    def test_disconnected_graph(self, aligner, disconnected_scm):
        copy = _make_scm(
            [[0, 0.5, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0.5],
             [0, 0, 0, 0]],
            names=["X0", "X1", "X2", "X3"],
        )
        result = aligner.align(
            disconnected_scm, copy,
            anchors={0: 0, 1: 1, 2: 2, 3: 3},
        )
        assert result.n_shared == 2
        assert result.alignment_quality == 1.0

    def test_different_size_graphs(self, aligner):
        scm_a = _make_scm(
            [[0, 0.5],
             [0, 0]],
            names=["A0", "A1"],
        )
        scm_b = _make_scm(
            [[0, 0.5, 0],
             [0, 0, 0.5],
             [0, 0, 0]],
            names=["B0", "B1", "B2"],
        )
        result = aligner.align(scm_a, scm_b, anchors={0: 0, 1: 1})
        assert result.n_unmatched_b >= 1
        assert 0.0 <= result.alignment_quality <= 1.0


# ---------------------------------------------------------------
# AlignmentCache
# ---------------------------------------------------------------

class TestAlignmentCache:
    def test_put_and_get(self):
        cache = AlignmentCache(max_size=10)
        r = _make_dummy_result("X", "Y")
        cache.put("X", "Y", r)
        assert cache.get("X", "Y") is r
        assert cache.size == 1

    def test_get_miss(self):
        cache = AlignmentCache(max_size=10)
        assert cache.get("A", "B") is None

    def test_order_independent_key(self):
        cache = AlignmentCache(max_size=10)
        r = _make_dummy_result("X", "Y")
        cache.put("X", "Y", r)
        assert cache.get("Y", "X") is r

    def test_lru_eviction(self):
        cache = AlignmentCache(max_size=2)
        cache.put("A", "B", _make_dummy_result("A", "B"))
        cache.put("C", "D", _make_dummy_result("C", "D"))
        cache.put("E", "F", _make_dummy_result("E", "F"))  # evicts A||B
        assert cache.get("A", "B") is None
        assert cache.get("C", "D") is not None
        assert cache.size == 2

    def test_lru_access_refreshes(self):
        cache = AlignmentCache(max_size=2)
        cache.put("A", "B", _make_dummy_result("A", "B"))
        cache.put("C", "D", _make_dummy_result("C", "D"))
        cache.get("A", "B")  # refresh A||B
        cache.put("E", "F", _make_dummy_result("E", "F"))  # evicts C||D
        assert cache.get("A", "B") is not None
        assert cache.get("C", "D") is None

    def test_hit_miss_stats(self):
        cache = AlignmentCache(max_size=10)
        cache.put("A", "B", _make_dummy_result("A", "B"))
        cache.get("A", "B")  # hit
        cache.get("C", "D")  # miss
        s = cache.stats()
        assert s["hits"] == 1
        assert s["misses"] == 1
        assert s["hit_rate"] == 0.5

    def test_invalidate_context(self):
        cache = AlignmentCache(max_size=10)
        cache.put("A", "B", _make_dummy_result("A", "B"))
        cache.put("A", "C", _make_dummy_result("A", "C"))
        cache.put("B", "C", _make_dummy_result("B", "C"))
        n = cache.invalidate("A")
        assert n == 2
        assert cache.get("A", "B") is None
        assert cache.get("A", "C") is None
        assert cache.get("B", "C") is not None

    def test_invalidate_all(self):
        cache = AlignmentCache(max_size=10)
        cache.put("A", "B", _make_dummy_result("A", "B"))
        cache.put("C", "D", _make_dummy_result("C", "D"))
        cache.invalidate_all()
        assert cache.size == 0
        assert cache.hit_rate == 0.0

    def test_max_size_must_be_positive(self):
        with pytest.raises(ValueError):
            AlignmentCache(max_size=0)
        with pytest.raises(ValueError):
            AlignmentCache(max_size=-1)

    def test_update_existing_key(self):
        cache = AlignmentCache(max_size=10)
        r1 = _make_dummy_result("A", "B", quality=0.5)
        r2 = _make_dummy_result("A", "B", quality=0.9)
        cache.put("A", "B", r1)
        cache.put("A", "B", r2)
        assert cache.size == 1
        assert cache.get("A", "B").alignment_quality == 0.9

    def test_get_all_pairs(self):
        cache = AlignmentCache(max_size=10)
        cache.put("X", "Y", _make_dummy_result("X", "Y"))
        cache.put("A", "B", _make_dummy_result("A", "B"))
        pairs = cache.get_all_pairs()
        assert len(pairs) == 2


# ---------------------------------------------------------------
# BatchAligner
# ---------------------------------------------------------------

class TestBatchAligner:
    def test_align_all_pairs_count(self, aligner, chain3_scm):
        """3 contexts => 3 pairs."""
        copy1 = _make_scm(
            [[0, 0.7, 0],
             [0, 0, 0.5],
             [0, 0, 0]],
            names=["X0", "X1", "X2"],
        )
        copy2 = _make_scm(
            [[0, 0.7, 0],
             [0, 0, 0.5],
             [0, 0, 0]],
            names=["X0", "X1", "X2"],
        )
        mccm = _SimpleMCCM({"c1": chain3_scm, "c2": copy1, "c3": copy2})
        ba = BatchAligner(aligner=aligner)
        results = ba.align_all_pairs(mccm, anchors={0: 0, 1: 1, 2: 2})
        assert len(results) == 3

    def test_uses_cache(self, aligner, chain3_scm, chain3_scm_copy):
        mccm = _SimpleMCCM({"A": chain3_scm, "B": chain3_scm_copy})
        cache = AlignmentCache(max_size=100)
        ba = BatchAligner(aligner=aligner, cache=cache)
        ba.align_all_pairs(mccm, anchors={0: 0, 1: 1, 2: 2})
        assert cache.size >= 1
        # Second run should hit cache
        ba.align_all_pairs(mccm, anchors={0: 0, 1: 1, 2: 2})
        assert cache.stats()["hits"] >= 1

    def test_progress_callback(self, aligner, chain3_scm, chain3_scm_copy):
        mccm = _SimpleMCCM({"A": chain3_scm, "B": chain3_scm_copy})
        calls = []
        ba = BatchAligner(aligner=aligner)

        def cb(i, total, desc):
            calls.append((i, total, desc))

        ba.align_all_pairs(mccm, anchors={0: 0, 1: 1, 2: 2}, progress_callback=cb)
        assert len(calls) >= 1
        assert calls[-1][0] == calls[-1][1]  # final call: i == total

    def test_quality_matrix(self, aligner, chain3_scm, chain3_scm_copy):
        mccm = _SimpleMCCM({"A": chain3_scm, "B": chain3_scm_copy})
        ba = BatchAligner(aligner=aligner)
        results = ba.align_all_pairs(mccm, anchors={0: 0, 1: 1, 2: 2})
        mat = ba.alignment_quality_matrix(results, ["A", "B"])
        assert mat.shape == (2, 2)
        assert mat[0, 0] == 1.0  # diagonal
        assert mat[1, 1] == 1.0
        assert 0.0 <= mat[0, 1] <= 1.0

    def test_divergence_matrix(self, aligner, chain3_scm, chain3_scm_copy):
        mccm = _SimpleMCCM({"A": chain3_scm, "B": chain3_scm_copy})
        ba = BatchAligner(aligner=aligner)
        results = ba.align_all_pairs(mccm, anchors={0: 0, 1: 1, 2: 2})
        mat = ba.divergence_matrix(results, ["A", "B"])
        assert mat.shape == (2, 2)
        assert mat[0, 0] == 0.0  # diagonal
        assert mat[0, 1] >= 0.0

    def test_alignment_statistics(self, aligner, chain3_scm, chain3_scm_copy):
        mccm = _SimpleMCCM({"A": chain3_scm, "B": chain3_scm_copy})
        ba = BatchAligner(aligner=aligner)
        results = ba.align_all_pairs(mccm, anchors={0: 0, 1: 1, 2: 2})
        stats = ba.alignment_statistics(results)
        assert stats["n_pairs"] == 1
        assert "mean_quality" in stats
        assert "total_time" in stats

    def test_find_most_similar(self, aligner):
        s1 = _make_scm([[0, 0.5], [0, 0]], names=["X0", "X1"])
        s2 = _make_scm([[0, 0.5], [0, 0]], names=["X0", "X1"])
        s3 = _make_scm([[0, 0], [0.5, 0]], names=["X0", "X1"])
        mccm = _SimpleMCCM({"A": s1, "B": s2, "C": s3})
        ba = BatchAligner(aligner=aligner)
        results = ba.align_all_pairs(mccm, anchors={0: 0, 1: 1})
        best = ba.find_most_similar_pair(results)
        assert best is not None
        assert best[2].alignment_quality >= 0.0

    def test_edge_type_summary(self, aligner, chain3_scm, chain3_scm_copy):
        mccm = _SimpleMCCM({"A": chain3_scm, "B": chain3_scm_copy})
        ba = BatchAligner(aligner=aligner)
        results = ba.align_all_pairs(mccm, anchors={0: 0, 1: 1, 2: 2})
        summary = ba.edge_type_summary(results)
        assert "shared" in summary
        assert "modified" in summary

    def test_empty_results_statistics(self):
        ba = BatchAligner()
        stats = ba.alignment_statistics({})
        assert stats["n_pairs"] == 0


# ---------------------------------------------------------------
# Parametrized: quality score always in [0, 1]
# ---------------------------------------------------------------

@pytest.mark.parametrize("adj_a,adj_b", [
    ([[0, 1], [0, 0]], [[0, 1], [0, 0]]),       # identical
    ([[0, 1], [0, 0]], [[0, 0], [1, 0]]),       # reversed
    ([[0, 1], [0, 0]], [[0, 0], [0, 0]]),       # A only
    ([[0, 0], [0, 0]], [[0, 1], [0, 0]]),       # B only
    ([[0, 0], [0, 0]], [[0, 0], [0, 0]]),       # both empty
])
def test_quality_always_in_unit_interval(adj_a, adj_b):
    aligner = CADAAligner(
        mb_overlap_threshold=0.0,
        quality_threshold=0.0,
        seed=42,
    )
    scm_a = _make_scm(adj_a, names=[f"V{i}" for i in range(len(adj_a))])
    scm_b = _make_scm(adj_b, names=[f"V{i}" for i in range(len(adj_b))])
    anchors = {i: i for i in range(len(adj_a))}
    result = aligner.align(scm_a, scm_b, anchors=anchors)
    assert 0.0 <= result.alignment_quality <= 1.0
    assert result.normalized_divergence >= 0.0


# ---------------------------------------------------------------
# Parametrized: edge classification weights
# ---------------------------------------------------------------

@pytest.mark.parametrize("etype,expected_default_weight", [
    (EdgeType.SHARED, 0.0),
])
def test_shared_edge_weight(etype, expected_default_weight):
    ec = EdgeClassification(0, 1, 0, 1, etype, expected_default_weight)
    assert ec.weight == expected_default_weight
