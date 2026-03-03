"""Unit tests for cpa.alignment.scoring module."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from cpa.alignment.scoring import (
    CIFingerprintScorer,
    MarkovBlanketOverlap,
)

# Try importing optional classes
try:
    from cpa.alignment.scoring import DistributionShapeSimilarity
except ImportError:
    DistributionShapeSimilarity = None

try:
    from cpa.alignment.scoring import ScoreMatrix
except ImportError:
    ScoreMatrix = None

try:
    from cpa.alignment.scoring import AnchorValidator
except ImportError:
    AnchorValidator = None


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------

@pytest.fixture
def identity_cov():
    """3x3 identity covariance (independent variables)."""
    return np.eye(3)


@pytest.fixture
def correlated_cov():
    """3x3 covariance with known correlation structure."""
    cov = np.array([
        [1.0, 0.5, 0.2],
        [0.5, 1.0, 0.3],
        [0.2, 0.3, 1.0],
    ])
    return cov


@pytest.fixture
def known_precision_cov():
    """4x4 covariance constructed from a known precision matrix.

    Precision matrix P has P[0,1]=-0.6, P[0,2]=0, P[0,3]=-0.2
    so partial_corr(0,1) = 0.6/sqrt(P[0,0]*P[1,1]) etc.
    """
    P = np.array([
        [ 2.0, -0.6,  0.0, -0.2],
        [-0.6,  1.5,  0.0,  0.0],
        [ 0.0,  0.0,  1.0,  0.0],
        [-0.2,  0.0,  0.0,  1.2],
    ])
    cov = np.linalg.inv(P)
    return cov, P


@pytest.fixture
def cosine_scorer():
    return CIFingerprintScorer(method="cosine")


@pytest.fixture
def spearman_scorer():
    return CIFingerprintScorer(method="spearman")


@pytest.fixture
def overlap_scorer():
    return CIFingerprintScorer(method="overlap", significance_threshold=0.1)


@pytest.fixture
def mb_overlap():
    return MarkovBlanketOverlap(overlap_threshold=0.3)


@pytest.fixture
def chain_adjacency():
    """Chain graph: 0 -> 1 -> 2 -> 3."""
    adj = np.zeros((4, 4))
    adj[0, 1] = 1
    adj[1, 2] = 1
    adj[2, 3] = 1
    return adj


@pytest.fixture
def fork_adjacency():
    """Fork: 1 <- 0 -> 2."""
    adj = np.zeros((3, 3))
    adj[0, 1] = 1
    adj[0, 2] = 1
    return adj


@pytest.fixture
def collider_adjacency():
    """Collider: 0 -> 2 <- 1."""
    adj = np.zeros((3, 3))
    adj[0, 2] = 1
    adj[1, 2] = 1
    return adj


# ---------------------------------------------------------------
# CIFingerprintScorer — constructor
# ---------------------------------------------------------------

class TestCIFingerprintScorerInit:

    def test_default_method(self):
        s = CIFingerprintScorer()
        assert s.method == "cosine"

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="Unknown method"):
            CIFingerprintScorer(method="pearson")

    @pytest.mark.parametrize("method", ["cosine", "spearman", "overlap"])
    def test_valid_methods(self, method):
        s = CIFingerprintScorer(method=method)
        assert s.method == method


# ---------------------------------------------------------------
# CIFingerprintScorer — precision / partial correlations
# ---------------------------------------------------------------

class TestPrecisionAndPartialCorrelations:

    def test_precision_of_identity(self, cosine_scorer, identity_cov):
        prec = cosine_scorer._compute_precision(identity_cov)
        assert_allclose(prec, np.eye(3), atol=1e-6)

    def test_partial_corr_identity_are_zero(self, cosine_scorer, identity_cov):
        """Independent variables should have zero partial correlations."""
        pc = cosine_scorer.partial_correlations(0, identity_cov)
        assert_allclose(pc[1:], 0.0, atol=1e-6)
        assert pc[0] == 0.0  # self-entry

    def test_partial_corr_from_known_precision(self, cosine_scorer, known_precision_cov):
        cov, P = known_precision_cov
        pc = cosine_scorer.partial_correlations(0, cov)

        # rho(0,1) = -P[0,1] / sqrt(P[0,0]*P[1,1]) = 0.6 / sqrt(2*1.5)
        expected_01 = 0.6 / np.sqrt(2.0 * 1.5)
        # rho(0,2) = 0 (P[0,2]=0)
        expected_02 = 0.0
        # rho(0,3) = -P[0,3] / sqrt(P[0,0]*P[3,3]) = 0.2 / sqrt(2*1.2)
        expected_03 = 0.2 / np.sqrt(2.0 * 1.2)

        assert_allclose(pc[1], expected_01, atol=1e-4)
        assert_allclose(pc[2], expected_02, atol=1e-4)
        assert_allclose(pc[3], expected_03, atol=1e-4)

    def test_partial_corr_range(self, cosine_scorer, correlated_cov):
        pc = cosine_scorer.partial_correlations(0, correlated_cov)
        assert np.all(pc >= -1.0) and np.all(pc <= 1.0)


# ---------------------------------------------------------------
# CIFingerprintScorer — cosine similarity
# ---------------------------------------------------------------

class TestCosineSimilarity:

    def test_parallel_vectors(self, cosine_scorer):
        a = np.array([1.0, 2.0, 3.0])
        assert cosine_scorer._cosine_similarity(a, a) == pytest.approx(1.0)

    def test_antiparallel_vectors(self, cosine_scorer):
        a = np.array([1.0, 2.0, 3.0])
        assert cosine_scorer._cosine_similarity(a, -a) == pytest.approx(0.0)

    def test_orthogonal_vectors(self, cosine_scorer):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cosine_scorer._cosine_similarity(a, b) == pytest.approx(0.5)

    def test_both_zero(self, cosine_scorer):
        z = np.zeros(3)
        assert cosine_scorer._cosine_similarity(z, z) == 1.0

    def test_one_zero(self, cosine_scorer):
        a = np.array([1.0, 2.0])
        z = np.zeros(2)
        assert cosine_scorer._cosine_similarity(a, z) == 0.0


# ---------------------------------------------------------------
# CIFingerprintScorer — spearman similarity
# ---------------------------------------------------------------

class TestSpearmanSimilarity:

    def test_identical_vectors(self, spearman_scorer):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        sim = spearman_scorer._spearman_similarity(a, a)
        assert sim == pytest.approx(1.0)

    def test_reversed_vectors(self, spearman_scorer):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([4.0, 3.0, 2.0, 1.0])
        sim = spearman_scorer._spearman_similarity(a, b)
        assert sim == pytest.approx(0.0, abs=1e-6)

    def test_fallback_short_vectors(self, spearman_scorer):
        """Vectors with < 3 elements should fall back to cosine."""
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        sim = spearman_scorer._spearman_similarity(a, b)
        assert sim == pytest.approx(0.5)

    def test_range(self, spearman_scorer):
        rng = np.random.RandomState(42)
        a = rng.randn(10)
        b = rng.randn(10)
        sim = spearman_scorer._spearman_similarity(a, b)
        assert 0.0 <= sim <= 1.0


# ---------------------------------------------------------------
# CIFingerprintScorer — overlap similarity
# ---------------------------------------------------------------

class TestOverlapSimilarity:

    def test_identical_significant(self, overlap_scorer):
        a = np.array([0.5, 0.0, 0.3])
        sim = overlap_scorer._overlap_similarity(a, a)
        assert sim == pytest.approx(1.0)

    def test_disjoint_significant(self, overlap_scorer):
        a = np.array([0.5, 0.0, 0.0])
        b = np.array([0.0, 0.0, 0.5])
        sim = overlap_scorer._overlap_similarity(a, b)
        assert sim == pytest.approx(0.0)

    def test_partial_overlap(self, overlap_scorer):
        # sig_a = {0, 2}, sig_b = {0, 1}  => J = 1/3
        a = np.array([0.5, 0.0, 0.3])
        b = np.array([0.5, 0.3, 0.0])
        sim = overlap_scorer._overlap_similarity(a, b)
        assert sim == pytest.approx(1.0 / 3.0)

    def test_both_empty(self, overlap_scorer):
        a = np.array([0.0, 0.0])
        sim = overlap_scorer._overlap_similarity(a, a)
        assert sim == 1.0

    def test_one_empty(self, overlap_scorer):
        a = np.array([0.5, 0.0])
        b = np.array([0.0, 0.0])
        sim = overlap_scorer._overlap_similarity(a, b)
        assert sim == 0.0


# ---------------------------------------------------------------
# CIFingerprintScorer — fingerprint_similarity / score_pair
# ---------------------------------------------------------------

class TestFingerprintIntegration:

    def test_identical_covariances_high_similarity(self, cosine_scorer, correlated_cov):
        sim = cosine_scorer.score_pair(0, 0, correlated_cov, correlated_cov)
        assert sim > 0.99

    def test_identity_vs_correlated_lower(self, cosine_scorer, identity_cov, correlated_cov):
        sim = cosine_scorer.score_pair(0, 0, identity_cov, correlated_cov)
        # partial corrs of identity are ~0 vs non-zero → should be low
        assert sim < 0.8

    def test_alignment_mapping(self, cosine_scorer, correlated_cov):
        alignment = {0: 0, 1: 1, 2: 2}
        sim = cosine_scorer.score_pair(
            0, 0, correlated_cov, correlated_cov, alignment=alignment,
        )
        assert sim > 0.99

    def test_empty_alignment(self, cosine_scorer, correlated_cov):
        alignment = {0: None, 1: None}
        sim = cosine_scorer.fingerprint_similarity(
            np.array([0.5, 0.3]), np.array([0.5, 0.3]), alignment=alignment,
        )
        assert sim == 0.0


# ---------------------------------------------------------------
# CIFingerprintScorer — cache
# ---------------------------------------------------------------

class TestPrecisionCache:

    def test_cache_stores_result(self, cosine_scorer, correlated_cov):
        cosine_scorer._compute_precision(correlated_cov, cache_key="ctx1")
        assert "ctx1" in cosine_scorer._precision_cache

    def test_cache_returns_same_object(self, cosine_scorer, correlated_cov):
        p1 = cosine_scorer._compute_precision(correlated_cov, cache_key="ctx1")
        p2 = cosine_scorer._compute_precision(correlated_cov, cache_key="ctx1")
        assert p1 is p2

    def test_clear_cache(self, cosine_scorer, correlated_cov):
        cosine_scorer._compute_precision(correlated_cov, cache_key="ctx1")
        cosine_scorer.clear_cache()
        assert len(cosine_scorer._precision_cache) == 0

    def test_no_cache_without_key(self, cosine_scorer, correlated_cov):
        cosine_scorer._compute_precision(correlated_cov)
        assert len(cosine_scorer._precision_cache) == 0


# ---------------------------------------------------------------
# MarkovBlanketOverlap — construction & validation
# ---------------------------------------------------------------

class TestMarkovBlanketOverlapInit:

    def test_default_threshold(self):
        m = MarkovBlanketOverlap()
        assert m.overlap_threshold == 0.3

    def test_invalid_threshold(self):
        with pytest.raises(ValueError):
            MarkovBlanketOverlap(overlap_threshold=1.5)


# ---------------------------------------------------------------
# MarkovBlanketOverlap — markov_blanket_from_adjacency
# ---------------------------------------------------------------

class TestMarkovBlanketFromAdjacency:

    def test_chain_middle_node(self, chain_adjacency):
        # 0->1->2->3, MB(1) = {0 (parent), 2 (child)} no co-parents
        mb = MarkovBlanketOverlap.markov_blanket_from_adjacency(chain_adjacency, 1)
        assert mb == {0, 2}

    def test_chain_leaf_node(self, chain_adjacency):
        # MB(3) = {2 (parent)}
        mb = MarkovBlanketOverlap.markov_blanket_from_adjacency(chain_adjacency, 3)
        assert mb == {2}

    def test_chain_root_node(self, chain_adjacency):
        # MB(0) = {1 (child)}
        mb = MarkovBlanketOverlap.markov_blanket_from_adjacency(chain_adjacency, 0)
        assert mb == {1}

    def test_fork_center(self, fork_adjacency):
        # 0->1, 0->2. MB(0) = {1, 2}
        mb = MarkovBlanketOverlap.markov_blanket_from_adjacency(fork_adjacency, 0)
        assert mb == {1, 2}

    def test_fork_child(self, fork_adjacency):
        # MB(1) = {0 (parent)}
        mb = MarkovBlanketOverlap.markov_blanket_from_adjacency(fork_adjacency, 1)
        assert mb == {0}

    def test_collider_child(self, collider_adjacency):
        # 0->2<-1. MB(2) = {0, 1} (parents)
        mb = MarkovBlanketOverlap.markov_blanket_from_adjacency(collider_adjacency, 2)
        assert mb == {0, 1}

    def test_collider_parent(self, collider_adjacency):
        # MB(0) = {2 (child), 1 (co-parent of child 2)}
        mb = MarkovBlanketOverlap.markov_blanket_from_adjacency(collider_adjacency, 0)
        assert mb == {1, 2}

    def test_isolated_node(self):
        adj = np.zeros((3, 3))
        mb = MarkovBlanketOverlap.markov_blanket_from_adjacency(adj, 0)
        assert mb == set()


# ---------------------------------------------------------------
# MarkovBlanketOverlap — jaccard_index
# ---------------------------------------------------------------

class TestJaccardIndex:

    def test_identical_sets(self, mb_overlap):
        assert mb_overlap.jaccard_index({1, 2, 3}, {1, 2, 3}) == pytest.approx(1.0)

    def test_disjoint_sets(self, mb_overlap):
        assert mb_overlap.jaccard_index({1, 2}, {3, 4}) == pytest.approx(0.0)

    def test_empty_sets(self, mb_overlap):
        assert mb_overlap.jaccard_index(set(), set()) == 1.0

    def test_one_empty(self, mb_overlap):
        assert mb_overlap.jaccard_index({1}, set()) == pytest.approx(0.0)

    def test_partial_overlap(self, mb_overlap):
        # |{1,2} ∩ {2,3}| / |{1,2} ∪ {2,3}| = 1/3
        assert mb_overlap.jaccard_index({1, 2}, {2, 3}) == pytest.approx(1 / 3)

    def test_subset(self, mb_overlap):
        # |{1}∩{1,2}| / |{1}∪{1,2}| = 1/2
        assert mb_overlap.jaccard_index({1}, {1, 2}) == pytest.approx(0.5)


# ---------------------------------------------------------------
# MarkovBlanketOverlap — anchored_jaccard
# ---------------------------------------------------------------

class TestAnchoredJaccard:

    def test_full_anchors(self, mb_overlap):
        mb_a = {0, 1}
        mb_b = {10, 11}
        anchor = {0: 10, 1: 11}
        j = mb_overlap.anchored_jaccard(mb_a, mb_b, anchor)
        assert j == pytest.approx(1.0)

    def test_partial_anchors(self, mb_overlap):
        # mb_a={0,1,2}, only 0,1 anchored → translated_a={10,11}
        # mb_b={10,11,12}, restricted to anchored vals {10,11} → {10,11}
        mb_a = {0, 1, 2}
        mb_b = {10, 11, 12}
        anchor = {0: 10, 1: 11}
        j = mb_overlap.anchored_jaccard(mb_a, mb_b, anchor)
        assert j == pytest.approx(1.0)

    def test_no_anchored_overlap(self, mb_overlap):
        mb_a = {0}
        mb_b = {11}
        anchor = {0: 10, 1: 11}
        # translated_a = {10}, restricted_b = {11} → disjoint
        j = mb_overlap.anchored_jaccard(mb_a, mb_b, anchor)
        assert j == pytest.approx(0.0)

    def test_no_anchors_at_all(self, mb_overlap):
        mb_a = {0, 1}
        mb_b = {2, 3}
        anchor: dict = {}
        # Both translated_a and restricted_b empty → 1.0
        j = mb_overlap.anchored_jaccard(mb_a, mb_b, anchor)
        assert j == 1.0


# ---------------------------------------------------------------
# Optional: DistributionShapeSimilarity
# ---------------------------------------------------------------

@pytest.mark.skipif(DistributionShapeSimilarity is None, reason="DistributionShapeSimilarity not available")
class TestDistributionShapeSimilarity:

    def test_instantiation(self):
        dss = DistributionShapeSimilarity()
        assert dss is not None


# ---------------------------------------------------------------
# Optional: ScoreMatrix
# ---------------------------------------------------------------

@pytest.mark.skipif(ScoreMatrix is None, reason="ScoreMatrix not available")
class TestScoreMatrix:

    def test_instantiation(self):
        sm = ScoreMatrix()
        assert sm is not None


# ---------------------------------------------------------------
# Optional: AnchorValidator
# ---------------------------------------------------------------

@pytest.mark.skipif(AnchorValidator is None, reason="AnchorValidator not available")
class TestAnchorValidator:

    def test_instantiation(self):
        av = AnchorValidator()
        assert av is not None
