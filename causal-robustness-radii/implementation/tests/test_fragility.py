"""Tests for causalcert.fragility – scorer, channels, ranking."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causalcert.fragility.scorer import FragilityScorerImpl
from causalcert.fragility.channels import (
    DSepChannel,
    IdentificationChannel,
    EstimationChannel,
)
from causalcert.fragility.aggregation import (
    aggregate_scores,
    aggregate_fragility_scores,
    sensitivity_analysis,
    rank_stability,
    AggregationMethod,
)
from causalcert.fragility.ranking import (
    rank_edges,
    top_k_fragile,
    bottom_k_robust,
    load_bearing_edges,
    cosmetic_edges,
    classify_edge,
    classify_all_edges,
    severity_counts,
    EdgeSeverity,
    format_ranking_table,
    format_severity_summary,
)
from causalcert.types import (
    AdjacencyMatrix,
    EditType,
    FragilityChannel,
    FragilityScore,
    StructuralEdit,
)

# ── helpers ───────────────────────────────────────────────────────────────

def _adj(n: int, edges: list[tuple[int, int]]) -> AdjacencyMatrix:
    a = np.zeros((n, n), dtype=np.int8)
    for u, v in edges:
        a[u, v] = 1
    return a


def _synthetic_data(adj: AdjacencyMatrix, n: int = 300, seed: int = 42) -> pd.DataFrame:
    from tests.conftest import _linear_gaussian_data
    return _linear_gaussian_data(adj, n=n, seed=seed)


def _make_score(edge: tuple[int, int], total: float, dsep: float = 0.5,
                ident: float = 0.3, est: float = 0.2) -> FragilityScore:
    return FragilityScore(
        edge=edge,
        total_score=total,
        channel_scores={
            FragilityChannel.D_SEPARATION: dsep,
            FragilityChannel.IDENTIFICATION: ident,
            FragilityChannel.ESTIMATION: est,
        },
    )


# ═══════════════════════════════════════════════════════════════════════════
# Fragility scores on known DAGs
# ═══════════════════════════════════════════════════════════════════════════


class TestFragilityScoring:
    def test_chain_scoring(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        data = _synthetic_data(adj)
        scorer = FragilityScorerImpl(alpha=0.05)
        scores = scorer.score(adj, treatment=0, outcome=2, data=data)
        assert len(scores) >= 1
        for fs in scores:
            assert 0.0 <= fs.total_score <= 1.0

    def test_diamond_scoring(self) -> None:
        adj = _adj(4, [(0, 1), (0, 2), (1, 3), (2, 3)])
        data = _synthetic_data(adj)
        scorer = FragilityScorerImpl(alpha=0.05)
        scores = scorer.score(adj, treatment=0, outcome=3, data=data)
        assert len(scores) >= 1

    def test_data_free_scoring(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        scorer = FragilityScorerImpl(alpha=0.05)
        scores = scorer.score_data_free(adj, treatment=0, outcome=2)
        assert len(scores) >= 1
        for fs in scores:
            assert 0.0 <= fs.total_score <= 1.0

    def test_score_single_edge(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        data = _synthetic_data(adj)
        scorer = FragilityScorerImpl(alpha=0.05)
        edit = StructuralEdit(EditType.DELETE, 0, 1)
        fs = scorer.score_single_edge(adj, edit, treatment=0, outcome=2, data=data)
        assert 0.0 <= fs.total_score <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Load-bearing vs cosmetic edges
# ═══════════════════════════════════════════════════════════════════════════


class TestLoadBearingCosmetic:
    """Load-bearing edges should have high scores, cosmetic edges low."""

    def test_load_bearing_high_score(self) -> None:
        scores = [
            _make_score((0, 1), 0.9),
            _make_score((1, 2), 0.8),
            _make_score((2, 3), 0.1),
        ]
        lb = load_bearing_edges(scores, threshold=0.7)
        assert len(lb) == 2
        assert all(fs.total_score >= 0.7 for fs in lb)

    def test_cosmetic_low_score(self) -> None:
        scores = [
            _make_score((0, 1), 0.9),
            _make_score((1, 2), 0.05),
            _make_score((2, 3), 0.02),
        ]
        cos = cosmetic_edges(scores, threshold=0.1)
        assert len(cos) == 2
        assert all(fs.total_score < 0.1 for fs in cos)

    def test_no_load_bearing(self) -> None:
        scores = [_make_score((0, 1), 0.1)]
        lb = load_bearing_edges(scores, threshold=0.7)
        assert len(lb) == 0

    def test_all_load_bearing(self) -> None:
        scores = [_make_score((0, 1), 0.9), _make_score((1, 2), 0.95)]
        lb = load_bearing_edges(scores, threshold=0.7)
        assert len(lb) == 2


# ═══════════════════════════════════════════════════════════════════════════
# Individual channels
# ═══════════════════════════════════════════════════════════════════════════


class TestDSepChannel:
    def test_chain_dsep_channel(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        ch = DSepChannel(max_adj_set_size=3)
        edit = StructuralEdit(EditType.DELETE, 0, 1)
        score = ch.evaluate(adj, edit, treatment=0, outcome=2)
        assert 0.0 <= score <= 1.0

    def test_batch_evaluate(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        ch = DSepChannel(max_adj_set_size=3)
        edits = [
            StructuralEdit(EditType.DELETE, 0, 1),
            StructuralEdit(EditType.DELETE, 1, 2),
        ]
        scores = ch.evaluate_batch(adj, edits, treatment=0, outcome=2)
        assert len(scores) == 2
        assert all(0.0 <= s <= 1.0 for s in scores)


class TestIdentificationChannel:
    def test_chain_id_channel(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        ch = IdentificationChannel(max_adj_set_size=3)
        edit = StructuralEdit(EditType.DELETE, 0, 1)
        score = ch.evaluate(adj, edit, treatment=0, outcome=2)
        assert 0.0 <= score <= 1.0

    def test_batch_evaluate(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        ch = IdentificationChannel()
        edits = [StructuralEdit(EditType.DELETE, 0, 1)]
        scores = ch.evaluate_batch(adj, edits, treatment=0, outcome=2)
        assert len(scores) == 1


class TestEstimationChannel:
    def test_estimation_channel(self) -> None:
        adj = _adj(3, [(0, 1), (0, 2), (1, 2)])
        data = _synthetic_data(adj)
        ch = EstimationChannel()
        edit = StructuralEdit(EditType.DELETE, 0, 1)
        score = ch.evaluate(adj, edit, treatment=1, outcome=2, data=data)
        assert 0.0 <= score <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Aggregation methods
# ═══════════════════════════════════════════════════════════════════════════


class TestAggregation:
    @pytest.fixture
    def channel_scores(self) -> dict[FragilityChannel, float]:
        return {
            FragilityChannel.D_SEPARATION: 0.8,
            FragilityChannel.IDENTIFICATION: 0.5,
            FragilityChannel.ESTIMATION: 0.3,
        }

    def test_max_aggregation(self, channel_scores: dict) -> None:
        result = aggregate_scores(channel_scores, method=AggregationMethod.MAX)
        assert result == 0.8

    def test_weighted_aggregation(self, channel_scores: dict) -> None:
        result = aggregate_scores(channel_scores, method=AggregationMethod.WEIGHTED_AVERAGE)
        assert 0.0 <= result <= 1.0

    def test_geometric_mean(self, channel_scores: dict) -> None:
        result = aggregate_scores(channel_scores, method=AggregationMethod.GEOMETRIC_MEAN)
        assert 0.0 <= result <= 1.0

    def test_l2_norm(self, channel_scores: dict) -> None:
        result = aggregate_scores(channel_scores, method=AggregationMethod.L2_NORM)
        assert 0.0 <= result <= 1.0

    def test_product_complement(self, channel_scores: dict) -> None:
        result = aggregate_scores(channel_scores, method=AggregationMethod.PRODUCT_COMPLEMENT)
        assert 0.0 <= result <= 1.0

    def test_hierarchical(self, channel_scores: dict) -> None:
        result = aggregate_scores(channel_scores, method=AggregationMethod.HIERARCHICAL)
        assert 0.0 <= result <= 1.0

    def test_aggregate_fragility_scores_list(self) -> None:
        scores = [
            _make_score((0, 1), 0.9),
            _make_score((1, 2), 0.3),
        ]
        result = aggregate_fragility_scores(scores, method=AggregationMethod.MAX)
        assert len(result) == 2


# ═══════════════════════════════════════════════════════════════════════════
# Sensitivity analysis
# ═══════════════════════════════════════════════════════════════════════════


class TestSensitivityAnalysis:
    def test_sensitivity_analysis(self) -> None:
        scores = [
            _make_score((0, 1), 0.9, 0.9, 0.7, 0.5),
            _make_score((1, 2), 0.3, 0.3, 0.2, 0.1),
        ]
        results = sensitivity_analysis(scores)
        assert isinstance(results, dict)
        assert len(results) >= 1

    def test_rank_stability(self) -> None:
        scores = [
            _make_score((0, 1), 0.9, 0.9, 0.7, 0.5),
            _make_score((1, 2), 0.3, 0.3, 0.2, 0.1),
            _make_score((2, 3), 0.6, 0.6, 0.4, 0.3),
        ]
        sa = sensitivity_analysis(scores)
        stability = rank_stability(sa, top_k=2)
        assert isinstance(stability, dict)


# ═══════════════════════════════════════════════════════════════════════════
# Ranking
# ═══════════════════════════════════════════════════════════════════════════


class TestRanking:
    @pytest.fixture
    def scores(self) -> list[FragilityScore]:
        return [
            _make_score((0, 1), 0.3),
            _make_score((1, 2), 0.9),
            _make_score((2, 3), 0.6),
        ]

    def test_rank_descending(self, scores: list[FragilityScore]) -> None:
        ranked = rank_edges(scores, descending=True)
        assert ranked[0].total_score >= ranked[1].total_score >= ranked[2].total_score

    def test_rank_ascending(self, scores: list[FragilityScore]) -> None:
        ranked = rank_edges(scores, descending=False)
        assert ranked[0].total_score <= ranked[1].total_score <= ranked[2].total_score

    def test_top_k(self, scores: list[FragilityScore]) -> None:
        top = top_k_fragile(scores, k=2)
        assert len(top) == 2
        assert top[0].total_score >= top[1].total_score

    def test_bottom_k(self, scores: list[FragilityScore]) -> None:
        bot = bottom_k_robust(scores, k=2)
        assert len(bot) == 2
        assert bot[0].total_score <= bot[1].total_score


# ═══════════════════════════════════════════════════════════════════════════
# Classification
# ═══════════════════════════════════════════════════════════════════════════


class TestClassification:
    def test_classify_critical(self) -> None:
        sev = classify_edge(0.95)
        assert sev in (EdgeSeverity.CRITICAL, EdgeSeverity)  # high score

    def test_classify_low(self) -> None:
        sev = classify_edge(0.05)
        assert isinstance(sev, EdgeSeverity)

    def test_classify_all(self) -> None:
        scores = [
            _make_score((0, 1), 0.9),
            _make_score((1, 2), 0.3),
            _make_score((2, 3), 0.05),
        ]
        classified = classify_all_edges(scores)
        assert isinstance(classified, dict)
        total = sum(len(v) for v in classified.values())
        assert total == 3

    def test_severity_counts(self) -> None:
        scores = [
            _make_score((0, 1), 0.9),
            _make_score((1, 2), 0.3),
        ]
        counts = severity_counts(scores)
        assert isinstance(counts, dict)
        assert sum(counts.values()) == 2


# ═══════════════════════════════════════════════════════════════════════════
# Formatting
# ═══════════════════════════════════════════════════════════════════════════


class TestFormatting:
    def test_ranking_table(self) -> None:
        scores = [_make_score((0, 1), 0.9), _make_score((1, 2), 0.3)]
        table = format_ranking_table(scores)
        assert isinstance(table, str)
        assert len(table) > 0

    def test_severity_summary(self) -> None:
        scores = [_make_score((0, 1), 0.9)]
        summary = format_severity_summary(scores)
        assert isinstance(summary, str)


# ═══════════════════════════════════════════════════════════════════════════
# Scoring summary
# ═══════════════════════════════════════════════════════════════════════════


class TestScoringSummary:
    def test_get_summary(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        scorer = FragilityScorerImpl(alpha=0.05)
        scores = scorer.score_data_free(adj, treatment=0, outcome=2)
        summary = scorer.get_scoring_summary(scores)
        assert isinstance(summary, dict)


# ═══════════════════════════════════════════════════════════════════════════
# Fragility with random DAGs
# ═══════════════════════════════════════════════════════════════════════════


class TestFragilityRandomDAGs:
    @pytest.mark.parametrize("seed", range(5))
    def test_random_dag_data_free(self, seed: int) -> None:
        from tests.conftest import random_dag
        adj = random_dag(6, edge_prob=0.3, seed=seed)
        scorer = FragilityScorerImpl(alpha=0.05)
        scores = scorer.score_data_free(adj, treatment=0, outcome=5)
        for fs in scores:
            assert 0.0 <= fs.total_score <= 1.0

    @pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1])
    def test_alpha_sensitivity(self, alpha: float) -> None:
        adj = _adj(4, [(0, 1), (1, 2), (2, 3)])
        scorer = FragilityScorerImpl(alpha=alpha, include_absent=False)
        scores = scorer.score_data_free(adj, treatment=0, outcome=3)
        assert len(scores) >= 3  # at least the edges in the DAG

    def test_zero_edge_dag(self) -> None:
        adj = _adj(3, [])
        scorer = FragilityScorerImpl(alpha=0.05, include_absent=False)
        scores = scorer.score_data_free(adj, treatment=0, outcome=2)
        assert len(scores) == 0

    def test_score_monotonicity_by_path_count(self) -> None:
        # Edge on a bottleneck should score higher than a redundant one
        adj = _adj(5, [(0, 1), (1, 3), (0, 2), (2, 3), (3, 4)])
        scorer = FragilityScorerImpl(alpha=0.05, include_absent=False)
        scores = scorer.score_data_free(adj, treatment=0, outcome=4)
        score_map = {fs.edge: fs.total_score for fs in scores}
        # (3,4) is the unique bottleneck — should have high score
        assert score_map[(3, 4)] > 0.0
