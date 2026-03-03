"""Unit tests for cpa.alignment.hungarian module."""

from __future__ import annotations

import numpy as np
import pytest

from cpa.alignment.hungarian import (
    MatchResult,
    PaddedHungarianSolver,
    QualityFilter,
)


# ── helpers ──────────────────────────────────────────────────────────
def _solver(min_score: float = 0.0, **kw) -> PaddedHungarianSolver:
    return PaddedHungarianSolver(
        quality_filter=QualityFilter(min_score=min_score), **kw
    )


# ── MatchResult properties ──────────────────────────────────────────
class TestMatchResult:
    def test_n_matched(self):
        r = MatchResult([0, 1], [1, 0], [0.9, 0.8], [], [], 1.7, 0.85, 2, 2)
        assert r.n_matched == 2

    def test_match_ratio(self):
        r = MatchResult([0], [0], [1.0], [1], [], 1.0, 1.0, 2, 1)
        assert r.match_ratio == pytest.approx(0.5)

    def test_match_ratio_empty(self):
        r = MatchResult([], [], [], [], [], 0.0, 0.0, 0, 0)
        assert r.match_ratio == 0.0

    def test_as_dict(self):
        r = MatchResult([0], [1], [0.9], [1], [], 0.9, 0.9, 2, 2)
        d = r.as_dict()
        assert d == {0: 1, 1: None}

    def test_inverse_dict(self):
        r = MatchResult([0], [1], [0.9], [], [0], 0.9, 0.9, 1, 2)
        d = r.inverse_dict()
        assert d == {1: 0, 0: None}

    def test_to_dict_keys(self):
        r = MatchResult([0], [0], [1.0], [], [], 1.0, 1.0, 1, 1)
        d = r.to_dict()
        expected_keys = {
            "row_indices", "col_indices", "scores",
            "unmatched_rows", "unmatched_cols",
            "total_score", "quality",
            "n_original_rows", "n_original_cols",
            "n_matched", "match_ratio",
        }
        assert set(d.keys()) == expected_keys
        assert d["n_matched"] == 1
        assert d["match_ratio"] == pytest.approx(1.0)


# ── QualityFilter ───────────────────────────────────────────────────
class TestQualityFilter:
    def test_invalid_min_score(self):
        with pytest.raises(ValueError):
            QualityFilter(min_score=1.5)

    def test_min_score_filters_low(self):
        qf = QualityFilter(min_score=0.5)
        kept_r, kept_c, kept_s, um_r, um_c = qf.filter(
            [0, 1], [0, 1], [0.8, 0.3], 2, 2
        )
        assert kept_r == [0]
        assert kept_s == [0.8]
        assert 1 in um_r
        assert 1 in um_c

    def test_relative_threshold(self):
        qf = QualityFilter(min_score=0.0, relative_threshold=0.6)
        kept_r, kept_c, kept_s, um_r, um_c = qf.filter(
            [0, 1, 2], [0, 1, 2], [1.0, 0.7, 0.4], 3, 3
        )
        # threshold = 0.6 * 1.0 = 0.6 → keeps 1.0 and 0.7
        assert len(kept_r) == 2
        assert 0.4 not in kept_s

    def test_empty_scores(self):
        qf = QualityFilter()
        kept_r, _, _, um_r, um_c = qf.filter([], [], [], 3, 2)
        assert kept_r == []
        assert um_r == [0, 1, 2]
        assert um_c == [0, 1]


# ── PaddedHungarianSolver ──────────────────────────────────────────
class TestPaddedHungarianSolver:
    # ── square matrix ──
    def test_identity_matrix(self):
        mat = np.eye(3)
        result = _solver().solve(mat)
        pairs = set(zip(result.row_indices, result.col_indices))
        assert pairs == {(0, 0), (1, 1), (2, 2)}
        assert result.total_score == pytest.approx(3.0)

    def test_square_optimal(self):
        mat = np.array([[1.0, 0.0], [0.0, 1.0]])
        result = _solver().solve(mat)
        assert set(zip(result.row_indices, result.col_indices)) == {(0, 0), (1, 1)}

    # ── non-square ──
    @pytest.mark.parametrize(
        "shape",
        [(3, 2), (2, 3)],
        ids=["tall", "wide"],
    )
    def test_non_square(self, shape):
        m, n = shape
        mat = np.ones((m, n), dtype=np.float64)
        result = _solver().solve(mat)
        assert result.n_matched == min(m, n)
        assert result.n_original_rows == m
        assert result.n_original_cols == n
        if m > n:
            assert len(result.unmatched_rows) == m - n
        else:
            assert len(result.unmatched_cols) == n - m

    def test_non_square_3x2_optimal(self):
        mat = np.array([[0.9, 0.1], [0.1, 0.9], [0.5, 0.5]])
        result = _solver().solve(mat)
        pairs = set(zip(result.row_indices, result.col_indices))
        assert (0, 0) in pairs
        assert (1, 1) in pairs
        assert result.n_matched == 2

    # ── padding ──
    def test_pad_to_square(self):
        solver = _solver()
        mat = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        padded, orig_m, orig_n = solver._pad_to_square(mat)
        assert padded.shape == (3, 3)
        assert orig_m == 2
        assert orig_n == 3
        np.testing.assert_array_equal(padded[:2, :3], mat)

    def test_pad_square_noop(self):
        solver = _solver()
        mat = np.eye(2)
        padded, m, n = solver._pad_to_square(mat)
        assert padded.shape == (2, 2)
        assert m == 2 and n == 2

    # ── edge cases ──
    def test_empty_matrix(self):
        mat = np.empty((0, 0))
        result = _solver().solve(mat)
        assert result.n_matched == 0
        assert result.total_score == 0.0
        assert result.quality == 0.0

    def test_single_element(self):
        mat = np.array([[0.8]])
        result = _solver().solve(mat)
        assert result.n_matched == 1
        assert result.row_indices == [0]
        assert result.col_indices == [0]
        assert result.scores[0] == pytest.approx(0.8)

    def test_all_zeros(self):
        mat = np.zeros((2, 2))
        result = _solver().solve(mat)
        for s in result.scores:
            assert s == pytest.approx(0.0)

    def test_all_nan(self):
        mat = np.full((2, 3), np.nan)
        result = _solver().solve(mat)
        assert result.n_matched == 0
        assert result.unmatched_rows == [0, 1]
        assert result.unmatched_cols == [0, 1, 2]

    def test_partial_nan(self):
        mat = np.array([[np.nan, 0.9], [0.8, np.nan]])
        result = _solver().solve(mat)
        pairs = set(zip(result.row_indices, result.col_indices))
        assert (0, 1) in pairs
        assert (1, 0) in pairs

    # ── equal scores → deterministic ──
    def test_equal_scores_deterministic(self):
        mat = np.ones((3, 3))
        r1 = _solver().solve(mat)
        r2 = _solver().solve(mat)
        assert r1.row_indices == r2.row_indices
        assert r1.col_indices == r2.col_indices

    # ── quality filtering via solver ──
    def test_quality_filter_removes_low(self):
        mat = np.array([[0.9, 0.1], [0.1, 0.3]])
        result = _solver(min_score=0.5).solve(mat)
        assert 0.3 not in result.scores
        assert 0.1 not in result.scores
        assert 0.9 in result.scores

    # ── constraints ──
    def test_must_match(self):
        mat = np.array([[0.9, 0.1], [0.1, 0.9]])
        result = _solver().solve_with_constraints(mat, must_match=[(0, 1)])
        pairs = set(zip(result.row_indices, result.col_indices))
        assert (0, 1) in pairs

    def test_cannot_match(self):
        mat = np.array([[0.9, 0.1], [0.1, 0.9]])
        result = _solver().solve_with_constraints(mat, cannot_match=[(0, 0)])
        pairs = set(zip(result.row_indices, result.col_indices))
        assert (0, 0) not in pairs

    def test_must_and_cannot_combined(self):
        mat = np.array([
            [0.9, 0.5, 0.1],
            [0.1, 0.8, 0.2],
            [0.3, 0.2, 0.7],
        ])
        result = _solver().solve_with_constraints(
            mat, must_match=[(0, 2)], cannot_match=[(1, 1)]
        )
        pairs = set(zip(result.row_indices, result.col_indices))
        assert (0, 2) in pairs
        assert (1, 1) not in pairs

    # ── minimize mode ──
    def test_minimize(self):
        mat = np.array([[1.0, 3.0], [3.0, 1.0]])
        solver = PaddedHungarianSolver(
            quality_filter=QualityFilter(min_score=0.0), maximize=False
        )
        result = solver.solve(mat)
        pairs = set(zip(result.row_indices, result.col_indices))
        assert pairs == {(0, 0), (1, 1)}

    # ── validation helpers ──
    def test_unmatched_sets_correct(self):
        mat = np.array([[0.9, 0.1], [0.1, 0.8], [0.5, 0.5]])
        result = _solver().solve(mat)
        matched_rows = set(result.row_indices)
        matched_cols = set(result.col_indices)
        assert set(result.unmatched_rows) == set(range(3)) - matched_rows
        assert set(result.unmatched_cols) == set(range(2)) - matched_cols


# ── parametrize: various matrix sizes ──
@pytest.mark.parametrize(
    "m,n",
    [(1, 1), (1, 5), (5, 1), (4, 4), (3, 7)],
    ids=["1x1", "1x5", "5x1", "4x4", "3x7"],
)
def test_solver_various_sizes(m, n):
    rng = np.random.default_rng(123)
    mat = rng.random((m, n))
    result = _solver().solve(mat)
    assert result.n_matched <= min(m, n)
    assert result.n_original_rows == m
    assert result.n_original_cols == n
    assert len(set(result.row_indices)) == result.n_matched
    assert len(set(result.col_indices)) == result.n_matched
