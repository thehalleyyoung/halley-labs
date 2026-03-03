"""
Hungarian algorithm wrapper and utilities for optimal bipartite matching.

Wraps scipy.optimize.linear_sum_assignment with support for:
    - Non-square matrices (padding)
    - Quality-based post-filtering
    - Structured match results with confidence scores
    - Edge cases: empty matrices, all-negative scores, ties

Used by CADA Phase 4 (ALG1) for optimal variable matching.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

_EPSILON = 1e-12
_NEG_INF_PROXY = -1e9  # proxy for -infinity in padding


# ===================================================================
#  MatchResult
# ===================================================================
@dataclass
class MatchResult:
    """Structured result of bipartite matching.

    Attributes
    ----------
    row_indices : list of int
        Matched row indices.
    col_indices : list of int
        Matched column indices (aligned with row_indices).
    scores : list of float
        Match scores for each pair.
    unmatched_rows : list of int
        Row indices that were not matched (assigned to bot/None).
    unmatched_cols : list of int
        Column indices that were not matched.
    total_score : float
        Sum of all match scores.
    quality : float
        Mean of matched scores (0 if no matches).
    n_original_rows : int
        Number of rows in the original (unpadded) matrix.
    n_original_cols : int
        Number of columns in the original (unpadded) matrix.
    """

    row_indices: List[int]
    col_indices: List[int]
    scores: List[float]
    unmatched_rows: List[int]
    unmatched_cols: List[int]
    total_score: float
    quality: float
    n_original_rows: int
    n_original_cols: int

    @property
    def n_matched(self) -> int:
        """Number of matched pairs."""
        return len(self.row_indices)

    @property
    def match_ratio(self) -> float:
        """Ratio of matched to total variables."""
        total = max(self.n_original_rows, self.n_original_cols)
        return self.n_matched / total if total > 0 else 0.0

    def as_dict(self) -> Dict[int, Optional[int]]:
        """Return mapping from row indices to column indices.

        Unmatched rows map to None.
        """
        result: Dict[int, Optional[int]] = {}
        for r, c in zip(self.row_indices, self.col_indices):
            result[r] = c
        for r in self.unmatched_rows:
            result[r] = None
        return result

    def inverse_dict(self) -> Dict[int, Optional[int]]:
        """Return mapping from column indices to row indices.

        Unmatched columns map to None.
        """
        result: Dict[int, Optional[int]] = {}
        for r, c in zip(self.row_indices, self.col_indices):
            result[c] = r
        for c in self.unmatched_cols:
            result[c] = None
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "row_indices": self.row_indices,
            "col_indices": self.col_indices,
            "scores": self.scores,
            "unmatched_rows": self.unmatched_rows,
            "unmatched_cols": self.unmatched_cols,
            "total_score": self.total_score,
            "quality": self.quality,
            "n_original_rows": self.n_original_rows,
            "n_original_cols": self.n_original_cols,
            "n_matched": self.n_matched,
            "match_ratio": self.match_ratio,
        }

    def __repr__(self) -> str:
        return (
            f"MatchResult(matched={self.n_matched}, "
            f"unmatched_rows={len(self.unmatched_rows)}, "
            f"unmatched_cols={len(self.unmatched_cols)}, "
            f"quality={self.quality:.4f})"
        )


# ===================================================================
#  QualityFilter
# ===================================================================
class QualityFilter:
    """Post-matching quality filtering.

    After the Hungarian algorithm finds optimal assignments, this filter
    removes low-quality matches that fall below a threshold.

    Parameters
    ----------
    min_score : float
        Minimum match score to keep. Default 0.5.
    relative_threshold : float or None
        If set, also filter matches below this fraction of the best match.
        E.g., 0.5 means keep only matches with score >= 0.5 * best_score.
    max_unmatched_fraction : float
        Maximum fraction of variables that can be unmatched. Default 1.0 (no limit).
    """

    def __init__(
        self,
        min_score: float = 0.5,
        relative_threshold: Optional[float] = None,
        max_unmatched_fraction: float = 1.0,
    ) -> None:
        if not 0.0 <= min_score <= 1.0:
            raise ValueError(f"min_score must be in [0,1], got {min_score}")
        self.min_score = min_score
        self.relative_threshold = relative_threshold
        self.max_unmatched_fraction = max_unmatched_fraction

    def filter(
        self,
        row_indices: List[int],
        col_indices: List[int],
        scores: List[float],
        n_rows: int,
        n_cols: int,
    ) -> Tuple[List[int], List[int], List[float], List[int], List[int]]:
        """Filter matches by quality.

        Parameters
        ----------
        row_indices, col_indices : list of int
            Matched indices from Hungarian algorithm.
        scores : list of float
            Match scores.
        n_rows, n_cols : int
            Original matrix dimensions.

        Returns
        -------
        (kept_rows, kept_cols, kept_scores, unmatched_rows, unmatched_cols)
        """
        if not scores:
            return [], [], [], list(range(n_rows)), list(range(n_cols))

        # Compute threshold
        threshold = self.min_score
        if self.relative_threshold is not None:
            best = max(scores)
            relative = self.relative_threshold * best
            threshold = max(threshold, relative)

        # Filter
        kept_rows = []
        kept_cols = []
        kept_scores = []
        rejected_rows = []
        rejected_cols = []

        for r, c, s in zip(row_indices, col_indices, scores):
            if s >= threshold:
                kept_rows.append(r)
                kept_cols.append(c)
                kept_scores.append(s)
            else:
                rejected_rows.append(r)
                rejected_cols.append(c)

        # Check max unmatched fraction
        matched_rows = set(kept_rows)
        matched_cols = set(kept_cols)
        unmatched_rows = sorted(set(range(n_rows)) - matched_rows)
        unmatched_cols = sorted(set(range(n_cols)) - matched_cols)

        total = max(n_rows, n_cols)
        if total > 0:
            unmatched_frac = max(len(unmatched_rows), len(unmatched_cols)) / total
            if unmatched_frac > self.max_unmatched_fraction:
                logger.warning(
                    "Unmatched fraction %.2f exceeds limit %.2f; "
                    "relaxing threshold to maintain matches",
                    unmatched_frac,
                    self.max_unmatched_fraction,
                )
                # Re-add rejected matches in order of decreasing score
                rejected = sorted(
                    zip(rejected_rows, rejected_cols, [s for r, c, s in zip(row_indices, col_indices, scores) if r in rejected_rows]),
                    key=lambda x: -x[2],
                )
                for r, c, s in rejected:
                    if r not in matched_rows and c not in matched_cols:
                        kept_rows.append(r)
                        kept_cols.append(c)
                        kept_scores.append(s)
                        matched_rows.add(r)
                        matched_cols.add(c)

                        unmatched_rows = sorted(set(range(n_rows)) - matched_rows)
                        unmatched_cols = sorted(set(range(n_cols)) - matched_cols)
                        new_frac = max(len(unmatched_rows), len(unmatched_cols)) / total
                        if new_frac <= self.max_unmatched_fraction:
                            break

        return kept_rows, kept_cols, kept_scores, unmatched_rows, unmatched_cols


# ===================================================================
#  PaddedHungarianSolver
# ===================================================================
class PaddedHungarianSolver:
    """Hungarian algorithm solver with non-square matrix support.

    Handles:
        - Non-square matrices via padding with -infinity scores
        - Empty matrices
        - All-negative scores
        - Score ties (deterministic tie-breaking)
        - Quality-based post-filtering

    Parameters
    ----------
    quality_filter : QualityFilter or None
        Post-matching quality filter. If None, uses default (min_score=0.5).
    maximize : bool
        If True, maximize total score. If False, minimize. Default True.
    pad_value : float
        Value to use for padding dummy entries. Default -1e9.
    """

    def __init__(
        self,
        quality_filter: Optional[QualityFilter] = None,
        maximize: bool = True,
        pad_value: float = _NEG_INF_PROXY,
    ) -> None:
        self.quality_filter = quality_filter if quality_filter is not None else QualityFilter()
        self.maximize = maximize
        self.pad_value = pad_value

    def _pad_to_square(
        self,
        matrix: NDArray[np.floating],
    ) -> Tuple[NDArray[np.floating], int, int]:
        """Pad a rectangular matrix to square.

        Parameters
        ----------
        matrix : NDArray, shape (m, n)
            Input score matrix.

        Returns
        -------
        (padded_matrix, original_rows, original_cols)
        """
        m, n = matrix.shape
        size = max(m, n)

        if m == n:
            return matrix.copy(), m, n

        padded = np.full((size, size), self.pad_value, dtype=np.float64)
        padded[:m, :n] = matrix

        return padded, m, n

    def solve(
        self,
        score_matrix: NDArray[np.floating],
        row_labels: Optional[List[Any]] = None,
        col_labels: Optional[List[Any]] = None,
    ) -> MatchResult:
        """Solve the assignment problem.

        Parameters
        ----------
        score_matrix : NDArray, shape (m, n)
            Score matrix. Higher = better if maximize=True.
        row_labels : list or None
            Labels for rows (for reporting).
        col_labels : list or None
            Labels for columns.

        Returns
        -------
        MatchResult
            Structured matching result.
        """
        score_matrix = np.asarray(score_matrix, dtype=np.float64)

        # Handle edge cases
        if score_matrix.size == 0:
            m = score_matrix.shape[0] if score_matrix.ndim >= 1 else 0
            n = score_matrix.shape[1] if score_matrix.ndim >= 2 else 0
            return MatchResult(
                row_indices=[],
                col_indices=[],
                scores=[],
                unmatched_rows=list(range(m)),
                unmatched_cols=list(range(n)),
                total_score=0.0,
                quality=0.0,
                n_original_rows=m,
                n_original_cols=n,
            )

        if score_matrix.ndim != 2:
            raise ValueError(f"score_matrix must be 2D, got {score_matrix.ndim}D")

        m, n = score_matrix.shape

        # Handle all-nan
        if np.all(np.isnan(score_matrix)):
            return MatchResult(
                row_indices=[],
                col_indices=[],
                scores=[],
                unmatched_rows=list(range(m)),
                unmatched_cols=list(range(n)),
                total_score=0.0,
                quality=0.0,
                n_original_rows=m,
                n_original_cols=n,
            )

        # Replace NaN with pad value
        clean_matrix = np.where(np.isnan(score_matrix), self.pad_value, score_matrix)

        # Pad to square
        padded, orig_m, orig_n = self._pad_to_square(clean_matrix)

        # scipy.optimize.linear_sum_assignment minimizes, so negate for maximization
        if self.maximize:
            cost = -padded
        else:
            cost = padded.copy()

        # Add small noise for deterministic tie-breaking
        rng = np.random.default_rng(42)
        cost += rng.uniform(0, 1e-10, size=cost.shape)

        row_ind, col_ind = linear_sum_assignment(cost)

        # Extract matches within original bounds
        raw_rows = []
        raw_cols = []
        raw_scores = []

        for r, c in zip(row_ind, col_ind):
            if r < orig_m and c < orig_n:
                raw_rows.append(int(r))
                raw_cols.append(int(c))
                raw_scores.append(float(score_matrix[r, c]))

        # Apply quality filter
        kept_rows, kept_cols, kept_scores, unmatched_rows, unmatched_cols = (
            self.quality_filter.filter(raw_rows, raw_cols, raw_scores, orig_m, orig_n)
        )

        total = sum(kept_scores) if kept_scores else 0.0
        quality = float(np.mean(kept_scores)) if kept_scores else 0.0

        return MatchResult(
            row_indices=kept_rows,
            col_indices=kept_cols,
            scores=kept_scores,
            unmatched_rows=unmatched_rows,
            unmatched_cols=unmatched_cols,
            total_score=total,
            quality=quality,
            n_original_rows=orig_m,
            n_original_cols=orig_n,
        )

    def solve_with_constraints(
        self,
        score_matrix: NDArray[np.floating],
        must_match: Optional[List[Tuple[int, int]]] = None,
        cannot_match: Optional[List[Tuple[int, int]]] = None,
    ) -> MatchResult:
        """Solve with hard constraints.

        Parameters
        ----------
        score_matrix : NDArray, shape (m, n)
            Score matrix.
        must_match : list of (row, col) or None
            Pairs that must be matched.
        cannot_match : list of (row, col) or None
            Pairs that must not be matched.

        Returns
        -------
        MatchResult
        """
        modified = score_matrix.copy()

        # Enforce cannot_match by setting to pad value
        if cannot_match:
            for r, c in cannot_match:
                if 0 <= r < modified.shape[0] and 0 <= c < modified.shape[1]:
                    modified[r, c] = self.pad_value

        # Enforce must_match by setting extremely high scores
        if must_match:
            max_val = float(np.max(np.abs(score_matrix[np.isfinite(score_matrix)]))) if np.any(np.isfinite(score_matrix)) else 1.0
            boost = max_val * 1000.0 + 1e6
            for r, c in must_match:
                if 0 <= r < modified.shape[0] and 0 <= c < modified.shape[1]:
                    modified[r, c] = boost

        result = self.solve(modified)

        # Verify must_match constraints
        if must_match:
            matched_set = set(zip(result.row_indices, result.col_indices))
            for r, c in must_match:
                if (r, c) not in matched_set:
                    logger.warning(
                        "must_match constraint (%d, %d) not satisfied", r, c
                    )

        return result


def solve_assignment(
    score_matrix: NDArray[np.floating],
    min_quality: float = 0.5,
    maximize: bool = True,
) -> MatchResult:
    """Convenience function for solving assignment problems.

    Parameters
    ----------
    score_matrix : NDArray, shape (m, n)
        Score matrix.
    min_quality : float
        Minimum match score threshold.
    maximize : bool
        Whether to maximize (True) or minimize (False).

    Returns
    -------
    MatchResult
    """
    solver = PaddedHungarianSolver(
        quality_filter=QualityFilter(min_score=min_quality),
        maximize=maximize,
    )
    return solver.solve(score_matrix)


def validate_match_result(
    result: MatchResult,
    score_matrix: NDArray[np.floating],
) -> Dict[str, Any]:
    """Validate a match result against its source score matrix.

    Checks:
        1. No row or column is matched twice (bijectivity)
        2. Matched pairs have valid indices
        3. Scores in result match the score matrix
        4. Unmatched sets are correct

    Parameters
    ----------
    result : MatchResult
        Match result to validate.
    score_matrix : NDArray, shape (m, n)
        Original score matrix.

    Returns
    -------
    dict with keys:
        - 'valid': bool
        - 'errors': list of str
        - 'warnings': list of str
    """
    errors: List[str] = []
    warns: List[str] = []
    m, n = score_matrix.shape

    # Check bijectivity
    if len(set(result.row_indices)) != len(result.row_indices):
        errors.append("Duplicate row indices in match")
    if len(set(result.col_indices)) != len(result.col_indices):
        errors.append("Duplicate column indices in match")

    # Check index bounds
    for r in result.row_indices:
        if r < 0 or r >= m:
            errors.append(f"Row index {r} out of bounds [0, {m})")
    for c in result.col_indices:
        if c < 0 or c >= n:
            errors.append(f"Column index {c} out of bounds [0, {n})")

    # Check scores match
    for r, c, s in zip(result.row_indices, result.col_indices, result.scores):
        if 0 <= r < m and 0 <= c < n:
            expected = float(score_matrix[r, c])
            if abs(s - expected) > 1e-8:
                warns.append(
                    f"Score mismatch at ({r},{c}): {s:.8f} vs {expected:.8f}"
                )

    # Check unmatched sets
    matched_rows = set(result.row_indices)
    matched_cols = set(result.col_indices)
    expected_unmatched_rows = sorted(set(range(m)) - matched_rows)
    expected_unmatched_cols = sorted(set(range(n)) - matched_cols)

    if result.unmatched_rows != expected_unmatched_rows:
        warns.append(
            f"Unmatched rows mismatch: got {result.unmatched_rows}, "
            f"expected {expected_unmatched_rows}"
        )
    if result.unmatched_cols != expected_unmatched_cols:
        warns.append(
            f"Unmatched cols mismatch: got {result.unmatched_cols}, "
            f"expected {expected_unmatched_cols}"
        )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warns,
    }


def greedy_assignment(
    score_matrix: NDArray[np.floating],
    min_quality: float = 0.5,
) -> MatchResult:
    """Greedy assignment as a fast alternative to Hungarian.

    Iteratively selects the best remaining (row, col) pair. Not optimal
    but O(min(m,n) * m * n) instead of O(max(m,n)^3).

    Parameters
    ----------
    score_matrix : NDArray, shape (m, n)
        Score matrix.
    min_quality : float
        Minimum score threshold.

    Returns
    -------
    MatchResult
    """
    score_matrix = np.asarray(score_matrix, dtype=np.float64)
    if score_matrix.size == 0:
        m = score_matrix.shape[0] if score_matrix.ndim >= 1 else 0
        n = score_matrix.shape[1] if score_matrix.ndim >= 2 else 0
        return MatchResult(
            row_indices=[], col_indices=[], scores=[],
            unmatched_rows=list(range(m)),
            unmatched_cols=list(range(n)),
            total_score=0.0, quality=0.0,
            n_original_rows=m, n_original_cols=n,
        )

    m, n = score_matrix.shape
    available_rows = set(range(m))
    available_cols = set(range(n))
    row_indices: List[int] = []
    col_indices: List[int] = []
    scores: List[float] = []

    working = score_matrix.copy()
    working[np.isnan(working)] = -np.inf

    while available_rows and available_cols:
        best_val = -np.inf
        best_r = -1
        best_c = -1
        for r in available_rows:
            for c in available_cols:
                if working[r, c] > best_val:
                    best_val = working[r, c]
                    best_r = r
                    best_c = c

        if best_val < min_quality:
            break

        row_indices.append(best_r)
        col_indices.append(best_c)
        scores.append(float(score_matrix[best_r, best_c]))
        available_rows.remove(best_r)
        available_cols.remove(best_c)

    unmatched_rows = sorted(available_rows)
    unmatched_cols = sorted(available_cols)
    total = sum(scores) if scores else 0.0
    quality = float(np.mean(scores)) if scores else 0.0

    return MatchResult(
        row_indices=row_indices,
        col_indices=col_indices,
        scores=scores,
        unmatched_rows=unmatched_rows,
        unmatched_cols=unmatched_cols,
        total_score=total,
        quality=quality,
        n_original_rows=m,
        n_original_cols=n,
    )
