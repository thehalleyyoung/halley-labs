"""
Per-edge fragility scorer (ALG 3).

For each existing edge and each candidate absent edge, computes a composite
fragility score by evaluating three channels: d-separation impact,
identification impact, and estimation impact.

Algorithm 3 — Per-Edge Fragility Scoring
-----------------------------------------
Input: DAG G, treatment X, outcome Y, (optional) data D
Output: Ranked list of (edge, fragility_score) pairs

1. Compute ancestral set A = an({X, Y}) in G
2. Enumerate candidate edits E within A
3. For each edit e ∈ E:
   a. Apply e to copy G' = G + e
   b. If G' is cyclic, skip
   c. Compute F_dsep(e), F_id(e), F_est(e)
   d. Aggregate: F(e) = agg(F_dsep, F_id, F_est)
4. Sort by F(e) descending
5. Return ranked list
"""

from __future__ import annotations

import logging
import time
from typing import Any, Sequence

import numpy as np
import pandas as pd

from causalcert.types import (
    AdjacencyMatrix,
    CITestResult,
    EditType,
    FragilityChannel,
    FragilityScore,
    NodeId,
    NodeSet,
    StructuralEdit,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _apply_edit(adj: np.ndarray, edit: StructuralEdit) -> np.ndarray:
    """Apply edit to adjacency matrix copy."""
    result = adj.copy()
    if edit.edit_type == EditType.ADD:
        result[edit.source, edit.target] = 1
    elif edit.edit_type == EditType.DELETE:
        result[edit.source, edit.target] = 0
    elif edit.edit_type == EditType.REVERSE:
        result[edit.source, edit.target] = 0
        result[edit.target, edit.source] = 1
    return result


def _is_dag(adj: np.ndarray) -> bool:
    """Quick Kahn-based acyclicity test."""
    n = adj.shape[0]
    in_deg = adj.sum(axis=0).astype(int).copy()
    queue = [i for i in range(n) if in_deg[i] == 0]
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for c in np.nonzero(adj[node])[0]:
            in_deg[int(c)] -= 1
            if in_deg[int(c)] == 0:
                queue.append(int(c))
    return visited == n


def _ancestors_inclusive(adj: np.ndarray, targets: set[int]) -> set[int]:
    """Ancestors of targets (inclusive)."""
    from collections import deque
    visited: set[int] = set()
    queue = deque(targets)
    while queue:
        v = queue.popleft()
        if v in visited:
            continue
        visited.add(v)
        for p in np.nonzero(adj[:, v])[0]:
            p = int(p)
            if p not in visited:
                queue.append(p)
    return visited


# ---------------------------------------------------------------------------
# Scorer implementation
# ---------------------------------------------------------------------------


class FragilityScorerImpl:
    """Concrete implementation of the per-edge fragility scorer (ALG 3).

    Parameters
    ----------
    ci_results : Sequence[CITestResult] | None
        Pre-computed CI test results.  If ``None``, the scorer runs its
        own tests.
    alpha : float
        Significance level for CI tests.
    include_absent : bool
        Whether to score candidate edge additions (absent edges) in
        addition to existing edges.
    max_adj_set_size : int
        Maximum adjustment set size for channel evaluation.
    restrict_to_ancestral : bool
        If True, only consider edits within the ancestral set of {X, Y}.
    aggregation_method : str
        How to aggregate channel scores: 'max', 'weighted', 'hierarchical'.
    channel_weights : dict[FragilityChannel, float] | None
        Weights for weighted aggregation.
    """

    def __init__(
        self,
        ci_results: Sequence[CITestResult] | None = None,
        alpha: float = 0.05,
        include_absent: bool = True,
        max_adj_set_size: int = 4,
        restrict_to_ancestral: bool = True,
        aggregation_method: str = "max",
        channel_weights: dict[FragilityChannel, float] | None = None,
    ) -> None:
        self.ci_results = list(ci_results) if ci_results is not None else None
        self.alpha = alpha
        self.include_absent = include_absent
        self.max_adj_set_size = max_adj_set_size
        self.restrict_to_ancestral = restrict_to_ancestral
        self.aggregation_method = aggregation_method
        self.channel_weights = channel_weights or {
            FragilityChannel.D_SEPARATION: 0.4,
            FragilityChannel.IDENTIFICATION: 0.4,
            FragilityChannel.ESTIMATION: 0.2,
        }

        # Lazily initialised channels
        self._dsep_channel: Any = None
        self._id_channel: Any = None
        self._est_channel: Any = None

    def _init_channels(self) -> None:
        """Lazily initialise the three fragility channels."""
        from causalcert.fragility.channels import (
            DSepChannel,
            EstimationChannel,
            IdentificationChannel,
        )

        if self._dsep_channel is None:
            self._dsep_channel = DSepChannel(
                max_adj_set_size=self.max_adj_set_size
            )
        if self._id_channel is None:
            self._id_channel = IdentificationChannel(
                max_adj_set_size=self.max_adj_set_size
            )
        if self._est_channel is None:
            self._est_channel = EstimationChannel()

    def _aggregate(
        self, channel_scores: dict[FragilityChannel, float]
    ) -> float:
        """Aggregate channel scores into a single total score."""
        scores = list(channel_scores.values())
        if not scores:
            return 0.0

        if self.aggregation_method == "max":
            return max(scores)
        elif self.aggregation_method == "weighted":
            total_weight = 0.0
            weighted_sum = 0.0
            for ch, s in channel_scores.items():
                w = self.channel_weights.get(ch, 1.0)
                weighted_sum += w * s
                total_weight += w
            return weighted_sum / total_weight if total_weight > 0 else 0.0
        elif self.aggregation_method == "hierarchical":
            # Hierarchical: F_id first (binary), then F_dsep, then F_est
            f_id = channel_scores.get(FragilityChannel.IDENTIFICATION, 0.0)
            if f_id >= 1.0:
                return 1.0
            f_dsep = channel_scores.get(FragilityChannel.D_SEPARATION, 0.0)
            f_est = channel_scores.get(FragilityChannel.ESTIMATION, 0.0)
            return max(f_id, f_dsep, f_est)
        else:
            return max(scores)

    def score(
        self,
        adj: AdjacencyMatrix,
        treatment: NodeId,
        outcome: NodeId,
        data: pd.DataFrame | None = None,
    ) -> list[FragilityScore]:
        """Score all edges for fragility, sorted by decreasing total score.

        Parameters
        ----------
        adj : AdjacencyMatrix
            DAG adjacency matrix.
        treatment, outcome : NodeId
            Treatment and outcome nodes.
        data : pd.DataFrame | None
            Observational data (required if ``ci_results`` is ``None``).

        Returns
        -------
        list[FragilityScore]
            Edges ranked by total fragility (most fragile first).
        """
        self._init_channels()
        adj_arr = np.asarray(adj, dtype=np.int8)

        # Step 1: enumerate candidate edits
        edits = self._candidate_edits(adj_arr, treatment, outcome)
        logger.info("Scoring %d candidate edits", len(edits))

        # Step 2: batch-evaluate each channel
        t0 = time.time()
        dsep_scores = self._dsep_channel.evaluate_batch(
            adj_arr, edits, treatment, outcome, data
        )
        id_scores = self._id_channel.evaluate_batch(
            adj_arr, edits, treatment, outcome, data
        )
        est_scores = self._est_channel.evaluate_batch(
            adj_arr, edits, treatment, outcome, data
        )
        t1 = time.time()
        logger.info("Channel evaluation completed in %.2fs", t1 - t0)

        # Step 3: aggregate and build FragilityScore objects
        results: list[FragilityScore] = []
        for i, edit in enumerate(edits):
            channel_scores = {
                FragilityChannel.D_SEPARATION: dsep_scores[i],
                FragilityChannel.IDENTIFICATION: id_scores[i],
                FragilityChannel.ESTIMATION: est_scores[i],
            }
            total = self._aggregate(channel_scores)

            # Find witness CI test if available
            witness_ci = self._find_witness_ci(edit)

            results.append(FragilityScore(
                edge=(edit.source, edit.target),
                total_score=total,
                channel_scores=channel_scores,
                witness_ci=witness_ci,
            ))

        # Step 4: sort by decreasing total score
        results.sort(key=lambda s: s.total_score, reverse=True)
        return results

    def score_single_edge(
        self,
        adj: AdjacencyMatrix,
        edit: StructuralEdit,
        treatment: NodeId,
        outcome: NodeId,
        data: pd.DataFrame | None = None,
    ) -> FragilityScore:
        """Compute the fragility score for a single edge edit.

        Parameters
        ----------
        adj : AdjacencyMatrix
            Current DAG adjacency matrix.
        edit : StructuralEdit
            The candidate edit.
        treatment, outcome : NodeId
            Treatment and outcome nodes.
        data : pd.DataFrame | None
            Observational data.

        Returns
        -------
        FragilityScore
        """
        self._init_channels()
        adj_arr = np.asarray(adj, dtype=np.int8)

        f_dsep = self._dsep_channel.evaluate(adj_arr, edit, treatment, outcome, data)
        f_id = self._id_channel.evaluate(adj_arr, edit, treatment, outcome, data)
        f_est = self._est_channel.evaluate(adj_arr, edit, treatment, outcome, data)

        channel_scores = {
            FragilityChannel.D_SEPARATION: f_dsep,
            FragilityChannel.IDENTIFICATION: f_id,
            FragilityChannel.ESTIMATION: f_est,
        }
        total = self._aggregate(channel_scores)
        witness_ci = self._find_witness_ci(edit)

        return FragilityScore(
            edge=(edit.source, edit.target),
            total_score=total,
            channel_scores=channel_scores,
            witness_ci=witness_ci,
        )

    def score_data_free(
        self,
        adj: AdjacencyMatrix,
        treatment: NodeId,
        outcome: NodeId,
    ) -> list[FragilityScore]:
        """Score edges using only structural channels (F_dsep + F_id).

        Parameters
        ----------
        adj : AdjacencyMatrix
            DAG adjacency matrix.
        treatment, outcome : NodeId
            Treatment and outcome nodes.

        Returns
        -------
        list[FragilityScore]
            Edges ranked by structural fragility.
        """
        self._init_channels()
        adj_arr = np.asarray(adj, dtype=np.int8)
        edits = self._candidate_edits(adj_arr, treatment, outcome)

        dsep_scores = self._dsep_channel.evaluate_batch(
            adj_arr, edits, treatment, outcome
        )
        id_scores = self._id_channel.evaluate_batch(
            adj_arr, edits, treatment, outcome
        )

        results: list[FragilityScore] = []
        for i, edit in enumerate(edits):
            channel_scores = {
                FragilityChannel.D_SEPARATION: dsep_scores[i],
                FragilityChannel.IDENTIFICATION: id_scores[i],
            }
            total = max(channel_scores.values()) if channel_scores else 0.0
            results.append(FragilityScore(
                edge=(edit.source, edit.target),
                total_score=total,
                channel_scores=channel_scores,
                witness_ci=self._find_witness_ci(edit),
            ))

        results.sort(key=lambda s: s.total_score, reverse=True)
        return results

    def _candidate_edits(
        self,
        adj: AdjacencyMatrix,
        treatment: NodeId | None = None,
        outcome: NodeId | None = None,
    ) -> list[StructuralEdit]:
        """Enumerate all candidate single-edge edits.

        If restrict_to_ancestral is True and treatment/outcome are given,
        only consider edits within the ancestral set of {X, Y}.

        Parameters
        ----------
        adj : AdjacencyMatrix
            DAG adjacency matrix.
        treatment, outcome : NodeId | None
            Treatment and outcome for ancestral restriction.

        Returns
        -------
        list[StructuralEdit]
        """
        adj_arr = np.asarray(adj, dtype=np.int8)
        n = adj_arr.shape[0]

        # Determine relevant node set
        if (
            self.restrict_to_ancestral
            and treatment is not None
            and outcome is not None
        ):
            relevant = _ancestors_inclusive(adj_arr, {treatment, outcome})
        else:
            relevant = set(range(n))

        edits: list[StructuralEdit] = []
        for i in sorted(relevant):
            for j in sorted(relevant):
                if i == j:
                    continue
                if adj_arr[i, j]:
                    # Existing edge: can delete or reverse
                    edits.append(StructuralEdit(EditType.DELETE, i, j))
                    trial = adj_arr.copy()
                    trial[i, j] = 0
                    trial[j, i] = 1
                    if _is_dag(trial):
                        edits.append(StructuralEdit(EditType.REVERSE, i, j))
                elif self.include_absent:
                    # Absent edge: can add
                    trial = adj_arr.copy()
                    trial[i, j] = 1
                    if _is_dag(trial):
                        edits.append(StructuralEdit(EditType.ADD, i, j))

        return edits

    def _find_witness_ci(self, edit: StructuralEdit) -> CITestResult | None:
        """Find the CI test result most affected by this edit."""
        if self.ci_results is None:
            return None

        # Look for CI tests involving the edge endpoints
        best: CITestResult | None = None
        best_pval = 1.0
        for ci in self.ci_results:
            endpoints = {ci.x, ci.y}
            if edit.source in endpoints or edit.target in endpoints:
                if ci.p_value < best_pval:
                    best = ci
                    best_pval = ci.p_value
        return best

    def score_parallel(
        self,
        adj: AdjacencyMatrix,
        treatment: NodeId,
        outcome: NodeId,
        data: pd.DataFrame | None = None,
        n_jobs: int = 1,
    ) -> list[FragilityScore]:
        """Score edges with optional parallel evaluation.

        For n_jobs=1 (default), behaves identically to :meth:`score`.
        For n_jobs>1, evaluates edits in parallel using concurrent.futures.

        Parameters
        ----------
        adj : AdjacencyMatrix
            DAG adjacency matrix.
        treatment, outcome : NodeId
            Treatment and outcome nodes.
        data : pd.DataFrame | None
            Observational data.
        n_jobs : int
            Number of parallel workers.

        Returns
        -------
        list[FragilityScore]
        """
        if n_jobs <= 1:
            return self.score(adj, treatment, outcome, data)

        self._init_channels()
        adj_arr = np.asarray(adj, dtype=np.int8)
        edits = self._candidate_edits(adj_arr, treatment, outcome)

        if not edits:
            return []

        # Split edits into chunks
        import concurrent.futures

        chunk_size = max(1, len(edits) // n_jobs)
        chunks = [
            edits[i: i + chunk_size] for i in range(0, len(edits), chunk_size)
        ]

        def _score_chunk(
            chunk: list[StructuralEdit],
        ) -> list[FragilityScore]:
            results: list[FragilityScore] = []
            for edit in chunk:
                fs = self.score_single_edge(adj_arr, edit, treatment, outcome, data)
                results.append(fs)
            return results

        all_results: list[FragilityScore] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(_score_chunk, chunk) for chunk in chunks]
            for future in concurrent.futures.as_completed(futures):
                all_results.extend(future.result())

        all_results.sort(key=lambda s: s.total_score, reverse=True)
        return all_results

    def progress_score(
        self,
        adj: AdjacencyMatrix,
        treatment: NodeId,
        outcome: NodeId,
        data: pd.DataFrame | None = None,
        callback: Any | None = None,
    ) -> list[FragilityScore]:
        """Score edges with progress reporting via callback.

        Parameters
        ----------
        adj : AdjacencyMatrix
            DAG adjacency matrix.
        treatment, outcome : NodeId
            Treatment and outcome nodes.
        data : pd.DataFrame | None
            Observational data.
        callback : callable | None
            Called with (current_index, total, edit, score) after each edge.

        Returns
        -------
        list[FragilityScore]
        """
        self._init_channels()
        adj_arr = np.asarray(adj, dtype=np.int8)
        edits = self._candidate_edits(adj_arr, treatment, outcome)
        total = len(edits)

        results: list[FragilityScore] = []
        for i, edit in enumerate(edits):
            fs = self.score_single_edge(adj_arr, edit, treatment, outcome, data)
            results.append(fs)
            if callback is not None:
                callback(i, total, edit, fs)

        results.sort(key=lambda s: s.total_score, reverse=True)
        return results

    def get_scoring_summary(
        self,
        scores: list[FragilityScore],
    ) -> dict[str, Any]:
        """Compute summary statistics for a set of fragility scores.

        Returns
        -------
        dict
            Keys: 'n_scored', 'mean', 'max', 'min', 'std', 'n_critical',
            'n_cosmetic', 'channel_means'.
        """
        if not scores:
            return {
                "n_scored": 0,
                "mean": 0.0,
                "max": 0.0,
                "min": 0.0,
                "std": 0.0,
                "n_critical": 0,
                "n_cosmetic": 0,
                "channel_means": {},
            }

        totals = [s.total_score for s in scores]
        channel_sums: dict[FragilityChannel, list[float]] = {}
        for s in scores:
            for ch, val in s.channel_scores.items():
                channel_sums.setdefault(ch, []).append(val)

        return {
            "n_scored": len(scores),
            "mean": float(np.mean(totals)),
            "max": float(np.max(totals)),
            "min": float(np.min(totals)),
            "std": float(np.std(totals)),
            "n_critical": sum(1 for t in totals if t >= 0.7),
            "n_cosmetic": sum(1 for t in totals if t < 0.1),
            "channel_means": {
                ch.value: float(np.mean(vals))
                for ch, vals in channel_sums.items()
            },
        }
