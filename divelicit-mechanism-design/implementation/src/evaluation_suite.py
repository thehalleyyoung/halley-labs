"""Evaluation suite for diverse elicitation quality.

Provides comprehensive metrics for evaluating how well a selection
method achieves diversity: coverage, novelty, redundancy, and
alignment with human preferences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .embedding import TextEmbedder, embed_texts, project_to_sphere
from .diversity_metrics import cosine_diversity, sinkhorn_diversity_metric as sinkhorn_diversity, log_det_diversity
from .coverage import CoverageCertificate, estimate_coverage, _effective_dimension
from .kernels import RBFKernel


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CoverageMetrics:
    """Coverage evaluation metrics."""
    certificate: Optional[CoverageCertificate]
    fill_distance: float
    mean_nn_distance: float
    fraction_covered: float
    effective_dim: int


@dataclass
class NoveltyMetrics:
    """Per-item novelty metrics."""
    per_item_novelty: List[float]
    mean_novelty: float
    max_novelty: float
    min_novelty: float
    n_highly_novel: int  # items with novelty > 0.5


@dataclass
class RedundancyMetrics:
    """Redundancy/overlap metrics."""
    mean_pairwise_similarity: float
    max_pairwise_similarity: float
    n_near_duplicates: int
    redundancy_ratio: float  # fraction that are near-duplicates
    dispersion: float  # minimum pairwise distance


@dataclass
class PreferenceAlignment:
    """Alignment with human preferences."""
    kendall_tau: float
    spearman_rho: float
    top_k_overlap: float
    preference_satisfaction: float


@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    coverage: CoverageMetrics
    novelty: NoveltyMetrics
    redundancy: RedundancyMetrics
    preference: Optional[PreferenceAlignment]
    aggregate_score: float
    method_name: str
    n_selected: int
    n_candidates: int
    metrics_dict: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core evaluator
# ---------------------------------------------------------------------------

class DiversityEvaluator:
    """Evaluate the quality of a diverse elicitation/selection.

    Computes coverage, novelty, redundancy, and optional preference
    alignment metrics for a selected subset against a candidate pool.

    Example::

        evaluator = DiversityEvaluator()
        report = evaluator.evaluate(selected_items, all_candidates)
        print(report.coverage.fill_distance)
        print(report.novelty.mean_novelty)
        print(report.redundancy.dispersion)
        print(report.aggregate_score)
    """

    def __init__(
        self,
        embed_dim: int = 64,
        embed_fn: Optional[Callable[[Any], np.ndarray]] = None,
        near_duplicate_threshold: float = 0.9,
        coverage_epsilon: float = 0.3,
    ):
        self.embed_dim = embed_dim
        self.embed_fn = embed_fn
        self.near_duplicate_threshold = near_duplicate_threshold
        self.coverage_epsilon = coverage_epsilon
        self._embedder = TextEmbedder(dim=embed_dim)

    def evaluate(
        self,
        selected: List[Any],
        candidates: Optional[List[Any]] = None,
        *,
        human_ranking: Optional[List[int]] = None,
        method_name: str = "unknown",
    ) -> EvaluationReport:
        """Run full evaluation suite.

        Args:
            selected: The selected subset.
            candidates: The full candidate pool. If None, evaluates
                selected in isolation.
            human_ranking: Optional human preference ranking (list of indices
                into selected, from best to worst).
            method_name: Name of the selection method for labeling.

        Returns:
            EvaluationReport with all metrics.
        """
        sel_emb = self._embed(selected)
        cand_emb = self._embed(candidates) if candidates else sel_emb

        coverage = self._evaluate_coverage(sel_emb, cand_emb)
        novelty = self._evaluate_novelty(sel_emb, cand_emb)
        redundancy = self._evaluate_redundancy(sel_emb)

        pref = None
        if human_ranking is not None:
            pref = self._evaluate_preference(sel_emb, human_ranking)

        # Aggregate score: weighted combination of key metrics
        agg = (
            0.3 * (1.0 - redundancy.redundancy_ratio)
            + 0.3 * novelty.mean_novelty
            + 0.2 * coverage.fraction_covered
            + 0.2 * min(redundancy.dispersion * 2, 1.0)
        )

        metrics_dict = {
            "fill_distance": coverage.fill_distance,
            "mean_nn_distance": coverage.mean_nn_distance,
            "fraction_covered": coverage.fraction_covered,
            "mean_novelty": novelty.mean_novelty,
            "mean_pairwise_sim": redundancy.mean_pairwise_similarity,
            "dispersion": redundancy.dispersion,
            "redundancy_ratio": redundancy.redundancy_ratio,
            "aggregate": agg,
        }
        if pref:
            metrics_dict["kendall_tau"] = pref.kendall_tau
            metrics_dict["preference_satisfaction"] = pref.preference_satisfaction

        return EvaluationReport(
            coverage=coverage,
            novelty=novelty,
            redundancy=redundancy,
            preference=pref,
            aggregate_score=agg,
            method_name=method_name,
            n_selected=len(selected),
            n_candidates=len(candidates) if candidates else len(selected),
            metrics_dict=metrics_dict,
        )

    # ------------------------------------------------------------------
    # Coverage metrics
    # ------------------------------------------------------------------

    def _evaluate_coverage(
        self,
        sel_emb: np.ndarray,
        cand_emb: np.ndarray,
    ) -> CoverageMetrics:
        """Evaluate how well selected points cover the candidate space."""
        if len(sel_emb) == 0:
            return CoverageMetrics(None, float('inf'), float('inf'), 0.0, 0)

        cert = estimate_coverage(sel_emb, epsilon=self.coverage_epsilon)

        # Fill distance and mean NN distance
        dists = self._pairwise_dist(cand_emb, sel_emb)
        nn_dists = np.min(dists, axis=1)
        fill_dist = float(np.max(nn_dists))
        mean_nn = float(np.mean(nn_dists))

        # Fraction of candidates within epsilon of a selected point
        covered = np.sum(nn_dists <= self.coverage_epsilon) / max(len(cand_emb), 1)

        eff_dim = _effective_dimension(sel_emb)

        return CoverageMetrics(
            certificate=cert,
            fill_distance=fill_dist,
            mean_nn_distance=mean_nn,
            fraction_covered=float(covered),
            effective_dim=eff_dim,
        )

    # ------------------------------------------------------------------
    # Novelty metrics
    # ------------------------------------------------------------------

    def _evaluate_novelty(
        self,
        sel_emb: np.ndarray,
        cand_emb: np.ndarray,
    ) -> NoveltyMetrics:
        """Evaluate per-item novelty of selected items.

        Novelty = 1 - max_similarity to any OTHER selected item.
        Also considers distance to centroid of candidate pool.
        """
        n = len(sel_emb)
        if n == 0:
            return NoveltyMetrics([], 0.0, 0.0, 0.0, 0)

        if n == 1:
            return NoveltyMetrics([1.0], 1.0, 1.0, 1.0, 1)

        sim = sel_emb @ sel_emb.T
        np.fill_diagonal(sim, -1.0)
        max_sim = np.max(sim, axis=1)
        per_item = [float(1.0 - s) for s in max_sim]

        # Also factor in distance from candidate centroid
        centroid = np.mean(cand_emb, axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-12
        dist_to_centroid = 1.0 - sel_emb @ centroid
        # Blend: items far from centroid get a novelty boost
        for i in range(n):
            per_item[i] = 0.7 * per_item[i] + 0.3 * float(dist_to_centroid[i])

        mean_nov = float(np.mean(per_item))
        max_nov = float(np.max(per_item))
        min_nov = float(np.min(per_item))
        n_novel = sum(1 for v in per_item if v > 0.5)

        return NoveltyMetrics(
            per_item_novelty=per_item,
            mean_novelty=mean_nov,
            max_novelty=max_nov,
            min_novelty=min_nov,
            n_highly_novel=n_novel,
        )

    # ------------------------------------------------------------------
    # Redundancy metrics
    # ------------------------------------------------------------------

    def _evaluate_redundancy(self, sel_emb: np.ndarray) -> RedundancyMetrics:
        """Evaluate pairwise redundancy in the selected set."""
        n = len(sel_emb)
        if n <= 1:
            return RedundancyMetrics(0.0, 0.0, 0, 0.0, float('inf'))

        sim = sel_emb @ sel_emb.T
        np.fill_diagonal(sim, 0.0)

        # Upper triangle values (avoid double counting)
        triu_idx = np.triu_indices(n, k=1)
        pairwise_sims = sim[triu_idx]

        mean_sim = float(np.mean(pairwise_sims))
        max_sim = float(np.max(pairwise_sims))

        # Near duplicates
        n_near_dup = int(np.sum(pairwise_sims >= self.near_duplicate_threshold))
        n_pairs = len(pairwise_sims)
        redundancy_ratio = n_near_dup / max(n_pairs, 1)

        # Dispersion (minimum pairwise distance)
        dists = self._pairwise_dist(sel_emb, sel_emb)
        np.fill_diagonal(dists, float('inf'))
        dispersion = float(np.min(dists))

        return RedundancyMetrics(
            mean_pairwise_similarity=mean_sim,
            max_pairwise_similarity=max_sim,
            n_near_duplicates=n_near_dup,
            redundancy_ratio=redundancy_ratio,
            dispersion=dispersion,
        )

    # ------------------------------------------------------------------
    # Preference alignment
    # ------------------------------------------------------------------

    def _evaluate_preference(
        self,
        sel_emb: np.ndarray,
        human_ranking: List[int],
    ) -> PreferenceAlignment:
        """Evaluate alignment between diversity ranking and human preferences.

        Args:
            sel_emb: Selected item embeddings.
            human_ranking: Human preference ranking (indices into selected,
                from best to worst).
        """
        n = len(sel_emb)

        # Diversity-based ranking: rank by novelty (1 - max sim to others)
        sim = sel_emb @ sel_emb.T
        np.fill_diagonal(sim, -1.0)
        max_sim = np.max(sim, axis=1)
        novelty = 1.0 - max_sim
        div_ranking = list(np.argsort(-novelty))

        # Kendall's tau (simplified: count concordant vs discordant pairs)
        concordant = 0
        discordant = 0
        hr_rank = np.zeros(n, dtype=int)
        dr_rank = np.zeros(n, dtype=int)
        for i, idx in enumerate(human_ranking):
            if idx < n:
                hr_rank[idx] = i
        for i, idx in enumerate(div_ranking):
            dr_rank[idx] = i
        for i in range(n):
            for j in range(i + 1, n):
                h_diff = hr_rank[i] - hr_rank[j]
                d_diff = dr_rank[i] - dr_rank[j]
                if h_diff * d_diff > 0:
                    concordant += 1
                elif h_diff * d_diff < 0:
                    discordant += 1
        n_pairs = max(n * (n - 1) / 2, 1)
        tau = (concordant - discordant) / n_pairs

        # Spearman's rho (rank correlation)
        d_sq = np.sum((hr_rank - dr_rank) ** 2)
        rho = 1.0 - 6.0 * d_sq / max(n * (n ** 2 - 1), 1)

        # Top-k overlap (top 3)
        top_k = min(3, n)
        human_top = set(human_ranking[:top_k])
        div_top = set(div_ranking[:top_k])
        overlap = len(human_top & div_top) / max(top_k, 1)

        # Preference satisfaction: what fraction of human top picks are "novel"
        satisfaction = 0.0
        for idx in human_ranking[:top_k]:
            if idx < n and novelty[idx] > np.median(novelty):
                satisfaction += 1.0
        satisfaction /= max(top_k, 1)

        return PreferenceAlignment(
            kendall_tau=float(tau),
            spearman_rho=float(rho),
            top_k_overlap=float(overlap),
            preference_satisfaction=float(satisfaction),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _embed(self, items: List[Any]) -> np.ndarray:
        """Embed items to numpy arrays."""
        if not items:
            return np.empty((0, self.embed_dim))
        if self.embed_fn is not None:
            return project_to_sphere(np.array([self.embed_fn(it) for it in items]))
        sample = items[0]
        if isinstance(sample, np.ndarray):
            return project_to_sphere(np.array(items))
        if isinstance(sample, str):
            return project_to_sphere(embed_texts(items, dim=self.embed_dim))
        raise TypeError(
            f"Cannot embed type {type(sample).__name__}. Provide embed_fn."
        )

    @staticmethod
    def _pairwise_dist(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Euclidean distance matrix."""
        sq_x = np.sum(X ** 2, axis=1, keepdims=True)
        sq_y = np.sum(Y ** 2, axis=1, keepdims=True)
        D2 = sq_x + sq_y.T - 2.0 * X @ Y.T
        return np.sqrt(np.maximum(D2, 0.0))


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def compare_methods(
    items: List[Any],
    selections: Dict[str, List[int]],
    *,
    embed_fn: Optional[Callable] = None,
    embed_dim: int = 64,
    human_ranking: Optional[List[int]] = None,
) -> Dict[str, EvaluationReport]:
    """Compare multiple selection methods using the evaluation suite.

    Args:
        items: Full candidate pool.
        selections: Dict mapping method name to list of selected indices.
        embed_fn: Embedding function.
        embed_dim: Embedding dimension.
        human_ranking: Optional human preference ranking.

    Returns:
        Dict mapping method name to EvaluationReport.
    """
    evaluator = DiversityEvaluator(
        embed_dim=embed_dim,
        embed_fn=embed_fn,
    )
    results: Dict[str, EvaluationReport] = {}
    for method, indices in selections.items():
        selected = [items[i] for i in indices]
        results[method] = evaluator.evaluate(
            selected, items,
            human_ranking=human_ranking,
            method_name=method,
        )
    return results


def quick_evaluate(
    selected: List[Any],
    candidates: Optional[List[Any]] = None,
    embed_fn: Optional[Callable] = None,
) -> Dict[str, float]:
    """Quick evaluation returning just a metrics dict.

    Args:
        selected: Selected items.
        candidates: Optional full candidate pool.
        embed_fn: Embedding function.

    Returns:
        Dict of metric name to value.
    """
    evaluator = DiversityEvaluator(embed_fn=embed_fn)
    report = evaluator.evaluate(selected, candidates)
    return report.metrics_dict
