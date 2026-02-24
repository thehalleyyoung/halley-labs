"""DivElicit public API — high-level interface for diverse LLM elicitation.

Usage:
    from divelicit import elicit_diverse, select_diverse_subset, compute_coverage_certificate

    result = elicit_diverse("What are creative uses for paperclips?", agents, k=5)
    subset = select_diverse_subset(items, k=3, embed_fn=my_embedder)
    cert = compute_coverage_certificate(selected, universe)
    comparison = mechanism_compare(items, k=5, methods=["flow", "mmr", "dpp"])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .mechanism import MMRMechanism, KMedoidsMechanism, DirectMechanism
from .coverage import CoverageCertificate, estimate_coverage
from .diversity_metrics import (
    cosine_diversity,
    log_det_diversity,
    sinkhorn_diversity_metric as sinkhorn_diversity,
    mmd as mmd_diversity,
)
from .dpp import DPP, greedy_map
from .embedding import TextEmbedder, embed_texts, project_to_sphere
from .kernels import RBFKernel, AdaptiveRBFKernel
from .transport import sinkhorn_divergence, sinkhorn_candidate_scores


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ElicitationResult:
    """Result from eliciting diverse responses from multiple agents.

    Attributes:
        responses: The text responses selected for diversity.
        embeddings: Embedding vectors for the selected responses.
        diversity_score: Aggregate diversity measure (Sinkhorn divergence).
        quality_scores: Per-response quality scores if available.
        agent_contributions: Maps agent index to number of responses selected.
        all_responses: Full set of candidate responses before selection.
        selected_indices: Indices into all_responses of chosen items.
        coverage: Optional coverage certificate for the selected set.
    """
    responses: List[str]
    embeddings: np.ndarray
    diversity_score: float
    quality_scores: List[float]
    agent_contributions: Dict[int, int]
    all_responses: List[str]
    selected_indices: List[int]
    coverage: Optional[CoverageCertificate] = None


@dataclass
class CoverageResult:
    """Coverage certificate with diagnostic information.

    Wraps CoverageCertificate with additional metadata about the analysis.
    """
    certificate: CoverageCertificate
    fill_distance: float
    dispersion: float
    effective_dim: int
    n_selected: int
    n_universe: int


@dataclass
class ComparisonResult:
    """Result of comparing multiple selection mechanisms.

    Attributes:
        method_results: Per-method selection results.
        rankings: Methods ranked by each metric (higher is better).
        best_method: Method with highest aggregate score.
        metrics: Raw metric values per method.
    """
    method_results: Dict[str, List[int]]
    rankings: Dict[str, List[str]]
    best_method: str
    metrics: Dict[str, Dict[str, float]]


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def elicit_diverse(
    prompt: str,
    agents: List[Callable],
    k: int,
    *,
    n_per_agent: int = 5,
    quality_weight: float = 0.3,
    sinkhorn_epsilon: float = 0.1,
    embed_fn: Optional[Callable[[str], np.ndarray]] = None,
    embed_dim: int = 64,
) -> ElicitationResult:
    """Elicit diverse responses from multiple agents, then select a diverse subset.

    Each agent callable should accept a prompt string and return a string response.
    Agents are queried ``n_per_agent`` times each, producing a candidate pool.
    The top-k diverse responses are selected using Sinkhorn dual potentials.

    Args:
        prompt: The prompt to send to each agent.
        agents: List of callables ``agent(prompt) -> str``.
        k: Number of diverse responses to select.
        n_per_agent: Number of times to query each agent.
        quality_weight: Weight for quality vs. diversity in [0, 1].
        sinkhorn_epsilon: Regularization for Sinkhorn divergence.
        embed_fn: Custom embedding function ``text -> np.ndarray``.
            Falls back to deterministic hash embeddings if None.
        embed_dim: Embedding dimension when using default embedder.

    Returns:
        ElicitationResult with selected responses and diagnostics.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    if not agents:
        raise ValueError("Must provide at least one agent")

    # --- Collect candidate responses from all agents ---
    all_responses: List[str] = []
    agent_ids: List[int] = []
    for agent_idx, agent in enumerate(agents):
        for _ in range(n_per_agent):
            try:
                resp = agent(prompt)
                if isinstance(resp, str):
                    all_responses.append(resp)
                    agent_ids.append(agent_idx)
            except Exception:
                continue

    if len(all_responses) == 0:
        raise RuntimeError("All agents failed to produce responses")

    k = min(k, len(all_responses))

    # --- Embed ---
    if embed_fn is not None:
        embeddings = np.array([embed_fn(r) for r in all_responses])
    else:
        embeddings = embed_texts(all_responses, dim=embed_dim)

    embeddings = project_to_sphere(embeddings)

    # --- Quality heuristic: length + lexical diversity ---
    quality_scores = _estimate_quality(all_responses)

    # --- Select via Sinkhorn dual potentials ---
    selected_indices = _flow_select(embeddings, quality_scores, k,
                                     quality_weight=quality_weight,
                                     sinkhorn_epsilon=sinkhorn_epsilon)

    # --- Compute diversity ---
    sel_emb = embeddings[selected_indices]
    div_score = float(sinkhorn_diversity(sel_emb, reg=sinkhorn_epsilon))

    # --- Agent contribution map ---
    contributions: Dict[int, int] = {}
    for idx in selected_indices:
        aid = agent_ids[idx]
        contributions[aid] = contributions.get(aid, 0) + 1

    # --- Coverage certificate ---
    cert = estimate_coverage(sel_emb, epsilon=0.3)

    return ElicitationResult(
        responses=[all_responses[i] for i in selected_indices],
        embeddings=sel_emb,
        diversity_score=div_score,
        quality_scores=[quality_scores[i] for i in selected_indices],
        agent_contributions=contributions,
        all_responses=all_responses,
        selected_indices=list(selected_indices),
        coverage=cert,
    )


def select_diverse_subset(
    items: List[Any],
    k: int,
    *,
    embed_fn: Optional[Callable[[Any], np.ndarray]] = None,
    quality_fn: Optional[Callable[[Any], float]] = None,
    quality_weight: float = 0.3,
    method: str = "flow",
    sinkhorn_epsilon: float = 0.1,
) -> List[Any]:
    """Select k diverse items from a list.

    Works on any list — strings, dicts, objects — as long as ``embed_fn``
    can map each item to a numpy vector.

    Args:
        items: Pool of candidate items.
        k: Number of items to select.
        embed_fn: Maps item to embedding vector. Defaults to hash embeddings
            for strings, or treats items as numpy arrays if they already are.
        quality_fn: Maps item to quality score in [0, 1]. Default: uniform.
        quality_weight: Trade-off between quality (1.0) and diversity (0.0).
        method: Selection method — "flow", "mmr", "dpp", "kmedoids".
        sinkhorn_epsilon: Regularization for Sinkhorn (flow method only).

    Returns:
        List of k selected items (in selection order).
    """
    if not items:
        return []
    k = min(k, len(items))
    if k == len(items):
        return list(items)

    # --- Embed ---
    embeddings = _embed_items(items, embed_fn)
    embeddings = project_to_sphere(embeddings)

    # --- Quality ---
    if quality_fn is not None:
        quality_scores = np.array([quality_fn(it) for it in items])
    else:
        quality_scores = np.ones(len(items))

    # --- Select ---
    indices = _select_by_method(
        embeddings, quality_scores, k, method,
        quality_weight=quality_weight,
        sinkhorn_epsilon=sinkhorn_epsilon,
    )
    return [items[i] for i in indices]


def compute_coverage_certificate(
    selected: List[Any],
    universe: List[Any],
    *,
    embed_fn: Optional[Callable[[Any], np.ndarray]] = None,
    epsilon: float = 0.3,
    confidence: float = 0.95,
) -> CoverageResult:
    """Compute a coverage certificate for a selected subset against a universe.

    Measures how well the selected set covers the embedding space defined
    by the universe, using Hoeffding-based finite-sample bounds.

    Args:
        selected: The chosen subset.
        universe: The full candidate pool.
        embed_fn: Embedding function for items.
        epsilon: Coverage ball radius.
        confidence: Desired confidence level (e.g. 0.95).

    Returns:
        CoverageResult with certificate and diagnostics.
    """
    sel_emb = project_to_sphere(_embed_items(selected, embed_fn))
    uni_emb = project_to_sphere(_embed_items(universe, embed_fn))

    # Coverage certificate
    cert = estimate_coverage(sel_emb, epsilon=epsilon)

    # Fill distance (max nearest-neighbor distance from universe to selected)
    from scipy.spatial.distance import cdist
    try:
        dists = cdist(uni_emb, sel_emb)
    except Exception:
        dists = _pairwise_distances(uni_emb, sel_emb)
    nn_dists = np.min(dists, axis=1)
    fill_dist = float(np.max(nn_dists))
    dispersion_val = float(np.mean(nn_dists))

    # Effective dimension
    from .coverage import _effective_dimension
    eff_dim = _effective_dimension(sel_emb)

    return CoverageResult(
        certificate=cert,
        fill_distance=fill_dist,
        dispersion=dispersion_val,
        effective_dim=eff_dim,
        n_selected=len(selected),
        n_universe=len(universe),
    )


def mechanism_compare(
    items: List[Any],
    k: int,
    methods: Optional[List[str]] = None,
    *,
    embed_fn: Optional[Callable[[Any], np.ndarray]] = None,
    quality_fn: Optional[Callable[[Any], float]] = None,
    quality_weight: float = 0.3,
) -> ComparisonResult:
    """Compare multiple selection mechanisms on the same candidate pool.

    Runs each method and reports diversity/quality metrics so users can
    choose the best approach for their data.

    Args:
        items: Candidate pool.
        k: Selection budget.
        methods: List of method names. Default: ["flow", "mmr", "dpp", "kmedoids"].
        embed_fn: Embedding function.
        quality_fn: Quality scoring function.
        quality_weight: Quality vs. diversity trade-off.

    Returns:
        ComparisonResult comparing all methods.
    """
    if methods is None:
        methods = ["flow", "mmr", "dpp", "kmedoids"]

    embeddings = project_to_sphere(_embed_items(items, embed_fn))
    if quality_fn is not None:
        quality_scores = np.array([quality_fn(it) for it in items])
    else:
        quality_scores = np.ones(len(items))

    k = min(k, len(items))

    method_results: Dict[str, List[int]] = {}
    metrics: Dict[str, Dict[str, float]] = {}

    for method in methods:
        indices = _select_by_method(
            embeddings, quality_scores, k, method, quality_weight=quality_weight,
        )
        method_results[method] = list(indices)

        sel_emb = embeddings[indices]
        sel_q = quality_scores[indices]

        # Compute metrics
        pairwise = _pairwise_distances(sel_emb, sel_emb)
        np.fill_diagonal(pairwise, np.inf)
        dispersion = float(np.min(pairwise)) if len(indices) > 1 else 0.0
        np.fill_diagonal(pairwise, 0.0)

        metrics[method] = {
            "cosine_diversity": float(cosine_diversity(sel_emb)),
            "dispersion": dispersion,
            "mean_quality": float(np.mean(sel_q)),
            "sinkhorn_diversity": float(sinkhorn_diversity(sel_emb, reg=0.1)),
        }

    # Rankings per metric (higher is better)
    metric_names = ["cosine_diversity", "dispersion", "mean_quality", "sinkhorn_diversity"]
    rankings: Dict[str, List[str]] = {}
    for mn in metric_names:
        ranked = sorted(methods, key=lambda m: metrics[m][mn], reverse=True)
        rankings[mn] = ranked

    # Best method: most first-place finishes, tie-break by sinkhorn diversity
    first_place_counts = {m: 0 for m in methods}
    for mn in metric_names:
        first_place_counts[rankings[mn][0]] += 1
    best = max(methods, key=lambda m: (first_place_counts[m], metrics[m]["sinkhorn_diversity"]))

    return ComparisonResult(
        method_results=method_results,
        rankings=rankings,
        best_method=best,
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _embed_items(
    items: List[Any], embed_fn: Optional[Callable] = None
) -> np.ndarray:
    """Embed a list of items to numpy array."""
    if embed_fn is not None:
        return np.array([embed_fn(it) for it in items])
    # Auto-detect: numpy arrays, strings, or fail
    if len(items) == 0:
        return np.empty((0, 64))
    sample = items[0]
    if isinstance(sample, np.ndarray):
        return np.array(items)
    if isinstance(sample, str):
        return embed_texts(items, dim=64)
    raise TypeError(
        f"Cannot auto-embed items of type {type(sample).__name__}. "
        "Provide embed_fn."
    )


def _estimate_quality(texts: List[str]) -> np.ndarray:
    """Heuristic quality score based on length and lexical diversity."""
    scores = np.zeros(len(texts))
    for i, t in enumerate(texts):
        words = t.split()
        length_score = min(len(words) / 100.0, 1.0)
        unique_ratio = len(set(words)) / max(len(words), 1)
        scores[i] = 0.5 * length_score + 0.5 * unique_ratio
    return scores


def _flow_select(
    embeddings: np.ndarray,
    quality_scores: np.ndarray,
    k: int,
    quality_weight: float = 0.3,
    sinkhorn_epsilon: float = 0.1,
) -> List[int]:
    """Greedy Sinkhorn-potential selection."""
    n = embeddings.shape[0]
    selected: List[int] = []
    remaining = set(range(n))
    reference = embeddings  # full pool is the reference distribution

    for _ in range(k):
        if not remaining:
            break
        rem_list = sorted(remaining)
        if not selected:
            # First pick: highest quality
            best_global = rem_list[int(np.argmax(quality_scores[rem_list]))]
        else:
            scores = sinkhorn_candidate_scores(
                embeddings[rem_list],
                embeddings[selected],
                reference,
                reg=sinkhorn_epsilon,
            )
            combined = (
                (1 - quality_weight) * scores
                + quality_weight * quality_scores[rem_list]
            )
            best_local = int(np.argmax(combined))
            best_global = rem_list[best_local]
        selected.append(best_global)
        remaining.discard(best_global)
    return selected


def _select_by_method(
    embeddings: np.ndarray,
    quality_scores: np.ndarray,
    k: int,
    method: str,
    quality_weight: float = 0.3,
    sinkhorn_epsilon: float = 0.1,
) -> List[int]:
    """Dispatch to the appropriate selection method."""
    n = embeddings.shape[0]

    if method == "flow":
        return _flow_select(embeddings, quality_scores, k,
                            quality_weight=quality_weight,
                            sinkhorn_epsilon=sinkhorn_epsilon)

    elif method == "mmr":
        return _mmr_select(embeddings, quality_scores, k, quality_weight)

    elif method == "dpp":
        kernel = RBFKernel(bandwidth=1.0)
        L = kernel.gram_matrix(embeddings)
        # Add quality on diagonal
        L += np.diag(quality_scores * quality_weight)
        return greedy_map(L, k)

    elif method == "kmedoids":
        return _kmedoids_select(embeddings, k)

    else:
        raise ValueError(f"Unknown method: {method}. Choose from: flow, mmr, dpp, kmedoids")


def _mmr_select(
    embeddings: np.ndarray,
    quality_scores: np.ndarray,
    k: int,
    lam: float = 0.5,
) -> List[int]:
    """Maximal Marginal Relevance selection."""
    n = embeddings.shape[0]
    sim = embeddings @ embeddings.T
    selected: List[int] = []
    remaining = set(range(n))

    # First: highest quality
    first = int(np.argmax(quality_scores))
    selected.append(first)
    remaining.discard(first)

    for _ in range(k - 1):
        if not remaining:
            break
        rem_list = sorted(remaining)
        max_sim_to_selected = np.max(sim[np.ix_(rem_list, selected)], axis=1)
        mmr_scores = (
            lam * quality_scores[rem_list]
            - (1 - lam) * max_sim_to_selected
        )
        best_local = int(np.argmax(mmr_scores))
        best_global = rem_list[best_local]
        selected.append(best_global)
        remaining.discard(best_global)
    return selected


def _kmedoids_select(embeddings: np.ndarray, k: int) -> List[int]:
    """Simple k-medoids via iterative swap."""
    n = embeddings.shape[0]
    dists = _pairwise_distances(embeddings, embeddings)

    # Initialize with farthest-point sampling
    rng = np.random.RandomState(42)
    medoids = [rng.randint(n)]
    for _ in range(k - 1):
        min_dists = np.min(dists[:, medoids], axis=1)
        medoids.append(int(np.argmax(min_dists)))

    # Swap refinement (up to 10 iterations)
    for _ in range(10):
        # Assign to nearest medoid
        assignments = np.argmin(dists[:, medoids], axis=1)
        changed = False
        for mi in range(k):
            cluster = np.where(assignments == mi)[0]
            if len(cluster) == 0:
                continue
            costs = np.sum(dists[np.ix_(cluster, cluster)], axis=1)
            new_medoid = cluster[int(np.argmin(costs))]
            if new_medoid != medoids[mi]:
                medoids[mi] = new_medoid
                changed = True
        if not changed:
            break
    return medoids


def _pairwise_distances(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix without scipy dependency."""
    x_sq = np.sum(X ** 2, axis=1, keepdims=True)
    y_sq = np.sum(Y ** 2, axis=1, keepdims=True)
    D2 = x_sq + y_sq.T - 2.0 * X @ Y.T
    return np.sqrt(np.maximum(D2, 0.0))
