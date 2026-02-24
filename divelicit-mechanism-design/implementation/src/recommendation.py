"""
Diverse recommendation algorithms.

Provides functions for generating recommendations that balance relevance
to a user profile with diversity across categories, novelty, and
serendipity.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np

from .embedding import TextEmbedder, embed_texts
from .dpp import greedy_map
from .diversity_metrics import cosine_diversity
from .kernels import RBFKernel
from .utils import log_det_safe


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Item:
    """A recommendable item."""
    id: str
    text: str
    category: str = ""
    embedding: Optional[np.ndarray] = None
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserProfile:
    """User preference profile."""
    interests: List[str] = field(default_factory=list)
    history_ids: Set[str] = field(default_factory=set)
    embedding: Optional[np.ndarray] = None
    category_weights: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_embeddings(
    items: List[Item], embedder: Optional[TextEmbedder] = None
) -> Tuple[List[Item], np.ndarray]:
    """Embed items that lack an embedding vector."""
    if embedder is None:
        embedder = TextEmbedder(dim=64)
    texts = [it.text for it in items]
    embs = embedder.embed_batch(texts)
    for i, it in enumerate(items):
        if it.embedding is None:
            it.embedding = embs[i]
    return items, embs


def _ensure_user_embedding(
    profile: UserProfile, embedder: Optional[TextEmbedder] = None
) -> np.ndarray:
    """Return (or compute) the user profile embedding."""
    if profile.embedding is not None:
        return profile.embedding
    if embedder is None:
        embedder = TextEmbedder(dim=64)
    if profile.interests:
        embs = embedder.embed_batch(profile.interests)
        profile.embedding = embs.mean(axis=0)
    else:
        profile.embedding = np.zeros(64)
    return profile.embedding


def _relevance_scores(
    items: List[Item],
    user_emb: np.ndarray,
    item_embs: np.ndarray,
) -> np.ndarray:
    """Cosine similarity between each item and the user profile."""
    u_norm = user_emb / (np.linalg.norm(user_emb) + 1e-12)
    i_norms = np.linalg.norm(item_embs, axis=1, keepdims=True)
    normed = item_embs / np.maximum(i_norms, 1e-12)
    return normed @ u_norm


# ---------------------------------------------------------------------------
# Diverse recommendations (DPP + relevance)
# ---------------------------------------------------------------------------

def diverse_recommendations(
    items: List[Item],
    user_profile: UserProfile,
    k: int = 10,
    diversity_weight: float = 0.5,
    embedder: Optional[TextEmbedder] = None,
) -> List[Item]:
    """Select *k* items that are both relevant to the user and diverse.

    Uses a quality-diversity DPP where the L-ensemble kernel is modulated
    by per-item relevance scores.

    Parameters
    ----------
    items : list of Item
        Candidate items.
    user_profile : UserProfile
        The user's preference profile.
    k : int
        Number of recommendations.
    diversity_weight : float
        Trade-off between relevance (0) and diversity (1).
    embedder : TextEmbedder, optional

    Returns
    -------
    list of Item
        Selected diverse recommendations.
    """
    if len(items) <= k:
        return list(items)

    items, embs = _ensure_embeddings(items, embedder)
    user_emb = _ensure_user_embedding(user_profile, embedder)

    rel = _relevance_scores(items, user_emb, embs)
    rel = (rel - rel.min()) / (rel.max() - rel.min() + 1e-12)

    # Build quality-diversity kernel
    kernel = RBFKernel()
    S = kernel.gram_matrix(embs)
    # Quality modulation
    q = (1.0 - diversity_weight) * rel + diversity_weight * np.ones(len(items))
    q = np.maximum(q, 1e-6)
    Q = np.diag(q)
    L = Q @ S @ Q + np.eye(len(items)) * 1e-6

    selected_idx = greedy_map(L, k)
    return [items[i] for i in selected_idx]


# ---------------------------------------------------------------------------
# Explore-exploit balance
# ---------------------------------------------------------------------------

def explore_exploit_balance(
    items: List[Item],
    history: List[Item],
    exploration_rate: float = 0.3,
    k: int = 10,
    embedder: Optional[TextEmbedder] = None,
) -> List[Item]:
    """Balance exploiting known preferences with exploring new territory.

    A fraction ``exploration_rate`` of the *k* slots are filled with items
    maximally distant from the user's history (explore); the rest are filled
    with items most similar to positively-rated history (exploit).

    Parameters
    ----------
    items : list of Item
        Candidate items.
    history : list of Item
        User interaction history (assumed positive signals).
    exploration_rate : float
        Fraction of slots reserved for exploration.
    k : int
        Number of recommendations.
    embedder : TextEmbedder, optional

    Returns
    -------
    list of Item
    """
    if len(items) <= k:
        return list(items)

    emb = embedder or TextEmbedder(dim=64)
    items, item_embs = _ensure_embeddings(items, emb)

    n_explore = max(1, int(k * exploration_rate))
    n_exploit = k - n_explore

    # History centroid
    if history:
        _, hist_embs = _ensure_embeddings(history, emb)
        centroid = hist_embs.mean(axis=0)
    else:
        centroid = item_embs.mean(axis=0)

    # Exploit: closest to centroid
    dists = np.linalg.norm(item_embs - centroid, axis=1)
    exploit_idx = np.argsort(dists)[:n_exploit].tolist()

    # Explore: farthest from centroid, then diversify via max-min
    remaining = [i for i in range(len(items)) if i not in exploit_idx]
    if not remaining:
        return [items[i] for i in exploit_idx]

    rem_embs = item_embs[remaining]
    rem_dists = np.linalg.norm(rem_embs - centroid, axis=1)
    far_order = np.argsort(-rem_dists)

    explore_pool = [remaining[far_order[i]] for i in range(min(len(far_order), n_explore * 3))]
    if len(explore_pool) <= n_explore:
        explore_idx = explore_pool
    else:
        # Greedy max-min within explore pool
        pool_embs = item_embs[explore_pool]
        norms = np.linalg.norm(pool_embs, axis=1, keepdims=True)
        normed = pool_embs / np.maximum(norms, 1e-12)
        sim = normed @ normed.T
        dist_mat = 1.0 - sim

        selected = [0]
        for _ in range(n_explore - 1):
            min_d = dist_mat[selected].min(axis=0)
            min_d[selected] = -np.inf
            selected.append(int(np.argmax(min_d)))
        explore_idx = [explore_pool[s] for s in selected]

    result_idx = exploit_idx + explore_idx
    return [items[i] for i in result_idx]


# ---------------------------------------------------------------------------
# Serendipity boost
# ---------------------------------------------------------------------------

def serendipity_boost(
    recommendations: List[Item],
    user_history: List[Item],
    boost_fraction: float = 0.2,
    embedder: Optional[TextEmbedder] = None,
) -> List[Item]:
    """Re-rank recommendations to boost serendipitous (surprising yet relevant) items.

    An item is serendipitous if it is *dissimilar* to the user's history
    but still has reasonable relevance.

    Parameters
    ----------
    recommendations : list of Item
        Initial recommendation list.
    user_history : list of Item
        User's past interactions.
    boost_fraction : float
        Fraction of recommendations to replace with serendipitous items.
    embedder : TextEmbedder, optional

    Returns
    -------
    list of Item
        Re-ranked recommendations.
    """
    if not user_history or len(recommendations) < 3:
        return list(recommendations)

    emb = embedder or TextEmbedder(dim=64)
    _, rec_embs = _ensure_embeddings(recommendations, emb)
    _, hist_embs = _ensure_embeddings(user_history, emb)
    hist_centroid = hist_embs.mean(axis=0)

    # Distance from history (novelty) and original rank (relevance proxy)
    dists = np.linalg.norm(rec_embs - hist_centroid, axis=1)
    dists_norm = (dists - dists.min()) / (dists.max() - dists.min() + 1e-12)

    n = len(recommendations)
    rank_scores = np.linspace(1.0, 0.0, n)  # higher for earlier

    # Serendipity = novelty * relevance proxy
    serendipity = dists_norm * (0.3 + 0.7 * rank_scores)

    n_boost = max(1, int(n * boost_fraction))
    boost_idx = np.argsort(-serendipity)[:n_boost]
    non_boost_idx = [i for i in range(n) if i not in boost_idx]

    # Interleave: place serendipitous items in spread-out positions
    result: List[Item] = []
    boost_list = [recommendations[i] for i in boost_idx]
    normal_list = [recommendations[i] for i in non_boost_idx]

    step = max(1, len(normal_list) // (n_boost + 1))
    bi = 0
    for i, item in enumerate(normal_list):
        if bi < len(boost_list) and i > 0 and i % step == 0:
            result.append(boost_list[bi])
            bi += 1
        result.append(item)
    # Append any remaining boosted items
    while bi < len(boost_list):
        result.append(boost_list[bi])
        bi += 1

    return result


# ---------------------------------------------------------------------------
# Category coverage
# ---------------------------------------------------------------------------

def category_coverage(
    items: List[Item],
    categories: List[str],
    k: int = 10,
    min_per_category: int = 1,
    embedder: Optional[TextEmbedder] = None,
) -> List[Item]:
    """Select *k* items ensuring coverage of all *categories*.

    First, ``min_per_category`` items are selected from each category
    (highest-scoring items).  Remaining slots are filled via DPP
    diversity selection over the full pool.

    Parameters
    ----------
    items : list of Item
        Candidate items; each should have a ``category`` attribute.
    categories : list of str
        Target categories.
    k : int
        Total number of items to select.
    min_per_category : int
        Minimum items from each category.
    embedder : TextEmbedder, optional

    Returns
    -------
    list of Item
    """
    # Group items by category
    cat_items: Dict[str, List[int]] = {c: [] for c in categories}
    for i, it in enumerate(items):
        if it.category in cat_items:
            cat_items[it.category].append(i)

    selected_set: Set[int] = set()

    # Guarantee min_per_category from each category
    for cat in categories:
        pool = cat_items.get(cat, [])
        if not pool:
            continue
        # Sort by score descending
        pool_sorted = sorted(pool, key=lambda i: items[i].score, reverse=True)
        for idx in pool_sorted[:min_per_category]:
            selected_set.add(idx)
            if len(selected_set) >= k:
                break
        if len(selected_set) >= k:
            break

    # Fill remaining slots with DPP diversity
    remaining_slots = k - len(selected_set)
    if remaining_slots > 0:
        remaining_idx = [i for i in range(len(items)) if i not in selected_set]
        if remaining_idx:
            items_copy, embs = _ensure_embeddings(items, embedder)
            rem_embs = embs[remaining_idx]
            kernel = RBFKernel()
            S = kernel.gram_matrix(rem_embs) + np.eye(len(remaining_idx)) * 1e-6
            dpp_sel = greedy_map(S, min(remaining_slots, len(remaining_idx)))
            for j in dpp_sel:
                selected_set.add(remaining_idx[j])

    return [items[i] for i in sorted(selected_set)]
