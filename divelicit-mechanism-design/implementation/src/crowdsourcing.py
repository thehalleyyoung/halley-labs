"""Crowdsource diverse ideas with human+AI mixed elicitation.

Provides an API for collecting ideas from multiple human and AI sources,
deduplicating via semantic similarity, tracking coverage of the idea space,
and maintaining a quality-diversity portfolio.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from .embedding import TextEmbedder, embed_texts, project_to_sphere
from .diversity_metrics import cosine_diversity, sinkhorn_diversity_metric as sinkhorn_diversity
from .coverage import CoverageCertificate, estimate_coverage, _effective_dimension
from .transport import sinkhorn_candidate_scores


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Idea:
    """A single idea with metadata."""
    id: str
    text: str
    source: str  # "human", "ai", or specific agent name
    timestamp: float = field(default_factory=time.time)
    embedding: Optional[np.ndarray] = None
    quality_score: float = 0.5
    novelty_score: float = 0.0
    cluster_id: int = -1
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class IdeaCluster:
    """A cluster of related ideas."""
    cluster_id: int
    representative_id: str
    member_ids: List[str]
    centroid: Optional[np.ndarray] = None
    label: str = ""


@dataclass
class CoverageReport:
    """Report on how well the idea space is covered."""
    certificate: Optional[CoverageCertificate]
    total_ideas: int
    unique_ideas: int
    n_clusters: int
    coverage_gaps: List[str]
    effective_dim: int
    diversity_score: float
    per_source_counts: Dict[str, int]
    per_source_unique: Dict[str, int]


@dataclass
class QualityDiversityItem:
    """An item in the quality-diversity portfolio."""
    idea_id: str
    quality: float
    novelty: float
    qd_score: float  # combined quality-diversity score


@dataclass
class QDPortfolio:
    """Quality-diversity portfolio tracking."""
    items: List[QualityDiversityItem]
    pareto_front: List[str]  # idea IDs on the Pareto front
    mean_quality: float
    mean_novelty: float
    portfolio_diversity: float


# ---------------------------------------------------------------------------
# Core crowdsourcing class
# ---------------------------------------------------------------------------

class IdeaCrowdsourcer:
    """API for human+AI mixed idea elicitation with deduplication.

    Collects ideas from multiple sources, deduplicates semantically,
    tracks coverage of the idea space, and maintains a quality-diversity
    portfolio.

    Example::

        cs = IdeaCrowdsourcer(embed_dim=64)
        cs.add_idea("Use solar panels on rooftops", source="human")
        cs.add_idea("Solar rooftop installations", source="ai")  # duplicate
        cs.add_idea("Wind turbines in urban areas", source="human")

        print(cs.unique_count)       # 2
        report = cs.coverage_report()
        print(report.diversity_score)
    """

    def __init__(
        self,
        embed_dim: int = 64,
        similarity_threshold: float = 0.85,
        quality_fn: Optional[Callable[[str], float]] = None,
    ):
        self.embed_dim = embed_dim
        self.similarity_threshold = similarity_threshold
        self.quality_fn = quality_fn
        self._embedder = TextEmbedder(dim=embed_dim)
        self._ideas: Dict[str, Idea] = {}
        self._embeddings: List[np.ndarray] = []
        self._idea_order: List[str] = []  # maintains insertion order
        self._next_id = 0
        self._clusters: Optional[List[IdeaCluster]] = None

    @property
    def count(self) -> int:
        """Total number of ideas (including duplicates)."""
        return len(self._ideas)

    @property
    def unique_count(self) -> int:
        """Number of unique (non-duplicate) ideas."""
        return sum(1 for idea in self._ideas.values() if not idea.is_duplicate)

    @property
    def ideas(self) -> List[Idea]:
        """All ideas in insertion order."""
        return [self._ideas[iid] for iid in self._idea_order]

    @property
    def unique_ideas(self) -> List[Idea]:
        """Only non-duplicate ideas."""
        return [idea for idea in self.ideas if not idea.is_duplicate]

    def add_idea(
        self,
        text: str,
        source: str = "unknown",
        quality: Optional[float] = None,
        tags: Optional[List[str]] = None,
    ) -> Tuple[str, bool]:
        """Add an idea to the pool.

        Returns:
            (idea_id, is_new) — is_new is False if the idea was a duplicate.
        """
        idea_id = f"idea_{self._next_id}"
        self._next_id += 1

        # Embed
        embedding = project_to_sphere(
            self._embedder.embed(text).reshape(1, -1)
        )[0]

        # Quality score
        if quality is not None:
            q = quality
        elif self.quality_fn is not None:
            q = self.quality_fn(text)
        else:
            words = text.split()
            q = 0.5 * min(len(words) / 50.0, 1.0) + 0.5 * (
                len(set(words)) / max(len(words), 1)
            )

        idea = Idea(
            id=idea_id,
            text=text,
            source=source,
            embedding=embedding,
            quality_score=q,
            tags=tags or [],
        )

        # Check for duplicates
        is_new = True
        if self._embeddings:
            existing_emb = np.array(self._embeddings)
            sims = embedding @ existing_emb.T
            max_sim_idx = int(np.argmax(sims))
            max_sim = float(sims[max_sim_idx])
            if max_sim >= self.similarity_threshold:
                idea.is_duplicate = True
                idea.duplicate_of = self._idea_order[max_sim_idx]
                is_new = False

        # Compute novelty: average distance to all existing unique ideas
        if self._embeddings:
            unique_emb = np.array([
                self._embeddings[i] for i, iid in enumerate(self._idea_order)
                if not self._ideas[iid].is_duplicate
            ]) if self.unique_count > 0 else np.empty((0, self.embed_dim))
            if len(unique_emb) > 0:
                sims_unique = embedding @ unique_emb.T
                idea.novelty_score = float(1.0 - np.mean(sims_unique))
            else:
                idea.novelty_score = 1.0
        else:
            idea.novelty_score = 1.0

        self._ideas[idea_id] = idea
        self._embeddings.append(embedding)
        self._idea_order.append(idea_id)
        self._clusters = None  # invalidate cache

        return idea_id, is_new

    def add_batch(
        self,
        texts: List[str],
        source: str = "unknown",
    ) -> List[Tuple[str, bool]]:
        """Add multiple ideas at once."""
        return [self.add_idea(t, source=source) for t in texts]

    def get_idea(self, idea_id: str) -> Optional[Idea]:
        """Retrieve an idea by ID."""
        return self._ideas.get(idea_id)

    def remove_idea(self, idea_id: str) -> bool:
        """Remove an idea."""
        if idea_id not in self._ideas:
            return False
        idx = self._idea_order.index(idea_id)
        del self._ideas[idea_id]
        self._idea_order.pop(idx)
        self._embeddings.pop(idx)
        self._clusters = None
        return True

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    def cluster_ideas(self, n_clusters: Optional[int] = None) -> List[IdeaCluster]:
        """Cluster unique ideas using k-medoids.

        Args:
            n_clusters: Number of clusters. Default: sqrt(n_unique).

        Returns:
            List of IdeaClusters.
        """
        unique = self.unique_ideas
        if len(unique) < 2:
            if unique:
                cluster = IdeaCluster(0, unique[0].id, [unique[0].id])
                self._clusters = [cluster]
            else:
                self._clusters = []
            return self._clusters

        if n_clusters is None:
            n_clusters = max(2, int(np.sqrt(len(unique))))
        n_clusters = min(n_clusters, len(unique))

        emb = np.array([idea.embedding for idea in unique])
        emb = project_to_sphere(emb)

        # k-medoids via farthest-point init + iterative swap
        dists = self._pairwise_dist(emb)
        medoids = [0]
        for _ in range(n_clusters - 1):
            min_d = np.min(dists[:, medoids], axis=1)
            medoids.append(int(np.argmax(min_d)))

        for _ in range(10):
            assignments = np.argmin(dists[:, medoids], axis=1)
            changed = False
            for mi in range(n_clusters):
                members = np.where(assignments == mi)[0]
                if len(members) == 0:
                    continue
                costs = np.sum(dists[np.ix_(members, members)], axis=1)
                new_med = members[int(np.argmin(costs))]
                if new_med != medoids[mi]:
                    medoids[mi] = new_med
                    changed = True
            if not changed:
                break

        assignments = np.argmin(dists[:, medoids], axis=1)
        clusters: List[IdeaCluster] = []
        for ci in range(n_clusters):
            member_indices = np.where(assignments == ci)[0]
            member_ids = [unique[j].id for j in member_indices]
            rep_id = unique[medoids[ci]].id
            centroid = np.mean(emb[member_indices], axis=0) if len(member_indices) > 0 else None
            for j in member_indices:
                unique[j].cluster_id = ci
            clusters.append(IdeaCluster(
                cluster_id=ci,
                representative_id=rep_id,
                member_ids=member_ids,
                centroid=centroid,
            ))

        self._clusters = clusters
        return clusters

    # ------------------------------------------------------------------
    # Coverage analysis
    # ------------------------------------------------------------------

    def coverage_report(self, epsilon: float = 0.3) -> CoverageReport:
        """Generate a coverage report for the current idea pool.

        Args:
            epsilon: Coverage ball radius.

        Returns:
            CoverageReport with metrics and gap analysis.
        """
        unique = self.unique_ideas
        if not unique:
            return CoverageReport(
                certificate=None, total_ideas=0, unique_ideas=0,
                n_clusters=0, coverage_gaps=[], effective_dim=0,
                diversity_score=0.0, per_source_counts={}, per_source_unique={},
            )

        emb = np.array([idea.embedding for idea in unique])
        emb = project_to_sphere(emb)

        # Coverage certificate
        cert = estimate_coverage(emb, epsilon=epsilon)

        # Clustering for gap analysis
        if self._clusters is None:
            self.cluster_ideas()
        clusters = self._clusters or []

        # Identify underrepresented clusters (fewer than mean members)
        mean_size = len(unique) / max(len(clusters), 1)
        gaps = []
        for c in clusters:
            if len(c.member_ids) < mean_size * 0.5:
                rep = self._ideas.get(c.representative_id)
                if rep:
                    gaps.append(f"Cluster {c.cluster_id}: only {len(c.member_ids)} ideas "
                                f"(e.g., '{rep.text[:60]}...')")

        # Effective dimension
        eff_dim = _effective_dimension(emb)

        # Diversity
        div_score = float(sinkhorn_diversity(emb, reg=0.1)) if len(emb) > 1 else 0.0

        # Per-source stats
        source_counts: Dict[str, int] = {}
        source_unique: Dict[str, int] = {}
        for idea in self.ideas:
            source_counts[idea.source] = source_counts.get(idea.source, 0) + 1
            if not idea.is_duplicate:
                source_unique[idea.source] = source_unique.get(idea.source, 0) + 1

        return CoverageReport(
            certificate=cert,
            total_ideas=self.count,
            unique_ideas=self.unique_count,
            n_clusters=len(clusters),
            coverage_gaps=gaps,
            effective_dim=eff_dim,
            diversity_score=div_score,
            per_source_counts=source_counts,
            per_source_unique=source_unique,
        )

    # ------------------------------------------------------------------
    # Quality-diversity portfolio
    # ------------------------------------------------------------------

    def qd_portfolio(self, qd_weight: float = 0.5) -> QDPortfolio:
        """Compute a quality-diversity portfolio.

        Ranks ideas by combined quality and novelty, identifies the
        Pareto front (ideas not dominated in both quality and novelty).

        Args:
            qd_weight: Weight for quality vs. novelty. 0=all novelty, 1=all quality.

        Returns:
            QDPortfolio with rankings and Pareto front.
        """
        unique = self.unique_ideas
        if not unique:
            return QDPortfolio([], [], 0.0, 0.0, 0.0)

        items: List[QualityDiversityItem] = []
        for idea in unique:
            qd_score = qd_weight * idea.quality_score + (1 - qd_weight) * idea.novelty_score
            items.append(QualityDiversityItem(
                idea_id=idea.id,
                quality=idea.quality_score,
                novelty=idea.novelty_score,
                qd_score=qd_score,
            ))

        items.sort(key=lambda x: x.qd_score, reverse=True)

        # Pareto front: find non-dominated items
        pareto: List[str] = []
        for item in items:
            dominated = False
            for other in items:
                if other.idea_id == item.idea_id:
                    continue
                if other.quality >= item.quality and other.novelty >= item.novelty:
                    if other.quality > item.quality or other.novelty > item.novelty:
                        dominated = True
                        break
            if not dominated:
                pareto.append(item.idea_id)

        mean_q = float(np.mean([it.quality for it in items]))
        mean_n = float(np.mean([it.novelty for it in items]))

        # Portfolio diversity
        emb = np.array([self._ideas[it.idea_id].embedding for it in items])
        emb = project_to_sphere(emb)
        port_div = float(cosine_diversity(emb)) if len(emb) > 1 else 0.0

        return QDPortfolio(
            items=items,
            pareto_front=pareto,
            mean_quality=mean_q,
            mean_novelty=mean_n,
            portfolio_diversity=port_div,
        )

    # ------------------------------------------------------------------
    # Suggest next idea
    # ------------------------------------------------------------------

    def suggest_gap_direction(self) -> Optional[str]:
        """Suggest what kind of idea would most increase coverage.

        Returns a textual hint about the underrepresented region,
        or None if coverage is already high.
        """
        report = self.coverage_report()
        if report.coverage_gaps:
            return (
                f"The idea space has gaps. Consider ideas related to: "
                f"{report.coverage_gaps[0]}"
            )
        if report.unique_ideas < 5:
            return "More ideas needed — the pool is still small."
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pairwise_dist(X: np.ndarray) -> np.ndarray:
        """Euclidean pairwise distance matrix."""
        sq = np.sum(X ** 2, axis=1, keepdims=True)
        D2 = sq + sq.T - 2.0 * X @ X.T
        return np.sqrt(np.maximum(D2, 0.0))
