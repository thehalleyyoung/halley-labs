"""Diverse survey and polling engine for DivFlow.

Generates synthetic diverse responses across audience segments, computes
diversity metrics, detects consensus and polarization, and produces
representative samples.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

from .coverage import estimate_coverage
from .diversity_metrics import cosine_diversity
from .embedding import TextEmbedder, embed_texts

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SurveyQuestion:
    """A single survey question."""

    text: str
    question_type: Literal["open", "likert", "multiple_choice", "ranking"]
    options: Optional[List[str]] = None
    required: bool = True


@dataclass
class ResponseRecord:
    """One respondent's answer to a single question."""

    question_id: int
    segment: str
    response: str
    confidence: float
    timestamp: float
    embedding: Optional[np.ndarray] = None


@dataclass
class SurveyResults:
    """Aggregated collection of survey responses."""

    questions: List[SurveyQuestion]
    responses: List[ResponseRecord]
    segments: List[str]
    response_rate: Dict[str, float]


@dataclass
class DiversityMetrics:
    """Diversity statistics for a set of survey responses."""

    coverage_score: float
    segment_balance: float
    perspective_spread: float
    unique_themes: int
    simpson_index: float


@dataclass
class SurveyAnalysis:
    """Full analysis output from a survey."""

    results: SurveyResults
    diversity: DiversityMetrics
    theme_clusters: Dict[str, List[str]]
    summary_by_segment: Dict[str, str]
    overall_summary: str


@dataclass
class ConsensusResult:
    """Result of consensus detection across segments."""

    consensus_level: float  # 0-1
    consensus_statement: str
    agreement_areas: List[str]
    disagreement_areas: List[str]
    holdout_segments: List[str]


@dataclass
class PolarizationMap:
    """Polarization analysis of survey responses."""

    polarization_score: float
    poles: List[Dict[str, Any]]
    bridge_positions: List[str]
    segment_alignments: Dict[str, int]


@dataclass
class Sample:
    """A representative sample drawn from the respondent pool."""

    members: List[str]
    representativeness_score: float
    segment_distribution: Dict[str, int]
    population_distribution: Dict[str, int]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _simpsons_diversity_index(counts: List[int]) -> float:
    """Compute Simpson's diversity index: 1 - sum(n_i*(n_i-1)) / (N*(N-1)).

    Returns 0 when all items belong to one group and approaches 1 when items
    are evenly distributed across many groups.
    """
    total = sum(counts)
    if total <= 1:
        return 0.0
    numerator = sum(n * (n - 1) for n in counts)
    denominator = total * (total - 1)
    return 1.0 - numerator / denominator


def _stratified_sample(
    segments: Dict[str, List[Any]],
    proportions: Dict[str, float],
    n: int,
) -> List[Any]:
    """Draw a stratified sample that follows *proportions*.

    Each segment contributes ``round(proportion * n)`` items.  When the total
    doesn't sum to *n* due to rounding, the remainder is drawn from the
    largest segments first.
    """
    rng = np.random.RandomState(42)
    allocation: Dict[str, int] = {}
    for seg, prop in proportions.items():
        allocation[seg] = int(np.floor(prop * n))

    # Distribute remainder by fractional parts (largest-remainder method)
    remainders = {
        seg: (prop * n) - allocation[seg] for seg, prop in proportions.items()
    }
    deficit = n - sum(allocation.values())
    for seg in sorted(remainders, key=remainders.get, reverse=True):  # type: ignore[arg-type]
        if deficit <= 0:
            break
        allocation[seg] += 1
        deficit -= 1

    sampled: List[Any] = []
    for seg, count in allocation.items():
        pool = segments.get(seg, [])
        if len(pool) == 0:
            continue
        count = min(count, len(pool))
        indices = rng.choice(len(pool), size=count, replace=False)
        sampled.extend(pool[i] for i in indices)
    return sampled


def _cluster_responses(embeddings: np.ndarray, n_clusters: int) -> List[int]:
    """Simple k-means style clustering on *embeddings*.

    Uses k-means++ initialisation followed by Lloyd iterations.
    Returns cluster assignment for each embedding.
    """
    n, d = embeddings.shape
    if n <= n_clusters:
        return list(range(n))

    rng = np.random.RandomState(0)

    # --- k-means++ initialisation ---
    centres = np.empty((n_clusters, d), dtype=embeddings.dtype)
    first_idx = rng.randint(n)
    centres[0] = embeddings[first_idx]
    for k in range(1, n_clusters):
        dists = np.min(
            np.linalg.norm(
                embeddings[:, None, :] - centres[None, :k, :], axis=2
            ),
            axis=1,
        )
        probs = dists ** 2
        probs_sum = probs.sum()
        if probs_sum > 0:
            probs /= probs_sum
        else:
            probs = np.ones(n) / n
        centres[k] = embeddings[rng.choice(n, p=probs)]

    # --- Lloyd iterations ---
    labels = np.zeros(n, dtype=int)
    for _ in range(50):
        dists = np.linalg.norm(
            embeddings[:, None, :] - centres[None, :, :], axis=2
        )
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for k in range(n_clusters):
            mask = labels == k
            if mask.any():
                centres[k] = embeddings[mask].mean(axis=0)

    return labels.tolist()


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class DiverseSurvey:
    """Diverse survey and polling engine.

    Generates synthetic diverse responses across audience segments, computes
    diversity metrics, detects consensus and polarization, and draws
    representative samples.
    """

    def __init__(
        self,
        questions: List[SurveyQuestion],
        audience_segments: List[str],
    ) -> None:
        self.questions = questions
        self.segments = audience_segments
        self._embedder = TextEmbedder(dim=64, seed=7)
        self._results: Optional[SurveyResults] = None

    # ------------------------------------------------------------------
    # Response collection
    # ------------------------------------------------------------------

    def collect(self, n_per_segment: int = 10) -> SurveyResults:
        """Generate synthetic diverse responses for each segment and question.

        Uses hash-based determinism to simulate different segment perspectives.
        Creates ``ResponseRecord`` objects with embeddings and tracks response
        rates per segment.
        """
        responses: List[ResponseRecord] = []
        response_counts: Dict[str, int] = {seg: 0 for seg in self.segments}
        expected_counts: Dict[str, int] = {
            seg: n_per_segment * len(self.questions) for seg in self.segments
        }

        for seg_idx, segment in enumerate(self.segments):
            for q_idx, question in enumerate(self.questions):
                for resp_idx in range(n_per_segment):
                    # Deterministic seed from segment + question + index
                    seed_str = f"{segment}|{question.text}|{resp_idx}"
                    seed_hash = int(
                        hashlib.sha256(seed_str.encode()).hexdigest(), 16
                    )
                    rng = np.random.RandomState(seed_hash % (2 ** 31))

                    response_text = self._synthesise_response(
                        question, segment, rng
                    )
                    confidence = float(np.clip(rng.beta(2 + seg_idx, 2), 0, 1))
                    ts = 1_700_000_000.0 + rng.uniform(0, 86_400)
                    emb = self._embedder.embed(
                        f"{segment} {question.text} {response_text}"
                    )

                    responses.append(
                        ResponseRecord(
                            question_id=q_idx,
                            segment=segment,
                            response=response_text,
                            confidence=confidence,
                            timestamp=ts,
                            embedding=emb,
                        )
                    )
                    response_counts[segment] += 1

        response_rate = {
            seg: response_counts[seg] / max(expected_counts[seg], 1)
            for seg in self.segments
        }

        self._results = SurveyResults(
            questions=self.questions,
            responses=responses,
            segments=self.segments,
            response_rate=response_rate,
        )
        return self._results

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def analyze(self) -> SurveyAnalysis:
        """Compute diversity metrics, cluster into themes, and summarise.

        Diversity metrics include Simpson's index for segment balance,
        coverage score via ``estimate_coverage``, and perspective spread
        via ``cosine_diversity``.
        """
        results = self._ensure_results()
        embeddings = self._all_embeddings(results)

        # --- Diversity metrics ---
        seg_counts = [
            sum(1 for r in results.responses if r.segment == seg)
            for seg in results.segments
        ]
        simpson = _simpsons_diversity_index(seg_counts)
        segment_balance = simpson

        coverage_cert = estimate_coverage(embeddings, epsilon=1.0)
        coverage_score = float(coverage_cert.coverage_fraction)

        perspective_spread = cosine_diversity(embeddings)

        # Cluster into themes
        n_clusters = min(max(3, len(results.responses) // 20), 12)
        labels = _cluster_responses(embeddings, n_clusters)
        theme_clusters: Dict[str, List[str]] = {}
        for idx, label in enumerate(labels):
            key = f"theme_{label}"
            theme_clusters.setdefault(key, []).append(
                results.responses[idx].response
            )
        unique_themes = len(theme_clusters)

        diversity = DiversityMetrics(
            coverage_score=coverage_score,
            segment_balance=segment_balance,
            perspective_spread=perspective_spread,
            unique_themes=unique_themes,
            simpson_index=simpson,
        )

        # --- Summaries per segment ---
        summary_by_segment: Dict[str, str] = {}
        for seg in results.segments:
            seg_responses = [
                r.response for r in results.responses if r.segment == seg
            ]
            n = len(seg_responses)
            top = seg_responses[:3]
            summary_by_segment[seg] = (
                f"Segment '{seg}': {n} responses. "
                f"Representative: {'; '.join(top)}."
            )

        overall_summary = (
            f"Survey with {len(results.questions)} questions across "
            f"{len(results.segments)} segments. "
            f"{len(results.responses)} total responses. "
            f"Simpson diversity={simpson:.3f}, "
            f"coverage={coverage_score:.3f}, "
            f"spread={perspective_spread:.3f}."
        )

        return SurveyAnalysis(
            results=results,
            diversity=diversity,
            theme_clusters=theme_clusters,
            summary_by_segment=summary_by_segment,
            overall_summary=overall_summary,
        )

    # ------------------------------------------------------------------
    # Consensus detection
    # ------------------------------------------------------------------

    def find_consensus(self) -> ConsensusResult:
        """Find areas of agreement and disagreement across segments.

        Computes pairwise segment agreement using mean cosine similarity
        of response embeddings.  High cross-segment similarity indicates
        consensus; low similarity indicates disagreement.
        """
        results = self._ensure_results()

        # Group embeddings by segment
        seg_embs: Dict[str, np.ndarray] = {}
        for seg in results.segments:
            embs = [
                r.embedding
                for r in results.responses
                if r.segment == seg and r.embedding is not None
            ]
            if embs:
                seg_embs[seg] = np.vstack(embs)

        # Pairwise mean cosine similarity between segment centroids
        centroids = {
            seg: emb.mean(axis=0) for seg, emb in seg_embs.items()
        }
        seg_list = list(centroids.keys())
        n_segs = len(seg_list)
        sim_matrix = np.zeros((n_segs, n_segs))
        for i in range(n_segs):
            for j in range(n_segs):
                a = centroids[seg_list[i]]
                b = centroids[seg_list[j]]
                na = np.linalg.norm(a)
                nb = np.linalg.norm(b)
                if na > 1e-12 and nb > 1e-12:
                    sim_matrix[i, j] = float(np.dot(a, b) / (na * nb))

        # Overall consensus = mean off-diagonal similarity
        if n_segs > 1:
            mask = ~np.eye(n_segs, dtype=bool)
            consensus_level = float(np.clip(np.mean(sim_matrix[mask]), 0, 1))
        else:
            consensus_level = 1.0

        # Identify agreement areas (question-level high similarity)
        agreement_areas: List[str] = []
        disagreement_areas: List[str] = []
        for q_idx, question in enumerate(results.questions):
            q_embs: Dict[str, np.ndarray] = {}
            for seg in results.segments:
                embs = [
                    r.embedding
                    for r in results.responses
                    if r.segment == seg
                    and r.question_id == q_idx
                    and r.embedding is not None
                ]
                if embs:
                    q_embs[seg] = np.vstack(embs).mean(axis=0)
            if len(q_embs) < 2:
                continue
            sims: List[float] = []
            keys = list(q_embs.keys())
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    a, b = q_embs[keys[i]], q_embs[keys[j]]
                    na, nb = np.linalg.norm(a), np.linalg.norm(b)
                    if na > 1e-12 and nb > 1e-12:
                        sims.append(float(np.dot(a, b) / (na * nb)))
            mean_sim = float(np.mean(sims)) if sims else 0.0
            if mean_sim > 0.3:
                agreement_areas.append(question.text)
            else:
                disagreement_areas.append(question.text)

        # Holdout segments: those most different from the overall centroid
        overall_centroid = np.mean(list(centroids.values()), axis=0)
        holdout_segments: List[str] = []
        for seg, c in centroids.items():
            nc = np.linalg.norm(c)
            no = np.linalg.norm(overall_centroid)
            if nc > 1e-12 and no > 1e-12:
                sim = float(np.dot(c, overall_centroid) / (nc * no))
                if sim < 0.2:
                    holdout_segments.append(seg)

        consensus_statement = (
            f"Consensus level {consensus_level:.2f} across "
            f"{len(results.segments)} segments with "
            f"{len(agreement_areas)} agreement areas."
        )

        return ConsensusResult(
            consensus_level=consensus_level,
            consensus_statement=consensus_statement,
            agreement_areas=agreement_areas,
            disagreement_areas=disagreement_areas,
            holdout_segments=holdout_segments,
        )

    # ------------------------------------------------------------------
    # Polarization detection
    # ------------------------------------------------------------------

    def find_polarization(self) -> PolarizationMap:
        """Detect polarized opinions via spectral analysis on embeddings.

        Uses eigengap heuristic on the similarity matrix to identify poles,
        and finds bridge positions between them.
        """
        results = self._ensure_results()
        embeddings = self._all_embeddings(results)
        n = embeddings.shape[0]

        # Build cosine similarity matrix
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        normed = embeddings / norms
        sim = normed @ normed.T

        # Degree matrix and graph Laplacian
        degree = np.diag(sim.sum(axis=1))
        laplacian = degree - sim

        # Eigenvalues of normalised Laplacian for spectral clustering
        d_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(np.diag(degree), 1e-12)))
        norm_lap = d_inv_sqrt @ laplacian @ d_inv_sqrt
        eigenvalues = np.sort(np.linalg.eigvalsh(norm_lap))

        # Eigengap heuristic: find largest gap in first few eigenvalues
        max_k = min(6, len(eigenvalues) - 1)
        gaps = np.diff(eigenvalues[: max_k + 1])
        n_poles = int(np.argmax(gaps) + 1) if len(gaps) > 0 else 2
        n_poles = max(2, min(n_poles, 4))

        # Cluster into poles
        labels = _cluster_responses(embeddings, n_poles)
        label_arr = np.array(labels)

        # Build pole descriptions
        poles: List[Dict[str, Any]] = []
        for k in range(n_poles):
            mask = label_arr == k
            member_indices = np.where(mask)[0]
            pole_responses = [
                results.responses[int(i)].response for i in member_indices[:5]
            ]
            pole_segments = [
                results.responses[int(i)].segment for i in member_indices
            ]
            poles.append(
                {
                    "id": k,
                    "size": int(mask.sum()),
                    "sample_responses": pole_responses,
                    "dominant_segment": max(
                        set(pole_segments), key=pole_segments.count
                    )
                    if pole_segments
                    else "",
                }
            )

        # Polarization score: variance of cluster sizes normalised by n
        sizes = np.array([p["size"] for p in poles], dtype=float)
        size_entropy = -np.sum(
            (sizes / n) * np.log(np.maximum(sizes / n, 1e-12))
        )
        max_entropy = np.log(max(n_poles, 1))
        polarization_score = float(
            1.0 - size_entropy / max_entropy if max_entropy > 0 else 0.0
        )

        # Bridge positions: responses closest to multiple cluster centroids
        centres = np.array(
            [embeddings[label_arr == k].mean(axis=0) for k in range(n_poles)]
        )
        dists_to_centres = np.linalg.norm(
            embeddings[:, None, :] - centres[None, :, :], axis=2
        )
        # Sort distances for each point; small gap ⇒ bridge
        sorted_dists = np.sort(dists_to_centres, axis=1)
        bridge_scores = sorted_dists[:, 1] - sorted_dists[:, 0]
        bridge_indices = np.argsort(bridge_scores)[:5]
        bridge_positions = [
            results.responses[int(i)].response for i in bridge_indices
        ]

        # Segment alignments: which pole each segment's centroid is closest to
        segment_alignments: Dict[str, int] = {}
        for seg in results.segments:
            seg_embs = [
                r.embedding
                for r in results.responses
                if r.segment == seg and r.embedding is not None
            ]
            if not seg_embs:
                segment_alignments[seg] = 0
                continue
            seg_centroid = np.mean(seg_embs, axis=0)
            dists = np.linalg.norm(centres - seg_centroid, axis=1)
            segment_alignments[seg] = int(np.argmin(dists))

        return PolarizationMap(
            polarization_score=polarization_score,
            poles=poles,
            bridge_positions=bridge_positions,
            segment_alignments=segment_alignments,
        )

    # ------------------------------------------------------------------
    # Representative sampling
    # ------------------------------------------------------------------

    def representative_sample(
        self,
        population: Dict[str, int],
        n: int,
    ) -> Sample:
        """Select a representative sample matching *population* distribution.

        Uses stratified sampling with proportional allocation and computes a
        representativeness score based on chi-squared-like distance between
        sample and population proportions.
        """
        results = self._ensure_results()

        # Build per-segment pools of response texts
        seg_pools: Dict[str, List[str]] = {}
        for seg in results.segments:
            seg_pools[seg] = [
                r.response for r in results.responses if r.segment == seg
            ]

        pop_total = sum(population.values())
        proportions = {
            seg: count / max(pop_total, 1) for seg, count in population.items()
        }

        sampled = _stratified_sample(seg_pools, proportions, n)

        # Compute actual segment distribution of the sample
        seg_dist: Dict[str, int] = {seg: 0 for seg in results.segments}
        seg_response_map: Dict[str, str] = {}
        for seg in results.segments:
            for r in results.responses:
                if r.segment == seg:
                    seg_response_map[r.response] = seg
        for item in sampled:
            seg = seg_response_map.get(item, "")
            if seg in seg_dist:
                seg_dist[seg] += 1

        # Representativeness: 1 - normalised chi-squared distance
        sample_total = max(len(sampled), 1)
        chi2 = 0.0
        for seg in proportions:
            expected = proportions[seg] * sample_total
            observed = seg_dist.get(seg, 0)
            if expected > 0:
                chi2 += (observed - expected) ** 2 / expected
        representativeness = float(np.clip(1.0 - chi2 / max(len(proportions), 1), 0, 1))

        return Sample(
            members=sampled,
            representativeness_score=representativeness,
            segment_distribution=seg_dist,
            population_distribution=dict(population),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_results(self) -> SurveyResults:
        """Return stored results or auto-collect."""
        if self._results is None:
            self.collect()
        assert self._results is not None
        return self._results

    def _all_embeddings(self, results: SurveyResults) -> np.ndarray:
        """Stack all response embeddings into an (N, D) array."""
        embs = []
        for r in results.responses:
            if r.embedding is not None:
                embs.append(r.embedding)
            else:
                embs.append(self._embedder.embed(r.response))
        return np.vstack(embs)

    @staticmethod
    def _synthesise_response(
        question: SurveyQuestion,
        segment: str,
        rng: np.random.RandomState,
    ) -> str:
        """Produce a deterministic synthetic response based on question type.

        Uses the segment name and question text to create realistic variation.
        """
        if question.question_type == "likert":
            scale = rng.randint(1, 6)
            return str(scale)

        if question.question_type == "multiple_choice" and question.options:
            idx = rng.randint(len(question.options))
            return question.options[idx]

        if question.question_type == "ranking" and question.options:
            perm = rng.permutation(len(question.options)).tolist()
            return ",".join(question.options[i] for i in perm)

        # Open-ended: hash-based deterministic phrase generation
        seed_int = int(
            hashlib.md5(f"{segment}:{question.text}".encode()).hexdigest(), 16
        )
        phrases = [
            f"From {segment} perspective",
            f"regarding {question.text[:30]}",
            f"priority-{(seed_int % 5) + 1}",
            f"approach-{'ABCDE'[seed_int % 5]}",
        ]
        selected = [phrases[i] for i in rng.choice(len(phrases), size=min(3, len(phrases)), replace=False)]
        return "; ".join(selected)
