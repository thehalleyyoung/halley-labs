"""Team cognitive diversity optimization for DivFlow.

Provides multi-dimensional diversity assessment, gap analysis, optimal team
composition via submodular maximization, collaboration network analysis,
and groupthink detection for diverse LLM generation pipelines.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .coverage import estimate_coverage
from .diversity_metrics import cosine_diversity
from .embedding import TextEmbedder, embed_texts

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TeamMember:
    """A single team member profile."""

    name: str
    skills: List[str]
    thinking_style: str
    experience_years: float
    domain: str
    background: str


@dataclass
class TeamDiversityScore:
    """Multi-dimensional diversity assessment result."""

    overall_score: float
    cognitive_diversity: float
    skill_diversity: float
    experience_diversity: float
    domain_coverage: float
    dimension_scores: Dict[str, float]
    recommendations: List[str]


@dataclass
class Gap:
    """A gap between team capabilities and requirements."""

    area: str
    severity: float
    description: str
    suggested_profiles: List[str]


@dataclass
class NetworkAnalysis:
    """Result of collaboration network analysis."""

    density: float
    clustering_coefficient: float
    central_members: List[str]
    bridges: List[str]
    isolated_members: List[str]
    communication_health: float


@dataclass
class GroupthinkRisk:
    """Groupthink risk assessment for a discussion."""

    risk_level: float  # 0-1
    risk_category: str  # "low" / "medium" / "high" / "critical"
    indicators: List[str]
    contributing_factors: List[str]
    mitigation_strategies: List[str]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

_EXPECTED_THINKING_STYLES = {"analytical", "creative", "practical", "interpersonal"}
_DEFAULT_EXPECTED_DOMAINS = 6  # sensible baseline for domain coverage denominator

_DISAGREEMENT_WORDS = frozenset(
    [
        "disagree", "however", "but", "actually", "incorrect", "wrong",
        "alternatively", "contrast", "oppose", "challenge", "reconsider",
        "counterpoint", "issue", "concern", "differ", "doubt",
    ]
)


def _coefficient_of_variation(values: np.ndarray) -> float:
    """Return the coefficient of variation (std / mean).

    Returns 0.0 when the mean is zero to avoid division errors.
    """
    mean = np.mean(values)
    if mean == 0.0:
        return 0.0
    return float(np.std(values, ddof=0) / mean)


def _greedy_submodular_max(
    scores: np.ndarray, similarity: np.ndarray, k: int
) -> List[int]:
    """Greedy submodular maximisation for diversity-aware selection.

    At each step the candidate maximising *score_i + λ · marginal_diversity*
    is added, where marginal diversity is the minimum distance to every
    already-selected item.

    Parameters
    ----------
    scores:
        1-D relevance scores for each candidate (higher is better).
    similarity:
        Square similarity matrix (n × n) between candidates.
    k:
        Number of items to select.

    Returns
    -------
    List of selected indices.
    """
    n = len(scores)
    k = min(k, n)
    if k <= 0:
        return []

    # Distance matrix from similarity
    distance = 1.0 - np.clip(similarity, 0.0, 1.0)

    selected: List[int] = []
    remaining = set(range(n))

    # Seed with highest-scoring candidate
    first = int(np.argmax(scores))
    selected.append(first)
    remaining.discard(first)

    lam = 0.5  # trade-off between relevance and diversity

    while len(selected) < k and remaining:
        best_idx = -1
        best_gain = -np.inf
        for idx in remaining:
            min_dist = np.min(distance[idx, selected])
            gain = scores[idx] + lam * min_dist
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        selected.append(best_idx)
        remaining.discard(best_idx)

    return selected


def _adjacency_to_laplacian(adj: np.ndarray) -> np.ndarray:
    """Compute the graph Laplacian L = D - A from an adjacency matrix."""
    degree = np.diag(np.sum(adj, axis=1))
    return degree - adj


def _member_text(member: TeamMember) -> str:
    """Build a textual representation of a member for embedding."""
    return (
        f"{' '.join(member.skills)} {member.thinking_style} "
        f"{member.domain} {member.background}"
    )


def _hash_embed(text: str, dim: int = 64) -> np.ndarray:
    """Deterministic hash-based pseudo-embedding (no LLM call)."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    rng = np.random.RandomState(int.from_bytes(digest[:4], "big"))
    vec = rng.randn(dim).astype(np.float64)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


# ---------------------------------------------------------------------------
# Module-level functions
# ---------------------------------------------------------------------------


def assess_team_diversity(team_profiles: List[TeamMember]) -> TeamDiversityScore:
    """Compute a multi-dimensional diversity score for *team_profiles*.

    Dimensions assessed:
    * **Cognitive diversity** – cosine diversity of member profile embeddings.
    * **Skill diversity** – Simpson's diversity index over skill mentions.
    * **Experience diversity** – normalised coefficient of variation of tenure.
    * **Domain coverage** – unique domains / expected domain count.

    The overall score is a weighted average of the four dimensions.
    Recommendations are generated for any dimension scoring below 0.5.
    """
    if not team_profiles:
        return TeamDiversityScore(
            overall_score=0.0,
            cognitive_diversity=0.0,
            skill_diversity=0.0,
            experience_diversity=0.0,
            domain_coverage=0.0,
            dimension_scores={},
            recommendations=["Team is empty – add members to assess diversity."],
        )

    # --- Cognitive diversity via embeddings ---
    texts = [_member_text(m) for m in team_profiles]
    embeddings = np.array([_hash_embed(t) for t in texts])
    cognitive = float(cosine_diversity(embeddings))

    # --- Skill diversity (Simpson's index) ---
    all_skills: List[str] = []
    for m in team_profiles:
        all_skills.extend(s.lower().strip() for s in m.skills)
    skill_counts = Counter(all_skills)
    total_mentions = sum(skill_counts.values())
    if total_mentions > 1:
        simpson = 1.0 - sum(
            c * (c - 1) for c in skill_counts.values()
        ) / (total_mentions * (total_mentions - 1))
    else:
        simpson = 0.0
    skill_div = float(np.clip(simpson, 0.0, 1.0))

    # --- Experience diversity (CV, capped to [0, 1]) ---
    experience_vals = np.array(
        [m.experience_years for m in team_profiles], dtype=np.float64
    )
    cv = _coefficient_of_variation(experience_vals)
    experience_div = float(np.clip(cv, 0.0, 1.0))

    # --- Domain coverage ---
    unique_domains = len({m.domain.lower().strip() for m in team_profiles})
    domain_cov = float(
        np.clip(unique_domains / _DEFAULT_EXPECTED_DOMAINS, 0.0, 1.0)
    )

    # --- Weighted overall ---
    weights = {
        "cognitive_diversity": 0.35,
        "skill_diversity": 0.25,
        "experience_diversity": 0.15,
        "domain_coverage": 0.25,
    }
    dim_scores: Dict[str, float] = {
        "cognitive_diversity": cognitive,
        "skill_diversity": skill_div,
        "experience_diversity": experience_div,
        "domain_coverage": domain_cov,
    }
    overall = sum(dim_scores[k] * w for k, w in weights.items())

    # --- Recommendations ---
    recommendations: List[str] = []
    if cognitive < 0.5:
        recommendations.append(
            "Low cognitive diversity – consider adding members with different "
            "thinking styles and backgrounds."
        )
    if skill_div < 0.5:
        recommendations.append(
            "Skill diversity is low – broaden the range of technical and soft "
            "skills represented on the team."
        )
    if experience_div < 0.5:
        recommendations.append(
            "Experience levels are homogeneous – mix junior and senior members "
            "for richer perspective."
        )
    if domain_cov < 0.5:
        recommendations.append(
            "Domain coverage is narrow – recruit from additional domains to "
            "improve cross-functional breadth."
        )

    return TeamDiversityScore(
        overall_score=overall,
        cognitive_diversity=cognitive,
        skill_diversity=skill_div,
        experience_diversity=experience_div,
        domain_coverage=domain_cov,
        dimension_scores=dim_scores,
        recommendations=recommendations,
    )


def diversity_gaps(
    team: List[TeamMember], required_skills: List[str]
) -> List[Gap]:
    """Identify gaps between current team capabilities and requirements.

    Checks skill coverage, thinking-style coverage, experience range, and
    domain coverage.  For every detected gap a severity (0-1) and suggested
    ideal profiles are returned.
    """
    gaps: List[Gap] = []

    # --- Skill coverage ---
    team_skills = {s.lower().strip() for m in team for s in m.skills}
    missing_skills = [
        s for s in required_skills if s.lower().strip() not in team_skills
    ]
    if missing_skills:
        severity = len(missing_skills) / max(len(required_skills), 1)
        gaps.append(
            Gap(
                area="skills",
                severity=float(np.clip(severity, 0.0, 1.0)),
                description=(
                    f"Missing required skills: {', '.join(missing_skills)}."
                ),
                suggested_profiles=[
                    f"Specialist in {s}" for s in missing_skills[:5]
                ],
            )
        )

    # --- Thinking-style coverage ---
    team_styles = {m.thinking_style.lower().strip() for m in team}
    missing_styles = _EXPECTED_THINKING_STYLES - team_styles
    if missing_styles:
        severity = len(missing_styles) / len(_EXPECTED_THINKING_STYLES)
        gaps.append(
            Gap(
                area="thinking_style",
                severity=float(severity),
                description=(
                    f"Missing thinking styles: {', '.join(sorted(missing_styles))}."
                ),
                suggested_profiles=[
                    f"{style.capitalize()} thinker" for style in sorted(missing_styles)
                ],
            )
        )

    # --- Experience range ---
    if team:
        exps = np.array([m.experience_years for m in team], dtype=np.float64)
        exp_range = float(np.ptp(exps))
        if exp_range < 5.0:
            gaps.append(
                Gap(
                    area="experience",
                    severity=float(np.clip(1.0 - exp_range / 5.0, 0.0, 1.0)),
                    description=(
                        f"Experience range is only {exp_range:.1f} years – "
                        "consider mixing junior and senior profiles."
                    ),
                    suggested_profiles=[
                        "Junior contributor (< 2 years)",
                        "Senior expert (> 10 years)",
                    ],
                )
            )

    # --- Domain coverage ---
    team_domains = {m.domain.lower().strip() for m in team}
    if len(team_domains) < 3:
        severity = 1.0 - len(team_domains) / 3.0
        gaps.append(
            Gap(
                area="domain",
                severity=float(np.clip(severity, 0.0, 1.0)),
                description=(
                    f"Only {len(team_domains)} domain(s) represented – "
                    "cross-functional coverage is limited."
                ),
                suggested_profiles=["Cross-domain generalist", "Adjacent-field expert"],
            )
        )

    return gaps


def optimal_team_composition(
    candidates: List[TeamMember], team_size: int, task: str
) -> List[str]:
    """Select an optimally diverse team via greedy submodular maximisation.

    Each candidate receives a *task relevance* score (hash-based cosine
    similarity to the task description) and a *diversity bonus* computed
    during greedy selection.  The function returns the names of the chosen
    members.
    """
    if not candidates:
        return []
    team_size = min(team_size, len(candidates))

    # Task-relevance scores
    task_vec = _hash_embed(task)
    member_vecs = np.array([_hash_embed(_member_text(c)) for c in candidates])
    relevance = member_vecs @ task_vec  # cosine (vectors are unit-norm)
    relevance = (relevance - relevance.min()) / (relevance.ptp() + 1e-9)

    # Pairwise similarity matrix
    similarity = member_vecs @ member_vecs.T

    selected_indices = _greedy_submodular_max(relevance, similarity, team_size)
    return [candidates[i].name for i in selected_indices]


def collaboration_network(
    team: List[TeamMember],
    interactions: List[Tuple[str, str, float]],
) -> NetworkAnalysis:
    """Analyse a team's collaboration network.

    Parameters
    ----------
    team:
        Team member profiles (used for name→index mapping).
    interactions:
        Triples of (name_a, name_b, weight) representing interaction
        intensity between members.

    Returns
    -------
    NetworkAnalysis with density, clustering coefficient, central /
    bridge / isolated members, and an overall communication-health score.
    """
    n = len(team)
    if n < 2:
        names = [m.name for m in team]
        return NetworkAnalysis(
            density=0.0,
            clustering_coefficient=0.0,
            central_members=names,
            bridges=[],
            isolated_members=[],
            communication_health=0.0,
        )

    name_to_idx = {m.name: i for i, m in enumerate(team)}
    adj = np.zeros((n, n), dtype=np.float64)

    for a, b, w in interactions:
        ia, ib = name_to_idx.get(a), name_to_idx.get(b)
        if ia is not None and ib is not None and ia != ib:
            adj[ia, ib] = max(adj[ia, ib], w)
            adj[ib, ia] = max(adj[ib, ia], w)

    binary = (adj > 0).astype(np.float64)

    # --- Density ---
    possible_edges = n * (n - 1) / 2.0
    actual_edges = np.sum(binary) / 2.0
    density = actual_edges / possible_edges if possible_edges > 0 else 0.0

    # --- Clustering coefficient (average local) ---
    degrees = binary.sum(axis=1)
    clustering_vals: List[float] = []
    for i in range(n):
        neighbours = np.where(binary[i] > 0)[0]
        ki = len(neighbours)
        if ki < 2:
            clustering_vals.append(0.0)
            continue
        triangles = sum(
            1
            for a_idx in range(ki)
            for b_idx in range(a_idx + 1, ki)
            if binary[neighbours[a_idx], neighbours[b_idx]] > 0
        )
        clustering_vals.append(2.0 * triangles / (ki * (ki - 1)))
    clustering_coeff = float(np.mean(clustering_vals))

    # --- Central members (top-k by weighted degree) ---
    weighted_deg = adj.sum(axis=1)
    top_k = min(3, n)
    central_idx = np.argsort(weighted_deg)[-top_k:][::-1]
    central_members = [team[i].name for i in central_idx if weighted_deg[i] > 0]

    # --- Bridges (connect otherwise weakly-connected groups) ---
    # Approximate: members whose removal increases Laplacian Fiedler value change
    laplacian = _adjacency_to_laplacian(binary)
    eigvals = np.sort(np.linalg.eigvalsh(laplacian))
    fiedler = eigvals[1] if n > 1 else 0.0
    bridges: List[str] = []
    for i in range(n):
        reduced = np.delete(np.delete(binary, i, axis=0), i, axis=1)
        if reduced.shape[0] < 2:
            continue
        lap_r = _adjacency_to_laplacian(reduced)
        eig_r = np.sort(np.linalg.eigvalsh(lap_r))
        fiedler_r = eig_r[1] if len(eig_r) > 1 else 0.0
        if fiedler_r < fiedler * 0.5:
            bridges.append(team[i].name)

    # --- Isolated members ---
    mean_deg = np.mean(degrees)
    std_deg = np.std(degrees)
    threshold = max(mean_deg - 2.0 * std_deg, 0.0)
    isolated_members = [
        team[i].name for i in range(n) if degrees[i] <= threshold
    ]

    # --- Communication health (heuristic) ---
    health = float(
        np.clip(
            0.4 * density
            + 0.3 * clustering_coeff
            + 0.3 * (1.0 - len(isolated_members) / max(n, 1)),
            0.0,
            1.0,
        )
    )

    return NetworkAnalysis(
        density=float(density),
        clustering_coefficient=clustering_coeff,
        central_members=central_members,
        bridges=bridges,
        isolated_members=isolated_members,
        communication_health=health,
    )


def groupthink_detector(
    discussion_history: List[Dict[str, str]],
) -> GroupthinkRisk:
    """Detect groupthink risk from a list of discussion messages.

    Each entry in *discussion_history* should have keys ``"author"`` and
    ``"message"``.  The detector analyses four signals:

    1. **Opinion diversity** – embedding spread of unique messages.
    2. **Agreement speed** – how quickly positions converge.
    3. **Dissent frequency** – proportion of messages containing
       disagreement language.
    4. **Leadership dominance** – imbalance in message count and length.

    Returns a ``GroupthinkRisk`` with a 0-1 risk level, categorical label,
    observed indicators, contributing factors, and mitigation strategies.
    """
    if not discussion_history:
        return GroupthinkRisk(
            risk_level=0.0,
            risk_category="low",
            indicators=[],
            contributing_factors=["No discussion data available."],
            mitigation_strategies=[],
        )

    messages = [d.get("message", "") for d in discussion_history]
    authors = [d.get("author", "unknown") for d in discussion_history]

    # --- 1. Opinion diversity (embedding spread) ---
    embeddings = np.array([_hash_embed(m) for m in messages])
    if len(embeddings) > 1:
        sim_matrix = embeddings @ embeddings.T
        upper = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
        mean_sim = float(np.mean(upper))
        opinion_diversity = 1.0 - mean_sim  # high similarity → low diversity
    else:
        opinion_diversity = 0.0

    # --- 2. Agreement speed (cosine between successive message pairs) ---
    if len(embeddings) > 2:
        half = len(embeddings) // 2
        first_half_sim = float(
            np.mean(embeddings[:half] @ embeddings[:half].T)
        )
        second_half_sim = float(
            np.mean(embeddings[half:] @ embeddings[half:].T)
        )
        convergence = max(second_half_sim - first_half_sim, 0.0)
    else:
        convergence = 0.0

    # --- 3. Dissent frequency ---
    dissent_count = 0
    for msg in messages:
        tokens = set(re.findall(r"\w+", msg.lower()))
        if tokens & _DISAGREEMENT_WORDS:
            dissent_count += 1
    dissent_ratio = dissent_count / max(len(messages), 1)
    low_dissent_signal = 1.0 - dissent_ratio  # lack of dissent → risk

    # --- 4. Leadership dominance ---
    author_counts = Counter(authors)
    counts = np.array(list(author_counts.values()), dtype=np.float64)
    if len(counts) > 1:
        dominance = float(counts.max() / counts.sum())
    else:
        dominance = 1.0

    author_lengths: Dict[str, int] = {}
    for auth, msg in zip(authors, messages):
        author_lengths[auth] = author_lengths.get(auth, 0) + len(msg)
    length_vals = np.array(list(author_lengths.values()), dtype=np.float64)
    if len(length_vals) > 1:
        length_dominance = float(length_vals.max() / length_vals.sum())
    else:
        length_dominance = 1.0
    leadership_dom = (dominance + length_dominance) / 2.0

    # --- Composite risk ---
    risk_level = float(
        np.clip(
            0.30 * (1.0 - opinion_diversity)
            + 0.20 * convergence
            + 0.25 * low_dissent_signal
            + 0.25 * leadership_dom,
            0.0,
            1.0,
        )
    )

    # --- Categorise ---
    if risk_level < 0.3:
        risk_category = "low"
    elif risk_level < 0.55:
        risk_category = "medium"
    elif risk_level < 0.8:
        risk_category = "high"
    else:
        risk_category = "critical"

    # --- Indicators ---
    indicators: List[str] = []
    if opinion_diversity < 0.3:
        indicators.append("Very low opinion diversity among messages.")
    if convergence > 0.2:
        indicators.append("Rapid convergence of positions detected.")
    if dissent_ratio < 0.1:
        indicators.append("Almost no dissenting language observed.")
    if leadership_dom > 0.6:
        indicators.append("Discussion dominated by a single participant.")

    # --- Contributing factors ---
    contributing: List[str] = []
    if len(set(authors)) < 3:
        contributing.append("Very few unique contributors.")
    if len(messages) < 5:
        contributing.append("Discussion is too short for robust analysis.")
    if dominance > 0.5:
        contributing.append("Unequal participation in message count.")

    # --- Mitigation strategies ---
    strategies: List[str] = []
    if risk_category in {"high", "critical"}:
        strategies.append("Introduce a designated devil's advocate role.")
        strategies.append("Use anonymous idea-submission rounds.")
    if opinion_diversity < 0.3:
        strategies.append("Encourage independent idea generation before discussion.")
    if leadership_dom > 0.6:
        strategies.append("Implement round-robin speaking to equalise participation.")
    if dissent_ratio < 0.1:
        strategies.append("Explicitly invite counter-arguments and alternative views.")

    return GroupthinkRisk(
        risk_level=risk_level,
        risk_category=risk_category,
        indicators=indicators,
        contributing_factors=contributing,
        mitigation_strategies=strategies,
    )
