"""Structured deliberation processes for collective decision-making.

Implements deliberative democracy methods including structured debate,
Socratic dialogue, Toulmin argument mapping, consensus building,
deliberative polling (Fishkin), and citizens' assemblies with sortition.

Unlike debate.py (which focuses on adversarial debate with diversity
constraints), this module models cooperative deliberation where
participants seek shared understanding and consensus through structured
processes.
"""

from __future__ import annotations

import itertools
import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Position:
    """A participant's stance represented as a vector in opinion space.

    The vector encodes opinions on multiple dimensions of an issue.
    Values typically range from -1 (strongly disagree) to +1 (strongly agree).
    """

    vector: np.ndarray
    conviction: float = 1.0
    label: str = ""

    def distance_to(self, other: Position) -> float:
        """Euclidean distance between two positions."""
        return float(np.linalg.norm(self.vector - other.vector))

    def copy(self) -> Position:
        return Position(
            vector=self.vector.copy(),
            conviction=self.conviction,
            label=self.label,
        )


@dataclass
class Statement:
    """A deliberative statement with Toulmin model components.

    Attributes:
        claim: The central assertion being made.
        grounds: Evidence or facts supporting the claim.
        warrant: The reasoning connecting grounds to claim.
        backing: Support for the warrant itself.
        qualifier: Degree of certainty (0-1 scale).
        rebuttal: Conditions under which the claim would not hold.
        embedding: Optional vector representation for similarity computation.
    """

    claim: str
    grounds: str = ""
    warrant: str = ""
    backing: str = ""
    qualifier: float = 1.0
    rebuttal: str = ""
    embedding: Optional[np.ndarray] = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


@dataclass
class Participant:
    """A participant in a deliberative process.

    Attributes:
        id: Unique identifier.
        name: Display name.
        position: Current position in opinion space.
        expertise_weights: Per-issue expertise levels (0-1).
        open_mindedness: How much the participant updates toward others (0-1).
        position_history: Track of position changes over time.
    """

    id: str
    name: str
    position: Position
    expertise_weights: Optional[np.ndarray] = None
    open_mindedness: float = 0.5
    position_history: List[np.ndarray] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.position_history.append(self.position.vector.copy())

    def record_position(self) -> None:
        """Snapshot current position into history."""
        self.position_history.append(self.position.vector.copy())


@dataclass
class Argument:
    """An argument node in an argument map.

    Includes Toulmin components plus graph metadata for
    support/attack relationship tracking.
    """

    id: str
    statement: Statement
    supports: List[str] = field(default_factory=list)
    attacks: List[str] = field(default_factory=list)
    strength: float = 0.0
    centrality: float = 0.0

    @property
    def claim(self) -> str:
        return self.statement.claim


@dataclass
class ArgumentMap:
    """A directed graph of arguments with support/attack relationships.

    Attributes:
        arguments: Map from argument ID to Argument.
        support_edges: List of (source_id, target_id, weight) support relations.
        attack_edges: List of (source_id, target_id, weight) attack relations.
        strongest_arguments: IDs of the highest-strength arguments.
        central_claims: IDs of arguments with highest centrality.
        key_rebuttals: IDs of arguments that serve as key rebuttals.
    """

    arguments: Dict[str, Argument] = field(default_factory=dict)
    support_edges: List[Tuple[str, str, float]] = field(default_factory=list)
    attack_edges: List[Tuple[str, str, float]] = field(default_factory=list)
    strongest_arguments: List[str] = field(default_factory=list)
    central_claims: List[str] = field(default_factory=list)
    key_rebuttals: List[str] = field(default_factory=list)

    @property
    def n_arguments(self) -> int:
        return len(self.arguments)

    @property
    def n_edges(self) -> int:
        return len(self.support_edges) + len(self.attack_edges)


@dataclass
class DebateResult:
    """Result of a structured multi-round debate.

    Attributes:
        final_positions: Each participant's final position.
        argument_graph: Directed graph of all arguments made.
        consensus_measure: 0-1 measure of final agreement (1 = full consensus).
        rounds_data: Per-round statistics (novelty, quality, position spread).
        position_trajectories: Per-participant position history.
        total_arguments: Total number of arguments submitted.
    """

    final_positions: List[Position]
    argument_graph: ArgumentMap
    consensus_measure: float
    rounds_data: List[Dict[str, Any]]
    position_trajectories: Dict[str, List[np.ndarray]]
    total_arguments: int


@dataclass
class DialogueResult:
    """Result of a Socratic dialogue.

    Attributes:
        final_positions: Participants' final positions.
        position_evolution: Per-participant list of position vectors over time.
        questioner_sequence: Who was the questioner at each iteration.
        convergence_iteration: Iteration at which convergence was reached, or -1.
        final_spread: Standard deviation of final positions.
        challenges_made: Total number of challenges issued.
    """

    final_positions: List[Position]
    position_evolution: Dict[str, List[np.ndarray]]
    questioner_sequence: List[str]
    convergence_iteration: int
    final_spread: float
    challenges_made: int


@dataclass
class ConsensusResult:
    """Result of a consensus-building process.

    Attributes:
        final_positions: Positions after consensus building.
        agreement_level: 0-1 measure of agreement (inverse of spread).
        convergence_round: Round at which convergence was detected, or -1.
        holdout_positions: Positions that did not move significantly toward consensus.
        position_history: Per-round list of all positions.
        centroid_history: Per-round centroid position.
    """

    final_positions: List[Position]
    agreement_level: float
    convergence_round: int
    holdout_positions: List[Position]
    position_history: List[List[np.ndarray]]
    centroid_history: List[np.ndarray]


@dataclass
class PollResult:
    """Result of a deliberative polling process.

    Attributes:
        pre_poll_positions: Initial positions before deliberation.
        post_poll_positions: Final positions after deliberation.
        opinion_change: Per-participant magnitude of position change.
        polarization_before: Polarization measure before deliberation.
        polarization_after: Polarization measure after deliberation.
        information_gain: Estimated information gain from the process.
        group_assignments: Which small group each participant was assigned to.
        phase_positions: Positions at each phase boundary.
    """

    pre_poll_positions: List[np.ndarray]
    post_poll_positions: List[np.ndarray]
    opinion_change: List[float]
    polarization_before: float
    polarization_after: float
    information_gain: float
    group_assignments: Dict[str, int]
    phase_positions: Dict[str, List[np.ndarray]]


@dataclass
class AssemblyResult:
    """Result of a citizens' assembly simulation.

    Attributes:
        recommendations: Per-issue recommendation vectors (centroid of final positions).
        confidence_levels: Per-issue confidence (inverse of final spread).
        minority_reports: Per-issue list of dissenting position summaries.
        selected_participants: IDs of participants chosen by sortition.
        voting_results: Per-issue dict with approval counts and thresholds.
        group_assignments: Per-issue mapping of participant to discussion group.
        position_trajectories: Per-issue per-participant position evolution.
    """

    recommendations: Dict[str, np.ndarray]
    confidence_levels: Dict[str, float]
    minority_reports: Dict[str, List[Dict[str, Any]]]
    selected_participants: List[str]
    voting_results: Dict[str, Dict[str, Any]]
    group_assignments: Dict[str, Dict[str, int]]
    position_trajectories: Dict[str, Dict[str, List[np.ndarray]]]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _compute_novelty(
    argument_vec: np.ndarray,
    prior_arguments: List[np.ndarray],
) -> float:
    """Compute novelty of an argument relative to prior arguments.

    Novelty is defined as the minimum cosine distance to any prior argument.
    Returns 1.0 if there are no prior arguments.  Values are clipped to [0, 1].

    Args:
        argument_vec: Embedding vector of the new argument.
        prior_arguments: List of embedding vectors of previous arguments.

    Returns:
        Novelty score in [0, 1].
    """
    if len(prior_arguments) == 0:
        return 1.0
    norm_a = np.linalg.norm(argument_vec)
    if norm_a < 1e-12:
        return 0.0
    similarities: List[float] = []
    for prior in prior_arguments:
        norm_p = np.linalg.norm(prior)
        if norm_p < 1e-12:
            continue
        cos_sim = float(np.dot(argument_vec, prior) / (norm_a * norm_p))
        similarities.append(cos_sim)
    if not similarities:
        return 1.0
    max_similarity = max(similarities)
    novelty = 1.0 - max(0.0, max_similarity)
    return float(np.clip(novelty, 0.0, 1.0))


def _update_positions(
    positions: np.ndarray,
    influence_matrix: np.ndarray,
) -> np.ndarray:
    """Update positions based on an influence matrix.

    Each position moves toward a weighted average of all positions,
    where weights come from the influence matrix.  The influence matrix
    should have rows that sum to 1 (or will be normalized).

    Args:
        positions: (n, d) array of position vectors.
        influence_matrix: (n, n) matrix where entry (i, j) is the
            influence of participant j on participant i.

    Returns:
        Updated (n, d) positions array.
    """
    n = positions.shape[0]
    row_sums = influence_matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-12, 1.0, row_sums)
    normed = influence_matrix / row_sums
    return normed @ positions


def _assign_diverse_groups(
    participants: List[Participant],
    n_groups: int,
) -> Dict[str, int]:
    """Assign participants to groups maximizing inter-group diversity.

    Uses a round-robin assignment on participants sorted by their position's
    first principal component, which ensures that each group gets a spread
    of viewpoints rather than clustering similar participants together.

    Args:
        participants: List of participants to assign.
        n_groups: Number of groups to create.

    Returns:
        Mapping from participant ID to group index.
    """
    if n_groups <= 0:
        n_groups = 1
    n_groups = min(n_groups, len(participants))

    positions = np.array([p.position.vector for p in participants])
    centroid = positions.mean(axis=0)
    centered = positions - centroid

    if centered.shape[1] > 1:
        # Project onto first principal component for sorting
        cov = centered.T @ centered
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        pc1 = eigenvectors[:, -1]
        projections = centered @ pc1
    else:
        projections = centered[:, 0]

    sorted_indices = np.argsort(projections)
    assignments: Dict[str, int] = {}
    for rank, idx in enumerate(sorted_indices):
        group = rank % n_groups
        assignments[participants[idx].id] = group

    return assignments


def _measure_polarization(positions: np.ndarray) -> float:
    """Measure polarization as the variance of positions.

    High variance indicates polarized opinions; low variance indicates
    consensus.  The measure is the mean variance across all opinion
    dimensions, which is equivalent to the mean squared distance to
    the centroid.

    Args:
        positions: (n, d) array of position vectors.

    Returns:
        Non-negative polarization score.
    """
    if len(positions) < 2:
        return 0.0
    centroid = positions.mean(axis=0)
    diffs = positions - centroid
    return float(np.mean(np.sum(diffs ** 2, axis=1)))


def _convergence_check(
    positions_history: List[np.ndarray],
    threshold: float = 0.01,
) -> bool:
    """Check whether positions have converged.

    Convergence is detected when the maximum change in any position
    between the last two snapshots falls below the threshold.

    Args:
        positions_history: List of (n, d) position arrays over time.
        threshold: Maximum per-participant movement for convergence.

    Returns:
        True if converged.
    """
    if len(positions_history) < 2:
        return False
    prev = positions_history[-2]
    curr = positions_history[-1]
    per_participant_change = np.linalg.norm(curr - prev, axis=1)
    return bool(np.max(per_participant_change) < threshold)


def _toulmin_parse(statement: Statement) -> Dict[str, Any]:
    """Extract Toulmin model components from a Statement.

    Returns a dictionary with all six Toulmin elements plus metadata
    useful for argument mapping.

    Args:
        statement: A Statement dataclass instance.

    Returns:
        Dictionary of Toulmin components and derived metadata.
    """
    has_grounds = bool(statement.grounds.strip()) if statement.grounds else False
    has_warrant = bool(statement.warrant.strip()) if statement.warrant else False
    has_backing = bool(statement.backing.strip()) if statement.backing else False
    has_rebuttal = bool(statement.rebuttal.strip()) if statement.rebuttal else False

    completeness = sum([
        1.0,  # claim always present
        0.2 if has_grounds else 0.0,
        0.2 if has_warrant else 0.0,
        0.2 if has_backing else 0.0,
        0.2 if has_rebuttal else 0.0,
    ])
    # qualifier contributes to completeness weighting
    completeness *= statement.qualifier

    return {
        "claim": statement.claim,
        "grounds": statement.grounds,
        "warrant": statement.warrant,
        "backing": statement.backing,
        "qualifier": statement.qualifier,
        "rebuttal": statement.rebuttal,
        "has_grounds": has_grounds,
        "has_warrant": has_warrant,
        "has_backing": has_backing,
        "has_rebuttal": has_rebuttal,
        "completeness": completeness,
        "id": statement.id,
    }


def _generate_embedding(text: str, dim: int = 32, seed: Optional[int] = None) -> np.ndarray:
    """Generate a deterministic pseudo-embedding from text.

    Uses a hash-based RNG seeded by the text content so the same text
    always produces the same embedding.  The result is L2-normalized.

    Args:
        text: Input string.
        dim: Embedding dimensionality.
        seed: Optional extra seed for variation.

    Returns:
        Unit-norm vector of shape (dim,).
    """
    hash_val = hash(text) & 0xFFFFFFFF
    if seed is not None:
        hash_val = (hash_val ^ seed) & 0xFFFFFFFF
    rng = np.random.RandomState(hash_val)
    vec = rng.randn(dim)
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        vec = np.ones(dim) / np.sqrt(dim)
    else:
        vec = vec / norm
    return vec


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors, safe for zero vectors."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# Main deliberation functions
# ---------------------------------------------------------------------------


def structured_debate(
    topic: str,
    participants: List[Participant],
    rounds: int = 5,
    dim: int = 16,
    rng_seed: int = 42,
) -> DebateResult:
    """Multi-round structured debate among participants.

    Each round, every participant submits an argument represented as a
    position vector plus a quality score.  Arguments are evaluated for
    novelty (distance from all prior arguments) and strength (quality ×
    novelty).  Participant positions evolve toward strong arguments made
    by others, weighted by open-mindedness.  The process builds an
    argument graph tracking support/attack relationships.

    Args:
        topic: The debate topic string.
        participants: List of Participant objects with initial positions.
        rounds: Number of debate rounds.
        dim: Dimensionality for argument embeddings.
        rng_seed: Random seed for reproducibility.

    Returns:
        DebateResult with final positions, argument graph, consensus, etc.
    """
    rng = np.random.RandomState(rng_seed)
    n = len(participants)
    if n == 0:
        return DebateResult(
            final_positions=[],
            argument_graph=ArgumentMap(),
            consensus_measure=1.0,
            rounds_data=[],
            position_trajectories={},
            total_arguments=0,
        )

    arg_map = ArgumentMap()
    all_arg_vecs: List[np.ndarray] = []
    rounds_data: List[Dict[str, Any]] = []
    trajectories: Dict[str, List[np.ndarray]] = {
        p.id: [p.position.vector.copy()] for p in participants
    }

    prev_round_arg_ids: List[str] = []

    for r in range(rounds):
        round_arg_ids: List[str] = []
        round_novelties: List[float] = []
        round_qualities: List[float] = []

        for p in participants:
            # Simulate argument as perturbation of current position
            noise = rng.randn(dim) * 0.3
            arg_vec = p.position.vector[:dim] + noise if len(p.position.vector) >= dim else np.concatenate([p.position.vector, rng.randn(dim - len(p.position.vector))]) + noise
            arg_vec = arg_vec[:dim]
            norm = np.linalg.norm(arg_vec)
            if norm > 1e-12:
                arg_vec = arg_vec / norm

            quality = float(np.clip(0.5 + rng.randn() * 0.2, 0.1, 1.0))
            novelty = _compute_novelty(arg_vec, all_arg_vecs)
            strength = quality * (0.5 + 0.5 * novelty)

            arg_id = f"arg_{p.id}_r{r}_{uuid.uuid4().hex[:6]}"
            stmt = Statement(
                claim=f"{p.name} argument on '{topic}' round {r}",
                grounds=f"Based on position {p.position.label}",
                qualifier=quality,
                id=arg_id,
            )
            argument = Argument(
                id=arg_id,
                statement=stmt,
                strength=strength,
            )

            # Determine support/attack relations to previous round's arguments
            for prev_id in prev_round_arg_ids:
                prev_arg = arg_map.arguments[prev_id]
                if prev_arg.statement.embedding is not None and arg_vec is not None:
                    sim = _cosine_similarity(arg_vec, prev_arg.statement.embedding)
                else:
                    sim = rng.uniform(-0.5, 0.5)

                if sim > 0.3:
                    arg_map.support_edges.append((arg_id, prev_id, float(sim)))
                    argument.supports.append(prev_id)
                elif sim < -0.2:
                    arg_map.attack_edges.append((arg_id, prev_id, float(abs(sim))))
                    argument.attacks.append(prev_id)

            stmt.embedding = arg_vec.copy()
            arg_map.arguments[arg_id] = argument
            all_arg_vecs.append(arg_vec.copy())
            round_arg_ids.append(arg_id)
            round_novelties.append(novelty)
            round_qualities.append(quality)

        # Update participant positions toward strong arguments from others
        for i, p in enumerate(participants):
            total_pull = np.zeros_like(p.position.vector)
            total_weight = 0.0
            for j, other in enumerate(participants):
                if i == j:
                    continue
                other_arg_id = round_arg_ids[j]
                other_strength = arg_map.arguments[other_arg_id].strength
                diff = other.position.vector - p.position.vector
                weight = other_strength * p.open_mindedness
                total_pull += weight * diff
                total_weight += weight

            if total_weight > 1e-12:
                step = total_pull / total_weight
                learning_rate = 0.15 * p.open_mindedness
                p.position.vector = p.position.vector + learning_rate * step

            p.record_position()
            trajectories[p.id].append(p.position.vector.copy())

        # Compute round statistics
        current_positions = np.array([p.position.vector for p in participants])
        polarization = _measure_polarization(current_positions)
        rounds_data.append({
            "round": r,
            "mean_novelty": float(np.mean(round_novelties)),
            "mean_quality": float(np.mean(round_qualities)),
            "polarization": polarization,
            "n_arguments": len(round_arg_ids),
            "n_support_edges": sum(
                1 for s, _, _ in arg_map.support_edges
                if s in round_arg_ids
            ),
            "n_attack_edges": sum(
                1 for s, _, _ in arg_map.attack_edges
                if s in round_arg_ids
            ),
        })
        prev_round_arg_ids = round_arg_ids

    # Compute centrality for all arguments (in-degree based)
    in_degree: Dict[str, int] = {aid: 0 for aid in arg_map.arguments}
    for _, target, _ in arg_map.support_edges:
        if target in in_degree:
            in_degree[target] += 1
    for _, target, _ in arg_map.attack_edges:
        if target in in_degree:
            in_degree[target] += 1
    max_deg = max(in_degree.values()) if in_degree else 1
    for aid, arg in arg_map.arguments.items():
        arg.centrality = in_degree[aid] / max(max_deg, 1)

    # Identify strongest and most central arguments
    sorted_by_strength = sorted(
        arg_map.arguments.values(), key=lambda a: a.strength, reverse=True
    )
    arg_map.strongest_arguments = [a.id for a in sorted_by_strength[:5]]
    sorted_by_centrality = sorted(
        arg_map.arguments.values(), key=lambda a: a.centrality, reverse=True
    )
    arg_map.central_claims = [a.id for a in sorted_by_centrality[:5]]
    arg_map.key_rebuttals = [
        a.id for a in arg_map.arguments.values()
        if len(a.attacks) > 0
    ][:5]

    # Consensus measure: 1 - normalized polarization
    final_positions_arr = np.array([p.position.vector for p in participants])
    final_polarization = _measure_polarization(final_positions_arr)
    initial_positions_arr = np.array([
        trajectories[p.id][0] for p in participants
    ])
    initial_polarization = _measure_polarization(initial_positions_arr)
    denom = max(initial_polarization, 1e-12)
    consensus = float(np.clip(1.0 - final_polarization / denom, 0.0, 1.0))

    return DebateResult(
        final_positions=[p.position.copy() for p in participants],
        argument_graph=arg_map,
        consensus_measure=consensus,
        rounds_data=rounds_data,
        position_trajectories=trajectories,
        total_arguments=len(arg_map.arguments),
    )


def socratic_dialogue(
    question: str,
    participants: List[Participant],
    max_iterations: int = 50,
    challenge_threshold: float = 0.3,
    convergence_threshold: float = 0.01,
) -> DialogueResult:
    """Simulated Socratic dialogue among participants.

    At each iteration the participant whose position is most distant
    from the group centroid is selected as the questioner.  The questioner
    challenges others' positions: each other participant evaluates the
    challenge strength (proportional to the questioner's conviction and
    the distance between positions) and updates their position toward
    the challenge if the strength exceeds the threshold.

    The process terminates when positions converge or max_iterations
    is reached.

    Args:
        question: The guiding question for the dialogue.
        participants: Participants with initial positions.
        max_iterations: Maximum dialogue iterations.
        challenge_threshold: Minimum challenge strength to cause an update.
        convergence_threshold: Position-change threshold for convergence.

    Returns:
        DialogueResult with position evolution, questioner sequence, etc.
    """
    n = len(participants)
    if n == 0:
        return DialogueResult(
            final_positions=[],
            position_evolution={},
            questioner_sequence=[],
            convergence_iteration=-1,
            final_spread=0.0,
            challenges_made=0,
        )

    evolution: Dict[str, List[np.ndarray]] = {
        p.id: [p.position.vector.copy()] for p in participants
    }
    questioner_seq: List[str] = []
    challenges_made = 0
    convergence_iter = -1
    positions_history: List[np.ndarray] = []

    for iteration in range(max_iterations):
        positions = np.array([p.position.vector for p in participants])
        positions_history.append(positions.copy())

        # Check convergence
        if _convergence_check(positions_history, convergence_threshold):
            convergence_iter = iteration
            break

        centroid = positions.mean(axis=0)

        # Select questioner: participant farthest from centroid
        distances_to_centroid = np.linalg.norm(positions - centroid, axis=1)
        questioner_idx = int(np.argmax(distances_to_centroid))
        questioner = participants[questioner_idx]
        questioner_seq.append(questioner.id)

        # Questioner challenges each other participant
        for i, p in enumerate(participants):
            if i == questioner_idx:
                continue

            diff = questioner.position.vector - p.position.vector
            distance = float(np.linalg.norm(diff))
            challenge_strength = distance * questioner.position.conviction

            if challenge_strength > challenge_threshold:
                challenges_made += 1
                # Update toward questioner weighted by open-mindedness and
                # inversely by own conviction
                update_rate = (
                    p.open_mindedness
                    * min(challenge_strength, 1.0)
                    / (1.0 + p.position.conviction)
                )
                p.position.vector = (
                    p.position.vector + update_rate * diff
                )
                # Slightly reduce conviction when challenged successfully
                p.position.conviction = max(
                    0.1, p.position.conviction * 0.98
                )

            p.record_position()
            evolution[p.id].append(p.position.vector.copy())

        # Questioner also softens slightly toward centroid
        q_diff = centroid - questioner.position.vector
        questioner.position.vector = (
            questioner.position.vector
            + 0.05 * questioner.open_mindedness * q_diff
        )
        questioner.record_position()
        evolution[questioner.id].append(questioner.position.vector.copy())

    # Final spread
    final_positions = np.array([p.position.vector for p in participants])
    final_spread = float(np.std(final_positions))

    return DialogueResult(
        final_positions=[p.position.copy() for p in participants],
        position_evolution=evolution,
        questioner_sequence=questioner_seq,
        convergence_iteration=convergence_iter,
        final_spread=final_spread,
        challenges_made=challenges_made,
    )


def argument_mapping(
    statements: List[Statement],
    similarity_threshold: float = 0.3,
    attack_threshold: float = -0.2,
    embedding_dim: int = 32,
) -> ArgumentMap:
    """Build a Toulmin argument map from a list of statements.

    Each statement is parsed into its Toulmin components (claim, grounds,
    warrant, backing, qualifier, rebuttal).  Embeddings are computed for
    each claim, and pairwise cosine similarity determines support/attack
    relationships.  The resulting directed graph identifies strongest
    arguments (by completeness × qualifier), central claims (by degree
    centrality), and key rebuttals.

    Args:
        statements: List of Statement objects to map.
        similarity_threshold: Cosine similarity above which a support
            edge is created.
        attack_threshold: Cosine similarity below which an attack edge
            is created.
        embedding_dim: Dimensionality for generated embeddings.

    Returns:
        ArgumentMap with arguments, edges, and analysis.
    """
    arg_map = ArgumentMap()

    if not statements:
        return arg_map

    # Parse and embed all statements
    parsed: List[Dict[str, Any]] = []
    for stmt in statements:
        p = _toulmin_parse(stmt)
        if stmt.embedding is None:
            stmt.embedding = _generate_embedding(stmt.claim, dim=embedding_dim)
        parsed.append(p)

        arg = Argument(
            id=stmt.id,
            statement=stmt,
            strength=p["completeness"],
        )
        arg_map.arguments[stmt.id] = arg

    # Compute pairwise relationships
    for i, stmt_i in enumerate(statements):
        for j, stmt_j in enumerate(statements):
            if i >= j:
                continue
            sim = _cosine_similarity(stmt_i.embedding, stmt_j.embedding)

            if sim > similarity_threshold:
                arg_map.support_edges.append((stmt_i.id, stmt_j.id, float(sim)))
                arg_map.arguments[stmt_i.id].supports.append(stmt_j.id)
                arg_map.arguments[stmt_j.id].supports.append(stmt_i.id)
            elif sim < attack_threshold:
                arg_map.attack_edges.append((stmt_i.id, stmt_j.id, float(abs(sim))))
                arg_map.arguments[stmt_i.id].attacks.append(stmt_j.id)
                arg_map.arguments[stmt_j.id].attacks.append(stmt_i.id)

            # Rebuttal cross-linking: if stmt_i's rebuttal text is similar
            # to stmt_j's claim, create an attack edge
            if stmt_i.rebuttal:
                reb_emb = _generate_embedding(stmt_i.rebuttal, dim=embedding_dim)
                reb_sim = _cosine_similarity(reb_emb, stmt_j.embedding)
                if reb_sim > similarity_threshold:
                    arg_map.attack_edges.append(
                        (stmt_i.id, stmt_j.id, float(reb_sim))
                    )
                    arg_map.arguments[stmt_i.id].attacks.append(stmt_j.id)

    # Compute centrality (total degree)
    degree: Dict[str, int] = {aid: 0 for aid in arg_map.arguments}
    for src, tgt, _ in arg_map.support_edges:
        degree[src] = degree.get(src, 0) + 1
        degree[tgt] = degree.get(tgt, 0) + 1
    for src, tgt, _ in arg_map.attack_edges:
        degree[src] = degree.get(src, 0) + 1
        degree[tgt] = degree.get(tgt, 0) + 1
    max_degree = max(degree.values()) if degree else 1
    for aid in arg_map.arguments:
        arg_map.arguments[aid].centrality = degree[aid] / max(max_degree, 1)

    # Identify key elements
    sorted_strength = sorted(
        arg_map.arguments.values(), key=lambda a: a.strength, reverse=True
    )
    arg_map.strongest_arguments = [a.id for a in sorted_strength[:5]]

    sorted_centrality = sorted(
        arg_map.arguments.values(), key=lambda a: a.centrality, reverse=True
    )
    arg_map.central_claims = [a.id for a in sorted_centrality[:5]]

    arg_map.key_rebuttals = [
        a.id for a in arg_map.arguments.values()
        if len(a.attacks) > 0
    ][:5]

    return arg_map


def consensus_building(
    positions: List[Position],
    mediator_fn: Optional[Callable[[List[Position], np.ndarray], np.ndarray]] = None,
    max_rounds: int = 100,
    convergence_threshold: float = 0.01,
    holdout_threshold: float = 0.1,
) -> ConsensusResult:
    """Find consensus among diverse positions via iterative compromise.

    Each round, positions move toward the conviction-weighted centroid.
    The step size is proportional to each position's open-mindedness
    analog (inverse conviction).  If a mediator function is provided,
    it proposes a compromise position that attracts all positions.

    Args:
        positions: List of Position objects to reconcile.
        mediator_fn: Optional function (positions, centroid) -> proposed
            compromise vector.
        max_rounds: Maximum number of consensus rounds.
        convergence_threshold: Threshold for convergence detection.
        holdout_threshold: Minimum movement to not be considered a holdout.

    Returns:
        ConsensusResult with agreement level, holdouts, and history.
    """
    n = len(positions)
    if n == 0:
        return ConsensusResult(
            final_positions=[],
            agreement_level=1.0,
            convergence_round=-1,
            holdout_positions=[],
            position_history=[],
            centroid_history=[],
        )

    working = [p.copy() for p in positions]
    initial_vecs = np.array([p.vector for p in working])
    position_history: List[List[np.ndarray]] = [
        [p.vector.copy() for p in working]
    ]
    centroid_history: List[np.ndarray] = []
    convergence_round = -1

    for r in range(max_rounds):
        vecs = np.array([p.vector for p in working])
        convictions = np.array([p.conviction for p in working])

        # Conviction-weighted centroid
        weights = convictions / max(convictions.sum(), 1e-12)
        centroid = (weights[:, None] * vecs).sum(axis=0)
        centroid_history.append(centroid.copy())

        # If mediator is available, blend centroid with mediator proposal
        if mediator_fn is not None:
            proposal = mediator_fn(working, centroid)
            centroid = 0.6 * centroid + 0.4 * proposal

        # Move each position toward centroid
        new_vecs = []
        for i, p in enumerate(working):
            diff = centroid - p.vector
            # Lower-conviction participants move more
            step_rate = 0.2 / (1.0 + p.conviction)
            p.vector = p.vector + step_rate * diff
            # Conviction decays slightly each round toward 0.5
            p.conviction = p.conviction * 0.99 + 0.5 * 0.01
            new_vecs.append(p.vector.copy())

        position_history.append([p.vector.copy() for p in working])

        # Convergence check
        curr = np.array([p.vector for p in working])
        history_arrays = [np.array(h) for h in position_history[-2:]]
        if _convergence_check(history_arrays, convergence_threshold):
            convergence_round = r
            break

    # Agreement level
    final_vecs = np.array([p.vector for p in working])
    final_polarization = _measure_polarization(final_vecs)
    initial_polarization = _measure_polarization(initial_vecs)
    denom = max(initial_polarization, 1e-12)
    agreement_level = float(np.clip(1.0 - final_polarization / denom, 0.0, 1.0))

    # Identify holdouts: positions that moved less than threshold
    holdouts: List[Position] = []
    for i, p in enumerate(working):
        movement = float(np.linalg.norm(p.vector - initial_vecs[i]))
        if movement < holdout_threshold:
            holdouts.append(p.copy())

    return ConsensusResult(
        final_positions=[p.copy() for p in working],
        agreement_level=agreement_level,
        convergence_round=convergence_round,
        holdout_positions=holdouts,
        position_history=position_history,
        centroid_history=centroid_history,
    )


def deliberative_polling(
    topic: str,
    participants: List[Participant],
    n_groups: int = 4,
    information_shift: float = 0.15,
    deliberation_rounds: int = 5,
    rng_seed: int = 42,
) -> PollResult:
    """Simulated deliberative polling following the Fishkin method.

    Three phases:
    1. **Pre-poll**: Record initial positions.
    2. **Information phase**: Present balanced information, shifting
       positions toward the overall center (simulating informed
       moderation).
    3. **Deliberation phase**: Small-group discussions where participants
       influence each other based on proximity, expertise, and
       open-mindedness.
    4. **Post-poll**: Record final positions and compute statistics.

    Args:
        topic: Topic being polled.
        participants: Participants with initial positions.
        n_groups: Number of small discussion groups.
        information_shift: How much the information phase shifts positions
            toward center (0-1).
        deliberation_rounds: Rounds of small-group deliberation.
        rng_seed: Random seed.

    Returns:
        PollResult with opinion changes, polarization changes, etc.
    """
    rng = np.random.RandomState(rng_seed)
    n = len(participants)

    if n == 0:
        return PollResult(
            pre_poll_positions=[],
            post_poll_positions=[],
            opinion_change=[],
            polarization_before=0.0,
            polarization_after=0.0,
            information_gain=0.0,
            group_assignments={},
            phase_positions={},
        )

    working = [
        Participant(
            id=p.id,
            name=p.name,
            position=p.position.copy(),
            expertise_weights=p.expertise_weights.copy() if p.expertise_weights is not None else None,
            open_mindedness=p.open_mindedness,
        )
        for p in participants
    ]

    # --- Phase 0: Pre-poll ---
    pre_poll = np.array([p.position.vector.copy() for p in working])
    polarization_before = _measure_polarization(pre_poll)

    phase_positions: Dict[str, List[np.ndarray]] = {
        "pre_poll": [v.copy() for v in pre_poll],
    }

    # --- Phase 1: Information ---
    centroid = pre_poll.mean(axis=0)
    for p in working:
        diff = centroid - p.position.vector
        # Participants with lower expertise shift more
        expertise_factor = 1.0
        if p.expertise_weights is not None:
            expertise_factor = float(1.0 - np.mean(p.expertise_weights))
        shift = information_shift * expertise_factor
        p.position.vector = p.position.vector + shift * diff

    post_info = np.array([p.position.vector.copy() for p in working])
    phase_positions["post_information"] = [v.copy() for v in post_info]

    # --- Phase 2: Deliberation (small-group discussions) ---
    group_assignments = _assign_diverse_groups(working, n_groups)

    # Build groups
    groups: Dict[int, List[int]] = {}
    for i, p in enumerate(working):
        g = group_assignments[p.id]
        groups.setdefault(g, []).append(i)

    for d_round in range(deliberation_rounds):
        for g_id, member_indices in groups.items():
            if len(member_indices) < 2:
                continue

            group_positions = np.array([
                working[i].position.vector for i in member_indices
            ])
            group_centroid = group_positions.mean(axis=0)

            for idx in member_indices:
                p = working[idx]
                diff = group_centroid - p.position.vector
                # Influence depends on open-mindedness and group dynamics
                rate = 0.1 * p.open_mindedness / (1.0 + p.position.conviction * 0.5)
                noise = rng.randn(*p.position.vector.shape) * 0.02
                p.position.vector = p.position.vector + rate * diff + noise

    post_delib = np.array([p.position.vector.copy() for p in working])
    phase_positions["post_deliberation"] = [v.copy() for v in post_delib]

    # --- Phase 3: Post-poll statistics ---
    polarization_after = _measure_polarization(post_delib)

    opinion_change = [
        float(np.linalg.norm(post_delib[i] - pre_poll[i]))
        for i in range(n)
    ]

    # Information gain: reduction in variance (entropy proxy)
    info_gain = max(0.0, polarization_before - polarization_after) / max(
        polarization_before, 1e-12
    )

    return PollResult(
        pre_poll_positions=[v.copy() for v in pre_poll],
        post_poll_positions=[v.copy() for v in post_delib],
        opinion_change=opinion_change,
        polarization_before=polarization_before,
        polarization_after=polarization_after,
        information_gain=float(info_gain),
        group_assignments=group_assignments,
        phase_positions=phase_positions,
    )


def citizens_assembly(
    issues: List[str],
    participants: List[Participant],
    assembly_size: int = 0,
    n_groups: int = 4,
    expert_rounds: int = 2,
    deliberation_rounds: int = 5,
    voting_threshold: float = 0.6,
    dim: int = 8,
    rng_seed: int = 42,
) -> AssemblyResult:
    """Simulated citizens' assembly with sortition and structured deliberation.

    Process for each issue:
    1. **Sortition**: Randomly select a representative subset of participants.
    2. **Expert testimony**: Shift positions toward expert-informed centers,
       weighted by each participant's expertise openness.
    3. **Small-group deliberation**: Diversity-maximizing group assignment
       followed by iterative position updating within groups.
    4. **Plenary voting**: Threshold-based consensus on the recommendation.

    Args:
        issues: List of issue descriptions.
        participants: Full pool of potential participants.
        assembly_size: Number of participants to select (0 = all).
        n_groups: Number of discussion groups per issue.
        expert_rounds: Rounds of expert testimony.
        deliberation_rounds: Rounds of small-group deliberation.
        voting_threshold: Fraction required for consensus.
        dim: Position vector dimensionality.
        rng_seed: Random seed.

    Returns:
        AssemblyResult with recommendations, confidence, minority reports.
    """
    rng = np.random.RandomState(rng_seed)

    if not participants:
        return AssemblyResult(
            recommendations={},
            confidence_levels={},
            minority_reports={},
            selected_participants=[],
            voting_results={},
            group_assignments={},
            position_trajectories={},
        )

    # --- Sortition ---
    if assembly_size <= 0 or assembly_size >= len(participants):
        selected_indices = list(range(len(participants)))
    else:
        selected_indices = sorted(
            rng.choice(len(participants), size=assembly_size, replace=False).tolist()
        )

    selected_ids = [participants[i].id for i in selected_indices]

    recommendations: Dict[str, np.ndarray] = {}
    confidence_levels: Dict[str, float] = {}
    minority_reports: Dict[str, List[Dict[str, Any]]] = {}
    voting_results: Dict[str, Dict[str, Any]] = {}
    all_group_assignments: Dict[str, Dict[str, int]] = {}
    all_trajectories: Dict[str, Dict[str, List[np.ndarray]]] = {}

    for issue_idx, issue in enumerate(issues):
        # Create working copies for this issue
        assembly = []
        for idx in selected_indices:
            orig = participants[idx]
            # Generate issue-specific initial position if dimension doesn't match
            if len(orig.position.vector) == dim:
                vec = orig.position.vector.copy()
            else:
                issue_seed = rng_seed + issue_idx * 1000 + idx
                issue_rng = np.random.RandomState(issue_seed)
                base = orig.position.vector
                if len(base) > dim:
                    vec = base[:dim].copy()
                else:
                    vec = np.concatenate([
                        base, issue_rng.randn(dim - len(base)) * 0.3
                    ])

            member = Participant(
                id=orig.id,
                name=orig.name,
                position=Position(vector=vec, conviction=orig.position.conviction),
                expertise_weights=orig.expertise_weights,
                open_mindedness=orig.open_mindedness,
            )
            assembly.append(member)

        trajectories: Dict[str, List[np.ndarray]] = {
            m.id: [m.position.vector.copy()] for m in assembly
        }

        # --- Expert testimony ---
        for e_round in range(expert_rounds):
            # Simulate expert position as a perturbation of the centroid
            positions_arr = np.array([m.position.vector for m in assembly])
            expert_center = positions_arr.mean(axis=0)
            expert_noise = rng.randn(dim) * 0.1
            expert_position = expert_center + expert_noise

            for m in assembly:
                diff = expert_position - m.position.vector
                # Expertise weight affects receptiveness
                if m.expertise_weights is not None and issue_idx < len(m.expertise_weights):
                    openness = float(1.0 - m.expertise_weights[issue_idx]) * 0.3
                else:
                    openness = 0.15
                m.position.vector = m.position.vector + openness * diff
                trajectories[m.id].append(m.position.vector.copy())

        # --- Small-group deliberation ---
        group_assign = _assign_diverse_groups(assembly, n_groups)
        all_group_assignments[issue] = group_assign

        groups: Dict[int, List[int]] = {}
        for i, m in enumerate(assembly):
            g = group_assign[m.id]
            groups.setdefault(g, []).append(i)

        for d_round in range(deliberation_rounds):
            for g_id, member_indices in groups.items():
                if len(member_indices) < 2:
                    continue

                group_pos = np.array([
                    assembly[i].position.vector for i in member_indices
                ])
                group_centroid = group_pos.mean(axis=0)

                # Build within-group influence matrix
                n_g = len(member_indices)
                influence = np.zeros((n_g, n_g))
                for a_local in range(n_g):
                    for b_local in range(n_g):
                        if a_local == b_local:
                            continue
                        dist = float(np.linalg.norm(
                            group_pos[a_local] - group_pos[b_local]
                        ))
                        # Closer positions have more influence (bounded)
                        influence[a_local, b_local] = (
                            assembly[member_indices[b_local]].open_mindedness
                            / (1.0 + dist)
                        )
                    # Self-retention
                    influence[a_local, a_local] = 1.0 + assembly[
                        member_indices[a_local]
                    ].position.conviction

                # Normalize and apply
                updated = _update_positions(group_pos, influence)

                # Apply with damping
                damping = 0.3
                for local_idx, global_idx in enumerate(member_indices):
                    m = assembly[global_idx]
                    m.position.vector = (
                        (1.0 - damping) * m.position.vector
                        + damping * updated[local_idx]
                    )
                    trajectories[m.id].append(m.position.vector.copy())

        all_trajectories[issue] = trajectories

        # --- Plenary voting ---
        final_positions = np.array([m.position.vector for m in assembly])
        recommendation = final_positions.mean(axis=0)
        recommendations[issue] = recommendation

        # Confidence: inverse of spread
        spread = _measure_polarization(final_positions)
        confidence = float(1.0 / (1.0 + spread))
        confidence_levels[issue] = confidence

        # Voting: count those whose final position is close to recommendation
        distances_to_rec = np.linalg.norm(
            final_positions - recommendation, axis=1
        )
        median_dist = float(np.median(distances_to_rec))
        approvals = int(np.sum(distances_to_rec <= median_dist * 1.5))
        approval_rate = approvals / max(len(assembly), 1)
        consensus_reached = approval_rate >= voting_threshold

        voting_results[issue] = {
            "approvals": approvals,
            "total": len(assembly),
            "approval_rate": float(approval_rate),
            "threshold": voting_threshold,
            "consensus_reached": consensus_reached,
        }

        # Minority reports: participants far from recommendation
        minority: List[Dict[str, Any]] = []
        for i, m in enumerate(assembly):
            if distances_to_rec[i] > median_dist * 2.0:
                minority.append({
                    "participant_id": m.id,
                    "participant_name": m.name,
                    "distance_from_recommendation": float(distances_to_rec[i]),
                    "final_position": m.position.vector.tolist(),
                    "conviction": float(m.position.conviction),
                })
        minority_reports[issue] = minority

    return AssemblyResult(
        recommendations=recommendations,
        confidence_levels=confidence_levels,
        minority_reports=minority_reports,
        selected_participants=selected_ids,
        voting_results=voting_results,
        group_assignments=all_group_assignments,
        position_trajectories=all_trajectories,
    )
