"""Collective intelligence algorithms for diverse problem solving.

Implements collective problem solving, idea marketplaces, innovation
tournaments, crowd labeling with diversity requirements, and swarm
intelligence with diversity maintenance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Problem:
    """Optimisation problem defined by an objective and search bounds.

    Attributes:
        objective: Callable mapping a position vector to a scalar value
            (lower is better).
        bounds: Sequence of (lo, hi) pairs, one per dimension.
        description: Human-readable description of the problem.
    """

    objective: Callable[[np.ndarray], float]
    bounds: Sequence[Tuple[float, float]]
    description: str = ""

    @property
    def dim(self) -> int:
        return len(self.bounds)


@dataclass
class SwarmAgent:
    """Agent participating in swarm or collective optimisation.

    Attributes:
        position: Current position vector in search space.
        velocity: Current velocity vector.
        perspective: A bias vector representing the agent's unique viewpoint.
        agent_type: Optional categorical label (e.g. for diversity tracking).
    """

    position: np.ndarray
    velocity: np.ndarray
    perspective: np.ndarray
    agent_type: str = "default"


@dataclass
class Solution:
    """Result of collective problem solving.

    Attributes:
        best_position: The best solution vector found.
        best_value: Objective value at best_position.
        diversity: Mean pairwise distance among final candidate solutions.
        convergence_history: Objective value of best solution at each iteration.
        all_solutions: All candidate solutions retained at termination.
    """

    best_position: np.ndarray
    best_value: float
    diversity: float
    convergence_history: List[float] = field(default_factory=list)
    all_solutions: List[np.ndarray] = field(default_factory=list)


@dataclass
class Idea:
    """An idea in the marketplace.

    Attributes:
        idea_id: Unique identifier.
        embedding: Dense vector representation of the idea.
        initial_quality: Creator's quality estimate in [0, 1].
        description: Human-readable summary.
    """

    idea_id: int
    embedding: np.ndarray
    initial_quality: float
    description: str = ""


@dataclass
class MarketplaceResult:
    """Result of an idea marketplace simulation.

    Attributes:
        rankings: Idea ids sorted best-to-worst by final market price.
        final_prices: Mapping from idea id to converged market price.
        undervalued: Ids of ideas with high diversity but low price.
        market_efficiency: Correlation between final prices and true quality.
        price_history: Dict mapping idea id to list of prices over rounds.
    """

    rankings: List[int]
    final_prices: dict[int, float]
    undervalued: List[int]
    market_efficiency: float
    price_history: dict[int, List[float]]


@dataclass
class Challenge:
    """A challenge posed in an innovation tournament.

    Attributes:
        challenge_id: Unique identifier.
        objective: Callable scoring a solution vector (higher is better).
        bounds: Search bounds per dimension.
        description: Human-readable description.
    """

    challenge_id: int
    objective: Callable[[np.ndarray], float]
    bounds: Sequence[Tuple[float, float]]
    description: str = ""


@dataclass
class TournamentResult:
    """Result of a multi-stage innovation tournament.

    Attributes:
        winners: List of (participant_index, best_score) for final winners.
        stage_scores: Per-stage list of (participant_index, score) lists.
        innovation_quality: Mean score improvement across stages.
        diversity_of_winners: Mean pairwise distance among winning solutions.
        winning_solutions: Actual solution vectors of winners.
    """

    winners: List[Tuple[int, float]]
    stage_scores: List[List[Tuple[int, float]]]
    innovation_quality: float
    diversity_of_winners: float
    winning_solutions: List[np.ndarray]


@dataclass
class LabelItem:
    """An item to be labeled by the crowd.

    Attributes:
        item_id: Unique identifier.
        true_label: Ground-truth label (may be unknown / -1).
        features: Optional feature vector.
    """

    item_id: int
    true_label: int = -1
    features: Optional[np.ndarray] = None


@dataclass
class Labels:
    """Result of diverse crowd labeling.

    Attributes:
        estimated_labels: Mapping from item id to estimated true label.
        labeler_quality: Mapping from labeler index to quality score in [0,1].
        inter_annotator_agreement: Fleiss' kappa or similar agreement metric.
        confident_items: Item ids where label confidence exceeds a threshold.
        label_probabilities: Mapping from item id to array of class probs.
    """

    estimated_labels: dict[int, int]
    labeler_quality: dict[int, float]
    inter_annotator_agreement: float
    confident_items: List[int]
    label_probabilities: dict[int, np.ndarray]


@dataclass
class SwarmResult:
    """Result of swarm intelligence optimisation.

    Attributes:
        best_position: Global best position found.
        best_value: Objective value at best_position.
        convergence_history: Best value at each iteration.
        diversity_history: Swarm diversity at each iteration.
        final_positions: Agent positions at termination.
    """

    best_position: np.ndarray
    best_value: float
    convergence_history: List[float] = field(default_factory=list)
    diversity_history: List[float] = field(default_factory=list)
    final_positions: List[np.ndarray] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _diverse_selection(solutions: List[np.ndarray], k: int) -> List[np.ndarray]:
    """Select *k* most diverse solutions via greedy farthest-point sampling.

    Starting from the solution with the highest norm (arbitrary but
    deterministic seed), iteratively pick the candidate farthest from the
    already-selected set.

    Args:
        solutions: Pool of candidate solution vectors.
        k: Number of solutions to select.

    Returns:
        List of *k* diverse solution vectors.
    """
    if k >= len(solutions):
        return list(solutions)
    if k <= 0:
        return []

    arr = np.array(solutions)
    n = len(arr)

    norms = np.linalg.norm(arr, axis=1)
    selected_indices: List[int] = [int(np.argmax(norms))]
    min_dists = np.full(n, np.inf)

    for _ in range(k - 1):
        last = arr[selected_indices[-1]]
        dists = np.linalg.norm(arr - last, axis=1)
        min_dists = np.minimum(min_dists, dists)
        # Exclude already selected
        min_dists_copy = min_dists.copy()
        for idx in selected_indices:
            min_dists_copy[idx] = -1.0
        selected_indices.append(int(np.argmax(min_dists_copy)))

    return [arr[i].copy() for i in selected_indices]


def _dawid_skene_em(
    annotations: dict[int, dict[int, int]],
    n_classes: int,
    n_iterations: int = 20,
) -> Tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Dawid-Skene EM algorithm for crowd label aggregation.

    Estimates the true label distribution for each item and per-labeler
    confusion matrices via expectation-maximisation.

    Args:
        annotations: ``{item_id: {labeler_id: label}}``.
        n_classes: Number of distinct label classes.
        n_iterations: EM iterations.

    Returns:
        Tuple of (class_probabilities, error_rates) where
        class_probabilities maps item_id -> array of class probabilities,
        error_rates maps labeler_id -> (n_classes x n_classes) confusion matrix
        (row = true class, col = observed class), normalised per row.
    """
    items = sorted(annotations.keys())
    labelers: set[int] = set()
    for ann in annotations.values():
        labelers.update(ann.keys())
    labeler_list = sorted(labelers)

    # Initialise class probs via majority vote
    class_probs: dict[int, np.ndarray] = {}
    for item in items:
        counts = np.zeros(n_classes, dtype=np.float64)
        for lab_label in annotations[item].values():
            if 0 <= lab_label < n_classes:
                counts[lab_label] += 1
        total = counts.sum()
        if total > 0:
            class_probs[item] = counts / total
        else:
            class_probs[item] = np.ones(n_classes) / n_classes

    # Initialise error rate matrices (identity-ish)
    error_rates: dict[int, np.ndarray] = {}
    for j in labeler_list:
        mat = np.eye(n_classes) * 0.7 + 0.3 / n_classes
        # Normalise rows
        mat = mat / mat.sum(axis=1, keepdims=True)
        error_rates[j] = mat

    # Class prior
    class_prior = np.ones(n_classes) / n_classes

    for _ in range(n_iterations):
        # --- E step: update class_probs for each item ---
        for item in items:
            log_posterior = np.log(class_prior + 1e-300)
            for j, label in annotations[item].items():
                if 0 <= label < n_classes:
                    for c in range(n_classes):
                        log_posterior[c] += math.log(error_rates[j][c, label] + 1e-300)
            # Normalise in log space
            log_posterior -= log_posterior.max()
            posterior = np.exp(log_posterior)
            posterior /= posterior.sum() + 1e-300
            class_probs[item] = posterior

        # --- M step: update error_rates and class_prior ---
        # Update class prior
        prior_counts = np.zeros(n_classes)
        for item in items:
            prior_counts += class_probs[item]
        class_prior = prior_counts / (prior_counts.sum() + 1e-300)

        # Update error rates
        for j in labeler_list:
            mat = np.zeros((n_classes, n_classes))
            for item in items:
                if j in annotations[item]:
                    label = annotations[item][j]
                    if 0 <= label < n_classes:
                        mat[:, label] += class_probs[item]
            # Normalise rows with Laplace smoothing
            mat += 1e-6
            mat = mat / mat.sum(axis=1, keepdims=True)
            error_rates[j] = mat

    return class_probs, error_rates


def _market_price_update(
    current_price: float,
    new_evaluation: float,
    decay: float = 0.1,
) -> float:
    """Exponential moving average price update.

    Args:
        current_price: Current market price of an idea.
        new_evaluation: New evaluation score from a trader.
        decay: Smoothing factor (higher = more weight on new eval).

    Returns:
        Updated market price.
    """
    return (1.0 - decay) * current_price + decay * new_evaluation


def _pso_update(
    positions: np.ndarray,
    velocities: np.ndarray,
    pbest: np.ndarray,
    gbest: np.ndarray,
    w: float,
    c1: float,
    c2: float,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Standard PSO velocity and position update.

    Args:
        positions: (n_agents, dim) current positions.
        velocities: (n_agents, dim) current velocities.
        pbest: (n_agents, dim) personal best positions.
        gbest: (dim,) global best position.
        w: Inertia weight.
        c1: Cognitive coefficient.
        c2: Social coefficient.
        rng: Numpy random generator (for reproducibility).

    Returns:
        Tuple of (new_positions, new_velocities).
    """
    if rng is None:
        rng = np.random.default_rng()

    n, dim = positions.shape
    r1 = rng.random((n, dim))
    r2 = rng.random((n, dim))

    new_velocities = (
        w * velocities
        + c1 * r1 * (pbest - positions)
        + c2 * r2 * (gbest - positions)
    )
    new_positions = positions + new_velocities
    return new_positions, new_velocities


def _swarm_diversity(positions: np.ndarray) -> float:
    """Mean pairwise Euclidean distance among positions.

    Args:
        positions: (n, dim) array of agent positions.

    Returns:
        Scalar mean pairwise distance. Returns 0 for fewer than 2 agents.
    """
    n = positions.shape[0]
    if n < 2:
        return 0.0
    # Efficient pairwise distance via broadcasting
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    # Upper triangle only (avoid double-counting and diagonal)
    return float(dists[np.triu_indices(n, k=1)].mean())


def _tournament_bracket(
    n_participants: int,
    n_stages: int,
) -> List[List[int]]:
    """Organise tournament stages determining how many participants advance.

    At each stage roughly the top half advances (minimum 1).

    Args:
        n_participants: Total number of participants.
        n_stages: Number of tournament stages.

    Returns:
        List of lists; each inner list gives indices of participants
        advancing into that stage. Stage 0 contains all participants.
    """
    stages: List[List[int]] = [list(range(n_participants))]
    current = n_participants
    for _ in range(1, n_stages):
        current = max(1, current // 2)
        # Placeholder: actual indices filled during tournament execution
        stages.append(list(range(current)))
    return stages


def _decompose_problem(
    problem: Problem,
    n_subproblems: int,
) -> List[Problem]:
    """Decompose a problem into sub-problems by partitioning dimensions.

    Each sub-problem optimises a disjoint subset of dimensions while fixing
    the others at the midpoint of their bounds.

    Args:
        problem: The original problem.
        n_subproblems: Desired number of sub-problems.

    Returns:
        List of sub-problems. Each has the full objective but restricted
        bounds (other dimensions are fixed at midpoint).
    """
    dim = problem.dim
    n_subproblems = min(n_subproblems, dim)
    n_subproblems = max(n_subproblems, 1)

    indices = np.array_split(np.arange(dim), n_subproblems)
    midpoint = np.array([(lo + hi) / 2.0 for lo, hi in problem.bounds])

    sub_problems: List[Problem] = []
    for idx_group in indices:
        idx_set = set(int(i) for i in idx_group)

        def _make_objective(
            active: set[int],
            mid: np.ndarray,
            obj: Callable[[np.ndarray], float],
        ) -> Callable[[np.ndarray], float]:
            def sub_objective(x: np.ndarray) -> float:
                full = mid.copy()
                j = 0
                for i in sorted(active):
                    full[i] = x[j]
                    j += 1
                return obj(full)
            return sub_objective

        sub_bounds = [problem.bounds[i] for i in sorted(idx_set)]
        sub_problems.append(
            Problem(
                objective=_make_objective(idx_set, midpoint, problem.objective),
                bounds=sub_bounds,
                description=f"Sub-problem dims {sorted(idx_set)}",
            )
        )
    return sub_problems


def _clip_to_bounds(
    positions: np.ndarray,
    bounds: Sequence[Tuple[float, float]],
) -> np.ndarray:
    """Clip positions to stay within search bounds.

    Args:
        positions: (n, dim) or (dim,) array.
        bounds: Per-dimension (lo, hi) bounds.

    Returns:
        Clipped array of same shape.
    """
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    return np.clip(positions, lo, hi)


# ---------------------------------------------------------------------------
# Main API functions
# ---------------------------------------------------------------------------

def collective_problem_solving(
    problem: Problem,
    agents: List[SwarmAgent],
    method: str = "diverse",
    n_iterations: int = 50,
    rng: np.random.Generator | None = None,
) -> Solution:
    """Solve an optimisation problem collectively using diverse agents.

    Three methods are supported:

    * ``"diverse"``: Each agent independently proposes solutions (local
      search biased by its perspective), then the most diverse top
      solutions are selected.
    * ``"iterative"``: Agents iteratively refine each other's best
      solution, mixing in their own perspective.
    * ``"decompose"``: The problem is decomposed into sub-problems
      assigned to agents; sub-solutions are recombined.

    Args:
        problem: The optimisation problem (minimisation).
        agents: List of agents with position/perspective vectors.
        method: One of ``"diverse"``, ``"iterative"``, ``"decompose"``.
        n_iterations: Number of refinement iterations.
        rng: Numpy random generator for reproducibility.

    Returns:
        A :class:`Solution` containing the best result and diagnostics.
    """
    if rng is None:
        rng = np.random.default_rng()

    dim = problem.dim
    bounds = problem.bounds
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    convergence: List[float] = []

    if method == "diverse":
        all_solutions: List[np.ndarray] = []
        all_values: List[float] = []

        for agent in agents:
            # Initialise near agent perspective, clipped to bounds
            perspective = agent.perspective[:dim] if len(agent.perspective) >= dim else np.concatenate(
                [agent.perspective, rng.uniform(0, 1, dim - len(agent.perspective))]
            )
            x = _clip_to_bounds(lo + perspective * (hi - lo), bounds)
            best_x = x.copy()
            best_val = problem.objective(x)

            for _ in range(n_iterations):
                # Local search: random perturbation biased by perspective
                step = rng.normal(0, 0.1, dim) * (hi - lo)
                candidate = _clip_to_bounds(x + step, bounds)
                val = problem.objective(candidate)
                if val < best_val:
                    best_val = val
                    best_x = candidate.copy()
                x = candidate if val < problem.objective(x) else x

            all_solutions.append(best_x)
            all_values.append(best_val)

        # Track convergence as running best
        running_best = float("inf")
        for v in all_values:
            running_best = min(running_best, v)
            convergence.append(running_best)

        # Select diverse subset
        k = max(1, len(agents) // 2)
        diverse_sols = _diverse_selection(all_solutions, k)

        best_idx = int(np.argmin(all_values))
        diversity = _swarm_diversity(np.array(all_solutions)) if len(all_solutions) > 1 else 0.0

        return Solution(
            best_position=all_solutions[best_idx],
            best_value=all_values[best_idx],
            diversity=diversity,
            convergence_history=convergence,
            all_solutions=diverse_sols,
        )

    elif method == "iterative":
        # Start from first agent's perspective
        current = _clip_to_bounds(
            lo + agents[0].perspective[:dim] * (hi - lo) if len(agents[0].perspective) >= dim
            else lo + np.resize(agents[0].perspective, dim) * (hi - lo),
            bounds,
        )
        best_x = current.copy()
        best_val = problem.objective(current)
        convergence.append(best_val)
        all_solutions_iter: List[np.ndarray] = [best_x.copy()]

        for it in range(n_iterations):
            agent = agents[it % len(agents)]
            persp = np.resize(agent.perspective, dim)
            # Mix current solution with agent perspective
            alpha = 0.3
            mixed = (1 - alpha) * current + alpha * _clip_to_bounds(
                lo + persp * (hi - lo), bounds
            )
            mixed = _clip_to_bounds(mixed, bounds)

            # Local perturbation
            step = rng.normal(0, 0.05, dim) * (hi - lo)
            candidate = _clip_to_bounds(mixed + step, bounds)
            val = problem.objective(candidate)

            if val < best_val:
                best_val = val
                best_x = candidate.copy()
            current = candidate
            convergence.append(best_val)
            all_solutions_iter.append(candidate.copy())

        diversity = _swarm_diversity(np.array(all_solutions_iter)) if len(all_solutions_iter) > 1 else 0.0

        return Solution(
            best_position=best_x,
            best_value=best_val,
            diversity=diversity,
            convergence_history=convergence,
            all_solutions=_diverse_selection(all_solutions_iter, max(1, len(agents))),
        )

    elif method == "decompose":
        n_sub = min(len(agents), problem.dim)
        sub_problems = _decompose_problem(problem, n_sub)
        midpoint = np.array([(b[0] + b[1]) / 2.0 for b in bounds])
        combined = midpoint.copy()

        dim_groups = np.array_split(np.arange(dim), n_sub)
        all_solutions_dec: List[np.ndarray] = []

        for sp_idx, (sub_prob, idx_group) in enumerate(zip(sub_problems, dim_groups)):
            agent = agents[sp_idx % len(agents)]
            sub_dim = len(idx_group)
            sub_lo = np.array([sub_prob.bounds[i][0] for i in range(sub_dim)])
            sub_hi = np.array([sub_prob.bounds[i][1] for i in range(sub_dim)])
            persp = np.resize(agent.perspective, sub_dim)
            x = _clip_to_bounds(sub_lo + persp * (sub_hi - sub_lo), sub_prob.bounds)
            best_x = x.copy()
            best_val = sub_prob.objective(x)

            for _ in range(n_iterations):
                step = rng.normal(0, 0.1, sub_dim) * (sub_hi - sub_lo)
                candidate = _clip_to_bounds(x + step, sub_prob.bounds)
                val = sub_prob.objective(candidate)
                if val < best_val:
                    best_val = val
                    best_x = candidate.copy()
                x = candidate if val < sub_prob.objective(x) else x

            # Place sub-solution into combined vector
            for k_idx, global_idx in enumerate(sorted(int(i) for i in idx_group)):
                combined[global_idx] = best_x[k_idx]

        combined_val = problem.objective(combined)
        convergence.append(combined_val)
        all_solutions_dec.append(combined.copy())

        return Solution(
            best_position=combined,
            best_value=combined_val,
            diversity=0.0,
            convergence_history=convergence,
            all_solutions=all_solutions_dec,
        )

    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'diverse', 'iterative', or 'decompose'.")


def idea_marketplace(
    ideas: List[Idea],
    evaluators: List[np.ndarray],
    n_rounds: int = 50,
    decay: float = 0.1,
    rng: np.random.Generator | None = None,
) -> MarketplaceResult:
    """Simulate a double-auction idea marketplace.

    Evaluators are represented by preference vectors; their rating of an
    idea is the cosine similarity between their preference and the idea's
    embedding, scaled and shifted to [0, 1].  In each round a random
    subset of evaluators "trade" (rate) a random subset of ideas and
    market prices are updated via exponential moving average.

    Args:
        ideas: List of ideas with embeddings and initial quality.
        evaluators: List of evaluator preference vectors.
        n_rounds: Number of trading rounds.
        decay: EMA decay factor for price updates.
        rng: Numpy random generator.

    Returns:
        A :class:`MarketplaceResult` with rankings, prices, etc.
    """
    if rng is None:
        rng = np.random.default_rng()

    prices: dict[int, float] = {idea.idea_id: idea.initial_quality for idea in ideas}
    price_history: dict[int, List[float]] = {idea.idea_id: [prices[idea.idea_id]] for idea in ideas}

    eval_arr = [e / (np.linalg.norm(e) + 1e-300) for e in evaluators]

    for _ in range(n_rounds):
        # Each round: random subset of evaluators rate random subset of ideas
        n_eval = max(1, len(evaluators) // 2)
        n_ideas_sample = max(1, len(ideas) // 2)
        eval_indices = rng.choice(len(evaluators), size=n_eval, replace=False)
        idea_indices = rng.choice(len(ideas), size=n_ideas_sample, replace=False)

        # Double-auction: compute bid/ask for each (evaluator, idea) pair
        for ii in idea_indices:
            idea = ideas[ii]
            emb = idea.embedding / (np.linalg.norm(idea.embedding) + 1e-300)

            bids: List[float] = []
            for ei in eval_indices:
                cos_sim = float(np.dot(eval_arr[ei], emb))
                # Map cosine similarity from [-1,1] to [0,1]
                rating = (cos_sim + 1.0) / 2.0
                # Add noise to simulate diverse opinions
                rating += rng.normal(0, 0.05)
                rating = float(np.clip(rating, 0.0, 1.0))
                bids.append(rating)

            # Double auction: sort bids descending, asks ascending
            # Market-clearing price is the average of matched bids
            bids_sorted = sorted(bids, reverse=True)
            # Simple clearing: weighted average as the "trade" price
            if bids_sorted:
                trade_price = float(np.mean(bids_sorted[:max(1, len(bids_sorted) // 2)]))
                prices[idea.idea_id] = _market_price_update(
                    prices[idea.idea_id], trade_price, decay
                )

            price_history[idea.idea_id].append(prices[idea.idea_id])

    # Compute diversity contribution of each idea
    embeddings = np.array([idea.embedding for idea in ideas])
    emb_norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-300
    embeddings_normed = embeddings / emb_norms

    diversity_contributions: dict[int, float] = {}
    for i, idea in enumerate(ideas):
        # Mean distance to all other ideas
        if len(ideas) > 1:
            dists = np.linalg.norm(embeddings_normed - embeddings_normed[i], axis=1)
            diversity_contributions[idea.idea_id] = float(dists.sum() / (len(ideas) - 1))
        else:
            diversity_contributions[idea.idea_id] = 0.0

    # Identify undervalued ideas: high diversity contribution but low price
    div_values = np.array([diversity_contributions[idea.idea_id] for idea in ideas])
    price_values = np.array([prices[idea.idea_id] for idea in ideas])

    if div_values.std() > 1e-10 and price_values.std() > 1e-10:
        div_z = (div_values - div_values.mean()) / (div_values.std() + 1e-300)
        price_z = (price_values - price_values.mean()) / (price_values.std() + 1e-300)
        undervalued = [
            ideas[i].idea_id
            for i in range(len(ideas))
            if div_z[i] > 0.5 and price_z[i] < -0.5
        ]
    else:
        undervalued = []

    # Market efficiency: correlation between final prices and initial quality
    true_qualities = np.array([idea.initial_quality for idea in ideas])
    if true_qualities.std() > 1e-10 and price_values.std() > 1e-10:
        corr_matrix = np.corrcoef(true_qualities, price_values)
        market_efficiency = float(corr_matrix[0, 1])
    else:
        market_efficiency = 0.0

    # Rankings by final price (descending)
    ranked = sorted(ideas, key=lambda idea: prices[idea.idea_id], reverse=True)
    rankings = [idea.idea_id for idea in ranked]

    return MarketplaceResult(
        rankings=rankings,
        final_prices=dict(prices),
        undervalued=undervalued,
        market_efficiency=market_efficiency,
        price_history=price_history,
    )


def innovation_tournament(
    challenges: List[Challenge],
    participants: List[SwarmAgent],
    n_stages: int = 3,
    n_iterations_per_stage: int = 20,
    rng: np.random.Generator | None = None,
) -> TournamentResult:
    """Run a multi-stage innovation tournament.

    Stage 1: All participants submit solutions to each challenge.
    Stage 2+: Top participants from previous stage refine solutions;
    scores include a diversity bonus rewarding novel approaches.

    Args:
        challenges: List of challenges with objective functions.
        participants: List of agents / participants.
        n_stages: Number of tournament stages.
        n_iterations_per_stage: Local search iterations per stage.
        rng: Numpy random generator.

    Returns:
        A :class:`TournamentResult` with winners and metrics.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_participants = len(participants)
    bracket = _tournament_bracket(n_participants, n_stages)

    # Track per-participant best solutions across all challenges
    active_indices = list(range(n_participants))
    participant_solutions: dict[int, List[np.ndarray]] = {i: [] for i in range(n_participants)}
    stage_scores: List[List[Tuple[int, float]]] = []

    for stage in range(n_stages):
        scores_this_stage: List[Tuple[int, float]] = []
        solutions_this_stage: List[np.ndarray] = []

        for p_idx in active_indices:
            agent = participants[p_idx]
            total_score = 0.0

            for challenge in challenges:
                cdim = len(challenge.bounds)
                c_lo = np.array([b[0] for b in challenge.bounds])
                c_hi = np.array([b[1] for b in challenge.bounds])

                # Initialise from agent perspective
                persp = np.resize(agent.perspective, cdim)
                x = _clip_to_bounds(c_lo + persp * (c_hi - c_lo), challenge.bounds)

                # If we have a previous solution, start from a mix
                if participant_solutions[p_idx]:
                    prev = participant_solutions[p_idx][-1]
                    if len(prev) == cdim:
                        x = _clip_to_bounds(0.5 * x + 0.5 * prev, challenge.bounds)

                best_x = x.copy()
                best_val = challenge.objective(x)

                for _ in range(n_iterations_per_stage):
                    step = rng.normal(0, 0.1, cdim) * (c_hi - c_lo)
                    candidate = _clip_to_bounds(x + step, challenge.bounds)
                    val = challenge.objective(candidate)
                    if val > best_val:  # Challenges: higher is better
                        best_val = val
                        best_x = candidate.copy()
                    x = candidate if val > challenge.objective(x) else x

                total_score += best_val
                participant_solutions[p_idx].append(best_x)
                solutions_this_stage.append(best_x)

            # Diversity bonus: distance from mean of all submitted solutions
            if solutions_this_stage:
                mean_sol = np.mean(solutions_this_stage, axis=0)
                last_sol = participant_solutions[p_idx][-1]
                if len(last_sol) == len(mean_sol):
                    div_bonus = float(np.linalg.norm(last_sol - mean_sol))
                else:
                    div_bonus = 0.0
                total_score += 0.1 * div_bonus

            scores_this_stage.append((p_idx, total_score))

        stage_scores.append(scores_this_stage)

        # Advance top half
        scores_this_stage.sort(key=lambda t: t[1], reverse=True)
        n_advance = max(1, len(active_indices) // 2)
        active_indices = [t[0] for t in scores_this_stage[:n_advance]]

    # Final winners
    final_scores = stage_scores[-1]
    final_scores.sort(key=lambda t: t[1], reverse=True)
    winners = final_scores

    winning_solutions = []
    for p_idx, _ in winners:
        if participant_solutions[p_idx]:
            winning_solutions.append(participant_solutions[p_idx][-1])

    # Innovation quality: mean score improvement from first to last stage
    if len(stage_scores) >= 2:
        first_mean = float(np.mean([s for _, s in stage_scores[0]]))
        last_mean = float(np.mean([s for _, s in stage_scores[-1]]))
        innovation_quality = last_mean - first_mean
    else:
        innovation_quality = 0.0

    # Diversity of winning solutions
    if len(winning_solutions) > 1:
        diversity_of_winners = _swarm_diversity(np.array(winning_solutions))
    else:
        diversity_of_winners = 0.0

    return TournamentResult(
        winners=winners,
        stage_scores=stage_scores,
        innovation_quality=innovation_quality,
        diversity_of_winners=diversity_of_winners,
        winning_solutions=winning_solutions,
    )


def crowd_labeling_diverse(
    items: List[LabelItem],
    labelers: dict[int, dict[int, int]],
    n_classes: int = 2,
    diversity_req: float = 0.3,
    labeler_types: dict[int, str] | None = None,
    confidence_threshold: float = 0.8,
) -> Labels:
    """Diverse crowd labeling using Dawid-Skene EM.

    Multiple labelers label items.  The algorithm estimates true labels
    and labeler quality via EM.  A diversity requirement ensures that
    confident labels are only accepted when they come from a sufficiently
    diverse set of labeler types.

    Args:
        items: Items to be labeled.
        labelers: ``{item_id: {labeler_id: label}}``.
        n_classes: Number of label classes.
        diversity_req: Minimum fraction of distinct labeler types that
            must be represented among an item's labelers for the label
            to be considered confident.
        labeler_types: ``{labeler_id: type_string}``. If ``None``, each
            labeler is its own type.
        confidence_threshold: Minimum max-class probability to consider
            a label confident.

    Returns:
        A :class:`Labels` result with estimated labels and diagnostics.
    """
    # Build annotations dict in Dawid-Skene format
    annotations: dict[int, dict[int, int]] = {}
    all_labeler_ids: set[int] = set()
    for item in items:
        if item.item_id in labelers:
            annotations[item.item_id] = labelers[item.item_id]
            all_labeler_ids.update(labelers[item.item_id].keys())
        else:
            annotations[item.item_id] = {}

    if labeler_types is None:
        labeler_types = {lid: str(lid) for lid in all_labeler_ids}

    # Run Dawid-Skene EM
    class_probs, error_rates = _dawid_skene_em(annotations, n_classes)

    # Estimated labels
    estimated_labels: dict[int, int] = {}
    label_probabilities: dict[int, np.ndarray] = {}
    for item in items:
        if item.item_id in class_probs:
            probs = class_probs[item.item_id]
            estimated_labels[item.item_id] = int(np.argmax(probs))
            label_probabilities[item.item_id] = probs
        else:
            estimated_labels[item.item_id] = 0
            label_probabilities[item.item_id] = np.ones(n_classes) / n_classes

    # Labeler quality: trace of confusion matrix / n_classes (perfect = 1)
    labeler_quality: dict[int, float] = {}
    for lid in all_labeler_ids:
        if lid in error_rates:
            labeler_quality[lid] = float(np.trace(error_rates[lid]) / n_classes)
        else:
            labeler_quality[lid] = 0.5

    # Confident items: high probability AND diverse labelers
    all_types = set(labeler_types.values())
    n_total_types = len(all_types) if all_types else 1
    confident_items: List[int] = []

    for item in items:
        iid = item.item_id
        probs = label_probabilities.get(iid, np.ones(n_classes) / n_classes)
        max_prob = float(probs.max())

        if max_prob < confidence_threshold:
            continue

        # Check diversity of labelers for this item
        item_labelers = annotations.get(iid, {})
        types_present = set(labeler_types.get(lid, str(lid)) for lid in item_labelers.keys())
        type_diversity = len(types_present) / n_total_types if n_total_types > 0 else 0.0

        if type_diversity >= diversity_req:
            confident_items.append(iid)

    # Inter-annotator agreement (observed agreement)
    agreement_pairs = 0
    agreement_count = 0
    for item in items:
        item_labels_list = list(annotations.get(item.item_id, {}).values())
        n_lab = len(item_labels_list)
        for i in range(n_lab):
            for j in range(i + 1, n_lab):
                agreement_pairs += 1
                if item_labels_list[i] == item_labels_list[j]:
                    agreement_count += 1

    observed_agreement = agreement_count / agreement_pairs if agreement_pairs > 0 else 0.0

    # Expected agreement under random assignment
    all_labels_flat: List[int] = []
    for item in items:
        all_labels_flat.extend(annotations.get(item.item_id, {}).values())
    if all_labels_flat:
        label_counts = np.bincount(all_labels_flat, minlength=n_classes).astype(float)
        label_freq = label_counts / label_counts.sum()
        expected_agreement = float((label_freq ** 2).sum())
    else:
        expected_agreement = 1.0 / n_classes

    if abs(1.0 - expected_agreement) < 1e-10:
        inter_annotator_agreement = 1.0
    else:
        inter_annotator_agreement = (observed_agreement - expected_agreement) / (
            1.0 - expected_agreement
        )

    return Labels(
        estimated_labels=estimated_labels,
        labeler_quality=labeler_quality,
        inter_annotator_agreement=inter_annotator_agreement,
        confident_items=confident_items,
        label_probabilities=label_probabilities,
    )


def swarm_intelligence(
    objective: Callable[[np.ndarray], float],
    agents: List[SwarmAgent],
    n_iterations: int = 100,
    w: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
    diversity_threshold: float | None = None,
    repulsion_strength: float = 0.5,
    bounds: Sequence[Tuple[float, float]] | None = None,
    rng: np.random.Generator | None = None,
) -> SwarmResult:
    """Particle swarm optimisation with diversity maintenance.

    Standard PSO with an additional repulsion force activated when swarm
    diversity (mean pairwise distance) drops below a threshold.  The
    repulsion pushes each particle away from the swarm centroid.

    Args:
        objective: Scalar function to minimise.
        agents: Initial swarm agents with position/velocity.
        n_iterations: Number of PSO iterations.
        w: Inertia weight.
        c1: Cognitive coefficient.
        c2: Social coefficient.
        diversity_threshold: If swarm diversity drops below this value,
            repulsion is activated. Defaults to 10% of initial diversity.
        repulsion_strength: Magnitude of repulsion force.
        bounds: Optional per-dimension (lo, hi) bounds for clamping.
        rng: Numpy random generator.

    Returns:
        A :class:`SwarmResult` with best position, convergence, and
        diversity histories.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(agents)
    dim = len(agents[0].position)
    positions = np.array([a.position.copy() for a in agents], dtype=np.float64)
    velocities = np.array([a.velocity.copy() for a in agents], dtype=np.float64)

    # Evaluate initial positions
    values = np.array([objective(positions[i]) for i in range(n)])
    pbest = positions.copy()
    pbest_values = values.copy()
    gbest_idx = int(np.argmin(values))
    gbest = positions[gbest_idx].copy()
    gbest_value = float(values[gbest_idx])

    initial_diversity = _swarm_diversity(positions)
    if diversity_threshold is None:
        diversity_threshold = 0.1 * initial_diversity if initial_diversity > 0 else 1.0

    convergence_history: List[float] = [gbest_value]
    diversity_history: List[float] = [initial_diversity]

    for _ in range(n_iterations):
        # Standard PSO update
        positions, velocities = _pso_update(
            positions, velocities, pbest, gbest, w, c1, c2, rng
        )

        # Diversity maintenance: repulsion if diversity too low
        current_diversity = _swarm_diversity(positions)
        if current_diversity < diversity_threshold:
            centroid = positions.mean(axis=0)
            for i in range(n):
                direction = positions[i] - centroid
                norm = np.linalg.norm(direction)
                if norm > 1e-10:
                    positions[i] += repulsion_strength * direction / norm
                else:
                    positions[i] += rng.normal(0, repulsion_strength, dim)

        # Clip to bounds if provided
        if bounds is not None:
            positions = _clip_to_bounds(positions, bounds)

        # Evaluate
        values = np.array([objective(positions[i]) for i in range(n)])

        # Update personal bests
        improved = values < pbest_values
        pbest[improved] = positions[improved]
        pbest_values[improved] = values[improved]

        # Update global best
        current_best_idx = int(np.argmin(pbest_values))
        if pbest_values[current_best_idx] < gbest_value:
            gbest = pbest[current_best_idx].copy()
            gbest_value = float(pbest_values[current_best_idx])

        convergence_history.append(gbest_value)
        diversity_history.append(_swarm_diversity(positions))

    return SwarmResult(
        best_position=gbest,
        best_value=gbest_value,
        convergence_history=convergence_history,
        diversity_history=diversity_history,
        final_positions=[positions[i].copy() for i in range(n)],
    )
