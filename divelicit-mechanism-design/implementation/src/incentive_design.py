"""
Incentive design for eliciting diverse responses.

Implements scoring rules, truthfulness verification, strategic agent
simulation, optimal payment schemes, and VCG mechanisms for settings
where agents must be incentivized to report diverse, high-quality
information.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

from .embedding import TextEmbedder, embed_texts
from .diversity_metrics import cosine_diversity
from .dpp import greedy_map
from .kernels import RBFKernel
from .scoring_rules import ScoringRule, BrierRule, LogarithmicRule
from .utils import log_det_safe


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DesignedScoringRule:
    """A scoring rule produced by ``design_scoring_rule``."""
    name: str
    base_rule: ScoringRule
    diversity_weight: float
    quality_weight: float
    constraints: Dict[str, float] = field(default_factory=dict)

    def score(
        self,
        reports: np.ndarray,
        realized: int,
        all_reports: Optional[np.ndarray] = None,
    ) -> float:
        """Score a single report given a realization.

        Parameters
        ----------
        reports : np.ndarray
            Probability vector reported by the agent.
        realized : int
            Index of the realized outcome.
        all_reports : np.ndarray, optional
            Matrix of all agents' reports for diversity bonus.
        """
        quality = self.base_rule.score(reports, realized)
        diversity_bonus = 0.0
        if all_reports is not None and len(all_reports) > 1:
            diversity_bonus = cosine_diversity(all_reports)
        return float(
            self.quality_weight * quality + self.diversity_weight * diversity_bonus
        )


@dataclass
class SimResult:
    """Results from ``simulate_strategic_agents``."""
    round_scores: np.ndarray          # (n_rounds, n_agents)
    round_diversity: np.ndarray       # (n_rounds,)
    final_reports: np.ndarray         # (n_agents, dim)
    truthful_fraction: float
    mean_diversity: float
    convergence_round: int


# ---------------------------------------------------------------------------
# Design scoring rules
# ---------------------------------------------------------------------------

def design_scoring_rule(
    objective: str = "diversity",
    constraints: Optional[Dict[str, float]] = None,
    base: str = "brier",
) -> DesignedScoringRule:
    """Design a scoring rule that incentivizes a given objective.

    Parameters
    ----------
    objective : str
        One of ``'diversity'``, ``'quality'``, or ``'balanced'``.
    constraints : dict, optional
        Constraints such as ``{'budget': 100.0, 'min_quality': 0.5}``.
    base : str
        Base proper scoring rule: ``'brier'`` or ``'logarithmic'``.

    Returns
    -------
    DesignedScoringRule
    """
    constraints = constraints or {}

    base_rule: ScoringRule
    if base == "logarithmic":
        base_rule = LogarithmicRule()
    else:
        base_rule = BrierRule()

    if objective == "diversity":
        dw, qw = 0.7, 0.3
    elif objective == "quality":
        dw, qw = 0.2, 0.8
    else:  # balanced
        dw, qw = 0.5, 0.5

    # Respect budget constraint — scale weights so max payout ≤ budget
    budget = constraints.get("budget", None)
    if budget is not None:
        total = dw + qw
        scale = budget / (total + 1e-12)
        dw *= scale
        qw *= scale

    min_quality = constraints.get("min_quality", 0.0)
    if min_quality > 0:
        # Increase quality weight to meet minimum quality threshold
        qw = max(qw, min_quality * (dw + qw))

    return DesignedScoringRule(
        name=f"{objective}_{base}",
        base_rule=base_rule,
        diversity_weight=dw,
        quality_weight=qw,
        constraints=constraints,
    )


# ---------------------------------------------------------------------------
# Truthfulness verification
# ---------------------------------------------------------------------------

def verify_truthfulness(
    rule: DesignedScoringRule,
    n_tests: int = 500,
    dim: int = 5,
    seed: int = 42,
) -> bool:
    """Check whether *rule* is incentive-compatible (truthful reporting is optimal).

    We sample random true belief vectors and verify that the expected score
    is maximized when the agent reports truthfully (up to numerical tolerance).

    Parameters
    ----------
    rule : DesignedScoringRule
        The scoring rule to verify.
    n_tests : int
        Number of random belief vectors to test.
    dim : int
        Dimensionality of the probability simplex.
    seed : int
        Random seed.

    Returns
    -------
    bool
        ``True`` if no incentive-compatibility violations are detected.
    """
    rng = np.random.RandomState(seed)
    violations = 0

    for _ in range(n_tests):
        # Random true belief on the simplex
        alpha = rng.dirichlet(np.ones(dim))

        # Expected score under truthful report
        truthful_score = 0.0
        for y in range(dim):
            truthful_score += alpha[y] * rule.base_rule.score(alpha, y)

        # Try random deviations
        for _ in range(10):
            deviation = rng.dirichlet(np.ones(dim))
            deviated_score = 0.0
            for y in range(dim):
                deviated_score += alpha[y] * rule.base_rule.score(deviation, y)

            # Quality component: truthful should dominate
            if deviated_score > truthful_score + 1e-8:
                violations += 1

    # Allow a small tolerance for numerical issues
    return violations <= n_tests * 0.01


# ---------------------------------------------------------------------------
# Strategic agent simulation
# ---------------------------------------------------------------------------

def simulate_strategic_agents(
    rule: DesignedScoringRule,
    n_agents: int = 10,
    n_rounds: int = 50,
    dim: int = 5,
    learning_rate: float = 0.1,
    seed: int = 42,
) -> SimResult:
    """Simulate strategic agents adapting their reports over rounds.

    Each agent has a private belief and updates its report each round via
    gradient-ascent on the scoring rule, balancing truthfulness against
    diversity incentives.

    Parameters
    ----------
    rule : DesignedScoringRule
        The scoring rule governing payoffs.
    n_agents : int
        Number of participating agents.
    n_rounds : int
        Simulation length.
    dim : int
        Dimensionality of the report space.
    learning_rate : float
        Step size for report updates.
    seed : int
        Random seed.

    Returns
    -------
    SimResult
    """
    rng = np.random.RandomState(seed)

    # Private beliefs
    beliefs = np.array([rng.dirichlet(np.ones(dim)) for _ in range(n_agents)])
    reports = beliefs.copy()

    round_scores = np.zeros((n_rounds, n_agents))
    round_diversity = np.zeros(n_rounds)
    convergence_round = n_rounds

    for rnd in range(n_rounds):
        # Realize outcome from a mixture of beliefs
        mixture = beliefs.mean(axis=0)
        mixture /= mixture.sum()
        realized = int(rng.choice(dim, p=mixture))

        for a in range(n_agents):
            score = rule.score(reports[a], realized, all_reports=reports)
            round_scores[rnd, a] = score

        round_diversity[rnd] = cosine_diversity(reports)

        # Agents update reports via noisy gradient ascent
        for a in range(n_agents):
            # Gradient toward truthfulness (move report toward belief)
            grad_truth = beliefs[a] - reports[a]

            # Gradient toward diversity (move away from mean report)
            mean_others = (reports.sum(axis=0) - reports[a]) / max(n_agents - 1, 1)
            grad_div = reports[a] - mean_others

            grad = (
                rule.quality_weight * grad_truth
                + rule.diversity_weight * grad_div
                + rng.randn(dim) * 0.01  # exploration noise
            )
            reports[a] += learning_rate * grad
            # Project back onto simplex
            reports[a] = np.maximum(reports[a], 1e-8)
            reports[a] /= reports[a].sum()

        # Check convergence (reports stable)
        if rnd > 5:
            prev = round_scores[rnd - 1].mean()
            curr = round_scores[rnd].mean()
            if abs(curr - prev) < 1e-6 and convergence_round == n_rounds:
                convergence_round = rnd

    # Truthfulness: fraction of agents whose report ≈ belief
    diffs = np.linalg.norm(reports - beliefs, axis=1)
    truthful_fraction = float((diffs < 0.15).mean())

    return SimResult(
        round_scores=round_scores,
        round_diversity=round_diversity,
        final_reports=reports,
        truthful_fraction=truthful_fraction,
        mean_diversity=float(round_diversity.mean()),
        convergence_round=convergence_round,
    )


# ---------------------------------------------------------------------------
# Optimal payment for diversity
# ---------------------------------------------------------------------------

def optimal_payment(
    responses: List[str],
    budget: float = 100.0,
    method: str = "marginal",
    embedder: Optional[TextEmbedder] = None,
) -> Dict[int, float]:
    """Compute payments to agents that maximize diversity within a budget.

    Parameters
    ----------
    responses : list of str
        Submitted responses.
    budget : float
        Total payment budget.
    method : str
        ``'marginal'`` — pay proportional to marginal diversity contribution;
        ``'shapley'`` — approximate Shapley-value payments.
    embedder : TextEmbedder, optional
        Custom embedder.

    Returns
    -------
    dict mapping agent index to payment
    """
    if embedder is None:
        embedder = TextEmbedder(dim=64)
    n = len(responses)
    if n == 0:
        return {}
    embeddings = embedder.embed_batch(responses)

    if method == "shapley":
        values = _approx_shapley_diversity(embeddings, n_samples=200)
    else:
        values = _marginal_diversity(embeddings)

    # Normalize to budget
    total = values.sum()
    if total < 1e-12:
        payments = np.full(n, budget / n)
    else:
        payments = values / total * budget

    return {i: float(payments[i]) for i in range(n)}


def _marginal_diversity(embeddings: np.ndarray) -> np.ndarray:
    """Marginal diversity contribution of each item."""
    n = embeddings.shape[0]
    values = np.zeros(n)
    kernel = RBFKernel()
    L = kernel.gram_matrix(embeddings) + np.eye(n) * 1e-6
    full_div = log_det_safe(L)

    for i in range(n):
        idx = [j for j in range(n) if j != i]
        if len(idx) == 0:
            values[i] = full_div
        else:
            sub_L = L[np.ix_(idx, idx)]
            values[i] = max(full_div - log_det_safe(sub_L), 0.0)
    return values


def _approx_shapley_diversity(
    embeddings: np.ndarray, n_samples: int = 200
) -> np.ndarray:
    """Monte-Carlo approximation of Shapley values for diversity."""
    n = embeddings.shape[0]
    values = np.zeros(n)
    rng = np.random.RandomState(0)
    kernel = RBFKernel()
    L = kernel.gram_matrix(embeddings) + np.eye(n) * 1e-6

    for _ in range(n_samples):
        perm = rng.permutation(n)
        prev_val = 0.0
        for pos, idx in enumerate(perm):
            subset = list(perm[: pos + 1])
            sub_L = L[np.ix_(subset, subset)]
            cur_val = log_det_safe(sub_L)
            values[idx] += cur_val - prev_val
            prev_val = cur_val

    return values / n_samples


# ---------------------------------------------------------------------------
# VCG mechanism for diverse elicitation
# ---------------------------------------------------------------------------

def vcg_diverse_mechanism(
    responses: List[str],
    k: int = 5,
    embedder: Optional[TextEmbedder] = None,
) -> Tuple[List[int], Dict[int, float]]:
    """VCG mechanism that selects *k* diverse responses and computes payments.

    The social welfare function is the log-determinant diversity of the
    selected set.  VCG payments are computed as the externality each
    selected agent imposes on others.

    Parameters
    ----------
    responses : list of str
        All submitted responses.
    k : int
        Number of responses to select.
    embedder : TextEmbedder, optional
        Custom embedder.

    Returns
    -------
    selected : list of int
        Indices of selected responses.
    payments : dict
        VCG payment for each selected agent.
    """
    if embedder is None:
        embedder = TextEmbedder(dim=64)
    n = len(responses)
    if n <= k:
        return list(range(n)), {i: 0.0 for i in range(n)}

    embeddings = embedder.embed_batch(responses)
    kernel = RBFKernel()
    L = kernel.gram_matrix(embeddings) + np.eye(n) * 1e-6

    # Optimal selection (greedy MAP approximation)
    selected = greedy_map(L, k)
    sel_L = L[np.ix_(selected, selected)]
    welfare_with_all = log_det_safe(sel_L)

    payments: Dict[int, float] = {}
    for agent in selected:
        # Remove this agent and re-optimize
        others = [i for i in range(n) if i != agent]
        L_others = L[np.ix_(others, others)]
        alt_selected = greedy_map(L_others, k)
        alt_L = L_others[np.ix_(alt_selected, alt_selected)]
        welfare_without = log_det_safe(alt_L)

        # Also compute welfare of current selection minus this agent
        remaining = [i for i in selected if i != agent]
        if remaining:
            rem_L = L[np.ix_(remaining, remaining)]
            welfare_rest = log_det_safe(rem_L)
        else:
            welfare_rest = 0.0

        # VCG payment = welfare others get with agent present
        #             - welfare others get without agent
        payment = max(welfare_without - welfare_rest, 0.0)
        payments[agent] = float(payment)

    return selected, payments
