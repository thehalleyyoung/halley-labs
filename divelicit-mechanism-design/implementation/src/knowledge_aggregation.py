"""Knowledge aggregation methods for diverse LLM generation.

Implements prediction markets, Delphi method, superforecasting, and
wisdom-of-crowds aggregation to combine answers from multiple sources
into high-quality consensus outputs with calibrated confidence.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .embedding import TextEmbedder, embed_texts
from .diversity_metrics import cosine_diversity
from .kernels import AdaptiveRBFKernel


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AggregatedAnswer:
    """Result of aggregating answers from multiple sources."""

    question: str
    answer: str
    confidence: float
    source_contributions: Dict[str, float]
    agreement_level: float
    dissenting_views: List[str]


@dataclass
class MarketResult:
    """Result of a prediction market simulation."""

    question: str
    final_price: float
    price_history: List[float]
    trader_profits: Dict[str, float]
    convergence_round: int
    liquidity: float


@dataclass
class DelphiResult:
    """Result of an iterative Delphi method process."""

    question: str
    final_consensus: str
    round_summaries: List[str]
    convergence_score: float
    remaining_disagreements: List[str]
    expert_confidence: Dict[str, float]


@dataclass
class Forecast:
    """Result of a superforecasting aggregation."""

    question: str
    point_estimate: float
    confidence_interval: Tuple[float, float]
    calibration_score: float
    forecaster_weights: Dict[str, float]
    reasoning_summary: str


@dataclass
class CrowdWisdom:
    """Result of wisdom-of-crowds aggregation."""

    question: str
    aggregated_answer: str
    crowd_size: int
    diversity_score: float
    accuracy_estimate: float
    outlier_insights: List[str]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _deterministic_hash(text: str) -> float:
    """Return a deterministic float in [0, 1) from *text*."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / (1 << 64)


def _deterministic_hash_array(text: str, size: int) -> np.ndarray:
    """Return a deterministic array of floats in [0, 1) from *text*."""
    values = np.empty(size)
    for i in range(size):
        values[i] = _deterministic_hash(f"{text}__dim{i}")
    return values


def _lmsr_cost(q: np.ndarray, b: float) -> float:
    """Logarithmic market scoring rule cost function.

    Parameters
    ----------
    q : np.ndarray
        Outstanding quantity vector for each outcome.
    b : float
        Liquidity parameter controlling price sensitivity.

    Returns
    -------
    float
        The LMSR cost C(q) = b * log(sum(exp(q_i / b))).
    """
    # Numerically stable log-sum-exp
    q_shifted = q - np.max(q)
    return b * (np.max(q) + np.log(np.sum(np.exp(q_shifted / b))))


def _extremize(probability: float, factor: float) -> float:
    """Push *probability* away from 0.5 by *factor*.

    Uses the log-odds transformation: convert to log-odds, multiply by
    *factor*, convert back.  A factor > 1 extremises, < 1 moderates.

    Parameters
    ----------
    probability : float
        Input probability in (0, 1).
    factor : float
        Extremising multiplier (>1 pushes toward 0 or 1).

    Returns
    -------
    float
        Extremised probability clamped to [0.001, 0.999].
    """
    prob = np.clip(probability, 1e-9, 1.0 - 1e-9)
    log_odds = np.log(prob / (1.0 - prob))
    adjusted = log_odds * factor
    result = 1.0 / (1.0 + np.exp(-adjusted))
    return float(np.clip(result, 0.001, 0.999))


def _bootstrap_confidence(
    values: np.ndarray,
    confidence: float = 0.9,
    n_bootstrap: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """Compute a bootstrap confidence interval for the mean of *values*.

    Parameters
    ----------
    values : np.ndarray
        Sample values.
    confidence : float
        Desired confidence level (e.g. 0.9 for 90 %).
    n_bootstrap : int
        Number of bootstrap resamples.
    rng : np.random.Generator, optional
        Random generator for reproducibility.

    Returns
    -------
    Tuple[float, float]
        (lower, upper) bounds of the confidence interval.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        means[i] = np.mean(sample)
    alpha = (1.0 - confidence) / 2.0
    lower = float(np.percentile(means, 100 * alpha))
    upper = float(np.percentile(means, 100 * (1.0 - alpha)))
    return (lower, upper)


def _calibration_score(
    predictions: np.ndarray, outcomes: np.ndarray
) -> float:
    """Compute calibration via Brier score (lower is better).

    Parameters
    ----------
    predictions : np.ndarray
        Predicted probabilities in [0, 1].
    outcomes : np.ndarray
        Binary outcomes (0 or 1).

    Returns
    -------
    float
        Mean Brier score.
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.float64)
    if len(predictions) == 0:
        return 0.0
    return float(np.mean((predictions - outcomes) ** 2))


def _pairwise_cosine_similarity(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix.

    Parameters
    ----------
    embeddings : np.ndarray
        Matrix of shape (n, d) where each row is an embedding vector.

    Returns
    -------
    np.ndarray
        Similarity matrix of shape (n, n) with values in [-1, 1].
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normed = embeddings / norms
    return normed @ normed.T


def _centroid_distances(embeddings: np.ndarray) -> np.ndarray:
    """Compute cosine similarity of each embedding to the centroid.

    Parameters
    ----------
    embeddings : np.ndarray
        Matrix of shape (n, d).

    Returns
    -------
    np.ndarray
        Array of shape (n,) with cosine similarity to centroid.
    """
    centroid = np.mean(embeddings, axis=0, keepdims=True)
    c_norm = np.linalg.norm(centroid)
    if c_norm < 1e-12:
        return np.ones(len(embeddings))
    centroid_normed = centroid / c_norm
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normed = embeddings / norms
    return (normed @ centroid_normed.T).ravel()


# ---------------------------------------------------------------------------
# Main aggregation functions
# ---------------------------------------------------------------------------


def aggregate_answers(
    question: str, sources: List[str]
) -> AggregatedAnswer:
    """Aggregate answers from multiple sources via embedding similarity.

    Embeds all source answers, computes pairwise cosine similarities,
    selects the answer closest to the centroid as the aggregate, and
    identifies dissenting views that are far from the centroid.

    Parameters
    ----------
    question : str
        The question being answered.
    sources : List[str]
        List of answer strings from different sources.

    Returns
    -------
    AggregatedAnswer
        Aggregated result with confidence and source contributions.
    """
    if not sources:
        return AggregatedAnswer(
            question=question,
            answer="",
            confidence=0.0,
            source_contributions={},
            agreement_level=0.0,
            dissenting_views=[],
        )

    # Embed all source answers
    embeddings = embed_texts(sources)  # (n, d)
    n = len(sources)

    # Pairwise similarity and agreement level
    sim_matrix = _pairwise_cosine_similarity(embeddings)
    if n > 1:
        upper_tri = sim_matrix[np.triu_indices(n, k=1)]
        agreement_level = float(np.mean(upper_tri))
    else:
        agreement_level = 1.0

    # Distance of each source to centroid
    centroid_sims = _centroid_distances(embeddings)

    # Select answer closest to centroid
    best_idx = int(np.argmax(centroid_sims))
    answer = sources[best_idx]

    # Source contributions weighted by similarity to centroid
    raw_weights = np.maximum(centroid_sims, 0.0)
    weight_sum = raw_weights.sum()
    if weight_sum < 1e-12:
        normalised = np.ones(n) / n
    else:
        normalised = raw_weights / weight_sum
    source_contributions = {
        f"source_{i}": float(normalised[i]) for i in range(n)
    }

    # Dissenting views: similarity to centroid below mean - 1 std
    threshold = float(np.mean(centroid_sims) - np.std(centroid_sims))
    dissenting_views = [
        sources[i] for i in range(n) if centroid_sims[i] < threshold
    ]

    # Confidence derived from agreement and centroid proximity
    confidence = float(
        0.5 * agreement_level + 0.5 * centroid_sims[best_idx]
    )
    confidence = float(np.clip(confidence, 0.0, 1.0))

    return AggregatedAnswer(
        question=question,
        answer=answer,
        confidence=confidence,
        source_contributions=source_contributions,
        agreement_level=float(agreement_level),
        dissenting_views=dissenting_views,
    )


def prediction_market(
    question: str,
    agents: List[str],
    rounds: int = 10,
) -> MarketResult:
    """Simulate a prediction market using LMSR pricing.

    Each agent derives a private belief from a deterministic hash of
    agent+question.  Agents trade to push the market price toward their
    belief.  The market converges when price changes drop below epsilon.

    Parameters
    ----------
    question : str
        The question being priced.
    agents : List[str]
        List of agent identifiers.
    rounds : int
        Maximum number of trading rounds.

    Returns
    -------
    MarketResult
        Market result with price history and trader profits.
    """
    n_agents = len(agents)
    if n_agents == 0:
        return MarketResult(
            question=question,
            final_price=0.5,
            price_history=[0.5],
            trader_profits={},
            convergence_round=0,
            liquidity=0.0,
        )

    # Liquidity parameter
    b = max(2.0, math.log(n_agents + 1) * 2.0)

    # Outstanding quantities for binary outcomes (yes / no)
    q = np.zeros(2)

    # Derive agent beliefs deterministically
    beliefs = np.array(
        [_deterministic_hash(f"{agent}|{question}") for agent in agents]
    )
    # Clamp beliefs away from extremes
    beliefs = np.clip(beliefs, 0.05, 0.95)

    price_history: List[float] = []
    trader_positions: Dict[str, np.ndarray] = {
        agent: np.zeros(2) for agent in agents
    }
    trader_costs: Dict[str, float] = {agent: 0.0 for agent in agents}

    epsilon = 1e-4
    convergence_round = rounds

    for r in range(rounds):
        # Current market price for outcome 0 (yes)
        exp_q = np.exp(q / b)
        price = float(exp_q[0] / np.sum(exp_q))
        price_history.append(price)

        prev_price = price
        for i, agent in enumerate(agents):
            # Agent trades proportional to (belief - current price)
            belief = beliefs[i]
            exp_q_now = np.exp(q / b)
            current_price = exp_q_now[0] / np.sum(exp_q_now)

            delta = belief - current_price
            trade_size = delta * b * 0.5  # tempered trade

            old_cost = _lmsr_cost(q, b)
            if trade_size > 0:
                q[0] += abs(trade_size)
            else:
                q[1] += abs(trade_size)
            new_cost = _lmsr_cost(q, b)

            cost = new_cost - old_cost
            trader_costs[agent] += cost
            if trade_size > 0:
                trader_positions[agent][0] += abs(trade_size)
            else:
                trader_positions[agent][1] += abs(trade_size)

        # Check convergence
        exp_q_end = np.exp(q / b)
        end_price = float(exp_q_end[0] / np.sum(exp_q_end))
        if abs(end_price - prev_price) < epsilon:
            convergence_round = r + 1
            price_history.append(end_price)
            break

    # Final price
    exp_q_final = np.exp(q / b)
    final_price = float(exp_q_final[0] / np.sum(exp_q_final))
    if price_history[-1] != final_price:
        price_history.append(final_price)

    # Compute trader profits (value of position - cost paid)
    # Assume outcome = round(final_price), payoff = 1 for correct, 0 else
    outcome = int(round(final_price))
    trader_profits: Dict[str, float] = {}
    for agent in agents:
        payoff = float(trader_positions[agent][outcome])
        trader_profits[agent] = payoff - trader_costs[agent]

    return MarketResult(
        question=question,
        final_price=final_price,
        price_history=price_history,
        trader_profits=trader_profits,
        convergence_round=convergence_round,
        liquidity=float(b),
    )


def delphi_method(
    question: str,
    experts: List[str],
    rounds: int = 3,
) -> DelphiResult:
    """Implement the iterative Delphi method for expert consensus.

    Round 1 collects independent estimates.  Subsequent rounds share the
    group median and IQR, allowing experts to revise their estimates
    while maintaining independent reasoning.  Convergence is tracked via
    the decreasing IQR across rounds.

    Parameters
    ----------
    question : str
        The question posed to the expert panel.
    experts : List[str]
        List of expert identifiers.
    rounds : int
        Number of Delphi rounds (minimum 1).

    Returns
    -------
    DelphiResult
        Consensus result with round summaries and remaining disagreements.
    """
    n_experts = len(experts)
    if n_experts == 0:
        return DelphiResult(
            question=question,
            final_consensus="",
            round_summaries=[],
            convergence_score=0.0,
            remaining_disagreements=[],
            expert_confidence={},
        )

    rounds = max(1, rounds)

    # Initial independent estimates derived from hash
    estimates = np.array(
        [_deterministic_hash(f"{expert}|{question}|r0") for expert in experts]
    )

    round_summaries: List[str] = []
    iqr_history: List[float] = []

    for r in range(rounds):
        median = float(np.median(estimates))
        q1 = float(np.percentile(estimates, 25))
        q3 = float(np.percentile(estimates, 75))
        iqr = q3 - q1
        iqr_history.append(iqr)

        summary = (
            f"Round {r + 1}: median={median:.4f}, "
            f"IQR={iqr:.4f}, range=[{float(np.min(estimates)):.4f}, "
            f"{float(np.max(estimates)):.4f}]"
        )
        round_summaries.append(summary)

        if r < rounds - 1:
            # Experts revise toward median but keep some independence
            revision_strength = 0.3 + 0.1 * r  # increases each round
            noise = np.array(
                [
                    _deterministic_hash(f"{expert}|{question}|r{r + 1}")
                    for expert in experts
                ]
            )
            # Move toward median with some expert-specific noise
            estimates = (
                estimates * (1.0 - revision_strength)
                + median * revision_strength
                + (noise - 0.5) * iqr * 0.1
            )
            estimates = np.clip(estimates, 0.0, 1.0)

    # Final consensus: the expert whose final estimate is closest to median
    final_median = float(np.median(estimates))
    distances = np.abs(estimates - final_median)
    consensus_idx = int(np.argmin(distances))

    # Convergence score: ratio of IQR reduction
    if len(iqr_history) >= 2 and iqr_history[0] > 1e-12:
        convergence_score = float(
            1.0 - iqr_history[-1] / iqr_history[0]
        )
    else:
        convergence_score = 1.0
    convergence_score = float(np.clip(convergence_score, 0.0, 1.0))

    # Remaining disagreements: experts far from median
    std = float(np.std(estimates))
    threshold = final_median + 1.5 * std
    remaining_disagreements = [
        f"{experts[i]} (estimate={estimates[i]:.4f})"
        for i in range(n_experts)
        if abs(estimates[i] - final_median) > max(std, 0.01)
    ]

    # Expert confidence: inversely proportional to distance from median
    max_dist = float(np.max(distances)) if np.max(distances) > 1e-12 else 1.0
    expert_confidence = {
        experts[i]: float(1.0 - distances[i] / max_dist)
        for i in range(n_experts)
    }

    return DelphiResult(
        question=question,
        final_consensus=(
            f"Consensus estimate: {final_median:.4f} "
            f"(closest expert: {experts[consensus_idx]})"
        ),
        round_summaries=round_summaries,
        convergence_score=convergence_score,
        remaining_disagreements=remaining_disagreements,
        expert_confidence=expert_confidence,
    )


def superforecasting(
    question: str,
    forecasters: List[str],
    deadline: str,
) -> Forecast:
    """Aggregate forecasts using superforecasting methodology.

    Weights each forecaster by simulated past calibration (Brier score
    derived from a deterministic hash).  Applies extremising to push the
    aggregate away from 50 % by a factor proportional to forecaster
    diversity.  Computes confidence intervals via bootstrap.

    Parameters
    ----------
    question : str
        The question being forecast.
    forecasters : List[str]
        List of forecaster identifiers.
    deadline : str
        Forecast deadline (used in hash for reproducibility).

    Returns
    -------
    Forecast
        Aggregated forecast with confidence interval and calibration.
    """
    n = len(forecasters)
    if n == 0:
        return Forecast(
            question=question,
            point_estimate=0.5,
            confidence_interval=(0.5, 0.5),
            calibration_score=0.0,
            forecaster_weights={},
            reasoning_summary="No forecasters provided.",
        )

    # Simulate each forecaster's probability estimate and calibration
    raw_estimates = np.array(
        [
            _deterministic_hash(f"{f}|{question}|{deadline}")
            for f in forecasters
        ]
    )

    # Simulate past calibration (Brier scores; lower = better)
    n_past = 20
    brier_scores = np.empty(n)
    for i, f in enumerate(forecasters):
        preds = _deterministic_hash_array(f"{f}|past_preds", n_past)
        outcomes = (
            _deterministic_hash_array(f"{f}|past_outcomes", n_past) > 0.5
        ).astype(float)
        brier_scores[i] = _calibration_score(preds, outcomes)

    # Weights: inverse Brier score (better calibrated = higher weight)
    inv_brier = 1.0 / (brier_scores + 1e-6)
    weights = inv_brier / inv_brier.sum()

    forecaster_weights = {
        forecasters[i]: float(weights[i]) for i in range(n)
    }

    # Weighted aggregate
    weighted_estimate = float(np.sum(weights * raw_estimates))

    # Extremising factor based on diversity of estimates
    diversity = float(np.std(raw_estimates))
    # More diverse panel → more extremising (range ~1.0 to 2.5)
    extremise_factor = 1.0 + 3.0 * diversity
    point_estimate = _extremize(weighted_estimate, extremise_factor)

    # Bootstrap confidence interval
    ci = _bootstrap_confidence(raw_estimates, confidence=0.9)

    # Overall calibration score of the panel
    panel_calibration = float(np.mean(brier_scores))

    # Reasoning summary from top forecasters
    top_k = min(3, n)
    top_indices = np.argsort(weights)[-top_k:][::-1]
    top_names = [forecasters[i] for i in top_indices]
    top_estimates = [f"{raw_estimates[i]:.3f}" for i in top_indices]
    reasoning_summary = (
        f"Top forecasters: {', '.join(top_names)} "
        f"(estimates: {', '.join(top_estimates)}). "
        f"Panel diversity: {diversity:.3f}, "
        f"extremising factor: {extremise_factor:.2f}. "
        f"Deadline: {deadline}."
    )

    return Forecast(
        question=question,
        point_estimate=point_estimate,
        confidence_interval=ci,
        calibration_score=panel_calibration,
        forecaster_weights=forecaster_weights,
        reasoning_summary=reasoning_summary,
    )


def wisdom_of_crowds(
    question: str,
    responses: List[str],
) -> CrowdWisdom:
    """Aggregate crowd responses using embedding-based analysis.

    Embeds all responses, finds the one closest to the centroid, and
    detects outlier insights that may contain unique value.  Estimates
    accuracy using a Condorcet-jury-theorem-inspired formula that
    accounts for crowd size and response diversity.

    Parameters
    ----------
    question : str
        The question posed to the crowd.
    responses : List[str]
        List of crowd response strings.

    Returns
    -------
    CrowdWisdom
        Aggregated result with diversity score and outlier insights.
    """
    crowd_size = len(responses)
    if crowd_size == 0:
        return CrowdWisdom(
            question=question,
            aggregated_answer="",
            crowd_size=0,
            diversity_score=0.0,
            accuracy_estimate=0.0,
            outlier_insights=[],
        )

    # Embed all responses
    embeddings = embed_texts(responses)  # (n, d)

    # Diversity score via pairwise cosine diversity
    diversity_score = float(cosine_diversity(embeddings))

    # Centroid similarity
    centroid_sims = _centroid_distances(embeddings)
    best_idx = int(np.argmax(centroid_sims))
    aggregated_answer = responses[best_idx]

    # Outlier detection: responses whose similarity to centroid is
    # more than 1.5 standard deviations below the mean
    mean_sim = float(np.mean(centroid_sims))
    std_sim = float(np.std(centroid_sims))
    outlier_threshold = mean_sim - 1.5 * std_sim
    outlier_insights = [
        responses[i]
        for i in range(crowd_size)
        if centroid_sims[i] < outlier_threshold
    ]

    # Accuracy estimate inspired by Condorcet jury theorem:
    # P(correct) = 1 - exp(-k * n * p_eff) where p_eff accounts for
    # diversity (diverse crowds are more informative)
    p_individual = 0.55 + 0.1 * diversity_score  # base individual accuracy
    k = 0.1  # scaling constant
    accuracy_estimate = float(
        1.0 - math.exp(-k * crowd_size * p_individual)
    )
    accuracy_estimate = float(np.clip(accuracy_estimate, 0.0, 1.0))

    return CrowdWisdom(
        question=question,
        aggregated_answer=aggregated_answer,
        crowd_size=crowd_size,
        diversity_score=diversity_score,
        accuracy_estimate=accuracy_estimate,
        outlier_insights=outlier_insights,
    )
