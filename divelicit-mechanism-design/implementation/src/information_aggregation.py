"""Information aggregation mechanisms for eliciting and combining beliefs.

Implements Bayesian Truth Serum (Prelec 2004), peer prediction, the
surprisingly-popular algorithm, expert-opinion pooling, calibrated
forecast aggregation, and superforecasting-style extremization.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Response:
    """A single respondent's answer together with a prediction of the
    distribution of answers across the population."""

    respondent_id: str
    answer: str
    prediction: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExpertOpinion:
    """An expert's point estimate with optional confidence interval."""

    expert_id: str
    estimate: float
    confidence_interval: Optional[Tuple[float, float]] = None
    expertise_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Forecast:
    """A single forecaster's probability estimate for a binary event."""

    forecaster_id: str
    probability: float
    past_forecasts: Optional[List[float]] = None
    past_outcomes: Optional[List[int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BTSResult:
    """Result returned by :func:`bayesian_truth_serum`."""

    scores: Dict[str, float]
    estimated_truth: str
    confidence: float
    high_information_respondents: List[str]
    answer_frequencies: Dict[str, float]
    information_scores: Dict[str, float]
    prediction_scores: Dict[str, float]
    raw_responses: List[Response]


@dataclass
class PeerPredResult:
    """Result returned by :func:`peer_prediction`."""

    scores: Dict[str, float]
    adjusted_payments: Dict[str, float]
    mechanism_properties: Dict[str, Any]
    agreement_matrix: Dict[str, Dict[str, float]]
    reference_scores: Dict[str, float]


@dataclass
class SPResult:
    """Result returned by :func:`surprisingly_popular`."""

    surprisingly_popular_answer: str
    votes: Dict[str, float]
    sp_scores: Dict[str, float]
    confidence: float
    actual_frequencies: Dict[str, float]
    mean_predicted_frequencies: Dict[str, float]
    frequency_gaps: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]


@dataclass
class AggregateResult:
    """Result returned by :func:`wisdom_of_experts`."""

    weighted_mean: float
    trimmed_mean: float
    median: float
    linear_pool: float
    confidence_interval: Tuple[float, float]
    disagreement: float
    outlier_ids: List[str]
    weights_used: Dict[str, float]


@dataclass
class CalibratedForecast:
    """Result returned by :func:`calibrated_aggregation`."""

    aggregated_probability: float
    individual_calibrated: Dict[str, float]
    calibration_scores: Dict[str, Dict[str, float]]
    aggregate_brier: float
    aggregate_log_loss: float
    weights_used: Dict[str, float]
    recalibrated: bool


# ---------------------------------------------------------------------------
# Helper / private functions
# ---------------------------------------------------------------------------

def _information_score(
    answer: str,
    prediction: Dict[str, float],
    all_responses: List[Response],
) -> Tuple[float, float, float]:
    """Compute the BTS information score for a single respondent.

    Following Prelec (2004), the information score measures how much more
    common a respondent's answer is *among those who predicted similarly*
    compared to the overall population.

    Returns
    -------
    information_score : float
        log(geometric-mean frequency among same-prediction peers)
        − log(overall frequency of own answer).
    prediction_score : float
        Measures the accuracy of the respondent's prediction of the
        answer distribution (negative KL-divergence direction).
    total_score : float
        ``information_score + prediction_score``.
    """
    n = len(all_responses)
    if n == 0:
        return 0.0, 0.0, 0.0

    # --- overall answer frequencies ---
    answer_counts: Counter[str] = Counter(r.answer for r in all_responses)
    all_answers = sorted(answer_counts.keys())
    overall_freq: Dict[str, float] = {
        a: answer_counts[a] / n for a in all_answers
    }

    # --- geometric mean frequency of *own answer* among peers who gave the
    #     same answer (Prelec's "surprisingly common" measure) ---
    # Group respondents by answer
    same_answer_peers = [r for r in all_responses if r.answer == answer]
    if not same_answer_peers:
        return 0.0, 0.0, 0.0

    # For each peer who gave the same answer, look at their prediction for
    # *this* answer.  The geometric mean of those predictions is the
    # "predicted commonality" within the endorsing group.
    log_preds = []
    for peer in same_answer_peers:
        pred_val = peer.prediction.get(answer, 1e-10)
        pred_val = max(pred_val, 1e-10)
        log_preds.append(math.log(pred_val))

    geom_mean_peer = math.exp(sum(log_preds) / len(log_preds))

    # Overall geometric mean prediction for this answer across *all*
    # respondents.
    log_preds_all = []
    for r in all_responses:
        pred_val = r.prediction.get(answer, 1e-10)
        pred_val = max(pred_val, 1e-10)
        log_preds_all.append(math.log(pred_val))

    geom_mean_all = math.exp(sum(log_preds_all) / len(log_preds_all))

    # Information score: how much more common own answer is than the
    # population predicted.
    info_score = math.log(max(overall_freq.get(answer, 1e-10), 1e-10)) - math.log(
        max(geom_mean_all, 1e-10)
    )

    # --- prediction score (quadratic scoring rule on predicted distribution) ---
    pred_score = 0.0
    for a in all_answers:
        p_a = prediction.get(a, 0.0)
        f_a = overall_freq.get(a, 0.0)
        if f_a > 0 and p_a > 1e-10:
            pred_score += f_a * math.log(max(p_a, 1e-10))
    # Normalise by subtracting entropy so that a perfect predictor scores 0
    entropy = 0.0
    for a in all_answers:
        f_a = overall_freq.get(a, 0.0)
        if f_a > 0:
            entropy += f_a * math.log(f_a)
    pred_score = pred_score - entropy  # higher = better prediction

    total = info_score + pred_score
    return info_score, pred_score, total


def _log_linear_pool(
    probabilities: Sequence[float],
    weights: Sequence[float],
) -> float:
    """Weighted geometric mean of probabilities, renormalised.

    Computes  exp(Σ w_i log p_i) / [exp(Σ w_i log p_i) + exp(Σ w_i log(1-p_i))]
    which is the log-linear (logarithmic) opinion pool for binary events.

    Parameters
    ----------
    probabilities : sequence of float
        Each value in (0, 1).
    weights : sequence of float
        Non-negative weights (need not sum to 1; will be normalised).

    Returns
    -------
    float
        Aggregated probability in (0, 1).
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)

    # Normalise weights
    w_sum = w.sum()
    if w_sum <= 0:
        w = np.ones_like(w) / len(w)
    else:
        w = w / w_sum

    # Clamp to avoid log(0)
    eps = 1e-12
    probs = np.clip(probs, eps, 1.0 - eps)

    log_p = np.sum(w * np.log(probs))
    log_q = np.sum(w * np.log(1.0 - probs))

    # Numerically stable softmax-style normalisation
    max_val = max(log_p, log_q)
    p_agg = np.exp(log_p - max_val) / (np.exp(log_p - max_val) + np.exp(log_q - max_val))
    return float(np.clip(p_agg, eps, 1.0 - eps))


def _brier_score(
    forecasts: Sequence[float],
    outcomes: Sequence[int],
) -> float:
    """Mean Brier score  (1/N) Σ (f_i − o_i)².

    Lower is better.  Perfect calibration on a large sample → ~variance.
    """
    f = np.asarray(forecasts, dtype=np.float64)
    o = np.asarray(outcomes, dtype=np.float64)
    if len(f) == 0:
        return 0.0
    return float(np.mean((f - o) ** 2))


def _calibration_error(
    forecasts: Sequence[float],
    outcomes: Sequence[int],
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE).

    Bins forecasts into *n_bins* equal-width bins and computes the
    weighted-average absolute difference between predicted probability
    and observed frequency.
    """
    f = np.asarray(forecasts, dtype=np.float64)
    o = np.asarray(outcomes, dtype=np.float64)
    n = len(f)
    if n == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        if b == n_bins - 1:
            mask = (f >= lo) & (f <= hi)
        else:
            mask = (f >= lo) & (f < hi)
        count = int(mask.sum())
        if count == 0:
            continue
        avg_pred = float(f[mask].mean())
        avg_outcome = float(o[mask].mean())
        ece += (count / n) * abs(avg_pred - avg_outcome)
    return float(ece)


def _isotonic_recalibrate(
    forecasts: Sequence[float],
    outcomes: Sequence[int],
) -> np.ndarray:
    """Pool-adjacent-violators (isotonic regression) recalibration.

    Given paired (forecast, outcome) data, fits a monotone non-decreasing
    step function mapping raw forecasts → calibrated probabilities using
    the pool-adjacent-violators algorithm.

    Returns the recalibrated forecasts (same length as input).
    """
    f = np.asarray(forecasts, dtype=np.float64)
    o = np.asarray(outcomes, dtype=np.float64)
    n = len(f)
    if n == 0:
        return np.array([], dtype=np.float64)

    # Sort by forecast value
    order = np.argsort(f)
    f_sorted = f[order]
    o_sorted = o[order]

    # Pool-adjacent-violators
    blocks: List[List[int]] = [[i] for i in range(n)]
    values: List[float] = [float(o_sorted[i]) for i in range(n)]

    i = 0
    while i < len(blocks) - 1:
        if values[i] > values[i + 1]:
            # Merge block i and i+1
            merged_indices = blocks[i] + blocks[i + 1]
            merged_value = sum(o_sorted[j] for j in merged_indices) / len(merged_indices)
            blocks[i] = merged_indices
            values[i] = merged_value
            del blocks[i + 1]
            del values[i + 1]
            # Step back to check previous block
            if i > 0:
                i -= 1
        else:
            i += 1

    # Assign calibrated values
    calibrated = np.empty(n, dtype=np.float64)
    for block_indices, val in zip(blocks, values):
        for idx in block_indices:
            calibrated[idx] = val

    # Un-sort back to original order
    result = np.empty(n, dtype=np.float64)
    for out_pos, orig_pos in enumerate(order):
        result[orig_pos] = calibrated[out_pos]

    return result


def _apply_isotonic_model(
    new_forecasts: Sequence[float],
    train_forecasts: Sequence[float],
    train_outcomes: Sequence[int],
) -> np.ndarray:
    """Apply a learned isotonic calibration model to new forecasts.

    Fits isotonic regression on (train_forecasts, train_outcomes) to
    learn a step-function mapping, then applies it to *new_forecasts*
    via interpolation.
    """
    tf = np.asarray(train_forecasts, dtype=np.float64)
    to = np.asarray(train_outcomes, dtype=np.float64)
    nf = np.asarray(new_forecasts, dtype=np.float64)

    if len(tf) == 0:
        return nf.copy()

    # Build isotonic step function from training data
    order = np.argsort(tf)
    tf_sorted = tf[order]
    to_sorted = to[order]

    # Pool-adjacent-violators on training data
    blocks: List[List[int]] = [[i] for i in range(len(tf_sorted))]
    values: List[float] = [float(to_sorted[i]) for i in range(len(tf_sorted))]

    i = 0
    while i < len(blocks) - 1:
        if values[i] > values[i + 1]:
            merged = blocks[i] + blocks[i + 1]
            merged_val = sum(to_sorted[j] for j in merged) / len(merged)
            blocks[i] = merged
            values[i] = merged_val
            del blocks[i + 1]
            del values[i + 1]
            if i > 0:
                i -= 1
        else:
            i += 1

    # Build (x, y) pairs for the step function (use block midpoints)
    xs: List[float] = []
    ys: List[float] = []
    for block_indices, val in zip(blocks, values):
        block_xs = [tf_sorted[j] for j in block_indices]
        xs.append(float(np.mean(block_xs)))
        ys.append(val)

    xs_arr = np.array(xs)
    ys_arr = np.array(ys)

    # Interpolate new forecasts
    calibrated = np.interp(nf, xs_arr, ys_arr)
    return np.clip(calibrated, 1e-4, 1.0 - 1e-4)


# ---------------------------------------------------------------------------
# Main mechanism functions
# ---------------------------------------------------------------------------

def bayesian_truth_serum(
    responses: List[Response],
    prior: Optional[Dict[str, float]] = None,
) -> BTSResult:
    """Bayesian Truth Serum (Prelec 2004).

    Each response contains an answer and a prediction of the overall
    distribution of answers.  The mechanism rewards respondents whose
    answers are *more common than collectively predicted* — the key
    insight being that truthful respondents' answers will be
    "surprisingly common" relative to the average prediction.

    Parameters
    ----------
    responses : list of Response
        Each respondent's answer and probabilistic prediction.
    prior : dict mapping answer → probability, optional
        Prior distribution over answers.  If ``None`` the empirical
        distribution is used as a baseline.

    Returns
    -------
    BTSResult
    """
    n = len(responses)
    if n == 0:
        return BTSResult(
            scores={},
            estimated_truth="",
            confidence=0.0,
            high_information_respondents=[],
            answer_frequencies={},
            information_scores={},
            prediction_scores={},
            raw_responses=[],
        )

    # Empirical answer frequencies
    answer_counts: Counter[str] = Counter(r.answer for r in responses)
    all_answers = sorted(answer_counts.keys())
    freq: Dict[str, float] = {a: answer_counts[a] / n for a in all_answers}

    if prior is None:
        prior = freq.copy()

    # Score every respondent
    info_scores: Dict[str, float] = {}
    pred_scores: Dict[str, float] = {}
    total_scores: Dict[str, float] = {}

    for r in responses:
        i_score, p_score, t_score = _information_score(
            r.answer, r.prediction, responses
        )
        info_scores[r.respondent_id] = i_score
        pred_scores[r.respondent_id] = p_score
        total_scores[r.respondent_id] = t_score

    # Identify high-information respondents (above-median total score)
    if total_scores:
        median_score = float(np.median(list(total_scores.values())))
        high_info = [
            rid for rid, s in total_scores.items() if s >= median_score
        ]
    else:
        high_info = []

    # Estimated truth: the answer whose actual frequency most exceeds the
    # geometric-mean prediction (the "surprisingly popular" signal within BTS).
    sp_gaps: Dict[str, float] = {}
    for a in all_answers:
        log_preds = []
        for r in responses:
            p = r.prediction.get(a, 1e-10)
            log_preds.append(math.log(max(p, 1e-10)))
        geom_mean_pred = math.exp(sum(log_preds) / len(log_preds))
        sp_gaps[a] = freq[a] - geom_mean_pred

    estimated_truth = max(sp_gaps, key=sp_gaps.get)  # type: ignore[arg-type]

    # Confidence: normalised gap for winning answer (sigmoid-transformed)
    max_gap = sp_gaps[estimated_truth]
    confidence = 1.0 / (1.0 + math.exp(-10.0 * max_gap))

    return BTSResult(
        scores=total_scores,
        estimated_truth=estimated_truth,
        confidence=confidence,
        high_information_respondents=high_info,
        answer_frequencies=freq,
        information_scores=info_scores,
        prediction_scores=pred_scores,
        raw_responses=responses,
    )


def peer_prediction(
    responses: List[Response],
    reference_fn: Optional[Callable[[Response, Response], float]] = None,
) -> PeerPredResult:
    """Peer-prediction mechanism (Miller, Resnick, Zeckhauser 2005).

    Each respondent is scored based on how well their response predicts a
    randomly-matched peer's response.  Under mild conditions, truthful
    reporting is a strict Bayesian-Nash equilibrium.

    When *reference_fn* is ``None`` the default 1/prior output-agreement
    method is used: a respondent receives a payment of ``1 / prior(peer_answer)``
    whenever their answer matches the peer's answer.

    Parameters
    ----------
    responses : list of Response
    reference_fn : callable, optional
        ``(focal, peer) → score`` custom scoring function.

    Returns
    -------
    PeerPredResult
    """
    n = len(responses)
    empty_result = PeerPredResult(
        scores={},
        adjusted_payments={},
        mechanism_properties={},
        agreement_matrix={},
        reference_scores={},
    )
    if n < 2:
        return empty_result

    # Empirical prior
    answer_counts: Counter[str] = Counter(r.answer for r in responses)
    all_answers = sorted(answer_counts.keys())
    prior: Dict[str, float] = {a: answer_counts[a] / n for a in all_answers}

    # Default reference function: 1/prior output agreement
    if reference_fn is None:
        def _default_ref(focal: Response, peer: Response) -> float:
            if focal.answer == peer.answer:
                return 1.0 / max(prior[peer.answer], 1e-10)
            return 0.0
        reference_fn = _default_ref

    # Build agreement matrix and raw scores via leave-one-out matching
    agreement_matrix: Dict[str, Dict[str, float]] = {}
    raw_scores: Dict[str, List[float]] = defaultdict(list)

    for i, focal in enumerate(responses):
        agreement_matrix[focal.respondent_id] = {}
        for j, peer in enumerate(responses):
            if i == j:
                continue
            score = reference_fn(focal, peer)
            raw_scores[focal.respondent_id].append(score)
            agreement_matrix[focal.respondent_id][peer.respondent_id] = score

    # Average score for each respondent
    scores: Dict[str, float] = {
        rid: float(np.mean(vals)) if vals else 0.0
        for rid, vals in raw_scores.items()
    }

    # Adjusted payments: normalise so that mean payment = 0 (budget-balanced).
    mean_score = float(np.mean(list(scores.values()))) if scores else 0.0
    adjusted: Dict[str, float] = {
        rid: s - mean_score for rid, s in scores.items()
    }

    # Reference scores (expected score under truthful reporting)
    ref_scores: Dict[str, float] = {}
    for r in responses:
        expected = 0.0
        for a in all_answers:
            p_a = prior.get(a, 0.0)
            if r.answer == a:
                expected += p_a * (1.0 / max(p_a, 1e-10))
            # else contribution is 0 for default mechanism
        ref_scores[r.respondent_id] = expected

    # Mechanism properties
    properties: Dict[str, Any] = {
        "num_respondents": n,
        "num_distinct_answers": len(all_answers),
        "budget_balanced": True,
        "truthful_equilibrium": True,
        "mean_raw_score": mean_score,
        "score_variance": float(np.var(list(scores.values()))) if scores else 0.0,
        "prior_used": prior,
    }

    return PeerPredResult(
        scores=scores,
        adjusted_payments=adjusted,
        mechanism_properties=properties,
        agreement_matrix=agreement_matrix,
        reference_scores=ref_scores,
    )


def surprisingly_popular(
    responses: List[Response],
) -> SPResult:
    """Surprisingly-popular algorithm (Prelec, Seung, McCoy 2017).

    Each respondent provides an answer *and* a prediction of the
    distribution of answers.  The SP answer is the one whose actual
    frequency most exceeds the *average* predicted frequency — i.e. the
    answer that is "surprisingly popular".

    Confidence intervals are bootstrap-based (percentile method, 1000
    resamples) on the frequency gap.

    Parameters
    ----------
    responses : list of Response

    Returns
    -------
    SPResult
    """
    n = len(responses)
    empty = SPResult(
        surprisingly_popular_answer="",
        votes={},
        sp_scores={},
        confidence=0.0,
        actual_frequencies={},
        mean_predicted_frequencies={},
        frequency_gaps={},
        confidence_intervals={},
    )
    if n == 0:
        return empty

    # Collect all answer labels that appear either as an answer or in a
    # prediction dictionary.
    all_answers_set: set[str] = set()
    for r in responses:
        all_answers_set.add(r.answer)
        all_answers_set.update(r.prediction.keys())
    all_answers = sorted(all_answers_set)

    # Actual frequencies
    answer_counts: Counter[str] = Counter(r.answer for r in responses)
    actual_freq: Dict[str, float] = {
        a: answer_counts.get(a, 0) / n for a in all_answers
    }

    # Mean predicted frequency
    mean_pred: Dict[str, float] = {}
    for a in all_answers:
        preds = [r.prediction.get(a, 0.0) for r in responses]
        mean_pred[a] = float(np.mean(preds))

    # Frequency gaps
    gaps: Dict[str, float] = {
        a: actual_freq[a] - mean_pred[a] for a in all_answers
    }

    # SP answer
    sp_answer = max(gaps, key=gaps.get)  # type: ignore[arg-type]

    # SP scores: per-answer, the gap normalised so that the maximum gap = 1
    max_abs_gap = max(abs(g) for g in gaps.values()) if gaps else 1.0
    if max_abs_gap < 1e-12:
        max_abs_gap = 1.0
    sp_scores: Dict[str, float] = {a: gaps[a] / max_abs_gap for a in all_answers}

    # Votes: the number of "votes" each answer receives in a simple tally
    votes: Dict[str, float] = {a: float(answer_counts.get(a, 0)) for a in all_answers}

    # Bootstrap confidence intervals on the gap for each answer
    rng = np.random.default_rng(seed=42)
    n_boot = 1000
    boot_gaps: Dict[str, List[float]] = {a: [] for a in all_answers}

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_responses = [responses[int(i)] for i in idx]
        b_counts: Counter[str] = Counter(r.answer for r in boot_responses)
        for a in all_answers:
            b_freq = b_counts.get(a, 0) / n
            b_preds = [boot_responses[int(j)].prediction.get(a, 0.0) for j in range(n)]
            b_mean_pred = float(np.mean(b_preds))
            boot_gaps[a].append(b_freq - b_mean_pred)

    ci: Dict[str, Tuple[float, float]] = {}
    for a in all_answers:
        arr = np.array(boot_gaps[a])
        lo = float(np.percentile(arr, 2.5))
        hi = float(np.percentile(arr, 97.5))
        ci[a] = (lo, hi)

    # Confidence: 1 − p-value-like measure.  Fraction of bootstrap samples
    # where the SP answer has a positive gap.
    sp_boot = np.array(boot_gaps[sp_answer])
    frac_positive = float(np.mean(sp_boot > 0))
    confidence = frac_positive

    return SPResult(
        surprisingly_popular_answer=sp_answer,
        votes=votes,
        sp_scores=sp_scores,
        confidence=confidence,
        actual_frequencies=actual_freq,
        mean_predicted_frequencies=mean_pred,
        frequency_gaps=gaps,
        confidence_intervals=ci,
    )


def wisdom_of_experts(
    expert_opinions: List[ExpertOpinion],
    expertise_weights: Optional[Dict[str, float]] = None,
) -> AggregateResult:
    """Weighted aggregation of expert point estimates.

    Implements four aggregation methods:

    1. **Weighted mean** using *expertise_weights* (equal if ``None``).
    2. **Trimmed mean** (symmetric 10 % trim).
    3. **Median**.
    4. **Linear opinion pool** (weighted average, same as weighted mean
       for point estimates; included for API symmetry with probabilistic
       settings).

    Outliers are detected via the 1.5·IQR rule.  Disagreement is
    measured as the coefficient of variation of the estimates.

    Parameters
    ----------
    expert_opinions : list of ExpertOpinion
    expertise_weights : dict mapping expert_id → weight, optional

    Returns
    -------
    AggregateResult
    """
    n = len(expert_opinions)
    default = AggregateResult(
        weighted_mean=0.0,
        trimmed_mean=0.0,
        median=0.0,
        linear_pool=0.0,
        confidence_interval=(0.0, 0.0),
        disagreement=0.0,
        outlier_ids=[],
        weights_used={},
    )
    if n == 0:
        return default

    # Build weight vector
    if expertise_weights is None:
        weights = {e.expert_id: 1.0 / n for e in expert_opinions}
    else:
        total_w = sum(expertise_weights.get(e.expert_id, 1.0) for e in expert_opinions)
        if total_w <= 0:
            total_w = 1.0
        weights = {
            e.expert_id: expertise_weights.get(e.expert_id, 1.0) / total_w
            for e in expert_opinions
        }

    estimates = np.array([e.estimate for e in expert_opinions], dtype=np.float64)
    w_arr = np.array([weights[e.expert_id] for e in expert_opinions], dtype=np.float64)

    # Weighted mean
    weighted_mean = float(np.average(estimates, weights=w_arr))

    # Linear pool (same as weighted mean for point estimates)
    linear_pool = weighted_mean

    # Trimmed mean (10 % each side)
    sorted_est = np.sort(estimates)
    trim_k = max(1, int(math.floor(0.1 * n)))
    if n - 2 * trim_k > 0:
        trimmed = sorted_est[trim_k: n - trim_k]
        trimmed_mean = float(np.mean(trimmed))
    else:
        trimmed_mean = float(np.mean(estimates))

    # Median
    median_val = float(np.median(estimates))

    # Outliers: 1.5 × IQR rule
    q1 = float(np.percentile(estimates, 25))
    q3 = float(np.percentile(estimates, 75))
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    outlier_ids = [
        e.expert_id
        for e in expert_opinions
        if e.estimate < lower_fence or e.estimate > upper_fence
    ]

    # Disagreement: coefficient of variation (std / |mean|)
    std = float(np.std(estimates, ddof=1)) if n > 1 else 0.0
    disagreement = std / abs(weighted_mean) if abs(weighted_mean) > 1e-12 else std

    # Confidence interval: if experts supply CIs, combine them; otherwise
    # use ±1.96 × weighted standard deviation.
    has_ci = all(e.confidence_interval is not None for e in expert_opinions)
    if has_ci and n > 0:
        lo_vals = np.array([e.confidence_interval[0] for e in expert_opinions])  # type: ignore[index]
        hi_vals = np.array([e.confidence_interval[1] for e in expert_opinions])  # type: ignore[index]
        ci_lo = float(np.average(lo_vals, weights=w_arr))
        ci_hi = float(np.average(hi_vals, weights=w_arr))
    else:
        w_std = math.sqrt(
            float(np.average((estimates - weighted_mean) ** 2, weights=w_arr))
        )
        ci_lo = weighted_mean - 1.96 * w_std
        ci_hi = weighted_mean + 1.96 * w_std

    return AggregateResult(
        weighted_mean=weighted_mean,
        trimmed_mean=trimmed_mean,
        median=median_val,
        linear_pool=linear_pool,
        confidence_interval=(ci_lo, ci_hi),
        disagreement=disagreement,
        outlier_ids=outlier_ids,
        weights_used=weights,
    )


def calibrated_aggregation(
    forecasts: List[Forecast],
    calibration_data: Optional[Dict[str, Tuple[List[float], List[int]]]] = None,
) -> CalibratedForecast:
    """Aggregate probability forecasts with calibration adjustment.

    Workflow:

    1. If *calibration_data* is provided (mapping ``forecaster_id`` →
       ``(past_forecasts, past_outcomes)``), each forecaster is
       recalibrated via isotonic regression before aggregation.
    2. Aggregation uses the **log-linear (logarithmic) opinion pool**
       weighted by the *inverse* of each forecaster's calibration error
       (ECE).  Forecasters with lower ECE receive more weight.
    3. Individual Brier and log-loss scores are computed for forecasters
       that have calibration data.

    Parameters
    ----------
    forecasts : list of Forecast
    calibration_data : dict, optional
        ``{forecaster_id: (past_forecasts, past_outcomes)}``.

    Returns
    -------
    CalibratedForecast
    """
    n = len(forecasts)
    empty = CalibratedForecast(
        aggregated_probability=0.5,
        individual_calibrated={},
        calibration_scores={},
        aggregate_brier=0.0,
        aggregate_log_loss=0.0,
        weights_used={},
        recalibrated=False,
    )
    if n == 0:
        return empty

    recalibrated = calibration_data is not None and len(calibration_data) > 0

    # --- Step 1: recalibrate individual forecasts if data is available ---
    calibrated_probs: Dict[str, float] = {}
    cal_scores: Dict[str, Dict[str, float]] = {}
    ece_values: Dict[str, float] = {}

    for f in forecasts:
        fid = f.forecaster_id
        raw_prob = f.probability

        # Determine calibration source: explicit parameter or attached history
        past_f: Optional[List[float]] = None
        past_o: Optional[List[int]] = None
        if calibration_data and fid in calibration_data:
            past_f, past_o = calibration_data[fid]
        elif f.past_forecasts is not None and f.past_outcomes is not None:
            past_f = f.past_forecasts
            past_o = f.past_outcomes

        if past_f is not None and past_o is not None and len(past_f) >= 2:
            # Recalibrate current forecast using learned isotonic model
            recal = _apply_isotonic_model([raw_prob], past_f, past_o)
            calibrated_probs[fid] = float(recal[0])

            # Compute calibration quality metrics
            brier = _brier_score(past_f, past_o)
            ece = _calibration_error(past_f, past_o)

            # Log loss
            pf = np.clip(np.asarray(past_f, dtype=np.float64), 1e-10, 1 - 1e-10)
            po = np.asarray(past_o, dtype=np.float64)
            log_loss = -float(np.mean(po * np.log(pf) + (1 - po) * np.log(1 - pf)))

            cal_scores[fid] = {
                "brier": brier,
                "ece": ece,
                "log_loss": log_loss,
                "n_calibration_points": len(past_f),
            }
            ece_values[fid] = ece
        else:
            calibrated_probs[fid] = raw_prob
            cal_scores[fid] = {
                "brier": float("nan"),
                "ece": float("nan"),
                "log_loss": float("nan"),
                "n_calibration_points": 0,
            }
            ece_values[fid] = 0.5  # default moderate ECE when unknown

    # --- Step 2: compute weights (inverse ECE) ---
    weights: Dict[str, float] = {}
    for fid in calibrated_probs:
        ece_val = ece_values.get(fid, 0.5)
        # Inverse ECE weighting with floor to avoid division by zero
        weights[fid] = 1.0 / (ece_val + 0.01)

    total_w = sum(weights.values())
    if total_w > 0:
        weights = {fid: w / total_w for fid, w in weights.items()}
    else:
        weights = {fid: 1.0 / n for fid in calibrated_probs}

    # --- Step 3: log-linear pool ---
    prob_list = [calibrated_probs[f.forecaster_id] for f in forecasts]
    w_list = [weights[f.forecaster_id] for f in forecasts]
    agg_prob = _log_linear_pool(prob_list, w_list)

    # --- aggregate Brier & log-loss (across forecasters that have scores) ---
    valid_briers = [
        v["brier"] for v in cal_scores.values()
        if not math.isnan(v["brier"])
    ]
    agg_brier = float(np.mean(valid_briers)) if valid_briers else 0.0

    valid_ll = [
        v["log_loss"] for v in cal_scores.values()
        if not math.isnan(v["log_loss"])
    ]
    agg_ll = float(np.mean(valid_ll)) if valid_ll else 0.0

    return CalibratedForecast(
        aggregated_probability=agg_prob,
        individual_calibrated=calibrated_probs,
        calibration_scores=cal_scores,
        aggregate_brier=agg_brier,
        aggregate_log_loss=agg_ll,
        weights_used=weights,
        recalibrated=recalibrated,
    )


def extremize_forecasts(
    forecasts: Sequence[float],
    factor: float = 2.5,
) -> List[float]:
    """Super-forecasting extremization (Tetlock & Gardner 2015).

    Algorithm:

    1. Compute the geometric mean of odds across all forecasters.
       ``odds_i = p_i / (1 - p_i)``
       ``geo_mean_odds = exp( (1/N) Σ log(odds_i) )``
    2. Extremize by raising the aggregated odds to *factor*:
       ``extremized_odds = geo_mean_odds ** factor``
    3. Convert back to probability:
       ``extremized_p = extremized_odds / (1 + extremized_odds)``
    4. Clamp the result to [0.01, 0.99].

    The intuition is that independent forecasters who agree on a
    direction should be collectively *more* confident than their average
    suggests, because each has private information.

    Parameters
    ----------
    forecasts : sequence of float
        Individual probability forecasts in (0, 1).
    factor : float
        Extremization exponent. 1.0 = no extremization. Values > 1
        push the aggregate towards 0 or 1.

    Returns
    -------
    list of float
        One extremized probability for each input forecast.  All outputs
        share the *same* extremized value (the aggregate); the list
        length matches the input for downstream convenience.
    """
    eps = 1e-10
    probs = np.asarray(forecasts, dtype=np.float64)
    probs = np.clip(probs, eps, 1.0 - eps)

    if len(probs) == 0:
        return []

    # Step 1: geometric mean of odds
    log_odds = np.log(probs / (1.0 - probs))
    mean_log_odds = float(np.mean(log_odds))

    # Step 2: extremize
    extremized_log_odds = mean_log_odds * factor

    # Step 3: convert back
    # Use sigmoid for numerical stability
    if extremized_log_odds > 20:
        extremized_p = 1.0 / (1.0 + math.exp(-extremized_log_odds))
    elif extremized_log_odds < -20:
        extremized_p = math.exp(extremized_log_odds) / (1.0 + math.exp(extremized_log_odds))
    else:
        extremized_p = 1.0 / (1.0 + math.exp(-extremized_log_odds))

    # Step 4: clamp
    extremized_p = max(0.01, min(0.99, extremized_p))

    return [extremized_p] * len(probs)


# ---------------------------------------------------------------------------
# Convenience / orchestration helpers
# ---------------------------------------------------------------------------

def aggregate_with_all_methods(
    responses: List[Response],
    expert_opinions: Optional[List[ExpertOpinion]] = None,
    forecasts: Optional[List[Forecast]] = None,
    calibration_data: Optional[Dict[str, Tuple[List[float], List[int]]]] = None,
    extremization_factor: float = 2.5,
) -> Dict[str, Any]:
    """Run every aggregation mechanism and return a consolidated dict.

    This is a convenience wrapper that calls :func:`bayesian_truth_serum`,
    :func:`peer_prediction`, :func:`surprisingly_popular`,
    :func:`wisdom_of_experts`, :func:`calibrated_aggregation`, and
    :func:`extremize_forecasts` as applicable, collecting all outputs
    into a single dictionary.

    Parameters
    ----------
    responses : list of Response
        Required for BTS, peer prediction, and SP.
    expert_opinions : list of ExpertOpinion, optional
        Required for wisdom-of-experts aggregation.
    forecasts : list of Forecast, optional
        Required for calibrated aggregation and extremization.
    calibration_data : dict, optional
        Passed to :func:`calibrated_aggregation`.
    extremization_factor : float
        Passed to :func:`extremize_forecasts`.

    Returns
    -------
    dict
        Keys: ``bts``, ``peer_prediction``, ``surprisingly_popular``,
        ``wisdom_of_experts``, ``calibrated``, ``extremized``.
        Values are the corresponding result dataclass instances (or
        ``None`` when the required inputs are missing).
    """
    result: Dict[str, Any] = {}

    # BTS
    if responses:
        result["bts"] = bayesian_truth_serum(responses)
        result["peer_prediction"] = peer_prediction(responses)
        result["surprisingly_popular"] = surprisingly_popular(responses)
    else:
        result["bts"] = None
        result["peer_prediction"] = None
        result["surprisingly_popular"] = None

    # Expert opinions
    if expert_opinions:
        result["wisdom_of_experts"] = wisdom_of_experts(expert_opinions)
    else:
        result["wisdom_of_experts"] = None

    # Forecasts
    if forecasts:
        result["calibrated"] = calibrated_aggregation(
            forecasts, calibration_data=calibration_data
        )
        probs = [f.probability for f in forecasts]
        result["extremized"] = extremize_forecasts(
            probs, factor=extremization_factor
        )
    else:
        result["calibrated"] = None
        result["extremized"] = None

    return result


def compute_mechanism_comparison(
    responses: List[Response],
) -> Dict[str, Any]:
    """Compare BTS, peer-prediction, and SP on the same response set.

    Returns a dictionary with each mechanism's top answer and a
    concordance flag indicating whether all three agree.
    """
    bts_result = bayesian_truth_serum(responses)
    sp_result = surprisingly_popular(responses)
    pp_result = peer_prediction(responses)

    # For peer prediction the "winning answer" is the answer given by
    # the highest-scored respondent.
    if pp_result.scores:
        best_pp_id = max(pp_result.scores, key=pp_result.scores.get)  # type: ignore[arg-type]
        pp_answer = next(
            (r.answer for r in responses if r.respondent_id == best_pp_id),
            "",
        )
    else:
        pp_answer = ""

    answers = {
        "bts": bts_result.estimated_truth,
        "surprisingly_popular": sp_result.surprisingly_popular_answer,
        "peer_prediction": pp_answer,
    }
    concordant = len(set(answers.values())) == 1

    return {
        "answers": answers,
        "concordant": concordant,
        "bts_confidence": bts_result.confidence,
        "sp_confidence": sp_result.confidence,
        "bts_result": bts_result,
        "sp_result": sp_result,
        "pp_result": pp_result,
    }
