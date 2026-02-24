"""Ethical elicitation mechanisms for divergence-aware mechanism design.

Implements manipulation detection, fairness auditing, differential privacy,
informed consent verification, stakeholder analysis, and moral uncertainty
aggregation for ethical preference elicitation.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ManipulationAttempt:
    """A detected manipulation attempt in elicitation responses."""

    agent_id: str
    manipulation_type: str  # strategic_misreporting | collusion | sybil | anchoring
    confidence: float  # 0-1
    evidence: Dict[str, Any] = field(default_factory=dict)
    description: str = ""


@dataclass
class DemographicGroup:
    """A demographic group for fairness auditing."""

    name: str
    member_ids: List[str] = field(default_factory=list)
    selection_rate: float = 0.0
    average_quality: float = 0.0
    quality_scores: List[float] = field(default_factory=list)


@dataclass
class FairnessReport:
    """Report from a fairness audit of a selection mechanism."""

    representation_parity: Dict[str, float] = field(default_factory=dict)
    quality_parity: Dict[str, float] = field(default_factory=dict)
    diversity_parity: Dict[str, float] = field(default_factory=dict)
    individual_fairness_score: float = 0.0
    disparate_impact_ratios: Dict[str, float] = field(default_factory=dict)
    equalized_odds: Dict[str, Dict[str, float]] = field(default_factory=dict)
    overall_fairness_score: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    passes_80_percent_rule: bool = True


@dataclass
class PrivateResult:
    """Result of a differentially private elicitation query."""

    noisy_aggregate: np.ndarray = field(default_factory=lambda: np.array([]))
    true_aggregate: Optional[np.ndarray] = None
    epsilon_spent: float = 0.0
    delta: float = 0.0
    accuracy_estimate: float = 0.0
    noise_scale: float = 0.0
    num_agents: int = 0
    composition_budget_remaining: float = 0.0


@dataclass
class ConsentCriterion:
    """A single criterion in an informed consent check."""

    name: str
    description: str
    score: float = 0.0  # 0-1
    satisfied: bool = False
    evidence: str = ""


@dataclass
class ConsentReport:
    """Report from an informed consent check."""

    criteria: List[ConsentCriterion] = field(default_factory=list)
    overall_score: float = 0.0
    consent_valid: bool = False
    recommendations: List[str] = field(default_factory=list)


@dataclass
class Stakeholder:
    """A stakeholder in a decision."""

    id: str
    name: str
    concerns: List[float] = field(default_factory=list)  # embedding vector
    power: float = 0.0  # 0-1
    interest: float = 0.0  # 0-1
    impact: float = 0.0  # can be negative
    group: str = ""


@dataclass
class StakeholderReport:
    """Report from a stakeholder analysis."""

    stakeholder_impacts: Dict[str, float] = field(default_factory=dict)
    power_interest_grid: Dict[str, List[str]] = field(default_factory=dict)
    underrepresented: List[str] = field(default_factory=list)
    engagement_strategy: Dict[str, str] = field(default_factory=dict)
    total_positive_impact: float = 0.0
    total_negative_impact: float = 0.0


@dataclass
class EthicalView:
    """An ethical framework's view on a decision."""

    framework: str  # e.g. "utilitarian", "deontological", "virtue_ethics"
    recommendation: np.ndarray = field(default_factory=lambda: np.array([]))
    confidence: float = 0.5
    choiceworthiness: float = 0.0  # how choice-worthy the recommended action is


@dataclass
class AggregatedView:
    """Aggregated recommendation across moral uncertainty."""

    recommendation: np.ndarray = field(default_factory=lambda: np.array([]))
    method: str = ""
    framework_contributions: Dict[str, float] = field(default_factory=dict)
    regret_bound: float = 0.0
    parliament_seats: Dict[str, int] = field(default_factory=dict)
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _laplace_noise(scale: float, shape: Tuple[int, ...] | int) -> np.ndarray:
    """Sample Laplace noise calibrated for differential privacy.

    The Laplace mechanism adds noise drawn from Lap(0, scale) where
    scale = sensitivity / epsilon.  This satisfies epsilon-DP because for
    neighbouring datasets D, D' differing in one record the ratio
    Pr[M(D)=t] / Pr[M(D')=t] = exp(-|f(D)-t|/scale) / exp(-|f(D')-t|/scale)
                                <= exp(|f(D)-f(D')|/scale) <= exp(sensitivity/scale) = exp(epsilon).
    """
    return np.random.laplace(loc=0.0, scale=scale, size=shape)


def _disparate_impact_ratio(group_rates: Dict[str, float]) -> Dict[str, float]:
    """Compute disparate impact ratios using the 80% rule.

    The 80% rule (four-fifths rule) states that a selection rate for any
    group should be at least 80% of the highest group's rate.

    Returns a dict mapping each group name to its ratio relative to the
    most-favoured group.  Values below 0.8 indicate a potential violation.
    """
    if not group_rates:
        return {}
    max_rate = max(group_rates.values())
    if max_rate == 0:
        return {g: 1.0 for g in group_rates}
    return {g: rate / max_rate for g, rate in group_rates.items()}


def _cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarities for an (n, d) matrix.

    Returns an (n, n) matrix where entry (i, j) is the cosine similarity
    between embeddings[i] and embeddings[j].
    """
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        return np.array([[]])
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    normed = embeddings / norms
    return normed @ normed.T


def _z_score_outliers(
    values: np.ndarray, threshold: float = 3.0
) -> List[int]:
    """Detect outlier indices using z-score method.

    An observation is flagged if |z| > threshold.  With threshold=3.0 this
    corresponds to roughly the 0.13% tails of a normal distribution.
    """
    if len(values) < 2:
        return []
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    if std < 1e-12:
        return []
    z_scores = np.abs((values - mean) / std)
    return [int(i) for i in np.where(z_scores > threshold)[0]]


def _power_interest_grid(
    stakeholders: List[Stakeholder],
) -> Dict[str, List[str]]:
    """Classify stakeholders into a power-interest grid.

    Quadrants (using 0.5 as the boundary):
      - high_power_high_interest:  Manage closely
      - high_power_low_interest:   Keep satisfied
      - low_power_high_interest:   Keep informed
      - low_power_low_interest:    Monitor
    """
    grid: Dict[str, List[str]] = {
        "high_power_high_interest": [],
        "high_power_low_interest": [],
        "low_power_high_interest": [],
        "low_power_low_interest": [],
    }
    for s in stakeholders:
        if s.power >= 0.5 and s.interest >= 0.5:
            grid["high_power_high_interest"].append(s.id)
        elif s.power >= 0.5 and s.interest < 0.5:
            grid["high_power_low_interest"].append(s.id)
        elif s.power < 0.5 and s.interest >= 0.5:
            grid["low_power_high_interest"].append(s.id)
        else:
            grid["low_power_low_interest"].append(s.id)
    return grid


def _expected_choiceworthiness(
    views: List[EthicalView], weights: np.ndarray
) -> np.ndarray:
    """Maximise Expected Choiceworthiness (MEC) aggregation.

    Each ethical framework f_i assigns a choiceworthiness score c_i to an
    action a.  MEC selects the action maximising sum_i w_i * c_i where w_i
    is the credence (weight) assigned to framework f_i.

    Here, each view provides a recommendation vector (e.g. a probability
    distribution over actions).  We compute the weighted average
    recommendation, weighting by credence * choiceworthiness.
    """
    if not views:
        return np.array([])
    dim = views[0].recommendation.shape[0]
    result = np.zeros(dim, dtype=np.float64)
    total_weight = 0.0
    for v, w in zip(views, weights):
        effective_w = w * v.choiceworthiness
        result += effective_w * v.recommendation
        total_weight += effective_w
    if total_weight > 0:
        result /= total_weight
    return result


def _minimax_regret(views: List[EthicalView]) -> Tuple[np.ndarray, float]:
    """Compute the minimax-regret action across ethical frameworks.

    Regret of action a under framework f = max choiceworthiness under f minus
    choiceworthiness of a under f.  We discretise the action space using the
    set of recommendations provided by the views themselves, then find the
    action (possibly a mixture) that minimises the maximum regret across
    frameworks.

    For simplicity we evaluate each view's recommendation as a candidate action
    and pick the one with the lowest worst-case regret.  If all views provide
    recommendation vectors, we also try the uniform mixture.
    """
    if not views:
        return np.array([]), 0.0

    # Candidate actions: each view's recommendation + uniform mixture
    candidates: List[np.ndarray] = [v.recommendation for v in views]
    dim = views[0].recommendation.shape[0]
    uniform = np.ones(dim, dtype=np.float64) / dim
    candidates.append(uniform)

    # For each framework, the best choiceworthiness is the one from its own
    # recommendation.  Regret of a candidate under framework f is computed as
    # the dot-product loss relative to f's own recommendation.
    best_cw = [v.choiceworthiness for v in views]

    best_action = candidates[0]
    best_max_regret = float("inf")

    for cand in candidates:
        max_regret = 0.0
        for i, v in enumerate(views):
            # choiceworthiness of candidate under framework i is proportional
            # to cosine similarity between candidate and framework's rec
            norm_c = np.linalg.norm(cand)
            norm_r = np.linalg.norm(v.recommendation)
            if norm_c > 0 and norm_r > 0:
                sim = float(np.dot(cand, v.recommendation) / (norm_c * norm_r))
            else:
                sim = 0.0
            cw_candidate = sim * best_cw[i]
            regret = best_cw[i] - cw_candidate
            max_regret = max(max_regret, regret)
        if max_regret < best_max_regret:
            best_max_regret = max_regret
            best_action = cand.copy()

    return best_action, best_max_regret


# ---------------------------------------------------------------------------
# Main functions
# ---------------------------------------------------------------------------

def detect_manipulation(
    responses: List[Dict[str, Any]],
) -> List[ManipulationAttempt]:
    """Detect manipulation attempts in elicitation responses.

    Each response dict should contain at minimum:
        - agent_id: str
        - value: float or array-like  (the reported preference/value)
        - timestamp: float             (epoch seconds)
        - metadata: dict (optional)

    Detection strategies:
      (a) Strategic misreporting — z-score outlier detection on reported values.
      (b) Collusion — high cosine similarity among response vectors from
          distinct agents.
      (c) Sybil attacks — near-identical metadata fingerprints or suspiciously
          close timestamps from "different" agents.
      (d) Anchoring manipulation — early responses that are extreme relative
          to later ones, suggesting intent to bias.

    Returns a list of ManipulationAttempt objects with confidence scores.
    """
    if not responses:
        return []

    attempts: List[ManipulationAttempt] = []

    # --- Extract numeric values ---
    agent_ids = [r["agent_id"] for r in responses]
    raw_values = []
    for r in responses:
        v = r["value"]
        if isinstance(v, (list, np.ndarray)):
            raw_values.append(np.mean(v))
        else:
            raw_values.append(float(v))
    values = np.array(raw_values, dtype=np.float64)
    timestamps = np.array([r.get("timestamp", 0.0) for r in responses], dtype=np.float64)

    # (a) Strategic misreporting — z-score outliers
    outlier_indices = _z_score_outliers(values, threshold=3.0)
    mean_val = float(np.mean(values))
    std_val = float(np.std(values, ddof=1)) if len(values) > 1 else 1.0
    for idx in outlier_indices:
        z = abs((values[idx] - mean_val) / std_val) if std_val > 0 else 0.0
        conf = min(1.0, (z - 3.0) / 3.0 + 0.5)  # scales from 0.5 at z=3 to 1.0 at z=6
        attempts.append(ManipulationAttempt(
            agent_id=agent_ids[idx],
            manipulation_type="strategic_misreporting",
            confidence=float(np.clip(conf, 0.0, 1.0)),
            evidence={"z_score": round(z, 4), "value": float(values[idx]),
                       "mean": round(mean_val, 4), "std": round(std_val, 4)},
            description=(
                f"Agent {agent_ids[idx]} reported value {values[idx]:.4f} "
                f"which is {z:.1f} standard deviations from the mean."
            ),
        ))

    # (b) Collusion — cosine similarity among response vectors
    embeddings = []
    for r in responses:
        v = r["value"]
        if isinstance(v, (list, np.ndarray)):
            embeddings.append(np.asarray(v, dtype=np.float64))
        else:
            embeddings.append(np.array([float(v)], dtype=np.float64))
    max_dim = max(e.shape[0] for e in embeddings)
    padded = np.zeros((len(embeddings), max_dim), dtype=np.float64)
    for i, e in enumerate(embeddings):
        padded[i, : e.shape[0]] = e

    sim_matrix = _cosine_similarity_matrix(padded)
    collusion_threshold = 0.98
    flagged_pairs: set = set()
    for i in range(len(responses)):
        for j in range(i + 1, len(responses)):
            if agent_ids[i] == agent_ids[j]:
                continue
            if sim_matrix[i, j] >= collusion_threshold:
                pair = tuple(sorted([agent_ids[i], agent_ids[j]]))
                if pair not in flagged_pairs:
                    flagged_pairs.add(pair)
                    conf = float(np.clip((sim_matrix[i, j] - 0.95) / 0.05, 0.0, 1.0))
                    attempts.append(ManipulationAttempt(
                        agent_id=f"{pair[0]},{pair[1]}",
                        manipulation_type="collusion",
                        confidence=conf,
                        evidence={"cosine_similarity": round(float(sim_matrix[i, j]), 4),
                                  "agents": list(pair)},
                        description=(
                            f"Agents {pair[0]} and {pair[1]} have suspiciously "
                            f"similar responses (cosine sim = {sim_matrix[i, j]:.4f})."
                        ),
                    ))

    # (c) Sybil attacks — fingerprint + timing analysis
    def _fingerprint(r: Dict[str, Any]) -> str:
        meta = r.get("metadata", {})
        parts = [str(meta.get("ip", "")), str(meta.get("user_agent", "")),
                 str(meta.get("session_id", ""))]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]

    fingerprints: Dict[str, List[int]] = {}
    for idx, r in enumerate(responses):
        fp = _fingerprint(r)
        fingerprints.setdefault(fp, []).append(idx)

    for fp, indices in fingerprints.items():
        if len(indices) < 2:
            continue
        ids_in_group = [agent_ids[i] for i in indices]
        unique_ids = set(ids_in_group)
        if len(unique_ids) > 1:
            conf = min(1.0, len(indices) / 5.0)
            attempts.append(ManipulationAttempt(
                agent_id=",".join(sorted(unique_ids)),
                manipulation_type="sybil",
                confidence=conf,
                evidence={"fingerprint": fp, "agent_ids": sorted(unique_ids),
                          "count": len(indices)},
                description=(
                    f"Agents {sorted(unique_ids)} share the same metadata "
                    f"fingerprint, suggesting a sybil attack."
                ),
            ))

    # Timing-based sybil detection: clusters of responses within a small window
    if len(timestamps) > 1 and np.any(timestamps > 0):
        sorted_idx = np.argsort(timestamps)
        time_diffs = np.diff(timestamps[sorted_idx])
        burst_threshold = np.median(time_diffs) * 0.1 if np.median(time_diffs) > 0 else 0.5
        burst_threshold = max(burst_threshold, 0.01)
        burst_start = 0
        for k in range(len(time_diffs)):
            if time_diffs[k] > burst_threshold:
                burst_indices = sorted_idx[burst_start: k + 1]
                if len(burst_indices) >= 3:
                    burst_agents = list({agent_ids[int(bi)] for bi in burst_indices})
                    if len(burst_agents) > 1:
                        conf = min(1.0, len(burst_indices) / 6.0)
                        attempts.append(ManipulationAttempt(
                            agent_id=",".join(sorted(burst_agents)),
                            manipulation_type="sybil",
                            confidence=conf,
                            evidence={"timing_window": float(time_diffs[k]),
                                      "burst_size": len(burst_indices)},
                            description=(
                                f"{len(burst_indices)} responses arrived in a "
                                f"suspiciously short window ({time_diffs[k]:.3f}s)."
                            ),
                        ))
                burst_start = k + 1

    # (d) Anchoring manipulation — early extreme responses
    if len(values) >= 5:
        early_count = max(2, len(values) // 5)
        order = np.argsort(timestamps) if np.any(timestamps > 0) else np.arange(len(values))
        early_indices = order[:early_count]
        later_indices = order[early_count:]
        later_mean = float(np.mean(values[later_indices]))
        later_std = float(np.std(values[later_indices], ddof=1)) if len(later_indices) > 1 else 1.0
        for idx in early_indices:
            if later_std < 1e-12:
                continue
            deviation = abs(values[int(idx)] - later_mean) / later_std
            if deviation > 2.5:
                conf = float(np.clip((deviation - 2.5) / 2.5, 0.3, 1.0))
                attempts.append(ManipulationAttempt(
                    agent_id=agent_ids[int(idx)],
                    manipulation_type="anchoring",
                    confidence=conf,
                    evidence={"early_value": float(values[int(idx)]),
                              "later_mean": round(later_mean, 4),
                              "deviation_sigma": round(deviation, 4)},
                    description=(
                        f"Early response from {agent_ids[int(idx)]} "
                        f"({values[int(idx)]:.4f}) deviates {deviation:.1f}σ "
                        f"from later consensus, suggesting anchoring intent."
                    ),
                ))

    return attempts


def fairness_audit(
    mechanism: Dict[str, Any],
    demographic_groups: List[DemographicGroup],
) -> FairnessReport:
    """Audit a selection mechanism for fairness across demographic groups.

    Parameters
    ----------
    mechanism : dict
        Must contain:
          - selections: Dict[str, bool]  (agent_id -> selected)
          - scores: Dict[str, float]     (agent_id -> quality score)
          - features: Dict[str, np.ndarray] (agent_id -> feature vector, optional)
    demographic_groups : list of DemographicGroup
        Groups with member_ids populated.

    Checks
    ------
    (a) Representation parity — proportional selection rates.
    (b) Quality parity — average quality consistent across groups.
    (c) Diversity parity — within-group variance consistent.
    (d) Individual fairness — similar feature vectors ⇒ similar outcomes.

    Returns detailed FairnessReport with recommendations.
    """
    selections: Dict[str, bool] = mechanism.get("selections", {})
    scores: Dict[str, float] = mechanism.get("scores", {})
    features: Dict[str, np.ndarray] = mechanism.get("features", {})

    report = FairnessReport()

    # --- (a) Representation parity ---
    group_rates: Dict[str, float] = {}
    for g in demographic_groups:
        if not g.member_ids:
            group_rates[g.name] = 0.0
            continue
        selected_count = sum(1 for m in g.member_ids if selections.get(m, False))
        rate = selected_count / len(g.member_ids)
        group_rates[g.name] = rate
        g.selection_rate = rate
    report.representation_parity = group_rates

    # Disparate impact
    di_ratios = _disparate_impact_ratio(group_rates)
    report.disparate_impact_ratios = di_ratios
    report.passes_80_percent_rule = all(r >= 0.8 for r in di_ratios.values())

    # --- (b) Quality parity ---
    quality_by_group: Dict[str, float] = {}
    for g in demographic_groups:
        member_scores = [scores[m] for m in g.member_ids if m in scores]
        if member_scores:
            avg_q = float(np.mean(member_scores))
            quality_by_group[g.name] = avg_q
            g.average_quality = avg_q
            g.quality_scores = member_scores
        else:
            quality_by_group[g.name] = 0.0
    report.quality_parity = quality_by_group

    # --- (c) Diversity parity (within-group variance of scores) ---
    diversity_by_group: Dict[str, float] = {}
    for g in demographic_groups:
        if len(g.quality_scores) > 1:
            diversity_by_group[g.name] = float(np.std(g.quality_scores, ddof=1))
        else:
            diversity_by_group[g.name] = 0.0
    report.diversity_parity = diversity_by_group

    # --- (d) Individual fairness ---
    # For each pair of agents with features, check that similar features lead
    # to similar selection outcomes (Lipschitz condition).
    if features:
        agent_list = [a for a in features if a in selections]
        if len(agent_list) >= 2:
            feat_matrix = np.array([features[a] for a in agent_list])
            sim_matrix = _cosine_similarity_matrix(feat_matrix)
            violations = 0
            total_pairs = 0
            for i in range(len(agent_list)):
                for j in range(i + 1, len(agent_list)):
                    feat_sim = sim_matrix[i, j]
                    outcome_diff = abs(
                        float(selections[agent_list[i]])
                        - float(selections[agent_list[j]])
                    )
                    total_pairs += 1
                    # Violation: very similar features but different outcomes
                    if feat_sim > 0.9 and outcome_diff > 0.5:
                        violations += 1
            report.individual_fairness_score = 1.0 - (violations / max(total_pairs, 1))
        else:
            report.individual_fairness_score = 1.0
    else:
        report.individual_fairness_score = 1.0

    # --- Equalized odds ---
    # True positive rate and false positive rate by group (using median score
    # as the "ground truth" threshold).
    if scores:
        median_score = float(np.median(list(scores.values())))
        for g in demographic_groups:
            tp, fp, fn, tn = 0, 0, 0, 0
            for m in g.member_ids:
                qualified = scores.get(m, 0.0) >= median_score
                selected = selections.get(m, False)
                if qualified and selected:
                    tp += 1
                elif not qualified and selected:
                    fp += 1
                elif qualified and not selected:
                    fn += 1
                else:
                    tn += 1
            tpr = tp / max(tp + fn, 1)
            fpr = fp / max(fp + tn, 1)
            report.equalized_odds[g.name] = {"tpr": round(tpr, 4), "fpr": round(fpr, 4)}

    # --- Overall score ---
    component_scores = [
        1.0 if report.passes_80_percent_rule else 0.5,
        report.individual_fairness_score,
    ]
    # Quality parity: penalise if max-min quality gap is large
    if quality_by_group:
        q_vals = list(quality_by_group.values())
        q_range = max(q_vals) - min(q_vals) if q_vals else 0.0
        q_max = max(abs(v) for v in q_vals) if q_vals else 1.0
        quality_score = 1.0 - min(q_range / max(q_max, 1e-9), 1.0)
        component_scores.append(quality_score)
    report.overall_fairness_score = float(np.mean(component_scores))

    # --- Recommendations ---
    if not report.passes_80_percent_rule:
        worst_group = min(di_ratios, key=di_ratios.get)  # type: ignore[arg-type]
        report.recommendations.append(
            f"Group '{worst_group}' has a disparate impact ratio of "
            f"{di_ratios[worst_group]:.2f}, violating the 80% rule. "
            f"Consider adjusting selection criteria."
        )
    if report.individual_fairness_score < 0.8:
        report.recommendations.append(
            "Individual fairness score is low. Similar candidates are "
            "receiving different outcomes—review the decision boundary."
        )
    if not report.recommendations:
        report.recommendations.append("No major fairness concerns detected.")

    return report


def privacy_preserving_elicitation(
    query: np.ndarray,
    agents: List[Dict[str, Any]],
    epsilon: float = 1.0,
    total_budget: float = 10.0,
    sensitivity: float = 1.0,
) -> PrivateResult:
    """Differentially private elicitation via the Laplace mechanism.

    Parameters
    ----------
    query : np.ndarray
        The query vector (e.g. which dimensions of preference to elicit).
    agents : list of dict
        Each dict must have ``"value"`` (float or array) — the agent's true
        response.
    epsilon : float
        Per-query privacy parameter.  Smaller ⇒ more privacy, more noise.
    total_budget : float
        Total privacy budget (for sequential composition tracking).
    sensitivity : float
        Global sensitivity of the query function (L1 norm of max change when
        one agent's data changes).

    Privacy guarantee
    -----------------
    Each agent's response is perturbed with Lap(sensitivity/epsilon) noise
    before aggregation.  By the Laplace mechanism this satisfies
    epsilon-differential privacy per agent:

        Pr[M(D) ∈ S] ≤ e^ε · Pr[M(D') ∈ S]

    for neighbouring datasets D, D' differing in one agent's response.

    Sequential composition: k queries each at epsilon cost a total of
    k*epsilon budget.
    """
    n = len(agents)
    if n == 0:
        return PrivateResult(epsilon_spent=0.0, composition_budget_remaining=total_budget)

    # Determine dimensionality
    sample_val = agents[0]["value"]
    if isinstance(sample_val, (list, np.ndarray)):
        dim = len(sample_val)
    else:
        dim = 1

    # Collect true values
    true_values = np.zeros((n, dim), dtype=np.float64)
    for i, a in enumerate(agents):
        v = a["value"]
        if isinstance(v, (list, np.ndarray)):
            true_values[i] = np.asarray(v, dtype=np.float64)
        else:
            true_values[i, 0] = float(v)

    # True aggregate (mean)
    true_agg = np.mean(true_values, axis=0)

    # Each agent adds calibrated Laplace noise
    # scale = sensitivity / epsilon  (per-agent)
    scale = sensitivity / epsilon
    noisy_values = np.zeros_like(true_values)
    for i in range(n):
        noise = _laplace_noise(scale, (dim,))
        noisy_values[i] = true_values[i] + noise

    # Aggregate noisy responses (mean)
    noisy_agg = np.mean(noisy_values, axis=0)

    # Accuracy estimate: expected L2 error of the mean with Laplace noise
    # Var[Lap(0,b)] = 2b^2, so Var[mean of n Lap] = 2b^2/n^2 * n = 2b^2/n
    # Expected L2 error ≈ sqrt(dim * 2 * scale^2 / n)
    expected_l2_error = math.sqrt(dim * 2.0 * scale ** 2 / n)
    accuracy = 1.0 / (1.0 + expected_l2_error)

    budget_remaining = total_budget - epsilon

    return PrivateResult(
        noisy_aggregate=noisy_agg,
        true_aggregate=true_agg,
        epsilon_spent=epsilon,
        delta=0.0,  # pure epsilon-DP, no delta
        accuracy_estimate=accuracy,
        noise_scale=scale,
        num_agents=n,
        composition_budget_remaining=max(budget_remaining, 0.0),
    )


def informed_consent_check(
    mechanism: Dict[str, Any],
) -> ConsentReport:
    """Check whether a mechanism satisfies informed consent requirements.

    Parameters
    ----------
    mechanism : dict
        Should contain metadata about the mechanism:
          - description: str        (human-readable description)
          - data_usage: str         (how data will be used)
          - opt_out_available: bool (can agents decline?)
          - opt_out_penalty: float  (cost of opting out, 0 = no penalty)
          - complexity_score: float (0-1, higher = more complex)
          - purpose: str            (stated purpose)
          - secondary_uses: list    (any other uses of the data)

    Criteria
    --------
    (a) Transparency — agents know how their data is used.
    (b) Voluntariness — agents can opt out without penalty.
    (c) Understanding — mechanism is simple enough to understand.
    (d) Purpose limitation — data used only for stated purpose.
    """
    report = ConsentReport()
    recommendations: List[str] = []

    # (a) Transparency
    description = mechanism.get("description", "")
    data_usage = mechanism.get("data_usage", "")
    transparency_score = 0.0
    if description:
        transparency_score += 0.4
    if data_usage:
        transparency_score += 0.3
    if len(description) > 50:
        transparency_score += 0.15
    if len(data_usage) > 30:
        transparency_score += 0.15
    transparency_score = min(transparency_score, 1.0)
    trans_criterion = ConsentCriterion(
        name="transparency",
        description="Agents know how their data is used.",
        score=transparency_score,
        satisfied=transparency_score >= 0.7,
        evidence=f"Description length: {len(description)}, data usage length: {len(data_usage)}",
    )
    report.criteria.append(trans_criterion)
    if not trans_criterion.satisfied:
        recommendations.append(
            "Improve transparency: provide a detailed description and "
            "explicit data usage statement."
        )

    # (b) Voluntariness
    opt_out = mechanism.get("opt_out_available", False)
    penalty = mechanism.get("opt_out_penalty", 1.0)
    vol_score = 0.0
    if opt_out:
        vol_score = 1.0 - min(float(penalty), 1.0)
    vol_criterion = ConsentCriterion(
        name="voluntariness",
        description="Agents can opt out without penalty.",
        score=vol_score,
        satisfied=vol_score >= 0.8,
        evidence=f"Opt-out available: {opt_out}, penalty: {penalty}",
    )
    report.criteria.append(vol_criterion)
    if not vol_criterion.satisfied:
        recommendations.append(
            "Ensure agents can opt out without material penalty. "
            f"Current opt-out penalty is {penalty}."
        )

    # (c) Understanding
    complexity = mechanism.get("complexity_score", 0.5)
    understanding_score = 1.0 - float(complexity)
    und_criterion = ConsentCriterion(
        name="understanding",
        description="Mechanism is simple enough for agents to understand.",
        score=understanding_score,
        satisfied=understanding_score >= 0.6,
        evidence=f"Complexity score: {complexity}",
    )
    report.criteria.append(und_criterion)
    if not und_criterion.satisfied:
        recommendations.append(
            "Simplify the mechanism or provide better explanations. "
            f"Current complexity score is {complexity:.2f}."
        )

    # (d) Purpose limitation
    purpose = mechanism.get("purpose", "")
    secondary_uses = mechanism.get("secondary_uses", [])
    purpose_score = 1.0 if purpose else 0.3
    if secondary_uses:
        purpose_score -= 0.15 * len(secondary_uses)
    purpose_score = max(purpose_score, 0.0)
    purp_criterion = ConsentCriterion(
        name="purpose_limitation",
        description="Data is used only for the stated purpose.",
        score=purpose_score,
        satisfied=purpose_score >= 0.7,
        evidence=f"Purpose: '{purpose}', secondary uses: {secondary_uses}",
    )
    report.criteria.append(purp_criterion)
    if not purp_criterion.satisfied:
        recommendations.append(
            "Limit data usage to the stated purpose. "
            f"Found {len(secondary_uses)} secondary use(s)."
        )

    # Overall
    all_scores = [c.score for c in report.criteria]
    report.overall_score = float(np.mean(all_scores))
    report.consent_valid = all(c.satisfied for c in report.criteria)
    report.recommendations = recommendations if recommendations else [
        "All informed consent criteria are satisfied."
    ]

    return report


def stakeholder_analysis(
    decision: Dict[str, Any],
    stakeholders: List[Stakeholder],
) -> StakeholderReport:
    """Analyse the impact of a decision on different stakeholders.

    Parameters
    ----------
    decision : dict
        Must contain:
          - dimensions: np.ndarray  (embedding of decision attributes)
          - magnitude: float        (scale of the decision's effect)
    stakeholders : list of Stakeholder
        Each stakeholder has a ``concerns`` embedding vector, ``power``, and
        ``interest`` fields populated.

    For each stakeholder:
      - Compute impact via cosine similarity between stakeholder concerns and
        decision dimensions, scaled by decision magnitude.
      - Populate power/interest scores.
      - Classify into power-interest grid.
      - Identify underrepresented stakeholders (low power, high interest).
    """
    decision_dims = np.asarray(decision.get("dimensions", []), dtype=np.float64)
    magnitude = float(decision.get("magnitude", 1.0))

    report = StakeholderReport()
    total_pos = 0.0
    total_neg = 0.0

    for s in stakeholders:
        concerns = np.asarray(s.concerns, dtype=np.float64)

        # Pad to equal length
        max_len = max(len(concerns), len(decision_dims))
        c_padded = np.zeros(max_len)
        d_padded = np.zeros(max_len)
        c_padded[: len(concerns)] = concerns
        d_padded[: len(decision_dims)] = decision_dims

        # Impact = cosine_similarity * magnitude * interest_weight
        norm_c = np.linalg.norm(c_padded)
        norm_d = np.linalg.norm(d_padded)
        if norm_c > 0 and norm_d > 0:
            sim = float(np.dot(c_padded, d_padded) / (norm_c * norm_d))
        else:
            sim = 0.0

        impact = sim * magnitude * (0.5 + 0.5 * s.interest)
        s.impact = impact
        report.stakeholder_impacts[s.id] = round(impact, 4)

        if impact >= 0:
            total_pos += impact
        else:
            total_neg += impact

    report.total_positive_impact = round(total_pos, 4)
    report.total_negative_impact = round(total_neg, 4)

    # Power-interest grid
    grid = _power_interest_grid(stakeholders)
    report.power_interest_grid = grid

    # Underrepresented: low power but high interest (most affected, least heard)
    report.underrepresented = grid["low_power_high_interest"]

    # Engagement strategy
    strategy_map = {
        "high_power_high_interest": "Manage closely: regular meetings, co-design",
        "high_power_low_interest": "Keep satisfied: periodic updates, consult on key decisions",
        "low_power_high_interest": "Keep informed: newsletters, feedback channels, advocacy",
        "low_power_low_interest": "Monitor: occasional updates, open channels",
    }
    for s in stakeholders:
        for quadrant, members in grid.items():
            if s.id in members:
                report.engagement_strategy[s.id] = strategy_map[quadrant]
                break

    return report


def moral_uncertainty_aggregation(
    ethical_views: List[EthicalView],
    credences: Optional[Dict[str, float]] = None,
    parliament_seats: int = 100,
) -> AggregatedView:
    """Aggregate recommendations under moral uncertainty.

    Parameters
    ----------
    ethical_views : list of EthicalView
        Each view has a framework label, a recommendation vector (e.g.
        probability distribution over actions), a confidence level, and a
        choiceworthiness score for its recommended action.
    credences : dict, optional
        Mapping from framework name to credence (probability that this
        framework is correct).  If None, uniform credences are used.
    parliament_seats : int
        Number of seats in the moral parliament.

    Methods implemented
    -------------------
    (a) Maximise Expected Choiceworthiness (MEC) — weighted average of
        recommendations, weighted by credence × choiceworthiness.
    (b) Moral parliament — proportional representation; each framework gets
        seats proportional to its credence, then majority vote.
    (c) My favourite theory — go with the highest-credence view.
    (d) Moral hedging (minimax regret) — choose the action minimising the
        worst-case regret across frameworks.

    Returns the MEC recommendation by default, with diagnostics for all
    methods.
    """
    if not ethical_views:
        return AggregatedView(method="none", confidence=0.0)

    n_views = len(ethical_views)
    dim = ethical_views[0].recommendation.shape[0]

    # Build credence weights
    if credences is None:
        weights = np.ones(n_views) / n_views
    else:
        raw = np.array([credences.get(v.framework, 1.0 / n_views) for v in ethical_views])
        total = raw.sum()
        weights = raw / total if total > 0 else np.ones(n_views) / n_views

    result = AggregatedView()

    # (a) MEC
    mec_rec = _expected_choiceworthiness(ethical_views, weights)

    # (b) Moral parliament — assign seats proportionally, then weighted vote
    seats: Dict[str, int] = {}
    remaining_seats = parliament_seats
    for i, v in enumerate(ethical_views):
        s = int(np.floor(weights[i] * parliament_seats))
        seats[v.framework] = s
        remaining_seats -= s
    # Distribute remaining seats to highest fractional parts
    fractional = [(weights[i] * parliament_seats - int(np.floor(weights[i] * parliament_seats)), i)
                  for i in range(n_views)]
    fractional.sort(reverse=True)
    for _, idx in fractional[:remaining_seats]:
        seats[ethical_views[idx].framework] += 1

    parliament_rec = np.zeros(dim, dtype=np.float64)
    for v in ethical_views:
        parliament_rec += seats[v.framework] * v.recommendation
    if parliament_seats > 0:
        parliament_rec /= parliament_seats

    result.parliament_seats = seats

    # (c) My favourite theory
    best_idx = int(np.argmax(weights))
    favourite_rec = ethical_views[best_idx].recommendation.copy()

    # (d) Minimax regret
    mmr_rec, mmr_bound = _minimax_regret(ethical_views)

    # Use MEC as the primary recommendation
    result.recommendation = mec_rec
    result.method = "maximise_expected_choiceworthiness"
    result.regret_bound = mmr_bound

    # Framework contributions (weight of each framework in MEC)
    for i, v in enumerate(ethical_views):
        effective_w = float(weights[i] * v.choiceworthiness)
        result.framework_contributions[v.framework] = round(effective_w, 4)

    # Overall confidence: weighted average of view confidences
    result.confidence = float(np.sum(weights * np.array([v.confidence for v in ethical_views])))

    return result
