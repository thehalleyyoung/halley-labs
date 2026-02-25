"""
CABER – Coalgebraic Behavioral Auditing of Foundation Models
Evaluation metrics for behavioral auditing.

Provides data-models, composite metric computations, statistical utilities,
and human-readable reporting.  Only stdlib dependencies (math, statistics,
dataclasses, random, collections, typing).
"""

from __future__ import annotations

import math
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# 1.  Data models
# ---------------------------------------------------------------------------


@dataclass
class FidelityMetrics:
    """Accuracy / cross-validation / confusion-matrix bundle."""

    prediction_accuracy: float
    cross_val_scores: List[float]
    mean_cv_score: float
    std_cv_score: float
    confusion_matrix: Dict[str, Dict[str, int]]
    per_class_f1: Dict[str, float]

    def __post_init__(self) -> None:
        if not 0.0 <= self.prediction_accuracy <= 1.0:
            raise ValueError("prediction_accuracy must be in [0, 1]")


@dataclass
class QueryComplexityMetrics:
    """Statistics about the queries issued during learning."""

    total_queries: int
    queries_per_state: float
    membership_queries: int
    equivalence_queries: int
    adversarial_queries: int
    query_entropy: float
    average_query_length: float


@dataclass
class CoverageMetrics:
    """Specification-coverage information."""

    total_specs: int
    tested_specs: int
    coverage_ratio: float
    uncovered_specs: List[str]
    per_category_coverage: Dict[str, float]


@dataclass
class CertificateSoundnessMetrics:
    """How well certificates reflect ground-truth properties."""

    total_certificates: int
    sound_certificates: int
    soundness_rate: float
    false_positive_specs: List[str]
    false_negative_specs: List[str]


@dataclass
class BisimulationMetrics:
    """Distance between two behavioural automata."""

    state_matching_accuracy: float
    transition_distance: float
    output_distance: float
    overall_distance: float
    matched_pairs: List[Tuple[str, str]]


@dataclass
class ComplexityMeasures:
    """Behavioural complexity of observed responses."""

    response_entropy: float
    behavioral_diversity: int
    vocabulary_complexity: float
    estimated_states: int
    myhill_nerode_lower_bound: int
    distinguishing_sequences_count: int


@dataclass
class MetricsSummary:
    """Aggregated summary across all metric dimensions."""

    fidelity: Optional[FidelityMetrics] = None
    query_complexity: Optional[QueryComplexityMetrics] = None
    coverage: Optional[CoverageMetrics] = None
    soundness: Optional[CertificateSoundnessMetrics] = None
    bisimulation: Optional[BisimulationMetrics] = None
    complexity: Optional[ComplexityMeasures] = None
    overall_score: float = 0.0


# ---------------------------------------------------------------------------
# 3.  Statistical utility functions  (placed before the class that uses them)
# ---------------------------------------------------------------------------


def _accuracy(predictions: list, ground_truth: list) -> float:
    """Proportion of correct predictions."""
    if len(predictions) != len(ground_truth):
        raise ValueError("predictions and ground_truth must have equal length")
    if not predictions:
        return 0.0
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    return correct / len(predictions)


def _precision_recall_f1(
    predictions: list,
    ground_truth: list,
    positive_label: str,
) -> Tuple[float, float, float]:
    """Binary precision, recall, F1 for *positive_label*."""
    tp = fp = fn = 0
    for p, g in zip(predictions, ground_truth):
        if p == positive_label and g == positive_label:
            tp += 1
        elif p == positive_label and g != positive_label:
            fp += 1
        elif p != positive_label and g == positive_label:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


def _confusion_matrix(
    predictions: list,
    ground_truth: list,
) -> Dict[str, Dict[str, int]]:
    """Build a label→label confusion matrix as nested dicts.

    Keys of the outer dict are ground-truth labels; inner keys are predicted
    labels.
    """
    labels = sorted(set(predictions) | set(ground_truth))
    matrix: Dict[str, Dict[str, int]] = {
        gt: {pred: 0 for pred in labels} for gt in labels
    }
    for p, g in zip(predictions, ground_truth):
        matrix[g][p] += 1
    return matrix


def _shannon_entropy(distribution: Dict[str, float]) -> float:
    """Shannon entropy in *nats* (natural-log base).

    The *distribution* values must be non-negative and are normalised
    internally so they need not sum to 1.
    """
    total = sum(distribution.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in distribution.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log(p)
    return entropy


def _kl_divergence(
    p: Dict[str, float],
    q: Dict[str, float],
) -> float:
    """Kullback–Leibler divergence D_KL(P || Q).

    Both *p* and *q* are normalised internally.  Keys present in *p* but
    absent in *q* contribute ``+inf``; keys in *q* but not in *p* are
    ignored (standard convention).
    """
    total_p = sum(p.values())
    total_q = sum(q.values())
    if total_p <= 0 or total_q <= 0:
        return 0.0

    q_norm: Dict[str, float] = {k: v / total_q for k, v in q.items()}
    divergence = 0.0
    for k, v in p.items():
        pk = v / total_p
        if pk <= 0:
            continue
        qk = q_norm.get(k, 0.0)
        if qk <= 0:
            return float("inf")
        divergence += pk * math.log(pk / qk)
    return divergence


def _l1_distance(
    p: Dict[str, float],
    q: Dict[str, float],
) -> float:
    """L1 (total-variation) distance between two distributions.

    Both inputs are normalised internally.
    """
    total_p = sum(p.values()) or 1.0
    total_q = sum(q.values()) or 1.0
    all_keys = set(p) | set(q)
    dist = 0.0
    for k in all_keys:
        pk = p.get(k, 0.0) / total_p
        qk = q.get(k, 0.0) / total_q
        dist += abs(pk - qk)
    return dist


def _jaccard_similarity(set_a: set, set_b: set) -> float:
    """Jaccard index  |A ∩ B| / |A ∪ B|."""
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _bootstrap_confidence_interval(
    values: List[float],
    num_samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Non-parametric bootstrap CI for the mean of *values*.

    Returns the (lower, upper) bounds at the requested *confidence* level.
    Uses a fixed *seed* for reproducibility.
    """
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means: List[float] = []
    for _ in range(num_samples):
        sample = [rng.choice(values) for _ in range(n)]
        means.append(statistics.mean(sample))
    means.sort()
    alpha = 1.0 - confidence
    lo_idx = max(0, int(math.floor((alpha / 2) * num_samples)))
    hi_idx = min(num_samples - 1, int(math.ceil((1 - alpha / 2) * num_samples)) - 1)
    return (means[lo_idx], means[hi_idx])


def _cohen_kappa(predictions: list, ground_truth: list) -> float:
    """Cohen's κ — inter-rater agreement corrected for chance.

    κ = (p_o − p_e) / (1 − p_e)
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("predictions and ground_truth must have equal length")
    n = len(predictions)
    if n == 0:
        return 0.0

    labels = sorted(set(predictions) | set(ground_truth))
    cm = _confusion_matrix(predictions, ground_truth)

    # Observed agreement
    p_o = sum(cm[l][l] for l in labels) / n

    # Expected agreement
    p_e = 0.0
    for l in labels:
        row_sum = sum(cm[l].values()) / n           # ground-truth marginal
        col_sum = sum(cm[gt][l] for gt in labels) / n  # predicted marginal
        p_e += row_sum * col_sum

    if abs(1.0 - p_e) < 1e-12:
        return 1.0 if abs(p_o - 1.0) < 1e-12 else 0.0
    return (p_o - p_e) / (1.0 - p_e)


def _cramers_v(contingency: Dict[str, Dict[str, int]]) -> float:
    """Cramér's V for a contingency table (dict-of-dicts).

    V = sqrt(χ² / (n · min(r−1, c−1)))
    """
    row_labels = sorted(contingency.keys())
    col_labels_set: set = set()
    for inner in contingency.values():
        col_labels_set.update(inner.keys())
    col_labels = sorted(col_labels_set)

    r = len(row_labels)
    c = len(col_labels)
    if r <= 1 or c <= 1:
        return 0.0

    n = 0
    row_totals: Dict[str, int] = {}
    col_totals: Dict[str, int] = defaultdict(int)
    for rl in row_labels:
        rt = 0
        for cl in col_labels:
            val = contingency[rl].get(cl, 0)
            rt += val
            col_totals[cl] += val
        row_totals[rl] = rt
        n += rt

    if n == 0:
        return 0.0

    chi2 = 0.0
    for rl in row_labels:
        for cl in col_labels:
            observed = contingency[rl].get(cl, 0)
            expected = (row_totals[rl] * col_totals[cl]) / n
            if expected > 0:
                chi2 += (observed - expected) ** 2 / expected

    denominator = n * (min(r, c) - 1)
    if denominator <= 0:
        return 0.0
    return math.sqrt(chi2 / denominator)


# ---------------------------------------------------------------------------
# Additional statistical helpers (private)
# ---------------------------------------------------------------------------


def _normalize_distribution(d: Dict[str, float]) -> Dict[str, float]:
    """Return a copy of *d* with values summing to 1."""
    total = sum(d.values())
    if total <= 0:
        return {k: 0.0 for k in d}
    return {k: v / total for k, v in d.items()}


def _cosine_similarity(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Cosine similarity between two sparse vectors."""
    keys = set(a) | set(b)
    dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
    norm_a = math.sqrt(sum(v * v for v in a.values())) or 1e-12
    norm_b = math.sqrt(sum(v * v for v in b.values())) or 1e-12
    return dot / (norm_a * norm_b)


def _weighted_mean(values: List[float], weights: List[float]) -> float:
    """Weighted arithmetic mean."""
    if not values or not weights:
        return 0.0
    total_w = sum(weights)
    if total_w <= 0:
        return 0.0
    return sum(v * w for v, w in zip(values, weights)) / total_w


def _median_absolute_deviation(values: List[float]) -> float:
    """MAD – a robust measure of spread."""
    if not values:
        return 0.0
    med = statistics.median(values)
    deviations = [abs(v - med) for v in values]
    return statistics.median(deviations)


def _gini_impurity(distribution: Dict[str, float]) -> float:
    """Gini impurity: 1 − Σ p_i²."""
    total = sum(distribution.values())
    if total <= 0:
        return 0.0
    return 1.0 - sum((v / total) ** 2 for v in distribution.values())


def _harmonic_mean(a: float, b: float) -> float:
    """Harmonic mean of two non-negative values."""
    if a + b <= 0:
        return 0.0
    return 2.0 * a * b / (a + b)


def _stratified_split(
    data: list,
    labels: list,
    k: int,
    seed: int = 0,
) -> List[List[int]]:
    """Stratified k-fold index split.

    Returns *k* lists of indices so that each fold preserves the label
    distribution as closely as possible.
    """
    rng = random.Random(seed)
    label_indices: Dict[str, List[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        label_indices[lab].append(i)

    # Shuffle within each label group
    for indices in label_indices.values():
        rng.shuffle(indices)

    folds: List[List[int]] = [[] for _ in range(k)]
    for indices in label_indices.values():
        for fold_idx, idx in enumerate(indices):
            folds[fold_idx % k].append(idx)

    for fold in folds:
        rng.shuffle(fold)
    return folds


def _majority_class_accuracy(labels: list) -> float:
    """Accuracy of a majority-class classifier."""
    if not labels:
        return 0.0
    counter = Counter(labels)
    return counter.most_common(1)[0][1] / len(labels)


def _response_type_classifier(response: str) -> str:
    """Classify a response into a coarse behavioural type.

    Uses the first significant word/token pattern to bucket responses.
    """
    stripped = response.strip()
    if not stripped:
        return "EMPTY"

    lowered = stripped.lower()
    # Classify by first-word pattern
    first_word = lowered.split()[0] if lowered.split() else ""

    affirmative = {"yes", "sure", "absolutely", "correct", "right", "true", "ok", "okay"}
    negative = {"no", "not", "never", "false", "incorrect", "wrong", "nope"}
    interrogative = {"what", "why", "how", "when", "where", "who", "which", "is", "are", "do", "does", "can", "could", "would", "should"}
    imperative = {"please", "let", "try", "go", "run", "stop", "start", "open", "close", "set", "get"}
    explanatory = {"because", "since", "therefore", "thus", "hence", "so", "the", "a", "an", "this", "that", "it", "i"}
    numeric_start = first_word and (first_word[0].isdigit() or first_word in {"one", "two", "three", "four", "five"})

    if first_word in affirmative:
        return "AFFIRMATIVE"
    if first_word in negative:
        return "NEGATIVE"
    if first_word in interrogative:
        return "INTERROGATIVE"
    if first_word in imperative:
        return "IMPERATIVE"
    if numeric_start:
        return "NUMERIC"
    if first_word in explanatory:
        return "EXPLANATORY"

    # Fall back on length-based heuristic
    word_count = len(stripped.split())
    if word_count <= 3:
        return "SHORT"
    if word_count <= 15:
        return "MEDIUM"
    return "LONG"


def _simple_hash_bucket(text: str, num_buckets: int = 64) -> int:
    """Deterministic hash-based bucketing (no external deps)."""
    h = 5381
    for ch in text:
        h = ((h << 5) + h + ord(ch)) & 0xFFFFFFFF
    return h % num_buckets


def _type_token_ratio(texts: List[str]) -> float:
    """Type-token ratio across all *texts*.

    TTR = |unique tokens| / |total tokens|.
    """
    all_tokens: List[str] = []
    for t in texts:
        all_tokens.extend(t.lower().split())
    if not all_tokens:
        return 0.0
    return len(set(all_tokens)) / len(all_tokens)


def _count_distinguishing_suffixes(responses: List[str], max_suffix_len: int = 5) -> int:
    """Estimate the number of distinguishing suffixes needed.

    Two responses are *distinguished* by a suffix if appending that suffix
    would (heuristically) change one's classification but not the other's.
    We approximate this by counting how many distinct suffix-equivalence
    classes exist.
    """
    suffix_classes: Set[str] = set()
    for resp in responses:
        tokens = resp.lower().split()
        for length in range(1, min(max_suffix_len + 1, len(tokens) + 1)):
            suffix = " ".join(tokens[-length:])
            suffix_classes.add(suffix)
    return len(suffix_classes)


def _transition_probability_vector(
    transitions: List[dict],
    source: str,
    all_targets: List[str],
    all_symbols: List[str],
) -> Dict[str, float]:
    """Build a probability vector keyed by (target, symbol) for a source state."""
    vec: Dict[str, float] = {}
    for t in all_targets:
        for s in all_symbols:
            vec[f"{t}|{s}"] = 0.0

    for tr in transitions:
        if tr.get("source") == source:
            key = f"{tr['target']}|{tr['symbol']}"
            vec[key] = tr.get("probability", 0.0)

    return vec


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division that returns *default* when denominator is zero."""
    if abs(denominator) < 1e-15:
        return default
    return numerator / denominator


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp *value* to [lo, hi]."""
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# 2.  EvaluationMetrics class
# ---------------------------------------------------------------------------


class EvaluationMetrics:
    """Composite evaluator for CABER behavioural auditing pipelines."""

    # Weights for the overall score
    _WEIGHT_FIDELITY: float = 0.3
    _WEIGHT_COVERAGE: float = 0.2
    _WEIGHT_SOUNDNESS: float = 0.3
    _WEIGHT_BISIMULATION: float = 0.2

    def __init__(self) -> None:
        # Prediction / ground-truth pairs accumulated over time
        self._predictions: List[str] = []
        self._ground_truth: List[str] = []

        # Query log – each entry is a dict with at least {"type": ..., "content": ...}
        self._query_log: List[dict] = []

        # Specification coverage tracker
        self._tested_properties: Set[str] = set()
        self._all_properties: Set[str] = set()

        # Certificate verification results
        self._certificate_results: List[dict] = []

    # ------------------------------------------------------------------
    # Public accumulation helpers
    # ------------------------------------------------------------------

    def record_prediction(self, prediction: str, truth: str) -> None:
        """Append a single prediction / ground-truth pair."""
        self._predictions.append(prediction)
        self._ground_truth.append(truth)

    def record_query(self, query: dict) -> None:
        """Append a query entry to the internal log."""
        self._query_log.append(query)

    def record_tested_property(self, prop: str) -> None:
        self._tested_properties.add(prop)

    def register_properties(self, props: List[str]) -> None:
        self._all_properties.update(props)

    def record_certificate_result(self, result: dict) -> None:
        self._certificate_results.append(result)

    # ------------------------------------------------------------------
    # Core metric computations
    # ------------------------------------------------------------------

    def compute_automaton_fidelity(
        self,
        predictions: List[str],
        ground_truth: List[str],
        k_folds: int = 5,
    ) -> FidelityMetrics:
        """Compute automaton-fidelity metrics.

        Parameters
        ----------
        predictions : list[str]
            Predicted labels from the learned automaton.
        ground_truth : list[str]
            True labels.
        k_folds : int
            Number of cross-validation folds.

        Returns
        -------
        FidelityMetrics
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("predictions and ground_truth must have equal length")

        # Overall accuracy
        pred_acc = _accuracy(predictions, ground_truth)

        # Confusion matrix
        cm = _confusion_matrix(predictions, ground_truth)

        # Per-class F1
        labels = sorted(set(predictions) | set(ground_truth))
        per_class: Dict[str, float] = {}
        for lab in labels:
            _, _, f1 = _precision_recall_f1(predictions, ground_truth, lab)
            per_class[lab] = round(f1, 6)

        # k-fold cross-validation (simulated with index splits)
        n = len(predictions)
        k_folds = min(k_folds, n) if n > 0 else 1
        k_folds = max(k_folds, 2) if n >= 2 else 1

        cv_scores: List[float] = []
        if n > 1 and k_folds >= 2:
            folds = _stratified_split(predictions, ground_truth, k_folds)
            for fold_idx in range(k_folds):
                test_indices = set(folds[fold_idx])
                train_labels = [
                    ground_truth[i] for i in range(n) if i not in test_indices
                ]
                test_preds = [predictions[i] for i in range(n) if i in test_indices]
                test_truth = [
                    ground_truth[i] for i in range(n) if i in test_indices
                ]

                if not test_preds:
                    continue

                # Majority-class baseline from training split
                if train_labels:
                    majority = Counter(train_labels).most_common(1)[0][0]
                else:
                    majority = Counter(test_truth).most_common(1)[0][0]

                # Score = accuracy of actual predictions on test split, adjusted
                # by majority baseline (gives more credit when beating baseline).
                fold_acc = _accuracy(test_preds, test_truth)
                baseline_acc = sum(1 for g in test_truth if g == majority) / len(
                    test_truth
                )
                # Report raw fold accuracy (interpretable)
                cv_scores.append(fold_acc)
        else:
            cv_scores = [pred_acc]

        mean_cv = statistics.mean(cv_scores) if cv_scores else 0.0
        std_cv = statistics.pstdev(cv_scores) if len(cv_scores) > 1 else 0.0

        return FidelityMetrics(
            prediction_accuracy=round(pred_acc, 6),
            cross_val_scores=[round(s, 6) for s in cv_scores],
            mean_cv_score=round(mean_cv, 6),
            std_cv_score=round(std_cv, 6),
            confusion_matrix=cm,
            per_class_f1=per_class,
        )

    def compute_query_complexity(
        self,
        query_log: List[dict],
    ) -> QueryComplexityMetrics:
        """Analyse the query log.

        Each entry in *query_log* should be a dict with at least:
        - ``type``: one of ``"membership"``, ``"equivalence"``, ``"adversarial"``,
          or any other string.
        - ``content``: the query string (used to compute average length).
        - Optionally ``num_states``: int (used once to derive queries_per_state).
        """
        total = len(query_log)
        type_counts: Dict[str, int] = Counter()
        total_length = 0
        num_states: Optional[int] = None

        membership = 0
        equivalence = 0
        adversarial = 0

        for entry in query_log:
            qtype = entry.get("type", "unknown")
            type_counts[qtype] += 1

            content = entry.get("content", "")
            total_length += len(str(content))

            if qtype == "membership":
                membership += 1
            elif qtype == "equivalence":
                equivalence += 1
            elif qtype == "adversarial":
                adversarial += 1

            ns = entry.get("num_states")
            if ns is not None:
                num_states = int(ns)

        # Queries per state
        if num_states and num_states > 0:
            qps = total / num_states
        else:
            qps = float(total)

        # Query entropy over type distribution
        q_entropy = _shannon_entropy(
            {k: float(v) for k, v in type_counts.items()}
        )

        avg_len = total_length / total if total > 0 else 0.0

        return QueryComplexityMetrics(
            total_queries=total,
            queries_per_state=round(qps, 6),
            membership_queries=membership,
            equivalence_queries=equivalence,
            adversarial_queries=adversarial,
            query_entropy=round(q_entropy, 6),
            average_query_length=round(avg_len, 6),
        )

    def compute_coverage(
        self,
        tested_properties: List[str],
        all_properties: List[str],
        category_map: Optional[Dict[str, str]] = None,
    ) -> CoverageMetrics:
        """Compute specification coverage.

        Parameters
        ----------
        tested_properties : list[str]
            Properties that have been tested.
        all_properties : list[str]
            Full set of properties that *should* be tested.
        category_map : dict[str, str] | None
            Optional mapping from property name to category name.
        """
        tested_set = set(tested_properties)
        all_set = set(all_properties)
        total = len(all_set)
        tested_count = len(tested_set & all_set)
        coverage = tested_count / total if total > 0 else 0.0
        uncovered = sorted(all_set - tested_set)

        per_cat: Dict[str, float] = {}
        if category_map is not None:
            cat_all: Dict[str, int] = defaultdict(int)
            cat_tested: Dict[str, int] = defaultdict(int)
            for prop in all_properties:
                cat = category_map.get(prop, "uncategorized")
                cat_all[cat] += 1
                if prop in tested_set:
                    cat_tested[cat] += 1
            for cat in sorted(cat_all):
                per_cat[cat] = round(
                    cat_tested[cat] / cat_all[cat] if cat_all[cat] > 0 else 0.0,
                    6,
                )

        return CoverageMetrics(
            total_specs=total,
            tested_specs=tested_count,
            coverage_ratio=round(coverage, 6),
            uncovered_specs=uncovered,
            per_category_coverage=per_cat,
        )

    def compute_certificate_soundness(
        self,
        certificates: List[dict],
        ground_truth_properties: Dict[str, bool],
    ) -> CertificateSoundnessMetrics:
        """Verify certificates against ground truth.

        Each certificate dict must have:
        - ``property``: str — the property name.
        - ``verdict``: bool — whether the certificate claims the property holds.

        *ground_truth_properties* maps property names to their true status.
        """
        total = len(certificates)
        sound = 0
        false_positives: List[str] = []
        false_negatives: List[str] = []

        for cert in certificates:
            prop = cert.get("property", "")
            verdict = cert.get("verdict", False)
            truth = ground_truth_properties.get(prop)

            if truth is None:
                # Property not in ground truth – skip but still count
                continue

            if verdict == truth:
                sound += 1
            elif verdict and not truth:
                false_positives.append(prop)
            elif not verdict and truth:
                false_negatives.append(prop)

        rate = sound / total if total > 0 else 0.0

        return CertificateSoundnessMetrics(
            total_certificates=total,
            sound_certificates=sound,
            soundness_rate=round(rate, 6),
            false_positive_specs=sorted(false_positives),
            false_negative_specs=sorted(false_negatives),
        )

    def compute_bisimulation_distance(
        self,
        automaton_a: dict,
        automaton_b: dict,
    ) -> BisimulationMetrics:
        """Compute bisimulation distance between two automata.

        Each automaton dict has:
        - ``states``: list of ``{"state_id": str, "dominant_behavior": str, ...}``
        - ``transitions``: list of ``{"source": str, "target": str,
          "symbol": str, "probability": float}``

        The distance is 0 when the automata are bisimilar.
        """
        states_a = automaton_a.get("states", [])
        states_b = automaton_b.get("states", [])
        trans_a = automaton_a.get("transitions", [])
        trans_b = automaton_b.get("transitions", [])

        # ---- State matching by dominant_behavior ---
        behavior_to_a: Dict[str, List[str]] = defaultdict(list)
        behavior_to_b: Dict[str, List[str]] = defaultdict(list)
        for s in states_a:
            behavior_to_a[s.get("dominant_behavior", "")].append(s["state_id"])
        for s in states_b:
            behavior_to_b[s.get("dominant_behavior", "")].append(s["state_id"])

        matched_pairs: List[Tuple[str, str]] = []
        matched_a: Set[str] = set()
        matched_b: Set[str] = set()

        for beh in behavior_to_a:
            if beh in behavior_to_b:
                for sa in behavior_to_a[beh]:
                    if sa in matched_a:
                        continue
                    for sb in behavior_to_b[beh]:
                        if sb in matched_b:
                            continue
                        matched_pairs.append((sa, sb))
                        matched_a.add(sa)
                        matched_b.add(sb)
                        break

        total_matchable = max(len(states_a), len(states_b), 1)
        state_match_acc = len(matched_pairs) / total_matchable

        # ---- Transition distance (L1 on probability vectors) ----
        all_targets_a = sorted({s["state_id"] for s in states_a})
        all_targets_b = sorted({s["state_id"] for s in states_b})
        all_symbols = sorted(
            {t["symbol"] for t in trans_a} | {t["symbol"] for t in trans_b}
        )

        trans_distances: List[float] = []
        for sa, sb in matched_pairs:
            vec_a = _transition_probability_vector(trans_a, sa, all_targets_a, all_symbols)
            vec_b = _transition_probability_vector(trans_b, sb, all_targets_b, all_symbols)

            # Re-key vec_a through the matching so keys are comparable
            mapped_a: Dict[str, float] = {}
            pair_map_a_to_b: Dict[str, str] = {pa: pb for pa, pb in matched_pairs}
            for key, val in vec_a.items():
                tgt, sym = key.split("|", 1)
                mapped_tgt = pair_map_a_to_b.get(tgt, tgt)
                mapped_key = f"{mapped_tgt}|{sym}"
                mapped_a[mapped_key] = mapped_a.get(mapped_key, 0.0) + val

            dist = _l1_distance(mapped_a, vec_b)
            trans_distances.append(dist)

        transition_dist = (
            statistics.mean(trans_distances) if trans_distances else 1.0
        )

        # ---- Output distance ----
        output_distances: List[float] = []
        for sa, sb in matched_pairs:
            out_a = self._state_output_distribution(states_a, sa)
            out_b = self._state_output_distribution(states_b, sb)
            output_distances.append(_l1_distance(out_a, out_b))

        output_dist = (
            statistics.mean(output_distances) if output_distances else 1.0
        )

        # ---- Overall weighted distance ----
        overall = _clamp(
            0.4 * (1.0 - state_match_acc)
            + 0.3 * _clamp(transition_dist)
            + 0.3 * _clamp(output_dist)
        )

        return BisimulationMetrics(
            state_matching_accuracy=round(state_match_acc, 6),
            transition_distance=round(transition_dist, 6),
            output_distance=round(output_dist, 6),
            overall_distance=round(overall, 6),
            matched_pairs=matched_pairs,
        )

    def compute_behavioral_complexity(
        self,
        responses: List[str],
    ) -> ComplexityMeasures:
        """Measure behavioural complexity of a set of LLM responses."""
        if not responses:
            return ComplexityMeasures(
                response_entropy=0.0,
                behavioral_diversity=0,
                vocabulary_complexity=0.0,
                estimated_states=0,
                myhill_nerode_lower_bound=0,
                distinguishing_sequences_count=0,
            )

        # Response-type distribution & entropy
        type_counts: Dict[str, float] = defaultdict(float)
        for r in responses:
            rtype = _response_type_classifier(r)
            type_counts[rtype] += 1.0
        resp_entropy = _shannon_entropy(type_counts)

        # Behavioural diversity via hash bucketing
        buckets: Set[int] = set()
        for r in responses:
            buckets.add(_simple_hash_bucket(r, num_buckets=128))
        behavioral_div = len(buckets)

        # Vocabulary complexity (type-token ratio)
        vocab_complexity = _type_token_ratio(responses)

        # Estimated states
        estimated = max(behavioral_div, math.ceil(resp_entropy * 2))

        # Myhill-Nerode lower bound
        mn_lower = max(1, len(type_counts))

        # Distinguishing sequences
        dist_seqs = _count_distinguishing_suffixes(responses)

        return ComplexityMeasures(
            response_entropy=round(resp_entropy, 6),
            behavioral_diversity=behavioral_div,
            vocabulary_complexity=round(vocab_complexity, 6),
            estimated_states=estimated,
            myhill_nerode_lower_bound=mn_lower,
            distinguishing_sequences_count=dist_seqs,
        )

    # ------------------------------------------------------------------
    # Scoring & comparison
    # ------------------------------------------------------------------

    def compute_overall_score(self, summary: MetricsSummary) -> float:
        """Weighted combination of available metric dimensions.

        Returns a score in [0, 1].
        """
        components: List[float] = []
        weights: List[float] = []

        if summary.fidelity is not None:
            components.append(summary.fidelity.prediction_accuracy)
            weights.append(self._WEIGHT_FIDELITY)

        if summary.coverage is not None:
            components.append(summary.coverage.coverage_ratio)
            weights.append(self._WEIGHT_COVERAGE)

        if summary.soundness is not None:
            components.append(summary.soundness.soundness_rate)
            weights.append(self._WEIGHT_SOUNDNESS)

        if summary.bisimulation is not None:
            # Lower distance is better → convert to a "quality" score
            components.append(1.0 - _clamp(summary.bisimulation.overall_distance))
            weights.append(self._WEIGHT_BISIMULATION)

        if not components:
            return 0.0

        score = _weighted_mean(components, weights)
        return round(_clamp(score), 6)

    def compare_with_baseline(
        self,
        caber_metrics: MetricsSummary,
        baseline_metrics: MetricsSummary,
    ) -> dict:
        """Compare CABER metrics against a baseline.

        Returns a dict mapping dimension names to improvement dicts with
        ``caber_value``, ``baseline_value``, ``improvement_pct``, and
        ``status`` (``"improved"`` / ``"degraded"`` / ``"unchanged"``).
        """
        comparison: Dict[str, dict] = {}

        pairs: List[Tuple[str, Optional[float], Optional[float]]] = []

        # Fidelity
        fid_c = caber_metrics.fidelity.prediction_accuracy if caber_metrics.fidelity else None
        fid_b = baseline_metrics.fidelity.prediction_accuracy if baseline_metrics.fidelity else None
        pairs.append(("fidelity_accuracy", fid_c, fid_b))

        fid_cv_c = caber_metrics.fidelity.mean_cv_score if caber_metrics.fidelity else None
        fid_cv_b = baseline_metrics.fidelity.mean_cv_score if baseline_metrics.fidelity else None
        pairs.append(("fidelity_mean_cv", fid_cv_c, fid_cv_b))

        # Coverage
        cov_c = caber_metrics.coverage.coverage_ratio if caber_metrics.coverage else None
        cov_b = baseline_metrics.coverage.coverage_ratio if baseline_metrics.coverage else None
        pairs.append(("coverage_ratio", cov_c, cov_b))

        # Soundness
        snd_c = caber_metrics.soundness.soundness_rate if caber_metrics.soundness else None
        snd_b = baseline_metrics.soundness.soundness_rate if baseline_metrics.soundness else None
        pairs.append(("soundness_rate", snd_c, snd_b))

        # Bisimulation (lower is better, so flip sign)
        bis_c = caber_metrics.bisimulation.overall_distance if caber_metrics.bisimulation else None
        bis_b = baseline_metrics.bisimulation.overall_distance if baseline_metrics.bisimulation else None
        pairs.append(("bisimulation_distance", bis_c, bis_b))

        # Query complexity
        qc_c = caber_metrics.query_complexity.total_queries if caber_metrics.query_complexity else None
        qc_b = baseline_metrics.query_complexity.total_queries if baseline_metrics.query_complexity else None
        pairs.append(("query_total", float(qc_c) if qc_c is not None else None, float(qc_b) if qc_b is not None else None))

        # Overall
        pairs.append(("overall_score", caber_metrics.overall_score, baseline_metrics.overall_score))

        for name, cv, bv in pairs:
            if cv is None or bv is None:
                comparison[name] = {
                    "caber_value": cv,
                    "baseline_value": bv,
                    "improvement_pct": None,
                    "status": "unavailable",
                }
                continue

            # For bisimulation_distance and query_total, lower is better
            lower_better = name in ("bisimulation_distance", "query_total")

            if abs(bv) < 1e-12:
                pct = 0.0 if abs(cv) < 1e-12 else float("inf")
            else:
                pct = ((bv - cv) / abs(bv)) * 100.0 if lower_better else ((cv - bv) / abs(bv)) * 100.0

            if abs(pct) < 1e-6:
                status = "unchanged"
            elif pct > 0:
                status = "improved"
            else:
                status = "degraded"

            comparison[name] = {
                "caber_value": round(cv, 6) if isinstance(cv, float) else cv,
                "baseline_value": round(bv, 6) if isinstance(bv, float) else bv,
                "improvement_pct": round(pct, 2),
                "status": status,
            }

        return comparison

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def format_report(self, summary: MetricsSummary) -> str:
        """Render a human-readable text report."""
        lines: List[str] = []
        sep = "=" * 68

        lines.append(sep)
        lines.append("  CABER  –  Behavioral Auditing Evaluation Report")
        lines.append(sep)
        lines.append("")

        # Fidelity
        if summary.fidelity is not None:
            f = summary.fidelity
            lines.append("── Automaton Fidelity ──────────────────────────────")
            lines.append(f"  Prediction accuracy    : {f.prediction_accuracy:.4f}")
            lines.append(f"  Mean CV score          : {f.mean_cv_score:.4f}")
            lines.append(f"  Std  CV score          : {f.std_cv_score:.4f}")
            lines.append(f"  CV scores              : {', '.join(f'{s:.4f}' for s in f.cross_val_scores)}")
            if f.per_class_f1:
                lines.append("  Per-class F1:")
                for cls, f1 in sorted(f.per_class_f1.items()):
                    lines.append(f"    {cls:<24s} {f1:.4f}")
            if f.confusion_matrix:
                labels = sorted(f.confusion_matrix.keys())
                lines.append("  Confusion matrix (rows=truth, cols=predicted):")
                header = "    {:>12s}".format("") + "".join(f"  {l:>8s}" for l in labels)
                lines.append(header)
                for gt in labels:
                    row_vals = "".join(f"  {f.confusion_matrix[gt].get(p, 0):>8d}" for p in labels)
                    lines.append(f"    {gt:>12s}{row_vals}")
            lines.append("")

        # Query complexity
        if summary.query_complexity is not None:
            q = summary.query_complexity
            lines.append("── Query Complexity ────────────────────────────────")
            lines.append(f"  Total queries          : {q.total_queries}")
            lines.append(f"  Queries per state      : {q.queries_per_state:.4f}")
            lines.append(f"  Membership queries     : {q.membership_queries}")
            lines.append(f"  Equivalence queries    : {q.equivalence_queries}")
            lines.append(f"  Adversarial queries    : {q.adversarial_queries}")
            lines.append(f"  Query entropy          : {q.query_entropy:.4f}")
            lines.append(f"  Avg query length       : {q.average_query_length:.2f}")
            lines.append("")

        # Coverage
        if summary.coverage is not None:
            c = summary.coverage
            lines.append("── Specification Coverage ──────────────────────────")
            lines.append(f"  Total specifications   : {c.total_specs}")
            lines.append(f"  Tested specifications  : {c.tested_specs}")
            lines.append(f"  Coverage ratio         : {c.coverage_ratio:.4f}")
            if c.uncovered_specs:
                lines.append(f"  Uncovered ({len(c.uncovered_specs)}):")
                for spec in c.uncovered_specs[:10]:
                    lines.append(f"    • {spec}")
                if len(c.uncovered_specs) > 10:
                    lines.append(f"    ... and {len(c.uncovered_specs) - 10} more")
            if c.per_category_coverage:
                lines.append("  Per-category coverage:")
                for cat, cov in sorted(c.per_category_coverage.items()):
                    bar_len = int(cov * 20)
                    bar = "█" * bar_len + "░" * (20 - bar_len)
                    lines.append(f"    {cat:<20s}  {bar}  {cov:.2%}")
            lines.append("")

        # Soundness
        if summary.soundness is not None:
            s = summary.soundness
            lines.append("── Certificate Soundness ───────────────────────────")
            lines.append(f"  Total certificates     : {s.total_certificates}")
            lines.append(f"  Sound certificates     : {s.sound_certificates}")
            lines.append(f"  Soundness rate         : {s.soundness_rate:.4f}")
            if s.false_positive_specs:
                lines.append(f"  False positives ({len(s.false_positive_specs)}):")
                for fp in s.false_positive_specs[:5]:
                    lines.append(f"    ✗ {fp}")
            if s.false_negative_specs:
                lines.append(f"  False negatives ({len(s.false_negative_specs)}):")
                for fn in s.false_negative_specs[:5]:
                    lines.append(f"    ✗ {fn}")
            lines.append("")

        # Bisimulation
        if summary.bisimulation is not None:
            b = summary.bisimulation
            lines.append("── Bisimulation Distance ───────────────────────────")
            lines.append(f"  State matching acc.    : {b.state_matching_accuracy:.4f}")
            lines.append(f"  Transition distance    : {b.transition_distance:.4f}")
            lines.append(f"  Output distance        : {b.output_distance:.4f}")
            lines.append(f"  Overall distance       : {b.overall_distance:.4f}")
            if b.matched_pairs:
                lines.append(f"  Matched state pairs ({len(b.matched_pairs)}):")
                for sa, sb in b.matched_pairs[:8]:
                    lines.append(f"    {sa} ⟷  {sb}")
                if len(b.matched_pairs) > 8:
                    lines.append(f"    ... and {len(b.matched_pairs) - 8} more")
            lines.append("")

        # Complexity
        if summary.complexity is not None:
            x = summary.complexity
            lines.append("── Behavioral Complexity ───────────────────────────")
            lines.append(f"  Response entropy       : {x.response_entropy:.4f}")
            lines.append(f"  Behavioral diversity   : {x.behavioral_diversity}")
            lines.append(f"  Vocabulary complexity  : {x.vocabulary_complexity:.4f}")
            lines.append(f"  Estimated states       : {x.estimated_states}")
            lines.append(f"  Myhill-Nerode lower bd : {x.myhill_nerode_lower_bound}")
            lines.append(f"  Distinguishing seqs    : {x.distinguishing_sequences_count}")
            lines.append("")

        # Overall
        lines.append(sep)
        lines.append(f"  OVERALL SCORE : {summary.overall_score:.4f}")
        lines.append(sep)
        lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _state_output_distribution(
        states: List[dict],
        state_id: str,
    ) -> Dict[str, float]:
        """Extract an output distribution for a state.

        Falls back to using ``dominant_behavior`` as a degenerate distribution
        when no explicit ``output_distribution`` key is present.
        """
        for s in states:
            if s.get("state_id") == state_id:
                if "output_distribution" in s:
                    return dict(s["output_distribution"])
                # Degenerate: single dominant behaviour
                return {s.get("dominant_behavior", "unknown"): 1.0}
        return {"unknown": 1.0}

    # ------------------------------------------------------------------
    # Convenience: build full summary from raw data
    # ------------------------------------------------------------------

    def build_summary(
        self,
        predictions: Optional[List[str]] = None,
        ground_truth: Optional[List[str]] = None,
        query_log: Optional[List[dict]] = None,
        tested_properties: Optional[List[str]] = None,
        all_properties: Optional[List[str]] = None,
        category_map: Optional[Dict[str, str]] = None,
        certificates: Optional[List[dict]] = None,
        ground_truth_properties: Optional[Dict[str, bool]] = None,
        automaton_a: Optional[dict] = None,
        automaton_b: Optional[dict] = None,
        responses: Optional[List[str]] = None,
    ) -> MetricsSummary:
        """One-shot computation of all available metrics."""
        summary = MetricsSummary()

        if predictions is not None and ground_truth is not None:
            summary.fidelity = self.compute_automaton_fidelity(
                predictions, ground_truth
            )

        if query_log is not None:
            summary.query_complexity = self.compute_query_complexity(query_log)

        if tested_properties is not None and all_properties is not None:
            summary.coverage = self.compute_coverage(
                tested_properties, all_properties, category_map
            )

        if certificates is not None and ground_truth_properties is not None:
            summary.soundness = self.compute_certificate_soundness(
                certificates, ground_truth_properties
            )

        if automaton_a is not None and automaton_b is not None:
            summary.bisimulation = self.compute_bisimulation_distance(
                automaton_a, automaton_b
            )

        if responses is not None:
            summary.complexity = self.compute_behavioral_complexity(responses)

        summary.overall_score = self.compute_overall_score(summary)
        return summary


# ---------------------------------------------------------------------------
# 4.  Tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import traceback

    _passed = 0
    _failed = 0

    def _assert(cond: bool, msg: str = "") -> None:
        global _passed, _failed
        if cond:
            _passed += 1
        else:
            _failed += 1
            frame = traceback.extract_stack(limit=2)[0]
            print(f"  FAIL  line {frame.lineno}: {msg}")

    def _assert_close(a: float, b: float, tol: float = 1e-4, msg: str = "") -> None:
        _assert(abs(a - b) < tol, msg or f"{a} != {b} (tol={tol})")

    # ---- Test statistical utilities ----
    print("▸ Statistical utilities")

    # accuracy
    _assert_close(_accuracy(["a", "b", "a"], ["a", "b", "b"]), 2 / 3, msg="accuracy")

    # precision / recall / f1
    p, r, f1 = _precision_recall_f1(
        ["a", "a", "b", "b", "a"], ["a", "b", "b", "a", "a"], "a"
    )
    _assert(0 < p < 1, "precision in (0,1)")
    _assert(0 < r < 1, "recall in (0,1)")
    _assert(0 < f1 < 1, "f1 in (0,1)")

    # confusion matrix
    cm = _confusion_matrix(["x", "y", "x", "y"], ["x", "x", "y", "y"])
    _assert(cm["x"]["x"] == 1, "cm true-pos x")
    _assert(cm["y"]["y"] == 1, "cm true-pos y")
    _assert(cm["x"]["y"] == 1, "cm x predicted y")

    # Shannon entropy
    _assert_close(_shannon_entropy({"a": 1.0, "b": 1.0}), math.log(2), msg="entropy uniform 2")
    _assert_close(_shannon_entropy({"a": 1.0}), 0.0, msg="entropy single")

    # KL divergence
    _assert_close(_kl_divergence({"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.5}), 0.0, msg="kl same")
    _assert(_kl_divergence({"a": 1.0}, {"b": 1.0}) == float("inf"), "kl disjoint")

    # L1 distance
    _assert_close(_l1_distance({"a": 1}, {"a": 1}), 0.0, msg="l1 same")
    _assert_close(_l1_distance({"a": 1}, {"b": 1}), 2.0, msg="l1 disjoint")

    # Jaccard
    _assert_close(_jaccard_similarity({1, 2, 3}, {2, 3, 4}), 2 / 4, msg="jaccard")
    _assert_close(_jaccard_similarity(set(), set()), 1.0, msg="jaccard empty")

    # Bootstrap CI
    lo, hi = _bootstrap_confidence_interval([1.0, 2.0, 3.0, 4.0, 5.0])
    _assert(lo < hi, "bootstrap lo < hi")
    _assert(1.0 <= lo <= 5.0, "bootstrap lo in range")
    _assert(1.0 <= hi <= 5.0, "bootstrap hi in range")

    # Cohen's kappa
    kappa = _cohen_kappa(["a", "a", "b", "b"], ["a", "a", "b", "b"])
    _assert_close(kappa, 1.0, msg="kappa perfect agreement")
    kappa_rand = _cohen_kappa(["a", "b", "a", "b"], ["b", "a", "b", "a"])
    _assert(kappa_rand < 0.01, f"kappa random ~ 0, got {kappa_rand}")

    # Cramér's V
    cm_perfect = {"a": {"a": 50, "b": 0}, "b": {"a": 0, "b": 50}}
    _assert(_cramers_v(cm_perfect) > 0.9, "cramers_v perfect")
    cm_rand = {"a": {"a": 25, "b": 25}, "b": {"a": 25, "b": 25}}
    _assert_close(_cramers_v(cm_rand), 0.0, tol=0.01, msg="cramers_v random")

    # Additional helpers
    _assert_close(_cosine_similarity({"a": 1}, {"a": 1}), 1.0, msg="cosine identical")
    _assert_close(_weighted_mean([1.0, 2.0], [1.0, 1.0]), 1.5, msg="weighted mean")
    _assert(_median_absolute_deviation([1, 2, 3, 4, 5]) > 0, "MAD > 0")
    _assert_close(_gini_impurity({"a": 1.0}), 0.0, msg="gini pure")
    _assert_close(_gini_impurity({"a": 1.0, "b": 1.0}), 0.5, msg="gini uniform 2")
    _assert_close(_harmonic_mean(1.0, 1.0), 1.0, msg="harmonic mean")
    _assert_close(_safe_div(1.0, 0.0, default=-1.0), -1.0, msg="safe_div by zero")
    _assert_close(_clamp(1.5), 1.0, msg="clamp high")
    _assert_close(_clamp(-0.5), 0.0, msg="clamp low")

    print(f"  {_passed} passed, {_failed} failed")
    _stat_fail = _failed
    _passed = _failed = 0

    # ---- Test fidelity ----
    print("▸ Automaton fidelity")
    ev = EvaluationMetrics()
    preds = ["a"] * 40 + ["b"] * 30 + ["c"] * 30
    truth = ["a"] * 35 + ["b"] * 5 + ["b"] * 25 + ["c"] * 5 + ["c"] * 28 + ["a"] * 2
    fid = ev.compute_automaton_fidelity(preds, truth)
    _assert(0.0 <= fid.prediction_accuracy <= 1.0, "fid accuracy range")
    _assert(len(fid.cross_val_scores) > 0, "fid has cv scores")
    _assert(0.0 <= fid.mean_cv_score <= 1.0, "fid mean_cv range")
    _assert(fid.std_cv_score >= 0.0, "fid std_cv >= 0")
    _assert(len(fid.confusion_matrix) > 0, "fid has confusion matrix")
    _assert("a" in fid.per_class_f1, "fid has f1 for class a")
    _assert(all(0.0 <= v <= 1.0 for v in fid.per_class_f1.values()), "fid f1 range")

    # Edge: small dataset
    fid_small = ev.compute_automaton_fidelity(["a", "b"], ["a", "b"], k_folds=2)
    _assert(fid_small.prediction_accuracy == 1.0, "fid small perfect")

    # Edge: single class
    fid_single = ev.compute_automaton_fidelity(["a"] * 10, ["a"] * 10)
    _assert_close(fid_single.prediction_accuracy, 1.0, msg="fid single class")

    print(f"  {_passed} passed, {_failed} failed")
    _fid_fail = _failed
    _passed = _failed = 0

    # ---- Test query complexity ----
    print("▸ Query complexity")
    ql = (
        [{"type": "membership", "content": "hello world", "num_states": 5}] * 20
        + [{"type": "equivalence", "content": "test eq"}] * 10
        + [{"type": "adversarial", "content": "adversarial prompt here"}] * 5
    )
    qcm = ev.compute_query_complexity(ql)
    _assert(qcm.total_queries == 35, f"qc total {qcm.total_queries}")
    _assert(qcm.membership_queries == 20, "qc membership")
    _assert(qcm.equivalence_queries == 10, "qc equivalence")
    _assert(qcm.adversarial_queries == 5, "qc adversarial")
    _assert(qcm.queries_per_state == 35 / 5, "qc per state")
    _assert(qcm.query_entropy > 0, "qc entropy > 0")
    _assert(qcm.average_query_length > 0, "qc avg length > 0")

    # Edge: empty log
    qcm_empty = ev.compute_query_complexity([])
    _assert(qcm_empty.total_queries == 0, "qc empty total")
    _assert_close(qcm_empty.query_entropy, 0.0, msg="qc empty entropy")

    print(f"  {_passed} passed, {_failed} failed")
    _qc_fail = _failed
    _passed = _failed = 0

    # ---- Test coverage ----
    print("▸ Coverage metrics")
    all_props = ["p1", "p2", "p3", "p4", "p5"]
    tested = ["p1", "p3", "p5"]
    cat_map = {"p1": "safety", "p2": "safety", "p3": "fairness", "p4": "fairness", "p5": "robustness"}
    cov = ev.compute_coverage(tested, all_props, cat_map)
    _assert(cov.total_specs == 5, "cov total")
    _assert(cov.tested_specs == 3, "cov tested")
    _assert_close(cov.coverage_ratio, 0.6, msg="cov ratio")
    _assert(set(cov.uncovered_specs) == {"p2", "p4"}, "cov uncovered")
    _assert_close(cov.per_category_coverage["safety"], 0.5, msg="cov safety")
    _assert_close(cov.per_category_coverage["fairness"], 0.5, msg="cov fairness")
    _assert_close(cov.per_category_coverage["robustness"], 1.0, msg="cov robustness")

    # No category map
    cov2 = ev.compute_coverage(tested, all_props)
    _assert(cov2.per_category_coverage == {}, "cov no categories")

    # Full coverage
    cov3 = ev.compute_coverage(all_props, all_props)
    _assert_close(cov3.coverage_ratio, 1.0, msg="cov full")
    _assert(cov3.uncovered_specs == [], "cov full uncovered empty")

    print(f"  {_passed} passed, {_failed} failed")
    _cov_fail = _failed
    _passed = _failed = 0

    # ---- Test certificate soundness ----
    print("▸ Certificate soundness")
    certs = [
        {"property": "safe", "verdict": True},
        {"property": "fair", "verdict": True},
        {"property": "robust", "verdict": False},
        {"property": "private", "verdict": True},
        {"property": "explainable", "verdict": False},
    ]
    gt_props = {
        "safe": True,
        "fair": False,       # FP
        "robust": True,      # FN
        "private": True,
        "explainable": False,
    }
    snd = ev.compute_certificate_soundness(certs, gt_props)
    _assert(snd.total_certificates == 5, "snd total")
    _assert(snd.sound_certificates == 3, f"snd sound {snd.sound_certificates}")
    _assert_close(snd.soundness_rate, 0.6, msg="snd rate")
    _assert("fair" in snd.false_positive_specs, "snd fp fair")
    _assert("robust" in snd.false_negative_specs, "snd fn robust")

    # Perfect soundness
    certs_ok = [{"property": "a", "verdict": True}, {"property": "b", "verdict": False}]
    gt_ok = {"a": True, "b": False}
    snd_ok = ev.compute_certificate_soundness(certs_ok, gt_ok)
    _assert_close(snd_ok.soundness_rate, 1.0, msg="snd perfect")

    print(f"  {_passed} passed, {_failed} failed")
    _snd_fail = _failed
    _passed = _failed = 0

    # ---- Test bisimulation distance ----
    print("▸ Bisimulation distance")
    auto_a = {
        "states": [
            {"state_id": "s0", "dominant_behavior": "greeting"},
            {"state_id": "s1", "dominant_behavior": "farewell"},
            {"state_id": "s2", "dominant_behavior": "question"},
        ],
        "transitions": [
            {"source": "s0", "target": "s1", "symbol": "bye", "probability": 0.8},
            {"source": "s0", "target": "s2", "symbol": "ask", "probability": 0.2},
            {"source": "s1", "target": "s0", "symbol": "hi", "probability": 1.0},
            {"source": "s2", "target": "s0", "symbol": "hi", "probability": 0.5},
            {"source": "s2", "target": "s1", "symbol": "bye", "probability": 0.5},
        ],
    }
    # Identical automaton → distance should be 0
    bsim = ev.compute_bisimulation_distance(auto_a, auto_a)
    _assert_close(bsim.state_matching_accuracy, 1.0, msg="bsim self match")
    _assert_close(bsim.transition_distance, 0.0, msg="bsim self trans")
    _assert_close(bsim.output_distance, 0.0, msg="bsim self output")
    _assert_close(bsim.overall_distance, 0.0, msg="bsim self overall")
    _assert(len(bsim.matched_pairs) == 3, "bsim self pairs")

    # Different automaton
    auto_b = {
        "states": [
            {"state_id": "q0", "dominant_behavior": "greeting"},
            {"state_id": "q1", "dominant_behavior": "farewell"},
        ],
        "transitions": [
            {"source": "q0", "target": "q1", "symbol": "bye", "probability": 1.0},
            {"source": "q1", "target": "q0", "symbol": "hi", "probability": 1.0},
        ],
    }
    bsim2 = ev.compute_bisimulation_distance(auto_a, auto_b)
    _assert(0 < bsim2.overall_distance, "bsim diff > 0")
    _assert(bsim2.state_matching_accuracy < 1.0, "bsim diff match < 1")

    print(f"  {_passed} passed, {_failed} failed")
    _bsim_fail = _failed
    _passed = _failed = 0

    # ---- Test behavioral complexity ----
    print("▸ Behavioral complexity")
    responses = [
        "Yes, that is correct.",
        "No, I disagree with that.",
        "The answer is 42.",
        "Why would you ask that?",
        "Please try again later.",
        "Because the sun is a star.",
        "Sure, I can help with that.",
        "It depends on the context.",
        "3.14159 is the value of pi.",
        "Hello, how can I help you?",
        "That is an interesting question.",
        "I'm not sure about that.",
        "Let me think about it.",
        "The quick brown fox jumps.",
        "No way that can be right.",
    ]
    cplx = ev.compute_behavioral_complexity(responses)
    _assert(cplx.response_entropy > 0, "complexity entropy > 0")
    _assert(cplx.behavioral_diversity > 0, "complexity diversity > 0")
    _assert(0 < cplx.vocabulary_complexity <= 1.0, "complexity vocab in (0,1]")
    _assert(cplx.estimated_states > 0, "complexity estimated states > 0")
    _assert(cplx.myhill_nerode_lower_bound >= 1, "complexity MN >= 1")
    _assert(cplx.distinguishing_sequences_count > 0, "complexity dist seqs > 0")

    # Edge: empty responses
    cplx_empty = ev.compute_behavioral_complexity([])
    _assert(cplx_empty.response_entropy == 0.0, "complexity empty entropy")
    _assert(cplx_empty.behavioral_diversity == 0, "complexity empty diversity")

    # Single response
    cplx_single = ev.compute_behavioral_complexity(["Hello"])
    _assert(cplx_single.behavioral_diversity >= 1, "complexity single diversity >= 1")

    print(f"  {_passed} passed, {_failed} failed")
    _cplx_fail = _failed
    _passed = _failed = 0

    # ---- Test overall score & comparison ----
    print("▸ Overall score & comparison")

    summary = ev.build_summary(
        predictions=preds,
        ground_truth=truth,
        query_log=ql,
        tested_properties=tested,
        all_properties=all_props,
        category_map=cat_map,
        certificates=certs,
        ground_truth_properties=gt_props,
        automaton_a=auto_a,
        automaton_b=auto_b,
        responses=responses,
    )
    _assert(0.0 <= summary.overall_score <= 1.0, f"overall range {summary.overall_score}")
    _assert(summary.fidelity is not None, "summary has fidelity")
    _assert(summary.query_complexity is not None, "summary has qc")
    _assert(summary.coverage is not None, "summary has coverage")
    _assert(summary.soundness is not None, "summary has soundness")
    _assert(summary.bisimulation is not None, "summary has bisimulation")
    _assert(summary.complexity is not None, "summary has complexity")

    # Comparison
    baseline = MetricsSummary(
        fidelity=FidelityMetrics(
            prediction_accuracy=0.5, cross_val_scores=[0.5], mean_cv_score=0.5,
            std_cv_score=0.0, confusion_matrix={}, per_class_f1={},
        ),
        coverage=CoverageMetrics(
            total_specs=5, tested_specs=2, coverage_ratio=0.4,
            uncovered_specs=["p3", "p4", "p5"], per_category_coverage={},
        ),
        soundness=CertificateSoundnessMetrics(
            total_certificates=5, sound_certificates=2, soundness_rate=0.4,
            false_positive_specs=[], false_negative_specs=[],
        ),
        bisimulation=BisimulationMetrics(
            state_matching_accuracy=0.5, transition_distance=0.5,
            output_distance=0.5, overall_distance=0.5, matched_pairs=[],
        ),
        overall_score=0.4,
    )
    comp = ev.compare_with_baseline(summary, baseline)
    _assert("fidelity_accuracy" in comp, "comp has fidelity")
    _assert("coverage_ratio" in comp, "comp has coverage")
    _assert("soundness_rate" in comp, "comp has soundness")
    _assert("bisimulation_distance" in comp, "comp has bisimulation")
    _assert("overall_score" in comp, "comp has overall")
    _assert(all("status" in v for v in comp.values()), "comp all have status")
    _assert(
        all("improvement_pct" in v for v in comp.values()),
        "comp all have improvement_pct",
    )

    # Compare with self → unchanged
    self_comp = ev.compare_with_baseline(summary, summary)
    _assert(
        all(v["status"] in ("unchanged", "unavailable") for v in self_comp.values()),
        "self-comparison all unchanged",
    )

    # Partial summaries
    partial = MetricsSummary(fidelity=fid)
    partial.overall_score = ev.compute_overall_score(partial)
    _assert(0.0 <= partial.overall_score <= 1.0, "partial score range")

    empty_summary = MetricsSummary()
    _assert_close(ev.compute_overall_score(empty_summary), 0.0, msg="empty score")

    print(f"  {_passed} passed, {_failed} failed")
    _score_fail = _failed
    _passed = _failed = 0

    # ---- Test report formatting ----
    print("▸ Report formatting")
    report = ev.format_report(summary)
    _assert(isinstance(report, str), "report is str")
    _assert(len(report) > 200, "report has content")
    _assert("OVERALL SCORE" in report, "report has overall")
    _assert("Fidelity" in report, "report has fidelity section")
    _assert("Coverage" in report, "report has coverage section")
    _assert("Soundness" in report, "report has soundness section")
    _assert("Bisimulation" in report, "report has bisimulation section")
    _assert("Complexity" in report, "report has complexity section")

    # Empty summary report
    empty_report = ev.format_report(MetricsSummary())
    _assert("OVERALL SCORE" in empty_report, "empty report has overall")

    print(f"  {_passed} passed, {_failed} failed")
    _rpt_fail = _failed

    # ---- Summary ----
    total_fail = _stat_fail + _fid_fail + _qc_fail + _cov_fail + _snd_fail + _bsim_fail + _cplx_fail + _score_fail + _rpt_fail
    print()
    if total_fail == 0:
        print("All tests passed ✓")
    else:
        print(f"{total_fail} test(s) FAILED")
        sys.exit(1)
