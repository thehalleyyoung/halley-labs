"""Ground truth labels and validation for benchmark scenarios.

Provides the canonical expected verdict for every benchmark scenario,
plus helper functions to validate detection results against these labels
and compute aggregate quality statistics.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from collusion_proof.evaluation.scenarios import (
    ScenarioSpec,
    generate_scenario_prices,
    get_all_scenarios,
    get_scenario_by_id,
)

# ---------------------------------------------------------------------------
# Ground truth label table
# ---------------------------------------------------------------------------

# scenario_id → (expected_verdict, confidence, description)
GROUND_TRUTH_LABELS: Dict[str, Tuple[str, float, str]] = {
    # --- Collusive (10) ---
    "col_01_calvano_duopoly": (
        "collusive", 0.95,
        "Calvano-style Q-learning converges to supra-competitive prices."
    ),
    "col_02_high_discount": (
        "collusive", 0.98,
        "Very high discount factor enables robust collusion."
    ),
    "col_03_memory1_ql": (
        "collusive", 0.90,
        "Memory-1 Q-learning sustains collusion via tit-for-tat punishment."
    ),
    "col_04_slow_convergence": (
        "collusive", 0.85,
        "Slow learner that eventually converges to collusive prices."
    ),
    "col_05_asymmetric_costs": (
        "collusive", 0.88,
        "Asymmetric cost firms still collude above the symmetric Nash."
    ),
    "col_06_three_player": (
        "collusive", 0.82,
        "Three players sustain collusion despite larger deviation incentive."
    ),
    "col_07_grim_trigger": (
        "collusive", 0.97,
        "Grim trigger maintains monopoly pricing indefinitely."
    ),
    "col_08_tit_for_tat": (
        "collusive", 0.93,
        "Tit-for-tat sustains collusion with brief punishment episodes."
    ),
    "col_09_dqn_collusion": (
        "collusive", 0.87,
        "DQN agents learn supra-competitive pricing via implicit coordination."
    ),
    "col_10_freq_coordination": (
        "collusive", 0.86,
        "Frequency-based coordination converges to near-monopoly."
    ),
    # --- Competitive (10) ---
    "comp_01_static_nash": (
        "competitive", 0.99,
        "Static Nash: price is always at equilibrium."
    ),
    "comp_02_random_pricing": (
        "competitive", 0.95,
        "Random pricing yields no supra-competitive structure."
    ),
    "comp_03_myopic_br": (
        "competitive", 0.95,
        "Myopic best response converges to Nash."
    ),
    "comp_04_low_discount_ql": (
        "competitive", 0.90,
        "Low discount Q-learning cannot sustain collusion."
    ),
    "comp_05_noisy_competitive": (
        "competitive", 0.92,
        "Noisy Nash play: mean price at Nash despite variance."
    ),
    "comp_06_bertrand_eq": (
        "competitive", 0.98,
        "Bertrand undercutting drives price to marginal cost."
    ),
    "comp_07_differentiated_comp": (
        "competitive", 0.88,
        "Differentiated product Nash equilibrium (above MC but competitive)."
    ),
    "comp_08_ucb_competitive": (
        "competitive", 0.90,
        "UCB bandit without opponent modelling converges to Nash."
    ),
    "comp_09_gradient_comp": (
        "competitive", 0.90,
        "Policy gradient optimising single-round profit reaches Nash."
    ),
    "comp_10_high_epsilon": (
        "competitive", 0.85,
        "High exploration prevents convergence to collusive prices."
    ),
    # --- Boundary (5) ---
    "bnd_01_near_nash": (
        "competitive", 0.65,
        "Prices slightly above Nash; hard to distinguish from competitive."
    ),
    "bnd_02_edgeworth_cycles": (
        "competitive", 0.60,
        "Edgeworth cycles look collusive on average but are competitive."
    ),
    "bnd_03_tacit_parallel": (
        "competitive", 0.70,
        "Parallel pricing from similar costs, not coordination."
    ),
    "bnd_04_cost_plus": (
        "competitive", 0.68,
        "Fixed cost-plus markup between Nash and monopoly."
    ),
    "bnd_05_demand_learning": (
        "competitive", 0.72,
        "Transient high prices during exploration converge to Nash."
    ),
    # --- Adversarial (5) ---
    "adv_01_randomised_collusion": (
        "collusive", 0.75,
        "Noise injected to mask collusion; mean still supra-competitive."
    ),
    "adv_02_alternating_strategy": (
        "collusive", 0.70,
        "Alternating collusive/competitive phases; net profit above Nash."
    ),
    "adv_03_delayed_punishment": (
        "collusive", 0.72,
        "Delayed punishment hides retaliation pattern."
    ),
    "adv_04_asymmetric_camouflage": (
        "collusive", 0.70,
        "Asymmetric price split yields supra-competitive joint profits."
    ),
    "adv_05_phase_shifting": (
        "collusive", 0.68,
        "Drifting collusive price to evade stationary window tests."
    ),
}


class GroundTruthValidator:
    """Validate detection results against ground truth labels."""

    def __init__(self) -> None:
        self.labels: Dict[str, Tuple[str, float, str]] = dict(GROUND_TRUTH_LABELS)

    def validate(
        self, scenario_id: str, predicted_verdict: str,
    ) -> Dict[str, Any]:
        """Check a single prediction against ground truth.

        Returns a dict with ``correct``, ``expected``, ``predicted``,
        ``label_confidence``, and ``description``.
        """
        if scenario_id not in self.labels:
            return {
                "correct": None,
                "expected": None,
                "predicted": predicted_verdict,
                "label_confidence": None,
                "description": "Unknown scenario",
                "error": f"No ground truth for scenario_id={scenario_id!r}",
            }

        expected, confidence, desc = self.labels[scenario_id]
        correct = predicted_verdict.lower() == expected.lower()

        return {
            "correct": correct,
            "expected": expected,
            "predicted": predicted_verdict,
            "label_confidence": confidence,
            "description": desc,
        }

    def validate_batch(
        self, predictions: Dict[str, str],
    ) -> Dict[str, Any]:
        """Validate a batch of predictions.

        Parameters
        ----------
        predictions : {scenario_id: predicted_verdict}

        Returns
        -------
        Dict with ``per_scenario`` results, ``accuracy``, ``correct``, ``total``.
        """
        per_scenario: Dict[str, Dict[str, Any]] = {}
        correct = 0
        total = 0

        for scenario_id, pred in predictions.items():
            result = self.validate(scenario_id, pred)
            per_scenario[scenario_id] = result
            if result["correct"] is not None:
                total += 1
                if result["correct"]:
                    correct += 1

        return {
            "per_scenario": per_scenario,
            "correct": correct,
            "total": total,
            "accuracy": correct / total if total > 0 else 0.0,
        }

    def get_label(self, scenario_id: str) -> Optional[str]:
        """Return the expected verdict for a scenario, or None."""
        entry = self.labels.get(scenario_id)
        return entry[0] if entry else None

    def get_all_labels(self) -> Dict[str, str]:
        """Return {scenario_id: expected_verdict} for all scenarios."""
        return {sid: entry[0] for sid, entry in self.labels.items()}

    def accuracy(self, predictions: Dict[str, str]) -> float:
        """Compute accuracy over a batch of predictions."""
        result = self.validate_batch(predictions)
        return result["accuracy"]

    def confusion_matrix(self, predictions: Dict[str, str]) -> np.ndarray:
        """Compute a 2×2 confusion matrix [[TN, FP], [FN, TP]].

        Positive class = collusive.
        """
        tp = fp = fn = tn = 0
        for scenario_id, pred in predictions.items():
            entry = self.labels.get(scenario_id)
            if entry is None:
                continue
            expected = entry[0]
            p_col = pred.lower() == "collusive"
            e_col = expected.lower() == "collusive"
            if e_col and p_col:
                tp += 1
            elif (not e_col) and p_col:
                fp += 1
            elif e_col and (not p_col):
                fn += 1
            else:
                tn += 1

        return np.array([[tn, fp], [fn, tp]])


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------

def generate_ground_truth_prices(
    scenario_id: str, seed: int = 42,
) -> Tuple[np.ndarray, str]:
    """Generate prices for a scenario and return with its ground truth label.

    Returns ``(prices, expected_verdict)``.
    """
    scenario = get_scenario_by_id(scenario_id)
    if scenario is None:
        raise ValueError(f"Unknown scenario_id: {scenario_id!r}")

    prices = generate_scenario_prices(scenario, seed=seed)
    entry = GROUND_TRUTH_LABELS.get(scenario_id)
    expected = entry[0] if entry else scenario.expected_verdict
    return prices, expected


def validate_type_i_error(
    competitive_predictions: List[str], alpha: float = 0.05,
) -> Dict[str, Any]:
    """Validate type I error control on competitive scenarios.

    Parameters
    ----------
    competitive_predictions : list of predicted verdicts for scenarios
        whose true label is competitive.
    alpha : nominal significance level.

    Returns
    -------
    Dict with ``observed_rate``, ``nominal_alpha``, ``controlled``,
    ``n_false_positives``, ``n_total``.
    """
    n = len(competitive_predictions)
    if n == 0:
        return {
            "observed_rate": 0.0, "nominal_alpha": alpha,
            "controlled": True, "n_false_positives": 0, "n_total": 0,
        }

    fp = sum(1 for p in competitive_predictions if p.lower() == "collusive")
    rate = fp / n

    # Use a binomial test to check whether the observed rate is consistent
    # with the nominal alpha (one-sided test: is rate > alpha?)
    from scipy import stats as sp_stats
    p_value = float(sp_stats.binom_test(fp, n, alpha, alternative="greater"))

    return {
        "observed_rate": rate,
        "nominal_alpha": alpha,
        "controlled": rate <= alpha or p_value > 0.05,
        "n_false_positives": fp,
        "n_total": n,
        "binomial_p_value": p_value,
    }


def validate_power(
    collusive_predictions: List[str], min_power: float = 0.8,
) -> Dict[str, Any]:
    """Validate detection power on collusive scenarios.

    Parameters
    ----------
    collusive_predictions : list of predicted verdicts for scenarios
        whose true label is collusive.
    min_power : minimum acceptable power.

    Returns
    -------
    Dict with ``observed_power``, ``min_power``, ``sufficient``,
    ``n_detected``, ``n_total``.
    """
    n = len(collusive_predictions)
    if n == 0:
        return {
            "observed_power": 0.0, "min_power": min_power,
            "sufficient": False, "n_detected": 0, "n_total": 0,
        }

    detected = sum(1 for p in collusive_predictions if p.lower() == "collusive")
    power = detected / n

    return {
        "observed_power": power,
        "min_power": min_power,
        "sufficient": power >= min_power,
        "n_detected": detected,
        "n_total": n,
    }
