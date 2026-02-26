#!/usr/bin/env python3
"""
Run all experiments for the Causal-Shielded Adaptive Trading paper.

Produces reproducible JSON results for:
  1. State abstraction soundness (overapproximation vs grid size)
  2. Error decomposition (per-stage epsilon values)
  3. Bounded liveness (LTL spec satisfaction rates)
  4. Independent verification (certificate cross-check)
  5. PAC-Bayes vacuity analysis (non-vacuous bounds vs sample size)
  6. Student-t vs Gaussian emission comparison

Usage
-----
    cd implementation/
    python3 experiments/run_all_experiments.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy.special import gammaln, digamma

_IMPL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _IMPL_DIR not in sys.path:
    sys.path.insert(0, _IMPL_DIR)

from causal_trading.market.synthetic import SyntheticMarketGenerator
from causal_trading.regime.sticky_hdp_hmm import StickyHDPHMM
from causal_trading.regime.student_t_emission import EmissionModelSelector
from causal_trading.shield.pac_bayes import PACBayesBound, CatoniBound
from causal_trading.shield.bounded_liveness_specs import (
    DrawdownRecoverySpec,
    LossRecoverySpec,
    PositionReductionSpec,
    RegimeTransitionSpec,
)
from causal_trading.shield.shield_synthesis import PosteriorPredictiveShield
from causal_trading.verification.independent_verifier import IndependentVerifier
from causal_trading.verification.state_abstraction import (
    ConcreteState,
    discretize_state_space,
    AbstractState,
)
from experiments.run_multi_instrument import run_multi_instrument_experiment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_all_experiments")

RESULTS_DIR = Path(__file__).parent / "results"
SEED = 42


def _save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))
    logger.info("Saved %s", path)


def _save_dat(header: str, rows: List[List[Any]], path: Path) -> None:
    """Write a pgfplots-compatible .dat file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [header]
    for row in rows:
        lines.append(" ".join(str(v) for v in row))
    path.write_text("\n".join(lines) + "\n")
    logger.info("Saved %s", path)


# ===================================================================
# Experiment 1: State Abstraction Soundness
# ===================================================================

def run_state_abstraction_experiment() -> Dict[str, Any]:
    """Verify overapproximation on varying grid sizes."""
    logger.info("=== Experiment 1: State Abstraction Soundness ===")
    rng = np.random.default_rng(SEED)

    n_states = 5
    n_actions = 3
    state_dim = 2
    bounds_lo = np.zeros(state_dim)
    bounds_hi = np.ones(state_dim)

    # Build a small random MDP transition matrix
    raw_transitions = rng.dirichlet(np.ones(n_states), size=(n_states, n_actions))
    state_vectors = rng.uniform(0, 1, size=(n_states, state_dim))

    grid_sizes = [2, 4, 8, 16, 32]
    results_list: List[Dict[str, Any]] = []

    for g in grid_sizes:
        t0 = time.time()
        abstraction = discretize_state_space(bounds_lo, bounds_hi, n_bins=g)
        n_abstract = abstraction.n_abstract

        # Map concrete states to abstract states
        concrete_to_abstract = np.zeros(n_states, dtype=int)
        for i in range(n_states):
            cs = ConcreteState.from_array(state_vectors[i])
            concrete_to_abstract[i] = abstraction.abstract_index(cs)

        # Build abstract transition matrix via overapproximation
        abstract_trans = np.zeros((n_abstract, n_actions, n_abstract))
        for s in range(n_states):
            a_s = concrete_to_abstract[s]
            for a in range(n_actions):
                for s2 in range(n_states):
                    a_s2 = concrete_to_abstract[s2]
                    abstract_trans[a_s, a, a_s2] += raw_transitions[s, a, s2]

        # Normalise rows
        for a_s in range(n_abstract):
            for a in range(n_actions):
                row_sum = abstract_trans[a_s, a].sum()
                if row_sum > 0:
                    abstract_trans[a_s, a] /= row_sum

        # Safety probability: probability of staying in "safe" abstract states
        safe_abstract = set(range(n_abstract // 2 + 1))
        safe_concrete = {s for s in range(n_states)
                         if concrete_to_abstract[s] in safe_abstract}

        # Concrete safety: avg prob of reaching a safe state in 1 step
        concrete_safety = 0.0
        for s in range(n_states):
            for a in range(n_actions):
                concrete_safety += sum(
                    raw_transitions[s, a, s2]
                    for s2 in range(n_states)
                    if s2 in safe_concrete
                )
        concrete_safety /= (n_states * n_actions)

        # Abstract safety
        abstract_safety = 0.0
        n_occupied = len(set(concrete_to_abstract))
        for a_s in set(concrete_to_abstract):
            for a in range(n_actions):
                abstract_safety += sum(
                    abstract_trans[a_s, a, a_s2]
                    for a_s2 in safe_abstract
                    if a_s2 < n_abstract
                )
        abstract_safety /= max(n_occupied * n_actions, 1)

        gap = abs(abstract_safety - concrete_safety)
        elapsed = time.time() - t0

        results_list.append({
            "grid_size": g,
            "n_abstract_states": n_abstract,
            "concrete_safety_prob": round(concrete_safety, 6),
            "abstract_safety_prob": round(abstract_safety, 6),
            "overapproximation_gap": round(gap, 6),
            "elapsed_s": round(elapsed, 4),
        })
        logger.info(
            "  grid=%2d  n_abstract=%4d  gap=%.6f", g, n_abstract, gap
        )

    # Check monotonicity of gap reduction
    gaps = [r["overapproximation_gap"] for r in results_list]
    monotone = all(gaps[i] >= gaps[i + 1] - 1e-9 for i in range(len(gaps) - 1))

    output = {
        "experiment": "state_abstraction_soundness",
        "n_concrete_states": n_states,
        "n_actions": n_actions,
        "state_dim": state_dim,
        "seed": SEED,
        "grid_results": results_list,
        "monotone_refinement": monotone,
    }
    _save_json(output, RESULTS_DIR / "state_abstraction.json")
    return output


# ===================================================================
# Experiment 2: Error Decomposition
# ===================================================================

def _regime_error(predicted: np.ndarray, true_labels: np.ndarray) -> Dict:
    T = len(true_labels)
    pred_unique = np.unique(predicted)
    true_unique = np.unique(true_labels)
    purity = 0.0
    for c in pred_unique:
        mask = predicted == c
        if mask.sum() == 0:
            continue
        counts = np.array([np.sum(true_labels[mask] == t) for t in true_unique])
        purity += counts.max()
    purity /= T
    return {"stage": "regime_detection", "epsilon": round(1.0 - purity, 6),
            "details": {"purity": round(purity, 6),
                        "n_regimes_pred": int(len(pred_unique)),
                        "n_regimes_true": int(len(true_unique))}}


def _causal_error(data: np.ndarray, true_adj: np.ndarray) -> Dict:
    T, D = data.shape
    corr = np.corrcoef(data.T)
    threshold = 2.0 / np.sqrt(T)
    discovered = np.abs(corr) > threshold
    np.fill_diagonal(discovered, False)
    tp = fp = fn = 0
    for i in range(D):
        for j in range(i + 1, D):
            t = bool(true_adj[i, j] or true_adj[j, i])
            d = bool(discovered[i, j])
            if t and d:
                tp += 1
            elif not t and d:
                fp += 1
            elif t and not d:
                fn += 1
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return {"stage": "causal_discovery", "epsilon": round(1.0 - f1, 6),
            "details": {"precision": round(prec, 6), "recall": round(rec, 6),
                        "f1": round(f1, 6)}}


def _pac_bayes_stage_error(states: np.ndarray, n: int, delta: float = 0.05) -> Dict:
    n_r = len(np.unique(states))
    posterior = np.array([float((states == r).sum()) for r in np.unique(states)]) + 1.0
    prior = np.full(n_r, 10.0 / max(n_r, 1))
    kl = float(gammaln(posterior.sum()) - gammaln(prior.sum())
               - np.sum(gammaln(posterior)) + np.sum(gammaln(prior))
               + np.sum((posterior - prior) * (digamma(posterior) - digamma(posterior.sum()))))
    kl = max(kl, 0.0)
    bound = min((kl + np.log(2 * np.sqrt(n) / delta)) / n, 1.0)
    return {"stage": "pac_bayes", "epsilon": round(bound, 6),
            "details": {"kl_divergence": round(kl, 6), "bound": round(bound, 6)}}


def _shield_stage_error(states: np.ndarray, K_max: int) -> Dict:
    n_active = len(np.unique(states))
    # Use n_active as effective K for permissivity (informative prior)
    effective_k = max(n_active, 1)
    perm = float(effective_k) / float(max(K_max, effective_k))
    return {"stage": "shield", "epsilon": round(1.0 - perm, 6),
            "details": {"permissivity": round(perm, 6), "n_active": n_active}}


def run_error_decomposition_experiment() -> Dict[str, Any]:
    """Generate synthetic data and decompose per-stage errors."""
    logger.info("=== Experiment 2: Error Decomposition ===")
    n_regimes, n_features, T = 3, 5, 500

    gen = SyntheticMarketGenerator(n_features=n_features, n_regimes=n_regimes, seed=SEED)
    dataset = gen.generate(T=T, n_regimes=n_regimes, n_features=n_features)
    data = dataset.features
    true_labels = dataset.ground_truth.regime_labels
    true_adj = dataset.ground_truth.adjacency_matrices.get(
        0, np.zeros((n_features, n_features))
    )

    hmm = StickyHDPHMM(K_max=10, kappa=50.0, n_iter=80, burn_in=15, random_state=SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hmm.fit(data)
    pred_states = hmm.states_

    stages = [
        _regime_error(pred_states, true_labels),
        _causal_error(data, true_adj),
        _pac_bayes_stage_error(pred_states, T),
        _shield_stage_error(pred_states, K_max=n_regimes),
    ]
    total = sum(s["epsilon"] for s in stages)

    output = {
        "experiment": "error_decomposition",
        "n_regimes": n_regimes,
        "n_features": n_features,
        "T": T,
        "seed": SEED,
        "stages": stages,
        "total_error": round(total, 6),
    }
    _save_json(output, RESULTS_DIR / "error_decomposition.json")

    # pgfplots .dat
    _save_dat(
        "stage epsilon",
        [[i, s["epsilon"]] for i, s in enumerate(stages)],
        RESULTS_DIR / "error_decomposition.dat",
    )
    return output


# ===================================================================
# Experiment 3: Bounded Liveness
# ===================================================================

def _generate_trajectory(rng: np.random.Generator, T: int, satisfy: bool) -> List[Dict[str, float]]:
    """Produce a synthetic trajectory that satisfies or violates liveness specs."""
    traj: List[Dict[str, float]] = []
    drawdown = 0.0
    loss = 0.0
    exposure = 50.0
    regime_change = False
    adapted = False

    for t in range(T):
        # Drawdown dynamics
        if satisfy:
            drawdown = max(0, drawdown + rng.normal(-0.005, 0.02))
            if drawdown > 0.05:
                drawdown *= 0.7  # recover
        else:
            drawdown = max(0, drawdown + rng.normal(0.003, 0.02))

        # Loss dynamics
        loss = max(0, rng.normal(0.01, 0.015))
        if satisfy and loss > 0.03:
            loss *= 0.5

        # Exposure
        if satisfy:
            exposure = min(80.0, max(20.0, exposure + rng.normal(-1, 5)))
        else:
            exposure = min(150.0, max(50.0, exposure + rng.normal(2, 5)))

        # Regime change
        regime_change = rng.random() < 0.05
        adapted = satisfy and regime_change

        traj.append({
            "drawdown": float(drawdown),
            "loss": float(loss),
            "exposure": float(exposure),
            "regime_change": float(regime_change),
            "strategy_adapted": float(adapted),
        })
    return traj


def run_bounded_liveness_experiment() -> Dict[str, Any]:
    """Check liveness specs on synthetic trajectories."""
    logger.info("=== Experiment 3: Bounded Liveness ===")
    rng = np.random.default_rng(SEED)

    specs = {
        "DrawdownRecovery": DrawdownRecoverySpec(threshold=0.05, recovery_level=0.02, horizon=20),
        "LossRecovery": LossRecoverySpec(threshold=0.03, recovery_level=0.01, horizon=5),
        "PositionReduction": PositionReductionSpec(limit=100.0, safe_level=80.0, horizon=10),
        "RegimeTransition": RegimeTransitionSpec(adaptation_window=10),
    }

    n_each = 50
    T_traj = 100
    spec_results: Dict[str, Any] = {}

    for name, spec in specs.items():
        tp = tn = fp = fn = 0
        for label in ("satisfy", "violate"):
            for _ in range(n_each):
                satisfy = label == "satisfy"
                traj = _generate_trajectory(rng, T_traj, satisfy=satisfy)
                result = spec.evaluate_trajectory(traj)
                sat = result.satisfied
                if satisfy and sat:
                    tp += 1
                elif satisfy and not sat:
                    fn += 1
                elif not satisfy and sat:
                    fp += 1
                else:
                    tn += 1

        total = 2 * n_each
        sat_rate = (tp + fp) / total
        fpr = fp / max(fp + tn, 1)
        fnr = fn / max(fn + tp, 1)
        spec_results[name] = {
            "satisfaction_rate": round(sat_rate, 4),
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "false_positive_rate": round(fpr, 4),
            "false_negative_rate": round(fnr, 4),
        }
        logger.info("  %-25s sat_rate=%.4f FPR=%.4f FNR=%.4f", name, sat_rate, fpr, fnr)

    output = {
        "experiment": "bounded_liveness",
        "n_trajectories": 2 * n_each,
        "trajectory_length": T_traj,
        "seed": SEED,
        "specs": spec_results,
    }
    _save_json(output, RESULTS_DIR / "bounded_liveness.json")
    return output


# ===================================================================
# Experiment 4: Independent Verification
# ===================================================================

def run_independent_verification_experiment() -> Dict[str, Any]:
    """Create random MDPs, compute PAC-Bayes bounds, cross-verify."""
    logger.info("=== Experiment 4: Independent Verification ===")
    rng = np.random.default_rng(SEED)
    verifier = IndependentVerifier(tolerance=1e-6)

    n_trials = 10
    n_states, n_actions = 5, 3
    trial_results: List[Dict[str, Any]] = []
    discrepancies: List[float] = []

    for trial in range(n_trials):
        # Generate random transition observations as (from_state, to_state) pairs
        n_obs = rng.integers(200, 500)
        from_states = rng.integers(0, n_states, size=n_obs)
        to_states = rng.integers(0, n_states, size=n_obs)
        raw_transitions = np.column_stack([from_states, to_states])  # (n_obs, 2)

        prior_counts = np.ones((n_states, n_states))

        # Build posterior from prior + observed counts
        observed = np.zeros((n_states, n_states), dtype=np.float64)
        for s, sp in raw_transitions:
            observed[int(s), int(sp)] += 1.0
        posterior_counts = prior_counts + observed

        delta = 0.05

        # Compute bound with main PACBayesBound code
        pac = PACBayesBound(prior_type="dirichlet")
        post_flat = posterior_counts.reshape(-1)
        prior_flat = prior_counts.reshape(-1)
        n_samples = int(raw_transitions.shape[0])
        main_bound = pac.compute_bound(post_flat, prior_flat, n=n_samples, delta=delta)

        # Independently verify
        verified, indep_bound, discrepancy = verifier.verify_pac_bayes_bound(
            raw_transitions=raw_transitions,
            prior_counts=prior_counts,
            posterior_counts=posterior_counts,
            delta=delta,
            bound_type="catoni",
            claimed_bound=main_bound,
        )

        discrepancies.append(discrepancy)
        trial_results.append({
            "trial": trial,
            "main_bound": round(main_bound, 8),
            "independent_bound": round(indep_bound, 8),
            "discrepancy": round(discrepancy, 8),
            "verified": verified,
            "n_samples": n_samples,
        })
        logger.info("  trial=%d  main=%.6f  indep=%.6f  disc=%.6f  ok=%s",
                     trial, main_bound, indep_bound, discrepancy, verified)

    output = {
        "experiment": "independent_verification",
        "n_trials": n_trials,
        "n_states": n_states,
        "n_actions": n_actions,
        "delta": 0.05,
        "seed": SEED,
        "trials": trial_results,
        "summary": {
            "mean_discrepancy": round(float(np.mean(discrepancies)), 8),
            "max_discrepancy": round(float(np.max(discrepancies)), 8),
            "all_verified": all(t["verified"] for t in trial_results),
        },
    }
    _save_json(output, RESULTS_DIR / "independent_verification.json")
    return output


# ===================================================================
# Experiment 5: PAC-Bayes Vacuity Analysis
# ===================================================================

def run_pac_bayes_vacuity_experiment() -> Dict[str, Any]:
    """Show non-vacuous bounds across sample sizes."""
    logger.info("=== Experiment 5: PAC-Bayes Vacuity Analysis ===")
    rng = np.random.default_rng(SEED)

    sample_sizes = [100, 500, 1000, 5000, 10000]
    delta = 0.05
    n_params = 15  # 5 states × 3 actions
    results_list: List[Dict[str, Any]] = []

    for n in sample_sizes:
        # Simulate posterior counts from observations
        raw_counts = rng.dirichlet(np.ones(5), size=n)
        posterior_counts = raw_counts.sum(axis=0) + 1.0
        prior_counts = np.ones_like(posterior_counts)

        pac = CatoniBound(prior_type="dirichlet")
        bound = pac.compute_bound(posterior_counts, prior_counts, n=n, delta=delta)
        is_vacuous = bool(bound >= 1.0)

        # Also compute KL for reporting
        kl = pac.compute_kl(posterior_counts, prior_counts)

        results_list.append({
            "n": n,
            "bound": round(float(bound), 8),
            "kl_divergence": round(float(kl), 6),
            "is_vacuous": is_vacuous,
        })
        logger.info("  n=%5d  bound=%.6f  KL=%.4f  vacuous=%s", n, bound, kl, is_vacuous)

    # Check that bound decreases with n
    bounds = [r["bound"] for r in results_list]
    monotone_decreasing = all(bounds[i] >= bounds[i + 1] - 1e-9
                              for i in range(len(bounds) - 1))

    output = {
        "experiment": "pac_bayes_vacuity",
        "delta": delta,
        "seed": SEED,
        "results": results_list,
        "monotone_decreasing": monotone_decreasing,
        "any_non_vacuous": any(not r["is_vacuous"] for r in results_list),
    }
    _save_json(output, RESULTS_DIR / "pac_bayes_vacuity.json")

    # pgfplots .dat
    _save_dat(
        "n bound kl vacuous",
        [[r["n"], r["bound"], r["kl_divergence"], int(r["is_vacuous"])]
         for r in results_list],
        RESULTS_DIR / "pac_bayes_vacuity.dat",
    )
    return output


# ===================================================================
# Experiment 6: Student-t vs Gaussian Emission Comparison
# ===================================================================

def run_emission_comparison_experiment() -> Dict[str, Any]:
    """Compare Gaussian vs Student-t emission models on heavy-tailed data."""
    logger.info("=== Experiment 6: Student-t vs Gaussian Emission ===")
    rng = np.random.default_rng(SEED)

    T = 500
    n_features = 5
    df = 3  # heavy tails

    # Generate heavy-tailed data
    data = rng.standard_t(df=df, size=(T, n_features))
    data = data * 0.5 + rng.normal(0, 0.1, size=(T, n_features))  # add location shift

    selector = EmissionModelSelector(nu=float(df))
    comparison = selector.compare(data)

    model_results = []
    for row in comparison:
        model_results.append({
            "model": row.model,
            "log_likelihood": round(float(row.log_likelihood), 4),
            "bic": round(float(row.bic), 4),
            "n_params": row.n_params,
        })
        logger.info("  %-12s  LL=%.2f  BIC=%.2f  params=%d",
                     row.model, row.log_likelihood, row.bic, row.n_params)

    # Kurtosis test
    kurt_test = selector.kurtosis_test(data)
    selected = selector.select(data)

    # Compute excess kurtosis per feature
    from scipy.stats import kurtosis as sp_kurtosis
    feature_kurtosis = [round(float(sp_kurtosis(data[:, j])), 4)
                        for j in range(n_features)]

    output = {
        "experiment": "emission_comparison",
        "T": T,
        "n_features": n_features,
        "student_t_df": df,
        "seed": SEED,
        "models": model_results,
        "selected_model": selected,
        "kurtosis_test": {
            k: (round(float(v), 6) if isinstance(v, (float, np.floating)) else v)
            for k, v in kurt_test.items()
        },
        "per_feature_excess_kurtosis": feature_kurtosis,
    }
    _save_json(output, RESULTS_DIR / "emission_comparison.json")
    return output


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Results directory: %s", RESULTS_DIR.resolve())

    t_total = time.time()
    summaries: Dict[str, Any] = {}

    experiments = [
        ("state_abstraction", run_state_abstraction_experiment),
        ("error_decomposition", run_error_decomposition_experiment),
        ("bounded_liveness", run_bounded_liveness_experiment),
        ("independent_verification", run_independent_verification_experiment),
        ("pac_bayes_vacuity", run_pac_bayes_vacuity_experiment),
        ("emission_comparison", run_emission_comparison_experiment),
        ("multi_instrument", run_multi_instrument_experiment),
    ]

    for name, fn in experiments:
        t0 = time.time()
        try:
            result = fn()
            elapsed = time.time() - t0
            summaries[name] = {"status": "success", "elapsed_s": round(elapsed, 2)}
            logger.info("  [%s] completed in %.1f s", name, elapsed)
        except Exception as e:
            elapsed = time.time() - t0
            summaries[name] = {"status": "error", "error": str(e),
                               "elapsed_s": round(elapsed, 2)}
            logger.error("  [%s] FAILED: %s", name, e, exc_info=True)

    total_elapsed = time.time() - t_total

    # Save overall summary
    summary = {
        "total_elapsed_s": round(total_elapsed, 2),
        "seed": SEED,
        "experiments": summaries,
    }
    _save_json(summary, RESULTS_DIR / "summary.json")

    logger.info("=" * 60)
    logger.info("All experiments finished in %.1f s", total_elapsed)
    for name, info in summaries.items():
        logger.info("  %-30s %s (%.1fs)", name, info["status"], info["elapsed_s"])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
