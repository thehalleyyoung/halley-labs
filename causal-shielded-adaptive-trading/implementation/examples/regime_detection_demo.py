#!/usr/bin/env python
"""
Regime detection demo for Causal-Shielded Adaptive Trading.

Demonstrates:
  1. Generate regime-switching time series with known regimes
  2. Fit a Sticky HDP-HMM and examine posterior regime assignments
  3. Visualise regime assignments (text-based)
  4. Estimate and display the transition matrix with credible intervals
  5. Online regime tracking with the sliding-window tracker
  6. Bayesian online change-point detection
"""

from __future__ import annotations

import sys
import time
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from causal_trading.market import SyntheticMarketGenerator
from causal_trading.regime import (
    StickyHDPHMM,
    BayesianRegimeDetector,
    OnlineRegimeTracker,
    TransitionMatrixEstimator,
)
from causal_trading.evaluation import RegimeAccuracyEvaluator

# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

SEED = 42
N_FEATURES = 8
N_REGIMES = 3
T = 2000


def print_header(title: str) -> None:
    width = 72
    print("\n" + "═" * width)
    print(f"  {title}")
    print("═" * width)


def print_section(title: str) -> None:
    print(f"\n── {title} {'─' * max(0, 60 - len(title))}")


def text_regime_plot(
    labels: np.ndarray,
    title: str = "Regimes",
    width: int = 70,
    max_t: int | None = None,
) -> None:
    """Print a text-based regime timeline."""
    symbols = "▓░▒█▄▀●○"
    if max_t is not None:
        labels = labels[:max_t]
    T_local = len(labels)
    n_regimes = int(labels.max()) + 1

    print(f"\n  {title} (T={T_local}, K={n_regimes})")
    print("  " + "─" * width)

    # Downsample to width
    bin_size = max(1, T_local // width)
    n_bins = T_local // bin_size

    for k in range(n_regimes):
        row = f"  R{k} │"
        for b in range(min(n_bins, width)):
            start = b * bin_size
            end = min(start + bin_size, T_local)
            frac = np.mean(labels[start:end] == k)
            if frac > 0.7:
                row += symbols[k % len(symbols)]
            elif frac > 0.3:
                row += "·"
            else:
                row += " "
        print(row + "│")

    # Time axis
    print("  " + " " * 4 + "└" + "─" * min(n_bins, width) + "┘")
    axis = "  " + " " * 5
    for p in [0, 25, 50, 75, 100]:
        pos = int(p / 100 * min(n_bins, width))
        t_val = int(p / 100 * T_local)
        label = str(t_val)
        pad = pos - len(axis.rstrip()) + len(axis) - len(axis.lstrip())
        axis = axis[:5 + pos] + label
        axis += " " * (5 + min(n_bins, width) - len(axis))
    print(f"  {'t':>4s}  0{' ' * (min(n_bins, width) - 5)}{T_local}")


def print_matrix(mat: np.ndarray, title: str, row_labels: List[str] | None = None) -> None:
    """Print a matrix with nice formatting."""
    n = mat.shape[0]
    m = mat.shape[1]
    if row_labels is None:
        row_labels = [f"  {i}" for i in range(n)]

    print(f"\n  {title}")
    # Column headers
    header = "       " + "".join(f"  {j:>6d}" for j in range(m))
    print(header)
    print("       " + "─" * (m * 8))
    for i in range(n):
        row = f"  {row_labels[i]:>3s} │"
        for j in range(m):
            row += f" {mat[i, j]:7.4f}"
        print(row)


# ═══════════════════════════════════════════════════════════════════════════
# Part 1: Generate Data
# ═══════════════════════════════════════════════════════════════════════════

def generate_regime_data():
    """Generate synthetic data with regime-switching dynamics."""
    print_header("Part 1: Generate Regime-Switching Data")

    gen = SyntheticMarketGenerator(
        n_features=N_FEATURES,
        n_regimes=N_REGIMES,
        regime_persistence=0.97,
        edge_density=0.12,
        invariant_ratio=0.4,
        fat_tail_df=5.0,
        use_garch=True,
        seed=SEED,
    )
    dataset = gen.generate(T=T)
    gt = dataset.ground_truth

    print(f"  Generated T={T} observations, p={N_FEATURES} features")
    print(f"  True regimes: K={gt.n_regimes}")

    counts = np.bincount(gt.regime_labels, minlength=N_REGIMES)
    for k in range(N_REGIMES):
        pct = 100.0 * counts[k] / T
        print(f"    Regime {k}: {counts[k]:5d} obs ({pct:5.1f}%)")

    print(f"\n  True transition matrix:")
    print_matrix(gt.regime_transition_matrix, "P_true")

    # Compute regime durations
    print_section("Regime Duration Statistics")
    durations: Dict[int, List[int]] = {k: [] for k in range(N_REGIMES)}
    current_regime = gt.regime_labels[0]
    current_duration = 1
    for t in range(1, T):
        if gt.regime_labels[t] == current_regime:
            current_duration += 1
        else:
            durations[current_regime].append(current_duration)
            current_regime = gt.regime_labels[t]
            current_duration = 1
    durations[current_regime].append(current_duration)

    for k in range(N_REGIMES):
        if durations[k]:
            d = np.array(durations[k])
            print(f"  Regime {k}: mean={d.mean():.1f}, median={np.median(d):.0f}, "
                  f"max={d.max()}, n_visits={len(d)}")

    text_regime_plot(gt.regime_labels, "True Regime Assignments")

    return dataset


# ═══════════════════════════════════════════════════════════════════════════
# Part 2: Sticky HDP-HMM
# ═══════════════════════════════════════════════════════════════════════════

def fit_sticky_hdp_hmm(dataset):
    """Fit a Sticky HDP-HMM to the data."""
    print_header("Part 2: Fit Sticky HDP-HMM")

    hmm = StickyHDPHMM(
        K_max=10,
        alpha=5.0,
        gamma=3.0,
        kappa=50.0,
    )

    print("  Fitting Sticky HDP-HMM (K_max=10, κ=50.0)...")
    t0 = time.time()
    hmm.fit(dataset.features, verbose=False)
    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s")

    # Predict regime assignments
    predicted = hmm.predict(dataset.features)
    n_active = len(set(predicted))
    print(f"  Active regimes discovered: {n_active}")

    counts = np.bincount(predicted, minlength=n_active)
    for k in range(n_active):
        pct = 100.0 * counts[k] / T
        print(f"    Regime {k}: {counts[k]:5d} obs ({pct:5.1f}%)")

    text_regime_plot(predicted, "HDP-HMM Predicted Regimes")

    # Posterior probabilities
    posteriors = hmm.predict_proba(dataset.features)
    print(f"\n  Posterior shape: {posteriors.shape}")
    print(f"  Mean max-posterior confidence: {posteriors.max(axis=1).mean():.4f}")

    # Show a few time-step posteriors
    print_section("Sample Posterior Probabilities")
    sample_times = [0, T // 4, T // 2, 3 * T // 4, T - 1]
    for t in sample_times:
        probs = posteriors[t, :n_active]
        prob_str = ", ".join(f"{p:.3f}" for p in probs)
        pred_k = predicted[t]
        print(f"  t={t:5d}: [{prob_str}]  →  regime {pred_k}")

    # Evaluate accuracy
    print_section("Accuracy Against Ground Truth")
    gt = dataset.ground_truth
    evaluator = RegimeAccuracyEvaluator()
    metrics = evaluator.evaluate(
        true_regimes=gt.regime_labels,
        predicted_regimes=predicted,
        delay_tolerance=5,
    )
    print(f"  Adjusted Rand Index:     {metrics.adjusted_rand_index:.4f}")
    print(f"  Normalised Mutual Info:  {metrics.normalized_mutual_info:.4f}")
    print(f"  V-measure:               {metrics.v_measure:.4f}")
    print(f"  Homogeneity:             {metrics.homogeneity:.4f}")
    print(f"  Completeness:            {metrics.completeness:.4f}")
    print(f"  Mean detection delay:    {metrics.mean_detection_delay:.1f} steps")

    return hmm, predicted


# ═══════════════════════════════════════════════════════════════════════════
# Part 3: Transition Matrix Estimation
# ═══════════════════════════════════════════════════════════════════════════

def estimate_transition_matrix(predicted, n_active):
    """Estimate transition matrix with Bayesian posterior."""
    print_header("Part 3: Bayesian Transition Matrix Estimation")

    estimator = TransitionMatrixEstimator(
        n_states=n_active,
        prior_alpha=1.0,
        sticky_kappa=10.0,
        n_posterior_samples=2000,
        random_state=SEED,
    )

    estimator.fit(predicted)

    # MAP estimate
    map_est = estimator.get_map_estimate()
    print_matrix(map_est, "MAP Transition Matrix")

    # Posterior mean
    post_mean = estimator.get_posterior_mean()
    print_matrix(post_mean, "Posterior Mean Transition Matrix")

    # Credible intervals
    ci = estimator.credible_intervals(level=0.95)
    print_section("95% Credible Intervals (selected entries)")
    for i in range(min(n_active, 3)):
        for j in range(min(n_active, 3)):
            lo, hi = ci[i, j]
            mean_val = post_mean[i, j]
            print(f"  P({i}→{j}): {mean_val:.4f} [{lo:.4f}, {hi:.4f}]")

    # Stationary distribution
    stationary = estimator.stationary_distribution()
    print(f"\n  Stationary distribution: [{', '.join(f'{p:.4f}' for p in stationary)}]")

    # Mixing time and spectral gap
    mixing = estimator.mixing_time(epsilon=0.25)
    spectral = estimator.spectral_gap()
    print(f"  Mixing time (ε=0.25):   {mixing:.1f}")
    print(f"  Spectral gap:           {spectral:.4f}")
    print(f"  Ergodic:                {estimator.is_ergodic()}")
    print(f"  Reversible:             {estimator.is_reversible()}")

    return estimator


# ═══════════════════════════════════════════════════════════════════════════
# Part 4: Online Regime Tracking
# ═══════════════════════════════════════════════════════════════════════════

def online_tracking_demo(dataset):
    """Demonstrate online regime tracking."""
    print_header("Part 4: Online Regime Tracking")

    tracker = OnlineRegimeTracker(
        n_regimes=N_REGIMES,
        window_size=200,
        forgetting_factor=0.98,
        alert_threshold=0.7,
        transition_prior=0.8,
        random_state=SEED,
    )

    gt = dataset.ground_truth
    T_online = min(T, 500)  # track first 500 steps

    print(f"  Tracking {T_online} observations online...")
    online_predictions = np.zeros(T_online, dtype=int)
    regime_change_times = []
    prev_regime = -1

    for t in range(T_online):
        x_t = float(dataset.features[t, 0])
        posterior = tracker.update(x_t)
        current_regime = np.argmax(posterior)
        online_predictions[t] = current_regime

        if current_regime != prev_regime and t > 0:
            regime_change_times.append(t)
        prev_regime = current_regime

    print(f"  Regime changes detected: {len(regime_change_times)}")
    if regime_change_times:
        print(f"  Change-point times: {regime_change_times[:15]}")
        if len(regime_change_times) > 15:
            print(f"    ... ({len(regime_change_times) - 15} more)")

    # Compare with ground truth
    true_changes = []
    for t in range(1, T_online):
        if gt.regime_labels[t] != gt.regime_labels[t - 1]:
            true_changes.append(t)

    print(f"\n  True regime changes in [0, {T_online}): {len(true_changes)}")
    if true_changes:
        print(f"  True change-point times: {true_changes[:15]}")

    # Detection delay analysis
    if true_changes and regime_change_times:
        delays = []
        for tc in true_changes:
            closest = min(regime_change_times, key=lambda x: abs(x - tc))
            if closest >= tc:
                delays.append(closest - tc)
        if delays:
            print(f"\n  Detection delay statistics:")
            print(f"    Mean:   {np.mean(delays):.1f} steps")
            print(f"    Median: {np.median(delays):.0f} steps")
            print(f"    Max:    {np.max(delays)} steps")

    text_regime_plot(online_predictions, "Online Regime Tracking", max_t=T_online)
    text_regime_plot(gt.regime_labels, "True Regimes", max_t=T_online)

    return tracker, online_predictions


# ═══════════════════════════════════════════════════════════════════════════
# Part 5: Bayesian Change-Point Detection
# ═══════════════════════════════════════════════════════════════════════════

def changepoint_detection_demo(dataset):
    """Demonstrate Bayesian online change-point detection."""
    print_header("Part 5: Bayesian Online Change-Point Detection")

    detector = BayesianRegimeDetector(
        n_regimes=N_REGIMES,
        n_particles=500,
        hazard_rate=1.0 / 200.0,
        resample_threshold=0.5,
        random_state=SEED,
    )

    gt = dataset.ground_truth
    T_detect = min(T, 1000)
    features_1d = dataset.features[:T_detect, :1]

    print(f"  Running Bayesian change-point detection (T={T_detect})...")
    t0 = time.time()
    states, posteriors = detector.detect(features_1d)
    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s")

    n_detected_regimes = len(set(states))
    print(f"  Detected {n_detected_regimes} regimes")

    counts = np.bincount(states)
    for k in range(len(counts)):
        if counts[k] > 0:
            print(f"    Regime {k}: {counts[k]:5d} obs ({100.0 * counts[k] / T_detect:5.1f}%)")

    # Posterior entropy as a measure of uncertainty
    print_section("Posterior Uncertainty Over Time")
    n_regimes_post = posteriors.shape[1]
    entropy = -np.sum(posteriors * np.log(posteriors + 1e-12), axis=1)
    max_entropy = np.log(n_regimes_post)

    # Show entropy at sampled time points
    sample_times = np.linspace(0, T_detect - 1, 10, dtype=int)
    for t in sample_times:
        bar_len = int(30 * entropy[t] / max_entropy)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  t={t:5d}  H={entropy[t]:.3f}  |{bar}|  regime={states[t]}")

    # Change-point probability
    print_section("Change-Point Probability")
    cp_prob = np.zeros(T_detect)
    for t in range(1, T_detect):
        cp_prob[t] = 1.0 - posteriors[t, states[t - 1]] if states[t - 1] < n_regimes_post else 0.5

    # Find peaks in change-point probability
    threshold = 0.5
    change_points = []
    for t in range(1, T_detect):
        if cp_prob[t] > threshold:
            if not change_points or t - change_points[-1] > 5:
                change_points.append(t)

    print(f"  Detected change points (P > {threshold}): {len(change_points)}")
    for cp in change_points[:20]:
        true_label = gt.regime_labels[cp] if cp < len(gt.regime_labels) else "?"
        print(f"    t={cp:5d}  P(change)={cp_prob[cp]:.3f}  true_regime={true_label}")

    # Compare to ground truth
    true_cps = [t for t in range(1, T_detect)
                if gt.regime_labels[t] != gt.regime_labels[t - 1]]
    print(f"\n  True change points: {len(true_cps)}")

    # Precision / recall for change-point detection
    if true_cps and change_points:
        tolerance = 10
        tp = 0
        matched_true = set()
        for cp in change_points:
            for tcp in true_cps:
                if abs(cp - tcp) <= tolerance and tcp not in matched_true:
                    tp += 1
                    matched_true.add(tcp)
                    break

        precision = tp / len(change_points) if change_points else 0
        recall = tp / len(true_cps) if true_cps else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"  Change-point detection (tolerance={tolerance}):")
        print(f"    Precision: {precision:.4f}")
        print(f"    Recall:    {recall:.4f}")
        print(f"    F1:        {f1:.4f}")

    text_regime_plot(states, "Bayesian Change-Point Detector Output", max_t=T_detect)

    return detector, states


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Run all regime detection demonstrations."""
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║     Causal-Shielded Adaptive Trading — Regime Detection    ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # Step 1: Generate data
    dataset = generate_regime_data()

    # Step 2: Sticky HDP-HMM
    hmm, predicted = fit_sticky_hdp_hmm(dataset)
    n_active = len(set(predicted))

    # Step 3: Transition matrix
    estimate_transition_matrix(predicted, n_active)

    # Step 4: Online tracking
    online_tracking_demo(dataset)

    # Step 5: Change-point detection
    changepoint_detection_demo(dataset)

    print("\n" + "═" * 72)
    print("  Regime detection demo complete.")
    print("═" * 72 + "\n")


if __name__ == "__main__":
    main()
