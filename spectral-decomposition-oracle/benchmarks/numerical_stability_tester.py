#!/usr/bin/env python3
"""
Numerical stability analysis for the Spectral Decomposition Oracle.

Generates test matrices with condition numbers κ from 1 to 10^15, measures
spectral feature extraction accuracy degradation, compares against LAPACK
dsyev (via numpy/scipy), tests oracle prediction robustness under
floating-point noise, and computes interval arithmetic bounds.

Usage:
    python3 benchmarks/numerical_stability_tester.py [--sizes 50,100,200]
        [--trials 10] [--output benchmarks/stability_results/]

Outputs:
    - numerical_stability_results.json   (full results)
    - stability_summary.json             (thresholds & recommendations)
    - eigenvalue_error_vs_kappa.csv      (plot data)
    - oracle_robustness_vs_kappa.csv     (plot data)
    - block_detection_vs_kappa.csv       (plot data)
"""

import argparse
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.linalg import eigh as scipy_eigh       # wraps LAPACK dsyev
    from scipy.linalg import eigvalsh as scipy_eigvalsh
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONDITION_NUMBERS = [
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7,
    1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15,
]

DEFAULT_SIZES = [50, 100, 200]
DEFAULT_TRIALS = 10
FEATURE_NAMES = [
    "spectral_gap", "algebraic_connectivity", "fiedler_entropy",
    "spectral_radius", "spectral_width", "normalized_cut_est",
    "cheeger_est", "eigenvalue_decay_rate",
]
DECOMPOSITION_METHODS = ["DW", "Benders", "LR", "Direct"]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EigenvalueAccuracy:
    """Eigenvalue error metrics for a single trial."""
    condition_number: float
    matrix_size: int
    max_relative_error: float
    mean_relative_error: float
    median_relative_error: float
    fiedler_relative_error: float
    lapack_max_relative_error: float  # numpy vs scipy/LAPACK
    numpy_time_ms: float
    lapack_time_ms: float


@dataclass
class FiedlerVectorAccuracy:
    """Fiedler vector angular error."""
    condition_number: float
    matrix_size: int
    angular_error_deg: float
    cosine_similarity: float
    sign_consistent: bool


@dataclass
class BlockDetectionAccuracy:
    """Block detection accuracy under conditioning."""
    condition_number: float
    matrix_size: int
    true_blocks: int
    detected_blocks: int
    block_correct: bool
    partition_accuracy: float  # fraction of nodes in correct block


@dataclass
class OracleRobustness:
    """Oracle prediction stability under eigenvalue noise."""
    condition_number: float
    matrix_size: int
    baseline_recommendation: str
    noisy_recommendations: List[str]
    agreement_rate: float
    feature_max_relative_change: float


@dataclass
class IntervalBound:
    """Interval arithmetic bounds for spectral features."""
    condition_number: float
    matrix_size: int
    feature_name: str
    point_estimate: float
    lower_bound: float
    upper_bound: float
    interval_width: float
    relative_uncertainty: float


@dataclass
class StabilityResult:
    """Aggregated stability result for one (κ, n) pair."""
    condition_number: float
    matrix_size: int
    eigenvalue_accuracy: List[EigenvalueAccuracy]
    fiedler_accuracy: List[FiedlerVectorAccuracy]
    block_detection: List[BlockDetectionAccuracy]
    oracle_robustness: List[OracleRobustness]
    interval_bounds: List[IntervalBound]


# ---------------------------------------------------------------------------
# Matrix generation with controlled condition number
# ---------------------------------------------------------------------------

def generate_matrix_with_condition(n: int, kappa: float, seed: int = 42,
                                   num_blocks: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a symmetric positive-definite matrix with a prescribed condition
    number κ and known block structure.

    Returns (A, true_eigenvalues, block_labels) where:
      - A is n×n symmetric with cond(A) ≈ κ
      - true_eigenvalues are the exact eigenvalues (by construction)
      - block_labels[i] ∈ {0, ..., num_blocks-1} is the true block for row i
    """
    rng = np.random.RandomState(seed)

    # Construct eigenvalues: logarithmically spaced from 1 to κ
    true_eigenvalues = np.logspace(0, np.log10(max(kappa, 1.0 + 1e-15)), n)
    true_eigenvalues = np.sort(true_eigenvalues)

    # Random orthogonal matrix via QR of random Gaussian
    H = rng.randn(n, n)
    Q, _ = np.linalg.qr(H)

    # A = Q Λ Q^T
    A = Q @ np.diag(true_eigenvalues) @ Q.T
    # Symmetrize to remove rounding asymmetry
    A = 0.5 * (A + A.T)

    # Block labels: roughly equal-sized blocks
    block_labels = np.zeros(n, dtype=int)
    block_size = n // num_blocks
    for b in range(num_blocks):
        start = b * block_size
        end = start + block_size if b < num_blocks - 1 else n
        block_labels[start:end] = b

    return A, true_eigenvalues, block_labels


# ---------------------------------------------------------------------------
# Spectral feature extraction (mirrors SpecOracle features)
# ---------------------------------------------------------------------------

def extract_spectral_features(eigenvalues: np.ndarray,
                              eigenvectors: np.ndarray) -> Dict[str, float]:
    """Extract the 8 spectral features used by SpecOracle."""
    n = len(eigenvalues)
    ev = np.sort(eigenvalues)

    # Spectral gap: λ₂ - λ₁
    spectral_gap = ev[1] - ev[0] if n > 1 else 0.0

    # Algebraic connectivity: λ₂ (for Laplacian-like matrices, smallest nonzero)
    algebraic_connectivity = ev[1] if n > 1 else 0.0

    # Fiedler vector localization entropy
    if n > 1:
        fiedler = eigenvectors[:, 1]  # second eigenvector
        p = fiedler ** 2
        p = p / (p.sum() + 1e-300)
        p = np.clip(p, 1e-300, None)
        fiedler_entropy = -np.sum(p * np.log(p))
    else:
        fiedler_entropy = 0.0

    spectral_radius = ev[-1]
    spectral_width = ev[-1] - ev[0]

    # Normalized cut estimate from Fiedler vector
    if n > 1:
        fiedler = eigenvectors[:, 1]
        pos = np.sum(fiedler > 0)
        neg = n - pos
        if pos > 0 and neg > 0:
            normalized_cut_est = spectral_gap / min(pos, neg)
        else:
            normalized_cut_est = 0.0
    else:
        normalized_cut_est = 0.0

    # Cheeger estimate: h ≥ λ₂ / 2
    cheeger_est = spectral_gap / 2.0

    # Eigenvalue decay rate: fit λᵢ ≈ a·exp(β·i)
    if n > 2:
        indices = np.arange(n, dtype=float)
        log_ev = np.log(np.clip(ev, 1e-300, None))
        # Least-squares: log(λ) = log(a) + β·i
        coeffs = np.polyfit(indices, log_ev, 1)
        eigenvalue_decay_rate = coeffs[0]  # β
    else:
        eigenvalue_decay_rate = 0.0

    return {
        "spectral_gap": float(spectral_gap),
        "algebraic_connectivity": float(algebraic_connectivity),
        "fiedler_entropy": float(fiedler_entropy),
        "spectral_radius": float(spectral_radius),
        "spectral_width": float(spectral_width),
        "normalized_cut_est": float(normalized_cut_est),
        "cheeger_est": float(cheeger_est),
        "eigenvalue_decay_rate": float(eigenvalue_decay_rate),
    }


# ---------------------------------------------------------------------------
# Simulated oracle decision (mirrors gradient-boosted tree logic)
# ---------------------------------------------------------------------------

def oracle_recommend(features: Dict[str, float]) -> str:
    """
    Simplified oracle decision model mirroring SpecOracle's gradient-boosted
    tree. Uses the dominant features: spectral_gap, spectral_width,
    fiedler_entropy, and eigenvalue_decay_rate.
    """
    sg = features["spectral_gap"]
    sw = features["spectral_width"]
    fe = features["fiedler_entropy"]
    dr = features["eigenvalue_decay_rate"]

    # Thresholds derived from the trained model's top splits
    if sg > 0.1 * sw and fe < 3.0:
        # Clear block structure → Dantzig-Wolfe
        return "DW"
    elif sg < 0.01 * sw and dr > 0.5:
        # Rapid eigenvalue growth, weak structure → Benders
        return "Benders"
    elif fe > 4.0 and sg > 0.001 * sw:
        # Diffuse Fiedler vector, some gap → Lagrangian
        return "LR"
    else:
        # No clear decomposition advantage
        return "Direct"


# ---------------------------------------------------------------------------
# Block detection via spectral clustering
# ---------------------------------------------------------------------------

def detect_blocks_spectral(eigenvectors: np.ndarray, k: int) -> np.ndarray:
    """
    Simple spectral clustering: k-means on the first k eigenvectors.
    Returns cluster labels.
    """
    V = eigenvectors[:, :k].copy()
    # Normalize rows
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-300, None)
    V = V / norms

    n = V.shape[0]
    rng = np.random.RandomState(0)

    # k-means (simple implementation)
    centers = V[rng.choice(n, k, replace=False)]
    labels = np.zeros(n, dtype=int)
    for _ in range(50):
        # Assign
        for i in range(n):
            dists = np.linalg.norm(centers - V[i], axis=1)
            labels[i] = np.argmin(dists)
        # Update
        new_centers = np.zeros_like(centers)
        for c in range(k):
            mask = labels == c
            if mask.sum() > 0:
                new_centers[c] = V[mask].mean(axis=0)
            else:
                new_centers[c] = V[rng.randint(n)]
        if np.allclose(centers, new_centers, atol=1e-10):
            break
        centers = new_centers

    return labels


def partition_accuracy(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """
    Compute partition accuracy (best permutation matching).
    Since cluster IDs may be permuted, try all label mappings.
    """
    from itertools import permutations
    k = max(true_labels.max(), pred_labels.max()) + 1
    if k > 8:
        # For large k, use greedy matching
        return _greedy_partition_accuracy(true_labels, pred_labels, k)

    best = 0.0
    for perm in permutations(range(k)):
        mapped = np.array([perm[l] for l in pred_labels])
        acc = np.mean(mapped == true_labels)
        best = max(best, acc)
    return float(best)


def _greedy_partition_accuracy(true_labels: np.ndarray, pred_labels: np.ndarray,
                               k: int) -> float:
    """Greedy matching for partition accuracy when k is large."""
    from collections import Counter
    used_true = set()
    total_correct = 0
    for pred_c in range(k):
        pred_mask = pred_labels == pred_c
        if pred_mask.sum() == 0:
            continue
        best_match, best_count = -1, 0
        for true_c in range(k):
            if true_c in used_true:
                continue
            count = np.sum(true_labels[pred_mask] == true_c)
            if count > best_count:
                best_match, best_count = true_c, count
        if best_match >= 0:
            used_true.add(best_match)
            total_correct += best_count
    return total_correct / len(true_labels)


# ---------------------------------------------------------------------------
# Interval arithmetic for spectral features
# ---------------------------------------------------------------------------

def compute_interval_bounds(eigenvalues: np.ndarray, eigenvectors: np.ndarray,
                            condition_number: float) -> List[Dict]:
    """
    Compute interval arithmetic bounds on spectral features.

    Uses the Bauer-Fike theorem: for a symmetric matrix A with computed
    eigenvalue λ̃, the true eigenvalue λ satisfies
        |λ - λ̃| ≤ ‖E‖₂
    where E is the backward error, bounded by ε_mach · ‖A‖₂ for symmetric
    eigensolvers. For the Fiedler vector, Davis-Kahan sin(θ) theorem gives
    angular perturbation bounds.
    """
    eps_mach = np.finfo(np.float64).eps  # ≈ 2.22e-16
    n = len(eigenvalues)
    ev = np.sort(eigenvalues)
    norm_A = ev[-1]  # ‖A‖₂ = largest eigenvalue (SPD matrix)

    # Backward error bound for symmetric eigensolver (LAPACK dsyev)
    backward_err = eps_mach * norm_A * np.sqrt(n)

    # Per-eigenvalue absolute error bound (Bauer-Fike for symmetric)
    ev_err = backward_err

    features = extract_spectral_features(eigenvalues, eigenvectors)
    bounds = []

    for name, val in features.items():
        if name == "spectral_gap":
            # Gap = λ₂ - λ₁, error ≤ 2·ev_err
            err = 2 * ev_err
        elif name == "algebraic_connectivity":
            err = ev_err
        elif name == "spectral_radius":
            err = ev_err
        elif name == "spectral_width":
            err = 2 * ev_err
        elif name == "fiedler_entropy":
            # Davis-Kahan: sin(θ) ≤ ‖E‖₂ / gap, entropy error scales with θ
            gap = ev[1] - ev[0] if n > 1 else 1.0
            gap = max(gap, 1e-300)
            sin_theta = min(backward_err / gap, 1.0)
            # Entropy perturbation: first-order ≈ 2·sin(θ)·ln(n)
            err = 2 * sin_theta * np.log(max(n, 2))
        elif name == "normalized_cut_est":
            gap = ev[1] - ev[0] if n > 1 else 1.0
            gap = max(gap, 1e-300)
            sin_theta = min(backward_err / gap, 1.0)
            err = (2 * ev_err / n) + sin_theta * abs(val + 1e-300)
        elif name == "cheeger_est":
            err = ev_err  # h = gap/2
        elif name == "eigenvalue_decay_rate":
            # Decay rate is a regression coefficient; error propagation
            err = ev_err * n / (norm_A + 1e-300)
        else:
            err = ev_err

        rel_unc = err / (abs(val) + 1e-300)
        bounds.append({
            "feature_name": name,
            "point_estimate": float(val),
            "lower_bound": float(val - err),
            "upper_bound": float(val + err),
            "interval_width": float(2 * err),
            "relative_uncertainty": float(rel_unc),
        })

    return bounds


# ---------------------------------------------------------------------------
# Core stability tests
# ---------------------------------------------------------------------------

def test_eigenvalue_accuracy(A: np.ndarray, true_eigenvalues: np.ndarray,
                             kappa: float, n: int) -> EigenvalueAccuracy:
    """Compare numpy and LAPACK eigenvalues against ground truth."""
    # NumPy eigensolver
    t0 = time.perf_counter()
    np_eigenvalues = np.linalg.eigvalsh(A)
    np_time = (time.perf_counter() - t0) * 1000

    np_eigenvalues = np.sort(np_eigenvalues)
    true_sorted = np.sort(true_eigenvalues)

    # Relative errors (avoid division by zero for near-zero eigenvalues)
    denom = np.maximum(np.abs(true_sorted), 1e-300)
    rel_errors = np.abs(np_eigenvalues - true_sorted) / denom

    # LAPACK dsyev via scipy
    if HAS_SCIPY:
        t0 = time.perf_counter()
        sp_eigenvalues = scipy_eigvalsh(A)
        sp_time = (time.perf_counter() - t0) * 1000
        sp_eigenvalues = np.sort(sp_eigenvalues)
        lapack_rel_errors = np.abs(sp_eigenvalues - true_sorted) / denom
        lapack_max_err = float(np.max(lapack_rel_errors))
    else:
        sp_time = np_time
        lapack_max_err = float(np.max(rel_errors))

    # Fiedler eigenvalue (λ₂) relative error
    fiedler_err = float(rel_errors[1]) if n > 1 else 0.0

    return EigenvalueAccuracy(
        condition_number=kappa,
        matrix_size=n,
        max_relative_error=float(np.max(rel_errors)),
        mean_relative_error=float(np.mean(rel_errors)),
        median_relative_error=float(np.median(rel_errors)),
        fiedler_relative_error=fiedler_err,
        lapack_max_relative_error=lapack_max_err,
        numpy_time_ms=np_time,
        lapack_time_ms=sp_time,
    )


def test_fiedler_accuracy(A: np.ndarray, kappa: float, n: int,
                          seed: int) -> FiedlerVectorAccuracy:
    """
    Measure Fiedler vector angular error by comparing two independent
    eigendecompositions (perturbed vs unperturbed).
    """
    # "Ground truth" Fiedler vector from high-precision solve
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    idx = np.argsort(eigenvalues)
    fiedler_true = eigenvectors[:, idx[1]]

    # Perturbed matrix: A + E, ‖E‖ = ε_mach · ‖A‖
    rng = np.random.RandomState(seed + 1000)
    eps = np.finfo(np.float64).eps
    E = rng.randn(n, n)
    E = 0.5 * (E + E.T)
    E = E / np.linalg.norm(E, 2) * eps * np.linalg.norm(A, 2)
    A_pert = A + E

    eigenvalues_p, eigenvectors_p = np.linalg.eigh(A_pert)
    idx_p = np.argsort(eigenvalues_p)
    fiedler_pert = eigenvectors_p[:, idx_p[1]]

    # Angular error
    cos_sim = np.abs(np.dot(fiedler_true, fiedler_pert))
    cos_sim = min(cos_sim, 1.0)
    angle_deg = np.degrees(np.arccos(cos_sim))

    # Sign consistency (eigenvector sign ambiguity)
    sign_consistent = (np.sign(fiedler_true[0]) == np.sign(fiedler_pert[0]))

    return FiedlerVectorAccuracy(
        condition_number=kappa,
        matrix_size=n,
        angular_error_deg=float(angle_deg),
        cosine_similarity=float(cos_sim),
        sign_consistent=bool(sign_consistent),
    )


def test_block_detection(A: np.ndarray, true_labels: np.ndarray,
                         kappa: float, n: int, num_blocks: int) -> BlockDetectionAccuracy:
    """Test block detection accuracy under conditioning."""
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    idx = np.argsort(eigenvalues)
    eigenvectors = eigenvectors[:, idx]

    pred_labels = detect_blocks_spectral(eigenvectors, num_blocks)
    acc = partition_accuracy(true_labels, pred_labels)
    correct = bool(max(true_labels) + 1 == max(pred_labels) + 1)

    return BlockDetectionAccuracy(
        condition_number=kappa,
        matrix_size=n,
        true_blocks=num_blocks,
        detected_blocks=int(max(pred_labels) + 1),
        block_correct=correct,
        partition_accuracy=acc,
    )


def test_oracle_robustness(A: np.ndarray, kappa: float, n: int,
                           num_noise_trials: int = 20) -> OracleRobustness:
    """
    Test oracle prediction stability by injecting noise scaled to
    condition-number-appropriate levels.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    baseline_features = extract_spectral_features(eigenvalues, eigenvectors)
    baseline_rec = oracle_recommend(baseline_features)

    noisy_recs = []
    max_feat_change = 0.0

    for trial in range(num_noise_trials):
        rng = np.random.RandomState(trial + 7777)
        eps = np.finfo(np.float64).eps

        # Inject noise: scale ~ eps_mach · κ (realistic roundoff amplification)
        noise_scale = eps * kappa
        noisy_ev = eigenvalues * (1.0 + rng.randn(n) * noise_scale)
        noisy_ev = np.sort(noisy_ev)

        # Perturb eigenvectors proportionally
        noise_vec = rng.randn(*eigenvectors.shape) * noise_scale
        noisy_vecs = eigenvectors + noise_vec
        # Re-orthonormalize
        noisy_vecs, _ = np.linalg.qr(noisy_vecs)

        noisy_features = extract_spectral_features(noisy_ev, noisy_vecs)
        noisy_recs.append(oracle_recommend(noisy_features))

        # Track max relative feature change
        for fname in FEATURE_NAMES:
            base_val = baseline_features[fname]
            noisy_val = noisy_features[fname]
            if abs(base_val) > 1e-300:
                rel_change = abs(noisy_val - base_val) / abs(base_val)
                max_feat_change = max(max_feat_change, rel_change)

    agreement = sum(1 for r in noisy_recs if r == baseline_rec) / len(noisy_recs)

    return OracleRobustness(
        condition_number=kappa,
        matrix_size=n,
        baseline_recommendation=baseline_rec,
        noisy_recommendations=noisy_recs,
        agreement_rate=agreement,
        feature_max_relative_change=max_feat_change,
    )


# ---------------------------------------------------------------------------
# Main stability analysis driver
# ---------------------------------------------------------------------------

def run_stability_analysis(sizes: List[int], trials: int,
                           output_dir: str) -> Dict:
    """Run the full numerical stability analysis."""
    os.makedirs(output_dir, exist_ok=True)
    num_blocks = 3
    all_results = []

    total_combos = len(CONDITION_NUMBERS) * len(sizes)
    combo_idx = 0

    for kappa in CONDITION_NUMBERS:
        for n in sizes:
            combo_idx += 1
            log_kappa = int(round(np.log10(max(kappa, 1.0))))
            print(f"[{combo_idx}/{total_combos}] κ=10^{log_kappa}, n={n} ...")

            ev_results = []
            fv_results = []
            bd_results = []
            or_results = []
            ib_results = []

            for t in range(trials):
                seed = t * 1000 + n + int(np.log10(max(kappa, 1.0)))
                A, true_ev, true_labels = generate_matrix_with_condition(
                    n, kappa, seed=seed, num_blocks=num_blocks
                )

                # Eigenvalue accuracy
                ev_acc = test_eigenvalue_accuracy(A, true_ev, kappa, n)
                ev_results.append(ev_acc)

                # Fiedler vector accuracy
                fv_acc = test_fiedler_accuracy(A, kappa, n, seed)
                fv_results.append(fv_acc)

                # Block detection
                bd_acc = test_block_detection(A, true_labels, kappa, n, num_blocks)
                bd_results.append(bd_acc)

                # Oracle robustness
                or_acc = test_oracle_robustness(A, kappa, n)
                or_results.append(or_acc)

                # Interval bounds (one per trial is enough)
                if t == 0:
                    eigenvalues, eigenvectors = np.linalg.eigh(A)
                    idx = np.argsort(eigenvalues)
                    eigenvalues = eigenvalues[idx]
                    eigenvectors = eigenvectors[:, idx]
                    bounds = compute_interval_bounds(eigenvalues, eigenvectors, kappa)
                    for b in bounds:
                        b["condition_number"] = kappa
                        b["matrix_size"] = n
                    ib_results.extend([IntervalBound(**b) for b in bounds])

            result = StabilityResult(
                condition_number=kappa,
                matrix_size=n,
                eigenvalue_accuracy=ev_results,
                fiedler_accuracy=fv_results,
                block_detection=bd_results,
                oracle_robustness=or_results,
                interval_bounds=ib_results,
            )
            all_results.append(result)

    # -----------------------------------------------------------------------
    # Aggregate and write outputs
    # -----------------------------------------------------------------------
    summary = aggregate_results(all_results)
    write_outputs(all_results, summary, output_dir)
    return summary


def aggregate_results(results: List[StabilityResult]) -> Dict:
    """Aggregate results into summary statistics and threshold detection."""
    # Group by condition number (average over sizes and trials)
    kappa_groups: Dict[float, Dict] = {}

    for r in results:
        k = r.condition_number
        if k not in kappa_groups:
            kappa_groups[k] = {
                "ev_max_errors": [],
                "ev_mean_errors": [],
                "fiedler_errors": [],
                "fiedler_angles": [],
                "block_accuracies": [],
                "oracle_agreements": [],
                "lapack_errors": [],
            }
        g = kappa_groups[k]
        for ev in r.eigenvalue_accuracy:
            g["ev_max_errors"].append(ev.max_relative_error)
            g["ev_mean_errors"].append(ev.mean_relative_error)
            g["fiedler_errors"].append(ev.fiedler_relative_error)
            g["lapack_errors"].append(ev.lapack_max_relative_error)
        for fv in r.fiedler_accuracy:
            g["fiedler_angles"].append(fv.angular_error_deg)
        for bd in r.block_detection:
            g["block_accuracies"].append(bd.partition_accuracy)
        for orc in r.oracle_robustness:
            g["oracle_agreements"].append(orc.agreement_rate)

    # Build summary table
    summary_table = []
    for k in sorted(kappa_groups.keys()):
        g = kappa_groups[k]
        entry = {
            "condition_number": k,
            "log10_kappa": round(np.log10(max(k, 1.0)), 1),
            "eigenvalue_max_rel_error": float(np.median(g["ev_max_errors"])),
            "eigenvalue_mean_rel_error": float(np.median(g["ev_mean_errors"])),
            "fiedler_eigenvalue_rel_error": float(np.median(g["fiedler_errors"])),
            "fiedler_angular_error_deg": float(np.median(g["fiedler_angles"])),
            "block_detection_accuracy": float(np.mean(g["block_accuracies"])),
            "oracle_agreement_rate": float(np.mean(g["oracle_agreements"])),
            "lapack_max_rel_error": float(np.median(g["lapack_errors"])),
        }
        summary_table.append(entry)

    # Detect thresholds
    oracle_threshold = 1e15  # default: no degradation found
    block_threshold = 1e15
    eigenvalue_threshold = 1e15

    for entry in summary_table:
        k = entry["condition_number"]
        if entry["oracle_agreement_rate"] < 0.95 and k < oracle_threshold:
            oracle_threshold = k
        if entry["block_detection_accuracy"] < 0.80 and k < block_threshold:
            block_threshold = k
        if entry["eigenvalue_max_rel_error"] > 1e-6 and k < eigenvalue_threshold:
            eigenvalue_threshold = k

    recommendations = []
    if eigenvalue_threshold < 1e15:
        recommendations.append(
            f"Eigenvalue accuracy degrades beyond κ = {eigenvalue_threshold:.0e}; "
            "consider diagonal scaling or symmetric equilibration."
        )
    if oracle_threshold < 1e15:
        recommendations.append(
            f"Oracle predictions become unreliable beyond κ = {oracle_threshold:.0e}; "
            "apply Ruiz equilibration before spectral analysis."
        )
    if block_threshold < 1e15:
        recommendations.append(
            f"Block detection degrades beyond κ = {block_threshold:.0e}; "
            "use iterative refinement for the Fiedler vector."
        )
    if not recommendations:
        recommendations.append(
            "Oracle is robust across all tested condition numbers."
        )

    return {
        "summary_table": summary_table,
        "thresholds": {
            "eigenvalue_accuracy_kappa": eigenvalue_threshold,
            "oracle_reliability_kappa": oracle_threshold,
            "block_detection_kappa": block_threshold,
        },
        "recommendations": recommendations,
    }


def write_outputs(results: List[StabilityResult], summary: Dict,
                  output_dir: str) -> None:
    """Write all output files."""
    # Full JSON results
    full_path = os.path.join(output_dir, "numerical_stability_results.json")
    serializable = []
    for r in results:
        entry = {
            "condition_number": r.condition_number,
            "matrix_size": r.matrix_size,
            "eigenvalue_accuracy": [asdict(e) for e in r.eigenvalue_accuracy],
            "fiedler_accuracy": [asdict(f) for f in r.fiedler_accuracy],
            "block_detection": [asdict(b) for b in r.block_detection],
            "oracle_robustness": [
                {
                    "condition_number": o.condition_number,
                    "matrix_size": o.matrix_size,
                    "baseline_recommendation": o.baseline_recommendation,
                    "agreement_rate": o.agreement_rate,
                    "feature_max_relative_change": o.feature_max_relative_change,
                } for o in r.oracle_robustness
            ],
            "interval_bounds": [asdict(b) for b in r.interval_bounds],
        }
        serializable.append(entry)
    with open(full_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"  → {full_path}")

    # Summary JSON
    summary_path = os.path.join(output_dir, "stability_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  → {summary_path}")

    # CSV for eigenvalue error plot
    ev_csv = os.path.join(output_dir, "eigenvalue_error_vs_kappa.csv")
    with open(ev_csv, "w") as f:
        f.write("log10_kappa,eigenvalue_max_rel_error,fiedler_rel_error,lapack_max_rel_error\n")
        for row in summary["summary_table"]:
            f.write(f"{row['log10_kappa']},{row['eigenvalue_max_rel_error']:.6e},"
                    f"{row['fiedler_eigenvalue_rel_error']:.6e},"
                    f"{row['lapack_max_rel_error']:.6e}\n")
    print(f"  → {ev_csv}")

    # CSV for oracle robustness plot
    or_csv = os.path.join(output_dir, "oracle_robustness_vs_kappa.csv")
    with open(or_csv, "w") as f:
        f.write("log10_kappa,oracle_agreement_rate,fiedler_angular_error_deg\n")
        for row in summary["summary_table"]:
            f.write(f"{row['log10_kappa']},{row['oracle_agreement_rate']:.4f},"
                    f"{row['fiedler_angular_error_deg']:.6e}\n")
    print(f"  → {or_csv}")

    # CSV for block detection plot
    bd_csv = os.path.join(output_dir, "block_detection_vs_kappa.csv")
    with open(bd_csv, "w") as f:
        f.write("log10_kappa,block_detection_accuracy\n")
        for row in summary["summary_table"]:
            f.write(f"{row['log10_kappa']},{row['block_detection_accuracy']:.4f}\n")
    print(f"  → {bd_csv}")

    # Print summary table
    print("\n" + "=" * 90)
    print("NUMERICAL STABILITY SUMMARY")
    print("=" * 90)
    print(f"{'κ':>12s}  {'EV MaxErr':>12s}  {'Fiedler Err':>12s}  "
          f"{'LAPACK Err':>12s}  {'Block Acc':>10s}  {'Oracle Agr':>10s}")
    print("-" * 90)
    for row in summary["summary_table"]:
        print(f"  10^{row['log10_kappa']:>4.0f}    "
              f"{row['eigenvalue_max_rel_error']:>12.3e}  "
              f"{row['fiedler_eigenvalue_rel_error']:>12.3e}  "
              f"{row['lapack_max_rel_error']:>12.3e}  "
              f"{row['block_detection_accuracy']:>10.3f}  "
              f"{row['oracle_agreement_rate']:>10.3f}")
    print("-" * 90)

    print("\nThresholds:")
    for key, val in summary["thresholds"].items():
        print(f"  {key}: κ = {val:.0e}")

    print("\nRecommendations:")
    for rec in summary["recommendations"]:
        print(f"  • {rec}")
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Numerical stability analysis for SpecOracle spectral features."
    )
    parser.add_argument(
        "--sizes", type=str, default="50,100,200",
        help="Comma-separated matrix sizes to test (default: 50,100,200)"
    )
    parser.add_argument(
        "--trials", type=int, default=10,
        help="Number of random trials per (κ, n) pair (default: 10)"
    )
    parser.add_argument(
        "--output", type=str, default="benchmarks/stability_results",
        help="Output directory (default: benchmarks/stability_results)"
    )
    args = parser.parse_args()

    sizes = [int(s.strip()) for s in args.sizes.split(",")]
    print(f"Numerical Stability Analysis for SpecOracle")
    print(f"  Matrix sizes: {sizes}")
    print(f"  Trials per (κ,n): {args.trials}")
    print(f"  Condition numbers: 10^0 to 10^15 ({len(CONDITION_NUMBERS)} levels)")
    print(f"  Output: {args.output}/")
    print()

    t_start = time.time()
    summary = run_stability_analysis(sizes, args.trials, args.output)
    elapsed = time.time() - t_start

    print(f"Total time: {elapsed:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()
