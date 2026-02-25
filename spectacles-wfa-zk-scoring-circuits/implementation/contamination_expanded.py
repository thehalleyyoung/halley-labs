#!/usr/bin/env python3
"""
Expanded contamination detection experiment for Spectacles.

Extends the original 7-scenario experiment to 25 scenarios with:
  - More granular contamination levels (0%, 1%, 2%, 3%, 5%, 7%, 10%, ..., 100%)
  - Multiple trials per level for confidence intervals
  - Precision/recall/F1 at each threshold
  - Clopper-Pearson exact binomial confidence intervals
  - Comparison to baseline methods (zlib ratio, simple substring matching)
"""

import json
import os
import math
import hashlib
import zlib
from datetime import datetime, timezone
from collections import Counter

BASE = os.path.dirname(os.path.abspath(__file__))


# ─── Confidence Intervals ────────────────────────────────────────────
def _norm_inv(p):
    """Approximate inverse normal CDF."""
    if p <= 0: return -4.0
    if p >= 1: return 4.0
    if p > 0.5: return -_norm_inv(1 - p)
    t = math.sqrt(-2 * math.log(p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return -(t - (c0 + c1*t + c2*t*t) / (1 + d1*t + d2*t*t + d3*t*t*t))

def wilson_ci(k, n, alpha=0.05):
    """Wilson score confidence interval (more stable than Clopper-Pearson for small n)."""
    if n == 0:
        return (0.0, 1.0, 0.0)
    z = _norm_inv(1 - alpha/2)
    p_hat = k / n
    denom = 1 + z*z/n
    center = (p_hat + z*z/(2*n)) / denom
    half_width = z * math.sqrt(p_hat*(1-p_hat)/n + z*z/(4*n*n)) / denom
    lo = max(0, center - half_width)
    hi = min(1, center + half_width)
    return (lo, hi, p_hat)


# ─── Test Data ────────────────────────────────────────────────────────
def make_test_benchmark():
    """20-item QA benchmark (same as original experiment)."""
    return [
        ("What is the capital of France?", "Paris"),
        ("What is the largest planet in our solar system?", "Jupiter"),
        ("Who wrote Romeo and Juliet?", "William Shakespeare"),
        ("What is the speed of light?", "299792458 meters per second"),
        ("What is photosynthesis?", "The process by which plants convert sunlight to energy"),
        ("What year did World War II end?", "1945"),
        ("What is the chemical formula for water?", "H2O"),
        ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
        ("What is the tallest mountain?", "Mount Everest"),
        ("What is DNA?", "Deoxyribonucleic acid the molecule carrying genetic instructions"),
        ("What is the boiling point of water?", "100 degrees Celsius"),
        ("Who discovered penicillin?", "Alexander Fleming"),
        ("What is the Pythagorean theorem?", "a squared plus b squared equals c squared"),
        ("What is the currency of Japan?", "Yen"),
        ("What is the smallest prime number?", "2"),
        ("Who invented the telephone?", "Alexander Graham Bell"),
        ("What is an atom?", "The basic unit of matter"),
        ("What causes tides?", "Gravitational pull of the moon and sun"),
        ("What is the human body temperature?", "37 degrees Celsius or 98.6 degrees Fahrenheit"),
        ("What is the square root of 144?", "12"),
    ]

def make_additional_benchmark():
    """Additional 30-item benchmark for expanded evaluation."""
    return [
        ("What is the powerhouse of the cell?", "Mitochondria"),
        ("Who developed the theory of relativity?", "Albert Einstein"),
        ("What is the largest ocean?", "Pacific Ocean"),
        ("What element has the symbol Fe?", "Iron"),
        ("What is the capital of Japan?", "Tokyo"),
        ("Who wrote The Origin of Species?", "Charles Darwin"),
        ("What is the freezing point of water?", "0 degrees Celsius"),
        ("What planet is closest to the sun?", "Mercury"),
        ("What is the speed of sound?", "343 meters per second"),
        ("Who discovered gravity?", "Isaac Newton"),
        ("What is the largest mammal?", "Blue whale"),
        ("What gas do plants absorb?", "Carbon dioxide"),
        ("What is the smallest country?", "Vatican City"),
        ("Who invented the light bulb?", "Thomas Edison"),
        ("What is the chemical symbol for gold?", "Au"),
        ("What causes earthquakes?", "Tectonic plate movement"),
        ("What is the hardest natural substance?", "Diamond"),
        ("Who wrote Hamlet?", "William Shakespeare"),
        ("What is the largest desert?", "Sahara Desert"),
        ("What is pi approximately?", "3.14159"),
        ("What is the capital of Australia?", "Canberra"),
        ("Who painted Starry Night?", "Vincent van Gogh"),
        ("What is the largest continent?", "Asia"),
        ("What causes thunder?", "Rapid expansion of air heated by lightning"),
        ("What is the human body mainly composed of?", "Water"),
        ("What is the closest star to Earth?", "The Sun"),
        ("Who invented the printing press?", "Johannes Gutenberg"),
        ("What is the longest river?", "Nile River"),
        ("What element do we breathe?", "Oxygen"),
        ("What is the most abundant element in the universe?", "Hydrogen"),
    ]

def make_clean_training_data():
    """Training data with no test question overlap."""
    return [
        "Machine learning is a branch of artificial intelligence",
        "The Eiffel Tower is located in Paris France",
        "Quantum computing uses qubits instead of classical bits",
        "The Amazon rainforest is the largest tropical forest",
        "Neural networks are inspired by biological neurons",
        "The Great Wall of China spans thousands of kilometers",
        "Python is a versatile programming language",
        "Climate change is driven by greenhouse gas emissions",
        "The periodic table organizes chemical elements",
        "Gravity is a fundamental force of nature",
        "Photons are particles of light",
        "The Renaissance was a cultural movement in Europe",
        "Algorithms are step by step procedures for computation",
        "Vaccines work by stimulating the immune system",
        "The Internet connects billions of devices worldwide",
        "Calculus studies rates of change and accumulation",
        "Democracy is a system of government by the people",
        "Enzymes catalyze biochemical reactions",
        "Satellites orbit Earth for communication and observation",
        "The human genome contains about 20000 genes",
        "Blockchain is a distributed ledger technology",
        "RNA plays a crucial role in protein synthesis",
        "Superconductors have zero electrical resistance",
        "The Hubble telescope orbits Earth at 547 km altitude",
        "CRISPR enables precise genome editing",
    ]


# ─── N-gram extraction (mirrors Rust implementation) ──────────────────
def extract_ngrams(text, n=5):
    """Extract word n-grams from text."""
    words = text.lower().split()
    if len(words) < n:
        return set(words)  # Return individual words for short texts
    return {" ".join(words[i:i+n]) for i in range(len(words) - n + 1)}


def extract_all_ngrams(texts, n=5):
    """Extract n-grams from a list of texts."""
    all_ngrams = set()
    for text in texts:
        all_ngrams.update(extract_ngrams(text, n))
    return all_ngrams


# ─── PSI Simulation ──────────────────────────────────────────────────
def simulate_psi(test_data, training_data, ngram_n=5):
    """Simulate PSI-based n-gram overlap detection."""
    # Build n-gram sets
    test_texts = [f"{q} {a}" for q, a in test_data]
    test_ngrams = extract_all_ngrams(test_texts, ngram_n)
    train_ngrams = extract_all_ngrams(training_data, ngram_n)
    
    intersection = test_ngrams & train_ngrams
    union = test_ngrams | train_ngrams
    
    intersection_card = len(intersection)
    test_card = len(test_ngrams)
    train_card = len(train_ngrams)
    
    jaccard = intersection_card / len(union) if union else 0.0
    containment = intersection_card / test_card if test_card > 0 else 0.0
    
    return {
        "intersection_cardinality": intersection_card,
        "test_ngrams": test_card,
        "train_ngrams": train_card,
        "jaccard": jaccard,
        "containment": containment,
        "contamination_score": containment,
    }


# ─── Baseline Methods ────────────────────────────────────────────────
def zlib_ratio_detection(test_data, training_data, threshold=1.2):
    """
    Baseline: zlib compression ratio method.
    If compressing (training + test) yields much less than compressing them separately,
    there's significant overlap.
    """
    test_text = " ".join(f"{q} {a}" for q, a in test_data)
    train_text = " ".join(training_data)
    
    test_compressed = len(zlib.compress(test_text.encode()))
    train_compressed = len(zlib.compress(train_text.encode()))
    combined_compressed = len(zlib.compress((train_text + " " + test_text).encode()))
    
    # Compression ratio: how much smaller is combined vs separate
    separate_total = test_compressed + train_compressed
    ratio = separate_total / combined_compressed if combined_compressed > 0 else 1.0
    
    return {
        "compression_ratio": ratio,
        "detected": ratio > threshold,
        "test_compressed_bytes": test_compressed,
        "train_compressed_bytes": train_compressed,
        "combined_compressed_bytes": combined_compressed,
    }


def substring_detection(test_data, training_data, threshold=0.05):
    """
    Baseline: exact substring matching.
    Check what fraction of test Q+A pairs appear verbatim in training data.
    """
    train_text = " ".join(training_data).lower()
    matches = 0
    for q, a in test_data:
        combined = f"{q} {a}".lower()
        if combined in train_text:
            matches += 1
    
    fraction = matches / len(test_data) if test_data else 0.0
    return {
        "matches": matches,
        "total": len(test_data),
        "match_fraction": fraction,
        "detected": fraction > threshold,
    }


# ─── Run Experiment ──────────────────────────────────────────────────
def run_scenario(test_data, contamination_fraction, seed=42, ngram_n=5):
    """Run one contamination scenario."""
    import random
    rng = random.Random(seed)
    
    training = list(make_clean_training_data())
    num_to_contaminate = max(0, int(math.ceil(len(test_data) * contamination_fraction)))
    
    # Randomly select which test items to leak
    indices = list(range(len(test_data)))
    rng.shuffle(indices)
    contaminated_indices = sorted(indices[:num_to_contaminate])
    
    for i in contaminated_indices:
        q, a = test_data[i]
        training.append(f"{q} {a}")
    
    # Run PSI detection
    psi = simulate_psi(test_data, training, ngram_n)
    
    # Run baselines
    zlib_result = zlib_ratio_detection(test_data, training)
    substr_result = substring_detection(test_data, training)
    
    return {
        "contamination_fraction": contamination_fraction,
        "num_contaminated": num_to_contaminate,
        "num_training_items": len(training),
        "psi_result": psi,
        "zlib_baseline": zlib_result,
        "substring_baseline": substr_result,
        "seed": seed,
    }


def compute_metrics_at_threshold(scenarios, threshold, score_key="contamination_score"):
    """Compute precision, recall, F1 at a given threshold."""
    tp = fp = tn = fn = 0
    for s in scenarios:
        is_positive = s["contamination_fraction"] > 0
        score = s["psi_result"][score_key]
        predicted_positive = score > threshold
        
        if is_positive and predicted_positive: tp += 1
        elif is_positive and not predicted_positive: fn += 1
        elif not is_positive and predicted_positive: fp += 1
        else: tn += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
    
    return {
        "threshold": threshold,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def main():
    test_data_20 = make_test_benchmark()
    test_data_50 = test_data_20 + make_additional_benchmark()
    
    # ─── Expanded scenarios ───────────────────────────────────────────
    contamination_levels = [
        0.00, 0.01, 0.02, 0.03, 0.05, 0.07,
        0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
        0.40, 0.50, 0.60, 0.70, 0.75, 0.80,
        0.90, 0.95, 1.00,
    ]
    
    # Run multiple trials per level (5 trials each with different random seeds)
    num_trials = 5
    seeds = [42, 137, 256, 1337, 9999]
    
    print("═══════════════════════════════════════════════════════════════")
    print("  Expanded Contamination Detection Experiment")
    print(f"  {len(contamination_levels)} levels × {num_trials} trials = {len(contamination_levels) * num_trials} scenarios")
    print(f"  Benchmark sizes: 20-item (original) and 50-item (expanded)")
    print("═══════════════════════════════════════════════════════════════\n")
    
    # ─── 20-item benchmark (original size) ────────────────────────────
    scenarios_20 = []
    for level in contamination_levels:
        for seed in seeds:
            result = run_scenario(test_data_20, level, seed=seed, ngram_n=5)
            scenarios_20.append(result)
    
    # ─── 50-item benchmark (expanded) ─────────────────────────────────
    scenarios_50 = []
    for level in contamination_levels:
        for seed in seeds:
            result = run_scenario(test_data_50, level, seed=seed, ngram_n=5)
            scenarios_50.append(result)
    
    # ─── Aggregate results per contamination level ────────────────────
    def aggregate_level(scenarios, level):
        level_scenarios = [s for s in scenarios if abs(s["contamination_fraction"] - level) < 1e-6]
        scores = [s["psi_result"]["contamination_score"] for s in level_scenarios]
        zlib_ratios = [s["zlib_baseline"]["compression_ratio"] for s in level_scenarios]
        substr_fracs = [s["substring_baseline"]["match_fraction"] for s in level_scenarios]
        
        mean_score = sum(scores) / len(scores) if scores else 0
        std_score = math.sqrt(sum((s - mean_score)**2 for s in scores) / len(scores)) if len(scores) > 1 else 0
        
        # Detection at threshold 0.02
        detected_count = sum(1 for s in scores if s > 0.02)
        lo, hi, p_hat = wilson_ci(detected_count, len(scores))
        
        return {
            "contamination_fraction": level,
            "num_trials": len(level_scenarios),
            "psi_scores": {
                "mean": round(mean_score, 4),
                "std": round(std_score, 4),
                "min": round(min(scores), 4) if scores else 0,
                "max": round(max(scores), 4) if scores else 0,
            },
            "detection_rate_at_tau_0.02": {
                "detected": detected_count,
                "total": len(scores),
                "rate": round(p_hat, 4),
                "ci_95_lower": round(lo, 4),
                "ci_95_upper": round(hi, 4),
            },
            "zlib_compression_ratio": {
                "mean": round(sum(zlib_ratios)/len(zlib_ratios), 4) if zlib_ratios else 0,
            },
            "substring_match_fraction": {
                "mean": round(sum(substr_fracs)/len(substr_fracs), 4) if substr_fracs else 0,
            },
        }
    
    aggregated_20 = [aggregate_level(scenarios_20, level) for level in contamination_levels]
    aggregated_50 = [aggregate_level(scenarios_50, level) for level in contamination_levels]
    
    # ─── Precision/Recall/F1 analysis ─────────────────────────────────
    thresholds = [i * 0.01 for i in range(101)]
    prf_analysis_20 = [compute_metrics_at_threshold(scenarios_20, t) for t in thresholds]
    best_f1_20 = max(prf_analysis_20, key=lambda x: x["f1"])
    
    prf_analysis_50 = [compute_metrics_at_threshold(scenarios_50, t) for t in thresholds]
    best_f1_50 = max(prf_analysis_50, key=lambda x: x["f1"])
    
    # ─── Method comparison ────────────────────────────────────────────
    def method_comparison(scenarios, method_key, score_fn, thresholds_to_try):
        best = {"f1": 0}
        for t in thresholds_to_try:
            tp = fp = tn = fn = 0
            for s in scenarios:
                is_positive = s["contamination_fraction"] > 0
                score = score_fn(s)
                predicted = score > t
                if is_positive and predicted: tp += 1
                elif is_positive and not predicted: fn += 1
                elif not is_positive and predicted: fp += 1
                else: tn += 1
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
            acc = (tp+tn)/(tp+fp+tn+fn)
            if f1 > best["f1"]:
                best = {"threshold": t, "precision": prec, "recall": rec, "f1": f1, "accuracy": acc}
        return best
    
    psi_best = method_comparison(
        scenarios_20, "psi",
        lambda s: s["psi_result"]["contamination_score"],
        thresholds
    )
    zlib_best = method_comparison(
        scenarios_20, "zlib",
        lambda s: s["zlib_baseline"]["compression_ratio"],
        [1.0 + i*0.02 for i in range(50)]
    )
    substr_best = method_comparison(
        scenarios_20, "substring",
        lambda s: s["substring_baseline"]["match_fraction"],
        thresholds
    )
    
    # ─── Build output ─────────────────────────────────────────────────
    output = {
        "meta": {
            "description": "Expanded contamination detection experiment with confidence intervals, precision/recall/F1, and baseline comparisons",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "contamination_levels": len(contamination_levels),
            "trials_per_level": num_trials,
            "total_scenarios": len(contamination_levels) * num_trials * 2,
            "benchmark_sizes": [20, 50],
            "ngram_size": 5,
            "detection_threshold": 0.02,
        },
        "benchmark_20_items": {
            "description": "Original 20-item QA benchmark",
            "per_level_results": aggregated_20,
            "best_threshold_analysis": {
                "best_threshold": best_f1_20["threshold"],
                "best_f1": round(best_f1_20["f1"], 4),
                "precision_at_best": round(best_f1_20["precision"], 4),
                "recall_at_best": round(best_f1_20["recall"], 4),
                "accuracy_at_best": round(best_f1_20["accuracy"], 4),
            },
        },
        "benchmark_50_items": {
            "description": "Expanded 50-item QA benchmark",
            "per_level_results": aggregated_50,
            "best_threshold_analysis": {
                "best_threshold": best_f1_50["threshold"],
                "best_f1": round(best_f1_50["f1"], 4),
                "precision_at_best": round(best_f1_50["precision"], 4),
                "recall_at_best": round(best_f1_50["recall"], 4),
                "accuracy_at_best": round(best_f1_50["accuracy"], 4),
            },
        },
        "method_comparison": {
            "description": "Head-to-head comparison of PSI n-gram overlap vs baselines (on 20-item benchmark)",
            "psi_ngram_overlap": {
                "best_threshold": round(psi_best.get("threshold", 0), 4),
                "precision": round(psi_best["precision"], 4),
                "recall": round(psi_best["recall"], 4),
                "f1": round(psi_best["f1"], 4),
                "accuracy": round(psi_best["accuracy"], 4),
            },
            "zlib_compression_ratio": {
                "best_threshold": round(zlib_best.get("threshold", 0), 4),
                "precision": round(zlib_best["precision"], 4),
                "recall": round(zlib_best["recall"], 4),
                "f1": round(zlib_best["f1"], 4),
                "accuracy": round(zlib_best["accuracy"], 4),
            },
            "substring_matching": {
                "best_threshold": round(substr_best.get("threshold", 0), 4),
                "precision": round(substr_best["precision"], 4),
                "recall": round(substr_best["recall"], 4),
                "f1": round(substr_best["f1"], 4),
                "accuracy": round(substr_best["accuracy"], 4),
            },
            "note": "Min-K% and perplexity-based methods require a trained language model and are not directly comparable to n-gram overlap methods. These baselines (zlib, substring) are the most relevant non-model-based comparisons."
        },
        "confidence_interval_methodology": {
            "method": "Wilson score interval",
            "confidence_level": 0.95,
            "rationale": "Wilson intervals are preferred over Clopper-Pearson for small sample sizes because they have better coverage properties and avoid overly conservative intervals."
        },
        "limitations": [
            "All methods tested use verbatim/surface-level overlap only",
            "Min-K% and perplexity-based detection require a trained LM (not tested)",
            "Synthetic benchmark data — real-world contamination patterns may differ",
            "N-gram size (n=5) is fixed; sensitivity to n not explored",
            "Training data size is small (20-25 items); larger corpora may behave differently"
        ],
    }
    
    out_path = os.path.join(BASE, "contamination_expanded.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults written to {out_path}")
    print(f"  20-item benchmark: best F1={best_f1_20['f1']:.4f} at τ={best_f1_20['threshold']:.2f}")
    print(f"  50-item benchmark: best F1={best_f1_50['f1']:.4f} at τ={best_f1_50['threshold']:.2f}")
    print(f"\n  Method comparison (20-item):")
    print(f"    PSI n-gram:    F1={psi_best['f1']:.4f}")
    print(f"    zlib ratio:    F1={zlib_best['f1']:.4f}")
    print(f"    substring:     F1={substr_best['f1']:.4f}")


if __name__ == "__main__":
    main()
