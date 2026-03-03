#!/usr/bin/env python3
"""CPA Benchmark Evaluation — run benchmarks and compare results.

Runs the three benchmark generators (FSVP, CSVM, TPS), computes
classification and tipping-point metrics, and prints a summary table.

Usage
-----
    python examples/benchmark_evaluation.py
"""

from __future__ import annotations

import time
from typing import Dict, List

import numpy as np

from benchmarks.generators import (
    BenchmarkResult,
    FSVPGenerator,
    CSVMGenerator,
    TPSGenerator,
)
from benchmarks.metrics import (
    ClassificationMetrics,
    TippingPointMetrics,
    CertificateMetrics,
    ArchiveMetrics,
)
from cpa.pipeline import CPAOrchestrator, PipelineConfig
from cpa.pipeline.orchestrator import MultiContextDataset
from cpa.pipeline.results import AtlasResult, MechanismClass


def _make_config() -> PipelineConfig:
    """Small config suitable for benchmark demos."""
    cfg = PipelineConfig.fast()
    cfg.search.n_iterations = 5
    cfg.certificate.n_bootstrap = 30
    cfg.certificate.n_permutations = 30
    return cfg


def _run_benchmark(bench: BenchmarkResult, label: str) -> AtlasResult:
    """Run the CPA pipeline on a BenchmarkResult."""
    dataset = MultiContextDataset(
        context_data=bench.context_data,
        variable_names=bench.variable_names,
        context_ids=bench.context_ids,
    )
    cfg = _make_config()
    orch = CPAOrchestrator(cfg)

    t0 = time.perf_counter()
    atlas = orch.run(dataset)
    elapsed = time.perf_counter() - t0
    print(f"  [{label}] completed in {elapsed:.2f}s")
    return atlas


def _classification_metrics(
    atlas: AtlasResult, ground_truth
) -> Dict[str, float]:
    """Compute classification metrics against ground truth."""
    predicted = {}
    for var in atlas.variable_names:
        cls = atlas.get_classification(var)
        predicted[var] = cls.value if cls else "unclassified"

    true_labels = ground_truth.classifications

    tp = fp = fn = tn = 0
    for var in atlas.variable_names:
        pred_plastic = predicted.get(var, "unclassified") not in (
            "invariant", MechanismClass.INVARIANT.value,
        )
        true_plastic = true_labels.get(var, "invariant") not in (
            "invariant", "INVARIANT",
        )
        if pred_plastic and true_plastic:
            tp += 1
        elif pred_plastic and not true_plastic:
            fp += 1
        elif not pred_plastic and true_plastic:
            fn += 1
        else:
            tn += 1

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def run_fsvp_benchmark() -> Dict[str, float]:
    """Run Generator 1 (FSVP) benchmark."""
    print("\n--- Generator 1: FSVP (Fixed Structure, Varying Parameters) ---")
    gen = FSVPGenerator(p=8, K=4, n=300, density=0.3,
                        plasticity_fraction=0.5, seed=42)
    bench = gen.generate()
    atlas = _run_benchmark(bench, "FSVP")
    metrics = _classification_metrics(atlas, bench.ground_truth)
    return metrics


def run_csvm_benchmark() -> Dict[str, float]:
    """Run Generator 2 (CSVM) benchmark."""
    print("\n--- Generator 2: CSVM (Changing Structure, Variable Mismatch) ---")
    gen = CSVMGenerator(p=8, K=4, n=300, density=0.3,
                        emergence_fraction=0.2,
                        structural_change_fraction=0.3, seed=42)
    bench = gen.generate()
    atlas = _run_benchmark(bench, "CSVM")
    metrics = _classification_metrics(atlas, bench.ground_truth)
    return metrics


def run_tps_benchmark() -> Dict[str, float]:
    """Run Generator 3 (TPS) benchmark with tipping-point evaluation."""
    print("\n--- Generator 3: TPS (Tipping-Point Scenario) ---")
    gen = TPSGenerator(p=5, K=10, n=200, density=0.4,
                       n_tipping_points=2, seed=99)
    bench = gen.generate()
    atlas = _run_benchmark(bench, "TPS")

    cls_metrics = _classification_metrics(atlas, bench.ground_truth)

    # Tipping-point metrics
    true_tps = bench.ground_truth.tipping_points
    detected_tps: List[int] = []
    if (
        atlas.validation is not None
        and atlas.validation.tipping_points is not None
    ):
        tp_res = atlas.validation.tipping_points
        if hasattr(tp_res, "tipping_points") and tp_res.tipping_points:
            for tp in tp_res.tipping_points:
                loc = getattr(tp, "location", getattr(tp, "index", None))
                if loc is not None:
                    detected_tps.append(int(loc))

    if true_tps and detected_tps:
        mad = np.mean([
            min(abs(d - t) for d in detected_tps) for t in true_tps
        ])
        cls_metrics["tp_mad"] = mad
        cls_metrics["tp_detected"] = len(detected_tps)
        cls_metrics["tp_true"] = len(true_tps)
    else:
        cls_metrics["tp_mad"] = float("nan")
        cls_metrics["tp_detected"] = len(detected_tps)
        cls_metrics["tp_true"] = len(true_tps)

    return cls_metrics


def print_metrics_table(results: Dict[str, Dict[str, float]]) -> None:
    """Print a formatted metrics comparison table."""
    print("\n" + "=" * 75)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 75)

    header = f"{'Metric':>20s}"
    for label in results:
        header += f"  {label:>12s}"
    print(header)
    print("-" * 75)

    all_keys = set()
    for v in results.values():
        all_keys.update(v.keys())
    ordered_keys = sorted(all_keys)

    for key in ordered_keys:
        row = f"{key:>20s}"
        for label in results:
            val = results[label].get(key, float("nan"))
            if isinstance(val, float):
                row += f"  {val:>12.4f}"
            else:
                row += f"  {val:>12}"
        print(row)

    print("=" * 75)


def main() -> None:
    print("=" * 60)
    print("CPA Benchmark Evaluation")
    print("=" * 60)

    results = {}
    results["FSVP"] = run_fsvp_benchmark()
    results["CSVM"] = run_csvm_benchmark()
    results["TPS"] = run_tps_benchmark()

    print_metrics_table(results)

    print("\n" + "=" * 60)
    print("Benchmark evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
