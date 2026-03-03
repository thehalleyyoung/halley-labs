#!/usr/bin/env python3
"""CPA Benchmark Suite — runs CPA + 7 baselines on 3 benchmark families.

Usage:
    cd causal-plasticity-atlas
    PYTHONPATH=implementation python3 experiments/run_benchmarks.py

Outputs experiments/results/core_benchmarks.json with all results.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Setup paths
IMPL = Path(__file__).resolve().parent.parent / "implementation"
sys.path.insert(0, str(IMPL))
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

from benchmarks.generators import FSVPGenerator, CSVMGenerator, TPSGenerator
from cpa.baselines import (
    CDNODBaseline, GESBaseline, ICPBaseline,
    IndependentPHC, JCIBaseline, LSEMPooled, PooledBaseline,
)
from cpa.core.types import PlasticityClass
from cpa.pipeline import CPAOrchestrator, PipelineConfig
from cpa.pipeline.orchestrator import MultiContextDataset
from cpa.pipeline.results import MechanismClass

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── enum → string maps ───────────────────────────────────────────────
P2S = {
    PlasticityClass.INVARIANT: "invariant",
    PlasticityClass.PARAMETRIC_PLASTIC: "parametrically_plastic",
    PlasticityClass.STRUCTURAL_PLASTIC: "structurally_plastic",
    PlasticityClass.MIXED: "fully_plastic",
    PlasticityClass.EMERGENT: "emergent",
}
M2S = {m: m.value for m in MechanismClass}

NORM = {
    "fully_plastic": "fully_plastic", "mixed": "fully_plastic",
    "parametric_plastic": "parametrically_plastic",
    "structural_plastic": "structurally_plastic",
}

def _norm(c: str) -> str:
    c = c.lower().replace(" ", "_")
    return NORM.get(c, c)


# ── helpers ───────────────────────────────────────────────────────────
def aggregate_edges(
    edge_plast: Dict[Tuple[int, int], PlasticityClass],
    variable_names: List[str],
) -> Dict[str, str]:
    """Per-edge → per-variable classification (take most-plastic edge)."""
    H = {"invariant": 0, "parametrically_plastic": 1,
         "structurally_plastic": 2, "fully_plastic": 3, "emergent": 4}
    result: Dict[str, str] = {}
    for vi in range(len(variable_names)):
        ml, mc = -1, "invariant"
        for (t, _), pc in edge_plast.items():
            if t == vi:
                cs = P2S.get(pc, "invariant")
                if H.get(cs, 0) > ml:
                    ml, mc = H.get(cs, 0), cs
        result[variable_names[vi]] = mc
    return result


def compute_metrics(
    predicted: Dict[str, str], ground_truth: Dict[str, str],
) -> Dict[str, float]:
    """Macro-F1, accuracy, per-class precision/recall."""
    all_classes = sorted({_norm(v) for v in
                          list(ground_truth.values()) + list(predicted.values())})
    common = sorted(set(predicted) & set(ground_truth))
    if not common:
        return {"macro_f1": 0.0, "accuracy": 0.0,
                "macro_precision": 0.0, "macro_recall": 0.0}

    correct = sum(1 for v in common if _norm(predicted[v]) == _norm(ground_truth[v]))
    accuracy = correct / len(common)

    f1s, ps, rs = [], [], []
    for c in all_classes:
        tp = sum(1 for v in common if _norm(predicted[v]) == c and _norm(ground_truth[v]) == c)
        fp = sum(1 for v in common if _norm(predicted[v]) == c and _norm(ground_truth[v]) != c)
        fn = sum(1 for v in common if _norm(predicted[v]) != c and _norm(ground_truth[v]) == c)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        f1s.append(f1); ps.append(p); rs.append(r)

    return {
        "macro_f1": float(np.mean(f1s)),
        "accuracy": accuracy,
        "macro_precision": float(np.mean(ps)),
        "macro_recall": float(np.mean(rs)),
    }


# ── runners ───────────────────────────────────────────────────────────
def run_cpa(bench: Any) -> Tuple[Dict[str, str], float]:
    cfg = PipelineConfig.fast()
    cfg.run_phase_2 = False
    cfg.run_phase_3 = False
    ds = MultiContextDataset(
        context_data=bench.context_data,
        variable_names=bench.variable_names,
    )
    t0 = time.time()
    atlas = CPAOrchestrator(cfg).run(ds)
    elapsed = time.time() - t0
    pred = {v: M2S[atlas.get_classification(v)] for v in bench.variable_names}
    return pred, elapsed


BASELINES = {
    "ICP": ICPBaseline,
    "CD-NOD": CDNODBaseline,
    "JCI": JCIBaseline,
    "GES": GESBaseline,
    "Ind-PHC": IndependentPHC,
    "Pooled": PooledBaseline,
    "LSEM": LSEMPooled,
}


def run_baseline(name: str, cls: type, bench: Any,
                 timeout: float = 120.0) -> Tuple[Dict[str, str], float]:
    import signal

    def _handler(signum, frame):
        raise TimeoutError(f"{name} timed out after {timeout}s")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(int(timeout))
    try:
        t0 = time.time()
        obj = cls()
        obj.fit(bench.context_data)
        ep = (obj.predict_plasticity_all_targets()
              if hasattr(obj, "predict_plasticity_all_targets")
              else obj.predict_plasticity())
        elapsed = time.time() - t0

        if ep and isinstance(next(iter(ep.keys())), tuple):
            pred = aggregate_edges(ep, bench.variable_names)
        else:
            pred = {
                (bench.variable_names[k] if isinstance(k, int) else k): P2S.get(v, str(v))
                for k, v in ep.items()
            }
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    return pred, elapsed


# ── benchmark configs ─────────────────────────────────────────────────
CONFIGS = [
    ("FSVP-small",  FSVPGenerator, {"p": 6,  "K": 3, "n": 200}),
    ("FSVP-medium", FSVPGenerator, {"p": 10, "K": 4, "n": 300}),
    ("FSVP-large",  FSVPGenerator, {"p": 15, "K": 4, "n": 300}),
    ("CSVM-small",  CSVMGenerator, {"p": 6,  "K": 3, "n": 200}),
    ("CSVM-medium", CSVMGenerator, {"p": 10, "K": 4, "n": 300}),
    ("CSVM-large",  CSVMGenerator, {"p": 15, "K": 4, "n": 300}),
    ("TPS-small",   TPSGenerator,  {"p": 6,  "K": 5,  "n": 150}),
    ("TPS-medium",  TPSGenerator,  {"p": 8,  "K": 6, "n": 200}),
]

REPS = 3
METHODS = ["CPA"] + list(BASELINES.keys())


# ── scalability configs ──────────────────────────────────────────────
SCALE_P = [5, 10, 15, 20]
SCALE_K = [2, 3, 5, 8]


# ── main ──────────────────────────────────────────────────────────────
def main() -> None:
    all_results: List[Dict[str, Any]] = []

    # ─── 1. Core benchmarks ──────────────────────────────────────────
    print("=" * 65)
    print("  [1/3] CORE BENCHMARKS  (8 scenarios × 3 reps × 8 methods)")
    print("=" * 65)

    for sc_name, gen_cls, gen_kw in CONFIGS:
        for rep in range(REPS):
            seed = 42 + rep * 1000
            bench = gen_cls(**gen_kw, seed=seed).generate()
            gt = bench.ground_truth.classifications

            # CPA
            try:
                pred, t = run_cpa(bench)
            except Exception:
                pred = {v: "unclassified" for v in bench.variable_names}
                t = 0.0
            m = compute_metrics(pred, gt)
            all_results.append({"method": "CPA", "scenario": sc_name, "seed": seed,
                                "time": t, **m, "predicted": pred, "ground_truth": gt})

            # Baselines
            for bn, bc in BASELINES.items():
                try:
                    pred, t = run_baseline(bn, bc, bench)
                except Exception:
                    pred = {v: "unclassified" for v in bench.variable_names}
                    t = 0.0
                m = compute_metrics(pred, gt)
                all_results.append({"method": bn, "scenario": sc_name, "seed": seed,
                                    "time": t, **m, "predicted": pred, "ground_truth": gt})

        print(f"  ✓ {sc_name}")

    # ─── 2. Scalability sweep ────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  [2/3] SCALABILITY SWEEP")
    print("=" * 65)

    scale_results: List[Dict[str, Any]] = []

    for p in SCALE_P:
        bench = FSVPGenerator(p=p, K=5, n=500, seed=42).generate()
        gt = bench.ground_truth.classifications
        for mname in METHODS:
            try:
                if mname == "CPA":
                    pred, t = run_cpa(bench)
                else:
                    pred, t = run_baseline(mname, BASELINES[mname], bench)
            except Exception:
                pred = {v: "unclassified" for v in bench.variable_names}
                t = 0.0
            m = compute_metrics(pred, gt)
            scale_results.append({"method": mname, "sweep": "p", "value": p,
                                  "time": t, **m})
        print(f"  ✓ p={p}")

    for K in SCALE_K:
        bench = FSVPGenerator(p=10, K=K, n=300, seed=42).generate()
        gt = bench.ground_truth.classifications
        for mname in METHODS:
            try:
                if mname == "CPA":
                    pred, t = run_cpa(bench)
                else:
                    pred, t = run_baseline(mname, BASELINES[mname], bench)
            except Exception:
                pred = {v: "unclassified" for v in bench.variable_names}
                t = 0.0
            m = compute_metrics(pred, gt)
            scale_results.append({"method": mname, "sweep": "K", "value": K,
                                  "time": t, **m})
        print(f"  ✓ K={K}")

    # ─── 3. Ablation ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  [3/3] ABLATION STUDY")
    print("=" * 65)

    ablation_results: List[Dict[str, Any]] = []
    ablation_cfgs = {
        "CPA-full":       {"run_phase_2": True,  "run_phase_3": True},
        "CPA-no-QD":      {"run_phase_2": False, "run_phase_3": True},
        "CPA-no-cert":    {"run_phase_2": True,  "run_phase_3": False},
        "CPA-foundation": {"run_phase_2": False, "run_phase_3": False},
    }

    for ab_name, ab_kw in ablation_cfgs.items():
        for rep in range(3):
            seed = 42 + rep * 1000
            bench = FSVPGenerator(p=10, K=5, n=500, seed=seed).generate()
            gt = bench.ground_truth.classifications
            cfg = PipelineConfig.fast()
            for k, v in ab_kw.items():
                setattr(cfg, k, v)
            ds = MultiContextDataset(
                context_data=bench.context_data,
                variable_names=bench.variable_names,
            )
            t0 = time.time()
            try:
                atlas = CPAOrchestrator(cfg).run(ds)
                pred = {v: M2S[atlas.get_classification(v)] for v in bench.variable_names}
            except Exception:
                pred = {v: "unclassified" for v in bench.variable_names}
            elapsed = time.time() - t0
            m = compute_metrics(pred, gt)
            ablation_results.append({"variant": ab_name, "seed": seed,
                                     "time": elapsed, **m})
        print(f"  ✓ {ab_name}")

    # ─── Save results ────────────────────────────────────────────────
    output = {
        "core": all_results,
        "scalability": scale_results,
        "ablation": ablation_results,
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "reps": REPS,
            "configs": [(n, c.__name__, kw) for n, c, kw in CONFIGS],
        },
    }
    out_path = RESULTS_DIR / "core_benchmarks.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n✓ Results saved to {out_path}")

    # ─── Print summary tables ────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  CLASSIFICATION F1  (mean ± std over 3 reps)")
    print("=" * 65)
    scenarios = sorted({r["scenario"] for r in all_results})
    hdr = f"{'':14s}" + "".join(f" {m:>9s}" for m in METHODS)
    print(hdr)
    print("-" * len(hdr))
    for sc in scenarios:
        row = f"{sc:14s}"
        for method in METHODS:
            entries = [r for r in all_results
                       if r["scenario"] == sc and r["method"] == method]
            f1s = [e["macro_f1"] for e in entries]
            row += f" {np.mean(f1s):6.3f}±{np.std(f1s):.2f}"
        print(row)

    # Overall
    row = f"{'OVERALL':14s}"
    for method in METHODS:
        entries = [r for r in all_results if r["method"] == method]
        row += f" {np.mean([e['macro_f1'] for e in entries]):6.3f}    "
    print(row)

    # Timing
    print(f"\n{'TIMING (s)':14s}" + "".join(f" {m:>9s}" for m in METHODS))
    for sc in scenarios:
        row = f"{sc:14s}"
        for method in METHODS:
            entries = [r for r in all_results
                       if r["scenario"] == sc and r["method"] == method]
            row += f" {np.mean([e['time'] for e in entries]):9.2f}"
        print(row)


if __name__ == "__main__":
    main()
