#!/usr/bin/env python3
"""CausalQD Experiment Suite — Quality-Diversity Illumination for Causal Discovery.

Evaluates CausalQD against GES, PC, MMHC baselines on standard Bayesian
network benchmarks (Asia, Sachs, Child, Insurance, Alarm).

Research Questions:
  RQ1: Edge-probability calibration (AUROC, Brier score)
      — CausalQD archive frequency vs bootstrap resampling baselines
  RQ2: Structure accuracy (SHD, F1) of consensus predictions
  RQ3: Quality-Diversity metrics (QD-Score, coverage, diversity)
  RQ4: Edge certificate quality (TP/FP separation)
  RQ5: Scalability and time-quality Pareto frontier
"""
from __future__ import annotations

import json
import sys
import time
from collections import deque
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from causal_qd.benchmarks.standard_networks import (
    AsiaBenchmark, SachsBenchmark, ChildBenchmark,
    InsuranceBenchmark, AlarmBenchmark,
)
from causal_qd.baselines import PCAlgorithm, GESBaseline, MMHCAlgorithm
from causal_qd.operators import (
    EdgeFlipMutation, AcyclicEdgeAddition, EdgeRemoveMutation,
    EdgeReverseMutation, UniformCrossover,
)
from causal_qd.operators.local_search import GreedyLocalSearch, HillClimbingRefiner
from causal_qd.descriptors import StructuralDescriptor
from causal_qd.scores import BICScore, CachedScore
from causal_qd.engine.map_elites import CausalMAPElites, MAPElitesConfig
from causal_qd.metrics.structural import SHD, F1, skeleton_f1
from causal_qd.certificates.bootstrap import BootstrapCertificateComputer

SEED = 42
N_SAMPLES = 1000
N_SEEDS = 1
N_BOOT = 20  # bootstrap resamples for baselines

BENCHMARKS = [
    ("Asia", AsiaBenchmark, 8),
    ("Sachs", SachsBenchmark, 11),
    ("Child", ChildBenchmark, 20),
    ("Insurance", InsuranceBenchmark, 27),
    ("Alarm", AlarmBenchmark, 37),
]

ITER_BUDGET = {8: 300, 11: 400, 20: 600, 27: 700, 37: 500}


# ── DAG utilities ────────────────────────────────────────────────────

def _is_dag(adj):
    n = adj.shape[0]
    ideg = adj.sum(axis=0).copy()
    q = deque(i for i in range(n) if ideg[i] == 0)
    cnt = 0
    while q:
        nd = q.popleft(); cnt += 1
        for c in range(n):
            if adj[nd, c]:
                ideg[c] -= 1
                if ideg[c] == 0: q.append(c)
    return cnt == n


def _safe_run(alg, data, n):
    try:
        d = alg.run(data)
        if d.shape == (n, n) and _is_dag(d):
            return d.astype(np.int8)
    except Exception:
        pass
    return None


# ── CausalQD engine ─────────────────────────────────────────────────

def _warmstart_dags(data, n):
    dags = []
    for alg in [PCAlgorithm(alpha=0.01), PCAlgorithm(alpha=0.05),
                PCAlgorithm(alpha=0.1), GESBaseline(), MMHCAlgorithm(alpha=0.05)]:
        d = _safe_run(alg, data, n)
        if d is not None:
            dags.append(d)
    return dags


def _run_causalqd(data, n_nodes, seed=42, iters=None, refine=True):
    if iters is None:
        iters = ITER_BUDGET.get(n_nodes, 500)
    scorer = CachedScore(BICScore())
    desc = StructuralDescriptor(features=["edge_density", "max_in_degree"])
    config = MAPElitesConfig(
        archive_dims=(20, 20),
        archive_ranges=((0.0, 1.0), (0.0, 1.0)),
        seed=seed,
        adaptive_operators=True,
        log_interval=iters + 1,
        selection_strategy="quality_proportional",
    )
    engine = CausalMAPElites(
        mutations=[EdgeFlipMutation().mutate, AcyclicEdgeAddition().mutate,
                   EdgeRemoveMutation().mutate, EdgeReverseMutation().mutate],
        crossovers=[UniformCrossover().crossover],
        descriptor_fn=desc.compute,
        score_fn=scorer.score,
        config=config,
    )
    ws = _warmstart_dags(data, n_nodes)
    archive = engine.run(data, n_iterations=iters, batch_size=32, initial_dags=ws)

    # Refine top archive members with hill-climbing
    if refine and archive.size > 0:
        refiner = HillClimbingRefiner(
            local_search=GreedyLocalSearch(max_iterations=20),
            max_refine=min(20, archive.size),
        )
        rng = np.random.default_rng(seed)
        refined = refiner.refine_archive(archive.entries, scorer, data, rng)
        for idx, result in refined:
            entry = archive.entries[idx]
            from causal_qd.archive.archive_base import ArchiveEntry
            new_entry = ArchiveEntry(
                solution=result.dag,
                quality=result.score,
                descriptor=desc.compute(result.dag, data),
                metadata={"refined": True},
            )
            archive.add(new_entry)

    return archive


def _edge_probs(archive):
    """Quality-weighted edge frequency across archive elites."""
    entries = archive.entries
    n = entries[0].solution.shape[0]
    q = np.array([e.quality for e in entries])
    s = q - q.min()
    t = s.sum()
    w = s / t if t > 1e-15 else np.ones(len(entries)) / len(entries)
    freq = np.zeros((n, n), dtype=np.float64)
    for wi, e in zip(w, entries):
        freq += wi * e.solution.astype(np.float64)
    return freq


def _consensus_dag(archive, threshold=0.5):
    freq = _edge_probs(archive)
    return (freq >= threshold).astype(np.int8), freq


# ── Bootstrap baselines ─────────────────────────────────────────────

def _bootstrap_probs(alg_factory, data, n, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    freq = np.zeros((n, n), dtype=np.float64)
    ns = data.shape[0]
    valid = 0
    for _ in range(n_boot):
        idx = rng.integers(0, ns, size=ns)
        alg = alg_factory()
        d = _safe_run(alg, data[idx], n)
        if d is not None:
            freq += d.astype(np.float64)
            valid += 1
    return freq / max(valid, 1)


# ── Metrics ──────────────────────────────────────────────────────────

def _eval_dag(pred, true_dag):
    f1m = F1()
    f1v = f1m.compute(pred, true_dag)
    return {
        "shd": int(SHD.compute(pred, true_dag)),
        "f1": round(f1v, 4),
        "precision": round(f1m.precision(), 4),
        "recall": round(f1m.recall(), 4),
        "skeleton_f1": round(skeleton_f1(pred, true_dag), 4),
    }


def _edge_prob_metrics(probs, true_dag):
    n = true_dag.shape[0]
    mask = ~np.eye(n, dtype=bool)
    yt = true_dag[mask].ravel().astype(int)
    ys = np.clip(probs[mask].ravel(), 0, 1)
    if yt.sum() == 0 or yt.sum() == len(yt):
        return {"auroc": 0.0, "auprc": 0.0, "brier": 1.0}
    return {
        "auroc": round(roc_auc_score(yt, ys), 4),
        "auprc": round(average_precision_score(yt, ys), 4),
        "brier": round(brier_score_loss(yt, ys), 4),
    }


def _calibration_error(probs, true_dag, n_bins=10):
    """Expected Calibration Error for edge probabilities."""
    n = true_dag.shape[0]
    mask = ~np.eye(n, dtype=bool)
    yt = true_dag[mask].ravel().astype(float)
    ys = np.clip(probs[mask].ravel(), 0, 1)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (ys >= bins[i]) & (ys < bins[i + 1])
        if in_bin.sum() == 0:
            continue
        avg_conf = ys[in_bin].mean()
        avg_acc = yt[in_bin].mean()
        ece += in_bin.sum() * abs(avg_conf - avg_acc)
    return round(ece / len(yt), 4)


# ══════════════════════════════════════════════════════════════════════
# RQ1: Edge Probability Calibration
# ══════════════════════════════════════════════════════════════════════

def rq1_calibration():
    print("\nRQ1: Edge Probability Calibration (AUROC / AUPRC / Brier / ECE)")
    print("=" * 70)
    results = {}

    baseline_factories = {
        "GES-Boot": lambda: GESBaseline(),
        "PC-Boot": lambda: PCAlgorithm(alpha=0.05),
        "MMHC-Boot": lambda: MMHCAlgorithm(alpha=0.05),
    }

    for bname, BenchClass, n_nodes in BENCHMARKS:
        bench = BenchClass()
        results[bname] = {}
        print(f"\n  {bname} ({n_nodes} nodes):")

        for s in range(N_SEEDS):
            rng = np.random.default_rng(SEED + s)
            data = bench.generate_data(N_SAMPLES, rng=rng)

            # CausalQD
            t0 = time.perf_counter()
            archive = _run_causalqd(data, n_nodes, seed=SEED + s)
            t_qd = time.perf_counter() - t0
            probs = _edge_probs(archive)
            m = _edge_prob_metrics(probs, bench.true_dag)
            m["ece"] = _calibration_error(probs, bench.true_dag)
            m["time_s"] = round(t_qd, 2)
            m["n_elites"] = archive.size
            results[bname].setdefault("CausalQD", []).append(m)

            # Bootstrap baselines (skip slow combos)
            for bl_name, factory in baseline_factories.items():
                if bl_name == "GES-Boot" and n_nodes > 8:
                    continue
                if bl_name == "MMHC-Boot" and n_nodes > 27:
                    continue
                    continue
                t0 = time.perf_counter()
                bp = _bootstrap_probs(factory, data, n_nodes, seed=SEED + s)
                t_bl = time.perf_counter() - t0
                bm = _edge_prob_metrics(bp, bench.true_dag)
                bm["ece"] = _calibration_error(bp, bench.true_dag)
                bm["time_s"] = round(t_bl, 2)
                results[bname].setdefault(bl_name, []).append(bm)

        # Print averages
        header = f"  {'Algorithm':<14} {'AUROC':>7} {'AUPRC':>7} {'Brier':>7} {'ECE':>7} {'Time':>8}"
        print(header)
        for alg in results[bname]:
            runs = results[bname][alg]
            avg = {k: round(np.mean([r[k] for r in runs]), 4)
                   for k in ["auroc", "auprc", "brier", "ece", "time_s"]}
            print(f"  {alg:<14} {avg['auroc']:>7.3f} {avg['auprc']:>7.3f} "
                  f"{avg['brier']:>7.3f} {avg['ece']:>7.3f} {avg['time_s']:>7.1f}s")

    # Average results for each method
    for alg in results[bname]:
        runs_all = results[bname][alg]
        results[bname][alg] = {
            k: round(np.mean([r[k] for r in runs_all if k in r]), 4)
            for k in ["auroc", "auprc", "brier", "ece", "time_s"]
        }

    return results


# ══════════════════════════════════════════════════════════════════════
# RQ2: Structure Accuracy (Consensus)
# ══════════════════════════════════════════════════════════════════════

def rq2_structure_accuracy():
    print("\nRQ2: Structure Accuracy — Consensus vs Point Estimates")
    print("=" * 70)
    results = {}

    for bname, BenchClass, n_nodes in BENCHMARKS:
        bench = BenchClass()
        results[bname] = {}
        rng = np.random.default_rng(SEED)
        data = bench.generate_data(N_SAMPLES, rng=rng)

        # CausalQD consensus at multiple thresholds
        t0 = time.perf_counter()
        archive = _run_causalqd(data, n_nodes, seed=SEED)
        t_qd = time.perf_counter() - t0

        best_f1 = 0
        best_thr = 0.5
        for thr in np.arange(0.3, 0.85, 0.05):
            cons, _ = _consensus_dag(archive, threshold=thr)
            m = _eval_dag(cons, bench.true_dag)
            if m["f1"] > best_f1:
                best_f1 = m["f1"]
                best_thr = thr

        cons, _ = _consensus_dag(archive, threshold=best_thr)
        m = _eval_dag(cons, bench.true_dag)
        m["threshold"] = round(best_thr, 2)
        m["time_s"] = round(t_qd, 2)
        results[bname]["CausalQD-Consensus"] = m

        # Best single DAG from archive
        best = archive.best()
        if best:
            m = _eval_dag(best.solution, bench.true_dag)
            m["time_s"] = round(t_qd, 2)
            results[bname]["CausalQD-Best"] = m

        # Baselines
        for alg_name, alg in [("GES", GESBaseline()), ("PC", PCAlgorithm(alpha=0.05)),
                                ("MMHC", MMHCAlgorithm(alpha=0.05))]:
            t0 = time.perf_counter()
            pred = _safe_run(alg, data, n_nodes)
            t_bl = time.perf_counter() - t0
            if pred is None:
                pred = np.zeros((n_nodes, n_nodes), dtype=np.int8)
            m = _eval_dag(pred, bench.true_dag)
            m["time_s"] = round(t_bl, 2)
            results[bname][alg_name] = m

        print(f"\n  {bname} ({n_nodes} nodes):")
        print(f"  {'Algorithm':<20} {'SHD':>5} {'F1':>7} {'Prec':>7} {'Rec':>7} {'SkF1':>7} {'Time':>8}")
        for alg in results[bname]:
            r = results[bname][alg]
            thr_str = f" @{r['threshold']}" if 'threshold' in r else ""
            print(f"  {alg+thr_str:<20} {r['shd']:>5} {r['f1']:>7.3f} {r['precision']:>7.3f} "
                  f"{r['recall']:>7.3f} {r['skeleton_f1']:>7.3f} {r['time_s']:>7.1f}s")

    return results


# ══════════════════════════════════════════════════════════════════════
# RQ3: Quality-Diversity Metrics (unique to CausalQD)
# ══════════════════════════════════════════════════════════════════════

def rq3_quality_diversity():
    print("\nRQ3: Quality-Diversity Metrics")
    print("=" * 70)
    results = {}

    for bname, BenchClass, n_nodes in BENCHMARKS:
        bench = BenchClass()
        rng = np.random.default_rng(SEED)
        data = bench.generate_data(N_SAMPLES, rng=rng)

        archive = _run_causalqd(data, n_nodes, seed=SEED)
        entries = archive.entries
        qualities = [e.quality for e in entries]
        descs = np.array([e.descriptor for e in entries])

        # Diversity
        if len(descs) > 1:
            dists = [np.linalg.norm(descs[i] - descs[j])
                     for i, j in combinations(range(len(descs)), 2)]
            diversity = float(np.mean(dists))
        else:
            diversity = 0.0

        # SHD distribution across archive
        shds = [SHD.compute(e.solution, bench.true_dag) for e in entries]

        # How many archive members beat each baseline?
        baseline_shds = {}
        for alg_name, alg in [("GES", GESBaseline()), ("PC", PCAlgorithm(alpha=0.05)),
                                ("MMHC", MMHCAlgorithm(alpha=0.05))]:
            pred = _safe_run(alg, data, n_nodes)
            if pred is not None:
                baseline_shds[alg_name] = SHD.compute(pred, bench.true_dag)
            else:
                baseline_shds[alg_name] = 999

        n_beat = {bl: sum(1 for s in shds if s < v) for bl, v in baseline_shds.items()}

        results[bname] = {
            "qd_score": round(archive.qd_score(), 2),
            "coverage": round(archive.coverage(), 4),
            "n_elites": archive.size,
            "diversity": round(diversity, 4),
            "best_quality": round(max(qualities), 2),
            "mean_quality": round(float(np.mean(qualities)), 2),
            "best_shd": min(shds),
            "median_shd": int(np.median(shds)),
            "mean_shd": round(float(np.mean(shds)), 1),
            "n_beat_ges": n_beat.get("GES", 0),
            "n_beat_pc": n_beat.get("PC", 0),
            "n_beat_mmhc": n_beat.get("MMHC", 0),
        }

        print(f"\n  {bname}: {archive.size} elites, QD={archive.qd_score():.0f}, "
              f"cov={archive.coverage():.3f}, div={diversity:.3f}")
        print(f"    SHD: best={min(shds)} med={int(np.median(shds))} mean={np.mean(shds):.1f}")
        for bl, cnt in n_beat.items():
            print(f"    beat {bl} (SHD={baseline_shds[bl]}): {cnt}/{archive.size}")

    return results


# ══════════════════════════════════════════════════════════════════════
# RQ4: Edge Certificates
# ══════════════════════════════════════════════════════════════════════

def rq4_certificates():
    print("\nRQ4: Edge Certificate Quality")
    print("=" * 70)
    results = {}

    for bname, BenchClass, n_nodes in BENCHMARKS[:3]:  # Asia, Sachs, Child
        bench = BenchClass()
        rng = np.random.default_rng(SEED)
        data = bench.generate_data(N_SAMPLES, rng=rng)

        archive = _run_causalqd(data, n_nodes, seed=SEED)
        best = archive.best()
        if not best:
            continue
        scorer = BICScore()

        cert_computer = BootstrapCertificateComputer(
            score_fn=scorer.score,
            n_bootstrap=50,
            confidence=0.95,
            seed=SEED,
        )
        certs = cert_computer.compute(best.solution, data)

        true_edges = set(zip(*np.where(bench.true_dag)))
        pred_edges = set(zip(*np.where(best.solution)))

        tp_certs = []
        fp_certs = []
        for (i, j), cert in certs.items():
            if (i, j) in true_edges:
                tp_certs.append(cert.value)
            else:
                fp_certs.append(cert.value)

        # Archive-based edge reliability
        freq = _edge_probs(archive)
        tp_freq = [freq[i, j] for i, j in true_edges if best.solution[i, j]]
        fp_freq = [freq[i, j] for i, j in pred_edges - true_edges]

        results[bname] = {
            "n_edges": len(certs),
            "n_tp": len(tp_certs),
            "n_fp": len(fp_certs),
            "mean_tp_cert": round(float(np.mean(tp_certs)), 4) if tp_certs else 0.0,
            "mean_fp_cert": round(float(np.mean(fp_certs)), 4) if fp_certs else 0.0,
            "cert_separation": round(float(np.mean(tp_certs) - np.mean(fp_certs)), 4) if tp_certs and fp_certs else 0.0,
            "mean_tp_archive_freq": round(float(np.mean(tp_freq)), 4) if tp_freq else 0.0,
            "mean_fp_archive_freq": round(float(np.mean(fp_freq)), 4) if fp_freq else 0.0,
        }

        print(f"\n  {bname}: {len(certs)} edges certified")
        print(f"    Bootstrap cert: TP={results[bname]['mean_tp_cert']:.3f} "
              f"FP={results[bname]['mean_fp_cert']:.3f} sep={results[bname]['cert_separation']:.3f}")
        print(f"    Archive freq:   TP={results[bname]['mean_tp_archive_freq']:.3f} "
              f"FP={results[bname]['mean_fp_archive_freq']:.3f}")

    return results


# ══════════════════════════════════════════════════════════════════════
# RQ5: Scalability
# ══════════════════════════════════════════════════════════════════════

def rq5_scalability():
    print("\nRQ5: Scalability — Wall-Clock Time")
    print("=" * 70)
    results = {}

    for bname, BenchClass, n_nodes in BENCHMARKS:
        bench = BenchClass()
        rng = np.random.default_rng(SEED)
        data = bench.generate_data(N_SAMPLES, rng=rng)

        timings = {}

        # CausalQD
        t0 = time.perf_counter()
        archive = _run_causalqd(data, n_nodes, seed=SEED)
        timings["CausalQD"] = round(time.perf_counter() - t0, 2)

        # CausalQD quality at this time
        best = archive.best()
        qd_f1 = _eval_dag(best.solution, bench.true_dag)["f1"] if best else 0.0

        # Baselines
        baseline_f1 = {}
        for alg_name, alg in [("GES", GESBaseline()), ("PC", PCAlgorithm(alpha=0.05)),
                                ("MMHC", MMHCAlgorithm(alpha=0.05))]:
            t0 = time.perf_counter()
            pred = _safe_run(alg, data, n_nodes)
            timings[alg_name] = round(time.perf_counter() - t0, 2)
            if pred is not None:
                baseline_f1[alg_name] = _eval_dag(pred, bench.true_dag)["f1"]
            else:
                baseline_f1[alg_name] = 0.0

        results[bname] = {
            "n_nodes": n_nodes,
            "timings": timings,
            "causalqd_f1": qd_f1,
            "baseline_f1": baseline_f1,
        }

        print(f"  {bname} ({n_nodes}n): " +
              "  ".join(f"{k}={v:.1f}s" for k, v in timings.items()))

    return results


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  CausalQD — Quality-Diversity Illumination for Causal Discovery")
    print("  Experiment Suite")
    print("=" * 70)

    all_results = {}
    all_results["rq1_calibration"] = rq1_calibration()
    all_results["rq2_structure_accuracy"] = rq2_structure_accuracy()
    all_results["rq3_quality_diversity"] = rq3_quality_diversity()
    all_results["rq4_certificates"] = rq4_certificates()
    all_results["rq5_scalability"] = rq5_scalability()

    # ── Headline summary ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  HEADLINE RESULTS")
    print("=" * 70)

    rq1 = all_results["rq1_calibration"]
    # Aggregate AUROC across benchmarks
    for alg in ["CausalQD", "GES-Boot", "PC-Boot", "MMHC-Boot"]:
        aucs = []
        for b in rq1:
            if alg in rq1[b]:
                if isinstance(rq1[b][alg], list):
                    aucs.append(np.mean([r["auroc"] for r in rq1[b][alg]]))
                elif isinstance(rq1[b][alg], dict) and "auroc" in rq1[b][alg]:
                    aucs.append(rq1[b][alg]["auroc"])
        if aucs:
            print(f"  {alg}: mean AUROC={np.mean(aucs):.3f}")

    rq3 = all_results["rq3_quality_diversity"]
    mean_cov = np.mean([rq3[b]["coverage"] for b in rq3])
    total_elites = sum(rq3[b]["n_elites"] for b in rq3)
    print(f"\n  Archive diversity: {total_elites} total elites, {mean_cov:.3f} coverage")

    # ── Save ────────────────────────────────────────────────────────
    out_path = Path(__file__).parent / "results.json"

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super().default(obj)

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, cls=NpEncoder)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
