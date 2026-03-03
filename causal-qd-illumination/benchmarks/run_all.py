#!/usr/bin/env python3
"""Comprehensive benchmark suite for CausalQD.

Runs MAP-Elites against standard baselines (PC, GES, MMHC) across multiple
graph scales and data regimes, measuring:
  - SHD (Structural Hamming Distance) to ground truth
  - F1-score on edge predictions
  - QD-Score (sum of qualities in archive)
  - Coverage (fraction of archive cells filled)
  - Diversity (number of unique MEC classes in archive)
  - Runtime
  - Bootstrap certificate accuracy

Usage:
    python benchmarks/run_all.py [--tier {quick,standard,full,stress}] [--output results.json]
"""

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

warnings.filterwarnings("ignore")

from causal_qd.core import DAG
from causal_qd.data import LinearGaussianSCM
from causal_qd.scores import BICScore, BDeuScore
from causal_qd.engine import CausalMAPElites, MAPElitesConfig
from causal_qd.descriptors import StructuralDescriptor
from causal_qd.operators import (
    EdgeFlipMutation, EdgeAddMutation, EdgeRemoveMutation,
    TopologicalMutation, EdgeReverseMutation, VStructureMutation,
    SkeletonMutation, PathMutation,
    UniformCrossover, SkeletonCrossover, MarkovBlanketCrossover,
)
from causal_qd.baselines import PCAlgorithm, GESAlgorithm, MMHCAlgorithm
from causal_qd.metrics import SHD, F1
from causal_qd.mec import CPDAGConverter, MECEnumerator
from causal_qd.certificates import BootstrapCertificateComputer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mut(op_cls):
    op = op_cls()
    return lambda adj, rng: op.mutate(adj, rng)

def _make_desc():
    d = StructuralDescriptor(features=["edge_density", "max_in_degree"])
    return lambda adj, data: d.compute(adj, data)

def _make_scorer():
    s = BICScore()
    return lambda adj, data: s.score(adj, data)

def _make_cross(op_cls):
    op = op_cls()
    def fn(a1, a2, rng):
        c1, c2 = op.crossover(a1, a2, rng)
        return c1
    return fn

def _random_dag(n_nodes: int, edge_prob: float, rng: np.random.Generator) -> np.ndarray:
    adj = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.random() < edge_prob:
                adj[i, j] = 1.0
    return adj

def _generate_data(adj: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    dag = DAG(adj)
    n = adj.shape[0]
    weights = adj * rng.uniform(0.5, 2.0, size=adj.shape)
    noise_std = np.ones(n)
    scm = LinearGaussianSCM(dag=dag, weights=weights, noise_std=noise_std)
    return scm.sample(n=n_samples, rng=rng)


# ---------------------------------------------------------------------------
# Benchmark Configs
# ---------------------------------------------------------------------------

TIER_CONFIGS = {
    "quick": [
        {"n_nodes": 5, "edge_prob": 0.3, "n_samples": 500, "iters": 200, "batch": 16},
        {"n_nodes": 8, "edge_prob": 0.25, "n_samples": 500, "iters": 200, "batch": 16},
    ],
    "standard": [
        {"n_nodes": 5, "edge_prob": 0.3, "n_samples": 500, "iters": 500, "batch": 20},
        {"n_nodes": 8, "edge_prob": 0.25, "n_samples": 800, "iters": 500, "batch": 20},
        {"n_nodes": 10, "edge_prob": 0.2, "n_samples": 1000, "iters": 500, "batch": 20},
        {"n_nodes": 15, "edge_prob": 0.15, "n_samples": 1500, "iters": 300, "batch": 32},
    ],
    "full": [
        {"n_nodes": 5, "edge_prob": 0.3, "n_samples": 500, "iters": 1000, "batch": 32},
        {"n_nodes": 8, "edge_prob": 0.25, "n_samples": 800, "iters": 1000, "batch": 32},
        {"n_nodes": 10, "edge_prob": 0.2, "n_samples": 1000, "iters": 1000, "batch": 32},
        {"n_nodes": 15, "edge_prob": 0.15, "n_samples": 1500, "iters": 500, "batch": 32},
        {"n_nodes": 20, "edge_prob": 0.1, "n_samples": 2000, "iters": 500, "batch": 32},
        {"n_nodes": 30, "edge_prob": 0.08, "n_samples": 3000, "iters": 300, "batch": 32},
    ],
    "stress": [
        {"n_nodes": 50, "edge_prob": 0.05, "n_samples": 5000, "iters": 200, "batch": 32},
        {"n_nodes": 75, "edge_prob": 0.04, "n_samples": 7500, "iters": 100, "batch": 32},
        {"n_nodes": 100, "edge_prob": 0.03, "n_samples": 10000, "iters": 100, "batch": 32},
    ],
}


# ---------------------------------------------------------------------------
# Benchmark Runners
# ---------------------------------------------------------------------------

def run_map_elites(adj: np.ndarray, data: np.ndarray,
                   iters: int, batch: int, seed: int = 42) -> Dict[str, Any]:
    """Run CausalQD MAP-Elites and return metrics."""
    config = MAPElitesConfig(
        archive_dims=(8, 8), seed=seed, log_interval=9999,
        adaptive_operators=True, mutation_prob=0.8,
    )
    mutations = [_make_mut(c) for c in [
        EdgeFlipMutation, EdgeAddMutation, EdgeRemoveMutation,
        TopologicalMutation, EdgeReverseMutation, VStructureMutation,
        SkeletonMutation, PathMutation,
    ]]
    crossovers = [_make_cross(c) for c in [
        UniformCrossover, SkeletonCrossover, MarkovBlanketCrossover,
    ]]

    engine = CausalMAPElites(
        mutations=mutations, crossovers=crossovers,
        descriptor_fn=_make_desc(), score_fn=_make_scorer(), config=config,
    )

    t0 = time.time()
    archive = engine.run(data=data, n_iterations=iters, batch_size=batch)
    elapsed = time.time() - t0

    best = archive.best()
    f1_m = F1()
    shd_val = int(SHD.compute(best.solution, adj)) if best else -1
    f1_val = float(f1_m.compute(best.solution, adj)) if best else 0.0

    # Count unique MECs
    converter = CPDAGConverter()
    seen_cpdags = set()
    for entry in archive.entries:
        try:
            dag_obj = DAG(entry.solution)
            cpdag = converter.dag_to_cpdag(dag_obj)
            seen_cpdags.add(cpdag.tobytes())
        except Exception:
            pass

    return {
        "algorithm": "CausalQD",
        "time": round(elapsed, 3),
        "shd": shd_val,
        "f1": round(f1_val, 4),
        "qd_score": round(archive.qd_score(), 4),
        "coverage": round(archive.coverage(), 4),
        "n_elites": archive.size,
        "n_unique_mecs": len(seen_cpdags),
        "best_quality": round(best.quality, 4) if best else None,
    }


def run_baseline(name: str, algo, adj: np.ndarray, data: np.ndarray) -> Dict[str, Any]:
    """Run a baseline algorithm and return metrics."""
    f1_m = F1()
    t0 = time.time()
    try:
        result_adj = algo.run(data)
        elapsed = time.time() - t0
        shd_val = int(SHD.compute(result_adj, adj))
        f1_val = float(f1_m.compute(result_adj, adj))
    except Exception as e:
        elapsed = time.time() - t0
        shd_val = -1
        f1_val = 0.0

    return {
        "algorithm": name,
        "time": round(elapsed, 4),
        "shd": shd_val,
        "f1": round(f1_val, 4),
        "qd_score": None,
        "coverage": None,
        "n_elites": 1,
        "n_unique_mecs": 1,
        "best_quality": None,
    }


def run_certificate_benchmark(adj: np.ndarray, data: np.ndarray) -> Dict[str, Any]:
    """Benchmark bootstrap certificate computation."""
    scorer = BICScore()
    score_fn = lambda a, d: scorer.score(a, d)
    cert = BootstrapCertificateComputer(n_bootstrap=100, score_fn=score_fn)

    t0 = time.time()
    certs = cert.compute_edge_certificates(adj, data)
    elapsed = time.time() - t0

    n_edges = int(adj.sum())
    freqs = [c.bootstrap_frequency for c in certs.values()]
    mean_freq = float(np.mean(freqs)) if freqs else 0.0

    return {
        "n_edges": n_edges,
        "n_certified": len(certs),
        "mean_bootstrap_freq": round(mean_freq, 4),
        "time": round(elapsed, 3),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CausalQD Benchmark Suite")
    parser.add_argument("--tier", default="standard",
                        choices=["quick", "standard", "full", "stress"])
    parser.add_argument("--output", default="benchmarks/results.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    configs = TIER_CONFIGS[args.tier]
    all_results: List[Dict[str, Any]] = []

    print(f"=== CausalQD Benchmark Suite (tier={args.tier}) ===\n")

    for cfg in configs:
        n = cfg["n_nodes"]
        print(f"--- n={n} nodes, edge_p={cfg['edge_prob']}, "
              f"samples={cfg['n_samples']} ---")

        adj = _random_dag(n, cfg["edge_prob"], rng)
        data = _generate_data(adj, cfg["n_samples"], rng)
        n_edges = int(adj.sum())
        print(f"  Ground truth: {n_edges} edges")

        # MAP-Elites
        me_result = run_map_elites(adj, data, cfg["iters"], cfg["batch"], args.seed)
        me_result.update({"n_nodes": n, "n_edges": n_edges,
                          "n_samples": cfg["n_samples"]})
        all_results.append(me_result)
        print(f"  CausalQD:  SHD={me_result['shd']:3d}  F1={me_result['f1']:.3f}  "
              f"elites={me_result['n_elites']}  MECs={me_result['n_unique_mecs']}  "
              f"time={me_result['time']:.1f}s")

        # Baselines
        for bname, bcls in [("PC", PCAlgorithm), ("GES", GESAlgorithm),
                            ("MMHC", MMHCAlgorithm)]:
            br = run_baseline(bname, bcls(), adj, data)
            br.update({"n_nodes": n, "n_edges": n_edges,
                       "n_samples": cfg["n_samples"]})
            all_results.append(br)
            print(f"  {bname:8s}:  SHD={br['shd']:3d}  F1={br['f1']:.3f}  "
                  f"time={br['time']:.3f}s")

        # Certificates
        if n <= 20:
            cert_result = run_certificate_benchmark(adj, data)
            print(f"  Certs: {cert_result['n_certified']} edges certified, "
                  f"mean_freq={cert_result['mean_bootstrap_freq']:.3f}, "
                  f"time={cert_result['time']:.1f}s")

        print()

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"tier": args.tier, "seed": args.seed,
                   "results": all_results}, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
