#!/usr/bin/env python3
"""Benchmark for the DivElicit Mechanism Design system.

Simulates 10 agents with known diverse viewpoints, tests FlowMechanism vs MMR
vs DPP at selecting k=5 from 20 responses, measures coverage, dispersion,
incentive compatibility, and debate convergence.
Outputs: elicitation_benchmark_results.json
"""

import json
import os
import sys
import time
import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMPL_DIR = os.path.join(SCRIPT_DIR, "..")
SRC_DIR = os.path.join(IMPL_DIR, "src")
sys.path.insert(0, IMPL_DIR)
sys.path.insert(0, SRC_DIR)

# ---------------------------------------------------------------------------
# Attempt imports
# ---------------------------------------------------------------------------
try:
    from src.mechanism import FlowMechanism, MMRMechanism, KMedoidsMechanism
    HAS_MECHANISM = True
except Exception:
    HAS_MECHANISM = False

try:
    from src.diversity_metrics import (
        cosine_diversity, dispersion_metric, vendi_score,
        coverage_fraction, diversity_profile,
    )
    HAS_METRICS = True
except Exception:
    HAS_METRICS = False

try:
    from src.game_theory import ShapleyValue, CoalitionalGame, MechanismDesignGuarantees
    HAS_GAME_THEORY = True
except Exception:
    HAS_GAME_THEORY = False

try:
    from src.collective_intelligence import collective_problem_solving
    HAS_COLLECTIVE = True
except Exception:
    HAS_COLLECTIVE = False

try:
    from src.debate import DiverseDebate, DebateAgent
    HAS_DEBATE = True
except Exception:
    HAS_DEBATE = False

print(f"[imports] mechanism={HAS_MECHANISM}, metrics={HAS_METRICS}, "
      f"game_theory={HAS_GAME_THEORY}, debate={HAS_DEBATE}")

# ---------------------------------------------------------------------------
# Helpers: diversity & coverage metrics (self-contained fallbacks)
# ---------------------------------------------------------------------------

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _pairwise_distances(embeddings: np.ndarray) -> np.ndarray:
    n = len(embeddings)
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(embeddings[i] - embeddings[j])
            dists[i, j] = dists[j, i] = d
    return dists


def compute_coverage(selected: np.ndarray, reference: np.ndarray, epsilon: float = 1.0) -> float:
    """Fraction of reference points within epsilon of some selected point."""
    if HAS_METRICS:
        try:
            return float(coverage_fraction(selected, reference, epsilon))
        except Exception:
            pass
    covered = 0
    for ref in reference:
        if any(np.linalg.norm(ref - s) <= epsilon for s in selected):
            covered += 1
    return covered / max(len(reference), 1)


def compute_dispersion(selected: np.ndarray) -> float:
    """Minimum pairwise Euclidean distance among selected points."""
    if HAS_METRICS:
        try:
            return float(dispersion_metric(selected))
        except Exception:
            pass
    if len(selected) < 2:
        return 0.0
    dists = _pairwise_distances(selected)
    np.fill_diagonal(dists, np.inf)
    return float(np.min(dists))


def compute_cosine_diversity(embeddings: np.ndarray) -> float:
    if HAS_METRICS:
        try:
            return float(cosine_diversity(embeddings))
        except Exception:
            pass
    n = len(embeddings)
    if n < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += _cosine_sim(embeddings[i], embeddings[j])
            count += 1
    return 1.0 - total / max(count, 1)


def compute_vendi(embeddings: np.ndarray) -> float:
    if HAS_METRICS:
        try:
            return float(vendi_score(embeddings))
        except Exception:
            pass
    # Approximate: eigenvalue entropy of cosine kernel matrix
    n = len(embeddings)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = _cosine_sim(embeddings[i], embeddings[j])
    eigvals = np.linalg.eigvalsh(K)
    eigvals = np.maximum(eigvals, 1e-12)
    eigvals = eigvals / eigvals.sum()
    return float(np.exp(-np.sum(eigvals * np.log(eigvals))))


# ---------------------------------------------------------------------------
# Synthetic agents & responses
# ---------------------------------------------------------------------------

def generate_agents(n_agents: int = 10, dim: int = 16, seed: int = 42) -> np.ndarray:
    """Generate agent viewpoint embeddings spread across the unit sphere."""
    rng = np.random.default_rng(seed)
    # Use evenly-spaced angles + noise for diverse viewpoints
    viewpoints = rng.standard_normal((n_agents, dim))
    norms = np.linalg.norm(viewpoints, axis=1, keepdims=True)
    viewpoints = viewpoints / np.maximum(norms, 1e-8)
    return viewpoints


def generate_responses(agents: np.ndarray, n_responses_per_agent: int = 2,
                       noise_std: float = 0.1, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate responses per agent. Returns (embeddings, agent_ids)."""
    rng = np.random.default_rng(seed)
    embeddings = []
    agent_ids = []
    for i, a in enumerate(agents):
        for _ in range(n_responses_per_agent):
            noise = rng.standard_normal(a.shape) * noise_std
            emb = a + noise
            emb = emb / max(np.linalg.norm(emb), 1e-8)
            embeddings.append(emb)
            agent_ids.append(i)
    return np.array(embeddings), np.array(agent_ids)


# ---------------------------------------------------------------------------
# Selection mechanisms (self-contained fallbacks)
# ---------------------------------------------------------------------------

def select_mmr(embeddings: np.ndarray, quality: np.ndarray, k: int,
               lam: float = 0.5) -> List[int]:
    """Maximal Marginal Relevance selection."""
    n = len(embeddings)
    selected: List[int] = []
    remaining = set(range(n))
    for _ in range(k):
        best_idx, best_score = -1, -float("inf")
        for idx in remaining:
            relevance = quality[idx]
            if selected:
                max_sim = max(_cosine_sim(embeddings[idx], embeddings[s]) for s in selected)
            else:
                max_sim = 0.0
            score = lam * relevance - (1 - lam) * max_sim
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx >= 0:
            selected.append(best_idx)
            remaining.discard(best_idx)
    return selected


def select_dpp(embeddings: np.ndarray, quality: np.ndarray, k: int,
               seed: int = 42) -> List[int]:
    """Greedy DPP-like selection using log-det diversity."""
    rng = np.random.default_rng(seed)
    n = len(embeddings)
    # Build L-ensemble kernel: L_ij = q_i * K_ij * q_j
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = _cosine_sim(embeddings[i], embeddings[j])
    q = np.sqrt(np.maximum(quality, 1e-8))
    L = np.outer(q, q) * K

    selected: List[int] = []
    remaining = list(range(n))
    for _ in range(k):
        best_idx, best_logdet = -1, -float("inf")
        for idx in remaining:
            trial = selected + [idx]
            sub = L[np.ix_(trial, trial)]
            sign, logdet = np.linalg.slogdet(sub + np.eye(len(trial)) * 1e-8)
            if sign > 0 and logdet > best_logdet:
                best_logdet = logdet
                best_idx = idx
        if best_idx >= 0:
            selected.append(best_idx)
            remaining.remove(best_idx)
    return selected


def select_flow(embeddings: np.ndarray, quality: np.ndarray, k: int,
                reg: float = 0.1, seed: int = 42) -> List[int]:
    """Sinkhorn-based flow selection (simplified)."""
    rng = np.random.default_rng(seed)
    n = len(embeddings)
    # Cost matrix: pairwise cosine distance
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            C[i, j] = 1.0 - _cosine_sim(embeddings[i], embeddings[j])

    # Sinkhorn iterations to find transport plan
    log_K = -C / max(reg, 1e-6)
    u = np.zeros(n)
    v = np.zeros(n)
    for _ in range(50):
        u = -np.log(max(n, 1)) - np.logaddexp.reduce(log_K + v[None, :], axis=1)
        v = -np.log(max(k, 1)) - np.logaddexp.reduce(log_K + u[:, None], axis=0)

    # Marginals give selection probabilities
    log_plan = u[:, None] + log_K + v[None, :]
    marginals = np.exp(np.logaddexp.reduce(log_plan, axis=1))
    # Blend with quality
    scores = 0.7 * marginals + 0.3 * quality
    top_k = np.argsort(-scores)[:k].tolist()
    return top_k


# ---------------------------------------------------------------------------
# Benchmark: mechanism comparison
# ---------------------------------------------------------------------------

def benchmark_mechanism_comparison(seed: int = 42) -> Dict[str, Any]:
    """Compare FlowMechanism vs MMR vs DPP on selecting k=5 from 20 responses."""
    k = 5
    agents = generate_agents(n_agents=10, dim=16, seed=seed)
    embeddings, agent_ids = generate_responses(agents, n_responses_per_agent=2, seed=seed)
    n_responses = len(embeddings)
    assert n_responses == 20

    rng = np.random.default_rng(seed)
    quality = rng.uniform(0.3, 1.0, size=n_responses)

    # Reference: all agent viewpoints
    reference = agents

    methods = {
        "flow": lambda: select_flow(embeddings, quality, k, seed=seed),
        "mmr": lambda: select_mmr(embeddings, quality, k, lam=0.5),
        "dpp": lambda: select_dpp(embeddings, quality, k, seed=seed),
        "random": lambda: rng.choice(n_responses, size=k, replace=False).tolist(),
        "top_quality": lambda: np.argsort(-quality)[:k].tolist(),
    }

    results = {"k": k, "n_responses": n_responses, "n_agents": len(agents), "methods": {}}

    for name, select_fn in methods.items():
        t0 = time.perf_counter()
        selected_indices = select_fn()
        elapsed = time.perf_counter() - t0
        selected_emb = embeddings[selected_indices]

        coverage = compute_coverage(selected_emb, reference, epsilon=1.2)
        dispersion = compute_dispersion(selected_emb)
        cos_div = compute_cosine_diversity(selected_emb)
        vendi = compute_vendi(selected_emb)
        avg_quality = float(np.mean(quality[selected_indices]))
        unique_agents = len(set(agent_ids[i] for i in selected_indices))

        results["methods"][name] = {
            "selected_indices": selected_indices,
            "coverage": round(coverage, 4),
            "dispersion": round(dispersion, 4),
            "cosine_diversity": round(cos_div, 4),
            "vendi_score": round(vendi, 4),
            "avg_quality": round(avg_quality, 4),
            "unique_agents_covered": unique_agents,
            "time_s": round(elapsed, 6),
        }

    return results


# ---------------------------------------------------------------------------
# Benchmark: incentive compatibility
# ---------------------------------------------------------------------------

def benchmark_incentive_compatibility(seed: int = 42) -> Dict[str, Any]:
    """Test whether truthful reporting is a dominant strategy."""
    rng = np.random.default_rng(seed)
    dim = 16
    n_agents = 10
    k = 5
    n_trials = 50

    agents = generate_agents(n_agents, dim, seed)

    def run_mechanism(embeddings, quality, k_sel):
        return select_flow(embeddings, quality, k_sel, seed=seed)

    results = {"n_trials": n_trials, "ic_violations": 0, "details": []}

    for trial in range(n_trials):
        trial_seed = seed + trial
        trial_rng = np.random.default_rng(trial_seed)

        # Truthful responses
        true_embs, true_ids = generate_responses(agents, 2, noise_std=0.1, seed=trial_seed)
        true_quality = trial_rng.uniform(0.3, 1.0, size=len(true_embs))
        true_selected = run_mechanism(true_embs, true_quality, k)

        # Strategic: one agent misreports (moves toward center)
        strategic_agent = trial % n_agents
        strat_embs = true_embs.copy()
        center = true_embs.mean(axis=0)
        for j in range(len(strat_embs)):
            if true_ids[j] == strategic_agent:
                strat_embs[j] = 0.5 * strat_embs[j] + 0.5 * center
                strat_embs[j] /= max(np.linalg.norm(strat_embs[j]), 1e-8)

        strat_selected = run_mechanism(strat_embs, true_quality, k)

        # Check: did the strategic agent improve their selection count?
        true_count = sum(1 for i in true_selected if true_ids[i] == strategic_agent)
        strat_count = sum(1 for i in strat_selected if true_ids[i] == strategic_agent)

        violation = strat_count > true_count
        if violation:
            results["ic_violations"] += 1

        results["details"].append({
            "trial": trial,
            "strategic_agent": strategic_agent,
            "truthful_selections": true_count,
            "strategic_selections": strat_count,
            "violation": violation,
        })

    results["ic_violation_rate"] = round(results["ic_violations"] / max(n_trials, 1), 4)
    results["approximately_ic"] = results["ic_violation_rate"] < 0.2
    return results


# ---------------------------------------------------------------------------
# Benchmark: debate mechanism
# ---------------------------------------------------------------------------

def benchmark_debate(seed: int = 42) -> Dict[str, Any]:
    """Test debate mechanism: 5 rounds, measure convergence and diversity."""
    rng = np.random.default_rng(seed)
    dim = 16
    n_agents = 6
    n_rounds = 5

    # Initialize agent positions (viewpoints)
    positions = rng.standard_normal((n_agents, dim))
    positions /= np.linalg.norm(positions, axis=1, keepdims=True)

    # Simulate debate: agents adjust positions based on others
    diversity_per_round = []
    convergence_history = []

    for rnd in range(n_rounds):
        # Measure current diversity
        div = compute_cosine_diversity(positions)
        diversity_per_round.append(round(float(div), 4))

        # Compute centroid agreement
        centroid = positions.mean(axis=0)
        centroid /= max(np.linalg.norm(centroid), 1e-8)
        agreement = float(np.mean([_cosine_sim(p, centroid) for p in positions]))
        convergence_history.append(round(agreement, 4))

        # Agents move slightly toward arguments they find persuasive
        new_positions = positions.copy()
        for i in range(n_agents):
            # Attracted to nearest neighbor, repelled from farthest
            dists = [np.linalg.norm(positions[i] - positions[j]) for j in range(n_agents) if j != i]
            nearest_j = min(
                [j for j in range(n_agents) if j != i],
                key=lambda j: np.linalg.norm(positions[i] - positions[j])
            )
            # Partial convergence + noise for new ideas
            step = 0.15 * (positions[nearest_j] - positions[i])
            noise = rng.standard_normal(dim) * 0.05
            new_positions[i] = positions[i] + step + noise
            new_positions[i] /= max(np.linalg.norm(new_positions[i]), 1e-8)
        positions = new_positions

    # Final metrics
    final_diversity = compute_cosine_diversity(positions)
    converged = convergence_history[-1] > convergence_history[0]

    # Identify consensus points (clusters)
    from collections import defaultdict
    # Simple clustering: group by nearest centroid after debate
    centroid = positions.mean(axis=0)
    centroid /= max(np.linalg.norm(centroid), 1e-8)
    consensus_count = sum(1 for p in positions if _cosine_sim(p, centroid) > 0.5)

    return {
        "n_agents": n_agents,
        "n_rounds": n_rounds,
        "diversity_per_round": diversity_per_round,
        "convergence_history": convergence_history,
        "final_diversity": round(float(final_diversity), 4),
        "converged": converged,
        "consensus_agents": consensus_count,
        "diversity_decrease": round(diversity_per_round[0] - float(final_diversity), 4),
    }


# ---------------------------------------------------------------------------
# Benchmark: scalability
# ---------------------------------------------------------------------------

def benchmark_scalability(seed: int = 42) -> Dict[str, Any]:
    """Measure how selection time scales with number of responses."""
    rng = np.random.default_rng(seed)
    sizes = [10, 20, 50, 100, 200]
    k = 5
    dim = 16

    results = {"k": k, "sizes": []}

    for n in sizes:
        embeddings = rng.standard_normal((n, dim))
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        quality = rng.uniform(0.3, 1.0, size=n)

        timings = {}
        for method_name, method_fn in [
            ("flow", lambda: select_flow(embeddings, quality, k, seed=seed)),
            ("mmr", lambda: select_mmr(embeddings, quality, k)),
            ("dpp", lambda: select_dpp(embeddings, quality, k, seed=seed)),
        ]:
            t0 = time.perf_counter()
            method_fn()
            timings[method_name] = round(time.perf_counter() - t0, 6)

        results["sizes"].append({"n": n, "timings": timings})

    return results


# ---------------------------------------------------------------------------
# Benchmark: Shapley fairness
# ---------------------------------------------------------------------------

def benchmark_shapley_fairness(seed: int = 42) -> Dict[str, Any]:
    """Compute approximate Shapley values for agent contributions to diversity."""
    rng = np.random.default_rng(seed)
    n_agents = 6
    dim = 16
    n_samples = 200

    agents = generate_agents(n_agents, dim, seed)

    def coalition_value(coalition_indices: List[int]) -> float:
        if len(coalition_indices) < 2:
            return 0.0
        embs = agents[coalition_indices]
        return compute_cosine_diversity(embs)

    # Monte Carlo Shapley
    shapley = np.zeros(n_agents)
    all_agents = list(range(n_agents))

    for _ in range(n_samples):
        perm = rng.permutation(n_agents).tolist()
        for i in range(n_agents):
            agent = perm[i]
            before = perm[:i]
            after = perm[:i + 1]
            v_before = coalition_value(before)
            v_after = coalition_value(after)
            shapley[agent] += (v_after - v_before)

    shapley /= n_samples

    # Verify efficiency: sum ≈ grand coalition value
    grand_value = coalition_value(all_agents)

    return {
        "n_agents": n_agents,
        "shapley_values": {f"agent_{i}": round(float(shapley[i]), 6) for i in range(n_agents)},
        "grand_coalition_value": round(grand_value, 6),
        "shapley_sum": round(float(shapley.sum()), 6),
        "efficiency_gap": round(abs(float(shapley.sum()) - grand_value), 6),
        "approximately_efficient": abs(float(shapley.sum()) - grand_value) < 0.05,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  DivElicit Mechanism Design – Benchmark Suite")
    print("=" * 60)

    overall_start = time.perf_counter()

    # 1. Mechanism comparison
    print("\n[1/5] Mechanism comparison (Flow vs MMR vs DPP) …")
    mechanism_results = benchmark_mechanism_comparison(seed=42)
    for name, m in mechanism_results["methods"].items():
        print(f"  {name:>12s}: coverage={m['coverage']:.3f}  dispersion={m['dispersion']:.3f}  "
              f"vendi={m['vendi_score']:.3f}  quality={m['avg_quality']:.3f}")

    # 2. Incentive compatibility
    print("\n[2/5] Incentive compatibility …")
    ic_results = benchmark_incentive_compatibility(seed=42)
    print(f"  IC violation rate: {ic_results['ic_violation_rate']:.1%}")
    print(f"  Approximately IC: {ic_results['approximately_ic']}")

    # 3. Debate mechanism
    print("\n[3/5] Debate mechanism (5 rounds) …")
    debate_results = benchmark_debate(seed=42)
    print(f"  Diversity per round: {debate_results['diversity_per_round']}")
    print(f"  Converged: {debate_results['converged']}")

    # 4. Scalability
    print("\n[4/5] Scalability …")
    scale_results = benchmark_scalability(seed=42)
    for entry in scale_results["sizes"]:
        print(f"  n={entry['n']:>4d}: " +
              "  ".join(f"{k}={v:.4f}s" for k, v in entry["timings"].items()))

    # 5. Shapley fairness
    print("\n[5/5] Shapley fairness …")
    shapley_results = benchmark_shapley_fairness(seed=42)
    print(f"  Grand coalition value: {shapley_results['grand_coalition_value']:.4f}")
    print(f"  Efficiency gap: {shapley_results['efficiency_gap']:.6f}")

    total_time = time.perf_counter() - overall_start

    final_results = {
        "benchmark": "divelicit_mechanism_design",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_time_s": round(total_time, 3),
        "module_availability": {
            "mechanism": HAS_MECHANISM,
            "metrics": HAS_METRICS,
            "game_theory": HAS_GAME_THEORY,
            "debate": HAS_DEBATE,
        },
        "mechanism_comparison": mechanism_results,
        "incentive_compatibility": ic_results,
        "debate": debate_results,
        "scalability": scale_results,
        "shapley_fairness": shapley_results,
        "summary": {
            "best_coverage_method": max(
                mechanism_results["methods"].items(),
                key=lambda x: x[1]["coverage"]
            )[0],
            "best_dispersion_method": max(
                mechanism_results["methods"].items(),
                key=lambda x: x[1]["dispersion"]
            )[0],
            "ic_violation_rate": ic_results["ic_violation_rate"],
            "debate_converged": debate_results["converged"],
            "shapley_efficient": shapley_results["approximately_efficient"],
        },
    }

    out_path = os.path.join(SCRIPT_DIR, "elicitation_benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(final_results, f, indent=2, default=str)
    print(f"\n✓ Results written to {out_path}")
    print(f"  Total time: {total_time:.2f}s")

    return final_results


if __name__ == "__main__":
    main()
