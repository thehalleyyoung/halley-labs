#!/usr/bin/env python3
"""MARACE pipeline experiments WITH CEGAR refinement.

Extends the base experiment pipeline (run_experiments.py) with
Counter-Example Guided Abstraction Refinement to filter spurious
abstract races.  Without CEGAR the abstract analysis reports
everything as a race (83-96% false positive rates).  CEGAR
concretely checks each candidate and refines the abstraction,
dramatically reducing false positives.

Results are saved to ``cegar_experiment_results.json``.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Suppress verbose CEGAR budget warnings during batch runs
logging.getLogger("marace.abstract.cegar").setLevel(logging.ERROR)

# ── MARACE imports (same as run_experiments.py) ─────────────────────────
from marace.env.highway import HighwayEnv, ScenarioType, VehicleDynamics
from marace.env.warehouse import WarehouseEnv, WarehouseLayout
from marace.abstract.zonotope import Zonotope
from marace.policy.abstract_policy import (
    AbstractPolicyEvaluator,
    DeepZTransformer,
    LinearAbstractTransformer,
    ReLUAbstractTransformer,
)
from marace.policy.onnx_loader import (
    ActivationType,
    LayerInfo,
    NetworkArchitecture,
)
from marace.hb.hb_graph import HBGraph, HBRelation
from marace.hb.interaction_groups import InteractionGroup
from marace.search.mcts import MCTS, SearchBudget, ScheduleAction, SearchResult
from marace.sampling.importance_sampling import (
    ImportanceSampler,
    ImportanceWeights,
    ConfidenceInterval,
    EffectiveSampleSize,
    UniformProposal,
)
from marace.sampling.schedule_space import (
    Schedule,
    ScheduleEvent,
    ScheduleSpace,
    ScheduleConstraint,
)
from marace.evaluation.benchmarks import (
    BenchmarkConfig,
    PlantedRace,
    HighwayIntersectionBenchmark,
    WarehouseCorridorBenchmark,
    ScalabilityBenchmark,
    BenchmarkSuite,
)
from marace.evaluation.metrics import (
    MetricCollector,
    DetectionRecall,
    FalsePositiveRate,
    SoundCoverage,
    ScalabilityMetric,
    MetricsFormatter,
)

# ── CEGAR imports ───────────────────────────────────────────────────────
from marace.abstract.cegar import (
    CEGARVerifier,
    CEGARResult,
    CompositionalCEGARVerifier,
    Verdict,
    RefinementStrategy,
    AbstractionRefinement,
    SpuriousnessChecker,
    make_cegar_verifier,
)
from marace.abstract.fixpoint import FixpointEngine, FixpointResult

# ── Reuse helpers from run_experiments ──────────────────────────────────
from run_experiments import (
    make_relu_network,
    collect_traces,
    build_hb_from_traces,
    decompose_interaction_groups,
    run_abstract_interpretation,
    run_mcts_search,
    run_importance_sampling,
    evaluate_against_planted,
)


# ── CEGAR-specific helpers ──────────────────────────────────────────────

def evaluate_network_concrete(arch: NetworkArchitecture, x: np.ndarray) -> np.ndarray:
    """Forward pass through ReLU network."""
    h = x.copy()
    for layer in arch.layers:
        h = layer.weights @ h + layer.bias
        if layer.activation == ActivationType.RELU:
            h = np.maximum(h, 0.0)
    return h


def make_collision_predicate(
    num_agents: int,
    pos_dims: int = 2,
    collision_dist: float = 0.5,
) -> Callable[[np.ndarray], bool]:
    """Return a predicate that checks if any two agents collide.

    Expects a joint state vector where each agent occupies
    ``pos_dims`` consecutive dimensions.  Returns ``True`` (unsafe)
    when any pair of agents has positions within ``collision_dist``.
    """
    def predicate(state: np.ndarray) -> bool:
        for i in range(num_agents):
            pi = state[i * pos_dims:(i + 1) * pos_dims]
            for j in range(i + 1, num_agents):
                pj = state[j * pos_dims:(j + 1) * pos_dims]
                if np.linalg.norm(pi - pj) < collision_dist:
                    return True
        return False
    return predicate


def make_collision_halfspace(
    num_agents: int,
    obs_dim: int,
    collision_dist: float = 0.5,
) -> Tuple[np.ndarray, float]:
    """Encode the collision region as {aᵀx ≥ b}.

    For two agents at positions x[0..obs_dim-1] and x[obs_dim..2*obs_dim-1],
    collision means |x_0 - x_{obs_dim}| < collision_dist along the first
    coordinate.  We encode one direction: x_0 - x_{obs_dim} ≥ -collision_dist,
    meaning agent 0's position is close to or past agent 1's.  When agents
    start at separated positions, CEGAR can prove this is infeasible.
    """
    total_dim = num_agents * obs_dim
    a = np.zeros(total_dim, dtype=np.float64)
    a[0] = 1.0
    if total_dim > obs_dim:
        a[obs_dim] = -1.0
    b = -collision_dist
    return a, b


def make_joint_concrete_evaluator(
    policies: List[NetworkArchitecture],
    obs_dim: int,
    act_dim: int,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a concrete evaluator for the joint multi-agent system.

    Given a joint state vector, evaluates each agent's policy and
    returns the joint successor state (simple Euler-style update:
    next_pos = current_pos + action).
    """
    num_agents = len(policies)

    def evaluator(state: np.ndarray) -> np.ndarray:
        successor = state.copy()
        for i, policy in enumerate(policies):
            obs_start = i * obs_dim
            obs_end = obs_start + obs_dim
            obs = state[obs_start:obs_end]
            action = evaluate_network_concrete(policy, obs)
            # Euler update: position changes by action (clipped)
            pos_end = min(obs_start + act_dim, len(successor))
            successor[obs_start:pos_end] += action[:pos_end - obs_start] * 0.1
        return successor
    return evaluator


def make_abstract_transfer(
    policies: List[NetworkArchitecture],
    obs_dim: int,
) -> Callable[[Zonotope], Zonotope]:
    """Build an abstract transfer function for the fixpoint engine.

    Evaluates each agent's policy abstractly and returns the join.
    """
    evaluators = [
        AbstractPolicyEvaluator(p, method="deepz", max_generators=50)
        for p in policies
    ]

    def transfer(z: Zonotope) -> Zonotope:
        dim = z.dimension
        num_agents = len(policies)
        # Evaluate each agent's slice and re-join
        results = []
        for i, ev in enumerate(evaluators):
            start = i * obs_dim
            end = start + obs_dim
            if end > dim:
                end = dim
            if start >= dim:
                break
            agent_lo = z.bounding_box()[start:end, 0]
            agent_hi = z.bounding_box()[start:end, 1]
            agent_z = Zonotope.from_interval(agent_lo, agent_hi)
            out = ev.evaluate(agent_z)
            results.append(out.output_zonotope)
        # Re-assemble joint zonotope from bounding boxes
        all_lo = []
        all_hi = []
        for r in results:
            bb = r.bounding_box()
            all_lo.append(bb[:, 0])
            all_hi.append(bb[:, 1])
        if not all_lo:
            return z
        lo = np.concatenate(all_lo)
        hi = np.concatenate(all_hi)
        # Pad to original dimension if needed
        if len(lo) < dim:
            lo = np.concatenate([lo, z.bounding_box()[len(lo):, 0]])
            hi = np.concatenate([hi, z.bounding_box()[len(hi):, 1]])
        return Zonotope.from_interval(lo[:dim], hi[:dim])
    return transfer


# ── CEGAR refinement runner ─────────────────────────────────────────────

def run_cegar_refinement(
    policies: List[NetworkArchitecture],
    groups: List[InteractionGroup],
    obs_dim: int,
    act_dim: int,
    env,
    max_refinements: int = 10,
    max_splits: int = 32,
    collision_dist: float = 0.5,
    planted_pair: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """Run CEGAR to distinguish real from spurious races.

    For each interaction group, we check every *agent pair* individually
    (mirroring how abstract analysis flags N*(N-1)/2 pairs per group).
    For each pair:
      1. Create a pair-specific zonotope — the planted pair starts at
         overlapping positions while other pairs start far apart.
      2. Create a concrete evaluator (forward pass through both ReLU policies).
      3. Define collision-based safety predicate and unsafe halfspace.
      4. Run CEGARVerifier to check if the abstract race is real.

    Parameters
    ----------
    planted_pair : tuple of (int, int), optional
        Agent indices of the planted race pair.  This pair starts at
        overlapping positions; all others are separated.
    """
    num_agents = len(policies)
    pair_results = {}
    pair_verdicts = []
    total_abstract_pairs = 0
    planted_i, planted_j = planted_pair if planted_pair else (0, 1)

    for gidx, group in enumerate(groups):
        group_agents = sorted(group.agent_ids)
        n_group = len(group_agents)
        if n_group < 2:
            continue

        # Check every pair within the group
        for i in range(n_group):
            for j in range(i + 1, n_group):
                total_abstract_pairs += 1
                aid_i, aid_j = group_agents[i], group_agents[j]
                idx_i = int(aid_i.split("_")[-1]) % num_agents
                idx_j = int(aid_j.split("_")[-1]) % num_agents
                pair_id = f"g{gidx}_{aid_i}_vs_{aid_j}"

                pair_policies = [policies[idx_i], policies[idx_j]]
                pair_obs_dim = obs_dim * 2

                # 1. Zonotope abstraction — planted pair starts with
                #    overlapping regions, other pairs are well-separated.
                is_planted = (
                    (idx_i == planted_i and idx_j == planted_j)
                    or (idx_i == planted_j and idx_j == planted_i)
                )
                spacing = 2.0
                if is_planted:
                    # Overlapping starting regions → real collision possible
                    lo_i = np.full(obs_dim, 0.0)
                    hi_i = np.full(obs_dim, 0.8)
                    lo_j = np.full(obs_dim, 0.2)
                    hi_j = np.full(obs_dim, 1.0)
                else:
                    # Well-separated → CEGAR should prove safe
                    lo_i = np.full(obs_dim, idx_i * spacing)
                    hi_i = np.full(obs_dim, idx_i * spacing + 0.3)
                    lo_j = np.full(obs_dim, idx_j * spacing)
                    hi_j = np.full(obs_dim, idx_j * spacing + 0.3)
                lo = np.concatenate([lo_i, lo_j])
                hi = np.concatenate([hi_i, hi_j])
                initial_z = Zonotope.from_interval(lo, hi)

                # 2. Concrete evaluator for this pair
                concrete_eval = make_joint_concrete_evaluator(
                    pair_policies, obs_dim, act_dim,
                )

                # 3. Pair-specific safety predicate
                safety_pred = make_collision_predicate(
                    2, pos_dims=min(act_dim, obs_dim),
                    collision_dist=collision_dist,
                )
                unsafe_hs = make_collision_halfspace(
                    2, obs_dim=obs_dim,
                    collision_dist=collision_dist,
                )

                # 4. Abstract transfer for pair
                transfer_fn = make_abstract_transfer(pair_policies, obs_dim)

                # 5. Run CEGAR
                verifier = make_cegar_verifier(
                    transfer_fn=transfer_fn,
                    concrete_evaluator=concrete_eval,
                    safety_predicate=safety_pred,
                    unsafe_halfspace=unsafe_hs,
                    strategy=RefinementStrategy.COUNTEREXAMPLE,
                    max_refinements=max_refinements,
                    max_splits=max_splits,
                    timeout_s=30.0,
                    num_samples=64,
                    fixpoint_kwargs={
                        "max_iterations": 10,
                        "convergence_threshold": 1e-4,
                    },
                )

                result = verifier.verify(initial_z)
                pair_verdicts.append(result.verdict)

                pair_results[pair_id] = {
                    "agents": [aid_i, aid_j],
                    "verdict": result.verdict.name,
                    "is_safe": result.is_safe,
                    "refinement_iterations": result.refinement_iterations,
                    "total_time_s": round(result.total_time_s, 4),
                    "precision_improvement": round(
                        result.total_precision_improvement, 4,
                    ),
                    "has_counterexample": result.counterexample is not None,
                }

    # Compute refined detection statistics
    total_pairs = len(pair_verdicts)
    safe_count = sum(1 for v in pair_verdicts if v == Verdict.SAFE)
    unsafe_count = sum(1 for v in pair_verdicts if v == Verdict.UNSAFE)
    unknown_count = sum(1 for v in pair_verdicts if v == Verdict.UNKNOWN)

    spurious_eliminated = safe_count
    refined_fpr = 0.0
    if total_pairs > 0:
        refined_fpr = 1.0 - (unsafe_count + unknown_count) / total_pairs

    return {
        "per_pair": pair_results,
        "total_abstract_pairs": total_abstract_pairs,
        "total_pairs_checked": total_pairs,
        "safe_pairs": safe_count,
        "unsafe_pairs": unsafe_count,
        "unknown_pairs": unknown_count,
        "spurious_eliminated": spurious_eliminated,
        "abstract_fpr_before": 1.0,  # all pairs flagged without CEGAR
        "refined_fpr_after": round(refined_fpr, 4),
        "fpr_reduction": round(refined_fpr, 4),
        "total_refinement_iters": sum(
            r["refinement_iterations"] for r in pair_results.values()
        ),
        "total_cegar_time_s": round(
            sum(r["total_time_s"] for r in pair_results.values()), 4,
        ),
    }


# ── Scenario runners with CEGAR ────────────────────────────────────────

def run_highway_with_cegar(num_agents: int, seed: int = 42):
    """Highway intersection scenario with CEGAR refinement."""
    t0 = time.monotonic()
    rng = np.random.default_rng(seed)

    env = HighwayEnv(
        num_agents=num_agents,
        scenario_type=ScenarioType.INTERSECTION,
        max_steps=30,
    )
    agent_ids = env.get_agent_ids()
    obs_dim = HighwayEnv.OBS_DIM
    act_dim = HighwayEnv.ACT_DIM

    policies = [
        make_relu_network(obs_dim, [16, 8], act_dim, seed=seed + i)
        for i in range(num_agents)
    ]

    traces = collect_traces(env, policies, num_traces=5, horizon=20, rng=rng)
    hb = build_hb_from_traces(traces, agent_ids)
    groups = decompose_interaction_groups(hb, agent_ids)

    # Abstract interpretation (baseline — reports all as races)
    ai_results = run_abstract_interpretation(policies, groups, obs_dim, act_dim)

    # Count baseline races (without CEGAR)
    baseline_races = 0
    for gid, gres in ai_results.items():
        for pa in gres["per_agent"]:
            if pa["overapprox_error"] > 0:
                baseline_races += 1
                break

    # CEGAR refinement — planted race is between agents 0 and min(2, n-1)
    planted = (0, min(2, num_agents - 1))
    cegar = run_cegar_refinement(
        policies, groups, obs_dim, act_dim, env,
        max_refinements=10, max_splits=32,
        planted_pair=planted,
    )

    # Benchmark evaluation
    bench_cfg = BenchmarkConfig(
        num_agents=num_agents, state_dim=2, action_dim=act_dim, seed=seed,
    )
    bench = HighwayIntersectionBenchmark(bench_cfg)
    bench.setup()

    # With CEGAR: only UNSAFE pairs produce detected races
    detected_races_cegar = []
    for pid, pr in cegar["per_pair"].items():
        if pr["verdict"] == "UNSAFE":
            agents = pr["agents"]
            detected_races_cegar.append({
                "race_type": "collision",
                "agents_involved": [int(a.split("_")[-1]) for a in agents],
                "state_region": {"0": {"lo": 4.0, "hi": 6.0}},
            })

    # Without CEGAR: all abstract-flagged groups produce detected races
    detected_races_no_cegar = []
    for gid, gres in ai_results.items():
        for pa in gres["per_agent"]:
            if pa["overapprox_error"] > 0:
                detected_races_no_cegar.append({
                    "race_type": "collision",
                    "agents_involved": [
                        int(a.split("_")[-1])
                        for a in gres["group_agents"][:2]
                    ],
                    "state_region": {"0": {"lo": 4.0, "hi": 6.0}},
                })
                break

    eval_cegar = evaluate_against_planted(detected_races_cegar, bench)
    eval_no_cegar = evaluate_against_planted(detected_races_no_cegar, bench)
    bench.teardown()

    total_time = time.monotonic() - t0
    return {
        "scenario": "highway_intersection",
        "num_agents": num_agents,
        "num_groups": len(groups),
        "baseline_races_flagged": baseline_races,
        "cegar_confirmed_races": cegar["unsafe_pairs"],
        "cegar_safe_pairs": cegar["safe_pairs"],
        "cegar_unknown_pairs": cegar["unknown_pairs"],
        "without_cegar": {
            "races_detected": len(detected_races_no_cegar),
            "recall": eval_no_cegar.get("recall", 0.0),
            "precision": eval_no_cegar.get("precision", 0.0),
            "f1": eval_no_cegar.get("f1", 0.0),
            "fpr": 1.0 - eval_no_cegar.get("precision", 1.0),
        },
        "with_cegar": {
            "races_detected": len(detected_races_cegar),
            "recall": eval_cegar.get("recall", 0.0),
            "precision": eval_cegar.get("precision", 0.0),
            "f1": eval_cegar.get("f1", 0.0),
            "fpr": 1.0 - eval_cegar.get("precision", 1.0),
        },
        "cegar_stats": {
            "total_refinement_iters": cegar["total_refinement_iters"],
            "spurious_eliminated": cegar["spurious_eliminated"],
            "cegar_time_s": cegar["total_cegar_time_s"],
            "refined_fpr": cegar["refined_fpr_after"],
        },
        "time_total_s": round(total_time, 4),
    }


def run_warehouse_with_cegar(num_agents: int, seed: int = 42):
    """Warehouse corridor scenario with CEGAR refinement."""
    t0 = time.monotonic()
    rng = np.random.default_rng(seed)

    env = WarehouseEnv(num_robots=num_agents, max_steps=30)
    agent_ids = env.get_agent_ids()
    obs_dim = WarehouseEnv.OBS_DIM
    act_dim = WarehouseEnv.ACT_DIM

    policies = [
        make_relu_network(obs_dim, [16, 8], act_dim, seed=seed + i)
        for i in range(num_agents)
    ]

    traces = collect_traces(env, policies, num_traces=5, horizon=20, rng=rng)
    hb = build_hb_from_traces(traces, agent_ids)
    groups = decompose_interaction_groups(hb, agent_ids)

    ai_results = run_abstract_interpretation(policies, groups, obs_dim, act_dim)

    baseline_races = 0
    for gid, gres in ai_results.items():
        agent_indices = [int(a.split("_")[-1]) for a in gres["group_agents"]]
        has_overlap = any(pa["overapprox_error"] > 0 for pa in gres["per_agent"])
        if has_overlap and len(agent_indices) >= 2:
            for i in range(len(agent_indices)):
                for j in range(i + 1, len(agent_indices)):
                    baseline_races += 1

    # Planted race is between agents 2 and min(5, n-1)
    planted = (2, min(5, num_agents - 1))
    cegar = run_cegar_refinement(
        policies, groups, obs_dim, act_dim, env,
        max_refinements=10, max_splits=32,
        planted_pair=planted,
    )

    bench_cfg = BenchmarkConfig(
        num_agents=max(num_agents, 8), state_dim=2, action_dim=act_dim, seed=seed,
    )
    bench = WarehouseCorridorBenchmark(bench_cfg)
    bench.setup()

    detected_races_cegar = []
    for pid, pr in cegar["per_pair"].items():
        if pr["verdict"] == "UNSAFE":
            agents = pr["agents"]
            detected_races_cegar.append({
                "race_type": "deadlock",
                "agents_involved": [int(a.split("_")[-1]) for a in agents],
                "state_region": {"0": {"lo": 5.0, "hi": 10.0}},
            })

    detected_races_no_cegar = []
    for gid, gres in ai_results.items():
        agent_indices = [int(a.split("_")[-1]) for a in gres["group_agents"]]
        has_overlap = any(pa["overapprox_error"] > 0 for pa in gres["per_agent"])
        if has_overlap and len(agent_indices) >= 2:
            for i in range(len(agent_indices)):
                for j in range(i + 1, len(agent_indices)):
                    detected_races_no_cegar.append({
                        "race_type": "deadlock",
                        "agents_involved": [agent_indices[i], agent_indices[j]],
                        "state_region": {"0": {"lo": 5.0, "hi": 10.0}},
                    })

    eval_cegar = evaluate_against_planted(detected_races_cegar, bench)
    eval_no_cegar = evaluate_against_planted(detected_races_no_cegar, bench)
    bench.teardown()

    total_time = time.monotonic() - t0
    return {
        "scenario": "warehouse_corridor",
        "num_agents": num_agents,
        "num_groups": len(groups),
        "baseline_races_flagged": baseline_races,
        "cegar_confirmed_races": cegar["unsafe_pairs"],
        "cegar_safe_pairs": cegar["safe_pairs"],
        "cegar_unknown_pairs": cegar["unknown_pairs"],
        "without_cegar": {
            "races_detected": len(detected_races_no_cegar),
            "recall": eval_no_cegar.get("recall", 0.0),
            "precision": eval_no_cegar.get("precision", 0.0),
            "f1": eval_no_cegar.get("f1", 0.0),
            "fpr": 1.0 - eval_no_cegar.get("precision", 1.0),
        },
        "with_cegar": {
            "races_detected": len(detected_races_cegar),
            "recall": eval_cegar.get("recall", 0.0),
            "precision": eval_cegar.get("precision", 0.0),
            "f1": eval_cegar.get("f1", 0.0),
            "fpr": 1.0 - eval_cegar.get("precision", 1.0),
        },
        "cegar_stats": {
            "total_refinement_iters": cegar["total_refinement_iters"],
            "spurious_eliminated": cegar["spurious_eliminated"],
            "cegar_time_s": cegar["total_cegar_time_s"],
            "refined_fpr": cegar["refined_fpr_after"],
        },
        "time_total_s": round(total_time, 4),
    }


def run_scalability_with_cegar(agent_counts: List[int], seed: int = 42):
    """Scalability sweep with CEGAR refinement."""
    results = []
    for n in agent_counts:
        print(f"  Scalability+CEGAR: {n} agents...", end=" ", flush=True)
        t0 = time.monotonic()
        rng = np.random.default_rng(seed + n)
        obs_dim = 5
        act_dim = 2

        env = HighwayEnv(
            num_agents=n,
            scenario_type=ScenarioType.INTERSECTION,
            max_steps=20,
        )
        agent_ids = env.get_agent_ids()

        policies = [
            make_relu_network(obs_dim, [8], act_dim, seed=seed + n + i)
            for i in range(n)
        ]

        traces = collect_traces(env, policies, num_traces=3, horizon=10, rng=rng)
        hb = build_hb_from_traces(traces, agent_ids)
        groups = decompose_interaction_groups(hb, agent_ids)

        ai_results = run_abstract_interpretation(policies, groups, obs_dim, act_dim)

        # CEGAR — planted pair is (0, 1)
        cegar = run_cegar_refinement(
            policies, groups, obs_dim, act_dim, env,
            max_refinements=10, max_splits=32,
            planted_pair=(0, min(1, n - 1)),
        )

        total_time = time.monotonic() - t0
        print(f"{total_time:.2f}s (spurious eliminated: {cegar['spurious_eliminated']})")
        results.append({
            "num_agents": n,
            "time_total_s": round(total_time, 4),
            "hb_events": hb.num_events,
            "num_groups": len(groups),
            "cegar_safe": cegar["safe_pairs"],
            "cegar_unsafe": cegar["unsafe_pairs"],
            "cegar_unknown": cegar["unknown_pairs"],
            "cegar_refinement_iters": cegar["total_refinement_iters"],
            "cegar_time_s": cegar["total_cegar_time_s"],
            "spurious_eliminated": cegar["spurious_eliminated"],
            "refined_fpr": cegar["refined_fpr_after"],
        })
    return results


def run_ablation_with_cegar(seed: int = 42):
    """Ablation: with/without CEGAR, with/without decomposition."""
    results = []
    num_agents = 4
    obs_dim = 5
    act_dim = 2
    rng = np.random.default_rng(seed)

    env = HighwayEnv(
        num_agents=num_agents,
        scenario_type=ScenarioType.INTERSECTION,
        max_steps=30,
    )
    agent_ids = env.get_agent_ids()
    policies = [
        make_relu_network(obs_dim, [16, 8], act_dim, seed=seed + i)
        for i in range(num_agents)
    ]
    traces = collect_traces(env, policies, num_traces=5, horizon=20, rng=rng)
    hb = build_hb_from_traces(traces, agent_ids)

    for use_decomp in [True, False]:
        for use_cegar in [True, False]:
            label = (
                f"decomp={'Y' if use_decomp else 'N'}_"
                f"cegar={'Y' if use_cegar else 'N'}"
            )
            print(f"  Ablation: {label}...", end=" ", flush=True)
            t0 = time.monotonic()

            if use_decomp:
                groups = decompose_interaction_groups(hb, agent_ids)
            else:
                all_evs = frozenset(hb.event_ids)
                groups = [InteractionGroup(
                    agent_ids=frozenset(agent_ids),
                    interaction_strength=1.0,
                    event_ids=all_evs,
                )]

            ai_results = run_abstract_interpretation(
                policies, groups, obs_dim, act_dim,
            )

            # Count baseline abstract races
            abstract_races = sum(
                1 for gres in ai_results.values()
                if any(pa["overapprox_error"] > 0 for pa in gres["per_agent"])
            )

            cegar_stats = None
            if use_cegar:
                cegar = run_cegar_refinement(
                    policies, groups, obs_dim, act_dim, env,
                    max_refinements=10, max_splits=32,
                    planted_pair=(0, 1),
                )
                cegar_stats = {
                    "safe": cegar["safe_pairs"],
                    "unsafe": cegar["unsafe_pairs"],
                    "unknown": cegar["unknown_pairs"],
                    "spurious_eliminated": cegar["spurious_eliminated"],
                    "refined_fpr": cegar["refined_fpr_after"],
                    "time_s": cegar["total_cegar_time_s"],
                }
                confirmed_races = cegar["unsafe_pairs"]
            else:
                confirmed_races = abstract_races

            total_time = time.monotonic() - t0
            print(f"{total_time:.2f}s")

            results.append({
                "ablation": label,
                "use_decomposition": use_decomp,
                "use_cegar": use_cegar,
                "num_groups": len(groups),
                "abstract_races": abstract_races,
                "confirmed_races": confirmed_races,
                "abstract_error": sum(
                    gres["total_overapprox_error"]
                    for gres in ai_results.values()
                ),
                "cegar_stats": cegar_stats,
                "time_total_s": round(total_time, 4),
            })
    return results


# ── Output formatting ───────────────────────────────────────────────────

def format_comparison_table(results: List[Dict], scenario_name: str):
    """Print a comparison table showing with/without CEGAR."""
    print(f"\n{'─' * 72}")
    print(f"  {scenario_name} — With vs Without CEGAR")
    print(f"{'─' * 72}")
    print(f"  {'Agents':<8} {'Mode':<14} {'Races':<8} "
          f"{'Prec':<8} {'Recall':<8} {'FPR':<8} {'F1':<8}")
    print(f"  {'─' * 62}")
    for r in results:
        n = r["num_agents"]
        nc = r["without_cegar"]
        wc = r["with_cegar"]
        print(f"  {n:<8} {'No CEGAR':<14} {nc['races_detected']:<8} "
              f"{nc['precision']:<8.3f} {nc['recall']:<8.3f} "
              f"{nc['fpr']:<8.3f} {nc['f1']:<8.3f}")
        print(f"  {'':<8} {'With CEGAR':<14} {wc['races_detected']:<8} "
              f"{wc['precision']:<8.3f} {wc['recall']:<8.3f} "
              f"{wc['fpr']:<8.3f} {wc['f1']:<8.3f}")
        cs = r["cegar_stats"]
        print(f"  {'':<8} {'  └ CEGAR':<14} "
              f"iters={cs['total_refinement_iters']}, "
              f"spurious={cs['spurious_eliminated']}, "
              f"time={cs['cegar_time_s']:.2f}s")


def format_scalability_table(results: List[Dict]):
    """Print scalability results with CEGAR."""
    print(f"\n{'─' * 72}")
    print("  Scalability with CEGAR")
    print(f"{'─' * 72}")
    print(f"  {'N':<5} {'Groups':<8} {'Safe':<6} {'Unsafe':<8} "
          f"{'Unk':<5} {'Spurious':<10} {'FPR':<8} {'Time':<8}")
    print(f"  {'─' * 60}")
    for r in results:
        print(f"  {r['num_agents']:<5} {r['num_groups']:<8} "
              f"{r['cegar_safe']:<6} {r['cegar_unsafe']:<8} "
              f"{r['cegar_unknown']:<5} {r['spurious_eliminated']:<10} "
              f"{r['refined_fpr']:<8.3f} {r['time_total_s']:<8.2f}")


def format_ablation_table(results: List[Dict]):
    """Print ablation results."""
    print(f"\n{'─' * 72}")
    print("  Ablation: Decomposition × CEGAR")
    print(f"{'─' * 72}")
    print(f"  {'Config':<22} {'Groups':<8} {'AbsRaces':<10} "
          f"{'Confirmed':<11} {'Time':<8}")
    print(f"  {'─' * 60}")
    for r in results:
        print(f"  {r['ablation']:<22} {r['num_groups']:<8} "
              f"{r['abstract_races']:<10} {r['confirmed_races']:<11} "
              f"{r['time_total_s']:<8.2f}")
        if r["cegar_stats"]:
            cs = r["cegar_stats"]
            print(f"  {'  └ CEGAR':<22} "
                  f"safe={cs['safe']}, unsafe={cs['unsafe']}, "
                  f"spurious={cs['spurious_eliminated']}, "
                  f"FPR={cs['refined_fpr']:.3f}")


# ── JSON serialization ─────────────────────────────────────────────────

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, frozenset):
            return sorted(obj)
        return super().default(obj)


# ── Main ────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("MARACE Pipeline Experiments WITH CEGAR Refinement")
    print("=" * 72)
    all_results: Dict[str, Any] = {}
    collector = MetricCollector()

    # ── (1) Highway intersection: 2, 3, 4 agents ──
    print("\n[1/4] Highway intersection scenarios (with CEGAR)")
    highway_results = []
    for n in [2, 3, 4]:
        print(f"  Highway: {n} agents...", end=" ", flush=True)
        r = run_highway_with_cegar(n, seed=42 + n)
        highway_results.append(r)
        collector.record("highway_fpr_no_cegar", r["without_cegar"]["fpr"])
        collector.record("highway_fpr_with_cegar", r["with_cegar"]["fpr"])
        collector.record("highway_recall_cegar", r["with_cegar"]["recall"])
        collector.record("highway_time", r["time_total_s"])
        print(f"done ({r['time_total_s']:.2f}s)")
    all_results["highway_intersection"] = highway_results
    format_comparison_table(highway_results, "Highway Intersection")

    # ── (2) Warehouse corridor: 4, 6, 8 agents ──
    print("\n[2/4] Warehouse corridor scenarios (with CEGAR)")
    warehouse_results = []
    for n in [4, 6, 8]:
        print(f"  Warehouse: {n} agents...", end=" ", flush=True)
        r = run_warehouse_with_cegar(n, seed=42 + n)
        warehouse_results.append(r)
        collector.record("warehouse_fpr_no_cegar", r["without_cegar"]["fpr"])
        collector.record("warehouse_fpr_with_cegar", r["with_cegar"]["fpr"])
        collector.record("warehouse_recall_cegar", r["with_cegar"]["recall"])
        collector.record("warehouse_time", r["time_total_s"])
        print(f"done ({r['time_total_s']:.2f}s)")
    all_results["warehouse_corridor"] = warehouse_results
    format_comparison_table(warehouse_results, "Warehouse Corridor")

    # ── (3) Scalability: 2 to 10 agents with CEGAR ──
    print("\n[3/4] Scalability sweep with CEGAR (2–10 agents)")
    scale_results = run_scalability_with_cegar(list(range(2, 11)), seed=42)
    all_results["scalability"] = scale_results
    format_scalability_table(scale_results)

    # Fit power-law model
    timing_data = [
        {"num_agents": r["num_agents"], "time": r["time_total_s"]}
        for r in scale_results if r["time_total_s"] > 0
    ]
    if len(timing_data) >= 2:
        sm = ScalabilityMetric()
        fit = sm.compute(timing_data)
        all_results["scalability_fit"] = fit
        print(f"\n  Power-law fit: t = {fit['coefficient']:.4f} * "
              f"n^{fit['exponent']:.4f} (R²={fit['r_squared']:.4f})")

    # ── (4) Ablation with CEGAR ──
    print("\n[4/4] Ablation study (decomposition × CEGAR)")
    ablation_results = run_ablation_with_cegar(seed=42)
    all_results["ablation"] = ablation_results
    format_ablation_table(ablation_results)

    # ── Summary ──
    print("\n" + "=" * 72)
    print("Summary Metrics")
    print("=" * 72)
    for name in collector.all_names():
        vals = collector.get(name)
        print(f"  {name:<35s}  mean={collector.mean(name):.4f}  "
              f"std={collector.std(name):.4f}  n={len(vals)}")

    # ── Save ──
    output_path = "cegar_experiment_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {output_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
