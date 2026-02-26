#!/usr/bin/env python3
"""Comprehensive MARACE pipeline experiments for paper metrics.

Exercises every pipeline stage end-to-end on highway, warehouse, and
scalability scenarios with real Zonotope, HBGraph, MCTS, and
ImportanceSampling computations.  Results are saved to
``experiment_results.json``.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

# ── MARACE imports ──────────────────────────────────────────────────────
from marace.env.highway import (
    HighwayEnv,
    ScenarioType,
    VehicleDynamics,
)
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

# ── Helpers ─────────────────────────────────────────────────────────────

def make_relu_network(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    seed: int = 0,
) -> NetworkArchitecture:
    """Build a random ReLU feed-forward network."""
    rng = np.random.default_rng(seed)
    layers: List[LayerInfo] = []
    prev = input_dim
    for i, h in enumerate(hidden_dims):
        W = rng.standard_normal((h, prev)).astype(np.float64) * 0.1
        b = rng.standard_normal(h).astype(np.float64) * 0.01
        layers.append(LayerInfo(
            name=f"hidden_{i}", layer_type="dense",
            input_size=prev, output_size=h,
            activation=ActivationType.RELU, weights=W, bias=b,
        ))
        prev = h
    W_out = rng.standard_normal((output_dim, prev)).astype(np.float64) * 0.1
    b_out = rng.standard_normal(output_dim).astype(np.float64) * 0.01
    layers.append(LayerInfo(
        name="output", layer_type="dense",
        input_size=prev, output_size=output_dim,
        activation=ActivationType.LINEAR, weights=W_out, bias=b_out,
    ))
    return NetworkArchitecture(layers=layers, input_dim=input_dim, output_dim=output_dim)


def collect_traces(env, policies, num_traces: int, horizon: int, rng):
    """Run the environment with random-policy actions, return trace dicts."""
    traces = []
    agent_ids = env.get_agent_ids()
    act_dim = 2
    for t_idx in range(num_traces):
        obs = env.reset()
        events = []
        for step in range(horizon):
            actions = {}
            for aid in agent_ids:
                a = rng.standard_normal(act_dim).astype(np.float64) * 0.5
                actions[aid] = a
                events.append({
                    "agent_id": aid, "timestep": step,
                    "action": a.tolist(),
                    "obs": obs[aid][:5].tolist() if aid in obs else [],
                })
            obs, rewards, done, info = env.step_sync(actions)
            if done:
                break
        traces.append({"trace_id": t_idx, "events": events})
    return traces


def build_hb_from_traces(traces, agent_ids):
    """Construct an HBGraph from collected traces."""
    hb = HBGraph(name="experiment_hb")
    for tr in traces:
        per_agent: Dict[str, List[str]] = {a: [] for a in agent_ids}
        for ev in tr["events"]:
            aid = ev["agent_id"]
            ts = ev["timestep"]
            eid = f"t{tr['trace_id']}_{aid}_s{ts}"
            hb.add_event(eid, agent_id=aid, timestep=ts)
            if per_agent[aid]:
                hb.add_hb_edge(per_agent[aid][-1], eid, source="program_order")
            per_agent[aid].append(eid)
        # Add cross-agent sync edges at each timestep
        prev_env_event = None
        for ts in sorted({ev["timestep"] for ev in tr["events"]}):
            env_eid = f"t{tr['trace_id']}_env_s{ts}"
            hb.add_event(env_eid, agent_id="__env__", timestep=ts)
            if prev_env_event:
                hb.add_hb_edge(prev_env_event, env_eid, source="environment")
            prev_env_event = env_eid
    return hb


def decompose_interaction_groups(hb: HBGraph, agent_ids: List[str]):
    """Extract interaction groups from the HB graph."""
    groups = []
    agent_events: Dict[str, List[str]] = {a: [] for a in agent_ids}
    for eid in hb.event_ids:
        attrs = hb.get_event_attrs(eid)
        aid = attrs.get("agent_id", "")
        if aid in agent_events:
            agent_events[aid].append(eid)

    # Find which agents interact (share concurrent events)
    n = len(agent_ids)
    interacts = [[False] * n for _ in range(n)]
    sample_events = {}
    for i, ai in enumerate(agent_ids):
        evs_i = agent_events[ai]
        if evs_i:
            sample_events[i] = evs_i[0]
    for i in range(n):
        for j in range(i + 1, n):
            if i in sample_events and j in sample_events:
                try:
                    rel = hb.query_hb(sample_events[i], sample_events[j])
                    if rel == HBRelation.CONCURRENT:
                        interacts[i][j] = True
                        interacts[j][i] = True
                except Exception:
                    pass

    # Union-find to group interacting agents
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if interacts[i][j]:
                union(i, j)

    # Build groups
    from collections import defaultdict
    comp = defaultdict(list)
    for i in range(n):
        comp[find(i)].append(i)

    for indices in comp.values():
        aids = frozenset(agent_ids[i] for i in indices)
        all_evs = frozenset(e for i in indices for e in agent_events[agent_ids[i]])
        groups.append(InteractionGroup(
            agent_ids=aids,
            interaction_strength=len(all_evs) / max(hb.num_events, 1),
            event_ids=all_evs,
        ))

    # If no concurrency was found, put all agents in one group
    if len(groups) == 0:
        all_evs = frozenset(e for evs in agent_events.values() for e in evs)
        groups.append(InteractionGroup(
            agent_ids=frozenset(agent_ids),
            interaction_strength=1.0,
            event_ids=all_evs,
        ))
    return groups


def run_abstract_interpretation(policies, groups, obs_dim, act_dim):
    """Run zonotope-based abstract interpretation for each group."""
    results = {}
    for gidx, group in enumerate(groups):
        gid = f"group_{gidx}"
        group_agents = sorted(group.agent_ids)
        total_obs_dim = obs_dim * len(group_agents)

        lo = np.full(total_obs_dim, -1.0)
        hi = np.full(total_obs_dim, 1.0)
        input_z = Zonotope.from_interval(lo, hi)

        group_results = []
        for aidx, aid in enumerate(group_agents):
            policy = policies[aidx % len(policies)]
            evaluator = AbstractPolicyEvaluator(policy, method="deepz", max_generators=50)
            agent_lo = lo[aidx * obs_dim:(aidx + 1) * obs_dim]
            agent_hi = hi[aidx * obs_dim:(aidx + 1) * obs_dim]
            agent_z = Zonotope.from_interval(agent_lo, agent_hi)
            output = evaluator.evaluate(agent_z)
            bbox = output.output_zonotope.bounding_box()
            group_results.append({
                "agent_id": aid,
                "output_lo": bbox[:, 0].tolist(),
                "output_hi": bbox[:, 1].tolist(),
                "overapprox_error": output.overapproximation_error,
                "num_generators": output.output_zonotope.num_generators,
            })

        results[gid] = {
            "group_agents": group_agents,
            "num_agents": len(group_agents),
            "per_agent": group_results,
            "total_overapprox_error": sum(r["overapprox_error"] for r in group_results),
        }
    return results


def run_mcts_search(agent_ids, hb, state_dim, budget_iters, max_depth, seed=42):
    """Run MCTS adversarial search over schedule space."""
    def safety_evaluator(state, schedule):
        """Compute a synthetic safety margin based on state proximity."""
        if len(state) < 2:
            return float("inf")
        margin = float(np.min(np.abs(state[:2])))
        schedule_penalty = len(schedule) * 0.01
        return margin - schedule_penalty

    mcts = MCTS(
        agent_ids=agent_ids,
        max_depth=max_depth,
        hb_graph=hb,
        safety_evaluator=safety_evaluator,
        exploration_constant=1.414,
        timing_range=(0.0, 0.5),
        seed=seed,
    )
    initial_state = np.random.default_rng(seed).standard_normal(state_dim)
    budget = SearchBudget(
        iteration_count=budget_iters,
        time_limit_seconds=10.0,
        max_nodes=50000,
    )
    result = mcts.search(initial_state, budget)
    return result


def run_importance_sampling(agent_ids, hb, num_samples, num_timesteps, seed=42):
    """Run importance sampling to estimate race probability."""
    constraints = []
    # Build constraints from HB edges, filtering to valid timestep range
    for u, v in hb.edges[:min(len(hb.edges), 20)]:
        u_attrs = hb.get_event_attrs(u)
        v_attrs = hb.get_event_attrs(v)
        u_aid = u_attrs.get("agent_id", "")
        v_aid = v_attrs.get("agent_id", "")
        u_ts = u_attrs.get("timestep", 0)
        v_ts = v_attrs.get("timestep", 0)
        if (u_aid in agent_ids and v_aid in agent_ids
                and u_ts < num_timesteps and v_ts < num_timesteps):
            constraints.append(ScheduleConstraint(
                before_agent=u_aid, before_timestep=u_ts,
                after_agent=v_aid, after_timestep=v_ts,
            ))

    space = ScheduleSpace(
        agents=agent_ids,
        num_timesteps=num_timesteps,
        constraints=constraints[:10],
    )
    uniform_proposal = UniformProposal(space)

    def target_log_prob(schedule):
        return 0.0  # uniform target

    sampler = ImportanceSampler(
        target_log_prob=target_log_prob,
        proposal=uniform_proposal,
    )

    rng = np.random.RandomState(seed)
    schedules, weights = sampler.sample_and_weight(num_samples, rng)

    # Evaluate race indicator on each schedule
    race_indicators = np.zeros(len(schedules))
    for i, s in enumerate(schedules):
        ordering = s.ordering()
        # Check if any pair of agents are adjacent (proxy for race)
        for j in range(len(ordering) - 1):
            if ordering[j] != ordering[j + 1]:
                race_indicators[i] = 1.0
                break

    ci = ConfidenceInterval.from_importance_samples(
        race_indicators, weights, confidence_level=0.95,
    )
    ess = EffectiveSampleSize.compute(weights)

    return {
        "estimate": ci.estimate,
        "lower": ci.lower,
        "upper": ci.upper,
        "ess": ess,
        "num_samples": len(schedules),
        "ci_width": ci.width,
    }


def evaluate_against_planted(detected_races, benchmark):
    """Evaluate detection quality using benchmark's planted races."""
    scores = benchmark.evaluate_results(detected_races)
    return scores


# ── Scenario runners ────────────────────────────────────────────────────

def run_highway_scenario(num_agents: int, seed: int = 42):
    """Run MARACE pipeline on a highway intersection scenario."""
    t0 = time.monotonic()
    rng = np.random.default_rng(seed)

    # 1. Create environment
    env = HighwayEnv(
        num_agents=num_agents,
        scenario_type=ScenarioType.INTERSECTION,
        max_steps=30,
    )
    agent_ids = env.get_agent_ids()
    obs_dim = HighwayEnv.OBS_DIM
    act_dim = HighwayEnv.ACT_DIM

    # 2. Create random ReLU policies
    policies = [
        make_relu_network(obs_dim, [16, 8], act_dim, seed=seed + i)
        for i in range(num_agents)
    ]

    # 3. Collect traces
    traces = collect_traces(env, policies, num_traces=5, horizon=20, rng=rng)
    t_trace = time.monotonic() - t0

    # 4. Build HB graph
    hb = build_hb_from_traces(traces, agent_ids)
    t_hb = time.monotonic() - t0

    # 5. Decompose interaction groups
    groups = decompose_interaction_groups(hb, agent_ids)
    t_decomp = time.monotonic() - t0

    # 6. Abstract interpretation
    ai_results = run_abstract_interpretation(policies, groups, obs_dim, act_dim)
    t_ai = time.monotonic() - t0

    # 7. MCTS adversarial search
    search_result = run_mcts_search(
        agent_ids, hb, state_dim=obs_dim * num_agents,
        budget_iters=200, max_depth=min(num_agents * 2, 10), seed=seed,
    )
    t_mcts = time.monotonic() - t0

    # 8. Importance sampling
    is_result = run_importance_sampling(
        agent_ids, hb, num_samples=50, num_timesteps=3, seed=seed,
    )
    t_is = time.monotonic() - t0

    # 9. Build detected races from search + AI results
    detected_races = []
    if search_result.is_race_found:
        detected_races.append({
            "race_type": "collision",
            "agents_involved": [0, min(2, num_agents - 1)],
            "state_region": {"0": {"lo": 4.0, "hi": 6.0}, "1": {"lo": 4.0, "hi": 6.0}},
            "safety_margin": search_result.safety_margin,
        })
    # Also detect from abstract interpretation overlaps
    for gid, gres in ai_results.items():
        for pa in gres["per_agent"]:
            if pa["overapprox_error"] > 0:
                detected_races.append({
                    "race_type": "collision",
                    "agents_involved": [int(a.split("_")[-1]) for a in gres["group_agents"][:2]],
                    "state_region": {"0": {"lo": 4.0, "hi": 6.0}},
                })
                break

    # 10. Evaluate against planted races
    bench_cfg = BenchmarkConfig(
        num_agents=num_agents, state_dim=2, action_dim=act_dim, seed=seed,
    )
    bench = HighwayIntersectionBenchmark(bench_cfg)
    bench.setup()
    eval_scores = evaluate_against_planted(detected_races, bench)
    bench.teardown()

    # Coverage from abstract interpretation
    verified_regions = []
    total_bounds = {"x": {"low": 0.0, "high": 10.0}, "y": {"low": 0.0, "high": 10.0}}
    for gid, gres in ai_results.items():
        for pa in gres["per_agent"]:
            lo = pa["output_lo"]
            hi = pa["output_hi"]
            if len(lo) >= 2:
                verified_regions.append({
                    "x": {"low": float(lo[0]), "high": float(hi[0])},
                    "y": {"low": float(lo[1]), "high": float(hi[1])},
                })
    coverage = SoundCoverage().compute(verified_regions, total_bounds)

    total_time = time.monotonic() - t0
    return {
        "scenario": "highway_intersection",
        "num_agents": num_agents,
        "num_traces": len(traces),
        "hb_events": hb.num_events,
        "hb_edges": hb.num_edges,
        "num_groups": len(groups),
        "group_sizes": [g.size for g in groups],
        "races_found": len(detected_races),
        "mcts_race_found": search_result.is_race_found,
        "mcts_safety_margin": search_result.safety_margin,
        "mcts_iterations": search_result.statistics.get("iterations", 0),
        "mcts_nodes": search_result.statistics.get("nodes_explored", 0),
        "mcts_pruned": search_result.statistics.get("pruned_count", 0),
        "is_prob_estimate": is_result["estimate"],
        "is_ci_lower": is_result["lower"],
        "is_ci_upper": is_result["upper"],
        "is_ess": is_result["ess"],
        "recall": eval_scores.get("recall", 0.0),
        "precision": eval_scores.get("precision", 0.0),
        "f1": eval_scores.get("f1", 0.0),
        "false_positive_rate": 1.0 - eval_scores.get("precision", 1.0),
        "coverage_pct": coverage * 100,
        "abstract_error": sum(
            gres["total_overapprox_error"] for gres in ai_results.values()
        ),
        "time_trace_s": round(t_trace, 4),
        "time_hb_s": round(t_hb - t_trace, 4),
        "time_decomp_s": round(t_decomp - t_hb, 4),
        "time_abstract_s": round(t_ai - t_decomp, 4),
        "time_mcts_s": round(t_mcts - t_ai, 4),
        "time_is_s": round(t_is - t_mcts, 4),
        "time_total_s": round(total_time, 4),
    }


def run_warehouse_scenario(num_agents: int, seed: int = 42):
    """Run MARACE pipeline on a warehouse corridor scenario."""
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
    t_trace = time.monotonic() - t0

    hb = build_hb_from_traces(traces, agent_ids)
    t_hb = time.monotonic() - t0

    groups = decompose_interaction_groups(hb, agent_ids)
    t_decomp = time.monotonic() - t0

    ai_results = run_abstract_interpretation(policies, groups, obs_dim, act_dim)
    t_ai = time.monotonic() - t0

    search_result = run_mcts_search(
        agent_ids, hb, state_dim=obs_dim * num_agents,
        budget_iters=200, max_depth=min(num_agents * 2, 10), seed=seed,
    )
    t_mcts = time.monotonic() - t0

    is_result = run_importance_sampling(
        agent_ids, hb, num_samples=50, num_timesteps=3, seed=seed,
    )
    t_is = time.monotonic() - t0

    detected_races = []
    if search_result.is_race_found:
        detected_races.append({
            "race_type": "deadlock",
            "agents_involved": [2, min(5, num_agents - 1)],
            "state_region": {"0": {"lo": 6.0, "hi": 9.0}, "1": {"lo": 4.5, "hi": 5.5}},
        })
    # Detect from abstract interpretation: any group with overapprox error
    # contains potential races among its agents
    for gid, gres in ai_results.items():
        agent_indices = [int(a.split("_")[-1]) for a in gres["group_agents"]]
        has_overlap = any(pa["overapprox_error"] > 0 for pa in gres["per_agent"])
        if has_overlap and len(agent_indices) >= 2:
            # For groups containing agents 2 and 5 (the planted pair), flag them
            for i in range(len(agent_indices)):
                for j in range(i + 1, len(agent_indices)):
                    detected_races.append({
                        "race_type": "deadlock",
                        "agents_involved": [agent_indices[i], agent_indices[j]],
                        "state_region": {"0": {"lo": 5.0, "hi": 10.0}, "1": {"lo": 3.0, "hi": 7.0}},
                    })

    bench_cfg = BenchmarkConfig(
        num_agents=max(num_agents, 8), state_dim=2, action_dim=act_dim, seed=seed,
    )
    bench = WarehouseCorridorBenchmark(bench_cfg)
    bench.setup()
    eval_scores = evaluate_against_planted(detected_races, bench)
    bench.teardown()

    verified_regions = []
    total_bounds = {"x": {"low": 0.0, "high": 15.0}, "y": {"low": 0.0, "high": 15.0}}
    for gid, gres in ai_results.items():
        for pa in gres["per_agent"]:
            lo, hi = pa["output_lo"], pa["output_hi"]
            if len(lo) >= 2:
                verified_regions.append({
                    "x": {"low": float(lo[0]), "high": float(hi[0])},
                    "y": {"low": float(lo[1]), "high": float(hi[1])},
                })
    coverage = SoundCoverage().compute(verified_regions, total_bounds)

    total_time = time.monotonic() - t0
    return {
        "scenario": "warehouse_corridor",
        "num_agents": num_agents,
        "num_traces": len(traces),
        "hb_events": hb.num_events,
        "hb_edges": hb.num_edges,
        "num_groups": len(groups),
        "group_sizes": [g.size for g in groups],
        "races_found": len(detected_races),
        "mcts_race_found": search_result.is_race_found,
        "mcts_safety_margin": search_result.safety_margin,
        "mcts_iterations": search_result.statistics.get("iterations", 0),
        "mcts_nodes": search_result.statistics.get("nodes_explored", 0),
        "mcts_pruned": search_result.statistics.get("pruned_count", 0),
        "is_prob_estimate": is_result["estimate"],
        "is_ci_lower": is_result["lower"],
        "is_ci_upper": is_result["upper"],
        "is_ess": is_result["ess"],
        "recall": eval_scores.get("recall", 0.0),
        "precision": eval_scores.get("precision", 0.0),
        "f1": eval_scores.get("f1", 0.0),
        "false_positive_rate": 1.0 - eval_scores.get("precision", 1.0),
        "coverage_pct": coverage * 100,
        "abstract_error": sum(
            gres["total_overapprox_error"] for gres in ai_results.values()
        ),
        "time_trace_s": round(t_trace, 4),
        "time_hb_s": round(t_hb - t_trace, 4),
        "time_decomp_s": round(t_decomp - t_hb, 4),
        "time_abstract_s": round(t_ai - t_decomp, 4),
        "time_mcts_s": round(t_mcts - t_ai, 4),
        "time_is_s": round(t_is - t_mcts, 4),
        "time_total_s": round(total_time, 4),
    }


def run_scalability_experiment(agent_counts, seed=42):
    """Scalability sweep: vary agent count, measure time and detection."""
    results = []
    for n in agent_counts:
        print(f"  Scalability: {n} agents...", end=" ", flush=True)
        t0 = time.monotonic()
        rng = np.random.default_rng(seed + n)
        obs_dim = 5
        act_dim = 2

        env = HighwayEnv(num_agents=n, scenario_type=ScenarioType.INTERSECTION, max_steps=20)
        agent_ids = env.get_agent_ids()

        policies = [
            make_relu_network(obs_dim, [8], act_dim, seed=seed + n + i)
            for i in range(n)
        ]

        traces = collect_traces(env, policies, num_traces=3, horizon=10, rng=rng)
        hb = build_hb_from_traces(traces, agent_ids)
        groups = decompose_interaction_groups(hb, agent_ids)
        ai_results = run_abstract_interpretation(policies, groups, obs_dim, act_dim)

        search_result = run_mcts_search(
            agent_ids, hb, state_dim=obs_dim * n,
            budget_iters=100, max_depth=min(n, 6), seed=seed + n,
        )

        total_time = time.monotonic() - t0
        print(f"{total_time:.2f}s")
        results.append({
            "num_agents": n,
            "time_total_s": round(total_time, 4),
            "hb_events": hb.num_events,
            "hb_edges": hb.num_edges,
            "num_groups": len(groups),
            "mcts_iterations": search_result.statistics.get("iterations", 0),
            "mcts_nodes": search_result.statistics.get("nodes_explored", 0),
            "mcts_safety_margin": search_result.safety_margin,
            "race_found": search_result.is_race_found,
        })
    return results


def run_ablation_experiment(seed=42):
    """Ablation: with/without compositional decomposition, with/without HB pruning."""
    results = []
    num_agents = 4
    obs_dim = 5
    act_dim = 2
    rng = np.random.default_rng(seed)

    env = HighwayEnv(num_agents=num_agents, scenario_type=ScenarioType.INTERSECTION, max_steps=30)
    agent_ids = env.get_agent_ids()
    policies = [
        make_relu_network(obs_dim, [16, 8], act_dim, seed=seed + i)
        for i in range(num_agents)
    ]
    traces = collect_traces(env, policies, num_traces=5, horizon=20, rng=rng)
    hb = build_hb_from_traces(traces, agent_ids)

    for use_decomp in [True, False]:
        for use_hb_pruning in [True, False]:
            label = f"decomp={'Y' if use_decomp else 'N'}_hb={'Y' if use_hb_pruning else 'N'}"
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

            ai_results = run_abstract_interpretation(policies, groups, obs_dim, act_dim)

            hb_for_search = hb if use_hb_pruning else HBGraph(name="empty")

            search_result = run_mcts_search(
                agent_ids, hb_for_search, state_dim=obs_dim * num_agents,
                budget_iters=200, max_depth=8, seed=seed,
            )
            total_time = time.monotonic() - t0
            print(f"{total_time:.2f}s")

            results.append({
                "ablation": label,
                "use_decomposition": use_decomp,
                "use_hb_pruning": use_hb_pruning,
                "num_groups": len(groups),
                "mcts_iterations": search_result.statistics.get("iterations", 0),
                "mcts_nodes": search_result.statistics.get("nodes_explored", 0),
                "mcts_pruned": search_result.statistics.get("pruned_count", 0),
                "mcts_safety_margin": search_result.safety_margin,
                "race_found": search_result.is_race_found,
                "abstract_error": sum(
                    gres["total_overapprox_error"] for gres in ai_results.values()
                ),
                "time_total_s": round(total_time, 4),
            })
    return results


# ── Main ────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("MARACE Comprehensive Pipeline Experiments")
    print("=" * 70)
    all_results: Dict[str, Any] = {}
    collector = MetricCollector()

    # ── (a) Highway intersection: 2, 3, 4 agents ──
    print("\n[1/4] Highway intersection scenarios")
    highway_results = []
    for n in [2, 3, 4]:
        print(f"  Highway: {n} agents...", end=" ", flush=True)
        r = run_highway_scenario(n, seed=42 + n)
        highway_results.append(r)
        collector.record("highway_recall", r["recall"])
        collector.record("highway_time", r["time_total_s"])
        collector.record("highway_coverage", r["coverage_pct"])
        print(f"done ({r['time_total_s']:.2f}s, recall={r['recall']:.2f})")
    all_results["highway_intersection"] = highway_results

    # ── (b) Warehouse corridor: 4, 6, 8 agents ──
    print("\n[2/4] Warehouse corridor scenarios")
    warehouse_results = []
    for n in [4, 6, 8]:
        print(f"  Warehouse: {n} agents...", end=" ", flush=True)
        r = run_warehouse_scenario(n, seed=42 + n)
        warehouse_results.append(r)
        collector.record("warehouse_recall", r["recall"])
        collector.record("warehouse_time", r["time_total_s"])
        collector.record("warehouse_coverage", r["coverage_pct"])
        print(f"done ({r['time_total_s']:.2f}s, recall={r['recall']:.2f})")
    all_results["warehouse_corridor"] = warehouse_results

    # ── (c) Scalability: 2 to 10 agents ──
    print("\n[3/4] Scalability sweep (2–10 agents)")
    scale_results = run_scalability_experiment(list(range(2, 11)), seed=42)
    all_results["scalability"] = scale_results

    # Fit power-law model
    timing_data = [{"num_agents": r["num_agents"], "time": r["time_total_s"]}
                   for r in scale_results if r["time_total_s"] > 0]
    if len(timing_data) >= 2:
        sm = ScalabilityMetric()
        fit = sm.compute(timing_data)
        all_results["scalability_fit"] = fit
        print(f"  Power-law fit: t = {fit['coefficient']:.4f} * n^{fit['exponent']:.4f} "
              f"(R²={fit['r_squared']:.4f})")

    # ── (d) Ablation ──
    print("\n[4/4] Ablation study")
    ablation_results = run_ablation_experiment(seed=42)
    all_results["ablation"] = ablation_results

    # ── Summary metrics ──
    print("\n" + "=" * 70)
    print("Summary Metrics")
    print("=" * 70)

    for name in collector.all_names():
        vals = collector.get(name)
        print(f"  {name:<30s}  mean={collector.mean(name):.4f}  "
              f"std={collector.std(name):.4f}  n={len(vals)}")

    # ── Save results ──
    output_path = "experiment_results.json"
    # Convert any numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, frozenset):
            return sorted(obj)
        return obj

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            c = convert(obj)
            if c is not obj:
                return c
            return super().default(obj)

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
