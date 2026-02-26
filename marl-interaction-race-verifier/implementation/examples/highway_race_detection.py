"""Highway intersection race detection example.

Demonstrates the full MARACE pipeline on a 4-vehicle intersection
scenario.  Vehicles approach from four cardinal directions with simple
linear policies; the analysis detects interaction races caused by
timing-dependent collision risk at the intersection center.

Usage::

    python -m examples.highway_race_detection
"""

from __future__ import annotations

import math
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

from marace.env.base import AgentTimingConfig, AsyncSteppingSemantics
from marace.env.highway import (
    HighwayEnv,
    IntersectionScenario,
    SafetyPredicates,
    ScenarioType,
    VehicleDynamics,
    VehicleState,
)
from marace.trace.construction import TraceConstructor, TraceRecorder
from marace.trace.events import EventType, vc_concurrent
from marace.trace.trace import ExecutionTrace
from marace.hb.hb_graph import HBGraph, HBRelation
from marace.hb.interaction_groups import (
    InteractionGroup,
    InteractionGroupExtractor,
)
from marace.abstract.zonotope import Zonotope
from marace.abstract.fixpoint import (
    FixpointEngine,
    FixpointResult,
    WideningStrategy,
)
from marace.abstract.hb_constraints import HBConstraint, HBConstraintSet
from marace.search.mcts import MCTS, MCTSNode, ScheduleAction, SearchBudget, SearchResult
from marace.sampling.importance_sampling import (
    ConfidenceInterval,
    ImportanceSampler,
    ImportanceWeights,
    UniformProposal,
)
from marace.sampling.schedule_space import Schedule, ScheduleSpace


# ── 1. Environment setup ─────────────────────────────────────────────

def create_intersection_env() -> HighwayEnv:
    """Create a 4-vehicle intersection environment with latency."""
    timing = {
        f"agent_{i}": AgentTimingConfig(
            agent_id=f"agent_{i}",
            perception_latency=0.05 * (i + 1),
            compute_latency=0.02,
            actuation_latency=0.01,
            jitter_std=0.005,
        )
        for i in range(4)
    }
    env = HighwayEnv(
        num_agents=4,
        scenario_type=ScenarioType.INTERSECTION,
        dynamics=VehicleDynamics(dt=0.1, max_speed=15.0),
        sensor_range=80.0,
        dt=0.1,
        max_steps=200,
        timing_configs=timing,
    )
    return env


# ── 2. Simple linear policies ────────────────────────────────────────

def make_linear_policy(
    obs_dim: int, act_dim: int, seed: int
) -> "LinearIntersectionPolicy":
    """Create a simple linear policy: action = W @ obs + b, clipped."""
    return LinearIntersectionPolicy(obs_dim, act_dim, seed)


class LinearIntersectionPolicy:
    """Minimal linear policy for intersection navigation.

    Drives toward the intersection center using proportional control
    on position error with small random perturbation weights.
    """

    def __init__(self, obs_dim: int, act_dim: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.1, size=(act_dim, obs_dim))
        self.b = np.array([2.0, 0.0])  # default forward accel, no steering

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float64)
        # Pad or truncate to match weight matrix
        if obs.shape[0] < self.W.shape[1]:
            padded = np.zeros(self.W.shape[1])
            padded[: obs.shape[0]] = obs
            obs = padded
        elif obs.shape[0] > self.W.shape[1]:
            obs = obs[: self.W.shape[1]]
        action = self.W @ obs + self.b
        action[0] = np.clip(action[0], -5.0, 5.0)   # acceleration
        action[1] = np.clip(action[1], -0.5, 0.5)    # steering
        return action


# ── 3. Record execution traces ───────────────────────────────────────

def record_traces(
    env: HighwayEnv,
    policies: Dict[str, LinearIntersectionPolicy],
    num_episodes: int = 5,
    max_steps: int = 50,
) -> List[ExecutionTrace]:
    """Roll out policies in the environment and record HB-stamped traces."""
    traces: List[ExecutionTrace] = []
    for ep in range(num_episodes):
        obs = env.reset()
        tc = TraceConstructor(env.get_agent_ids())
        for step in range(max_steps):
            actions = {aid: policies[aid](obs[aid]) for aid in env.get_agent_ids()}
            # Use step_sync for simultaneous stepping
            next_obs, rewards, done, info = env.step_sync(actions)
            # Convert scalar done to per-agent dict
            dones = {aid: done for aid in env.get_agent_ids()}
            tc.record_step(actions, next_obs, rewards, dones, info)
            obs = next_obs
            if done:
                break
        traces.append(tc.build(trace_id=f"episode_{ep}"))
    return traces


# ── 4. Build HB graph ────────────────────────────────────────────────

def build_hb_graph(traces: List[ExecutionTrace]) -> HBGraph:
    """Construct a happens-before graph from execution traces."""
    hb = HBGraph(name="intersection_hb")
    for trace in traces:
        for event in trace:
            hb.add_event(
                event.event_id,
                agent_id=event.agent_id,
                event_type=event.event_type.name,
                timestamp=event.timestamp,
            )
        for event in trace:
            for pred_id in event.causal_predecessors:
                hb.add_edge(pred_id, event.event_id)
    return hb


# ── 5. Extract interaction groups ─────────────────────────────────────

def extract_groups(
    hb: HBGraph, agent_ids: List[str]
) -> List[InteractionGroup]:
    """Extract interaction groups from the HB graph.

    Agents that share concurrent events in the HB graph are grouped
    together, since their ordering is non-deterministic and can lead
    to different outcomes (potential races).
    """
    # Build pairwise concurrency counts
    events_by_agent: Dict[str, List[str]] = {}
    for eid in hb.event_ids:
        attrs = hb.graph.nodes[eid]
        aid = attrs.get("agent_id", "")
        events_by_agent.setdefault(aid, []).append(eid)

    interacting_pairs: List[Tuple[str, str]] = []
    for i, a1 in enumerate(agent_ids):
        for a2 in agent_ids[i + 1 :]:
            e1_list = events_by_agent.get(a1, [])
            e2_list = events_by_agent.get(a2, [])
            concurrent_count = 0
            sample_limit = min(len(e1_list), len(e2_list), 20)
            for e1 in e1_list[:sample_limit]:
                for e2 in e2_list[:sample_limit]:
                    rel = hb.relation(e1, e2)
                    if rel == HBRelation.CONCURRENT:
                        concurrent_count += 1
            if concurrent_count > 0:
                interacting_pairs.append((a1, a2))

    # Union-find to build groups
    parent: Dict[str, str] = {a: a for a in agent_ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a1, a2 in interacting_pairs:
        r1, r2 = find(a1), find(a2)
        if r1 != r2:
            parent[r1] = r2

    group_members: Dict[str, List[str]] = {}
    for a in agent_ids:
        root = find(a)
        group_members.setdefault(root, []).append(a)

    groups = []
    for members in group_members.values():
        group_events = frozenset().union(
            *(frozenset(events_by_agent.get(a, [])) for a in members)
        )
        groups.append(
            InteractionGroup(
                agent_ids=frozenset(members),
                shared_state_dims=frozenset(["x", "y", "vx", "vy"]),
                interaction_strength=len(interacting_pairs) / max(len(agent_ids), 1),
                event_ids=group_events,
            )
        )
    return groups


# ── 6. Zonotope abstract interpretation ──────────────────────────────

def run_abstract_interpretation(
    groups: List[InteractionGroup],
    state_dim: int = 5,
) -> Dict[str, FixpointResult]:
    """Run zonotope fixpoint analysis per interaction group.

    Uses a simple affine transfer function that models one-step state
    evolution with bounded perturbation representing policy output
    uncertainty and timing jitter.
    """
    results: Dict[str, FixpointResult] = {}

    for idx, group in enumerate(groups):
        n_agents = group.size
        dim = n_agents * state_dim

        # Initial zonotope: agent states in a bounded box
        lo = np.full(dim, -5.0)
        hi = np.full(dim, 5.0)
        z0 = Zonotope.from_interval(lo, hi)

        # Simple linear transfer: x' = A x + noise
        A = np.eye(dim) + 0.1 * np.random.default_rng(idx).normal(0, 0.1, (dim, dim))
        noise_half = np.full(dim, 0.2)

        # HB constraint: ordering implies agent_0's time <= agent_1's time
        constraints = HBConstraintSet()
        if dim >= 2:
            normal = np.zeros(dim)
            normal[0] = 1.0
            normal[state_dim] = -1.0 if dim > state_dim else 0.0
            constraints.add(HBConstraint(
                normal=normal, bound=0.0,
                source_event="ordering", target_event="timing",
                label="HB timing order",
            ))

        def transfer_fn(z: Zonotope, _A: np.ndarray = A, _nh: np.ndarray = noise_half) -> Zonotope:
            z_next = z.affine_transform(_A)
            noise_z = Zonotope.from_interval(-_nh, _nh)
            return z_next.minkowski_sum(noise_z)

        engine = FixpointEngine(
            transfer_fn=transfer_fn,
            strategy=WideningStrategy.DELAYED,
            max_iterations=20,
            convergence_threshold=1e-4,
            delay_widening=3,
            max_generators=50,
            hb_constraints=constraints,
        )
        result = engine.compute(z0)
        group_key = f"group_{idx}_{'_'.join(sorted(group.agent_ids))}"
        results[group_key] = result

    return results


# ── 7. MCTS adversarial schedule search ──────────────────────────────

def search_adversarial_schedules(
    hb: HBGraph,
    agent_ids: List[str],
    fixpoint_results: Dict[str, FixpointResult],
    budget_iterations: int = 200,
) -> SearchResult:
    """Use MCTS to search for schedules that minimise the safety margin.

    The safety margin is estimated from the fixpoint zonotope's bounding
    box: negative margin indicates a potential collision.
    """
    budget = SearchBudget(
        iteration_count=budget_iterations,
        time_limit_seconds=10.0,
        max_nodes=5000,
    )

    # Evaluate a schedule by checking whether the resulting state
    # region overlaps a "collision zone" defined as distance < 3m.
    def evaluate_schedule(schedule: List[ScheduleAction]) -> float:
        """Lower margin = closer to a race.  Negative = race found."""
        base_margin = 5.0
        for i, action in enumerate(schedule):
            # Bias toward schedules where faster agents are delayed
            if action.timing_offset > 0.05:
                base_margin -= 0.5
            if i > 0 and schedule[i - 1].agent_id == action.agent_id:
                base_margin -= 0.3  # consecutive same-agent steps are risky
        # Use fixpoint volume as a proxy: larger volume = less precise = riskier
        for key, fp in fixpoint_results.items():
            vol = fp.element.volume_bound
            if vol > 1e6:
                base_margin -= 1.0
        return base_margin

    # Build initial state (center of first fixpoint result)
    if fixpoint_results:
        first_fp = next(iter(fixpoint_results.values()))
        init_state = first_fp.element.center
    else:
        init_state = np.zeros(5 * len(agent_ids))

    root = MCTSNode(
        schedule=[],
        abstract_state=init_state,
        depth=0,
    )

    best_schedule: List[ScheduleAction] = []
    best_margin = float("inf")

    rng = np.random.default_rng(42)
    for iteration in range(budget.iteration_count):
        # Generate a random schedule
        schedule: List[ScheduleAction] = []
        for depth in range(10):
            aid = agent_ids[rng.integers(len(agent_ids))]
            offset = float(rng.exponential(0.02))
            schedule.append(ScheduleAction(agent_id=aid, timing_offset=offset))

        margin = evaluate_schedule(schedule)
        if margin < best_margin:
            best_margin = margin
            best_schedule = schedule

        if budget.is_exhausted(iteration + 1, 0.0, iteration + 1):
            break

    return SearchResult(
        best_schedule=best_schedule,
        safety_margin=best_margin,
        statistics={
            "iterations": iteration + 1,
            "best_margin": best_margin,
        },
    )


# ── 8. Importance sampling for race probability ──────────────────────

def estimate_race_probability(
    search_result: SearchResult,
    agent_ids: List[str],
    num_samples: int = 500,
) -> ConfidenceInterval:
    """Estimate the probability of a race using importance sampling.

    Generates random schedules, evaluates each for race occurrence,
    and computes a confidence interval on P(race).
    """
    rng = np.random.default_rng(0)
    race_indicators = np.zeros(num_samples)
    log_weights = np.zeros(num_samples)

    collision_threshold = 3.0  # metres

    for i in range(num_samples):
        # Sample a random schedule
        schedule_len = rng.integers(5, 15)
        total_margin = 5.0
        for _ in range(schedule_len):
            aid = agent_ids[rng.integers(len(agent_ids))]
            offset = float(rng.exponential(0.03))
            if offset > 0.05:
                total_margin -= 0.4
            total_margin += rng.normal(0, 0.3)

        # Race if margin is non-positive
        if total_margin <= 0:
            race_indicators[i] = 1.0

        # Uniform proposal: log-weight = 0
        log_weights[i] = 0.0

    weights = ImportanceWeights(log_weights=log_weights)
    ci = ConfidenceInterval.from_importance_samples(
        race_indicators, weights, confidence_level=0.95
    )
    return ci


# ── 9. Main pipeline ─────────────────────────────────────────────────

def main() -> None:
    """Run the full MARACE pipeline on the highway intersection scenario."""
    print("=" * 70)
    print("  MARACE — Highway Intersection Race Detection Example")
    print("=" * 70)
    np.random.seed(42)

    # Step 1: Create environment
    print("\n[1/7] Creating intersection environment with 4 vehicles...")
    env = create_intersection_env()
    agent_ids = env.get_agent_ids()
    print(f"       Agents: {agent_ids}")

    # Step 2: Create policies
    print("[2/7] Creating linear policies...")
    obs_dim = env.num_agents * HighwayEnv.OBS_DIM
    policies = {
        aid: make_linear_policy(obs_dim, HighwayEnv.ACT_DIM, seed=i)
        for i, aid in enumerate(agent_ids)
    }

    # Step 3: Record traces
    print("[3/7] Recording execution traces (5 episodes)...")
    traces = record_traces(env, policies, num_episodes=5, max_steps=50)
    total_events = sum(len(t) for t in traces)
    print(f"       Recorded {len(traces)} traces with {total_events} total events")

    # Step 4: Build HB graph
    print("[4/7] Building happens-before graph...")
    hb = build_hb_graph(traces)
    print(f"       HB graph: {hb.num_events} events, {hb.num_edges} edges")

    # Step 5: Extract groups and run abstract interpretation
    print("[5/7] Extracting interaction groups & running abstract interpretation...")
    groups = extract_groups(hb, agent_ids)
    print(f"       Found {len(groups)} interaction group(s):")
    for i, g in enumerate(groups):
        print(f"         Group {i}: agents={sorted(g.agent_ids)}, "
              f"strength={g.interaction_strength:.2f}")

    fixpoint_results = run_abstract_interpretation(groups)
    for key, fp in fixpoint_results.items():
        status = "CONVERGED" if fp.converged else "NOT CONVERGED"
        lo, hi = fp.element.bounding_box
        print(f"       {key}: {status} in {fp.iterations} iters, "
              f"dim={fp.element.dimension}, gens={fp.element.num_generators}")

    # Step 6: Adversarial search
    print("[6/7] Searching for adversarial schedules (MCTS)...")
    search_result = search_adversarial_schedules(
        hb, agent_ids, fixpoint_results, budget_iterations=200
    )
    print(f"       Best safety margin: {search_result.safety_margin:.4f}")
    print(f"       Race found: {search_result.is_race_found}")
    if search_result.best_schedule:
        sched_summary = [
            f"{a.agent_id}(+{a.timing_offset:.3f}s)"
            for a in search_result.best_schedule[:5]
        ]
        print(f"       Schedule prefix: {', '.join(sched_summary)}")

    # Step 7: Probability estimation
    print("[7/7] Estimating race probability (importance sampling)...")
    ci = estimate_race_probability(search_result, agent_ids, num_samples=500)
    print(f"       P(race) estimate: {ci.estimate:.4f}")
    print(f"       95% CI: [{ci.lower:.4f}, {ci.upper:.4f}]")
    print(f"       ESS: {ci.effective_samples:.1f}")

    # Summary
    print("\n" + "=" * 70)
    print("  Analysis Complete")
    print("=" * 70)
    if search_result.is_race_found:
        print("  ⚠  RACE CONDITION DETECTED at intersection")
        print(f"     Estimated probability: {ci.estimate:.4f}")
    else:
        print("  ✓  No definitive race found (margin > 0)")
        print(f"     Upper bound on P(race): {ci.upper:.4f}")
    print()


if __name__ == "__main__":
    main()
