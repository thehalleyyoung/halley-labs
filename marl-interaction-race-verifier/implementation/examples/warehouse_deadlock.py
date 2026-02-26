"""Warehouse corridor deadlock detection example.

Demonstrates MARACE analysis on a warehouse environment where 6 robots
operate in narrow corridors.  The scenario plants a corridor conflict
by having robots approach each other in a shared corridor, revealing
a deadlock race condition under certain schedules.

Usage::

    python -m examples.warehouse_deadlock
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np

from marace.env.base import AgentTimingConfig, AsyncSteppingSemantics, SteppingOrder
from marace.env.warehouse import (
    CellType,
    CorridorConflict,
    DeadlockDetection,
    RobotDynamics,
    RobotState,
    TaskAssignment,
    WarehouseEnv,
    WarehouseLayout,
    WarehouseSafetyPredicates,
)
from marace.trace.construction import TraceConstructor
from marace.trace.events import EventType
from marace.trace.trace import ExecutionTrace
from marace.hb.hb_graph import HBGraph, HBRelation
from marace.hb.interaction_groups import InteractionGroup
from marace.abstract.zonotope import Zonotope
from marace.abstract.fixpoint import FixpointEngine, FixpointResult, WideningStrategy
from marace.abstract.hb_constraints import HBConstraint, HBConstraintSet
from marace.search.mcts import SearchBudget, SearchResult, ScheduleAction
from marace.sampling.importance_sampling import ConfidenceInterval, ImportanceWeights


# ── 1. Environment & layout ──────────────────────────────────────────

def create_corridor_layout() -> WarehouseLayout:
    """Build a warehouse layout with narrow corridors between shelves.

    The layout has two parallel shelf rows with a single narrow corridor
    between them.  This forces robots to negotiate passage.
    """
    layout = WarehouseLayout(width=16, height=12, cell_size=1.0, corridor_width=1.2)
    assert layout.grid is not None

    # Horizontal shelf rows leaving a narrow corridor at row 5
    for col in range(2, 14):
        for row in [3, 4, 6, 7]:
            if col % 5 != 0:  # leave gaps for cross-corridors
                layout.set_cell(row, col, CellType.SHELF)

    # Pickup stations on left wall
    layout.set_cell(1, 0, CellType.PICKUP_STATION)
    layout.set_cell(5, 0, CellType.PICKUP_STATION)
    layout.set_cell(9, 0, CellType.PICKUP_STATION)

    # Delivery stations on right wall
    layout.set_cell(1, 15, CellType.DELIVERY_STATION)
    layout.set_cell(5, 15, CellType.DELIVERY_STATION)
    layout.set_cell(9, 15, CellType.DELIVERY_STATION)

    return layout


def create_warehouse_env() -> WarehouseEnv:
    """Create a 6-robot warehouse with per-robot latency."""
    layout = create_corridor_layout()
    timing = {
        f"robot_{i}": AgentTimingConfig(
            agent_id=f"robot_{i}",
            perception_latency=0.03 + 0.01 * i,
            compute_latency=0.02,
            actuation_latency=0.01,
            jitter_std=0.005,
        )
        for i in range(6)
    }
    env = WarehouseEnv(
        num_robots=6,
        layout=layout,
        dynamics=RobotDynamics(dt=0.1, max_linear=1.0, max_angular=math.pi / 2),
        sensor_range=4.0,
        max_steps=500,
        dt=0.1,
        auto_task=True,
        timing_configs=timing,
    )
    return env


# ── 2. Corridor-seeking policies ─────────────────────────────────────

class CorridorPolicy:
    """Simple policy that drives robots toward a target through corridors.

    Uses proportional control with obstacle avoidance.
    """

    def __init__(self, target_x: float, target_y: float, seed: int = 0) -> None:
        self.target = np.array([target_x, target_y])
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float64)
        # Extract ego state: [x, y, heading, carrying, battery]
        ego_x, ego_y, heading = obs[0], obs[1], obs[2]
        ego_pos = np.array([ego_x, ego_y])

        # Direction to target
        to_target = self.target - ego_pos
        dist = np.linalg.norm(to_target)
        if dist < 0.1:
            return np.array([0.0, 0.0])

        desired_heading = math.atan2(to_target[1], to_target[0])
        heading_error = desired_heading - heading
        # Wrap to [-pi, pi]
        heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi

        angular_vel = np.clip(2.0 * heading_error, -math.pi / 2, math.pi / 2)
        linear_vel = np.clip(0.8 * dist, 0.0, 1.0)

        # Slow down when turning sharply
        if abs(heading_error) > 0.5:
            linear_vel *= 0.3

        return np.array([linear_vel, angular_vel])


def make_corridor_policies(env: WarehouseEnv) -> Dict[str, CorridorPolicy]:
    """Create policies that route robots through the narrow corridor.

    Robots 0-2 drive left-to-right; robots 3-5 drive right-to-left,
    all through the same corridor row (y ≈ 5.5).
    """
    policies: Dict[str, CorridorPolicy] = {}
    layout = env.layout
    corridor_y = 5.5  # center of the corridor between shelf rows

    for i, aid in enumerate(env.get_agent_ids()):
        if i < 3:
            # Drive toward right side
            target_x = layout.real_width - 1.0
            target_y = corridor_y
        else:
            # Drive toward left side
            target_x = 1.0
            target_y = corridor_y
        policies[aid] = CorridorPolicy(target_x, target_y, seed=i)

    return policies


# ── 3. Trace recording ───────────────────────────────────────────────

def record_warehouse_traces(
    env: WarehouseEnv,
    policies: Dict[str, CorridorPolicy],
    num_episodes: int = 3,
    max_steps: int = 60,
) -> List[ExecutionTrace]:
    """Record multi-agent execution traces in the warehouse."""
    traces: List[ExecutionTrace] = []

    for ep in range(num_episodes):
        obs = env.reset()
        tc = TraceConstructor(env.get_agent_ids())

        for step in range(max_steps):
            actions = {aid: policies[aid](obs[aid]) for aid in env.get_agent_ids()}
            next_obs, rewards, done, info = env.step_sync(actions)
            dones = {aid: done for aid in env.get_agent_ids()}
            tc.record_step(actions, next_obs, rewards, dones)
            obs = next_obs
            if done:
                break

        traces.append(tc.build(trace_id=f"warehouse_ep_{ep}"))

    return traces


# ── 4. HB graph construction ─────────────────────────────────────────

def build_hb_graph(traces: List[ExecutionTrace]) -> HBGraph:
    """Build happens-before graph from recorded traces."""
    hb = HBGraph(name="warehouse_hb")
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


# ── 5. Interaction group extraction ──────────────────────────────────

def extract_corridor_groups(
    hb: HBGraph, agent_ids: List[str]
) -> List[InteractionGroup]:
    """Group robots that share corridor interactions.

    In the corridor deadlock scenario, robots 0-2 (left→right) and
    robots 3-5 (right→left) form an interaction group because they
    share the narrow corridor and their orderings are concurrent in
    the HB graph.
    """
    events_by_agent: Dict[str, List[str]] = {}
    for eid in hb.event_ids:
        attrs = hb.graph.nodes[eid]
        aid = attrs.get("agent_id", "")
        events_by_agent.setdefault(aid, []).append(eid)

    # Check pairwise concurrency to find interacting agents
    parent: Dict[str, str] = {a: a for a in agent_ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a1 in enumerate(agent_ids):
        for a2 in agent_ids[i + 1 :]:
            e1s = events_by_agent.get(a1, [])[:10]
            e2s = events_by_agent.get(a2, [])[:10]
            has_concurrency = any(
                hb.relation(e1, e2) == HBRelation.CONCURRENT
                for e1 in e1s
                for e2 in e2s
            )
            if has_concurrency:
                r1, r2 = find(a1), find(a2)
                if r1 != r2:
                    parent[r1] = r2

    group_members: Dict[str, List[str]] = {}
    for a in agent_ids:
        root = find(a)
        group_members.setdefault(root, []).append(a)

    groups = []
    for members in group_members.values():
        all_events = frozenset().union(
            *(frozenset(events_by_agent.get(a, [])) for a in members)
        )
        groups.append(
            InteractionGroup(
                agent_ids=frozenset(members),
                shared_state_dims=frozenset(["x", "y", "heading"]),
                interaction_strength=0.8,
                event_ids=all_events,
            )
        )
    return groups


# ── 6. Abstract interpretation ────────────────────────────────────────

def run_abstract_interpretation(
    groups: List[InteractionGroup],
) -> Dict[str, FixpointResult]:
    """Run zonotope analysis on each interaction group.

    The transfer function models robot dynamics with bounded
    uncertainty from timing jitter and observation staleness.
    """
    results: Dict[str, FixpointResult] = {}
    state_dim = 3  # x, y, heading per robot

    for idx, group in enumerate(groups):
        dim = group.size * state_dim
        rng = np.random.default_rng(idx)

        # Initial state: corridor positions with uncertainty
        lo = np.full(dim, -2.0)
        hi = np.full(dim, 2.0)
        z0 = Zonotope.from_interval(lo, hi)

        # Linear dynamics with coupling between nearby robots
        A = np.eye(dim)
        for i in range(group.size):
            for j in range(group.size):
                if i != j:
                    # Small coupling in position dimensions
                    row, col = i * state_dim, j * state_dim
                    A[row, col] = 0.02
                    A[row + 1, col + 1] = 0.02

        noise_half = np.full(dim, 0.15)  # timing jitter uncertainty

        def transfer_fn(z: Zonotope, _A: np.ndarray = A, _nh: np.ndarray = noise_half) -> Zonotope:
            z_next = z.affine_transform(_A)
            noise = Zonotope.from_interval(-_nh, _nh)
            return z_next.minkowski_sum(noise)

        engine = FixpointEngine(
            transfer_fn=transfer_fn,
            strategy=WideningStrategy.DELAYED,
            max_iterations=25,
            convergence_threshold=1e-4,
            delay_widening=3,
            max_generators=40,
        )

        result = engine.compute(z0)
        key = f"group_{idx}"
        results[key] = result

    return results


# ── 7. Adversarial schedule search ────────────────────────────────────

def search_deadlock_schedules(
    agent_ids: List[str],
    fixpoint_results: Dict[str, FixpointResult],
    iterations: int = 300,
) -> SearchResult:
    """Search for schedules that cause corridor deadlock.

    A deadlock schedule interleaves opposing-direction robots so they
    enter the corridor simultaneously and block each other.
    """
    rng = np.random.default_rng(7)
    best_schedule: List[ScheduleAction] = []
    best_margin = float("inf")

    # Agents 0-2 go left→right, 3-5 go right→left
    left_to_right = [a for a in agent_ids if int(a.split("_")[1]) < 3]
    right_to_left = [a for a in agent_ids if int(a.split("_")[1]) >= 3]

    for _ in range(iterations):
        schedule: List[ScheduleAction] = []
        margin = 5.0

        # Interleave opposing robots
        for step in range(8):
            if step % 2 == 0:
                aid = left_to_right[rng.integers(len(left_to_right))]
            else:
                aid = right_to_left[rng.integers(len(right_to_left))]
            offset = float(rng.exponential(0.02))
            schedule.append(ScheduleAction(agent_id=aid, timing_offset=offset))

            # Opposing robots entering corridor simultaneously reduces margin
            if step > 0 and schedule[-1].agent_id in right_to_left:
                prev_is_left = schedule[-2].agent_id in left_to_right
                if prev_is_left and offset < 0.03:
                    margin -= 1.5  # Near-simultaneous opposing entry
            margin += rng.normal(0, 0.2)

        if margin < best_margin:
            best_margin = margin
            best_schedule = schedule

    return SearchResult(
        best_schedule=best_schedule,
        safety_margin=best_margin,
        statistics={"iterations": iterations, "best_margin": best_margin},
    )


# ── 8. Race probability estimation ───────────────────────────────────

def estimate_deadlock_probability(
    agent_ids: List[str], num_samples: int = 500
) -> ConfidenceInterval:
    """Estimate probability of corridor deadlock via importance sampling."""
    rng = np.random.default_rng(0)
    race_indicators = np.zeros(num_samples)
    log_weights = np.zeros(num_samples)

    left_ids = [a for a in agent_ids if int(a.split("_")[1]) < 3]
    right_ids = [a for a in agent_ids if int(a.split("_")[1]) >= 3]

    for i in range(num_samples):
        # Simulate random schedule and check for deadlock condition
        opposing_simultaneous = 0
        for _ in range(8):
            l_delay = rng.exponential(0.03)
            r_delay = rng.exponential(0.03)
            if abs(l_delay - r_delay) < 0.01:
                opposing_simultaneous += 1
        # Deadlock if enough simultaneous opposing entries
        if opposing_simultaneous >= 2:
            race_indicators[i] = 1.0

    weights = ImportanceWeights(log_weights=log_weights)
    return ConfidenceInterval.from_importance_samples(
        race_indicators, weights, confidence_level=0.95
    )


# ── 9. Deadlock detection check ──────────────────────────────────────

def check_deadlock_conditions(env: WarehouseEnv) -> None:
    """Run the built-in deadlock detector on the current state."""
    robots = env.robots
    detector = DeadlockDetection(proximity_threshold=1.5)
    deadlocks = detector.detect(robots)

    conflict_detector = CorridorConflict(env.layout, corridor_width=1.2)
    conflicts = conflict_detector.detect(robots)

    if deadlocks:
        print(f"       Deadlock cycles detected: {len(deadlocks)}")
        for dl in deadlocks:
            print(f"         Cycle: {sorted(dl)}")
    else:
        print("       No deadlocks in current state")

    if conflicts:
        print(f"       Corridor conflicts: {conflicts}")
    else:
        print("       No corridor conflicts in current state")


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    """Run the full warehouse deadlock detection pipeline."""
    print("=" * 70)
    print("  MARACE — Warehouse Corridor Deadlock Detection Example")
    print("=" * 70)
    np.random.seed(123)

    # Step 1: Environment
    print("\n[1/8] Creating warehouse environment (6 robots, narrow corridors)...")
    env = create_warehouse_env()
    agent_ids = env.get_agent_ids()
    print(f"       Robots: {agent_ids}")
    print(f"       Layout: {env.layout.width}x{env.layout.height} grid, "
          f"corridor width={env.layout.corridor_width}m")

    # Step 2: Policies
    print("[2/8] Creating corridor-seeking policies...")
    policies = make_corridor_policies(env)
    for aid in agent_ids[:2]:
        p = policies[aid]
        print(f"       {aid} → target=({p.target[0]:.1f}, {p.target[1]:.1f})")
    print(f"       ... ({len(agent_ids)} policies total)")

    # Step 3: Initial deadlock check
    print("[3/8] Running initial deadlock detection...")
    obs = env.reset()
    check_deadlock_conditions(env)

    # Step 4: Record traces
    print("[4/8] Recording execution traces (3 episodes, 60 steps)...")
    traces = record_warehouse_traces(env, policies, num_episodes=3, max_steps=60)
    total_events = sum(len(t) for t in traces)
    print(f"       {len(traces)} traces, {total_events} events total")

    # Step 5: Build HB graph
    print("[5/8] Building happens-before graph...")
    hb = build_hb_graph(traces)
    print(f"       {hb.num_events} events, {hb.num_edges} HB edges")

    # Step 6: Extract groups and abstract interpretation
    print("[6/8] Extracting interaction groups & abstract interpretation...")
    groups = extract_corridor_groups(hb, agent_ids)
    print(f"       {len(groups)} interaction group(s):")
    for i, g in enumerate(groups):
        print(f"         Group {i}: {sorted(g.agent_ids)} "
              f"(strength={g.interaction_strength:.2f})")

    fp_results = run_abstract_interpretation(groups)
    for key, fp in fp_results.items():
        status = "CONVERGED" if fp.converged else "NOT CONVERGED"
        print(f"       {key}: {status}, {fp.iterations} iters, "
              f"dim={fp.element.dimension}")

    # Step 7: Adversarial search
    print("[7/8] Searching for deadlock-inducing schedules...")
    search_result = search_deadlock_schedules(
        agent_ids, fp_results, iterations=300
    )
    print(f"       Safety margin: {search_result.safety_margin:.4f}")
    print(f"       Race found: {search_result.is_race_found}")
    if search_result.best_schedule:
        summary = [
            f"{a.agent_id}(+{a.timing_offset:.3f}s)"
            for a in search_result.best_schedule[:4]
        ]
        print(f"       Schedule: {', '.join(summary)} ...")

    # Step 8: Probability estimation
    print("[8/8] Estimating deadlock probability...")
    ci = estimate_deadlock_probability(agent_ids, num_samples=500)
    print(f"       P(deadlock) = {ci.estimate:.4f}")
    print(f"       95% CI: [{ci.lower:.4f}, {ci.upper:.4f}]")

    # Summary
    print("\n" + "=" * 70)
    print("  Analysis Complete")
    print("=" * 70)
    if search_result.is_race_found or ci.estimate > 0.01:
        print("  ⚠  CORRIDOR DEADLOCK RACE DETECTED")
        print(f"     Estimated probability: {ci.estimate:.4f}")
        print("     Cause: opposing robots enter narrow corridor simultaneously")
        print("     Remediation: add corridor reservation protocol or ")
        print("                  increase perception frequency")
    else:
        print("  ✓  No significant deadlock risk detected")
    print()


if __name__ == "__main__":
    main()
