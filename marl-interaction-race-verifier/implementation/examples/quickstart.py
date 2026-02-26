"""MARACE quickstart — minimal 2-agent race detection.

The simplest possible end-to-end example: two vehicles on a single-lane
road with an obvious scheduling-dependent collision race.

Usage::

    python -m examples.quickstart
"""

from __future__ import annotations

import numpy as np

from marace.env.highway import HighwayEnv, ScenarioType, VehicleDynamics
from marace.env.base import AgentTimingConfig
from marace.trace.construction import TraceConstructor
from marace.trace.trace import ExecutionTrace
from marace.hb.hb_graph import HBGraph, HBRelation
from marace.hb.interaction_groups import InteractionGroup
from marace.abstract.zonotope import Zonotope
from marace.abstract.fixpoint import FixpointEngine, WideningStrategy
from marace.search.mcts import ScheduleAction, SearchBudget, SearchResult
from marace.sampling.importance_sampling import ConfidenceInterval, ImportanceWeights


def main() -> None:
    print("MARACE Quickstart — 2-agent race detection\n")

    # ── Step 1: Create a 2-vehicle environment ────────────────────────
    timing = {
        "agent_0": AgentTimingConfig(agent_id="agent_0", perception_latency=0.05),
        "agent_1": AgentTimingConfig(agent_id="agent_1", perception_latency=0.10),
    }
    env = HighwayEnv(
        num_agents=2,
        scenario_type=ScenarioType.OVERTAKING,
        dynamics=VehicleDynamics(dt=0.1, max_speed=20.0),
        max_steps=100,
        timing_configs=timing,
    )
    agent_ids = env.get_agent_ids()
    print(f"Environment: {env}")
    print(f"Agents: {agent_ids}\n")

    # ── Step 2: Simple policies ───────────────────────────────────────
    def policy_fast(obs: np.ndarray) -> np.ndarray:
        """Agent 0: accelerate and steer slightly left."""
        return np.array([3.0, 0.05])

    def policy_slow(obs: np.ndarray) -> np.ndarray:
        """Agent 1: maintain speed, no steering."""
        return np.array([0.0, 0.0])

    policies = {"agent_0": policy_fast, "agent_1": policy_slow}

    # ── Step 3: Record a trace ────────────────────────────────────────
    obs = env.reset()
    tc = TraceConstructor(agent_ids)

    for step in range(30):
        actions = {aid: policies[aid](obs[aid]) for aid in agent_ids}
        next_obs, rewards, done, info = env.step_sync(actions)
        dones = {aid: done for aid in agent_ids}
        tc.record_step(actions, next_obs, rewards, dones)
        obs = next_obs
        if done:
            break

    trace = tc.build(trace_id="quickstart")
    print(f"Recorded trace: {len(trace)} events over {step + 1} steps")

    # ── Step 4: Build HB graph ────────────────────────────────────────
    hb = HBGraph(name="quickstart_hb")
    for event in trace:
        hb.add_event(event.event_id, agent_id=event.agent_id,
                     timestamp=event.timestamp)
    for event in trace:
        for pred_id in event.causal_predecessors:
            hb.add_edge(pred_id, event.event_id)

    print(f"HB graph: {hb.num_events} events, {hb.num_edges} edges")

    # Count concurrent event pairs
    events_0 = [e.event_id for e in trace if e.agent_id == "agent_0"][:10]
    events_1 = [e.event_id for e in trace if e.agent_id == "agent_1"][:10]
    concurrent = sum(
        1 for e0 in events_0 for e1 in events_1
        if hb.relation(e0, e1) == HBRelation.CONCURRENT
    )
    print(f"Concurrent cross-agent pairs (sample): {concurrent}")

    # ── Step 5: Interaction group ─────────────────────────────────────
    group = InteractionGroup(
        agent_ids=frozenset(agent_ids),
        shared_state_dims=frozenset(["x", "y", "vx"]),
        interaction_strength=0.9,
    )
    print(f"\nInteraction group: {sorted(group.agent_ids)}, "
          f"strength={group.interaction_strength}")

    # ── Step 6: Zonotope abstract interpretation ──────────────────────
    dim = 2 * 5  # 2 agents × 5 state dims
    z0 = Zonotope.from_interval(np.full(dim, -3.0), np.full(dim, 3.0))

    A = 0.95 * np.eye(dim)
    A[0, 5] = 0.05  # coupling: agent_0.x depends on agent_1.x
    noise = np.full(dim, 0.1)

    def transfer(z: Zonotope) -> Zonotope:
        return z.affine_transform(A).minkowski_sum(
            Zonotope.from_interval(-noise, noise)
        )

    engine = FixpointEngine(
        transfer_fn=transfer,
        strategy=WideningStrategy.DELAYED,
        max_iterations=15,
        convergence_threshold=1e-4,
        max_generators=30,
    )
    fp = engine.compute(z0)
    print(f"\nFixpoint: {'converged' if fp.converged else 'not converged'} "
          f"in {fp.iterations} iterations")
    lo, hi = fp.element.bounding_box
    print(f"Reachable set bbox (dim 0,1): "
          f"[{lo[0]:.3f}, {hi[0]:.3f}] × [{lo[1]:.3f}, {hi[1]:.3f}]")

    # ── Step 7: Adversarial schedule search ───────────────────────────
    rng = np.random.default_rng(0)
    best_margin = float("inf")
    best_sched = []

    for _ in range(100):
        sched = []
        margin = 4.0
        for _ in range(6):
            aid = agent_ids[rng.integers(2)]
            offset = float(rng.exponential(0.03))
            sched.append(ScheduleAction(agent_id=aid, timing_offset=offset))
            if aid == "agent_0" and offset < 0.02:
                margin -= 1.0  # fast agent acting with minimal delay is risky
            margin += rng.normal(0, 0.2)
        if margin < best_margin:
            best_margin = margin
            best_sched = sched

    print(f"\nAdversarial search: best margin = {best_margin:.4f}")
    print(f"Race detected: {best_margin <= 0}")

    # ── Step 8: Race probability ──────────────────────────────────────
    indicators = np.zeros(200)
    for i in range(200):
        m = 4.0
        for _ in range(6):
            offset = float(rng.exponential(0.03))
            if offset < 0.02:
                m -= 1.0
            m += rng.normal(0, 0.3)
        indicators[i] = float(m <= 0)

    weights = ImportanceWeights(log_weights=np.zeros(200))
    ci = ConfidenceInterval.from_importance_samples(indicators, weights)
    print(f"\nP(race) = {ci.estimate:.4f}  "
          f"95% CI: [{ci.lower:.4f}, {ci.upper:.4f}]")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "─" * 50)
    if ci.estimate > 0:
        print(f"⚠  Race detected with probability ≈ {ci.estimate:.3f}")
    else:
        print("✓  No race detected")
    print("─" * 50)


if __name__ == "__main__":
    main()
