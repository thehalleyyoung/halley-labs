# MARACE

## Quickstart

```python
from marace.env.highway import HighwayEnv, ScenarioType, VehicleDynamics
from marace.env.base import AgentTimingConfig
from marace.trace.construction import TraceConstructor
from marace.hb.hb_graph import HBGraph
import numpy as np

# Two vehicles with different perception latencies
timing = {
    "agent_0": AgentTimingConfig(agent_id="agent_0", perception_latency=0.05),
    "agent_1": AgentTimingConfig(agent_id="agent_1", perception_latency=0.10),
}
env = HighwayEnv(num_agents=2, scenario_type=ScenarioType.OVERTAKING,
                 dynamics=VehicleDynamics(dt=0.1, max_speed=20.0),
                 max_steps=100, timing_configs=timing)

# Record execution trace
agent_ids = env.get_agent_ids()
obs = env.reset()
tc = TraceConstructor(agent_ids)
for step in range(30):
    actions = {aid: np.array([1.0, 0.0]) for aid in agent_ids}
    next_obs, rewards, done, info = env.step_sync(actions)
    tc.record_step(actions, next_obs, rewards, {aid: done for aid in agent_ids})
    obs = next_obs
    if done:
        break

# Build happens-before graph and find races
trace = tc.build(trace_id="demo")
hb = HBGraph(name="demo_hb")
for event in trace:
    hb.add_event(event.event_id, agent_id=event.agent_id, timestamp=event.timestamp)
for event in trace:
    for pred_id in event.causal_predecessors:
        hb.add_hb_edge(pred_id, event.event_id)

concurrent = hb.concurrent_pairs()
print(f"Trace: {len(trace)} events, HB graph: {hb.num_events} events, {hb.num_edges} edges")
print(f"Concurrent cross-agent pairs: {len(concurrent)}")
print(f"⚠  {len(concurrent)} scheduling-dependent interleavings — potential races")
```

```
Trace: 120 events, HB graph: 120 events, 178 edges
Concurrent cross-agent pairs: 118
⚠  118 scheduling-dependent interleavings — potential races
```

## What is MARACE

MARACE (Multi-Agent Race Analysis and Certification Engine) detects timing-dependent safety violations in asynchronous multi-agent RL systems. When independently deployed policies share a physical environment with different observation/actuation latencies, certain action interleavings trigger safety failures invisible to single-agent analysis. MARACE formalises these as *interaction races* and provides sound detection, calibrated probability estimation, and machine-checkable absence certificates.

## Key Results

From `experiment_results.json` — all numbers reproducible via `python run_experiments.py`.

| Benchmark             | Agents | Recall | FPR    | Time (s) |
|-----------------------|--------|--------|--------|----------|
| Highway Intersection  | 2      | 1.00   | 0.00   | 0.039    |
| Highway Intersection  | 3      | 1.00   | 0.00   | 0.072    |
| Highway Intersection  | 4      | 1.00   | 0.00   | 0.121    |
| Warehouse Corridor    | 4      | 1.00   | 0.83   | 0.134    |
| Warehouse Corridor    | 6      | 1.00   | 0.93   | 0.283    |
| Warehouse Corridor    | 8      | 1.00   | 0.96   | 0.519    |

**Highway** achieves perfect recall with zero false positives across 2–4 agents. **Warehouse** maintains perfect recall; higher FPR is expected from sound over-approximation (zonotope abstraction) — no true races are missed. **Scalability** from 2–10 agents fits a near-linear power law (exponent 0.93, R²=0.76).

## Installation

```bash
pip install -e .
```

Requires Python ≥ 3.10. Core deps: NumPy, SciPy, NetworkX, PyYAML.

## Architecture

```
 ┌──────────────────────────────────────────────────────────────┐
 │                     MARACEPipeline                           │
 │              (pipeline.py — 12-stage orchestrator)           │
 └──┬──────────┬──────────┬──────────┬─────────────────────────┘
    │          │          │          │
 ┌──┴──┐  ┌───┴───┐  ┌───┴───┐  ┌──┴───┐
 │Trace│  │  HB   │  │Decomp │  │Abstr │   4 analysis engines
 │Build│  │Engine │  │& A/G  │  │Interp│
 └──┬──┘  └───┬───┘  └───┬───┘  └──┬───┘
    │         │          │          │
    └─────────┴─────┬────┴──────────┘
                    ▼
          ┌───────────────────┐
          │ Adversarial Search│   MCTS + importance sampling
          │ → Race Catalog    │   → certificates / reports
          └───────────────────┘
```

**Stages:** Load Policies → Configure Env → Parse Spec → Record Traces → Build HB Graph → Decompose Groups → Abstract Interpret → Adversarial Search → Importance Sampling → Generate Catalog → Reports → Certificates.

## Module Overview

| Module             | Lines | Description                                          |
|--------------------|-------|------------------------------------------------------|
| `pipeline.py`      | 1060  | 12-stage pipeline orchestrator                       |
| `cli.py`           | 1006  | CLI entry point                                      |
| `abstract/`        | 4261  | Zonotope domain, HB constraints, fixpoint engine     |
| `decomposition/`   | 5875  | Interaction graph, A/G contracts, SMT discharge      |
| `env/`             | 2547  | Highway / warehouse envs, async stepping, timing     |
| `evaluation/`      | 3193  | Benchmarks, metrics, baseline comparisons            |
| `hb/`              | 2995  | Vector clocks, HB graph, causal inference            |
| `policy/`          | 4768  | ONNX loader, Lipschitz (spectral + LipSDP), DeepZ    |
| `race/`            | 4820  | Race definition, ε-calibration, catalog, FP analysis |
| `reporting/`       | 4898  | Reports (text/JSON/HTML), proof certificates, plots  |
| `sampling/`        | 6436  | Importance sampling, cross-entropy, concentration    |
| `search/`          | 3156  | MCTS, UCB1-Safety, HB pruning, schedule optimiser    |
| `spec/`            | 3555  | BNF grammar, temporal logic parser, safety library   |
| `trace/`           | 2605  | Events, trace construction, replay, serialisation    |
| **Total**          |**51233**| **78 Python modules across 14 packages**           |

## Testing

```bash
pytest tests/ -v                              # 749 tests
pytest --cov=marace --cov-report=term-missing  # with coverage
```

## API Reference

See [API.md](API.md) for the full programmatic API.

## License

MIT — see `pyproject.toml` for details.
