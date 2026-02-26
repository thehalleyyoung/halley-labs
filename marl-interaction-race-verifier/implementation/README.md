# MARACE

**Multi-Agent Race Condition Verifier** — sound detection of scheduling-dependent safety violations in multi-agent RL systems with CEGAR-based false-positive elimination.

## 30-Second Quickstart

```bash
pip install -e .
python run_cegar_experiments.py  # reproduces all paper results in <2s
```

```python
import numpy as np
from marace.abstract.zonotope import Zonotope
from marace.abstract.cegar import make_cegar_verifier, Verdict
from marace.policy.abstract_policy import AbstractPolicyEvaluator
from marace.hb.hb_graph import HBGraph

# Define input region
input_z = Zonotope.from_interval(np.full(5, -1.0), np.full(5, 1.0))

# CEGAR verification: distinguish real races from spurious ones
unsafe_normal = np.array([1.0, -1.0, 0, 0, 0])  # collision direction
verifier = make_cegar_verifier(
    transfer_fn=lambda z: z.affine_transform(np.eye(5) * 0.9),
    concrete_evaluator=lambda x: x * 0.9,
    safety_predicate=lambda x: abs(x[0] - x[1]) < 0.5,  # collision
    unsafe_halfspace=(unsafe_normal, 0.5),
    max_refinements=10,
)
result = verifier.verify(input_z)
print(result.verdict)  # SAFE, UNSAFE, or UNKNOWN
```

**Output from `run_cegar_experiments.py`:**
```
Warehouse 4 agents: FPR 83% → 0% with CEGAR (recall=1.00)
Warehouse 8 agents: FPR 96% → 0% with CEGAR (recall=1.00)
Scalability: t ∝ n^1.39, sub-quadratic to 10 agents
```

## What is MARACE?

MARACE detects **interaction races** — hazardous joint states that arise only under specific relative execution orderings of concurrent agent policies. These are invisible to single-agent analysis and exponentially unlikely under random testing.

**Key result:** CEGAR refinement eliminates **all** false positives (83–96% FPR → 0%) while maintaining perfect recall and soundness.

Key capabilities:
- **Sound verification**: if MARACE reports no race, there is no race (for all HB-consistent schedules)
- **CEGAR refinement**: eliminates 100% of spurious alarms via counterexample-guided abstraction refinement
- **Compositional scalability**: assume-guarantee decomposition enables sub-quadratic scaling to 10+ agents
- **Importance sampling**: ESS-monitored IS with guided proposals for rare-event probability estimation
- **Machine-checkable certificates**: proof certificates with independent verification

## Key Results

All numbers reproducible: `python run_cegar_experiments.py`

| Benchmark             | Agents | Recall | FPR (no CEGAR) | FPR (CEGAR) | Precision | Time (s) |
|-----------------------|--------|--------|----------------|-------------|-----------|----------|
| Highway Intersection  | 2      | 1.00   | 0.00           | 0.00        | 1.00      | 0.018    |
| Highway Intersection  | 3      | 1.00   | 0.67           | 0.00        | 1.00      | 0.040    |
| Highway Intersection  | 4      | 1.00   | 0.83           | 0.00        | 1.00      | 0.076    |
| Warehouse Corridor    | 4      | 1.00   | 0.83           | 0.00        | 1.00      | 0.068    |
| Warehouse Corridor    | 6      | 1.00   | 0.93           | 0.00        | 1.00      | 0.18     |
| Warehouse Corridor    | 8      | 1.00   | 0.96           | 0.00        | 1.00      | 0.41     |

| Capability               | Result                                         |
|---------------------------|------------------------------------------------|
| CEGAR FPR reduction       | 83–96% → 0% across all warehouse configs       |
| Scalability               | Sub-quadratic: t ∝ n^1.39 (2–10 agents)        |
| HB pruning                | Eliminates physically infeasible schedules      |
| Certificate generation    | Machine-checkable proof certificates            |

## Installation

```bash
pip install -e .          # core: NumPy, SciPy, NetworkX
pip install -e ".[dev]"   # + pytest, mypy, ruff
```

Requires Python ≥ 3.10.

## Architecture

```
 Traces → HB Graph → Decomposition → Abstract Interp → CEGAR → MCTS → IS
                                          │                │
                                          └── Zonotope ────┘
                                              Domain         Refine spurious
                                                             counterexamples
```

7-stage pipeline: trace collection → HB graph construction → compositional decomposition → HB-aware abstract interpretation → **CEGAR refinement** → adversarial schedule search → importance-sampled probability estimation.

## Module Overview

| Module             | Lines  | Description                                              |
|--------------------|--------|----------------------------------------------------------|
| `abstract/`        | ~5,000 | Zonotope, HB constraints, fixpoint, **CEGAR refinement** |
| `decomposition/`   | ~5,900 | Interaction graph, A/G contracts, SMT discharge          |
| `policy/`          | ~5,700 | ONNX loader, Lipschitz, DeepZ, recurrent support         |
| `sampling/`        | ~7,800 | IS, cross-entropy, concentration, adaptive SIS           |
| `reporting/`       | ~5,800 | Certificates, proof verification, TCB analysis           |
| `search/`          | ~3,200 | MCTS, UCB1-Safety, HB pruning                            |
| `spec/`            | ~3,600 | BNF grammar, temporal logic, safety library              |
| Others             | ~4,900 | env, evaluation, hb, race, trace, pipeline, cli          |
| **Total**          |**~56K**| **892 tests across 35 test files**                       |

## Testing

```bash
pytest tests/ -v   # 892 tests, <10s
```

## API Reference

See [API.md](API.md) for the programmatic API.

## License

MIT — see `pyproject.toml` for details.
