# TensorGuard: Static Verification of PyTorch `nn.Module` Computation Graphs

A constraint-based static verifier that catches shape, device, and phase bugs
in PyTorch models — **zero annotations required**.

TensorGuard extracts the computation graph from your `nn.Module`, encodes shape
arithmetic into Z3 constraints with custom theory propagators, and either
proves the model safe or returns a concrete counterexample.

## Quickstart

```bash
pip install -e ".[smt]"   # requires z3-solver
```

```python
from src.model_checker import verify_model

result = verify_model(
    source=open("my_model.py").read(),
    input_shapes={"x": ("batch", 3, 224, 224)},
)
if result.safe:
    print("✓ All shapes, devices, and phases are compatible")
else:
    for v in result.counterexample.violations:
        print(f"✗ {v.message}")
```

## What It Catches

TensorGuard tracks shape information across multi-step flows — the class of
bugs that survive linting and even dynamic test runs with lucky shapes.

| Bug class | Example |
|-----------|---------|
| **Residual dimension mismatch** | Skip connection adds tensors of shape `(B, 64, H, W)` and `(B, 128, H, W)` |
| **Broadcasting through projections** | Linear projects to `(B, 64)`, then broadcasts against `(B, 1, 64)` in an unexpected way |
| **Matmul inner-dim mismatch** | `x @ y.T` where `x` is `(B, 64, 32)` and `y` is `(B, 48, 32)` — inner dims 32 vs 48 after transpose |
| **Cross-submodule device inconsistency** | Encoder on `cuda:0`, decoder parameters still on `cpu` |
| **Phase-dependent bugs** | `Dropout` or `BatchNorm` behaves differently in train vs eval; forgetting `model.eval()` at inference |

## How It Works

1. **Graph extraction** — Parses `__init__` and `forward` to build a typed
   computation graph of tensor operations.

2. **Z3 constraint encoding** — Shape dimensions become Z3 integer variables.
   Four custom `UserPropagator` plugins handle domain-specific reasoning:
   - `BroadcastPropagator` — NumPy-style broadcasting rules
   - `StridePropagator` — convolution / pooling output-size arithmetic
   - `DevicePropagator` — device-placement consistency
   - `PhasePropagator` — train/eval phase tracking

3. **Contract discovery (CEGAR)** — A counterexample-guided loop
   (`shape_cegar.py`) discovers implicit shape contracts that the module
   relies on but never states. Guards harvested from `isinstance` checks,
   assertions, and conditional branches seed the predicate pool.

4. **Verdict** — If the solver finds `UNSAT`, the model is safe for all
   valid inputs matching the declared shapes. If `SAT`, TensorGuard returns a
   concrete counterexample with the offending operation and witness shapes.

## Evaluation Results

Compared against a syntactic shape-checking baseline:

| Suite | Description | TensorGuard F1 | Syntactic F1 |
|-------|-------------|:-----------:|:------------:|
| **A** | 18 theory-exercising benchmarks | **1.00** | 0.00 |
| **B** | 15 production architectures | **1.00** | 0.92 |
| **C+D** | 17 real-world models (0 false positives) | **1.00** | 0.82 |

## Directory Structure

```
src/
├── model_checker.py        # Main constraint-based verifier
├── shape_cegar.py          # Contract discovery loop (CEGAR)
├── guard_extractor.py      # Guard harvesting from source
├── pipeline.py             # Integrated analysis pipeline
├── tensor_shapes.py        # Shape arithmetic utilities
├── smt/
│   ├── z3_backend.py       # Core Z3 solver interface
│   ├── encoder.py          # Constraint encoder
│   ├── solver.py           # Solver orchestration
│   ├── broadcast_theory.py # BroadcastPropagator plugin
│   ├── stride_theory.py    # StridePropagator plugin
│   ├── device_theory.py    # DevicePropagator plugin
│   └── phase_theory.py     # PhasePropagator plugin
├── domains/                # Abstract domains
├── types.py                # Core type definitions
└── config.py               # Configuration

experiments/                # Benchmark scripts and result JSON files
tests/                      # 1260 unit tests
setup.py                    # pip install -e ".[smt]"
```

## Installation

Requires Python ≥ 3.9.

```bash
git clone <repo-url>
cd implementation
pip install -e ".[smt]"    # installs z3-solver
```

Run the test suite:

```bash
pytest tests/
```

## License

MIT
