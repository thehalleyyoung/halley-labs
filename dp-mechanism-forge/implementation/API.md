# DP-Forge API Reference

## CLI Commands

### `dp-forge synthesize`

Synthesize an optimal DP mechanism via CEGIS.

**Query source (pick one):**
- `--query-type`, `-q` — Built-in query type: `counting`, `histogram`, `range`, `workload`.
- `--spec-file`, `-f` — Path to a JSON query specification file (for arbitrary queries).

**Parameters:**
- `--epsilon`, `-e` — Privacy parameter ε (required with `--query-type`; read from spec file with `--spec-file`).
- `--delta`, `-d` — Approximate DP parameter δ (default: 0.0).
- `--k` — Number of discretization bins (default: 50).
- `--loss` — Loss function: `l1`, `l2`, `linf` (default: `l2`).
- `--domain-size`, `-n` — Query domain size (default: 2, used with `--query-type`).
- `--output`, `-o` — Output path for mechanism file.
- `--format` — Output format: `json`, `python`, `cpp`, `rust` (default: `json`).
- `--solver` — LP/SDP solver: `highs`, `glpk`, `scs`, `mosek`, `auto` (default: `auto`).
- `--max-iter` — Maximum CEGIS iterations (default: 50).

```bash
# Built-in query type
dp-forge synthesize --query-type counting --epsilon 1.0 -n 2

# Custom query from spec file
dp-forge synthesize --spec-file my_query.json

# With code generation
dp-forge synthesize --spec-file my_query.json --format python -o my_mech.py
```

### `dp-forge check-spec <spec_file>`

Validate a query specification file without running synthesis.

```bash
dp-forge check-spec my_query.json
# ✓ my_query.json is a valid query specification
#   Query values: 5 distinct outputs
#   Sensitivity: 1.0
#   Privacy: ε=1.0
#   Discretization: k=50
#   Loss function: L2
```

### `dp-forge init-spec [name] --template <template>`

Generate a starter JSON query specification file.

**Arguments:**
- `name` — Output filename (default: `my_query.json`).
- `--template`, `-t` — Template: `counting`, `sum`, `median`, `custom` (default: `counting`).

```bash
dp-forge init-spec my_query.json --template median
```

### `dp-forge verify`

Verify that a mechanism satisfies (ε, δ)-DP.

**Options:**
- `--mechanism`, `-m` — Path to mechanism JSON file (required).
- `--epsilon`, `-e` — Target ε (default: from file).
- `--delta`, `-d` — Target δ (default: from file).

```bash
dp-forge verify --mechanism mech.json --epsilon 1.0
```

### `dp-forge compare`

Compare a synthesized mechanism against baselines.

```bash
dp-forge compare --mechanism mech.json --baselines laplace gaussian
```

### `dp-forge benchmark`

Run synthesis benchmarks.

```bash
dp-forge benchmark --tier 1 --output-dir results/
```

### `dp-forge codegen`

Generate standalone deployment code from a mechanism file.

```bash
dp-forge codegen --mechanism mech.json --language python --output mech.py
dp-forge codegen --mechanism mech.json --language rust --output mech.rs
```

### `dp-forge info`

Display information about a mechanism file.

```bash
dp-forge info --mechanism mech.json
```

---

## JSON Spec File Format

A query specification file defines what query the mechanism should answer:

```json
{
  "query_values": [0.0, 1.0, 2.0, 3.0, 4.0],
  "sensitivity": 1.0,
  "epsilon": 1.0,
  "delta": 0.0,
  "k": 50,
  "loss": "l2",
  "domain": "description of query",
  "adjacency": "consecutive"
}
```

| Field | Required | Type | Description |
|---|---|---|---|
| `query_values` | ✅ | `list[float]` | Distinct query output values f(x₁), …, f(xₙ) |
| `sensitivity` | ✅ | `float` | Global sensitivity Δf > 0 |
| `epsilon` | ✅ | `float` | Privacy parameter ε > 0 |
| `delta` | | `float` | Approximate DP δ ∈ [0, 1) (default: 0.0) |
| `k` | | `int` | Discretization bins ≥ 2 (default: 100) |
| `loss` | | `str` | `"l1"`, `"l2"`, or `"linf"` (default: `"l2"`) |
| `domain` | | `str` | Human-readable description |
| `adjacency` | | `str` | `"consecutive"` (Hamming-1) or `"complete"` |

---

## Python Library API

### Synthesis

```python
from dp_forge.types import QuerySpec
from dp_forge.cegis_loop import CEGISSynthesize
from dp_forge.extractor import compute_mechanism_mse
import numpy as np

# Built-in query type
spec = QuerySpec.counting(n=10, epsilon=1.0, delta=0.0, k=50)

# Custom query (any values)
spec = QuerySpec(
    query_values=np.array([0.0, 2.5, 5.0, 7.5, 10.0]),
    domain="custom sum",
    sensitivity=10.0,
    epsilon=1.0,
    k=50,
)

result = CEGISSynthesize(spec)
print(f"Optimal objective: {result.obj_val:.4f}")
print(f"CEGIS iterations: {result.iterations}")
```

### Verification

```python
from dp_forge.types import ExtractedMechanism
from dp_forge.verifier import verify_dp

mechanism = ExtractedMechanism(p_final=result.mechanism)
is_valid, violation = verify_dp(mechanism, epsilon=1.0, delta=0.0)
```

### Code Generation

```python
from dp_forge.codegen import PythonCodeGenerator

gen = PythonCodeGenerator()
code = gen.generate(mechanism, spec)
```

---

## Benchmarking

### Run Experiments

```bash
cd experiments
python run_benchmarks.py
# Results written to experiments/benchmark_results.json
```

### Benchmark Results Format

The output JSON contains five experiment suites:

| Suite | Description |
|-------|-------------|
| `counting` | ε × n sweep for counting queries vs Laplace/Staircase |
| `histogram` | Multi-bin histogram queries |
| `loss_comparison` | L1 vs L2 loss objectives |
| `approx_dp` | (ε,δ)-DP vs calibrated Gaussian |
| `scalability` | Synthesis time vs domain size |

Each entry includes `cegis_worst_mse`, `laplace_mse`, `improvement_vs_laplace`, `synthesis_time_s`, and `converged`.
