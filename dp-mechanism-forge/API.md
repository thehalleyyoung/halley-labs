# DP-Forge API Reference

Complete API reference for the DP-Forge toolkit — CLI commands, spec file formats,
and the full Python library covering all 22 subpackages.

---

## Table of Contents

- [CLI Commands](#cli-commands)
- [Spec File Format](#spec-file-format-json-yaml--csv)
- [Python Library API](#python-library-api)
  - [Core Synthesis](#core-synthesis)
  - [Query Specifications](#query-specifications)
  - [Extracted Mechanisms](#extracted-mechanisms)
  - [Synthesis Configuration](#synthesis-configuration)
  - [Verification](#verification)
  - [Baseline Mechanisms](#baseline-mechanisms)
  - [Code Generation](#code-generation)
  - [Privacy Accounting](#privacy-accounting)
  - [Advanced Composition](#advanced-composition)
  - [Rényi DP Accounting](#rényi-dp-accounting)
  - [zCDP Accounting](#zcdp-accounting)
  - [Game-Theoretic Synthesis](#game-theoretic-synthesis)
  - [Grid Refinement](#grid-refinement)
  - [Infinite-Dimensional Synthesis](#infinite-dimensional-synthesis)
  - [Lattice Methods](#lattice-methods)
  - [Sparse Optimization](#sparse-optimization)
  - [Multi-Dimensional Mechanisms](#multi-dimensional-mechanisms)
  - [Verification Backends](#verification-backends)
  - [SMT-Based Verification](#smt-based-verification)
  - [Robust Synthesis](#robust-synthesis)
  - [Interpolation-Based Verification](#interpolation-based-verification)
  - [Certificates](#certificates)
  - [Local Differential Privacy](#local-differential-privacy)
  - [Streaming Mechanisms](#streaming-mechanisms)
  - [Privacy Amplification](#privacy-amplification)
  - [Workload Optimization](#workload-optimization)
  - [Automatic Differentiation](#automatic-differentiation)
  - [Types and Enums](#types-and-enums)
- [Benchmarking](#benchmarking)

---

## CLI Commands

### `dp-forge synthesize`

Synthesize an optimal DP mechanism via CEGIS.

**Query source (pick one):**
- `--query-type`, `-q` — Built-in query type: `counting`, `histogram`, `range`, `workload`.
- `--spec-file`, `-f` — Path to a JSON, YAML, or CSV query specification file (for arbitrary queries).

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
- `--output-format` — Console output format: `text` (default), `json`, `python-code`.
- `--compare-baseline` — Compare against Laplace, Gaussian, and Exponential baselines.
- `--export-opendp` — After synthesis, output an OpenDP Measurement definition as Python code.

```bash
# Built-in query type
dp-forge synthesize --query-type counting --epsilon 1.0 -n 2

# Custom query from spec file (JSON or YAML)
dp-forge synthesize --spec-file my_query.json
dp-forge synthesize --spec-file my_query.yaml

# With code generation
dp-forge synthesize --spec-file my_query.json --format python -o my_mech.py

# Machine-readable JSON console output
dp-forge synthesize -q counting -e 1.0 --output-format json

# Emit ready-to-use Python code to stdout
dp-forge synthesize -q counting -e 1.0 --output-format python-code

# Compare against baselines inline
dp-forge synthesize -q counting -e 1.0 --compare-baseline

# CSV query workload with OpenDP export
dp-forge synthesize --spec-file workload.csv --export-opendp
```

### `dp-forge check-spec <spec_file>`

Validate a query specification file (JSON or YAML) without running synthesis.

```bash
dp-forge check-spec my_query.json
dp-forge check-spec my_query.yaml
# ✓ my_query.yaml is a valid query specification
#   Query values: 5 distinct outputs
#   Sensitivity: 1.0
#   Privacy: ε=1.0
#   Discretization: k=50
#   Loss function: L2
```

### `dp-forge init-spec [name] --template <template>`

Generate a starter query specification file (JSON or YAML).

**Arguments:**
- `name` — Output filename (default: `my_query.json`). Use `.yaml`/`.yml` extension for YAML output.
- `--template`, `-t` — Template: `counting`, `sum`, `median`, `custom` (default: `counting`).

```bash
dp-forge init-spec my_query.json --template median
dp-forge init-spec my_query.yaml --template counting
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

## Spec File Format (JSON, YAML & CSV)

A query specification file defines what query the mechanism should answer.
JSON, YAML, and CSV formats are supported.

### JSON

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

### YAML

```yaml
query_values: [0.0, 1.0, 2.0, 3.0, 4.0]
sensitivity: 1.0
epsilon: 1.0
delta: 0.0
k: 50
loss: l2
domain: description of query
adjacency: consecutive
```

### CSV (Query Workload)

Each row is a query specification with columns `query_type`, `sensitivity`,
and optionally `description`:

```csv
query_type,sensitivity,description
counting,1.0,Count of matching rows
sum,10.0,Sum of salaries
mean,1.0,Average age
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

### Core Synthesis

#### `CEGISSynthesize(spec, max_iter=300)`

Top-level function that runs the full CEGIS (Counter-Example Guided Inductive
Synthesis) loop to produce an optimal differentially private mechanism for
a given query specification.

```python
from dp_forge.cegis_loop import CEGISSynthesize
from dp_forge.types import QuerySpec

spec = QuerySpec.counting(n=10, epsilon=1.0, delta=0.0, k=50)
result = CEGISSynthesize(spec, max_iter=300)

print(result.obj_val)    # Optimal objective value (worst-case expected loss)
print(result.iterations) # Number of CEGIS iterations used
print(result.mechanism)  # The probability matrix (numpy array)
print(result.converged)  # True if CEGIS converged within max_iter
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `spec` | `QuerySpec` | required | Query specification defining the synthesis problem |
| `max_iter` | `int` | `300` | Maximum number of CEGIS refinement iterations |

**Returns:** `CEGISResult` — contains `.obj_val`, `.iterations`, `.mechanism`, `.converged`.

#### `CEGISEngine`

Low-level engine class providing fine-grained control over the CEGIS loop.
Use this when you need access to intermediate states, custom termination
conditions, or step-by-step iteration.

```python
from dp_forge.cegis_loop import CEGISEngine
from dp_forge.types import QuerySpec, SynthesisConfig

spec = QuerySpec.counting(n=5, epsilon=1.0, delta=0.0, k=50)
config = SynthesisConfig(max_iter=100)

engine = CEGISEngine(spec, config=config)
engine.initialize()

while not engine.converged:
    engine.step()
    print(f"Iteration {engine.iteration}: obj={engine.current_obj:.6f}")

result = engine.extract_result()
```

**Constructor:** `CEGISEngine(spec: QuerySpec, config: SynthesisConfig = None)`

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `initialize()` | `None` | Set up initial LP and counterexample set |
| `step()` | `None` | Execute one CEGIS iteration (solve LP, verify, add counterexample) |
| `extract_result()` | `CEGISResult` | Build the final result from the current engine state |

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `converged` | `bool` | Whether the loop has converged |
| `iteration` | `int` | Current iteration count |
| `current_obj` | `float` | Current objective value |

#### `quick_synthesize()`

Convenience function for rapid one-line synthesis with sensible defaults.
Wraps `CEGISSynthesize` with simplified arguments.

```python
from dp_forge.cegis_loop import quick_synthesize

result = quick_synthesize(
    query_type="counting",
    n=10,
    epsilon=1.0,
    delta=0.0,
    k=50,
    loss="l2",
)
print(f"MSE: {result.obj_val:.4f}, converged: {result.converged}")
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `query_type` | `str` | required | One of `"counting"`, `"histogram"`, `"range"`, `"workload"` |
| `n` | `int` | `2` | Domain size |
| `epsilon` | `float` | required | Privacy parameter ε |
| `delta` | `float` | `0.0` | Approximate DP parameter δ |
| `k` | `int` | `50` | Discretization bins |
| `loss` | `str` | `"l2"` | Loss function: `"l1"`, `"l2"`, or `"linf"` |

**Returns:** `CEGISResult`

---

### Query Specifications

#### `QuerySpec`

Dataclass defining the synthesis problem: what query is being answered,
its sensitivity, the privacy parameters, and the discretization granularity.

```python
from dp_forge.types import QuerySpec
import numpy as np

spec = QuerySpec(
    query_values=np.array([0.0, 2.5, 5.0, 7.5, 10.0]),
    domain="custom sum query",
    sensitivity=10.0,
    epsilon=1.0,
    k=50,
    delta=0.0,
    loss_fn="l2",
)
```

**Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `query_values` | `np.ndarray` | required | Distinct query output values f(x₁), …, f(xₙ) |
| `domain` | `str` | `""` | Human-readable description of the query domain |
| `sensitivity` | `float` | `1.0` | Global sensitivity Δf |
| `epsilon` | `float` | required | Privacy budget ε > 0 |
| `k` | `int` | `50` | Number of discretization bins |
| `delta` | `float` | `0.0` | Approximate DP parameter δ |
| `loss_fn` | `str` | `"l2"` | Loss function identifier |

#### `QuerySpec.counting(n, epsilon, delta, k)`

Factory method for counting queries (how many rows match a predicate).
Query values are integers 0, 1, …, n−1 with sensitivity 1.

```python
spec = QuerySpec.counting(n=100, epsilon=1.0, delta=1e-6, k=80)
print(spec.query_values)  # array([0, 1, 2, ..., 99])
print(spec.sensitivity)   # 1.0
```

#### `QuerySpec.histogram(n_bins, epsilon, delta, k)`

Factory method for histogram queries. Produces a multi-bin histogram
specification with sensitivity 1 (one individual contributes to one bin).

```python
spec = QuerySpec.histogram(n_bins=10, epsilon=0.5, delta=1e-5, k=60)
print(spec.query_values)  # array([0, 1, 2, ..., 9])
```

---

### Extracted Mechanisms

#### `ExtractedMechanism`

Represents a synthesized mechanism extracted from the CEGIS solution.
Wraps the raw probability matrix and exposes its structural properties.

```python
from dp_forge.types import ExtractedMechanism
import numpy as np

mechanism = ExtractedMechanism(p_final=result.mechanism)
print(mechanism.n)        # Number of input values (rows)
print(mechanism.k)        # Number of output bins (columns)
print(mechanism.p_final)  # The n × k probability matrix
```

**Constructor:** `ExtractedMechanism(p_final: np.ndarray)`

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `n` | `int` | Number of input values (rows of the matrix) |
| `k` | `int` | Number of output bins (columns of the matrix) |
| `p_final` | `np.ndarray` | The n × k conditional probability matrix |

---

### Synthesis Configuration

#### `SynthesisConfig`

Configuration dataclass controlling CEGIS behavior: iteration limits,
solver selection, convergence tolerances, and logging.

```python
from dp_forge.types import SynthesisConfig

config = SynthesisConfig(
    max_iter=500,
    solver="highs",
    tol=1e-10,
    verbose=True,
)
```

**Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_iter` | `int` | `300` | Maximum CEGIS iterations |
| `solver` | `str` | `"auto"` | LP solver backend (`"highs"`, `"glpk"`, `"scs"`, `"mosek"`, `"auto"`) |
| `tol` | `float` | `1e-9` | Convergence tolerance |
| `verbose` | `bool` | `False` | Enable detailed logging |

---

### Verification

#### `quick_verify(p, epsilon, delta, tol)`

Fast convenience function to check whether a probability matrix satisfies
(ε, δ)-differential privacy.

```python
from dp_forge.verifier import quick_verify

is_private = quick_verify(result.mechanism, epsilon=1.0, delta=0.0, tol=1e-9)
print(f"Satisfies DP: {is_private}")
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `p` | `np.ndarray` | required | Probability matrix to verify |
| `epsilon` | `float` | required | Target privacy parameter ε |
| `delta` | `float` | `0.0` | Target approximate DP parameter δ |
| `tol` | `float` | `1e-9` | Numerical tolerance for constraint checking |

**Returns:** `bool` — `True` if the mechanism satisfies (ε, δ)-DP within tolerance.

#### `verify(p, epsilon, delta, edges, tol)`

Full verification function that returns a detailed `VerifyResult` with
information about any constraint violations found.

```python
from dp_forge.verifier import verify

vr = verify(result.mechanism, epsilon=1.0, delta=0.0, edges=None, tol=1e-9)
print(vr.valid)      # True/False
print(vr.violation)  # Details of the worst violation, or None
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `p` | `np.ndarray` | required | Probability matrix to verify |
| `epsilon` | `float` | required | Target ε |
| `delta` | `float` | `0.0` | Target δ |
| `edges` | `list[tuple]` | `None` | Adjacency pairs; `None` means consecutive |
| `tol` | `float` | `1e-9` | Numerical tolerance |

**Returns:** `VerifyResult` — with `.valid` (bool) and `.violation` (details or `None`).

#### `verify_extracted_mechanism(mechanism, spec)`

Verify an `ExtractedMechanism` object against its originating `QuerySpec`.

```python
from dp_forge.verifier import verify_extracted_mechanism
from dp_forge.types import ExtractedMechanism

mechanism = ExtractedMechanism(p_final=result.mechanism)
vr = verify_extracted_mechanism(mechanism, spec)
assert vr.valid, f"Verification failed: {vr.violation}"
```

**Returns:** `VerifyResult`

#### `PrivacyVerifier`

Class providing multiple verification methods for different DP variants.

```python
from dp_forge.verifier import PrivacyVerifier

pv = PrivacyVerifier(tol=1e-9)

# Check pure ε-DP
pure_result = pv.verify_pure_dp(mechanism.p_final, epsilon=1.0)

# Check approximate (ε,δ)-DP
approx_result = pv.verify_approx_dp(mechanism.p_final, epsilon=1.0, delta=1e-5)

# Unified method dispatching to the right check
result = pv.verify_mechanism(mechanism.p_final, epsilon=1.0, delta=0.0)
print(result.valid)
```

**Constructor:** `PrivacyVerifier(tol: float = 1e-9)`

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `verify_mechanism(p, epsilon, delta)` | `VerifyResult` | Dispatch to pure or approximate DP check |
| `verify_pure_dp(p, epsilon)` | `VerifyResult` | Verify pure ε-differential privacy |
| `verify_approx_dp(p, epsilon, delta)` | `VerifyResult` | Verify (ε, δ)-differential privacy |

#### `MonteCarloVerifier`

Statistical verification via repeated sampling. Useful for large mechanisms
where exact LP-based verification is too expensive.

```python
from dp_forge.verifier import MonteCarloVerifier

mc = MonteCarloVerifier(n_samples=100_000, confidence=0.99)
result = mc.verify(mechanism.p_final, epsilon=1.0, delta=0.0)
print(f"Statistically valid: {result.valid}")
```

**Constructor:** `MonteCarloVerifier(n_samples: int = 100000, confidence: float = 0.99)`

#### `hockey_stick_divergence(p, q, epsilon)`

Compute the hockey-stick divergence D_{e^ε}(p ∥ q) = Σ max(0, p_i − e^ε · q_i)
between two distributions. This is the fundamental quantity in (ε, δ)-DP:
a mechanism satisfies (ε, δ)-DP iff the hockey-stick divergence is ≤ δ for
all adjacent pairs.

```python
from dp_forge.verifier import hockey_stick_divergence
import numpy as np

p = np.array([0.6, 0.3, 0.1])
q = np.array([0.3, 0.4, 0.3])
div = hockey_stick_divergence(p, q, epsilon=1.0)
print(f"Hockey-stick divergence: {div:.6f}")
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `p` | `np.ndarray` | First distribution (1-D probability vector) |
| `q` | `np.ndarray` | Second distribution (1-D probability vector) |
| `epsilon` | `float` | Privacy parameter ε |

**Returns:** `float` — the hockey-stick divergence value.

---

### Baseline Mechanisms

All baselines live in `dp_forge.baselines` and implement standard DP mechanisms
for comparison against CEGIS-synthesized results.

#### `LaplaceMechanism`

The classic Laplace mechanism for real-valued queries. Adds Lap(sensitivity/ε)
noise, satisfying pure ε-DP.

```python
from dp_forge.baselines import LaplaceMechanism

lap = LaplaceMechanism(epsilon=1.0, sensitivity=1.0)
print(f"MSE:  {lap.mse():.6f}")
print(f"MAE:  {lap.mae():.6f}")

sample = lap.sample()         # Draw one noisy sample
pdf_val = lap.pdf(0.5)        # PDF at x=0.5
cdf_val = lap.cdf(1.0)        # CDF at x=1.0
```

**Constructor:** `LaplaceMechanism(epsilon: float, sensitivity: float = 1.0)`

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `mse()` | `float` | Mean squared error: 2(Δf/ε)² |
| `mae()` | `float` | Mean absolute error: Δf/ε |
| `sample()` | `float` | Draw one noisy sample from the Laplace distribution |
| `pdf(x)` | `float` | Probability density at point x |
| `cdf(x)` | `float` | Cumulative distribution at point x |

#### `GaussianMechanism`

The Gaussian mechanism for (ε, δ)-approximate DP. Adds N(0, σ²) noise with
σ calibrated to achieve the target (ε, δ).

```python
from dp_forge.baselines import GaussianMechanism

gauss = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=1.0)
print(f"MSE: {gauss.mse():.6f}")
print(f"MAE: {gauss.mae():.6f}")
```

**Constructor:** `GaussianMechanism(epsilon: float, delta: float, sensitivity: float = 1.0)`

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `mse()` | `float` | Mean squared error: σ² |
| `mae()` | `float` | Mean absolute error: σ√(2/π) |

#### `StaircaseMechanism`

The optimal staircase mechanism for pure ε-DP, which can beat Laplace in
worst-case error for certain query structures.

```python
from dp_forge.baselines import StaircaseMechanism

stair = StaircaseMechanism(epsilon=1.0, sensitivity=1.0)
print(f"MSE: {stair.mse():.6f}")
```

**Constructor:** `StaircaseMechanism(epsilon: float, sensitivity: float = 1.0)`

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `mse()` | `float` | Mean squared error of the staircase distribution |

#### `GeometricMechanism`

Discrete analogue of the Laplace mechanism for integer-valued queries with
integer sensitivity (e.g., counting queries).

```python
from dp_forge.baselines import GeometricMechanism

geo = GeometricMechanism(epsilon=1.0, sensitivity=1)
print(f"MSE: {geo.mse():.6f}")
```

**Constructor:** `GeometricMechanism(epsilon: float, sensitivity: int)`

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `mse()` | `float` | Mean squared error |

#### `ExponentialMechanism`

The exponential mechanism for utility-based selection from a discrete set
of candidates. Selects output r with probability proportional to
exp(ε · u(x, r) / (2Δu)).

```python
from dp_forge.baselines import ExponentialMechanism
import numpy as np

utilities = np.array([1.0, 3.0, 2.0, 5.0])
exp_mech = ExponentialMechanism(epsilon=1.0, sensitivity=1.0)
selected_index = exp_mech.select(utilities)
print(f"Selected index: {selected_index}")
```

**Constructor:** `ExponentialMechanism(epsilon: float, sensitivity: float = 1.0)`

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `select(utilities)` | `int` | Sample an index proportional to exp(ε · u / 2Δu) |

#### `MatrixMechanism`

Workload-aware Gaussian mechanism that optimizes the noise covariance
for a given workload matrix W, achieving lower error than per-query Gaussian.

```python
from dp_forge.baselines import MatrixMechanism
import numpy as np

W = np.eye(5)  # identity workload (5 counting queries)
mm = MatrixMechanism(epsilon=1.0, delta=1e-5, workload=W)
print(f"Total MSE: {mm.mse():.6f}")
```

**Constructor:** `MatrixMechanism(epsilon: float, delta: float, workload: np.ndarray)`

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `mse()` | `float` | Total mean squared error across the workload |

---

### Code Generation

Generate deployable source code from a synthesized mechanism. Supports
Python, C++, and Rust backends.

#### `PythonCodeGenerator`

```python
from dp_forge.codegen import PythonCodeGenerator
from dp_forge.types import ExtractedMechanism

gen = PythonCodeGenerator()
mechanism = ExtractedMechanism(p_final=result.mechanism)
code = gen.generate(mechanism, spec)
print(code)  # Standalone Python module implementing the mechanism
```

#### `CppCodeGenerator`

```python
from dp_forge.codegen import CppCodeGenerator

gen = CppCodeGenerator()
code = gen.generate(mechanism, spec)
with open("mechanism.cpp", "w") as f:
    f.write(code)
```

#### `RustCodeGenerator`

```python
from dp_forge.codegen import RustCodeGenerator

gen = RustCodeGenerator()
code = gen.generate(mechanism, spec)
with open("mechanism.rs", "w") as f:
    f.write(code)
```

**Common interface for all generators:**

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `generate` | `(mechanism: ExtractedMechanism, spec: QuerySpec)` | `str` | Emit standalone source code implementing the mechanism |

---

### Privacy Accounting

Module `dp_forge.privacy_accounting` provides budget tracking and advanced
composition tools.

#### `PrivacyBudgetTracker`

Track cumulative privacy expenditure across multiple mechanism invocations.
Raises an error if spending would exceed the total budget.

```python
from dp_forge.privacy_accounting import PrivacyBudgetTracker

tracker = PrivacyBudgetTracker(total_epsilon=5.0, total_delta=1e-5)

tracker.spend(epsilon=1.0, delta=0.0)
print(tracker.remaining_epsilon)  # 4.0

tracker.spend(epsilon=2.0, delta=1e-6)
print(tracker.remaining_epsilon)  # 2.0
```

**Constructor:** `PrivacyBudgetTracker(total_epsilon: float, total_delta: float = 0.0)`

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `remaining_epsilon` | `float` | Unused ε budget |
| `remaining_delta` | `float` | Unused δ budget |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `spend(epsilon, delta)` | `None` | Deduct (ε, δ) from the remaining budget |

#### `MomentsAccountant`

Advanced composition via the moments accountant technique (Abadi et al., 2016).
Provides tighter (ε, δ) bounds for sequences of adaptive mechanisms than
naive composition.

```python
from dp_forge.privacy_accounting import MomentsAccountant

ma = MomentsAccountant()
ma.add_mechanism(epsilon=1.0, delta=0.0)
ma.add_mechanism(epsilon=1.0, delta=0.0)

total_eps = ma.get_epsilon(delta=1e-5)
print(f"Composed ε at δ=1e-5: {total_eps:.4f}")
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `add_mechanism(epsilon, delta)` | `None` | Record one mechanism invocation |
| `get_epsilon(delta)` | `float` | Compute total ε for a given target δ |
| `get_delta(epsilon)` | `float` | Compute total δ for a given target ε |

#### `RenyiDPAccountant`

Rényi differential privacy accountant operating in the (α, ε)-RDP framework.

```python
from dp_forge.privacy_accounting import RenyiDPAccountant

rdp = RenyiDPAccountant()
rdp.add_mechanism(noise_multiplier=1.0, sample_rate=0.01, steps=1000)
eps = rdp.get_epsilon(delta=1e-5)
print(f"RDP ε at δ=1e-5: {eps:.4f}")
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `add_mechanism(noise_multiplier, sample_rate, steps)` | `None` | Record a subsampled Gaussian mechanism |
| `get_epsilon(delta)` | `float` | Convert RDP guarantee to (ε, δ)-DP |

---

### Advanced Composition

Module `dp_forge.composition` provides advanced privacy accounting using
Fourier-based methods, privacy loss distributions, and adaptive filters.

#### `FourierAccountant`

Numerically exact privacy accounting via Fourier transform of the privacy
loss distribution (Koskela et al., 2020). Provides the tightest known
composition bounds.

```python
from dp_forge.composition import FourierAccountant

fa = FourierAccountant(n_points=2**16)
fa.add_mechanism(noise_multiplier=1.0, sample_rate=0.01, steps=500)
eps = fa.get_epsilon(delta=1e-5)
print(f"Fourier ε at δ=1e-5: {eps:.6f}")
```

**Constructor:** `FourierAccountant(n_points: int = 65536)`

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `add_mechanism(noise_multiplier, sample_rate, steps)` | `None` | Add a subsampled Gaussian mechanism |
| `get_epsilon(delta)` | `float` | Compute the tightest ε for a target δ |

#### `PrivacyLossDistribution`

Explicit representation of the privacy loss random variable for a mechanism.
Supports arithmetic composition (convolution) of multiple PLDs.

```python
from dp_forge.composition import PrivacyLossDistribution

pld1 = PrivacyLossDistribution.from_gaussian(sigma=1.0)
pld2 = PrivacyLossDistribution.from_gaussian(sigma=2.0)
composed = pld1.compose(pld2)
eps = composed.get_epsilon(delta=1e-5)
print(f"Composed ε: {eps:.4f}")
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `from_gaussian(sigma)` | `PrivacyLossDistribution` | Construct PLD for Gaussian mechanism (class method) |
| `compose(other)` | `PrivacyLossDistribution` | Convolve two PLDs (sequential composition) |
| `get_epsilon(delta)` | `float` | Convert to (ε, δ)-DP |
| `self_compose(n)` | `PrivacyLossDistribution` | n-fold self-composition |

#### `PrivacyFilter`

Adaptive composition tool that halts queries when the privacy budget is
exhausted. Unlike `PrivacyBudgetTracker`, the filter provides valid
composition even for adaptively chosen mechanisms.

```python
from dp_forge.composition import PrivacyFilter

pf = PrivacyFilter(total_epsilon=5.0, total_delta=1e-5)
can_continue = pf.check_and_spend(epsilon=1.0, delta=1e-6)
print(f"Can continue: {can_continue}")  # True
```

**Constructor:** `PrivacyFilter(total_epsilon: float, total_delta: float)`

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `check_and_spend(epsilon, delta)` | `bool` | Spend budget and return whether budget remains |

#### `PrivacyOdometer`

Continuously tracks cumulative privacy loss with real-time guarantees.
Reports the current total (ε, δ) at any point during a sequence of queries.

```python
from dp_forge.composition import PrivacyOdometer

odo = PrivacyOdometer()
odo.record(epsilon=1.0, delta=0.0)
odo.record(epsilon=0.5, delta=1e-6)
current_eps, current_delta = odo.current_budget()
print(f"Running total: ε={current_eps:.2f}, δ={current_delta:.2e}")
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `record(epsilon, delta)` | `None` | Log one mechanism invocation |
| `current_budget()` | `tuple[float, float]` | Return cumulative (ε, δ) so far |

#### `MixedAccountant`

Combines multiple accounting strategies (basic, moments, RDP, Fourier) and
returns the tightest bound across all of them.

```python
from dp_forge.composition import MixedAccountant

ma = MixedAccountant()
ma.add_mechanism(noise_multiplier=1.0, sample_rate=0.01, steps=1000)
eps = ma.get_epsilon(delta=1e-5)
print(f"Tightest ε across all methods: {eps:.4f}")
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `add_mechanism(noise_multiplier, sample_rate, steps)` | `None` | Record a mechanism across all internal accountants |
| `get_epsilon(delta)` | `float` | Return the minimum ε across all accountant strategies |

---

### Rényi DP Accounting

Module `dp_forge.rdp` provides dedicated Rényi DP tracking and conversion.

#### `RDPAccountant`

Full-featured Rényi DP accountant supporting heterogeneous mechanism
composition and optimal conversion to (ε, δ)-DP.

```python
from dp_forge.rdp import RDPAccountant

acct = RDPAccountant(alphas=[2, 5, 10, 20, 50, 100])
acct.add_gaussian(sigma=1.0)
acct.add_gaussian(sigma=2.0)
eps = acct.get_epsilon(delta=1e-5)
print(f"RDP → (ε,δ)-DP: ε={eps:.4f}")
```

**Constructor:** `RDPAccountant(alphas: list[float] = None)`

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `add_gaussian(sigma)` | `None` | Record a Gaussian mechanism |
| `get_epsilon(delta)` | `float` | Optimal conversion to (ε, δ)-DP across all α |
| `get_rdp(alpha)` | `float` | Return the Rényi divergence at order α |

---

### zCDP Accounting

Module `dp_forge.zcdp` provides zero-Concentrated DP accounting.

#### `ZCDPAccountant`

Track privacy loss in the zCDP framework (Bun & Dwork, 2016).
zCDP composes linearly and converts to (ε, δ)-DP.

```python
from dp_forge.zcdp import ZCDPAccountant

acct = ZCDPAccountant()
acct.add_mechanism(rho=0.5)
acct.add_mechanism(rho=0.3)
eps = acct.get_epsilon(delta=1e-5)
print(f"zCDP → (ε,δ)-DP: ε={eps:.4f}")
```

**Constructor:** `ZCDPAccountant()`

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `add_mechanism(rho)` | `None` | Record a mechanism with zCDP parameter ρ |
| `get_epsilon(delta)` | `float` | Convert total ρ to (ε, δ)-DP |

---

### Game-Theoretic Synthesis

Module `dp_forge.game_theory` formulates mechanism design as a two-player
game between the mechanism designer and an adversary.

#### `MinimaxSolver`

Solves the minimax formulation: minimize the maximum expected loss over
all adjacent database pairs.

```python
from dp_forge.game_theory import MinimaxSolver

solver = MinimaxSolver(spec)
result = solver.solve()
print(f"Minimax objective: {result.obj_val:.6f}")
```

**Constructor:** `MinimaxSolver(spec: QuerySpec)`

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `solve()` | `CEGISResult` | Solve the minimax LP and return the optimal mechanism |

#### `NashSolver`

Computes a Nash equilibrium of the mechanism design game, where neither
the designer nor the adversary can unilaterally improve.

```python
from dp_forge.game_theory import NashSolver

solver = NashSolver(spec)
result = solver.solve()
print(f"Nash equilibrium value: {result.obj_val:.6f}")
```

**Constructor:** `NashSolver(spec: QuerySpec)`

#### `StackelbergSolver`

Solves the Stackelberg (leader-follower) formulation where the designer
commits first and the adversary best-responds.

```python
from dp_forge.game_theory import StackelbergSolver

solver = StackelbergSolver(spec)
result = solver.solve()
print(f"Stackelberg value: {result.obj_val:.6f}")
```

**Constructor:** `StackelbergSolver(spec: QuerySpec)`

---

### Grid Refinement

Module `dp_forge.grid` provides adaptive discretization for the output space.

#### `AdaptiveGridRefiner`

Iteratively refines the output discretization grid, concentrating bins
where the mechanism probability mass is highest. Improves solution quality
without increasing overall problem size.

```python
from dp_forge.grid import AdaptiveGridRefiner

refiner = AdaptiveGridRefiner(initial_k=20, max_k=200, refine_ratio=0.5)
refined_spec = refiner.refine(spec, result.mechanism)
print(f"Refined grid: {refined_spec.k} bins")
```

**Constructor:** `AdaptiveGridRefiner(initial_k: int, max_k: int, refine_ratio: float = 0.5)`

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `refine(spec, p)` | `QuerySpec` | Produce a new spec with a refined grid based on the current solution |

---

### Infinite-Dimensional Synthesis

Module `dp_forge.infinite` lifts the synthesis problem to a continuous
output space using semi-infinite LP techniques.

#### `InfiniteLPSolver`

Solves the mechanism design LP over a continuous output domain via
cutting-plane methods, avoiding discretization error entirely.

```python
from dp_forge.infinite import InfiniteLPSolver

solver = InfiniteLPSolver(spec, tol=1e-8)
result = solver.solve()
print(f"Infinite LP objective: {result.obj_val:.6f}")
```

**Constructor:** `InfiniteLPSolver(spec: QuerySpec, tol: float = 1e-8)`

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `solve()` | `CEGISResult` | Solve the semi-infinite LP to global optimality |

---

### Lattice Methods

Module `dp_forge.lattice` provides branch-and-bound optimization for
discrete mechanism design.

#### `BranchAndBound`

Exact optimization via branch-and-bound over the lattice of feasible
mechanism parameters. Guarantees global optimality with LP relaxation bounds.

```python
from dp_forge.lattice import BranchAndBound

bb = BranchAndBound(spec, branch_strategy="most_fractional")
result = bb.solve()
print(f"B&B optimal: {result.obj_val:.6f}")
```

**Constructor:** `BranchAndBound(spec: QuerySpec, branch_strategy: str = "most_fractional")`

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `solve()` | `CEGISResult` | Run branch-and-bound to find the global optimum |

---

### Sparse Optimization

Module `dp_forge.sparse` provides scalable decomposition methods for
large-scale mechanism synthesis.

#### `ColumnGenerator`

Column generation for large output spaces. Iteratively adds promising
output columns to the LP, avoiding the need to enumerate all bins up front.

```python
from dp_forge.sparse import ColumnGenerator

cg = ColumnGenerator(spec, initial_columns=10)
result = cg.solve()
print(f"Column gen objective: {result.obj_val:.6f}")
```

**Constructor:** `ColumnGenerator(spec: QuerySpec, initial_columns: int = 10)`

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `solve()` | `CEGISResult` | Run column generation to convergence |

#### `BendersDecomposer`

Benders decomposition that splits the mechanism design problem into a
master problem (mechanism structure) and subproblems (privacy verification
per adjacent pair).

```python
from dp_forge.sparse import BendersDecomposer

bd = BendersDecomposer(spec)
result = bd.solve()
print(f"Benders objective: {result.obj_val:.6f}")
```

**Constructor:** `BendersDecomposer(spec: QuerySpec)`

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `solve()` | `CEGISResult` | Run Benders decomposition to convergence |

#### `LagrangianRelaxer`

Lagrangian relaxation of the privacy constraints. Dualizes the DP
constraints, solving a sequence of unconstrained subproblems with
multiplier updates.

```python
from dp_forge.sparse import LagrangianRelaxer

lr = LagrangianRelaxer(spec, step_size=0.01)
result = lr.solve()
print(f"Lagrangian bound: {result.obj_val:.6f}")
```

**Constructor:** `LagrangianRelaxer(spec: QuerySpec, step_size: float = 0.01)`

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `solve()` | `CEGISResult` | Run Lagrangian relaxation with subgradient updates |

---

### Multi-Dimensional Mechanisms

Module `dp_forge.multidim` extends CEGIS to multi-dimensional query
workloads and budget allocation across queries.

#### `ProjectedCEGIS`

CEGIS variant for multi-dimensional queries that projects the
high-dimensional problem onto manageable subspaces.

```python
from dp_forge.multidim import ProjectedCEGIS
import numpy as np

workload = np.eye(5)
solver = ProjectedCEGIS(workload=workload, epsilon=1.0, delta=1e-5)
result = solver.solve()
print(f"Projected CEGIS objective: {result.obj_val:.6f}")
```

**Constructor:** `ProjectedCEGIS(workload: np.ndarray, epsilon: float, delta: float = 0.0)`

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `solve()` | `CEGISResult` | Synthesize the multi-dimensional mechanism |

#### `MultiDimMechanism`

Represents a synthesized multi-dimensional mechanism that answers a full
query workload.

```python
from dp_forge.multidim import MultiDimMechanism

mdm = MultiDimMechanism(strategy_matrix=result.mechanism, workload=workload)
noisy_answers = mdm.answer(true_data=np.array([10, 20, 30, 40, 50]))
print(f"Noisy answers: {noisy_answers}")
```

**Constructor:** `MultiDimMechanism(strategy_matrix: np.ndarray, workload: np.ndarray)`

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `answer(true_data)` | `np.ndarray` | Apply the mechanism to produce noisy answers |

#### `BudgetAllocator`

Optimally allocate a total privacy budget (ε, δ) across multiple queries
to minimize total workload error.

```python
from dp_forge.multidim import BudgetAllocator
import numpy as np

allocator = BudgetAllocator(total_epsilon=2.0, total_delta=1e-5)
workload = np.eye(5)
epsilons = allocator.allocate(workload)
print(f"Per-query ε: {epsilons}")
```

**Constructor:** `BudgetAllocator(total_epsilon: float, total_delta: float = 0.0)`

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `allocate(workload)` | `np.ndarray` | Return per-query epsilon allocations |

---

### Verification Backends

Module `dp_forge.verification` provides multiple verification strategies
with different precision/performance trade-offs.

#### `IntervalVerifier`

Verification using interval arithmetic for guaranteed numerical soundness.
Avoids floating-point errors that could produce false positives.

```python
from dp_forge.verification import IntervalVerifier

iv = IntervalVerifier()
result = iv.verify(mechanism.p_final, epsilon=1.0, delta=0.0)
print(f"Interval-verified: {result.valid}")
```

**Constructor:** `IntervalVerifier()`

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `verify(p, epsilon, delta)` | `VerifyResult` | Verify with interval arithmetic bounds |

#### `RationalVerifier`

Exact verification using rational (arbitrary-precision) arithmetic.
Eliminates all numerical error at the cost of increased computation time.

```python
from dp_forge.verification import RationalVerifier

rv = RationalVerifier()
result = rv.verify(mechanism.p_final, epsilon=1.0, delta=0.0)
print(f"Exact verification: {result.valid}")
```

**Constructor:** `RationalVerifier()`

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `verify(p, epsilon, delta)` | `VerifyResult` | Verify with exact rational arithmetic |

#### `CEGARVerifier`

Counter-Example Guided Abstraction Refinement verifier. Starts with a
coarse abstraction and refines only the regions where violations are found.

```python
from dp_forge.verification import CEGARVerifier

cv = CEGARVerifier(max_refinements=50)
result = cv.verify(mechanism.p_final, epsilon=1.0, delta=0.0)
print(f"CEGAR result: {result.valid}")
```

**Constructor:** `CEGARVerifier(max_refinements: int = 50)`

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `verify(p, epsilon, delta)` | `VerifyResult` | Verify via abstraction-refinement |

---

### SMT-Based Verification

Module `dp_forge.smt` encodes the DP property as an SMT formula for
sound and complete verification.

#### `SMTVerifier`

Encodes DP constraints as Satisfiability Modulo Theories (SMT) queries
and delegates to an SMT solver (e.g., Z3). Provides a formal proof of
privacy or a concrete counterexample.

```python
from dp_forge.smt import SMTVerifier

sv = SMTVerifier(solver="z3")
result = sv.verify(mechanism.p_final, epsilon=1.0, delta=0.0)
print(f"SMT verified: {result.valid}")
if not result.valid:
    print(f"Counterexample: {result.violation}")
```

**Constructor:** `SMTVerifier(solver: str = "z3")`

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `verify(p, epsilon, delta)` | `VerifyResult` | Formally verify or produce a counterexample |

---

### Robust Synthesis

Module `dp_forge.robust` handles synthesis under numerical uncertainty,
guaranteeing privacy even with floating-point implementation errors.

#### `RobustCEGISEngine`

CEGIS variant that inflates constraints by a robustness margin, ensuring
the synthesized mechanism remains private under implementation-level
rounding and truncation.

```python
from dp_forge.robust import RobustCEGISEngine

engine = RobustCEGISEngine(spec, robustness_margin=1e-6)
engine.initialize()
while not engine.converged:
    engine.step()
result = engine.extract_result()
print(f"Robust mechanism, obj: {result.obj_val:.6f}")
```

**Constructor:** `RobustCEGISEngine(spec: QuerySpec, robustness_margin: float = 1e-6)`

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `initialize()` | `None` | Set up the robust LP formulation |
| `step()` | `None` | Execute one robust CEGIS iteration |
| `extract_result()` | `CEGISResult` | Build the final robust result |

#### `IntervalMatrix`

Represents a probability matrix with interval-valued entries, tracking
worst-case bounds on each probability.

```python
from dp_forge.robust import IntervalMatrix
import numpy as np

p = np.array([[0.6, 0.4], [0.4, 0.6]])
im = IntervalMatrix(p, margin=1e-6)
print(im.lower)  # Lower bounds on each entry
print(im.upper)  # Upper bounds on each entry
```

**Constructor:** `IntervalMatrix(p: np.ndarray, margin: float = 1e-6)`

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `lower` | `np.ndarray` | Element-wise lower bounds |
| `upper` | `np.ndarray` | Element-wise upper bounds |

---

### Interpolation-Based Verification

Module `dp_forge.interpolation` uses Craig interpolants to strengthen
verification proofs and accelerate convergence.

#### `InterpolantEngine`

Generates interpolants between the mechanism constraints and privacy
violations to prune the search space during CEGIS.

```python
from dp_forge.interpolation import InterpolantEngine

ie = InterpolantEngine(spec)
interpolant = ie.compute(mechanism.p_final, epsilon=1.0, delta=0.0)
print(f"Interpolant computed: {interpolant is not None}")
```

**Constructor:** `InterpolantEngine(spec: QuerySpec)`

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `compute(p, epsilon, delta)` | `CraigInterpolant` | Compute a Craig interpolant for the current state |

#### `CraigInterpolant`

Represents a Craig interpolant: a formula that is implied by the mechanism
constraints and implies the privacy property. Used to generalize
counterexamples and speed up CEGIS convergence.

```python
from dp_forge.interpolation import CraigInterpolant

interpolant = ie.compute(mechanism.p_final, epsilon=1.0, delta=0.0)
strengthened = interpolant.strengthen(factor=2.0)
print(f"Interpolant dimension: {interpolant.dimension}")
```

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `dimension` | `int` | Dimensionality of the interpolant space |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `strengthen(factor)` | `CraigInterpolant` | Tighten the interpolant by the given factor |

---

### Certificates

Module `dp_forge.certificates` provides formal optimality and privacy
certificates based on LP duality.

#### `CertificateGenerator`

Generates machine-checkable certificates proving that a mechanism is both
private and optimal (or near-optimal).

```python
from dp_forge.certificates import CertificateGenerator

cg = CertificateGenerator()
cert = cg.generate(result, spec)
print(f"Certificate valid: {cert.verify()}")
```

**Constructor:** `CertificateGenerator()`

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `generate(result, spec)` | `CertificateChain` | Produce a full certificate chain for the synthesis result |

#### `LPOptimalityCertificate`

An LP duality certificate proving that the synthesized mechanism achieves
the optimal objective value. Contains the dual solution and complementary
slackness conditions.

```python
from dp_forge.certificates import LPOptimalityCertificate

cert = LPOptimalityCertificate(
    primal_obj=result.obj_val,
    dual_obj=result.obj_val,
    dual_solution=dual_vars,
)
print(f"Duality gap: {cert.duality_gap():.2e}")
print(f"Optimal: {cert.is_optimal()}")
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `duality_gap()` | `float` | Absolute gap between primal and dual objectives |
| `is_optimal()` | `bool` | True if the duality gap is within tolerance |

#### `CertificateChain`

A chain of certificates covering privacy verification, optimality, and
any intermediate proof steps. Designed to be serialized and independently
verified.

```python
from dp_forge.certificates import CertificateChain

chain = cg.generate(result, spec)
print(f"Chain length: {len(chain)}")
print(f"All valid: {chain.verify()}")
chain.save("certificate.json")
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `verify()` | `bool` | Verify all certificates in the chain |
| `save(path)` | `None` | Serialize the chain to a JSON file |
| `__len__()` | `int` | Number of certificates in the chain |

---

### Local Differential Privacy

Module `dp_forge.local_dp` provides mechanisms for the local DP model
where each user perturbs their own data before sending it to the server.

#### `RandomizedResponse`

Classic randomized response mechanism for binary or categorical data.

```python
from dp_forge.local_dp import RandomizedResponse

rr = RandomizedResponse(epsilon=2.0, num_categories=5)
perturbed = rr.perturb(true_value=3)
print(f"Perturbed: {perturbed}")

estimated_freq = rr.aggregate(responses=[2, 3, 3, 1, 0, 3, 2, 4])
print(f"Estimated frequencies: {estimated_freq}")
```

**Constructor:** `RandomizedResponse(epsilon: float, num_categories: int = 2)`

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `perturb(true_value)` | `int` | Randomize one user's response |
| `aggregate(responses)` | `np.ndarray` | Estimate true frequencies from collected responses |

---

### Streaming Mechanisms

Module `dp_forge.streaming` provides mechanisms for continual observation
and streaming data release.

#### `BinaryTreeMechanism`

Continual counting mechanism using a binary tree structure (Dwork et al., 2010;
Chan et al., 2011). Answers prefix-sum queries over a stream with polylogarithmic
error.

```python
from dp_forge.streaming import BinaryTreeMechanism

btm = BinaryTreeMechanism(epsilon=1.0, max_time=1024)
btm.update(value=1)
btm.update(value=0)
btm.update(value=1)
count = btm.query()  # Noisy prefix sum up to current time
print(f"Running count: {count:.2f}")
```

**Constructor:** `BinaryTreeMechanism(epsilon: float, max_time: int)`

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `update(value)` | `None` | Ingest one stream element |
| `query()` | `float` | Return noisy prefix sum at current time |

---

### Privacy Amplification

Module `dp_forge.amplification` implements privacy amplification results
for subsampling and shuffling.

#### `ShuffleAmplifier`

Computes the amplified privacy guarantee when a shuffler randomly permutes
user messages before they reach the analyzer (Erlingsson et al., 2019).

```python
from dp_forge.amplification import ShuffleAmplifier

amp = ShuffleAmplifier(local_epsilon=2.0, n_users=10000)
central_eps = amp.amplified_epsilon(delta=1e-6)
print(f"Amplified ε: {central_eps:.4f}")
```

**Constructor:** `ShuffleAmplifier(local_epsilon: float, n_users: int)`

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `amplified_epsilon(delta)` | `float` | Central ε after shuffle amplification |

#### `SubsamplingRDPAmplifier`

Privacy amplification by Poisson subsampling in the Rényi DP framework
(Mironov, 2017; Wang et al., 2019).

```python
from dp_forge.amplification import SubsamplingRDPAmplifier

amp = SubsamplingRDPAmplifier(base_rdp_func=lambda a: a / (2 * sigma**2), sample_rate=0.01)
amplified_rdp = amp.amplified_rdp(alpha=10)
print(f"Amplified RDP at α=10: {amplified_rdp:.6f}")
```

**Constructor:** `SubsamplingRDPAmplifier(base_rdp_func: Callable, sample_rate: float)`

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `amplified_rdp(alpha)` | `float` | Amplified Rényi divergence at order α |

---

### Workload Optimization

Module `dp_forge.workload_optimizer` optimizes noise strategies for
matrix-valued query workloads.

#### `HDMMOptimizer`

Implements the High-Dimensional Matrix Mechanism (McKenna et al., 2018)
for optimizing a strategy matrix that minimizes total workload error.

```python
from dp_forge.workload_optimizer import HDMMOptimizer
import numpy as np

workload = np.eye(10)
optimizer = HDMMOptimizer(workload=workload, epsilon=1.0, delta=1e-5)
strategy = optimizer.optimize()
print(f"Optimized MSE: {optimizer.evaluate(strategy):.6f}")
```

**Constructor:** `HDMMOptimizer(workload: np.ndarray, epsilon: float, delta: float)`

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `optimize()` | `np.ndarray` | Compute the optimal strategy matrix |
| `evaluate(strategy)` | `float` | Compute total MSE for a given strategy |

#### `KroneckerStrategy`

Exploits Kronecker product structure in multi-marginal workloads to
scale the matrix mechanism to high dimensions.

```python
from dp_forge.workload_optimizer import KroneckerStrategy
import numpy as np

marginals = [np.eye(10), np.eye(5)]
ks = KroneckerStrategy(marginals=marginals, epsilon=1.0, delta=1e-5)
strategy = ks.optimize()
print(f"Kronecker strategy MSE: {ks.evaluate(strategy):.6f}")
```

**Constructor:** `KroneckerStrategy(marginals: list[np.ndarray], epsilon: float, delta: float)`

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `optimize()` | `np.ndarray` | Compute the optimal Kronecker-structured strategy |
| `evaluate(strategy)` | `float` | Compute total MSE |

---

### Automatic Differentiation

Module `dp_forge.autodiff` provides gradient computation for mechanism
optimization, supporting both forward-mode (dual numbers) and
reverse-mode (computation tape) differentiation.

#### `ComputationTape`

Reverse-mode automatic differentiation tape that records operations for
backpropagation through mechanism parameters.

```python
from dp_forge.autodiff import ComputationTape

tape = ComputationTape()
x = tape.variable(2.0, name="x")
y = tape.variable(3.0, name="y")
z = x * y + tape.exp(x)
grads = tape.gradient(z, [x, y])
print(f"dz/dx = {grads[0]:.4f}, dz/dy = {grads[1]:.4f}")
```

**Constructor:** `ComputationTape()`

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `variable(value, name)` | `TracedVar` | Create a tracked variable on the tape |
| `exp(x)` | `TracedVar` | Exponentiation with gradient tracking |
| `log(x)` | `TracedVar` | Logarithm with gradient tracking |
| `gradient(output, inputs)` | `list[float]` | Compute gradients via backpropagation |

#### `DualNumber`

Forward-mode automatic differentiation via dual numbers. Efficiently
computes one directional derivative per forward pass.

```python
from dp_forge.autodiff import DualNumber

x = DualNumber(2.0, 1.0)  # value=2.0, derivative seed=1.0
y = x * x + DualNumber(3.0, 0.0) * x
print(f"f(2) = {y.real:.4f}")     # 4 + 6 = 10
print(f"f'(2) = {y.dual:.4f}")    # 2*2 + 3 = 7
```

**Constructor:** `DualNumber(real: float, dual: float = 0.0)`

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `real` | `float` | Function value |
| `dual` | `float` | Derivative value |

Supports arithmetic operators: `+`, `-`, `*`, `/`, `**`, and functions
via `DualNumber.exp()`, `DualNumber.log()`, `DualNumber.abs()`.

#### `MechanismOptimizer`

Gradient-based optimizer for mechanism parameters. Uses automatic
differentiation to compute gradients of the loss with respect to the
probability matrix, then projects back onto the feasible set.

```python
from dp_forge.autodiff import MechanismOptimizer

optimizer = MechanismOptimizer(spec, learning_rate=0.01, max_steps=1000)
result = optimizer.optimize(initial_p=result.mechanism)
print(f"Optimized objective: {result.obj_val:.6f}")
```

**Constructor:** `MechanismOptimizer(spec: QuerySpec, learning_rate: float = 0.01, max_steps: int = 1000)`

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `optimize(initial_p)` | `CEGISResult` | Gradient-descend from initial_p to a local optimum |

---

### Types and Enums

All core types are defined in `dp_forge.types`.

#### `LossFunction`

Enum specifying the loss metric to optimize.

```python
from dp_forge.types import LossFunction

LossFunction.L1     # Mean absolute error
LossFunction.L2     # Mean squared error (default)
LossFunction.LINF   # Worst-case (max) error
```

#### `MechanismFamily`

Enum specifying the structural family of the mechanism.

```python
from dp_forge.types import MechanismFamily

MechanismFamily.PIECEWISE_CONST  # Piecewise-constant density
```

#### `AdjacencyRelation`

Enum defining which database pairs are considered "adjacent" (neighbors)
for the DP guarantee.

```python
from dp_forge.types import AdjacencyRelation

AdjacencyRelation.CONSECUTIVE  # Hamming distance 1 between consecutive rows
AdjacencyRelation.COMPLETE     # All pairs are adjacent
```

#### `VerifyResult`

Result of a privacy verification check.

```python
from dp_forge.types import VerifyResult

vr: VerifyResult
vr.valid      # bool — True if mechanism satisfies (ε,δ)-DP
vr.violation  # Optional detail about the worst constraint violation
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `valid` | `bool` | Whether the mechanism passes verification |
| `violation` | `object or None` | Details of the worst violation, or `None` if valid |

#### `CEGISResult`

Result returned by all synthesis functions.

```python
from dp_forge.types import CEGISResult

r: CEGISResult
r.mechanism   # np.ndarray — the n × k probability matrix
r.obj_val     # float — optimal objective (worst-case expected loss)
r.iterations  # int — number of CEGIS iterations used
r.converged   # bool — whether CEGIS converged
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `mechanism` | `np.ndarray` | The synthesized probability matrix |
| `obj_val` | `float` | Optimal objective value |
| `iterations` | `int` | Number of iterations used |
| `converged` | `bool` | Whether the loop converged within `max_iter` |

#### `PrivacyBudget`

Lightweight container for an (ε, δ) privacy budget.

```python
from dp_forge.types import PrivacyBudget

budget = PrivacyBudget(epsilon=1.0, delta=1e-5)
print(budget.epsilon)  # 1.0
print(budget.delta)    # 1e-5
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `epsilon` | `float` | Privacy parameter ε |
| `delta` | `float` | Privacy parameter δ |

#### `OptimalityCertificate`

LP duality certificate attesting to the optimality of a synthesized mechanism.

```python
from dp_forge.types import OptimalityCertificate

cert: OptimalityCertificate
print(cert.primal_obj)   # Primal LP objective
print(cert.dual_obj)     # Dual LP objective
print(cert.duality_gap)  # |primal - dual|
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `primal_obj` | `float` | Primal LP objective value |
| `dual_obj` | `float` | Dual LP objective value |
| `duality_gap` | `float` | Absolute gap between primal and dual |

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
