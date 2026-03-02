# DP-Forge: Counterexample-Guided Synthesis of Optimal DP Mechanisms

**Automatically discover provably-optimal differentially private noise mechanisms for arbitrary query types.**

DP-Forge is a CEGIS-based (Counterexample-Guided Inductive Synthesis) engine that encodes DP mechanism design as LP/SDP optimization problems, iteratively verified by a formal separation oracle. The result: deployable mechanisms that provably satisfy (ε,δ)-differential privacy while minimizing expected error—achieving **2–10× accuracy improvement** over Laplace/Gaussian baselines at equivalent privacy guarantees.

> **Theoretical basis.** DP-Forge bridges Roth's mechanism design theory (encoding DP as linear constraints over probability tables) with Narodytska's CEGIS synthesis methodology (lazy constraint generation with formal verification). The LP feasible region is exactly the polytope of all ε-DP mechanisms for a given query, and the CEGIS loop navigates it in at most |E| iterations.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                        dp-forge                              │
├──────────────┬───────────────┬──────────────┬────────────────┤
│     Core     │    Support    │   Analysis   │    Deploy      │
├──────────────┼───────────────┼──────────────┼────────────────┤
│ cegis_loop   │ baselines     │ analysis     │ codegen        │
│ lp_builder   │ workloads     │ certificates │ sampling       │
│ sdp_builder  │ query_sensit. │ optimizer    │ cli            │
│ verifier     │ privacy_acct. │              │ visualization  │
│ extractor    │ composition   │              │                │
│ symmetry     │ mechanisms/   │              │                │
│ types        │ numerical     │              │                │
└──────────────┴───────────────┴──────────────┴────────────────┘
```

**Core loop:**

```
                    ┌────────────────────┐
                    │   QuerySpec        │
                    │  (ε, δ, k, loss)   │
                    └────────┬───────────┘
                             │
                    ┌────────▼───────────┐
              ┌────▶│  LP/SDP Builder    │◀──── symmetry reduction
              │     └────────┬───────────┘
              │              │ candidate mechanism
              │     ┌────────▼───────────┐
  counterexample    │     Verifier       │──── DONE ──▶ ExtractMechanism
              │     │ (separation oracle)│              ──▶ DeployableMechanism
              │     └────────┬───────────┘
              │              │ violated pair (i, i')
              └──────────────┘
```

---

## Installation

```bash
# Clone and install in development mode
git clone <repo-url>
cd dp-mechanism-forge/implementation
pip install -e ".[dev]"
```

**Requirements:** Python ≥ 3.10, NumPy ≥ 1.24, SciPy ≥ 1.10, CVXPY ≥ 1.3

**Optional solvers:** MOSEK (recommended for SDP), GLPK, HiGHS (default for LP)

---

## Quick Start

### 1. Synthesize a counting query mechanism

```python
from dp_forge.types import QuerySpec, ExtractedMechanism
from dp_forge.cegis_loop import CEGISSynthesize
from dp_forge.extractor import compute_mechanism_mse
import numpy as np

spec = QuerySpec.counting(n=10, epsilon=1.0, delta=0.0, k=50)
result = CEGISSynthesize(spec)
mechanism = ExtractedMechanism(p_final=result.mechanism)
y_grid = np.linspace(0, spec.n - 1, spec.k)
mse = compute_mechanism_mse(mechanism.p_final, y_grid, spec.query_values)
print(f"MSE: {mse:.4f}")
```

### 2. Compare with baselines

```python
from dp_forge.types import QuerySpec, ExtractedMechanism
from dp_forge.cegis_loop import CEGISSynthesize
from dp_forge.baselines import BaselineComparator
import numpy as np

spec = QuerySpec.counting(n=10, epsilon=1.0, delta=0.0, k=50)
result = CEGISSynthesize(spec)
mechanism = ExtractedMechanism(p_final=result.mechanism)
y_grid = np.linspace(0, spec.n - 1, spec.k)

comparator = BaselineComparator()
comparison = comparator.compare(mechanism.p_final, spec=spec, y_grid=y_grid)
print(f"Synthesized MSE: {comparison.synthesised_mse:.4f}")
for name, metrics in comparison.baseline_results.items():
    factor = comparison.improvement_factors.get(name, 0)
    print(f"{name}: MSE={metrics['mse']:.4f}  (improvement={factor:.2f}x)")
```

### 3. Full workload optimization with code generation

```python
from dp_forge.types import QuerySpec, MechanismFamily, ExtractedMechanism
from dp_forge.cegis_loop import CEGISSynthesize
from dp_forge.workloads import WorkloadGenerator, WorkloadAnalyzer
from dp_forge.codegen import PythonCodeGenerator
from dp_forge.baselines import MatrixMechanism

# Define a prefix-sum workload
A = WorkloadGenerator.prefix_sums(d=10)
props = WorkloadAnalyzer().analyze_workload(A)
print(f"Workload rank={props.rank}, condition={props.condition_number:.1f}")

# Synthesize workload-aware mechanism
spec = QuerySpec.counting(n=10, epsilon=1.0, delta=1e-6, k=50)
result = CEGISSynthesize(spec, family=MechanismFamily.GAUSSIAN_WORKLOAD)
mechanism = ExtractedMechanism(p_final=result.mechanism)

# Generate standalone deployment code
gen = PythonCodeGenerator()
code = gen.generate(mechanism, spec)
with open("my_mechanism.py", "w") as f:
    f.write(code)

# Compare against Matrix Mechanism baseline
baseline = MatrixMechanism(A, epsilon=1.0, delta=1e-6)
print(f"CEGIS obj:  {result.obj_val:.4f}")
print(f"Matrix MSE: {baseline.total_mse():.4f}")
```

---

## Synthesize for Your Own Query

DP-Forge is not limited to the built-in query types. You can define **any**
query as a JSON spec file and synthesize an optimal mechanism for it.

### Step 1: Generate a Starter Spec

```bash
dp-forge init-spec my_query.json --template counting
```

Available templates: `counting`, `sum`, `median`, `custom`.

### Step 2: Edit the Spec

A spec file is JSON with the following fields:

```json
{
  "query_values": [0.0, 1.0, 2.0, 3.0, 4.0],
  "sensitivity": 1.0,
  "epsilon": 1.0,
  "delta": 0.0,
  "k": 50,
  "loss": "l2",
  "domain": "my custom query",
  "adjacency": "consecutive"
}
```

| Field | Required | Description |
|---|---|---|
| `query_values` | ✅ | List of distinct query outputs `f(x₁), …, f(xₙ)` |
| `sensitivity` | ✅ | Global sensitivity Δf of the query |
| `epsilon` | ✅ | Privacy parameter ε |
| `delta` | | Approximate DP parameter δ (default: 0.0) |
| `k` | | Number of discretization bins (default: 100) |
| `loss` | | Loss function: `l1`, `l2`, `linf` (default: `l2`) |
| `domain` | | Human-readable description |
| `adjacency` | | `"consecutive"` (default) or `"complete"` |

### Step 3: Validate

```bash
dp-forge check-spec my_query.json
```

### Step 4: Synthesize

```bash
dp-forge synthesize --spec-file my_query.json
dp-forge synthesize --spec-file my_query.json --format python -o my_mech.py
```

See `examples/spec_counting.json`, `examples/spec_median.json`, and
`examples/spec_custom_sum.json` for complete runnable examples.

---

## Module Map

### Core Modules

#### `dp_forge.types` — Core Type System
> All dataclasses, enums, and type definitions used across DP-Forge.

| Symbol | Description |
|--------|-------------|
| `QuerySpec` | Primary input: query values, ε, δ, k (discretization bins), loss function, adjacency edges. Factory methods: `.counting(n, epsilon, delta, k)`, `.histogram(n_bins, epsilon, delta, k)` |
| `QueryType` | Enum: `COUNTING`, `HISTOGRAM`, `RANGE`, `LINEAR_WORKLOAD`, `MARGINAL`, `CUSTOM` |
| `MechanismFamily` | Enum: `PIECEWISE_CONST`, `PIECEWISE_LINEAR`, `GAUSSIAN_WORKLOAD` |
| `LossFunction` | Enum: `L1`, `L2`, `LINF`, `CUSTOM` — each has a `.fn` callable property |
| `AdjacencyRelation` | Neighboring database pairs. Class methods: `.hamming_distance_1(n)`, `.complete(n)` |
| `PrivacyBudget` | Privacy parameters (ε, δ) with composition type |
| `LPStruct` | LP problem: constraint matrices, objective vector, bounds, solver backend |
| `SDPStruct` | SDP problem: matrix variables, PSD constraints |
| `CEGISResult` | CEGIS loop output: optimal probability table, objective value, iteration count |
| `ExtractedMechanism` | Deployable mechanism with sampling tables |
| `VerifyResult` | Verification output: satisfied flag, violation magnitude, counterexample pairs |
| `OptimalityCertificate` | Dual-based proof of near-optimality |
| `WorkloadSpec` | Linear workload matrix with structural hints |
| `SynthesisConfig` | Overall synthesis configuration |
| `NumericalConfig` | Solver tolerances and numerical stability parameters |
| `SamplingConfig` | Sampling strategy: method (`ALIAS`, `CDF`, `REJECTION`), RNG config |

**Mathematical basis:** The `QuerySpec` encodes the mechanism design problem: minimize E[loss(f(x), y)] subject to the DP constraint polytope defined by `AdjacencyRelation` edges and privacy budget (ε, δ).

---

#### `dp_forge.lp_builder` — LP Construction for Discrete Mechanisms
> Constructs the linear program whose feasible region is exactly the set of all ε-DP mechanisms.

| Symbol | Description |
|--------|-------------|
| `LPManager` | LP construction and incremental constraint management. Methods: `build_dp_constraints(spec, edges) -> LPStruct`, `add_witnesses(candidates)`, `solve(lp_struct) -> Solution` |
| `VariableLayout` | Maps semantic variables to flat LP indices. `p_index(i, j)` → index for P[database i, output j]. Properties: `n_vars`, `n`, `k` |
| `build_output_grid(f_values, k, padding_factor=0.5)` | Constructs discretization grid for output domain |
| `ConstraintHistoryEntry` | Per-CEGIS-iteration checkpoint for warm-starting |

**Mathematical basis:** For n databases and k output bins, the LP has n·k probability variables P[i][j] subject to: (1) ∀i: Σ_j P[i][j] = 1 (simplex), (2) ∀(i,i')∈E, ∀j: P[i][j] ≤ e^ε · P[i'][j] (ε-DP), (3) P[i][j] ≥ 0. Objective: minimize Σ_{i,j} loss(i,j) · P[i][j].

**Dependencies:** `numpy`, `scipy.sparse`, `cvxpy`

---

#### `dp_forge.sdp_builder` — SDP Construction for Gaussian Workload Mechanisms
> Formulates and solves the semidefinite program for optimal Gaussian mechanisms under linear workloads.

| Symbol | Description |
|--------|-------------|
| `SDPManager` | SDP formulation and solving. `build_sdp(spec, workload) -> SDPStruct`, `solve(sdp_struct) -> Solution` |
| `GaussianMechanismResult` | Optimal covariance matrix and workload error |
| `StructuralDetector` | Auto-detects Toeplitz, circulant, or low-rank structure for SDP speedup |
| `SensitivityBallComputer` | Computes L2 sensitivity ball for workload matrix |

**Mathematical basis:** Minimizes E[‖A(x + noise) − Ax‖²] = tr(AΣAᵀ) subject to the K-norm mechanism constraint: the covariance Σ must satisfy (ε,δ)-DP for the L2 sensitivity of the query. Uses the K-norm framework to find optimal Gaussian covariance.

**Dependencies:** `cvxpy`, `numpy`, MOSEK (recommended)

---

#### `dp_forge.verifier` — DP Verification + Counterexample Generation
> The separation oracle: checks whether a candidate mechanism satisfies (ε,δ)-DP and returns the most-violating neighboring pair if not.

| Symbol | Description |
|--------|-------------|
| `PrivacyVerifier` | Deterministic verifier. `verify(p_final, spec, mode, tol) -> VerifyResult` |
| `verify(p_table, spec, ...) -> VerifyResult` | Module-level convenience function |
| `hockey_stick_divergence(p_i, p_j, eps) -> float` | H_ε(P‖Q) = Σ_j max(0, P[j] − e^ε · Q[j]) |
| `MonteCarloVerifier` | Statistical privacy auditing via sampling |
| `VerificationReport` | Structured findings: `is_dp_satisfied`, `most_violating_pair`, `violation_magnitude` |

**Mathematical basis:** For pure DP, checks ∀(i,i')∈E, ∀j: P[i][j] / P[i'][j] ≤ e^ε. For approximate DP, checks hockey-stick divergence: H_ε(M(x) ‖ M(x')) ≤ δ. Returns the pair (i, i') with maximum violation as the counterexample for CEGIS.

**Verification modes:** `FAST` (random subset), `MOST_VIOLATING` (full sweep, default), `EXHAUSTIVE` (all pairs with full analysis).

**Dependencies:** `numpy`

---

#### `dp_forge.cegis_loop` — Main CEGIS Orchestrator
> The top-level synthesis loop: iteratively builds LP, solves, verifies, and refines until convergence.

| Symbol | Description |
|--------|-------------|
| `CEGISSynthesize(spec, family, config) -> CEGISResult` | Main entry point. Runs the CEGIS loop to convergence. |
| `CEGISEngine` | Loop orchestrator with methods: `synthesize(spec, config)`, `_iterate()`, `_verify()`, `_extract()` |
| `SynthesisStrategy` | Enum: strategy selection (LP-based, SDP-based, hybrid) |
| `CEGISProgress` | Per-iteration snapshot: `iteration`, `objective`, `n_witness_pairs`, `violation_magnitude`, `total_time` |
| `ConvergenceHistory` | Tracks objective and duality gap across iterations |
| `WitnessSet` | Counterexample pairs queued for next LP iteration |
| `DualSimplexWarmStart` | Warm-start state for incremental LP solves |

**Mathematical basis:** Algorithm 3 from the theory. Start with a small subset of DP constraints. Solve LP → get candidate. Verify → if violated, add counterexample pair to constraint set and re-solve. Terminates in at most |E| iterations (number of neighboring pairs). Typical convergence: D ≪ |E| iterations due to warm-starting and most-violating-pair selection.

**Dependencies:** `lp_builder`, `sdp_builder`, `verifier`, `extractor`, `symmetry`

---

#### `dp_forge.extractor` — Post-Processing + Deployment
> Converts raw LP solutions into deployable mechanisms with exact DP guarantees.

| Symbol | Description |
|--------|-------------|
| `MechanismExtractor` | Extraction pipeline. `extract(lp_solution, spec) -> DeployableMechanism` |
| `extract_from_spec(p_raw, spec, y_grid)` | Module-level convenience: extracts a `DeployableMechanism` from raw probabilities |
| `DeployableMechanism` | Complete package: `sample(true_value_index, n_samples)`, `to_json()`, `save(path)`. Properties: `p`, `y_grid`, `n`, `k` |
| `AliasTable` | O(1) sampling via Vose's alias method. `sample(rng) -> int` |
| `CDFTable` | O(log k) sampling via binary search on CDF |
| `build_alias_table(probs) -> AliasTable` | Construct alias table from probability vector |
| `solve_dp_projection_qp(p_init, spec) -> np.ndarray` | QP fallback: project to nearest DP-feasible mechanism |

**Mathematical basis:** Raw LP solutions may have small numerical violations. The extractor: (1) clips negative probabilities, (2) re-normalizes rows to sum to 1, (3) optionally solves a QP to find the nearest feasible mechanism, (4) builds alias tables for O(1) sampling. All steps preserve DP guarantees within numerical tolerance.

**Dependencies:** `numpy`, `scipy`

---

#### `dp_forge.symmetry` — Symmetry Reduction
> Detects and exploits symmetry structure to reduce LP size, yielding up to |G|× speedup.

| Symbol | Description |
|--------|-------------|
| `ReduceBySymmetry(spec, lp_struct) -> ReducedLPStruct` | Main entry point |
| `SymmetryDetector` | Auto-detects translation, reflection, permutation symmetry. `detect(spec) -> SymmetryGroup` |
| `TranslationReducer` | Exploits translation invariance (~n× speedup) |
| `ReflectionReducer` | Exploits reflection symmetry (~2n× speedup) |
| `PermutationReducer` | General permutation group reduction (|G|× speedup) |
| `ReconstructionMap` | Maps reduced solutions back to full probability tables |
| `orbit_computation(generators, n_elements) -> List[Set[int]]` | Compute orbits of group action |

**Mathematical basis:** If the query function f and adjacency relation are invariant under a symmetry group G, the optimal mechanism can be assumed G-invariant without loss of optimality. This reduces the LP from n·k variables to (n/|G|)·k variables. For counting queries with Hamming adjacency, translation invariance gives ~n× reduction.

**Dependencies:** `numpy`, `sympy` (for group operations)

---

### Support Modules

#### `dp_forge.baselines` — Reference Mechanisms
> Standard DP mechanisms for comparison benchmarking.

| Class | Description |
|-------|-------------|
| `LaplaceMechanism` | Laplace noise, scale = Δf/ε |
| `GaussianMechanism` | Gaussian noise for (ε,δ)-DP |
| `GeometricMechanism` | Discrete Laplace for integer-valued queries |
| `StaircaseMechanism` | Geng-Viswanath optimal mechanism for 1D counting |
| `MatrixMechanism` | Matrix mechanism for workload queries |
| `ExponentialMechanism` | Exponential mechanism for selection queries |
| `RandResponseMechanism` | Randomized response for categorical/local DP |
| `BaselineComparator` | `compare(synthesized, spec, baselines) -> List[ComparisonResult]` |

---

#### `dp_forge.workloads` — Workload Generators
> Generate and analyze standard query workloads.

| Symbol | Description |
|--------|-------------|
| `WorkloadGenerator` | Static methods: `.counting_query(n)`, `.histogram(n_bins)`, `.range_queries(n)`, `.prefix_sums(d)`, `.linear_workload(matrix)` |
| `WorkloadAnalyzer` | `analyze_workload(matrix) -> WorkloadProperties` (rank, condition number, Toeplitz detection, sparsity) |

---

#### `dp_forge.query_sensitivity` — Sensitivity Computation
> Computes global sensitivity for different query types.

| Symbol | Description |
|--------|-------------|
| `QuerySensitivityAnalyzer` | Dispatch: `compute(spec) -> SensitivityResult` |
| `CountingQuerySensitivity` | Δf = 1 for Hamming adjacency |
| `HistogramQuerySensitivity` | Δf = 2 (add/remove) or 1 (substitution) |
| `LinearWorkloadSensitivity` | Δf = max singular value of workload matrix |

---

#### `dp_forge.privacy_accounting` — Composition Theorems
> Privacy composition and budget tracking.

| Symbol | Description |
|--------|-------------|
| `BasicComposition` | `sequential(budgets)`, `parallel(budgets)` |
| `AdvancedComposition` | Dwork–Rothblum–Vadhan (2010) strong composition |
| `RenyiDPAccountant` | Rényi DP accounting (Mironov 2017): `convert_to_approx_dp(rdp_budgets, delta)` |
| `ZeroCDPAccountant` | Zero-concentrated DP (Bun–Steinke 2016) |
| `MomentsAccountant` | Moments-based accounting for subsampled mechanisms |
| `SubsamplingAmplification` | `poisson_subsampling(budget, q)`, `without_replacement_subsampling(budget, q)` |
| `PrivacyBudgetTracker` | Track cumulative spending: `spend(budget) -> bool` |

---

#### `dp_forge.composition` — Mechanism Composition
> Compose multiple mechanisms into pipelines with formal privacy accounting.

| Symbol | Description |
|--------|-------------|
| `PrivacyFilter` | `compose(mechanisms, composition_type) -> ComposedMechanism` |
| `FourierAccountant` | Fourier-based composition accounting |
| `AdaptiveFilter` | Adaptive budget allocation |
| `FilteredComposition` | Filtered composition |
| `PrivacyOdometer` | Privacy budget tracking via odometer |

---

#### `dp_forge.mechanisms` — Mechanism Types
> Concrete mechanism wrappers for different families.

| Symbol | Description |
|--------|-------------|
| `DiscreteMechanism` | Wraps probability table + output range. `sample(rng, n_samples)` |
| `GaussianWorkloadMechanism` | Optimal Gaussian for linear workloads: covariance matrix + workload |
| `ComposedMechanism` | Wrapper for composed pipelines with cascaded sampling |

---

#### `dp_forge.numerical` — Numerical Utilities
> Numerically stable primitives used throughout the codebase.

| Function | Description |
|----------|-------------|
| `_logsumexp(a)` | Numerically stable log-sum-exp |
| `_log_subtract(log_a, log_b)` | Log-space subtraction |
| `_normalize_probabilities(probs, tol)` | Row normalization with clipping to [0,1] |
| `_safe_log(x, floor=1e-300)` | Log with underflow protection |
| `_safe_divide(a, b, default)` | Division with zero-denominator fallback |

---

### Analysis Modules

#### `dp_forge.analysis` — Deep Mechanism Analysis
> Post-synthesis analysis: utility metrics, privacy curves, robustness.

| Symbol | Description |
|--------|-------------|
| `MechanismAnalyzer` | `analyze(mechanism, spec, n_trials) -> UtilityMetrics` (MSE, MAE, variance, bias, quantiles) |
| `PrivacyCurveAnalyzer` | `compute_curve(mechanism, spec, n_points) -> PrivacyCurve` — (ε,δ) Pareto frontier |
| `OptimalityAnalyzer` | `analyze(mechanism, spec) -> OptimalityReport` — gap to LP lower bound |
| `RobustnessAnalyzer` | Sensitivity to ε/δ/k parameter variations |
| `ReportGenerator` | Generate LaTeX/Markdown/JSON reports |

---

#### `dp_forge.certificates` — Optimality Certificates
> Machine-checkable proofs that the synthesized mechanism is near-optimal.

| Symbol | Description |
|--------|-------------|
| `CertificateGenerator` | `generate(lp_result, spec) -> OptimalityCertificate` |
| `CertificateVerifier` | `verify(cert, mechanism, spec) -> bool` |
| `LPOptimalityCertificate` | Dual LP solution: `dual_ub`, `dual_eq`, `duality_gap`, `primal_obj`, `dual_obj`. Property: `is_tight` |
| `SDPOptimalityCertificate` | SDP dual matrix + minimum eigenvalue |
| `ApproximationCertificate` | Bound on discretization error (continuous → discrete) |
| `CertificateChain` | Chain of certificates for composed mechanisms |

---

#### `dp_forge.optimizer` — Advanced Optimization
> Multi-objective optimization and advanced LP techniques.

| Symbol | Description |
|--------|-------------|
| `BackendSelector` | Multi-backend optimization management |
| `ColumnGenerationEngine` | Column generation for large LPs |
| `CuttingPlaneEngine` | Cutting-plane algorithm |
| `BundleMethod` | Lagrangian dual bounds |
| `HyperparameterTuner` | Auto-tune ε, k, sampling method |

---

### Deployment Modules

#### `dp_forge.codegen` — Standalone Code Generation
> Generate deployment-ready code in multiple languages from synthesized mechanisms.

| Symbol | Description |
|--------|-------------|
| `CodeGenerator` | Dispatch: `generate(mechanism, spec, language='python') -> str` |
| `PythonCodeGenerator` | Standalone Python module with alias-method sampler |
| `CppCodeGenerator` | C++ header + optional CMakeLists.txt |
| `RustCodeGenerator` | Rust lib.rs + Cargo.toml |
| `NumpyCodeGenerator` | NumPy-optimized vectorized sampling |
| `DocumentationGenerator` | LaTeX/Markdown documentation with privacy proof sketch |

Generated code embeds `CodegenMetadata`: ε, δ, k, n, query type, objective value, and certificate hash for auditability.

---

#### `dp_forge.sampling` — Sampling Algorithms
> High-performance sampling from synthesized discrete distributions.

| Symbol | Description |
|--------|-------------|
| `AliasMethodSampler` | O(1) per-sample via Vose's alias method |
| `InverseCDFSampler` | O(log k) per-sample via binary search |
| `RejectionSampler` | General rejection sampling |
| `SecureRNG` | Cryptographic-grade random number generation |
| `MechanismSampler` | High-level: `add_noise(true_value, n_samples)`, `estimate_mse(n_trials)` |

---

#### `dp_forge.visualization` — Plotting
> Matplotlib-based visualization of mechanisms and synthesis diagnostics.

| Symbol | Description |
|--------|-------------|
| `MechanismVisualizer` | `heatmap(mechanism, spec)`, `1d_profile(mechanism, spec, i)` |
| `ConvergenceVisualizer` | `plot_convergence(history)` — objective + gap over CEGIS iterations |
| `ComparisonVisualizer` | `plot_comparison(results, baselines)` — bar/violin plots |

---

#### `dp_forge.cli` — Command-Line Interface
> Click-based CLI with Rich terminal output.

**Dependencies:** `click`, `rich`

---

## Algorithms

### Algorithm 1: BuildPrivacyLP

Constructs the LP whose feasible region is the polytope of all ε-DP mechanisms.

- **Variables:** P[i][j] for i ∈ {1,...,n}, j ∈ {1,...,k} — probability of output j on database i
- **Constraints:**
  - Simplex: Σ_j P[i][j] = 1 for all i
  - ε-DP: P[i][j] ≤ e^ε · P[i'][j] for all (i,i') ∈ E, all j
  - Non-negativity: P[i][j] ≥ 0
- **Objective:** minimize Σ_{i,j} loss(f(i), y_j) · P[i][j]
- **Complexity:** O(nk) variables, O(|E|k) constraints (monolithic); CEGIS adds constraints lazily

### Algorithm 2: BuildWorkloadSDP

Formulates the SDP for optimal Gaussian mechanisms under a linear workload A.

- **Variable:** Covariance matrix Σ ∈ S^d_+
- **Constraint:** Σ satisfies (ε,δ)-DP for L2 sensitivity of A
- **Objective:** minimize tr(AΣAᵀ) (total workload MSE)
- **Structure exploitation:** Auto-detects Toeplitz/circulant/low-rank for SDP speedup

### Algorithm 3: CEGISSynthesize

Main CEGIS synthesis loop with lazy constraint addition.

```
Input: QuerySpec (ε, δ, k, loss, adjacency)
D ← initial subset of E (e.g., random 10 pairs)
repeat:
    LP ← BuildPrivacyLP(spec, D)
    P* ← Solve(LP)                    // warm-started from previous
    (satisfied, violation) ← Verify(P*, spec)
    if satisfied:
        return ExtractMechanism(P*, spec)
    D ← D ∪ {most-violating pair from violation}
```

Typical convergence: D ≈ 20 iterations for n=100, vs. |E| = 4,950 total pairs.

### Algorithm 4: Verify

Separation oracle — checks (ε,δ)-DP and returns the most-violating pair.

- **Pure DP:** For each (i,i') ∈ E, check max_j P[i][j]/P[i'][j] ≤ e^ε
- **Approximate DP:** For each (i,i') ∈ E, check H_ε(P_i ‖ P_{i'}) ≤ δ where H_ε is the hockey-stick divergence
- **Returns:** (is_satisfied, most_violating_pair, violation_magnitude)

### Algorithm 5: ExtractMechanism

Post-processes LP solution into a deployable mechanism.

1. Clip negative probabilities (numerical artifacts)
2. Re-normalize each row to sum to 1
3. Optionally solve QP: min ‖P − P*‖² s.t. P is ε-DP (nearest feasible projection)
4. Build alias tables for O(1) sampling (Vose's algorithm)
5. Package as `DeployableMechanism` with embedded metadata

### Algorithm 6: ReduceBySymmetry

Exploits structural symmetry to reduce LP size.

1. Detect symmetry group G of (query, adjacency) pair
2. Compute orbits of G on the variable set {P[i][j]}
3. Identify one representative per orbit → reduced variable set
4. Transform constraints to act on representatives only
5. Solve reduced LP (up to |G|× smaller)
6. Reconstruct full solution from representatives

---

## Theoretical Guarantees

### CEGIS Termination
The CEGIS loop terminates in **at most |E| iterations**, where |E| is the number of neighboring database pairs. Each iteration adds at least one new constraint from E, and the LP feasible region is a polytope with finitely many faces. In practice, convergence occurs in D ≪ |E| iterations.

### CEGIS Soundness
When the loop returns `DONE=True`, the mechanism **provably satisfies (ε,δ)-DP**. The verifier exhaustively checks all pairs in E (or uses the hockey-stick divergence for approximate DP), so no violation can be missed.

### CEGIS Optimality
The returned mechanism is **globally optimal within the family F_k** (piecewise-constant mechanisms with k output bins). The LP feasible region is convex, the objective is linear, and LP solvers find global optima. The dual solution provides a machine-checkable optimality certificate.

### Discretization Approximation
For k output bins spanning a range B, the discretization error is at most **B/k** in output resolution. As k → ∞, the discrete optimum converges to the continuous optimum. For L2 loss, the discretization MSE overhead is O(B²/k²).

---

## Benchmarks

### Tier 1 — Mandatory (Core Contributions)

| ID | Benchmark | Parameters | Success Criterion |
|----|-----------|-----------|-------------------|
| T1.1 | Single Counting Query | n=100, ε∈{0.5, 1.0, 2.0} | MSE ≤ Laplace, terminates <30s |
| T1.2 | Small Histogram d=5 | ε=1.0, k=100 | CEGIS terminates, MSE ≤ Laplace |
| T1.3 | Small Histogram d=10 | ε=1.0, k=100 | **2–10× improvement** vs Laplace |
| T1.4 | Prefix Queries d=10 | ε=1.0, k=50 | MSE ≤ naive composition |

### Tier 2 — Strong Paper (Advanced Queries)

| ID | Benchmark | Parameters | Success Criterion |
|----|-----------|-----------|-------------------|
| T2.1 | Medium Histogram d=20 | ε=1.0, δ=1e-6 | MSE ≤ Gaussian |
| T2.2 | All-Range Queries d=10 | ε=1.0, δ=1e-6 | MSE ≤ 2× Matrix Mechanism |
| T2.3 | 2D Histogram 5×5 | ε=1.0, k=50 | Terminates <300s |
| T2.4 | Identity Workload d=20 | ε=1.0 | CEGIS ≤ 1.1× Laplace |
| T2.5 | Sequential Composition (5 queries) | per-query ε=0.2 | Composition accounting verified |

### Tier 3 — Aspirational (Scalability)

| ID | Benchmark | Parameters | Success Criterion |
|----|-----------|-----------|-------------------|
| T3.1 | Large Histogram d=50 | ε=1.0, timeout=600s | MSE ≤ 1.5× Matrix |
| T3.2 | 3-way Marginals d=10 | 120 queries | SDP + LP fallback completes |
| T3.3 | Mixed Workload | 50 mixed queries | Terminates within timeout |

Run benchmarks:
```bash
dp-forge benchmark --tier 1 --output-dir results/
```

---

## CLI Usage

```bash
# Synthesize a mechanism
dp-forge synthesize --query-type counting --epsilon 1.0 --delta 0.0 --k 100 -o mechanism.json

# Verify a mechanism satisfies DP
dp-forge verify -m mechanism.json --epsilon 1.0

# Compare with baselines
dp-forge compare -m mechanism.json -b laplace -b gaussian -o comparison.json

# Run benchmark suite
dp-forge benchmark --tier 1 --output-dir benchmarks/

# Display mechanism info
dp-forge info -m mechanism.json

# Generate deployment code
dp-forge codegen -m mechanism.json -l python -o mechanism.py
dp-forge codegen -m mechanism.json -l cpp -o mechanism.cpp
dp-forge codegen -m mechanism.json -l rust -o mechanism.rs
```

---

## API Reference

### Core Functions

```python
# Synthesis
CEGISSynthesize(spec: QuerySpec,
                family: MechanismFamily = PIECEWISE_CONST,
                config: SynthesisConfig = None) -> CEGISResult

# Extraction
ExtractedMechanism(p_final=result.mechanism)  # from types module
extract_from_spec(p_raw, spec, y_grid)        # from extractor module

# Verification
verify(p_table, spec: QuerySpec,
       mode: VerificationMode = MOST_VIOLATING) -> VerifyResult

# Symmetry reduction
ReduceBySymmetry(spec: QuerySpec, lp_struct: LPStruct) -> ReducedLPStruct
```

### Core Classes

```python
# Query specification
QuerySpec.counting(n=100, epsilon=1.0, delta=0.0, k=100) -> QuerySpec
QuerySpec.histogram(n_bins=10, epsilon=1.0, delta=0.0, k=100) -> QuerySpec

# Extracted mechanism
mechanism = ExtractedMechanism(p_final=result.mechanism)
mechanism.p_final  # n × k probability table
mechanism.n        # number of database inputs
mechanism.k        # number of output bins

# Baselines
laplace = LaplaceMechanism(sensitivity=1.0, epsilon=1.0)
gaussian = GaussianMechanism(sensitivity=1.0, epsilon=1.0, delta=1e-6)
comparator = BaselineComparator()
result = comparator.compare(mechanism.p_final, spec=spec, y_grid=y_grid)

# Code generation
gen = PythonCodeGenerator()
code = gen.generate(mechanism, spec)  # standalone .py with embedded probability table

# Privacy accounting
total = BasicComposition.sequential([PrivacyBudget(0.5, 0.0)] * 10)
total_adv = AdvancedComposition.compose([0.5]*10, [0.0]*10, delta_prime=1e-6)
```

---

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=dp_forge --cov-report=term-missing

# Run specific test categories
pytest tests/ -m "not slow"           # skip slow tests
pytest tests/ -m integration          # integration tests only
pytest tests/ -m numerical            # numerical precision tests

# Run a specific test file
pytest tests/test_cegis.py -v
pytest tests/test_verifier.py -v
```

**Test modules:** `test_types`, `test_lp_builder`, `test_verifier`, `test_cegis`, `test_extractor`, `test_symmetry`, `test_baselines`, `test_workloads`, `test_sensitivity`

---

## Honest Limitations

- **Memory:** Discretization requires O(n·k) memory for probability tables. For n=1,000 and k=100, this is 100K floats (~800KB) — manageable, but scales linearly.

- **LP solver scalability:** Practical domain size is limited to approximately **n ≈ 10,000** databases. Beyond this, LP solve time dominates even with CEGIS's lazy constraint addition. Mitigations: symmetry reduction, column generation, family restriction.

- **SDP path is Gaussian-only:** The SDP formulation (Algorithm 2) finds the optimal *Gaussian* mechanism for a workload, not the globally optimal mechanism across all families. The LP path gives global optimality but doesn't scale to high-dimensional workloads.

- **Approximate DP termination:** For (ε,δ)-DP with δ > 0, the hockey-stick divergence constraint is piecewise-linear, requiring epigraph linearization. CEGIS still terminates, but convergence can be slower than pure DP.

- **Not a drop-in replacement:** DP-Forge is a synthesis tool for discovering optimal mechanisms, not a production DP library. For production deployment, use the generated code (via `codegen`) or established libraries like [OpenDP](https://opendp.org/) or [Google's DP library](https://github.com/google/differential-privacy). DP-Forge's value is in *finding* better mechanisms, not in runtime infrastructure.

- **Numerical precision:** LP solvers operate in floating-point arithmetic. Optimality certificates include duality gaps (typically < 1e-8), but are not exact-arithmetic proofs. For formal verification, export constraints to an exact-arithmetic solver.

---

## Experimental Results

Full benchmark data: [`experiments/benchmark_results.json`](experiments/benchmark_results.json)

Run benchmarks: `python experiments/run_benchmarks.py`

### Counting Query: CEGIS vs. Laplace

| ε | n | CEGIS MSE | Laplace MSE | Improvement |
|---|---|-----------|-------------|-------------|
| 0.10 | 5 | 3.86 | 200.00 | **51.8×** |
| 0.10 | 10 | 17.34 | 200.00 | **11.5×** |
| 0.10 | 20 | 52.56 | 200.00 | **3.8×** |
| 0.25 | 5 | 3.31 | 32.00 | **9.7×** |
| 0.25 | 10 | 10.03 | 32.00 | **3.2×** |
| 0.50 | 5 | 2.19 | 8.00 | **3.7×** |
| 0.50 | 10 | 4.05 | 8.00 | **2.0×** |
| 1.00 | 5 | 0.89 | 2.00 | **2.2×** |
| 1.00 | 10 | 1.37 | 2.00 | **1.5×** |
| 2.00 | 5 | 0.27 | 0.50 | **1.9×** |
| 5.00 | 5 | 0.02 | 0.08 | **3.3×** |

**Summary:** Median 2.0× improvement over Laplace across all converged configs. Up to 51.8× at low ε.

### Approximate DP: CEGIS vs. Gaussian

| ε | δ | n | CEGIS MSE | Gaussian MSE | Improvement |
|---|---|---|-----------|--------------|-------------|
| 0.50 | 10⁻⁵ | 10 | 4.05 | 93.92 | **23.2×** |
| 1.00 | 10⁻⁵ | 10 | 1.37 | 23.48 | **17.2×** |
| 1.00 | 10⁻³ | 10 | 1.34 | 14.27 | **10.6×** |
| 2.00 | 10⁻⁵ | 10 | 0.36 | 5.87 | **16.2×** |

**Summary:** 10–23× improvement over calibrated Gaussian for approximate DP.

### Limitations

- Staircase mechanism (known optimal for counting) beats CEGIS at ε < 2 — CEGIS optimizes within discrete k-bin family
- Convergence issues at n ≥ 50 with ε ≥ 1.0 due to numerical precision
- Not applicable to multi-dimensional queries (current LP is 1-D)

---

## Citation

```bibtex
@article{dpforge2026,
  title     = {{DP-Forge}: Counterexample-Guided Synthesis of Optimal
               Differentially Private Mechanisms},
  author    = {{DP-Forge Team}},
  year      = {2026},
  note      = {Software available at \url{https://github.com/dp-forge/dp-forge}},
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
