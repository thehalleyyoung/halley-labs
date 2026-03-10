# CausalCert API Reference

> **CausalCert** — structural stress-testing of causal DAGs.
> Compute robustness radii, per-edge fragility scores, and certified causal-effect estimates.

## Table of Contents

- [Quick Start](#quick-start)
- [causalcert.types](#causalcerttypes) — enums, dataclasses, protocols
- [causalcert.dag](#causalcertdag) — DAG representation & d-separation
- [causalcert.ci\_testing](#causalcertci_testing) — conditional independence tests
- [causalcert.solver](#causalcertsolver) — robustness radius solvers
- [causalcert.fragility](#causalcertfragility) — per-edge fragility scoring
- [causalcert.estimation](#causalcertestimation) — causal effect estimation
- [causalcert.data](#causalcertdata) — data loading & synthetic generation
- [causalcert.pipeline](#causalcertpipeline) — end-to-end orchestration
- [causalcert.reporting](#causalcertreporting) — report generation
- [causalcert.benchmarks](#causalcertbenchmarks) — benchmark suite
- [causalcert.evaluation](#causalcertevaluation) — published DAGs
- [causalcert.treewidth](#causalcerttreewidth) — tree decomposition
- [causalcert.utils](#causalcertutils) — utilities
- [CLI Reference](#cli-reference)

---

## Quick Start

### Full audit in five lines

```python
from causalcert import CausalCertAnalysis
from causalcert.data import load_csv, load_dag

adj, names = load_dag("sachs.dot")
data = load_csv("sachs.csv")
result = CausalCertAnalysis(adj, data, treatment=0, outcome=5, node_names=names).run()

print(f"Robustness radius: [{result.radius_lower}, {result.radius_upper}]")
print(f"Most fragile edge: {result.most_fragile_edge}")
```

### Generate synthetic data and compute fragility

```python
from causalcert.dag import CausalDAG
from causalcert.data import generate_linear_gaussian
from causalcert.fragility import FragilityScorerImpl, rank_edges

dag = CausalDAG.from_edges(5, [(0,1), (1,2), (0,3), (3,4), (2,4)])
data, weights = generate_linear_gaussian(dag.adj, n=2000, seed=42)

scorer = FragilityScorerImpl(alpha=0.05)
scores = scorer.score(dag.adj, treatment=0, outcome=4, data=data)
for fs in rank_edges(scores)[:3]:
    print(f"  Edge {fs.edge}: score={fs.total_score:.3f}")
```

### Compute robustness radius with the ILP solver

```python
from causalcert.solver import ILPSolver
from causalcert.estimation import satisfies_backdoor

def predicate(adj, data, *, treatment, outcome):
    return satisfies_backdoor(adj, treatment, outcome, frozenset())

solver = ILPSolver(time_limit_s=60.0)
radius = solver.solve(dag.adj, predicate, data, treatment=0, outcome=4, max_k=10)
print(f"Radius: [{radius.lower_bound}, {radius.upper_bound}]")
```

---

## `causalcert.types`

### Enumerations

| Enum | Values | Description |
|------|--------|-------------|
| `EditType` | `ADD`, `DELETE`, `REVERSE` | Kind of structural edit |
| `FragilityChannel` | `D_SEPARATION`, `IDENTIFICATION`, `ESTIMATION` | Fragility channel |
| `SolverStrategy` | `ILP`, `LP_RELAXATION`, `FPT`, `CDCL`, `AUTO` | Solver selection |
| `VariableType` | `CONTINUOUS`, `ORDINAL`, `NOMINAL`, `BINARY` | Variable type |
| `CITestMethod` | `KERNEL`, `PARTIAL_CORRELATION`, `RANK`, `CRT`, `ENSEMBLE`, `HSIC`, `MUTUAL_INFO`, `CLASSIFIER`, `ADAPTIVE` | CI test method |
| `SolverStatus` | `OPTIMAL`, `FEASIBLE`, `INFEASIBLE`, `TIMEOUT`, `UNKNOWN` | Solver termination |
| `FormatType` | `DOT`, `DAGITTY`, `BIF`, `JSON`, `CSV`, `TETRAD`, `PCALG`, `GML` | DAG file format |

### Type Aliases

```python
NodeId          = int
AdjacencyMatrix = NDArray[np.int8]
NodeSet         = frozenset[NodeId]
EdgeTuple       = tuple[NodeId, NodeId]
```

### Dataclasses

All core dataclasses use `frozen=True, slots=True`.

```python
@dataclass(frozen=True, slots=True)
class StructuralEdit:
    edit_type: EditType
    source: NodeId
    target: NodeId
    edge -> EdgeTuple          # property
    cost -> int                # property, always 1

@dataclass(frozen=True, slots=True)
class CITestResult:
    x: NodeId
    y: NodeId
    conditioning_set: NodeSet
    statistic: float
    p_value: float
    method: CITestMethod
    reject: bool
    alpha: float = 0.05

@dataclass(frozen=True, slots=True)
class FragilityScore:
    edge: EdgeTuple
    total_score: float                             # Aggregate score in [0, 1]
    channel_scores: dict[FragilityChannel, float]  # Per-channel breakdown
    witness_ci: CITestResult | None = None

@dataclass(frozen=True, slots=True)
class RobustnessRadius:
    lower_bound: int
    upper_bound: int
    witness_edits: tuple[StructuralEdit, ...] = ()
    solver_strategy: SolverStrategy = SolverStrategy.AUTO
    solver_time_s: float = 0.0
    gap: float = 0.0
    certified: bool = False                        # True when lower == upper

@dataclass(frozen=True, slots=True)
class EstimationResult:
    ate: float          # Average treatment effect
    se: float           # Standard error
    ci_lower: float
    ci_upper: float
    adjustment_set: NodeSet
    method: str = "aipw"
    n_obs: int = 0

@dataclass(slots=True)
class AuditReport:
    treatment: NodeId
    outcome: NodeId
    n_nodes: int
    n_edges: int
    radius: RobustnessRadius
    fragility_ranking: list[FragilityScore] = []
    baseline_estimate: EstimationResult | None = None
    perturbed_estimates: list[EstimationResult] = []
    ci_results: list[CITestResult] = []
    metadata: dict[str, Any] = {}
```

### Configuration Dataclasses

```python
@dataclass(slots=True)
class PipelineConfig:
    treatment: NodeId = 0;  outcome: NodeId = 1;  alpha: float = 0.05
    ci_method: CITestMethod = CITestMethod.ENSEMBLE
    solver_strategy: SolverStrategy = SolverStrategy.AUTO
    max_k: int = 10;  n_folds: int = 5;  fdr_method: str = "by"
    n_jobs: int = 1;  seed: int = 42;  cache_dir: str | None = None
```

### Protocols

Runtime-checkable protocols for pluggable components:

| Protocol | Core method | Returns |
|----------|-------------|---------|
| `ConclusionPredicate` | `__call__(adj, data, *, treatment, outcome)` | `bool` |
| `CITester` | `test(x, y, conditioning_set, data)` | `CITestResult` |
| `CausalEstimator` | `estimate(adj, data, treatment, outcome, adjustment_set)` | `EstimationResult` |
| `RobustnessSolver` | `solve(adj, predicate, data, treatment, outcome, max_k)` | `RobustnessRadius` |
| `FragilityScorer` | `score(adj, treatment, outcome)` | `Sequence[FragilityScore]` |

---

## `causalcert.dag`

### `CausalDAG`

Primary DAG class backed by an `int8` adjacency matrix.

**Constructors:**

```python
CausalDAG(adj: AdjacencyMatrix | int, node_names=None, validate=True)
CausalDAG.empty(n, node_names=None) -> CausalDAG
CausalDAG.from_edges(n, edges: list[EdgeTuple], node_names=None, validate=True) -> CausalDAG
CausalDAG.from_adjacency_matrix(matrix, node_names=None, validate=True) -> CausalDAG
```

**Properties:** `n_nodes`, `n_edges`, `adj`, `node_names`, `density`, `max_in_degree`, `max_out_degree`

**Node & edge queries:**

| Method | Returns | Description |
|--------|---------|-------------|
| `node_name(v)` | `str` | Name for node index |
| `node_id(name)` | `int` | Index for node name |
| `has_edge(u, v)` | `bool` | Edge existence |
| `parents(v)` / `children(v)` | `NodeSet` | Direct parents / children |
| `ancestors(v)` / `descendants(v)` | `NodeSet` | Transitive ancestors / descendants |
| `roots()` / `leaves()` | `NodeSet` | Source / sink nodes |
| `edges()` | `Iterator[EdgeTuple]` | All directed edges |
| `topological_sort()` | `list[NodeId]` | Topological ordering |
| `has_directed_path(u, v)` | `bool` | Reachability |

**Mutations:** `add_edge(u, v)`, `delete_edge(u, v)`, `reverse_edge(u, v)`, `apply_edit(edit: StructuralEdit)`, `copy()`

**Subgraphs:** `subgraph(nodes)`, `ancestral_subgraph(targets)`

### `DSeparationOracle`

```python
DSeparationOracle(adj: AdjacencyMatrix)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `is_d_separated(x, y, conditioning)` | `bool` | Test d-separation |
| `is_d_connected(x, y, conditioning)` | `bool` | Test d-connection |
| `d_connected_set(x, conditioning)` | `NodeSet` | Nodes d-connected to x |
| `find_separating_set(x, y, max_size=None)` | `NodeSet \| None` | Minimal separating set |
| `all_d_separations(x, y, max_size=None)` | `list[NodeSet]` | All separating sets |
| `all_ci_implications(max_cond_size=None)` | `list[tuple]` | All CI implications |
| `active_paths(x, y, conditioning)` | `list[list[NodeId]]` | Active paths |
| `markov_blanket(v)` | `NodeSet` | Markov blanket |

### Format Conversions

```python
to_dot(adj, node_names=None) -> str
from_dot(dot_str: str) -> tuple[AdjacencyMatrix, list[str]]
to_json(adj, node_names=None, metadata=None) -> str
from_json(json_str: str) -> tuple[AdjacencyMatrix, list[str]]
```

### Moral Graph

```python
moral_graph(adj: AdjacencyMatrix) -> np.ndarray
moralize_ancestral(adj: AdjacencyMatrix, targets: NodeSet) -> np.ndarray
```

---

## `causalcert.ci_testing`

All testers implement `CITester` and return `CITestResult`.

### `PartialCorrelationTest`

Fast parametric CI test via partial correlations. Best for Gaussian data.

```python
PartialCorrelationTest(alpha=0.05, regularization=0.0, seed=42)
  .test(x, y, conditioning_set, data: pd.DataFrame) -> CITestResult
  .test_semi_partial(x, y, conditioning_set, data) -> CITestResult
```

### `KernelCITest`

Nonparametric kernel CI test with optional Nyström approximation.

```python
KernelCITest(
    alpha=0.05, bandwidth="median", nystrom_approximation=True,
    n_inducing_points=None, regularization=1e-5, seed=42,
)
  .test(x, y, conditioning_set, data: pd.DataFrame) -> CITestResult
```

### `CauchyCombinationTest`

Ensemble CI test combining p-values via the Cauchy method: `T = Σ wᵢ tan((0.5 − pᵢ)π)`.

```python
CauchyCombinationTest(
    methods=None,    # Default: [PARTIAL_CORRELATION, KERNEL]
    weights=None,    # Default: uniform
    alpha=0.05, seed=42,
)
  .test(x, y, conditioning_set, data: pd.DataFrame) -> CITestResult
```

---

## `causalcert.solver`

All solvers implement `RobustnessSolver` and return `RobustnessRadius`.

### `ILPSolver`

Exact ILP solver (Algorithm 4). Finds the minimum edit set to invalidate a causal conclusion.

```python
ILPSolver(time_limit_s=300.0, gap_tolerance=1e-4, threads=1, verbose=False)
  .set_warm_start(edge_vals: dict[tuple[int,int], float])
  .set_warm_start_from_adj(adj)
  .solve(adj, predicate, data, treatment, outcome, max_k=10, ci_results=None) -> RobustnessRadius
```

### `LPRelaxationSolver`

LP relaxation (Algorithm 5) for fast lower bounds with iterative cutting planes.

```python
LPRelaxationSolver(time_limit_s=60.0, verbose=False, max_cutting_rounds=20)
  .solve(adj, predicate, data, treatment, outcome, max_k=10, ci_results=None) -> RobustnessRadius
```

### `FPTSolver`

FPT-DP on nice tree decomposition (Algorithm 7). Runtime `O(3^tw · n)`.

```python
FPTSolver(time_limit_s=300.0, decomposition_method="min_fill", verbose=False)
  .solve(adj, predicate, data, treatment, outcome, max_k=10) -> RobustnessRadius
```

### `CDCLSolver`

Conflict-driven clause-learning search with restarts.

```python
CDCLSolver(time_limit_s=300.0, verbose=False, restart_strategy="luby")
  .solve(adj, predicate, data, treatment, outcome, max_k=10) -> RobustnessRadius
```

### `UnifiedSolver`

Auto-selects the best solver based on graph size and treewidth.

```python
UnifiedSolver(strategy: SolverStrategy = SolverStrategy.AUTO, **kwargs)
  .solve(adj, predicate, data, treatment, outcome, max_k=10) -> RobustnessRadius
```

**`AUTO` selection logic:**

| Condition | Solver chosen |
|-----------|---------------|
| n ≤ 20 | `ILPSolver` |
| treewidth ≤ 6 | `FPTSolver` |
| n ≤ 50 | `ILPSolver` with LP warm-start |
| otherwise | `LPRelaxationSolver` (bounds only) |

---

## `causalcert.fragility`

### `FragilityScorerImpl`

Computes per-edge fragility scores across three channels (Algorithm 3).

```python
FragilityScorerImpl(
    ci_results=None, alpha=0.05, include_absent=True,
    max_adj_set_size=4, restrict_to_ancestral=True,
    aggregation_method="max",  # "max" | "weighted" | "hierarchical"
    channel_weights: dict[FragilityChannel, float] | None = None,
)
  .score(adj, treatment, outcome, data=None) -> list[FragilityScore]
```

### Edge Ranking

```python
class EdgeSeverity(enum.Enum):
    CRITICAL  = "critical"    # score ≥ 0.7
    IMPORTANT = "important"   # score ∈ [0.4, 0.7)
    MODERATE  = "moderate"    # score ∈ [0.1, 0.4)
    COSMETIC  = "cosmetic"    # score < 0.1

classify_edge(score: float, thresholds=None) -> EdgeSeverity
rank_edges(scores, descending=True) -> list[FragilityScore]
top_k_fragile(scores, k=10) -> list[FragilityScore]
bottom_k_robust(scores, k=10) -> list[FragilityScore]
```

**Example — channel-level inspection:**

```python
from causalcert.fragility import FragilityScorerImpl, rank_edges, classify_edge
from causalcert.types import FragilityChannel

scores = FragilityScorerImpl(alpha=0.05).score(adj, 0, 4, data=data)
for fs in rank_edges(scores):
    sev = classify_edge(fs.total_score)
    d = fs.channel_scores.get(FragilityChannel.D_SEPARATION, 0.0)
    i = fs.channel_scores.get(FragilityChannel.IDENTIFICATION, 0.0)
    e = fs.channel_scores.get(FragilityChannel.ESTIMATION, 0.0)
    print(f"  {fs.edge} [{sev.value}] d-sep={d:.2f} id={i:.2f} est={e:.2f}")
```

---

## `causalcert.estimation`

### `AIPWEstimator`

Cross-fitted augmented inverse probability weighting.

```python
AIPWEstimator(
    n_folds=5, propensity_model="logistic", outcome_model="linear",
    seed=42, trim_lower=0.01, trim_upper=0.99,
    clip_bounds=(0.01, 0.99), ci_method="wald", n_bootstrap=1000,
)
  .estimate(adj, data, treatment, outcome, adjustment_set: NodeSet) -> EstimationResult
```

### Adjustment Set Operations

```python
satisfies_backdoor(adj, treatment, outcome, conditioning: NodeSet) -> bool
enumerate_adjustment_sets(adj, treatment, outcome, *, forbidden=None, required=None,
                          max_size=None, minimal=False) -> list[NodeSet]
find_optimal_adjustment_set(adj, treatment, outcome) -> NodeSet
find_minimal_adjustment_set(adj, treatment, outcome) -> NodeSet
```

---

## `causalcert.data`

### Loading

```python
load_csv(path, *, columns=None, dtype=None, na_values=None, nrows=None, delimiter=",") -> pd.DataFrame
load_parquet(path, *, columns=None) -> pd.DataFrame
load_auto(path, *, columns=None, dtype=None) -> pd.DataFrame   # dispatches by extension
```

### Synthetic Generation & DAG I/O

```python
generate_linear_gaussian(adj, n=1000, noise_scale=1.0, edge_weight_range=(0.5,1.5),
                         seed=42, intercepts=None) -> tuple[pd.DataFrame, np.ndarray]
generate_linear_gaussian_with_treatment(adj, treatment, outcome, n=1000, noise_scale=1.0,
                                        true_ate=1.0, seed=42) -> tuple[pd.DataFrame, np.ndarray, float]
load_dag(path) -> tuple[AdjacencyMatrix, list[str]]   # dispatches by .dot/.json/.csv/.bif
save_dag(adj, path, node_names=None, metadata=None) -> None
```

---

## `causalcert.pipeline`

### `CausalCertPipeline`

Full pipeline orchestrator (Algorithm 8). Steps: validate → CI testing → fragility → radius → estimation → report.

```python
class CausalCertPipeline:
    STEPS = ("validate", "ci_testing", "fragility", "radius", "estimation", "report")

    def __init__(self, config: PipelineRunConfig, progress_callback=None, checkpoint=None): ...
    def run(self, adj, data, predicate=None, node_names=None) -> AuditReport: ...
    def run_step(self, step: str, adj, data, predicate=None) -> Any: ...
```

**`PipelineRunConfig`** — see `PipelineConfig` in [Configuration Dataclasses](#configuration-dataclasses).

### `CausalCertAnalysis`

High-level convenience wrapper.

```python
CausalCertAnalysis(
    dag: AdjacencyMatrix | CausalDAG, data: pd.DataFrame,
    treatment: NodeId | str, outcome: NodeId | str,
    node_names=None, alpha=0.05, ci_method=CITestMethod.ENSEMBLE,
    solver_strategy=SolverStrategy.AUTO, seed=42,
)
  .run(predicate=None, max_k=10, n_jobs=1) -> AnalysisResult
```

**`AnalysisResult`:** Properties — `robustness_radius`, `radius_lower`, `radius_upper`, `is_certified`, `fragility_scores`, `most_fragile_edge`, `witness_edits`. Methods — `top_k_fragile(k=5)`, `fragility_by_channel(channel)`, `to_dict()`, `to_json(indent=2)`.

---

## `causalcert.reporting`

```python
to_json_report(report: AuditReport, node_names=None, indent=2) -> str
to_json_dict(report: AuditReport, node_names=None) -> dict
to_html_report(report, output_path=None, node_names=None, template_name="audit_report.html.j2") -> str
to_latex_tables(report, include_fragility=True, include_ci_results=False) -> str
```

---

## `causalcert.benchmarks`

```python
list_benchmarks() -> list[str]
get_benchmark(name: str) -> BenchmarkDAG          # fields: name, adj_matrix, node_names,
                                                   # treatment, outcome, expected_radius_range,
                                                   # expected_n_load_bearing, true_ate, source, tags
generate_benchmark_data(dag, n_samples=1000, seed=42) -> BenchmarkDataset
run_stress_suite(time_limit_s=300.0, max_n_nodes=50) -> dict[str, Any]
run_full_comparison(benchmarks=None) -> dict[str, dict[str, Any]]
compute_e_value(radius: int, n: int, treatment_effect: float, se: float) -> float
```

---

## `causalcert.evaluation`

17 real-world DAGs: `asia`, `sachs`, `alarm`, `insurance`, `child`, and more.

```python
list_published_dags() -> list[str]
get_published_dag(name: str) -> tuple[AdjacencyMatrix, list[str]]

# Synthetic DAG generators
random_dag_erdos_renyi(n, density=0.2, rng=None) -> AdjacencyMatrix
random_dag_scale_free(n, m_edges=2, rng=None) -> AdjacencyMatrix
chain_dag(n: int) -> AdjacencyMatrix
```

---

## `causalcert.treewidth`

```python
compute_treewidth_upper_bound(adj, method="min_fill") -> TreewidthBound  # fields: lower, upper, exact
compute_treewidth_bounds_from_adj(adj) -> TreewidthBound
compute_tree_decomposition(graph: nx.Graph, method="min_fill") -> TreeDecomposition
compute_tree_decomposition_from_adj(adj, method="min_fill") -> TreeDecomposition  # fields: bags, width, root_id
width_of_decomposition(td) -> int
```

Methods: `"min_fill"`, `"min_degree"`, `"min_width"`, `"mcs"`, `"best"` (tries all, keeps tightest).

---

## `causalcert.utils`

### `math_utils`

`nearest_positive_definite(M)`, `spectral_radius(M)`, `condition_number(M)`, `effective_rank(M, tol=1e-6)`, `safe_log(x, floor=1e-300)`, `safe_divide(a, b, fill=0.0)`, `log_sum_exp(a)`, `softmax(logits)`

### `graph_utils`

`bfs_shortest_path(adj, source, target)`, `all_simple_paths(adj, source, target, max_depth=None)`, `descendants(adj, node)`, `ancestors(adj, node)`, `topological_sort(adj)`, `is_dag(adj)`, `find_cycle(adj)`

### `stat_utils`

`cauchy_combine_pvalues(p_values, weights)`, `empirical_cdf(x, xs)`, `bootstrap_se(data, n_boot=1000)`, `percentile_ci(data, alpha=0.05)`, `fdr_correct_pvalues(pvalues, method="by")`

### `io_utils`

`save_pickle(obj, path)`, `load_pickle(path)`, `save_npz(data, path)`, `load_npz(path)`

### `parallel_utils`

```python
ParallelExecutor(n_jobs=1, backend="threading")
  .map(func, items) -> list
  .submit(func, *args, **kwargs) -> Any
```

---

## CLI Reference

Install the package and invoke via `causalcert`.

### `causalcert audit`

```bash
causalcert audit --dag sachs.dot --data sachs.csv --treatment 0 --outcome 5 \
  --output report.html --format html
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--dag` | PATH | required | DAG file (.dot, .json, .csv, .bif) |
| `--data` | PATH | required | Data file (.csv, .parquet) |
| `--treatment` | INT | required | Treatment node index |
| `--outcome` | INT | required | Outcome node index |
| `--output` | PATH | stdout | Output file |
| `--format` | STR | `json` | `html` \| `json` \| `latex` |
| `--alpha` | FLOAT | `0.05` | Significance level |
| `--solver` | STR | `auto` | Solver strategy |
| `--max-k` | INT | `10` | Maximum edit-set size |
| `--seed` | INT | `42` | Random seed |

### `causalcert fragility`

```bash
causalcert fragility --dag sachs.dot --data sachs.csv --top-k 10
```

### `causalcert validate`

```bash
causalcert validate --dag sachs.dot --data sachs.csv
```
