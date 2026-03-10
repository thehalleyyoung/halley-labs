# Architecture Document

## Bounded-Rational Usability Oracle — System Architecture

This document describes the internal architecture of the Bounded-Rational Usability
Oracle, including design principles, layer decomposition, data flow, module dependencies,
extension points, and performance characteristics.

---

## Table of Contents

- [Design Principles](#design-principles)
- [Layer Architecture](#layer-architecture)
- [Data Flow](#data-flow)
- [Module Dependency Graph](#module-dependency-graph)
- [Core Abstractions](#core-abstractions)
- [Pipeline Architecture](#pipeline-architecture)
- [Extension Points](#extension-points)
- [Performance Characteristics](#performance-characteristics)
- [Error Handling Strategy](#error-handling-strategy)
- [Caching Architecture](#caching-architecture)
- [Concurrency Model](#concurrency-model)

---

## Design Principles

### 1. CPU-Only Execution

The entire system runs on commodity laptop CPUs without GPU, cloud, or network
dependencies. This is not a constraint but a design principle:

- **Saliency** is computed from structural accessibility tree features, not pixel-level
  vision models
- **Monte Carlo** trajectory sampling is embarrassingly parallel across CPU cores
- **SMT solvers** (Z3) are CPU-native
- **Bisimulation** keeps abstract MDPs small (≤10K states)
- **No training phase** — all cognitive laws use published psychophysical parameters

### 2. Protocol-Driven Architecture

All public interfaces are defined as `Protocol` classes in `core/protocols.py`. This
enables:

- Independent testing of each component against its protocol
- Swappable implementations (e.g., different parsers, cost models, solvers)
- Clear contractual boundaries between modules

The 28 protocol interfaces include: `Parser`, `Aligner`, `CostModel`, `MDP`, `Policy`,
`PolicyComputer`, `BottleneckClassifier`, `RepairSynthesizer`, `Validator`,
`OutputFormatter`, `AccessibilityNode`, `AccessibilityTree`, `CacheProvider`,
`Serializable`, and more.

### 3. Consistency Oracle, Not Fidelity Oracle

The system does not predict absolute human performance. It detects *relative* cost
changes between UI versions. This weaker requirement enables:

- Formal error bounds via the Paired-Comparison Theorem
- Correlated error cancellation under shared bisimulation partitions
- Ordinal (rank-order) validation against published data instead of absolute calibration

### 4. Immutable Configuration

All configuration is expressed as frozen `dataclass` instances:

- `OracleConfig` — top-level configuration
- Per-subsystem configs: `ParserConfig`, `CognitiveConfig`, `MDPConfig`, etc.
- `FullPipelineConfig` — aggregates all configs with per-stage settings
- Loaded from YAML (`from_yaml()`), environment (`from_env()`), or defaults (`DEFAULT()`)

### 5. Compositional Error Propagation

Uncertainty is tracked through the entire pipeline via `Interval` arithmetic:

- Cognitive law predictions return `Interval` ranges
- Cost elements carry variance (σ²) alongside mean (μ)
- Error bounds are computed at each stage and composed formally
- Final verdicts include confidence intervals

---

## Layer Architecture

The system is organized into three layers, corresponding to the three theoretical
contributions plus a foundation and output layer.

### Layer 0: Foundation

**Packages:** `core/`, `utils/`, `interval/`

Provides the type system, protocol interfaces, configuration, error hierarchy, and
shared utilities that all other layers depend on.

```
core/
├── types.py        Point2D, BoundingBox, Interval, CostTuple, Trajectory
├── enums.py        AccessibilityRole, BottleneckType, RegressionVerdict, ...
├── protocols.py    28 Protocol interfaces
├── config.py       OracleConfig and sub-configs
├── errors.py       Exception hierarchy (UsabilityOracleError → ...)
└── constants.py    Psychophysical constants as Interval ranges

interval/
├── interval.py     Interval arithmetic: [low, high] with +, -, ×, ÷, ^, sqrt, exp, log
├── arithmetic.py   Extended operations
├── comparison.py   Interval comparison predicates
└── propagation.py  Uncertainty propagation through expressions

utils/
├── entropy.py      Shannon entropy, mutual information, KL divergence
├── graph.py        Topological sort, shortest path, connected components
├── math.py         log-sum-exp, stable softmax, safe division
├── sampling.py     Random sampling utilities
├── logging.py      Structured logging
├── timing.py       Performance timing decorators
└── serialization.py  JSON/YAML helpers
```

### Layer 1: Lean Profiler (Parsing and Cost Estimation)

**Packages:** `accessibility/`, `cognitive/`, `algebra/`, `alignment/`, `taskspec/`

Parses UI accessibility trees, computes cognitive costs using psychophysical laws, and
composes costs using the cost algebra. This layer implements the **minimal viable
system** (additive composition with Fitts' and Hick–Hyman laws).

```
accessibility/             Parses HTML/ARIA and JSON into AccessibilityTree
├── models.py              AccessibilityNode, AccessibilityTree, AccessibilityState
├── html_parser.py         HTMLAccessibilityParser.parse(html) → AccessibilityTree
├── json_parser.py         JSONAccessibilityParser.parse(json_str) → AccessibilityTree
├── normalizer.py          Normalize roles, names, coordinates; remove decorative
├── roles.py               RoleTaxonomy: ARIA role hierarchy and semantics
├── spatial.py             SpatialAnalyzer: layout, grouping, Fitts' metrics
└── validators.py          TreeValidator: structural validation

cognitive/                 Psychophysical law implementations
├── fitts.py               FittsLaw.predict(distance, width) → float
├── hick.py                HickHymanLaw.predict(n_alternatives) → float
├── working_memory.py      WorkingMemoryModel.predict_recall_probability(...)
├── visual_search.py       VisualSearchModel.predict_serial/parallel/guided(...)
├── motor.py               Motor execution models
├── perception.py          Perceptual channel models
├── calibration.py         Parameter calibration utilities
├── parameters.py          Published psychophysical parameter values
└── models.py              Shared cognitive data models

algebra/                   Compositional cognitive cost algebra
├── models.py              CostElement(μ, σ², κ, λ), CostExpression tree
├── sequential.py          SequentialComposer: ⊕ operator
├── parallel.py            ParallelComposer: ⊗ operator with MRT
├── context.py             ContextModulator: Δ operator
├── composer.py            TaskGraphComposer: DAG-level composition
├── soundness.py           SoundnessVerifier: algebraic axiom verification
└── optimizer.py           AlgebraicOptimizer: expression simplification

alignment/                 Semantic tree differencing
├── differ.py              SemanticDiffer: 3-pass alignment → AlignmentResult
├── exact_match.py         Exact matching by semantic hash
├── fuzzy_match.py         Fuzzy matching (role, name, spatial proximity)
├── classifier.py          Edit operation classification (ADD/REMOVE/MODIFY/REORDER)
├── cost_model.py          Edit operation cost assignment
├── models.py              AlignmentResult, NodeMapping, EditOperation
└── visualizer.py          Visual diff output

taskspec/                  Task specification DSL
├── dsl.py                 TaskDSLParser: YAML parsing and serialization
├── models.py              TaskStep, TaskFlow, TaskSpec, TaskGraph
├── templates.py           TaskTemplates: pre-built login, search, form specs
├── inference.py           Automatic task inference from UI structure
├── recorder.py            Task recording from interaction traces
└── validator.py           Task specification validation
```

### Layer 2: Bounded-Rational Theory (Bisimulation and Policy)

**Packages:** `mdp/`, `bisimulation/`, `policy/`

Constructs task-state MDPs, performs bounded-rational bisimulation for state-space
reduction, and computes softmax/free-energy policies. This layer implements
**Contribution 1** (bisimulation) and the policy computation required for
**Contribution 3** (bottleneck classification).

```
mdp/                       Markov Decision Process construction
├── models.py              State, Action, Transition, MDP, MDPStatistics
├── builder.py             MDPBuilder.build(tree, task_spec) → MDP
├── solver.py              Value iteration, policy iteration
├── trajectory.py          Trajectory sampling and statistics
├── features.py            State feature extraction
├── reward.py              Reward function definitions
└── visualization.py       MDP graph visualization

bisimulation/              Bounded-rational state abstraction
├── cognitive_distance.py  CognitiveDistanceComputer: d_cog metric
├── partition.py           PartitionRefinement: ε-bisimulation computation
├── quotient.py            QuotientMDPBuilder: abstract MDP construction
├── clustering.py          State clustering heuristics
├── models.py              Partition, Block data models
└── validators.py          Partition validation

policy/                    Bounded-rational policy computation
├── softmax.py             SoftmaxPolicy: π_β(a|s) ∝ p₀·exp(−β·Q)
├── free_energy.py         FreeEnergyComputer: F(π) = E[c] + (1/β)·D_KL
├── monte_carlo.py         MonteCarloEstimator: first-visit / every-visit MC
├── value_iteration.py     Exact value iteration
├── optimal.py             Fully rational baseline policy
└── models.py              Policy data models
```

### Layer 3: Scale (Comparison, Classification, Repair)

**Packages:** `comparison/`, `bottleneck/`, `repair/`, `fragility/`, `evaluation/`

Performs paired comparison with hypothesis testing, classifies bottlenecks using the
information-theoretic taxonomy, and (optionally) synthesizes repairs via Z3.

```
comparison/                Regression detection
├── paired.py              PairedComparator: shared-partition comparison
├── hypothesis.py          RegressionTester: Welch's t, Mann-Whitney, bootstrap
├── error_bounds.py        ErrorBoundComputer: Hoeffding bounds, sampling error
├── union_mdp.py           Union MDP for shared abstraction
├── parameter_free.py      Parameter-free comparison methods
├── reporter.py            Regression report generation
└── models.py              Comparison result models

bottleneck/                Cognitive bottleneck taxonomy
├── classifier.py          BottleneckClassifier: 5-type classification
├── signatures.py          SignatureComputer: information-theoretic signatures
├── models.py              BottleneckSignature, BottleneckResult, BottleneckReport
├── perceptual.py          H(S|display) > τ_p detection
├── choice.py              log|A| − I(S;A) > τ_c detection
├── motor.py               Motor difficulty detection
├── memory.py              I(S_t; S_{t−k}) < τ_μ detection
├── interference.py        I(A^(1); A^(2) | S) > τ_ι detection
└── repair_map.py          Bottleneck → repair strategy mapping

repair/                    SMT-backed repair synthesis (stretch goal)
├── synthesizer.py         RepairSynthesizer: Z3-based constraint solving
├── mutations.py           MutationOperator: resize, regroup, simplify, landmark
├── constraints.py         Z3 constraint encoding
├── strategies.py          Strategy selection by bottleneck type
├── validator.py           Repair validation
└── models.py              Repair data models

fragility/                 Robustness analysis
├── analyzer.py            FragilityAnalyzer: cost curves, discontinuities
├── sensitivity.py         Parameter sensitivity analysis
├── cliff.py               Cliff detection in cost landscapes
├── adversarial.py         Adversarial perturbation testing
├── inclusive.py            Inclusive design across population β-distributions
└── models.py              Fragility data models

evaluation/                Validation framework
├── ordinal.py             OrdinalValidator: Spearman ρ, Kendall τ
├── ablation.py            Ablation study framework
├── baselines.py           Baseline comparators (axe-core, KLM, scalar)
├── regression.py          Regression detection metrics
└── reporting.py           Evaluation report generation
```

### Output & Orchestration Layer

**Packages:** `pipeline/`, `output/`, `cli/`, `benchmarks/`

Orchestrates the full pipeline, formats results, and provides CLI and CI integration.

```
pipeline/                  Orchestration
├── runner.py              PipelineRunner.run(config, source_a, source_b, task)
├── stages.py              10 StageExecutors + StageRegistry
├── config.py              FullPipelineConfig
├── cache.py               ResultCache: content-addressed, TTL, hit-rate tracking
└── parallel.py            ParallelExecutor: thread/process pool

output/                    Result formatting
├── console.py             ConsoleFormatter: Rich terminal output
├── json_output.py         JSONFormatter: structured JSON
├── sarif.py               SARIFFormatter: SARIF 2.1.0
├── html_report.py         HTML report generation
└── models.py              Output data models

cli/                       Command-line interface
├── main.py                Click CLI: diff, analyze, benchmark, validate, init
├── formatters.py          CLI formatting utilities
└── github_action.py       GitHubActionIntegration: PR comments, annotations

benchmarks/                Benchmarking
├── suite.py               BenchmarkSuite.run() → BenchmarkReport
├── generators.py          Synthetic UI pair generation
├── metrics.py             Precision, recall, F1, accuracy metrics
├── mutations.py           Controlled mutation injection
└── datasets.py            Benchmark dataset management
```

---

## Data Flow

### Full Pipeline Execution

```
1. PARSE       ─── HTML/JSON → AccessibilityTree (×2 for diff)
                   TaskDSL YAML → TaskSpec
2. NORMALIZE   ─── AccessibilityTree → normalized AccessibilityTree
3. VALIDATE    ─── AccessibilityTree → ValidationResult
4. ALIGN       ─── (TreeA, TreeB) → AlignmentResult (3-pass semantic diff)
5. BUILD MDP   ─── (Tree, TaskSpec) → MDP (×2)
6. COST        ─── MDP → weighted MDP (cognitive costs on transitions)
7. BISIMULATE  ─── MDP → abstract MDP (bounded-rational ε-bisimulation)
8. POLICY      ─── abstract MDP → Policy (softmax / free-energy optimal)
9. SAMPLE      ─── (MDP, Policy) → trajectory distributions (Monte Carlo)
10. COMPARE    ─── (trajs_A, trajs_B) → HypothesisResult, RegressionVerdict
11. BOTTLENECK ─── (MDP, Policy, trajs) → BottleneckReport
12. REPAIR     ─── (bottlenecks, constraints) → RepairCandidates [optional]
13. FORMAT     ─── PipelineResult → console / JSON / SARIF / HTML
```

### Stage Dependencies

```
PARSE_A ──┐
           ├──→ ALIGN ──→ BUILD_MDP_A ──→ BISIMULATE_A ──→ POLICY_A ──┐
PARSE_B ──┘                                                             │
           ├──→ ALIGN ──→ BUILD_MDP_B ──→ BISIMULATE_B ──→ POLICY_B ──┤
PARSE_TASK─┘                                                            │
                                                                        ▼
                                                                    COMPARE
                                                                        │
                                                                        ▼
                                                                   BOTTLENECK
                                                                        │
                                                                        ▼
                                                                     REPAIR
                                                                        │
                                                                        ▼
                                                                     OUTPUT
```

### Key Data Types at Each Stage

| Stage | Input Type | Output Type |
|-------|-----------|-------------|
| Parse | `str` (HTML/JSON) | `AccessibilityTree` |
| Align | `(AccessibilityTree, AccessibilityTree)` | `AlignmentResult` |
| Build MDP | `(AccessibilityTree, TaskSpec)` | `MDP` |
| Bisimulate | `MDP` | `MDP` (abstract) |
| Policy | `MDP` | `Policy` |
| Compare | `(Policy, Policy, MDP, MDP)` | `HypothesisResult` |
| Bottleneck | `(MDP, Policy, TrajectoryStats)` | `BottleneckReport` |
| Repair | `(list[BottleneckResult], Constraints)` | `RepairResult` |
| Output | `PipelineResult` | `str` |

---

## Module Dependency Graph

Arrows indicate "depends on" relationships. Circular dependencies are prohibited.

```
                         ┌─────────┐
                         │  core/  │ ◄──── everything depends on core
                         └────┬────┘
                              │
               ┌──────────────┼──────────────┐
               │              │              │
          ┌────▼────┐   ┌────▼────┐   ┌─────▼─────┐
          │interval/ │   │ utils/  │   │ taskspec/ │
          └────┬─────┘   └────┬────┘   └─────┬─────┘
               │              │              │
     ┌─────────┴──────┐      │              │
     │                │      │              │
┌────▼──────┐   ┌─────▼──────▼──┐     ┌────▼───────┐
│cognitive/ │   │accessibility/ │     │ alignment/ │
└─────┬─────┘   └───────┬───────┘     └─────┬──────┘
      │                 │                    │
      │         ┌───────┴────────┐           │
      │         │                │           │
┌─────▼─────┐   │          ┌─────▼───────────▼──┐
│  algebra/ │   │          │       mdp/         │
└─────┬─────┘   │          └─────────┬──────────┘
      │         │                    │
      │   ┌─────▼──────┐    ┌───────▼───────┐
      │   │bisimulation/│    │    policy/    │
      │   └─────┬───────┘    └───────┬───────┘
      │         │                    │
      │   ┌─────▼────────────────────▼──┐
      │   │        comparison/          │
      │   └─────────────┬───────────────┘
      │                 │
      │   ┌─────────────▼───────────────┐
      └──►│        bottleneck/          │
          └─────────────┬───────────────┘
                        │
          ┌─────────────▼───────────────┐
          │          repair/            │
          └─────────────┬───────────────┘
                        │
    ┌───────┬───────────▼─────┬──────────┐
    │       │                 │          │
┌───▼──┐ ┌─▼──────┐ ┌────────▼──┐ ┌────▼──────┐
│ cli/ │ │output/ │ │ pipeline/ │ │evaluation/│
└──────┘ └────────┘ └───────────┘ └───────────┘
```

### Dependency Rules

1. **`core/`** depends on nothing (except Python stdlib and `numpy`)
2. **`interval/`** and **`utils/`** depend only on `core/`
3. **`cognitive/`** depends on `core/`, `interval/`
4. **`accessibility/`** depends on `core/`, `utils/`
5. **`algebra/`** depends on `core/`, `interval/`, `cognitive/`
6. **`mdp/`** depends on `core/`, `accessibility/`, `cognitive/`, `algebra/`
7. **`bisimulation/`** depends on `core/`, `mdp/`, `policy/`
8. **`policy/`** depends on `core/`, `mdp/`
9. **`comparison/`** depends on `core/`, `mdp/`, `policy/`, `bisimulation/`
10. **`bottleneck/`** depends on `core/`, `mdp/`, `policy/`, `algebra/`
11. **`repair/`** depends on `core/`, `bottleneck/`, `accessibility/`
12. **`pipeline/`** depends on all analysis packages
13. **`cli/`** and **`output/`** depend on `pipeline/` and `core/`

---

## Core Abstractions

### CostElement — The 4-Tuple

Every cognitive operation is represented as a `CostElement(μ, σ², κ, λ)`:

| Field | Meaning | Range |
|-------|---------|-------|
| `μ` (mu) | Expected time cost | [0, ∞) seconds |
| `σ²` (sigma_sq) | Cost variance | [0, ∞) seconds² |
| `κ` (kappa) | Capacity utilization | [0, 1] fraction |
| `λ` (lam) | Interference susceptibility | [0, 1] fraction |

### CostExpression — The Expression Tree

Cost composition builds an expression tree:

```
       Sequential(⊕)
       /            \
   Leaf(click)    Parallel(⊗)
                  /          \
            Leaf(read)    ContextMod(Δ)
                              |
                          Leaf(search)
```

The `AlgebraicOptimizer` simplifies these trees, and the `SoundnessVerifier` checks
algebraic axioms (monotonicity, triangle inequality, commutativity of ⊗).

### Partition — State Abstraction

A `Partition` maps concrete MDP states to abstract equivalence classes (blocks). The
`PartitionRefinement` algorithm iteratively splits blocks until the ε-bisimulation
condition is satisfied:

```
Concrete states: {s1, s2, s3, s4, s5, s6, s7, s8}
                          ↓ refine(ε=0.005)
Abstract blocks:  {s1,s2,s3}  {s4,s5}  {s6,s7,s8}
                     B1          B2        B3
```

---

## Pipeline Architecture

### StageExecutor Pattern

Each pipeline stage is a `BaseStageExecutor` subclass with:

- **Timing**: automatic execution timing
- **Retry**: configurable retry with exponential backoff
- **Error handling**: stage-specific error wrapping
- **Caching**: content-addressed result caching

```python
class ParseStageExecutor(BaseStageExecutor):
    stage = PipelineStage.PARSE

    def _run(self, **kwargs) -> Any:
        source = kwargs["source"]
        parser = HTMLAccessibilityParser(...)
        return parser.parse(source)
```

### StageRegistry

The `StageRegistry` manages stage registration and lookup:

```python
registry = StageRegistry.default(stage_configs)
executor = registry.get(PipelineStage.PARSE)
result = executor.execute(source=html_content)
```

### PipelineRunner

The `PipelineRunner` orchestrates the full pipeline:

```python
runner = PipelineRunner(config=oracle_config)
result = runner.run(
    config=full_config,
    source_a="<html>...",
    source_b="<html>...",
    task_spec=task_spec,
)
# result.verdict: RegressionVerdict
# result.cost_delta: float
# result.bottleneck_report: BottleneckReport
# result.timing: dict[PipelineStage, float]
```

---

## Extension Points

### 1. Custom Parsers

Implement the `Parser` protocol:

```python
class MyCustomParser:
    def parse(self, source: str) -> AccessibilityTree:
        # Parse your custom format
        ...
```

### 2. Custom Cognitive Laws

Implement the `CostModel` protocol:

```python
class MyLaw:
    def compute(self, context: Any) -> CostTuple:
        # Return cognitive cost for this operation
        ...
```

### 3. Custom Output Formats

Implement the `OutputFormatter` protocol:

```python
class MyFormatter:
    def format(self, data: Any, options: dict) -> str:
        # Serialize pipeline results
        ...
```

### 4. Custom Pipeline Stages

Extend `BaseStageExecutor`:

```python
class MyStageExecutor(BaseStageExecutor):
    stage = PipelineStage.MY_STAGE

    def _run(self, **kwargs) -> Any:
        ...
```

### 5. Custom Bottleneck Detectors

Add new bottleneck types by extending the signature classification:

```python
# In bottleneck/signatures.py, add new threshold checks
# In bottleneck/classifier.py, register new detector
# In bottleneck/repair_map.py, map to repair strategies
```

---

## Performance Characteristics

### Time Complexity by Stage

| Stage | Complexity | Typical Time |
|-------|-----------|-------------|
| Parse | O(n) where n = DOM nodes | <1s for 2000 elements |
| Align | O(n²) worst case, O(n log n) typical | <2s |
| Build MDP | O(n × \|A\|) where \|A\| = action types | <5s |
| Bisimulate | O(\|S\|² × \|A\| × k) where k = refinement iterations | <10s |
| Policy (value iteration) | O(\|Ŝ\|² × \|A\|) on abstract MDP | <5s |
| Monte Carlo sampling | O(T × H) where T = trajectories, H = horizon | <5s |
| Comparison | O(T) for hypothesis test | <1s |
| Bottleneck classification | O(\|Ŝ\| × \|A\|) | <2s |
| Repair synthesis | O(exponential) but Z3-bounded by timeout | ≤60s |
| **Total (typical)** | | **<30s** |

### Space Complexity

| Component | Memory |
|-----------|--------|
| AccessibilityTree (2000 nodes) | ~5 MB |
| Raw MDP (50K states) | ~50 MB |
| Abstract MDP (10K states) | ~10 MB |
| Trajectory samples (1000 × 500 steps) | ~20 MB |
| **Total pipeline peak** | **~100 MB** |

### Scaling Targets

| UI Size | Elements | Abstract States | Wall Clock |
|---------|----------|----------------|------------|
| Small | ≤100 | ≤500 | <10s |
| Medium | ≤500 | ≤5,000 | <60s |
| Large | ≤2,000 | ≤10,000 | <300s |
| XL | ≤5,000 | ≤20,000 | <600s |

### Parallelism

The `ParallelExecutor` exploits CPU parallelism at two levels:

1. **Trajectory sampling**: independent trajectories sampled in parallel across
   `max_workers` threads
2. **Version analysis**: before/after UI analysis can run concurrently for
   independent stages (Parse, Build MDP)

---

## Error Handling Strategy

### Exception Hierarchy

```
UsabilityOracleError
├── ParseError
│   ├── InvalidAccessibilityTreeError
│   └── MalformedHTMLError
├── AlignmentError
│   └── IncompatibleTreesError
├── CostModelError
│   ├── InvalidParameterError
│   └── ConvergenceError
├── MDPError
│   ├── StateSpaceExplosionError
│   └── UnreachableStateError
├── PolicyError
│   └── NumericalInstabilityError
├── BisimulationError
│   └── PartitionError
├── BottleneckError
│   └── ClassificationError
├── ComparisonError
├── RepairError
├── ConfigError
├── PipelineError
└── CacheError
```

### Error Propagation

1. Each stage catches its specific errors and wraps them in `StageResult.errors`
2. `fail_fast=True` (default) aborts the pipeline on first error
3. `fail_fast=False` continues through remaining stages, collecting errors
4. The final `PipelineResult` aggregates all stage results and errors

---

## Caching Architecture

### Content-Addressed Cache

The `ResultCache` uses content-addressed keys computed from stage inputs:

```python
key = cache.compute_key(
    stage=PipelineStage.PARSE,
    inputs={"source_hash": sha256(html), "config_hash": sha256(config)},
)
```

### Cache Layers

1. **Memory cache**: in-process dictionary with LRU eviction (`max_memory_entries`)
2. **Disk cache**: filesystem-backed cache in `cache_dir` with TTL expiration

### Cache Invalidation

- TTL-based expiration (default: 3600 seconds)
- Manual invalidation via `cache.invalidate(key)`
- Full clear via `cache.clear()`
- Cache hit rate tracked via `cache.hit_rate` property

---

## Concurrency Model

### Thread Safety

- `OracleConfig` and all config dataclasses are immutable (frozen)
- `CostElement` is a frozen dataclass with `__hash__`
- `ResultCache` uses thread-safe dictionary access
- `ParallelExecutor` supports both thread and process pools

### Process Isolation

For CPU-bound stages (bisimulation, Monte Carlo), `ParallelExecutor` can use
`ProcessPoolExecutor` (`use_threads=False`) to bypass the GIL:

```python
executor = ParallelExecutor(max_workers=4, use_threads=False)
results = executor.execute_map(sample_trajectory, mdp_states)
```

---

*For API-level documentation, see [README.md](README.md). For mathematical foundations,
see [docs/theory.md](docs/theory.md).*
