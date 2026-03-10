# Bounded-Rational Usability Oracle

> Information-theoretic cognitive cost analysis for automated usability regression testing in CI/CD pipelines.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

---

## Table of Contents

- [Overview](#overview)
- [Architecture Overview](#architecture-overview)
- [Module Map](#module-map)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Usage](#cli-usage)
- [Supported Formats](#supported-formats)
- [Mathematical Foundations](#mathematical-foundations)
- [Configuration Reference](#configuration-reference)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

---

## Overview

The Bounded-Rational Usability Oracle detects **structural usability regressions** by
modeling users as information-theoretic bounded-rational agents. Given two versions of a
UI (before and after a code change) and a task specification, it constructs task-state
MDPs from accessibility trees, annotates transitions with compositional cognitive costs
(Fitts' law, Hick–Hyman law, visual search, working memory), computes bounded-rational
policies, and produces a regression verdict with formal error bounds.

**Key properties:**

- **Deterministic** — same inputs always produce the same verdict
- **Quantitative** — scalar cost differentials with confidence intervals
- **Monotone** — strictly worse UIs are never reported as improvements
- **Bounded** — formal error bounds on approximation and sampling error
- **Fast** — laptop-CPU execution within CI/CD wall-clock constraints (≤60s for ≤500 elements)

**What it detects:** regressions in information architecture, interaction flows, element
counts, groupings, navigation depth, and affordance structure.

**What it does not detect:** visual regressions (CSS, color, spacing, animation).
Use screenshot-diff tools (Chromatic, Percy, BackstopJS) for those.

---

## Architecture Overview

The system is organized into three conceptual layers comprising 24 packages:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          CLI & Output Layer                            │
│  ┌──────────┐  ┌──────────────┐  ┌────────────┐  ┌─────────────────┐  │
│  │   cli/   │  │   output/    │  │ pipeline/  │  │   benchmarks/   │  │
│  │  main.py │  │  console.py  │  │  runner.py │  │    suite.py     │  │
│  │  github_ │  │  sarif.py    │  │  stages.py │  │  generators.py  │  │
│  │  action   │  │  json_output │  │  cache.py  │  │   metrics.py    │  │
│  └────┬─────┘  └──────┬───────┘  └─────┬──────┘  └────────┬────────┘  │
│       │               │                │                   │           │
├───────┴───────────────┴────────────────┴───────────────────┴───────────┤
│                    Analysis & Comparison Layer                         │
│  ┌────────────┐  ┌─────────────┐  ┌────────────┐  ┌───────────────┐   │
│  │ comparison/ │  │ bottleneck/ │  │  repair/   │  │  evaluation/  │   │
│  │  paired.py  │  │ classifier  │  │ synthesizer│  │  ordinal.py   │   │
│  │ hypothesis  │  │ signatures  │  │ mutations  │  │  ablation.py  │   │
│  │ error_bounds│  │  models.py  │  │ strategies │  │  baselines    │   │
│  └──────┬─────┘  └──────┬──────┘  └─────┬──────┘  └──────┬────────┘   │
│         │               │                │                │            │
├─────────┴───────────────┴────────────────┴────────────────┴────────────┤
│                  Cognitive Modeling & MDP Layer                        │
│  ┌────────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────────────┐  │
│  │ cognitive/ │  │   mdp/   │  │  policy/  │  │   bisimulation/   │  │
│  │  fitts.py  │  │ builder  │  │ softmax   │  │ cognitive_distance │  │
│  │  hick.py   │  │ models   │  │free_energy│  │    partition       │  │
│  │ visual_    │  │ solver   │  │monte_carlo│  │    quotient        │  │
│  │  search    │  │trajectory│  │ optimal   │  │   clustering       │  │
│  │ working_   │  │ features │  │value_iter │  │   validators       │  │
│  │  memory    │  │ reward   │  └─────┬─────┘  └────────┬───────────┘  │
│  └─────┬──────┘  └────┬─────┘        │                 │              │
│        │              │              │                 │              │
├────────┴──────────────┴──────────────┴─────────────────┴──────────────┤
│                   Foundation & Parsing Layer                          │
│  ┌──────────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ │
│  │accessibility/│ │ algebra/ │ │alignment/│ │taskspec/ │ │  core/ │ │
│  │ html_parser  │ │ models   │ │ differ   │ │  dsl.py  │ │ types  │ │
│  │ json_parser  │ │sequential│ │exact_    │ │ models   │ │ enums  │ │
│  │  models      │ │ parallel │ │  match   │ │templates │ │protocol│ │
│  │ normalizer   │ │ context  │ │fuzzy_    │ │inference │ │ config │ │
│  │   roles      │ │ composer │ │  match   │ │recorder  │ │ errors │ │
│  │  spatial     │ │soundness │ │visualizer│ │validator │ │constant│ │
│  │ validators   │ │optimizer │ │classifier│ └────┬─────┘ └───┬────┘ │
│  └──────┬───────┘ └────┬─────┘ └────┬─────┘      │          │      │
│         │              │            │             │          │      │
│  ┌──────┴──────┐ ┌─────┴──────┐ ┌──┴───────┐ ┌──┴──────────┴────┐ │
│  │ fragility/  │ │ interval/  │ │  utils/  │ │   Cross-cutting  │ │
│  │  analyzer   │ │ interval   │ │ entropy  │ │   dependencies   │ │
│  │sensitivity  │ │ arithmetic │ │  graph   │ │                  │ │
│  │   cliff     │ │propagation │ │  math    │ │  numpy, scipy,   │ │
│  │ adversarial │ │ comparison │ │sampling  │ │  z3, networkx,   │ │
│  │  inclusive   │ └────────────┘ │ logging  │ │  click, rich,    │ │
│  └─────────────┘                │ timing   │ │  lxml, html5lib  │ │
│                                 │serializ. │ │                  │ │
│                                 └──────────┘ └──────────────────┘ │
└───────────────────────────────────────────────────────────────────────┘
```

**Data flow:**

```
  HTML/ARIA or JSON          Task YAML
  accessibility tree         specification
        │                        │
        ▼                        ▼
  ┌──────────┐            ┌──────────┐
  │  Parse   │            │  Parse   │
  │ (access.)│            │(taskspec)│
  └────┬─────┘            └────┬─────┘
       │   AccessibilityTree   │  TaskSpec
       ▼                       ▼
  ┌──────────────────────────────────┐
  │         Align (differ)           │  ← 3-pass semantic diff
  └───────────────┬──────────────────┘
                  │  AlignmentResult
                  ▼
  ┌──────────────────────────────────┐
  │    Build MDPs (mdp/builder)      │  ← per-version task-state MDPs
  └───────────────┬──────────────────┘
                  │  MDP_A, MDP_B
                  ▼
  ┌──────────────────────────────────┐
  │  Annotate costs (algebra)        │  ← ⊕ sequential, ⊗ parallel, Δ context
  └───────────────┬──────────────────┘
                  │  Weighted MDPs
                  ▼
  ┌──────────────────────────────────┐
  │  Bisimulate (bisimulation)       │  ← bounded-rational ε-bisimulation
  └───────────────┬──────────────────┘
                  │  Abstract MDPs
                  ▼
  ┌──────────────────────────────────┐
  │  Compute policies (policy)       │  ← softmax / free-energy optimal
  └───────────────┬──────────────────┘
                  │  Trajectory distributions
                  ▼
  ┌──────────────────────────────────┐
  │  Compare (comparison)            │  ← hypothesis tests, error bounds
  └───────────────┬──────────────────┘
                  │  RegressionVerdict
                  ▼
  ┌──────────────────────────────────┐
  │  Classify bottlenecks            │  ← 5-type taxonomy
  └───────────────┬──────────────────┘
                  │  BottleneckReport
                  ▼
  ┌──────────────────────────────────┐
  │  Synthesize repairs (optional)   │  ← SMT/Z3-backed
  └───────────────┬──────────────────┘
                  │  RepairCandidates
                  ▼
  ┌──────────────────────────────────┐
  │  Format output                   │  ← console / JSON / SARIF
  └──────────────────────────────────┘
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full architecture document.

---

## Module Map

### `core/` — Foundation types and contracts

| File | Description |
|------|-------------|
| `types.py` | Core data types: `Point2D`, `BoundingBox`, `Interval`, `CostTuple`, `Trajectory`, `TrajectoryStep`, `PolicyDistribution` |
| `enums.py` | Enumerations: `AccessibilityRole`, `CognitiveLaw`, `BottleneckType`, `EditOperationType`, `RegressionVerdict`, `PipelineStage`, `ComparisonMode`, `OutputFormat`, `Severity`, `MotorChannel`, `PerceptualChannel` |
| `protocols.py` | 28 protocol interfaces: `Parser`, `Aligner`, `CostModel`, `MDP`, `Policy`, `PolicyComputer`, `BottleneckClassifier`, `RepairSynthesizer`, `Validator`, `OutputFormatter`, `AccessibilityNode`, `AccessibilityTree`, `CacheProvider`, `Serializable`, etc. |
| `config.py` | Configuration dataclasses: `OracleConfig` (top-level), `ParserConfig`, `AlignmentConfig`, `CognitiveConfig`, `MDPConfig`, `PolicyConfig`, `BisimulationConfig`, `ComparisonConfig`, `RepairConfig`, `OutputConfig`, `PipelineConfig` |
| `errors.py` | Exception hierarchy rooted at `UsabilityOracleError`: `ParseError`, `AlignmentError`, `CostModelError`, `MDPError`, `PolicyError`, `BisimulationError`, `BottleneckError`, `ComparisonError`, `RepairError`, `ConfigError`, `PipelineError`, `CacheError` |
| `constants.py` | Psychophysical constants as `Interval` ranges with literature citations: Fitts' law parameters, Hick–Hyman parameters, working-memory capacity, visual-search slopes, decay rates |

### `accessibility/` — UI tree parsing and normalization

| File | Description |
|------|-------------|
| `models.py` | `AccessibilityNode` (with `is_interactive()`, `is_visible()`, `semantic_hash()`, `find_by_role()`, tree traversal), `AccessibilityTree` (with `get_interactive_nodes()`, `get_visible_nodes()`, `iter_bfs()`, `iter_dfs()`, `validate()`), `AccessibilityState`, `BoundingBox` |
| `html_parser.py` | `HTMLAccessibilityParser` — parses HTML/ARIA markup into `AccessibilityTree` via `parse(html: str)`. Infers roles from HTML elements, extracts ARIA attributes, computes bounding boxes |
| `json_parser.py` | `JSONAccessibilityParser` — parses Chrome DevTools, axe-core, and generic JSON accessibility tree formats via `parse(json_str: str)` and `parse_file(path)` |
| `normalizer.py` | `AccessibilityNormalizer` — normalizes roles, names, bounding boxes; removes decorative elements; collapses wrapper nodes; assigns semantic levels |
| `roles.py` | `RoleTaxonomy` — ARIA role hierarchy with `is_interactive()`, `is_landmark()`, `semantic_similarity()`, `taxonomy_distance()`, `can_contain()` |
| `spatial.py` | `SpatialAnalyzer` — layout analysis: `compute_layout()`, `detect_groups()`, `fitts_distance()`, `fitts_index_of_difficulty()`, `compute_reading_order()`, `find_overlapping_pairs()` |
| `validators.py` | `TreeValidator` — structural validation: unique IDs, role validity, name presence for interactive elements, parent-child consistency, containment checks. Returns `ValidationResult` |

### `algebra/` — Compositional cognitive cost algebra

| File | Description |
|------|-------------|
| `models.py` | `CostElement(μ, σ², κ, λ)` — 4-tuple encoding mean, variance, capacity utilization, and interference susceptibility. `CostExpression` abstract tree with `Leaf`, `Sequential`, `Parallel`, `ContextMod` nodes |
| `sequential.py` | `SequentialComposer` — the ⊕ operator: `compose(a, b, coupling)`, `compose_chain(elements, couplings)`, `compose_interval()`, `compose_matrix()`, `sensitivity()` |
| `parallel.py` | `ParallelComposer` — the ⊗ operator with Multiple Resource Theory: `compose(a, b, interference)`, `compose_group()`, `compose_with_channels()`, `interference_factor()`. Includes `INTERFERENCE_MATRIX` |
| `context.py` | `ContextModulator` — the Δ operator: `modulate(element, context)` applies fatigue, working-memory load, practice, stress, and age modulations via `CognitiveContext` |
| `composer.py` | `TaskGraphComposer` — DAG-level composition: `compose(task_graph, cost_map)`, `critical_path_cost()`, `parallelism_factor()`, `bottleneck_nodes()` |
| `soundness.py` | `SoundnessVerifier` — axiom verification: `verify_sequential()`, `verify_parallel()`, `verify_monotonicity()`, `verify_triangle_inequality()`, `verify_commutativity()`, `verify_identity()`, `verify_all()` |
| `optimizer.py` | `AlgebraicOptimizer` — expression simplification: flatten, eliminate zeros, factor common terms, reorder commutative sub-expressions |

### `alignment/` — Semantic tree differencing

| File | Description |
|------|-------------|
| `differ.py` | `SemanticDiffer` — 3-pass alignment: `_pass1_exact()` (structural match), `_pass2_fuzzy()` (heuristic match), `_pass3_classify()` (edit classification). Returns `AlignmentResult` |
| `exact_match.py` | Exact node matching by semantic hash |
| `fuzzy_match.py` | Fuzzy matching using role similarity, name edit distance, spatial proximity |
| `classifier.py` | Edit operation classifier (ADD, REMOVE, MODIFY, REORDER) |
| `cost_model.py` | Cost assignment for edit operations |
| `models.py` | `AlignmentResult`, `NodeMapping`, `EditOperation` data models |
| `visualizer.py` | Visual diff output |

### `cognitive/` — Psychophysical law implementations

| File | Description |
|------|-------------|
| `fitts.py` | `FittsLaw` — `predict(distance, width)`, `predict_interval()`, `index_of_difficulty()`, `throughput()`, `effective_width()`, `crossing_time()`, `predict_batch()`, `error_rate()` |
| `hick.py` | `HickHymanLaw` — `predict(n_alternatives)`, `predict_interval()`, `predict_unequal_probabilities()`, `entropy()`, `information_gain()`, `effective_alternatives()`, `predict_with_practice()` |
| `working_memory.py` | `WorkingMemoryModel` — `predict_recall_probability()`, `chunk_count()`, `rehearsal_cost()`, `interference_factor()`, `load_cost()`, `proactive_interference()`, `total_memory_cost()` |
| `visual_search.py` | `VisualSearchModel` — `predict_serial()`, `predict_parallel()`, `predict_guided()`, `saliency_from_structure()`, `effective_set_size()`, `eccentricity_cost()` |
| `motor.py` | Motor execution models |
| `perception.py` | Perceptual channel models |
| `calibration.py` | Parameter calibration utilities |
| `parameters.py` | Published psychophysical parameter values |
| `models.py` | Shared cognitive data models |

### `mdp/` — Markov Decision Process construction

| File | Description |
|------|-------------|
| `models.py` | `State` (with `node_id`, `task_progress`, `working_memory_load`), `Action` (CLICK, TYPE, TAB, SCROLL, NAVIGATE, READ, SELECT, BACK), `Transition`, `MDP` (with `get_actions()`, `get_transitions()`, `reachable_states()`, `to_networkx()`, `statistics()`), `MDPStatistics` |
| `builder.py` | `MDPBuilder` — `build(tree, task_spec)` constructs task-state MDPs from accessibility trees. `MDPBuilderConfig` controls max states (default 50K) and per-action costs |
| `solver.py` | MDP solvers (value iteration, policy iteration) |
| `trajectory.py` | Trajectory sampling and statistics |
| `features.py` | State feature extraction |
| `reward.py` | Reward function definitions |
| `visualization.py` | MDP graph visualization |

### `bisimulation/` — Bounded-rational state abstraction

| File | Description |
|------|-------------|
| `cognitive_distance.py` | `CognitiveDistanceComputer` — `compute_distance(s1, s2, mdp, beta)`, `compute_distance_matrix()`. Implements d_cog(s1, s2) = sup_{β'≤β} d_TV(π_{β'}(·|s1), π_{β'}(·|s2)) |
| `partition.py` | `PartitionRefinement` — `refine(mdp, beta, epsilon)` iteratively refines partitions to achieve ε-bisimulation. Abstract policy signatures guide splitting decisions |
| `quotient.py` | `QuotientMDPBuilder` — `build(mdp, partition)` constructs abstract quotient MDPs. `verify_quotient()` checks abstraction quality |
| `clustering.py` | State clustering heuristics |
| `models.py` | Partition and block data models |
| `validators.py` | Partition validation |

### `policy/` — Bounded-rational policy computation

| File | Description |
|------|-------------|
| `softmax.py` | `SoftmaxPolicy` — `from_q_values(q_values, beta, prior)` implements π_β(a|s) ∝ p₀(a|s)·exp(−β·Q(s,a)). `kl_divergence()`, `mutual_information()`, `beta_sweep()`, `effective_rationality()` |
| `free_energy.py` | `FreeEnergyComputer` — `compute()` evaluates F(π) = E_π[c] + (1/β)·D_KL(π‖p₀). `decompose()`, `optimal_policy()`, `rate_distortion_curve()` |
| `monte_carlo.py` | `MonteCarloEstimator` — `estimate_value()`, `estimate_q_values()`, `estimate_free_energy()`, `confidence_intervals()`. First-visit and every-visit MC methods |
| `value_iteration.py` | Exact value iteration solver |
| `optimal.py` | Optimal (fully rational) policy baseline |
| `models.py` | Policy data models |

### `comparison/` — Regression detection

| File | Description |
|------|-------------|
| `paired.py` | `PairedComparator` — paired UI version comparison using shared bisimulation partition for correlated error cancellation |
| `hypothesis.py` | `RegressionTester` — `test(cost_samples_a, cost_samples_b, alpha)` via Welch's t-test, Mann–Whitney U, or bootstrap. `test_multiple()` with FDR correction. Returns `HypothesisResult` |
| `error_bounds.py` | `ErrorBoundComputer` — `compute_abstraction_error()`, `compute_sampling_error()`, `compute_required_samples()`, `full_analysis()`. Hoeffding-based bounds |
| `models.py` | Comparison result data models |
| `reporter.py` | Regression report generation |
| `union_mdp.py` | Union MDP construction for shared abstraction |
| `parameter_free.py` | Parameter-free comparison methods |

### `bottleneck/` — Cognitive bottleneck taxonomy

| File | Description |
|------|-------------|
| `classifier.py` | `BottleneckClassifier` — `classify(mdp, policy, trajectory_stats, cost_breakdown)` returns classified bottlenecks. `classify_to_report()` produces aggregate `BottleneckReport` |
| `signatures.py` | `SignatureComputer` — `compute(mdp, policy, state)` extracts information-theoretic `BottleneckSignature`. `classify_signature()` maps signatures to bottleneck types |
| `models.py` | `BottleneckSignature`, `BottleneckResult` (with `severity_score`, `impact_score`), `BottleneckReport` (with `generate_summary()`, `by_type()`, `type_distribution()`) |
| `perceptual.py` | Perceptual overload detection (H(S_t \| display) > τ_p) |
| `choice.py` | Choice paralysis detection (log\|A_t\| − I(S_t; A_t) > τ_c) |
| `motor.py` | Motor difficulty detection |
| `memory.py` | Working-memory decay detection (I(S_t; S_{t−k}) < τ_μ) |
| `interference.py` | Cross-channel interference detection (I(A^(1)_t; A^(2)_t \| S_t) > τ_ι) |
| `repair_map.py` | Maps bottleneck types to repair strategies |

### `repair/` — SMT-backed repair synthesis

| File | Description |
|------|-------------|
| `synthesizer.py` | `RepairSynthesizer` — `synthesize(mdp, bottlenecks, constraints, timeout)` uses Z3 to find minimal UI mutations that restore the cost envelope |
| `mutations.py` | `MutationOperator` — `apply(tree, mutation)`: resize, reposition, regroup, simplify menus, add landmarks |
| `constraints.py` | Z3 constraint encoding |
| `strategies.py` | Repair strategy selection by bottleneck type |
| `models.py` | Repair data models |
| `validator.py` | Repair validation (verifies fix doesn't introduce new bottlenecks) |

### `taskspec/` — Task specification DSL

| File | Description |
|------|-------------|
| `dsl.py` | `TaskDSLParser` — `parse(yaml_str)`, `parse_file(path)`, `parse_directory()`, `to_yaml()`. YAML-based task specification language |
| `models.py` | `TaskStep` (action_type, target_role, target_name, preconditions, postconditions), `TaskFlow` (ordered steps with success criteria), `TaskSpec` (named collection of flows), `TaskGraph` (dependency DAG) |
| `templates.py` | `TaskTemplates` — pre-built specs: `login_form()`, `search_and_select()`, `form_fill()`, `multi_step_wizard()`, `navigation()`, `shopping_cart()`, `settings()` |
| `inference.py` | Automatic task inference from UI structure |
| `recorder.py` | Task recording from interaction traces |
| `validator.py` | Task specification validation |

### `pipeline/` — Orchestration

| File | Description |
|------|-------------|
| `runner.py` | `PipelineRunner` — `run(config, source_a, source_b, task_spec)` orchestrates the full analysis pipeline. Returns `PipelineResult` with per-stage timing |
| `stages.py` | 10 `StageExecutor` implementations: Parse, Align, Cost, MDP, Bisimulation, Policy, Comparison, Bottleneck, Repair, Output. `StageRegistry` for registration and lookup |
| `config.py` | `FullPipelineConfig` — `from_yaml()`, `from_env()`, `DEFAULT()`. Per-stage enable/disable, timeout, retry configuration |
| `cache.py` | `ResultCache` — content-addressed caching with TTL, memory bounds, and hit-rate tracking |
| `parallel.py` | `ParallelExecutor` — thread/process pool execution: `execute_parallel()`, `execute_map()`, `execute_with_results()` |

### `output/` — Result formatting

| File | Description |
|------|-------------|
| `console.py` | `ConsoleFormatter` — Rich-based terminal output with verdict panels, cost tables, bottleneck trees, and timing summaries |
| `json_output.py` | `JSONFormatter` — structured JSON output with `format()`, `format_compact()`, `schema()` |
| `sarif.py` | `SARIFFormatter` — SARIF 2.1.0 output for IDE and CI integration |
| `html_report.py` | HTML report generation |
| `models.py` | Output data models |

### `cli/` — Command-line interface

| File | Description |
|------|-------------|
| `main.py` | Click-based CLI with commands: `diff`, `analyze`, `benchmark`, `validate`, `init` |
| `formatters.py` | CLI output formatting utilities |
| `github_action.py` | `GitHubActionIntegration` — `run(event_payload)` for CI: PR comment posting, annotation emission, check-run formatting |

### `evaluation/` — Validation and benchmarking

| File | Description |
|------|-------------|
| `ordinal.py` | `OrdinalValidator` — `validate(model_orderings, human_orderings)` computes Spearman ρ, Kendall τ with bootstrap confidence intervals |
| `ablation.py` | Ablation study framework |
| `baselines.py` | Baseline comparators (axe-core, KLM, scalar threshold) |
| `regression.py` | Regression detection metrics |
| `reporting.py` | Evaluation report generation |

### `fragility/` — Robustness analysis

| File | Description |
|------|-------------|
| `analyzer.py` | `FragilityAnalyzer` — `analyze(mdp, task, beta_range)`: fragility scores, cost curves, discontinuity detection, population impact analysis |
| `sensitivity.py` | Parameter sensitivity analysis |
| `cliff.py` | Cliff detection in cost landscapes |
| `adversarial.py` | Adversarial perturbation testing |
| `inclusive.py` | Inclusive design analysis across population β-distributions |
| `models.py` | Fragility data models |

### `interval/` — Interval arithmetic

| File | Description |
|------|-------------|
| `interval.py` | `Interval` — closed interval [low, high] with rigorous arithmetic (+, −, ×, ÷, ^), elementary functions (sqrt, exp, log), set operations (union, intersection), predicates |
| `arithmetic.py` | Extended interval arithmetic operations |
| `comparison.py` | Interval comparison predicates |
| `propagation.py` | Uncertainty propagation through expressions |

### `utils/` — Shared utilities

| File | Description |
|------|-------------|
| `entropy.py` | Shannon entropy, mutual information, KL divergence |
| `graph.py` | Graph algorithms (topological sort, shortest path, connected components) |
| `math.py` | Numerical utilities (log-sum-exp, softmax, stable division) |
| `sampling.py` | Random sampling utilities |
| `logging.py` | Structured logging configuration |
| `timing.py` | Performance timing decorators and context managers |
| `serialization.py` | JSON/YAML serialization helpers |

### `formats/` — Platform-specific accessibility tree parsers

| File | Description |
|------|-------------|
| `aria.py` | ARIA/HTML accessibility format parsing |
| `chrome_devtools.py` | Chrome DevTools accessibility tree format |
| `axe_core.py` | axe-core JSON result format |
| `android.py` | Android accessibility service format |
| `ios.py` | iOS accessibility format |
| `windows.py` | Windows UI Automation format |
| `converters.py` | Cross-format conversion utilities |

### `simulation/` — Bounded-rational agent simulation

| File | Description |
|------|-------------|
| `agent.py` | Simulated bounded-rational user agent |
| `environment.py` | UI environment for agent interaction |
| `interaction.py` | Interaction trace recording |
| `scenarios.py` | Pre-built simulation scenarios |
| `metrics.py` | Simulation outcome metrics |
| `recorder.py` | Interaction recording utilities |

### `visualization/` — Plotting and visual output

| File | Description |
|------|-------------|
| `cost_viz.py` | Cost curve and algebra visualizations |
| `bottleneck_viz.py` | Bottleneck analysis visualizations |
| `fragility_viz.py` | Fragility and cliff visualizations |
| `mdp_viz.py` | MDP graph visualizations |
| `tree_viz.py` | Accessibility tree visualizations |
| `report_viz.py` | Report-level composite visualizations |
| `colors.py` | Color scheme and palette definitions |

### `analysis/` — Statistical and information-theoretic analysis

| File | Description |
|------|-------------|
| `statistical.py` | Statistical analysis utilities |
| `information.py` | Information-theoretic measures |
| `sensitivity.py` | Sensitivity analysis framework |
| `convergence.py` | Convergence diagnostics |
| `stability.py` | Stability analysis |
| `complexity.py` | Computational complexity analysis |

### `benchmarks/` — Benchmark suite

| File | Description |
|------|-------------|
| `suite.py` | Benchmark test suite runner |
| `generators.py` | Synthetic UI tree generators |
| `mutations.py` | Controlled mutation operators for benchmarks |
| `metrics.py` | Benchmark evaluation metrics |
| `datasets.py` | Dataset loading and management |

---

## Installation

### Requirements

- Python 3.10 or later
- No GPU required — all computation runs on CPU

### From source

```bash
git clone https://github.com/your-org/bounded-rational-usability-oracle.git
cd bounded-rational-usability-oracle/proposals/proposal_00/implementation

# Install in development mode
pip install -e ".[dev]"
```

### Dependencies

Core dependencies (installed automatically):

| Package | Purpose |
|---------|---------|
| `numpy>=1.24` | Numerical computation |
| `scipy>=1.10` | Statistical tests, optimization |
| `pyyaml>=6.0` | Configuration and task spec parsing |
| `lxml>=4.9` | HTML/XML parsing |
| `html5lib>=1.1` | HTML5-compliant parsing |
| `z3-solver>=4.12` | SMT constraint solving for repair synthesis |
| `networkx>=3.0` | Graph algorithms for MDP analysis |
| `click>=8.1` | CLI framework |
| `dataclasses-json>=0.6` | Serialization |
| `rich>=13.0` | Terminal output formatting |

Development dependencies:

| Package | Purpose |
|---------|---------|
| `pytest>=7.4` | Testing |
| `pytest-cov>=4.1` | Coverage |
| `hypothesis>=6.80` | Property-based testing |
| `mypy>=1.5` | Static type checking |
| `ruff>=0.1` | Linting |

---

## Quick Start

### 1. Initialize a project

```bash
usability-oracle init --output-dir .usability
```

This creates a `.usability/` directory with default configuration and example task specs.

### 2. Write a task specification

Create `task_login.yaml`:

```yaml
spec_id: login-flow
name: User Login
description: Standard username/password login flow

flows:
  - flow_id: login
    name: Login with credentials
    steps:
      - step_id: enter-username
        action_type: TYPE
        target_role: TEXTBOX
        target_name: "Username"
        input_value: "testuser"

      - step_id: enter-password
        action_type: TYPE
        target_role: TEXTBOX
        target_name: "Password"
        input_value: "secret"
        depends_on: [enter-username]

      - step_id: click-submit
        action_type: CLICK
        target_role: BUTTON
        target_name: "Sign In"
        depends_on: [enter-password]

    success_criteria:
      - type: state_reached
        condition: "logged_in"
    max_time: 30.0
```

### 3. Run a usability diff

```python
from usability_oracle.accessibility import HTMLAccessibilityParser
from usability_oracle.taskspec import TaskDSLParser
from usability_oracle.pipeline import PipelineRunner, FullPipelineConfig

# Parse UI versions
parser = HTMLAccessibilityParser()
tree_before = parser.parse(open("login_v1.html").read())
tree_after = parser.parse(open("login_v2.html").read())

# Parse task specification
task_parser = TaskDSLParser()
task_spec = task_parser.parse_file("task_login.yaml")

# Run the pipeline
config = FullPipelineConfig.DEFAULT()
runner = PipelineRunner(config=config.oracle)
result = runner.run(
    config=config,
    source_a=tree_before,
    source_b=tree_after,
    task_spec=task_spec,
)

# Check verdict
print(f"Verdict: {result.verdict}")        # REGRESSION | IMPROVEMENT | NO_CHANGE
print(f"Cost delta: {result.cost_delta}")
print(f"Bottlenecks: {result.bottleneck_report.generate_summary()}")
```

### 4. Use pre-built task templates

```python
from usability_oracle.taskspec import TaskTemplates

# Generate common task specs automatically
login_spec = TaskTemplates.login_form(
    username_label="Email",
    password_label="Password",
    submit_label="Log In",
)

search_spec = TaskTemplates.search_and_select(
    search_label="Search products",
    result_type="product-card",
    has_filters=True,
)
```

### 5. Analyze a single UI

```python
from usability_oracle.cognitive import FittsLaw, HickHymanLaw, WorkingMemoryModel
from usability_oracle.accessibility import HTMLAccessibilityParser, SpatialAnalyzer

parser = HTMLAccessibilityParser()
tree = parser.parse(html_content)

# Spatial analysis
spatial = SpatialAnalyzer()
layout = spatial.compute_layout(tree)
print(f"Visual density: {layout.visual_density}")
print(f"Element groups: {len(layout.groups)}")

# Cognitive cost estimation
interactive = tree.get_interactive_nodes()
hick_rt = HickHymanLaw.predict(n_alternatives=len(interactive))
print(f"Choice reaction time for {len(interactive)} options: {hick_rt:.3f}s")
```

---

## CLI Usage

### `diff` — Compare two UI versions

```bash
# Basic diff
usability-oracle diff before.html after.html --task-spec task.yaml

# With JSON output
usability-oracle diff before.html after.html \
    --task-spec task.yaml \
    --output-format json \
    --output report.json

# With SARIF output for IDE integration
usability-oracle diff before.html after.html \
    --task-spec task.yaml \
    --output-format sarif \
    --output results.sarif

# Custom beta range for bounded-rational analysis
usability-oracle diff before.html after.html \
    --task-spec task.yaml \
    --beta-range 1.0,10.0

# Verbose mode with detailed timing
usability-oracle diff before.html after.html \
    --task-spec task.yaml \
    --verbose

# With custom configuration
usability-oracle diff before.html after.html \
    --task-spec task.yaml \
    --config .usability/config.yaml
```

### `analyze` — Analyze a single UI

```bash
# Analyze a single HTML file
usability-oracle analyze ui.html --task-spec task.yaml

# Analyze a JSON accessibility tree
usability-oracle analyze tree.json --task-spec task.yaml

# Output as JSON
usability-oracle analyze ui.html \
    --task-spec task.yaml \
    --output-format json
```

### `benchmark` — Run benchmark suite

```bash
# Run the default benchmark suite
usability-oracle benchmark

# Run a specific suite with multiple iterations
usability-oracle benchmark --suite regression-detection --n-runs 10

# Save benchmark results
usability-oracle benchmark --output benchmark_results.json --verbose
```

### `validate` — Validate task specifications

```bash
# Validate a task spec file
usability-oracle validate --task-spec-file task.yaml

# Validate against a UI source
usability-oracle validate --task-spec-file task.yaml --ui-source ui.html
```

### `init` — Initialize project

```bash
# Create default config directory
usability-oracle init

# Custom output directory
usability-oracle init --output-dir my-config

# Overwrite existing configuration
usability-oracle init --output-dir .usability --force
```

---

## Supported Formats

### Input Formats

| Format | Parser | Description |
|--------|--------|-------------|
| HTML with ARIA | `HTMLAccessibilityParser` | Standard HTML5 with ARIA attributes (`role`, `aria-label`, `aria-describedby`, etc.) |
| JSON (Chrome DevTools) | `JSONAccessibilityParser` | Chrome DevTools Protocol accessibility tree snapshots |
| JSON (axe-core) | `JSONAccessibilityParser` | axe-core accessibility audit results |
| JSON (generic) | `JSONAccessibilityParser` | Generic JSON accessibility tree format |
| YAML (task specs) | `TaskDSLParser` | YAML-based task specification DSL |

### Output Formats

| Format | Formatter | Use Case |
|--------|-----------|----------|
| Console (Rich) | `ConsoleFormatter` | Interactive terminal output with color, tables, trees |
| JSON | `JSONFormatter` | Machine-readable output for scripting and dashboards |
| SARIF 2.1.0 | `SARIFFormatter` | IDE integration (VS Code, GitHub Code Scanning) |
| HTML Report | `HTMLReportFormatter` | Standalone HTML reports for sharing |

---

## Mathematical Foundations

The system rests on three theoretical contributions. See [docs/theory.md](docs/theory.md)
for full details.

### Bounded-Rational Framework

Users are modeled as bounded-rational agents that minimize free energy:

```
F(π) = E_π[cost] + (1/β) · D_KL(π ‖ p₀)
```

where `β` is the rationality parameter (information-processing capacity), `π` is the
agent's policy, and `p₀` is the prior (default behavior). This recovers:

- **Fitts' Law** as a capacity-constrained motor channel
- **Hick–Hyman Law** as a capacity-constrained choice channel
- **Visual search** as a capacity-constrained perceptual channel
- **Working memory decay** as temporal information loss

### Compositional Cost Algebra

Three operators compose cognitive costs (see [docs/cost_algebra.md](docs/cost_algebra.md)):

| Operator | Symbol | Semantics |
|----------|--------|-----------|
| Sequential | ⊕ | `μ₁₊₂ = μ₁ + μ₂ + γ·κ₁·μ₂` — prior load amplifies subsequent cost |
| Parallel | ⊗ | `μ₁ₓ₂ = max(μ₁,μ₂) + α·λ₁λ₂·min(μ₁,μ₂)` — interference between concurrent operations |
| Context | Δ | Scales (μ, σ²) by fatigue, memory load, practice, stress, age |

**Soundness guarantee:** The composed cost upper-bounds the mutual information of the
full discrete-event cognitive simulation: `C_alg(G) ≥ Σ_t I(S_t; A_t)`.

### Bounded-Rational Bisimulation

States are merged when a capacity-limited agent cannot distinguish them:

```
d_cog(s₁, s₂) = sup_{β'≤β} d_TV(π_{β'}(·|s₁), π_{β'}(·|s₂))
```

**Paired-Comparison Theorem:** When both UI versions are abstracted under the same
partition, regression detection error satisfies `|(Ĉ_B − Ĉ_A) − (C_B − C_A)| ≤ O(ε)`
rather than the naive `O(Hβε)` bound — correlated errors cancel.

### Bottleneck Taxonomy

Five information-theoretic bottleneck types (see [docs/bottleneck_taxonomy.md](docs/bottleneck_taxonomy.md)):

| Type | Signature | Repair Strategy |
|------|-----------|-----------------|
| Perceptual | H(S_t \| display) > τ_p | Grouping, contrast |
| Choice | log\|A_t\| − I(S_t; A_t) > τ_c | Progressive disclosure |
| Motor | I(A_t; target) < τ_m | Target sizing, placement |
| Memory | I(S_t; S_{t−k}) < τ_μ | Persistent state cues |
| Interference | I(A^(1); A^(2) \| S_t) > τ_ι | Modality separation |

---

## Configuration Reference

### Default configuration (`oracle_config.yaml`)

```yaml
# Parser settings
parser:
  use_html5lib: true
  include_text_nodes: false
  id_prefix: "node"

# Alignment settings
alignment:
  fuzzy_threshold: 0.7
  max_edit_distance: 5

# Cognitive model parameters
cognitive:
  fitts_a: 0.050          # Fitts' law intercept (seconds)
  fitts_b: 0.150          # Fitts' law slope (seconds/bit)
  hick_a: 0.200           # Hick-Hyman intercept (seconds)
  hick_b: 0.155           # Hick-Hyman slope (seconds/bit)
  wm_capacity: 4          # Working memory capacity (chunks)
  wm_decay_rate: 0.077    # Memory decay rate (per second)
  visual_search_slope: 0.025  # Serial search slope (seconds/item)

# MDP construction
mdp:
  max_states: 50000
  click_cost: 0.3
  type_cost: 0.5
  tab_cost: 0.2
  scroll_cost: 0.4
  read_cost: 0.3
  navigate_cost: 0.6

# Bounded-rational policy
policy:
  beta: 5.0               # Default rationality parameter
  beta_min: 1.0            # Minimum beta for sweep
  beta_max: 20.0           # Maximum beta for sweep
  discount: 0.99
  max_iterations: 5000
  convergence_epsilon: 1e-6

# Bisimulation
bisimulation:
  epsilon: 0.005           # Bisimulation granularity
  max_abstract_states: 10000

# Comparison
comparison:
  method: "welch_t"        # welch_t | mann_whitney | bootstrap
  alpha: 0.05              # Significance level
  n_trajectories: 1000
  max_trajectory_steps: 500
  n_bootstrap: 10000
  regression_threshold: 0.05  # 5% cost increase threshold

# Repair synthesis
repair:
  enabled: false           # Stretch goal — disabled by default
  max_mutations: 5
  timeout: 60              # Z3 solver timeout (seconds)

# Output
output:
  format: "console"        # console | json | sarif | html
  verbose: false
  include_timing: true

# Pipeline
pipeline:
  log_level: "INFO"
  cache_enabled: true
  cache_ttl: 3600
  max_workers: 4
  fail_fast: true
```

### Environment variable overrides

```bash
USABILITY_ORACLE_CONFIG=path/to/config.yaml
USABILITY_ORACLE_BETA=5.0
USABILITY_ORACLE_ALPHA=0.05
USABILITY_ORACLE_FORMAT=json
USABILITY_ORACLE_VERBOSE=1
USABILITY_ORACLE_CACHE_DIR=/tmp/usability_cache
```

### Per-stage configuration

Each pipeline stage can be individually configured:

```yaml
stages:
  parse:
    enabled: true
    timeout: 30
    retry: 1
    fail_fast: true
  bisimulation:
    enabled: true
    timeout: 120
    retry: 0
    fail_fast: true
  repair:
    enabled: false     # Disable repair synthesis
    timeout: 60
    retry: 0
    fail_fast: false
```

---

## API Reference

### Core Types

```python
from usability_oracle.core.types import (
    Point2D,
    BoundingBox,
    Interval,
    CostTuple,
    Trajectory,
    TrajectoryStep,
    PolicyDistribution,
)

# Point2D
pt = Point2D(x=100.0, y=200.0)
dist = pt.distance(Point2D(150.0, 250.0))

# BoundingBox
bbox = BoundingBox(x=10, y=20, width=100, height=50)
print(bbox.center, bbox.area)

# Interval (for uncertainty propagation)
iv = Interval(low=0.8, high=1.2)
result = iv * Interval(2.0, 3.0)  # [1.6, 3.6]

# CostTuple
cost = CostTuple(time=0.5, information=2.3, motor=0.3, perceptual=1.0)
```

### Parsing

```python
from usability_oracle.accessibility import (
    HTMLAccessibilityParser,
    JSONAccessibilityParser,
    AccessibilityNormalizer,
    SpatialAnalyzer,
    TreeValidator,
)

# HTML parsing
parser = HTMLAccessibilityParser(use_html5lib=True)
tree = parser.parse("<html>...</html>")

# JSON parsing (Chrome DevTools format)
json_parser = JSONAccessibilityParser()
tree = json_parser.parse_file("accessibility_tree.json")

# Normalization
normalizer = AccessibilityNormalizer(
    remove_decorative=True,
    collapse_wrappers=True,
    normalize_coordinates=True,
)
normalized = normalizer.normalize(tree)

# Validation
validator = TreeValidator(max_depth=20, require_names_for_interactive=True)
result = validator.validate(tree)
print(f"Valid: {result.valid}, Errors: {result.error_count}")

# Spatial analysis
spatial = SpatialAnalyzer()
layout = spatial.compute_layout(tree)
id_fitts = spatial.fitts_index_of_difficulty(distance=200.0, width=40.0)
```

### Cost Algebra

```python
from usability_oracle.algebra import (
    CostElement,
    SequentialComposer,
    ParallelComposer,
    ContextModulator,
    CognitiveContext,
    TaskGraphComposer,
    SoundnessVerifier,
)

# Create cost elements (μ, σ², κ, λ)
click = CostElement(mu=0.3, sigma_sq=0.01, kappa=0.2, lam=0.1)
read = CostElement(mu=0.8, sigma_sq=0.04, kappa=0.6, lam=0.3)

# Sequential composition (⊕)
seq = SequentialComposer()
combined = seq.compose(read, click, coupling=0.15)

# Parallel composition (⊗)
par = ParallelComposer()
concurrent = par.compose(read, click, interference=0.2)

# Context modulation (Δ)
ctx = CognitiveContext(fatigue=0.3, wm_load=0.5, practice=100)
mod = ContextModulator()
adjusted = mod.modulate(click, ctx)

# Verify algebraic soundness
verifier = SoundnessVerifier()
results = verifier.verify_all(expression)
```

### MDP and Policy

```python
from usability_oracle.mdp import MDPBuilder, MDPBuilderConfig
from usability_oracle.policy import (
    SoftmaxPolicy,
    FreeEnergyComputer,
    MonteCarloEstimator,
)

# Build MDP from accessibility tree
builder = MDPBuilder(MDPBuilderConfig(max_states=50000))
mdp = builder.build(tree, task_spec)
stats = mdp.statistics()
print(f"States: {stats.n_states}, Actions: {stats.n_actions}")

# Compute bounded-rational policy
fe = FreeEnergyComputer()
policy = fe.optimal_policy(mdp, beta=5.0, discount=0.99)

# Decompose free energy
decomp = fe.decompose(policy, mdp, beta=5.0)
print(f"Expected cost: {decomp.expected_cost:.3f}")
print(f"Information cost: {decomp.information_cost:.3f}")

# Monte Carlo estimation
mc = MonteCarloEstimator(discount=0.99)
values = mc.estimate_value(mdp, policy, n_trajectories=1000)
ci = mc.confidence_intervals(values, alpha=0.05)

# Rate-distortion curve
curve = fe.rate_distortion_curve(mdp, betas=[1, 2, 5, 10, 20])
```

### Bottleneck Classification

```python
from usability_oracle.bottleneck import (
    BottleneckClassifier,
    SignatureComputer,
)

classifier = BottleneckClassifier()
report = classifier.classify_to_report(
    mdp=mdp,
    policy=policy,
    trajectory_stats=stats,
    cost_breakdown=breakdown,
)

print(report.generate_summary())
for bn in report.by_type(BottleneckType.CHOICE):
    print(f"  Choice paralysis at state {bn.state_id}: severity={bn.severity_score:.2f}")
```

---

## Testing

### Run all tests

```bash
pytest
```

### Run with coverage

```bash
pytest --cov=usability_oracle --cov-report=html
```

### Run specific test modules

```bash
# Core types
pytest tests/test_core.py

# Cognitive models
pytest tests/test_cognitive.py

# Algebra
pytest tests/test_algebra.py

# Integration tests
pytest tests/test_pipeline.py
```

### Type checking

```bash
mypy usability_oracle
```

### Linting

```bash
ruff check usability_oracle
ruff format --check usability_oracle
```

### Property-based tests

The project uses Hypothesis for property-based testing of mathematical invariants:

```bash
pytest tests/test_algebra_properties.py -v
```

---

## Contributing

### Development setup

```bash
# Clone and install
git clone https://github.com/your-org/bounded-rational-usability-oracle.git
cd bounded-rational-usability-oracle/proposals/proposal_00/implementation
pip install -e ".[dev]"

# Run checks
ruff check usability_oracle
mypy usability_oracle
pytest
```

### Code style

- Python 3.10+ with full type annotations
- Line length: 100 characters (configured in `pyproject.toml`)
- Linter: `ruff` with default rules
- Type checker: `mypy` in strict mode
- Docstrings: Google style

### Architecture guidelines

1. All public interfaces are defined as `Protocol` classes in `core/protocols.py`
2. Configuration is immutable dataclasses in `core/config.py`
3. Errors follow the hierarchy in `core/errors.py`
4. Mathematical constants include literature citations and are expressed as `Interval` ranges
5. New cognitive laws must implement the `CostModel` protocol
6. New output formats must implement the `OutputFormatter` protocol
7. Pipeline stages must extend `BaseStageExecutor` from `pipeline/stages.py`

### Pull request checklist

- [ ] All tests pass (`pytest`)
- [ ] Type checks pass (`mypy usability_oracle`)
- [ ] Linting passes (`ruff check usability_oracle`)
- [ ] New code has type annotations
- [ ] New public APIs have docstrings
- [ ] Mathematical claims include proof sketches or references
- [ ] Configuration changes are documented
- [ ] `ARCHITECTURE.md` updated if module structure changes

---

## License

This project is licensed under the MIT License.

```
MIT License

Copyright (c) 2024 Bounded-Rational Usability Oracle Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Citation

If you use this software in academic work, please cite:

```bibtex
@software{bounded_rational_usability_oracle,
  title     = {Bounded-Rational Usability Oracle: Information-Theoretic
               Cognitive Cost Analysis for CI/CD},
  author    = {Bounded-Rational Usability Oracle Team},
  year      = {2024},
  url       = {https://github.com/your-org/bounded-rational-usability-oracle},
  license   = {MIT},
  version   = {0.1.0},
}
```

### Related publications

The theoretical foundations draw on:

- **Bounded rationality:** Ortega, P.A. & Braun, D.A. (2013). Thermodynamics as a theory of decision-making with information-processing costs. *Proceedings of the Royal Society A*, 469(2153).
- **Fitts' Law as capacity:** MacKenzie, I.S. (1992). Fitts' law as a research and design tool in human-computer interaction. *Human-Computer Interaction*, 7(1), 91–139.
- **Multiple Resource Theory:** Wickens, C.D. (2002). Multiple resources and performance prediction. *Theoretical Issues in Ergonomics Science*, 3(2), 159–177.
- **CogTool:** John, B.E. & Salvucci, D.D. (2005). Multipurpose prototypes for assessing user interfaces in pervasive computing systems. *IEEE Pervasive Computing*, 4(4), 27–34.
- **Rate-distortion cognition:** Tishby, N., Pereira, F.C. & Bialek, W. (2000). The information bottleneck method. *Proceedings of the 37th Allerton Conference*.

---

*Built with ♥ and information theory.*
