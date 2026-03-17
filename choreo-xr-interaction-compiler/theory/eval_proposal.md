# Choreo Evaluation Proposal: Falsifiable Empirical Validation Plan

**Author**: Empirical Scientist Role  
**Stage**: Theory  
**Status**: Draft for peer review  
**Date**: 2026-03-08

---

## 0. Scope and Constraints

All experiments described in this document must be:

- **Fully automated**: no headsets, no human participants, no manual annotation.
- **Reproducible**: deterministic seeds, pinned dependencies, containerized execution.
- **CPU-only**: runnable on a single laptop (≥8 cores, ≥16 GB RAM, no GPU).
- **Falsifiable**: every hypothesis has explicit pass/fail thresholds.
- **Self-contained**: no dependency on proprietary XR runtimes at verification time (extraction is offline).

Evaluation code budget: ~10–13K LoC (subsystem 14 in impl_scope.md), comprising parametric scene generators, benchmark harnesses, metric collectors, and analysis scripts.

---

## 1. Hypotheses (Falsifiable)

### H1: Spatial CEGAR Finds Real Interaction Protocol Bugs

**Claim**: The spatial CEGAR verifier discovers genuine interaction protocol anomalies (deadlocks, unreachable states, non-deterministic transitions) in interaction choreographies extracted from open-source XR projects.

| Criterion | Threshold (PASS) | Marginal | Threshold (FAIL) |
|-----------|-------------------|----------|-------------------|
| Distinct anomalies found | ≥ 5 | 3–4 | < 3 |
| Corroborated by issue tracker | ≥ 2 of the anomalies match filed issues or developer-acknowledged bugs | 1 | 0 |
| False positive rate | ≤ 30% | 30–70% | > 70% |
| Unique bug classes | ≥ 3 distinct categories (e.g., deadlock, unreachable, race) | 2 | 1 |

**Corroboration protocol**: An anomaly is "corroborated" if (a) a matching GitHub issue exists in the source project, OR (b) we file an issue and a maintainer confirms within 90 days, OR (c) we construct a minimal reproducer that demonstrates the anomalous behavior in the original framework's simulator.

**Target projects for extraction** (selected for open-source availability, issue tracker activity, and interaction complexity):

| Project | Repository | Interaction Patterns | Est. Zones |
|---------|-----------|----------------------|------------|
| MRTK3 samples | microsoft/MixedRealityToolkit-Unity | Hand menu, near/far grab, solvers | 8–15 |
| MRTK2 examples | microsoft/MixedRealityToolkit-Unity (v2.x) | Bounding box, manipulation handler | 6–12 |
| XR Interaction Toolkit samples | Unity-Technologies/XR-Interaction-Toolkit-Examples | Socket, ray, direct interactors | 5–10 |
| Meta Interaction SDK samples | oculus-samples/Unity-* | Hand tracking, poke, grab | 8–14 |
| Godot XR Tools | GodotVR/godot-xr-tools | Snap zones, pickable, climber | 4–8 |

Minimum extraction target: **15 distinct interaction scenes** yielding ≥ 50 interaction patterns total.

### H2: Geometric Pruning Provides Meaningful State-Space Reduction

**Claim**: Geometric consistency pruning—eliminating predicate valuations unrealizable in any spatial configuration—achieves significant reduction of the abstract state space.

| Criterion | Threshold (PASS) | Marginal | Threshold (FAIL) |
|-----------|-------------------|----------|-------------------|
| Pruning ratio ≥ 3× | On ≥ 80% of benchmark scenes | 60–80% | < 50% of scenes |
| Pruning ratio ≥ 10× | On ≥ 50% of benchmark scenes | 30–50% | < 20% |
| Monotonicity in zone count | Pruning ratio non-decreasing as zones increase (5→50) | Holds on ≥ 70% of parameter sweeps | Decreasing on > 50% |
| Pruning preserves soundness | 0 missed bugs (vs. unpruned baseline) | — | Any missed bug |

**Pruning ratio** is defined precisely in §4.

### H3: Spatial Type Checker Rejects Ill-Formed Interaction Patterns

**Claim**: The spatial type checker (assuming T1 decidability result holds) catches interaction patterns with spatial-realizability violations—patterns that specify geometrically impossible configurations.

| Criterion | Threshold (PASS) | Marginal | Threshold (FAIL) |
|-----------|-------------------|----------|-------------------|
| Seeded violation catch rate | ≥ 80% of seeded spatial-realizability errors rejected at type-check time | 60–80% | < 60% |
| Well-typed acceptance rate | ≥ 95% of known-good patterns type-check successfully | 90–95% | < 90% |
| Error message actionability | ≥ 70% of error messages identify the violated spatial constraint | 50–70% | < 50% |
| Type-check latency | < 2 seconds per pattern (single-pattern type check) | 2–10 sec | > 10 sec |

**Seeded violation categories** (20 violations per category, 100 total):

1. **Volume impossibility**: Object A specified inside object B, but A's bounding volume exceeds B's interior.
2. **Proximity contradiction**: `Proximity(A, B, r₁)` and `Proximity(B, C, r₂)` with `¬Proximity(A, C, r₁ + r₂)`.
3. **Containment cycle**: `Inside(A, B) ∧ Inside(B, A)`.
4. **Gaze cone infeasibility**: Gaze cone specified with aperture/range making target geometrically unreachable.
5. **Temporal-spatial conflict**: Simultaneous spatial predicates requiring an object to be in two disjoint regions.

### H4: Verification Scales with Geometric Structure

**Claim**: Verification time scales gracefully with scene complexity when geometric pruning and spatial CEGAR are enabled, bounded by the geometric structure (not the raw state-space size).

| Criterion | Threshold (PASS) | Marginal | Threshold (FAIL) |
|-----------|-------------------|----------|-------------------|
| 20-zone scenes | Verify in < 60 sec (median over benchmark suite) | 60–120 sec | > 120 sec |
| 15-zone scenes | Verify in < 30 sec (median) | 30–60 sec | > 60 sec |
| 10-zone scenes | Verify in < 5 sec (median) | 5–15 sec | > 15 sec |
| Scaling exponent | Empirical time ∝ n^α with α < 4 (where n = zones) | α ∈ [4, 6] | α > 6 |
| 15-zone absolute ceiling | < 300 sec (worst case over all 15-zone benchmarks) | — | > 300 sec on any 15-zone scene |

**Scaling analysis method**: Fit `log(time) = α · log(zones) + β` via OLS on the parametric benchmark sweep. Report α with 95% CI.

### H5: Compositional Verification Extends Scalability

**Claim**: Spatial separability decomposition enables verification of scenes beyond the monolithic verifier's practical limit by exploiting tree-decomposable interaction structure.

| Criterion | Threshold (PASS) | Marginal | Threshold (FAIL) |
|-----------|-------------------|----------|-------------------|
| 50-zone scenes (treewidth ≤ 5) | Verify in < 300 sec | 300–600 sec | > 600 sec |
| Compositional speedup | ≥ 5× vs. monolithic on 30+-zone scenes | 2–5× | < 2× |
| Decomposition overhead | < 10% of total verification time | 10–25% | > 25% |
| Soundness preservation | 0 missed bugs vs. monolithic | — | Any missed bug |
| 100-zone scenes (treewidth ≤ 4) | Verify in < 600 sec | 600–1200 sec | > 1200 sec |

---

## 2. Benchmark Suite Design

### 2.1 Real-World Extraction Benchmarks (Category R)

**Purpose**: Validate that Choreo operates on real XR interaction patterns, not just synthetic ones.

**Extraction targets**:

| ID | Source Project | Scene | Expected Patterns | Expected Zones |
|----|---------------|-------|-------------------|----------------|
| R-01 | MRTK3 | HandMenu | 4–6 (open, close, select, pin) | 8 |
| R-02 | MRTK3 | NearFarGrab | 6–8 (approach, grab, move, release, throw) | 10 |
| R-03 | MRTK3 | BoundsControl | 5–7 (select, scale, rotate, translate) | 12 |
| R-04 | MRTK3 | Solvers | 3–5 (follow, orbital, surface mag.) | 6 |
| R-05 | MRTK2 | ManipulationHandler | 4–6 (one-hand, two-hand, constrained) | 8 |
| R-06 | MRTK2 | InteractableStates | 5–7 (focus, press, toggled, disabled) | 6 |
| R-07 | XRI Toolkit | SocketInteractor | 3–4 (hover, select, attach, detach) | 5 |
| R-08 | XRI Toolkit | RayInteractor | 4–5 (point, hover, select, activate) | 7 |
| R-09 | Meta SDK | HandGrab | 5–7 (approach, touch, grab, pinch, release) | 10 |
| R-10 | Meta SDK | PokeInteraction | 3–4 (approach, contact, press, release) | 6 |
| R-11 | Godot XR | SnapZone | 2–3 (highlight, snap, release) | 4 |
| R-12 | Godot XR | ClimbingMechanic | 3–4 (grab, pull, release, fall) | 5 |
| R-13 | MRTK3 | EyeTracking+Hand | 6–8 (gaze, dwell, confirm, dismiss) | 14 |
| R-14 | XRI Toolkit | Locomotion | 4–5 (teleport aim, confirm, snap-turn) | 8 |
| R-15 | MRTK3 | Dialog+Tooltip | 3–4 (appear, follow, dismiss, transition) | 6 |

**Total**: 15 scenes, ~60–85 interaction patterns, 5–14 zones each.

**Extraction fidelity validation**:
1. **Structural check**: Every MonoBehaviour/component in the source scene's interaction graph maps to exactly one Choreo pattern or zone declaration. Coverage ≥ 90%.
2. **Behavioral check**: For each extracted pattern, simulate 1,000 random traces in both the original framework (via headless Unity in batch mode) and the Choreo runtime. Trace divergence rate ≤ 5%.
3. **Manual audit (one-time)**: The first 5 extractions (R-01 through R-05) are manually verified against source code to calibrate the automated extraction pipeline. This is a one-time calibration cost, not a per-experiment requirement.

### 2.2 Parametric Synthetic Benchmarks (Category S)

**Purpose**: Controlled experiments isolating the effect of specific parameters on verification performance.

**Generation parameters**:

| Parameter | Values | Distribution |
|-----------|--------|-------------|
| Zone count (n) | {5, 8, 10, 15, 20, 30, 50, 75, 100} | Swept exhaustively |
| Patterns per zone (p) | {2, 4, 6, 10, 15, 20, 30} | Swept, cross-product with n |
| Spatial density (d) | {sparse=0.1, moderate=0.3, dense=0.6, full=0.9} | Fraction of zone pairs with spatial predicates |
| Hierarchy depth (h) | {flat=1, shallow=2, moderate=3, deep=5} | Containment tree depth |
| Treewidth (tw) | {2, 3, 5, 8, ∞} | Interaction graph treewidth (for H5) |
| Timing constraints | {none, loose=10s, tight=1s, mixed} | MTL constraint tightness |

**Scene generator** (`synth_gen`):
1. Generate a containment tree of depth `h` with `n` zone nodes.
2. Assign convex-polytope bounding volumes (axis-aligned boxes + random affine transforms) ensuring geometric realizability.
3. For each zone pair at density `d`, add a spatial predicate (Proximity, Inside, GazeAt) sampled uniformly.
4. Generate `p` interaction patterns per zone as random finite automata with 3–8 states, transitions guarded by the zone's spatial predicates.
5. Inject timing constraints at the specified tightness.
6. For treewidth-controlled scenes, generate the interaction graph via the treewidth-`tw` graph generator of Bodlaender & Koster (2008).

**Total synthetic benchmarks**:
- Primary sweep: 9 zone-counts × 4 densities × 4 hierarchy depths = **144 configurations**.
- At 5 random seeds per configuration = **720 benchmark instances**.
- Treewidth sweep (for H5): 9 zone-counts × 5 treewidths × 3 seeds = **135 instances**.
- Pattern-count sweep: 5 zone-counts × 7 pattern-counts × 3 seeds = **105 instances**.
- **Grand total**: ~960 synthetic benchmark instances.

### 2.3 Stress Benchmarks (Category X)

**Purpose**: Adversarial inputs designed to expose worst-case behavior.

| ID | Description | Expected Difficulty | Target Hypothesis |
|----|-------------|---------------------|-------------------|
| X-01 | Complete spatial graph: every zone pair has every predicate | Maximum predicate density | H2 (pruning under density) |
| X-02 | Chain topology: n zones in a linear chain, each zone sees only neighbors | Maximum diameter | H4 (scaling) |
| X-03 | Star topology: one central zone connected to n-1 peripheral zones | Maximum degree-1 node | H5 (composition) |
| X-04 | Nested containment: 10-deep nesting of zones | Maximum hierarchy depth | H2 (containment pruning) |
| X-05 | Predicate oscillation: patterns with rapid Proximity → ¬Proximity cycling | Maximum CEGAR iterations | H1, H4 (CEGAR termination) |
| X-06 | Near-deadlock: system with a single escape path from apparent deadlock | Hardest for incomplete methods | H1 (bug detection precision) |
| X-07 | Symmetric zones: n identical zones with identical patterns | Symmetry explosion | H4, H5 |
| X-08 | Contradictory constraints: deliberately unsatisfiable spatial layout | Must reject, not loop | H3 (type checking) |
| X-09 | Tight MTL: sub-second timing constraints on 20-zone scene | Temporal-spatial interaction | H4 |
| X-10 | Disconnected components: 5 independent 10-zone sub-scenes | Decomposition opportunity | H5 |

**Total stress benchmarks**: 10 instances × 3 scale variants (small, medium, large) = **30 instances**.

### 2.4 Regression Benchmarks (Category B — Known Bugs)

**Purpose**: Validate that Choreo's verifier detects known-to-be-real XR interaction bugs.

These are hand-encoded Choreo DSL specs corresponding to documented bugs in XR frameworks:

| ID | Bug Pattern | Source | Choreo Encoding |
|----|------------|--------|-----------------|
| B-01 | MRTK2 ManipulationHandler deadlock when two hands grab simultaneously | MRTK#7293 (or similar) | Two-hand simultaneous grab, shared state, no arbitration |
| B-02 | BoundsControl rotation lock fails under reparenting | MRTK issue tracker | Containment hierarchy change during active manipulation |
| B-03 | NearInteractionGrabbable unreachable state after teleport | Community report | Proximity predicate invalidated by non-continuous position change |
| B-04 | Tooltip follow solver oscillation | MRTK solver bug | Spatial predicate feedback loop creating livelock |
| B-05 | Socket interactor accepts two objects | XRI known limitation | Missing mutual exclusion on zone occupancy |
| B-06 | Ray interactor selection persists through occlusion | XRI behavior | Gaze-cone predicate not re-evaluated on occlusion change |
| B-07 | Hand menu opens during two-hand manipulation | MRTK UX conflict | Missing guard: menu-open requires ¬(either hand in grab state) |
| B-08 | Poke interaction double-fire on boundary | Meta SDK edge case | Non-deterministic transition at proximity boundary |
| B-09 | Snap zone detach race condition | Godot XR Tools | Two objects competing for same snap zone |
| B-10 | Eye-tracking dwell activates behind occluder | MRTK eye tracking | Gaze cone doesn't account for spatial occlusion |

**Total regression benchmarks**: **10 hand-encoded bug specifications**.

**Validation**: Each regression benchmark must:
1. Be flagged as a violation by the verifier (true positive).
2. Produce a counterexample trace that reproduces the bug.
3. The counterexample trace must be simulatable in the Choreo headless runtime.

---

## 3. Baseline Comparisons

### 3.1 iv4XR Agent-Based Exploration

**What**: iv4XR is a BDI agent framework for testing 3D virtual environments. We compare Choreo's formal verification against iv4XR's exploratory testing on the same interaction scenes.

**Setup**:
- Encode 10 representative benchmarks (R-01 through R-10) in both Choreo DSL and iv4XR agent scripts.
- iv4XR agents use the default exploration strategy (curiosity-driven) with a 600-second time budget per scene.
- Choreo runs spatial CEGAR with no time limit (but wall-clock is recorded).

**Metrics compared**:

| Metric | iv4XR Measurement | Choreo Measurement |
|--------|--------------------|--------------------|
| State coverage | Fraction of reachable abstract states visited by agents | States explored / total reachable states (from BDD) |
| Bugs found | Anomalies detected by agent assertions | Counterexamples produced by CEGAR |
| Wall-clock time | Agent exploration time to plateau | Verification time to completion |
| False positives | Agent-flagged anomalies that are not bugs | CEGAR counterexamples that are spurious |
| Coverage completeness | Can agents reach 100%? | Verification is exhaustive by construction |

**Expected outcome**: Choreo achieves exhaustive coverage; iv4XR achieves partial coverage but may find bugs faster in simple scenes.

### 3.2 UPPAAL Manual Encoding

**What**: UPPAAL is the state-of-the-art timed automata model checker. We compare the effort and capability of encoding XR interaction protocols as UPPAAL timed automata (without native spatial predicates) vs. Choreo DSL specs.

**Setup**:
- Select 10 representative protocols spanning the complexity range: R-01, R-02, R-05, R-07, R-09, B-01, B-03, B-05, B-08, S-{20-zone moderate}.
- Manually encode each as a UPPAAL timed automata network, abstracting spatial predicates as boolean variables with manual feasibility constraints.
- Run UPPAAL verification queries (deadlock-freedom, reachability) with default settings.

**Metrics compared**:

| Metric | UPPAAL | Choreo |
|--------|--------|--------|
| Encoding effort | Lines of UPPAAL XML / person-hours | Lines of Choreo DSL |
| Expressiveness | Can it represent the spatial predicate? (yes/no per predicate type) | Native support |
| Verification time | UPPAAL wall-clock | Choreo wall-clock |
| Bugs found | Violations found by UPPAAL | Violations found by Choreo |
| Spatial soundness | Are spatial infeasibility constraints correct? (manual audit) | Automatic geometric consistency |
| State space size | UPPAAL explored states | Choreo explored states (after pruning) |
| False positive rate | Spurious CEs due to missing spatial constraints | Spurious CEs |

**Expected outcome**: UPPAAL finds similar bugs on small scenes but requires 5–10× more encoding effort and misses spatially-infeasible counterexamples.

### 3.3 Random Simulation Baseline

**What**: Quantify the coverage gap between random testing and formal verification to establish the value-add of exhaustive checking.

**Setup**:
- For each benchmark in categories R, S (≤30 zones), and B: generate 10,000 random traces using the Choreo headless simulator.
- Each trace: random initial spatial configuration, random input events sampled uniformly, up to 500 steps.
- Record which abstract states are visited and which violations are triggered.

**Metrics compared**:

| Metric | Random Simulation | Choreo CEGAR |
|--------|-------------------|--------------|
| State coverage | Fraction of reachable states visited | Exhaustive |
| Bug detection recall | Fraction of known bugs (Category B) detected | Target: 100% |
| Time to first bug | Wall-clock to first violation | Wall-clock to first CE |
| Coverage plateau | Traces needed to stop discovering new states | N/A (complete) |

**Coverage gap** := (Choreo reachable states − random-visited states) / Choreo reachable states.

**Expected outcome**: Random simulation covers 40–70% of reachable states, misses subtle bugs (B-01, B-06, B-10), establishing the need for formal verification.

### 3.4 Ablation: No Geometric Pruning

**What**: Choreo with geometric consistency pruning disabled—all 2^|P| predicate valuations are treated as potentially realizable.

**Setup**: Run all Category S benchmarks (720 instances) and Category R benchmarks (15 instances) with and without pruning.

**Metrics**:
- Pruning ratio: states explored (pruned) / states explored (unpruned).
- Verification time ratio.
- Memory usage ratio.
- Soundness check: identical bug-finding results.

### 3.5 Ablation: No CEGAR (Naive BFS)

**What**: Replace CEGAR refinement loop with naive BFS exploration of the full abstract state space.

**Setup**: Run all benchmarks with naive BFS (with pruning still enabled, to isolate CEGAR's contribution).

**Metrics**:
- States explored: BFS total vs. CEGAR total.
- Counterexample quality: BFS finds shortest CE; CEGAR finds feasible CE. Compare CE length and spuriousness.
- Verification time.
- Scalability limit: largest scene verifiable in 600 seconds.

### 3.6 Ablation: No Compositional Decomposition

**What**: Choreo without spatial separability decomposition—monolithic verification only.

**Setup**: Run all Category S treewidth-sweep benchmarks (135 instances) and Category X stress benchmarks with decomposition enabled vs. disabled.

**Metrics**:
- Verification time ratio.
- Maximum verifiable scene size within 600-second budget.
- Decomposition overhead (time spent computing tree decomposition).

---

## 4. Metrics (Precise Definitions)

### 4.1 Pruning Ratio

**Definition**: Let P be the set of spatial predicates in a scene, and C ⊆ 2^P be the set of geometrically consistent predicate valuations (those realizable by some spatial configuration). The pruning ratio is:

```
pruning_ratio = |2^P| / |C|
```

A pruning ratio of k means the pruned state space is k× smaller than the unpruned space.

**Measurement method**:
1. Enumerate all predicate valuations (feasible for |P| ≤ 30; for larger |P|, estimate via sampling with Clopper-Pearson confidence interval).
2. For each valuation, invoke the geometric consistency checker (LP feasibility for convex polytopes).
3. Count feasible valuations → |C|.
4. Report pruning_ratio = 2^|P| / |C|.

For |P| > 30: sample 100,000 random valuations, count feasible fraction f, estimate |C| ≈ f · 2^|P|, report pruning_ratio ≈ 1/f with 95% Clopper-Pearson CI.

### 4.2 Verification Wall-Clock Time

**Definition**: Total elapsed time from invocation of the Choreo verifier to the production of a verdict (SAFE or COUNTEREXAMPLE), measured via `std::time::Instant` (Rust monotonic clock).

**What is included**: DSL parsing, type checking, Event Calculus lowering, automata compilation, R-tree construction, CEGAR loop (abstraction, model checking, refinement), result formatting.

**What is excluded**: Benchmark generation, extraction from source projects, result analysis.

**Reporting**: Median and 95th-percentile over random seeds for each configuration. Timeout: 600 seconds (results beyond timeout reported as ⊥).

### 4.3 State-Space Explored

**Definition**: The number of distinct abstract states `(q₁, ..., qₖ, v)` visited during verification, where `qᵢ` is the local state of automaton `i` and `v ∈ C` is the predicate valuation.

**Measurement**: Instrumented counter in the BDD exploration engine and CEGAR loop, incremented on each new state addition to the explored set.

### 4.4 Bug Detection Precision

**Definition**:

```
precision = TP / (TP + FP)
```

Where:
- **True Positive (TP)**: A counterexample trace that (a) violates the specification AND (b) is geometrically realizable (the spatial configuration sequence is feasible).
- **False Positive (FP)**: A counterexample trace that violates the specification but is geometrically infeasible (a spurious counterexample that survives CEGAR refinement—indicating a CEGAR incompleteness).

**Assessment protocol**: For each reported counterexample, run the geometric feasibility checker on every step of the trace. If any step has an infeasible spatial configuration, classify as FP.

### 4.5 Bug Detection Recall

**Definition**:

```
recall = TP / (TP + FN)
```

Where **False Negative (FN)** = bugs that exist but are not detected.

**Challenge**: The true bug count is unknown for real-world benchmarks.

**Estimation method**:
1. **Category B (regression)**: Known bugs → recall = (bugs detected) / 10.
2. **Category R (real-world)**: Use random simulation coverage gap as an upper bound on FN. If random simulation with 10,000 traces finds k bugs and Choreo finds k' ≥ k bugs, then recall ≥ k/k'. If Choreo finds bugs that random misses, that is additional evidence of recall superiority.
3. **Category S (synthetic, seeded bugs)**: Inject 5 bugs per synthetic instance at known locations. Recall = (detected seeded bugs) / (total seeded bugs).

### 4.6 Compilation Throughput

**Definition**: Patterns compiled per second at each phase of the compilation pipeline.

**Phases measured**:
1. Parse: DSL text → AST (patterns/sec).
2. Type check: AST → typed AST (patterns/sec).
3. EC lowering: typed AST → Event Calculus axioms (patterns/sec).
4. Automata compilation: EC axioms → spatial event automata (patterns/sec).
5. R-tree construction: spatial layout → indexed structure (zones/sec).

**Measurement**: Process each phase independently on the full Category S suite. Report median throughput and 5th-percentile (worst case).

### 4.7 Type Error Quality

**Definition**: The information content and actionability of type error messages produced by the spatial type checker.

**Measurement protocol** (automated, no human judges):

1. **Structural completeness** (binary per error): Does the error message contain (a) the violated constraint, (b) the involved zones/objects, (c) the conflicting predicate values? Score: fraction of errors with all three components.
2. **Localization accuracy** (binary per error): Does the error message point to the correct source location (line/column in DSL)? Score: fraction correctly localized.
3. **Minimal witness**: Does the error include a minimal spatial configuration witnessing the violation? Score: fraction that include a witness.

**Composite score**: Average of the three sub-scores, reported as a percentage.

### 4.8 Trace Fidelity (Cross-Platform Equivalence)

**Definition**: Given an interaction trace T = [(e₁, t₁, σ₁), ..., (eₖ, tₖ, σₖ)] where eᵢ is an event, tᵢ a timestamp, and σᵢ a spatial configuration, two executions are ε-equivalent if:

```
∀i: same event sequence AND |tᵢ - tᵢ'| ≤ ε_t AND ‖σᵢ - σᵢ'‖ ≤ ε_s
```

With ε_t = 50ms (temporal tolerance) and ε_s = 1cm (spatial tolerance).

**Measurement**: Run the same Choreo trace in the headless simulator and (where possible) in the Unity headless batch mode. Report the fraction of traces that are ε-equivalent.

**Scope**: This metric applies only to Category R benchmarks where the source framework supports headless execution.

---

## 5. Statistical Methodology

### 5.1 Sample Size Justification

**Parametric benchmarks** (Category S): 720 instances across 144 configurations with 5 seeds each. For the primary claim (H2: pruning ratio ≥ 3× on ≥ 80% of scenes), this provides:

- At the configuration level (n=144): a two-sided 95% CI on the proportion meeting the threshold has width ≤ ±8% (Clopper-Pearson).
- At the instance level (n=720): CI width ≤ ±3.5%.
- Power analysis: to detect a true proportion of 80% vs. a null of 50% with α=0.05, β=0.05, required n ≥ 30. We have 144 configurations, far exceeding this.

**Real-world benchmarks** (Category R): 15 scenes is small for parametric statistics. These are treated as a case study with per-scene reporting, not as a statistical sample. Aggregate statistics are reported with the caveat of small n.

### 5.2 Confidence Intervals

All timing and ratio measurements are reported with **bootstrapped 95% confidence intervals** (10,000 bootstrap replicates, percentile method).

**Protocol**:
1. For each configuration, collect 5 measurements (one per random seed).
2. Compute the median (primary statistic).
3. Bootstrap the median: resample with replacement 10,000 times, report 2.5th and 97.5th percentiles.

For pruning ratios (which can span orders of magnitude), report CIs on `log₂(pruning_ratio)` and back-transform.

### 5.3 Handling Variance Across Scene Structures

Different scene topologies (chain, star, tree, mesh) exhibit fundamentally different verification behavior. We handle this via **stratified analysis**:

1. **Report per-topology results separately** before aggregating.
2. Use **linear mixed-effects models**: `log(time) ~ zones + density + hierarchy + (1|topology)` to account for topology as a random effect.
3. For the scaling exponent (H4), fit separate regression lines per topology and report the range of α values.
4. Outlier handling: Winsorize at 5th/95th percentile. Report both Winsorized and raw results.

### 5.4 Multiple Comparisons

With 5 hypotheses and multiple metrics per hypothesis, we apply the **Holm-Bonferroni correction** when making pass/fail decisions across hypotheses. Individual metric p-values are reported uncorrected; hypothesis-level decisions use corrected thresholds.

### 5.5 Threats to Validity

#### Internal Validity

| Threat | Mitigation |
|--------|------------|
| Benchmark generator bias | Use multiple random seeds; audit generator for systematic biases in spatial layouts |
| Implementation bugs | Differential testing: compare CEGAR results against BFS on small instances |
| Measurement noise | Warm-up runs; pin CPU frequency; report median over 5 runs |
| Pruning ratio inflation | Verify that "unpruned" baseline is correctly computing the full 2^P space |
| Timeout bias | Report both median-over-completed and fraction-timed-out |

#### External Validity

| Threat | Mitigation |
|--------|------------|
| Synthetic benchmarks may not represent real XR scenes | Category R provides real-world grounding; report synthetic/real correlation |
| MRTK extraction fidelity | Manual audit of first 5 extractions; behavioral trace comparison |
| Open-source projects may not represent commercial XR apps | Acknowledged limitation; scope claim to "open-source XR interaction patterns" |
| Convex-polytope assumption may not hold | Gate 0 (week 1): audit MRTK volumes; if < 50% convex, trigger scope reduction |

#### Construct Validity

| Threat | Mitigation |
|--------|------------|
| "Bug" definition may be subjective | Define precisely: deadlock, unreachable state, non-deterministic resolution, livelock (cycle with no accepting state) |
| Pruning ratio may not predict practical speedup | Report both pruning ratio and wall-clock improvement; they need not correlate perfectly |
| Type error "quality" measurement is a proxy | Automated structural metrics are a lower bound on true quality; acknowledged limitation |

---

## 6. Concrete Experimental Protocol

### Experiment E1: Bug-Finding on Real-World Extractions (Tests H1)

**Input**: 15 extracted Choreo DSL specs (Category R), each with its source project's issue tracker URL.

**Process**:
```
for scene in R-01..R-15:
    choreo extract --source $scene.unity --output $scene.choreo
    choreo typecheck $scene.choreo                    # Record type errors
    choreo verify --property deadlock-free $scene.choreo  # Record CEs
    choreo verify --property all-reachable $scene.choreo
    choreo verify --property deterministic $scene.choreo
    for ce in counterexamples:
        choreo simulate --trace $ce --check-feasibility  # Filter FPs
    cross-reference violations against issue tracker
```

**Output collected**:
- Per-scene: number of anomalies, anomaly type, counterexample trace, feasibility verdict, issue tracker match (URL or "no match").
- Aggregate: total TP, FP, FN (estimated), precision, recall.

**Analysis**:
- Table: scene × anomaly-type matrix.
- Summary: "Found X anomalies across Y scenes, Z corroborated."
- Pass/fail against H1 thresholds.

**Presentation format**:

| Scene | Deadlocks | Unreachable | Non-det | Total | Corroborated | FP |
|-------|-----------|-------------|---------|-------|--------------|-----|
| R-01 | ... | ... | ... | ... | ... | ... |

### Experiment E2: Pruning Ratio Measurement (Tests H2)

**Input**: 720 Category S instances + 15 Category R instances.

**Process**:
```
for scene in benchmarks:
    # Unpruned: count all 2^|P| valuations
    unpruned_count = 2 ** scene.num_predicates
    # Pruned: count geometrically consistent valuations
    pruned_count = choreo count-consistent --scene $scene
    pruning_ratio = unpruned_count / pruned_count
    # Verify soundness: run both pruned and unpruned verification
    result_pruned = choreo verify --pruning=on $scene
    result_unpruned = choreo verify --pruning=off $scene
    assert result_pruned.bugs == result_unpruned.bugs  # Soundness
```

**Output**: Per-instance: |P|, |C|, pruning ratio, zone count, density, hierarchy depth. Aggregate: fraction meeting ≥3× threshold, stratified by topology.

**Presentation format**:

| Zones | Density | Hierarchy | Median Pruning Ratio [95% CI] | % ≥ 3× | % ≥ 10× |
|-------|---------|-----------|-------------------------------|---------|----------|
| 5 | sparse | flat | ... | ... | ... |

**Figure**: Log-scale scatter plot of pruning ratio vs. zone count, colored by density.

### Experiment E3: Verification Scaling (Tests H4)

**Input**: Category S zone-count sweep (5, 8, 10, 15, 20, 30, 50 zones), moderate density, moderate hierarchy, 5 seeds each.

**Process**:
```
for n in [5, 8, 10, 15, 20, 30, 50]:
    for seed in [1..5]:
        scene = synth_gen(zones=n, density=0.3, hierarchy=2, seed=seed)
        time, states, result = choreo verify --property deadlock-free $scene
        record(n, seed, time, states)
```

**Output**: Per-(n, seed): verification time, states explored, verdict. Aggregate: median time per zone-count, 95% CI, scaling exponent α.

**Analysis**:
1. Fit `log(time) = α · log(n) + β` via OLS. Report α with 95% CI.
2. Plot: log-log time vs. zones with regression line and CIs.
3. Report per-zone-count: median, p5, p95 time.
4. Pass/fail against H4 thresholds.

**Presentation format**:

| Zones | Median Time (s) [95% CI] | States Explored | Pass H4? |
|-------|---------------------------|-----------------|----------|
| 5 | ... | ... | ... |
| 10 | ... | ... | ... |
| 20 | ... | ... | ... |

**Figure**: Log-log plot with regression line, annotated with scaling exponent.

### Experiment E4: Compositional Verification (Tests H5)

**Input**: Category S treewidth sweep (135 instances) + stress benchmarks X-03, X-07, X-10.

**Process**:
```
for tw in [2, 3, 5, 8, INF]:
    for n in [5, 10, 20, 30, 50, 75, 100]:
        for seed in [1..3]:
            scene = synth_gen(zones=n, treewidth=tw, ...)
            time_mono = choreo verify --compositional=off $scene
            time_comp = choreo verify --compositional=on $scene
            record(n, tw, seed, time_mono, time_comp)
```

**Output**: Per-instance: monolithic time, compositional time, speedup, decomposition overhead. Aggregate: speedup by (n, tw).

**Presentation format**:

| Zones | Treewidth | Mono Time (s) | Comp Time (s) | Speedup | Decomp Overhead |
|-------|-----------|----------------|----------------|---------|-----------------|
| 50 | 3 | ... | ... | ... | ... |

**Figure**: Heatmap of speedup (zones × treewidth).

### Experiment E5: Type Checker Validation (Tests H3)

**Input**: 100 seeded violations (20 per category, §1 H3) + 60–85 known-good patterns from Category R.

**Process**:
```
for violation in seeded_violations:
    result = choreo typecheck $violation.choreo
    record(violation.id, violation.category, result.rejected?, result.error_msg)
for pattern in known_good_patterns:
    result = choreo typecheck $pattern.choreo
    record(pattern.id, result.accepted?)
```

**Output**:
- Seeded violations: catch rate per category and overall.
- Known-good: false rejection rate.
- Error quality: structural completeness, localization, witness scores.

**Presentation format**:

| Violation Category | Seeded | Caught | Catch Rate |
|-------------------|--------|--------|------------|
| Volume impossibility | 20 | ... | ...% |
| Proximity contradiction | 20 | ... | ...% |
| Containment cycle | 20 | ... | ...% |
| Gaze infeasibility | 20 | ... | ...% |
| Temporal-spatial conflict | 20 | ... | ...% |
| **Total** | **100** | ... | ...% |

### Experiment E6: Baseline Comparisons (Tests H1–H4 Comparatively)

**Input**: 10 selected benchmarks per baseline (see §3).

**Process**: As described in §3 for each baseline. All baselines run on the same machine with the same time budget (600 seconds).

**Presentation format**:

**Table: Choreo vs. Baselines (Summary)**

| Metric | Choreo | iv4XR | UPPAAL | Random | No-Prune | No-CEGAR | No-Comp |
|--------|--------|-------|--------|--------|----------|----------|---------|
| Bugs found | ... | ... | ... | ... | ... | ... | — |
| FP rate | ... | ... | ... | ... | ... | ... | — |
| Coverage | 100% | ...% | 100% | ...% | 100% | 100% | 100% |
| Med. time (s) | ... | ... | ... | ... | ... | ... | ... |
| Encoding effort | ... | N/A | ...× | N/A | — | — | — |

### Experiment E7: Regression Suite (Tests H1 Recall)

**Input**: 10 Category B bug specifications.

**Process**:
```
for bug in B-01..B-10:
    result = choreo verify --property all $bug.choreo
    assert result.verdict == COUNTEREXAMPLE
    choreo simulate --trace result.ce --animate  # Verify CE is meaningful
    record(bug.id, detected?, ce_length, ce_feasible?)
```

**Output**: Per-bug: detected (yes/no), counterexample length, feasibility. Aggregate: regression recall = detected / 10.

**Pass criterion**: 10/10 detected (this is a regression suite, not a statistical sample).

---

## 7. Kill Criteria

### Per-Hypothesis Abandon Triggers

| Hypothesis | ABANDON Trigger | Scope Reduction Trigger | Contingency |
|------------|-----------------|-------------------------|-------------|
| **H1** (bug-finding) | < 3 anomalies found across all 15 real-world scenes AND < 7/10 regression bugs detected | 3–4 anomalies found but 0 corroborated; OR FP rate 50–70% | Pivot to synthetic-only evaluation; reframe as "verification methodology" paper without empirical bug-finding claims |
| **H2** (pruning) | Pruning ratio < 2× on > 50% of Category S instances | Pruning ≥ 3× on only 50–79% of instances | Restrict claim to specific scene topologies where pruning is effective; add topology precondition to theorem statement |
| **H3** (type checking) | Catch rate < 60% on seeded violations OR decidability proof (T1) fails | Catch rate 60–79% | If T1 fails: drop type-checking hypothesis entirely, remove from paper, present pruning + CEGAR as the contributions. If catch rate marginal: restrict to the violation categories where it works |
| **H4** (scaling) | 15-zone scenes take > 300 seconds (median) | 20-zone scenes > 120 seconds but 15-zone < 120 seconds | Reduce claimed scalability threshold; focus paper on quality of bugs found (H1) rather than scale |
| **H5** (composition) | Compositional verification provides < 2× speedup on scenes with treewidth ≤ 5 | 50-zone scenes > 600 seconds with composition | Drop H5 from paper; present composition as future work. Paper survives on H1–H4 |

### Early Gate Kill Decisions (Before Full Evaluation)

These align with the kill gates defined in the final approach document:

| Gate | Timing | Check | Kill Action |
|------|--------|-------|-------------|
| **Gate 0** | Week 1 | Convexity audit: ≥ 50% of MRTK bounding volumes are convex polytopes | If < 50%: extend type system to handle non-convex volumes (scope increase) or restrict to axis-aligned boxes (scope decrease) |
| **Gate 1** | Week 3 | Extract 3 MRTK scenes, find ≥ 1 known bug via manual analysis | If 0 bugs in any of 3 scenes: extraction pipeline is fundamentally flawed; reassess extraction approach |
| **Gate 2** | Week 2 | Treewidth ≤ 5 for ≥ 80% of real-world interaction graphs | If < 80%: drop H5 (compositional verification) from evaluation; it relies on low treewidth |
| **Gate 3** | Month 4 | Decidability proof sketch for spatial subtyping | If no proof sketch: drop H3 and T1 from paper; accept B-grade portfolio |
| **Gate 4** | Month 8 | Bug-finding checkpoint: ≥ 3 anomalies, FP ≤ 60% | If fails: switch to CAV/TACAS algorithm paper (minimal publishable unit) |

### Cascading Failure Scenarios

**Scenario A: Bug-finding underperforms (H1 marginal or fails)**
1. Check if extraction fidelity is the bottleneck (trace divergence > 5%).
2. If extraction is sound but bugs are scarce: the XR frameworks are more robust than expected. Reframe: "Choreo as a certification tool" — proving absence of bugs is also a result.
3. If extraction is lossy: report what fraction of interaction patterns survive extraction and focus evaluation on those.
4. Fallback evaluation: Run Category B regression suite as the primary bug-finding evidence. If 10/10 detected, the verifier works—real-world projects are just well-tested.

**Scenario B: Pruning disappoints (H2 fails)**
1. Check if the pruning ratio measurement is correct (implementation bug?).
2. If correct: the convex-polytope fragment doesn't provide enough pruning. Check if non-convex extensions (OBB trees, convex decomposition) improve it.
3. If pruning is inherently weak: drop pruning as a contribution, focus on CEGAR refinement (H1, H4 without pruning).

**Scenario C: Type checker decidability fails (T1, H3 abandoned)**
1. This is anticipated (45% risk).
2. Paper proceeds with T2, T3, T4 contributions and H1, H2, H4, H5 evaluation.
3. Type checker becomes a best-effort tool (semi-decision procedure with timeout) rather than a decidable algorithm.
4. Remove H3 from evaluation entirely.

**Scenario D: Nothing works (H1–H4 all fail)**
1. Probability: < 5% (H2 and H4 are low-risk).
2. Action: Publish the negative result. "Why Formal Verification of XR Interactions Is Harder Than It Looks."
3. Alternative: Extract the geometric CEGAR algorithm as a standalone contribution and submit to CAV/TACAS without the XR application story.

---

## 8. Resource Budget

| Resource | Budget | Notes |
|----------|--------|-------|
| Benchmark generation + infrastructure | ~3K LoC (Python) | Scene generators, harness scripts |
| Metric collection + analysis | ~2K LoC (Python) | Timing, state counting, aggregation |
| Baseline implementations | ~3K LoC | iv4XR adapter, UPPAAL encodings, random sim driver |
| Visualization + reporting | ~1K LoC (Python) | Tables, plots (matplotlib/seaborn) |
| MRTK extraction scripts | ~2K LoC (Python + C#) | AST parsing, scene graph traversal |
| **Total evaluation code** | **~11K LoC** | Within the 10–13K budget |

| Compute resource | Estimate |
|------------------|----------|
| Full Category S sweep (720 instances, 5 seeds) | ~100 CPU-hours (assuming median 5 min/instance) |
| Category R extraction + verification | ~10 CPU-hours |
| Baseline comparisons | ~50 CPU-hours |
| Total | ~160 CPU-hours (~7 days on 1 machine) |

---

## 9. Deliverables

| Deliverable | Format | Location |
|------------|--------|----------|
| Benchmark suite (all categories) | Choreo DSL files + generator scripts | `eval/benchmarks/` |
| Raw experimental data | CSV per experiment | `eval/data/` |
| Analysis scripts | Python + SQL | `eval/analysis/` |
| Result tables + figures | LaTeX + PDF | `eval/results/` |
| Reproduction instructions | Makefile + README | `eval/README.md` |
| Per-hypothesis verdict | JSON summary | `eval/verdicts.json` |

All experiments are orchestrated by a single entry point:

```bash
make -C eval run-all     # Run all experiments (~160 CPU-hours)
make -C eval run-quick   # Run Category R + Category B only (~2 CPU-hours)
make -C eval analyze     # Generate tables and figures from collected data
make -C eval verdict     # Print pass/fail for each hypothesis
```

---

## 10. Summary: Hypothesis → Experiment → Kill Decision Map

```
H1 (bug-finding)   ← E1 (real-world) + E7 (regression) + E6 (baselines)
                      KILL if: <3 anomalies AND <7/10 regression
                      SCOPE REDUCE if: 0 corroborated OR FP >50%

H2 (pruning)       ← E2 (pruning measurement) + E6 (no-prune ablation)
                      KILL if: <2× on >50% of instances
                      SCOPE REDUCE if: <3× on >20% of instances

H3 (type checking) ← E5 (seeded violations) + Gate 3 (decidability)
                      KILL if: T1 fails OR catch rate <60%
                      SCOPE REDUCE if: catch rate 60-79%

H4 (scaling)       ← E3 (scaling sweep) + E6 (no-CEGAR ablation)
                      KILL if: 15-zone >300s
                      SCOPE REDUCE if: 20-zone >120s

H5 (composition)   ← E4 (treewidth sweep) + E6 (no-comp ablation) + Gate 2
                      KILL if: <2× speedup at tw≤5
                      SCOPE REDUCE if: 50-zone >600s
```

**Minimum viable evaluation** (if all stretch goals fail): H1 + H2 + H4 using Categories R, B, and a reduced Category S sweep (~200 instances). This is sufficient for a systems paper at OOPSLA/PLDI with bug-finding + pruning contributions.
