# Algebraic Repair Calculus: Provably Correct Incremental Maintenance of Data Pipelines Under Schema Evolution, Quality Drift, and Partial Outages

## Problem and Approach

Modern data pipelines are brittle. When an upstream Postgres table adds a column, a Spark job's input schema silently shifts, or a third-party API begins returning nulls where it shouldn't, the downstream cascade of failures is catastrophic. Industry surveys report that data engineers spend 40–60% of their time on pipeline maintenance—not building new features, but firefighting breakages. Today's response is crude: detect the failure (often hours later via stale dashboards), manually diagnose the root cause by tracing lineage, write a one-off fix, and pray it doesn't break something else. There is no formal framework for reasoning about pipeline perturbations, no algebra for composing repairs, and no system that can automatically compute a provably correct repair plan.

We propose a **dataflow repair engine** grounded in a novel **three-sorted delta algebra** that treats schema evolution, data-quality drift, and partial source outages as first-class algebraic objects—**perturbation deltas**—over typed dependency graphs extracted by static analysis from real SQL and Python ETL code. The engine statically analyzes SQL (PostgreSQL, Spark SQL) and Python data transformations (pandas, PySpark, dbt models) to construct a **typed pipeline dependency graph** where nodes carry schema types, quality invariants, and availability contracts. When a perturbation arrives, the delta algebra computes its propagation through the graph, and a **cost-optimal repair planner** synthesizes the minimal-cost incremental plan that restores all downstream invariants—provably correct with respect to full recomputation while executing only the necessary subset of transformations.

The intellectual core is the **three-sorted delta algebra** Δ = (Δ_S, Δ_D, Δ_Q, ∘, ⁻¹, push), where Δ_S is the monoid of schema changes (column additions, type widenings, renames), Δ_D is the group of data-level changes (inserts, deletes, updates, including corrections to quality violations), and Δ_Q is the lattice of quality-contract changes (tightened null constraints, new range bounds, shifted distributions). These three sorts interact via **interaction homomorphisms**: a schema delta φ(δ_s) lifts into a data-delta transformer (new columns need defaults), and ψ(δ_s) lifts into a quality-delta transformer (new columns need quality contracts). The algebra defines composition (δ₁ ∘ δ₂), inversion (δ⁻¹), and propagation (push_f(δ) through transformation f) operators with algebraic laws guaranteeing associativity, identity, and—critically—a **bounded commutation theorem**: for any pipeline *G* in the deterministic, order-independent fragment *F* and perturbation *σ*, `apply(repair(σ), state(G)) = recompute(evolve(G, σ))`—incremental repair yields the same pipeline state as full recomputation. For pipelines outside *F* (containing floating-point aggregations, non-deterministic functions, or external calls), we provide a computable deviation bound *ε(G, σ)*.

This is a fundamental departure from classical incremental view maintenance (IVM), which assumes fixed schemas and handles only data deltas. Systems like DBSP (Budiu et al., VLDB 2023) and Differential Dataflow (McSherry et al., CIDR 2013) provide elegant algebraic frameworks for data-level incrementality but cannot express "the upstream table's schema changed" or "the data quality contract was violated." Schema evolution has been studied separately (Curino et al.'s PRISM), as has data quality monitoring, and pairwise combinations exist in the literature. Our system is the first to formalize all three perturbation classes in a single algebraic framework with cross-sort interaction homomorphisms, prove an **encoding impossibility theorem** showing that no data-domain encoding of schema deltas into DBSP preserves both type safety and incrementality, prove that cost-optimal repair is NP-hard in general but polynomial for acyclic pipeline topologies (which cover >90% of real pipelines), and build a system that exploits this theory for provably correct repairs with significant cost savings over recomputation.

## Value Proposition

**Who needs this.** Every organization running data pipelines at scale—thousands of companies with data engineering teams of 5–500 people. Schema evolution is the #1 cause of pipeline breakage (per dbt Labs surveys, Fivetran incident reports). Today, the response to schema changes is manual: an engineer traces the lineage, writes a migration, tests it, and deploys. For a 200-table warehouse with weekly schema changes, this consumes 2–3 FTE permanently.

**What becomes possible.** (1) **Provably correct** repair plans generated in seconds—the repaired pipeline is formally guaranteed to produce the same output as full recomputation for deterministic pipelines, eliminating the "did my fix break something else?" anxiety that dominates on-call experience. (2) **Compound perturbation handling**: when a schema change and a quality violation arrive simultaneously, the interaction homomorphisms correctly compose the repairs—a capability no existing tool or LLM-based approach provides. (3) Cost-optimal repairs that touch only the minimal set of downstream transformations, reducing recomputation costs by 2–5× over best-practice selective recomputation (dbt `--select` with lineage awareness) and 10–50× over full recomputation. (4) A mathematical framework that makes pipeline maintenance a principled engineering discipline rather than an art.

**Why desperately.** Pipeline breakage is not an edge case—it is the dominant operational cost of modern data infrastructure. Today's mitigations are duct tape: schema registries detect changes but don't repair downstream consumers; quality-monitoring tools (Great Expectations, Monte Carlo) flag anomalies but offer no automated resolution path; incremental-view-maintenance engines (Materialize, Noria) assume fixed schemas; dbt can re-run models but provides no correctness guarantees for selective recomputation. LLM-based tools can suggest repairs for simple perturbations (column adds, renames) but cannot guarantee correctness, cannot handle compound perturbations, and cannot provide cost-optimal plans. The key missing piece: given a perturbation, what is the provably correct and cost-optimal repair? This gap costs the industry hundreds of millions of dollars annually in wasted compute, secondary breakages, and engineering time.

## Technical Difficulty

The system requires novel contributions in five hard areas, built on existing libraries (sqlglot for SQL parsing, scipy for statistical tests, networkx for graph operations, DuckDB for evaluation).

**Scope tiers** ensure a publishable result at every level:

- **Tier 1 — Minimum Viable Paper (~35K LoC):** Two-sorted algebra (Δ_S, Δ_D) over SQL-only pipelines, commutation theorem for the deterministic fragment, DP algorithm for acyclic pipelines, DBSP encoding impossibility separation, TPC-DS evaluation at SF=10.
- **Tier 2 — Full Paper (~61K LoC):** Three-sorted algebra (+Δ_Q), Python idiom matching, cost-optimal planning with LP relaxation approximation, full evaluation suite.
- **Tier 3 — Stretch (~75K LoC):** Additional SQL dialects (MySQL, BigQuery), streaming quality monitor, saga-based executor with compensating actions.

1. **SQL static analysis on sqlglot** (7K LoC): Semantic column-level lineage extraction for PostgreSQL and Spark SQL, built on sqlglot's multi-dialect parser. Custom semantic visitors handle CTEs, correlated subqueries, window functions, and lateral joins. The architecture supports additional dialects via sqlglot's AST with only front-end parser changes, not algebra or planner modifications.

2. **Python idiom-matching analyzer** (5K LoC): Pattern-matching on 15–20 common pandas/PySpark operations (rename, drop, merge, groupby, pivot, melt, assign, filter, join, concat, apply-with-known-schema). Operations not matching a known idiom are treated conservatively as opaque (all-columns-depend-on-all-columns). This is explicitly NOT full abstract interpretation—it is pragmatic idiom coverage targeting ≥85% of column-level dependencies in real dbt projects, validated against a corpus of 500 public dbt repositories. UDFs and dynamic column creation are flagged for conservative handling.

3. **Three-sorted delta algebra engine** (9K LoC): The novel delta algebra with composition, inversion, propagation, and cross-sort interaction homomorphisms. The engine computes delta propagation through pipeline DAGs while maintaining algebraic invariants. Implemented as a term-rewriting system with the cost model for repair planning integrated at the algebra level.

4. **Cost-optimal repair planner** (6K LoC): (a) A polynomial-time O(|V| · k²) dynamic-programming algorithm for acyclic pipelines (the practically dominant case covering >90% of real pipelines) that finds the exact cost-optimal repair plan. (b) An (ln k + 1)-approximation via LP relaxation with randomized rounding for general topologies. (c) An optional ILP exact solver (via HiGHS) for small cyclic instances, with timeout fallback to the approximation algorithm. The cost model accounts for data volume, transformation complexity, and delta annihilation (stages where the propagated delta is provably zero).

5. **Pipeline type system** (5K LoC): A refinement type system where types carry schema structure and quality predicates (non-null, positivity, range bounds). Schema deltas induce type-level transformations; type checking ensures repair plans produce well-typed outputs at every stage. Subtyping handles schema widening. Statistical distribution constraints are handled at the monitoring layer (runtime), not the type layer (static)—this is a deliberate separation of concerns.

6. **Repair executor with saga-based consistency** (5K LoC): Executing repair plans with pipeline-level consistency guarantees—no downstream consumer sees a partially-repaired state. Uses a saga pattern with compensating actions, checkpoint/rollback support, and partial-failure recovery. Explicitly provides eventual consistency, not ACID guarantees, acknowledging that heterogeneous backends (databases, files, APIs) have different transactional capabilities.

7. **Data quality monitoring** (4K LoC): Statistical drift detection (KS tests, PSI, Wasserstein distance via scipy.stats) with streaming wrappers for bounded-memory operation. Quality deltas are inferred from observed distribution shifts and integrated into the delta algebra. False-positive control via Bonferroni correction with configurable significance thresholds.

**Subsystem breakdown** (~61K LoC, Tier 2):

| Subsystem | LoC | Difficulty | Key Library |
|---|---|---|---|
| SQL Semantic Analyzer (2 dialects) | 7,000 | Hard | sqlglot |
| Python Idiom Matcher (pandas/PySpark) | 5,000 | Medium-Hard | ast (stdlib) |
| Typed Dependency Graph | 5,000 | Medium | networkx |
| Refinement Type System | 5,000 | Hard | — |
| Delta Algebra Engine | 9,000 | Very Hard | — |
| Repair Planner | 6,000 | Very Hard | scipy, HiGHS |
| Repair Executor | 5,000 | Medium-Hard | — |
| Data Quality Monitor | 4,000 | Medium | scipy.stats |
| Pipeline State Management | 3,000 | Medium | SQLite |
| Evaluation & Test Infrastructure | 12,000 | Medium | DuckDB, hypothesis |

## New Mathematics Required

Three genuinely novel mathematical contributions, two solid applications of standard techniques, and one required infrastructure result:

### Genuinely Novel

1. **Three-Sorted Delta Algebra with Interaction Homomorphisms.** Define the algebraic structure *(Δ_S, Δ_D, Δ_Q, ∘, ⁻¹, push)* where Δ_S is the **monoid** of schema deltas, Δ_D is the **group** of data deltas, and Δ_Q is the **lattice** of quality deltas. The key novel structure is the **interaction homomorphism** *φ: Δ_S → (Δ_D → Δ_D)* that lifts schema changes into data-delta transformers, and the analogous *ψ: Δ_S → (Δ_Q → Δ_Q)* for quality. While each individual sort draws on known algebraic structures (monoids from schema evolution, groups from IVM, lattices from abstract interpretation), the cross-sort interaction homomorphisms and the **propagation coherence lemma** (proving that delta propagation through pipeline stages preserves interaction laws) are genuinely unprecedented.

2. **Bounded Commutation Theorem.** For any pipeline DAG *G* in the deterministic, order-independent fragment *F* (formally: all stages are pure functions over bag semantics with exact arithmetic), source perturbation *σ = (δ_s, δ_d, δ_q)*, and repair plan *R* produced by the planner: `apply(R, state(G)) = recompute(evolve(G, σ))`. For pipelines outside *F*, we provide a **bounded deviation guarantee**: `distance(apply(R, state(G)), recompute(evolve(G, σ))) ≤ ε(G, σ)` where *ε* is computable from the pipeline's non-determinism profile (number of floating-point aggregations, non-deterministic function calls). The proof proceeds by structural induction on *G*, with the key inductive step requiring the interaction homomorphism to commute with stage-local computation—a property that holds exactly for *F* and approximately outside it. Precisely characterizing *F* and proving the bound on *ε* is a non-trivial contribution that preempts the obvious reviewer objection.

3. **DBSP Encoding Impossibility Theorem.** A formal proof that no encoding of schema deltas into DBSP's data domain preserves both type safety and incrementality simultaneously. Specifically: DBSP circuits are parametric in data (Z-sets over a fixed tuple type *T*) but not in schema (*T* is fixed at circuit construction time). Encoding schema changes as data deltas over a meta-table forces either (a) a universal tuple type (Map<String, Any>) that destroys the type-directed optimization DBSP relies on, or (b) circuit reconstruction that sacrifices incrementality (equivalent to full recomputation for the affected subgraph). The proof shows that the three-sorted algebra avoids this dilemma because schema deltas are first-class objects with their own propagation rules. This is a non-trivial theoretical contribution—not merely showing that DBSP lacks a feature, but proving that no data-domain workaround can substitute for first-class schema deltas.

### Solid Applications of Standard Techniques

4. **Repair Complexity Classification.** Optimal repair plan synthesis is NP-hard by reduction from weighted set cover (standard technique). The tractable fragment: acyclic pipeline topologies admit optimal repair in *O(|V| · k²)* time via dynamic programming, where *k* is the maximum fan-out. For general topologies, an *(ln k + 1)*-approximation via LP relaxation with randomized rounding (inheriting the tight bound from set cover). Since >90% of real pipelines are DAGs by construction, the polynomial algorithm is practically dominant. The novel element is the specific reduction and the DP algorithm over the delta-algebraic cost model, not the proof techniques.

5. **Type Soundness for Pipeline Types.** Define a type judgment Γ ⊢ stage : τ_in → τ_out where types τ encode (schema, quality-predicate) pairs. Prove **progress** (well-typed repair plans always have a next step) and **preservation** (applying one repair step to a well-typed state yields a well-typed state). This is standard PL metatheory (Wright-Felleisen) applied to the novel pipeline type system. The contribution is the type system design, not the proof technique.

### Required Infrastructure

6. **Idiom Coverage Lemma.** For the set of 15–20 supported pandas/PySpark idioms, prove that the idiom matcher's dependency extraction is sound (no false negatives for matched idioms) and measure empirical coverage against a corpus of 500 public dbt projects (target: ≥85% of column-level dependencies correctly captured). For unmatched idioms, the conservative all-to-all fallback guarantees soundness at the cost of precision. This replaces the original abstract interpretation soundness claim with an honest, empirically grounded result.

## Best Paper Argument

This paper merits best paper consideration on four axes. First, it **solves the right problem with formal guarantees**: pipeline maintenance is the #1 pain point in data engineering, and no existing system provides correctness guarantees for repair—this is the "garbage collection" moment for data pipelines, where formal methods replace ad-hoc manual intervention. The bounded commutation theorem provides the correctness guarantee that no competitor (including LLM-based tools) can offer. Second, the **interaction homomorphisms and DBSP encoding impossibility** are genuine mathematical novelties—the interaction homomorphisms are a new algebraic structure with no prior analog, and the impossibility theorem shows this isn't just a matter of adding features to DBSP but a fundamental architectural difference. Third, the **clean complexity dichotomy** (NP-hard in general, *O(|V|·k²)* for the practically-dominant acyclic case) is the kind of result that theorists and practitioners both appreciate. Fourth, it **unifies three fragmented subfields**: schema evolution (SIGMOD lineage), data quality (VLDB monitoring), and incremental view maintenance (PODS/SIGMOD theory) have been studied as separate communities. This work demonstrates they are three faces of a single algebraic structure, connected by interaction homomorphisms—extending a 40-year line of IVM research (Gupta & Mumick 1995, Nikolic et al. 2018, Budiu et al. 2023).

## Evaluation Plan

All evaluation is fully automated with zero human involvement. The benchmark harness generates perturbations, executes all systems, and collects metrics programmatically.

**Benchmarks:**
- **TPC-DS Pipeline Corpus**: 99 TPC-DS queries organized into 15 ETL pipeline DAGs with realistic transformation chains (SF=10 default, SF=100 as stretch). Schema evolution traces generated by systematically applying column additions, type changes, renames, and drops.
- **Synthetic Perturbation Suites**: 500 randomly generated pipeline topologies (10–500 nodes, including chain, tree, DAG, and DAG-with-cycles topologies) with parameterized perturbation sequences covering all combinations of schema/data/quality deltas. Stress tests at 200, 500, and 1000 stages measure planner scalability.
- **Real-World Schema Evolution Traces**: Extracted from public schema migration histories in open-source projects (Rails migrations from GitHub, Liquibase changelogs, Alembic migrations).
- **Python Idiom Coverage Corpus**: 500 public dbt projects with pandas/PySpark transformations, measuring idiom matcher recall against ground-truth dependencies from instrumented execution.

**Baselines** (5):
1. *Full Recomputation*: Tear down and rebuild all affected downstream nodes (secondary baseline — the worst case).
2. *Lineage-Aware Selective Recomputation*: Simulate best-practice dbt `--select` with column-level lineage awareness—recompute only nodes with affected input columns. **This is the primary baseline** representing the current state of practice.
3. *DBSP-style Data-Only IVM*: Apply DBSP's algebra ignoring schema/quality deltas (requires manual schema migration first). Exposes the expressiveness gap.
4. *Naive Incremental*: Recompute only directly-affected nodes without transitive optimization or delta annihilation.
5. *Rule-Based Repair*: Hand-written heuristic rules (column added → add default; type changed → cast) as in sqlmesh-style tools.

**Metrics:**
- **Correctness**: 100% semantic equivalence with full recomputation for deterministic pipelines; bounded deviation for non-deterministic pipelines (automated diff of all pipeline outputs).
- **Repair Cost vs. Selective Recomputation**: Cost(algebraic repair) / Cost(lineage-aware selective recompute) — target <0.5 (2× improvement over best practice). Report Cost(algebraic repair) / Cost(full recompute) as secondary metric.
- **Correctness Guarantee Coverage**: Fraction of pipeline stages for which the commutation theorem provides exact guarantees (target: >90% of SQL-only stages).
- **Planning Time**: Wall-clock time to synthesize repair plan — target <1 second for 95th percentile.
- **Perturbation Coverage**: Fraction of perturbation types handled correctly — target 100% of typed perturbations.
- **Algebra Verification**: Automated property-based testing (Hypothesis) of all algebraic laws (associativity, identity, interaction-homomorphism coherence, commutation) over 10M random delta combinations.
- **Idiom Matcher Recall**: Fraction of true column-level dependencies captured by the Python idiom matcher vs. ground-truth traces — target ≥85%.
- **SQL Analysis Recall**: Fraction of true column-level dependencies captured for SQL stages vs. ground-truth — target ≥95%.
- **End-to-End Latency**: Time from perturbation detection to full consistency restoration.

## Laptop CPU Feasibility

This system is inherently CPU-friendly. **Static analysis** is parsing (via sqlglot) plus idiom matching and semantic visitors—tree traversals, entirely CPU-bound. The **delta algebra** is symbolic term rewriting—pattern matching and algebraic simplification with no numerical computation requiring GPUs. **Repair planning** is combinatorial optimization (DP for acyclic, LP relaxation for general)—solver workloads native to CPU. **Quality monitoring** uses streaming statistics (KS tests, histogram comparisons via scipy)—lightweight numerical computations. **No ML/training**: the system uses formal methods exclusively; there are no models to train, no embeddings to compute, no inference to run.

The entire evaluation suite targets pipelines of 10–500 nodes with schemas of 10–200 columns—well within laptop memory and CPU budgets. TPC-DS at SF=10 fits comfortably in 16GB RAM; SF=100 is included as a stretch goal for scalability testing. Full benchmarks complete in 6–8 hours on a modern laptop (8-core, 16GB RAM). ILP solver instances include a configurable timeout (default: 30 seconds) with fallback to LP relaxation approximation.

## Slug

`algebraic-repair-calculus`
