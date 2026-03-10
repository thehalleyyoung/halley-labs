# Competing Approaches: Algebraic Repair Calculus

Three genuinely different approaches to building a provably correct incremental repair engine for data pipelines under schema evolution, quality drift, and partial outages.

---

## Approach A: "Algebraic-First" — The Algebra IS the Product

### Core Strategy

Push the three-sorted delta algebra to its theoretical limits: full axiomatization of pipeline perturbation spaces, a denotational semantics for repair plans, and interaction homomorphisms as a new algebraic structure. The system is a proof-of-concept REPL on DuckDB. Engineering is minimal; the algebra is the contribution.

### 1. Extreme Value and Who Needs It

**Persona: The database theory researcher** at a top-20 institution working on IVM for 5–15 years. IVM theory is stuck at the "fixed schema" barrier — DBSP, Differential Dataflow, and the Gupta-Mumick lineage all assume the schema never changes. This person needs a clean algebraic generalization that extends IVM to handle schema evolution, quality drift, and their interactions without abandoning formal rigor. The algebra provides the *lingua franca* for reasoning about pipeline perturbations.

**Secondary persona: PL researchers working on migration verification** (PRISM lineage) who want to connect schema evolution to data-level correctness but lack algebraic machinery.

### 2. Why Genuinely Difficult

- **Axiomatization completeness:** Defining correct algebraic laws for interaction homomorphisms φ: Δ_S → (Δ_D → Δ_D) that are compatible with delta composition in *both* sorts simultaneously. No existing template.
- **Propagation coherence across pipeline stages:** The key lemma requires characterizing which transformations preserve the interaction structure — analogous to naturality but over a three-sorted algebra with non-trivial interactions.
- **DBSP encoding impossibility proof:** Requires formalizing DBSP's circuit model well enough to prove a *negative* result. The formalization must be honest enough that DBSP researchers accept it.
- **Term-rewriting engine:** Correct, confluent, terminating rewrite system for a three-sorted algebra (~9K LoC of research-grade code).

### 3. New Math Required (Load-Bearing Only)

1. **Three-sorted delta algebra with interaction homomorphisms.** Monoid (Δ_S) × Group (Δ_D) × Lattice (Δ_Q) connected by φ and ψ. The coherence conditions (propagation through pipeline stages commutes with interaction) are genuinely novel.

2. **Bounded commutation theorem.** Exact for deterministic fragment F; bounded ε for non-deterministic pipelines. Proof by structural induction with interaction homomorphism commutativity at each stage.

3. **DBSP encoding impossibility via parametricity.** Prove that no encoding of schema deltas into DBSP's Z-set data domain preserves both type safety and incrementality. Must cover a class of encodings (universal tuple, tagged unions, deletion-reinsertion, meta-table).

4. **Decidability of fragment F.** For SQL without UDFs, membership in F is a syntactic check. Load-bearing: converts the commutation theorem from mathematical statement to engineering guarantee.

### 4. Best-Paper Potential

**PODS/ICDT candidate.** A new algebra generalizing a 40-year IVM lineage. The impossibility theorem is a sharp, citable separation. Risk: PODS reviewers may demand mechanized proofs (Coq/Lean).

### 5. Hardest Technical Challenge

**Proving equational completeness of the three-sorted algebra.** Cross-sort equations resist standard Birkhoff-style arguments. Mitigation: start with two-sorted (analogous to semidirect product of monoids, completeness known), extend via lattice modules, use Knuth-Bendix completion for critical pairs.

### 6. Scores

| Axis | Score | Rationale |
|---|---|---|
| Value | 6/10 | Transforms IVM theory but limited practitioner impact. Narrow audience. |
| Difficulty | 9/10 | Equational completeness, categorical semantics, encoding impossibility. |
| Potential | 9/10 | Clean generalization of 40-year research direction. Independently publishable results. |
| Feasibility | 5/10 | Completeness proofs are unpredictable. High risk of 60% math, no system. |

---

## Approach B: "Systems-First" — The Engine IS the Product

### Core Strategy

Build a production-grade repair engine that plugs into existing data infrastructure. Users never see delta terms — they see repair plans as SQL migrations and re-run commands. The algebra is internal, simplified to two sorts (Δ_S, Δ_D), with quality handled as tagged data deltas. The product is a CLI: "Schema change detected. Repair plan: re-cast 3 views, re-aggregate 1 table. Cost: 2.3 GB (vs. 47 GB full recompute). Correctness: guaranteed. Execute? [y/N]"

### 1. Extreme Value and Who Needs It

**Persona: The senior data engineer** at a Series B–D company, on-call owner of a 200-table warehouse. Paged at 2 AM because an upstream Postgres migration added a nullable column and 37 downstream dbt models produce wrong results. Spends 4 hours tracing lineage, writing migrations, testing, deploying. Needs: "Here is the provably correct fix. It touches 8 of your 37 models and costs 12 minutes instead of 6 hours."

**The pain:** 60% of their job is pipeline maintenance. Every fix is a high-wire act with no safety net. Company loses $50K/month in engineering time and $20K/month in over-provisioned compute.

**Why must-have:** Converts maintenance from art to engineering. The correctness guarantee is the killer feature no LLM can provide.

### 2. Why Genuinely Difficult

- **Multi-dialect SQL semantic analysis at column granularity.** sqlglot gives the AST; you need the semantics. Column-level lineage through correlated subqueries, lateral joins, window functions, recursive CTEs. State-of-art handles ~70%; getting to 95% requires solving known hard cases.
- **Saga-based repair execution across heterogeneous backends.** Compensating actions across DuckDB, Postgres, S3 — each with different transactional capabilities.
- **Cost model calibration.** Must predict actual execution cost accurately enough for the planner to beat naive recomputation.
- **Delta annihilation analysis.** Detecting that a propagated delta is provably zero at a stage — the primary speedup mechanism over lineage-aware selective recomputation.

### 3. New Math Required (Load-Bearing Only)

1. **Two-sorted delta algebra (Δ_S, Δ_D) with push operators.** Minimal algebra enabling correct repair planning. Quality handled as tagged data deltas.

2. **Commutation theorem for deterministic fragment.** The system's unique selling proposition. Without it, the system is just another heuristic repair tool.

3. **DP cost-optimal repair for acyclic topologies.** O(|V| · k²) algorithm exploiting delta-dependent cost model with annihilation.

4. **Delta annihilation analysis.** Static analysis determining when push_f(δ) = 0. The primary source of cost savings over dbt.

### 4. Best-Paper Potential

**VLDB/SIGMOD candidate** — but unlikely best paper. Systems papers need theoretical surprise; the 2-sorted algebra is a clean extension, not a breakthrough. Strong accept is realistic.

### 5. Hardest Technical Challenge

**Achieving ≥95% recall on column-level SQL lineage across two dialects.** Mitigation: extend sqlglot's lineage module for known hard cases, build 500-pattern ground-truth test suite, cross-validate with DuckDB EXPLAIN, fail conservatively with all-to-all fallback.

### 6. Scores

| Axis | Score | Rationale |
|---|---|---|
| Value | 9/10 | Directly solves #1 pain point for thousands of data engineers. |
| Difficulty | 7/10 | Multi-dialect analysis, saga execution, integration complexity. |
| Potential | 7/10 | Strong VLDB paper but lacks theoretical novelty for best paper. |
| Feasibility | 8/10 | Well-scoped, uses proven libraries, tiered scope ensures publishable result. |

---

## Approach C: "Hybrid: Algebra + Pragmatic Engineering" — The Bridge IS the Contribution

### Core Strategy

Neither pure algebra nor pure engineering. Build the full three-sorted delta algebra but ground every algebraic construct in a concrete system operation. The interaction homomorphism φ isn't just math — it's a code generator emitting SQL ALTER/CAST/DEFAULT statements. The commutation theorem isn't just a theorem — it's a test oracle the system checks at runtime. The impossibility theorem isn't just impossibility — it's a design justification for three sorts instead of encoding everything as data.

Every theorem earns its place by enabling a named engineering capability:

| Theorem | Engineering Capability Enabled |
|---|---|
| Interaction homomorphisms + coherence | Compound perturbation handling (schema + quality simultaneously) |
| Bounded commutation theorem | Correctness guarantee: repair = recompute for fragment F |
| DBSP encoding impossibility | Architectural justification for three-sorted design |
| Complexity dichotomy | Dual-algorithm planner (DP for DAGs, LP for cycles) |
| Delta annihilation | Repair plan pruning; primary speedup mechanism |

### 1. Extreme Value and Who Needs It

**Persona: The data platform team lead** at a large enterprise owning pipeline infrastructure serving 50+ consumers. Needs simultaneously: (1) correctness guarantees for downstream consumers, (2) cost optimization for $200K/month compute bills, (3) compound perturbation handling — because schema changes, quality violations, and outages arrive in the same week.

**Why compound perturbations are the killer feature:** When a schema change and quality violation arrive simultaneously, the interaction homomorphisms correctly compose the repairs — the schema change's default values must satisfy the quality constraint. No heuristic tool handles this. No LLM handles this. This is where the algebra becomes engineering necessity.

### 2. Why Genuinely Difficult

- **The algebra-to-code bridge.** ~200 push-operator definitions (20 SQL operators × 3 sorts + 15 Python idioms × 3 sorts). Each must individually be correct AND satisfy hexagonal coherence. One error invalidates the commutation theorem.
- **Three-sorted coherence is genuinely harder than two-sorted.** Interaction homomorphisms φ and ψ must be compatible with composition in all three sorts, forming a hexagonal coherence diagram.
- **Property-based testing at algebra scale.** Generating well-typed random deltas requires a custom generator infrastructure understanding the type system.
- **Delta annihilation with cross-sort interactions.** Compound deltas can cancel across sorts — analysis becomes constraint satisfaction over the three-sorted structure.

### 3. New Math Required (Load-Bearing Only)

1. **Three-sorted delta algebra with full coherence conditions.** The hexagonal diagram: `push_f(φ(δ_s)(δ_d)) = φ(push_f^S(δ_s))(push_f^D(δ_d))` — lifting a schema delta into a data-delta transformer commutes with propagation through f. Analogous condition for ψ. "Triangle" condition: applying compound (δ_s, δ_d, δ_q) is order-independent in the deterministic fragment. Load-bearing: without this, compound repair is order-dependent and nondeterministic.

2. **Bounded commutation theorem with explicit fragment characterization.** Three parts: (a) exact for F, (b) constructive characterization of F (bag semantics, exact arithmetic, no non-deterministic functions), (c) computable ε(G, σ) for pipelines outside F. Part (c) makes the system usable in production where floating-point aggregations are unavoidable.

3. **DBSP encoding impossibility via type-parametricity.** Proof that DBSP circuits, being natural transformations over Z-modules parametric in tuple type T, cannot internalize schema deltas without losing type safety or incrementality. Covers four encoding strategies: universal tuple, tagged unions, deletion-reinsertion, meta-table.

4. **Cost-optimal repair with compound deltas.** DP extended to compound perturbations: state space O(|V| · k² · 2³) — still polynomial for DAGs. Load-bearing: enables cross-sort annihilation (schema change renders quality constraint vacuous).

5. **Decidability of fragment F.** For SQL: syntactic check (no ORDER BY, LIMIT with ties, non-deterministic functions, external calls). For Python: idiom matcher success implies deterministic semantics. Converts commutation theorem to engineering guarantee.

### 4. Best-Paper Potential

**Strongest candidate across all approaches, targeting VLDB/SIGMOD.** The theory-practice bridge is the signature move of best papers at systems venues. Spark showed lineage enables fault-tolerant distributed computation. DBSP showed Z-sets enable unified IVM. This shows three-sorted delta algebras enable provably correct repair under compound perturbations — something no existing algebra or system can do. The evaluation (property-based algebraic verification + end-to-end benchmarks + 5 baselines) is what best-paper committees demand.

**Probability distribution:** 25% best paper / 45% strong accept / 25% accept after revision / 5% reject.

### 5. Hardest Technical Challenge

**Defining and verifying ~200 push-operator instances across three delta sorts and all supported operators.**

Mitigation strategy:
1. **Stratify by priority.** 8 SQL operators × 3 sorts + 5 Python idioms × 3 sorts = 39 definitions for MVP.
2. **Template-based generation.** 5-6 templates reduce to ~15 unique definitions + ~24 instantiations.
3. **Property-based testing.** 100K random deltas per operator verifying coherence conditions.
4. **Formal proofs for 5 critical operators.** SELECT, JOIN, GROUP BY, FILTER, UNION × 3 sorts = 15 lemmas.
5. **Conservative fallback.** Unverified operators get all-to-all treatment, preserving soundness.

### 6. Scores

| Axis | Score | Rationale |
|---|---|---|
| Value | 9/10 | Compound perturbation handling with correctness guarantees. Serves enterprise data platform teams. |
| Difficulty | 8/10 | Three-sorted coherence + ~200 push operators + property-based verification + full system. |
| Potential | 9/10 | Theory-practice bridge is best-paper profile. Three-sorted algebra + impossibility + system + evaluation. |
| Feasibility | 7/10 | Higher risk than B but tiered scope (start 2-sorted, extend to 3) provides graceful degradation. |

---

## Comparative Summary

| Dimension | A: Algebraic-First | B: Systems-First | C: Hybrid |
|---|---|---|---|
| Primary audience | DB theory researchers | Data engineers | Systems researchers + practitioners |
| Primary venue | PODS/ICDT | VLDB/SIGMOD | VLDB/SIGMOD |
| Core novelty | Free algebra, categorical semantics | Production repair engine | Theory-practice bridge |
| Math depth | Deep (completeness, Galois connection) | Moderate (2-sorted, DP, annihilation) | Deep where load-bearing (3-sorted coherence, impossibility) |
| Engineering depth | Minimal (REPL + DuckDB) | Deep (multi-dialect, saga, orchestrators) | Moderate (focused on algebra-to-SQL bridge) |
| Risk profile | High theory, low engineering | Low theory, moderate engineering | Moderate both |
| Best-paper probability | ~15% (PODS) | ~15% (VLDB) | ~25% (VLDB/SIGMOD) |
| Graceful degradation | 2-sorted algebra is still a paper | SQL-only tool is still useful | Two-sorted + benchmarks is strong |
| Estimated LoC | ~48,500 | ~63,000 | ~60,500 |
| Novel LoC | ~18,000 | ~8,000 | ~16,000 |
| P(publishable) | 75% | 60% | 70% |

---

## Recommendation

**Approach C is the strongest.** It is the only approach that can win best paper because the interaction homomorphisms are not ornamental math — they are the *mechanism* enabling compound perturbation handling, which no competitor can match. The tiered scope ensures graceful degradation: even if three-sorted coherence proves harder than expected, the two-sorted version + impossibility theorem + benchmarks is a strong accept. Approach A maximizes mathematical ceiling but risks delivering beautiful theory nobody implements. Approach B is the safe choice but leaves the best-paper on the table — the 2-sorted algebra makes it "DBSP + schema deltas," competing on engineering alone.
