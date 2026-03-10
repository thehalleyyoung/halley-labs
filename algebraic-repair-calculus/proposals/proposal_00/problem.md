# Final Approach: Algebraic Repair Calculus

## 1. Approach Name and Core Thesis

**"Grounded Algebra" — Interaction Homomorphisms as the Mechanism for Provably Correct Pipeline Repair**

We build a three-sorted delta algebra (Δ_S, Δ_D, Δ_Q) whose genuinely novel contribution — interaction homomorphisms connecting schema evolution, data changes, and quality drift — is grounded in a working repair engine that demonstrates every theorem's engineering necessity. The paper's argument is not "here is a beautiful algebra" or "here is a useful tool" but: "interaction homomorphisms are the minimal algebraic structure required for provably correct repair under compound perturbations, and here is the system that proves it." Every algebraic result earns its place by enabling a named capability that no heuristic or single-sorted approach can replicate. We do not claim paradigm shift; we claim a clean algebraic insight with demonstrable engineering consequence.

## 2. Extreme Value and Who Needs It

**Primary persona: The data platform team lead** at a mid-to-large company (Series C+ or enterprise), on-call owner of a 100–500 table warehouse with 20–50+ downstream consumers. This person manages 2–5 data engineers who collectively spend 40–60% of their time on pipeline maintenance (per dbt Labs and Fivetran surveys).

**The pain (quantified):**
- **$50–200K/month** in engineering time on manual pipeline repair (2–3 FTE at market rates).
- **$20–80K/month** in over-provisioned compute from full recomputation as the "safe" repair strategy.
- **4–8 hours** mean time to resolution per schema-change incident, during which downstream consumers receive stale or incorrect data.
- **Zero formal guarantees** on selective recomputation: every `dbt --select` repair is a bet that the engineer traced lineage correctly.

**Why this approach specifically addresses it:**
1. The **commutation theorem** provides the guarantee no existing tool offers: for pipelines in fragment F (decidable by syntactic check), incremental repair = full recomputation. This eliminates the "did my fix break something else?" anxiety.
2. **Delta annihilation analysis** — detecting when a perturbation has zero effect at a stage — is the primary cost-saving mechanism. Lineage-aware selective recomputation (dbt `--select`) cannot detect this: it recomputes every transitively affected stage. Our algebra prunes provably-zero deltas, targeting 2–5× cost reduction over the real baseline.
3. **Compound perturbation handling** via interaction homomorphisms. When a schema change and quality violation arrive in the same repair window, φ(δ_s)(δ_d) correctly transforms the data delta to operate on the new schema. No existing tool, LLM, or heuristic handles this correctly.

**Addressing the Skeptic:** The Skeptic correctly notes that "provably correct" is a researcher's value proposition, not an on-call engineer's. We reframe: the value proposition is **automated repair with machine-checkable correctness certificates**. The engineer doesn't read the proof — the system checks the preconditions (pipeline in fragment F), generates the repair plan, and reports "repair provably correct" or "repair within deviation bound ε = 0.003." The engineer cares about the green checkmark, not the algebra behind it.

**Addressing compound perturbation rarity:** The Skeptic challenges that compound perturbations may be rare. We concede this is unverified and commit to measuring it empirically against public schema migration traces. However, even if compound perturbations are 10–15% of cases, the interaction homomorphisms are still load-bearing: they enable the algebra to *prove* that sequential handling is safe when it is, rather than assuming it. The homomorphisms formalize the independence condition that practitioners implicitly rely on.

## 3. Technical Architecture

### Subsystem Breakdown with Honest LoC Estimates

| Subsystem | LoC | Difficulty | Key Library | Novel LoC |
|---|---|---|---|---|
| SQL Semantic Analyzer (PostgreSQL, 8 core operators) | 5,500 | 6/10 | sqlglot | ~2,000 |
| Python Idiom Matcher (8–10 idioms) | 3,500 | 5/10 | ast (stdlib) | ~1,500 |
| Typed Dependency Graph | 4,000 | 4/10 | networkx | ~1,000 |
| Refinement Type System (schema + non-null + range) | 4,500 | 7/10 | — | ~3,000 |
| **Delta Algebra Engine** | **9,000** | **9/10** | **—** | **~8,000** |
| **Repair Planner (DP + LP fallback)** | **5,500** | **8/10** | scipy, HiGHS | **~4,000** |
| Repair Executor (DuckDB, checkpoint/rollback) | 3,000 | 4/10 | DuckDB | ~500 |
| Data Quality Monitor (batch, KS + PSI) | 3,000 | 4/10 | scipy.stats | ~500 |
| Pipeline State Management | 2,500 | 3/10 | SQLite | ~200 |
| Evaluation & Property Tests | 10,000 | 5/10 | DuckDB, Hypothesis | ~2,000 |
| **Total (Tier 1+)** | **~50,500** | | | **~22,700** |

### Key Design Decisions and Rationale

1. **Three-sorted from the start, two-sorted as escape hatch.** We implement Δ_S, Δ_D, Δ_Q with interaction homomorphisms φ and ψ from week 1. If three-sorted coherence fails for >3 of 8 operators by the 40% time mark, we retreat to two-sorted (Δ_S, Δ_D) with quality as tagged data deltas. This is still a publishable paper (the preliminary synthesis confirms "two-sorted + benchmarks is a strong accept").

2. **Column-level interaction homomorphisms.** φ and ψ operate at column granularity, not schema-level. This gives precision but costs O(n²) composition for n columns. For pipelines with schemas <200 columns (the practical universe), this is acceptable. The decision is irreversible and permeates the algebra.

3. **DuckDB as sole execution backend.** We explicitly drop heterogeneous backend support. The Difficulty Assessor rates saga executor bugs at 40% probability; we eliminate this risk entirely. DuckDB provides: OLAP performance for evaluation, transactional semantics for checkpoint/rollback, and EXPLAIN-based lineage cross-validation.

4. **Sound over-approximation as default for SQL analysis.** When the analyzer cannot determine lineage for a subexpression, it defaults to all-to-all (every output column depends on every input column). This guarantees soundness (no missed repairs) at a measured precision cost. Target: ≥95% recall on TPC-DS single-dialect queries, with all-to-all fallback covering <5% of stages.

5. **Property-based testing as the bridge, not a substitute for proofs.** Formal proofs for 5 critical operators (SELECT, JOIN, GROUP BY, FILTER, UNION) × 3 sorts = 15 coherence lemmas. Property-based testing (100K+ random deltas per operator via Hypothesis) for remaining operators. The paper states this split honestly: "provably correct for the core relational operators; empirically verified for extended operators; conservatively sound for unrecognized operators."

## 4. New Mathematics Required

### Result 1: Three-Sorted Delta Algebra with Interaction Homomorphisms

**Statement:** Define the algebraic structure (Δ_S, Δ_D, Δ_Q, ∘, ⁻¹, push) where Δ_S is the monoid of schema deltas, Δ_D is the group of data deltas, Δ_Q is the lattice of quality deltas, connected by interaction homomorphisms φ: Δ_S → End(Δ_D) and ψ: Δ_S → End(Δ_Q). Prove the hexagonal coherence condition: `push_f(φ(δ_s)(δ_d)) = φ(push_f^S(δ_s))(push_f^D(δ_d))` for 8 core SQL operators, and the analogous condition for ψ. Prove the triangle condition: applying compound (δ_s, δ_d, δ_q) is order-independent in the deterministic fragment.

**Why load-bearing:** Without φ, a schema change and data correction cannot be composed into a single repair — you must handle them sequentially, risking inconsistent intermediate states. Without the coherence condition, propagation through multi-stage pipelines is not compositional — you cannot reason about each stage independently. The interaction homomorphisms are the mechanism that makes compound repair plans well-defined.

**Honest novelty rating: 6/10.** The Math Assessor is correct: the pattern (multi-sorted algebra with cross-sort morphisms) is standard in universal algebra. A category theorist would recognize this as a fibration. The novelty is in the specific instantiation connecting schema evolution, data maintenance, and quality contracts — no prior work defines these cross-sort maps or proves coherence for relational operators. This is "new application of known algebraic structures to a genuinely new domain," not "new mathematics."

**Proof strategy:** Case analysis on each of the 8 SQL operators × 3 delta sorts × 2 interaction maps. For each (operator, delta-sort, interaction-map) triple, verify the coherence condition by expanding definitions under bag semantics. The 5 critical operators (SELECT, JOIN, GROUP BY, FILTER, UNION) get full formal proofs; the remaining 3 (WINDOW, CTE-reference, set operations) get detailed proof sketches + property-based verification.

**Fallback:** If coherence fails for >3 operators, retreat to two-sorted (Δ_S, Δ_D). Quality becomes tagged data deltas. The commutation theorem still holds for the two-sorted fragment. This is Approach B's strategy and is still a publishable contribution.

### Result 2: Bounded Commutation Theorem with Decidable Fragment Characterization

**Statement:** For any pipeline DAG G in fragment F, source perturbation σ = (δ_s, δ_d, δ_q), and repair plan R produced by the planner: `apply(R, state(G)) = recompute(evolve(G, σ))`. For pipelines outside F: `distance(apply(R, state(G)), recompute(evolve(G, σ))) ≤ ε(G, σ)` where ε is computable. **Critically:** membership in F is decidable — for SQL, F = {bag semantics, exact arithmetic, no ORDER BY, no LIMIT under ties, no non-deterministic functions, no external calls}; this is a syntactic check on the AST.

**Why load-bearing:** This theorem is the system's unique selling proposition. Without it, the system is a heuristic repair tool with no correctness story beyond testing. The decidability of F is what the Math Assessor calls "the single most important missing result" — it converts the theorem from a mathematical statement into an engineering guarantee the system can check at runtime and report to the user.

**Honest novelty rating: 5/10 (theorem) + 4/10 (decidability).** The commutation theorem is structural induction with a new inductive hypothesis — the technique is standard, the domain is new. The decidability of F for SQL-only pipelines is a syntactic check, not deep. The combination (theorem + decidability + computable ε) is useful but no single component is surprising. **If we precisely characterize F including SQL's semantic dark corners (NULLs, three-valued logic, implicit coercions), the rating rises to 7/10** — this is where we should invest.

**Proof strategy:** Structural induction on the pipeline DAG G. Base case: source nodes (trivial). Inductive step: for each SQL operator f, show that the coherence condition from Result 1 implies `apply(push_f(R_in), f(state_in)) = f(apply(R_in, state_in))`. This uses the interaction homomorphism commutativity at each stage. For the bounded case outside F: derive ε from the pipeline's non-determinism profile using backward error analysis (connection to numerical analysis for floating-point aggregation chains).

**Fallback:** If the tight characterization of F proves harder than expected, we define F conservatively (pure bag algebra with exact arithmetic, no NULLs) and report coverage as a metric: "F covers X% of stages in TPC-DS pipelines." Even a conservative F with high coverage is useful.

### Result 3: DBSP Encoding Impossibility

**Statement:** No encoding of schema deltas into DBSP's Z-set data domain preserves both type safety and incrementality. Formalized: DBSP circuits are natural transformations over Z-modules, parametric in tuple type T. For any functor F: SchDelta → Z-setDelta, either F forgets type information (destroying type-directed optimization) or F does not preserve the monoidal structure that enables incrementalization (reducing to full recomputation). The proof covers four encoding strategies: (1) universal tuple type (Map<String, Any>), (2) tagged unions, (3) deletion-reinsertion, (4) meta-table encoding.

**Why load-bearing:** This result justifies the three-sorted design architecturally. Without it, the obvious reviewer objection is "why not just extend DBSP?" The impossibility shows this isn't a feature gap but a fundamental limitation: DBSP's parametricity over a fixed tuple type is incompatible with first-class schema evolution.

**Honest novelty rating: 4–7/10 (range reflects formalization uncertainty).** The Skeptic's challenge is direct: "Isn't this trivially true? DBSP has fixed types, schema changes change types, QED." The trivial version is a 2/10 observation. The non-trivial version must show that *clever encodings* fail — specifically, that tagged unions (which partially work) lose incrementality for the tag-dispatch overhead, and deletion-reinsertion (which technically works in Z-set algebra) reduces to full recomputation. **We commit to proving the non-trivial version covering all four encoding strategies, or demoting this to a remark.** We will produce a concrete proof sketch within the first 3 weeks to determine if it's deep or trivial.

**Proof strategy:** Formalize "preserves type safety" as: the encoding respects typed operations (projection, join) without dynamic dispatch. Formalize "preserves incrementality" as: the encoding maps schema deltas to Z-set deltas whose application cost is proportional to the *change size*, not the *data size*. Show each encoding violates at least one condition. The representation independence argument parallels Reynolds' abstraction theorem.

**Fallback:** If the proof reduces to a trivial observation ("DBSP has fixed types"), demote it from a theorem to a design motivation paragraph. The three-sorted algebra stands on its own merits (compound perturbation handling) regardless. This weakens the paper's positioning but doesn't kill it.

### Result 4: Cost-Optimal Repair with Delta Annihilation

**Statement:** (a) Cost-optimal repair plan synthesis is NP-hard by reduction from weighted set cover. (b) For acyclic pipeline topologies (>90% of real pipelines), the exact optimal plan is computable in O(|V| · k²) via dynamic programming, where k is maximum fan-out. (c) For general topologies, an (ln k + 1)-approximation via LP relaxation with randomized rounding. (d) **Delta annihilation analysis:** a static analysis determining when push_f(δ) = 0 for stage f and incoming delta δ. This is the primary cost-saving mechanism.

**Why load-bearing:** The DP algorithm is the planner's core. Delta annihilation is the measurable differentiator vs. dbt selective recomputation — when a column-add propagates through a SELECT that doesn't reference the new column, the output delta is zero and the stage is pruned from the repair plan. dbt cannot detect this. The Math Assessor explicitly calls annihilation "under-sold" and "the algebraic result with the most direct performance impact."

**Honest novelty rating: 4/10 (complexity results), 6/10 (annihilation analysis).** The NP-hardness reduction and LP relaxation are textbook. The DP algorithm's specific formulation over the delta-algebraic cost model is novel — particularly handling non-monotone costs from delta annihilation (where a subtree's cost drops to zero, breaking standard optimal substructure). The annihilation analysis itself (symbolic reasoning about when push_f(δ) = 0 for relational operators) is a genuine algorithmic contribution that no prior system implements.

**Proof strategy:** NP-hardness: standard weighted set cover reduction (construct a pipeline where each stage covers a subset of perturbation effects). DP optimality: prove that delta annihilation restores optimal substructure — when a subtree's delta is zero, its cost is zero regardless of parent decisions, so subproblems decouple at annihilation boundaries. Annihilation: case analysis on each operator × delta-sort, determining conditions under which the output delta is provably empty.

**Fallback:** If non-monotone DP proves intractable, use greedy topological-order planning with annihilation pruning (no optimality guarantee, but still correct and still faster than dbt). This weakens the "cost-optimal" claim to "cost-efficient" but preserves the annihilation story.

### Result 5: Decidability of Fragment F

**Statement:** For SQL pipelines without UDFs: membership in F is decidable by a syntactic check on the sqlglot AST (no ORDER BY, no LIMIT under ties, no non-deterministic functions, no floating-point GROUP BY, no external calls). For Python stages: idiom matcher success implies known deterministic semantics; unmatched idioms receive conservative (outside-F) treatment. The system reports at pipeline-registration time: "87% of your stages are in fragment F and receive exact correctness guarantees."

**Why load-bearing:** Without decidability, the commutation theorem is a mathematical curio. A user cannot know whether their pipeline gets a formal guarantee or a deviation bound. This result converts the commutation theorem from theory to engineering.

**Honest novelty rating: 3/10.** This is a syntactic check, not deep mathematics. But it is *essential infrastructure* that all experts agree must exist.

**Proof strategy:** Define F syntactically on the sqlglot AST. Prove that every AST node satisfying the syntactic predicate denotes a deterministic, order-independent function under bag semantics. This is a straightforward soundness argument.

**Fallback:** None needed — this is low-risk.

## 5. Why Genuinely Difficult as Software Artifact

### The 3 Hardest Subproblems

**1. Interaction homomorphism correctness through real SQL operators. (Difficulty: 9/10)**

The algebra must correctly propagate three-sorted deltas through joins, aggregations, window functions, and CTEs. Each push-operator definition must individually satisfy hexagonal coherence, and there are 8 operators × 3 sorts × 2 interaction maps = 48 coherence conditions to verify. A single error invalidates the commutation theorem for all pipelines using that operator. The Difficulty Assessor rates a 30% probability that interaction homomorphisms don't compose cleanly — rated FATAL. No existing template, library, or prior system provides guidance. This must be invented.

**2. Non-monotone DP for delta-aware repair planning. (Difficulty: 8/10)**

Standard DP on DAGs assumes optimal substructure: the optimal solution to a subproblem is independent of decisions made in other subproblems. When the cost function depends on delta composition — and delta annihilation can drive a subtree's cost to zero — subproblems interact. We must prove that annihilation boundaries restore independence, or invent a modified DP that accounts for delta-dependent costs. The Difficulty Assessor flags this as a potential "hidden show-stopper" for the "polynomial for DAGs" claim.

**3. SQL column-level lineage at ≥95% recall. (Difficulty: 6/10, but foundation-critical)**

sqlglot gives the AST; we need the semantics. Column-level lineage through correlated subqueries, lateral joins, window functions with PARTITION BY, and CTEs requires a scope-aware name resolver and semantic visitor infrastructure. The Difficulty Assessor identifies a 25–35% probability of incorrect lineage causing silent data corruption — the worst possible failure mode for a "provably correct" system. This is not research-hard, but it is the foundation on which every algebraic claim rests.

### Novel Algorithms That Must Be Invented

1. **Interaction-aware delta propagation:** Propagate a three-sorted delta through a pipeline DAG where each node applies a different relational operator. Must handle operator-specific rules for each delta sort and their interactions. No existing algorithm does this.

2. **Delta annihilation detector:** Given a delta arriving at a pipeline stage, determine symbolically if push_f(δ) = 0. Requires reasoning about the algebraic interaction between delta types and SQL operators' column-dependency structures.

3. **Annihilation-aware DP planner:** Modified dynamic programming that exploits annihilation boundaries to restore optimal substructure under non-monotone delta-dependent cost functions.

### What Existing Libraries Provide and What They Can't

| Library | Provides (~% of subsystem) | Cannot Provide |
|---|---|---|
| sqlglot | Parsing, AST, dialect normalization (~70% of analyzer) | Semantic column resolution, lineage extraction, scope chains |
| networkx | Graph storage, traversal, topological sort (~60% of graph) | Typed annotations, incremental type propagation, delta-aware subgraphs |
| scipy/HiGHS | LP/ILP solving, statistical tests (~30% of planner, ~70% of quality) | Problem formulation, delta-algebraic cost encoding, quality-delta inference |
| DuckDB | Query execution, EXPLAIN plans (~80% of executor) | Cross-stage transactional guarantees (we use checkpoint/rollback) |
| Hypothesis | Property-based test framework | Well-typed three-sorted delta generators (we write these — ~1,500 LoC of genuinely hard test infrastructure) |

## 6. Best-Paper Argument

### Honest Assessment (Incorporating Skeptic's Deflation)

The Skeptic is right: this is not in the lineage of Spark, Naiad, or DBSP. Those were paradigm shifts adopted by millions or reshaping how an entire community thinks. This is a well-executed niche contribution with a clean algebraic insight. We stop claiming otherwise.

**What makes this paper stand out from "another pipeline tool":** The theory-practice bridge is genuine, not cosmetic. Specifically:

| Without the Algebra | With the Algebra |
|---|---|
| Lineage-aware selective recomputation (dbt `--select`): recompute all transitively affected stages | Delta annihilation prunes provably-zero deltas → 2–5× cost reduction over dbt baseline |
| No correctness guarantee for selective recomputation | Commutation theorem guarantees repair = recompute for decidable fragment F |
| Compound perturbations handled sequentially, hoping order doesn't matter | Interaction homomorphisms formally compose cross-sort repairs |
| "Does my fix break something else?" — unknowable | Machine-checkable correctness certificate for every repair plan |

**The one result that carries the paper:** Delta annihilation analysis grounded in interaction homomorphisms. This is the result where the algebra directly and measurably outperforms every baseline. A reviewer can look at the evaluation and see: "For this perturbation, dbt recomputes 37 stages. The algebraic repair touches 8, because annihilation proves 29 stages are unaffected. The output is identical." That's the moment the algebra earns its place.

**Honest best-paper probability: ~8%.** (Skeptic's unconditional estimate: P(implementation lands) × P(best paper | lands) ≈ 0.55 × 0.15 ≈ 8%.) The realistic target is **strong accept at VLDB/SIGMOD** with 50–100 citations over 5 years. This is a good outcome. We optimize for the floor (publishable result), not the ceiling (best paper).

## 7. Hardest Technical Challenge and Mitigation Strategy

**#1 Risk: Interaction homomorphisms don't compose cleanly through real SQL operators.**

The hexagonal coherence condition requires that lifting a schema delta into a data-delta transformer commutes with propagation through each SQL operator. For simple operators (PROJECT, FILTER, UNION), this likely falls out from definitions. For complex operators (JOIN on a renamed column, GROUP BY with a widened type, WINDOW with PARTITION BY on a column referenced in a quality constraint), the coherence condition may fail.

**Step-by-step mitigation:**

1. **Weeks 1–2:** Implement push operators for the 3 simplest operators (SELECT/PROJECT, FILTER, UNION) × 3 sorts. Run 100K property-based tests per operator via Hypothesis. If coherence fails here, the algebra has a fundamental design flaw — **discover this in week 2, not month 3.**

2. **Weeks 3–4:** Add JOIN and GROUP BY. These are the first operators where cross-sort interaction is non-trivial. If coherence fails, diagnose: is the failure in the operator definition (fixable) or the coherence condition itself (fundamental)? If the condition needs loosening (e.g., coherence up to bounded permutation), characterize the bound and adjust the commutation theorem.

3. **Weeks 5–6:** Add WINDOW, CTE-reference, set operations. These are the hardest cases. If coherence fails here, use the **conservative fallback:** treat these operators as opaque (all-to-all dependency) for delta propagation. Soundness is preserved; we lose precision for pipelines containing these operators. Measure the precision penalty on TPC-DS.

4. **Detection criterion:** A single coherence failure on a random Hypothesis test input triggers investigation. Use Hypothesis shrinking to find minimal counterexamples.

5. **Escape hatch (week 7):** If coherence fails for >3 of 8 operators, retreat to two-sorted algebra (Δ_S, Δ_D). Quality becomes tagged data deltas. Redraft the paper as a two-sorted contribution with impossibility theorem + commutation + annihilation + benchmarks. This is still a strong VLDB submission — the preliminary synthesis confirms Approach B at 7/10 potential.

## 8. Tiered Scope

### Tier 1 — Minimum Viable Paper (~38K LoC)

**Scope:** Three-sorted algebra over SQL-only (PostgreSQL) pipelines, 8 core operators, DP planner for acyclic topologies, DuckDB execution, TPC-DS evaluation.

| Component | Scope | LoC |
|---|---|---|
| SQL Semantic Analyzer | PostgreSQL only, 8 core operators, ≥90% recall on TPC-DS | 5,000 |
| Python Idiom Matcher | SKIP | 0 |
| Typed Dependency Graph | Basic typed DAG on networkx | 3,500 |
| Refinement Type System | Schema types + non-null predicates only | 3,500 |
| Delta Algebra Engine | Three-sorted, 8 SQL operators, coherence verified | 8,000 |
| Repair Planner | DP for acyclic only | 4,000 |
| Repair Executor | DuckDB only, checkpoint/rollback | 2,500 |
| Data Quality Monitor | Batch-mode, KS test + PSI | 2,500 |
| Evaluation & Tests | TPC-DS SF=10, 3 baselines, 1M property tests | 9,000 |
| **Total** | | **~38,000** |

**Theorems delivered:** (1) Three-sorted coherence for 8 operators, (2) Bounded commutation for F, (3) Decidability of F, (4) DBSP encoding impossibility, (5) DP optimality for acyclic topologies.

**Baselines:** Full recomputation, lineage-aware selective recomputation (dbt `--select`), DBSP data-only IVM.

**What makes it publishable:** Novel algebra + impossibility theorem + correctness guarantee + measurable cost savings over dbt baseline on TPC-DS. Strong VLDB submission even without Python or compound perturbation optimization.

### Tier 2 — Target Paper (~55K LoC)

**Adds:** Python support, Spark SQL dialect, compound perturbation optimization, LP fallback, full evaluation.

| Addition | LoC Added |
|---|---|
| Spark SQL dialect (via sqlglot) | +2,000 |
| Python Idiom Matcher (10 idioms) | +5,000 |
| Compound delta DP extension (O(|V| · k² · 2³)) | +1,500 |
| LP relaxation for cyclic topologies | +1,500 |
| Saga-lite executor (compensating actions, single-backend) | +2,000 |
| Full quality integration (streaming wrappers, Bonferroni) | +1,500 |
| Expanded evaluation (5 baselines, 500 topologies, 10M property tests) | +4,000 |
| **Total added / Cumulative** | **~17,500 / ~55,500** |

**Additional theorems:** (6) Cross-sort annihilation in compound perturbations, (7) Idiom soundness for 10 Python patterns.

**What makes it the target:** Three-sorted algebra demonstrated end-to-end across SQL + Python. Compound perturbation handling that no baseline can match. Five-baseline evaluation. This is the full submission.

### Tier 3 — Stretch (~68K LoC)

**Adds:** Quantitative bridge theorems, incremental replanning, real-world schema traces, categorical connection.

| Addition | LoC Added |
|---|---|
| Bridge theorems ("repair touches ≤ f(m,n,k) stages") | +1,000 |
| Incremental replanning (perturbation during repair) | +2,500 |
| Real-world schema traces (Rails/Alembic migrations) | +3,000 |
| Additional SQL dialects (MySQL, BigQuery) | +2,000 |
| Scale evaluation (TPC-DS SF=100, 500–1000 stages) | +2,000 |
| Categorical connection paragraph (fibered category framing) | +2,000 |
| **Total added / Cumulative** | **~12,500 / ~68,000** |

**What makes it best-paper push:** Quantitative guarantees practitioners can check mechanically + real-world schema traces showing beyond-benchmark validity + categorical framing connecting to 40-year IVM literature. The bridge from algebra to measurable engineering impact is complete.

**Hard scope rules:**
- No Tier 2 work begins until ALL Tier 1 completion criteria are met.
- At 60% of available time: if Tier 1 incomplete, cut Python and Spark SQL permanently.
- At 80% of available time: freeze features, focus on evaluation + writing.
- If at 60% time Tier 1 is incomplete: pivot to whichever half (algebra or system) is more complete.

## 9. Risk Assessment

### Top 5 Risks

| # | Risk | Probability | Impact | Mitigation |
|---|---|---|---|---|
| 1 | Interaction homomorphisms don't compose through real SQL operators | 30% | FATAL | Stratified implementation (simple → complex), property-based early detection, two-sorted escape hatch |
| 2 | Scope creep from balancing algebra + system | 40% | HIGH | Hard tier gates, calendar-based cutoffs, cognitive mode separation (contiguous algebra/engineering blocks) |
| 3 | SQL lineage incorrectness invalidates algebraic claims | 25% | HIGH | Sound over-approximation default, 300+ ground-truth test suite, DuckDB EXPLAIN cross-validation |
| 4 | DBSP impossibility proof is trivially true | 35% | MEDIUM | Produce concrete proof sketch covering 4 encodings within first 3 weeks; if trivial, demote to remark |
| 5 | Commutation theorem requires too many restrictions (fragment F covers <50% of real stages) | 25% | HIGH | Precisely characterize F early; if coverage <50%, reframe: "ε bound for all pipelines" with "exact for core fragment" |

### Honest Probability Estimates

**P(publishable result): 55%.** This incorporates: 50% full implementation probability (Difficulty Assessor), 70% publishable given partial (preliminary synthesis), discounted for scope risk and Math Assessor's novelty deflation. The tiered scope is the primary protection — Tier 1 alone is a credible submission.

**P(best paper): 8%.** Unconditional: P(all components land cleanly) ≈ 0.35 × P(best paper | everything lands) ≈ 0.22 ≈ 8%. The Skeptic is right that the honest ceiling is "solid VLDB accept." We design for that outcome and treat best paper as upside, not expectation.

**P(useful artifact regardless of publication): 75%.** Even if the paper doesn't land, the SQL analyzer + delta algebra + repair planner is a useful open-source tool for the data engineering community.

## 10. What We Explicitly Drop (and Why)

| Dropped | Source | Justification |
|---|---|---|
| **Categorical semantics (Galois connections, fibrations, free algebra)** | Approach A | Math Assessor: would push to 8–9/10 depth. Difficulty Assessor: A has 55% implementation probability. One paragraph noting the fibration connection signals sophistication; full treatment is a follow-up paper. We cannot afford multi-month investment in categorical formalism with unpredictable payoff. |
| **Equational completeness** | Approach A | "Completeness proofs for multi-sorted algebras are notoriously difficult." The simplifier is sound but not provably complete. We measure the precision penalty empirically and state the limitation honestly. |
| **Full saga executor with heterogeneous backends** | Approach B | Difficulty Assessor: 40% saga bug probability. DuckDB-only with checkpoint/rollback eliminates distributed systems failure modes entirely. Multi-backend is Tier 3+. |
| **Orchestrator integration (dbt, Airflow, Dagster adapters)** | Approach B | Each adapter is "unglamorous but genuinely hard" engineering (Difficulty Assessor). Adds LoC without novelty. Evaluation uses TPC-DS pipelines, not live orchestrators. |
| **Streaming quality monitor** | Problem statement Tier 3 | Batch quality-delta inference is sufficient. Streaming adds engineering complexity without theoretical contribution. |
| **Full Python idiom matcher (15–20 idioms) in Tier 1** | Problem statement | Reduced to 8–10 idioms covering ≥85% of pandas usage in Tier 2. Tier 1 is SQL-only. Idiom coverage is engineering validation (2/10 novelty per Math Assessor). |
| **Type soundness proof in main paper** | All experts agree | Unanimously ornamental. State the type system, describe guarantees informally, cite the proof in appendix/technical report. Wright-Felleisen metatheory impresses no one at VLDB. |
| **Comparison to Spark/Naiad/DBSP lineage** | Skeptic | The Skeptic is right: "comparing this to Spark is delusion." We frame honestly as "a clean algebraic insight with demonstrable engineering consequence," not a paradigm shift. |
| **Claims of "unprecedented" novelty** | Math Assessor | Interaction homomorphisms are 6/10, not "unprecedented." We call them "a novel application of multi-sorted algebra to pipeline perturbation spaces" — accurate and defensible. |

## 11. Evaluation Plan

### Benchmarks

1. **TPC-DS Pipeline Corpus:** 99 TPC-DS queries organized into 15 ETL pipeline DAGs with realistic transformation chains. SF=10 default, SF=100 stretch. Schema evolution traces generated systematically: column additions, type widenings, renames, drops.

2. **Synthetic Perturbation Suites:** 500 randomly generated pipeline topologies (10–500 nodes: chain, tree, DAG). Parameterized perturbation sequences covering all combinations of schema/data/quality deltas. Stress tests at 200, 500, and 1000 stages for planner scalability.

3. **Real-World Schema Evolution Traces (Tier 3):** Public schema migration histories from GitHub (Rails migrations, Alembic changelogs). This addresses the Skeptic's challenge that synthetic benchmarks don't convince systems reviewers.

### Baselines (5)

1. **Full Recomputation:** Rebuild all downstream nodes. Secondary baseline (worst case).
2. **Lineage-Aware Selective Recomputation (dbt `--select`):** Recompute nodes with affected input columns. **Primary baseline** — current state of practice.
3. **DBSP-style Data-Only IVM:** Apply DBSP's algebra ignoring schema/quality deltas. Exposes the expressiveness gap.
4. **Naive Incremental:** Recompute directly-affected nodes only, no transitive optimization.
5. **Rule-Based Heuristic Repair:** Hand-written rules (column added → add default; type changed → cast) as in sqlmesh-style tools.

### Metrics

| Metric | Target | What It Proves |
|---|---|---|
| **Correctness** (semantic equivalence with full recompute for F) | 100% | The commutation theorem holds in practice |
| **Cost ratio** (algebraic repair / dbt selective recompute) | < 0.5 (2×+ improvement) | Delta annihilation provides real savings |
| **Correctness coverage** (fraction of stages in F) | > 80% of SQL-only stages | Fragment F is practically useful |
| **Planning time** (p95 latency) | < 1 second for pipelines ≤ 500 stages | The planner is interactive-speed |
| **Annihilation rate** (fraction of stages pruned by annihilation) | Report honestly (key metric) | How often the algebra outperforms naive lineage |
| **Algebra verification** (property-based tests) | 10M+ delta combinations, 0 coherence failures | The implementation matches the algebra |
| **SQL analysis recall** | ≥ 95% on TPC-DS | The lineage foundation is sound |
| **Compound perturbation correctness** | 100% on synthetic suite | Interaction homomorphisms work in practice |

### How the Evaluation Proves the Value of the Algebra (Not Just the System)

The critical evaluation question is: **does the algebra buy anything a well-engineered heuristic system cannot?** The Math Assessor warns that "80% of the capability is achievable without algebra." We must demonstrate the 20% concretely.

**Ablation study (the key experiment):**
1. Run our system with full algebraic analysis (including annihilation + coherence).
2. Run a **no-algebra baseline**: same SQL analyzer and graph infrastructure, but replace the delta algebra with heuristic rules (column added → recompute all downstream; type changed → cast and recompute). This is the 80% system the Math Assessor describes.
3. Measure: (a) correctness (does the heuristic get repairs wrong?), (b) cost (does annihilation save stages?), (c) compound handling (does sequential heuristic handling produce different results?).

If the ablation shows the algebra saves ≥30% of unnecessary recomputation via annihilation AND catches ≥1 compound perturbation case where heuristic sequential handling is incorrect, the algebra has demonstrated its engineering value. If not, the paper must honestly report that the algebra's marginal value is theoretical (correctness guarantee), not practical (cost savings).

## 12. Scores

| Axis | Score | Rationale |
|---|---|---|
| **Value** | **8/10** | Solves the #1 pain point in data engineering (pipeline maintenance under schema evolution) with formal correctness guarantees. Downgraded from 9/10 because, as the Skeptic notes, practitioners may not demand formal guarantees — "good enough" heuristics serve 80% of cases. The 8 reflects genuine need from the enterprise data platform persona, not universal demand. |
| **Difficulty** | **8/10** | Requires inventing interaction homomorphisms (9/10 subsystem), proving non-trivial coherence conditions, building a non-monotone DP planner, and achieving ≥95% SQL lineage recall. The Difficulty Assessor's 7.5/10 for Approach C, rounded up because we inherit Approach A's hardest subproblem (interaction homomorphisms at 9/10) while also requiring a working system. Genuinely novel algorithms must be invented; no library solves the core problems. |
| **Potential** | **7/10** | Downgraded from the Visionary's 9/10 to reflect the Math Assessor's honest ratings: interaction homomorphisms at 6/10 novelty, commutation theorem at 5/10, DBSP impossibility at 4–7/10. The Skeptic is right: this is "a collection of B+ results" with one genuine novelty. The 7 reflects that a well-executed paper with clean algebraic insight, demonstrable cost savings, and honest ablation is a strong VLDB accept — not a paradigm shift, but a contribution that advances the field. A 9 requires categorical depth or completeness results we explicitly drop. |
| **Feasibility** | **6/10** | The Difficulty Assessor gives 50% implementation probability for the full hybrid approach; the Skeptic gives 55% publishable. The tiered scope provides protection: Tier 1 at ~38K LoC is achievable with disciplined execution. The 30% fatal risk on interaction homomorphisms, mitigated by the two-sorted escape hatch, brings feasibility to 6/10 — realistic with appropriate risk management, but not comfortable. The Visionary's 7/10 is optimistic; the raw numbers support 5–6. |

---

*This document is designed to directly seed the `theory/` directory (Results 1–5 map to proof obligations) and the `implementation/` directory (subsystem table maps to modules). Tier 1 completion criteria are testable, tier gates are calendar-enforced, and every algebraic claim is grounded in a specific engineering capability that the evaluation measures.*
