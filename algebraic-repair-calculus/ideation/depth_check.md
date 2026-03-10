# Depth Check: Algebraic Repair Calculus

**Slug:** `algebraic-repair-calculus`
**Evaluation method:** Three-expert adversarial panel (Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer) with cross-critique synthesis.

---

## Scores

| Axis | Score | Threshold | Status |
|---|---|---|---|
| 1. Extreme and Obvious Value | **7/10** | ≥ 7 | ✅ PASS |
| 2. Genuine Difficulty as Software Artifact | **6/10** | ≥ 7 | ❌ FAIL — amendments required |
| 3. Best-Paper Potential | **6/10** | ≥ 7 | ❌ FAIL — amendments required |
| 4. Laptop CPU + No Humans | **8/10** | ≥ 9 | ❌ FAIL — minor amendments |
| **Composite** | **27/40 (6.75)** | | **CONDITIONAL PASS — amendments applied** |

---

## Axis 1: Extreme and Obvious Value — 7/10

**The problem is real.** Pipeline maintenance consumes 40–60% of data engineer time (dbt Labs surveys, Fivetran incident reports). Schema evolution is the #1 cause of pipeline breakage. The gap is genuine: schema registries detect but don't repair; Great Expectations flags but doesn't fix; Materialize/Noria assume fixed schemas; dbt always does full recomputation. No tool unifies detection + repair + correctness guarantees.

**Partial LLM erosion.** Within 2 years, LLM-based tools will handle 60–70% of simple perturbations (column adds, renames, type widenings) interactively. However, LLMs cannot provide:
- **Correctness guarantees** (proving equivalence to full recomputation)
- **Compound perturbation handling** (schema change + quality violation simultaneously)
- **Cost-optimal repair planning** (minimizing reprocessing)
- **Autonomous continuous operation** (detect and repair without human intervention)

These hard cases justify formal methods and represent the durable value proposition.

**Baseline comparison is misleading.** The 10–50× speedup over full recomputation is technically correct but dishonest. Against best-practice selective recomputation (dbt `--select` with lineage awareness), the honest speedup is 2–5× for cost, with the unique advantage being formal correctness guarantees. The proposal must lead with correctness, not raw speedup.

**Graceful degradation.** Even a SQL-only, two-sorted version provides clear value. The worst-case outcome ("a really good dbt plugin with formal correctness guarantees") is still useful.

---

## Axis 2: Genuine Difficulty as Software Artifact — 6/10

**172K LoC is inflated by ~2.8×.** Subsystem-by-subsystem analysis using existing libraries (sqlglot, scipy, networkx, DuckDB):

| Subsystem | Claimed | Realistic | Assumptions |
|---|---|---|---|
| SQL Analysis (on sqlglot) | 22,000 | 7,000 | sqlglot parses all 4 dialects; custom semantic visitors for lineage |
| Python Analysis | 18,000 | 5,000 | Idiom-matching on 15–20 common pandas/PySpark ops, NOT abstract interpretation |
| Typed Dependency Graph | 14,000 | 5,000 | networkx for graph ops, custom type annotations |
| Type System | 12,000 | 5,000 | Refinement types over schemas, not full dependent types |
| Delta Algebra Engine | 20,000 | 9,000 | Core algebra + cost model + simplifier |
| Repair Planner | 16,000 | 6,000 | DP + LP relaxation via scipy/HiGHS |
| Executor | 14,000 | 5,000 | DuckDB-only backend, saga pattern |
| Quality Monitor | 13,000 | 4,000 | scipy.stats + streaming wrappers |
| Pipeline State | 10,000 | 3,000 | SQLite-backed checkpoint store |
| Eval & Tests | 33,000 | 12,000 | TPC-DS pipelines, property tests, benchmarks |
| **Total** | **172,000** | **~61,000** | **Using sqlglot, scipy, networkx, DuckDB** |

**Genuinely novel and hard subsystems:** Delta algebra engine (~9K), repair planner (~6K), interaction homomorphisms, and the Python idiom matcher (~5K). These total ~20–25K LoC of research-grade code. The remainder is engineering leveraging existing libraries.

**The 150K+ LoC threshold is not met.** Even with generous estimates, the necessary complexity is ~61K LoC. The proposal must either (a) honestly state the scope as ~60–75K LoC, or (b) dramatically expand the genuinely novel components.

**Python abstract interpretation is a scope bomb.** Sound analysis of arbitrary pandas code (pivots, dynamic column creation, eval(), UDFs) is an open research problem. The proposal treats it as one subsystem; it's actually a PhD thesis. Scoping to idiom-matching on common operations is the only feasible path.

**Three genuinely novel math contributions (not six).** The "6 mathematical contributions" overcounts:
- **Genuinely novel (3):** Interaction homomorphisms with propagation coherence, commutation theorem, DBSP expressiveness separation (if reformulated)
- **Solid but standard (2):** NP-hardness via weighted set cover, approximation via LP relaxation
- **Required infrastructure (1):** Type soundness (progress + preservation) — standard PL metatheory applied to new domain

---

## Axis 3: Best-Paper Potential — 6/10

**Base quality: 8/10.** The unification of schema evolution + IVM + data quality into a single algebra is a legitimately novel framing. The DBSP separation (if properly reformulated as an encoding impossibility) is a sharp, citable result. The complexity dichotomy is clean. The evaluation plan with property-based algebraic law verification is unusually rigorous.

**Risk-adjusted: 6/10.** Best papers require flawless execution across theory AND experiments. The probability of fully executing all mathematical contributions AND building the system AND running comprehensive evaluations is ~20–25%. Outcome probability distribution:

| Outcome | Probability | Description |
|---|---|---|
| Tier 1: Best paper at VLDB/SIGMOD | ~20% | Everything lands: clean commutation proof, strong DBSP separation, 10× speedup over IVM baseline |
| Tier 2: Strong accept, no best paper | ~45% | Algebra works for SQL-only; Python is idiom-matching; DBSP separation proved in strong form; 5–20× over full recompute |
| Tier 3: Accept after revision | ~25% | Commutation needs restrictive assumptions; quality deltas underdeveloped; smaller benchmarks |
| Tier 4: Workshop/reject | ~10% | Algebra is sound but complexity results are standard; Python doesn't work; DBSP separation trivially stated |

**The DBSP separation must be reformulated.** As currently stated ("construct a scenario DBSP can't handle"), it is trivially true by construction — DBSP was never designed for schema changes. The interesting and publishable version: prove that no encoding of schema deltas into DBSP's data domain preserves both type safety and incrementality simultaneously. This requires showing DBSP circuits are parametric in data but not in schema.

**The commutation theorem needs explicit scope.** The claim `apply(repair(σ), state(G)) = recompute(evolve(G, σ))` for ALL pipelines is likely false for: floating-point aggregations (order-dependent), non-deterministic functions (RAND, NOW, UUID), external API calls. Honest formulation: exact commutation for the deterministic, order-independent fragment; bounded deviation guarantee otherwise.

**"6 math contributions" is marketing.** Honest framing (3 novel + 2 standard + 1 infrastructure) builds reviewer trust. The 3 genuinely novel contributions are strong enough to carry the paper.

---

## Axis 4: Laptop CPU + No Humans — 8/10

**Fundamentally CPU-friendly.** All components (static analysis, symbolic term rewriting, combinatorial optimization, streaming statistics) are inherently CPU-bound workloads. No ML, no training, no GPUs needed. Evaluation uses generated benchmarks and public schema evolution traces — zero human annotation.

**Minor concerns:**
- **ILP solver scalability:** For cyclic pipelines (the <10% case), HiGHS/GLPK may take minutes per instance on a laptop. With 500 synthetic topologies including cycles, the ILP could strain the time budget. Mitigation: fall back to LP relaxation approximation for large cyclic instances.
- **TPC-DS at SF=100 on 16GB RAM:** Marginal for large join materializations. SF=10 is safe. Recommendation: default to SF=10, include SF=100 as stretch.
- **Time budget:** The "under 4 hours" claim is optimistic. Realistic estimate: 6–8 hours for the full suite. Not a deal-breaker but should be honestly stated.
- **Abstract interpretation convergence:** On complex SQL with deeply nested CTEs, the fixed-point computation may require widening operators. The proposal doesn't discuss iteration bounds.

**To reach 9/10:** Explicitly state 6–8 hour evaluation time budget. Default to SF=10 with SF=100 as stretch. Include ILP timeout with fallback to approximation. Drop abstract interpretation in favor of idiom-matching (eliminates convergence concern).

---

## Axis 5: Fatal Flaws

### Flaw 1 (CRITICAL): Python Abstract Interpretation Is a Scope Bomb
Sound analysis of arbitrary pandas/PySpark code is an open research problem. Operations like `pivot()`, `melt()`, dynamic column creation, and UDFs create schemas dependent on runtime data values. Sound over-approximation degrades to "everything depends on everything," collapsing to full recomputation. The proposal's speedup claims require precise lineage; sound abstract interpretation guarantees imprecise lineage. These are in direct contradiction.

**Fix:** Replace with idiom-matching on 15–20 common pandas/PySpark operations. Treat unknown patterns as opaque (conservative all-to-all dependency). State coverage target: ≥85% of column-level dependencies in real dbt projects.

### Flaw 2 (CRITICAL): DBSP Separation Is Trivially Stated
The current formulation ("construct a scenario DBSP can't handle") is trivially true. DBSP operates on fixed schemas; of course it can't handle schema changes. A DBSP reviewer would dismiss this as a category error.

**Fix:** Reformulate as encoding impossibility: prove that no data-domain encoding of schema deltas preserves DBSP's type safety AND incrementality guarantees simultaneously.

### Flaw 3 (IMPORTANT): LoC Inflation Erodes Credibility
172K LoC is inflated by ~2.8× over realistic estimates using existing libraries. This signals either inexperience with available tooling or deliberate padding. Either damages reviewer trust.

**Fix:** State honest scope (~60–75K LoC) with explicit library dependencies. Define tiered scope (MVP at ~35K, full at ~61K, stretch at ~75K).

### Flaw 4 (IMPORTANT): Commutation Theorem Overclaims
Bitwise equivalence for all pipelines is false for floating-point aggregations, non-deterministic functions, and external calls. A reviewer who spots this attacks the headline theorem.

**Fix:** State exact commutation for deterministic, order-independent fragment. Provide bounded deviation guarantee for pipelines outside this fragment.

### Flaw 5 (MINOR): "ACID-like" Mischaracterization
Saga pattern explicitly sacrifices isolation and atomicity. Calling it "ACID-like" will trigger DB-theory reviewers. The executor provides pipeline-level consistency (no consumer sees partial repairs) via checkpoint/rollback — a legitimate guarantee, just not ACID.

**Fix:** Replace with "saga-based eventual consistency with compensating actions."

---

## Required Amendments Summary

| # | Amendment | Severity | Impact |
|---|---|---|---|
| 1 | Scope Python to idiom-matching, drop abstract interpretation | CRITICAL | Removes fatal flaw #1; reduces scope by ~13K LoC |
| 2 | Reformulate DBSP separation as encoding impossibility | CRITICAL | Transforms trivial claim into publishable theorem |
| 3 | Honest LoC (~61K) with tiered scope (MVP/full/stretch) | IMPORTANT | Builds credibility; ensures publishable result under pressure |
| 4 | Bounded commutation theorem with explicit fragment | IMPORTANT | Pre-empts reviewer attack on headline result |
| 5 | Lead with correctness guarantees, not 10–50× speedup | IMPORTANT | Honest baseline comparison; stronger value proposition |
| 6 | Two SQL dialects (Postgres + Spark SQL), not four | IMPORTANT | Reduces low-novelty engineering by ~40% |
| 7 | Drop "ACID-like" → saga-based consistency | MINOR | Avoids reviewer objection |
| 8 | Reframe "6 math contributions" as "3 novel + 2 standard + 1 infrastructure" | MINOR | Builds trust through intellectual honesty |
| 9 | State 6–8 hour eval time; default SF=10 | MINOR | Honest laptop feasibility claim |

**Verdict:** All amendments have been applied to the crystallized problem statement. With amendments, the proposal is a strong VLDB/SIGMOD submission with ~45% probability of strong accept and ~20% probability of best-paper selection.

---

## Expert Signoff

- **Independent Auditor:** Scores are evidence-based and defensible. Amendments address all identified risks. ✅
- **Fail-Fast Skeptic:** Fatal flaws are resolved by scoping (Python), reformulation (DBSP), and honest framing (LoC, commutation). The core idea survives aggressive testing. ✅
- **Scavenging Synthesizer:** Graceful degradation paths preserved. Hidden value (standalone static analyzer, typed dep graph as IR, delta algebra generalization) noted for future exploitation. Tiered scope ensures publishable minimum. ✅
