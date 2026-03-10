# Scavenging Synthesis: Algebraic Repair Calculus (proposal_00)

**Role**: Scavenging Synthesizer — salvage maximum value, find what carries the paper
**Date**: 2026-03-04
**Inputs**: Three prior evaluations (Skeptic, Mathematician, Community Expert), all proposal materials

---

## SCORES

| Axis | Score | Justification |
|------|-------|---------------|
| **1. Extreme and Obvious Value** | **6/10** | |
| **2. Genuine Difficulty as Software Artifact** | **6/10** | |
| **3. Best-Paper Potential** | **4/10** | |

**VERDICT: CONTINUE** — with delta annihilation as the load-bearing hero, two-sorted as the primary scope, and strict week-2 kill gates.

---

## Axis 1: EXTREME AND OBVIOUS VALUE — 6/10

### What's genuinely strong

The pain is real and enormous. 40–60% of data engineer time on maintenance is not disputed by anyone. Schema evolution is the #1 cause of pipeline breakage. Every organization with a data warehouse lives this problem weekly. The addressable market is thousands of companies, millions of wasted engineering hours.

**Delta annihilation is the value anchor.** When `ADD_COLUMN(priority)` propagates through `SELECT customer_id, SUM(amount)`, the delta is provably zero and the stage is pruned. This is something dbt *cannot* do — it will recompute every transitively downstream stage. The algebra enables detecting "this perturbation has zero effect here" through symbolic reasoning about operator semantics, not just lineage tracing. This is immediately understandable to a practitioner: "37 stages affected by lineage; 8 actually need recomputation; 29 provably unaffected." That's a sentence that sells a paper *and* a tool.

**The commutation theorem provides a guarantee no competitor offers.** For pipelines in Fragment F, incremental repair = full recomputation. This eliminates "did my fix break something else?" — the dominant anxiety of on-call data engineering. No LLM, no heuristic, no existing tool can make this promise. The guarantee is machine-checkable at runtime ("your pipeline is 87% in Fragment F; those stages get exact correctness certificates").

### What limits the score

- **2–5× cost claim is pure prediction.** Zero benchmarks, zero empirical evidence. If annihilation rate is <15% on real pipelines, the cost story collapses. All three prior evaluators flagged this.
- **Modern dbt (1.6+) with column-level lineage** narrows the gap. The marginal improvement is likely 1.5–3×, not 2–5×. The proposal benchmarks against the old dbt baseline.
- **"Provably correct" is a researcher's value proposition.** Practitioners want green checkmarks, not algebraic guarantees. A well-tested heuristic that's right 95% of the time is often indistinguishable in practice.
- **LLM erosion**: 60–70% of simple perturbations will be LLM-handled within 2 years, narrowing the addressable space to hard compound cases whose frequency is unverified.

### Strongest possible framing

"First system that can *prove* a pipeline stage is unaffected by an upstream schema change — and use that proof to skip unnecessary recomputation." Lead with the concrete optimization (annihilation), not the abstract algebra (interaction homomorphisms). The algebra is the *mechanism*; the annihilation is the *value*.

### Salvage if full vision fails

Even if three-sorted coherence completely fails and Fragment F covers only 40% of stages:
- **Two-sorted (Δ_S, Δ_D) annihilation** still works for the dominant perturbation class (schema evolution + data changes).
- **Bounded deviation ε** for non-F stages still provides information no existing tool offers.
- A paper titled "Delta Annihilation: Schema-Aware Pruning for Incremental Pipeline Repair" with an honest ablation study is a solid VLDB contribution.

---

## Axis 2: GENUINE DIFFICULTY AS SOFTWARE ARTIFACT — 6/10

### What's genuinely hard

**1. Interaction homomorphism coherence through real SQL operators (8/10 local difficulty).**
48 coherence conditions across 8 operators × 3 sorts × 2 interaction maps. The hexagonal condition `push_f(φ(δ_s)(δ_d)) = φ(push_f^S(δ_s))(push_f^D(δ_d))` must hold per-operator. The JOIN and GROUP BY cases with renamed/widened columns are non-trivial. A single error invalidates the commutation theorem for pipelines using that operator. No existing template, library, or prior system provides guidance.

**2. SQL column-level lineage at ≥95% recall (7/10 local difficulty).**
The Mathematician correctly identifies this as the *underrated* difficulty center. Column-level lineage through correlated subqueries, lateral joins, window functions with PARTITION BY, and CTEs requires a scope-aware semantic visitor that sqlglot's AST alone doesn't give you. This is foundation-critical: if lineage is wrong, "provably correct" becomes "provably wrong." The failure mode is silent data corruption — the worst kind.

**3. Non-monotone DP for annihilation-aware planning (7/10 local difficulty).**
Delta annihilation breaks standard optimal substructure. When a subtree's cost drops to zero because its delta annihilates, subproblems interact in non-standard ways. The algorithm must be invented, not adapted.

### What deflates the difficulty

- **Massive library leverage.** sqlglot provides ~70% of the analyzer, networkx ~60% of the graph layer, scipy/HiGHS ~30% of the planner, DuckDB ~80% of the executor. The genuinely novel code is ~12–15K LoC (all three evaluators agree), not the claimed 22.7K.
- **Most of the 48 coherence conditions are tedious, not hard.** ~30 are straightforward definitional calculations (SELECT, FILTER, UNION). The genuine difficulty concentrates in 5–8 cases involving JOIN/GROUP BY/WINDOW with cross-sort interactions.
- **The Python idiom matcher is pattern matching (5/10).** The type system is standard refinement types. The executor is DuckDB checkpoint/rollback. These are competent engineering, not research-grade challenges.
- **The Skeptic's "12–15K genuinely hard LoC embedded in ~25K infrastructure" is accurate.** Substantial but not extraordinary for a research prototype.

### What carries the difficulty story

The *integration risk* is the real difficulty multiplier. Each subsystem must be correct for the end-to-end guarantee to hold. A subtle bug in SQL lineage extraction → false annihilation detection → silent data corruption. The coherence conditions aren't just mathematical exercises — they're the load-bearing joints where the entire correctness story can fail. This is the kind of difficulty that produces good papers: systems difficulty where the math and engineering must agree exactly.

---

## Axis 3: BEST-PAPER POTENTIAL — 4/10

### The honest assessment

This is a "collection of B+ results" paper — the proposal's own accurate self-description. No single theorem exceeds 7/10 novelty. The evaluator consensus:

| Contribution | Honest Novelty | Status |
|---|---|---|
| Interaction homomorphisms | 6/10 | Indexed monoid actions from universal algebra — new domain, not new math |
| Commutation theorem | 5/10 | Structural induction with new hypothesis — standard technique |
| DBSP impossibility | 2–7/10 (55–60% trivially true) | 3/4 encoding failures are definitional; only tagged-union may be deep |
| Delta annihilation | 6/10 | Genuine algorithmic contribution, undersold by the proposal |
| Complexity results | 4/10 | Textbook reductions |
| Fragment F decidability | 3/10 | Syntactic check — essential infrastructure, not a theorem |

**Best-paper probability: 5–8%.** All three evaluators converge here. The proposal's own final_approach.md gives 8%. The theory/approach.json's 25–35% was retracted.

### What could push best-paper

The *only* path to best-paper is a compelling demo of delta annihilation that makes a reviewer viscerally feel the difference:

> "This TPC-DS pipeline has 99 stages. Schema change: ADD COLUMN to dimension table. dbt recomputes 37 downstream stages. Our system proves 29 have zero delta, touches 8. Identical output. 4.6× cost reduction. Zero correctness violations across 10M property tests."

If the numbers land like that, and the ablation against an 80%-heuristic baseline shows the algebra catches cases the heuristic misses, a reviewer might champion it. But this requires empirical results that don't yet exist.

### What blocks best-paper

1. **No paradigm shift.** This is not Spark, Naiad, or DBSP. It is a well-scoped niche contribution.
2. **The "80% without algebra" problem.** If a well-engineered heuristic system (column-level lineage + repair rules) achieves 80% of the capability, the marginal value of the algebra must be demonstrated concretely. If annihilation rate is low, the margin is thin.
3. **DBSP impossibility is the weakest "theorem."** If it's trivially true (majority probability), the paper loses its sharpest positioning tool.
4. **Fragment F coverage may be disappointing.** If <50% of real analytical SQL stages are in F, the "provably correct" headline applies to less than half the pipeline.

### Realistic target

**Solid accept at VLDB/SIGMOD.** 50–100 citations over 5 years. This is a good outcome for a well-executed systems paper with a clean algebraic insight. The paper will be cited by the IVM community, the pipeline maintenance tooling community, and anyone working on formal methods for data systems.

---

## WHAT'S GENUINELY STRONG ENOUGH TO CARRY A PAPER

### Tier 1: Carry the paper alone (each sufficient for a submission)

1. **Delta annihilation + ablation study.** "We detect when upstream perturbations have zero effect at pipeline stages, skip those stages, and prove the result is identical to full recomputation." With a strong ablation showing ≥30% annihilation rate on TPC-DS and concrete cost savings over dbt, this alone is a paper. The algebra is the mechanism; the annihilation is the punchline.

2. **Commutation theorem for Fragment F + decidability.** "For the decidable fragment F of SQL pipelines (covering X% of stages in TPC-DS), incremental repair is provably equivalent to full recomputation." Even if the algebra is two-sorted, even if annihilation rates are modest, a machine-checkable correctness guarantee for incremental pipeline repair is novel and useful. No existing tool offers this.

### Tier 2: Strengthen the paper significantly (additive value)

3. **Two-sorted (Δ_S, Δ_D) interaction homomorphisms with coherence verification.** Formalizing how schema deltas and data deltas interact through relational operators, with 100K+ property tests per operator, provides the algebraic backbone. Not a standalone contribution, but makes the annihilation and commutation results principled rather than ad-hoc.

4. **Non-monotone DP planner exploiting annihilation boundaries.** If the algorithm is novel (not just standard DP with a zero-cost shortcut), the algorithmic contribution adds depth. The annihilation-boundary restoration of optimal substructure is the specific claim that could be interesting.

### Tier 3: Nice to have (marginal value)

5. **DBSP encoding impossibility.** Only if the tagged-union case yields a genuine asymptotic separation. Otherwise, demote to a design-motivation paragraph.

6. **Three-sorted extension (Δ_Q).** Only if coherence holds. Insurance policy, not a primary contribution.

---

## STRONGEST POSSIBLE FRAMING

**Title**: "Delta Annihilation: Algebraic Pruning for Provably Correct Incremental Pipeline Repair"

**Abstract hook**: "When an upstream schema changes, how many downstream stages actually need recomputation? We introduce delta annihilation — a static analysis that proves when a perturbation has zero effect at a pipeline stage — grounded in a two-sorted delta algebra with interaction homomorphisms connecting schema evolution and data changes. For pipelines in a decidable deterministic fragment (covering X% of SQL stages), our bounded commutation theorem guarantees that incremental repair produces identical results to full recomputation. On TPC-DS pipelines, delta annihilation prunes Y% of unnecessary stages, achieving Z× cost reduction over lineage-aware selective recomputation."

**Key narrative moves**:
- Lead with annihilation (the concrete win), not the algebra (the mechanism)
- Frame as "schema-aware IVM" — extending the IVM tradition to handle the perturbation class it can't
- Two-sorted is the *contribution*, not a fallback — most perturbations are schema + data
- The ablation against the 80%-heuristic is the centerpiece experiment
- Honest Fragment F coverage reporting converts a potential weakness into intellectual honesty that reviewers respect

---

## SALVAGE PLAN IF FULL VISION FAILS

### If three-sorted coherence fails entirely (30–40% probability)
→ Two-sorted (Δ_S, Δ_D) algebra. Drop Δ_Q to future work. Paper becomes "schema-aware incremental pipeline maintenance with formal correctness guarantees." Still novel: no existing system provides this for schema deltas. **Publishable at VLDB.**

### If Fragment F covers <50% (25–35% probability)
→ Reframe around delta annihilation and bounded deviation ε, not "provably correct." Lead sentence becomes "we *bound* the deviation from full recomputation and *eliminate* provably-zero stages." The ε story + annihilation is still a contribution. **Publishable at VLDB with weaker framing.**

### If annihilation rate is <15% on TPC-DS (unknown probability, the make-or-break empirical question)
→ Reframe around correctness guarantees only. "Even if you must recompute the same stages, you now *know* the result is correct." Weaker paper. **Publishable at CIDR/workshop, marginal at VLDB.**

### If DBSP impossibility is trivially true (55–60% probability)
→ Demote to one paragraph of design motivation. Zero impact on the rest of the paper. **No salvage needed — just honest framing.**

### Worst realistic case (two-sorted + low annihilation + trivial DBSP)
→ "Schema-Aware Incremental View Maintenance with Bounded Correctness Guarantees." A modest contribution formalizing schema deltas in the IVM framework, with the commutation theorem for Fragment F as the main result. **Publishable at CIDR or industrial track. Not VLDB main.**

---

## PROBABILITY ASSESSMENT

| Outcome | Probability |
|---------|-------------|
| Publishable at VLDB/SIGMOD (main track) | 50–55% |
| Publishable at CIDR/workshop/industrial track | 20% |
| Useful artifact, no top-venue paper | 15% |
| Complete failure | 10% |
| Best paper at VLDB/SIGMOD | 5–8% |
| P(at least one critical risk materializes) | 78–85% |

---

## VERDICT: **CONTINUE**

### Rationale

**Three things that are genuinely strong enough to carry a paper:**

1. **Delta annihilation is real, novel, and measurable.** No existing tool can prove a pipeline stage is unaffected by an upstream perturbation and prune it. dbt recomputes all transitively affected stages. SQLMesh approximates column-level lineage but doesn't reason symbolically about zero-delta propagation through operators. The algebra makes annihilation *provably sound*, not heuristic. This single capability justifies the algebraic framework.

2. **The commutation theorem provides a guarantee that has no competitor.** "Incremental repair = full recomputation" for a decidable fragment is a claim no LLM, no heuristic tool, and no existing IVM system can make for schema-evolving pipelines. The guarantee is machine-checkable. Even if Fragment F is smaller than hoped, the *existence* of a provably correct fragment is a novel contribution.

3. **The two-sorted escape hatch makes failure *graceful*, not catastrophic.** If three-sorted fails, two-sorted (Δ_S, Δ_D) covers 90%+ of real perturbations and is still a novel algebraic contribution. If annihilation rates are low, correctness guarantees alone are still publishable. If DBSP impossibility is trivial, it was never the load-bearing result. The degradation path is unusually well-defined for a research proposal.

### Why CONTINUE and not STRONG CONTINUE

- 50% publishable probability is coin-flip territory
- No single result exceeds 7/10 novelty
- The make-or-break empirical question (annihilation rate) is unanswered
- 78–85% chance of at least one critical-path failure
- The "80% without algebra" challenge is the most dangerous argument and has not been empirically rebutted

### Mandatory conditions (inheriting from prior evaluations, endorsed)

1. **Two-sorted (Δ_S, Δ_D) is primary scope.** Three-sorted only if coherence holds by 40% mark.
2. **DBSP impossibility resolved week 1.** Deep → theorem. Trivial → paragraph.
3. **Coherence verified for SELECT, FILTER, JOIN by week 2.** Failure on any → immediate pivot.
4. **Delta annihilation is the headline.** Not interaction homomorphisms.
5. **Annihilation rate ≥30% on TPC-DS** as binding evidence gate.
6. **Build the 80% heuristic baseline first.** Ablation study is the key experiment.
7. **Fragment F coverage reported honestly.** No marketing.

---

*Evaluation complete. As Scavenging Synthesizer: this proposal has one genuinely novel capability (delta annihilation), one genuinely unique guarantee (bounded commutation), and a well-defined degradation path. The ceiling is modest (solid VLDB accept, not best paper), but the floor is protected. That's enough to continue.*
