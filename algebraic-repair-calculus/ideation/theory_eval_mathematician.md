# Theory Evaluation: Algebraic Repair Calculus (proposal_00)

**Evaluator**: Deep mathematician — evaluating load-bearing math, not ornament
**Method**: Three-expert adversarial team (Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer) with cross-critique synthesis
**Date**: 2026-03-04

---

## SCORES

| Dimension | Score | Justification |
|-----------|-------|---------------|
| **Extreme Value** | **7/10** | Pain is undeniable (9/10). Marginal value over modern dbt with column-level lineage is likely 1.5–3× (not the claimed 2–5×). Correctness guarantees provide real differentiation at academic venues. Delta annihilation is the true value driver. |
| **Genuine Software Difficulty** | **7/10** | ~12K genuinely novel LoC (not the claimed 22.7K). Delta algebra is laborious (48 case analyses) but not breakthrough-hard — 6–7/10, not 9/10. SQL semantic analysis is the hidden difficulty center (7–8/10), underrated by the proposal. System-integration risk (analyzer bugs → false correctness certificates) elevates total difficulty. |
| **Best-Paper Potential** | **5/10** | No theorem exceeds 7/10 novelty. "Collection of B+ results" is the proposal's own accurate self-assessment. P(best paper) ≈ 5%. Strong accept at VLDB plausible (28%). The engineering story (annihilation ablation) is the most likely path to reviewer excitement, not the theory. |
| **Laptop-CPU Feasibility** | **8/10** | DuckDB + sqlglot + scipy stack is well-suited. TPC-DS SF=10 trivial on laptop. LP solver for 500-variable problems is instant. Property-based testing (3–28 hrs) is the only wall-clock concern. |
| **Overall Feasibility** | **5/10** | P(publishable) = 50%. 78% chance of ≥1 critical-path failure. Single-developer timeline: 7–13 months. Tiered scope and two-sorted fallback provide genuine protection, but each fallback weakens the paper. |

**Composite: 32/50**

---

## LOAD-BEARING MATH ASSESSMENT

This is the central question: **Is the math load-bearing, or is it ornament?**

### The Irreducible Mathematical Core (2.5 results)

**1. Delta Annihilation Soundness (T5) — 9/10 load-bearing. THE result.**

When `ADD_COLUMN(priority)` propagates through `SELECT customer_id, SUM(amount)` and the new column isn't referenced, the delta is zero and the stage is pruned. This is what dbt cannot do. This is what no heuristic can replicate without reimplementing the algebra under a different name. The three-sorted structure is necessary specifically to enable annihilation reasoning across schema/data/quality boundaries. Without T5, the system degrades to dbt-level performance and the 50K LoC buys nothing.

**2. Bounded Commutation Theorem (T2) — 7/10 load-bearing. The correctness backbone.**

For pipelines in Fragment F (deterministic, order-independent): incremental repair = full recomputation. This is THE unique selling proposition vs. every heuristic alternative, including LLMs. The proof is structural induction on the pipeline DAG — standard technique, but the depth lives in precisely characterizing Fragment F for real SQL (NULLs, three-valued logic, implicit coercions). If F handles SQL's semantic dark corners, this is 7/10 novelty. If F is just "pure bag algebra, no NULLs," it drops to 4/10.

**Critical gate**: Fragment F coverage on real pipelines. The proposal targets 60–75%. Our assessment: 40–55% on production workloads with FLOAT aggregations and timestamp columns. If <50%, the correctness guarantee applies to only half the pipeline.

**3. Hexagonal Coherence (T1) — 5/10 load-bearing. Essential infrastructure, not a theorem.**

48 case analyses (8 operators × 3 sorts × 2 interaction maps) verifying that the algebra's definitions are internally consistent. The Skeptic is right: this is a verification task, not a theorem. Each case is a calculation, not a proof requiring insight. But it IS necessary infrastructure — without it, T2 and T5 cannot be stated. The genuine difficulty hides in 3–5 hard cases: JOIN with renamed columns, GROUP BY with widened types, WINDOW with quality constraints.

**Resolution**: Present as a "coherence verification" or "verification lemma," not as a standalone theorem.

### Non-Load-Bearing Results (should not headline)

**DBSP Encoding Impossibility (T3) — 2/10 load-bearing. Ornamental positioning.**

55–60% probability of being trivially true. 3 of 4 encoding failure cases are trivially expected. Only the tagged-union case is potentially interesting, but the incrementality-loss proof is missing. Does not drive any implementation decision or enable any capability. **Invest at most 1 week; demote to remark if tagged-union argument doesn't crystallize.**

**Cost-Optimal Repair Planning (T4) — 3/10 load-bearing (potentially 6/10).**

NP-hardness via weighted set cover: textbook. DP for DAGs: textbook. LP relaxation: textbook. The one interesting element is the non-monotone DP handling delta annihilation boundaries — if worked out carefully, this is genuine algorithmic novelty.

**Interaction Composition Associativity (T6) — 1/10 load-bearing.**

One-line consequence of the homomorphism properties. Expected to hold. Not a theorem.

### Mathematical Depth Verdict

| Aspect | Rating |
|--------|--------|
| Mathematical novelty | **4/10** — known math (multi-sorted algebra, fibered categories, structural induction) in a genuinely new domain |
| Engineering novelty | **7/10** — the instantiation for SQL pipeline repair is novel and useful |
| Theory-practice bridge | **7/10** — the algebra enables capabilities (annihilation, provable correctness) that engineering alone cannot replicate |

**Bottom line**: This is "known math, new domain." The value is not deep mathematics — it is the *correct* algebraic model for a real problem, enabling optimizations that no heuristic alternative can match. That is a perfectly valid paper at VLDB/SIGMOD. It is not a paper for STOC/FOCS.

---

## THE "80% WITHOUT ALGEBRA" QUESTION

All three assessors agree: ~80% of practical value is achievable with ~15K LoC of heuristic engineering (tag propagation rules + column-level lineage + sequential application). The algebra adds:

1. **Delta annihilation** — the measurable differentiator (no heuristic can replicate)
2. **Provable correctness** for Fragment F (formal guarantee vs. "tested and seemed fine")
3. **Compound perturbation handling** (provably correct composition vs. "apply sequentially and hope")

**Verdict**: The "80% without algebra" is not fatal — it IS the ablation study's control condition. The paper's story must be: "here is what the algebra adds beyond what engineering rules can achieve." If the algebra adds nothing (annihilation rate ≈ 0, no compound errors caught), THEN the 80% alternative becomes fatal. The evidence gates below test this.

---

## FATAL FLAWS

**No fatal flaw sufficient to ABANDON was identified by any assessor.**

Potential fatal flaws, all mitigated:

| Risk | Probability | Mitigation | Residual Impact |
|------|-------------|------------|-----------------|
| Interaction homomorphisms don't compose for >3/8 operators | 30–40% | Two-sorted fallback (Δ_S, Δ_D only) | Paper weakens from "novel 3-sorted algebra" to "incremental schema-aware IVM" |
| 2–5× cost claim collapses to <1.5× vs modern dbt | 30–40% | Reframe around correctness, not cost | Paper viable but less exciting |
| Fragment F covers <50% of real stages | 25–35% | Computable ε bound for non-F stages | Correctness story weakens |
| DBSP impossibility is trivially true | 55–60% | Demote to remark | Loses positioning argument |

The proposal's tiered scope, calendar-based gates, and explicit fallbacks provide genuine protection against each risk.

---

## SPECIFIC CONCERNS

### Inflated Claims to Drop or Revise

1. **"Minimal algebraic structure required"** — Unsupported. Minimality requires a representation theorem (showing any correct system must contain an isomorphic structure). No such result is attempted. Replace with: "a sufficient algebraic structure."

2. **"6 theorems"** — Overstates by ~3×. Honestly: 2 core theorems (T2, T5), 1 verification lemma (T1), 1 probably-trivial observation (T3), 2 textbook applications (T4, T6).

3. **"22.7K novel LoC"** — Inflated by ~45%. Genuinely novel: ~12K. Novel-application-of-known-techniques: ~15K. The remaining ~8K is library glue and boilerplate.

4. **"9/10 difficulty for delta algebra engine"** — Inflated. Laborious (48 case analyses), not breakthrough-hard. Honest rating: 6–7/10. The SQL semantic analyzer (rated 6/10 by proposal) is actually the harder component at 7–8/10.

### The Modern dbt Baseline Problem

The proposal compares against dbt `--select` which is the *old* interface. Modern dbt (1.6+) has column-level lineage. The algebraic advantage specifically over modern-dbt is:
- Annihilation detection (column not referenced → zero propagation) — dbt can approximate this heuristically
- Formal correctness guarantee — dbt cannot provide this
- Cross-sort annihilation (schema change that doesn't affect quality constraints) — dbt doesn't model quality

The honest marginal improvement is likely **1.5–3×**, not 2–5×. The proposal should benchmark against the strongest realistic baseline.

---

## BINDING EVIDENCE GATES

Adopted from the Skeptic's assessment with one modification:

| Gate | Requirement | Deadline | Consequence of Failure |
|------|-------------|----------|----------------------|
| **G1** | Hexagonal coherence verified for SELECT, JOIN, FILTER with all delta sorts (100K+ property tests each) | 30% of implementation time | Fail >1 of 3 → retreat to two-sorted immediately |
| **G2** | Annihilation rate ≥30% on TPC-DS mixed perturbation corpus | 30% of implementation time | <30% → reframe value proposition around correctness, not cost |
| **G3** | SQL column-level lineage recall ≥85% on 30 TPC-DS queries | 30% of implementation time | <85% → simplify to table-level lineage + conservative all-to-all column deps |
| **G4** | DBSP impossibility non-triviality assessment for tagged-union case | 1 week | Trivially true → demote to remark, do not count as theorem |

---

## PROBABILITY ESTIMATES

| Metric | Proposal Self-Assessment | Team Adjudication |
|--------|--------------------------|-------------------|
| P(publishable result) | 55% | **50%** |
| P(strong accept at VLDB) | 25–35% | **28%** |
| P(best paper) | 8% | **5%** |
| P(≥1 critical-path failure) | ~70% | **78%** |

---

## VERDICT: **CONTINUE**

### Why CONTINUE

1. **Delta annihilation (T5) is a genuinely novel optimization** that no heuristic alternative can replicate. This single result justifies the algebraic approach.

2. **Bounded commutation (T2) provides a unique correctness guarantee** that no existing tool — including LLMs — can offer. For the 50–75% of pipeline stages in Fragment F, incremental repair is provably equivalent to full recomputation.

3. **The two-sorted fallback ensures a publishable result** even if the three-sorted algebra partially fails. The worst realistic outcome (two-sorted algebra + modest cost savings + honest ablation) is still a viable VLDB submission.

4. **The Skeptic's own verdict was CONTINUE.** Even the most adversarial assessment could not find a fatal flaw sufficient to justify ABANDON.

5. **The math is load-bearing, not ornamental.** It is not deep (4/10 novelty), but it is correct for the domain and enables capabilities — annihilation-based pruning, provable correctness — that engineering alone cannot replicate. The paper's value is the theory-practice bridge, which is how applied formal methods papers work at systems venues.

### Why This Is Not a Strong CONTINUE

1. **50% publishable probability** is sober. One in two attempts at this paper fails.
2. **No single result exceeds 7/10 novelty.** This is a "collection of B+ results" paper, not a best-paper contender.
3. **The marginal value over modern dbt is likely 1.5–3×, not 2–5×.** The cost story is weaker than presented.
4. **12K genuinely novel LoC** is substantial but the 22.7K claim is inflated.
5. **The "6 theorems" are actually 2 theorems, 1 verification lemma, and 3 non-novel items.**

### Recommendation

Proceed to implementation with the 4 binding evidence gates. Prioritize:
1. Delta annihilation implementation and benchmarking (the make-or-break result)
2. Fragment F characterization for real SQL
3. Ablation study design against the 80% heuristic baseline

Deprioritize: DBSP impossibility proof, type soundness proof, Tier 3 stretch goals.

Drop or revise: "minimal algebraic structure" claim, 9/10 difficulty self-rating for algebra engine, "6 theorems" framing.

---

*Evaluation produced by three-expert adversarial team with cross-critique synthesis. All scores cite specific evidence from proposal files. Disagreements adjudicated by weighing evidence quality, not averaging. Independent verifier signoff: all three assessors reached CONTINUE (Auditor unconditional, Skeptic conditional with 4 gates, Synthesizer unconditional with priority recommendations).*
