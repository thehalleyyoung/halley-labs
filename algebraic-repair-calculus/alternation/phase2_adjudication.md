# Phase 2 Adjudication: Algebraic Repair Calculus (proposal_00)

**Role**: Team Lead — adversarial cross-critique mediation
**Date**: 2026-03-04
**Inputs**: Independent Auditor (CONTINUE weak), Fail-Fast Skeptic (ABANDON), Scavenging Synthesizer (CONTINUE)

---

## Disagreement 1: CONTINUE vs ABANDON

### Skeptic's kill case

The Skeptic's strongest arguments:
1. **Zero lines of code exist.** The theory is 115KB of JSON; the implementation is 0 LoC. Every claim is unverified.
2. **The 80% heuristic argument.** A simple rule engine (column added → skip if unused downstream; column renamed → rewrite references; type widened → cast) handles ~80% of real schema perturbations. This costs ~2K LoC and 2 weeks, not 12–15K LoC and 4+ months.
3. **30–40% FATAL risk on coherence.** The theory's own `risk_assessment` gives 40% probability that hexagonal coherence verification hits blocking complexity. If coherence fails, the commutation theorem is unproven, and the paper's central guarantee evaporates.
4. **Build the heuristic first.** If the heuristic already captures 80% of the value, the marginal return on 10× more engineering effort is unproven.

### Auditor/Synthesizer's counter-case

1. **Delta annihilation is categorically different from heuristics.** A heuristic says "column `priority` was added; downstream stages probably don't use it." The algebra *proves* `push_SELECT(ADD_COLUMN(priority))` = ∅ when `priority ∉ referenced_columns(SELECT)`. The heuristic guesses; the algebra certifies. This is not a quantitative difference — it is a qualitative one. When a pipeline has 200 stages and a wrong guess causes silent data corruption, the guarantee matters.
2. **The two-sorted fallback is real.** Even if three-sorted coherence fails entirely and Fragment F covers only 40% of stages, a paper on two-sorted (Δ_S, Δ_D) annihilation with bounded commutation for an honest fragment is still a genuine VLDB contribution. The Synthesizer's proposed fallback title — "Delta Annihilation: Schema-Aware Pruning for Incremental Pipeline Repair" — is publishable.
3. **The DBSP impossibility is a structural result.** Even if 3 of 4 encoding strategies are trivially impossible, the tagged-union case requires non-trivial analysis. And the *existence* of the impossibility — schema deltas cannot be reduced to data deltas without cost — justifies a new algebraic sort. This is positioning, not just a theorem.

### My adjudication

**The Skeptic is right that building a heuristic first is good engineering practice, but wrong that it invalidates the research contribution.** The proposal isn't competing with heuristics — it's establishing a formal framework that explains *why* heuristics work when they do and *when* they fail. The heuristic catches 80% of cases; the algebra catches 80% + the 15% of hard compound cases where heuristics silently corrupt data. That 15% is precisely where the research contribution lives.

However, the Skeptic's risk assessment is uncomfortably accurate:
- 40% risk on coherence (theory's own number)
- 70% risk on implementation complexity underestimation (theory's own number)
- 0 LoC written

**The two-sorted fallback is the decisive factor.** It converts a 40% FATAL risk into a graceful degradation. If three-sorted fails → two-sorted paper. If Fragment F is only 40% → honest coverage reporting. If annihilation rate is low → the impossibility theorem + bounded commutation still contribute.

**Verdict on Disagreement 1: CONTINUE (weak), with mandatory kill gates.**

The Skeptic's ABANDON is too aggressive because it ignores the two-sorted fallback floor. But the Skeptic's *conditions* should be adopted: (1) Week 2 kill gate — if two-sorted coherence isn't verified for JOIN and GROUP BY, ABANDON. (2) Week 4 kill gate — if annihilation rate < 15% on TPC-DS corpus, pivot to pure theory paper or ABANDON.

---

## Disagreement 2: Value Scoring (3 vs 5 vs 6)

### Evidence inventory

**For Value = 3 (Skeptic):**
- Modern dbt 1.6+ with column-level lineage already does selective recomputation. Marginal improvement is 1.2–1.5× in common cases.
- "Provably correct" matters to researchers, not practitioners. A 95%-accurate heuristic is operationally equivalent.
- LLM-based repair will handle 60–70% of simple cases within 2 years.
- The 2–5× cost claim has zero empirical backing.

**For Value = 5 (Auditor):**
- Delta annihilation is a capability no existing tool provides — *proving* a stage is unaffected, not *guessing*.
- The commutation theorem provides a machine-checkable correctness certificate. This is qualitatively different from "tests pass."
- Compound perturbations (schema change + quality violation simultaneously) are genuinely unhandled by any existing system.

**For Value = 6 (Synthesizer):**
- The problem is undeniably real and costs hundreds of millions annually.
- Delta annihilation + commutation theorem together create a new capability class.
- The framework unifies three fragmented subfields.

### Who has stronger evidence

**The Skeptic is right about the marginal value gap narrowing.** Modern dbt with column-level lineage is a much tighter baseline than the proposal acknowledges. The 2–5× claim is pure speculation with zero code to back it.

**The Synthesizer is right that annihilation is qualitatively different.** dbt can trace lineage, but it cannot prove `push_f(δ) = 0`. When dbt says "these 37 stages are downstream," it recomputes all 37. When the algebra says "29 of those have zero delta," it skips 29. That's not a marginal improvement — it's a new operation that didn't exist before.

**The Auditor's middle position is closest to correct.** The pain is real (supporting 6), but the marginal value over modern tools is unproven (capping at 5). Compound perturbations are a genuine gap but their *frequency* is unverified (the proposal estimates 10–15% of episodes, but this is a prediction, not data).

### Resolved score: **5/10**

The Auditor's score holds. Delta annihilation saves the value proposition from being "a nice dbt plugin" (Skeptic) but the absence of any empirical validation caps it at 5. If annihilation rates prove to be >25% on real pipelines, this climbs to 6–7. If <15%, it drops to 3–4.

---

## Disagreement 3: Difficulty Scoring (4 vs 6 vs 6)

### Evidence inventory

**For Difficulty = 4 (Skeptic):**
- All three evaluators agree the genuinely novel code is ~12–15K LoC.
- Massive library leverage: sqlglot, networkx, scipy, DuckDB, HiGHS do heavy lifting.
- "12–15K of novel code on top of libraries is a master's thesis, not impressive for VLDB."
- 30 of 48 coherence conditions are tedious-not-hard definitional calculations.

**For Difficulty = 6 (Auditor & Synthesizer):**
- The 5–8 hard coherence conditions (JOIN/GROUP BY/WINDOW with cross-sort interactions) are genuinely novel.
- SQL column-level lineage at ≥95% recall through correlated subqueries, CTEs, and window functions is hard — and the foundation is load-bearing.
- Integration risk: every subsystem must be correct for the end-to-end guarantee. A subtle lineage bug → false annihilation → silent corruption.
- The DP algorithm for annihilation-aware planning breaks optimal substructure and must be invented, not adapted.

### Who has stronger evidence

**The Skeptic undervalues integration difficulty.** The claim "12–15K LoC is a master's thesis" is true in isolation but misleading. A master's thesis with 12K LoC where *every component must be formally correct for the system guarantee to hold* is harder than 50K LoC of independent features. The failure mode isn't "my feature doesn't work" — it's "my system silently produces wrong data and the correctness certificate says it's right."

**The Auditor/Synthesizer correctly identify SQL lineage as the underrated difficulty center.** This is the subsystem most likely to contain subtle bugs, most load-bearing for correctness, and least helped by existing libraries (sqlglot gives you the AST but not semantic column-level lineage through correlated subqueries).

**However, the Skeptic is right that library leverage deflates the difficulty.** Without sqlglot, this is a 9/10 difficulty. With sqlglot, the SQL analysis is a 7/10 locally. The delta algebra engine is genuinely novel at 8/10 locally but is 9K LoC of symbolic manipulation — hard but tractable.

### Resolved score: **5/10**

I split the difference but lean toward the Skeptic more than the optimists might expect. The integration difficulty is real (pushing above 4), but the heavy library leverage and the fact that most coherence conditions are definitional (pushing below 6) balance out. A 5 says: "This is a strong systems research project requiring genuine novelty in 3–4 components, built on substantial library foundations." It is harder than a master's thesis but not at the frontier of systems difficulty. The 12–15K of genuinely novel code must be *correct*, not just functional — that's the difficulty multiplier the Skeptic underweights.

---

## Disagreement 4: Best-Paper Scoring (2 vs 4 vs 4)

### Evidence inventory

**For Best-Paper = 2 (Skeptic):**
- No individual result exceeds B+ quality.
- Interaction homomorphisms are indexed monoid actions in a new domain — new *application*, not new *math*.
- Commutation theorem is structural induction — standard technique.
- DBSP impossibility is 55–60% likely to be trivially true (3 of 4 encodings fail definitionally).
- Best-paper probability 3–5%. Noise floor.

**For Best-Paper = 4 (Auditor & Synthesizer):**
- "Collection of B+ results" in a single coherent system can be a VLDB accept.
- Delta annihilation as a practical optimization makes the theory tangible.
- Unifying three fragmented subfields (schema evolution, data quality, IVM) is the kind of synthesis VLDB rewards.
- Best-paper probability 5–8%.

### Who has stronger evidence

**The Skeptic is right that no individual result is best-paper quality.** The theorems are individually 4–6/10 novelty. None would stand alone as a best-paper contribution. The Skeptic's characterization as "B+ results" is accurate.

**The Auditor/Synthesizer are right that the *combination* can exceed the sum.** VLDB best papers are often systems papers where no single theorem is groundbreaking but the system as a whole demonstrates something new. The annihilation optimization + commutation guarantee + impossibility separation is a stronger package than any individual piece.

**However**, best-paper at VLDB/SIGMOD requires either: (a) a stunning empirical result (10× improvement that makes reviewers gasp), or (b) a deep theoretical insight that reshapes thinking in the field. This proposal offers neither. The cost improvement is predicted at 2–5× (likely 1.5–3× against modern baselines), and the theory is novel application rather than novel mathematics.

**The honest best-paper probability is 5–8%.** All three evaluators and the proposal's own revised assessment converge here. A 4/10 score is slightly generous for 5–8% probability, but a 2/10 is too harsh — there are plausible worlds where the annihilation numbers land perfectly and a reviewer champions it.

### Resolved score: **3/10**

I compromise. A 3/10 says: "This is a solid VLDB accept candidate (60–70% accept probability if well-executed) with a small but non-negligible best-paper shot (5–8%)." The Skeptic's 2 is too pessimistic about accept probability; the Auditor/Synthesizer's 4 is too optimistic about best-paper probability given that no single result is above B+ and the empirical validation is entirely speculative.

---

## Final Adjudicated Scores

| Axis | Auditor | Skeptic | Synthesizer | **Adjudicated** | Reasoning |
|------|---------|---------|-------------|-----------------|-----------|
| Value | 5 | 3 | 6 | **5** | Delta annihilation is genuine differentiator; marginal value unproven empirically |
| Difficulty | 6 | 4 | 6 | **5** | Integration difficulty is real but library leverage deflates; split |
| Best-Paper | 4 | 2 | 4 | **3** | Solid accept candidate, small best-paper shot; no result exceeds B+ |
| **Composite** | **15** | **9** | **16** | **13/30** | |

---

## Final Verdict: **CONTINUE (weak)** — Composite 13/30

### Reasoning

The proposal survives because of three load-bearing facts:

1. **Delta annihilation is a genuinely new operation** that no existing system (heuristic, LLM, or algebraic) performs. Proving `push_f(δ) = 0` is categorically different from lineage tracing. If it works in practice, it's a significant contribution. If it doesn't, we find out at the Week 4 kill gate.

2. **The two-sorted fallback provides a publishable floor.** Even if three-sorted coherence fails, Fragment F is only 40%, and the DBSP impossibility is trivial, a paper on two-sorted annihilation with honest scope is a genuine VLDB contribution. The expected value calculation: 60% chance of a solid paper × value of that paper > the opportunity cost of 4–6 weeks of effort.

3. **The Skeptic's ABANDON case assumes worst-case on every axis simultaneously.** Coherence fails AND annihilation rate is low AND Fragment F is tiny AND the impossibility is trivial. While each risk is 25–40%, they are partially independent. The probability of *all* failing is ~10–15%, not the 30–40% the Skeptic implies.

### Why not a stronger CONTINUE

- Zero implementation exists. The theory is entirely speculative until code runs.
- The Skeptic's core insight — that an 80% heuristic captures most practical value — is probably correct. The algebraic approach's advantage is in the remaining 20%, whose importance is unverified.
- Best-paper probability is low (5–8%). This is a "solid publication" project, not a "breakthrough" project.

### Mandatory kill gates

| Gate | Deadline | Condition | Action if failed |
|------|----------|-----------|-----------------|
| **KG-1** | Week 2 | Two-sorted coherence verified for SELECT, JOIN, GROUP BY, FILTER via property-based tests (≥100K cases each, 0 violations) | ABANDON algebra; pivot to heuristic-with-lineage paper |
| **KG-2** | Week 3 | SQL column-level lineage ≥90% recall on TPC-DS queries (ground truth from manual annotation of 20 representative queries) | ABANDON system; publish theory-only paper |
| **KG-3** | Week 4 | Delta annihilation prunes ≥15% of stages on TPC-DS pipeline corpus under realistic schema evolution traces | Demote to theory contribution; drop cost claims |
| **KG-4** | Week 6 | End-to-end repair correctness on ≥10 TPC-DS pipelines (repair output = full recomputation output) | ABANDON; results are unreliable |

### Recommended execution order

1. **Weeks 1–2**: Delta algebra engine (two-sorted) + property-based coherence tests → KG-1
2. **Weeks 2–3**: SQL semantic analyzer on sqlglot → KG-2
3. **Weeks 3–4**: Annihilation detection + TPC-DS pipeline corpus → KG-3
4. **Weeks 4–6**: Repair planner + end-to-end evaluation → KG-4
5. **Weeks 6–8**: Three-sorted extension (if time permits), paper writing

### Point of maximum leverage

If the team has limited bandwidth, **invest disproportionately in annihilation detection and SQL lineage quality.** These two subsystems determine whether the paper has a compelling story (annihilation) told with integrity (lineage accuracy). The interaction homomorphisms and commutation theorem are the mathematical skeleton; annihilation is the muscle that makes it move.

---

*Adjudication complete. The Skeptic's risk analysis is the most accurate assessment of what can go wrong. The Synthesizer's identification of delta annihilation as the load-bearing contribution is the most actionable strategic insight. The Auditor's balanced scoring is closest to ground truth. Final composite: 13/30, CONTINUE (weak) with kill gates.*
