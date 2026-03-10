# Theory Evaluation: Community Expert (Data Management & Databases)

**Proposal:** Algebraic Repair Calculus — Three-Sorted Delta Algebra for Provably Correct Pipeline Repair  
**Evaluation method:** Claude Code Agent Teams (3 experts + adversarial synthesis + independent verification)  
**Team roles:** Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer  
**Date:** 2026-03-04

---

## SCORES

| Axis | Score | Confidence |
|---|---|---|
| 1. Extreme Value | **5/10** | HIGH |
| 2. Genuine Software Difficulty | **6/10** | HIGH |
| 3. Best-Paper Potential | **4/10** | MEDIUM |
| 4. Laptop-CPU Feasibility & No-Humans | **9/10** | HIGH |
| 5. Feasibility | **5/10** | MEDIUM |

**VERDICT: CONTINUE**

---

## Axis 1: Extreme Value — 5/10

The pipeline maintenance problem is unambiguously real. Schema evolution is the #1 cause of pipeline breakage, and data engineers spend 40–60% of their time firefighting. No existing system unifies schema evolution, data maintenance, and quality monitoring with formal correctness guarantees.

**However, the marginal value of algebraic formalism over existing practice is narrower than claimed:**

- **dbt** already does lineage-aware selective recomputation. **sqlmesh** handles incremental models with schema evolution awareness. **DBSP/Differential Dataflow** handle data-level incrementality elegantly. Practitioners compose these tools and get 80% of the way there (per the proposal's own Math Assessor).
- The **2–5× cost reduction** claim over dbt is completely unsupported — zero benchmarks exist. This is a target, not a measurement.
- **Compound perturbation frequency is unverified** — the entire three-sorted value proposition rests on simultaneous schema+data+quality changes being common enough to matter. The proposal concedes: "We concede this is unverified."
- **"Provably correct"** is a researcher's value proposition, not a practitioner's. The proposal itself acknowledges this: "the engineer cares about the green checkmark, not the algebra." But if the engineer only needs a green checkmark, a well-tested heuristic system that's right 95% of the time is indistinguishable — and vastly simpler.
- **LLM erosion** will handle 60–70% of simple perturbations within 2 years, narrowing the addressable problem space to hard compound cases.

**What saves this from a 4:** Delta annihilation — detecting when push_f(δ) = 0 and pruning unnecessary stages — is a genuine, concrete differentiator that no existing tool provides. This single capability is immediately understandable and potentially valuable even without the full algebraic machinery.

---

## Axis 2: Genuine Software Difficulty — 6/10

The difficulty is concentrated in the right places for a database paper:

- **Delta algebra engine** (~9K LoC, 9/10 difficulty): 48 coherence conditions across 8 SQL operators × 3 sorts × 2 interaction maps. A single error invalidates the commutation theorem. Genuinely hard research-grade code with no existing template.
- **Non-monotone DP planner** (~5.5K LoC, 8/10): Delta annihilation breaks standard optimal substructure. Novel algorithm required.
- **SQL column-level lineage** (~5.5K LoC, 6/10): Foundation-critical and error-prone through correlated subqueries, lateral joins, window functions, CTEs.

**What deflates this from the self-assessed 8:**

- Massive library leverage: sqlglot (~70% of analyzer), networkx (~60% of graph), scipy/HiGHS (~30% of planner), DuckDB (~80% of executor). Novel code is ~22K out of ~50K LoC.
- The Python idiom matcher is pattern matching (5/10 difficulty), not research. The type system is standard refinement types. The executor is DuckDB checkpoint/rollback.
- ~15K LoC of genuinely novel, research-grade code embedded in ~25K LoC of infrastructure. This is a solid systems paper, not an exceptionally difficult artifact.

**Team consensus:** The Skeptic's characterization of "hard algebra wrapped around medium engineering" is partially right but undersells the verification challenge. 48 coherence conditions aren't busywork — each one could harbor subtle bugs that invalidate the entire correctness story.

---

## Axis 3: Best-Paper Potential — 4/10

The proposal's own honest novelty ratings tell the story:

| Contribution | Self-Assessed Novelty |
|---|---|
| Interaction homomorphisms | 6/10 — "new application of known algebraic structures" |
| Commutation theorem | 5/10 — "structural induction with new inductive hypothesis" |
| DBSP impossibility | 4–7/10 — "we don't know if it's trivial" |
| Complexity results | 4/10 — "textbook reduction" |
| Delta annihilation | 6/10 — "genuine algorithmic contribution" |

This is a "collection of B+ results" (the proposal's own characterization). No single contribution reaches the A+ threshold needed for best paper. The proposal honestly estimates 8% unconditional best-paper probability, and the self-assessment of "not in the lineage of Spark, Naiad, or DBSP" is accurate.

**Key reviewer attack vectors:**
1. "Interaction homomorphisms are indexed monoid actions — Section 3.2 of any universal algebra textbook."
2. "Fragment F covers 60–75% of stages — your guarantee is for the easy part."
3. "Show me one production pipeline where you find a repair dbt misses."
4. "80% of the capability without algebra — why do I need the algebra?"

**The DBSP impossibility** has a 35% chance of being trivially true ("DBSP has fixed types, schema changes change types, QED"). If trivial, the paper loses its sharpest positioning weapon.

**What could elevate this:** If the paper were reframed around delta annihilation as the hero result, with the algebra as the mechanism that enables it and the commutation theorem as the guarantee, the narrative becomes stronger: "This is the first system that can tell you 'this pipeline stage is provably unaffected by that schema change' — and be right." That's a sentence that makes a reviewer nod.

**Realistic target:** Solid accept at VLDB with 50–100 citations over 5 years. Not best paper — but a genuine contribution.

---

## Axis 4: Laptop-CPU Feasibility & No-Humans — 9/10

This is the proposal's quiet strength and a near-ideal match for the CPU-only constraint:

- **All computation is CPU-native:** AST parsing, symbolic term rewriting, DP/LP optimization, streaming statistics, DuckDB queries
- **No ML, no training, no GPUs, no human annotation, no user studies**
- **DuckDB-only backend** eliminates distributed systems concerns
- **TPC-DS at SF=10** fits comfortably in 16GB RAM
- **Evaluation is fully automated** with property-based testing (Hypothesis) and benchmark generation
- **ILP solver timeout** mitigated by LP relaxation fallback

Only concern: full evaluation suite is 6–8 hours, and 10M Hypothesis tests are CPU-intensive but parallelizable. These are timing concerns, not feasibility blockers.

---

## Axis 5: Feasibility — 5/10

**The compound risk profile is severe:**

| Risk | Probability | Impact |
|---|---|---|
| Hexagonal coherence fails for ≥3 operators | 30-40% | FATAL (retreat to two-sorted) |
| Scope creep from algebra+system balance | 40% | HIGH |
| SQL lineage incorrectness | 25% | HIGH (invalidates "provably correct") |
| DBSP impossibility trivially true | 35% | MEDIUM (lose positioning) |
| Fragment F covers <50% of stages | 25% | HIGH (guarantee covers minority) |
| Implementation complexity underestimated | 70% | MEDIUM |

P(no major risk materializes) ≈ 15–20%. P(at least one major risk) ≈ 80–85%.

**What saves this from a 4:** The tiered scope with two-sorted escape hatch. Tier 1 (two-sorted, SQL-only, acyclic DAGs, ~38K LoC) is achievable at ~65% probability and produces a publishable result even if three-sorted coherence fails and DBSP impossibility is trivial. The front-loaded risk testing (coherence in weeks 1–4, DBSP proof sketch by week 3) means fatal risks are discovered early, not late.

**Self-assessed 55% publishable probability** is reasonable. The theory document's 7/10 feasibility is inflated; the final approach's 6/10 is closer, and the evidence supports 5/10 after adversarial critique.

---

## Fatal Flaws

### Flaw 1: 2–5× Cost Claim is Completely Unsubstantiated ⚠️
The practitioner-facing value proposition rests on "2–5× cost reduction over dbt." Zero benchmarks exist. If delta annihilation savings are <1.3× in practice, the engineering value proposition evaporates and the paper becomes pure theory — harder to place at VLDB.

### Flaw 2: Compound Perturbation Frequency is Unvalidated ⚠️
The three-sorted algebra's differentiator is handling compound perturbations (simultaneous schema+data+quality changes). Prediction P5 targets 10–15% of episodes — but no data validates this. If <8% (the proposal's own falsification threshold), the three-sorted motivation collapses and the paper shrinks to "IVM extended with schema deltas."

### Flaw 3: 30–40% FATAL Risk on Core Theorem ⚠️
48 hexagonal coherence conditions across 8 operators. If coherence fails for JOIN or GROUP_BY, the commutation theorem doesn't hold for most real pipelines. The two-sorted escape hatch converts "project dies" to "paper is less interesting" — but the gap between the full vision and the fallback is large.

**None of these flaws are individually project-killing** given the tiered scope and escape hatches. But collectively they constrain the upside significantly.

---

## Team Process Summary

### Phase 1: Independent Proposals
| | Auditor | Skeptic | Synthesizer |
|---|---|---|---|
| Value | 6 | 4 | 7 |
| Difficulty | 6 | 5 | 7 |
| Best-Paper | 4 | 3 | 6 |
| CPU | 9 | 8 | 9 |
| Feasibility | 5 | 4 | 7 |
| Verdict | CONTINUE | CONTINUE (reluctant) | CONTINUE (optimistic) |

### Phase 2: Adversarial Critique (Key Challenge Outcomes)
- **Skeptic → Synthesizer:** Skeptic wins on Value (7 is ceiling, not expectation). Synthesizer wins on escape hatch viability.
- **Auditor → Skeptic:** Auditor wins on Difficulty (5 is too low; coherence proofs are genuinely hard). Split on Value (problem is real but marginal benefit uncertain).
- **Synthesizer → Auditor:** Synthesizer wins on Best-Paper arithmetic (4 maps below base rate). Auditor wins on Feasibility framing (Tier 1 publishable ≠ venue-appropriate).

### Phase 3: Synthesis
Post-critique consensus: Value 5.5, Difficulty 6, Best-Paper 5, CPU 9, Feasibility 5.5

### Phase 4: Independent Verification
Final adjudicated scores: Value 5, Difficulty 6, Best-Paper 4, CPU 9, Feasibility 5. Verifier sided slightly with Skeptic/Auditor on Value and Best-Paper, finding the Synthesizer's optimism insufficiently grounded in the proposal's own risk assessment.

---

## Key Recommendations

1. **Reframe the paper around delta annihilation, not interaction homomorphisms.** All three team members agree annihilation is the hero result. Lead with: "This is the first system that can prove a pipeline stage is unaffected by an upstream change."

2. **Validate the 2–5× cost claim immediately.** Before further theory work, prototype delta annihilation on TPC-DS pipelines. If savings are <1.3×, the cost story dies and the paper must pivot to correctness-only framing.

3. **Start with Tier 1, two-sorted.** Do not attempt the quality sort (Δ_Q) until schema+data coherence is verified. The two-sorted escape hatch is the project's insurance policy — use it.

4. **Resolve DBSP impossibility early.** If the proof sketch (due week 3) is shallow, demote to a design-motivation remark. Don't waste a theorem number on a tautology.

5. **Benchmark against dbt --select with column-level lineage.** The Skeptic's "80% achievable without algebra" is the most damaging argument. If true, the paper is marginal. If false, the paper is strong. This is empirically testable and must be tested before the theory is finalized.

6. **Use the self-assessed 8% best-paper probability, not the theory document's 25–35%.** The 8% figure from the final approach incorporates red-team criticism and is more honest.

---

## VERDICT: CONTINUE

**Rationale:** Despite significant risks (compound failure probability ~80–85%, self-assessed 55% publishable probability), three factors justify continuation:

1. **The problem is real and unsolved.** No existing system provides formal correctness guarantees for incremental repair under schema evolution. Even a partial solution advances the state of the art in data management.

2. **The tiered scope provides a credible safety net.** Tier 1 (two-sorted algebra, SQL-only, DP for acyclic, commutation theorem for Fragment F) is achievable at ~65% probability and produces a publishable VLDB submission — not the full vision, but a genuine contribution.

3. **Delta annihilation is a concrete, demonstrable win.** Statically proving that perturbation deltas annihilate at specific pipeline stages — and therefore skipping entire subgraphs — is a clean advance over dbt-style lineage-aware recomputation. This single result can anchor a good paper regardless of whether the full three-sorted theory lands.

**This is not a paradigm shift. It is a well-scoped niche contribution with a clean algebraic insight and a measurable engineering payoff.** The realistic ceiling is a solid VLDB accept with 50–100 citations over 5 years. That's a good outcome worth pursuing.

---

*Evaluation complete. Team: Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer, Independent Verifier. All teammates shut down. Verification signoff obtained.*
