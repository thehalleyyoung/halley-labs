# Theory Gate Report: MutSpec-Hybrid (mutation-contract-synth)

| Field | Value |
|---|---|
| **Slug** | `mutation-contract-synth` |
| **Area** | 009 — Programming Languages and Formal Methods |
| **Date** | 2026-03-08 |
| **Stage** | Verification (post-theory, pre-implementation) |
| **Evaluator** | Best-Paper Committee Chair (Team Lead) |
| **Panel** | Independent Auditor · Fail-Fast Skeptic · Scavenging Synthesizer |
| **Process** | Independent proposals → Adversarial cross-critiques → Lead synthesis |
| **Prior Evaluations** | Skeptic 3.55 ABANDON · Mathematician 5.05 COND. CONTINUE · Community Expert 5.15 COND. CONTINUE |
| **Final Composite** | **3.29/10** |
| **Verdict** | **ABANDON with salvage** |

---

## Executive Summary

Three independent experts evaluated MutSpec-Hybrid across five dimensions after its theory phase. Each produced independent proposals, then adversarially critiqued each other's work. Post-critique scores converged strongly toward ABANDON:

| Expert | Initial Score | Post-Critique Score | Verdict |
|--------|--------------|-------------------|---------|
| Independent Auditor | 3.70/10 | 3.63/10 | ABANDON |
| Fail-Fast Skeptic | 2.68/10 | 2.43/10 | ABANDON |
| Scavenging Synthesizer | 4.58/10 | 3.80/10 | CONDITIONAL CONTINUE (narrowly) |

**Panel vote: 2-1 ABANDON.** The Synthesizer dissents with CONDITIONAL CONTINUE but concedes most of the Skeptic's factual claims and dropped 0.78 points during cross-critique. The Auditor independently shifted from prior panel optimism (5.2/10) to ABANDON (3.63/10) after re-examining evidence.

**The project has produced zero proofs, zero code, zero experiments, and zero bugs after completing its theory phase.** The 63KB of theory artifacts consists entirely of theorem *statements* and architectural plans — not a single completed proof. The crown jewel theorem (A3: ε-completeness) is unstarted at 65% achievability. The practical deliverable is achievable by a ~500-line Z3 script atop PIT. The panel unanimously agrees the mutation-specification duality insight is genuinely novel; the panel also unanimously agrees the project cannot execute on it.

---

## Panel Process

### Phase 1: Independent Proposals (Parallel)
Three experts independently scored the proposal across five weighted pillars, producing assessments ranging from 2.68/10 (Skeptic) to 4.58/10 (Synthesizer).

### Phase 2: Adversarial Cross-Critiques (Parallel)
Each expert critiqued the other two. Key moves:
- **Auditor** identified three areas where the Skeptic was too harsh (max(a,b) conflation with T1, consultant test overcounting, treating all execution failure as fungible) and four areas where the Synthesizer was too generous (insight ≠ output, conflating independent salvage bets, process/concept distinction being non-actionable, inflated difficulty).
- **Skeptic** challenged the Auditor's Value=4 as smuggling in hypothetical theorems (deliverables=0 means value=0), and dismantled the Synthesizer's MutGap-Lite estimate as 2x optimistic on both LoC and timeline.
- **Synthesizer** conceded five major Skeptic claims (Gap Theorem circularity, max(a,b) expressiveness gap, ~80% consultant replicability, "zero bugs found" kill sentence, compounding probability direction). Defended against the Skeptic's 1.5/10 best-paper via venue calibration and maintained that structured output has real engineering delta over the 500-line script.

### Phase 3: Synthesis (This Document)
Lead resolved disagreements using evidence weight, not vote counting.

---

## Score Reconciliation

### Pillar 1: Extreme Value — 3.5/10

| Expert | Score | Post-Critique |
|--------|-------|--------------|
| Auditor | 4.0 | 3.8 |
| Skeptic | 2.0 | 2.0 |
| Synthesizer | 4.5 | 3.5 |
| **Lead** | | **3.5** |

**Resolution:** The mutation-specification duality is genuinely novel (all experts confirm; prior art audit survives scrutiny). The marginal-cost-on-PIT argument is structurally sound. However:

1. **The consultant test is devastating.** The project self-acknowledges 70-80% replicability. Post-SyGuS elimination, the panel converges on ~80%. The marginal non-replicable value depends entirely on Theorem A3, which does not exist.
2. **The 500-line Z3 script achieves the practical deliverable.** PIT → Z3 equivalence check → model extraction produces ranked bug reports with distinguishing inputs. The Gap Theorem formalizes but does not enable this pipeline.
3. **Zero demand-side evidence.** No user interviews, no feature requests, no adoption signals. The Skeptic's challenge — "name a user who asked for this" — remains unanswered after three phases.

The Synthesizer's concession of consultant replicability to ~78-82% and the Auditor's acknowledgment that novelty ≠ value cap this pillar. The Skeptic's 2/10 is slightly harsh (the insight has real pedagogical/survey value), but the Auditor's 4/10 gives too much credit for potential over deliverables.

### Pillar 2: Genuine Difficulty — 3.5/10

| Expert | Score | Post-Critique |
|--------|-------|--------------|
| Auditor | 3.5 | 3.5 |
| Skeptic | 3.0 | 3.0 |
| Synthesizer | 5.0 | 4.0 |
| **Lead** | | **3.5** |

**Resolution:** SyGuS elimination removed the only genuinely novel open problem. What remains:

- **Lattice-walk:** Standard lattice fixpoint theory (Cousot & Cousot 1977) applied to a new domain. The Mathematician confirms: "Parts (a)-(c) are straightforward lattice theory."
- **Theorem A3:** The only genuinely hard component, but at 65% achievability it is a *risk*, not a difficulty *credit*. Unstarted. May be unprovable.
- **WP differencing:** The Synthesizer correctly identifies batch WP with incremental Z3 sharing as non-trivial systems work. The Skeptic undersells this at 3/10. But it is engineering novelty, not algorithmic novelty.

The Auditor's "difficulty-value paradox" is decisive: making the project feasible (eliminating SyGuS) made it uninteresting. The Synthesizer's post-critique concession to 4.0 (from 5.0) acknowledges this. The honest assessment: one hard unproved theorem + competent integration engineering.

### Pillar 3: Best-Paper Potential — 2.0/10

| Expert | Score | Post-Critique |
|--------|-------|--------------|
| Auditor | 2.5 | 2.5 |
| Skeptic | 1.5 | 1.5 |
| Synthesizer | 3.5 | 2.5 |
| **Lead** | | **2.0** |

**Resolution:** All three legs of the stool (completeness theorem + lattice-walk algorithm + real bugs) are missing:

- **Theorem A3:** Unproved, 65% achievable, covers loop-free QF-LIA only (~10-20% of real Java methods). The depth check challenge — "Show one PLDI best paper with main theorem restricted to loop-free code" — was never met.
- **Lattice-walk:** Unimplemented. Post-SyGuS pivot narrowed the gap with SpecFuzzer.
- **Bug finding:** Zero bugs found. The Skeptic's kill sentence — *"The authors present a bug-finding tool that has found zero bugs"* — is accurate and fatal for any reviewer.

The Synthesizer's venue-calibration argument has merit: P(FormaliSE acceptance) ≈ 50-60%, which is not consistent with 1.5/10. However, FormaliSE is a workshop, and workshop acceptance does not constitute "best-paper potential." I weight the Skeptic's calibration to top venues more heavily. The realistic ceiling is a solid ASE tool paper or FormaliSE note — neither is best-paper territory.

**P(best paper at any top venue) ≈ 1-3%.** P(any top-venue publication) ≈ 15-20%.

### Pillar 4: Laptop-CPU Feasibility — 6.0/10

| Expert | Score | Post-Critique |
|--------|-------|--------------|
| Auditor | 6.5 | 6.5 |
| Skeptic | 5.0 | 5.0 |
| Synthesizer | 7.0 | 7.0 |
| **Lead** | | **6.0** |

**Resolution:** The pipeline is genuinely CPU-only (PIT/JVM, Z3/C++, lattice-walk/polynomial). No GPUs, cloud APIs, or human annotation. This is the project's strongest pillar. The Skeptic's philosophical objection — the constraint isn't genuinely binding because the fragment is where everything is easy — has merit but overweights it. Z3 at scale (O(|D|² · SMT(n)) with hundreds of dominators) is non-trivial engineering even for QF-LIA.

The QF-LIA/QF-BV soundness gap (Java 32-bit integers vs. unbounded LIA) is a real credibility risk but not a feasibility blocker.

### Pillar 5: Feasibility — 2.5/10

| Expert | Score | Post-Critique |
|--------|-------|--------------|
| Auditor | 2.5 | 2.5 |
| Skeptic | 2.0 | 2.0 |
| Synthesizer | 3.5 | 3.0 |
| **Lead** | | **2.5** |

**Resolution:** This is the most damning pillar.

1. **Zero deliverables after three phases.** The project completed ideation, depth check, and theory. It produced 150KB+ of planning documents and zero results (proofs, code, experiments).
2. **Binding gates from depth check ignored.** The prior CONDITIONAL CONTINUE was explicitly conditioned on empirical validation. The theory phase produced formalism instead.
3. **Compounding probability.** Using the most charitable estimates: P(A3 proves) × P(implementation succeeds) × P(empirical validation) = 0.65 × 0.60 × 0.70 ≈ 27% for full delivery. The Skeptic's 22% accounts for optimism bias in self-assessed achievability. Neither is above threshold.
4. **The Synthesizer's process-vs-concept distinction collapses under cross-critique.** The Auditor's decisive argument: "The CONDITIONAL CONTINUE verdict requires predicting future execution success. The strongest predictor of future execution is past execution." A project that has never produced output gets a lower prior on future output, regardless of whether the failure was "conceptual" or "process."

The Synthesizer's concession to 3.0 (from 3.5) and the Auditor's Bayesian update (from 60-70% prior to 35-45% on execution ability) anchor this score.

---

## Composite Score

| Pillar | Score | Weight | Weighted |
|--------|-------|--------|----------|
| Extreme Value | 3.5 | 25% | 0.875 |
| Genuine Difficulty | 3.5 | 20% | 0.700 |
| Best-Paper Potential | 2.0 | 25% | 0.500 |
| Laptop-CPU Feasibility | 6.0 | 15% | 0.900 |
| Feasibility | 2.5 | 15% | 0.375 |
| **Composite** | | **100%** | **3.35** |

**Rounded composite: 3.35/10** — well below the 5.0 continuation threshold.

Cross-check against prior evaluations:

| Evaluation | Composite | Verdict |
|-----------|-----------|---------|
| Prior Skeptic (theory_eval_skeptic.md) | 3.55 | ABANDON |
| Prior Mathematician (theory_eval_mathematician.md) | 5.05 | CONDITIONAL CONTINUE |
| Prior Community Expert (theory_eval_community_expert.md) | 5.15 | CONDITIONAL CONTINUE |
| **This panel (post-cross-critique)** | **3.35** | **ABANDON** |

The downward revision from the Mathematician (5.05) and Community Expert (5.15) reflects: (1) those panels scored before the cross-critique phase exposed the full weight of the evidence, (2) this panel's Auditor independently revised downward from 5.2 after fresh examination, and (3) the Synthesizer — the most optimistic voice — dropped 0.78 points to 3.80 after conceding five major Skeptic claims.

---

## Fatal Flaws

### CRITICAL (independently sufficient for ABANDON)

**F1: Zero Deliverables After Theory Stage**
- Evidence: `theory_bytes: 0` (State.json measurement), `code_loc: 0`, `impl_loc: 0`. The 63KB in theory/ consists of theorem *statements*, not proofs. Zero lines contain proof content (QED, ∎, "by induction," "assume for contradiction").
- Impact: Demonstrates inability to execute. Three phases of excellent planning, zero bytes of results.
- All three experts agree this is CRITICAL.

**F2: Practical Deliverable Achievable by Simpler Means**
- Evidence: PIT → Z3 `∃x. f(x) ≠ m(x)` → model extraction achieves the flagship deliverable (ranked bug reports with distinguishing inputs) in ~500 lines.
- Impact: The value proposition collapses. The remaining delta depends on Theorem A3, which does not exist.
- Auditor and Skeptic agree CRITICAL. Synthesizer concedes ~80% replicability but argues structured output has delta (engineering delta, not research delta).

**F3: Depth Check Binding Conditions Ignored**
- Evidence: Prior CONDITIONAL CONTINUE required empirical validation (Gate 0: MutGap-Lite by week 4). The theory phase produced formalism instead.
- Impact: Process failure predicts future process failure. A second CONDITIONAL CONTINUE with "harder gates" has no credibility given the first gates were ignored.
- All three experts agree this is CRITICAL.

### SERIOUS

**F4: Crown Jewel Theorem A3 Unstarted at 65% Achievability**
- 35% probability the project's intellectual foundation is false. Zero proof progress.

**F5: Cross-Site Expressiveness Gap**
- `max(a,b)` counterexample: lattice-walk cannot express relational postconditions. Tier 1 fails for methods with non-trivial input-output relationships.
- Synthesizer concedes this is "devastating for Tier 1."

**F6: Gap Theorem Circularity**
- All three experts agree: trivially true by definitions. The Synthesizer concedes: "rename to Gap Characterization."

**F7: QF-LIA Loop-Free Restriction Covers ~10-20% of Real Java**
- The formal guarantee applies to a toy fragment. The depth check challenge ("show one PLDI best paper restricted to loop-free code") remains unanswered.

### MODERATE

- F8: Equivalent mutant FP rate unvalidated (target ≤10%, realistic 10-20%)
- F9: Math inflation (18 "theorems," ~3 genuine after honest filtering)
- F10: QF-LIA/QF-BV soundness gap undermines "formally grounded" claim
- F11: SpecFuzzer differentiation narrowed post-SyGuS-pivot

---

## Key Disagreements Resolved

### D1: Is "Zero Deliverables" a Process Failure or a Prediction?

**The Synthesizer argues:** "Zero deliverables is a process failure (phase misdesign), not a concept failure. The insight is real; the execution plan was wrong."

**The Auditor and Skeptic argue:** "The strongest predictor of future execution is past execution. A project that produced planning documents when proofs were demanded will produce planning documents when code is demanded."

**Resolution: The Skeptic and Auditor are correct.** The Synthesizer's distinction between process and concept failure is intellectually valid but operationally useless. A CONDITIONAL CONTINUE requires predicting future success. The Bayesian update after three phases of non-delivery should reduce P(execution) to ~35-45% (Auditor's estimate). The depth check already set binding gates that were ignored; setting new binding gates has no credibility unless accompanied by a fundamentally different execution mechanism.

### D2: Does the Salvage Plan Justify CONDITIONAL CONTINUE?

**The Synthesizer argues:** MutGap-Lite (3K LoC, 4 weeks) + FormaliSE note have P(accept) ≥ 50% each.

**The Skeptic argues:** MutGap-Lite is 5.5-7K LoC (not 3K), 8-10 weeks (not 4), because the PIT↔symbolic bridge alone is ~3.5K LoC per the project's own estimates.

**Resolution: The Skeptic's critique is substantially correct.** The Synthesizer's LoC and timeline estimates are ~2x optimistic. However, the *existence* of a salvage path with non-trivial P(accept) is real — even at the Skeptic's inflated estimates, MutGap-Lite is buildable in 8-10 weeks and has P(accept at ASE tool track) ≈ 40-50%. This argues for ABANDON-with-salvage (Skeptic's recommendation), not CONDITIONAL CONTINUE (Synthesizer's recommendation). The distinction matters: "continue the project" implies the theoretical apparatus and full vision are still active. "Salvage" implies killing the theory/lattice-walk vision and building a simple tool.

### D3: What Is the 500-Line Script's Actual Replication Percentage?

**Skeptic:** 80-85%. **Auditor:** 75-80%. **Synthesizer (post-concession):** 78-82%.

**Resolution: ~80%.** The Synthesizer's argument about structured output, provenance, and minimality adds ~5% real delta over the flat script. The Auditor's argument about batch WP differencing adds ~5% performance delta. Total non-replicable: ~20%. But this 20% includes A3 (which may not exist) as its largest component.

---

## Publication Probability Estimates (Panel Consensus)

| Outcome | Skeptic | Auditor | Synthesizer | **Consensus** |
|---------|---------|---------|-------------|--------------|
| PLDI | 3-4% | 5-8% | 8-12% | **5-8%** |
| OOPSLA | 8% | 12-18% | 12-18% | **10-15%** |
| ASE/ISSTA | 15% | 25-35% | 40-50% | **30-40%** |
| FormaliSE | 25% | 40-50% | 45-55% | **40-50%** |
| No publication | 55-60% | 45-55% | 25-35% | **35-45%** |
| Best paper (any) | <1% | 1-2% | <3% | **1-2%** |

---

## Verdict: ABANDON

**Composite: 3.35/10 — well below 5.0 continuation threshold.**

**Panel vote: 2-1 ABANDON (Auditor + Skeptic ABANDON; Synthesizer dissents CONDITIONAL CONTINUE at 3.80).**

The project has five fatal or serious flaws, zero completed deliverables after three phases, and a primary value proposition achievable by dramatically simpler means. The theoretical contribution — the only potential differentiator — is unstarted, has a 35% chance of being false, and covers only a toy fragment of real programs. Binding gates from the prior evaluation were ignored. The evidence pattern is clear and predictive.

### What Unanimous Agreement Exists

All three experts agree on:
1. The mutation-specification duality insight is genuinely novel
2. Zero completed proofs exist in any artifact
3. The Gap Theorem is definitional (rename to "Gap Characterization")
4. Zero empirical validation exists
5. Depth check binding conditions were not addressed
6. The scope restriction (loop-free QF-LIA) covers 5-20% of real functions
7. Laptop-CPU feasibility is a genuine strength (5-7/10)
8. SpecFuzzer overlap remains unresolved
9. Consultant replicability is ~80%
10. The SyGuS elimination reduced both difficulty and novelty

### What the Synthesizer Gets Right (Incorporated into Salvage)

The Synthesizer's salvage analysis is the highest-value output of this verification. Three items have independent merit:

1. **MutGap tool demo (ASE/MSR tool track):** PIT + Z3 non-equivalence + model extraction. ~5-7K LoC (Skeptic's corrected estimate). 8-10 weeks. P(accept) ≈ 40-50%. This is the honest core of what MutSpec attempted.
2. **FormaliSE theory note:** Mutation-specification duality formalization (A1+A2+A4). 6-8 weeks. P(accept) ≈ 40-50%. Establishes priority on the insight.
3. **Prior art audit:** 19KB, thorough, immediately reusable as Related Work. Zero additional effort.

### Recommended Salvage Path

**Option A (8-10 weeks):** Build MutGap tool demo. Honest framing ("PIT + Z3 for gap analysis"), no theoretical claims, concrete bug finding on Defects4J. ASE/MSR tool track.

**Option B (6-8 weeks):** FormaliSE/ICSE-NIER note formalizing the duality. Establishes priority. Workshop-level venue.

**Option C (0 weeks):** Archive everything. The prior art audit and math_spec.md are available for future reference.

**Recommendation: Option A.** MutGap is the honest core. Build it, find bugs (or don't), and know.

---

## Appendix: Evidence Inventory

| Claim | Status | Source |
|---|---|---|
| theory_bytes = 0 (no completed proofs) | **CONFIRMED** | State.json; grep for proof content yields 0 hits |
| code_loc = 0 | **CONFIRMED** | State.json; implementation/ contains only scoping.md |
| 63KB theory planning docs exist | **CONFIRMED** | theory/math_spec.md (31KB) + approach.json (13KB) + prior_art_audit.md (19KB) |
| 17+ theorem statements, 0 proofs | **CONFIRMED** | math_spec.md: 17+ "Theorem" declarations, 0 completed proofs |
| A3 at 65% achievability, unstarted | **CONFIRMED** | math_spec.md lines 92-115 |
| Gap Theorem is definitional | **CONFIRMED** | All 3 experts + all 3 prior evaluations agree |
| SyGuS eliminated | **CONFIRMED** | final_approach.md: lattice-walk replaces SyGuS |
| max(a,b) counterexample for lattice-walk | **CONFIRMED** | approach_debate.md; theory_eval_skeptic.md |
| ~80% consultant-replicable | **CONFIRMED** | Project self-reports 70-80%; panel converges ~80% |
| Depth check gates not met | **CONFIRMED** | Gate 0 (MutGap-Lite by week 4) never attempted |
| SpecFuzzer comparison never run | **CONFIRMED** | No comparison artifacts exist |

---

## JSON Verdict

```json
{
  "rankings": [
    {
      "proposal_id": "proposal_00",
      "score": 3.35,
      "verdict": "ABANDON",
      "reason": "Zero deliverables after three phases (0 proofs, 0 code, 0 experiments). Practical deliverable achievable by ~500-line Z3 script (~80% consultant-replicable). Crown jewel theorem A3 unstarted at 65% achievability. Gap Theorem definitionally circular. Depth check binding gates ignored. Cross-site expressiveness gap limits Tier 1 to non-relational properties. Panel vote: 2-1 ABANDON (Auditor 3.63, Skeptic 2.43, Synthesizer 3.80). Composite 3.35/10, well below 5.0 threshold. Salvage: MutGap tool demo (PIT+Z3, ~5-7K LoC, ASE/MSR tool track, P(accept)≈40-50%).",
      "scavenge_from": []
    }
  ]
}
```

---

*Filed by Team Lead (Best-Paper Committee Chair). Panel: Independent Auditor (3.63/10) · Fail-Fast Skeptic (2.43/10) · Scavenging Synthesizer (3.80/10). Lead Composite: 3.35/10. Verdict: ABANDON with salvage.*
