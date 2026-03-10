# Theory Gate Report: Verification Stage

**Problem:** bounded-rational-usability-oracle  
**Area:** area-042-human-computer-interaction  
**Date:** 2026-03-04  
**Stage:** Verification (post-theory evaluation)  
**Method:** Claude Code Agent Teams — 3-expert adversarial panel with independent proposals, cross-critiques, synthesis, and independent verifier signoff

---

## Team Composition

| Role | Function |
|------|----------|
| **Independent Auditor** | Evidence-based scoring and challenge testing |
| **Fail-Fast Skeptic** | Aggressively reject under-supported claims |
| **Scavenging Synthesizer** | Salvage value and identify strongest framing |
| **Independent Verifier** | Process quality review and final signoff |

---

## Process Summary

### Phase 1: Independent Proposals (no cross-visibility)

| Axis | Auditor | Skeptic | Synthesizer |
|------|---------|---------|-------------|
| Value | 6 | 4 | 7 |
| Difficulty | 6 | 5 | 6 |
| Best-Paper | 4 | 3 | 6 |
| CPU/No-Humans | 8 | 8 | 9 |
| Feasibility | 5 | 4 | 6 |
| **Total** | **29** | **24** | **34** |
| **Verdict** | CONTINUE | **ABANDON** | CONTINUE |

Spread: 10 points. Genuine disagreement — the Skeptic recommended ABANDON.

### Phase 2: Adversarial Cross-Critiques (6 challenges exchanged)

**Skeptic → Synthesizer (challenges upheld):**
- Best-Paper 6→5: "Reframing potential" is not an accomplishment. Cannot name a zero-data best paper.
- Value 7→6: Layer 1 alone is trivially competitive with axe-core diff.

**Synthesizer → Skeptic (challenges upheld):**
- Value 4→5: The problem exists with zero existing solutions. A 4 implies barely worth solving.

**Auditor mediation findings:**
- Confirmed Skeptic's "motte-and-bailey" diagnosis: Layer 1 defended by Layer 2's features, Layer 2 defended by Layer 1's feasibility. Valid structural critique.
- Confirmed Synthesizer's "fragility is the diamond" argument: all evidence points to cognitive fragility as the most novel and publishable contribution.
- Identified task-specification bottleneck as under-examined across all evaluations (groupthink gap).

### Phase 3: Post-Critique Positions

| Axis | Auditor | Skeptic | Synthesizer | **Median** |
|------|---------|---------|-------------|------------|
| Value | 5 | 5 | 6 | **5** |
| Difficulty | 6 | 5 | 6 | **6** |
| Best-Paper | 4 | 3 | 5 | **4** |
| CPU/No-Humans | 8 | 8 | 9 | **8** |
| Feasibility | 5 | 4 | 6 | **5** |
| **Total** | **28** | **25** | **32** | **28** |
| **Verdict** | CONTINUE | ABANDON | CONTINUE | **CONTINUE** |

Spread narrowed from 10 to 7 points. 2-to-1 CONTINUE with meaningful convergence.

### Phase 4: Independent Verifier Signoff

**SIGNOFF: APPROVED.** Verifier confirmed:
- Process rigor: A- (genuine adversarial disagreement, not rubber-stamp)
- Groupthink residual: LOW (Skeptic maintained ABANDON through full process)
- Score consistency: PASS (all scores traceable to specific evidence)
- Verdict justified: YES (conditional CONTINUE with binding gate)

---

## Final Verified Scores

| Axis | Score | Rationale |
|------|-------|-----------|
| **Extreme Value** | **5/10** | Real unaddressed gap (no automated structural usability regression in CI/CD). But: trivial baseline may capture parameter-free cases, 50-70% coverage claim fabricated, CogTool adoption failure unaddressed, ordinal validity undemonstrated. |
| **Genuine Difficulty** | **6/10** | Paired-comparison proof and fragility computation are genuine. But: Layer 1 (guaranteed deliverable) is difficulty 4-5. Two of four theorems self-described as "easy." Probability-weighted expected difficulty ≈ 6. |
| **Best-Paper Potential** | **4/10** | Novel consistency-oracle framing. Fragility concept genuinely interesting. But: zero empirical results (disqualifying for CHI/UIST best paper), incomplete proofs, one conjectured theorem, three modest results rather than one stunning contribution. ~15-20% best-paper probability. |
| **CPU & No-Humans** | **8/10** | Strongest axis (unanimous). Principled CPU-native design: accessibility trees, embarrassingly parallel MC, CPU-native SMT, published psychophysical parameters. Minor gaps: state-space bound unsubstantiated, CI/CD timing for Layer 2. |
| **Feasibility** | **5/10** | Incremental architecture is the saving grace. But: zero implementation code, accessibility-tree parser is 8+ weeks alone, cross-browser normalization is unsolved at industry scale, compound risk P(≥1 failure) ≈ 84%. |
| **TOTAL** | **28/50** | |

---

## Prior Evaluation Context

Six evaluations total (3 prior + 3 verification team):

| Evaluator | Total | Verdict |
|-----------|-------|---------|
| Prior Skeptic | 33/50 | CONTINUE |
| Prior Mathematician | 31/50 | CONTINUE |
| Prior Community Expert | 28/50 | CONTINUE |
| Verification Auditor | 28/50 | CONTINUE |
| Verification Skeptic | 25/50 | **ABANDON** |
| Verification Synthesizer | 32/50 | CONTINUE |
| **Mean** | **29.5/50** | 5× CONTINUE, 1× ABANDON |

The verification team scored 2-5 points lower than prior evaluations on average, reflecting more aggressive scrutiny. The first explicit ABANDON recommendation emerged during verification.

---

## Fatal Flaw Analysis

| Flaw | Severity | Mitigable? | Status |
|------|----------|-----------|--------|
| Retrospective validation data may not exist | CRITICAL | Yes — pivot to modern UI pairs | **Hard gate: Week 2** |
| 50-70% coverage claim fabricated | SEVERE | Yes — drop number or measure | Must fix immediately |
| Evaluation circularity | SEVERE | Partially — fragility metric is self-referential | Reframe around fragility |
| Trivial baseline captures same parameter-free cases | HIGH | Must test — implement as benchmark | **Hard gate: Week 4** |
| Theorem 4 might be false | MODERATE | Contained — defer to future work | Already recommended |
| k-transition degradation makes Theorem 1 vacuous for large changes | MODERATE | Must characterize k empirically | Condition for Layer 2 |
| Task-specification adoption barrier | MODERATE | Under-examined across all evaluations | New finding from verification |

**No single flaw is definitively fatal.** The closest: retrospective validation data availability (30% failure risk) threatens the entire consistency-oracle claim.

---

## Key Insights from Verification

### 1. The Motte-and-Bailey Problem (Skeptic, validated by Auditor)
Layer 1 is defended by invoking Layer 2's features (paired comparison, fragility). Layer 2 is defended by invoking Layer 1's feasibility (incremental architecture, standalone value). This circular defense is structurally valid — incremental architecture IS genuine risk management — but it masks the fact that Layer 1 alone has thin research novelty.

### 2. Cognitive Fragility Is the Diamond (All three agree)
The most novel, most publishable, most mathematically deep contribution. Currently buried as a Layer 2 add-on. Should be elevated to co-lead with paired comparison. The "Chaos Monkey for usability" framing (from abandoned Approach C) dramatically improves the elevator pitch.

### 3. The Task-Specification Bottleneck (Auditor, new finding)
Every CI/CD tool requiring per-task manual specification has failed to achieve adoption. The proposal's "automatic inference for standard flows" is hand-waved in one clause with no algorithm. This is the largest under-examined adoption barrier.

### 4. Self-Score Inflation
The proposal self-scores Value 9, Difficulty 7, Best-Paper 8, Feasibility 8. Verified scores: Value 5, Difficulty 6, Best-Paper 4, Feasibility 5. Gap of 2-4 points on three axes signals miscalibration.

---

## Binding Conditions for Continuation

| # | Condition | Deadline | Fail → |
|---|-----------|----------|--------|
| 1 | **Validate retrospective data:** Name 5 specific UI pairs with confirmed accessibility-tree extractability AND published human orderings | Week 2 | ABANDON consistency-oracle claim |
| 2 | **Proof-of-life gate:** Parse one real accessibility tree into task-path MDP (≥10 states), compute additive cost + fragility F(M) with non-trivial cliff or rigorous null explanation | Week 3 | ABANDON |
| 3 | **Trivial baseline:** Implement 50-line axe-core diff benchmark; Layer 1 must demonstrate Δτ ≥ 0.08 improvement in ordinal agreement | Week 4 | Rethink cost model |
| 4 | **Remove fabricated numbers:** Drop "50-70% coverage" and "70-85% coverage" claims immediately | Immediate | Required for honest self-assessment |
| 5 | **Scope to Layers 1-2 only:** Defer Layer 3 (bisimulation, cost algebra, Theorem 4) to future work | Immediate | Required |
| 6 | **Lead with fragility:** Reframe paper around cognitive fragility as co-lead contribution | Before paper writing | Required for best-paper path |

---

## Proposal Rankings

### proposal_00: The Cognitive Regression Prover

**Score: 28/50**  
**Theory Score: 5.6/10** (= 28/50 normalized)  
**Verdict: CONTINUE (conditional)**

The proposal addresses a genuine unmet need with a principled CPU-only design and one genuinely novel contribution (cognitive fragility). The incremental architecture provides meaningful downside protection. However: zero implementation exists, zero empirical validation exists, self-scores are significantly inflated, the flagship theorem degrades for non-trivial changes, and the guaranteed deliverable (Layer 1) is thin. Best-paper probability is ~15-20%, strong-UIST-contribution probability ~45-55%.

**CONTINUE is justified because:**
1. The problem is real and unaddressed (unanimous)
2. The fragility concept is genuinely novel (unanimous)
3. No single fatal flaw is unrecoverable
4. The incremental architecture bounds downside
5. Expected value of 6-month investment is positive given ~50% probability of UIST-quality result

**CONTINUE is narrow because:**
1. The Skeptic's ABANDON was sustained through full adversarial process
2. Zero evidence that the pipeline produces correct results on any real UI
3. The motte-and-bailey between Layer 1 feasibility and Layer 2 value is structural
4. Binding Week 2 and Week 3 gates must pass or verdict converts to ABANDON

---

```json
{
  "rankings": [
    {
      "proposal_id": "proposal_00",
      "score": 28,
      "verdict": "CONTINUE",
      "reason": "Addresses genuine unmet need (automated structural usability regression in CI/CD) with principled CPU-only design and one genuinely novel contribution (cognitive fragility as robustness metric). Verified 28/50 across 3-expert adversarial panel (2-to-1 CONTINUE, 1 ABANDON sustained). Zero empirical validation and thin Layer 1 novelty are serious weaknesses. Conditional on Week 2 validation-data gate and Week 3 proof-of-life gate passing.",
      "scavenge_from": []
    }
  ]
}
```

---

## What Would Most Improve This Proposal

**Produce one end-to-end worked example on a real UI pair.** Take Material UI v4→v5, extract accessibility trees, run the parameter-free analysis, compute the fragility metric, show the full output. One concrete demonstration would simultaneously validate the accessibility-tree pipeline, ground the coverage claims, test the fragility concept, and reveal whether the system produces useful or vacuous results. The absence of this — despite being achievable in days — is the single most concerning signal about the proposal's maturity.
