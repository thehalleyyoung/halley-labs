# Theory Gate Report — Causal-Plasticity Atlas

**Date:** 2026-03-02  
**Stage:** Verification (post-theory)  
**Method:** Three-expert adversarial team (Independent Auditor + Fail-Fast Skeptic + Scavenging Synthesizer) with cross-critique synthesis and independent verification signoff.

---

## Executive Summary

The Causal-Plasticity Atlas (proposal_00) was evaluated by a six-evaluator pipeline: three original reviewers (Skeptic, Mathematician, Community Expert) followed by a three-person verification team with adversarial cross-critique. The proposal receives a final composite score of **4.0/10** and verdict **ABANDON**.

The core insight — classifying causal mechanism change as a continuous geometric object rather than a binary label — is genuine but insufficiently developed. The full proposal (5 algorithms, 14 definitions, 8 theorems) is overengineered for the problem it solves, with no real-data validation, a missing primary baseline (CD-NOD), self-contradicted runtime claims, and certificates that are vacuous in the regime where practitioners need them most.

---

## Scoring Trail

### Original Evaluations

| Evaluator | Value | Difficulty | Best-Paper | Feasibility | Composite | Verdict |
|-----------|-------|------------|------------|-------------|-----------|---------|
| Skeptic | 5/10 | 5/10 | 4/10 | 5/10 | 5.0/10 | ABANDON |
| Mathematician | 5/10 | 6/10 | 4/10 | 7/10 | 5.4/10 | CONTINUE* |
| Community Expert | 6/10 | 7/10 | 5/10 | 6/10 | 6.0/10 | CONTINUE* |

*Conditional on scope reductions, real-data validation, CD-NOD baseline, and dropping QD search.

### Verification Team — Independent Proposals

| Expert | Value | Difficulty | Best-Paper | Composite | Verdict |
|--------|-------|------------|------------|-----------|---------|
| Independent Auditor | 4/10 | 5/10 | 3/10 | 4.0/10 | ABANDON |
| Fail-Fast Skeptic | 3/10 | 3.5/10 | 4/10 | 3.5/10 | ABANDON |
| Scavenging Synthesizer | 6.5/10* | 5.5/10 | 5/10 | 4.8/10 | SALVAGE |

*Synthesizer's 6.5 was for an unsubmitted salvaged MVP; full-proposal value was 5.0.

### Post-Debate Consensus (after adversarial cross-critique)

| Dimension | Score |
|-----------|-------|
| Extreme Value | **4.2/10** |
| Genuine Difficulty | **4.2/10** |
| Best-Paper Potential | **3.7/10** |
| **Composite** | **4.0/10** |

### Verified Final Score: **4.0/10 — ABANDON**

---

## Three Pillars Assessment

### Pillar 1: Extreme and Obvious Value — FAIL (4.2/10)

The 4D plasticity taxonomy (invariant / parametrically plastic / structurally plastic / emergent) and the robustness certificate framework are genuine contributions with no direct competitor. However:

- **Zero demonstrated value.** All 12 falsifiable claims use synthetic data satisfying CPA's assumptions by construction. No real-data pilot exists.
- **80% of practical value achievable in ~100 LOC.** Per-context GES + pairwise √JSD comparison captures most of the actionable insight without the 5-algorithm pipeline.
- **Self-defeating risk assessment.** The proposal rates DAG estimation errors (TR1) and unmeasured confounders (AR2) as HIGH severity / HIGH likelihood. The 6-assumption conjunction holds with ~6–16% probability on realistic observational data.
- **Niche audience.** ~50–200 researchers working on multi-context observational causal discovery with 15–50 variables.

### Pillar 2: Genuine Difficulty — FAIL (4.2/10)

After removing components all evaluators unanimously recommend cutting (QD search, atlas completeness, full-scale certificates), what remains is:

- Per-context GES (~0 LOC, library call)
- √JSD mechanism distance (~50 LOC)
- 4D plasticity descriptor (~100 LOC, four independent 1D statistics)
- Classification by thresholds (~50 LOC)
- Bootstrap/stability-selection certificates (~300 LOC)
- Pipeline integration (~200 LOC)

Total: **~800–1,500 LOC**, buildable by a competent research engineer in 2–3 weeks. The perturbation-model design for certificates (choosing which structural DAG changes to test) is the only component requiring genuine research; it accounts for ~30% of effort.

### Pillar 3: Best-Paper Potential — FAIL (3.7/10)

- **No surprising theorems.** T1 follows from known √JSD metric properties. T2 is McDiarmid + union bound. T3 is bootstrap consistency. T6 is coupon-collector. All are correct compositions of known results.
- **Zero empirical results.** No comparison demonstrates CPA outperforming any baseline.
- **Too sprawling.** 14 definitions, 8 theorems, 5 algorithms in a 186KB draft (50+ pages) — the antithesis of best-paper elegance.
- **Missing primary baseline.** CD-NOD (the most relevant competitor) is cited but never benchmarked.

**Three-Pillars Result: 0/3 pass.**

---

## Fatal Flaws (5 fully triggered, 2 partially)

| # | Flaw | Status | Evidence |
|---|------|--------|----------|
| 1 | No CD-NOD baseline | **Fully triggered** | Most relevant competitor absent from all evaluation plans |
| 2 | FC7 self-contradicted | **Fully triggered** | Claims 120 min for p=100; own bottleneck analysis shows 8+ hours |
| 3 | No real-data validation | **Fully triggered** | 12/12 falsifiable claims on synthetic data |
| 4 | QD search unjustified | **Fully triggered** | 4D descriptor space exhaustively computable in seconds |
| 5 | 4D descriptor trivially computable | **Fully triggered** | Four 1D statistics, no interaction terms, ~100 LOC |
| 6 | T8 perturbation bounds vacuous for ψ_P | **Partially triggered** | Vacuous at s≥3 for ψ_P; tight and useful for ψ_S |
| 7 | Certificates vacuous | **Partially triggered** | ψ_P certificates vacuous; ψ_S certificates informative |

---

## Key Disagreements and Resolutions

### "CONTINUE or ABANDON?"

The original evaluators split 2-1 for CONTINUE. The verification team split 2-1 for ABANDON after determining that the CONTINUE verdicts were actually verdicts for a *different, smaller project* (CPA-Lite) — not the proposal as designed. The conditions attached to CONTINUE (Oracle-DAG gap ≤25% F1, F1≥0.75 at N=500) are predicted to fail by the proposal's own analysis (T4 requires N_min ≈15,000–29,000; T8's ψ_P bound exceeds [0,1] at realistic SHD).

### "Is the sweet spot vanishingly narrow?"

The Skeptic's original argument that CPA operates in a vanishingly narrow sweet spot was shown to overreach logically (it would invalidate all downstream causal analysis). The refined argument prevails: CPA's usable zone is strictly contained within the DAG-learning method's usable zone, making it narrower by construction.

### "Are certificates worth building?"

The certificate framework (D14, ALG5, T3) has zero competitors. However, certificates condition on correct DAGs (which are never fully correct from observational data). Resolution: ψ_S certificates are genuinely useful; ψ_P/ψ_E certificates are currently vacuous. A ψ_S-only certificate paper is the highest-value spin-off.

---

## Salvageable Components

| Component | Value | Recommendation |
|-----------|-------|----------------|
| 4D plasticity descriptors (D7–D8) | 8/10 | Core intellectual contribution; cheap to implement |
| Robustness certificates (D14, ALG5, T3) | 8.5/10 (ψ_S only) | Zero competitors; demote to "calibrated diagnostics" |
| √JSD mechanism distance (D5, T1) | 7.5/10 | Load-bearing foundation; proper metric |
| Structural stability bounds (T8) | 7/10 | Informative for ψ_S; vacuous for ψ_P |
| MCCM formalization (D1–D3) | 7/10 | First framework for variable-set mismatch |
| QD search (ALG3) | DROP | Exhaustive computation in seconds |
| Tipping-point detection (ALG4, T5) | DEFER | PELT wrapper; separate follow-up |
| Atlas completeness (T6) | DROP | Vacuous coupon-collector bound |

---

## Ranking

```json
{
  "rankings": [
    {
      "proposal_id": "proposal_00",
      "score": 4.0,
      "verdict": "ABANDON",
      "reason": "Fails all three pillars: value is moderate not extreme (4.2/10), difficulty is moderate after cutting overengineered components (4.2/10), best-paper potential absent with no surprising theorems and zero empirical results (3.7/10). Self-contradicted runtime claims, missing CD-NOD baseline, vacuous certificates for 2/4 descriptor dimensions, and no real-data validation. Original CONTINUE verdicts were conditional on scope reductions transforming this into a different project.",
      "scavenge_from": []
    }
  ]
}
```

---

## Process Notes

**Team composition:** Independent Auditor (evidence-based scoring), Fail-Fast Skeptic (aggressive rejection of unsupported claims), Scavenging Synthesizer (salvage maximization).

**Workflow:** Independent proposals → adversarial cross-critique (3 paired exchanges) → synthesis → independent verification signoff.

**Key concessions extracted during cross-critique:**
1. Synthesizer conceded 6.5/10 value was for unsubmitted salvage (revised to 5.0 for full proposal)
2. Auditor conceded difficulty inflated by including components recommended for cutting (revised 5.0→4.0)
3. Skeptic conceded "narrow sweet spot" argument overreached; ψ_S certificates are robust (revised value 3.0→3.5)
4. All three agreed: 4D descriptor framing oversold with only 1–2 reliable dimensions

**Verification signoff:** APPROVED. No procedural irregularities, no systematic biases, kill conditions corrected from 7/7 to 5/7 fully triggered.

---

*Report generated by verification team with independent signoff. All scores derived from evidence in proposal artifacts (approach.json 166KB, paper.tex 186KB) and six evaluator assessments.*
