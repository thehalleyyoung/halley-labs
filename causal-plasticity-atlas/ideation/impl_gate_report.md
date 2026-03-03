# Implementation Gate Report — Causal-Plasticity Atlas

**Date:** 2026-03-02  
**Methodology:** Claude Code Agent Teams — 3 independent experts + adversarial cross-critique + lead verification signoff  
**Evaluator roles:** Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer

---

## Final Scores

| Axis | Score | Rationale |
|------|-------|-----------|
| **Extreme and Obvious Value** | **5/10** | Real unsolved problem (multi-context mechanism plasticity), but audience is narrow (~500–2000 causal inference researchers). Not a problem a large community desperately needs solved. Pipeline cannot produce results end-to-end. |
| **Genuine Difficulty as Software** | **6/10** | ~12K lines of genuine algorithmic code (CADA alignment, √JSD mechanism distance, PELT changepoint, MAP-Elites QD, bootstrap certificates). 90% from-scratch with numpy/scipy only. But: components don't compose — orchestrator has confirmed API mismatches. Novel *composition* of known methods, not novel algorithms per se. |
| **Best-Paper Potential** | **3/10** | Zero experimental results. Broken end-to-end pipeline. JSD inconsistency (plasticity.py computes symmetric KL, not JSD). QD evaluator is simulation stub. No real-data validation. No baseline comparisons (PC, GES, FCI, ICP, CD-NOD absent). Would not survive peer review at any top venue. |
| **Total** | **14/30 (4.7/10)** | |

---

## Verdict: ABANDON

**Confidence: HIGH** — All three independent experts recommended ABANDON. Cross-critique consensus confirmed at 4.7/10. Lead verification confirmed all critical flaws.

---

## Expert Score Summary

| Expert | Value | Difficulty | Best-Paper | Composite | Verdict |
|--------|-------|-----------|------------|-----------|---------|
| Independent Auditor | 7 | 8 | 6 | 7.0 | CONTINUE (conditional) |
| Fail-Fast Skeptic | 3 | 5 | 1 | 3.0 | ABANDON |
| Scavenging Synthesizer | 6 | 7 | 5 | 6.0 | SALVAGE & ITERATE |
| **Cross-Critique Consensus** | **5** | **6** | **3** | **4.7** | **ABANDON** |
| **Lead Verification** | **5** | **6** | **3** | **4.7** | **ABANDON** |

---

## Verified Fatal Flaws

### 1. Pipeline Cannot Run End-to-End (CONFIRMED)
The orchestrator has API mismatches with its own components:
- `orchestrator.py` calls `qd_engine.search()` but `QDSearchEngine` only has `.run()` (wrong method name AND wrong return type)
- `orchestrator.py` passes `cost_matrix=, context_ids=, method=, penalty=` to `detector.detect()`, but `PELTDetector.detect()` expects `adjacencies, datasets, context_order, target_idx`
- All errors silently caught by broad `try/except` handlers

### 2. JSD Computation Inconsistency (CONFIRMED)
- `cpa/descriptors/plasticity.py:267-286` computes **symmetric KL divergence** `0.5*(KL(P‖Q) + KL(Q‖P))` and labels it "JSD approximation"
- `cpa/core/mechanism_distance.py:346-400` computes **correct JSD** via moment-matched Gaussian mixture
- These produce different distance orderings for the same distributions
- The plasticity module (the paper's central contribution) uses the wrong one

### 3. QD Search Evaluator Is a Stub (CONFIRMED)
- `_default_evaluator` in `qd_search.py:76-139` generates synthetic classifications using seeded RNG
- Docstring explicitly states: "When the full pipeline is not yet available, this produces synthetic but structurally valid results"
- The QD exploration — the primary claimed novelty beyond existing multi-context methods — never connects to real causal analysis

### 4. Test Suite Has Systematic Gaps (CONFIRMED)
- **59 except→pass blocks** in test files that swallow failures
- **4 `assert X or True`** patterns that cannot fail
- **13 except→pass blocks** in production code
- No test verifies JSD numerical accuracy against known values
- No test exercises orchestrator→component interface contracts
- 1,452 unit tests pass but miss all integration-level bugs

### 5. Zero External Validation
- No benchmark datasets (Sachs, ALARM, Asia, Dream4)
- No baseline comparisons (CD-NOD, ICP, LPCMCI, FCI)
- No real-data demonstrations
- All evaluation on synthetic generators the authors control

---

## Disagreement Resolution

### Auditor (7.0) vs Skeptic (3.0)
The Auditor was too generous — missed the 59 except→pass patterns and scored difficulty 8/10 despite broken integration. The Skeptic correctly identified all fatal flaws but overstated severity by calling CADA "zero novelty" (it's a genuine novel composition) and Best-Paper 1/10 (workshop potential exists). **Resolution: Cross-critique median of 4.7 is evidence-aligned.**

### "Is the JSD bug fixable?"
Yes, in ~10 lines. But fixability is not the evaluation criterion — the artifact is scored as-submitted. The bug's existence after 0 polish rounds indicates the system was never tested against ground truth.

### "Is 90% from-scratch impressive?"
The Synthesizer correctly noted that most code is from-scratch numpy/scipy without external causal libraries. However, the algorithms being implemented (Hungarian matching, PELT, MAP-Elites, bootstrap) are all well-known. The novelty is in their composition for multi-context causal analysis, not in any individual algorithm.

---

## Salvageable Components

| Component | LOC | Quality | Ecosystem Gap? |
|-----------|-----|---------|----------------|
| `core/mechanism_distance.py` — √JSD metric | 2,373 | High | Yes — no Python library provides mechanism-level JSD |
| `alignment/cada.py` + scoring + hungarian | 3,541 | Moderate | Yes — no cross-context DAG alignment tool exists |
| `detection/changepoint.py` — PELT | 1,167 | High | No — `ruptures` library exists |
| `core/scm.py` + `core/mccm.py` — Data model | 2,436 | High | Partial — pgmpy/causal-learn exist but lack multi-context |
| `stats/` — Statistical primitives | 1,943 | High | No — scipy exists |
| `certificates/robustness.py` | 1,296 | Moderate | Partial — novel application |

A focused ~8–12K line extraction (mechanism distance + CADA + plasticity descriptors + PELT) could form a standalone "CPA-lite" library worth pursuing as a separate project.

---

## Process Notes

1. Three independent background agents ran full codebase analysis (~6 min each)
2. Cross-critique agent resolved 6 inter-expert disagreements with evidence citations
3. Lead verified 5 critical claims: JSD bug ✅, QD stub ✅, API mismatches ✅, test suite (1,452 pass ✅, 59 except-pass ✅, 4 assert-or-True ✅)
4. Final verdict is unanimous across Skeptic, Cross-critique, and Lead verification
5. Auditor's CONTINUE was overruled: Auditor missed except-pass patterns and overscored difficulty

---

## Rankings

```json
{
  "rankings": [
    {
      "proposal_id": "proposal_00",
      "score": 4.7,
      "verdict": "ABANDON",
      "reason": "Pipeline cannot run end-to-end (orchestrator API mismatches with QD search and tipping-point detector). Core JSD computation in plasticity.py is mathematically wrong (symmetric KL, not JSD). QD evaluator is a simulation stub. 59 except-pass blocks in tests mask failures. Zero real-data validation. Zero baseline comparisons. 0 polish rounds completed. Theory gate scored 4.0 ABANDON. Three pillars: 0/3 met.",
      "scavenge_from": []
    }
  ],
  "best_proposal": "proposal_00"
}
```

---

*Report generated by three-expert verification team with adversarial cross-critique and lead verification signoff. All scores derived from direct source code examination and independent test execution (1,452 passed in 180s).*
