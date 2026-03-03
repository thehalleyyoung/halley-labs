# Theory Evaluation — Skeptic Review

## Proposal: Causal-Plasticity Atlas (proposal_00)
**Stage:** Post-theory verification
**Review Team:** Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer
**Adjudication:** Cross-critique mediation + Independent Verifier signoff

---

## SCORES

| Dimension | Auditor | Skeptic | Synthesizer | Cross-Critique Consensus | Final |
|---|---|---|---|---|---|
| Extreme Value | 5/10 | 3/10 | 7/10 | 5/10 | **5/10** |
| Genuine Software Difficulty | 7/10 | 6/10 | 4/10 | 5/10 | **5/10** |
| Best-Paper Potential | 4/10 | 2/10 | 5/10 | 4/10 | **4/10** |
| Laptop-CPU Feasibility | 6/10 | 3/10 | 7/10 | 5/10 | **5/10** |
| Feasibility | 6/10 | 4/10 | 8/10 | 6/10 | **6/10** |
| **Overall** | **5.3** | **3.4** | **~6** | **5.0** | **5.0/10** |

---

## THREE PILLARS ASSESSMENT

### Pillar 1: Extreme and Obvious Value — FAIL (5/10)

The 4D plasticity descriptor taxonomy (invariant / parametrically plastic / structurally plastic / emergent) and the MCCM formalization for variable-set mismatch are genuinely novel contributions. No prior multi-context causal framework handles variable appearance/disappearance as a first-class phenomenon.

**However**, the value is academic, not extreme:
- The target community is small (~50–200 researchers working on multi-context observational causal discovery with 15–50 variables).
- The proposal's own risk analysis rates its foundational risks as HIGH/HIGH (TR1: DAG errors dominate, AR2: unmeasured confounders are the norm). A tool whose guarantees erode in precisely the regime where users need it most does not deliver *extreme* value.
- 80% of practical value is achievable with per-context GES + pairwise JSD comparison (~100 lines of Python).
- No domain scientist has requested this capability; no pilot results on real data exist.

### Pillar 2: Genuinely Difficult Software Artifact — FAIL (5/10)

The full CPA as proposed includes genuinely hard components (NP-hard DAG alignment, stability-selection certificates). But the cross-critique consensus correctly identifies that:
- QD search is overengineered for a 4D descriptor space (exhaustive computation takes 30–60 seconds).
- Robustness certificates are computationally infeasible at stated scale (17+ hours for p=100).
- What remains after removing these overbuilt components (CPA-Lite) is a moderate-difficulty research codebase (~500–2000 LOC), not a genuinely hard software artifact.

### Pillar 3: Real Best-Paper Potential — FAIL (4/10)

The theory has genuine depth (14 definitions, 8 theorems, 5 algorithms), but:
- Theorems are standard compositions of known results: T1 (DAG metric) follows from √JSD metric properties + GED theory; T2 (classification correctness) is McDiarmid + union bound; T3 (certificate soundness) is bootstrap consistency; T6 (atlas completeness) is coupon-collector.
- The evaluation is entirely synthetic/semi-synthetic with no real-data results.
- The paper tries to do too many things (DAG alignment + descriptors + QD search + tipping points + certificates), diluting the core contribution.
- Publishable as a regular paper at UAI/AISTATS, but not competitive for best paper at any top venue.

**Three-Pillars Result: 0/3 pass. Clear fail.**

---

## FATAL FLAWS ANALYSIS

Six significant flaws were identified. None is individually unfixable, but their collective weight is damning:

### Flaw 1: Self-Defeating Risk Assessment
The proposal rates TR1 (DAG estimation errors dominate everything downstream) and AR2 (unmeasured confounders are the norm in observational data) as HIGH severity / HIGH likelihood. The 6-assumption conjunction (Markov + Faithfulness + ICM + Causal Sufficiency + Context Regularity + Bounded Parents) holds with ~6% probability on realistic observational data. This is not an edge case — it is the default condition.

**Cross-critique adjudication:** Valid concern but proves too much; all causal discovery methods share similar assumption stacks. CPA is no worse than ICP on this dimension.

### Flaw 2: Vacuous Certificates
Robustness certificates guarantee correctness *conditional on correct DAGs*, but DAGs from observational data are never correct (the proposal's own TR1). Stability selection measures robustness to *sampling variability*, not *model misspecification*. The 17-hour computational investment produces statements conditional on assumptions the authors admit don't hold.

**Cross-critique adjudication:** Valid. Demote certificates to diagnostics. Keep Theorem T3 as a theoretical result, not a practical deliverable.

### Flaw 3: Self-Falsified Runtime Claim (FC7)
FC7 claims p=100, K=5 in under 120 minutes on a single CPU. The proposal's own bottleneck analysis shows certificate generation requires ~8+ hours sequential for p=100. This is a 4× discrepancy within the same document.

**Cross-critique adjudication:** Valid. FC7 must be revised to match honest bottleneck estimates.

### Flaw 4: No Real-Data Validation
All 11 falsifiable claims use synthetic generators that satisfy CPA's assumptions by construction. The semi-synthetic Sachs benchmark (11 variables) is trivially small. GTEx and Penn World Table are mentioned speculatively with no pilot results. The evaluation design guarantees success.

**Cross-critique adjudication:** Valid but fixable. Real data is a mandatory addition before publication.

### Flaw 5: Missing CD-NOD Baseline
CD-NOD (Zhang/Huang et al., 2017/2020) already identifies changing causal modules across heterogeneous data without requiring ICM or causal sufficiency. It is not included as a baseline despite being the most relevant prior work.

**Cross-critique adjudication:** Valid but not fatal. CPA's 4D descriptor taxonomy genuinely extends beyond CD-NOD's binary change detection. But CD-NOD must be the primary baseline.

### Flaw 6: Unnecessary QD Search
The plasticity descriptor for every variable can be computed exhaustively in O(p × K²) time (~30–60 seconds). MAP-Elites adds CVT tessellation, curiosity signals, and mutation operators to "explore" a 4D space that can be fully enumerated. FC5's coverage improvement is tautological (curiosity-driven search fills empty cells by definition).

**Cross-critique adjudication:** Valid. QD search should be cut from implementation. The descriptor space is too small to need stochastic exploration.

---

## WHAT SURVIVES (Scavenging Synthesizer Analysis)

### Components Worth Preserving

| Component | Standalone Value | Rationale |
|---|---|---|
| MCCM formalization (D1–D3) | 7/10 | First framework handling variable-set mismatch in multi-context causal inference |
| Emergence relation (D12) | 7/10 | Testable information-theoretic criterion for causal emergence; genuinely novel |
| Mechanism distance via √JSD (D5) | 5/10 | Proper metric, necessary building block |
| 4D plasticity descriptors (D7–D8) | **8/10** | Core intellectual contribution; clean decomposition into actionable taxonomy |
| Tipping-point detection (ALG4) | 4/10 | Simple PELT wrapper; useful but trivial |

### Components to Cut

| Component | Rationale |
|---|---|
| QD search (ALG3) | 4D space is exhaustively computable; MAP-Elites adds complexity without value |
| Full robustness certificates (ALG5) | Computationally infeasible and epistemically fragile |
| CVT tessellation | Visualization technique masquerading as algorithm |
| DAG alignment (ALG1) | Only needed for unknown variable correspondences (niche case) |

### CPA-Lite: The Viable Core

A focused paper on MCCM + 4D plasticity descriptors would be publishable at UAI/AISTATS:
- ~500 lines of Python using `causal-learn` + `scipy` + `ruptures`
- Theorems T2 (classification correctness), T4 (sample complexity), T8 (perturbation bounds)
- Evaluation on generators 1–3 + Sachs network + one real-data case study
- CD-NOD as primary baseline
- Honest discussion of DAG-estimation-error limitation (preserving the risk analysis)

**Publication title suggestion:** *"Beyond Invariance Testing: A Continuous Plasticity Spectrum for Multi-Context Causal Mechanisms"*

---

## KEY DISAGREEMENTS AND RESOLUTIONS

### "Is CD-NOD adequate?" (Skeptic vs. Synthesizer)
**Resolution:** CD-NOD detects *which* modules change (binary). CPA classifies *how* they change (4D continuous). This is a genuine extension, not incremental feature engineering. But CD-NOD must be included as baseline.

### "Are certificates worthless?" (Skeptic vs. Auditor)
**Resolution:** Certificates are formally correct but practically undeliverable. The mathematical contribution (T3) has value; the claim of providing actionable per-mechanism guarantees from observational data does not. Demote to diagnostics.

### "Is the difficulty genuine?" (Auditor 7/10 vs. Synthesizer 4/10)
**Resolution:** They scored different scopes. Full CPA is 7/10 difficulty. CPA-Lite (what should be built) is 4–5/10. The NP-hard DAG alignment is the only genuinely hard novel component; whether it's needed depends on whether variable-set mismatch is common.

---

## VERDICT: **ABANDON**

### Rationale

The Causal-Plasticity Atlas fails all three pillars of the best-paper standard:
1. **Value is moderate, not extreme** (5/10). The contribution is incremental: a useful taxonomy atop existing methods.
2. **Difficulty is moderate, not genuinely hard** (5/10). After removing the overbuilt components (QD search, certificates), what remains is a standard research codebase.
3. **Best-paper potential is absent** (4/10). Theorems are standard compositions. No real-data results. Too many contributions dilute focus.

The operational constraints (laptop CPU, no humans) are met only by CPA-Lite, which is an even smaller project further from the extreme-value bar.

### What Should Happen to This Idea

The MCCM formalization and 4D plasticity descriptors are genuine contributions worth pursuing through normal academic channels (UAI/AISTATS submission). They should not consume implementation resources reserved for three-pillars-clearing projects.

### Conditions for Reconsideration

1. **Demonstrate extreme value:** A real-data case study where CPA reveals causal structure that no existing method can detect.
2. **Find genuine difficulty:** A computational challenge in the CPA pipeline that is both necessary and hard (not QD search, not infeasible certificates).
3. **Articulate best-paper narrative:** A theoretical result with surprising implications, or an empirical finding that changes how the field thinks about context-dependent causation.
