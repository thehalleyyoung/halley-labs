# Independent Verification Report: nlp-metamorphic-localizer

**Verifier:** Independent Verifier (automated)
**Date:** 2026-03-08
**Documents reviewed:** `final_approach.md`, `problem_statement.md`, `approach_debate.md`, `depth_check.md`, proposals A/B/C

---

## 1. Consistency Check — PASS

The final approach is consistent with the problem statement's core claims. Key alignments:

- **Problem statement** defines the gap as "no existing tool localizes faults to a specific pipeline stage." Final approach preserves this framing exactly.
- **Problem statement** targets spaCy + HuggingFace + Stanza. Final approach targets spaCy + HuggingFace (Stanza demoted to stretch goal). This is a **tightening**, not a contradiction — acceptable.
- **Problem statement** claims M4 and M5 as the two genuinely new math contributions. Final approach preserves both and adds N1, N4(a,b,d), N3-simplified from the debate. No conflicts.

**Two controlled divergences (not contradictions):**

| Problem Statement Claim | Final Approach Change | Verdict |
|---|---|---|
| 107K LoC, 18K Rust grammar compiler | ~40.5K LoC, lightweight feature-checker replaces grammar compiler | Deliberate scope reduction; consistent with problem statement's own go/no-go milestone concept |
| ≥25 bugs target (Tier 1) | ≥10 bugs target | Honest recalibration after debate; problem statement's own fallback already acknowledged <5 scenario |

Neither divergence contradicts the problem statement — both are refinements that the problem statement's own risk analysis anticipated.

---

## 2. Skeptic Survival — PASS (4/5 fully resolved, 1/5 adequately mitigated)

### F-A1: Grammar compiler tar pit → **RESOLVED**
The 18K-LoC grammar compiler is completely eliminated. Replaced by a ~2–3K LoC feature-unification checker in Rust that answers only "is this candidate grammatical w.r.t. the 15 transformations' constraints?" — not a generative grammar. Scope is bounded: ~80 feature constraints across ~15 clause types. Corpus-based generation (Penn Treebank + 300 hand-crafted seeds) replaces grammar-compiled generation. This is the single most important design decision and it is correct.

### F-B1: Uncomputable C(T) → **RESOLVED**
N2 (information-theoretic localization bounds depending on C(T)) is demoted to a conditional two-track strategy: attempt proof in 4 weeks, include if successful, relegate to future work if not. The tool is built regardless as a Thompson-sampling heuristic. C(T) no longer gates any core claim. The final approach explicitly states "~40% proof failure risk" — honest.

### F-B2: Locally-linear assumption (N3) → **RESOLVED**
N3(b,c) (observational identifiability requiring locally-linear condition) is cut entirely. Only N3-simplified is retained: DCE/IE formalization with interventional sufficiency, which is "trivially true" by construction (the engine always does interventional replay). The locally-linear condition is never invoked. Clean resolution.

### F-C1: spaCy-as-proxy killing M5 → **RESOLVED**
The final approach explicitly rejects spaCy-as-proxy for the convergence guarantee. The Rust feature-unification checker provides a deterministic oracle, preserving M5's convergence proof. spaCy is used only for corpus parsing (generation) and as a naturalness proxy (Tier 3 evaluation), never as the grammaticality oracle in the shrinking loop.

### F-C2: Accidental corpus coverage → **ADEQUATELY MITIGATED** (not fully resolved)
The concern was that corpus-based generation might accidentally fail to cover rare constructions. The final approach adds 300 hand-crafted seed sentences targeting rare constructions + explicit transformation-coverage annotations on seeds + the N1 discriminability matrix as a pre-test diagnostic. This is a reasonable mitigation but not a guarantee — corpus coverage gaps could still exist for constructions not anticipated by the seed author. The N1 rank check provides a formal diagnostic, which is the right answer. **Residual risk: low.**

---

## 3. Math Portfolio Sanity — PASS

### M4 (Causal-Differential Fault Localization)
- Statement is clean: per-stage differentials, argmax localization, interventional refinement for introduction-vs-amplification.
- Complexity O(N · n · C_pipeline) is correctly stated and realistic.
- Framed as "validated heuristic" — honest about circularity concern.
- **Achievability: HIGH.** This is an algorithm, not a deep theorem. Implementable.

### N1 (Stage Discriminability Matrix)
- Parts (a) and (b) are standard linear algebra (rank = n iff columns span ℝⁿ; rank < n iff equivalence classes exist).
- "One-paragraph proofs" is accurate. These are column-space arguments.
- Calibration from ~100 samples is feasible (estimating means, not distributions — correctly distinguished from N2's KL divergence problem).
- **Achievability: HIGH.**

### N4(a): NP-hardness of global minimality
- Reduction from Minimum Grammar-Consistent String is standard.
- **Achievability: HIGH.** Straightforward reduction.

### N4(b): 1-minimality convergence
- O(|T|² · |R|) bound for GCHDD is a careful but routine extension of delta debugging.
- The conditional O(|T| · log|T| · |R|) improvement (requires monotonicity lemma) is honestly flagged as uncertain.
- **Achievability: HIGH for base bound, MODERATE for improvement.**

### N4(d): Expected shrinking ratio
- E[|x'|] ≤ |x|/b + O(α · log|x|) for bounded ambiguity α, branching b.
- The 40→8-13 word estimate for b≈3, α≤5 is plausible.
- **Achievability: MODERATE-HIGH.** The bound involves distributional assumptions that need empirical validation.

### N3-simplified (DCE/IE formalization)
- Stated as a formalization, not a theorem. Interventional sufficiency is trivially true by construction.
- **Achievability: HIGH.** This is a definition + one observation.

**Concern:** No result is overclaimed. The two-track strategy for N2 is appropriately hedged. All included results are either standard (N1, N4a) or achievable extensions (M4, N4b, N4d, N3-simplified).

---

## 4. Feasibility Check — PASS

### LoC Estimate
- **~40.5K total (24K Python + 11.5K Rust)** is realistic for a well-scoped tool.
- Compare to problem statement's 107K: the 2.6× reduction is entirely attributable to eliminating the grammar compiler (18K), reducing the fault localizer (12K→6K), and shrinking infrastructure. Each reduction is justified by the debate.
- The per-component estimates are credible: 15 transformations × ~200 lines = 3K (stated as 5K with MR checker — reasonable). Adapters for 2 frameworks at ~4K each = 8K. Rust shrinker at 7K. Feature-checker at 3K.
- **Minimum viable paper at ~18K Python** is an honest fallback.

### Architecture
- Python-first with Rust acceleration via PyO3 is a proven pattern (pydantic, tokenizers, polars).
- The Rust surface is confined to two well-defined components (shrinker + feature-checker) — no Rust sprawl.
- Pure-Python fallback for environments where Rust doesn't build — practical.
- pip-installable + Docker is the right delivery model.

### Score Honesty
- **Value 8/10:** Appropriate. Real gap, real audience, but contracting market. The 2-point docking for LLM shift is honest.
- **Difficulty 7/10:** Appropriate. Multi-domain synthesis, two novel algorithms, but known techniques in new combination. Not 8+ because no single subproblem is research-frontier hard.
- **Best-Paper 7/10:** Appropriate. Three-punch evaluation is strong. Bug discovery quality is the uncontrollable variable. The conditional 8–9 if N2 succeeds is fair.
- **Feasibility 8/10:** Appropriate. Major risk (grammar compiler) eliminated. Feature-checker is scoped. Staged rollout mitigates adapter risk.

---

## 5. Portfolio Differentiation — PASS

The differentiation analysis in §10 is thorough and accurate:

- **vs. ml-pipeline-selfheal:** Training repair ≠ inference localization. Different pipeline type, different techniques. ✅
- **vs. cross-lang-verifier:** Cross-language translation equivalence ≠ NLP pipeline stage consistency. ✅
- **vs. dp-verify-repair, tensorguard, tensor-train-modelcheck:** Intra-model verification ≠ inter-stage causal localization. ✅
- **vs. algebraic-repair-calculus:** Repair (treatment) ≠ localization (diagnosis). ✅
- **vs. causal-risk-bounds, causal-robustness-radii, causal-plasticity-atlas:** Statistical causal analysis ≠ software fault localization via do-calculus. ✅
- **vs. zk-nlp-scoring:** Completely different technique and goal. ✅

The closest neighbor is `ml-pipeline-selfheal`, but the training-vs-inference and repair-vs-localization distinctions are genuine. No portfolio overlap.

---

## Remaining Concerns

1. **300 hand-crafted seeds is a hidden time bomb.** "A linguist can write these in 1–2 days" is optimistic. Each seed needs transformation-applicability annotations across 15 transformations. If seed quality is poor, generation coverage suffers silently (N1 rank check helps but doesn't catch within-construction distribution gaps). **Severity: LOW-MEDIUM.**

2. **Feature-checker calibration risk.** The checker covers ~80 feature constraints for ~15 clause types. If it is too restrictive, shrunk counterexamples are non-minimal (acknowledged in Risk #3). If too permissive, ungrammatical outputs damage credibility. The mitigation (test against 10K spaCy-parsed sentences) is necessary but may be insufficient for rare constructions. **Severity: LOW.**

3. **Tier 1 bug yield.** The 30% risk of <5 actionable bugs is the single largest uncontrollable variable. The fallback (Tier 2 + Tier 3 + GPT-4 paper) is viable but drops the "10 Bugs, 10 Words" centerpiece. The pre-screen in week 1 is the right mitigation. **Severity: MEDIUM** (acknowledged, well-mitigated).

4. **N2 two-track carries zero expected value into the paper.** At 40% failure risk with a 4-week budget, the expected contribution is thin. If it fails, it's wasted time. If it succeeds, the benefit is significant (research track eligibility). The decision to attempt it is rational given the payoff asymmetry, but the plan should not depend on it. **It does not — the plan is self-contained without N2.** No concern.

---

## Final Verdict

### **SIGNOFF** ✅

The final approach is internally consistent, addresses all fatal flaws from the debate, includes correctly stated and achievable math results, has a realistic LoC estimate and sound architecture, provides honest scores, and is genuinely differentiated from the portfolio.

**Conditions (non-blocking):**
- Ensure the 300 hand-crafted seeds are developed early (week 1–2) and validated against the N1 rank check before committing to the full test campaign.
- Calibrate the feature-checker against ≥10K parsed sentences as stated in Risk #3 mitigation — this should be a hard milestone, not a suggestion.
- Run the Tier 1 pre-screen in week 1 as planned. If <3 bugs from passivization+NER+sentiment on spaCy, trigger the scope expansion immediately.

### Recommended Composite Score: **7.5/10**

This matches the final approach's self-assessment. The score is honest: the project fills a genuine gap with formal contributions and a strong evaluation plan, but the contracting market, moderate difficulty (known techniques in new combination), and bug-yield uncertainty prevent a higher rating. If N2 is proved and Tier 1 yields ≥10 bugs, the score rises to 8–8.5. If Tier 1 yields <5, the fallback paper is a solid 6.5.

---

*Verified by Independent Verifier. All checks passed. No blocking issues found.*
