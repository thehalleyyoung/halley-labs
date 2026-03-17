# Math Depth Assessor Verification Report

## Verdict: APPROVE WITH CONDITIONS

The final approach is mathematically coherent and addresses the overwhelming majority of my original critique points. Three minor classification issues and one unresolved tension require correction before the approach is fully sound on paper.

---

## Checklist Results

### 1. Were your mathematical soundness gaps addressed?

**ρ-at-merge-points:** YES. Amendment D1 explicitly specifies ρ-after-join with inner fixpoint iteration, bounded convergence via cache geometry (O(|sets| × |ways|) iterations), and a monotone-by-construction single-pass fallback. The diamond-CFG precision test in Phase Gate 3 validates the approach empirically. This fully addresses my §2 concern.

**Bounded speculation unsoundness:** YES. Amendment D8 introduces a residual leakage term: at most log₂(⌈F/64⌉) bits per misspeculation point beyond depth W, where F is the function's memory footprint. For crypto functions with F < 64KB, the residual is ≤ 10 bits—conservative but sound. This is included in the A1/A2 proof obligations. My concern from the Approach B analysis (§2 of the critique) is resolved.

**Type inference soundness coupling (Approach C):** N/A. The final approach borrows only C's contract-typed notation, not its type inference machinery. Binary-level type inference—the fatal challenge I identified in C—is correctly excluded.

### 2. Were your novelty deflation points incorporated?

**A5 (reduction operator) at ~60% genuine novelty:** YES. The Mathematical Contributions table labels A5 as "DEEP NOVEL" with "~60% genuinely new after deflation." The prose explicitly states "No direct prior art for spec×cache×quant reduction" while acknowledging the reduced-product framework itself is classical. This is honest.

**A7 (composition rule) as instantiation of Smith (2009):** PARTIALLY. A7 is correctly labeled INSTANTIATION in the Mathematical Contributions table, consistent with my deflation. However, the accompanying note—"making it work soundly over cache-state domain with taint-restricted counting under speculative semantics is substantially more than an instantiation"—hedges against its own label. If it is substantially more than an instantiation, it should be labeled NOVEL. If it is an instantiation, the note should be dropped or softened. This internal contradiction must be resolved. My recommendation: keep the INSTANTIATION label and change the note to "domain-specific engineering to instantiate under speculative cache semantics is non-trivial but follows established composition theory."

**A6 (quantitative widening) deferred:** PARTIALLY. The approach correctly notes A6 is "Not required for v1 fixed-iteration targets; deferred to evaluation on variable-iteration code" and marks it ENABLING rather than load-bearing. However, A6 retains the DEEP NOVEL classification despite being deferred. My original critique was unambiguous: "Novel but deferred — cannot claim as a contribution unless it appears in the evaluation." The label should be downgraded to "NOVEL (deferred)" or "DEEP NOVEL (deferred, not claimed for v1)" to avoid overclaiming in a future paper introduction.

### 3. Were your crown jewel stress test findings incorporated?

**4–7 person-month estimate for ρ:** YES. §Hard Subproblems #1 states "Estimated effort: 4–7 person-months," directly reflecting my assessment.

**60% success probability for "provable AND useful":** YES, implicitly. The risk table assigns 25% to ρ precision failure and 15% to ρ soundness/termination failure, yielding ~60% combined success probability (1 − 0.25)(1 − 0.15) ≈ 64%. This is consistent with my 60% estimate. The decomposition into orthogonal failure modes is actually cleaner than my combined figure.

**Fallback (direct product without ρ):** YES. §Hard Subproblems #1 explicitly states: "If ρ fails entirely, the direct product still yields sound bounds—the paper degrades from 'novel reduced product' to 'CacheAudit extended with speculation + composition,' which remains publishable."

### 4. Were your independence condition findings incorporated?

**ChaCha20 as trivial benchmark:** YES. Amendment D2 explicitly notes "ChaCha20's trivial satisfaction of independence noted as a limitation." Phase Gate 1 success criteria counts ChaCha20 but flags it as "non-stress-testing."

**4th benchmark with genuine cache-state correlation:** YES. T-table AES with related subkeys is added (Amendment D2), and a scatter/gather AES implementation is mentioned. This directly addresses my §5 requirement.

**Rényi fallback honest difficulty:** YES. Amendment D3 and §Hard Subproblems #3 characterize the Rényi path as "insurance against abandonment, not a path to a best paper," with 1–2 months additional research and 2–5× precision loss. This is the honest assessment I demanded.

### 5. Are mathematical contributions correctly classified?

**Classification consistency:** MOSTLY YES, with three issues:

1. **A6 (widening): DEEP NOVEL but deferred.** As noted above, the label overclaims for v1. Condition: add "(deferred)" qualifier.

2. **A8 (independence condition): labeled NOVEL at ~25% genuinely new.** My critique assessed this as "~75% routine application" of Smith (2009). At ~25% genuine novelty, NOVEL is generous. APPLIED or INSTANTIATION would be more accurate—it is applied verification of a known algebraic property on specific crypto patterns. Condition: downgrade to "APPLIED NOVEL" or add honest caveat that the genuinely new content is the characterization on specific crypto patterns, not a new theorem.

3. **A1 (speculative collecting semantics): inconsistency between tables.** The "What Is Novel vs. Adapted" table classifies the speculative trace semantics as "ADAPTED (~30% new)," but the Mathematical Contributions table classifies A1 as "NOVEL." These should be reconciled. At 30% genuine novelty, NOVEL is defensible (the new 30%—quantitative observation function over speculative traces—is theoretically meaningful), but the tension should be acknowledged.

**Load-bearing status:** CORRECT. A5 is correctly load-bearing. A6 is correctly ENABLING (not critical for v1). A7 is correctly load-bearing.

**Crown jewel identification:** CORRECT. A5 is the right choice—it is the deepest genuinely novel component after systematic deflation, and it is what makes the combination tractable rather than vacuously imprecise.

### 6. Is the phase gate structure mathematically sound?

**Phase Gate 1 (composition theorem):** YES, with a caveat. It tests independence on 4 patterns (AES distinct subkeys, AES related subkeys, ChaCha20, Curve25519) with success criteria ≥3 of 4. Since ChaCha20 is trivially satisfied, this effectively requires ≥2 of 3 non-trivial patterns. This is a reasonable threshold—it correctly identifies when the independence condition is viable.

**Phase Gate 2 (precision canary):** YES. It tests D_cache ⊗ D_quant on AES T-table under LRU at 3×/5×/10× thresholds. This validates the quantitative counting machinery *independently* of ρ, which is precisely its purpose. The debate correctly noted this does NOT validate ρ, and the final approach correctly positions ρ validation in Phase Gate 3.

**Kill triggers:** APPROPRIATE. PG1 kills only if composition fails for ALL patterns AND Rényi yields vacuous bounds—this is a conjunction that avoids premature termination. PG2 kills at 10× after redesign—aggressive but justified for the core counting machinery. PG3 uses a "fallback trigger" (not kill), pivoting to Reduced-B—this is the right structure, preserving the composition contribution even if speculation proves vacuous.

---

## Mathematical Soundness Assessment

The approach is mathematically coherent. The three-way reduced product D_spec ⊗ D_cache ⊗ D_quant is well-defined over domains with bounded height (64 sets × 8 ways ensures finite lattice). The ρ-after-join strategy with bounded inner fixpoint is the correct resolution of the merge-point problem I identified. The residual leakage term for bounded speculation is a clean soundness argument.

**One remaining tension:** The composition rule B_{f;g}(s) = B_f(s) + B_g(τ_f(s)) is stated as an equality in the notation but is actually an *upper bound* (as Smith 2009 establishes). The text occasionally treats it as an equality (e.g., the contract-typed interface notation), which could confuse the soundness story. This is a presentational issue, not a mathematical one, but should be corrected in any paper draft to avoid reviewer confusion.

**No remaining mathematical gaps** that threaten the core construction. The approach has correctly identified all hard subproblems and provided either solutions (ρ-at-merge-points, residual speculation term) or honest risk assessments with phase-gated detection (independence condition, ρ precision).

---

## Novelty Assessment Post-Synthesis

After all deflation and honest characterization, the genuine novelty inventory is:

| Result | Honest Classification | Genuine Novelty |
|--------|----------------------|-----------------|
| A5 (ρ) | DEEP NOVEL | ~60% — anchors the paper |
| A4 (D_quant) | NOVEL | ~40% — taint-restricted counting is non-trivial |
| A1 (speculative semantics) | ADAPTED/NOVEL borderline | ~30% — quantitative observation function |
| A7 (composition) | INSTANTIATION | ~25% — domain-specific instantiation of Smith 2009 |
| A8 (independence) | APPLIED | ~25% — verification on specific patterns |
| A6 (widening) | DEFERRED | N/A for v1 |

**Total genuine novelty: one DEEP NOVEL result (ρ) supported by one NOVEL result (taint-restricted counting) and several competent INSTANTIATION/ADAPTED components.** This is sufficient for a solid CCS paper. It is honest: the paper's contribution is the three-way reduction making the combination tractable, not a collection of individually deep results. The narrative coherence (answering LeaVe's open question, bridging PL and security) provides the publication-worthy framing around this core novelty.

---

## Conditions for Approval

1. **Resolve the A7 label/note contradiction.** Either upgrade A7 to NOVEL or soften the "substantially more than an instantiation" note. Internal contradiction undermines credibility with reviewers who will check.

2. **Add "(deferred)" qualifier to A6's DEEP NOVEL classification.** This prevents overclaiming in a paper that does not exercise widening in evaluation. If widening is later implemented and evaluated, the qualifier can be removed.

3. **Reconcile A1 classification between the two tables** (ADAPTED vs. NOVEL). Either consistently use NOVEL with a "~30% new" caveat, or ADAPTED with a note identifying what the 30% contributes theoretically.

These are classification/presentation conditions, not structural changes. The mathematical architecture is sound.

---

## Updated Risk Assessment

| Risk | Original (Critique) | Final Approach | My Revised Assessment |
|------|---------------------|----------------|----------------------|
| ρ provable AND useful | 60% success | 60% success (via 25%+15% decomposition) | **60% success** — unchanged; merge-point treatment helps but doesn't change fundamentals |
| Independence condition | "more serious than presented" | 30% failure; 4 benchmarks; Rényi fallback honestly characterized | **30–35% failure** — 4th benchmark and honest Rényi characterization are improvements; still the second-highest risk |
| Speculative bounds vacuous | Not explicitly assessed | 30% | **25–30%** — residual term and empirical context-count treatment are correct mitigations |
| Overall kill probability | ~35% (debate) | 30–35% | **30–35%** — consistent; the amendments reduce risk at margins but don't change the structural profile |
| Publishable paper probability | 55% (debate) | Not restated | **55–60%** — slight improvement from better risk mitigation and honest framing |
| Best-paper probability | 8% (debate) | Not restated | **8–10%** — honest novelty characterization paradoxically helps (reviewers trust honest papers more) |

The final approach's risk profile is mathematically sound and honestly stated. The phase-gate structure ensures that the 30–35% kill probability is *detected early* (months 2–8), not discovered after full implementation. This is the most important structural property for a research project with genuine technical risk.
