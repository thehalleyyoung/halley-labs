# Independent Verifier Final Assessment: Certified Leakage Contracts (proposal_00)

**Role:** Independent Verifier — Mathematician performing final signoff  
**Date:** 2026-03-08  
**Orientation:** Ruthlessly honest. Last line of defense.

---

## EXECUTIVE SUMMARY

**Verdict: CONDITIONAL CONTINUE — ENDORSED with score corrections.**

I endorse the panel's verdict but correct three scores downward. The panel was slightly generous on Difficulty and Best-Paper, and the Feasibility score deserves nuanced decomposition. The binding conditions (sketch proofs in 4 weeks, precision canary in 8 weeks) are exactly the right de-risking mechanism and represent the minimum rational investment before committing 16–22 months.

| Axis | Panel Score | My Score | Direction | Rationale |
|------|:----------:|:--------:|:---------:|-----------|
| Value | 6 | **6** | — | Agree. Real problem, narrow audience, regression detection compelling. |
| Difficulty | 6 | **5** | ↓ | One novel theorem, ~15–20K genuinely novel LoC. Skeptic was right. |
| Best-Paper | 4 | **3** | ↓ | Zero proofs + zero code + synthesis narrative = ~2–3% probability. |
| Laptop | 7 | **7** | — | Agree. Abstract interpretation on bounded cache domains is efficient. |
| Feasibility | 5 | **5** | — | Agree. ~55–60% publishable; ~35–40% at CCS specifically. |
| **Composite** | **5.6** | **5.2** | ↓ | |

**Confidence in verdict:** 65%.

---

## 1. ARE THE PANEL'S SCORES INTERNALLY CONSISTENT?

### V6/D6/BP4/L7/F5 — Mostly consistent, two tensions

**Consistency check passed:**
- V6 + F5 is coherent: the problem is real (V6) but delivery is uncertain (F5). These are independent axes — one measures the world's need, the other measures this project's capacity.
- L7 + F5 is coherent: laptop-tractable (L7) but feasibility includes mathematical and precision risk, not just computational cost.
- BP4 + F5 is coherent: low best-paper probability (4) is compatible with moderate publication probability (5) because fallback venues exist.

**Tension 1: D6 is too high given the evidence.**

The panel's own cross-critique concluded that the Skeptic's deflation of novelty was "closest to right" on theory_bytes. The Skeptic estimated 15–20K genuinely novel LoC and one modestly novel theorem (ρ). The fail-fast evaluation's decomposition (Table in §AXIS 2) is meticulous and, upon verification, correct:

- ρ operator: ~1.5–2.5K genuinely novel LoC (monotonicity/termination is ~1K; rest is reduced-product plumbing following Cousot & Cousot 1979)
- D_spec: ~3–5K novel (tagged powerset is standard; quantitative observation function integration is new)
- D_cache: ~2–3K novel (CacheAudit core is adapted; taint layer is new)
- D_quant: ~3–4K novel (CacheAudit counting extended; taint restriction is new)
- Fixpoint engine: ~1–2K novel (Bourdoncle WTO is textbook)
- Composition: ~2–3K novel (Smith 2009 instantiation + independence characterization)
- CLI/Tests: 0K novel

**Total genuinely novel: ~13–20K LoC within a ~65K artifact.** This is the difficulty of a solid single-author PhD paper — score 5, not 6. The ideation-stage depth check's D7 was anchored to the full 55–65K estimate (including adaptation), which conflated "not directly copy-pasted" with "genuinely novel." The Skeptic's correction is sound: reimplementing CacheAudit's cache model in a new language with per-line taint is real work, but it is *adaptation*, not novelty. A D6 still implicitly overcounts.

**My correction: D = 5.**

**Tension 2: BP4 is too high given theory_bytes=0.**

The fail-fast evaluation's BP3 is better supported than the panel's BP4. The evidence:

- Zero proofs exist anywhere in the repository. The theory stage ran for ~2 hours and produced architecture documents, not mathematics.
- Zero implementation exists. impl_loc = 0.
- The narrative is synthesis ("we combined CacheAudit + Spectector + Smith 2009 + LeaVe").
- 16–22 months of development lie ahead.
- Cache side channels peaked in 2018–2020; LeaVe is the exception, not the trend.

The base rate for best paper at CCS is ~2%. Adjustments:
- Answers a Distinguished Paper's open question: +2%
- Novel ρ operator: +1%
- Bridges PL and security: +0.5%
- Zero proofs: −2%
- Zero implementation: −1.5%
- Synthesis critique: −1%
- 16-month horizon: −1%
- Net: ~2–3%, which maps to 3/10.

The gap between BP4 (~4–6%) and BP3 (~2–3%) is small, but intellectual honesty demands the lower score. A BP4 implies that with some luck, this project could compete for best paper. A BP3 says: produce a solid paper and be grateful. The latter is the honest framing.

**My correction: BP = 3.**

---

## 2. IS THE VERDICT (CONDITIONAL CONTINUE) JUSTIFIED?

### Yes, but barely. The asymmetric payoff structure is the key argument.

**Expected value calculation:**

| Outcome | Probability | Value | EV |
|---------|:----------:|:-----:|:---:|
| Sketch proofs succeed → precision canary succeeds → CCS paper | 25% | +10 (career units) | +2.5 |
| Sketch proofs succeed → precision canary succeeds → SAS/VMCAI paper | 15% | +5 | +0.75 |
| Sketch proofs succeed → precision canary fails → redesign → eventual publication | 10% | +3 | +0.30 |
| Sketch proofs succeed → precision canary fails → abandon | 10% | −1 (8 weeks lost) | −0.10 |
| Sketch proofs fail → abandon at 4 weeks | 25% | −0.5 (4 weeks lost) | −0.125 |
| Sketch proofs succeed → all subsequent stages fail → abandon at 16–22 months | 15% | −8 (full project lost) | −1.20 |
| **Total expected value** | | | **+2.13** |

The expected value is positive because the binding conditions create an early exit at low cost. The worst-case scenario (15% probability of losing 16–22 months) is mitigated by the phase-gate structure detecting failure by month 4–8. The best-case scenario (25% × CCS paper) carries high reward.

**The critical question: is theory_bytes=0 disqualifying?**

No, but it is close. Here is why it is not disqualifying:

1. **The proofs are not deep.** The approach.json already contains a 3-sentence correctness argument for ρ that is essentially a proof sketch: "Each iteration decreases taint counts and capacity upper bounds (monotone decreasing on finite lattice). Lattice has bounded height ≤ S×W = 512. Therefore inner fixpoint terminates." Expanding this to a rigorous proof is 2–3 pages of careful lattice-theoretic work. It is not a Fields Medal problem.

2. **The composition theorem is genuinely an instantiation.** Smith 2009 Theorem 4.3 does the heavy lifting. The domain-specific work is threading cache-state transformers through the rule and verifying independence on crypto patterns. This is important *engineering* verification but follows a known template.

3. **The theory stage's failure to produce proofs is likely a pipeline issue, not a mathematical obstacle.** The stage ran for 2 hours and produced 24KB of JSON architecture + 40KB of evaluation planning. This is consistent with a pipeline that generates planning artifacts, not with a mathematician failing to prove theorems. (Under Interpretation 2 — that the proofs were attempted and failed — the picture is much darker, but the evidence favors Interpretation 1.)

**However:** The Skeptic is correct that every risk estimate in the planning documents is *conditioned* on the math working. The conditions have never been checked. The conditional continue says: check them, *then* continue. This is rational.

### Are the binding conditions sufficient?

**Mostly yes.** The six binding conditions map to the six identified risks:

| Condition | Risk Addressed | Sufficient? |
|-----------|---------------|:-----------:|
| Sketch proofs in 4 weeks | Unvalidated math (FLAW 1) | **YES** — if sketch proofs fail, the math is harder than estimated |
| Precision canary in 8 weeks | Vacuous bounds (FLAW 2) | **YES** — D_cache ⊗ D_quant on AES under LRU is the minimal test |
| Redefine scope to "CacheAudit-next" | Overclaiming novelty | **YES** — honest framing improves reception |
| Drop best-paper aspiration | Scope distortion | **YES** — targeting "solid CCS paper" is correct |
| Phase Gate 1 (composition theorem) | Independence fragility (FLAW 3) | **PARTIALLY** — testing on 4 patterns is necessary but not sufficient; real crypto has more diverse composition boundaries |
| Phase Gate 3 (speculation bounds) | ρ empirical hollowness (FLAW 4) | **PARTIALLY** — tests ρ on real code, but only after 4–8 months of investment |

**Gap identified:** The conditions don't adequately test ρ's *precision* at low cost. The precision canary (Phase Gate 2) tests D_cache ⊗ D_quant *without ρ*. The first real test of ρ's value is Phase Gate 3 at month 4–8, by which point 4–8 person-months are invested. I recommend adding:

**Additional condition (from this verification):** Within the 4-week sketch-proof period, construct a pencil-and-paper example showing ρ *strictly* tightens bounds on at least one non-trivial CFG (a diamond with one speculative branch). If no such example can be constructed by hand, ρ may be provably correct but empirically useless, and the project should pivot to the direct product formulation before investing in implementation.

---

## 3. HONEST MATHEMATICAL ASSESSMENT

### How many genuinely new theorems survive after deflation?

I enumerate every claimed mathematical contribution, classify it honestly, and count only results that constitute NEW mathematics — not adaptations, instantiations, engineering, or routine applications.

| ID | Claim | Honest Classification | New Math? | Reasoning |
|----|-------|----------------------|:---------:|-----------|
| A5 | ρ monotonicity, termination, soundness | **Modestly novel theorem** | ✓ | The monotonicity argument for the specific three-way interaction (spec→cache taint pruning→quant capacity zeroing) at merge points is genuinely new. But the *proof technique* is standard lattice theory on bounded-height domains. The *domain specificity* makes it a real contribution but not a deep one. **One theorem, ~2–3 pages.** |
| A5+ | ρ *precision* (strict improvement over direct product) | **Open question** | ? | The approach claims ρ "strictly tightens bounds." This is neither proved nor demonstrated. It may be true on hand-crafted examples and false on real crypto code. **Not yet a theorem — it is a conjecture.** |
| A2 | γ-only soundness for D_spec | **Application of Cousot & Cousot 1992** | ✗ | The γ-only framework is textbook. Applying it to a speculative trace domain follows the template: define the concretization, show the abstract transfer functions are sound under γ. Real work, not new mathematics. |
| A1 | Quantitative observation function over speculative traces | **Adaptation (~30% new)** | ✗ | Extends Guarnieri et al.'s trace model with a quantitative observation function. The extension is natural: replace boolean observations with counting. A definition, not a theorem. |
| A4 | Taint-restricted counting | **Minor extension** | ✗ | Restricting CacheAudit's counting to tainted configurations. The soundness argument is one paragraph: tainted configurations are a superset of secret-distinguishable configurations, so counting them gives a valid upper bound on min-entropy leakage. |
| A7 | Composition rule | **Instantiation of Smith 2009** | ✗ | The approach.json itself labels this INSTANTIATION. The algebraic identity B_{f;g}(s) ≤ B_f(s) + B_g(τ_f(s)) is Smith 2009 Theorem 4.3 with domain-specific plumbing. The plumbing is non-trivial *engineering* but is not a new theorem. |
| A8 | Independence characterization | **Case analysis** | ✗ | Verifying that the independence condition holds for specific crypto patterns (AES distinct subkeys, ChaCha20, Curve25519). This is domain-specific case analysis, not a general result. |
| A6 | Quantitative widening | **Deferred** | N/A | Not claimed for v1. Would be genuinely novel if completed, but it does not exist. |

### Count: ONE genuinely new theorem survives (ρ, A5).

The Skeptic's characterization — "ONE modestly novel theorem surrounded by instantiations" — is the correct summary. I concur.

### Is that theorem the reason the artifact is hard to build?

**Partially, but mostly no.** The artifact is hard because engineering abstract interpretation tools is hard. CacheAudit took ~3 years. The mathematics (ρ) is ~2–3K LoC out of ~65K — about 4% of the codebase. The remaining 96% is adaptation, integration, testing, and infrastructure. The difficulty is dominated by engineering, not mathematics.

### Is that theorem the reason it delivers value?

**For absolute bounds: yes. For regression detection: no.**

ρ's value is tightening absolute bounds by pruning speculative infeasibility from the cache taint and then from the capacity count. Without ρ (direct product), bounds are sound but looser. For the "killer app" of regression detection, loose bounds are tolerable because you're comparing two analyses, not thresholding against an absolute. ρ matters for the paper's *narrative* ("novel reduced product") more than for the tool's *primary use case* (regression detection).

This is a critical insight: **the project's value and its mathematical novelty are partially decoupled.** A direct-product version (no ρ) would still be a useful tool for regression detection. It would also still be publishable as "CacheAudit extended to x86-64 with speculation and composition" — a solid tools paper at SAS or VMCAI. ρ is what elevates the paper from SAS to CCS consideration by providing a novel theoretical contribution. But ρ carries the dual risk of being correct but empirically useless (the precision risk), in which case the paper degrades to a tools paper regardless.

---

## 4. THE theory_bytes=0 QUESTION — IS THE MATH ACHIEVABLE?

### Yes. The math is achievable. It is also modest.

Let me assess each proof obligation as a working mathematician:

**A5 (ρ monotonicity + termination): Achievable in 2–4 weeks of focused work.**

The proof has three parts:
1. `reduce_cache_from_spec` is monotone: clearing taint can only decrease the cache taint annotation. If the input lattice element is higher (more tainted), the output can only be higher (more tainted) or equal. This is a straightforward monotonicity argument on a boolean lattice.
2. `reduce_quant_from_cache` is monotone: zeroing capacity for untainted sets can only decrease the capacity count. Same argument.
3. The composition (step 1 then step 2, iterated at merge points) terminates because each iteration either decreases the combined measure |tainted lines| + Σ capacity, or is a fixpoint. The measure is bounded by S × W = 512. Therefore at most 512 iterations.

This is *not hard mathematics*. It is careful, domain-specific lattice theory. A graduate student in abstract interpretation could produce this proof in 1–2 weeks. The "no direct prior art" claim is true in the narrow sense that no one has combined these *three specific domains*, but the proof *technique* is entirely standard.

**Probability of success: 85%.** The 15% failure mode is discovering that `reduce_cache_from_spec` is NOT monotone when speculative context pruning interacts non-trivially with cache join operations at merge points. The approach's "inner fixpoint" strategy at merge points is designed to handle this, but it has never been tested against a concrete counterexample.

**A7 (Composition soundness): Achievable in 1–2 weeks.**

This is an instantiation of Smith 2009 Theorem 4.3. The proof is:
1. Define the min-entropy channel for f and g over cache observations.
2. Show that under the independence condition (g's cache observations are conditionally independent of f's secrets given τ_f(s)), the leakage adds: H_∞(S|O_{f;g}) ≥ H_∞(S|O_f) − H_0(O_g|τ_f(s)).
3. Convert to the B notation via B = log₂|S| − H_∞(S|O).

The domain-specific work is: formally stating the independence condition in terms of cache-state equivalence classes and verifying it for specific crypto patterns. This is case analysis, not deep proof.

**Probability of success: 90%.** The remaining 10% failure mode is discovering that the cache-state transformer τ_f cannot be made sufficiently precise to make the independence condition hold for any interesting pattern.

**A8 (Independence verification on crypto patterns): Achievable in 2–3 weeks.**

Case analysis on 4 patterns:
- AES with distinct subkeys: different rounds use different key bytes → cache footprints of consecutive rounds don't overlap → independence holds. (**High confidence: 90%.**)
- ChaCha20: data-independent permutation → zero cache leakage → independence trivially holds. (**Certain: 99%.**)
- AES with related subkeys: key schedule creates algebraic correlations → independence *fails*. Correction term d needed. (**Expected to fail; correction term is the contribution.**)
- Curve25519: conditional swap creates 1-bit correlation → independence holds if τ captures the swap's cache effect. (**Medium confidence: 70%.**)

**Overall probability that independence holds for ≥3 of 4 patterns: ~65%.** This is the panel's estimate, and I concur.

### Is the remaining math portfolio sufficient for a CCS paper?

**Marginal.** One modestly novel theorem (ρ), one instantiation (composition), and case analysis (independence) do not constitute a strong theory contribution. The paper's strength must come from the *systems evaluation* — demonstrating tight bounds on real crypto code, detecting known CVEs, showing composition overhead ≤ 2×. If the empirical results are strong, the theory provides sufficient scaffolding. If the empirical results are mediocre, the theory alone is not CCS-worthy.

**The honest framing:** This is a tools paper with a novel theoretical hook (ρ). CCS publishes such papers, but they need impressive evaluation results. The mathematical novelty alone would be a SAS poster, not a CCS paper.

### Is it achievable within 16–22 months?

**The math: yes, in 2–4 months.** The implementation: probably, if the implementer is experienced with abstract interpretation frameworks. The evaluation: another 4–6 months. Total: 8–14 months for the core work plus buffer. 16–22 months is conservative enough to absorb one major redesign.

**The Skeptic is right that this is ONE modestly novel theorem surrounded by instantiations.** But the Skeptic underweights the systems contribution. Producing a working tool that matches CacheAudit's precision while adding speculation and composition would be a genuine advance, even if the mathematical novelty is concentrated in 2–3K LoC of ρ. CCS publishes tools papers. The question is whether the tool works, not whether the math is deep.

---

## 5. FINAL VERDICT

### My Scores

| Axis | Score | Justification |
|------|:-----:|---------------|
| **Value** | **6/10** | Real problem, explicitly identified as open by a CCS Distinguished Paper. Narrow direct audience (~30–50 people). Regression detection is compelling. LLM competition erodes marginal value. |
| **Difficulty** | **5/10** | After honest deflation: ONE novel theorem (ρ), ~15–20K genuinely novel LoC within a ~65K artifact. Comparable to 1.5× CacheAudit in genuine novelty. A strong PhD chapter, not a multi-person multi-year effort. |
| **Best-Paper** | **3/10** | ~2–3% probability. Zero proofs, zero code, synthesis narrative, declining subfield, 16-month horizon. The Skeptic's 3 is correct. |
| **Laptop** | **7/10** | Abstract interpretation on L1 LRU cache domains with bounded speculation is demonstrably efficient. CacheAudit runs in seconds. Even at 100× overhead, CI-compatible. |
| **Feasibility** | **5/10** | ~55–60% chance of a publishable paper (at some venue). ~35–40% chance at CCS specifically. ~20–25% chance of the *proposed* paper with ρ as crown jewel, composition as centerpiece, and impressive empirical results. |
| **Composite** | **5.2/10** | |

### Verdict: **CONDITIONAL CONTINUE**

**Confidence: 65%.**

### I ENDORSE the panel's verdict with the following corrections:

1. **Difficulty: 6 → 5.** The panel's reconciliation overweighted the ideation-stage D7 anchor. The fail-fast evaluation's decomposition of genuinely novel LoC is meticulous and correct. The Skeptic was right.

2. **Best-Paper: 4 → 3.** The panel gave one point of credit that the evidence does not support. With theory_bytes=0 and impl_loc=0, there is no basis for assigning even ~5% best-paper probability. The honest estimate is 2–3%.

3. **Additional binding condition:** Within the 4-week sketch-proof period, construct a pencil-and-paper example demonstrating that ρ *strictly* tightens bounds on a non-trivial CFG (a diamond with one speculative and one non-speculative branch). If no such example can be constructed, ρ may be provably correct but empirically vacuous, and the project should commit to the direct-product formulation before investing 4–8 more months.

### Why not ABANDON?

Four reasons, in order of importance:

1. **The asymmetric payoff.** The binding conditions cost 4–8 weeks. If they fail, the project loses 4–8 weeks. If they succeed and the project continues to completion, the payoff is a CCS paper. The expected value is positive.

2. **The regression detection use case.** Even a "degraded" version (direct product, no ρ, composition on easy patterns) delivers genuine value for CI-integrated regression detection on crypto libraries. This is achievable with moderate confidence.

3. **The LeaVe positioning.** Answering a Distinguished Paper's explicitly identified open question provides a natural publication story that survives even if ρ proves empirically useless.

4. **The math is not deep enough to be a trap.** Ironically, the modest novelty of the math portfolio is a *reason to continue*, not a reason to abandon. Deep, novel math can harbor hidden obstacles. Shallow math — one lattice-theoretic theorem, one instantiation, case analysis — is unlikely to harbor surprises. The risk is not "the math is impossible" but "the math is correct but the tool doesn't work well empirically."

### Why not unconditional CONTINUE?

Because **theory_bytes=0 means every mathematical claim in the project is unvalidated.** The proofs *probably* work (my estimate: 85% for ρ, 90% for composition, 65% for independence on ≥3 patterns; joint: ~50% for all three). A 50% joint success probability does not justify unconditional commitment of 16–22 months. The conditional structure — prove the math first, then build — is the only rational approach.

---

## MATHEMATICIAN'S POSTSCRIPT

The deepest concern I have as a mathematician is not about the *correctness* of the proofs but about the *significance* of the results. The panel, across all stages, has spent considerable effort assessing whether ρ is "DEEP NOVEL" or "MODERATE NOVEL" or merely an instantiation. This debate, while important for scoring, misses the more fundamental question:

**Does ρ solve a problem that matters to the people who would use this tool?**

Crypto library maintainers want to know: "did this commit introduce a timing leak?" For that question, they need a sound analyzer with reasonable precision. They do not care whether the analyzer uses a reduced product or a direct product. They do not care whether ρ is novel or textbook. They care whether the tool runs in their CI pipeline and catches regressions.

The mathematical novelty of ρ serves the *paper*, not the *user*. This is not inherently bad — papers need novelty — but it means the project has two goals that may conflict: (1) build a useful tool, and (2) publish a novel paper. If ρ proves empirically useless, goal (1) is still achievable (direct product → regression detection tool) but goal (2) is severely compromised (tools paper at SAS, not CCS). The project should be evaluated primarily on goal (1), with goal (2) as a bonus.

The binding conditions correctly de-risk goal (2). But the project should proceed even if goal (2) fails, because goal (1) — a working regression detection tool for cache side channels in crypto binaries — is independently valuable.

---

## SUMMARY

| Question | Answer |
|----------|--------|
| Are scores internally consistent? | Mostly. D6 and BP4 are each ~1 point too high. |
| Is CONDITIONAL CONTINUE justified? | Yes. Asymmetric payoffs favor the conditional structure. |
| Honest math assessment? | ONE modestly novel theorem (ρ). Everything else is instantiations and engineering. |
| Is the math achievable? | Yes. The proofs are 2–4 months of work, not deep enough to harbor hidden obstacles. |
| Final verdict | **CONDITIONAL CONTINUE (65% confidence). ENDORSE with score corrections V6/D5/BP3/L7/F5 = 5.2.** |

*Signed: Independent Verifier*
