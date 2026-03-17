# Community Expert Verification Report: certified-leakage-contracts

## Panel Composition

| Expert | Role | Orientation |
|--------|------|-------------|
| **Independent Auditor** | Evidence-based scoring and challenge testing | Anchored to concrete data: LoC comparisons against real tools, hardware specs, publication records. Default stance: trust evidence, not narrative. |
| **Fail-Fast Skeptic** | Adversarial attack-vector enumeration and kill-probability estimation | Seeks project-fatal flaws. Default stance: the proposal is wrong until proven right. Assigns severity ratings and explicit failure probabilities per risk. |
| **Scavenging Synthesizer** | Value preservation, scope surgery, risk-adjusted optimization | Identifies the "diamond" worth preserving, proposes scope reductions, computes expected value under uncertainty. Default stance: find what's worth saving and cut the rest. |
| **Independent Verifier** | Final signoff, score consistency, contradiction detection | Checks all axes for internal consistency and evidence grounding. |

## Methodology

The verification followed a four-phase adversarial process:

1. **Independent Proposals.** Each expert produced a full assessment in isolation. No expert saw the others' work.
2. **Adversarial Cross-Critique.** Each expert attacked the weakest arguments of the other two, with explicit teammate-to-teammate challenges. The Auditor challenged the Skeptic's "trivial engineering" framing and the Synthesizer's optimistic publication probability. The Skeptic attacked the Auditor's Value score ("name 5 organizations that would adopt this") and the Synthesizer's risk-adjusted computation ("garbage-in, garbage-out"). The Synthesizer countered the Skeptic's "cache channels peaked" claim with the LeaVe 2023 Distinguished Paper and challenged the Auditor's CacheAudit timeline comparator.
3. **Panel Synthesis.** The lead reconciled all six documents (3 assessments + 3 cross-critiques), weighting arguments by evidence quality.
4. **Independent Verifier Signoff.** A final review checked score consistency, evidence gaps, internal contradictions, and community calibration.

---

## Axis 1: EXTREME AND OBVIOUS VALUE — Score: 6/10

### The Case For Value

The problem is real, documented with concrete CVEs, and positioned against a Distinguished Paper's explicit open question. The **Auditor** anchored this: CVE-2018-0734 and CVE-2018-0735 (OpenSSL ECDSA nonce timing leaks introduced by compiler optimization) are genuine vulnerabilities that survive source-level review. CVE-2022-4304 (RSA timing oracle) is high-impact. libsodium's documentation explicitly disclaims binary-level constant-time guarantees. The FIPS 140-3 audit pipeline processes ~200 CMVP modules/year with IG 2.4.A requiring side-channel resistance evidence.

The **Synthesizer** made the strongest strategic argument: LeaVe won Distinguished Paper at CCS 2023 for verifying hardware-side leakage contracts on RISC-V. The LeaVe authors (Guarnieri, Reineke, and collaborators) explicitly identified software-side contract verification as an open question in their 2025 SETSS tutorial. A CCS reviewer who remembers LeaVe will immediately understand why this matters.

The **regression detection use case** is the most robust value argument, identified by all three experts as the "killer app": even when absolute bounds are 5–10× conservative (LRU→PLRU over-approximation, widening imprecision), *changes* between binary versions reliably flag introduced leaks. This transforms modeling imprecision from an existential threat into a tolerable over-approximation.

### The Case Against

The **Skeptic** delivered devastating counter-evidence: the direct user base is ~50–100 crypto library maintainers globally. Google (BoringSSL) already uses cachegrind-diff in CI. OpenSSL has not adopted CacheAudit, Spectector, or Binsec/Rel in 10+ years. libsodium is maintained by essentially one person. No organization has publicly expressed interest in quantitative speculative cache analysis for CI. Zero real-world CVEs have been attributed to speculative cache leakage in crypto library code (as opposed to cross-process Spectre attacks, which are an OS/hardware concern).

LLM-based side-channel triage handles ~90% of practical cases (secret-dependent branches, variable-time lookups, missing `cmov`) in seconds. The tool targets the residual 10%, but even within that residual, cachegrind-diff handles regression detection for non-speculative leaks today.

### Reconciliation

The Skeptic's 4/10 discounts the genuine publication value of answering a Distinguished Paper's open question. The Synthesizer's 7/10 assumes the full supply-chain vision is achievable in scope. The publication value (clear CCS slot via LeaVe positioning) justifies 6, but practical deployment value is near-zero against deployed baselines. The LLM competition argument prevents 7+.

**Scope caveat (from Verifier):** This score is a project-level assessment assuming speculative analysis is eventually delivered. Under the recommended Reduced-A-minus scope (no speculation in Paper 1), the Paper 1 value is 5–5.5. The full 6 requires the speculative complement to LeaVe in Paper 2.

---

## Axis 2: GENUINE SOFTWARE DIFFICULTY — Score: 6.5/10

### The LoC Debate

This axis produced the widest disagreement (Skeptic 5, Auditor 7, Synthesizer 7).

**Auditor's evidence-based breakdown (most credible):**

| Module | Proposal LoC | Auditor Adjusted | Notes |
|--------|:---:|:---:|---|
| Lifter adapter | 4–6K | 4–5K | Tedious, correctness-critical, no template for vpshufb/aesenc/pclmulqdq |
| D_spec (speculative) | 8–12K | 8–10K | Follows Spectector's published semantics; tagged powerset is non-trivial |
| D_cache (tainted LRU) | 9–13K | 6–9K | CacheAudit core (~4–5K) + taint annotations; inflated in proposal |
| D_quant (counting) | 8–12K | 5–8K | Taint-restricted counting extends CacheAudit; inflated in proposal |
| ρ + fixpoint engine | 10–14K | 10–12K | Crown jewel. 3–5K correctness-critical ρ code. Credible. |
| Contracts + composition | 7–10K | 7–9K | Reasonable |
| CLI/CI | 4–6K | 4K | Standard |
| Tests | 18–22K | 15–18K | Proportional |
| **Total** | **75–85K** | **55–70K** | **35–45K genuinely novel** |

**The Skeptic's counter:** Only 3–8K LoC is genuinely novel research code (ρ ~3–5K, quantitative widening ~2K, novel lattice operations ~2K). Everything else — cache domain (extends CacheAudit), fixpoint engine (Bourdoncle's WTO), speculative domain (standard powerset) — is "textbook abstract interpretation with a twist."

**The Auditor's rebuttal was decisive:** The Skeptic conflates "uses known techniques" with "easy to implement." CacheAudit itself (a simpler system without speculation, composition, or taint annotations) was ~15K LoC and took Boris Köpf's group ~3 years. The claim that the combined system requires only 3–8K LoC of genuine work is not credible. However, second-mover advantage is real: published algorithms, existing binary lifters, and mature fixpoint engine designs substantially reduce development time compared to CacheAudit's pioneering effort.

**Reconciled at 6.5 (Verifier-endorsed):** The Skeptic's 5 ignores integration difficulty and the engineering reality of abstract interpretation tools. The Auditor's and Synthesizer's 7 is slightly high given ~40–50% of the system follows published blueprints. The genuinely hard work is concentrated in ρ and the system-level integration of three novel domains — substantial but not majority-novel.

---

## Axis 3: BEST-PAPER POTENTIAL — Score: 5/10

### The Synthesis vs. Paradigm Shift Debate

The **Skeptic** challenged: "Name ONE recent best paper that was a synthesis of known techniques." Cache side channels peaked 2018–2020. The composition rule B_{f;g}(s) = B_f(s) + B_g(τ_f(s)) instantiates Smith (2009) Theorem 4.3. The "first to combine four properties" claim is a conjunction fallacy. Best papers create new categories; this creates a new SKU in an existing category.

The **Synthesizer** countered with dispositive evidence: LeaVe won CCS Distinguished Paper in 2023, three years after the claimed peak. The CCS PC awarded its highest recognition to a leakage contracts paper. The "community has moved on" narrative is empirically falsified. ProVerif (CCS 2001, subsequent awards) was fundamentally a synthesis of Dolev-Yao models, Horn clause resolution, and process algebra — three established techniques combined via a novel encoding.

The **Auditor** offered the most calibrated view: LeaVe won because it introduced a genuinely new *framing* (hardware-side contracts verified on RISC-V silicon), not because reviewers love cache analysis per se. The software side is inherently less novel because software-side verification is a mature area. The reduced product domain ρ is a genuine contribution to abstract interpretation theory (~60% new), but the novelty concentration in one operator (3–5K LoC) is a thin pedestal for best-paper recognition.

**Reconciled at 5/10:** ~8% best-paper probability (the debate consensus from ideation). Sufficient for a solid CCS accept. The LeaVe positioning and regression detection framing are genuinely compelling. But the synthesis headwind, narrow novelty concentration, and the proposal's own acknowledgment of structural challenges cap the ceiling. Best-paper requires either a dramatic precision result (ρ tightens bounds 3×+ over direct product) or a CVE caught by no other tool.

---

## Axis 4: LAPTOP CPU + NO HUMANS — Score: 7/10

### Where All Experts Agree

L1 LRU cache analysis via abstract interpretation is demonstrably tractable on laptop hardware. CacheAudit analyzed AES in seconds on decade-old hardware. For a 32KB L1D (64 sets × 8 ways × 1 taint bit), per-abstract-state size is ~128 bytes. Per-function analysis avoids whole-program fixpoints. The analysis is purely static — no hardware-in-the-loop, no human annotation, no GPU. A single `make eval` invocation runs the full pipeline.

### Where They Disagree: Speculative Context Count

The speculative context count C is the key unknown. The original "10–20 contexts" claim was correctly retracted. The Skeptic raises a valid technical point: a function with 20 branches within a 50-instruction speculation window can have up to 2^20 speculative contexts in the worst case. The approach.json's O(|sets| × |ways|) bound applies to ρ's inner fixpoint, not to C itself.

The **Synthesizer's resolution** is the most practical: cut D_spec from Paper 1's critical path. Non-speculative analysis is clearly laptop-feasible (8/10). Speculative analysis has genuine uncertainty about C (5–6/10). Under the recommended reduced scope, this concern is deferred.

**Reconciled at 7/10:** Non-speculative analysis on LRU with bounded geometry is trivially laptop-feasible. The 30–90 minute full-library analysis budget is plausible for non-speculative mode. Memory (300–600MB for Curve25519 with hash-consing) is manageable. Zero-annotation binary analysis is achievable with instruction subset restrictions.

---

## Axis 5: FEASIBILITY — Score: 5/10

### The Central Red Flag: theory_bytes=0

State.json is unambiguous:

```json
"theory_bytes": 0,
"theory_score": null,
"impl_loc": 0,
"monograph_bytes": 0,
"code_loc": 0
```

Every metric is zero. The theory stage was designed to produce proofs. It produced a 24KB JSON specification (approach.json) and a 40KB evaluation plan (empirical_proposal.md) — detailed, well-structured, and valuable as engineering blueprints — but zero formal proofs.

**The Skeptic's diagnosis is partially valid:** Two pipeline stages have produced increasingly detailed descriptions of work that hasn't been done. Each stage has generated more prose — specifications, contingency plans, fallback paths — without producing any artifact demonstrating the core construction works. The ρ operator, the intellectual crown jewel, exists only as pseudocode. The composition theorem has not been proved. The independence condition has not been verified on any crypto pattern.

**The Synthesizer's counter has merit:** The approach.json is unusually detailed — pseudocode for every transfer function, complexity bounds, correctness arguments, fallback paths. The soundness of ρ follows from a simple lattice-theoretic argument (monotone decreasing on a finite lattice of height S×W=512). These are verification tasks for well-specified designs, not deep open problems.

**The Auditor's framing is most honest:** The plan is excellent. The gap between "plan" and "implementation" is where projects die. At this stage, what exists is a research proposal, not a research project. The difference matters.

### Timeline Assessment

- CacheAudit: ~15K LoC, ~3 person-years, established group.
- Spectector: multi-year, multi-institution collaboration.
- This proposal: ~55–65K LoC (Reduced-A-minus), single researcher, 16–22 months claimed.

The Auditor estimates 24–36 months (1.5–2× the claimed timeline). With 30–35% research kill probability and ~35–40% execution risk for timeline overrun, the combined probability of the full claimed contribution is ~20–25%.

**However:** The fallback hierarchy genuinely reduces tail risk. Even partial success (Reduced-B without speculation and ρ, or direct product with composition and regression detection) produces a publishable artifact. The Synthesizer's expected value computation — E[continue]=5.78 vs. E[abandon]=1.5, differential of 4.28 — is robust even under conservative assumptions.

**Reconciled at 5/10 (conditional):** This score is contingent on the binding conditions below being enforced. Without them, 4.5 is more appropriate. The fallback structure and detailed specifications earn the half-point above the Auditor's 4; the zero-artifact starting point prevents the Synthesizer's 6.

---

## Fatal Flaws

| # | Flaw | Severity | Kill Probability | Mitigation |
|---|------|----------|:---:|---|
| 1 | **ρ precision failure** — crown jewel adds no measurable precision improvement over direct product | HIGH | 25–30% | Precision canary (PG2); diamond-CFG test (PG3); single-pass monotone fallback; degrades to "CacheAudit + composition" if ρ vacuous |
| 2 | **Independence condition failure** — min-entropy composition breaks on primary benchmarks (AES with related subkeys) | MEDIUM-HIGH | 30% | 4 crypto pattern tests; correction terms; Rényi entropy fallback (1–2 months, 2–5× looser) |
| 3 | **Speculative bounds vacuous** — D_spec adds nothing over non-speculative analysis on real programs | MEDIUM | 25–30% | Reduced-B fallback drops speculation; still novel for composition + quantitative |
| 4 | **LRU/PLRU model gap** — formal guarantees 10–50× loose on Intel hardware | MEDIUM (known) | 80% (limitation, not risk) | Regression detection is model-tolerant; ARM Cortex-A (LRU) as primary absolute-bounds platform |
| 5 | **Single-researcher execution risk** — 55–65K LoC of correctness-critical Rust in 16–22 months | HIGH | 35–40% | Scope reduction; phase gates; but no mitigation for fundamental timeline pressure |
| 6 | **theory_bytes=0 meta-risk** — two stages of planning with zero artifacts predicts continued planning | MEDIUM | 15–25% | Phase gates with hard kill triggers force confrontation with technical risk |
| 7 | **Binary lifter coverage gap** — BAP/angr may not correctly lift SIMD/AES-NI instructions | LOW-MEDIUM | 15–20% | Restrict initial evaluation to non-SIMD benchmarks; validate lifter on instruction subset first |

**Combined kill probability: 30–40%.** The fallback hierarchy ensures that even under partial failure, a publishable artifact exists in ~60–70% of outcomes.

---

## Expert Disagreements (Unresolved)

### 1. Is ρ Genuinely Hard or Trivially Implementable?

The **Skeptic** rates ρ at "6/10 difficulty" and claims it's "3–5K LoC of constraint propagation code" — meaning the entire crown jewel of an 85K LoC project represents 4–6% of the codebase. The **Auditor** and **Synthesizer** counter that ρ's correctness-critical nature (convergence, monotonicity, precision at merge points) makes it the hardest component despite its small size, analogous to how a compiler's register allocator is 2% of the codebase but 20% of the difficulty.

**Panel assessment:** The Skeptic correctly identifies that ρ is *conceptually compact* but underweights its *engineering difficulty*. The approach.json specifies a single-pass monotone fallback (provably sound), but the iterative ρ at merge points with inner fixpoint has never been implemented and tested. The Synthesizer's observation that the Skeptic's D5 + F3 scores are contradictory (trivial engineering doesn't have 40–45% kill probability) is dispositive.

### 2. Does theory_bytes=0 Signal a Planning Spiral?

The **Skeptic** diagnoses a "planning spiral" — indefinitely refining plans to avoid confronting hard technical risk. The **Synthesizer** argues the approach.json is "unusually detailed" and the proof obligations are "straightforward to discharge." The **Auditor** splits the difference.

**Panel assessment:** Both interpretations have merit. The specifications are genuine intellectual work product that the theory_bytes metric fails to count. But the absence of completed proofs after a theory phase is a real gap — "straightforward to discharge" is not "discharged." The binding conditions (C1–C3) are the mechanism that resolves this by forcing artifact production within hard deadlines.

### 3. Is the LLM Competition Argument Fatal?

The **Skeptic** argues LLMs cover 90% of practical side-channel triage. The **Auditor** and **Synthesizer** argue the tool targets the residual 10% plus provides formal guarantees LLMs cannot.

**Panel assessment:** LLMs handle pattern-matching cases but cannot produce formal bounds, certificates, or compositional guarantees. The tool's value is in the hard tail and in CI-integrated formal regression detection. The LLM argument limits the value ceiling (preventing 8+) but doesn't eliminate value.

---

## Composite Score

| Axis | Auditor | Skeptic | Synthesizer | Reconciled | Notes |
|------|:---:|:---:|:---:|:---:|---|
| 1. Extreme Value | 6 | 4 | 7 | **6** | Publication value strong (LeaVe); deployment value weak |
| 2. Genuine Difficulty | 7 | 5 | 7 | **6.5** | ρ is genuinely hard; ~40–50% follows published blueprints |
| 3. Best-Paper Potential | 5 | 3 | 6 | **5** | ~8% BP probability; synthesis headwind real but not fatal |
| 4. Laptop CPU + No Humans | 7 | 5 | 8 | **7** | Non-speculative clearly feasible; speculative C unknown but deferrable |
| 5. Feasibility | 4 | 3 | 6 | **5** | theory_bytes=0; fallback hierarchy helps; conditional on binding gates |
| **Composite** | **5.8** | **4.0** | **6.8** | **5.9** | — |

---

## VERDICT: CONDITIONAL CONTINUE

**Publication probability estimates:**

| Outcome | Probability |
|---------|:---:|
| Full contribution (all RQs, ρ works, composition works, speculative analysis adds value) — CCS/S&P | 15–20% |
| Reduced contribution (Reduced-A-minus: no speculation, ρ + composition + regression) — CCS/ESORICS | 25–30% |
| Degraded contribution (no ρ, composition + regression detection) — NDSS-BAR/DIMVA | 15–20% |
| Abandoned after phase gate failure | 30–40% |

**P(publishable at strong venue) ≈ 40–50%**
**P(publishable anywhere) ≈ 60–70%**
**P(best paper) ≈ 5–8%**

### Binding Conditions (Violation → ABANDON)

**C1: Precision canary within 4 months.** D_cache ⊗ D_quant running on AES T-table with bounds ≤3× exhaustive ground truth. This is the minimum viable evidence that quantitative counting produces useful bounds. >10× after redesign → KILL.

**C2: Composition theorem paper proof within 3 months.** Not a specification — an actual written proof with the independence condition precisely stated and checked against AES round composition. If independence excludes AES, pivot to correction terms or Rényi immediately. All patterns fail AND Rényi vacuous → KILL.

**C3: ρ precision demonstration within 8 months.** On at least one non-trivial function, the reduced product must beat the direct product by >10%. Zero measurable improvement → degrade scope to Reduced-B.

### Strong Recommendations

**C4: Adopt Reduced-A-minus scope for Paper 1.** Defer full speculative analysis. This eliminates 30% of kill probability and produces a cleaner paper. Speculation appears as a prototyped extension.

**C5: Budget 24–30 months, not 16–22.** The CacheAudit comparator (3 years / 15K LoC for a simpler system) suggests the claimed timeline is 1.5–2× too optimistic.

**C6: Confront ρ immediately.** The Skeptic correctly notes that ρ can be tested on AES T-table with ~500 LoC in ~2 weeks. Every month spent refining specifications without testing ρ increases the probability of late discovery that the crown jewel doesn't help.

### Why CONTINUE

1. **The research question is genuine.** LeaVe's open question (software-side contracts) is real, and the Guarnieri/Köpf/Reineke program is actively seeking this work.
2. **The fallback structure is well-designed.** Each phase gate has a degraded-but-publishable exit path. Even partial success produces a useful contribution.
3. **The evaluation plan is exceptional.** Seven research questions with falsification criteria, theorem connections, and concrete protocols. Better than 90% of published CCS papers.
4. **The regression detection framing is practical.** Robust to modeling imprecision, demonstrated with 3 real CVEs, and novel (no prior tool does formal regression detection for cache side channels).
5. **The risk-adjusted expected value is positive.** E[continue] ≈ 5.78 vs. E[abandon] ≈ 1.5; the differential of 4.28 survives even conservative probability adjustments.

### Why the Conditions Are Non-Negotiable

1. **Zero artifacts exist.** theory_bytes=0, impl_loc=0. Every claim is paper-only. The history of formal methods for side channels is littered with proposals that looked sound on paper and failed during implementation.
2. **The timeline is unrealistic for one researcher.** 55–65K LoC of correctness-critical Rust in 16–22 months requires everything to go right at every phase gate.
3. **The crown jewel is unvalidated.** ρ has never been prototyped, let alone demonstrated to tighten bounds. A project that invests 8 months before testing its core construction is accepting unnecessary risk.
4. **The competitive landscape is accelerating.** By 2028, LLM-assisted side-channel analysis may handle 95% of practical cases, compressing the target audience further.

---

## Salvage Analysis (If ABANDON Were Chosen)

If the project were abandoned today, the following diamonds could be salvaged:

1. **Composition theorem + independence characterization** as a standalone theory paper at SAS/VMCAI (~2 months to prove and write). Value: 2/10.
2. **approach.json as a detailed blueprint** for another group (Guarnieri/Köpf/Reineke) to build. Value: 1/10 standalone, but high leverage if adopted.
3. **The Reduced-A-minus scope** as a focused follow-up by a group with existing abstract interpretation infrastructure (CEA List / Binsec team, Saarland / CacheAudit team). Value: conditional on adoption.

The E[abandon] = 1.5/10 vs. E[continue] = 5.78/10 differential decisively favors continuation.

---

## Community Expert Assessment: Would the Security/Crypto Community Care?

As someone who knows this field's practitioners, open problems, and review culture:

**What practitioners want:** A tool that runs `./analyze --binary libcrypto.so --regression v1.1.1 v1.1.2` and outputs a JSON report flagging functions where cache leakage increased, with per-function bit-level bounds. They want it to run in CI in under an hour, require zero configuration, and produce zero false positives on known-good code. This proposal aspires to this but won't deliver it at v1 quality.

**What CCS reviewers want:** A paper that introduces a genuinely new abstract domain (ρ is this), proves a non-trivial composition theorem (the independence characterization is this), demonstrates precision on real benchmarks (the evaluation plan targets this), and positions against prior work with clear improvement (the CacheAudit + Spectector baselines do this). This proposal has the ingredients for a solid CCS accept.

**What would make this a best paper:** Catching a real, previously unknown speculative cache leak in a production crypto library via automated analysis. Alternatively, demonstrating that ρ tightens bounds by 5×+ over the direct product on a non-trivial benchmark. Either would transform the narrative from "competent synthesis" to "this tool finds things nothing else can find."

**What the community would find underwhelming:** Absolute bounds that are 10× loose on all benchmarks. A paper where the ρ ablation shows ≤5% improvement. Composition overhead >3× (making compositional analysis worse than monolithic). Any of these would reduce the paper from "solid accept" to "borderline reject."

**Bottom line:** The proposal targets a real gap that practitioners have acknowledged but not prioritized — because they've been living without formal side-channel guarantees for 20 years and simpler tools handle the easy cases. The value is in the hard tail: speculative leaks, quantitative bounds, formal composition. If the tool works, it will be cited and used as a reference point. It won't change how most practitioners work day-to-day, but it will advance the formal-methods frontier for those who care about provable guarantees. That is sufficient for a strong venue paper, conditional on the technical bets paying off.

---

*Community Expert verification produced via adversarial team process: 3 independent assessments → 3 cross-critiques → panel synthesis → independent verifier signoff. Composite 5.9/10. Verdict: CONDITIONAL CONTINUE with binding gates at months 3, 4, and 8.*
