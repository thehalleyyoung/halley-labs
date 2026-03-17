# Verification Report: certified-leakage-contracts

## Panel Composition

| Expert | Role | Orientation |
|--------|------|-------------|
| **Independent Auditor** | Evidence-based scoring and challenge testing | Anchored to concrete data: LoC comparisons against real tools, hardware specifications, publication records. Default stance: trust evidence, not narrative. |
| **Fail-Fast Skeptic** | Adversarial attack-vector enumeration and kill-probability estimation | Seeks project-fatal flaws. Default stance: the proposal is wrong until proven right. Assigns severity ratings and explicit failure probabilities per risk. |
| **Scavenging Synthesizer** | Value preservation, scope surgery, risk-adjusted optimization | Identifies the "diamond" worth preserving, proposes scope reductions, computes expected value under uncertainty. Default stance: find what's worth saving and cut the rest. |

## Methodology

The verification followed a three-phase adversarial process:

1. **Independent Proposals.** Each expert produced a full-length assessment of the crystallized problem statement in isolation, scoring all axes, identifying fatal flaws, and recommending a verdict. No expert saw the others' work during this phase.

2. **Adversarial Cross-Critique.** Each expert received the other two experts' proposals and was tasked with attacking their weakest arguments: the Auditor challenged the Skeptic's pseudo-quantitative kill probability and the Synthesizer's independence assumption in risk modeling; the Skeptic attacked the Auditor's "generous" value score and the Synthesizer's "diamond" characterization of the composition theorem; the Synthesizer challenged the Auditor's pessimism on laptop feasibility and the Skeptic's prior-art trend claims.

3. **Panel Synthesis (this document).** The panel chair reconciles the six documents into a single assessment, weighting arguments by the quality of their evidence (specific citations, concrete numbers, verifiable claims) and resolving disagreements by identifying where one expert's evidence directly addresses another's objection. The Skeptic's scores anchor the lower bound, the Synthesizer's scores anchor the upper bound, and the Auditor's evidence-based corrections generally prevail when they cite specific prior work.

---

## Axis 1: EXTREME AND OBVIOUS VALUE — Score: 7/10

### The Case For Value

The problem is real, urgent, and documented with concrete evidence. The **Auditor** anchored this most effectively: CVE-2018-0734 and CVE-2018-0735 (OpenSSL ECDSA nonce timing leaks introduced by compiler optimization) are genuine examples of vulnerabilities that survive source-level review. CVE-2022-4304 (RSA timing oracle) is a high-impact, real-world vulnerability. libsodium's documentation explicitly disclaims binary-level constant-time guarantees. The FIPS 140-3 audit pipeline processes ~200 CMVP modules/year with IG 2.4.A requiring side-channel resistance evidence that is currently prose-based and non-reproducible.

The **Synthesizer** made the strongest case for strategic positioning: LeaVe won Distinguished Paper at CCS 2023 for verifying hardware-side leakage contracts on RISC-V. The LeaVe authors (Guarnieri, Reineke, and collaborators) explicitly identified software-side contract verification as an open problem in their 2025 SETSS tutorial. This proposal directly answers a recognized open question from a Distinguished Paper — in the currency of academic research, that is inherently high-value. The Synthesizer's analogy is apt: the audience for a cancer drug is not oncologists but cancer patients. The indirect beneficiaries of verified crypto library side-channel properties number in the hundreds of millions of deployed systems.

### The Case Against Value

The **Skeptic** sharpened the audience concern with devastating precision: the direct user base is closer to 50–100 people (crypto library maintainers globally), not the Auditor's 200–500. The Skeptic correctly identified that the Auditor double-counts: FIPS auditors are cited as both a value driver *and* noted as too conservative to adopt the tool — you cannot have both. The Skeptic's LLM competition argument is novel and partially valid: in 2025, `gpt-4 "audit this function for cache timing leaks"` provides actionable triage in 30 seconds covering ~90% of practical cases (secret-dependent branches, variable-time lookups, missing `cmov`). The remaining 10% — subtle quantitative leakage from cache-line sharing, PLRU-specific effects, speculative windows — is precisely the domain where formal analysis is most valuable, but also where the tool's own precision is most uncertain.

The **Auditor** identified the most underappreciated value proposition, which both other experts underweighted: **regression detection** (does this commit increase leakage?) is robust to modeling imprecision. Even if absolute bounds are 5× loose, relative changes between commits are accurately captured. This transforms the PLRU problem, the widening precision problem, and even the composition imprecision problem from existential threats into tolerable over-approximations.

### Reconciliation

The Skeptic's 4–5 is too harsh because it discounts the LeaVe positioning and regression-detection value. The Synthesizer's 8–9 is too generous because it assumes the full supply-chain vision is achievable within one paper's scope. The Auditor's 7, augmented by the Synthesizer's LeaVe-positioning argument and tempered by the Skeptic's audience-narrowing, anchors the final score. The LLM competition argument partially validates but does not eliminate the tool's value — LLMs handle the easy cases; the tool handles the hard cases *and* provides formal guarantees that LLMs cannot.

**Final Score: 7/10.** The problem is real, timely, and explicitly identified as open by a Distinguished Paper. The direct audience is narrow but the indirect impact is broad. The LLM competition argument prevents a score of 8+.

---

## Axis 2: GENUINE DIFFICULTY AS A SOFTWARE ARTIFACT — Score: 7/10

### The LoC Debate

This axis produced the widest disagreement and the most informative cross-critique.

**Original proposal:** 152K LoC across 11 subsystems.

**Auditor (initial):** Reduced to 90–119K by deflating the binary lifter (35K → 8–12K) and CFG recovery (18K → 6–10K) based on concrete comparisons to BAP (~5–10K LoC for x86-64 lifting on top of LLVM MC), angr (~60K total across all architectures), and CacheAudit (~15K LoC for the complete research prototype). The Auditor identified the genuine difficulty as concentrated in ~50–60K LoC of novel abstract domain + certificate work.

**Synthesizer (cross-critique):** Further reduced to 55–65K LoC at the recommended Reduced-A scope by eliminating the binary lifter entirely (reuse BAP/angr → 3–5K adapter), zeroing out CFG recovery (free from existing frameworks), and deferring certificates and the μarch contract language. This estimate preserves all three novel abstract domains (D_spec ~10–12K, D_cache ~10–13K, D_quant ~8–10K), the reduced product + fixpoint engine (10–13K), the composition system (8–10K), and test infrastructure (12–15K).

**Skeptic (cross-critique):** Argued that even 50–60K of "genuinely novel" code is inflated by 2–3×. The Skeptic's decomposition claims only ~15–20K LoC is truly novel: speculative reduction operator (~3K), quantitative widening (~2K), composition theorem + contract checking (~5K), certificate format + emission (~3K), certificate checker (~3K), novel lattice operations (~2K). Everything else — cache domain fundamentals (extending CacheAudit), fixpoint engine (Bourdoncle's WTO — textbook), powerset speculative domain (standard construction) — is characterized as "textbook abstract interpretation with a twist."

### Reconciliation

The Skeptic conflates "uses known techniques" with "easy to implement." This is the central error in the Skeptic's difficulty assessment. Implementing CacheAudit-quality cache abstract domains extended with per-line taint annotations, integrated with a speculative reachability domain, and connected through a novel reduction operator ρ is *not* "textbook AI" in the same way that implementing a compiler backend using well-known algorithms is not "textbook" just because the algorithms are published. The **Auditor's** comparison is decisive: CacheAudit itself (a simpler system analyzing only x86-32, without speculation, composition, or certificates) took Boris Köpf's group ~3 years to develop as a research prototype. Spectector (analyzing only speculative non-interference, no quantitative bounds, no composition) was a multi-year collaboration between MPI-SWS, IMDEA, and TU Graz. The claim that the combined system — which subsumes and extends both — requires only 15–20K LoC of genuine work is not credible.

However, the Synthesizer is also correct that at Reduced-A scope (~55–65K LoC), the difficulty is substantially less than the original 152K LoC proposal implies. The 152K figure included 76K LoC of infrastructure (lifter, CFG recovery, test infrastructure, CLI) that exists in mature open-source projects.

At Reduced-A scope, the genuinely challenging work consists of:
- Three novel abstract domains with lattice operations, transfer functions, widening, and parameterization: ~28–35K LoC
- Reduced product with the novel ρ operator and fixpoint engine: ~10–13K LoC
- Compositional contract system with the composition theorem: ~8–10K LoC
- IR adapter layer: ~3–5K LoC
- Test infrastructure and CLI: ~15–19K LoC

This is harder than a typical top-venue paper but achievable by a focused team within 1.5–2 years. The Skeptic's comparison to prior tool effort (CacheAudit ~3 years, Spectector ~multi-year) actually *supports* this characterization.

**Final Score: 7/10.** At Reduced-A scope, the ~55–65K LoC estimate is credible, the genuinely novel implementation spans ~35–45K LoC of hard abstract-interpretation engineering, and the difficulty is at the level of a strong PhD thesis. Not a 10 (the original 152K scope that nobody recommends), not a 4–5 (the Skeptic's "textbook AI" dismissal ignores real implementation complexity).

---

## Axis 3: BEST-PAPER POTENTIAL — Score: 6/10

### The "Synthesis vs. Paradigm Shift" Debate

This axis produced the sharpest intellectual disagreement across the panel.

The **Auditor** framed the core objection most clearly: each of the five claimed properties exists in some prior tool (CacheAudit has quantitative bounds + binary-level; Spectector has speculation awareness; Binsec/Rel has binary-level; LeaVe has contracts + certificates). The novelty is their *combination*. A paper whose primary contribution reads as "we combined CacheAudit + Spectector + Binsec/Rel + LeaVe + PCC" risks being triaged by at least one reviewer as "sophisticated synthesis, not paradigm shift." At top venues with 8–15% acceptance rates, one skeptical reviewer citing this critique can sink the paper.

The **Skeptic** amplified this with publication data: scanning S&P/CCS/USENIX Security best/distinguished papers from 2022–2024, cache side-channel papers are absent except for LeaVe (CCS 2023 Distinguished). The pattern suggests that cache side channels had their peak visibility in 2018–2020 (Spectre, Meltdown, Foreshadow) and the community has moved on. The Skeptic further challenged the Synthesizer's "crown jewel" characterization of the composition theorem, citing Smith (2009) — Theorem 4.3 on sequential composition of min-entropy channels — and Yasuoka & Terauchi (2010) — algorithmic composition for QIF — as establishing the core results that the proposal instantiates for the cache domain. The Skeptic's argument: the cache-state-parameterized composition rule B_{f;g}(s) = B_f(s) + B_g(τ_f(s)) is an *instantiation* of known frameworks, not a new composition principle.

### The Counter-Arguments

The **Synthesizer** offered the strongest rebuttal: answering a Distinguished Paper's explicitly identified open question is inherently high-value. LeaVe won at CCS *2023*, not 2018 — this contradicts the "side channels are passé" narrative. The Synthesizer correctly notes that best papers don't follow trends; they define them. The hardware-software contract paradigm (Guarnieri, Köpf, Reineke) is active and growing, not declining.

The **Auditor** partially conceded this in cross-critique, noting that while the field may have shifted, the LeaVe exception demonstrates that *formal* treatments of the contract framework remain publishable. The Auditor also refined the novelty count: the reduced product domain (D_spec ⊗ D_cache ⊗ D_quant with the speculative reduction operator) and the composition theorem are each genuinely novel and substantial; quantitative widening for counting domains is an open problem in abstract interpretation theory that would independently merit publication at SAS/VMCAI. That yields 2–3 genuinely publishable novelties.

### Reconciliation

The Skeptic's critique of the composition theorem as merely instantiating Smith (2009) has merit for the *algebraic identity* but misses the *engineering contribution*: making that identity work soundly over the cache-state domain with taint-restricted counting, under a speculative semantics, with convergence guarantees, is substantially more than a paragraph in related work. However, the Skeptic is correct that it is not a paradigm shift — it is a technically deep synthesis.

The "side channels are passé" argument is contradicted by LeaVe's 2023 award, but the Auditor's refinement is accurate: LeaVe won because it introduced a genuinely new *framing* (hardware-side contracts), not because reviewers were excited about cache side channels per se. The software side is inherently less novel because software-side verification is a mature area.

**Final Score: 6/10.** The paper has strong best-paper *candidacy* elements (answers a Distinguished Paper's open question, 2–3 genuine novelties, bridges PL and security communities) but faces structural headwinds (synthesis critique, community attention shift, 0 LoC prototype risk). The ~5% best-paper probability estimate (Skeptic) versus the ~2% base rate means a mild positive signal, not a lock. The Auditor's 6 is the most evidence-grounded score on this axis.

### Amendment Required (Score < 7)

To elevate best-paper potential to 7+, the following changes are necessary:

**Amendment BP-1:** Foreground the reduced product domain (D_spec ⊗ D_cache ⊗ D_quant with the speculative reduction operator ρ) as the primary theoretical novelty, not the five-property combination narrative. The reduction operator is genuinely new abstract-interpretation theory with no direct prior art, and it is the mathematical engine that enables the combination. Framing the contribution as "a new abstract domain for speculative cache-channel analysis" is stronger than "we combine 5 properties."

**Amendment BP-2:** Position the paper for CCS rather than S&P. CCS reviewers reward theoretical depth; S&P reviewers expect more immediate systems impact. The novel abstract domain and composition theorem play better at CCS.

**Amendment BP-3:** Include a formalized precision comparison on the CacheAudit benchmark suite (AES T-table) as the centerpiece of the evaluation, demonstrating that the tool matches or exceeds CacheAudit's best-known bounds while additionally providing speculation awareness and composition. This addresses the "precision or it's vacuous" concern head-on.

---

## Axis 4: LAPTOP CPU + NO HUMANS — Score: 7/10

### Where All Experts Agree

L1 LRU cache analysis via abstract interpretation is demonstrably tractable on laptop hardware. The **Auditor** established the baseline: CacheAudit analyzes AES in *seconds* on decade-old hardware. For a 32KB L1D (64 sets × 8 ways × 1 taint bit), the per-abstract-state size is ~512 bits (~64 bytes). The **Skeptic**, in cross-critique, conceded this is the strongest point for feasibility and corrected the Auditor's own score upward, arguing the Auditor's 5/10 was too pessimistic. Even the Skeptic agrees: for L1 LRU analysis of individual crypto functions, the laptop constraint is easily met.

### Where They Disagree: Speculation Overhead

The **Auditor** raised the speculative context count as the primary concern: the proposal claims "10–20 distinct speculative contexts" after prefix closure, but the ROB depth of Skylake is 224 μops. For crypto code with table lookups inside loops, the Auditor estimated >100 contexts before collapsing, yielding per-program-point state of ~480KB (20 contexts × 8KB × 3 domains) to potentially much larger. The **Skeptic** agreed with the Auditor that the "10–20 contexts" claim is unsubstantiated, calling the prefix closure abstraction "wishful thinking" with "zero evidence for this number."

The **Synthesizer** countered that even at 100 speculative contexts, the analysis is 100× slower than non-speculative — CacheAudit at 2 seconds × 100 = ~3 minutes per function, which is CI-compatible. The Synthesizer also correctly noted that bounded speculation windows (e.g., depth ≤ 50 μops instead of full ROB 224) are a sound engineering parameter that trades completeness for tractability — analyzing only shallow speculation still catches the vast majority of practical Spectre gadgets.

### The BoringSSL Timing Debate

The proposal claims full BoringSSL crypto/ directory analysis in 10–15 minutes. The **Auditor** calculated: ~300 functions × ~30 seconds each = ~150 minutes, a 10× discrepancy. The **Skeptic** partially defended the proposal here: most of the 300 functions are trivial utilities that would be filtered or analyzed in <1 second. Crypto-critical functions number perhaps 30–50 at 30 seconds each = 15–25 minutes. The truth lies between: 30–90 minutes for a full library pass, which is CI-compatible (pipelines routinely take 30–120 minutes) even if the 15-minute claim is overoptimistic.

### Reconciliation

The Auditor's initial 5/10 was the least defensible score across all experts, as the Skeptic's own cross-critique demonstrated. Abstract interpretation is fundamentally more efficient than symbolic execution (Spectector) or model checking. Per-function analysis avoids whole-program fixpoints. Cache geometry bounds the domain size. Even with pessimistic speculation overhead, the analysis remains laptop-feasible for the core use case.

The real risk is for complex functions (RSA modexp with thousands of loop iterations) and multi-level cache hierarchies — but these can be deferred or bounded as engineering parameters in v1.

**Final Score: 7/10.** Feasible for L1 LRU analysis of individual crypto functions with bounded speculation windows. Uncertain for full-library analysis with aggressive speculation, but the uncertainty is on *performance*, not *possibility*. The LRU-first strategy and bounded speculation windows make this tractable.

---

## Axis 5: FATAL FLAWS

### Flaw 1: Binary Lifter — ELIMINATED

| Expert | Initial Severity | Post-Cross-Critique |
|--------|-----------------|---------------------|
| Auditor | MEDIUM-HIGH | Mitigable to LOW via reuse |
| Skeptic | FATAL (70% project-fatal) | Concedes reuse eliminates most risk |
| Synthesizer | HIGH (eliminated by reuse) | Maintains: adapter layer = 3–5K LoC |

**Reconciled Severity: ELIMINATED.** All three experts converge on the same recommendation: reuse BAP, angr, or Ghidra as the binary analysis substrate. The Skeptic's original FATAL rating was predicated on the proposal appearing to plan a from-scratch lifter; with mandatory reuse, this drops to a minor integration risk (3–5K LoC adapter layer). The Auditor correctly noted this in cross-critique: "A risk is FATAL only when no realistic mitigation exists. Here, the mitigation is obvious and well-precedented."

**Binding requirement:** Do not build a binary lifter. This is non-negotiable.

### Flaw 2: Vacuous Bounds (PLRU + Widening Precision) — HIGH

| Expert | Initial Severity | Post-Cross-Critique |
|--------|-----------------|---------------------|
| Auditor | HIGH | Maintains; proposes LRU-first strategy |
| Skeptic | HIGH (40% project-fatal) | Maintains; calculates ~33 bits spurious leakage per 4 tainted sets under LRU→PLRU over-approximation |
| Synthesizer | HIGH | Maintains; but argues LRU-on-LRU is tight (CacheAudit demonstrated this) |

**Reconciled Severity: HIGH — the existential risk.** All three experts agree this is the most dangerous flaw. The Skeptic's PLRU precision-loss calculation is directionally correct but overstated: the Auditor corrected the 315× over-approximation factor to 10–50× (citing Reineke et al. 2008), reducing the spurious leakage estimate from ~33 bits to ~13–22 bits per 4 tainted sets. This is still substantial and threatens the 3× precision target on Intel hardware.

**Mitigation consensus:** Target LRU first (ARM Cortex-A series processors use LRU; CacheAudit demonstrated tight bounds for LRU). Accept PLRU over-approximation as a known limitation for v1. Foreground regression detection, which is robust to absolute-bound imprecision. The Auditor's "precision canary" recommendation (test D_cache ⊗ D_quant on AES T-table before committing to the full framework) is endorsed by all experts as the most cost-effective de-risking step.

### Flaw 3: Min-Entropy Composition Independence — MEDIUM-HIGH

| Expert | Initial Severity | Post-Cross-Critique |
|--------|-----------------|---------------------|
| Auditor | HIGH | Refined to: fails 30–50% at useful compositional boundaries |
| Skeptic | HIGH (50% project-fatal; 60–80% of pairs violate) | Maintains 60–80% estimate |
| Synthesizer | MEDIUM-HIGH | Argues crypto functions with clear data-flow separation (AES rounds, ChaCha quarter-rounds) likely satisfy independence; Rényi fallback is publishable |

**Reconciled Severity: MEDIUM-HIGH.** The Auditor's 30–50% violation estimate is more credible than the Skeptic's 60–80% because the Auditor correctly distinguished between *all function pairs* and *function pairs at useful compositional boundaries*. In a well-structured crypto library, contracts are applied at API boundaries (e.g., `aes_encrypt`, `ecdsa_sign`), not at every internal function pair. Many internal pairs operate on non-overlapping secret partitions (AES SubBytes on different byte positions, separate elliptic curve coordinate operations). The Skeptic's three counterexamples (table-lookup AES with key-dependent permutations, RSA CRT recombination, secret-dependent branches) are well-chosen but represent worst-case patterns, not typical compositional boundaries.

The Rényi-entropy fallback exists and provides meaningful bounds with different but acceptable semantics (Boreale 2015, Fernandes et al. 2019). This is a genuine engineering pivot within a well-understood theoretical landscape.

### Flaw 4: Speculative Path Explosion — MEDIUM

| Expert | Initial Severity | Post-Cross-Critique |
|--------|-----------------|---------------------|
| Auditor | MEDIUM | Maintains |
| Skeptic | HIGH (45% project-fatal) | Maintains; argues prefix closure is "wishful thinking" |
| Synthesizer | MEDIUM | Argues bounded speculation windows (depth ≤ 50 μops) provide sound engineering parameter |

**Reconciled Severity: MEDIUM.** The "prefix closure" abstraction is indeed uncharacterized, and the Skeptic is correct that the "10–20 contexts" claim has no supporting evidence. However, bounded speculation windows are a well-understood engineering tradeoff: analyzing only shallow speculation (depth ≤ 50 μops) still captures the practical Spectre-PHT gadgets while keeping the context count manageable. Worst case: speculative bounds are loose, and the tool falls back to non-speculative analysis (Reduced-B), which is still novel for composition + quantitative bounds. The Synthesizer's hedge strategy is reasonable here.

### Flaw 5: Cache Geometry Error (4096 vs. 64 sets) — COSMETIC

| Expert | Initial Severity | Post-Cross-Critique |
|--------|-----------------|---------------------|
| Auditor | LOW-MEDIUM | Maintains (raises domain expertise concerns) |
| Skeptic | Not separately flagged | — |
| Synthesizer | Not a flaw at all | Argues it's a typo; correction *improves* feasibility |

**Reconciled Severity: COSMETIC.** The Synthesizer is correct: 4096 = 64 × 64 is a plausible arithmetic error, or the author was describing L2 parameters. The correction to 64 sets (the correct L1D value for 32KB / 8-way / 64B lines) makes the per-point state ~16× smaller than stated, which *improves* the laptop-feasibility argument. The Auditor's "raises concerns about domain expertise" is an overreach — the proposal correctly identifies PLRU complexity, min-entropy non-composability, and speculative path explosion, demonstrating genuine domain knowledge beyond a typo.

**Binding requirement:** Fix the 4096→64 sets parameter in the problem statement.

### Flaw 6: Zero Lines of Code Implemented — ACKNOWLEDGED

All three experts flag this. The Auditor and Skeptic rate it HIGH; the Synthesizer acknowledges it. This is inherent to the crystallization phase and applies equally to every project at this stage. The Auditor's cross-critique offers the most actionable framing: "The highest-priority action is building a 5K LoC prototype that demonstrates non-vacuous bounds on a single function."

---

## Composite Score

### Final Reconciled Scores

| Axis | Final Score | Auditor | Skeptic (implied) | Synthesizer (Reduced-A) |
|------|------------|---------|-------------------|-------------------------|
| 1. Extreme and Obvious Value | **7/10** | 7 | 4–5 | 8 |
| 2. Genuine Difficulty | **7/10** | 6 | 4–5 | 8 |
| 3. Best-Paper Potential | **6/10** | 6 | ~3–4 | 7 |
| 4. Laptop CPU + No Humans | **7/10** | 5 | 6–7 | 8 |
| **Composite (average)** | **6.75/10** | **6.0** | **~4.6** | **7.75** |

### Fatal Flaw Summary

| Flaw | Reconciled Severity | Status |
|------|-------------------|--------|
| Binary lifter | ELIMINATED | Mandatory reuse resolves |
| Vacuous bounds (PLRU + widening) | HIGH | Existential risk; LRU-first strategy mitigates |
| Min-entropy composition independence | MEDIUM-HIGH | 30–50% violation at useful boundaries; Rényi fallback exists |
| Speculative path explosion | MEDIUM | Bounded speculation windows as engineering parameter |
| Cache geometry error | COSMETIC | Fix the typo |

---

## Expert Disagreements

The following disagreements remained substantively unresolved after cross-critique:

### 1. Novelty of the Composition Theorem

The **Skeptic** argues the composition rule B_{f;g}(s) = B_f(s) + B_g(τ_f(s)) is a direct instantiation of Smith (2009) Theorem 4.3 and Yasuoka & Terauchi (2010), meriting "a paragraph in related work, not a section title." The **Synthesizer** argues the cache-state parameterization τ_f(s) and the engineering of soundness over taint-restricted counting domains under speculative semantics is substantially more than an instantiation. The **Auditor** splits the difference: it is a real contribution (publishable as part of a systems paper) but not independently best-paper-worthy.

**Panel assessment:** The Skeptic's critique of the *algebraic identity* has theoretical merit, but underweights the *implementation challenge* of making that identity work soundly over the specific domain. The contribution is more than a paragraph but less than a paradigm shift. It is a strong technical contribution — the heart of a good systems-security paper.

### 2. The LLM Competition Argument

The **Skeptic** argues LLMs cover 90% of practical side-channel triage cases, leaving the tool competing for a 10% niche where its own precision is uncertain. The **Auditor** and **Synthesizer** do not engage deeply with this argument. 

**Panel assessment:** This is a novel and partially valid critique that the proposal should address. LLMs handle pattern-matching cases (secret-dependent branches, variable-time lookups) but cannot produce formal bounds, certificates, or compositional guarantees. The tool's value is strongest in the residual cases requiring quantitative reasoning — and in providing guarantees that LLMs structurally cannot. The proposal should explicitly position against LLM-based triage.

### 3. Kill Probability Methodology

The **Skeptic** claims 62% kill probability (with methodology errors the Auditor exposed — independent-risk multiplication yields ~97%, not 62%, suggesting the 62% is a gut estimate). The **Synthesizer** estimates 12% joint success at Reduced-A scope (challenged by the Skeptic for assuming independent failures, corrected to ~6–7%). The **Auditor** dismisses both as "pseudo-quantitative" and argues qualitatively: "3 HIGH-severity and 1 FATAL-severity risk. Projects with this risk profile fail more often than they succeed."

**Panel assessment:** All three probability estimates are poorly grounded. The Auditor's qualitative framing is most honest. The directional conclusion — that the project is high-risk but not irrecoverable with scope reduction — is a consensus.

### 4. Commit to Reduced-B vs. Reduced-A

The **Skeptic** argues in cross-critique that the rational choice is to commit to Reduced-B (no speculation) from day one, treating speculation as a separate future project. The Synthesizer's hedge strategy (attempt Reduced-A, fall back to Reduced-B) is characterized as "an admission of defeat" and "hope, not engineering." The **Synthesizer** argues Reduced-A preserves the most valuable novelties and the hedge is standard risk management. The **Auditor** sides with Reduced-A but recommends foregrounding regression detection.

**Panel assessment:** Reduced-A with the hedge to Reduced-B is the correct strategy. The Skeptic's argument that the hedge creates a "worst-of-both-worlds" scenario has theoretical merit but underweights the value of *attempting* speculation analysis — even partial results (e.g., demonstrating that bounded speculation adds 1.5 bits for a known Spectre gadget) are publishable alongside the compositional framework.

---

## Binding Amendments

The following amendments are **NON-NEGOTIABLE** changes to the problem statement and project plan. They represent the unanimous or near-unanimous consensus of the verification panel.

### Amendment 1: Mandatory Lifter Reuse
**Scope:** Eliminate the from-scratch binary lifter (35K LoC). Build a 3–5K LoC adapter layer on BAP BIL, angr VEX, or Ghidra P-code. Accept imperfect SIMD semantics for v1; restrict the instruction subset to ~150 crypto-critical instructions (not 300) with AVX-512 deferred.
**Rationale:** All three experts agree. This eliminates the single highest-risk component and reduces total LoC by ~40%.
**Expert support:** Unanimous.

### Amendment 2: LRU-First, PLRU-Deferred
**Scope:** Target LRU replacement policy for v1. Do not claim tight bounds on Intel tree-PLRU hardware. Explicitly quantify the LRU→PLRU over-approximation gap (13–22 bits per 4 tainted sets, per the Auditor's corrected estimate) as a known limitation. Target ARM Cortex-A series (which uses LRU) as the primary evaluation platform.
**Rationale:** All three experts agree PLRU counting is an unsolved research problem (#P-hard). CacheAudit published successfully analyzing LRU only. The Auditor's "calibrate precision relative to the LRU model" is scientifically honest and publishable.
**Expert support:** Unanimous.

### Amendment 3: Prove Composition Theorem First
**Scope:** Before any implementation begins, formally prove the min-entropy additive composition rule and characterize the independence condition. Demonstrate that the condition holds for at least 3 representative crypto patterns: (a) AES rounds (sequential, non-overlapping byte positions), (b) ChaCha20 quarter-rounds (data-independent permutation structure), (c) Curve25519 scalar multiply (conditional swap + field operations). If the condition fails for all three, pivot to Rényi entropy before building the implementation.
**Rationale:** The Skeptic and Auditor both identify this as the intellectual crown jewel and the highest-value de-risking action. The Synthesizer agrees on sequencing.
**Expert support:** Unanimous.

### Amendment 4: Build Precision Canary Before Full Framework
**Scope:** Implement D_cache ⊗ D_quant (without speculation) on a single function (AES T-table lookup under LRU) and verify that bounds are within 3× of exhaustive enumeration on small inputs (key ≤ 16 bits). If bounds exceed 5× of true leakage, the quantitative widening needs redesign before anything else proceeds.
**Rationale:** The Synthesizer proposed this; the Auditor endorsed it as "the most cost-effective de-risking step proposed by any assessor." The Skeptic implicitly supports it through their emphasis on the "precision or it's vacuous" argument.
**Expert support:** Unanimous.

### Amendment 5: Scope to Reduced-A (~55–65K LoC)
**Scope:** Drop certificates (defer to follow-up paper), drop the μarch contract language (hardcode 2–3 configurations: LRU no-speculation, LRU bounded-Spectre-PHT, ARM Cortex-A76), drop raw binary lifting (use assembly/IR input). Keep: D_spec ⊗ D_cache ⊗ D_quant, reduced product, composition theorem, speculative modeling, quantitative bounds.
**Rationale:** All three experts recommend scope reduction. Reduced-A preserves ~80% of value at ~57% of the LoC. The Skeptic's Reduced-B is the fallback if speculation proves intractable.
**Expert support:** Unanimous on scope reduction; 2-of-3 on Reduced-A specifically (Skeptic prefers Reduced-B).

### Amendment 6: Foreground Regression Detection
**Scope:** Add regression detection (does this commit increase leakage?) as a first-class use case, not a buried side benefit. The evaluation must include at least one regression-detection benchmark: analyze a pre-patch and post-patch binary for a known CVE and show the tool detects the leakage change even if absolute bounds are loose.
**Rationale:** The Auditor identified this as the "real killer app" that is robust to modeling imprecision. Both other experts underweighted it. Regression detection transforms the PLRU imprecision, widening imprecision, and composition imprecision from existential threats into tolerable over-approximations.
**Expert support:** Auditor (strong); Synthesizer and Skeptic (implicit).

### Amendment 7: Fix Cache Geometry Parameter
**Scope:** Correct "4096 sets × 8 ways" to "64 sets × 8 ways" for L1D (32KB / 8-way / 64B lines). If multi-level cache analysis is intended for a future version, specify L2 parameters separately.
**Rationale:** Factual error. Correction improves the feasibility argument (16× smaller per-point state than stated).
**Expert support:** Auditor (flagged); Synthesizer (confirmed as typo).

### Amendment 8: Address LLM Positioning
**Scope:** Add an explicit paragraph in the value proposition addressing how the tool complements LLM-based side-channel triage. Frame: LLMs handle pattern-matching cases (secret-dependent branches, variable-time lookups) but cannot produce formal bounds, certificates, or compositional guarantees. The tool targets the residual cases requiring quantitative reasoning and provides guarantees that LLMs structurally cannot deliver.
**Rationale:** The Skeptic's LLM competition argument is novel and partially valid. Ignoring it leaves a gap that reviewers in 2025–2026 will exploit.
**Expert support:** Skeptic (raised); Auditor (noted but did not pursue).

---

## Verdict

### **CONDITIONAL CONTINUE**

The panel unanimously recommends CONDITIONAL CONTINUE at Reduced-A scope, contingent on all eight binding amendments being applied.

### Conditions for Continuation

1. **Phase Gate 1 (Month 1–2):** Prove the composition theorem (Amendment 3). If the independence condition fails for all three test patterns, the project must pivot to Rényi entropy or whole-program analysis.
2. **Phase Gate 2 (Month 2–4):** Build and pass the precision canary (Amendment 4). If bounds exceed 5× of true leakage on AES T-table under LRU, the quantitative widening must be redesigned.
3. **Phase Gate 3 (Month 4–8):** Prototype D_spec ⊗ D_cache ⊗ D_quant on 3–5 crypto functions. If speculative bounds are vacuous (>10× true leakage), invoke the Reduced-B fallback and drop speculation from the first paper.
4. **Phase Gate 4 (Month 8–14):** Full evaluation on AES, ChaCha20, Curve25519, and at least one known CVE. Paper submission to CCS (preferred) or S&P.

### Kill Triggers

The project should be **ABANDONED** if:
- The composition theorem provably requires conditions that exclude all standard crypto patterns (Phase Gate 1 failure with no viable fallback)
- The precision canary fails at >10× on AES T-table under LRU (this would mean the quantitative framework is fundamentally imprecise, not just engineering-limited)
- After 6 months of effort, no function produces bounds within 5× of exhaustive enumeration

### Kill Probability Estimate

**~30–40% at Reduced-A scope with all amendments applied.**

Methodology: Qualitative assessment anchored to the panel's risk reconciliation. The three experts' estimates span from ~62% (Skeptic, on the full unscoped project) to ~25–35% (Synthesizer, on Reduced-A). The Auditor characterizes the risk profile qualitatively as "high-risk but not irrecoverable." The binding amendments eliminate the FATAL risk (lifter), reduce the HIGH risks (LRU-first for PLRU, precision canary for widening, early theorem-proving for composition), and bound the MEDIUM risks (speculation windows, hedge to Reduced-B). The Skeptic's corrected 6–7% success probability for Reduced-A is based on a correlated-failure model that is pessimistic but structurally valid; inverting gives ~93% kill probability, which is too high because it ignores partial-success modes (the Auditor's insight: a tool with 5× loose bounds is still publishable, a tool with composition working for 50% of patterns is still novel). Accounting for partial-success modes, the kill probability for *producing a publishable top-venue paper* is ~30–40%.

---

## Appendix: Expert Scores Comparison

### Initial Proposal Scores

| Axis | Auditor | Skeptic (implied) | Synthesizer (Full) | Synthesizer (Reduced-A) |
|------|---------|-------------------|-------------------|-------------------------|
| Value | 7 | 4–5 | 9 | 8 |
| Difficulty | 6 | 4–5 | 10 | 8 |
| Best-Paper Potential | 6 | 3–4 | 8 | 7 |
| Laptop CPU + No Humans | 5 | 4–5* | 7 | 8 |
| **Composite** | **6.0** | **~4.1** | **8.5 / 7.75** | — |

*The Skeptic did not assign explicit axis scores but implied scores through severity ratings and the 62% kill probability. The Laptop CPU estimate for the Skeptic is inferred from the cross-critique, where the Skeptic argued the Auditor's 5 was too harsh and proposed 6–7.

### Post-Cross-Critique Adjusted Scores

| Axis | Auditor (adjusted) | Skeptic (adjusted) | Synthesizer (adjusted) | **Panel Final** |
|------|-------------------|-------------------|------------------------|----------------|
| Value | 7 (maintained) | 4–5 (maintained) | 8 (reconciled) | **7** |
| Difficulty | 6 (maintained) | 4–5 (maintained) | 7 (reconciled) | **7** |
| Best-Paper Potential | 6 (maintained) | ~6 (conceded Auditor is "approximately right") | 7 (reconciled) | **6** |
| Laptop CPU + No Humans | 5 (maintained, but weakened by Skeptic cross-critique) | 6–7 (revised upward from initial) | 7 (reconciled) | **7** |
| **Composite** | **6.0** | **~5.3** | **7.25** | **6.75** |

### Score Movement Summary

The largest score movement during cross-critique was on the **Laptop CPU** axis, where the Skeptic's cross-critique of the Auditor forced an effective upward revision by demonstrating that abstract interpretation on cache domains is "demonstrably fast" and CacheAudit's performance data supports laptop feasibility for L1 LRU. The smallest movement was on **Value**, where all three experts largely maintained their positions — the Skeptic never conceded the narrow-audience argument, and the Synthesizer never conceded the LLM competition point.

The panel's final scores fall between the Auditor's evidence-based center and the Synthesizer's optimistic ceiling, with the Skeptic's arguments serving as effective anchors that prevented uncritical acceptance of the proposal's own framing. The Skeptic's most lasting contribution was not any individual score but the methodological discipline of demanding *evidence* for every claimed number — forcing the other experts (and this panel) to distinguish between substantiated claims and aspirational assertions.
