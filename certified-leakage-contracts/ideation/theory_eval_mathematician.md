# Deep Mathematician Verification Report: certified-leakage-contracts

## Verdict: CONDITIONAL CONTINUE

**Composite Score: V6/D5/BP3/L7/F5 = 5.2/10**

This project has ONE genuinely novel theorem (the reduction operator ρ), a solid engineering plan, and a credible publication narrative — but zero proofs exist after the theory stage, and the math is entirely unvalidated. Continue only if sketch proofs and a precision canary are produced within 8 weeks.

---

## Panel Composition

| Expert | Role | Orientation |
|--------|------|-------------|
| **Independent Auditor** | Evidence-based scoring and challenge testing | Trust evidence, not narrative. Anchor every claim to concrete data. |
| **Fail-Fast Skeptic** | Adversarial attack-vector enumeration | The proposal is wrong until proven right. Assign explicit failure probabilities. |
| **Scavenging Synthesizer** | Value preservation and scope surgery | Find the diamond worth preserving. Cut the rest. |
| **Independent Verifier** | Final signoff | Last line of defense. Ruthlessly honest. |

## Methodology

The verification followed a four-phase adversarial process:

1. **Independent Proposals.** Each expert produced a full-length assessment in isolation, scoring all axes, identifying fatal flaws, and recommending a verdict. No expert saw the others' work.

2. **Adversarial Cross-Critique.** The Panel Chair identified six key disagreements and forced point-by-point resolution: novelty of ρ (30% vs 60%), genuinely novel LoC (15K vs 60K), theory_bytes=0 interpretation, best-paper potential, kill probability, and the "CacheAudit-next" framing.

3. **Panel Synthesis.** Disagreements reconciled by evidence quality. The Skeptic's scores anchor the lower bound, the Synthesizer's scores anchor the upper bound, the Auditor's evidence-based corrections prevail where specific prior work is cited.

4. **Independent Verification.** A final verifier assessed internal consistency, endorsed or overrode the panel's scores, and produced the binding verdict.

---

## The theory_bytes=0 Problem

### What happened

State.json records `theory_bytes: 0` and `theory_score: null` for proposal_00. The `proposals/proposal_00/theory/` directory is empty. The project-level `theory/` directory contains `approach.json` (24KB) and `empirical_proposal.md` (40KB) — architectural planning and evaluation design, not mathematical proofs.

### What this means

**The theory stage produced zero proofs.** Not "proofs in the wrong directory" — no `.lean`, `.v`, `.tex`, or sketch-proof files exist anywhere in the repository. The 64KB of theory-stage output consists of:
- A JSON architecture specification with pseudocode and complexity estimates
- A detailed evaluation plan with 7 research questions, 11 benchmarks, and 5 baselines

These are excellent planning artifacts. They are not mathematics. Every claimed proof (A1–A9) exists only as an English-language description of what *would* be proved.

### Panel assessment

The Skeptic argued this is devastating; the Synthesizer argued it reflects a framework mismatch (AI proofs are short and not appropriate for theory-stage formalization); the Auditor suggested a pipeline bookkeeping bug. **The Skeptic is closest to right.** The theory stage's purpose was to validate mathematical feasibility. It validated planning feasibility instead. The proofs are likely short (ρ monotonicity is ~2–3 pages via finite lattice argument), but *no one has written even a sketch*. This adds ~10% unvalidated-math risk beyond what the prior panel estimated.

**Conclusion:** Not fatal, but a significant risk factor. Proofs must be produced in the first 4 weeks of the next stage — this is the panel's hardest binding condition.

---

## Axis 1: EXTREME AND OBVIOUS VALUE — Score: 6/10

### What the math buys

The mathematical contributions serve two distinct value propositions:

1. **Regression detection** (the "killer app"): Requires sound abstract interpretation but does NOT require the crown-jewel ρ operator or the composition theorem. A modernized CacheAudit for x86-64 with a diff mode would deliver ~70% of regression-detection value in ~15–20K LoC. The full mathematical framework is overkill for this use case.

2. **Compositional contracts** (the research contribution): Requires the composition theorem (A7), independence condition (A8), and ρ for precision. This is where the math is load-bearing — without ρ, the direct product produces vacuously imprecise bounds that cannot support meaningful contracts.

### Audience

The Skeptic performed the most rigorous audience count: ~35–50 crypto library maintainers globally who would directly use this tool. The Auditor's 50–100 double-counts. The indirect beneficiaries (deployed systems) number in hundreds of millions but don't distinguish this from any security research. **Direct audience: ~35–50 people.**

### LLM competition

In 2025–2026, GPT-4/5 handles ~90% of practical side-channel triage (secret-dependent branches, variable-time lookups, missing `cmov`). The tool targets the residual 10% requiring quantitative reasoning plus formal guarantees LLMs cannot produce. The LLM trajectory argument (Skeptic: the residual shrinks faster than this project ships) is partially valid but underestimates the structural gap — LLMs cannot produce compositional formal bounds by construction.

### LeaVe positioning

The strongest value argument: LeaVe won CCS 2023 Distinguished Paper for hardware-side leakage contracts. The LeaVe authors explicitly identified software-side contract verification as an open question. This proposal directly answers that question. In academic currency, answering a Distinguished Paper's open question is inherently high-value.

### Score justification

Real problem, strong positioning, but: microscopic audience, regression detection doesn't need 80% of what's proposed, LLM competition constrains the ceiling. The math is load-bearing for the *research* value (compositional contracts), not for the *practical* value (regression detection). This tension between what the math enables and what users actually need prevents a score above 6.

---

## Axis 2: GENUINE SOFTWARE DIFFICULTY — Score: 5/10

### The novelty deflation

This axis produced the widest disagreement across the panel. The core question: how much genuinely NEW math is required?

**The Skeptic's deflation was the most rigorous and is substantially correct.** The Skeptic went component-by-component:

| Result | Claimed | Deflated | Surviving Novel Theorems |
|--------|---------|----------|--------------------------|
| A5 (ρ reduction operator) | DEEP NOVEL, 60% new | ~40% new | **1–2 novel lemmas** (monotonicity at merge points, termination bounded by cache geometry) |
| A4 (D_quant taint-restricted counting) | NOVEL, 40% new | ~25% new | **~0.5 theorem** (taint restriction is a one-paragraph soundness argument) |
| A1 (speculative collecting semantics) | NOVEL, 30% new | ~30% new | **~0.5 theorem** (quantitative observation function extension) |
| A2 (γ-only soundness for D_spec) | NOVEL | Routine application | ~0 (standard γ-only argument on new domain) |
| A7 (composition rule) | INSTANTIATION | INSTANTIATION | 0 (Smith 2009 Thm 4.3 instantiation) |
| A8 (independence condition) | APPLIED, 25% new | APPLIED | 0 (verification on specific patterns, not a new theorem) |
| A6 (quantitative widening) | DEFERRED | DEFERRED | N/A for v1 |
| A3 (D_cache with taint) | INSTANTIATION | INSTANTIATION | 0 |
| A9 (fixpoint convergence) | INSTANTIATION | TEXTBOOK | 0 |

**Total surviving genuinely novel theorems: 2–3.** The crown jewel (ρ, A5) contributes 1–2 of these. Everything else is adaptation, instantiation, or textbook application.

### The genuinely novel LoC debate

| Expert | Estimate | Methodology |
|--------|:--------:|-------------|
| Proposal | 50–60K | Everything not copy-pasted from CacheAudit |
| Auditor | 50–60K | Implicitly accepts proposal framing |
| Synthesizer | ~35–45K | Code requiring novel lattice operations |
| Panel Chair | 25–35K | Split the difference |
| Skeptic | 15–20K | Only code implementing genuinely new math |
| Verifier | 15–20K | Agrees with Skeptic's decomposition |

**Reconciled: ~20–30K genuinely novel LoC within a ~65–73K total artifact.** The Skeptic's decomposition (fail_fast_evaluation.md lines 125–137) is the most detailed: ρ operator ~2K, D_spec novel ~4K, D_cache novel ~3K, D_quant novel ~4K, fixpoint novel ~2K, composition novel ~3K. Everything else (CLI, tests, lifter adapter, CacheAudit-derived code) is engineering.

### Is the math the reason this is hard?

**Partially.** ρ's three-way interaction (speculative infeasibility → cache taint pruning → capacity zeroing) is genuinely subtle. The ρ-at-merge-points problem (join operations re-introducing pruned states) has no template in the literature. CacheAudit (~15K LoC, ~3 years) and Spectector (multi-year, multi-institution) demonstrate that simpler systems in this space require significant effort. But: the Skeptic is right that after deflation, this is ~1.5× CacheAudit in genuine novelty, not the paradigm-shifting "first abstract domain simultaneously modeling speculative reachability and cache channel capacity" that the proposal claims.

### Score justification

The Verifier concurred with the Skeptic: ONE genuinely novel theorem (ρ) plus competent engineering. The difficulty is real (implementing CacheAudit-quality domains correctly is hard even when the theory is known) but concentrated in ~20K LoC of genuinely new work. Score 5 reflects: harder than a typical paper, but not paradigm-defining after honest deflation. The depth_check panel's 7 was before the Skeptic's detailed LoC decomposition.

---

## Axis 3: BEST-PAPER POTENTIAL — Score: 3/10

### The synthesis critique

Every panel member acknowledged this structural headwind. The paper's narrative is: "we combined CacheAudit (quantitative bounds) + Spectector (speculative analysis) + Smith 2009 (composition) + CacheAudit/Köpf (counting) via a new glue operator (ρ)." Each component exists in prior work. The novelty is the combination. Best papers introduce paradigms; this paper instantiates one someone else introduced (Guarnieri, Köpf, Reineke's contract framework).

### The "answers a Distinguished Paper" argument

Valid but overrated. LeaVe won because it introduced hardware-side contracts — a *new framing*. Answering an open question within that framing is inherently second-wave. The software side is less novel because software verification is mature. Reviewers will correctly identify this as "the expected follow-up."

### Zero proofs, zero code

With theory_bytes=0 and impl_loc=0, the project is at maximum distance from a submittable paper. The base rate for a 16–22 month project producing a best paper from zero starting point is <1%. The LeaVe positioning and ρ novelty push this to ~2–3%, but not higher.

### Score justification

The Skeptic's 3/10 is correct. The Auditor and Synthesizer's 5/10 was too generous — they counted "above base rate" as competitive. A 3/10 = "~2–3% probability, above the ~1.5% base rate, but not competitive." The Verifier agreed: zero proofs + zero code + synthesis narrative + declining subfield attention = 3/10.

---

## Axis 4: LAPTOP CPU + NO HUMANS — Score: 7/10

### Where all experts agree

L1 LRU cache analysis via abstract interpretation is demonstrably tractable on laptop hardware. CacheAudit analyzes AES in *seconds* on decade-old hardware. For a 32KB L1D (64 sets × 8 ways × 1 taint bit), the per-abstract-state size is ~128 bytes. Domain size is bounded by cache geometry (fixed constants). Per-function analysis avoids whole-program fixpoints. Crypto loops have fixed iteration counts. Ground truth computation (exhaustive enumeration for ≤16-bit keys) is trivially parallelizable.

The speculative domain multiplies state by the speculative context count, but bounded W ≤ 50 keeps this tractable. Even at 100 contexts × 128 bytes × 3 domains ≈ 37.5KB per program point, this fits comfortably in L2 cache.

### Risks

Curve25519 memory (255 iterations × speculative contexts) could push to ~3GB without hash-consing. RSA-2048 modexp is the scalability stress test and may timeout. Full BoringSSL library: ~30–90 minutes (CI-compatible). These are performance concerns, not feasibility blockers.

### Score justification

No GPU, no cluster, no human annotation required. All baselines (CacheAudit, Spectector, cachegrind) are single-machine tools. Zero hidden requirements. The Auditor scored this 8/10; we apply a mild correction for the Curve25519/RSA uncertainty.

---

## Axis 5: FEASIBILITY — Score: 5/10

### The probability chain

| Gate | Pass Probability | Cumulative |
|------|:----------------:|:----------:|
| PG1: Composition theorem (month 1–2) | ~70% | 70% |
| PG2: Precision canary within 3× (month 2–4) | ~75% | 53% |
| PG3: Speculative bounds within 10× (month 4–8) | ~55% | 29% |
| PG4: Full evaluation + paper (month 8–14) | ~80% | 23% |

**Joint probability of the proposed CCS paper: ~23%.** With Reduced-B fallback (drop speculation): ~42%. With further degradation (direct product, Rényi): ~55–60% of producing *something* publishable.

### The fallback cascade

Each fallback strips novelty:
- **Full Reduced-A** (proposed): CCS-targeted, ρ + composition + speculation. ~23% success.
- **Reduced-B** (drop speculation): D_cache ⊗ D_quant + composition + ρ-lite. ESORICS/CSF-targeted. ~42% success.
- **Reduced-C** (drop ρ): CacheAudit modernized + composition. Workshop/tool-track targeted. ~60% success.
- **Reduced-D** (regression only): CacheAudit ported to x86-64 + diff mode. ~75% success. Not novel.

### The timeline risk

16–22 months for one researcher producing 65–73K LoC from zero is aggressive. Months 4–8 require ~35K LoC in 4 months = 8.75K LoC/month of novel abstract-interpretation code. The 3–4 month buffer is consumed by a single gate-triggered redesign. Two redesigns = project overrun.

### Score justification

The honest probability of producing a CCS paper is ~23–30%. The probability of producing *any* publication is ~60%. The phase-gate structure is the strongest feature — it ensures failure is detected early and fallbacks are activated rationally. Score 5 reflects: probable that *something* publishable emerges, but the proposed paper is the minority outcome.

---

## Fatal Flaws

### Flaw 1: Unvalidated Mathematics — Severity: HIGH

**All experts agree.** theory_bytes=0 means every mathematical claim is aspirational, not validated. The ρ monotonicity/termination proof has never been sketched. The composition theorem's independence condition has never been formally stated. The speculative collecting semantics has never been written down formally. The entire project rests on mathematics that has never been attempted in any form.

**The Skeptic's asymmetric-payoff argument is decisive:** the cost of validating the math (4 weeks for sketch proofs) is trivial compared to discovering it fails at month 12. Proofs must come first.

### Flaw 2: ρ Precision May Be Empirically Hollow — Severity: HIGH

The ρ operator can be proved correct (monotone, sound, terminating) relatively quickly. The hard question is whether it provides non-trivial precision improvement over the direct product on real crypto code. The ρ-at-merge-points problem (CFG joins re-introduce pruned states) could silently undo most of ρ's precision gains. If ρ-ratio ≤ 1.0 on >50% of benchmarks, the paper's crown jewel is decorative.

**Mitigation:** Phase Gate 3 tests this, but by month 4–8. The Verifier added a binding condition: before implementation, produce a pencil-and-paper worked example proving ρ strictly tightens bounds on a specific 4–6 instruction program.

### Flaw 3: Composition Independence Fragility — Severity: MEDIUM-HIGH

Min-entropy does not satisfy the chain rule. The additive composition requires independence that fails for:
- T-table AES with related subkeys (cache-set aliasing creates correlations)
- RSA CRT (CRT recombination creates feedback between p-branch and q-branch)
- ECDSA (bit-by-bit processing means each step depends on all prior secret bits)

Estimated violation rate: ~30–50% at useful compositional boundaries. The Rényi fallback exists but is honestly characterized as "insurance against abandonment" with 2–5× additional precision loss.

### Flaw 4: Speculative Bounds May Be Vacuous — Severity: MEDIUM

30% probability that speculation analysis adds >10× imprecision, forcing Reduced-B fallback. The "10–20 speculative contexts" claim has been dropped (correctly), and context count is now treated as empirical. But: if speculative bounds are vacuous, the paper loses its most distinctive positioning (the LeaVe complement story).

### No flaw is independently fatal

Each flaw has a fallback. The risk is *correlated degradation*: if ρ is hollow AND independence needs large corrections AND speculative bounds are vacuous, the residual contribution is "CacheAudit for x86-64" — a workshop paper, not CCS.

---

## Novelty Assessment: The Deep Mathematician's View

### What genuinely new math exists in this proposal?

After deflation by four independent assessors across two verification rounds:

| Result | Honest Classification | Math Quality | Load-Bearing? |
|--------|----------------------|:------------:|:-------------:|
| **A5 (ρ operator)** | **NOVEL (~40% new)** | **B+** | **YES — the reason the artifact is hard AND the reason it delivers value** |
| A4 (taint-restricted counting) | ADAPTED (~25% new) | C+ | Partially — tightens bounds meaningfully |
| A1 (speculative semantics) | ADAPTED (~30% new) | B- | YES — enables the Spectre story |
| A7 (composition rule) | INSTANTIATION | C | YES — but the math is Smith 2009 |
| A8 (independence characterization) | APPLIED | C | YES — but verification, not theory |
| A6 (widening) | DEFERRED | N/A | Not in v1 |
| A2, A3, A9 | INSTANTIATION/TEXTBOOK | D+ | Necessary but not novel |

**Total genuinely novel mathematical content: ONE novel theorem (ρ) with ~2 supporting lemmas.** The rest is instantiation, adaptation, and competent engineering. This is the novelty portfolio of a solid PhD paper — sufficient for CCS acceptance if ρ delivers empirically, insufficient for best-paper.

### Is the math load-bearing?

**ρ (A5) is genuinely load-bearing.** Without ρ, the direct product D_spec × D_cache × D_quant yields vacuously imprecise bounds — speculation taints everything and the counting domain has no way to discharge infeasible paths. ρ is the mechanism that converts a trivial "put three known domains next to each other" into something that produces tight bounds. This is not ornamental math. ρ is the reason the artifact is hard to build (three-way interaction with domain-specific monotonicity proof) and the reason it delivers value (precision enables meaningful contracts).

**A7 (composition) is load-bearing but not novel.** The composition theorem is what makes contracts composable — the defining feature of the system — but the algebraic identity is Smith (2009) Thm 4.3. The domain-specific instantiation requires non-trivial engineering (threading cache-state transformers, characterizing independence) but produces no new mathematical insight.

### The Skeptic's "ONE theorem" characterization

The Verifier agreed with the Skeptic: after honest deflation, one genuinely novel theorem remains. **This is harsh but defensible.** The counter-argument (Auditor/Synthesizer) is that domain-specific proofs in unexplored territory count as novel even if the proof technique is classical. Both sides have merit. The reconciliation: ρ is one *construction* requiring ~2–3 genuinely novel *proof obligations* (monotonicity at merge points, termination bounded by cache geometry, precision advantage over direct product). Calling it "one theorem" or "three lemmas in service of one construction" is a framing choice, not a factual disagreement.

---

## Panel Disagreements — Resolved

### 1. Novelty of ρ: 30% vs 40% vs 60%

**Resolved: ~40% genuinely new.** The *idea* of constraint propagation in a reduced product is ~30% new (classical technique). The *proof obligations* for this specific three-domain combination are ~55% new (no template for the merge-point interaction). Weighted: ~40%. The Skeptic's 30% treats "follows a known pattern" as "easy"; the Auditor/Synthesizer's 60% conflates "novel domain" with "novel technique."

### 2. Genuinely novel LoC: 15K vs 35K vs 60K

**Resolved: ~20–30K.** The Skeptic's detailed decomposition (ρ ~2K, D_spec ~4K, D_cache ~3K, D_quant ~4K, fixpoint ~2K, composition ~3K = ~18K) is the most rigorous but slightly too narrow. Adding domain-specific transfer functions and lattice operations that are novel even if the pattern is known: ~25–30K. The proposal's 50–60K is indefensible and should be corrected for the paper.

### 3. theory_bytes=0

**Resolved: The Skeptic is right on fact, the Synthesizer is right on interpretation.** Zero proofs exist. This is not a pipeline bug. But the proofs are likely short (~5 pages total for ρ + composition). The theory stage produced planning instead of proofs because the proofs were not the existential risk — precision is. Still, sketch proofs must be produced before implementation.

### 4. Best-paper potential: 3 vs 4 vs 5

**Resolved: 3/10.** The Verifier sided with the Skeptic. Zero proofs + zero code + synthesis narrative + 16-month horizon from zero = ~2–3% best-paper probability. The LeaVe positioning provides a mild positive signal but "answering an open question" < "posing a new question."

### 5. Kill probability

**Resolved: ~35–40% kill for CCS paper; ~25–30% kill for any publication.** The Auditor's multiplicative chain (23% success at Reduced-A) is structurally correct but ignores partial-success modes. The Synthesizer's 60% success rate underweights theory_bytes=0 risk. Reconciled: ~60–65% chance of publishing *something*; ~35–40% chance at CCS specifically.

---

## Binding Conditions for CONTINUE

| # | Condition | Deadline | Kill Trigger |
|---|-----------|----------|--------------|
| BC-1 | **Sketch proofs** of ρ monotonicity/termination (A5) and composition soundness (A7/E.2.1). Paper-style, 2–3 pages each, with key lemmas formally stated. | 4 weeks | ABANDON if no proofs in 4 weeks |
| BC-2 | **Pencil-and-paper worked example** showing ρ *strictly* tightens bounds over direct product on a specific 4–6 instruction program fragment. | 4 weeks | ABANDON if no strict improvement exists on any example |
| BC-3 | **Independence validation** on ≥3 of 4 crypto patterns (AES distinct subkeys, AES related subkeys, ChaCha20, Curve25519). | 6 weeks | Pivot to Rényi if <2 non-trivial patterns pass |
| BC-4 | **Precision canary**: D_cache ⊗ D_quant on AES T-table within 3× of exhaustive enumeration (small keys ≤ 16 bits). ~3–4K LoC. | 8 weeks | REDESIGN if >5×; ABANDON if >10× after redesign |
| BC-5 | **No from-scratch binary lifter.** 3–5K LoC adapter on BAP/angr/Ghidra only. | Immediate | Non-negotiable |
| BC-6 | **Honest framing.** Paper positions as "novel reduced-product domain for speculative cache analysis with compositional contracts." Not "50–60K novel LoC" or "5 independent novelties." | Paper draft | Reviewers will deflate dishonest framing |

---

## Expected Value Assessment

### Cost
~16–22 person-months for one senior PL/security researcher.

### Value (conditioned on phase-gate discipline)

| Outcome | Probability | Value |
|---------|:-----------:|:-----:|
| CCS strong accept | ~15% | 7.0 |
| CCS borderline accept | ~20% | 5.0 |
| ESORICS/CSF via Reduced-B | ~15% | 3.0 |
| Workshop/tool paper | ~15% | 1.5 |
| Kill (nothing publishable) | ~35% | 0.0 |

**E[value] = 0.15×7 + 0.20×5 + 0.15×3 + 0.15×1.5 + 0.35×0 = 1.05 + 1.00 + 0.45 + 0.225 = 2.725/10**

The conditional expected value *given survival past PG2* (month 4) rises to ~4.5/10, which justifies the remaining 12–16 months of effort. The phase-gate structure ensures that the first 8 weeks (which produces proofs + precision canary) is the go/no-go decision point. Cost of validating: ~2 person-months. Cost of discovering the math fails at month 12: ~12 person-months wasted. The Skeptic's asymmetric-payoff argument is decisive.

---

## Kill Triggers

The project should be **ABANDONED** if:
1. Sketch proofs of ρ cannot be produced in 4 weeks (BC-1 fails)
2. No pencil-and-paper example shows ρ strictly tightens bounds (BC-2 fails)
3. Precision canary exceeds 10× on AES T-table after redesign (BC-4 kills)
4. After 6 months of effort, no function produces bounds within 5× of exhaustive enumeration
5. The composition theorem provably requires conditions excluding all standard crypto patterns AND Rényi yields vacuous bounds

---

## Final Reconciled Scores

| Axis | Score | Prior Panel | Change | Justification |
|------|:-----:|:-----------:|:------:|---------------|
| **Value (V)** | **6** | 7 | ↓1 | Real problem, strong LeaVe positioning, but microscopic direct audience (~35–50), regression detection doesn't need 80% of the math, LLM competition constrains ceiling |
| **Difficulty (D)** | **5** | 7 | ↓2 | After honest deflation: ONE novel theorem (ρ), ~20–30K genuinely novel LoC. The Skeptic's decomposition is substantively correct. ~1.5× CacheAudit novelty, not paradigm-defining |
| **Best-Paper (BP)** | **3** | 6 | ↓3 | Zero proofs, zero code, synthesis headwind, ~2–3% probability. The Verifier agreed. One novel theorem + instantiations is a solid venue paper, not a best paper |
| **Laptop (L)** | **7** | 7 | = | All experts agree: AI on bounded cache geometry is tractable on laptop CPU. CacheAudit performance data is conclusive. Least contested axis |
| **Feasibility (F)** | **5** | 7 | ↓2 | ~35% CCS probability; ~65% any-publication probability. theory_bytes=0 adds risk. 16–22 month timeline assumes zero setbacks. Phase gates help but sequential gating reduces joint probability |
| **Composite** | **5.2** | 6.75 | ↓1.55 | — |

---

## Verdict

### **CONDITIONAL CONTINUE** — Confidence: 65%

**Rationale:** The diamond (ρ) is real. The LeaVe positioning is strong. The regression-detection use case is genuinely valuable. The phase-gate structure ensures early failure detection. The expected value is positive *conditional on phase-gate discipline*.

**What this is:** A solid CCS submission attempt with ~35% success probability, backed by fallbacks to ESORICS/CSF at ~55% and to any publication at ~65%. One genuinely novel mathematical contribution (ρ) that fills a real gap in the abstract interpretation literature. A useful tool for crypto library maintainers *if* the precision canary passes.

**What this is NOT:** A best-paper candidate (3/10). A paradigm shift (synthesis, not creation). A project with validated mathematics (theory_bytes=0). Easy (35–40% kill probability).

**The binding conditions are non-negotiable.** If BC-1 (sketch proofs, 4 weeks) and BC-2 (worked example of ρ tightening, 4 weeks) fail, ABANDON without hesitation. The project has already consumed its entire theory budget producing plans instead of proofs. It cannot afford to repeat this pattern with its implementation budget.

---

## Appendix: Expert Score Comparison

| Axis | Auditor | Skeptic | Synthesizer | Panel Chair | Verifier | **Final** |
|------|:-------:|:-------:|:-----------:|:-----------:|:--------:|:---------:|
| Value | 6 | 5 | 7 | 6 | 6 | **6** |
| Difficulty | 7 | 5 | 7 | 6 | 5 | **5** |
| Best-Paper | 5 | 3 | 5 | 4 | 3 | **3** |
| Laptop | 8 | — | — | 7 | 7 | **7** |
| Feasibility | 5 | — | — | 5 | 5 | **5** |
| **Composite** | **6.2** | **~4.3** | **~6.3** | **5.6** | **5.2** | **5.2** |

### Score Movement from Ideation Verification

| Axis | Ideation Panel | This Panel | Movement | Driver |
|------|:--------------:|:----------:|:--------:|--------|
| Value | 7 | 6 | ↓1 | Skeptic's audience deflation + LLM competition |
| Difficulty | 7 | 5 | ↓2 | Skeptic's LoC decomposition (15–20K novel, not 50–60K) |
| Best-Paper | 6 | 3 | ↓3 | theory_bytes=0, zero code, synthesis headwind, Verifier confirmation |
| Laptop | 7 | 7 | = | Consensus |
| Feasibility | 7 | 5 | ↓2 | Multiplicative gate-passing probability, theory_bytes=0 risk |

**Key insight:** The ideation panel assessed the *plan*. This panel assessed the *evidence of execution*. The plan was good (6.75). The execution evidence (theory_bytes=0, impl_loc=0) forces a significant downward correction.
