# Post-Theory Skeptic Verification: certified-leakage-contracts (proposal_00)

## Verdict: CONDITIONAL CONTINUE at Reduced-C Scope

**Composite Score: V5 / D6 / BP4 / Laptop8 / F4 = 5.4/10** (down from ideation-stage 6.75/10)

Kill probability: **45–50%** at full proposed scope; **25–30%** at Reduced-C.

---

## Panel Composition

| Expert | Role | Verdict |
|--------|------|---------|
| **Independent Auditor** | Evidence-based scoring, challenge testing | CONDITIONAL CONTINUE (composite 6.0) |
| **Fail-Fast Skeptic** | Aggressively reject under-supported claims | ABANDON (composite 4.4) |
| **Scavenging Synthesizer** | Salvage value, scope surgery, risk-adjusted optimization | CONDITIONAL CONTINUE at Reduced-C (composite 6.2) |

## Methodology

Three-phase adversarial process:
1. **Independent Proposals.** Each expert assessed the post-theory state in isolation, scoring all axes and recommending a verdict.
2. **Adversarial Cross-Critique.** Each expert challenged the other two's weakest arguments. The Auditor challenged the Skeptic's prior art citations; the Skeptic attacked the Auditor's "generous" continuation and the Synthesizer's optimistic probabilities; the Synthesizer challenged the Skeptic's ABANDON verdict and the Auditor's full-scope recommendation.
3. **Panel Synthesis.** The chair reconciled all arguments, verified contested claims via web search, and produced reconciled scores.

---

## The Critical Finding: theory_bytes = 0

### What the Theory Phase Was Supposed to Produce

The ideation-stage verification panel issued a **CONDITIONAL CONTINUE** with unanimously agreed conditions. The single highest-priority condition was:

> **Amendment 3: Prove Composition Theorem First.** Before any implementation begins, formally prove the min-entropy additive composition rule and characterize the independence condition. Demonstrate that the condition holds for at least 3 representative crypto patterns. Expert support: **Unanimous.**

The theory phase was designed to execute Phase Gate 1: paper proofs of the composition theorem, independence condition characterization on 4 crypto patterns, and formal specification of the reduction operator ρ.

### What Was Actually Delivered

| Artifact | Size | Content Type |
|----------|------|-------------|
| `theory/approach.json` | 23,973 bytes | Programmatic specification: algorithms, domain definitions, pseudocode, phase gates, risk assessment |
| `theory/empirical_proposal.md` | 39,702 bytes | Evaluation plan: 7 RQs with falsification criteria, 11 benchmarks, 5 baselines, 4 ablations, threats to validity |
| **Total theory output** | **63,675 bytes** | **Planning/specification documents — zero mathematical proofs** |

State.json:
```json
"theory_bytes": 0,
"impl_loc": 0,
"status": "theory_complete"
```

### Assessment

**The theory_bytes = 0 is substantively accurate.** The 64KB of output consists of planning and specification documents, not mathematical theory. The approach.json is a JSON-formatted restatement of the ideation-stage final_approach.md with added pseudocode. The empirical_proposal.md is a detailed experiment design for a tool that does not exist.

**What is missing:**
- Zero formal proofs of any kind (no Lean, no Coq, no LaTeX derivations, no pen-and-paper proof sketches)
- The composition theorem B_{f;g}(s) ≤ B_f(s) + B_g(τ_f(s)) is **stated** but never **proved**
- The γ-only soundness for D_spec (A2) is **described** but never **derived**
- The ρ correctness argument is one sentence: "monotone decreasing on finite lattice, therefore terminates"
- No independence condition verification for any crypto pattern

**Mitigating context:** State.json timestamps show the theory phase ran for approximately 2 hours (created_at: 06:27:19Z to updated_at: 08:22:26Z). This is insufficient time to prove a composition theorem for min-entropy over cache-state domains. The failure may reflect a pipeline timing constraint rather than mathematical inability — but the result is the same: Phase Gate 1 was not attempted.

**The planning artifacts have genuine value.** The empirical_proposal.md is publication-quality evaluation design — 7 research questions with falsification criteria, ground-truth methodology with exhaustive enumeration for small keys and KSG Monte Carlo for large keys, and a comprehensive threats-to-validity analysis. The approach.json contains actionable pseudocode for the reduction operator, transfer functions, and fixpoint engine. These artifacts save an estimated 6–8 weeks of design work in future phases.

**But planning ≠ theory.** The project has now completed ideation (5 documents), depth check, 2 verification reports, and "theory" — producing ~250KB of natural-language planning and zero executable artifacts.

---

## Prior Art Challenge: Verified Novelty Erosion

### The Skeptic's Citations — Verified via Web Search

The Fail-Fast Skeptic cited 6 recent papers as eroding novelty. The panel chair verified these via web search:

| Citation | Status | Relevance to This Proposal |
|----------|--------|---------------------------|
| **Mitchell & Wang, ECOOP 2025** — "Quantifying Cache Side-Channel Leakage by Refining Set-Based Abstractions" (LIPIcs Vol 333, pp 22:1–22:28, DOI: 10.4230/LIPIcs.ECOOP.2025.22) | **✅ CONFIRMED REAL** | **HIGH.** Directly advances CacheAudit-style quantitative cache analysis with refined transfer functions and finite powerset construction. Improves precision on crypto benchmarks. Artifact available at github.com/jlmitche23/ecoop25CacheQuantification. |
| **BINSEC v0.11 (Jan 2026)** — QRSE plugin (Quantitative Robust Symbolic Execution) + Binsec/Haunted (speculative analysis) | **✅ CONFIRMED REAL** | **MEDIUM.** BINSEC v0.11 exists with QRSE. However, QRSE measures "quantitative robustness" (vulnerability reachability counting), NOT cache channel capacity in bits. Binsec/Haunted handles speculative constant-time but not quantitative leakage bounds. The overlap is partial, not complete. |
| **SCAFinder (TIFS 2024)** — Zhang et al., formal verification of cache hardware designs | **✅ REAL** | **LOW.** Hardware-side cache RTL verification. Competes with LeaVe, not this proposal. |
| **SpecLFB (USENIX Security 2024)** — Hardware defense in SonicBOOM RISC-V | **✅ REAL** | **NONE.** Hardware mitigation mechanism, not analysis tool. |
| **Contract Shadow Logic (arXiv 2024)** — RTL verification for secure speculation | **✅ REAL** | **LOW.** Hardware-side. Actually strengthens the case for software-side tools. |
| **VeriCache (TDSC 2025)** — Verified fine-grained partitioned cache | **✅ REAL** | **LOW.** Hardware cache architecture. |

### Reconciled Novelty Assessment

**2 of 6 papers have genuine relevance** to software-side analysis:

1. **Mitchell & Wang (ECOOP 2025)** is the most significant. It advances CacheAudit-style quantitative cache analysis — exactly the territory D_cache ⊗ D_quant occupies. If Mitchell & Wang's precision matches this proposal's 3× target on AES T-table, the precision canary becomes "matching ECOOP 2025 published results" rather than "advancing the state of the art." Key differentiators this proposal retains: (a) compositional contracts (Mitchell & Wang appear monolithic), (b) speculative awareness (not in Mitchell & Wang), (c) binary-level analysis via existing lifter.

2. **BINSEC v0.11** now combines quantitative analysis (QRSE) with speculative constant-time checking (Haunted). However, QRSE measures robustness, not channel capacity in bits, and uses symbolic execution (doesn't scale to full crypto functions). The proposal's abstract-interpretation scalability and compositional contracts remain differentiators.

**Revised novelty erosion: MEDIUM.** The four-property combination (quantitative + speculative + binary + compositional) remains novel, but two properties individually have closer prior art than the proposal acknowledges. The proposal MUST cite and differentiate from Mitchell & Wang.

---

## Three Pillars Assessment

### Pillar 1: Extreme and Obvious Value — 5/10

| Expert | Score | Key Argument |
|--------|:-----:|-------------|
| Auditor | 6 | Real problem, LeaVe positioning, but value hypothetical without execution |
| Skeptic | 4 | 20–50 person audience, LLM competition, zero execution evidence |
| Synthesizer | 6 | Regression detection is the killer app, LeaVe's open question stands |

**Reconciled: 5/10.** The problem is real and the LeaVe positioning is genuine. However:
- Direct audience: ~30–50 crypto library maintainers globally. The Skeptic's narrowing is evidenced.
- Mitchell & Wang (ECOOP 2025) erodes the quantitative cache analysis novelty more than the ideation panel anticipated.
- LLM-based triage handles ~90% of practical cases (the proposal's own admission). The tool competes for a 10% niche where its own precision is most uncertain.
- CacheAudit (published 2013/2015) has approximately zero FIPS 140-3 adoption despite being available for over a decade — the claim that auditors will use this tool requires evidence beyond aspiration.
- Zero execution evidence after the theory phase means value remains entirely hypothetical.

### Pillar 2: Genuine Software Difficulty — 6/10

| Expert | Score | Key Argument |
|--------|:-----:|-------------|
| Auditor | 7 | CacheAudit took 3 years; this subsumes CacheAudit + Spectector |
| Skeptic | 6 | Real but inflated; LoC estimate not credible; crown jewel ρ is 3–5K LoC |
| Synthesizer | 7 | Abstract interpretation engineering is hard regardless of scale |

**Reconciled: 6/10.** The difficulty is real and concentrated in the right places (ρ at ~3–5K LoC, taint-restricted counting, composition with independence). However:
- After honest deflation, genuinely novel LoC is ~15–20K, not 50–60K. The remainder is adaptation, integration, testing, and infrastructure.
- The novelty is in combination, not paradigm creation. Each component has clear prior art (CacheAudit, Spectector, Smith 2009, Cousot & Cousot).
- Mitchell & Wang (ECOOP 2025) demonstrated that refined quantitative cache analysis is achievable — reducing the perceived difficulty of D_cache ⊗ D_quant.
- At Reduced-C scope (~12–18K LoC), the difficulty is honest ~6/10: a strong research contribution, not a paradigm shift.

### Pillar 3: Best-Paper Potential — 4/10

| Expert | Score | Key Argument |
|--------|:-----:|-------------|
| Auditor | 5 | Zero proofs/code; best papers require exceptional execution |
| Skeptic | 3 | Synthesis headwind, ECOOP erosion, timeline risk, zero output |
| Synthesizer | 4 | Reduced-C targets SAS/VMCAI, not CCS best-paper |

**Reconciled: 4/10.** The prior ideation panel's 6/10 was already the lower bound of their range (Skeptic wanted 5, Synthesizer wanted 7). Post-theory, two factors reduce further:

1. **Zero proofs, zero code.** Best papers require exceptional execution, and the project has demonstrated zero execution capability. The pre-theory assessment credited the "genuinely new abstract-interpretation theory" of ρ and the composition theorem — neither has been proved.

2. **Mitchell & Wang competition.** The quantitative precision story is partially told by ECOOP 2025. If this proposal's precision results are similar, it reads as an incremental extension, not a breakthrough.

3. **Scope reduction.** At Reduced-C scope, the paper is "CacheAudit revival + composition for x86-64" — a solid SAS/VMCAI paper but not a CCS best-paper candidate. Best-paper probability at SAS/VMCAI: ~10–15%. At CCS (if full scope recovered): ~3%.

### Pillar 4: Laptop CPU + No Humans — 8/10

| Expert | Score | Key Argument |
|--------|:-----:|-------------|
| Auditor | 7 | Tractable but speculation overhead is uncertain |
| Skeptic | 7 | The one clearly viable axis |
| Synthesizer | 8 | Reduced-C removes speculation overhead entirely |

**Reconciled: 8/10.** Universal agreement. Abstract interpretation on bounded cache domains (64 sets × 8 ways) is polynomial by construction. CacheAudit analyzed AES in seconds on decade-old hardware. At Reduced-C scope (no speculation domain), the context explosion problem vanishes entirely. Fully automated evaluation with exhaustive enumeration for small keys and Monte Carlo for large keys — zero human involvement.

### Pillar 5: Feasibility — 4/10

| Expert | Score | Key Argument |
|--------|:-----:|-------------|
| Auditor | 5 | Theoretically sound but zero execution evidence |
| Skeptic | 2 | Theory phase failure is project-critical signal; timeline impossible |
| Synthesizer | 6 | Reduced-C is achievable in 6–12 months |

**Reconciled: 4/10.** This is the axis with the widest expert disagreement, reflecting fundamentally different interpretations of the theory_bytes=0 signal.

The timeline critique is validated: 50–60K novel LoC for one researcher in 16–22 months requires 10–25× the productivity of CacheAudit and Spectector teams (both of which had teams and took multiple years). Even with the blueprint advantage (detailed pseudocode, evaluation plan), corrected estimates yield 45–99 person-months for full scope — far exceeding the claimed 16–22 months.

At Reduced-C scope (~12–18K LoC), feasibility improves substantially. Realistic estimate: 6–12 months for one researcher, with the composition theorem proof as the gating risk (60–65% success probability).

The zero-proofs outcome from the theory phase is a genuine red flag. The project has passed through 5+ pipeline stages producing ~250KB of planning and zero executable artifacts. This pattern ("planning as procrastination" per the Skeptic) may or may not continue, but it cannot be ignored.

---

## Fatal Flaws

### Flaw 1: Zero Mathematical Output After Theory Phase — HIGH

**Severity: HIGH.** Phase Gate 1 (composition theorem, months 1–2) was the unanimously agreed first deliverable. It was not attempted. The conditions for the prior CONDITIONAL CONTINUE have not been met.

**Mitigation:** The 30-day proof-sketch deadline (Phase Gate 0, see Binding Amendments) forces engagement with the mathematics. If no proof sketch exists after 30 focused days, the project should be abandoned.

### Flaw 2: Timeline Infeasibility at Full Scope — HIGH

**Severity: HIGH.** Corrected productivity analysis:
- Genuinely novel LoC: 15–20K at 300–500 LoC/person-month (with blueprint) = 30–67 person-months
- Adaptation/integration: 25–30K at 1,500–2,500 LoC/person-month = 10–20 person-months
- Test/CLI infrastructure: 17–24K at 2,000–3,000 LoC/person-month = 6–12 person-months
- **Total: 46–99 person-months (3.8–8.3 years for one researcher)**

The claimed 16–22 month timeline is not credible by any historical comparison. CacheAudit (~15K LoC, simpler scope) took ~3 years with a team.

**Mitigation:** Reduce scope to Reduced-C (~12–18K LoC, 6–12 months). This is achievable for one researcher.

### Flaw 3: Novelty Erosion from Mitchell & Wang (ECOOP 2025) — MEDIUM-HIGH

**Severity: MEDIUM-HIGH.** Mitchell & Wang directly advance CacheAudit-style quantitative cache analysis with improved precision — exactly the territory D_cache ⊗ D_quant occupies. The proposal does not cite or position against this work.

**Mitigation:** The proposal must differentiate on compositional contracts (which Mitchell & Wang do not have). Reduced-C's primary contribution becomes "the first tool enabling cross-function leakage bound composition for x86-64 crypto binaries" rather than "improved quantitative cache analysis."

### Flaw 4: Independence Condition Untested — MEDIUM-HIGH

**Severity: MEDIUM-HIGH.** Min-entropy does not compose (established theorem). The additive rule requires an independence condition that has never been empirically validated. Phase Gate 1 was supposed to do exactly this validation. The Rényi fallback is honestly characterized as "insurance, not a path to strong results" with 2–5× additional precision loss.

**Mitigation:** Phase Gate 0 (30-day proof sketches) tests this. If independence fails for all non-trivial patterns AND Rényi is vacuous, ABANDON.

### Flaw 5: Vacuous Bounds on Real Hardware (PLRU) — MEDIUM

**Severity: MEDIUM.** Intel/AMD CPUs use pseudo-LRU, not LRU. The proposal admits 10–50× over-approximation. Regression detection partially mitigates this.

**Mitigation:** LRU-first strategy (ARM Cortex-A as primary platform). Regression detection as primary use case. Unchanged from ideation stage.

---

## Expert Disagreements

### 1. Is theory_bytes=0 Fatal?

**Skeptic:** Yes — "a project that produced zero proofs in its dedicated theory phase will not subsequently produce those proofs while simultaneously building 75K LoC." Evidence of inability. FATAL.

**Auditor:** No — recoverable within 30 days. The 2-hour pipeline window is insufficient for proofs. The planning quality demonstrates domain competence.

**Synthesizer:** No — the artifacts are valuable engineering blueprints worth ~$25–35K in researcher-time savings. The error was labeling it "theory" when it was "architectural planning."

**Panel resolution:** The Auditor and Synthesizer are correct that a 2-hour pipeline window is not a fair test of mathematical capability. The Skeptic is correct that the pattern (5 stages of planning, zero execution) is a red flag. **Resolution: impose a 30-day execution-forcing deadline (Phase Gate 0) as a genuine test. If this fails, the Skeptic's ABANDON becomes the consensus.**

### 2. Should the Scope be Full or Reduced?

**Auditor:** Full scope with hard deadlines (30 + 60 days).

**Skeptic:** ABANDON entirely — no scope is viable.

**Synthesizer:** Reduced-C (~12–18K LoC, SAS/VMCAI target).

**Panel resolution:** Full scope is infeasible for one researcher in 16–22 months (timeline analysis confirms). The Auditor's CONDITIONAL CONTINUE at full scope with 30+60-day deadlines would effectively kill the project before full scope is reached — making the Auditor and Synthesizer's positions closer than they appear. **Resolution: Reduced-C scope with upgrade path.** The full four-property vision is preserved as aspirational but not committed.

### 3. Best-Paper Probability

**Auditor:** 5–8%. **Skeptic:** ~2%. **Synthesizer:** 10–15% at SAS/VMCAI.

**Panel resolution:** At Reduced-C targeting SAS/VMCAI: ~10%. At full scope targeting CCS: ~3%. The Synthesizer's venue-adjusted estimate is most realistic.

---

## Verdict: CONDITIONAL CONTINUE at Reduced-C Scope

### Scope Definition

**Reduced-C: D_cache ⊗ D_quant + compositional leakage contracts.** No speculation domain. No reduction operator ρ.

- **LoC:** ~12–18K
- **Timeline:** 6–12 months
- **Target venue:** SAS 2027 or VMCAI 2027
- **Primary contribution:** The first tool enabling cross-function quantitative leakage bound composition for x86-64 crypto binaries
- **Upgrade path:** If Phase Gates 0+1 pass, add D_spec for speculative analysis → Reduced-B → retarget CCS

### Why Not ABANDON

1. **The novelty gap is real.** No tool combines quantitative cache bounds + compositional contracts for x86-64 binaries. Mitchell & Wang (ECOOP 2025) advanced quantitative analysis but without composition. BINSEC QRSE measures robustness, not channel capacity.

2. **Expected value favors continuation.** At P(Phase 1 success) = 60–65%, E[V] for Reduced-C exceeds ABANDON across all reasonable utility assumptions.

3. **The 2-hour theory phase is not strong evidence of mathematical inability.** The 30-day Phase Gate 0 is the genuine test.

4. **Fallback value is non-zero.** Planning artifacts (evaluation plan, pseudocode) have reuse value. Even a failed proof attempt produces useful mathematical insight.

### Why Not Full Scope

1. **Timeline impossibility.** 50–60K novel LoC for one researcher requires 3.8–8.3 years, not 16–22 months.

2. **Novelty erosion risk.** Every month of delay increases preemption probability. The Guarnieri/Reineke group is actively working on the software-side contract problem.

3. **Feasibility at 4/10 does not support full commitment.** Reduced-C at 6/10 feasibility is the right match for the evidence.

### Why Reduced-C Specifically

1. **Mitchell & Wang is the clarifying constraint.** They advanced CacheAudit quantification without composition or speculation. Reduced-C must deliver what they did NOT: compositional contracts. This is the minimal differentiator.

2. **The composition theorem is the highest-value deliverable per unit effort.** If proved, it's independently publishable. If it fails, early discovery saves months.

3. **Time-adjusted expected value.** Reduced-C delivers >2× the value per month of Full Reduced-A (Synthesizer's analysis, Chair-validated).

---

## Binding Amendments

### Amendment 1: Phase Gate 0 — Proof Sketches (30 Calendar Days)

**Deliverable:** Written proof sketches (≥3 pages each) for:
- (a) Composition soundness: B_{f;g}(s) ≤ B_f(s) + B_g(τ_f(s)) under explicit independence conditions
- (b) Independence condition verification for ≥2 of {AES T-table rounds, ChaCha20 quarter-rounds, Curve25519 scalar multiply}
- (c) D_quant taint-restricted counting soundness sketch

**Kill trigger:** No written proof sketch after 30 days → **ABANDON.**

**Pivot trigger:** Independence fails for all 3 patterns → pivot to Rényi. If Rényi also vacuous → **ABANDON.**

### Amendment 2: Phase Gate 1 — Precision Canary (90 Calendar Days from PG0 Pass)

**Deliverable:** Working implementation of D_cache ⊗ D_quant on AES T-table under LRU with exhaustive ground truth on ≤16-bit keys.

**Success threshold:** Bounds within 3× of exhaustive enumeration.

**Mandatory comparison:** Results must be compared to Mitchell & Wang (ECOOP 2025) if artifact available.

**Kill trigger:** Bounds >10× → **ABANDON.** Bounds 5–10× → one redesign iteration (30 days), then kill if still >5×.

### Amendment 3: Scope Lock

Reduced-C scope is **locked** until PG0 and PG1 both pass. No scope expansion to speculation or CCS targeting until:
- Composition proof sketches verified
- Precision canary meets 3× threshold
- ≥4K LoC of working analytical code exists

### Amendment 4: Prior Art Integration (Mandatory)

The proposal must cite and differentiate from:
- Mitchell & Wang (ECOOP 2025) — mandatory, HIGH priority
- BINSEC v0.11 QRSE — mandatory, MEDIUM priority
- SCAFinder, SpecLFB, Contract Shadow Logic, VeriCache — recommended in related work

### Amendment 5: Honest Positioning

The paper must:
- NOT claim to "answer LeaVe's open question" until speculation is added
- NOT claim "50–60K novel LoC" (honest for Reduced-C: 8–12K genuinely novel)
- Position as "compositional quantitative cache analysis for x86-64 binaries"
- Target SAS/VMCAI, not CCS

### Amendment 6: No Further Planning Phases

Any future pipeline stage must produce either mathematical proofs or working code. Additional specification, reformulation, or planning documents do not count toward phase completion.

### Amendment 7: Stall Detection

If fewer than 2K LoC of working analytical code exist 60 days after PG0 pass → **ABANDON.** Execution inability confirmed.

---

## Kill Triggers

| Trigger | Deadline | Condition | Action |
|---------|----------|-----------|--------|
| PG0: No proof sketches | Day 30 | No written proof sketch for composition soundness | **ABANDON** |
| PG0: Independence fails | Day 30 | All 3 patterns fail AND Rényi vacuous | **ABANDON** |
| PG1: Precision >10× | Day 120 | Bounds exceed 10× on AES T-table | **ABANDON** |
| PG1: Precision >5× post-redesign | Day 150 | Bounds exceed 5× after redesign | **ABANDON** |
| Stall detection | Day 90 | <2K LoC working code | **ABANDON** |
| Prior art preemption | Any time | Guarnieri et al. or Mitchell & Wang publish compositional contracts | **ABANDON or pivot** |
| Budget exhaustion | Month 12 | No submittable draft | **ABANDON** |

---

## Composite Scores

| Axis | Ideation (Pre-Theory) | Post-Theory | Δ | Rationale |
|------|:---------------------:|:-----------:|:---:|-----------|
| Extreme Value | 7 | **5** | −2 | Mitchell & Wang erosion; LLM competition; zero execution evidence; microscopic audience |
| Genuine Difficulty | 7 | **6** | −1 | After honest LoC deflation (15–20K genuinely novel); individual components have prior art |
| Best-Paper Potential | 6 | **4** | −2 | Zero proofs/code; synthesis headwind; Mitchell & Wang; Reduced-C targets SAS/VMCAI |
| Laptop CPU + No Humans | 7 | **8** | +1 | Reduced-C removes speculation overhead; universally agreed |
| Feasibility | 7 | **4** | −3 | theory_bytes=0; timeline infeasible at full scope; 60–65% at Reduced-C |
| **Composite** | **6.75** | **5.4** | **−1.35** | |

### Fatal Flaw Summary

| Flaw | Severity | Status |
|------|----------|--------|
| Zero theory output | HIGH | Phase Gate 0 (30 days) forces resolution |
| Timeline infeasibility (full scope) | HIGH | Resolved by scope reduction to Reduced-C |
| Mitchell & Wang novelty erosion | MEDIUM-HIGH | Differentiate on composition contracts |
| Independence condition untested | MEDIUM-HIGH | Phase Gate 0 tests this |
| Vacuous PLRU bounds | MEDIUM | LRU-first + regression detection (unchanged) |

---

## Dissent Record

The **Fail-Fast Skeptic** recommends ABANDON (composite 4.4/10, kill probability 55–65%). The Skeptic's position is the strongest on three points:

1. **The "planning as procrastination" diagnosis.** Five pipeline stages, ~250KB of planning, zero executable artifacts. This pattern is a red flag regardless of the 2-hour pipeline window.

2. **The timeline impossibility.** Full scope at 50–60K novel LoC for one researcher in 16–22 months is 10–25× historical productivity. The panel confirms this is not credible.

3. **The prior art landscape is more active than the ideation panel recognized.** Mitchell & Wang (ECOOP 2025), BINSEC v0.11, and the Guarnieri/Reineke research program are all converging on overlapping territory.

The Skeptic's ABANDON verdict becomes the consensus if:
- PG0 yields no proof sketches (Day 30)
- PG1 exceeds 10× precision (Day 120)
- The Guarnieri/Reineke group publishes software-side contracts before PG1

The 30-day Phase Gate 0 is the Skeptic's primary contribution to this synthesis: it forces a decision point where ABANDON becomes the default if evidence is not produced.

---

## Appendix: What Should Have Been Produced in the Theory Phase

For reference, a successful theory phase would have contained:

1. **Composition theorem proof** (10–20 pages): Formal statement and proof of B_{f;g}(s) ≤ B_f(s) + B_g(τ_f(s)) with precise independence conditions.
2. **Independence condition characterization** (5–10 pages): Formal definition, proof/counterexample for AES rounds with related subkeys.
3. **ρ specification** (5–15 pages): Formal definition, monotonicity proof, termination proof, soundness proof, worked example.
4. **Soundness framework** (5–10 pages): γ-only soundness argument for the relevant domains.

What was delivered: a JSON specification and an evaluation plan. The gap is total. The 30-day Phase Gate 0 gives the project one final opportunity to demonstrate mathematical capability before the Skeptic's ABANDON becomes the panel consensus.

---

## Salvage Value (If Abandoned)

Even under ABANDON, the following artifacts have residual value:

| Artifact | Value | Reuse Path |
|----------|-------|-----------|
| empirical_proposal.md | HIGH | Publication-quality evaluation design reusable in any cache side-channel paper |
| approach.json | MEDIUM-HIGH | Algorithmic pseudocode directly implementable; risk assessment reusable |
| Ideation documents (~250KB) | MEDIUM | Comprehensive survey of the design space for speculative cache-channel analysis |
| Composition theorem statement | LOW | The algebraic identity is known (Smith 2009); the domain-specific instantiation attempt has value as failed-approach documentation |

Estimated salvage: ~$25–35K equivalent in researcher-time savings if artifacts are transferred to a new project in the same space.
