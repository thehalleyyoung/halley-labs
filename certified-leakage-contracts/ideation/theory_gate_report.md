# Theory Gate Verification Report: certified-leakage-contracts

## Verdict: CONDITIONAL CONTINUE (proposal_00) at Reduced-C Scope

**Composite Score: V6 / D5.5 / BP3 / L7 / F4 = 5.1/10**

Kill probability: **~45%** (any publication); **~70%** (CCS specifically).
Publication probability: **~55%** (any venue); **~25–30%** (CCS).

---

## Panel Composition

| Expert | Role | Verdict | Composite |
|--------|------|---------|:---------:|
| **Independent Auditor** | Evidence-based scoring, challenge testing | CONDITIONAL CONTINUE | 5.0 |
| **Fail-Fast Skeptic** | Adversarial kill-path enumeration | STOP (ABANDON) | 4.4 |
| **Scavenging Synthesizer** | Value preservation, scope surgery | CONDITIONAL CONTINUE (Diamond Cut) | 5.7 |

**Panel Chair reconciled verdict: CONDITIONAL CONTINUE at Reduced-C scope.**

---

## Methodology

Four-phase adversarial process:

1. **Independent Proposals.** Each expert produced a full assessment in isolation, scoring all axes and recommending a verdict. No expert saw the others' work.
2. **Adversarial Cross-Critique.** Each expert attacked the weakest arguments of the other two with direct teammate-to-teammate challenges. Six critique documents produced (3 assessments + 3 cross-critiques).
3. **Panel Synthesis.** The chair reconciled all disagreements, weighting arguments by evidence quality, and produced reconciled scores.
4. **Verification Signoff.** Final consistency check against three prior evaluation panels (Skeptic: 5.4, Mathematician: 5.2, Community Expert: 5.9).

---

## The Critical Finding: theory_bytes = 0

### What the Theory Phase Was Supposed to Produce

The ideation-stage verification panels unanimously required **Phase Gate 1: Composition Theorem Proof** before any implementation. Specifically:

> Formally prove the min-entropy additive composition rule and characterize the independence condition. Demonstrate that the condition holds for at least 3 representative crypto patterns.

### What Was Actually Delivered

| Artifact | Size | Content |
|----------|------|---------|
| `theory/approach.json` | 23,973 bytes | Algorithmic specification, pseudocode, domain definitions, complexity estimates |
| `theory/empirical_proposal.md` | 39,702 bytes | Publication-quality evaluation plan: 7 RQs, 11 benchmarks, 5 baselines, falsification criteria |
| **Total** | **63,675 bytes** | **Planning/specification documents — zero mathematical proofs** |

### Panel Assessment

**theory_bytes = 0 is substantively accurate.** The 64KB of output is architectural planning and evaluation design, not mathematics. No `.lean`, `.v`, `.tex`, or sketch-proof files exist anywhere in the repository. Every mathematical claim (A1–A9) exists only as an English-language description.

**Mitigating context:** State.json timestamps show the theory phase ran ~2 hours (06:27–08:22 UTC). This is insufficient to prove a composition theorem. The failure reflects a pipeline timing constraint.

**However: the consequence is identical regardless of cause.** Every mathematical claim is unvalidated. Every feasibility estimate is conditioned on unproved theorems. The project has completed 5+ pipeline stages producing ~250KB of planning and zero executable artifacts.

**Planning artifacts have genuine value.** The approach.json contains actionable pseudocode (ρ reduction sequence, convergence bounds, fixpoint strategy). The empirical_proposal.md is publication-ready evaluation design. These save an estimated 4–6 weeks of design work. But planning ≠ theory, and theory ≠ implementation.

---

## Prior Art: Verified Novelty Erosion

| Citation | Status | Relevance |
|----------|--------|-----------|
| **Mitchell & Wang, ECOOP 2025** — Refined quantitative cache analysis | ✅ REAL | **HIGH.** Directly advances CacheAudit-style D_cache ⊗ D_quant territory. Differentiator: no composition. |
| **BINSEC v0.11 QRSE** — Quantitative robust symbolic execution | ✅ REAL | **MEDIUM.** Measures robustness, not channel capacity. Partial overlap. |
| **SCAFinder (TIFS 2024)** — Cache hardware verification | ✅ REAL | LOW. Hardware-side. |
| **SpecLFB (USENIX Security 2024)** — Hardware defense | ✅ REAL | NONE. Hardware mitigation. |
| **Contract Shadow Logic (arXiv 2024)** — RTL speculation verification | ✅ REAL | LOW. Hardware-side. |

**Revised novelty erosion: MEDIUM.** Mitchell & Wang is the primary threat. The four-property combination (quantitative + speculative + binary + compositional) remains novel, but D_cache ⊗ D_quant precision is no longer the frontier. **The proposal MUST differentiate on compositional contracts** — the one capability Mitchell & Wang lack.

---

## Reconciled Axis Scores

### Axis 1: Extreme and Obvious Value — 6/10

| Expert | Score | Key Argument |
|--------|:-----:|-------------|
| Auditor | 6 | Real problem, LeaVe positioning, CVE evidence |
| Skeptic | 4 | 35–50 person audience, LLM triage handles 90% |
| Synthesizer | 7 | Regression detection is killer app, LeaVe gap is real |

**Reconciled: 6.** The Auditor's evidence-anchored score prevails. Regression detection framing is the strongest value argument — even with 5–10× imprecise absolute bounds, version-to-version changes flag introduced leaks. The LeaVe positioning (answering a CCS Distinguished Paper's open question) is genuine publication currency. Direct audience is narrow (~50 crypto library maintainers) but the CI-integration framing reaches security engineering teams more broadly. The Skeptic's LLM competition argument limits the ceiling (preventing 7+) but doesn't eliminate the formal-guarantee tail.

### Axis 2: Genuine Software Difficulty — 5.5/10

| Expert | Score | Key Argument |
|--------|:-----:|-------------|
| Auditor | 5 | ONE novel theorem, ~15–20K genuinely novel LoC |
| Skeptic | 5 | ρ is 3–5K LoC; everything else is adaptation |
| Synthesizer | 8.5 | PhD-thesis difficulty, CacheAudit took 3 years |

**Reconciled: 5.5.** The Synthesizer's D8.5 is indefensible — the Mathematician independently scored D5, and the independent verification decomposes genuinely novel LoC to ~15–20K within a ~65K artifact. One genuinely novel theorem (ρ) with ~2–3 supporting lemmas. CacheAudit (~15K LoC, simpler scope) is the correct comparator. D5.5 acknowledges that ρ's three-way domain interaction is genuinely subtle while recognizing the proof technique is standard lattice theory on bounded-height domains.

### Axis 3: Best-Paper Potential — 3/10

| Expert | Score | Key Argument |
|--------|:-----:|-------------|
| Auditor | 3 | Zero proofs + zero code = ~2–3% probability |
| Skeptic | 2 | Synthesis headwind, Mitchell & Wang erosion |
| Synthesizer | 5 | Ceiling if everything works; LeaVe positioning |

**Reconciled: 3.** The Skeptic and Auditor anchor on evidence: zero proofs, zero code, 10–14 month horizon, synthesis narrative. The Synthesizer's BP5 prices the ceiling of the completed work, but best-paper probability must weight current execution state. At Reduced-C targeting SAS/VMCAI: ~10% best-paper. At CCS: ~2–3%. The synthesis headwind (combining CacheAudit + Spectector + Smith 2009 via ρ) is real — best papers create paradigms, this instantiates one.

### Axis 4: Laptop CPU + No Humans — 7/10

| Expert | Score | Key Argument |
|--------|:-----:|-------------|
| Auditor | 7 | Bounded cache geometry, polynomial complexity |
| Skeptic | 8 | Only non-controversial axis |
| Synthesizer | 7 | Non-speculative clearly feasible |

**Reconciled: 7.** Universal agreement. Abstract interpretation on bounded L1 LRU cache (64 sets × 8 ways × 1 taint bit = ~128 bytes per abstract state) is polynomial by construction. CacheAudit analyzed AES in seconds on decade-old hardware. At Reduced-C (no speculation domain), context explosion is eliminated. Fully automated with zero annotation.

### Axis 5: Feasibility — 4/10

| Expert | Score | Key Argument |
|--------|:-----:|-------------|
| Auditor | 4 | PG1 not attempted; gap between planning and implementation |
| Skeptic | 3 | 0/4 phase gates passed; joint survival ≤25% |
| Synthesizer | 4 | Full scope impossible; Reduced-C plausible |

**Reconciled: 4.** The Auditor's evidence-based correction prevails. theory_bytes=0 is a genuine red flag — feasibility cannot exceed 4 until ρ is sketched on paper. Full scope (50–60K LoC in 16–22 months) is timeline-impossible (corrected estimate: 46–99 person-months). At Reduced-C (~12–18K LoC), feasibility improves to ~60–65% for any publication. The fallback hierarchy (Reduced-A → B → C → D) limits catastrophic loss.

---

## Expert Disagreements — Resolved

### 1. Is theory_bytes=0 Recoverable?

**Skeptic:** Fatal signal — "planning as procrastination." Five stages, ~250KB planning, zero execution.

**Auditor:** Recoverable — 2-hour pipeline window was insufficient. Planning quality demonstrates competence.

**Synthesizer:** Recoverable — proofs are likely short (ρ monotonicity ~2–3 pages). Theory stage produced planning because proofs weren't the existential risk.

**Resolution: Recoverable, but only with a hard 4-week gate.** The Auditor and Synthesizer are correct that 2 hours is insufficient for proofs. The Skeptic is correct that the pattern (planning over execution) is a red flag. Phase Gate 0 (4 weeks, proof sketches) forces resolution. If no proof sketch exists after 30 focused days, the Skeptic's ABANDON becomes consensus.

### 2. What Is the Correct Scope?

**Auditor:** CONDITIONAL CONTINUE at original scope with hard gates.

**Skeptic:** STOP, or at most 4-week PoC.

**Synthesizer:** Diamond Cut (~30K LoC, two-paper strategy).

**Resolution: Reduced-C (~12–18K LoC, SAS/VMCAI).** The Synthesizer's Diamond Cut (30K LoC) is still too ambitious given zero existing code. Full scope is timeline-impossible. Reduced-C targets the minimum viable contribution: ρ operator + LRU D_cache + D_quant + composition on crypto patterns. Scope expansion to speculation is locked behind gates.

### 3. Kill Probability

**Skeptic:** 70%+ (assumes independent risks).

**Auditor:** 35–45% (correlated risks).

**Synthesizer:** 25–30% at Reduced-C.

**Resolution: ~45%.** The Skeptic's 86% raw calculation assumes independence across heavily correlated risks (ρ precision and independence condition share mathematical roots). The Auditor's correction (35–45%) is methodologically sound. At Reduced-C scope, the kill probability is ~40–45% for any venue publication, ~70% for CCS specifically.

---

## Fatal Flaws

| # | Flaw | Severity | Status |
|---|------|----------|--------|
| 1 | **Zero mathematical output after theory phase** | HIGH | Phase Gate 0 (4 weeks) forces resolution |
| 2 | **Timeline infeasibility at full scope** | HIGH | Resolved by scope reduction to Reduced-C |
| 3 | **Mitchell & Wang (ECOOP 2025) novelty erosion** | MEDIUM-HIGH | Differentiate on composition contracts |
| 4 | **Independence condition untested** | MEDIUM-HIGH | Phase Gate 0 tests this |
| 5 | **ρ precision may be empirically hollow** | HIGH | Phase Gate 1 tests this (pencil-and-paper + canary) |
| 6 | **Vacuous PLRU bounds** | MEDIUM | LRU-first + regression detection (unchanged) |

---

## Binding Conditions for CONTINUE

| Gate | Deadline | Deliverable | Kill Trigger |
|------|----------|-------------|--------------|
| **G0: Proof Sketches** | **4 weeks** | (a) ρ monotonicity/termination proof sketch (≥3 pages, formal lemma statements), (b) Composition soundness proof sketch on ≥2 crypto patterns, (c) Pencil-and-paper worked example showing ρ *strictly* tightens bounds over direct product on a 4–6 instruction fragment | No written proof sketch → **ABANDON** |
| **G1: Precision Canary** | **12 weeks** (8 weeks after G0) | Working D_cache ⊗ D_quant on AES T-table with bounds ≤3× exhaustive enumeration (small keys ≤16 bits). ~3–4K LoC. Mandatory comparison to Mitchell & Wang if artifact available. | Bounds >10× → **ABANDON**. Bounds 5–10× → one 4-week redesign, then kill if still >5× |
| **G2: Composition Sanity** | **16 weeks** (4 weeks after G1) | Composition overhead κ ≤ 3.0× on AES-128 (10 rounds composed vs. monolithic analysis) | κ >5× → **ABANDON** |
| **Scope Lock** | Until G0+G1 pass | No scope expansion beyond Reduced-C. No speculation. No CCS targeting. | Non-negotiable |
| **Stall Detection** | 10 weeks after G0 pass | ≥2K LoC of working analytical code | <2K LoC → **ABANDON** |
| **Prior Art Gate** | Any time | Mitchell & Wang or Guarnieri group publishes compositional contracts | **ABANDON or pivot** |

---

## Recommended Scope, Venue, Timeline

**Scope: Reduced-C**
- D_cache (tainted LRU) + D_quant (taint-restricted counting) + two-way reduction ρ₂ (cache→quant)
- Compositional leakage contracts with additive composition rule
- Regression detection mode on 3 CVE binary pairs
- ~12–18K LoC total, ~8–12K genuinely novel
- No speculation domain. No three-way ρ. No D_spec.

**Primary venue: SAS 2027 or VMCAI 2027**
- Contribution: "First compositional quantitative cache analysis for x86-64 crypto binaries"
- Differentiator from Mitchell & Wang: composition + regression detection

**Stretch venue: CCS 2027** (requires G0+G1+G2 pass AND scope expansion to speculation)

**Timeline: 10–14 months to submission.** Gates at weeks 4/12/16 provide three off-ramps.

**Upgrade path:** If G0+G1+G2 all pass AND ≥4K LoC working code exists, unlock scope expansion to Reduced-B (add D_spec for speculative analysis) and retarget CCS.

---

## Expected Value Assessment

| Outcome | Probability | Value |
|---------|:-----------:|:-----:|
| SAS/VMCAI accept (Reduced-C) | ~35% | 5.0 |
| CCS borderline accept (if scope upgraded) | ~10% | 7.0 |
| Workshop/tool paper | ~10% | 2.0 |
| Kill (nothing publishable) | ~45% | 0.0 |

**E[CONTINUE] = 0.35×5.0 + 0.10×7.0 + 0.10×2.0 = 2.65**
**E[ABANDON] = salvage value ≈ 0.5**

**Differential: +2.15.** Continuation is positive-EV conditional on phase-gate discipline.

---

## Dissent Record

The **Fail-Fast Skeptic** recommends STOP (composite 4.4/10). The Skeptic's position is strongest on three points:

1. **"Planning as procrastination."** Five stages, ~250KB of planning, zero executable artifacts. A researcher who prioritized evaluation-plan prose over the composition proof that every evaluator demanded is exhibiting revealed preference for planning over execution.

2. **Timeline impossibility at full scope.** 50–60K novel LoC for one researcher in 16–22 months requires 10–25× historical productivity. Confirmed infeasible by all panelists.

3. **ρ vacuity is cheaply testable.** A pencil-and-paper example proving ρ strictly tightens bounds could be produced in 2 weeks. Investing months without this basic validation is irrational.

The Skeptic's STOP becomes consensus if G0 yields no proof sketches (day 30). The 4-week Phase Gate 0 is the Skeptic's primary contribution to this synthesis.

---

## Composite Score Comparison

| Source | V | D | BP | L | F | Composite |
|--------|:-:|:-:|:--:|:-:|:-:|:---------:|
| Skeptic panel (prior) | 5 | 6 | 4 | 8 | 4 | 5.4 |
| Mathematician panel (prior) | 6 | 5 | 3 | 7 | 5 | 5.2 |
| Community Expert panel (prior) | 6 | 6.5 | 5 | 7 | 5 | 5.9 |
| **This panel (reconciled)** | **6** | **5.5** | **3** | **7** | **4** | **5.1** |

The 5.1 composite is slightly below prior panels (range 5.2–5.9), closest to the Mathematician's 5.2, reflecting the BP downgrade (3 vs. prior 3–5 range) and the Skeptic's contribution on feasibility risk.

---

## Salvage Value (If Abandoned)

| Artifact | Value | Reuse Path |
|----------|-------|-----------|
| empirical_proposal.md (40KB) | HIGH | Publication-quality evaluation design reusable in any cache side-channel paper |
| approach.json (24KB) | MEDIUM-HIGH | Algorithmic pseudocode directly implementable; convergence bounds reusable |
| Ideation documents (~250KB) | MEDIUM | Comprehensive design space survey for speculative cache analysis |
| Composition theorem statement | LOW | The algebraic identity is Smith (2009); domain-specific instantiation has value as documentation |

Estimated salvage: ~4–6 weeks of researcher-time savings if artifacts transfer to a new project.

---

## Rankings

```json
{
  "rankings": [
    {
      "proposal_id": "proposal_00",
      "score": 5.1,
      "verdict": "CONDITIONAL CONTINUE",
      "reason": "CONDITIONAL CONTINUE at Reduced-C scope (12-18K LoC, SAS/VMCAI target). ONE genuinely novel theorem (ρ reduction operator). theory_bytes=0 but recoverable with 4-week proof-sketch gate. Mitchell & Wang (ECOOP 2025) erodes quantitative novelty but not compositional contracts. Kill probability ~45%. Binding gates at weeks 4/12/16. Positive expected value (E[CONTINUE]=2.65 vs E[ABANDON]=0.5) conditional on phase-gate discipline. Skeptic dissents ABANDON (composite 4.4).",
      "scavenge_from": []
    }
  ]
}
```
