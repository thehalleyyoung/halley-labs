# Theory Evaluation: Skeptical Verification

**Proposal:** proposal_00 — The Cognitive Regression Prover: A Three-Layer Usability Oracle
**Stage:** Post-theory verification
**Method:** Three-expert adversarial panel (Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer) with independent proposals → cross-critiques → synthesis → independent verifier signoff
**Date:** 2026-03-04

---

## VERDICT: CONTINUE

**Final Scores (post-critique, post-verification):**

| Axis | Score | Rationale |
|------|-------|-----------|
| 1. Extreme Value | **7/10** | Real unaddressed gap (no automated structural usability regression detection in CI/CD). Parameter-independence gives zero-calibration verdicts on Day 1. Docked for zero empirical validation and task-specification adoption barrier. |
| 2. Genuine Software Difficulty | **6/10** | With Layer 3 deferred: MDP construction is genuinely hard (prior tools died here), paired-comparison proof is medium-difficulty, cliff detection requires novel algorithmics. ~40% is standard engineering. |
| 3. Best-Paper Potential | **5/10** | Fragility-as-robustness framing is novel. Paired-comparison error cancellation is a real result. But: zero empirical evidence, conjectured Theorem 4 deferred, modest validation threshold. Acceptance-worthy at CHI/UIST, not award-worthy without execution. |
| 4. Laptop-CPU & No-Humans | **9/10** | Principled CPU-native design throughout. Accessibility trees (no vision models), embarrassingly parallel sampling, CPU-native SMT, published psychophysical parameters (no training), retrospective validation against existing data. Strongest axis. |
| 5. Feasibility | **6/10** | Incremental architecture correctly isolates risk. Layer 1 deliverable in 10–12 weeks. But: zero implementation progress, accessibility-tree parsing is a known graveyard, retrospective validation data availability is uncertain. |
| **TOTAL** | **33/50** | |

---

## Team Process Summary

### Phase 1: Independent Proposals

Three experts scored independently before seeing each other's work:

| Axis | Auditor | Skeptic | Synthesizer |
|------|---------|---------|-------------|
| Value | 7 | 6 | 7 |
| Difficulty | 7 | 5 | 7 |
| Best-Paper | 6 | 4 | 7 |
| CPU/No-Humans | 9 | 8 | 9 |
| Feasibility | 7 | 6 | 7 |
| **Total** | **36** | **29** | **37** |

Spread: 8 points (29–37). All three recommended CONTINUE.

### Phase 2: Adversarial Cross-Critiques (6 challenges)

**Challenges upheld (scores changed):**
- **Skeptic → Auditor:** Best-Paper 6→5. Zero data cannot justify 6; a best paper needs proved theorems or stunning data, and this has neither in hand.
- **Skeptic → Synthesizer:** Best-Paper 7→6. Framework elegance alone doesn't reach strong-accept without empirical validation or proved central theorems.
- **Auditor → Skeptic:** Difficulty 5→6. Integration difficulty exceeds component difficulty; the engineering-theory interface (noisy trees → clean formal MDPs) is a genuine difficulty multiplier.
- **Synthesizer → Auditor:** Feasibility 7→6. CogTool-class system from zero code in 9 months is a stretch even with better APIs.

**Challenges that changed framing (no score change):**
- **Synthesizer → Skeptic:** Fragility "trivial" withdrawn to "straightforward." The definition is simple; the computation via cliff-location theorem is a genuine (if modest) algorithmic contribution.
- **Auditor → Synthesizer:** 50%-cut narrative accepted. The revised three-act structure (parameter-independence → fragility → trust hierarchy) is tighter and more publishable than the full vision.

Post-critique spread: 6 points (30–36). Meaningful convergence.

### Phase 3: Independent Verifier Signoff

**SIGNOFF: APPROVED.** Verifier confirmed:
- Scores internally consistent with evidence
- CONTINUE justified (no missed fatal flaws)
- Mild groupthink detected but mitigated by substantive go/no-go gate
- Difficulty adjusted −1 from median (Layer 3 deferral reduces difficulty)
- Added one condition: prototype Layer 1 on ≥3 real component libraries within 4 weeks

---

## Fatal Flaws Identified

### Flaw 1: "Zero False Positives" Is Tautological (HIGH)
The claim "zero false positives on parameter-independent verdicts" is true by construction: the test fires when `n_after > n_before`, and the system *defines* that as a regression. There is no external validation that "more options = worse usability" in all cases (expert-targeted or information-dense UIs may benefit from more options). **Required fix:** Qualify with the assumption. Replace with "zero false positives under the assumption that monotone structural cost increases constitute regressions."

### Flaw 2: Retrospective Validation Data May Not Exist (CRITICAL)
CogTool-era datasets contain task-completion times, not accessibility-tree representations. Published UI comparison studies (Findlater 2004, Gajos 2010, Cockburn 2007) predate modern accessibility standards. "Extract or reconstruct accessibility trees" for 15-year-old UIs may be impossible. If validation data doesn't exist, the consistency-oracle claim is unfalsifiable. **Required fix:** Verify data availability within 2 weeks as a hard go/no-go gate.

### Flaw 3: 70–85% Coverage Number Is Unsupported (HIGH)
The claim that Layer 1 "catches 70–85% of structural usability regressions" has no empirical or analytical backing. The proposal body says "estimated 50–70%"; the Layer 1 deliverable says "70–85%." The numbers are internally inconsistent and externally unvalidated. **Required fix:** Drop the number or derive it from systematic analysis of real regression types in open-source changelogs.

### Flaw 4: Ordinal Cost-Algebra Soundness Is Conjectured (MODERATE)
Theorem 4 is explicitly "conjectured" with "incomplete proof sketch." The dimensional mismatch (time vs. bits) that killed the original theorem is acknowledged but unresolved. The inductive step for sequential composition is "plausible but unproven." **Required fix:** Defer to future work. Layers 1–2 don't require it.

### Flaw 5: Layer 1 Novelty Is Thin (MODERATE)
Layer 1 is substantially CogTool + interval arithmetic + CI/CD integration. The engineering is valuable (CogTool never shipped as a CI/CD tool), but the theoretical novelty of Layer 1 alone is near-zero. The parameter-independence observation is the one genuine insight, but it follows trivially from monotonicity of standard cost functions. **Implication:** The project's research novelty is concentrated in Layer 2 (paired-comparison, fragility). If Layer 2 underdelivers, the project is a useful tool with a thin paper.

### Flaw 6: Fragility Confound (MODERATE)
F(M) = max_β − min_β E[C(τ)] does not distinguish "good for experts, bad for novices" from "bad for everyone at different levels." A uniformly terrible UI scores low fragility. Needs an absolute-cost floor to be interpretable. **Required fix:** Add a minimum-cost qualifier or position fragility explicitly as a *secondary* signal after cost regression.

**No fatal flaw justifies ABANDON.** All flaws have identified mitigations, and the incremental architecture prevents any single failure from destroying the project.

---

## Claims That Survived Challenge

1. **The problem is real and unaddressed.** No existing tool provides automated structural usability regression detection in CI/CD. The gap between accessibility linters (rule-based, no cognitive model) and manual usability studies (expensive, quarterly) is genuine.

2. **Parameter-independence is a genuine insight.** The observation that dominant failure modes (option proliferation, target shrinking, depth increase) yield parameter-free regression verdicts via interval arithmetic is unstated in prior automated usability evaluation literature. Easy to prove (monotonicity) but impactful for adoption.

3. **Paired-comparison error cancellation is a real mathematical contribution.** Differential estimation under shared analysis achieves O(ε) error vs O(Hβε) for independent analysis. The correlated-bias cancellation argument is non-trivial and has implications beyond usability.

4. **Cognitive fragility is a genuinely new metric category.** "Usability as robustness to user capacity variation" connects to inclusive design, adversarial robustness, and chaos engineering. The framing is novel and the "cognitive cliff" concept is vivid and actionable.

5. **The incremental architecture correctly isolates risk.** Layer 1 ships regardless of Layer 2–3 outcomes. Each layer has standalone value. This is genuine risk management.

6. **CPU-only design is principled.** Every component is chosen for CPU execution, not compromised into it. WCAG mandates are improving accessibility-tree quality for free.

---

## Conditions for Continuation

### Condition 1 (GO/NO-GO GATE — 2 weeks): Verify Retrospective Validation Data
Confirm that at least ONE dataset (CogTool, Oulasvirta, or modern UI comparison studies) provides accessibility-tree-level UI pairs with human performance orderings. If no such data exists, pivot to "internal consistency only" — a significantly weaker claim. **This is a hard stop.**

### Condition 2: Drop "Zero False Positives" Language
Replace with qualified statement: "Parameter-independent verdicts are correct by construction under the assumption that monotone structural cost increases constitute regressions."

### Condition 3: Drop or Empirically Support Coverage Numbers
Either measure the fraction of real regressions that are parameter-free on the benchmark suite, or remove all quantitative coverage claims.

### Condition 4: Center Fragility as Lead Contribution
All three experts agree fragility-as-robustness is the most publishable and most novel contribution. The paper should lead with it, not with cost differencing.

### Condition 5: Defer Layer 3 to Future Work
Bisimulation scaling and cost-algebra ordinal soundness are correctly isolated as high-risk. Attempting to prove conjectured theorems in a first submission risks sinking the whole paper.

### Condition 6 (Added by Verifier): Prototype Layer 1 on ≥3 Real Libraries (4 weeks)
Build a working Layer 1 prototype on Material UI, Ant Design, and one other library before investing in Layer 2 theory. The accessibility-tree-to-MDP reduction is the project-killing engineering risk.

---

## The Key Question: Is This Just CogTool With Interval Arithmetic?

**Layer 1: Substantially yes.** Fitts' + Hick's + structural predicates + interval arithmetic + CI/CD. The engineering integration is a genuine advance (CogTool never shipped as a CI/CD tool), but the theoretical novelty is near-zero.

**Layer 2: No, if delivered.** The bounded-rational MDP formulation, paired-comparison theorem, and cognitive fragility analysis are genuinely new. The paired-comparison error-cancellation argument has real mathematical content. But Layer 2's value depends on: (a) the MDP construction working on real UIs, and (b) retrospective validation confirming model orderings match human perception.

**Bottom line:** The proposal is CogTool-with-interval-arithmetic at launch, with a plausible but uncertain path to something genuinely novel. The novelty is real but contingent on Layer 2 delivering.

---

## Underappreciated Strengths

1. **The trust hierarchy (parameter-free → interval → behavioral) is a natural calibration spectrum.** "Layer 1 says regression with certainty; Layer 2 says by how much" is the right trust architecture.

2. **The consistency-oracle framing resolves a 25-year impasse.** Since Ivory & Hearst (2001), the field has been stuck between "automated tools aren't accurate enough" and "manual testing doesn't scale." Ordinal consistency (not absolute accuracy) is the correct resolution.

3. **WCAG mandates are a tailwind.** Accessibility-tree quality improves for free as regulatory compliance improves. The system's input quality gets better without any effort from the authors.

4. **The "Why Not LLMs" positioning is defensible and durable.** Determinism, quantitative comparability, monotonicity, and formal error bounds are genuine requirements for CI/CD gates that LLMs cannot satisfy.

---

## Score Trajectory

| Phase | Value | Difficulty | Best-Paper | CPU | Feasibility | Total |
|-------|-------|------------|------------|-----|-------------|-------|
| Prior depth-check | 6 | 7 | 5 | 9 | — | 27/40 |
| Post-amendment (Auditor) | 7 | 7 | 6 | 9 | 7 | 36/50 |
| Post-amendment (Skeptic) | 6 | 5 | 4 | 8 | 6 | 29/50 |
| Post-amendment (Synthesizer) | 7 | 7 | 7 | 9 | 7 | 37/50 |
| Post-cross-critique mean | 6.7 | 6.7 | 5.0 | 8.7 | 6.3 | 33.3/50 |
| **Final (verified)** | **7** | **6** | **5** | **9** | **6** | **33/50** |

---

## FINAL VERDICT

### proposal_00: **CONTINUE**

The Cognitive Regression Prover addresses a genuine unmet need with a sound incremental architecture, principled CPU-only design, and two real (if modest) intellectual contributions: the paired-comparison error-cancellation theorem and the cognitive fragility framing. The project's main risk is not failure but mediocrity — succeeding as a useful engineering tool with thin research novelty if Layer 2 underdelivers. Centering fragility and clearing the retrospective-data gate within 2 weeks are the critical path items.

The proposal survives skeptical review, but just barely. It is not yet a best-paper candidate. It is a solid project with best-paper *potential* contingent on: (1) confirmed retrospective validation data, (2) a working Layer 1 prototype on real UIs, and (3) a tight paper centered on fragility-as-robustness.
