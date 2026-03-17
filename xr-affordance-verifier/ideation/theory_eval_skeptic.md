# Skeptic Verification: xr-affordance-verifier (proposal_00)

**Stage:** Verification (post-theory)
**Date:** 2026-03-08
**Panel:** Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer
**Verifier:** Independent signoff (APPROVED WITH RESERVATIONS)
**Input documents:** ~495KB across 14 structured documents (ideation/*, theory/*, problem_statement.md, State.json)
**Prior scores:** Depth check 20/40 (V4/D7/BP4/L5) → Final approach 25/40 (V6/D6/P6/F7)

---

## Executive Summary

Three independent experts evaluated proposal_00 (Coverage-Certified XR Accessibility Verifier) after theory development. The project proposes a two-tier system: Tier 1 is an interval-arithmetic linter using affine arithmetic for reachability envelopes (<2s in Unity editor); Tier 2 is a sampling-guided symbolic engine with coverage certificates ⟨S, V, U, ε_analytical, ε_estimated, δ, κ⟩. Crown jewel: C1 coverage certificate soundness theorem (B+ grade). Two-paper strategy targets CAV (certificates) + UIST (linter). ~43–68K LoC total, ~21–37K novel. State.json shows theory_bytes=0 (path bug — approach.json is ~66KB), impl_loc=0.

**Panel process:** Independent proposals → adversarial cross-critique → synthesis of strongest elements → independent verifier signoff. The Auditor scored 18/50 ABANDON; the Skeptic scored ~14/50 ABANDON; the Synthesizer scored 25/50 full / 32/50 salvage SALVAGE-AND-REDIRECT. Cross-critique resolved disagreements. The Independent Verifier approved with three required corrections (Laptop/No-Humans underscored, Best-Paper inconsistent with acknowledged novelty, Fatal Flaw F4 presented assertion as fact).

**Verdict: CONDITIONAL CONTINUE at reduced scope.** Composite 21/50 (full project, corrected) / 27/50 (salvage path, corrected). The full project as specified faces irrecoverable constraints (no-humans kills UIST paper and demand validation, zero demand signal, ε numerically worse than Clopper-Pearson). The salvage path — domain-general coverage certificates targeting TACAS/FMCAD, with Tier 1 linter as parallel standalone — survives with 8 mandatory amendments and a 2-month hard kill-chain.

---

## Panel Composition and Process

### Independent Auditor
Evidence-based scoring. Challenged every claim against specific textual citations from the ~495KB record. Key contribution: identified the irrecoverable loop created by the no-humans constraint (cannot validate demand → cannot publish tool paper → cannot demonstrate value).

### Fail-Fast Skeptic
Maximally hostile reviewer. Key contribution: the "formalism theater" attack — that the system uses heavyweight formal methods providing numerically weaker guarantees than a 2K-LoC Monte Carlo tool. While the label was rejected as a strawman in cross-critique (the certificate provides spatial structure, not just a number), the kernel of truth — that the certificate's ε is numerically looser than CP from the same samples — survived as a serious concern.

### Scavenging Synthesizer
Identified three concrete salvage paths with LoC estimates and publication probabilities:
- **Path A:** Domain-general coverage certificates (15–22K LoC, TACAS/FMCAD, 20–30%)
- **Path B:** Tier 1 standalone linter (12–18K LoC, ICSE Tool Track/ASSETS, 20–30%)
- **Path C:** Robotics workspace certification (22–35K LoC, ICRA/IROS, 20–30%)

### Independent Verifier
Approved with reservations. Required three corrections: (1) Laptop/No-Humans grossly underscored at 3/10 — should be 6–7/10 since all computation is CPU-based and no-humans kills a venue, not execution; (2) Best-Paper inconsistent — panel acknowledges κ-completeness as genuinely novel but scores 3/10; (3) Fatal Flaw F4's "97% MC detection" is an untested assertion, not an established fact.

---

## Scoring (Corrected per Verifier)

| Axis | Auditor | Skeptic | Synthesizer (full/salvage) | Consensus | Corrected |
|------|---------|---------|---------------------------|-----------|-----------|
| **1. Extreme Value** | 3 | 3 | 4 / 5 | 3 | **3** |
| **2. Genuine Software Difficulty** | 5 | 5 | 6 / 5 | 5 | **5** |
| **3. Best-Paper Potential** | 3 | 2→3 | 4 / 5 | 3 | **4** |
| **4. Laptop-CPU & No-Humans** | 4→3 | 3 | 6 / 8 | 3 | **6** |
| **5. Feasibility** | 3 | 2→3 | 5 / 7 | 3 | **3** |
| **Composite (full)** | **18** | **14→17** | **25** | **17** | **21** |
| **Composite (salvage)** | — | — | **32→26** | **23** | **27** |

### Axis 1: Extreme Value — 3/10

**All three agree (high confidence):** Zero demonstrated demand. The demand validation gate (D7: survey ≥20 developers) cannot be passed under the no-humans constraint. No developer surveys, no platform-holder interest, no XR-specific accessibility regulation exists today.

**The simpler-alternative problem:** A stratified Monte Carlo tool (~2–5K LoC) captures the vast majority of detection value. The formal certificate adds *certification structure* (spatial verification map, κ-completeness, counterexample generation) rather than *detection capability*. Whether certification structure has value is an open question — the project cannot answer it within the no-humans constraint.

**The empty-niche advantage:** There are zero competing XR accessibility tools of any kind. The Tier 1 linter, even as a simple interval-arithmetic checker, would be the first. In software ecosystems, first-movers in empty niches frequently become de facto standards (ESLint, axe-core). The Verifier correctly notes this is underweighted.

**Regulatory timing:** The EU Accessibility Act (effective June 2025) does not name XR, but its scope will likely be interpreted to include immersive interfaces. Section 508/ADA Title I already apply to enterprise XR (Boeing, Lockheed Martin). The regulatory window may open faster than assumed — but this is speculative upside, not demonstrated demand.

**Score rationale:** Real problem, zero demand signal, dominant simpler alternative on detection, speculative regulatory driver. The certification-structure value is genuine but unvalidated.

### Axis 2: Genuine Software Difficulty — 5/10

**All three agree (high confidence):** The project involves 15–22K lines of genuinely difficult code concentrated in three areas: (1) affine-arithmetic FK engine with wrapping control through 7-joint revolute chains (3–5K novel), (2) coverage certificate engine with piecewise-Lipschitz partitioning and κ-completeness tracking (5–8K novel), (3) adaptive stratified sampler with frontier detection and SMT orchestration (5–8K novel).

The remaining ~25–40K LoC is engineering: Unity YAML parser, editor plugin, benchmark infrastructure, population reporting. Important but not algorithmically novel.

**Library reuse is substantial:** Pinocchio (FK evaluation), Z3 (SMT), established affine-arithmetic libraries. The novelty is in composition and domain-specific adaptation, not in inventing new algorithms.

**The honest difficulty:** C1's proof is a "genuine but incremental" composition of known techniques (Hoeffding bounds + stratified sampling + volume subtraction) applied to a new domain. The difficulty is making the certificate *tight* (empirical, not theoretical) and handling piecewise-Lipschitz boundaries (engineering + careful analysis).

### Axis 3: Best-Paper Potential — 4/10 (corrected from 3)

**Crown jewel assessment:** C1 (Coverage Certificate Soundness) is graded B+ by the project's own Math Assessor. "Individual components known; composition is new." Conceptual distance from Younes & Simmons (statistical model checking) is shorter than originally claimed.

**What survived cross-critique as genuinely novel:**
1. **κ-completeness framework** — explicitly quantifying what a verification certificate covers vs. doesn't is a genuine contribution to parameterized testing methodology. No existing framework tracks this.
2. **Decision 7** — crediting Tier 1 affine-arithmetic green regions as symbolically verified volume in a statistical certificate. This connects abstract interpretation to statistical testing in a potentially novel way.
3. **Dual-ε reporting** (Decision 8) — reporting both ε_analytical (provably sound, may be loose) and ε_estimated (tight, not provably sound). Honest reporting of soundness-tightness tradeoffs.
4. **Piecewise-Lipschitz certificates** — handling joint-limit discontinuities as explicit partition boundaries rather than ignoring or excluding them.

**Publication probability (corrected):**

| Venue | Probability | Notes |
|-------|-------------|-------|
| TACAS / FMCAD (domain-general certificates) | 20–30% | Best fit for B+ composition; values new application domains |
| ISSTA / FSE (testing methodology) | 15–25% | κ-completeness + Decision 7 as testing contribution |
| ICSE Tool Track (linter) | 20–30% | If automated benchmarks are sufficiently compelling |
| CAV (formal verification) | 10–20% | Below CAV's typical bar for decidability-theoretic results |
| UIST (developer tool) | 0% as planned | Requires developer study; blocked by no-humans constraint |

**Best-paper probability at any venue: 3–5%.** The κ-completeness framework could attract attention at ISSTA if positioned as a contribution to parameterized testing theory.

**Verifier correction:** The panel cannot reject "formalism theater" as a strawman, acknowledge κ-completeness as genuinely novel, and score 3/10. Corrected to 4/10.

### Axis 4: Laptop-CPU Feasibility & No-Humans — 6/10 (corrected from 3)

**Laptop-CPU: PASS.** All computation is CPU-friendly:
- Tier 1: Affine-arithmetic FK at ~0.1ms/eval → 10K evals ≈ 1s. Memory <100MB. ✅
- Tier 2 (single-step): ~10s sampling + ~4 min SMT (6,000 queries × 100ms). Memory 1–10GB. ✅
- Tier 2 (multi-step k≤3): ~15 min, ~1GB. Tight but feasible on 16GB. ✅
- No GPU required anywhere in pipeline. Confirmed by all analyses. ✅
- ANSUR-II anthropometric data: public, freely downloadable, no annotation. ✅

**No-Humans: PARTIAL FAIL.** The developer study (15–22 participants, IRB-required) violates the no-humans constraint. This kills:
- The UIST paper's primary evaluation
- Gate D7 (demand validation: survey ≥20 developers)
- Any claim about tool usability or developer workflow integration

**What survives without humans:** The theory paper (CAV/TACAS/FMCAD) requires only automated benchmarks on procedural scenes + injected bugs. The mutation testing + baseline ladder evaluation design is fully automated and methodologically sound.

**Verifier correction:** The panel conflated "no-humans kills a publication venue" with "no-humans prevents execution." All computation runs on a laptop CPU within stated time budgets. The score should reflect that the constraint limits the publication strategy, not the technical execution. Corrected from 3 to 6.

### Axis 5: Feasibility — 3/10

**Compound failure probability:** Panel consensus ~83–88% for the full project (corrected from Skeptic's 93% which erroneously included dropped gate D7 at P=1.0).

| Gate | P(fail) | Source |
|------|---------|--------|
| D1 (wrapping ≤5× on 4-joint, ≤10× on 7-joint) | 20–25% | Red-Team Attack 3.1; subdivision mitigates |
| D2 (ε < 0.05 on 10-object benchmark) | 40–50% | Red-Team back-of-envelope: ε ≈ 0.06 baseline |
| D3 (≥3× improvement over CP) | 40–50% | Frontier-resolution unproven; volume-subtraction alone gives 1.4× |
| D4 (Lipschitz violations <20%) | 20–25% | 420 potential joint-limit surfaces per scene |
| Unity parser on real scenes | 30–35% | 30–40% of scenes unanalyzable without DSL |
| Frontier-resolution works | 40–50% | "Plausible but unproven" per synthesis |

**Compound survival (top-4 risks):** ~0.55 × 0.55 × 0.75 × 0.65 ≈ 0.15. P(at least one critical failure) ≈ 85%.

**Salvage path compound failure:** ~55–65% (Path A eliminates Unity parser risk, D7, and reduces multi-step risk).

**The 2-month kill-chain as a mitigant:** The Verifier correctly notes that the risk profile is asymmetric — gates D1/D2/D3 fire within 2 months, limiting maximum sunk cost. A 55% failure probability with 2-month bounded loss is fundamentally different from 55% failure probability with 9-month unbounded loss. The panel underweights this.

**Zero code written:** impl_loc=0. Every risk estimate is theoretical. Implementation may surface additional problems.

### Axis 6: Fatal Flaws

| # | Flaw | Severity | Survives Salvage? |
|---|------|----------|-------------------|
| **F1** | Developer study violates no-humans constraint | **FATAL** | ✅ Salvage drops developer study entirely |
| **F2** | Zero demand signal; cannot validate under constraints | **FATAL** | ⚠️ Partially — theory paper doesn't need market demand |
| **F3** | Lipschitz fails for disability populations; κ=10–30% | **NEAR-FATAL** | ✅ κ-completeness reframed as contribution, not limitation |
| **F4** | Smart MC likely dominates on detection rate (untested) | **NEAR-FATAL** | ⚠️ D3 gate tests this; ε-vs-CP is category error |
| **F5** | UIST paper unsubmittable without human study | **FATAL** | ✅ Salvage targets TACAS/FMCAD/ICSE instead |
| **F6** | Frontier-resolution load-bearing but unproven | **NEAR-FATAL** | ⚠️ D3 tests it; fallback is structural framing |
| **F7** | SMT volumetric contribution is 10⁻⁹ (negligible) | **SERIOUS** | ✅ Decision 7 reframes: Tier 1 green regions = verified volume |
| **F8** | theory_bytes=0 in State.json | **MINOR** | ✅ Path bug — approach.json is ~66KB |

**Cross-critique corrections on fatal flaws:**
- F4 ("97% MC detection at 1/20th cost") is an untested assertion, not an established fact. The actual marginal detection rate is an open empirical question answered at gate D3. The 2K-LoC MC comparison omits: spatial verification map, κ-tracking, verified-volume accounting, and coverage certificates. (Verifier RC-3)
- F7 is resolved by Decision 7 (synthesis): Tier 1 affine-arithmetic green regions provide 30–60% symbolically verified volume — far exceeding SMT's negligible contribution. The "sampling-symbolic hybrid" becomes "sampling + interval verification + targeted SMT."

---

## Cross-Critique Highlights

### Skeptic's "Formalism Theater" — Rejected as Strawman, Kernel Retained

The Skeptic charged that the project uses formal methods cosmetically. Cross-critique found this is a strawman of the project's actual claims: the coverage certificate provides spatial localization and population segmentation that CP fundamentally cannot. However, the kernel — that ε_certificate is numerically worse than ε_CP and reviewers will conflate them — is a genuine presentation risk. The paper must lead with certificate *structure* (spatial map, κ-completeness, counterexample generation), not ε *tightness*.

### ε vs. Clopper-Pearson — Category Error

The panel confirms this is a category error exploited by both Auditor and Skeptic:
- **CP ε** bounds the *population failure rate* (fraction of body parameterizations that fail)
- **Certificate ε** bounds *P(an undetected failure exists in unverified parameter-space regions)*

These are not comparable. A CP bound of 10⁻⁶ and a certificate ε of 0.05 can both be correct simultaneously. The certificate's value is the spatial verification map, not a tighter ε.

### κ-Exclusion — Genuine Contribution, Not "Silently Defeating Purpose"

The Skeptic argued κ-exclusion (10–30% of interesting parameter space excluded) "silently defeats the purpose." Cross-critique found: κ is *explicitly reported* in every certificate. A system that says "we guarantee ε < 0.05 for 90% of the target population and here are the 10% we can't guarantee" is more honest than one that says "we found no bugs in 10M samples" (which silently says nothing about adversarial configurations). The explicit exclusion tracking is arguably the project's most intellectually honest contribution.

### Tier 1 Green Regions as Verified Volume (Decision 7) — Best Improvement

Both Auditor and Synthesizer identify Decision 7 as the single most impactful insight from the theory stage. When Tier 1 classifies an element as "definitely reachable" for a body-parameter range, this constitutes symbolically verified volume backed by affine-arithmetic soundness. If 40% of (element, body-range) pairs are green, |V|/|Θ| ≈ 0.40 without any SMT queries — providing 1.67× ε improvement from volume subtraction alone.

---

## Salvage Paths (Synthesizer + Cross-Critique Corrected)

### Path A (Recommended): Domain-General Coverage Certificates → TACAS/FMCAD

**What is built:** Coverage certificate engine (C1 + implementation), affine-arithmetic FK for revolute chains (B1), adaptive stratified sampler, linearized-kinematics SMT encoding. Two case studies: 7-DOF human arm (XR scenario), 6-DOF industrial manipulator (robotics scenario).

**Novel LoC:** 15–22K. **Target venue:** TACAS 2026, FMCAD 2026. **P(acceptance):** 20–30%. **Compound failure:** ~55–65%. **Key risk:** ε tightness — if ε > 0.05 on 10-minute budgets, lead with structural advantages instead.

### Path B (Parallel): Tier 1 Standalone Linter → ICSE Tool Track

**What is built:** Unity scene parser, affine-arithmetic FK engine, Unity editor integration with visual annotations, population-fraction estimation, procedural benchmark suite.

**Novel LoC:** 12–18K. **Target venue:** ICSE Tool Track 2026, ASSETS 2026. **P(acceptance):** 20–30%. **P(delivery):** 75–85%. **Key risk:** False-positive rate if wrapping factor >5× on 7-joint realistic chains.

### Path C (Contingency): Robotics Workspace Certification → ICRA/IROS

**What is built:** Certificate pipeline retargeted to collaborative robot workspace certification (UR5, UR10, Franka Panda). ISO 10218/ISO/TS 15066 compliant workspace partitioning.

**Novel LoC:** 22–35K (corrected upward per cross-critique — includes simulation infrastructure). **Target venue:** ICRA 2026, IROS 2026. **P(acceptance):** 20–30%. **Key risk:** Competition from existing robotics safety tools (COMPAS FAB, MoveIt). Requires new domain expertise.

---

## Mandatory Amendments

### A1: Scope Reduction (BINDING)
The project must commit to exactly ONE salvage path by end of Week 2. Recommended: Path A (domain-general coverage certificates → TACAS) + Path B (Tier 1 linter → ICSE Tool Track) as synergistic pair sharing the affine-arithmetic FK engine.

### A2: D1 Gate — Hard Kill at Month 1 (BINDING)
If 4-joint wrapping factor exceeds 5× or 7-joint wrapping factor exceeds 10× after subdivision (max 6 levels), the linter approach is dead. Fall back to Taylor models as last resort. If Taylor models also exceed 10× — ABANDON linter.

### A3: D2/D3 Gate — Certificate Viability at Month 2 (BINDING)
If ε > 0.10 on 10-object benchmark with 5-minute budget — ABANDON certificate framework. If ε improvement over Clopper-Pearson is < 1.5× — reframe paper around structural advantages (spatial map, κ-completeness), not ε tightness.

### A4: Drop All Human Studies (BINDING)
No developer study, no demand validation survey, no user evaluation. The project stands on automated evaluation alone. Do not claim "future work: developer study" — the paper must be self-contained with automated benchmarks.

### A5: Lead with Certificate Structure, Not ε Tightness (BINDING)
The theory paper must frame the coverage certificate as a *spatial verification map with explicit completeness tracking*, not as "tighter bounds than MC." The κ-completeness metric and dual-ε reporting are the intellectual contributions. The ε number is a secondary result.

### A6: Publication Target Adjustment (BINDING)
Primary: TACAS 2027 or FMCAD 2026 (not CAV — B+ crown jewel is below CAV main track bar). Secondary: ICSE Tool Track (linter standalone). Do not target UIST, CHI, or ASSETS (all require user evaluation under standard reviewing practice).

### A7: Monthly Kill-Chain Reviews (BINDING)
If at any monthly checkpoint the compound probability of remaining gates exceeds 75%, the project terminates.

### A8: Honest Framing of Marginal Value (BINDING)
The paper must explicitly compare against a strong Monte Carlo baseline (frontier-adaptive importance-sampled MC per synthesis Decision 9) and honestly report the marginal detection rate. If MDR < 5%, the paper leads with certification value, not detection value.

---

## What Would Change This Assessment

1. **Demonstrate frontier-resolution works** (D3 gate, Month 2): ≥3× improvement over CP from same sample count would validate the core technical bet. This alone would raise Best-Paper to 5/10 and Feasibility to 5/10.

2. **Evidence of platform-holder interest:** If Meta Accessibility, Apple Accessibility, or Unity Technologies expressed interest in formal verification tooling, Value jumps to 5–6/10.

3. **Regulatory action:** An XR-specific accessibility lawsuit or regulatory guidance would create immediate demand.

4. **Relax no-humans constraint:** Enables developer study (UIST paper viable), demand validation (D7 executable), usability evaluation. Would raise Laptop/No-Humans to 8/10 and open the second publication path.

5. **Show κ ≤ 0.10 on representative scenes** (D4 gate, Month 3): Would defang the most damaging reviewer attack ("your certificate excludes the people who need it most").

---

## Final Verdict

### Full Project: ABANDON (21/50 corrected)

The full project as specified faces three independently fatal flaws (no-humans kills UIST paper, zero demand signal, Lipschitz exclusion of disability populations) and ~85% compound technical failure probability. The theory stage was exceptionally rigorous — nearly 500KB of adversarial cross-critique that discovered its own fatal flaws before a line of code was written. This rigor deserves credit but does not rescue the project.

### Salvage Path: CONDITIONAL CONTINUE (27/50 corrected)

**Conditions:** All 8 amendments above are binding. The 2-month kill-chain (D1/D2/D3) limits maximum sunk cost. Recommended path: domain-general coverage certificates (TACAS/FMCAD) + Tier 1 linter (ICSE Tool Track) as synergistic pair sharing the affine-arithmetic FK engine. Total novel LoC: 20–30K (down from 43–68K). Compound failure: ~55–65%.

**P(any publication):** ~40–50%. **P(best-paper):** ~3–5%. **Kill probability:** ~55%.

The strongest surviving contributions are: (1) the κ-completeness framework for honest verification certificates, (2) Decision 7 connecting abstract interpretation to statistical testing, (3) the dual-ε reporting mechanism, and (4) the Tier 1 linter as the first tool of any kind in an empty niche. These deserve to be pursued — but at reduced scope, with ruthless kill gates, and without the pretense that XR developers are clamoring for formal verification.

---

## Dissent Record

**Synthesizer dissent:** "The consensus undervalues the robotics transfer opportunity (Path C). The mathematical substrate transfers cleanly; ICRA/IROS acceptance rates for formal-methods-meets-robotics are 30–40%. A 28–30/50 salvage score is appropriate."

**Auditor/Skeptic response:** "The robotics path requires new benchmarks, new baselines (COMPAS FAB, MoveIt), and new experimental infrastructure. Until a single robotics workspace certificate is demonstrated, Path C remains aspirational."

**Skeptic partial dissent:** "The corrected Laptop/No-Humans score of 6/10 is too generous. The no-humans constraint doesn't just kill a venue — it eliminates the only validation pathway for the tool's practical utility. Score should be 4–5/10."

**Verifier response:** "The axis measures whether the project CAN run on a laptop with no humans, not whether the results are maximally publishable. The computation works. The evaluation design (mutation testing + baseline ladder) works. 6/10 is correct."

---

*Panel assessment produced by: Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer. Cross-critique and synthesis by team lead. Independent verifier signoff: APPROVED WITH RESERVATIONS (3 corrections applied). 2026-03-08.*
