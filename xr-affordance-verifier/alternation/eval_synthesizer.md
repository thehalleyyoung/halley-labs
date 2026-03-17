# Scavenging Synthesizer Assessment: Coverage-Certified XR Accessibility Verifier (proposal_00)

**Role:** Scavenging Synthesizer — find the strongest reduced-scope path to publication
**Date:** 2026-03-08
**Input:** 3 prior evaluations (Skeptic 21/50, Mathematician 22/50, Community Expert 25/50), ~500KB theory documents, synthesis with 10 binding decisions

---

## Axis Scores

| # | Axis | Score | Strongest Interpretation |
|---|------|-------|-------------------------|
| 1 | **Extreme & Obvious Value** | **4/10** | Empty niche is real — zero competing XR accessibility tools. First-mover advantage is underweighted by prior panels. Enterprise XR (Boeing, Lockheed Martin surgical sim) already faces Section 508/ADA Title I mandates. The Tier 1 linter solves an immediate pain point for the ~30–50K XR developers who currently rely on manual testing with diverse body types. The value ceiling is capped by small market, but the value *floor* — first tool of any kind in this space — is genuinely nonzero. |
| 2 | **Genuine Difficulty** | **5/10** | The 15–22K difficult LoC (rescoped) centers on three genuine challenges: (1) affine-arithmetic FK wrapping through 7-joint revolute chains with subdivision control, (2) piecewise-Lipschitz partitioning of body-parameter space at joint-limit surfaces, (3) κ-completeness metric computation. These are B+ composition problems — not A-grade open math, but not trivial engineering either. The affine-arithmetic wrapping factor (D1 gate exists because the answer is genuinely uncertain) is a real numerical-analysis challenge. |
| 3 | **Best-Paper Potential** | **4/10** | Decision 7 (Tier 1 interval-arithmetic verdicts credited as verified volume in statistical certificates) is the strongest intellectual contribution. It connects abstract interpretation to parameterized testing in a way I cannot find in prior art from a preliminary search. κ-completeness as a metric for honest coverage reporting is genuinely novel. The ISSTA/FSE framing ("Coverage Certificates for Parameterized Testing") generalizes beyond XR. P(best paper) is low (~2–4%) but the contribution is real enough that a strong presentation at a receptive venue could land. |
| 4 | **Laptop-CPU & No-Humans** | **8/10** | This is the project's strongest axis. ALL computation is CPU-native: affine arithmetic, FK evaluation via Pinocchio, stratified sampling, Z3 on QF_LRA. Tier 1 runs in <2s. Full pipeline in <10min. Peak memory <2GB. No GPU, no human annotation, no human studies needed for the core system. The mutation-testing + baseline-ladder evaluation strategy replaces the developer study entirely. The only deduction: multi-step k=3 is borderline at d_eff=26, and Unity plugin builds require platform-specific toolchains. |
| 5 | **Feasibility** | **5/10** | The termination chain is well-designed: D1 (Month 1, wrapping factor) and D2/D3 (Month 2, ε + CP comparison) test all critical mechanisms within 2 months. Maximum sunk cost before termination: 2 person-months. Corrected ε ≈ 0.022 (Mathematician's strata fix: 3⁵=243, not 3⁷=2,187) meets the hard-pass threshold and substantially de-risks D2. Tier 1 linter has ~85% standalone delivery probability. P(any publication) on the optimal salvage path ≈ 40–55%. |

**Composite: 26/50**

---

## Optimal Salvage Path: "AccessCert" — Two-Track with Shared Engine

The highest-EV path is a **two-track strategy sharing a common affine-arithmetic FK engine**, with independent termination criteria and publication targets.

### Track A: Tier 1 Standalone Linter — "AccessLint-XR" (12–15K LoC)

**What it is:** First-ever spatial accessibility linter for Unity XR. Ingests .unity YAML + .prefab, evaluates reachability of every interactable element across ANSUR-II body parameter ranges using affine-arithmetic forward kinematics with adaptive subdivision. Produces green/yellow/red verdicts with counterexample body types. Runs <2s in Unity Editor.

**Why it survives independently:**
- Zero dependencies on the certificate framework
- Affine-arithmetic FK engine is the only technically uncertain component (D1 gate)
- 85% standalone delivery probability (all three panels agree)
- Fills a completely empty niche — literally the first tool of its kind
- Evaluation: mutation testing on 500+ procedural scenes + 10 real scenes, baseline ladder (lookup table → sphere model → FK model → affine FK)

**Target venue:** ICSE 2027 Tool Track or ASSETS 2027 (if κ-exclusion is properly disclosed)
**P(acceptance):** 25–35%
**Timeline:** Working prototype Month 2, paper-ready Month 4

### Track B: Domain-General Coverage Certificates (10–15K LoC additional)

**What it is:** The κ-completeness framework + Decision 7 insight, extracted from XR-specific context and presented as a contribution to parameterized testing methodology. Core claim: "When an abstract-interpretation pass can prove correctness for a subset V of a continuous parameter space Θ, the remaining space Θ\V requires fewer samples to achieve the same statistical guarantee, yielding tighter coverage certificates than pure sampling." Demonstrated on XR accessibility + 2 toy domains (robotic workspace reachability, dosage-weight safety verification).

**Why it survives:**
- Decision 7 is the panel's consensus "most promising intellectual contribution"
- κ-completeness is genuinely novel — no prior parameterized testing framework explicitly quantifies coverage completeness
- Domain-general framing sidesteps XR-specific market concerns
- The corrected ε ≈ 0.022 (sampling only) → ε ≈ 0.015 (with 30% Tier 1 verified volume) is a concrete 1.5× improvement from Decision 7 alone, before frontier-resolution
- Cross-domain toy examples (robotics, medical) demonstrate generality without requiring full implementations

**Target venue:** ISSTA 2027 or FSE 2027 (testing methodology track)
**P(acceptance):** 15–25%
**Timeline:** Framework prototype Month 3, evaluation Month 4–5, paper-ready Month 5–6

### Shared Infrastructure

- **Affine-arithmetic FK engine** (~5–8K LoC, C++/Python): Used by both tracks. Built Month 1. D1 gate applies to both.
- **Stratified sampling engine** (~3–5K LoC, Python): Latin hypercube + frontier detection. Used by Track B.
- **ANSUR-II population model** (~2K LoC): Body parameter distributions. Used by both.
- **Benchmark infrastructure** (~3–5K LoC): Procedural scene generation + mutation testing.

### What's Cut

| Component | Reason | LoC Saved |
|-----------|--------|-----------|
| SMT/Z3 integration | Volume contribution ≈ 10⁻⁹; frontier-resolution unproven | ~8–12K |
| Multi-step (k > 1) | Curse of dimensionality; certificates too loose | ~5K |
| DSL for interaction annotations | Zero evidence of developer willingness | ~3K |
| Developer study / UIST paper | No-humans constraint | N/A |
| WebXR/OpenXR format support | Scope reduction | ~3K |
| Full PGHA formalism | Unnecessary for Tier 1; unnecessary for domain-general certificate | ~5K |

**Total rescoped: ~25–35K LoC, ~12–18K difficult. Down from 43–68K / 21–37K.**

---

## Cross-Domain Transfer Value

The strongest argument for CONTINUE is that the core contributions transfer beyond XR:

| Domain | What Transfers | Effort to Adapt | Venue |
|--------|---------------|-----------------|-------|
| **Robotic workspace certification** | FK engine + coverage certificates. Robot arm kinematics are structurally identical to human arm. | ~3K LoC (swap kinematic model) | ICRA/IROS |
| **Medical dosage verification** | κ-completeness + Decision 7 framework. Parameter space = patient weight/age/metabolism. | ~2K LoC (toy implementation) | AMIA/toy example in ISSTA paper |
| **Automotive sensor coverage** | Coverage certificates over sensor placement parameters. | Conceptual only (future work) | IV/ITSC |
| **General parameterized testing** | Decision 7 framework is fully domain-general. Any system with continuous parameters + abstract-interpretation-checkable properties. | 0 LoC (is the framework) | ISSTA/FSE |

The robotic workspace transfer is particularly strong: Pinocchio already supports arbitrary URDF models. Swapping the human kinematic chain for a robot arm is a configuration change, not a reimplementation. A single Track B paper could demonstrate XR + robotics + medical as three instantiations.

---

## Expected Value Calculation

### Downside (Termination Chain)

| Gate | Month | P(terminate) | Sunk Cost | Expected Sunk |
|------|-------|--------------|-----------|---------------|
| A9 (literature review) | Week 1 | 15% | 0.25 months | 0.04 |
| D1 (wrapping factor) | Month 1 | 12% | 1 month | 0.12 |
| D2 (ε > 0.10) | Month 2 | 8% | 2 months | 0.16 |
| D3 (certificate value) | Month 2 | 15% | 2 months | 0.30 |

**Expected sunk cost if terminated: 0.62 person-months** (weighted by termination probability × timing).
**Maximum sunk cost before critical gate: 2 months.**

### Upside (Publication Outcomes)

| Outcome | P(outcome) | Value | Expected Value |
|---------|------------|-------|---------------|
| Track A (ICSE Tool) accepted | 25% | 1.0 pub | 0.25 |
| Track B (ISSTA/FSE) accepted | 18% | 1.0 pub | 0.18 |
| Both accepted | 8% | 2.0 pubs | 0.16 |
| Track A only (B stopped at D3) | 17% | 1.0 pub | 0.17 |
| Workshop paper fallback | 15% | 0.5 pub | 0.075 |
| Total failure | 17% | 0 | 0 |

**E[publications] ≈ 0.68 pub-equivalents.**
**P(at least one publication) ≈ 45–55%.**
**P(best paper at any venue) ≈ 2–4%.**

### ROI Assessment

- **Expected cost:** 4.5 months (weighted by termination probability distribution)
- **Expected output:** 0.68 pub-equivalents + useful open-source artifact (Tier 1 linter, ~75% delivery probability)
- **ROI comparison:** A typical 6-month research project with P(pub) ≈ 30% yields 0.30 pub-equivalents. This project's 0.68 expected pubs in 4.5 expected months is **above the baseline research ROI**, primarily because the 2-month termination chain prevents the worst case (6 months for nothing).

---

## Verdict: **CONDITIONAL CONTINUE**

**Confidence: 60%**

The optimal salvage path produces positive expected value. The termination chain bounds downside to ≤2 months. Decision 7 is a genuinely novel intellectual contribution that transfers across domains. The Tier 1 linter fills a completely empty niche with 85% delivery probability.

### Conditions for CONTINUE (all binding)

1. **A9 clears within Week 1.** Literature search for "abstract-interpretation verdicts as verified volume in statistical testing" across ISSTA/FSE/ICSE/CAV 2018–2025. If prior art found: re-evaluate Track B. Track A proceeds regardless.

2. **D1 passes at Month 1.** Wrapping factor ≤ 10× on 7-DOF chains after subdivision. If fail: ABANDON both tracks (shared FK engine is dead).

3. **D2 passes at Month 2.** Empirical ε ≤ 0.10 on 10-object benchmark. With corrected strata (3⁵=243), expected ε ≈ 0.022 — this should pass comfortably. If fail: ABANDON Track B, continue Track A.

4. **External gate authority enforced.** An external reviewer receives D1/D2/D3 results and has binding authority to terminate. The project's demonstrated optimistic reframing tendency (5× ε target relaxation, repeated scope narrowing) makes self-enforcement insufficient.

5. **SMT stays cut unless frontier-resolution validates empirically.** Do not re-add Z3 integration speculatively. If D3 shows certificate improvement <2× over CP with Tier 1 verified volume alone, frontier-resolution is the only mechanism that could help — prototype it at Month 3 as a stretch goal, not a dependency.

6. **No scope creep.** Multi-step (k>1), DSL, WebXR/OpenXR support, and developer studies remain cut. The two-track strategy succeeds by being focused, not ambitious.

### What Would Flip to ABANDON

- D1 fail (wrapping catastrophic) → both tracks dead → ABANDON
- A9 finds exact prior art AND D3 fails → no novel contribution → ABANDON
- Month 3 with zero working code → execution failure → ABANDON
- External reviewer exercises termination authority → respect it

---

*Assessment by: Scavenging Synthesizer. Composite 26/50 (V4/D5/BP4/L8/F5). The project's best path is narrower than originally envisioned but has genuine intellectual content (Decision 7, κ-completeness) and practical value (first XR accessibility tool). The termination chain makes this a bounded gamble with positive expected value.*
