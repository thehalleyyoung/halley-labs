# Theory Evaluation: Deep Mathematician Assessment

**Proposal:** proposal_00 — The Cognitive Regression Prover: A Three-Layer Usability Oracle with Incremental Formal Guarantees  
**Area:** area-042-human-computer-interaction  
**Evaluator:** Deep Mathematician (verification team: Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer)  
**Method:** Three-expert adversarial panel with independent proposals, cross-critiques, adversarial challenges, synthesis, and independent verification signoff  
**Date:** 2026-03-04

---

## Evaluation Methodology

Three independent experts scored this proposal, then engaged in direct adversarial challenges:

- **Independent Auditor:** Evidence-based scoring with mathematical scrutiny of all four theorems
- **Fail-Fast Skeptic:** Maximally adversarial attack on value, novelty, and feasibility claims
- **Scavenging Synthesizer:** Identified strongest elements and proposed reframings to rescue weak areas

Disagreements were resolved through direct argumentation (not averaging). An independent verifier confirmed internal consistency, evidence quality, and verdict justification. **Verification status: VERIFIED.**

---

## Load-Bearing vs Ornamental Math Assessment

This proposal's core mathematical evaluation, as a deep mathematician, centers on whether the math *drives* the system — whether it is the reason the artifact is hard to build and the reason it delivers extreme value. Ornamental math that dresses up engineering is worthless.

### Theorem 1: Paired-Comparison Tightness — ★★★★☆ LOAD-BEARING (with critical qualification)

$$|(\hat{C}_B - \hat{C}_A) - (C_B - C_A)| \leq 2\varepsilon \cdot L_R$$

**Status:** Proof sketch exists; O(ε) scaling established; tight constant TBD.  
**Assessment:** Genuinely load-bearing. Error cancellation under shared bisimulation abstraction is a real result — not trivial, not previously formalized in the MDP/HCI context. The insight that a loose absolute bound (TV ≤ 0.53) becomes a tight differential bound (≤ 20ms) is the mathematical core of the consistency-oracle claim.

**Critical qualification (unanimously flagged):** For UIs differing on k transitions, error degrades to O(k·ε). At k=100, ε=0.005: error ≤ 2s ≈ the task duration itself. The theorem is **tightest when least needed** (trivial changes) and **loosest when most needed** (major refactors). The k-distribution for real PRs is unknown. This must be empirically characterized or the theorem must be explicitly restricted to small-change regressions.

**Verdict:** Prove this first. It is the paper — but only if k is empirically shown to be small for typical PRs.

### Theorem 2: Parameter-Independence — ★★☆☆☆ ORNAMENTAL (useful observation, not a theorem)

**Status:** Proven (trivially).  
**Assessment:** All three experts unanimously agree: this is a tautology dressed as a theorem. "If f is monotone increasing and x₂ > x₁, then f(x₂) > f(x₁)" is the definition of monotonicity, not a research contribution. The proposal's own assessment: "Proof difficulty: Easy. Follows directly from monotonicity of Fitts'/Hick's laws."

The *framing* — recognizing that monotonicity enables parameter-free CI/CD verdicts — is a useful practical observation. But calling it a "theorem" alongside genuine mathematical results inflates the contribution dishonestly.

**Verdict:** Keep as a practical corollary. Stop calling it a theorem. Its value is engineering, not mathematics.

### Theorem 3: Cognitive Fragility & Cliff Location — ★★★★★ MOST NOVEL (the diamond)

$$F(M) = \max_{\beta \in \mathcal{B}_{\text{human}}} \mathbb{E}_{\pi_\beta}[C(\tau)] - \min_{\beta \in \mathcal{B}_{\text{human}}} \mathbb{E}_{\pi_\beta}[C(\tau)]$$

**Status:** Cliff location proven (straightforward). Fragility decomposition sketched.  
**Assessment:** Unanimous convergence across all three experts: this is the most genuinely novel contribution. Comparing a UI to *itself* across the human capacity space sidesteps the entire calibration/validation circularity problem. The cliff-location theorem — analytically identifying β* values where softmax policies undergo phase transitions via Q-value crossings — is compact, provable, and computationally enabling (reduces 200 MDP solves to ~50 targeted evaluations).

**Why this is the deepest math:** It defines a new computable quantity (cognitive fragility), connects it to an actionable detection algorithm (cliff location), and sidesteps the hardest objection to the entire framework (evaluation circularity). This is load-bearing math of the highest quality: it simultaneously makes the system *possible* and *valuable*.

**Currently under-exploited.** The proposal treats fragility as a Layer 2 add-on. It should be co-lead alongside paired comparison. The "Chaos Monkey for usability" framing (from abandoned Approach C) would dramatically improve the elevator pitch.

**Verdict:** Elevate to co-lead contribution. This is the idea most likely to open a new research direction.

### Theorem 4: Ordinal Soundness of Cost Algebra — ★★☆☆☆ CONJECTURED (aspirational, not load-bearing)

**Status:** Conjectured. Original proof had dimensional mismatch (time vs bits). Restated ordinal version has incomplete proof sketch.  
**Assessment:** Universal agreement: demote to future work. The original proof was *false as stated* (dimensional error). The restated version is unproven with 35% self-assessed failure probability. If it fails, Layer 3's formal guarantees evaporate. But Layers 1–2 are unaffected.

**Verdict:** State the conjecture, provide empirical evidence, move on. The paper does not need this theorem.

### Summary: Mathematical Load-Bearing Status

| Theorem | Load-Bearing | Novel | Proven | Recommendation |
|---------|-------------|-------|--------|----------------|
| 1: Paired-Comparison | ★★★★★ | ★★★★ | Sketch | Prove first; qualify k-dependence |
| 2: Parameter-Independence | ★★★ | ★★ | Trivially | Keep as observation, stop calling it a theorem |
| 3: Fragility/Cliff | ★★★★★ | ★★★★★ | Partially | Elevate to co-lead |
| 4: Cost Algebra | ★★ | ★★★ | Conjectured | Demote to future work |

**Overall mathematical assessment:** One genuinely novel contribution (fragility/cliff), one competent application of known principles (paired comparison), one trivial observation dressed as a theorem (parameter-independence), and one unproven conjecture (cost algebra). This is not mathematical poverty, but it is not mathematical wealth. The novelty concentrates in Theorem 3, which is currently under-exploited. A reframed paper leading with fragility alongside paired comparison has legitimate, if not overwhelming, mathematical depth.

---

## Scores

| Dimension | Score | Key Evidence |
|-----------|-------|-------------|
| **Extreme Value** | **6/10** | Real gap between accessibility linters and human usability testing. But value is conditional on unexecuted retrospective validation (30% self-assessed failure risk). The 50–70% parameter-free coverage claim is fabricated — stated without any data, dataset, or citation. Fragility adds genuinely new capability but requires Layer 2 (months 3–6). The "desperate need" framing is overblown: design-system teams manage via design review, not automated oracles. |
| **Genuine Software Difficulty** | **6/10** | Research contributions (bisimulation, paired-comparison proof, fragility computation, MDP reduction) are genuine but concentrated in high-risk Layers 2–3. The guaranteed deliverable (Layer 1) is difficulty 4–5 by the proposal's own admission: interval arithmetic over Fitts'/Hick's laws. Two of four theorems are self-described as "easy." Probability-weighted expected difficulty ≈ 6. |
| **Best-Paper Potential** | **5/10** | Novel framing (usability regression as incremental formal verification). Fragility concept is genuinely interesting. But zero empirical results, zero worked examples, incomplete proofs, and one conjectured theorem. Recent CHI/UIST best papers universally require extensive human data or working systems — this has neither. Three small results rather than one stunning contribution. Could rise to 7 with the proposed reframing + one worked example + proofs. |
| **Laptop-CPU Feasibility & No Humans** | **8/10** | Excellent CPU-only design: no ML training, no GPU dependency, accessibility-tree-native, small MDPs (<1K states). Z3/CBC CPU-optimized. Zero-human runtime. Minor gaps: accessibility-tree reconstruction for retrospective validation is hand-waved; ≤10K post-bisimulation state bound is unsubstantiated; evaluation may require minimal human input (5 developers, 15-minute study) as validation fallback. |
| **Overall Feasibility** | **6/10** | Incremental architecture is the saving grace — each layer delivers standalone value. But compound risk is high (84% probability at least one of five risks triggers). Layer 1 timeline (6–8 weeks) is aggressive given cross-browser normalization complexity. Retrospective validation is on critical path and underscoped. Layer 2 requires three simultaneous research contributions. |

**Composite: 31/50**

---

## Fatal Flaws

1. **Retrospective validation may be infeasible (CRITICAL).** CogTool datasets (2006–2014) lack accessibility trees. The cited published studies (Findlater & McGrenere 2004, Gajos et al. 2010) involve visual-spatial manipulations — the exact class the proposal disclaims. With n=10 UI pairs, Kendall's τ has 95% CI ≈ ±0.4, giving essentially zero statistical power to distinguish τ=0.6 from τ=0.4. *Mitigation:* Pivot to modern open-source UI pairs (Material UI v4→v5) with extractable accessibility trees.

2. **50–70% parameter-free coverage is fabricated (SEVERE).** This number appears in no literature, no dataset, no pilot study. The Skeptic's devastating observation: the parameter-free regression types (more options, smaller targets, deeper navigation) are precisely those **already trivially detectable** by existing accessibility linters or simple DOM diffing. If so, Layer 1 reduces to "an accessibility linter with Fitts'/Hick's labels attached."

3. **Evaluation circularity persists (SEVERE).** The system defines "regression" as "cost increase under our model," then tests whether it detects cost increases under its model. The retrospective validation is supposed to break this circle, but per Flaw 1, it may be infeasible. Fragility analysis partially mitigates (self-referential metric), which is why the reframing is strategically critical.

4. **k-transition degradation may make Theorem 1 vacuous (MODERATE).** The paired-comparison bound degrades as O(k·ε). For non-trivial structural changes (k > 50–100), error approaches task duration. The theorem is tightest when least needed and loosest when most needed. k-distribution for real PRs is unknown.

5. **Theorem 4 might be false (MODERATE, contained to Layer 3).** Original proof had dimensional mismatch. Restated version is unproven. Impact contained if Layer 3 is demoted to future work (unanimous recommendation).

---

## VERDICT: **CONTINUE** (conditional)

All three independent experts converged on CONDITIONAL CONTINUE. The independent verifier confirmed this verdict is justified and evidence-based.

### Conditions (Week 1–2, non-negotiable):

1. **Validate the 50–70% claim empirically.** Classify 50+ usability issues from Material UI, Ant Design, or React-Bootstrap issue trackers. If parameter-free coverage < 30%, restructure around Layer 2 (fragility) as core contribution.

2. **Name 5 specific UI pairs for retrospective validation** with confirmed accessibility-tree availability. Not abstract dataset references — specific applications, specific versions, confirmed extractability. If infeasible, design a minimal human-validation fallback (≤10 participants, ≤30 minutes).

3. **Characterize k-distribution for real PRs.** Analyze 20+ design-system PRs and report the number of differing transitions. If median k > 20, prominently qualify Theorem 1's scope.

4. **Adopt the Synthesizer's reframing.** Lead with cognitive fragility ("Chaos Monkey for usability"), not cost regression. This makes the primary claim immune to evaluation circularity.

5. **Demote Layer 3 (bisimulation) and Theorem 4 (cost algebra) to future work.** Sharpen to a two-layer, three-result paper.

**If conditions 1–2 fail by Week 2: ABANDON.** The proposal's survival depends on empirical grounding that currently does not exist.

### What Would Most Improve This Proposal

**Produce one end-to-end worked example on a real UI pair.** Take Material UI v4→v5 (or any real design-system version bump), extract accessibility trees, run the parameter-free analysis, compute the fragility metric, and show the full output. One vivid, concrete demonstration would simultaneously validate the accessibility-tree pipeline, ground the coverage claims, test the fragility concept, produce a compelling figure, and reveal whether the system produces useful or vacuous results. The fact that this hasn't been done — despite being achievable in days — is the single most concerning signal about the proposal's maturity.

---

## Theory Score Assignment

Based on the mathematical assessment:
- One genuinely novel contribution (fragility/cliff detection) — sufficient for a solid paper, not sufficient for best paper alone
- One load-bearing but not surprising formal result (paired-comparison) — sound but depends on k empirics
- One trivial observation dressed as a theorem (parameter-independence) — useful engineering, not math
- One conjectured and potentially false result (cost algebra) — should be cut
- Zero empirical validation of any mathematical claim

**Theory score: 6.0/10**

The score reflects genuine mathematical content (fragility is novel and load-bearing; paired-comparison is competent) tempered by the absence of any empirical grounding, the triviality of one claimed contribution, the conjectural status of another, and the critical k-transition qualification on the flagship theorem. With the proposed reframing (fragility as co-lead), empirical validation, and proof completion, this could rise to 7.5–8.0.
