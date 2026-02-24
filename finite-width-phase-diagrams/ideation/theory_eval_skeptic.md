# Skeptic Verification: proposal_00 — Finite-Width Phase Diagrams

**Stage:** Verification (post-theory development)
**Date:** 2026-02-22
**Method:** Claude Code Agent Teams — 3 independent reviewers + adversarial cross-challenge + synthesis

## Team Composition & Process

| Role | Task | Method |
|------|------|--------|
| Independent Auditor | Evidence-based scoring, challenge tests | Independent proposal |
| Fail-Fast Skeptic | Aggressively reject under-supported claims | Independent proposal |
| Scavenging Synthesizer | Salvage value, identify minimum viable paper | Independent proposal |
| Lead Synthesizer | Cross-challenge, resolve disagreements | Adversarial synthesis |

**Process:** Independent proposals → Adversarial critiques → Cross-challenge resolution → Consolidated verdict with verification signoff.

---

## PROPOSAL SUMMARY

Build the first integrated system computing finite-width phase diagrams predicting lazy-to-rich training regime transitions for neural networks. Core pipeline: compute NTKs at calibration widths → extract 1/N corrections → integrate kernel-residual ODE → locate phase boundaries via Lyapunov exponent zero-crossing → track boundaries via continuation. Claims novel H-tensor derivations for ConvNet/ResNet. Self-rates at 70%/35%/10% across three tiers. Targets AUC > 0.90, <15% boundary error, all on CPU.

---

## THREE PILLARS EVALUATION

### Pillar 1: Extreme Value — Score: 5/10

**Consensus finding:** The value proposition is real but narrow. The primary audience (NTK/mean-field/µP theorists) numbers ~200–500 active researchers. Transformer exclusion eliminates the dominant architecture class, gutting the practitioner value proposition. The "30–50% sweep reduction" claim is asserted without evidence. The Synthesizer argues H-tensor derivations for ConvNets/ResNets are independently valuable math — this is true but doesn't constitute "extreme and obvious" value.

**Challenge test:** If a one-line heuristic baseline (predict rich if γ = η·N^{-(1-a-b)} > C_fit) achieves AUC > 0.85, the 50K+ line system delivers marginal value over a trivial alternative.

### Pillar 2: Genuine Software Difficulty — Score: 7/10

**Consensus finding:** All three reviewers agree the engineering is legitimately hard. The H-tensor derivations have no reference implementation, the kernel-residual ODE coupling is non-trivial, and correct composition of multiple mathematical backends is multiplicatively complex. However, each individual piece (autodiff NTK, ODE solving, Nyström) is well-understood. Difficulty is in integration, not invention.

### Pillar 3: Best-Paper Potential — Score: 4/10

**Consensus finding:** The proposal extends Dyer & Gur-Ari (2020) to new architectures — useful but incremental. The "universality class" narrative (architecture-dependent critical exponents) is self-rated at 10% probability — too speculative to anchor a best-paper argument. The Lyapunov reformulation is interesting but the monotonicity proof gap (see Fatal Flaws) undermines the theoretical contribution. No transformers means limited reviewer enthusiasm at NeurIPS/ICML. The Skeptic notes: NTK-adjacent papers face real reviewer fatigue.

**Upgrade condition:** If architecture-dependent critical exponents differ by >2σ across families AND match experiment within 10%, score jumps to 7. But this is a 10% probability event.

### Additional Dimensions

**Laptop-CPU Feasibility: 3/10 (original) → 7/10 (restructured)**

The Skeptic's arithmetic wins this debate decisively:
- ~4000 training runs × ~15 min each = 1000 CPU-hours = **42 days single-core / 5+ days 8-core**
- The proposal's "12–24 hours" claim is fiction; even the revised "60–80 hours" is optimistic
- PCA of CIFAR-10 to 100 dims destroys spatial structure, invalidating ConvNet-specific evaluation
- **Restructured version** (MLP + Fashion-MNIST/MNIST only, 2K subsample, ~800 runs) reduces to ~150–200 CPU-hours → feasible in 2–3 days

**Overall Feasibility: 4/10 (original) → 6/10 (restructured)**

The three-tier structure (70%/35%/10%) is an admission that the core math contribution has a coin-flip chance of working. If only Tier 1 succeeds, the system is an over-engineered width interpolator (the Skeptic's "Baseline E in disguise" argument). The restructured version (base tier using known MLP math at ~92%, core tier with ConvNet H-tensors at ~65%) has an acceptable risk profile.

---

## FATAL FLAWS (Consolidated, Ranked)

### 1. CRITICAL — Lyapunov Exponent Monotonicity Unproven

All three reviewers flagged this. The bisection algorithm (Algorithm 5) for phase boundary detection **requires** Λ_T(γ) to be monotonically increasing. The "proof" in Appendix A admits it cannot handle non-commutativity of the non-autonomous system and invokes a comparison theorem that doesn't apply to the indefinite off-diagonal perturbation structure. If Λ_T is non-monotone, bisection converges to spurious crossings or misses the real boundary entirely.

**Fix:** Replace bisection with a full γ-sweep (~20x more compute). Drop the monotonicity claim. Report the sweep results and note monotonicity as an empirical observation, not a theorem.

### 2. HIGH — KADR Evaluation Circularity

Both Auditor and Skeptic independently identified this. The system predicts KADR via perturbative kernel ODE. Ground truth uses KADR as 1 of 4 indicators. The phase boundary is defined as steepest-gradient of KADR in **both** prediction and evaluation. The "methodological independence" argument is inadequate — the system is being evaluated on whether its ODE approximates gradient descent, not whether phase boundaries are predictive.

**Fix:** Run evaluation without KADR in ground truth. Use only CKA + weight displacement + linear probe gap as ground truth indicators. If AUC drops below 0.80, the system is measuring ODE accuracy, not phase prediction accuracy.

### 3. HIGH — Tier 1 Without H-Tensor ≈ Width Interpolation

If the H-tensor derivation fails (65% probability in original, 35% in restructured), Tier 1 reduces to: train at 5 calibration widths, fit Θ(N) = Θ^(0) + Θ^(1)/N by regression, integrate ODE. Baseline E (linear interpolation from the same 5 widths) does essentially the same thing. The ODE provides some extrapolation structure, but near the phase boundary where the 1/N expansion breaks down, both approaches degrade similarly.

**Fix:** Pre-register minimum acceptable AUC improvement over Baseline E (e.g., ≥0.10). If not met, report as negative result about ODE vs interpolation.

### 4. HIGH — ReLU Excluded from All Theorems

Every theorem requires C³ activations (GELU, Swish, Softplus). ReLU — the activation practitioners use — is relegated to Conjecture 3.12 with no proof strategy. The Gaussian moment closure κ₄=0 is wrong for ReLU (half-Gaussian pre-activations). The system's validated domain has approximately zero overlap with the practitioner domain.

**Fix:** Commit to GELU-only scope with honest framing. Stop claiming general applicability.

### 5. MODERATE — Frozen-H Self-Defeating Near Boundary

The frozen-H approximation (H = H(t=0)) is justified by structural stability when kernel drift is small. But the phase boundary is defined as where kernel drift *begins* — the approximation hypothesis is violated exactly at the transition. The validity parameter ε is not controllable a priori.

### 6. MODERATE — CPU Budget 5–10x Underestimated (Original)

Original claims 12–24 hours; realistic estimate is 300–1000+ CPU-hours. The restructured version at ~150–200 hours is feasible but still requires multi-day runs, not "minutes."

### 7. MODERATE — Prior Art Thinner Novelty Than Claimed

The ConvNet H-tensor (Theorem 3.3) is described as a "hypothesis, not a theorem" with ~50% failure probability. If it reduces to "standard weight-sharing index contraction" (as the Skeptic argues), the novelty budget shrinks to the Lyapunov reformulation alone — a theoretical reframing, not a new capability.

---

## SCORES SUMMARY

### Original Proposal

| Dimension | Score | Justification |
|-----------|-------|---------------|
| Extreme Value | 5/10 | Real but narrow audience; no transformers; practitioner value unsubstantiated |
| Software Difficulty | 7/10 | Legitimately hard integration of novel math; no reference implementations |
| Best-Paper Potential | 4/10 | Incremental extension of Dyer & Gur-Ari; NTK reviewer fatigue; speculative universality claims |
| CPU Feasibility | 3/10 | 5–10x underestimated compute; PCA destroys ConvNet evaluation |
| Overall Feasibility | 4/10 | 65K LoC scope + 30% base-tier failure risk; monotonicity gap; math risk at the core |

### Restructured Proposal (per Synthesizer)

| Dimension | Score | Justification |
|-----------|-------|---------------|
| Extreme Value | 7/10 | H-tensors for ConvNets are genuinely new math; focused tool fills real gap |
| Software Difficulty | 6/10 | ~20K LoC, JAX-heavy; main risk is numerical stability of H-tensor extraction |
| Best-Paper Potential | 5/10 | Strong theory+systems combo but toy-scale networks on toy datasets limit excitement |
| CPU Feasibility | 7/10 | Kernel-space formulation genuinely CPU-friendly; ~150–200 hours feasible |
| Overall Feasibility | 6/10 | Base tier at ~92%; ConvNet tier at ~65%; eliminated most engineering risk |

---

## VERDICTS

### Original Proposal: **ABANDON**

The original scope (65K LoC, 7 subsystems, computation-graph IR, three backends, full UQ) is a software product masquerading as a research paper. The CPU budget is 5–10x underestimated. The monotonicity proof gap undermines the algorithmic pipeline. The three-tier probability structure (70%/35%/10%) means a 30% chance the base tier fails and a 65% chance the novel math doesn't materialize. The practitioner value proposition collapses without transformers. This is a 6–12 month engineering project with a coin-flip on the math, not a paper.

### Restructured Proposal: **CONTINUE** (conditional)

The restructured version (~20K LoC, focused on H-tensors + kernel ODE + phase boundary detection, MLP base with ConvNet stretch) is a tractable project with genuine novelty. The base tier using known MLP math is ~92% feasible. The ConvNet H-tensor derivation is the real gamble but is independently publishable even if the phase diagram tool underperforms.

---

## MANDATORY CONDITIONS FOR CONTINUATION

1. **Replace bisection with full γ-sweep.** Drop monotonicity claim. Accept 20x compute cost.
2. **Add KADR-free ground truth evaluation.** Evaluate with CKA + weight displacement + linear probe gap only. Report both KADR-included and KADR-excluded AUC.
3. **Honest CPU budget.** State 150–200 hours for full evaluation. Plan for multi-day runs.
4. **Gate on H-tensor validation.** Before building the phase diagram system, implement ConvNet H-tensor for a single 3×3 conv layer and verify against finite-difference H at width 512. If disagreement >5% Frobenius norm, stop and debug.
5. **Drop ReLU claims.** Commit to GELU/Swish/smooth activations. Frame honestly.
6. **Pre-register Baseline E margin.** Define minimum acceptable improvement over width interpolation. If not met, report as negative result.
7. **Run pilot first.** 2-layer ReLU MLP on MNIST-2K end-to-end. If KADR steepest-gradient boundary doesn't match ground truth within 15%, reassess.

---

## WHAT SURVIVES (For Implementation)

If continuing with the restructured proposal, the implementation priority order is:

1. **Pilot validation** (1 week): 2-layer MLP, known H-tensor, MNIST-2K. Gate: boundary within 15%.
2. **H-tensor gate** (2 weeks): ConvNet H-tensor derivation + finite-difference validation. Gate: <5% error.
3. **MLP phase diagrams** (3 weeks): Full MLP results with ablations. This is the base paper.
4. **ConvNet phase diagrams** (3 weeks): If H-tensor gate passed. This upgrades the paper.
5. **Cross-architecture comparison** (2 weeks): If both MLP and ConvNet work. This is the stretch contribution.

Total: ~11 weeks for full scope, with exit ramps at weeks 1, 3, and 6.

---

*Verification signoff: All three reviewers independently reached CONTINUE. Cross-challenge resolved 5 disagreements. Consolidated verdict downgrades original to ABANDON (scope/feasibility) but endorses restructured version with 7 mandatory conditions. The core intellectual contribution (H-tensor derivations + Lyapunov phase boundary) is worth pursuing if properly scoped.*
