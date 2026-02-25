# Mathematician's Verification: Proposal 00 — Finite-Width Phase Diagrams

## Team Evaluation Summary

Evaluated by a three-member verification team (Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer) with adversarial cross-critique and independent verifier signoff.

---

## Scores

| Axis | Score | Justification |
|------|-------|---------------|
| **Extreme Value** | **6/10** | Value is real but narrowly theoretical. The practitioner narrative (sweep triage, µTransfer validation) is not credible — the system excludes transformers, SGD, BatchNorm, and uses 200 probe points. The genuine value is for the training dynamics theory community (~400–700 researchers): first computable finite-width phase diagrams, and if scaling exponents work, a quantitative bridge to statistical mechanics. |
| **Software Difficulty** | **6/10** | Novel algorithmic code is modest (~500–800 lines; ~3K LoC total MVP). The 65K LoC claim in the problem statement is disconnected from reality. However, the numerical stability challenges are substantial: ill-conditioned kernels near phase boundaries, stiff ODEs, Lyapunov exponent convergence, and multiplicative fragility across architecture types. Getting 500 lines numerically stable across edge cases is harder than 5000 lines of straightforward code. |
| **Best-Paper Potential** | **5/10** | The ceiling is genuinely high if finite-size scaling exponents work (statistical mechanics × ML universality classes). Best-paper probability: ~6–8%. The most likely outcome is a solid ICML/NeurIPS theory paper, not best paper. Headwinds: transformer exclusion, toy-scale experiments (MNIST/CIFAR at n=2K, width≤1024, depth≤6), small audience relative to best-paper standards. |
| **Laptop-CPU Feasibility** | **5/10** | The prediction system is genuinely laptop-feasible — kernel-space ODE solving on n×n matrices (n≤200) completes in 1–2 hours. This is a real technical achievement. But ground-truth validation requires ~3050 CPU-hours (~3–4 weeks on laptop or 1 day on modest cluster). The proposal should be honest about this. Many theory papers use clusters for experiments; this isn't fatal but shouldn't be hidden. |
| **Feasibility** | **5/10** | The mathematical core (Theorems 1–3, 5) is provable with ~70% confidence. Theorem 4 Part 2 (monotonicity of Lyapunov exponent) is likely false in general — the Skeptic's argument that rapid residual convergence at large γ can re-stabilize the trajectory is persuasive. Top failure modes: perturbative expansion doesn't converge at interesting widths (40%), phase boundaries too fuzzy to detect (35%), frozen-H fails at boundary (30%). |

**Weighted Score: 5.4/10**

---

## Load-Bearing Math Assessment

### Genuinely New and Hard (Load-Bearing)
- **Theorem 2\* (H-tensor for ConvNets):** The spatial averaging decomposition H^{conv} = (1/|S|)Σ_α H^{dense}(α) + R with the CLT-based bound on R via spatial correlation length ξ is genuinely novel. This is the hardest new math in the proposal.
- **Theorem 3\* (H-tensor for ResNets):** The cross-term Ξ_ℓ between skip and branch paths, and the O(1/L) per-block depth scaling, are new. Less hard than Theorem 2\* but substantive.
- **Theorem 4\* (Lyapunov stability boundary):** Reformulating lazy-to-rich as a zero-crossing of finite-time Lyapunov exponents is new in the NN context. The concept draws on standard dynamical systems (Benettin et al. 1980); novelty is in the application and the variational system construction. **Part 2 (monotonicity) is likely false** and should be dropped.
- **Structural Stability Lemma (Frozen-H):** Under-sold. This is essentially a phase boundary persistence theorem (Fenichel-type result). Should be elevated to a named theorem.
- **Lemma 3 (Spatial averaging bound):** Clean CLT for weakly dependent spatial contributions. Required for Theorem 2\*.

### Routine (Necessary Infrastructure)
- Theorem 1\* (1/N expansion): Extension of Dyer & Gur-Ari 2020 to weight-sharing. Bookkeeping, not conceptual breakthrough.
- Theorem 5\* (Perturbative bounds): Standard Grönwall argument. Exponential dependence on γ·T makes bound vacuous in practice.
- Algorithms A, G (NTK, Nyström): Textbook implementations.

### Ornamental (Does Not Drive System)
- Conjecture 1 (ReLU extension): Stated but unproven. Should be a remark, not a numbered conjecture.
- Definition 1.15 (Pseudo-arclength continuation): Textbook method (Keller 1977). Defining it formally adds length without insight.
- ~30% of assumptions are standard regularity conditions that could be stated once.

**Math ratio: ~40% genuinely new, ~30% routine, ~30% ornamental.** The new math is concentrated correctly — H-tensor decompositions are exactly what's needed.

---

## Fatal Flaws

1. **Ground-truth compute budget breaks laptop constraint.** 3050 CPU-hours cannot run on a laptop in any reasonable timeframe. Must acknowledge cluster compute for validation.

2. **Theorem 4 Part 2 (monotonicity) is probably false.** At large γ, rapid residual convergence can re-stabilize the NTK trajectory, creating non-monotone Λ_T(γ). The bisection algorithm (Alg E\*) depends on monotonicity for unique zero-crossing. **Fix: Drop monotonicity, use grid search.**

3. **Self-referential validation.** System predicts kernel drift; ground truth measures kernel drift via KADR. Correlated failure modes are possible. Partially mitigated by multi-indicator conjunction (CKA, weight displacement, linear probe gap) but the circularity concern remains.

4. **n_probe=200 subsampling risk.** Phase boundaries computed on 200 points may not match those on full datasets. No evaluator quantified this risk. **Needs an early gate test: compare boundaries at n=200 vs n=500 vs n=1000.**

---

## Hidden Strengths (Under-Sold)

1. **Finite-size scaling exponents are the real payload.** The prediction Δη/η\* ~ N^{-β} with measurable, architecture-dependent β would establish universality classes for training dynamics — a genuine quantitative bridge to statistical mechanics. This is the best-paper-level result, not the phase diagrams.

2. **Definition concordance experiment.** Comparing three independent definitions of the phase boundary (Lyapunov, KADR-gradient, sigmoid inflection) is methodologically novel and rare in ML theory. If they agree: strong evidence the boundary is real. If they disagree: itself an interesting finding.

3. **Frozen-H structural stability lemma.** A Fenichel-type persistence result for phase boundaries. The constant C encodes geometry of the transition. More fundamental than the phase diagrams it enables.

4. **Strong fallback positions.** 70% probability of ≥6/10 paper via fallback hierarchy: MLP-only + scaling exponents (8/10 conditional) → KADR as unified order parameter (6/10) → pure H-tensor math (5/10).

---

## Verdict: CONTINUE

### Conditions for Continuation

1. **Front-load MLP validation gate (Week 1).** If 1/N expansion doesn't converge (two-loop check ε₃ fails) at N≤256 for 2-layer GELU MLP, **ABANDON**.
2. **Kill sweep triage narrative.** Reframe as pure theory paper centered on finite-size scaling exponents.
3. **Fix Theorem 4 Part 2.** Drop monotonicity claim. Use grid search for boundary detection.
4. **Add n_probe gate test.** Compare phase boundaries at n=200 vs n=500 vs n=1000 for one configuration.
5. **Be honest about compute.** Predictions: laptop. Validation: modest cluster.
6. **Elevate frozen-H to a theorem.** Rename "Phase Boundary Persistence Theorem."
7. **Prioritize:** (1) MLP phase diagrams → (2) scaling exponents β → (3) ConvNet H-tensor → (4) structural stability polish.

### Walk-Away Triggers (Unanimous)
- Two-loop check fails at N ≤ 256 for MLP → **ABANDON**
- AUC < 0.75 on MLP lazy-vs-rich classification → **ABANDON**
- Baseline E (empirical interpolation) matches system within 5% → **PIVOT to pure math paper**

### Publication Probability
- Top venue acceptance (ICML/NeurIPS/ICLR): **25–35%**
- Best paper: **6–8%**
- Expected paper quality: **6.1/10** (thick middle of solid outcomes)

### Risk-Adjusted Assessment
This is a high-ceiling, well-cushioned project. The primary risk is not failure but mediocrity — the most likely outcome is a solid 6/10 theory paper, not a 9/10 breakthrough. The restructuring around scaling exponents raises the ceiling without changing the floor. The fallback positions are unusually strong: even significant partial failure yields a publishable paper.

**The math is load-bearing where it counts.** Theorems 2\*–3\* (H-tensor for structured architectures) and the Lyapunov reformulation (Theorem 4\*, minus the monotonicity claim) are genuinely new and directly enable the computational system. The remaining ~60% of the mathematical content is necessary infrastructure or ornamental, which is an acceptable ratio.

---

*Evaluation by: Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer*
*Synthesis by: Team Lead with adversarial cross-critique*
*Verified by: Independent Verifier — SIGNED OFF*
