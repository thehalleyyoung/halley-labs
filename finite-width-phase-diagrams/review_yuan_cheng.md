# Review: PhaseKit

**Reviewer:** Yuan Cheng (Probabilistic Modeling Researcher)  
**Expertise:** Probabilistic models, Bayesian inference, statistical methodology, uncertainty quantification  
**Score:** 7/10  
**Recommendation:** Weak Accept

---

## Summary

PhaseKit delivers a mean-field framework computing finite-width phase diagrams for neural network initialization with O(1/N)+O(1/N²) variance corrections validated across 358 configurations. The soft phase classification with posterior probabilities and independent calibration study (75% three-class / 89% binary accuracy) represent a serious effort to ground theoretical predictions statistically. The probabilistic methodology is largely sound but requires tightening in several key areas.

## Strengths

**1. Rigorous Moment Expansion with Computable Error Bounds.** The O(1/N)+O(1/N²) correction hierarchy follows classical perturbative moment-closure methodology with a formal O(1/N³) truncation bound |R₃| ≤ σ_w⁸·M₈(q)/N³. Computing the eighth moment E[φ(z)⁸] analytically for ReLU and numerically for smooth activations to make this bound computable is unusual diligence. The error budget is clearly delineated at each perturbative order, and the perturbative validity monitor (|Θ⁽¹⁾/Θ⁽⁰⁾| < 0.5) provides a self-diagnosing convergence check. This level of error accounting exceeds the standard in ML systems papers and reflects genuine probabilistic modeling discipline.

**2. Comprehensive 358-Configuration Validation Design.** The factorial design — 4 activations × 3 depths × 8 σ_w values × 5 widths with 80 Monte Carlo trials per configuration — constitutes a well-structured validation study. The variance improvement ratios (1.1–6.4× at W=32) are measured across this grid, enabling systematic detection of failure modes. The zero-dangerous-error finding across all 358 configurations is a strong statistical result: under a binomial model, P(0 dangerous errors | p_danger ≥ 1%) < 0.03, giving meaningful confidence that catastrophic mispredictions are rare.

**3. Independent Calibration Against Training Outcomes.** The calibration study using 36 configurations with empirical phase assignment based on actual SGD training dynamics (loss ratio criterion after 500 steps, ≥20 seeds per σ_w) closes a theory-to-practice loop that most mean-field papers leave open. The 89% binary trainability prediction accuracy against ground-truth training runs constitutes genuine validation, not just internal consistency checking. The ECE/Brier score diagnostics via the CalibrationDiagnostics module add proper calibration evaluation infrastructure.

**4. Honest Epistemic Accounting Throughout.** The relabeling from "Bayesian" to "soft" classifier and "Soundness Theorem" to "Soundness Conjecture" demonstrates exemplary epistemic discipline. The activation-specific calibration constants (c_relu=1.0, c_tanh=1.2, c_gelu=c_silu=1.8) are presented as heuristic scaling choices, not as principled Bayesian posteriors. The confidence interval construction for χ₁ via `find_edge_of_chaos_with_ci` properly accounts for finite-width fluctuations. This intellectual honesty increases the paper's credibility substantially.

## Weaknesses

**1. Calibration Study Sample Size Limits Statistical Power.** With n=36 configurations, the Wilson 95% CI for 75% three-class accuracy spans [58%, 87%] — nearly 30 percentage points. The 89% binary accuracy yields CI ≈ [74%, 96%]. These intervals are too wide to distinguish good calibration from mediocre performance. More critically, per-class precision and recall are not reported: the 25% three-class error could concentrate entirely in "critical" misclassifications, which are the most operationally relevant for edge-of-chaos initialization. Expanding to ≥150 configurations or reporting stratified error rates would substantially strengthen the claims.

**2. Soft Phase Probabilities Lack Principled Statistical Foundation.** The critical window ε(N, φ) = c_φ·ln(D)/D is justified by a scaling argument about gradient accumulation, not by a probability model. A proper approach would place a prior over the phase boundary location and compute posterior predictive distributions given finite-width fluctuations. The resulting "probabilities" are not calibrated in the frequentist sense — P(chaotic)=0.75 has no operational meaning as a coverage guarantee. The CalibrationDiagnostics module exists but is not used to recalibrate these pseudo-posteriors via Platt scaling or isotonic regression.

**3. Missing Confidence Intervals on Fitted Correction Coefficients.** The O(1/N) and O(1/N²) coefficients are fitted from 80 Monte Carlo trials across 5 width values per activation-depth combination, but no bootstrap CIs or standard errors are reported on the fitted coefficients themselves. If the CI on the O(1/N²) coefficient includes zero, the claim of a meaningful second-order correction is weakened. The "1.1–6.4× improvement" range needs error bars to distinguish genuine improvement from sampling variability.

**4. Moment Closure Sensitivity Analysis Not Conducted.** The Gaussian closure κ₄≈0 is standard but known to degrade near phase transitions where non-Gaussianity peaks — precisely where predictions matter most. The problem statement itself proposes ±50% κ₄ perturbation analysis, yet this experiment is not reported. Without quantifying how phase boundary predictions shift under moment-closure violations, the formal truncation bound is rigorous only under an assumption whose validity is uncharacterized at the critical point.

## Verdict

PhaseKit makes a genuine contribution by bringing structured moment-expansion techniques with formal error budgets to neural network initialization analysis, validated across a substantial 358-configuration grid. The probabilistic methodology is sound in its foundations but statistically underpowered in its calibration claims and missing key sensitivity analyses.

**Score: 7/10** — Strong moment-expansion framework with real utility and honest uncertainty reporting; needs larger calibration studies, correction coefficient CIs, and moment-closure sensitivity analysis to fully substantiate its probabilistic claims.
