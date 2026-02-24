# Review by Yuan Cheng (probabilistic_modeling_researcher)

## Project: PhaseKit — Finite-Width Phase Diagrams for Neural Network Initialization

**Reviewer Expertise:** Probabilistic modeling, Bayesian methods, mean-field theory, uncertainty quantification.

---

### Summary

PhaseKit extends infinite-width mean-field theory with O(1/N²) finite-width corrections, a Bayesian phase classifier, and calibration diagnostics. The grounding.json traceability is exemplary. However, several claims require stronger statistical methodology, and the Bayesian classification framework has fundamental design issues.

### Strengths

The O(1/N²) correction derivation (Theorem 1) is the most substantive contribution. The cross-layer accumulation term C_acc captures error propagation through depth—an effect overlooked by prior O(1/N) analyses. Validation across 358 configurations with 80 Monte Carlo trials per configuration is thorough. The ReLU closed-form verifications (8 tests, exact to 6+ decimals) and chi_2 bifurcation analysis are clean contributions.

### Weaknesses

**1. The Bayesian classifier is not truly Bayesian.** The posteriors (Eqs. 10–12) use an ad hoc Gaussian bump and logistic sigmoids, not posteriors derived from a generative model. The ε(N) = 2/√N critical window width is heuristic—why 2 and not 1.5? No calibration study of this hyperparameter is provided.

**2. The 83.3% accuracy is misleading.** This comes from 18 σ_w × 5 seeds = 90 evaluations—far too small for reliable accuracy estimates. The ground truth itself uses a chi_1 threshold, so the classifier is evaluated against its own theoretical framework, not an independent definition of trainability.

**3. The perturbative convergence radius is not rigorously bounded.** Theorem 4 states validity when L·|χ₁−1|·D/N ≪ 1, but "≪ 1" is informal. The code uses a heuristic clamp at 0.5. A region-of-validity analysis is needed showing where the correction provably improves over uncorrected mean field.

**4. Calibration diagnostics lack independent ground truth.** Phase labels derive from the same mean-field theory being evaluated. ECE and reliability diagrams measure self-consistency, not accuracy. Breaking this circularity requires defining phases operationally (e.g., via training loss convergence).

**5. Statistical reporting is incomplete.** Table 1 reports point estimates without confidence intervals. With 80 trials, standard errors should be provided. The 1.1–6.4× improvement range across 358 configurations likely has substantial heterogeneity—what is the distribution of improvement factors?

### Grounding Assessment

The grounding.json is strong. However, chi_2 values for GELU (9.622) and SiLU (3.902) lack a "code" field—they only reference the data JSON without an independent verification script.

### Path to Best Paper

Replace the ad hoc classifier with a proper generative model. Provide rigorous convergence bounds. Define phase ground truth operationally. Report distributional statistics. Scale experiments to width ≥4096 and depth ≥50.

### Score: 5/10 — Solid engineering, methodologically incomplete.
