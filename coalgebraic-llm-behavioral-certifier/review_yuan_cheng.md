# Review: CABER — Coalgebraic Behavioral Auditing of Foundation Models

**Reviewer:** Yuan Cheng
**Persona:** Probabilistic Modeling Researcher
**Expertise:** Probabilistic modeling, uncertainty quantification, Bayesian inference, calibration, distributional robustness

---

## Summary

CABER applies coalgebraic automata-learning methods to audit black-box LLM behavior, learning finite-state behavioral automata from API queries and quantifying divergence via Kantorovich bisimulation distance. The Bayesian analysis is the paper's strongest probabilistic component—the Beta-Binomial posterior for divergence rates is correctly applied with credible intervals rather than point estimates. However, the probabilistic framework exhibits several critical deficiencies: the PAC bounds central to Theorem 1 (PCL* convergence in Õ(2^β · n₀/ε²) queries) are vacuous at operating sample sizes, the graded satisfaction scores have calibration error of 0.28–0.73 rendering quantitative probabilistic reasoning unreliable, and conventional statistical baselines (chi-squared, MMD) achieve comparable divergence detection power—raising fundamental questions about the marginal value of the coalgebraic probabilistic machinery.

## Strengths

1. **Sound Bayesian posterior analysis.** The Beta-Binomial posterior for divergence rate (P(rate > 25%) = 97.3%, 95% HPD [25.0%, 69.9%]) is correctly constructed using a conjugate prior. Reporting full posterior distributions with highest-posterior-density intervals rather than p-values or point estimates reflects mature statistical practice. The choice of Beta(1,1) uniform prior is appropriately non-informative given the exploratory nature of the study.

2. **Rigorous baseline battery.** Comparing against chi-squared tests, maximum mean discrepancy (MMD) with RBF kernel, KL divergence estimation, token-frequency counting, random sampling baselines, and simplified ALERGIA-based PDFAs provides a thorough probabilistic contextualization. That 7/15 prompts yield divergent behavior across system-prompt configurations is confirmed by multiple independent statistical methods, lending credibility to the core finding.

3. **Transparent PAC bound analysis.** The corrected query complexity bound Õ(2^β · n₀/ε²) from Theorem 1—which the authors themselves corrected from the originally stated O(β · n₀)—is honestly analyzed: at ε = 0.05, δ = 0.05, the required ~143K samples vs. the 54–90 actually collected makes the PAC guarantee vacuous. Rather than hiding this, the paper recommends Bayesian posteriors as the operative uncertainty quantification. This combination of theoretical honesty and practical alternative is commendable.

4. **Multi-configuration factorial design.** The 3 system-prompt configurations × 15 prompts × 5–6 trials design provides structured exploration with replication. The analysis of between-configuration variance vs. within-configuration variance is a sensible decomposition of behavioral variability sources.

5. **Kantorovich metric grounding.** The use of Kantorovich (Wasserstein-1) distance for bisimulation quantification, rather than an ad-hoc distance, connects to the established optimal transport literature. The behavioral functor's graded coalgebraic structure is compatible with the Kantorovich lifting, providing a principled probabilistic distance metric with known mathematical properties.

## Weaknesses

1. **PAC bounds are vacuous at operating scale and the query complexity correction is severe.** Theorem 1 claims PCL* convergence in Õ(2^β · n₀/ε²) queries. The original paper stated O(β · n₀), which was corrected to O(2^β · n₀)—an exponential blowup in the alphabet-size parameter β. At β ≈ 4–5 behavioral atoms, this means 16–32× more queries than originally claimed. With operating sizes of 54–90 samples against the ~143K required, the PAC guarantee provides a confidence interval wider than the entire parameter space. This is not merely a gap—the theoretical convergence rate is operationally irrelevant, and the exponential correction suggests the underlying PAC analysis may have additional issues.

2. **Calibration error of 0.28–0.73 destroys the graded probabilistic semantics.** The framework's key probabilistic innovation over binary testing is graded satisfaction in [0,1]. Calibration error of 0.28 at the low end is concerning; at 0.73 it means the graded values are essentially uncorrelated with true satisfaction probabilities. This undermines every downstream probabilistic computation that depends on these values: the QCTL_F model-checking results that consume graded satisfaction as inputs, the bisimulation distance computations that use graded state-match scores, and any threshold-based decision making. Without calibration, a reported satisfaction degree of 0.8 might correspond to a true probability anywhere in [0.07, 1.0]. The framework should apply Platt scaling or isotonic regression before reporting graded results.

3. **No posterior predictive validation.** The Beta-Binomial posterior is sound conditionally on its distributional assumptions, but no posterior predictive checks are performed. The Dirichlet posteriors over behavioral atom distributions assume exchangeability of responses within a configuration—a strong assumption for LLMs, which exhibit context-dependent behavior. No held-out posterior predictive p-values, no comparison against overdispersed alternatives (Beta-Binomial vs. Dirichlet-Multinomial vs. hierarchical Dirichlet), and no assessment of the prior's influence via sensitivity analysis. For 54–90 samples, prior sensitivity can be substantial.

4. **Kantorovich distances lack uncertainty quantification.** The bisimulation distances that constitute the primary behavioral divergence measure are reported as point estimates. Given 54–90 queries per configuration, the bootstrap variance on Kantorovich distance estimates is likely substantial—the 1-Wasserstein distance is known to have convergence rate O(n^{-1/d}) in d dimensions, which for the behavioral atom simplex (d ≈ 4) implies slow convergence. No confidence intervals, no bootstrap distributions, and no sensitivity to the underlying metric on the behavioral atom space are reported. This is a major gap for a probabilistic framework.

5. **Statistical baselines achieve comparable detection power.** Chi-squared and MMD tests detect the same 7/15 divergent prompts at p < 0.05, with comparable effect sizes. The paper argues that coalgebraic methods provide structural advantages (reusable automata, temporal reasoning, compositional properties) but demonstrates no case where these structural advantages produce: (a) detection of a divergence that statistical baselines miss, (b) faster detection with fewer queries, or (c) more actionable diagnostic information. Theorem 3's end-to-end error composition (five sources compose additively) is a theoretical contribution, but without showing it produces tighter bounds than simple Bonferroni correction over baseline tests, the practical advantage is undemonstrated.

6. **Distributional robustness is unaddressed.** The framework assumes the LLM's behavioral distribution is stationary within each configuration. No analysis of distributional shift (the LLM's behavior may change over time due to API updates), no DRO (distributionally robust optimization) framework for worst-case behavioral guarantees, and no consideration of adversarial distributional perturbations. For a framework claiming to provide behavioral safety auditing, distributional robustness is a critical missing component. Theorem 2 (Bandwidth-Sample Complexity Bound) implicitly assumes stationarity but does not bound the effect of distribution drift on the learned automaton's validity.

## Questions for Authors

- Can you provide bootstrap confidence intervals on the Kantorovich distances, and how does the width of these intervals compare to the distances themselves? If the 95% CI for a bisimulation distance of 0.3 is [0.05, 0.55], the quantitative distance becomes uninformative.
- What is the posterior predictive p-value for the Dirichlet model fit to observed behavioral atom distributions? Specifically, under the fitted Dirichlet posterior, what fraction of simulated datasets exhibit more extreme summary statistics than the observed data?
- At what sample size do the PAC bounds from Theorem 1 become non-vacuous (ε < width of 95% HPD) for your operating tolerance, and is this achievable at reasonable API cost?

## Overall Assessment

CABER's probabilistic framework is more statistically honest than virtually any comparable LLM behavioral analysis paper—the Bayesian posterior analysis is correctly constructed, the PAC bound vacuity is transparently reported with practical alternatives, and multiple statistical baselines contextualize the contributions. However, the framework's probabilistic core is undermined by three critical issues: (1) the graded satisfaction calibration error of 0.28–0.73 renders the quantitative probabilistic reasoning that distinguishes CABER from binary testing unreliable, (2) the Kantorovich distances that constitute the primary divergence measure lack any uncertainty quantification, and (3) conventional statistical baselines achieve comparable detection power without the coalgebraic complexity. The theoretical contributions (Theorems 1–3) are mathematically interesting but either vacuous at operating scale (Theorem 1) or unvalidated as tighter than naive alternatives (Theorem 3). The probabilistic framework needs calibration, uncertainty quantification on its core outputs, and a demonstrated advantage over simpler alternatives to justify the coalgebraic machinery.

**Score:** 5/10
**Confidence:** 4/5
