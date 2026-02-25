# Review: CABER — Coalgebraic Behavioral Auditing of Foundation Models

**Reviewer:** Yuan Cheng (Probabilistic Modeling Researcher)  
**Expertise:** PAC learning theory, statistical hypothesis testing, probabilistic automata, Markov decision processes, optimal transport metrics  
**Score:** 7/10  
**Recommendation:** Weak Accept

---

## Summary

CABER proposes a principled coalgebraic framework for black-box LLM auditing that extracts probabilistic finite automata and verifies temporal behavioral properties. The PAC-style convergence analysis for PCL* is the paper's strongest theoretical contribution, though several statistical assumptions require deeper scrutiny.

## Strengths

**1. Rigorous PAC-style query complexity analysis.** The Õ(β·n·log(1/δ)) bound for PCL* convergence is carefully derived, and the dependence on the bisimilarity tolerance β is well-motivated. The authors correctly identify that classical L* sample complexity results do not directly apply to stochastic oracles and provide a non-trivial extension that accounts for hypothesis testing at each membership query. The Hoeffding-based confidence intervals for transition probability estimation are appropriate for the bounded output setting.

**2. Principled use of Kantorovich metric for behavioral equivalence.** The choice of Kantorovich (Wasserstein-1) distance over total variation or KL divergence is well-justified for this application: it respects the metric structure of the output space induced by the embedding clustering, and it provides a genuine metric (not just a divergence), enabling the fixed-point characterization of bisimilarity. The connection to optimal transport gives computational tractability via linear programming for finite state spaces.

**3. Sound statistical framework for equivalence queries.** The two-sample testing approach for approximate equivalence queries using the permutation-based maximum mean discrepancy test is statistically sound and avoids parametric assumptions about the response distribution. This is a meaningful improvement over naive frequency-based comparison.

**4. Explicit treatment of confidence parameters.** The (ε,δ)-bisimilarity guarantee cleanly separates approximation error from confidence, following best practices in PAC learning. The paper correctly propagates these parameters through the CEGAR refinement loop, maintaining valid coverage guarantees at each iteration.

## Weaknesses

**1. Union bound accumulation is overly conservative.** The paper applies a naive union bound across all O(n²) state pairs and all CEGAR iterations to maintain the global δ guarantee. For realistic automata with 50-100 states, this yields per-test significance levels on the order of 10⁻⁶, requiring enormous sample sizes (~10⁶ queries per equivalence check). The authors acknowledge this but do not explore tighter concentration arguments—for instance, a Bonferroni-Holm step-down procedure or, better, a Rademacher complexity-based uniform convergence bound over the hypothesis class would substantially reduce the query overhead while maintaining formal guarantees.

**2. Classifier error propagation undermines PAC framework.** The alphabet abstraction relies on a learned clustering model whose error rate is bounded empirically but not incorporated into the PAC analysis. If the classifier misassigns a response to the wrong abstract symbol with probability p_err, the effective transition probabilities observed by PCL* are corrupted by a convolution with the confusion matrix. The paper's convergence proof assumes noise-free symbol observations; without a formal treatment of this misspecification, the (ε,δ) guarantee is informal at best. A clean fix would be to model the classifier as a noisy channel and derive a corrected convergence rate under bounded misclassification.

**3. Stationarity assumption is questionable for LLM APIs.** The PAC analysis assumes the LLM's response distribution is stationary across queries, but production APIs undergo continuous fine-tuning, load balancing across model variants, and temperature/sampling parameter drift. The paper's Phase 0 results (which are notably absent) would need to demonstrate that the stationarity assumption holds over the timescale of automaton extraction, or provide a formal treatment of non-stationarity via, e.g., a sliding-window adaptation of the convergence bounds.

**4. Missing comparison with Bayesian automata learning.** The frequentist PAC framework is one valid approach, but Bayesian methods for probabilistic automata learning (e.g., Bayesian Automaton Learning with Dirichlet priors on transitions) can provide posterior concentration guarantees that are often tighter in practice for moderate sample sizes. The paper does not discuss or compare against this alternative statistical framework, which weakens the claim that PCL* is the right algorithmic choice.

## Verdict

CABER's statistical foundations are solid in structure but require tightening in key areas—particularly classifier error propagation and union bound conservatism. With these fixes and empirical validation of the stationarity assumption, the PAC analysis would be genuinely novel and practically meaningful.
