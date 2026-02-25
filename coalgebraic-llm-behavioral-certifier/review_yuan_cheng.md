# Review: CABER — Coalgebraic Behavioral Auditing of Foundation Models

**Reviewer:** Yuan Cheng
**Persona:** Probabilistic Modeling Researcher
**Expertise:** Statistical rigor, uncertainty quantification, PAC bounds, Monte Carlo analysis, probabilistic convergence guarantees

---

## Summary

CABER proposes a verification pipeline that treats LLMs as coalgebras, learns finite behavioral automata via PCL* (a probabilistic variant of L*), and model-checks temporal properties with PAC-style error bounds. The theoretical framework is impressive in ambition and the error composition analysis (Theorem 3) is a genuine contribution. However, the empirical evaluation relies entirely on stochastic mock LLMs (Markov chains), which are precisely the class of systems where L*-style algorithms are known to succeed. The PAC guarantees, while formally stated, depend on assumptions (local stationarity, independent sampling) that are unlikely to hold for real LLMs.

## Strengths

1. **Rigorous end-to-end error composition (Theorem 3).** The additive composition of five error sources (learning, abstraction, classifier, model checking, drift) into a total error bound is a principled framework. The numerical example (total ≤ 0.173) provides concrete intuition and is consistent with the empirical ≥92% accuracy.

2. **Classifier robustness analysis is thorough.** The Monte Carlo simulation (2,000 trials per error rate across 6 rates) provides strong empirical evidence that the pipeline maintains ≥99% verdict accuracy under 20% classifier error. The theoretical bound ε_total ≤ ε_learn/(1−ρ) + ε_mc + ρ is validated against simulation data.

3. **Functor bandwidth provides a meaningful complexity measure.** The ε-covering number definition of bandwidth (β) and its connection to metric entropy (Kolmogorov-Tikhomirov) provide a principled characterization of how many effective behaviors an LLM exhibits. The sublinear growth (3.1 to 14.2 vs. naïve alphabet 500) demonstrates the utility of the behavioral abstraction.

4. **Convergence theorem (Theorem 1) with explicit sample complexity.** The PCL* convergence guarantee with explicit query complexity Õ(β·n₀·log(1/δ)) connects the theoretical bandwidth to practical resource requirements.

## Weaknesses

1. **Mock LLMs are stochastic Markov chains — the ideal case for L*-style learning.** The four mock models (3-6 states, Markov dynamics) are precisely the system class that L*-based algorithms are designed to learn. Success on these mock models provides no evidence that the approach would work on real LLMs with context-dependent, non-Markovian behavior. The local stationarity assumption (monitored via CUSUM drift detection) is particularly problematic — real LLMs exhibit state-dependent response distributions that change with conversation history in non-stationary ways.

2. **Independent sampling assumption is violated in practice.** The convergence guarantee assumes independent sampling at each query. Real LLM API interactions involve: (a) temperature-dependent sampling from the same softmax, (b) potential server-side batching effects, (c) context window dependencies. Session isolation is noted as an approximation, but the magnitude of this approximation error is not analyzed.

3. **The PAC parameters are set without justification.** The ε and δ parameters in the PAC guarantee are not connected to any deployment requirement or risk tolerance. What values of ε and δ are needed for the certificates to be meaningful? The numerical example (ε=0.05, δ=0.05) is chosen without justification, and the total error bound of 0.173 may be too loose for safety-critical applications.

4. **Specification soundness degrades rapidly with model size.** The reported soundness drops from 98.3% (n₀≤50) to 94.2% (n₀=200). Extrapolating to real LLMs with thousands or millions of effective states suggests potentially unacceptable degradation. No analysis of how soundness scales beyond n₀=200 is provided.

5. **Query cost ($200-$600 per audit) is computed from mock model experiments.** The query count (71K-94K) is determined by mock model complexity (3-6 states). Real LLMs with far more behavioral states would require proportionally more queries, potentially making audits prohibitively expensive.

## Novelty Assessment

The theoretical framework combining coalgebraic semantics with PAC learning guarantees is genuinely novel. The functor bandwidth concept and its connection to sample complexity are original contributions. However, the gap between theory and validation (mock models only) substantially limits the demonstrated novelty. **High theoretical novelty, low empirical validation.**

## Suggestions

1. Validate on at least one real LLM API (even a small one like GPT-2 or a fine-tuned model) to demonstrate that the approach extracts meaningful behavioral structure from non-Markovian systems.
2. Analyze how specification soundness scales with model complexity beyond n₀=200, either theoretically or by extrapolation.
3. Justify the PAC parameter choices (ε, δ) by connecting them to deployment risk requirements.
4. Provide a sensitivity analysis of the convergence guarantee to violations of the stationarity assumption.

## Overall Assessment

CABER has a theoretically impressive framework with genuine contributions in the form of functor bandwidth and end-to-end error composition. However, the entire empirical validation is on toy Markov chains that are the ideal case for the proposed algorithm. Until the approach is validated on real LLMs, the practical relevance of the theoretical guarantees remains uncertain. The work reads as a strong theoretical proposal with proof-of-concept validation rather than a validated system.

**Score:** 6/10
**Confidence:** 4/5
