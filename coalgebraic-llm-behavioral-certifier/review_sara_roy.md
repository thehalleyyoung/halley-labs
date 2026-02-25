# Review: CABER — Coalgebraic Behavioral Auditing of Foundation Models

**Reviewer:** Sara Roy
**Persona:** Machine Learning and Formal Verification
**Expertise:** ML-verification integration, learning-based verification, experimental methodology, practical deployment considerations

---

## Summary

CABER combines active machine learning (PCL*) with formal model checking (QCTL_F) to verify behavioral properties of LLMs. The ML component learns finite automata from query responses, while the verification component checks temporal specifications. The integration is deeper than typical ML+verification systems — the learning algorithm is specifically designed for the coalgebraic verification pipeline, and error bounds are composed end-to-end. However, the entire evaluation is on stochastic mock LLMs, and the practical deployment considerations (cost, latency, API constraints) are not adequately addressed.

## Strengths

1. **End-to-end error composition is properly done.** Theorem 3 composes errors from five sources (learning, abstraction, classifier, model checking, drift) with explicit dependence on PAC parameters. This is the correct way to handle a multi-stage ML+verification pipeline.

2. **Classifier robustness analysis is methodologically strong.** 2,000 Monte Carlo trials per error rate with 6 error rates provides good statistical coverage. The agreement between theoretical bound and simulation results validates the error model.

3. **The learning component is purpose-built.** PCL* is not a generic learning algorithm applied to verification — it is specifically designed for the coalgebraic setting, with tolerance parameters and sample counts that connect directly to the verification guarantees.

4. **Certificate generation provides a tangible output.** The AuditCertificate with hash chain, PAC parameters, and verification results provides a concrete, timestamped deliverable that can be independently checked.

5. **~99K lines of code demonstrates implementation depth.** The Rust workspace with separate crates for core, CLI, examples, and integration shows a well-structured implementation effort.

## Weaknesses

1. **Mock LLM evaluation invalidates all practical claims.** The four mock models (Markov chains with 3-6 states) bear no resemblance to real LLMs. Key differences: (a) real LLMs have context-dependent distributions over unbounded sequences, (b) real LLM behavior varies with prompt engineering, (c) real LLMs have non-stationary behavior across API updates. Success on Markov chains provides zero evidence of practical viability.

2. **No real-LLM experiment, even a small one.** Even a single experiment with GPT-2 or a small fine-tuned model would provide some evidence that the approach extracts meaningful behavioral structure. The complete absence of real-LLM validation makes the claims entirely theoretical.

3. **Deployment cost analysis is unrealistic.** The $200-$600 estimate is based on mock model query counts (71K-94K). Real LLMs with more complex behavior would require far more queries. Additionally, the estimate doesn't account for: (a) API rate limits that extend wall-clock time, (b) prompt construction costs, (c) embedding model inference costs for alphabet abstraction, (d) re-auditing frequency.

4. **No comparison to existing LLM evaluation frameworks.** There is no comparison against HELM, CheckList, DecodingTrust, or other established evaluation frameworks. The paper argues that CABER provides temporal properties that these frameworks cannot express, but this argument should be supported by showing a concrete property that matters in practice that existing frameworks miss.

5. **The specification templates assume known behavioral failure modes.** The six templates presuppose knowledge of what properties to check. In practice, unknown failure modes (emergent behaviors, capability overhangs) may be more dangerous than known ones. The framework provides no mechanism for discovering unexpected behavioral patterns.

6. **AALpy+PRISM baseline receives an unfair handicap.** The baseline comparison gives AALpy a manually defined alphabet while CABER discovers its own. This makes CABER's task harder but also makes the comparison non-controlled — any performance difference could be due to alphabet quality, not framework quality.

## Novelty Assessment

The ML-verification integration is deeper than typical approaches. The purpose-built learning algorithm with verification-aware error composition is genuinely novel. However, without real-LLM validation, the demonstrated novelty is theoretical. **High theoretical novelty, no practical novelty demonstrated.**

## Suggestions

1. Conduct at least one real-LLM experiment (even GPT-2) to bridge the theory-practice gap.
2. Compare against HELM or CheckList on a shared task to contextualize the contribution.
3. Provide a realistic deployment cost analysis including API rate limits, latency, and re-auditing frequency.
4. Add a behavioral anomaly detection mode that discovers unexpected patterns rather than only checking specified properties.
5. Fix the AALpy baseline by giving both systems the same alphabet or neither system a manual one.

## Overall Assessment

CABER has the best ML-verification integration of the projects reviewed, with purpose-built learning algorithms and properly composed error bounds. However, the complete absence of real-LLM validation is a critical gap. The mock model evaluation demonstrates that the implementation works on the ideal case but provides no evidence of practical viability. The work is a strong theoretical contribution with a significant validation gap.

**Score:** 6/10
**Confidence:** 4/5
