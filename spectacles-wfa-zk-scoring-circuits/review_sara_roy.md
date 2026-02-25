# Review: Spectacles — Contamination-Certified Evaluation Certificates

**Reviewer:** Sara Roy
**Persona:** Machine Learning and Formal Verification
**Expertise:** ML evaluation methodology, benchmark contamination, verified computation, practical deployment of formal verification

---

## Summary

Spectacles addresses a real and important problem: trustworthy ML benchmark evaluation with contamination detection. The approach of compiling scoring functions to verifiable circuits is sound in principle, and the triple-implementation methodology provides strong empirical evidence of correctness. However, the system's practical utility is limited by the weakness of the contamination detection (n-gram overlap only), the partial formal verification (Lean proofs cover semiring axioms but not the compilation pipeline), and the absence of end-to-end benchmarking on real datasets at scale.

## Strengths

1. **Addresses a genuine need in ML evaluation.** Benchmark contamination is a real and growing problem. A system that can certify both score correctness and data separation fills an important gap in the ML evaluation ecosystem.

2. **Triple-implementation methodology is rigorous.** Three independent implementations of each metric, cross-validated on 100K random pairs, is a strong empirical verification methodology that exceeds standard practice.

3. **Mathematical bugs discovered demonstrate value.** Finding and fixing the Montgomery inverse constant error and Lagrange interpolation bug shows that the verification methodology catches real errors that standard testing might miss.

4. **Lean-Rust correspondence documentation is honest.** The three-layer empirical methodology (Lean proofs, proptest, differential testing) is clearly distinguished from verified extraction, and the comparison to CompCert/CakeML honestly acknowledges the gap.

5. **Seven metrics supported with honest coverage reporting.** Supporting exact match, token F1, BLEU, ROUGE-N, ROUGE-L, regex, and pass@k, with per-metric WFA coverage percentages, provides a useful toolkit with transparent capabilities.

## Weaknesses

1. **Contamination detection is too weak for the "certified" framing.** N-gram PSI detects only literal overlap. The paper acknowledges that paraphrase memorization has TPR ≈ 0, but the title and abstract still use "contamination-certified." This framing will mislead practitioners who need actual contamination detection.

2. **No real-dataset evaluation at scale.** The end-to-end tests use "benchmark-style data, not full datasets." Without evaluation on actual MMLU, SQuAD, or other standard benchmarks at full scale, the practical performance (proof generation time, memory usage, proof size) is unknown.

3. **Lean verification covers only the easy parts.** Semiring axioms are algebraic identities that are straightforward to formalize. The hard parts — WFA compilation, STARK circuit generation, PSI protocol correctness — are not Lean-verified. The formalization creates a misleading impression of verification depth.

4. **The STARK prover is simulated, not implemented.** Proof sizes and verification times are computed from analytical formulas, not from running an actual STARK prover. This means the "under 20ms verification" claim is an estimate, not a measurement.

5. **No cost-benefit analysis for deployment.** What is the computational overhead of certified evaluation vs. standard evaluation? If proof generation takes 100x longer than scoring, the system may be impractical for routine benchmark evaluation. Without benchmarking, this question is unanswered.

6. **Post-processing operations are a significant unverified gap.** The paper acknowledges that aggregation gadgets are tested but not formally verified. For metrics like BLEU where 35% of the computation is tested-only, the "verified scoring" claim overstates what is actually proved.

## Novelty Assessment

The WFA decomposition approach is novel and the combination with STARK proofs is an original contribution. However, the contamination detection using PSI is a straightforward application of existing cryptographic techniques. **High novelty for the verification pipeline, low novelty for contamination detection.**

## Suggestions

1. Evaluate on at least one full benchmark dataset (e.g., 1000 MMLU questions) with actual proof generation and timing.
2. Implement the STARK prover and measure real proof sizes and verification times.
3. Rename the contamination component to accurately reflect its capabilities (n-gram overlap detection, not general contamination detection).
4. Provide a deployment cost-benefit analysis comparing certified vs. standard evaluation.
5. Prioritize Lean verification of the WFA-to-circuit compilation over additional semiring proofs.

## Overall Assessment

Spectacles tackles an important problem with an elegant approach. The WFA decomposition is a genuine insight and the triple-implementation methodology is rigorous. However, the gap between the "certified evaluation" framing and the actual verification coverage is significant: the STARK prover is simulated, the Lean proofs cover only semiring axioms, and the contamination detection is limited to literal overlap. With real-scale evaluation, actual STARK implementation, and honest repositioning, this could be a strong practical contribution.

**Score:** 7/10
**Confidence:** 4/5
