# Review: Spectacles — Verified WFA-ZK Scoring Circuits for Contamination-Certified Evaluation

**Reviewer:** Sara Roy (Machine Learning & Formal Verification)  
**Expertise:** ML system deployment, formal verification of ML pipelines, software engineering for AI, large-scale system integration, benchmark platform design  
**Score:** 7/10  
**Recommendation:** Weak Accept

---

## Summary

Spectacles addresses a genuine and growing need—trustworthy, contamination-aware evaluation of language models—with an ambitious architecture spanning formal verification, zero-knowledge proofs, and private set intersection. While the conceptual contribution is strong, significant practical adoption barriers related to engineering scale, ecosystem integration, and the Lean-Rust semantic gap temper my assessment.

## Strengths

**1. Addresses a Real and Urgent Problem.** Benchmark contamination is arguably the most pressing credibility crisis in LLM evaluation today. Spectacles is the first system to propose cryptographic contamination certificates that are both verifiable and privacy-preserving. Unlike statistical contamination detection (e.g., min-k% prob, perplexity-based methods), which are inherently heuristic and gameable, the PSI-based approach provides a hard guarantee: the evaluator cannot have seen more than τ fraction of the test set without the certificate revealing this. This is a qualitative improvement over the status quo.

**2. Specification-Level Correctness is the Right Abstraction.** Most ZK-based computation verification proves "this program executed correctly on these inputs," but Spectacles proves "this score equals the mathematical definition of the metric." This distinction matters: a buggy BLEU implementation can execute correctly (in the ZK sense) while producing wrong scores. By compiling from mathematical specifications via WFA, Spectacles shifts the trusted computing base from implementation to specification, which is a meaningful security improvement.

**3. Differential Testing Methodology is Pragmatic.** The 100K pair differential testing between the Rust implementation and reference Python implementations (SacreBLEU, HuggingFace evaluate) is the right pragmatic validation strategy for a system where full formal verification is infeasible. The choice of reference implementations is well-motivated: SacreBLEU is the community standard for BLEU, and HuggingFace evaluate covers the remaining six metrics. Testing across edge cases (empty strings, single-token outputs, maximum-length inputs) shows engineering maturity.

**4. Laptop-CPU Feasibility Lowers Adoption Barriers.** The explicit design goal of running on commodity hardware (laptop CPUs with AES-NI) rather than requiring GPU clusters or specialized FPGA setups dramatically lowers the adoption barrier. The use of BLAKE3 for hashing (which is SIMD-optimized) and AES-NI for OPRF operations means the cryptographic overhead is bounded by hardware-accelerated primitives, making integration into existing evaluation pipelines at least architecturally feasible.

## Weaknesses

**1. Engineering Scale Risk is Severe.** The projected 117–142K LoC across Lean 4, Rust, and TLA+ is comparable to a production database engine. The paper proposes a 12-month timeline, which implies roughly 10–12K LoC per month of production-quality, formally verified code. For comparison, the seL4 microkernel (10K LoC C + 200K LoC Isabelle) took approximately 20 person-years. Even accounting for Lean 4's superior ergonomics over Isabelle, the proposed timeline appears optimistic by a factor of 3–5×. No staffing plan or team composition is provided to justify the schedule.

**2. Lean-Rust Semantic Gap Undermines End-to-End Guarantees.** The 800 LoC Lean pilot formalizes metric specifications and WFA equivalence, but the production Rust implementation is connected only via differential testing. This means the end-to-end guarantee has a trust gap: the Lean proofs certify properties of abstract WFA, while the STARK circuit operates on a Rust-compiled concrete circuit. Without verified extraction (à la CertiCoq for Coq or code extraction in Isabelle/HOL), a subtle bug in the Rust WFA-to-circuit compiler could invalidate all formal guarantees while passing differential tests.

**3. Ecosystem Coordination Requirements are Underestimated.** For Spectacles to achieve its stated goal, benchmark platforms (HELM, LM-Eval-Harness, OpenCompass) must integrate the certificate verification pipeline. This requires these platforms to (a) adopt the EvalSpec DSL for metric specification, (b) run the PSI protocol with model developers, and (c) verify STARK proofs. The paper assumes this coordination will happen but provides no analysis of the incentive structures, standardization requirements, or migration costs. In practice, ecosystem adoption is the hardest part of any verification infrastructure.

**4. Semi-Honest Security Model Contradicts the Threat Scenario.** The threat model posits adversaries who actively inflate scores (A1) and substitute outputs (A2), but the PSI protocol assumes semi-honest behavior. A malicious evaluator could submit fabricated model outputs to the PSI protocol—outputs that were never actually generated by the model—and receive a valid contamination certificate for these fake outputs. The commit-then-reveal protocol (G3) partially addresses this, but only if the commitment scheme is binding against computationally bounded adversaries, which requires explicit security reduction not provided in the paper.

**5. No User Study or Benchmark Platform Integration Prototype.** The paper claims practical impact on LLM evaluation but provides no evidence of usability: no user study with benchmark maintainers, no prototype integration with any existing platform, no API design for certificate generation and verification. For a systems paper claiming practical relevance, the absence of any empirical evaluation of the human and organizational factors is a significant gap. Even a small pilot with one benchmark suite would substantially strengthen the practical contribution claim.

## Verdict

Spectacles identifies the right problem and proposes a technically sophisticated solution, but the gap between the formal architecture and practical deployment is substantial. The engineering scale, ecosystem coordination, and Lean-Rust semantic gap are not merely implementation details—they are fundamental feasibility concerns that the paper must address more convincingly to merit a strong accept.
