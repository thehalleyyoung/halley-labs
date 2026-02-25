# Review: Spectacles — Verified Compiler from Semiring-Weighted Automata to Zero-Knowledge Scoring Circuits

**Reviewer:** Sara Roy
**Persona:** Machine Learning & Formal Verification
**Expertise:** ML evaluation methodology, formal verification of ML systems, experimental methodology, statistical rigor, ablation studies

---

## Summary

Spectacles tackles the ML evaluation integrity problem by compiling NLP metrics into weighted finite automata (WFA) over typed semirings and generating STARK zero-knowledge proofs of score correctness. The evaluation methodology — 57,518 triple-verification checks with 0 disagreements, Lean 4 formalization, and 2,825 benchmark-level checks — is among the most thorough I have reviewed for a formal-methods-meets-ML paper. However, critical experimental gaps undermine the production-readiness claims: the test corpus is not representative of real NLP text, no comparison with the most relevant prior work (VerifiableEvals) is provided, the STARK scaling analysis stops at 512 states, and the Lean-to-Rust verification gap means the running code is not formally verified.

## Strengths

1. **Rigorous triple-verification methodology.** 57,518 three-way agreement checks (reference × WFA × circuit) across 10 seeds × 5 metrics × 1,015 pairs with 0 disagreements provides the strongest empirical correctness evidence available for this type of system. The use of three _independent_ implementations as mutual cross-checks is methodologically exemplary.
2. **Real bugs discovered by the methodology.** Two genuine math bugs caught by triple verification — a Montgomery reduction error and a Lagrange interpolation error — demonstrate that the verification methodology has practical teeth, not just confirmatory value. These bugs would likely have escaped conventional testing.
3. **Comprehensive benchmark coverage.** 2,825 additional checks across MMLU, SQuAD, translation, and random datasets extend correctness validation to representative NLP benchmarks, not just synthetic inputs. The 0-disagreement result across all benchmarks strengthens confidence.
4. **Deterministic reproducibility.** Fixed random seeds and a reproduce.sh script enabling full result regeneration is exemplary reproducibility practice that should be standard in all ML evaluation work.
5. **Honest limitation reporting.** The test corpus constraints (30-word vocabulary, max 9 tokens), the Lean-to-Rust gap, and the end-to-end integration gap are all explicitly documented rather than hidden.

## Weaknesses

1. **Test corpus is fundamentally not representative of production NLP.** The 30-word vocabulary and max 9 tokens per string are adequate for verifying algebraic correctness of small WFA, but they bear no resemblance to production NLP text. Real BLEU computation involves 30K+ BPE tokens, sequences of 50–500 tokens, Unicode text with mixed scripts, and edge cases (empty strings, single-token outputs). The 57,518 tests validate correctness on a _qualitatively different_ input domain from production use. No analysis bridges this gap: no argument that algebraic correctness on small inputs implies correctness on large inputs (it does not, in general, due to state-count scaling and potential overflow in Proposition E.1's [0, 2^62) bound). The 9.8K property tests help but still operate on synthetic distributions.

2. **No quantitative comparison with VerifiableEvals.** South et al. (2024) is the most directly comparable prior work — also targeting ZK proofs for NLP evaluation. The comparison is entirely theoretical (specification-level vs. program-level verification). No empirical head-to-head benchmark compares proof size, generation time, verification time, or metric coverage on the same inputs. For an ML evaluation paper, failing to run the baseline on your benchmarks is a significant methodological gap.

3. **STARK scaling analysis is truncated at 512 states.** The 76 verified proofs go up to 512 states, but no extrapolation or scaling model predicts performance at production scale. The circuit width scales as O(|Q|² × |Σ|) per step, so a 5,000-state WFA would require ~156× more constraints per step than the 400-state benchmarks. Will proof generation remain under 60 seconds? Under 10 minutes? The paper provides no scaling model, no regression fit, and no extrapolation. The 400-state BLEU-4 timing (3,821 ± 271 ms) is the largest data point, but production BLEU on realistic vocabularies could require orders of magnitude more states.

4. **Lean-to-Rust verification gap is structurally unaddressable by testing.** The formal proofs cover a mathematical model in Lean 4; the deployed system is Rust code bridged by 57K differential tests and 9.8K property tests. This is fundamentally weaker than machine-checked extraction (CompCert, CakeML, sel4). Differential testing can find bugs (and did: 2 bugs found) but cannot prove absence of bugs. The paper should state clearly: "the Rust implementation is _tested_, not _verified_." Currently, the "verified compiler" framing could mislead readers into believing the running code has formal guarantees.

5. **Performance benchmarks lack statistical rigor for production claims.** Proof generation times (3,821 ± 271 ms for 400-state BLEU-4; 1.5 ms verify; 270 KiB proof) report mean ± std but no confidence intervals, no percentile analysis (P50, P95, P99), no warm-up/cool-down methodology, no memory profiling, and no comparison across hardware configurations. The verification time of 1.5 ms is reported without specifying whether this includes deserialization, hash recomputation, or only the FRI query check. For a system targeting production deployment, this level of performance characterization is insufficient.

6. **No adversarial evaluation of contamination detection.** Contamination detection reports F1 = 1.00 at τ = 0.03 for verbatim n-gram overlap, but no adversarial testing is performed: no paraphrasing attacks, no back-translation, no synonym substitution, no character-level perturbations. The paper acknowledges that "heavy paraphrasing evades detection," but in an era where contamination is predominantly non-verbatim (pre-training data overlaps through reformulation), this limitation is not merely theoretical — it is the _primary_ failure mode in practice. An ML evaluation paper claiming to address evaluation integrity should test against realistic contamination strategies.

## Questions for Authors

- Can you run VerifiableEvals (South et al. 2024) on your MMLU and SQuAD benchmarks and provide a quantitative comparison of proof size, generation time, and verification time?
- What is the projected state count and proof time for BLEU-4 on WMT'14 EN-DE with standard BPE tokenization (32K vocabulary, sequences up to 128 tokens)?
- Does triple-verification agreement hold for Unicode inputs (Chinese, Arabic, emoji, mixed-script), or does the WFA construction assume ASCII?

## Overall Assessment

Spectacles has exceptional evaluation methodology for its tested domain — the triple verification, Lean formalization, bug discovery, and benchmark coverage collectively provide strong evidence of algebraic correctness. However, the evidence is confined to a toy-scale input domain (30 words, 9 tokens) that is qualitatively different from production NLP, no comparison with the most relevant prior work is provided, the scaling analysis stops well short of production requirements, and the "verified compiler" framing overstates what the Lean proofs actually guarantee about the running Rust code. The contribution is real and valuable, but the experimental evaluation does not yet support the production-readiness claims that the paper's framing implies.

**Score:** 6/10
**Confidence:** 4/5
