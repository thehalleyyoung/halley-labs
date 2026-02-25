# Review: Spectacles — Contamination-Certified Evaluation Certificates

**Reviewer:** Yuan Cheng
**Persona:** Probabilistic Modeling Researcher
**Expertise:** Statistical rigor, calibration, hypothesis testing, contamination detection methodology, false positive/negative analysis

---

## Summary

Spectacles combines WFA-based scoring, STARK zero-knowledge proofs, and PSI-based contamination detection to produce certified benchmark evaluations. The differential testing methodology (800K pairs, 0 disagreements) is impressive, and the triple-implementation approach provides strong empirical confidence in scoring correctness. However, the contamination detection component conflates n-gram overlap with statistical contamination, and the paper's honest limitations section acknowledges that paraphrase memorization has TPR ≈ 0. The "certified" framing overstates what the certificate actually attests.

## Strengths

1. **Triple implementation with differential testing is a gold standard for empirical correctness.** Implementing each metric three times (reference, WFA, circuit) and cross-validating on 100K random pairs per metric with 0 disagreements across 800K total pairs provides very strong evidence that the implementations agree. This is superior to unit testing alone.

2. **Honest WFA coverage decomposition.** Table 2 transparently reports the percentage of each metric that is WFA-proved vs. gadget-assisted vs. tested-only. BLEU at 60% WFA, 5% gadget, 35% tested-only is an honest admission that not everything is formally verified.

3. **The contamination detection addresses a real problem.** N-gram overlap as a necessary-but-not-sufficient condition for contamination is correctly scoped, and the ROC analysis by attack type (paraphrase memorization → TPR ≈ 0) is commendably honest.

4. **Mathematical bugs found and fixed.** Discovering and correcting the Montgomery inverse constant and Lagrange interpolation errors through property-based testing demonstrates the value of the verification methodology.

## Weaknesses

1. **N-gram PSI is a weak contamination detector.** The paper honestly acknowledges that paraphrase memorization, indirect contamination, and contamination-aware fine-tuning are undetectable by n-gram overlap. This means the contamination certificate attests only to literal overlap, not to statistical contamination in any meaningful sense. The "contamination-certified" framing in the title is therefore misleading — it should be "n-gram-overlap-certified."

2. **Threshold τ requires domain-specific calibration with no guidance.** The contamination threshold is described as needing calibration at the 99th percentile of clean-pair distribution, but no empirical guidance, recommended ranges, or calibration methodology is provided. Without this, users cannot meaningfully set τ.

3. **Proof sizes are estimated, not measured.** The 45-750 KiB proof sizes come from an analytical formula (constraints × 16 + wires × 32 + ...), not from actual STARK prover execution. End-to-end wall-clock timing on realistic corpora is acknowledged as future work.

4. **The per-metric WFA coverage varies enormously.** Exact match and regex are 100% WFA-proved, but ROUGE-L is only 65% WFA with 20% tested-only. The "verified scoring" claim is strong only for simple metrics — complex metrics have significant unverified components.

5. **No statistical power analysis for the differential testing.** With 100K pairs per metric, the probability of missing a 10⁻⁴ disagreement rate is ~e⁻¹⁰ ≈ 4.5×10⁻⁵. This should be stated explicitly to contextualize what "0 disagreements" means.

## Novelty Assessment

The WFA decomposition of NLP metrics is a genuine contribution. The combination with STARK proofs and PSI contamination detection is novel in this application domain. However, the contamination detection component is weak enough to undermine the "contamination-certified" positioning. **Moderate to high novelty for the WFA+STARK pipeline, low novelty for contamination detection.**

## Suggestions

1. Rename from "contamination-certified" to "overlap-certified" or add prominent qualifiers about the scope of contamination detection.
2. Provide empirical guidance for τ calibration with examples from actual benchmark datasets.
3. Measure actual proof sizes and verification times from running the STARK prover.
4. Report the statistical power of the differential testing (detection probability for specific disagreement rates).

## Overall Assessment

Spectacles makes a genuine contribution with the WFA decomposition of NLP metrics and the triple-implementation verification methodology. The differential testing provides strong empirical confidence. However, the contamination detection is limited to literal n-gram overlap, making the "contamination-certified" framing misleading. The verification depth varies significantly by metric, with complex metrics having substantial unverified components. With honest repositioning of the contamination claims, this is a solid contribution to verified evaluation.

**Score:** 7/10
**Confidence:** 4/5
