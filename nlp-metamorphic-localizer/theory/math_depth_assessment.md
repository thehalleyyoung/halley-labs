# Math Depth Assessment: nlp-metamorphic-localizer

**Assessor Role:** Math Depth Assessor
**Scope:** All mathematical claims across Approaches A, B, and C

---

## Per-Result Assessment

---

### M3: MR Composition Theorem (Formal Specification)

**Claim:** Composed transformations on disjoint syntactic positions preserve meaning with probability ≥ 1−δ via Clopper-Pearson bounds. Reduces test count from O(|T|²) to O(|T| · log|R|).

| Criterion | Assessment |
|-----------|------------|
| **Load-bearing?** | **Important.** Removing it eliminates the coverage efficiency argument — you need ~10× more tests to achieve pairwise coverage. The system still works; it's just slower. |
| **Correctly stated?** | **Yes, with a caveat.** The statistical form is honest and the Clopper-Pearson machinery is standard. However, "disjoint syntactic positions" is under-defined for natural language. Passivization rearranges the VP; adverb repositioning moves within the VP — are these "disjoint"? The predicate needs a formal definition in terms of tree nodes, not a hand-wave. |
| **Achievable?** | **Trivially.** Standard statistics applied to a new domain. No proof risk. |
| **Novel?** | **No.** The authors correctly disclaim novelty. The contribution is the domain-specific formalization, not the mathematics. |
| **Key concern** | The "disjoint syntactic positions" precondition will fail for many transformation pairs in practice (e.g., passivization + topicalization both restructure the clause), limiting the coverage reduction to a subset of pairs. The O(|T| · log|R|) claim may hold only for ~40-60% of pairs, not universally. |

---

### M4: Causal-Differential Fault Localization (NEW)

**Claim:** Per-stage differentials Δₖ(x, τ) = d_k(sₖ(prefix_k(x)), sₖ(prefix_k(τ(x)))). Localize to k* = argmax_k [Δₖ − E[Δₖ | τ is meaning-preserving]]. Interventional refinement distinguishes introduction from amplification. Complexity O(N · n · C_pipeline).

| Criterion | Assessment |
|-----------|------------|
| **Load-bearing?** | **Essential.** This is the core contribution of the entire project. Remove it and you have CheckList — detection without localization. Every approach depends on it. |
| **Correctly stated?** | **Mostly.** The formulation is clean and the interventional logic is sound. Two technical issues: (1) The baseline E[Δₖ | τ is meaning-preserving] requires a calibration corpus of non-faulty pipeline executions — the proposal acknowledges the circularity but doesn't formalize when the bootstrap estimate converges. (2) The argmax rule implicitly assumes a single faulty stage. For multiple simultaneous faults (common in practice: tokenizer + parser both struggle with passivization), the argmax identifies only the most salient fault, and the interventional analysis may produce misleading introduction/amplification labels because replacing stage k's output doesn't isolate stage k when stages k-2 and k are both faulty. |
| **Achievable?** | **Yes.** The algorithm is implementable. The formal statement is a validated heuristic, not a provably optimal procedure, which is the right framing. The complexity bound is straightforward. |
| **Novel?** | **Yes, in combination.** SBFL is standard. Causal intervention is standard. Their synthesis for NLP pipeline stages — where "coverage" means linguistic feature processing and the intervention operates on typed intermediate representations — is novel. The introduction-vs-amplification distinction is the genuine conceptual contribution. Closest prior art: spectrum-based fault localization for Java programs (Wong et al. 2016), which operates on statement coverage, not pipeline-stage IR distances. |
| **Key concern** | **Multiple simultaneous faults.** Real NLP pipelines frequently exhibit correlated failures (the tokenizer mishandles a construction, AND the parser compounds the error). The argmax heuristic degrades to "identifies the noisiest stage" rather than "identifies the causal stage." The interventional refinement partially addresses this, but replacing stage k's output when stage k-2 is also faulty means the counterfactual is still corrupted. Needs explicit treatment of the multi-fault case. |

---

### M5: Grammar-Constrained Shrinking Convergence (NEW)

**Claim (A/C):** GCHDD terminates in O(|T|² · |R|) grammar-validity checks, producing 1-minimal counterexamples.
**Claim (B, improved):** O(|T_x| · log|T_x| · |R|) via binary search over subtree orderings.

| Criterion | Assessment |
|-----------|------------|
| **Load-bearing?** | **Important.** Without this, the shrinker has no convergence guarantee — it might loop or produce non-minimal output. The 1-minimality guarantee is what makes counterexamples actionable ("this is the shortest proof sentence that still triggers the bug"). But the localizer (M4) works without the shrinker; M5 is load-bearing for counterexample quality, not for localization. |
| **Correctly stated?** | **The O(|T|² · |R|) bound (A/C) is plausible.** It follows the structure of Zeller's original delta debugging proof extended to tree operations: O(|T|) subtrees × O(|T|) rounds × O(|R|) validity check per attempt. **The O(|T_x| · log|T_x| · |R|) improvement (B) requires scrutiny.** The binary-search-within-levels argument assumes subtree criticality is monotone with respect to some ordering — this needs to be proved, not assumed. Subtree importance for fault-exposure is not necessarily monotone in any natural ordering (size, depth, leftmost position). If monotonicity fails, the binary search is unsound and the bound reverts to quadratic. |
| **Achievable?** | **Yes for the O(|T|² · |R|) bound.** This is a careful but routine extension of delta debugging. **Uncertain for the O(|T_x| · log|T_x| · |R|) improvement.** The binary-search speedup requires a structural lemma about subtree criticality that may not hold. |
| **Novel?** | **Moderately.** TreeReduce (Herfert et al. 2017) handles programming-language parse trees. The extension to natural-language parse trees with feature unification constraints is genuinely new. The constraint that the shrunk sentence must remain transformation-applicable (not just grammatical) is a meaningful addition. The conceptual novelty is moderate; the engineering novelty is high. |
| **Key concern** | **Approach C's pragmatic weakening may undermine the theorem.** Using spaCy's parser as a "grammaticality proxy" instead of a unification grammar means the validity oracle has false negatives (grammatical sentences that spaCy misparsed) and false positives (ungrammatical sentences that spaCy accepted). The O(|T|² · |R|) bound assumes a correct oracle. With an approximate oracle, 1-minimality becomes "1-minimality modulo parser errors," which is much weaker. If Approach C is chosen, M5 is a *heuristic with empirical validation*, not a *theorem with a proof*. |

---

### M7: Behavioral Fragility Index (Formal Specification)

**Claim:** BFI(P, τ) = E_x[dist_out(P(x), P(τ(x)))] / E_x[dist_in(x, τ(x))]. Ratio metric for per-stage amplification.

| Criterion | Assessment |
|-----------|------------|
| **Load-bearing?** | **Nice-to-have.** BFI provides an interpretable summary metric but is not required for localization or shrinking. The system works without it. It adds a quantitative dimension to the behavioral atlas ("transformer NER has BFI 4.7 for passivization") but any amplification metric would serve this purpose. |
| **Correctly stated?** | **Problematic denominator.** When τ barely changes the input (e.g., synonym substitution of a near-synonym), dist_in → 0 and BFI → ∞, creating unbounded artifacts. The definition needs a regularization term or a minimum-distance threshold. Additionally, the per-stage version (used in the proposal) divides stage-k output distance by stage-(k-1) output distance, but when stage k-1 is the source of amplification, BFI_k inherits that amplification in its denominator, producing misleadingly low BFI for the actual fault-introducing stage. |
| **Achievable?** | **Trivially.** It's a ratio of empirical means. |
| **Novel?** | **No.** Amplification ratios appear in adversarial robustness (Lipschitz constant estimation), signal processing (gain), and control theory. The application to NLP pipelines is new but the metric form is not. |
| **Key concern** | Denominator instability and interpretive ambiguity when stages have heterogeneous output types (you're dividing tree-edit-distance by token-edit-distance — what does BFI = 3.2 *mean* when the numerator and denominator use incompatible distance scales?). |

---

### N1: Stage Discriminability Matrix and Separation Theorem

**Claim:** M ∈ ℝⁿˣᵐ where M_{k,j} = E_x[Δₖ(x, τⱼ)]. rank(M) = n iff T can localize to unique stage. Structural + lexical + morphological transformations achieve full rank for n ≤ 7.

| Criterion | Assessment |
|-----------|------------|
| **Load-bearing?** | **Important.** Provides a pre-test diagnostic: "can this transformation set localize at all?" Without it, the engine might run 5,000 tests that all probe the same stage boundary. This is a genuine operational contribution — the rank check prevents wasted computation. |
| **Correctly stated?** | **Parts (a) and (b) are correct** — this is standard linear algebra (column space dimensionality determines which rows are distinguishable). **Part (c) is an empirical domain claim masquerading as a theorem.** The assertion that structural + lexical + morphological transformations achieve full rank assumes specific pipeline architectures and that different transformation types produce linearly independent differential signatures. This is a reasonable empirical observation for spaCy-like pipelines, but it is *not* a theorem — it could fail for pipelines with shared-representation stages (e.g., a multi-task transformer where POS, parsing, and NER share the same encoder, making their differentials highly correlated). Part (c) should be stated as a conjecture validated on specific pipelines, not a theorem. |
| **Achievable?** | **Yes for (a)-(b).** These are one-paragraph proofs. **Partially for (c).** Empirical validation on spaCy/Stanza/HuggingFace is achievable; a general proof is not. |
| **Novel?** | **Moderately.** The matrix M and its rank-localizability connection are a clean formalization. The idea that "different test types probe different pipeline components" is folk wisdom in testing; the contribution is making it precise. Analogous constructions exist in compressed sensing (measurement matrix design) and group testing (separating matrix), but the instantiation for NLP pipelines is new. |
| **Key concern** | **Estimation of M from finite samples.** The entries M_{k,j} = E_x[Δₖ(x, τⱼ)] are population means estimated from finite calibration data. With small calibration sets (100-200 tests, as suggested), the rank of the *empirical* M̂ may be numerically full even when the population M is not (noise inflates rank). Need a statistical test for effective rank (e.g., condition number threshold), not just numerical rank. |

---

### N2: Information-Theoretic Localization Bounds (Crown Jewel)

**Claim:**
- Lower: m ≥ (ln(n-1) + ln(1/δ)) / C(T)
- Upper: m ≤ (2 ln n + ln(1/δ)) / C(T) + O(n)
- Non-adaptive penalty: n-fold increase
- C(T) = max_w min_{i≠j} Σⱼ wⱼ · D(τⱼ; hᵢ, hⱼ) where D is KL divergence

| Criterion | Assessment |
|-----------|------------|
| **Load-bearing?** | **Essential for Approach B's thesis, ornamental for the tool.** N2 transforms the narrative from "we built a tool" to "we established fundamental limits." Without it, Approach B collapses to Approach A with extra notation. But the *tool* works fine without N2 — ADAPTIVE-LOCATE is just Bayesian sequential testing, which is implementable without proving the bounds are tight. |
| **Correctly stated?** | **Mostly, with two significant concerns.** (1) **The Markov factorization lemma** (DKL(P_i ‖ P_j) = Σₖ E[DKL(Pᵢ(Δₖ\|Δ<ₖ) ‖ Pⱼ(Δₖ\|Δ<ₖ))]) assumes the pipeline has a strict Markov structure — Δₖ depends only on Δₖ₋₁. This is true for simple sequential pipelines but fails when stages share global state (e.g., HuggingFace pipelines where all stages read from a shared transformer encoder, meaning Δ₃ depends on Δ₁ directly, not only through Δ₂). The factorization may need to be replaced with a chain-graph decomposition that accounts for shared latent variables. (2) **The O(n) additive term in the upper bound** is not negligible. For n=5 stages and small C(T), this additive term can dominate, making the bound say "you need ~20 tests" when the lower bound says "you need ~5." A 4× gap between upper and lower bounds is respectable for information theory but undermines the "tight bounds" marketing. (3) **The non-adaptive penalty of n** via coupon-collector argument is clean but assumes each transformation independently targets a single stage. If transformations have "spread" (perturbing multiple stages simultaneously, as most NLP transformations do), the penalty may be less than n. |
| **Achievable?** | **High risk.** The lower bound via Fano's inequality is a well-trodden path, but the extension to sequential testing with correlated observations is genuinely hard. The upper bound analysis of greedy MI selection in correlated settings extends Naghshvar & Javidi 2013, which took a full journal paper for independent observations. The pipeline correlation structure adds substantial technical difficulty. **The 3-4 week estimate is wildly optimistic for a research-level result.** Realistic estimate: 6-10 weeks for a complete proof with tight constants, assuming no dead ends. There is a ~30% chance the proof "works in the limit" but the constants are so loose that the bounds are vacuous for n ≤ 7 (the practical range). |
| **Novel?** | **Yes, genuinely.** No prior work provides information-theoretic bounds for pipeline fault localization. The closest analogues are group testing theory (Du & Hwang 2000) and sequential hypothesis testing (Chernoff 1959), but the structured hypothesis space (pipeline Markov model) and constrained actions (grammar-limited transformations) make this a novel instantiation requiring new technical work. This is not "apply Fano's inequality to a new domain" — it requires the factorization lemma and the handling of correlated observations, which are non-trivial contributions. |
| **Key concern** | **Calibration-to-reality gap.** The bounds depend on C(T), which depends on KL divergences D(τ; hᵢ, hⱼ) between distributions P_k(Δ\|τ). These distributions are unknown and must be estimated from calibration data. If KDE estimates of the KL divergence have high variance (they do — KL estimation from finite samples is notoriously unstable), then the computed C(T) may bear little relation to the true capacity, and the sample-complexity bounds become numerical artifacts, not theoretical guarantees. The robust variant (using lower confidence bounds on KL) addresses this in principle but may make C(T) so conservative that the bounds say "you need 10,000 tests" for a 5-stage pipeline — worse than the non-adaptive approach. |

---

### N3: Causal Identifiability of Fault Origin

**Claim:** SCM: Sₖ = fₖ(Sₖ₋₁) + εₖ. DCE_k(τ) = E[Δₖ | do(Sₖ₋₁^τ := Sₖ₋₁)]. Observationally identifiable iff P(Δₖ | Δₖ₋₁) is injective. Sufficient condition: fₖ locally linear.

| Criterion | Assessment |
|-----------|------------|
| **Load-bearing?** | **Important.** Provides theoretical backing for the introduction-vs-amplification distinction that all three approaches rely on. But the tool implements interventional replay regardless (it doesn't check identifiability before replaying). The practical impact is the "cheap-first strategy" — avoid replay when observational identification suffices. |
| **Correctly stated?** | **The formalism is clean but the model is wrong for NLP.** The additive noise SCM (Sₖ = fₖ(Sₖ₋₁) + εₖ) assumes continuous outputs with additive stochastic perturbation. NLP pipeline stages produce *discrete* outputs (POS tags, dependency labels, entity spans). A POS tagger doesn't output f(input) + noise — it outputs a deterministic sequence of discrete labels (for fixed model weights). The additive noise model is a category error. The "locally linear" sufficient condition requires computing the Jacobian of fₖ, which doesn't exist for discrete-output functions. The proposal acknowledges this and suggests "soft-max relaxations," but this amounts to saying "replace the real pipeline with a smooth approximation, prove the theorem for the approximation, and hope the real pipeline behaves similarly." This is defensible but should be stated explicitly as an approximation theorem, not an exact identifiability result. |
| **Achievable?** | **Yes for part (a)** (interventional sufficiency — trivially true by definition). **Partially for part (b)** — the continuous case is clean, the discrete extension via soft-max relaxation is achievable but weakens the guarantee. **Uncertain for the Lip₂/Lip₁ bound in part (c)** — computing Lipschitz constants for NLP pipeline stages is itself a hard problem. |
| **Novel?** | **Moderate.** SCM + do-calculus (Pearl 2009) is well-established. The application to NLP pipeline fault localization is new but the causal machinery is borrowed. The "cheap-first strategy" (try observational identification first, fall back to intervention) is a practical contribution. |
| **Key concern** | **The locally linear condition almost never holds for NLP stages.** Tokenizers are discontinuous (adding a space changes the tokenization completely). POS taggers and NER models are discrete classifiers — locally constant, not locally linear. Dependency parsers use argmax decoding — piecewise constant. The sufficient condition in (b) is satisfied approximately only for the soft-max relaxation of these stages, which is a theoretical convenience, not a practical reality. In practice, the engine will almost always fall back to interventional replay, making N3's observational identifiability result a theorem about a world that doesn't exist. |

---

### N4: Grammar-Constrained Minimization: Hardness and Optimality

**Claim:** (a) NP-hardness of global minimization via reduction from Min Grammar-Consistent String. (b) 1-minimality in O(|T_x| · log|T_x| · |R|). (c) No polynomial c-approximation unless P=NP. (d) Expected shrinking ratio E[|x'|] ≤ |x|/b + O(α·log|x|).

| Criterion | Assessment |
|-----------|------------|
| **Load-bearing?** | **Important.** Part (a) justifies the design decision to target 1-minimality instead of global minimality — without it, a reviewer could ask "why not find the globally shortest counterexample?" Parts (b) is the convergence guarantee. Part (d) sets user expectations. |
| **Correctly stated?** | **Concerns with parts (b) and (c).** Part (a) is a straightforward reduction — fine. Part (b)'s improvement from O(|T|² · |R|) to O(|T| · log|T| · |R|) depends on a binary-search argument that assumes criticality is monotone in some subtree ordering — this needs proof, as discussed under M5. **Part (c) is the most problematic.** The claim that no polynomial-time algorithm can guarantee a counterexample shorter than c · |OPT| for any constant c is an *inapproximability* result. Such results typically require either PCP-based hardness (not just NP-hardness) or a specific gap-amplification argument. The proposal says it "follows from (a) plus gap-preserving reduction" but gap-preservation is non-trivial — it requires that the reduction from Min Grammar-Consistent String preserves approximation ratios, which must be verified. The claim may be correct but the proof sketch is insufficient. Part (d) is reasonable but the "bounded ambiguity α" assumption may not hold for English grammars (which are massively ambiguous). |
| **Achievable?** | **Yes for (a), (b) with the weaker quadratic bound, and (d).** The inapproximability result (c) requires more careful work. |
| **Novel?** | **Moderate.** The hardness reduction (a) is a standard technique. The improved convergence bound (b) and expected shrinking ratio (d) are new for grammar-constrained delta debugging. Part (c), if correct, is the most novel sub-result. |
| **Key concern** | **The inapproximability claim (c) may not survive scrutiny.** NP-hardness alone does not imply inapproximability. The reduction must be gap-preserving, and the gap structure of Min Grammar-Consistent String is not well-established in the literature. If (c) falls, the narrative weakens from "1-minimality is *provably optimal*" to "1-minimality is *the best we know how to achieve*" — still fine, but less impressive. |

---

### N5: Submodularity of Localization Information

**Claim:** F(S) = I(H; {Δ(xᵢ, τᵢ)}_{i∈S}) is monotone submodular. Greedy achieves (1-1/e) approximation.

| Criterion | Assessment |
|-----------|------------|
| **Load-bearing?** | **Nice-to-have.** This provides a theoretical guarantee for the batch/CI mode (non-adaptive test selection). Without it, greedy test selection still works empirically — you just can't prove the 63% approximation ratio. The system's primary mode is adaptive (ADAPTIVE-LOCATE), making N5 a fallback result. |
| **Correctly stated?** | **Yes, modulo a subtlety.** MI of a set of observations about a discrete random variable H is indeed monotone submodular (Krause & Guestrin 2005). The pipeline structure doesn't break this. However, the claim requires that the observations {Δ(xᵢ, τᵢ)} are conditionally independent given H — this holds if each test uses an independently sampled input x, which it does. The formulation is correct. |
| **Achievable?** | **Yes.** This is a direct instantiation of known results. The pipeline-specific contribution is the derivation of the explicit sample-complexity bound in part (c), which is a calculation, not a research problem. |
| **Novel?** | **Low.** Submodularity of MI is known. The contribution is recognizing that this framework applies to pipeline fault localization and connecting it to N2. |
| **Key concern** | **Practical irrelevance.** The greedy algorithm requires computing F(S ∪ {(τ,x)}) − F(S) for every candidate test, which requires evaluating mutual information — the same computation that makes ADAPTIVE-LOCATE work. If you can afford to compute MI, you should use ADAPTIVE-LOCATE (which is n-fold better). If you can't afford MI computation, you can't run the greedy algorithm either. N5 is a theorem in search of a use case. |

---

## 6. The Optimal Math Portfolio

For a best-paper submission at ISSTA/ICSE, here is the recommended portfolio:

### INCLUDE

| Result | Role in Paper | Justification |
|--------|--------------|---------------|
| **M4** | Core contribution | Non-negotiable. The introduction-vs-amplification distinction is the diamond. |
| **N1** | Diagnostic foundation | Clean, achievable, operationally useful. The rank-check-before-testing idea is elegant and practical. Takes 1 page to state and prove. |
| **N4(a,b)** | Hardness + convergence | NP-hardness justifies 1-minimality; convergence bound (even the weaker O(|T|² · |R|)) makes the shrinker trustworthy. Drop (c) unless the gap-preserving reduction is airtight. Keep (d) for user expectations. |
| **N3 (simplified)** | Causal formalization | State the introduction/amplification distinction formally via the DCE/IE decomposition, prove interventional sufficiency (trivial), state observational identifiability as a heuristic for optimization, skip the locally-linear sufficient condition (it doesn't hold for real NLP). |
| **N2 (if proved)** | Crown jewel | If the proof is complete and the bounds are non-vacuous for n ≤ 7, this is the result that elevates the paper from "good tool" to "best-paper candidate." But include it ONLY if fully proved. A half-proved N2 destroys credibility. |

### CUT

| Result | Reason for Cutting |
|--------|--------------------|
| **M3** | Correctly identified as standard statistics. Mention in 2 sentences as engineering methodology. Not worth page space. |
| **M7** | Problematic denominator, not load-bearing. Define the metric in the system section, don't claim it as a mathematical contribution. |
| **N5** | Theorem in search of a use case. The greedy guarantee is nice but if you can compute MI, you should use ADAPTIVE-LOCATE. One sentence in Related Work: "In the non-adaptive setting, submodularity of localization information provides a (1-1/e) greedy guarantee (details in appendix)." |
| **N3(b,c)** | The locally-linear sufficient condition is wrong for NLP. Keep the DCE/IE decomposition and interventional sufficiency; cut the observational identifiability analysis. |
| **N4(c)** | The inapproximability claim is likely unprovable without PCP machinery. Cut it rather than risk a flawed proof. |

### CONTINGENCY

If N2's proof is not complete in time, the portfolio becomes: **M4 + N1 + N4(a,b,d) + N3(simplified)** — a strong tool paper with clean formalization, but not a theory paper. This is still publishable at ISSTA as a tools track paper with solid evaluation. The paper's strength shifts from "tight bounds" to "real bugs found + localization accuracy."

---

## 7. Crown Jewel Assessment: Is N2 Achievable and Novel?

### Novelty: GENUINE

N2 is genuinely novel. The closest prior work:
- **Group testing** (Du & Hwang 2000): Binary outcomes, independent tests, no pipeline structure. N2 handles continuous differential vectors with correlated components along a Markov chain.
- **Sequential hypothesis testing** (Chernoff 1959): Two hypotheses, independent observations. N2 handles n hypotheses with grammar-constrained actions.
- **Bayesian experimental design** (Jedynak et al. 2012, Naghshvar & Javidi 2013): Independent observations. N2's pipeline correlation structure is genuinely new.

The conceptual contribution — characterizing localization difficulty by a single computable quantity C(T) — is clean and would be recognized as a real advance by the SE testing community.

### Achievability: HIGH RISK (~40% chance of failure)

**What could go wrong:**

1. **The KL factorization lemma may not hold cleanly.** The Markov assumption (Δₖ ⊥ Δ₁:ₖ₋₂ | Δₖ₋₁) fails for pipelines with shared encoders. Fixing this requires a chain-graph or factor-graph generalization that significantly complicates the proof.

2. **The constants may be vacuous.** Information-theoretic bounds often have beautiful asymptotic form but terrible constants. For n=5 (the typical case), if the upper bound says "≤ 47 tests" and the lower bound says "≥ 3 tests," the gap is practically useless. The 3-4 week estimate doesn't account for the 2-4 additional weeks of tightening constants.

3. **KL estimation from finite data is unstable.** The bounds depend on C(T), which requires estimating KL divergences. KL estimation from finite samples has known bias issues (the naive plug-in estimator is downward-biased, the KSG estimator has high variance for small samples). If C(T) is poorly estimated, the bounds become numbers printed on paper, not operational guarantees.

4. **The proof is 3-4 weeks only in the best case.** Realistically, extending Fano to structured sequential testing with correlated observations is a research paper in itself. If the factorization lemma requires substantial modification, the proof timeline could double.

### Verdict

N2 is **the right theorem to prove** but carries substantial execution risk. I recommend a **two-track strategy**: (1) Begin the N2 proof immediately and in parallel implement ADAPTIVE-LOCATE as a heuristic (which works regardless of whether the bounds are tight). (2) Set a 4-week proof checkpoint. If the factorization lemma is proved and the constants are reasonable by that point, proceed to completion. If not, pivot to the contingency portfolio (M4 + N1 + N4 + N3-simplified) and relegate N2 to a "future work" conjecture with computational evidence.

---

## 8. Math-Value Tradeoff: Does Approach B's Extra Math Help Users?

### What the extra math buys for users

| Result | Practical User Impact |
|--------|-----------------------|
| **N1 (rank check)** | **Real.** "Your transformation set cannot distinguish the POS tagger from the dependency parser — add a morphological transformation" is directly actionable. Estimated user time saved: 30-60 minutes per pipeline configuration. |
| **N2 (sample bounds)** | **Marginal.** "You need ≈ 23 tests at δ=0.05" is nice but a heuristic stopping rule ("stop when localization confidence > 95%") gives comparable practical results. The principled stopping criterion saves maybe 10-20% of test executions vs. a reasonable heuristic. |
| **N3 (identifiability)** | **Negligible.** The observational identification path almost never applies to real NLP stages (discrete outputs violate local linearity). In practice, the engine always does interventional replay, which is what A/C do anyway. |
| **N4 (hardness)** | **Indirect.** Users don't see the NP-hardness proof. They see "shrunk to 8 words in 12 seconds." The convergence bound affects developer confidence in the tool's predictability, but this is an internal quality, not a user-facing feature. |
| **N5 (submodularity)** | **None.** No user will notice whether the CI test suite captures 63% or 55% of maximum localization information. |

### The honest tradeoff

Approach B's extra math improves the **paper** substantially (from ISSTA-tools to ICSE-research track, from "solid" to "best-paper candidate") at the cost of:
- 6-10 additional weeks of proof work
- Increased risk of incomplete results at submission deadline
- Potential credibility damage if N2 or N3 contain errors discovered by reviewers
- No meaningful improvement in the tool's practical performance for users

The math is not ornamental — N1 and the N2-derived stopping criterion provide real operational value. But the honest assessment is that **80% of the user value comes from M4 + M5 (shared across all approaches), and only ~5% comes from N1-N5's extra theoretical machinery.** The remaining 15% of the value comes from the behavioral atlas and engineering quality, which are approach-independent.

### Recommendation

If the goal is **maximum practical impact**: Choose Approach C's math portfolio (M4 + M5 practical) and invest the saved proof-writing time in evaluation quality and real bug discovery.

If the goal is **maximum academic impact**: Choose Approach B's math portfolio but with the contingency plan above. The N2 crown jewel, if achieved, genuinely changes the theoretical landscape. A tool paper with 25 real bugs is good; a tool paper with 25 real bugs *and* tight information-theoretic bounds is exceptional.

If the goal is **maximum probability of a strong paper**: Choose Approach A's math portfolio (M4 + M5 + M3 + M7) supplemented with N1 from Approach B. This gives clean formalization with no proof risk, and the paper succeeds or fails on evaluation strength alone — which is the most controllable variable.

---

## Summary Table

| Result | Load-Bearing | Correct | Achievable | Novel | Include? |
|--------|-------------|---------|------------|-------|----------|
| M3 | Important | Yes (caveat) | Trivial | No | Cut (mention only) |
| **M4** | **Essential** | Mostly | **Yes** | **Yes** | **Must include** |
| M5 | Important | Yes | Yes | Moderate | Include (weaker bound) |
| M7 | Nice-to-have | Problematic | Trivial | No | Cut (define as metric) |
| **N1** | Important | (a,b) Yes; (c) empirical | Yes | Moderate | **Include** |
| **N2** | Essential for B | Mostly (concerns) | **High risk** | **Yes** | **Conditional include** |
| N3 | Important | Model is wrong | Partially | Moderate | Include simplified |
| N4 | Important | (c) uncertain | Mostly | Moderate | Include (a,b,d) |
| N5 | Nice-to-have | Yes | Yes | Low | Cut (appendix) |
