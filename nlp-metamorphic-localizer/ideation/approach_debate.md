# Approach Debate: nlp-metamorphic-localizer

**Date:** 2026-03-08
**Debaters:** Adversarial Skeptic, Math Depth Assessor
**Method:** Independent critique of all three approaches with cross-challenge and synthesis recommendation.

---

## PART I: Adversarial Skeptic's Critique

### Approach A: "The Pipeline MRI" — Death by Grammar Compiler

#### Fatal Flaws

**F-A1: The grammar compiler is a tar pit disguised as a scoped subproblem.** The proposal claims "~200 productions for 15 transformations." This is fantasy. Passivization *alone* requires: auxiliary `be` conjugation across 6 tenses × 2 aspects × 2 numbers × 3 persons (72 forms), irregular past participle generation (~200 common irregular verbs), ditransitive/monotransitive subcategorization frames, modal interaction ("should have been being written"), and the get-passive with different syntactic structure. That's ~30–40 productions for *one* transformation. Clefting requires expletive `it`, copular agreement, relative clause complementizer selection (that/who/which/∅), and focus-sensitive extraction constraints. Realistic count: 400–600 productions with 80+ feature types — 2–3× the claimed scope. The go/no-go milestone assumes the grammar works; if it takes 50% longer than expected (historically normal), the entire project timeline collapses because shrinker, generator, and every downstream component depend on it.

**F-A2: The 85% top-1 localization accuracy is evaluated on synthetic faults, not real ones.** Tier 2 injects faults by corrupting POS labels, perturbing dependency arcs, adding NER noise — all *uniform* noise. Real NLP pipeline faults are *systematic*: a parser consistently misanalyzes a specific construction, not randomly flipping arcs. The calibration distribution won't match the fault distribution, making the 85% number meaningless outside the evaluation. The paper's only externally valid data is Tier 1 (real bugs), which can't be quantified as a controlled experiment.

#### Serious Weaknesses

- **S-A1:** 107K LoC Rust+Python with PyO3 bridging, custom grammar compiler, and morphological FSTs is effectively unreproducible. Artifact evaluation committees will spend days building it. Contrast with CheckList (pip-installable). Complexity is the enemy of adoption, citation, and artifact badges.
- **S-A2:** BFI conflates signal magnitude with signal abnormality. Stages with near-zero baseline differential show BFI → ∞ on the one transformation that affects them, swamping all other signals. Needs normalization by null distribution.
- **S-A3:** The "regulated-industry" value claim assumes pipeline debugging is the bottleneck, not model accuracy. In healthcare NLP circa 2026, the primary pain is model accuracy on rare entities, not pipeline stage interactions.

#### Unfounded Assumptions

- "20+ percentage-point gap over vanilla SBFL" — if 80% of real faults are single-stage, the gap shrinks to <10 points.
- "Grammar scoped to 200 productions avoids the multi-year rabbit hole" — no grammar engineering effort in history has stayed within initial scope.
- "spaCy, Stanza, HuggingFace cover ~80% of production multi-stage NLP" — many production systems use Spark NLP, NLTK+custom, Flair, or custom serving.

#### Strongest Attack

**The grammar compiler is load-bearing for both generation AND shrinking, creating a single point of failure.** If buggy → inflated false positives. If incomplete → shrinker can't find short counterexamples. If slow → exceeds 75-minute budget. Every failure path runs through the grammar compiler, and grammar engineering has a 40-year track record of taking 3–10× longer than planned.

#### What's Right

M4 is genuinely novel and well-motivated. The introduction-vs-amplification distinction maps cleanly to real engineering decisions. The "10 bugs, 10 words" framing is brilliant paper marketing. Honest acknowledgment of circularity concern shows intellectual maturity.

---

### Approach B: "Information-Theoretic Causal Localizer" — Elegant Theory, Uncomputable Practice

#### Fatal Flaws

**F-B1: N2's localization capacity C(T) is uncomputable in practice.** C(T) requires KL divergence D(τ; hᵢ, hⱼ) between distributions P_k(Δ|τ). For transformer pipelines, Δₖ at the embedding stage is a difference in ℝ⁷⁶⁸. KL divergence estimation between two distributions in ℝ⁷⁶⁸ from finite samples is catastrophically sample-inefficient: k-nearest-neighbor KL estimators require O(exp(d/2)) samples for constant multiplicative error. The proposal claims "100–200 calibration tests suffice." This suffices for *means* (N1's discriminability matrix) but not *distributions* (N2's KL divergences). With 200 samples in ℝ⁷⁶⁸, the KL estimate has infinite variance. **The crown jewel depends on a quantity that is mathematically well-defined but empirically uncomputable.**

**F-B2: The "3–4 weeks for the proof" of N2 is a schedule fantasy.** N2 extends sequential hypothesis testing to correlated observations from composed pipeline stages. The key factorization lemma requires the Markov property for differentials: Δₖ depends on Δ<ₖ only through Δₖ₋₁. This is false in general NLP pipelines: NER's differential depends not just on the parser's differential but on the *actual* parser output (which carries information about *all* prior differentials). The pipeline is Markovian in stage outputs Sₖ but *not* in differentials Δₖ, because Δₖ is a nonlinear function of two correlated execution traces. Fixing this requires either (a) invertible stages (absurd), (b) conditioning on full traces (breaks tractability), or (c) looser bounds with pipeline-depth-dependent slack. A real proof: 2–3 months, not "3–4 weeks."

#### Serious Weaknesses

- **S-B1:** N3's "locally linear" assumption is violated by every discrete-output NLP stage. POS tagger → argmax over softmax (piecewise constant, Jacobian = 0 a.e.). Parser → graph algorithm (combinatorially discrete). The "softmax relaxation" mitigation proves the theorem for a cardboard model, not the real pipeline.
- **S-B2:** 100–200 calibration samples for estimating 7 stages × 15 transformations = 105 distribution-stage-transformation cells means ~2 samples per cell. Mutual information computation is garbage in, garbage out.
- **S-B3:** 5 theorems create 5× reviewing burden. If any one has a bug, the paper's credibility collapses. Approach A's 2 theorems are a more defensible reviewing surface.

#### Strongest Attack

**N2 is a theorem about a quantity that cannot be reliably estimated from feasible data, making "provable near-optimality" vacuous in practice.** The entire value proposition over Approach A is provable guarantees. But if the guarantees depend on an uncomputable quantity, you're back to heuristics — except you've spent 3 extra months proving theorems that don't affect what the tool actually does. A reviewer who asks "what does C(T) equal for the spaCy pipeline?" and receives an answer with 300% confidence intervals will reject the paper.

#### What's Right

Modeling fault localization as sequential hypothesis testing is the right abstraction. N1 (discriminability matrix) is clean, practical, cheap. N4(b)'s improved shrinking bound is genuine. "Cheap-first strategy" is sensible engineering even if the theorem supporting it is flawed.

---

### Approach C: "The Pragmatic Maximizer" — Fast, Fragile, and Intellectually Thin

#### Fatal Flaws

**F-C1: spaCy-as-grammaticality-proxy destroys M5's convergence guarantee.** M5 requires a *deterministic* validity oracle. spaCy's parser is statistical — it never says "ungrammatical," just gives a (possibly wrong) parse with a confidence score. The convergence proof doesn't apply. The paper is left with one theorem (M4), and a single-theorem tool paper at ISSTA is a hard sell.

**F-C2: Killing the grammar compiler makes linguistic coverage accidental.** Without a grammar, generation is "parse seed sentences from curated corpora and apply transformations." If the corpus doesn't contain sentences satisfying transformation-specific preconditions (there-insertable, cleftable, dative-alternation-applicable), coverage is silently incomplete. Penn Treebank is WSJ articles — overwhelmingly third-person, past-tense, declarative. Good luck finding there-insertions, clefts, or dative alternations in sufficient quantity.

#### Serious Weaknesses

- **S-C1:** GPT-4 baseline is a strawman unless prompt engineering is rigorous. Without systematic prompt ablation (5 strategies, report best-of-5), reviewers will dismiss it.
- **S-C2:** Lemma-level alignment fails for 4/15 transformations (synonym substitution, negation insertion, embedding depth change, agreement perturbation), meaning causal refinement doesn't apply to ~27% of transformations.
- **S-C3:** Pure Python shrinker is 10–50× slower than Rust. With 60-second timeout, explores 2–10% of search space. "10 Bugs, 10 Words" becomes "10 Bugs, 20 Words."

#### Strongest Attack

**By killing the grammar compiler and using spaCy-as-proxy, Approach C converts both mathematical contributions from provable results to heuristics, leaving a tool paper with zero theorems.** M4's formal guarantees depend on well-defined distance functions over grammatically controlled inputs. M5's convergence bound requires a deterministic oracle. What remains is engineering: "we built a pipeline debugging tool and found some bugs." Fine demo paper. Not best-paper material.

#### What's Right

Feasibility analysis is the most honest. Staged rollout is sensible. Minimum viable paper fallback shows clear-eyed project management. Scope cuts table is the best project management artifact. Emphasis on evaluation over theorem count reflects mature understanding of what reviewers reward.

---

### Cross-Approach Comparison

**Most fixable flaws: Approach C.** Its problems are ambition-reduction, not incorrect claims:
- F-C1 fixable: build lightweight feature-unification checker (~3–5K LoC) as deterministic validity oracle — between A's full grammar and C's parser proxy.
- F-C2 fixable: augment Penn Treebank with ~300 hand-crafted seed sentences targeting rare constructions.
- S-C1 fixable: systematic prompt engineering with 5 strategies, report best-of-5.
- S-C3 fixable: implement ONLY the shrinker in Rust via PyO3 (~5–8K LoC), keep everything else Python.

**Deepest unfixable problems: Approach B.** C(T) requiring density estimation in high-dimensional spaces is a mathematical fact about nonparametric estimation (curse of dimensionality) — not fixable with better estimators. N3's locally-linear assumption is unfixable because NLP stages are inherently discrete-output.

**Approach A's problems are fixable but expensive.** The grammar compiler could work with 6 months of effort. The question is whether 6 months of grammar engineering is the best use of time when it could be spent on evaluation.

---

## PART II: Math Depth Assessor's Critique

### Per-Result Assessment

| Result | Load-Bearing | Correct | Achievable | Novel | Verdict |
|--------|-------------|---------|------------|-------|---------|
| M3 (Composition) | Important | Yes (caveat: "disjoint positions" under-defined) | Trivially | No | **Cut** (mention only) |
| **M4 (Localization)** | **Essential** | Mostly (multi-fault case needs treatment) | **Yes** | **Yes** | **Must include** |
| M5 (Shrinking) | Important | Yes (O(|T|²·|R|) bound plausible) | Yes | Moderate | **Include** |
| M7 (BFI) | Nice-to-have | Problematic (denominator instability) | Trivially | No | **Cut** (define as metric only) |
| **N1 (Discriminability)** | Important | (a,b) Yes; (c) empirical claim | Yes | Moderate | **Include** |
| N2 (Sample Complexity) | Essential for B, ornamental for tool | Mostly (Markov factorization concern) | **High risk (~40% failure)** | **Genuinely novel** | **Conditional** |
| N3 (Identifiability) | Important | Model wrong for NLP (locally-linear violated) | Partially | Moderate | **Simplify** (keep DCE/IE, cut local-linearity) |
| N4 (Shrinking Hardness) | Important | (c) inapproximability uncertain | Mostly | Moderate | **Include (a,b,d)** |
| N5 (Submodularity) | Nice-to-have | Yes | Yes | Low | **Cut** (appendix) |

### Crown Jewel Assessment: N2

**Novelty: GENUINE.** No prior work provides information-theoretic bounds for pipeline fault localization. The structured hypothesis space (pipeline Markov model) and constrained actions (grammar-limited transformations) make this novel.

**Achievability: HIGH RISK (~40% failure).** Four failure modes:
1. KL factorization lemma may not hold for shared-encoder pipelines.
2. Constants may be vacuous for n ≤ 7 (upper bound "≤47 tests" vs lower bound "≥3 tests").
3. KL estimation from finite data is unstable (naive estimator downward-biased, KSG has high variance).
4. Proof timeline 6–10 weeks realistic, not 3–4.

**Verdict:** Right theorem to prove, but two-track strategy essential. 4-week checkpoint; if factorization lemma proved with reasonable constants, proceed. Otherwise, relegate to future work.

### Math-Value Tradeoff

| Result | Practical User Impact |
|--------|-----------------------|
| N1 (rank check) | **Real** — "add a morphological transformation" is directly actionable. |
| N2 (sample bounds) | **Marginal** — heuristic stopping gives comparable results in practice. |
| N3 (identifiability) | **Negligible** — observational path almost never applies to real NLP stages. |
| N4 (hardness) | **Indirect** — users see "shrunk to 8 words in 12s," not the NP-hardness proof. |
| N5 (submodularity) | **None** — no user notices 63% vs 55% of maximum localization information. |

**80% of user value comes from M4 + M5.** Only ~5% from N1–N5's extra theory. The extra math improves the *paper* (tools track → research track, "solid" → "best-paper candidate") but not the *tool*.

### Optimal Math Portfolio

**INCLUDE:** M4 (essential), N1 (cheap, practical), N4(a,b,d) (hardness + convergence).

**CONDITIONAL:** N2 if proved with non-vacuous constants by 4-week checkpoint.

**SIMPLIFIED:** N3 — keep DCE/IE decomposition and interventional sufficiency. Cut locally-linear condition.

**CUT:** M3, M7, N5, N4(c), N3(b,c).

**Contingency without N2:** M4 + N1 + N4(a,b,d) + N3-simplified. Strong tool paper. Paper succeeds on evaluation.

---

## PART III: Skeptic's Recommended Synthesis

**The Hardened Pragmatic Localizer (~35K LoC, Python + minimal Rust shrinker)**

**From C, keep:** Python architecture for localizer/adapters/transformations/evaluation. Corpus-based generation. Staged rollout. "10 Bugs, 10 Words" centerpiece. GPT-4 baseline. Minimum viable paper fallback.

**From A, steal:** Shrinker in Rust via PyO3 (~5–8K LoC). Lightweight feature-unification checker (~2–3K LoC Rust) as deterministic validity oracle — not spaCy's parser, not a full grammar compiler. Preserves M5 as a real theorem.

**From B, steal:** N1 (discriminability matrix — cheap, useful, no density estimation). Thompson-sampling for adaptive selection (not full ADAPTIVE-LOCATE). N4(b) improved shrinking bound.

**Reject from B:** N2 (uncomputable C(T)), N3 (model wrong for NLP), N5 (no use case).

**Fix C's flaws:** 300 hand-crafted seed sentences. 5-strategy GPT-4 prompt ablation. Feature-checker as validity oracle.

**Resulting paper:** M4 + M5 + N1. ~35K LoC. 2 theorems + 1 clean definition. pip-installable with optional Rust. Manageable reviewing surface.

*"Approach A is a 2-year project pretending to be a 6-month project. Approach B is a PhD thesis pretending to be a paper. Approach C is a paper pretending it doesn't need theorems. The synthesis is a paper that knows exactly what it is."*

---

## PART IV: Math Assessor's Recommended Synthesis

Agrees with Skeptic on M4 + N1 + M5 (with deterministic oracle). Adds:

- **Include N4(a,b,d)** — NP-hardness justifies 1-minimality, convergence bound makes shrinker trustworthy, expected shrinking ratio sets user expectations. All achievable.
- **Attempt N2 on two-track strategy** — if proved with non-vacuous constants by week 4, include as crown jewel that elevates paper from tools to research track. If not, relegate to future work conjecture with computational evidence. The tool is built regardless.
- **N3-simplified** — keep DCE/IE decomposition as formalization. Cut locally-linear condition. State interventional sufficiency (trivially true). This gives the introduction-vs-amplification distinction formal backing without claiming identifiability results that don't hold.

**Key disagreement with Skeptic:** The Skeptic recommends against attempting N2 at all. The Math Assessor argues the two-track strategy has no downside — you implement ADAPTIVE-LOCATE as a heuristic regardless, and the proof, if achieved, transforms the paper. The ~40% failure risk is acceptable because the contingency portfolio (M4 + N1 + M5 + N4) is already strong.
