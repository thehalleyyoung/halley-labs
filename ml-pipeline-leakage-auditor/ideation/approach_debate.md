# Approach Debate: Skeptic and Mathematician Critiques

## Approach A: LeakageIR — Sound Abstract Interpretation

### Skeptic's Critique

Approach A is the most fragile of the three proposals, with a **composite P_fail ≈ 0.70**. Three interlocking fatal flaws doom it.

**Fatal Flaw 1: The 85% coverage ceiling is aspirational fiction.** The proposal claims its static Python frontend resolves ~85% of real pipeline dynamism. This is not a measured quantity — it is a wish. Real Kaggle pipelines use `exec()`, config-driven column names (`df[config['features']]`), `**kwargs` cascaded through three abstraction layers, and decorator metaprogramming. Pyright, with 100K+ LoC and over 100 person-years of investment, still fails on pandas-heavy code. A fresh heuristic frontend will realistically cover 50–65% of real pipelines. The remaining 35–50% get ⊤ — trivially sound, completely useless. The proposed mitigation (user annotations like `# taintflow: schema(...)`) requires users to understand the tool's internal abstract domain, which annihilates the zero-configuration value proposition. **P_fail = 0.55.**

**Fatal Flaw 2: Galois connection proofs for 130 transfer functions are infeasible.** Each proof requires operation-specific reasoning. `pd.merge` alone has 6 join types × 4 index modes × 2 suffixing strategies = 48 behavioral variants, each needing a separate case in the soundness argument. Extrapolating from CompCert (~6 person-years for verified C compilation), 130 hand-verified transfer functions is a 3–5 person-year effort, not a 6-month sprint. The project will silently downgrade from "formally verified" to "we tested a few cases" — at which point the Galois connection machinery buys nothing over Approach B's testing-based validation. **P_fail = 0.70.**

**Fatal Flaw 3: Purely static analysis cannot know row provenance.** `train_test_split` returns two DataFrames via tuple unpacking. Tracking which variable holds which partition through arbitrary Python control flow (function arguments, dicts, reassignment) is the full alias analysis problem — unsound or undecidable in general. Every pipeline needs this resolved, since a single misidentified partition assignment makes all downstream bounds meaningless. **P_fail = 0.40.**

**Vacuity attack:** Without execution data, bounds require worst-case assumptions for group cardinalities, key overlaps, and data distributions. A `groupby('user_id').transform('mean')` with 50,000 groups gets a bound of ~15.6 bits when true leakage is 0.001 bits. The "10× within true leakage" target will fail for any operation whose bound depends on data-dependent quantities — which is most of them. The reported bounds will be 100–1000× loose on real pipelines, making the formal machinery an expensive way to produce useless numbers.

**Competitor kill:** A 500-line Python script using `ast.parse()` + a lookup table of 10 channel capacity formulas delivers 80% of the practical value. The Galois connection infrastructure adds theoretical elegance but zero practical capability beyond this script.

### Mathematician's Critique

Approach A's mathematical portfolio is top-heavy: one crown jewel surrounded by ornamental scaffolding.

**M-A1 (Galois-Connected Partition-Taint Lattice): ★★☆ novelty.** The lattice $\mathcal{T} = (\mathcal{P}(\mathcal{O}) \times [0, B_{max}], \sqsubseteq)$ is structurally necessary, but the insistence on a *full Galois insertion* is over-engineered. The origin set $\mathcal{P}(\mathcal{O})$ is a standard powerset lattice; the bit-bound $[0, B_{max}]$ is a standard interval; their product is a lattice by Theorem 2.16 of Davey & Priestley. The claim that "prior abstract domains use either qualitative or quantitative measures, never both" overstates the novelty — the reduced product of a security-label lattice with a numerical domain achieves the same effect, and reduced products are standard since Cousot & Cousot 1979. A sound abstraction relation (not a Galois connection) would suffice for the soundness theorem. Proof feasibility: <1 month.

**M-A2 (Widening-Narrowing Convergence): ★☆☆ novelty.** Lexicographic widening on product lattices is textbook (Bagnara et al., 2005). The only novelty is verifying that widening bit-bounds while holding origins fixed preserves soundness for this specific lattice — a mechanical check, not an intellectual contribution. Moreover, ~40% of pipelines have "loops" but they are nearly all `GridSearchCV`/`cross_val_score` with rigid structure, handleable by pattern-specific unrolling. This contribution should be dropped entirely.

**M-A3 (Soundness Theorem & Fit-Transform Decomposition): ★★★ novelty.** This is the crown jewel — the single most important mathematical contribution across all three approaches. The fit-transform channel decomposition lemma, which factors `fit_transform`'s feedback loop into an aggregation channel and a pointwise application channel, is genuinely new. No prior QIF work handles the pattern where an operation simultaneously reads input taint and writes state that modifies the same input's transformation. However, the proof is high-risk (2–4 months): the decomposition assumes `fit` computes sufficient statistics, silently excluding KNNImputer, IterativeImputer, and custom transformers. Proof feasibility: non-trivial to high risk.

**Overall assessment:** Keep M-A3 (the fit-transform lemma). Simplify M-A1 by dropping the Galois insertion requirement. Drop M-A2 entirely. Total load-bearing math: one hard theorem (M-A3) dressed in two months of ornamental proof engineering.

### Cross-Critique: Skeptic Challenges Mathematician

The Mathematician rates M-A1 as ★★☆ novel and suggests simplifying it. I argue the novelty rating is irrelevant because **the entire lattice is operationally useless without execution data**. The Mathematician evaluates the lattice in isolation — "is this a valid construction?" — but ignores that the lattice's concretization $\gamma(O, b)$ requires knowing the mutual information $I(D_P^{te}; v)$, which depends on the actual data distribution. Without execution, the tool must use worst-case parameterizations that inflate bounds by 100–1000×. A beautiful lattice producing vacuous bounds is a mathematical curiosity, not a tool. The Mathematician's recommendation to "simplify M-A1" presumes M-A1 is worth keeping at all. It isn't — the entire static-only architecture should be abandoned, rendering M-A1 moot.

Furthermore, the Mathematician praises M-A3 as "★★★ — the crown jewel." I agree the fit-transform lemma is intellectually novel. But novelty does not imply feasibility. The sufficiency assumption (that `fit` computes sufficient statistics) excludes a significant fraction of real estimators. The Mathematician estimates 2–4 months for the proof; I estimate a 40% chance the theorem statement must be weakened to the point where it covers only `StandardScaler`, `MinMaxScaler`, and `PCA` — three operations a lookup table handles trivially.

### Cross-Critique: Mathematician Challenges Skeptic

The Skeptic claims that "a 500-line Python script delivers 80% of the practical value" (the competitor kill). This is technically true for detecting *syntactic* leakage patterns — but fundamentally wrong about *quantitative* analysis. Pattern matching detects "there is leakage"; it cannot distinguish 0.001 bits from 50 bits. The entire value proposition of TaintFlow — quantitative measurement in bits — is inaccessible to a lookup-table approach. The Skeptic's competitor kill attacks a strawman: a tool that merely detects leakage, not one that measures it. The Galois connection framework, while over-engineered in its current form, provides the formal anchor necessary for any quantitative claim. Without *some* formal foundation, the tool's bit-count output is a meaningless number with no relationship to actual mutual information.

Additionally, the Skeptic's P_fail estimates compound risks multiplicatively (0.55 × 0.70 × 0.40) when the risks are partially correlated. If the Python frontend achieves 65% coverage (not 50%), many of the "infeasible proofs" become unnecessary (fewer operations to verify), and row provenance resolution improves (simpler pipelines are easier to track). The true composite P_fail is closer to 0.55–0.60, not the implied 0.70. Still the worst of the three, but not as catastrophic as the Skeptic suggests.

### Verdict on Approach A

**Strengths:** M-A3's fit-transform decomposition lemma is the single most important mathematical contribution in the entire project. The no-execution property has genuine value for auditing pipelines without data access. The certificate infrastructure, if achievable, provides the strongest trust guarantees.

**Fatal weaknesses:** The Python static frontend is a project-killing risk that two of three evaluators flag as the #1 concern. Bounds will be vacuously loose without observed data distributions. The proof engineering timeline is 3–5× longer than estimated.

**Overall viability: Low.** Approach A should not be built as proposed. Its architecture should be cannibalized: extract M-A3's fit-transform lemma and transplant it into a dynamic-first architecture.

---

## Approach B: TaintFlow — Hybrid Dynamic-Static

### Skeptic's Critique

Approach B is the hardest to kill — which is why I respect it the least. It survives by making *weaker claims*, not by solving harder problems. **Composite P_fail ≈ 0.30.**

**Fatal Flaw 1: Conditional soundness is not soundness.** The entire differentiator over LeakageDetector is supposed to be formal guarantees. But "sound for the observed execution path" means: no leakage is missed *on this particular data with this particular control flow*. If a `try/except` block routes the pipeline differently on the user's next dataset, the guarantee evaporates. A reviewer will immediately note: "This is dynamic analysis with extra steps. What does it buy over running the pipeline twice?" **P_fail = 0.25** — not because the tool breaks, but because the paper's theoretical contribution is undermined.

**Fatal Flaw 2: The trace semantics formalization is an unfounded assumption.** Python's `sys.settrace` does not intercept C-extension calls. pandas operations execute in C/Cython. A `df.groupby('a').transform('mean')` dispatches to C for the actual computation; the instrumentation sees entry/exit but not internal data flow. Proving faithfulness would require a formal model of CPython's C-extension API interacting with pandas' Cython dispatch — a dissertation-scale effort. **P_fail = 0.30.**

**Fatal Flaw 3: Row provenance tracking at scale will OOM.** For a `merge` with duplicate keys producing a 1000 × 1000 Cartesian product, the provenance bitmap is 1M bits per original row across 2000 rows — 250 MB for a single merge. Chained merges on real pipelines will consume 10–50 GB of bitmap memory. The "roaring bitmaps" optimization helps but doesn't eliminate the combinatorial explosion. **P_fail = 0.35.**

**Vacuity attack:** The empirical refinement phase (M-B3) does all the real work. The abstract bounds from Phase 2 are the same channel capacity formulas as Approach A — just parameterized by observed ρ instead of worst-case. If those abstract bounds are still 10–20× loose, the KSG estimator provides the tight numbers. But KSG on high-dimensional data with finite samples has bias scaling as $O(d/n)$. The tool's output is either the loose abstract bound (unimpressive) or the unreliable empirical estimate (untrustworthy). The reduced product papers over this gap with notation.

**Why B survives despite these attacks:** B's worst-case failure mode is still useful. If abstract bounds are loose, empirical refinement helps. If instrumentation misses edge cases, 95% of leakage through standard APIs is still caught. If conditional soundness bothers a reviewer, the paper honestly frames it as sufficient for interactive debugging. B has graceful degradation; A and C do not.

### Mathematician's Critique

Approach B has the best ratio of load-bearing to total math. Two of three contributions carry real weight; one is ornamental.

**M-B1 (Provenance-Parameterized Capacity Bounds): ★★☆ novelty.** The individual channel capacity formulas ($C_{mean}(\rho)$, $C_{PCA}(\rho, d)$) are textbook Gaussian channel results. The novelty is exploiting exact observed provenance from dynamic instrumentation to parameterize bounds — a clean insight that genuinely improves precision over static worst-case analysis. The systematic catalog for ~30 operations is useful engineering but not mathematical novelty. Proof feasibility: <1 month.

**M-B2 (Execution-Path-Conditional Soundness): ★★☆ novelty.** Conditional soundness is a known concept (concolic testing, hybrid type inference), but formalizing it for quantitative information-flow bounds with a trace semantics for Python instrumentation is a meaningful adaptation. The DPI argument on the observed DAG is standard; the trace semantics formalization is the genuinely novel part. However, the proposal underspecifies what "faithful instrumentation" means formally — what simulation relation between instrumented and uninstrumented execution? This is the hard open question. Proof feasibility: 1–2 months, but the trace semantics may require significant simplifying assumptions.

**M-B3 (Reduced Product of Abstract and Empirical Domains): ★☆☆ novelty.** This is ornamental. The reduction operator $\rho(\tau, \hat{I}) = \min(b, \hat{I} + \epsilon_\alpha)$ is just clamping an abstract bound to an empirical estimate plus a confidence margin. The "strict precision improvement theorem" — that $\min(a, b) \leq a$ when $b < a$ — is tautological. A hostile reviewer would write: "Theorem 3 states that taking the minimum of two upper bounds is tighter than either. This is obvious." Drop this entirely; replace with a one-paragraph note.

**Overall assessment:** B's math is honestly scoped. M-B1 and M-B2 are genuinely load-bearing. M-B3 is padding. Total proof effort: ~2–3 months for a realistic, non-ornamental mathematical foundation. No research-blocking steps on the critical path.

### Cross-Critique: Skeptic Challenges Mathematician

The Mathematician rates M-B2 (conditional soundness) as ★★☆ and calls the trace semantics formalization "the genuinely novel part." I challenge this: the trace semantics formalization is not just novel, it is **impossible to complete rigorously**. A formal simulation relation between instrumented and uninstrumented CPython execution would require modeling `sys.settrace`'s interaction with C extensions, reference counting, the GIL, and pandas' internal Cython dispatch. No one has formalized CPython's tracing semantics — not even CPython's own developers document it precisely. The Mathematician is rating the novelty of a proof that *cannot be written* at the required level of rigor. The pragmatic solution — defining a concrete list of instrumentation guarantees and proving soundness conditioned on those guarantees — is what the Mathematician recommends but is operationally just "we assume the instrumentation works and prove things conditional on that assumption." That is an assumption, not a theorem. Rate M-B2 honestly: ★★☆ novelty for the *idea*, ★☆☆ for the *achievable proof*.

### Cross-Critique: Mathematician Challenges Skeptic

The Skeptic claims conditional soundness "is not soundness" and compares it unfavorably to "running the pipeline twice." This conflates two fundamentally different things. Running the pipeline twice gives an accuracy delta — a scalar that tells you *something changed* but not *what* or *how much information leaked where*. Conditional soundness gives per-feature, per-stage bit-bounds with a formal anchor in the data-processing inequality. The DPI-based induction over the observed DAG is a genuine mathematical argument, not "dynamic analysis with extra steps." The bounds are *provably correct for the observed execution*, which is the execution the user actually cares about — they are debugging their specific pipeline on their specific data. The Skeptic's "different path on next dataset" objection is real but marginal: most leakage in practice is structural (fit before split), not path-dependent, and structural leakage is path-independent.

Furthermore, the Skeptic's P_fail = 0.35 for OOM on provenance tracking is inflated. Roaring bitmaps compress sparse provenance efficiently; in practice, few pipelines contain Cartesian-product merges. The typical case — inner joins on unique keys — produces 1:1 provenance maps that compress to near-zero overhead. The pathological case (many-to-many merges) is real but affects <5% of Kaggle pipelines. A more realistic P_fail for the OOM concern is 0.10–0.15.

### Verdict on Approach B

**Strengths:** Eliminates the highest-risk component (Python static analysis). Achieves 100% pipeline coverage through observation. Provenance-parameterized bounds are materially tighter than worst-case analysis. Shortest time-to-demo (2–3 months). Graceful degradation under every failure mode.

**Fatal weaknesses:** Conditional soundness is a genuinely weaker claim that some reviewers will find unsatisfying. The trace semantics formalization cannot be completed with full rigor. Requires pipeline execution, excluding the "audit someone else's code" use case.

**Overall viability: High.** Approach B is the correct architectural foundation. Its weaknesses are bounded engineering challenges, not open research questions. The execution requirement is an acceptable tradeoff for the primary use case.

---

## Approach C: BitLeak — Sensitivity-Indexed Channel Types

### Skeptic's Critique

Approach C is a beautiful theory waiting for reality to disappoint it. **Composite P_fail ≈ 0.50.**

**Fatal Flaw 1: Tight channel capacity bounds for non-linear operations do not exist in closed form.** The tight capacity catalog (M-C2) is the centerpiece. But `RobustScaler` (median, IQR) involves order-statistic distributions without clean closed forms. `TruncatedSVD` capacity depends on eigenvalue distributions — a random matrix theory problem where Marchenko-Pastur applies only asymptotically. `TargetEncoder` capacity depends on $H(Y|group)$, which is dataset-specific with no universal tight bound. The catalog will contain ~15 genuinely tight bounds and ~65 dressed-up loose bounds. **P_fail = 0.45.** The mitigation (report Gaussian and distribution-free bounds) means the tool always shows two numbers, and practitioners won't know which to trust.

**Fatal Flaw 2: The type system framing is a venue mismatch.** ML reviewers at ICML/NeurIPS don't evaluate type systems — they evaluate empirical results. They will see $\Gamma \vdash e : \tau$ as obscurantism. PL reviewers at POPL/OOPSLA will note that sensitivity types are Lipschitz conditions on mutual information following trivially from DPI, and that the "principal type inference" is a forward DAG traversal — not the unification-based inference they associate with principality. The paper falls between two stools: too theoretical for ML, too applied for PL. **P_fail = 0.40.**

**Fatal Flaw 3: The ~90% coverage claim contradicts Approach A's 85%.** C's type inference must infer sensitivity *functions* (richer than taint labels), requiring *more* information than A's simpler inference. If A achieves 85% with simpler types, C achieving 90% with richer types is contradictory. Realistic coverage: 75–85%. **P_fail = 0.35.**

**Vacuity attacks:** "Tight within $O(\log d)$" for $d = 200$ features means $\approx 7.6\times$ looseness. The tool reports "≤ 28.5 bits" when truth is 3.7 bits. Tightness only works for $d < 20$, excluding most real feature engineering. Additionally, the calibration layer reintroduces execution — at which point BitLeak becomes "Approach B with extra type-theoretic notation." The $\delta_{id}$ fallback for custom transformers is *unsound* for amplifying operations and *vacuous* for attenuating operations — "graceful degradation" means "wrong or useless" outside the catalog.

**Competitor kill:** The tight capacity catalog (M-C2) is a standalone reference result publishable as a 4-page workshop paper without any type system machinery. A reviewer will ask: "If the catalog is the main contribution, why do I need the type system? Just apply capacity formulas directly to the DAG."

### Mathematician's Critique

Approach C has the highest theoretical ceiling and the most concentrated research risk.

**M-C1 (Sensitivity-Indexed Channel Types): ★★☆ novelty.** The sensitivity function concept is genuinely useful — tracking how leakage transforms through downstream operations enables tighter composition than naive DPI. But packaging it as a *type system* is ornamental dressing for what is operationally a forward dataflow analysis with operation-specific transfer functions — exactly what Approach A does, with different notation. The compositionality and monotonicity properties follow directly from DPI and monotonicity of mutual information, not from the type system structure. The claim that this is "the information-theoretic analog of Lipschitz types" is apt but cuts both ways: if the analogy is that clean, the contribution is "applying a known framework to a new domain," not inventing one. Proof feasibility: 1–2 months (essentially the same proof as M-A3 with different notation).

**M-C2 (Tight Capacity Catalog): ★★★ novelty.** This is the most practically important contribution and the most likely to be cited as a standalone reference. Tight bounds with proved tightness factors ($\kappa$ such that $C/\kappa \leq I_{true} \leq C$) are what distinguish "3.7 bits (tight within 2×)" from "370 bits." Individual entries range from ★☆☆ (mean/sum — textbook) to ★★★ (rank channel, Wishart channel — require bespoke information-theoretic derivations). However, feasibility is the critical concern: rank statistics lack clean closed-form capacity expressions; the PCA tightness proof "for $d \leq 50$" suggests a numerical rather than analytical argument. Proof feasibility: **high risk, 3–6 months**, with significant probability of failure for the hard operation classes.

**M-C3 (Principal Type Inference): ★☆☆ to ★★☆ novelty.** The forward-propagation algorithm is standard dataflow analysis. The principality claim requires greatest lower bounds in an infinite-dimensional sensitivity function space — non-trivial and not argued. The graceful degradation to $\delta_{id}$ shows this is really "forward propagation with a default fallback," which is what abstract interpretation already does. If principality is proved rigorously for the full infinite-dimensional space, upgrade to ★★☆. Proof feasibility: 1–2 months for the algorithm, uncertain for principality in the infinite-dimensional case.

**Overall assessment:** Invest in M-C2 (the capacity catalog) — it has standalone value regardless of framing. Drop the type system framing from M-C1; present sensitivity functions as transfer function annotations. Drop the "principal type" claim from M-C3 unless the infinite-dimensional GLB question is resolved.

### Cross-Critique: Skeptic Challenges Mathematician

The Mathematician rates M-C2 as ★★★ and calls it "the most practically important contribution." I challenge the word *practical*. A tight capacity bound for `RobustScaler` is information theory research — publishable, citable, and useless for building a tool. The Mathematician evaluates math in isolation: "Is this bound novel and tight?" But the Skeptic evaluates math in context: "Does this bound help a practitioner?" If tight bounds exist only for 15 operations and the other 65 get loose bounds, the tool's output is a patchwork of precise and vacuous numbers with no way for the user to distinguish which is which. The catalog's *standalone reference value* is real, but the Mathematician's claim that it is "practically important" confuses academic impact with engineering utility.

Furthermore, the Mathematician gives M-C1 ★★☆ and notes it is "operationally a forward dataflow analysis." I argue this is too generous. If M-C1 is operationally identical to A's abstract interpretation, then C's unique contribution is *solely* M-C2. And if M-C2 is a standalone reference result, the entire type system framing is marketing, not mathematics.

### Cross-Critique: Mathematician Challenges Skeptic

The Skeptic's "competitor kill" — that M-C2 can be published as a standalone workshop paper — proves too much. By the same logic, every component of every approach can be published independently, and no integrated system is ever worth building. The capacity catalog *gains value* from integration into a compositional analysis framework. A standalone catalog tells you "StandardScaler leaks ≤ 0.5 bits"; an integrated system tells you "after StandardScaler feeds into PCA which feeds into a cross-validated model, the end-to-end leakage of feature $j$ is ≤ 1.7 bits." The composition is where the value lives. The type system framing, while ornamental, does capture a real insight: sensitivity functions *compose*, and this composition is the mechanism for end-to-end bounds. Strip the notation and you still need the compositional engine — call it a type system or a dataflow analysis, the math is the same.

The Skeptic also claims the ~90% coverage figure "contradicts" Approach A's 85%. This is a non sequitur. Sensitivity functions require more information *per operation* but C's type inference operates on a smaller fragment (only typed operations, with $\delta_{id}$ fallback for the rest). The 90% refers to operations that get *non-trivial* types, not Python programs that are fully analyzable. C can achieve 90% operation coverage while having 80% Python coverage by assigning $\delta_{id}$ to unanalyzable Python constructs — worse than B's 100% but not contradictory with A's 85%.

### Verdict on Approach C

**Strengths:** M-C2's tight capacity catalog is a genuine, standalone contribution to information theory. The sensitivity-function composition insight (when stripped of type-theoretic notation) is the right way to propagate bounds through complex pipelines. The theoretical ceiling is the highest of all three approaches.

**Fatal weaknesses:** Tight bounds exist only for simple operations; complex operations degrade to loose-or-conditional bounds, collapsing C into a less-practical version of A. The type system framing is ornamental and creates a venue mismatch. The ~90% coverage claim is unsupported. Research risk on M-C2 is high (3–6 months, may fail for hard cases).

**Overall viability: Moderate as standalone; high as a component.** C should not be built as an independent system. Its capacity catalog and sensitivity-function composition should be extracted and integrated into B's architecture.

---

## Final Debate Summary

### Points of Agreement

1. **The fit-transform feedback channel is the central mathematical challenge.** Both evaluators agree that factoring `fit_transform` into a composition of an aggregation channel and a pointwise application channel (M-A3's decomposition lemma) is the single most important theorem across all proposals. It is ★★★ novel, genuinely load-bearing, and required by every approach.

2. **Approach A's Python static frontend is a project-killer.** The Skeptic estimates 50–65% real coverage (P_fail = 0.55); the Mathematician flags it as the "highest-risk component." The Difficulty Assessor rates it 🔴 genuinely hard. All evaluators agree: avoid static Python analysis.

3. **M-B3 (Reduced Product) and M-A2 (Widening) are ornamental.** The Skeptic calls M-B3 "notation over a min operation." The Mathematician calls it "nearly trivial" and rates it ★☆☆. M-A2 is textbook widening that should be replaced by pattern-specific unrolling. Both should be dropped.

4. **Approach B has the lowest risk and shortest critical path.** The Difficulty Assessor gives B the best difficulty-to-feasibility ratio with no research-blocking steps. The Skeptic assigns the lowest P_fail (0.30). The Mathematician calls B's math "the most honestly scoped."

5. **C's tight capacity catalog (M-C2) has standalone reference value.** Both evaluators rate it ★★★ and acknowledge it as the contribution most likely to be cited independently. The disagreement is over whether it needs a type system wrapper or can be applied directly.

### Points of Disagreement

1. **Is conditional soundness sufficient?** The Skeptic argues it is "not soundness" and undermines the paper's theoretical contribution. The Mathematician argues it is "honestly scoped" and sufficient for the primary use case. The Skeptic demands universal guarantees; the Mathematician accepts path-conditional guarantees as a pragmatic and publishable compromise.

2. **Is the type system framing valuable?** The Mathematician gives M-C1 ★★☆ and acknowledges the sensitivity-function insight, even while calling the type-theoretic packaging ornamental. The Skeptic argues the framing is pure marketing with ★☆☆ novelty and creates a venue mismatch that endangers publication. This disagreement is fundamentally about whether formal notation that captures a real insight adds value even when it adds no computational capability.

3. **Can the fit-transform lemma be proved?** The Mathematician estimates 2–4 months with "significant probability of needing to weaken the theorem statement." The Skeptic estimates a 40% chance it collapses to covering only three trivial estimators. The difference is whether the sufficiency assumption (that `fit` computes sufficient statistics) holds broadly enough to be useful — the Mathematician sees this as a provable-if-scoped claim, the Skeptic sees it as a research gamble.

4. **Is M-C2 practically important?** The Mathematician calls it "the most practically important contribution in Approach C." The Skeptic argues tight bounds for 15 operations surrounded by vacuous bounds for 65 others produces a patchwork output that confuses rather than helps practitioners. This reflects a fundamental tension: academic impact (novel bounds are citable) versus engineering utility (partial tightness may be worse than consistent looseness).

### Consensus Recommendation

The debate converges on a clear synthesis: **Approach B's dynamic-first architecture + Approach C's tight capacity catalog + Approach A's fit-transform decomposition lemma.**

Concretely, the winning approach should combine:

- **B's instrumentation-first architecture** (dynamic DAG extraction eliminates the Python static analysis problem — the highest-risk component across all proposals)
- **B's provenance-parameterized bound framework** (M-B1: exploit exact observed provenance $\rho$ for tighter bounds)
- **B's conditional soundness theorem** (M-B2: weaker but achievable and publishable, with simplified trace semantics — an explicit list of instrumentation guarantees rather than a full CPython formalization)
- **C's tight capacity catalog** (M-C2: the 15–20 operations with provably tight bounds replace B's generic channel capacity formulas for those operations, making output actionable)
- **A's fit-transform decomposition lemma** (from M-A3: the key theorem resolving the feedback-channel challenge, transplanted into B's conditional-soundness framework)

**Drop:** M-A1 (Galois insertion — use a sound abstraction relation), M-A2 (widening — use pattern-specific unrolling), M-B3 (reduced product — use a one-line min operation), M-C1's type-theoretic framing (present sensitivity functions as annotated transfer functions), M-C3's principality claim (use standard forward propagation).

**Estimated total math effort:** ~8 person-months for a non-ornamental, feasibility-tested mathematical foundation. The resulting system has B's coverage and pragmatism, C's precision where it matters, and A's strongest theorem — the best elements of each approach without their architectural liabilities.
