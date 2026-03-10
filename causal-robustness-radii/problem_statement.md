# CausalCert: Structural Stress-Testing for Causal DAGs

## Problem Statement

Every observational causal analysis rests on a directed acyclic graph (DAG) encoding the analyst's structural assumptions — which variables cause which, which confounders exist, which pathways are absent. When a reviewer asks "what if your DAG is wrong?", the analyst has no principled quantitative answer. Standard sensitivity analysis (Rosenbaum 2002, Cinelli & Hazlett 2020, the E-value of VanderWeele & Ding 2017) holds the DAG fixed and perturbs confounding strength: it asks "how strong would an unmeasured confounder need to be to explain away the effect?" but never asks the deeper question — *which structural assumptions are load-bearing and which are cosmetic?* Structure learning algorithms (PC, GES, FCI) find a DAG from data but provide no guarantee that the learned structure supports the same causal conclusion as the true one. There exists no tool for systematically stress-testing a causal DAG's assumptions against data and quantifying which edges are fragile. This is the gap CausalCert fills.

We introduce two complementary diagnostics. First, **per-edge fragility scores** — for each edge (or absent edge) in the assumed DAG, the score quantifies how sensitive the causal conclusion is to misspecification of that specific assumption. A fragility score of 1.0 means removing/adding/reversing that edge alone destroys the conclusion; a score near 0 means the edge is cosmetic. The fragility report tells the analyst: "your conclusion depends critically on the absence of a direct effect from Z to Y, but is robust to whether W→V is present." This is the primary, immediately actionable output. Second, the **structural robustness radius** — the minimum number of simultaneous edge additions, deletions, or reversals that would overturn the conclusion (render the target effect non-identifiable, reverse its sign, or push it outside a claimed bound). A radius of 1 means a single misspecified edge can break the conclusion; a radius of 4 means the conclusion survives any combination of up to 3 structural errors. The radius provides a single-number summary; the fragility scores provide the actionable detail. Together, they transform the qualitative debate about DAG correctness into a quantitative stress test.

**Relation to prior work.** IDA (Maathuis et al. 2009) computes the range of causal effects across all DAGs in the Markov equivalence class (MEC), providing structural robustness *within* the MEC. CausalCert goes beyond: it searches the k-edit neighborhood of the assumed DAG (which may span multiple MECs) and evaluates conclusion validity at each perturbation. SID (Peters & Bühlmann 2015) measures structural intervention distance between DAGs — a conclusion-agnostic metric — whereas CausalCert's radius is conclusion-specific: the same DAG may have radius 4 for one query and radius 1 for another. The E-value (VanderWeele & Ding 2017) quantifies robustness to unmeasured confounding strength (continuous, calibratable against observed covariates). The robustness radius quantifies robustness to structural misspecification (discrete, calibratable against specific named edges). These are complementary, not competing, axes of robustness.

Computing fragility scores and the robustness radius requires solving two interlocking problems. **Falsification:** systematically testing whether the assumed DAG is compatible with the observed data by attacking conditional independence (CI) implications of the Markov property with an ensemble of nonparametric CI tests (kernel-based via KCI with Nyström approximation for smooth nonlinear dependence, plus rank-based tests for monotone dependence) aggregated via the Cauchy combination test. We confront the Shah & Peters (2020) impossibility result — no uniformly valid nonparametric CI test exists — head-on: our ensemble provides *practically useful* falsification by combining tests with complementary power profiles, while honestly reporting that a passing audit means "unfalsified by this battery," not "validated." CRT (conditional randomization test) is available as an optional add-on for mixed/high-dimensional conditioning sets when runtime budget permits. Multiplicity across CI implications is controlled via the Benjamini-Yekutieli procedure, with ancestral-set pruning to limit the number of tests. **Repair & Search:** we search for the nearest data-consistent DAG via a conflict-driven search over Markov equivalence classes (MECs), inspired by CDCL in SAT solving. Failed repair attempts generate no-good clauses — minimal subsets of edge edits proven inconsistent — that prune the search tree, with LP relaxation providing anytime lower bounds on the optimal edit distance. The search is capped at k_max = 3 by default (configurable), keeping computation tractable while covering the empirically relevant regime.

The system outputs a **structural audit report**: per-edge fragility scores, the robustness radius (up to k_max), a ranked list of nearby data-consistent DAGs with their edit distances, and for each surviving DAG, the set of valid back-door identification strategies with corresponding effect estimates and confidence intervals via cross-fitted AIPW. The report answers: "Here are the edges whose misspecification would break your conclusion, and here are the nearest alternative DAGs consistent with your data."

**Scope boundary.** CausalCert focuses on structural robustness — which edges matter and how many can be wrong. Parametric sensitivity analysis (DRO bounds, covariate-calibrated sensitivity, breakdown frontiers combining structural and parametric uncertainty) is an important extension deferred to a follow-on system (CausalCert-Sensitivity). This separation keeps the core system focused, computationally tractable, and evaluable within a single paper. Front-door and IV identification strategies are similarly deferred: the core system handles back-door adjustment, which covers the majority of applied causal analyses. The complete ID algorithm, cross-strategy comparison, and estimand separation (ATE vs. LATE) are planned extensions.

## Value Proposition

**Who needs this.** Applied researchers in epidemiology, economics, political science, and social science who specify causal DAGs as the basis for effect estimates. Reviewers and journal editors who must evaluate whether a DAG's assumptions are defensible. Methodologists developing sensitivity analysis tools who need structural-robustness diagnostics complementing parametric-sensitivity frameworks.

**Why desperately.** Every empirical researcher who has drawn a causal DAG has faced the question "but what if your DAG is wrong?" and had no principled response beyond appealing to domain knowledge. Sensitivity analysis (E-value, sensemakr) quantifies robustness to unmeasured confounding *within* a fixed graph but is silent on structural misspecification — the possibility that edges are missing, reversed, or spurious. Structure learning (PC, GES) finds a graph from data but does not connect the learned structure to a specific causal conclusion's validity. No existing tool tells the analyst *which specific edges are load-bearing for their conclusion*. Per-edge fragility scores fill this gap directly and immediately: "your estimate of the effect of X on Y depends critically on the assumption that Z does not directly cause Y — consider whether this assumption is defensible." This is the structural analogue of covariate-calibrated sensitivity: instead of "how strong would unmeasured confounding need to be?", it answers "which specific structural assumptions carry the weight?"

**What becomes possible.** Researchers can report fragility scores alongside effect estimates, directing reviewer scrutiny to the assumptions that matter most. A fragility report identifies the 2–3 edges whose presence/absence is load-bearing, enabling focused domain-expert deliberation rather than diffuse "is your DAG right?" debate. The robustness radius provides a single summary statistic for overall structural robustness. Together, they give the causal analyst a structural stress-test analogous to how sensitivity analysis provides a parametric stress-test.

**Goldilocks validation gate.** Before committing to the full build, a pilot study will compute fragility scores on 5+ published causal DAGs (Hernán & Robins textbook examples, Tennant et al. 2021 survey). If fragility scores show meaningful variation (some edges fragile, some robust), the concept has bite. If all edges are trivially fragile or none are, the concept needs rethinking. This pilot gates the full implementation.

## Technical Difficulty

The system presents three core hard problems, each requiring genuine algorithmic innovation.

**1. Robustness radius computation is NP-hard; FPT in treewidth via tree decomposition DP.** The decision problem — does there exist a DAG within edit distance k that overturns the conclusion? — is NP-hard by reduction from minimum vertex cover on the moral graph (Theorem 2). Brute-force enumeration of the k-edit neighborhood is infeasible: for p = 30 and k = 3, the candidate space exceeds 10⁸. Tractability comes from two results. First, the locality property (Theorem 1c): edits outside the ancestors of {X, Y} cannot change the robustness radius, restricting the search to O(|An(X,Y)|²) potential edits. Second, FPT in treewidth (Theorem 3): dynamic programming on the tree decomposition of the moral graph solves the problem in O(2^{O(w²)} · p) time. The practical regime is w ≤ 5 of the moral graph, which is polynomial. At w = 6, runtime is hours; beyond w = 6, exact computation is infeasible and the system switches to LP-relaxation lower bounds with heuristic search upper bounds, clearly flagging that the reported radius is a bound rather than exact. **Treewidth of published DAGs:** We will survey the moral-graph treewidths of DAGs in the DAGitty repository, Tennant et al. (2021) catalogue, and Hernán & Robins textbook. Preliminary inspection of canonical examples (LaLonde: p=8, smoking/birthweight: p=12, IHDP: p=25) suggests moral-graph treewidth ≤ 5 for p ≤ 15 and 5–8 for p = 15–30. The system honestly reports when it exits the exact-computation regime.

**2. Conflict-driven MEC search is a genuinely new algorithm combining SAT-solver techniques with graphical model structure.** No existing algorithm solves minimum-edit DAG repair — GES maximizes a score from scratch, not edit distance from a reference. Our search adapts CDCL: when a partial set of edits produces a DAG inconsistent with surviving CI constraints, conflict analysis extracts the minimal inconsistent subset as a no-good clause that prunes all supersets. LP relaxation of the integer program (binary edit variables, acyclicity and CI-consistency constraints) provides anytime lower bounds. Correctness requires proving a monotonicity lemma — that the inconsistency of a partial edit set implies inconsistency of all supersets containing its conflicts — and completeness of the clause-learning scheme. The default search depth is k_max = 3 (configurable to 5), keeping the branching factor bounded.

**3. Ensemble CI testing with honest impossibility handling.** Shah & Peters (2020) prove that no nonparametric CI test has non-trivial power against all alternatives uniformly. Our response is to work within this impossibility: deploy an ensemble with complementary power profiles (KCI/Nyström for smooth nonlinear dependence, rank-based for monotone alternatives), aggregate via the Cauchy combination test (valid under arbitrary dependence; Liu & Xie 2020), and honestly report the power envelope. CRT is available as an optional high-power mode (at ~10× runtime cost) for settings where the default ensemble lacks power. The audit explicitly states "robust to structural perturbations *detectable by this test battery at this sample size*." Multiplicity is controlled via Benjamini-Yekutieli with ancestral-set pruning reducing the effective test count.

**Additional algorithmic contributions:**
- **Incremental d-separation (Theorem 7):** O(p · d_max) algorithm identifying which CI relations change under a single edge edit, avoiding O(p²) full recomputation per search step. Clean result, essential for search efficiency.
- **Per-edge fragility scoring:** For each edge in An(X,Y), compute whether adding/removing/reversing it (a) changes identification status, (b) changes the sign of the identified effect, or (c) introduces a testable CI violation. This is a structured BFS over single-edit perturbations — simpler than the full radius computation but already novel and useful.

### Subsystem Breakdown

| # | Subsystem | Est. LoC | Key Challenge |
|---|-----------|----------|---------------|
| 1 | DAG Engine & d-Separation Compiler | ~5K | Incremental d-separation under edge edits; ancestral-set enumeration; CI implication generation with pruning. Uses networkx for graph primitives; novel code for incremental updates. |
| 2 | CI Testing Engine | ~8K | KCI with Nyström approximation (m=500 inducing points), rank-based CI tests, Cauchy combination, BY FDR control. Optional CRT module. Wraps sklearn kernels; novel code for test aggregation and warm-start kernel caching. |
| 3 | Conflict-Driven MEC Search | ~10K | CDCL-style no-good learning on the MEC lattice; LP relaxation lower bounds; k_max-bounded search. This algorithm does not exist in any prior implementation. |
| 4 | Fragility Scorer & Radius Computer | ~7K | Per-edge fragility via single-edit BFS; radius via k-edit search using subsystem 3; incremental ID evaluation at each candidate. |
| 5 | Back-Door Identification & Estimation | ~6K | Back-door adjustment set enumeration (wraps existing algorithms); cross-fitted AIPW estimation via DoubleML/EconML; influence-function variance. Novel code: strategy→estimator mapping and incremental re-enumeration under DAG edits. |
| 6 | Reporting & Audit Generator | ~4K | Structured JSON + rendered HTML: fragility scores, radius, repair frontier, effect estimates. Plain-language narrative via templates. |
| 7 | Data Layer | ~3K | CSV/Parquet ingestion; type inference (continuous/ordinal/nominal); routing to appropriate CI test family. Standard engineering. |
| 8 | Pipeline Orchestrator & Caching | ~3K | DAG of subsystem dependencies; memoized CI test results and kernel matrices; deterministic replay via logged seeds. |
| 9 | Evaluation Framework | ~12K | Synthetic DGP generation with known radii (exhaustive verification on small graphs); semi-synthetic benchmarks; scalability profiling; ablation harness; coverage verification via Monte Carlo (200 DGPs). |
| | **Total** | **~58K** | |

**Honest accounting of novelty vs. wrapping:** Subsystems 3 (CDCL search, ~10K) and 4 (fragility/radius, ~7K) are entirely novel algorithms. Subsystem 1 (incremental d-sep, ~2K novel within 5K) and subsystem 2 (test aggregation/caching, ~3K novel within 8K) mix novel code with library wrapping. Subsystems 5–8 are primarily integration engineering with novel glue. Subsystem 9 (evaluation) has ~3K of novel DGP-construction code. **Total genuinely novel: ~25K LoC. Total with integration + testing + evaluation: ~58K LoC.**

## New Mathematics Required

Four load-bearing theorems plus one algorithmic result form the mathematical backbone. Each directly enables a computational capability.

**Theorem 1 (Structural Robustness Radius — Well-Definedness, Monotonicity, Locality).** For DAG G, causal query Q = E[Y | do(X=x)], and conclusion C (e.g., "ATE > 0"), define r(G, Q, C) = min{d_edit(G, G') : G' violates C}, where violation means Q is non-identifiable in G', or the identified effect contradicts C. Prove: (a) r is well-defined and finite; (b) r(G, Q, C) ≥ r(G, Q, C') whenever C' ⊆ C (monotonicity); (c) r depends only on the ancestors of {X, Y} in G (locality). Property (c) restricts the search from O(p²) to O(|An(X,Y)|²) potential edits. *Enables:* tractable radius computation (subsystem 4).

**Theorem 2 (Minimum-Edit DAG Repair is NP-Hard).** The decision problem — given DAG G, CI constraints S, integer k, does G' exist with d_edit(G, G') ≤ k consistent with S? — is NP-hard by reduction from minimum vertex cover on the moral graph. *Enables:* justification for conflict-driven search rather than polynomial-time algorithms.

**Theorem 3 (FPT in Treewidth).** Minimum-edit DAG repair is fixed-parameter tractable in the treewidth w of G's moral graph: O(2^{O(w²)} · p) via DP on the tree decomposition. Practical regime: w ≤ 5 (polynomial time). At w = 6: hours. Beyond w = 6: the system provides LP-relaxation lower bounds and heuristic upper bounds rather than exact solutions. *Enables:* exact computation for the sparse DAGs common in applied work; graceful degradation for denser DAGs.

**Theorem 4 (Cauchy Combination Validity for CI Ensembles).** For CI tests T₁, ..., T_m with arbitrary dependence, the Cauchy combination p-value controls Type I error at level α (Liu & Xie 2020). Verify applicability to the CI testing setting with overlapping variable subsets. Combined with BY FDR across CI implications. *Enables:* valid ensemble CI testing (subsystem 2).

**Theorem 7 (Incremental d-Separation Under Edge Edits).** O(p · d_max) algorithm identifying all CI relations that change between G and G' (single edge edit). Correctness by structural induction. *Enables:* efficient search (subsystem 3) without O(p²) recomputation per step.

**Deferred mathematics (CausalCert-Sensitivity, Paper 2):**
- Theorem 5 (DAG-Structured DRO Dimensionality Reduction): O(p · max|pa|) dual variables under DAG compatibility constraints.
- Theorem 6 (Finite-Sample Pipeline Validity): End-to-end coverage guarantee.
- These are genuine contributions but belong in the sensitivity extension, not the core structural-robustness paper.

**Load-bearing assumptions.** (A1) i.i.d. sampling; (A2) positivity/overlap for back-door identification; (A3) Markov property; (A4) faithfulness — the most controversial, as it can fail at measure-zero parameter values and near-faithfulness violations degrade CI test power. The system explicitly reports sensitivity to faithfulness violations via the power envelope characterization. Without faithfulness, CI testing cannot reliably detect structural violations, and the audit honestly reports this limitation.

## Best Paper Argument

**Focused contribution: structural robustness as a new diagnostic axis for causal inference.**

The structural robustness radius is a genuinely new inferential primitive (novelty 7/10 after accounting for IDA and SID). No prior paper computes conclusion-specific minimum structural break distance or per-edge fragility scores for a reference DAG. The closest work: IDA (effect ranges within MECs — within-MEC only, not edit-distance-specific), SID (DAG-to-DAG distance — conclusion-agnostic), sensitivity analysis (parametric robustness — DAG fixed).

The CDCL adaptation to MEC search is a genuinely new algorithm (novelty 8/10). The minimum-edit DAG repair problem has never been formalized, let alone solved. Conflict-driven clause learning, LP relaxation bounds, and the NP-hardness + FPT results constitute a solid computational contribution.

**Paper structure (8 pages + appendix):**
1. Define structural robustness radius and per-edge fragility scores (Section 2)
2. Prove NP-hardness (Theorem 2) and FPT in treewidth (Theorem 3) (Section 3)
3. Present the CDCL MEC search algorithm with correctness proof (Section 4)
4. Ensemble CI testing with Cauchy combination (Section 5)
5. Empirical: fragility analysis of 10+ published causal DAGs (Section 6)
6. Synthetic evaluation: radius recovery, scalability, ablation (Section 7)

**The killer empirical result:** "We stress-tested 15 published causal DAGs from epidemiology and economics. In 11/15, we identified at least one edge whose misspecification alone would overturn the stated conclusion. The median robustness radius was 2. Here are the fragile edges in each DAG." This result — if it materializes — would be immediately impactful. Applied researchers would want to run this tool on their own DAGs.

**Best venue:** UAI or NeurIPS causal inference track. The contribution is both conceptual (a new diagnostic primitive) and computational (new algorithms with complexity results). Best-paper potential depends on the empirical results: if fragility scores reveal interesting structure in published DAGs, this is a strong contender at UAI.

**The pitch:** *"Which edges in your causal DAG actually matter for your conclusion?"*

## Evaluation Plan

All evaluation is fully automated — no human annotation, no user studies, no subjective judgments.

**Goldilocks pilot (Stage 0 — gates full build).** Compute fragility scores on 5 published causal DAGs (LaLonde, IHDP, Cattaneo smoking/birthweight, Hernán & Robins textbook Chapter 6, Tennant et al. 2021 Example 1). Success criterion: at least 3/5 DAGs show meaningful fragility variation (some edges fragile, some robust). If all edges are uniformly fragile or uniformly robust, the concept needs fundamental rethinking. This pilot requires only subsystems 1, 2, and the fragility-scoring portion of 4 — implementable in ~15K LoC.

**Synthetic DGP benchmark.** Generate DAGs with *known* robustness radii by construction: start with a DAG G and conclusion C, verify by exhaustive search (feasible for p ≤ 12) that the true radius is r. Test whether CausalCert recovers r. Vary p (8–30), treewidth (2–6), and r (1–3).

**Semi-synthetic benchmark.** Real covariate distributions from standard causal datasets with synthetic treatment/outcome mechanisms calibrated to produce known effects and known robustness properties.

**Published DAG re-audit.** Apply CausalCert to 10–15 published causal DAGs from epidemiology and economics. Compute fragility scores and robustness radii. Report: which DAGs are falsified? What are typical radii? Which edges are fragile? This is the empirical contribution demonstrating real-world bite.

**Scalability benchmarks.** Wall-clock time vs. p, w, k, n. Demonstrate: (a) under 30 minutes for p ≤ 15, w ≤ 4, n ≤ 50K on a laptop CPU; (b) under 2 hours for p ≤ 25, w ≤ 5, n ≤ 100K; (c) graceful degradation (LP bounds + heuristic) for p = 30–50.

**Ablation.** Replace CI ensemble with single test (Fisher-Z); replace CDCL with brute-force enumeration; remove locality pruning (Theorem 1c). Measure: radius accuracy, runtime, fragility-score agreement.

**Coverage verification.** Monte Carlo with 200 DGPs from known models, checking that fragility scores correctly identify the truly fragile edges and the reported radius lower-bounds the true radius at the stated confidence level.

**Baselines.** dagitty (testable implications only), DoWhy (identification + estimation, no structural robustness), sensemakr (parametric sensitivity, DAG fixed), IDA via pcalg (effect range within MEC — complementary, not competing). CausalCert provides fragility scores and structural radii that no baseline offers.

## Laptop CPU Feasibility

All computation is combinatorial search, linear algebra, and kernel evaluations — nothing requires GPUs.

**Honest runtime estimates (single laptop CPU core):**

| Regime | p | w (moral) | n | CI Tests | Search | Estimation | Total |
|--------|---|-----------|---|----------|--------|------------|-------|
| Pilot (Stage 0) | ≤15 | ≤4 | 10K | 5–15 min | 2–5 min | 5 min | **15–30 min** |
| Standard | ≤25 | ≤5 | 50K | 30–90 min | 10–30 min | 15 min | **1–2 hours** |
| Extended | ≤35 | ≤6 | 100K | 2–6 hours | 1–4 hours | 30 min | **4–10 hours** |
| Large (approx.) | ≤50 | >6 | 100K | 4–10 hours | LP bounds only | 1 hour | **6–12 hours** |

**Key design choices for CPU feasibility:**
- **Default CI ensemble:** KCI (Nyström m=500) + rank-based tests. CRT is optional (adds ~10× per test). KCI at n=50K with Nyström: O(n·m²) = O(50K × 250K) ≈ 12.5B FLOPs → ~1.5s per test. With ~500 tests after ancestral pruning (p=25): ~750s ≈ 13 min. Kernel matrix caching across overlapping tests provides 2–3× speedup.
- **k_max = 3 default:** Caps search depth, bounding the branching factor. Configurable to 5 for users willing to wait.
- **Precomputed kernel caching:** Kernel matrices are computed once and reused across CI tests sharing variables.
- **Back-door only (core system):** Avoids the combinatorial explosion of enumerating all identification strategies. DoubleML/EconML handle estimation.

**Memory:** Nyström kernel matrices at n=50K, m=500 → 50K × 500 × 8 bytes = 200MB per matrix. With ~10 unique conditioning sets active at a time: ~2GB. Fits in laptop RAM.

**Evaluation runtime:** 200 DGPs × ~15 min each (pilot regime) = 50 hours. Parallelizable across CPU cores (4 cores → ~12 hours). Feasible on a laptop over a weekend.

No human annotation or human studies appear anywhere in the pipeline.

## Risks and Honest Limitations

**Goldilocks problem for radii (HIGH — gated by pilot).** If most published DAGs have radius 0 (already falsified) or 1 (trivially fragile), the radius concept is less interesting — though fragility scores may still show useful variation. If typical radii are > 3, exact computation is expensive. The Goldilocks pilot (Stage 0) resolves this before the full build. Risk is HIGH but mitigable and gated.

**Treewidth ceiling (MEDIUM-HIGH).** Exact robustness radii are computable for moral-graph treewidth ≤ 5. Beyond this, the system provides LP-relaxation lower bounds and heuristic upper bounds. We will survey published DAGs' moral-graph treewidths; if most fall in w ≤ 5, the exact algorithm covers the applied regime. If not, the system remains useful via bounds but loses the "exact radius" claim for large DAGs.

**Shah-Peters impossibility (MEDIUM).** The audit is conditional on the CI test battery detecting violations. Adversarial dependence structures evading all tests yield vacuously high radii. Mitigation: honest reporting ("unfalsified by this battery at this sample size") rather than unconditional guarantees. The system is a stress-test, not a formal certificate.

**Faithfulness assumption (MEDIUM).** Without faithfulness, CI testing is unreliable and radii are meaningless. Near-faithfulness violations (e.g., two causal paths that nearly cancel) degrade CI test power, producing false negatives. The power envelope characterization honestly reports detection capability as a function of departure strength. The system explicitly warns when it cannot distinguish "robust" from "low power."

**Error compounding across pipeline stages (MEDIUM).** CI testing errors (false positives/negatives) propagate to repair and estimation. BY correction is conservative, reducing power. Mitigation: the system reports individual CI test results alongside the aggregate, allowing users to examine borderline cases. The fragility scores (single-edit perturbations) are less sensitive to multiplicity than the full radius computation (k-edit neighborhoods).

**Computational tractability at scale (MEDIUM).** For DAGs with p > 35 and k > 3, exact search is expensive (hours). The system provides anytime LP bounds and heuristic search for larger instances. The honest runtime table above reflects real expectations, not aspirational claims.

---

`causal-robustness-radii`
