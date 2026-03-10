# CausalCert — Final Synthesized Approach

## 1. Title and Pitch

**CausalCert: Per-Edge Fragility Scores and Exact Robustness Radii for Causal DAGs via ILP-Backed Structural Stress Testing.**

Every published causal DAG contains unjustified edges — CausalCert tells you exactly which ones would overturn your conclusions, and how many simultaneous misspecifications your analysis can survive, backed by NP-hardness theory that explains why this is fundamentally hard and FPT algorithms that solve it anyway.

## 2. Approach Overview

CausalCert is a structural stress-testing system for causal directed acyclic graphs (DAGs). Given a user-specified DAG, treatment-outcome pair, observational data, and a causal conclusion (e.g., "treatment X has a back-door admissible adjustment set for outcome Y"), CausalCert computes two outputs: (1) **per-edge fragility scores** quantifying each edge's contribution to maintaining the conclusion, and (2) a **robustness radius** — the minimum number of simultaneous edge additions/deletions/reversals that would overturn the conclusion, reported as an honest bracket [lower bound, upper bound].

The system fuses the best elements from three competing designs. From Approach A, it inherits the fragility-first philosophy: single-edit BFS over the ancestral set computes per-edge scores that are immediately actionable for applied researchers. From Approach B, it inherits the theoretical backbone: NP-hardness of minimum-edit DAG repair (justifying why exact computation requires serious algorithms) and fixed-parameter tractability in treewidth (explaining when the problem is solvable). From the debate consensus, it inherits the ILP-first strategy: formulate radius computation as an integer linear program and let Gurobi/CPLEX solve it before building any custom solver. CDCL is an upgrade path, not a launch requirement.

The development is strictly phased with a pilot gate. Phase 0 computes fragility scores on 5 published DAGs (Sachs, Asia, Alarm, Insurance, a user-contributed epidemiology DAG). If fewer than 3/5 show meaningful fragility variation (spread > 0.3 on [0,1]), the project stops. This pilot costs 6 weeks and ~4K LoC — a bounded bet. Only after the pilot passes does full development proceed through core fragility scoring, ILP-based radius computation, and paper-ready evaluation.

## 3. Value Proposition

### Primary User: Applied Causal Researcher

**Persona:** Dr. Maria Chen, epidemiologist studying air pollution → childhood asthma with a 12-node DAG. A reviewer writes: *"You assume no direct SES → Diagnosis edge. Justify this."* Maria has no tool to answer systematically. She can reason about individual edges but cannot assess which of her 15 assumed absences are load-bearing.

**What CausalCert gives her:** `causalcert audit --dag dag.dot --treatment Pollution --outcome Asthma --data cohort.csv` returns a ranked fragility table:

```
Edge              Fragility   Impact
SES → Diagnosis   0.92        Removes all valid adjustment sets
Genetics → SES    0.41        Changes AIPW estimate by 34%
Pollution → SES   0.03        No effect on identification
```

**Time saved:** From 2–3 days of manual sensitivity analysis to 8 minutes of automated audit for p ≤ 20.

### Secondary User: Causal Inference Methodologist

**Persona:** Prof. Jonas Petersson, who studies computational complexity of causal problems. CausalCert provides: (a) the first formal complexity classification of minimum-edit DAG repair, (b) FPT algorithms he can extend, and (c) empirical benchmarks for the robustness radius problem.

### Why Desperately Needed

1. **No existing tool computes structural robustness.** dagitty tests implied CIs but doesn't score edges by fragility. sensemakr handles unmeasured confounding but not structural misspecification. DoWhy refutes but doesn't rank.
2. **Reviewer pressure is real.** Causal DAGs in epidemiology, economics, and social science face increasing scrutiny. Researchers need defensible answers about which assumptions are load-bearing.
3. **The E-value analogy.** VanderWeele & Ding (2017) gave a number — "unmeasured confounding must be this strong to explain away the effect." CausalCert gives an analogous number for structural assumptions: "this many edges must be wrong to overturn the conclusion."

## 4. Technical Architecture

### Subsystem Breakdown

| Subsystem | Description | LoC (Total) | LoC (Novel) | Wrapping |
|-----------|-------------|-------------|-------------|----------|
| **DAG Model & Ancestral Set** | Graph representation, ancestral set computation, edit operations (add/delete/reverse), acyclicity enforcement | 2,500 | 1,200 | networkx |
| **Incremental d-Separation** | O(p·d_max) d-separation update under single-edge edit, avoiding full BFS recomputation | 1,800 | 1,800 | — |
| **CI Test Ensemble** | Fisher-Z (continuous), χ² (discrete), KCI/Nyström (nonparametric), Cauchy combination test, BY multiplicity correction | 3,500 | 1,500 | scipy, sklearn, causal-learn |
| **Per-Edge Fragility Scorer** | Single-edit enumeration over An(X,Y), fragility score = max over CI/ID/estimation impact channels | 2,500 | 2,500 | — |
| **ILP Radius Solver** | Minimum-edit DAG repair as integer linear program: edge indicator variables, acyclicity constraints (topological), d-separation constraints, Markov blanket constraints | 3,000 | 2,500 | python-mip or gurobipy |
| **LP Relaxation (Lower Bound)** | Continuous relaxation of ILP for anytime lower bound; integrality gap ≤ O(w) | 1,200 | 800 | scipy.optimize or python-mip |
| **Effect Estimation** | AIPW doubly-robust estimator under original and perturbed DAGs | 2,000 | 400 | econml, DoWhy |
| **Evaluation Harness** | Synthetic DGP generation, known-radius ground truth, benchmark suite | 4,000 | 2,000 | — |
| **CLI & Report** | Command-line interface, JSON/HTML output, fragility heatmap | 3,500 | 1,000 | click, jinja2 |
| **Tests & CI** | Unit tests, integration tests, property-based tests | 8,000 | 2,000 | pytest, hypothesis |
| **FPT / Tree Decomposition (Theory Only)** | Reference implementation for w ≤ 3, primarily for paper validation | 2,000 | 2,000 | — |
| **TOTAL** | | **~34,000** | **~17,700** | |

### Dependency Stack

**Core (no optional dependencies):** Python 3.10+, networkx, numpy, scipy, scikit-learn, click, jinja2.

**Optional:** gurobipy (commercial ILP; fallback: python-mip with CBC), econml, DoWhy, causal-learn.

**Explicitly excluded:** R dependencies (no rpy2), JavaScript visualization (plain HTML/SVG only).

### Data Flow

```
User DAG (DOT/GML) + Data (CSV) + Query (treatment, outcome, conclusion)
  │
  ▼
[Ancestral Set Extraction] ── restricts all computation to An_G(X,Y)
  │
  ├──▶ [Single-Edit Enumeration] ──▶ [Per-Edge Fragility Scores]
  │         │
  │         ├── d-sep channel: incremental d-sep check
  │         ├── ID channel: back-door criterion check under edit
  │         └── EST channel: AIPW re-estimation under edit (optional, data-dependent)
  │
  ├──▶ [ILP Radius Solver] ──▶ [Exact Radius or Timeout → Upper Bound]
  │
  ├──▶ [LP Relaxation] ──▶ [Lower Bound on Radius]
  │
  └──▶ [Report Generator] ──▶ JSON + HTML fragility table + radius bracket
```

## 5. New Mathematics

### Theorem 1: Fragility Locality

**Statement.** Let G = (V, E) be a DAG, X the treatment, Y the outcome, and C a back-door conclusion. For any edge e = (i → j) with {i, j} ∩ An_G(X, Y) = ∅, the fragility score F(e) = 0: editing e cannot change C.

**Why load-bearing.** Restricts computation from O(p²) candidate edges to O(|An|²). For typical published DAGs, |An|/p ≈ 0.4–0.7, yielding 2–6× speedup. Without this, the tool is impractically slow for p > 15.

**Proof difficulty:** 2/10. Essentially folklore (Verma 1993, Lauritzen 1996), but needs formal statement for the edit-perturbation context. The proof is a straightforward argument: edges outside An(X,Y) cannot create or destroy paths between X and Y, hence cannot affect d-separation or back-door adjustment.

**Known risks:** None. This is rock-solid.

---

### Theorem 2: NP-Hardness of Minimum-Edit DAG Repair

**Statement.** Given a DAG G, a set of CI constraints C derived from data, and an integer k, deciding whether there exists a set of ≤ k edge edits (additions, deletions, reversals) transforming G into G' such that G' satisfies all constraints in C and maintains a specified back-door conclusion is NP-complete.

**Why load-bearing.** Justifies the entire algorithmic program. Without NP-hardness, a reviewer asks "why not just enumerate?" This theorem proves that exact radius computation requires exponential time in the worst case, motivating the ILP formulation and FPT algorithms.

**Proof difficulty:** 4/10. Reduction from minimum vertex cover on the moral graph. The key insight: each vertex in the moral graph corresponds to an edge in G, and covering all "violated" CI constraints corresponds to covering edges in the moral graph. Encoding acyclicity adds a polynomial-time check that doesn't change the complexity class.

**Known risks:** Low. The reduction is standard in structure. The main subtlety is ensuring acyclicity constraints don't collapse the problem to P (they don't, because acyclicity checking is in P and composing an NP-hard problem with a P-time constraint preserves NP-hardness).

---

### Theorem 3: Fixed-Parameter Tractability in Treewidth

**Statement.** Minimum-edit DAG repair is FPT parameterized by the treewidth w of the moral graph of An_G(X, Y): solvable in O(3^{O(w²)} · p) time via dynamic programming on a tree decomposition.

**Why load-bearing.** Explains *when* the problem is tractable despite NP-hardness. Published causal DAGs typically have treewidth w ≤ 4 (sparse, locally connected). This theorem guarantees polynomial-time exact solutions for the practical regime.

**Proof difficulty:** 7/10. This is the deepest mathematics in the project. The DP must track, for each bag in the tree decomposition, the edit status of all edges incident to bag vertices (3 states per edge: original, deleted, added) and verify d-separation constraints locally. The join operation combining child-bag states is the hardest step — requires proving that local d-separation checks compose correctly across the tree decomposition.

**Known risks:** Medium. The constant in 3^{O(w²)} determines practical feasibility. At w = 3: 3^9 = 19,683 states per bag (tractable). At w = 4: 3^16 ≈ 43M states per bag (feasible with pruning). At w = 5: 3^25 ≈ 847B states per bag (infeasible). **The paper must honestly state the practical regime as w ≤ 3–4, not w ≤ 5.** The reference implementation targets w ≤ 3 with optional w = 4 support.

---

### Theorem 4: LP Relaxation Gap Bound

**Statement.** The integrality gap of the LP relaxation of the minimum-edit DAG repair ILP is at most O(w), where w is the treewidth of the moral graph.

**Why load-bearing.** Provides a *lower bound* on the true radius. Combined with the ILP upper bound (or greedy upper bound when ILP times out), this gives an honest bracket [LP_lower, ILP_upper] on the radius. Without this, the tool can only report upper bounds.

**Proof difficulty:** 5/10. Standard LP duality argument. The main technical content is constructing a feasible dual solution that certifies the gap bound. The treewidth dependence follows from the structure of d-separation constraints.

**Known risks:** Medium. The O(w) gap may be loose for specific instances. Empirically, the gap is often 0–1 for published DAGs (conjecture, to be validated in Phase 3). If the gap is consistently large, the lower bound is uninformative, reducing the tool to upper-bound-only reporting.

---

### Non-Theorem: CI Error Propagation (Open Problem)

**Statement (informal).** How do Type I/II errors in conditional independence tests propagate to errors in the robustness radius?

**Status:** This is identified as a gap by all three debate experts. No approach addresses it formally. The honest position: fragility scores and radii are *conditional on CI test results*. The tool reports "assuming these CI verdicts are correct, the radius is [l, u]." Formal error propagation (e.g., "if each CI test has error rate α, the radius estimate has error rate ≤ f(α, k, p)") is flagged as future work and explicitly listed as a limitation.

## 6. Phased Development Plan

### Phase 0: Goldilocks Pilot (Weeks 1–6) — GATES EVERYTHING

**Objective:** Determine whether per-edge fragility scores produce informative variation on real DAGs.

**Deliverable:** Fragility score computation on 5 published DAGs using single-edit BFS + Fisher-Z CI testing.

**DAGs:**
1. **Sachs (2005):** 11 nodes, 17 edges. Protein signaling. Well-studied ground truth.
2. **Asia (Lauritzen & Spiegelhalter 1988):** 8 nodes, 8 edges. Canonical toy example.
3. **Alarm (Beinlich 1989):** 37 nodes, 46 edges. Medical diagnosis. Tests scalability.
4. **Insurance (Binder 1997):** 27 nodes, 52 edges. Dense graph. Stress-tests ancestral set pruning.
5. **User-contributed epidemiology DAG:** To be sourced from published AJE/Epidemiology paper. Real-world messy structure.

**Implementation (~4K LoC):**
- DAG model + ancestral set extraction (800 LoC)
- Single-edit d-separation check (600 LoC)
- Fisher-Z CI test (400 LoC)
- Back-door criterion check (300 LoC)
- Fragility scorer loop (500 LoC)
- Pilot evaluation script (400 LoC)
- Tests (1,000 LoC)

**Go/No-Go Gate (end of Week 6):**
- **GO** if ≥ 3/5 DAGs show fragility spread > 0.3 (i.e., not all edges are fragility ≈ 0 or ≈ 1).
- **GO** if at least 1 DAG reveals a non-obvious fragile edge (one that domain experts wouldn't flag first).
- **NO-GO** if fragility scores are uniformly uninformative. Salvage: publish a negative result ("structural fragility is rare in published DAGs").

**Milestone:** Internal report with 5 fragility heatmaps, runtime measurements, and go/no-go recommendation.

---

### Phase 1: Core Fragility System (Weeks 7–14)

**Objective:** Production-quality per-edge fragility scoring with ensemble CI testing.

**Deliverables:**
- Incremental d-separation engine (O(p·d_max) per edit)
- Full CI test ensemble: Fisher-Z + KCI/Nyström + Cauchy combination + BY correction
- Three fragility channels: d-separation impact, identification impact, estimation impact
- Per-edge fragility score = max(d-sep channel, ID channel, EST channel) — no composite weighting, no designed metric
- CLI: `causalcert fragility --dag G.dot --treatment X --outcome Y --data D.csv`
- JSON + HTML output with ranked fragility table

**LoC:** ~12K total, ~7K novel.

**Go/No-Go Gate (end of Week 14):**
- **GO** if fragility scores on pilot DAGs match or improve Phase 0 results.
- **GO** if runtime ≤ 30 minutes for p ≤ 20 on laptop CPU.
- **CONCERN** if KCI calibration produces > 20% disagreement with Fisher-Z on continuous data (indicates kernel choice sensitivity).

**Milestone:** Reproducible fragility tables for all 5 pilot DAGs + 5 additional published DAGs (10 total). Internal benchmark report.

---

### Phase 2: Exact Radius via ILP (Weeks 15–22)

**Objective:** Compute exact robustness radius as minimum-edit DAG repair via ILP, with LP relaxation lower bound.

**Deliverables:**
- ILP formulation: binary variables x_e ∈ {0,1} for each candidate edit e ∈ An(X,Y)², objective min Σ x_e, subject to:
  - Acyclicity: topological ordering constraints (Δ_ij variables)
  - CI constraints: for each CI relation (A ⊥⊥ B | S) supported by data, d-separation must hold in edited DAG
  - Conclusion preservation: back-door criterion must fail in edited DAG (we seek the *smallest* set of edits that *breaks* the conclusion)
- LP relaxation solver for anytime lower bound
- Radius bracket output: [LP_lower, ILP_exact_or_timeout_upper]
- CLI: `causalcert radius --dag G.dot --treatment X --outcome Y --data D.csv --timeout 1800`
- NP-hardness proof (written, for the paper)
- FPT reference implementation for w ≤ 3 (for paper validation, not production use)

**LoC:** ~8K total, ~5K novel.

**Go/No-Go Gate (end of Week 22):**
- **GO** if ILP solves p ≤ 20 instances in < 30 minutes (CBC solver) or < 5 minutes (Gurobi).
- **GO** if LP lower bound is within 1 of ILP optimum on ≥ 80% of test instances.
- **DECISION POINT:** If ILP is too slow for p > 15, evaluate whether CDCL upgrade path is warranted. This is a research decision, not a code decision. Estimated CDCL timeline: +4–6 months.

**Milestone:** Exact radii for all 10 benchmark DAGs. Comparison: ILP exact vs. greedy upper bound vs. LP lower bound.

---

### Phase 3: Paper-Ready Evaluation (Weeks 23–30)

**Objective:** Complete evaluation suite, write and submit paper.

**Deliverables:**
- Synthetic DGP suite: Erdős-Rényi DAGs (p ∈ {8, 12, 16, 20}, expected degree ∈ {1.5, 2.5}), scale-free DAGs, known-radius instances (plant k random misspecifications, verify tool recovers k)
- Semi-synthetic suite: published DAG structures with simulated data (n ∈ {500, 1000, 5000})
- Published DAG evaluation: 15 DAGs from epidemiology, economics, and biology literature
- Baselines: (a) dagitty implied-CI test + manual enumeration, (b) random edge perturbation, (c) sensemakr E-value (for comparison, different robustness notion)
- Runtime profiling: wall-clock time vs. p, vs. |An|, vs. treewidth
- Full paper draft targeting UAI (8 pages + appendix)

**LoC:** ~6K total, ~3K novel (DGP generators, benchmark harness).

**Milestone:** Submitted paper + public GitHub repository + reproducible evaluation script.

## 7. Hardest Technical Challenges

### Challenge 1: KCI Calibration and the "Fragile vs. Low-Power" Distinction

**Problem:** The kernel conditional independence (KCI) test is the nonparametric workhorse for continuous data, but its calibration is sensitive to kernel bandwidth, Nyström approximation rank, and sample size. A rejection might mean "this CI relation is truly violated" (fragile edge) or "the test lacks power at this sample size" (low-power artifact). Conflating the two produces spurious fragility scores.

**Severity:** High. If fragility scores are dominated by power artifacts, the entire tool is unreliable.

**Mitigation strategy:**
1. **Multi-resolution testing.** Run KCI at multiple Nyström ranks (50, 100, 200) and bandwidths (median heuristic × {0.5, 1.0, 2.0}). Report fragility only when ≥ 2/3 configurations agree.
2. **Fisher-Z as anchor.** For Gaussian data, Fisher-Z is exact. Use Fisher-Z/KCI agreement rate as a calibration diagnostic. If agreement < 70%, warn user that nonparametric results are unreliable.
3. **Cauchy combination.** The Cauchy combination test (Liu & Xie 2020) combines p-values from heterogeneous tests without assuming independence. This replaces ad-hoc minimum/maximum aggregation.
4. **Power diagnostics.** Report estimated power for each CI test at the observed sample size (using permutation-based power estimation). Flag edges where power < 0.5 as "inconclusive" rather than "robust."

**Estimated effort:** 2–3 weeks of careful statistical engineering in Phase 1.

---

### Challenge 2: ILP Formulation of Acyclicity Constraints

**Problem:** Encoding "the edited graph G' must be a DAG" as linear constraints is non-trivial. The standard approach uses topological ordering variables: integer variables π_i representing the position of node i in a topological order, with π_i < π_j whenever edge i → j exists. But edge edits change which ordering constraints are active, creating a bilinear interaction between edge indicator variables and ordering variables.

**Severity:** High. An incorrect or loose ILP formulation either misses valid edits (overly conservative radius) or allows cyclic graphs (incorrect radius).

**Mitigation strategy:**
1. **Big-M formulation.** For each potential edge (i,j) in the edited graph: π_j - π_i ≥ 1 - M(1 - e_ij), where e_ij is the edge indicator and M = p. This linearizes the bilinear term at the cost of weak LP relaxation.
2. **Cycle-elimination cuts.** Iteratively solve the LP relaxation, check for cycles in the fractional solution, add violated cycle-elimination constraints (sum of edge indicators around any cycle ≤ cycle_length - 1), re-solve. This is a standard cutting-plane approach that tightens the formulation.
3. **Ancestral set restriction.** Only model edges within An(X,Y), reducing the variable count from O(p²) to O(|An|²). For typical DAGs, this cuts the ILP size by 2–6×.
4. **Fallback:** If ILP formulation proves intractable, fall back to exhaustive enumeration for k ≤ 3 (feasible for |An| ≤ 15: C(225, 3) ≈ 1.9M candidates, pruned to ~50K by fragility ranking).

**Estimated effort:** 3–4 weeks in Phase 2. The cycle-elimination approach is well-studied in combinatorial optimization.

---

### Challenge 3: Evaluating Correctness Without Ground Truth

**Problem:** For real DAGs, the true robustness radius is unknown. We cannot directly verify that the tool's output is correct. For fragility scores, there is no "ground truth" — only the question of whether scores are *useful* and *consistent*.

**Severity:** Medium. Without validation, the tool's claims are unverifiable.

**Mitigation strategy:**
1. **Synthetic instances with known radius.** Generate DAGs, plant exactly k misspecifications, verify the tool recovers radius = k. This validates correctness for the synthetic regime.
2. **Exhaustive enumeration on small instances.** For p ≤ 10, k ≤ 2, exhaustively enumerate all edit sets and compute exact radius by brute force. Compare ILP output. Any discrepancy is a bug.
3. **Consistency checks.** Fragility scores must satisfy: (a) locality (Theorem 1), (b) monotonicity under edge removal from conclusion set, (c) fragility(e) > 0 implies e is in An(X,Y). Automated property-based tests (hypothesis library) check these invariants.
4. **Semi-synthetic validation.** Take real DAG structure, simulate data from known DGP, introduce known misspecifications, verify tool detects them.
5. **Cross-solver validation.** Compare ILP radius against FPT DP (for w ≤ 3 instances) and brute-force (for small instances). Three-way agreement builds confidence.

**Estimated effort:** 3–4 weeks in Phase 3, integrated into evaluation harness.

## 8. Evaluation Plan

All evaluation is fully automated and reproducible via a single script: `python -m causalcert.eval.run_all`.

### 8.1 Synthetic Evaluation (Known Ground Truth)

| Parameter | Values | Purpose |
|-----------|--------|---------|
| Nodes (p) | 8, 12, 16, 20 | Scalability |
| Expected degree | 1.5, 2.5 | Sparse vs. medium density |
| Graph model | Erdős-Rényi, scale-free (Barabási-Albert) | Structure diversity |
| Planted misspecifications (k) | 1, 2, 3 | Known radius |
| Sample size (n) | 500, 1000, 5000 | Power variation |
| Repetitions | 50 per configuration | Statistical reliability |

**Total synthetic instances:** 4 × 2 × 2 × 3 × 3 × 50 = 7,200.

**Metrics:** (a) Radius recovery rate (ILP radius = planted k), (b) Fragility score AUC (do planted edges rank highest?), (c) Runtime (wall-clock seconds), (d) LP gap (ILP optimum - LP relaxation).

### 8.2 Semi-Synthetic Evaluation (Known Structure, Simulated Data)

Take the 5 pilot DAGs + 10 additional published DAGs. For each:
1. Simulate data from a linear Gaussian SEM with standardized coefficients ~ Uniform([−1, −0.3] ∪ [0.3, 1]).
2. Introduce k ∈ {1, 2, 3} random edge misspecifications.
3. Run CausalCert. Check fragility scores rank misspecified edges in top-k.

**Total semi-synthetic instances:** 15 × 3 × 3 × 20 = 2,700 (3 sample sizes, 20 repetitions).

### 8.3 Published DAG Evaluation (No Ground Truth)

15 published DAGs from:
- **Epidemiology:** DAGs from American Journal of Epidemiology (2018–2024), 5 DAGs.
- **Economics:** DAGs from Angrist & Pischke examples, 3 DAGs.
- **Biology:** Sachs (2005), plus 2 additional biological network DAGs.
- **Social science:** DAGs from Morgan & Winship textbook examples, 5 DAGs.

**Metrics:** (a) Number of fragile edges (fragility > 0.5), (b) Radius, (c) Runtime, (d) Domain expert validation (for 3 DAGs where we have collaborator access).

### 8.4 Baseline Comparisons

| Baseline | Description | Purpose |
|----------|-------------|---------|
| dagitty + for-loop | dagitty::adjustmentSets() per edge removal | Show CausalCert adds value beyond trivial enumeration |
| Random perturbation | Random edge edits, check if conclusion breaks | Lower bound on fragility score utility |
| sensemakr E-value | Unmeasured confounding robustness | Different robustness concept; compare on same DAGs |
| Greedy beam search | Fragility-ranked greedy multi-edit (Approach A) | Justify ILP over greedy: how often does greedy overestimate? |

### 8.5 Runtime Budget

| Instance size | Target runtime | Solver |
|---------------|---------------|--------|
| p ≤ 10 | < 2 minutes | ILP (CBC) |
| p ≤ 15 | < 10 minutes | ILP (CBC or Gurobi) |
| p ≤ 20 | < 30 minutes | ILP (Gurobi preferred) |
| p > 20 | Fragility only (radius may timeout) | LP lower bound + greedy upper bound |

All benchmarks on a single laptop CPU (Apple M2 or Intel i7, 16GB RAM). No GPU, no cluster.

## 9. Best-Paper Argument

### Target Venue: UAI (Conference on Uncertainty in Artificial Intelligence)

UAI is the ideal venue: it values the intersection of computational complexity, causal inference, and practical algorithms. Recent best papers (IDA, GES, NOTEARS) share the structure of "new computational problem + hardness + algorithm + evaluation."

### Paper Structure (8 pages + appendix)

1. **Introduction (1 page).** Motivating example: published DAG where a single edge misspecification overturns the treatment effect. "How fragile is your DAG?" as a question every causal analyst should ask.

2. **Problem Formulation (1 page).** Formal definition of robustness radius, fragility score, minimum-edit DAG repair. Clean, self-contained.

3. **Complexity Results (2 pages).** Theorem 2 (NP-hardness), Theorem 3 (FPT in treewidth), Theorem 4 (LP gap bound). The theoretical core.

4. **Algorithms (1.5 pages).** Per-edge fragility via incremental d-separation. ILP formulation. LP relaxation. Fragility-ranked pruning.

5. **Experiments (2 pages).** Synthetic (radius recovery, AUC), semi-synthetic (ranking accuracy), published DAGs (case studies). Runtime scaling.

6. **Discussion (0.5 pages).** Limitations, future work (CDCL, error propagation, beyond back-door).

### Killer Result

*"We stress-tested 15 published causal DAGs and found: (a) in 11/15, a single edge misspecification overturns the stated causal conclusion (radius = 1); (b) per-edge fragility scores identify the load-bearing edges with AUC > 0.9 on semi-synthetic benchmarks; (c) exact radii via ILP are computable in < 30 minutes for all DAGs with p ≤ 20."*

### Why This Wins

1. **New problem.** Minimum-edit DAG repair has not been formally studied. The NP-hardness result is new.
2. **Complete complexity picture.** NP-hard in general, FPT in treewidth, LP gap O(w). This is the kind of clean complexity trifecta reviewers love.
3. **Immediately useful.** The fragility table is something every causal analyst can use tomorrow. Not just theory.
4. **Reproducible.** Public code, automated evaluation, published DAG benchmarks.

## 10. Honest Limitations

### What CausalCert Cannot Do

1. **Cannot certify a DAG is correct.** CausalCert identifies *fragile* assumptions, not *wrong* assumptions. A fragile edge might be correct; a robust edge might be wrong. The tool measures sensitivity to structural perturbation, not ground truth.

2. **Cannot handle unmeasured confounding.** The tool assumes all variables in the DAG are measured. Latent variable models (MAGs, PAGs) are out of scope. sensemakr/E-values address unmeasured confounding; CausalCert addresses structural misspecification. They are complementary, not competing.

3. **Cannot handle non-back-door identification.** Phase 1 supports only back-door/adjustment-set identification. Front-door criterion, instrumental variables, and do-calculus-based identification are future work.

4. **Cannot scale beyond p ≈ 20 for exact radius.** The ILP is exponential in the worst case. For p > 20, only fragility scores (polynomial) and radius bounds (LP lower, greedy upper) are available. This is a fundamental complexity barrier (Theorem 2), not an engineering limitation.

5. **Cannot distinguish fragile-from-misspecification vs. fragile-from-low-power.** If a CI test rejects due to low power rather than true dependence, the fragility score is inflated. Power diagnostics mitigate but do not eliminate this issue.

6. **Cannot handle cyclic causal models.** DAGs only. Cyclic SEMs, feedback loops, and time-varying systems are out of scope.

### Load-Bearing Assumptions

1. **Faithfulness.** CI test results are interpreted under the faithfulness assumption. Violations (measure-zero but possible) can cause missed fragile edges.

2. **Correct variable set.** The DAG must include all causally relevant variables. Missing variables create unmeasured confounding, which CausalCert does not address.

3. **Sufficient sample size.** CI tests require adequate power. For p = 20 with conditioning sets of size 5+, n ≥ 1000 is practically necessary for Fisher-Z; n ≥ 2000 for KCI.

4. **CI test validity.** The tool's outputs are conditional on CI test results being correct. No formal error propagation bound exists (see Section 5, Non-Theorem).

## 11. Scores

| Axis | Score | Justification |
|------|-------|---------------|
| **Value** | **7/10** | Per-edge fragility scores are immediately actionable for applied researchers (Maria persona). The E-value analogy is strong. Docked from 8 because single-edit fragility is achievable with dagitty + for-loop in ~200 lines of R; the genuine value-add is the multi-edit radius and ensemble CI testing. The radius bracket [LP, ILP] is a real contribution no existing tool provides. |
| **Difficulty** | **6/10** | ~17.7K novel LoC spanning incremental d-separation (5/10), ILP formulation with acyclicity constraints (6/10), LP relaxation (5/10), KCI calibration (4/10), and NP-hardness + FPT proofs (4/10 + 7/10). No single component is research-frontier hard (no CDCL), but the combination is solidly above engineering-only. The FPT proof is genuine math. |
| **Potential** | **7/10** | Clean complexity trifecta (NP-hard + FPT + LP gap) is publishable at UAI. Empirical results on published DAGs add impact. Not 8 because without CDCL, the algorithmic contribution is "formulate as ILP and call Gurobi" — real but not a new algorithmic paradigm. The FPT result and LP gap bound elevate above methods-note territory. |
| **Feasibility** | **7/10** | 30-week timeline with strict phase gates. Phase 0 (6 weeks) is low-risk. Phase 1 (8 weeks) is solid engineering. Phase 2 (8 weeks) has moderate risk: ILP formulation of acyclicity constraints is tricky but well-studied in combinatorial optimization. Phase 3 (8 weeks) is evaluation and writing. Main risk: ILP solver performance on p > 15 instances. Docked from 9 (Approach A) because ILP and FPT add real implementation complexity beyond simple fragility scoring. |

**Composite: 6.75/10.** This reflects a deliberate trade-off: more theoretical depth than Approach A (Potential 7 vs. 5), more practical grounding than Approach B (Feasibility 7 vs. 5), and more intellectual honesty than Approach C (no ungrounded composite scores).

## 12. Risk Registry

| # | Risk | Severity | Probability | Mitigation |
|---|------|----------|-------------|------------|
| R1 | **Goldilocks failure:** fragility scores show no meaningful variation on published DAGs (all edges fragility ≈ 0 or ≈ 1) | **Critical** | 25% | Phase 0 pilot on 5 DAGs gates all subsequent work. 6-week bounded bet. If failure: publish negative result, pivot to different robustness notion. |
| R2 | **ILP scalability wall:** CBC/Gurobi cannot solve p > 15 instances within 30-minute budget | **High** | 30% | (a) Fragility-ranked variable ordering to warm-start ILP. (b) LP lower bound as anytime fallback. (c) Greedy upper bound if ILP times out. (d) Report bracket [LP, greedy] instead of exact radius. (e) CDCL upgrade path for future work. |
| R3 | **KCI miscalibration:** kernel CI tests produce systematically biased fragility scores due to bandwidth/power sensitivity | **High** | 35% | Multi-resolution testing (multiple bandwidths, Nyström ranks). Fisher-Z anchor for Gaussian data. Cauchy combination for principled aggregation. Power diagnostics flagging inconclusive tests. |
| R4 | **FPT proof error:** Tree decomposition DP join operation is incorrect, invalidating Theorem 3 | **Medium** | 15% | (a) Verify on exhaustive enumeration for w ≤ 2 instances. (b) Cross-check with ILP on w ≤ 3 instances. (c) Paper contribution survives without FPT if NP-hardness + ILP + LP gap remain. (d) Theorem is not load-bearing for the software — only for the paper. |
| R5 | **Acyclicity constraint encoding:** ILP formulation is incorrect, allowing cyclic solutions or over-constraining | **High** | 20% | (a) Big-M formulation is well-studied (standard in DAG learning ILP literature). (b) Post-hoc acyclicity check on every ILP solution. (c) Cycle-elimination cuts as iterative refinement. (d) Exhaustive-enumeration crosscheck on small instances. |
| R6 | **"Fragile vs. low-power" confusion:** Users misinterpret fragility scores as evidence of misspecification rather than sensitivity to perturbation | **Medium** | 40% | (a) Clear documentation: "fragile ≠ wrong." (b) Power diagnostics in output. (c) Separate columns for d-sep fragility, ID fragility, EST fragility — users see *why* an edge is fragile. (d) "Inconclusive" label for low-power settings. |
| R7 | **Scope creep into CDCL:** ILP underperformance triggers premature CDCL development, blowing the timeline | **Medium** | 20% | Strict phase gate at end of Phase 2. CDCL is explicitly deferred to future work. The paper's contribution does not depend on CDCL. If ILP is slow, report honest brackets and move on. |
| R8 | **Competing work:** Another group publishes structural robustness for DAGs before submission | **Low** | 10% | (a) Move fast — 30-week timeline. (b) NP-hardness + FPT trifecta is a differentiator even if someone publishes fragility scores. (c) Monitor arXiv monthly. (d) If scooped on fragility: pivot paper to focus on complexity results. |
| R9 | **LP gap is too loose:** LP relaxation lower bound is consistently 0 or 1, providing no useful information | **Medium** | 25% | (a) Tighten formulation with cutting planes (cycle-elimination, clique inequalities). (b) If gap remains large, report upper bound only with explicit caveat. (c) FPT DP provides exact solution for w ≤ 3 as an alternative lower/exact bound. |
| R10 | **Mixed variable types:** Real datasets mix continuous, binary, ordinal variables; CI test ensemble struggles with heterogeneous data | **Medium** | 30% | (a) Fisher-Z for continuous, χ² for discrete, KCI as nonparametric fallback. (b) Recommend users discretize or model variable types explicitly. (c) Phase 1 CI test ensemble handles mixed types via appropriate test selection per variable pair. |
