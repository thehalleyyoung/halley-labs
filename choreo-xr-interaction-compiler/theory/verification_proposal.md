# Choreo XR Interaction Compiler — Theory Stage Verification Framework

**Document type**: Verification Chair Quality-Gate Specification
**Project**: Choreo — DSL Compiler & Verifier for Spatial-Temporal Interaction Choreographies
**Stage**: Theory (approach.json + paper.tex)
**Date**: 2025-07-18
**Prior evaluation**: Depth Check Composite 6.25/10 — CONDITIONAL CONTINUE (2-1, Skeptic dissents)
**Prior scores**: Value 5/10, Difficulty 7/10, Best-Paper Potential 5/10, Laptop CPU 8/10

---

## 0. Executive Summary

This document defines the quality gates, scoring rubrics, and CONTINUE/ABANDON decision framework for the Choreo theory stage. The theory stage must produce two deliverables:

1. **`theory/approach.json`** — Structured theory specification: algorithms, complexity analysis, mathematical foundations, implementation mapping.
2. **`theory/paper.tex`** — Publication-quality document (>50 KB) with full formalism, proofs, and experimental plan.

The project advances four theorem-level contributions:

| ID | Claim | Risk Profile | Grade Target |
|----|-------|-------------|--------------|
| **T1** | Decidable spatial type checking | Moonshot, ~45% failure | B+ to A- |
| **T2** | Geometric consistency pruning | Guaranteed, ~20% failure | B to B+ |
| **T3** | Spatial CEGAR with GJK/EPA | Guaranteed, ~25% failure | B |
| **T4** | Compositional spatial separability | Stretch, ~30% failure | B |

The prior depth check's composite 6.25/10 reflects genuine uncertainty. This framework specifies what "good enough" looks like so the next evaluation is not subjective.

---

## 1. Quality Criteria for Theory Deliverables

### 1.1 `approach.json` Requirements

The structured theory file must satisfy **all** of the following. Each item is scored PASS / PARTIAL / FAIL.

#### 1.1.1 Algorithmic Precision

| Criterion | PASS | PARTIAL | FAIL |
|-----------|------|---------|------|
| **Pseudocode coverage** | Every algorithm (CEGAR loop, GJK oracle, R-tree pruning, type checking, compositional decomposition) has line-numbered pseudocode with typed inputs/outputs | ≥3 of 5 core algorithms have pseudocode; others have structured descriptions | Algorithms described only in prose |
| **Termination arguments** | Every loop/recursion has an explicit variant/measure that decreases | Most loops have arguments; ≤1 gap acknowledged | No termination reasoning |
| **Correctness invariants** | Loop invariants stated for every non-trivial loop | Invariants stated for ≥ main CEGAR loop | No invariants |

#### 1.1.2 Complexity Bounds

| Criterion | PASS | PARTIAL | FAIL |
|-----------|------|---------|------|
| **Derivation present** | Every complexity bound (time and space) has a step-by-step derivation from structural properties | Bounds stated with sketch-level justification | Bounds stated without justification |
| **Tightness discussion** | Lower bounds or tightness arguments for ≥ T2 and T3 | At least one tightness argument | No discussion of tightness |
| **Parameter identification** | All parameters in O(·) notation are defined (e.g., n = number of zones, k = number of predicates, w = treewidth) | Most parameters defined | Ambiguous parameters |

#### 1.1.3 Assumption Audit

| Criterion | PASS | PARTIAL | FAIL |
|-----------|------|---------|------|
| **Explicit listing** | Every theorem has a numbered list of assumptions | Assumptions present but not systematically enumerated | Assumptions implicit |
| **Justification** | Each assumption has a justification (empirical, definitional, or conservative) | Most assumptions justified | Assumptions stated without rationale |
| **Relaxation analysis** | For each assumption, states what happens if it fails | Relaxation discussed for ≥ critical assumptions | No relaxation analysis |

#### 1.1.4 Implementation Mapping

| Criterion | PASS | PARTIAL | FAIL |
|-----------|------|---------|------|
| **Theorem → Module map** | Every theorem maps to specific code modules (e.g., "T2 → `src/pruning/geometric_consistency.rs`") | Mapping at subsystem level (e.g., "T2 → Reachability Checker") | No mapping |
| **Interface contracts** | Module interfaces specified with pre/post conditions | Interfaces described informally | No interface specification |
| **Dependency DAG** | Explicit dependency graph between components with critical-path annotation | Dependencies listed but not graphed | Dependencies implicit |

#### 1.1.5 Component Dependencies

| Criterion | PASS | PARTIAL | FAIL |
|-----------|------|---------|------|
| **Theorem dependency graph** | Explicitly states which theorems depend on which (e.g., "T3 uses T2 as subroutine; T1 is independent") | Dependencies mentioned but not systematized | Not discussed |
| **Graceful degradation** | For each optional theorem (T1, T4), describes system behavior if theorem is unavailable | Degradation discussed for T1 only | No degradation analysis |
| **Build order** | Specifies implementation order respecting dependencies | Partial ordering | No ordering |

### 1.2 `paper.tex` Requirements

#### 1.2.1 Structural Requirements

| Criterion | PASS | PARTIAL | FAIL |
|-----------|------|---------|------|
| **Size** | >50 KB raw LaTeX source | 35–50 KB | <35 KB |
| **Compilation** | `pdflatex` (or `lualatex`) compiles without errors; ≤5 warnings | Compiles with ≤3 non-critical errors | Does not compile |
| **Standard structure** | Abstract, Introduction, Related Work, Formal Model, Technical Contributions (T1–T4), Algorithms, Evaluation Plan, Conclusion | Missing ≤1 major section | Missing ≥2 major sections |
| **Bibliography** | ≥30 references; all cited in text; BibTeX compiles cleanly | 20–30 references | <20 references |

#### 1.2.2 Definition Quality

| Criterion | PASS | PARTIAL | FAIL |
|-----------|------|---------|------|
| **Implementable precision** | Every definition (spatial predicate, event automaton, choreography type, etc.) is stated in set-builder or BNF notation with all metavariables bound | Most definitions are formal; ≤2 rely on informal description | Definitions are prose-level |
| **Consistent notation** | A notation table is provided; notation is used consistently throughout | Minor inconsistencies (≤3 instances) | Significant inconsistency or notation table absent |
| **Running example** | A single non-trivial XR interaction example threads through all definitions | Example present but not threaded through all sections | No running example |

#### 1.2.3 Proof Quality

| Criterion | PASS | PARTIAL | FAIL |
|-----------|------|---------|------|
| **T2 proof** | Complete, self-contained proof with all steps justified | Proof sketch with ≤2 identified gaps | Absent or hand-wavy |
| **T3 proof** | Complete, self-contained proof with all steps justified | Proof sketch with ≤2 identified gaps | Absent or hand-wavy |
| **T1 proof** | At minimum: binary-case (2 predicates) complete proof + general-case proof sketch | Binary-case proof sketch only | No proof content |
| **T4 proof** | At minimum: disjoint-case complete proof + general-case conjecture with evidence | Disjoint-case proof sketch only | No proof content |
| **Citation of prior techniques** | Every proof technique cites its origin (e.g., "by CEGAR [Clarke et al. 2003]") | Most techniques attributed | Proof techniques appear ex nihilo |

#### 1.2.4 Algorithmic Content

| Criterion | PASS | PARTIAL | FAIL |
|-----------|------|---------|------|
| **Pseudocode** | All core algorithms in `algorithm` or `algorithmic` LaTeX environments with line numbers | ≥3 core algorithms in formal pseudocode | <3 algorithms formalized |
| **Correctness arguments** | Each algorithm has a theorem or lemma establishing its correctness | ≥ CEGAR loop and pruning have correctness results | No correctness arguments |
| **Complexity analysis** | Each algorithm has a complexity theorem with proof | Complexity stated for ≥3 algorithms | Complexity not analyzed |

#### 1.2.5 Experimental Plan

| Criterion | PASS | PARTIAL | FAIL |
|-----------|------|---------|------|
| **Falsifiable thresholds** | Concrete numerical thresholds (e.g., "CEGAR finds ≥5 bugs in MRTK corpus", "geometric pruning reduces state space by ≥10× on scenes with ≥8 zones") | Thresholds stated but soft (e.g., "significant improvement") | No thresholds |
| **Baselines** | ≥3 comparison baselines defined (e.g., naive BFS, UPPAAL, random simulation) with justification | 1–2 baselines | No baselines |
| **Benchmark specification** | Benchmark suite defined: source (MRTK, Meta SDK), selection criteria, parametric scaling method | Benchmarks mentioned but not specified | No benchmark plan |
| **Statistical methodology** | States how significance will be assessed (e.g., median over N runs, confidence intervals) | Mentions repetition but no methodology | No statistical plan |

---

## 2. Theorem Verification Checklist

### 2.1 T1: Decidable Spatial Type Checking

**Claim**: Well-typed Choreo programs compile to spatially realizable automata. Type checking is decidable for convex polytopes (via LP) and NP-complete for bounded-depth CSG.

#### Statement Clarity

- [ ] The type system is defined with formal inference rules (not just English descriptions)
- [ ] "Spatially realizable" is precisely defined: a configuration σ is realizable iff ∃ a geometric embedding in ℝ³ satisfying all active spatial predicates
- [ ] The decidability result states the exact fragment (convex polytopes) and the exact complexity (polynomial via LP)
- [ ] The NP-completeness result for CSG states the reduction source problem
- [ ] A third party with access to the paper alone could implement a type checker

#### Proof Completeness

- [ ] Binary-case (2 predicates): complete proof establishing that LP feasibility decides realizability
- [ ] General convex case: proof or detailed sketch reducing to LP feasibility
- [ ] CSG case: NP-hardness reduction from a known NP-complete problem (e.g., 3-SAT, subset intersection)
- [ ] CSG case: membership in NP via witness construction
- [ ] Soundness: well-typed ⇒ all reachable states are spatially realizable

#### Assumption Audit

- [ ] Convexity of geometric primitives is explicitly assumed and its scope defined
- [ ] Finite precision arithmetic: does the proof hold under floating-point? If not, states the gap
- [ ] Bounded-depth CSG: the depth bound is stated and justified for XR scenes
- [ ] Static vs. dynamic geometry: states whether the type system handles time-varying shapes
- [ ] Independence from T2–T4: T1 is self-contained

#### Counterexample Search

- [ ] Has an example where type checking succeeds but a naive spatial check would fail (demonstrates value)
- [ ] Has an example where type checking correctly rejects an unrealizable configuration
- [ ] Has attempted to construct a configuration that is realizable but rejected by the type system (false positive search)

#### Prior Art Comparison

- [ ] Compares to spatial logic (Aiello et al.), region connection calculus (RCC-8), constraint-based spatial reasoning
- [ ] States clearly what is new: the connection to XR interaction semantics, not the LP feasibility itself
- [ ] Acknowledges that LP-based geometric feasibility is classical; novelty is in the type-system integration

#### Load-Bearing Test

- [ ] If T1 is false: the compiler still works but cannot statically reject unrealizable configurations → runtime errors possible but system degrades gracefully
- [ ] T2, T3, T4 do NOT depend on T1
- [ ] Quantify: what fraction of real XR bugs would T1 alone catch? (Honest answer: likely small — most bugs are temporal, not spatial-type errors)

---

### 2.2 T2: Geometric Consistency Pruning

**Claim**: Monotonicity, triangle inequality, and containment consistency reduce the feasible predicate set C ⊆ 2^P from 2^|P| to a polynomially-bounded (in practice) subset, enabling exponential state-space reduction.

#### Statement Clarity

- [ ] The feasible predicate set C is formally defined: C = {v ∈ 2^P | ∃ geometric configuration realizing v}
- [ ] Each pruning rule (monotonicity, triangle inequality, containment) is stated as a formal constraint
- [ ] The reduction factor is quantified: from 2^(n²·m) to O(m^(n²)) for m proximity thresholds over n entities
- [ ] "Exponential reduction" is qualified: exponential in what parameter, under what conditions

#### Proof Completeness

- [ ] Soundness proof: every pruned valuation is genuinely unrealizable (no false pruning)
- [ ] Completeness discussion: are there unrealizable valuations that survive pruning? (Expected: yes — pruning is sound but incomplete)
- [ ] The O(m^(n²)) bound has a complete derivation
- [ ] The derivation identifies the tightest known bound, not just an upper bound

#### Assumption Audit

- [ ] Metric space properties: triangle inequality assumes a proper metric (not a pseudo-metric)
- [ ] Monotonicity: requires that spatial predicates are monotone in distance (stated explicitly)
- [ ] Containment: assumes a proper containment partial order on regions
- [ ] Independence from T1: T2 works even if T1 fails (the pruning is a standalone preprocessing step)
- [ ] Independence from T3: T2 feeds into T3 but T3 can work (less efficiently) without T2

#### Counterexample Search

- [ ] Has an example where pruning eliminates ≥50% of states on a realistic scene
- [ ] Has an example where pruning eliminates 0% of states (worst case; identifies when pruning is useless)
- [ ] Has verified that no reachable state in the MRTK menu example is incorrectly pruned

#### Prior Art Comparison

- [ ] Compares to BDD-based symmetry reduction, partial-order reduction, predicate abstraction
- [ ] Acknowledges that geometric consistency as a pruning criterion is novel for model checking but classical in computational geometry
- [ ] Cites relevant computational geometry work (Edelsbrunner, de Berg et al.)

#### Load-Bearing Test

- [ ] If T2 is false (pruning is unsound): CEGAR produces spurious counterexamples → false bug reports → **critical failure**
- [ ] If T2 is weak (pruning provides <2× reduction): verification is slow but correct → system degrades in performance, not correctness
- [ ] T3 depends on T2 for efficiency but not correctness

---

### 2.3 T3: Spatial CEGAR with GJK/EPA

**Claim**: A CEGAR loop using GJK/EPA as the geometric oracle can verify spatial-temporal event automata, with termination guaranteed for finite-state abstractions and sound geometric refinement.

#### Statement Clarity

- [ ] The CEGAR loop is precisely defined: abstract → check → counterexample → refine → repeat
- [ ] The abstraction function α and concretization function γ are formally defined
- [ ] GJK/EPA is specified as the geometric feasibility oracle with its exact interface
- [ ] Termination condition is stated: finite abstract domain ⇒ termination in ≤|abstract states| iterations
- [ ] The soundness claim is precise: if CEGAR reports "safe", the property holds in the concrete system; if CEGAR reports a counterexample, it is geometrically realizable

#### Proof Completeness

- [ ] CEGAR soundness: complete proof that abstract counterexample + geometric feasibility ⇒ concrete counterexample
- [ ] CEGAR completeness: proof that refinement eventually eliminates all spurious counterexamples (for finite domains)
- [ ] Termination: proof that the abstract domain is finite and refinement strictly increases precision
- [ ] GJK/EPA correctness: either proves correctness or cites a standard reference and states the interface contract
- [ ] Progress lemma: each refinement step eliminates at least one spurious counterexample

#### Assumption Audit

- [ ] Convexity for GJK: GJK requires convex shapes; states how non-convex shapes are handled (convex decomposition)
- [ ] Finite abstraction: states that the abstract domain is finite (justified by bounded predicates and bounded time horizon)
- [ ] Numerical stability: GJK/EPA uses floating-point; states the tolerance and its impact on soundness
- [ ] The Event Calculus formalization is assumed correct (dependency on the formal model, not on T1/T2/T4)
- [ ] The R-tree spatial index is assumed correct (implementation dependency, not a theorem dependency)

#### Counterexample Search

- [ ] Has a concrete example where CEGAR finds a genuine spatial-temporal bug
- [ ] Has a concrete example where CEGAR correctly proves safety
- [ ] Has attempted to construct a scenario where GJK/EPA gives a wrong answer due to numerical issues (and documented the result)
- [ ] Has a scenario where CEGAR requires ≥3 refinement iterations (demonstrates non-trivial behavior)

#### Prior Art Comparison

- [ ] Cites CEGAR origin (Clarke et al., 2000/2003) and subsequent developments (SLAM, BLAST, CPAchecker)
- [ ] States clearly what is new: spatial predicates as the abstraction domain, GJK/EPA as the feasibility oracle
- [ ] Compares to existing hybrid-system CEGAR (Alur, Dang, Ivančić) and states the differences
- [ ] Acknowledges that CEGAR is a well-known technique; novelty is in the instantiation, not the framework

#### Load-Bearing Test

- [ ] T3 is the **crown jewel**: if T3 is fatally flawed, the entire project loses its primary verification capability → **ABANDON trigger**
- [ ] If T3 is merely weak (slow, limited scalability): the project still has value as a DSL + runtime, but verification story is diminished → **scope reduction**
- [ ] T3 depends on the formal model being correct; does not depend on T1 or T4

---

### 2.4 T4: Compositional Spatial Separability

**Claim**: Spatially separated regions can be verified independently, enabling compositional verification that scales beyond monolithic state exploration.

#### Statement Clarity

- [ ] "Spatially separated" is formally defined (e.g., regions R₁, R₂ are separated iff no spatial predicate spans both)
- [ ] The compositionality theorem states: if R₁ and R₂ are separated, then Verify(R₁ ∪ R₂) = Verify(R₁) ∧ Verify(R₂)
- [ ] The interface between separated regions is formally defined (shared events, temporal constraints)
- [ ] Complexity gain is quantified: from O(|S₁| · |S₂|) to O(|S₁| + |S₂|) for separated regions

#### Proof Completeness

- [ ] Disjoint case: complete proof that fully disjoint regions (no shared predicates or events) can be verified independently
- [ ] Interface case: at least a proof sketch for regions sharing temporal events but not spatial predicates
- [ ] Soundness: compositional verification is at least as conservative as monolithic verification (no missed bugs)
- [ ] Completeness discussion: can compositional verification miss bugs that monolithic catches? (Expected: yes, for interleaving bugs at interfaces)

#### Assumption Audit

- [ ] Separation criterion: states exactly when two regions are "separated enough" for the theorem to apply
- [ ] Event independence: assumes certain events are local to regions; states which events can be shared
- [ ] Monotonicity of composition: adding a region never makes a previously safe region unsafe (if this is claimed)
- [ ] T4 builds on T2 and T3 but is not required by them

#### Counterexample Search

- [ ] Has an example where compositional verification gives the same result as monolithic (correctness demo)
- [ ] Has an example where compositional verification is unsound because regions are not truly separated (boundary case)
- [ ] Has quantified the separation condition on ≥1 realistic MRTK scene

#### Prior Art Comparison

- [ ] Compares to assume-guarantee reasoning (Jones, 1983; Pnueli, 1985)
- [ ] Compares to compositional model checking (Grumberg, Long; Clarke et al.)
- [ ] States what is new: the spatial separation criterion, not the compositional principle itself
- [ ] Acknowledges that for most practical XR scenes, regions may NOT be cleanly separated

#### Load-Bearing Test

- [ ] If T4 is false: verification is monolithic → scalability bottleneck for scenes with >15 zones, but correctness is unaffected
- [ ] T4 is a scalability theorem, not a correctness theorem: failure reduces performance, not soundness
- [ ] No other theorem depends on T4

---

## 3. CONTINUE/ABANDON Decision Framework

### 3.1 STRONG CONTINUE (Theory Quality ≥ 7/10)

All of the following must hold:

**Proofs:**
- T2 (geometric pruning) proof is complete and self-contained. A reviewer can check every step.
- T3 (spatial CEGAR) proof is complete: soundness, progress, and termination are all established.
- T1 (spatial types) has at minimum a complete proof for the binary-predicate case (2 spatial predicates, convex geometry) and a plausible proof sketch for the general convex case.
- T4 (compositionality) has at minimum a complete proof for the disjoint case (fully separated regions) and a conjecture with evidence for the interface case.

**Algorithms:**
- All five core algorithms (CEGAR loop, geometric pruning, type checking, compositional decomposition, R-tree-backed simulation) have formal pseudocode with typed inputs/outputs.
- Each algorithm has an explicit complexity bound with derivation.
- Each algorithm has a correctness argument (theorem, lemma, or detailed invariant-based sketch).

**Experimental plan:**
- ≥5 falsifiable numerical thresholds stated (e.g., "≥5 bugs found", "≥10× state reduction", "≤60s verification for ≤20 zones").
- ≥3 baselines defined with justification.
- Benchmark suite specified with source and selection criteria.

**Integrity:**
- No fatal logical contradictions between theorems.
- All assumptions explicitly listed and justified.
- Limitations section is honest and substantive (not pro-forma).
- Prior art comparison is thorough and fair.

**Deliverable quality:**
- `approach.json` has ≥80% PASS scores on §1.1 criteria.
- `paper.tex` is >50 KB, compiles cleanly, has ≥30 references.
- A competent PhD student in formal methods could reproduce the core proofs from the paper alone.

### 3.2 CONDITIONAL CONTINUE (Theory Quality 5–7/10)

At least the following must hold:

**Proofs:**
- T2 proof is sketched with ≤2 identified gaps; the gaps have plausible closure strategies.
- T3 proof is sketched: soundness is argued but progress/termination may have gaps. The CEGAR loop structure is correct.
- T1 is explicitly acknowledged as open with a concrete attack plan and timeline (≤3 months to resolution).
- T4 has at least the disjoint-case argument.

**Algorithms:**
- ≥3 of 5 core algorithms have formal pseudocode.
- Complexity bounds are stated for all algorithms; derivations present for ≥3.
- Correctness arguments present for CEGAR loop and pruning at minimum.

**Experimental plan:**
- Plan exists with ≥3 thresholds, but some may be soft ("significant improvement" rather than "≥10×").
- ≥2 baselines defined.

**Integrity:**
- No known fatal contradictions (but some untested interactions).
- Most assumptions listed; ≤3 may be implicit.
- Limitations discussed but may not be comprehensive.

**Deliverable quality:**
- `approach.json` has ≥60% PASS scores and ≤10% FAIL scores on §1.1 criteria.
- `paper.tex` is >35 KB and compiles (possibly with warnings).
- A reader with domain expertise could follow the arguments with effort.

**Amendments required (must be listed explicitly):**
- Each gap in T2/T3 proofs must have a named person/timeline for closure.
- Missing pseudocode must be delivered before implementation begins.
- Soft thresholds must be hardened before evaluation begins.

### 3.3 ABANDON (Theory Quality < 5/10)

Any ONE of the following triggers ABANDON:

**Fatal flaw triggers (any single one suffices):**

1. **T3 has a fatal logical flaw.** The CEGAR loop does not converge, the soundness argument has an unfixable gap, or the GJK/EPA oracle interface is fundamentally incompatible with the abstraction. *Rationale: T3 is the crown jewel. Without sound spatial CEGAR, the project's primary contribution—verification of XR interactions—collapses.*

2. **T2 pruning is unsound.** Geometric consistency pruning discards reachable states. *Rationale: unsound pruning means the verifier produces false "safe" results. This is worse than no verifier.*

3. **T2 pruning provides negligible reduction.** On realistic scenes (≥8 zones, ≥3 proximity thresholds), pruning reduces state space by <2×. *Rationale: if pruning doesn't help, the verification is brute-force BFS on a state space that is well-known to be intractable. The "geometric insight" narrative collapses.*

4. **No plausible path to bug-finding evaluation.** The theory cannot connect to an evaluation where bugs are found in real XR code within the project timeline. *Rationale: the project's primary empirical claim is bug-finding in MRTK/Meta SDK. If the theory doesn't support this, the paper has no evaluation.*

5. **Hidden contradictions in the formal model.** The spatial-temporal event automaton model, the Event Calculus formalization, or the R-tree semantics contain internal contradictions discovered during proof attempts. *Rationale: the formal model is the foundation; contradictions invalidate everything built on it.*

6. **The problem is fundamentally misframed.** Evidence emerges that XR interaction bugs are not spatial-temporal in nature (e.g., they are primarily rendering bugs, networking bugs, or input-parsing bugs), making the entire formalization irrelevant. *Rationale: if the problem doesn't exist, the solution has no value regardless of its elegance.*

**Soft abandon triggers (≥3 required for ABANDON):**

7. T1 has no plausible attack plan and no partial results after good-faith effort.
8. T4 disjoint case fails (regions cannot be verified independently even when fully separated).
9. Fewer than 2 of 5 core algorithms have formal pseudocode.
10. The paper has <20 references and no serious related-work section.
11. Assumptions are so strong that the theorems are vacuously true for all practical XR scenes.
12. The writing quality is too poor for a PhD student to reproduce the work.

---

## 4. Cross-Cutting Concerns

### 4.1 Proof Dependencies

**Dependency graph:**

```
T1 (spatial types)          — INDEPENDENT, moonshot
T2 (geometric pruning)      — INDEPENDENT, feeds efficiency into T3
T3 (spatial CEGAR)           — USES T2 for efficiency (not correctness)
T4 (compositional)           — USES T2 and T3 as black boxes
```

**Critical question: If T1 fails, does anything else break?**

Answer: **No.** T1 is architecturally independent.
- T2 does not reference T1.
- T3's CEGAR loop uses geometric feasibility (GJK/EPA), not the type system.
- T4's separation criterion is topological, not type-theoretic.
- The compiler still works without a spatial type checker; it simply cannot statically reject unrealizable configurations before compilation.

**Verification**: The approach.json must explicitly include a "degradation matrix" showing system behavior when each theorem is absent:

| Missing | Impact | Severity |
|---------|--------|----------|
| T1 | No static spatial-type errors; runtime detection only | Low |
| T2 | Verification is correct but ≤100× slower on large scenes | Medium |
| T3 | No automated verification; manual testing only | **Critical** |
| T4 | Verification is monolithic; does not scale past ~15 zones | Medium |

### 4.2 Honest Self-Assessment

The theory must include an explicit "Limitations and Honest Assessment" section addressing:

- [ ] **Scalability ceiling**: For what scene sizes does verification become intractable even with all optimizations?
- [ ] **Assumption fragility**: Which assumptions are most likely to be violated in practice?
- [ ] **Novelty modesty**: Which parts of the theory are genuinely novel vs. standard instantiations of known techniques?
- [ ] **Evaluation realism**: Is the bug-finding target (≥5 bugs in MRTK) achievable, or is it aspirational?
- [ ] **Market honesty**: The realistic user base is 50–200 engineers at 4 companies. Does the theory serve this audience or a hypothetical academic one?

**Red flag**: If the theory claims all contributions are "novel" without qualification, this is a honesty failure.

### 4.3 Novelty Claims Audit

For each claimed novelty, the paper must provide:

| Contribution | Claimed Novelty | Prior Art That Must Be Cited | What Is Actually New |
|-------------|----------------|------------------------------|---------------------|
| T1: Spatial type checking | Type system for XR interactions | Spatial logics (Aiello), RCC-8, dependent types for geometry (Coquand) | Integration with XR interaction semantics; LP-based decidability for this specific type system |
| T2: Geometric pruning | Pruning via geometric constraints | Symmetry reduction, partial-order reduction, constraint propagation in CSPs | Application of geometric constraints to model-checking state spaces |
| T3: Spatial CEGAR | CEGAR with spatial oracle | CEGAR (Clarke), hybrid CEGAR (Alur/Dang), GJK/EPA (Gilbert, Bergen) | Instantiation of CEGAR with GJK/EPA as refinement oracle for spatial predicates |
| T4: Compositionality | Spatial separability | Assume-guarantee (Jones/Pnueli), compositional model checking (Grumberg) | Spatial separation as the compositionality criterion |

**Acceptable novelty claim**: "We instantiate CEGAR in a new domain (spatial-temporal XR interactions) with a geometrically-grounded refinement oracle, which to our knowledge has not been done before."

**Unacceptable novelty claim**: "We invent a fundamentally new verification paradigm."

### 4.4 Practical Relevance

The theory must pass the **ornamental theory test**: every formal definition and theorem must trace to either (a) a compiler module or (b) an evaluation metric. If a theorem exists only for intellectual interest with no path to code or experiments, it is ornamental and should be flagged.

Checklist:
- [ ] T1 → spatial type checker module → catches type errors in Choreo programs
- [ ] T2 → pruning module → measurable state-space reduction (experimental metric)
- [ ] T3 → CEGAR verifier module → bug-finding on MRTK corpus (primary evaluation)
- [ ] T4 → compositional verifier → scalability beyond 15 zones (scalability evaluation)
- [ ] Every definition in the formal model → parser or compiler phase
- [ ] Every complexity bound → performance prediction testable in evaluation

### 4.5 Reproducibility

A competent PhD student in formal methods (familiar with model checking and computational geometry, but not this specific project) should be able to:

- [ ] Re-derive T2 proof from the paper in ≤4 hours
- [ ] Re-derive T3 soundness proof in ≤8 hours
- [ ] Implement the CEGAR loop from the pseudocode in ≤2 weeks (excluding GJK/EPA, using a library)
- [ ] Implement the pruning algorithm from the pseudocode in ≤1 week
- [ ] Understand the full formal model from the paper in ≤1 day

If any of these estimates seem unreasonable, the paper lacks sufficient detail.

---

## 5. Scoring Rubric

Each axis is scored 1–10 with precise anchors. The composite theory quality is the **unweighted arithmetic mean** of all six axes, consistent with the project's prior scoring methodology.

### 5.1 Rigor (1–10)

| Score | Description |
|-------|-------------|
| 1–2 | No proofs. Claims without justification. |
| 3–4 | Proof sketches with major gaps. Key steps are "clearly" true without argument. |
| 5–6 | T2 and T3 have proof sketches with ≤2 identified gaps each. Gaps have plausible closure strategies. Key lemmas are stated but some are unproven. |
| 7–8 | T2 and T3 have complete proofs. T1 has a binary-case proof. T4 has a disjoint-case proof. All lemmas are proven or have detailed proof sketches. Minor gaps (≤1 per theorem) are identified and bounded. |
| 9–10 | All four theorems have complete, formally-checkable proofs. No gaps. Could be mechanized in a proof assistant with moderate effort. |

### 5.2 Novelty (1–10)

| Score | Description |
|-------|-------------|
| 1–2 | All results are restatements of known results. No new insight. |
| 3–4 | Results are straightforward instantiations of known techniques in a new domain. The domain choice is the only novelty. |
| 5–6 | The instantiation requires non-trivial adaptation: GJK/EPA as CEGAR oracle introduces new technical challenges (geometric refinement semantics, numerical stability). The combination is novel even if components are known. |
| 7–8 | At least one result (likely T1 or T4) provides a new insight that could influence other domains. The geometric pruning analysis reveals structural properties not previously known. |
| 9–10 | Multiple results are publishable independently. The framework introduces new theoretical concepts adopted by the community. |

### 5.3 Load-Bearing (1–10)

| Score | Description |
|-------|-------------|
| 1–2 | Theorems are ornamental. Removing any theorem has no impact on the system. |
| 3–4 | Only 1 of 4 theorems is genuinely load-bearing. Others are included for paper padding. |
| 5–6 | T3 is clearly load-bearing. T2 provides measurable performance benefit. T1 and T4 are defensible but could be omitted without critical system impact. |
| 7–8 | T2 and T3 are both load-bearing (pruning is essential for tractability; CEGAR is the verification engine). T1 or T4 provides a meaningful additional capability. |
| 9–10 | All four theorems are essential. Removing any one significantly degrades the system. Each maps to a distinct, irreplaceable capability. |

### 5.4 Completeness (1–10)

| Score | Description |
|-------|-------------|
| 1–2 | Major components of the theory are missing. The formal model is incomplete. |
| 3–4 | Formal model exists but algorithms are informal. Or: algorithms exist but the formal model is sketchy. |
| 5–6 | Formal model and algorithms are present. Proofs have identified gaps. Complexity analysis is incomplete for ≤2 algorithms. Experimental plan exists but is underspecified. |
| 7–8 | Formal model is complete. All algorithms have pseudocode. All proofs are present (some with minor gaps). Complexity analysis is complete. Experimental plan has falsifiable thresholds. |
| 9–10 | Theory is self-contained and comprehensive. No gaps. All proofs are complete. Implementation mapping is precise. Evaluation plan is fully specified with statistical methodology. |

### 5.5 Honesty (1–10)

| Score | Description |
|-------|-------------|
| 1–2 | Claims are inflated. Limitations are not discussed. Novelty is overstated. |
| 3–4 | Some limitations mentioned but buried. Novelty claims are imprecise ("our novel framework"). |
| 5–6 | Limitations section exists. Novelty claims are qualified. But some risks are understated (e.g., T1 failure risk described as "low" when it is ~45%). |
| 7–8 | Honest limitations section. Each theorem has its assumptions and failure modes stated. Novelty claims precisely delineate what is new vs. known. Risk assessment is calibrated to actual probabilities. |
| 9–10 | Exemplary intellectual honesty. Negative results or failed proof attempts are reported. The paper explicitly states where the theory is weakest and where the authors are least confident. Market size and practical impact are honestly assessed. |

### 5.6 Implementability (1–10)

| Score | Description |
|-------|-------------|
| 1–2 | Theory cannot be implemented. Definitions are too abstract. No pseudocode. |
| 3–4 | Theory is implementable in principle but would require significant intellectual effort to bridge from definitions to code. |
| 5–6 | Most algorithms have pseudocode. Implementation mapping exists at the subsystem level. A competent engineer could implement with ~2 weeks of ramp-up per module. |
| 7–8 | All algorithms have typed pseudocode. Module interfaces are specified. A competent engineer could implement from the paper with ≤1 week ramp-up per module. Libraries (GJK/EPA, R-tree, SAT solver) are identified with version requirements. |
| 9–10 | Pseudocode is directly translatable to code. Data structures are specified. Edge cases are enumerated. A reference implementation could be produced mechanically from the paper. |

### 5.7 Composite Score

```
Theory Quality = (Rigor + Novelty + Load-Bearing + Completeness + Honesty + Implementability) / 6
```

**Interpretation:**
- ≥ 7.0 → STRONG CONTINUE
- 5.0–6.9 → CONDITIONAL CONTINUE (with mandatory amendments)
- < 5.0 → ABANDON

**Interaction with prior depth-check scores:**

The final project composite integrates theory quality with the prior evaluation dimensions:

```
Project Composite = 0.3 × Theory Quality + 0.2 × Value + 0.2 × Difficulty + 0.15 × Best-Paper + 0.15 × Feasibility
```

Using prior scores (V=5, D=7, BP=5, F=8):

| Theory Quality | Project Composite | Verdict |
|---|---|---|
| 8/10 | 0.3(8) + 0.2(5) + 0.2(7) + 0.15(5) + 0.15(8) = 6.35 | STRONG CONTINUE |
| 6/10 | 0.3(6) + 0.2(5) + 0.2(7) + 0.15(5) + 0.15(8) = 5.75 | CONDITIONAL CONTINUE |
| 4/10 | 0.3(4) + 0.2(5) + 0.2(7) + 0.15(5) + 0.15(8) = 5.15 | ABANDON (theory too weak) |
| 3/10 | 0.3(3) + 0.2(5) + 0.2(7) + 0.15(5) + 0.15(8) = 4.85 | ABANDON |

---

## 6. Known Risks and Mitigations

### Risk R1: T3 CEGAR Loop Does Not Converge in Practice

| Attribute | Detail |
|-----------|--------|
| **Risk** | The CEGAR loop oscillates between refinements without converging to a verdict, even though termination is theoretically guaranteed for finite domains. In practice, the abstract domain may be too large for convergence within reasonable time. |
| **Probability** | 25% (consistent with T3's stated ~25% failure risk) |
| **Impact** | Critical — T3 is the crown jewel |
| **Mitigation** | (1) Bound the refinement depth (e.g., max 50 iterations). (2) Use widening operators from abstract interpretation to accelerate convergence. (3) Fall back to bounded model checking (BMC) for fixed time horizons. |
| **Fallback if mitigation fails** | Reduce scope to bounded verification only (check properties up to depth k). This is still publishable as "bounded spatial CEGAR" but loses the completeness claim. Reduces grade from B to B-. |

### Risk R2: Geometric Pruning (T2) Provides Negligible Benefit on Real Scenes

| Attribute | Detail |
|-----------|--------|
| **Risk** | Real XR scenes have low spatial predicate density (few predicates per entity pair), so the theoretical exponential reduction factor does not materialize. Pruning removes <2× states in practice. |
| **Probability** | 20% |
| **Impact** | High — undermines the "geometric insight" narrative. T3 works but is unacceptably slow. |
| **Mitigation** | (1) Quantify pruning effectiveness on ≥10 MRTK scenes during theory stage (a counting argument, not implementation). (2) Identify the "breakeven" scene complexity where pruning pays off. (3) If pruning is weak, reframe as "sound overapproximation" rather than "efficient pruning." |
| **Fallback if mitigation fails** | Drop geometric pruning as a headline contribution. Merge into T3 as an optimization section. Reduce T2 from a standalone theorem to a lemma within T3. Reduces composite difficulty and novelty scores. |

### Risk R3: T1 Type System Is Vacuously Useful

| Attribute | Detail |
|-----------|--------|
| **Risk** | The spatial type system is decidable but rejects so few programs that it provides no practical value. Alternatively, it accepts all programs that a simple syntactic check would also accept. |
| **Probability** | 35% (T1 has ~45% failure risk; of the 55% where it "works," ~35% chance it's vacuous) |
| **Impact** | Medium — T1 is a moonshot and the system degrades gracefully without it |
| **Mitigation** | (1) During theory, construct ≥3 examples of programs that are syntactically valid but spatially unrealizable, where T1 catches the error. (2) Estimate the false-positive and false-negative rates on realistic programs. |
| **Fallback if mitigation fails** | Demote T1 from a theorem to a "preliminary result" or "observation." Move to a future-work section. Reduces grade target from B+/A- to B+. Frees effort for strengthening T2/T3. |

### Risk R4: Formal Model Has Hidden Inconsistencies

| Attribute | Detail |
|-----------|--------|
| **Risk** | The spatial-temporal event automaton model, which is the foundation for all four theorems, contains an internal contradiction (e.g., the interaction between spatial predicates and temporal constraints admits paradoxical states). |
| **Probability** | 10% |
| **Impact** | Fatal — invalidates all theorems |
| **Mitigation** | (1) Exhaustive review of the formal model before proceeding to proofs. (2) Construct a small (≤5 state) model instance and manually verify all properties. (3) Check for known pitfalls in Event Calculus (frame problem, ramification problem). |
| **Fallback if mitigation fails** | Fix the model if the inconsistency is local. If the inconsistency is fundamental (e.g., the combination of spatial and temporal semantics is incoherent), ABANDON. |

### Risk R5: GJK/EPA Numerical Instability Breaks Soundness

| Attribute | Detail |
|-----------|--------|
| **Risk** | GJK/EPA operates in floating-point arithmetic. For near-boundary configurations (objects barely touching / barely separated), GJK may give incorrect answers, causing the CEGAR loop to accept spurious counterexamples or reject genuine ones. |
| **Probability** | 30% |
| **Impact** | Medium-High — affects T3 soundness |
| **Mitigation** | (1) Use ε-thickened predicates: object A is "in region R" iff distance(A, R) ≤ ε, eliminating boundary cases. (2) Cite existing GJK stability analyses (e.g., Montanari et al. 2017). (3) In proofs, separate the "exact arithmetic" correctness from the "approximate arithmetic" implementation, and state the approximation bounds. |
| **Fallback if mitigation fails** | Use exact arithmetic (rational GJK) at the cost of ~100× slowdown. This preserves soundness but may make the verifier impractical for large scenes. State the tradeoff explicitly. |

### Risk R6: Bug-Finding Evaluation Finds Zero Bugs

| Attribute | Detail |
|-----------|--------|
| **Risk** | The MRTK/Meta SDK corpus has no spatial-temporal bugs detectable by the verifier, or the bugs are trivial and already known. |
| **Probability** | 25% |
| **Impact** | High — the primary evaluation claim is bug-finding |
| **Mitigation** | (1) During theory, perform a manual inspection of ≥10 MRTK interaction patterns to estimate bug prevalence. (2) Define "bug" broadly: unreachable states, deadlocks, and accessibility violations all count. (3) Inject synthetic bugs as a secondary evaluation dimension. |
| **Fallback if mitigation fails** | Pivot evaluation to scalability and specification expressiveness. Demonstrate that Choreo can express real XR interactions and verify their properties, even if no bugs are found. Reframe as "verified correct" rather than "found bugs." Publication venue may shift from SE to FM. |

### Risk R7: T4 Compositional Separation Never Applies

| Attribute | Detail |
|-----------|--------|
| **Risk** | Real XR scenes have too many cross-region dependencies (shared events, transitive spatial predicates) for the separation criterion to be satisfied. T4 is correct but vacuous. |
| **Probability** | 30% |
| **Impact** | Medium — T4 is a stretch goal |
| **Mitigation** | (1) Analyze ≥5 MRTK scenes for spatial separation structure during theory. (2) If exact separation is rare, weaken to "approximate separation" (≤k cross-region dependencies). (3) Quantify the k threshold where compositional verification still provides speedup. |
| **Fallback if mitigation fails** | Drop T4 from the headline contributions. Mention as future work. Reduces scope from "full" to "Reduced-A." The paper is still publishable with T2+T3 as the core contributions. |

---

## 7. Final Verdict Template

To be filled after the theory deliverables (`approach.json` and `paper.tex`) are complete and evaluated against this framework.

```
═══════════════════════════════════════════════════════════════════════
              CHOREO THEORY STAGE — FINAL VERDICT
═══════════════════════════════════════════════════════════════════════

VERDICT: [STRONG CONTINUE / CONDITIONAL CONTINUE / ABANDON]

───────────────────────────────────────────────────────────────────────
THEORY QUALITY SCORES
───────────────────────────────────────────────────────────────────────

  Rigor:            ___ / 10   [justification in ≤2 sentences]
  Novelty:          ___ / 10   [justification in ≤2 sentences]
  Load-Bearing:     ___ / 10   [justification in ≤2 sentences]
  Completeness:     ___ / 10   [justification in ≤2 sentences]
  Honesty:          ___ / 10   [justification in ≤2 sentences]
  Implementability: ___ / 10   [justification in ≤2 sentences]

  THEORY QUALITY (mean):  ___ / 10

───────────────────────────────────────────────────────────────────────
PROJECT COMPOSITE
───────────────────────────────────────────────────────────────────────

  Value (V):             5 / 10  [from depth check; update if warranted]
  Difficulty (D):        7 / 10  [from depth check; update if warranted]
  Best-Paper Pot. (BP):  5 / 10  [from depth check; update if warranted]
  Feasibility (F):       8 / 10  [from depth check; update if warranted]
  Theory Quality (TQ):   ___ / 10

  PROJECT COMPOSITE = 0.3(TQ) + 0.2(V) + 0.2(D) + 0.15(BP) + 0.15(F) = ___ / 10

───────────────────────────────────────────────────────────────────────
THEOREM STATUS
───────────────────────────────────────────────────────────────────────

  T1 (Spatial Types):        [PROVEN / SKETCHED / OPEN / FAILED]
  T2 (Geometric Pruning):    [PROVEN / SKETCHED / OPEN / FAILED]
  T3 (Spatial CEGAR):        [PROVEN / SKETCHED / OPEN / FAILED]
  T4 (Compositionality):     [PROVEN / SKETCHED / OPEN / FAILED]

───────────────────────────────────────────────────────────────────────
FATAL FLAWS
───────────────────────────────────────────────────────────────────────

  Count: ___
  List:
    1. [description, severity, affected theorems]
    2. ...

───────────────────────────────────────────────────────────────────────
AMENDMENTS REQUIRED (for CONDITIONAL CONTINUE only)
───────────────────────────────────────────────────────────────────────

  Count: ___
  List:
    1. [amendment description, owner, deadline, blocking status]
    2. ...

───────────────────────────────────────────────────────────────────────
DELIVERABLE STATUS
───────────────────────────────────────────────────────────────────────

  approach.json:
    - PASS criteria met:  ___ / [total]
    - PARTIAL:            ___
    - FAIL:               ___

  paper.tex:
    - Size:               ___ KB  [≥50 KB required]
    - Compiles:           [YES / NO]
    - References:         ___     [≥30 required]
    - PASS criteria met:  ___ / [total]
    - PARTIAL:            ___
    - FAIL:               ___

───────────────────────────────────────────────────────────────────────
PROBABILITY ESTIMATES
───────────────────────────────────────────────────────────────────────

  Publication probability:       ___ %
    (top venue: CAV/PLDI/OOPSLA/ICSE)
  Best-paper probability:        ___ %
  Bug-finding evaluation success: ___ %
  On-time delivery:              ___ %

───────────────────────────────────────────────────────────────────────
RECOMMENDED SCOPE
───────────────────────────────────────────────────────────────────────

  [Full / Reduced-A / Reduced-B / CAV-only]

  Scope definitions:
    Full:       T1 + T2 + T3 + T4, all five math results, full MRTK evaluation
    Reduced-A:  T2 + T3 + T1(binary only), drop T4, reduced evaluation
    Reduced-B:  T2 + T3 only, drop T1 and T4, focused bug-finding evaluation
    CAV-only:   T3 only with T2 as supporting lemma, minimal evaluation

───────────────────────────────────────────────────────────────────────
EVALUATOR VOTES
───────────────────────────────────────────────────────────────────────

  Evaluator 1 (___):  [CONTINUE / ABANDON]  —  [1-sentence rationale]
  Evaluator 2 (___):  [CONTINUE / ABANDON]  —  [1-sentence rationale]
  Evaluator 3 (___):  [CONTINUE / ABANDON]  —  [1-sentence rationale]

  Consensus: [UNANIMOUS / MAJORITY (2-1) / SPLIT — Chair decides]

───────────────────────────────────────────────────────────────────────
CHAIR'S SUMMARY (≤5 sentences)
───────────────────────────────────────────────────────────────────────

  [Chair's overall assessment, key concerns, and forward-looking guidance]

═══════════════════════════════════════════════════════════════════════
```

---

## Appendix A: Verification Procedure

The following procedure should be followed when evaluating the theory deliverables:

1. **Structural check** (30 min): Verify `approach.json` and `paper.tex` exist, meet size requirements, and compile. Score §1.1 and §1.2 structural criteria.

2. **Formal model review** (2 hr): Read the formal model (spatial-temporal event automata, spatial predicates, Event Calculus integration). Check for internal consistency. Score definitions against §1.2.2.

3. **Theorem-by-theorem audit** (4 hr): For each T1–T4, walk through the §2 checklist. Score each sub-item. Flag any FAIL items.

4. **Cross-cutting review** (1 hr): Evaluate §4 concerns (dependencies, honesty, novelty, relevance, reproducibility).

5. **Scoring** (30 min): Assign scores per §5 rubric. Compute composite.

6. **Verdict** (30 min): Apply §3 decision framework. Fill §7 template. Three evaluators vote independently, then discuss to consensus.

**Total estimated time**: 8–9 hours for a thorough evaluation.

## Appendix B: Precedent Calibration

To calibrate scoring, the following reference points from the prior depth check are noted:

- **Composite 6.25/10** was assessed as borderline, warranting CONDITIONAL CONTINUE with one dissent.
- **Value 5/10** reflects that the target audience is narrow (50–200 XR engineers) but the formal methods contribution has broader relevance.
- **Difficulty 7/10** reflects that 14 subsystems, ~157K LoC, and 5 math results are genuinely hard but not unprecedented.
- **Best-Paper 5/10** reflects that the work is solid but unlikely to win best paper at a top venue without a strong evaluation.
- **Laptop CPU 8/10** reflects that the verification workload is bounded (finite-state, bounded scenes) and feasible on commodity hardware.

The theory stage evaluation should produce scores consistent with this calibration. If theory quality is high (≥7), it should pull the composite above 6.25. If theory quality is low (<5), it should pull the composite below 6.25, triggering ABANDON.

## Appendix C: Glossary

| Term | Definition |
|------|-----------|
| **CEGAR** | Counterexample-Guided Abstraction Refinement |
| **GJK** | Gilbert-Johnson-Keerthi distance algorithm for convex shapes |
| **EPA** | Expanding Polytope Algorithm (penetration depth for overlapping convex shapes) |
| **EC** | Event Calculus |
| **BDD** | Binary Decision Diagram |
| **CSG** | Constructive Solid Geometry |
| **LP** | Linear Programming |
| **MRTK** | Mixed Reality Toolkit (Microsoft) |
| **XR** | Extended Reality (VR + AR + MR) |
| **BMC** | Bounded Model Checking |
| **Treewidth** | Graph-theoretic parameter bounding the "tree-likeness" of a graph |
