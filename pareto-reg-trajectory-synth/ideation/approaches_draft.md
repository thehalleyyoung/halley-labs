# RegSynth: Competing Approaches

**Slug:** `pareto-reg-trajectory-synth`
**Stage:** Ideation — Approach Design
**Date:** 2026-03-08

---

## Approach A: "ConflictProver" — MUS-First Regulatory Conflict Detection Engine

### Thesis

Build the world's first formal conflict prover for multi-jurisdictional AI regulation: a lean, sharply focused tool that answers the single most valuable question in cross-border compliance — *"Are these obligations simultaneously satisfiable, and if not, which specific articles are in irreducible conflict?"* — with machine-checkable certificates.

### 1. Extreme Value Delivered

**Who desperately needs this:**

- **Chief Compliance Officers at multinationals** facing EU AI Act enforcement (August 2026, €35M penalties). They currently rely on legal teams manually reading regulations from multiple jurisdictions and hoping to spot conflicts. When conflicts are discovered post-implementation, remediation costs 5–50× the upfront analysis cost.
- **Regulatory bodies** (EU AI Office, NIST) need systematic impact analysis of how their regulations interact with other jurisdictions. Currently done by lawyers reading foreign law — no formal analysis exists.
- **AI auditing firms** (Holistic AI, Credo AI) need defensible conflict assessments that go beyond "we think these might conflict" to "here is a machine-verified proof that these three articles are mutually unsatisfiable."

**Why this specific approach maximizes value:**

ConflictProver does one thing brilliantly: it detects, proves, and diagnoses regulatory conflicts. No existing tool — not GRC platforms, not LLM assistants, not deontic logic systems — can produce a formal proof that specific articles across specific jurisdictions are in irreducible conflict. This is commercially irreplaceable. The depth check identified this as "the most commercially and academically defensible feature" (Amendment A8).

By stripping away Pareto optimization, temporal trajectories, and remediation planning, ConflictProver achieves:
- **Faster time-to-value**: smaller artifact, fewer dependencies, earlier ship date
- **Sharper narrative**: "we prove regulatory conflicts exist" is a crisper story than "we optimize compliance trajectories"
- **Higher credibility**: a lean tool that does one thing provably correct is more trustworthy than a sprawling system with many unvalidated components

**What becomes possible:**
- An organization presents a formal infeasibility certificate to regulators: "Articles 12(1) of the EU AI Act and 5(1)(c) of GDPR are mutually unsatisfiable for high-risk AI systems processing personal data. Here is the machine-checkable proof. We request regulatory guidance on priority."
- A regulatory body runs ConflictProver on a draft regulation to identify conflicts with existing frameworks *before* publication.
- An auditing firm provides clients with certified conflict analyses as part of due diligence.

### 2. Genuine Difficulty as a Software Artifact

**Hard Subproblems:**

**H1. Regulatory DSL with Conflict-Oriented Type System (Research-grade PL design)**
A typed DSL optimized for expressing obligations that may conflict. Obligation types (OBL/PERM/PROH) indexed by jurisdiction with explicit conflict annotations. Simpler than the full formalizability-graded DSL — no temporal sorts, no confidence algebra — but still requires: conditional triggers, cross-references, exemptions, jurisdiction-specific scoping, and a composition algebra (⊗, ⊕, ▷, ⊘) with algebraic laws. The key design challenge: the type system must be rich enough to capture real regulatory structure but simple enough that every well-typed obligation has a sound SMT encoding.

**H2. Faithful Encoding of 300+ Regulatory Articles (Domain engineering)**
Translating 5 regulatory frameworks (~300 articles) into the DSL with semantic fidelity. Each article requires careful analysis of logical structure, hard vs. soft constraint classification, and cross-jurisdictional concept alignment. The EU AI Act alone has 113 articles. Different frameworks use different terminology, risk taxonomies, and enforcement models. A cross-jurisdictional ontology maps equivalent concepts (e.g., "high-risk" in EU ≈ "high impact" in NIST ≈ "critical information infrastructure" in China).

**H3. Sound Compositional Encoding to SMT (Compiler correctness)**
Translation function τ from typed obligations to QF-LIA + arrays + UF must be provably sound: if τ(O) is SAT, then O is genuinely satisfiable. Compositionality: τ(O₁ ⊗ O₂) = τ(O₁) ∧ τ(O₂). This is the load-bearing correctness property — a wrong encoding produces wrong conflict verdicts with formal-looking but incorrect certificates. Requires careful treatment of: soft vs. hard constraints, conditional obligations, jurisdictional overrides, and exemption clauses.

**H4. Comprehensive MUS Enumeration with Regulatory Provenance (Primary algorithmic challenge)**
When constraints are UNSAT, extract *all* minimal unsatisfiable subsets (MUS), not just one. Each MUS represents an irreducible conflict core. Map each MUS back through τ⁻¹ to specific regulatory articles. Challenges: (a) the number of MUS can be exponential — need bounded enumeration with prioritization (smallest MUS first, cross-jurisdictional MUS prioritized over intra-jurisdictional); (b) provenance tracking through the compilation pipeline requires maintaining a bidirectional mapping between SMT variables and regulatory article numbers; (c) MUS extraction quality varies across solvers — need multi-solver cross-validation.

**H5. Machine-Checkable Infeasibility Certificates (Proof-carrying compliance)**
Two certificate types: (a) *compliance certificates* — a satisfying assignment mapped back to a concrete compliance strategy; (b) *infeasibility certificates* — resolution proofs extracted from UNSAT cores, independently verifiable by a standalone verifier (<2,500 LoC). The certificates must be sound: if a certificate says "infeasible," the obligations are genuinely unsatisfiable (no false alarms). Certificate extraction from Z3/CVC5 proof traces is non-trivial — proof formats differ, proofs can be enormous, and minimization is needed for human interpretability.

**H6. Multi-Solver Cross-Validation (Engineering reliability)**
Run Z3 and CVC5 on identical instances. Verify: same satisfiability verdicts, compatible MUS extractions (MUS may differ but union should cover the same conflict space). Cross-validation catches encoding bugs that single-solver testing misses. The solver abstraction layer must handle API differences, timeout policies, and proof format translation.

**Architectural Challenge:** End-to-end provenance. Every variable in the SMT encoding must trace back to a specific obligation in a specific article in a specific jurisdiction. Every MUS must decompose into a human-readable regulatory conflict diagnosis. This requires provenance annotations throughout the pipeline: DSL → IR → SMT formula → solver result → MUS → diagnosis. Provenance breaks easily when optimizations (common subexpression elimination, constraint simplification) are applied.

### 3. New Math Required (Load-Bearing Only)

**M1. Obligation Algebra Soundness (Grade: B)**
Prove τ is a homomorphism from (Obligations, ⊗, ⊕, ▷, ⊘) to (SMT, ∧, ∨, priority-ite, except-ite). Soundness: SAT(τ(O)) → Satisfiable(O). Partial completeness for the fully-formalizable fragment. This is a standard compiler-correctness result instantiated for a novel domain, but it's absolutely load-bearing: without it, the certificates are meaningless.

**M2. MUS Coverage Theorem (Grade: B)**
Given a bounded enumeration budget of k MUS extractions, prove that the prioritized enumeration algorithm (smallest-first, cross-jurisdictional-first) covers all "important" conflict cores. Define "importance" via a regulatory relevance metric (cross-jurisdictional conflicts are more important than intra-jurisdictional ones; conflicts involving hard deadlines are more important than those involving soft recommendations). Prove: with budget k = O(n log n) where n is the number of obligations, the algorithm covers all cross-jurisdictional conflict cores of size ≤ c for a constant c depending on the jurisdiction count.

**M3. Certificate Correctness (Grade: C+)**
Prove: (a) compliance certificates are sound — the strategy satisfies all hard constraints; (b) infeasibility certificates are sound — the resolution proof is valid and the MUS is genuinely minimal (removing any obligation makes the remaining set satisfiable). Standard proof theory, but the composition with provenance tracking is novel.

**M4. Conflict Density Characterization (Grade: B-)**
Characterize the expected number and size distribution of MUS as a function of jurisdiction count, obligation density, and constraint tightness. Prove: for randomly structured regulatory instances with n obligations across j jurisdictions and conflict probability p, the expected number of MUS is Θ(n^c · j^d) for constants c, d depending on p. This provides theoretical backing for the scalability claims and guides the enumeration budget.

### 4. Best-Paper Argument

**Why this has best-paper potential:**

1. **First of its kind.** No existing tool can formally prove multi-jurisdictional regulatory infeasibility. This is a genuine "0 to 1" contribution — not an incremental improvement over prior work.

2. **Perfect timing.** EU AI Act enforcement begins August 2026. The paper arrives when multinationals are desperately seeking tools that can identify regulatory conflicts formally.

3. **Clean, sharp contribution.** The paper tells a crisp story: "We show that multi-jurisdictional AI regulatory compliance can be formalized as a constraint satisfaction problem, we prove that real-world regulatory corpora contain irreducible conflicts, and we provide the first machine-checkable certificates of regulatory infeasibility." No bloat, no over-engineering.

4. **The artifact speaks for itself.** Running ConflictProver on the EU AI Act + GDPR + China Interim Measures and producing concrete, verified conflict reports is a compelling demonstration. The demo practically writes itself.

5. **Target venue fit.** ICSE Tools Track values substantial, well-evaluated tools with formal properties. ConflictProver is exactly this — a novel tool backed by formal guarantees. FAccT values accountability tools with real-world relevance. ConflictProver addresses both.

**Weakness:** The mathematical depth is moderate — soundness proofs are standard compiler-correctness results in a new domain. A reviewer seeking deep theoretical novelty may find this insufficient. Best-paper potential depends on the tool's impact being valued over mathematical depth.

### 5. Hardest Technical Challenge

**MUS enumeration with regulatory provenance at scale.**

The number of MUS in an unsatisfiable constraint system can be exponential. For a 300-obligation, 5-jurisdiction instance with moderate conflict density, there may be thousands of distinct conflict cores. Enumerating all of them is intractable; enumerating the "important" ones (small, cross-jurisdictional) requires a prioritized search strategy that:

1. Uses incremental SAT solving to avoid re-solving from scratch for each MUS
2. Maintains provenance through the MUS extraction (each element of the MUS maps to a specific regulatory article)
3. Deduplicates "equivalent" conflict cores (two MUS that differ only in intra-jurisdictional obligations that are logical consequences of each other)
4. Presents results in a ranked, human-interpretable format

**Mitigation:** Start with MARCO (Liffiton & Malik, 2016) for MUS enumeration with an incremental SAT oracle. Add provenance as annotations on the formula's clause-to-obligation mapping. Implement a regulatory relevance ranking that prioritizes cross-jurisdictional, small-sized, deadline-sensitive conflict cores. Kill gate: if MUS enumeration on a 100-obligation, 3-jurisdiction instance takes >60 seconds, switch to single-MUS extraction with iterative relaxation.

### 6. Scores

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| **Value** | 8/10 | Solves the single most valuable problem (conflict detection) with zero existing alternatives |
| **Difficulty** | 6/10 | Genuine compiler + solver integration challenge; MUS enumeration is hard; but scope is contained |
| **Potential** | 6/10 | Clean first-of-kind contribution; but moderate math depth limits best-paper ceiling |
| **Feasibility** | 9/10 | Lean scope, well-understood solver technology, no temporal complexity |

### 7. LoC Breakdown

| Component | LoC | Language | Novel? |
|-----------|-----|----------|--------|
| DSL & Type System (lexer, parser, IR, type checker) | 10,000 | Rust | Yes — conflict-oriented obligation types |
| Regulatory Data Corpus (5 frameworks, ~300 articles) | 15,000 | Custom DSL | Domain data, not algorithmic novelty |
| Constraint Encoding Engine (obligation→SMT, hard/soft) | 8,000 | Rust | Yes — sound compositional translation |
| Solver Backend (Z3/CVC5 integration, MUS extraction) | 9,000 | Rust | Yes — multi-solver MUS with provenance |
| Certificate Generator (compliance + infeasibility proofs) | 4,500 | Rust | Yes — proof extraction + standalone verifier |
| Evaluation Framework (benchmarks, metrics, regression) | 14,000 | Python + Rust | Standard methodology |
| CLI / Reporting | 4,000 | Rust + HTML | Standard |
| Infrastructure | 4,500 | Rust | Standard |
| **TOTAL** | **~69,000** | | |
| **Novel algorithmic core** | **~31,500** | | 46% of total |

### 8. Key Risks & Kill Gates

| Gate | Week | Condition | Fail Action |
|------|------|-----------|-------------|
| G1: DSL expressiveness | 4 | Encode 50+ EU AI Act articles | Simplify type system |
| G2: MUS quality | 6 | Extract meaningful MUS from 100-obligation instance in <30s | Switch to single-MUS |
| G3: Cross-jurisdiction conflicts | 8 | Find ≥5 genuine conflicts in EU+GDPR+China corpus | Encoding methodology needs revision |
| G4: Certificate verification | 10 | Standalone verifier confirms 100% of generated certificates | Certificate format needs redesign |
| G5: External validation | 14 | ≥80% expert agreement on 50-article encoding sample | Major encoding overhaul |

---

## Approach B: "RegSynth-Complete" — Formalizability-Graded Synthesis with Bounded Uncertainty

### Thesis

Build a layered constraint-solving engine whose intellectual crown jewel is the **formalizability completeness theorem**: the first formal proof that partial regulatory formalization — honestly admitting that 30–40% of obligations resist formalization — still yields Pareto-optimal compliance strategies with quantified, bounded uncertainty over the opaque fragment. This is the "full architecture" approach with all depth check amendments applied.

### 1. Extreme Value Delivered

**Who desperately needs this:**

All stakeholders from Approach A, plus:
- **Strategic compliance planners** who need not just "these conflict" but "here are your 12 best compliance strategies, ranked by cost/time/risk/burden, with formal guarantees on what we can prove and honest bounds on what we can't."
- **Boards of directors** making multi-million-dollar compliance investment decisions who need to see the full trade-off space, not a single "recommended" strategy.
- **Regulatory harmonization bodies** (OECD AI Policy Observatory, Global Partnership on AI) who need to understand not just where regulations conflict, but what the optimal compliance landscape looks like and how it changes as regulations evolve.

**Why this specific approach maximizes value:**

The key insight: real-world regulatory compliance is not a binary "feasible/infeasible" question. Most multi-jurisdictional deployments face a spectrum of partially-compliant strategies with vastly different cost profiles. Two strategies achieving 95% compliance coverage may differ by 10× in implementation cost. Without optimization, organizations routinely choose dominated strategies.

But the deeper insight — and what makes this approach intellectually distinctive — is the **honest treatment of partial formalizability**. Every prior attempt to formalize regulations either (a) pretends all obligations are formalizable (intellectually dishonest, produces overconfident results) or (b) gives up entirely when some obligations resist formalization. RegSynth-Complete is the first system to prove that *partial formalization is formally valid*: the computed Pareto frontier is an ε-approximation of the true frontier restricted to the formalizable fragment, with bounded worst-case impact from opaque obligations. This turns a weakness (we can't formalize everything) into a strength (we can formally characterize what we can and can't say).

**What becomes possible:**
- Everything from Approach A (conflict detection, infeasibility proofs, MUS diagnosis)
- Plus: "Here are 15 Pareto-optimal compliance strategies. Strategy 7 costs $2.1M and achieves 94% coverage in 8 months. Strategy 12 costs $4.8M and achieves 99% coverage in 4 months. The remaining 6% of obligations are opaque — our worst-case bound on their impact is +$300K/+2months."
- Plus: "When the EU AI Act Phase 3 takes effect in August 2026, strategies 3, 7, and 11 become infeasible. Here is the updated Pareto frontier."

### 2. Genuine Difficulty as a Software Artifact

**Hard Subproblems (extends Approach A with 4 additional challenges):**

**H1–H6: All from Approach A** (DSL, encoding, sound compilation, MUS enumeration, certificates, cross-validation) — these are prerequisites, not duplicated effort.

**H7. Formalizability Grading System with Confidence Algebra (Novel PL contribution)**
Each obligation in the DSL carries a formalizability grade: Full (encoding captures legal intent completely), Partial(α) where α ∈ [0,1] (encoding captures fraction α of the obligation's semantic content), or Opaque (obligation resists formalization — tracked but excluded from optimization). The grades propagate through the composition algebra: γ(O₁ ⊗ O₂) = min(γ(O₁), γ(O₂)) for conjunction; γ(O₁ ⊕ O₂) = max(γ(O₁), γ(O₂)) for disjunction; jurisdiction override and exception have their own propagation rules. The confidence algebra must be: (a) compositional — grades of compound obligations are determined by grades of components; (b) sound — the propagated grade never overstates confidence; (c) informative — the grade provides a meaningful bound on the gap between the formal encoding and legal intent.

**H8. Pareto Frontier Computation via Iterative Weighted Partial MaxSMT (Novel algorithm)**
Computing ε-approximate Pareto frontiers over a 4-dimensional cost space (implementation cost, time-to-compliance, residual risk, operational burden). Each Pareto point requires a MaxSMT solve; the frontier may contain hundreds of points. The algorithm: (a) solve MaxSMT with dimension d as objective, other dimensions as ε-bounded constraints; (b) add blocking clause excluding the found point's dominance cone; (c) iterate until no new non-dominated points exist within ε. Must guarantee ε-coverage, avoid enumerating dominated points, and terminate in bounded time. ILP (Gurobi) provides the baseline implementation; MaxSMT extends with richer logical constraint handling.

**H9. Temporal Regulatory Transition Model (Research extension)**
Regulations phase in over multi-year timelines. Model compliance as a trajectory through a time-varying constraint landscape with bounded transition budgets (an organization can't restructure everything in one quarter). ILP handles the static case; MaxSMT temporal unrolling is the research delta. Prove: per-timestep optimization can produce dominated trajectories (constructive 3-timestep example).

**H10. Bounded Uncertainty Propagation for Opaque Obligations (Novel)**
Opaque obligations are excluded from optimization but their worst-case impact on the cost vector must be bounded. For each opaque obligation, compute a worst-case cost multiplier assuming the most expensive possible compliance interpretation. Propagate these bounds through the Pareto computation to produce a "confidence envelope" around each Pareto point: the true cost lies within this envelope with probability ≥ 1-δ. This is the technical core of the formalizability completeness theorem.

**Architectural Challenge:** The full pipeline has 11 components that must compose correctly: DSL → type checker → formalizability grader → IR → constraint encoder → ILP backend → MaxSMT backend → Pareto enumerator → temporal unroller → certificate generator → remediation planner. End-to-end provenance must be maintained through all stages. The formalizability grade must propagate correctly through every transformation.

### 3. New Math Required (Load-Bearing Only)

**M1. Formalizability Completeness Theorem (Grade: B+ — Crown Jewel)**
*Statement:* Let O = O_F ∪ O_P ∪ O_X be a set of regulatory obligations partitioned into Fully formalizable, Partially formalizable (with confidence α_i), and Opaque. Let τ be the sound encoding function and P* the true Pareto frontier over all obligations. Let P_F be the computed Pareto frontier restricted to O_F ∪ {o ∈ O_P : α_o ≥ threshold}. Then:
- P_F is an ε-approximation of the true frontier P*_F restricted to the formalizable fragment
- For each point p ∈ P_F, the true cost c*(p) satisfies c*(p) ∈ [c_F(p), c_F(p) + Δ_X] where Δ_X is computable from the opaque obligations' worst-case cost bounds
- The uncertainty bound Δ_X is tight: there exist regulatory instances where the gap is achieved

*Why load-bearing:* Without this theorem, any critic can dismiss RegSynth by pointing to a single vague regulation. With it, the system's outputs are formally sound within quantified uncertainty. This is what makes partial formalization rigorous rather than ad hoc.

**M2. Obligation Algebra Soundness (Grade: B)**
Same as Approach A M1 — soundness of τ as a homomorphism. Extended to handle formalizability grades: τ preserves the grade structure, and the soundness guarantee is qualified by the grade.

**M3. Pareto ε-Coverage via Iterative MaxSMT (Grade: B-)**
Prove: (a) each returned point is Pareto-optimal; (b) the algorithm terminates in at most O(1/ε^{d-1}) iterations where d is the cost dimension; (c) the returned set is an ε-cover (every point on the true frontier is within ε of some returned point). Adaptation of ε-constraint scalarization to MaxSMT — the algorithmic contribution is the blocking clause construction and the termination proof for the MaxSMT setting (not guaranteed by standard OR results due to the richer constraint language).

**M4. Temporal Trajectory Dominance (Grade: B- — Motivating Formalization)**
Per-timestep Pareto optimality does not imply trajectory Pareto optimality. Constructive proof via 3-timestep, 2-jurisdiction instance. This is a known phenomenon (Bellman 1957) instantiated for the regulatory domain — honestly graded as motivating formalization, not a novel theorem.

**M5. Confidence Algebra Soundness (Grade: C+)**
The formalizability grade propagation rules form a sound abstract interpretation: the computed grade of a compound obligation is a lower bound on the true formalizability of the composition. Proof by structural induction on the obligation algebra.

**M6. Incremental Pareto Maintenance (Grade: C+)**
When obligations change (regulatory amendment), the existing Pareto frontier can be incrementally updated: test surviving points against new constraints, re-solve only in uncovered regions, merge and filter. Prove correctness: the incremental frontier is an ε-cover of the true new frontier.

### 4. Best-Paper Argument

**Why this has best-paper potential:**

1. **The formalizability completeness theorem is genuinely novel.** No prior work provides a formal treatment of what partial regulatory formalization guarantees. This is the kind of "honest formalism" that best-paper committees love — it doesn't sweep difficulties under the rug, it formalizes them.

2. **Three-layered contribution, each independently publishable.** (a) Formal infeasibility detection (from Approach A); (b) Pareto-optimal synthesis with bounded uncertainty; (c) temporal trajectory extension. Even if one layer is weak, the others carry the paper.

3. **The intellectual honesty is a selling point.** Most formal methods papers pretend their formalization is complete. RegSynth-Complete proves that *incomplete formalization is still formally useful* — a meta-result about the limits and guarantees of formal compliance analysis.

4. **Comprehensive artifact.** ~115K LoC system covering the full pipeline from regulatory text to certified compliance strategies. ICSE Tools Track values exactly this level of completeness.

5. **ILP-vs-MaxSMT comparison as primary evaluation axis.** This provides an honest, quantitative answer to "does MaxSMT add value over ILP for regulatory compliance?" — a question of independent interest.

**Weakness:** The scope is ambitious. If temporal trajectories, Pareto synthesis, and the completeness theorem all need to work, there are many potential failure points. The kill probability is higher than Approach A.

### 5. Hardest Technical Challenge

**Proving the formalizability completeness theorem (M1) in a way that's both rigorous and non-trivial.**

The risk: the theorem might be "obvious" — if you formalize the easy parts correctly and bound the hard parts worst-case, of course you get a valid approximation. A reviewer could dismiss it as "just sensitivity analysis." The theorem must be deeper than this:

1. **The ε-approximation must be tight.** Show instances where the gap between P_F and P* is exactly Δ_X, not just bounded by it. This requires constructing adversarial regulatory instances.
2. **The composition of formalizability grades must be non-trivial.** Show that naïve grade propagation (e.g., always taking min) is too conservative, and the actual propagation rules exploit structural properties of the obligation algebra to give tighter bounds.
3. **The relationship between formalizability threshold and frontier quality must be characterized.** Prove a trade-off curve: as the threshold increases (more obligations excluded as opaque), the uncertainty Δ_X decreases but the frontier P_F shrinks. Characterize the optimal threshold.

**Mitigation:** Start with the simple version of the theorem (just worst-case bounds) and iterate toward tighter characterization. If the tight version is too hard, the simple version is still publishable — it's the first formal treatment of partial formalization in any domain.

### 6. Scores

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| **Value** | 8/10 | Everything from A plus Pareto synthesis and temporal planning |
| **Difficulty** | 8/10 | Full 11-component pipeline, 6 math contributions, dual-solver backend |
| **Potential** | 8/10 | Formalizability theorem is genuinely novel; three-layered contribution |
| **Feasibility** | 6/10 | Ambitious scope; many potential failure points; temporal extension may not work at scale |

### 7. LoC Breakdown

| Component | LoC | Language | Novel? |
|-----------|-----|----------|--------|
| DSL & Type System (with formalizability grading) | 15,200 | Rust | Yes |
| Regulatory Data Corpus (5 core frameworks) | 18,000 | Custom DSL | Domain data |
| Temporal Model (transition system, phase-in) | 6,500 | Rust | Yes |
| Constraint Encoding Engine (ILP + SMT dual) | 11,200 | Rust | Yes |
| Solver Backend (Z3/CVC5/Gurobi, MUS, Pareto) | 12,500 | Rust | Yes |
| Pareto Synthesis Engine (multi-objective, trajectory) | 7,500 | Rust | Yes |
| Remediation Planner | 6,500 | Rust | Partially novel |
| Certificate Generator (3 types + verifier) | 6,000 | Rust | Yes |
| Evaluation Framework | 20,000 | Python + Rust | Standard |
| CLI / Reporting | 5,500 | Rust + HTML | Standard |
| Infrastructure | 6,100 | Rust | Standard |
| **TOTAL** | **~115,000** | | |
| **Novel algorithmic core** | **~55,400** | | 48% of total |

### 8. Key Risks & Kill Gates

| Gate | Week | Condition | Fail Action |
|------|------|-----------|-------------|
| G1: DSL expressiveness | 4 | 50+ EU AI Act articles with formalizability grades | Simplify grading to binary |
| G1.5: Scalability | 6 | 3×100×5 in <10 min (ILP or MaxSMT) | Reduce temporal depth |
| G2: ILP baseline | 8 | ILP Pareto frontier for static 5×300 case | Reduce to 3 jurisdictions |
| G3: MaxSMT delta | 12 | Measurable advantage on ≥1 axis | Pivot: paper is DSL + ILP + M1 |
| G4: Frontier quality | 14 | Hypervolume ratio >0.9 on planted benchmarks | Refine ε |
| G5: End-to-end | 18 | Full pipeline on 5-jurisdiction corpus | Reduce temporal depth |

---

## Approach C: "RegVerify" — Regulatory Type Theory with Verified Compilation

### Thesis

Recast multi-jurisdictional regulatory compliance as a **type-theoretic problem**: obligations are types, compliance strategies are inhabitants, regulatory conflicts are type errors, and the compilation from types to constraints is *verified* (proven correct within the system). The crown jewel is a **verified regulatory constraint compiler** — the first tool where the translation from legal obligations to formal constraints is itself formally verified, not just tested.

### 1. Extreme Value Delivered

**Who desperately needs this:**

All stakeholders from Approaches A and B, plus:
- **High-assurance compliance environments** (financial services, healthcare, defense-adjacent AI) where a compliance tool's own correctness is a liability question. If RegSynth says "you're compliant" and you're not, the tool vendor faces legal exposure. A verified compiler eliminates this class of errors.
- **Regulatory technology (RegTech) vendors** building compliance automation who need a formally-verified core engine they can embed in their products with confidence.
- **Standards bodies** (ISO/IEC JTC 1/SC 42) developing compliance assessment standards who need reference implementations with provable properties.

**Why this specific approach maximizes value:**

The fundamental problem with any compliance tool is trust: why should an organization trust that the tool's formalization of regulations is correct? Approaches A and B address this through external expert validation — a human-in-the-loop quality check. Approach C addresses it at the *architectural level*: the compilation from regulatory obligations (expressed in a typed DSL) to constraint systems is formally verified, meaning the compiler itself cannot introduce errors. If the obligations are correctly expressed in the DSL (validated by legal experts), the constraint encoding is *guaranteed* correct by construction.

This is a qualitatively different trust model:
- Approaches A/B: "We tested the encoding and experts reviewed a sample"
- Approach C: "The encoding is provably correct — here's the machine-checked proof"

**What becomes possible:**
- Everything from Approach A (conflict detection, infeasibility certificates)
- Plus: the infeasibility certificates are *doubly* certified — the conflict proof is valid AND the encoding that produced it is verified
- Plus: regulatory compliance tooling with a formally verified core, suitable for high-assurance environments
- Plus: a reusable verified framework that future regulatory DSLs can build on

### 2. Genuine Difficulty as a Software Artifact

**Hard Subproblems:**

**H1. Dependently-Typed Regulatory DSL (Deep PL contribution)**
A DSL where obligation types carry rich information: jurisdiction index (dependent on a jurisdiction lattice), temporal interval (dependent on a regulatory timeline), obligation modality (OBL/PERM/PROH with deontic semantics), and constraint arity (dependent on the compliance strategy space). Conflict detection becomes type-checking: if two obligations have a conjunction type (O₁ ⊗ O₂) that is uninhabited, they conflict. The type system must be: (a) decidable — type-checking terminates; (b) sound — well-typed obligations have sound encodings; (c) expressive — can capture the regulatory patterns identified in the corpus.

This is substantially harder than a simple typed DSL. Dependent types indexed by jurisdiction and temporal interval require a non-trivial metatheory (normalization, decidability, type preservation under the obligation algebra operations). The payoff: type-checking provides a fast, compositional conflict pre-filter before invoking the expensive SMT solver.

**H2. Verified Compilation Pipeline (Core challenge)**
The compilation from typed obligations to constraint formulas must be *verified* — accompanied by a machine-checked proof (in Lean 4 or Coq) that the compilation preserves semantics. Specifically:
- Define a denotational semantics ⟦·⟧ for the obligation DSL (what does each obligation "mean" in terms of compliance strategy sets)
- Define the standard semantics of SMT formulas
- Prove: for all well-typed obligations O, ⟦O⟧ = Solutions(τ(O)) — the set of compliant strategies under the DSL semantics equals the set of satisfying assignments of the compiled formula

This is a genuine verification challenge. The compilation involves: type erasure, constraint linearization (obligations with non-linear conditions), Boolean flattening (disjunctive obligations), and optimization of the constraint representation. Each transformation must preserve semantics, and the proof must compose.

**H3. Modular, Multi-Backend Compilation (Architectural novelty)**
The verified compiler targets multiple backends: ILP (Gurobi), MaxSMT (Z3/CVC5), and Answer Set Programming (Clingo). Each backend has different expressiveness: ILP handles linear constraints efficiently; MaxSMT handles arbitrary Boolean structure; ASP handles defaults and preferences naturally. The compilation must be *modular*: a common IR with backend-specific lowering, each lowering separately verified.

**H4. Conflict Detection via Type Inhabitation (Novel algorithm)**
Regulatory conflicts are detected by checking whether the conjunction type O₁ ⊗ O₂ is inhabited — whether any compliance strategy satisfies both obligations simultaneously. For the type-theoretic fragment (obligations expressible as types), this is decidable and fast (type-checking, not SAT solving). For the full constraint fragment, inhabitation reduces to SAT/SMT solving. The algorithm: (a) type-level conflict pre-filter catches "obvious" conflicts in O(n²) time; (b) constraint-level solver handles the remainder. The pre-filter dramatically reduces solver load.

**H5. MUS Extraction + Certificates (Same as Approaches A/B)**
MUS enumeration, provenance mapping, and certificate generation — same technical challenge, but certificates now include an additional layer: a reference to the compilation verification proof.

**H6. Lean 4 / Coq Formalization (Verification engineering)**
The compilation correctness proof must be mechanized — not just a paper proof. This requires: (a) formalizing the obligation DSL's syntax and semantics in Lean 4; (b) formalizing the SMT formula syntax and semantics; (c) implementing the compilation in Lean 4's programming language; (d) proving the correctness theorem. Estimated: 5,000–8,000 lines of Lean 4 proof code for the core compilation; additional proofs for each backend. This is the single largest risk — mechanized proofs are notoriously time-consuming.

**Architectural Challenge:** The system has two "languages": the DSL implemented in Rust for production use, and the formalized DSL in Lean 4 for verification. These must stay synchronized. Any change to the Rust DSL requires a corresponding update to the Lean 4 formalization and re-verification. This is the "verified-compiler maintenance problem" and it's a known pain point.

### 3. New Math Required (Load-Bearing Only)

**M1. Obligation Type Theory Metatheory (Grade: A- — Crown Jewel)**
*Statement:* Define ObligationType, a type theory with: (a) base types for obligations indexed by jurisdiction and temporal interval; (b) type constructors for conjunction (⊗), disjunction (⊕), override (▷), exception (⊘); (c) a subtyping relation capturing obligation refinement (a stronger obligation is a subtype of a weaker one); (d) a type inhabitation judgment capturing compliance. Prove:
- **Decidability:** Type-checking and inhabitation checking are decidable (termination of the checking algorithm)
- **Soundness:** If ⊢ σ : O (strategy σ has type O), then σ genuinely satisfies obligation O under the denotational semantics
- **Completeness for the decidable fragment:** If a strategy satisfies O and O is fully decidable, then ⊢ σ : O is derivable

*Why load-bearing:* The type theory provides the semantic foundation for the entire system. Soundness ensures that type-checking-based conflict detection never produces false negatives. Decidability ensures the pre-filter terminates. Completeness (for the decidable fragment) ensures no false positives.

**M2. Verified Compilation Correctness (Grade: B+)**
*Statement:* The compilation function τ : ObligationType → SMTFormula preserves semantics: for all well-typed O, ⟦O⟧ = Solutions(τ(O)). The proof is mechanized in Lean 4.

*Why load-bearing:* This is what distinguishes Approach C from A/B. Without verification, the compilation is "tested but not proven correct." With it, the system provides end-to-end formal guarantees: if the obligations are correctly expressed (human responsibility), the constraint encoding is *provably* correct (machine-verified), and the solver results are valid (solver correctness).

**M3. Type-Level Conflict Decidability Characterization (Grade: B)**
Characterize the fragment of obligations for which conflict detection (inhabitation of O₁ ⊗ O₂) is decidable in polynomial time via type-checking alone, without invoking the SAT solver. Prove: for obligations in the "linear" fragment (no disjunction, no exception), conflict detection is in P. For the full fragment including disjunction, inhabitation is NP-complete (reduces from SAT).

*Why load-bearing:* Determines the boundary of the fast pre-filter. If the linear fragment covers 60–70% of real-world obligations (as we hypothesize), the type-level pre-filter handles most conflict detection cheaply.

**M4. Modular Backend Correctness (Grade: C+)**
Each backend lowering (IR → ILP, IR → MaxSMT, IR → ASP) is independently verified. The modular proof structure: prove each lowering correct, then compose to get end-to-end correctness for any backend.

**M5. Obligation Subtyping Soundness (Grade: C+)**
The subtyping relation O₁ <: O₂ (O₁ is a stronger obligation) is sound with respect to the compliance semantics: if σ satisfies O₁ and O₁ <: O₂, then σ satisfies O₂. This enables obligation refinement during regulatory amendments.

### 4. Best-Paper Argument

**Why this has best-paper potential:**

1. **Verified compilation is deep.** Unlike Approaches A/B where the encoding is "tested," Approach C provides a machine-checked proof that the encoding is correct. This is a qualitative jump in assurance that PL and FM communities will recognize.

2. **Novel type theory for regulatory obligations.** The dependent type system indexed by jurisdiction and temporal interval, with a decidability characterization, is a genuine PL contribution that stands independent of the compliance application.

3. **Bridges PL and governance.** This is exactly the kind of interdisciplinary work that top venues celebrate — deep PL theory applied to a timely, high-stakes domain.

4. **The trust argument is compelling.** "Why should you trust a compliance tool?" is a question that resonates with every audience. "Because we proved the compiler correct" is a powerful answer.

5. **PLDI/OOPSLA venue fit.** Unlike Approaches A/B (ICSE/FAccT), Approach C targets the PL community directly. A verified compiler for a novel domain with practical relevance is PLDI bread-and-butter.

**Weakness:** The Lean 4 verification is extremely high risk. Mechanized proofs take 3–5× longer than estimated. If the verification falls behind, the entire crown jewel is lost. The Approach falls back to Approach A (unverified but tested encoding).

### 5. Hardest Technical Challenge

**Mechanizing the compilation correctness proof in Lean 4.**

Formalizing the obligation DSL's semantics, the SMT formula semantics, and proving their equivalence under compilation is a substantial verification engineering effort. The specific challenges:

1. **Semantic formalization of regulatory obligations.** What does "an organization must ensure appropriate human oversight" *mean* formally? The denotational semantics must be rich enough to capture real regulatory intent but simple enough to prove properties about.

2. **Handling partiality.** The compilation is partial — some obligations are Opaque and cannot be compiled. The correctness theorem must be qualified: "for all *compilable* obligations, the encoding is correct." Formalizing "compilable" and proving the qualification is tricky.

3. **Proof engineering at scale.** 5,000–8,000 lines of Lean 4 for the core proof, plus backend-specific proofs. This is a 4–6 month effort for an experienced Lean developer. If the proof encounters unexpected difficulties (as they always do), it can balloon to 12+ months.

**Mitigation:** 
- Start with the simplest sub-language (linear, no disjunction, no temporal) and verify that. Extend incrementally.
- Use Lean 4's tactic automation (simp, omega, decide) aggressively to reduce proof burden.
- Kill gate: if the core proof is not complete by week 12, abandon verification and fall back to Approach A's tested-but-unverified encoding.
- Backup: even an incomplete verification (e.g., verified for the linear fragment only) is publishable as "partial verification with characterization of the verified fragment."

### 6. Scores

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| **Value** | 7/10 | Verified trust model is compelling for high-assurance environments; slightly less immediate commercial value than A/B |
| **Difficulty** | 9/10 | Verified compilation in Lean 4 + novel type theory + multi-backend = extremely challenging |
| **Potential** | 9/10 | If pulled off, the verified regulatory compiler is a landmark contribution |
| **Feasibility** | 4/10 | Lean 4 verification is extremely high risk; realistic P(completion) ≈ 40–50% |

### 7. LoC Breakdown

| Component | LoC | Language | Novel? |
|-----------|-----|----------|--------|
| Dependently-Typed DSL (lexer, parser, type checker, elaborator) | 18,000 | Rust | Yes — dependent obligation types |
| Lean 4 Formalization & Proofs | 7,000 | Lean 4 | Yes — mechanized compilation proofs |
| Regulatory Data Corpus (5 frameworks) | 15,000 | Custom DSL | Domain data |
| Verified Constraint Compiler (obligation→IR→backends) | 12,000 | Rust | Yes — multi-backend compilation |
| Solver Backend (Z3/CVC5/Gurobi/Clingo) | 10,000 | Rust | Partially novel — ASP backend is new |
| MUS Extraction + Certificates | 5,500 | Rust | Yes |
| Evaluation Framework | 16,000 | Python + Rust | Standard |
| CLI / Reporting | 4,000 | Rust + HTML | Standard |
| Infrastructure | 5,500 | Rust | Standard |
| **TOTAL** | **~93,000** | | |
| **Novel algorithmic core** | **~52,500** | | 56% of total |

### 8. Key Risks & Kill Gates

| Gate | Week | Condition | Fail Action |
|------|------|-----------|-------------|
| G1: Type theory metatheory | 4 | Decidability proof sketch for linear fragment | Simplify to non-dependent types |
| G2: Lean 4 core formalization | 8 | DSL syntax + semantics formalized in Lean 4 | Abandon verification; fall back to tested encoding |
| G3: Compilation correctness proof | 12 | Lean 4 proof of τ correctness for linear fragment | Scope verification to linear fragment only |
| G4: MUS + certificates | 14 | Working conflict detection on 100-obligation instance | Standard Approach A fallback |
| G5: Multi-backend | 16 | ILP + MaxSMT backends both working | Drop ASP backend |
| G6: External validation | 18 | Expert agreement ≥80% on encoding sample | Encoding methodology revision |

---

## Cross-Approach Comparison

| Dimension | A: ConflictProver | B: RegSynth-Complete | C: RegVerify |
|-----------|-------------------|---------------------|--------------|
| **Crown Jewel** | First formal conflict prover for multi-jurisdictional AI regulation | Formalizability completeness theorem | Verified regulatory constraint compiler |
| **Value** | 8 | 8 | 7 |
| **Difficulty** | 6 | 8 | 9 |
| **Potential** | 6 | 8 | 9 |
| **Feasibility** | 9 | 6 | 4 |
| **Composite** | **7.25** | **7.50** | **7.25** |
| **Novel LoC** | ~31K | ~55K | ~52K |
| **Total LoC** | ~69K | ~115K | ~93K |
| **Kill Prob** | ~15% | ~35% | ~55% |
| **P(best-paper)** | ~4% | ~8% | ~12% if completed |
| **P(any pub)** | ~80% | ~65% | ~45% |
| **Primary Venue** | ICSE Tools | ICSE Tools / FAccT | PLDI / OOPSLA |
| **Math Depth** | Moderate (B) | Moderate-High (B+) | High (A-) |
| **Risk Profile** | Low risk, moderate reward | Moderate risk, high reward | High risk, very high reward |
