# RegSynth: Formal Tractability of Multi-Jurisdictional AI Regulatory Compliance via Infeasibility Detection, Pareto Synthesis, and Temporal Trajectory Optimization

**Slug:** `pareto-reg-trajectory-synth`

---

## 1. Problem Statement and Approach

Organizations deploying AI systems across jurisdictional boundaries face a combinatorial explosion of regulatory obligations that no human compliance team can tractably navigate. The EU AI Act (Regulation 2024/1689, enforcement beginning August 2026 with €35M penalties), the NIST AI Risk Management Framework, ISO/IEC 42001, China’s Interim Measures for Generative AI, and GDPR’s AI-relevant provisions collectively impose approximately 300 machine-encodable obligations spanning risk classification, transparency, documentation, human oversight, data governance, post-market surveillance, and cross-border data transfer. These obligations frequently conflict: the EU AI Act mandates extensive logging for high-risk systems while GDPR requires data minimization; China requires algorithmic disclosure while US trade secret law protects proprietary methods; the EU’s precautionary approach clashes with lighter-touch voluntary frameworks. A multinational deploying a single AI system across these five jurisdictions faces thousands of pairwise obligation interactions, many of which are jointly infeasible — and no existing tool can detect, prove, or resolve these conflicts formally.

Current approaches to this problem are fundamentally inadequate. GRC platforms (ServiceNow, AuditBoard, OneTrust) provide checklist workflows and audit trails but cannot reason about regulatory interactions or detect cross-jurisdictional conflicts. The emerging wave of LLM-powered compliance assistants generates natural-language guidance but cannot provide formal guarantees, cannot prove infeasibility, and cannot compute optimal trade-offs. Academic work on deontic logic and normative systems has formalized individual regulatory frameworks but has never addressed the multi-jurisdictional tractability problem. No existing tool answers the most basic question a multinational compliance officer faces: *"Are these regulatory obligations simultaneously satisfiable — and if not, which specific articles across which jurisdictions are in irreducible conflict?"*

We propose **RegSynth**, a constraint-solving engine that makes multi-jurisdictional AI regulatory compliance formally tractable. The system delivers three contributions in a layered architecture:

**Primary contribution: Formal infeasibility detection with MUS-mapped regulatory diagnoses.** RegSynth formalizes multi-jurisdictional AI regulations in a typed regulatory Domain-Specific Language (DSL) and encodes them as constraint systems. When simultaneous full compliance is provably infeasible, RegSynth extracts minimal unsatisfiable subsets (MUS) of conflicting obligations and maps them back through a provenance-preserving inverse translation to produce human-interpretable regulatory diagnoses: specific conflicting articles across specific jurisdictions, with machine-checkable certificates of infeasibility. This is the first system capable of formally proving that regulatory conflict exists, identifying its irreducible core, and certifying the proof.

**Secondary contribution: Pareto-optimal compliance strategy synthesis.** When feasible strategies exist, RegSynth computes the Pareto frontier over a 4-dimensional cost vector (implementation cost, time-to-compliance, residual regulatory risk, ongoing operational burden) via iterative weighted partial MaxSMT, enabling decision-makers to see the full trade-off space rather than picking a single dominated strategy. An ILP (Gurobi) baseline provides the foundational optimization layer; the MaxSMT formulation extends this with richer logical constraint handling as the research delta.

**Research extension: Temporal trajectory optimization.** Regulations phase in over multi-year timelines (the EU AI Act alone has four enforcement milestones between February 2025 and August 2027), and organizations have bounded implementation budgets per quarter. RegSynth models compliance as a temporal trajectory optimization problem, computing Pareto-optimal compliance *paths* through a time-varying regulatory constraint landscape. We prove that the naïve approach — independently optimizing compliance at each timestep — can produce trajectories that are Pareto-dominated by the trajectory-aware solution.

The system is structured as a pipeline: (1) a typed regulatory DSL with jurisdiction-indexed obligation types, temporal annotations, and explicit formalizability grading — itself a PL contribution; (2) a compositional encoding function translating typed obligations to quantifier-free SMT formulas over linear arithmetic, arrays, and uninterpreted functions; (3) an ILP baseline encoding for foundational constraint solving; (4) a temporal constraint unroller that produces time-indexed MaxSMT instances with inter-step transition budget constraints; (5) a Pareto synthesis engine that computes ε-approximate Pareto frontiers via iterative MaxSMT with lexicographic refinement; (6) a remediation roadmap planner that transforms Pareto-optimal compliance strategies into scheduled implementation plans with resource constraints; and (7) a certificate generator producing machine-verifiable proofs of compliance, infeasibility, and Pareto optimality relative to the formalizable fragment. The entire system runs on a laptop CPU, with Z3, CVC5, and Gurobi as backend solvers, targeting solution of 5-jurisdiction, 300-obligation instances in under 15 minutes (core scope), with a stretch target of 10 jurisdictions in under 30 minutes.

The central intellectual contribution is *making AI regulatory compliance formally tractable*: demonstrating that the regulatory DSL can capture a meaningful fragment of real-world obligations (with honest formalizability grading), that infeasibility can be formally detected and diagnosed, that the formalizable fragment admits a completeness theorem (M5), and that Pareto-optimal strategies can be synthesized and certified. The temporal trajectory extension demonstrates where static optimization is provably insufficient, but the paper stands on the tractability result even without it.

---

## 2. Value Proposition

### Who Needs This

**Compliance officers and AI governance teams at multinational technology companies** operating AI systems across jurisdictional boundaries. This includes every Fortune 500 technology company, every major financial institution deploying AI for risk assessment, every healthcare AI company operating in both the EU and US, and every autonomous vehicle company with multi-jurisdictional deployment. The addressable market encompasses thousands of organizations, each spending $1–10M annually on AI compliance.

**Regulatory bodies and standards organizations** (EU AI Office, NIST, ISO/IEC JTC 1/SC 42) need tools to analyze whether their regulations create unintended conflicts with other jurisdictions’ frameworks. Currently, regulatory impact assessment is done by lawyers reading other jurisdictions’ laws — no systematic formal analysis exists.

**AI auditing firms** (Holistic AI, Credo AI, ForHumanity) need to provide clients with defensible compliance assessments that go beyond checklists to demonstrate optimality of chosen compliance strategies.

### Why Desperately

The EU AI Act enters full enforcement in August 2026. Organizations face fines up to €35M or 7% of global turnover. Simultaneously, multiple other major jurisdictions are implementing or have implemented AI-specific legislation. For the first time in regulatory history, companies face a situation where:

1. **Simultaneous compliance may be provably impossible**: When Jurisdiction A mandates logging everything and Jurisdiction B mandates deleting everything, there exists no compliant configuration. Currently, companies discover this through painful trial and error with regulators, often after investing millions in a compliance strategy that cannot work. No existing tool can *prove* this infeasibility or identify which specific articles conflict.

2. **The strategy space is exponentially large**: With 300+ obligations and binary compliance choices for each, the strategy space is combinatorially explosive. Even accounting for constraint structure, the feasible region is too large for human enumeration.

3. **Compliance is temporally complex**: The EU AI Act alone has four phase-in dates. A strategy that is compliant today may become non-compliant in six months. Organizations need roadmaps, not snapshots.

4. **Cost differences between strategies are enormous**: Two strategies that both achieve 95% compliance coverage may differ by 10× in implementation cost. Without optimization, organizations routinely choose dominated strategies.

### What Becomes Possible

- **Formal infeasibility proofs with regulatory diagnoses**: For the first time, an organization can formally prove to a regulator that simultaneous compliance with jurisdictions A and B is logically impossible, accompanied by a minimal unsatisfiable core identifying the exact conflicting articles — not just “it’s infeasible” but “these three specific articles across these two jurisdictions are in irreducible conflict, and removing any one resolves it.”
- **Pareto-optimal strategy selection**: Decision-makers see the full trade-off frontier between cost, speed, risk, and coverage, enabling informed rather than guesswork-based compliance decisions.
- **Minimum-cost remediation roadmaps**: Instead of “implement everything now,” organizations get phased implementation plans that respect organizational transition budgets and regulatory phase-in schedules.
- **Continuous re-optimization**: When a regulation is amended, the Pareto frontier is incrementally recomputed rather than compliance strategies being rebuilt from scratch.

---

## 3. Technical Difficulty

### The Hard Subproblems

Building RegSynth requires solving at least 8 genuinely difficult technical problems, each of which individually constitutes a publishable contribution:

**H1. Regulatory DSL Design with Formalizability Grading (Research-grade PL design)**
Designing a typed DSL that can express the full range of regulatory obligations — temporal phasing, jurisdictional scoping, conditional triggers, cross-references, exemptions — while honestly tracking what fraction of each obligation has been faithfully formalized. The type system includes obligation types (OBL/PERM/PROH) indexed by jurisdiction and temporal interval, a composition algebra with conjunction (⊗), disjunction (⊕), jurisdictional override (▷), and exception (⊘) operators satisfying algebraic laws, and a formalizability grade (Full, Partial(α), Opaque) that propagates uncertainty through a confidence algebra. No existing DSL handles this combination. The DSL itself, with its formalizability grading system, is a PL contribution independent of the solver backend.

**H2. Faithful Encoding of Regulatory Articles Across Jurisdictions (Domain engineering at scale)**
Translating real regulatory text into the typed DSL while maintaining semantic fidelity. This is not boilerplate — each article requires careful analysis of its logical structure, identification of hard vs. soft constraints, assignment of temporal bounds, and mapping to the cross-jurisdictional ontology. The EU AI Act alone has 113 articles and 13 annexes; NIST RMF has 4 functions, 19 categories, and 72 subcategories. Each framework uses different terminology, different risk taxonomies, and different enforcement models. Core scope: 5 frameworks, ~300 obligations; stretch scope: 10 frameworks, ~500 obligations.

**H3. Compositional Obligation-to-Constraint Encoding (Compiler correctness)**
Building a translation function τ from the typed obligation algebra to quantifier-free SMT formulas that is provably sound (if τ(O) is satisfiable then O is genuinely satisfiable) and compositionally structured (τ(O₁ ⊗ O₂) = τ(O₁) ∧ τ(O₂)). Soundness is load-bearing: a wrong encoding produces wrong compliance verdicts with formal-looking but incorrect certificates.

**H4. Temporal Constraint Unrolling with Transition Budgets (Novel algorithm design)**
Producing time-indexed MaxSMT instances from regulatory transition systems where cross-timestep constraints enforce bounded organizational change per period. This is analogous to bounded model checking but with soft constraints, multi-objective optimization, and the novel requirement that strategy variables across timesteps satisfy implementability constraints (an organization cannot restructure its entire data governance in a single quarter).

**H5. Pareto Frontier Computation via Iterative MaxSMT (Novel algorithm)**
Computing ε-approximate Pareto frontiers over a 4-dimensional cost space using iterative MaxSMT queries. Each Pareto point requires a MaxSMT solve; the frontier may contain hundreds of non-dominated points. The algorithm must: (a) avoid enumerating dominated points, (b) guarantee ε-approximation coverage of the true frontier, (c) terminate in bounded time on a laptop CPU. Existing multi-objective optimization literature uses evolutionary algorithms or ILP; adapting Pareto enumeration to MaxSMT with regulatory constraint structure is novel.

**H6. Temporal Pareto Trajectory Optimization (Research extension — novel formulation)**
Computing Pareto-optimal compliance *paths* through a time-varying constraint landscape with bounded transition budgets. This is fundamentally harder than static Pareto computation because: (a) the constraint set changes at each timestep, (b) inter-step transition costs create coupling between timesteps, (c) the trajectory space is exponential in the time horizon. We prove that naïve per-timestep optimization can produce dominated trajectories and develop an algorithm that finds globally Pareto-optimal trajectories via temporal MaxSMT unrolling.

**H7. Conflict Core Extraction and Regulatory Diagnosis (Primary contribution — actionable formal methods)**
When hard constraints are unsatisfiable, extracting minimal unsatisfiable subsets (MUS) and mapping them back through τ⁻¹ to produce human-interpretable regulatory diagnoses: “Articles 6(1)(a) of the EU AI Act, Section 2.3.4 of NIST RMF, and Clause 5.2 of ISO 42001 are mutually unsatisfiable. Removing any one resolves the conflict.” The inverse mapping τ⁻¹ from SMT variables back to regulatory article numbers requires maintaining full provenance through the compilation pipeline. This is the headline capability of the system.

**H8. Machine-Checkable Compliance Certificates (Proof-carrying compliance)**
Generating proof witnesses for three verdict types: compliance certificates (strategy σ satisfies all hard constraints — witness is a satisfying assignment), infeasibility certificates (no strategy satisfies the hard constraints — witness is a resolution proof extracted from the UNSAT core), and Pareto optimality certificates (no feasible strategy dominates the proposed one on all cost dimensions — witness is a sequence of bounded MaxSMT proofs). Certificates must be independently verifiable by a standalone verifier under 2,500 LoC.

### Subsystem Breakdown Totaling ~115K LoC

| Component | LoC | Language | Key Challenge |
|-----------|-----|----------|---------------|
| **A. DSL & Type System** (lexer, parser, 3-stage IR, type checker, elaborator, error reporting, pretty printer) | 15,200 | Rust | Novel obligation type system with jurisdiction indexing, temporal sorts, formalizability grading |
| **B. Regulatory Data Corpus** (cross-jurisdictional ontology, 5 core + 5 stretch framework encodings, cross-jurisdiction mappings, regulatory test oracles) | 18,000 | Custom DSL | Faithful encoding of ~300 core articles (stretch: ~500) with per-article regression tests; cross-jurisdiction concept alignment ontology |
| **C. Temporal Model** (regulatory transition system, version lattice, phase-in operators, temporal diff engine) | 6,500 | Rust | Correct composition of concurrent multi-jurisdictional amendments; exploiting temporal structure for incremental solving |
| **D. Constraint Encoding Engine** (obligation-to-SMT translation, ILP encoding, hard/soft classification, temporal unrolling, jurisdiction weighting) | 11,200 | Rust | Provably sound encoding; dual ILP/SMT backends; compositional translation preserving algebraic structure; temporal unrolling with transition budgets |
| **E. Solver Backend** (Z3/CVC5/Gurobi integration, MaxSMT interface, ILP interface, Pareto enumeration, conflict core extraction, incremental solving) | 12,500 | Rust | ILP baseline + MaxSMT research delta; Pareto frontier computation; MUS extraction with provenance; multi-solver abstraction |
| **F. Pareto Synthesis Engine** (multi-objective cost model, trajectory optimizer, strategy representation, dominance checking) | 7,500 | Rust | Temporal Pareto trajectory optimization; ILP vs MaxSMT comparison on same problem instances |
| **G. Remediation Planner** (schedule optimizer, dependency tracker, resource allocator, roadmap generator) | 6,500 | Rust | RCPSP with regulatory deadline constraints; Pareto trajectory to actionable phased plan |
| **H. Certificate Generator** (compliance/infeasibility/Pareto proofs, standalone verifier) | 6,000 | Rust | Three certificate types; standalone verifier <2,500 LoC; proof extraction from solver traces |
| **I. Evaluation Framework** (benchmark generator, metric suite, solver comparison, ablation harness, regression oracle) | 20,000 | Python + Rust | Planted-solution synthetic benchmarks at 1000+ instances; ILP vs MaxSMT evaluation; automated end-to-end pipeline |
| **J. CLI / API / Reporting** (CLI interface, report generator, visualization, Pareto front explorer) | 5,500 | Rust + HTML/SVG | Pareto front visualization in 3+ dimensions; regulatory-domain report templates |
| **K. Infrastructure** (error handling, logging, config, build system, integration tests) | 6,100 | Rust + TOML/YAML | End-to-end integration tests covering full pipeline; cross-solver regression tests |
| **TOTAL** | **~115,000** | | |

*Component breakdown: 15,200 + 18,000 + 6,500 + 11,200 + 12,500 + 7,500 + 6,500 + 6,000 + 20,000 + 5,500 + 6,100 = 115,000.*

**By category:** ~55K novel algorithmic core (Components A, C, D, E, F, G, H) + ~18K regulatory data corpus (Component B) + ~20K evaluation framework (Component I) + ~12K infrastructure and tooling (Components J, K) + ~10K DSL data portion within B.

**The novel research contribution is ~55K LoC.** The regulatory data corpus (~28K including DSL encodings and test oracles) is essential domain engineering but is not algorithmic novelty. The evaluation framework (~20K) and infrastructure (~12K) are necessary for a complete artifact but employ standard techniques.

**By language:** ~72K Rust, ~28K Custom DSL (regulatory encodings), ~12K Python, ~2K HTML/SVG, ~1K config/build.

**Novel vs. standard:** ~55K genuinely novel algorithmic core (48%), ~28K domain-specific regulatory data (24%), ~20K evaluation infrastructure (17%), ~12K standard infrastructure (11%).

---

## 4. New Mathematics Required

### Genuinely New Theory (6 contributions)

**M1. Temporal Pareto Trajectory Optimization (MOTIVATING FORMALIZATION — Grade: B-)**

*Formulation.* Given a regulatory transition system R = (S, s₀, Σ, →) where states are sets of active typed obligations, a 4-dimensional cost function C(σ,s) over compliance strategies σ in regulatory state s, and a transition budget function B(t) bounding organizational change between consecutive timesteps, find the set of Pareto-optimal trajectories (σ₀, σ₁, ..., σ_T) such that: (a) each σ_t satisfies all hard constraints active at timestep t, (b) transition cost Δ(σ_t, σ_{t+1}) ≤ B(t) for all t, and (c) no other feasible trajectory Pareto-dominates the total cost vector Σ_t C(σ_t, s_t).

*Key theorem.* Per-timestep Pareto optimality does not imply trajectory Pareto optimality: there exist regulatory instances where independently optimizing each timestep yields trajectories strictly dominated by trajectory-aware solutions. Proof is constructive, exhibiting a 3-timestep, 2-jurisdiction instance.

*Algorithm.* ε-approximate Pareto enumeration over temporally-unrolled MaxSMT instances with iterative exclusion constraints and transition budget coupling. Termination and ε-completeness proofs.

*Role in paper.* This is the research extension that motivates the temporal MaxSMT approach over static ILP. It demonstrates the theoretical ceiling of per-timestep optimization but is not the primary contribution. The paper stands without this result if the ILP baseline proves sufficient for practical instances.

**M2. Pareto Frontier Computation via Iterative Weighted Partial MaxSMT (Grade: B)**

Extension of standard MaxSMT optimization to multi-objective settings via iterative ε-constraint scalarization with blocking clauses. For each objective dimension d, solve MaxSMT with dimension d as the objective and other dimensions as ε-bounded constraints, then add a blocking clause excluding the found point’s dominance cone. Prove: (a) each returned point is Pareto-optimal, (b) the algorithm terminates in finitely many iterations, (c) the returned set is an ε-cover of the true Pareto frontier.

**M3. Compliance-Preserving Bisimulation over SMT-Encoded Strategy Spaces (Grade: B-)**

A bisimulation relation ≈_C on regulatory transition system states such that s₁ ≈_C s₂ iff Compliant(s₁) = Compliant(s₂) — the set of satisfying strategy assignments is identical. Adaptation of partition refinement to work over SMT-definable strategy spaces rather than finite-state spaces. Termination proof for the regulatory domain (finite obligation set implies finite partition).

*Why load-bearing.* Enables the incremental Pareto maintenance algorithm (M4) to skip recomputation when regulatory amendments are compliance-preserving.

**M4. Incremental Pareto Frontier Maintenance Under Regulatory Transitions (Grade: B-)**

Given a Pareto frontier F for regulatory state s and a transition s →^σ s’, compute the new frontier F’ without full recomputation. Algorithm: (a) test each point in F against new constraints — surviving points remain Pareto-optimal, (b) solve MaxSMT for new constraints only in the region not covered by survivors, (c) merge and filter dominated points. Prove correctness: F’ is an ε-cover of the true frontier for s’.

**M5. Formalizability Completeness Theorem (Grade: B+)**

Define a formalizability grade γ ∈ {Full, Partial(α), Opaque} for each obligation, where α ∈ [0,1] is confidence that the formal encoding captures legal intent. Prove: if all Full and Partial(α ≥ threshold) obligations are correctly encoded, then the computed Pareto frontier is an ε-approximation of the true frontier restricted to the formalizable fragment. The Opaque obligations are tracked but excluded from optimization, with their impact on the cost vector bounded by a worst-case multiplier. This is the formal treatment of the 60–70% formalizability ceiling and is the key theoretical result that makes the entire approach intellectually honest: rather than pretending all regulations are formalizable, it proves that the system’s outputs are sound and complete *relative to the formalizable fragment*, with explicit, bounded uncertainty for the rest.

*Why elevated.* This theorem is what separates RegSynth from naïve formalization: it provides the formal guarantee that partial formalization still yields useful results, with quantified uncertainty. Without it, any critic can dismiss the system by pointing to a single vague regulation. With it, vague regulations are explicitly tracked, bounded, and excluded from the completeness guarantee.

**M6. Obligation Composition Algebra Soundness (Grade: C+)**

Prove that the translation function τ from the typed obligation algebra to SMT formulas preserves the algebraic structure: τ is a homomorphism from (Obligations, ⊗, ⊕, ▷, ⊘) to (SMTFormulas, ∧, ∨, priority-ite, except-ite). Soundness: if the SMT encoding is satisfiable, then the obligation set is genuinely satisfiable (no false positives in compliance verdicts). Partial completeness: if the obligation set is satisfiable in the formalizable fragment, the encoding is satisfiable (no false negatives for fully-formalized obligations).

### New Applications of Existing Theory (12 contributions)

- **Deontic obligation type system** (A1): Standard Deontic Logic as types via Curry-Howard — known idea, new domain
- **Jurisdiction-indexed type families** (A2): Parameterized types indexed by jurisdiction lattice
- **Temporal obligation sorts** (A3): LTL-inspired temporal annotations on obligations
- **Formalizability grading system** (A4): Fuzzy/probabilistic typing for encoding confidence
- **Regulatory transition system** (B1): Labeled transition system instantiated for regulatory evolution
- **Phase-in/sunset temporal operators** (B3): Domain-specific LTL operators for regulatory lifecycle
- **Regulatory version lattice** (B4): Version control theory applied to concurrent regulatory amendments
- **Obligation-to-constraint encoding** (C1): Compiler-style translation to SMT — standard technique, novel domain
- **Temporal constraint unrolling** (C3): BMC-style unrolling with strategy continuity constraints
- **Conflict core extraction** (C5): MUS extraction with regulatory provenance mapping
- **Multi-objective compliance cost model** (D1): 4D cost vector for regulatory compliance
- **Synthetic regulatory benchmark generator** (F6): Planted-solution methodology for ground-truth evaluation

### Adapted Standard Techniques (6 items)

- Weighted partial MaxSMT solving via Z3/CVC5
- Multi-objective ILP solving via Gurobi (baseline)
- Jurisdiction priority weighting scheme
- MUS/UNSAT core algorithms
- Hypervolume and spread metrics for Pareto quality
- Statistical significance testing for evaluation

---

## 5. Best Paper Argument

### Why a Committee Would Select This

**1. It establishes the first formal tractability result for multi-jurisdictional AI regulation.** No existing academic work or commercial tool can formally prove that a set of regulatory obligations across jurisdictions is infeasible, identify the minimal conflicting core, and certify the proof. RegSynth demonstrates that a meaningful fragment of real-world AI regulation is formally tractable — that obligations can be faithfully encoded, conflicts can be detected and diagnosed, and optimal strategies can be synthesized with machine-checkable certificates. This is a foundational result that changes how the field thinks about regulatory compliance: from an inherently informal, lawyer-dependent process to one amenable to formal methods with quantified completeness guarantees.

**2. The timing is perfect.** The EU AI Act enters full enforcement in August 2026. Every major technology company is currently scrambling to build compliance programs. The paper arrives at exactly the moment when the problem transitions from theoretical to urgent. A best paper that speaks to an immediate, trillion-dollar industry problem has enormous impact.

**3. It bridges three communities with a genuine PL contribution.** RegSynth connects formal methods (SMT solving, type theory), operations research (multi-objective optimization, Pareto frontiers), and AI governance (regulatory compliance, responsible AI) in a way that creates genuine cross-pollination. The regulatory DSL with its formalizability grading system is an independent PL contribution: a type system that honestly tracks the boundary between formal and informal knowledge, with a completeness theorem (M5) that makes partial formalization rigorous.

**4. The contribution triad is clean and layered.** The paper delivers three progressively deeper contributions: (i) formal infeasibility detection with MUS-mapped regulatory diagnoses — immediately useful, no prior art; (ii) Pareto-optimal compliance synthesis over the formalizable fragment with completeness guarantees; (iii) temporal trajectory optimization proving that static approaches are theoretically suboptimal. Each layer stands independently, and each is a “first” in its own right.

**5. The artifact is substantial and honest.** At ~115K total LoC (~55K novel algorithmic core), RegSynth is not a proof-of-concept but a complete, usable system with an honestly accounted codebase. It covers the full pipeline from regulatory text encoding to Pareto frontier computation to remediation roadmap generation to machine-checkable certificates. The evaluation includes external validation of regulatory encodings by domain experts, ILP-vs-MaxSMT comparison as a primary axis, and comprehensive automated benchmarks.

**6. It is the first of its kind.** No existing academic work or commercial tool computes Pareto-optimal multi-jurisdictional compliance strategies. No existing work provides machine-checkable certificates of compliance or infeasibility across jurisdictions. No existing work offers a formal completeness theorem for partial regulatory formalization. RegSynth creates a new subfield at the intersection of formal methods and AI governance.

### Target Venue

Primary: **ICSE 2027 (Tools Track)**. RegSynth is fundamentally a novel software tool with formal properties — a constraint-solving engine backed by rigorous theory. The ICSE Tools Track values exactly this: substantial, well-evaluated tools that advance the state of the practice with formal underpinnings. The regulatory DSL, the multi-solver architecture (ILP + MaxSMT), the certificate generator, and the end-to-end pipeline are all software engineering contributions that ICSE reviewers are best positioned to evaluate.

Secondary: **FAccT 2027**. The AI governance domain contribution — formal tractability of regulatory compliance, infeasibility detection, and the formalizability completeness theorem — speaks directly to the FAccT community’s interest in accountability, transparency, and formal guarantees for AI systems. FAccT values interdisciplinary work that makes governance problems technically rigorous.

---

## 6. Evaluation Plan

All automated evaluation requires zero human involvement. External encoding validation (Section 6.7) is the sole human-in-the-loop evaluation component.

### 6.1 Synthetic Regulatory Benchmarks (Ground Truth)

A parameterized benchmark generator produces synthetic regulatory instances with known properties:

- **Controlled parameters**: jurisdiction count (2–10), obligations per jurisdiction (20–200), conflict density (0–50% of pairwise interactions), temporal complexity (1–20 timesteps), formalizability ratio (0.5–1.0), cost heterogeneity.
- **Planted-solution methodology**: Generate a known Pareto frontier first, then construct obligations that produce it. This provides ground truth for measuring precision and recall of frontier computation.
- **Scale**: 1,000+ benchmark instances across the parameter space.
- **Metrics against ground truth**: Pareto frontier hypervolume ratio (computed/true), generational distance (average distance from computed to true frontier), spread metric (coverage uniformity), inverted generational distance.

### 6.2 Real Regulatory Corpora

Encode regulatory articles from jurisdictions in the RegSynth DSL. Five frameworks are **core scope** (required for paper); five additional frameworks are **stretch scope** (included if timeline permits):

| Framework | Articles/Requirements | Type | Scope |
|-----------|----------------------|------|-------|
| EU AI Act (Reg. 2024/1689) | ~113 articles, 13 annexes | Binding regulation | **Core** |
| NIST AI RMF 1.0 | 72 subcategories | Voluntary framework | **Core** |
| ISO/IEC 42001:2023 | ~90 clauses | International standard | **Core** |
| China Interim Measures for GenAI | ~30 articles | Binding regulation | **Core** |
| GDPR (AI-relevant) | ~40 articles | Binding regulation | **Core** |
| Singapore AIGA | ~30 guidelines | Voluntary framework | Stretch |
| UK AI Framework | ~45 principles | Voluntary framework | Stretch |
| South Korea AI Act | ~40 articles | Binding regulation | Stretch |
| Canada AIDA | ~35 sections | Proposed legislation | Stretch |
| Brazil LGPD (AI provisions) | ~25 articles | Binding regulation | Stretch |

**Core scope total:** 5 frameworks, ~300 obligations. **Stretch scope total:** 10 frameworks, ~500 obligations.

**Metrics on real corpora**: Number of cross-jurisdictional conflicts detected, MUS sizes, Pareto frontier sizes, solver runtime, strategy cost distributions, roadmap schedule lengths.

### 6.3 Baselines and Comparisons

The **ILP (Gurobi) vs. MaxSMT (Z3/CVC5) comparison** is the primary evaluation axis. The ILP encoding is not merely one baseline among many — it is the mandatory first implementation milestone, and the MaxSMT temporal trajectory approach is evaluated as the “research delta” built on top of a working ILP baseline.

| Baseline | Description | Expected Result |
|----------|-------------|-----------------|
| **ILP encoding (Gurobi) — PRIMARY BASELINE** | Encode the same multi-jurisdictional compliance problem as multi-objective ILP | ILP handles the static optimization problem well; MaxSMT provides advantages on deeply nested logical constraints, infeasibility diagnosis (MUS quality), and temporal trajectory formulation |
| **Single-objective MaxSMT** | Optimize one cost dimension, ignore others | Dominated by Pareto frontier on other dimensions |
| **Per-timestep independent Pareto** | Optimize each timestep separately | Dominated by trajectory-aware optimization (M1 theorem) |
| **Greedy compliance** | Greedily satisfy obligations by penalty weight | Feasible but dominated strategies |
| **Random feasible search** | Sample random feasible strategies | Far from Pareto frontier |
| **No formalizability grading** | Treat all obligations as fully formalizable | Overconfident certificates; silent errors on vague obligations |

**ILP vs. MaxSMT comparison metrics**: solution quality (Pareto hypervolume), solve time, scalability curves, MUS extraction quality (minimality, regulatory interpretability), expressiveness (which regulatory patterns require workarounds in ILP), and temporal trajectory capability.

### 6.4 Ablation Studies

Remove each major mathematical component one at a time and measure impact:
- Without temporal unrolling → per-step optimization (quantify dominance gap)
- Without formalizability grading → all-or-nothing formalization
- Without jurisdiction weighting → equal-weight soft constraints
- Without compliance bisimulation → full recomputation on every regulatory change
- Without certificate generation → uncertified verdicts
- Without MUS-based diagnosis → opaque “infeasible” verdicts

### 6.5 Scalability Evaluation

- Obligations: 50, 100, 200, 500 per jurisdiction
- Jurisdictions: 2, 3, 5, 10
- Timesteps: 1, 5, 10, 20
- Measure: solver wall-clock time, peak memory, Pareto frontier size, ε-approximation quality
- **Core target**: 5 jurisdictions × 300 obligations × 10 timesteps in <15 minutes on laptop CPU
- **Stretch target**: 10 jurisdictions × 500 obligations × 10 timesteps in <30 minutes on laptop CPU

**Honest caveat**: MaxSMT scaling is non-linear; temporal unrolling multiplies instance size. The 15-minute core target and 30-minute stretch target are engineering targets to be validated empirically, not theoretical guarantees. If these targets prove infeasible, the fallback is reduced temporal depth (fewer timesteps) or modular decomposition by regulatory domain to bring solve times within budget.

### 6.6 Cross-Solver Validation

Run identical benchmarks on Z3, CVC5, and Gurobi. Verify:
- Same conflicts detected (modulo solver-specific MUS extraction)
- Same Pareto frontier (within ε tolerance)
- Certificate cross-verification (Z3-generated certificates verified by CVC5-based verifier and vice versa)
- ILP vs. MaxSMT solution quality and runtime comparison across the full benchmark suite

### 6.7 External Encoding Validation

The fidelity of regulatory encodings is critical to the system’s real-world relevance. Automated evaluation cannot assess whether a DSL encoding faithfully captures legal intent. We therefore include a human evaluation component for encoding validation:

- **Expert panel**: 2–3 external regulatory domain experts (AI law specialists or regulatory consultants with jurisdiction-specific expertise) independently review a stratified sample of encodings.
- **Sample design**: ≥50 articles spanning ≥3 jurisdictions (at minimum: EU AI Act, NIST RMF, and one of ISO 42001 / China Interim Measures / GDPR), stratified by obligation type (OBL/PERM/PROH), formalizability grade (Full/Partial/Opaque), and regulatory domain (data governance, transparency, risk classification, etc.).
- **Evaluation criteria**: (a) Semantic fidelity — does the DSL encoding capture the legal intent of the article? (b) Completeness — are all material obligations in the article encoded? (c) Formalizability grade accuracy — is the assigned grade (Full/Partial/Opaque) appropriate? (d) Cross-jurisdictional mapping correctness — are equivalent concepts across jurisdictions correctly aligned?
- **Inter-annotator agreement**: Compute Cohen’s κ (pairwise) and Fleiss’ κ (all annotators) for each evaluation criterion. Target: κ ≥ 0.6 (substantial agreement) on semantic fidelity and formalizability grading.
- **Budget**: ~$15–30K for legal review (expert time at ~$300–500/hr, ~40–80 hours total across 2–3 experts).
- **Impact on results**: Report the expert-validated fidelity rate alongside automated metrics. If expert agreement is low on certain obligation types, flag those as areas where the formalizability grading system must assign lower confidence.

---

## 7. Laptop CPU Feasibility

### Why This Runs on a Laptop CPU

**MaxSMT and ILP solvers are CPU-native.** Z3, CVC5, and Gurobi are highly optimized C/C++ programs designed for CPU execution. They do not benefit from GPU parallelism because SAT/SMT solving is inherently sequential (DPLL/CDCL is a backtracking search, not a data-parallel computation) and ILP branch-and-bound is similarly CPU-bound. The state of the art in SMT and ILP solving is, and for the foreseeable future will remain, CPU-bound.

**The problem has exploitable structure.** Regulatory constraint systems are not adversarial SAT instances. They have significant structure: hierarchical jurisdiction scoping, temporal locality (most obligations don’t change most of the time), modular decomposition by regulatory domain (data governance constraints are largely independent of transparency constraints). This structure enables:

1. **Modular solving**: Decompose the constraint system by regulatory domain and solve sub-problems independently, combining results.
2. **Incremental solving**: When only a few obligations change (new regulation or amendment), use incremental SMT to avoid full recomputation. Z3’s incremental API supports push/pop and assumption-based solving.
3. **ε-approximation**: The Pareto frontier need not be computed exactly — an ε-cover suffices, dramatically reducing the number of MaxSMT queries.
4. **Temporal sparsity**: In a 10-timestep unrolling of a 5-jurisdiction system, most timesteps have identical constraint sets (regulations change at discrete events, not every timestep). Only the transition timesteps require new solving.

**Empirical calibration.** Individual MaxSMT instances in similar domains (software product line configuration, XACML policy analysis) with 500–2000 soft constraints solve in seconds to minutes on modern laptop CPUs. Our temporal unrolling increases instance size linearly in the time horizon (5–10× for core scope), bringing per-instance solve times to minutes. Pareto enumeration requires 50–200 MaxSMT queries per frontier, yielding total solve times in the 5–15 minute range for the core 5-jurisdiction scope. This is well within laptop CPU feasibility. The 30-minute stretch target for 10 jurisdictions is an engineering target to be empirically validated.

**Memory is bounded.** The constraint system, Pareto frontier, and certificates are compact data structures. A 300-obligation, 5-jurisdiction system produces constraint systems of ~30K clauses (fits in <1GB RAM). The Pareto frontier contains at most a few hundred points. Certificates are polynomial in the constraint system size.

### How Evaluation Avoids Human Involvement (Automated Components)

- **Regulatory encoding is pre-done** as part of the artifact (Component B, ~18K LoC for core scope). No runtime human annotation.
- **Benchmark generation is fully automated** via the planted-solution generator (Component I).
- **All metrics are computed programmatically**: hypervolume, generational distance, spread, runtime, memory.
- **No human judgment required for correctness**: certificates are machine-verified; Pareto frontier quality is measured against planted ground truth.
- **No user studies**: the evaluation measures the system’s formal properties, not user satisfaction.
- **Exception**: External encoding validation (Section 6.7) is a one-time human evaluation of encoding fidelity, not part of the runtime evaluation loop.

---

## 8. Risk Analysis and Kill Gates

| Gate | Timeline | Pass Condition | Fail Action |
|------|----------|----------------|-------------|
| **G1: DSL expressiveness** | Week 4 | Can encode 50+ articles from EU AI Act with type-checking | Simplify type system; pivot to constraint language without types |
| **G1.5: Early scalability** | Week 6 | Prototype solves 3 jurisdictions × 100 obligations × 5 timesteps in <10 minutes on laptop (ILP or MaxSMT) | Fundamental scalability problem — reduce temporal depth, switch to pure ILP, or decompose by regulatory domain before proceeding |
| **G2: ILP baseline working** | Week 8 | ILP (Gurobi) solves static 5-jurisdiction, 300-obligation instance with Pareto enumeration; MUS extraction functional | ILP is the foundation — if it fails, the project scope must be radically reduced (2–3 jurisdictions, fewer obligations) |
| **G3: MaxSMT research delta** | Week 12 | MaxSMT formulation demonstrates measurable advantage over ILP on at least one axis (MUS quality, nested logical constraints, or temporal trajectories). If ILP is sufficient for all practical instances, the temporal trajectory dominance result (M1) still provides the paper contribution by demonstrating *where* ILP fails. | If MaxSMT adds nothing over ILP, pivot: paper contribution becomes the DSL + ILP-based infeasibility detection + formalizability completeness theorem (still publishable at ICSE Tools Track) |
| **G4: Pareto frontier quality** | Week 14 | ε-cover with hypervolume ratio >0.9 on planted benchmarks | Refine ε; accept approximate guarantees |
| **G5: End-to-end pipeline** | Week 18 | Full pipeline runs on 5-jurisdiction core corpus with certificates | Reduce temporal depth; extend timeline for stretch jurisdictions |

**Key risk: ILP sufficiency.** If the ILP baseline proves sufficient for all practical instances (static and temporal), the MaxSMT temporal extension is still the paper contribution — it formally characterizes the problem class and demonstrates where ILP’s expressiveness limitations matter (deeply nested conditional obligations, complex regulatory cross-references). The paper narrative shifts from “MaxSMT is better” to “here is the formal tractability result; ILP works for the common case; MaxSMT is needed for the hard tail.” This is an honest and publishable outcome.

---

## 9. Differentiation from Portfolio

RegSynth is **distinctly different** from all 28 existing portfolio projects:

- **vs. dp-verify-repair**: DP focuses on differential privacy mechanisms; RegSynth focuses on multi-jurisdictional AI governance regulations. Different domain (privacy theory vs. regulatory compliance), different techniques (DP composition vs. MaxSMT/ILP optimization).
- **vs. cross-lang-verifier**: Cross-lang verifies program equivalence across programming languages; RegSynth verifies regulatory compliance across legal jurisdictions. Different domain entirely.
- **vs. tensorguard**: Tensorguard verifies neural network safety properties; RegSynth optimizes organizational compliance strategies. Different level of abstraction (model properties vs. organizational processes).
- **vs. synbio-verifier**: Synbio verifies synthetic biology circuit designs; RegSynth operates in AI governance. Completely orthogonal domains.
- **vs. tlaplus-coalgebra-compress**: TLA+ focuses on temporal logic state space compression for concurrent systems; RegSynth uses temporal MaxSMT for regulatory trajectory optimization. Different application domain and different algorithmic approach (coalgebraic compression vs. Pareto synthesis).
- **vs. market-manipulation-prover**: Market manipulation proves financial market abuse; RegSynth optimizes AI regulatory compliance. Different domain (finance vs. AI governance).
- **vs. algebraic-repair-calculus**: Algebraic repair operates on data/schema repair; RegSynth operates on regulatory compliance strategies. Different domain and different mathematical foundations.

The closest potential overlap is with any sibling project in area-075 focusing on regulatory conflict detection or accountability certification. RegSynth’s key differentiator: **it provides the first formal tractability result for multi-jurisdictional AI regulation** — not just detecting conflicts but proving infeasibility with minimal cores, synthesizing Pareto-optimal strategies, and certifying all results with machine-checkable proofs. Conflict detection is the headline capability (Component E, MUS extraction + provenance mapping); Pareto synthesis and temporal trajectory optimization build on top of it. Accountability certification operates on organizational role structures, not compliance cost optimization — a fundamentally different problem.

---

## 10. Summary

**RegSynth** is a ~115K LoC constraint-solving engine (~55K novel algorithmic core, in Rust + Python) that makes multi-jurisdictional AI regulatory compliance formally tractable. Its primary contribution is **formal infeasibility detection with MUS-mapped regulatory diagnoses**: the first system capable of proving that a set of regulatory obligations across jurisdictions is unsatisfiable, identifying the minimal conflicting core mapped back to specific articles, and certifying the proof. Its secondary contribution is **Pareto-optimal compliance strategy synthesis** over the formalizable fragment, with a formalizability completeness theorem (M5) that makes partial formalization rigorous by providing ε-approximation guarantees relative to the honestly-graded formalizable fragment. Its research extension demonstrates that **temporal trajectory optimization** is provably superior to per-timestep optimization, motivating the MaxSMT formulation over static ILP. The regulatory DSL with its formalizability grading system is an independent PL contribution. The system is built on an ILP baseline (Gurobi) with MaxSMT (Z3/CVC5) as the research delta, runs entirely on a laptop CPU, targets 5 core jurisdictions (~300 obligations) in under 15 minutes, evaluates against both synthetic benchmarks with planted ground truth and expert-validated real regulatory corpora, and includes ILP-vs-MaxSMT comparison as a primary evaluation axis. External regulatory domain experts validate encoding fidelity on a stratified sample. Target venue: ICSE 2027 (Tools Track), with FAccT 2027 as secondary.

---

**Slug:** `pareto-reg-trajectory-synth`
