# RegSynth: Competing Approaches

**Slug:** `pareto-reg-trajectory-synth`
**Stage:** Ideation — Approach Design
**Date:** 2026-03-08
**Expert Panel:** Domain Visionary, Math Depth Assessor, Difficulty Assessor, Adversarial Skeptic

---

## Approach A: "ConflictProver" — MUS-First Regulatory Conflict Detection Engine

### Thesis

Build the world's first formal conflict prover for multi-jurisdictional AI regulation: a lean, sharply focused tool that answers the single most valuable question in cross-border compliance — *"Are these obligations simultaneously satisfiable, and if not, which specific articles are in irreducible conflict?"* — with machine-checkable certificates.

### 1. Extreme Value Delivered

**Who desperately needs this:** Chief Compliance Officers at multinationals facing EU AI Act enforcement (August 2026, €35M penalties). They rely on legal teams manually reading regulations from multiple jurisdictions, hoping to spot conflicts. When conflicts surface post-implementation, remediation costs 5–50× the upfront analysis. Regulatory bodies (EU AI Office, NIST) need systematic impact analysis of how their regulations interact with other jurisdictions. AI auditing firms (Holistic AI, Credo AI) need defensible conflict assessments.

**Why this approach maximizes value:** ConflictProver does one thing brilliantly: detect, prove, and diagnose regulatory conflicts. No existing tool — not GRC platforms, not LLM assistants, not deontic logic systems — can produce a formal proof that specific articles across specific jurisdictions are in irreducible conflict. This is commercially irreplaceable. The depth check identified this as "the most commercially and academically defensible feature" (Amendment A8).

By stripping away Pareto optimization, temporal trajectories, and remediation planning, ConflictProver achieves faster time-to-value, a sharper narrative ("we prove regulatory conflicts exist"), and higher credibility (lean tool doing one thing provably correct).

**What becomes possible:** An organization presents a formal infeasibility certificate to regulators: "Articles 12(1) of the EU AI Act and 5(1)(c) of GDPR are mutually unsatisfiable for high-risk AI systems processing personal data. Here is the machine-checkable proof." A regulatory body runs ConflictProver on a draft regulation to identify conflicts *before* publication.

### 2. Genuine Difficulty

**H1. Regulatory DSL with Conflict-Oriented Type System.** Obligation types (OBL/PERM/PROH) indexed by jurisdiction with composition algebra (⊗, ⊕, ▷, ⊘) satisfying algebraic laws. Must be rich enough for real regulatory structure but simple enough that every well-typed obligation has a sound SMT encoding. 10K LoC Rust.

**H2. Faithful Encoding of 300+ Regulatory Articles.** 5 frameworks (~300 articles) into DSL with semantic fidelity. Each article requires careful logical analysis, hard/soft classification, and cross-jurisdictional concept alignment. This is legal interpretation, not programming. 15K LoC Custom DSL.

**H3. Sound Compositional Encoding to SMT.** Translation τ from typed obligations to QF-LIA + arrays + UF, provably sound: SAT(τ(O)) → Satisfiable(O). Compositionality: τ(O₁ ⊗ O₂) = τ(O₁) ∧ τ(O₂). Wrong encoding → wrong certificates. 8K LoC Rust.

**H4. Comprehensive MUS Enumeration with Regulatory Provenance.** Extract *all* minimal unsatisfiable subsets (not just one). Map each MUS back through τ⁻¹ to specific regulatory articles. Provenance tracking through the entire compilation pipeline. Bounded enumeration with prioritization (smallest MUS first, cross-jurisdictional first). 9K LoC Rust.

**H5. Machine-Checkable Infeasibility Certificates.** Two certificate types: compliance certificates (satisfying assignment) and infeasibility certificates (resolution proofs from UNSAT cores). Standalone verifier <2,500 LoC. Proof extraction from Z3/CVC5 is non-trivial (formats differ, proofs enormous). 4.5K LoC Rust.

**H6. Multi-Solver Cross-Validation.** Z3 and CVC5 on identical instances, verifying same verdicts and compatible MUS extractions.

### 3. New Math Required

**M1. Obligation Algebra Soundness (Grade: B).** τ is a homomorphism from (Obligations, ⊗, ⊕, ▷, ⊘) to (SMT, ∧, ∨, priority-ite, except-ite). Soundness: SAT(τ(O)) → Satisfiable(O). Partial completeness for fully-formalizable fragment. Standard compiler-correctness instantiated for novel domain. Load-bearing: without it, certificates are meaningless. Proof: 1–2 weeks.

**M2. MUS Coverage Theorem (Grade: C+).** With budget k = O(n log n), prioritized enumeration covers all cross-jurisdictional conflict cores of bounded size. Partially load-bearing (system works without guarantee). Proof: 2–4 weeks, but may require unrealistic structural assumptions.

**M3. Certificate Correctness (Grade: C+).** Compliance certificates are sound; infeasibility certificates are valid resolution proofs; MUS is genuinely minimal. Standard proof theory + provenance composition. Load-bearing. Proof: weekend.

**M4. Conflict Density Characterization (Grade: C).** Expected MUS count as function of jurisdiction count, obligation density, constraint tightness under a random regulatory model. Not load-bearing — guides engineering decisions. Proof: days.

### 4. Best-Paper Argument

First-of-its-kind formal conflict prover for multi-jurisdictional AI regulation. Perfect timing (EU AI Act enforcement August 2026). Clean, sharp contribution. Artifact speaks for itself. Good ICSE Tools Track fit.

**Weakness:** Moderate math depth — soundness proofs are standard compiler-correctness in new domain. Best-paper depends on tool impact being valued over mathematical depth.

### 5. Hardest Technical Challenge

**MUS enumeration with regulatory provenance at scale.** Number of MUS can be exponential; bounded enumeration with prioritization, provenance maintenance through the pipeline, multi-solver deduplication. Mitigation: MARCO algorithm with incremental SAT oracle, provenance as clause-to-obligation annotations, regulatory relevance ranking.

### 6. Scores

| Dimension | Self-Assessed | Math Review | Difficulty Review | Skeptic Review | **Calibrated** |
|-----------|:---:|:---:|:---:|:---:|:---:|
| **Value** | 8 | — | — | Partially real | **7** |
| **Difficulty** | 6 | 3 (math) | 5 (eng) | — | **5** |
| **Potential** | 6 | — | — | 1% BP | **5** |
| **Feasibility** | 9 | 0.95 completion | 0.85 completion | 0.65 pub | **8** |

### 7. LoC Breakdown

| Component | LoC | Novel? |
|-----------|:---:|--------|
| DSL & Type System | 10,000 | Yes — obligation types |
| Regulatory Data Corpus | 15,000 | Domain data |
| Constraint Encoding Engine | 8,000 | Yes — sound translation |
| Solver Backend + MUS | 9,000 | Yes — provenance MUS |
| Certificate Generator | 4,500 | Yes — proof extraction |
| Evaluation Framework | 14,000 | Standard |
| CLI / Reporting | 4,000 | Standard |
| Infrastructure | 4,500 | Standard |
| **TOTAL** | **~69,000** | **~31.5K novel (46%)** |

*Difficulty review adjusts genuine novel code to ~18–22K LoC.*

### 8. Key Risks

| Gate | Week | Condition | Fail Action |
|------|:---:|-----------|-------------|
| G1: DSL expressiveness | 4 | 50+ EU AI Act articles | Simplify types |
| G2: MUS quality | 6 | Meaningful MUS from 100-obligation instance in <30s | Single-MUS |
| G3: Real conflicts | 8 | ≥5 genuine conflicts in EU+GDPR+China | Encoding revision |
| G4: Certificates | 10 | Standalone verifier confirms 100% | Redesign format |
| G5: External validation | 14 | ≥80% expert agreement | Encoding overhaul |

**Kill probability: ~15%. P(any pub): ~65%. P(best-paper): ~1–4%.**

---

## Approach B: "RegSynth-Complete" — Formalizability-Graded Synthesis with Bounded Uncertainty

### Thesis

Build a layered constraint-solving engine whose intellectual crown jewel is the **formalizability completeness theorem**: the first formal proof that partial regulatory formalization — honestly admitting that 30–40% of obligations resist formalization — still yields Pareto-optimal compliance strategies with quantified, bounded uncertainty over the opaque fragment.

### 1. Extreme Value Delivered

**Who desperately needs this:** All stakeholders from Approach A, plus strategic compliance planners who need "here are your 12 best compliance strategies, ranked by cost/time/risk/burden, with formal guarantees on what we can prove and honest bounds on what we can't." Boards making multi-million-dollar compliance investment decisions need the full trade-off space.

**Why this approach maximizes value:** The key insight: real-world compliance is not binary "feasible/infeasible." Most deployments face a spectrum of partially-compliant strategies with vastly different cost profiles. Two strategies at 95% coverage may differ by 10× in cost. The deeper insight: honest treatment of partial formalizability. Every prior attempt either pretends all obligations are formalizable (overconfident) or gives up entirely. RegSynth-Complete proves that partial formalization is formally valid with quantified uncertainty.

**What becomes possible:** Everything from Approach A, plus: "Here are 15 Pareto-optimal strategies. Strategy 7 costs $2.1M/94% coverage/8 months. Strategy 12 costs $4.8M/99% coverage/4 months. The remaining 6% of obligations are opaque — worst-case bound on their impact is +$300K/+2 months." Plus temporal re-optimization when regulations change.

### 2. Genuine Difficulty

**H1–H6: All from Approach A** (prerequisites, not duplicated effort).

**H7. Formalizability Grading System with Confidence Algebra.** Each obligation carries a grade: Full, Partial(α), or Opaque. Grades propagate through composition algebra: γ(O₁ ⊗ O₂) = min(γ(O₁), γ(O₂)). Must be compositional, sound, and informative. Implementation is trivial (~500–1000 LoC); the research value is the theorem, not the code.

**H8. Pareto Frontier via Iterative Weighted Partial MaxSMT.** ε-approximate Pareto frontiers over 4D cost space. Each point requires a MaxSMT solve; frontier may have hundreds of points. Blocking clause construction for dominance cones. ε-coverage guarantee with termination. ILP (Gurobi) provides baseline; MaxSMT extends with richer logical constraints. This is the hardest engineering subproblem — expect 6–10 weeks.

**H9. Temporal Regulatory Transition Model.** Compliance trajectories through time-varying constraint landscapes with bounded transition budgets. ILP for static; MaxSMT temporal unrolling as research delta. Temporal unrolling multiplies constraint size linearly → performance risk.

**H10. Bounded Uncertainty Propagation.** Worst-case cost multiplier for opaque obligations; "confidence envelope" around Pareto points. Engineering is trivial (~200–500 LoC); research value is in the theorem.

### 3. New Math Required

**M1. Formalizability Completeness Theorem (Grade: B-/C+ — Crown Jewel).**
Let O = O_F ∪ O_P ∪ O_X (Full, Partial, Opaque). P_F is computed Pareto frontier restricted to formalizable fragment. Prove:
- P_F is ε-approximation of true frontier P*_F
- For each point p ∈ P_F, true cost c*(p) ∈ [c_F(p), c_F(p) + Δ_X] where Δ_X is computable
- Bound Δ_X is tight (adversarial construction)

Math review assessment: the first bullet is a property of the Pareto algorithm, not a formalizability result. The second is interval arithmetic (1960s). Only the tightness claim has teeth, and it may require unrealistic independence assumptions. Self-grade B+ is optimistic; honest grade B-/C+. Risk: reviewers dismiss as "just sensitivity analysis."

**M2. Obligation Algebra Soundness (Grade: B).** Same as Approach A M1.

**M3. Pareto ε-Coverage via Iterative MaxSMT (Grade: C+/B-).** ε-constraint scalarization adapted to MaxSMT with blocking clauses. O(1/ε^{d-1}) iteration bound. The MaxSMT adaptation has a genuine wrinkle (non-convex feasible regions) but is not deep.

**M4. Temporal Trajectory Dominance (Grade: C — Motivating Formalization).** Per-timestep Pareto optimality ≠ trajectory Pareto optimality. Known since Bellman 1957. Constructive 3-timestep proof. An afternoon's work. Honestly graded as motivating formalization; reputational risk if presented as a theorem.

**M5. Confidence Algebra Soundness (Grade: C+).** Propagation rules form sound abstract interpretation. Structural induction, standard.

**M6. Incremental Pareto Maintenance (Grade: C).** Test old points, re-solve gaps, merge. Algorithmic recipe, not theorem.

Math review verdict: "Approach B has the most inflated grades. Six contributions graded B+ to C+ that honestly range from C to B-. The crown jewel risks being sensitivity analysis."

### 4. Best-Paper Argument

Formalizability completeness theorem is genuinely novel framing — no prior work formalizes what partial regulatory formalization guarantees. Three-layered contribution (each independently publishable). Intellectual honesty as selling point. Comprehensive artifact (~115K LoC). ILP-vs-MaxSMT comparison as independent evaluation contribution.

**Weakness:** Scope is ambitious. Crown jewel may be dismissed as sensitivity analysis. ILP baseline may deliver 80–90% of value. 11 components → none done deeply enough.

### 5. Hardest Technical Challenge

**Proving the formalizability completeness theorem in a way that's non-trivial.** Risk: reduces to "bound the easy parts, worst-case the hard parts." Must show: tightness is achieved, naïve grade propagation is too conservative, and the threshold-quality trade-off is non-trivial.

### 6. Scores

| Dimension | Self-Assessed | Math Review | Difficulty Review | Skeptic Review | **Calibrated** |
|-----------|:---:|:---:|:---:|:---:|:---:|
| **Value** | 8 | — | — | Wrong form factor | **7** |
| **Difficulty** | 8 | 4 (math) | 7 (eng) | — | **7** |
| **Potential** | 8 | B-/C+ crown jewel | — | 3% BP | **6** |
| **Feasibility** | 6 | 0.75 completion | 0.55 completion | 0.50 pub | **5** |

### 7. LoC Breakdown

| Component | LoC | Novel? |
|-----------|:---:|--------|
| DSL & Type System (with grading) | 15,200 | Yes |
| Regulatory Data Corpus | 18,000 | Domain data |
| Temporal Model | 6,500 | Yes |
| Constraint Encoding (ILP + SMT) | 11,200 | Yes |
| Solver Backend (Z3/CVC5/Gurobi) | 12,500 | Yes |
| Pareto Synthesis Engine | 7,500 | Yes |
| Remediation Planner | 6,500 | Partially (will be descoped) |
| Certificate Generator | 6,000 | Yes |
| Evaluation Framework | 20,000 | Standard |
| CLI / Reporting | 5,500 | Standard |
| Infrastructure | 6,100 | Standard |
| **TOTAL** | **~115,000** | **~55K novel (48%)** |

*Difficulty review adjusts genuine novel code to ~35–40K LoC. Remediation planner will be cut. Realistic: ~90–100K LoC total.*

### 8. Key Risks

| Gate | Week | Condition | Fail Action |
|------|:---:|-----------|-------------|
| G1: DSL expressiveness | 4 | 50+ articles with formalizability grades | Simplify to binary |
| G1.5: Scalability | 6 | 3×100×5 in <10 min | Reduce temporal depth |
| G2: ILP baseline | 8 | ILP Pareto for static 5×300 | Reduce to 3 jurisdictions |
| G3: MaxSMT delta | 12 | Advantage on ≥1 axis | Pivot: DSL + ILP + M1 only |
| G4: Frontier quality | 14 | Hypervolume >0.9 on planted benchmarks | Refine ε |
| G5: End-to-end | 18 | Full pipeline with certificates | Reduce temporal depth |

**Kill probability: ~35%. P(any pub): ~50%. P(best-paper): ~3%.**

---

## Approach C: "RegVerify" — Regulatory Type Theory with Verified Compilation

### Thesis

Recast multi-jurisdictional regulatory compliance as a **type-theoretic problem**: obligations are types, compliance strategies are inhabitants, regulatory conflicts are type errors, and the compilation from types to constraints is *verified* (proven correct in Lean 4). Crown jewel: a **verified regulatory constraint compiler**.

### 1. Extreme Value Delivered

**Who desperately needs this:** High-assurance compliance environments (financial services, healthcare, defense-adjacent AI) where a compliance tool's own correctness is a liability question. RegTech vendors needing a formally-verified core engine. Standards bodies needing reference implementations.

**Why this approach maximizes value:** The fundamental trust problem: why should an organization trust the tool's formalization? Approaches A/B address this through expert validation (human-in-the-loop). Approach C addresses it architecturally: verified compilation guarantees encoding correctness by construction.

**What becomes possible:** Everything from Approach A, with certificates doubly certified (conflict proof valid AND encoding provably correct). A formally verified core suitable for high-assurance deployment. Reusable verification framework for future regulatory DSLs.

### 2. Genuine Difficulty

**H1. Dependently-Typed Regulatory DSL.** Obligation types carrying jurisdiction index (dependent on jurisdiction lattice), temporal interval, obligation modality, and constraint arity. Conflict detection = type inhabitation. Requires normalization engine, unification with dependent types, decidability proof constraining implementation. 18K LoC Rust. Genuinely hard — 2–3 months for working dependent type checker.

**H2. Verified Compilation Pipeline.** Denotational semantics ⟦·⟧ for DSL, standard semantics for SMT, machine-checked proof that ⟦O⟧ = Solutions(τ(O)). In Lean 4 or Coq. Involves type erasure, constraint linearization, Boolean flattening, backend-specific lowering — each transformation preserving semantics. 7K LoC claimed → **12–22K LoC realistic** per difficulty review.

**H3. Multi-Backend Compilation.** Verified compiler targets ILP, MaxSMT, ASP. Common IR with backend-specific lowering, each separately verified. ASP verification is hardest (stable model semantics harder to formalize).

**H4. Conflict Detection via Type Inhabitation.** Type-level pre-filter catches "obvious" conflicts in O(n²); solver handles remainder. Pre-filter must be conservative (no false negatives). Speedup depends on fraction in "linear fragment" — hypothesized 60–70%.

**H5. MUS + Certificates.** Same as A/B, with additional verification layer reference.

**H6. Lean 4 Formalization.** The project-killer. Claimed 5–8K lines, **realistic 12–22K lines** per difficulty review. Proof brittleness: changing definitions cascades through dozens of proofs. 50–100 iteration cycles over 6 months. P(full completion) ≈ 30–40%.

### 3. New Math Required

**M1. Obligation Type Theory Metatheory (Grade: B+ to A- — Crown Jewel).**
Define ObligationType with: base types indexed by jurisdiction and temporal interval; type constructors ⊗, ⊕, ▷, ⊘; subtyping for obligation refinement; inhabitation judgment for compliance. Prove:
- **Decidability:** Type-checking terminates (non-trivial for dependent types)
- **Soundness:** ⊢ σ : O → σ genuinely satisfies O
- **Completeness (decidable fragment):** σ satisfies O and O is decidable → ⊢ σ : O

Math review: A- is defensible if dependent types involve genuine computation (not just finite-set indexing). Multi-month proof effort. Most mathematically novel contribution across all approaches.

**M2. Verified Compilation Correctness (Grade: B+ paper / A- mechanized).** Mechanized in Lean 4: ⟦O⟧ = Solutions(τ(O)). Published verified compilers prove harder results (CompCert), but regulatory DSL is novel domain. 4–6 month effort for experienced Lean developer.

**M3. Type-Level Conflict Decidability Characterization (Grade: B).** Linear fragment (no disjunction/exception) has P-time inhabitation; full fragment is NP-complete. Genuine complexity-theoretic work for new type system.

**M4. Modular Backend Correctness (Grade: C+/B-).** Independent verification per backend. Cumulative effort substantial (3 backends × 1–2 months per).

**M5. Obligation Subtyping Soundness (Grade: C+).** Corollary of M1.

Math review verdict: "The only approach with genuinely hard math. Type theory metatheory is real PL theory. But feasibility risk: probability of completing math in time ≤50%."

### 4. Best-Paper Argument

Verified compilation is deep (qualitative jump in assurance). Novel type theory for regulatory obligations. Bridges PL and governance. Compelling trust argument. PLDI/OOPSLA venue fit.

**Weakness:** Lean 4 verification extremely high risk. Timeline incompatible with regulatory urgency. Skeptic: "Verification addresses a secondary concern (compiler bugs) at enormous cost — encoding bugs from misinterpreting legal text are the real problem."

### 5. Hardest Technical Challenge

**Mechanizing the compilation correctness proof in Lean 4.** Semantic formalization gap, proof brittleness, compilation transformation tower. Every verified compiler project exceeds proof estimates by 3–10×. Mitigation: start with simplest sub-language, use Lean 4 automation, kill gate at week 12, partial verification fallback.

### 6. Scores

| Dimension | Self-Assessed | Math Review | Difficulty Review | Skeptic Review | **Calibrated** |
|-----------|:---:|:---:|:---:|:---:|:---:|
| **Value** | 7 | — | — | Premature | **5** |
| **Difficulty** | 9 | 7 (math) | 8 (eng) | — | **8** |
| **Potential** | 9 | A- crown jewel | — | 2% BP (uncond.) | **7** |
| **Feasibility** | 4 | 0.45 completion | 0.35 full / 0.65 descoped | 0.30 pub | **3** |

### 7. LoC Breakdown

| Component | LoC | Novel? |
|-----------|:---:|--------|
| Dependently-Typed DSL | 18,000 | Yes — dependent obligation types |
| Lean 4 Formalization | 7,000 (claimed) / 15,000 (realistic) | Yes — mechanized proofs |
| Regulatory Data Corpus | 15,000 | Domain data |
| Verified Constraint Compiler | 12,000 | Yes — multi-backend |
| Solver Backend (4 solvers) | 10,000 | Partially novel |
| MUS + Certificates | 5,500 | Yes |
| Evaluation Framework | 16,000 | Standard |
| CLI / Reporting | 4,000 | Standard |
| Infrastructure | 5,500 | Standard |
| **TOTAL** | **~93K–101K** | **~52K novel (56%)** |

*Difficulty review adjusts genuine novel code to ~40–45K LoC.*

### 8. Key Risks

| Gate | Week | Condition | Fail Action |
|------|:---:|-----------|-------------|
| G1: Type theory metatheory | 4 | Decidability proof sketch for linear fragment | Simplify to non-dependent |
| G2: Lean 4 core | 8 | DSL syntax + semantics formalized | Abandon verification → Approach A |
| G3: Compilation proof | 12 | Correctness for linear fragment | Scope to linear only |
| G4: MUS + certificates | 14 | Working conflict detection 100-obligation | Approach A fallback |
| G5: Multi-backend | 16 | ILP + MaxSMT both working | Drop ASP |
| G6: External validation | 18 | Expert ≥80% agreement | Encoding revision |

**Kill probability: ~55%. P(any pub): ~30%. P(best-paper): ~2% (unconditional).**

---

## Cross-Approach Comparison

| Dimension | A: ConflictProver | B: RegSynth-Complete | C: RegVerify |
|-----------|:---:|:---:|:---:|
| **Crown Jewel** | First formal conflict prover | Formalizability completeness | Verified regulatory compiler |
| **Value** | 7 | 7 | 5 |
| **Difficulty** | 5 | 7 | 8 |
| **Potential** | 5 | 6 | 7 |
| **Feasibility** | 8 | 5 | 3 |
| **Calibrated Composite** | **6.25** | **6.25** | **5.75** |
| **Novel LoC** | ~20K | ~37K | ~42K |
| **Total LoC** | ~69K | ~100K | ~97K |
| **Kill Prob** | ~15% | ~35% | ~55% |
| **P(best-paper)** | ~2% | ~3% | ~2% uncond. |
| **P(any pub)** | ~65% | ~50% | ~30% |
| **Math Depth** | 3/10 | 4/10 | 7/10 |
| **Eng Difficulty** | 5/10 | 7/10 | 8/10 |
| **Risk-Adj Math** | 2.85 | 3.00 | 3.15 |
| **Primary Venue** | ICSE Tools | ICSE Tools / FAccT | PLDI / OOPSLA |
| **Realistic Timeline** | 7–9 months | 15–18 months | 15–24 months |
| **Risk Profile** | Low risk, moderate reward | Moderate risk, moderate reward | High risk, high reward |

### Expert Consensus

**Math Depth Assessor:** "None is a math paper. C has genuinely hard math (A- crown jewel) but P(completion) ≤ 50%. B is the weakest — tries to look like a math paper without doing hard math. Choose C if you have Lean 4 expertise, A otherwise."

**Difficulty Assessor:** "A is achievable (5/10, 7–9 months). B descopes to A + ILP Pareto in practice. C requires dedicated Lean 4 expertise the team may not have. All three underweight the regulatory encoding bottleneck."

**Adversarial Skeptic:** "None is fundable as stated. The entire project rests on an unvalidated assumption: that formal proofs of regulatory conflict are what practitioners want. A comes closest to viable. B's scope is the enemy. C puts the verification cart before the viability horse."
