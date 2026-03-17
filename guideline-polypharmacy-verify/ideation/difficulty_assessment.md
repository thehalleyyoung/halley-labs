# GuardPharma: Difficulty Assessment

**Date:** 2025-07-18
**Phase:** Domain Visionary — Difficulty Assessment
**Assessor:** Difficulty Assessor

---

## Preamble: Scope of Assessment

This assessment evaluates the genuine software artifact difficulty of each of the three proposed approaches. The goal is precision about what is *hard* versus what is *tedious*, and what is *novel* versus what is *known-hard-with-known-solutions*. LoC estimates are stress-tested against comparable systems. Feasibility is evaluated against a 12–16 week timeline with 1–2 strong engineers.

---

## Approach 1: PTA-Contract Compositional Model Checking

### A. Hard Subproblems Identification

**1. CQL-to-PTA Semantic Compilation (claimed ~10K novel LoC)**

- **WHY hard:** CQL is a non-trivial language with temporal operators, interval arithmetic, FHIR-path navigation, terminology bindings, and a rich type system. The compilation target (hybrid automata transitions with PK state guards) has no precedent — this is defining a formal semantics for CQL by construction. The hard part is *semantic fidelity*: ensuring that the automaton transitions faithfully represent the clinical decision logic, including edge cases in temporal operators (e.g., "during" vs. "overlaps" in CQL interval logic) and dynamic data requirements (lab values that arrive asynchronously).
- **Known or unsolved?** KNOWN hard problem (compiler construction for a DSL into formal models) with NO known solution for this specific source/target pair. The closest analogy is Lustre-to-automata compilation in synchronous languages, but CQL's asynchronous, data-dependent semantics are fundamentally different. The HL7 CQL spec is ~500 pages; achieving even 70% coverage is a multi-month effort.
- **Effort:** 3–4 person-months for a strong compiler engineer familiar with both CQL and hybrid automata. 5–6 person-months for someone learning CQL semantics on the job. **However, the problem statement's phased delivery wisely defers this to post-paper.** For the paper phase, guidelines are manually encoded as PTA.

**2. Contract Extraction from PK Models (claimed ~14K novel LoC)**

- **WHY hard:** The (assume, guarantee) pairs over the CYP-enzyme activity vector require computing worst-case enzyme loads from population PK parameters — a parametric reachability problem over Metzler ODEs. The circular dependency (drug concentrations depend on enzyme activity, which depends on drug concentrations) must be resolved via a monotonicity argument. The monotonicity proof for competitive CYP inhibition is sketched in the problem statement and appears sound, but *non-competitive inhibition and mixed mechanisms break monotonicity*, requiring a fallback to direct product verification.
- **Known or unsolved?** NOVEL instantiation of assume-guarantee reasoning. The assume-guarantee framework is well-established (Pnueli 1985, Giannakopoulou 2005, Pacti 2022), but the enzyme-interface contract abstraction is new. The monotonicity argument for Metzler PK dynamics is the genuinely novel mathematical contribution. The fixed-point resolution via worst-case guarantees is elegant *if* the monotonicity holds — and the problem statement is honest that it holds only for competitive inhibition (~70% of clinically significant PK DDIs).
- **Effort:** 2–3 person-months. The mathematics is well-scoped; the implementation requires careful parameterization against published PopPK data.

**3. CEGAR with Clinical Domain Abstractions (claimed ~14K novel LoC)**

- **WHY hard:** The PK region graph for 5+ concurrent drugs can have ~10^15 regions. The CEGAR loop must converge, but convergence for PK-structured state spaces is *unproven*. The clinical domain abstractions (drug-class equivalence, lab-value coarsening, temporal aggregation) are heuristic — there's no guarantee they produce useful refinements. If CEGAR doesn't converge, the fallback is SAT-based bounded model checking with a 365-day horizon, which is sound but may be slow.
- **Known or unsolved?** CEGAR is a KNOWN technique (Clarke et al. 2003). The clinical domain abstractions are novel but heuristic. The convergence question for this specific domain is genuinely OPEN. This is the "sounds routine but could be a nightmare" component — CEGAR works beautifully on some domains and fails to converge on others, and you don't know which until you try.
- **Effort:** 3–4 person-months. Much of this is empirical tuning of abstraction granularity.

**4. Zonotopic Reachability for Metzler Systems (claimed ~10K novel LoC)**

- **WHY hard:** The reachability algorithm exploits the Metzler property (off-diagonal entries ≥ 0 in the system matrix) for monotone propagation. Drug discontinuation resets break this property and require partitioned analysis. Zonotope order reduction must preserve soundness while ensuring convergence — standard order-reduction schemes (Girard 2005, Kopetzki et al. 2017) are not guaranteed to maintain useful precision for PK state spaces where therapeutic windows are narrow (often 2× between therapeutic and toxic).
- **Known or unsolved?** KNOWN technique (CORA, SpaceEx do zonotopic reachability) applied to a NOVEL domain. The Metzler-specific optimizations are new but build on established computational geometry. The partitioned analysis for drug resets is a novel engineering contribution, not a fundamental research problem.
- **Effort:** 2–3 person-months. Significant existing code can be adapted from CORA (MATLAB) or SpaceEx (C++), though re-implementation in the project's language is needed.

**5. MTL Model Checker with PK Region Graph (claimed ~22K LoC)**

- **WHY hard:** The pharmacokinetic region graph partitions the continuous PK state space at clinical thresholds. The correctness argument (that this discretization is sound for MTL properties referencing only clinical threshold predicates) is the technical core. Region graph construction for N drugs with T thresholds each produces O(T^N) regions. For 5 drugs with 3 thresholds each (sub-therapeutic, therapeutic-toxic boundary, toxic-lethal boundary), that's 3^5 = 243 regions — manageable. For 10 drugs: 3^10 ≈ 59K regions — still feasible. For 20 drugs: 3^20 ≈ 3.5 billion — intractable without the compositional decomposition.
- **Known or unsolved?** Region-based model checking is KNOWN (Alur-Dill 1994). The PK-specific region construction is a NOVEL domain application. The soundness argument is where the real intellectual work lies.
- **Effort:** 3–4 person-months. The BDD and SAT backends (BuDDy/CUDD, CaDiCaL) are existing tools, but the PK region encoding into BDD variables is non-trivial.

**6. Validated Interval ODE Integration (claimed part of ~12K PK library)**

- **WHY hard:** δ-decidability requires validated interval arithmetic with directed rounding and wrapping-effect control. Wrapping CAPD or VNODE-LP correctly is notoriously fiddly — off-by-one errors in rounding modes produce unsound results. The integration must handle the Metzler-specific structure to avoid catastrophic wrapping-effect growth.
- **Known or unsolved?** KNOWN hard problem with KNOWN solutions (CAPD, VNODE-LP, dReal). The difficulty is integration engineering, not fundamental research. But integration engineering for validated numerics is one of those "2 weeks estimated, 2 months actual" tasks.
- **Effort:** 1.5–2.5 person-months. Highly dependent on the quality of existing library bindings.

### B. Architectural Challenges

**Hardest integration points:**
1. **CQL → PTA → PK Model → Region Graph → Model Checker pipeline.** Each interface is a potential semantic gap. The PTA must faithfully represent CQL semantics *and* be amenable to PK region construction *and* be checkable by the MTL model checker. Any mismatch means bugs that are invisible until the final evaluation.
2. **Validated ODE solver ↔ Zonotopic reachability ↔ CEGAR.** The CEGAR loop needs to query the reachability engine, which uses the ODE solver. Latency in this inner loop determines overall verification time. If the ODE solver is slow (validated arithmetic is 10–100× slower than floating-point), the CEGAR loop may time out before convergence.
3. **Terminology resolution at compilation time.** CQL references concepts by ValueSet URIs that must be expanded against terminology services (SNOMED CT, RxNorm, LOINC). This requires either a live terminology server or a pre-loaded snapshot. Either way, it's a brittle external dependency.

**Where the system most likely breaks under real-world inputs:**
- **CQL guidelines that use unsupported language features.** CQL 1.5 has aggregate functions, tuple types, and query syntax that are non-trivial to compile. Real-world CQL libraries (e.g., CMS eCQMs) use constructs the compiler doesn't support.
- **Population PK parameters that aren't published for the specific drug.** The PK model library needs published compartmental parameters for every drug in every guideline. For common drugs (metformin, warfarin, atorvastatin), these exist. For newer drugs or drug combinations, they may not.
- **Enzyme interactions that don't fit the competitive-inhibition model.** The contract-based decomposition only covers CYP-mediated competitive inhibition. Non-competitive inhibition, enzyme induction, and PD interactions require the monolithic fallback, which may not scale.

**Critical path dependencies:**
- PTA formalism definition → everything else (all components depend on the PTA data structure)
- PK model library → contract extraction → compositional verification
- Region graph construction → MTL model checker → CEGAR loop

### C. LoC Estimates Reality Check

| Subsystem | Claimed LoC | Realistic LoC | Assessment |
|-----------|-------------|---------------|------------|
| CQL-to-PTA Compiler (deferred) | ~18K | 12–25K | **Plausible but high variance.** CQL coverage is the variable. 70% coverage: ~12K. 95% coverage: ~25K+. |
| FHIR PlanDef Compiler (deferred) | ~10K | 6–12K | **Slightly overestimated.** PlanDefinition is complex but well-documented. HAPI FHIR does most parsing. |
| Guideline Corpus Pipeline (deferred) | ~10K | 8–15K | **Underestimated.** Heterogeneous source handling, dependency resolution, and ValueSet expansion are notoriously tedious. Integration code always grows. |
| Clinical State Space Model | ~8K | 5–8K | **Plausible.** Data modeling with domain invariants. |
| PK Model Library | ~12K | 10–18K | **Underestimated.** Validated interval ODE integration alone could be 5–8K. PopPK parameterization for ~50 drugs adds bulk. |
| PTA Construction & Composition | ~18K | 12–20K | **Plausible.** Product construction is algorithmic; contract extraction is novel. |
| Zonotopic Reachability | ~16K | 10–16K | **Plausible.** Much of this is computational geometry with known algorithms. |
| MTL Model Checker + CEGAR | ~22K | 15–25K | **Plausible but depends heavily on backend integration.** Z3/CUDD wrappers are significant code. |
| Counterexample Generator | ~7K | 5–10K | **Underestimated.** Clinical narrative generation is deceptively complex (mapping formal states back to clinical concepts). |
| Clinical Significance Filter | ~10K | 8–12K | **Plausible.** Multi-source data integration is tedious but well-scoped. |
| Terminology Layer (deferred) | ~6K | 4–8K | **Plausible.** HAPI FHIR wrapping. |
| Evaluation Engine | ~13K | 10–18K | **Underestimated.** Reproducible benchmark harnesses always have more infrastructure than expected. TMR baseline re-implementation alone is 2–4K. |

**Paper-phase total:** Claimed ~95K. Realistic: **70–120K.** The estimate is within the right order of magnitude but the variance is high.

**Novel code ratio:** Claimed ~65K novel / ~95K total (68%). Realistic: ~55–75K novel / 70–120K total (55–65%). The glue/integration fraction is typically underestimated.

**Most underestimated subsystems:**
1. PK Model Library (validated numerics always takes longer than expected)
2. Evaluation Engine (benchmark infrastructure is invisible work)
3. Counterexample Generator (clinical narrative is hard NLG)

**Most overestimated subsystems:**
1. FHIR PlanDefinition Compiler (HAPI does heavy lifting)
2. Terminology Layer (wrapper code)

### D. Feasibility Assessment

**Can 1–2 engineers build this in 12–16 weeks?**

For the **full 95K LoC paper-phase**: **No.** At a generous 200 LoC/day of production-quality novel code (including tests, debugging, and integration), 95K LoC requires ~475 person-days = ~24 person-months. Even with 2 engineers working 16 weeks, that's 8 person-months — about one-third of what's needed.

For a **minimum viable artifact**: **Possible, with aggressive scoping.**

**What must be cut to hit the deadline (16 weeks, 2 engineers):**
1. ~~CQL/FHIR compilation~~ (already deferred) — manually encode 3–5 guideline pairs as PTA
2. Simplify PK models to 1-compartment only (eliminates multi-compartment ODE complexity)
3. Drop CEGAR — use direct bounded model checking (SAT-based) with a 90-day horizon
4. Drop zonotopic reachability — use interval-based over-approximation (less precise but much simpler)
5. Simplify clinical significance filter to DrugBank lookup only (drop FAERS processing)
6. Limit evaluation to E1, E3, E5, E6 (drop E2, E4, E7, E8, E9)

**Minimum viable artifact for a best-paper submission:**
- 3–5 manually encoded guideline pairs as PTA
- 1-compartment PK models with CYP-inhibition (competitive only)
- Contract-based compositional verification (Theorem 3) — this is the crown jewel
- SAT-based bounded model checking as the verification backend
- Simple counterexample extraction (no clinical narrative — just formal traces)
- E1 (temporal ablation) + E3 (known-conflict recall) + E6 (compositionality speedup)
- **Estimated scope: ~25–35K LoC, ~18–22K novel — achievable in 12–16 weeks by 2 engineers**

### E. Difficulty Scores

| Metric | Score | Rationale |
|--------|-------|-----------|
| **Raw Difficulty** | **8/10** | Three-domain intersection (FM + PK + clinical informatics). Novel formalism. Novel composition theorem. ~65K novel LoC at full scope. Only an 8 not a 9 because all the individual techniques (timed automata, CEGAR, zonotopes, A/G contracts) are known — the novelty is their combination and domain-specific instantiation. |
| **Minimum Viable Difficulty** | **5/10** | With manual PTA encoding, 1-compartment PK, and SAT-based BMC, the core contribution (contract-based composition + temporal ablation experiment) is implementable by strong engineers in 12–16 weeks. The math is the hard part; the code is tractable once the math is settled. |
| **Risk of Non-Termination** | **6/10** | Three independent kill risks: (1) CEGAR doesn't converge for realistic guidelines — mitigated by SAT fallback. (2) E1 temporal ablation shows <15% temporal-only conflicts — mitigated by fallback narrative. (3) PTA encoding of real guidelines turns out to be infeasible — partially mitigated by the 3-pair pilot gate. The pilot gate is well-designed but the project could stall if all three risks materialize simultaneously. |

### F. Red Flags

1. **Validated interval ODE integration is likely 3–5× harder than estimated.** Wrapping CAPD/VNODE-LP correctly, handling rounding modes, and debugging soundness issues in validated numerics is notoriously time-consuming. The problem statement treats this as "wraps existing validated ODE integrator" — in practice, getting the wrapping right *is* the hard part.

2. **The E1 experiment is a high-variance gamble with no fallback that's equally compelling.** The fallback narrative ("explanation quality" and "existence proof") is weaker than the primary narrative ("X% of conflicts require temporal reasoning"). If X < 15%, the paper's impact argument is significantly diminished. This is not a technical difficulty issue — it's a research risk issue — but it affects the practical difficulty of producing a best-paper-quality artifact.

3. **Population PK parameter availability.** The system needs published compartmental PK parameters for every drug pair it verifies. For the 3 pilot guideline pairs (diabetes + hypertension, diabetes + CKD, hypertension + anticoagulation), the relevant drugs (metformin, lisinopril, atorvastatin, warfarin, etc.) have well-published PK data. But scaling to 80 guideline artifacts requires PK data for potentially hundreds of drugs, and not all have published compartmental models.

4. **"Sounds easy but is actually nightmare": counterexample clinical narratives.** Translating a formal trace (sequence of automaton locations with continuous state values) into a readable clinical narrative ("Patient with eGFR < 30 starts metformin day 1, adds lisinopril day 14, warfarin concentration reaches 6.2 μg/mL by day 45...") requires a rich clinical ontology mapping. This is not just string formatting — it's clinical NLG that must be medically accurate.

---

## Approach 2: Abstract Interpretation over Pharmacokinetic Lattices

### A. Hard Subproblems Identification

**1. PK Abstract Domain Design (core of the ~50K novel LoC)**

- **WHY hard:** The precision-coarseness tradeoff is the central challenge. The therapeutic window for many drugs is a 2× range (e.g., warfarin therapeutic 2–5 μg/mL, toxic >5 μg/mL). An abstract domain that widens beyond this 2× range classifies everything as "possibly unsafe" — useless. An abstract domain that maintains sub-μg/mL precision has too many elements for fast convergence. Finding the sweet spot requires deep domain expertise in both abstract interpretation theory and clinical pharmacology.
- **Known or unsolved?** Designing application-specific abstract domains is a KNOWN research methodology (Cousot & Cousot 1977, Mine 2006), but the PK-specific domain is NOVEL. The closest analogy is interval abstract domains for control systems (Blanchet et al. 2003, Astrée), but PK dynamics have fundamentally different structure (Metzler matrices, enzyme-mediated coupling) and precision requirements (narrow therapeutic windows).
- **Effort:** 2–3 person-months for the domain design and implementation. This is the intellectually hardest part but not the most code.

**2. Pharmacokinetic Widening Operator (novel)**

- **WHY hard:** Standard interval widening (jump to ±∞) is catastrophic for PK. The PK-aware widening must exploit two domain properties: (a) no drug concentration exceeds the lethal dose (physical upper bound), and (b) all drugs are eventually eliminated (Metzler eigenvalues have negative real parts → steady-state convergence). Designing a widening that terminates in bounded iterations *and* preserves therapeutic-vs-toxic discrimination requires proving Theorem B (bounded convergence in D iterations, D = number of drugs). The proof relies on steady-state convergence of compartmental PK models, which is physiologically sound but mathematically requires the system matrix to be Hurwitz (all eigenvalues with negative real parts). This holds for elimination-dominant PK models but may fail for drugs with extremely long half-lives (amiodarone: half-life 40–55 days) where steady state is not reached within a clinically relevant horizon.
- **Known or unsolved?** Domain-specific widening is a KNOWN technique (Cousot & Cousot 1992, delayed widening, etc.) but the PK-specific widening with bounded convergence proof is NOVEL. The Metzler-monotonicity argument is the same insight used in Approach 1's contract extraction — applied differently.
- **Effort:** 1.5–2.5 person-months. The proof is the hard part; the implementation is straightforward once the operator is defined.

**3. Reduced Enzyme-Coupling Product (novel)**

- **WHY hard:** When N guidelines share CYP enzymes, the naive product of N abstract domains is N-dimensional in the number of drugs. The reduced product tracks only enzyme-coupled drugs jointly, reducing dimensionality. The key question is whether the reduced product loses critical precision: if drugs A and B share CYP3A4, and drugs B and C share CYP2D6, then A and C are transitively coupled through B — but the reduced product may not track this transitive coupling unless B appears in both enzyme groups. Getting the decomposition right while preserving soundness and useful precision is a careful engineering problem.
- **Known or unsolved?** Reduced product construction is a KNOWN technique (Cousot & Cousot 1979). The enzyme-coupling-specific decomposition is a NOVEL application. The transitive coupling issue is a genuine precision concern.
- **Effort:** 1–2 person-months.

**4. Abstract Transformers for CQL Decision Steps (~15–20K LoC)**

- **WHY hard:** Each CQL decision (medication initiation, dose adjustment, lab check) must be modeled as an abstract transformer over the PK domain. The transformer for "initiate drug X at dose D" must compute the new concentration interval for X, the new enzyme-activity interval for all CYP enzymes X affects, and the resulting concentration changes for all co-administered drugs sharing those enzymes. This requires composing the PK ODE abstraction with the enzyme-coupling model with the clinical decision logic — three layers of abstraction in each transformer.
- **Known or unsolved?** Abstract transformer construction is a KNOWN methodology. The PK-specific transformers are NOVEL. The difficulty is moderate — more tedious than fundamentally hard.
- **Effort:** 2–3 person-months. Mostly implementation work once the domain is designed.

### B. Architectural Challenges

**Hardest integration points:**
1. **Abstract domain ↔ guideline decision logic.** The transformers must faithfully represent clinical decisions in the abstract domain. A transformer for "if eGFR < 30, reduce metformin dose to 500mg" must correctly update the metformin concentration interval given the current abstract state including renal function. This requires the abstract domain to represent renal function as a variable, not just drug concentrations.
2. **Widening ↔ narrowing iteration control.** The initial widening pass produces a sound but imprecise overapproximation. Narrowing recovers precision, but standard narrowing is not guaranteed to converge to a useful fixpoint. Balancing widening and narrowing iterations for useful precision is an empirical art.

**Where the system most likely breaks under real-world inputs:**
- **False positives from imprecise enzyme-coupling.** CYP3A4 is shared by ~50% of drugs. If the abstract transformer for CYP3A4-sharing drug pairs is too imprecise, every combination involving CYP3A4 substrates will be flagged as "possibly unsafe" — making the tool useless for the most clinically relevant drug class. This is the acknowledged hardest challenge.
- **Drugs with extremely long half-lives.** Amiodarone (half-life 40–55 days) doesn't reach steady state within a typical 90-day verification horizon. The widening operator assumes steady-state convergence; for amiodarone-class drugs, the widening may produce excessively wide intervals.

**Critical path dependencies:**
- PK abstract domain definition → widening operator → transformers → everything else
- Much simpler dependency graph than Approach 1, which is a significant feasibility advantage.

### C. LoC Estimates Reality Check

| Subsystem | Claimed LoC (implied) | Realistic LoC | Assessment |
|-----------|----------------------|---------------|------------|
| PK abstract domain + Galois connection | ~10K | 6–10K | **Plausible.** The domain definition is compact; the proof machinery (testing soundness) adds bulk. |
| Widening/narrowing operators | ~5K | 3–6K | **Plausible.** Algorithmically compact but needs extensive testing. |
| Reduced product construction | ~5K | 4–7K | **Plausible.** Graph decomposition + product construction. |
| Abstract transformers for CQL steps | ~15K | 12–20K | **Slightly underestimated.** One transformer per guideline action type; combinatorial variety in CQL actions. |
| Guideline encoding (manual for paper) | ~5K | 3–5K | **Plausible.** Manual encoding of guideline pairs. |
| PK model library (shared with Approach 1) | ~8K | 6–10K | **Plausible.** Simpler than Approach 1 because abstract transformers don't need validated interval ODE — they use the steady-state formulas directly. |
| Evaluation engine | ~10K | 8–15K | **Underestimated.** Same benchmark infrastructure issue as Approach 1. |

**Paper-phase total:** Claimed ~75K. Realistic: **50–80K.** The simpler architecture reduces variance.

**Novel code ratio:** Claimed ~50K novel / ~75K total (67%). Realistic: ~35–55K novel / 50–80K total (60–70%). Closer to accurate than Approach 1's estimates because the architecture has fewer integration points.

**Most underestimated subsystems:**
1. Abstract transformers (combinatorial variety in clinical actions)
2. Evaluation engine (always underestimated)

**Most overestimated subsystems:**
1. PK model library (simpler than Approach 1's validated ODE requirement)

### D. Feasibility Assessment

**Can 1–2 engineers build this in 12–16 weeks?**

For the **full 75K LoC**: **Borderline.** At 200 LoC/day, 75K requires ~375 person-days = ~19 person-months. Two engineers × 16 weeks = ~8 person-months — still short, but closer than Approach 1.

For a **minimum viable artifact**: **Yes, achievable.**

**Minimum viable artifact for a best-paper submission:**
- PK abstract domain with interval-based concentration tracking + enzyme-activity intervals
- PK-aware widening with steady-state bounds (Theorem B)
- Reduced product for enzyme-coupled guidelines (Theorem C)
- Abstract transformers for the 3–5 most common guideline action types
- 5–10 manually encoded guideline pairs
- Speed comparison against an atemporal checker and Approach 1's model checking (if available)
- **Estimated scope: ~20–30K LoC, ~15–20K novel — comfortably achievable in 12–16 weeks by 1–2 engineers**

**What must be cut:**
1. CQL/FHIR compilation (already out of scope for paper phase)
2. Specialized CYP3A4 relational domain (polyhedra sub-domain) — use intervals only, accept higher false-positive rate
3. FAERS processing — DrugBank lookup only
4. Narrowing iterations — widening-only analysis with manual precision assessment

### E. Difficulty Scores

| Metric | Score | Rationale |
|--------|-------|-----------|
| **Raw Difficulty** | **7/10** | Novel abstract domain design requiring deep expertise in both abstract interpretation and pharmacokinetics. Simpler architecture than Approach 1 — no automata, no model checking, no CEGAR — but the precision challenge (avoiding catastrophic false positives) is genuinely hard. |
| **Minimum Viable Difficulty** | **4/10** | With interval-only domains, manual guideline encoding, and acceptance of moderate false-positive rates, the core contribution (PK-aware abstract interpretation with bounded convergence) is achievable by a strong PL/FM engineer in 12–16 weeks. The three theorems are well-scoped and provable. |
| **Risk of Non-Termination** | **4/10** | Abstract interpretation is inherently more predictable than model checking — the widening operator *guarantees* termination, and the soundness argument is clean. The main risk is *precision* (false positives making the tool useless), not *termination* or *correctness*. Precision is a gradient, not a cliff — partial success is always possible. |

### F. Red Flags

1. **CYP3A4 false-positive explosion is the single biggest risk.** If ~50% of drugs are CYP3A4 substrates, and the interval abstract domain can't distinguish safe from unsafe CYP3A4-sharing combinations, then ~25% of all drug pairs are flagged — catastrophic false-positive rate. The proposed mitigation (polyhedra sub-domain for CYP3A4) is sound but adds significant implementation complexity. **This could be 2–3× harder than estimated** if the interval domain proves insufficient and the polyhedra sub-domain is needed for the paper.

2. **No counterexample generation.** Abstract interpretation proves safety or says "possibly unsafe" — it cannot produce the counterexample patient trajectory that Approach 1 generates. For a clinical audience (AIME, AMIA), the lack of actionable diagnostic output is a significant weakness. For a PL/FM audience (SAS, VMCAI, POPL), this is fine.

3. **"Sounds easy but is actually nightmare": abstract transformer testing.** Each abstract transformer must be verified to be a sound overapproximation of the concrete semantics. For simple transformers (dose initiation), this is straightforward. For complex ones (dose adjustment based on lab value + current concentration + renal function), testing soundness requires generating concrete test cases that exercise the transformer's precision boundaries. This testing infrastructure is invisible but essential work.

4. **Theorem A (Galois connection) is load-bearing but may be fragile.** The claim that "Metzler dynamics preserve interval structure — the image of a box under Metzler flow is contained in a computable box" is true for single-drug dynamics but requires careful handling for enzyme-coupled multi-drug dynamics. The enzyme coupling introduces non-diagonal terms in the system matrix that may cause the box image to be non-box (requiring overapproximation that loses precision). The Galois connection proof must handle this carefully; if it can't, the entire approach loses its theoretical foundation.

---

## Approach 3: Pharmacokinetic Safety Games with Safe-Schedule Synthesis

### A. Hard Subproblems Identification

**1. Decidability of Pharmacokinetic Timed Games (Theorem I)**

- **WHY hard:** The decidability claim rests on three pillars: (a) adversary extremalization (worst-case PK parameters are extremal due to Metzler monotonicity), (b) scheduler discretization (continuous administration times can be discretized to a finite grid), and (c) finite game reduction. Pillar (a) is sound for competitive CYP inhibition by the same monotonicity argument as Approach 1. Pillar (b) is the **genuinely hard step**: the grid granularity depends on PK time constants, which vary across drugs. For drugs with fast elimination (e.g., antibiotics with 2-hour half-lives), the grid needs ~30-minute granularity over a 365-day horizon = ~17,500 time points per drug. For D drugs, the scheduler's action space is 17,500^D — astronomically large. The "finite game reduction" doesn't help if the finite game is too large to solve.
- **Known or unsolved?** Decidability of hybrid games with ODE dynamics is in general UNDECIDABLE (Henzinger et al. 1999). The Metzler restriction is claimed to restore decidability, but this is a CONJECTURE in the proposal, not a proven theorem. The adversary extremalization step is novel and plausible but unproven. This is the highest-risk mathematical component across all three approaches.
- **Effort:** 2–4 person-months just for the proof, assuming it succeeds. If the proof fails, the entire approach collapses.

**2. Pareto-Optimal Safe Schedule Computation (Theorem II)**

- **WHY hard:** Computing the Pareto frontier of safe schedules requires solving a multi-objective optimization problem over the schedule polytope, for each of 2^p extremal adversary parameter vectors. The claimed O(D · |grid|^D · 2^p) complexity is exponential in both D (number of drugs) and p (number of independent PK parameters). For D=5, |grid|=100 (coarse), p=5: 5 × 100^5 × 32 = 1.6 × 10^12 operations — intractable on a laptop. The enzyme-group decomposition is supposed to reduce effective D to ~3 per group, giving 3 × 100^3 × 8 = 2.4 × 10^7 — manageable. But this decomposition requires Theorem III (compositional synthesis), creating a dependency chain.
- **Known or unsolved?** Multi-objective optimization in games is a KNOWN research area (Chatterjee et al. 2006, Brenguier et al. 2016) but has not been applied to PK systems. The Pareto polytope characterization (at most D+1 vertices) depends on the safety constraint being a single halfspace per drug — if the therapeutic window creates a bounded interval constraint (both sub-therapeutic and toxic bounds), the Pareto structure may be more complex.
- **Effort:** 3–4 person-months. The mathematical analysis and the implementation are both substantial.

**3. Strategy Synthesis for Continuous-State Games**

- **WHY hard:** Winning strategies for games with continuous state are in general infinite-memory objects. The proposal asserts that Metzler structure permits memoryless or bounded-memory strategies, but this is unproven. If bounded-memory strategies don't exist, the synthesized schedules can't be represented as finite objects — making the entire "output an actionable schedule" narrative impossible.
- **Known or unsolved?** Memory requirements for winning strategies in hybrid games are an active RESEARCH AREA. For timed games (without ODE dynamics), memoryless strategies exist for safety objectives (de Alfaro et al. 2007). For hybrid games with ODE dynamics, the situation is OPEN.
- **Effort:** 1–2 person-months for the proof, if it succeeds. Potentially unbounded if it doesn't.

**4. Temporal Flexibility Extraction from Guidelines**

- **WHY hard:** Clinical guidelines specify dosing frequencies ("daily," "twice daily," "every 8 hours") but not exact administration times. Extracting the set of guideline-compliant schedules requires interpreting FHIR TimingRepeat constraints as mathematical constraint sets over administration times. This is a constraint compilation problem — translating clinical timing language into formal timing constraints. FHIR TimingRepeat has 15+ parameters (frequency, frequencyMax, period, periodMax, periodUnit, dayOfWeek, timeOfDay, when, offset, count, countMax, duration, durationMax, durationUnit, event) with complex interactions.
- **Known or unsolved?** KNOWN compilation problem, but the FHIR TimingRepeat spec is complex enough that faithful formalization is non-trivial. Not a research problem — an engineering problem.
- **Effort:** 1–1.5 person-months.

**5. Product-Game Construction for N Guidelines**

- **WHY hard:** Composing N guidelines into a single game arena with shared PK state requires the product of N guideline automata with synchronization on shared enzyme state. This is the same exponential blowup as Approach 1's product automaton, but now with game structure (alternating scheduler/adversary moves). The enzyme-group decomposition (Theorem III) is supposed to address this, but it requires the inter-group scheduling compatibility check to work — which may fail for drugs appearing in multiple enzyme groups.
- **Known or unsolved?** Product-game construction is KNOWN. The enzyme-group decomposition for games is NOVEL and unproven.
- **Effort:** 2–3 person-months.

### B. Architectural Challenges

**Hardest integration points:**
1. **Decidability proof → implementation.** The entire system depends on Theorem I being provable. If the proof fails, there is no graceful degradation — the approach doesn't become "verification with weaker guarantees," it becomes "heuristic scheduling with no formal backing." This is a binary dependency that makes the approach fundamentally riskier than Approaches 1 or 2.
2. **Game solver ↔ PK ODE dynamics.** The game solver must evaluate PK trajectories for each scheduler action × adversary parameter combination. The inner loop is PK ODE evaluation, which must be efficient. For the extremal adversary approach, 2^p ODE evaluations per scheduler step — each must be fast.
3. **Pareto optimization ↔ clinical interpretability.** The Pareto frontier must be presented in a way clinical pharmacists can act on. This requires translating schedule-space tradeoffs into clinical language ("Schedule A maintains warfarin in therapeutic range 95% of the time but atorvastatin only 80%...").

**Where the system most likely breaks under real-world inputs:**
- **Adversary strategy blowup for complex polypharmacy.** For a patient on 15 medications (realistic for elderly polypharmacy), even with enzyme-group decomposition, some groups may have 5+ drugs with 10+ independent PK parameters. 2^10 = 1024 extremal scenarios per group × grid^5 scheduler actions = potentially intractable.
- **Drugs with no temporal flexibility.** Some guidelines specify exact timing ("take with breakfast," "take at bedtime"). If most drugs have fixed timing, the scheduler's optimization space collapses and the game formulation adds complexity without value.
- **Non-competitive interactions.** Same issue as Approach 1 — the adversary extremalization relies on monotonicity that doesn't hold for non-competitive inhibition or induction.

**Critical path dependencies:**
- Theorem I (decidability) → everything. If this proof fails, the project has no foundation.
- Theorem II (Pareto) → Theorem III (composition) → scalable synthesis
- This is the riskiest dependency chain across all three approaches.

### C. LoC Estimates Reality Check

| Subsystem | Claimed LoC (implied) | Realistic LoC | Assessment |
|-----------|----------------------|---------------|------------|
| Temporal flexibility extraction | ~5K | 3–6K | **Plausible.** FHIR TimingRepeat parsing + constraint set construction. |
| Hybrid game construction | ~20K | 15–25K | **Underestimated.** Product-game with ODE dynamics, alternating moves, enzyme synchronization. |
| Strategy synthesis engine | ~25K | 20–35K | **Significantly underestimated.** This is the core algorithm — finite-game solver + continuous-state handling + memory-bounded strategy extraction. |
| Pareto optimization | ~10K | 8–15K | **Plausible.** Multi-objective optimization within the game framework. |
| PK model library (shared) | ~10K | 8–12K | **Plausible.** Same as other approaches. |
| Compositional synthesis (Theorem III) | ~10K | 8–15K | **Underestimated.** Enzyme-group decomposition for games is harder than for verification (must also synthesize compatible schedules). |
| Schedule presentation layer | ~5K | 5–10K | **Underestimated.** Clinically interpretable schedule output with Pareto tradeoff visualization. |
| Evaluation engine | ~12K | 10–18K | **Same benchmark infrastructure issue.** |

**Paper-phase total:** Claimed ~110K. Realistic: **85–140K.** The highest variance of all three approaches.

**Novel code ratio:** Claimed ~70K novel / ~110K total (64%). Realistic: ~55–90K novel / 85–140K total (60–65%).

**Most underestimated subsystems:**
1. Strategy synthesis engine (the algorithmic core is genuinely hard)
2. Hybrid game construction (product-game with ODE dynamics)
3. Compositional synthesis (game-specific decomposition)

### D. Feasibility Assessment

**Can 1–2 engineers build this in 12–16 weeks?**

For the **full 110K LoC**: **Absolutely not.** Even the realistic lower bound of 85K LoC at 200 LoC/day requires ~425 person-days = ~21 person-months. Two engineers × 16 weeks = ~8 person-months — less than half what's needed. And this doesn't account for the proof work on Theorems I–III, which could consume 3–6 person-months of a mathematician's time.

For a **minimum viable artifact**: **Extremely challenging.** The minimum viable version still requires the decidability proof (Theorem I), which is a binary dependency. Without it, the approach has no formal foundation.

**Minimum viable artifact for a best-paper submission:**
- Prove Theorem I (decidability) — this alone could take 2–4 months and may fail
- 2–3 manually encoded guideline pairs with temporal flexibility
- 1-compartment PK models
- Extremal adversary with 2–3 independent PK parameters per group
- Scheduler synthesis over a coarse time grid (2-hour granularity)
- Simple Pareto computation (enumerate small grid, filter dominated schedules)
- Drop Theorem III (compositional synthesis) — verify/synthesize small groups only
- **Estimated scope: ~30–40K LoC, ~20–25K novel — achievable in 12–16 weeks IF the decidability proof succeeds in the first 2–3 weeks**

**What must be cut:**
1. Compositional synthesis (Theorem III) — verify small groups only
2. Memory-bounded strategy proof — output schedules heuristically, claim empirical (not formal) optimality
3. Complex Pareto visualization — text-only schedule output
4. Non-competitive interactions — competitive CYP inhibition only
5. Large-scale evaluation — proof-of-concept on 2–3 guideline pairs only

### E. Difficulty Scores

| Metric | Score | Rationale |
|--------|-------|-----------|
| **Raw Difficulty** | **9/10** | The hardest approach by a significant margin. Hybrid game decidability is a conjecture. Strategy synthesis for continuous-state games is an open research problem. Multi-objective Pareto optimization within a game framework compounds the difficulty. ~70K novel LoC at full scope. The only reason this isn't 10/10 is that the domain-specific structure (Metzler monotonicity) genuinely helps. |
| **Minimum Viable Difficulty** | **7/10** | Even the minimum viable version requires proving a novel decidability result. If the proof succeeds, the implementation is tractable (SAT-based game solving over a coarse grid). If the proof fails, there is no minimum viable version — the approach collapses to heuristic scheduling without formal guarantees. |
| **Risk of Non-Termination** | **8/10** | The decidability proof is a binary dependency. If it fails (and the proposal honestly labels it a conjecture), the project has no foundation. Even if it succeeds, the exponential blowup in adversary strategies (2^p) and scheduler actions (|grid|^D) could make the implementation intractable for realistic inputs. The enzyme-group decomposition (Theorem III) is supposed to address this but is itself unproven. Three stacked conjectures, each dependent on the previous one. |

### F. Red Flags

1. **🚨 Theorem I (decidability of PTG) is a conjecture, not a theorem.** This is the single most critical red flag across all three approaches. The entire approach depends on proving a novel decidability result for hybrid games. If the proof fails — which is entirely possible, since hybrid games with ODE dynamics are in general undecidable — the project has no graceful degradation path. This is fundamentally different from Approaches 1 and 2, where the proof obligations are domain-specific instantiations of known results. Here, the proof obligation is a *novel decidability claim* in a domain (hybrid games) where undecidability is the norm.

2. **🚨 Exponential blowup is baked into the complexity bounds.** The claimed complexity O(D · |grid|^D · 2^p) is exponential in BOTH D and p. The enzyme-group decomposition reduces effective D but is itself unproven (Theorem III). For realistic polypharmacy (D ≥ 5 within a single enzyme group), the computation may be intractable even with decomposition.

3. **Strategy memory requirements are unproven.** The proposal asserts that bounded-memory strategies exist for PTGs with Metzler dynamics but does not prove it. If infinite-memory strategies are required, the synthesized schedules cannot be represented as finite objects — making the primary deliverable (actionable medication schedules) impossible.

4. **"Sounds easy but is actually nightmare": Pareto frontier computation.** Computing the Pareto frontier over a multi-dimensional objective space (one dimension per drug) requires enumerating a potentially exponential number of non-dominated schedules. The claim that the Pareto set has at most D+1 vertices depends on the safety constraint being a single halfspace per drug — but therapeutic windows are *intervals* (bounded above by toxicity, bounded below by sub-therapeutic threshold), doubling the constraint count and potentially shattering the polytope structure.

5. **Clinical actionability may be illusory.** The approach promises "take metformin at 08:00, atorvastatin at 20:00" — but in clinical practice, medication scheduling is constrained by meal timing, sleep schedules, patient compliance patterns, and formulation-specific requirements (enteric coating, sustained release). The "temporal flexibility" extracted from guidelines may be much smaller than expected when real-world constraints are added, potentially leaving no room for optimization.

---

## Cross-Approach Comparison

### Genuine Difficulty vs. Tedium

| Component | Genuinely Hard | Tedious but Not Hard |
|-----------|---------------|---------------------|
| PK abstract domain design | ✓ (Approach 2) | |
| CQL-to-PTA compilation | | ✓ (deferred — compiler engineering) |
| Validated interval ODE integration | ✓ (fragile, debugging-intensive) | |
| Contract extraction from PK models | ✓ (Approach 1 — novel math) | |
| CEGAR convergence tuning | ✓ (empirically hard, unpredictable) | |
| Terminology integration | | ✓ (well-scoped wrapping) |
| FAERS data processing | | ✓ (batch ETL at scale) |
| Evaluation benchmark harness | | ✓ (infrastructure work) |
| Clinical narrative generation | ✓ (deceptively hard NLG) | |
| Hybrid game decidability proof | ✓✓ (Approach 3 — unsolved) | |
| Pareto schedule synthesis | ✓ (exponential in dimension) | |
| Drug discontinuation partitioning | | ✓ (engineering) |
| DrugBank/Beers integration | | ✓ (data wrangling) |

### Which Approach Has the Best Difficulty-Adjusted Return?

| Metric | Approach 1 | Approach 2 | Approach 3 |
|--------|-----------|-----------|-----------|
| Raw Difficulty | 8 | 7 | 9 |
| Minimum Viable Difficulty | 5 | 4 | 7 |
| Risk of Non-Termination | 6 | 4 | 8 |
| **Difficulty-Weighted Feasibility** | **Medium** | **High** | **Low** |

**Approach 2 has the best difficulty profile for a 12–16 week timeline.** The minimum viable artifact is achievable (4/10 difficulty), the risk of non-termination is low (4/10), and the theoretical contributions (three clean theorems) are well-scoped. The main risk (false-positive rate) is a gradient, not a cliff — partial success is always publishable.

**Approach 1 is the safest middle ground** if the E1 temporal ablation gamble is accepted. The minimum viable artifact (5/10) is achievable, and the contract composition theorem is the single strongest mathematical contribution across all three approaches.

**Approach 3 has the highest ceiling but unacceptable risk** for a best-paper-or-nothing timeline. The binary dependency on Theorem I (unproven decidability conjecture) means the project has a ~30–40% chance of complete failure (proof doesn't go through), regardless of engineering effort.

### LoC Estimates Summary

| | Full Vision | Paper Phase | Minimum Viable | Realistic Min. Viable |
|--|-------------|-------------|----------------|----------------------|
| **Approach 1** | ~135K | ~95K | ~25–35K | ~30–40K (with testing) |
| **Approach 2** | ~75K | ~75K | ~20–30K | ~25–35K (with testing) |
| **Approach 3** | ~110K | ~110K | ~30–40K | ~35–50K (with testing) |

The problem statement's 135K estimate for Approach 1's full vision is *in the right ballpark* but likely represents an upper-middle estimate. The 95K paper-phase estimate is optimistic by ~20–30% for what would actually be delivered in 16 weeks. The minimum viable artifacts for all three approaches are significantly smaller than the paper-phase estimates, which is appropriate — a best-paper submission needs a compelling artifact, not a complete system.

### Final Assessment

The genuinely hard problems in this project cluster around three areas:

1. **Novel formalisms with domain-specific decidability** (PTA δ-decidability, PTG decidability, PK Galois connection). These are mathematical challenges where the outcome is binary — the proofs either work or they don't.

2. **Precision under abstraction** (CEGAR convergence, abstract domain false-positive rate, widening precision). These are engineering challenges where the outcome is a gradient — you can always get *some* precision, but useful precision is not guaranteed.

3. **Validated numerical computation** (interval ODE, zonotopic reachability, PK parameter sensitivity). These are implementation challenges where the difficulty is consistently underestimated by ~2–3×.

For a best-paper submission, the critical insight is that **mathematical novelty matters more than system completeness.** A clean theorem with a small but convincing prototype beats a large system without theoretical depth. Approach 2's three clean theorems + speed demonstration may be more best-paper-competitive than Approach 1's larger system with a high-variance E1 experiment, even though Approach 1's contract composition theorem is individually stronger.
