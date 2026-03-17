# Depth Check: BiCut — Bilevel Optimization Compiler

**Panel:** 3-expert verification team (Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer)  
**Method:** Independent scoring → adversarial cross-critiques → consensus synthesis  
**Document Under Review:** `crystallized_problem.md` (BiCut, slug: `bilevel-compiler-intersection-cuts`)  
**Date:** 2026-03-08

---

## Scoring Summary

| Axis | Auditor | Skeptic | Synthesizer | **Consensus** | Notes |
|------|---------|---------|-------------|---------------|-------|
| 1. Extreme & Obvious Value | 4 | 3 | 5 | **4** | Small user base caps value |
| 2. Genuine Difficulty | 6 | 5 | 7 | **6** | 3–5 hard subproblems in large routine infrastructure |
| 3. Best-Paper Potential | 5 | 4 | 6 | **5** | Bilevel intersection cuts are real new math, but narrow corridor |
| 4. Laptop CPU + No Humans | 8 | 7 | 8 | **8** | All computation CPU-native; zero annotation |
| **Total** | **23** | **19** | **26** | **23/40** | |

---

## Axis 1: Extreme and Obvious Value — 4/10

### What's good
- **"Compile once, solve anywhere"** with .mps output is genuinely useful for reproducibility in a field with no common bilevel interchange format.
- **Correctness certificates** are a first — no existing bilevel tool (BilevelJuMP, PAO, GAMS EMP, YALMIP, MibS) provides machine-checkable proofs that a reformulation preserves bilevel optimality.
- **Bilevel intersection cuts** have no substitute — this is new cutting-plane technology unavailable anywhere.
- The pain is real: manual reformulation derivation, big-M debugging, and solver-specific coding take weeks per problem.

### What's missing for "extreme"
- **User base is tiny: 50–500 people.** The proposal documents 50–200 MIBLP researchers and 200–500 applied bilevel practitioners. Even at the ceiling, this is three orders of magnitude smaller than CVXPY's user base. The CVXPY analogy is structurally flawed — CVXPY succeeded because convex optimization has millions of downstream users (ML, finance, control). Bilevel optimization is a niche within a niche.
- **Existing tools cover the 80% case.** BilevelJuMP.jl (INFORMS JOC 2023) handles KKT→MPEC with per-constraint mode selection. PAO offers FA, PCCG, REG solvers. MibS handles MILP lower levels natively. GAMS EMP does specification→reformulation→dispatch. BiCut's delta is: automatic strategy-level selection, correctness certificates, and novel cuts. Only the cuts are irreplaceable.
- **Industrial adoption is unlikely.** The adversarial critique is devastating: "Industrial users already have custom implementations tuned to their specific problem structures. They will not switch to a general-purpose compiler." Energy companies, defense contractors, and logistics firms that use bilevel optimization have custom in-house tools.
- **LLM erosion.** An LLM can generate KKT reformulations for specific bilevel problems from a specification. The compiler's convenience value erodes for one-off formulations.
- **The reproducibility argument is real but modest.** The Synthesizer argued that reproducibility certificates expand the effective user base 5–10×. The Auditor countered: "A reviewer who can evaluate bilevel reformulation correctness is exactly the kind of expert who doesn't need a machine-checkable certificate." The marginal gain is perhaps 100–200 additional users, not 2,500.

### What would raise this to 7
- Demonstrate the "CVXPY effect": a prototype showing that lowering the barrier creates new demand from OR researchers who currently avoid bilevel optimization.
- Broaden scope beyond MIBLPs to QP/conic lower levels, reaching a wider practitioner base.
- Reframe BiCut explicitly as a **bilevel reproducibility/verification layer**, not just a performance tool — this provides value even when cuts are modest.

---

## Axis 2: Genuine Difficulty as Software Artifact — 6/10

### What's genuinely hard
- **Intersection cut engine (S5, 11K LoC, 4K novel):** Characterizing bilevel-infeasible sets via value-function epigraph geometry and building a separation oracle with >90% cache hit rate — no off-the-shelf solution.
- **Value-function oracle (S6, 10K LoC, 3.7K novel):** Parametric LP for continuous lower levels, sampling-based MILP approximation with error bounds for integer lower levels, Gomory-Johnson lifting. Exact evaluation is limited to ~15–20 upper-level variables.
- **Big-M computation (S4):** Automated bound-tightening via auxiliary LPs with numerical soundness across three solvers. Subtle — a single tolerance mistake corrupts all downstream results.
- **Three-backend callback emission (S8):** Gurobi `cbLazy()`, SCIP `CONSHDLR` plugins (C callbacks with thread-safety requirements), HiGHS iterative MPS fallback. No existing bilevel tool has attempted this.
- **Sound CQ verification (S2, S9):** Conservative approximation hierarchy for a co-NP-hard problem that must be sound without being so conservative it rejects most instances.

### What limits the score
- **Novel logic is ~15K LoC out of 105K total.** The adversarial critique estimates 30–50K of genuinely challenging code; the rest is standard compiler infrastructure (AST, expression DAGs, serialization, logging, CLI, Python bindings, test harnesses, benchmark loading).
- **The 150K LoC threshold is met only via deferred extensions.** The scoped system is 105K — 30% below threshold. The 155K "full vision" includes 45K of extensions (QP, conic, pessimistic, multi-follower, CPLEX, regularization) explicitly listed as non-goals.
- **The "4 engineering breakthroughs" are heterogeneous.** Only S5 and S6 require genuine algorithmic invention. CQ verification (S2) is conservative approximation over known theory. Solver emission (S8) is painful integration work, not research.
- **Infrastructure is standard.** CVXPY, Pyomo, and JuMP all have typed IRs, backend emission, and test infrastructure. Much of BiCut's 90K of infrastructure is the same kind of work every DSL-to-solver pipeline does.

### What would raise this to 7
- Bring QP lower levels (E1, +12K LoC) and CPLEX backend (E5, +4K LoC) into evaluated scope, pushing the scoped system to ~121K and reducing the LoC gap.
- Demonstrate that the intersection cut separation oracle and value-function oracle require novel algorithmic solutions (not just known parametric programming from the 1970s applied to a new setting).

---

## Axis 3: Best-Paper Potential — 5/10

### What supports best-paper
- **Bilevel intersection cuts are genuinely new polyhedral theory.** Extending Balas 1971 to optimality-defined infeasible sets is unexplored territory. The facet-defining conditions, separation complexity analysis, and finite convergence proof are new results that bridge the cutting-plane theory and bilevel optimization communities.
- **"Creates a new software category"** is the strongest narrative argument. If BiCut achieves for bilevel optimization what CVXPY/CVX did for convex optimization, the historical significance is clear.
- **Clean, falsifiable benchmark story.** BOBILib (2600+ instances) + MibS baseline + standard metrics (root gap closure, solve time, node count) = no ambiguity.
- **IJOC is the natural home** and would value both the cutting-plane theory and the systems contribution.

### What undermines best-paper
- **The narrow viability corridor is the critical risk.** The adversarial critique identifies: "If the bilevel-infeasible convex set turns out to have a simple closed-form for LP lower levels, then the intersection cut derivation is mechanical." Conversely, if separation is too hard, cuts are impractical. The entire mathematical contribution exists in a narrow band between triviality and intractability, and **no empirical evidence validates that this corridor is navigable.**
- **The 15–25% gap closure claim has zero empirical backing.** It is calibrated against GMI cuts for pure IPs — a structurally different setting. The Skeptic argues that bilevel infeasibility is continuous (value-function violation), not combinatorial (integrality violation), making the GMI analogy unreliable. If actual gap closure is 5–10%, the paper becomes "we tried Balas on bilevel with modest results."
- **Value-function lifting (T1.2) is speculative.** Extending Gomory-Johnson to a function (V(x)) that cannot even be efficiently evaluated for MILP lower levels makes the lifting theory difficult to instantiate computationally.
- **Math Programming vs. IJOC tension.** Without the facet-defining proof, the contribution is computational rather than theoretical. The compiler soundness theorem (T1.3, Difficulty B) is "a straightforward case analysis" per the adversarial critique. Math Programming requires deep, self-contained mathematical contributions; IJOC values systems contributions. Best-paper at IJOC is achievable; best-paper at Math Programming is unlikely.
- **No preliminary empirical signal.** The gap closure target, the cache hit rate assumption, and the speedup claims are all theoretical projections with zero validation.

### What would raise this to 7
- **Prove facet-defining conditions** for bilevel intersection cuts. This elevates the contribution from "applied Balas" to "new facet theory for bilevel polyhedra" — the difference between a computational study and a theoretical contribution.
- **Conduct a 2-week prototype experiment** on 50+ BOBILib instances to validate gap closure. If ≥15%, the paper stands. If <5%, pivot early.
- **Commit to explicit go/no-go thresholds:** gap closure ≥10% on ≥30% of LP-lower-level instances, separation oracle overhead <50ms on 90% of callbacks, cache hit rate ≥80%.

---

## Axis 4: Laptop CPU + No Humans — 8/10

### What works
- **Compilation is entirely symbolic** — parsing, structural analysis, reformulation selection, and solver emission require zero numerical optimization. Timing: <100ms parsing, <1s analysis, <10ms selection, <500ms lowering, <200ms emission.
- **Cut generation uses small auxiliary LPs** (lower-level dimension, typically <100 constraints) with warm-starting and caching.
- **BOBILib instances up to ~500 variables** are solvable in minutes on a modern laptop. The benchmark infrastructure is pre-existing (2600+ instances with known optimal values).
- **Zero human annotation.** All evaluation is fully automated. BOBILib instances are pre-existing. Bilevel feasibility verification is automated.
- **No GPU at any stage.** Explicitly listed as a non-goal.

### Minor concerns
- **Large instances (>500 vars)** may require hours per instance on laptop CPU. The full BOBILib suite across 3 backends × multiple configurations could require days of wall-clock time — CPU-feasible but slow.
- **Value-function oracle scalability:** Exact parametric LP evaluation limited to ~15–20 upper-level variables. Beyond this, sampling-based approximation is used.
- **"Fast compilation, slow solving" paradox:** A KKT reformulation of a 5000-constraint lower level produces a 5000-binary MILP that may be intractable on laptop even with Gurobi. BiCut compiles in milliseconds but the output may take hours to solve.

### Verdict
All concerns are about scale, not about fundamental hardware requirements. The system is designed for laptop CPU and requires no GPU or human involvement. Score 8 is appropriate.

---

## Axis 5: Fatal Flaws

### Flaw 1: Narrow Viability Corridor for Intersection Cuts (CRITICAL)

The bilevel intersection cut contribution — the mathematical crown jewel — exists in a narrow band between triviality and intractability. If the bilevel-infeasible set has a simple closed-form for LP lower levels, the cut derivation is mechanical (applying Balas to one more class). If separation requires solving hard auxiliary problems, cuts are too expensive to be practical.

**No evidence is presented that this corridor is navigable.** The adversarial critique's Challenge 3 ("quantify gap closure on ≥50 instances") is identified as "the single most important empirical question in the entire project" — and it remains unanswered. The 15–25% gap closure claim is calibrated on GMI cuts from a structurally different domain.

**Mitigation:** A 2-week prototype experiment on BOBILib instances with LP lower levels would validate or kill the cutting-plane contribution. This MUST happen before full implementation.

### Flaw 2: Value-Function Oracle Collapse Risk (HIGH)

The adversarial critique identifies: "If the cache hit rate drops below ~90%, the system collapses." Parametric LP is limited to ≤15–20 upper-level variables. MILP value functions are discontinuous with no closed-form.

If the oracle is slow, BOTH novel contributions (intersection cuts AND value-function lifting) become impractical — they share the same oracle dependency, creating correlated failure risk.

**Mitigation:** Define explicit performance thresholds: separation oracle overhead <50ms on 90% of callbacks, cache hit rate ≥80%. If missed, the compiler still provides value via reformulation selection and certificates — the cut contributions degrade gracefully.

### Flaw 3: LoC Gap (MODERATE)

The scoped system is 105K; the threshold is 150K. The gap is bridged only by deferred extensions. The load-bearing code is ~81K.

**Mitigation:** Bring QP lower levels (E1, +12K) and CPLEX backend (E5, +4K) into evaluated scope, pushing to ~121K. Provide detailed overhead accounting for the remaining gap.

### Flaw 4: Tiny User Base (MODERATE)

50–500 direct users. The "CVXPY effect" (lowering barriers creates new demand) is plausible but unproven. Industrial adoption is unlikely. The realistic outcome is a highly cited paper, not a widely used tool.

**Mitigation:** Reframe explicitly as a reproducibility/verification infrastructure layer for computational optimization research, not a tool competing with BilevelJuMP for daily use.

### Flaw 5: No Preliminary Empirical Signal (MODERATE)

Gap closure, cache hit rates, speedup claims — all theoretical projections with zero validation. The entire project rests on the plausibility of cutting-plane effectiveness with no empirical evidence.

**Mitigation:** The 2-week prototype experiment (Flaw 1 mitigation) also addresses this flaw.

---

## Consensus Amendments (All 3 Experts Agree)

### Amendment 1 (BINDING): Prove Facet-Defining Conditions
Elevate T1.1 from computational heuristic to polyhedral theory. The intersection cut contribution MUST include facet-defining conditions for bilevel intersection cuts, not just a separation procedure. This is the difference between IJOC and IPCO, between "applied Balas" and "new facet theory." Add this as an explicit deliverable in §4 T1.1.

### Amendment 2 (BINDING): 2-Week Prototype Gate
Before full implementation, conduct a prototype experiment: implement the intersection cut separation oracle on ≥50 BOBILib instances with LP lower levels. Measure root gap closure. If <5% geometric mean, the cutting-plane contribution should be descoped and the project repositioned as a pure compiler/verification tool. Add this as a go/no-go gate in the evaluation plan.

### Amendment 3 (BINDING): Explicit Go/No-Go Thresholds
Define measurable criteria in the evaluation plan:
- Gap closure ≥10% geometric mean on ≥30% of LP-lower-level BOBILib instances
- Separation oracle overhead <50ms on 90% of callbacks
- Cache hit rate ≥80% across cut rounds
- If any threshold is missed, document the pivot strategy

### Amendment 4 (RECOMMENDED): Bring QP Lower Levels + CPLEX Into Scope
Move E1 (QP Lower Levels, +12K LoC) and E5 (CPLEX Backend, +4K LoC) from non-goals to evaluated scope. This pushes the scoped system from 105K to ~121K LoC, partially closing the LoC gap, and broadens the applicability corridor beyond LP-lower-level MIBLPs.

### Amendment 5 (RECOMMENDED): Reframe as Reproducibility Layer
Position BiCut explicitly as bilevel reproducibility/verification infrastructure, not primarily a performance tool. This provides value even when intersection cuts are modest, broadens the audience to the wider computational optimization community, and hedges against the risk that gap closure is below target.

### Amendment 6 (RECOMMENDED): Widen Gap Closure Range
Replace "15–25% root gap closure" with "10–30% root gap closure (with 10% as the go/no-go floor)" to honestly reflect uncertainty. The original calibration against GMI cuts is from a structurally different domain.

---

## Overall Verdict

### CONDITIONAL CONTINUE (3-0 consensus)

**Rationale:** BiCut is a competent, honest proposal with genuinely novel mathematics in the bilevel intersection cuts. The compiler architecture is sound, the evaluation plan is clean and falsifiable, and the system is fully CPU-feasible. However, the proposal is undermined by a tiny user base (capping value at 4/10), unvalidated empirical claims at the core of the contribution (capping best-paper at 5/10), and a narrow viability corridor for the crown-jewel math contribution.

**The single most important action is Amendment 2:** a 2-week prototype experiment to validate or kill the 15–25% gap closure claim. This experiment determines whether BiCut is a paper about new polyhedral theory (gap closure ≥15%) or a paper about compiler infrastructure with modest cuts (gap closure <10%). Both papers are publishable; only the first has best-paper potential.

**Conditions for full approval:**
1. Amendments 1–3 (binding) are incorporated into the problem statement.
2. The prototype experiment (Amendment 2) shows gap closure ≥10% on ≥30% of test instances.
3. If the prototype fails, the pivot to pure compiler/verification tool is documented.

**Kill condition:** If the prototype shows gap closure <5% AND the value-function oracle cache hit rate is <70%, the cutting-plane contribution should be abandoned entirely. The compiler alone (reformulation selection + certificates + solver-agnostic emission) is a solid IJOC paper but scores ~18/40 on this rubric and would not merit best-paper consideration.

---

*Assessment by 3-expert verification panel. Consensus reached after independent scoring, adversarial cross-critiques, and synthesis.*
