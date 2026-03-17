# BiCut Difficulty Assessment: Honest Evaluation of Three Approaches

**Reviewer role:** Difficulty Assessor  
**Date:** 2025-07-18

---

## Approach A: "Cuts-First" — Deep Cutting Plane Theory

### 1. What's Genuinely Hard (3–5 hardest subproblems)

1. **Bilevel-infeasible set polyhedrality + facet-defining proof (Novel algorithmic design).** This is the hardest problem across all three approaches. You must characterize the set of (x,y) pairs where y is *suboptimal* for the follower — a set defined by optimality failure, not by algebraic constraints. Proving this set is a finite union of polyhedra (for LP lower levels) requires connecting lower-level dual vertex enumeration to the geometry of the bilevel-feasible set's convex hull. The facet-defining conditions require constructing dim(P) affinely independent feasible points on the cut hyperplane. This is genuinely novel polyhedral theory with no close precedent. The Balas 1971/1979 proofs handled disjunctions defined by integrality — a fundamentally simpler structure than optimality-defined infeasibility. **Verdict: Genuinely hard. This is the kind of problem where you might work for 6 months and fail.**

2. **Separation oracle with parametric warm-starting and >90% cache hit rate (Novel engineering).** Each separation call solves an auxiliary LP. Making this practical (<50ms per call) requires exploiting the fact that consecutive LP relaxation points in a branch-and-bound tree differ by small perturbations. You need basis tracking, perturbation detection, and cache invalidation — essentially building a custom parametric LP solver layer. No off-the-shelf LP solver API exposes the right hooks for this. **Verdict: Hard systems engineering with novel design decisions, but the difficulty is bounded — you're optimizing a known algorithmic pattern, not inventing one.**

3. **Value-function oracle for MILP lower levels (Novel algorithm).** Exact evaluation of V(x) for integer lower levels is NP-hard per query. The sampling-based piecewise-linear overestimator with provable error bounds is a real algorithmic contribution. The Gomory–Johnson lifting step — extending subadditivity theory from integer lattices to polyhedral domains — is mathematically nontrivial. **Verdict: Genuinely hard, though the LP lower-level case (exact parametric LP) is tractable. The MILP case is where the difficulty concentrates.**

4. **Finite convergence proof under bilevel non-degeneracy (Hard but precedented).** Standard for cutting-plane methods, but the bilevel feasibility structure makes the proof non-routine. The key difficulty is handling the interaction between the cut loop's convergence and the bilevel-infeasible set's geometry. **Verdict: Hard but achievable by someone who knows both cutting-plane theory and bilevel optimization.**

### 2. What's Routine

- **Parser (~3K LoC):** Standard recursive descent. Not even interesting.
- **Value-function reformulation pass (~4K LoC):** Known theory from Outrata et al. 1998. Textbook implementation.
- **MPS emission (~2K LoC):** Standard file format. A weekend's work.
- **SCIP callback integration (~3K LoC):** Well-documented API with many examples in SCIP's own test suite. Tedious but not hard.
- **BOBILib loader (~2K LoC):** File parsing.

The approaches document correctly identifies these as routine. No inflation here.

### 3. Novel LoC Estimate

| Component | Claimed Novel LoC | Honest Estimate | Rationale |
|-----------|-------------------|-----------------|-----------|
| Intersection cut separation oracle | ~4,000 | ~3,000–3,500 | The polyhedral characterization and ray-intersection logic is genuinely novel. But ~500–1,000 LoC is boilerplate LP setup, result parsing, and data structures that any experienced optimization programmer has written before. |
| Value-function oracle + Gomory lifting | ~3,700 | ~2,500–3,000 | The parametric LP solver interface and sampling grid are standard numerical computing. The genuinely novel parts are the error bound derivation, the piecewise-linear overestimator construction, and the subadditivity lifting. |
| Supporting infrastructure | ~6,300 | ~1,500–2,000 | Parser, reformulation pass, emission, callbacks — all standard. |
| **Total** | **~14,000** | **~7,000–8,500** | About half of the claimed "novel" code is standard optimization infrastructure dressed up as novel. |

The genuinely novel algorithmic logic is probably **7,000–8,500 LoC** — still substantial, but the document inflates by counting standard LP/MIP interface code as novel.

### 4. Architectural Risk

**Low architectural risk, high mathematical risk.** The system is deliberately narrow (one solver, one reformulation, one cut family), which makes the architecture trivially coherent. There are no coupling problems because there's almost nothing to couple. The risk is entirely mathematical: if the polyhedrality theorem doesn't hold cleanly, or if facet-defining conditions require case analysis too complex to prove in general, the entire approach collapses. The architecture can't save you from bad math.

**One lurking issue:** Locking into SCIP-only means the separation oracle's performance is coupled to SCIP's branch-and-bound implementation. If SCIP's node processing order interacts badly with the cut cache (e.g., SCIP jumps around the tree more than expected, killing cache hit rates), there's no fallback without significant rework.

### 5. Implementation Timeline Risk

- **Facet-defining proof:** 2–6 months, with non-trivial probability of failure. This is research, not engineering — you can't schedule a proof.
- **Separation oracle + cache:** 2–3 months for a competent systems programmer with optimization background.
- **Value-function oracle (LP case):** 1–2 months.
- **Value-function oracle (MILP case + Gomory lifting):** 2–4 months, partly concurrent with the proof work.
- **Infrastructure (parser, reformulation, emission, benchmarking):** 1–2 months.

**Realistic total: 6–12 months for a strong 2-person team.** The variance is dominated by the proof work. A team that gets lucky with the math could finish in 6 months; a team that struggles could burn 12 months and still not have the facet-defining conditions.

### 6. Difficulty Score (Honest): **8/10**

The claimed 9/10 is slightly inflated. The separation oracle engineering, while challenging, has clear design patterns from parametric LP literature. The value-function oracle for LP lower levels is hard but precedented. The genuine 9–10 difficulty concentrates in exactly one place: the facet-defining proof and the polyhedrality theorem. If you remove the proof requirement and settle for "computationally effective cuts without full theoretical characterization," this drops to a 6. The difficulty is extremely concentrated.

---

## Approach B: "Compiler-First" — Full Compiler Architecture

### 1. What's Genuinely Hard (3–5 hardest subproblems)

1. **Sound CQ verification with conservative three-tier approximation (Moderately novel algorithm).** Verifying LICQ/Slater for parametric lower-level programs is co-NP-hard. Building a three-tier hierarchy (syntactic → LP-based → sampling-based) that is *sound* (no false positives) while rejecting <20% of valid instances requires careful algorithm design. The LP-based rank test for LICQ is standard, but calibrating the rejection rate across the full BOBILib instance set — without being so conservative that you reject everything — requires empirical tuning with theoretical backing. **Verdict: Genuinely challenging design problem, but each tier individually uses known techniques. The novelty is in the composition and the soundness guarantee.**

2. **Four-backend solver emission with semantic preservation (Painful systems engineering).** This is the subproblem the approaches document identifies as hardest, and I partly disagree. The *design* is standard — strategy pattern, capability profiles, per-solver encoders. The *effort* is large because you're learning four different solver APIs and their idiosyncrasies. But there's no novel algorithm here. You're not inventing indicator constraint encoding; you're mapping a known mathematical construct to four different API calls. **Verdict: High effort, moderate difficulty. A senior developer who knows one solver API well can learn the others in weeks. The cross-solver verification testing is the truly painful part, not the emission logic itself.**

3. **Automatic big-M computation with numerical soundness (Hard engineering).** Bound-tightening via auxiliary LPs is known (Pineda & Morales 2019). The hard part is handling edge cases: unbounded duals, near-zero reduced costs, solver-dependent feasibility tolerances. A single numerical error corrupts all downstream results, and the failure mode is silent (you get a wrong answer, not a crash). **Verdict: Genuinely hard in practice, but the difficulty is debugging and testing, not algorithm design. An experienced numerical optimization engineer has seen all these failure modes before.**

4. **Reformulation selection with cost model (Moderately novel).** Mapping structural signatures to ranked strategies with soundness guarantees. The soundness part (never select an invalid reformulation) is straightforward case analysis. The performance prediction part (which valid reformulation will be fastest) requires empirical calibration against BOBILib, which is standard ML-for-optimization work. **Verdict: The soundness proof is easy. The cost model is a calibration exercise, not a novel algorithm. The composition rules for chained passes are the only genuinely interesting design problem here.**

5. **QP lower-level reformulation (Known with engineering challenges).** McCormick envelopes for bilinear KKT terms are textbook (McCormick 1976, Al-Khayyal & Falk 1983). The SOCP reformulation for convex QP lower levels is also known. The engineering challenge is integrating these with the compiler's big-M computation and making the McCormick bounds tight. **Verdict: Standard theory, moderate implementation effort.**

### 2. What's Routine

- **Parser, expression DAG, CLI, Python bindings, logging:** Standard software infrastructure. ~20K LoC of completely standard code.
- **MPS/LP file writing:** Trivial.
- **BOBILib instance loader:** File parsing.
- **KKT reformulation pass (for LP/continuous lower levels):** Known theory, straightforward implementation. The novelty is only in the *automation* — hand-derivation of KKT for specific problems is a textbook exercise.
- **Strong duality reformulation:** Known theory from Fortuny-Amat & McCarl 1981. Implementation is mechanical.
- **Value-function reformulation:** Known theory from Outrata et al. 1998.
- **Column-and-constraint generation:** Known algorithm, standard implementation.

**The approaches document is mostly honest here**, but I'd add: the correctness certificate *format* design (JSON/protobuf schema for machine-checkable certificates) is routine software engineering, even though the *content* of the certificates involves non-trivial mathematical verification.

### 3. Novel LoC Estimate

| Component | Claimed Novel LoC | Honest Estimate | Rationale |
|-----------|-------------------|-----------------|-----------|
| Reformulation selection engine | 1,100 | 600–800 | The cost model is essentially a lookup table with regression-calibrated weights. The soundness proof is a few pages of math, not code. |
| CQ verification (three-tier) | 900 (within S2) | 700–900 | The LP-based tier is genuinely novel in composition. Fair estimate. |
| Big-M computation | ~500 (within S4) | 400–500 | Bound-tightening LP setup is standard; the numerical edge-case handling is the novel part. |
| Four-backend emission | 800 | 300–400 | The per-solver encoding strategies use known techniques. The "novel" part is the capability-profile abstraction and the automated encoding selection — a design pattern, not an algorithm. |
| Certificate system | 700 | 500–600 | The verification logic is genuinely novel; the serialization and format are not. |
| QP extension | 4,000 | 1,500–2,000 | McCormick and SOCP reformulations are known. The novel part is the automated tightening and integration with the compiler pipeline. |
| Intersection cuts (basic, no facet theory) | ~4,000 | ~2,000 | Without facet-defining conditions, you're implementing Balas's procedure with a bilevel-feasibility check. Still novel but much less so than Approach A's full theory. |
| **Total** | **~19,300** | **~6,000–7,200** | The honest novel LoC is about a third of the claimed figure. Most of the 121K system is standard compiler/optimization infrastructure. |

### 4. Architectural Risk

**Moderate.** The pass-manager architecture is well-precedented (LLVM, CVXPY), but there are real coupling risks:

- **Reformulation pass ↔ emission layer coupling:** Each reformulation produces different MILP structures (KKT has complementarity binaries, strong duality has dual variables, value-function has oracle callbacks). The emission layer must handle all of these, meaning the backend abstraction is leaky — you can't emit a value-function reformulation to HiGHS (no callback API) without falling back to an iterative MPS approach that changes the solve semantics.
- **Certificate system ↔ everything coupling:** Certificates must reference specific reformulation preconditions, structural analysis results, and solver-specific encoding decisions. This creates a transitive dependency from certificates through every other subsystem.
- **Scale of testing:** 2,600 instances × 4 solvers × 4 reformulations = 41,600 configurations. The combinatorial testing burden is the real architectural risk — not the design, but the validation.

### 5. Implementation Timeline Risk

- **IR + parser + structural analysis:** 2–3 months.
- **Reformulation passes (KKT, SD, VF, C&CG):** 2–3 months.
- **Four-backend emission:** 2–3 months (parallelizable across developers).
- **Certificates + CQ verification:** 1–2 months.
- **QP extension:** 1–2 months.
- **Basic intersection cuts (no facet theory):** 1–2 months.
- **Benchmark harness + full evaluation:** 1–2 months.
- **Integration testing + debugging:** 2–3 months (this is where schedule overruns live).

**Realistic total: 8–14 months for a 3–4 person team.** The work parallelizes well (backends are independent, reformulation passes are independent). The risk is in integration — getting all the pieces to work together across 41,600 configurations. No individual component is a schedule risk; the system-level integration is.

### 6. Difficulty Score (Honest): **4/10**

The claimed 6/10 is inflated. No single subproblem in Approach B requires a novel algorithm. Every component builds on known techniques: KKT reformulation, strong duality, value-function, McCormick envelopes, SOCP, bound-tightening LPs, solver APIs. The difficulty is in *scale* (many components, many solvers, many test configurations) and *correctness* (the certificate system must be sound), not in *novelty*. This is a large, carefully-engineered software system — closer to building a well-designed web framework than to proving a new theorem. The "basic" intersection cuts (Balas adaptation without facet theory) add some novelty, pushing it to 4 rather than 3. But calling this a 6 conflates effort with difficulty.

---

## Approach C: "Hybrid" — Compiler + Cuts as Co-Primary Contributions

### 1. What's Genuinely Hard (3–5 hardest subproblems)

1. **Cut engine ↔ compiler integration with bidirectional space mapping (Novel systems design).** This is the subproblem unique to Approach C and it's legitimately hard. The cut engine operates in the *reformulated* MILP space, but bilevel infeasibility is defined in the *original* bilevel space. You need a bidirectional mapping that depends on which reformulation was applied: KKT requires extracting primal-dual pairs, strong duality requires reconstructing follower solutions from duals, value-function requires oracle evaluation. The `BilevelAnnotation` interface is a clean design, but implementing it correctly for all four reformulation strategies — and testing that the mapping preserves feasibility semantics — is a genuine integration challenge with no precedent. **Verdict: Genuinely hard. Not because any single piece is novel, but because the composition of reformulation-dependent space mappings with a generic cut engine is an unexplored design space.**

2. **The same facet-defining proof as Approach A.** If pursued. Same difficulty, same risk.

3. **Certificate composition across three layers (Novel formalism).** Composing structural analysis certificates, reformulation certificates, and cut validity certificates into an end-to-end guarantee is a real formal methods problem. The closest precedent (proof-carrying code) operates in a much simpler domain. The challenge is that cut validity depends on the reformulation, and the reformulation's correctness depends on structural analysis — so the composition is not a simple conjunction but a chain of conditional guarantees. **Verdict: Genuinely novel formal methods work, though the actual implementation is modest in LoC (~500–800 lines of certificate logic). The difficulty is in the design, not the code.**

4. **Reformulation-aware cut selection (Novel algorithm).** Different reformulations produce different LP relaxation geometries. Adapting the separation strategy to exploit reformulation-specific structure (complementarity constraints in KKT, dual feasibility in strong duality) is a genuinely novel algorithmic idea with no precedent. **Verdict: Novel but bounded in scope — this is a decision layer, not a core algorithm. The implementation is probably ~300–500 lines of selection logic.**

5. **All of Approach B's engineering challenges at full scale.** Four-backend emission, big-M computation, CQ verification, QP extension. Same difficulty as Approach B.

### 2. What's Routine

Same as Approach B: parser, DAG, MPS, BOBILib loader, CLI, Python bindings, logging, individual reformulation passes (KKT, SD, VF, C&CG). The approaches document correctly identifies these.

**Additional routine items the document overstates:**
- The "scalable benchmark evaluation" (80,000+ configurations) is a parallelization and orchestration problem, not a difficulty problem. This is what CI/CD systems and batch job schedulers are for.
- The `BilevelAnnotation` interface *design* is routine software engineering (strategy pattern + visitor). The *content* of each reformulation's annotation implementation is the hard part.

### 3. Novel LoC Estimate

| Component | Claimed Novel LoC | Honest Estimate | Rationale |
|-----------|-------------------|-----------------|-----------|
| Intersection cut engine (full, with facet theory) | 4,000 | 3,000–3,500 | Same as Approach A estimate. |
| Value-function oracle + Gomory lifting | 3,700 | 2,500–3,000 | Same as Approach A estimate. |
| Cut-compiler integration (BilevelAnnotation, bidirectional mapping) | ~1,000 (implicit) | 800–1,200 | Genuinely novel integration logic. |
| Reformulation-aware cut selection | ~500 (implicit) | 300–500 | Novel but small. |
| Certificate composition | ~500 (within S9) | 400–600 | Novel formal methods logic. |
| Compiler infrastructure (selection, CQ, big-M, emission) | ~5,000 | ~2,500–3,500 | Same as Approach B estimate. |
| QP extension | 4,000 | 1,500–2,000 | Same as Approach B estimate. |
| **Total** | **~19,300** | **~11,000–13,800** | More genuinely novel code than either A or B alone, because the integration layer is itself novel. |

### 4. Architectural Risk

**Highest of the three approaches.** The tension between modularity (cuts as an optional pass) and deep integration (cuts must understand reformulation structure) is real and architecturally hazardous:

- **The `BilevelAnnotation` interface is a leaky abstraction.** Each reformulation produces a structurally different MILP. The annotation must expose enough reformulation-specific detail for the cut engine to work, but not so much that adding a new reformulation requires rewriting the cut engine. Getting this interface right requires iterating across all four reformulations, which means you can't design it upfront — it emerges from implementation.
- **Certificate composition creates a transitive fragility.** A bug in any reformulation's certificate logic can silently invalidate the end-to-end guarantee. Testing this requires exhaustive coverage of (reformulation × cut strategy × instance structure) triples.
- **The cut engine's performance depends on the reformulation.** If KKT reformulations produce LP relaxations where the separation oracle is slow (because complementarity big-M constraints create degenerate LPs), the cuts may work beautifully for value-function reformulation but fail for KKT — making the system's performance highly configuration-dependent.

### 5. Implementation Timeline Risk

- **Everything from Approach B:** 8–14 months for the compiler.
- **Everything from Approach A's cut engine:** 4–8 months for the full cut theory + implementation.
- **Integration layer (BilevelAnnotation, certificate composition, reformulation-aware selection):** 2–3 months.
- **But** significant parallelism is possible: compiler backbone and cut engine can be developed concurrently, with integration as a final phase.

**Realistic total: 10–16 months for a 4–5 person team.** The critical path runs through the facet-defining proof (if pursued) and the integration phase. The integration phase cannot begin until both the compiler and cut engine are substantially complete, creating a waterfall dependency at the end.

**Schedule risk is the dominant concern.** This is the most ambitious system, and the integration phase (where the cut engine meets the compiler) is where unknown unknowns live. Budget 3 months for integration and debugging; plan for 5.

### 6. Difficulty Score (Honest): **7/10**

The claimed 8/10 is slightly inflated. The integration challenges (bidirectional mapping, certificate composition, reformulation-aware selection) are genuinely novel but bounded in scope — each is ~500–1,200 LoC of design-intensive code. The bulk of the difficulty comes from inheriting Approach A's hard math (which I rated 8/10) and diluting it with Approach B's engineering (which I rated 4/10). The weighted combination lands at 7. If the facet-defining proof is descoped (settling for computational cuts without full theory), this drops to 5–6.

---

## Cross-Approach Comparison

### Most Concentrated Difficulty: Approach A

Approach A has **2 genuinely hard subproblems** (facet-defining proof, separation oracle design) that account for ~80% of the total difficulty. Everything else is routine. This is a "two peaks and a plain" difficulty profile. If you can solve the two hard problems, the rest is straightforward. If you can't, you have nothing.

### Most Distributed Difficulty: Approach B

Approach B has **5–6 medium subproblems** (CQ verification, big-M computation, four-backend emission, QP extension, reformulation selection, basic intersection cuts) with no single component being genuinely hard. The difficulty is in doing all of them correctly and making them work together. This is a "rolling hills" difficulty profile — no cliffs, but a long march.

### Most Likely Buildable on Time: Approach B

Approach B has no existential risks. Every component uses known techniques. The work parallelizes well across a team. The integration testing is the main schedule risk, but it's a bounded risk — you know what "done" looks like. **Probability of delivering a complete system on a 12-month timeline: ~75%.**

Approach C is next (**~40–50% on 14 months**), because the compiler fallback means partial delivery is still publishable. Approach A is riskiest (**~30–40% on 10 months**), because the proof work has unbounded variance.

### Recommendation: Difficulty Profile for Best-Paper Artifact

**For a best-paper artifact, Approach C's difficulty profile is optimal, but with a critical caveat.**

The reasoning:

1. **Best papers need both depth and breadth.** Approach A has depth without breadth (tiny system, one solver, one reformulation). Approach B has breadth without depth (large system, no novel algorithms). Approach C has both — the cut theory provides depth, the compiler provides breadth, and the integration provides a novel contribution unique to the hybrid.

2. **The difficulty must be *visible* to reviewers.** Approach A's difficulty is concentrated in a proof that reviewers may or may not appreciate depending on whether they're polyhedral combinatorics experts. Approach B's difficulty is distributed across engineering that reviewers may dismiss as "just implementation." Approach C's difficulty is visible at multiple levels — novel math, novel systems design, and a large validated artifact — giving every type of reviewer something to respect.

3. **The fallback matters.** If the facet-defining proof fails, Approach C degrades to "compiler + computational cuts" — still a strong IJOC paper. Approach A degrades to nothing.

**The critical caveat:** Approach C's difficulty profile only works if the team has both a strong theorist (for the cut math) and strong systems engineers (for the compiler). If the team is theory-heavy, Approach A is better. If the team is engineering-heavy, Approach B is better. Approach C requires a balanced team and excellent project management to hit the integration milestones.

### Summary Table (Honest Scores)

| | Approach A | Approach B | Approach C |
|---|---|---|---|
| **Difficulty (claimed)** | 9 | 6 | 8 |
| **Difficulty (honest)** | **8** | **4** | **7** |
| **Novel LoC (claimed)** | ~14K | ~19.3K | ~19.3K |
| **Novel LoC (honest)** | ~7–8.5K | ~6–7.2K | ~11–13.8K |
| **Hardest single subproblem** | Facet-defining proof (9/10) | Four-backend emission (5/10) | Cut-compiler integration (7/10) |
| **# of genuinely hard subproblems** | 2 | 0 | 3–4 |
| **# of medium subproblems** | 1 | 5–6 | 5–6 |
| **Architectural risk** | Low | Moderate | High |
| **Schedule risk** | High (proof variance) | Moderate (integration testing) | High (integration + proof) |
| **Probability of on-time delivery** | ~35% | ~75% | ~45% |
| **Best-paper probability if delivered** | ~40% | ~10% | ~25% |
| **Expected best-paper value** | ~14% | ~7.5% | ~11% |

The expected value calculation (delivery probability × best-paper-if-delivered probability) slightly favors Approach A, but this hides the variance. Approach A is a high-variance bet; Approach C is a medium-variance bet with a floor. **For a risk-adjusted recommendation: Approach C with aggressive prototype gating.**
