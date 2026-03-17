# Adversarial Critique: Three Competing Framings for a Bilevel Optimization Compiler

**Panel Date:** Crystallization Phase Review  
**Status:** STRESS TEST — no framing is accepted until all challenges are answered  

---

## 1. CRITIC 1: Systems/Engineering Skeptic

### Framing A: BiCL — Universal Bilevel Compiler

**Is 150K LoC genuinely necessary?**  
No. This estimate is inflated by at least 40%. The six solver backends share enormous structural overlap — Gurobi, CPLEX, and HiGHS all consume MPS/LP files and expose nearly identical callback APIs for lazy constraints. Claiming 6 backends × 15K LoC each is dishonest accounting. A well-designed emission layer with a common interface and thin per-solver adapters should be ~25K LoC total, not ~90K. The IR and structural analysis — the actually novel parts — are probably ~30K LoC of real logic. The remaining 95K is boilerplate, tests, and benchmarks. **Challenge: strip the estimate to novel logic only. If it's under 50K, the "150K" claim undermines credibility.**

**Hardest engineering subproblem that could block progress:**  
Convexity/CQ detection. The math specification (M3.2) admits CQ-VERIFICATION is co-NP-hard. DCP-style rules are sound but incomplete — they will misclassify problems that are convex but not DCP-representable. This means the compiler will *reject valid problems* or *fall back to the most expensive reformulation* (value function) unnecessarily. The system will appear broken to any user whose problem doesn't fit the DCP template, which includes most real-world nonlinear bilevel problems. This is not a bug to be fixed later; it is a fundamental architectural limitation that must be honestly disclosed.

**Can this run on a laptop CPU at meaningful scale?**  
The compilation step itself — parsing, type inference, structural analysis, reformulation selection — is cheap. But this is a sleight of hand. The *compilation* is fast; the *compiled problem* may be unsolvable. A bilevel LP with 5000 lower-level constraints generates a KKT reformulation with 5000 dual variables, 5000 complementarity constraints, and 5000 binary variables (for big-M). That's a 5000-binary MILP that Gurobi on a laptop may take hours to solve. The compiler produced it in milliseconds. **Celebrating fast compilation of an unsolvable problem is an empty achievement.**

**Will benchmarks be convincing?**  
Unlikely. The "CVXPY for bilevel" framing invites comparison against CVXPY's benchmark rigor — but CVXPY benchmarks on problems with known global optima (convex programs). Bilevel benchmarks lack standardization: BOLIB has 173 problems (too small), BOBILib has 2600 MIBLP instances (only linear). There is no standard bilevel benchmark suite spanning LP, QP, conic, and NLP lower levels. Framing A will either benchmark on toy problems (unconvincing) or construct its own benchmarks (self-serving). **Reviewers will rightly ask: compared to what? On what instances?**

### Framing B: MIBiL-Cut — Breakthrough MIBLP Solver Technology

**Is 150K LoC genuinely necessary?**  
More credible here. Cut generation, value-function computation, lifting, branching rules, and deep solver callback integration each involve substantial, non-overlapping logic. The 25K LoC for bilevel intersection cuts alone is believable — the separation oracle requires solving auxiliary LPs/MILPs per cut, managing warm-starts, and handling degeneracy. The 15K for solver callbacks is also credible given the genuinely different APIs across Gurobi/SCIP/CPLEX. **However:** 20K for "benchmark infrastructure" is padding. Benchmarks should not count toward technical complexity. Real estimate: ~110-120K LoC.

**Hardest engineering subproblem that could block progress:**  
Value-function oracle performance. The entire cutting plane machinery depends on fast evaluation of V(x) — the lower-level optimal value as a function of upper-level decisions. For LP lower levels, this is a parametric LP (tractable). For MILP lower levels, this requires solving a fresh MILP per evaluation point. During cut generation, the oracle is called *thousands* of times. If each call takes even 0.1 seconds, cut generation for a single node takes minutes, making the branch-and-cut tree intractable. **The entire approach lives or dies on oracle caching and warm-starting. If the cache hit rate drops below ~90%, the system collapses.**

**Can this run on a laptop CPU at meaningful scale?**  
Yes, for small-to-medium MIBLP instances (up to ~500 variables, ~200 constraints per level). This is the sweet spot where root-node cutting plane generation pays off. For larger instances (1000+ variables), root-node cut generation itself becomes the bottleneck. However, this is *still meaningful* — MibS also struggles at this scale, and demonstrating superiority on 200-500 variable instances would be publishable. **The key risk is not scale but diminishing returns: if bilevel intersection cuts only close 5-10% of the root gap (instead of the hoped-for 20-30%), the entire contribution shrinks to incremental.**

**Will benchmarks be convincing?**  
Yes, if executed properly. MibS is the clear baseline. The BOBILib benchmark library provides standardized instances. The comparison is clean: same instances, same solver (Gurobi/SCIP), compare MibS vs. MIBiL-Cut on solve time, root gap closed, and nodes explored. This is the cleanest benchmark story of the three framings. **The risk is that MibS is a moving target — if they incorporate similar ideas before publication, the comparison becomes muddled.**

### Framing C: BiCompile — Domain-Driven

**Is 155K LoC genuinely necessary?**  
This is the most honest estimate precisely because it is the most bloated. 25K for energy market modeling, 20K for neural network MILP encoding, 20K for Benders decomposition, 15K for relaxation cascades, 15K for C&CG, 20K for solver backends, 20K for benchmarks, 10K for validation. But honesty about scope does not excuse the scope itself. **This is two PhD theses stapled together and called a compiler.** The energy market modeling alone (unit commitment with PTDF, N-1 contingencies, reserve requirements) is a mature research subfield with decades of specialized work. Building this from scratch, correctly, is a multi-year effort. Similarly, neural network MILP encoding with CROWN-style bound propagation is α-β-CROWN's core contribution — reimplementing it as a side component is hubristic.

**Hardest engineering subproblem that could block progress:**  
Faithful energy market lower-level modeling. Getting unit commitment right is not an optimization problem — it's a *domain knowledge* problem. Minimum up/down time constraints, hot/warm/cold startup costs, must-run obligations, ancillary service markets, locational marginal pricing, transmission switching — each of these has subtleties that energy engineers spend careers mastering. A compiler team that gets even one of these wrong will produce a tool that energy professionals dismiss instantly. **And if the model is simplified to avoid these issues, the "full-fidelity" selling point evaporates.**

**Can this run on a laptop CPU at meaningful scale?**  
No, not for the adversarial robustness domain. A modest CNN for CIFAR-10 (e.g., 6 convolutional layers, ~100K parameters) produces a MILP with tens of thousands of binary variables after ReLU encoding. Even with aggressive bound propagation and neuron fixing, the remaining MILP is enormous. α-β-CROWN uses GPU-accelerated bound propagation specifically because CPU-only approaches cannot scale. **Running bilevel optimization over this MILP on a laptop CPU is science fiction.** For energy markets, a simplified model (IEEE 30-bus, single time period) is feasible. A realistic model (IEEE 118-bus, 24-hour horizon) is not.

**Will benchmarks be convincing?**  
They will convince nobody in either community. Energy market researchers will see simplified models and say "we already do this better with domain-specific tools." ML robustness researchers will see CPU-only verification times 100x slower than α-β-CROWN and dismiss the work entirely. **The benchmark story requires being worse than specialized tools in both domains while claiming the generality is the contribution — but generality across two domains is not a compelling value proposition when each domain has superior specialized tools.**

---

## 2. CRITIC 2: Theory Skeptic

### Framing A: BiCL — Universal Bilevel Compiler

**Is the mathematics genuinely new?**  
Mostly no. The math specification identifies 26 load-bearing results, of which 6 are "genuinely new." Let's examine these:

1. **Selection Correctness (M2.2):** The function ρ(σ) mapping problem signatures to valid reformulations is presented as a theorem, but it is a *definition* with a correctness proof that is a straightforward case analysis over the conditions in M1.1–M1.5. Any graduate student who has read Dempe (2002) could construct this table in an afternoon. **Calling this a theorem is generous; it's a classification table with a completeness argument.**

2. **CQ-VERIFICATION is co-NP-hard (M3.2):** This is a valid complexity result, but it is a *negative* result that says "the compiler can't do this efficiently." It does not enable anything; it constrains the architecture. The proof (reduction from parametric linear independence checking) is a straightforward exercise in computational complexity. **It's correct but not publishable on its own.**

3. **Compositional Error Bound (M4.3):** The chain error bound is the most interesting new result, but the math specification itself admits it "follows from Lipschitz analysis." The bound is a telescoping product of per-pass amplification factors — a standard technique in numerical analysis. The exponential blowup warning is important but unsurprising. **This is a useful engineering result, not a mathematical contribution.**

4. **Type System Soundness (M5.2):** Extending DCP rules to bilevel problems is incremental. Grant & Boyd (2006) established DCP for convex optimization; extending the type system to track upper/lower variable scoping and CQ status is a modest extension. **The PL community would find this toy-sized; the optimization community won't care about type theory.**

**Which "new" theorems are actually straightforward corollaries?**  
- M2.2 (Selection Correctness) is a corollary of M1.1–M1.5 combined.
- M5.2 (Type Inference Decidability) is a corollary of DCP decidability + the observation that bilevel scoping adds only a finite annotation layer.
- M2.4 (Approximation Bounds) with the Lipschitz argument is a corollary of standard perturbation theory.

**Risk of triviality once stated?**  
High. The "compiler IR" framing is the genuine novelty, but it is a *software architecture* contribution, not a *mathematical* contribution. Once you state "represent bilevel problems as typed ASTs and use DCP rules to determine valid reformulations," the rest follows mechanically. **The risk is that a reviewer says: "This is a well-engineered piece of software with a thin mathematical veneer."**

**Would a Math Programming reviewer find sufficient depth?**  
No. Math Programming publishes papers with deep, self-contained mathematical contributions (new algorithms with convergence proofs, new complexity results, new structural theorems). Framing A's math is a collage of known results organized by a software architecture. **Target INFORMS Journal on Computing instead, where the systems contribution is valued.**

### Framing B: MIBiL-Cut — Breakthrough MIBLP Solver Technology

**Is the mathematics genuinely new?**  
Yes, conditionally. Bilevel intersection cuts — extending Balas's (1971) intersection cut framework to the bilevel feasible region — are a genuinely unexplored direction. The key question is whether the bilevel-infeasible set can be efficiently characterized. For LP lower levels, the bilevel-infeasible set is the complement of the set where the lower-level value function constraint holds; characterizing this requires understanding the piecewise-linear structure of V(x). **This is real mathematics, and the difficulty is genuine.**

Value-function lifting — strengthening value-function cuts by exploiting upper-level integrality — is analogous to Gomory-Johnson lifting for standard cuts. The analogy is well-motivated, but the details are non-trivial because the lifting function depends on V(x), which is itself piecewise-defined. **This could be a substantial contribution if the lifted cuts are demonstrably stronger.**

**Which "new" theorems are actually straightforward corollaries?**  
- The basic bilevel intersection cut derivation (intersecting a maximal bilevel-infeasible convex set with the simplex cone) follows the standard Balas template. The novelty is in *constructing* the bilevel-infeasible set, not in the intersection cut machinery itself.
- Bilevel-aware branching (scoring variables by their impact on the lower-level response) is heuristic, not a theorem. Calling it a "contribution" overstates the mathematical content.

**Risk of triviality once stated?**  
Moderate. If the bilevel-infeasible convex set turns out to have a simple closed-form for LP lower levels (e.g., a polyhedral set derived from the dual), then the intersection cut derivation is mechanical and the contribution shrinks to "apply Balas's framework to one more class of problems." **The contribution stands or falls on the computational difficulty of the separation problem.** If separation requires solving a hard auxiliary problem, the cuts are theoretically interesting but practically useless. If separation is cheap, the result may be seen as an obvious application of known machinery. **There's a narrow corridor of genuine contribution between these failure modes.**

**Would a Math Programming reviewer find sufficient depth?**  
Yes, if the cut theory is developed rigorously with facet-defining properties, worst-case gap closure bounds, and the separation complexity is fully characterized. This is exactly the kind of paper Math Programming publishes. **But the paper must include the theory, not just the computational results. "We ran experiments and the cuts helped" is Computers & OR territory, not Math Programming.**

### Framing C: BiCompile — Domain-Driven

**Is the mathematics genuinely new?**  
No. Every algorithmic component is a known technique applied to a specific domain:
- Network-exploiting Benders decomposition is standard Benders with a graph-partitioned substructure. The observation that power networks decompose by bus is well-known in the power systems literature.
- Adaptive relaxation cascades are a heuristic framework for LP relaxation → partial integer → full integer, guided by dual sensitivity. This is not a theorem; it's an algorithmic design choice.
- Warm-started C&CG using historical data or PGD-generated adversarial examples is an engineering trick, not a mathematical contribution.

**Which "new" theorems are actually straightforward corollaries?**  
All of them. The "network-exploiting decomposition" theorem is: "If the lower-level constraint matrix is block-diagonal with linking constraints, Benders decomposition can be applied per block." This is textbook Benders. The "adaptive relaxation cascade" has no theorem — it's a heuristic with no formal convergence guarantee beyond "the full-integer solve is exact." The C&CG warm-starting has no theorem beyond "starting with good initial columns reduces iterations" — which is obvious.

**Risk of triviality once stated?**  
Extreme. **There is no mathematical contribution here that isn't already known.** The contribution is purely in the *combination* and *domain-specific instantiation* of known techniques. This is a valid systems/applications contribution, but it has no place at a theory venue.

**Would a Math Programming reviewer find sufficient depth?**  
Absolutely not. This paper would be desk-rejected at Math Programming. It could work at Operations Research (applications section), European Journal of Operational Research, or NeurIPS (if the adversarial robustness results are strong enough). **But calling this a "compiler" paper overpromises theoretical sophistication that isn't delivered.**

---

## 3. CRITIC 3: Impact/Value Skeptic

### Framing A: BiCL — Universal Bilevel Compiler

**Who would ACTUALLY use this?**  
OR graduate students working on bilevel optimization who currently use BilevelJuMP, PAO, or manual reformulation. This is a real but small community — perhaps 200-500 active researchers worldwide. The "CVXPY for bilevel" analogy is aspirational but misleading: CVXPY succeeded because convex optimization has a massive user base (ML, signal processing, finance, control). Bilevel optimization is a niche within a niche. **The tool would be used by the people who already know they need bilevel optimization — it doesn't create new demand.**

Industrial users (energy companies, logistics firms) already have custom implementations tuned to their specific problem structures. They will not switch to a general-purpose compiler that may produce slower formulations than their hand-tuned code. **"Correct but slow" is not a value proposition for industry.**

**Is the value immediately obvious?**  
Yes, to bilevel optimization researchers. No, to everyone else. The pitch "you specify your bilevel problem and we automatically choose the best reformulation" is compelling *if you already understand what a reformulation is*. For practitioners who just want to solve their problem, the value is indirect — they need to first understand that their problem is bilevel, then learn the modeling language, then trust the compiler's reformulation choice. **The adoption funnel is long and narrow.**

**Can we demonstrate 10x improvement over status quo?**  
Over manual reformulation by a novice: probably yes (time savings, not solve speed). Over expert manual reformulation: probably no. Over BilevelJuMP: maybe 2-3x on problems where strategy selection matters (i.e., where KKT is not the obviously correct choice). **The 10x claim would require finding instances where automatic selection discovers a reformulation strategy the user would never have considered. Such instances must exist and must be demonstrated.**

**Tool vs. paper?**  
This is the strongest "tool people will use" framing of the three. If well-executed, it could become the default bilevel modeling tool in Python. But "well-executed" requires: (a) a clean API that feels like CVXPY, (b) documentation and tutorials, (c) active maintenance, (d) a community. **Without sustained investment post-publication, this becomes abandonware within 2 years — like most research software.**

### Framing B: MIBiL-Cut — Breakthrough MIBLP Solver Technology

**Who would ACTUALLY use this?**  
MIBLP researchers (perhaps 50-100 worldwide) who currently use MibS or hand-code branch-and-cut algorithms. This is an extremely small user base. However, the *output* of the compiler — strengthened MILPs — can be consumed by anyone with access to Gurobi/SCIP/CPLEX. This means the indirect user base is larger: anyone who can formulate their problem as an MIBLP can benefit from the strengthened formulation, even if they don't understand the cutting plane theory. **The "compile once, solve anywhere" value proposition is genuinely useful for reproducibility and benchmarking.**

**Is the value immediately obvious?**  
To the target audience (MIBLP researchers): extremely obvious. "Here are new cuts that close the root gap by X%" is the most directly measurable contribution in integer programming. **But to anyone outside this narrow community, the value requires significant explanation.**

**Can we demonstrate 10x improvement over status quo?**  
Plausible but uncertain. Standard intersection cuts for MILPs (GMI cuts) close 30-50% of the root gap for typical instances. If bilevel intersection cuts achieve even half of that (15-25% root gap closure for MIBLPs), the resulting branch-and-bound speedup could be 5-50x depending on instance structure. **But if the cuts are weak (5-10% gap closure), the speedup may be only 1.5-2x — publishable but not transformative.**

The MibS comparison is the key: if MIBiL-Cut solves instances that MibS cannot solve within a time limit (1 hour), that's a clear 10x+ demonstration. **But this requires carefully selecting hard instances, which invites accusations of cherry-picking.**

**Tool vs. paper?**  
This is overwhelmingly a **paper people will cite**. The cutting plane theory is the contribution; the software is the evidence. Researchers will cite the bilevel intersection cut results and may implement them in their own frameworks. Very few will download and run MIBiL-Cut itself, because: (a) the user base is tiny, (b) researchers in this space write their own solvers, (c) the code will be tightly coupled to specific solver versions. **This is fine — a highly cited paper is a successful outcome.**

### Framing C: BiCompile — Domain-Driven

**Who would ACTUALLY use this?**  
In theory: energy market participants and ML robustness researchers. In practice: neither.

Energy market participants already have commercial tools (PLEXOS, PROMOD, Aurora) and in-house optimization stacks developed over decades. They will not adopt an academic prototype that models a simplified version of their market. The gap between "IEEE 30-bus test system" and "PJM's actual market" is enormous — and no academic tool bridges it. **Energy companies that would benefit most from bilevel optimization already have optimization teams that can (and do) build custom solutions.**

ML robustness researchers use α-β-CROWN, auto_LiRPA, or custom MILP verifiers. They benchmark on VNN-COMP instances with strict time limits. A CPU-only bilevel compiler that is 100x slower than GPU-accelerated α-β-CROWN will not be adopted, regardless of its generality. **The ML community optimizes for speed, not generality. A slower but more general tool is a non-starter.**

**Is the value immediately obvious?**  
No. The value requires believing simultaneously that: (a) bilevel optimization is the right framework for both energy and robustness, (b) existing specialized tools are inadequate, (c) a general compiler can outperform specialized tools. Points (a) and (b) are debatable; point (c) is almost certainly false. **The value proposition requires too much squinting.**

**Can we demonstrate 10x improvement over status quo?**  
No. The status quo in energy markets is manual MPEC reformulation by domain experts — the compiler may match their speed on standard formulations. The status quo in adversarial robustness is α-β-CROWN on GPU — the compiler will be orders of magnitude slower. **There is no scenario where this framing demonstrates 10x improvement in either domain.**

**Tool vs. paper?**  
A paper that people will skim the abstract of, note the energy + ML combination as interesting, and never download or cite. **The two-domain structure means it will be submitted to a venue that values breadth (AAAI, IJCAI) where it will be reviewed by people who are experts in neither domain and may pass — or it will be submitted to a domain-specific venue where it will be found lacking in depth.**

---

## 4. CROSS-CRITIC SYNTHESIS

### Survivability Matrix

| Criterion | Framing A | Framing B | Framing C |
|-----------|-----------|-----------|-----------|
| Engineering feasibility | ⚠️ Over-scoped but modular | ✅ Focused and buildable | ❌ Two PhDs in a trenchcoat |
| Mathematical novelty | ⚠️ Thin—mostly classification | ✅ Genuine new cutting theory | ❌ Zero new math |
| Practical impact | ⚠️ Small community, real need | ⚠️ Tiny community, deep need | ❌ No community adoption path |
| Benchmark story | ⚠️ No standard suite exists | ✅ Clean MibS comparison | ❌ Loses to specialists in both domains |
| CPU feasibility | ✅ Compilation is cheap | ✅ MILP solving is CPU-native | ❌ Neural network verification needs GPU |
| Venue fit | ⚠️ IJOC (not top theory) | ✅ Math Prog / IPCO | ❌ No clear home |

**Which framing survives all three critics best?**  
**Framing B** survives with the least damage. It has genuine mathematical novelty (bilevel intersection cuts), a clean benchmark story (vs. MibS on BOBILib), CPU feasibility, and a focused scope. Its weakness is narrow impact — but a highly cited paper in Math Programming or IPCO is a successful outcome.

**Framing A** survives as a secondary option with a different success metric: if the goal is building lasting research infrastructure (not maximizing paper impact), the compiler architecture is the most durable contribution. But it needs mathematical sharpening.

**Framing C** does not survive. It should be abandoned as a standalone framing.

### Elements to Preserve from Each Framing

**From Framing A (preserve):**
- The typed IR and formal grammar for bilevel programs (M5.1, M5.2) — this is genuinely novel and useful regardless of the downstream application
- The reformulation selection framework (M2.2) — even if the math is lightweight, the classification is practically valuable
- Solver-agnostic emission architecture — the "compile once, solve anywhere" philosophy
- The compositional error bound for reformulation chains (M4.3) — useful engineering theory

**From Framing B (preserve):**
- Bilevel intersection cut theory — the crown jewel of mathematical novelty
- Value-function lifting framework — strengthens the cutting plane contribution
- Progressive strengthening pipeline — clean algorithmic narrative
- MibS benchmark comparison methodology — the strongest empirical story

**From Framing C (preserve):**
- Domain-specific warm-starting for C&CG — a small but practical trick worth including as a feature, not a centerpiece
- Energy market strategic bidding as a *motivating example* (not a full domain) — illustrates why bilevel compilation matters
- Nothing else. The neural network verification angle adds scope without adding value.

### THE OPTIMAL HYBRID

**Architecture:** Framing A's compiler IR (typed AST, structural analysis, reformulation selection) as the foundation.

**Mathematical core:** Framing B's bilevel intersection cuts and value-function lifting as the primary new contribution — but embedded as a compiler pass within Framing A's architecture, not as a standalone solver.

**Scope:** Restrict to **mixed-integer bilevel linear programs** (Framing B's scope) for the primary paper. The compiler IR supports broader problem classes by design, but the evaluation focuses on MIBLPs where the contribution is deepest.

**Benchmarks:** MibS comparison on BOBILib (Framing B's methodology), plus a strategic bidding case study (from Framing C, simplified) to demonstrate practical relevance.

**Narrative:** "We built a bilevel optimization compiler with a typed IR that automatically selects reformulation strategies. For MIBLPs, the compiler implements novel bilevel intersection cuts and value-function lifting that close X% of the root gap, yielding Y× speedups over MibS. The compiler architecture generalizes to other bilevel problem classes, which we demonstrate on a strategic energy bidding example."

**LoC estimate for the hybrid:** ~100K LoC (cutting the neural network verification, the full energy market modeling stack, and the duplicative solver backends).

### What to EXPLICITLY CUT

1. **Neural network / adversarial robustness** — entirely. It requires GPU, is slower than α-β-CROWN, and adds scope without adding citations.
2. **Full-fidelity energy market modeling** — cut the full unit commitment stack. Keep a simplified strategic bidding example (LP-relaxed market clearing, single bus or small network) as a case study.
3. **Ipopt and OR-Tools backends** — cut. Focus on Gurobi, SCIP, and HiGHS. Three backends demonstrate solver-agnosticism; six is vanity.
4. **NLP and conic bilevel support** — defer to future work. The IR can represent these problem classes, but the reformulation and evaluation infrastructure should not attempt them in v1.
5. **Pessimistic bilevel formulations** — defer. Optimistic formulations cover 90%+ of applications. Pessimistic support adds theoretical complexity (robust optimization reformulations) without proportional practical return.
6. **Regularization methods** (Scholtes, penalty) — cut from the primary contribution. Mention as compiler options but don't evaluate. They produce approximate solutions and will confuse the correctness certificate narrative.

---

## 5. DIRECT CHALLENGES

These must be answered with concrete evidence before any framing is accepted:

### Challenge 1: Demonstrate that automatic reformulation selection beats expert manual selection

**Requirement:** Produce at least 5 bilevel problem instances where the compiler's automatically selected reformulation strategy is *at least 2x faster in solve time* than the reformulation a domain expert would choose by default (typically KKT/big-M for convex lower levels, value-function for integer lower levels). 

**Why this matters:** If experts always choose correctly, the compiler's selection engine adds complexity without value. The selection engine must demonstrate non-obvious choices — e.g., selecting strong duality over KKT for an LP lower level where the dual formulation produces a tighter relaxation, or selecting C&CG over KKT for a problem with a high-dimensional lower level where KKT blows up the problem size. **Without this evidence, the compiler is a glorified code generator, not an intelligent system.**

### Challenge 2: Prove that correctness certificates add value beyond "the user should know duality theory"

**Requirement:** Design and execute a user study or construct a concrete scenario where: (a) a user formulates a bilevel problem, (b) applies a reformulation that appears valid but is actually incorrect (e.g., applying KKT when a constraint qualification fails), (c) obtains a solution that is bilevel-infeasible, and (d) the compiler's correctness certificate would have caught the error. Alternatively, show that at least 10% of bilevel problems in BOLIB/BOBILib, when reformulated with the "default" strategy, yield incorrect solutions due to violated assumptions.

**Why this matters:** Correctness certificates are a major selling point ("no existing tool provides them"). But if reformulation errors are rare in practice — because experienced users know when KKT is valid — the certificates are solving a problem that doesn't exist. **The certificates must catch real bugs, not hypothetical ones.**

### Challenge 3: Quantify the bilevel intersection cut gap closure

**Requirement:** On a suite of at least 50 MIBLP instances from BOBILib, compute: (a) the LP relaxation bound, (b) the LP relaxation + bilevel intersection cuts bound, (c) the optimal solution value. Report the percentage of the integrality gap closed by bilevel intersection cuts alone.

**Why this matters:** The entire value proposition of Framing B rests on these cuts being effective. GMI cuts close 30-50% of the gap for standard MILPs. If bilevel intersection cuts close less than 10%, the contribution is incremental. If they close 20%+, the contribution is significant. **This is the single most important empirical question in the entire project. It should be answered first, before committing to the full 100K+ LoC implementation.**

### Challenge 4: Show that the compiler architecture is not a premature abstraction

**Requirement:** Implement a minimal viable compiler (IR + one reformulation pass + one solver backend) in under 5K LoC and demonstrate it solves a non-trivial bilevel problem correctly. Then incrementally add a second reformulation pass and a second solver backend, measuring the marginal cost of each addition.

**Why this matters:** Compiler architectures are seductive but often premature. If adding a new reformulation strategy requires modifying the IR, the type system, and the emission layer, the "compiler" framing is an architectural liability, not an asset. The architecture must prove that new reformulations are **truly modular** — pluggable without touching existing code. **If the MVP takes 20K LoC instead of 5K, the compiler abstraction is already leaking.**

### Challenge 5: Resolve the "fast compilation, slow solving" paradox

**Requirement:** For at least 10 problem instances, report both: (a) compilation time (parsing + analysis + reformulation + emission), and (b) downstream solve time (solver time on the compiled problem). If compilation time is <1% of solve time in all cases, justify why the compilation step matters at all — the bottleneck is the solver, not the reformulation choice. Conversely, if compilation time is >10% of solve time, justify why the overhead is acceptable.

**Why this matters:** A compiler that produces output in milliseconds but the output takes hours to solve is not solving the user's real problem. The user's problem is: "my bilevel optimization takes too long." If the compiler's contribution is "I chose a reformulation 50% faster to solve, but the solve still takes 3 hours instead of 6," that's useful but not transformative. **The compiler must demonstrate that reformulation choice, not solver performance, is the dominant factor in at least some problem instances.**

---

## APPENDIX: Summary Verdict

| | Framing A (BiCL) | Framing B (MIBiL-Cut) | Framing C (BiCompile) |
|---|---|---|---|
| **Critic 1 (Systems)** | Over-scoped, padded LoC, weak benchmarks | Focused, credible, oracle bottleneck risk | Infeasible scope, fails on GPU constraint |
| **Critic 2 (Theory)** | Thin math, classification masquerading as theorems | Genuine new math with narrow viability corridor | Zero new mathematics |
| **Critic 3 (Impact)** | Small community, real but modest value | Tiny community, high citation value | No adoption path in either domain |
| **Verdict** | Viable as infrastructure play, not as theory paper | **Strongest framing**; best risk/reward | **Abandon** |
| **Recommended action** | Merge IR/architecture into hybrid | Core mathematical contribution | Scavenge warm-starting trick and one case study; discard rest |

**Final recommendation:** Build the hybrid. Framing B's mathematics in Framing A's architecture, scoped to MIBLPs, with a single motivating domain example. Cut everything else ruthlessly. Answer Challenges 3 and 4 first — they are the fastest way to validate or kill the project.
