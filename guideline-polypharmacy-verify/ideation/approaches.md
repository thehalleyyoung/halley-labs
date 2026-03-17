# GuardPharma: Three Competing Approaches

**Slug:** `guideline-polypharmacy-verify`  
**Date:** 2025-07-18  
**Phase:** Domain Visionary — Approach Generation

---

## Approach 1: PTA-Contract Compositional Model Checking

**One-line summary:** Compile clinical guidelines into Pharmacological Timed Automata with compartmental ODE state, then verify N-guideline polypharmacy safety compositionally via CYP-enzyme interface contracts and CEGAR-based model checking.

### Extreme Value Delivered

**Who desperately needs this:** Hospital CDS committees at large academic medical centers (Mayo, Partners, UPMC) that are deploying FHIR-based clinical decision support under ONC HTI-1 mandates. These committees currently review guideline interactions manually — most interactions are never checked. The Joint Commission already cites CDS configuration errors as a contributing factor in medication safety events. EHR vendors (Epic, Oracle Health) ship CDS content to thousands of hospitals and rely on post-deployment adverse event monitoring to find interaction bugs — a reactive posture that exposes patients to harm before errors are discovered.

**Why they need it now:** The 21st Century Cures Act mandates FHIR-based CDS interoperability in certified EHR systems. Computable guidelines are transitioning from research prototypes to deployed clinical infrastructure. The window between "computable guidelines exist" and "computable guidelines are deployed at scale" is the critical moment for pre-deployment verification infrastructure. GuardPharma provides the first formal safety certificate for multi-guideline polypharmacy — saying not "this specific prescription pair is flagged" but "these two guidelines can *never* produce a dangerous state for *any* patient matching the comorbidity profile."

### Genuine Software Difficulty

The system must be simultaneously correct across three domains — formal methods, pharmacokinetics, and clinical informatics — each of which alone is a strong research program.

**Hard subproblems:**

1. **CQL-to-PTA semantic compilation (~10K novel LoC).** CQL has temporal operators, interval arithmetic, FHIR-path expressions, terminology bindings, and a rich type system. Translating this faithfully into hybrid automata transitions — where guards involve both discrete clinical state predicates and continuous PK concentration thresholds — implicitly creates the first formal semantics of CQL. No existing compiler does this; the closest analogy is CompCert-level semantic translation for a clinical DSL.

2. **Contract extraction from PK models (~14K novel LoC).** Each guideline must be modeled as an open system with an enzyme-interface contract: an (assume, guarantee) pair over the shared CYP-enzyme activity vector. Extracting these contracts requires computing worst-case enzyme loads from population PK parameters, which involves solving parametric reachability over Metzler ODEs — bridging formal methods and pharmacology in a way neither community has done.

3. **CEGAR with clinical domain abstractions (~14K novel LoC).** The PK region graph for 5+ concurrent drugs can have ~10^15 regions before pruning. The CEGAR loop must use domain-specific abstractions — drug-class equivalence, lab-value coarsening at clinical thresholds, temporal aggregation — to make this tractable. Convergence of CEGAR on PK-structured state spaces is unproven; the system needs a bounded model checking fallback.

4. **Zonotopic reachability for Metzler systems (~10K novel LoC).** Novel reachability algorithm exploiting the Metzler property of PK dynamics. Drug discontinuation resets break monotonicity and require partitioned analysis.

**Total estimated scope:** ~95K LoC for the paper-phase artifact (~65K novel algorithmic code). The full-vision system with CQL/FHIR compilation is ~135K LoC (~83K novel).

### New Math Required

**Proposition 1 (δ-decidability of PTA reachability).** The reachability problem for PTA with Metzler ODE dynamics and bounded population PK parameters is δ-decidable: for clinically meaningful δ (minimum pharmacologically significant concentration difference, ~0.1 μg/mL), the verifier answers "unsafe" (exact) or "δ-safe." This is an application of dReal's δ-decidability framework to PK dynamics — the novelty is the domain-specific δ-calibration giving the result clinical meaning, not a new decidability technique. **Why necessary:** without δ-decidability, the verifier may not terminate, and GuardPharma degrades from a verification tool to a testing tool.

**Theorem 2 (MTL model checking for PTA).** The pharmacokinetic region graph partitions the continuous PK state space at clinical thresholds (toxic level, therapeutic range boundaries) rather than at infinitesimal clock equivalences (Alur-Dill). This domain-specific discretization yields a finite graph for bounded-horizon MTL model checking. The correctness argument — that this discretization is sound for MTL properties referencing only clinical threshold predicates — is the technical core. **Why necessary:** without PK regions, model checking falls back to general hybrid automata methods that are either undecidable or impractically expensive.

**Theorem 3 (contract-based compositional safety — the crown jewel).** N-guideline safety decomposes into N individual guideline checks plus enzyme-compatibility checks (linear in N per enzyme), avoiding exponential product-automaton construction. The key insight: for competitive CYP inhibition, enzyme-load guarantee functions are monotone in assumed enzyme capacity due to Metzler dynamics. Worst-case guarantees computed under minimum assumed capacity are sound upper bounds, resolving the circular dependency between drug concentrations and enzyme activity in a single pass. **Why necessary:** without compositionality, verifying a patient on 5+ guidelines is intractable — product-automaton construction produces 2^N states. With compositionality, cost is O(N · single-guideline-cost + N · M) for M shared enzymes.

### Best-Paper Potential

The centerpiece is experiment E1 (temporal ablation): "X% of guideline conflicts require temporal pharmacokinetic reasoning to detect." This is clean and memorable. If X ≥ 20%, it demonstrates that the PTA+PK machinery discovers real dangers invisible to all prior approaches. The contract composition theorem (Theorem 3) is a genuine mathematical contribution — the first assume-guarantee framework organized around metabolic pathway interfaces. The system bridges formal methods and clinical informatics with a working tool, which is catnip for cross-disciplinary venues (AIME, AMIA). Weaknesses: E1 is a high-variance gamble (depth check estimates X at 10–30%); individual theorems are thin for pure FM venues (CAV).

**Target venue:** AIME (8–12% P(best paper)), AMIA Annual (5–8%), JAMIA (3–5%).

### Hardest Technical Challenge

**The E1 temporal ablation experiment.** If the fraction of conflicts requiring temporal PK reasoning (X%) is below ~15%, the entire PTA+zonotope+CEGAR machinery appears over-engineered relative to a simpler atemporal checker. The critical analytical distinction (from the depth check): PK interactions *being temporal in nature* ≠ guideline conflicts *requiring temporal reasoning to detect*. Fluconazole + warfarin is temporal, but an atemporal checker flags it fine as a CYP2C9 interaction. Temporal reasoning adds *detection* value only for schedule-dependent conflicts where the specific timing prescribed by guidelines determines whether the combination is dangerous.

**How to address it:** (a) Pre-commit to a fallback narrative — if X < 20%, pivot to explanation quality (PK trajectory counterexamples are unprecedented), compositionality speedup (E6), and existence proof (first formal safety certificate). (b) Construct 5–10 synthetic guideline pairs with known temporal interactions as proof-of-concept before running E1 on the full corpus. (c) Literature calibration: estimate X from published PK DDI studies to set expectations before implementation.

### Scores

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Value | 6 | Real problem, genuine LLM-proof moat, but zero demand signal and CQL treatment-logic adoption near-zero |
| Difficulty | 8 | Three-domain intersection, ~65K novel LoC, novel formalism + composition theorem |
| Potential | 6 | 5–12% P(best paper) at best venue; E1 gamble could elevate or collapse the narrative |
| Feasibility | 6 | CEGAR convergence unknown; E1 may disappoint; pilot on 3 guideline pairs is a hard gate |

---

## Approach 2: Abstract Interpretation over Pharmacokinetic Lattices

**One-line summary:** Define pharmacokinetically-structured abstract domains (concentration intervals × enzyme-load intervals × clinical-state predicates) and compute multi-guideline safety as a fixed-point over these domains, avoiding automaton construction and state-space exploration entirely.

### Extreme Value Delivered

**Who desperately needs this:** The same stakeholders as Approach 1 (hospital CDS committees, EHR vendors, guideline developers), but with a critical practical advantage: **speed and interpretability.** Abstract interpretation converges in seconds even for 20+ concurrent guidelines, producing concentration-interval bounds directly interpretable by clinical pharmacists ("under these guidelines, warfarin concentration could reach 4.2–6.8 μg/mL — above the 5.0 μg/mL toxicity threshold"). This is the language pharmacists think in, unlike formal counterexample traces.

**Specific additional stakeholders:** (a) Clinical pharmacists conducting polypharmacy medication therapy management (MTM) reviews — they currently use heuristic checklists and could instead get per-patient concentration-interval reports in seconds. (b) Pharmacy benefit managers (PBMs) who need to screen formulary combinations at scale — abstract interpretation can check thousands of drug combinations per minute, enabling formulary-level safety screening that model checking cannot reach.

### Genuine Software Difficulty

**Hard subproblems:**

1. **Designing the PK abstract domain.** The domain elements are tuples of (concentration interval per drug, enzyme-activity interval per CYP, clinical-state predicate set). The domain must be precise enough to distinguish therapeutic from toxic (typical therapeutic window is 2× — e.g., warfarin therapeutic 2–5 μg/mL, toxic >5 μg/mL) while being coarse enough to terminate quickly. Too precise → the domain has too many elements and the fixed-point doesn't converge. Too coarse → every combination is "possibly unsafe" (useless). This precision-coarseness tradeoff is the central design challenge.

2. **Pharmacokinetic widening operator.** Standard interval widening (jump to [−∞, +∞]) is catastrophic for PK — it says "concentration could be anything." The widening must exploit pharmacological bounds (no drug concentration can exceed the lethal dose; all drugs are eventually eliminated) and steady-state convergence (Metzler eigenvalues determine convergence rate). Designing a widening that terminates in bounded iterations while preserving therapeutic-vs-toxic precision is a novel algorithmic challenge.

3. **Reduced product for enzyme-coupled guidelines.** When N guidelines share CYP enzymes, the naive product of N abstract domains is N-dimensional. The system must identify enzyme-sharing structure and construct a reduced product that tracks only enzyme-coupled drugs jointly, reducing dimensionality from O(∏ |drugs_e|) to O(Σ |drugs_e|).

4. **Abstract transformers for guideline steps.** Each CQL decision step (medication initiation, dose adjustment, lab check) must be modeled as an abstract transformer. The transformer must faithfully represent the step's effect on the PK abstract domain — including dose-dependent concentration changes, CYP-inhibition effects on co-administered drugs, and time-dependent concentration evolution.

**Estimated scope:** ~75K LoC total (~50K novel). Simpler architecture than Approach 1 — no automaton construction, no region graph, no CEGAR loop, no BDD/SAT backends. The complexity is concentrated in the abstract domain design and the pharmacokinetic transformers.

### New Math Required

**Theorem A (Galois connection for PK dynamics).** Define concrete domain C as the powerset of (ℝⁿ × L) — sets of (concentration vector, discrete clinical location) pairs — ordered by inclusion. Define abstract domain A as the lattice of (concentration-interval tuples × enzyme-interval tuples × clinical-predicate sets) with componentwise ordering. Prove that the standard abstraction (componentwise interval hull + predicate extraction) and concretization (Cartesian product of intervals × predicate satisfaction) form a Galois connection. The load-bearing insight: Metzler dynamics preserve interval structure — the image of a box under Metzler flow is contained in a computable box — so the abstract transformer for PK evolution is *exact* for single drugs. Over-approximation enters only through enzyme-interaction coupling. **Why necessary:** without the Galois connection, there is no guarantee that abstract-domain safety implies concrete safety. The soundness of the entire analysis depends on this theorem.

**Theorem B (PK-aware widening with bounded convergence).** Define a widening operator ∇_PK on the PK abstract domain:
- For concentration intervals: widen to [0, C_lethal] if the interval grows across iterations; otherwise, widen to [0, C_ss,max(φ_worst)] where C_ss,max is the worst-case steady-state concentration under worst-case population PK parameters φ_worst.
- For enzyme-activity intervals: widen to [E_min(full-inhibition), E_max(no-inhibition)].

Prove that ∇_PK ensures convergence in at most D iterations (D = number of drugs), because each drug independently reaches its worst-case steady-state under Metzler dynamics with stable eigenvalues. The key insight: steady-state convergence of compartmental PK models (physiologically: all drugs are eventually eliminated) provides a natural fixpoint that limits widening. **Why necessary:** without guaranteed convergence, the analysis may loop forever. Standard widening converges but destroys all precision. PK-aware widening converges *and* preserves therapeutic-vs-toxic discrimination.

**Theorem C (reduced enzyme-coupling product).** For N guidelines where enzyme e is shared by a subset S_e of guidelines, define the reduced product that tracks drugs in S_e jointly and all other drugs independently. Prove that the reduced product is a sound abstraction of the full Cartesian product, with size O(Σ_e |S_e|) instead of O(∏_e |S_e|). **Why necessary:** without the reduced product, 20 guidelines × 5 drugs each = 100-dimensional abstract domain. With the reduced product organized by CYP enzymes (typically 3–5 drugs share each enzyme), the effective dimensionality drops to ~15–20 joint variables.

### Best-Paper Potential

**The memorable result:** "Abstract interpretation over pharmacokinetic lattices verifies 50 concurrent guidelines in 0.3 seconds on a laptop, producing concentration-interval safety reports directly usable by clinical pharmacists." The speed story is dramatic — orders of magnitude faster than any model-checking approach, with a clean theoretical explanation (fixed-point computation over a domain designed to converge fast). The novel abstract domain design (PK-aware widening, enzyme-coupling product) is a genuine contribution to static analysis theory applied to an undeniably important domain. The theoretical cleanness (three crisp theorems, each load-bearing) is a strength at PL/FM venues.

**Target venue:** SAS (Static Analysis Symposium, 10–15% P(best paper)), POPL (abstract domain novelty, 3–5%), VMCAI (5–10%), AIME (8–12% with clinical framing).

**Weakness:** Abstract interpretation over-approximates — it can prove safety but cannot produce counterexamples when verification fails. It says "possibly unsafe" rather than "here is the exact patient trajectory that reaches toxicity." This limits the diagnostic value compared to Approach 1's counterexample traces.

### Hardest Technical Challenge

**Precision of the enzyme-coupling abstract transformer.** When drugs A and B both inhibit CYP3A4, the abstract transformer must compute how A's increased concentration (due to B's inhibition) further increases B's concentration (due to A's increased inhibition). This circular dependency is the same fixed-point problem as Approach 1's contract resolution, but in the abstract domain it manifests as imprecision: the interval for A's concentration depends on B's interval, which depends on A's interval, creating widening pressure. If the abstract transformer is too imprecise for enzyme-coupled drugs, it will flag every CYP3A4-sharing combination as "possibly unsafe" — producing unacceptable false-positive rates.

**How to address it:** (a) Use narrowing iterations after the initial widening pass to recover precision — the Metzler monotonicity guarantees that narrowing converges to a tighter fixpoint. (b) For CYP3A4 specifically (the most promiscuous enzyme, shared by ~50% of drugs), implement a specialized "CYP3A4 relational domain" that tracks the pairwise inhibition relationship between CYP3A4 substrates using a polyhedra sub-domain instead of intervals. This adds precision where it matters most at modest cost. (c) Report confidence levels: "definitely safe" (abstract domain doesn't intersect unsafe region), "possibly unsafe — recommend detailed review" (abstract domain intersects but concrete intersection unknown), and "definitely unsafe" (lower bound of abstract domain exceeds threshold).

### Scores

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Value | 7 | Same problem as Approach 1 but faster, more interpretable output; scales to formulary-level screening |
| Difficulty | 7 | Simpler architecture (~50K novel LoC) but novel abstract domain design is hard; fewer moving parts |
| Potential | 7 | Cleaner theoretical contribution (3 tight theorems); natural fit for PL venues; speed story is dramatic |
| Feasibility | 7 | Abstract interpretation is more predictable than model checking; main risk is precision, not termination |

---

## Approach 3: Pharmacokinetic Safety Games with Safe-Schedule Synthesis

**One-line summary:** Model multi-guideline polypharmacy as a two-player timed game between a scheduler (choosing drug administration times) and an adversarial physiology (choosing worst-case PK parameters), then synthesize provably safe medication schedules that maximize therapeutic efficacy.

### Extreme Value Delivered

**Who desperately needs this:** Clinical pharmacists conducting polypharmacy medication therapy management (MTM) reviews. The Bureau of Labor Statistics counts ~340,000 pharmacists in the US; MTM is a billable service under Medicare Part D. Currently, when a pharmacist identifies a polypharmacy conflict, the only intervention is "recommend discontinuing one drug" or "recommend dose reduction" — both of which compromise therapeutic intent. What pharmacists actually want is: "Here is how to give all prescribed drugs safely by adjusting their timing."

**Why this is transformatively different:** Approaches 1 and 2 produce *diagnostic* output — they identify conflicts. Approach 3 produces *prescriptive* output — it synthesizes safe medication schedules that preserve therapeutic intent while eliminating pharmacokinetic hazards. The output is actionable: "Take metformin 500mg at 08:00, atorvastatin 20mg at 20:00 (12h separation), hold lisinopril until 72h post-fluconazole cessation." This is the difference between a compiler that reports errors and an IDE that offers auto-fixes.

**Additional stakeholders:** (a) Automated medication dispensing systems (Pyxis, Omnicell) at hospitals could use synthesized schedules to time-gate medication availability. (b) Long-term care facilities managing 10+ concurrent medications per resident — scheduling is their primary safety lever. (c) Oncology pharmacists managing chemotherapy regimens with narrow therapeutic indices and severe DDI consequences — optimal scheduling can be the difference between efficacy and fatal toxicity.

### Genuine Software Difficulty

**Hard subproblems:**

1. **Extracting temporal flexibility from guidelines.** Clinical guidelines say "daily" but not "at what hour." They say "twice daily" but permit 8h-16h or 12h-12h splits. This temporal slack — the set of guideline-compliant administration schedules — must be formally extracted from CQL/FHIR artifacts. The extraction requires interpreting FHIR TimingRepeat constraints (frequency, period, dayOfWeek, timeOfDay, when) and CQL temporal expressions into a constraint set over administration times. This is a novel compilation problem.

2. **Hybrid game construction.** The game arena combines discrete clinical state (guideline decisions), continuous PK state (drug concentrations evolving under Metzler ODEs), and adversarial uncertainty (population PK parameters). The scheduler moves by choosing administration times; the adversary responds by choosing PK parameters for the interval until the next dose. Constructing this game from compiled guidelines requires solving the product-game problem — composing N guidelines into a single game arena with shared PK state.

3. **Strategy synthesis for continuous-state games.** Winning strategies for games with continuous state are in general infinite-memory objects — they depend on the entire history of continuous state evolution. Computing a finite-memory winning strategy (one implementable as a finite schedule) requires proving that the Metzler structure permits memoryless strategies or strategies with bounded memory.

4. **Multi-objective Pareto optimization.** A safe schedule must not only avoid toxicity but also maintain therapeutic efficacy — drug concentrations must stay in the therapeutic window (between sub-therapeutic and toxic thresholds). The system must compute Pareto-optimal schedules trading off between drugs (sometimes you can't keep all drugs in the therapeutic window simultaneously, and you must choose which drug to prioritize). This is a multi-objective optimization embedded within a safety-game framework.

**Estimated scope:** ~110K LoC total (~70K novel). Comparable complexity to Approach 1 but the difficulty is concentrated differently: less in state-space exploration (no CEGAR), more in game solving and optimization.

### New Math Required

**Theorem I (decidability of PK safety games).** Define a Pharmacokinetic Timed Game (PTG) where Player 1 (scheduler) chooses administration times within guideline-permitted intervals and Player 2 (adversary) chooses PK parameters φ ∈ Φ from bounded intervals. The continuous state evolves as ẋ = M(φ)x + B·d(t) (Metzler). The safety game — "does Player 1 have a strategy ensuring concentrations remain below toxic thresholds for treatment horizon H, regardless of Player 2?" — is decidable for PTGs with Metzler dynamics.

**Proof strategy:** (a) *Adversary extremalization:* Metzler monotonicity ensures that the worst-case adversary strategy is extremal — minimum clearance parameters maximize drug concentrations. This reduces the adversary's infinite strategy space to a finite set of 2^p extremal parameter vectors (p = number of independent PK parameters). (b) *Scheduler discretization:* for each fixed adversary strategy, the scheduler's problem reduces to choosing administration times on a finite grid (discretization of the dosing interval). The grid granularity is bounded by the PK time constants (faster elimination → finer grid needed). (c) *Finite game reduction:* the resulting discrete game is finite-state and solvable by backward induction in time polynomial in the game size.

**Why necessary:** without decidability, there is no guarantee the synthesis algorithm terminates. The Metzler structure is load-bearing — without it, hybrid games with ODE dynamics are undecidable.

**Theorem II (Pareto-optimal safe schedules).** Given a PTG with D drugs, define the therapeutic efficacy vector E(σ) = (E₁(σ), …, E_D(σ)) where Eᵢ(σ) is the fraction of the treatment horizon during which drug i's concentration is in its therapeutic window under schedule σ, minimized over all adversary strategies. The set of Pareto-optimal safe schedules {σ : ¬∃σ' with E(σ') ≥ E(σ) and E(σ') ≠ E(σ)} is a polytope in schedule-parameter space, representable by at most D+1 vertices and computable in O(D · |grid|^D · 2^p) time.

**Why necessary:** without Pareto optimality, the system could return any safe schedule — including one that keeps all drugs at sub-therapeutic concentrations (safe but useless). The Pareto characterization ensures the system returns schedules that are maximally effective within the safety constraint. The polytope structure enables presenting the pharmacist with a menu of tradeoffs: "Schedule A prioritizes warfarin efficacy; Schedule B prioritizes atorvastatin efficacy; Schedule C is the balanced compromise."

**Theorem III (compositional schedule synthesis).** When N guidelines share CYP enzymes, safe schedules for enzyme groups can be synthesized independently and combined if the inter-group scheduling constraints are compatible (no timing conflicts between enzyme-group schedules). Define compatibility as: for each drug appearing in multiple enzyme groups, its scheduled administration times agree across groups. Prove that checking compatibility and merging group schedules is O(N · log N) (sorting administration times and checking feasibility).

**Why necessary:** without compositionality, synthesis for 10+ guidelines faces the product-game explosion. With enzyme-group decomposition, synthesis scales to realistic polypharmacy.

### Best-Paper Potential

**The memorable result:** "RxScheduler synthesizes provably safe medication schedules for patients on 10+ concurrent guidelines, producing Pareto-optimal timing protocols that maintain therapeutic efficacy while eliminating pharmacokinetic hazards — the first system that resolves polypharmacy conflicts rather than merely reporting them." This flips the narrative from defensive ("here are your problems") to constructive ("here is how to make it work"). The game-theoretic framing is novel and mathematically interesting. The Pareto-optimal schedule theorem produces a clinically actionable menu of tradeoffs.

**Target venue:** HSCC (Hybrid Systems: Computation and Control, 8–15% P(best paper) — natural fit for timed games with ODE dynamics), CAV (5–8% — game-based synthesis track), AIME (10–15% — if framed around clinical pharmacist utility), AAAI (3–5% — health track with AI planning/optimization angle).

**Weakness:** The approach is the most technically ambitious and the most likely to hit fundamental tractability barriers. If the adversary extremalization produces 2^p scenarios for p > 15 independent PK parameters, synthesis becomes intractable without aggressive decomposition.

### Hardest Technical Challenge

**Exponential blowup of extremal adversary strategies.** For N drugs with independent PK parameters, the adversary has 2^N extremal parameter vectors. At N=10, that's 1024 scenarios; at N=20, it's 10^6 — each requiring a full scheduling optimization.

**How to address it:** (a) *Enzyme-group decomposition:* drugs sharing the same CYP enzyme have correlated PK parameters (inhibited clearance is correlated through shared enzyme activity), reducing independent parameters from N to the number of enzyme groups (~4–6 for typical polypharmacy). This brings extremal adversary count to 2^6 = 64, manageable. (b) *Counterexample-guided synthesis (CEGS):* start with nominal PK parameters, synthesize a schedule, then ask "can the adversary break this schedule?" (a robustness verification subproblem). If broken, add the breaking parameter vector and re-synthesize. This iterative approach converges because the polytope of safe schedules is convex under Metzler dynamics and finitely determined — each counterexample cuts off a half-space from the schedule polytope, and the number of cuts is bounded by the number of non-redundant active constraints. (c) *Lazy adversary evaluation:* for many drug pairs, the nominal-parameter schedule is robust to the full parameter range (because the drug pair doesn't share CYP enzymes and doesn't interact). Only drug pairs with tight therapeutic windows and shared enzymes require full adversarial analysis. A screening pass identifies these pairs in O(N²) time, and full game solving is applied only to the critical subset.

### Scores

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Value | 9 | Prescriptive output is what clinicians actually want; preserves therapeutic intent instead of just flagging problems |
| Difficulty | 9 | Hybrid game synthesis is genuinely hard; novel decidability claim; multi-objective optimization within games |
| Potential | 8 | Dramatic result if it works; novel game formulation; bridges game theory + pharmacology + CDS |
| Feasibility | 4 | Decidability of PTG is a conjecture pending proof; adversary blowup may be intractable for complex cases; ~70K novel LoC |

---

## Comparative Summary

| Dimension | Approach 1: PTA-Contract | Approach 2: Abstract Interp. | Approach 3: Safety Games |
|-----------|--------------------------|------------------------------|--------------------------|
| **Foundation** | Timed automata + model checking | Lattice theory + fixed-point computation | Game theory + strategy synthesis |
| **Output** | Safety certificate or ranked conflict list with counterexamples | Concentration-interval safety bounds per drug | Pareto-optimal safe medication schedules |
| **Composition** | Enzyme-interface A/G contracts | Reduced abstract domain product | Enzyme-group game decomposition |
| **Scalability bottleneck** | Product-automaton state explosion | Widening precision loss | Extremal adversary blowup |
| **Primary venue** | AIME / AMIA | SAS / POPL / VMCAI | HSCC / CAV |
| **Value** | 6 | 7 | 9 |
| **Difficulty** | 8 | 7 | 9 |
| **Potential** | 6 | 7 | 8 |
| **Feasibility** | 6 | 7 | 4 |
| **Composite** | 6.5 | 7.0 | 7.5 (risk-adjusted: 6.0) |

### Recommendation

**Approach 2 (Abstract Interpretation)** offers the best risk-adjusted profile: comparable value and potential to Approach 1 with better feasibility, a cleaner theoretical contribution, and a dramatic speed story. It avoids the high-variance E1 gamble that Approach 1 depends on.

**Approach 3 (Safety Games)** offers the highest ceiling — prescriptive schedule synthesis is transformatively more valuable than diagnostic conflict detection — but carries substantial feasibility risk. It is recommended as a second-phase project if the PTG decidability proof succeeds.

**Approach 1 (PTA-Contract)** is the baseline closest to the original vision. Its composite score is the lowest because the depth check identified significant risks (E1 temporal ablation gamble, zero demand signal) that the other approaches partially mitigate through different framings. However, it has the most mature problem statement and evaluation plan, and the contract composition theorem (Theorem 3) is the single strongest individual mathematical contribution across all three approaches.

A hybrid strategy is possible: implement the abstract interpretation core (Approach 2) for fast screening, then use the PTA model checker (Approach 1's Subsystems 6–8) as a precision backend for cases flagged as "possibly unsafe" by the abstract analysis. This two-tier architecture combines Approach 2's speed with Approach 1's counterexample precision.
