# Math Depth Assessment — GuardPharma

**Slug:** `guideline-polypharmacy-verify`
**Date:** 2025-07-18
**Assessor role:** Math Depth Assessor — evaluating whether the mathematics in each approach is load-bearing or ornamental

---

## Approach 1: PTA-Contract Compositional Model Checking

### A. Load-Bearing Analysis

#### Proposition 1: δ-Decidability of PTA Reachability

| Criterion | Assessment |
|-----------|------------|
| **Genuinely new?** | **No.** This is explicitly an application of Gao et al.'s dReal δ-decidability framework to PK dynamics. The document is honest about this (credit for self-awareness). The sole novel content is choosing δ = 0.1 μg/mL as "pharmacologically meaningful." That is a calibration decision, not a theorem. |
| **Software breaks without it?** | **Weakly yes.** Without *some* decidability guarantee, the verifier may not terminate. But the guarantee is inherited from the dReal/CAPD backend — the system gets δ-decidability for free by using an existing validated integrator. GuardPharma does not prove anything new; it invokes a library. |
| **Proof difficulty** | **2/10.** One sentence of insight: "PK dynamics are Metzler, Metzler dynamics are within dReal's fragment, therefore δ-decidable." The domain-specific δ-calibration is an observation, not a proof. |
| **Risk of being wrong?** | **Very low.** dReal is well-established. The application is straightforward. |
| **Risk of being vacuous?** | **Medium.** δ-decidability says "unsafe or δ-safe." If clinically important distinctions fall within δ (e.g., the difference between therapeutic and toxic is 0.08 μg/mL for a narrow-therapeutic-index drug like digoxin), the guarantee is vacuous for that drug. The document's claim that δ = 0.1 μg/mL is universally "pharmacologically meaningful" is dubious — NTI drugs have much tighter margins. |

**Verdict: Ornamental as a "Proposition."** The depth check already recommended downgrading from "Theorem" to "Proposition." Even "Proposition" is generous — this is closer to a "Remark" or "Observation." The system *needs* decidability, but it gets it for free from its backend. No new math is contributed here.

---

#### Theorem 2: MTL Model Checking for PTA (PSPACE-completeness)

| Criterion | Assessment |
|-----------|------------|
| **Genuinely new?** | **The complexity result is not new; the PK region graph is a novel engineering contribution.** PSPACE-hardness is trivial (timed automata MTL model checking is already PSPACE-hard, and PTA subsume timed automata). PSPACE membership via the PK region graph is the actual content — but the key bisimulation proof (showing the PK region graph is a sound abstraction) is *unspecified*. The depth check flagged this (M3). |
| **Software breaks without it?** | **No.** Standard timed-automata model checking (UPPAAL-style clock regions) would work — just be slower. The PK region graph is a performance optimization disguised as a theorem. Without it, the system still verifies; it just explores more regions. |
| **Proof difficulty** | **4/10 if completed.** PSPACE-hardness: 1/10 (trivial reduction). PSPACE membership: 4/10 (needs soundness of PK region construction). The bisimulation argument (the hard part) is missing. |
| **Risk of being wrong?** | **Medium.** The soundness of partitioning at clinical thresholds depends on the dynamics between thresholds being "well-behaved" in a sense that needs formal definition. If a drug's concentration oscillates near a threshold due to dosing intervals (which is common — trough concentrations for many drugs dip below therapeutic range before the next dose), the region graph may need to track finer dynamics than clinical thresholds suggest. The proof sketch is plausible but incomplete. |
| **Risk of being vacuous?** | **Low-medium.** PSPACE-completeness is real complexity theory. But calling this a "Theorem" overstates the contribution — the hard part (bisimulation proof) is missing, and the complexity class is inherited from the timed automata subcase. |

**Verdict: Partially load-bearing, but incomplete.** The PK region graph is a genuine engineering insight — partitioning at clinical thresholds rather than infinitesimal clock regions is smart and domain-appropriate. But: (a) the soundness proof is missing, (b) PSPACE-completeness is inherited, and (c) the software doesn't break without it. This is a solid engineering contribution masquerading as a complexity-theoretic result.

---

#### Theorem 3: Contract-Based Compositional Safety (the "crown jewel")

| Criterion | Assessment |
|-----------|------------|
| **Genuinely new?** | **Novel instantiation of a known technique.** Assume-guarantee reasoning (Pnueli 1985, Giannakopoulou 2005, Pacti/Incer 2022) is well-established. The novelty is: (a) identifying CYP-enzyme activity as the interface abstraction, (b) proving Metzler monotonicity enables sound single-pass worst-case guarantee computation, (c) demonstrating this captures ~70% of clinically significant PK interactions. The document is honest about this being an instantiation, not a new technique. The monotonicity insight connecting PK structure to contract soundness is genuine. |
| **Software breaks without it?** | **Yes, absolutely.** Without compositionality, verifying 5+ concurrent guidelines requires product-automaton construction — exponential state explosion that makes the tool practically useless for realistic polypharmacy (average Medicare beneficiary: 6.8 conditions). This is the theorem that makes GuardPharma a tool rather than a toy. |
| **Proof difficulty** | **5/10.** The monotonicity argument for competitive inhibition is real but follows from the structure of Michaelis-Menten kinetics (lower clearance → higher concentration → higher inhibition load — a three-step monotone chain). The fixed-point resolution via worst-case guarantees under Metzler monotonicity is the genuine insight. The restriction to competitive inhibition is a significant limitation (~70% coverage). Extending to non-competitive/mixed inhibition would substantially increase difficulty (7/10+). |
| **Risk of being wrong?** | **Low for competitive inhibition.** The monotonicity proof sketch in the problem statement is convincing — the chain CL↓ → C↑ → inhibition↑ is straightforward Michaelis-Menten. **Medium for the overall claim.** The ~30% of interactions outside the contract framework (enzyme induction, pharmacodynamic interactions) require monolithic fallback, and the boundary between "contract-safe" and "needs monolithic verification" must be correctly classified. Misclassification could produce unsound results. |
| **Risk of being vacuous?** | **Low.** The exponential-to-polynomial reduction is real and necessary. The ~70% coverage claim is empirically calibrated against known CYP-mediated DDI literature. |

**Verdict: Genuinely load-bearing.** This is the real mathematical contribution of Approach 1. It is a novel instantiation (category b) rather than a novel theorem (category a), but it is well-motivated, practically essential, and correctly scoped. The depth check's assessment of this as the "crown jewel" is accurate. The main weakness is the restriction to competitive inhibition — the monotonicity argument breaks for other mechanisms, and the document is honest about this.

---

### B. Math Depth Score: Approach 1

**Overall: 4.5/10**

- Proposition 1: 1.5/10 (known framework applied with a calibration choice)
- Theorem 2: 3.5/10 (novel engineering insight, but incomplete proof and inherited complexity class)
- Theorem 3: 6/10 (novel instantiation, load-bearing, practically essential)

The math is concentrated in Theorem 3. Proposition 1 and Theorem 2 are padding — not in the sense of being unnecessary, but in the sense of not contributing new mathematical knowledge. The system *needs* decidability and model checking, but it gets them from existing techniques. Theorem 3 is where original mathematical thinking occurs.

**Category distribution:** (a) No genuinely novel theorems. (b) One novel instantiation (Theorem 3). (c) Two standard applications (Proposition 1, Theorem 2).

---

### C. Red Flags: Approach 1

1. **Overclaiming on Proposition 1.** Even "Proposition" is generous for what amounts to "we used dReal." The δ-calibration at 0.1 μg/mL is presented as universally meaningful but fails for narrow-therapeutic-index drugs (digoxin, warfarin, lithium, phenytoin) where clinically relevant concentration differences can be <0.1 μg/mL in appropriate units.

2. **Missing bisimulation proof for Theorem 2.** The depth check flagged this (M3). Without it, the PSPACE membership claim is unproven. The document says "the correctness argument … is the technical core" — but doesn't provide it.

3. **Enzyme-interaction misclassification risk.** The boundary between contract-eligible interactions (competitive CYP inhibition) and monolithic-fallback interactions is critical for soundness. If a non-competitive interaction is incorrectly routed through the contract path, the monotonicity assumption fails and the result is unsound. No mechanism is described for validating this classification.

4. **Missing math that IS needed:** A formal semantics for the CQL-to-PTA compilation would be genuinely novel and load-bearing (the document hints at this: "implicitly creates the first formal semantics of CQL"). This is a harder and more valuable mathematical contribution than any of the three stated results. It is conspicuously absent from the theorem list.

---

## Approach 2: Abstract Interpretation over Pharmacokinetic Lattices

### A. Load-Bearing Analysis

#### Theorem A: Galois Connection for PK Dynamics

| Criterion | Assessment |
|-----------|------------|
| **Genuinely new?** | **No.** Galois connections for interval domains are textbook Cousot & Cousot (1977). The claim that "Metzler dynamics preserve interval structure" (the image of a box under Metzler flow is a computable box) is a known property of positive linear systems — it is exactly *why* interval/zonotopic methods work for them. This is in any positive systems textbook (Farina & Rinaldi 2000). The Galois connection construction is routine: componentwise interval hull as abstraction, Cartesian product as concretization. |
| **Software breaks without it?** | **Yes, for formal soundness.** Without the Galois connection, there is no guarantee that abstract safety implies concrete safety. But the theorem is trivially true — it would be remarkable if it *didn't* hold. |
| **Proof difficulty** | **2/10.** Textbook construction. The domain-specific content (Metzler monotonicity for interval preservation) is a known property being cited, not proved. |
| **Risk of being wrong?** | **Essentially zero.** This is standard abstract interpretation theory applied to a standard dynamical system class. |
| **Risk of being vacuous?** | **Low.** It's necessary for soundness. But its *novelty contribution* is zero — nobody would question that an interval domain over positive linear systems forms a Galois connection. |

**Verdict: Necessary scaffolding, zero novelty.** This is the mathematical equivalent of proving that your programming language has a type system that is sound. Necessary, but not a contribution. Anyone writing an abstract interpretation for PK dynamics would derive this as a warm-up exercise.

---

#### Theorem B: PK-Aware Widening with Bounded Convergence

| Criterion | Assessment |
|-----------|------------|
| **Genuinely new?** | **Partially.** Domain-specific widening operators are a known technique (Cousot 1978, numerous applications since). The PK-specific design — using lethal dose bounds and steady-state convergence under Metzler eigenvalues — is a novel instantiation. The bound of D iterations (D = number of drugs) is the real claim. |
| **Software breaks without it?** | **Yes.** Without controlled widening, the analysis either (a) doesn't terminate (no widening), or (b) terminates with useless results (standard widening: every interval goes to [−∞, +∞], meaning "every drug could be at any concentration" — formally sound but practically worthless). This theorem makes the tool *useful*. |
| **Proof difficulty** | **5/10.** The D-iteration convergence bound requires arguing that each drug independently reaches its worst-case steady state under Metzler dynamics. This is plausible for non-interacting drugs but less obvious for enzyme-coupled drugs. If drug A's steady state depends on drug B's concentration (through CYP inhibition), and B's steady state depends on A, the convergence to a mutual steady state might require more than D iterations — it might require a fixed-point iteration whose convergence rate depends on the spectral radius of the Jacobian of the coupled system. The claim "D iterations suffice" may be wrong for strongly coupled drugs. |
| **Risk of being wrong?** | **Medium-high for the D-iteration bound as stated.** For D non-interacting drugs, convergence in D iterations is trivial (each drug converges independently in 1 iteration). For enzyme-coupled drugs, the coupled steady-state computation is a fixed-point problem whose iteration count depends on the coupling strength. Strongly coupled drugs (e.g., two potent CYP3A4 inhibitors) may require O(D²) or more iterations. The document doesn't address this. |
| **Risk of being vacuous?** | **Low if the bound is correct.** The PK-aware widening that preserves therapeutic-vs-toxic precision is genuinely useful. |

**Verdict: The most load-bearing theorem in Approach 2, but the convergence bound is suspect.** The widening design is the real contribution — it's where domain expertise meets abstract interpretation theory. But the D-iteration convergence claim needs a more careful treatment of enzyme coupling. I suspect the correct bound is O(D²) or O(D · k) where k is the maximum enzyme-coupling degree, which is still good but different from the claimed O(D).

---

#### Theorem C: Reduced Enzyme-Coupling Product

| Criterion | Assessment |
|-----------|------------|
| **Genuinely new?** | **No.** Reduced product domains are standard in abstract interpretation (Cousot & Cousot 1979, "Systematic design of program analysis frameworks"). The enzyme-coupling structure provides an obvious decomposition — non-interacting drugs are independent, interacting drugs share enzyme state. This is a standard application of reduced products to a problem with natural independence structure. |
| **Software breaks without it?** | **No.** Without the reduced product, the analysis uses the full Cartesian product. For 20 guidelines, this is slower and may lose precision, but it still works. The reduced product is a performance optimization. |
| **Proof difficulty** | **2/10.** Standard reduced product construction. The soundness follows from independence of non-interacting drugs, which is definitional (non-interacting means their PK dynamics are decoupled). |
| **Risk of being wrong?** | **Very low.** |
| **Risk of being vacuous?** | **Low-medium.** Useful for performance, but this is standard technique. |

**Verdict: Standard application, no novelty.** The enzyme-coupling decomposition is obvious once you know abstract interpretation and pharmacology. It's good engineering but not a mathematical contribution.

---

### B. Math Depth Score: Approach 2

**Overall: 3.5/10**

- Theorem A: 1.5/10 (textbook Galois connection)
- Theorem B: 5/10 (novel widening design, but convergence bound is suspect)
- Theorem C: 2/10 (standard reduced product)

Approach 2 presents itself as having "three crisp theorems, each load-bearing." In reality, Theorems A and C are routine, and Theorem B is the only one with genuine content. The approach document's self-assessment of "cleaner theoretical contribution" than Approach 1 is backwards — Approach 1's Theorem 3 (contract composition with Metzler monotonicity) is a more novel and deeper contribution than anything in Approach 2.

**However**, Approach 2's math is more *honest* in the sense that it doesn't overclaim. The Galois connection and reduced product are necessary scaffolding, correctly identified as such. The approach doesn't try to make routine results sound impressive.

**Category distribution:** (a) No genuinely novel theorems. (b) One novel instantiation (Theorem B, PK-aware widening). (c) Two standard applications (Theorems A and C).

---

### C. Red Flags: Approach 2

1. **D-iteration convergence bound (Theorem B) is likely wrong for coupled drugs.** The claim that widening converges in D iterations assumes each drug converges independently. For enzyme-coupled drugs, convergence is a coupled fixed-point problem. Strongly inhibiting drug pairs could require more iterations. This should be stated as a conjecture or given a tighter analysis accounting for coupling.

2. **Precision claim is unvalidated.** The entire approach hinges on the abstract domain being precise enough to distinguish safe from unsafe. The hardest technical challenge section acknowledges this: "If the abstract transformer is too imprecise for enzyme-coupled drugs, it will flag every CYP3A4-sharing combination as 'possibly unsafe.'" This is the real risk — and no theorem addresses it. A theorem bounding the false-positive rate or the precision loss from abstraction would be genuinely novel and load-bearing.

3. **No counterexamples.** The document acknowledges that abstract interpretation cannot produce counterexamples when verification fails — it only says "possibly unsafe." This is a fundamental limitation. A theorem characterizing *when* the abstraction is precise enough to give definitive "safe" or "unsafe" (rather than "possibly unsafe") answers would be valuable and is missing.

4. **Overclaiming "cleaner theoretical contribution."** The approaches document rates Approach 2's Potential at 7 vs. Approach 1's 6, partly based on "cleaner theoretical contribution (3 tight theorems)." Two of the three theorems are textbook. Approach 1's single real theorem (Theorem 3) is deeper than all three of Approach 2's combined.

5. **Missing math that IS needed:** A precision analysis — bounding the gap between the abstract and concrete domains for clinically relevant drug combinations — is the critical missing piece. Without it, the approach might produce "possibly unsafe" for every non-trivial drug combination, rendering it useless.

---

## Approach 3: Pharmacokinetic Safety Games with Safe-Schedule Synthesis

### A. Load-Bearing Analysis

#### Theorem I: Decidability of PK Safety Games

| Criterion | Assessment |
|-----------|------------|
| **Genuinely new?** | **Yes, if true.** Hybrid games with ODE dynamics are generally undecidable. The claim that Metzler structure enables decidability via adversary extremalization is genuinely novel. The proof strategy (extremalize adversary via monotonicity → discretize scheduler → reduce to finite game) is creative and non-obvious. |
| **Software breaks without it?** | **Absolutely.** Without decidability, the synthesis algorithm may not terminate. This is the mathematical foundation of the entire approach. |
| **Proof difficulty** | **7/10.** Three non-trivial steps: (1) adversary extremalization requires proving that optimal adversary strategies for Metzler systems are extremal — this is plausible but needs a formal minimax argument over the continuous parameter space; (2) scheduler discretization requires bounding the grid granularity needed, which depends on PK time constants in a way that needs careful analysis; (3) the finite game reduction must show that the discrete game preserves the safety property of the continuous game. Each step is non-trivial, and the composition of all three is a real proof. |
| **Risk of being wrong?** | **HIGH.** The approaches document itself rates feasibility at 4/10 and acknowledges this is a "conjecture pending proof." Specific concerns: (a) Adversary extremalization assumes that worst-case PK parameters are constant across the treatment horizon. In reality, PK parameters can change (disease progression, renal function changes) — if the adversary can *adapt* parameters over time, extremalization may fail. (b) The scheduler discretization bound depends on PK time constants, but drugs with very different time constants (e.g., amiodarone t½ = 58 days vs. metformin t½ = 6 hours) create a stiff system where the grid must be very fine for fast drugs and very long for slow drugs — potentially making the grid size unmanageable. (c) The 2^p extremal parameter vectors could be exponentially many for complex polypharmacy. |
| **Risk of being vacuous?** | **Low if true.** This would be a genuine decidability result for an important class of hybrid games. |

**Verdict: The most ambitious mathematical claim across all three approaches. Potentially a significant contribution, but likely wrong as stated.** The adversary extremalization is the key step, and it makes strong assumptions about the adversary model (static parameters, only competitive inhibition, Metzler structure throughout). The depth check's skepticism is well-founded. If the proof goes through under restricted assumptions (static parameters, single enzyme group, competitive inhibition only), it would still be a meaningful result — but the generality claimed is too broad.

---

#### Theorem II: Pareto-Optimal Safe Schedules

| Criterion | Assessment |
|-----------|------------|
| **Genuinely new?** | **Partially, but the key claim is likely false.** Multi-objective optimization in constrained settings is well-studied. The specific claim that the Pareto set of safe schedules forms a polytope with at most D+1 vertices is the novel content — and it is almost certainly wrong. |
| **Software breaks without it?** | **No.** You can compute Pareto-optimal schedules numerically without this characterization. The polytope structure would enable efficient enumeration, but numerical Pareto computation works without it. |
| **Proof difficulty** | **6/10 if true.** The polytope characterization would require showing that the safety constraint and therapeutic efficacy objectives interact linearly in schedule-parameter space. For linear objectives over linear constraints, the Pareto set is indeed a polytope (fundamental theorem of linear programming applied to multi-objective LP). But therapeutic efficacy E_i(σ) — the fraction of time drug i is in its therapeutic window — is a *nonlinear* function of the schedule σ (it's a piecewise integral of PK dynamics). Nonlinear objectives over convex constraints do NOT generally produce polytopic Pareto fronts. |
| **Risk of being wrong?** | **VERY HIGH.** The D+1 vertex claim assumes linearity that almost certainly doesn't hold. Therapeutic efficacy as a function of dosing time involves piecewise-exponential integrals (from compartmental PK dynamics). The Pareto front of piecewise-exponential objectives over box constraints is generically a curved manifold, not a polytope. The D+1 bound is a property of multi-objective LP, not multi-objective nonlinear programming. **This theorem is likely false as stated.** |
| **Risk of being vacuous?** | **N/A — likely incorrect.** |

**Verdict: RED FLAG — this theorem is likely false.** The polytope/vertex claim appears to conflate multi-objective linear programming (where Pareto sets are polytopes) with multi-objective nonlinear programming (where they are not). The complexity bound O(D · |grid|^D · 2^p) is honestly stated as exponential in D, which already undermines the tractability narrative. The D+1 vertex claim would save this, but it is almost certainly wrong.

**What would be correct:** The Pareto front for PK scheduling objectives is likely a semialgebraic set (defined by polynomial/exponential inequalities) that can be approximated by a polytope with more than D+1 vertices. A theorem characterizing the *smoothness* or *dimension* of the Pareto front (rather than claiming it's a polytope) would be more defensible.

---

#### Theorem III: Compositional Schedule Synthesis

| Criterion | Assessment |
|-----------|------------|
| **Genuinely new?** | **Novel instantiation.** The enzyme-group decomposition mirrors Approach 1's Theorem 3, applied to the game setting rather than the verification setting. The O(N · log N) compatibility check is a simple sorting argument. The novelty is modest — it's the game-synthesis version of a decomposition already proposed in Approach 1. |
| **Software breaks without it?** | **Yes, for scalability.** Without decomposition, synthesis for 10+ guidelines faces product-game explosion. |
| **Proof difficulty** | **3-4/10.** The decomposition follows from independence of non-interacting enzyme groups. The compatibility check (that a drug appearing in multiple enzyme groups gets consistent administration times) is a constraint satisfaction problem, not an optimization problem — the O(N · log N) claim is for checking, not for resolving conflicts. If conflicts exist (a drug needs different timing in different groups), the theorem doesn't say how to resolve them. |
| **Risk of being wrong?** | **Low for the basic claim.** Medium for practical applicability — drugs often appear in multiple enzyme groups (e.g., a drug metabolized by both CYP3A4 and CYP2D6), and the compatibility check may frequently fail, requiring fallback to monolithic synthesis. |
| **Risk of being vacuous?** | **Medium.** If compatibility frequently fails for realistic polypharmacy, the decomposition doesn't help in practice. |

**Verdict: Modest contribution.** Sound but not deep. The enzyme-group decomposition is a natural idea; the O(N · log N) compatibility check is trivial; the hard problem (what to do when compatibility fails) is punted.

---

### B. Math Depth Score: Approach 3

**Overall: 6/10 (ceiling), 3.5/10 (expected)**

- Theorem I: 8/10 if true, but likely 4/10 after accounting for the high probability of being wrong or requiring severe restrictions
- Theorem II: 2/10 (likely false as stated)
- Theorem III: 3.5/10 (modest instantiation of decomposition idea)

Approach 3 has the highest mathematical *ambition* but the worst *reliability*. Theorem I, if proven under appropriate restrictions, would be a genuine contribution to hybrid games theory. Theorem II is likely false. Theorem III is modest. The expected math depth, accounting for the probability that Theorem I fails or requires severe restrictions and Theorem II is wrong, is ~3.5/10.

**Category distribution:** (a) One potentially novel theorem (Theorem I — IF it survives scrutiny). (b) One novel instantiation (Theorem III). (c) One likely-false claim (Theorem II).

---

### C. Red Flags: Approach 3

1. **Theorem II is likely false.** The polytope/D+1 vertex claim for the Pareto set of safe schedules appears to assume linearity that doesn't hold for PK dynamics. This is the most serious mathematical overclaim across all three approaches.

2. **Theorem I is a conjecture presented as a theorem.** The approaches document acknowledges "decidability of PTG is a conjecture pending proof" but still labels it "Theorem I." It should be labeled "Conjecture I" until proved. The proof strategy is plausible but has at least three non-trivial gaps.

3. **Adversary model is unrealistic.** Theorem I assumes the adversary chooses PK parameters from a bounded set and is Metzler-constrained. Real PK variability includes time-varying parameters (disease progression, drug-drug induction effects that emerge over weeks). A static adversary choosing parameters once at the start is a straw adversary that doesn't capture the most dangerous clinical scenarios.

4. **Complexity is honestly exponential.** O(D · |grid|^D · 2^p) is exponential in the number of drugs (D) and the number of PK parameters (p). For D = 5 and a grid of 100 time points, |grid|^D = 10^10 — infeasible even before the 2^p factor. The enzyme-group decomposition (Theorem III) reduces this, but only if decomposition succeeds — and the document doesn't analyze how often it will.

5. **Missing math that IS needed:** A robustness analysis for synthesized schedules — quantifying how much a safe schedule can tolerate PK parameter deviations beyond the assumed bounds — would be clinically essential and mathematically interesting. This is absent.

---

## D. Comparative Recommendations

### Math Depth Summary

| | Approach 1 | Approach 2 | Approach 3 |
|---|---|---|---|
| **Depth Score** | **4.5/10** | **3.5/10** | **6/10 ceiling, 3.5/10 expected** |
| **Strongest theorem** | Theorem 3 (contract composition) | Theorem B (PK widening) | Theorem I (game decidability) |
| **Weakest theorem** | Proposition 1 (δ-decidability) | Theorem A (Galois connection) | Theorem II (Pareto polytope — likely false) |
| **Novel theorems (category a)** | 0 | 0 | 0–1 (Theorem I, if it survives) |
| **Novel instantiations (category b)** | 1 (Theorem 3) | 1 (Theorem B) | 1 (Theorem III) |
| **Standard applications (category c)** | 2 | 2 | 0 |
| **Likely-false claims** | 0 | 0 | 1 (Theorem II) |
| **Overclaimed results** | Proposition 1 | "3 tight theorems" framing | Theorem I ("theorem" vs conjecture), Theorem II |

### Best Math-to-Value Ratio

**Approach 1 wins.** The reason is simple: Theorem 3 (contract-based composition) is the single deepest and most practically essential mathematical contribution across all nine theorems/propositions. It is:
- Novel (first A/G framework for CYP-enzyme interfaces)
- Load-bearing (the system is useless without it for realistic polypharmacy)
- Correctly scoped (honest about competitive-inhibition restriction)
- Practically motivated (maps to real pharmacology)
- Provably correct (the monotonicity argument is convincing for competitive inhibition)

Approach 2's math is shallower but more *honest* — it doesn't overclaim. Approach 3's math is more ambitious but less reliable — it overclaims (Theorem II) and bets on an unproved conjecture (Theorem I).

### Mathematical Strengthening Recommendations

#### Approach 1
1. **Add a formal CQL-to-PTA compilation correctness theorem.** This is the missing math that would be most valuable. A bisimulation or trace-equivalence result between CQL semantics and the compiled PTA would be genuinely novel (first formal semantics of CQL) and highly load-bearing (compilation bugs produce unsound verification). Proof difficulty: 6-7/10. This would be a stronger contribution than Proposition 1 and Theorem 2 combined.
2. **Complete the bisimulation proof for Theorem 2.** Without it, PSPACE membership is unproven.
3. **Extend Theorem 3 to competitive + time-dependent inhibition.** The current restriction to competitive inhibition covers ~70%. Adding time-dependent inhibition (mechanism-based inactivation, which is irreversible) would increase coverage to ~85% and require a more sophisticated monotonicity argument (proof difficulty: 7/10).

#### Approach 2
1. **Prove a precision bound.** A theorem bounding the false-positive rate (fraction of abstract "possibly unsafe" that are concretely safe) for clinically representative drug combinations would transform the approach from "nice theory" to "useful tool." This is the critical missing math.
2. **Fix the convergence bound in Theorem B.** Analyze the coupled case properly — the bound is probably O(D · k) where k is the maximum enzyme-coupling degree, not O(D).
3. **Add a counterexample recovery procedure.** When the abstract analysis says "possibly unsafe," provide a concrete witness using a targeted concrete analysis in the flagged region. A theorem showing this targeted analysis is decidable would bridge the gap between abstract interpretation and model checking.

#### Approach 3
1. **Downgrade Theorem I to a conjecture and prove it for a restricted subclass.** Static adversary, single enzyme group, competitive inhibition only, constant PK parameters. This restricted version is likely provable (difficulty: 5/10) and still novel.
2. **Retract or substantially weaken Theorem II.** Replace the polytope/D+1 vertex claim with a characterization of the Pareto front's dimension and smoothness. Or prove the polytope claim only for the restricted case where PK dynamics are linearized around steady state.
3. **Add a robustness theorem.** Quantify how much PK parameter variation a synthesized schedule can tolerate. This would be both novel and clinically essential.

---

## Overall Verdict

None of the three approaches contributes genuinely novel mathematics in the strong sense (new techniques, new complexity results, new decidability boundaries). All three are **novel instantiations of known frameworks applied to pharmacokinetic systems**. This is not damning — many good papers are instantiations — but none should overclaim novelty.

The mathematical depth ranking is:

1. **Approach 1** — best math-to-value ratio. One genuinely deep instantiation (Theorem 3) that is load-bearing, correctly scoped, and provably correct. Two ornamental/routine results that should be downgraded.

2. **Approach 3** — highest ceiling but highest risk. Theorem I is the most ambitious claim and would be the most impressive result if proved, but it's a conjecture with multiple gaps. Theorem II is likely false. The expected value after accounting for proof-failure risk is similar to the other approaches.

3. **Approach 2** — shallowest math, but most honest. No overclaiming, no likely-false results, but also no deep contributions. The PK-aware widening (Theorem B) is the only non-routine element, and even it has a suspect convergence bound.

**The uncomfortable truth:** The mathematical value of all three approaches is primarily in their *domain synthesis* — connecting formal methods to pharmacokinetics in ways that are genuinely new — rather than in any individual theorem. The theorems are largely instantiations of known techniques. The deepest *mathematical* contribution would be a formal semantics of CQL (compilation correctness), which none of the approaches proposes as a theorem despite all of them depending on it.
