# Approach Debate — GuardPharma

**Slug:** `guideline-polypharmacy-verify`  
**Date:** 2025-07-18  
**Personas:** Adversarial Skeptic + Mathematician Critic  
**Purpose:** Ruthlessly stress-test all three approaches to make the survivor bulletproof.

---

## Approach 1: PTA-Contract Compositional Model Checking

### Adversarial Skeptic

**Fatal Flaws**

1. **The E1 temporal ablation experiment is a coin flip masquerading as a research plan.** The depth check estimates the temporal-only detection rate X at 10–30%. The paper's *entire narrative* — "temporal PK reasoning discovers dangers invisible to existing approaches" — requires X ≥ 20%. If X lands at 12%, you've built a 65K-novel-LoC cathedral to discover that an atemporal checker with a DrugBank lookup gets 88% of the same results. The fallback narratives ("explanation quality," "existence proof") are consolation prizes, not best-paper arguments. Mechanism of death: E1 produces X ≈ 12%, reviewers write "the machinery is impressive but the marginal value over simpler approaches is unclear," and the paper is a strong reject at every target venue.

2. **Zero demand signal from any actual stakeholder.** The value proposition table lists five stakeholder categories — hospital CDS committees, EHR vendors, guideline developers, regulators, patients — but not one of them has asked for this. Not a letter of interest, not a conversation, not an email. The proposal describes hypothetical needs, not expressed needs. Every EHR vendor I'm aware of (Epic, Oracle Health, Cerner legacy) deploys CDS interaction checking via proprietary rulesets and Lexicomp/Micromedex lookups. They are not waiting for PTA-based formal verification. Mechanism of death: the tool is built, works correctly, and nobody adopts it because the workflow integration problem (getting formal verification output into existing CDS committee processes) was never solved — because nobody was consulted.

**Serious Flaws**

1. **CQL treatment-logic adoption is near-zero in production.** The pipeline is designed to verify CQL-encoded treatment guidelines, but the honest count of production CQL treatment-decision artifacts is ~30–50, with the vast majority being research prototypes or CMS quality measures (which are retrospective, not prospective treatment logic). You're building a verification engine for a corpus that barely exists. The ONC HTI-1 mandate is real but it mandates FHIR *interoperability*, not CQL *treatment logic*. Most CDS at scale is still proprietary Epic/Oracle rules.

2. **CEGAR convergence is faith-based.** The CEGAR loop's convergence on PK-structured state spaces is "unproven" (the difficulty assessor's word). The SAT-based bounded model checking fallback is sound but potentially slow. No estimate exists for how often the fallback is needed or how long it takes. If CEGAR fails to converge on 40% of realistic guideline pairs and the BMC fallback takes 45 minutes each, the tool is unusable for the evaluation.

3. **The 175K→135K→95K→25-35K LoC telescope.** The proposal started at 175K, the depth check shrank it to 135K (honest total) with 83K novel, the paper phase is 95K, and the minimum viable artifact is 25–35K. That's an 80% reduction from headline to reality. This isn't necessarily bad (honest scoping is good), but it means the *paper will describe a prototype*, not a system. Reviewers at clinical venues (AMIA, JAMIA) expect working systems with clinical evaluation, not proof-of-concept prototypes with 3 hand-encoded guideline pairs.

**Hidden Assumptions**

- *Guidelines can be faithfully encoded as timed automata.* No one has done this. The pilot is supposed to validate this assumption, but the pilot itself is manually encoding 3 pairs — which proves only that a skilled researcher can do it, not that the process is generalizable.
- *Population PK parameters are available for all drugs of interest.* Compartmental PK models with published parameters exist for common drugs (metformin, warfarin, atorvastatin) but may not exist for newer drugs, combination formulations, or drugs used off-label — which are precisely the drugs most likely to appear in complex polypharmacy.
- *CYP-mediated competitive inhibition captures the interaction mechanisms that matter.* The ~70% coverage claim comes from the pharmacology literature on PK DDIs specifically. But the clinically most dangerous polypharmacy interactions include PD interactions (QT prolongation from multiple QT-prolonging drugs, serotonin syndrome from multiple serotonergic drugs) that are completely outside the contract framework. A patient harmed by QT-prolongation-induced torsades de pointes won't be consoled that the system correctly verified CYP2D6 interactions.

**Demand Signal Challenge**

Who has actually asked for formal verification of clinical guidelines? The answer is *nobody*. The Joint Commission cites CDS configuration errors, but their recommended solution is better human review processes, not formal verification. ONC mandates CDS interoperability, not CDS safety verification. The FHIR community is focused on data exchange, not formal safety guarantees. This is a solution looking for a problem that the domain experts haven't articulated as needing this specific solution. The "verify before deploy" framing is intellectually compelling but practically disconnected from how CDS governance actually works in health systems.

**Competitive Threat**

- **LLMs + drug interaction databases (DrugBank, Lexicomp, Clinical Pharmacology).** GPT-4 with RAG over DrugBank flags most DDIs with mechanism explanations. The formal-vs-probabilistic distinction is real but the practical margin is narrow for the ~70% of interactions that are well-characterized CYP-mediated DDIs.
- **EHR-embedded interaction checking.** Epic's CDS already runs interaction alerts at prescription time. It doesn't do exhaustive multi-guideline verification, but it catches the vast majority of dangerous combinations at the point of care.
- **Regulatory evolution.** If ONC or CMS mandates CDS safety testing, they're more likely to mandate test-case-based validation (like FDA software validation) than formal verification. Formal verification is alien to medical device regulation.

**Best-Paper Killer**

"The paper presents an impressive formal framework (PTA formalism, contract composition, CEGAR with clinical abstractions) applied to 3 hand-encoded guideline pairs. The temporal ablation experiment shows that X% of conflicts require temporal reasoning, but X=14%, meaning an atemporal checker with DrugBank gets 86% of the benefit at 1% of the complexity. No clinical validation is provided. The mathematical contributions are individually thin — Proposition 1 is a library invocation, Theorem 2 has an incomplete proof, and Theorem 3, while novel, is an instantiation of well-known assume-guarantee reasoning. The system cannot be used by any clinical stakeholder in its current form. Reject."

**Verdict: CONDITIONAL CONTINUE**

The project survives if: (1) the pilot succeeds (PTA encoding works, model checker terminates), (2) E1 delivers X ≥ 20%, (3) minimal clinical validation is added. Without all three, this is a theory paper, not a best paper. The contract composition theorem (Theorem 3) is a genuine contribution worth preserving regardless of outcome.

---

### Mathematician Critic

**Proof Gaps**

1. **Theorem 2's bisimulation proof is completely missing.** PSPACE membership requires showing the PK region graph is a sound abstraction of the continuous dynamics. The "correctness argument" is called "the technical core" — and then isn't provided. Without it, the PSPACE claim is an unsubstantiated assertion, not a theorem. The soundness depends on dynamics between clinical thresholds being "well-behaved," but drug concentrations oscillate near thresholds during dosing intervals (trough concentrations routinely dip below therapeutic range), which could break the region abstraction.

2. **Theorem 3's monotonicity for the fixed-point resolution is sketched but not proved.** The chain CL↓ → C↑ → inhibition↑ is intuitively convincing for competitive inhibition, but the formal argument requires showing this holds *across the entire parameter space* Φ, not just pointwise. For population PK parameters near the boundary of the competitive-inhibition regime, the monotonicity may be marginal or fail for specific parameter combinations.

3. **The δ-calibration at 0.1 μg/mL is pharmacologically naive.** Digoxin has a therapeutic range of 0.8–2.0 ng/mL (note: *nanograms*, not micrograms). The proposed δ = 0.1 μg/mL = 100 ng/mL exceeds digoxin's *lethal* dose. For narrow-therapeutic-index drugs (digoxin, lithium, phenytoin, warfarin in INR terms), a universal δ is meaningless — it must be drug-specific. This isn't a mathematical error, but it reveals that the δ-decidability result's clinical applicability is narrower than claimed.

**Vacuity Risks**

- Proposition 1 is vacuously true in the sense that δ-decidability for bounded Metzler ODEs is an *immediate* consequence of applying dReal's framework. There's no mathematical content beyond "our system falls within the decidable fragment." The δ-calibration adds domain flavor but not mathematical substance.
- Theorem 2 may be vacuously complex: if the practical verification always uses SAT-based bounded model checking (because CEGAR converges faster), the PK region graph construction and its PSPACE-completeness result are theoretically interesting but never executed. A theorem about a code path that's never taken is an ornament.

**Over-Claiming**

- Calling Proposition 1 a "Proposition" overclaims; "Observation" or "Remark" is accurate. The math depth assessor already flagged this; the problem statement implemented the downgrade from "Theorem" to "Proposition" but should go further.
- The claim "first formal semantics of CQL" is made in passing about the compiler but never formalized as a theorem. If it's true, it's a much more valuable mathematical contribution than Proposition 1 — so why isn't it a theorem? Because formalizing it would be genuinely hard (proof difficulty 6–7/10), and the team is avoiding the hard math in favor of easier results.

**Missing Math**

- **CQL compilation correctness theorem.** A bisimulation or trace-refinement result between CQL operational semantics and the compiled PTA. This is the most valuable missing math across all three approaches — it would be genuinely novel (first formal semantics of CQL), highly load-bearing (compilation bugs produce unsound verification), and hard (6–7/10). It is conspicuously absent.
- **Enzyme-interaction classification correctness.** The boundary between contract-eligible (competitive CYP inhibition) and monolithic-fallback interactions must be correctly drawn. Misclassification produces unsound results. No formal mechanism for validating this classification is proposed.

**Depth Verdict**

The math is a **4.5/10** — one genuine contribution (Theorem 3 at 6/10) surrounded by padding (Proposition 1 at 1.5/10, Theorem 2 at 3.5/10). For a clinical venue (AIME, AMIA), this is adequate — the math is secondary to the system's clinical value. For a formal methods venue (CAV, TACAS), this is insufficient — one novel instantiation of A/G reasoning is a workshop paper, not a main-conference contribution. **Strengthening path:** add the CQL compilation correctness theorem and complete Theorem 2's bisimulation proof.

---

## Approach 2: Abstract Interpretation over Pharmacokinetic Lattices

### Adversarial Skeptic

**Fatal Flaws**

1. **The false-positive rate may render the tool useless.** Abstract interpretation over-approximates. If the abstract domain isn't precise enough, every CYP3A4-sharing drug combination is flagged as "possibly unsafe." CYP3A4 metabolizes ~50% of all drugs. So ~25% of all drug pairs would be flagged — a catastrophic false-positive rate that makes the tool clinically worthless. The proposal acknowledges this ("If the abstract transformer is too imprecise… it will flag every CYP3A4-sharing combination as 'possibly unsafe'") but offers no theorem or empirical evidence that precision will be sufficient. **No theorem bounds the false-positive rate.** The entire approach is a bet that the abstract domain will be precise enough, with no formal analysis supporting that bet. Mechanism of death: the tool flags everything as "possibly unsafe," clinical pharmacists ignore it after the first day, and the paper reports precision metrics that are embarrassingly low.

2. **No counterexamples = no actionable output.** When the analysis says "possibly unsafe," it provides no diagnostic information — no patient trajectory, no timeline, no mechanism. A clinical pharmacist needs to know *why* a combination is dangerous and *when* the danger manifests. "Drug X and Drug Y: possibly unsafe" is what they already get from Lexicomp. The proposal's response — report confidence levels ("definitely safe" / "possibly unsafe" / "definitely unsafe") — is a classification, not an explanation. The depth check and difficulty assessor both flag this as a fundamental limitation but the approach offers no remedy. Mechanism of death: the output is indistinguishable from existing drug interaction databases, reviewers at clinical venues ask "what does this add beyond Lexicomp?", and the paper fails the novelty threshold.

**Serious Flaws**

1. **The "three crisp theorems" framing is misleading.** The math depth assessor demolished this: Theorem A (Galois connection) is textbook with zero novelty (1.5/10). Theorem C (reduced product) is standard (2/10). Only Theorem B (PK-aware widening) has genuine content, and its D-iteration convergence bound is "likely wrong for coupled drugs." So the "clean theoretical contribution" is actually one partially-correct theorem plus two warm-up exercises dressed as results.

2. **The speed advantage is real but may not matter.** "50 concurrent guidelines in 0.3 seconds" is impressive — but 50 concurrent guidelines is a scenario that doesn't exist in practice. No patient is governed by 50 concurrent treatment guidelines. A realistic polypharmacy patient has 5–8 active guidelines. Approach 1's contract-based verification handles 20 guidelines in <120 seconds. Is the difference between 0.3 seconds and 120 seconds clinically meaningful? For an interactive tool, maybe. For a pre-deployment verification pipeline, absolutely not — you run it once before deployment, not in a clinical interaction loop.

3. **Formulary-level screening is a distraction.** The proposal positions abstract interpretation's speed advantage as enabling "formulary-level safety screening" — checking thousands of drug combinations per minute. But formularies are curated by pharmacy and therapeutics committees that already screen for interactions using Lexicomp and Clinical Pharmacology. They don't need formal verification of all pairwise combinations; they need analysis of the specific combinations that their clinical population actually receives. The use case is fabricated to justify the speed advantage.

**Hidden Assumptions**

- *Interval abstraction is precise enough for therapeutic-vs-toxic discrimination.* For drugs with a 2× therapeutic window (therapeutic 2–5 μg/mL, toxic >5 μg/mL), the abstract interval must be tighter than this window to distinguish "definitely safe" from "possibly unsafe." Whether Metzler-flow box preservation achieves this precision for enzyme-coupled drugs is *assumed*, not proved.
- *Steady-state convergence occurs within the verification horizon.* The widening operator assumes drugs reach steady state, but amiodarone (half-life 40–55 days) doesn't reach steady state within a 90-day horizon. For such drugs, the widening produces excessively wide intervals — potentially the entire [0, C_lethal] range.
- *Clinical decisions can be faithfully modeled as abstract transformers.* The transformer for "if eGFR < 30, reduce metformin dose to 500mg" requires the abstract domain to track renal function as a variable. But the proposal describes the domain as "concentration intervals × enzyme-load intervals × clinical-state predicates." If renal function is a predicate (eGFR < 30 yes/no), the transformer works. If renal function is a continuous variable (eGFR = 25), the abstract domain must be extended — and every additional continuous variable widens the domain and reduces precision.

**Demand Signal Challenge**

Same as Approach 1 — zero demand signal. The additional stakeholder claim (PBMs using this for formulary screening) is fabricated. PBMs optimize formularies for cost, not safety; safety screening is done by P&T committees using existing tools. No PBM has expressed interest in abstract-interpretation-based drug interaction analysis.

**Competitive Threat**

- **LLMs + databases** (same as Approach 1, but more acute here because the abstract interpretation output — concentration intervals — is less informative than LLM-generated mechanism explanations).
- **Approach 1 as a two-tier backend.** The approaches document itself proposes using abstract interpretation as a fast front-end with model checking as a precision backend. This cannibalization risk means Approach 2 may be demoted to a preprocessing step in a hybrid system rather than standing as an independent contribution.

**Best-Paper Killer**

"The paper applies textbook abstract interpretation (Galois connection, widening, reduced product) to a pharmacokinetic domain. Of the three theorems, two are standard constructions with zero novelty, and the third (PK-aware widening with D-iteration convergence) has a suspect convergence bound for enzyme-coupled drugs. The speed is impressive but the clinical value is unclear: the analysis cannot produce counterexamples, the false-positive rate is unreported, and the output ('possibly unsafe') is less informative than existing drug interaction databases. The paper does not compare against any existing drug interaction tool. Reject."

**Verdict: CONDITIONAL CONTINUE**

The approach survives if: (1) empirical false-positive rates are acceptable (≤15% of flagged combinations are actually safe), (2) the D-iteration convergence bound is fixed for coupled drugs, (3) a counterexample-recovery mechanism is added (targeted concrete analysis in flagged regions). Without these, the theoretical contribution is too shallow and the practical contribution is too imprecise for a best paper.

---

### Mathematician Critic

**Proof Gaps**

1. **Theorem B's D-iteration convergence bound is almost certainly wrong for enzyme-coupled drugs.** For D non-interacting drugs, convergence in D iterations is trivial (each drug converges independently in 1 iteration). For enzyme-coupled drugs, the coupled steady-state computation is a fixed-point problem where the iteration count depends on the spectral radius of the Jacobian of the coupled system. Strongly coupled drug pairs (two potent CYP3A4 inhibitors) may require O(D²) or O(D · k) iterations where k is the maximum enzyme-coupling degree. The stated D-iteration bound silently assumes independence.

2. **Theorem A's Galois connection proof needs careful handling of enzyme coupling.** The claim that "Metzler dynamics preserve interval structure" (box image under Metzler flow is a computable box) is true for diagonal (non-interacting) Metzler systems. For off-diagonal Metzler systems (enzyme coupling), the box image may not be a box — the coupling introduces correlation between components that interval abstraction loses. The proof must show that the over-approximation from losing this correlation doesn't destroy therapeutic-vs-toxic precision. This is assumed but not argued.

**Vacuity Risks**

- Theorem A is trivially true — it would be remarkable if an interval domain over positive linear systems *didn't* form a Galois connection. The novelty content is zero. Presenting this as a "theorem" rather than a "construction" or "observation" inflates the mathematical contribution.
- Theorem C (reduced product) is vacuously standard — the decomposition follows from independence of non-interacting drugs, which is true by definition. This is good engineering, not mathematics.

**Over-Claiming**

The approaches document rates Approach 2's Potential at 7 vs. Approach 1's 6, partly based on "cleaner theoretical contribution (3 tight theorems)." The math depth assessor correctly calls this "backwards" — Approach 1's single real theorem (Theorem 3, contract composition) is deeper than all three of Approach 2's combined. The "3 tight theorems" framing trades quantity for quality and hopes nobody notices.

**Missing Math**

- **A precision bound.** The critical missing theorem: for drug pairs with therapeutic window ratio ≥ R (toxic threshold / therapeutic threshold), the abstract domain's false-positive rate is bounded by f(R, coupling_strength). Without this, the approach has no guarantee it produces useful results.
- **A counterexample recovery theorem.** When the abstract analysis says "possibly unsafe," prove that a targeted concrete reachability analysis (restricted to the flagged abstract region) is decidable and terminates in bounded time. This would bridge abstract interpretation and model checking and could be the approach's real mathematical contribution.

**Depth Verdict**

The math is **3.5/10** — the shallowest across all three approaches. One partially novel result (Theorem B) with a suspect bound, plus two textbook constructions. This is adequate for a tool paper at VMCAI/SAS if the tool works well empirically, but it's not best-paper mathematics at any venue. **Strengthening path:** prove the precision bound and the counterexample recovery theorem — these would be genuinely novel, load-bearing, and would distinguish this from routine abstract interpretation applications.

---

## Approach 3: Pharmacokinetic Safety Games with Safe-Schedule Synthesis

### Adversarial Skeptic

**Fatal Flaws**

1. **Theorem I is a conjecture presented as a theorem. The entire approach depends on proving it. It may be unprovable as stated.** Hybrid games with ODE dynamics are *in general undecidable* (Henzinger et al. 1999). The claim that Metzler structure restores decidability via adversary extremalization is creative but unproven. Three specific gaps: (a) adversary extremalization assumes worst-case PK parameters are *static* across the treatment horizon — if the adversary can adapt parameters over time (reflecting disease progression), the extremalization argument collapses; (b) scheduler discretization requires bounding the grid granularity, but drugs with wildly different half-lives (amiodarone 58 days vs. metformin 6 hours) create a stiff system requiring grid points at 30-minute resolution over 365 days = 17,500 points per drug; (c) the finite game reduction must show that discretization preserves the safety property, which is non-trivial for continuous-state games. Mechanism of death: the decidability proof fails after 3 months of effort, and the team has nothing — no verification tool, no synthesis tool, no paper.

2. **Theorem II is almost certainly false.** The claim that the Pareto set of safe schedules forms a polytope with at most D+1 vertices assumes linearity that doesn't hold. Therapeutic efficacy E_i(σ) — the fraction of time drug i is in its therapeutic window — is a *nonlinear* function of the schedule (piecewise-exponential integrals from compartmental PK dynamics). The Pareto front of nonlinear objectives over box constraints is generically a *curved manifold*, not a polytope. The D+1 vertex bound is a property of multi-objective *linear* programming, not multi-objective nonlinear programming. This is a mathematical error, not a gap. Mechanism of death: a reviewer with multi-objective optimization expertise identifies the error in the first page of review, and the paper's credibility is destroyed.

**Serious Flaws**

1. **The complexity is honestly exponential and the decomposition may not help.** O(D · |grid|^D · 2^p) is exponential in both D and p. For D=5 drugs in one enzyme group with grid=100 and p=5 PK parameters: 5 × 10^10 × 32 ≈ 1.6 × 10^12 operations. This is intractable on a laptop. The enzyme-group decomposition (Theorem III) reduces D within each group, but groups with 5+ drugs sharing CYP3A4 are realistic. And Theorem III's compatibility check (can drug schedules from different enzyme groups be merged?) doesn't address what to do when compatibility fails — it just says "fallback to monolithic synthesis," which is intractable.

2. **Clinical actionability may be illusory.** The promise is beautiful: "Take metformin at 08:00, atorvastatin at 20:00." But clinical scheduling constraints include meal timing (metformin must be taken with food), formulation requirements (extended-release must be taken at bedtime), patient lifestyle (shift workers, travelers), and compliance patterns (patients consolidate medications to reduce pill burden). The "temporal flexibility" extracted from FHIR TimingRepeat is the *formal* flexibility; the *real* flexibility after clinical constraints is much smaller — potentially zero for many drug regimens. A system that synthesizes schedules ignoring these constraints produces academically correct but clinically useless output.

3. **Bounded-memory strategy existence is unproven.** If winning strategies for PTGs require infinite memory (tracking the entire history of continuous state), then synthesized schedules can't be represented as finite objects. The proposal asserts bounded-memory strategies exist for Metzler PTGs but provides no proof or even a proof sketch. This is a second binary dependency stacked on top of the first (decidability).

**Hidden Assumptions**

- *The adversary is static.* Theorem I's adversary chooses PK parameters from a bounded set but doesn't adapt over time. Real PK variability includes time-varying kidney function (disease progression), weight change (affecting volume of distribution), and induced enzyme activity (which *increases* over weeks, not instantaneously). A static adversary misses the most clinically dangerous scenario: a patient whose kidney function degrades over months, causing cumulative drug toxicity.
- *Temporal flexibility in guidelines is substantial.* If most guidelines specify "take with breakfast" or "take at bedtime," the scheduler's optimization space collapses and the game formulation adds complexity without value.
- *Pharmacists want computed schedules.* Pharmacists may prefer to make scheduling decisions using their clinical judgment, informed by interaction alerts, rather than receive computer-generated schedules. The tool assumes pharmacists would trust and follow automated scheduling recommendations — a strong behavioral assumption with no evidence.

**Demand Signal Challenge**

The MTM pharmacist use case is the most plausible demand signal across all three approaches — pharmacists *do* struggle with polypharmacy scheduling, and MTM *is* a billable Medicare Part D service. But "pharmacists struggle with scheduling" → "pharmacists want algorithmically synthesized schedules" is a large leap. Pharmacists are more likely to want decision support (flagging conflicts, suggesting timing separations) than fully automated schedule synthesis. The tool assumes a level of pharmacist trust in automated recommendations that doesn't exist in practice.

**Competitive Threat**

- **Clinical pharmacist expertise.** Pharmacists already solve the scheduling problem using heuristic rules (separate QT-prolonging drugs by 12h, take CYP3A4 substrates 4h before inhibitors). These heuristics handle most cases. The marginal value of formal optimality over "good enough heuristic" is unclear.
- **Drug interaction timing databases.** Tools like Lexi-Interact already provide timing-based separation recommendations for interacting drugs. The formal game-theoretic framing adds rigor but the practical output may be similar.

**Best-Paper Killer**

"The paper claims decidability of pharmacokinetic timed games (Theorem I) but this is acknowledged as a conjecture with no proof. Theorem II (Pareto polytope characterization) is false — the Pareto front of nonlinear PK objectives is not a polytope. The evaluation covers 2–3 hand-encoded guideline pairs. The complexity is exponential in the number of drugs. The clinical actionability of synthesized schedules is not validated. The paper presents ambitious claims without the mathematical or empirical backing to support them. Strong reject."

**Verdict: ABANDON as primary approach; PRESERVE key ideas for synthesis.**

The decidability conjecture is too risky as a primary bet. The false Theorem II is disqualifying in its current form. However, the *prescriptive framing* (synthesize safe schedules, not just flag conflicts) is the single most valuable conceptual contribution across all three approaches and should be incorporated into whichever approach is selected. Specifically: if Approach 1 or 2 finds a conflict, add a "timing recommendation" module that suggests schedule modifications using heuristic optimization (no formal game required) — this captures 80% of Approach 3's value at 10% of its risk.

---

### Mathematician Critic

**Proof Gaps**

1. **Theorem I: three independent non-trivial gaps stacked in sequence.** (a) Adversary extremalization: requires a formal minimax argument showing that for Metzler systems with competitive inhibition, the worst-case adversary strategy over the *continuous* parameter space Φ is achieved at an extremal point. This is plausible by Berge's maximum theorem if the safety predicate is continuous in parameters, but the safety predicate involves threshold crossings (discontinuous). The argument needs careful handling of boundary cases. (b) Scheduler discretization: the grid granularity bound depends on PK time constants and must be formally derived. For stiff systems (mixed fast/slow drugs), the bound may be impractically large. (c) Finite game preservation: the discrete game must preserve the safety property of the continuous game — this requires an ε-approximation argument whose error bound depends on the grid granularity and the ODE discretization.

2. **Theorem II: the polytope claim is mathematically incorrect.** Therapeutic efficacy is a piecewise integral of PK concentration trajectories — a nonlinear function of the schedule. Multi-objective nonlinear programming does not produce polytopic Pareto fronts. The D+1 vertex bound would hold only if efficacy were a *linear* function of schedule parameters, which it manifestly is not (it involves exponentials from compartmental dynamics). This is not a gap — it's an error.

3. **Theorem III: the hard case is punted.** The O(N · log N) compatibility check is trivial (sorting and constraint checking). But when compatibility fails (a drug appearing in multiple enzyme groups needs different timing), the theorem says nothing about resolution. The practically important question — "can incompatible schedules be reconciled, and at what cost to efficacy?" — is unanswered.

**Vacuity Risks**

- Theorem III is at risk of vacuity: if the compatibility check frequently fails for realistic polypharmacy (drugs metabolized by multiple CYP enzymes are common), the decomposition doesn't help in practice and the theorem's conclusion ("O(N · log N) if compatible") applies to an empty set of inputs.

**Over-Claiming**

- Theorem I is called a "Theorem" but is explicitly a conjecture. This is the most egregious overclaim across all nine results.
- Theorem II's complexity bound O(D · |grid|^D · 2^p) is presented as if it's tractable, but it's exponential in D. For D=5, |grid|=100: 10^10 per adversary scenario. The paper would need to acknowledge this is only tractable under decomposition.
- The overall framing — "the first system that resolves polypharmacy conflicts rather than merely reporting them" — overclaims the maturity of results that are largely conjectural.

**Missing Math**

- **Bounded-memory strategy theorem.** Essential for finite representability of synthesized schedules. Without it, the output may not exist as a finite object.
- **Robustness analysis for synthesized schedules.** If the adversary model assumes parameter bounds [φ_lo, φ_hi], how much violation of these bounds can a synthesized schedule tolerate before becoming unsafe? This robustness margin is clinically essential (PK parameters are estimated, not known exactly) and mathematically interesting (robust game theory).
- **Corrected Pareto characterization.** Replace the false polytope claim with a semialgebraic characterization of the Pareto front's dimension and smoothness, or prove the polytope claim under the restricted assumption of linearized steady-state dynamics.

**Depth Verdict**

Expected math depth is **3.5/10** after risk adjustment (ceiling 6/10 if Theorem I is proved, but the math depth assessor estimates high failure probability). The highest *ambition* but the worst *reliability*. One potentially significant result (Theorem I, *if* it survives), one likely-false result (Theorem II — this is disqualifying without correction), and one modest instantiation (Theorem III). **Strengthening path:** downgrade Theorem I to a conjecture and prove it for a restricted subclass; retract Theorem II and replace with a correct characterization; add the robustness theorem.

---

## Cross-Approach Comparison

### Which approach survives the hardest scrutiny?

**Approach 2** survives the hardest scrutiny — not because it's the best, but because it has the fewest ways to *completely fail*. The widening operator guarantees termination. The abstract domain may be imprecise, but imprecision is a gradient (degraded results), not a cliff (no results). Every theoretical obligation is a known-technique instantiation that will succeed — the question is whether the *precision* is useful, which is an empirical question with partial answers. Approach 1 has the E1 gamble and the CEGAR convergence uncertainty. Approach 3 has a binary dependency on an unproven decidability conjecture. Approach 2 has no binary dependencies.

### Which approach has the best failure mode?

**Approach 1.** Even if everything goes wrong — E1 disappoints, CEGAR doesn't converge well, the corpus is tiny — you still have: (a) the PTA formalism (novel, publishable as a theory paper at HSCC), (b) Theorem 3 on contract composition (the strongest individual theorem, publishable alone), (c) a proof-of-concept that formal guideline verification is possible (existence proof, publishable at AMIA as a vision paper), (d) the zonotopic reachability work for Metzler systems (publishable at HSCC/ADHS). The depth check estimates the salvage floor at 5 potential publications with no zero-paper failure mode. This is the best downside protection.

Approach 2's failure mode (everything flagged as "possibly unsafe") produces one paper about the abstract domain design — modest but publishable. Approach 3's failure mode (decidability proof fails) produces nothing — the game formulation without decidability is a research proposal, not a paper.

### What elements from each approach should be preserved in a synthesis?

1. **From Approach 1:** Theorem 3 (contract composition via CYP-enzyme interfaces). This is the single deepest mathematical contribution across all approaches and is practically essential for scalability. Also: the CEGAR-with-clinical-abstractions idea and the counterexample generation pipeline.
2. **From Approach 2:** The PK-aware widening operator (Theorem B, with corrected convergence bound). This is a legitimate algorithmic contribution and could serve as a fast screening layer. Also: the speed story and the clinical interpretability of concentration-interval output.
3. **From Approach 3:** The *prescriptive framing* — the idea that the tool should suggest schedule modifications, not just flag conflicts. This can be implemented as a heuristic optimization post-processing step in Approach 1 or 2, capturing 80% of Approach 3's clinical value without the decidability risk. Also: the adversary model (worst-case PK parameter reasoning) which strengthens robustness claims in any approach.

### Single strongest element across all 3 approaches?

**Theorem 3 from Approach 1 (contract-based composition via CYP-enzyme interfaces).** It is the only result that is simultaneously: novel (first A/G framework for metabolic pathway interfaces), load-bearing (the system is useless for realistic polypharmacy without it), correctly scoped (honest about competitive-inhibition restriction), practically motivated (maps directly to real pharmacology), and provably correct (the monotonicity argument is convincing). No other result across all nine theorems/propositions meets all five criteria.

### Single weakest element across all 3 approaches?

**Theorem II from Approach 3 (Pareto polytope characterization).** It is *likely mathematically false*. The polytope/D+1 vertex claim assumes linearity that PK dynamics don't satisfy. It is the only result across all three approaches where the math depth assessor identifies it as *incorrect*, not merely thin or incomplete. Including it in a submission would be disqualifying — a single false theorem destroys the credibility of the entire paper.

---

## Challenge Messages

### Challenge to the Domain Visionary

You built a value proposition around stakeholders who don't know they need you. Your table lists hospital CDS committees, EHR vendors, guideline developers, regulators, and patients — but your evidence for demand is entirely *structural* ("the gap exists, therefore someone must want to fill it"). That's the reasoning of every failed startup. The Joint Commission cites CDS configuration errors — but recommends better human processes, not formal verification. ONC mandates FHIR interoperability — but says nothing about formal safety verification. Epic already ships interaction checking — and has expressed zero interest in PTA-based model checking.

Your CQL ecosystem assessment is especially troubling. You quote "~30–50 CQL treatment-decision guidelines" as if that's a usable corpus, but most are research prototypes from CDS Connect that have never been deployed in production. The real count of *production-deployed* CQL treatment-logic artifacts is closer to 5–10 across all US health systems. You're building a verification engine for a format that barely exists in production.

Here's my challenge: **Before any implementation work begins, can you produce one piece of evidence that a real stakeholder wants this?** Not "would find it interesting" — *wants* it enough to allocate time to evaluate it. A 15-minute call with an Epic CDS architect, a CDS Connect curator, or a Joint Commission surveyor would either validate or invalidate the demand hypothesis. If you can't find a single person who says "yes, I would use this," the project's value proposition is an elegant fantasy.

### Challenge to the Math Depth Assessor

Your assessment is careful and largely correct, but I think you're too generous on Approach 2 and too harsh on the *cross-approach synthesis opportunity*.

You rate Approach 2's math at 3.5/10 and call it "shallowest but most honest." I'd argue that honesty in presentation doesn't compensate for shallowness in content. Theorem A is a Galois connection that anyone in the field would derive in 20 minutes. Theorem C is a reduced product that's standard since 1979. Theorem B's convergence bound is probably wrong for coupled drugs. Your generous 3.5/10 includes credit for "not overclaiming," but a best-paper committee doesn't award points for correctly labeling routine results as routine.

More critically: you identified the single most important missing mathematical contribution across all approaches — **a formal CQL compilation correctness theorem** — and then moved on. This is worth much more attention. You estimated it at 6–7/10 proof difficulty, called it "the deepest mathematical contribution" that "none of the approaches proposes," and noted it would be "a stronger contribution than Proposition 1 and Theorem 2 combined." So why isn't this the central recommendation? If this theorem is the most novel and load-bearing math available, shouldn't the project be reorganized around *proving it*? The uncomfortable truth you articulated — that the mathematical value is in *domain synthesis*, not individual theorems — implies the biggest mathematical wins come from formalizing the *interface between domains* (CQL ↔ PTA, or more generally, clinical informatics ↔ formal methods). The CQL compilation correctness theorem is exactly that interface. Push harder on this.

### Challenge to the Difficulty Assessor

Your LoC estimates and feasibility assessments are rigorous, but I think you're underweighting the *integration nightmare* for Approach 1 and overweighting the *precision risk* for Approach 2.

For Approach 1, you identify six critical integration points (CQL→PTA→PK→Region Graph→Model Checker; ODE solver↔reachability↔CEGAR; terminology resolution). Each interface is a potential semantic gap. In my experience with multi-domain formal verification tools, integration bugs — not algorithmic bugs — are the dominant failure mode. The PTA must faithfully represent CQL semantics *and* be amenable to PK region construction *and* be checkable by the MTL model checker. Any mismatch produces bugs invisible until the final evaluation. Your minimum viable artifact estimate of 25–35K LoC achieves this by cutting everything down to manually-encoded examples with 1-compartment PK, but even at this reduced scope, the CQL→PTA→model checker pipeline has three critical interfaces that each took 2–4 person-months in comparable systems (CompCert frontend, SLAM/SDV abstraction layer, CBMC GOTO compilation). Your 12–16 week timeline for 2 engineers gives ~8 person-months total — which needs to cover *both* the math and all three interfaces. This is extremely tight and likely requires 20+ weeks in reality.

For Approach 2, you rate false-positive risk as the "single biggest risk" and suggest it "could be 2–3× harder than estimated if the interval domain proves insufficient." But you also note that "precision is a gradient, not a cliff — partial success is always possible." These two claims are in tension. If precision is a gradient, then the false-positive rate is a tunable parameter (adjustable via narrowing iterations, domain refinement, CYP3A4 relational sub-domain). If it's 2–3× harder than estimated, it's more cliff-like. **Which is it?** My read: for non-CYP3A4 drugs, the interval domain will be precise enough (precision is a gradient). For CYP3A4-sharing drugs, the interval domain will fail and you'll need the polyhedra sub-domain (which is a cliff in implementation complexity). The true difficulty assessment should separate these two cases rather than blending them.

---

*Debate produced by the Adversarial Skeptic and Mathematician Critic personas operating as a stress-test team. The goal is a bulletproof final approach, not project termination. Every attack above should be treated as a reviewer's objection that must be addressed before submission.*
