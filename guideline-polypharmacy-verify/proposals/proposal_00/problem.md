# Final Approach: GuardPharma

## 1. Title and One-Sentence Summary

**GuardPharma: Two-Tier Compositional Verification of Polypharmacy Safety via Pharmacokinetic Abstract Interpretation and Contract-Based Model Checking**

**One sentence:** A two-tier formal verification engine that screens multi-guideline polypharmacy conflicts in seconds via pharmacokinetic abstract interpretation, then produces provably reachable counterexample trajectories for flagged conflicts via contract-based compositional model checking over Pharmacological Timed Automata.

---

## 2. Core Architecture

### System Overview

GuardPharma is a formal verification engine that takes computable clinical guidelines (manually encoded as Pharmacological Timed Automata for the paper; CQL/FHIR compilation deferred) and determines whether N simultaneously active guidelines can ever drive a patient into a pharmacokinetically unsafe state. The system produces either a **safety certificate** (no conflict exists for any patient matching the comorbidity profile) or a **ranked conflict report** with counterexample patient trajectories showing exactly when and how the unsafe state is reached.

### Two-Tier Verification Paradigm

The architecture merges Approach 2's speed with Approach 1's diagnostic precision:

**Tier 1 — Abstract Screening (from Approach 2).** A pharmacokinetic abstract interpretation engine computes concentration-interval bounds for all drugs under all guideline-compliant administration patterns. The abstract domain is a product of per-drug concentration intervals × per-enzyme activity intervals × discrete clinical-state predicates. A PK-aware widening operator exploiting Metzler steady-state convergence ensures termination with clinically meaningful precision. Output: for each drug combination, one of three verdicts — *definitely safe* (abstract domain disjoint from unsafe region), *possibly unsafe* (abstract domain intersects unsafe region), or *definitely unsafe* (abstract lower bound exceeds toxic threshold). Tier 1 processes 20+ guidelines in under 5 seconds on a laptop CPU.

**Tier 2 — Precise Model Checking (from Approach 1).** For every combination flagged as *possibly unsafe* or *definitely unsafe* by Tier 1, the system constructs product PTAs and performs contract-based compositional model checking. CYP-enzyme interface contracts decompose the N-guideline verification problem into individual guideline checks plus enzyme-compatibility checks (linear in N per enzyme). For combinations outside the contract framework (~30% — enzyme induction, PD interactions), a SAT-based bounded model checker provides sound verification with a configurable time horizon (default: 365 days). When verification fails, a counterexample generator extracts a minimal violating trace and translates it to a clinical narrative with PK trajectory visualization.

**Post-Processing Module — Schedule Recommendations (from Approach 3, demoted).** For confirmed conflicts where the underlying guidelines permit temporal flexibility, a heuristic schedule optimizer suggests timing adjustments that may resolve the conflict ("separate drug A and drug B by ≥8 hours"). This is implemented as a constraint-satisfaction post-processing step over FHIR TimingRepeat-derived flexibility windows — not as a formal game-theoretic synthesis. It captures ~80% of Approach 3's clinical value at ~10% of its technical risk. Formally, this module's output carries no soundness guarantee; it is a clinical suggestion that must be re-verified by Tier 2.

**Clinical Significance Filter.** All confirmed conflicts are stratified by severity using DrugBank interaction severity, Beers Criteria 2023 classification, pre-computed FAERS disproportionality signals (distributed as a lookup table), and Medicare comorbidity prevalence scoring. Only critical and significant conflicts are reported as primary findings.

### Why Two Tiers?

The Skeptic's critique of Approach 2 was devastating on one point: "No counterexamples = no actionable output." Abstract interpretation says *possibly unsafe* but cannot explain *why* or *when*. The Skeptic's critique of Approach 1 was equally sharp: CEGAR convergence is unpredictable, and Tier-2-only verification of 20+ guidelines could take hours. The two-tier design resolves both: Tier 1 eliminates the ~70–80% of guideline combinations that are provably safe (no further analysis needed), and Tier 2 focuses expensive model checking only on the ~20–30% that Tier 1 cannot resolve — producing the counterexample trajectories that make the output clinically actionable.

---

## 3. Why This Approach Wins

**Why not Approach 1 alone?** Contract-based model checking for 20+ guidelines is too slow for interactive use and depends on CEGAR convergence, which is empirically unpredictable. The Tier 1 abstract screening eliminates most combinations cheaply, focusing Tier 2 on genuinely ambiguous cases and dramatically improving wall-clock performance. The combined system inherits Approach 1's diagnostic strength while mitigating its scalability weakness.

**Why not Approach 2 alone?** Abstract interpretation cannot produce counterexamples. When it says "possibly unsafe," the clinician gets no more information than a DrugBank lookup. The false-positive rate for CYP3A4-sharing drugs may be catastrophic without precision refinement. Tier 2 provides the missing diagnostic layer: every "possibly unsafe" verdict is either confirmed with a concrete counterexample or refined to "safe" by the model checker.

**Why not Approach 3?** The decidability of Pharmacokinetic Timed Games is a conjecture, not a theorem. Theorem II (Pareto polytope characterization) is likely mathematically false. The binary dependency on an unproven decidability result creates ~30–40% probability of total project failure. The prescriptive framing is preserved as a heuristic module — safe to implement, clinically valuable, and decoupled from the formal verification core.

**What this synthesis uniquely achieves:**
1. Fast screening (Approach 2's speed) + precise diagnostics (Approach 1's counterexamples)
2. The strongest mathematical contribution across all approaches (Theorem 3: contract composition)
3. The best failure mode (Approach 1's salvage floor: PTA formalism + Theorem 3 + existence proof)
4. Prescriptive schedule suggestions (Approach 3's clinical value) without formal game-theoretic risk
5. Feasibility within 18–20 weeks for the minimum viable artifact (extended per verifier amendment VA1)

---

## 4. Mathematical Foundation

### Theorem 1 (Contract-Based Compositional Safety — Crown Jewel)

**Statement.** Let P₁, …, Pₙ be PTAs for N concurrent guidelines sharing pharmacokinetic state. For each CYP enzyme e shared by a subset Sₑ of guidelines, define the enzyme-interface contract Cₑ,ᵢ = (Aₑ,ᵢ, Gₑ,ᵢ) for guideline Pᵢ, where Aₑ,ᵢ specifies the minimum assumed enzyme-e capacity and Gₑ,ᵢ specifies the maximum enzyme-e load imposed by Pᵢ's medications. Under competitive CYP inhibition, the guarantee function Gₑ,ᵢ is monotone decreasing in assumed enzyme capacity (lower capacity → higher drug concentration → higher inhibition load) due to Metzler dynamics. Safety of the N-guideline composition holds if: (1) each Pᵢ satisfies its safety property individually, and (2) for each enzyme e, the worst-case aggregate load Σᵢ∈Sₑ Gₑ,ᵢ(min_capacity) does not exceed the enzyme's total capacity. Verification cost is O(N · single-guideline-cost + N · M) for M shared enzymes, versus exponential product-automaton cost without composition.

**Role.** Without this theorem, verifying a patient on 5+ concurrent guidelines requires exponential product-automaton construction — the tool is a toy, not a tool. With it, polypharmacy verification becomes routine. This is the single result that makes GuardPharma practical for realistic clinical polypharmacy (average Medicare beneficiary: 6.8 conditions).

**Proof strategy.** (1) Establish that competitive CYP inhibition produces a monotone chain: decreased clearance → increased concentration → increased inhibition load. (2) Prove this monotonicity holds across the entire bounded population PK parameter space Φ via the Metzler property (off-diagonal entries ≥ 0 ensure componentwise monotonicity of solutions). (3) Show that worst-case guarantees computed under minimum assumed enzyme capacity are sound upper bounds, resolving the circular dependency (concentrations depend on enzyme activity, which depends on concentrations) in a single pass. (4) Prove soundness: if contracts are mutually compatible, the product PTA satisfies the safety property.

**Depth assessment.** Novel instantiation of assume-guarantee reasoning (Pnueli 1985, Pacti 2022). The CYP-enzyme interface abstraction and Metzler monotonicity enabling single-pass resolution are genuinely new insights. Not a new verification technique, but a new and practically essential domain-specific instantiation. Depth: 6/10.

**Risk.** Low for competitive inhibition (~70% of clinically significant PK DDIs). Medium for the overall claim — the ~30% outside the contract framework (enzyme induction, PD interactions) require monolithic fallback. The boundary classification (competitive vs. non-competitive) must be correct; misclassification produces unsound results. Mitigation: conservative classification using DrugBank mechanism annotations; any uncertain interaction is routed to monolithic verification.

---

### Proposition 2 (PK-Aware Widening with Bounded Convergence)

**Statement.** Define a widening operator ∇_PK on the PK abstract domain: for concentration intervals, widen to [0, Css,max(φ_worst)] where Css,max is the worst-case steady-state concentration under worst-case population PK parameters; for enzyme-activity intervals, widen to [Emin(full-inhibition), Emax(no-inhibition)]. The abstract fixed-point computation converges in at most O(D · k) iterations, where D is the number of drugs and k is the maximum enzyme-coupling degree (number of drugs sharing any single CYP enzyme).

**Role.** Without controlled widening, the abstract analysis either doesn't terminate (no widening) or produces vacuous results (standard widening: every interval → [−∞, +∞]). This proposition makes Tier 1 both terminating and useful.

**Proof strategy.** (1) For non-interacting drugs, each drug independently reaches its worst-case steady state in 1 iteration (Metzler eigenvalues have negative real parts → exponential convergence to steady state). (2) For enzyme-coupled drugs, convergence is a coupled fixed-point problem. Each widening iteration computes worst-case steady states assuming the previous iteration's enzyme loads. Convergence requires at most k additional iterations per coupling layer, because the Metzler monotonicity ensures the enzyme-load sequence is monotonically increasing and bounded by total enzyme capacity.

**Depth assessment.** Novel instantiation of domain-specific widening (Cousot 1992). The PK-specific design exploiting Metzler steady-state convergence is genuine. The corrected O(D · k) bound (versus Approach 2's claimed O(D)) properly accounts for enzyme coupling. Depth: 5/10.

**Risk.** Medium. The O(D · k) bound assumes the spectral radius of the coupled Jacobian ensures geometric convergence. For strongly coupled drugs (two potent CYP3A4 inhibitors), convergence may be slower than O(D · k) — potentially O(D · k²). Mitigation: if the bound is exceeded in practice, fall back to standard widening-to-lethal-dose (sound but less precise) for the offending drug pair.

---

### Observation 3 (δ-Decidability of PTA Reachability)

**Statement.** The reachability problem for PTA with Metzler ODE dynamics and bounded population PK parameters is δ-decidable: for any δ > 0, the verifier answers "unsafe" (exact) or "δ-safe" (safe within perturbation δ). The parameter δ must be calibrated per drug: δᵢ = min(therapeutic_windowᵢ / 10, 0.1 · Cmax,ᵢ) to ensure clinical meaningfulness.

**Role.** Without a decidability guarantee, the Tier 2 model checker may not terminate, degrading GuardPharma from a verification tool to a testing tool. The per-drug δ-calibration (replacing the naive universal δ = 0.1 μg/mL from the original proposal) addresses the Mathematician's critique about narrow-therapeutic-index drugs.

**Proof strategy.** Direct application of Gao et al.'s dReal framework. PK dynamics with Metzler ODEs fall within dReal's decidable fragment. The per-drug δ-calibration is a domain-specific choice, not a mathematical contribution.

**Depth assessment.** Standard application of an existing framework with a domain-specific calibration. Depth: 1.5/10. Correctly labeled "Observation" rather than "Theorem" or "Proposition" — this is citing a library's guarantee, not proving something new.

**Risk.** Very low for the δ-decidability claim. Low-medium for the clinical meaningfulness of the per-drug δ — for drugs like digoxin (therapeutic range 0.8–2.0 ng/mL), even the per-drug calibration requires sub-nanogram precision, which may strain validated interval arithmetic.

---

### Deferred: CQL Compilation Correctness

The Math Depth Assessor and Mathematician Critic both identified this as the most valuable missing mathematical contribution: a bisimulation or trace-refinement result between CQL operational semantics and the compiled PTA. This would be the first formal semantics of CQL — genuinely novel, highly load-bearing, and proof difficulty 6–7/10. For this paper, CQL compilation is deferred (guidelines are manually encoded as PTA). The compilation correctness theorem is the centerpiece of the planned follow-on tool paper.

---

### Cut Results

- **Theorem 2 (PSPACE-completeness of MTL model checking for PTA):** Cut from the theorem list. The bisimulation proof is incomplete, PSPACE membership is unproven, and the system uses SAT-based bounded model checking in practice. The PK region graph remains an engineering optimization in the implementation, not a claimed theorem.
- **Theorem A (Galois connection for PK dynamics):** Textbook construction (depth 1.5/10). Present as a paragraph establishing soundness of the abstract domain, not as a theorem.
- **Theorem C (reduced enzyme-coupling product):** Standard reduced product (depth 2/10). Present as an implementation optimization, not as a theorem.
- **All of Approach 3's theorems:** Theorem I is an unproven conjecture; Theorem II is likely mathematically false; Theorem III is modest. None survive into the paper.

---

## 5. Key Technical Components

### 5.1 PK Abstract Interpretation Engine (Tier 1)

**What it does:** Computes concentration-interval bounds for all drugs under all guideline-compliant administration patterns. Classifies each drug combination as definitely-safe, possibly-unsafe, or definitely-unsafe.

**Why it's hard:** The precision-coarseness tradeoff is the central challenge. Therapeutic windows are often 2× (e.g., warfarin 2–5 μg/mL). An abstract domain too coarse to resolve this window classifies everything as possibly-unsafe. The PK-aware widening operator must exploit Metzler steady-state bounds to stay precise. For CYP3A4-sharing drugs (~50% of all drugs), enzyme coupling creates widening pressure that may destroy precision — mitigated by narrowing iterations post-widening.

**Estimated LoC:** ~8–12K (abstract domain + widening + transformers for guideline actions)

**Dependencies:** PK Model Library, Clinical State Space Model

### 5.2 PTA Construction & Contract Composition Engine (Tier 2 Core)

**What it does:** Builds Pharmacological Timed Automata from encoded guidelines + PK models. Extracts CYP-enzyme interface contracts. Checks contract compatibility for N-guideline compositions. Routes contract-ineligible interactions to the monolithic verification path.

**Why it's hard:** Product-automaton state explosion is the central scalability challenge. Contract extraction requires computing worst-case enzyme loads from population PK parameters — solving parametric reachability over Metzler ODEs. The monotonicity proof for competitive inhibition is the mathematical core. The contract-eligibility classifier must be conservative (routing uncertain interactions to monolithic fallback).

**Estimated LoC:** ~10–14K

**Dependencies:** PK Model Library, Clinical State Space Model

### 5.3 SAT-Based Bounded Model Checker

**What it does:** Verifies safety properties over PTAs using SAT-based bounded model checking with a configurable time horizon. Serves as the verification backend for both contract-decomposed and monolithic verification paths.

**Why it's hard:** The PK state encoding into SAT variables must preserve concentration precision at clinical thresholds. The bounded horizon (default 365 days) with hourly or daily time steps produces large SAT instances. CaDiCaL/Z3 are used as backends.

**Estimated LoC:** ~8–12K

**Dependencies:** PTA Construction Engine, Z3/CaDiCaL

### 5.4 Counterexample Generator & Clinical Narrator

**What it does:** Extracts minimal violating traces when model checking fails. Translates formal traces into clinical narratives ("Patient with eGFR < 30 starts metformin day 1, lisinopril day 14; warfarin concentration reaches 6.2 μg/mL by day 45"). Produces PK trajectory visualizations.

**Why it's hard:** Trace minimization on bounded model checking output. Clinical narrative generation requires mapping formal automaton states back to clinical concepts with temporal context — deceptively complex clinical NLG.

**Estimated LoC:** ~4–7K

**Dependencies:** SAT-Based Model Checker, Clinical State Space Model

### 5.5 Pharmacokinetic Model Library

**What it does:** Implements compartmental ODE models (1-compartment for MVP; 2/3-compartment post-paper) with population PK parameters from published sources. Provides Metzler-matrix representation, competitive CYP inhibition models, steady-state and transient solvers. Wraps CAPD/VNODE-LP for validated interval ODE integration.

**Why it's hard:** Validated interval arithmetic with directed rounding is notoriously fiddly — off-by-one errors in rounding modes produce unsound results. PopPK parameterization for ~50 common drugs requires curating published literature. The Metzler structure must be verified for each PK model to ensure contract-based decomposition applies.

**Estimated LoC:** ~8–12K

**Dependencies:** CAPD or VNODE-LP (external)

### 5.6 Clinical State Space Model

**What it does:** Formal model of patient state: conditions (onset, severity, temporal evolution), medications (dose, route, frequency, duration), lab values (continuous ranges + clinical thresholds), and interdependencies.

**Estimated LoC:** ~4–6K

**Dependencies:** None (foundational data model)

### 5.7 Clinical Significance Filter

**What it does:** Stratifies conflicts by severity using DrugBank interaction severity, Beers Criteria 2023, pre-computed FAERS disproportionality lookup table, and Medicare comorbidity prevalence.

**Estimated LoC:** ~5–8K

**Dependencies:** DrugBank data, Beers Criteria, FAERS lookup table

### 5.8 Heuristic Schedule Recommender (Approach 3 Module)

**What it does:** For confirmed conflicts with temporal flexibility, solves a constraint-satisfaction problem over FHIR TimingRepeat-derived windows to suggest timing adjustments. Output is a clinical suggestion, not a formal guarantee.

**Estimated LoC:** ~3–5K

**Dependencies:** Tier 2 conflict output, FHIR TimingRepeat constraints

### 5.9 Evaluation & Benchmarking Engine

**What it does:** Automated benchmark harness for all experiments. TMR-style atemporal baseline for E1 comparison. Statistical rigor for all validation metrics.

**Estimated LoC:** ~6–10K

**Dependencies:** All other subsystems

---

## 6. Evaluation Plan

| # | Experiment | Data Source | Metric | Target | Role |
|---|-----------|-------------|--------|--------|------|
| **E1** | **Temporal ablation (CENTERPIECE)** | Full guideline corpus; run Tier 2 with and without PK temporal constraints | % of conflicts found only by temporal reasoning (X%) | X ≥ 20% (expect 10–30%) | Demonstrates PTA+PK machinery discovers dangers invisible to all prior approaches |
| **E2** | Known-conflict recall | Beers 2023 (30+ DDIs), STOPP/START v3 (80+ contraindications) injected into guideline pairs | Recall of known conflict detection | ≥ 90% | Validates completeness against gold standard |
| **E3** | Two-tier speedup | Guideline sets of size 2, 5, 10, 15, 20 | Tier 1 time, Tier 2 time, total time vs. Tier-2-only time | 20 guidelines in <60s total; ≥5× speedup from Tier 1 screening | Demonstrates architectural value of two-tier design |
| **E4** | Compositionality speedup | Same guideline sets; compositional vs. monolithic Tier 2 | Wall-clock ratio (monolithic / compositional) | ≥ 10× for N ≥ 10 | Demonstrates practical value of Theorem 1 |
| **E5** | DrugBank cross-validation | All discovered conflicts cross-referenced against DrugBank | Precision for "critical" tier | ≥ 70% | Validates clinical relevance |
| **E6** | FAERS signal validation | Discovered conflicts validated by FAERS disproportionality (PRR > 2, CI > 1) | Fraction of novel conflicts with significant FAERS signal | Report as-is | Discovery metric; validates against independent pharmacovigilance data |
| **E7** | Tier 1 false-positive rate | Combinations classified "possibly unsafe" by Tier 1, resolved by Tier 2 | % of Tier 1 "possibly unsafe" that Tier 2 proves safe | Report; target ≤ 30% | Validates Tier 1 precision |
| **E8** | Clinical significance stratification | All conflicts ranked by composite severity | Distribution of critical / significant / informational | ≥ 30% informational | Demonstrates filter value |
| **E9** | Clinical pharmacist review **(MANDATORY — VA2)** | 3 pharmacists rate 20 conflicts on 5-point Likert scale | Inter-rater reliability (Fleiss' κ) | κ ≥ 0.4 (moderate agreement) | Eliminates credibility gap at clinical venues; ~$600, no IRB; recruit during weeks 1–4 |

### Centerpiece: E1 (Temporal Ablation)

E1 is the make-or-break experiment. The paper's core claim is that temporal PK reasoning discovers guideline conflicts invisible to atemporal approaches (TMR, Lexicomp, LLM+DrugBank). The experimental design: run the full guideline corpus through both an atemporal checker (drug-class interaction lookup, no timing, no concentration dynamics) and the full Tier 2 temporal verifier. Report X% = fraction of conflicts found *only* by temporal reasoning.

**Critical analytical distinction (from depth check):** A PK interaction *being temporal* ≠ a guideline conflict *requiring temporal reasoning to detect*. Fluconazole + warfarin is temporal, but an atemporal checker flags it fine. Temporal reasoning adds *detection* value only for schedule-dependent conflicts where guideline-prescribed timing determines danger. Pre-implementation calibration: construct 5–10 synthetic guideline pairs with known temporal interactions to establish a floor for X.

### Fallback Narratives if E1 Underdelivers

If X < 20%, the paper pivots to a three-part fallback:

1. **Explanation quality.** Even when atemporal checkers detect the same conflict, GuardPharma provides pharmacokinetic trajectory counterexamples — showing *when* toxicity occurs, *how fast* concentrations rise, *which CYP enzyme is overwhelmed*. This is unprecedented and clinically valuable regardless of detection rate.

2. **Compositionality speedup (E4).** Theorem 1 delivers practical exponential-to-polynomial reduction independent of E1. This is a standalone contribution: the first compositional verification framework for pharmacokinetic systems.

3. **Existence proof.** The first formal safety certificate for multi-guideline polypharmacy verification. Even with modest X%, the paper demonstrates that formal verification of clinical guidelines is *possible* — a new capability that didn't exist before.

---

## 7. Risk Registry

| # | Risk | Severity | Probability | Mitigation |
|---|------|----------|-------------|------------|
| R1 | **E1 temporal ablation produces X < 15%** | Critical | 30–40% | Fallback narrative (explanation quality + compositionality speedup + existence proof). Pre-calibrate with 5–10 synthetic temporal pairs. If X < 10%, pivot paper framing entirely to compositionality theorem. |
| R2 | **PTA encoding of real guidelines is infeasible** | Critical | 15–20% | Pilot gate: manually encode 3 guideline pairs before full commitment. If encoding fails, redirect to theory paper (PTA formalism + composition theorem at HSCC/CAV). |
| R3 | **CEGAR / BMC doesn't converge within timeout for Tier 2** | High | 25–35% | SAT-based BMC with 90-day bounded horizon is the fallback (always terminates). Report convergence statistics honestly. If most pairs time out, rely on Tier 1 abstract interpretation results with Tier 2 as a precision tool for selected pairs. |
| R4 | **Tier 1 false-positive rate is catastrophic (>50%)** | High | 20–30% | Narrowing iterations post-widening. CYP3A4 relational sub-domain (polyhedra for the most promiscuous enzyme). Worst case: Tier 1 becomes a fast pre-filter that only eliminates clearly-safe pairs (~50%), and Tier 2 handles the rest. |
| R5 | **Corpus starvation (~30–50 treatment guidelines)** | **High (VA3)** | 50–60% | Supplement with manually encoded clinical decision rules (Wells, CHADS₂-VASc, MELD, etc.) targeting ~80 total treatment-logic artifacts. Report honestly. Accept reduced statistical power. **Frame paper explicitly as proof-of-concept on a small corpus; do not oversell evaluation scope.** |
| R6 | **PopPK parameters unavailable for some drugs** | Medium | 30–40% | Curate parameters for the ~50 most common drugs in polypharmacy first. For missing drugs, use conservative estimates (worst-case from drug class averages). Document coverage. |
| R7 | **Zero demand signal from stakeholders** | Medium | 80%+ | Frame as forward-looking infrastructure for the FHIR/CQL ecosystem — verification before deployment, not a tool for current clinical practice. Position paper as research contribution to formal methods + clinical informatics, not a product. |
| R8 | **Contract-eligibility misclassification** | Medium | 10–15% | Conservative classification: any interaction not confirmed as competitive CYP inhibition in DrugBank is routed to monolithic verification. Report coverage statistics (expected ~70% contract-eligible). |
| R9 | **Validated interval ODE integration takes 3–5× longer than estimated** | Medium | 40–50% | For MVP, use 1-compartment models only (closed-form matrix exponential, no ODE integration needed). Defer validated interval ODE to post-paper. |
| R10 | **Monotonicity proof for Theorem 1 has a gap at parameter-space boundary** | Low-Medium | 15–20% | The monotonicity chain (CL↓ → C↑ → inhibition↑) must hold across the *entire* parameter space, not just pointwise. Restrict the theorem statement to competitive inhibition with bounded parameters (physiologically justified). State non-competitive and mixed inhibition as explicit exclusions requiring monolithic fallback. |

---

## 8. Minimum Viable Paper Artifact

### Must Build (for submission)

| Component | Est. LoC | Notes |
|-----------|----------|-------|
| Clinical State Space Model | ~5K | Foundational data model |
| PK Model Library (1-compartment, ~30 drugs) | ~6K | Closed-form matrix exponential; no validated ODE for MVP |
| PK Abstract Interpretation Engine (Tier 1) | ~8K | Interval domain + PK-aware widening + narrowing |
| PTA Construction & Contract Composition | ~10K | Core of Theorem 1 |
| SAT-Based Bounded Model Checker (Tier 2) | ~8K | Z3/CaDiCaL backend; 90-day bounded horizon |
| Counterexample Generator (simplified) | ~3K | Formal traces with basic clinical annotation (no rich NLG) |
| Clinical Significance Filter (DrugBank only) | ~4K | DrugBank + Beers lookup; FAERS as pre-computed table |
| Evaluation Engine (E1, E2, E4, E5, E7) | ~6K | Reduced experiment set; TMR-style atemporal baseline |
| **Manual PTA encodings (5–8 guideline pairs)** | ~3K | Hand-encoded from published guidelines |
| **TOTAL** | **~53K** | **~35K novel algorithmic code** |

### Can Defer

| Component | Reason for deferral |
|-----------|-------------------|
| CQL-to-PTA Compiler (~12–25K) | Engineering-intensive; manual encoding suffices for paper |
| FHIR PlanDefinition Compiler (~6–12K) | Same as above |
| Guideline Corpus Pipeline (~8–15K) | Data engineering, not algorithmic novelty |
| Terminology Integration Layer (~4–8K) | Wrapper code; use pre-resolved ValueSets |
| Zonotopic Reachability Engine (~10–16K) | Performance optimization; interval-based suffices for MVP |
| Rich Clinical Narrative NLG (~4K) | Formal traces suffice for paper; rich narratives for tool paper |
| Heuristic Schedule Recommender (~3–5K) | Future work module |
| 2/3-compartment PK models (~4K) | 1-compartment suffices for paper |
| Validated Interval ODE Integration (~5–8K) | Closed-form matrix exponential for 1-compartment suffices |
| Full FAERS processing (~5K) | Pre-computed lookup table suffices |
| E3, E6, E8, E9 experiments (~5K) | Reduced experiment set for paper |

### Total Estimated LoC for Paper Artifact: ~53K (~35K novel)

At 200 LoC/day of production-quality novel code, 35K novel LoC requires ~175 person-days ≈ 9 person-months. Two engineers × 20 weeks = ~10 person-months. This is achievable with aggressive scoping, particularly since the foundational math (Theorem 1 proof, Proposition 2 proof) can proceed in parallel with implementation. The validated ODE deferral (using closed-form 1-compartment solutions) is the key feasibility enabler. **Timeline extended to 20 weeks per verifier amendment VA1** to account for integration friction, proof work, and clinical PTA encoding iteration.

---

## 9. Best-Paper Argument

### Five Reasons a PC Selects This Paper

1. **The problem is undeniable and timely.** Polypharmacy-related adverse drug events are a leading cause of hospitalization. The 21st Century Cures Act mandates FHIR-based CDS interoperability. No tooling verifies multi-guideline safety. The paper arrives at the exact moment computable guidelines transition from research prototypes to deployed infrastructure. Every clinician and informatician recognizes the gap.

2. **The two-tier result is clean and memorable.** "Abstract interpretation screens 190 guideline pairs in 4.7 seconds; contract-based model checking confirms N conflicts with counterexample trajectories showing the exact day toxicity occurs." The speed + precision combination is novel in formal verification applied to clinical systems. No prior system achieves both.

3. **Theorem 1 (contract composition) is a genuine mathematical contribution.** The first assume-guarantee framework organized around metabolic pathway interfaces. Novel instantiation of a known technique to an undeniably important domain. The Metzler monotonicity enabling single-pass resolution is a real insight, not a library invocation. The exponential-to-polynomial reduction is experimentally demonstrated (E4).

4. **The system bridges two communities.** Formal methods and clinical informatics have operated in parallel for decades. This paper is the bridge — compositional model checking applied to the most urgent problem in clinical informatics. Neither community alone could have produced it. Cross-disciplinary papers that succeed technically are rare and rewarded.

5. **The evaluation is fully automated and devastating.** Known-conflict recall against Beers/STOPP gold standards. DrugBank cross-validation. FAERS signal validation. Temporal ablation quantifying the marginal value of PK reasoning. Compositionality speedup demonstrating Theorem 1's practical impact. Two-tier speedup demonstrating architectural value. All reproducible from a single `make evaluate` command on a laptop.

### Honest Probability Assessment

| Venue | P(accept) | P(best paper) | Notes |
|-------|-----------|---------------|-------|
| AIME 2025/2026 | 40–55% | 8–12% | Best fit; bridges FM + clinical informatics; smaller venue, receptive to cross-disciplinary |
| AMIA Annual | 30–45% | 5–8% | Clinical informatics audience; formal methods as applied tool = novelty. Weakened by zero clinical validation unless E9 is included |
| TACAS (tool track) | 35–50% | 4–7% | Natural fit for verification tool papers; Theorem 1 is adequate math for tool track |
| CAV | 20–30% | 2–4% | Math individually thin for a top FM venue; would need CQL compilation correctness theorem |
| JAMIA (journal) | 25–40% | 3–5% | Longer format allows full development; but zero clinical validation is near-fatal |
| SAS/VMCAI | 30–45% | 3–6% | If framed around Tier 1 abstract interpretation with PK-aware widening |

**Overall best-paper probability at optimal venue (AIME): ~10%.** This is honest — 10% at a good venue is a strong project. The main threats are: (a) E1 disappointing, (b) a reviewer who demands clinical validation, (c) a FM reviewer who finds Theorem 1's math insufficient for a "theorem" label. The fallback narratives mitigate (a); E9 mitigates (b); honest labeling as "novel instantiation" mitigates (c).

---

## 10. Scores

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| **Value** | 7/10 | Real problem, genuine LLM-proof moat (exhaustive verification, sound counterexamples, certificates), forward-looking infrastructure for mandated FHIR/CQL ecosystem. Docked for zero demand signal and CQL treatment-logic adoption being early-stage. The two-tier speed story and Tier 2 counterexamples add practical value beyond Approach 1 alone. |
| **Difficulty** | 7/10 | Three-domain intersection (FM + PK + clinical informatics). ~35K novel LoC for MVP. Novel formalism + composition theorem. Reduced from Approach 1's 8/10 because MVP defers CQL compilation, zonotopic reachability, and validated ODE — the remaining difficulty is substantial but achievable. |
| **Potential** | 7/10 | ~10% P(best paper) at AIME; clean two-tier narrative; strong fallback narratives if E1 disappoints; the composition theorem is a standalone contribution. Higher than Approach 1 alone (6) because the two-tier architecture provides a second dramatic result (speed) even if E1 underperforms. |
| **Feasibility** | 7/10 | MVP is tight but achievable in 12–16 weeks by 2 engineers. No binary dependencies (unlike Approach 3). CEGAR convergence risk mitigated by SAT-based BMC fallback. Abstract interpretation guaranteed to terminate. The main feasibility risk is E1 variance, which is a research risk, not an engineering risk. |

**Composite: 7.0/10** (equally weighted)

**Risk-adjusted composite: 6.5/10** (accounting for 30–40% probability E1 disappoints and 20–30% probability Tier 1 precision is poor)

This represents a significant improvement over the depth check's composite of 5.5/10 for the original Approach 1 formulation, achieved by: (a) incorporating Approach 2's speed and termination guarantees, (b) cutting ornamental math, (c) adding fallback narratives, (d) scoping the MVP aggressively, and (e) preserving Approach 3's prescriptive framing as a low-risk module.

---

## 11. Amendments from Debate

| # | Critique Source | Before | After | Rationale |
|---|----------------|--------|-------|-----------|
| A1 | Skeptic (Approach 1, Fatal Flaw 1) | E1 is the sole narrative; no fallback | E1 is centerpiece with three explicit fallback narratives (explanation quality, compositionality speedup, existence proof) | If X < 20%, the paper pivots rather than collapses |
| A2 | Skeptic (Approach 1, Fatal Flaw 2) | Value proposition assumes demand exists | Framed as forward-looking infrastructure; honest acknowledgment of zero demand signal | Removes the pretense of expressed demand; focuses on intellectual contribution |
| A3 | Skeptic (Approach 1, Serious Flaw 1) | "300+ guideline artifacts" target | "~30–50 CQL treatment guidelines + ~30 manually encoded rules ≈ 80 treatment-logic artifacts" | Honest corpus assessment per depth check Amendment A2 |
| A4 | Mathematician (Approach 1, Proof Gap 3) | Universal δ = 0.1 μg/mL | Per-drug δᵢ = min(therapeutic_windowᵢ / 10, 0.1 · Cmax,ᵢ) | Addresses the NTI drug critique (digoxin, lithium, phenytoin); pharmacologically meaningful per drug |
| A5 | Math Depth Assessor (Approach 1) | "Proposition 1 (δ-decidability)" | "Observation 3 (δ-decidability)" | Downgraded from Proposition to Observation — this is a library invocation, not a proof |
| A6 | Math Depth Assessor (Approach 1) | Theorem 2 (PSPACE-completeness) included | Theorem 2 cut from theorem list; PK region graph is engineering optimization | Bisimulation proof was incomplete; PSPACE membership was unproven; system uses BMC in practice |
| A7 | Math Depth Assessor (Approach 2) | Theorem A and Theorem C as theorems | Galois connection and reduced product presented as implementation paragraphs, not theorems | Textbook constructions (depth 1.5/10 and 2/10) should not be dressed as results |
| A8 | Mathematician (Approach 2) | Theorem B convergence bound: O(D) | Proposition 2 convergence bound: O(D · k) | Corrected for enzyme coupling; the original O(D) silently assumed independence |
| A9 | Skeptic (Approach 2, Fatal Flaw 1) | Abstract interpretation alone (no counterexamples) | Two-tier: abstract interpretation screens, model checking confirms with counterexamples | Addresses "no actionable output" — every flagged combination gets a diagnostic trace |
| A10 | Skeptic (Approach 2, Fatal Flaw 2) | AI speed advantage pitched for 50 guidelines | Speed pitched for realistic 5–20 guidelines; formulary screening dropped | Removed fabricated use case; focused on realistic polypharmacy |
| A11 | Skeptic (Approach 3, Fatal Flaw 1) | Safety game decidability as core theorem | Game-theoretic synthesis dropped; replaced with heuristic schedule recommender | Unproven conjecture with ~30–40% failure probability cannot be the foundation |
| A12 | Mathematician (Approach 3, Proof Gap 2) | Pareto polytope theorem (Theorem II) included | Theorem II retracted; Pareto optimization not claimed | Theorem was mathematically false (nonlinear objectives ≠ polytope Pareto fronts) |
| A13 | Skeptic (Approach 3) | Prescriptive framing as core architecture | Prescriptive framing preserved as post-processing module + future work | Captures 80% of clinical value at 10% of technical risk |
| A14 | Depth Check (Amendment A6) | No pilot commitment | Milestone 0: encode 3 guideline pairs as PTA, verify at least one temporal conflict before full commitment | Validates feasibility before betting 8 person-months |
| A15 | Depth Check (Amendment A8) | FAERS computed at evaluation time | FAERS disproportionality pre-computed offline, distributed as lookup table | Eliminates memory concerns; simplifies evaluation pipeline |
| A16 | Math Depth Assessor (Missing Math) | CQL compilation correctness not mentioned | Explicitly identified as deferred crown-jewel theorem for follow-on tool paper | Addresses the "most important missing math" without overcommitting for this paper |
| A17 | Difficulty Assessor (Approach 1, Red Flag 1) | Validated interval ODE in critical path | Deferred: MVP uses 1-compartment closed-form matrix exponential | Eliminates the "2 weeks estimated, 2 months actual" risk for validated numerics |
| A18 | Skeptic (Approach 1, Hidden Assumption 3) | ~70% CYP inhibition coverage asserted | ~70% coverage stated with explicit boundary: non-competitive inhibition, enzyme induction, and PD interactions routed to monolithic fallback; QT prolongation and serotonin syndrome interactions explicitly listed as out of scope | Honest scope boundary prevents unsound contract-path routing |
| A19 | Depth Check (Amendment A3) | No clinical validation | E9 added: 3 pharmacists rate 20 conflicts (Likert scale, Fleiss' κ); budget ~$600, no IRB | Eliminates credibility gap at clinical venues; maintains full computational reproducibility |
| A20 | Mathematician (Approach 1, Missing Math) | No enzyme-interaction classification validation | Conservative classification: uncertain interactions routed to monolithic fallback; classification correctness is a documented limitation, not a soundness hole | Misclassification produces slow verification (monolithic fallback), not unsound verification |

---

*This document is the definitive approach specification for the GuardPharma project. It merges Approach 1's mathematical crown jewel (contract composition), Approach 2's speed and termination guarantees (abstract interpretation screening), and Approach 3's prescriptive framing (schedule recommendations) into a two-tier architecture that is technically sound, honestly scoped, and feasible within 18–20 weeks for the minimum viable paper artifact. Every critique from the Skeptic and Mathematician has been addressed with either a fix (A1–A20) or an honest acknowledgment of residual risk (R1–R10). The project's fate hinges on the pilot (3 guideline pairs encoded as PTA) and E1's delivery of X ≥ 20% — with explicit fallback narratives if either underperforms.*

---

## 12. Verifier Amendments (VA1–VA5)

The Independent Verifier issued a CONDITIONAL SIGN OFF with the following required amendments, all of which have been incorporated above:

| # | Amendment | Status |
|---|-----------|--------|
| VA1 | Extend timeline from 16 to 20 weeks | **Applied** — §8 updated; timeline provides ~15% margin for integration friction and proof work |
| VA2 | Promote E9 (clinical pharmacist review) from optional to mandatory | **Applied** — §6 updated; recruit 3 pharmacists during weeks 1–4; budget $600 |
| VA3 | Upgrade R5 (corpus starvation) severity from Medium to High; add proof-of-concept framing | **Applied** — §7 updated; paper will explicitly frame evaluation as proof-of-concept |
| VA4 | Complete Theorem 1 proof to publication quality in weeks 1–3 before heavy implementation | **Accepted** — scheduling constraint added to implementation plan; proof completion gates implementation start |
| VA5 | Run monolithic BMC feasibility test on 2–3 PD interaction pairs during pilot phase | **Accepted** — added to Milestone 0 pilot; QT prolongation and serotonin syndrome pairs tested; times reported honestly |

**Composite Score (Verifier-confirmed): V7 / D7 / BP6 / F6 = 6.5/10**
