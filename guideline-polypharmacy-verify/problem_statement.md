# GuardPharma: Contract-Based Temporal Verification of Polypharmacy Safety Across Interacting Clinical Guidelines with Pharmacokinetic Semantics

**Slug:** `guideline-polypharmacy-verify`

---

## Problem Statement

### The Structural Failure Nobody Checks

Clinical practice guidelines are the backbone of evidence-based medicine. The ADA Standards of Care for diabetes, the ACC/AHA guidelines for hypertension, the KDIGO recommendations for chronic kidney disease — each is a carefully curated decision algorithm specifying when to initiate drugs, adjust doses, order labs, and escalate treatment. These guidelines are increasingly published as *computable artifacts*: CQL (Clinical Quality Language) libraries, FHIR PlanDefinition resources, and eCQM (electronic Clinical Quality Measure) documents. The HL7 FHIR Clinical Reasoning specification, CDS Connect, and the CQFramework reference implementation collectively form a growing ecosystem of machine-executable clinical logic. Under the 21st Century Cures Act, ONC now mandates FHIR-based CDS interoperability in certified EHR systems — meaning computable guidelines are transitioning from research prototypes to deployed clinical infrastructure.

But here is the structural failure: **each guideline is authored, validated, and published in isolation.** The ADA does not check its diabetes guideline against ACC/AHA's hypertension guideline. KDIGO does not verify that its renal dosing constraints are consistent with the CHEST anticoagulation protocol. No tooling exists — anywhere in the HL7 ecosystem, anywhere in the CDS vendor stack, anywhere in regulatory oversight — to ask: *"If guidelines G₁, G₂, …, Gₖ activate simultaneously for a patient with comorbidities C₁, …, Cₘ, do they ever produce contradictory, dangerous, or pharmacokinetically unsafe recommendations?"*

The average Medicare beneficiary has 6.8 chronic conditions. Most such patients are governed by multiple interacting guidelines that nobody has verified for mutual consistency. This is not a hypothetical concern: polypharmacy-related adverse drug events are a leading cause of hospitalization in older adults, and published case reports document patients harmed by recommendations that individually followed best-practice guidelines but collectively produced dangerous drug combinations. As computable CDS adoption accelerates under federal mandate, the absence of multi-guideline safety verification is a gap that must be closed *before* wide deployment — not after.

### Why Existing Approaches Don't Solve It

The clinical informatics community is aware of the problem. Five lines of prior work exist, and all fall short in specific, identifiable ways.

**(1) TMR (Transition-based Medical Recommendation model)** by Zamborlini et al. (Semantic Web Journal, 2017; extended with argumentation-based severity ranking, 2024) uses first-order Semantic Web rules to detect interactions between guideline recommendations. TMR is *atemporal* — it reasons over recommendation-level abstractions ("recommend drug X," "avoid drug Y") without modeling *when* recommendations fire, *how long* drugs persist in the body, or *what sequence* of guideline decisions produces the conflict. It handles only pairwise interactions.

**(2) MitPlan** (Wilk et al., Artificial Intelligence in Medicine, 2021) uses AI planning with durative operators to detect and mitigate N-guideline conflicts. MitPlan handles temporal aspects and multi-guideline composition, but it performs *planning-based heuristic search*, not exhaustive formal verification — it finds mitigation strategies that happen to avoid conflicts, but cannot prove that *no* conflict exists across all possible patient trajectories.

**(3) GLARE/META-GLARE** (Terenziani et al., 2023) executes multiple guidelines concurrently and detects conflicts via constraint propagation along a *single execution path*. It does not perform exhaustive verification.

**(4) UPPAAL-based clinical protocol verification** (various, surveyed in MDPI Sensors, 2025) applies timed-automata model checking to *individual* clinical protocols. These are one-off, hand-modeled case studies: no system automatically parses standard CIG representations or composes multiple guidelines.

**(5) Classical CIG frameworks** (Asbru, PROforma, SAGE) provide guideline representation languages with varying temporal and hierarchical capabilities, but none include formal verification engines for multi-guideline safety properties.

The gap is precise: **no existing system combines (i) automated compilation of standard computable guideline formats into formal models, (ii) automatic composition of N guidelines into a verifiable product model, (iii) temporal reasoning over pharmacokinetic constraints, and (iv) sound verification with counterexample generation.** GuardPharma fills this gap.

**Why not runtime monitoring?** An alternative to pre-deployment verification is runtime CDS monitoring — checking for interactions at prescription time (e.g., via SMART-on-FHIR apps or drug interaction databases like Lexicomp/Micromedex). Runtime checking is valuable but fundamentally different: it catches individual prescription-time conflicts but cannot prove that a *protocol* is safe across all possible patient trajectories. A runtime monitor says "this specific prescription pair is flagged"; GuardPharma says "these two guidelines can *never* produce a dangerous state for *any* patient matching the comorbidity profile." Pre-deployment verification catches design-level errors in guidelines before they reach patients; runtime monitoring catches execution-level errors at the point of care. Both are needed; GuardPharma provides the former.

**Why not LLMs with drug interaction databases?** Large language models with retrieval-augmented access to DrugBank, Lexicomp, or Micromedex can flag most known drug-drug interactions with mechanism explanations and dose adjustment suggestions. For individual prescription checking, this is often sufficient. But LLMs fundamentally cannot perform *exhaustive verification* — they cannot prove that no patient trajectory under a set of guidelines ever reaches a dangerous state. LLMs are stochastic and may miss edge cases; they cannot produce *sound counterexamples* (provably reachable violating traces) or *safety certificates* (formal proofs that two guideline sets are mutually safe). In a regulatory context where the distinction between "probably safe" and "provably safe" matters — as computable CDS moves from research prototypes to federally mandated infrastructure — formal verification provides the certification layer that probabilistic tools cannot. GuardPharma does not replace LLM-based or database-based drug interaction checking; it provides the formal guarantee layer above it.

### What GuardPharma Does

GuardPharma is a formal verification engine for multi-guideline clinical decision support under polypharmacy. It operates in four stages.

**Stage 1: Guideline Compilation.** The system ingests computable clinical guidelines via the existing HL7 CQL-to-ELM reference parser (cqframework/clinical_quality_language) and FHIR PlanDefinition resources via HAPI FHIR. It then *semantically compiles* these artifacts into a unified intermediate representation: *Pharmacological Timed Automata* (PTA). The CQL-to-PTA semantic compiler translates clinical decision logic — condition checks, medication initiations, dose adjustments, lab-value guards, timing constraints — into automaton transitions guarded by predicates over both discrete clinical state and continuous pharmacokinetic state. This compilation step has no precedent: no tool translates CQL/FHIR into hybrid automata.

**Stage 2: Pharmacokinetic State Modeling.** Each PTA augments classical timed automata with *pharmacokinetic state variables*: continuous variables whose evolution is governed by compartmental ODE systems (one-, two-, three-compartment models with first-order absorption and elimination) parameterized from published population PK models. Drug concentrations, AUC, and steady-state trough levels are first-class elements of the automaton state. Drug-drug interactions are modeled at the metabolic interface: CYP-enzyme inhibition modulates clearance rates, producing nonlinear concentration dynamics for co-administered drugs sharing metabolic pathways.

**Stage 3: Contract-Based Compositional Verification.** When N guidelines are active for a multimorbid patient, direct product-automaton verification faces exponential state explosion. GuardPharma addresses this through *contract-based compositional verification* organized around shared metabolic pathways. Each guideline is modeled as an open system with an *enzyme-interface contract*: an (assume, guarantee) pair over the shared CYP-enzyme activity vector. The assume component specifies the expected enzyme capacity range; the guarantee specifies the enzyme load the guideline's medications impose. Safety of the N-guideline composition is verified by checking that all enzyme-interface contracts are mutually compatible — that the cumulative enzyme load from all guidelines remains within bounds that keep drug concentrations below toxic thresholds. This contract-based approach avoids the three-body problem that defeats naive pairwise checking: the contracts explicitly track cumulative enzyme effects, so three drugs sharing CYP3A4 are handled by checking the aggregate CYP3A4 load, not by checking drug pairs independently. For guidelines without shared metabolic pathways, verification trivially decomposes. A CEGAR (counterexample-guided abstraction refinement) loop with clinical-domain abstractions manages remaining complexity.

**Stage 4: Clinical Significance Filtering.** Not every formal conflict is clinically dangerous. GuardPharma stratifies discovered conflicts by severity using: (a) DrugBank interaction severity, (b) Beers Criteria classification, (c) FAERS disproportionality signal strength, and (d) prevalence of the triggering comorbidity profile. Only critical and significant conflicts are reported as primary findings. This addresses the gap between formal inconsistency and clinical relevance without requiring human annotation.

### What an Honest Novelty Assessment Reveals

**Genuinely novel:** (a) The PTA formalism — timed automata with compartmental ODE state variables and CYP-enzyme interface semantics — does not exist in the hybrid automata literature; the closest decidable subclasses (rectangular automata, o-minimal automata) do not cover this model. (b) The δ-decidability result for PTA reachability via validated ODE integration over Metzler dynamics is a new domain-specific application with clinically meaningful error bounds. (c) The contract-based compositional verification organized around metabolic pathway interfaces is a novel instantiation of assume-guarantee reasoning for pharmacokinetic systems. (d) The CQL-to-PTA semantic compilation pipeline is unprecedented.

**Repackaged but necessary:** (e) CEGAR is a known technique; our clinical-domain abstractions are novel but the CEGAR framework is not. (f) Z3 is used as a backend for bounded model checking; we contribute the encoding, not the solver. (g) Zonotopic reachability computation follows techniques pioneered in CORA and SpaceEx; our contribution is the domain-specific decidability analysis, not the computational geometry. (h) The clinical significance filter integrates existing databases. The paper's intellectual contribution is concentrated in the PTA formalism, the contract-based decomposition, and the end-to-end compilation pipeline. Everything else is systems building — necessary, but not independently publishable.

---

## Value Proposition

| Stakeholder | Current State | With GuardPharma |
|---|---|---|
| **Hospital CDS committees** | Review guideline interactions manually; most never checked. Joint Commission cites CDS configuration errors as a top safety concern. | Upload computable guidelines, receive a formal safety certificate or a ranked list of conflicts with counterexample patient trajectories. |
| **EHR vendors** (Epic, Oracle Health) | Ship CDS content to thousands of hospitals; rely on post-deployment adverse event monitoring. | Pre-deployment formal verification of guideline interactions before FHIR-based distribution. |
| **Guideline developers** (ACC/AHA, ADA, KDIGO) | Publish guidelines independently with no cross-specialty consistency check. | Receive a compatibility report before publication: "Your new recommendation conflicts with KDIGO §4.3 for patients with eGFR < 30." |
| **Regulators** (ONC, CMS) | Mandating CDS interoperability under 21st Century Cures with no way to verify that interoperable CDS is *safe* CDS. | A verification standard for CDS safety certificates. |
| **Patients** | Polypharmacy-related ADEs are a leading cause of hospitalization in older adults, often from regimens where each drug individually followed guidelines. | Verified guideline consistency before deployment means the safety net works before the prescription is filled. |

**Framing:** GuardPharma is forward-looking infrastructure for the FHIR/CQL ecosystem — verification *before* deployment, not a retroactive fix. As computable guideline adoption accelerates under federal mandate, the question is not whether multi-guideline verification will be needed, but whether it will exist when it is needed.

---

## Technical Difficulty

| # | Subsystem | Description | Est. LoC | Why It's Hard |
|---|---|---|---|---|
| 1 | **CQL-to-PTA Semantic Compiler** | Consumes ELM (from HL7 reference parser); compiles clinical decision logic, temporal operators, terminology bindings, and medication actions into PTA transitions with PK state guards | ~18K | CQL has temporal operators, interval arithmetic, FHIR-path expressions, and dynamic data requirements. Semantic translation to hybrid automata transitions requires faithful modeling of clinical state evolution — this is a compiler for a domain-specific language into a formal model. |
| 2 | **FHIR PlanDefinition Compiler** | Parses PlanDefinition R4 resources (nested actions, triggers, timing constraints, conditions, dynamic values); compiles action workflows into PTA control flow | ~10K | PlanDefinitions have nested actions with complex timing (before-start, before-end, concurrent), related-action dependencies, and multiple expression language bindings. The workflow-to-automaton translation must preserve all timing semantics. |
| 3 | **Guideline Corpus Pipeline** | Automated ingestion from CDS Connect, CQFramework, and academic CQL repositories; dependency resolution; ValueSet expansion via terminology services; normalization and deduplication. **Important distinction:** CMS eCQMs are retrospective quality measures, not prospective treatment guidelines — they are ingested for compilation testing but do not produce polypharmacy conflicts. The core verification corpus consists of ~30–50 treatment-decision CQL guidelines supplemented by manually encoded clinical decision rules (Wells, CHADS₂-VASc, MELD, etc.) targeting ~80 total treatment-logic artifacts. | ~10K | Heterogeneous sources with different packaging, versioning, and ValueSet conventions. Must resolve transitive CQL library dependencies and expand ValueSets against SNOMED CT, ICD-10, RxNorm, LOINC hierarchies. |
| 4 | **Clinical State Space Model** | Formal model of patient state: conditions (onset, severity, temporal evolution), medications (dose, route, frequency, duration), lab values (continuous ranges + clinical thresholds), and their interdependencies | ~8K | Must faithfully represent clinical semantics while remaining amenable to formal verification. Condition-medication-lab interactions create a rich state space with domain-specific invariants. |
| 5 | **Pharmacokinetic Model Library** | Compartmental ODE models (1/2/3-compartment) with population PK parameters; Metzler-matrix representation; CYP-enzyme interaction models (competitive/noncompetitive inhibition, induction); steady-state and transient solvers; validated interval ODE integration (wraps existing validated ODE integrator, CAPD or VNODE-LP) | ~12K | Pharmacological fidelity is essential: models must match published PopPK parameters. CYP inhibition produces nonlinear dynamics (Michaelis-Menten at enzyme level). Validated interval arithmetic for δ-decidability requires directed rounding and wrapping-effect control. |
| 6 | **PTA Construction & Composition Engine** | Build PTA from compiled guidelines + PK models; construct product PTA; implement CYP-enzyme interface contracts; symmetry/partial-order reduction for product state space | ~18K | Product-automaton state explosion is the central scalability challenge. Contract construction requires extracting enzyme-load guarantees from PK model parameters — bridging formal methods and pharmacology. |
| 7 | **Zonotopic Reachability Engine** | Compute zonotopic over-approximations of reachable PK states; implement Metzler-monotone propagation; order-reduction schemes for convergence; interface with interval ODE solver for δ-decidability | ~16K | Novel reachability algorithm exploiting Metzler structure. Must handle drug discontinuation resets (non-monotone events) via partitioned analysis. Zonotope order reduction must preserve soundness while ensuring convergence. |
| 8 | **MTL Model Checker with CEGAR** | Region-based model checking for metric temporal logic over PTA; clinical-domain CEGAR loop (drug-class abstraction, lab-value coarsening, temporal aggregation); BDD and SAT backends for bounded model checking | ~22K | PK region construction (partitioning continuous concentration space at clinical thresholds) is domain-novel. CEGAR must balance precision against region count explosion. Bounded model checking with SAT for fallback when region graph is too large. |
| 9 | **Counterexample Generator & Clinical Narrator** | Extract minimal violating traces; translate to clinical narrative ("Patient with eGFR < 30 on metformin + lisinopril reaches toxic state at day 45..."); visualization of PK trajectories along counterexample | ~7K | Trace minimization on region graph with continuous PK dynamics. Clinical narrative generation requires mapping formal states back to clinical concepts with temporal context. |
| 10 | **Clinical Significance Filter** | DrugBank interaction severity; Beers Criteria 2023 lookup; FAERS disproportionality computation (PRR, ROR, BCPNN with Bonferroni/FDR correction); Medicare comorbidity prevalence scoring; composite severity ranking | ~10K | Multi-source data integration across different identifier systems and severity scales. FAERS processing at scale (20M+ reports) with rigorous statistical methodology. |
| 11 | **Terminology Integration Layer** | Interface to FHIR terminology services (wraps HAPI FHIR terminology services); SNOMED CT subsumption; ICD-10-CM hierarchy; RxNorm ingredient/brand resolution; LOINC panel structure; ValueSet caching and expansion | ~6K | 350K+ SNOMED concepts, full ICD-10 hierarchy. Must handle concept equivalence across coding systems (RxNorm ingredient ↔ DrugBank compound ↔ FAERS drug name). Performance-critical for compilation. |
| 12 | **Evaluation & Benchmarking Engine** | Automated benchmark harness: Beers/STOPP injection, TMR-comparison baseline, temporal ablation, compositionality measurement, DrugBank cross-validation, clinical report generation | ~13K | Fully reproducible pipeline from raw data to final tables. Must implement TMR-style atemporal checking as baseline for head-to-head comparison. Statistical rigor for all validation metrics. |
| | **TOTAL** | | **~135K** | |

**Phased delivery.** The ~135K total represents the full-vision system (of which ~83K is novel algorithmic code). The minimum viable best-paper artifact focuses on subsystems 4–9 (the novel verification core, ~86K LoC) plus subsystem 10 (clinical significance filter, ~10K) and subsystem 12 (evaluation, ~13K), totaling ~95K LoC — with guidelines manually encoded as PTA for the initial paper. Subsystems 1–3 (CQL/FHIR compilation, ~42K) and subsystem 11 (terminology, ~12K) are delivered as a follow-on tool contribution. Both phases are necessary for the full vision; the paper-phase contains ~95K LoC including ~65K novel algorithmic code. The first implementation milestone is a pilot on 3 guideline pairs (diabetes + hypertension, diabetes + CKD, hypertension + anticoagulation) to validate that PTA encoding is feasible, the model checker terminates, and at least one non-trivial temporal conflict is found.

The ~135K estimate reflects genuine complexity. ~83K LoC is novel algorithmic code; the remainder is necessary integration with existing tools (HAPI FHIR, Z3, CUDD, CQL reference parser, CAPD). Subsystems 1–3 constitute a semantic compiler for a non-trivial clinical language ecosystem. Subsystems 5–8 are the novel formal verification core.

---

## New Mathematics Required

### Proposition 1: δ-Decidability of PTA Reachability (the foundation)

**Statement.** Let P = (L, l₀, X, C, Φ, E, I) be a Pharmacological Timed Automaton where continuous state X evolves according to ẋ = M(u)x + B·d(t), where M(u) is a state-dependent Metzler matrix parameterized by CYP-enzyme inhibition effects u, d(t) is the dosing input, and Φ ∈ [φ_lo, φ_hi] ⊂ ℝᵖ are population PK parameters ranging over bounded intervals. The reachability problem — "does there exist φ ∈ Φ and a dosing schedule consistent with the guideline such that a designated unsafe location is reached?" — is δ-decidable: for any δ > 0, the verifier answers either "unsafe" (exact) or "δ-safe" (safe up to perturbation δ in continuous state variables).

**Proof strategy.** Apply validated interval ODE integration (in the style of Gao et al.'s dReal framework, using an existing validated integrator such as CAPD or VNODE-LP as the ODE backend) to the PTA's continuous dynamics. The key domain-specific contribution is establishing that δ can be chosen as the minimum pharmacologically meaningful concentration difference (typically 0.1 μg/mL for most drugs), giving the δ-decidability result *clinical* meaning: "δ-safe" means safe up to a perturbation below the threshold of pharmacological significance. Note: this is an application of established δ-decidability theory to a new domain, not a new decidability technique. The novelty is the domain-specific δ-calibration and the integration with the PTA formalism.

**Conditional strengthening.** For the subclass of PTA where M is constant per location and asymptotically stable (all eigenvalues have negative real part — physiologically: all drugs are eventually eliminated), we conjecture that exact decidability holds via convergence of zonotopic reachable-set approximations under Metzler-monotone order reduction. This is stated as a conjecture, not a theorem, pending resolution of the finite-stabilization question for zonotope sequences under order reduction. If proven, this strengthens the result; if not, δ-decidability remains the operational guarantee.

**Why this matters.** Without decidability/δ-decidability, the verifier may not terminate. This is the mathematical foundation that makes GuardPharma a *verification* tool rather than a *testing* tool. The distinction: existing approaches (GLARE, MitPlan) explore specific execution paths; GuardPharma exhaustively checks all paths within the δ-decidability guarantee.

**Relationship to existing tools.** CORA and SpaceEx perform zonotopic reachability for positive linear systems as sound over-approximations without decidability guarantees. dReal provides δ-decidability for general nonlinear hybrid systems. Our contribution is the domain-specific instantiation: establishing that pharmacologically meaningful δ values yield clinically actionable safety guarantees, and that the Metzler structure of PK dynamics enables efficient reachable-set computation (lower complexity than general nonlinear δ-decidability).

### Theorem 2: MTL Model Checking for PTA

**Statement.** Given a PTA P and a bounded-horizon MTL formula φ expressing a temporal safety property (e.g., "the predicted steady-state concentration of drug X never exceeds the toxic threshold while the patient is concurrently on drug Y within 90 days"), δ-decide whether all runs of P satisfy φ. The problem is PSPACE-complete for bounded-horizon MTL (hardness by reduction from timed-automata MTL; membership via the PK region graph construction).

**Key technical contribution.** The *pharmacokinetic region graph* partitions the continuous PK state space at clinical thresholds (toxic level, therapeutic range boundaries, sub-therapeutic level for each drug). This domain-specific discretization — as opposed to the infinitesimal clock-region equivalence of Alur-Dill — yields a finite graph whose size is bounded by the product of clinical threshold counts across drugs. The correctness argument (that this discretization is a sound abstraction for MTL properties referencing only clinical threshold predicates) is the technical core.

### Theorem 3: Contract-Based Compositional Safety (the crown jewel)

**Statement.** Let P₁, …, Pₙ be PTAs for N concurrent guidelines. For each CYP enzyme e shared by two or more guidelines, define an *enzyme-interface contract* Γᵢᵉ = (Aᵢᵉ, Gᵢᵉ) where:
- Aᵢᵉ (assume): the total CYP-e inhibition load from *other* guidelines is at most αᵢᵉ
- Gᵢᵉ (guarantee): guideline i's medications impose CYP-e inhibition load at most γᵢᵉ

**Safety Composition Rule:** The N-guideline composition satisfies a safety property φ (concentrations remain below toxic thresholds) if:
1. Each guideline Pᵢ satisfies φ under its assume-guarantee contract (verified individually with assumed enzyme capacity)
2. For each enzyme e, the sum of all guarantees satisfies every assumption: ∀i, Σⱼ≠ᵢ γⱼᵉ ≤ αᵢᵉ (mutual compatibility of contracts)
3. The enzyme activity vector under cumulative load remains in the safe region (enzyme saturation check)

**Fixed-point resolution.** Because each guarantee γᵢᵉ depends on the assumed enzyme capacity αᵢᵉ (drug concentrations depend on clearance rates, which depend on enzyme activity), the compatibility check is a fixed-point problem. We resolve this by computing *worst-case guarantees*: for each guideline, compute γᵢᵉ under the *minimum* assumed enzyme capacity (maximum inhibition from other guidelines). Metzler monotonicity ensures this is sound — lower enzyme capacity produces higher drug concentrations, which produce higher enzyme loads, so the worst-case guarantee is an upper bound. The resulting system of inequalities Σⱼ≠ᵢ γⱼᵉ(min-capacity) ≤ αᵢᵉ is checked once, without iteration. If it fails, the system reports a potential conflict and falls back to direct product verification for the enzyme group.

**Monotonicity proof sketch (competitive CYP inhibition).** For competitive inhibition, the enzyme clearance rate for drug i is CL_i = CL_i^0 · (1 - Σⱼ≠ᵢ I_j/(I_j + K_ij)), where I_j is the inhibitor concentration from guideline j and K_ij is the inhibition constant. Lower assumed enzyme capacity (higher external inhibition load) → lower CL_i → higher steady-state drug concentration C_i = dose_i/CL_i → higher enzyme inhibition contribution from guideline i. This chain is monotonically increasing, confirming that worst-case guarantees computed under minimum assumed enzyme capacity are sound upper bounds. For non-competitive inhibition and mixed mechanisms, this monotonicity may not hold; such interactions require direct product-PTA verification rather than contract-based decomposition.

**Why this works where pairwise checking fails.** The three-body problem — drugs A, B, C all inhibiting CYP3A4, where pairwise combinations are safe but the triple is not — is handled because condition (2) checks the *aggregate* CYP3A4 load (γ_B + γ_C ≤ α_A), not just pairwise loads. The contract mechanism tracks cumulative enzyme effects through the shared metabolic interface.

**Complexity reduction.** Verification decomposes into N individual guideline checks (each polynomial in single-guideline size) plus enzyme compatibility checks (linear in N per enzyme, polynomial in the number of shared enzymes). For M shared enzymes, total complexity is O(N · single-guideline-cost + N · M), compared to exponential product-automaton verification.

**What this is and isn't.** This is a novel *instantiation* of assume-guarantee reasoning (Pnueli 1985; Giannakopoulou et al. 2005; see also the Pacti contract-based design framework by Incer et al. 2022) for pharmacokinetic systems, not a fundamentally new compositional verification technique. The novelty is: (a) identifying CYP-enzyme activity as the correct interface abstraction for clinical guideline composition, (b) proving that Metzler PK dynamics support sound contract extraction (enzyme loads are monotone in drug concentrations for inhibition interactions), and (c) demonstrating that this decomposition captures the clinically dominant interaction class (CYP-mediated DDIs account for ~70% of clinically significant pharmacokinetic drug interactions).

**Honest limitations.** The contract-based decomposition covers CYP-mediated inhibition interactions (the dominant class). Enzyme *induction* (which decreases co-drug concentrations) and pharmacodynamic interactions (QT prolongation, serotonin syndrome) are outside the contract framework and require direct product-PTA verification with CEGAR. We estimate ≥70% of clinically significant PK interactions are covered by the compositional path; the remainder use the monolithic fallback. This boundary is stated upfront, not buried in limitations.

---

## Best Paper Argument

A best-paper committee selects this paper for five reasons:

1. **The problem is undeniable and timely.** Computable clinical guidelines are transitioning from research to deployment under federal mandate. Multi-guideline safety verification is infrastructure the field needs *now*, before wide deployment creates the next class of patient safety incidents. The framing — "verify before you deploy" — is forward-looking and constructive, not alarmist.

2. **The primary experiment is a temporal ablation that demonstrates unique value.** The centerpiece evaluation (E1) runs the same guideline corpus through GuardPharma (with temporal PK reasoning) and through a TMR-style atemporal checker, showing that a substantial fraction of real conflicts are *invisible* to existing approaches because they depend on temporal drug concentration dynamics. This is a clean, memorable result: "X% of guideline conflicts require temporal pharmacokinetic reasoning to detect."

3. **The mathematical contribution is genuine and honest.** The PTA formalism and its δ-decidability are new. The contract-based composition via CYP-enzyme interfaces is a novel and practically motivated instantiation of assume-guarantee reasoning. The paper is transparent about what is new (the domain-specific formalism and decomposition) versus what is applied (CEGAR, SMT solving, zonotopic reachability). This honesty strengthens rather than weakens the contribution.

4. **It bridges two communities with a working system.** Formal methods and clinical informatics have operated in parallel. This paper is the bridge — it takes compositional verification and applies it to polypharmacy safety. Neither community alone could have produced it. A JAMIA or AMIA best-paper committee that sees formal methods enabling patient safety will recognize its significance.

5. **Everything is fully automated and reproducible.** Zero human annotation. Every experiment runs from public data sources to final results tables without human intervention. Any reviewer can reproduce the evaluation on a laptop. This is a best-paper differentiator at venues where reproducibility is increasingly valued.

---

## Evaluation Plan (Fully Automated)

**Milestone 0 (pre-submission feasibility pilot):** Manually encode 3 guideline pairs as PTA — (1) ADA diabetes + ACC/AHA hypertension, (2) ADA diabetes + KDIGO CKD, (3) ACC/AHA hypertension + CHEST anticoagulation — and verify at least one non-trivial temporal conflict using the model checker. This pilot validates that PTA encoding is feasible, the model checker terminates on realistic instances, and temporal reasoning finds conflicts invisible to atemporal checking. The pilot is a hard gate: if PTA encoding fails or the model checker does not terminate, the project redirects to a theory paper on the PTA formalism.

| # | Experiment | Data Source | Metric | Role |
|---|---|---|---|---|
| **E1** | **Temporal ablation (PRIMARY)** | Full guideline corpus | Conflicts found by temporal PK reasoning vs. atemporal checking | **Centerpiece:** demonstrates unique value of PTA temporal semantics over TMR-style approaches. **A priori expectation:** temporal-only conflicts should be substantial because the dominant polypharmacy hazard pattern — drug A reaches steady state, drug B is added later and inhibits A's metabolism via CYP, A's concentration rises over days to toxic levels — is inherently temporal. Atemporal checkers flag "A and B interact" but miss the critical question of *when* and *whether* the concentration actually reaches toxicity given the specific dosing schedule and PK parameters. Published pharmacokinetic studies of time-dependent DDIs (e.g., fluconazole + warfarin onset at 3–7 days, amiodarone + digoxin onset at 1–3 weeks) suggest a significant fraction of clinically dangerous interactions have temporal dynamics invisible to atemporal methods. **Fallback narrative:** If temporal-only detection rate is modest (<20%), the paper pivots to: (a) *explanation quality* — PK trajectory counterexamples showing when and how toxicity develops are unprecedented regardless of detection rate, (b) *compositionality speedup* — E6 demonstrates practical value of Theorem 3 independently of E1, (c) *existence proof* — GuardPharma produces the first formal safety certificate for multi-guideline polypharmacy verification, a result no prior system has achieved. |
| **E2** | Head-to-head vs. TMR baseline | Same guideline corpus; re-implement TMR atemporal pairwise logic | Additional conflicts found; false negatives of TMR approach | Direct comparison with closest prior art |
| **E3** | Known-conflict recall | Beers Criteria 2023 (30+ interactions), STOPP/START v3 (80+ contraindications) injected into guideline pairs | Recall of known conflict detection | Validates basic correctness; target ≥90% recall |
| **E4** | DrugBank cross-validation | All discovered conflicts cross-referenced against DrugBank interaction severity | Precision for "critical" tier conflicts | Validates that formal conflicts correspond to pharmacological reality; target ≥70% precision |
| **E5** | Scalability | Synthetic guideline sets of size 2, 5, 10, 15, 20 | Wall-clock on laptop CPU (M-series MacBook, 16GB RAM) | Target: 20 guidelines in <120s with contract-based decomposition |
| **E6** | Compositionality speedup | Same guideline sets; contract-based vs. monolithic verification | Wall-clock ratio | Demonstrates practical value of Theorem 3 |
| **E7** | Clinical significance stratification | All conflicts ranked by composite severity | Distribution of critical / significant / informational | Demonstrates filter value; expect ≥30% classified as informational |
| **E8** | Compilation fidelity | CDS Connect + CQFramework CQL libraries | Semantic round-trip fidelity of CQL-to-PTA-to-clinical-narrative | Validates compilation pipeline |
| **E9** | Clinical pharmacist review | Top 20 discovered conflicts | Inter-rater reliability (Fleiss' κ) on 5-point clinical relevance Likert scale | 3 clinical pharmacists independently rate discovered conflicts for clinical plausibility and severity. Adds clinical credibility while maintaining fully reproducible computational pipeline. Cost: ~$600. No IRB required (no patient data, no human subjects). |

---

## Laptop CPU Feasibility

Every computational component is CPU-native:

- **CQL compilation:** The HL7 reference CQL-to-ELM parser is a standard compiler; the PTA semantic compiler performs AST-to-automaton translation — pure CPU work completing in seconds for individual guidelines.
- **PK ODE solving:** Linear compartmental models have closed-form matrix-exponential solutions. For 1–3 compartment models, this is a 3×3 matrix exponential — microseconds per evaluation. Interval ODE solving for δ-decidability is fast for small-dimensional systems (≤5 drugs = ≤15 ODE variables).
- **Zonotopic reachability:** Zonotope operations (Minkowski sum, linear map) are polynomial in zonotope order and state dimension. For m drugs, the zonotope is m-dimensional — tractable on CPU for m ≤ 10.
- **Model checking with CEGAR:** BDD-based model checking (BuDDy/CUDD) and SAT-based bounded model checking (CaDiCaL/MiniSat) are pure CPU. CEGAR dramatically reduces explored state space.
- **SMT solving:** Z3 runs on CPU; no GPU support exists.
- **Contract checking:** Enzyme-interface compatibility is a system of linear inequalities (sum of guarantees ≤ each assumption) — solvable in microseconds.

**Memory budget:** With contract-based decomposition, the largest in-memory structure is a single-guideline PTA with its PK model (~10MB) or a two-guideline product for shared-enzyme verification (~100MB after CEGAR pruning). Well within 16GB laptop RAM.

**FAERS pre-processing:** FAERS disproportionality measures (PRR, ROR, BCPNN) for all drug pairs are pre-computed offline and distributed as a lookup table (~500MB). This converts a large-scale data processing task into a table lookup at evaluation time, eliminating FAERS as a runtime bottleneck.

---

## Key Risks and Mitigations

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| 1 | **δ-decidability runtime too slow for 5+ concurrent drugs** | Medium | Contract-based decomposition means we verify guidelines individually or in small groups sharing enzymes, not the full N-drug product. Expected: 2–3 drugs per enzyme-group verification. |
| 2 | **Contract-based decomposition covers only CYP-mediated interactions (~70%)** | Medium | Stated upfront as a design boundary. Non-CYP interactions (QT prolongation, serotonin syndrome, nephrotoxicity) use direct product verification with CEGAR. Report coverage statistics honestly. |
| 3 | **Small computable guideline corpus** | Medium | The honest count: ~30–50 CQL treatment-decision guidelines from CDS Connect, CQFramework, and academic sources, supplemented by ~30 manually encoded clinical decision rules (Wells, CHADS₂-VASc, MELD, etc.), targeting ~80 treatment-logic artifacts. CMS eCQMs (hundreds of measures) are ingested for compilation testing but are retrospective quality measures, not treatment guidelines — they do not produce polypharmacy conflicts. CQL treatment-logic adoption is early-stage but accelerating under ONC HTI-1 mandates (effective Jan 2024). |
| 4 | **Formal conflicts may not be clinically significant** | High | Clinical significance filter with severity stratification. Results reported by tier — headline uses only critical + significant conflicts. Experiment E7 quantifies filter value. |
| 5 | **CQL language coverage gaps** | Medium | Implement full CQL 1.5 expression language (covers >90% of guideline logic). For unsupported constructs (aggregate functions over unbounded collections), use sound over-approximation. Document exact coverage boundary. |
| 6 | **CEGAR may not converge for complex guideline pairs** | Medium | Fall back to bounded model checking (SAT-based) with 365-day horizon. This is sound for bounded temporal properties and avoids region enumeration. |
| 7 | **Drug discontinuation resets break Metzler monotonicity** | Medium | Partition analysis: separate reachability computation for "on drug" and "off drug" phases, connected by reset transitions. Each phase maintains Metzler structure. The partitioned analysis is sound and handles the clinically common scenario of medication start/stop sequences. |

---

## Prior Art Positioning Summary

| System | Temporal | N-guideline | Exhaustive | Auto-parsing | PK modeling | Output |
|---|---|---|---|---|---|---|
| TMR (2017, 2024) | No | Pairwise | No (rules) | No | No | Conflict type |
| MitPlan (2021) | Partial (durations) | Yes | No (planning) | No | No | Mitigation plan |
| GLARE (2023) | Partial (execution) | Yes | No (single-path) | Partial | No | Conflict log |
| UPPAAL studies | Yes (clocks) | Single | Yes | No | No | Counterexample |
| Asbru/PROforma | Partial | Manual | No | Partial | No | Varies |
| **GuardPharma** | **Yes (PK ODEs)** | **Yes** | **Yes (δ-decidable)** | **Yes (CQL/FHIR)** | **Yes (compartmental)** | **Ranked conflicts + counterexamples** |
