# Depth Check: RegSynth — Temporal Pareto-Optimal Compliance Trajectory Synthesis

**Slug:** `pareto-reg-trajectory-synth`
**Stage:** Verification (depth check)
**Date:** 2026-03-08
**Evaluator:** Best-paper committee chair (3-expert adversarial team)

---

## Team & Methodology

Three independent experts evaluated the crystallized problem statement through two rounds:

| Role | R1 Composite | R2 Composite | Movement |
|------|-------------|-------------|----------|
| Independent Auditor | 4.5 | 4.5 | 0 |
| Fail-Fast Skeptic | 3.0 | 3.0 | 0 (maintains ABANDON) |
| Scavenging Synthesizer | 6.75 | 5.5 | -1.25 (conceded on crown jewel, LoC, venue) |

**R2 Team Average:** V=4.7 / D=5.3 / BP=3.0 / L=4.3 → Composite 4.3

After adversarial synthesis, the lead assigns calibrated consensus scores below.

---

## Pillar 1: EXTREME AND OBVIOUS VALUE — Score: 5/10

### What's Real

The problem is genuine. The EU AI Act enters full enforcement August 2026. Organizations deploying AI across jurisdictional boundaries face a combinatorial explosion of obligations that genuinely interact. The Synthesizer cites concrete, live regulatory tensions:

- **EU AI Act Art. 12 (logging) vs. GDPR Art. 5(1)(c) (data minimization)**: The EDPB has issued multiple guidance opinions because this tension is real and unresolved. Organizations must make explicit trade-offs that are formally conflicting at the constraint level.
- **China Interim Measures Art. 17 (algorithmic disclosure) vs. U.S. DTSA (trade secret protections)**: Genuine impossibility if disclosure granularity exceeds safe harbors.
- **EU precautionary classification vs. Singapore AIGA proportionality**: Different risk categorizations for identical systems.

The Skeptic's claim that "the problem doesn't exist" is empirically wrong. Harmonization is aspirational, not achieved, especially across EU/China/US regulatory regimes.

### What's Missing

**No evidence of practitioner demand.** The entire value proposition is argued from first principles. Zero interviews, surveys, letters of intent, or case studies with compliance teams. The Auditor correctly identifies that compliance officers want *defensible checklists*, not *Pareto frontiers*. GRC platforms (OneTrust, ServiceNow) dominate because they match practitioner workflows.

**Wrong form factor.** A "4-dimensional Pareto frontier over temporally-unrolled MaxSMT encodings" is not what any compliance officer has asked for. The Synthesizer's rebuttal — "compliance officers didn't ask for spreadsheets before VisiCalc" — is clever but unpersuasive. VisiCalc matched an existing workflow (calculation tables). RegSynth imposes a new workflow (formal trajectory optimization) on a domain governed by satisficing and legal precedent.

**The real value is buried.** Infeasibility proofs and conflict detection — the ability to formally prove "Articles X, Y, Z from EU, US, and China are mutually unsatisfiable" — are the commercially irreplaceable features. No GRC platform or LLM can do this. This should be the headline, not the Pareto trajectory optimization. The Synthesizer identified this correctly; the original problem statement buries it.

**The 60–70% formalizability ceiling is a selling point AND a liability.** The formalizability grading (Full/Partial(α)/Opaque) is intellectually honest, but it means the system cannot reason about 30–40% of obligations — precisely the ambiguous, principles-based obligations that create the most compliance uncertainty.

### Lead Verdict on Value

The problem is real but the solution's form factor is mismatched to the market. Infeasibility proofs have genuine standalone value. Pareto trajectory optimization is academically interesting but practically over-engineered. **Score: 5/10.**

---

## Pillar 2: GENUINE DIFFICULTY AS A SOFTWARE ARTIFACT — Score: 6/10

### What's Genuinely Hard

- **DSL with formalizability grading**: A typed DSL with obligation types indexed by jurisdiction, temporal intervals, and formalizability grades, with a composition algebra satisfying algebraic laws. No existing DSL handles this combination. This is real PL design.
- **11-component integration**: DSL → compiler → temporal unroller → MaxSMT solver → Pareto enumerator → roadmap planner → certificate generator is a genuine end-to-end systems challenge with provenance tracking throughout.
- **Dual-solver backend with cross-validation**: Running Z3 and CVC5 on identical instances catches solver bugs and encoding errors. The abstraction layer is non-trivial.
- **Machine-checkable certificates**: Three certificate types (compliance, infeasibility, Pareto optimality) with a standalone verifier <2,500 LoC. Novel in the regulatory domain.

### Honest LoC Accounting

The 153K LoC headline is inflated. All three experts converged on this:

| Category | Claimed | Actual Novel | Notes |
|----------|---------|-------------|-------|
| Genuinely novel algorithmic code | ~65K (42%) | ~50-65K | DSL (A), Encoding Engine (D), Solver (E), Pareto Synthesis (F), Temporal Model (C) |
| Regulatory Knowledge Base (B) | 38.5K | ~10K engineering + 28K data | Encoding 500+ articles is labor, not research. The ontology/mapping logic (~10K) is engineering; the article encodings (~28K) are domain data. |
| Evaluation Framework (I) | 20.7K | Standard | Benchmark generation and metrics — necessary but routine |
| Infrastructure (J, K) | 16.5K | Standard | CLI, reporting, logging, build — boilerplate |

**Effective research-novel LoC: ~50–65K.** Still substantial — roughly on par with a real compiler or verification tool — but not 153K.

### The ILP Challenge

Both the Auditor and Skeptic argue that ILP (Gurobi) could deliver 80-90% of practical value at 20-30% of the complexity. The original problem statement includes ILP as a baseline and concedes "comparable quality." For static single-timestep optimization, this is likely true. The MaxSMT approach adds value specifically for: (a) deeply nested logical implications ill-suited to ILP linearization, (b) formal infeasibility certificates via UNSAT cores, and (c) temporal trajectory coupling. Whether this marginal value justifies the complexity premium is debatable.

### Lead Verdict on Difficulty

Real end-to-end systems complexity with a genuine novel core of ~55K LoC. The DSL and certificate generator are legitimately novel artifacts. But the 153K headline inflates by ~2.5× and the ILP alternative significantly undercuts the necessity of the MaxSMT approach. **Score: 6/10.**

---

## Pillar 3: BEST-PAPER POTENTIAL — Score: 3/10

### The Crown Jewel Is Not Novel

All three experts converged on this after the adversarial round:

- **Auditor**: "Textbook observation in multi-period optimization. Grade: C+."
- **Skeptic**: "Trivially true. Bellman 1957 established that myopic per-stage optimization produces suboptimal trajectories under coupling."
- **Synthesizer** (after conceding): "B-level insight. Clean, publishable, but not paradigm-shifting."

The theorem — "per-timestep Pareto optimality does not imply trajectory Pareto optimality" — is the motivating insight behind dynamic programming, known for 70 years. The constructive proof via a 3-timestep, 2-jurisdiction instance is pedagogically useful but mathematically trivial. The self-grade of B+/A- is generous; the team consensus is B-/C+.

### Mathematical Novelty is Thin

| Contribution | Claimed Grade | Team Assessment |
|-------------|--------------|-----------------|
| M1: Temporal Pareto trajectory | B+/A- | B-/C+ — known phenomenon, new domain |
| M2: Pareto via iterative MaxSMT | B | C+ — ε-constraint scalarization is standard in OR |
| M3: Compliance bisimulation | B- | C+ — partition refinement over finite sets, known since Kanellakis-Smolka 1990 |
| M4: Incremental Pareto maintenance | B- | C — test old points against new constraints, re-solve gaps |
| M5: Relative completeness under formalizability | B | B — most intellectually honest contribution; bounds what formal tools can say |
| M6: Obligation algebra soundness | C+ | C+ — standard compiler correctness homomorphism |

The Synthesizer's recommendation to elevate M5 to co-crown-jewel has merit — it's the most novel *intellectual* contribution. But the Auditor and Skeptic both note it's conditional on unvalidated encoding correctness.

### Venue Fit is Awkward

| Venue | Fit | Problem |
|-------|-----|---------|
| ICSE | Best | Requires user studies the evaluation plan excludes |
| FAccT | Possible | Values empirical sociotechnical analysis, not formal systems |
| CAV | Poor | Regulatory domain seen as toy; formalization is shallow for CAV |
| AAAI | Poor | Optimization is standard; no AI contribution |

The Synthesizer argues FAccT would value the "compliance is a trajectory, not a checklist" reframing. The Auditor counters that FAccT wants empirical work with real-world impact. Neither argument is conclusive, but the realistic best-paper probability is **3–6%** at any venue.

### Lead Verdict on Best-Paper

The crown jewel is a clean formalization of a known phenomenon. The mathematical contributions are applications of existing techniques to a new domain. No venue is a natural home. The paper is publishable but not best-paper material without significant reframing. **Score: 3/10.**

---

## Pillar 4: LAPTOP CPU + NO HUMANS — Score: 6/10

### CPU Feasibility is Credible

All three experts agree: MaxSMT solvers (Z3, CVC5) are CPU-native. DPLL/CDCL is inherently sequential backtracking search that does not benefit from GPU parallelism. The structural argument is correct.

### The 30-Minute Target is Unvalidated

The proposal targets 10 jurisdictions × 500 obligations × 10 timesteps in <30 minutes. The Skeptic estimates 17 hours based on SPL benchmark extrapolation; the Synthesizer calls this "fear-based." **Neither side has evidence.** Both are extrapolating from different domains with different constraint structures. The proposal cites "empirical calibration" from "similar domains" but provides no specific benchmarks or prototype data.

The kill gate G2 (week 8: 200-obligation, 3-jurisdiction in <5 minutes) is appropriate. The team consensus is: **plausible but unproven, with moderate-to-high risk of blow-up at target scale.**

### The Human Encoding Problem

The 38.5K LoC regulatory knowledge base (Component B) requires expert human legal-to-formal translation. The Synthesizer's defense — "this is artifact construction, like SPEC CPU benchmarks" — is valid: the evaluation *execution* is fully automated even though the artifact *creation* requires human expertise. However:

- **Encoding correctness has no automated validation path.** Cross-solver validation catches solver bugs, not encoding bugs.
- **Circular validation**: The same team writes encodings and tests.
- **30–40% of obligations are Opaque** — excluded from formal reasoning entirely.

### The Planted-Solution Methodology is Sound

Generating synthetic regulatory instances with known Pareto frontiers provides automated ground truth. This is a well-established technique (planted clique, planted partition) and is genuinely clever for this domain.

### Lead Verdict on Laptop+NoHumans

Solver is CPU-native (good). Planted-solution benchmarks are clever (good). 30-minute target is unvalidated (risky). Encoding is pre-done human artifact (acceptable for evaluation, problematic for correctness claims). **Score: 6/10.**

---

## Pillar 5: Fatal Flaws

### Flaw 1: Encoding Correctness is Unvalidatable
- **Severity: SEVERE (borderline FATAL)**
- The system's formal guarantees are only as good as τ (the translation from regulatory text to SMT). Legal text is inherently ambiguous; two lawyers may reasonably disagree on the formalization of any article. No automated process can validate τ captures legal intent. The formalizability grading (M5) is an honest mitigation but doesn't resolve the fundamental problem.
- **Fixable?** Partially. External legal review of a sample of encodings would break the circularity. But full validation of 500+ encodings requires a parallel legal review project.

### Flaw 2: Crown Jewel Theorem is Not Novel
- **Severity: SEVERE**
- The central mathematical claim is a well-known observation in multi-stage optimization. Reviewers in OR, optimization, or AI planning will recognize this immediately. The paper needs a different crown jewel or honest framing as a "systems contribution with formal grounding" rather than "novel mathematical result."
- **Fixable?** Yes. Reframe M1 as "motivating formalization" and elevate M5 (formalizability completeness) plus infeasibility detection as primary contributions.

### Flaw 3: No Evidence of Practitioner Demand
- **Severity: SEVERE**
- Zero user research, interviews, or adoption evidence. The demand is argued as "latent" but unvalidated. If compliance officers don't want formal optimization, the system is an academic exercise.
- **Fixable?** Yes, with even minimal practitioner engagement (2-3 interviews with compliance officers at multinationals).

### Flaw 4: LoC Inflation
- **Severity: MODERATE**
- 153K LoC headline includes 38.5K of domain data, 20.7K of evaluation infrastructure, and 16.5K of boilerplate. Genuine novel core is ~50-65K. Inflated claims undermine credibility.
- **Fixable?** Yes. Honest accounting: present ~85K LoC system core + ~28K regulatory data + ~20K evaluation separately.

### Flaw 5: Scalability Unvalidated
- **Severity: MODERATE**
- The 30-minute target for 10×500×10 instances has no prototype evidence. Kill gate G2 (week 8) is appropriate but the headline claim should be qualified.
- **Fixable?** Yes. Add a week-6 scalability prototype gate. Qualify the 30-minute claim as a target, not a guarantee.

### Flaw 6: Venue Mismatch
- **Severity: MODERATE**
- No single top venue is a natural home. The multi-venue targeting signals indecision.
- **Fixable?** Yes. Commit to ICSE (tools track) as primary, with FAccT as secondary.

---

## Final Scores

| Pillar | Score | Below 7? | Key Issue |
|--------|-------|----------|-----------|
| **Value** | **5/10** | YES | Real problem, wrong form factor. Infeasibility proofs are the real value; Pareto trajectories are over-engineered. No practitioner evidence. |
| **Difficulty** | **6/10** | YES | ~55K novel LoC, not 153K. DSL and certificates are genuinely hard. ILP captures most practical value at far lower complexity. |
| **Best-Paper** | **3/10** | YES | Crown jewel is a known observation. Math contributions are applications of standard techniques. No natural venue home. P(best-paper) ≈ 3-6%. |
| **Laptop+NoHumans** | **6/10** | YES | SMT is CPU-native. Planted benchmarks are clever. But 30-min unvalidated, encoding is human work, 30-40% obligations excluded. |
| **Composite** | **5.0/10** | — | |

---

## Required Amendments

All four scores are below 7. The following amendments are **mandatory** for CONDITIONAL CONTINUE:

### Amendment A1: Reframe the Crown Jewel
**Problem:** M1 (temporal Pareto trajectory) is a known phenomenon from dynamic programming applied to a new domain. Self-grading as B+/A- is indefensible.
**Fix:** Demote M1 to "motivating formalization" (Grade: B-). Elevate the contribution triad: (1) formal infeasibility detection with MUS-mapped regulatory diagnoses, (2) formalizability completeness theorem (M5), (3) the regulatory DSL with formalizability grading as a PL contribution. Frame the paper as "making AI regulatory compliance formally tractable" rather than "proving per-step optimization is suboptimal."

### Amendment A2: Honest LoC Accounting
**Problem:** 153K LoC headline inflates by ~2.5× relative to novel core.
**Fix:** Present as: ~55K novel algorithmic core + ~28K regulatory data corpus + ~20K evaluation framework + ~12K infrastructure = ~115K total. The novel research contribution is ~55K LoC. Drop the 153K headline.

### Amendment A3: Scope Reduction to 4–5 Jurisdictions
**Problem:** 10 jurisdictions × 500+ obligations is enormously ambitious and the 38.5K knowledge base is a maintenance nightmare.
**Fix:** Core scope: EU AI Act, NIST RMF, ISO 42001, China Interim Measures, GDPR (AI-relevant) = 5 frameworks, ~300 obligations. Stretch goal: add 5 more. This reduces Component B from 38.5K to ~18K LoC while covering the most commercially relevant cross-jurisdictional tensions (EU-US, EU-China).

### Amendment A4: ILP Baseline-First Architecture
**Problem:** ILP (Gurobi) may deliver 80-90% of practical value at far lower complexity.
**Fix:** Mandate ILP-based static compliance optimization as the first implementation milestone (week 6). MaxSMT temporal trajectory optimization is the *research delta* built on top of a working ILP baseline. If ILP is sufficient for practitioner needs, the temporal extension becomes the paper contribution; if it's insufficient, the MaxSMT approach is justified empirically.

### Amendment A5: External Encoding Validation
**Problem:** Circular validation — same team writes encodings and tests.
**Fix:** Require 2–3 external regulatory domain experts to independently validate a stratified sample (≥50 articles spanning 3+ jurisdictions) of DSL encodings against the original regulatory text. Report inter-annotator agreement metrics. Budget ~$15–30K for legal review.

### Amendment A6: Early Scalability Gate
**Problem:** The 30-minute target for 10×500×10 is unvalidated.
**Fix:** Add kill gate G1.5 at week 6: run a prototype on 3 jurisdictions × 100 obligations × 5 timesteps. If solve time exceeds 10 minutes, restructure temporal unrolling (approximate methods, decomposition) before committing to full scale.

### Amendment A7: Venue Commitment
**Problem:** Multi-venue targeting (ICSE, FAccT, AAAI, CAV) signals indecision.
**Fix:** Primary: ICSE 2027 (tools track — the artifact is a novel software system). Secondary: FAccT 2027 (if the infeasibility/conflict-detection angle resonates). Drop AAAI and CAV.

### Amendment A8: Lead with Infeasibility Detection
**Problem:** The most commercially and academically defensible feature (formal infeasibility proofs with MUS-mapped regulatory diagnoses) is buried in H7. The headline is Pareto trajectory optimization, which is over-engineered for the market and mathematically non-novel.
**Fix:** Restructure the narrative: (1) Conflict detection and infeasibility proofs are the primary contribution; (2) Pareto-optimal strategy synthesis is the secondary contribution; (3) Temporal trajectory optimization is the research extension. The "aha" moment is "we can formally prove that these regulations conflict" — not "per-step optimization is suboptimal."

---

## Verdict

**CONDITIONAL CONTINUE** at reduced scope with 8 mandatory amendments.

The project has a real problem (multi-jurisdictional AI compliance), a defensible technical approach (constraint solving with formal guarantees), and one genuinely underserved capability (formal infeasibility proofs). But the current formulation over-engineers the optimization angle, inflates the LoC count, mis-identifies the crown jewel, and lacks any evidence of practitioner demand.

With amendments A1–A8 applied, the project becomes a ~90K LoC system covering 5 jurisdictions with honest framing, an ILP baseline, external validation, and a clear venue target. Expected quality: solid ICSE tools-track paper, ~5% best-paper probability, ~60–70% acceptance probability. This is a viable research contribution, not a best-paper contender.

**Kill gates** (from amended proposal):

| Gate | Week | Condition | Fail Action |
|------|------|-----------|-------------|
| G1: DSL expressiveness | 4 | Encode 50+ EU AI Act articles with type-checking | Simplify; pivot to constraint language |
| G1.5: Scalability prototype | 6 | 3-juris × 100-oblig × 5-timestep in <10 min | Restructure temporal approach |
| G2: ILP baseline | 8 | ILP produces valid Pareto frontier for static case | MaxSMT approach may be unnecessary |
| G3: Trajectory dominance | 10 | Constructive example on real corpus (not synthetic) | Pivot to static Pareto (still publishable) |
| G4: External validation | 12 | ≥80% inter-annotator agreement on 50-article sample | Encoding methodology needs revision |
| G5: End-to-end pipeline | 18 | Full pipeline on 5-jurisdiction corpus with certificates | Reduce scope; extend timeline |

**Confidence:** 60% that the amended project produces a publishable paper. 5% best-paper probability. 40% kill probability at gates G1.5–G3.

---

## Team Score Summary

| Pillar | Auditor R2 | Skeptic R2 | Synthesizer R2 | Lead Final |
|--------|-----------|-----------|----------------|------------|
| Value | 5 | 3 | 6 | **5** |
| Difficulty | 6 | 4 | 6 | **6** |
| Best-Paper | 3 | 2 | 4 | **3** |
| Laptop+NoHumans | 4 | 3 | 6 | **6** |
| **Composite** | **4.5** | **3.0** | **5.5** | **5.0** |

*Skeptic recommended ABANDON. Auditor and Synthesizer recommended CONDITIONAL CONTINUE at reduced scope. Lead breaks tie: CONDITIONAL CONTINUE with mandatory amendments.*
