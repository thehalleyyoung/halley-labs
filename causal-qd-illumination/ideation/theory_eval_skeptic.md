# Skeptic Verification — Theory-Stage Evaluation

**Proposal:** proposal_00 — CausalQD: Quality-Diversity Illumination for Causal Structure Discovery
**Stage:** Verification (post-theory)
**Date:** 2026-03-02
**Method:** Claude Code Agent Teams (Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer → Adversarial Cross-Critique → Synthesis)

---

## Executive Summary

Three independent evaluators assessed proposal_00 on five pillars. After adversarial cross-critique and forced resolution of disagreements, the team reached consensus. **The original proposal is dominated by existing Bayesian methods (Order MCMC + BMA) and carries vacuous theoretical claims.** A salvageable kernel exists in the behavioral descriptor space design, but exploiting it requires a major pivot.

---

## Consensus Scores

| Dimension | Auditor | Skeptic | Synthesizer | **CONSENSUS** |
|-----------|---------|---------|-------------|---------------|
| Extreme Value | 5 | 3 | 5 | **4** |
| Genuine Software Difficulty | 6 | 5 | 6 | **6** |
| Best-Paper Potential | 4 | 2 | 3 | **3** |
| Laptop-CPU Feasibility | 4 | 3 | 7 | **4** |
| Feasibility | 6 | 5 | 7 | **6** |

**Composite Score: 4.6/10**

---

## VERDICT: CONTINUE — conditional on mandatory modifications

The verdict is **CONTINUE** only if ALL of the following non-negotiable modifications are adopted. If any condition is rejected, the verdict reverts to **ABANDON**.

---

## Pillar Analysis

### 1. Extreme Value (4/10)

**The core problem is real but already addressed by existing methods.**

The proposal identifies a genuine limitation: traditional causal discovery returns a single DAG or MEC, missing alternative plausible explanations. However:

- **Order MCMC** (Friedman & Koller 2003) and **Partition MCMC** (Kuipers & Moffa 2017) already sample diverse DAGs from the posterior with proper probabilistic semantics. They provide calibrated `P(X→Y | D)` — a number practitioners can act on. CausalQD's archive provides a bag of DAGs with no probabilistic interpretation.
- **Stability selection** (Meinshausen & Bühlmann 2010) already quantifies edge robustness under resampling.
- **Bootstrap aggregation** of GES outputs is standard practice for diversity.

**What is genuinely new:** Organizing diverse DAGs by a behavioral descriptor space (structural features × MI profiles × equivalence class features) is a novel lens. If this reveals interpretable structure in the space of plausible causal explanations, it has real value. But this value is incremental, not extreme.

**Kill shot identified by Skeptic:** The proposal does not include Order MCMC as a baseline — the single strongest existing method for diverse structure recovery with uncertainty quantification. The comparison would likely be unfavorable to CausalQD on every dimension practitioners care about (calibration, probabilistic semantics, theoretical guarantees, available implementations).

### 2. Genuine Software Difficulty (6/10)

**Moderate engineering challenge, not research-grade difficulty.**

Genuinely hard components:
- Acyclicity-preserving crossover with topological repair (cycle-breaking in merged graphs)
- Efficient behavioral descriptor computation (MI profiles with conditioning sets, O(n³·k_max³))
- CVT-MAP-Elites integration for high-dimensional descriptor spaces
- Nauty-based canonical graph labeling for MEC identification

Not genuinely hard (but presented as if they were):
- MAP-Elites archive (a dictionary; open-source implementations exist: pyribs, QDax)
- BIC scoring (available in causal-learn)
- Topological ordering-based mutation (textbook)
- Bootstrap certificates (a resampling loop)

**Assessment:** ~40% genuine systems engineering challenges, ~60% standard component integration. A strong PhD student could implement this in 3-4 weeks.

### 3. Best-Paper Potential (3/10)

**No chance in current form. Very slim chance with major pivot.**

The theoretical contribution is hollow:
- **Theorem 1** (topological mutation preserves acyclicity): Trivially follows from the definition of topological ordering.
- **Theorem 2** (crossover acyclicity): Straightforward given topological repair.
- **Theorem 3** (coverage under ergodicity): **CIRCULAR** — conditions on unproven ergodicity, which is the hard part. "If our method works, then our method works."
- **Theorem 4** (supermartingale convergence): **TAUTOLOGICAL** — "elitist archive quality is non-decreasing" is the definition of an elitist archive. Supermartingale language adds notation, not content. Convergence to optimum requires Theorem 3's unproven ergodicity.
- **Theorem 5** (MEC separation): Restates Verma & Pearl (1990). Zero novelty.
- **Theorem 5b** (PCA approximate separation): States the obvious — PCA loses information.
- **Theorem 7** (Lipschitz certificate): Standard perturbation analysis. **Also contains a factor-of-2 error** — statement claims `C(e;D') ≥ C(e;D) - L_BIC·δ` but proof derives `C(e;D') ≥ C(e;D) - 2·L_BIC·δ`.
- **Theorem 8** (path certificate): **Assumes independent edge perturbations**, which is mathematically wrong for correlated data. Edges sharing a node have correlated perturbations under bootstrap resampling.
- **Theorem 9** (Boltzmann stability): A weighted average with a temperature parameter. This is a definition, not a theorem.

**No theorem in this paper is both correct and non-trivial.**

The "first application of MAP-Elites to causal discovery" framing is a novelty of combination — reviewers increasingly penalize this pattern. The experimental plan is overscoped (6 research questions, 3200 CPU-hours) while under-baselined (no Order MCMC).

### 4. Laptop-CPU Feasibility (4/10)

**The proposal as written requires a 96-core server. A heavily scoped version is borderline laptop-feasible.**

The proposal explicitly states: "96-core server or equivalent cloud resources, parallelizable across runs" with 3200 CPU-hours total budget. This is 400 hours on an 8-core laptop (16.7 days continuous).

Breakdown by scale:
- **n=10**: Feasible on laptop (minutes per run)
- **n=20**: Feasible with patience (hours per run)
- **n=30**: Problematic (behavioral descriptors at O(n³), multiple hours per run × 30 replicates = days)
- **n=50**: Infeasible on laptop in reasonable time

The proposal mentions "GPU acceleration" in implementation notes, conflicting with the CPU-only constraint.

**A stripped-down version** (n≤20, 10 replicates, 3-5 datasets, ≤100 CPU-hours) could run on a laptop in ~4 days.

### 5. Feasibility (6/10)

**The core algorithm is buildable. The experimental plan is overambitious. The theoretical claims are unvalidatable.**

Feasible:
- MAP-Elites with DAG representation and grid archive
- Topological mutation operators
- BIC scoring with caching (leverage causal-learn)
- Structural behavioral descriptors
- Bootstrap certificates

Risky:
- **FIND-ALL-CYCLES in crossover repair is intractable** — simple cycle enumeration is exponential. Needs replacement with polynomial-time feedback arc set heuristic.
- **Ergodicity verification** (required for Theorem 3) has no specified methodology and may be impossible for super-exponential search spaces.
- **MI profile computation for nonlinear data** is acknowledged as invalid but nonlinear experiments are still listed.
- **6 research questions** for one paper guarantees nothing gets adequate attention.

---

## Fatal Flaws (Unanimous or Majority Agreement)

### F1. Missing Order MCMC Baseline [ALL THREE EVALUATORS]
The strongest existing method for diverse structure recovery with uncertainty quantification is not baselined. This is the most natural competitor, and the comparison would likely be unfavorable. The proposal even inadvertently admits this in Theorem 9's proof, where it says the Boltzmann weighting can be "tuned to match posterior sampling methods (such as Order MCMC)."

### F2. Circular Coverage Guarantee [ALL THREE EVALUATORS]
Theorem 3 (archive coverage) is conditional on ergodicity, which is never proven and deferred to "empirical verification." But empirical verification of ergodicity requires exactly what the theorem claims to guarantee — visiting all reachable cells. Theorem 4 (convergence) inherits this circularity. The entire theoretical edifice rests on sand.

### F3. Tautological Convergence Theorem [ALL THREE EVALUATORS]
Theorem 4's supermartingale argument just restates that elitist archives are monotonically non-decreasing — which is their definition. The supermartingale language adds notation without content.

### F4. No Decision-Theoretic Framework [SKEPTIC + SYNTHESIZER]
The archive produces a bag of DAGs with no framework for making decisions. Bayesian model averaging provides `P(X→Y | D)`. CausalQD provides... a browsing experience. The Boltzmann-weighted certificate (Theorem 9) is a post-hoc attempt with an arbitrary hyperparameter β and no probabilistic interpretation.

### F5. Math Error in Lipschitz Certificate [AUDITOR]
Theorem 7 statement says `C(e;D') ≥ C(e;D) - L_BIC·δ` but the proof derives `C(e;D') ≥ C(e;D) - 2·L_BIC·δ`. The stability condition is off by a factor of 2. Fixable but erodes trust in theoretical rigor.

### F6. Intractable FIND-ALL-CYCLES Subroutine [AUDITOR]
The topological repair algorithm in crossover calls FIND-ALL-CYCLES, which has exponential worst-case complexity. For moderately dense graphs, the number of simple cycles can be astronomical. Needs replacement with bounded cycle-breaking.

### F7. Independent Edge Perturbation Assumption [SKEPTIC + AUDITOR]
Theorem 8 (path certificate) assumes edge perturbations are independent, which is mathematically wrong. Edges sharing a node have correlated perturbations under bootstrap. This makes path certificates anti-conservative — the wrong direction for a "robustness" guarantee.

### F8. Scalability Claims Self-Contradictory [ALL THREE EVALUATORS]
The paper claims n≤50 but by the time all approximations are applied (PCA for n>20 → destroys MEC separation, skeleton restriction for n>30 → introduces error, CVT for n>20 → reduces resolution), the method has degraded theoretical guarantees. Honest scope is n≤20 with full guarantees.

### F9. No Downstream Task Evaluation [AUDITOR]
The proposal evaluates structural metrics (SHD, F1, coverage) but never evaluates whether the archive helps practitioners do anything — no causal effect estimation, no interventional prediction, no decision-making evaluation.

---

## Strongest Salvageable Element

**The behavioral descriptor space design** — organizing DAGs by MI profiles + CI signatures + equivalence class features rather than by raw structure.

This is genuinely novel because:
1. No prior work proposes a systematic taxonomy of DAGs by information-theoretic behavioral properties for diversity enumeration
2. The descriptor space is algorithm-agnostic — it works with MCMC, GES, random search, or MAP-Elites
3. "What dimensions of variation exist among high-scoring DAGs?" is scientifically interesting and unanswered
4. It connects to interpretability: regions of descriptor space may correspond to qualitatively different causal stories

---

## Mandatory Modifications for CONTINUE

### M1. Hard Pivot on Framing
The primary contribution becomes **"Characterizing the DAG Fitness Landscape via Behavioral Descriptors."** MAP-Elites is demoted to one of several search methods studied. The paper is NOT about MAP-Elites for causal discovery.

### M2. Drop All Theoretical Claims
Theorems 1-9 are removed entirely. The paper is empirical/systems. If theory is retained, it must be limited to genuinely non-trivial results — no tautologies, no circular proofs, no restatements.

### M3. Include Order MCMC as Co-Method
The experimental design compares how GES, random restarts, Order MCMC posterior sampling, and MAP-Elites each explore the descriptor space. The research question becomes "how do different algorithms traverse the DAG landscape?" — not "is MAP-Elites better?"

### M4. Reduce Scope to Laptop-Feasible
- Maximum n=20 nodes
- At most 5 benchmark datasets
- At most 10 replicates
- Target total compute: ≤100 CPU-hours (~4 days on modern laptop)
- Fix crossover repair to use polynomial-time feedback arc set, not FIND-ALL-CYCLES

### M5. Define Success Criteria Before Experiments
Before running experiments, specify what findings would constitute a publishable result:
- "MAP-Elites covers ≥20% more descriptor cells than MCMC"
- "Descriptor clusters correspond to meaningfully different causal explanations"
- "The landscape has consistent topological structure that predicts algorithm performance"

If none of these hold, the result is still publishable as a negative finding ("MCMC suffices") at a workshop.

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Landscape analysis yields no surprises | 40% | High — weak publication | Define success criteria upfront; target workshop venues |
| Order MCMC dominates on coverage too | 30% | Medium — "MCMC suffices" is itself a finding | Frame as legitimate empirical result |
| Scope creep back to original framing | 25% | High — wastes time on dead end | Enforce modified scope; weekly check against plan |
| Computational budget exceeds laptop | 20% | Low — can use cloud credits | Pre-compute cost estimates; hard budget caps |

---

## Recommended Next Steps

1. Write 1-page revised problem statement incorporating the pivot
2. Implement descriptor computation for one dataset (proof of concept)
3. Run Order MCMC (via BiDAG) on one dataset; visualize descriptor space coverage
4. Compare with MAP-Elites coverage on same dataset
5. Evaluate whether comparison reveals anything interesting before committing to full experiments

---

## Team Signoff

| Role | Verdict | Score | Key Concern |
|------|---------|-------|-------------|
| Independent Auditor | CONTINUE-WITH-MODS | 5.0 | Circular theory, math errors, missing baseline |
| Fail-Fast Skeptic | ABANDON | 3.6 | Order MCMC dominates; all theorems vacuous |
| Scavenging Synthesizer | CONTINUE-WITH-MODS | 5.6 | Salvageable descriptor contribution; needs major pivot |
| **Cross-Critique Consensus** | **CONTINUE-WITH-MODS** | **4.6** | **Original framing dead; pivot to landscape characterization** |

**The original proposal — MAP-Elites as the contribution with theoretical guarantees — should be abandoned.** The CONTINUE verdict applies exclusively to the reframed version centered on behavioral descriptor space design with empirical landscape analysis using multiple search algorithms.
