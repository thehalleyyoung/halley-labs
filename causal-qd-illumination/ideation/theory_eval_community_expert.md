# Community Expert Evaluation: CausalQD (proposal_00)

## Proposal
**CausalQD: Quality-Diversity Illumination for Causal Structure Discovery**

A MAP-Elites engine for causal DAG discovery with information-theoretic behavioral descriptors, producing diverse archives of plausible causal models and robustness certificates.

## Evaluation Method
Three-expert adversarial team review with cross-critique synthesis:
- **Independent Auditor**: Evidence-based scoring with challenge testing
- **Fail-Fast Skeptic**: Aggressive rejection of under-supported claims
- **Scavenging Synthesizer**: Hidden strengths and reframing opportunities

---

## Scores

| Criterion | Score | Rationale |
|-----------|-------|-----------|
| **Extreme Value** | 4/10 | The structured archive concept is genuinely novel — it provides a *map* of causal hypotheses organized by behavioral properties, unlike Bayesian posterior samples which are unstructured. However, the proposal never articulates a concrete decision scenario where this map enables an action that Bayesian model averaging (Order MCMC) cannot. The causal sufficiency assumption (no hidden confounders) rules out the majority of real-world applications in pharma, epidemiology, and tech. Practitioners already have PC, GES, NOTEARS, causal-learn, bnlearn — CausalQD offers "competitive accuracy" (not superior) with "superior diversity" whose utility is unproven. Pearl would ask why we need diverse DAGs rather than do-calculus; Spirtes would note CPDAG already captures the MEC; Peters would point out that under identifiability (LiNGAM, nonlinear), diversity is a bug not a feature. |
| **Genuine Software Difficulty** | 6/10 | No individual component is algorithmically novel — topological ordering, partial correlations, CVT tessellation, MAP-Elites are all well-understood. But integration creates genuine difficulty: acyclicity-preserving crossover with topological repair must compose correctly with MI-profile computation, CPDAG/Nauty canonical hashing, and PCA-compressed archive management. Silent bugs (cycle introduction, descriptor miscomputation) are hard to detect with no test oracle. A competent implementation requires ~4,000–6,000 lines of Python with careful numerical stability handling. Prototype in 2–3 weeks; correct + efficient in 3–4 months. |
| **Best-Paper Potential** | 2/10 | The theoretical contributions do not meet top-venue standards. Theorems 1–2 (acyclicity preservation) are definitional consequences of topological orderings. Theorem 3 (coverage) is conditional on empirically-unverifiable ergodicity. Theorem 4 (supermartingale convergence) is trivially true for any elitist archive. Theorem 5a (MEC separation) restates Verma & Pearl 1990. **Most damaging**: Theorems 7–9 (Lipschitz certificates) are mathematically vacuous — the Lipschitz constant L_BIC = O(1/N) vs certificate C(e) = O(N) means every edge with detectable effect passes, providing zero discrimination. The experimental plan is unfocused (6 RQs, 9 baselines, 6 ablations) and missing the most relevant baseline (Order MCMC). Even restructured, this is a workshop/short paper contribution, not a top-venue oral. |
| **Laptop-CPU Feasibility** | 6/10 | Core MAP-Elites loop is CPU-friendly and laptop-feasible for n ≤ 30 (~2–3 hours). No GPU, no human annotation, no human studies required. All dependencies (NumPy, SciPy, NetworkX, causal-learn, Numba) are standard. **However**, the bootstrap certificate procedure (rerunning full CausalQD 200–1000× on resampled data) is infeasible on a laptop — 200 reruns at n=30 ≈ 67 hours per configuration. The experimental plan explicitly calls for a "96-core server with 512GB RAM" and 3,200 CPU-hours. Parametric bootstrap (re-evaluate existing archive on perturbed data) would fix this but is not in the current proposal. |
| **Feasibility** | 5/10 | Implementable with existing tools (pyribs, NetworkX, causal-learn). All 8 algorithms have detailed pseudocode. Key risks: (1) Ergodicity may fail in practice — archive may get stuck in a DAG subspace, voiding coverage claims. (2) PCA compression may destroy MEC separation (Theorem 5b explicitly acknowledges this). (3) MI profiles assume linear Gaussian SCMs — meaningless for nonlinear data (gene expression, climate, finance). (4) Bootstrap certificates as specified are computationally infeasible. Results will likely show "more diverse DAGs found" without demonstrating why that matters for any downstream decision. |

---

## Fatal Flaws

### Flaw 1: Robustness Certificates Are Mathematically Vacuous (CONFIRMED — FATAL TO CLAIMED CONTRIBUTION)
The Lipschitz bound L_BIC = O(1/N) makes Theorems 7–9 vacuously satisfied. For a true edge with effect β = 0.15 at N = 1000: C(e) ≈ 38, threshold ≈ 0.01. Ratio = 3,800×. Every edge with detectable BIC evidence is automatically "certified." The certificate provides zero discrimination beyond what BIC already provides. Three theorems and a claimed core contribution are invalidated. The bootstrap stability analysis underneath is standard (Efron & Tibshirani 1993; Meinshausen & Bühlmann 2010) and not novel.

### Flaw 2: No Articulated Use Case (CRITICAL)
The proposal never explains what a practitioner *does* with a QD archive that they cannot do with Bayesian posterior inference. "Explore alternative explanations" is a slogan, not a decision procedure. Until there is a concrete scenario — "given this archive, I would make decision X instead of decision Y" — the value proposition is speculative.

### Flaw 3: Bayesian Alternatives Dominate on Claimed Axes (SIGNIFICANT)
Order MCMC provides calibrated posterior probabilities over DAG structures with proven mixing guarantees. The archive gives an unweighted set without probabilistic interpretation. Boltzmann weighting (Theorem 9) is an ad-hoc approximation of what Bayesian inference does exactly. The proposal omits Order MCMC as a baseline, likely because it would dominate on every meaningful diversity metric.

### Flaw 4: Theoretical Contributions Are Weak (SIGNIFICANT)
Of 9 theorems: ~4 are trivially true for any elitist algorithm, ~2 restate known results, ~1 is conditional on an unverifiable assumption, ~1 admits it undermines the practical story (PCA destroys MEC separation), and ~1 is vacuous (certificates). No theorem provides a genuinely novel insight about causal discovery.

### Flaw 5: Linear Gaussian Restriction (MODERATE)
MI profiles assume linear Gaussian SCMs (explicitly stated in approach.json). This excludes most real-world applications (gene regulation, neural circuits, climate systems — all nonlinear). The proposed "real-world" experiments are in precisely the domains where this assumption fails.

---

## What Survives the Critique

Despite significant flaws, the team identified genuine value in specific components:

1. **Acyclicity-preserving genetic operators** — The topological-ordering-based mutation/crossover operators are well-designed and useful for any evolutionary optimization over DAGs (Bayesian network learning, neural architecture search). Standalone contribution.

2. **Behavioral descriptor space** — The three-component descriptor (structural + information-theoretic + equivalence class) provides a principled similarity metric for causal structures. Useful for DAG clustering, method comparison, and meta-analysis independent of MAP-Elites.

3. **Identifiability landscape visualization** — If the archive reveals surprising structure in the space of high-scoring DAGs (clusters by MEC, ridges, valleys), this is a finding about causal identifiability itself, independent of the algorithm. This is the make-or-break experiment.

4. **Embarrassingly parallel architecture** — MAP-Elites batches are independent. MCMC is inherently sequential. For large problems where MCMC mixing is poor, CausalQD's parallelism is a genuine engineering advantage.

5. **Sensitivity analysis reframing** — Repositioned as "systematic sensitivity analysis for causal discovery" rather than "QD optimization for DAGs," the proposal avoids the "evolutionary algorithm applied to X" stigma and connects to established statistical methodology (Rosenbaum, Imbens).

---

## Strongest Possible Version of This Idea

If restructured around the surviving components:

1. **Lead with sensitivity analysis, not QD optimization.** The archive is a sensitivity tool, not a replacement for Bayesian inference.
2. **Strip Lipschitz certificates.** Replace with honest bootstrap stability scores using parametric bootstrap (not full re-optimization). Drop Theorems 7–9.
3. **Add Order MCMC baseline.** Head-to-head on: diversity of recovered MECs, wall-clock time, identifiability boundary detection.
4. **Run the identifiability landscape experiment** on Sachs (11 nodes) as a go/no-go gate. If the archive fails to separate known equivalence classes: abandon.
5. **Concrete use case.** Show that archive-based analysis leads to a different (better) decision than single-DAG methods on one real-world problem.
6. **Honest scope statement.** "CausalQD is a structured sensitivity analysis tool for linear Gaussian causal models with up to 30 variables, providing organized diversity exploration with bootstrap stability scores. It complements — but does not replace — Bayesian posterior inference."

---

## VERDICT: CONTINUE (Conditional)

**Conditions for continuation:**
1. Run identifiability landscape experiment on Sachs/ALARM within 2 weeks as go/no-go gate
2. Strip Lipschitz certificate theorems (7–9) — replace with empirical bootstrap
3. Add Order MCMC as primary diversity baseline
4. Reframe as sensitivity analysis, not QD optimization
5. Scope to linear Gaussian, n ≤ 30, with honest scalability benchmarks

**Kill condition:** If the identifiability landscape experiment fails to reveal known structural features (known unidentifiable edges, known equivalence classes), or if Order MCMC dominates on diversity metrics at equal compute budget → ABANDON.

**Confidence:** 60%. The proposal has a viable core buried under overclaimed theory. Success depends entirely on whether the identifiability landscape experiment reveals something genuinely interesting about causal structure diversity. Without surprising empirical findings, this is incremental engineering — publishable at a workshop but not at a top venue.

---

*Evaluated by: Community Expert (causal-discovery) with 3-expert adversarial team review*
*Theory files reviewed: approach.json (63KB), paper.tex (91KB)*
*Date: 2026-03-02*
