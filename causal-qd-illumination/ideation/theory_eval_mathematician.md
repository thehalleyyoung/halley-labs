# Theory Evaluation — Deep Mathematician

**Proposal:** CausalQD — Quality-Diversity Illumination for Causal Structure Discovery  
**Evaluator:** Deep Mathematician (team-verified evaluation)  
**Method:** 3-agent adversarial team (Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer) with cross-evaluation critique round and independent verifier signoff.  
**Date:** 2026-03-02  
**Materials:** theory/approach.json (63KB), theory/paper.tex (91KB), 6 definitions, 9 theorems, 7 algorithms, 5 assumptions

---

## SCORES

| Pillar | Score | Assessment |
|--------|:-----:|-----------|
| **Extreme Value** | **5/10** | Real but bounded. The behavioral descriptor space (B_struct × B_info × B_equiv) is genuinely novel — no prior work organizes causal DAGs by information-theoretic behavioral fingerprints. But the core problem ("find diverse plausible causal structures") is already solved by Order MCMC and Partition MCMC with superior probabilistic semantics. The contribution is a recombination novelty (QD + causal discovery), not a conceptual breakthrough. Scalability ceiling (n≤50), linear Gaussian assumption, and causal sufficiency requirement cap practical impact. |
| **Genuine Software Difficulty** | **5/10** | Moderate engineering challenge, thin mathematical novelty. Acyclicity-preserving crossover with topological repair and incremental MI profile computation (O(n³·k_max³)) are genuine engineering tasks. But ~60% of the system is standard component integration: MAP-Elites is a dictionary, BIC scoring exists in causal-learn, bootstrap is a loop. No theorem in the paper requires genuinely novel mathematical technique. |
| **Best-Paper Potential** | **3/10** | No. All 9 theorems were independently verified as either trivial, circular, known, or containing errors. The theoretical presentation (volume-over-substance) would draw justified criticism at any strong venue. Zero implemented code means zero empirical validation. Best case: solid accept at a focused venue (UAI, CLeaR), not a spotlight/oral at a flagship. |
| **Laptop-CPU Feasibility** | **5/10** | Core MAP-Elites algorithm is laptop-feasible for n≤30 (~10 minutes for n=20). But bootstrap robustness certificates — a headline contribution — require running full CausalQD 1000×, costing ~33 hours for n=20 alone. The proposal itself requests a "96-core server" and budgets 3,200 CPU-hours. Cannot score 7/10 while ignoring a headline contribution's infeasibility. |
| **Feasibility** | **5/10** | Buildable in principle (algorithms well-specified, dependencies mature), but: impl_loc = 0 (nothing exists), the crossover repair algorithm (FIND-ALL-CYCLES) has exponential worst-case complexity, ergodicity "verification" is ill-defined, and the experimental plan (6 RQs, 3200 CPU-hours) is overambitious for the contribution size. |

**Composite: 23/50 (46%)**

---

## LOAD-BEARING MATH ASSESSMENT

The central question for a deep mathematician: **is the math load-bearing, or ornamental?**

### Theorem-by-Theorem Verdict

| Theorem | Classification | Justification |
|---------|---------------|---------------|
| T1: Topological Mutation Preservation | **Trivial** | "Forward edges in a topological order can't create cycles." This is a loop invariant, not a theorem. Textbook graph theory exercise. |
| T2: Crossover Acyclicity | **Engineering, not math** | Edge selection via π_m(Xi) < π_m(Xj) trivially guarantees acyclicity. The interesting part (topological repair for cyclic unions) is algorithmic, not rigorously proven. |
| T3: Archive Coverage (Ergodicity) | **Circular** | Assumes ergodicity — the very thing that would make the theorem non-trivial. Given ergodicity, it's a standard coupon-collector argument. The ergodicity assumption is likely *false* for archive-biased MAP-Elites (selection concentrates on well-populated regions, creating absorbing-like behavior). |
| T4: Supermartingale Convergence | **Tautological** | The archive quality is monotonically non-decreasing by definition of elitist replacement. Doob's theorem applies trivially. "Convergence to zero" relies on the unproven ergodicity from T3. Dressing this in martingale language adds formality but zero insight. |
| T5: MEC Separation (Full) | **Known result restated** | Direct restatement of Verma & Pearl (1990): "same skeleton + same v-structures ↔ Markov equivalent." The proof literally says "by the MEC characterization theorem." Zero novel content. |
| T5b: MEC Separation (PCA) | **Obvious** | "PCA projection loses information." This is the definition of dimensionality reduction. The separation bound is a triangle inequality. |
| T7: Lipschitz Certificate | **Only load-bearing theorem; has errors** | Standard perturbation analysis of BIC, but requires actual derivation. However: (a) statement claims C(e;D') ≥ C(e;D) − L·δ but proof derives 2L·δ (factor-of-2 error), (b) L_BIC = O(1/N) obscures an O(√n) factor from the trace term (should be O(√n/N)), (c) assumes well-conditioned covariance, which fails for near-singular Σ (precisely the hard cases). |
| T8: Path Certificate | **Incorrect** | Assumes independent edge perturbations. Edges sharing a node have correlated BIC responses because they share sufficient statistics. This makes path certificates anti-conservative. The composition is elementary probability even if the assumption held. |
| T9: Boltzmann Stability | **Definition, not theorem** | "As β→∞, Boltzmann distribution concentrates on the mode" is a definitional property. The "proof" computes textbook limits of softmax. |

### Summary

**No theorem in this paper is both correct and non-trivial.** Theorem 7 comes closest but has a factor-of-2 error and unstated regularity conditions. The interesting mathematical questions — geometry of BIC-constant surfaces in DAG space, topology of the CPDAG lattice restricted to high-scoring regions, mixing time analysis for the archive-biased Markov chain — are never asked, let alone answered.

The math is **overwhelmingly ornamental**. Strip away all 9 theorems and you have: a MAP-Elites implementation with domain-specific mutation operators and a BIC-based scoring function. That's a solid engineering contribution, not a theory paper.

---

## FATAL FLAWS

| # | Flaw | Severity | Fixable? |
|---|------|----------|----------|
| 1 | **Bootstrap certificates computationally infeasible on laptop** | High | Yes — drop as contribution, use analytical Lipschitz bound only |
| 2 | **Ergodicity assumption ungrounded and likely false** | Medium-High | Partially — downgrade to proposition, add empirical diagnostics |
| 3 | **Theorem 7 factor-of-2 error** | Medium | Yes — update statement to 2L·δ |
| 4 | **Theoretical padding: 9 theorems, ≤1 non-trivial** | Medium | Yes — strip to 3-4 theorems |
| 5 | **Scope overextension** | Medium | Yes — focus on 3 contributions |
| 6 | **Missing Order MCMC baseline** | High | Yes — must include for publishability |
| 7 | **Crossover repair (FIND-ALL-CYCLES) is exponential** | Medium | Yes — replace with polynomial FAS heuristic |

No single fatal flaw, but flaws 1 + 2 + 4 compound: the paper's theoretical and computational claims are its weakest elements, yet they consume the most space. The strongest element (behavioral descriptor design) is underemphasized.

---

## WHAT IS GENUINELY NOVEL AND VALUABLE

**One idea worth stealing:** MI profiles conditioned on parent sets as behavioral descriptors for organizing causal model diversity. This specific technical choice — computing I(Xi; Xj | Pa_G(Xi) ∪ Pa_G(Xj)) for all pairs and using the resulting vector as a continuous signature — has no direct precedent in either the QD or causal discovery literatures. Even without MAP-Elites, this descriptor could be used for:
- Clustering DAGs from Bayesian posterior samples
- Visualizing the landscape of causal uncertainty
- Defining meaningful distances between causal models
- Feature engineering for causal model selection

**Solid engineering:** The acyclicity-preserving operators (topological ordering invariant + order-based crossover + subgraph transplant via Markov blankets) are a reusable library contribution for any evolutionary search over DAGs.

**Unexploited synergy:** Descriptor variance across bootstraps → decomposed uncertainty quantification. High variance in B_info means information-flow is unstable; high variance in B_equiv means MEC assignment is fragile. This connection is more valuable than the current aggregate certificates.

---

## VERDICT: CONDITIONAL CONTINUE

### Conditions (ALL required):

1. **Drop bootstrap certificates** as a contribution. Mention analytical Lipschitz bound as a useful byproduct. This eliminates the computational infeasibility and raises laptop-CPU to a genuine 7/10.

2. **Strip theorems to 3-4 max.** Keep: acyclicity preservation (compress T1-T2 into one lemma), MEC separation (T5), and the corrected Lipschitz certificate (T7 with 2L·δ). Cut T3, T4, T5b, T8, T9 entirely.

3. **Reframe ergodicity.** Present coverage as a "proposition under stated assumptions" with explicit discussion of when the assumption fails (archive-biased selection, multimodal score landscapes). Add multi-seed convergence diagnostics to experiments.

4. **Elevate behavioral descriptors as centerpiece.** The MI-profiles-conditioned-on-parent-sets design should be the main contribution, not a subsection.

5. **Target UAI or CLeaR**, not NeurIPS/ICML. The contribution matches a focused venue.

6. **Include Order MCMC as primary baseline.** Without this, the paper is un-publishable — reviewers will immediately request it.

7. **Fix crossover repair algorithm.** Replace FIND-ALL-CYCLES (exponential) with polynomial-time minimum feedback arc set heuristic.

8. **Fix Theorem 7.** Statement must use 2·L_BIC·δ, explicitly state well-conditioned covariance assumption, report L_BIC = O(√n/N).

9. **Reduce honest scope to n ≤ 20** for full guarantees (PCA and skeleton restrictions degrade all theoretical properties for larger n).

10. **Define success criteria before implementation.** Pre-specify: what coverage gap between MAP-Elites and MCMC constitutes a meaningful finding? What descriptor clustering structure would be publishable?

### Abandonment Triggers:
- Feasibility experiment on Sachs (n=11) shows <10% diversity gain over 30-restart GES
- Behavioral descriptors fail to separate known-different MECs on synthetic data
- Order MCMC with same descriptors achieves comparable archive diversity
- n=20 takes >4 hours per run on available hardware

### If all conditions met:
Acceptance probability at UAI/CLeaR: **25-40%**. The paper becomes a clean, scoped contribution with a genuine novel idea (behavioral descriptors), solid engineering (topological operators), and useful empirical results.

### If conditions NOT met:
**ABANDON.** The current version has too many undercooked components, and the theoretical presentation will draw justified criticism that overshadows the genuine contribution.

---

## TEAM PROCESS SUMMARY

| Phase | Key Finding |
|-------|------------|
| **Independent Auditor** | No fatal flaws. Behavioral descriptors are strongest element. Factor-of-2 error in T7. Scores slightly optimistic. |
| **Fail-Fast Skeptic** | 9/10 theorems are trivial/circular/known. Bootstrap certificates computationally catastrophic. Ergodicity near-fatal. |
| **Scavenging Synthesizer** | Behavioral descriptor is "the one idea worth stealing." Strip to 3 contributions, target focused venue. |
| **Adversarial Critique** | Skeptic wins on laptop-CPU (bootstrap infeasible). Synthesizer wins on strategic direction. Ergodicity is serious but addressable. |
| **Verification Signoff** | APPROVED. "No theorem is both correct and non-trivial" confirmed. 5 missing conditions appended. Evaluation slightly lenient on Extreme Value and Feasibility. |
