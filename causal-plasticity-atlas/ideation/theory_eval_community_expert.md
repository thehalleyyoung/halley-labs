# Community Expert Evaluation: Causal-Plasticity Atlas (proposal_00)

**Evaluator**: Domain Expert in Causal Discovery (Community Perspective)
**Stage**: Post-Theory Verification
**Method**: Three-role adversarial team (Independent Auditor + Fail-Fast Skeptic + Scavenging Synthesizer) with mediated synthesis and independent verification signoff

---

## Proposal Summary

A causal-plasticity atlas engine that, given heterogeneous multi-context observational datasets, uses curiosity-driven quality-diversity search to systematically map which causal mechanisms are invariant versus plastic versus emergent across contexts — producing a navigable archive of mechanism-change patterns via novel context-aware DAG alignment operators, information-theoretic plasticity descriptors, automated tipping-point detection, and robustness certificates over mechanism stability classes.

**Theory materials reviewed**: approach.json (166KB, 14 definitions, 8 theorems, 5 algorithms, 12 falsifiable claims, 6 assumptions, full risk analysis) + paper.tex (186KB, ~4100 lines, 9 theorems with proofs, 14 definitions, 5 algorithms, 6 synthetic + 4 semi-synthetic benchmarks, 8 baselines, 7 ablations, appendices with supporting lemmas and extended complexity analysis).

---

## PILLAR SCORES

### 1. Extreme Value: 6/10

**The problem is real but the demonstrated value is zero.** ICP's binary invariant/non-invariant classification is a genuine limitation — the paper correctly identifies five gaps (binary classification, fixed variable sets, no mechanism-level metric, no robustness certificates, no systematic exploration). The four-dimensional plasticity descriptor (parametric, structural, emergence, context-sensitivity) addresses this gap with a principled taxonomy that would benefit multi-site genomics, cross-country economics, and clinical subgroup analysis.

However, three issues limit the score:
- **No real-data validation.** 12 falsifiable claims, all on synthetic/semi-synthetic data. The value remains theoretical.
- **CD-NOD already gets you 80% of the way.** CD-NOD's changed/unchanged module detection + standard graph comparison covers most practical needs. The marginal value of the four-class taxonomy over {changed, unchanged} × {edge present, absent} is undemonstrated.
- **The QD search and atlas framing add complexity without proportional value.** For practical K and n, exhaustive pairwise comparison is tractable. The QD archive is a visualization convenience, not a computational necessity.

**What would raise this to 8+**: A real-world case study (e.g., GTEx tissues, Sachs perturbations) where the four-class taxonomy produces an actionable insight that binary ICP misses.

### 2. Genuine Software Difficulty: 7/10

**The integration challenge is genuine; the individual components are known.** The five-algorithm pipeline (CADA → PDC → CD-QD → TPD → RCG) requires:
- **CADA alignment**: Six-phase pipeline with NP-hardness circumvented by Markov-blanket pruning and CI-fingerprint scoring. The FPT result (O(n²d² + u³) for bounded-degree DAGs) requires careful combinatorial optimization wrapped in statistical inference.
- **Certificate generation**: Stability selection (50% subsample × 100 rounds) with three-level parallelism (per-context, per-pair, per-mechanism). The computational budget analysis shows this consuming 47% of wall-clock time.
- **Numerical stability**: √JSD computation for near-singular covariance matrices requires Tikhonov regularization. Bit-identical reproducibility (FC10) demands careful floating-point management.

Core algorithms (Hungarian matching, PELT, GES, MAP-Elites, bootstrap) have reference implementations. The difficulty is in composition, not invention — but that composition is non-trivial.

### 3. Best-Paper Potential: 5/10

**Strong ingredients, too sprawling for a crisp narrative.**

Strengths:
- 9 theorems with full proofs and clean metric-space foundations
- 12 pre-registered falsifiable claims with explicit falsification conditions (exemplary methodology)
- Novel problem formulation: the space of mechanism-change patterns as a geometric object
- Compelling narrative potential: "We give every causal mechanism a stability passport"

Weaknesses:
- **Theorems are competent but not groundbreaking.** T1 (metric properties) is a routine verification. T2 (classification correctness) has circular conditioning — requires the signal to be far from decision boundaries. T6 (atlas completeness) is practically vacuous (convergence bound exponential in genome space dimension). Only T5 (tipping-point consistency, O_p(1/K) localization) and T8 (structural stability under DAG errors) would impress reviewers.
- **Gaussian linear SEM restriction** limits generality claims. No nonlinear evaluation.
- **Missing CD-NOD baseline** — the closest competitor is conspicuously absent.
- **Paper falls between two stools**: too many moving parts for a theory contribution, too little empirical grounding for a systems contribution.

**Best-paper path**: Focus ruthlessly on plasticity descriptors + certificates (Theorems 2, 3, 4, 8). Drop QD search. Add real data + CD-NOD baseline. Target UAI/AISTATS. The "passport" narrative is strong enough if the scope matches.

### 4. Laptop-CPU Feasibility & No-Humans: 6/10

**Feasible for medium-scale problems; the headline scalability claim will likely fail.**

The pipeline is fully automated (no human judgment in evaluation loop), CPU-only (NumPy/SciPy/scikit-learn/causal-learn), and runs on standard hardware (≤8 cores, ≤32 GB RAM). Concrete wall-clock times from the paper's own budget analysis:

| Problem size | Time | Verdict |
|-------------|------|---------|
| p=20, K=5, n=500 | ~5 min | ✅ Easily feasible |
| p=50, K=10, n=500 | ~30 min | ✅ Feasible |
| p=100, K=5, n=500 | ~120 min | ⚠️ At risk (FC7) |
| p=100, K=20, n=500 | ~8+ hours | ❌ Certificate phase alone |

The binding constraint is certificate generation: O(n_inv × 100 × K × T_DAG). For p=100 with K=20, stability selection requires ~100,000 DAG re-estimations. The proposal's own risk analysis rates this HIGH severity, HIGH likelihood.

**No-humans**: Fully satisfied. All benchmarks use public data (Sachs, Asia, Alarm, Insurance from bnlearn). No annotation or human studies required.

**No-GPUs**: Fully satisfied. Pure CPU computation throughout.

### 5. Feasibility: 6/10

**Achievable with scope reduction; full scope is multi-year.**

Of 12 falsifiable claims:
- **High confidence (5-6 claims)**: FC1 (F1 ≥ 0.90 on FSVP), FC8 (Sachs 80%), FC10 (reproducibility), FC5 (QD coverage), FC6 (JSD ablation superiority)
- **Achievable with effort (3 claims)**: FC2 (emergence detection), FC3 (tipping-point localization), FC9 (SHD ablation)
- **At risk (3-4 claims)**: FC4 (certificate calibration ECE ≤ 0.04), FC7 (p=100 ≤120 min), FC11 (latent confounder F1 drop ≤ 0.15), FC12 (Theorem 8 compliance 95%)

Critical credibility gap: the theoretical sample complexity requires n_min ≈ 15,000+ per context (from Theorem 4), but the evaluation uses n = 500–1,000. The formal guarantees do not apply to the experimental regime. This is standard practice in statistical learning theory but should be transparently acknowledged.

The full proposal (5 algorithms, 12 claims, 7 ablations, 10 implementation modules) is ambitious for a single project. The focused "Plasticity Certificates" version (PDC + classification + certificates + Theorems 2,3,4,8) is a 3–6 month project.

---

## FATAL FLAW ANALYSIS

### Claimed Fatal Flaws (5 identified by team)

| # | Flaw | Severity | Truly Fatal? | Mitigation |
|---|------|----------|--------------|------------|
| 1 | Theory-practice sample size gap (n_min ≈ 15K vs. eval at n=500) | HIGH | **No** | Acknowledge explicitly + empirical convergence curves |
| 2 | DAG estimation errors dominate downstream classification | HIGH | **No** | Oracle-DAG baseline isolates; Theorem 8 quantifies; framework is modular |
| 3 | Atlas completeness theorem (T6) is practically vacuous | MODERATE | **No** | Drop QD search (all evaluators recommend this) |
| 4 | Zero real-world validation | HIGH | **No** | Add Sachs or GTEx experiment (2–4 weeks of work) |
| 5 | Over-engineering: 5 algorithms when 2 suffice for 80% of value | MODERATE | **No** | Restructure as modular core + optional extensions |

**Compound risk**: Flaws #1 + #2 + #4 together create a credibility risk — formal guarantees that don't apply to the evaluation regime, applied to synthetic-only data, on top of unreliable DAG estimates. This compound risk is the closest thing to a fatal flaw. Mitigation: radical transparency about limitations + empirical utility demonstration.

**No genuinely fatal flaws identified.** All issues are manageable with specific, actionable mitigations.

---

## COMPONENT VALUE ASSESSMENT

| Component | Novelty | Value | Keep/Drop |
|-----------|---------|-------|-----------|
| √JSD mechanism distance | 4/10 | 8/10 | **KEEP** (foundation) |
| 4D plasticity descriptors | 7/10 | 9/10 | **KEEP** (crown jewel) |
| Robustness certificates | 8/10 | 9/10 | **KEEP** (killer feature) |
| Structural stability (T8) | 7/10 | 8/10 | **KEEP** (practical essential) |
| MCCM framework | 6/10 | 7/10 | **KEEP** (definitional) |
| Tipping-point detection | 5/10 | 6/10 | OPTIONAL (nice-to-have) |
| Context-aware DAG alignment | 7/10 | 5/10 | OPTIONAL (niche use case) |
| QD search over mechanism space | 5/10 | 4/10 | **DROP** (not load-bearing) |
| Evaluation framework | 7/10 | 7/10 | **KEEP** (exemplary design) |

**The strongest standalone contribution**: The plasticity descriptor framework with robustness certificates. A focused "Beyond Binary Invariance" paper with just this subset would be publishable at UAI/AISTATS.

---

## VERDICT

### CONTINUE — with focused scope

**Consensus**: 30/50 (all three evaluators independently reached CONTINUE)

**Mandatory conditions**:
1. **Reduce scope** to plasticity descriptors + certificates + Theorems 2,3,4,8. Drop QD search from core.
2. **Add ≥1 real-data experiment** (Sachs flow cytometry with actual perturbation data, or GTEx multi-tissue).
3. **Add CD-NOD as baseline** — the closest competitor is conspicuously absent.

**Strongly recommended**:
4. Weaken FC7 to p=50 single-core / p=100 with 8-core parallelism.
5. Add one nonlinear SEM generator to address Gaussian restriction.
6. Include empirical convergence curves (F1 vs. n) to bridge theory-practice gap.

**Reframing**: Drop "Causal-Plasticity Atlas" branding (implies exhaustive QD exploration). Adopt **"Plasticity Certificates for Multi-Context Causal Mechanisms"** — focused, defensible, actionable.

**One-line pitch**: *"We give every causal mechanism a stability passport: a four-dimensional fingerprint of how it changes across contexts, plus a formal certificate guaranteeing the classification is correct."*

**Target venue**: UAI or AISTATS (primary). JMLR if comprehensive evaluation warrants journal-length.

**Timeline**: 3–6 months for focused version. QD search, CADA, and extended TPD as follow-up work.

**Competitive window**: 12–18 months before Peters/Bühlmann (ETH) or Mooij (Amsterdam) groups produce something in this space.

---

## SCORING SUMMARY

| Dimension | Score |
|-----------|-------|
| Extreme Value | **6/10** |
| Genuine Software Difficulty | **7/10** |
| Best-Paper Potential | **5/10** |
| Laptop-CPU Feasibility & No-Humans | **6/10** |
| Feasibility | **6/10** |
| **TOTAL** | **30/50** |
| Fatal Flaws | **None (5 manageable risks)** |
| **VERDICT** | **CONTINUE** |

---

*Evaluation produced by three-role adversarial team with independent verification signoff. Verification status: APPROVED.*
