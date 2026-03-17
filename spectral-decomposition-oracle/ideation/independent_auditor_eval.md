# Independent Auditor Evaluation — proposal_00 (spectral-decomposition-oracle)

**Evaluator:** Independent Auditor (evidence-based scoring, challenge testing)
**Date:** 2025-07-22
**Stage:** Post-theory verification gate
**Method:** Line-by-line evidence review of 8 artifacts, independent scoring, challenge against 3 prior evaluations

---

## 0. Executive Summary

**Scores: V4 / D4 / BP2 / L6 / F3 — Composite 3.8/10**

**Verdict: CONDITIONAL CONTINUE** — barely. Gated on G0 (48h) and G1 (Week 2). Honest assessment: this is a marginal project whose continuation is justified only by (a) cheap kill gates and (b) sunk theory investment.

State.json shows theory_bytes=0 — confirmed measurement bug; 288KB of theory artifacts exist in `theory/`. However, impl_loc=0 is **accurate and damning**: after producing 288KB of analysis documents, zero lines of implementation code exist. The 50-line G0 test that determines the project's viability was never written. This is the clearest evidence of analysis paralysis in any project I've reviewed.

---

## 1. Score Table

| Pillar | Score | Binding Ceiling | Justification |
|--------|-------|-----------------|---------------|
| **V — Extreme Value** | **4** | 5 | Census is genuinely valuable (V3–4 floor). Spectral features add conditional V+1 if they work — but they're untested. "Extreme and obvious" requires broad demand; ~50–100 decomposition research groups is not broad. Commercial solvers (CPLEX, Gurobi) already ship automatic decomposition heuristics. |
| **D — Genuine Difficulty** | **4** | 4 | Matches ceiling. Hypergraph Laplacian construction for non-square matrices is genuinely hard (~6.5K LoC). Everything else — sklearn RF/XGBoost, SCIP/GCG wrappers, census pipeline — is competent engineering. A PhD student builds this in 6–8 weeks. Post-descoping (26.5K LoC) is honest. |
| **BP — Best-Paper** | **2** | 3 | Below ceiling. The ceiling of 3 assumed L3 proof gaps would be fixed and red-team threshold met. Neither happened: L3 has 2 unfixed gaps; only 4–6.5/12 SERIOUS findings addressed vs. 8/12 threshold. T2 is vacuous. No mechanism to generate surprise. P(best-paper) ≈ 0.02. |
| **L — Laptop-CPU** | **6** | 6 | Matches ceiling. Spectral annotation: ~9h for 1,065 instances. ML pipeline: seconds. 500-instance census: ~5 days on 4 cores (batch-feasible). No GPUs, no human annotation. Memory is the only constraint near d_max=200 clique-expansion threshold. |
| **F — Feasibility** | **3** | — | Zero code. Broken main proof. Untested core hypothesis. 4–6.5/12 red-team threshold missed. AutoFolio baseline absent. Compound gate survival ~25–30%. These are not theoretical risks — they are current failures. The theory stage's primary deliverable (complete proofs) FAILED. |

**Composite: 3.8/10** (arithmetic mean of V4/D4/BP2/L6/F3)

---

## 2. Evidence-Based Pillar Analysis

### V=4: Evidence for and against

**FOR (V≥4):**
- No public cross-method MIPLIB decomposition census exists (depth_check.md §1.1; verified in final_approach.md §1 P2). This is a genuine gap.
- Spectral features are a new continuous feature family for MIP characterization (paper.tex abstract, lines 39–57). Conceptually distinct from syntactic features.
- Cross-method reformulation selection (Benders vs. DW vs. neither) extends Kruber et al. 2017 which only predicted DW yes/no (paper.tex §1.1, lines 66–83).

**AGAINST (V≤4):**
- Audience is ~50–100 decomposition research groups (Mathematician eval §1: "not OR practitioners at large"). Skeptic eval §2: "Nobody will solve MIPs differently because of this work."
- Only 10–25% of MIPLIB has exploitable block structure (Bergner et al. 2015, cited in depth_check.md S2). Oracle is relevant for ~150/1,065 instances at best.
- CPLEX 22.1 and Gurobi already ship automatic decomposition heuristics (Mathematician eval §1).
- The target improvement is ≥5pp over syntactic features — textbook incremental (Skeptic eval §2).
- T2 delivers zero predictive value (vacuous on 60–70% of MIPLIB; depth_check.md F1, red-team T2-5).

**MY ASSESSMENT:** The census has standalone scholarly value (V3–4 floor). The spectral features add conditional value IF they prove non-redundant with density (untested). The audience cap at ~100 research groups makes "extreme" a stretch. V=4, below the binding ceiling of 5.

**WHERE I DISAGREE WITH PRIOR EVALS:** The Skeptic eval, Community Expert eval, and depth check all score V=5. I score V=4. The prior V=5 scores implicitly credit the spectral features as working. With impl_loc=0 and the G0 test never run, the spectral contribution is pure speculation. The census alone is V3–4. Crediting an untested hypothesis at full value is inconsistent with evidence-based scoring.

### D=4: Evidence

Post-descoping to 26.5K LoC with 30K cap (approach.json, depth_check.md §4 Amendment 3). Novel intellectual content concentrates in ~6.5K of spectral engine (Mathematician eval §2). ML pipeline is "a scikit-learn homework assignment" (Skeptic eval §2). Solver wrappers are API configuration. Census is engineering grind. The genuine difficulty is numerical robustness of eigensolves on near-singular Laplacians (red-team ALG-1: shift-invert on κ~10¹⁰ matrices produces silent errors). This is real but not impressive. Consensus at D=4 is well-calibrated.

### BP=2: Evidence for scoring below ceiling

The depth-check ceiling is BP=3. I score BP=2 because:
1. **L3 proof gaps are UNFIXED.** Verification report §2: two non-trivial gaps in the paper's main theorem (Step 3 dual feasibility for general A, Step 5 (n_e−1) derivation). A paper with an incomplete proof of its main theorem cannot be a best-paper candidate.
2. **Red-team threshold missed.** 4–6.5/12 SERIOUS findings vs. required 8/12 (verification report §3). Three findings fully unaddressed: S-1 (dual degeneracy), S-4 (T2 missing δ<γ/2), S-8 (AutoFolio baseline). These are exactly the items a JoC reviewer will demand.
3. **T2 is vacuous.** C = O(k·κ⁴·‖c‖∞) → 10²⁴+ on typical MIPLIB instances (depth_check.md F1). Informative on ~8–14% of MIPLIB (Skeptic eval §3). Paper has no quantitative theoretical contribution.
4. **Zero empirical results.** The core experiment is entirely unrun. Best-paper requires surprising findings; you cannot surprise anyone with zero data.

The Mathematician eval scores BP=2 on the same logic. The depth check, Skeptic eval, and Community Expert all score BP=3. I agree with the Mathematician: post-theory-stage evidence justifies dropping below the ceiling.

### L=6: Evidence

Spectral annotation: ~30s/instance × 1,065 = ~9h (paper.tex §6.1). ML training: seconds. Pilot (50 instances): hours. Paper evaluation (500 instances): ~5 days on 4 cores (Skeptic eval §3). Full census: ~12 days with 1hr caps. No GPUs required. Fully automated. Memory constraint near d_max=200 threshold (red-team ALG-5: clique expansion can hit 6–10GB). L=6 is well-calibrated; not 7 because the full census is a multi-day batch job.

### F=3: Evidence for aggressive feasibility score

This is where I diverge most sharply from prior evaluations.

**Current state of failures (not risks — actualized failures):**
1. **impl_loc=0** — zero code after 288KB of analysis. The G0 test (50 lines of Python) was never written (Community Expert §3 AC3).
2. **L3 proof FAILED** — verification report §2 verdict: "FAIL". Main theorem is broken.
3. **Red-team threshold FAILED** — 4–6.5/12 vs. 8/12.
4. **AutoFolio baseline MISSING** — "a glaring gap" (red-team SCOPE-2).
5. **Core hypothesis UNTESTED** — spectral features may be pure density proxies (red-team EVAL-8; P(proxy)≈0.30–0.40 per Mathematician eval FF1).

**Compound gate survival:** ~25–30% (Skeptic eval §3: 27%; Mathematician eval §5: 37% for JoC path).

**P(at least one serious risk materializes):** ~63% (Mathematician eval §5).

The Community Expert scores F=5, the Skeptic scores F=4, the Mathematician scores F=4. All of these assume that the theory stage was a partial success. I weigh the theory stage as a **partial failure**: the primary deliverable (complete proofs) has two gaps, the secondary deliverables (red-team resolution) missed threshold, and the one thing that would have de-risked the project (G0) was never run.

**F=3** reflects: well-designed kill gates (good), but the project has already failed its most recent gate (theory verification), and the existential uncertainty remains at its maximum level because nothing was tested.

---

## 3. Fatal Flaws — Severity-Ranked

| # | Flaw | Severity | Evidence | Fixable? | Est. Effort |
|---|------|----------|----------|----------|-------------|
| **FF1** | **Core hypothesis entirely untested** | EXISTENTIAL | State.json: impl_loc=0. G0 never run. Spectral features may be density proxies (red-team EVAL-8, KQ4). P(proxy)≈30–40%. | Yes — run G0 | 1 day |
| **FF2** | **L3 proof has 2 non-trivial gaps** | CRITICAL | Verification report §2: Step 3 (dual feasibility of ȳ for general A with mixed-sign entries); Step 5 ((n_e−1) factor conflates constraint-dropping and variable-duplication models). | Yes — rewrite | 3–5 days |
| **FF3** | **Red-team resolution below threshold** | SERIOUS | Verification report §3: 4–6.5/12 SERIOUS addressed vs. 8/12 required. Unresolved: S-1 (dual degeneracy), S-4 (δ<γ/2 for T2), S-8 (AutoFolio). | Yes | 2 weeks |
| **FF4** | **T2 is vacuous on 60–70% of MIPLIB** | SERIOUS (mitigated) | Depth check F1, red-team T2-5. C=O(k·κ⁴·‖c‖∞)→10²⁴+ for κ~10⁶. | Mitigated by demotion to motivational. But δ²/γ² (Feature 2) loses theoretical justification (red-team §6.6). | N/A |
| **FF5** | **Analysis paralysis** | PROCESS | 288KB of theory, 0 lines of code. Community Expert §3 AC3: "The 50-line G0 test could have been written in the time spent on the 40.7KB red-team report." | Yes — binding condition: next artifact must be Python. | Immediate |
| **FF6** | **AutoFolio baseline absent** | SERIOUS | Red-team SCOPE-2: "a glaring gap." Verification report §3 residual issue #3. The most natural experiment for a JoC reviewer is absent. | Yes | 1 week |
| **FF7** | **"Complete" in title is false** | MODERATE | Depth check S5, red-team §6.3. Only spectral annotations are complete (1,065); decomposition evaluation covers 500 with 60% expected timeouts. | Yes — rename to "Systematic" | 1 hour |
| **FF8** | **Two incompatible Laplacians with hard switch** | MODERATE | Red-team ALG-4, KQ3. Clique-expansion (d_max≤200) and incidence-matrix (d_max>200) have different spectra. Classifier trains on mixed features from different distributions. | Partially — normalize or unify | 2–3 days |

---

## 4. Probability Estimates

### Methodology

I use the Skeptic's conditional-chain approach (decomposing into testable sub-events) rather than holistic estimation, because sub-events are individually calibratable.

| Factor | My Estimate | Evidence |
|--------|-------------|----------|
| P(code works, wrappers operational) | 0.85 | Mature external dependencies (SCIP, GCG, scipy). Standard wrapping. |
| P(G0 passes: spectral ≠ density proxy) | 0.60 | Untested. Red-team EVAL-8 argues γ₂ correlates with density. Community Expert gives 0.65. I shade lower: there's a substantive theoretical reason (Cheeger-type inequalities) to expect correlation. |
| P(G1 passes: ρ ≥ 0.4) | 0.55 given G0 | Correlated with G0 (if spectral ≠ proxy, likely predicts). Lower than some estimates because the retrospective nature of the quality metric (L3 requires LP duals) limits utility. |
| P(spectral > syntactic at G3) | 0.65 given G1 | If G1 passes strongly, modest advantage likely. But the 30% label noise ceiling (red-team KQ5) caps achievable improvement. |
| P(census produces interesting findings) | 0.75 | Largely independent of spectral hypothesis. JoC values open artifacts. |
| P(L3 proof fixable) | 0.90 | Underlying bound is correct (Lagrangian duality). Gaps are presentation, not correctness. |
| P(reviewer acceptance given completed paper) | 0.65 | JoC is natural venue, but T2 vacuity, missing AutoFolio comparison, and class imbalance are reviewer magnets. |

### Outcome Probabilities

| Outcome | Probability | Derivation |
|---------|-------------|------------|
| **P(JoC publication)** | **0.30** | Serial chain: 0.85 × 0.60 × 0.55 × 0.65 × 0.90 × 0.65 ≈ 0.11 (all-gates path). Add dual-path correction for census-led paper surviving spectral failure: +0.05. Add partial-success path (census + modest features): +0.14. Blended ≈ 0.30. |
| **P(any reputable venue)** | **0.55** | Degradation ladder: JoC (0.30) → C&OR census paper (0.15) → CPAIOR data/workshop (0.07) → negative result (0.03). With overlap correction. |
| **P(best-paper at any venue)** | **0.02** | Census would need to reveal genuinely surprising structure AND spectral features dominate by ≥15pp AND L3 is tightened. P(all three) ≈ 0.02. |
| **P(abandon at gates)** | **0.35** | G0 failure (0.40) is the dominant risk. P(G0 fails) + P(G0 passes, G1 fails) + P(both pass, later gate fails) ≈ 0.40 × 0.65 + 0.60 × 0.45 × 0.20 + ... ≈ 0.35. |

---

## 5. Areas Where I DISAGREE with Prior Evaluations

### 5.1 Value: I score V=4; three priors score V=5

**My reasoning:** The V=5 consensus credits the spectral features as contributing value. With zero empirical validation, I cannot credit a hypothesis as delivering value. The census is V3–4. Untested spectral features are V+0 until tested, not V+1. The depth check's V=5 was set pre-theory-stage, assuming the theory stage would provide evidence. It didn't — it provided broken proofs and no experiments.

### 5.2 Best-Paper: I score BP=2; depth check and Skeptic/Community Expert score BP=3

**My reasoning:** BP=3 was set as a ceiling assuming successful theory completion. The theory stage produced L3 with two proof gaps, missed the red-team threshold, and generated zero empirical evidence. The Mathematician eval agrees at BP=2. A paper whose main theorem doesn't prove what it claims, whose crown jewel is vacuous, and whose experiments are entirely unrun cannot score 3 on best-paper potential.

### 5.3 Feasibility: I score F=3; Community Expert scores F=5, Skeptic F=4

**My reasoning:** The Community Expert's F=5 weights the "well-designed kill gates" and "option value of testing." I agree the gates are well-designed — but the project already failed one gate (theory verification: L3 proof FAIL, red-team threshold missed). Scoring F=5 after a failed gate is inconsistent. The Skeptic's F=4 is closer but still generous. The compound probability of gate survival (~27%) combined with actualized failures justifies F=3.

### 5.4 P(JoC): I estimate 0.30; Skeptic eval 0.45, Community Expert 0.35, depth check 0.55

**My reasoning:** The depth check's 0.55 assumed successful theory completion. Theory partially failed. The Skeptic eval's 0.45 is conditional on binding conditions being met; unconditional is closer to 0.19 (their own calculation). My 0.30 accounts for dual-path correction (census can partially compensate for spectral failure) but is more pessimistic than the Community Expert's 0.35 because I weight the red-team threshold failure more heavily. A paper going to JoC with 3 unaddressed SERIOUS findings faces an uphill battle in review.

### 5.5 I AGREE with the Mathematician on novel theorem-equivalents

The Mathematician's assessment of ~1.0 novel theorem-equivalents (discounted from claimed ~3.2) is the most rigorous math evaluation in the corpus. L3-C Benders (~0.3) and L3-C DW (~0.3) are the genuinely novel items. L3 itself is Geoffrion 1974 in hypergraph notation (~0.2). T2 is vacuous (0.0). F1/F2 are textbook (0.1). The math is not load-bearing — it neither drives difficulty nor delivers value. I endorse this assessment fully.

### 5.6 I AGREE with the Skeptic on analysis paralysis but DISAGREE on ABANDON

The Skeptic's diagnosis that 288KB of markdown and 0 lines of code represents inverted priorities is correct and devastating. The Community Expert's AC3 ("The 50-line G0 test could have been written in the time spent on the 40.7KB red-team report") is the single most damning observation in any evaluation. However, the Synthesizer's option-value argument is decisive: testing G0 costs 1 day, abandoning forfeits 40–60 hours. The asymmetry favors testing.

---

## 6. Challenge Testing — Claims vs. Evidence

| Claim | Source | Evidence Status | My Assessment |
|-------|--------|----------------|---------------|
| "Spectral features improve decomposition selection by ≥5pp" | final_approach.md abstract | **ZERO EVIDENCE.** Never tested on a single instance. | Unverifiable. The hedged parenthetical "(or honestly reporting if smaller)" makes this unfalsifiable (red-team EVAL-1). |
| "First complete MIPLIB 2017 decomposition census" | paper.tex title | **Overclaimed.** Spectral annotations: 1,065 (complete). Decomposition evaluation: 500 (not complete). ~60% of full census will timeout (depth check S5). | Change "Complete" to "Systematic." |
| "L3 is the partition-to-bound bridge (main theorem)" | paper.tex §4, approach.json | **Proof is broken.** Two gaps (verification report §2). Underlying bound likely correct (Lagrangian duality) but proof as written doesn't establish it. | Fixable in 3–5 days. But a "main theorem" that doesn't prove its claim is a current failure, not a future risk. |
| "T2 provides theoretical justification for δ²/γ²" | paper.tex §5 | **Vacuous.** C=O(k·κ⁴·‖c‖∞)→10²⁴+ on 60–70% of MIPLIB. | Correctly demoted to motivational. But if T2 is vacuous, Feature 2 (δ²/γ²) has no theoretical justification — it's an empirical feature, not a principled one. |
| "Spectral futility predictor at ≥80% precision" | final_approach.md S1 | **ZERO EVIDENCE.** Renamed from "no-go certificate" (honest). But precision target is entirely aspirational. | Cannot evaluate. Depends on spectral features working at all. |
| "26.5K LoC with 30K cap" | approach.json, depth check Amendment 3 | **Credible.** Post-descoping from inflated 155K. Itemized breakdown is reasonable. | Honest scoping. The 155K→26.5K revision is to the team's credit. |
| "Laptop-feasible" | Multiple sources | **Mostly credible.** Eigensolves fast. Memory concern near d_max=200. Full census is a multi-day batch job. | L=6 is fair. Not "trivially laptop" (not 8+), but feasible with patience. |

---

## 7. Risk Assessment: What Goes Wrong

| Risk | P(materializes) | Impact | Detection | Evidence |
|------|-----------------|--------|-----------|----------|
| Spectral features are density proxies (G0 fail) | 0.40 | Fatal for spectral thesis | G0, Day 1 | Red-team EVAL-8, KQ4. Cheeger inequality gives theoretical reason to expect γ₂-density correlation. |
| Spectral features don't predict decomposition benefit (G1 fail) | 0.25 (given G0 pass) | Fatal for spectral thesis | G1, Week 2 | L3 is retrospective (requires LP duals); spectral features are prospective. Mismatch may break correlation. |
| GCG integration failure | 0.15 | Blocks DW evaluation | G2, Week 4 | GCG API is brittle, version-specific (Skeptic eval §2). |
| L3 proof gaps are deeper than expected | 0.10 | Lose main theorem | Week 1 | Verification report §2. Underlying bound is correct → 90% fixable. |
| Eigensolve numerical instability | 0.30 | Corrupts 5–15% of features | Development | Red-team ALG-1. Silent ARPACK errors on κ~10¹⁰ matrices. |
| Class imbalance → inconclusive results | 0.40 | Weak claims | G4, Week 14 | 75% "neither" class. McNemar on 15–20 minority test samples has near-zero power. |
| Timeline overrun | 0.35 | No slack for iteration | Continuous | 16 weeks for proofs + 26.5K LoC + census + paper. Zero buffer. |

**P(at least one serious risk materializes) ≈ 0.70.** This is higher than the Mathematician's 0.63 because I include the timeline risk (zero code as evidence of execution speed concerns).

---

## 8. Verdict

### **CONDITIONAL CONTINUE** — composite 3.8/10

**Why not ABANDON:**
1. **G0 is cheap.** 1 day of compute resolves the existential question (spectral ≠ density proxy). The option value of testing before forfeiting 40–60 hours of design work is positive.
2. **Census has standalone value.** Even if spectral features fail entirely, the MIPLIB decomposition census is a publishable artifact at C&OR or as a data paper (~P=0.20–0.25).
3. **Kill gates are well-designed.** G0 (Day 1) → G1 (Week 2) → G2 (Week 4) provides fast, cheap exits. The maximum wasted investment before definitive kill is ~2 weeks.

**Why not CONTINUE (unqualified):**
1. **The project already failed its most recent gate** (theory verification: L3 proof FAIL, red-team threshold missed).
2. **Zero code after 288KB of analysis** is a process failure, not just a measurement anomaly.
3. **The core empirical hypothesis has never been tested on a single instance.**
4. **F=3 means the most likely outcome is partial failure** — a census-only data paper (not the proposed JoC computational study).

### Binding Conditions (failure on ANY → ABANDON)

| # | Condition | Deadline | Kill Criterion |
|---|-----------|----------|----------------|
| BC1 | **G0: spectral ≠ density proxy** | 48 hours | R²(γ₂ ~ syntactic) ≥ 0.70 (OLS) OR R² ≥ 0.80 (RF) on ≥5/8 features |
| BC2 | **Next artifact is Python, not markdown** | 48 hours | If next deliverable is a .md file → ABANDON for analysis paralysis |
| BC3 | **L3 proof gaps fixed** | Week 1 | Proof reviewed by independent checker; gaps remain → CONDITIONAL HOLD |
| BC4 | **G1: Spearman ρ(δ²/γ², bound degradation) ≥ 0.4** | Week 2 | ρ < 0.4 → ABANDON spectral thesis; evaluate census-only pivot |
| BC5 | **AutoFolio + SPEC-8 baseline operational** | Week 4 | Cannot begin G3 evaluation without it |
| BC6 | **Title changed: "Complete" → "Systematic"** | Week 1 | Contradiction with depth check S5 |
| BC7 | **All metrics use balanced accuracy** | Before any evaluation | Trivial "always-neither" baseline gets ~75% raw accuracy |

---

## 9. Final Score Comparison

| Pillar | Depth Check | Mathematician | Skeptic | Community Expert | **My Score** | Delta from Consensus |
|--------|-------------|---------------|---------|-----------------|-------------|---------------------|
| V | 5 | 4 | 5 | 5 | **4** | −1 from consensus (5) |
| D | 4 | 4 | 4 | 4 | **4** | 0 |
| BP | 3 | 2 | 3 | 3 | **2** | −1 from consensus (3) |
| L | 6 | 6 | 5 | 6 | **6** | 0 |
| F | — | 4 | 4 | 5 | **3** | −1 from others |
| Composite | 4.5 | 4.0 | 4.2 | 4.6 | **3.8** | Lowest |
| P(JoC) | 0.55 | 0.16 | 0.45 | 0.35 | **0.30** | Between Mathematician and Community Expert |
| P(any pub) | 0.72 | 0.63 | 0.55 | 0.60 | **0.55** | Matches Skeptic |
| P(best-paper) | 0.03 | 0.02 | 0.02 | 0.03 | **0.02** | Matches Mathematician/Skeptic |
| P(abandon) | 0.25 | 0.30–0.40 | 0.30 | 0.30 | **0.35** | On the pessimistic side |

### Summary of Disagreements with Prior Evaluations

1. **V=4 not V=5** — cannot credit untested hypothesis as delivering value.
2. **BP=2 not BP=3** — theory stage failed; proof broken; red-team threshold missed.
3. **F=3 not F=4–5** — actualized failures (broken proof, zero code, missed threshold) are not risks; they're facts.
4. **P(JoC)=0.30 not 0.45–0.55** — conditional chain with realistic gate survival; red-team failures weigh on reviewer acceptance.
5. **I agree with Mathematician** that the math is ornamental (~1.0 novel theorem-equivalents, none load-bearing). The census and empirical features are the real contributions.
6. **I agree with Skeptic** on analysis paralysis diagnosis but disagree on ABANDON because G0's cheapness makes test-then-decide strictly dominant over immediate abandonment.
7. **I agree with Community Expert** that the census is the diamond, the spectral features are the gamble, and the next artifact must be Python.

---

*Independent Auditor Evaluation — spectral-decomposition-oracle — 2025-07-22*
*Verdict: CONDITIONAL CONTINUE at 3.8/10 — barely above the ABANDON threshold.*
*The project survives on option value, not on demonstrated merit.*
