# Theory Gate Report: Spectral Decomposition Oracle

**Verification Chair:** Impartial Best-Paper Committee Chair
**Team:** Independent Auditor · Fail-Fast Skeptic · Scavenging Synthesizer
**Date:** 2026-03-08
**Stage:** Verification (post-theory gate)
**Process:** Independent parallel proposals → adversarial cross-critiques → synthesis → verification signoff

---

## 0. Executive Summary

**Composite: 3.8/10** (V4 / D4 / BP2 / L6 / F3)
**Verdict: CONDITIONAL CONTINUE** — gated on G0/G1 pilot within 2 weeks.

One proposal evaluated (proposal_00). Three independent expert agents produced parallel evaluations, engaged in adversarial cross-critique, and reached a synthesized consensus. The Skeptic dissents ABANDON at composite 2.8/10; the Synthesizer dissents upward at 4.6/10. The chair resolves at 3.8/10 — the lowest composite this project has received — reflecting that actualized failures (broken proof, zero code, missed red-team threshold) must be scored as failures, not as risks.

The project survives on **option value alone**: a 2-week G0/G1 pilot costing ~10% of the remaining budget resolves ~57% of the existential uncertainty. If G0 or G1 fails, ABANDON immediately. If both pass, the project profile improves enough to justify continued investment.

---

## 1. Team Process

### 1.1 Independent Evaluations (Parallel)

| Expert | V | D | BP | L | F | Composite | P(JoC) | P(any pub) | P(abandon) | Verdict |
|--------|---|---|----|---|---|-----------|--------|------------|------------|---------|
| Independent Auditor | 4 | 4 | 2 | 6 | 3 | 3.8 | 0.30 | 0.55 | 0.35 | CONDITIONAL CONTINUE (barely) |
| Fail-Fast Skeptic | 3 | 3 | 1 | 5 | 2 | 2.8 | 0.08 | 0.20 | 0.55 | CONDITIONAL CONTINUE (dissent: ABANDON) |
| Scavenging Synthesizer | 5 | 4 | 3 | 6 | 5 | 4.6 | 0.27 | 0.68 | 0.20 | CONDITIONAL CONTINUE |

### 1.2 Cross-Critique Round

**5 key disagreements identified and resolved:**

**Dispute 1 — Value (V3 vs V4 vs V5):**
- Skeptic scores expected value (untested × P(works) = low); Synthesizer credits census floor + conditional spectral upside.
- **Resolution: V=4.** Census alone earns V3–4 (genuine gap, no comparable public artifact). Untested spectral hypothesis gets fractional credit (theoretically motivated, not yet demonstrated). V=5 is too generous for an unvalidated feature family; V=3 underweights the census.

**Dispute 2 — Best-Paper (BP1 vs BP2 vs BP3):**
- Synthesizer argues BP=3 per binding ceiling; Auditor argues theory stage failures justify dropping below.
- **Resolution: BP=2.** The depth-check ceiling of BP=3 assumed successful theory completion. The theory stage produced: L3 proof with 2 gaps (FAIL verdict from own verification), T2 vacuous (demoted), red-team 4–6.5/12 vs 8/12 threshold. These are actualized failures, not risks. P(best-paper) ≈ 0.02 at any venue.

**Dispute 3 — Feasibility (F2 vs F3 vs F5):**
- Skeptic weights actualized failures; Synthesizer weights forward-looking kill-gate design.
- **Resolution: F=3.** 536KB of markdown and 0 lines of code after theory stage is a strong negative signal about execution capability. The L3 proof FAIL and red-team threshold miss are current failures, not future risks. Well-designed kill gates partially offset (cheap exits exist) but cannot erase a revealed preference for analysis over execution.

**Dispute 4 — P(JoC) (0.08 vs 0.27 vs 0.30):**
- Skeptic's conditional chain (0.12 gate survival × 0.65 = 0.08) ignores correlated gate success; Synthesizer's portfolio model (0.27) overcredits undemonstrated execution.
- **Resolution: P(JoC) = 0.20.** Built from calibrated conditional components with correlation correction. This is the geometric mean of the three estimates, which is appropriate when experts use fundamentally different estimation methodologies.

**Dispute 5 — Continue vs Abandon:**
- Skeptic argues opportunity cost (20 person-weeks better spent on fresh project); Synthesizer argues option value (2-week test dominates abandon by 8:1).
- **Resolution: Option value wins.** 2 weeks at 10% of budget resolving 57% of uncertainty is strictly dominant over immediate abandonment, regardless of the alternative project's quality. But G1 (ρ ≥ 0.4) is a hard kill gate — no renegotiation.

### 1.3 Verification Signoff

Chair verified: no score inflation from Synthesizer accepted into consensus; Skeptic's attacks on actualized failures incorporated into F and BP scores; probability estimates use the most defensible methodology from each expert. **APPROVED.**

---

## 2. Pillar Assessment

### Pillar 1: Extreme and Obvious Value — 4/10

**Census (V3–4 floor):** No public cross-method MIPLIB 2017 decomposition census exists. The first systematic evaluation of which instances benefit from Benders vs. DW vs. neither fills a genuine gap for ~50–100 decomposition research groups. This is real, lasting infrastructure value with a long citation tail — but the audience is niche.

**Spectral features (conditional +1, unvalidated):** Eight spectral features from the constraint hypergraph Laplacian are conceptually distinct from syntactic features. Cross-method reformulation selection extends Kruber et al. 2017. But with impl_loc=0 and the G0 test never run, the spectral contribution is speculation, not demonstrated value.

**Limiting factors:** Only 10–25% of MIPLIB has exploitable block structure (Bergner et al. 2015). Commercial solvers already ship automatic decomposition. The target improvement is ≥5pp — incremental by definition. T2 delivers zero predictive value (vacuous on 60–70% of MIPLIB).

**Score rationale:** V=4 reflects genuine census value + fractional credit for theoretically motivated but unvalidated features. Cannot reach V=5 without at least one instance demonstrating spectral utility.

### Pillar 2: Genuine Software Difficulty — 4/10

**Genuinely hard (~6.5K LoC):** Hypergraph Laplacian construction for non-square, mixed-sign constraint matrices. Numerical robustness of eigensolves on near-singular Laplacians (κ~10¹⁰). Two incompatible Laplacian variants (clique-expansion vs. incidence-matrix) with different spectral properties.

**Routine (~20K LoC):** sklearn RF/XGBoost pipeline (~3K). SCIP/GCG solver wrappers (~6K). Census infrastructure with timeout handling (~8K). Tests and analysis (~7K).

**Score rationale:** Post-descoping to 26.5K LoC, the novel intellectual content concentrates in ~6.5K lines. A competent PhD student builds the complete system in 6–8 weeks. Moderate engineering difficulty; low mathematical difficulty. The math (~1.0 novel theorem-equivalents, per Mathematician) is ornamental — it neither drives the difficulty nor delivers the value.

### Pillar 3: Best-Paper Potential — 2/10

**Why best-paper is unreachable:**
- T2 is vacuous → theory venues (MPC/IPCO) eliminated (P=0.05)
- L3 is Geoffrion 1974 in hypergraph notation → no novel mathematical headline
- L3 proof has 2 unfixed gaps → paper's main theorem doesn't prove its claim
- Core experiment is unrun → no data, no possibility of surprise
- Red-team 4–6.5/12 vs required 8/12 → quality gate missed
- Census finding is predictable ("75% of instances: nothing helps")
- Audience: ~50–100 research groups

**Best-case scenario (P ≈ 0.02):** Census reveals 30–40% unexploited decomposable structure (contradicting Bergner 2015) AND spectral features dominate by ≥15pp AND L3 yields tight computable bound. All three simultaneously is near-impossible.

**Score rationale:** BP=2, below the depth-check ceiling of 3, because the theory stage failed its primary deliverables (proofs, red-team resolution). Publishable at JoC with significant revision; not competitive for any best-paper award.

### Pillar 4: Laptop-CPU Feasibility — 6/10

| Component | Time | Feasible? |
|-----------|------|-----------|
| Spectral features (per instance) | ~30s | ✅ Trivial |
| Full spectral annotation (1,065) | ~9 hours | ✅ Easy |
| Pilot evaluation (50 instances) | ~2 hours | ✅ Trivial |
| Dev evaluation (200 instances) | ~3–4 days | ✅ Comfortable |
| Paper evaluation (500 instances) | ~5 days on 4 cores | ⚠ Tight but doable |
| Full census (1,065 instances) | ~12 days on 4 cores | ⚠ Batch job |

No GPUs required. No human annotation. No human studies. Fully automated. Memory constraint near d_max=200 clique-expansion threshold (up to 3GB/instance). The tiered design (50→200→500→1065) enables iterative development.

### Pillar 5: Feasibility — 3/10

**Actualized failures (not risks):**
1. **impl_loc=0** — 536KB of markdown, zero lines of code after theory stage
2. **L3 proof: FAIL** — own verification report assigned FAIL; 2 non-trivial gaps
3. **Red-team: FAIL** — 4–6.5/12 SERIOUS findings addressed vs 8/12 threshold
4. **Core hypothesis: UNTESTED** — the 50-line G0 test was never written
5. **AutoFolio baseline: ABSENT** — the most natural comparison is missing

**Mitigating factors:**
- Kill gates G0 (Day 1) and G1 (Week 2) provide cheap, fast exits
- Census has standalone publication value (P~0.20–0.25 as data paper)
- External dependencies are mature (SCIP, GCG, scipy, sklearn)
- Degradation ladder prevents total failure (P(zero output) ≈ 0.05)

**Compound gate survival: ~15–25%** (range reflects Skeptic's 0.12 to Synthesizer's 0.27; chair resolves at ~0.20).

**Score rationale:** F=3 reflects the weight of actualized failures. The project already failed its most recent gate (theory verification). Well-designed kill gates partially offset, but a revealed preference for analysis over execution is a strong negative signal for a project that requires 26.5K LoC of implementation.

---

## 3. Fatal Flaws

### FF-1: Core empirical hypothesis entirely untested (EXISTENTIAL)

**Evidence:** State.json: impl_loc=0. The G0 test (50 lines of Python) was never written. The entire project rests on one claim — "spectral features predict decomposition benefit better than syntactic features" — that has never been tested on a single instance.

**Fix:** Run G0 within 48 hours. If spectral features are density proxies (R² ≥ 0.70 OLS or R² ≥ 0.80 RF on ≥5/8 features), ABANDON spectral thesis.

### FF-2: L3 proof has 2 non-trivial gaps (CRITICAL, fixable)

**Evidence:** Verification report §2: Step 3 asserts dual feasibility of ȳ without establishing it for general mixed-sign A. Step 5 conflates constraint-dropping and variable-duplication Lagrangian models. Verification verdict: FAIL.

**Fix:** Rewrite via variable-duplication Lagrangian relaxation. Estimated 3–5 days. The underlying bound is correct (Lagrangian duality).

### FF-3: Red-team threshold missed (SERIOUS)

**Evidence:** 4–6.5/12 SERIOUS findings addressed vs. 8/12 threshold. Unresolved: S-1 (dual degeneracy), S-4 (δ<γ/2 assumption), S-8 (AutoFolio baseline).

### FF-4: T2 is vacuous (SERIOUS, mitigated)

**Evidence:** C = O(k·κ⁴·‖c‖∞) evaluates to >10²⁴ on typical MIPLIB instances. Informative on ~8–14% of MIPLIB; vacuous on 60–70%. Correctly demoted to motivational.

### FF-5: Analysis paralysis (PROCESS)

**Evidence:** 536KB of prose, 0 bytes of code. The 50-line G0 test could have been written in the time spent on any single evaluation document. This is the most severe case of analysis paralysis in the portfolio.

### FF-6: AutoFolio baseline missing (SERIOUS)

**Evidence:** Red-team S-8/SCOPE-2: "a glaring gap." Adding 8 spectral features to AutoFolio's existing feature pipeline is the most natural comparison. A JoC reviewer will demand this.

### FF-7: Trivial baseline dominance (METHODOLOGICAL)

**Evidence:** ~75% of MIPLIB has no exploitable block structure. A "always say neither" classifier achieves ~75% raw accuracy. The ≥65% method-prediction target is below this trivial floor. All metrics must use balanced accuracy or macro-F1.

---

## 4. Probability Estimates (Consensus)

| Outcome | Auditor | Skeptic | Synthesizer | **Chair Estimate** |
|---------|---------|---------|-------------|-------------------|
| P(JoC) | 0.30 | 0.08 | 0.27 | **0.20** |
| P(any reputable venue) | 0.55 | 0.20 | 0.68 | **0.45** |
| P(best-paper) | 0.02 | 0.01 | 0.02 | **0.02** |
| P(abandon at gates) | 0.35 | 0.55 | 0.20 | **0.40** |
| P(zero output) | — | 0.80 | 0.05 | **0.10** |

**Methodology notes:**
- P(JoC)=0.20 uses correlation-corrected conditional chain with dual-path adjustment
- P(any pub)=0.45 includes degradation ladder (JoC → C&OR → CPAIOR → data paper → negative result)
- P(abandon)=0.40 weights G0 as the dominant risk (P(G0 fail)≈0.35–0.40)
- Unconditional estimates — not conditioned on binding conditions being met

---

## 5. What This Project IS and IS NOT

**What this project IS:**
- A computational study introducing spectral features for MIP decomposition selection
- The first cross-method MIPLIB 2017 census (empirical infrastructure artifact)
- Feature engineering with theoretical motivation from spectral perturbation theory
- Target: INFORMS JoC (primary), C&OR / CPAIOR (secondary)
- Budget: 26.5K LoC / 30K cap
- A project that survives on option value, not demonstrated merit

**What this project IS NOT:**
- A bridging theorem connecting two fields (T2 is vacuous)
- A best-paper contender at any venue (P ≈ 0.02)
- A project with validated core claims (hypothesis untested, proof broken)
- A project with demonstrated execution capability (536KB docs, 0 code)
- Appropriate for theory venues (MPC/IPCO: P = 0.05)

---

## 6. Binding Conditions for Continuation

Failure on **any** condition triggers ABANDON.

| # | Condition | Deadline | Kill Criterion |
|---|-----------|----------|----------------|
| **BC1** | G0: spectral ≠ density proxy | 48 hours | R²(spectral ~ syntactic) ≥ 0.70 OLS or ≥ 0.80 RF on ≥5/8 features |
| **BC2** | Next artifact is Python, not markdown | 48 hours | If next deliverable is .md → ABANDON |
| **BC3** | L3 proof gaps fixed | Week 1 | Complete proof survives independent review |
| **BC4** | G1: Spearman ρ(δ²/γ², bound degradation) ≥ 0.4 | Week 2 | ρ < 0.4 → ABANDON spectral thesis |
| **BC5** | AutoFolio + SPEC-8 baseline operational | Week 4 | Required before G3 evaluation |
| **BC6** | Title: "Complete" → "Systematic" | Week 1 | Non-negotiable |
| **BC7** | All metrics use balanced accuracy | Before any evaluation | Trivial baseline gets ~75% raw accuracy |

**Convert to ABANDON if:**
- G0 fails (spectral features are density proxies)
- G1 fails (spectral ratio is not predictive)
- Team rejects Amendment E (insists on T2-centered paper or MPC/IPCO targeting)
- No code exists 2 weeks from now
- Next deliverable is markdown, not Python

---

## 7. Kill Gates

| Gate | Week | Condition | Kill If |
|------|------|-----------|---------|
| G0 | 0 | Spectral ≠ density proxy | R² ≥ 0.70 (OLS) or R² ≥ 0.80 (RF) |
| G1 | 2 | Spectral premise validation | ρ < 0.40 |
| G2 | 4 | Solver wrappers operational | <80% of pilot produces valid bounds |
| G3 | 8 | Spectral > syntactic | Spectral balanced accuracy ≤ syntactic |
| G4 | 14 | Full evaluation | ρ < 0.5 AND balanced acc < 65% AND precision < 80% |
| G5 | 18 | Internal review | Fundamental problems flagged |

---

## 8. Dissent Record

### Fail-Fast Skeptic (Agent-1) — Dissent: ABANDON at 2.8/10

"This project is a documentation exercise masquerading as research software. 536KB of markdown and zero lines of code is not a theory stage — it is analysis paralysis. P(JoC) unconditional = 0.08. The expected return per person-week is 4× higher on a fresh project. I do not block CONDITIONAL CONTINUE under the option-value argument (2-week G0/G1 test is cheap), but I record that my independent scores yield ABANDON and that the consensus has been systematically pulled upward by the Synthesizer's optimistic salvage accounting."

**Skeptic's independent scores:** V3/D3/BP1/L5/F2 → composite 2.8/10

### Scavenging Synthesizer (Agent-2) — Dissent: Upward at 4.6/10

"The consensus underweights the degradation ladder and salvage value. P(any pub) = 0.68 via 7 outcome paths — the census alone provides a publication floor surviving nearly every failure mode. P(zero output) = 0.05. The option value of testing dominates abandonment by 8:1. F should be 5 (kill gates are well-designed), not 3 (which scores only actualized failures without crediting the forward-looking mitigants)."

**Synthesizer's independent scores:** V5/D4/BP3/L6/F5 → composite 4.6/10

---

## 9. Comparison with Prior Evaluations

| Metric | Depth Check | Skeptic Eval | Mathematician Eval | Community Expert | **This Report** |
|--------|-------------|-------------|-------------------|-----------------|----------------|
| V | 5 | 5 | 4 | 5 | **4** (↓1) |
| D | 4 | 4 | 4 | 4 | **4** (=) |
| BP | 3 | 3 | 2 | 3 | **2** (↓1) |
| L | 6 | 5 | 6 | 6 | **6** (=) |
| F | — | 4 | 4 | 5 | **3** (↓1–2) |
| P(JoC) | 0.55 | 0.45 | 0.16 | 0.35 | **0.20** (↓) |
| P(any pub) | 0.72 | 0.55 | 0.63 | 0.60 | **0.45** (↓) |
| P(best-paper) | 0.03 | 0.02 | 0.02 | 0.03 | **0.02** (=) |

**Downward revisions justified by:** L3 proof FAIL in theory stage, red-team threshold miss, zero code after theory stage, core hypothesis untested. The depth-check scores assumed a successful theory stage; the theory stage partially failed.

---

## 10. Verdict

### CONDITIONAL CONTINUE — Composite V4/D4/BP2/L6/F3 (3.8/10)

**Vote:**
| Expert | Verdict | Composite |
|--------|---------|-----------|
| Independent Auditor | CONDITIONAL CONTINUE | 3.8 |
| Fail-Fast Skeptic | CONDITIONAL CONTINUE (dissent: ABANDON at 2.8) | 2.8 |
| Scavenging Synthesizer | CONDITIONAL CONTINUE | 4.6 |

**Rationale:** The project survives on option value, not demonstrated merit. The 2-week G0/G1 pilot resolves the existential question (spectral features ≠ density proxy AND predict decomposition benefit) at minimal cost. If both gates pass (P ≈ 0.35–0.44), the project profile improves substantially. If either fails, ABANDON immediately with ≤2 weeks marginal cost.

**The honest framing:** This is a marginal project with a 20% chance of its headline outcome (JoC), a 45% chance of any publication, and a 40% chance of abandonment. It continues because testing is cheap and abandonment forfeits 40–60 hours of design work for zero output. The next artifact must be Python, not markdown.

---

## Rankings

```json
{
  "rankings": [
    {
      "proposal_id": "proposal_00",
      "score": 3.8,
      "verdict": "CONTINUE",
      "reason": "Composite V4/D4/BP2/L6/F3. CONDITIONAL CONTINUE gated on G0/G1 pilot within 2 weeks. Census has genuine standalone value. Spectral hypothesis untested but cheaply testable. Option value of 2-week pilot dominates abandonment. P(JoC)=0.20, P(any pub)=0.45, P(best-paper)=0.02. Hard kill: G0 fail (spectral=density proxy) or G1 fail (rho<0.4) or next artifact is markdown not Python.",
      "scavenge_from": []
    }
  ]
}
```

---

*Theory Gate Report — spectral-decomposition-oracle — 2026-03-08*
*Team: Independent Auditor + Fail-Fast Skeptic + Scavenging Synthesizer*
*Process: Independent parallel evaluation → adversarial cross-critique → synthesis → verification signoff*
*Verdict: CONDITIONAL CONTINUE — V4/D4/BP2/L6/F3 (3.8/10)*
*Next action: Write Python. Run G0. 48 hours.*
