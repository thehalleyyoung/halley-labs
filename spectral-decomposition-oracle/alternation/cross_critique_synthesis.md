# Cross-Critique Synthesis: Spectral Decomposition Oracle (proposal_00)

**Panel Chair:** Team Lead (Cross-Critique Phase)
**Date:** 2026-03-08
**Status:** FINAL CONSENSUS

---

## 0. Panel Composition & Methodology

| Role | Agent | Composite | Verdict |
|------|-------|-----------|---------|
| Independent Auditor | Agent-0 | 3.8/10 | CONDITIONAL CONTINUE (barely) |
| Fail-Fast Skeptic | Agent-1 | 2.8/10 | CONDITIONAL CONTINUE (dissent: ABANDON) |
| Scavenging Synthesizer | Agent-2 | 4.6/10 | CONDITIONAL CONTINUE |

**Methodology:** Each disagreement is resolved by (1) identifying which expert's methodology best fits the evidence, (2) checking against hard facts from the repository (676KB markdown, 0 code files, 0 theory bytes, depth-check ceiling scores), and (3) applying the principle that *scores should reflect the current state of the project, not the best-case future state*.

---

## 1. Resolution of Disagreements

### 1.1 Value Score: V3 vs V4 vs V5

**Positions:**
- **Skeptic (V3):** Scores expected value. Untested hypothesis × P(works) = low. The spectral contribution is vaporware.
- **Auditor (V4):** Census has real value (V3–4 floor), but can't credit the spectral hypothesis at full value without any empirical validation.
- **Synthesizer (V5):** Census floor is V4, plus conditional spectral adds V+1.

**Evidence adjudication:**
1. The depth-check ceiling for Value is **V=5** (median of three pre-critique experts). This is a *ceiling*, not a guaranteed score.
2. The census is genuinely novel — no systematic cross-method decomposition annotation of MIPLIB 2017 exists. This is confirmed by all three experts and the depth check.
3. The spectral hypothesis (ρ ≥ 0.4 between δ²/γ² and observed bound degradation) is **completely untested** — `impl_loc: 0`, `theory_bytes: 0`, zero pilot data.
4. L3 (partition-to-bound bridge) is sound and provides theoretical grounding, but has not been applied to real data.

**Resolution: V = 4**

The Auditor's position is best-supported. The census alone is V=3–4 (a useful artifact with venue fit at JoC/C&OR). Theoretical motivation via L3 and the qualitative insight from T2 (even with vacuous constant) earns a half-step above bare census, rounding to V=4. The Synthesizer's V=5 credits the spectral hypothesis as if it has partial empirical validation — it does not. The Skeptic's V=3 undervalues the census, which is a durable contribution surviving all failure modes.

**Key principle applied:** Untested but theoretically motivated features get *fractional* credit (~0.5 points), not full credit (+1). The theoretical motivation is real (L3 is sound, spectral features have a principled basis), but zero empirical validation means the value is conditional, not actualized.

---

### 1.2 Best-Paper Score: BP1 vs BP2 vs BP3

**Positions:**
- **Skeptic (BP1):** No mechanism for surprise. No novel theory (T2 broken), no data, no implementation. Best-paper requires *something surprising to exist*, and nothing exists yet.
- **Auditor (BP2):** Theory stage produced a broken proof. The most recent gate was failed. Can't award ceiling score after a failure.
- **Synthesizer (BP3):** Matches the depth-check binding ceiling. JoC venue fit is strong for computational studies with artifacts.

**Evidence adjudication:**
1. The depth-check ceiling is **BP=3** (unanimous across all three pre-critique experts). This was explicitly described as "publishable at JoC, not competitive for best-paper at any target venue."
2. T2's proof is not merely incomplete — its constant is *vacuously large* (κ⁴ term). This is a concrete intellectual failure, not just an unfinished task.
3. Zero code and zero pilot data mean there is no empirical surprise to point to.
4. However: BP scores are forward-looking (probability of achieving best-paper). The census *could* reveal surprising structural patterns. The 2-week pilot *could* validate the spectral hypothesis with unexpectedly strong ρ.
5. BP=3 at a ceiling of 3 means the project is at *maximum* best-paper potential — this contradicts the broken-proof reality.

**Resolution: BP = 2**

The Auditor's position is correct. The depth-check ceiling of BP=3 was set *conditional on successful theory-stage execution*. T2's vacuous constant is an actualized failure, not a risk. Awarding the ceiling score after a gate failure violates the purpose of ceilings. BP=1 (Skeptic) is too harsh because: (a) L3 remains sound and novel, (b) the census concept has strong JoC venue fit, and (c) the kill-gate structure means surprising empirical results *could* emerge from the pilot. But the mechanism for best-paper surprise requires something the project does not currently possess.

**Key principle applied:** Ceiling scores require clean execution through the relevant phase. A broken proof in the theory phase drops BP below ceiling by at least 1.

---

### 1.3 Feasibility Score: F2 vs F3 vs F5

**Positions:**
- **Skeptic (F2):** Actualized failures compound multiplicatively. Zero code after theory phase. Broken proof. 5× LoC inflation. P(JoC) = 0.08. The project has *demonstrated* inability to execute.
- **Auditor (F3):** The project failed its most recent gate (theory stage). This is a concrete negative signal, not a risk factor.
- **Synthesizer (F5):** Well-designed kill gates limit downside. P(any pub) = 0.68. The 2-week test resolves core uncertainty cheaply. Tools exist (SCIP, GCG, ARPACK). Descoped 25–30K LoC is tractable.

**Evidence adjudication:**
1. **Hard facts supporting low F:** 676KB of markdown, 0 lines of code. Theory phase produced no formal proofs (`theory_bytes: 0`). The LoC estimate was inflated 5–6×. These are execution failures, not theoretical risks.
2. **Hard facts supporting higher F:** Kill gates are well-designed (G1 at week 2 resolves core question for ~10% of total investment). Descoped scope (25–30K LoC) uses existing tools (SCIP 7.0+ Benders, GCG DW). Spectral eigendecomposition is a solved computational problem (ARPACK/SciPy). The team has demonstrated strong analytical capability (depth check is thorough).
3. **The critical asymmetry:** Feasibility should weight *actualized failures* more heavily than *future risk mitigants*. A project that has produced 676KB of documentation and 0 bytes of code or proofs has demonstrated a pattern, not just bad luck.

**Resolution: F = 3**

The Auditor's position is most defensible. F=5 (Synthesizer) is unjustifiable for a project with zero code, zero proofs, and a broken theoretical centerpiece — "well-designed kill gates" describe risk management, not current feasibility. F=2 (Skeptic) overstates the compound failure risk because: (a) the failures have been *diagnosed and have concrete remediation paths* (amendments), (b) the descoped scope uses battle-tested tools, and (c) the 2-week pilot genuinely limits downside exposure. F=3 correctly reflects "the project has failed its most recent gate but has a credible, cheap path to resolution."

**Key principle applied:** Feasibility scores should primarily reflect demonstrated execution capability, secondarily reflect risk management structure. Zero code after a completed theory phase is a strong negative signal that risk management alone cannot offset.

---

### 1.4 P(JoC): 0.08 vs 0.27 vs 0.30

**Methodological comparison:**

| Expert | Method | P(JoC) | Strength | Weakness |
|--------|--------|--------|----------|----------|
| Skeptic | Conditional chain: P(all 5 gates pass) = 0.12, × P(JoC\|pass) = 0.65 | 0.08 | Conservative; models compounding risk; no double-counting | Assumes near-independence of gates (G1→G3 are highly correlated); ignores census-only JoC path |
| Synthesizer | Portfolio model with 7 outcome paths | 0.27 | Models multiple landing zones; accounts for partial success | May overweight census-only JoC path; portfolio diversification assumptions may be optimistic |
| Auditor | Dual-path correction: spectral path + census-only path | 0.30 | Accounts for path diversity; corrects for census floor | May double-count some probability mass; 0.30 is high for zero-code project |

**Resolution methodology — build from components:**

**Path A (Spectral success → JoC):**
- P(G1 pass: ρ ≥ 0.4) = 0.55 [depth-check estimate, reasonable given theoretical motivation]
- P(successful implementation | G1 pass) = 0.60 [descoped scope, existing tools, but zero track record]
- P(G3 pass: spectral > syntactic | implemented) = 0.70 [conditional on G1, features that predict bound quality likely predict selector quality]
- P(JoC acceptance | full results) = 0.55 [competitive venue, but census + spectral is a strong package]
- **Path A total: 0.55 × 0.60 × 0.70 × 0.55 ≈ 0.13**

**Path B (Spectral fails → census-only JoC):**
- P(spectral fails but census completed) = 0.45 × 0.70 = 0.315
- P(JoC accepts census-only paper) = 0.15 [possible but a census without a selection thesis is thin for JoC]
- **Path B total: 0.315 × 0.15 ≈ 0.05**

**Combined P(JoC) ≈ 0.13 + 0.05 = 0.18**

**Resolution: P(JoC) = 0.18, rounded to 0.20 for communication**

The Skeptic's 0.08 is too low primarily because it ignores gate correlation (G1 pass dramatically boosts G3 pass) and the census fallback. The Auditor's 0.30 and Synthesizer's 0.27 are too high for a project with zero executed code — they implicitly assume implementation success probabilities above 0.7, which is generous given the track record. The component-build model yields 0.18, which I round to 0.20 acknowledging residual uncertainty.

**Key principle applied:** Build P(JoC) from conditional components with explicit gate correlations. Do not assume gate independence (Skeptic's error) or implicitly credit undemonstrated execution capability (Auditor/Synthesizer's error).

---

### 1.5 ABANDON vs Option Value

**Skeptic's argument:** Expected return per person-week is 4× higher on a fresh project. 20 person-weeks on this project yields E[value] ≈ 0.08 × JoC_value. A new project with P(pub) ≈ 0.35 and similar effort yields 4× the expected return.

**Synthesizer's counter:** The option value of "test 2 weeks then decide" dominates "abandon now" by 8:1. Spending 2/20 weeks (10% of budget) resolves 57% of the uncertainty (G1: does ρ ≥ 0.4?).

**Resolution: The option-value argument wins, but with strict conditions.**

The Skeptic's opportunity-cost analysis has a critical flaw: it **compares a known project (with diagnosed problems and concrete remediation) against an idealized alternative** (a fresh project with assumed P(pub) ≈ 0.35). In reality:
- A fresh project also has uncertainty, ramp-up costs, and failure modes
- The sunk cost in problem formulation, evaluation design, and amendment structure has *real* reuse value if G1 passes
- The 2-week test is genuinely cheap: 10% of budget resolves a binary question

However, the Skeptic is correct that **beyond the 2-week test, continued investment requires strong justification**. The option-value argument is time-limited:

| If G1 result | Action | Rationale |
|--------------|--------|-----------|
| ρ ≥ 0.4 | CONTINUE to G2 | Core hypothesis validated; remaining risk is execution, not concept |
| 0.3 ≤ ρ < 0.4 | REASSESS with reduced scope | Weak signal; census-only path may dominate |
| ρ < 0.3 | ABANDON immediately | No spectral signal → census-only paper at C&OR *might* justify 4 more weeks, but not the full 20 |

**Key principle applied:** Option value dominates opportunity cost *when the test is cheap and resolves substantial uncertainty*. But option value is not a perpetual license — each gate must clear its threshold or the opportunity-cost argument reasserts.

---

## 2. Final Consensus Scores

| Pillar | Score | Justification |
|--------|-------|---------------|
| **Value (V)** | **4** | Census floor (V=3–4) + fractional credit for theoretically motivated but untested spectral features. Cannot credit V=5 without any empirical validation. |
| **Difficulty (D)** | **4** | Depth-check median, uncontested. Post-descoping, the novel technical work is moderate: hypergraph Laplacian adaptation (hard), ML pipeline (standard), solver wrappers (routine). |
| **Best-Paper (BP)** | **2** | Below depth-check ceiling of 3 due to actualized theory-stage failure (T2 vacuous constant). L3 soundness and census venue fit prevent drop to 1. |
| **Laptop CPU (L)** | **6** | Uncontested. Spectral analysis is trivially laptop-feasible. Census bottleneck is batch time, not hardware. |
| **Feasibility (F)** | **3** | Project failed most recent gate (theory phase: broken proof, zero code). Credible remediation path (cheap pilot, existing tools, descoped scope) prevents drop to 2. |

**Composite: (4 + 4 + 2 + 6 + 3) / 5 = 3.8 / 10**

*Note: This composite uses equal weights. If Value and Feasibility are weighted 2× (as they most directly determine publishability), the weighted composite is (2×4 + 4 + 2 + 6 + 2×3) / 7 = 26/7 ≈ 3.7.*

---

## 3. Final Probability Estimates

| Outcome | Estimate | 80% CI | Methodology |
|---------|----------|--------|-------------|
| **P(JoC)** | **0.20** | [0.10, 0.30] | Component-build: spectral path (0.13) + census-only path (0.05), rounded up for model uncertainty |
| **P(any reputable pub)** | **0.45** | [0.25, 0.60] | Includes C&OR, CPAIOR, EJOR. Census-only paper has ~0.25 standalone probability; spectral success adds ~0.20 |
| **P(best-paper)** | **0.02** | [0.005, 0.04] | Requires census to reveal genuinely surprising structural insight AND spectral features to dominate. Uncontested across panel. |
| **P(abandon)** | **0.40** | [0.30, 0.55] | P(G1 fail: ρ < 0.3) ≈ 0.30 + P(G1 pass but later failure) ≈ 0.10. Higher than Synthesizer's 0.05, lower than Skeptic's 0.55. |
| **P(zero output)** | **0.10** | [0.05, 0.20] | Census artifact survives most failure modes; only total team dissolution or SCIP/GCG integration impossibility yields zero |

**Probability reconciliation notes:**
- P(JoC) = 0.20 is the geometric mean region of the three estimates (0.08, 0.27, 0.30), pulled toward the Skeptic because zero-code projects warrant conservative estimates.
- P(any pub) = 0.45 is notably lower than the Synthesizer's 0.68 and the depth-check's 0.72, reflecting the penalty for zero demonstrated execution.
- P(abandon) = 0.40 is the arithmetic mean of Skeptic (0.55) and Auditor (0.35), reflecting that the kill gates are real and the project has genuine vulnerabilities.

---

## 4. Final Verdict

### **CONDITIONAL CONTINUE — 2-week pilot, then hard gate**

The project proceeds **only** under all of the following binding conditions:

**Immediate (Week 0):**
1. Team explicitly accepts Amendment E restructuring (census-first, 25–30K LoC, honest terminology)
2. Original proposal framing is formally abandoned (no "bridging theorem as headline," no "155K LoC," no "MPC/IPCO target")

**Gate 0 — Week 1:**
3. SCIP Benders wrapper + GCG DW wrapper operational on ≥5 test instances
4. Hypergraph Laplacian construction + eigendecomposition running on ≥5 instances
5. *Failure = ABANDON* (if basic tooling can't be stood up in 1 week, execution capacity is insufficient)

**Gate 1 — Week 2:**
6. 50-instance pilot: compute Spearman ρ between δ²/γ² and observed bound degradation
7. **ρ ≥ 0.4 → CONTINUE** to full development
8. **0.3 ≤ ρ < 0.4 → REASSESS** with census-only scope (reduced to ~8 weeks, C&OR target)
9. **ρ < 0.3 → ABANDON** (spectral premise falsified)

**Gate 2 — Week 4:**
10. SCIP Benders + GCG DW produce dual bounds on ≥80% of 50-instance pilot
11. *Failure = ABANDON*

**Gates 3–5:** Per depth-check schedule (weeks 8, 14, 18)

**Resource cap:** 20 person-weeks maximum. No extensions without new depth check.

---

## 5. Dissent Record

### Fail-Fast Skeptic (Agent-1) — PARTIAL DISSENT

**Accepts:** Final scores V=4, D=4, L=6. Accepts 2-week pilot as the minimum viable test.

**Dissents on:**
- **BP=2:** Maintains BP=1. "A broken proof and zero data cannot earn BP=2. L3 is a lemma, not a best-paper mechanism. The census is useful but mundane — no one gives best-paper awards for data collection." *Chair response: Noted. BP=2 reflects the conditional probability that the pilot reveals a surprising empirical result, which L3 can frame. This is a low probability (reflected in P(best-paper)=0.02) but not zero.*
- **F=3:** Maintains F=2. "676KB of markdown and 0 bytes of code is a documentation project, not a research project. F=3 is generous for a project that has never demonstrated it can write code." *Chair response: Noted. F=3 credits the risk management structure (cheap gates) and the availability of existing tools. The Skeptic's concern about execution capability is reflected in P(abandon)=0.40.*
- **P(JoC)=0.20:** Maintains 0.08–0.12. "The component-build model assigns P(successful implementation | G1 pass) = 0.60, which is unjustified for a team that has written zero code." *Chair response: Partially accepted. The 0.60 assumption is generous; if reduced to 0.40, P(JoC) drops to 0.14. The range [0.10, 0.30] captures this uncertainty.*
- **P(abandon)=0.40:** Maintains 0.50–0.55. *Chair response: The difference is within the 80% CI.*

**Skeptic's standing condition:** "If G0 is not passed by end of Week 1, my dissent converts to a full ABANDON recommendation with no further gates."

### Independent Auditor (Agent-0) — ACCEPTS CONSENSUS

Accepts all final scores and probabilities. Notes that the consensus composite (3.8) matches the Auditor's independent composite exactly, providing convergent validation.

### Scavenging Synthesizer (Agent-2) — PARTIAL DISSENT

**Accepts:** Final scores V=4, D=4, BP=2, L=6. Accepts all gate conditions.

**Dissents on:**
- **F=3:** Maintains F=4–5. "The kill-gate structure is the *definition* of good feasibility engineering. A project that will spend 10% of its budget to resolve 57% of its uncertainty before committing is better-managed than most." *Chair response: Noted. Feasibility scores should reflect both management quality and execution track record. The management is good; the execution record is empty. F=3 is the compromise.*
- **P(any pub)=0.45:** Maintains 0.60–0.68. "The census alone has P(pub) ≈ 0.40 at C&OR/EJOR. Combined with any spectral result, P(pub) ≥ 0.60." *Chair response: The census-only P(pub) estimate of 0.25–0.40 depends on whether the team actually builds and runs the census. With zero code written, discounting for execution risk is appropriate.*
- **P(abandon)=0.40:** Maintains 0.15–0.20. *Chair response: The Synthesizer's low abandon probability assumes the team will pivot to census-only if spectral fails. This is a rational response but assumes team flexibility that hasn't been demonstrated.*

---

## 6. Summary Decision Matrix

```
                    Skeptic    Auditor    Synthesizer    CONSENSUS
Value (V)              3          4            5            4
Difficulty (D)         3          4            4            4
Best-Paper (BP)        1          2            3            2
Laptop CPU (L)         5          6            6            6
Feasibility (F)        2          3            5            3
─────────────────────────────────────────────────────────
Composite            2.8        3.8          4.6          3.8
Verdict            ABANDON*   COND.CONT   COND.CONT   COND.CONT
P(JoC)              0.08       0.30         0.27         0.20
P(any pub)          0.20       0.55         0.68         0.45
P(best-paper)       0.01       0.02         0.03         0.02
P(abandon)          0.55       0.35         0.05         0.40

* Skeptic accepts CONDITIONAL CONTINUE for 2-week pilot only
```

---

## 7. Binding Timeline

```
Week 0  ─── Accept amendments, set up environment
Week 1  ─── G0: Basic tooling operational (SCIP/GCG wrappers + eigensolve)
Week 2  ─── G1: 50-instance pilot, compute ρ ──┬── ρ ≥ 0.4 → CONTINUE
                                                 ├── 0.3 ≤ ρ < 0.4 → REASSESS
                                                 └── ρ < 0.3 → ABANDON
Week 4  ─── G2: Solver wrappers at 80% coverage
Week 8  ─── G3: Spectral > syntactic on 200 instances
Week 14 ─── G4: 500-instance stratified evaluation
Week 18 ─── G5: Draft paper for external review
Week 20 ─── RESOURCE CAP (hard stop)
```

---

*This synthesis is final. The project proceeds under the stated conditions or is abandoned. No renegotiation of gate thresholds is permitted without a full new depth check.*
