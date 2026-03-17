# Verification Report: Final Approach

**Verifier**: Independent Verifier
**Date**: 2026-03-08
**Document Under Review**: `ideation/final_approach.md` — The Collusion Detection Barrier — Certified Algorithmic Audit via Compositional Testing with Automaton-Theoretic Completeness

---

## Verification Checklist

### Structural Completeness

- ✅ **Has all required sections.** Overview, Extreme Value, Architecture, Math Contributions (with CORE/STRETCH), Difficulty, Best-Paper, Hardest Challenge, Evaluation Plan, Risk Assessment, Timeline, Scores — all present and substantive.

- ✅ **Each mathematical contribution has: statement, proof strategy, load-bearing justification, achievability%, person-months.** Verified for M1, C3', M8, M6, M7 (CORE) and M1-broad, C3'-stochastic, M4'-lower (STRETCH). Each has all five required fields. C3' has a particularly detailed proof strategy via Mealy machine / product state space argument.

- ✅ **CORE vs STRETCH separation is explicit and clear.** CORE summary table (line 190–198) lists 5 items with ~75% joint achievability. STRETCH summary table (line 243–248) lists 3 items explicitly labeled as stretch goals with honest "what happens if it fails" sections for each. The separation is unambiguous.

- ✅ **Kill gates are defined with specific conditions.** Three kill gates (KG1 at Month 3, KG2 at Month 5, KG3 at Month 7) with specific pass/fail criteria and explicit consequences (kill, pivot to theory-only, debug/descope). These are genuinely actionable.

### Internal Consistency

- ✅ **LoC estimates are consistent across sections.** Subsystem table sums to exactly 60,000 LoC (12K+8K+4K+10K+9K+5K+2.5K+6.5K+3K). Totals in the table, the "Core MVP ~60K LoC" heading, the MVP scope list, and the score justifications all reference ~60K consistently. Extended scope consistently referenced as ~110–130K LoC.

- ⚠️ **Rust/Python split minor discrepancy.** The final approach states "Rust ~37K / Python ~23K" but summing individual subsystem LoC by language (based on subsystem descriptions and the problem_statement's per-subsystem language assignments) yields closer to Rust ~42K / Python ~18K — the split from the problem_statement. The discrepancy is ~5K, likely from reclassifying some mixed Rust+Python subsystems. Not a structural error, but the split should be reconciled.

- ✅ **Score justifications are consistent with risk assessment.** Value 7/10 deductions align with risk matrix entries (regulatory window, cooperative access). Difficulty 7/10 deductions align with Layer 0 avoiding hardest problems. Potential 7/10 deductions align with M1's formulation-level novelty and M8's "trivially true" risk. Feasibility 6/10 deductions align with the ~75% math completion probability and axiom stabilization risk.

- ✅ **Timeline is consistent with difficulty assessment.** 8-month timeline with 9–14 person-months of CORE math, overlapping with infrastructure (months 2–5). Kill gates at months 3/5/7 provide checkpoint structure. The timeline allows 2–3 months per hard theorem (C3', M1-medium) and 6–8 weeks for axiom stabilization, consistent with the difficulty estimates.

- ✅ **Math program is consistent between CORE list and proof checker dependencies.** The proof checker (S6/M6) encodes axioms about M1 statistical conclusions and C3' collusion-detectability claims. M8 is a standalone impossibility theorem that does not require proof-checker encoding. M7 is an ordering optimization on M1. All dependencies are coherent.

### Honest Self-Assessment

- ✅ **Scores incorporate Skeptic's corrections.** The score justification table (lines 462–468) explicitly shows self-scores, Skeptic's corrections, and final scores. Finals are Value 7 (Skeptic: 6, self: 7–8), Difficulty 7 (Skeptic: 6–8), Potential 7 (Skeptic: 5–7), Feasibility 6 (Skeptic: 4–6). The finals are calibrated between self-assessment and Skeptic, consistently landing at or just above the Skeptic's range — an honest reconciliation, not inflated.

- ✅ **Risk matrix covers the "uncomfortable truth" (H₀-broad trivial power).** Explicitly listed as the first risk entry: "M1 H₀-broad bound is vacuous" at 55–65% probability, Medium impact. The mitigation (tiered hierarchy as primary strategy) is honest. The "what if it happens" section correctly states the fallback contribution.

- ✅ **Kill gates are genuine.** KG1 (Month 3): kills the project if M1 H₀-narrow or basic C3' fails. This is genuine — without these two, there is no contribution. KG2 (Month 5): pivots to theory-only paper if proof checker fails. This is genuine — it abandons the artifact framing. KG3 (Month 7): descope on Type-I inflation. This is genuine — it addresses the statistical validity question directly.

- ✅ **C3 conditionality is properly framed.** The "Design Principle" section (lines 87–91) states: "Soundness is UNCONDITIONAL... Completeness is conditional: proved unconditionally for deterministic bounded-recall automata... conditional on C3 for stochastic strategies." This framing appears throughout: the overview, the math contributions, the best-paper argument, and the scores. Soundness-first is the clear primary guarantee.

- ✅ **Feasibility score accounts for math-engineering coupling risk.** Feasibility 6/10 explicitly lists: "(d) any late-stage C3' gap cascades into proof checker and certificate rework; (e) the Skeptic's valid concern that math-engineering coupling creates schedule risk, though the CORE/STRETCH separation mitigates this."

### Critique Integration

- ✅ **Addresses Skeptic's fatal flaws.** (1) H₀-broad power: tiered null hierarchy as primary strategy, H₀-broad relegated to STRETCH (35–45% achievability). (2) C3 dependency: resolved via deterministic C3' as CORE (85% achievable), stochastic as STRETCH. (3) Evaluation circularity: 4 adversarial red-team scenarios added, Type-I validation seed count increased to 200+ per scenario.

- ✅ **Incorporates Math Assessor's recommendations.** (1) Deterministic C3': adopted as CORE contribution with detailed proof strategy. (2) M8 impossibility: adopted as CORE with A− grade (downgraded from B's A). (3) A+B hybrid strategy: explicitly adopted — "A's achievable scope + B's two strongest theorems + C's engineering practices."

- ✅ **Adopts Approach C's engineering practices.** Phantom-type segment isolation, dual-path rational verification, and interval arithmetic propagation are all described in Hard Subproblems 1 and 2 (lines 253–263) with specific LoC estimates. These are engineering practices within the approach, not paper-level contributions — exactly as both critics recommended.

- ✅ **Avoids Approach B's "5-theorem suicide pact."** The CORE has 5 items but M7 (D grade, 95%+ achievable) and M6 (C grade, 85% achievable) are low-risk. The hard bets are M1, C3', and M8, and even C3' and M8 individually have ≥85% and ≥90% achievability. The 3 STRETCH goals (M1-broad, C3'-stochastic, M4'-lower) are honestly segregated with 35–60% individual achievability. This is structurally different from B's all-or-nothing narrative.

### Differentiation from Portfolio

- ⚠️ **Portfolio overlap check is partially verifiable.** The specific projects named in the checklist (market-manipulation-prover, marl-race-detect, causal-trading-shields, dp-verify-repair, dp-mechanism-forge) do not exist in this pipeline's portfolio. The actual sibling projects in area-064 are:
  - **mechanism-design-auditor**: Audits mechanism design properties — different from collusion detection. No overlap.
  - **walrasian-equilibrium-compiler**: Computes Walrasian equilibria — different from collusion certificates. No overlap.
  - **mech-audit-smt-game-property-verifier** (area-024): SMT-based game property verification — closest potential overlap, but focuses on mechanism design properties verified via SMT solvers, not statistical collusion detection with proof-carrying certificates. Different methodology and domain.

  The final approach does not explicitly address portfolio differentiation. This is a minor gap — the project occupies a unique niche (collusion certification via statistical testing + PCC), but the document should state how it differs from the mechanism-design-auditor sibling.

### Depth Check Binding Conditions

1. ✅ **Tiered oracle access model (Layer 0/1/2) with Layer 0 independently publishable.** Explicit table (lines 52–58) defining three layers with access model, capabilities, and settings. Layer 0 described as "self-contained and independently publishable" (line 58).

2. ✅ **C3-conditional framing — unconditional soundness as primary guarantee.** "Design Principle" section (lines 87–91) and multiple reinforcing statements throughout. Soundness-first framing is consistent and pervasive.

3. ✅ **Scope reduction to ~60K LoC MVP.** Subsystem breakdown targets ~60K LoC as "CollusionProof-Lite (Core MVP)." Extended scope (~110–130K LoC) is clearly separated.

4. ✅ **Three-tier evaluation budget with --smoke mode.** Evaluation section (lines 323–331) defines --smoke (<30 min), --standard (~4 days), --full (~20 days) with specific scenario counts, algorithms, seeds, and CPU-hours for each.

5. ✅ **LoC revision to ~110–130K with honest split.** MVP at ~60K, extended at ~110–130K. Honest breakdown states "~50K novel research code + ~10K essential infrastructure" for the core. The depth check required ~55K novel + rest as infrastructure — the final approach is actually *more conservative* (shifting 5K from "novel" to "infrastructure"), which is the honest direction.

6. ✅ **Tiered null hierarchy (narrow/medium/broad) for practical power.** M1 contribution explicitly defines three tiers with separate achievability estimates (90%, 70%, 35–45%) and explains that "the tiered null hierarchy is the practical power strategy, not a fallback."

7. ✅ **Prove C3 for restricted strategy classes.** C3' for deterministic bounded-recall automata is a CORE contribution with 85% achievability and a detailed proof strategy. Covers grim-trigger, tit-for-tat, Q-learning with discretized Q-tables, and all lookup-table strategies.

8. ✅ **Adversarial red-team scenarios in evaluation.** Ground-truth benchmark suite includes 4 adversarial red-team scenarios: "collusion with injected noise, randomized punishment timing, correlation-mimicking strategies" (line 340). Adversarial evasion rate is a tracked metric.

9. ❌ **Bertrand CP fix — absolute-margin fallback for zero-profit equilibria.** The problem_statement.md addresses this explicitly (delta_p metric for zero-profit Bertrand), and it was binding condition #9 from the depth check. However, the final_approach.md contains **zero mention** of the absolute-margin fallback, delta_p, the zero-profit boundary case, or the Bertrand CP fix. The M5 Collusion Premium is referenced only in the oracle layer table and subsystem S5 description, without addressing the zero-profit edge case. This is a gap: a binding condition is unaddressed in the synthesis document.

10. ✅ **Drop "defining a new subfield" language.** The final approach contains no "defining a new subfield" language. The closest phrasing is "Category creation" in the best-paper argument (line 291), which is a factual claim about the artifact category, not a community-building overclaim. The problem_statement already uses "opening a new direction at the intersection of computational game theory and formal verification," which is the recommended replacement.

---

## Issues Found

### ❌ FAIL: Binding Condition #9 — Bertrand CP Fix Missing

The depth check required an absolute-margin fallback (delta_p) for Collusion Premium computation when competitive equilibrium profit is zero (homogeneous Bertrand). This was binding condition #9 and was flagged as Flaw 5 (LOW-MODERATE) in the depth check. The problem_statement.md addresses it explicitly with a delta_p metric and the statement "for zero-profit competitive equilibria (e.g., homogeneous Bertrand), the system reports an absolute supra-competitive margin delta_p rather than the relative CP." However, the final_approach.md — the synthesis document — does not mention this fix anywhere. Since homogeneous Bertrand is the canonical model of price competition and is one of the two market models in the MVP, this omission is a gap that must be remedied.

**Remediation**: Add a brief statement in the M5/Collusion Premium discussion (or in the subsystem S5 description) specifying the absolute-margin delta_p fallback for zero-profit equilibria, consistent with the problem_statement's formulation.

---

## Concerns Noted

### ⚠️ Rust/Python Split Minor Discrepancy

The final approach states "Rust ~37K / Python ~23K" while the problem_statement states "Rust ~42K / Python ~18K" for the same ~60K MVP. The individual subsystem LoC are consistent; only the aggregate language split differs. This likely reflects reclassification of mixed-language subsystems (S2, S5, S7, S9). Not a structural issue, but the two documents should be reconciled to avoid confusion during implementation planning.

### ⚠️ Novel Research Code Split Discrepancy

The final approach states "~50K novel research code + ~10K essential infrastructure" while the problem_statement states "~55K novel research code + ~5K essential infrastructure" for the core 60K. The final approach is more conservative (honest direction), but the inconsistency between the two documents could cause confusion. Recommend aligning to the final approach's numbers since it is the governing document.

### ⚠️ Portfolio Differentiation Not Explicit

The final approach does not discuss differentiation from sibling projects (mechanism-design-auditor, walrasian-equilibrium-compiler) or related projects in other areas. While no actual overlap exists, an explicit differentiation statement would strengthen the document, particularly since mechanism-design-auditor operates in a conceptually adjacent space (formal verification of game/mechanism properties).

### ⚠️ Extended Scope Lacks Detailed LoC Breakdown

The final approach references ~110–130K for the extended scope (lines 432–438) but only lists features without a per-subsystem LoC table. The problem_statement has this table. For implementation planning, the final approach should either include the table or explicitly reference the problem_statement for the extended breakdown.

### ⚠️ M5 (Collusion Premium) Not Listed as CORE Math Contribution

The Collusion Premium quantification (M5) is referenced in the architecture and evaluation but is not listed in the CORE or STRETCH math summary tables. M5 appears only implicitly in the Layer table ("partial M5 Collusion Premium" for Layer 1, "tight M5 bounds" for Layer 2). The depth check and problem_statement treat M5 as a distinct contribution. Either M5 should be explicitly listed (even at Grade C) or its absorption into other subsystems should be noted.

---

## Strengths

1. **The A+B+C hybrid is well-executed.** The synthesis genuinely captures Approach A's achievable scope, Approach B's strongest theorems (deterministic C3' and M8), and Approach C's engineering practices (phantom types, dual-path verification, interval arithmetic) — exactly as both critics independently recommended. The integration is seamless, not mechanical.

2. **The CORE/STRETCH separation is the document's best structural feature.** The CORE math program (M1 narrow/medium, C3' deterministic, M8, M6, M7) has ~75% joint achievability and independently constitutes a strong EC submission. The STRETCH goals (M1-broad, C3'-stochastic, M4'-lower) are honestly labeled with candid "what happens if it fails" sections. This is genuinely good risk management — the project cannot fail into irrelevance.

3. **The C3'/M8 dichotomy is the paper's strongest narrative element.** "Collusion by bounded-recall automata is certifiably detectable; collusion by unrestricted strategies is provably undetectable" — this is clean, quotable, and structurally appealing. The framing converts bounded recall from a limitation into a deep insight.

4. **Kill gates are real.** KG1 at Month 3 actually kills the project — not adjusts, not pivots, but kills — if the foundation (M1 H₀-narrow or basic C3') doesn't hold. This is rare honesty in research planning.

5. **The score calibration is honest.** Composite 6.75 with Feasibility at 6/10 is a sober self-assessment that respects the Skeptic's deflation without capitulating to it. The 55–65% EC acceptance probability and 15–20% best-paper probability are realistic ranges.

6. **The evaluation plan is thorough and addresses circularity.** Three-tier budget, 4 adversarial red-team scenarios, 200+ seeds per competitive scenario for Type-I validation, and tracked adversarial evasion rates directly address the Skeptic's evaluation circularity concern.

---

## Final Verdict

### APPROVE WITH CONDITIONS

The final approach is a high-quality synthesis that successfully integrates critiques from the adversarial debate, adopts the depth check's binding conditions, and presents an honest, achievable research plan. One binding condition (Bertrand CP fix) is missing from the document, and several minor inconsistencies exist between the final approach and the problem statement. These are readily remediable.

### Binding Conditions for Approval

1. **Add the Bertrand CP fix (absolute-margin delta_p fallback) to the final approach document.** This is depth check binding condition #9 and affects the MVP's core market model. A 2–3 sentence addition in the architecture or M5 discussion suffices.

2. **Reconcile the Rust/Python split** (37K/23K vs 42K/18K) between the final approach and problem statement. Choose one and update the other.

### Non-Binding Recommendations

3. Add explicit portfolio differentiation statement (1–2 sentences distinguishing from mechanism-design-auditor and walrasian-equilibrium-compiler).

4. Either list M5 explicitly in the math summary or note its absorption into S5.

5. Include or cross-reference the extended scope LoC table for implementation planning.
