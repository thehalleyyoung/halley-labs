# Theory Stage Synthesis: Coverage-Certified XR Accessibility Verifier

**Author:** Team Lead (Phase 2 Cross-Critique + Phase 3 Synthesis)
**Date:** 2026-03-08
**Status:** Binding synthesis — all proposals reviewed, disagreements adjudicated
**Input documents:**
1. `proposal_formal_methods.md` — Definitions, theorems, proofs (Formal Methods Lead)
2. `proposal_algorithms.md` — Pseudocode and complexity (Algorithm Designer)
3. `proposal_empirical.md` — Evaluation plan (Empirical Scientist)
4. `proposal_redteam.md` — Adversarial attacks (Red-Team Reviewer)
5. `proposal_verification.md` — Quality criteria (Verification Chair)

---

## Part 1: Cross-Critique Summary

### 1.1 Formal Methods vs. Red-Team

**Critique 1 (Attack 1.1 → Def 5): Lipschitz fails at joint limits; is piecewise-Lipschitz sufficient?**

The Red-Team identifies that joint-limit cutoffs create step-function discontinuities in the accessibility function — L = ∞ at every transition. The Formal Methods proposal *anticipated* this (Def 5 explicitly introduces piecewise-Lipschitz partitioning), and the C1 proof structure already operates on smooth cells between joint-limit surfaces.

However, the Red-Team raises a *residual* concern the FM proposal does not fully address: the excluded ε-neighborhoods of joint-limit surfaces form (d−1)-dimensional hypersurfaces in d-dimensional Θ. For d = 5 body parameters, these are 4-dimensional surfaces. The excluded volume scales as:

    μ(∪_l Nε(H_l)) / μ(Θ) ≤ Σ_l 2ε_l · Area(H_l ∩ Θ) / μ(Θ)

With 2mn = 2 × 30 × 7 = 420 potential surfaces (FM Open Question 4), the excluded volume depends critically on ε_l (neighborhood width). If ε_l ≈ 0.02 in normalized units and each surface spans ~50% of the remaining dimensions, the excluded fraction could reach 10–30% — not negligible.

The FM proposal's κ-completeness metric (Def 7) correctly tracks this, but the Red-Team is right that κ could dominate the certificate's practical utility. The wheelchair-user scenario (Attack 1.1 concrete failure) is particularly damning: the populations most affected by accessibility failures live *on* the joint-limit surfaces the certificate exempts.

**Assessment:** Piecewise-Lipschitz is *necessary* but not *sufficient*. The FM formulation is structurally correct, but the synthesis must mandate that κ ≤ 0.10 for a certificate to be reported as "full coverage." Certificates with κ > 0.10 must be labeled "partial" and the excluded body-parameter regions must be reported explicitly in the output. This resolves the Red-Team's concern without requiring the FM proof to be restructured.

---

**Critique 2 (Attack 1.2 → C1/C3): ε ≈ 0.04–0.06 is barely useful.**

The Red-Team's back-of-envelope is methodologically sound:
- 4M samples over 3⁵ = 243 strata → ~16,500 per stratum
- Hoeffding with Bonferroni correction: ε ≈ √(ln(2 × 243 × 30 × 3 / 0.01) / (2 × 16,500)) ≈ 0.043

The FM proposal's C3 frontier-resolution model (§C3, "Enhanced model") claims ε_enhanced can be much lower by crediting SMT queries with frontier-resolution rather than volume subtraction. The Algorithms proposal (Algorithm 5, lines 24–28) implements this model. But the Red-Team correctly notes that the volume-subtraction contribution of SMT is negligible (10⁻¹² per query × 6,000 queries ≈ 10⁻⁸ total).

The key question is whether frontier-resolution is *real* or merely *theoretical*. The FM proposal acknowledges this as Open Question 3 but does not resolve it. The Algorithms proposal implements it (Algorithm 4, Phase 5, lines 32–41) but the effective_radius = Δ_max / L̂ computation depends on L̂ being well-estimated — which the Red-Team attacks in 1.4.

**Assessment:** The Red-Team's ε ≈ 0.04–0.06 estimate for sampling-only is accepted as the realistic baseline. The frontier-resolution enhancement is a plausible mechanism for reaching 3–5× improvement (ε ≈ 0.01–0.02), but it is *unproven* and must be validated empirically at gate D3. The target is revised from ε < 0.01 (FM/Empirical) to ε < 0.05 as the *minimum* and ε < 0.02 as the *target*. This aligns with Verification Chair COND-6 (accept 3–5× if ≥ 1×).

---

**Critique 3 (Attack 1.4 → C1 Assumption M2): Lipschitz constant estimation is circular.**

The Red-Team identifies a genuine circularity: estimating L requires samples near the frontier, finding the frontier requires knowing L, and knowing L requires samples. The FM proposal's mitigation (cross-validation in Algorithm 2, Phase 5) validates *average* L but not *worst-case* L. The Red-Team's adversarial scene (99 smooth buttons, 1 near-discontinuous) directly defeats the estimator.

The Red-Team proposes a structural fix: derive an analytical L bound from the kinematic Jacobian. The FM proposal's C2 provides exactly this Jacobian structure (J_θ, J_q) but does not connect it to an L bound.

**Assessment:** Cross-validation is *necessary* (as a consistency check) but *not sufficient* (as a sound bound). The synthesis mandates:
1. Derive an analytical L_max from C2's Jacobian: L_max = ||J_θ||_op · max_link_sensitivity. This provides a provably conservative bound.
2. Report *two* ε values per certificate: ε_analytical (using L_max, provably sound but loose) and ε_estimated (using L̂ from cross-validation, tighter but not provably sound).
3. The CAV paper leads with ε_analytical for the soundness theorem; ε_estimated is a practical enhancement disclosed separately.

This eliminates the circularity for the formal result while preserving the practical tightness.

---

### 1.2 Algorithms vs. Red-Team

**Critique 4 (Algorithm 3 confirms Attack 1.2/2.2): SMT volume is negligible.**

Algorithm 3's complexity analysis (§3.4) independently computes: vol_query / vol_Θ ≈ (0.045/π)⁷ × (0.045)⁵ ≈ 10⁻¹² per query. With 6,000 queries: total volumetric coverage ≈ 6 × 10⁻⁹. The Algorithm Designer explicitly writes: "This confirms the Red-Team's concern."

This is the single most important cross-validation in the synthesis. Two independent analyses (Red-Team back-of-envelope, Algorithm Designer exact computation) agree: **SMT volume subtraction is a rounding error.** The entire "sampling-symbolic hybrid" framing collapses if the symbolic component's value is purely volumetric.

**Assessment:** The SMT value proposition must be reframed. It is *not* about volume elimination. It is about:
1. **Frontier resolution** (Algorithm 5 model): pinpointing where the accessibility boundary lies within a local neighborhood, enabling Lipschitz interpolation to tighten ε in the surrounding region.
2. **Proof certificates**: providing machine-checkable proofs for *specific* contested sub-regions (e.g., "this button IS reachable for all bodies in the 40th–60th percentile range" backed by an SMT proof).
3. **Structural understanding**: the SMT query's sat/unsat answer reveals the *reason* for accessibility/inaccessibility (which joint limit? which orientation constraint?), enabling actionable counterexamples.

The Red-Team's recommendation (Attack 2.2 mitigation #2) to use Tier 1 affine-arithmetic envelopes as the "symbolically verified volume" is compelling and is adopted (see Part 4, Decision 7).

---

**Critique 5 (Algorithm 5 frontier-resolution model): Does this save the 5× improvement claim?**

Algorithm 5 introduces the "frontier-resolution" ε enhancement (lines 24–31):

    resolution_factor = max(0, 1 − N_smt × Δ_resolved_per_query / Δ_frontier)

Where Δ_resolved_per_query = 2 × Δ_max ≈ 0.09 rad ≈ 5° of frontier. If the total frontier length (in the appropriate metric) is Δ_frontier ≈ 1000 × Δ_max (covering ~1000 linearization-width segments), then 6,000 queries resolve 6,000 / 1,000 ≈ 6× the frontier. This gives resolution_factor ≈ 0, meaning frontier is fully resolved, and ε_enhanced could approach zero.

But this model has multiple assumptions the Red-Team would attack:
- Δ_frontier must be estimable (it depends on the true frontier geometry, which is unknown)
- Queries must be placed *at* the frontier (requires accurate frontier detection from sampling)
- The Lipschitz interpolation from a resolved frontier point must actually tighten the bound in the surrounding region (requires L to be finite in that region — back to the Lipschitz assumption)

**Assessment:** Frontier-resolution is the *most promising* mechanism for meaningful SMT contribution, but its theoretical foundation is incomplete. The FM proof of C1 does not include a frontier-resolution theorem. The synthesis mandates:
1. Add a **Lemma B2** to the FM framework: "Frontier Resolution Improvement Bound" — formalizing the relationship between SMT queries at the frontier and the effective ε reduction in surrounding regions, under piecewise-Lipschitz.
2. This lemma must be *conditional* — it depends on L being known/bounded in the resolved neighborhood.
3. If B2 cannot be proven at the theory stage, frontier-resolution is classified as an *empirical enhancement* and gate D3 tests whether it works in practice. The CAV paper's headline result (C1 soundness) remains independent of B2.

---

**Critique 6 (Algorithm 6 multi-step complexity): 3-step limit is confirmed.**

Algorithm 6 §6.2 computes the curse of dimensionality:
- k=1: d_eff = 5 → manageable
- k=2: d_eff = 19 → sparse but feasible
- k=3: d_eff = 26 → very sparse, ε > 0.1 likely

The Empirical proposal's H6 tests ε < 0.1 on 75% of 3-step scenes with 15-minute budget. The Algorithm Designer estimates 100s for k=3 sampling — within budget.

The Red-Team's Attack 1.3 notes that 3⁷ = 2,187 strata at d=7 already strains the budget; at d=26 with even 2 strata per dimension, 2²⁶ ≈ 67M strata — obviously impossible with uniform stratification.

**Assessment:** Multi-step verification cannot use the same stratification scheme as single-step. The synthesis mandates:
1. For k ≥ 2, use *marginal* stratification: stratify over Θ (body params, d=5) only, not over the full trajectory space T^(k×7).
2. Joint-angle sampling is *conditional*: for each body θ, sample IK solutions (Algorithm 6 lines 9–21) rather than uniformly sampling the full trajectory space.
3. The ε guarantee for multi-step is *weaker* than single-step: ε < 0.10 for k=2, ε < 0.15 for k=3. If D5 fails even at these relaxed targets, restrict to k ≤ 2 (Verification Chair COND-3).

---

### 1.3 Empirical vs. Verification Chair

**Critique 7: Does the evaluation plan satisfy PT-1 through PT-10?**

| Criterion | Assessment | Notes |
|-----------|------------|-------|
| PT-1 (≥50KB content) | Not yet testable (no paper.tex yet) | Evaluation plan alone is ~47KB — promising |
| PT-2 (precise definitions) | **PASS** | Metrics in §4 are precisely defined with formulas |
| PT-3 (self-contained theorems) | Deferred to FM | Eval plan doesn't contain theorems |
| PT-4 (proof completeness ∝ novelty) | Deferred to FM | — |
| PT-5 (falsifiable hypotheses) | **PASS** — 6 hypotheses (H1–H6) with quantitative thresholds, tests, and kill criteria | Exceeds the minimum of 3 |
| PT-6 (statistical methodology) | **PASS** — power analysis, Bonferroni correction, effect sizes, Clopper-Pearson CIs | Thorough |
| PT-7 (honest related work) | Partially addressed | Eval plan doesn't cover related work directly |
| PT-8 (load-bearing) | **PASS** | Every hypothesis maps to a kill gate |
| PT-9 (notation) | **PASS** | Notation table in §0 |
| PT-10 ("so what?" test) | Partially | Eval plan supports both CAV (ε comparison) and UIST (TPR/FPR) framing |

**Assessment:** The Empirical plan is the strongest of the five proposals. It exceeds Verification Chair requirements on statistical methodology and falsifiability.

---

**Critique 8: Are the falsifiable hypotheses compatible with the formal analysis?**

| Hypothesis | FM Compatibility | Red-Team Compatibility |
|------------|-----------------|----------------------|
| H1 (5× over CP) | C3 predicts frontier-resolution can achieve this | Red-Team (Attack 1.2) predicts ~1.4× from volume alone; frontier-resolution untested |
| H2 (wrapping ≤ 5×) | B1 predicts 1.15–1.71 for ≤30° | Red-Team (Attack 3.1) predicts 4–7× for 7-joint ±60° |
| H3 (Tier 1 TPR > 95%) | C4 bounds detection as f(w, margin) | Red-Team agrees detection is good; FPR is the concern |
| H4 (Tier 2 TPR > 97%) | C1 supports this if ε is tight enough | Red-Team agrees sampling catches bugs; SMT adds certification, not detection |
| H5 (≥10% marginal over MC) | FM has no direct prediction | Red-Team (Attack 4.3) predicts <5% against strong MC |
| H6 (multi-step ε < 0.1) | C1 extends to k-step; complexity is O(d^k) | Algorithm 6 confirms feasibility for k ≤ 3 |

**Key tension:** H1 (5× improvement) and H5 (10% marginal detection) face conflicting predictions between FM/Algorithms (optimistic) and Red-Team (skeptical). The synthesis resolution is in Part 2.

---

**Critique 9: Is the statistical methodology sound?**

The Empirical plan's statistical framework is well-constructed:
- **Appropriate tests:** Wilcoxon signed-rank for paired ε comparisons (H1), exact binomial for rates (H3–H6), one-sample t-test on log-transformed wrapping (H2).
- **Multiple comparison correction:** Bonferroni across 8 hypothesis tests (conservative but safe).
- **Power analysis:** 400 bugs for TPR, 600 elements for FPR — well-powered for the stated effect sizes.
- **Cross-validation:** 80/20 split for L̂ estimation, 20% holdout for benchmark suite.

One concern: **H1's paired Wilcoxon is appropriate only if the ε_cert and CP values come from the same samples on the same scenes.** The Empirical plan confirms this (§1, H1 procedure: "paired observations"). Sound.

Second concern: **H5's binomial test on marginal detection has low power if MC misses few bugs.** The plan estimates 20–60 MC-missed bugs — at the low end (20), the test has only 80% power to detect MDR = 25% (not 10%). The Empirical plan acknowledges this implicitly but should state the power curve.

**Assessment:** Methodology is sound with minor power concerns on H5. No changes required.

---

### 1.4 Formal Methods vs. Algorithms

**Critique 10: Are complexity bounds consistent with theorem statements?**

| Theorem | Algorithm | Consistency |
|---------|-----------|-------------|
| C2 (Δ_max ∝ √(η / C_FK·L_sum)) | Algorithm 3 lines 9–11 | **Consistent.** Algorithm uses exact same formula. C_FK = n/2 = 3.5 for 7-DOF. |
| C4 (wrapping w ≤ ∏(1 + c_i·Δ_i²)) | Algorithm 1 (AffineFK) | **Consistent.** B1's bound is the analytical prediction; Algorithm 1 implements the computation. |
| C1 (n_min per stratum) | Algorithm 2 (sampling) | **Consistent.** Algorithm 2 Phase 2 allocates samples per stratum matching C1's formula. |
| C3 (budget allocation) | Algorithm 5 | **Consistent.** Algorithm 5 implements C3's optimization. |

**Critique 11: Does C_FK = n/2 in C2 match Algorithm 3's Δ_max?**

Algorithm 3 line 9 computes: Δ_q = √(η / (2 × C_FK × L_sum)). With C_FK = 3.5 (n/2 for n=7), L_sum = 0.7m, η = 0.005m (half of 1cm margin):

    Δ_q = √(0.005 / (2 × 3.5 × 0.7)) = √(0.005 / 4.9) = √(0.00102) ≈ 0.032 rad ≈ 1.8°

This is *tighter* than the FM proposal's estimate of Δ_q ≈ 2.6° (which used η = 0.01m, the full margin). The discrepancy is because Algorithm 3 line 8 sets η = margin × 0.5, consuming half the margin for soundness. This is conservative and correct.

FM Open Question 1 asks whether C_FK = n/2 is tight. If the true constant is n/4, Δ_max doubles, and each SMT query covers 2^12 ≈ 4096× more volume. This is a potential empirical win but cannot be resolved at the theory stage.

**Assessment:** FM and Algorithms are internally consistent. C_FK = n/2 is conservative; empirical measurement may improve it.

---

## Part 2: Resolved Disagreements

### Disagreement 1: ε Achievability

| Position | Source | Claim |
|----------|--------|-------|
| Optimistic | FM (C1 + C3) | ε < 0.01 achievable with frontier-resolution |
| Pessimistic | Red-Team (Attack 1.2) | ε ≈ 0.04–0.06 realistic; 5× over CP impossible |
| Middle | Algorithms (Alg. 5) | ε depends on frontier-resolution model validity |

**Resolution:** The sampling-only ε baseline is ε ≈ 0.04–0.06 (Red-Team is correct for this scenario). The frontier-resolution model (FM + Algorithms) is a *plausible but unproven* mechanism that could reduce ε to 0.01–0.02.

**Synthesis decision:**
- **Hard target:** ε < 0.05 (must achieve or gate D3 fails)
- **Stretch target:** ε < 0.02 (requires frontier-resolution to work)
- **ε < 0.01 is dropped** as a first-pass target for 10-minute budgets. It may be achievable with longer budgets (report ε vs. budget curve per Red-Team recommendation).
- The Empirical plan's H1 is revised: median(ρ) ≤ 0.33 (3× improvement) as the hard pass, median(ρ) ≤ 0.20 (5×) as the stretch goal.

---

### Disagreement 2: Lipschitz Handling

| Position | Source | Claim |
|----------|--------|-------|
| FM | Def 5, C1 | Piecewise-Lipschitz with explicit partition; κ-completeness tracks excluded volume |
| Red-Team | Attack 1.1 | (d−1)-dimensional exclusion zones can dominate; wheelchair users live on these surfaces |

**Resolution:** Both are correct. Piecewise-Lipschitz is the right formal framework, AND the excluded volume can be large for populations near joint limits.

**Synthesis decision:**
- Accept piecewise-Lipschitz formulation for C1 (FM position accepted).
- Mandate κ ≤ 0.10 threshold for "full coverage" certificates. Certificates with κ > 0.10 are labeled "partial" and must list the excluded body-parameter regions (Red-Team concern addressed).
- Add *focused sampling* within the ε-neighborhoods of joint-limit surfaces: instead of excluding these regions entirely, sample them densely and report a separate ε_boundary for the boundary regions. This is a new mechanism not in any individual proposal.
- For the CAV paper: present the κ-completeness bound as a *feature*, not a limitation. The certificate explicitly quantifies what it covers and what it doesn't — this is more honest than MC (which implicitly ignores frontier structure).

---

### Disagreement 3: 5× Improvement over Clopper-Pearson

| Position | Source | Claim |
|----------|--------|-------|
| FM + Algorithms | C3, Algorithm 5 | Frontier-resolution model predicts 3–10× improvement |
| Red-Team | Attacks 1.2, 2.2 | Volume subtraction gives ~1.4×; frontier-resolution is speculative |

**Resolution:** The Red-Team is correct that *volume subtraction alone* gives ~1.4× (from 30% SMT coverage reducing ρ to 0.7). The 5× claim depends entirely on frontier-resolution.

**Synthesis decision:**
- The 5× claim is *conditional* on frontier-resolution working. It is NOT a guaranteed feature of the certificate framework.
- Gate D3 is the definitive test. If median(ρ) > 0.33 (less than 3× improvement), the certificate's quantitative advantage over CP is marginal, and the paper must reframe around *structural* advantages (spatial map, counterexample generation, proof certificates) rather than ε tightness. This is Verification Chair COND-6.
- The FM proposal's C3 is revised to present *both* models: volume-subtraction (proven, ~1.4×) and frontier-resolution (conjectured, 3–10×). The paper leads with the proven result and presents frontier-resolution as an empirically validated enhancement.
- The Red-Team's suggestion (Attack 2.2 mitigation #3) to use Tier 1 affine-arithmetic envelopes as symbolically verified volume is adopted as a third improvement mechanism (see Decision 7). This could provide 30–60% verified volume without any SMT queries, giving 1.7–2.5× improvement from Tier 1 alone.

---

### Disagreement 4: Wrapping Factor

| Position | Source | Claim |
|----------|--------|-------|
| FM (B1) | Lemma B1 | w ≈ 1.15–1.71 for k=7, Δ=30°–60° |
| Red-Team | Attack 3.1 | w ≈ 4–7× realistic for 7-joint with wide ranges; bilinear product wrapping dominates |

**Resolution:** Both estimates are valid for different scenarios. B1's bound is *per-joint* multiplicative and doesn't fully account for bilinear cross-term accumulation in the matrix chain. The Red-Team's 4–7× estimate includes bilinear effects but is based on extrapolation from published affine-arithmetic benchmarks (not specific kinematic computation).

**Synthesis decision:**
- Accept B1's formula as the *analytical* prediction. Accept Red-Team's 4–7× as the *empirical* expectation range.
- Gate D1 must test **both** scenarios:
  - D1a: 4-joint chain, ±30° (B1 predicts w ≈ 1.15; should easily pass w ≤ 5)
  - D1b: 7-joint chain, realistic ranges (shoulder ±60°, elbow ±70°, wrist ±45°). Red-Team predicts w ≈ 4–7×.
- If D1a passes but D1b fails (w > 5×): implement joint-angle subdivision (Red-Team suggestion in Attack 3.1 mitigation #3). Split shoulder range into 2 sub-ranges, doubling evaluation count but halving wrapping for that joint. This brings the 7-joint case back under control.
- If D1b fails even with subdivision (w > 10×): fall back to Taylor models (Verification Chair COND-4). Add ~2K LoC, 3× slower Tier 1, still < 5s.

---

## Part 3: Unresolved Issues (Honest)

These issues **cannot be resolved at the theory stage** and must be tested empirically:

### 3.1 Is Frontier-Resolution SMT Real or Theoretical?

The frontier-resolution model (Algorithm 5, FM C3 enhanced model) predicts that SMT queries at the accessibility frontier provide information-theoretic value far exceeding their volumetric coverage. This is *plausible* — resolving the boundary location allows Lipschitz interpolation to tighten ε nearby — but the actual magnitude of improvement is unknown.

**Why it can't be resolved theoretically:** The improvement depends on the empirical structure of accessibility frontiers (their smoothness, dimensionality, and how well they're approximated by piecewise-linear surfaces). No closed-form characterization of frontier structure exists for general XR scenes.

**Resolution path:** Gate D3 (Month 2). Run the certificate pipeline with and without frontier-resolution accounting on 100 benchmark scenes. If the improvement is < 1.5× over sampling-only, frontier-resolution is reclassified as insignificant and the paper's SMT contribution is reframed.

### 3.2 Is Piecewise-Lipschitz Partitioning Tractable?

Def 5 requires identifying all joint-limit transition surfaces. For a 30-element scene with 7 joints: up to 420 surfaces. The Algorithm proposal (Algorithm 4, Phase 1) provides pseudocode for detection, but the question is whether detection has acceptable false-negative rates.

**Why it can't be resolved theoretically:** Detection relies on finding critical angles q_j*(e) for each element — the joint angle at which a specific joint becomes the limiting factor for reaching element e. This requires solving an optimization problem per (element, joint) pair. Whether the optimization reliably finds all critical angles depends on the FK landscape's complexity.

**Resolution path:** Gate D4 (Month 3). Measure violation frequency and detection rate on 100+ benchmark scenes. If undetected violations > 20%, the κ-completeness guarantee is unreliable, and the certificate must include a "detection confidence" factor (weakening ε further).

### 3.3 What Is the Actual ε on Benchmark Scenes?

All ε estimates in this synthesis are analytical predictions. The actual ε depends on:
- Empirical wrapping factor (affects Tier 1 frontier estimates, which seed Tier 2)
- Empirical L values (affect the Hoeffding bound's tightness)
- Empirical frontier-resolution effectiveness (affects ε_enhanced)
- SMT timeout rates (affect the volume of verified regions)

**Resolution path:** Gate D2 (Month 2). Prototype the certificate engine on 10-object scenes and measure ε directly. If ε > 0.10 on simple scenes, the certificate framework needs fundamental revision or abandonment.

### 3.4 Is the Analytical L Bound from Kinematics Useful?

Part 1 (Critique 3) mandated deriving L_max from C2's Jacobian. The resulting bound may be extremely conservative — if L_max is 10× the empirical L̂, then ε_analytical ≈ 10× ε_estimated, potentially exceeding 0.5 (vacuous).

**Why it can't be resolved theoretically:** The gap between analytical L_max and empirical L̂ depends on how much of the Jacobian's maximum is actually realized in practice. This is scene-dependent.

**Resolution path:** Compute L_max analytically for the 50th-percentile 7-DOF arm. Compare with empirical L̂ from 100 benchmark scenes. If L_max / L̂ > 5×, the dual-ε reporting strategy (Critique 3 resolution) is necessary. If L_max / L̂ > 20×, ε_analytical is vacuous and the CAV paper must weaken its formal guarantee to ε_estimated + "consistent with cross-validation" rather than "provably sound."

### 3.5 What Is the Actual Wrapping Factor on 7-Joint Chains?

B1 and the Red-Team disagree on the magnitude of wrapping for realistic configurations (see Disagreement 4). The true value depends on:
- Specific joint configurations (near-singular vs. generic)
- Implementation quality (how tight the Chebyshev approximations are)
- Whether bilinear cross-terms dominate over single-joint wrapping

**Resolution path:** Gate D1 (Month 1). This is the *first* empirical test and has the earliest kill potential.

### 3.6 How Many Joint-Limit Transition Surfaces Exist in Practice?

FM's bound of 2mn (420 for a typical scene) is a worst-case count. Many surfaces may not intersect Θ_target, and many may be geometrically redundant (the same joint is never limiting for multiple elements at similar body sizes).

**Resolution path:** Gate D4. Measure on benchmark scenes.

---

## Part 4: Synthesis Decisions

### Decision 1: C1 Formulation — Piecewise-Lipschitz with Explicit Violation Tracking

**Accepted.** C1 is stated for piecewise-Lipschitz frontiers (Def 5), not global Lipschitz. The proof structure (union bound + per-cell Hoeffding + volume subtraction) operates identically within each smooth cell. Joint-limit transition surfaces define the cell boundaries.

**Rationale:** The Red-Team (Attack 1.1) demonstrated that global Lipschitz is unrealistic. The FM proposal already anticipated this with Def 5. The piecewise formulation is both mathematically sound and practically necessary. It also satisfies Verification Chair CONT-8 (rigorous Lipschitz treatment).

**Implication:** C1's ε guarantee is *conditional* on the smooth region Θ_smooth. The certificate explicitly reports κ (excluded fraction). This is stricter than the original C1 statement but more honest.

---

### Decision 2: SMT Value Model — Frontier-Resolution (Conditional)

**Conditionally accepted.** The frontier-resolution model (Algorithm 5, C3 enhanced) is the proposed mechanism for SMT contribution. However, it is classified as an *empirical enhancement* until validated at gate D3, not a *theorem*.

**Rationale:** Both the Algorithm Designer and Red-Team agree that volume subtraction is negligible (10⁻⁹). The only viable SMT value model is frontier-resolution. But frontier-resolution lacks a formal proof (it's a heuristic model in Algorithm 5). The synthesis mandates adding Lemma B2 (frontier-resolution bound) if possible at the theory stage; otherwise, D3 is the empirical gate.

**Implication:** The CAV paper's C1 theorem does NOT depend on frontier-resolution. C1 is sound with sampling alone (ε_sampling). Frontier-resolution is a practical enhancement reported separately.

---

### Decision 3: Tier 1 Strategy — Subdivision with Max 6 Levels

**Accepted.** Algorithm 1b's subdivision strategy (max_sub = 6, selecting worst-axis per iteration) is the Tier 1 approach. Maximum 2⁶ = 64 sub-problems per element.

**Rationale:** B1 predicts manageable wrapping (1.15–1.71) for ≤30° ranges. Subdivision reduces wrapping super-exponentially. At 6 levels: effective wrapping < 1.05 per the FM proposal's analysis. Even the Red-Team's pessimistic 4–7× estimate is for *unsub-divided* chains. With subdivision, the gap narrows.

**Modifications:**
- Add *joint-angle subdivision* (not just body-parameter subdivision) for joints with ranges > 45°. This doubles the evaluation count per subdivided joint but is necessary for shoulder (±60°) and elbow (±70°). Algorithm 1b line 12 already supports this.
- Set max_wrapping = 5 as the target (Algorithm 1b line 2). If a sub-problem still exceeds w = 5 after 6 subdivisions, it is flagged as "inconclusive" (not green or red).

---

### Decision 4: Multi-Step Limit — k ≤ 3

**Accepted.** Multi-step interaction verification is restricted to k ≤ 3 steps. Sequences with k > 3 are reported as "not certifiable."

**Rationale:** Algorithm 6 §6.2 shows the curse of dimensionality: d_eff = 5 + 7k. At k = 3, d_eff = 26. The Red-Team's Attack 1.3 confirms that stratified sampling in 26D is barely feasible (requires marginal stratification, not full-factorial). The Empirical plan's H6 tests ε < 0.1 for k = 3. The Verification Chair's COND-3 permits restricting to k ≤ 2 if D5 fails.

**Modifications:**
- Multi-step uses *marginal stratification* over Θ (d=5) only, with conditional IK sampling for joint angles (per Algorithm 6's approach). This avoids the infeasible 2²⁶-stratum stratification.
- ε targets are relaxed: ε < 0.05 (k=1), ε < 0.10 (k=2), ε < 0.15 (k=3). These are reported as separate certificate tiers.

---

### Decision 5: Budget Allocation — 95%+ to Sampling, ≤5% to SMT (Default)

**Accepted with modification.** Algorithm 5's analysis shows the optimal SMT fraction is α ≈ 0.05% under volume-subtraction and α ≈ 10–20% under frontier-resolution. Since frontier-resolution is unproven, the default is conservative: 95% sampling, 5% SMT.

**Modification:** Make α *adaptive* at runtime. Start with α = 5%. After 50 SMT queries, measure the effective frontier-resolution rate. If it exceeds the volume-subtraction model by > 2×, increase α toward the frontier-resolution optimum. This is a simple online optimization that hedges between the two models.

---

### Decision 6: Certificate Format — Include κ-Completeness Metric

**Accepted.** Every certificate C = ⟨S, V, U, ε, δ, κ⟩ includes the κ-completeness metric.

**Rationale:** This is the FM proposal's mechanism for honestly reporting the excluded volume. The Red-Team demanded it. The Verification Chair's CONT-8 requires it. It is the key difference between the coverage certificate and a naive MC bound: the certificate *knows* what it doesn't cover.

**Certificate tiers (new):**
- **Full certificate** (κ ≤ 0.10): Coverage guarantee applies to ≥ 90% of the target population.
- **Partial certificate** (0.10 < κ ≤ 0.30): Significant frontier exclusions. Excluded regions listed explicitly.
- **Weak certificate** (κ > 0.30): Certificate covers < 70% of the population. Formal guarantee is of limited practical value.

---

### Decision 7: Tier 1 Envelopes as Symbolically Verified Volume (NEW)

**Adopted from Red-Team recommendation.** When Tier 1 classifies an element as "definitely reachable" (green) for a body-parameter range [θ_min, θ_max], this constitutes a *symbolically verified region* backed by affine-arithmetic soundness. These green regions are credited to the verified set V in the coverage certificate.

**Rationale:** The Red-Team's Attack 2.2 mitigation #3 observes that Tier 1 already computes conservative envelopes for the entire parameter space. If 40% of (element, body-range) pairs are classified green by Tier 1, then |V|/|Θ| ≈ 0.40 without any SMT queries. This provides a 1/(1−0.40) ≈ 1.67× ε improvement from volume subtraction alone — comparable to or exceeding the optimistic SMT contribution.

**Implications:**
- Algorithm 4 (Certificate Assembly) is modified to include Tier 1 green regions in the verified set V.
- The "sampling-symbolic" hybrid is rebranded as "sampling + interval verification + targeted SMT." The interval verification (Tier 1) provides the bulk of symbolic coverage; SMT provides frontier-resolution.
- This is the single most impactful cross-proposal improvement identified in the synthesis.

---

### Decision 8: Analytical L Bound as Formal Backbone (NEW)

**Adopted from Red-Team recommendation + FM proposal synthesis.** Derive L_max analytically from C2's kinematic Jacobian structure. Report two ε values:
- ε_analytical: provably sound, using L_max (may be loose)
- ε_estimated: tighter, using cross-validated L̂ (not provably sound)

**Rationale:** Eliminates the circularity in Attack 1.4. The CAV paper's formal contribution is ε_analytical (provably sound). The practical system uses ε_estimated. This dual reporting satisfies both the Red-Team (soundness) and the FM Lead (tightness).

---

### Decision 9: Strong MC Baseline (NEW)

**Adopted from Red-Team recommendation.** The Empirical plan's MC baseline (Baseline 1) is upgraded from naive stratified MC to *frontier-adaptive importance-sampled MC*. This adds ~100 lines of code but dramatically strengthens the baseline.

**Rationale:** The Red-Team (Attack 4.3) correctly identifies that the naive MC baseline is a strawman. A strong MC baseline makes the comparison credible. If the coverage certificate cannot beat frontier-adaptive MC on detection rate, the paper must reframe around certification value (ε guarantee, spatial map) rather than detection rate.

**Implication:** Empirical H5 (≥10% marginal detection over MC) may fail against the strong baseline. The Empirical plan must be prepared for this outcome and have a fallback framing (see Verification Chair COND-6).

---

### Decision 10: D1 Gate Strengthened (MODIFIED)

Gate D1 is revised to test *both* easy and hard scenarios:
- **D1a (original):** 4-joint chain, ±30°. w ≤ 5× on 95% of geometries. Pass threshold.
- **D1b (new):** 7-joint chain, realistic ranges (shoulder ±60°, elbow ±70°, wrist ±45°). w ≤ 10× on 90% of geometries after subdivision (max 6 levels). Pass threshold.

**Rationale:** Red-Team Attack 3.1 and Verification Chair CONT-2 both demand testing on realistic parameters. D1a alone provides false confidence.

---

## Part 5: What Changes from Proposals

### 5.1 Changes to Formal Methods Proposal

| Item | Original | Revised | Justification |
|------|----------|---------|---------------|
| **C1 ε target** | ε < 0.01 | ε < 0.05 (hard), ε < 0.02 (stretch) | Red-Team back-of-envelope validated by Algorithms (Part 2, Disagreement 1) |
| **C1 Lipschitz assumption** | Global L (with violation detection) | Piecewise-Lipschitz (Def 5 elevated to primary assumption) | Already in FM proposal but now *mandatory*, not optional mitigation |
| **C3 SMT value model** | Volume subtraction with frontier-resolution "enhanced model" | Frontier-resolution is *conditional* (not proven); volume subtraction is the baseline | Algorithm 3 confirms negligible volume; frontier-resolution needs Lemma B2 or D3 gate |
| **Certificate format** | ⟨S, V, U, ε, δ⟩ | ⟨S, V, U, ε_analytical, ε_estimated, δ, κ⟩ | Dual-ε (Decision 8) + explicit κ-completeness (Decision 6) |
| **Verified set V** | SMT-verified regions only | SMT + Tier 1 green regions | Decision 7 (Red-Team recommendation) |
| **New required: Lemma B2** | Not present | Frontier Resolution Improvement Bound (conditional) | Decision 2 — needed to bridge C3 volume-subtraction to frontier-resolution |
| **New required: L_max derivation** | Open Question 2 (no resolution) | Mandatory analytical derivation from J_θ | Decision 8 — eliminates circularity (Attack 1.4) |

### 5.2 Changes to Algorithms Proposal

| Item | Original | Revised | Justification |
|------|----------|---------|---------------|
| **Algorithm 1b: subdivision** | Body-parameter and joint-angle axes | Mandatory joint-angle subdivision for ranges > 45° | Disagreement 4 resolution; Red-Team Attack 3.1 |
| **Algorithm 4: verified set** | SMT regions only | SMT regions + Tier 1 green regions | Decision 7 |
| **Algorithm 5: default α** | Grid search 0–50% | Default α = 5%, adaptive increase based on frontier-resolution measurement | Decision 5 |
| **Algorithm 6: stratification** | Full-factorial stratification implied | Marginal stratification over Θ (d=5) only for k ≥ 2 | Part 1, Critique 6 — full-factorial infeasible at d=26 |
| **New: Algorithm 4 Phase 5** | Frontier-resolution enhancement | Remains, but marked as "conditional enhancement" pending D3 | Decision 2 |
| **Risk table: ε > 0.05** | P=40%, Impact=High | P=50%, Impact=High (revised upward) | Red-Team analysis increases probability estimate |

### 5.3 Changes to Empirical Proposal

| Item | Original | Revised | Justification |
|------|----------|---------|---------------|
| **H1: 5× improvement** | median(ρ) ≤ 0.20 | Hard pass: median(ρ) ≤ 0.33 (3×); stretch: ≤ 0.20 (5×) | Disagreement 3 resolution |
| **Baseline 1: MC** | Stratified MC (1M samples) | Frontier-adaptive importance-sampled MC | Decision 9 (Red-Team Attack 4.3) |
| **H5: marginal detection** | MDR ≥ 10% | MDR ≥ 10% retained; add fallback framing if MDR 5–10% | Red-Team predicts < 5% vs. strong MC |
| **D1 gate** | 4-joint ±30° only | D1a (4-joint ±30°) + D1b (7-joint realistic ranges) | Decision 10 |
| **H6: multi-step ε** | ε < 0.1 on 75% of scenes | ε < 0.15 for k=3 (relaxed); ε < 0.10 for k=2 | Decision 4 |
| **New: ε vs. budget curve** | Not present | Add experiment: ε(T) for T ∈ {1, 5, 10, 30, 60} minutes | Red-Team Attack 1.3 recommendation |
| **New: strong MC in baseline** | Not present | Baseline 6: frontier-adaptive importance-sampled MC | Decision 9 |
| **New: Tier 1 as verified volume** | Not in ablation | Add ablation A6: certificate with vs. without Tier 1 green regions in V | Decision 7 validation |

### 5.4 Changes to Red-Team Proposal (Acknowledged Attacks)

| Attack | Status | Resolution |
|--------|--------|------------|
| 1.1 (Lipschitz at joint limits) | **Resolved** | Piecewise-Lipschitz (Def 5) with κ-completeness |
| 1.2 (ε too loose) | **Partially resolved** | Targets revised; frontier-resolution conditional |
| 1.3 (sampling density impractical) | **Partially resolved** | ε target relaxed; adaptive stratification adopted |
| 1.4 (circular L estimation) | **Resolved** | Analytical L_max + dual-ε reporting |
| 2.1 (soundness envelope tiny) | **Acknowledged** | Cannot be resolved; inherent to linearization |
| 2.2 (SMT volume negligible) | **Resolved** | Tier 1 envelopes replace SMT as primary symbolic volume |
| 3.1 (wrapping factor on 7-joint) | **Partially resolved** | D1b gate added; joint-angle subdivision mandated |
| 3.2 (sin/cos error) | **Acknowledged (minor)** | Subsumed by Attack 3.1 |
| 4.1 (procedural scenes unrepresentative) | **Partially resolved** | Real scene count target increased; results reported separately |
| 4.2 (bug injection detectable) | **Acknowledged (minor)** | Severity distribution specified in revised plan |
| 4.3 (MC baseline is strawman) | **Resolved** | Strong MC baseline adopted |
| 5.1 (Unity parser fragility) | **Acknowledged** | Engineering concern; target Unity 2022.3 LTS + XRI ≥ 2.3 |
| 5.2 (7-DOF model insufficient) | **Acknowledged** | Document limitation; defer trunk model to future work |
| 5.3 (device model oversimplified) | **Acknowledged (minor)** | Acceptable for research prototype |
| 6.1 (C1/C4 coverage gap) | **Resolved** | Bug classes explicitly characterized by tier (Part 1 resolution) |
| 6.3 (metrics internally inconsistent) | **Resolved** | All targets revised to achievable values |

### 5.5 Changes to Verification Proposal (Gate Impacts)

| Gate | Original Criterion | Revised Criterion |
|------|-------------------|-------------------|
| D1 | w ≤ 5× on 4-joint ±30° | D1a: w ≤ 5× on 4-joint ±30°; D1b: w ≤ 10× on 7-joint realistic |
| D3 | median(ρ) ≤ 0.20 | Hard: median(ρ) ≤ 0.33; Stretch: ≤ 0.20 |
| D5 | ε < 0.1 for k ≤ 3 | ε < 0.10 for k=2; ε < 0.15 for k=3 |
| CONT-7 | ε_cert ≤ ε_CP / g, g > 1 | g > 1 using ε_analytical (includes Tier 1 verified volume) |
| ABAND-3 | ε_cert > ε_CP for all |V|/|Θ| < 0.5 | ε_cert > ε_CP for all |V|/|Θ| < 0.5 (unchanged — still must beat CP) |
| COND-6 | CP improvement 3–5× | CP improvement 2–5× (lowered floor to 2×) |

---

## Part 6: Consolidated Risk Assessment

| Risk | P (revised) | Impact | Combined | Mitigation |
|------|-------------|--------|----------|------------|
| Certificate not tighter than strong MC | 40% | 9/10 | 3.6 | Frontier-resolution + Tier 1 verified volume. Fallback: reframe as certification, not detection. |
| Wrapping factor makes Tier 1 useless | 20% | 8/10 | 1.6 | Joint-angle subdivision; D1b gate; Taylor model fallback. |
| ε > 0.05 on typical scenes | 35% | 8/10 | 2.8 | Revised targets; ε vs. budget curve; frontier-resolution at D3. |
| Unity parser fails on real scenes | 35% | 6/10 | 2.1 | Target specific LTS version; parse confidence metric; DSL fallback. |
| Lipschitz violations pervasive (κ > 0.3) | 25% | 6/10 | 1.5 | Piecewise-Lipschitz; focused boundary sampling; partial cert labeling. |
| Multi-step certificates fail (k=3) | 40% | 4/10 | 1.6 | Restrict to k ≤ 2 (COND-3); marginal stratification. |
| Frontier-resolution doesn't work | 45% | 5/10 | 2.25 | Fall back to Tier 1 verified volume + sampling-only ε. |
| Analytical L_max is vacuous | 30% | 4/10 | 1.2 | Dual-ε reporting; lead with ε_estimated if L_max/L̂ > 20. |

**Compound risk (at least one critical failure):** ~65% (down from ~85% in depth check, due to revised targets and added mitigations). The project remains high-risk but the failure modes now have explicit fallback paths.

---

## Part 7: Theory Stage Exit Criteria (Binding)

The theory stage is COMPLETE when all of the following are delivered:

1. **C1 proof** — complete, piecewise-Lipschitz formulation, all assumptions stated, boundary cases checked.
2. **C2 constants** — computed for 7-DOF arm, ANSUR-II 50th percentile, ±30° and ±60° ranges.
3. **B1 numerical predictions** — for D1a and D1b configurations.
4. **B2 (conditional)** — Frontier Resolution Improvement Bound, or explicit statement that it is deferred to empirical validation.
5. **L_max derivation** — analytical bound from kinematic Jacobian.
6. **Algorithms 1–7** — pseudocode with complexity analysis (as submitted, with modifications from Part 5.2).
7. **Evaluation plan** — revised per Part 5.3 (strong MC baseline, D1b gate, relaxed H1/H5/H6).
8. **Assumption catalog** — unified across all proposals, with each theorem referencing its required assumptions by ID.
9. **κ-completeness analysis** — worked example showing κ for a 30-element scene with 7-DOF arm.

The Verification Chair's CONTINUE criteria (CONT-1 through CONT-8) are the binding quality gate. The synthesis decisions above ensure that the deliverables *can* satisfy these criteria; whether they *do* is determined at the Week 4 decision meeting.

---

*End of synthesis.*
