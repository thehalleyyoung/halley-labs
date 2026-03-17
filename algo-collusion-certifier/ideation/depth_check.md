# Depth Check: CollusionProof

**Evaluator**: Impartial Verifier (Best-Paper Committee Chair Model)
**Date**: 2026-03-08
**Method**: Three-expert adversarial team (Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer) with cross-critique synthesis
**Document Under Review**: `ideation/crystallized_problem.md` — CollusionProof: Proof-Carrying Collusion Certificates via Compositional Statistical Testing and Counterfactual Deviation Analysis for Black-Box Algorithmic Pricing Markets

---

## Pillar 1: EXTREME AND OBVIOUS VALUE

**Score: 5.5 / 10** — Below threshold. Amendments required.

### What works

The regulatory timing is genuine. The EU DMA entered full enforcement March 2024. The DOJ RealPage case (filed 2024) is the first federal antitrust action targeting algorithmic pricing coordination. Assad et al. (2024, *JPE*) provides empirical evidence of margin inflation in algorithmic duopolies. These are not hypothetical — they are active proceedings and mandates. The triple intersection of formal verification × game theory × competition law has no known active research group producing tools. This gap is real and verified by the prior art audit: Calvano et al. simulate without certifying, Gambit computes equilibria without black-box input, PRISM-games requires explicit state-space models, PrimeNash targets analytical games not empirical behavior, and EGTA extracts empirical game models without collusion formalization.

### What fails

**The oracle rewind assumption annihilates the regulatory use case.** The counterfactual deviation layer (M2, M3, M5) — three of six substantive mathematical contributions — requires that pricing algorithms "be replayable from arbitrary history prefixes." No real-world proprietary algorithm (Amazon, Uber, RealPage) satisfies this. These algorithms are embedded in distributed systems with external state dependencies, protected by trade secret law, and non-deterministic in ways that make replay fundamentally impossible. The proposal conflates "sandbox audit" with "regulatory enforcement." The EU DMA Article 6(1)(a) does not require or enable sandbox testing of pricing algorithms. The DOJ RealPage prosecution uses behavioral evidence, not interactive oracle access.

**Regulators have not asked for machine-checkable certificates.** Antitrust proceedings are adversarial legal processes evaluated by judges and juries who cannot read formal proofs. The Daubert standard for expert testimony asks whether methodology is generally accepted, testable, and peer-reviewed — not whether it produces machine-checkable proofs. An expert econometrician's report is currently *more useful* in court than a formal certificate.

**The passive-only mode (Layer 0) salvages partial value.** M1 (composite test) and M7 (directed closed testing) operate on observed trajectory data without oracle access. This is already more principled than existing screening methods (variance screens, Granger causality) because M1 has formal Type-I error control over a game-theoretic null. Layer 0 alone surpasses every existing deployable statistical screening tool. But it is a screening tool, not a "certifier" — a significantly narrower contribution than advertised.

### Required amendments for this pillar

1. **Restructure into tiered oracle access model**: Layer 0 (passive observation, M1+M7), Layer 1 (replay oracle, adds M2+partial M5), Layer 2 (full rewind, adds M3+tight M5). Layer 0 must be self-contained and independently publishable.
2. **Replace "regulatory enforcement tool" language with "algorithmic audit framework."** Position as research infrastructure for competition analysis, not a deployable courtroom tool.
3. **Add "regulatory sandbox" framing explicitly**: The full pipeline operates in voluntary/cooperative audit settings, not adversarial enforcement.

---

## Pillar 2: GENUINE DIFFICULTY AS A SOFTWARE ARTIFACT

**Score: 6.5 / 10** — Below threshold. Amendments required.

### What's genuinely hard

- **Compositional soundness** is the hidden challenge. Making M1's composite test, M2's adaptive sampling, M3's perturbation injection, and M7's closed testing compose with end-to-end α-control across different probability spaces and conditioning events is non-trivial integration that has no off-the-shelf solution.
- **S4 (Counterfactual Deviation Analysis, claimed 20K)**: Adaptive sampling with selection-bias control via peeling argument, punishment detection, Monte Carlo variance reduction. This is the core research subsystem.
- **S6 (Certificate DSL & Proof Checker, claimed 20K)**: A *de novo* proof language with ~15 axiom schemas, ~25 inference rules, and a 2,500 LoC trusted core. No existing proof checker handles game-theoretic collusion certificates. Writing a correct minimal trusted core for a new domain is extremely hard per line of code.
- **Performance constraints**: >100K rounds/sec simulation target and O(N·D·S·T) = ~600M round-steps for counterfactual analysis impose real systems engineering requirements that justify Rust for the simulation loop and proof checker kernel.

### LoC inflation

The 168K estimate is inflated by 25–40%. Subsystem-by-subsystem analysis:

| Subsystem | Claimed | Honest Estimate | Concern |
|---|---|---|---|
| S1 (Game Sim) | 30K | 12–18K | 3 models × 3 demand functions, each small. Bertrand ~500 lines. |
| S2 (Algo Interface) | 21K | 8–12K | 8 algorithms at ~200–1200 lines each + PyO3 sandbox |
| S3 (Equilibrium) | 17K | 5–12K | Could FFI to Gambit; reimplementation is unjustified for non-critical-path component |
| S4 (Counterfactual) | 20K | 15–20K | Core research. Estimate reasonable. |
| S5 (Collusion Premium) | 15K | 10–15K | M1+M7 implementation. Reasonable. |
| S6 (Certificate DSL) | 20K | 15–20K | Core research. Estimate reasonable. |
| S7 (Evidence Bundle) | 8K | 5–8K | Protobuf + Merkle. Reasonable. |
| S8 (Report Gen) | 10K | 3–5K | LaTeX/HTML templating is mature-library territory. Padding. |
| S9 (Evaluation) | 17K | 8–12K | Scenarios + metrics. Heavy scipy/sklearn use. |
| S10 (Orchestration) | 10K | 4–6K | CLI + config. Standard infrastructure. |
| **Total** | **168K** | **90–128K** | |

**Honest split**: ~55K core novel research code (S4+S5+S6+S7) + ~35–73K essential infrastructure (S1+S2+S3) + ~15–23K support/evaluation (S8+S9+S10). The 168K figure should be revised to ~110–130K with honest test-to-code ratios.

### Rust justification

Rust is necessary for two independent reasons:
1. **Trust model**: The proof checker kernel must be memory-safe, deterministic, and auditable. Python's runtime is non-deterministic, memory-unsafe at the C extension layer, and too large to audit.
2. **Performance**: At Python speeds (~1–5K rounds/sec), the evaluation would require 76,000–380,000 CPU-hours — infeasible on a laptop.

### Required amendments for this pillar

1. **Revise LoC to ~110–130K** with explicit core/infrastructure/support breakdown.
2. **Define minimum viable artifact at ~60K LoC** (CollusionProof-Lite: 2-player Bertrand/Cournot, tabular RL, analytical equilibria, M1+M2+M6 core).
3. **Justify or eliminate S3 reimplementation**: Use Gambit FFI for equilibrium computation unless Rust reimplementation is justified by the trusted computing base requirement.

---

## Pillar 3: BEST-PAPER POTENTIAL

**Score: 5.0 / 10** — Below threshold. Amendments required.

### What's strong

- **M1 is a legitimate new problem formulation.** The composite hypothesis test where H₀ is parameterized by demand systems × learning algorithms is unprecedented. While composite testing over infinite-dimensional nuisance parameters exists in semiparametric statistics (Andrews & Shi, 2013; Chernozhukov, Chetverikov & Kato, 2014), no existing framework tests against a null family parameterized by games and learning algorithms. The formulation novelty is real.
- **Proof-carrying certificates for economic properties** bring the PCC paradigm into an entirely new domain. No precedent exists in any community.
- **The uncontested triple intersection** survives scrutiny. No active group produces tools at this intersection.

### What's weak

- **M1 is formulation novelty, not mathematical novelty.** The semiparametric testing machinery already exists. The contribution is applying it to a game-theoretic null — interesting and publishable, but the techniques needed (bounding maximum cross-firm correlation over Lipschitz function spaces) are likely achievable with standard empirical process theory tools. Grade A is generous; Grade B+ pending proof details is more accurate.
- **M4's C3 dependency is a significant liability.** The completeness result depends entirely on Conjecture C3 (Folk Theorem converse for bounded-recall). An unproved conjecture in the *statement* of a main result is a red flag for program committees. However, conditional results are routine in cryptography and complexity theory (security under hardness assumptions, complexity relative to oracles). The question is whether C3 is "natural" enough that committees accept it. The Synthesizer's analogy to cryptographic hardness assumptions is apt: retain M4 at Grade A with an explicit asterisk, but present unconditional soundness as the primary guarantee.
- **Venue mismatch risk.** The work is too engineering-heavy for pure theory venues (WINE), too theoretical for systems venues (SOSP), and too domain-specific for ML venues (NeurIPS). EC is the correct primary target — it explicitly values CS-meets-econ contributions. The Synthesizer's suggestion of AAMAS/FAccT targets less prestigious venues. The paper should be structured as a **theory contribution with a system demonstration** for EC.
- **"Defining a new subfield" is overclaimed.** The follow-up community requires expertise in game theory + statistics + formal verification + antitrust economics — an intersection of approximately zero active researchers. The work opens an interesting direction, not a subfield.

### Required amendments for this pillar

1. **Target EC as primary venue.** Structure paper as theory (M1 + conditional M4 + M6) demonstrated through system.
2. **Decouple soundness from completeness.** Lead with: "Soundness (Type-I error control) is unconditional. Completeness is conditional on C3." This is the difference between a desk reject and a strong accept.
3. **Prove C3 for restricted strategy classes** (deterministic automata with ≤M states, grim trigger, tit-for-tat). Show the conjecture is "morally true" for practical cases even if the general case is open.
4. **Drop "defining a new subfield" language.** Replace with "opening a new direction at the intersection of computational game theory and formal verification."

---

## Pillar 4: LAPTOP CPU + NO HUMANS

**Score: 5.5 / 10** — Below threshold. Amendments required.

### What works

- **CPU-only execution is architecturally sound.** The critical path (game simulation, statistical testing, proof checking) is entirely CPU-bound arithmetic. Deep RL training (DQN, PPO) runs on CPU via PyTorch — slow but feasible and only needed for evaluation, not certification.
- **Memory budget is reasonable.** Peak 8 GB concurrent, well within 32 GB limit. The Skeptic's claim that memory exceeds 32GB is factually incorrect — it incorrectly sums sequential peaks.
- **Evaluation is fully automated.** 30 ground-truth scenarios with known labels, automated metrics, bootstrap CIs, sensitivity analysis. Zero human annotation. This is genuinely no-humans.

### What fails

- **3,800 CPU-hours is a development killer.** One full evaluation = ~20 days on 8 cores. Iterative development requires 5–14 full-equivalent runs. Total: ~100–280 days of continuous 8-core compute just for evaluation iteration. This is feasible only if evaluation is tiered.
- **Deep RL scenarios are expensive and infrequent.** DQN training at 2–4 hours per agent × multiple scenarios × multiple seeds means deep RL evaluation cannot run frequently during development. Bugs that manifest only with deep RL agents may go undetected for weeks.
- **PPO 3-player: 10–18 hours wall-clock per scenario.** A single bug-fix-rerun cycle costs 10–18 hours for one scenario. Five iterations = 50–90 hours = 2–3 days for ONE scenario.
- **"Embarrassingly parallel on 8 cores" is underwhelming.** The parallelism is real but 8× speedup on a 3,800-hour workload still yields 475 hours = 20 days. This is "feasible but painful."

### Required amendments for this pillar

1. **Implement three-tier evaluation budget:**
   - `--smoke`: 5 scenarios, tabular RL, 100K rounds, < 30 min wall-clock. For CI and development.
   - `--standard`: 15 scenarios, tabular + bandit, 2–3 players, ~800 CPU-hours (~4 days). For milestone validation.
   - `--full`: 30 scenarios, all algorithms, sensitivity analysis, ~3,800 CPU-hours (~20 days). For camera-ready only.
2. **Define "fast iteration" development mode.** Development uses `--smoke` exclusively; `--standard` runs at milestones; `--full` runs once for final evaluation.
3. **Reduce 3-player scenarios.** Full 3-player counterfactual analysis for 5 scenarios only (not all 30). Demonstrate scaling; defer comprehensive 3-player evaluation.

---

## Pillar 5: FATAL FLAWS

### Flaw 1: Conjecture C3 Dependency — SEVERITY: HIGH

M4 (Completeness of the Hybrid Certifier, designated Grade A) is explicitly conditional on the unproved Folk Theorem converse for bounded-recall strategies. If C3 is false, "stealth collusion" strategies may exist that sustain supra-competitive pricing without detectable punishment responses, and the completeness guarantee is vacuous.

**Mitigating factors**: (a) Soundness (Type-I error control) is unconditional — C3 affects only completeness, not false positive control. (b) Conditional results are standard in cryptography and complexity theory. (c) C3 can be proved for restricted strategy classes (deterministic automata).

**Verdict**: HIGH risk but not fatal if soundness is presented as the primary guarantee and completeness is explicitly conditional. The system retains value as a one-sided test: if it certifies collusion, the certification is valid; if it doesn't, the result is inconclusive.

### Flaw 2: Oracle Rewind Assumption — SEVERITY: HIGH

M2 (deviation oracle), M3 (punishment detection), and M5 (collusion premium quantification) require algorithms to be replayable from arbitrary history prefixes. No real-world proprietary pricing algorithm satisfies this. The regulatory use cases cited (EU DMA, DOJ RealPage) do not provide or require this access.

**Mitigating factors**: (a) The passive-only mode (M1+M7) operates without oracle access and is already more principled than existing screening methods. (b) The layered architecture (passive → replay → full rewind) provides graceful degradation. (c) Voluntary/cooperative audit settings do permit oracle access.

**Verdict**: HIGH risk for the regulatory enforcement framing, but addressable via the tiered oracle access model. The contribution is layered, not all-or-nothing.

### Flaw 3: Distribution-Freeness May Be Asymptotic Only — SEVERITY: MODERATE

M1 claims α-sound testing with Type-I error control uniform over infinite-dimensional nuisance parameters, but hedges: "Distribution-freeness may hold only asymptotically, requiring sufficiently large T." Uniform convergence over Lipschitz function spaces requires covering number arguments with potentially slow convergence rates. The "sufficiently large T" may be T = 10⁸ or higher — beyond the evaluation's 10⁵–10⁷ range.

**Mitigating factors**: (a) Permutation-based sub-tests (M3) provide exact finite-sample distribution-freeness. (b) Parametric fallback tests for small T preserve usability. (c) Empirical validation on 10 competitive scenarios across 50 seeds provides practical evidence.

**Verdict**: MODERATE. Weakens M1 from "new theoretical framework" to "asymptotically valid framework with parametric fallbacks" if finite-sample guarantees don't hold. The Grade A claim for M1 is premature without resolution.

### Flaw 4: Evaluation Circularity — SEVERITY: MODERATE

The 30 ground-truth scenarios are designed by the system builders. "Known-collusive" scenarios use textbook patterns (grim trigger, tit-for-tat) that any reasonable statistical test can detect. The evaluation measures "does our system detect the collusion patterns we designed it to detect?" — the answer is trivially yes.

**Mitigating factors**: (a) External baselines (Gambit NE check, statistical screening, Calvano-style comparison) provide non-trivial comparisons. (b) Adversarial stress tests can be added. (c) The 8 boundary/hard scenarios attempt to test discrimination.

**Verdict**: MODERATE. The evaluation needs adversarial red-team scenarios: strategies specifically designed to evade M1 detection. Without these, the evaluation proves engineering competence, not detection power.

### Flaw 5: Collusion Premium Undefined for Canonical Bertrand — SEVERITY: LOW-MODERATE

CP(p̄) is undefined when competitive profit at NE is zero (homogeneous Bertrand — the canonical model of price competition). The proposal acknowledges this but provides no workaround.

**Mitigating factors**: Simple fix — define an absolute-margin deviation metric as fallback for zero-profit equilibria. Restrict relative CP claims to positive-profit settings.

**Verdict**: LOW-MODERATE. Fixable with a straightforward amendment.

### Flaw 6: Null Family Too Broad for Practical Power — SEVERITY: MODERATE

H₀ = {all Lipschitz demand × all independent learners} may be so broad that statistical power against realistic collusion alternatives is impractically low. The proposal acknowledges "parametric restrictions serve as a fallback" but this reintroduces the expert-judgment dependency the system aims to eliminate.

**Mitigating factors**: The tiered null hierarchy (narrow/medium/broad) proposed by the Synthesizer addresses this elegantly — test at multiple specificity levels and report graded confidence.

**Verdict**: MODERATE. The tiered null hierarchy is the right fix and should be incorporated.

---

## Summary Scorecard

| Pillar | Score | Threshold | Status |
|---|---|---|---|
| 1. Extreme & Obvious Value | **5.5** | 7 | ❌ BELOW — Oracle rewind limits applicability; regulatory framing overpromises |
| 2. Genuine Difficulty | **6.5** | 7 | ❌ BELOW — Genuinely hard but LoC inflated; need honest 110–130K estimate |
| 3. Best-Paper Potential | **5.0** | 7 | ❌ BELOW — Formulation novelty is real but C3 conditional weakens completeness story |
| 4. Laptop CPU + No Humans | **5.5** | 9 | ❌ BELOW — Architecturally feasible but 3,800 CPU-hrs constrains iteration severely |
| 5. Fatal Flaws | **2 HIGH, 4 MODERATE** | 0 fatal | ⚠️ No strictly fatal flaws, but HIGH risks require structural amendments |

**Composite: 5.6 / 10**

---

## Verdict: CONDITIONAL CONTINUE

The project addresses a genuine problem at a real intersection with legitimate mathematical novelty. The passive-only mode (M1+M7) is independently valuable and surpasses existing screening tools. However, the full pipeline's value proposition is contingent on access models that don't exist in current regulatory practice, the completeness guarantee depends on an open conjecture, the LoC is inflated, and the evaluation budget requires tiering for development feasibility.

### Binding Conditions for CONTINUE

1. **Tiered oracle access model** (Layer 0/1/2) with Layer 0 independently publishable
2. **C3-conditional framing** — unconditional soundness as primary guarantee
3. **Scope reduction to ~60K LoC MVP** as initial implementation target
4. **Three-tier evaluation budget** with `--smoke` mode for development
5. **LoC revision to ~110–130K** with honest core/infrastructure/support split
6. **Tiered null hierarchy** (narrow/medium/broad) for practical power
7. **Prove C3 for restricted strategy classes** (deterministic automata, grim trigger)
8. **Adversarial red-team scenarios** in evaluation suite
9. **Bertrand CP fix** — absolute-margin fallback for zero-profit equilibria
10. **Drop "defining a new subfield" language** — replace with "opening a new direction"

### Recommended Scope

**Phase 1**: CollusionProof-Lite (~60K LoC). 2-player Bertrand/Cournot, tabular RL + DQN, analytical equilibria, M1+M2+M6 core, Layer 0+1. Evaluation: 15 scenarios, ~800 CPU-hours. Target: EC submission.

**Phase 2**: Full CollusionProof (~110K LoC). N-player, all algorithms, M1–M7, all layers. Evaluation: 30 scenarios + parametric sweeps, ~3,800 CPU-hours. Target: Systems/artifact companion.

### Team Dissent Record

- **Fail-Fast Skeptic** (composite ~4.2/10): Maintains the correct output is a theory paper (M1 + conditional M4, ~15 pages, ~10K supporting code). The systems artifact adds engineering cost without proportional research value. If C3 is false, the project has no completeness story and reduces to a statistical screening suite.
- **Independent Auditor** (composite ~6.5/10): The project is viable with honest scoping. The integration challenge is genuine difficulty the Skeptic underweights. EC is achievable.
- **Scavenging Synthesizer** (composite ~5.6/10): The amended proposal at ~60K LoC with unconditional soundness and tiered oracle access is a strong, honest research contribution. The regulatory window justifies the engineering investment.

---

## Amendments Required

All scores fell below threshold. An amended version of the full problem statement has been written to `ideation/crystallized_problem.md` incorporating all binding conditions above.
