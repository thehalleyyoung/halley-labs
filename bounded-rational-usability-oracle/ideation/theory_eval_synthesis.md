# Theory Evaluation: Scavenging Synthesis

**Proposal:** proposal_00 — The Cognitive Regression Prover: A Three-Layer Usability Oracle  
**Area:** area-042-human-computer-interaction  
**Evaluator:** Scavenging Synthesizer (cross-evaluation synthesis)  
**Prior Evaluations:** Skeptic (33/50), Mathematician (31/50), Community Expert (28/50)  
**Date:** 2026-03-04

---

## VERDICT: CONTINUE

**Final Scores:**

| Axis | Score | Rationale |
|------|-------|-----------|
| 1. Extreme Value | **7/10** | Real unsolved problem (unanimous across all 3 panels). Consistency-oracle framing is a genuine conceptual advance. Docked for unvalidated ordinal agreement and adoption uncertainty. Elevated by parameter-independence giving Day 1 value. |
| 2. Genuine Software Difficulty | **6/10** | Layer 1 is difficulty 4–5; Layer 2 (fragility + paired-comparison) is 7–8. Probability-weighted expected difficulty ≈ 6. The engineering-theory interface (noisy trees → clean MDPs) is a genuine difficulty multiplier underappreciated in raw component scores. |
| 3. Best-Paper Potential | **6/10** | Elevated above prior evaluations (5, 5, 4) because the REFRAMING changes everything. The current framing under-sells. With fragility as lead, "Chaos Monkey for usability" as pitch, and one worked example, this jumps from "solid contribution" to "genuine contender." See §3 below. |
| 4. Laptop-CPU & No-Humans | **9/10** | Unanimously the strongest axis. Every design choice is CPU-native by principle, not compromise. WCAG tailwind improves input quality for free. |
| 5. Feasibility | **6/10** | Incremental architecture is genuine risk management — best feature of the proposal. Layer 1 ships regardless. But compound risk across proofs/parsing/validation is real. |
| **TOTAL** | **34/50** | |

---

## 1. The STRONGEST Possible Elevator Pitch

### Version A (Technical audience — CHI/UIST):

> "We introduce *cognitive fragility* — a Chaos-Monkey-for-usability metric that stress-tests UI designs against the full spectrum of human cognitive capacity, finding 'cognitive cliffs' where small decreases in user ability cause catastrophic task failure. Because fragility compares a UI to *itself* across the capacity space, it requires no external calibration, runs on laptop CPUs in CI/CD pipelines, and provides the first formally grounded usability regression gate with zero parameter tuning for dominant failure modes."

### Version B (Industry/practitioner):

> "Every pull request gets an automatic cognitive cost diff — a 60-second analysis that catches structural usability regressions before merge, identifies *cognitive cliffs* where novice or impaired users suddenly can't complete tasks, and classifies exactly why (choice paralysis, motor difficulty, memory overload) with no human testers, no GPU, and no parameter calibration. Think of it as a Chaos Monkey for usability: it synthesizes worst-case users and tells you where your UI breaks."

### Why these pitches work:

The current proposal leads with "cost differencing" — a correct but uninspiring framing that immediately invites the "isn't this just CogTool?" objection. Both pitches instead lead with **cognitive fragility** (the diamond, per unanimous evaluator agreement) and **Chaos Monkey framing** (scavenged from abandoned Approach C). This accomplishes four things simultaneously:

1. **Sidesteps evaluation circularity.** "Did the UI become more fragile?" is self-contained; it doesn't need external ground truth.
2. **Connects to hot topics.** Chaos engineering, inclusive design, adversarial robustness — all active research fronts.
3. **Makes the contribution vivid.** "Cognitive cliff" is a concept reviewers will remember and cite.
4. **Preserves the full technical depth.** Fragility requires the bounded-rational MDP apparatus, paired-comparison theorem, and cliff-location algorithm — the actual novel math.

---

## 2. Minimum Viable Contribution That Justifies CONTINUE

The **absolute floor** — what ships even if everything else fails:

### Layer 1 alone (8 weeks, ~75% success probability):
- Working accessibility-tree-to-cost-diff pipeline on Chrome
- Parameter-independent regression verdicts for monotone structural changes  
- Interval arithmetic for robust verdicts under parameter uncertainty
- CI/CD integration (GitHub Action, CLI, SARIF output)
- Benchmark suite of 50+ UI pairs from open-source component libraries
- Retrospective ordinal validation (τ ≥ 0.6) against at least one published dataset

**Publishable as:** ICSE tool track or CHI late-breaking work. The parameter-independence insight alone is a publishable observation (confirmed by all evaluators, including the harshest Skeptic: "this framing alone is a publishable insight").

**Why this justifies CONTINUE:** Even this floor delivers something that does not currently exist: an automated structural usability regression detector in CI/CD with zero-calibration verdicts. The gap between accessibility linters (rule-based, no cognitive model) and manual usability studies (expensive, slow) is real and unanimous. A working tool that fills this gap has adoption potential regardless of theoretical depth.

### What raises it above the floor:

| Addition | Probability | Value-Add |
|----------|-------------|-----------|
| Paired-comparison theorem (proved) | 70% | Transforms from heuristic to formal tool |
| Cognitive fragility metric | 65% | Novel contribution, sidesteps circularity, inclusive-design angle |
| Cliff detection algorithm | 75% | Makes fragility computationally feasible in CI/CD |
| One end-to-end worked example | 90% | Single most impactful evidence for reviewers |
| Retrospective validation passes | 70% | External anchor for consistency-oracle claim |

**Expected contribution (probability-weighted):** Layer 1 + paired-comparison proof + fragility metric + cliff detection + worked example. This is a solid UIST full paper. ~40% chance of landing the full "best-paper contender" narrative.

---

## 3. Reframing to Maximize Best-Paper Potential

The current proposal is organized as: Layer 1 (profiler) → Layer 2 (theory) → Layer 3 (scaling). This is the correct *engineering* architecture but the wrong *paper* architecture. The paper should be organized around a **three-act intellectual narrative**:

### Act 1: "You don't need parameters" (The Hook)
Lead with the parameter-independence result. Not as a theorem — as a *demonstration*. Show a real UI pair (Material UI v4→v5 or equivalent) where the system produces a correct, zero-calibration regression verdict. Show the interval arithmetic collapsing. Show the structural predicate.

**Why this works:** Immediately answers "why should I care?" and "does it work?" in one move. Disarms the "is this just CogTool?" objection by showing something CogTool never did: a working CI/CD verdict with zero setup.

### Act 2: "But when you do, errors cancel" (The Surprise)
Introduce the paired-comparison theorem. Show that the consistency-oracle framing is not rhetoric but mathematics: shared-analysis error cancellation yields O(ε) differential accuracy from O(Hβε) absolute accuracy. This is the "aha" moment.

**Why this works:** The reviewer who just saw a working demo now learns there's a formal guarantee behind it. The gap between "0.53 TV for absolute" and "20ms for differential" is genuinely surprising.

### Act 3: "And you can stress-test against the worst-case user" (The Diamond)
Introduce cognitive fragility and cliff detection. Show a UI that passes Act 1's regression check (no monotone structural degradation) but reveals a cognitive cliff — a β threshold where novice users suddenly can't complete the task. This is the result no other tool can produce.

**Why this works:** Fragility is the contribution most likely to open a new research direction. "Usability as robustness to user capacity variation" connects to inclusive design, adversarial ML, and chaos engineering — three active, well-funded research fronts. The cliff-location theorem provides a compact, provable, computationally enabling result. And critically: fragility is immune to the evaluation-circularity objection that dogs the cost-regression claim.

### Specific changes to maximize best-paper probability:

1. **Title change:** "Cognitive Fragility: Stress-Testing UI Designs Against the Spectrum of Human Ability" (instead of "The Cognitive Regression Prover"). Lead with the novel concept, not the pipeline.

2. **Lead contribution: fragility, not cost differencing.** Cost regression is Layer 1 engineering. Fragility is the intellectual contribution. Currently fragility is buried as a Layer 2 add-on. It should be the paper's thesis.

3. **One stunning figure.** A "fragility surface" plot: x-axis = β (user capacity), y-axis = expected task cost, with cliff locations marked. Show two versions of the same UI, one with a cliff and one without. This single figure would be the most-cited element of the paper.

4. **Drop "three-layer" framing for the paper.** The engineering architecture is three layers; the paper should present a unified contribution. The layer decomposition is an implementation detail, not a research contribution.

5. **One end-to-end worked example.** Every evaluator flagged this as the single highest-impact missing element. Take Material UI v4→v5, extract accessibility trees, run the full pipeline, show parameter-free verdict + fragility analysis + cliff detection. This is achievable in days and would transform the proposal from blueprint to evidence.

6. **"Chaos Monkey for usability" as the pitch, not the subtitle.** This framing (scavenged from Approach C) was independently flagged by evaluators across all three panels as dramatically improving communicability. Use it in the abstract, introduction, and talk title.

7. **Demote Theorem 4 (cost algebra) entirely.** It's conjectured, previously falsified, and not needed. Its presence weakens the paper by creating an obligation the authors can't fulfill. Remove it; mention it as future work in one sentence.

8. **Rename Theorem 2 (parameter-independence).** Stop calling it a theorem. Call it a "Corollary" or "Observation." Three evaluations independently flagged this as trivial math inflated by framing. Keeping it as a theorem invites reviewer skepticism about the authors' calibration.

### Estimated best-paper probability after reframing:

| Configuration | Best-Paper Prob | Strong Accept Prob |
|---------------|----------------|-------------------|
| Current framing, no data | ~5% | ~25% |
| Current framing + worked example | ~10% | ~45% |
| **Reframed (fragility-led) + worked example** | **~20-25%** | **~55-65%** |
| Reframed + worked example + full proofs | ~30-35% | ~70% |

The reframing roughly doubles best-paper probability by (a) leading with the most novel contribution, (b) providing a vivid memorable concept, (c) sidestepping the weakest flank (evaluation circularity), and (d) connecting to multiple hot research directions.

---

## 4. Final Scores, Verdict, and Conditions

### Scores

| Axis | Skeptic | Mathematician | Community | **Synthesis** | Rationale for Synthesis Score |
|------|---------|--------------|-----------|---------------|-------------------------------|
| Value | 7 | 6 | 5 | **7** | Real gap confirmed by all; parameter-independence gives Day 1 value; the Skeptic's "trivial baseline" challenge is valid but the formal framework demonstrably does more (fragility, cliff detection) |
| Difficulty | 6 | 6 | 6 | **6** | Unanimous convergence. Layer 1 is moderate; Layer 2 is genuinely hard; probability-weighted ≈ 6 |
| Best-Paper | 5 | 5 | 4 | **6** | **Elevated.** The reframing to fragility-led narrative with one worked example changes the equation. The current score (4-5) reflects the current framing; the achievable score with modest reframing is 6-7 |
| CPU/No-Humans | 9 | 8 | 8 | **9** | Strongest axis by far. Principled CPU-native design. WCAG tailwind. No training. |
| Feasibility | 6 | 6 | 5 | **6** | Incremental architecture is real risk management. Compound risk is real but mitigated by layer independence. |
| **TOTAL** | **33** | **31** | **28** | **34** | |

### Cross-Evaluation Score Distribution

Across all four evaluations (Skeptic, Mathematician, Community Expert, Synthesis):

| Axis | Min | Max | Spread | Convergence |
|------|-----|-----|--------|-------------|
| Value | 5 | 7 | 2 | Moderate — disagreement on trivial-baseline threat |
| Difficulty | 6 | 6 | 0 | **Perfect convergence** |
| Best-Paper | 4 | 6 | 2 | Moderate — disagreement on reframing impact |
| CPU | 8 | 9 | 1 | **Strong convergence** |
| Feasibility | 5 | 6 | 1 | Strong convergence |
| **Total** | **28** | **34** | **6** | Moderate — within normal adversarial range |

**Key observation:** Difficulty and CPU scores are fully converged (spread ≤ 1). The remaining disagreement concentrates on Value (is the trivial baseline fatal?) and Best-Paper (does reframing change the odds?). Both are *answerable empirically within 2 weeks* — making the go/no-go gate decision tractable.

### VERDICT: **CONTINUE**

**Confidence: Moderate-to-High.** Four independent evaluations, each with three-expert adversarial panels, converge on CONTINUE. No evaluation recommended ABANDON. The Skeptic's sustained adversarial attacks identified real weaknesses but could not locate a fatal flaw. The incremental architecture ensures bounded downside.

### Conditions (ordered by criticality):

#### Hard Gates (binary go/no-go):

| # | Condition | Deadline | Failure → |
|---|-----------|----------|-----------|
| **G1** | **Validation data exists.** Name ≥5 specific UI pairs with confirmed accessibility-tree availability AND published/extractable human orderings. Not abstract references — specific applications, versions, confirmed extractability. | Week 2 | **ABANDON** consistency-oracle claim; pivot to fragility-only paper |
| **G2** | **End-to-end proof of life.** Layer 1 producing cost diffs on ≥3 real component library examples (Chrome/CDP). | Week 4 | **ABANDON** if parser infeasible |
| **G3** | **Trivial baseline differential.** Layer 1 must demonstrate Δτ ≥ 0.08 ordinal agreement over a 50-line axe-core diff script on ≥30 UI pairs. | Week 6 | Rethink cost model; complexity not warranted |

#### Required Amendments (non-negotiable for paper):

| # | Amendment | Rationale |
|---|-----------|-----------|
| **A1** | **Lead with fragility, not cost regression.** Paper title, abstract, and contribution list must center cognitive fragility and cliff detection. Cost regression becomes the "enabling infrastructure." | Unanimous across all evaluations: fragility is the diamond |
| **A2** | **Demote Theorem 2 to "Observation" or "Corollary."** Parameter-independence is useful but trivially follows from monotonicity. Calling it a theorem invites reviewer backlash. | Unanimous: "a tautology dressed as a theorem" |
| **A3** | **Demote Theorem 4 to future work.** Cost algebra ordinal soundness is conjectured, previously falsified, and not needed for Layers 1-2. | Unanimous across all evaluations |
| **A4** | **Drop "zero false positives" language.** Replace with: "Parameter-independent verdicts are correct by construction under the assumption that monotone structural cost increases constitute regressions." | Skeptic flagged as tautological |
| **A5** | **Drop or empirically ground the 50-70% coverage claim.** Classify 50+ real usability issues to determine actual parameter-free coverage. If < 30%, restructure contribution framing. | Mathematician flagged as "fabricated" |
| **A6** | **Add absolute-cost floor to fragility metric.** F(M) alone doesn't distinguish "good for experts, bad for novices" from "uniformly terrible." Qualify fragility as secondary signal after cost-regression check. | Skeptic flagged the confound |

#### Strategic Recommendations (high-value, not blocking):

| # | Recommendation | Impact |
|---|----------------|--------|
| **S1** | Produce one end-to-end worked example on a real UI pair within 2 weeks. Single most impactful action. | Transforms proposal from blueprint to evidence |
| **S2** | Characterize k-distribution for real PRs (analyze ≥20 design-system PRs). If median k > 20, prominently qualify Theorem 1's scope. | Addresses the paired-comparison theorem's weakest flank |
| **S3** | Adopt "Chaos Monkey for usability" as the primary external-facing description. | Dramatically improves communicability; connects to DevOps/SRE audience |
| **S4** | Produce a "fragility surface" figure for one real UI. This will be the paper's most-cited element. | Vivid, memorable, demonstrates the concept non-verbally |
| **S5** | Scope to Chrome-only for v1. Cross-browser normalization is a multi-year industry problem; don't let it block the research contribution. | Reduces engineering risk by ~40% |

---

## 5. Value in Abandoned Approaches to Recover

### From Approach C ("Adversarial Cognitive Fuzzer"):

**CRITICAL RECOVERY — Already partially done, but not enough.**

| Element | Status | Recovery Action |
|---------|--------|-----------------|
| "Chaos Monkey for usability" framing | Noted but not adopted | **Adopt as primary pitch.** Use in title, abstract, intro, talk. |
| Cognitive fragility metric F(M) | Incorporated into Layer 2 | **Elevate to co-lead contribution.** Currently buried as add-on. |
| Cliff-location theorem | Incorporated into Layer 2 | **Elevate to featured theorem.** It's the most compact, provable, computationally enabling result. |
| Adversarial policy synthesis (CA-MCTS) | Abandoned | **Recover as future work.** CA-MCTS enables richer adversarial analysis (per-step capacity adversary). Not needed for v1 but is a natural Layer 3 extension. |
| Inclusive design framing | Partially absorbed | **Make explicit.** "Cognitive cliffs disproportionately affect users with lower cognitive capacity" connects to accessibility, aging, cognitive disability — a powerful positioning for CHI. |
| Fragility decomposition ($F(M) \approx \sum_t F_t(M)$) | Sketch exists | **Complete.** Per-step fragility attribution is the diagnostic version of fragility — the "why" to fragility's "whether." Analogous to how bottleneck taxonomy adds "why" to cost regression. |

**The biggest missed opportunity in the current proposal:** Approach C's *entire framing* was superior to the current proposal's framing. The proposal adopted C's technical components (fragility, cliffs) but not C's *narrative architecture*. The narrative should be: "We stress-test UIs against the spectrum of human cognitive ability and find breaking points." The current narrative is: "We diff cognitive costs between UI versions." The former is vivid, novel, and memorable. The latter is correct but flat.

### From Approach A ("Full-Theory Bisimulation Machine"):

| Element | Status | Recovery Action |
|---------|--------|-----------------|
| Formal verification analogy | Partially adopted | **Strengthen.** "Usability regression testing as formal verification" is the framing that positions this at the intersection of PL and HCI — exactly where the most impactful recent work lives (e.g., Ringer et al. on proof repair, Bornholt et al. on Cosette). |
| Full cost algebra ($\oplus$, $\otimes$, $\Delta$) | Deferred to Layer 3 | **Keep deferred but maintain as future work.** The algebra's *structure* is elegant even if ordinal soundness is unproven. Present the operators; state the conjecture; don't claim the theorem. |
| Bottleneck taxonomy | In proposal, under-emphasized | **Feature as practical contribution.** Even the Skeptic acknowledged engineering value. The five-type classification (perceptual/choice/motor/memory/interference) with information-theoretic signatures is actionable and novel in the CI/CD context. |

### From Approach B ("Lean Profiler"):

| Element | Status | Recovery Action |
|---------|--------|-----------------|
| "Cognitive git diff" framing | Not adopted | **Recover for the CLI tool name.** `usability-diff` or `cogdiff` as the tool name. The Chaos Monkey framing is for the paper; the "cognitive diff" framing is for the developer-facing tool. |
| Explicit detection boundary characterization | Not in current proposal | **Add.** "Here is what the system can detect; here is what it cannot" is more trustworthy than sweeping claims. The boundary between parameter-free / parameter-dependent / undetectable regressions should be a figure. |
| 85% coverage → explicit coverage measurement | Overclaimed | **Replace with empirical measurement.** This is Amendment A5 above. |

### Net assessment of scavenged value:

The abandoned approaches collectively contain **more narrative value than technical value**. The current proposal absorbed the right technical components (fragility metric, cliff detection, interval arithmetic) but left the best *framings* on the cutting-room floor. Recovery is primarily a writing and positioning exercise, not an engineering one:

1. **Adopt Approach C's narrative architecture** (Chaos Monkey / stress testing / cognitive cliffs)
2. **Adopt Approach B's tool-naming convention** (cognitive diff for the CLI)  
3. **Adopt Approach A's formal-verification analogy** (usability regression ≈ type checking)
4. **Recover CA-MCTS as explicit future work** (adversarial per-step capacity)
5. **Feature the detection boundary** from Approach B's honest scoping

Total effort: ~2-3 days of writing. Zero engineering. High impact on reviewer perception.

---

## 6. The Meta-Observation

Across four independent evaluations, each with 3-expert adversarial panels, one pattern is striking:

**The proposal's *ideas* are consistently rated higher than its *presentation*.**

- The consistency-oracle framing: "genuinely novel" (Skeptic), "a publishable insight" (Community Expert), "resolves a 25-year impasse" (Skeptic Synthesizer).
- Cognitive fragility: "the diamond" (Mathematician), "most genuinely novel" (Mathematician), "genuinely interesting" (Community Expert).  
- CPU-only design: 8-9/10 across all evaluations.
- Incremental architecture: "genuine risk management" (unanimous).

Yet the composite scores are 28-34/50 — below what these individual assessments would suggest. The gap is explained entirely by two factors:

1. **Zero empirical evidence.** Every evaluator flagged this. One worked example would shift scores by 3-5 points.
2. **Suboptimal framing.** Leading with cost regression instead of fragility. Calling trivial results "theorems." Including a conjectured/falsified theorem. Overclaiming coverage.

Both are fixable in weeks, not months. The proposal's survival through four adversarial evaluations without an ABANDON recommendation, despite sustained attacks on every axis, is itself the strongest evidence for CONTINUE: the ideas are robust even when the presentation is not.

---

## FINAL VERDICT: **CONTINUE**

**Composite: 34/50** (Value 7, Difficulty 6, Best-Paper 6, CPU 9, Feasibility 6)

The Cognitive Regression Prover — reframed as a "Chaos Monkey for usability" with cognitive fragility as the lead contribution — addresses a genuine unsolved problem with principled CPU-only design, sound incremental architecture, and at least two real intellectual contributions (paired-comparison error cancellation; cognitive fragility as a computable, calibration-free robustness metric). The proposal's primary risk is not technical failure but presentational weakness — fixable with modest effort. Execute the go/no-go gates (G1-G3), adopt the reframing (A1-A6), produce one worked example (S1), and this becomes a legitimate UIST contender with ~20-25% best-paper probability.
