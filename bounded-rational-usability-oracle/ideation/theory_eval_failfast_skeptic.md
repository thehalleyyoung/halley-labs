# Theory Gate: Fail-Fast Skeptic Verification

**Proposal:** proposal_00 — The Cognitive Regression Prover  
**Role:** Fail-Fast Skeptic (adversarial verification after three prior CONTINUE verdicts)  
**Prior scores:** Skeptic 33/50, Mathematician 31/50, Community Expert 28/50  
**Date:** 2026-03-04

---

## VERDICT: ABANDON

---

## Scores

| Axis | Score | Rationale |
|------|-------|-----------|
| 1. Extreme Value | **4/10** | Trivial baseline kills Layer 1. CogTool adoption failure repeats. Fabricated coverage number. |
| 2. Genuine Software Difficulty | **5/10** | Layer 1 is difficulty 4 (proposal admits this). Layer 2 is contingent on an unsolved engineering problem. Layer 3 is deferred. |
| 3. Best-Paper Potential | **3/10** | Zero data, three of four theorems unproven, 17% best-paper probability by the most sympathetic prior evaluator. |
| 4. Laptop-CPU & No-Humans | **8/10** | Genuinely strong. Conceded. |
| 5. Feasibility | **4/10** | 84% compound risk, zero implementation progress, parser graveyard, fabricated timelines. |
| **TOTAL** | **24/50** | Below ABANDON threshold. |

---

## The Groupthink Problem

All three prior evaluators recommended CONTINUE. This unanimity is suspicious given the severity of the weaknesses they themselves identified. Here is the groupthink diagnosis:

### Evidence of convergent bias

1. **All three evaluators deferred to the "incremental architecture" defense.** The Skeptic evaluation explicitly states "the incremental architecture prevents any single failure from destroying the project." The Mathematician evaluation calls it "the saving grace." The Community Expert calls it "the most important finding." All three used the same escape hatch to avoid confronting the possibility that Layer 1 alone is insufficient.

2. **All three identified the same fatal flaws but none pulled the trigger.** Every evaluator flagged: fabricated 50-70% coverage, retrospective validation uncertainty, evaluation circularity, trivial baseline competition. Yet all three concluded CONTINUE. When three independent reviewers identify four potentially fatal flaws and none recommends rejection, anchoring bias is operative — the prior CONTINUE verdicts from the initial evaluation set an anchor that subsequent evaluators needed extraordinary evidence to overcome.

3. **The Community Expert scored 28/50 and still recommended CONTINUE.** A 28/50 is 56% — a failing grade by any standard. The justification was "the Skeptic's inability to recommend ABANDON despite six attack vectors is the strongest signal." This is circular reasoning: the Skeptic didn't abandon because the Auditor didn't abandon because the Synthesizer didn't abandon. Nobody wanted to be the first to say no.

4. **Score inflation on the Value axis.** The proposal self-scored Value at 9. The evaluators gave 7, 6, 5 — a consistent 2-4 point deflation, but starting from 9 anchors the range upward. If the self-score hadn't been visible, Value would likely have landed at 3-5 across the board. The "real unaddressed gap" framing accepts the proposal's problem definition uncritically, without asking: is this an unaddressed gap because nobody can solve it, or because nobody needs the solution at the proposed complexity level?

### The evaluation should have been structured as a pre-registration

The three evaluators should have committed to ABANDON thresholds *before* seeing each other's scores. Without pre-registered kill criteria, the group defaults to CONTINUE because CONTINUE is the low-stakes recommendation (you can always abandon later) and ABANDON is irreversible.

---

## The Single Strongest Reason to ABANDON

### The Trivial-Baseline Kill Chain

This is the argument that should end the project:

**Step 1: Layer 1's "parameter-free verdicts" ARE a trivial baseline.**

The proposal's parameter-free regression types are:
- Option proliferation: `n_after > n_before` → regression
- Target shrinking: `W_after < W_before` → regression  
- Navigation depth increase: `d_after > d_before` → regression

These are `if` statements. They require no cost model, no interval arithmetic, no accessibility-tree parsing beyond element counting, no Fitts' law, no Hick-Hyman law, and no information theory. A script that counts interactive elements, measures bounding boxes, and computes DOM depth performs identical regression detection for these cases. The proposal calls this "50-70% of real cases" (an unsubstantiated number), which means the proposal claims that 50-70% of its value comes from three `if` statements.

**Step 2: Layer 1's non-parameter-free verdicts are circular.**

For the remaining 30-50% of regressions, Layer 1 uses interval arithmetic over Fitts'/Hick's cost functions. The regression verdict is: "our cost model says cost increased." The external validation that this cost increase corresponds to a real usability regression requires retrospective ordinal validation — which has a 30% self-assessed failure probability, uses datasets that predate modern accessibility trees, and would have sample sizes too small for statistical significance even if the data existed.

Without external validation, these verdicts are definitionally circular: the system detects the thing it defines as a regression.

**Step 3: Layer 2 is the escape from circularity, but it depends on solving an unsolved problem.**

Layer 2 adds the bounded-rational MDP formulation, paired-comparison theorem, and cognitive fragility metric. The fragility metric is genuinely self-referential (comparing a UI to itself across capacity space) and could partially break the evaluation circle. But Layer 2 requires constructing task-path MDPs from accessibility trees — a problem the proposal itself identifies as "where prior automated usability tools have died" (Section 8). The proposal's mitigation is a "restricted UI grammar" covering "~80% of common interaction patterns" — another unsubstantiated coverage estimate.

**Step 4: The kill chain closes.**

- Layer 1 is achievable but trivially competitive → minimal value
- Layer 1's non-trivial contributions are circular → no external validation
- Layer 2 breaks circularity but depends on a known-hard problem → high execution risk
- Layer 3 is deferred → not evaluable

The guaranteed deliverable (Layer 1) adds marginal value over a trivial competitor. The valuable deliverable (Layer 2) is high-risk and months away. The full vision (Layers 1-3) is years of work with an unproven central theorem. At no point on this trajectory does the project produce something that is simultaneously achievable and significantly better than trivial alternatives.

**This is not a risk to manage. It is a structural deficiency in the value proposition.**

---

## Detailed Axis Analysis

### Value: 4/10

**The proposal's "desperate need" framing is wrong.** Design-system teams do not have "zero automated signal for structural usability regressions." They have:

- **axe-core / Pa11y:** Catches element proliferation, missing labels, navigation structure violations. ~100 rules, ~10-second runtime, zero cognitive modeling needed.
- **Storybook interaction tests:** Record-and-replay user flows, catch functional regressions that correlate with usability regressions.
- **Design token enforcement:** Component libraries constrain target sizes, spacing, hierarchy depth at the design-system level, preventing the exact "parameter-free" regressions this proposal targets.
- **Screenshot diff (Chromatic, Percy):** Catches layout changes that the proposal explicitly excludes.
- **Manual design review on PRs:** The actual workflow in every design-system team I'm aware of. Fast, effective, and sensitive to context.

The proposal positions itself in a gap that is narrower than claimed: structural changes that (a) don't violate accessibility rules, (b) aren't caught by interaction tests, (c) aren't prevented by design tokens, (d) aren't visible in screenshots, and (e) aren't caught by a human reviewer looking at the PR diff. This is a real but *thin* gap, not a desperate need.

**CogTool died for a reason.** The proposal acknowledges "15+ years of attention, approximately zero CI/CD adoption" but does not analyze *why*. CogTool failed because: (1) the setup cost (manual task specification) exceeded the perceived benefit, (2) cognitive cost predictions did not match developer intuitions about usability, and (3) the output was difficult to act on. This proposal's mitigation for (1) is "recording-based extraction for common patterns" (unbuilt), for (2) is the consistency-oracle framing (unvalidated), and for (3) is the bottleneck taxonomy (Layer 2+). The adoption barriers are acknowledged but not solved.

**The 50-70% coverage number is fabricated.** All three prior evaluators flagged this. The number appears nowhere in the literature. The proposal body says "estimated 50-70%"; the Layer 1 deliverable says "70-85%." These are internally inconsistent and externally ungrounded. If the true number is 20%, Layer 1's value proposition collapses.

### Difficulty: 5/10

**Counting difficulty by layers is misleading.** The proposal presents difficulty as a ladder: Layer 1 (4-5), Layer 2 (7-8), Layer 3 (8-9). But the realistic deliverable is Layer 1 + partial Layer 2. By the proposal's own estimates and the unanimous evaluator recommendation to defer Layer 3:

- Layer 1 difficulty: 4-5 (proposal admits "known algorithms, interval arithmetic")
- Layer 2 difficulty: 7-8, but discounted by:
  - MDP construction risk (25-40% failure)
  - Proof completion risk (sketch exists, not proven)
  - Fragility decomposition risk (sketch with unresolved interactions)

Expected difficulty = 0.6 × Layer1 + 0.4 × Layer2 = 0.6 × 4.5 + 0.4 × 7.5 = 5.7 → round to 5.

Two of the four theorems are self-described as "Easy." The Mathematician evaluation rates parameter-independence as "a tautology dressed as a theorem." The bisimulation algorithm (the hardest component) is deferred to future work. The remaining novel research (paired-comparison proof, fragility computation) is medium-difficulty — the proposal itself rates the paired-comparison proof as "Medium" and cliff location as "Easy."

### Best-Paper Potential: 3/10

**The evidence is devastating:**

- **Zero empirical results.** Not one worked example. Not one UI pair processed. Not one accessibility tree parsed. Not one cost estimate computed. The Community Expert is exactly right: "This is disqualifying at CHI, UIST, or any HCI venue for best-paper consideration."

- **Three of four theorems are incomplete.**
  - Theorem 1 (paired-comparison): proof sketch only, tight constant TBD, degrades for real use cases (k > 50)
  - Theorem 2 (parameter-independence): trivial / tautological
  - Theorem 3 (fragility): partially proven, decomposition sketch with unresolved interactions
  - Theorem 4 (cost algebra): conjectured, original proof was false, 35% self-assessed failure

- **The "three surprising results" narrative collapses under scrutiny.** The Skeptic evaluator's deflation is correct:
  - Parameter-independence = monotonicity of standard functions (textbook)
  - Paired-comparison tightness = error cancellation under shared bias (known statistical phenomenon, formalized for MDPs)
  - Fragility = max-min sensitivity analysis (known framework, applied to a new domain)
  
  The novelty is in the *application domain*, not the *techniques*. This is a reasonable CHI/UIST contribution, not a best-paper candidate.

- **The Community Expert's 17% probability estimate is generous.** That estimate assumed all three narrative moves land simultaneously. With zero implementation, zero data, and three unfinished proofs, I estimate best-paper probability at < 5%.

- **Venue mismatch is real.** UIST demands working demos. CHI demands human validation. This has neither and has no concrete plan to produce either within the evaluation window.

### CPU/No-Humans: 8/10

**Conceded.** This is genuinely well-designed. Structural saliency from accessibility trees (no vision models), embarrassingly parallel MC sampling, CPU-native solvers, published psychophysical parameters (no training). The only deduction is uncertainty about accessibility-tree reconstruction for retrospective validation and the unsubstantiated ≤10K state-space bound.

### Feasibility: 4/10

**The compound risk is disqualifying.**

The proposal's own risk register:
- 30% retrospective validation fails → consistency-oracle claim dies
- 25% tree quality insufficient → all layers produce garbage  
- 40% bisimulation intractable → Layer 3 fails
- 35% cost-algebra proof fails → Layer 3 formal guarantees evaporate

Probability that ALL risks are avoided: 0.70 × 0.75 × 0.60 × 0.65 = **20.5%**. Probability that at least one triggers: **79.5%**.

Even scoped to Layers 1-2 only (removing bisimulation and cost-algebra risks): 0.70 × 0.75 = **52.5%** success probability on the two critical risks alone.

**The timeline is fantasy.** Layer 1 is budgeted at 8 weeks. The components:
1. Accessibility-tree parser + cross-browser normalizer: The proposal budgets "40% of Layer 1 engineering time" (3.2 weeks) and separately budgets "3-4 weeks for cross-browser normalization." These overlap but don't sum — the parser alone, including normalization, is 6-8 weeks for a senior engineer based on the complexity described and prior failure history. The Community Expert correctly notes: "The accessibility-tree parser alone is 8+ weeks."
2. Semantic tree alignment (three-pass algorithm): 1-2 weeks
3. Task-flow specification DSL: 1-2 weeks
4. Additive cost model + interval arithmetic: 1 week
5. CI/CD integration: 1 week
6. Benchmark curation: 2-3 weeks

Realistic Layer 1 timeline: 14-20 weeks, not 8. The proposal's 8-week estimate is off by 2x.

**Zero implementation progress.** theory_bytes: 0, impl_loc: 0, code_loc: 0. The theory directory is empty. Not a single line of code, not a single proof, not a single experiment. The proposal is pure blueprint. At this stage, feasibility should be scored on demonstrated capability, not aspirational plans.

---

## Rebuttal of the "Incremental Architecture" Defense

The three prior evaluators all relied on the incremental architecture as the key reason to CONTINUE. I argue this defense is a motte-and-bailey:

**The bailey (what's being sold):** A formally grounded usability oracle with bounded-rational behavioral models, cognitive fragility analysis, and bisimulation-based scaling. Paired-comparison theorem provides order-of-magnitude tighter error bounds. Fragility metric is a genuinely new research direction. Best-paper potential at UIST/CHI.

**The motte (what's achievable):** A structural diff engine that flags when elements increase, targets shrink, or navigation deepens. Fitts'/Hick's cost labels attached. CI/CD integration. Difficulty 4-5, novelty near-zero.

**The retreat pattern:** When I challenge the bailey (Theorem 4 is false, bisimulation is intractable, retrospective validation fails), the response is "Layer 1 delivers standalone value." When I challenge the motte (Layer 1 is a trivial baseline), the response is "Layer 2 adds theoretical depth."

This is structurally identical to:
- "We're building a self-driving car" (bailey)
- "But for now we have a lane-departure warning" (motte)
- When challenged on self-driving: "Lane-departure warning is already useful"
- When challenged on lane-departure: "Self-driving is the real contribution"

The incremental architecture is not a risk-management strategy. It is a way to avoid committing to what this project actually is. If it's Layer 1, score it as Layer 1. If it's the full vision, accept the full risk. You cannot have it both ways.

---

## What About Fragility? (Steelmanning the Best Piece)

I acknowledge that the cognitive fragility metric (Theorem 3) is the proposal's genuine diamond. Comparing a UI to itself across the human capacity space is a genuinely novel formulation that sidesteps evaluation circularity. The cliff-location theorem is compact and provable. "Chaos Monkey for usability" is a compelling pitch.

**But fragility alone cannot save this proposal, for three reasons:**

1. **Fragility requires Layer 2's MDP infrastructure.** Computing F(M) requires softmax policy computation over task-path MDPs. Building task-path MDPs from accessibility trees is the known-hard problem. Without MDPs, fragility is a definition without a computation.

2. **Fragility is a secondary signal.** The proposal correctly positions fragility as activating "when the simpler analysis is inconclusive." But if the simpler analysis (Layer 1) is trivially competitive, and the harder analysis (fragility) requires solving the MDP problem, the project's value is concentrated in a narrow band: UIs where (a) Layer 1 is inconclusive, (b) the MDP construction succeeds, and (c) fragility reveals something actionable. The size of this band is unknown.

3. **Fragility has not been tested on a single UI.** The concept is appealing. The math is clean. The computational path exists. But nobody has computed F(M) for any real interface. It could produce vacuous results (all UIs are equally fragile), degenerate results (fragility is dominated by a single bottleneck), or surprising results (novel failure modes revealed). We don't know which, because there is zero empirical work.

If the proposal were "just" the fragility metric — a focused 3-month project to define, compute, and validate cognitive fragility on 20 real UI pairs — it would be a strong CONTINUE. But embedded in a 9-month three-layer system with fabricated coverage numbers, unproven theorems, and a parser graveyard, the diamond is buried too deep.

---

## Realistic Probability Assessment

| Outcome | Probability |
|---------|------------|
| Full vision delivered (all three layers, all theorems) | < 5% |
| Layers 1-2 delivered, strong UIST paper | ~20% |
| Layer 1 delivered, retrospective validation passes, CHI LBW / ICSE tool paper | ~30% |
| Layer 1 delivered, retrospective validation fails, unpublishable tool | ~25% |
| Project abandoned before Layer 1 completes (parser infeasible) | ~20% |

**Expected best-paper probability: < 5%.** Expected publication at a top venue (UIST/CHI full paper): ~20%. Expected any publication: ~50%.

For a project requiring 6-9 months of full-time work, a 20% probability of a top-venue publication and < 5% probability of best-paper is below the threshold for investment.

---

## Single Strongest Argument for CONTINUE

**The consistency-oracle framing is genuinely novel and the fragility metric is genuinely new.** No prior work formalizes usability regression detection as differential inference under shared analysis. No prior work defines cognitive fragility as capacity-space robustness. Even the Skeptic evaluations across all rounds acknowledge these as real contributions. If the MDP construction problem is solved (a hard but not impossible engineering challenge), the combination of paired-comparison tightness and fragility analysis is publishable at UIST/CHI. The incremental architecture means Layer 1 investment (8-14 weeks) provides a working tool regardless, and Layer 2 investment is gated on Layer 1 success. The call-option structure bounds downside.

**Why this argument fails:** "Novel framing" is a necessary but not sufficient condition for a research contribution. Novel framing + zero data + unfinished proofs + unbuilt system = a research proposal, not a research contribution. The question is not "is this idea worth pursuing?" (yes) but "does this proposal, as constituted, have sufficient probability of producing a best-paper-worthy result to justify the investment over alternative projects?" (no).

---

## Single Strongest Argument for ABANDON

**The guaranteed deliverable (Layer 1) is trivially competitive, and the valuable deliverable (Layer 2) depends on solving a problem that has killed every prior attempt.**

Layer 1's parameter-free verdicts are structurally identical to a 50-line axe-core wrapper. Layer 1's cost-model verdicts are circular without retrospective validation. Retrospective validation has a 30% failure probability using datasets that lack accessibility trees. Layer 2's genuinely novel contributions (fragility, paired-comparison) require constructing task-path MDPs from accessibility trees — the engineering problem that killed CogTool, SUPPLE, and every other automated usability tool that attempted it.

The project therefore occupies an uncomfortable position: what it can certainly build doesn't justify its theoretical framework, and what would justify its theoretical framework it may not be able to build. This is not a risk to manage — it is a gap between the guaranteed deliverable and the required deliverable for the research claims.

**The "incremental architecture" does not resolve this gap.** It merely separates the trivial deliverable (Layer 1) from the risky deliverable (Layer 2) in time, creating the illusion that Layer 1's existence de-risks Layer 2. It does not. Layer 2's risk is intrinsic: either MDP construction from accessibility trees works or it doesn't. Layer 1's existence does not change this probability.

---

## Final Assessment

| Metric | This Proposal | ABANDON Threshold |
|--------|--------------|-------------------|
| Best-paper probability | < 5% | < 10% |
| Top-venue probability | ~20% | < 25% |
| Fabricated claims | ≥ 2 (coverage %, validation feasibility) | ≥ 1 |
| Unproven load-bearing theorems | 2 of 4 | > 50% |
| Zero empirical results | Yes | Yes at theory gate |
| Trivial-baseline competition | Unaddressed | Unaddressed |
| Prior tool adoption failure | Unanalyzed | Unanalyzed |

Every indicator exceeds the ABANDON threshold. The three prior evaluators were anchored by the proposal's self-scores and by each other's CONTINUE verdicts. The incremental architecture provided a shared escape hatch that prevented any evaluator from confronting the core structural problem: this project's achievable deliverable is trivially competitive, and its competitive deliverable is not achievable with acceptable probability.

**VERDICT: ABANDON.**

If the team insists on CONTINUE despite this assessment, the following conditions are non-negotiable:

1. **Week 1:** Implement the trivial baseline (50-line axe-core element-count diff). If Layer 1 cannot demonstrate Δτ ≥ 0.15 over this baseline on 10 real UI pairs, ABANDON.
2. **Week 2:** Name 10 specific UI pairs with confirmed accessibility-tree availability AND published human performance orderings. Not "CogTool datasets" — specific files, specific URLs. If fewer than 10 exist, ABANDON.
3. **Week 3:** Parse one real accessibility tree (Material UI v4) into a task-path MDP with ≥ 10 states. If this cannot be done, Layer 2 is dead and the project reduces to Layer 1 (which fails the trivial-baseline test). ABANDON.
4. **Remove all fabricated numbers** (50-70%, 70-85%) from all documents immediately.
