# Approach Debate: Bounded-Rational Usability Oracle

> **Format:** For each approach the Skeptic attacks, the Math Assessor evaluates, the Difficulty Assessor grounds, and the Domain Visionary defends. A second round of direct challenges follows. The section closes with a tally of claims that survive the gauntlet.

---

## APPROACH A: Full-Theory Bisimulation Machine

### Round 1

#### Skeptic's Opening Attack

**Fatal Flaw 1 — The Markov assumption is false.**
The MDP formulation assumes the next state depends only on the current state and action. Real users carry task context, frustration, learned expectations, partial plans, and backtracking strategies. The problem statement's own working-memory decay model explicitly measures $I(S_t;\, S_{t-k})$, which *contradicts* the Markov assumption. You cannot simultaneously assume Markov transitions and model temporal memory decay.

**Fatal Flaw 2 — The "shared partition" may be unsatisfiable.**
The paired-comparison theorem requires both UI versions abstracted under the same bisimulation partition. But large structural changes—exactly the changes most likely to cause regressions—may share too little structure for any common partition to exist. The theorem is most useful precisely where it is least applicable.

**Devastating Critique.**
The entire theoretical apparatus has never been shown to detect a single real regression that simpler methods (count clicks and elements) would miss. There is zero evidence that the theory adds value over trivially computable heuristics.

---

#### Math Assessor's Evaluation

**On the Markov objection — partially valid.**
The standard resolution is *augmented state*: include working-memory contents in the state vector. This inflates the state space, but bisimulation is designed to compress it back. The real issue is that the soundness theorem's Condition (i) must carefully specify *which* Markov property is assumed.

**On paired comparison — probably salvageable.**
The theorem is likely true for UIs differing on a small number of transitions (the typical regression case). For large structural changes, error degrades to $O(k \cdot \varepsilon)$ where $k$ counts changed transitions. Getting the assumptions right is the hard part.

**Critical finding — dimensional mismatch in the cost algebra.**
The cost-algebra soundness theorem has a dimensional error: $\mu$ values are in time units (ms), while mutual information $I(S_t;\, A_t)$ is in bits. The claim that $\gamma \cdot \kappa_1 \cdot \mu_2$ upper-bounds $I(S_{t+1};\, A_{t+1} \mid S_t, A_t)$ conflates dimensions. This may render the soundness theorem *false as stated*. **Fix:** normalize everything to bits (losing the time-prediction connection) or restate as an ordinal bound (weaker but correct).

**Ornamental math identified:**

| Claim | Verdict |
|---|---|
| Trajectory approximation $\mathrm{TV} \le 0.53$ | Stepping stone only |
| Coarsening corollary | Empirical observation dressed as theorem |
| Exhaustiveness theorem | Tautological |
| Distinguishability theorem | Generic transversality |
| Intervention-specificity theorem | False — UI modifications have cross-cutting effects |

**Missing math:**

1. β calibration / sensitivity theory (**#1 gap**)
2. MDP construction formalization
3. Statistical test specification
4. Monotonicity proof for the full algebra

---

#### Difficulty Assessor's Ground Truth

| Dimension | Assessment |
|---|---|
| Novel algorithms required | 5 — bisimulation partition refinement (hard), shared partition construction (medium), cost algebra DAG composition (medium-hard), bottleneck signature extraction (medium), SMT encoding of cost constraints (medium) |
| Core engine size | 8,000–12,000 LoC |
| Time to MVP | 5–7 months |
| Integration complexity | **9/10** — monolithic system; components cannot be developed independently |

**Top risk.** Bisimulation computation doesn't scale. If progressive refinement can't beat $O(|S|^2)$, the laptop-CPU constraint kills it for UIs with >500 elements. Real accessibility trees are noisy enough to consume months of engineering unrelated to the theory.

---

#### Visionary's Defense

The Markov objection is addressable via augmented state—this is standard MDP practice. The real innovation is the bisimulation *reducing* this augmented state space back to tractability. The shared-partition concern is valid for dramatic restructurings, but the typical regression case (component refactor, option proliferation, navigation change) affects <20 % of states, keeping the shared partition useful.

On the "never detected a real regression" critique: this is the MVP-first argument, which I accept. But the theory provides *guarantees* the MVP cannot—and in regulated industries, guarantees matter. The paired-comparison theorem enables calibrated confidence thresholds that an ad-hoc scorer cannot.

**Concession.** The cost-algebra soundness theorem needs restating. Replace the time/bits inequality with an ordinal bound: *"the algebra's cost ordering is consistent with the mutual-information ordering."* This is weaker but correct, and still serves regression detection.

---

### Round 2

**Skeptic → Visionary:**
> "If the soundness theorem must be weakened to ordinal, and the paired-comparison theorem has a $k \cdot \varepsilon$ degradation for large changes, and the bisimulation algorithm doesn't exist yet—what formal guarantee actually survives? You're left with 'our heuristic approximately preserves orderings for small changes.' That's Approach B with extra steps."

**Math Assessor → Skeptic:**
> "Your 'count clicks and elements' baseline is a straw man. Fitts' law captures target size and distance; Hick's law captures choice complexity. These are quantitative models, not click counts. The real question is whether the *compositional* structure (algebra) adds signal over the *additive* model. This is an empirical question that should be answered, not assumed away."

**Difficulty Assessor → Both:**
> "The bisimulation algorithm is the make-or-break. If someone invents an efficient bounded-rational partition refinement—which is a genuinely novel algorithm problem—the entire approach works. If not, fall back to heuristic state-space reduction and lose the formal guarantees. The approach should be structured around this pivot point."

---

### Surviving Claims

| # | Claim | Verdict |
|---|---|---|
| 1 | Paired-comparison theorem (with $k \cdot \varepsilon$ qualification) | **Genuine, load-bearing** |
| 2 | Free-energy unification of cognitive laws | Genuine but adds complexity; value over additive unclear |
| 3 | Bisimulation for scaling | Needed only if additive on raw trees doesn't scale beyond ~200 elements |
| 4 | Cost algebra | Needs fundamental restatement; ordinal bound may survive |
| 5 | Bottleneck taxonomy | Engineering value, not theoretical value |

---

## APPROACH B: Lean Profiler — "Cognitive git diff"

### Round 1

#### Skeptic's Opening Attack

**Fatal Flaw — This is CogTool-in-CI, a 20-year-old idea that already failed.**
CogTool had 15 years of research attention and never achieved CI/CD adoption. Repackaging KLM prediction with JSON output does not solve the underlying adoption problem.

**Additive models are known wrong.**
Setting $\gamma = \alpha = 0$ ignores the primary mechanism of real cognitive difficulty: interaction effects. Confident verdicts from a model known to be wrong generate false negatives on subtle regressions—exactly the ones that matter.

**Devastating Critique.**
The MVP occupies the worst strategic position: too simple to be intellectually interesting, too complex to be casually adopted, and too similar to existing tools to differentiate itself.

---

#### Math Assessor's Evaluation

**On additive being "known wrong" — true but beside the point.**
The relevant question is whether additive produces correct *orderings*, not correct magnitudes. For the dominant failure modes (option proliferation, target shrinking, navigation depth increase), the regression verdict is parameter-independent: it depends only on structural changes. This is the interval-arithmetic insight, and it is genuinely powerful.

**Missing math.**
$\beta$ does not appear in the lean approach (there is no softmax policy), which eliminates the **#1 parameter problem**. But this also means there is no model of user *behavior*—just a cost estimate for assumed-optimal behavior. The lean approach implicitly assumes users take the shortest path, which is unrealistic for complex UIs.

**Key assessment.** The lean approach's math is **100 % load-bearing and 0 % ornamental.** Its weakness is not in what it claims but in what it *cannot capture*.

---

#### Difficulty Assessor's Ground Truth

| Dimension | Assessment |
|---|---|
| Novel algorithms required | 0 — all adaptation of existing techniques |
| Core engine size | 2,500–4,000 LoC |
| Time to MVP | 6–8 weeks |
| Integration complexity | **4/10** — clean linear pipeline |
| Difficulty rating | **4/10** — important engineering, not research |

**Top risks.** Element-matching fragility (refactors break matching → false positives/negatives). Additive model misses interaction-effect regressions. Bounding-box availability varies across platforms.

---

#### Visionary's Defense

CogTool failed because it required manual task specification and didn't target CI/CD. The lean approach addresses both: automatic inference for common patterns, first-class CI integration. The technology landscape has changed—accessibility trees are better (WCAG mandates), CI/CD pipelines are universal, developers expect automated quality gates.

The "too simple to be interesting" critique misses the point. The value is in the *integration* and *validation*, not the algorithms. If we show $F_1 \ge 0.85$ on 1,000+ real PRs—something no prior work has attempted—that is a strong empirical contribution.

**Concession.** The approach needs one distinguishing insight to be publishable. The parameter-independence result for dominant failure modes is that insight: *"for the most common regression types, the verdict is completely independent of parameter calibration."* This is elegant, surprising, and practically important.

---

### Round 2

**Skeptic → Visionary:**
> "Your $F_1 \ge 0.85$ target is measured against your own synthetic mutations and biased issue-tracker labels. Against a rigorous human-validated ground truth, I predict $F_1 \le 0.65$—too low for a CI gate."

**Math Assessor → Visionary:**
> "The parameter-independence result is real but narrow. It covers options-added and target-shrunk. It does NOT cover: reordering (depends on visual-search model), grouping changes (depends on perceptual model), memory-load changes (not in the additive model). The 85 % coverage claim needs empirical backing."

**Difficulty Assessor → Visionary:**
> "The 6–8 week timeline assumes clean accessibility trees. Budget 3–4 extra weeks for cross-browser accessibility-tree normalization alone. Realistic timeline: **10–12 weeks**."

---

### Surviving Claims

| # | Claim | Verdict |
|---|---|---|
| 1 | Parameter-independence for dominant failure modes | **Genuine, valuable** |
| 2 | Interval arithmetic for robust verdicts | Sound, practical |
| 3 | Structural tree diff as regression signal | Plausible, needs validation |
| 4 | 85 % regression coverage | Speculative, needs empirical backing |
| 5 | CI/CD integration value | Real but not novel |

---

## APPROACH C: Adversarial Cognitive Fuzzer — "Usability Chaos Engineering"

### Round 1

#### Skeptic's Opening Attack

**The fragility metric sounds great but may be meaningless.**
$F(M) = \max_\beta C(\tau) - \min_\beta C(\tau)$ penalizes UIs that are *specifically optimized* for a particular user capacity. A UI designed for expert users (high $\beta$) that also works for novices (low $\beta$) scores identically to a UI that is mediocre for everyone. Fragility does not distinguish "works well for some, badly for others" from "works badly for everyone."

**Cognitive cliffs require full MDP infrastructure from Approach A.**
You cannot compute $\pi_\beta(a \mid s)$ without first constructing the MDP and solving for Q-values. The adversarial fuzzer inherits all of Approach A's engineering difficulty (MDP construction, value iteration) *plus* adds adversarial optimization on top.

**200 MDP evaluations at <1 s each assumes abstract MDPs exist.**
Without bisimulation, you are solving the full MDP 200 times. With bisimulation, you have imported Approach A's hardest unsolved problem.

---

#### Math Assessor's Evaluation

**The cliff-location theorem is genuinely novel and load-bearing.**
It enables analytical pre-computation of cliff candidates from Q-value crossings—this is the key algorithmic insight that makes the adversarial approach computationally feasible. The theorem is straightforward to prove (softmax crossing analysis) and directly useful.

**The fragility decomposition's $O(\gamma \cdot F_{\max}^2)$ correction is problematic.**
If $\gamma$ is unknown (same calibration gap as Approach A), the correction bound is meaningless. For additive models ($\gamma = 0$), the decomposition is exact—but then you have lost the interaction effects that motivate the approach.

**Fisher-metric variant has worse β sensitivity.**
Fisher information scales as $\beta^2$, making geodesic distances more sensitive to $\beta$ misspecification than TV-based distances. Not recommended.

**The fragility metric IS model-independent in a meaningful sense.**
It compares the UI to *itself* across the capacity space. Two different cost models will generally agree on which UI has higher fragility, even if they disagree on absolute costs. This is a genuine advantage over Approaches A and B.

---

#### Difficulty Assessor's Ground Truth

| Dimension | Assessment |
|---|---|
| Novel algorithms required | 3 — CA-MCTS (medium-hard), Bayesian optimization with discontinuous kernels (hard), analytical cliff pre-computation (medium) |
| Core engine size | 6,000–10,000 LoC (assumes MDP construction exists) |
| Time to MVP | 4–6 months (assumes MDP infra from A/B is available) |
| Integration complexity | High — requires MDP infrastructure from Approach A |

**Top risk.** Adversarial optimization exceeds CI/CD time budgets on complex UIs. 200 MDP evaluations × 1–5 s each = 200–1,000 s, potentially exceeding the 300 s target for large UIs.

---

#### Visionary's Defense

The fragility metric's "model independence" is the key selling point. The Skeptic is right that fragility does not distinguish "good for some, bad for others" from "bad for everyone"—but we can augment with absolute-cost floor checks. The core insight survives: *variance across the capacity space is an independent signal not captured by any other metric.*

On MDP dependency: yes, we need MDP infrastructure. But we can use a *simplified* MDP (Approach B's task-path representation with softmax action selection) rather than Approach A's full bisimulation. Cliff detection only requires Q-values at states on the task path, not all states.

**Concession.** CI/CD wall-clock time is a real concern. The approach works best as a nightly/weekly analysis rather than a per-PR gate—which is still valuable, analogous to nightly performance-regression suites.

---

### Round 2

**Skeptic → Visionary:**
> "Nightly analysis means the PR that caused the regression is already merged. You've lost the core value proposition—catching regressions before merge."

**Math Assessor → Visionary:**
> "The simplified MDP (task-path with softmax) doesn't support the cliff-location theorem, which requires full Q-values at all reachable states. You need the analytical cliff pre-computation to make the approach feasible in CI/CD. This is a genuine technical tension."

**Difficulty Assessor → Visionary:**
> "The CA-MCTS algorithm is interesting but unproven. Standard MCTS with adversarial nature players has convergence issues when the nature player's action space is continuous ($\beta$ parameters). You'd need to discretize $\beta$ to a grid, losing the analytical cliff-detection advantage."

**Visionary → All:**
> "The cliff-location theorem identifies a *finite* set of critical $\beta$ values analytically. We only need to evaluate the MDP at these values plus a small neighborhood. This is not continuous optimization—it is targeted evaluation at analytically determined points. The computational budget is $O(|\text{cliff\_candidates}| \times \text{trajectory\_sampling\_cost})$, which for typical UIs with $\le 50$ cliff candidates is well within CI/CD budgets."

---

### Surviving Claims

| # | Claim | Verdict |
|---|---|---|
| 1 | Cognitive fragility metric | **Genuinely novel, model-independent, valuable signal** |
| 2 | Cliff-location theorem | **Load-bearing, enables feasible computation** |
| 3 | Inclusive-design angle | Unique positioning, strong at CHI |
| 4 | Adversarial synthesis over capacity space | Novel capability no existing tool provides |
| 5 | CI/CD feasibility | Questionable for per-PR; viable for nightly |

---

## CROSS-CUTTING DEBATE

### Skeptic's Universal Challenges

**To all approaches — the accessibility tree is a leaky abstraction.**
It was designed for assistive technology, not cognitive modeling. It omits visual-layout relationships, animation, scroll position, viewport-relative positioning, and dynamic content loading—all of which affect usability. Every approach is building models on incomplete data.

**To all approaches — does anyone actually want this?**
Teams that care about usability do user research. Teams that don't care won't adopt a CI tool. The assumed middle ground—teams who care enough to install a tool but not enough to test with humans—may be vanishingly small.

**To all approaches — the "consistency oracle" framing is weaker than it appears.**
A "consistent" oracle that is consistent with *itself* is trivially achievable. The hard claim is consistency with *human judgments*, which requires external validation—exactly what all approaches defer.

---

### Math Assessor's Universal Findings

**One theorem matters across all approaches.**
The paired-comparison argument: differential cost estimation is more accurate than absolute estimation under a shared analysis framework. This is approach-independent and should be the flagship result.

**The bottleneck-taxonomy theorems are universally ornamental.**
Drop the formal claims; keep the engineering taxonomy.

**$\beta$ is the #1 parameter problem.**
All approaches that use softmax policies depend on $\beta$. None provide principled calibration. The lean approach's advantage is avoiding $\beta$ entirely, at the cost of modeling user behavior.

---

### Difficulty Assessor's Universal Findings

**The accessibility-tree parser is the universal engineering bottleneck.**
All approaches need it; none discuss it adequately. Budget **40 % of engineering time** for parsing, normalization, and edge cases.

**MDP infrastructure is shared between A and C.**
If A is built first, C is an incremental extension. If C is attempted without A's infrastructure, it inherits A's hardest problems.

**Approach B is the only one deliverable in <3 months.**
Both A and C require ≥5 months to MVP.

---

### Visionary's Cross-Cutting Response

The accessibility-tree objection is valid but overstated. For *structural* regressions—which we explicitly scope to—the accessibility tree captures the relevant signal: navigation structure, element counts, groupings, interaction flows. We explicitly disclaim visual regressions and position as complementary to screenshot-diff tools.

The demand-side risk is real but mitigable. The target audience is not "all teams" but **design-system teams at scale** (50+ components, multiple products, CI/CD pipelines). These teams exist, they have this problem, and they currently have no automated signal. Even a modest signal is better than zero.

On external validation: the retrospective validation against published human data (CogTool datasets, Oulasvirta group) breaks the circularity without new human studies. **All approaches should commit to this validation.**

---

## SYNTHESIS: What Survives the Gauntlet

### Load-Bearing Results (keep and prove)

1. **Paired-comparison theorem** — differential estimation under shared framework is more accurate than absolute estimation. Approach-independent. Flagship result.
2. **Parameter-independence for dominant failure modes** — for option proliferation and target shrinking, the regression verdict requires no calibration. Lean approach's core insight.
3. **Cliff-location theorem** — analytical identification of critical $\beta$ values from Q-value crossings. Approach C's core insight.
4. **Cognitive fragility as model-independent signal** — variance across the capacity space is a novel, complementary metric.

### Weakened but Salvageable (restate carefully)

5. **Cost-algebra soundness** — ordinal bound survives if dimensional mismatch is corrected.
6. **Bisimulation for state-space reduction** — value depends on whether additive models on raw trees scale past ~200 elements.

### Ornamental (drop or demote to remarks)

7. Trajectory approximation TV bound
8. Coarsening corollary
9. Exhaustiveness / distinguishability / intervention-specificity theorems
10. Bottleneck taxonomy *theorems* (keep the taxonomy, drop the formalism)

### Unresolved Tensions

| Tension | Status |
|---|---|
| $\beta$ calibration | **Open problem.** No approach solves it; the lean approach sidesteps it. |
| Accessibility-tree completeness | **Acknowledged limitation.** Scope to structural regressions; disclaim visual. |
| Empirical validation of $F_1 \ge 0.85$ | **Must be tested.** No approach can claim this a priori. |
| CI/CD wall-clock budget for Approach C | **Partially resolved** by cliff-location theorem; needs benchmarking. |
| Market demand | **Mitigated** by scoping to design-system teams; not eliminated. |
