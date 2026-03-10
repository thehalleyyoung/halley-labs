# The Cognitive Regression Prover: A Three-Layer Usability Oracle with Incremental Formal Guarantees

**One-line summary:** Build a lean structural usability profiler first (weeks), layer bounded-rational theory for calibrated confidence and fragility analysis second (months), and add bisimulation-based scaling third — delivering immediate CI/CD value while accumulating provably tight differential guarantees that no LLM or heuristic tool can match.

---

## 1. Core Thesis

Usability regression detection is fundamentally a *differential* inference problem: it requires accurate estimation of cost *differences* between UI versions, not accurate absolute cost prediction. We prove that differential estimation under a shared analysis framework achieves error bounds an order of magnitude tighter than absolute estimation (the paired-comparison theorem), and that for the most common regression types the verdict is entirely parameter-free (the parameter-independence result). Together, these results establish a new class of automated usability oracle — one that is conservative, quantitative, deterministic, and incrementally improvable from a working MVP to a formally grounded system with cognitive fragility analysis.

---

## 2. Extreme Value Delivered

### Who needs this desperately

Design-system teams maintaining 50+ components across multiple products — enterprise SaaS, healthcare EHR systems, government digital services, financial platforms. These teams ship daily, maintain hundreds of UI components, and have *zero* automated signal for structural usability regressions. Accessibility linters (axe-core, Pa11y) catch WCAG violations but not cognitive cost increases. Screenshot-diff tools (Chromatic, Percy) catch visual regressions but miss structural ones entirely. The gap between "passes a11y lint" and "a human tested this" is where regressions live for months.

### What becomes possible

Every pull request that modifies UI structure receives an automatic cognitive cost diff within 60 seconds:

```
$ npx @usability-oracle/cli diff --before v2.3 --after HEAD --task checkout-flow

  Step 3 (product-selection):
    ⚠ +2.1 bits Hick cost (4→18 options)  [PARAMETER-FREE: structural increase]
    Bottleneck: Choice paralysis
    Fragility: cliff at β=2.3 (novice users lose optimal path)
    Suggested: Progressive disclosure

  Overall: +18% expected task cost | Fragility: +0.4 | Verdict: REGRESSION (high confidence)
```

Teams that currently run usability studies quarterly can restrict those studies to genuinely novel questions the oracle cannot answer, while catching structural regressions in hours rather than months.

### Why not LLMs

CI/CD regression detection imposes four requirements LLMs fundamentally cannot satisfy:

1. **Determinism.** Same UI pair → same verdict across runs. LLM outputs are stochastic and prompt-sensitive.
2. **Quantitative comparability.** Scalar cost differentials that can be thresholded, trended, and compared across releases. LLM outputs require further interpretation.
3. **Monotonicity.** If version B is strictly worse than A on all task dimensions, the system must never report improvement. LLMs provide no such guarantee.
4. **Formal error bounds.** Teams calibrate thresholds to their risk tolerance with meaningful confidence intervals. LLMs offer no error theory.

We position this system as *complementary* to LLM-based usability critique: LLMs provide broad qualitative feedback during design exploration; the oracle serves as a quantitative CI gate with provable properties.

---

## 3. Architecture: Three-Layer Design

The system is structured as three layers that build incrementally, each delivering standalone value while enabling the next.

### Layer 1: Lean Profiler (MVP, Weeks 1–8)

**What it is.** A structural diff engine operating directly on accessibility trees, applying calibrated cognitive cost functions with interval arithmetic, producing parameter-independent regression verdicts for dominant failure modes.

**Components:**

1. **Accessibility-tree parser and normalizer.** Extract semantic structure from platform accessibility APIs (web via ARIA/DOM, native via platform APIs). Normalize cross-browser/cross-platform differences. Output: a typed tree of interactive elements with roles, labels, spatial bounding boxes, and parent-child relationships. *Budget 40% of Layer 1 engineering time here* — this is the universal engineering bottleneck identified in debate.

2. **Semantic tree alignment.** Not DOM diffing — semantic alignment where identity is (role, label, position, relationships). Three-pass algorithm: (a) exact match on (role, name, description), (b) fuzzy match via weighted bipartite matching on remaining nodes, (c) classify unmatched as additions/removals. Adapted from RTED with domain-specific edit costs.

3. **Task-flow specification.** Tasks defined as annotated paths through the UI: YAML DSL for manual specification, recording-based extraction for common patterns, automatic inference for standard flows (form completion, navigation, search). Task paths are sequences of (state, action) pairs over the semantic tree.

4. **Additive cost model with interval arithmetic.** Each task step receives a cost interval $C(\text{step}) = [c(\theta_{\text{low}}), c(\theta_{\text{high}})]$ using published parameter ranges for Fitts' law ($\text{MT} = a + b \cdot \log_2(1 + D/W)$), Hick–Hyman law ($\text{RT} = a + b \cdot \log_2(n)$), and visual-search models. Composition is additive: $C_{\text{total}} = \bigoplus_i C(\text{step}_i)$ where $[a,b] \oplus [c,d] = [a+c, b+d]$. Regression test: $\inf(C_{\text{after}}) > \sup(C_{\text{before}})$.

5. **Parameter-independent verdict engine.** For dominant failure modes — option proliferation ($n_{\text{after}} > n_{\text{before}}$), target shrinking ($W_{\text{after}} < W_{\text{before}}$), navigation depth increase — the regression verdict reduces to a structural predicate independent of any cost parameter. The interval arithmetic collapses: the *sign* of the cost difference is determined entirely by the structural change.

6. **CI/CD integration.** Zero-config CLI, GitHub Action, GitLab CI template. JSON/SARIF output for integration with existing code-review workflows. Configurable severity thresholds.

**What Layer 1 explicitly cannot do:** Model user behavior (no softmax policies), capture interaction effects between cognitive operations, analyze worst-case users, or provide formal error bounds beyond the interval arithmetic. These limitations motivate Layer 2.

**Deliverable:** A working tool that catches 70–85% of structural usability regressions with zero false positives on parameter-independent verdicts, usable by any team with accessibility-tree-producing UIs.

### Layer 2: Bounded-Rational Theory (Months 3–6)

**What it is.** A bounded-rational MDP layer that models user behavior via softmax policies over free energy, enabling the paired-comparison theorem, cognitive fragility analysis, and cliff detection.

**Components:**

1. **Task-path MDP construction.** Convert Layer 1's task-flow specifications into lightweight MDPs. States encode the current task step plus relevant context (visible elements, focus position, working-memory contents via augmented state). Actions are interactive operations available at each state. Transitions are deterministic for most UI interactions, stochastic for user errors. *Simplified task-path MDPs for most analyses; full state-space MDPs only for fragility analysis.*

2. **Bounded-rational policy computation.** Softmax policies $\pi_\beta(a|s) \propto \exp(\beta \cdot Q(s,a))$ parameterized by cognitive capacity $\beta$. Q-values computed via value iteration on the task-path MDP (tractable because task-path MDPs are small — typically <1,000 states).

3. **Paired-comparison engine.** Given two UI versions and their task-path MDPs, construct the union MDP and compute cost differences under shared analysis. The paired-comparison theorem guarantees that cost-difference errors are $O(\varepsilon)$ rather than the $O(H\beta\varepsilon)$ of independent analysis — the flagship formal result.

4. **Cognitive fragility analyzer.** Compute $F(M) = \max_{\beta \in \mathcal{B}_{\text{human}}} \mathbb{E}_{\pi_\beta}[C(\tau)] - \min_{\beta \in \mathcal{B}_{\text{human}}} \mathbb{E}_{\pi_\beta}[C(\tau)]$ — a model-independent robustness measure comparing the UI to *itself* across the capacity space. This solves the evaluation circularity: "did the UI become more fragile to user variation?" is self-contained and empirically testable, independent of absolute cost calibration.

5. **Cliff detection via analytical pre-computation.** The cliff-location theorem identifies critical $\beta$ values where policies undergo phase transitions, computed analytically from Q-value crossings. Only these critical points (plus neighborhoods) need evaluation, keeping computation within CI/CD budgets. Typical UIs have ≤50 cliff candidates.

6. **Adversarial range-over-β for parameter robustness.** Rather than calibrating β to a single value, report verdicts that hold *across the entire human-plausible β range* $\mathcal{B}_{\text{human}}$. This is the principled answer to the #1 parameter problem: the system reports "regression for all plausible users" vs. "regression for users with β < 2.3" vs. "no regression for any plausible user."

**What Layer 2 adds over Layer 1:** Behavioral modeling (user errors, suboptimal paths), formal error bounds on differential cost estimation, fragility analysis for inclusive design, cliff detection for worst-case users, principled handling of β uncertainty. Layer 1's parameter-independent verdicts remain as fast-path short-circuits; Layer 2 activates only when the parameter-free analysis is inconclusive.

**Deliverable:** A theoretically grounded usability oracle with calibrated confidence thresholds, fragility reports, and inclusive-design diagnostics. Publishable at CHI/UIST.

### Layer 3: Scale (Months 6–9)

**What it is.** Bisimulation-based state-space reduction for production-scale UIs, plus the restated cost algebra for capturing nonlinear cognitive interactions.

**Components:**

1. **Bounded-rational bisimulation.** The cognitive distance metric $d_{\text{cog}}(s_1, s_2) = \sup_{\beta' \leq \beta} d_{\text{TV}}(\pi_{\beta'}(\cdot|s_1), \pi_{\beta'}(\cdot|s_2))$ defines when two states are indistinguishable to a capacity-limited user. Partition-refinement algorithm for computing ε-bisimulation quotients. *This algorithm does not yet exist* — it is a genuine research contribution. Heuristic fallback: feature-based state clustering using Layer 1's semantic similarity, validated against exact bisimulation on small MDPs.

2. **Restated cost algebra with ordinal soundness.** Replace the original time-vs-bits soundness claim with an ordinal bound: the algebra's cost *ordering* is consistent with the mutual-information ordering of the full discrete-event cognitive simulation. Operators: sequential composition ($\oplus$ with load-amplification $\gamma$), parallel interference ($\otimes$ with cross-channel coupling $\alpha$), context modulation ($\Delta_\theta$). The ordinal restatement avoids the dimensional mismatch that invalidated the original theorem.

3. **Compositional analysis for large UIs.** Bisimulation enables analysis of UIs with 2,000+ interactive elements by reducing state spaces from $10^5$–$10^6$ to $\leq 10^4$ abstract states while preserving cost orderings. Combined with the cost algebra's compositional structure, this enables tractable analysis of full enterprise applications.

**What Layer 3 adds:** Scaling to production-size UIs, richer cost models capturing interaction effects (memory-load amplification, cross-channel interference), and the formal guarantee that if a regression exists in the full MDP, it is detected in the abstract MDP.

---

## 4. New Math Required (Load-Bearing Only)

### Theorem 1: Paired-Comparison Tightness

**Statement.** For two UI versions $A$ and $B$ whose task-path MDPs are analyzed under a shared framework (shared state-partition $\varphi$, shared cost model, shared policy family), the regression detection error satisfies:

$$|(\hat{C}_B - \hat{C}_A) - (C_B - C_A)| \leq 2\varepsilon \cdot L_R$$

where $\varepsilon$ is the bisimulation granularity and $L_R$ is the reward Lipschitz constant. For typical parameters ($\varepsilon = 0.005$, $c_{\max} = 2\text{s}$), this yields error $\leq 20\text{ms}$ — well below the threshold of perceptible usability difference.

More precisely: when both UI versions are abstracted under the same partition, systematic abstraction errors are correlated across $A$ and $B$. In the cost difference $\hat{C}_B - \hat{C}_A$, these correlated errors cancel to first order, leaving only a residual proportional to $\varepsilon$ rather than the horizon-amplified $H\beta\varepsilon$ of independent analysis.

**Why it's load-bearing.** Without this theorem, the consistency-oracle claim rests on heuristic faith — "cost differences are probably more accurate than absolute costs." With it, teams can set regression thresholds with formally grounded confidence: any reported regression of magnitude $> 2\varepsilon L_R$ is a true regression. This is the theorem that transforms the system from a heuristic profiler into a formal verification tool.

**Proof difficulty.** Medium. The core argument is error cancellation under shared bias, formalized via coupling arguments on the joint trajectory distribution. The main technical challenge is bounding the correlation structure of abstraction errors across the two MDPs.

**Status.** Proof sketch exists. The $O(\varepsilon)$ scaling is established; the tight constant requires careful analysis of the coupling structure. Qualification: for UIs differing on $k$ transitions, error degrades to $O(k \cdot \varepsilon)$.

---

### Theorem 2: Parameter-Independence for Dominant Failure Modes

**Statement.** For regression types where the structural change is monotone in all cost parameters — specifically, option proliferation ($n_{\text{after}} > n_{\text{before}}$), target shrinking ($W_{\text{after}} < W_{\text{before}}$), and navigation-depth increase ($d_{\text{after}} > d_{\text{before}}$) — the regression verdict is independent of all cost-model parameters. Formally: for any two parameter vectors $\theta_1, \theta_2$ in the admissible range, $\text{sign}(C_{\theta_1}(\text{after}) - C_{\theta_1}(\text{before})) = \text{sign}(C_{\theta_2}(\text{after}) - C_{\theta_2}(\text{before}))$.

Interval arithmetic provides the computational mechanism: $C(\text{step}) = [c(\theta_{\text{low}}), c(\theta_{\text{high}})]$, and regression is flagged when $\inf(C_{\text{after}}) > \sup(C_{\text{before}})$ — a condition that collapses to the structural predicate for these failure modes.

**Why it's load-bearing.** Without this result, every regression verdict requires parameter calibration — the #1 unsolved problem across all approaches. With it, the most common regressions (estimated 50–70% of real cases) receive zero-parameter, zero-false-positive verdicts on Day 1. This is the result that makes Layer 1 immediately useful without any theory.

**Proof difficulty.** Easy. Follows directly from monotonicity of Fitts'/Hick's laws in their structural arguments and the definition of interval comparison.

**Status.** Proven. The monotonicity properties of standard cognitive cost functions are well-established in the psychophysics literature. The contribution is recognizing their implication for parameter-free regression detection.

---

### Theorem 3: Cognitive Fragility Metric and Cliff Location

**Statement (Fragility).** Define cognitive fragility as:

$$F(M) = \max_{\beta \in \mathcal{B}_{\text{human}}} \mathbb{E}_{\pi_\beta}[C(\tau)] - \min_{\beta \in \mathcal{B}_{\text{human}}} \mathbb{E}_{\pi_\beta}[C(\tau)]$$

$F(M)$ is a model-independent robustness measure: it compares a UI to *itself* across the human-plausible capacity space. Fragility regression ($F(M_B) > F(M_A) + \varepsilon_{\text{frag}}$) is strictly weaker than cost regression but independent of absolute calibration and robust to model misspecification.

**Statement (Cliff Location).** For softmax policies $\pi_\beta(a|s) \propto \exp(\beta \cdot Q(s,a))$, cognitive cliffs — discontinuities in optimal policy as a function of $\beta$ — occur at values $\beta^*$ satisfying:

$$\beta^* \cdot (Q(s, a^*) - Q(s, a')) = \ln\left(\frac{\pi_{\beta^*}(a'|s)}{\pi_{\beta^*}(a^*|s)}\right) \to 0$$

i.e., where the softmax probability ratio between the optimal and a suboptimal action crosses a decision-relevant threshold. Cliff severity is proportional to $|C(\tau_{a^*}) - C(\tau_{a'})|$ — the cost difference between the paths enabled by the two actions.

**Why they're load-bearing.** Fragility solves the evaluation circularity: "did the UI become more fragile?" requires no external ground truth. Cliff location makes fragility analysis *computationally feasible* within CI/CD budgets: instead of exhaustive search over $\beta$-space, evaluate only at analytically pre-computed cliff candidates (typically ≤50 per UI). Without cliff location, fragility analysis requires ~200 MDP solves; with it, ~10–50 targeted evaluations suffice.

**Proof difficulty.** Medium for fragility properties (continuity, decomposability); Easy for cliff location (direct analysis of softmax crossing conditions).

**Status.** Cliff location: proven (straightforward softmax analysis). Fragility decomposition: sketch exists; the per-step decomposition $F(M) \approx \sum_t F_t(M)$ with correction $O(\gamma \cdot F_{\max}^2)$ requires bounding interaction effects.

---

### Theorem 4: Ordinal Soundness of Cost Algebra (Restated)

**Statement.** For task graphs with Markov transitions, conditionally independent sensory channels, and bounded interference degree $k$: the compositional cost algebra's ordering over task graphs is consistent with the mutual-information ordering of the full discrete-event cognitive simulation. That is, if $C_{\text{alg}}(G_1) > C_{\text{alg}}(G_2)$, then $\sum_t I^{(1)}(S_t; A_t) > \sum_t I^{(2)}(S_t; A_t)$ up to a tolerance $O(k\varepsilon/\beta)$.

This is weaker than the original time-magnitude claim (which had a dimensional mismatch between time and bits) but correctly serves the regression-detection use case: we need ordinal agreement, not magnitude prediction.

**Why it's load-bearing.** Without ordinal soundness, the cost algebra is an unjustified heuristic — faster than simulation but ungrounded. With it, Layer 3's richer cost model (interaction effects, load amplification) has formal backing: detected regressions under the algebra correspond to genuine increases in cognitive information processing.

**Proof difficulty.** Hard. Requires induction on task-graph structure using the data-processing inequality, with careful handling of the dimensional normalization. The original proof sketch conflated time and bits; the restated ordinal version requires a different proof strategy using order-preserving maps between the time and information domains.

**Status.** Conjectured. Proof sketch for the ordinal version is incomplete. The base case (single operations) is straightforward via rate-distortion theory. The inductive step for sequential composition requires showing that the coupling parameter $\gamma$ preserves orderings, which is plausible but unproven.

---

## 5. What We Explicitly Do NOT Claim

1. **We do not predict absolute task-completion times.** The system is a consistency oracle, not a fidelity oracle. Cost magnitudes are approximations; cost *orderings* between UI versions are the guaranteed output.

2. **We do not detect visual usability regressions.** Changes in CSS, typography, color, spacing, animation, or layout aesthetics are invisible to accessibility-tree analysis. We detect structural regressions (information architecture, interaction flows, element counts, groupings, navigation depth). This is by design: structural regressions are the class most affected by code changes in PRs and least served by existing tools. Screenshot-diff tools (Chromatic, Percy) are the complement.

3. **We do not model all cognitive phenomena.** Emotion, learning, fatigue across sessions, cultural expectations, and aesthetic preference are outside scope. We model the information-processing bottlenecks captured by established psychophysical laws (Fitts', Hick–Hyman, visual search, working memory).

4. **We do not guarantee coverage of all regression types.** The parameter-independence result covers dominant failure modes (estimated 50–70%); remaining regressions require the full bounded-rational analysis (Layer 2) or may fall outside the system's detection boundary entirely. We explicitly characterize this boundary.

5. **We do not claim the bisimulation algorithm exists yet.** Layer 3's bounded-rational partition-refinement algorithm is a research contribution. If it proves intractable, the heuristic fallback (feature-based clustering) preserves practical value while losing formal guarantees.

6. **We do not claim SMT-based repair synthesis is feasible.** This remains a stretch goal, downscoped from the original proposal. The core value is regression detection + bottleneck classification + fragility analysis.

---

## 6. Evaluation Plan

### Retrospective Ordinal Validation (Breaking the Circle)

The system's core claim is that its cost orderings are consistent with human-perceived usability orderings. We validate against existing published human performance data — *no new human studies required:*

1. **CogTool validation datasets.** Human task-completion times for multiple interface variants across desktop and mobile applications. Extract/reconstruct accessibility trees for these UI versions, run the full pipeline, compute rank correlation (Kendall's τ, Spearman's ρ) between model-predicted and human-measured orderings.

2. **Oulasvirta group interaction logs.** Task-completion times and error rates for structured interaction tasks from Aalto University datasets.

3. **Published UI comparison studies.** Findlater & McGrenere 2004, Gajos et al. 2010, Cockburn et al. 2007 — studies reporting human preference or performance orderings between interface versions.

**Threshold:** τ ≥ 0.6 for ordinal agreement. If the system's orderings do not correlate with human orderings on published data, the consistency-oracle claim fails regardless of internal metrics.

### Regression Detection Metrics

- **Benchmark suite:** 200+ UI version pairs from open-source projects (Material UI, Ant Design, React-Bootstrap changelogs) with ground-truth labels from (a) issue-tracker annotations where designers flagged regressions and (b) synthetic mutations with known bottleneck types.
- **Targets:** F1 ≥ 0.80 on curated benchmark. False-positive rate on known-benign refactors ≤ 5%. Zero false positives on parameter-independent verdicts (by construction).
- **Baselines:** axe-core/Pa11y (rule-based), CogTool-style KLM (cognitive but non-compositional), scalar cost threshold without bottleneck decomposition.

### Fragility Validation

- Compute fragility scores on UI pairs where published data shows differential performance across user populations (e.g., novice vs. expert, young vs. elderly, able-bodied vs. motor-impaired).
- Validate that higher fragility correlates with larger performance gaps across user groups.

### Ablation Studies

1. **Layer 1 vs. Layer 2:** Does bounded-rational modeling improve F1 over additive interval arithmetic?
2. **With vs. without fragility:** Does fragility analysis catch regressions that cost analysis misses?
3. **Parameter-independent vs. full analysis:** What fraction of real regressions are parameter-free?
4. **Sensitivity over β range:** Ordinal stability of verdicts across $\beta \in [0.5, 20]$.

### Performance

- Wall-clock time on 4-core laptop CPU: ≤60s for ≤500 elements (Layer 1), ≤300s for ≤2,000 elements (Layer 2).
- Layer 1 alone: target ≤10s for ≤500 elements.

---

## 7. Best-Paper Argument

This synthesis is stronger than any single approach because it resolves the central tension the debate exposed: *practical value requires simplicity, but intellectual impact requires depth.* No single approach achieved both.

**Why not Approach A alone.** The full-theory bisimulation machine has feasibility problems (5+ months to MVP, novel algorithm that doesn't exist yet, monolithic integration complexity) and mathematical baggage (ornamental theorems, dimensional mismatch in cost algebra, TV bound too loose). Its flagship idea — paired-comparison tightness — is preserved and strengthened by embedding it in a system that can actually be built.

**Why not Approach B alone.** The lean profiler ships fast but caps out at Difficulty 5 and Potential 6. It has no behavioral model, no formal error bounds, and no story beyond "careful engineering." The parameter-independence insight is preserved as Layer 1, but the synthesis gives it a theoretical foundation (why it works) and a growth path (what to do when it's insufficient).

**Why not Approach C alone.** The adversarial fuzzer has the strongest best-paper potential (novel framing, hot-topic connections) but inherits Approach A's MDP infrastructure without proposing how to build it. The cognitive fragility metric and cliff-location theorem are preserved and placed in context: fragility is the *second* signal after cost regression, activated when the simpler analysis is inconclusive or when inclusive-design questions arise.

**The synthesis's best-paper argument proceeds in three moves:**

1. **The parameter-independence result** (from B): surprises the audience by showing that the most common usability regressions require *zero calibration* — the verdict is structural. This is the "you don't need theory for 70% of cases" move that captures the practical contingent.

2. **The paired-comparison theorem** (from A): surprises the theoretical contingent by showing that differential estimation under shared analysis is an order of magnitude tighter than absolute estimation. Loose TV bounds become tight for the regression-detection task — *the framing has mathematical consequences.*

3. **The fragility metric + cliff location** (from C): surprises the inclusive-design contingent by showing that usability-as-robustness is a well-defined, computable, model-independent signal that reveals problems invisible to mean-estimation approaches. Cognitive cliffs are vivid, actionable, and novel.

Each move stands alone; together they establish a new research direction: *usability regression testing as incremental formal verification, from parameter-free structural checks to bounded-rational behavioral analysis to inclusive-design robustness.*

**Target venue:** UIST (primary) or CHI (secondary). UIST rewards working systems with surprising theoretical depth. CHI rewards novel framings with human-centered impact. The synthesis serves both.

---

## 8. Hardest Technical Challenge

**The accessibility-tree-to-MDP reduction.**

This is where prior automated usability tools have died. Real accessibility trees are messy, inconsistent across platforms and browsers, rife with implicit state (collapsed menus, modals, scroll containers, dynamic content loading), and lacking information that affects usability (viewport-relative positioning, animation timing, scroll position).

**Why it's the hardest challenge, not bisimulation.**

The debate revealed that bisimulation is a *scaling* problem with a heuristic fallback (feature-based clustering). The MDP reduction is a *correctness* problem with no fallback: if the MDP doesn't accurately represent the user's task-state space, all downstream analysis — Layer 1, 2, and 3 — produces garbage.

**Mitigation strategy:**

1. **Layer 1 sidesteps the full MDP.** Task-path specifications + additive cost over accessibility-tree diffs require no MDP construction. This delivers value immediately while the MDP parser matures.

2. **Restricted UI grammar.** Define a formal grammar covering ~80% of common interaction patterns (linear forms, menu navigation, tabbed interfaces, modal dialogs, search-and-select). Each grammar production maps to a known MDP fragment. UIs matching the grammar get automatic MDP construction; others require task-path annotation.

3. **Incremental parser with MDP debugger.** Build the parser incrementally, starting with the restricted grammar and extending. Include a visualization/debugging tool that shows the constructed MDP alongside the source UI, enabling rapid identification of parsing errors.

4. **Property-based testing.** Generate random accessibility trees conforming to the restricted grammar, construct MDPs, and verify structural invariants (reachability, determinism of UI transitions, consistency with task specifications).

5. **Cross-browser normalization layer.** Budget 3–4 weeks for normalizing accessibility-tree differences across Chrome, Firefox, Safari, and platform-native APIs. This is unglamorous but essential.

---

## 9. Scores

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| **Value** | **9** | Layer 1 delivers immediate CI/CD value (zero-config, parameter-free verdicts for dominant regressions). Layer 2 adds calibrated confidence and fragility analysis for regulated industries. No existing tool provides any of this. |
| **Difficulty** | **7** | Layer 1 is careful engineering (Difficulty 4–5). Layer 2 requires novel theoretical work (paired-comparison proof, fragility computation — Difficulty 7–8). Layer 3 requires a novel algorithm (bisimulation — Difficulty 9). Blended: 7, with genuine research contributions concentrated in Layers 2–3. |
| **Potential** | **8** | Three surprising results (parameter-independence, paired-comparison tightness, cognitive fragility) form a coherent narrative. Not one big theorem but a *framework* with multiple "aha" moments. Strong UIST/CHI candidate. Deducted from 9 because the cost-algebra ordinal soundness is still conjectured. |
| **Feasibility** | **8** | Layer 1 in 6–8 weeks (known algorithms, interval arithmetic). Layer 2 in months 3–6 (task-path MDPs are small; proofs are medium difficulty). Layer 3 is the risk — bisimulation algorithm may not materialize — but Layers 1–2 deliver full value without it. The incremental architecture is the key feasibility enabler. |

**Composite: 32/40** — exceeds any single approach (A: 30, B: 29, C: 31) with better risk distribution.

---

## 10. Risk Register

### Risk 1: Retrospective Validation Fails (τ < 0.6)

**Likelihood:** Medium (30%). Published datasets may not include UI pairs with accessibility-tree-level structural differences, or the cost model's orderings may not align with human judgments on subtle regressions.

**Impact:** Critical. The consistency-oracle claim fails.

**Mitigation:** (a) Begin validation in Week 2, before investing in Layer 2. (b) If τ < 0.6 on raw additive costs, test whether interval-arithmetic verdicts (parameter-independent subset) achieve higher agreement — these are structurally determined and more likely to correlate. (c) If validation fails entirely, pivot to "structural change detector with calibrated heuristics" — weaker claim but still useful.

### Risk 2: Accessibility-Tree Quality Insufficient

**Likelihood:** Medium (25%). Cross-browser inconsistencies, missing ARIA annotations, and dynamic content may produce trees too noisy for reliable semantic alignment.

**Impact:** High. Layers 1–3 all depend on tree quality.

**Mitigation:** (a) Normalize aggressively in the parsing layer. (b) Scope initial support to well-annotated design systems (Material UI, Ant Design) where tree quality is high. (c) Provide a "tree quality score" so teams know when to trust the oracle's verdicts. (d) WCAG mandates are improving tree quality industry-wide.

### Risk 3: Bisimulation Algorithm Intractable (Layer 3)

**Likelihood:** Medium-High (40%). No existing algorithm computes bounded-rational bisimulation; the novel partition-refinement approach may not beat $O(|S|^2)$, violating laptop-CPU constraints for large UIs.

**Impact:** Medium. Layer 3 fails, but Layers 1–2 deliver full value for UIs ≤500 elements.

**Mitigation:** (a) Heuristic fallback: feature-based state clustering validated against exact bisimulation on small MDPs. (b) Approximate bisimulation via locality-sensitive hashing on state features. (c) Accept that Layer 3 is a research contribution with uncertain outcome; Layers 1–2 are the product.

### Risk 4: CI/CD Wall-Clock Budget Exceeded for Fragility Analysis

**Likelihood:** Low-Medium (20%). Cliff detection + targeted evaluation may exceed 60s for complex UIs with many cliff candidates.

**Impact:** Medium. Fragility analysis moves to nightly builds; per-PR analysis falls back to Layer 1 only.

**Mitigation:** (a) Cliff-location theorem reduces evaluation to ~50 targeted points rather than 200 MDP solves. (b) Hierarchical evaluation: cheap Monte Carlo for exploration, expensive analysis only at identified worst-case points. (c) Per-PR gate uses Layer 1 (fast); Layer 2 fragility runs as nightly/weekly analysis — still valuable, analogous to nightly performance-regression suites.

### Risk 5: Ordinal Cost-Algebra Soundness Proof Fails

**Likelihood:** Medium (35%). The dimensional normalization required for the ordinal restatement may not preserve the inductive structure of the original proof sketch.

**Impact:** Low-Medium. Layer 3's cost algebra becomes an empirically validated heuristic rather than a formally grounded tool. Layers 1–2 are unaffected.

**Mitigation:** (a) Attempt the proof early (Month 3) to determine feasibility. (b) If the full ordinal soundness fails, prove a weaker monotonicity property: $C_{\text{alg}}(\text{after}) > C_{\text{alg}}(\text{before}) \implies$ the full simulation also shows increase, for *additive* task graphs (a useful subcase). (c) Validate empirically that the algebra's orderings agree with simulation orderings on the benchmark suite, even without formal proof.

---

*This document synthesizes Approaches A, B, and C from the ideation phase, refined through adversarial debate and depth assessment. It represents the team's consensus on the strongest path forward for the Bounded-Rational Usability Oracle project.*
