# MARACE Approach Debate

This document synthesizes the critiques from the Math Depth Assessor, Difficulty Assessor, and Adversarial Skeptic on the three proposed approaches, followed by direct teammate-to-teammate challenges.

---

## Approach 1 — Sound Static Certification via HB-Constrained Zonotope Abstract Interpretation

### Math Depth Assessment

- **Math load-bearing?** Partially. The convergence proof and compositional soundness theorem are genuinely load-bearing—the approach cannot function without them. However, the PSPACE-hardness result and the complexity-theoretic separation are ornamental motivation: they frame the problem as important but do not contribute to the solution machinery.
- **Convergence proof gaps:**
  - "Stratified widening" is invoked but never formally defined. The proof sketch claims the widening operator produces chains of bounded length, but the tightening step (constraint-directed recovery of precision) makes the operator *non-monotone*, violating the foundational Cousot-Cousot requirement for abstract interpretation convergence.
  - Fourier-Motzkin elimination on pre-activation variables can *double* the constraint count at each ReLU layer, meaning |C| grows unboundedly through the network—contradicting the convergence bound's assumption that |C| is a fixed parameter.
- **Compositional soundness contradiction:** The proof assumes a static HB graph (fixed interaction-group decomposition), but the approach document explicitly states that interaction groups merge adaptively during analysis. These cannot both be true—adaptive merging invalidates the fixed-graph assumption on which the assume-guarantee proof relies.
- **Schedule-space complexity:** The O(N^{2N}) bound on topologically distinct orderings is vacuously worse than brute-force enumeration of all N! permutations for N ≥ 3. This bound adds no insight.
- **Scores:** Math difficulty 7/10, Math necessity 8/10.

### Difficulty Assessment

- **Implementation difficulty:** 9/10. **Feasibility: 3/10** (not the Visionary's claimed 5).
- The widening cascade problem—where widening one agent's abstract state forces widening all causally connected agents' dimensions—is *structural*, not a tunable parameter. No amount of engineering finesse patches a fundamentally non-convergent operator.
- The Rust numerics ecosystem is immature for this workload. The proposed 110K LoC estimate is fantasy; a realistic 6-month deliverable is a **2-agent ReLU-only toy prototype** and nothing more.

### Adversarial Skeptic Critique

- **HB constraints are NOT linear for order-book matching.** The limit order book uses priority queue semantics (price-time priority), which involve sorting and conditional branching. These operations cannot be faithfully represented as linear constraints over the joint state space. The entire zonotope-based approach rests on this linearity assumption.
- **Interaction groups are NOT small for exchange-mediated finance.** When all agents interact through the same order book, the interaction graph is a complete graph (k = N). The compositional decomposition into small groups—the key to tractability—does not apply.
- **ReLU convergence excludes real architectures.** The convergence proof is specific to ReLU networks. Real trading strategies use LSTMs and transformers, which involve sigmoid/tanh activations, attention mechanisms, and recurrence. The proof does not generalize.
- **"ThreadSanitizer" analogy is misleading.** ThreadSanitizer achieves near-zero false positive rates in production. This approach targets < 10% FPR on toy benchmarks. The comparison sets false expectations.
- **IP impossibility.** No trading firm will share ONNX checkpoints of proprietary strategies. The input assumption is unrealistic for the stated use case.

### Salvageable Elements (per Skeptic)

- The interaction race formalism and separation theorem could stand alone as a theory paper.
- The ε-race calibration idea (parameterizing how "close" a system is to a race) has independent value.

---

## Approach 2 — Scalable Race Hunting via Differentiable Schedule Optimization

### Math Depth Assessment

- **Math load-bearing?** Weakly. The math provides post-hoc guarantees on an essentially heuristic search procedure, rather than driving the approach's core logic.
- **Rounding guarantee issues:** The bound Δ(τ) ≤ L · N · τ · log(N) requires computing the Lipschitz constant L of the composed policy-environment system, which is NP-hard to compute tightly for deep networks. In practice, loose L estimates make the bound vacuous.
- **PAC coverage bound issues:** The coupon-collector-style argument assumes each optimization run discovers a race basin with probability ≥ p_min drawn uniformly. Gradient-based search is not uniform—it is biased toward basins with large gradients, meaning small-but-dangerous basins can be systematically missed. The bound doesn't account for this.
- **IS probability estimation issues:** The importance-weighted estimator assumes race basins are convex subsets of the relaxed schedule space. ReLU policy boundaries create non-convex basin geometries, invalidating the N_eff guarantee.
- **No novel math.** All techniques are off-the-shelf: Gumbel-Softmax (Jang et al. 2017), Sinkhorn normalization, coupon-collector bounds, importance sampling. The combination is workmanlike but not mathematically novel.
- **Scores:** Math difficulty 3/10, Math necessity 4/10.

### Difficulty Assessment

- **Implementation difficulty:** 6/10. **Feasibility: 6/10** (not the Visionary's claimed 8).
- The core PyTorch pipeline (Gumbel-Softmax relaxation, gradient-based schedule search, rounding) is buildable in 2–3 months by an experienced ML engineer.
- The **surrogate order-book model** is the make-or-break component. If the differentiable surrogate doesn't faithfully capture matching engine semantics, found races won't transfer to the real simulator. This is a research risk, not an engineering task.
- The claimed "50+ agents in minutes" requires GPU compute. On a laptop CPU—the stated development constraint—realistic throughput is 6–10 agents with slow iteration.
- **Realistic 6-month deliverable:** A 6–10 agent race finder that works but is slow on CPU.

### Adversarial Skeptic Critique

- **Rounding bound is vacuous at scale.** At N = 50, with modest Lipschitz constant L ≈ 1 and temperature τ = 0.1, the bound Δ(τ) ≤ L · N · τ · log(N) ≈ 1 · 50 · 0.1 · log(50) ≈ 19.6. For any safety predicate with violation margin < 20, the rounding guarantee says nothing. Even with aggressive annealing (τ = 0.01), the bound is still ≈ 1.96—often larger than meaningful safety margins.
- **Soft-attention surrogate can't represent exact queue position.** Real HFT races depend on an agent's exact position in the order queue (price-time priority). A soft-attention mechanism over price levels cannot represent this discrete, position-dependent logic. Races found in the surrogate may not correspond to real matching-engine behavior.
- **Competing methods exist.** The approach presents as novel but overlaps significantly with existing falsification tools: S-TaLiRo, Breach (for STL falsification), CMA-ES (derivative-free schedule search), and Bayesian adversarial testing. The paper must position against these explicitly.
- **$50M/year claim is fabricated.** The "opportunity cost per firm" figure has no citation or derivation. It undermines credibility.

### Salvageable Elements (per Skeptic)

- Gumbel-Softmax parameterization of the schedule space is a genuinely useful heuristic, even without the formal rounding guarantee.
- Importance-weighted race probability estimation is practical and could work as a standalone contribution paired with any race-finding method.

---

## Approach 3 — Causal Race Discovery via Interventional Schedule Analysis

### Math Depth Assessment

- **Math load-bearing?** Yes—the Race-Causality Correspondence theorem IS the approach. Without it, there is no reason to use causal inference machinery for race detection.
- **FATAL flaw in the correspondence theorem:** The theorem requires the *faithfulness assumption* for the SCM. Faithfulness states that every conditional independence in the data corresponds to a d-separation in the causal graph (no "accidental" independences). But neural network policies are *deterministic functions* of their inputs—they create exact functional dependencies that systematically violate faithfulness. A deterministic policy π(o) produces actions that are perfectly determined by observations, creating non-d-separation-based independences throughout the causal graph. The theorem's central assumption fails for the exact class of systems the approach targets.
- **ACE is the wrong estimand.** The Average Causal Effect averages over environment stochasticity (random seeds, noise). This means rare-but-catastrophic races (the ones that actually matter—flash crashes, Knight Capital events) are averaged away. A race that occurs under 0.1% of environment conditions but causes $440M in losses has ACE ≈ 0 and would not be detected.
- **Partial identification bounds likely trivially loose.** Without strong parametric assumptions, Manski-style bounds on P(φ | do(σ)) default to the interval [0, 1] for binary safety outcomes. The proposed tightening via "monotone treatment response" and "monotone instrumental variables" requires domain-specific assumptions that may not hold for adversarial agent interactions.
- **Scores:** Math difficulty 6/10, Math necessity 9/10.

### Difficulty Assessment

- **Implementation difficulty:** 8/10. **Feasibility: 4/10** (not the Visionary's claimed 6).
- The neural counterfactual predictor (diffusion model over order-book trajectories conditioned on schedule) is alone a 2-month research project with uncertain outcomes.
- Conditional independence testing at scale requires 10^8+ samples for reliable results in high-dimensional financial state spaces. Generating these samples takes days to weeks on a laptop.
- **Realistic 6-month deliverable:** A 4-agent, fully-observable causal analysis pipeline without the diffusion model and without partial identification bounds. Essentially a proof-of-concept on toy environments.

### Adversarial Skeptic Critique

- **Faithfulness violated by hedging strategies.** Financial strategies are explicitly designed to create statistical independence between causally coupled positions (that's what hedging *is*). A portfolio hedged against market risk will show A ⊥⊥ B in observational data even when A causally affects B through the hedge instrument. The CI testing framework will systematically miss these edges.
- **CI testing with BH correction is unsound.** The Benjamini-Hochberg procedure assumes independence (or positive regression dependency) among the test statistics. Adjacent CI tests in the causal race graph share conditioning variables, creating strong positive correlations that inflate the actual false discovery rate beyond the nominal level.
- **Neural counterfactual predictor: identification error.** The predictor is trained on *observational* execution data but must predict *interventional* distributions (outcomes under do(σ')). Without explicit causal adjustment, the predictor learns P(Y | σ) ≠ P(Y | do(σ)), and the causal effect estimates are biased.
- **"First bridge" claim is false.** Halpern and Pearl's actual causation framework has already been applied to verification and accountability in concurrent systems. The claimed novelty of bridging causal inference and verification is overstated.
- **ACE ≠ 0 is trivially satisfied.** For almost any pair of interacting agents, the average causal effect of schedule ordering on outcomes is nonzero (because ordering generally matters at least slightly). Without a principled threshold, every agent pair is flagged as a "race," rendering the detector useless.

### Salvageable Elements (per Skeptic)

- The *interventional race definition* (defining races via do-calculus rather than reachability) is a genuine conceptual contribution worth preserving.
- Fix synthesis via min-cut on the causal race graph is elegant and practical—*if* the graph is correctly constructed.
- The partial identification framework is the right theoretical tool for handling unobserved market participants, even if current bounds are too loose.

---

## Cross-Cutting Concerns

### Financial Domain as Trojan Horse (Skeptic)

The financial domain is presented as the motivating application, but it may be an impossible deployment target:
- **No firm will share ONNX checkpoints** of proprietary trading strategies. Intellectual property concerns make the core input assumption unrealistic.
- **ABIDES** (the proposed simulation environment) is a cartoon approximation of real exchange matching semantics. Results on ABIDES may not transfer to production exchanges.
- All three approaches assume access to agent internals that real-world deployments will not provide.

### Hybrid Strategy Critique (Skeptic)

The Visionary's recommended hybrid strategy ("build Approach 2 first, add Approach 3 as explanation layer, pursue Approach 1 long-term") is characterized as a dodge. The three approaches have *incompatible value propositions*:
- Approach 1 promises soundness but can't scale.
- Approach 2 promises scale but can't provide guarantees.
- Approach 3 promises explanation but can't reliably identify races.

Combining them doesn't resolve these tensions—it accumulates their weaknesses.

### PSPACE-Hardness (Skeptic)

The PSPACE-hardness of interaction race detection is presented as motivation ("this problem is hard, so novel solutions are needed"), but it actually *undermines* all three approaches: it means no polynomial-time algorithm can solve the general problem, so each approach must either (a) solve a restricted subproblem (limiting applicability) or (b) give up completeness/soundness (limiting guarantees). The hardness result is a ceiling, not a selling point.

---

## Direct Teammate-to-Teammate Challenges

### 1. Math Assessor → Visionary

> "PSPACE-hardness is not a pillar of this work. It's a known-hard-to-known-hard mapping—you're reducing an already-known-to-be-hard concurrency problem to another known-hard problem. This adds no technical insight and should not inflate the Potential score. Feasibility for Approach 1 should be 3–4, not 5. The convergence proof has fundamental gaps (non-monotone operator, unbounded constraint growth) that cannot be patched without rethinking the entire widening strategy."

### 2. Difficulty Assessor → Visionary

> "Your feasibility scores are inflated by 2–3 points across all three approaches. Approach 1 at feasibility 5 means 'coin flip whether it works'—the reality is closer to 3 ('probably won't work in 6 months'). The 110K LoC estimate is fantasy; realistic Rust development velocity for research-grade numerical code is 50–100 LoC/day, putting 110K LoC at 3–6 engineer-years. Approach 2's feasibility 8 ignores the surrogate model research risk. Approach 3's feasibility 6 ignores that the neural counterfactual predictor alone is a paper-worthy contribution that might not work."

### 3. Skeptic → Visionary

> "The financial domain is an IP impossibility that undermines the entire value proposition. No exchange operator or trading firm will provide the inputs your system requires. The ThreadSanitizer analogy is misleading—ThreadSanitizer works because it instruments *your own code* in *your own process*; MARACE requires instrumenting *other firms' strategies* in a *shared exchange*. The hybrid strategy is a dodge: you're proposing to build three systems instead of one, each with fundamental limitations that the combination doesn't resolve."

### 4. Skeptic → Math Assessor

> "You're too generous about Approach 1's convergence proof. You correctly identify the non-monotone operator problem but still rate math difficulty at 7/10. The deeper issue: adaptive decomposition of interaction groups (which the approach document explicitly requires for tractability) invalidates the fixed-constraint convergence argument entirely. You can't prove convergence over a graph that's being rewritten during iteration. This isn't a gap to be filled—it's a contradiction to be resolved, and resolving it likely requires abandoning either adaptivity or the current convergence strategy."

### 5. Math Assessor → Skeptic

> "The causal bridge concept in Approach 3 is genuinely novel even if the current execution is flawed. Don't throw out the baby with the bathwater. Yes, faithfulness fails for deterministic policies. Yes, ACE is the wrong estimand for rare events. But the *idea*—that races are causal effects of schedule interventions, identifiable via do-calculus—is a conceptual contribution that survives these specific technical failures. Fix the estimand (use distributional causal effects or worst-case interventional bounds instead of ACE), weaken the faithfulness requirement (use partial identification or sensitivity analysis), and the framework stands. Dismissing the entire approach because of fixable technical choices wastes the strongest conceptual innovation across all three proposals."

---

## Summary of Assessor Recommendations

| Assessor | Top Pick | Rationale |
|----------|----------|-----------|
| **Math Assessor** | Approach 1 (with Approach 2 fallback) | Highest math necessity; borrow Approach 3's causal explanation as post-hoc layer |
| **Difficulty Assessor** | Approach 2 | Only approach with a realistic 6-month deliverable beyond toy scale |
| **Adversarial Skeptic** | None as stated; salvageable elements from all three | Every approach has fatal flaws in its current form; recommends extracting standalone contributions |

### Adjusted Feasibility Scores

| Approach | Visionary's Score | Math Assessor | Difficulty Assessor | Skeptic |
|----------|-------------------|---------------|---------------------|---------|
| 1 — HB-Constrained Zonotopes | 5 | 3–4 | 3 | 2–3 |
| 2 — Differentiable Schedule Optimization | 8 | 6 | 6 | 5 |
| 3 — Causal Race Discovery | 6 | 4–5 | 4 | 3 |
