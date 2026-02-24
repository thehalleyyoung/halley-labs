# Community Expert Evaluation: Proposal proposal_00

## Finite-Width Phase Diagrams for Neural Network Training Dynamics via Lyapunov Stability of the NTK Trajectory

**Evaluation Date:** 2026-02-22
**Evaluator Role:** ML Research Community Expert
**Methodology:** Agent Teams — Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer, with adversarial cross-challenge and independent verification signoff.

---

## Summary

This proposal builds the first integrated system that computes finite-width phase diagrams predicting the lazy-to-rich transition boundary for neural network training, using a 1/N perturbative expansion of the NTK with novel H-tensor derivations for convolutional and residual architectures. The system uses a Lyapunov stability analysis of the NTK training trajectory to define phase boundaries, with KADR (Kernel Alignment Drift Rate) as the operational order parameter. Three tiers of results are proposed: working tool (70%), analytic H-tensors (35%), architecture-dependent critical exponents (10%).

---

## Pillar Scores

### 1) Extreme Value: 5/10

**Who needs this:** The primary audience is theoretical ML researchers studying NTK/mean-field/µP training dynamics — perhaps 50–500 active researchers worldwide. The H-tensor derivations for ConvNets and ResNets are genuinely novel mathematical objects with no prior derivation, giving this community a computational laboratory for testing conjectures. The KADR diagnostic and kernel evolution ODE solver have standalone reuse value.

**Who doesn't need this:** Practitioners rarely think in terms of lazy-vs-rich regimes. The "sweep triage" narrative (eliminating 30–50% of hyperparameter grid) is speculative and unvalidated. Transformer exclusion removes the architectures ~80% of the community currently cares about. The ML safety use case (phase boundary proximity as instability indicator) is a hypothesis without supporting evidence.

**Cross-challenge resolution:** The Auditor scored 4/10 (small audience); the Synthesizer argued the gap between Neural Tangents (infinite-width only) and Tensor Programs (scaling laws, no phase diagrams) is genuine. The resolved view: within its actual audience the value is 7/10, but to the broader ML community it's 3/10. KADR requires full NTK computation (O(nP) per evaluation), making it a theorist's diagnostic, not a practitioner's tool.

### 2) Genuine Software Difficulty: 6/10

**Realistic scope:** The claimed ~65K LoC is inflated. The Architecture IR is over-engineered for 3 architecture families (~3–5K realistic vs. 15K claimed). The Symbolic Kernel Engine overlaps with Neural Tangents. Realistic estimate: **35–45K LoC**, with ~20–25K genuinely novel code (H-tensor computation, moment closure, ODE integration, UQ indicators, Lyapunov exponent computation).

**What's genuinely hard:** No library computes H-tensors for structured architectures. No library does perturbative finite-width kernel corrections. The Lyapunov stability analysis of non-autonomous kernel trajectories is novel numerical machinery. The coupling between kernel evolution ODE and bifurcation detection with pseudo-arclength continuation is real integration work.

**What's not hard:** The infinite-width NTK engine, architecture parsing, evaluation harness, and Nyström approximation are standard. ~40% of the codebase leverages existing tools (JAX autodiff, SciPy ODE solvers, Neural Tangents).

### 3) Best-Paper Potential: 5/10

**Genuine contributions:** (1) The Lyapunov trajectory stability reformulation is a real conceptual advance — it identifies that prior steady-state bifurcation formulations produce block-triangular Jacobians with no feedback, and fixes this with a variational system that has genuine two-way coupling. This is the kind of "right formulation" insight that elevates a paper. (2) The H-tensor derivations for ConvNets (spatial averaging) and ResNets (O(1/L) per-block scaling) are novel mathematics. (3) The dual validation strategy (retrodiction of known results + prediction for new architectures) is epistemologically sound.

**Why not higher:** The Lyapunov reformulation, while novel, follows a standard dynamical systems template (apply to new domain). Reviewers may view this as "Dyer & Gur-Ari (2020) + better linearization + engineering." Best-paper candidacy requires the 10%-probability universality class result (architecture-dependent critical exponents), or a connection to edge-of-stability that the proposal mentions but does not develop. Without a surprising experimental finding, this is a solid main-conference paper, not a spotlight.

**Upgrade path identified by verifier:** If the Lyapunov framework is extended to discrete-time dynamics and successfully predicts edge-of-stability onset η*(N, L), this bridges NTK theory and edge-of-stability — two active research fronts developing independently. That connection could push best-paper potential to 7/10.

### 4) Laptop-CPU Feasibility & No-Humans: 6/10

**Prediction side (FEASIBLE):** The kernel-space reformulation is genuinely clever — the system solves ODEs *about* neural networks, never training them. ODE integration on 200×200 matrices: ~0.4 seconds per grid point. Full phase boundary computation: 2–4 hours. This is unambiguously CPU-feasible.

**Validation side (OVER-BUDGET):** Ground-truth training runs are the bottleneck. The proposal claims 12–24 hours; actual arithmetic yields:
- ~4000 training runs × ~15 min each = ~1000 CPU-hours per architecture family
- On 8-core laptop: **5–6 days per architecture family**, not 12–24 hours
- The proposal's own final approach document revised to "60–80 hours (3–4 days)" — internally inconsistent with the problem statement's 12–24 hour claim

**Realistic timeline:** With Strategy A (MLP-only, reduced seeds: 10 at boundary, 3 interior): ~2–3 days total. For all 3 architecture families with full validation: 2–3 weeks. No GPUs needed. No human annotation. Fully automated.

### 5) Overall Feasibility: 6/10

**Tier 1 (~70%): Empirically calibrated phase diagram tool.** Credible. Multi-width NTK regression for Θ^(1) extraction is straightforward numerical linear algebra. ODE integration is standard. The main risk is that n_probe=200 subsampling may not capture spectral structure relevant to phase boundaries.

**Tier 2 (~35% → revised ~40–50%): Analytic H-tensors.** The existence of a 124KB theory manuscript with full proofs substantially de-risks this. The ConvNet H-tensor relies on a spatial averaging hypothesis that may fail for non-uniform spatial data. The ResNet composition rule is more likely to hold.

**Tier 3 (~10%): Architecture-dependent critical exponents.** Honest assessment. With only 5 width values for log-log regression, statistical power is low. Systematic errors from the perturbative expansion compound across widths.

**Key approximation risks:**
- **Frozen-H:** H(t) ≈ H(0) is a lazy-regime approximation used to locate the lazy regime's boundary. Logically consistent (boundary = last point where approximation holds) but means all predictions inside the rich regime are unreliable by construction. The structural stability lemma's hypothesis is essentially the conclusion — circular at the boundary.
- **Gaussian moment closure:** κ₄ = 0 is exact for Gaussian init but degrades during training and for ReLU. The proposal correctly restricts theorems to C³ activations (GELU, Softplus, Swish) and handles ReLU as an empirical conjecture. The compound bias of frozen-H + Gaussian closure near the phase boundary is unquantified.
- **Grönwall bound:** The theoretical convergence guarantee (Theorem 3.11) has exponent ~10^5 for practical parameters — vacuous. All validity assessment relies on computable indicators without theoretical backing.

---

## 6) Fatal Flaws

**No fatal flaws identified.** One SERIOUS issue:

- **Compound approximation error near phase boundary.** Both frozen-H and Gaussian moment closure degrade toward the rich regime, and their joint effect is unquantified. The UQ system tracks each independently but doesn't model their correlation. The system is least reliable precisely where predictions matter most. Mitigated by the ablation study (frozen-H vs. periodically recomputed H) and moment-closure sensitivity analysis (±50% κ₄ perturbation).

**Former "fatal" flaw resolved:** The Skeptic initially flagged the ReLU/GELU contradiction as fatal (theorems exclude ReLU but experiments target it). Cross-challenge resolved this: the paper explicitly excludes ReLU from theorems and handles it as Conjecture 3.1 tested empirically. This is a stated scope restriction, not a hidden contradiction.

**Baseline risk (SERIOUS):** The one-parameter heuristic (predict rich if γ = η·N^{-(1-a-b)} > C) captures the correct scaling from µP theory and may achieve AUC > 0.80 on simple architectures/datasets. If this baseline reaches AUC > 0.85, the marginal value of the full 35–45K-line system is questionable. The proposal includes this as Baseline B with a 15-percentage-point outperformance target, which is the right experimental design but an aggressive target.

---

## VERDICT: CONTINUE

### Binding Pre-Conditions

1. **Baseline B pre-flight.** Before full evaluation, compute the γ > C heuristic AUC on a 50-point MLP pilot grid. If Baseline B AUC > 0.85, pivot emphasis from "prediction accuracy" to "theoretical framework + UQ + scaling exponents."

2. **Timeline correction.** Replace all instances of "12–24 hours" with "2–3 days for MLP-only evaluation; 2–4 hours for prediction alone." The current claim is 5–10× underestimated.

3. **Moment-closure validation.** Run κ₄ perturbation ablation on GELU networks first. If phase boundary displacement > 20% under ±50% κ₄ perturbation, revise precision claims downward.

4. **n_probe sensitivity.** Compute phase boundaries at n={500, 1000, 2000} for one MLP configuration. If boundaries shift >15% between n=1000 and n=2000, add as stated limitation.

### Kill Criteria

| Criterion | Test | Outcome if triggered |
|-----------|------|---------------------|
| γ > C heuristic AUC > 0.85 | Baseline B on pilot grid | Pivot emphasis; marginal value suspect |
| H_{ijk} factorization fails + empirical can't beat heuristic | Conv H-tensor verification | Abandon ConvNet claims |
| Frozen-H validation error > 30% | H(t) vs. H(0) comparison | Framework validity compromised |
| ε₂ (moment-closure) > 0.3 for GELU at depth ≥ 3 | κ₄ quality indicators | Gaussian closure unreliable |

### What This Project Is, Honestly

This is a **well-designed theoretical contribution with computational demonstration** for the NTK/mean-field theory community. Its genuine strengths are: novel H-tensor derivations, the Lyapunov stability reformulation (fixing a real deficiency in prior formulations), falsifiable predictions, and honest UQ. Its value is NOT: a practitioner tool, a best-paper contender (without the 10% universality result or edge-of-stability connection), or a system that runs in 12–24 hours.

**Recommended strategy:** Pursue MLP-only first (Strategy A), establish framework and publication, extend to ConvNets as follow-up. Explore edge-of-stability connection via discrete-time Lyapunov exponents as best-paper upgrade path.

### Upgrade Paths to 7–8/10

1. **Edge-of-stability connection.** Extend Lyapunov framework to discrete dynamics, predict η_EoS(N, L). Bridges NTK theory and edge-of-stability — two independently active research communities. Gate: test one config for clean zero-crossing under discrete-time dynamics.

2. **Clean finite-size scaling exponents.** Measure β in Δη/η* ~ N^{-β} across architectures. If β differs across MLP/ConvNet/ResNet, this connects neural network training to statistical-mechanics universality classes. Gate: R² > 0.9 on 5-width power law fit.

---

## Score Summary

| Dimension | Score | Evidence |
|-----------|-------|----------|
| Extreme Value | 5/10 | Genuine gap for theorists; practitioners don't need this; no transformers |
| Software Difficulty | 6/10 | 35–45K realistic LoC; ~20–25K genuinely novel; H-tensor + Lyapunov machinery is real |
| Best-Paper Potential | 5/10 | Novel reformulation + H-tensors = solid paper; best paper needs universality or EoS connection |
| Laptop-CPU & No-Humans | 6/10 | Prediction: hours. Validation: days. Fully automated, no GPUs, no humans |
| Overall Feasibility | 6/10 | Tier 1 credible at ~70%; complete theory manuscript de-risks math; compound errors near boundary unquantified |

**Weighted Average: 5.6/10 → CONTINUE with conditions**

---

*Evaluation produced by Agent Teams verification workflow: Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer, cross-challenge synthesis, and independent verification signoff. All binding conditions and kill criteria were unanimously endorsed.*
