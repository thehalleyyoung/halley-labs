# Final Adversarial Synthesis: Proposal 00 — Finite-Width Phase Diagrams

## Cross-Challenge Resolutions

### A. Value Score: Auditor (5) vs Synthesizer (7)

**Resolution: 6 (original), 7 (restructured)**

The Auditor is right that transformer exclusion severely limits the practitioner audience. The proposal itself acknowledges this ("honest about audience"). The 200–500 researcher estimate is plausible for the *core* theory audience. However, the Auditor underweights the H-tensor derivation as a standalone mathematical contribution — Reviewer 3 correctly identifies that ConvNet/ResNet H_{ijk} tensors are genuinely novel mathematics with publication value independent of the tool. The restructured version, scoped to MLP-only base with ConvNet H-tensor as the highlight contribution, earns its 7: it's a tight theory paper with a working artifact, not a bloated tool with a shrinking audience. The original's value stays at 6 because the 50K LoC ambition dilutes the contribution narrative.

**Winner: Synthesizer on restructured, Auditor on original. Split score.**

### B. Feasibility / CPU Budget: Auditor (4) vs Skeptic (~3) vs Synthesizer (7)

**Resolution: Original=4, Restructured=6**

The Skeptic's arithmetic is the most rigorous. Let's verify:

- Phase diagram: ~500 grid points × 5 calibration widths × NTK computation at each = non-trivial
- Ground truth: Even at 5 seeds × 500 grid points = 2500 training runs. Each run: width ≤ 1024, depth ≤ 4, 2K data, ~100 epochs. Estimate: ~3–5 min each on CPU = **125–210 hours**
- Boundary-adjacent 10-seed runs add another ~50 hours
- Ablation multiplier: 4 mandatory ablations on a subset still adds ~100 hours
- **Conservative total: 300–400 CPU-hours for MLP-only evaluation**

The proposal's "12–24 hours" is fantasy. The revised "60–80 hours" (from problem.md) is still optimistic by ~4x for the full original scope. However:

- The Skeptic's 1000+ hours estimate assumes the original 65K LoC scope with ConvNets + ResNets + all ablations
- The restructured MLP-only scope genuinely reduces this to ~150–200 CPU-hours (~6–8 days on 8-core)
- This is feasible but requires multi-day overnight runs, which the restructured proposal acknowledges

The Synthesizer's 7 for feasibility on the restructured version is slightly generous given CPU realities. Adjusted to 6.

**Winner: Skeptic on original, compromise on restructured.**

### C. Is Tier 1 Just Baseline E (Width Interpolation)?

**Resolution: No, but the distinction is thinner than the Synthesizer claims.**

The Synthesizer is technically correct: an ODE system with empirically calibrated 1/N coefficients is structurally different from linear width interpolation. The ODE provides:
1. Extrapolation to untested widths via the 1/N expansion structure
2. Temporal dynamics (how the kernel evolves during training, not just the endpoint)
3. Bifurcation detection from eigenvalue analysis of the ODE

Width interpolation (Baseline E) has none of these. However, the Skeptic is correct that *without validated H-tensor derivations*, the empirical calibration degrades toward a sophisticated curve-fitting exercise. The "extrapolation structure" of the ODE is only as good as the 1/N² model assumption, which is itself unvalidated at small N. If the 1/N expansion doesn't converge at N=64–256 (highly plausible near phase boundaries), then the ODE wrapper adds complexity without accuracy.

**Winner: Synthesizer wins the technical argument, but Skeptic wins the practical concern.** Tier 1 is NOT Baseline E in principle, but may perform like Baseline E in the regime that matters most (near boundaries, small N).

### D. KADR Circularity

**Resolution: MODERATE flaw, not fatal, but the Synthesizer's silence is indeed a weakness.**

Both Auditor and Skeptic are right to flag this. The proposal's evaluation plan uses "kernel alignment drift rate" (KADR) both as:
1. The system's *predicted* order parameter (from the kernel ODE)
2. The ground truth's *measured* order parameter (from actual training)

This is not inherently circular — measuring the same quantity via theory vs. experiment is standard physics methodology. The non-circularity holds IF:
- The ground-truth KADR is measured from actual gradient descent trajectories (it is)
- The predicted KADR comes from the ODE model (it does)
- These are genuinely independent computations (they are)

However, there IS a subtle problem: if KADR is a poor proxy for "lazy vs. rich" regime identity (which the Auditor flags as "ad hoc multi-criteria"), then BOTH prediction and ground truth could agree while being wrong about the actual regime. The proposal should include at least one KADR-independent validation metric (e.g., linear probing accuracy gap, or representation similarity index) as a sanity check.

**Winner: Auditor and Skeptic are right to flag it. The Synthesizer's silence is a genuine gap. But it's fixable: add one independent metric.**

### E. Monotonicity of Lyapunov Exponent

**Resolution: HIGH severity flaw, fixable but symptomatic.**

Both Auditor and Skeptic identify that the bisection algorithm for phase boundary detection assumes monotonicity of the leading eigenvalue (Lyapunov exponent) as a function of the coupling parameter γ. This assumption is:
- **Unproven** for the kernel ODE system
- **Likely false** near codimension-2 bifurcations (Hopf + transcritical interaction)
- **Possibly false** even for simple architectures with non-monotonic learning rate dependence

The Synthesizer's fix (replace bisection with full sweep) is correct and practical:
- A grid sweep over γ costs ~10x more than bisection but eliminates the monotonicity assumption
- At 500 grid points this is feasible within the CPU budget
- Adaptive refinement near detected crossings recovers most of bisection's efficiency

However, the deeper concern is whether the eigenvalue-crossing formulation itself is well-posed. If eigenvalues exhibit avoided crossings (generically true for parameter-dependent matrices), the "crossing zero" condition becomes a near-miss that requires careful numerical treatment. This is standard numerical bifurcation theory — solvable, but not trivial.

**Winner: Skeptic on diagnosis, Synthesizer on fix. The monotonicity issue is a HIGH-severity implementation flaw, not a fatal theoretical flaw. Full sweep + adaptive refinement is the right approach.**

---

## Consolidated Fatal Flaws (Ranked)

### CRITICAL

1. **CPU budget underestimated 4–10x (original scope).** The original proposal claims 12–24 hours; realistic estimate is 300–1000+ hours depending on architecture coverage. This makes the original scope infeasible on laptop CPU within any reasonable timeline. *Severity: CRITICAL for original, MODERATE for restructured (MLP-only is ~150–200 hours, feasible with multi-day runs).*

### HIGH

2. **Perturbation theory diverges at phase boundaries.** The 1/N expansion's effective parameter blows up precisely where predictions matter most. The proposal acknowledges this and has mitigations (UQ flagging, bifurcation detection doesn't require accuracy *at* the boundary), but this fundamentally limits the system's value proposition. A phase diagram tool that can't make confident predictions near phase boundaries is like a weather forecast that's only reliable when it's sunny. *The restructured version inherits this limitation but is honest about it.*

3. **Monotonicity assumption in bisection algorithm.** Unproven and likely false. Fixable by replacing with full sweep + adaptive refinement, but must be addressed before any results are credible. *Severity: HIGH (but fixable).*

4. **H_{ijk} factorization is a hypothesis, not a theorem.** The key factorization H^(conv)_{ijk} = H^(dense)_{ijk} ⊗ K^(patch)_{pp'} may not hold. The proposal has a graceful fallback (empirical calibration), but if this fails, the ConvNet tier's theoretical contribution evaporates. *Severity: HIGH for the paper's novelty claim, MODERATE for the tool's functionality.*

### MODERATE

5. **KADR evaluation concerns.** Not circular per se, but lacks an independent validation metric. Fixable by adding one non-KADR ground-truth indicator. *Severity: MODERATE.*

6. **PCA preprocessing destroys ConvNet spatial structure.** CIFAR-10 reduced to 100 PCA dimensions eliminates exactly the structure that makes ConvNets different from MLPs. ConvNet evaluation on PCA-reduced data is nearly meaningless. *Severity: MODERATE (restructured version can use 1D ConvNets on raw MNIST sequences as partial workaround).*

7. **ReLU excluded from analytic H-tensor theorems.** The smooth-activation requirement for the H_{ijk} derivation excludes ReLU, the most common activation. Empirical calibration still works for ReLU, but the theoretical contribution applies to a narrower domain than practitioners use. *Severity: MODERATE.*

8. **Race risk from Bordelon & Pehlevan.** ConvNet H-tensor derivations could be scooped. The defensible moat (working tool + KADR definition + Lyapunov formulation) is real but thin. *Severity: MODERATE (timing risk, not a flaw).*

---

## Final Scores

### ORIGINAL Proposal (50K LoC, ConvNet + ResNet + full evaluation)

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| **Value** | 5 | Transformer exclusion limits audience; tool ambition exceeds theoretical contribution; audience ~200-500 researchers |
| **Difficulty** | 7 | Genuine novel math (H_{ijk}) + substantial systems engineering; ~50K LoC is real work |
| **Best-Paper** | 4 | Too diffuse — 50K LoC tool paper dilutes the H-tensor contribution; evaluation gaps weaken claims |
| **CPU Feasibility** | 3 | 300–1000+ CPU-hours vs claimed 12–24; multi-week campaign on laptop, not overnight run |
| **Overall Feasibility** | 4 | Math risk (~35–50% for ConvNet H-tensor) × CPU infeasibility × scope creep = high failure probability |

### RESTRUCTURED Proposal (MLP base + ConvNet H-tensor highlight, ~20K LoC)

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| **Value** | 7 | Tight theory paper: H-tensor derivation is genuinely novel, MLP tool validates methodology, honest scope |
| **Difficulty** | 6 | Core math is hard but well-scoped; reduced engineering surface area; MLP math is known |
| **Best-Paper** | 5 | Strong if H-tensor factorization holds AND produces novel critical exponents; otherwise solid-not-spectacular |
| **CPU Feasibility** | 7 | MLP-only ground truth is ~150–200 hours (feasible multi-day); ConvNet H-tensor is analytic (cheap) |
| **Overall Feasibility** | 6 | MLP base at 92% × ConvNet H-tensor at 65% × CPU manageable = reasonable delivery probability |

---

## Final Verdicts

### ORIGINAL Proposal: **ABANDON**

The original 50K LoC scope is a trap. The CPU budget is 4–10x underestimated, the evaluation plan is infeasible on laptop hardware, and the theoretical contribution is diluted by engineering ambition. Three independent reviewers identified the same core problem from different angles: the proposal tries to build a tool when it should write a paper.

### RESTRUCTURED Proposal: **CONTINUE** (with mandatory conditions)

The restructured version focuses on what matters: novel H-tensor mathematics for ConvNets validated by a working MLP prototype. This is a publishable contribution with a realistic delivery path.

---

## Mandatory Conditions for Continuation (Restructured)

### Before committing (Week 1 gate):

1. **Replace bisection with full grid sweep + adaptive refinement.** The monotonicity assumption is unproven. Implement a coarse-to-fine sweep over γ that does not assume monotonicity. Budget: ~500 extra grid evaluations, acceptable within revised CPU estimate.

2. **Add one KADR-independent ground-truth metric.** Include linear probing accuracy gap OR representation similarity index as a secondary ground truth. The primary evaluation can still use KADR, but at least one ablation must show agreement with an independent metric.

3. **Revise CPU budget to 150–200 hours for MLP evaluation.** Acknowledge multi-day overnight run campaign. Plan evaluation as 4–5 overnight batches, not a single session.

### Before Phase 4 (ConvNet H-tensor, Week 7 gate):

4. **H_{ijk} factorization numerical test at N=8,16,32.** This gate already exists in the proposal. Enforce it strictly: if relative error exceeds 5% at ANY of these widths, abandon the analytic ConvNet H-tensor and report a negative result. Do NOT soften the threshold.

5. **Test on raw 1D sequences, not PCA-reduced images.** For ConvNet evaluation, use MNIST reshaped as 1D sequences (784-length) or synthetic 1D data where spatial structure is meaningful. PCA-reduced CIFAR-10 is not a valid ConvNet benchmark.

### Before paper submission:

6. **Report perturbative validity coverage.** What fraction of the phase diagram has V[Θ] < 0.5 (high-confidence)? If less than 60% of grid points are high-confidence, the system's practical value is too limited for the claimed contribution.

7. **One-loop convergence check at all reported widths.** The proposal drops two-loop verification (reasonable), but must verify one-loop self-consistency: predictions at N=256 should be consistent with extrapolation from N=128 and N=512 calibration. If not, restrict claims to widths where consistency holds.
