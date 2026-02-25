# Finite-Width Phase Diagrams: Competing Approaches

## Problem Summary

Build the first integrated system that computes finite-width phase diagrams predicting the lazy-to-rich transition boundary for neural network training, validated experimentally against ground-truth training runs. The system stitches together NTK theory, mean-field limits, and µP parameterization at finite width.

---

## Approach A: Theory-First — Perturbative Field Theory of Kernel Evolution

### Extreme Value

**Theoretical physicists and mathematical ML researchers** who study neural network training dynamics as a statistical field theory desperately need this. Currently, the communities working on NTK corrections (Dyer & Gur-Ari, Huang & Yau), mean-field limits (Mei, Montanari & Nguyen), and µP (Yang & Hu) each publish results in isolation with no way to computationally verify where their regimes of validity overlap, conflict, or break down. A researcher studying, say, the role of depth in feature learning currently has no tool to compute whether their 1/N expansion is even convergent for the architecture they care about. This approach gives them a rigorous perturbative calculus — a symbolic algebra of kernel evolution — where every approximation has a computable error bound and every regime boundary has a precise mathematical characterization as a bifurcation in a dynamical system. The desperate need: without this, theorists keep publishing results about infinite-width limits that practitioners rightly ignore because they cannot tell when those results apply at width 256.

### Genuine Difficulty

**Hard Subproblem 1: Deriving H_{ijk} for weight-shared architectures.** The Hessian-Jacobian contraction tensor for convolutional layers requires tracking spatial index structure through weight sharing. Unlike dense layers where H_{ijk} decomposes into independent per-neuron contributions, convolutional layers introduce cross-patch correlations in the higher cumulants. The combinatorics of index contraction through stride, padding, and multi-channel convolutions produces terms that don't simplify to known forms. This is genuinely new mathematics — not a straightforward extension of MLP results.

**Hard Subproblem 2: Two-loop convergence verification.** To validate the one-loop (1/N) truncation, we need the two-loop (1/N²) corrections at least approximately. The two-loop computation involves 6th-order cumulants (κ_6) and mixed contraction tensors that scale as O(L³) in depth. Computing these symbolically for structured architectures may be intractable; we need a strategy to estimate two-loop contributions without computing them exactly.

**Hard Subproblem 3: Bifurcation classification in high-dimensional parameter space.** The phase boundary in (γ, depth, data-complexity) space is not generically a smooth manifold. Codimension-2 bifurcations (where two eigenvalue curves cross simultaneously) create cusps, and Hopf bifurcations create oscillatory regions. The numerical continuation algorithm must handle these singularities without human intervention.

**Architectural Challenge:** The symbolic kernel engine must represent per-layer kernel recursions for arbitrary compositions of dense, convolutional, and residual blocks. This requires a term-rewriting system that can symbolically differentiate through architecture-dependent index contractions, then lower to numerical evaluation. The symbolic and numerical layers must be co-designed so that symbolic simplification (which can reduce computation by orders of magnitude for structured architectures) feeds into the numerical ODE solver.

### New Math Required

**1. Convolutional Hessian-Jacobian Contraction Tensor.** For a convolutional layer with kernel size k, C_in input channels, C_out output channels, and spatial dimension S:

$$H^{(\text{conv})}_{ijk} = \frac{1}{C_\text{out}} \sum_{c=1}^{C_\text{out}} \sum_{p,p'} K(p,p') \cdot \frac{\partial^2 f_i}{\partial W_{c,p} \partial W_{c,p'}} \cdot \frac{\partial f_j}{\partial W_{c,p}} \cdot \frac{\partial f_k}{\partial W_{c,p'}}$$

where K(p,p') is the patch-overlap kernel encoding weight-sharing structure. The key insight is that weight sharing constrains H_{ijk} to lie in a low-dimensional subspace determined by the patch-overlap geometry, reducing the effective rank of the correction. This factorization is load-bearing: without it, the per-step cost of the corrected ODE is O(n³S²) instead of O(n³ · rank(K)).

**2. Residual Skip-Connection Jacobian Composition.** For a residual block f(x) = x + g(x), the Jacobian factors as J_res = I + J_g, making the NTK decompose as Θ_res = Θ_skip + Θ_cross + Θ_branch. The 1/N correction to the cross-term Θ_cross involves mixed cumulants between skip and branch pathways:

$$\Theta^{(1)}_{\text{cross},ij} = \frac{1}{N} \sum_k \text{cum}_4(J^{\text{skip}}_i, J^{\text{branch}}_j, J^{\text{branch}}_k, r_k)$$

This 4th mixed cumulant has no analog in the MLP theory and requires a new factorization lemma for additive Jacobian composition.

**3. Perturbative Validity Functional.** Define V[Θ] = ||Θ^(1)||_op / ||Θ^(0)||_op as the perturbative validity ratio. The phase diagram is partitioned into three zones: V < 0.2 (high confidence, one-loop sufficient), 0.2 ≤ V < 0.5 (moderate confidence, report with UQ), V ≥ 0.5 (low confidence, flag as unreliable). The boundary ∂{V = 0.5} is itself a computable surface in hyperparameter space that bounds the domain of validity of the entire approach.

**4. Spectral Bifurcation Theory for Kernel ODEs.** The linearized kernel evolution around the NTK fixed point defines a linear operator L(γ) on the space of symmetric n×n matrices. The lazy-to-rich transition occurs at γ_c where the spectral abscissa s(L(γ_c)) = max Re(λ_i(L(γ_c))) crosses zero. For the parameterized family L(γ, d, σ_data), we derive conditions under which this crossing is: (a) transversal (generic, giving a smooth phase boundary), (b) tangential (codimension-1, giving cusps), or (c) oscillatory (Hopf, giving edge-of-stability connection). The bifurcation type determines the universality class of the transition — a novel prediction.

### Best Paper Potential

This wins best paper because it does for neural network training dynamics what renormalization group theory did for statistical mechanics: it provides a systematic, improvable perturbative framework that unifies previously disparate results and makes novel quantitative predictions. The specific killer result is **architecture-dependent universality classes** — demonstrating computationally that ConvNets, ResNets, and MLPs have distinct critical exponents at the lazy-to-rich transition, meaning the phase transition is not universal but depends on architectural symmetry (translation invariance for ConvNets, additive structure for ResNets). This would be the first concrete demonstration that architecture imposes symmetry constraints on training dynamics with measurable consequences, connecting neural network theory to the deep mathematical structure of phase transitions. The combination of new mathematics (H_{ijk} for structured architectures), a working computational tool, and sharp falsifiable predictions (critical exponents that differ across architectures) hits the trifecta that best papers require.

### Hardest Technical Challenge

**Deriving and validating H_{ijk} for convolutional architectures with correct index structure.**

The weight-sharing pattern of convolutions means that the standard MLP derivation (where each weight matrix entry is independent) breaks down. In a convolutional layer, the same kernel weights appear at every spatial position, introducing correlations in the Hessian that don't factor per-neuron. The contraction H_{ijk} involves sums over spatial positions weighted by the patch-overlap structure, and getting the combinatorial prefactors correct requires tracking 6 indices (input channel, output channel, two spatial positions, two data indices) through two levels of differentiation.

**How to address it:**
1. **Derive for 1D convolutions first** (where spatial structure is a single axis), verify against brute-force finite-difference computation of the exact finite-width kernel at small N (N=8, 16, 32).
2. **Factor the spatial structure.** Show that H^(conv)_{ijk} = H^(dense)_{ijk} ⊗ K^(patch)_{pp'} where K^(patch) is the patch-overlap Gram matrix. This factorization (if it holds) reduces the problem to the known dense case tensored with a data-independent geometric kernel.
3. **Validate convergence empirically.** For each architecture, compute the exact NTK at widths N = {32, 64, 128, 256, 512, 1024} via autodiff, extract the empirical Θ^(1) by regression against 1/N, and compare to the analytically derived Θ^(1). Agreement to <5% relative error at N=128 is the acceptance criterion.
4. **Implement symbolically first, then optimize.** Build the derivation in a symbolic computation framework (SymPy) where index contractions can be verified term-by-term, then lower to optimized NumPy for production evaluation.

### Scores

| Criterion | Score | Rationale |
|-----------|-------|-----------|
| **Value** | 9 | Fills a fundamental gap in theoretical ML; gives theorists a computational laboratory |
| **Difficulty** | 9 | New math for structured architectures, symbolic-numeric co-design, bifurcation theory |
| **Potential** | 10 | Architecture-dependent universality classes would reshape how we think about training dynamics |
| **Feasibility** | 5 | Risk concentrated in whether H_{ijk} factorization works for convolutions; two-loop verification may be intractable |

---

## Approach B: Systems-First — Adaptive Kernel Observatory with Empirical Calibration

### Extreme Value

**ML engineers at mid-size companies training ConvNets and ResNets for production** (medical imaging, autonomous driving perception, scientific computing) desperately need this. These teams run hyperparameter sweeps costing $10K–$100K in compute, and the most expensive failure mode is discovering *after* training that the network was in the wrong dynamical regime — e.g., a wide ResNet trained with a learning rate that kept it in the lazy regime, wasting depth and producing a model no better than a kernel machine. Currently, there is no pre-training diagnostic that predicts regime. Phase diagrams computed in minutes on a laptop before launching a sweep would let these teams eliminate 30–50% of their hyperparameter grid, saving days of GPU time per experiment cycle. The second desperate need is **µTransfer validation**: when transferring hyperparameters across widths, practitioners have no way to check whether the transfer crosses a phase boundary. Silent failures (transferred hyperparameters that change the qualitative training regime) are undetectable without this tool.

### Genuine Difficulty

**Hard Subproblem 1: Robust numerical kernel computation at scale.** The symbolic kernel engine must compute NTK and NNGP kernels for arbitrary compositions of dense, convolutional, and residual blocks on datasets of size n=2K–5K. For convolutional layers, kernel evaluation involves numerical integration over patch structure with no closed form. The system must handle this without numerical instability (kernel matrices must remain positive definite throughout the ODE integration) and without excessive memory consumption (n×n matrices at double precision for n=5K require 200MB each; the ODE solver needs ~10 such matrices simultaneously).

**Hard Subproblem 2: Adaptive phase boundary refinement.** A naive grid sweep over a 3D hyperparameter space (learning rate × width × initialization scale) at resolution 50³ = 125K points is too expensive (each point requires solving the kernel ODE). The system needs an adaptive refinement strategy that quickly identifies rough boundary locations with coarse sweeps, then concentrates computation near boundaries. This is a computational geometry problem: maintaining a mesh in hyperparameter space that refines near detected boundaries while respecting the ODE solver's error tolerances.

**Hard Subproblem 3: Ground-truth regime classification.** The evaluation harness must classify each training run as "lazy" or "rich" using multiple indicators (kernel alignment drift, representation similarity, weight displacement, linear probing gap). These indicators can disagree, and the classification must be robust to noise across random seeds. Defining a reliable multi-criteria ground truth that is neither too permissive (classifying everything as rich) nor too conservative is itself a methodological challenge.

**Architectural Challenge:** The compiler pipeline must be production-quality: robust to malformed architecture specifications, informative error messages when an architecture falls outside the supported class, graceful degradation when the perturbative expansion diverges. The system must feel like a tool, not a research prototype. This means investment in input validation, error handling, progress reporting, and output visualization that research code typically skips.

### New Math Required

**1. Empirically Calibrated Finite-Width Corrections.** Instead of deriving H_{ijk} symbolically for each architecture type, this approach computes it numerically. At a set of calibration widths N_cal = {32, 64, 128}, compute the exact NTK via autodiff and extract Θ^(1) by fitting the model Θ(N) = Θ^(0) + Θ^(1)/N + Θ^(2)/N² to the measured kernels. This gives a numerical H_{ijk} without requiring the symbolic derivation. The load-bearing math is the **regression estimator** for Θ^(1):

$$\hat{\Theta}^{(1)} = \left(\sum_{j} \frac{1}{N_j^2}\right)^{-1} \sum_j \frac{1}{N_j} \left(\Theta(N_j) - \hat{\Theta}^{(0)}\right)$$

with bootstrap confidence intervals from resampling over initialization seeds. This is mathematically simple but requires careful treatment of the estimation error: the uncertainty in Θ^(1) propagates to uncertainty in the phase boundary location.

**2. Nyström-Corrected Kernel Spectral Decomposition.** The Nyström approximation replaces the n×n kernel matrix with a rank-m approximation (m ≈ 500–2000). For phase boundary detection, we need the top eigenvalues of the kernel and their evolution under the ODE. The Nyström error in eigenvalue estimation is:

$$|\lambda_i - \hat{\lambda}_i| \leq \frac{||K - K_m||_2}{\delta_i}$$

where δ_i is the eigenvalue gap. Near phase boundaries, eigenvalue gaps shrink, amplifying Nyström error exactly where accuracy matters most. The fix: **adaptive rank selection** that increases m near detected boundaries until the eigenvalue error is below the bifurcation detection threshold.

**3. Multi-Indicator Regime Classification with Bayesian Fusion.** Define four binary indicators: (a) kernel alignment drift > threshold, (b) CKA between initial and final representations < threshold, (c) weight displacement > lazy-regime prediction, (d) linear probing gap > threshold. Each indicator has architecture-dependent noise characteristics. Fuse them via a simple Bayesian model:

$$P(\text{rich} | \text{indicators}) = \frac{P(\text{indicators} | \text{rich}) P(\text{rich})}{P(\text{indicators})}$$

where the per-indicator likelihoods are estimated from calibration runs. This gives a probabilistic ground truth with calibrated uncertainty, replacing the ad-hoc "≥3/4 agree" rule.

**4. Pseudo-Arclength Continuation with Error Control.** Track phase boundaries as curves in hyperparameter space using predictor-corrector continuation. The predictor uses a tangent extrapolation; the corrector solves a bordered system. The key addition is **error-controlled step size**: the step size along the boundary is adapted so that the eigenvalue gap at the corrector step matches the eigenvalue gap at the predictor step to within a tolerance. This prevents the continuation from "jumping" across nearby bifurcation curves in regions of complex phase topology.

### Best Paper Potential

This wins best paper by demonstrating **immediate practical impact with rigorous methodology**. The killer result is a head-to-head comparison: for 10 realistic architecture-dataset pairs, compare the cost of hyperparameter optimization with and without phase-diagram-guided sweep triage. If the phase diagram eliminates 30–50% of the sweep volume while maintaining the same final model quality, the paper pays for itself in the first experiment. The methodological contributions — empirically calibrated 1/N corrections, adaptive Nyström rank selection, Bayesian ground-truth fusion — are individually publishable techniques that generalize beyond this specific application. The paper would demonstrate that rigorous theoretical tools can have practical engineering value without requiring the user to understand the underlying mathematics, which is a narrative that resonates strongly with the ML community's current appetite for "theory that works." The robustness story (graceful degradation, calibrated UQ, honest failure reporting) would set a new standard for how theoretical ML tools should be engineered.

### Hardest Technical Challenge

**Making the empirically calibrated 1/N corrections accurate enough for phase boundary prediction while remaining computationally cheaper than ground-truth training.**

The fundamental tension: extracting Θ^(1) by fitting to exact kernels at calibration widths requires computing exact NTKs, which involves autodiff through the full network. For a ResNet-20 with width 128, the NTK has size n² × P where P is the parameter count (~500K). Computing this for n=2K data points is itself expensive (~10 minutes per width on CPU). The calibration set {32, 64, 128} requires 3 such computations per hyperparameter configuration, and the phase diagram sweeps over hundreds of configurations.

**How to address it:**
1. **Amortize calibration.** The architecture-dependent part of Θ^(1) (the H_{ijk} tensor structure) is data-independent. Compute it once per architecture on a small proxy dataset (n=200), then apply it to the full dataset. Only the data-dependent spectral alignment needs recomputation per dataset.
2. **Warm-start across the hyperparameter grid.** Adjacent grid points have similar kernels. Use the previous point's ODE solution as the initial condition for the next point, reducing ODE solver iterations by ~60%.
3. **Sparse eigenvalue tracking.** For bifurcation detection, we only need the top-k eigenvalues (k ≈ 10–20) of the linearized operator, not the full spectrum. Use iterative eigenvalue solvers (Lanczos/Arnoldi) that cost O(n² · k) per step instead of O(n³).
4. **Fallback to analytic corrections.** For architectures where empirical calibration is too expensive, fall back to the MLP-derived H_{ijk} with an architecture-dependent scaling correction fitted from a small number of calibration points. This is less accurate but much cheaper.

### Scores

| Criterion | Score | Rationale |
|-----------|-------|-----------|
| **Value** | 10 | Direct dollar savings for practitioners; immediately deployable |
| **Difficulty** | 7 | Systems engineering challenge with known mathematical ingredients; novelty is in integration |
| **Potential** | 7 | Strong applied contribution but less theoretical novelty; unlikely to reshape the field's understanding |
| **Feasibility** | 8 | Empirical calibration sidesteps the hardest math; well-scoped engineering with fallback strategies |

---

## Approach C: Hybrid — Renormalization Group Flow on Architecture Graphs

### Extreme Value

**The research community working at the intersection of deep learning theory and statistical physics** desperately needs this. This community (Roberts, Yaida & Hanin; Bahri, Kadmon, Schoenholz; Halverson, Maiti, Stoner) has established that neural networks at finite width exhibit phenomena analogous to phase transitions, critical phenomena, and renormalization group flows in statistical mechanics. But the analogy has remained qualitative — "neural networks are like spin systems" — because no one has built the computational infrastructure to make it quantitative. This approach takes the analogy seriously: it implements the architecture graph as a lattice, the per-layer kernel statistics as field variables, and the depth direction as RG flow. The result is a phase diagram computed not by brute-force parameter sweeps but by identifying fixed points and relevant operators of the RG flow — the same methodology that classifies universality in condensed matter physics. The desperate need: the physics-of-deep-learning community is producing increasingly sophisticated theoretical results with no computational validation, and is at risk of becoming disconnected from empirical ML. This tool reconnects theory and experiment.

### Genuine Difficulty

**Hard Subproblem 1: Defining the RG transformation for neural network layers.** In statistical physics, the RG transformation integrates out short-distance degrees of freedom while preserving long-distance behavior. For neural networks, the analogous operation is integrating out a layer's neurons while preserving the input-output kernel. This is well-defined in the infinite-width limit (the kernel recursion), but at finite width, integrating out neurons introduces corrections that depend on the specific weight realization. The challenge is defining an RG transformation that (a) is computable, (b) reduces to the infinite-width kernel recursion at leading order, and (c) systematically captures 1/N corrections as "relevant operators" near the Gaussian fixed point.

**Hard Subproblem 2: Fixed-point analysis of the kernel flow.** The infinite-width NTK is a fixed point of the depth-direction RG flow (the kernel converges to a fixed kernel as depth → ∞ for appropriate initialization). The lazy-to-rich transition corresponds to this fixed point losing stability — a relevant perturbation growing under RG flow. Computing the stability matrix of the fixed point (the "anomalous dimensions" of operators near the Gaussian fixed point) requires diagonalizing a linear operator on the space of kernels, which is infinite-dimensional. Finite-dimensional truncation (representing kernels by their top-m spectral coefficients) introduces truncation error that must be controlled.

**Hard Subproblem 3: Connecting the RG framework to the compiler pipeline.** The RG approach gives a different computational structure than the direct ODE approach. Instead of solving coupled ODEs over the full hyperparameter grid, the RG approach computes fixed points and their stability once, then reads off the phase diagram from the fixed-point structure. But the compiler pipeline (architecture → IR → backends → phase diagram) was designed for the direct approach. Refactoring the pipeline to use the RG framework as the primary computation while retaining the direct ODE approach as a validation backend requires a clean abstraction boundary between "phase diagram computation" and "phase diagram representation."

**Architectural Challenge:** The system must simultaneously support two computational paradigms — the direct perturbative ODE approach (for validation and for architectures where the RG framework doesn't apply) and the RG flow approach (for deeper theoretical insight and potentially faster computation for deep networks). The IR must be rich enough to express both: layer-by-layer kernel recursion (for direct) and coarse-graining operations (for RG). This dual-paradigm architecture is the engineering novelty.

### New Math Required

**1. Finite-Width Kernel Renormalization Group.** Define the single-layer RG transformation R_N: for a layer with N neurons, input kernel K_in, weights W ~ N(0, σ²_w/N), and nonlinearity φ:

$$R_N[K_{\text{in}}]_{ij} = \frac{1}{N}\sum_{\alpha=1}^{N} \phi\left(\sum_k W_{\alpha k} x_{ik}\right) \phi\left(\sum_k W_{\alpha k} x_{jk}\right)$$

At infinite width, R_∞[K] = E[φ(z_1)φ(z_2)] where (z_1,z_2) ~ N(0, K̃) is the known NNGP recursion. The finite-width correction is:

$$R_N[K] = R_\infty[K] + \frac{1}{N}\mathcal{R}^{(1)}[K] + O(1/N^2)$$

where R^(1)[K] is a linear operator on kernels computable from 4th moments of φ. The phase diagram is determined by the spectrum of the linearization DR_N at the fixed point K*: perturbations with |DR_N · δK| > |δK| are relevant (grow under RG flow = increasing depth) and drive the system away from the Gaussian/lazy fixed point toward the feature-learning regime.

**2. Architecture-Aware Coarse-Graining.** For convolutional architectures, the RG transformation must respect spatial structure. Define a spatial coarse-graining C_s that reduces spatial resolution (analogous to block-spin transformation) alongside the width-direction integration:

$$\tilde{R}_N = C_s \circ R_N$$

The combined transformation maps a layer's kernel (indexed by spatial positions and data points) to a coarse-grained kernel. The fixed points of R̃_N correspond to scale-invariant architectures, and their stability determines depth-dependent phase structure. For residual architectures, the skip connection modifies the RG transformation to R_N^{(res)}[K] = K + R_N[K], whose fixed-point structure differs qualitatively (the identity term ensures the Gaussian fixed point is never fully unstable, explaining why ResNets are easier to train at depth).

**3. Anomalous Dimensions and Critical Exponents.** Near the Gaussian fixed point K*, linearize the RG flow: δK_{l+1} = DR_N(K*) · δK_l + O(δK²). The eigenvalues λ_i of DR_N(K*) define scaling dimensions Δ_i = -log|λ_i| / log(scale_factor). Operators with Δ_i < 0 (|λ_i| > 1) are relevant and determine the phase boundary. The critical exponent ν governing the divergence of the crossover length near the phase boundary is:

$$\nu = -\frac{1}{\log |\lambda_{\text{max}}|}$$

where λ_max is the largest relevant eigenvalue. **This is the key novel prediction:** ν depends on the architecture through DR_N, which depends on the nonlinearity, weight-sharing structure, and skip connections. Different architectures generically have different ν, placing them in different universality classes. This prediction is falsifiable: measure the width of the crossover region as a function of depth (or width) and extract ν empirically.

**4. Phase Diagram from Fixed-Point Structure.** The full phase diagram in (γ, depth, data-complexity) space is reconstructed from the RG fixed-point structure without sweeping the hyperparameter grid:
- **Lazy phase:** RG flow converges to Gaussian fixed point K*. Criterion: all eigenvalues of DR_N(K*) have |λ_i| < 1 when evaluated at the target (γ, N).
- **Rich phase:** RG flow diverges from K*. Criterion: at least one eigenvalue has |λ_i| > 1.
- **Phase boundary:** max |λ_i| = 1. This is a codimension-1 surface computable by root-finding on the eigenvalue equation, avoiding the need for grid sweeps entirely.
- **Chaotic phase:** Multiple fixed points with complex stability structure. Detected by continuation of fixed points under parameter variation.

### Best Paper Potential

This wins best paper by **establishing a rigorous connection between neural network architecture and statistical-mechanical universality classes, with quantitative experimental validation.** The central claim — that different architectures exhibit different critical exponents at the lazy-to-rich transition, computable from the spectrum of the kernel RG operator — is the kind of deep structural insight that defines a best paper. It would be the first time the RG framework for neural networks produces a quantitative, experimentally verified prediction that goes beyond what simpler methods can achieve. The specific falsifiable prediction (architecture-dependent ν with distinct values for MLPs, ConvNets, and ResNets) gives reviewers something concrete to evaluate. If the predicted critical exponents match empirical measurements within error bars, it validates the entire physics-of-deep-learning program. The computational speedup over grid sweeps (computing phase boundaries from fixed-point analysis instead of point-by-point evaluation) demonstrates that the theoretical framework isn't just elegant but practically superior. The dual-paradigm architecture (RG + direct ODE with cross-validation) sets a methodological standard for how theoretical ML tools should be built: multiple independent computational paths to the same answer, with discrepancies flagged as opportunities for theoretical refinement.

### Hardest Technical Challenge

**Computing the spectrum of the linearized RG operator DR_N(K*) for convolutional and residual architectures with sufficient accuracy for phase boundary prediction.**

The operator DR_N acts on the space of n×n symmetric matrices (kernel perturbations), which is n(n+1)/2-dimensional. For n=2000, this is a ~2M-dimensional eigenvalue problem. Direct computation is impossible. The challenge is finding an efficient representation.

**How to address it:**
1. **Spectral truncation.** Expand kernel perturbations δK in the eigenbasis of K* itself: δK = Σ_i c_i v_i v_i^T + Σ_{i<j} c_{ij} (v_i v_j^T + v_j v_i^T). The RG operator DR_N approximately preserves this decomposition (diagonal and off-diagonal perturbations decouple at leading order in 1/N). Work in the truncated basis of the top-m eigenvectors of K*, reducing the problem to an m×m eigenvalue computation (m ≈ 50–100).
2. **Exploiting convolutional structure.** For convolutional architectures, K* has block-circulant structure (in the spatial indices). DR_N inherits this structure, enabling block-diagonalization via spatial Fourier transform. Each Fourier mode evolves independently under the RG, reducing the problem to m_spatial independent small eigenvalue problems.
3. **Iterative eigenvalue computation.** Use Arnoldi iteration to find only the dominant eigenvalues of DR_N without forming the full matrix. Each Arnoldi step requires one application of DR_N to a kernel perturbation, which costs O(n²) per layer (the same as one forward kernel evaluation). Finding the top-20 eigenvalues requires ~50 Arnoldi steps, totaling ~50 kernel evaluations per architecture — comparable to a single ODE solve.
4. **Cross-validation against direct ODE.** For architectures where both the RG approach and the direct perturbative ODE are computationally feasible (shallow networks, small datasets), compare phase boundaries computed by both methods. Agreement validates the RG truncation; disagreement identifies where the spectral truncation is insufficient and more eigenvectors are needed.

### Scores

| Criterion | Score | Rationale |
|-----------|-------|-----------|
| **Value** | 8 | Bridges deep learning theory and statistical physics; valued by both communities |
| **Difficulty** | 10 | Requires genuine mathematical novelty (kernel RG operator), new computational methods (spectral truncation of infinite-dimensional operators), and dual-paradigm system architecture |
| **Potential** | 10 | Architecture-dependent universality classes computed from first principles would be a landmark result |
| **Feasibility** | 4 | Multiple high-risk mathematical steps (RG operator well-definedness, spectral truncation accuracy, convolutional block-diagonalization); unclear if all will work for realistic architectures |

---

## Comparative Summary

| Criterion | A: Theory-First | B: Systems-First | C: Hybrid RG |
|-----------|-----------------|-------------------|--------------|
| **Value** | 9 | 10 | 8 |
| **Difficulty** | 9 | 7 | 10 |
| **Potential** | 10 | 7 | 10 |
| **Feasibility** | 5 | 8 | 4 |
| **Primary audience** | Theoretical ML researchers | ML engineers / practitioners | Physics-of-DL community |
| **Key novelty** | Symbolic H_{ijk} for ConvNets/ResNets | Empirically calibrated corrections with robust tooling | Kernel RG operator and universality classes |
| **Risk profile** | Medium-high (math may not factor cleanly) | Low-medium (engineering risk, not math risk) | Very high (multiple untested theoretical steps) |
| **Compute model** | Symbolic → numerical lowering | Empirical calibration + numerical ODE | RG fixed-point analysis + spectral methods |
| **Phase boundary method** | Direct bifurcation detection in ODE | Adaptive grid sweep with continuation | Fixed-point stability analysis |
| **Fallback if primary method fails** | Degrade to empirical calibration (→ Approach B) | Always works at some accuracy level | Fall back to direct ODE (→ Approach A) |

### Recommendation

**Approach B** is the safest path to a strong paper with high practical impact and meets all evaluation criteria. **Approach A** is the recommended stretch goal — if the H_{ijk} factorization for convolutions works, it produces a significantly stronger theoretical contribution. **Approach C** is the moonshot — highest ceiling but lowest floor. The optimal strategy may be to build the systems infrastructure of Approach B first, then attempt the theoretical contributions of Approach A on top, with Approach C's RG framework as a long-term research direction that the infrastructure enables.
