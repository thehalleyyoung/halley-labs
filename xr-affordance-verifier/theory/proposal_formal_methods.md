# Formal Methods Proposal: Coverage-Certified XR Accessibility Verifier

**Author role:** Formal Methods Lead
**Date:** 2026-03-08
**Status:** Theory-stage proposal — all definitions, theorems, and proofs for the formal foundation

---

## 1. Core Definitions

### Definition 1: XR Accessibility Scene Model

An **XR accessibility scene** is a tuple Ω = (E, G, D) where:

- **E = {e₁, ..., eₘ}** is a finite set of *interactable elements*. Each element eᵢ is described by:
  - **pᵢ ∈ ℝ³**: centroid position in world coordinates
  - **Vᵢ ⊂ ℝ³**: activation volume (a compact semialgebraic set — typically an axis-aligned box, sphere, or convex hull defined by polynomial inequalities)
  - **τᵢ ∈ {click, grab, slide, dial, proximity}**: interaction type
  - **aᵢ ∈ {one-hand-dominant, one-hand-nondominant, bimanual, gaze, head}**: required actuator
  - **rᵢ ⊂ SE(3)**: required end-effector pose region for successful interaction (orientation + position constraints; e.g., a grab requires the hand to be within Vᵢ with palm facing the object)

- **G = (E, Dep)** is the *interaction dependency graph*, a directed acyclic graph where (eᵢ, eⱼ) ∈ Dep means element eⱼ can only be activated after eᵢ has been successfully activated (models multi-step interaction sequences, conditional visibility, unlock gates).

- **D = {d₁, ..., dₚ}** is a finite set of *target device configurations*. Each device dⱼ specifies:
  - **Tⱼ ⊂ ℝ³**: tracking volume (compact set where hand/controller tracking is available)
  - **Iⱼ ⊂ {click, grab, slide, dial, proximity}**: supported interaction types
  - **Mⱼ ∈ {seated, standing, room-scale}**: supported movement modes
  - **cⱼ: SE(3) → SE(3)**: controller-to-hand transform (maps controller pose to effective hand pose; identity for hand tracking)

**Notation.** |E| = m (element count), |D| = p (device count), tw(G) denotes the treewidth of the undirected skeleton of G, depth(G) denotes the longest path in G.

**Well-formedness.** Ω is well-formed if G is acyclic, every Vᵢ is a compact semialgebraic set (expressible as a Boolean combination of polynomial inequalities in ℝ³), and every rᵢ is a compact semialgebraic set in SE(3) (after chart embedding — see Definition 6).

---

### Definition 2: Parameterized Kinematic Body Model

A **parameterized kinematic body model** is a tuple B = (Θ, K, J, FK) where:

- **Θ ⊂ ℝᵈ** is the *body parameter space*, a compact convex subset of ℝᵈ representing the range of anthropometric parameters. We use d = 5 primary parameters drawn from ANSUR-II:
  - θ₁: stature (cm), range [147.3, 198.6] (1st–99th percentile female/male combined)
  - θ₂: arm length (acromion to fingertip, cm), range [62.1, 85.2]
  - θ₃: shoulder breadth (biacromial, cm), range [33.0, 46.5]
  - θ₄: forearm length (radiale to stylion, cm), range [21.5, 30.2]
  - θ₅: hand length (cm), range [15.9, 22.3]

  The parameter space Θ is the axis-aligned hyperrectangle defined by these ranges, further refined by correlation constraints from ANSUR-II (e.g., arm length is positively correlated with stature).

- **K(θ)** is the *kinematic chain* for body parameters θ, a sequence of rigid links connected by revolute joints. We model a 7-DOF arm:
  - Links: upper arm (length l₁(θ) = f₁(θ₂, θ₄)), forearm (length l₂(θ) = f₂(θ₄)), hand (length l₃(θ) = f₃(θ₅))
  - Joints: shoulder (3-DOF: flexion/extension, abduction/adduction, internal/external rotation), elbow (1-DOF: flexion/extension), wrist (3-DOF: flexion/extension, radial/ulnar deviation, pronation/supination)
  - Link lengths lᵢ(θ) are affine functions of the body parameters θ

- **J(θ) = [q_min(θ), q_max(θ)] ⊂ T⁷** is the *joint limit set* for body parameters θ, where T⁷ = (S¹)⁷ is the 7-torus of joint angles. Joint limits depend on body parameters (larger individuals typically have different ROM):
  - Shoulder flexion: [−60°, 180°] (typical); varies with θ
  - Shoulder abduction: [0°, 180°]
  - Shoulder rotation: [−90°, 90°]
  - Elbow flexion: [0°, 145°]
  - Wrist flexion/extension: [−80°, 70°]
  - Wrist deviation: [−20°, 35°]
  - Wrist pronation/supination: [−85°, 90°]

  Joint limits are modeled as axis-aligned hyperrectangles in joint space: J(θ) = ∏ᵢ₌₁⁷ [qᵢ,min(θ), qᵢ,max(θ)] where each qᵢ,min, qᵢ,max is an affine function of θ.

- **FK: Θ × T⁷ → SE(3)** is the *forward kinematics* map. For body parameters θ and joint angles q = (q₁,...,q₇):

  FK(θ, q) = T_base(θ) · R₁(q₁) · T₁(θ) · R₂(q₂) · T₂(θ) · ... · R₇(q₇) · T₇(θ)

  where T_base(θ) ∈ SE(3) positions the shoulder in world coordinates (depends on stature and shoulder breadth), each Rᵢ(qᵢ) ∈ SO(3) is a rotation matrix for joint i, and each Tᵢ(θ) ∈ SE(3) is a translation along the link direction (depends on link lengths).

**Key properties of FK:**
1. FK is smooth (C^∞) in both θ and q (composition of smooth rotations and translations)
2. FK is periodic in each qᵢ with period 2π (but joint limits restrict to a subset)
3. The Jacobian ∂FK/∂q has rank ≤ 6 everywhere (since FK maps to SE(3) ≅ ℝ³ × SO(3), which is 6-dimensional)

---

### Definition 3: Reachable Workspace

The **reachable workspace** of body θ is:

W(θ) = {FK(θ, q) | q ∈ J(θ)} ⊂ SE(3)

For practical computation, we project to the position component:

W_pos(θ) = {pos(FK(θ, q)) | q ∈ J(θ)} ⊂ ℝ³

where pos: SE(3) → ℝ³ extracts the translation component.

The **reachable workspace with orientation constraint** r ⊂ SE(3) is:

W(θ, r) = {q ∈ J(θ) | FK(θ, q) ∈ r}

This is the set of joint configurations that place the end-effector in the required pose region r.

---

### Definition 4: Accessibility Predicates

**Single-step accessibility.** Element e = (p, V, τ, a, r) is accessible for body θ on device d if there exists a joint configuration that places the end-effector in the required region while the end-effector position lies within the activation volume and within the device tracking volume:

Acc(e, θ, d) ≡ ∃q ∈ J(θ). [pos(FK(θ, q)) ∈ V ∩ T_d] ∧ [FK(θ, q) ∈ r] ∧ [τ ∈ I_d]

where T_d is the tracking volume and I_d is the supported interaction set of device d.

**Multi-step accessibility.** For an interaction sequence π = (e₁, e₂, ..., eₖ) with k ≤ 3, multi-step accessibility requires sequential reachability without geometric trapping:

Acc_k(π, θ, d) ≡ ∃q₁,...,qₖ ∈ J(θ).
  [∀i ∈ {1,...,k}. pos(FK(θ, qᵢ)) ∈ Vᵢ ∩ T_d ∧ FK(θ, qᵢ) ∈ rᵢ ∧ τᵢ ∈ I_d]
  ∧ [∀i ∈ {1,...,k-1}. Feasible(qᵢ, qᵢ₊₁, θ)]

where Feasible(qᵢ, qᵢ₊₁, θ) asserts that the body can transition from configuration qᵢ to qᵢ₊₁ via a continuous path within J(θ) (no joint-limit violation along the path). For k ≤ 3, we approximate: Feasible(qᵢ, qᵢ₊₁, θ) ≡ the linear interpolation qᵢ + t(qᵢ₊₁ - qᵢ), t ∈ [0,1], remains within J(θ). This is exact when J(θ) is convex (which it is, since J(θ) is an axis-aligned box in joint space).

**Population accessibility.** For a target population Θ_target ⊆ Θ and device set D:

PAcc(e, Θ_target, D) ≡ ∀θ ∈ Θ_target. ∀d ∈ D. Acc(e, θ, d)

**Accessibility frontier.** The accessibility frontier for element e and device d is:

∂F(e, d) = {θ ∈ Θ | θ is on the boundary of {θ' | Acc(e, θ', d)}}

This is the set of body parameterizations at which accessibility transitions between true and false.

---

### Definition 5: Accessibility Frontier Regularity

The accessibility function for element e and device d is:

acc(e, d, ·) : Θ → {0, 1}, acc(e, d, θ) = 𝟙[Acc(e, θ, d)]

The accessibility frontier ∂F(e, d) is **piecewise L-Lipschitz** if there exists a finite partition Θ = Θ₁ ∪ ... ∪ Θ_N (where each Θᵢ is a compact connected set with piecewise-smooth boundary) such that:

1. Within each Θᵢ, acc(e, d, ·) is constant (either identically 0 or identically 1)
2. Each boundary ∂Θᵢ ∩ ∂Θⱼ is a Lipschitz manifold of codimension 1 (i.e., locally the graph of an L-Lipschitz function)
3. The number of partition cells N is finite and bounded by a function of the scene geometry

**Why piecewise-Lipschitz, not globally Lipschitz.** As the Red-Team will correctly identify, the accessibility function has discontinuities at joint-limit boundaries. A body parameterization θ where a joint limit qᵢ,max(θ) = qᵢ* (the critical angle for reaching element e) creates a step from accessible to inaccessible. These transitions are:
- **Predictable**: they occur on known hypersurfaces in Θ defined by qᵢ,max(θ) = qᵢ*(e) or qᵢ,min(θ) = qᵢ*(e)
- **Detectable**: the kinematic model provides closed-form expressions for joint-limit boundaries
- **Finite in number**: for a scene with m elements and n joints, there are at most 2mn joint-limit transition surfaces

The piecewise-Lipschitz condition holds within each cell of the partition defined by these surfaces. The Lipschitz constant L within each cell depends on the gradient of the reachability boundary with respect to body parameters (i.e., how the workspace boundary moves as body dimensions change).

---

### Definition 6: Chart Embedding for SE(3) Predicates

To express SE(3) predicates in terms of polynomial inequalities (required for semialgebraic formulation), we use the following embedding.

**Position component.** pos: SE(3) → ℝ³ is the projection to the translational part. Position predicates "pos(FK(θ,q)) ∈ V" where V is a semialgebraic set in ℝ³ are directly expressible as polynomial inequalities in θ and q (since FK is a composition of rotations and translations, the position is a trigonometric polynomial in q with polynomial coefficients in θ).

**Orientation component.** For interaction types requiring orientation constraints (e.g., grab requires palm-facing), we use the axis-angle representation within a single chart of SO(3). For small orientation regions (typical for interaction constraints), a single chart suffices. The orientation constraint "FK(θ,q) has orientation within angle α of target orientation R₀" becomes:

‖log(R₀⁻¹ · R(θ,q))‖ ≤ α

where R(θ,q) is the rotation component of FK(θ,q) and log: SO(3) → so(3) is the matrix logarithm. For |α| < π (which covers all practical interaction constraints), this is well-defined and smooth.

**Practical simplification.** For Tier 1 (interval arithmetic), we over-approximate orientation constraints by ignoring them (checking position only). This is sound (position-reachable ⊇ position-and-orientation-reachable) and avoids the complexity of SO(3) computation. For Tier 2 (SMT), orientation constraints are linearized along with position (see Theorem C2).

---

### Definition 7: Coverage Certificate

A **coverage certificate** for scene Ω, target population Θ_target, and device set D is a tuple C = ⟨S, V, U, ε, δ⟩ where:

- **S = {(θ_j, v_j)}_{j=1}^{N_s}** is a *sample set*: a finite set of body parameterizations θ_j ∈ Θ_target paired with verdicts v_j = (v_j^{e,d})_{e∈E, d∈D} where v_j^{e,d} ∈ {accessible, inaccessible}. Each verdict is obtained by evaluating Acc(e, θ_j, d) via direct forward-kinematics computation (ground truth for that specific θ_j).

- **V = {(R_i, π_i)}_{i=1}^{N_v}** is a *verified region set*: a finite collection of compact regions R_i ⊂ Θ_target with SMT-backed proofs π_i certifying either ∀θ∈R_i. Acc(e, θ, d) = true or ∀θ∈R_i. Acc(e, θ, d) = false for specific (e,d) pairs. Each R_i is an axis-aligned hyperrectangle in Θ (matching the linearization envelope from C2).

- **U = {(H_l, μ_l)}_{l=1}^{N_u}** is the *unverified set annotation*: H_l are detected Lipschitz-violation hypersurfaces (joint-limit transitions) with estimated measure μ_l of their ε-neighborhoods. This explicitly tracks the "exempted" regions.

- **ε ∈ [0, 1]** is the *error bound*: an upper bound on P_μ(∃e∈E, d∈D. Acc(e,θ,d) ≠ cert_verdict(e,d,θ) | θ ∈ Θ_target \ ∪_l Nε(H_l)) where μ is the uniform measure on Θ_target, cert_verdict is the certificate's claimed verdict, and Nε(H_l) is the ε-neighborhood of the l-th violation hypersurface.

- **δ ∈ [0, 1]** is the *confidence parameter*: P(the ε bound holds) ≥ 1 - δ.

**Certificate validity.** A certificate C is *valid* if the ε bound holds with probability at least 1-δ over the randomness in the sampling procedure.

**Certificate completeness.** A certificate C is *κ-complete* if the total exempted volume satisfies:

μ(∪_l Nε(H_l)) / μ(Θ_target) ≤ κ

A κ-complete certificate with κ < 0.1 means the certificate covers at least 90% of the target population.

---

### Definition 8: Linearized Kinematic Approximation

For a reference point (θ₀, q₀) ∈ Θ × T⁷, the **first-order Taylor approximation** of FK is:

FK_lin(θ₀, q₀)(θ, q) = FK(θ₀, q₀) + J_θ(θ₀, q₀)·(θ - θ₀) + J_q(θ₀, q₀)·(q - q₀)

where:
- J_θ(θ₀, q₀) = ∂FK/∂θ|_{(θ₀,q₀)} ∈ ℝ^(6×d) is the body-parameter Jacobian
- J_q(θ₀, q₀) = ∂FK/∂q|_{(θ₀,q₀)} ∈ ℝ^(6×7) is the joint-angle Jacobian

The **soundness envelope** for linearization at (θ₀, q₀) with error tolerance η > 0 is:

Δ_max(θ₀, q₀, η) = sup{Δ > 0 | ∀(θ,q) ∈ B_∞(θ₀,Δ) × B_∞(q₀,Δ). ‖FK(θ,q) − FK_lin(θ₀,q₀)(θ,q)‖ ≤ η}

where B_∞(x, Δ) is the ℓ∞-ball of radius Δ centered at x.

---

## 2. Theorem Statements

### Theorem C1: Coverage Certificate Soundness (Piecewise-Lipschitz)

**Statement.** Let Ω = (E, G, D) be a well-formed XR accessibility scene. Let B = (Θ, K, J, FK) be a parameterized kinematic body model. Let Θ_target ⊆ Θ be the target population (compact). Suppose the following assumptions hold:

**(A1) Piecewise-Lipschitz frontier.** For each (e, d) ∈ E × D, there exists a partition of Θ_target into finitely many cells {Θ_target^{(e,d,i)}}_{i=1}^{N_{e,d}} such that:
- Each cell Θ_target^{(e,d,i)} is a compact connected set
- acc(e, d, ·) is constant on the interior of each cell
- The boundary between adjacent cells is an L_{e,d}-Lipschitz manifold of codimension 1

The partition boundaries are contained in the *joint-limit transition surfaces*: {θ | ∃j. q_{j,max}(θ) = q_j*(e) or q_{j,min}(θ) = q_j*(e)} for critical angles q_j*(e) determined by the element geometry.

**(A2) Identified violation surfaces.** The certificate construction identifies a set of hypersurfaces H₁, ..., H_N_u that contain all partition boundaries from (A1). Each H_l has an associated ε-neighborhood Nε(H_l) = {θ | dist(θ, H_l) < ε_l} with ε_l > 0.

**(A3) Sufficient sampling density.** Let Θ_smooth = Θ_target \ ∪_l Nε(H_l) (the "smooth region" after excluding violation neighborhoods). The sample set S is obtained by stratified random sampling over Θ_smooth with at least n_min samples per stratum, where:

n_min = ⌈(1/(2ε²)) · ln(2|Strata| · |E| · |D| / δ)⌉

and |Strata| is the number of sampling strata.

**(A4) SMT verification soundness.** Each verified region (R_i, π_i) ∈ V satisfies: the SMT proof π_i correctly certifies the stated property over R_i (relying on the soundness of the SMT solver for QF_LRA and the linearization error bound from Theorem C2).

**(A5) Compactness.** Θ_target is compact and has positive Lebesgue measure.

**Conclusion.** Under assumptions (A1)–(A5), the coverage certificate C = ⟨S, V, U, ε, δ⟩ satisfies:

P_sampling[∃θ ∈ Θ_smooth \ ∪_i R_i. ∃(e,d). cert_verdict(e,d,θ) ≠ acc(e,d,θ)] ≤ ε

with probability at least 1 - δ over the randomness of the stratified sampling procedure.

Moreover, the total population fraction excluded from the guarantee is bounded:

μ(∪_l Nε(H_l)) / μ(Θ_target) ≤ ∑_l 2ε_l · L_l^{d-1} · Area(H_l ∩ Θ_target) / μ(Θ_target)

where L_l is the Lipschitz constant and Area(·) is the (d-1)-dimensional Hausdorff measure.

---

### Theorem C1 — Proof Structure

**Proof strategy:** Union bound + Hoeffding + volume subtraction.

**Step 1: Reduce to per-element, per-device, per-stratum bounds.**

By union bound over elements e ∈ E, devices d ∈ D, and strata s ∈ Strata:

P[∃ error in Θ_smooth \ ∪R_i] ≤ ∑_{e,d,s} P[error for (e,d) in stratum s \ ∪R_i]

Allocate failure probability: each (e, d, s) triple gets δ_{e,d,s} = δ / (|E| · |D| · |Strata|).

**Step 2: Within each stratum, apply Hoeffding to the smooth sub-stratum.**

Fix an element e, device d, and stratum s. Let Θ_s = (stratum s) ∩ Θ_smooth be the smooth part of the stratum. Let V_s = ∪{R_i | R_i ⊆ Θ_s} be the verified subregion within the stratum. The unverified part is U_s = Θ_s \ V_s.

Within U_s, the accessibility function acc(e,d,·) is continuous (by A1, since we've excluded violation neighborhoods). The samples in U_s are i.i.d. uniform (conditioned on falling in U_s, which adjusts the effective measure).

Define X_j = 𝟙[acc(e,d,θ_j) ≠ cert_prediction(e,d,θ_j)] for samples θ_j ∈ U_s. By Hoeffding's inequality:

P[|X̄ - E[X]| > t] ≤ 2·exp(−2n_s·t²)

where n_s is the number of samples in U_s and X̄ is the sample mean.

Setting t = ε and solving for n_s to make the right side ≤ δ_{e,d,s}:

n_s ≥ (1/(2ε²))·ln(2/δ_{e,d,s}) = (1/(2ε²))·ln(2|E|·|D|·|Strata|/δ)

This is exactly assumption (A3).

**Step 3: Volume accounting for verified regions.**

The verified regions V_s are exempt from the Hoeffding bound because they have SMT proofs (by A4). The effective unverified volume fraction is:

ρ_s = μ(U_s) / μ(Θ_s)

If ρ_s < 1 (some volume is verified), the sampling bound applies only to the unverified fraction. The total error probability is bounded by:

ε_total ≤ max_s (ρ_s · ε_stratum)

This gives the ε-improvement over pure sampling: if 30% of volume is SMT-verified (ρ_s = 0.7), the effective ε improves by factor 1/0.7 ≈ 1.43×.

**Step 4: Aggregate across strata, elements, devices.**

By union bound:

ε_global ≤ max over all (e,d,s) of ε_s ≤ ε

where the max arises because a bug in any stratum/element/device pair constitutes a global error.

**What's novel:** The composition of stratified Hoeffding bounds with SMT-verified volume subtraction in the context of kinematic parameter-space verification. The individual components (Hoeffding, stratified sampling, SMT) are known; the certificate framework assembling them with piecewise-Lipschitz handling and volume accounting is new.

**What could go wrong:** The stratum count |Strata| enters logarithmically in n_min, but if adaptive stratification creates too many strata (>10⁴), the per-stratum sample count drops below useful levels. This is addressed by bounding |Strata| ≤ S_max in the budget allocation algorithm (C3).

---

### Theorem C2: Linearization Soundness Envelope

**Statement.** Let FK: Θ × T^n → ℝ³ be the position component of forward kinematics for an n-joint revolute chain with link lengths l₁,...,lₙ satisfying lᵢ ≤ L_max for all i. Let (θ₀, q₀) ∈ Θ × T^n be a reference configuration. Then for all (θ, q) with ‖θ - θ₀‖_∞ ≤ Δ_θ and ‖q - q₀‖_∞ ≤ Δ_q:

‖FK(θ, q) − FK_lin(θ₀, q₀)(θ, q)‖₂ ≤ C_FK · (Δ_q² + Δ_θ · Δ_q) · L_sum

where:
- L_sum = ∑ᵢ₌₁ⁿ lᵢ (total arm length)
- C_FK = n/2 (a dimensionless constant depending only on joint count)
- The bound accounts for second-order terms in both joint angles and body parameters

For the specific case of a 7-DOF arm (n = 7) with L_sum ≈ 0.7m (typical human arm):

‖FK − FK_lin‖₂ ≤ 3.5 · (Δ_q² + Δ_θ·Δ_q) · 0.7 = 2.45 · (Δ_q² + Δ_θ·Δ_q) meters

**Soundness envelope.** For error tolerance η > 0, the maximum linearization radius is:

Δ_max = min(Δ_q, Δ_θ) where Δ_q = √(η / (2·C_FK·L_sum)) and Δ_θ is set so that Δ_θ·Δ_q·C_FK·L_sum ≤ η/2.

For η = 1 cm = 0.01m: Δ_q = √(0.005 / 2.45) ≈ 0.045 rad ≈ 2.6°, Δ_θ = η/(2·C_FK·L_sum·Δ_q) ≈ 0.01/(2·2.45·0.045) ≈ 0.045 (normalized parameter units).

---

### Theorem C2 — Proof Structure

**Proof strategy:** Taylor remainder theorem applied to the kinematic chain composition.

**Step 1.** Write FK as a composition of rotation and translation operations:

FK(θ, q) = ∑ᵢ₌₁ⁿ lᵢ(θ) · (∏ⱼ₌₁ⁱ Rⱼ(qⱼ)) · ûᵢ

where Rⱼ(qⱼ) is the rotation matrix for joint j and ûᵢ is the unit direction of link i in its local frame.

**Step 2.** The second derivative of Rⱼ(qⱼ) with respect to qⱼ satisfies ‖∂²Rⱼ/∂qⱼ²‖ ≤ 1 (since Rⱼ is a rotation matrix and its second derivative is bounded by the identity).

**Step 3.** By the multivariate Taylor remainder theorem:

‖FK(θ,q) − FK_lin(θ₀,q₀)(θ,q)‖ ≤ (1/2) · sup_{ξ∈[θ₀,θ], η∈[q₀,q]} ‖H_FK(ξ,η)‖ · (‖θ-θ₀‖² + ‖q-q₀‖²)

where H_FK is the Hessian of FK.

**Step 4.** Bound the Hessian. The key observation is that each joint j contributes at most lⱼ · 1 to the Hessian norm (from ‖∂²Rⱼ/∂qⱼ²‖ ≤ 1 scaled by link length). Summing over joints:

‖H_FK‖ ≤ ∑ᵢ lᵢ · n_downstream(i)

where n_downstream(i) is the number of joints downstream of link i. For a serial chain, n_downstream(i) = n - i, giving:

‖H_FK‖ ≤ ∑ᵢ₌₁ⁿ lᵢ · (n-i) ≤ n · L_sum

**Step 5.** Combine: ‖FK − FK_lin‖ ≤ (n/2) · L_sum · (Δ_θ² + Δ_q²). Cross terms Δ_θ·Δ_q arise from mixed partials (body parameter affecting link length × joint angle affecting rotation direction). Including these: C_FK = n/2.

**Novelty:** This is a careful Taylor-remainder computation for kinematic chains. The technique is standard; the explicit constant C_FK = n/2 for revolute chains with application to accessibility verification is new.

---

### Theorem C4: Tier 1 Completeness Gap Bound

**Statement.** Let the affine-arithmetic forward kinematics evaluate to an enclosure E(θ_range, q_range) ⊇ W_pos(θ_range, q_range) with wrapping factor w ≥ 1 (i.e., vol(E) ≤ w · vol(W_pos)). For an interaction element e with activation volume V_e, define the *spatial margin*:

margin(e, θ) = inf_{x ∈ ∂V_e} dist(x, ∂W_pos(θ))

(the minimum distance between the boundaries of the activation volume and the workspace).

Then Tier 1 (interval arithmetic) is guaranteed to correctly classify element e for body parameter range [θ_min, θ_max] if:

1. **Definitely reachable:** margin(e, θ) > (w^{1/3} - 1) · diam(W_pos(θ)) for all θ ∈ [θ_min, θ_max]
2. **Definitely unreachable:** dist(V_e, E(θ_range, q_range)) > 0

Conversely, Tier 1 reports "inconclusive" (yellow) whenever:

margin(e, θ) ≤ (w^{1/3} - 1) · diam(W_pos(θ))

for some θ in the range. The *maximum undetectable bug radius* — the largest accessibility violation that Tier 1 can miss — is:

r_max = (w^{1/3} - 1) · diam(W_pos)

For w = 5 (target wrapping factor), r_max ≈ 0.71 · diam(W_pos). For a typical arm with diam(W_pos) ≈ 1.2m, r_max ≈ 85 cm. This means Tier 1 can miss a button placed 85 cm inside the reachability boundary if the wrapping inflates the enclosure to include it.

**Refinement.** With s subdivisions per dimension:
- Effective wrapping factor per sub-problem: w_s ≈ w^{1/s^d} (for d-dimensional subdivision)
- r_max(s) = (w_s^{1/3} - 1) · diam(W_pos) / s

For s = 4 subdivisions in 3D position space (64 sub-problems) and w = 5: w_s ≈ 5^{1/64} ≈ 1.025, r_max(4) ≈ 0.008 · 1.2/4 ≈ 2.5 mm. This brings Tier 1 to practical precision at the cost of 64× evaluation time (still under 2 seconds for typical scenes).

---

### Theorem C4 — Proof Structure

**Proof strategy:** Geometric argument about the over-approximation envelope.

**Step 1.** The affine-arithmetic enclosure E satisfies W_pos ⊆ E and vol(E) ≤ w · vol(W_pos). Assuming both are approximately spherical (worst case for over-approximation), the Hausdorff distance between ∂E and ∂W_pos is at most:

d_H(∂E, ∂W_pos) ≤ (w^{1/3} - 1) · r(W_pos)

where r(W_pos) = (3·vol(W_pos)/(4π))^{1/3} is the effective radius.

**Step 2.** If margin(e, θ) > d_H, then V_e is either entirely inside W_pos (definitely reachable) or entirely outside E (definitely unreachable). If margin(e, θ) ≤ d_H, the enclosure E may include V_e even if W_pos does not (or vice versa), leading to an inconclusive result.

**Step 3.** The refinement bound follows from the property that subdivision reduces wrapping super-exponentially (each sub-problem has narrower input ranges, and wrapping grows polynomially in range width).

**Novelty:** Connecting wrapping factor to spatial margin in the kinematic accessibility domain. The geometric argument is elementary; the connection to accessibility detection rates is new.

---

### Lemma B1: Affine-Arithmetic Wrapping Factor for Revolute Chains

**Statement.** For a k-joint revolute chain with joint ranges [qᵢ - Δᵢ, qᵢ + Δᵢ] (symmetric about center qᵢ) and link lengths l₁,...,lₖ, the affine-arithmetic wrapping factor for position space satisfies:

w ≤ ∏ᵢ₌₁ᵏ (1 + cᵢ · Δᵢ²)

where cᵢ = (1/2)·(lᵢ/∑ⱼ≥ᵢ lⱼ) is the relative contribution of joint i to downstream motion.

For uniform Δᵢ = Δ and equal link lengths l:

w ≤ (1 + Δ²/(2k))^k ≈ exp(Δ²/2) for large k

**Numerical predictions:**
- k = 4, Δ = 30° = 0.524 rad: w ≤ (1 + 0.034)⁴ ≈ 1.145 → wrapping ~15%
- k = 7, Δ = 30° = 0.524 rad: w ≤ (1 + 0.020)⁷ ≈ 1.148 → wrapping ~15%
- k = 7, Δ = 45° = 0.785 rad: w ≤ (1 + 0.044)⁷ ≈ 1.356 → wrapping ~36%
- k = 7, Δ = 60° = 1.047 rad: w ≤ (1 + 0.078)⁷ ≈ 1.706 → wrapping ~71%

**Critical note:** These are bounds on the *position volume* wrapping factor. The actual wrapping depends on the specific joint configuration (center angles qᵢ) and link geometry. The bound is tight for generic configurations but can be loose near kinematic singularities. The sin/cos Chebyshev approximation error contributes an additional factor that is O(Δ³) per joint (cubic in range width), which is dominated by the O(Δ²) quadratic wrapping for Δ < 1 rad ≈ 57°.

---

### Lemma B1 — Proof Structure

**Proof strategy:** Affine-arithmetic error propagation through the chain.

**Step 1.** Model sin(qᵢ) and cos(qᵢ) over [qᵢ - Δᵢ, qᵢ + Δᵢ] using Chebyshev affine approximations:

sin(q) ≈ sin(qᵢ) + cos(qᵢ)·(q - qᵢ) + ηᵢ_sin
cos(q) ≈ cos(qᵢ) - sin(qᵢ)·(q - qᵢ) + ηᵢ_cos

where |ηᵢ_sin|, |ηᵢ_cos| ≤ Δᵢ²/2 (quadratic residual from the linearization of trig functions).

**Step 2.** In affine arithmetic, the noise symbol ηᵢ from joint i propagates through all downstream joints. The contribution of ηᵢ to the final position uncertainty is bounded by lᵢ + lᵢ₊₁ + ... + lₖ (the total length of the downstream chain).

**Step 3.** The wrapping factor is the ratio of enclosure volume to true workspace volume. Each joint contributes a multiplicative factor (1 + noise_i / signal_i) where noise_i ≈ Δᵢ² · (downstream length) and signal_i ≈ downstream length. This gives the per-joint factor (1 + cᵢ·Δᵢ²).

**Novelty:** Tight wrapping analysis for affine arithmetic on kinematic chains, with explicit constants. Prior work (Stolfi & de Figueiredo 2003) gives generic affine-arithmetic bounds; we specialize to the kinematic chain structure.

---

### Theorem C3: Budget Allocation Optimality

**Statement.** Given total compute budget T (in seconds), frontier Lipschitz estimate L̂, and current certificate state, the optimal allocation (N_samples, N_smt, SMT_priorities) that minimizes ε subject to the time constraint is the solution to:

minimize ε(N_samples, N_smt)
subject to: t_sample · N_samples + t_smt · N_smt ≤ T
            N_samples ≥ 0, N_smt ≥ 0

where:
- ε(N_samples, N_smt) = √(ln(2·K/δ) / (2·N_samples/|Strata|)) · (1 - vol_verified(N_smt)/vol(Θ_smooth))
- K = |E|·|D|·|Strata| (Bonferroni count)
- vol_verified(N_smt) = N_smt · vol_per_query (volume verified per SMT query)
- t_sample ≈ 50μs (FK evaluation time), t_smt ≈ 100ms mean (SMT query time with timeout)

This is a single-variable optimization (since N_smt = (T - t_sample·N_samples)/t_smt as a function of N_samples). The objective is quasi-convex in N_samples, admitting efficient binary search for the optimum.

**Closed-form approximation.** When vol_per_query is small (typical), the optimal allocation heavily favors sampling:

N_samples* ≈ T / t_sample · (1 - α)
N_smt* ≈ T / t_smt · α

where α = t_sample/(t_sample + t_smt · vol_per_query/vol_stratum) is the optimal SMT fraction. For typical values (t_sample = 50μs, t_smt = 100ms, vol_per_query/vol_stratum ≈ 10⁻⁶): α ≈ 0.05%, meaning 99.95% of budget goes to sampling.

**The real value of SMT.** The above analysis shows that SMT's volumetric contribution is negligible. The actual value of SMT queries is *targeted boundary resolution*: an SMT query at the accessibility frontier doesn't just verify a tiny volume, it *resolves the boundary location* within that volume, allowing the Lipschitz-based interpolation to tighten ε for the surrounding region. This *information-theoretic* contribution of SMT is not captured by the simple volume-subtraction model.

**Enhanced model.** With frontier-resolution accounting:

ε_enhanced(N_smt) ≈ ε_sampling · max(0, 1 - N_smt · Δ_resolved / Δ_frontier)

where Δ_frontier is the total length of the accessibility frontier and Δ_resolved is the frontier length resolved per SMT query. This gives a much more favorable allocation (α ≈ 10-20% to SMT).

---

## 3. Assumption Catalog

### Structural Assumptions

| ID | Assumption | Testable? | What breaks if false | Mitigation |
|----|-----------|-----------|---------------------|------------|
| S1 | XR interaction graphs have bounded treewidth (tw ≤ 6 for typical scenes) | Yes — compute tw(G) for benchmark scenes | Compositional decomposition degrades; model checking is exponential in tw | Report tw for each scene; degrade gracefully to monolithic verification for high-tw subgraphs |
| S2 | Interactions are k-local with k ≤ 4 joints | Yes — analyze interaction types | Zone abstraction dimension grows with k; if k > 6, CAD-based approaches become intractable | Not relevant for our approach (we use sampling, not CAD), but affects wrapping factor for Tier 1 |
| S3 | Multi-step interactions have depth ≤ 3 | Enforced by system design | Certificate quality degrades for longer sequences (trajectory space dimension = k·d) | Hard limit at k = 3; longer sequences reported as "not certifiable" |
| S4 | Activation volumes are compact semialgebraic sets | Enforced by DSL; checked at parse time | SMT encoding may be unsound for non-semialgebraic volumes | Approximate non-semialgebraic volumes by conservative bounding boxes |

### Mathematical Assumptions

| ID | Assumption | Testable? | What breaks if false | Mitigation |
|----|-----------|-----------|---------------------|------------|
| M1 | Piecewise-Lipschitz accessibility frontier | Partially — Lipschitz constant can be estimated; partition structure is predicted by kinematic model | Certificate ε bound is invalid | Detect violations (Def 5), exclude neighborhoods, report exclusion fraction (κ-completeness) |
| M2 | Lipschitz constant L is estimable from samples | Empirically testable (cross-validation) | L underestimated → certificate unsound; L overestimated → ε too loose | Conservative L estimation with cross-validation; inflate L by safety factor 1.5× |
| M3 | Linearization error is bounded by C2 | Yes — numerically verifiable | SMT verification is unsound (linearized model doesn't match true kinematics) | Validate C2 bound empirically on 10K random configurations; use conservative η |
| M4 | Joint limits are axis-aligned boxes in joint space | Approximately true; real joints have coupling | Over-approximation of J(θ) → false negatives (missing some accessibility); under-approximation → false positives | Conservative (larger) joint limit boxes ensure soundness of accessibility verdict |
| M5 | Body parameter correlations are captured by Θ | Partially — ANSUR-II provides correlations | Sampling density is wrong in correlated dimensions → ε bound may be loose | Use ANSUR-II correlation matrix to define Θ as an ellipsoidal region rather than a box |

### Computational Assumptions

| ID | Assumption | Testable? | What breaks if false | Mitigation |
|----|-----------|-----------|---------------------|------------|
| C1_comp | Z3 is sound on QF_LRA | Trusted (well-verified solver) | SMT verdicts are wrong → certificate unsound | Use certified Z3 proofs (LFSC proof format) when available |
| C2_comp | FK evaluation is numerically stable | Testable via extended precision | Floating-point errors in FK → incorrect sample verdicts | Use compensated summation; validate against extended-precision reference for 1% of samples |
| C3_comp | SMT queries complete within 2s timeout | Empirically testable | Timed-out queries leave regions unverified → looser ε | Timeout is safe (region stays unverified); adjust budget allocation if timeout rate > 50% |

### Domain Assumptions

| ID | Assumption | Testable? | What breaks if false | Mitigation |
|----|-----------|-----------|---------------------|------------|
| D1 | ANSUR-II is representative of target XR users | No (inherent limitation) | Tool's population coverage claims are biased | Document limitation explicitly; supplement with disability-specific ROM data from rehabilitation literature |
| D2 | 7-DOF arm model captures relevant accessibility | Partially — compare with full-body model | Misses seated posture, trunk rotation, bimanual interactions | Acknowledge as limitation; extend model in future work |
| D3 | Unity scene format is stable across versions | Testable (parse multiple Unity versions) | Parser breaks on new Unity versions | Version-specific adapters; test on Unity 2021 LTS, 2022 LTS, 2023 |

---

## 4. What's New vs. What's Known

| Component | Closest Prior Work | What's New |
|-----------|-------------------|------------|
| Coverage certificates (C1) | Statistical model checking (Younes & Simmons 2002; Legay et al. 2010) — probabilistic guarantees for stochastic system properties | Application to deterministic parameter-space verification (not stochastic temporal properties); integration of SMT-verified regions into sampling-based bounds; piecewise-Lipschitz handling of joint-limit discontinuities |
| Linearization soundness (C2) | Taylor-model arithmetic (Berz & Makino 1998); interval analysis of robot kinematics (Merlet 2004) | Explicit constant C_FK = n/2 for revolute chains; coupling between body parameters and joint angles; application to accessibility verification SMT encoding |
| Completeness gap (C4) | Interval analysis error bounds (Moore 1966); affine arithmetic (Stolfi & de Figueiredo 2003) | Connection between wrapping factor and accessibility detection; spatial-margin formulation specific to XR interaction volumes |
| Wrapping factor (B1) | Affine arithmetic wrapping (Stolfi & de Figueiredo 2003); interval FK (Merlet 2004) | Tight per-chain analysis with explicit multiplicative structure; prediction of wrapping as function of chain length and joint range |
| Budget allocation (C3) | Multi-fidelity optimization (Peherstorfer et al. 2018); adaptive sampling for reliability analysis (Au & Beck 2001) | Application to certificate optimization; frontier-resolution model for SMT value |
| XR accessibility scene model | W3C XR Accessibility Requirements (2021); robotics workspace analysis (Zacharias et al. 2007 — Reuleaux reachability maps) | Formal scene model with dependency graph; parametric accessibility predicates; device-capability integration |
| Piecewise-Lipschitz frontier | Discontinuity detection in sampling (Rall 1981); sensitivity analysis (Saltelli et al. 2004) | Application to kinematic accessibility with joint-limit surfaces as known discontinuity sources |

---

## 5. Open Questions Requiring Resolution

1. **Is C_FK = n/2 tight?** The Hessian bound in C2 may be loose. Empirical measurement of the actual linearization error for ANSUR-II-parameterized arms would refine this constant. If the true constant is <n/4, the soundness envelope doubles, significantly improving SMT coverage.

2. **What is the practical Lipschitz constant L?** Theorem C1 requires L to be estimated. Preliminary analysis suggests L depends on the activation volume radius and the workspace gradient — but no closed-form expression for L as a function of scene geometry exists. Empirical L estimation with cross-validation (described in the evaluation plan) is the practical path forward.

3. **Does frontier-resolution SMT provide the claimed 5× improvement over Clopper-Pearson?** The simple volume-subtraction model predicts negligible improvement; the frontier-resolution model predicts 3-10×. Which model is correct depends on the empirical structure of accessibility frontiers. This is the key uncertainty and is tested at gate D3.

4. **How many joint-limit transition surfaces exist in practice?** Theorem C1 bounds these at 2mn (m elements × n joints × 2 limits per joint). For a 30-element scene with a 7-DOF arm, this is 420 surfaces. The exempted volume depends on the ε-neighborhood width — if ε_l is large, the certificate may be sparsely applicable. Empirical measurement at gate D4.
