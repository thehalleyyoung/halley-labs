# Algorithm Design Proposal: Coverage-Certified XR Accessibility Verifier

**Author role:** Algorithm Designer
**Date:** 2026-03-08
**Status:** Theory-stage proposal — implementable algorithms with pseudocode and complexity analysis

---

## 1. Algorithm 1: Affine-Arithmetic Forward Kinematics (Tier 1)

### 1.1 Overview

Computes a conservative enclosure of the end-effector workspace for a given range of body parameters and joint angles. Uses affine arithmetic to track correlations between variables, reducing over-approximation compared to naive interval arithmetic.

### 1.2 Data Structures

```
AffineForm {
    center: f64           // central value
    coeffs: Vec<f64>      // noise symbol coefficients [ε₁, ε₂, ..., εₖ]
    error: f64            // accumulated rounding error (non-affine residual)
}

// Each noise symbol εᵢ ∈ [-1, 1] represents one source of uncertainty.
// Noise symbols ε₁..ε₇ correspond to joint angles q₁..q₇.
// Noise symbols ε₈..ε₁₂ correspond to body parameters θ₁..θ₅.
// Additional symbols introduced by nonlinear operations (sin, cos, multiply).
```

### 1.3 Core Operations

```
// Affine addition: exact (no new noise symbols)
fn aa_add(a: AffineForm, b: AffineForm) -> AffineForm:
    center = a.center + b.center
    coeffs[i] = a.coeffs[i] + b.coeffs[i]  for all i
    error = a.error + b.error
    return AffineForm { center, coeffs, error }

// Affine multiplication: introduces one new noise symbol
fn aa_mul(a: AffineForm, b: AffineForm) -> AffineForm:
    center = a.center * b.center
    // Linear terms from product rule
    coeffs[i] = a.center * b.coeffs[i] + b.center * a.coeffs[i]  for all i
    // Quadratic residual → new noise symbol
    rad_a = sum(|a.coeffs[i]|) + a.error
    rad_b = sum(|b.coeffs[i]|) + b.error
    new_error = rad_a * rad_b  // Conservative bound on cross terms
    error = a.error * |b.center| + |a.center| * b.error + new_error
    return AffineForm { center, coeffs, error }

// Affine sine via Chebyshev linearization
fn aa_sin(a: AffineForm) -> AffineForm:
    // a represents [a.center - rad, a.center + rad] where rad = sum(|coeffs|) + error
    lo = a.center - a.radius()
    hi = a.center + a.radius()
    
    // Chebyshev linear approximation of sin on [lo, hi]
    // slope α minimizes max |sin(x) - αx - β| over [lo, hi]
    if hi - lo < 1e-10:
        return AffineForm { center: sin(a.center), coeffs: [0...], error: 0 }
    
    α = (sin(hi) - sin(lo)) / (hi - lo)  // Secant slope
    // Find optimal intercept β by minimizing max deviation
    // For Chebyshev approximation on [lo, hi]:
    // The maximum error occurs at the extremum of sin(x) - αx
    // d/dx(sin(x) - αx) = cos(x) - α = 0 → x* = arccos(α)
    x_star = arccos(clamp(α, -1, 1))
    if lo <= x_star <= hi:
        max_above = sin(x_star) - α * x_star
    else:
        max_above = max(sin(lo) - α*lo, sin(hi) - α*hi)
    min_below = min(sin(lo) - α*lo, sin(hi) - α*hi)
    β = (max_above + min_below) / 2
    δ = (max_above - min_below) / 2  // Approximation error radius
    
    // Propagate through affine form
    result.center = α * a.center + β
    result.coeffs[i] = α * a.coeffs[i]  for all i
    result.error = |α| * a.error + δ  // Add Chebyshev residual
    return result

// Affine cosine: cos(x) = sin(x + π/2)
fn aa_cos(a: AffineForm) -> AffineForm:
    return aa_sin(aa_add_scalar(a, π/2))
```

### 1.4 Main Algorithm: Affine-Arithmetic FK

```
Algorithm 1: AffineFK
Input:
    θ_range: [θ_min, θ_max] ∈ ℝ^d     -- body parameter range
    q_range: [q_min, q_max] ∈ ℝ^7       -- joint angle range
    link_length_fns: (θ → l₁,...,lₖ)    -- link length as function of body params
    joint_axes: [â₁,...,â₇]              -- rotation axes
Output:
    enclosure: AxisAlignedBox ⊂ ℝ³       -- bounding box of end-effector positions

1.  // Initialize affine forms for inputs
2.  for i = 1 to 7:
3.      q[i] = AffineForm {
4.          center: (q_min[i] + q_max[i]) / 2,
5.          coeffs: [0,..., (q_max[i]-q_min[i])/2 at position i, ..., 0],
6.          error: 0
7.      }
8.  for j = 1 to d:
9.      θ[j] = AffineForm {
10.         center: (θ_min[j] + θ_max[j]) / 2,
11.         coeffs: [0,..., (θ_max[j]-θ_min[j])/2 at position 7+j, ..., 0],
12.         error: 0
13.     }
14.
15. // Compute link lengths as affine forms (affine functions of θ → exact)
16. for i = 1 to k:
17.     l[i] = link_length_fns[i](θ)  // Affine combination of θ → exact affine form
18.
19. // Propagate through kinematic chain
20. // pos = Σᵢ lᵢ · (∏ⱼ₌₁ⁱ Rⱼ(qⱼ)) · û_i
21. // We track position as 3 affine forms (x, y, z)
22. pos = [AffineForm(0), AffineForm(0), AffineForm(0)]
23. R_accum = [[AffineForm(1),AffineForm(0),AffineForm(0)],  // 3×3 identity
24.            [AffineForm(0),AffineForm(1),AffineForm(0)],
25.            [AffineForm(0),AffineForm(0),AffineForm(1)]]
26.
27. for i = 1 to 7:
28.     // Compute rotation matrix for joint i
29.     c_i = aa_cos(q[i])
30.     s_i = aa_sin(q[i])
31.     R_i = rotation_matrix(joint_axes[i], c_i, s_i)  // 3×3 affine matrix
32.     
33.     // Accumulate rotation: R_accum = R_accum · R_i
34.     R_accum = aa_mat_mul(R_accum, R_i)  // 3×3 affine matrix multiply
35.     
36.     // Add link contribution: pos += R_accum · (l[i] · û_i)
37.     link_vec = aa_scale_vec(unit_vec[i], l[i])  // 3-vector of affine forms
38.     contribution = aa_mat_vec_mul(R_accum, link_vec)
39.     pos = aa_vec_add(pos, contribution)
40.
41. // Extract bounding box from affine forms
42. enclosure = AxisAlignedBox {
43.     x: [pos[0].lower(), pos[0].upper()],
44.     y: [pos[1].lower(), pos[1].upper()],
45.     z: [pos[2].lower(), pos[2].upper()]
46. }
47. return enclosure
```

### 1.5 Subdivision Strategy

```
Algorithm 1b: SubdividedAffineFK
Input:
    Same as Algorithm 1 + max_wrapping: f64, max_subdivisions: int
Output:
    enclosure: union of AxisAlignedBoxes

1.  initial = AffineFK(θ_range, q_range, ...)
2.  est_wrapping = estimate_wrapping(initial, θ_range, q_range)
3.  
4.  if est_wrapping ≤ max_wrapping or max_subdivisions ≤ 0:
5.      return {initial}
6.  
7.  // Find axis with largest contribution to wrapping
8.  // (the noise symbol with largest total coefficient magnitude across x,y,z)
9.  worst_axis = argmax_i Σⱼ |pos[j].coeffs[i]|
10. 
11. // Subdivide along worst axis
12. if worst_axis ≤ 7:  // Joint angle axis
13.     (range_lo, range_hi) = split(q_range, worst_axis)
14.     results = SubdividedAffineFK(θ_range, range_lo, ..., max_subdivisions-1)
15.              ∪ SubdividedAffineFK(θ_range, range_hi, ..., max_subdivisions-1)
16. else:  // Body parameter axis
17.     param_idx = worst_axis - 7
18.     (range_lo, range_hi) = split(θ_range, param_idx)
19.     results = SubdividedAffineFK(range_lo, q_range, ..., max_subdivisions-1)
20.              ∪ SubdividedAffineFK(range_hi, q_range, ..., max_subdivisions-1)
21. return results
```

### 1.6 Intersection Test

```
Algorithm 1c: AccessibilityCheck_Tier1
Input:
    element: (position p, activation_volume V)
    body_range: [θ_min, θ_max]
    device: (tracking_volume T, interaction_types I)
Output:
    verdict: GREEN | YELLOW | RED with population_estimate

1.  q_range = joint_limits(body_range)  // Full joint range for this body range
2.  enclosures = SubdividedAffineFK(body_range, q_range, max_wrapping=5, max_sub=6)
3.  workspace = union(enclosures) ∩ T  // Intersect with device tracking volume
4.  
5.  if V ⊂ workspace:            // Activation volume fully inside workspace
6.      return GREEN              // Definitely accessible
7.  elif V ∩ workspace = ∅:      // No intersection
8.      return RED(100%)          // Definitely inaccessible for entire range
9.  else:
10.     // Estimate excluded population fraction via interval bisection
11.     // on body parameter space
12.     excluded = estimate_excluded_fraction(element, body_range, 4)  // 4 levels
13.     return YELLOW(excluded)
```

### 1.7 Complexity Analysis

| Operation | Time | Space |
|-----------|------|-------|
| AffineFK (no subdivision) | O(k · s²) where k=joints, s=noise symbols | O(s) per affine form, O(9s) for rotation matrix |
| Single subdivision | 2× the above | Same (sequential) |
| SubdividedAffineFK (max_sub=6) | O(2⁶ · k · s²) = O(64 · 7 · s²) | O(s) (reuse) |
| Full Tier 1 (m elements) | O(m · 64 · 7 · s²) | O(s) |

With s ≈ 20 noise symbols (7 joints + 5 body params + ~8 nonlinear), k = 7:
- Per element: 64 × 7 × 400 = 179,200 operations ≈ 0.2ms on modern CPU
- 50-element scene: 50 × 0.2ms = 10ms
- **Well under 2-second budget.**

Wrapping factor (from Lemma B1): w ≈ 1.15-1.71 for 30°-60° ranges before subdivision; with 6 levels of subdivision, effective w < 1.05.

---

## 2. Algorithm 2: Adaptive Stratified Sampling

### 2.1 Main Algorithm

```
Algorithm 2: AdaptiveStratifiedSampler
Input:
    scene: Ω = (E, G, D)
    Θ_target: target parameter space
    tier1_results: {element → (verdict, frontier_estimate)} from Tier 1
    N_budget: total sample budget
    δ: confidence parameter
Output:
    S: sample set with verdicts {(θⱼ, vⱼ)}
    F̂: frontier estimate {(element, device) → frontier_region}
    L̂: Lipschitz constant estimates per stratum

Phase 1: Initial Stratification
1.  // Create initial strata from Tier 1 frontier regions
2.  strata = []
3.  for each (element, verdict) in tier1_results:
4.      if verdict == YELLOW:
5.          // Tier 1's yellow region IS the frontier estimate
6.          frontier_region = tier1_frontier(element)
7.          strata.append(Stratum(region=frontier_region, priority=HIGH))
8.      elif verdict == RED:
9.          strata.append(Stratum(region=red_region(element), priority=LOW))
10.     // GREEN regions: skip (no frontier here)
11. 
12. // Add uniform background strata covering Θ_target
13. grid_strata = uniform_grid(Θ_target, cells_per_dim=3)  // 3⁵ = 243 strata
14. for s in grid_strata:
15.     if s not covered by existing strata:
16.         strata.append(Stratum(region=s, priority=MEDIUM))

Phase 2: Initial Uniform Sampling
17. N_initial = N_budget * 0.3  // 30% budget for initial uniform pass
18. samples_per_stratum = max(100, N_initial / |strata|)
19. S = {}
20. for each stratum s in strata:
21.     points = latin_hypercube_sample(s.region, samples_per_stratum)
22.     for θ in points:
23.         v = evaluate_accessibility(scene, θ)  // FK evaluation for all elements
24.         S = S ∪ {(θ, v)}

Phase 3: Lipschitz Estimation
25. for each stratum s:
26.     frontier_samples = {(θ,v) ∈ S | θ ∈ s.region and v has mixed verdicts nearby}
27.     if |frontier_samples| ≥ 10:
28.         // Estimate L from max gradient of accessibility across nearby sample pairs
29.         L̂[s] = max over pairs (θ₁,θ₂) in frontier_samples:
30.                     |acc(θ₁) - acc(θ₂)| / ‖θ₁ - θ₂‖
31.         // L is 0 or ∞ for binary functions; instead estimate the
32.         // "effective Lipschitz constant" = inverse of minimum distance
33.         // between accessible and inaccessible samples
34.         L̂[s] = 1 / min_frontier_gap(frontier_samples)
35.     else:
36.         L̂[s] = default_L  // Conservative default

Phase 4: Adaptive Refinement
37. N_remaining = N_budget - |S|
38. // Allocate remaining samples proportional to frontier uncertainty
39. for each stratum s:
40.     uncertainty[s] = estimate_frontier_volume(s, S) * L̂[s]
41. 
42. total_uncertainty = Σ uncertainty[s]
43. for each stratum s:
44.     N_adaptive[s] = N_remaining * uncertainty[s] / total_uncertainty
45. 
46. for each stratum s:
47.     if N_adaptive[s] > 0:
48.         // Concentrate samples near the frontier
49.         frontier_center = estimate_frontier_center(s, S)
50.         frontier_radius = estimate_frontier_width(s, S) * 2  // 2× for safety
51.         sample_region = s.region ∩ Ball(frontier_center, frontier_radius)
52.         points = latin_hypercube_sample(sample_region, N_adaptive[s])
53.         for θ in points:
54.             v = evaluate_accessibility(scene, θ)
55.             S = S ∪ {(θ, v)}

Phase 5: Cross-Validation for L̂
56. // Hold out 20% of samples for cross-validation
57. (S_train, S_test) = random_split(S, 0.8, 0.2)
58. for each stratum s:
59.     // Predict test verdicts using train verdicts + L̂
60.     predictions = lipschitz_predict(S_train, L̂[s], S_test)
61.     actual = {v | (θ,v) ∈ S_test, θ ∈ s.region}
62.     error_rate = mismatch_rate(predictions, actual)
63.     if error_rate > ε / 2:  // L̂ is too aggressive
64.         L̂[s] = L̂[s] * 2  // Double the Lipschitz estimate
65.         // Re-validate; if still failing, mark stratum as "L-unstable"

66. return (S, F̂, L̂)
```

### 2.2 Latin Hypercube Sampling

```
fn latin_hypercube_sample(region: HyperRect, n: int) -> Vec<Point>:
    d = region.dimension
    result = []
    for dim = 1 to d:
        permutation[dim] = random_permutation(0..n)
    for i = 0 to n-1:
        point = []
        for dim = 1 to d:
            lo = region.lo[dim] + (permutation[dim][i] / n) * region.width[dim]
            hi = region.lo[dim] + ((permutation[dim][i]+1) / n) * region.width[dim]
            point.push(uniform_random(lo, hi))
        result.push(point)
    return result
```

### 2.3 Complexity Analysis

| Phase | Time | Space |
|-------|------|-------|
| Stratification | O(m·p) where m=elements, p=devices | O(|strata|) |
| Initial sampling | O(N_initial · m · FK_cost) | O(N_initial · m) for verdicts |
| Lipschitz estimation | O(|S|² · d) per stratum (pairwise distances) | O(|strata|) |
| Adaptive refinement | O(N_remaining · m · FK_cost) | O(N_remaining · m) |
| Cross-validation | O(|S|²) | O(|S|) |

FK_cost = O(k) ≈ O(7) for a single evaluation via Pinocchio (microseconds).

With N_budget = 4M samples, m = 30 elements: 4M × 30 × 7 ≈ 840M operations ≈ 2-3 seconds. **Well within 10-minute budget.**

---

## 3. Algorithm 3: Linearized-Kinematics SMT Encoding

### 3.1 Main Algorithm

```
Algorithm 3: LinearizedSMTVerification
Input:
    θ₀, q₀: reference configuration (sample point)
    Δ_max: soundness envelope radius (from Theorem C2)
    element: (position p, activation_volume V)
    element_idx, device_idx: identifiers
Output:
    verdict: VERIFIED_ACCESSIBLE | VERIFIED_INACCESSIBLE | UNKNOWN
    proof: SMT proof object (if verified)

1.  // Step 1: Compute Jacobians at reference point
2.  J_θ = compute_body_jacobian(θ₀, q₀)    // ∂FK/∂θ ∈ ℝ³ˣ⁵
3.  J_q = compute_joint_jacobian(θ₀, q₀)    // ∂FK/∂q ∈ ℝ³ˣ⁷
4.  p₀ = FK_position(θ₀, q₀)               // Reference end-effector position
5.  
6.  // Step 2: Compute soundness envelope
7.  // From C2: ‖FK - FK_lin‖ ≤ C_FK · (Δ_q² + Δ_θ·Δ_q) · L_sum
8.  η = activation_volume_margin(V) * 0.5   // Error tolerance = half the margin
9.  Δ_q = sqrt(η / (2 * C_FK * L_sum))      // Max joint angle deviation
10. Δ_θ = η / (2 * C_FK * L_sum * Δ_q)      // Max body param deviation
11. Δ_max = min(Δ_q, Δ_θ)
12. 
13. // Step 3: Build SMT formula
14. // Variables: δθ ∈ ℝ⁵ (body param deviation), δq ∈ ℝ⁷ (joint angle deviation)
15. formula = new SMTFormula(solver=Z3, logic=QF_LRA)
16. 
17. // Declare variables
18. δθ = [formula.declare_real(f"dtheta_{i}") for i in 1..5]
19. δq = [formula.declare_real(f"dq_{j}") for j in 1..7]
20. 
21. // Bound variables to soundness envelope
22. for i in 1..5:
23.     formula.assert(-Δ_θ ≤ δθ[i] ≤ Δ_θ)
24. for j in 1..7:
25.     formula.assert(-Δ_q ≤ δq[j] ≤ Δ_q)
26.     // Also enforce joint limits
27.     formula.assert(q₀[j] + δq[j] ≥ q_min[j](θ₀))
28.     formula.assert(q₀[j] + δq[j] ≤ q_max[j](θ₀))
29.
30. // Linearized position: p_lin = p₀ + J_θ · δθ + J_q · δq
31. p_lin = [p₀[k] + sum(J_θ[k,i]*δθ[i] for i) + sum(J_q[k,j]*δq[j] for j)
32.          for k in {x,y,z}]
33. 
34. // Step 4: Check accessibility — does p_lin ∈ V (with η margin for soundness)?
35. // Expand V by η to account for linearization error
36. V_expanded = expand(V, η)
37. 
38. // Query 1: Is there a configuration in the envelope that reaches V?
39. reachability = formula.check_sat(p_lin ∈ V_expanded)
40.
41. if reachability == UNSAT:
42.     // No configuration in envelope can reach V → VERIFIED_INACCESSIBLE
43.     // (for body params in [θ₀-Δ_θ, θ₀+Δ_θ])
44.     return (VERIFIED_INACCESSIBLE, formula.proof())
45. 
46. // Query 2: Does EVERY configuration reach V? (negate: is there one that doesn't?)
47. non_reach = formula.check_sat(p_lin ∉ V_expanded)
48.
49. if non_reach == UNSAT:
50.     // Every configuration reaches V → VERIFIED_ACCESSIBLE
51.     // (need to shrink V by η instead of expanding for this direction)
52.     // Actually: re-check with V_shrunk = shrink(V, η)
53.     V_shrunk = shrink(V, η)
54.     universal_reach = formula.check_sat(p_lin ∉ V_shrunk)
55.     if universal_reach == UNSAT:
56.         return (VERIFIED_ACCESSIBLE, formula.proof())
57. 
58. return (UNKNOWN, null)
```

### 3.2 SMT-LIB Generation for Activation Volume

```
fn encode_activation_volume(V: ActivationVolume, p_lin: [Expr;3]) -> SMTExpr:
    match V.shape:
        AxisAlignedBox(lo, hi):
            return AND(
                lo.x ≤ p_lin[0] ≤ hi.x,
                lo.y ≤ p_lin[1] ≤ hi.y,
                lo.z ≤ p_lin[2] ≤ hi.z
            )  // Linear constraints → QF_LRA
        
        Sphere(center, radius):
            // (p - c)² ≤ r² is quadratic, not linear!
            // Over-approximate with inscribed cube for QF_LRA
            r_inner = radius / sqrt(3)
            return AND(
                |p_lin[0] - center.x| ≤ r_inner,
                |p_lin[1] - center.y| ≤ r_inner,
                |p_lin[2] - center.z| ≤ r_inner
            )
        
        ConvexHull(vertices):
            // Express as intersection of half-planes (QF_LRA compatible)
            constraints = compute_halfplane_representation(vertices)
            return AND(c.normal · p_lin ≤ c.offset for c in constraints)
```

### 3.3 Timeout and Skip

```
fn smt_with_timeout(formula: SMTFormula, timeout_ms: int = 2000) -> Result:
    Z3.set_timeout(timeout_ms)
    result = Z3.check_sat(formula)
    if result == TIMEOUT:
        return UNKNOWN  // Region stays unverified; certificate ε adjusts
    return result
```

### 3.4 Complexity Analysis

| Component | Time | Notes |
|-----------|------|-------|
| Jacobian computation | O(k · d) = O(35) | Analytical Jacobian, not finite differences |
| SMT formula construction | O(k + d) = O(12) | Linear in variable count |
| Z3 solving (QF_LRA) | O(?) empirically ~10-100ms | Simplex-based; polynomial in variable count for feasible instances |
| Timeout handling | O(1) | Hard cutoff at 2s |
| Per-query total | ~100ms average | With 2s timeout, median ~50ms |

With 10-minute budget and ~100ms per query: ~6,000 queries possible.
Each query covers a hypercube of radius Δ_max in 12D space (5 body + 7 joint).

**Volume per query:** For Δ_q ≈ 2.6° ≈ 0.045 rad, Δ_θ ≈ 0.045 (normalized):
vol_query / vol_Θ ≈ (0.045/π)⁷ × (0.045)⁵ ≈ 10⁻¹² per query.
6,000 queries × 10⁻¹² = 6 × 10⁻⁹ ≈ negligible volumetric coverage.

**This confirms the Red-Team's concern:** SMT volume coverage is negligible. The value of SMT must come from frontier resolution, not volume elimination (see Algorithm 5).

---

## 4. Algorithm 4: Coverage Certificate Assembly

### 4.1 Main Algorithm

```
Algorithm 4: AssembleCertificate
Input:
    S: sample set {(θⱼ, vⱼ)} with |S| = N_s
    V: verified regions {(Rᵢ, proofᵢ, verdictᵢ)} with |V| = N_v
    L̂: Lipschitz estimates per stratum
    strata: stratification of Θ_target
    scene: Ω = (E, G, D)
    δ: target confidence
Output:
    certificate: ⟨S, V, U, ε, δ⟩

Phase 1: Detect Lipschitz Violations
1.  U = []  // Unverified violation surfaces
2.  for each element e in E:
3.      for each joint j in 1..7:
4.          // Find critical angle where joint limit = required angle
5.          q_crit = critical_angle(e, j)  // Angle needed to reach element
6.          if q_crit is not None:
7.              // Joint limit surface: {θ | q_j_max(θ) = q_crit}
8.              // This is a hyperplane in Θ (since q_j_max is affine in θ)
9.              H = {θ | q_j_max(θ) = q_crit}
10.             ε_nbhd = compute_neighborhood_width(H, L̂)
11.             μ_nbhd = estimate_measure(neighborhood(H, ε_nbhd), Θ_target)
12.             U.append((H, μ_nbhd))

Phase 2: Partition Smooth Region
13. Θ_smooth = Θ_target
14. for (H, μ) in U:
15.     Θ_smooth = Θ_smooth \ neighborhood(H, ε_nbhd)
16. κ = 1 - vol(Θ_smooth) / vol(Θ_target)  // Excluded fraction

Phase 3: Compute ε per Stratum
17. K = |E| × |D| × |strata|  // Total Bonferroni count
18. for each stratum s in strata:
19.     s_smooth = s ∩ Θ_smooth
20.     // Count samples in smooth region of this stratum
21.     n_s = |{(θ,v) ∈ S | θ ∈ s_smooth}|
22.     // Compute verified volume in this stratum
23.     vol_verified_s = Σ{vol(Rᵢ) | Rᵢ ⊆ s_smooth}
24.     ρ_s = 1 - vol_verified_s / vol(s_smooth)  // Unverified fraction
25.     
26.     if n_s == 0:
27.         ε_s = 1.0  // No samples → no guarantee
28.     else:
29.         // Hoeffding bound for unverified region
30.         ε_s = ρ_s × sqrt(ln(2 × K / δ) / (2 × n_s))

Phase 4: Compute Global ε
31. ε = max(ε_s for s in strata)

Phase 5: Frontier-Resolution Enhancement
32. // For SMT queries near the frontier, credit their information value
33. // beyond pure volume elimination
34. for each verified region Rᵢ at the frontier:
35.     // This SMT query resolved the boundary location within Rᵢ
36.     // The Lipschitz interpolation from this resolved point
37.     // covers a neighborhood of radius ~ Δ_max / L̂
38.     effective_radius = Δ_max / max(L̂[containing_stratum(Rᵢ)], 1)
39.     effective_volume = vol(Ball(center(Rᵢ), effective_radius))
40.     // Credit this to verified volume
41.     vol_verified_s += effective_volume  // for containing stratum

42. // Recompute ε with enhanced volumes
43. for each stratum s:
44.     ρ_s_enhanced = 1 - vol_verified_s_enhanced / vol(s_smooth)
45.     ε_s_enhanced = ρ_s_enhanced × sqrt(ln(2 × K / δ) / (2 × n_s))
46. ε_enhanced = max(ε_s_enhanced for s in strata)

Phase 6: Assemble Certificate
47. certificate = {
48.     S: S,
49.     V: V,
50.     U: U,
51.     ε: min(ε, ε_enhanced),
52.     δ: δ,
53.     κ: κ,                    // Excluded fraction
54.     strata_detail: {s → (ε_s, n_s, ρ_s) for s in strata},
55.     metadata: {scene_hash, timestamp, tool_version}
56. }
57. return certificate
```

### 4.2 Certificate Verification (Standalone Checker)

```
Algorithm 4b: VerifyCertificate
Input:
    certificate: ⟨S, V, U, ε, δ⟩
    scene: Ω
Output:
    valid: bool

1.  // Recompute ε from scratch using the certificate's data
2.  ε_recomputed = recompute_epsilon(certificate.S, certificate.V,
3.                                     certificate.strata_detail, δ)
4.  
5.  // Verify SMT proofs
6.  for (Rᵢ, proofᵢ, verdictᵢ) in V:
7.      if not Z3.verify_proof(proofᵢ):
8.          return false
9.  
10. // Spot-check samples (re-evaluate 5% of samples)
11. spot_check = random_sample(S, 0.05)
12. for (θ, v) in spot_check:
13.     v_recomputed = evaluate_accessibility(scene, θ)
14.     if v ≠ v_recomputed:
15.         return false
16. 
17. // Verify ε is correctly computed
18. if |ε_recomputed - certificate.ε| > 1e-6:
19.     return false
20.
21. return true
```

### 4.3 Complexity Analysis

| Phase | Time | Space |
|-------|------|-------|
| Lipschitz violation detection | O(m · n · d) | O(m · n) surfaces |
| Volume computation | O(|strata| · |V|) | O(|strata|) |
| ε computation | O(|strata|) | O(|strata|) |
| Frontier enhancement | O(|V|) | O(|strata|) |
| Certificate verification | O(|V| · proof_check + 0.05 · |S| · m · FK) | O(|S|) |

Total assembly: O(|strata| · |V| + m·n·d) ≈ milliseconds. **Not a bottleneck.**

---

## 5. Algorithm 5: Budget Allocation

### 5.1 Main Algorithm

```
Algorithm 5: OptimalBudgetAllocation
Input:
    T: total time budget (seconds)
    tier1_results: Tier 1 frontier estimates
    N_elements: number of scene elements
    d: parameter space dimension (typically 5 body + 7 joint = 12 for SMT, 5 for sampling)
    t_fk: FK evaluation time (~50μs)
    t_smt: mean SMT query time (~100ms)
    δ: confidence parameter
Output:
    allocation: (N_samples, N_smt, smt_priorities)

1.  // Estimate frontier characteristics from Tier 1
2.  N_frontier_elements = count(e in tier1_results where verdict == YELLOW)
3.  frontier_fraction = N_frontier_elements / N_elements
4.  
5.  // Binary search for optimal split
6.  best_ε = 1.0
7.  best_split = (T / t_fk, 0)  // Default: all sampling
8.  
9.  for α in [0.0, 0.01, 0.02, ..., 0.50]:  // SMT fraction of budget
10.     T_smt = α * T
11.     T_sample = (1 - α) * T
12.     
13.     N_samples = floor(T_sample / t_fk)
14.     N_smt = floor(T_smt / t_smt)
15.     
16.     // Estimate ε from this allocation
17.     N_strata = 3^5  // 243 strata (3 per body-param dimension)
18.     n_per_stratum = N_samples / N_strata
19.     
20.     // Volume-based SMT credit (small)
21.     vol_credit = N_smt * vol_per_query(d)
22.     
23.     // Frontier-resolution SMT credit (large)
24.     // Each frontier SMT query resolves ~Δ_resolved of boundary
25.     // reducing effective frontier uncertainty
26.     Δ_frontier = estimate_frontier_length(frontier_fraction, d)
27.     Δ_resolved_per_query = 2 * Δ_max  // Each query resolves 2Δ of frontier
28.     resolution_factor = max(0, 1 - N_smt * Δ_resolved_per_query / Δ_frontier)
29.     
30.     ρ_effective = (1 - vol_credit) * resolution_factor
31.     ε_estimate = ρ_effective * sqrt(ln(2 * N_strata * N_elements * |D| / δ)
32.                                     / (2 * n_per_stratum))
33.     
34.     if ε_estimate < best_ε:
35.         best_ε = ε_estimate
36.         best_split = (N_samples, N_smt)
37. 
38. // Compute SMT priorities
39. smt_priorities = []
40. for each element e with YELLOW verdict:
41.     for each stratum s overlapping the frontier of e:
42.         // Priority = frontier uncertainty in this stratum
43.         // Higher uncertainty → higher priority
44.         priority = frontier_volume_estimate(e, s) * L̂[s]
45.         smt_priorities.append((e, s, priority))
46. sort smt_priorities by priority descending
47.
48. return (best_split[0], best_split[1], smt_priorities[:best_split[1]])
```

### 5.2 Complexity

O(50 × |strata|) for the grid search. Negligible compared to actual sampling/SMT.

---

## 6. Algorithm 6: Multi-Step Interaction Verification

### 6.1 Main Algorithm

```
Algorithm 6: MultiStepVerification
Input:
    π = (e₁, e₂, ..., eₖ) where k ≤ 3: interaction sequence
    θ: body parameters
    d: device
Output:
    accessible: bool

1.  if k == 1:
2.      return single_step_accessible(e₁, θ, d)
3.  
4.  // Find joint configurations for each step
5.  // q₁ must reach e₁, q₂ must reach e₂, ..., qₖ must reach eₖ
6.  // AND the transition q_i → q_{i+1} must be feasible (within joint limits)
7.  
8.  // Strategy: sample q₁ from IK solutions for e₁, then check forward
9.  Q₁ = sample_ik_solutions(e₁, θ, d, n=100)  // Sample configs reaching e₁
10. 
11. for q₁ in Q₁:
12.     Q₂ = sample_ik_solutions(e₂, θ, d, n=100)
13.     for q₂ in Q₂:
14.         // Check feasibility: linear path from q₁ to q₂ within J(θ)?
15.         if linear_path_feasible(q₁, q₂, J(θ)):
16.             if k == 2:
17.                 return true
18.             Q₃ = sample_ik_solutions(e₃, θ, d, n=100)
19.             for q₃ in Q₃:
20.                 if linear_path_feasible(q₂, q₃, J(θ)):
21.                     return true
22. 
23. return false  // No feasible sequence found (may be false negative for sampling)

// Note: This is a sampling-based check, not exact.
// For the coverage certificate, multi-step verdicts are obtained by
// treating the k-step problem as a higher-dimensional single-step problem
// over the product space T^(k·7) with constraints.
```

### 6.2 Multi-Step Certificate Extension

For multi-step interactions, the parameter space for the certificate expands:
- Single-step: Θ (d=5 body params) → sample in ℝ⁵
- 2-step: Θ × T⁷ × T⁷ (d=5+7+7=19) → sample in ℝ¹⁹
- 3-step: Θ × T⁷ × T⁷ × T⁷ (d=5+7+7+7=26) → sample in ℝ²⁶

The curse of dimensionality:
- At d=5 with 4M samples: ~22 samples per dimension → good coverage
- At d=19 with 4M samples: ~3.5 samples per dimension → sparse
- At d=26 with 4M samples: ~2.7 samples per dimension → very sparse

This is why the ε target relaxes: <0.01 for single-step → <0.1 for 3-step.

### 6.3 Complexity

| Steps k | Trajectory dimension | Samples for ε=0.1 | Time at 50μs/sample |
|---------|---------------------|-------------------|---------------------|
| 1 | 5 | ~50K | 2.5s |
| 2 | 19 | ~500K | 25s |
| 3 | 26 | ~2M | 100s |

All within the 10-minute budget for Tier 2.

---

## 7. End-to-End Pipeline

```
Algorithm 7: FullPipeline
Input:
    unity_scene: path to .unity file
    target_population: Θ_target
    target_devices: D
    tier: 1 | 2
    budget: time in seconds (2 for Tier 1, 600 for Tier 2)
Output:
    report: AccessibilityReport

// ===== Stage 1: Scene Parsing =====
1.  scene_graph = parse_unity_scene(unity_scene)     // Unity YAML → IR
2.  elements = extract_interactables(scene_graph)     // Pattern-match XRI idioms
3.  dep_graph = extract_dependencies(scene_graph)     // Build interaction graph G
4.  Ω = (elements, dep_graph, target_devices)

// ===== Stage 2: Tier 1 — Affine-Arithmetic Linter =====
5.  tier1_results = {}
6.  for each element e in elements:
7.      for each device d in D:
8.          verdict = AccessibilityCheck_Tier1(e, target_population, d)
9.          tier1_results[(e,d)] = verdict
10. tier1_report = format_tier1_report(tier1_results)

11. if tier == 1:
12.     return tier1_report  // Done for Tier 1

// ===== Stage 3: Tier 2 — Sampling + SMT + Certificate =====

// 3a. Budget allocation
13. allocation = OptimalBudgetAllocation(budget, tier1_results, |elements|, ...)

// 3b. Adaptive stratified sampling
14. (S, F̂, L̂) = AdaptiveStratifiedSampler(Ω, target_population,
15.                                          tier1_results, allocation.N_samples, δ=0.01)

// 3c. Targeted SMT verification
16. verified_regions = []
17. for (element, stratum, priority) in allocation.smt_priorities:
18.     // Pick sample near frontier in this stratum
19.     frontier_sample = nearest_frontier_sample(S, element, stratum)
20.     (θ₀, q₀) = frontier_sample
21.     result = LinearizedSMTVerification(θ₀, q₀, Δ_max, element)
22.     if result.verdict ≠ UNKNOWN:
23.         verified_regions.append((result.region, result.proof, result.verdict))
24.     if time_elapsed() > budget * 0.9:
25.         break  // Reserve 10% budget for certificate assembly

// 3d. Multi-step verification (for sequences in dep_graph)
26. sequences = enumerate_sequences(dep_graph, max_length=3)
27. for π in sequences:
28.     // Multi-step uses sampling from S (already computed)
29.     multistep_verdicts[π] = assess_multistep(π, S)

// 3e. Certificate assembly
30. certificate = AssembleCertificate(S, verified_regions, L̂, strata, Ω, δ=0.01)

// 3f. Report
31. report = {
32.     tier1: tier1_report,
33.     certificate: certificate,
34.     per_element: {e → (verdict, confidence, details) for e in elements},
35.     multistep: multistep_verdicts,
36.     population_exclusions: certificate.U,
37.     runtime: time_elapsed()
38. }
39. return report
```

---

## 8. Complexity Summary Table

| Algorithm | Time (worst case) | Time (expected) | Space | Bottleneck |
|-----------|-------------------|-----------------|-------|------------|
| **1. AffineFK** | O(2^s · k · n²) s=subdivisions, k=joints, n=noise symbols | O(m · 64 · 7 · 400) ≈ 10ms for 50 elements | O(n) per eval | Matrix multiply in affine arithmetic |
| **2. Adaptive Sampling** | O(N · m · k) N=samples, m=elements, k=joints | O(4M · 30 · 7) ≈ 3s | O(N · m) verdicts | FK evaluation (Pinocchio) |
| **3. SMT Encoding** | O(timeout) per query = 2s worst case | O(100ms) per query | O(d²) per formula | Z3 solver |
| **4. Certificate Assembly** | O(|strata| · |V| + m·n·d) | ~10ms | O(|strata| + |V|) | Volume computation |
| **5. Budget Allocation** | O(50 · |strata|) | ~1ms | O(|strata|) | Grid search |
| **6. Multi-Step** | O(IK_samples^k) | ~100s for k=3 | O(IK_samples · k) | IK sampling |
| **7. Full Pipeline** | O(N·m·k + N_smt·timeout) | ~5min for 30-element scene | O(N·m) | Sampling + SMT |

---

## 9. Implementation Mapping

| Algorithm | Language | Key Libraries | Est. LoC | Risk |
|-----------|----------|---------------|----------|------|
| 1. AffineFK | C++ | Custom affine arithmetic library | 5-8K | Medium: wrapping factor may exceed target |
| 1b. Subdivision | C++ | Same | Included above | Low |
| 1c. Tier 1 Check | C++ + C# (Unity bridge) | Unity Editor API | 3-5K | Low: engineering, not algorithmic |
| 2. Adaptive Sampling | Python + C++ (FK) | Pinocchio (FK), NumPy | 5-8K | Low: well-understood techniques |
| 3. SMT Encoding | Python | Z3 (pyz3), custom linearization | 5-8K | Medium: query time variability |
| 4. Certificate Assembly | Python | NumPy, custom | 6-10K | Low: straightforward computation |
| 5. Budget Allocation | Python | SciPy optimize | 1-2K | Low: simple optimization |
| 6. Multi-Step | Python + C++ | Pinocchio (IK) | 3-5K | Medium: IK sampling coverage |
| 7. Pipeline Orchestrator | Python | All of the above | 3-5K | Low: integration |
| **Total** | | | **31-51K** | |

### Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Wrapping factor > 5× on 7-DOF | 25% | High (Tier 1 useless) | Subdivision + Taylor models fallback |
| SMT timeout rate > 50% | 20% | Medium (ε loosens) | Aggressive linearization; reduce Δ_max |
| ε > 0.05 on typical scenes | 40% | High (certificate weak) | Frontier-resolution model; increase budget |
| Multi-step ε > 0.2 at k=3 | 35% | Medium (restrict to k≤2) | Focus on single-step; defer multi-step |
| Pinocchio integration issues | 15% | Low (alternatives exist) | KDL as fallback library |
