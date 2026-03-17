# Empirical Evaluation Proposal: Coverage-Certified XR Accessibility Verifier

**Author role:** Empirical Scientist
**Date:** 2026-03-08
**Status:** Theory-stage proposal — all experiments designed for fully automated execution

---

## 0. Notation and Conventions

| Symbol | Meaning |
|--------|---------|
| Θ | Full anthropometric × joint-angle parameter space |
| S ⊂ Θ | Sampled region with pointwise verdicts |
| V ⊂ Θ | Symbolically (SMT) verified region |
| ε | Certificate upper bound on P(undetected bug in Θ \ (S ∪ V)) |
| δ | Confidence level: P(ε bound holds) ≥ 1 − δ |
| L | Lipschitz constant of the accessibility frontier |
| w | Affine-arithmetic wrapping factor |
| CP(n, k, α) | Clopper-Pearson upper bound: n samples, k failures, confidence 1−α |
| FK(θ, b) | Forward kinematics: joint angles θ, body params b → end-effector pose |
| TPR | True positive rate (detection rate) |
| FPR | False positive rate |
| d | Kinematic chain DOF (default 7: 3-shoulder, 1-elbow, 3-wrist) |

All experiments target **Apple M-series laptop, 16 GB RAM, no GPU**. All random
procedures use explicit seeds recorded in experiment manifests. All statistical tests
use α = 0.05 unless stated otherwise.

---

## 1. Experimental Hypotheses (Falsifiable)

Each hypothesis corresponds to a project kill gate. We state null (H0) and alternative
(H1), the statistical test, sample size rationale, and the concrete failure criterion.

### H1: Coverage Certificate ε Improvement (Gate D3)

- **H0_1:** The coverage certificate ε is no better than the Clopper-Pearson 99% upper
  bound from the same sample count. Formally: E[ε_cert / CP(n, k, 0.01)] ≥ 1/3 across
  benchmark scenes.
- **H1_1:** ε_cert ≤ (1/5) × CP(n, k, 0.01) on ≥80% of benchmark scenes.
- **Test:** Paired one-sided Wilcoxon signed-rank test on ε_cert / CP ratios across N ≥ 50
  scenes. Null rejected if p < 0.05 and median ratio ≤ 0.20.
- **Sample:** 100 procedural scenes (20 per object-count level), each run at 3 sample
  budgets (10K, 50K, 200K samples). Total 300 paired observations.
- **Failure criterion:** If median ratio > 0.20, or if fewer than 80% of scenes achieve
  ratio ≤ 0.20, gate D3 fails. Downscope to tool-only paper.
- **Effect size justification:** A 5× improvement means the certificate's SMT-verified
  volume eliminates ≥80% of the statistical uncertainty. At 30% SMT coverage, the
  effective sample space shrinks to 70%, yielding ~1.4× tightening from volume alone; the
  5× target requires that adaptive sampling near the frontier further concentrates
  information. The 5× threshold distinguishes "novel formal contribution" from "modest
  engineering improvement."

### H2: Affine-Arithmetic Wrapping Factor (Gate D1)

- **H0_2:** The wrapping factor w of affine-arithmetic FK on a 4-joint revolute chain with
  ±30° joint ranges exceeds 5×.
- **H1_2:** w ≤ 5× on ≥95% of sampled configurations.
- **Test:** One-sample one-sided t-test on log(w) across 1000 random chain geometries
  (link lengths drawn from ANSUR-II arm-segment distributions). Null rejected if mean
  log(w) < log(5) at α = 0.05.
- **Procedure:** For each chain geometry, compute:
  1. Ground truth: exact reachable workspace volume via dense (10M-point) FK sampling.
  2. Affine-arithmetic enclosure volume via noise-symbol propagation.
  3. w = vol(enclosure) / vol(ground_truth).
- **Failure criterion:** If mean w > 5× or if >5% of geometries yield w > 10×, gate D1
  fails. Switch to Taylor-model propagation. If Taylor models also exceed 10×, abandon
  linter; fall back to lookup table.

### H3: Tier 1 Detection Rate and False Positive Rate

- **H0_3a:** Tier 1 detection rate ≤ 95% across all bug categories.
- **H1_3a:** TPR > 95% (one-sided binomial test, n ≥ 400 injected bugs).
- **H0_3b:** Tier 1 false positive rate ≥ 15%.
- **H1_3b:** FPR < 15% (one-sided binomial test, n ≥ 600 accessible elements).
- **Test:** Exact binomial tests. For TPR: reject H0 if observed TPR exceeds 95% with
  p < 0.05. For FPR: reject H0 if observed FPR is below 15% with p < 0.05.
- **Sample size rationale:** 400 bugs gives 80% power to detect TPR = 97% vs. H0 of 95%
  (two-percentage-point effect). 600 elements gives 80% power to detect FPR = 12% vs.
  H0 of 15%.
- **Failure criterion:** If TPR ≤ 95% or FPR ≥ 15% after Bonferroni correction for 8 bug
  categories, Tier 1 quality is insufficient for the UIST paper's "practical linter"
  claim.

### H4: Tier 2 Detection Rate and False Positive Rate

- **H0_4a:** Tier 2 detection rate ≤ 97%.
- **H1_4a:** TPR > 97%.
- **H0_4b:** Tier 2 false positive rate ≥ 5%.
- **H1_4b:** FPR < 5%.
- **Test:** Same binomial framework as H3. Sample: ≥400 bugs, ≥1000 accessible elements.
- **Failure criterion:** If TPR ≤ 97% or FPR ≥ 5%, the certificate-backed tier does not
  meaningfully improve over Tier 1, weakening the CAV paper.

### H5: Marginal Detection over Monte Carlo (Gate A3)

- **H0_5:** Formal verification (Tier 2) detects < 10% of bugs missed by 1M-sample
  stratified Monte Carlo.
- **H1_5:** Marginal detection rate ≥ 10%.
- **Test:** One-sided exact binomial test on the count of MC-missed bugs detected by Tier 2,
  out of total MC-missed bugs.
- **Procedure:**
  1. Run MC baseline (1M samples) on full benchmark suite. Record set B_MC of detected bugs.
  2. Run Tier 2 on same suite. Record set B_T2 of detected bugs.
  3. Compute marginal set: M = B_T2 \ B_MC (bugs found by Tier 2 but missed by MC).
  4. Marginal rate = |M| / |B_all \ B_MC| where B_all is full injected bug set.
- **Failure criterion:** If marginal rate < 10%, gate A3 fails. Formal verification does
  not justify its cost. If marginal rate 5–10%, soft fail: reframe as certification.
- **Expected MC-missed bugs:** Subtle anthropometric-boundary bugs (category 6), sequential
  traps (category 5), and bimanual impossibilities (category 8) are hardest for MC. With
  40 bug templates × 10 scenes per template = 400 bugs, we expect MC to miss 20–60
  (5–15%), giving 20–60 bugs in the denominator — sufficient for a binomial test.

### H6: Multi-Step Certificate Tightness (Gate D5)

- **H0_6:** Multi-step (≤3) coverage certificates cannot achieve ε < 0.1 within a
  15-minute budget on benchmark scenes.
- **H1_6:** ε < 0.1 on ≥75% of ≤3-step benchmark scenes within 15 minutes.
- **Test:** One-sided binomial test on the proportion of scenes achieving ε < 0.1.
- **Sample:** 60 scenes with 2-step interactions, 60 with 3-step interactions.
- **Failure criterion:** If <75% of scenes achieve ε < 0.1, gate D5 fails. Restrict to
  single-step certificates in the paper.

---

## 2. Benchmark Suite Design

### 2.1 Procedural Scene Generator

The generator is a deterministic Python program parameterized by a JSON manifest.
Every scene is reproducible from (manifest, seed) alone.

**Parameterization Axes (5 axes, independently controllable):**

| Axis | Levels | Values |
|------|--------|--------|
| Object count | 5 | 5, 10, 30, 50, 100 |
| Spatial distribution | 3 | clustered, uniform, adversarial |
| Interaction depth | 3 | 1-step, 2-step, 3-step |
| Anthropometric sensitivity | 3 | low, medium, high |
| Device mode | 3 | hand-tracking, wand-controller, seated |

**Scene Generation Algorithm:**

```
generate_scene(seed, n_objects, distribution, depth, sensitivity, device):
  rng = RandomState(seed)

  # 1. Place user origin (standing or seated)
  user_origin = (0, 0, 0) if device != "seated" else (0, -0.45, 0)

  # 2. Generate object positions
  if distribution == "uniform":
    positions = uniform_sphere_sample(rng, n_objects,
                                       r_min=0.2, r_max=1.5)
  elif distribution == "clustered":
    n_clusters = max(1, n_objects // 5)
    centers = uniform_sphere_sample(rng, n_clusters, r_min=0.3, r_max=1.2)
    positions = gaussian_cluster(rng, centers, sigma=0.1, n_objects)
  elif distribution == "adversarial":
    # Place objects near reachability boundary for target percentile
    boundary_r = ansur_reach_percentile(sensitivity_to_pct(sensitivity))
    positions = annular_sample(rng, n_objects,
                                r_inner=boundary_r - 0.05,
                                r_outer=boundary_r + 0.05)

  # 3. Assign interaction types based on depth
  if depth == 1:
    interactions = [SingleReach(pos) for pos in positions]
  elif depth == 2:
    pairs = rng.partition(positions, group_size=2)
    interactions = [SequentialReach(pair) for pair in pairs]
  elif depth == 3:
    triples = rng.partition(positions, group_size=3)
    interactions = [SequentialReach(triple) for triple in triples]

  # 4. Adjust sensitivity
  #    low:    all objects within 50th-percentile reach
  #    medium: 20% of objects within 10% of 25th-percentile boundary
  #    high:   40% of objects within 2% of 5th-percentile boundary
  adjust_for_sensitivity(interactions, sensitivity, rng)

  # 5. Assign activation volumes (sphere r=0.03–0.08m)
  for elem in interactions:
    elem.activation_radius = rng.uniform(0.03, 0.08)

  # 6. Generate DSL annotation file
  emit_scene_yaml(interactions, device)
  emit_dsl_annotations(interactions)

  return Scene(interactions, metadata={seed, params})
```

**Sensitivity-to-percentile mapping:**

| Sensitivity | Target percentile | Boundary tolerance |
|-------------|-------------------|--------------------|
| low | 50th | ±10 cm |
| medium | 25th | ±3 cm |
| high | 5th | ±1 cm |

**Full factorial coverage:** 5 × 3 × 3 × 3 × 3 = 405 configurations. We generate
2 seeds per configuration → **810 procedural scenes** total.

### 2.2 Bug Injection

**40 bug templates** (8 categories × 5 severity levels):

| ID | Category | Severity 1 (Obvious) | Severity 5 (Subtle) |
|----|----------|---------------------|---------------------|
| B1 | Unreachable elements | Button at 2.5m height (beyond 99th pct) | Button at exact 5th-pct reach boundary (within 1cm) |
| B2 | Occlusion deadlocks | A fully behind B, B requires A | A partially occluded by B at specific pose angles |
| B3 | Pose-impossible gestures | Simultaneous reach: 1m left + 1m right | Simultaneous reach requiring 178° shoulder sum (limit 180°) |
| B4 | Cross-device failures | Pinch gesture on wand-only device | Wrist-rotation gesture exceeding wand tracking cone by 5° |
| B5 | Sequential traps | Step 1 locks arm, step 2 requires same arm | Step 1 positions body such that step 3 is 2cm out of reach |
| B6 | Anthropometric exclusion | Reachable at 50th pct, 15cm beyond 5th pct | Reachable at 50th pct, 0.5cm beyond 5th pct |
| B7 | Seated-mode failures | Object at 2.2m (standing head height) | Object requiring 5° more forward lean than seated ROM allows |
| B8 | Bimanual impossibilities | Two hands must be 3m apart | Two hands must be 2cm farther apart than 5th-pct shoulder width |

**Injection procedure:**

```
inject_bugs(scene, bug_templates, n_bugs, seed):
  rng = RandomState(seed)
  selected = rng.choice(bug_templates, n_bugs, replace=True)
  for template in selected:
    # Replace a random accessible element with the bugged version
    target_idx = rng.randint(0, len(scene.elements))
    original = scene.elements[target_idx]
    bugged = template.instantiate(original, scene.user_origin, rng)
    scene.elements[target_idx] = bugged
    scene.ground_truth[target_idx] = BugLabel(
      category=template.category,
      severity=template.severity,
      affected_percentiles=template.compute_affected_range(),
      description=template.describe()
    )
  return scene
```

**Ground truth guarantee:** Each bug template computes the exact affected percentile
range analytically:
- B1/B6: Solve `||FK(θ*, b) - target|| = 0` for body param b; the failure set is
  `{b : max_θ ||FK(θ,b) - target|| > activation_radius}`, computed via bisection on the
  anthropometric dimension with verified bounds.
- B2: Occlusion computed geometrically from element positions and body pose; deadlock
  verified by checking the interaction-graph cycle.
- B3/B8: Joint-limit constraint satisfaction checked algebraically.
- B4: Device capability flags are discrete; ground truth is exact.
- B5: Sequential feasibility checked by composing FK reachability across steps.
- B7: Seated ROM limits applied as hard constraints; boundary computed analytically.

Each bug template includes a `verify_ground_truth(scene) → bool` method that
independently confirms the bug exists using 10M-sample brute force.

### 2.3 Real Scene Corpus (≥10 scenes)

**Selection criteria:**
1. Open-source with permissive license (MIT, Apache 2.0, CC-BY).
2. Contains ≥10 interactable elements.
3. Uses Unity XR Interaction Toolkit or recognizable interaction patterns.
4. Represents a distinct interaction domain (menu, control panel, room-scale, etc.).

**Candidate scenes (5 open-source):**

| ID | Source | Description | Elements | Interaction depth |
|----|--------|-------------|----------|-------------------|
| R1 | Unity XR Interaction Toolkit Starter Assets | VR hand interaction demo | ~15 | 1–2 step |
| R2 | Unity VR Template Project | Room-scale VR experience | ~20 | 1 step |
| R3 | XR Interaction Toolkit Examples (grab, poke, teleport) | Interaction sampler | ~25 | 1–2 step |
| R4 | Open Brush (Tilt Brush fork) menu system | Creative tool UI | ~30 | 1–2 step |
| R5 | MRTK3 Sample Hub for Unity | Mixed reality UI samples | ~40 | 1–3 step |

**Hand-crafted scenes (5 scenes):**

| ID | Description | Elements | Key challenge |
|----|-------------|----------|---------------|
| H1 | VR settings menu: nested panels, scroll views, toggles, sliders | 50 | Deep nesting, small targets |
| H2 | Industrial control panel: multi-step procedures, safety interlocks | 35 | Sequential dependencies, bimanual |
| H3 | Surgical instrument tray: precise grasps, handoff sequences | 25 | Fine motor, anthropometric sensitivity |
| H4 | Collaborative assembly: two-person task adapted for single user | 40 | Spatial spread, sequential traps |
| H5 | Accessibility stress test: all 8 bug categories present by design | 60 | Known ground truth for naturalistic layout |

**Ground truth for real scenes:**
1. Three independent annotators (the authors) label each element as accessible/inaccessible
   for 5th, 25th, 50th, 75th, 95th percentile body types.
2. Disagreements resolved by 10M-sample Monte Carlo at the disputed percentile.
3. Final labels stored as JSON alongside the scene.
4. For hand-crafted scenes, ground truth is exact by construction (H5 especially).

---

## 3. Baseline Implementations

### 3.1 Baseline 1: Stratified Monte Carlo (1M samples)

**Implementation:** Python orchestrator + C++ Pinocchio FK evaluation via pybind11.

**Sampling strategy:**
```
stratified_mc(scene, n_samples=1_000_000, seed=42):
  rng = RandomState(seed)

  # Stratify across 5 ANSUR-II percentile bands
  bands = [(0, 10), (10, 30), (30, 70), (70, 90), (90, 100)]
  samples_per_band = n_samples // len(bands)  # 200K each

  for band_lo, band_hi in bands:
    body_params = ansur2.sample_percentile_range(
      band_lo, band_hi, samples_per_band, rng
    )
    for b in body_params:
      # Sample 1000 joint configurations uniformly in ROM
      thetas = uniform_joint_sample(b.rom_limits, n=1000, rng=rng)
      for elem in scene.elements:
        reachable = any(
          ||FK(theta, b) - elem.position|| < elem.activation_radius
          for theta in thetas
        )
        record(elem.id, b, reachable)

  # Detection criterion: bug detected if ≥1 sample witnesses failure
  for elem in scene.elements:
    fail_count = count(not reachable for (_, _, reachable) in records[elem.id])
    elem.mc_detected = (fail_count > 0)
    elem.mc_fail_rate = fail_count / n_samples
```

**Detection criterion:** A bug is "detected" by MC if at least one sample in the failing
percentile range witnesses inaccessibility. For subtle bugs (severity 4–5), the failure
set may be so small that 1M samples have <50% probability of hitting it — this is
precisely where formal verification adds value.

**Runtime estimation:** Pinocchio FK evaluation: ~1 μs per call. 1M body params × 1K
joint configs = 10⁹ FK calls × 1 μs = 1000s ≈ 17 minutes. Parallelized across 8 cores:
~2 minutes per scene. Full 810-scene suite: ~27 hours (batched overnight).

### 3.2 Baseline 2: Clopper-Pearson Confidence Intervals

**Implementation:** 50 lines of Python using `scipy.stats.beta`.

**Procedure:**
```
clopper_pearson_baseline(mc_results, alpha=0.01):
  for elem in scene.elements:
    n = total_samples_for(elem)
    k = failure_count_for(elem)
    # 99% CP upper bound on true failure probability
    if k == 0:
      cp_upper = 1 - (alpha / 2) ** (1 / n)
    else:
      cp_upper = beta.ppf(1 - alpha / 2, k + 1, n - k)
    elem.cp_bound = cp_upper
```

**Comparison metric:** For each element, compute ratio:

    r_i = ε_cert(element_i) / cp_upper(element_i)

Report distribution of r_i across all elements and scenes. The certificate "wins" if
median(r_i) ≤ 0.20 (i.e., 5× improvement).

**Why CP is the right comparison:** CP is the tightest frequentist confidence interval for
binomial proportions. If the certificate cannot beat CP, it adds no value over "just
sample a lot and compute a confidence interval." The 5× threshold (gate D3) ensures the
improvement is not marginal.

### 3.3 Baseline 3: Lookup Table (20 Discrete Percentiles)

**Implementation:** ~500 lines of Python. The Skeptic's minimalist alternative.

```
lookup_table_baseline(scene, seed=42):
  # Pre-compute reachability for 20 fixed body types
  percentiles = [1, 5, 10, 15, 20, 25, 30, 40, 50, 60,
                 70, 75, 80, 85, 90, 95, 97, 99, 99.5, 99.9]
  bodies = [ansur2.percentile_body(p) for p in percentiles]

  for body in bodies:
    reach_envelope = compute_max_reach(body)  # Pinocchio FK, 10K samples
    for elem in scene.elements:
      dist = ||elem.position - body.shoulder_origin||
      elem.lookup_reachable[body.pct] = (dist <= reach_envelope)

  # Detection: flag if any percentile in target range is unreachable
  for elem in scene.elements:
    elem.lookup_detected = any(
      not elem.lookup_reachable[p]
      for p in percentiles if p >= 5 and p <= 95
    )
```

**What it captures:** Simple distance-based reachability failures (categories B1, B6, B7
partially). **What it misses:** Occlusion (B2), pose-impossible (B3), cross-device (B4),
sequential (B5), bimanual (B8), and any bug requiring joint-angle reasoning.

**Key comparison:** Tier 1's marginal detection rate over this baseline quantifies the
value of affine-arithmetic FK over naive distance checking. If Tier 1 catches <10% more
bugs than the lookup table, the UIST paper must justify the engineering complexity.

### 3.4 Baseline 4: Geometric Heuristic Checker

**Implementation:** ~800 lines of Python.

```
heuristic_checker(scene):
  for elem in scene.elements:
    # Bounding-box reachability: is element within max reach sphere?
    max_reach_5pct = ansur2.max_reach(percentile=5)
    max_reach_95pct = ansur2.max_reach(percentile=95)
    dist = ||elem.position - user_shoulder_position||

    elem.heuristic_reachable = (dist <= max_reach_5pct)
    elem.heuristic_flagged = (dist > max_reach_5pct * 0.85)  # 15% margin

    # Height check
    if elem.position.y > ansur2.max_overhead_reach(percentile=5):
      elem.heuristic_flagged = True
    if elem.position.y < ansur2.min_comfortable_reach(percentile=95):
      elem.heuristic_flagged = True

    # Bimanual spread check (if applicable)
    if elem.is_bimanual:
      spread = elem.left_target - elem.right_target
      if ||spread|| > ansur2.shoulder_width(percentile=5) * 1.5:
        elem.heuristic_flagged = True
```

**No state machines, no joint-angle reasoning, no sequential analysis.** This baseline
measures how much geometric heuristics alone capture vs. the full verification pipeline.

### 3.5 Baseline 5: Random Walk Exploration Agent

**Implementation:** ~400 lines of Python.

```
random_walk_agent(scene, n_walks=10000, max_steps=20, seed=42):
  rng = RandomState(seed)
  body = ansur2.percentile_body(rng.choice([5, 25, 50, 75, 95]))

  for walk_id in range(n_walks):
    state = scene.initial_state()
    for step in range(max_steps):
      available = state.available_interactions(body)
      if not available:
        break
      action = rng.choice(available)
      state = state.apply(action, body)
    # Record coverage
    for elem in state.visited:
      agent_coverage[elem.id][body.pct] += 1

  # Detection: element flagged if coverage < threshold
  for elem in scene.elements:
    for pct in [5, 25, 50, 75, 95]:
      visits = agent_coverage[elem.id][pct]
      expected = n_walks * 0.2 * (1 / len(scene.elements))
      if visits < expected * 0.1:  # 10× fewer visits than expected
        elem.agent_flagged = True
```

**Rationale:** Simulates a manual tester randomly exploring the scene. Measures how many
traversals approximate exhaustive testing. Expected to miss sequential traps and
anthropometric-boundary bugs.

**Calibration:** Run with n_walks ∈ {100, 1K, 10K, 100K} to plot detection rate vs.
exploration budget. Compare the human-equivalent cost: 10K walks ≈ 30 minutes of
simulated exploration.

---

## 4. Metrics (Precise Definitions)

### 4.1 Detection Rate (True Positive Rate)

**Definition:** For a set of injected bugs B and detected bugs D:

    TPR = |D ∩ B| / |B|

Computed per bug category c ∈ {B1, ..., B8}:

    TPR_c = |D ∩ B_c| / |B_c|

And per severity level s ∈ {1, ..., 5}:

    TPR_s = |D ∩ B_s| / |B_s|

**How computed:** An injected bug at element e is "detected" if the tool flags e as
inaccessible for any body parameterization in the bug's affected percentile range.

**Confidence interval:** Clopper-Pearson 95% CI on TPR treated as binomial proportion.

**Success criterion:** TPR > 95% (Tier 1), TPR > 97% (Tier 2). Report per-category
breakdown; no individual category below 80%.

### 4.2 False Positive Rate

**Definition:** For accessible elements A and flagged-as-inaccessible elements F:

    FPR = |F ∩ A| / |A|

An element is "accessible" if the ground truth labels it reachable for all percentiles in
the 5th–95th range. An element is "flagged" if the tool reports it as potentially
inaccessible.

**Confidence interval:** Clopper-Pearson 95% CI.

**Success criterion:** FPR < 15% (Tier 1), FPR < 5% (Tier 2).

### 4.3 Coverage Certificate ε

**Definition:** The certificate ⟨S, V, ε, δ⟩ asserts:

    P(∃ undetected accessibility bug in Θ \ (S ∪ V)) ≤ ε

with confidence ≥ 1 − δ, conditional on the accessibility frontier being L-Lipschitz in
Θ \ V_boundary where V_boundary is the set of identified Lipschitz-violation regions.

**How computed:** The certificate engine outputs ε directly. We validate ε empirically:
1. Generate a scene with a known bug at a known location in Θ.
2. Run the certificate engine with the bug location excluded from sampling.
3. Check whether ε correctly bounds the probability of the unsampled bug.
4. Repeat 1000 times with different bug locations. The fraction of times the bug falls
   in the "undetected" region should be ≤ ε (up to δ tolerance).

**Success criterion:** ε < 0.01 for single-step 30-object scenes in 10 minutes;
ε < 0.1 for ≤3-step scenes in 15 minutes.

### 4.4 ε Improvement Ratio over Clopper-Pearson

**Definition:**

    ρ_i = ε_cert(element_i) / CP(n_i, k_i, 0.01)

where n_i is the number of samples for element i, k_i is the number of failures observed.

**Aggregation:** Report median(ρ), mean(ρ), and 90th percentile of ρ across all elements
and scenes.

**Success criterion:** median(ρ) ≤ 0.20 (i.e., ≥5× improvement).

### 4.5 Verification Time vs. Scene Complexity (Scalability)

**Definition:** Wall-clock time T(n) to verify a scene with n interactable objects, from
scene ingestion to verdict/certificate emission.

**How computed:** `time.monotonic()` around the full pipeline. Report median and 95th
percentile across scenes of the same size.

**Presentation:** Log-log plot of T(n) vs. n for n ∈ {5, 10, 30, 50, 100}. Fit
power-law T(n) = a · n^b. Report exponent b with 95% CI.

**Success criterion:**
- Tier 1: T(n) < 2s for n ≤ 100. Exponent b < 1.5 (sub-quadratic).
- Tier 2: T(n) < 600s for n ≤ 50. Exponent b < 2.5.

### 4.6 Counterexample Precision (Biomechanical Validity)

**Definition:** For each reported counterexample trace (body params b, joint sequence θ₁,
..., θ_k):

    valid(trace) = all of:
      (a) FK(θ_i, b) is within activation volume of target element at step i
      (b) All joint angles within ROM limits for body b
      (c) Joint angular velocity between consecutive steps ≤ physiological maximum
      (d) No self-collision in the kinematic chain

    Precision = |valid counterexamples| / |total counterexamples|

**How computed:** Run each reported counterexample trace through the Pinocchio FK model
with collision checking and ROM validation.

**Success criterion:** Precision > 95%.

### 4.7 Marginal Detection Rate over Monte Carlo

**Definition:**

    MDR = |B_T2 \ B_MC| / |B_all \ B_MC|

where B_T2 is the set of bugs detected by Tier 2, B_MC is the set detected by MC, and
B_all is the full bug set.

**Confidence interval:** Clopper-Pearson 95% CI on MDR as binomial proportion.

**Success criterion:** MDR ≥ 10% (gate A3). Report per-category breakdown.

---

## 5. Ablation Studies

Each ablation removes exactly one component, holding all else constant. Run on 100
procedural scenes (balanced across object counts) with 50 injected bugs each.

### 5.1 No SMT Verification (Sampling Only)

**Removed:** All SMT queries in Tier 2. Certificate computed from sampling alone.
**Expected impact:** ε increases by 3–10× (the entire V region is empty, so all
uncertainty comes from sampling). This ablation directly measures SMT's contribution to
certificate tightness.
**Metrics:** ε, TPR, FPR, runtime.

### 5.2 No Tier 1 Seeding

**Removed:** Frontier-region information from Tier 1 is not passed to Tier 2's adaptive
sampler. Tier 2 starts with uniform sampling.
**Expected impact:** Tier 2 requires 2–5× more samples to achieve the same ε, because the
cold-start phase wastes samples in clearly-accessible/clearly-inaccessible regions.
**Metrics:** ε at fixed sample budget, number of samples to reach ε = 0.01.

### 5.3 No Lipschitz Detection

**Removed:** Lipschitz-violation boundary detection is disabled. Certificate assumes global
Lipschitz regularity.
**Expected impact:** Certificate becomes unsound on scenes with knife-edge boundaries.
Measure by checking ε validity on scenes with known Lipschitz-violating bugs (severity-5
B1 and B6 bugs). If ε is too optimistic (underestimates true failure probability on >5%
of scenes), this component is essential.
**Metrics:** ε soundness rate (fraction of scenes where ε correctly bounds true failure
probability), ε value.

### 5.4 No Adaptive Sampling (Uniform Only)

**Removed:** Replace frontier-adaptive Latin hypercube sampling with uniform Latin hypercube
sampling over full Θ.
**Expected impact:** ε increases by 2–4× at the same sample count. Frontier-adaptive
sampling concentrates information where it matters; uniform sampling wastes budget on the
interior of the accessible/inaccessible regions.
**Metrics:** ε at fixed sample budget, ε at fixed runtime.

### 5.5 No Subdivision in Affine Arithmetic

**Removed:** Disable parameter-space subdivision in Tier 1. Single evaluation over full
5th–95th percentile range.
**Expected impact:** Wrapping factor increases from ~5× to ~15× on wide-range joints,
causing FPR to increase from <15% to >30%.
**Metrics:** Wrapping factor w, FPR, TPR (should not decrease).

### 5.6 Summary Table

| Ablation | Component removed | Primary metric affected | Expected direction |
|----------|------------------|------------------------|--------------------|
| A1 | SMT verification | ε | ε increases 3–10× |
| A2 | Tier 1 seeding | ε (at fixed budget) | ε increases 2–5× |
| A3 | Lipschitz detection | ε soundness | Unsound on 5–20% of scenes |
| A4 | Adaptive sampling | ε | ε increases 2–4× |
| A5 | AA subdivision | FPR (Tier 1) | FPR increases 2–3× |

---

## 6. Scalability Experiments

### 6.1 Scene Complexity Curves

**Setup:** Generate scenes at n ∈ {5, 10, 20, 30, 50, 75, 100} objects, 10 scenes per
size, uniform distribution, 1-step interactions, medium sensitivity.

**Measure:** Tier 1 time, Tier 2 time (to certificate), peak memory, ε.

**Presentation:** Log-log plots:
- T₁(n) vs. n with fitted power law and 95% CI on exponent.
- T₂(n) vs. n with fitted power law.
- Memory(n) vs. n.

**Expected scaling:**
- Tier 1: O(n) — each element independently evaluated. Expected exponent b ≈ 1.0.
- Tier 2: O(n^α) with α ∈ [1.5, 2.5] — SMT query count scales with frontier complexity.

### 6.2 Parameter Dimension Curves

**Setup:** Fix scene (10 objects, uniform) and vary kinematic model:
- 3-DOF (shoulder only): d = 3
- 5-DOF (shoulder + elbow): d = 5
- 7-DOF (full arm): d = 7
- 10-DOF (arm + torso lean): d = 10
- 14-DOF (bimanual): d = 14

**Measure:** Tier 1 time, Tier 2 time, ε, wrapping factor.

**Presentation:** Semi-log plots of time and ε vs. d.

**Expected scaling:**
- Tier 1 wrapping factor: grows as O(d^0.5) to O(d) depending on joint-range width.
- Tier 2 ε: grows exponentially with d unless frontier dimensionality is lower than d
  (expected for most bugs).

### 6.3 Multi-Step Interaction Curves

**Setup:** Fix scene (20 objects) and vary interaction depth k ∈ {1, 2, 3, 4, 5}.

**Measure:** Tier 2 time, ε, TPR, certificate size (bytes).

**Presentation:**
- ε vs. k at fixed 15-minute budget.
- Time to reach ε = 0.01 vs. k.

**Expected scaling:**
- Effective parameter dimension scales as k × d. At k = 3, d = 7: 21 dimensions,
  manageable. At k = 5, d = 7: 35 dimensions, ε > 0.1 expected.

### 6.4 Scalability Summary Table

| Experiment | Independent var | Dependent vars | Range | Scenes per point |
|------------|----------------|----------------|-------|------------------|
| 6.1 Scene complexity | n (objects) | T₁, T₂, mem, ε | 5–100 | 10 |
| 6.2 DOF dimension | d (joints) | T₁, T₂, ε, w | 3–14 | 10 |
| 6.3 Multi-step depth | k (steps) | T₂, ε, TPR | 1–5 | 10 |

---

## 7. Threats to Validity

### 7.1 Internal Threats

| Threat | Description | Mitigation | Residual risk |
|--------|-------------|------------|---------------|
| **Bug injection bias** | Procedural bug templates may not capture the distribution of real-world accessibility failures. Templates are designed by the authors, not derived from field data. | Include 10 real scenes with naturally occurring bugs. Cross-reference injected bug categories with published XR accessibility literature (Mott et al., CHI 2019; Gerling et al., CHI 2020). | Moderate. No empirical data on XR spatial accessibility bug distributions exists. We report per-category TPR so readers can assess which categories are well-served. |
| **Procedural scene unrealism** | Procedural scenes may be easier or harder than real XR layouts due to simplified spatial structure. | Include 10 real/hand-crafted scenes. Report procedural and real-scene results separately with statistical comparison (Mann-Whitney U test on TPR distributions). | Moderate. Procedural scenes are necessary for statistical power; real scenes are necessary for external validity. Neither alone is sufficient. |
| **ANSUR-II population bias** | ANSUR-II is a US military dataset (ages 17–51, primarily able-bodied, fitness-selected). Does not represent elderly, pediatric, or many disability populations. | Supplement with published disability-specific ROM data (Boone & Azen 1979, Soucie et al. 2011). Add sensitivity analysis: re-run experiments with ±20% ROM reduction to simulate mobility impairment. | High. This is an inherent limitation of available anthropometric databases. Explicitly documented. Do not claim "verifies accessibility for disabled users" without disability-specific kinematic data. |
| **Ground truth validity** | Bug injection ground truth is computed analytically by the same kinematic model used for verification. | Cross-validate ground truth with 10M-sample brute force MC using independent FK implementation (Pinocchio vs. custom FK). Any discrepancy >0.1% triggers investigation. | Low. Kinematic FK is deterministic and well-understood. Cross-validation catches implementation bugs. |
| **Overfitting to benchmark suite** | System tuned to perform well on the specific benchmark configurations. | Hold out 20% of procedural scenes as a test set (generated with different seeds, not used during development). Report held-out vs. training results. Use 5-fold cross-validation for any hyperparameter tuning (e.g., SMT timeout, adaptive sampling parameters). | Low–moderate. Held-out test set and cross-validation are standard mitigations. |

### 7.2 External Threats

| Threat | Description | Mitigation | Residual risk |
|--------|-------------|------------|---------------|
| **Unity-only evaluation** | System only tested on Unity scenes. May not generalize to Unreal Engine, WebXR, or proprietary engines. | Scene IR is engine-agnostic; only the parser is Unity-specific. Document which IR features are Unity-dependent. | Moderate. The verification core is engine-independent; only the scene ingestion layer is Unity-specific. |
| **Limited body model** | 7-DOF arm model omits lower body, spine, neck. Seated mode is approximated. | Acknowledge explicitly. Measure impact by comparing 7-DOF results with 10-DOF (adding torso lean, hip rotation) on a subset of scenes. | Moderate. Lower-body interactions (foot pedals, floor-level targets) are out of scope. |
| **No real human validation** | No study with real humans attempting the flagged interactions. Biomechanical validity is model-based only. | Counterexample precision metric (§4.6) validates biomechanical plausibility computationally. Real-human validation is future work. | Moderate–high. Model-based validation cannot capture fatigue, pain, cognitive load, or individual variation beyond ANSUR-II. |
| **Single-user interactions only** | Multi-user collaborative interactions are not modeled. | Document as scope limitation. Multi-user XR is a distinct research problem (shared workspace negotiation). | Low for single-user settings; high for collaborative XR. |

### 7.3 Construct Threats

| Threat | Description | Mitigation | Residual risk |
|--------|-------------|------------|---------------|
| **ε vs. practical usefulness** | ε measures mathematical coverage of the parameter space, not user-perceived accessibility. A tool could achieve ε = 0.001 while missing the most impactful real-world failures. | Report detection rate per bug category alongside ε. Correlation analysis between ε and TPR across scenes. If correlation < 0.5, ε is a poor proxy for practical utility. | Moderate. ε is the formal contribution; TPR is the practical contribution. Both are reported. |
| **Injected TPR ≠ real-world value** | High TPR on injected bugs may not translate to value on naturally occurring bugs, which may have different characteristics. | Real-scene corpus (10 scenes) with naturally occurring bugs provides an independent TPR estimate. Compare injected vs. natural TPR. | Moderate. The 10 real scenes provide ground truth for natural bugs, but 10 is a small sample. |
| **FPR sensitivity to ground truth** | FPR depends on correctly labeling accessible elements. If ground truth is wrong (element labeled accessible but actually marginally inaccessible), FPR is inflated. | Ground truth verification via 10M-sample MC. Sensitivity analysis: report FPR at multiple accessibility thresholds (5th pct only, 5th–10th pct, 5th–25th pct). | Low. 10M samples with analytical cross-check makes ground truth errors unlikely. |
| **Verification time is hardware-dependent** | Reported times are specific to the test hardware. Different machines will yield different results. | Report hardware specs alongside all timing results. Normalize by a single-FK-evaluation benchmark to enable cross-machine comparison. Include regression model T = f(n, d, k, hardware_factor). | Low. Standard practice in systems evaluation. |

---

## 8. Experimental Timeline

### Phase Map

```
Month 1  ─── D1: Wrapping factor experiment
          │    - Implement AA FK engine
          │    - Run 1000-geometry wrapping measurement
          │    - Decision: proceed / Taylor models / abandon
          │
Month 2  ─── D2: Certificate prototype ε on 10-object scene
          │    - Implement sampling + SMT pipeline
          │    - Measure ε on 20 scenes, 3 budgets
          ├── D3: Certificate vs. Clopper-Pearson comparison
          │    - Run paired comparison on 100 scenes
          │    - Wilcoxon signed-rank test
          │    - Decision: proceed / downscope
          │
Month 3  ─── D4: Lipschitz violation frequency
          │    - Run boundary detector on 100+ scenes
          │    - Measure violation rate
          │    - Decision: proceed / downscope certificate
          ├── D7: Developer feedback (parallel, non-blocking for experiments)
          │
Month 4  ─── D5: Multi-step certificates
          │    - Extend Tier 2 to 2-step and 3-step interactions
          │    - Measure ε on 120 multi-step scenes
          │    - Decision: proceed / restrict to single-step
          │
Month 5  ─── Full benchmark suite execution (Phase 1)
          │    - Generate all 810 procedural scenes
          │    - Run all 5 baselines on full suite
          │    - Begin real-scene corpus analysis
          │
Month 6  ─── A3: Formal vs. MC marginal detection
          │    - Compute MDR across full suite
          │    - Decision: proceed / downscope
          ├── A4: Real scene corpus results
          │    - Complete 10-scene analysis
          │    - Ground truth validation
          │
Month 7  ─── Ablation studies + scalability experiments
          │    - 5 ablation conditions × 100 scenes
          │    - 3 scalability experiment sets
          │
Month 8  ─── Statistical analysis + paper writing
          │    - Full hypothesis testing pipeline
          │    - Generate all figures (matplotlib, reproducible)
          │    - Paper drafts: CAV (certificates) + UIST (tool)
          │
Month 9  ─── Paper revision + supplementary materials
               - Address internal review feedback
               - Reproducibility package preparation
```

### Gate Dependencies

```
D1 ──→ Tier 1 implementation ──→ D2 ──→ D3 ──→ D5
                                   │
                                   └──→ D4
                                         │
D7 (parallel) ─────────────────────────→ A3 ──→ Paper decisions
                                         │
                              A4 ────────┘
```

**Critical path:** D1 → D2 → D3 → D5 → A3. Total: 6 months to primary evaluation
results. Any gate failure triggers immediate rescoping per the kill-chain.

---

## 9. Data and Reproducibility

### 9.1 Random Seeds and Deterministic Reproduction

**Seed hierarchy:**
```
MASTER_SEED = 20260308  # Date-based, fixed for the project

scene_seed(config_id) = hash(MASTER_SEED, config_id)
bug_seed(scene_id, bug_id) = hash(MASTER_SEED, scene_id, bug_id)
mc_seed(scene_id) = hash(MASTER_SEED, "mc", scene_id)
sampler_seed(scene_id) = hash(MASTER_SEED, "sampler", scene_id)
```

**Determinism requirements:**
- All random number generators use `numpy.random.RandomState` or `PCG64` with explicit
  seeding. No calls to unseeded `random()`.
- Floating-point determinism: use `--fp-model=precise` compiler flag for C++ FK code.
  Pin BLAS/LAPACK to single-threaded operation during FK evaluation.
- SMT solver: fix Z3 random seed (`set_param("smt.random_seed", 42)`). Record solver
  version (Z3 must be pinned to specific release).

**Reproduction procedure:**
```bash
# 1. Install dependencies (pinned versions)
pip install -r requirements.txt  # numpy==1.26.4, scipy==1.12.0, z3-solver==4.12.6.0, ...

# 2. Generate benchmark suite
python generate_benchmarks.py --master-seed 20260308 --output benchmarks/

# 3. Run all experiments
python run_experiments.py --config experiments.yaml --output results/

# 4. Generate figures and statistics
python analyze.py --input results/ --output figures/

# 5. Validate reproducibility
python validate_reproducibility.py --results1 results/ --results2 results_rerun/
```

### 9.2 Hardware Specification Requirements

**Primary test machine (all timing results):**
- Apple MacBook Pro, M2 Pro, 16 GB RAM
- macOS 14.x
- Python 3.11, C++17 (Apple Clang)
- Z3 4.12.6, Pinocchio 2.7.x

**Normalization benchmark:** Time 10⁶ FK evaluations on the test machine and report as
`FK_baseline_ns`. All timing results accompanied by this number so readers can normalize
to their hardware.

**Minimum requirements for reproduction:**
- 8-core CPU, 16 GB RAM
- ~50 GB disk for full benchmark suite + results
- Estimated full-suite runtime: ~72 hours (parallelizable to ~12 hours on 8 cores)

### 9.3 Runtime Measurement Methodology

**Wall-clock time:** `time.monotonic()` (Python) or `std::chrono::steady_clock` (C++).
No `time.time()` (subject to NTP adjustments).

**Process isolation:** During timing measurements:
- No other user-level processes consuming >5% CPU.
- Thermal throttling check: if CPU frequency drops below 80% of base, discard run and
  re-run after cooldown.
- Warm-up: 3 untimed runs before each timed measurement to stabilize caches.
- Repetitions: 5 timed runs per configuration. Report median and IQR.

**Memory measurement:** `tracemalloc` (Python peak) + `getrusage(RUSAGE_SELF).ru_maxrss`
(process peak). Report both.

### 9.4 Statistical Analysis Pipeline

**Automated pipeline (no manual steps):**

```python
# analysis_pipeline.py

def run_analysis(results_dir):
    # 1. Load all results
    results = load_results(results_dir)

    # 2. Hypothesis tests
    h1 = test_epsilon_improvement(results)       # Wilcoxon signed-rank
    h2 = test_wrapping_factor(results)            # One-sample t-test on log(w)
    h3a = test_tier1_tpr(results)                 # Exact binomial
    h3b = test_tier1_fpr(results)                 # Exact binomial
    h4a = test_tier2_tpr(results)                 # Exact binomial
    h4b = test_tier2_fpr(results)                 # Exact binomial
    h5 = test_marginal_detection(results)         # Exact binomial
    h6 = test_multistep_epsilon(results)          # Exact binomial

    # 3. Multiple comparison correction
    p_values = [h1.p, h2.p, h3a.p, h3b.p, h4a.p, h4b.p, h5.p, h6.p]
    corrected = bonferroni_correction(p_values, alpha=0.05)

    # 4. Confidence intervals
    ci_tpr_tier1 = clopper_pearson_ci(h3a.successes, h3a.trials, 0.05)
    ci_tpr_tier2 = clopper_pearson_ci(h4a.successes, h4a.trials, 0.05)
    ci_fpr_tier1 = clopper_pearson_ci(h3b.successes, h3b.trials, 0.05)
    ci_fpr_tier2 = clopper_pearson_ci(h4b.successes, h4b.trials, 0.05)
    ci_mdr = clopper_pearson_ci(h5.successes, h5.trials, 0.05)

    # 5. Effect sizes
    effect_epsilon = compute_effect_size(results, metric='epsilon_ratio')
    effect_wrapping = compute_effect_size(results, metric='wrapping_factor')

    # 6. Generate figures
    plot_epsilon_comparison(results, output='figures/epsilon_comparison.pdf')
    plot_scalability_curves(results, output='figures/scalability.pdf')
    plot_ablation_results(results, output='figures/ablation.pdf')
    plot_category_breakdown(results, output='figures/category_tpr.pdf')
    plot_wrapping_factor_distribution(results, output='figures/wrapping.pdf')

    # 7. Generate LaTeX tables
    generate_hypothesis_table(corrected, output='tables/hypotheses.tex')
    generate_baseline_comparison(results, output='tables/baselines.tex')
    generate_ablation_table(results, output='tables/ablation.tex')

    # 8. Kill-gate assessment
    assess_gates(results, corrected)

    return AnalysisReport(hypotheses=corrected, figures=figures, tables=tables)
```

**Figure specifications (for both papers):**

| Figure | Paper | Description | Format |
|--------|-------|-------------|--------|
| F1 | CAV | ε_cert vs. CP bound, scatter with diagonal | PDF, 3.5" × 3" |
| F2 | CAV | ε vs. sample count (cert vs. CP), log-log | PDF, 3.5" × 3" |
| F3 | CAV | Ablation bar chart (5 conditions × ε) | PDF, 7" × 3" |
| F4 | CAV | Scalability log-log (time vs. n, time vs. d) | PDF, 7" × 3" |
| F5 | CAV | Multi-step ε vs. k, with 15-min budget line | PDF, 3.5" × 3" |
| F6 | UIST | TPR per bug category, grouped bar (Tier 1, Tier 2, baselines) | PDF, 7" × 3" |
| F7 | UIST | FPR comparison across tools | PDF, 3.5" × 3" |
| F8 | UIST | Marginal detection rate over MC per category | PDF, 3.5" × 3" |
| F9 | UIST | Real scene case study: annotated scene diagram | SVG + PDF |
| F10 | Both | Baseline comparison radar chart (5 baselines × 5 metrics) | PDF, 4" × 4" |

### 9.5 Artifact Package

**Released artifacts:**
1. `benchmarks/` — All 810 procedural scenes + 10 real scenes (JSON + DSL annotations).
2. `ground_truth/` — Bug labels for every element in every scene.
3. `results/` — Raw experimental results (CSV per experiment).
4. `analysis/` — Statistical analysis scripts and generated figures.
5. `baselines/` — All 5 baseline implementations.
6. `configs/` — Experiment manifests (JSON) with all seeds and parameters.
7. `Dockerfile` — Reproducible build environment.
8. `REPRODUCE.md` — Step-by-step reproduction instructions.

---

## 10. Summary: Experiment-to-Gate Mapping

| Gate | Experiment(s) | Hypothesis | Pass criterion | Kill action |
|------|--------------|------------|----------------|-------------|
| D1 (M1) | §2 H2: Wrapping factor | H1_2 | w ≤ 5× on 95% of geometries | Taylor models or abandon |
| D2 (M2) | §3 Cert prototype | — | ε < 0.05, 10 objects, 5 min | Abandon certificate |
| D3 (M2) | §2 H1: ε improvement | H1_1 | median(ρ) ≤ 0.20 | Downscope to tool paper |
| D4 (M3) | Lipschitz measurement | — | ≤20% undetectable violations | Downscope certificate claim |
| D5 (M4) | §2 H6: Multi-step | H1_6 | ε < 0.1 on 75% of scenes | Restrict to single-step |
| A3 (M6) | §2 H5: Marginal detection | H1_5 | MDR ≥ 10% | Downscope to theory + linter |
| A4 (M6) | Real scene corpus | — | ≥5 scenes complete Tier 2 | Procedural-only evaluation |

**Total experimental compute budget:** ~72 CPU-hours for full suite.
**Total wall-clock time:** ~12 hours with 8-core parallelization per run, plus
~27 hours for MC baseline (overnight batch).

---

*End of empirical evaluation proposal.*
