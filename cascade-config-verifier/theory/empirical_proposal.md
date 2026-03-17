# CascadeVerify: Empirical Evaluation Proposal

**Document type:** Falsifiable evaluation strategy for NSDI submission  
**Scope:** Retry amplification + timeout chain violations in microservice configs (CB-free, monotone model)  
**Tool summary:** Two-tier static analysis (graph + BMC) with MaxSAT repair on K8s/Istio/Envoy YAML  

---

## 1. Falsifiable Hypotheses

Each hypothesis specifies a quantitative threshold, a falsification criterion, and implications of falsification.

### H1 — Detection Effectiveness

**Statement:** CascadeVerify detects ≥95% of injected retry amplification and timeout chain violations in semi-synthetic topologies (N≥200 configs), with a false positive rate ≤10%.

**Operationalization:** Generate 200+ semi-synthetic configurations across four topology families (chain, tree, mesh, hub-and-spoke) at five scales (5, 10, 20, 30, 50 services). Inject exactly one ground-truth bug class per configuration drawn uniformly from {retry bomb, timeout chain, fan-in storm, multi-failure cascade}. A detection is a *true positive* if CascadeVerify reports a cascade risk whose minimal failure set overlaps ≥80% with the injected ground-truth failure set. A *false positive* is a reported risk with no corresponding injected bug and no independently verified cascade path.

**Falsification:** Recall < 95% OR FPR > 10% on the full 200-config corpus.

**Implications of falsification:**
- Recall 85–95%: Acceptable for a tool paper; reframe as "high-recall screening" and investigate missed bug classes. Likely cause: encoding gaps in fan-in or multi-failure scenarios.
- Recall < 85%: Fundamental encoding flaw. The BMC model does not faithfully capture retry-timeout interactions. Would require re-examination of the QF_LIA encoding (Subsystem S4) and load propagation semantics (Subsystem S3).
- FPR > 10%: Over-approximation in the failure model. Investigate whether the discrete-step load propagation model introduces spurious cascade paths not realizable under actual Envoy retry semantics (e.g., jitter absorption, retry budget capping).

---

### H2 — BMC Marginal Value

**Statement:** Tier 2 (BMC with failure set enumeration) discovers ≥15% additional cascade risks beyond Tier 1 (graph-only analysis) on topologies with fan-in degree ≥2.

**Operationalization:** Run both Tier 1 (product-of-retries along paths, sum-of-timeouts along paths) and Tier 2 (full BMC) on the same corpus. Tier 1 operates per-path; it cannot reason about simultaneous multi-caller load aggregation at shared dependencies or multi-failure scenarios requiring ≥2 simultaneous component failures. Compute the *marginal detection set*: bugs found by Tier 2 but missed by Tier 1. Restrict measurement to topologies with max fan-in ≥2, where the theoretical advantage of BMC is expected.

**Falsification:** |Tier2-only bugs| / |All bugs in fan-in≥2 subset| < 15%.

**Implications of falsification:**
- 10–15%: Tier 2 still adds value but the marginal cost may not justify the complexity. Reframe the two-tier architecture as "defense in depth" rather than claiming fundamental necessity.
- < 10%: Graph analysis is nearly sufficient for the CB-free monotone case. This would be an intellectually honest and publishable finding — it would mean the monotone retry-timeout model is simpler than anticipated. The paper's contribution shifts toward the repair synthesis and the monotonicity theorem itself.
- > 30%: Stronger than expected. Emphasize the critical importance of failure-set reasoning for cascade detection.

---

### H3 — Monotonicity Speedup

**Statement:** Monotonicity-aware antichain pruning (Theorem B6) reduces MinUnsat enumeration time by ≥10× compared to naive enumeration on topologies with ≥20 services.

**Operationalization:** Implement two MinUnsat enumeration strategies on the same MARCO-based framework: (a) naive enumeration without monotonicity exploitation (explores supersets of known cascading sets and subsets of known safe sets), (b) antichain-pruned enumeration that skips both. Measure wall-clock time to enumerate all minimal failure sets on the 20-, 30-, and 50-service topologies across all four topology families. Report geometric mean speedup with 95% confidence interval.

**Falsification:** Geometric mean speedup < 10× on 20+ service topologies.

**Implications of falsification:**
- 5–10×: Still a meaningful speedup; the monotonicity theorem is correct but the search space is already manageable at these scales (the lattice is shallow). Reframe as "enables practical enumeration" rather than "order-of-magnitude improvement."
- < 5×: The pruning overhead (maintaining antichains, subset/superset checks) approaches the savings. This would suggest that for topologies ≤50 services, the combinatorial explosion is mild enough that naive enumeration suffices. The monotonicity theorem retains theoretical value but the practical speedup claim must be weakened.
- > 50×: Stronger than expected. The exponential blowup in failure set space is real, and antichain pruning is essential for tractability.

---

### H4 — Repair Quality

**Statement:** MaxSAT repairs change ≤3 parameters on average, achieve 100% soundness (every repaired configuration passes re-verification), and produce repairs within 20% of optimal total weighted distance.

**Operationalization:** For each detected cascade risk in the 200-config corpus, invoke the MaxSAT repair synthesizer. Measure: (a) *parameter count* — number of retry counts or timeout budgets modified; (b) *soundness* — re-run Tier 1 + Tier 2 on the repaired configuration and verify zero cascade risks remain; (c) *optimality gap* — compare the total weighted deviation (Σ |original_i − repaired_i| × weight_i) against the known optimal repair from the ground-truth generator. The ground-truth generator exhaustively searches the (small, bounded) parameter space for each injected bug.

**Falsification:** Mean parameters changed > 3 OR soundness < 100% OR mean optimality gap > 20%.

**Implications of falsification:**
- Soundness < 100%: Critical bug in the MaxSAT hard-clause encoding. This is a correctness failure, not a performance issue. Would require debugging the constraint encoding in Subsystem S6. *This must not ship.*
- Mean params > 3: Repairs are more invasive than desired. Investigate whether the soft-clause weighting function properly reflects operational impact. May need domain-specific weights (e.g., retry count changes are cheaper than timeout changes).
- Optimality gap > 20%: The MaxSAT solver is finding suboptimal solutions, likely due to encoding artifacts or solver timeout. Investigate whether the weighted partial MaxSAT formulation has large symmetry groups that confuse the solver.

---

### H5 — Scalability

**Statement:** End-to-end verification (Tier 1 + Tier 2 + repair) completes in <30s for 30-service topologies and <90s for 50-service topologies on a single laptop CPU (Apple M-series or x86-64 equivalent, single-threaded).

**Operationalization:** Measure wall-clock time on a MacBook Pro M2 (or equivalent, specified in paper). Report Tier 1 time, Tier 2 time per query, full MinUnsat enumeration time, and MaxSAT repair time separately. Measure across 20 configurations (4 topologies × 5 scales). Report median and 95th percentile.

**Falsification:** Median end-to-end time > 30s at 30 services OR > 90s at 50 services.

**Implications of falsification:**
- 30s–60s at 30 services: Marginal. The tool is usable in CI/CD (most pipelines tolerate 60s) but the "fast enough for pre-merge" claim is weakened. Investigate bottleneck: if Tier 1 is fast but MinUnsat enumeration is slow, consider bounding enumeration to top-k minimal sets.
- > 60s at 30 services: Scalability ceiling is lower than claimed. Honestly report the ceiling and investigate compositional decomposition for larger topologies. The paper can still succeed if the ceiling is clearly characterized with a log-log regression.
- At 50 services, > 90s but < 300s: Acceptable with reframing. "Nightly batch verification" rather than "pre-merge gate."
- At 50 services, > 300s: The approach does not scale to 50 services without compositional decomposition. Reduce the scalability claim to 30 services and present 50-service results as motivation for future compositional work.

---

### H6 — Real-World Bugs

**Statement:** CascadeVerify identifies ≥3 previously unknown cascade risks in real open-source Kubernetes configurations.

**Operationalization:** Run CascadeVerify on the real-config corpus (§2, Axis 2). A "previously unknown cascade risk" is a retry amplification or timeout chain violation that: (a) is confirmed by manual inspection of the call graph and parameter values, (b) is not flagged by existing linters (kube-score, istioctl analyze), and (c) has not been reported in the project's issue tracker. Report each finding with the specific services, parameter values, failure set, and cascade mechanism.

**Falsification:** Fewer than 3 confirmed bugs across the entire real-config corpus.

**Implications of falsification:**
- 1–2 bugs: Weaken to "CascadeVerify identifies cascade risks in real configurations" (no count claim). Emphasize the semi-synthetic evaluation as primary evidence and present real-config results as existence proofs.
- 0 bugs: Most open-source demo configs use safe defaults (retry_count=1 or unset). This does not invalidate the tool — it means the problem is latent in configs with explicit resilience tuning. Discuss honestly: the value proposition targets production configs, which we cannot access. Shift emphasis to semi-synthetic and post-mortem case studies entirely.

---

## 2. Evaluation Axes with Detailed Methodology

### Axis 1: Semi-Synthetic Bug Detection (50% of evaluation effort)

**Rationale:** Semi-synthetic configs provide full ground-truth control — every injected bug has a known minimal failure set and known optimal repair. This is the only axis where precision, recall, and repair optimality can be rigorously measured.

#### Topology Generation

| Family | Parameters | Structure | Why included |
|--------|-----------|-----------|-------------|
| **Chain** | Depth d ∈ {3, 5, 7, 10} | Linear A→B→C→...→Z | Pure retry amplification (product along path) |
| **Tree** | Branching b ∈ {2, 3, 4}, depth d ∈ {3, 4, 5} | Root fans out, each child fans out | Timeout chain + fan-out amplification |
| **Mesh (random DAG)** | n services, edge probability p = 2ln(n)/n | Erdős–Rényi DAG (edges oriented by index) | Fan-in storms, shared dependencies |
| **Hub-and-spoke** | 1 hub, n−1 spokes, spoke→hub + hub→spoke edges | Star topology with bidirectional calls | Extreme fan-in on hub service |

**Scale:** 5, 10, 20, 30, 50 services per topology family.

**Total configurations:** 4 families × 5 scales × 10 random seeds = 200 base configs.

#### Parameter Sampling

All parameters are drawn independently per edge/service:

- `retry_count` ~ Uniform{1, 2, 3, 4, 5} (integers)
- `timeout_ms` ~ round(LogNormal(μ=ln(1000), σ=0.5)) — median ~1000ms, range ~400–2500ms
- `capacity` ~ max(10, round(Normal(μ=100, σ=20))) — requests/sec, floored at 10
- `entry_deadline_ms` ~ round(LogNormal(μ=ln(5000), σ=0.3)) — median ~5s, typical SLA

#### Bug Injection Protocol

Each configuration receives exactly one injected bug class (uniform random selection):

**Class 1 — Retry Bomb:** Select a path P of length ≥3. Set retry counts along P such that Π_retries > capacity of the terminal service. Ensure no single edge has retry_count > 5 (realistic constraint). The minimal failure set is: failure of the terminal service's downstream dependency (single failure).

**Class 2 — Timeout Chain:** Select a path P from entry to leaf. Set per-hop timeouts such that Σ(timeout_i × retry_count_i) > entry_deadline_ms. The cascade mechanism is silent deadline violation, not load amplification. Minimal failure set: slowdown (not failure) of any service on P.

**Class 3 — Fan-In Storm:** Select a service v with in-degree ≥2. Set parameters such that the *aggregate* retry load from all callers exceeds v's capacity, but *no single caller's* retry load exceeds v's capacity. This bug is invisible to per-path analysis (Tier 1). Minimal failure set: v fails, all callers retry simultaneously.

**Class 4 — Multi-Failure Cascade:** Construct a scenario requiring ≥2 simultaneous failures to trigger cascade. Service A depends on B and C; neither B-failure alone nor C-failure alone causes cascade, but {B, C} together cause A to exhaust retries against both, exceeding A's processing capacity. Only detectable by BMC with k≥2.

#### Ground Truth Specification

For each injected bug, the generator records:
- Minimal failure set F* (set of services that must fail)
- Cascade mechanism (retry amplification | timeout chain | fan-in storm)
- Cascade target (entry-point service that becomes unavailable)
- Optimal repair R* (minimum-cost parameter change eliminating the cascade)
- R* cost: Σ |original_i − repaired_i| × weight_i

#### Metrics

| Metric | Definition | Target |
|--------|-----------|--------|
| **Detection Recall** | |detected ∩ injected| / |injected| | ≥ 0.95 |
| **Detection Precision** | |detected ∩ injected| / |detected| | ≥ 0.90 |
| **Detection F1** | Harmonic mean of recall and precision | ≥ 0.92 |
| **Repair Soundness** | Fraction of repairs that pass re-verification | 1.00 |
| **Repair Minimality** | Mean parameters changed per repair | ≤ 3 |
| **Repair Optimality Gap** | Mean (CascadeVerify_cost − R*_cost) / R*_cost | ≤ 0.20 |

#### Statistical Design

- 200+ configurations, 5 independent random seeds per configuration for parameter sampling
- Report: mean ± standard deviation, median, and 95% confidence intervals via bootstrap (10,000 resamples)
- Stratify results by topology family and scale — report per-stratum to reveal systematic weaknesses
- Effect size: report Cohen's d for Tier 1 vs. Tier 2 detection rates

---

### Axis 2: Real Open-Source Configs (25% of effort)

#### Corpus Construction

| Project | Services | Source | Expected retry/timeout config |
|---------|----------|--------|-------------------------------|
| Google Online Boutique | 11 | github.com/GoogleCloudPlatform/microservices-demo | Some explicit timeout configs |
| Weaveworks Sock Shop | 14 | github.com/microservices-demo/microservices-demo | Minimal resilience config |
| Istio BookInfo | 4 | istio.io/docs/examples/bookinfo | VirtualService timeouts, retries |
| Helm charts from Artifact Hub | 15–20 | artifacthub.io (filtered for retry/timeout) | Varies widely |

**Total target:** 30–40 distinct service topologies.

#### Methodology

1. **Pre-filter:** Automatically scan for manifests containing explicit retry, timeout, or resilience keywords (`retries`, `timeout`, `perTryTimeout`, `retryOn`, `connectTimeout`, `idleTimeout`). Discard configs using only defaults.
2. **Run both tiers:** Execute Tier 1 (graph analysis) and Tier 2 (BMC) on every pre-filtered config.
3. **Manual verification:** For every reported cascade risk, two authors independently verify by tracing the call graph and computing retry amplification / timeout sums by hand. Disagreements resolved by discussion.
4. **Novelty check:** Search each project's GitHub issue tracker for previously reported cascade/retry/timeout issues.
5. **Linter comparison:** Run kube-score, `istioctl analyze`, and OPA with standard Rego rules on the same configs. Record what each tool reports.

#### Challenges and Mitigations

- **Default configs:** Most demo apps ship with retry_count=0 or unset (Envoy default: no retries). Pre-filtering eliminates these, but this reduces corpus size. Mitigation: supplement with Artifact Hub charts that explicitly configure resilience.
- **Simplistic topologies:** Demo apps have ≤15 services with shallow call graphs. Mitigation: this is why semi-synthetic is the primary axis; real configs demonstrate applicability, not comprehensive coverage.
- **No ground truth:** We cannot know all bugs in a real config. Report *confirmed* bugs (manually verified) without claiming completeness.

#### If < 3 Real Bugs Found

If the real-config corpus yields fewer than 3 confirmed bugs:
1. Report the finding honestly: "Open-source demo configurations predominantly use safe defaults."
2. Downgrade H6 to a directional claim without a specific count threshold.
3. Emphasize that the value proposition targets production configurations with explicit resilience tuning.
4. Ensure the semi-synthetic evaluation (Axis 1) carries the paper's primary claims.

---

### Axis 3: Baseline Comparison (15% of effort)

#### Baselines

**Baseline 1 — Graph-Analysis-Only (~2K LoC):**
- Product-of-retries along every path from entry to leaf
- Sum-of-timeouts (expanded by retries) along every path
- Capacity check: does worst-case retry amplification on any path exceed terminal service capacity?
- This is CascadeVerify's Tier 1 in isolation.

**Baseline 2 — Existing Linters:**
- `kube-score` v1.18+: scores K8s manifests, checks for missing resource limits, probes
- `istioctl analyze`: checks Istio configs for misconfigurations (missing sidecars, conflicting VirtualServices)
- OPA/Rego with Styra's default K8s rules
- *None of these reason about cross-service retry-timeout interactions*

**Baseline 3 — LDFI-Style Analysis (where applicable):**
- On semi-synthetic configs where we can generate execution traces (by simulating the topology), run a simplified LDFI analysis: inject failures, observe cascade in simulation, use SAT to find minimal failure sets from traces.
- Fair comparison requires: same failure model, same cascade definition.
- Note: LDFI operates on traces, not configs. This comparison is inherently asymmetric — LDFI requires a running system; CascadeVerify operates pre-deployment. Report this asymmetry explicitly.

#### Key Comparison Table

For each bug class, report:

| Bug class | CascadeVerify (Tier 1+2) | Tier 1 only | kube-score | istioctl analyze | OPA/Rego | LDFI-style |
|-----------|--------------------------|-------------|------------|-----------------|----------|------------|
| Retry bomb | ✓/✗ | ✓/✗ | ✓/✗ | ✓/✗ | ✓/✗ | ✓/✗ |
| Timeout chain | ✓/✗ | ✓/✗ | ✓/✗ | ✓/✗ | ✓/✗ | ✓/✗ |
| Fan-in storm | ✓/✗ | ✓/✗ | ✓/✗ | ✓/✗ | ✓/✗ | ✓/✗ |
| Multi-failure | ✓/✗ | ✓/✗ | ✓/✗ | ✓/✗ | ✓/✗ | ✓/✗ |

**Expected outcome:** CascadeVerify catches all four classes. Tier 1 catches retry bombs and timeout chains but misses fan-in storms and multi-failure. Existing linters catch none (they don't perform cross-service reasoning). LDFI-style catches what it can observe in traces.

---

### Axis 4: Scalability (10% of effort)

#### Experimental Setup

- **Machine:** MacBook Pro M2 (or specified equivalent), single-threaded execution
- **Topology sizes:** 5, 10, 20, 30, 50 services
- **Topology shapes:** chain, tree, mesh, hub-and-spoke (4 shapes × 5 sizes = 20 configurations)
- **Repetitions:** 5 random seeds per configuration = 100 total measurements
- **Timeout:** Hard cutoff at 600s; if exceeded, report as timeout

#### Measurements

For each configuration, record:

| Phase | What is measured |
|-------|-----------------|
| **Tier 1** | Wall-clock time for graph analysis (path enumeration, product/sum computation) |
| **Tier 2 (single query)** | Time for one BMC satisfiability query at bound k |
| **Tier 2 (full enumeration)** | Time to enumerate all minimal failure sets via MARCO |
| **Antichain-pruned enumeration** | Time for monotonicity-aware enumeration (same output as above) |
| **MaxSAT repair** | Time for repair synthesis per detected cascade risk |
| **Peak memory** | RSS high-water mark during full pipeline |
| **SMT variables** | Count of QF_LIA variables in the BMC formula |
| **SMT clauses** | Count of clauses in the CNF-level encoding |

#### Presentation

1. **Log-log plot** of end-to-end time vs. service count, one line per topology family. Fit power-law regression y = a·n^b; report exponent b with 95% CI. Label the inflection point where the curve transitions from "fast" to "slow."
2. **Stacked bar chart** showing time breakdown (Tier 1 / Tier 2 / repair) at each scale point.
3. **Honest scalability ceiling:** Report the largest topology size at which 95th percentile end-to-end time remains below 60s, 120s, and 300s. If the ceiling is < 50, state this clearly.
4. **SMT formula size plot:** Variables and clauses vs. service count, to characterize encoding growth rate.

---

### Axis 5: Repair Quality (5% of effort, bonus)

#### Metrics

| Metric | Definition |
|--------|-----------|
| **Parameters changed** | Count of retry_count or timeout_ms values modified |
| **Total weighted deviation** | Σ_i |original_i − repaired_i| × weight_i (weight reflects operational impact) |
| **Soundness** | Does the repaired config pass re-verification? (binary, target: 100%) |
| **Operational sensibility** | Are all repaired values within operator-specified bounds? (retry ∈ [0,5], timeout ∈ [100ms, 30s]) |

#### Baselines

- **Reset-to-defaults:** Set all retry_count=1, timeout=1000ms. Always sound but maximally invasive.
- **Uniform reduction:** Halve all retry counts and double all timeouts. Sound for retry bombs, unsound for timeout chains.
- **Random perturbation:** Randomly adjust parameters until verification passes. Report median attempts needed.

#### Visualization

For 5 representative examples (one per topology family + one real config), plot a **Pareto frontier** of repair minimality vs. robustness margin (the maximum additional failure budget before the repaired config cascades). Show CascadeVerify's MaxSAT repair as a point on or near the frontier.

---

## 3. Threats to Validity

### Internal Validity

| Threat | Severity | Mitigation |
|--------|----------|-----------|
| **Synthetic bug injection may not match real-world patterns** | High | Use parameter distributions calibrated from real configs; inject only bug classes observed in post-mortems |
| **Ground-truth generator may have bugs** | Medium | Cross-validate: verify ground-truth failure sets by independent BMC run with brute-force enumeration on small (n≤10) instances |
| **Z3 non-determinism** | Low | Fix random seed; report variance across seeds; use deterministic solver mode where available |
| **Measurement noise** | Low | 5 repetitions per config; report confidence intervals; run on quiescent machine with CPU frequency locked |

### External Validity

| Threat | Severity | Mitigation |
|--------|----------|-----------|
| **Open-source configs are simpler than production** | High | Acknowledged limitation. Semi-synthetic configs stress-test at production-like scales. Discuss explicitly in paper. |
| **Production configs are proprietary** | High | Cannot mitigate fully. If an industry partner provides sanitized configs, include them. Otherwise, semi-synthetic is primary. |
| **Config format evolution** | Low | Pin Kubernetes, Istio, Envoy versions. Test with 2 Istio versions (1.20, 1.22) to show format stability. |

### Construct Validity

| Threat | Severity | Mitigation |
|--------|----------|-----------|
| **"Cascade risk" is model-defined, not runtime-verified** | High | The BMC model uses conservative over-approximation. Any flagged risk *could* cascade under the model's assumptions. Acknowledge that runtime behavior depends on factors not modeled (network jitter, OS scheduling, GC pauses). |
| **Capacity model is simplified** | Medium | We model capacity as a static integer. Real capacity varies with load, resource contention, etc. The model catches worst-case risks under the static assumption. |
| **No circuit breakers in model** | Medium | Explicit scope limitation. State clearly and position CB-inclusive analysis as future work. |

### Statistical Validity

| Threat | Severity | Mitigation |
|--------|----------|-----------|
| **Small sample size for real configs (N<30)** | High | Use exact binomial confidence intervals, not normal approximation. Do not overclaim statistical significance on the real-config axis. |
| **Multiple hypothesis testing** | Low | 6 hypotheses; apply Bonferroni correction for any p-value claims (α = 0.05/6 ≈ 0.0083). For most hypotheses, we report confidence intervals rather than p-values. |

---

## 4. Concrete Predictions with Confidence Intervals

### H1 — Detection Effectiveness

| Metric | Point Estimate | 80% CI | If below CI |
|--------|---------------|--------|-------------|
| Recall | 0.97 | [0.94, 0.99] | Below 0.94: debug encoding for missed bug class. Below 0.90: fundamental encoding gap. |
| Precision | 0.93 | [0.88, 0.97] | Below 0.88: investigate over-approximation in load model. Tighten capacity bounds. |
| F1 | 0.95 | [0.91, 0.98] | Below 0.91: likely one bug class systematically missed. Stratify by class to isolate. |

**Reasoning:** Retry bombs and timeout chains are directly encoded in the BMC formulation; near-perfect detection expected. Fan-in storms require correct aggregation semantics. Multi-failure is the riskiest class — requires k≥2, which increases the SMT formula size substantially.

### H2 — BMC Marginal Value

| Metric | Point Estimate | 80% CI | If below CI |
|--------|---------------|--------|-------------|
| Marginal detection rate | 0.22 | [0.15, 0.30] | Below 0.15: graph analysis is surprisingly effective for monotone case. Reframe Tier 2 as "defense in depth." |

**Reasoning:** Fan-in storms and multi-failure cascades constitute ~50% of injected bugs, and Tier 1 should miss most of them. But some fan-in storms may be detectable by per-path analysis if the paths happen to individually exceed capacity. Estimated ~22% marginal value.

### H3 — Monotonicity Speedup

| Metric | Point Estimate | 80% CI | If below CI |
|--------|---------------|--------|-------------|
| Geometric mean speedup (20+ svcs) | 25× | [10×, 80×] | Below 10×: antichain pruning overhead is significant relative to savings. May need to optimize antichain data structure (e.g., use Zdd for antichain representation). |

**Reasoning:** Wide CI reflects uncertainty about the shape of the minimal failure set lattice. For chain topologies, the lattice is narrow (few minimal sets) and pruning helps less. For mesh topologies with high fan-in, the lattice is deep and pruning is essential.

### H4 — Repair Quality

| Metric | Point Estimate | 80% CI | If below CI |
|--------|---------------|--------|-------------|
| Mean params changed | 2.1 | [1.5, 2.8] | Above 2.8: MaxSAT soft-clause weights may need recalibration. Investigate per-bug-class. |
| Soundness | 1.00 | [1.00, 1.00] | Below 1.00: Hard-clause encoding bug. Critical priority fix. |
| Mean optimality gap | 0.08 | [0.03, 0.15] | Above 0.15: MaxSAT solver finding local optima. Try increasing solver timeout or alternative MaxSAT solver (e.g., RC2 → Open-WBO). |

### H5 — Scalability

| Config | Point Estimate | 80% CI | If above CI |
|--------|---------------|--------|-------------|
| 30-service end-to-end (median) | 12s | [6s, 25s] | Above 25s: bottleneck in MinUnsat enumeration. Cap at top-5 minimal sets for CI/CD mode. |
| 50-service end-to-end (median) | 45s | [20s, 80s] | Above 80s: scalability ceiling is real. Report honestly; propose compositional decomposition. |
| 50-service peak memory | 800 MB | [400 MB, 1.5 GB] | Above 1.5 GB: SMT formula is too large. Investigate cone-of-influence reduction aggressiveness. |

### H6 — Real-World Bugs

| Metric | Point Estimate | 80% CI | If below CI |
|--------|---------------|--------|-------------|
| Confirmed real bugs | 4 | [2, 8] | Below 2: open-source configs are too simple. Shift primary evidence to semi-synthetic. Report this as a finding about the state of OSS resilience configs. |

---

## 5. Minimum Viable Evaluation

If time or resources are constrained, the following is the *absolute minimum* evaluation that still supports the paper's core claims.

### Must Have (Paper cannot be submitted without these)

| Component | Effort | Supports |
|-----------|--------|----------|
| **Semi-synthetic bug detection (100 configs, 2 topology families)** | 3 weeks | H1, H2 |
| **Scalability measurements (chain + mesh, 5 sizes)** | 1 week | H5 |
| **Graph-analysis baseline comparison** | 1 week | H2, uniqueness of BMC contribution |
| **Repair soundness check (re-verification)** | 0.5 weeks | H4 (soundness only) |

**Minimum total: ~5.5 weeks.** This covers the two-tier architecture story (Tier 1 finds some bugs, Tier 2 finds more, repairs are sound, tool scales to 30+ services).

### Should Have (Paper is significantly stronger with these)

| Component | Effort | Supports |
|-----------|--------|----------|
| **Full semi-synthetic corpus (200+ configs, 4 families)** | +2 weeks | H1 (statistical power) |
| **Real open-source configs** | +2 weeks | H6 |
| **Monotonicity speedup measurement** | +1 week | H3 |
| **Linter baselines (kube-score, istioctl analyze)** | +0.5 weeks | Comparison story |

### Nice to Have (Best-paper-candidate additions)

| Component | Effort | Supports |
|-----------|--------|----------|
| **Post-mortem case studies** | +2 weeks | Narrative, real-world relevance |
| **LDFI-style comparison** | +2 weeks | Differentiation story |
| **Repair Pareto frontier visualization** | +1 week | H4 (full optimality analysis) |
| **Repair baselines (reset, uniform, random)** | +0.5 weeks | H4 (relative quality) |
| **Additional Istio versions** | +0.5 weeks | External validity |

### What to Cut First

1. **Cut Axis 5 (repair quality beyond soundness)** — soundness is the critical check; optimality gap is bonus.
2. **Cut LDFI comparison** — the asymmetry (static vs. dynamic) makes direct comparison inherently imperfect.
3. **Cut post-mortem case studies** — they are narrative, not quantitative. Lose the "real incident" story but retain all quantitative claims.
4. **Never cut the semi-synthetic evaluation or scalability measurements.** These are the backbone.

---

## 6. Post-Mortem Case Study Methodology

### Selected Incidents

| Incident | Date | Services involved | Root cause mechanism | Source |
|----------|------|-------------------|---------------------|--------|
| **AWS S3 outage** | Feb 2017 | S3 subsystems (index, placement, storage) | Retry storm during restart; subsystem retries amplified across dependencies | AWS post-mortem blog |
| **Google Cloud outage** | Jun 2019 | Multiple GCP services (Compute, App Engine, Cloud Console) | Configuration change triggered cascading failures across multiple systems | Google Cloud blog |
| **Cloudflare outage** | Jul 2019 | Edge servers, WAF, control plane | Regex-triggered CPU exhaustion cascaded through retry/failover mechanisms | Cloudflare blog |

### Methodology: Configuration Reconstruction

**Step 1 — Extract facts from prose.**
Read the post-mortem narrative and extract every stated or implied fact about:
- Service names and dependencies (call graph)
- Retry policies (counts, backoff, conditions)
- Timeout configurations (per-request, per-connection)
- Capacity limits (request rates, thread pools)
- The failure sequence described

Document each extracted fact with a direct quote from the post-mortem.

**Step 2 — Construct plausible configurations.**
For each extracted fact, create corresponding K8s/Istio/Envoy YAML:
- Where the post-mortem states a specific value (e.g., "3 retries"), use it exactly.
- Where the post-mortem describes behavior without specific values (e.g., "retries overwhelmed the system"), choose parameter values that are *consistent with the described behavior* and within typical ranges.
- Where the post-mortem is silent, use Istio/Envoy defaults.

**Step 3 — Validate the reconstruction.**
Verify that the constructed configuration, when analyzed by CascadeVerify, produces a cascade risk whose mechanism matches the post-mortem description. If it does not, adjust parameters within the plausible range until it does (documenting all adjustments).

**Step 4 — Run CascadeVerify and report.**
Report: (a) the detected cascade risk, (b) the minimal failure set, (c) the cascade mechanism, (d) the proposed repair, (e) whether the repair is consistent with the fix described in the post-mortem.

### Epistemic Boundaries

**What we CAN claim:**
- "Given a plausible configuration consistent with the described failure, CascadeVerify identifies a cascade risk matching the post-mortem's root cause description."
- "The tool's proposed repair is consistent with the mitigation described in the post-mortem."
- "This demonstrates that CascadeVerify's model captures the failure mechanisms described in real incidents."

**What we CANNOT claim:**
- "CascadeVerify would have prevented this outage." (We don't know the exact configuration.)
- "This is the configuration that was running." (Post-mortems omit many details.)
- "The tool detected the exact same failure." (Our model is an abstraction.)

**Framing in the paper:** Section title: "Case Studies: Plausible Configurations from Public Post-Mortems." Opening paragraph must state: "We reconstruct plausible service configurations consistent with publicly described failure mechanisms. These case studies demonstrate that CascadeVerify's cascade model captures real-world failure patterns; they do not claim to reproduce the exact conditions of each outage."

### Documentation Requirements

For each case study, the paper must include:
1. A table mapping post-mortem quotes to configuration parameters
2. The complete reconstructed YAML (in appendix or artifact)
3. CascadeVerify's output: detected risk, failure set, mechanism, repair
4. Explicit listing of all assumptions made beyond the post-mortem text
5. Sensitivity analysis: how much can the assumed parameters vary before the cascade risk disappears?

---

## Appendix A: Reproducibility Artifact Checklist

Per NSDI artifact evaluation criteria:

- [ ] All semi-synthetic topology generator code with fixed random seeds
- [ ] Complete real-config corpus with provenance documentation
- [ ] Post-mortem reconstruction configs with assumption documentation
- [ ] Baseline implementations (graph-analysis, reset-to-defaults repair)
- [ ] Measurement scripts with exact machine specification
- [ ] Raw timing data in CSV format
- [ ] Statistical analysis scripts (R or Python) producing all figures and tables
- [ ] Docker container with pinned Z3 version for reproducible solver behavior
- [ ] README with single-command reproduction instructions

## Appendix B: Timeline Estimate

| Week | Activity | Deliverable |
|------|----------|-------------|
| 1–2 | Implement semi-synthetic topology generator | Generator code + 200 configs |
| 3–4 | Implement graph-analysis baseline | Baseline 1 code + initial comparison |
| 5–6 | Run full semi-synthetic evaluation | H1, H2 results |
| 7 | Scalability measurements | H3, H5 results |
| 8 | Real-config corpus assembly + analysis | H6 results |
| 9 | Post-mortem case studies | 3 reconstructions |
| 10 | Repair quality evaluation + baselines | H4 results |
| 11 | Statistical analysis + figure generation | All plots and tables |
| 12 | Write evaluation section of paper | Draft evaluation text |

## Appendix C: Decision Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Primary evaluation axis | Semi-synthetic | Full ground-truth control; real configs lack ground truth |
| Statistical method | Bootstrap CIs | Non-parametric; no distribution assumptions needed |
| Scalability presentation | Log-log regression | Reveals power-law scaling behavior honestly |
| Post-mortem framing | "Plausible configurations" | Epistemically honest; avoids overclaiming |
| Minimum config corpus size | 200 | Provides ≥80% statistical power for detecting 5% recall differences |
| Real config target | 30–40 | Constrained by availability of configs with explicit resilience settings |
| Repair optimality benchmark | Exhaustive search on bounded parameter space | Only feasible for small parameter spaces; sufficient for our domain |
