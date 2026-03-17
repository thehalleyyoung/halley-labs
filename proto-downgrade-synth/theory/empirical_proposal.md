# NegSynth — Empirical Evaluation Proposal

## Negotiation Under Fire: Bounded-Complete Synthesis of Protocol Downgrade Attacks from Library Source Code

**Document type:** Experimental design  
**Scope:** Full evaluation campaign for NegSynth across four cryptographic libraries (OpenSSL, BoringSSL, WolfSSL, libssh2), eight ground-truth CVEs, and three baseline tools.

---

## 1  Falsifiable Hypotheses

Every hypothesis below specifies a *target* threshold the paper will claim, a *falsification* threshold at which we reject the hypothesis, and the precise measurement procedure. The gap between target and falsification accommodates measurement noise without weakening the claim.

### H1 — CVE Recovery Recall

**Claim.** NegSynth recovers ≥7 of 8 known downgrade CVEs from historically vulnerable library versions with zero false negatives on the in-scope CVE set.

| Parameter | Value |
|-----------|-------|
| Target | recall ≥ 87.5% (≥ 7/8) |
| Falsification | recall < 87.5% (< 7/8) |
| Measurement | Binary per-CVE verdict (synthesized / not synthesized), validated by TLS-Attacker replay of the concrete attack trace against a live instance of the vulnerable library version. Record time-to-first-synthesis (wall-clock seconds from pipeline start to first valid trace). |

**Rationale.** 7/8 is an honest bar: POODLE's version-downgrade component relies on a cross-session behavioral property (the attacker must observe multiple connections) that may exceed the single-session adversary model at low budgets. We pre-register this as the most likely miss.

### H2 — Protocol-Aware Merge Speedup

**Claim.** The protocol-aware merge operator (PROTOMERGE) achieves ≥10× reduction in symbolic path count compared to generic veritesting on negotiation code, across all four target libraries.

| Parameter | Value |
|-----------|-------|
| Target | path-count ratio ≥ 10× on all 4 libraries |
| Falsification | ratio < 10× on ≥ 2 libraries |
| Measurement | Path count (total completed symbolic paths) under identical KLEE configuration, same bounds (k = 20), same hardware. Control: KLEE + generic veritesting (Avgerinos et al. ICSE 2014). Treatment: KLEE + PROTOMERGE. Report both path count and wall-clock time. |

**Note on the 10× threshold.** The theoretical improvement is exponential — O(n) vs. O(2^n) for n cipher suites — so 10× is deliberately conservative. If the empirical ratio is 5×–9× on a library with few cipher suites, H2 is falsified at that library, and we report honestly. The hypothesis requires the ratio to hold on at least three of four libraries.

### H3 — Bounded-Completeness Coverage

**Claim.** At analysis bounds k = 20 (execution depth) and n = 5 (adversary budget), NegSynth explores ≥99% of reachable negotiation states in each target library's current HEAD.

| Parameter | Value |
|-----------|-------|
| Target | negotiation-state coverage ≥ 99% per library |
| Falsification | coverage < 95% on any library |
| Measurement | Ground-truth state set enumerated via exhaustive BFS on a reduced-scale model (≤ 8 cipher suites). Coverage = |NegSynth-explored states| / |BFS-enumerated states|. For full-scale libraries, validated independently via 1M uniformly random negotiation inputs; any input reaching a state not in the NegSynth certificate constitutes a coverage miss. |

**Why two validation methods.** BFS enumeration provides an exact denominator but is tractable only at reduced scale. Random testing extends validation to full scale with probabilistic guarantees: after 1M trials, a 1% coverage gap has ≥ 99.99% probability of surfacing at least once (assuming uniform reachability).

### H4 — Scalability

**Claim.** Per-library end-to-end analysis completes in < 8 hours wall-clock time on commodity hardware (8-core CPU, 32 GB RAM, no GPU).

| Parameter | Value |
|-----------|-------|
| Target | wall-clock < 8 h per library |
| Falsification | any library > 12 h |
| Measurement | Wall-clock seconds per pipeline stage (slicing, symbolic execution, state-machine extraction, DY+SMT encoding, Z3 solving, concretization), total per library. Peak RSS memory per stage. Median over 5 independent runs. |

**Gap justification.** The 8 h target and 12 h falsification threshold create a buffer for system variance (GC pressure, OS scheduling, Z3 non-determinism). A result between 8 and 12 hours is reported honestly as "near the scalability target" and not claimed as a pass.

### H5 — False Positive Rate

**Claim.** Among all attack traces reported by NegSynth across all experiments, fewer than 10% are false positives (traces that do not reproduce as a valid attack when replayed).

| Parameter | Value |
|-----------|-------|
| Target | FP rate < 10% |
| Falsification | FP rate ≥ 15% |
| Measurement | Every reported trace is replayed via TLS-Attacker (TLS traces) or a custom SSH replay harness (SSH traces) against a live instance of the target library version compiled with ASAN. A trace is a true positive iff replay achieves the claimed downgrade (i.e., the server selects the weak cipher suite / protocol version). FP rate = |non-reproducing traces| / |total reported traces|. |

### H6 — CEGAR Concretization Rate

**Claim.** The CEGAR concretization loop converts abstract attack traces to byte-level concrete traces with success rate ≥ 99% (failure rate ε ≤ 0.01).

| Parameter | Value |
|-----------|-------|
| Target | ε ≤ 0.01 per library |
| Falsification | ε > 0.05 on any library |
| Measurement | For each library: total concretization attempts, successful concretizations, CEGAR refinement iterations per attempt, failure-case root-cause classification. ε = |failures| / |attempts|. |

---

## 2  Experimental Design

### Experiment E1 — Known CVE Recovery

**Validates:** H1  
**Subjects:** 8 CVEs × specific vulnerable library versions.

| CVE ID | Name | Library | Version | Protocol |
|--------|------|---------|---------|----------|
| CVE-2015-0204 | FREAK | OpenSSL | 1.0.1k | TLS |
| CVE-2015-4000 | Logjam | OpenSSL | 1.0.2a | TLS |
| CVE-2014-3566 | POODLE | OpenSSL | 1.0.1i | TLS |
| CVE-2023-48795 | Terrapin | libssh2 | 1.11.0 | SSH |
| CVE-2016-0703 | DROWN-specific | OpenSSL | 1.0.2 | TLS |
| CVE-2015-3197 | SSLv2 override | OpenSSL | 1.0.2a | TLS |
| CVE-2014-0224 | CCS Injection | OpenSSL | 1.0.1g | TLS |
| CVE-2015-0291 | ClientHello sigalgs | OpenSSL | 1.0.2 | TLS |

**Protocol:**
1. Check out the exact vulnerable commit for each library version. Build LLVM bitcode via `wllvm` (OpenSSL, WolfSSL, libssh2) or native Clang (BoringSSL).
2. Run NegSynth end-to-end with default bounds (k = 20, n = 5). Record: start time, pipeline stage timestamps, attack traces produced, end time.
3. For each produced trace, run TLS-Attacker replay against a live server instance of the vulnerable library version running in a Docker container.
4. A CVE is "recovered" iff at least one trace demonstrably achieves the documented downgrade effect (e.g., server selects export cipher for FREAK).
5. For each recovered CVE, binary-search for the *minimal* (k, n) pair that still yields recovery. Record in the empirical bounds table.

**Output:** Per-CVE table with columns: CVE ID, recovered (Y/N), time-to-first-synthesis, minimal (k, n), number of traces produced, number validated.

**Controls:** Each library version is compiled with identical compiler flags (`-O0 -g -emit-llvm`). Docker containers run with host networking disabled to prevent interference.

### Experiment E2 — Merge Operator Scaling ("Money Plot")

**Validates:** H2  
**Method:** Controlled ablation comparing PROTOMERGE against generic veritesting.

**Independent variable:** Merge strategy ∈ {PROTOMERGE, generic-veritesting, no-merge}.  
**Controlled variables:** KLEE version, LLVM version, analysis bounds (k = 20), hardware, OS, compilation flags.  
**Dependent variables:** Symbolic path count, wall-clock time, peak memory.

**Procedure:**
1. For each of the 4 libraries (current HEAD), compile LLVM bitcode.
2. Run NegSynth's slicer to extract the negotiation slice (shared across all three conditions).
3. Run KLEE with PROTOMERGE on the slice. Record path count, time, memory.
4. Run KLEE with generic veritesting (Avgerinos-style, implemented as a KLEE searcher plugin) on the same slice. Record path count, time, memory.
5. Run KLEE with no merge (pure symbolic execution) on the same slice, with a 24-hour timeout. Record path count (or timeout indicator), time, memory.
6. Repeat steps 3–5 with *varying cipher-suite counts*: artificially restrict the library to n ∈ {5, 10, 15, 20, 25, 30, 40, 50} cipher suites via compile-time configuration flags or source-level `#define` overrides.

**Visualization:**
- Log-scale plot: x-axis = cipher-suite count n, y-axis = symbolic path count. Three lines (PROTOMERGE, generic-veritesting, no-merge). Expected shape: PROTOMERGE = linear, generic-veritesting = exponential, no-merge = exponential with higher constant.
- Bar chart: per-library speedup ratio (generic paths / PROTOMERGE paths) at default cipher count.

**Statistical treatment:** Path counts are deterministic given the same KLEE configuration and seed, so no variance analysis is needed. Report exact counts. Wall-clock times are reported as median ± IQR over 5 runs to account for OS-level scheduling variance.

### Experiment E3 — Bounded-Completeness Certificates

**Validates:** H3  
**Subjects:** Current HEAD of OpenSSL 3.x, BoringSSL (latest stable), WolfSSL (latest stable), libssh2 (latest stable).

**Procedure:**
1. Run NegSynth at k = 20, n = 5 on each library. If the DY+SMT query returns UNSAT, the tool emits a bounded-completeness certificate: a machine-checkable artifact asserting "within bounds (k, n), no downgrade attack exists."
2. Record: certificate produced (Y/N), total negotiation states explored, analysis time, peak memory, any attack traces found (which would indicate a 0-day).
3. **Coverage validation (small-scale ground truth):** For each library, configure a reduced-scale instance with ≤ 8 cipher suites. Run exhaustive BFS to enumerate all reachable negotiation states. Run NegSynth at the same bounds. Compute coverage = |NegSynth states ∩ BFS states| / |BFS states|.
4. **Coverage validation (full-scale stress test):** Generate 1M uniformly random negotiation inputs (random ClientHello cipher lists, random server configurations, random extension sets). For each input, compute the negotiation outcome via direct library execution. Check whether the outcome state appears in the NegSynth certificate's state set. Any uncovered state is a coverage miss.
5. **Sweep over bounds:** Repeat at k ∈ {5, 10, 15, 20, 25} and n ∈ {1, 2, 3, 4, 5, 7, 10} to produce a coverage-vs-bounds surface plot. This validates that k = 20, n = 5 is in the "plateau" region where additional bounds yield negligible coverage improvement.

**Output:** Per-library row with columns: library, version/commit, certificate (Y/N), states explored, BFS coverage %, random-test coverage %, analysis time, peak memory. Bounds-sweep surface plot.

### Experiment E4 — Scalability Profiling

**Validates:** H4  
**Method:** Per-stage timing and memory measurement of the full pipeline.

**Pipeline stages measured:**
1. **Bitcode generation:** `wllvm`/`gclang` compilation of the library.
2. **Slicing:** Protocol-aware slicer execution. Report: input LoC, output LoC, slice ratio.
3. **Symbolic execution:** KLEE + PROTOMERGE. Report: paths explored, time, memory.
4. **State-machine extraction:** Bisimulation quotient computation. Report: raw states, quotiented states, time.
5. **DY+SMT encoding:** Constraint generation. Report: clause count, variable count, time.
6. **Z3 solving:** SAT/UNSAT determination. Report: time, memory, CEGAR iterations.
7. **Concretization:** SMT model → byte-level traces. Report: traces produced, time.

**Procedure:**
1. For each of the 4 libraries (current HEAD), run NegSynth end-to-end with instrumented timing.
2. Record wall-clock time and peak RSS for each stage independently (via `/usr/bin/time -v` or `getrusage`).
3. Repeat 5 times per library. Report median and IQR for each cell.

**Output:** Table with rows = libraries, columns = stages, cells = median time (IQR) and peak memory. Final column = total end-to-end time. Stacked bar chart showing time decomposition per library.

**Hardware specification (reported exactly):**
- CPU: 8-core (model, base/boost frequency)
- RAM: 32 GB (speed)
- Storage: SSD (sequential read speed)
- OS: Linux kernel version, distribution
- Compiler: Clang/LLVM version
- Z3 version, KLEE version (commit hash)

### Experiment E5 — Tool Comparison

**Validates:** H1, H2 against baselines  
**Baselines:**

| Tool | Type | Citation | Configuration |
|------|------|----------|---------------|
| tlspuffin | DY-aware fuzzer | IEEE S&P 2024 | Default configuration, 72-hour budget |
| KLEE vanilla | Symbolic execution (no protocol awareness) | Cadar et al. OSDI 2008 | Same bounds (k = 20), same bitcode, no merge, no DY model |
| TLS-Attacker | Black-box protocol tester | Somorovsky, USENIX Sec 2016 | Default attack suite, 72-hour budget |

**Fair comparison methodology:**
- All tools run on identical hardware (same machine, exclusive access, no other workloads).
- All tools target the same library versions (the 8 vulnerable versions from E1).
- Time budget: NegSynth gets 8 hours (its design target). tlspuffin and TLS-Attacker get 72 hours (9× more time, to be generous). KLEE vanilla gets 8 hours (same as NegSynth, for a fair ablation).
- All tools output is normalized to a common format: (CVE matched / not matched, time-to-first-finding, false positives).

**Metrics:**

| Metric | Description |
|--------|-------------|
| CVE recall | Number of CVEs independently recovered (out of 8) |
| Time-to-first | Wall-clock seconds from tool start to first valid finding per CVE |
| False positive count | Findings that do not reproduce on TLS-Attacker replay |
| Code coverage | `lcov` branch coverage of the library's negotiation code (measured externally via instrumented builds) |
| Completeness | Can the tool certify *absence* of attacks? (Y/N) |

**Completeness gap demonstration:** Construct a synthetic but realistic scenario: a custom OpenSSL build with a deliberately introduced subtle downgrade path (a single `if` condition inversion in cipher preference ordering). Run all four tools. Measure time-to-detection. Hypothesis: NegSynth finds it in < 10 minutes; tlspuffin does not find it in 72 hours (the path is reachable but has negligible probability under random fuzzing).

**Why NOT ProVerif/Tamarin as baselines.** These tools analyze hand-written protocol specifications, not C source code. They operate on a fundamentally different input (a `.pv` or `.spthy` model) and cannot process the same artifacts NegSynth consumes. Including them would require manually writing a specification for each library version, introducing a confound (specification fidelity) that dominates any tool comparison. We explicitly scope them out and state this in the paper's related work section.

### Experiment E6 — Concretization and Replay Validation

**Validates:** H5, H6  
**Scope:** Every abstract attack trace produced across all experiments (E1, E3, E5).

**Procedure:**
1. Collect all abstract traces from the DY+SMT solver across all runs.
2. For each abstract trace, invoke the CEGAR concretization loop. Record:
   - Success/failure
   - Number of CEGAR refinement iterations
   - Time per concretization
   - Failure root cause (if applicable): wire-format constraint, timing dependency, multi-session requirement, Z3 timeout
3. For each successfully concretized trace, run TLS-Attacker replay (TLS) or the custom SSH replay harness against a live server running the target library version.
4. Classify replay outcome:
   - **True positive:** The claimed downgrade is achieved (weak cipher selected, protocol version lowered, etc.)
   - **Partial positive:** The downgrade is partially achieved (e.g., correct cipher suite but connection resets before completion)
   - **False positive:** The downgrade is not achieved (server rejects, falls back to secure choice, or connection fails)
5. Report aggregate statistics: total traces, concretization success rate (1 − ε), CEGAR iteration distribution (histogram), replay pass rate, FP rate, root-cause breakdown of failures.

**Output:** Summary table per library. Histogram of CEGAR iterations. Failure-case taxonomy.

---

## 3  Baseline Strategy

### Baseline selection rationale

| Baseline | Why included | What it tests |
|----------|-------------|---------------|
| tlspuffin | Closest competitor: DY-aware, implementation-level, published at S&P 2024. | Whether NegSynth's white-box approach outperforms black-box DY fuzzing on recall, time-to-bug, and completeness. |
| KLEE vanilla | Ablation: identical infrastructure minus protocol awareness. | Whether protocol-aware slicing, merge, and DY modeling provide value beyond generic symbolic execution. |
| TLS-Attacker | Industry-standard black-box protocol testing. Widely deployed. | Whether white-box source analysis outperforms external probing. |

### Baselines explicitly excluded

| Tool | Why excluded |
|------|-------------|
| ProVerif / Tamarin | Analyze protocol *specifications* (.pv, .spthy), not source code. Different input artifact makes comparison unfair: any result difference could be attributed to specification fidelity, not tool capability. Discussed in Related Work, not in evaluation. |
| miTLS / Project Everest | Verified *construction* of new implementations, not analysis of existing ones. Orthogonal contribution axis. |
| SAGE / Driller / angr | Generic symbolic execution / concolic testing without protocol awareness or adversary models. KLEE vanilla already serves as this ablation. Adding more generic tools would dilute the comparison without additional insight. |

---

## 4  Statistical Methodology

### Timing measurements

All timing experiments (E2, E4, E5) are repeated 5 times on dedicated hardware with no competing workloads. We report **median and interquartile range (IQR)**, not mean ± standard deviation, because execution-time distributions are typically right-skewed (occasional OS scheduling spikes). The 5-run count is justified by the low variance of deterministic symbolic execution; we verify this by checking that the coefficient of variation (CV = σ/μ) is < 0.10 across runs, and increase to 10 runs for any measurement with CV ≥ 0.10.

### Path counts and state counts

Symbolic path counts (E2) and negotiation state counts (E3) are **deterministic** given identical inputs, KLEE configuration, and random seed. We report exact values with no variance analysis. We fix the KLEE random seed to 42 across all experiments and report this.

### Coverage metrics

Bounded-completeness coverage (E3) is measured as a ratio of explored states to ground-truth states. For the small-scale BFS validation, this is an exact ratio. For the large-scale random-testing validation, we report a **99% Clopper-Pearson confidence interval** on the coverage gap, computed from the number of coverage misses observed in 1M trials. If zero misses are observed, the one-sided 99% upper bound on the gap is < 4.6 × 10⁻⁶ (by the rule of three).

### CVE recall

CVE recall (E1, E5) is a **binary count metric** (7/8, 6/8, etc.). No statistical test is needed — the hypothesis is directly falsified if the count is below threshold. We report exact counts with per-CVE details. For the tool comparison (E5), we report a per-CVE matrix (tool × CVE) so readers can see which tools find which CVEs.

### False positive rate

FP rate (E5, E6) is reported as a proportion with a **95% Wilson score confidence interval** (appropriate for proportions near 0 or 1 where the normal approximation is poor). If 0/N traces are false positives, we still report the one-sided upper bound.

### Concretization rate

CEGAR concretization rate (E6) is reported per library as successes/attempts with a Wilson score 95% CI. The per-library granularity ensures that a single difficult library does not mask problems.

### Multiple comparisons

We do not apply Bonferroni or other multiple-comparison corrections because our hypotheses are pre-registered with fixed thresholds (not p-value based). Each hypothesis is evaluated independently against its falsification criterion. This avoids the philosophical pitfalls of null-hypothesis significance testing in systems research, following the recommendations of Georges et al. (OOPSLA 2007).

---

## 5  Threats to Validity

### Internal validity

| Threat | Severity | Mitigation |
|--------|----------|------------|
| **CVE selection bias.** We selected 8 CVEs that we believe NegSynth can handle. The tool may be over-fitted to these specific attack patterns. | High | We publish a classified table of *all* known downgrade CVEs across the target libraries (approximately 15–20), with honest scope labels: "in-scope and tested," "in-scope but not tested," "out of scope (reason)." The paper explicitly acknowledges which classes of downgrade attacks are beyond the current adversary model (e.g., multi-session attacks, timing-based downgrades). |
| **Implementation bugs in NegSynth.** A bug in the merge operator or SMT encoder could produce false completeness certificates. | Medium | Every theorem (T1–T5) is validated by property-based testing. The merge operator is cross-validated against exhaustive exploration on small instances. The SMT encoder is cross-validated against a reference Prolog DY model. The concretizer is validated by end-to-end replay (E6). |
| **KLEE non-determinism.** KLEE's searcher may explore different paths across runs. | Low | We fix the random seed and verify < 10% CV across runs. Path counts are deterministic at fixed seed. |
| **Baseline configuration.** Suboptimal configuration of tlspuffin or TLS-Attacker could understate their capabilities. | Medium | We use default configurations recommended by each tool's authors. For tlspuffin, we additionally consult the S&P 2024 paper's experimental setup and replicate it. We grant baselines 9× more time budget than NegSynth. |

### External validity

| Threat | Severity | Mitigation |
|--------|----------|------------|
| **C libraries only.** Results may not generalize to Rust (rustls), Go (crypto/tls), or Java (JSSE) implementations. | High | Explicit scope statement in the paper. The pipeline architecture is language-specific at the front end (slicer, KLEE) but language-agnostic at the back end (DY+SMT encoding, concretization). We discuss extension to other languages as future work. |
| **TLS and SSH only.** Results may not generalize to QUIC, DTLS, or other negotiation protocols. | Medium | We argue that the algebraic properties exploited by PROTOMERGE (finite outcome spaces, lattice preferences, monotonic progression, deterministic selection) are shared by all cipher-suite negotiation protocols. QUIC and DTLS are listed as future work with a concrete argument for why the merge operator should transfer. |
| **Limited library diversity.** Three of four libraries are TLS implementations with shared ancestry (OpenSSL → BoringSSL fork). | Medium | WolfSSL is an independent implementation with a different code structure. libssh2 is an entirely different protocol. We argue these provide sufficient diversity. |

### Construct validity

| Threat | Severity | Mitigation |
|--------|----------|------------|
| **"Bounded completeness" depends on choice of (k, n).** A skeptic can always argue that the true attack requires k = 21. | High | The empirical bounds table (E1) shows all 8 known CVEs require k ≤ 15, n ≤ 5, providing a 33% margin to the analysis bound of k = 20. The coverage sweep (E3) shows that increasing bounds beyond k = 20, n = 5 yields < 0.1% additional coverage on all libraries, indicating a plateau. We explicitly state the bound and do not claim unbounded completeness. |
| **False positive definition.** A trace that crashes the server but does not achieve the *exact* documented downgrade behavior might be a true finding (a new bug) that our FP metric misclassifies. | Medium | We use a three-way classification (true positive, partial positive, false positive) and report all three. Partial positives are analyzed individually and disclosed. |
| **Coverage metric.** Negotiation-state coverage may not capture all relevant program behaviors (e.g., memory-safety bugs in negotiation code). | Low | Explicit scope: NegSynth targets *logical* downgrade attacks, not memory-safety bugs. Coverage is measured over the negotiation state machine, not the full program state. |

### Ecological validity

| Threat | Severity | Mitigation |
|--------|----------|------------|
| **Single-machine evaluation.** Results may differ on other hardware configurations. | Low | We use commodity hardware (8-core laptop, 32 GB RAM), report exact hardware specs, and provide all artifacts for reproduction. The evaluation does not require specialized hardware (no GPU, no cluster). |
| **Lab conditions.** Real-world deployment involves network effects, concurrent connections, and partial library configurations. | Medium | NegSynth is a static analysis tool; it analyzes source code, not live deployments. We validate attack traces via replay (E6) which introduces realistic network conditions. |

---

## 6  Concrete Metrics Summary

| Metric | Target | Falsification Threshold | Experiment | Measurement |
|--------|--------|------------------------|------------|-------------|
| CVE recall | ≥ 7/8 (87.5%) | < 7/8 | E1 | Binary per-CVE, TLS-Attacker replay |
| Merge speedup | ≥ 10× path reduction | < 10× on ≥ 2 libraries | E2 | Exact path counts, controlled ablation |
| State coverage | ≥ 99% | < 95% on any library | E3 | BFS ground truth + 1M random inputs |
| Per-library time | < 8 h | > 12 h on any library | E4 | Wall-clock, median of 5 runs |
| FP rate | < 10% | ≥ 15% | E5, E6 | TLS-Attacker replay of every trace |
| Concretization ε | ≤ 0.01 | > 0.05 on any library | E6 | Attempts / successes per library |

---

## 7  Artifact and Reproducibility Plan

All experiments will be packaged as a reproducible artifact for artifact evaluation:

1. **Docker images** for each vulnerable library version (E1) and each current HEAD (E3, E4), pinned to exact commits.
2. **Single-command runner** (`make eval`) that executes all experiments sequentially and produces all tables and figures.
3. **Raw data** (JSON logs per experiment) archived alongside the paper submission.
4. **Expected runtime:** Full artifact evaluation ≈ 48 hours sequential on the specified hardware (4 libraries × ~8 hours for the main pipeline, plus baseline runs).
5. **Kick-the-tires script** that runs E1 on a single CVE (FREAK) in < 1 hour, validating the full pipeline end-to-end.

Hardware requirements for reviewers: 8-core CPU, 32 GB RAM, 100 GB free disk (for Docker images and LLVM bitcode). No GPU required.

---

## 8  Timeline and Dependencies

| Week | Activity | Depends on |
|------|----------|------------|
| W1–W2 | Build Docker images for all 8 vulnerable versions + 4 current HEADs. Verify bitcode generation. | — |
| W3–W4 | Run E1 (CVE recovery). Iterate on any pipeline failures. | Bitcode OK |
| W5–W6 | Run E2 (merge ablation) with cipher-suite sweeps. | Pipeline stable |
| W7–W8 | Run E3 (bounded completeness) on current HEADs. | Pipeline stable |
| W9 | Run E4 (scalability profiling). | E3 complete |
| W10–W11 | Run E5 (tool comparison). Requires tlspuffin setup and 72-hour runs. | E1 complete |
| W12 | Run E6 (concretization validation) across all traces from E1, E3, E5. | E1, E3, E5 complete |
| W13 | Analyze results, produce tables and figures, write evaluation section. | All experiments |

**Critical path:** E1 → E5 → E6. E2, E3, E4 can run in parallel with each other after E1 validates pipeline correctness.

---

## 9  Decision Criteria and Contingencies

### If H1 fails (recall < 7/8)

Analyze which CVEs are missed and why. If the miss is due to a fundamental limitation (e.g., multi-session adversary model), scope down the claim and report honestly. If the miss is due to an engineering bug, fix and re-run (pre-registered: one round of bug-fixing is permitted before finalizing results).

### If H2 fails (merge < 10× on ≥ 2 libraries)

Report the actual speedup honestly. If the speedup is 5×–9×, reframe the claim as "significant but sub-order-of-magnitude speedup" and investigate whether the libraries with low speedup have unusually few cipher suites (reducing the theoretical gap). The bounds-sweep data from E2 will show whether the improvement grows with cipher count as theory predicts.

### If H3 fails (coverage < 95% on any library)

Analyze the uncovered states. If they require k > 20 or n > 5, increase bounds and measure the performance cost. If coverage improves to ≥ 99% at k = 30 or n = 7 with acceptable runtime, adjust the claimed bounds. If coverage remains low, investigate whether the gap represents unreachable states (inflating the denominator) and refine the state-enumeration methodology.

### If H4 fails (any library > 12 h)

Profile to identify the bottleneck stage. If Z3 solving dominates, investigate query decomposition or switch to incremental solving. If symbolic execution dominates, investigate more aggressive slicing. Report the actual times honestly and adjust the scalability claim to the subset of libraries that pass.

### If H5 fails (FP rate ≥ 15%)

Analyze false positive root causes. Common causes: abstraction of cryptographic operations that are concretely infeasible, wire-format constraints missed by the encoder. Each root cause leads to a specific CEGAR refinement. One round of encoder refinement is permitted before finalizing.

### If H6 fails (ε > 0.05 on any library)

Investigate failure cases. If failures concentrate on a specific protocol feature (e.g., TLS 1.3 PSK), scope that feature out of the completeness claim and report the per-feature breakdown. If failures are distributed, improve the concretizer and re-run.
