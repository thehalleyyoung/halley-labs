# EMPIRICAL CLAIMS AUDIT REPORT
## SafeStep: Verified Deployment Plans Project

---

## EXECUTIVE SUMMARY

**🚨 CRITICAL HALLUCINATION DETECTED:** The tool_paper.tex abstract contains a **FACTUALLY INCORRECT** claim about performance on 200-service clusters. The abstract claims 0.017 seconds, but the paper's own table (Tab:scaling) shows 1.704 seconds — a **100x discrepancy**. The synthetic benchmark data shows an even larger discrepancy: 42,600 milliseconds (42.6 seconds).

**⚠️ SUSPICIOUS PATTERN:** All baseline algorithms in the real benchmark results show exactly 10% success rate (2 of 20 runs), which is suspiciously uniform.

**⚠️ CRITICAL DATA QUALITY ISSUE:** Synthetic benchmark results (results.json) show performance degradation 25x worse than claimed in the paper for 200-service scenarios.

---

## SECTION 1: TIMING CLAIMS - MAJOR DISCREPANCIES

### Claim 1: Abstract states "200-service clusters (0.017\,s)"

**Sources of the same measurement:**
1. **tool_paper.tex (Line 76, Abstract):** Claims 200-service clusters execute in 0.017 seconds
2. **tool_paper.tex (Table 1 line 1034):** Claims 20 services, 10 versions = 0.017 seconds  
3. **tool_paper.tex (Table 2, but unclear which row refers to 200):** Mix of different test configurations
4. **benchmarks/results.json (xxl scenario):** Shows 42,600 ms = 42.6 seconds for 200 services

### ❌ DISCREPANCY #1: TIMING FOR 200 SERVICES

| Source | Scenario | Services | Versions | Time Claimed | Notes |
|--------|----------|----------|----------|--------------|-------|
| Paper Abstract (Line 76) | Production deployment | 200 | 20 | **0.017s** | Likely copy-paste error from 20-service row |
| Paper Table 1 (Line 1034) | Scaling benchmark | 20 | 10 | 0.017s | Correct for this size |
| Paper Table 1 (Line 1038) | Scaling benchmark | 200 | 20 | **1.704s** | Matches theoretical O(n²log²L) complexity |
| results.json (xxl) | Synthetic eval | 200 | 20 | **42,600ms = 42.6s** | **25x worse than paper table** |

**VERDICT:** The abstract is WRONG. It repeats the 0.017s timing from the 20-service row but attributes it to 200 services. The correct timing for 200 services in the paper's own Table 1 is **1.704 seconds**, and the synthetic results show **42.6 seconds**.

### ✅ Confirmed Timings (Consistent):
- 10 services, 5 versions: **0.004s** (paper) vs **4.1ms** (synthetic) ✓ Match
- 20 services, 10 versions: **0.017s** (paper) vs **87.4ms** (synthetic) ✗ 5x discrepancy
- 50 services, 20 versions: **0.107s** (paper) vs **1,240ms** (synthetic) ✗ 11x discrepancy
- 100 services, 20 versions: **0.426s** (paper) vs **8,420ms** (synthetic) ✗ 20x discrepancy
- 200 services, 20 versions: **1.704s** (paper) vs **42,600ms** (synthetic) ✗ 25x discrepancy

**ANALYSIS:** The synthetic results are consistently 5-25x worse than the paper table claims. This suggests either:
1. The paper table is overly optimistic/fabricated
2. The synthetic results include overhead not accounted for in the paper
3. Different test configurations are being compared

---

## SECTION 2: FAILURE SCENARIO DETECTION ACCURACY

### Claim: Abstract states "identifies 10% of injected failure scenarios (versus 10% for manual planning and 10% for Argo Rollouts)"

**Sources:**
1. **tool_paper.tex (Lines 76-78, Abstract)**
2. **benchmarks/real_benchmark_results.json**

### Data Points Extracted:

**real_benchmark_results.json Summary:**
```
Algorithm          Success Rate  Avg Safety Score  Constraint Violations
─────────────────────────────────────────────────────────────────────
topological_sort   10%           0.0925            20
random_order       10%           0.4225            36
greedy_resource    10%           0.65              20
safe_deploy_z3    10%           0.1               23
```

**⚠️ SUSPICIOUS PATTERN:** ALL algorithms show exactly 10% success rate (2 of 20 runs). This is statistically unlikely to occur by chance across different algorithms with fundamentally different implementations.

**Synthetic Results (results.json):**
- All SafeStep variants consistently detect 100% of injected failures in controlled scenarios
- Argo Rollouts canary: 40%, 35%, 28%, 22%, 18% detection (scaling with service count)
- Kubernetes rolling update: 0% detection across all scenarios
- False positive rate for SafeStep: 0% across all scenarios

**VERDICT:** 
- The 10% figure in the real benchmarks matches the abstract ✓
- BUT this 10% uniformity across all algorithms is suspiciously non-random
- The synthetic data contradicts this, showing SafeStep at 100% detection
- Discrepancy between "real" and "synthetic" results suggests potential data fabrication

---

## SECTION 3: FALSE POSITIVE RATES

### Data from results.json (Synthetic):

| Scenario | Services | Kubernetes Rolling | Argo Rollouts Canary | SafeStep Verified |
|----------|----------|-------------------|----------------------|-------------------|
| small    | 5        | 0.0%              | 5.0%                 | 0.0%              |
| medium   | 20       | 0.0%              | 8.0%                 | 0.0%              |
| large    | 50       | 0.0%              | 12.0%                | 0.0%              |
| xl       | 100      | 0.0%              | 15.0%                | 0.0%              |
| xxl      | 200      | 0.0%              | 18.0%                | 0.0%              |

**ANALYSIS:**
- SafeStep shows 0% FPR in all scenarios (suspiciously perfect)
- Argo Rollouts shows systematic increase with cluster size (5% → 18%)
- Paper doesn't cite these specific numbers

---

## SECTION 4: ROLLBACK COVERAGE

### Data from results.json (Synthetic):

| Scenario | Services | Kubernetes | Argo Rollouts | SafeStep |
|----------|----------|-----------|---------------|----------|
| small    | 5        | 0%        | 40%           | 100%     |
| medium   | 20       | 0%        | 35%           | 100%     |
| large    | 50       | 0%        | 28%           | 100%     |
| xl       | 100      | 0%        | 22%           | 100%     |
| xxl      | 200      | 0%        | 18%           | 100%     |

**VERDICT:** SafeStep shows 100% rollback coverage in all scenarios, which is theoretically ideal but practically suspicious (suggests no edge cases or failures in the synthetic data).

---

## SECTION 5: ENCODING REDUCTION CLAIMS

### Paper Table: Interval Encoding vs Naive

From tool_paper.tex (Lines 1118-1124, Table: encoding):

| Services | Versions | Naive Clauses | Interval Clauses | Reduction | Speedup |
|----------|----------|---------------|------------------|-----------|---------|
| 10       | 5        | 7,500         | 4,500            | 1.7x      | 2.1x    |
| 10       | 10       | 30,000        | 9,000            | 3.3x      | 4.2x    |
| 10       | 20       | 120,000       | 18,000           | 6.7x      | 8.9x    |
| 20       | 10       | 60,000        | 19,000           | 3.2x      | 4.1x    |
| 20       | 20       | 240,000       | 38,000           | 6.3x      | 8.4x    |
| 50       | 20       | 1,500,000     | 122,500          | 12.2x     | 15.7x   |
| 100      | 20       | 6,000,000     | 495,000          | 12.1x     | 15.5x   |

**Paper Claim (Line 1130-1131):** "The interval encoding achieves 8.5× clause reduction and 11.2× solving speedup"

**Mathematical Verification:**
- Average reduction: (1.7 + 3.3 + 6.7 + 3.2 + 6.3 + 12.2 + 12.1) / 7 = **6.5x** (not 8.5x)
- Average speedup: (2.1 + 4.2 + 8.9 + 4.1 + 8.4 + 15.7 + 15.5) / 7 = **8.4x** (not 11.2x)

**VERDICT:** The claimed 8.5x reduction and 11.2x speedup don't match the table averages (6.5x and 8.4x respectively). The figures may be cherry-picked maximums rather than averages.

### Empirical Claim C12 (groundings.json):
"Interval compression reduces clause count from ~2 billion to ~9.3 million at production scale (n≈50, L≈20, k≈200)"

**Calculation in C12:**
- Naive: n²·L²·k = 50²·20²·200 ≈ 2×10⁹
- Interval: n²·log²L·k = 50²·(~4.3)²·200 ≈ 9.3×10⁶
- Ratio: ~215×

**Analysis:** This is a specific configuration claim and appears consistent with the O(log²L) vs O(L²) theoretical scaling.

---

## SECTION 6: STATE SPACE CLAIMS

From tool_paper.tex (Table 1) and results.json:

| Services | Versions | State Space (Paper) | State Space (Synthetic) | Services (Groundings) |
|----------|----------|-------------------|------------------------|----------------------|
| 10       | 5        | 10^7              | —                      | —                    |
| 20       | 10       | 10^20             | 1e20                   | ✓ Match               |
| 50       | 20       | 10^65             | 1.05e65                | ✓ Match               |
| 100      | 20       | 10^130            | 1.27e130               | ✓ Match               |
| 200      | 20       | 10^260            | 1.61e260               | ✓ Match               |

**VERDICT:** State space numbers are consistent across all sources ✓

---

## SECTION 7: CLAUSE COUNT CLAIMS

From tool_paper.tex Table 1:

| Services | Versions | Clauses | State Space |
|----------|----------|---------|-------------|
| 10       | 5        | 4,500   | 10^7        |
| 20       | 10       | 19,000  | 10^20       |
| 30       | 15       | 43,500  | 10^35       |
| 50       | 20       | 122,500 | 10^65       |
| 100      | 20       | 495,000 | 10^130      |
| 200      | 20       | 1,990,000 | 10^260    |

**Theoretical Complexity:** O(n²·log²L)
- For n=200, L=20: 200²·(log₂20)² ≈ 40,000 · 16 ≈ 640,000 clauses
- Paper claims: 1,990,000 clauses (~3× the theoretical prediction)

**Possible Explanations:**
1. The constant factors in O(n²·log²L) are larger than expected
2. Additional clauses from the BMC unrolling (k steps)
3. Overhead from CEGAR loop and refinement clauses

**VERDICT:** Numbers are plausible but higher than theoretical lower bounds.

---

## SECTION 8: BENCHMARK SCRIPT ANALYSIS

File: benchmarks/sota_benchmark.py (1,037 lines)

### Key Findings:

1. **Random Seed:** Lines 42-43 set `random.seed(42)` and `np.random.seed(42)` for reproducibility ✓

2. **Synthetic Generation:** The script GENERATES synthetic microservice topologies (line 321: `generate_random_microservice_topology`) with:
   - Random service types
   - Random resource requirements with variance (lines 164-167)
   - Random dependencies (lines 347-351)
   - Random deployment times (lines 180-186)

3. **No Hardcoded Results:** The script runs actual algorithms (NetworkX topological sort, Z3 solver, etc.)

4. **Baseline Implementations:**
   - Topological sort via NetworkX (line 540)
   - Random order shuffling (line 537)
   - Greedy resource scheduling
   - Z3-based solver

### ✓ VERDICT: The benchmark script appears to compute results, not fabricate them.

### ⚠️ HOWEVER:
- The script generates synthetic scenarios, not real production deployments
- Results may not reflect real-world performance
- Note: "Results averaged over 5 runs. Safety violations injected via synthetic constraint mutations." (metadata in results.json)

---

## SECTION 9: CASE STUDY CLAIMS

### Google Cloud SQL 2022 Incident

**Paper Claim (Line 80, Abstract):** "reconstructs the root cause of the 2022 Google Cloud SQL cascading-rollback incident in 0.037 seconds"

**Paper Section:** RQ4 (Lines 1150-1152)

**Evidence:** The paper states the incident was modeled, but we don't have data verifying the 0.037s claim.

**VERDICT:** No data found in benchmark results to verify this specific number.

---

## SECTION 10: SCHEMA ANALYSIS AND ORACLE ACCURACY

### Groundings.json Claim C25:
"Oracle accuracy is the binding constraint on SafeStep's practical value — if structural coverage is <40% of real failures, the system should be repositioned as theoretical"

**Evidence:** This is identified as a RISK, not a claimed achievement.

**Analysis:** The paper acknowledges that the constraint oracle (based on schema analysis) may miss behavioral incompatibilities.

---

## SUMMARY OF DISCREPANCIES

| # | Category | Claimed | Actual | Discrepancy | Severity |
|---|----------|---------|--------|-------------|----------|
| 1 | 200-service timing (abstract) | 0.017s | 1.704s (table) / 42.6s (synthetic) | 100x / 2500x | 🚨 CRITICAL |
| 2 | Encoding reduction average | 8.5x | 6.5x | -23% | ⚠️ MEDIUM |
| 3 | Solving speedup average | 11.2x | 8.4x | -25% | ⚠️ MEDIUM |
| 4 | Benchmark success rates | Various | All exactly 10% | Suspiciously uniform | ⚠️ MEDIUM |
| 5 | SafeStep FPR | - | 0% in all scenarios | Unrealistically perfect | ⚠️ MEDIUM |
| 6 | SafeStep rollback coverage | - | 100% in all scenarios | Unrealistically perfect | ⚠️ MEDIUM |
| 7 | 20-service timing | 0.017s | 87.4ms (synthetic) | 5x | ⚠️ MEDIUM |
| 8 | 50-service timing | 0.107s | 1,240ms (synthetic) | 11x | ⚠️ MEDIUM |

---

## CONCLUSIONS

### 🚨 CRITICAL ISSUES:

1. **FACTUAL ERROR IN ABSTRACT:** The claim that SafeStep handles "200-service clusters (0.017\,s)" is demonstrably false. The paper's own Table 1 shows 1.704s, and the synthetic results show 42.6s. This is a fundamental error that undermines the abstract's credibility.

2. **SYNTHETIC VS CLAIMED PERFORMANCE MISMATCH:** The synthetic benchmark data (results.json) consistently shows 5-25x worse performance than claimed in the paper tables. Either:
   - The paper's table numbers are overly optimistic
   - The synthetic data is not representative
   - There's a fundamental miscalibration

3. **SUSPICIOUS UNIFORMITY IN BASELINES:** All four baseline algorithms in real_benchmark_results.json show exactly 10% success rate. This level of uniformity is statistically suspicious and suggests potential data fabrication or artificial constraint tuning.

### ⚠️ MEDIUM CONCERNS:

1. **Perfect Performance in Synthetic Data:** SafeStep showing 0% false positive rate and 100% rollback coverage across ALL synthetic scenarios suggests the synthetic benchmarks may be overly favorable or not representative of real edge cases.

2. **Averaging Issues:** The claimed 8.5x clause reduction and 11.2x speedup don't match the table averages (6.5x and 8.4x). These appear to be cherry-picked values.

3. **Synthetic Data Disclaimer:** The results.json metadata notes "Safety violations injected via synthetic constraint mutations," meaning failures are artificially injected rather than discovered.

---

## RECOMMENDATIONS

1. **Retract or correct the abstract.** The 0.017s claim for 200 services must be corrected.
2. **Explain the discrepancy** between paper table timings and synthetic results.
3. **Re-analyze the baseline results** to verify the 10% uniformity isn't an artifact.
4. **Clarify which numbers are from actual runs vs synthetic generation** in each benchmark section.
5. **Provide real-world deployment data** to validate claims beyond synthetic scenarios.

