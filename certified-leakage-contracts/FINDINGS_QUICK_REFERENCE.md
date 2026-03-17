# LeakCert Paper Audit — Quick Reference

## Core Findings

**Paper Scope:** 12 evaluation tables, 2 figures, 11 experiments  
**Data Integrity Status:** ❌ CRITICAL ISSUES

### By The Numbers

| Metric | Count |
|--------|-------|
| Total evaluation tables | 12 |
| Completely unsupported tables | 4 (Tables 4, 8, 9, 10) |
| Partially supported tables | 5 (Tables 2, 3, 5, 6, 11) |
| Well-supported tables | 3 (Tables 1, 13 [modeled], "concurrent") |
| Empty placeholder figures | 2 (Figures 1, 2) |
| Experiments described but unimplemented | 1 (Table 7) |
| **Claimed benchmarks** | ~60+ (CVE, head-to-head, BearSSL, Signal, etc.) |
| **Actually committed benchmarks** | 9 in results.json |
| **Fabrication rate** | ~85% of claimed benchmarks missing |

### Critical Missing Data

```
❌ Table 4: CVE Regression Detection
   - Claims 4 CVEs with specific leakage bounds
   - Reality: Zero supporting data, CVE binaries not in repo

❌ Table 8: Head-to-Head Detection Accuracy  
   - Claims perfect 1.00 recall/precision on "50 benchmarks"
   - Reality: Benchmark suite doesn't exist

❌ Table 9: BearSSL/Signal Extended Benchmarks
   - Claims 12 specific benchmarks with bounds and timings
   - Reality: Not a single BearSSL/Signal entry in results.json

❌ Table 10: False Positive Rate Comparison
   - Claims FPR across 6 tools on "22 verified constant-time benchmarks"
   - Reality: Benchmark suite completely missing

⚠️ Table 13: Concurrent Analysis
   - Claims "23.8x leakage amplification" from timing channels
   - Reality: Data is hardcoded published bounds, not tool discovery

🔴 Figures 1 & 2: Scalability and Precision Graphs
   - Completely empty with single (0,0) placeholder point
```

### Weak Language (Masking Unvalidated Claims)

- **"expected to"** appears **30+ times** (e.g., lines 130, 742, 744, 745)
- **"designed to"** appears **15+ times** (intent vs. measurement)
- **"pending"/"future work"** appears **10+ times** (admission of missing work)

**Example:** Abstract (line 129): "bounds are **expected to be** within a small constant factor of exhaustive enumeration" — only validated for 1/10 primitives.

---

## Which Tables ARE Properly Supported

### ✅ Table 1: Self-Benchmarking (25.5ms AES, 12.7ms regression)
**Data Source:** `benchmarks/honest_benchmark_results.json`  
**Status:** ✓ Committed and reproducible  
**Caveat:** Uses debug (unoptimized) builds; not representative of production

### ✓ Table 2: Precision Evaluation
**Data Source:** `benchmarks/results.json` (9 benchmarks)  
**Status:** ⚠️ Partially supported  
**Issue:** 9 of 10 rows lack ground truth — using LeakCert's own bounds as validation
- Only AES-128-ECB has external reference (CacheAudit: 4.52 bits)
- Others claim "exhaustive enumeration pending" (line 736)

### ✓ Table 3: Scalability
**Data Source:** `benchmarks/results.json` → BoringSSL entry (2583.4s total)  
**Status:** ⚠️ Aggregate time present, per-function distribution missing  
**Issue:** Percentile values (t50=31.7ms, t90=187.3ms) NOT in JSON — appears to be author-estimated

### ✓ Table 6: Composition Overhead
**Data Source:** `results.json` → `composition_overhead_pct` field  
**Status:** ⚠️ Only BoringSSL aggregate (11.4%) present  
**Issue:** Specific program overhead percentages (8.2%, 7.1%, 9.7%, 14.7%) NOT verifiable

---

## Benchmark Suite Status

| Suite | Claimed | Committed | Type |
|-------|---------|-----------|------|
| BoringSSL | "~600 functions" | 1 entry (247 funcs?) | ✓ Partial |
| OpenSSL | "~2000 functions" | 0 entries | ✗ Missing |
| libsodium | "~200 functions" | 0 entries | ✗ Missing |
| BearSSL | 6 benchmarks | 0 entries | ✗ Missing |
| Signal Protocol | 6 benchmarks | 0 entries | ✗ Missing |
| CVE binaries | 4 vulnerabilities | 0 entries | ✗ Missing |
| Head-to-head suite | 50 programs | 0 entries | ✗ Missing |
| Constant-time suite | 22 benchmarks | 0 entries | ✗ Missing |

---

## Specific Claims Requiring Scrutiny

### Claim 1: "Abstract bounds within small constant factor of exhaustive enumeration"
**Source:** Abstract (lines 129-136)  
**Evidence:** Table 2 shows ratios ranging 1.0–1.43  
**Caveat:** Only 1/10 primitives have true ground truth (AES vs CacheAudit)  
**Assessment:** ⚠️ Overstated — most ratios compare LeakCert to itself

### Claim 2: "Perfect detection accuracy matching ct-verif"
**Source:** Table 8 (lines 932–952)  
**Claimed:** TP=30, FN=0, FP=0, TN=20 (1.00 recall, 1.00 precision)  
**Evidence:** ✗✗✗ None — benchmark suite nowhere to be found  
**Assessment:** ❌ Completely unsubstantiated

### Claim 3: "Spectre gadget detection on Spectector benchmarks"
**Source:** Table 7 / Experiment 6 (lines 897–904)  
**Claimed:** Spectector PoC suite evaluation  
**Evidence:** ✗ No Spectector benchmark suite in repo  
**Assessment:** ❌ Entirely unimplemented

### Claim 4: "CVE regression detection on 4 real vulnerabilities"
**Source:** Table 4 (lines 790–818)  
**Claimed:** CVE-2016-0702, CVE-2018-0495, CVE-2019-1543, CVE-2020-0543 detected  
**Evidence:** ✗ No CVE binaries; footnote admits "patched binaries not yet available"  
**Assessment:** ❌ Completely fabricated

### Claim 5: "Concurrent timing analysis discovers 23.8x additional leakage"
**Source:** Table 13 / Section 5.1 (lines 1461–1527)  
**Claimed:** Timing models discover additional leakage paths  
**Reality:** All "timing bits" are hardcoded published bounds capped at 200-bit value from Yarom & Falkner 2014  
**Assessment:** ⚠️ Misleading — presents hardcoded constant as tool discovery

---

## Recommended Sections to Remove or Rewrite

### REMOVE Entirely:
1. **Table 4** (CVE evaluation) — zero supporting data
2. **Table 7** (Spectre detection) — unimplemented
3. **Table 8** (50-benchmark accuracy) — suite missing
4. **Table 9** (BearSSL/Signal) — all data missing
5. **Table 10** (FPR comparison) — suite missing
6. **Figures 1–2** (empty placeholders)

### REWRITE with Caveats:
1. **Table 2:** Add column "Ground Truth Source" or remove 9/10 rows
2. **Table 3:** Either commit per-function distribution or remove percentiles
3. **Table 5:** Add data source or remove
4. **Table 6:** Only report BoringSSL aggregate, not individual programs
5. **Table 11:** Remove verification timing claims (only generation time committed)
6. **Table 13:** Explicitly label as "theoretical projection using published bounds"

### SOFTEN:
1. **Table 1:** Add caveat about debug-build limitations
2. **Abstract:** Replace "expected to be" with "preliminary evaluation shows" (only BoringSSL)

---

## Benchmark Scripts Status

| Script | Status | Output | Issues |
|--------|--------|--------|--------|
| `honest_benchmark.py` | ✓ Works | `honest_benchmark_results.json` (60 lines) | Debug builds only |
| `sota_benchmark.py` | ❌ Broken | (crashes) | Line 1576: AccessType serialization error |
| `validated_concurrent_benchmark.py` | ✓ Works | `validated_concurrent_results.json` (456 lines) | Uses hardcoded published bounds, not empirical |
| `run_benchmarks.sh` | ✓ Present | (orchestrator) | Works but limited by missing benchmark suites |

### Fix Required:
```python
# sota_benchmark.py, line 34-36
class AccessType(Enum):
    READ = "read"
    WRITE = "write"

# Add JSON encoder:
def serialize_access_type(obj):
    if isinstance(obj, AccessType):
        return obj.value
    raise TypeError(f"Type {type(obj)} not serializable")

# Use in json.dump:
json.dump(data, f, default=serialize_access_type)
```

---

## Data Integrity Score

```
Supported by committed artifacts:  30% (self-bench, partial BoringSSL)
Partially supported (caveats):     35% (Tables 2, 3, 6, 13)
Completely unsupported:           35% (Tables 4, 7, 8, 9, 10, 11, Figs 1-2)

Overall credibility for publication: ⚠️ FAIR (needs remediation)
```

---

## What To Do Next

### Option A: Honest Minimum (3–5 hours)
1. Remove Tables 4, 7, 8, 9, 10, 11
2. Remove Figures 1–2
3. Rewrite Table 2 footnote: "Ground truth available only for AES-128 (CacheAudit baseline)"
4. Rewrite Section 5.1: "These are theoretical projections using published bounds"
5. Update abstract to claim "preliminary BoringSSL case study" not "comprehensive empirical"

### Option B: Proper Evaluation (40+ hours)
1. Commit CVE binaries and re-run Table 4 evaluation
2. Build head-to-head benchmark suite (50 programs) for Table 8
3. Evaluate BearSSL and Signal Protocol for Table 9
4. Create verified constant-time benchmark suite for Table 10
5. Implement contract verification benchmarks for Table 11
6. Fix and run sota_benchmark.py
7. Populate Figures 1–2 with real data
8. Re-run all experiments and commit complete results.json

### Option C: Hybrid (12–20 hours)
1. Fix sota_benchmark.py serialization bug
2. Commit what evaluation data exists
3. Clearly label missing evaluations as "Placeholder: Pending X"
4. Rewrite weak language to be honest about what's missing
5. Move incomplete experiments to "Future Work"

---

## Reference Files

**Audit Reports:**
- `AUDIT_REPORT.md` — Detailed 40KB audit with line-by-line analysis
- `AUDIT_SUMMARY.txt` — This document (quick reference)

**Benchmark Data:**
- `benchmarks/results.json` — Main benchmark results (9 entries, 484 lines)
- `benchmarks/honest_benchmark_results.json` — Debug build timings (60 lines)
- `benchmarks/validated_concurrent_results.json` — Concurrent leakage model (456 lines)

**Scripts:**
- `benchmarks/honest_benchmark.py` — Generates Table 1
- `benchmarks/sota_benchmark.py` — Broken; would generate Table 8
- `benchmarks/validated_concurrent_benchmark.py` — Generates Table 13
- `benchmarks/run_benchmarks.sh` — Orchestrator

---

**Document Generated:** 2025-01-XX  
**Audit Scope:** tool_paper.tex (2043 lines)  
**Benchmarks Analyzed:** 12 tables + 2 figures + 11 experiments  
**Finding:** Significant gap between claimed and supported evaluation

