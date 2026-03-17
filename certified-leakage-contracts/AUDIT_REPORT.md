# LeakCert Tool Paper (tool_paper.tex) — Audit Report
## Unsupported Benchmark Tables and Hardcoded/Simulated Content

**Audit Date:** 2025-01-XX  
**Scope:** Analysis of evaluation claims in tool_paper.tex against committed analysis artifacts  
**Key Finding:** Multiple benchmark tables rely on hardcoded/modeled data with weak language ("expected to," "designed to") rather than empirical results.

---

## Executive Summary

**Critical Issues:**
- **11 of 12 evaluation tables** contain unsupported claims or modeling artifacts
- **Perfect accuracy claims** (Table 8, 10) have NO supporting evidence in committed files
- **CVE detection claims** (Table 4) completely unsupported — CVE binaries missing
- **Concurrent leakage analysis** (Table 13) uses parametric modeling, not real measurements
- **Two figures** (scalability, precision) are **completely empty placeholders**
- **Weak language** pervasive: "expected to," "designed to," "pending validation," "left to future work"

---

## Table-by-Table Analysis

### ✅ Table 1: Self-Benchmarking (Lines 657–681)

**Status:** PARTIALLY SUPPORTED  
**Claimed Data:** 25.5ms (AES), 12.7ms (regression), 1μs (synthetic)

**Backed by Committed Files:**
- ✓ `benchmarks/honest_benchmark.py` — Script to generate timings
- ✓ `benchmarks/honest_benchmark_results.json` — Actual timings (but at 0.02551s median, NOT 25.5ms)
- ✗ Numbers in table **DO NOT MATCH** honest_benchmark_results.json
  - Table claims: 25.5 ms for AES T-table
  - JSON shows: 0.02551 s (25.51 ms) — **coincidentally close but labeled differently**
  - These are "debug-build shipped examples," NOT production analysis

**Issues:**
- Uses **unoptimized debug builds** (acknowledged in honest_benchmark.py line 101-104)
- No external tool comparison ("left to future work" line 693)
- Micro-benchmarks in release build use hardcoded cache geometry assumptions

**Recommendation:** Soften all claims in Table 1. Add caveat: "These are micro-benchmarks on synthetic examples, not representative of production crypto analysis. See Table 3 for real-world BoringSSL/OpenSSL benchmarks."

---

### ⚠️ Table 2: Precision Evaluation (Lines 709–737)

**Status:** PARTIALLY SUPPORTED (with major caveats)  
**Claimed Data:** Tightness ratios ranging 1.0–1.43 across 10 primitives

**Backed by Committed Files:**
- ✓ 9 benchmarks in `benchmarks/results.json` (AES-128-ECB, AES-256-GCM, ChaCha20, X25519, RSA-2048, ECDSA-P256, SHA-256, HKDF, BoringSSL-full)
- ✓ Leakage bounds present for all 9

**Critical Issues:**

1. **Ground truth missing for 9/10 rows:**
   - Footnote 734 states: "All numerical entries from benchmarks/results.json. $C_f$ for AES is the CacheAudit baseline bound; other $C_f$ values are **LeakCert bounds (exhaustive enumeration pending)**."
   - Only AES-128-ECB has external baseline (CacheAudit: 4.52 bits)
   - Other rows (AES-256-GCM, ChaCha20, X25519, RSA-2048, etc.) use **LeakCert's own bounds as "ground truth"**
   - This is circular: no true precision measurement for 9/10 benchmarks

2. **Confusing claims:**
   - Lines 741-746: "framework is designed to produce sound over-approximations... tightness ratios expected to remain within a small constant factor"
   - This is hedging language. "Expected" ≠ measured.

3. **Hard to interpret results:**
   - Row 3 (AES-256-GCM): 0.00 bits both directions → $r_f = 1.00$ (exact)
   - Row 8 (RSA-2048): 1.42 bits both directions → $r_f = 1.00$ (exact)
   - But no explanation why RSA matches exactly when others are 1.43x over

**Recommendation:** 
- **REMOVE Table 2 OR replace with honest table:**
  - Column: "Source of ground truth"
  - Only include AES-128-ECB (vs CacheAudit)
  - For others, either:
    1. Add exhaustive enumeration (feasible for small keys)
    2. Remove tightness ratio claim; report only "Bound (bits)" with footnote "No ground truth available"

---

### ⚠️ Table 3: Scalability (Lines 756–774)

**Status:** SUPPORTED (mostly)  
**Claimed Data:** Per-library analysis times (BoringSSL: 2583.4s total, OpenSSL: 892.1s, libsodium: 421.3s)

**Backed by Committed Files:**
- ✓ `benchmarks/results.json` has `boringssl-full` entry with `analysis_time_sec: 2583.4`
- ✓ BoringSSL entry has `instruction_count: 412,847` (consistent with "~600 functions" claim at line 752)
- ✗ OpenSSL and libsodium entries NOT in results.json
- ✗ Percentile times (t50=31.7, t90=187.3) NOT in results.json — these appear to be **per-function statistics**, not in the JSON

**Issues:**
- Per-function distribution (t50, t90, t99, t_max) claimed but data structure doesn't support it
- Only aggregate total time (2583.4s) is present for BoringSSL
- OpenSSL and libsodium timings completely absent from committed files

**Recommendation:** 
- For BoringSSL: Replace t50/t90/t_max with honest data or remove percentile claims
- For OpenSSL/libsodium: Mark as "pending validation" or remove entirely
- **CRITICAL:** Commit per-function timing distribution to results.json if claiming t50/t90/t99

---

### ❌ Table 4: CVE Regression Detection (Lines 790–818)

**Status:** COMPLETELY UNSUPPORTED  
**Claimed Data:** 4 CVEs (CVE-2016-0702, CVE-2018-0495, CVE-2019-1543, CVE-2020-0543) with leakage bounds (1.08–3.52 bits) and detection times (8.9–41.2s)

**Backed by Committed Files:**
- ✗ NO CVE entries in `benchmarks/results.json`
- ✗ CVE binaries NOT in repository
- ✗ No CVE benchmark script

**Issues:**
1. Footnote 807 admits: "Patch-vs-vulnerable delta analysis requires patched binaries **not yet available**."
2. This is a major claims table for **regression detection**, but entirely missing supporting data
3. Line 818: "left to future work" — contradicts the claim that CVEs are evaluated

**Recommendation:** 
- **REMOVE Table 4 entirely** OR replace with "PLACEHOLDER: Pending CVE binary availability"
- Move CVE regression to "Limitations" section (line 1532) with clear statement: "CVE evaluation is not yet complete."

---

### ⚠️ Table 5: Reduction Operator Effectiveness (Lines 829–849)

**Status:** PARTIALLY SUPPORTED  
**Claimed Data:** Direct vs. reduced product bounds showing ρ-ratio effectiveness (0.00–1.00)

**Backed by Committed Files:**
- ✗ Data NOT in results.json
- ✗ No source file referenced in footnote

**Issues:**
- Compares "direct product" vs "reduced product" but no benchmark script generates this comparison
- No committed data to verify claims

**Recommendation:** 
- Add footnote with data source, or
- Include full results in `benchmarks/results.json` as `"rho_effectiveness"` field

---

### ⚠️ Table 6: Composition Overhead (Lines 865–885)

**Status:** PARTIALLY SUPPORTED  
**Claimed Data:** Compositional vs. monolithic overhead ratios (κ=1.08–1.15)

**Backed by Committed Files:**
- ✓ `results.json` includes `"composition_overhead_pct"` field for BoringSSL entry
- ✗ No individual program entries (AES-128, ChaCha20, X25519, RSA-2048) in results.json
- ✗ Footnote 884 references "composition_overhead_pct field" but only BoringSSL entry has it

**Issues:**
- Claims 4 programs but only 1 BoringSSL aggregate data present
- Overhead percentages look correct but not verifiable for individual programs

**Recommendation:** 
- Add detailed per-function composition overhead breakdown to results.json, or
- Remove specific overhead claims (8.2%, 7.1%, etc.) and only report aggregate

---

### ⚠️ Table 7: Speculative Leakage Discovery (Lines 897–904)

**Status:** MENTIONED BUT EMPTY  
**Claimed Data:** Speculative uplift σ_f on Spectector benchmarks and Kocher's Spectre PoCs

**Backed by Committed Files:**
- ✗ No Spectector benchmark suite in repository
- ✗ No Kocher PoCs
- ✗ No "speculative uplift" field in results.json

**Issues:**
- Experiment completely unimplemented
- Only claims what "is expected to" happen; no actual results

**Recommendation:** 
- **Remove Experiment 7** or move to "Future Work"

---

### ❌ Table 8: Head-to-Head Detection Accuracy (Lines 932–952)

**Status:** COMPLETELY UNSUPPORTED  
**Claimed Data:** Perfect accuracy on 50 benchmarks (TP=30, FN=0, FP=0, TN=20) matching ct-verif

**Backed by Committed Files:**
- ✗ No "50 benchmarks" suite in repository
- ✗ No comparison data for ct-verif, Binsec/Rel, Spectector
- ✗ Footnote 951 vaguely references "shared benchmark suite" with no committed data
- ✗ **This table has ZERO supporting evidence**

**Issues:**
1. Identical perfect accuracy (1.00 recall, 1.00 precision) as ct-verif is suspicious
2. Claims Binsec/Rel has 0.79 recall — how is this measured against 50 unknown benchmarks?
3. Spectector column missing entirely from table but mentioned in text (line 958)

**Critical:** This is one of the main claims for LeakCert's superiority. **Complete removal required unless benchmark suite is committed.**

**Recommendation:** 
- **REMOVE Table 8** entirely
- If head-to-head comparison is important, commit:
  - 50 test programs with sources
  - Ground truth labels (safe/unsafe)
  - Tool configuration details
  - Exact invocation commands

---

### ❌ Table 9: Extended Benchmark Suite (BearSSL/Signal) (Lines 973–1002)

**Status:** COMPLETELY UNSUPPORTED  
**Claimed Data:** 12 benchmarks (6 BearSSL, 6 Signal Protocol) with leakage bounds

**Backed by Committed Files:**
- ✗ NO BearSSL entries in results.json
- ✗ NO Signal Protocol entries in results.json
- ✗ Footnote 1001 states: "All entries ." — incomplete sentence, no source

**Issues:**
- Completely fabricated or data not committed
- Claims like "BearSSL AES-CT (bitsliced): 0.00 bits bound / 34 bits baseline" with no justification
- Time data (t_50: 1.05, 1.07, etc.) impossibly precise

**Recommendation:** 
- **REMOVE Table 9** until data is committed, or
- Mark as "PLACEHOLDER: BearSSL/Signal analysis pending"

---

### ❌ Table 10: False Positive Rate Comparison (Lines 1024–1047)

**Status:** COMPLETELY UNSUPPORTED  
**Claimed Data:** FPR across 6 tools on "22 verified constant-time benchmarks"

**Backed by Committed Files:**
- ✗ The "22 verified constant-time benchmarks" suite NOT in repository
- ✗ No FPR measurements for any baseline tool (CacheAudit, Spectector, Binsec/Rel, etc.)
- ✗ Footnote 1046: "All false positive rate values ." — incomplete, no source

**Critical Issues:**
1. Footnote 1036 claims CacheAudit has 1 FP (2.9% on 35 benchmarks) — but we don't have this suite
2. Pitchfork claimed to have 2 FP (4.2%) — again, no data
3. LeakCert claims 0 FP on all 50 benchmarks — but Table 8 claimed 50 different benchmarks with 20 TN
   - **Inconsistent:** Are there 22 or 50 benchmarks?

**Recommendation:** 
- **REMOVE Table 10** — completely unsupported and internally inconsistent

---

### ⚠️ Table 11: Certificate Verification Time (Lines 1064–1086)

**Status:** UNSUPPORTED (data partially matches but claims unverified)  
**Claimed Data:** Generation vs. verification times with 47–138x speedup

**Backed by Committed Files:**
- ✓ BoringSSL timing (2583.4s generation) in results.json
- ✗ Verification time (18.7s) NOT in results.json
- ✗ No speedup calculations present

**Issues:**
- Footnote 1085 says "All verification times ." — incomplete, no source
- No committed verification benchmark script
- Claims about verification are theoretical, not measured

**Recommendation:** 
- Either implement and commit contract verification benchmarks, or
- Mark Table 11 as "Projected (not yet measured)"

---

### ❌ Figure 1: Scalability Graph (Lines 1097–1122)

**Status:** COMPLETELY EMPTY PLACEHOLDER  
**Claimed Data:** Wall-clock time vs. number of functions analyzed

**Current Content:**
```
\addplot[color=blue, mark=*, thick] coordinates { (0,0) };
\addlegendentry{\LeakCert{} (pending)}
```

**Issues:**
- Single point at (0,0) with "(pending)" label
- This is a broken placeholder, not a real figure
- Text claims "scalability" but provides no data

**Recommendation:** 
- **REMOVE Figure 1** or populate with real data from Table 3

---

### ❌ Figure 2: Precision Graph (Lines 1124–1156)

**Status:** COMPLETELY EMPTY PLACEHOLDER  
**Claimed Data:** Computed bound vs. true leakage with trend lines

**Current Content:**
```
\addplot[only marks, color=blue, mark=*, mark size=3pt] coordinates { (0,0) };
\addlegendentry{\LeakCert{} (pending)}
```

**Issues:**
- Single point at (0,0) with "(pending)" label
- This is a broken placeholder
- Trend lines for "Exact" and "2x over-approx" are present but no actual data

**Recommendation:** 
- **REMOVE Figure 2** or populate with real data from Table 2

---

### ⚠️ Table 13: Concurrent and Timing Channel Analysis (Lines 1465–1527)

**Status:** PARTIALLY SUPPORTED (but heavily modeled/simulated)  
**Claimed Data:** 10 concurrent scenarios with timing/concurrency leakage ratios

**Backed by Committed Files:**
- ✓ `benchmarks/validated_concurrent_results.json` (456 lines) contains 10 scenarios
- ✓ Data includes "st_bits", "timing_bits", "conc_bits" for each scenario
- ✗ **However, data is entirely parametric/modeled, NOT measured**

**Issues:**

1. **Modeled, not measured:**
   - Lines 1450-1463 describe "parametric model calibrated against published data"
   - Lines 1459-1463 describe Monte Carlo cache simulation, NOT real measurements
   - Footnote 1495-1501: "Timing column: calibrated parametric model (cache + branch-prediction + TLB channels) **capped at published per-primitive bounds**"

2. **Data integrity problems:**
   - All timing values are **capped** at published bounds (line 1496)
   - Example (from validated_concurrent_results.json):
     ```
     "st_bits": 3.17,           // Real ST analysis
     "timing_bits": 8.0,         // Capped at published bound
     "timing_ratio": 2.5x,        // 8.0 / 3.17 = modeled ratio
     ```
   - These are NOT new leakage paths discovered by the tool; they're **hardcoded published bounds plugged into a formula**

3. **Specific problems:**
   - RSA-CRT timing_ratio claims 23.8x but this comes from capping at 200-bit published bound divided by 8.91-bit LeakCert bound
   - Concurrent leakage values ("conc_bits") are mostly 0.0–2.0, not from real experiments
   - Bootstrap confidence intervals reported (e.g., mc_ci_lo, mc_ci_hi) but confidence column shows only 0.7 (line 1499)

**Recommendation:**
- **Rewrite Section 5.1 (lines 1448-1527) or remove entirely**
- Current language at lines 1504-1527 is **highly misleading**:
  - Claims "calibrated timing model" discovers "1.6–23.8x more leakage"
  - But this isn't LeakCert discovering leakage; it's **hardcoded published bounds**
  - Line 1513: "the branch-prediction bound is capped at the 200-bit single-trace recovery demonstrated by FLUSH+RELOAD" — this is NOT LeakCert's analysis
- If concurrent analysis is important, either:
  1. Implement real timing channel analysis in LeakCert, or
  2. Move to "Future Work" and acknowledge these are "projected estimates"

---

## Weak Language Audit

The paper uses persistent weak hedging language that masks missing evaluations:

| Count | Pattern | Example Lines | Status |
|-------|---------|----------------|--------|
| 30+ | "expected to" | 130, 742, 744, 745, 778, 851, 893, 903, 910, 955, 958, 960, 1006, 1009, 1050, 1089, 1181, 1183, 1195, 1210, 1213, 1239 | Indicates unvalidated claims |
| 15+ | "designed to" | 133, 587, 598, 741, 779, 851, 854, 889, 1603 | Indicates theoretical intent, not measured |
| 10+ | "left to future work" / "pending" | 693, 736, 810, 818, 1115, 1142, 1210, 1239, 1576 | Admission of missing data |

**Problem:** The abstract (line 127-136) claims evaluation targets "precision measurement against ground truth" and "practical scalability to full libraries," but **ground truth is missing for 9/10 primitives and full library scalability is only BoringSSL.**

---

## Missing Artifacts

| Artifact | Claimed | Committed | Status |
|----------|---------|-----------|--------|
| CVE binaries (4 vulnerabilities) | Table 4 | ✗ | Complete loss |
| Head-to-head benchmark suite (50 programs) | Table 8 | ✗ | Complete loss |
| BearSSL/Signal evaluation (12 progs) | Table 9 | ✗ | Complete loss |
| Verified constant-time benchmarks (22) | Table 10 | ✗ | Complete loss |
| Contract verification benchmark | Table 11 | ✗ | Theoretical only |
| Spectector PoCs / Kocher Spectre suite | Exp 7 | ✗ | Unimplemented |
| Per-function timing distribution | Table 3 | ✗ | Only aggregate total |
| Reduction operator effectiveness data | Table 5 | ✗ | No source |

---

## Recommendations

### Immediate (Before Publication)

1. **Table 2:** Replace or add disclaimer that 9/10 rows lack ground truth
2. **Tables 4, 8, 9, 10:** REMOVE entirely or mark "PLACEHOLDER — pending evaluation"
3. **Figures 1, 2:** Remove placeholder or populate with real data
4. **Section 5.1 (concurrent):** Rewrite to clarify that leakage ratios are **modeled**, not discovered
5. **Weak language:** Replace "expected to" with either:
   - Commit to measured results, or
   - Explicitly state "theoretical projection" or "future work"

### Best Case (If Data Exists Uncommitted)

1. Commit all benchmark suites (CVE binaries, head-to-head progs, BearSSL/Signal, constant-time suite)
2. Re-generate results.json with complete evaluation data
3. Populate empty figures
4. Add per-function timing distributions to results.json
5. Rewrite concurrent section with honest labeling of what is modeled vs. measured

### Honest Minimum (If Data Doesn't Exist)

1. **Reframe abstract:** Claim "framework design" not "empirical validation"
2. **Reduce evaluation section:** Keep only Tables 1, 2 (AES only), 3, 6
3. **Move to future work:**
   - Full CVE evaluation
   - Head-to-head tool comparison
   - Extended library evaluation (BearSSL/Signal)
   - False positive rate study
4. **Rewrite limitations (Section 7):** Be honest about what's missing, not speculative

---

## Summary Table

| Component | Supported? | Issues | Action |
|-----------|-----------|--------|--------|
| **Theoretical design** | ✓ | None — design is sound | Keep |
| **Implementation** | ✓ | 26K LoC Rust is real | Keep |
| **Table 1 (Self-bench)** | ✓ | Debug build only | Soften claims |
| **Table 2 (Precision)** | ⚠️ | 9/10 lack ground truth | Remove or replace |
| **Table 3 (Scalability)** | ⚠️ | Missing per-function distribution | Add data or remove percentiles |
| **Table 4 (CVE)** | ✗ | Zero supporting data | **REMOVE** |
| **Table 5 (Rho)** | ✗ | No committed data | Add source or remove |
| **Table 6 (Composition)** | ⚠️ | Only BoringSSL aggregate | Add per-program data |
| **Table 7 (Spectre)** | ✗ | Unimplemented | **REMOVE** |
| **Table 8 (Accuracy)** | ✗ | Perfect accuracy unsubstantiated | **REMOVE** |
| **Table 9 (Extended)** | ✗ | No BearSSL/Signal data | **REMOVE** |
| **Table 10 (FPR)** | ✗ | Inconsistent, no data | **REMOVE** |
| **Table 11 (Verification)** | ✗ | Only generation time committed | Remove timing claims |
| **Figure 1** | ✗ | Empty placeholder | **REMOVE** |
| **Figure 2** | ✗ | Empty placeholder | **REMOVE** |
| **Table 13 (Concurrent)** | ⚠️ | Modeled, not measured | Relabel as theoretical |

---

## Reference to Benchmark Scripts

- **honest_benchmark.py** (lines 1-150): Generates Table 1 data; references shipped examples in `implementation/target/debug/examples/`
- **sota_benchmark.py** (lines 1-150+): Intended for head-to-head comparison but crashes on JSON serialization of `AccessType` enum (line 1576 in LaTeX)
- **validated_concurrent_benchmark.py** (lines 1-200+): Generates Table 13; uses parametric model calibrated to published data
- **run_benchmarks.sh**: Orchestrates benchmark execution (referenced in benchmarks/README.md)

All scripts are present and functional, but most generate only **partial or modeled data**, not complete empirical evaluation.

---

## Conclusion

The paper presents a solid theoretical and implementation contribution, but **overstates empirical validation** through:
1. Missing ground truth for precision claims
2. Entirely fabricated tables (8, 9, 10) with zero supporting data
3. Conflating **parametric modeling** with **empirical measurement** (Table 13)
4. Placeholder figures presented as data
5. Heavy use of weak language to mask incomplete work

**Minimum credibility threshold:** Commit all missing benchmark suites and regenerate results before publication, or reframe as "theoretical design + preliminary BoringSSL case study" rather than "comprehensive empirical validation."

