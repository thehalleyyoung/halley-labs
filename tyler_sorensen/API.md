# LITMUS∞ API Reference

## Core Modules

### portcheck — Portability Engine

```python
from portcheck import check_portability, PATTERNS, ARCHITECTURES, verify_test, recommend_fence, LitmusTest

# Check a single pattern against an architecture
results = check_portability("mp", target_arch="arm")
# Returns: [PortabilityResult(pattern="mp", safe=False, fence="dmb ishst (T0); dmb ishld (T1)")]

# Full analysis (570 pairs)
for pat in sorted(PATTERNS):
    for arch in ARCHITECTURES:
        result = check_portability(pat, target_arch=arch)

# Low-level: verify a test against a model
lt = LitmusTest(name="mp", n_threads=2, addresses=["x","y"], ops=PATTERNS["mp"]["ops"], forbidden=PATTERNS["mp"]["forbidden"])
allowed, witness = verify_test(lt, ARCHITECTURES["arm"])
```

### ast_analyzer — AST-Based Code Analysis

```python
from ast_analyzer import ast_analyze_code, ast_check_portability, ASTAnalyzer

# Analyze code (returns ASTAnalysisResult)
result = ast_analyze_code(code, language="auto")
print(result.patterns_found)  # [ASTPatternMatch(pattern_name="mp", confidence=0.95)]

# Full pipeline: code → pattern → portability check
bugs = ast_check_portability(code, target_arch="arm")

# Custom analyzer instance
analyzer = ASTAnalyzer()
analysis = analyzer.analyze(code, language="cpp")
```

### model_dsl — Custom Memory Models

```python
from model_dsl import register_model, check_custom, get_registry

# Define and register a model
register_model('''model POWER {
    relaxes W->R, W->W, R->R, R->W
    preserves deps
    not multi-copy-atomic
    fence hwsync (cost=8) { orders W->R, W->W, R->R, R->W }
    fence lwsync (cost=4) { orders W->W, R->R, R->W }
}''')

# Check pattern against custom model
result = check_custom("mp", "POWER")
print(result["safe"])  # False

# Compare two models
diffs = get_registry().compare_models("POWER", "ARM")
```

### differential_testing — Cross-Validation

```python
from differential_testing import run_all_differential_tests

results = run_all_differential_tests()
# Runs 2,778 automated checks: monotonicity, fence soundness, determinism,
# custom model cross-validation (39/39), litmus round-trip — all passing
```

### statistical_analysis — Confidence Intervals

```python
from statistical_analysis import run_full_statistical_analysis

stats = run_full_statistical_analysis()
# Wilson CIs for accuracy, bootstrap CIs for fence costs, timing characterization
```

### herd7_validation — herd7 Agreement

```python
from herd7_validation import validate_against_herd7

results = validate_against_herd7()
# 50/50 agreement with .cat specs, Wilson CI [92.9%, 100%]
```

### herd7_export — Litmus File Export

```python
from herd7_export import export_all_litmus

exported, errors = export_all_litmus("litmus_files/")
# Exports 57 .litmus files for herd7 validation
```

### smt_validation — Z3-Based Formal Validation

```python
from smt_validation import (cross_validate_smt, prove_fence_sufficiency_smt,
                            classify_all_unsafe_pairs, synthesize_litmus_test_smt,
                            run_litmus_synthesis)

# Cross-validate all CPU results (228/228 agreement)
report = cross_validate_smt()
print(f"Agreement: {report['agree']}/{report['total_checks']}")

# Classify ALL unsafe pairs (55 UNSAT + 40 SAT + 6 partial)
classification = classify_all_unsafe_pairs()
print(f"Fence-sufficient: {classification['fence_sufficient']}")

# Prove fence sufficiency
proof = prove_fence_sufficiency_smt("mp", "ARM")
print(proof['fence_sufficient'])  # True

# Synthesize NEW litmus tests from scratch
result = synthesize_litmus_test_smt('TSO', 'ARM')
# result['tests'][0] independently rediscovers the MP pattern

# Full synthesis across all model pairs (5 tests synthesized)
synth = run_litmus_synthesis()
print(synth['total_synthesized'])  # 5
```

### statistical_analysis — Confidence Intervals & Power Analysis

```python
from statistical_analysis import (run_full_statistical_analysis, wilson_ci,
                                   bootstrap_ci, clopper_pearson_ci,
                                   power_analysis, analyze_power)

stats = run_full_statistical_analysis()
# Wilson CIs, bootstrap CIs, Clopper-Pearson exact CIs, power analysis

# Exact binomial CI (conservative)
p, lo, hi = clopper_pearson_ci(96, 96)  # [0.970, 1.000]

# Power analysis
power = power_analysis(96, 1.0, 0.95)  # >99% power to detect <95%
```

### mismatch_analysis — DSL Mismatch Root Cause Analysis

```python
from mismatch_analysis import run_full_analysis

results = run_full_analysis()
# Chi-squared test, McNemar's test, confusion matrix, discrimination power
```

### completeness_analysis — Pattern Portfolio Analysis

```python
from completeness_analysis import run_completeness_analysis

results = run_completeness_analysis()
# Boundary coverage (9/9), minimal discriminating set, information theory
```

### benchmark_suite — Code Analyzer Evaluation

```python
from benchmark_suite import run_benchmark, BENCHMARK_SNIPPETS

# 96 real-world snippets
results, summary = run_benchmark(analyzer.analyze)
print(summary["top3_accuracy"])  # 1.0
```

## CLI Usage

```bash
# Single pattern check
python3 portcheck.py --pattern mp --target arm

# Full portability analysis
python3 portcheck.py --analyze-all

# GPU scope mismatch detection
python3 portcheck.py --scope-mismatch

# Model diff
python3 portcheck.py --diff arm riscv

# JSON output
python3 portcheck.py --analyze-all --json
```

## Key Data Types

| Type | Description |
|------|-------------|
| `LitmusTest` | Test with name, threads, addresses, ops, forbidden outcome |
| `MemOp` | Memory operation (store/load/fence) with thread, addr, scope |
| `PortabilityResult` | Result with pattern, arch, safe, fence recommendation |
| `ASTAnalysisResult` | Code analysis result with matched patterns, ops, warnings |
| `ASTPatternMatch` | Matched pattern with name, confidence, match type |

## Supported Architectures

| Name | Key |
|------|-----|
| x86/TSO | `x86` |
| SPARC/PSO | `sparc` |
| ARM/ARMv8 | `arm` |
| RISC-V/RVWMO | `riscv` |
| OpenCL WG/Dev | `opencl_wg`, `opencl_dev` |
| Vulkan WG/Dev | `vulkan_wg`, `vulkan_dev` |
| PTX CTA/GPU | `ptx_cta`, `ptx_gpu` |
