# LITMUS∞ — API Reference

## CLI: `litmus-check`

The primary user-facing tool. Scans C/C++/CUDA files for concurrency
patterns and checks portability.

```bash
litmus-check --target arm src/           # scan directory
litmus-check --target arm myfile.c       # single file
litmus-check --target arm --stdin        # read from stdin
litmus-check --target arm --json src/    # JSON output for CI
```

**Flags:**
| Flag | Description |
|------|-------------|
| `--target`, `-t` | Target architecture (required): x86, sparc, arm, riscv, opencl_wg, etc. |
| `--source`, `-s` | Source architecture (default: x86) |
| `--stdin` | Read code from stdin |
| `--json` | Output JSON |
| `--verbose`, `-v` | Show safe patterns too |
| `--no-color` | Disable ANSI colors |

---

## Core: Pattern-Based Checking

### `check_portability(pattern_name, source_arch='x86', target_arch=None) → List[PortabilityResult]`

Check whether a litmus test pattern is safe to port between architectures.

```python
from portcheck import check_portability

result = check_portability("mp", target_arch="arm")
print(result[0].safe)                  # False
print(result[0].fence_recommendation)  # "dmb ishst (T0); dmb ishld (T1)"
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `pattern_name` | `str` | Litmus test name (75 built-in) |
| `source_arch` | `str` | Source architecture (default `"x86"`) |
| `target_arch` | `str` or `None` | Target architecture. `None` = all 10. |

**Returns:** `List[PortabilityResult]` with fields: `safe`, `fence_recommendation`, `compression_ratio`.

---

## AST-Based Code Analysis

### `ast_analyze_code(code, language='auto') → ASTAnalysisResult`

Analyze concurrent code using tree-sitter AST parsing (C/C++/CUDA) with regex fallback.

```python
from ast_analyzer import ast_analyze_code

result = ast_analyze_code("""
// Thread 0
data.store(42, std::memory_order_relaxed);
flag.store(1, std::memory_order_release);
// Thread 1
int r0 = flag.load(std::memory_order_acquire);
int r1 = data.load(std::memory_order_relaxed);
""", language="cpp")

print(result.patterns_found[0].pattern_name)  # "mp"
print(result.parse_method)                     # "ast"
print(result.memory_orders_used)               # {"relaxed", "release", "acquire"}
```

**Returns:** `ASTAnalysisResult` with fields:
| Field | Type | Description |
|-------|------|-------------|
| `patterns_found` | `List[ASTPatternMatch]` | Matched patterns with confidence |
| `extracted_ops` | `List[ExtractedOp]` | Extracted memory operations |
| `n_threads` | `int` | Number of threads detected |
| `parse_method` | `str` | `"ast"` or `"fallback_regex"` |
| `memory_orders_used` | `Set[str]` | Memory orderings found |
| `dependencies_found` | `List[Dict]` | Inferred data/address/control deps |
| `is_gpu` | `bool` | Whether GPU code detected |

### `ast_check_portability(code, target_arch=None, language='auto') → List[Dict]`

Full pipeline: AST parse → match patterns → check portability.

```python
from ast_analyzer import ast_check_portability

bugs = ast_check_portability(code, target_arch="arm")
for bug in bugs:
    print(f"{bug['pattern']}: safe={bug['safe']}, fix={bug['fence_fix']}")
```

---

## Custom Memory Model DSL

### `register_model(dsl_text) → CustomModel`

Register a custom memory model from DSL text.

```python
from model_dsl import register_model, check_custom

register_model("""
model POWER {
    relaxes W->R, W->W, R->R, R->W
    preserves deps
    not multi-copy-atomic
    fence hwsync (cost=8) { orders W->R, W->W, R->R, R->W }
    fence lwsync (cost=4) { orders W->W, R->R, R->W }
    fence isync  (cost=2) { orders R->R }
}
""")

result = check_custom("mp", "POWER")
print(result)  # {'pattern': 'mp', 'model': 'POWER', 'safe': False, ...}
```

### `check_custom(pattern, model_name) → Dict`

Check a pattern against a registered model.

### `get_registry().compare_models(model_a, model_b) → List[Dict]`

Find discriminating patterns between two models.

### `list_models() → List[str]`

List all available models (built-in + custom).

### DSL Syntax

```
model <Name> [extends <Parent>] {
    [description "<text>"]
    relaxes <pair> [, <pair>]*       # W->R, W->W, R->R, R->W
    [preserves deps]                  # dependency preservation
    [not multi-copy-atomic]           # multi-copy atomicity
    [scope <level>]                   # workgroup, device
    fence <name> [(cost=<N>)] {       # fence type definition
        orders <pair> [, <pair>]*
    }
}
```

---

## Differential Testing

### `run_all_differential_tests() → Dict`

Run all automated cross-validation checks.

```python
from differential_testing import run_all_differential_tests

results = run_all_differential_tests()
# results keys: monotonicity (342), fence_soundness (60), determinism (2280),
#               custom_model (39), litmus_roundtrip (57)
```

**Test categories:**
| Category | Checks | Description |
|----------|--------|-------------|
| **Meaningful semantic checks** | **498** | |
| Monotonicity | 342 | Stricter model ⊆ weaker model |
| Fence soundness | 60 | Adding fences never makes safe → unsafe |
| Custom model | 39 | DSL vs built-in cross-validation (39/39 agree) |
| Litmus round-trip | 57 | Export → re-import consistency |
| **Trivial stability checks** | **2,280** | |
| Determinism | 2,280 | Same input → same output (5 runs) |

---

## SMT-Based Formal Validation

### `cross_validate_smt() → Dict`

Cross-validate all CPU results using independent Z3 SMT encoding.

```python
from smt_validation import cross_validate_smt

report = cross_validate_smt()
print(f"Agreement: {report['agree']}/{report['total_checks']}")
# 228/228, Wilson CI [98.3%, 100%]
```

### `classify_all_unsafe_pairs() → Dict`

Classify all unsafe CPU pairs into fence-sufficient, inherently observable, or partial fence.

```python
from smt_validation import classify_all_unsafe_pairs

results = classify_all_unsafe_pairs()
print(f"Fence-sufficient: {len(results['fence_sufficient'])}")   # 55
print(f"Inherently observable: {len(results['inherently_observable'])}")  # 40
print(f"Partial fence: {len(results['partial_fence'])}")          # 6
```

**Returns:** `Dict` with keys:
| Key | Type | Description |
|-----|------|-------------|
| `fence_sufficient` | `List[Dict]` | Pairs where fences make the forbidden outcome UNSAT |
| `inherently_observable` | `List[Dict]` | Pairs where forbidden outcome persists despite fences |
| `partial_fence` | `List[Dict]` | Pairs with insufficient partial fences |

### `generate_discriminating_litmus_test(model_a, model_b) → Dict`

Generate an SMT-based litmus test that discriminates two memory models.

```python
from smt_validation import generate_discriminating_litmus_test

disc = generate_discriminating_litmus_test('ARM', 'RISC-V')
print(disc['discriminating_pattern'])  # Pattern that differs between models
```

### `generate_all_model_discriminators() → Dict`

Find the minimal set of patterns that discriminates all model pairs.

```python
from smt_validation import generate_all_model_discriminators

result = generate_all_model_discriminators()
print(result['minimal_discriminating_set'])  # ['isa2', 'mp_addr']
print(result['coverage'])                     # '5/6 pairs'
```

### `prove_fence_sufficiency_smt(pattern, model) → Dict`

Prove fence sufficiency via SMT (unfenced=sat, fenced=unsat).

```python
from smt_validation import prove_fence_sufficiency_smt

proof = prove_fence_sufficiency_smt("mp", "ARM")
print(proof['fence_sufficient'])  # True (unfenced=sat, fenced=unsat)
```

### `run_full_statistical_analysis() → Dict`

Compute Wilson CIs, bootstrap CIs, and variance estimates for all metrics.

```python
from statistical_analysis import run_full_statistical_analysis

stats = run_full_statistical_analysis()
# Returns: accuracy CIs, fence cost distributions, timing characterization
```

---

## herd7 Validation

### `validate_against_herd7() → Dict`

Validate all results against herd7 expected outcomes with CIs.

```python
from herd7_validation import validate_against_herd7

results = validate_against_herd7()
print(f"Agreement: {results['agreements']}/{results['total_checks']}")
# 50/50, Wilson CI [92.9%, 100%]
```

### `export_all_litmus(output_dir, fmt='C') → (List[str], List[Tuple])`

Export all 75 patterns as .litmus files.

---

## False-Negative Analysis

### `run_false_negative_analysis() → Dict`

Classify all 18 near-miss benchmark cases to verify zero false negatives.

```python
from false_negative_analysis import run_false_negative_analysis

results = run_false_negative_analysis()
print(f"SAFE: {results['safe_count']}")     # 9
print(f"NEUTRAL: {results['neutral_count']}") # 9
print(f"UNSAFE: {results['unsafe_count']}")   # 0
# Zero false negatives: 100% effective safety rate
```

**Returns:** `Dict` with keys:
| Key | Type | Description |
|-----|------|-------------|
| `classifications` | `List[Dict]` | Per-snippet classification with reasoning |
| `safe_count` | `int` | Conservative mismatches (safe, not bugs) |
| `neutral_count` | `int` | Identical portability profiles (harmless) |
| `unsafe_count` | `int` | Actual false negatives (should be 0) |

---

## Mismatch Analysis

### `run_full_analysis() → Dict`

Statistical analysis of DSL cross-validation mismatches.

```python
from mismatch_analysis import run_full_analysis

results = run_full_analysis()
# Includes: confusion matrix, chi-squared test, McNemar's test,
# root cause taxonomy, pattern discrimination power
```

---

## Completeness Analysis

### `run_completeness_analysis() → Dict`

Analyze pattern portfolio completeness and discrimination power.

```python
from completeness_analysis import run_completeness_analysis

results = run_completeness_analysis()
# Includes: boundary coverage (9/9), information theory,
# minimal discriminating set, structural coverage
```

---

## SMT-Based Litmus Test Synthesis

### `synthesize_litmus_test_smt(model_a, model_b, n_threads=2, n_ops_per_thread=2, n_addresses=2) → Dict`

Synthesize NEW litmus tests from scratch that discriminate two memory models.

```python
from smt_validation import synthesize_litmus_test_smt

result = synthesize_litmus_test_smt('TSO', 'ARM')
# result['synthesized'] == True
# result['tests'][0]['test_description'] == ['Thread 0: St x=1 ; St y=1', 'Thread 1: Ld y ; Ld x']
# Independently rediscovers the classic MP pattern!
```

### `run_litmus_synthesis() → Dict`

Run synthesis across all model pairs.

---

## Statistical Power Analysis

### `analyze_power(results) → Dict`

Post-hoc power analysis with Clopper-Pearson exact CIs.

```python
from statistical_analysis import analyze_power
power = analyze_power(benchmark_results)
# power['top3_match']['power_vs_95pct'] == 1.0  (>99% power)
# power['exact_match']['clopper_pearson_95ci'] == [0.725, 0.891]
```

### `clopper_pearson_ci(successes, total, alpha=0.05) → Tuple`

Conservative exact binomial confidence interval.

### `power_analysis(n, observed_rate, null_rate, alpha=0.05) → float`

Post-hoc statistical power for binomial proportion test.

---

## Portability Engine API

### `analyze_all_patterns() → Dict`
Full 750-pair analysis.

### `diff_architectures(arch1, arch2) → List`
Find discriminating patterns between two architectures.

### `detect_scope_mismatches() → Tuple[List, List]`
Detect GPU scope mismatch patterns. Returns (critical, warnings).

---

## API Layer (api.py)

### `find_fence_bugs(code, architecture) → List[FenceBug]`

Detect fence insufficiency and scope mismatch bugs.

### `minimize_fences(code, target_arch) → OptimizedCode`

Compute per-thread minimal fences with cost savings.

### `compare_architectures(test, archs) → ArchComparisonTable`

Compare behavior across multiple architectures.

### `validate_gpu_kernel(kernel, scope) → ValidationResult`

Validate GPU kernel scope correctness.

---

## Built-in Litmus Tests

75 built-in patterns across categories:

| Category | Count | Examples |
|----------|-------|---------|
| Basic ordering | 9 | MP, SB, LB, IRIW, WRC |
| Fenced | 28 | MP+fence, SB+fence, IRIW+fence, ISA2+fence, Dekker+fence |
| Asymmetric fence | 6 | MP+fence_wr, SB+fence_wr |
| GPU scope | 18 | gpu_mp_wg, gpu_sb_dev |
| Dependency | 6 | MP+data, LB+data, WRC+addr |
| Coherence | 4 | CoRR, CoWR, CoWW, CoRW |
| Multi-thread | 5 | ISA2, R, S, 3SB |
| Mutex | 2 | Dekker, Peterson |

## Supported Architectures

Built-in: `x86`, `sparc`, `arm`, `riscv`, `opencl_wg`, `opencl_dev`, `vulkan_wg`, `vulkan_dev`, `ptx_cta`, `ptx_gpu`

Custom (via DSL): unlimited
