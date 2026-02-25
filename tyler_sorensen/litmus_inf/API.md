# LITMUS∞ — API Reference

All features listed below are implemented and functional.
Install from source only (not on PyPI): `pip install -e .`

## CLI: `litmus-check`

```bash
litmus-check --target arm myfile.c       # single file
litmus-check --target arm src/           # scan directory
litmus-check --target arm --json src/    # JSON output for CI
litmus-check --target arm --stdin        # read from stdin
```

| Flag | Description |
|------|-------------|
| `--target`, `-t` | Target architecture: x86, sparc, arm, riscv, opencl_wg, opencl_dev, vulkan_wg, vulkan_dev, ptx_cta, ptx_gpu |
| `--source`, `-s` | Source architecture (default: x86) |
| `--stdin` | Read code from stdin |
| `--json` | Output JSON |
| `--verbose`, `-v` | Show safe patterns too |
| `--no-color` | Disable ANSI colors |

---

## Core: Pattern-Based Portability Checking

### `check_portability(pattern_name, source_arch='x86', target_arch=None)`

```python
from portcheck import check_portability
result = check_portability("mp", target_arch="arm")
# result[0].safe == False
# result[0].fence_recommendation == "dmb ishst (T0); dmb ishld (T1)"
```

### `analyze_all_patterns()` — Full 750-pair analysis

### `diff_architectures(arch1, arch2)` — Discriminating patterns between two models

### `detect_scope_mismatches()` — GPU scope mismatch detection (6 critical + 5 warning)

---

## AST-Based Code Analysis

### `ast_analyze_code(code, language='auto')`

Returns `ASTAnalysisResult` with `patterns_found`, `parse_method`, and `coverage_confidence` (0.0–1.0). Emits `UnrecognizedPatternWarning` when coverage < 50%.

```python
from ast_analyzer import ast_analyze_code
result = ast_analyze_code(code, language="cpp")
# result['patterns_found'][0]['pattern_name'] == "mp"
# result['coverage_confidence'] == 0.92
```

### `ast_check_portability(code, target_arch=None, language='auto')`

Full pipeline: AST parse → match patterns → check portability. Supports C11 atomics, GCC builtins, and Linux kernel macros.

---

## Custom Memory Model DSL

```python
from model_dsl import register_model, check_custom
register_model("""
model POWER {
    relaxes W->R, W->W, R->R, R->W
    preserves deps
    fence hwsync (cost=8) { orders W->R, W->W, R->R, R->W }
}
""")
result = check_custom("mp", "POWER")
```

170/171 (99.4%) empirical correspondence with herd7 `.cat` specs. DSL lacks formal semantics.

---

## SMT Validation and Certificate Export

### `cross_validate_all_750_smt()` — 750/750 Z3 certificates

### `cross_validate_smt()` — 228/228 CPU SMT internal consistency

### `cross_validate_gpu_smt()` — 108/108 GPU SMT internal consistency

### `prove_fence_sufficiency_smt(pattern, model)` — Single fence proof

### `generate_discriminating_litmus_test(model_a, model_b)` — SMT-based discriminator

### `generate_smtlib2_encoding(pattern_name, model_name)` — Single .smt2 certificate

```python
from smtlib_certificate_extractor import generate_smtlib2_encoding
cert = generate_smtlib2_encoding("mp", "ARM")
# cert['result'] == 'UNSAT'
# cert['unsat_core'] == ['rf_constraint_0', 'co_total_0_1', ...]
# cert['smtlib2_text'] starts with '(set-logic QF_LIA)'
```

### `generate_all_smtlib_certificates(output_dir)` — Export all 750 .smt2 files

Writes 459 files to `unsat_proofs/` and 291 to `sat_witnesses/`, plus `certificate_index.json`.

---

## Compositional Reasoning

### `analyze_program_compositionally(program, architecture)` — Multi-pattern analysis

```python
from compositional_reasoning import analyze_program_compositionally
result = analyze_program_compositionally(program, "arm")
# result['safe'] == True/False
# result['composition_type'] == 'disjoint' or 'shared_variable'
```

Limitation: shared-variable composition uses conservative analysis (flags all interactions).

### `identify_patterns_in_program(program)` — Pattern extraction

### `check_disjoint_composition(patterns)` — Theorem 6: disjoint ⇒ safe iff all safe

---

## Severity Classification

### `classify_all_unsafe_pairs()` (severity_classification.py)

```python
from severity_classification import classify_all_unsafe_pairs
report = classify_all_unsafe_pairs()
# report['severity_counts'] == {'data_race': 228, 'security_vulnerability': 44, 'benign': 70}
```

CWE-calibrated (not CVE-validated):
- **data_race** → CWE-362, CWE-366, CWE-820
- **security_vulnerability** → CWE-667, CWE-821
- **benign** → No direct CWE mapping

---

## Validation APIs

| API | Module | Result |
|-----|--------|--------|
| `validate_against_herd7()` | herd7_validation.py | 228/228 |
| `run_all_differential_tests()` | differential_testing.py | 642 meaningful + 3,000 determinism |
| `validate_all_models()` | dsl_cat_correspondence.py | 170/171 |
| `run_false_negative_analysis()` | false_negative_analysis.py | 4 SAFE, 3 NEUTRAL, 0 UNSAFE |

---

## Architectures

Built-in: `x86`, `sparc`, `arm`, `riscv`, `opencl_wg`, `opencl_dev`, `vulkan_wg`, `vulkan_dev`, `ptx_cta`, `ptx_gpu`

Custom (via DSL): user-defined models with arbitrary relaxation and fence specifications
