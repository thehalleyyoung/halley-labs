# LITMUS∞ — API Reference

All features listed below are implemented and functional.
Install from source only (not on PyPI): `cd litmus_inf && pip install -e .`

## CLI: `litmus-check`

```bash
litmus-check --target arm myfile.c           # single file
litmus-check --target arm src/               # scan directory
litmus-check --target arm --json src/        # JSON output for CI
litmus-check --target arm --stdin            # read from stdin
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

170/171 (99.4%) empirical correspondence with herd7 `.cat` specifications. DSL lacks formal semantics.

---

## SMT Validation and Certificate Export

| API | Module | Result |
|-----|--------|--------|
| `cross_validate_all_750_smt()` | smt_validation.py | 750/750 Z3 certificates |
| `cross_validate_smt()` | smt_validation.py | 228/228 CPU internal consistency |
| `cross_validate_gpu_smt()` | smt_validation.py | 108/108 GPU internal consistency |
| `prove_fence_sufficiency_smt(pattern, model)` | smt_validation.py | Single fence proof |
| `generate_discriminating_litmus_test(model_a, model_b)` | smt_validation.py | SMT discriminator |
| `generate_smtlib2_encoding(pattern, model)` | smtlib_certificate_extractor.py | Single .smt2 file |
| `generate_all_smtlib_certificates(output_dir)` | smtlib_certificate_extractor.py | 750 .smt2 files |

---

## Compositional Reasoning

| API | Module | Result |
|-----|--------|--------|
| `analyze_program_compositionally(program, arch)` | compositional_reasoning.py | Safe/unsafe with composition type |
| `identify_patterns_in_program(program)` | compositional_reasoning.py | Pattern extraction |
| `check_disjoint_composition(patterns)` | compositional_reasoning.py | Theorem 6: disjoint-variable soundness |

Limitation: shared-variable composition uses conservative analysis (flags all interactions).

---

## Severity Classification

```python
from severity_classification import classify_all_unsafe_pairs
report = classify_all_unsafe_pairs()
# {'data_race': 228, 'security_vulnerability': 44, 'benign': 70}
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
