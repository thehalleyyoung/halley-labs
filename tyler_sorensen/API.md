# LITMUS∞ — API Reference

All features listed below are implemented and functional.
Install from source only (not on PyPI): `cd litmus_inf && pip install -e .`

## CLI: `litmus-check`

```bash
litmus-check --target arm myfile.c           # single file
litmus-check --target arm src/               # scan directory
litmus-check --target arm --json src/        # JSON output for CI
litmus-check --target arm --stdin            # read from stdin
litmus-check --target arm --emit-certificate src/  # export .smt2 proofs
```

| Flag | Description |
|------|-------------|
| `--target`, `-t` | Target architecture: x86, sparc, arm, riscv, opencl_wg, opencl_dev, vulkan_wg, vulkan_dev, ptx_cta, ptx_gpu |
| `--source`, `-s` | Source architecture (default: x86) |
| `--stdin` | Read code from stdin |
| `--json` | Output JSON |
| `--emit-certificate` | Export SMT-LIB2 proof certificates (.smt2 files) |
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

### `analyze_all_patterns()` — Full 1,400-pair analysis (140 patterns × 10 architectures)

### `diff_architectures(arch1, arch2)` — Discriminating patterns between two models

### `detect_scope_mismatches()` — GPU scope mismatch detection

---

## AST-Based Code Analysis

### `ast_analyze_code(code, language='auto')`

Returns `ASTAnalysisResult` with `patterns_found`, `parse_method`, and `coverage_confidence` (0.0–1.0). Emits `UnrecognizedPatternWarning` when coverage < 50%.

### `ast_check_portability(code, target_arch=None, language='auto')`

Full pipeline: AST parse → match patterns → check portability. Supports C11 atomics, GCC builtins, and Linux kernel macros. **96.6% exact-match on 203 curated benchmark snippets.**

---

## LLM-Assisted Pattern Recognition

### `hybrid_check_portability(code, target_arch, use_llm=True, llm_model='gpt-4.1-nano')`

```python
from llm_pattern_recognizer import hybrid_check_portability
results = hybrid_check_portability(code, target_arch="arm", use_llm=True)
# Falls back to LLM when AST confidence is low
# All LLM-suggested patterns undergo full SMT verification (soundness preserved)
```

### `llm_recognize_patterns(code, target_arch=None, model='gpt-4.1-nano')`

Direct LLM pattern recognition. Returns `{'patterns': [...], 'confidence': 0.9, 'reasoning': '...'}`.

**Adversarial OOD accuracy: 85.0% with GPT-4.1 on 113 snippets across 12 domains (vs 6.2% AST-only).**

Requires `OPENAI_API_KEY` environment variable. Soundness: LLM affects recall only; all verdicts are SMT-certified.

---

## Cross-Solver Validation

```python
from cross_solver_validation import run_cross_solver_validation
report = run_cross_solver_validation()
# 1,400/1,400 agreement across Z3 and CVC5
# Wilson CI [99.7%, 100%]
```

Validates all SMT-LIB2 certificates through:
1. Z3 with proof production (seed=0)
2. Z3 diversity check (seed=42)
3. CVC5 (independent solver, v1.3.2)

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

170/171 (99.4%) empirical correspondence with herd7 `.cat` specifications.

---

## Alethe Proof Certificate Extraction

```python
from alethe_proof_extractor import Z3ProofExtractor, run_alethe_extraction
extractor = Z3ProofExtractor()
cert = extractor.extract_proof_for_pattern('mp_fence', 'arm')
# cert.verdict == 'unsat'
# cert.proof_size_steps == 88

summary = run_alethe_extraction()
# 993 UNSAT Alethe proofs + 407 SAT verified models
```

---

## SMT Validation and Certificate Export

| API | Module | Result |
|-----|--------|--------|
| `cross_validate_all_smt()` | smt_validation.py | 1,400/1,400 Z3 certificates |
| `generate_all_smtlib_certificates(output_dir)` | smtlib_certificate_extractor.py | 1,400 .smt2 files |

---

## Severity Classification

```python
from severity_classification import classify_all_unsafe_pairs
report = classify_all_unsafe_pairs()
# {'data_race': 689, 'security_vulnerability': 44, 'benign': 70}
```

CWE-calibrated (not CVE-validated):
- **data_race** → CWE-362, CWE-366, CWE-820
- **security_vulnerability** → CWE-667, CWE-821
- **benign** → No direct CWE mapping

---

## Architectures

Built-in: `x86`, `sparc`, `arm`, `riscv`, `opencl_wg`, `opencl_dev`, `vulkan_wg`, `vulkan_dev`, `ptx_cta`, `ptx_gpu`

Custom (via DSL): user-defined models with arbitrary relaxation and fence specifications

---

## Pattern Composition and Beyond-140 Analysis

### `compose_patterns(pattern1_name, pattern2_name, shared_vars=None)`

```python
from pattern_composition import compose_patterns
test = compose_patterns('mp', 'sb')
# Creates composite 4-thread test combining MP and SB patterns
```

### `generate_chain(base_pattern, n_threads)`

```python
from pattern_composition import generate_chain
chain = generate_chain('mp', 4)
# Generates 4-thread MP chain: T0→T1→T2→T3
```

### `bounded_model_check(n_threads, n_ops_per_thread, addresses, target_arch)`

```python
from pattern_composition import bounded_model_check
hazards = bounded_model_check(2, 2, ['x', 'y'], 'arm')
# Systematically enumerates all possible litmus test structures
```

### `run_pattern_composition_experiments()`

Runs all composition experiments and returns JSON-serializable results. 29 composite tests, 354 hazards found via BMC.

---

## DSL-to-.cat Mismatch Analysis

### `dsl_cat_mismatch_analysis.py`

```python
python3 dsl_cat_mismatch_analysis.py
```

Full characterization of the single DSL-to-.cat disagreement (mp_fence_wr on RISC-V). Proves mismatch is isolated (1/13 asymmetric-fence patterns), characterizes direction (false negative), and quantifies practical impact (minimal).
