# LITMUS∞: SMT-Verified Memory Model Portability Checker

Instantly check if your concurrent C/C++/CUDA code will break when ported from x86 to ARM, RISC-V, or GPU architectures. Every result has a Z3 proof certificate, cross-validated with independent solver configurations.

## 30-Second Quickstart

```bash
pip install -e .
litmus-check --target arm myfile.c
```

Output:
```
UNSAFE: mp (data_race) → dmb ishst (T0); dmb ishld (T1)  [Z3: SAT]
  ↳ 62.5% cheaper than full barrier
UNSAFE: sb (data_race) → dmb ish (T0); dmb ish (T1)      [Z3: SAT]
SAFE:   mp_fence                                           [Z3: UNSAT]
3 patterns matched, 2 unsafe, 1 safe  (12ms)
```

## Most Impressive Demo

```bash
# Check the full portability matrix: 140 patterns × 10 architectures = 1,400 pairs
python3 portcheck.py --analyze-all
# → 1,400/1,400 Z3 certificates in <1 second. Zero timeouts. Every result proven.

# Cross-solver validation: replay all certificates through independent Z3 config
python3 cross_solver_validation.py
# → 1,400/1,400 agreement (Wilson CI [99.7%, 100%])

# LLM-assisted recognition on out-of-distribution code
python3 llm_pattern_recognizer.py --benchmark --model gpt-4.1
# → 85.0% exact-match on 113 adversarial OOD snippets (vs 6.2% AST-only)

# Export all proofs as standalone SMT-LIB2 files
python3 smtlib_certificate_extractor.py --all
# → 597 UNSAT proofs + 803 SAT witnesses written to paper_results_v6/smtlib_certificates/
```

## Real Code Analysis

```python
from ast_analyzer import ast_check_portability

code = """
// Thread 0                     // Thread 1
data = value;                   if (flag) {
flag = 1;                           use(data);
                                }
"""
bugs = ast_check_portability(code, target_arch="arm")
# → {'pattern': 'mp', 'safe': False, 'fence_fix': 'dmb ishst (T0); dmb ishld (T1)'}
```

For out-of-distribution code, use LLM-assisted recognition:

```python
from llm_pattern_recognizer import hybrid_check_portability
results = hybrid_check_portability(code, target_arch="arm", use_llm=True)
```

## CLI Usage

```bash
litmus-check --target arm src/               # scan directory
litmus-check --target riscv --json src/      # JSON for CI pipelines
litmus-check --target arm --verbose src/     # include safe patterns

# Reproduce all paper results
python3 run_paper_experiments.py
python3 run_phase_b2_experiments.py
python3 cross_solver_validation.py
python3 llm_pattern_recognizer.py --benchmark
```

## Key Results

| Metric | Result | Evidence |
|--------|--------|----------|
| Z3 certificate coverage | **1,400/1,400 (100%)** | 597 UNSAT proofs + 803 SAT witnesses, zero timeouts |
| Cross-solver validation | **1,400/1,400 agreement** | Two Z3 configs (seed-0 proof, seed-42 diversity); Wilson CI [99.7%, 100%] |
| LLM-assisted OOD accuracy | **85.0% exact-match** | GPT-4.1 on 113 adversarial snippets across 12 domains (vs 6.2% AST-only) |
| Alethe proof extraction | **993 UNSAT proofs + 407 SAT models** | Avg 106.4 proof steps; independently checkable |
| herd7 internal consistency | **228/228** | All CPU models; Wilson CI [98.3%, 100%] |
| GPU external validation | **94/94** | Published litmus tests, PTX/Vulkan specs, known GPU bugs |
| DSL-to-.cat correspondence | **170/171 (99.4%)** | TSO, ARM, RISC-V |
| Code analyzer accuracy | **96.6% exact, 97.0% top-3** | n=203 curated benchmark |
| Severity classification | 689 data_race, 44 security, 70 benign | 803 unsafe pairs |
| Pattern library | **140 patterns** | Classical, RMW, lock-free, seqlock, RCU, hazard ptr, etc. |
| Analysis speed | **sub-second** (1,400 pairs) | <1ms per pattern |

## Supported Architectures

| Architecture | Model | Relaxations |
|-------------|-------|-------------|
| x86 / x86-64 | TSO | Store→load only |
| SPARC | PSO | Store→load, store→store |
| ARM (v7/v8) | ARMv8 | All four; preserves deps |
| RISC-V | RVWMO | All four; asymmetric fences |
| OpenCL | WG / Device | Scoped synchronization |
| Vulkan | WG / Device | SPIR-V memory model |
| PTX (CUDA) | CTA / GPU | Scoped membar |

Custom models: `model MyModel { relaxes W->R, W->W ... }`

## Limitations

- **Pattern-level only:** 140 built-in litmus test patterns, not arbitrary programs
- **No coverage audit:** 140-pattern universe not validated against real bug databases
- **Compositional reasoning is conservative:** exact for disjoint vars, sound but imprecise for shared vars
- **Severity taxonomy is CWE-calibrated**, not CVE-validated
- **Fence costs are analytical weights,** not measured hardware latencies
- **Z3 in trusted computing base:** cross-solver validation reduces but does not eliminate TCB
- **LLM recognition requires API key:** set OPENAI_API_KEY for LLM mode
- **GPU models lack hardware validation** (cross-validated against specs, not tested on hardware)
- **Theorems are paper proofs,** not mechanized in Coq/Isabelle/Lean
- **Not published to PyPI** — install from source

## Dependencies

- Python 3.8+
- z3-solver (`pip install z3-solver`)
- tree-sitter, tree-sitter-c, tree-sitter-cpp (optional, for AST analysis)
- openai (optional, for LLM-assisted recognition)
