# LITMUS∞: SMT-Verified Memory Model Portability Checker

Instantly check if your concurrent C/C++/CUDA code will break when ported from x86 to ARM, RISC-V, or GPU architectures. Every result has a Z3 proof certificate.

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
# Check the full portability matrix: 75 patterns × 10 architectures = 750 pairs
python3 portcheck.py --analyze-all
# → 750/750 Z3 certificates in <1 second. Zero timeouts. Every result proven.

# Export all proofs as standalone SMT-LIB2 files (verifiable by any solver)
python3 smtlib_certificate_extractor.py --all
# → 459 UNSAT proofs + 291 SAT witnesses written to paper_results_v6/smtlib_certificates/

# Find what breaks when porting from ARM to RISC-V
python3 portcheck.py --diff arm riscv
# → mp_fence_wr: sole discriminator (asymmetric fence semantics)
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

Recognizes C11 atomics, GCC builtins (`__sync_synchronize`), and kernel macros (`smp_store_release`, `READ_ONCE`, `WRITE_ONCE`).

## CLI Usage

```bash
litmus-check --target arm src/               # scan directory
litmus-check --target riscv --json src/      # JSON for CI pipelines
litmus-check --target arm --verbose src/     # include safe patterns

# Reproduce all paper results
python3 run_paper_experiments.py
python3 run_phase_b2_experiments.py
```

## Key Results

| Metric | Result | Evidence |
|--------|--------|----------|
| Z3 certificate coverage | **750/750 (100%)** | 459 UNSAT proofs + 291 SAT witnesses, zero timeouts |
| SMT-LIB2 proof export | **750 .smt2 files** | Independently verifiable by any SMT-LIB solver |
| herd7 internal consistency | **228/228** | All CPU models; Wilson CI [98.3%, 100%] |
| GPU SMT internal consistency | **108/108** | 18 patterns × 6 models; no external oracle |
| Machine-checked fence proofs | **55 UNSAT + 40 SAT** | Fence sufficiency / inherent observability |
| DSL-to-.cat correspondence | **170/171 (99.4%)** | TSO, ARM, RISC-V |
| Code analyzer accuracy | **93.0% exact, 94.0% top-3** | n=501; Wilson CI [90.4%, 94.9%] |
| Benchmark sources | **10 projects** | Linux kernel, Folly, LLVM, Abseil, Boost, DPDK, crossbeam, + 3 more |
| Compositional analysis | **5 multi-pattern programs** | Disjoint + shared-variable composition |
| Severity classification | 228 data_race, 44 security, 70 benign | 342 unsafe pairs |
| Analysis speed | **sub-second** (750 pairs) | <1ms per pattern |

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

- **Pattern-level only:** 75 built-in litmus test patterns, not arbitrary programs
- **No coverage audit:** 75-pattern universe not validated against real bug databases; practical coverage unknown
- **Compositional reasoning is conservative:** shared-variable programs flagged as potentially unsafe; exact composition only for disjoint variables; rely-guarantee extension is future work
- **Severity taxonomy is CWE-calibrated** (CWE-362 for data races, CWE-667 for security), but not validated against specific CVE instances
- **Fence costs are analytical weights,** not measured hardware latencies
- **Z3 in trusted computing base:** SMT-LIB2 exports enable solver cross-checking, but no independent LFSC/Alethe proof checking
- **SMT consistency is internal** (same-author encodings); herd7 provides independent evidence for CPU models
- **GPU models lack external validation** (108/108 is internal consistency only, no GPU-litmus comparison)
- **Theorems are paper proofs,** not mechanized in Coq/Isabelle/Lean
- **Benchmark is author-sampled** (501 snippets from 10 projects); not independently curated
- **Not published to PyPI** — install from source

## Dependencies

- Python 3.8+
- z3-solver (`pip install z3-solver`)
- tree-sitter, tree-sitter-c, tree-sitter-cpp (optional, for AST analysis)
