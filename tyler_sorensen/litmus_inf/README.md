# LITMUS∞: Cross-Architecture Memory Model Portability Checker

**Is your concurrent C/C++ code safe to port from x86 to ARM?**
LITMUS∞ answers in under 1ms per pattern, with per-thread minimal fence fixes, severity classification, and Z3 certificates for every result.

## 30-Second Quickstart

```bash
pip install -e .   # install litmus-check CLI

# Check a C++ file for x86→ARM portability issues:
litmus-check --target arm myfile.cpp

# Or use the Python API directly:
python3 -c "
from portcheck import check_portability
result = check_portability('mp', target_arch='arm')
print(result)
"
```

Output:
```
Pattern: mp  Target: arm  Result: UNSAFE
Severity: data_race
Fence fix: dmb ishst (T0); dmb ishld (T1)
Cost saving: 62.5% vs full dmb ish
Z3 certificate: SAT (forbidden outcome reachable)
```

### Analyze real code:
```python
from ast_analyzer import ast_analyze_code

result = ast_analyze_code("""
// Thread 0
data.store(42, std::memory_order_relaxed);
flag.store(1, std::memory_order_release);
// Thread 1
r0 = flag.load(std::memory_order_acquire);
r1 = data.load(std::memory_order_relaxed);
""")
print(result.patterns_found[0])  # mp (confidence=1.00)
```

## Key Results

| Metric | Value | Notes |
|--------|-------|-------|
| Z3 certificate coverage | **750/750 (100%)** | Every pair machine-checked |
| Severity classification | 228 data race, 44 security, 70 benign | 342 unsafe pairs classified |
| DSL-to-.cat correspondence | **170/171 (99.4%)** | TSO 100%, ARM 100%, RISC-V 98.2% |
| herd7 agreement | **228/228** | Wilson CI [98.3%, 100%] |
| Code analyzer accuracy | 96.6% exact, 98.0% top-3 (n=203) | Wilson CI [93.1%, 98.3%] |
| SMT consistency (CPU+GPU) | 228/228 + 108/108 | Independent Z3 validation |
| Fence proofs | 55 UNSAT + 40 SAT | Machine-checked certificates |
| Portability pairs | 750 (75 × 10 configs) | 4 CPU + 6 GPU scope |
| Speed | <200ms for all 750 pairs | Sub-ms per pattern |

## Usage

```bash
# Pattern-level checks
python3 portcheck.py --pattern mp --target arm
python3 portcheck.py --analyze-all
python3 portcheck.py --scope-mismatch          # GPU scope bugs
python3 portcheck.py --diff arm riscv           # model boundary

# Code analysis
litmus-check --target arm src/                  # scan directory
litmus-check --target arm --json src/           # JSON for CI

# Full experiments (Z3 certificates, severity, DSL-.cat)
python3 run_phase_b_experiments.py              # all Phase B experiments
python3 severity_classification.py              # severity classification
python3 dsl_cat_correspondence.py               # DSL-.cat validation

# SMT validation
python3 smt_validation.py                       # Z3 proofs
python3 herd7_validation.py                     # herd7 comparison
```

### Reproduce Paper Results

```bash
python3 run_phase_b_experiments.py      # all experiments (saves to paper_results_v5/)
python3 benchmark_suite.py              # 203-snippet benchmark
pdflatex paper.tex && pdflatex paper.tex  # compile paper
```

## Supported Architectures

| Architecture | Model | Key Feature |
|-------------|-------|-------------|
| x86 / x86-64 | TSO | Only relaxes store→load |
| SPARC | PSO | Also relaxes store→store |
| ARM (v7/v8) | ARMv8 | All reorderings; preserves dependencies |
| RISC-V | RVWMO | Asymmetric fences (fence pred,succ) |
| OpenCL | WG / Device | Scoped synchronization |
| Vulkan | WG / Device | SPIR-V memory model |
| PTX (CUDA) | CTA / GPU | Scoped membar |

## Limitations

- Operates on 75 built-in litmus patterns, not arbitrary programs
- Fence costs are analytical weights, not hardware latencies
- SMT consistency is internal (same-author encodings); herd7 provides independent validation
- Pattern-level safety does not compose to program-level safety (Proposition 7)
- GPU model is a conservative approximation (1 parameterized model, 6 scope instantiations)
- Theorems 1-3 are paper proofs, not mechanized in a proof assistant
- DSL cannot express RISC-V asymmetric fence pred/succ (1/171 mismatch)

## Dependencies

- Python 3.8+
- z3-solver (for SMT validation: `pip install z3-solver`)
- tree-sitter, tree-sitter-c, tree-sitter-cpp (for AST analysis, optional)
- herd7 (for independent validation, optional)
