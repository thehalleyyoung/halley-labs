# LITMUS∞: Cross-Architecture Memory Model Portability Checker

**Is your concurrent C/C++ code safe to port from x86 to ARM?**
LITMUS∞ answers in under 1ms per pattern, with per-thread minimal fence fixes.

## 30-Second Quickstart

```bash
pip install -e .   # install litmus-check CLI

# Check a C++ file for x86→ARM portability issues:
litmus-check --target arm myfile.cpp

# Or pipe a snippet:
echo '// Thread 0
data.store(42, std::memory_order_release);
flag.store(1, std::memory_order_release);
// Thread 1
r0 = flag.load(std::memory_order_acquire);
r1 = data.load(std::memory_order_acquire);' | litmus-check --target arm --stdin
```

Output:
```
1 portability issue(s) found (x86 → arm):
  <stdin>: ✗ UNSAFE  mp → arm  (confidence: 100%)
    Fix: dmb ishst (T0); dmb ishld (T1)

Summary: 5 pattern(s) checked, 1 issue(s) across 1 file(s)
```

## Key Results

| Metric | Value | Notes |
|--------|-------|-------|
| Portability pairs | 750 (75 patterns × 10 architectures) | CPU + GPU |
| Code analyzer accuracy | 92.7% exact, 100% top-3 (n=111) | Stratified per-category below |
| SMT internal consistency | 228/228 (CPU-only) | Same-author encodings; not external validation |
| Fence proofs | 55 UNSAT + 40 SAT (Z3) | Machine-checked certificates |
| herd7 agreement | 50/50 | Wilson CI [92.9%, 100%] |
| Hardware consistency | 25/25 | Published observations |
| Meaningful differential tests | 498 pass | + 2,280 trivial determinism checks |
| Real-code evaluation | 93.3% exact on 15 snippets | Linux kernel, Folly, LLVM provenance |

### Per-Category Accuracy (n=111)

| Category | N | Exact | Top-3 |
|----------|---|-------|-------|
| C++ atomics | 11 | 100% | 100% |
| kernel (Linux) | 8 | 100% | 100% |
| real_code (provenance) | 15 | 93.3% | 100% |
| gpu | 9 | 55.6% | 100% |

## Usage

### CLI: Scan source files

```bash
litmus-check --target arm src/           # scan directory
litmus-check --target arm --json src/    # JSON output for CI
litmus-check --target riscv myfile.c     # check single file
```

### Python API

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

### GitHub Actions

Add `.github/workflows/litmus-check.yml` to check PRs automatically.
See `.github/workflows/litmus-check.yml` for the ready-to-use workflow.

### Reproduce Paper Results

```bash
python3 run_paper_experiments.py         # All experiments
python3 benchmark_suite.py              # 111-snippet benchmark
python3 smt_validation.py              # Z3 proofs (CPU-only)
python3 differential_testing.py         # Differential tests
pdflatex paper.tex && pdflatex paper.tex  # Compile paper
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
- GPU models have **zero SMT validation** — formal claims are CPU-only
- Fence costs are analytical weights, not hardware latencies
- 228/228 SMT is internal consistency (same-author encodings), not external validation
- Pattern-level safety does not compose to program-level safety

## Dependencies

- Python 3.8+
- z3-solver (for SMT validation: `pip install z3-solver`)
- tree-sitter, tree-sitter-c, tree-sitter-cpp (for AST analysis, optional)
- herd7 (for independent validation, optional)
