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
| Portability pairs | 750 (75 patterns × 4 CPU + 1 GPU model ×6 scopes) | CPU + GPU |
| Code analyzer accuracy | 96.6% exact, 98.0% top-3 (n=203) | 16 categories, 4 languages |
| SMT internal consistency (CPU) | 228/228 | Wilson CI [98.3%, 100%] |
| SMT internal consistency (GPU) | 108/108 | Wilson CI [96.6%, 100%] |
| Fence proofs | 55 UNSAT + 40 SAT (Z3) | Machine-checked certificates |
| herd7 agreement | 228/228 | Wilson CI [98.3%, 100%] |
| Hardware consistency | 25/25 | Published observations |
| Meaningful differential tests | 642 pass | + 3,000 trivial determinism checks |
| Effective safety rate | 100% (203/203) | 4 SAFE + 3 NEUTRAL non-exact-matches, 0 UNSAFE |

### Per-Category Accuracy (n=203)

| Category | N | Exact | Top-3 |
|----------|---|-------|-------|
| application | 23 | 100% | 100% |
| data_structure | 15 | 100% | 100% |
| systems | 3 | 100% | 100% |
| riscv | 8 | 100% | 100% |
| basic | 5 | 100% | 100% |
| allocator | 3 | 100% | 100% |
| mca | 2 | 100% | 100% |
| fenced | 14 | 100% | 100% |
| multi_thread | 14 | 100% | 100% |
| gpu | 19 | 100% | 100% |
| coherence | 6 | 100% | 100% |
| real_code | 23 | 100% | 100% |
| synchronization | 27 | 96% | 96% |
| cpp_atomics | 20 | 90% | 90% |
| kernel | 16 | 88% | 94% |
| dependency | 5 | 60% | 100% |

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
python3 benchmark_suite.py              # 203-snippet benchmark
python3 smt_validation.py              # Z3 proofs (CPU + GPU)
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
- Fence costs are analytical weights, not hardware latencies
- 228/228 CPU SMT + 108/108 GPU SMT is internal consistency (same-author encodings)
- Pattern-level safety does not compose to program-level safety
- 0/7 non-exact-matches are UNSAFE (all SAFE or NEUTRAL)

## Dependencies

- Python 3.8+
- z3-solver (for SMT validation: `pip install z3-solver`)
- tree-sitter, tree-sitter-c, tree-sitter-cpp (for AST analysis, optional)
- herd7 (for independent validation, optional)
