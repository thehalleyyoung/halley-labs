# LITMUS∞: SMT-Verified Memory Model Portability Checking

Advisory pre-screening tool that checks whether concurrent C/C++/CUDA code is safe to port from x86 to ARM, RISC-V, or GPU architectures. Every result carries an independently checkable proof certificate (Alethe resolution proofs for UNSAT, self-certifying models for SAT). Pattern-level tool (137 known concurrency idioms), not a full-program verifier.

## 30-Second Quickstart

```bash
cd litmus_inf
pip install -e .           # install from source (not on PyPI)
litmus-check --target arm myfile.c
```

Output:
```
UNSAFE: mp (data_race) → dmb ishst (T0); dmb ishld (T1)  [Z3: SAT witness]
SAFE:   mp_fence                                           [Z3: UNSAT + Alethe proof]
2 patterns matched, 1 unsafe, 1 safe  (8ms)
```

Every UNSAFE result includes a self-certifying Z3 counterexample. Every SAFE result includes an Alethe-format resolution proof.

## Key Results

| Metric | Value | Evidence |
|--------|-------|----------|
| Alethe proof certificates | **1,370/1,370 (100%)** | 1,095 UNSAT Alethe proofs + 275 self-certifying SAT models |
| Avg proof size | **109.3 steps** | Median 94, max 169; avg 12.8ms extraction |
| CPU cross-validation | **432/432** | Enumeration vs SMT agreement on all CPU pairs |
| GPU external validation | **94/94** | Published litmus tests + PTX/Vulkan specs + known bugs |
| Denotational semantics | **428/432 (99.1%)** | 4 disagreements: RISC-V asymmetric fence gaps |
| DSL-to-.cat correspondence | **170/171 (99.4%)** | 1 mismatch (mp\_fence\_wr on RISC-V) |
| Code recognition (238 snippets) | **86.1% exact, 93.3% top-3** | 203 curated + 35 independently sourced |
| Adversarial benchmark (23 snippets) | **13.0% exact, 21.7% top-3** | 7 OOD domains; quantifies sampling bias |
| Fence proofs | **99 UNSAT + 72 SAT** | Fence sufficiency or inherent observability |
| TCB analysis | **11 components** | Verified / Trusted-Axiom / Implementation-Trust |
| Lean proof sketches | **Theorems 1,4,5,6,7** | sorry'd subgoals delineate proof obligations |
| Speed | **median <1ms per pattern** | Full 1,370-pair analysis: mean 397ms |

## Limitations

- **Advisory pre-screening only:** matches code against 137 fixed patterns — cannot discover novel bugs or verify whole programs
- **Adversarial benchmark shows 13% accuracy:** on out-of-distribution code (93%→13% gap quantifies author-sampling bias)
- **Z3 in TCB:** Alethe proofs are independently checkable but require an external validator (Carcara)
- **GPU models:** 94/94 external cross-validation; not hardware-tested
- **Lean proof sketches have sorry'd subgoals:** 12 sorry'd subgoals across 5 theorems
- **TCB:** 6 of 11 components at Implementation-Trust level (not formally verified)
- **Denotational semantics:** 4 disagreements on RISC-V asymmetric fences (known expressiveness gap)
- **Compositionality:** four strategies from exact (disjoint, SC-shared) to sound (Owicki-Gries, conservative RG)
- **Fence costs are analytical weights:** not measured hardware latencies
- **Severity taxonomy:** CWE-calibrated, not CVE-validated
- **Not on PyPI** — install from source only

## Supported Architectures

- **CPU:** x86-TSO, SPARC-PSO, ARMv8, RISC-V RVWMO
- **GPU:** OpenCL (WG/Device), Vulkan (WG/Device), PTX/CUDA (CTA/GPU)
- **Custom:** define via DSL (`model MyModel { relaxes W->R, W->W ... }`)

## Dependencies

- Python 3.8+
- z3-solver (`pip install z3-solver`)
- tree-sitter, tree-sitter-c, tree-sitter-cpp (optional, for AST analysis)

## Documentation

- [API.md](API.md) — API reference (CLI and Python)
- [litmus_inf/paper.tex](litmus_inf/paper.tex) — full paper with methodology and proofs
