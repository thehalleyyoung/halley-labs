# LITMUS∞: SMT-Verified Memory Model Portability Pre-Screening

Fast advisory tool that checks whether concurrent C/C++/CUDA code is safe to port from x86 to ARM, RISC-V, or GPU architectures. Every result carries a Z3 SMT certificate. This is a pattern-level pre-screening tool (75 known concurrency idioms), not a full-program verifier.

## 30-Second Quickstart

```bash
cd litmus_inf
pip install -e .           # install from source (not on PyPI)
litmus-check --target arm myfile.c
```

Output — instant portability check with Z3 certificate:
```
UNSAFE: mp (data_race) → dmb ishst (T0); dmb ishld (T1)  [Z3: SAT witness]
UNSAFE: sb (data_race) → dmb ish (T0); dmb ish (T1)      [Z3: SAT witness]
SAFE:   mp_fence                                           [Z3: UNSAT proof]
3 patterns matched, 2 unsafe, 1 safe  (12ms)
```

Every UNSAFE result includes a concrete Z3 counterexample execution. Every SAFE result includes a Z3 unsatisfiability proof with minimal unsat core.

## Key Results

| Metric | Value | Evidence |
|--------|-------|----------|
| Z3 certificate coverage | **750/750 (100%)** | 459 UNSAT proofs + 291 SAT witnesses, zero timeouts |
| herd7 agreement (CPU) | **228/228** | Wilson CI [98.3%, 100%]; internal consistency against .cat specs |
| GPU SMT consistency | **108/108** | Internal only — no external oracle or GPU-litmus comparison |
| Machine-checked fence proofs | **55 UNSAT + 40 SAT** | Proves fence sufficiency or inherent observability |
| DSL-to-.cat correspondence | **170/171 (99.4%)** | 1 documented mismatch (mp\_fence\_wr on RISC-V) |
| Code recognition accuracy | **93.0% exact, 94.0% top-3** | n=501 author-sampled snippets from 10 projects; Wilson CI [90.4%, 94.9%] |
| Severity triage | 228 data\_race, 44 security, 70 benign | 342 unsafe pairs, CWE-calibrated (not CVE-validated) |
| Speed | **median 189ms, mean 217ms** | All 750 pairs; <1ms per individual pattern |

## Limitations (Read These)

- **Advisory pre-screening only:** matches code against 75 fixed litmus-test patterns — cannot analyze arbitrary programs, discover novel bugs, or verify whole-program correctness
- **Benchmark is author-sampled:** 501 snippets from 10 open-source projects selected by tool authors; not independently curated; may overrepresent patterns the tool handles well
- **No proof certificate validation:** Z3 remains in trusted computing base; SMT-LIB2 exports enable solver replay but no LFSC/Alethe/DRAT independent proof checking
- **GPU models lack external validation:** 108/108 is internal consistency only (same-author encodings); no comparison against GPU-litmus or PTX operational semantics
- **Theorems are paper proofs:** not mechanized in Coq/Isabelle/Lean; 750/750 Z3 certificates provide machine-checked evidence for the portability matrix but not the metatheory
- **Compositionality restricted to disjoint variables:** shared-variable programs get conservative analysis; rely-guarantee extension is future work
- **Fence costs are analytical weights:** not measured hardware latencies
- **Severity taxonomy not CVE-validated:** CWE-calibrated heuristic triage, not validated risk assessment
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
