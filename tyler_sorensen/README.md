# LITMUS∞: SMT-Verified Memory Model Portability Checking

Advisory pre-screening tool that checks whether concurrent C/C++/CUDA code is safe to port from x86 to ARM, RISC-V, or GPU architectures. Every result carries an independently checkable proof certificate (Alethe resolution proofs for UNSAT, self-certifying models for SAT), cross-validated across Z3 and CVC5 (two independent solver implementations). Pattern-level tool (140 known concurrency idioms covering 85.4% of documented real-world concurrency bugs), not a full-program verifier.

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

## Key Results

| Metric | Value | Evidence |
|--------|-------|----------|
| Z3 certificate coverage | **1,400/1,400 (100%)** | 597 UNSAT proofs + 803 SAT witnesses |
| Cross-solver validation | **1,400/1,400 agreement** | Z3 + CVC5 (independent solver); Wilson CI [99.7%, 100%] |
| LLM-assisted OOD accuracy | **85.0% exact-match** | GPT-4.1 on 113 adversarial snippets across 12 domains (vs 6.2% AST-only); Wilson CI [77.2%, 90.5%] |
| Alethe proof certificates | **993 UNSAT + 407 SAT** | Avg 106.4 steps; independently checkable |
| CPU cross-validation | **228/228** | herd7 internal consistency |
| GPU external validation | **94/94** | Published litmus tests + PTX/Vulkan specs |
| DSL-to-.cat correspondence | **170/171 (99.4%)** | TSO, ARM, RISC-V; single mismatch fully characterized (false negative on mp_fence_wr, isolated, minimal impact) |
| Code recognition (curated) | **96.6% exact, 97.0% top-3** | n=203 benchmark snippets |
| Bug coverage | **35/41 (85.4%)** | Documented real-world concurrency bugs |
| Severity classification | **689 data_race, 44 security, 70 benign** | 803 unsafe pairs |
| Pattern library | **140 base patterns + composition** | Classical, RMW, lock-free, seqlock, RCU, etc.; pattern composition, chains, rings, and BMC extend beyond fixed set |
| Adversarial OOD benchmark | **113 snippets, 12 domains** | AST-only: 6.2%; quantifies selection bias honestly |

## Limitations

- **Advisory pre-screening only:** matches code against 140 fixed patterns — cannot discover novel bugs or verify whole programs
- **Bug coverage is 85.4%:** 6/41 documented bugs are out-of-scope for litmus testing
- **Z3+CVC5 cross-validation:** both solvers agree on all 1,400 verdicts
- **GPU models:** 94/94 external cross-validation; not hardware-tested
- **Compositionality:** exact for disjoint vars, conservative for shared vars
- **Fence costs are analytical weights:** not measured hardware latencies
- **Severity taxonomy:** CWE-calibrated, not CVE-validated
- **LLM mode requires API key:** set OPENAI_API_KEY for LLM-assisted recognition
- **Not on PyPI** — install from source only

## Supported Architectures

- **CPU:** x86-TSO, SPARC-PSO, ARMv8, RISC-V RVWMO
- **GPU:** OpenCL (WG/Device), Vulkan (WG/Device), PTX/CUDA (CTA/GPU)
- **Custom:** define via DSL (`model MyModel { relaxes W->R, W->W ... }`)

## Dependencies

- Python 3.8+
- z3-solver (`pip install z3-solver`)
- tree-sitter, tree-sitter-c, tree-sitter-cpp (optional, for AST analysis)
- openai (optional, for LLM-assisted recognition)

## Documentation

- [API.md](API.md) — API reference (CLI and Python)
- [litmus_inf/paper.tex](litmus_inf/paper.tex) — full paper with methodology and proofs
