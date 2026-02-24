# LITMUS∞: Cross-Architecture Memory Model Portability Checker

**Is your concurrent C/C++ code safe to port from x86 to ARM?**
LITMUS∞ answers in under 1ms per pattern, with per-thread minimal fence fixes.

## 30-Second Quickstart

```bash
cd litmus_inf
pip install -e .

# Check C/C++ code for x86→ARM portability:
echo '// Thread 0
data.store(42, std::memory_order_release);
flag.store(1, std::memory_order_release);
// Thread 1
r0 = flag.load(std::memory_order_acquire);
r1 = data.load(std::memory_order_acquire);' | litmus-check --target arm --stdin
# → ✗ UNSAFE  mp → arm  Fix: dmb ishst (T0); dmb ishld (T1)

# Scan a directory:
litmus-check --target arm src/
```

## Key Results

| Metric | Value | Notes |
|--------|-------|-------|
| Portability pairs | 750 (75 × 10 architectures) | CPU + GPU |
| Code accuracy | 92.7% exact, 100% top-3 (n=111) | Stratified by category |
| SMT internal consistency | 228/228 (CPU-only) | Same-author encodings |
| Fence proofs | 55 UNSAT + 40 SAT (Z3) | Machine-checked |
| herd7 agreement | 50/50 | Wilson CI [92.9%, 100%] |
| Meaningful diff tests | 498 pass | + 2,280 determinism checks |
| Real-code eval | 93.3% exact on 15 snippets | Linux, Folly, LLVM |

See [litmus_inf/README.md](litmus_inf/README.md) for full documentation and [litmus_inf/API.md](litmus_inf/API.md) for API reference.
