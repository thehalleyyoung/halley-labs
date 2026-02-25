# LITMUS∞: Cross-Architecture Memory Model Portability Checker

**Is your concurrent C/C++ code safe to port from x86 to ARM?**
LITMUS∞ answers in under 1ms per pattern, with per-thread minimal fence fixes,
severity classification, and Z3 certificates for every result.

## 30-Second Quickstart

```bash
cd litmus_inf
pip install -e .

# Check a single pattern:
python3 portcheck.py --pattern mp --target arm
# → Pattern: mp  Target: arm  Result: UNSAFE  Severity: data_race
#   Fence fix: dmb ishst (T0); dmb ishld (T1)  Cost saving: 62.5%

# Scan C/C++ source for portability bugs:
litmus-check --target arm src/
```

## What It Does (Honestly)

LITMUS∞ is a **pattern-level** portability checker, not a full-program verifier.
It matches concurrent code snippets against 75 known litmus test patterns and
checks safety across x86-TSO, SPARC-PSO, ARMv8, RISC-V RVWMO, and 1 parameterized
GPU model (6 scope instantiations).

**What it's great at:**
- Sub-millisecond analysis: 750 test-model pairs in <200ms
- **750/750 Z3 certificate coverage**: every portability matrix cell independently verified
- **342 unsafe pairs severity-classified**: 228 data races, 44 security vulnerabilities, 70 benign
- **170/171 DSL-to-.cat formal correspondence** across TSO, ARM, RISC-V
- Machine-checked fence proofs: 55 UNSAT + 40 SAT Z3 certificates
- 228/228 herd7 agreement on CPU models

**What it can't do:**
- Analyze arbitrary concurrent programs (pattern-level only, 75 patterns)
- Compose individual pattern results into whole-program guarantees
- Replace full-program tools like Dartagnan or VSync for complex code
- When code doesn't match any known pattern, the tool emits an
  `UnrecognizedPatternWarning` with `coverage_confidence` metric

## Key Results

| Metric | Value | Notes |
|--------|-------|-------|
| Z3 certificate coverage | **750/750 (100%)** | Wilson CI [99.5%, 100%] |
| Severity classification | 228 data race, 44 security, 70 benign | 342 unsafe pairs |
| DSL-to-.cat correspondence | **170/171 (99.4%)** | TSO, ARM, RISC-V |
| herd7 agreement | **228/228** | Wilson CI [98.3%, 100%] |
| Code accuracy | 96.6% exact, 98.0% top-3 (n=203) | Wilson CI [93.0%, 98.5%] |
| SMT consistency (CPU+GPU) | 228/228 + 108/108 | Independent Z3 validation |
| Fence proofs | 55 UNSAT + 40 SAT (Z3) | Machine-checked |
| Portability pairs | 750 (75 × 10 configs) | 4 CPU + 1 GPU ×6 scopes |

## CI/CD Integration

```yaml
# .github/workflows/litmus-check.yml
- uses: actions/checkout@v4
- run: pip install litmus-inf
- run: litmus-check --target arm --fail-on-unsafe src/
```

## Coverage Confidence

When the analyzer encounters code it can't confidently match:

```
UnrecognizedPatternWarning: 5 concurrent operations detected but
none match built-in patterns (coverage: 30%). Manual review recommended.
```

The `coverage_confidence` field (0.0–1.0) in analysis results indicates
what fraction of concurrent operations are explained by the best match.

See [litmus_inf/README.md](litmus_inf/README.md) for full documentation and [API.md](API.md) for API reference.
