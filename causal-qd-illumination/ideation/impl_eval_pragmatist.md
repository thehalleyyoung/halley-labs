# Implementation Evaluation — Hard-Nosed Pragmatist

**Proposal:** proposal_00 (CausalQD — Quality-Diversity Illumination for Causal Discovery)
**Evaluator:** Pragmatist (laptop-CPU-only, 150K LoC/day buildability, zero human involvement)
**Date:** 2026-03-02
**Method:** Claude Code Agent Teams — Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer, with adversarial cross-critique and independent verification signoff.

---

## Verdict: ABANDON

**Composite Score: 4.5/10**

| Dimension | Score | Summary |
|-----------|:-----:|---------|
| Code Quality | 7/10 | Clean architecture, consistent style, good docstrings, proper error handling. Deductions for duplication and dead code. |
| Genuine Difficulty | 4/10 | Every component is textbook. MAP-Elites, BIC/BDeu/BGe, Order MCMC, Parallel Tempering — all standard. Headline certificate contribution is mathematically vacuous. |
| Value Delivered | 3/10 | No evidence CausalQD outperforms any existing tool. No benchmark comparisons. Certificates provide zero actionable guarantees. No clear user. |
| Test Coverage | 6/10 | 1,197 tests pass (100% rate), but assertions are systematically weak. No ground-truth recovery tests. Strongest baseline (Order MCMC) deliberately excluded from comparisons. |

---

## Evidence Summary

### Tests: Quantity Without Substance
- **1,197 tests pass, 0 failures** — impressive volume.
- **No ground-truth recovery test exists.** Integration test computes SHD but only asserts `shd >= 0` (non-negative). Never checks that MAP-Elites actually recovers known causal structure.
- **"Competitive" baseline test is rigged.** `test_integration.py:376` asserts `bic_me >= worst_baseline * 2`. Since BIC scores are negative, this lets MAP-Elites be **half as good** as the worst baseline and still pass.
- **Order MCMC is implemented but zero-tested.** `grep -rn "OrderMCMC\|order_mcmc" tests/` returns nothing. The strongest competitor is deliberately excluded.
- **Certificate tests check formatting, not meaning.** All verify `value in [0,1]` and serialization — none verify calibration on known-ground-truth problems.

### Certificates: Confirmed Vacuous
- **Spectral Lipschitz bound: 1,410 for a 5-node chain with N=500.** Actual BIC deltas are O(1). The bound is ~700x too loose, confirmed by independent empirical test.
- Formula (`lipschitz.py:194`): `L = N * λ_max / (λ_min² * p)` grows linearly with N — always vacuously large.
- **Path certificates anti-conservative.** `path_certificate.py:86-96` multiplies edge bootstrap frequencies assuming independence. Edges sharing a node have correlated bootstrap frequencies, making path certificates optimistically biased — the wrong direction for a "robustness guarantee."
- The code's own docstring acknowledges: *"This bound grows linearly with N and can be vacuously large."*

### Theory Fixes: Zero Addressed
Prior theory evaluation required 10 mandatory modifications. Implementation status:

| Required Fix | Status |
|---|---|
| Pivot framing to behavioral descriptors | ❌ Not done |
| Drop all theoretical claims | ❌ Still validates 9 "theorems" |
| Include Order MCMC as co-method | ⚠️ Implemented, never tested against MAP-Elites |
| Reduce scope to n≤20 | ❌ Code targets n≤50 |
| Define success criteria | ❌ No pre-specified criteria |
| Fix Theorem 7 factor-of-2 error | ❌ Unchanged |
| Fix path certificate independence assumption | ❌ Unchanged |

### LOC Analysis
- **59,486 total LOC** (55,818 source + 16,087 test across 140 files).
- **~52% is scaffolding** (docstrings 33%, blanks, comments, imports).
- **Actual algorithmic code: ~27K lines.**
- Buildable in ~10 hours with 150K LoC/day capacity — within budget.
- 24 packages across 140 files — significantly over-modularized for what is a MAP-Elites loop with BIC scoring and bootstrap.

### Laptop CPU Feasibility
- Runs on CPU — no GPU dependencies. Only numpy/scipy.
- **n≥50 nodes is computationally infeasible** within reasonable time.
- Parallel module runs islands sequentially per generation (no real parallelization).
- Bootstrap certificates require ~500K additional score evaluations — hours on laptop for n≥20.

---

## Disagreements and Resolution

### Auditor (6/10 test coverage) vs Synthesizer (9/10)
**Resolution: 6/10.** The Synthesizer counted tests and pass rate. The Auditor examined what tests actually assert. Evidence shows assertions are systematically weak — SHD ≥ 0 is tautological, "competitive" test exploits negative BIC, no ground-truth recovery. Quantity without meaningful assertions is not coverage.

### Skeptic (3.0 composite) vs Auditor (5.5 composite)
**Resolution: 4.5/10.** The Auditor measured engineering quality; the Skeptic measured delivered value. Both are correct on their axis. Well-engineered code implementing broken theory lands at 4.5.

### Synthesizer (engineering quality 8/10) vs Skeptic ("52% scaffolding")
**Resolution: 7/10.** 40-55% non-logic lines is normal for well-documented Python. But dual ABCs, dead code, and duplicated fast/slow implementations justify deducting from 8 to 7.

---

## Salvage Assessment

The Scavenging Synthesizer identified genuinely useful components:

| Component | LOC | Ecosystem Value | Extraction Effort |
|-----------|-----|-----------------|-------------------|
| **MEC toolkit** (enumeration, canonical hashing, CPDAG) | ~2,550 | Fills genuine Python gap — no equivalent exists | 3-5 days |
| **MI-profile behavioral descriptors** | ~400 core | Genuinely novel DAG fingerprinting | 2-3 days |
| **Acyclicity-preserving DAG operators** | ~500 | No equivalent in any library | 2-3 days |
| **BIC/BDeu/BGe scoring** (EM, regularization, JIT) | ~4,221 | Exceeds causal-learn's scoring | 3-5 days |

**Recommended extraction: ~2 weeks for MEC + descriptors + scoring as standalone `causal-tools` package (~3,500 LoC). Abandon the remaining ~56K lines.**

---

## Final Assessment

CausalQD is well-engineered code implementing a theoretically unsound idea. The implementation does not address any of the 10 mandatory modifications from the theory evaluation. The headline contribution — Lipschitz certificates for causal claims — is empirically confirmed vacuous (700-16,000x too loose). The strongest available baseline (Order MCMC) is implemented in the codebase but deliberately excluded from comparison tests. No benchmark demonstrates CausalQD outperforms any existing tool on any dataset.

The ~3,500 lines of genuinely novel, useful code (MEC toolkit + behavioral descriptors + scoring) are buried under ~56K lines of standard QD infrastructure. Extract the diamonds; abandon the mine.

**VERDICT: ABANDON**
