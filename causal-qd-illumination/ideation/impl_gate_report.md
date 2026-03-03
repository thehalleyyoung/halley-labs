# Implementation Gate Report — CausalQD (proposal_00)

**Date:** 2026-03-02
**Stage:** Verification (Implementation Gate)
**Method:** Claude Code Agent Teams — 3 expert agents (Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer) with adversarial cross-critique, synthesis, and independent verification signoff.
**Prior evaluations:** 4 persona evaluations (Engineer, Pragmatist, Skeptic, Community Expert)

---

## Evaluation Axes

| Axis | Score | Description |
|------|:-----:|-------------|
| **1. Extreme and Obvious Value** | **3/10** | Does not solve a desperately-needed problem. No demonstrated advantage over existing tools. |
| **2. Genuine Difficulty as Software Artifact** | **5/10** | Competent engineering of known algorithms. Non-trivial but not impressively hard. |
| **3. Best-Paper Potential** | **2/10** | No result that would surprise a reviewer. Missing experiments, vacuous theory. |
| **Total** | **10/30** | |

---

## Team Consensus

| Evaluator | Axis 1 | Axis 2 | Axis 3 | Total | Verdict |
|-----------|:------:|:------:|:------:|:-----:|---------|
| Independent Auditor | 3 | 4 | 2 | 9/30 | ABANDON |
| Fail-Fast Skeptic | 3 | 4 | 2 | 9/30 | ABANDON |
| Scavenging Synthesizer | 4 | 7 | 3 | 14/30 | ABANDON (salvage) |
| **Synthesized (Lead)** | **3** | **5** | **2** | **10/30** | **ABANDON** |

Prior persona evaluations: 3/4 ABANDON, 1/4 CONTINUE (conditions unmet).
**Overall: 6 of 7 independent evaluations recommend ABANDON.**

---

## Key Findings (Verified)

### 1. No Demonstrated Value (Axis 1: 3/10)

- **MAP-Elites vs GES: 1000x slower, same quality.** On an 8-node DAG with N=500, MAP-Elites achieves SHD=4 in 1.18s; GES achieves SHD=4 in <0.001s. At 200 iterations (2.47s), MAP-Elites gets SHD=3, still worse than PC (SHD=2).
- **Zero benchmark results.** No experiments on ALARM, ASIA, Sachs, or any standard dataset. The `.benchmarks/` directory is empty.
- **Baseline test is vacuous.** `test_integration.py:376` asserts `bic_me >= worst_baseline * 2`. Since BIC scores are negative (e.g., −7000), `worst_baseline * 2 ≈ −14000`, which ANY non-degenerate DAG trivially satisfies.
- **No demonstrated user or use case** for QD-illuminated DAG archives.

### 2. Certificates Are Disconnected and Vacuous (Critical)

- **Never integrated.** `from causal_qd.certificates` appears zero times outside the certificates package itself. The flagship feature is literally not wired into the engine.
- **Self-documented as vacuous.** `lipschitz.py:166`: *"This bound grows linearly with N and can be vacuously large."* Empirically confirmed: spectral bound is 31x too loose at N=1000.
- **Fisher information bound** (tighter) is implemented but never used by the engine.

### 3. Engineering Is Competent, Not Novel (Axis 2: 5/10)

- **1,197 tests, all passing** (63.6s). Test infrastructure includes Hypothesis property-based testing — genuinely good practice.
- **BGe scorer** (715 LOC): Correct Normal-Wishart implementation with slogdet stability. Non-trivial but follows Geiger & Heckerman (2002).
- **MEC enumerator** (678 LOC): Full Meek R1-R3 with chain decomposition. Follows Chickering (2002).
- **MAP-Elites core** is ~300 lines of textbook QD algorithm. Curiosity selection stubs to uniform sampling.
- **~220 genuinely novel lines** (MI-profile descriptors + EM-for-missing-data BIC) out of 59,486 total (0.37%).

### 4. No Path to Best Paper (Axis 3: 2/10)

- **Theory score was 3.0/10.** All 9 theorems trivial, circular, or erroneous (prior evaluation).
- **No experiments = no paper** at any venue above workshop level.
- **"Theorem" tests** verify correctness invariants (mutation preserves acyclicity) dressed as theoretical contributions.
- **Competitive landscape:** GES, PC, NOTEARS, GOBNILP, Order MCMC all have established implementations with published benchmark results.

---

## Disagreement Resolution

### Difficulty: 4 (Auditor/Skeptic) vs 7 (Synthesizer) → **5**
The Synthesizer credited volume (55K LOC, 23 modules, 1197 tests) and mathematical care (BGe, MEC). The Auditor/Skeptic correctly noted every algorithm follows published pseudocode. Resolution: implementing BGe and MEC from scratch requires genuine mathematical understanding (not a weekend project), but it is engineering difficulty, not algorithmic novelty. Score: 5.

### Salvage value does not change the verdict
All experts agree on salvageable components (MEC toolkit ~1,350 LOC, scoring suite ~4,265 LOC, MI-profile descriptors ~400 LOC). These could form a standalone `causal-tools` package. But salvage ≠ continue — the proposal's value proposition (QD illumination + robustness certificates) fails.

### Prior CONTINUE conditions are insufficient
The one CONTINUE evaluator required: ablation study, scalability testing, fix mislabels, add MTC. Even if met, these address presentation, not the fundamental problem: no evidence QD illumination provides value over established methods.

---

## Salvageable Components

| Component | LOC | Ecosystem Gap | Extraction |
|-----------|-----|---------------|------------|
| MEC toolkit (enumeration, hashing, CPDAG) | ~1,350 | No Python equivalent | Easy |
| Scoring suite (BIC/BDeu/BGe + incremental) | ~4,265 | Exceeds causal-learn | Easy |
| MI-profile behavioral descriptors | ~400 | Novel DAG fingerprint | Moderate |
| Acyclicity-preserving DAG operators | ~500 | Unique in Python | Easy |
| KSG CMI estimator | ~580 | Clean standalone impl | Easy |

**Total salvageable: ~7,100 LOC out of 59,486 (12%).**

---

## Verdict

### ABANDON

**Rationale:** Six of seven independent evaluations converge on ABANDON. The implementation demonstrates competent engineering (7/10 code quality, 1,197 passing tests) wasted on a theoretically unsound value proposition. The headline contribution — Lipschitz certificates for causal claims — is self-documented as vacuous and never connected to the engine. MAP-Elites achieves the same quality as GES at 1000x the cost, with no benchmark demonstrating otherwise. The ~220 lines of genuinely novel code (0.37% of total) do not constitute a publishable contribution. Extract the MEC toolkit and scoring functions as standalone utilities; abandon the QD engine.

```json
{
  "rankings": [
    {
      "proposal_id": "proposal_00",
      "score": 10,
      "verdict": "ABANDON",
      "reason": "6/7 evaluators ABANDON. Certificates vacuous and disconnected from engine. MAP-Elites 1000x slower than GES with no quality gain. Zero benchmark results. 220 novel lines out of 59K. Theory score 3.0/10 unaddressed. Salvage MEC toolkit and scoring suite (~7K LOC); abandon remaining 52K."
    }
  ],
  "best_proposal": "proposal_00"
}
```
