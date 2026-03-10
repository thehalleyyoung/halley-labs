# Skeptic Verification: Algebraic Repair Calculus (proposal_00)

**Date**: 2026-03-04
**Evaluation method**: Claude Code Agent Teams — 3-expert adversarial evaluation with cross-critique and independent verification signoff
**Roles**: Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer, Independent Verifier

---

## Executive Summary

A three-sorted delta algebra (Δ_S, Δ_D, Δ_Q) with interaction homomorphisms for provably correct pipeline repair. The algebra connects schema evolution, data changes, and quality drift under a unified algebraic framework, with delta annihilation as the key practical optimization.

**VERDICT: CONDITIONAL CONTINUE** — with mandatory scope reduction from three-sorted to two-sorted primary, and week-1/week-2 kill criteria to front-load existential risks.

---

## Pillar Scores (Verified by Independent Signoff)

| Pillar | Score | Justification |
|--------|-------|---------------|
| **1. Extreme Value** | **6/10** | Real problem (40-60% engineer time on maintenance), but marginal value over advancing baselines (dbt + LLMs) is uncertain. Delta annihilation is a genuine unique capability. Cost savings (2-5×) are predicted, not demonstrated. Compound perturbation rarity is unverified. |
| **2. Genuine Software Difficulty** | **6/10** | ~12-15K genuinely hard novel LoC (from claimed 22.7K). 48 coherence conditions are a mix of hard (5-8 cases involving JOIN/GROUP BY/WINDOW) and tedious (30+ straightforward cases). SQL semantic analysis is the hardest practical subproblem. Not "years of research" hard but substantially beyond glue code. |
| **3. Best-Paper Potential** | **4/10** | "Known math, new domain" — interaction homomorphisms are indexed monoid actions from universal algebra. The DBSP impossibility has 55-60% probability of being trivially true. The commutation theorem is structural induction with a new hypothesis. Honest best-paper probability: 5-8% unconditional. Target: solid VLDB accept. |
| **4. Laptop-CPU Feasibility** | **8/10** | All components CPU-friendly: sqlglot, DuckDB, scipy, networkx. TPC-DS SF=10 fits 16GB RAM. DP planner handles 500-node pipelines in milliseconds. Evaluation runtime 12-24 hours (not claimed 6-8), but overnight batch is fine. No GPUs, no human annotation, no human studies. |
| **5. Feasibility** | **5/10** | 30% FATAL risk on coherence failure. 55% P(publishable at top venue), 75% P(publishable somewhere). Two-sorted escape hatch brings P(publishable) to ~88%. Single-developer timeline: 7-13 months. The proposal's own proponents give 55% publishable probability. |
| **Composite** | **29/50** | |

---

## Fatal Flaws (3 confirmed, 2 downgraded)

### CRITICAL — Confirmed Fatal Flaws

**1. Coherence failure for core operators (30% probability — self-assessed FATAL)**
The hexagonal coherence condition `push_f(φ(δ_s)(δ_d)) = φ(push_f^S(δ_s))(push_f^D(δ_d))` must hold for each SQL operator. The proposal's own difficulty assessor rates 30% probability that interaction homomorphisms don't compose cleanly through real SQL operators — and rates this FATAL. If coherence fails for JOIN (the most critical operator for real pipelines), the commutation theorem is invalid for most practical pipeline topologies.

**Mitigation**: Two-sorted (Δ_S, Δ_D) fallback. Verify SELECT, FILTER, JOIN by week 2 or pivot.

**2. DBSP Encoding Impossibility may be trivial (55-60% probability)**
Three of four encoding failure cases (universal tuple, deletion-reinsertion, meta-table) are trivially expected from definitions. The tagged-union case is the only potentially deep argument. The proposal itself gives 30-35% probability of triviality; the team consensus is 55-60%. A "theorem" whose own authors assign majority probability to triviality is speculation, not a contribution. Additionally, DBSP's own team (Feldera) is actively working on schema evolution support, which could invalidate the impossibility result.

**Mitigation**: Resolve in week 1. Tagged-union asymptotic argument is the litmus test. Deep → keep. Trivial → demote to design motivation paragraph.

**3. Fragment F coverage possibly <50% of real SQL**
Fragment F (where commutation theorem provides exact guarantees) excludes: ORDER BY, floating-point aggregations, window functions with ordering, non-deterministic functions, NULLs under three-valued logic, LIMIT with ties. These features appear in virtually all analytical SQL. The proposal's own risk assessment gives 60% probability of lower-than-expected F coverage. If F covers <40% of real pipeline stages, the "provably correct" headline is marketing, not mathematics.

**Mitigation**: Measure honestly on TPC-DS. If <50%, reframe paper around delta annihilation and bounded deviation guarantees rather than provable correctness.

### DOWNGRADED — Previously Claimed Fatal Flaws

**4. Compound perturbation rarity** — Downgraded. Two-sorted scope removes dependency on three-sorted compound handling. Quality deltas are future work.

**5. Best-paper probability inflation (25-35% claimed)** — Downgraded. The 25-35% figure from theory/approach.json contradicts every prior assessment (7-10% from approach_debate.md). Retracted to 5-8%. Not fatal, just dishonest marketing.

---

## Claim-by-Claim Assessment

| Claim | Verdict | Evidence |
|-------|---------|----------|
| "Interaction homomorphisms are minimal structure" | **UNSUPPORTED** | No minimality proof exists. No lower bound argument. Would φ and ψ as plain functions (not homomorphisms) work? The homomorphism property may be operationally unnecessary for DAG-processed pipelines. |
| "Bounded Commutation: repair = recompute for F" | **WEAKLY SUPPORTED** | Theorem technique is standard (structural induction). Fragment F likely covers <50% of real analytical SQL. The ε bound for pipelines outside F is hand-waved ("backward error analysis" in one sentence). |
| "DBSP Encoding Impossibility" | **WEAKLY SUPPORTED** | 55-60% probability of being trivially true. 3 of 4 encoding failures are definitional. Tagged-union case may provide genuine asymptotic separation if formalized. |
| "2-5× cost reduction over dbt" | **UNSUPPORTED** | Zero empirical evidence. Prediction only. Depends on annihilation rates which are untested. Modern dbt with column-level lineage may already achieve similar pruning. |
| "48 coherence conditions verified" | **WEAKLY SUPPORTED** | Only 15/48 get formal proofs (5 operators × 3 sorts). Remaining 33 are property-tested. 30% probability the hardest cases (JOIN, GROUP BY, WINDOW) fail. |
| "~50.5K LoC, ~22.7K novel" | **WEAKLY SUPPORTED** | Inflated ~33%. Realistic: ~33.5K total, ~12-15K genuinely novel. Still substantial but not extraordinary for a research prototype. |
| "No existing tool handles compound perturbations" | **WEAKLY SUPPORTED** | Formally true — no single tool composes schema + data + quality repairs algebraically. Practically misleading — sequential application of Liquibase + dbt + Great Expectations handles compound perturbations in practice. The proposal never demonstrates a concrete case where sequential handling fails. |
| "Delta annihilation prunes zero-effect stages" | **SUPPORTED** | Conceptually sound, clearly differentiated from lineage-aware recomputation (which cannot detect zero-effect through operators). The strongest claim in the proposal. |

---

## Team Disagreements and Resolutions

### 1. Continue vs. Abandon
- **Auditor**: Neutral (5/10 feasibility)
- **Skeptic**: "Cannot recommend acceptance in current form" → Conditional continue with mandatory restructuring
- **Synthesizer**: Continue with scope reduction
- **Resolution**: CONDITIONAL CONTINUE. The Skeptic's kill shot couldn't land — delta annihilation is genuinely useful and the two-sorted fallback ensures publishability. But the Skeptic's restructuring demands are correct.

### 2. Is two-sorted the right approach or consolation?
- **Auditor**: "Significantly weakens the contribution"
- **Skeptic**: Mandatory restructuring
- **Synthesizer**: "The RIGHT approach, not consolation"
- **Resolution**: **Synthesizer wins.** Two-sorted covers 90%+ of real perturbations (schema + data). Quality drift is the least common perturbation type. Coherence risk concentrates in the quality sort. Design for two-sorted from day 1.

### 3. Is delta annihilation actually novel vs. column-level lineage?
- **Auditor**: "Concrete differentiator" over dbt
- **Skeptic**: "SQLMesh already does annihilation-equivalent analysis"
- **Synthesizer**: "Crown jewel for salvage" (8/10 standalone)
- **Resolution**: **Annihilation is the formalized version of column-level lineage pruning.** The practical mechanism is close to existing tools, but the formalization enables the commutation theorem (correctness guarantee). The algebra's value is in the proof, not the optimization. For a *paper*, this is fine. For *adoption*, the marginal value over SQLMesh needs quantification.

---

## Mandatory Conditions for CONTINUE

### Week-1 Kill Criteria (violation → ABANDON)
1. DBSP impossibility is trivially true AND coherence fails for SELECT
2. SQL lineage recall < 80% on basic test cases
3. The team cannot articulate why φ as a homomorphism (not just a function) matters for a concrete example

### Binding Conditions
1. **Two-sorted (Δ_S, Δ_D) is the primary paper scope.** Three-sorted is stretch only.
2. **DBSP impossibility resolved by end of week 1.** Deep → theorem. Trivial → remark.
3. **Coherence verified for SELECT, FILTER, JOIN by end of week 2.** Any failure → pivot to annihilation-only engineering paper.
4. **Annihilation is the headline contribution.** Paper title: "Delta Annihilation: Schema-Aware Pruning for Incremental Pipeline Repair."
5. **Fragment F coverage reported honestly.** If <50% on TPC-DS, reframe around cost savings, not "provably correct."
6. **Annihilation rate ≥30% on TPC-DS corpus** as measurable gate.
7. **Build the "80% baseline" first.** Column-level lineage + heuristic rules. Measure marginal algebra benefit. If <20% marginal benefit, reconsider.
8. **Check Feldera/DBSP roadmap.** Address in related work.
9. **Add Clio/HECATAEUS to required related-work comparisons.**

---

## Minimum Viable Scope

**Target**: ~20-25K LoC, two-sorted algebra, annihilation-focused

| Component | LoC | Purpose |
|-----------|-----|---------|
| SQL semantic lineage analyzer | 4,000 | Column-level dependencies via sqlglot |
| Two-sorted delta algebra (Δ_S, Δ_D) | 5,000-6,000 | 5 operators: SELECT, FILTER, JOIN, GROUP BY, UNION |
| Delta annihilation detector | 1,500 | Static analysis: push_f(δ) = 0? |
| Greedy repair planner | 2,500 | Topological-order with annihilation pruning |
| DuckDB evaluation harness | 3,000 | TPC-DS SF=10, perturbation injection |
| Property-based test suite | 2,000 | Hypothesis tests for algebraic laws |
| Baselines | 2,000 | dbt-equivalent, full recompute, naive incremental |

### Explicitly Cut
- Δ_Q (quality deltas), ψ homomorphism → future work
- Python idiom matcher → future work
- Refinement type system → minimal internal use only
- LP/ILP for cyclic topologies → greedy fallback sufficient
- Saga-based executor → DuckDB checkpoint/rollback only
- Multiple SQL dialects → PostgreSQL only

### Stretch Goals (only if ahead)
1. Three-sorted extension (if coherence holds by 40% mark)
2. DP-optimal planner (if non-monotone substructure proof completes)
3. DBSP impossibility as standalone theorem (if deep version proves out)

---

## Probability Assessment

| Outcome | Probability |
|---------|-------------|
| Publishable at VLDB/SIGMOD (full paper) | 55% |
| Publishable at CIDR/workshop/industrial track | 20% |
| Useful artifact, no top-venue paper | 15% |
| Complete failure | 10% |
| Best paper at VLDB/SIGMOD | 5-8% |

---

## Verification Signoff

**APPROVED WITH AMENDMENTS** by Independent Verifier.

Amendments applied:
1. Extreme Value adjusted 5→6 (pain is real, unique capabilities undersold)
2. Best-Paper adjusted 3→4 (engineering story at systems venue slightly better)
3. Three follow-up items: measurable gates for conditions 4-5, Clio/HECATAEUS in related work, SQL analyzer correctness as monitored risk

**Evaluation quality**: HIGH — three-assessor adversarial structure with cross-critique produced genuine intellectual pressure-testing. The proposal's own self-critical materials were extensively cited. No rubber-stamping detected.

---

*Signed: Team Lead, Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer, Independent Verifier*
*Evaluation complete. Team disbanded.*
