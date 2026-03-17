# IsoSpec Panel Adjudication: Adversarial Cross-Critique & Consensus

**Panel:** Independent Auditor (IA) · Fail-Fast Skeptic (FFS) · Scavenging Synthesizer (SS)
**Adjudicator:** Team Lead
**Date:** 2026-03-08
**Subject:** proposal_00 (txn-isolation-verifier)

---

## Verified Ground Truth (Pre-Adjudication)

Before any expert speaks, the adjudicator establishes undisputed facts:

| Fact | Evidence | Status |
|------|----------|--------|
| theory_bytes = 0 | `State.json`: `"theory_bytes": 0, "impl_loc": 0` | **CONFIRMED** |
| implementation/ empty | `du -sh implementation/` → 0B | **CONFIRMED** |
| Zero executable artifacts | No .rs, .py, .z3, .smt2, .lean, .v files in repo | **CONFIRMED** |
| Total files: 33 | 22 .md + 7 .log + 3 .json + 1 .txt | **CONFIRMED** |
| Total prose: ~456KB | theory/ 176K, ideation/ 212K, proposals/ 68K | **CONFIRMED** |
| Phase: theory_complete | State.json `"status": "theory_complete"` | **CONFIRMED** |
| 4 FATAL flaws identified | verification_report.md, CRITIQUE_ANALYSIS.md | **CONFIRMED** |
| 0/17 amendments completed | CRITIQUE_EVIDENCE_CHECKLIST.md | **CONFIRMED** |
| CLOTHO acknowledged | prior_art_analysis.md exists | **CONFIRMED** |
| Elapsed: ~3.5 hours | created 05:11 → updated 08:29 | **CONFIRMED** |

---

## PHASE 1: ADVERSARIAL CROSS-CRITIQUE

---

### Round 1: Skeptic Challenges Synthesizer

#### Attack 1: The Feasibility Gap (F=6 vs F=2)

**SKEPTIC:** You scored Feasibility 6/10. I scored 2/10. This is not a philosophical disagreement — it's an empirical one. Let me state the evidence:

- **33 files, zero executable.** Not a single Z3 constraint, not one Rust function, not one SQL test case. The team produced 456KB of markdown during a phase labeled "theory." Theory in formal methods means *formalization* — definitions, lemmas, proofs, encodings. What was produced is *planning about formalization*.
- **Your F=6 implies a >50% chance of successful execution.** But the base rate for projects that produce zero artifacts during their theory phase and then attempt 55-60K LoC of novel formal methods code is not >50%. I'd estimate it below 15% based on comparable academic verification projects (e.g., seL4 took 20 person-years for ~8,700 lines of proof).
- **The "planning IS the hard part" claim is backwards.** In formal verification, planning is 10% of the work. The hard part is when your Z3 encoding returns `unknown` on a 200-constraint PostgreSQL SSI model at 3am, and you need to restructure your entire abstraction. That moment hasn't happened yet. You cannot score the feasibility of a marathon based on the runner buying good shoes.

**Specific challenge:** Name one comparable project that scored F≥6 with zero executable artifacts after completing its theory phase.

#### Attack 2: "All 4 FATALs Fixable in 3-4 Weeks"

**SKEPTIC:** You claim all 4 FATAL flaws are fixable in 3-4 weeks. Let me decompose:

- **FF1 (NULL/3VL):** Sound three-valued logic encoding into SMT-LIB is a research problem, not an engineering task. Gurevich & Matos (1996) spent an entire paper on just the semantics. The "NOT NULL restriction" escape hatch reduces scope to the point where most real-world migration queries are excluded — production schemas have NULLable columns everywhere. **Estimated fix: 2-4 weeks IF restricting scope; 2-3 months for sound 3VL.**
- **FF2 (PG SSI memory):** PostgreSQL's SIREAD lock summarization under memory pressure changes conflict detection semantics. Modeling this requires reading ~4,000 lines of `predicate.c` and `predicate_internals.h` in the PG source. Nobody has formalized this. **Estimated fix: 3-6 weeks minimum, with unknown unknowns.**
- **FF3 (k=3 proof):** The proof errors suggest the mathematical framework itself may need restructuring, not just patching. G1a requiring k=2 (not k=3) changes the state space bounds. **Estimated fix: 1-2 weeks IF errors are surface-level; 4-8 weeks if structural.**
- **FF4 (SMT performance):** This is an experimental question, not a fix. But if benchmarking reveals the constraint explosion is real (≥1000 constraints for k=3, n=10), then the entire tool becomes impractical and requires architectural redesign. **Estimated fix: 1 week to benchmark; 4-12 weeks if redesign needed.**

**Sum:** Best case 7-10 weeks, worst case 5+ months. Your "3-4 weeks" assumes everything goes right, which is the opposite of what happens in formal verification.

#### Attack 3: P(pub) 65-75% — Where's the Evidence?

**SKEPTIC:** Your P(pub) of 65-75% for Path B implies that a team which has produced zero code has a better-than-coin-flip chance of publishing at SIGMOD/VLDB. The base rate for submitted papers at these venues is ~20% acceptance. You need to show:

1. **The paper will be submitted** — given 0 artifacts, submission requires completing all implementation, evaluation, and writing. P(submission) ≤ 70% based on the artifact gap.
2. **The paper will be accepted** — conditional on submission, P(accept|submit) ≈ 20-25% for SIGMOD, higher for workshops.
3. **Compound:** P(pub) = P(submit) × P(accept|submit) ≈ 0.70 × 0.25 = 17.5% for tier-1, ~35-45% including workshops.

Your 65-75% is achievable only if "any publication" includes arXiv preprints and tech reports, which is not what the project targets.

---

#### SYNTHESIZER RESPONSE TO ROUND 1:

**On Feasibility (F=6):** The Skeptic compares this to seL4, which is a clean-room verified microkernel — a categorically different undertaking. IsoSpec is a *verification tool* that *uses* Z3, not a project that *extends* Z3. The comparable projects are CLOTHO itself (published), Hermitage (a bash script + PostgreSQL), and Elle (Clojure + Jepsen). All were smaller teams and produced publishable results. My F=6 reflects that the *hard intellectual work* — identifying which engine behaviors to model, how to encode them, what the predicate theory looks like — is substantially complete in the 456KB of planning. The remaining work is *translation to code*, which is mechanical given good specifications.

However, I concede the Skeptic has a point on the zero-artifact gap. I would revise F to **5/10** — still above the Skeptic's 2, but acknowledging that zero executable output after theory phase is genuinely concerning. The difference between my revised 5 and the Skeptic's 2 is whether you believe *this specific team's* planning quality predicts execution quality. I do, based on the depth of the theory documents. The Skeptic doesn't, based on the absence of proof-of-concept code.

**On FATAL Fix Timeline:** I concede the 3-4 week estimate was optimistic. Revised: **5-8 weeks** for all four, with the understanding that FF1 and FF4 are the high-variance items. I maintain that FF2 and FF3 are 1-3 weeks each.

**On P(pub):** I was including workshops (CIDR, PODS workshop, VLDB demo track) in the 65-75%. For tier-1 full papers only, I'd estimate 35-45%. The Skeptic's 17.5% assumes independence between submission and acceptance probabilities, but a team that completes implementation is *already selected* for quality.

---

### Round 2: Synthesizer Challenges Skeptic

#### Rebuttal 1: V=3 Denies the Problem Exists

**SYNTHESIZER:** You scored Value at 3/10. This is extraordinary. It implies the problem — that database engines implement isolation differently, causing silent data corruption during migrations — is barely worth investigating. But:

- **Oracle→PostgreSQL migrations are a $10B+ annual market.** This isn't speculative — Gartner and Forrester have published the numbers. Every Fortune 500 company on Oracle is evaluating PostgreSQL.
- **Hermitage empirically demonstrates the divergences exist.** It's not a theoretical concern.
- **No tool currently provides formal guarantees** about isolation behavior during migration. CLOTHO provides abstract specs. Jepsen tests partitions. Hermitage is manual. LLMs enumerate known issues but cannot guarantee completeness.

A V=3 says "this problem is barely worth a workshop paper." But three independent evaluation panels scored V at 6-7/10. Your outlier V=3 needs stronger justification than "LLMs will solve it by 2027."

**Specific challenge:** If the problem isn't valuable, why did CLOTHO get published? Why does Hermitage have 2,000+ GitHub stars?

#### Rebuttal 2: Your Compound Probability Assumes Independence

**SYNTHESIZER:** You calculate P(all 4 FATAL fixed) = 23.5% by multiplying individual fix probabilities. This assumes the four FATALs are independent events. They are not:

- **FF1 (NULL) and FF4 (SMT performance) are anti-correlated.** Restricting to NOT NULL *reduces* the constraint set, *improving* SMT performance. Solving FF1 by restriction partially solves FF4.
- **FF2 (PG SSI) and FF3 (k=3 proof) share a common factor.** Both require deeper understanding of the state space. Insights gained modeling PG SSI directly inform the proof bounds.
- **All four share a team competence factor.** If the team can solve FF1 (hardest), they can almost certainly solve FF3 (easiest). Your independent multiplication treats "team can solve hard formal methods problem" as independent across problems, but it's actually a single latent variable.

With reasonable positive correlation (ρ ≈ 0.4), P(all 4 fixed) ≈ 35-45%, not 23.5%. The compound probability framing is mathematically misleading.

#### Rebuttal 3: Your 14-Day Reversal Condition Is a Conditional Continue

**SYNTHESIZER:** You recommend ABANDON but then specify a 14-day reversal condition: "working Z3 encoding + fixed proofs + NULL-aware encoding." This is structurally identical to a CONDITIONAL CONTINUE with a tight gate. The only difference is framing:

- **Your version:** Default ABANDON, prove me wrong in 14 days.
- **My version:** Default CONTINUE, but must pass a gate in 14 days.

These have the same expected outcome. The Skeptic is, in practice, a conditional continue voter who has set a high bar. The honest statement is: **"I believe the project should continue if and only if the team demonstrates execution capability within 14 days."** That's a gate, not an abandonment.

#### Rebuttal 4: Academic Papers Don't Need Paying Customers

**SYNTHESIZER:** "Zero companies currently pay for formal isolation verification" is the wrong market test for a SIGMOD paper. Academic papers create categories, they don't validate markets. Counter-examples:

- **Jepsen** was published before any company paid for distributed systems formal testing. Now it's an industry standard.
- **CLOTHO** was published without a paying customer base for formal isolation specification.
- **Alloy** was published 15 years before Amazon used TLA+ in production.

The relevant question is: "Does this paper teach the database community something it doesn't know?" The answer is yes — engine-specific operational semantics and predicate-level conflict theory are novel contributions regardless of market demand.

---

#### SKEPTIC RESPONSE TO ROUND 2:

**On V=3:** Fair challenge. I concede V=3 underweights the academic value. The problem exists and Hermitage proves it. My concern is that the *marginal value over Hermitage + LLM* is shrinking rapidly. But I'll revise to **V=4** — the problem exists, formal treatment adds value, but the gap between "formal guarantees" and "LLM-generated comprehensive test suite" is narrower than the Synthesizer believes.

**On compound probability:** The Synthesizer is correct that full independence is wrong. I was using it as a conservative bound, but I should have been explicit. With ρ=0.4 correlation, I'd revise P(all 4 fixed) to ~30-35%. My core point stands: the probability of fixing ALL issues is well below 50%, and the project has zero margin for unforeseen problems.

**On 14-day reversal:** This is a legitimate structural observation. I am functionally a conditional continue with a very tight gate. The difference in framing matters, though: my default is "stop spending resources," which I believe is the correct default when a team has produced zero artifacts. The burden of proof should be on continuation, not abandonment.

**On market test:** I partially concede. Academic papers don't need paying customers. But academic papers at SIGMOD/VLDB *do* need to show practical relevance, and "this formalism could theoretically help migrations" without a single concrete case study is a weakness. I maintain this as a concern, but reduce its weight from "FATAL" to "SERIOUS weakness in evaluation section."

---

### Round 3: Auditor Adjudicates

The Auditor evaluates each disputed point with evidence citations.

---

#### Dispute 1: Feasibility (Skeptic F=2, Synthesizer F=6→5, Auditor F=4)

**Auditor's finding:** Both experts make valid but incomplete arguments.

**Where the Skeptic is RIGHT:**
- theory_bytes=0 after theory_complete is an objective execution failure. Evidence: `State.json` shows `theory_bytes: 0, impl_loc: 0`. The phase was designed to produce formalizations, not meta-analysis.
- The seL4 comparison, while extreme, correctly identifies that formal verification has a *qualitatively different* failure mode than software engineering. When Z3 returns `unknown`, you cannot debug it like a segfault.
- Zero artifacts means zero validated assumptions. Every design decision in the 456KB of planning is untested.

**Where the Skeptic is WRONG:**
- F=2 implies the project is nearly impossible. But CLOTHO exists as a published existence proof that SMT-based isolation verification works. The remaining novelty (engine-specific models) is incremental on proven architecture.
- The "zero companies pay" argument conflates commercial viability with technical feasibility. These are orthogonal dimensions.

**Where the Synthesizer is RIGHT:**
- The intellectual heavy lifting (identifying which PG/MySQL behaviors to model, formulating predicate conflict theory, designing the encoding strategy) IS substantially harder than translating to Z3. The 456KB contains genuine technical depth.
- Comparable tools (CLOTHO, Elle, Hermitage) were produced by small teams, suggesting the scope is achievable.

**Where the Synthesizer is WRONG:**
- F=6 (now revised to 5) still overweights planning quality relative to execution evidence. Good plans fail all the time in formal verification. The NULL/3VL issue alone could invalidate the core M5 contribution if the "restrict to NOT NULL" escape hatch proves too limiting.
- "3-4 weeks for all FATALs" was not credible and the revision to 5-8 weeks is still optimistic given zero baseline.

**Auditor's score: F=4.** The path exists (CLOTHO proves it), the plan is detailed, but zero execution evidence after a completed phase is a genuine red flag that prevents scoring above midpoint. This is the single most important pillar for the go/no-go decision.

---

#### Dispute 2: Value (Skeptic V=3→4, Synthesizer V=7, Auditor V=6)

**Auditor's finding:** The Synthesizer is substantially correct.

**Evidence for high value:**
- Hermitage (2,000+ stars) empirically proves the problem exists and the community cares.
- CLOTHO's publication at a top venue validates that formal isolation treatment meets the bar.
- Oracle→PostgreSQL migration is an active, high-stakes industry problem (evidence: `problem_statement.md`, lines citing Gartner/Forrester market data).
- M1 (engine-specific operational semantics) creates *new knowledge* — nobody has formalized PostgreSQL 16.x SSI at verification fidelity. This is a durable academic contribution.

**Evidence for lower value:**
- The LLM displacement argument has partial merit. By 2027-2028, LLMs will likely enumerate *known* isolation differences with high accuracy. The formal guarantee of *completeness* is IsoSpec's moat, but it's unclear how much the market (academic or commercial) values completeness over coverage.
- No production incident evidence (a real-world migration failure caused by isolation divergence) weakens the motivation section.

**Auditor's score: V=6.** The problem is real, the formal treatment adds genuine value over Hermitage/LLMs (completeness guarantees), and M1 creates new knowledge. Docked from 7 because the absence of a concrete migration incident weakens the urgency claim.

---

#### Dispute 3: Compound Probability and Independence

**Auditor's finding:** The Synthesizer is correct on the math, the Skeptic is correct on the implication.

- The Skeptic's independence assumption is mathematically wrong. The four FATALs share a latent competence factor and have structural dependencies (FF1↔FF4 anti-correlation, FF2↔FF3 shared state-space understanding).
- However, even with ρ=0.4 positive correlation, P(all 4 fixed) ≈ 30-35%. This is still well below 50%, and the Skeptic's *qualitative* conclusion — that the project faces substantial compound risk — is correct.
- The correct framing: the project needs *at least 3 of 4* FATALs fixed to be publishable (FF4 can be mitigated by scope restriction). P(≥3 of 4) ≈ 45-55% with correlation, which is a marginal bet.

---

#### Dispute 4: The 14-Day Gate

**Auditor's finding:** The Synthesizer's structural observation is correct — the Skeptic is functionally a conditional continue voter.

- All three experts agree on the *existence* of a mandatory gate before full implementation.
- The disagreement is on (a) the default if the gate fails, and (b) the gate's scope.
- The Skeptic's 14-day gate (working Z3 encoding + fixed proofs + NULL-aware encoding) is aggressive but reasonable as a *proof of concept* scope.
- The Auditor's 2-week vertical slice (PG SSI write skew end-to-end) is a similar scope with slightly different focus.
- The Synthesizer's 5 kill-gates spread over 9 months are too spread out — they delay the critical execution signal.

**Adjudicator's resolution:** Merge the gates. **A single 14-day spike** that produces:
1. A working Z3 encoding of PostgreSQL SSI write skew (≥50 constraints)
2. One FATAL flaw fix (FF3, the k=3 proof, as it's most tractable)
3. A NULL-handling decision with justification (restrict vs. encode)

This is achievable in 14 days, tests execution capability, and provides a clear go/no-go signal.

---

#### Dispute 5: Best-Paper Potential (Skeptic BP=3, Synthesizer BP=6, Auditor BP=5)

**Auditor's finding:** Split the difference, lean toward Synthesizer.

**For BP=6:**
- M1 (engine operational semantics) + M5 (predicate conflict theory) are a genuinely novel combination that no prior work offers.
- If the discovery bet pays off (≥3 Tier-1 migration-affecting findings), the paper writes itself.

**For BP=3:**
- Zero execution means the paper is currently at draft-minus. Best-paper requires flawless execution AND compelling results.
- The CLOTHO lineage creates a "incremental improvement" framing risk that must be overcome.

**Auditor's score: BP=5.** The *potential* is there (M1+M5+discoveries), but best-paper requires both intellectual novelty AND execution quality. Current execution evidence: zero. Score reflects the gap between potential and demonstrated capability.

---

#### Dispute 6: LLM Displacement Threat

**Auditor's finding:** Neither expert fully addresses this correctly.

- The Skeptic overestimates LLM displacement (70-80% by 2027). LLMs cannot provide *soundness guarantees* — they can enumerate known issues but cannot prove the absence of unknown issues. This is a fundamental limitation, not an engineering gap.
- The Synthesizer underestimates the practical impact. For most migrations, "enumerate known issues with 95% coverage" is sufficient. The remaining 5% that needs formal verification is a niche audience.
- **Net assessment:** LLMs narrow the *practical* value of IsoSpec (from "essential tool" to "gold standard for high-assurance migrations") but cannot eliminate the *academic* value (formal verification of engine semantics is new knowledge regardless of LLM capability).

---

## PHASE 2: CONSENSUS SYNTHESIS

### 1. Final Consensus Scores

| Pillar | Skeptic | Synthesizer | Auditor | **Consensus** | **Justification** |
|--------|:-------:|:-----------:|:-------:|:-------------:|-------------------|
| **Value (V)** | 3→4 | 7 | 6 | **6** | Problem is real (Hermitage proves it), formal treatment adds completeness guarantees, M1 creates new knowledge. Docked for no production incident evidence and narrowing LLM gap. Skeptic's V=4 underweights academic contribution; Synthesizer's V=7 is appropriate but Auditor docks for missing evidence. |
| **Difficulty (D)** | 5 | 7 | 7 | **7** | All experts agree the core work (engine semantics + predicate theory) is PhD-level. 55-60K corrected novel LoC is substantial. The Skeptic's D=5 underweights the engine modeling challenge — formalizing PG SSI from source code is genuinely hard. Consensus follows Auditor+Synthesizer. |
| **Best-Paper (BP)** | 3 | 6 | 5 | **5** | Potential exists (M1+M5+discoveries) but zero execution means it's entirely unrealized. BP=5 reflects: high ceiling, no demonstrated progress toward it. Skeptic's BP=3 is too pessimistic (the intellectual contributions ARE there); Synthesizer's BP=6 overweights potential vs. evidence. |
| **Likelihood (L)** | 6 | 8 | 8 | **7** | IF the team executes, the work fits SIGMOD/VLDB/CIDR well. The venue match is strong. Docked from 8→7 because the CLOTHO lineage creates a framing challenge that requires careful positioning. All experts agree the venue fit exists; disagreement is on execution conditioning. |
| **Feasibility (F)** | 2 | 6→5 | 4 | **4** | THE PIVOTAL SCORE. Zero artifacts after theory_complete is an objective execution failure. CLOTHO existence proof prevents scoring below 3. Detailed planning quality prevents scoring below 3. But untested assumptions + compound FATAL risk prevents scoring above 5. Consensus: marginal feasibility that requires immediate evidence. |

### 2. Final Composite Score

| | Skeptic | Synthesizer | Auditor | **Consensus** |
|---|:---:|:---:|:---:|:---:|
| **Total** | 19/50 | 34/50 | 30/50 | **29/50** |

**Interpretation:** 29/50 falls in the "marginal — requires immediate evidence to justify continuation" band. This is above the Skeptic's abandon threshold (~25) but below the Synthesizer's comfortable-continue threshold (~35).

### 3. Final Verdict: CONDITIONAL CONTINUE with BINDING GATE

**Verdict:** CONDITIONAL CONTINUE

**NOT ABANDON because:**
- The intellectual core (M1 + M5) is genuinely novel — this was unanimous
- CLOTHO proves the architecture works — the technical risk is in engine-specific extensions, not in the approach
- The problem has verified industry relevance
- All three experts agree on a gate mechanism, disagreeing only on framing

**NOT UNCONDITIONAL CONTINUE because:**
- theory_bytes=0 after theory_complete is an execution failure that cannot be hand-waved
- 0/17 amendments completed; 4 FATAL flaws unresolved
- Zero executable artifacts means every assumption is untested
- Compound FATAL risk is ~30-35% for full resolution (even with correlation correction)

---

### BINDING GATE: 14-Day Execution Spike

**Deadline:** 14 calendar days from decision date.

**Must produce (ALL THREE required):**

| Deliverable | Acceptance Criteria | Tests |
|-------------|-------------------|-------|
| **Z3 encoding of PG SSI write skew** | ≥50 SMT-LIB constraints; Z3 returns `sat` or `unsat` (not `unknown`) within 60 seconds; encodes SIREAD lock acquisition, dangerous structure detection, and abort for ≥2 concurrent transactions | Runs on Z3 4.12+; deterministic result |
| **FF3 resolution: k-bound proof** | Either: (a) corrected proof with explicit case analysis for G1a at k=2, or (b) explicit bounded claim with formal statement of what k=3 does NOT cover | Peer-reviewable LaTeX or markdown with no hand-waving |
| **NULL handling decision document** | Either: (a) sound 3VL encoding sketch with worked example, or (b) NOT NULL restriction with scope impact analysis (what % of TPC-C/TPC-H queries are excluded) | Quantitative scope impact assessment |

**If gate PASSES:** Full implementation begins under Path B (PG + MySQL, 2 engines, ~9 months).
**If gate FAILS:** Project is ABANDONED. Salvage M1 theory as a standalone workshop paper (PODS/CIDR).

---

### 4. Areas of Genuine Remaining Disagreement

#### Disagreement 1: LLM Displacement Timeline (UNRESOLVED)
- **Skeptic:** 70-80% displacement by 2027. Formal verification becomes niche.
- **Synthesizer:** LLMs cannot provide soundness. Formal verification remains essential.
- **Auditor:** LLMs narrow practical value but cannot eliminate academic value.
- **Status:** Unknowable. Mitigated by: if the paper is published before 2028, LLM displacement is moot for academic credit.

#### Disagreement 2: Feasibility Variance (PARTIALLY RESOLVED)
- **Skeptic:** F=2 (near-impossible).
- **Synthesizer:** F=5 (marginal but achievable).
- **Auditor:** F=4 (executable but high-risk).
- **Consensus F=4** but the Skeptic formally dissents. The 14-day gate resolves this empirically.

#### Disagreement 3: Optimal Scope (RESOLVED IN FAVOR OF SYNTHESIZER)
- **Skeptic:** Abandon entirely; salvage pieces.
- **Synthesizer:** Path B (2 engines, 9 months).
- **Auditor:** Vertical slice first, then decide scope.
- **Resolution:** The gate mechanism subsumes this disagreement. If the gate passes, Path B is the consensus path.

#### Disagreement 4: CLOTHO Differentiation Sufficiency (PARTIALLY RESOLVED)
- **Skeptic:** Strip engine models → CLOTHO with a SQL parser. Insufficient novelty.
- **Synthesizer:** M1 (engine models) IS the contribution. Cannot be stripped.
- **Auditor:** "Sufficient but fragile." Requires careful framing.
- **Resolution:** The Synthesizer's framing is correct — M1 is load-bearing and cannot be factored out. But the Skeptic is right that the *paper* must lead with M1/M5 discoveries, not with the tool. Framing as "discovery paper with a tool" (not "tool paper with discoveries") is the consensus strategy.

### 5. Publication Probability Estimates

| Scenario | P(best-paper) | P(tier-1 accept) | P(any publication) |
|----------|:---:|:---:|:---:|
| **Current state (0 artifacts)** | 0% | 2-5% | 15-25% |
| **After gate passes (14 days)** | 2-4% | 15-25% | 40-55% |
| **After all FATALs fixed** | 5-10% | 30-45% | 55-70% |
| **After full Path B execution** | 8-15% | 45-60% | 65-80% |
| **After Path B + ≥3 Tier-1 discoveries** | 12-20% | 55-70% | 75-85% |

**Methodology:** These are consensus estimates formed by:
- Starting from the Skeptic's base rates (most conservative, evidence-grounded)
- Applying the Synthesizer's correlation correction (mathematically sound)
- Conditioning on the Auditor's phased gate structure (removes unrealistic paths)

**Key insight:** The *largest single jump* in P(pub) is from "current state" to "after gate passes." The gate is worth ~25-30 percentage points because it resolves the core uncertainty (can this team execute?). This validates the gate as the correct decision mechanism.

---

## APPENDIX: Expert Scoring Reconciliation

### Why Each Expert Was Right

| Expert | Core Insight That Survived Adjudication |
|--------|----------------------------------------|
| **Skeptic** | Zero artifacts after theory phase IS a genuine execution failure, not a process artifact. The compound FATAL risk IS real. The project IS a marginal bet. |
| **Synthesizer** | M1 IS the crown jewel and IS anti-LLM. Path B IS the optimal scope. The FATALs ARE structurally correlated. Academic papers DON'T need paying customers. |
| **Auditor** | The execution gap IS the pivotal uncertainty. A time-bounded gate IS the correct decision mechanism. The truth IS between the extremes. |

### Why Each Expert Was Wrong

| Expert | Overclaim or Error |
|--------|-------------------|
| **Skeptic** | V=3 underweights verified academic value. Compound probability assumed full independence. "ABANDON" recommendation is structurally a conditional continue with a different label. |
| **Synthesizer** | "3-4 weeks for all FATALs" was not credible. F=6 overweights planning relative to execution. P(pub) 65-75% included unstated venue scope. |
| **Auditor** | Occupied a safe middle ground that risks being correct without being *useful*. The 2-week vertical slice was too similar to the Skeptic's 14-day spike to constitute an independent contribution. |

---

*Adjudication complete. The 14-day gate is now the single decision point. Everything else is commentary.*
