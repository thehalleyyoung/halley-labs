# Verification Gate: SafeStep (proposal_00) — Skeptic Panel Assessment

**Project:** SafeStep — Verified Deployment Plans with Rollback Safety Envelopes for Multi-Service Clusters  
**Slug:** `safe-deploy-planner`  
**Stage:** Theory → Implementation Gate  
**Method:** 3-expert adversarial panel (Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer) with cross-critique synthesis and independent verifier signoff  
**Date:** 2026-03-08  
**Artifacts Reviewed:** approach.json (50KB), paper.tex (2030 lines), verification_signoff.md (342 lines), depth_check.md (202 lines), final_approach.md (~350 lines), approaches.md, approach_debate.md, State.json  
**Prior Gate Scores:** Ideation V6/D6/BP5.5/L6.5 (composite 6.0). Theory verification_signoff V7/D7/BP6/L7.5/F7 (composite ~7.0).

---

## Executive Summary

SafeStep proposes a genuinely novel operational primitive — the rollback safety envelope — that fills a verified gap in the deployment toolchain. No existing tool, academic or industrial, computes bidirectional reachability under safety invariants for deployment states. This single concept justifies continued development. However, the project is fundamentally an *application paper* that composes known techniques (BMC, CEGAR, treewidth DP, interval encoding) in a new domain, its strongest empirical claims rest on a phantom dataset with zero documented methodology, and its practical value depends entirely on an unvalidated schema-based oracle. The prior review round (verification_signoff) scored this at V7/D7/BP6/L7.5/F7 — we find these scores inflated by approximately 1.0-1.5 points per dimension and correct downward. The project should continue, gated by Phase 0 oracle validation and 847-project dataset substantiation, targeting EuroSys/NSDI as the primary venue rather than SOSP/OSDI.

---

## Panel Composition and Process

| Expert | Role | Methodology | Composite Score |
|--------|------|-------------|----------------|
| Independent Auditor | Evidence-based scoring, challenge testing | Bottom-up artifact verification; tested paper's motivating incidents against oracle capability | 5.4/10 |
| Fail-Fast Skeptic | Aggressively reject under-supported claims | 6 attack vectors seeking fastest path to ABANDON | ~5.0/10 |
| Scavenging Synthesizer | Salvage value, identify reframing opportunities | Salvage analysis of 5 elements; 3 reframing options; risk-reward analysis | 6.8/10 |
| **Consensus** | Cross-critique synthesis + independent verifier signoff | Challenge-by-challenge adjudication with evidence-based resolution | **5.7/10** |

**Process:** Independent proposals (no cross-contamination) → adversarial cross-critique with 6 specific challenges → synthesis with evidence-based adjudication → independent verifier signoff (APPROVED WITH CONCERNS, 3 corrections issued). The cross-critique substantively moved positions: Synthesizer's composite dropped 1.0 points; Auditor's bilateral DC credit rose 0.5 points; Skeptic's dataset severity was elevated.

---

## Pillar 1: Extreme Value — 6/10

### Evidence FOR (what pushes above 5):
- **Real incidents cited.** Google Cloud SQL 2022 (4h47m), Cloudflare 2023 (2h+), AWS Kinesis 2020. Verifiable, named, consequential. (+2)
- **Combinatorial argument is sound.** 5^30 ≈ 9.3×10^20 intermediate states for a 30-service cluster. CI tests pairwise compatibility between adjacent versions; SafeStep reasons about the full N-service rollback state space. This is NOT "catching what CI catches." (+1)
- **No existing tool answers "is rollback safe from here?"** Verified against Spinnaker, ArgoCD, Flux, Aeolus/Zephyrus, McClurg et al. The gap is real. (+1)
- **Compliance value floor.** SOC2/HIPAA/PCI-DSS require documented change management. Machine-verified deployment plans with rollback annotations exceed current practice. (+1)

### Evidence AGAINST (what keeps this at 6):
- **Oracle coverage is unvalidated.** The paper claims ≥60% structural coverage but this is hypothesized, not measured. (-1)
- **The paper's own motivating incidents suggest 33-66% coverage.** The Auditor independently classified: Google Cloud SQL ✓ (schema-detectable), Cloudflare ? (ambiguous — token format change is partially structural), AWS Kinesis ✗ (thread-limit dependency — entirely behavioral/operational). If the paper's three showcase incidents yield 1 clear catch, 1 ambiguous, and 1 miss, baseline coverage expectations should be conservative. (-1)
- **Schema analysis may catch primarily low-value failures.** Structural breaking changes (removed fields, type changes) are already the *easiest* class to detect via CI contract testing. The high-value, 3:47 AM failures tend to be behavioral (changed semantics, performance regressions, race conditions) — explicitly outside oracle scope. (-0.5)
- **False confidence risk.** Operators seeing GREEN "structurally verified" stamps may reduce vigilance on the uncovered 40%. Confidence coloring mitigates but does not eliminate this risk. (-0.5)

### Panel Disagreement (5-5-7):
The Auditor and Skeptic anchor on the unvalidated oracle; the Synthesizer credits the envelope concept's standalone and compliance value. **Adjudication:** The envelope concept does have standalone value (even a theory paper is publishable), but the full value proposition — *preventing deployment failures* — requires the oracle to work. Value scored at 6, not 5 (the compliance floor and combinatorial argument are real) and not 7 (unvalidated oracle caps practical value).

---

## Pillar 2: Genuine Software Difficulty — 5.5/10

### What is genuinely novel (~15K LoC):
- **Rollback safety envelope computation** — bidirectional reachability under safety invariants with incremental SAT. Conceptually novel, algorithmically standard (bidirectional BFS in product graph via SAT queries). (+2)
- **Monotone sufficiency exchange argument** (Theorem 1) — including the bilateral DC discovery. A genuine, non-trivial proof with a real subtlety. (+1)
- **Interval encoding exploitation** — the observation that version-compatibility predicates have contiguous compatible ranges, enabling O(log²L) encoding. Domain-specific but useful. (+0.5)

### What is standard technique application (~20K LoC):
- **BMC** (Clarke et al., 2001): Standard 25-year-old technique. Paper acknowledges: "BMC is the computational mechanism, not the contribution." (0)
- **CEGAR** (Clarke et al., 2003): Standard CaDiCaL↔Z3 split. Theorem 4 is a standard correctness argument with no novel content. (0)
- **Treewidth DP** (Bodlaender, 1993): Textbook parameterized complexity. Domain-specific DP formulation is a trivial variation. (0)
- **k-robustness via brute-force enumeration** at k=2 with ≤45 SAT calls: A for-loop, not an algorithm. Theorem 5 (Adversary Budget Bound) provides post-hoc justification disconnected from implementation (theory says k_α=10; system uses k=2). (0)

### What is engineering (~25K LoC):
- **Schema oracle** (~10K LoC): OpenAPI + Protobuf parsing, schema diffing, breaking-change classification. Hard engineering but not novel.
- **K8s integration, evaluation infra, testing** (~15K LoC): Necessary, not novel.

### Honest theorem count:
- **2 genuinely important results:** Theorem 1 (Monotone Sufficiency), Theorem 2 (Interval Encoding)
- **2 standard results correctly applied:** Theorem 4 (CEGAR), Proposition A (Complexity characterization)
- **2 useful but removable results:** Theorem 3 (Treewidth DP), Proposition B (Replica Symmetry)
- **1 decorative result:** Theorem 5 (Adversary Budget — practically disconnected from implementation)

The verification_signoff's characterization of "5 theorems + 2 propositions, all with full proofs" over-credits. Honest: 2 key theorems + 5 supporting results (1 decorative). The bilateral DC discovery deserves partial credit (+0.5, not +1.0) as honest mathematical engagement.

---

## Pillar 3: Best-Paper Potential — 4/10

### The case FOR:
- **The rollback safety envelope could enter permanent vocabulary** — like "linearizability" or "eventual consistency," it names a concept operators intuit but lack formal language for. (+2)
- **The paper is unusually well-written.** The 3:47 AM scenario, the stuck-configuration witness walkthrough, the honest limitations framing. (+1)
- **The paper has the *shape* of a strong submission** — novel concept + theorems + system + evaluation design. (+1)

### The case AGAINST:
- **Every technique is borrowed.** BMC (2001), CEGAR (2003), treewidth DP (1993), interval encoding (standard SAT), bidirectional reachability (standard model checking). No technique advances the state of the art. (-2)
- **The SDN parallel is dangerous.** A hostile reviewer will frame this as "McClurg et al. (PLDI 2015) for Kubernetes." The differentiation (rollback analysis, oracle uncertainty) is real but may not overcome the "domain translation" framing. (-1)
- **The evaluation does not yet exist.** Plans produce zero credit. The killer result (finding a real unsafe rollback state in DeathStarBench) is prospective. (-2)
- **Conditional guarantees.** SafeStep's formal guarantees are relative to an unvalidated oracle. Ironfleet, CertiKOS, seL4 — actual best papers — provide unconditional guarantees. (-1)

### Best-paper probability estimate:
- P(best paper at any top venue) = **5-7%**
- Prior reviews' 8-12% is optimistic; the Skeptic's 3% is slightly too harsh.
- The project needs the DeathStarBench evaluation to discover a real unsafe rollback state AND the paper to be reframed concept-first (Synthesizer's Recommendation 1) to have realistic best-paper odds.

---

## Pillar 4: Laptop-CPU Feasibility & No-Humans — 7/10

**This is the strongest pillar.** All three panelists converge (7-7-7.5).

- **Clause counts check out.** 14.4M clauses at n=50, L=20, k=200. CaDiCaL routinely handles 50M+. Well within bounds. (+2)
- **Envelope computation with binary search is efficient.** ~8 incremental SAT calls for k=200 ≈ 16 seconds. (+2)
- **k-robustness is trivial.** 45 SAT calls at k=2 ≤ 55 seconds. (+1)
- **No GPU, no human annotation, no human studies.** Pipeline from Helm charts → verified plans is fully automated. (+1)
- **Treewidth DP honestly scoped.** tw≤3, L≤15 only. SAT/BMC handles all other cases. (+1)
- **Minor concerns:** CEGAR convergence (5-20 iterations) is empirical not proven; clause count discrepancies between approach.json (~20M) and paper.tex (~14.4M) need reconciliation; n>100 enters "hours" regime. (-1)

---

## Pillar 5: Feasibility — 6/10

### Evidence FOR:
- **Theorem-to-module mapping is explicit.** Each theorem → named Rust module with LoC estimate. (+1)
- **Dependencies available.** cadical-sys crate (v0.6.0), z3-sys crate (v0.8.x), helm template subprocess, prost for protobuf. (+1)
- **LoC estimates are honest.** 51-77K (midpoint ~60K) is realistic after the scope corrections. (+1)
- **Evaluation plan has concrete phases** with named benchmarks (DeathStarBench, TrainTicket, Sock Shop) and baselines. (+1)

### Evidence AGAINST:
- **847-project dataset may not exist** as a reproducible artifact. Methodology absent from 218KB+ of theory artifacts. If non-existent, 2-4 weeks to construct. (-1)
- **Phase 0 postmortem availability uncertain.** Finding 15-20 postmortems specifically about cross-service version incompatibility (not just "deployment went wrong") is non-trivial. (-0.5)
- **DeathStarBench adaptation is significant.** Synthetic version histories with controlled schema evolution require 4-6 weeks. (-0.5)
- **Timeline aggressive.** ~60K LoC Rust + evaluation + paper writing = realistically 9-12 months for a solo researcher. (-1)
- **Bilateral DC prevalence unquantified** under the strengthened bilateral condition. (-0.5)

---

## Fatal Flaw Analysis

### Flaw 1: Phantom 847-Project Dataset — SERIOUS (trending POTENTIALLY FATAL)

**Statement:** Three load-bearing empirical claims cite "847 open-source microservice projects" with zero documented methodology across 218KB of theory artifacts:
- ">92% interval structure prevalence" (determines encoding tractability — Theorem 2)
- "Median treewidth 3-5" (determines DP applicability — Theorem 3)
- "18-23% of outages involve version incompatibility" (determines problem importance)

**The Skeptic's analysis is most incisive:** These are likely "aspirational estimates presented as empirical findings." The 92% interval structure claim may rest on an observation that semver ranges are definitionally contiguous intervals — trivially true for package dependencies, a poor proxy for API compatibility predicates. The "200+ postmortems" supporting the 18-23% outage rate are never cited in any artifact.

**Rating: SERIOUS (trending POTENTIALLY FATAL)** — Presenting uncorroborated claims as empirical findings is an integrity issue, not a to-do item. However, the claims are plausible and constructible: semver ranges ARE intervals; microservice graphs ARE empirically sparse. A proper dataset can likely be built in 2-4 weeks.

**Resolution:** Produce the dataset with documented methodology within 4 weeks or explicitly weaken claims to "hypothesized structural properties to be validated during implementation." Measure bilateral DC prevalence. If interval structure under bilateral DC < 75%, revise tractability narrative.

### Flaw 2: Oracle Accuracy Untested — SERIOUS

**Statement:** The entire value proposition is conditional on schema analysis catching a meaningful fraction (≥40%) of real deployment failures. No experiment has been run. The 15-postmortem Phase 0 design has ±12% CI at 95%. The Auditor's independent analysis of the paper's 3 showcase incidents suggests 33-66% oracle coverage.

**The statistical inadequacy problem:** With n=15 and ±12% CI, Phase 0 cannot distinguish "viable" (60%) from "pivot" (<40%). The Skeptic's recommendation to increase to n≥20 is adopted. In the ambiguity zone (40-55%), the project should limit scope to a concept paper targeting HotOS/SoCC rather than the full system.

**Rating: SERIOUS** — not FATAL because the paper is honest about the limitation, has a pivot plan, and the envelope concept has independent theoretical value.

### Flaw 3: Bilateral DC Prevalence Unknown — SERIOUS

**Statement:** Theorem 1 (the most important theorem) requires bilateral DC — strictly stronger than the originally stated unilateral DC. The ">92%" figure was measured (if at all) under unilateral DC. Bilateral DC prevalence is unknown. If < 75%, monotone sufficiency applies to a minority of pairs and the completeness bound weakens.

**Rating: SERIOUS** — natural condition, graceful degradation designed, quantification planned in Phase 0.

### Flaw 4: Borrowed Techniques / Novelty Deficit — SERIOUS

**Statement:** Every algorithm in SafeStep (BMC, CEGAR, treewidth DP, interval encoding, bidirectional reachability) is a standard technique from the verification/parameterized complexity literature. The novelty is in the domain application and the envelope concept. A hostile SOSP/OSDI reviewer will write: "This reads as a solid EuroSys paper that has been scope-inflated to target SOSP."

**Rating: SERIOUS** — publication risk, not project risk. EuroSys/NSDI accept strong application papers.

### Flaw 5: Evaluation Has No Real-World Comparison — SERIOUS

**Statement:** No comparison to actual deployment tools (ArgoCD, Flux, Spinnaker) or actual deployment practices (canary + integration tests). Baselines are weak (topological sort, random plans, Fast Downward). The sample sizes (n=15 postmortems) are too small for statistical conclusions.

**Rating: SERIOUS** — the "find a real bug in DeathStarBench" goal is the strongest evaluation element if achieved.

### Flaw 6: Envelope Prefix Property Unproven — MODERATE

**Statement:** Binary search optimization (8 SAT calls instead of 200) assumes the envelope is a prefix of the plan. The verification_signoff identifies the argument as incomplete. If the prefix property fails, envelope computation degrades to 6-7 minutes (linear scan) instead of 16 seconds.

**Rating: MODERATE** — correctness unaffected; only performance. Linear scan is an acceptable fallback, but the paper's "under 20 seconds" claim must be qualified.

### Flaw 7: False Confidence Risk — MODERATE

**Statement:** Operators seeing GREEN "structurally verified" stamps may reduce vigilance on uncovered behavioral failures. Confidence coloring mitigates but does not eliminate this risk.

**Rating: MODERATE** — the paper should add prominent "STRUCTURALLY VERIFIED ≠ SAFE" disclaimers.

### Flaw 8: Theorem 5 is Decorative — LOW-MODERATE

**Statement:** The Adversary Budget Bound computes k_α=10 but the system uses k=2. The theorem provides post-hoc justification disconnected from implementation.

**Rating: LOW-MODERATE** — downgrade to Remark. Present k=2 robustness as a practical heuristic.

---

## Prior Art Comparison

| System | Problem | Rollback Analysis? | Schema Oracle? | Path Planning? | Gap SafeStep Fills |
|--------|---------|--------------------|----|---|----|
| **Aeolus** (ESOP 2014) | Configuration synthesis | No | No | No (target only) | Path planning + rollback envelope |
| **Zephyrus** (FACS 2015) | Configuration optimization | No | No | No | Same as Aeolus |
| **McClurg et al.** (PLDI 2015) | SDN consistent updates | **No — forward only** | No | Yes | **Rollback analysis genuinely absent from SDN work.** Oracle uncertainty model novel. |
| **Reitblatt et al.** (SIGCOMM 2012) | SDN consistent update | No | No | Partial | Same as McClurg |
| **Spinnaker/ArgoCD/Flux** | Deployment orchestration | Manual rollback only | No | No | Formal safety analysis |
| **PDDL/Fast Downward** | General planning | No | No | Yes (general) | Domain-specific encoding; rollback envelope |
| **EDOS/OPIUM** | Package dependency | No | No | No | Distributed multi-service path planning |

**Verdict:** No existing tool computes rollback safety envelopes. The gap is real and verified. The closest competitor (McClurg et al.) solves forward safe-transition synthesis for SDN — SafeStep's forward planning is structurally similar, but rollback analysis and oracle uncertainty are genuinely new.

---

## Score Summary and Trajectory

| Dimension | Ideation | Depth Check | Verification Signoff | **This Panel** | Delta from Signoff |
|-----------|----------|-------------|---------------------|---------------|-------------------|
| V (Value) | 6 | 6 | 7 | **6** | **-1** |
| D (Difficulty) | 6 | 6 | 7 | **5.5** | **-1.5** |
| BP (Best Paper) | 5.5 | 5.5 | 6 | **4** | **-2** |
| L (Laptop) | 6.5 | 6.5 | 7.5 | **7** | **-0.5** |
| F (Feasibility) | — | — | 7 | **6** | **-1** |

**Composite: 5.7/10** — corrects the inflationary trend from ideation (6.0) through verification signoff (~7.0). The correction is justified: prior reviews progressively gave credit for plans and promises; this panel gives credit only for demonstrated evidence.

### Why Our Scores Are Lower

1. **Value (-1):** We independently analyzed the 3 motivating incidents and found only 1/3 clearly schema-detectable. Prior reviews accepted the oracle coverage claim without testing it.
2. **Difficulty (-1.5):** We categorized every theorem's novelty: 2/7 genuinely novel, not 5-6 as prior reviews imply. "Application of known techniques to new domain" should be scored as such.
3. **Best Paper (-2):** Plans produce zero credit. Best-paper odds for an application paper with borrowed techniques, unexecuted evaluation, and conditional guarantees are 5-7%, not 8-12%.
4. **Laptop (-0.5):** Clause count discrepancies and unproven CEGAR convergence. Minor.
5. **Feasibility (-1):** The 847-project dataset's existence is unverified.

---

## Probability Estimates

| Outcome | Skeptic | Auditor | Synthesizer | **Consensus** |
|---------|---------|---------|-------------|---------------|
| P(SOSP/OSDI) | 20-25% | 20-25% | 35-40% | **22-28%** |
| P(EuroSys/NSDI) | 45-55% | 45-55% | 60-65% | **50-58%** |
| P(ICSE/FSE) | — | 55-65% | 70-75% | **55-65%** |
| P(best paper any venue) | 5-8% | 3-5% | 10-12% | **5-7%** |
| P(oracle coverage ≥60%) | ~40% | 40-50% | 55-65% | **45-55%** |
| P(should be abandoned) | 25-30% | 20% | 10-15% | **20-28%** |
| P(delivers on core claims) | 40-50% | 35-45% | 50-55% | **40-48%** |

---

## Verdict: CONDITIONAL CONTINUE

**All three panelists reach CONTINUE** (Skeptic reluctantly, Auditor conditionally, Synthesizer optimistically). No panelist found a genuinely FATAL flaw. The Skeptic prosecuted 6 attack vectors and concluded: "I looked hard for a FATAL flaw and did not find one." The independent verifier confirmed: "APPROVED WITH CONCERNS."

### Why CONTINUE:
1. **The rollback safety envelope is genuinely novel.** No prior tool computes bidirectional deployment reachability under safety invariants. The concept fills a verified gap.
2. **The formal framework is sound** (relative to the oracle). Theorem 1 and Theorem 2 are correct and useful.
3. **The paper is well-written** with unusual intellectual honesty about limitations.
4. **The worst-case outcome still produces a publishable contribution** (envelope concept as position paper or short paper).
5. **The pivot plan is designed** — if oracle coverage < 40%, redirect to theory/concept paper.

### Why This Is a Close Call:
1. **Phantom dataset.** Three load-bearing empirical claims rest on uncorroborated evidence.
2. **Untested oracle.** The value proposition depends on a schema-based oracle that has never been validated.
3. **Borrowed techniques.** Every algorithm is standard; novelty is in the domain application and concept.
4. **No results yet.** The evaluation is entirely prospective.
5. **P(should be abandoned) = 20-28%.** This is not a comfortable margin.

---

## Conditions for CONTINUE

### HARD GATES (failure → pivot or abandon)

**1. Phase 0 oracle validation: structural coverage ≥ 40%, n ≥ 20 postmortems.**
- Execute within 2 weeks as the first task.
- If coverage ≥ 60%: full system paper, target EuroSys/NSDI.
- If coverage 40-55%: limit implementation to core engine (~25K LoC), target concept paper at HotOS/SoCC.
- If coverage < 40%: ABANDON full system; pivot to theory/concept paper about the envelope abstraction.

**2. 847-project dataset: documented methodology within 4 weeks.**
- Produce: project selection criteria, compatibility predicate extraction method, interval structure verification procedure, bilateral DC test results.
- If interval structure under bilateral DC < 75%: revise tractability narrative.
- If dataset cannot be constructed: explicitly weaken claims to "hypothesized structural properties."

**3. Bilateral DC prevalence ≥ 75%.**
- Quantify during Phase 0 / dataset construction.
- If < 75%: provide explicit degradation analysis showing search space expansion is manageable.
- If < 60%: Theorem 1 applies to a minority of pairs; consider abandoning the monotone reduction as a primary contribution.

### SOFT CONDITIONS (before submission)

4. **Envelope prefix property:** Formalize as a lemma or drop binary search (use linear scan fallback; adjust performance claims from "under 20 seconds" to "under 7 minutes").
5. **Clause count reconciliation:** approach.json says ~20M total; paper.tex says ~14.4M. Pick one, justify, use consistently.
6. **Paper reframing (Synthesizer's recommendations adopted):**
   - Lead abstract/introduction with the envelope concept, not the BMC formalism.
   - Cut to tighter structure (concept + mechanism + ONE killer experiment).
   - Frame oracle limitation as intellectual contribution (confidence coloring + k-robustness = formal methods under uncertainty).
7. **Theorem 5 downgraded to Remark.** Present k=2 robustness as a practical heuristic, not a theorem-driven choice.
8. **Primary venue = EuroSys/NSDI.** SOSP/OSDI as a stretch submission only.
9. **Add prominent "STRUCTURALLY VERIFIED ≠ SAFE" disclaimer** to tool output framing and paper.
10. **Timeline reality check:** If solo researcher, target the tighter concept-paper version (~25-35K LoC).

---

## Reframing Recommendations (from Synthesizer, endorsed by panel)

1. **Lead with the envelope, not the BMC.** Current abstract: "SafeStep computes rollback safety envelopes via bounded model checking..." Proposed: "We introduce the rollback safety envelope: a new operational primitive that maps each intermediate deployment state to its rollback feasibility..." One sentence changes reviewers' frame from "BMC paper" to "new abstraction paper."

2. **Frame oracle limitation as a contribution.** "We contribute the first formal framework for reasoning about deployment safety under imperfect constraint knowledge. The framework cleanly separates the oracle from the verifier, enabling confidence-colored constraints, k-robustness certification, and principled degradation."

3. **State the "finding a real bug" evaluation criterion upfront.** "Our primary evaluation goal is to discover a genuine unsafe rollback state in DeathStarBench — a deployment ordering where rollback becomes unsafe due to cross-service version incompatibility."

4. **Add explicit compliance/audit subsection.** SOC2 CC6.1, HIPAA §164.312, PCI-DSS 6.5.6 all require documented change management. This provides a value floor independent of oracle quality.

5. **Consider database migration domain as a fallback.** The oracle problem is dramatically easier for database schemas (machine-readable by definition, compatibility well-defined). If microservice oracle fails, the envelope concept transfers directly to database migration orchestration.

---

## Salvage Priority (If Project Is Abandoned)

| Element | Standalone Value | Publication Vehicle |
|---------|-----------------|-------------------|
| Rollback safety envelope concept | HIGH | HotOS/HotNets position paper (2-4 pages) |
| Monotone sufficiency under bilateral DC | MODERATE | Workshop paper or CAV/TACAS short paper |
| Oracle confidence + k-robustness design pattern | MODERATE | Technical report or SE venue |
| Interval encoding observation | LOW-MODERATE | Supporting result in future work |

The envelope concept survives all failure modes and is worth preserving regardless of project outcome.

---

## Comparison to Prior Verification Gate

| Dimension | Prior Signoff | This Panel | Change | Justification |
|-----------|--------------|------------|--------|---------------|
| V | 7 | 6 | -1 | Auditor's incident analysis shows 33-66% oracle coverage from paper's own examples |
| D | 7 | 5.5 | -1.5 | Only 2/7 theorems genuinely novel; rest are standard technique application |
| BP | 6 | 4 | -2 | Plans get zero credit; borrowed techniques + no results = low best-paper odds |
| L | 7.5 | 7 | -0.5 | Minor; correcting error is debt payment not improvement |
| F | 7 | 6 | -1 | 847-project dataset existence unverified; timeline aggressive |
| **Composite** | **~7.0** | **5.7** | **-1.3** | Score inflation corrected |

The prior signoff identified 1 SERIOUS issue (bilateral DC prevalence) and 2 MODERATE issues (envelope prefix, 847 methodology). This panel identifies 5 SERIOUS issues and 3 MODERATE issues, including the phantom dataset as "SERIOUS trending POTENTIALLY FATAL." The scoring correction of -1.3 composite reflects stricter evidence requirements and the discovery of the dataset integrity concern.

---

## Panel Sign-off

| Expert | Verdict | Composite | Key Finding |
|--------|---------|-----------|-------------|
| Independent Auditor | CONDITIONAL CONTINUE | 5.4 | 1/3 motivating incidents clearly schema-detectable; prior reviews inflated |
| Fail-Fast Skeptic | CONTINUE (reluctant) | ~5.0 | No FATAL flaw found despite 6 attack vectors; phantom dataset is most damaging |
| Scavenging Synthesizer | CONDITIONAL CONTINUE | 6.8 | Envelope concept is "permanent vocabulary" potential; reframe concept-first |
| Cross-Critique | CONDITIONAL CONTINUE | 5.7 | Auditor's evidence-based anchor prevails; Synthesizer overcredits hypothetical value |
| Independent Verifier | APPROVED WITH CONCERNS | — | P(abandon) corrected to 20-28%; ambiguity zone needs concrete decision rule |
| **FINAL CONSENSUS** | **CONDITIONAL CONTINUE** | **V6/D5.5/BP4/L7/F6 = 5.7** | **Phase 0 is the binding gate. EuroSys/NSDI primary target.** |

---

*Assessment produced by 3-expert adversarial verification panel with cross-critique synthesis and independent verifier signoff. Independent proposals (no cross-contamination) → 6-challenge adversarial cross-critique → evidence-based synthesis → verifier signoff with 3 corrections. All scores justified by specific evidence. Score inflation from prior rounds explicitly identified and corrected.*
