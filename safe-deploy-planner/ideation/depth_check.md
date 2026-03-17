# Depth Check: SafeStep

**Proposal:** SafeStep: Verified Deployment Plans with Rollback Safety Envelopes for Multi-Service Clusters
**Slug:** `safe-deploy-planner`
**Evaluator:** Impartial Verification Panel (3 experts: Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer)
**Method:** Independent proposals → adversarial cross-critique → synthesis with explicit challenge/response
**Date:** 2026-03-07

---

## Verdict: CONDITIONAL CONTINUE

**Composite Score: 6.0/10** (V6/D6/BP5.5/L6.5)

The rollback safety envelope is a genuinely novel operational concept — no existing tool, academic or industrial, computes bidirectional reachability under safety invariants for deployment states. This single idea justifies continued development. However, the proposal substantially over-claims on multiple axes: inflated LoC, infeasible treewidth DP at stated parameters, internal encoding-bound inconsistency, unverified empirical assertions, and an unvalidated oracle that is the binding constraint on the entire system's practical value. The project should continue only if the 7 mandatory amendments below are applied and the oracle validation experiment yields ≥60% structural-failure coverage.

---

## Pillar 1: EXTREME AND OBVIOUS VALUE — 6/10

**The problem is real.** Cross-service version incompatibility during rolling updates causes catastrophic outages — Google Cloud SQL 2022, Cloudflare 2023, AWS Kinesis 2020. The combinatorial explosion of intermediate states (5^30 ≈ 9.3×10^20 for a 30-service cluster) defeats human reasoning. No existing tool answers "can I safely roll back from this specific intermediate state?" The rollback safety envelope fills a genuine gap.

**The value is conditional on oracle accuracy.** SafeStep's formal guarantees are only as strong as its schema-derived compatibility constraints. Schema analysis catches structural breaks (removed fields, type changes) but is blind to behavioral incompatibilities (changed semantics, different error handling, performance regressions). The proposal claims "even a 70% oracle catches the majority of failures" but provides no evidence for the 70% figure or the power-law distribution it assumes. This is the central empirical question.

**Panel disagreement (4-6-8).** The Skeptic argues false confidence is worse than no tool — operators who trust "formally verified" plans will lower their guard. The Synthesizer argues this proves too much (it would indict all static analysis), and frames the oracle as a designable parameter, not a flaw. The Auditor takes the middle: without an oracle validation experiment, both positions are unfalsifiable.

**What elevates this beyond 6:**
- An oracle validation experiment showing schema analysis catches ≥60% of the 15 reconstructed incident root causes.
- Explicit compliance/audit value framing — SOC2/HIPAA/PCI-DSS require documented change management; a machine-verified deployment plan with rollback annotations is exactly what auditors demand.
- Honest, qualified language throughout: "structurally verified" not "formally verified."

**What would drop this below 6:**
- If schema analysis catches <40% of real deployment failures, the value proposition collapses.
- If the paper implies unqualified safety guarantees without prominently disclaiming the oracle limitation.

---

## Pillar 2: GENUINE DIFFICULTY AS A SOFTWARE ARTIFACT — 6/10

**The core engine is genuinely hard.** The BMC engine with incremental assumption-based clause management, CEGAR abstraction-refinement, bidirectional reachability for rollback envelope computation, interval-compressed SAT encoding, and treewidth-based DP — all deeply integrated — constitutes ~55K LoC of novel algorithmic systems code. This exceeds typical SOSP/OSDI artifacts.

**The 175K LoC claim is inflated by 35-60%.** Panel consensus:
- ~55K LoC: novel algorithmic core (BMC + CEGAR + envelope + encoding + solver integration) — full credit
- ~30K LoC: schema compatibility analysis across 4 formats — hard engineering, partially library-replaceable
- ~25K LoC: Kubernetes integration — necessary engineering, not novel
- ~25K LoC: benchmark/evaluation infrastructure (Python) — standard, not system difficulty
- ~15K LoC: diagnostics/output — standard
- ~15K LoC: testing — expected
- ~10K LoC: Helm Go-template reimplementation in Rust — unnecessary risk

**The Helm reimplementation is a correctness liability.** All three experts agree: reimplementing Go's `text/template` + Sprig in Rust creates an enormous surface for semantic divergence. Any difference between the reimplementation and real Helm silently corrupts constraint extraction, breaking the formal guarantee at step one. The alternative — `helm template` as a subprocess — is trivial, correct, and adequate for a research paper.

**Honest difficulty estimate: ~115K LoC** (stripping padding, using `helm template` subprocess, deferring Avro/GraphQL). Still substantial and impressive, and every line serves the core contribution.

**Theoretical contributions:** 2 genuine results (Theorem 2: monotone sufficiency via exchange argument; Theorem 5: FPT treewidth tractability), 1 useful encoding optimization (Theorem 3: interval compression), 3 formalizations of structural observations (Theorems 1, 4, 6). The "6 load-bearing theorems" framing oversells — "2 key theorems + 4 supporting results" is more honest and more credible.

---

## Pillar 3: BEST-PAPER POTENTIAL — 5.5/10

**The rollback safety envelope is the strongest novelty claim.** This is a new operational primitive with potential to become permanent vocabulary — like "linearizability" or "eventual consistency." The concept is genuinely new, operationally meaningful, and applicable beyond deployment planning (database migrations, feature flag rollouts, IaC state transitions). If the paper leads with this concept rather than the BMC formalism, the impact narrative strengthens significantly.

**The techniques are borrowed, not invented.** BMC (Clarke et al., 2001), CEGAR (Clarke et al., 2003), treewidth DP (standard parameterized complexity), interval encoding (standard SAT optimization), bidirectional reachability (standard model checking). The combination is novel and the domain-specific reductions are useful, but no individual technique advances the state of the art. A hostile reviewer could frame this as "McClurg et al. (PLDI 2015) for Kubernetes."

**Panel consensus: "application paper" is not damning at systems venues.** Ironfleet, Verdi, CertiKOS, and seL4 all applied known verification techniques to new domains and won best papers. But those papers had either (a) deeper theoretical contributions or (b) dramatically more compelling evaluations than what SafeStep currently proposes.

**The evaluation plan has critical weaknesses:**
- Incident reconstruction from postmortems requires information (exact version sets, schema definitions) that public postmortems don't contain. Reconstruction involves heavy inference, making it closer to synthetic than real-world validation.
- 15 incidents is too small for statistical significance.
- The dual success criterion (find safe plan OR identify point of no return) makes success easy to claim.
- No prospective deployment on a real cluster.

**What would push this to best-paper:**
- SafeStep discovers a previously-unknown unsafe rollback state in an existing benchmark (DeathStarBench, TrainTicket) — finding a real bug with a new tool is the ultimate validation.
- A treewidth-vs-performance phase transition curve showing the theoretical tractability result is empirically relevant.
- Honest, focused narrative: envelope concept + 2 key theorems + killer evaluation. Everything else is supplementary.

**The SDN comparison must be handled carefully.** The structural parallel to Reitblatt et al. (SIGCOMM 2012) and McClurg et al. (PLDI 2015) is deep. The differentiation — different constraint domain, rollback analysis is new — is real but requires careful framing to avoid "just a domain translation" dismissal.

---

## Pillar 4: LAPTOP CPU + NO HUMANS — 6.5/10

**SAT/BMC is feasible at stated parameters.** The interval-compressed encoding for n=50, L=20, k=200 produces clauses well within CaDiCaL's capacity. Modern CDCL solvers routinely handle 10M+ clauses on laptop hardware. Incremental solving amortizes clause learning across BMC depths.

**Internal inconsistency in clause count.** The body text claims O(n² · log L) total encoding. Theorem 3 states O(log|Vᵢ| · log|Vⱼ|) per service pair per step, yielding O(n² · log² L · k) total. The 2.2M clause figure uses the body text formula; the correct figure under Theorem 3 is ~9.3M clauses (a 4.3× discrepancy). Both are within solver capacity, but the inconsistency suggests the theoretical analysis hasn't been fully pressure-tested.

**Treewidth DP is infeasible at claimed parameters.** Theorem 5 gives O(n · L^{2(w+1)}):
- tw=3, L=10: O(50 × 10^8) = 5×10^9 — feasible (~5 seconds)
- tw=3, L=20: O(50 × 20^8) ≈ 1.3×10^12 — borderline (~20 minutes)
- tw=5, L=20: O(50 × 20^12) ≈ 2×10^17 — completely infeasible
- tw=5, L=10: O(50 × 10^12) = 5×10^13 — infeasible

The proposal claims "DP fast path completes in under 60 seconds" for "the common case" (treewidth 3–5). This is **false** for tw≥4 at L≥15. The treewidth DP is a viable fast path only for tw≤3, L≤15.

**No-humans constraint is satisfied.** The pipeline from Helm charts to verified plans is fully automated. Schema analysis, constraint extraction, SAT solving, and envelope computation require no human annotation. The binding constraint is oracle quality, not automation.

**Graceful degradation is unaddressed.** What happens when n>100 or treewidth>8? The proposal should include a performance tier table:
| Services | Treewidth | Expected Performance | Method |
|----------|-----------|---------------------|--------|
| ≤30      | any       | <60s               | SAT/BMC |
| 30-50    | ≤3        | <60s               | Treewidth DP fast path |
| 30-50    | >3        | 3-17 min            | SAT/BMC |
| 50-100   | any       | 10-60 min           | SAT/BMC, envelope may be partial |
| 100-200  | any       | hours               | SAT/BMC, best-effort |

---

## Pillar 5: FATAL FLAWS

### Flaw 1: Oracle Accuracy Unvalidated (SERIOUS — all 3 experts flag)

The entire value pyramid rests on the schema-derived compatibility oracle. No experiment quantifies what fraction of real deployment failures schema analysis catches. Without this data, the value proposition is unfalsifiable speculation. The "70% oracle" defense and the "power-law distribution" claim are unsubstantiated.

**Required fix:** Design and report an oracle validation experiment. Classify the 15 postmortem root causes as structural (schema-detectable) vs. behavioral (invisible to schema analysis). Report the fraction. If <40%, pivot to a theory paper about the envelope concept.

### Flaw 2: Treewidth DP Infeasible at Stated Parameters (SERIOUS — Auditor identified)

The narrative implies the treewidth DP is the "common case fast path" at treewidth 3-5 with L=20. The math contradicts this: O(50 × 20^12) ≈ 2×10^17 at tw=5. The treewidth DP is only feasible for tw≤3, L≤15.

**Required fix:** Add explicit feasibility boundaries. Present the DP as a fast path for low-treewidth cases only, with SAT/BMC as the primary algorithm.

### Flaw 3: Clause Count Internal Inconsistency (MODERATE — Auditor identified)

Body text claims O(n² · log L); Theorem 3 implies O(n² · log² L · k). The feasibility analysis uses the wrong (smaller) formula. Corrected figure (~9.3M clauses) is still feasible but the 4.3× discrepancy is sloppy.

**Required fix:** Reconcile the formulas. Use the correct figure in feasibility analysis.

### Flaw 4: Helm Reimplementation Correctness Risk (MODERATE — all 3 flag)

Reimplementing Go templates + Sprig in Rust creates an enormous surface for divergence from real Helm behavior. Any difference silently invalidates constraint extraction.

**Required fix:** Use `helm template` as a subprocess. Note Rust reimplementation as future work for production performance.

### Flaw 5: Unverified Empirical Claims (MODERATE — all 3 flag)

Three load-bearing claims are asserted without methodology:
- ">92% of pairwise compatibility relations have interval structure" (from "847 microservice dependency graphs")
- "18-23% of multi-service outages involve version incompatibility" (from "200+ postmortems")
- "median treewidth 3-5 in production clusters"

**Required fix:** Describe the dataset and counting methodology for each claim. Ideally, make datasets available.

### Flaw 6: Model-Reality Gap (LOW-MODERATE — Skeptic identified)

The formal model assumes atomic sequential upgrades, binary compatibility, and deterministic state transitions. Real Kubernetes deployments are concurrent, asynchronous, and subject to race conditions, partial failures, and pod eviction.

**Required fix:** Explicitly discuss the gap between the sequential model and concurrent reality. Argue that the sequential model is a conservative overapproximation (if a concurrent execution is safe, any sequential interleaving is also safe) — or acknowledge the limitation.

### Flaw 7: Misleading Aeolus/Zephyrus Comparison (LOW — Auditor identified)

Comparing PSPACE (general, different problem) vs. NP (restricted, SafeStep's problem) is apples-to-oranges.

**Required fix:** Add explicit caveats about the restriction that enables tractability.

---

## Required Amendments (Mandatory — All 7)

1. **Oracle validation experiment.** Classify 15 postmortem root causes as structural vs. behavioral. Report coverage fraction.
2. **Fix clause count inconsistency.** Use correct O(n² · log² L · k) formula. Recompute feasibility.
3. **Treewidth DP feasibility boundary.** Explicit table: DP fast path for tw≤3 only. SAT/BMC is primary.
4. **Use `helm template` subprocess.** Drop Rust reimplementation for the paper.
5. **Qualify "formally verified" language.** Use "structurally verified" or "verified relative to modeled constraints."
6. **Provide methodology for empirical claims.** 92% interval structure, 18-23% outage rate, treewidth 3-5.
7. **Fix Aeolus/Zephyrus comparison.** Explicit caveat about restriction-dependent tractability.

## Recommended Improvements (Non-mandatory)

1. Narrow to OpenAPI + Protobuf for the paper. Defer Avro, GraphQL.
2. Lead with the envelope concept, not the BMC formalism.
3. Add oracle confidence coloring (green/yellow/red by constraint provenance).
4. Include warm-start and constraint caching for incremental deployments.
5. Add incremental re-planning capability discussion.
6. Discuss compliance/audit value explicitly.
7. Present theorems honestly: "2 key theorems + 4 supporting results."
8. Reduce LoC claims to honest core (~115K) with transparent breakdown.

---

## Probability Estimates

| Outcome | Probability |
|---------|-------------|
| P(publishable at SOSP/OSDI \| amendments applied) | 35-40% |
| P(publishable at EuroSys/NSDI/ICSE \| amendments applied) | 60-65% |
| P(best-paper at any top venue) | 8-12% |
| P(project delivers on core claims) | 50-55% |
| P(abandon as fundamentally flawed) | 10% |

## Panel Sign-off

| Expert | Verdict | Score | Key Condition |
|--------|---------|-------|---------------|
| Independent Auditor | CONDITIONAL CONTINUE | V6/D6/BP5/L6 | Oracle validation + clause count fix |
| Fail-Fast Skeptic | CONDITIONAL CONTINUE (reluctant) | V4→6/D5→6/BP5/L6 | Oracle must show ≥60% structural coverage, else ABANDON |
| Scavenging Synthesizer | CONTINUE | V8/D7/BP6.5/L7.5 | Lead with envelope concept, focus narrative |
| **Consensus** | **CONDITIONAL CONTINUE** | **V6/D6/BP5.5/L6.5** | **7 mandatory amendments; oracle validation is the gate** |

---

*Assessment produced by 3-expert adversarial verification panel. Independent proposals (no cross-contamination) → adversarial cross-critique with direct challenges → synthesis of strongest elements. All scores justified by specific evidence and explicit challenge/response exchanges.*
