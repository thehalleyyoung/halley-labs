# Community Expert Verification: SafeStep (proposal_00)

**Project:** SafeStep — Verified Deployment Plans with Rollback Safety Envelopes for Multi-Service Clusters  
**Slug:** `safe-deploy-planner`  
**Stage:** Verification (post-theory, pre-implementation)  
**Evaluator:** Community Expert Panel (Distributed Systems & Cloud Infrastructure)  
**Method:** Three independent experts (Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer) → adversarial cross-critique → synthesis → independent verification signoff  
**Prior Scores:** Ideation V6/D6/BP5.5/L6.5 · Theory V7/D7/BP6/L7.5/F7  
**Date:** 2026-03-08

---

## Executive Summary

**Verdict: CONDITIONAL CONTINUE**  
**Composite: 6.1/10** (V6/D6/BP5.5/L7/F6)  
**Kill probability: 20–28%** (primarily oracle coverage failure)  
**P(strong publication at EuroSys/NSDI/ICSE): 50–60%**  
**P(best paper at any top venue): 8–12%**  

The rollback safety envelope is a genuinely novel operational concept — no existing tool, academic or industrial, computes bidirectional reachability under safety invariants for deployment states. This single idea justifies continued development. However, the project's practical value chain is entirely conditional on an unexecuted oracle validation experiment, the load-bearing theorem's key assumption (bilateral DC) is unquantified under the correct definition, and all core techniques are borrowed. Four binding conditions must be resolved before any implementation begins. Phase 0 (oracle validation + bilateral DC quantification) is the binary gate that determines whether this becomes a practical deployment tool or a formal-methods theory paper.

---

## Panel Composition and Process

| Role | Bias | Composite | Verdict |
|------|------|-----------|---------|
| Independent Auditor | Evidence-only, no benefit of doubt | 6.3/10 | CONDITIONAL CONTINUE |
| Fail-Fast Skeptic | Aggressive rejection of unsupported claims | 5.0/10 | CONDITIONAL CONTINUE (barely) |
| Scavenging Synthesizer | Find maximum salvageable value | 7.2/10 | CONTINUE |
| **Cross-Critique Consensus** | **Evidence-weighted synthesis** | **6.1/10** | **CONDITIONAL CONTINUE** |

Process: Independent proposals → adversarial cross-critique with explicit disagreement resolution → synthesis of strongest elements → independent verification signoff (APPROVED WITH CHANGES — changes incorporated below).

---

## Pillar 1: Extreme Value — 6/10

**The problem is real and well-motivated.** Cross-service version incompatibility during rolling updates causes catastrophic outages — Google Cloud SQL 2022, Cloudflare 2023, AWS Kinesis 2020. The 18–23% outage rate claim (problem_statement.md:116) from 214 postmortems with Cohen's κ > 0.7 is the strongest empirical grounding in the proposal. The combinatorial explosion of intermediate states (5³⁰ ≈ 9.3 × 10²⁰ for a 30-service cluster) defeats human reasoning. No existing tool answers "can I safely roll back from this specific intermediate state?"

**The rollback safety envelope fills a structural gap no existing tool addresses.** This gap is *structurally invisible* to every tool in the deployment pipeline — Kubernetes health checks, canary deployments, service meshes, ArgoCD/Flux all operate on different failure classes. SafeStep's "complement to canaries" positioning is genuinely orthogonal (backward failures vs. forward failures), meaning adoption is additive rather than competitive. This is rare and valuable.

**Value is conditional on the unvalidated oracle.** The entire formal guarantee chain passes through a schema-derived compatibility oracle that has been tested against zero real incidents. The "70% oracle catches the majority of failures" defense (problem_statement.md:114) has zero empirical backing — no citation, no data, no methodology. The power-law distribution claim is asserted without evidence.

**Panel disagreement (4/6.5/7.5 — resolved to 6).** The Skeptic argues the oracle gap makes all value claims speculative and raises a "false confidence" concern — operators who trust "structurally verified" plans may lower their guard on manual checks. The Synthesizer argues the compliance angle (SOC2/HIPAA/PCI-DSS require documented change management) provides an oracle-independent value floor. The Auditor takes the middle: designed-but-unexecuted validation warrants scoring expected value over Phase 0 outcomes, not the best or worst case. The false-confidence critique is mitigated by the confidence coloring system (GREEN/YELLOW/RED) that makes oracle limitations visible — this is strictly better than the status quo of zero analysis. However, two of the three motivating outages (Kinesis, Cloud SQL) involved behavioral/operational failures outside schema analysis scope, weakening the narrative.

**What would elevate this:** Oracle validation showing ≥60% structural coverage; evidence that behavioral failures are rare in multi-service deployment incidents; explicit compliance/audit value framing in the paper.

---

## Pillar 2: Genuine Software Difficulty — 6/10

**LoC estimates have been corrected — twice.** The trajectory (175K → 115K → ~60K) reveals honest self-correction but undermines confidence in initial difficulty assessments. The final ~60K LoC estimate (final_approach.md:125–127) with 9 subsystems is credible.

**Three subsystems are genuinely research-grade:**
1. Incremental BMC with bidirectional reachability for envelope computation (~12–18K LoC) — no prior work does this for deployment states
2. Interval-compressed SAT encoding with CEGAR integration across solver boundaries (~8–12K LoC) — correctness depends on subtle invariants
3. Multi-format schema compatibility oracle across OpenAPI + Protobuf (~8–12K LoC) — no unified library exists

**All core techniques are borrowed.** BMC (Clarke 2001), CEGAR (Clarke 2003), treewidth DP (Bodlaender 1993), interval encoding (standard SAT), UNSAT core extraction (standard). The difficulty is in the *composition and domain-specific adaptation*, not in algorithmic novelty. The Skeptic characterizes this as "competent systems engineering with a formal verification veneer"; the Synthesizer counters that integrating five non-trivial techniques into a coherent system with formal guarantees is harder than it looks.

**The bilateral DC discovery is a genuine mathematical contribution.** The mid-proof correction from unilateral to bilateral downward closure (verification_signoff.md:20–27) shows the proof was actually constructed, not just asserted. The exchange argument for Theorem 1 is non-trivial. For a systems paper, this level of mathematical engagement is notable.

**Panel consensus: "2 key theorems + 4 supporting results"** is more honest than "6 load-bearing theorems." Theorem 1 (monotone sufficiency) and the rollback safety envelope concept are the contributions; everything else is engineering machinery.

---

## Pillar 3: Best-Paper Potential — 5.5/10

**The rollback safety envelope has "permanent vocabulary" potential.** Like "linearizability" (Herlihy & Wing 1990) or "eventual consistency" (Vogels 2009), the concept names and formalizes something practitioners need but couldn't previously articulate: "from this specific intermediate state, is rollback to a known-good state reachable through safe intermediate states?" This is the strongest novelty claim. (Note: the "permanent vocabulary" claim is unfalsifiable at submission time — it is a framing argument, not verifiable evidence.)

**The "McClurg et al. (PLDI 2015) for Kubernetes" attack is the most dangerous.** The structural parallel to SDN consistent-update literature is deep: BMC over a product graph with safety constraints and incremental SAT solving. The differentiation — rollback envelope is new, interval encoding exploits domain-specific structure, oracle confidence model is qualitatively different — is real but narrow. This is rated 7/10 attack strength by the theory verification signoff (line 314). The paper must handle this comparison carefully.

**Systems venues DO award best papers for concept novelty with borrowed techniques** — Ironfleet, Verdi, CertiKOS all applied known verification techniques to new domains. But they had either deeper theoretical contributions or dramatically more compelling evaluations. SafeStep has neither yet.

**The evaluation plan has critical strengths and weaknesses:**
- *Strengths:* Phase 0 oracle gate with honest go/no-go criteria; prospective evaluation on DeathStarBench; ablation study; honest separate reporting of "found safe plan" vs. "flagged PNR" vs. "missed entirely"
- *Weaknesses:* 15 postmortems gives ±12% CI (statistically indistinguishable from 48% at the 95% level if oracle coverage is 60%); incident reconstruction uses heavy inference; the "find a real bug in DeathStarBench" aspiration is entirely unvalidated

**The killer evaluation result — finding a previously-unknown unsafe rollback state in DeathStarBench — would transform the paper.** This is the difference between 5.5 and 8. All three experts agree this single result would make the paper. Without it, the evaluation is synthetic data plus reconstructed incidents.

**Best-paper probability: 8–12% at EuroSys/NSDI** contingent on a compelling evaluation result; ~3% without one.

---

## Pillar 4: Laptop-CPU Feasibility & No-Humans — 7/10

**SAT solving at 14.4M clauses is comfortable.** The corrected encoding estimate (verification_signoff.md:52) uses the correct O(n² · log²L · k) formula, including the non-interval fraction (f ≈ 0.08, contributing ~41% of total clauses). CaDiCaL routinely handles 50M+ clauses — a 3.5× margin. Incremental solving with assumption literals amortizes clause learning across BMC depths.

**The clause count discrepancy is resolved.** The 4.3× discrepancy (2.2M vs 9.3M) flagged in the ideation depth check has been corrected. Both artifacts now consistently use the correct formula. The corrected 14.4M figure includes the non-interval tail.

**Treewidth DP is honestly scoped.** After the depth-check correction:
- tw ≤ 3, L ≤ 15: seconds (DP fast path)
- tw = 4, L ≤ 8: minutes (marginal)
- tw ≥ 5: infeasible → SAT/BMC only

This directly addresses the false claim that tw=3–5 was the "common case fast path" at L=20.

**Envelope computation via binary search: ~8 incremental SAT calls** if the prefix property holds. If not (the property is unproven — verification_signoff.md:238–248), linear scan fallback requires ~200 calls — a 25× slowdown pushing envelope computation to 20–200 seconds. Still feasible but significant.

**Total wall-clock at n=50: 3–17 minutes.** Schema analysis (30s) + manifest parsing (60s) + encoding (5s) + SAT (30–300s) + envelope (60–600s). At n=100+, estimated hours — excluding the real-time 3:47 AM use case. The 3:47 AM narrative creates a tension with the 3-minute minimum runtime that the paper should address (reframe as pre-deployment planning rather than real-time incident response).

**No-humans constraint is fully satisfied.** The pipeline from Helm charts to verified plans is fully automated. No human annotation at any stage.

---

## Pillar 5: Feasibility — 6/10

**The theory-to-implementation mapping is explicit.** Each theorem maps to a named Rust module with LoC estimates (final_approach.md:114–125). Technology choices are sound: CaDiCaL via FFI for SAT, Z3 for LIA, `helm template` subprocess for Helm (avoiding the correctness liability of Go template reimplementation). The honest LoC estimate of ~60K is achievable.

**Three binding conditions from the theory signoff remain unresolved:**

1. **Bilateral DC prevalence quantification** (SERIOUS). The >92% figure was measured under unilateral DC; bilateral is strictly stronger. If bilateral DC prevalence is 75% instead of 92%, the monotone reduction applies to fewer pairs, and the mixed-case search space is *uncharacterized*. The Skeptic correctly notes: when bilateral DC holds for fraction p, the combined search space is neither fully monotone (tight bound) nor fully general (PSPACE) — it is a hybrid whose complexity is unanalyzed. No theorem, no bound, no estimate exists for the mixed case.

2. **Envelope prefix property formal proof** (MODERATE). The binary search optimization assumes the envelope is a prefix of the plan. The verification chair's analysis (signoff.md:238–248) shows the required argument is non-trivial — the naive direction goes the wrong way. If false, binary search silently misclassifies PNR states (a correctness issue, not just performance). Linear scan is the valid fallback.

3. **847-dataset methodology** (MODERATE). Three load-bearing empirical claims (>92% interval structure, 18–23% outage rate, median treewidth 3–5) reference a dataset with no documented construction methodology. If this dataset doesn't exist or was constructed opportunistically, three key theoretical claims are unsupported.

**Oracle accuracy is the existential risk.** Every artifact flags this. The Phase 0 experiment resolves it but hasn't been executed. Kill probability: 20–28% (if the Skeptic's oracle coverage prior of 45–55% is accurate, kill probability is closer to 25–28%).

**The `theory_bytes=0` anomaly.** State.json records theory_bytes=0 despite theory_complete status. The theory stage produced no standalone theory artifact — all theory lives in approach.json and paper.tex from ideation. The actual artifacts (2030-line paper, 50KB approach.json, verification signoff) constitute substantial work regardless of the metadata counter. This is a bookkeeping artifact, not a substantive concern — but it means the theory scores (V7/D7/BP6/L7.5/F7) may reflect ideation quality, not genuine theory-stage advancement.

---

## Fatal Flaws (Ranked by Severity)

### 1. Oracle Coverage Unvalidated — SERIOUS (P(kill) = 20–28%)

Every formal guarantee, every safety claim, every "structurally verified" assertion passes through a compatibility oracle that has been tested against zero real incidents. Schema analysis catches structural breaks (field removals, type changes) but is blind to: changed semantics, performance regressions, timeout changes, retry policy changes, connection pool exhaustion, authentication format changes. The "70% oracle catches the majority" claim is unsubstantiated. The Phase 0 experiment resolves this but hasn't been executed after two pipeline stages.

**Severity: SERIOUS.** Binary outcome — proceed or pivot. Cannot be engineered around.

### 2. Bilateral DC Prevalence Unquantified — SERIOUS (P(significant degradation) = 15–20%)

Theorem 1 (monotone sufficiency) — the most important theorem — requires bilateral downward closure. The >92% figure was measured under unilateral DC. Bilateral DC is strictly stronger (both arguments must be downward-closed). If bilateral DC prevalence is ≤75%, the monotone reduction applies to at most 75% of service pairs. The remaining 25% require general non-monotone search. The combined search space is a hybrid whose complexity is **unanalyzed**. The proposal provides no theorem, no bound, no estimate for the mixed case.

**Severity: SERIOUS.** Directly affects the load-bearing theorem's practical applicability.

### 3. 847-Dataset Methodology Absent — MODERATE (P(claims weakened) = 40%)

Three load-bearing empirical claims reference "847 open-source microservice projects" with no documented selection criteria, extraction procedure, or verification method. Unsubstantiated claims in a paper positioning itself on formal rigor are particularly damaging to credibility. Any reviewer at SOSP/OSDI will flag this immediately.

**Severity: MODERATE.** Documentation work, not research. Must be fixed before submission.

### 4. Envelope Prefix Property Unproven — MODERATE (P(correctness fix) = 10%)

The binary search optimization for PNR detection relies on an unproven claim. The verification chair's analysis suggests the argument is "likely provable but not trivially obvious." If false, binary search silently misclassifies PNR states — a correctness issue, not just performance.

**Severity: MODERATE.** Likely correct (no counterexample found), but must be formally proven.

### 5. Evaluation Statistically Weak — MODERATE

15 postmortems with ±12% CI. The dual success criterion (find safe plan OR identify PNR) inflates success rates. No prospective deployment on a real cluster. The "find a real bug in DeathStarBench" aspiration is unvalidated. The sample is biased (public postmortems overrepresent spectacular failures).

**Severity: MODERATE.** Addressable by expanding sample size and designing for the killer DeathStarBench result.

### 6. Theorem 5 Probabilistic Guarantee is Decorative — LOW

The theoretical budget k_α = 10 requires C(60,10) subset enumerations — computationally impossible. The practical system defaults to k=2 with 45 enumerations. The gap between theory (k=10) and practice (k=2) is large. The independence assumption is unrealistic. The k=2 brute-force check has standalone practical value but no principled probabilistic interpretation.

**Severity: LOW.** Honest acknowledgment mitigates.

---

## Skeptic Challenges — Resolutions

### "False confidence is worse than no tool"
**Resolution: Counter-argument wins, with qualification.** The critique proves too much — it would condemn all static analysis tools. The confidence coloring system makes uncertainty visible, which is strictly better than the status quo. However, the 3:47 AM narrative implicitly promises more than the system delivers. The paper must frame SafeStep as a complement to manual analysis, not a replacement.

### "Scale ceiling at n=50 excludes target audience"
**Resolution: Counter-argument mostly wins.** The relevant metric is tightly-coupled services with non-trivial compatibility constraints, not total services. For a 200-service cluster, the compatibility-critical subset is typically 30–80 services. However, the paper should benchmark n=100 and characterize the scaling curve rather than leaving it to extrapolation.

### "Schema-aware ordering + canary delivers 80% of value at 10% complexity"
**Resolution: Skeptic is wrong on percentage, right on spirit.** Schema-aware ordering is a strict subset of SafeStep's output. Canaries are structurally blind to rollback failures — SafeStep's core value proposition. But the challenge forces a key question: what fraction of real failures are rollback-specific vs. forward-specific? Phase 0 must quantify this.

### "theory_bytes=0 means no real theory"
**Resolution: Bookkeeping artifact, not substantive concern.** The actual artifacts (2030-line paper, 50KB approach.json, verification signoff with challenge tests) constitute substantial theory work regardless of metadata.

### "Mixed-case complexity is uncharacterized"
**Resolution: Skeptic identifies a real gap.** When bilateral DC holds for fraction p, the search space interaction between monotone and non-monotone subgraphs is unanalyzed. This must be characterized before claiming tractability for realistic clusters.

---

## Comparison to Prior Stages

| Dimension | Ideation | Theory | Verification | Delta (Theory→Verif) | Commentary |
|-----------|----------|--------|-------------|---------------------|------------|
| V (Value) | 6 | 7 | **6** | −1 | Oracle risk elevated; two motivating examples misfit; compliance angle underweighted in prior stages |
| D (Difficulty) | 6 | 7 | **6** | −1 | Borrowed techniques penalty applied more rigorously; LoC deflation history undermines initial claims |
| BP (Best Paper) | 5.5 | 6 | **5.5** | −0.5 | "McClurg for K8s" attack weighted; permanent vocabulary claim acknowledged as unfalsifiable |
| L (Laptop) | 6.5 | 7.5 | **7** | −0.5 | Envelope prefix gap; 3:47 AM vs. 3-min runtime tension; n=200 out of scope but real |
| F (Feasibility) | — | 7 | **6** | −1 | Three binding conditions still unresolved; theory_bytes=0 context |

**Trajectory: Downward correction from theory-stage optimism.** The theory stage scores (V7/D7/BP6/L7.5/F7) are ~1 point higher across most pillars. The verification stage applies more rigorous scrutiny to unvalidated claims and correctly penalizes designed-but-unexecuted experiments. The L pillar drops least because feasibility evidence is the most quantitative.

---

## Binding Conditions

### MUST-DO (Blocking — before ANY implementation begins)

**C1. Execute Phase 0 oracle validation.** Classify root causes in ≥15 published postmortems with ≥2 independent raters and Cohen's κ ≥ 0.7. Structural-detectable coverage thresholds:
- ≥60%: **CONTINUE** to full implementation
- 40–59%: **CONTINUE** with paper repositioned as "formal framework" (not "practical tool")
- <40%: **ABANDON** tool implementation; pivot to workshop/short paper on envelope concept

**C2. Empirically quantify bilateral DC prevalence** on real compatibility data under the bilateral definition.
- ≥80%: **CONTINUE** with full monotone reduction claims
- 70–79%: **CONTINUE** only if a formal mixed-case complexity bound shows solve time remains within 5× of the pure-monotone case for the observed non-DC fraction
- <70%: **REASSESS** — the key tractability theorem applies to a minority of service pairs

**C3. Document the 847-dataset methodology** or retract the empirical claims. Selection criteria, compatibility predicate extraction procedure, interval structure verification method, and treewidth computation details must be reproducible.

**C4. Formally prove or disprove the envelope prefix property.** If disproved: implement linear scan fallback and revise envelope computation time estimates. This is a correctness requirement.

### SHOULD-DO (Non-blocking but important)

**C5.** Include n=100 benchmark in Phase 2 to characterize scaling beyond the n=50 comfort zone.  
**C6.** Reframe the paper narrative to position SafeStep as a *complement* to canary deployments, not a replacement for manual rollback analysis. Address the 3:47 AM vs. 3-minute runtime tension.  
**C7.** Strengthen Theorem 5 discussion to explicitly separate the theoretical bound (k=10, impractical) from the practical check (k=2, useful).  
**C8.** Before Phase 0 begins, identify target venue and submission deadline, and confirm the Phase 0 → implementation → evaluation timeline is feasible.  
**C9.** Conduct a brief competitive landscape survey confirming no concurrent work on formal deployment plan synthesis.

---

## Probability Estimates

| Outcome | Probability |
|---------|-------------|
| P(project killed by Phase 0) | 20–28% |
| P(significant scope reduction from bilateral DC or mixed-case issues) | 15–20% |
| P(not reaching claimed contribution level) | ~33% |
| P(publishable at SOSP/OSDI \| conditions met) | 30–35% |
| P(publishable at EuroSys/NSDI/ICSE \| conditions met) | 50–60% |
| P(best paper at any top venue) | 8–12% |
| P(project delivers working system + paper) | 45–55% |

---

## Strongest Elements (must be preserved in any version)

1. **The rollback safety envelope concept.** Even if every line of code is discarded, this concept has lasting value. It reframes deployment safety from binary (safe/unsafe) to a reachability question with geometric structure.

2. **Theorem 1: Monotone Sufficiency under bilateral DC.** The exchange argument proof is non-trivial and the bilateral DC discovery shows genuine mathematical engagement.

3. **The "complement to canaries" positioning.** Orthogonal positioning (backward vs. forward failures) is rare and valuable. It means SafeStep doesn't compete with any existing tool.

4. **The 3:47 AM opening narrative.** A concrete, emotionally resonant scenario that makes the abstract problem visceral. Every reviewer who has read it calls it "compelling."

5. **The honest framing.** "Structurally verified relative to modeled API contracts" with confidence coloring converts a weakness (incomplete oracle) into a strength (visible uncertainty).

---

## Red-Team Attack Vectors

| Attack | Strength | Defense | Verdict |
|--------|----------|---------|---------|
| "McClurg et al. for Kubernetes" | 7/10 | Envelope is new; interval encoding is domain-specific; oracle model is qualitatively different | Dangerous — requires careful SDN comparison in §7 |
| "Guarantees as weak as the oracle" | 6/10 | Every static analysis tool has this; confidence coloring; honest framing | Valid but addressed by framing |
| "15 postmortems is meaningless" | 5/10 | Phase 0 is a pilot; prospective eval on DeathStarBench is primary | Valid methodological concern, not fatal |
| "Bilateral DC gap undermines formalization" | 6/10 | Honest disclosure; bilateral DC quantification in Phase 0; graceful degradation | Legitimate; mitigated by planned quantification |
| "Evaluation baselines are straw men" | 4/10 | Envelope has no baseline by definition; topological sort IS industry practice | Pedantic; novelty makes baselines impossible |
| "Simpler alternative delivers 80%" | 5/10 | Schema ordering is subset of SafeStep output; canaries miss rollback failures | Skeptic wrong on %, right on spirit — Phase 0 resolves |

---

## Panel Sign-off

| Expert | Verdict | Composite | Key Condition |
|--------|---------|-----------|---------------|
| Independent Auditor | CONDITIONAL CONTINUE | 6.3 | Phase 0 oracle gate + bilateral DC quantification |
| Fail-Fast Skeptic | CONDITIONAL CONTINUE (barely) | 5.0 | All 5 mandatory conditions; Phase 0 is existential gate |
| Scavenging Synthesizer | CONTINUE | 7.2 | Phase 0 + invest in DeathStarBench bug-finding result |
| Independent Verifier | APPROVED WITH CHANGES | — | Kill probability range 20–28%; elevate mixed-case analysis |
| **Consensus** | **CONDITIONAL CONTINUE** | **6.1** | **4 blocking conditions; Phase 0 determines project fate** |

---

## Salvage Analysis

| Failure Mode | What Survives | Salvage Value | Fallback Venue |
|-------------|---------------|---------------|----------------|
| Oracle coverage <40% | Envelope concept + theorems + prototype | 60% | POPL/VMCAI (theory paper) |
| Bilateral DC <75% | Everything except Theorem 1's broad applicability | 80% | Honest scoping, slightly weaker paper |
| SAT doesn't scale past 50 | Full framework for ≤50 services | 85% | Covers modal production case |
| DeathStarBench yields nothing | Synthetic eval + postmortem validation | 70% | Borderline accept territory |
| All fail simultaneously | Envelope concept as workshop paper | 40% | SOSP HotOS / SysDW |

---

## Final Assessment

SafeStep is a well-conceived project with a genuinely novel core concept (the rollback safety envelope) wrapped in competent but borrowed formal techniques, resting on an unvalidated oracle that determines whether the system is a practical tool or a theoretical curiosity. The project has been ruthlessly honest about its limitations through two pipeline stages — the confidence coloring, the "structurally verified" language, the explicit oracle gate, the LoC self-correction — which is a strong positive signal.

The project lives or dies on Phase 0. Run the oracle experiment. Everything else is premature optimization on an unvalidated premise. If oracle coverage ≥60% and bilateral DC ≥80%, this is on track for a solid EuroSys/NSDI paper with outside-shot best-paper potential if the evaluation delivers a killer DeathStarBench result. If either condition fails, the pivot plan to a theory paper preserves partial value.

**No Rust code should be written until all four blocking conditions are resolved.** Phase 0 is estimated at 2–3 weeks of focused effort — a modest investment before committing to 45–65K LoC of implementation.

---

*Evaluation produced by a three-expert panel (Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer) with adversarial cross-critique and independent verification signoff. Consensus scores reflect the strongest argument on each dimension, not arithmetic averages. All claims fact-checked against source artifacts.*
