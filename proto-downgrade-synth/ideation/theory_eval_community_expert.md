# Community Expert Verification Evaluation: NegSynth (proto-downgrade-synth)

**Evaluator Role:** Community Expert — Security, Privacy, and Cryptography  
**Evaluation Type:** Cross-critique synthesis of three independent expert analyses  
**Proposal:** "Negotiation Under Fire: Bounded-Complete Synthesis of Protocol Downgrade Attacks from Library Source Code"  
**Slug:** `proto-downgrade-synth` / proposal_00  
**Date:** 2026-03-08  
**Inputs:** Auditor analysis (evidence-based), Skeptic analysis (fail-fast), Synthesizer analysis (salvage-value)  
**State:** `theory_complete`, theory_bytes=0 (measurement bug; approach.json ≈ 50KB is the theory artifact)

---

## Executive Summary

Three independent expert analyses converge on a **conditional continue** verdict with composite scores in the 5.5–6.2 range against a self-assessment of 7.0. The proposal describes a genuinely novel pipeline — protocol-aware symbolic execution with algebraic merge, composed end-to-end from C source to bounded-completeness certificates — that would be a legitimate contribution to the security verification literature if realized. However, the full vision faces compounding risks (19% probability of full delivery) and the core technical claims require honest reframing: the O(n) bound applies only to an idealized subroutine, the certificates depend on an unproved slicer assumption (A0), and Z3 tractability on production-scale SMT encodings is an unvalidated bet.

No single flaw is independently fatal, but three flaws in concert create a fragile execution path. The critical question is not whether NegSynth *should* be built — it should — but whether the team can hit a publishable milestone before the kill gates close. The answer is yes, but only if the team plans for the MVP from day one and targets CCS/USENIX Security rather than S&P as the primary venue.

**Final Composite Score: 6.0/10**  
**Final Verdict: CONDITIONAL CONTINUE**

---

## 1. Cross-Critique: Disagreement Resolution

### Disagreement 1: VALUE — Auditor (6/10) vs. Synthesizer (7/10)

**Resolution: 6.5/10 — Synthesizer is closer, but Auditor's concerns are partially valid.**

The Auditor argues that declining CVE rates and TLS 1.3's anti-downgrade sentinel narrow the problem's relevance. The Skeptic *refuted* this empirically with five CVEs from 2023–2025: wolfSSL CVE-2024-5814, CVE-2025-11934, the uTLS bypass, .NET SMTP downgrade, and Terrapin (CVE-2023-48795). This evidence is decisive — the "problem is solved" narrative is factually wrong. Protocol downgrade vulnerabilities continue to appear in modern, audited code across multiple implementations and protocol families.

However, the Auditor is correct that the *stakeholder pool* is narrow: library maintainers at 4–6 major projects, IETF working group members, and a small cadre of protocol security auditors. This is not a mass-market tool. The Synthesizer's score of 7 slightly overstates excitement by conflating salvage-component reuse value (which is internal) with community value (which is external). The security community will find this *interesting and useful* but not *transformative* — it fills a real gap but serves a small audience.

**Who is right:** Skeptic provides the strongest evidence (recent CVEs demolish the drought narrative); Synthesizer correctly identifies deep-but-narrow value; Auditor overstates the decline but correctly identifies the small stakeholder pool. Blended score: **6.5/10**.

---

### Disagreement 2: FEASIBILITY — Auditor (6/10) vs. Synthesizer (5/10)

**Resolution: 5.5/10 — Synthesizer is closer; Auditor underweights the kill-gate cascade.**

The Synthesizer identifies three kill gates in the first 10 weeks: (1) slicer must produce ≤15K-line slices on OpenSSL, (2) merge operator must demonstrate measurable path reduction on real code, (3) Z3 must return SAT/UNSAT (not TIMEOUT) on at least one CVE encoding within the CEGAR loop. Any one of these failing terminates the full pipeline. The compound probability of clearing all three is the critical constraint.

The Auditor's 6/10 is generous because it weights the 55% minimal-viable probability without adequately discounting the serial gate structure. If gates are even weakly correlated (they share the same codebase complexity driver), the compound probability drops faster than independent multiplication suggests.

The Skeptic's publication probability (55–65%) is more pessimistic than the Synthesizer's (75–85%). The truth is between them: the salvage components (DY+SMT encoding, merge operator formalization) are independently publishable at ACSAC/ESORICS tier, which lifts the floor, but the composition into a top-4 venue paper requires clearing all three gates.

**Who is right:** Synthesizer's 5/10 for full feasibility is the most honest assessment. The 55% minimal-deliverable probability is credible. Auditor's 6/10 is optimistic. Blended score: **5.5/10**.

---

### Disagreement 3: BEST-PAPER PROBABILITY

| Source | Estimate |
|--------|----------|
| Self-assessment | 5–10% |
| Auditor | 3–5% |
| Skeptic | 1–3% |
| Synthesizer | 3–7% |

**Resolution: 2–4% — Skeptic and Auditor are closest.**

Best-paper awards at S&P/USENIX/CCS require either a paradigm shift (new attack class, new defense primitive) or extraordinary empirical depth (new dataset, new measurement at scale). NegSynth offers neither: it is a careful *composition* of known techniques (symbolic execution, DY modeling, bisimulation, SMT solving) applied to a specific domain. The core theorem T3 (merge correctness) scores 5/10 on mathematical depth — it is a domain instantiation of known congruence-closure techniques, not a new proof technique. The O(n) claim must be reframed, weakening the "money plot" narrative. The certificates are caveated by Assumption A0.

A best paper would require: (1) finding a new CVE in a current library HEAD, (2) the O(n) bound holding on production code without fallback, and (3) certificates with no caveats. The probability of all three is very low.

The self-assessment of 5–10% is aspirational; it conflates "best paper if everything works perfectly" with realistic probability. The Skeptic's 1–3% is slightly too harsh (the composition *is* novel and a new CVE discovery would be compelling). The Auditor's 3–5% with the Synthesizer's 3–7% bracket the truth.

**Final estimate: 2–4%.**

---

### Disagreement 4: P(TOP-4 VENUE ACCEPTANCE)

| Source | Full Vision | MVP/Minimal |
|--------|------------|-------------|
| Self-assessment | 45–55% | — |
| Auditor | 35–45% | — |
| Skeptic | 20–30% | 35–45% |
| Synthesizer | 15–20% | 35–45% |

**Resolution: Full vision 20–30%, MVP 35–45%.**

The full vision (4 libraries, 8 CVEs, bounded-completeness certificates, differential extension) is a maximalist paper that requires every pipeline stage to work. At S&P, reviewers will interrogate the A0 assumption, the O(n) reframing, and the practical utility of bounded certificates. The self-assessment of 45–55% assumes reviewers accept the framing at face value; the Skeptic and Synthesizer correctly predict skepticism.

The MVP (2 libraries, 4–5 CVEs, merge operator + attack synthesis without full certificates) is a more realistic submission that avoids the certificate caveats entirely. This is a cleaner story: "we found known CVEs automatically from source code using a novel merge operator." The convergence at 35–45% across Auditor, Skeptic, and Synthesizer for the MVP is credible.

The Synthesizer's 15–20% for the full vision is too pessimistic — it double-counts risks already priced into the caveated certificate framing. The Skeptic's 20–30% is the right range: achievable but far from guaranteed.

**Who is right:** Skeptic's ranges are the most calibrated. Full vision: **20–30%**. MVP: **35–45%**. Target CCS or USENIX Security, not S&P.

---

## 2. Strongest Points from Each Expert

### Auditor — Top 3 Contributions
1. **Fatal flaw decomposition with probability ranges.** The five-flaw analysis (F1–F5) with explicit probability bands (25–70%) and severity ratings is the most actionable risk framework produced. The finding that *no single flaw is independently fatal* while the *compound* of F1+F2 creates a 40–55% joint risk is the key strategic insight.
2. **~36K genuinely hard LoC estimate.** By subtracting protocol modules (human-authored, not algorithmically novel) and test infrastructure, the Auditor identifies the true intellectual core: slicer + merge operator + DY encoder + CEGAR loop ≈ 36K novel lines. This is a more honest difficulty assessment than the headline 50K.
3. **CCS-tier as realistic target.** The explicit recommendation to recalibrate from S&P to CCS/USENIX is pragmatically correct and aligns with the caveated certificate story.

### Skeptic — Top 3 Contributions
1. **Recent CVE evidence demolishing the "problem solved" narrative.** The five 2023–2025 CVEs (especially wolfSSL CVE-2024-5814 and CVE-2025-11934) are concrete evidence that protocol downgrade vulnerabilities persist. This is the single most important piece of evidence in the entire evaluation — it validates the problem's continued relevance.
2. **Downgrading three near-FATAL candidates with explicit reasoning.** C10 (bounded completeness vacuous), C11 (A0 backdoor), and C12 (18 months unrealistic) were each rigorously examined and downgraded with specific mitigations. This prevents false-negative abandonment of a viable project.
3. **MVP-from-day-one mandate.** The insistence on planning the minimal paper alongside the full vision is the correct engineering strategy. A team that builds only for the full vision has a 19% success probability; a team that builds for the MVP has a 55% probability and can extend to the full vision if gates clear early.

### Synthesizer — Top 3 Contributions
1. **Three independently valuable layers.** The insight that the pipeline decomposes into (a) DY+SMT encoding framework, (b) merge operator formalization, and (c) slicer heuristics — each publishable independently at lower-tier venues — transforms the risk calculus. Even total pipeline failure yields reusable artifacts.
2. **P(any publication) ≈ 75–85%.** By accounting for Approach B salvage (CipherReach merge predicate, ~1K LoC) and Approach C salvage (covering-design scenarios, ~2K LoC), the Synthesizer demonstrates that the *expected* outcome is positive even under pessimistic assumptions about the full pipeline.
3. **Kill-gate timeline.** The concrete 10-week gate structure (slicer → merge → Z3 tractability) provides a decision framework that converts vague feasibility concerns into binary checkpoints.

---

## 3. Consensus Findings (All Three Experts Agree)

| # | Consensus Finding | Confidence |
|---|-------------------|------------|
| C1 | **Difficulty is honestly 7/10.** All three experts agree with the self-assessment. ~50K total LoC with ~36K genuinely hard. The KLEE integration and protocol module work are substantial but not research-grade. | HIGH |
| C2 | **O(n) claim must be reframed.** The headline O(n) applies only to the cipher-selection subroutine under idealized axioms A1–A4. Production code introduces multiplicative factors (session resumption, ALPN, SNI, 0-RTT) yielding O(n·m·k) ≈ 1000+ paths. The paper must report both the idealized and realistic numbers. | HIGH |
| C3 | **Slicer soundness (A0) is the trust anchor.** The composition theorem (T4) is only as good as the slicer. A0 cannot be proved for production code within scope; it must be stated as an assumption with empirical validation (CVE reachability + random-path sampling). | HIGH |
| C4 | **Z3 tractability is the highest-variance risk.** The DY+SMT encoding for a full TLS 1.2 negotiation with n=5 adversary actions produces formulas that may exceed Z3's capacity. No prototype validates this. The 40% timeout risk (Auditor) is credible. | HIGH |
| C5 | **T3 mathematical depth is 5/10.** The merge operator correctness proof is a domain instantiation of congruence closure / bisimulation theory, not a new proof technique. Novel application, standard methodology. | MEDIUM-HIGH |
| C6 | **The composition is genuinely new.** No prior tool connects source → LLVM IR → protocol-aware slice → symbolic execution with merge → state machine extraction → DY+SMT encoding → concrete attack trace. The pipeline's novelty is in the composition, not the individual components. | HIGH |
| C7 | **18-month full timeline is aggressive but MVP is viable.** Full vision with 4 libraries, 8 CVEs, and certificates is a stretch. MVP with 2 libraries, 4–5 CVEs, and attack synthesis (no certificates) is achievable in 12–14 months. | MEDIUM-HIGH |
| C8 | **Bounded completeness applies to ALL bounded verification — not a unique weakness.** The Skeptic correctly identified and then correctly downgraded C10: any bounded model checker (CBMC, SPIN with depth bounds) faces the same limitation. The mitigation (bounds-sweep + coverage plateau experiment) is standard and sufficient. | HIGH |

---

## 4. Per-Dimension Unified Scoring

### 4.1 Value: 6.5/10

| Factor | Assessment |
|--------|-----------|
| Problem relevance | **Active.** Five CVEs in 2023–2025 refute the "solved problem" narrative. Terrapin (SSH, 2023), wolfSSL CVE-2024-5814 and CVE-2025-11934, uTLS bypass, .NET SMTP downgrade — the attack class persists across protocols and implementations. |
| Stakeholder breadth | **Narrow.** 4–6 library maintainer teams, IETF WG members, ~50–100 protocol security researchers worldwide. Not a mass-market tool. |
| TLS 1.3 narrowing | **Partially mitigated.** TLS 1.3's anti-downgrade sentinel helps compliant-only deployments, but legacy TLS (1.0–1.2) remains ubiquitous in IoT/embedded, SSH is fully in scope, and cross-version interaction paths in mixed-mode libraries are the primary attack surface. |
| Competitive landscape | **Favorable gap.** tlspuffin (S&P 2024) is the closest competitor; NegSynth's completeness guarantee and source-level analysis are genuine differentiators. ProVerif/Tamarin operate on specifications, not implementations. No existing tool occupies NegSynth's niche. |
| Delta from self-assessment (7/10) | **−0.5.** Self-assessment slightly overstates breadth of impact. |

### 4.2 Difficulty: 7/10

| Factor | Assessment |
|--------|-----------|
| Total novel LoC | ~50K (midpoint), of which ~36K is genuinely algorithmic (slicer, merge operator, DY encoder, CEGAR loop, state machine extractor). ~14K is protocol modules and integration. |
| Reuse leverage | **Substantial.** KLEE (~95K LoC), Z3 (reused as-is), tlspuffin DY algebra (adapted), TLS-Attacker (validation). The reuse is a strength, not a weakness — it follows standard systems security practice. |
| Integration complexity | **High.** KLEE C++ ↔ Rust FFI, Z3 bindings, LLVM IR manipulation, OpenSSL build system. Each integration point is a potential multi-day blocker. |
| Human-authored components | ~20K LoC of protocol modules (TLS 1.0–1.3, SSH v2 message grammars) are specification-driven, not algorithmically novel. This is honest engineering work but does not contribute to difficulty score. |
| Delta from self-assessment (7/10) | **0.** All three experts agree. |

### 4.3 Best-Paper Potential: 5.5/10

| Factor | Assessment |
|--------|-----------|
| Novelty of composition | **Genuine.** No prior tool connects all pipeline stages. The end-to-end loop from C source to concrete attack trace or bounded certificate is new. |
| T3 depth | **5/10.** Domain instantiation of known bisimulation / congruence-closure theory. Novel *application*, standard *methodology*. Not best-paper-caliber mathematics. |
| O(n) narrative | **Weakened by honest reframing.** The money plot must show idealized cipher-selection scaling *and* realistic full-negotiation path counts. A 10–100× improvement is still compelling but less dramatic than the headline O(n) vs. O(2^n). |
| Certificate caveats | **Significant.** Certificates depend on A0 (slicer soundness), which is assumed, not proved. Reviewers at S&P will flag this. Certificates are "conditional on A0" — a weaker claim than "unconditional." |
| New CVE discovery potential | **The wild card.** Finding a new, previously unknown downgrade vulnerability in a current library HEAD would transform the paper from "interesting tool" to "must-accept." Probability: 15–25% (negotiation code is well-audited but NegSynth explores exhaustively within bounds). |
| Delta from self-assessment (7/10) | **−1.5.** Self-assessment significantly overestimates. T3 depth and certificate caveats are the primary deflators. |

### 4.4 Laptop-CPU / No-Humans: 6/10

| Factor | Assessment |
|--------|-----------|
| Laptop feasibility | **Plausible with caveats.** The proposal claims 8-hour analysis on laptop CPU. This requires (1) slicer producing ≤7K-line slices, (2) merge operator achieving ≥10× path reduction, (3) Z3 solving the encoded formula within timeout. If any of these fail, runtime explodes. |
| Z3 timeout risk | **40% (Auditor estimate, consensus-endorsed).** The DY+SMT encoding for full TLS 1.2 negotiation with n=5 adversary budget in BV+Arrays+UF+LIA theory is large. No prototype validates Z3's performance on this class of formulas. CEGAR refinement may help but adds its own complexity. |
| Human-authored components | ~20K LoC of protocol modules require expert knowledge of TLS/SSH wire formats. These are not automatically generated. A domain expert must write and validate them. This does not violate "no humans at runtime" but does require significant human effort at development time. |
| Memory validation | **Unvalidated.** The proposal does not estimate peak memory consumption during symbolic execution of OpenSSL's negotiation logic. KLEE is known to be memory-hungry; 64GB may be needed for large libraries. |
| Benchmark reproducibility | **80–160 hours of CPU time (Synthesizer estimate).** Feasible on a modern laptop over several days, but not "push-button in 8 hours" for the full evaluation suite across 4 libraries × multiple historical versions. |
| Delta from self-assessment (7/10) | **−1.** Z3 risk and memory concerns are underweighted in self-assessment. |

### 4.5 Feasibility: 5.5/10

| Factor | Assessment |
|--------|-----------|
| P(full vision delivered) | **19%.** Compound of: slicer works (70%) × merge operator works (75%) × Z3 tractable (60%) × integration holds (65%) × evaluation reproduces (80%) ≈ 19%. All three experts endorse this range (±3%). |
| P(MVP delivered) | **55%.** MVP = 2 libraries, 4–5 CVEs, merge operator + attack synthesis, no certificates. Drops the certificate story and differential extension, halving the integration surface. |
| Kill-gate structure | **Three serial gates in 10 weeks.** (1) Slicer produces ≤15K-line slices on OpenSSL by Week 4. (2) Merge operator demonstrates measurable path reduction on sliced code by Week 7. (3) Z3 returns SAT/UNSAT on ≥1 CVE encoding by Week 10. Failure at any gate triggers pivot to MVP or salvage. |
| Timeline | **18 months (full) / 12–14 months (MVP).** The 5-month theory + 13-month implementation split is reasonable. However, the 50K novel LoC target implies ~200 LoC/day sustained, which is aggressive for research-quality systems code with formal properties. |
| Team risk | **Single-person or very small team.** The breadth of required expertise (LLVM IR, KLEE internals, Rust, Z3, TLS/SSH protocols, formal methods) is unusual. A single researcher faces context-switching overhead that degrades throughput 20–30%. |
| Delta from self-assessment (7/10) | **−1.5.** Self-assessment significantly underweights the compound probability and kill-gate cascade. |

---

## 5. Fatal Flaw Analysis

### Flaw Inventory

| ID | Flaw | Severity | P(Manifests) | Impact if Manifests | Mitigation |
|----|------|----------|:---:|------|------------|
| F1 | **Slicer soundness unproved** — silent unsoundness produces false certificates | CRITICAL | 25–35% | Invalidates certificates; paper must drop certificate claims | State as Assumption A0; empirical validation via CVE reachability + random-path sampling; certificate wording becomes "conditional on A0" |
| F2 | **Z3 tractability unvalidated** — DY+SMT encoding may exceed solver capacity | HIGH | 35–45% | No attack synthesis or certificates on production code; tool becomes a prototype | CEGAR refinement; incremental solving; CVC5 fallback; decompose per-CVE; reduce adversary budget for initial experiments |
| F3 | **O(n) claim misleading** — production code yields O(n·m·k), not O(n) | MEDIUM-HIGH | 60–70% | Weakens headline result; reviewers flag as overclaim | Reframe to cipher-selection subroutine; report both idealized and realistic numbers; show 10–100× improvement empirically |
| F4 | **OpenSSL 3.x provider architecture** — engine/provider dispatch breaks slicer assumptions | MEDIUM | 70% | OpenSSL 3.x excluded from evaluation; limits to OpenSSL 1.1.x + other libraries | Target OpenSSL 1.1.1 (still widely deployed); accept as known limitation |
| F5 | **Circular bound justification** — k=20, n=5 validated only against known CVEs | MEDIUM | 30% | Reviewers question whether bounds capture unknown attack classes | Structural argument from RFC message counts; bounds-sweep showing coverage plateau; scope-exclude multi-renegotiation |
| F6 | **Merge operator fallback frequency** — real code violates A1–A4 often, degrading O(n) to near-generic | MEDIUM | 40–50% | Path reduction less dramatic than claimed; runtime increases | Per-region property checker; report merge-hit rate per library; paper claims "up to O(n)" with measured improvement |

### Compound Risk Assessment

- **P(F1 ∧ F2):** If both slicer and Z3 fail, the entire pipeline produces nothing. P ≈ 10–15%. This is the nightmare scenario but remains below the abandonment threshold.
- **P(F1 ∨ F2):** At least one of the two critical risks manifests. P ≈ 50–60%. This is high but manageable — each has independent mitigations, and the MVP can survive one (but not both) failing.
- **P(no flaw manifests):** ≈ 5–10%. The team should *expect* at least one serious flaw to manifest and plan accordingly.

### Verdict on Fatal Flaws

**No single flaw is independently fatal.** F1 (slicer) degrades the paper from "certificate story" to "attack synthesis story" — still publishable. F2 (Z3) degrades from "production-scale" to "prototype on simplified models" — still publishable at lower tier. F3 (O(n)) is a presentation issue, not a technical failure. F4 and F5 are scope limitations, not blockers.

The combination F1+F2 *is* potentially fatal for a top-4 venue paper but not for any publication: salvage components remain publishable at ACSAC/ESORICS/NDSS-workshop tier.

---

## 6. Probability Estimates

### Publication Outcomes

| Outcome | Probability | Conditions |
|---------|:-----------:|-----------|
| Best paper at S&P/USENIX/CCS | 2–4% | Full pipeline + new CVE discovery + O(n) holds on production code |
| Accepted at S&P | 12–18% | Full pipeline with honest caveats; strong empirical evaluation |
| Accepted at USENIX Security | 18–25% | Full pipeline or strong MVP; USENIX values practical tools |
| Accepted at CCS | 20–28% | MVP sufficient; CCS accepts narrower formal-methods contributions |
| Accepted at any top-4 venue (full vision) | 20–30% | Everything works; honest framing accepted by reviewers |
| Accepted at any top-4 venue (MVP) | 35–45% | MVP + 4–5 CVE recoveries + merge operator empirical validation |
| Accepted at any top-4 venue (either path) | 40–50% | Max of full and MVP paths, accounting for correlation |
| Accepted at Tier 2 venue (NDSS, ACSAC, ESORICS) | 55–65% | Partial pipeline with strong component evaluation |
| Any peer-reviewed publication | 70–80% | Salvage components publishable independently |
| Zero publications | 15–25% | Total pipeline failure + no salvage effort |

### Delivery Outcomes

| Outcome | Probability |
|---------|:-----------:|
| Full vision delivered (4 libraries, 8 CVEs, certificates, differential) | 15–22% |
| MVP delivered (2 libraries, 4–5 CVEs, attack synthesis, no certificates) | 50–60% |
| Partial delivery (1 library, 2–3 CVEs, prototype quality) | 70–75% |
| Salvage-only (individual components, no integrated pipeline) | 85–90% |
| Complete failure (nothing publishable) | 10–15% |

### Kill-Gate Probabilities

| Gate | Deadline | P(Pass) | Failure Mode |
|------|----------|:-------:|-------------|
| G1: Slicer produces ≤15K-line OpenSSL slice | Week 4 | 65–75% | Slice too large → exclude OpenSSL 3.x, fall back to 1.1.1 or WolfSSL |
| G2: Merge operator shows ≥5× path reduction | Week 7 | 70–80% | Insufficient reduction → fallback frequency too high → O(n) story dead |
| G3: Z3 solves ≥1 CVE encoding (SAT) | Week 10 | 55–65% | Timeout → decompose formula → reduce bounds → prototype-only story |
| All three gates cleared | Week 10 | 25–35% | — |

---

## 7. Final Unified Scores

| Dimension | Self-Assessment | Auditor | Skeptic | Synthesizer | **Final Score** | **Rationale** |
|-----------|:---:|:---:|:---:|:---:|:---:|---------------|
| Value | 7 | 6 | ~6 | 7 | **6.5** | Recent CVEs validate relevance; narrow stakeholder pool caps excitement |
| Difficulty | 7 | 7 | ~7 | 7 | **7.0** | Full consensus; ~36K genuinely hard LoC on mature infrastructure |
| Best-Paper | 7 | 6 | ~5 | 6 | **5.5** | T3 depth 5/10; O(n) weakened; certificates caveated; new-CVE wild card |
| Laptop/No-Humans | 7 | 6 | ~6 | 6 | **6.0** | Z3 timeout 40%; unvalidated memory; 80–160hr benchmarks; human modules |
| Feasibility | 7 | 6 | ~5.5 | 5 | **5.5** | 19% full compound; 55% MVP; three serial kill gates in 10 weeks |
| **Composite** | **7.0** | **6.2** | **~5.8** | **6.2** | **6.0** | Weighted: 0.25V + 0.15D + 0.20BP + 0.15L + 0.25F |

**Delta from self-assessment: −1.0.** The self-assessment is consistently optimistic by approximately one point across all dimensions except Difficulty. The primary deflators are: feasibility compound probability (−1.5), best-paper T3 depth and certificate caveats (−1.5), and Z3 tractability risk on Laptop/No-Humans (−1.0).

---

## 8. Binding Conditions for CONTINUE

The following conditions are **mandatory**. Failure to meet any one triggers reassessment; failure to meet Conditions 1–3 triggers ABANDON.

### Condition 1: MVP-First Architecture (Immediate)

The implementation plan must be restructured around the MVP (2 libraries, 4–5 CVEs, attack synthesis) as the *primary* deliverable, with certificates and differential extension as stretch goals. Every design decision must ask: "Does this work for the MVP?" before "Does this work for the full vision?"

**Rationale:** P(full) = 19% is too low to bet on exclusively. P(MVP) = 55% is viable. The MVP is a publishable paper at CCS/USENIX.

### Condition 2: Kill-Gate Protocol (Weeks 4, 7, 10)

Establish explicit, binary kill gates with pre-committed decision criteria:
- **Week 4:** Slicer produces a slice of OpenSSL 1.1.1 or WolfSSL negotiation logic containing ≤15,000 lines and including all code paths for ≥2 known CVEs. FAIL → pivot to WolfSSL-only (smaller codebase) or ABANDON slicer in favor of manual annotation.
- **Week 7:** Merge operator on the sliced code demonstrates ≥5× path reduction versus vanilla KLEE on the same slice, measured by total explored states. FAIL → drop O(n) narrative; reframe as "protocol-aware search strategy with empirical speedup."
- **Week 10:** Z3 returns SAT on ≥1 CVE encoding (producing a concrete attack trace) within 2 hours on a single laptop core. FAIL → decompose formula; reduce bounds to k=10, n=3; if still fails, ABANDON SMT-based approach in favor of direct symbolic execution with property checking.

**Rationale:** These gates convert vague risk into concrete decisions. Pre-committing prevents sunk-cost fallacy.

### Condition 3: Z3 Feasibility Prototype (Before Week 6)

Before committing to the full DY+SMT encoding, build a *minimal* Z3 feasibility prototype: hand-encode the Dolev-Yao model for a single CVE (FREAK or POODLE-version) on a hand-written 200-line negotiation model. If Z3 cannot solve this simplified instance in <10 minutes, the production-scale encoding is extremely unlikely to work.

**Rationale:** This is the cheapest possible test of the highest-variance risk (F2). It costs ~1 week of effort and provides a definitive signal.

### Condition 4: Honest O(n) Framing (In Paper)

The paper must:
- State the O(n) bound as applying to the cipher-selection subroutine under axioms A1–A4
- Report the fallback frequency (% of code regions where merge fires vs. falls back to generic exploration)
- Show both idealized and realistic path counts in the evaluation
- Title the merge-operator evaluation section with language like "up to O(n) path reduction on negotiation-conformant code"

**Rationale:** All three experts flag O(n) as potentially misleading. Honest framing prevents reviewer rejection on overclaim grounds.

### Condition 5: A0 Validation Protocol (In Evaluation)

The paper must:
- State slicer soundness as Assumption A0, not a proved theorem
- Validate A0 via: (a) CVE reachability — all 8 CVE-vulnerable code paths present in the slice, (b) random-path sampling — 10K random negotiation traces on un-sliced library, verify each reaches a state in the extracted model, report miss rate
- Certificate wording: "Under Assumption A0, within bounds (k, n), no downgrade attack exists"
- Limitations section explicitly discusses slicer as trust anchor and failure mode of silent unsoundness

**Rationale:** Consensus across all three experts. This is the minimum honest framing that reviewers will accept.

### Condition 6: Venue Targeting (Strategic)

Primary targets: **CCS** and **USENIX Security** (practical tools track). Secondary target: **NDSS** (if MVP-only). S&P should be a stretch target only if the full vision delivers *and* a new CVE is discovered.

**Rationale:** CCS and USENIX are more receptive to practical security tools with caveated guarantees. S&P demands stronger formal claims than the certificates can deliver.

---

## 9. VERDICT

### CONDITIONAL CONTINUE

**Confidence in verdict: 75%.**

The proposal addresses a real, ongoing problem (validated by recent CVEs), proposes a genuinely novel pipeline composition, and has a credible path to publication at a top-4 venue via the MVP. The 55% MVP delivery probability and 40–50% top-4 venue acceptance probability (across full and MVP paths) exceed the threshold for continuation.

The risks are serious but manageable: Z3 tractability (testable via Condition 3), slicer soundness (mitigatable via A0 framing), and O(n) overclaim (fixable via honest reframing). The salvage value is high — even partial delivery produces independently publishable components with P(any publication) ≈ 70–80%.

**The proposal should proceed to implementation under all six binding conditions.** The team must internalize that the *expected* outcome is the MVP paper at CCS/USENIX, not the full-vision paper at S&P. The full vision is a 19% moonshot; the MVP is a 55% solid bet. Plan for the solid bet; celebrate if the moonshot lands.

### What Would Change This Verdict

- **To ABANDON:** Z3 feasibility prototype (Condition 3) fails on the simplified instance, *and* no decomposition strategy recovers tractability within 2 weeks. This would eliminate the core technical mechanism.
- **To UNCONDITIONAL CONTINUE:** Z3 feasibility prototype succeeds in <1 minute, *and* slicer produces ≤5K-line slice on OpenSSL 1.1.1, *and* merge operator shows ≥20× path reduction. This would shift P(full vision) from 19% to ~40%.

---

## Appendix A: Expert Calibration Assessment

| Expert | Bias Direction | Calibration Quality | Most Trusted Dimension |
|--------|---------------|-------------------|----------------------|
| Auditor | Slightly optimistic on feasibility; well-calibrated on value | HIGH — evidence-based with explicit probability ranges | Fatal flaw decomposition |
| Skeptic | Slightly pessimistic on best-paper; well-calibrated on feasibility | HIGH — fail-fast methodology correctly identified and then correctly dismissed three near-FATAL candidates | Recent CVE evidence; venue targeting |
| Synthesizer | Slightly optimistic on publication probability; well-calibrated on salvage | MEDIUM-HIGH — salvage analysis is uniquely valuable but P(any publication) may be inflated by 5–10% | Salvage value; kill-gate structure |

All three experts converge on the same verdict (CONDITIONAL CONTINUE) despite starting from different analytical frames (evidence-based, fail-fast, salvage-value). This convergence, despite different priors and methodologies, provides HIGH confidence that the verdict is correct.

---

## Appendix B: Comparison to Self-Assessment

| Claim | Self-Assessment | Community Expert Assessment | Gap |
|-------|-----------------|----------------------------|-----|
| Composite score | 7.0/10 | 6.0/10 | −1.0 |
| Theory health | 7.5/10 | 7.0/10 (validated by approach.json depth) | −0.5 |
| Best-paper probability | 5–10% | 2–4% | −3 to −6 pp |
| Top-4 venue acceptance | 45–55% | 20–30% (full) / 35–45% (MVP) | −10 to −25 pp |
| P(full delivery) | ~50% (implied) | 19% | −31 pp |
| P(MVP delivery) | Not stated | 55% | N/A |

The self-assessment is systematically optimistic by ~1 point on composite and ~15–25 percentage points on acceptance probability. This is within the normal range for researcher self-assessment (Dunning-Kruger moderated by domain expertise) and does not indicate dishonesty — it indicates the natural optimism of a researcher who has internalized the best-case execution path.

---

*This evaluation synthesizes three independent expert analyses into a unified assessment. The Community Expert endorses the CONDITIONAL CONTINUE verdict with the six binding conditions specified above. Implementation should proceed with the MVP as the primary target and kill gates as pre-committed decision points.*
