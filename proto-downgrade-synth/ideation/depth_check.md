# Verification Depth Check: proto-downgrade-synth

**Title:** "Negotiation Under Fire: Bounded-Complete Synthesis of Protocol Downgrade Attacks from Library Source Code"
**Stage:** Verification (team-based: Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer)
**Date:** 2026-03-08
**Method:** Three-expert independent assessment → adversarial cross-challenge → consensus synthesis

---

## Pillar Scores

### P1: EXTREME AND OBVIOUS VALUE — 7/10

**Evidence FOR (strong):**
- Protocol downgrade attacks are genuinely devastating and high-impact. FREAK, Logjam, POODLE, DROWN, and Terrapin collectively affected billions of devices. The specification-implementation gap is a well-documented, structurally unsolved problem (Bhargavan et al., S&P 2016; Cremers et al., CCS 2017).
- Terrapin (disclosed late 2023, in SSH) proves the gap is still lethal in modern, well-audited protocols. This is not a historical curiosity — new downgrade attacks continue to be discovered manually, years after vulnerable code ships.
- No existing tool closes the gap. ProVerif/Tamarin verify specs, not code. KLEE/SAGE lack adversary models. tlspuffin requires *hand-authored* protocol models that reintroduce the faithfulness gap. NegSynth fills a genuine blind spot.

**Evidence AGAINST (moderate):**
- TLS 1.3's anti-downgrade sentinel (RFC 8446 §4.1.3) progressively eliminates the TLS-specific attack surface. The future value diminishes for TLS 1.3-only deployments.
- The stakeholder market is narrow: ~20-50 library maintainers, a handful of auditing firms, and IETF working group members. This is high-impact-per-user but low-total-users.
- The Skeptic raised LLM competition (Google Big Sleep found a real bug in SQLite, 2024). **Cross-challenge resolution: this argument was rejected.** LLMs find pattern-matchable bugs; protocol downgrade attacks require compositional adversarial reasoning over state machines — precisely where symbolic methods dominate and LLMs hallucinate. No LLM has ever synthesized a Dolev-Yao attack trace from source code.

**Score rationale:** Problem is real, proven lethal by Terrapin, and structurally unsolvable by existing tools. Value is immediately clear to security researchers and protocol implementers. Docked from 8 to 7 because TLS 1.3 narrows the TLS-specific surface (though SSH, QUIC, and legacy TLS remain wide open).

---

### P2: GENUINE DIFFICULTY AS A SOFTWARE ARTIFACT — 6/10 (BELOW THRESHOLD)

**Evidence FOR:**
- The protocol-aware merge operator (T3), protocol-aware slicer, state machine extractor, and DY+SMT encoder with CEGAR refinement are genuinely novel, algorithmically hard subsystems. None is glue code.
- The end-to-end pipeline (source → slice → symex → extract → encode → solve → concretize) has many integration surfaces, each requiring careful correctness preservation.
- Building on KLEE adds genuine integration difficulty — KLEE's codebase is ~95K LoC of battle-tested but architecturally complex C++.

**Evidence AGAINST:**
- The original 155K LoC estimate is **significantly inflated**. Breakdown of non-novel components:
  - Test Infrastructure (15-20K): essential but not the contribution
  - CLI and Reporting (10-14K): SARIF formatter + HTML viz is engineering, not research
  - Evaluation Harness (15-18K): comparison scaffolding, expected not difficult
  - Protocol Modules (25-30K): tedious RFC transcription, not algorithmic novelty
  - **Total non-novel: ~65-82K LoC (~45% of headline)**
- The genuinely novel algorithmic core (symbolic engine + slicer + extractor + DY encoder + concretizer) is ~50-65K LoC. Substantial but dramatically less than "155K."
- The Skeptic's "KLEE + 5K DY harness gets 90%" claim was **rejected** (merge operator, slicer, and DY encoder are non-trivial), but the underlying point stands: the original proposal doesn't acknowledge leveraging existing tools (KLEE, tlspuffin DY model, TLS-Attacker replay).

**Why below 7:** The 155K headline invites justified skepticism from experienced reviewers. KLEE itself is ~95K LoC after 15+ years of development; claiming 155K of genuinely novel code for a single-paper artifact strains credulity. The honest figure (~50K novel + ~40K integration/protocol modules, built on KLEE) is still impressive but must be presented honestly.

**Amendment to reach 7:** Adopt two-tier LoC framing: "~50K lines of novel protocol-analysis code built on top of KLEE's symbolic execution infrastructure, with ~40K lines of protocol modules and integration." Explicitly disclose the reuse strategy (KLEE for symex, tlspuffin DY model for adversary algebra, TLS-Attacker for replay validation). This is standard practice and *strengthens* credibility.

---

### P3: BEST-PAPER POTENTIAL — 6/10 (BELOW THRESHOLD)

**Evidence FOR:**
- The end-to-end story (source code in → concrete attack traces out, with bounded completeness) is genuinely novel as a *composition*. No prior system claims bounded-complete downgrade synthesis from implementation source code.
- T3 (protocol-aware merge) + T4 (bounded completeness) together represent a real contribution: the composition must thread extraction soundness (T1) through merge correctness (T3) through encoding correctness (T5) into end-to-end guarantees.
- Finding even one new CVE in a production library would be a strong empirical result comparable to KLEE's OSDI 2008 best paper (bugs in GNU coreutils).

**Evidence AGAINST:**
- T3 in isolation is a domain instantiation of existing techniques (veritesting, Kuznetsov et al. PLDI 2012 state merging, Milner bisimulation). The protocol-specific specialization means restricting to *the easiest possible domain for state merging*: finite enumerations, deterministic transitions, monotonic orderings.
- T4 is a composition theorem over T1, T3, T5. If the components are adaptations of known techniques, composing them is moderate novelty — a legitimate contribution but not a "surprise" result.
- The "at least one new vulnerability" promise is an uncontrollable bet on the external world. Modern TLS libraries (BoringSSL, rustls) have been subjected to years of expert review, continuous fuzzing, and formal analysis. If no new vulnerability is found, the paper's most compelling evaluation result evaporates.
- "Big tool" papers face structural headwinds at top venues. The paper must foreground theory (T3, T4) and certificates, not the tool itself.

**Why below 7:** The novelty needs sharper articulation and the empirical strategy needs de-risking. Without a clear demonstration that protocol-aware merge achieves something generic veritesting cannot (the "money plot" showing O(2^n) → O(n) path reduction), T3 will be dismissed as "just veritesting for TLS."

**Amendments to reach 7:**
1. Include a formal comparison showing generic veritesting's exponential blowup vs. protocol-aware merge's linear behavior on negotiation code — this defeats the "just veritesting" critique.
2. Frame bounded-completeness *certificates* as the primary empirical contribution (novel artifacts with immediate practical value), not bug count.
3. Drop the hard commitment to finding new vulnerabilities. Lead with: "We produce the first bounded-completeness certificates for production TLS/SSH libraries."
4. Provide empirical k/n/ε characterization (table of minimal bounds per CVE, coverage at chosen bounds, measured concretization success rate).

---

### P4: LAPTOP CPU + NO HUMANS — 6/10 (BELOW THRESHOLD)

**Evidence FOR:**
- SMT solving is inherently sequential and branch-heavy — GPUs offer no advantage. The "no GPU" constraint is naturally satisfied.
- Protocol-aware slicing reducing 200K → 2-10K lines is **factually supported**: OpenSSL's negotiation decision logic lives in `ssl/statem/statem_clnt.c`, `statem_srvr.c`, `ssl_ciph.c`, and portions of `t1_lib.c` — roughly 3-7K lines out of 500K+ total (~1%). The Skeptic's claim of "10-15%" conflated negotiation *decision logic* with all *handshake-reachable* code (including cipher implementations, ASN.1, certificate handling).
- Z3 handles 25K-variable BV+Arrays+UF problems in seconds. The bounded adversary model (finite n) caps query complexity.
- Evaluation requires zero human annotation: CVE oracles automated, attack replay automated, all metrics computed programmatically.

**Evidence AGAINST:**
- The "25,000 variables per query" bound appears unsubstantiated. No profiling data, no preliminary measurements.
- 16GB RAM is tight. Z3 peak memory on complex BV queries can spike unpredictably. The proposal claims 12-16GB peak but hasn't validated this.
- **Arithmetic inconsistency**: 35 configurations × 4-8 hours ≠ 48-72 hours. This implies either aggressive parallelization (needs >8 cores) or many configurations are much faster — neither is explained.
- The 4-8 hour per-library estimate has no breakdown by pipeline stage (slicing vs. symex vs. SMT solving).

**Why below 7:** The architecture is sound but every quantitative claim is unvalidated. Experienced reviewers will notice the timing arithmetic error and the unsubstantiated variable-count bound. These are fixable but cannot be left as-is.

**Amendments to reach 7:**
1. Build a PoC on a single library (OpenSSL) to empirically validate: slicing ratio, per-library time breakdown, peak memory, and typical SMT query size.
2. Fix the 35×8hr ≠ 48-72hr arithmetic. Either reduce configurations or acknowledge multi-core parallelism.
3. Remove or empirically justify the 25K-variable claim.
4. Increase RAM target to 32GB as a safer bound (still laptop-class).

---

### P5: FATAL FLAWS

**ORIGINAL FATAL FLAWS IDENTIFIED:**

| # | Flaw | Source | Resolution |
|---|------|--------|------------|
| F1 | Multi-language symbolic execution (C+Rust+Go) with unified IR has never been achieved | Skeptic (independently fatal) | **RESOLVED by amendment**: Scope to C-only via LLVM IR. Build on KLEE. All four most-important libraries (OpenSSL, BoringSSL, WolfSSL, libssh2) are C. |
| F2 | The 1-5% slicing claim is unsubstantiated and likely 5x too optimistic | Skeptic (independently fatal) | **RESOLVED by factual analysis**: The 1-5% figure refers to negotiation *decision logic*, not all handshake-reachable code. OpenSSL's `ssl/statem/` + `ssl_ciph.c` + `t1_lib.c` ≈ 5K / 500K+ = ~1%. Skeptic's 10-15% conflated scope. PoC validation still required. |
| F3 | Bounded completeness with user-chosen k, n and unspecified ε is near-vacuous | Skeptic + Auditor | **RESOLVED by amendment**: Empirical k/n table for all historical CVEs, measured ε from CEGAR loop, coverage percentage at chosen bounds, structural argument that negotiation depth is inherently bounded. |
| F4 | POODLE and DROWN misclassified as negotiation attacks | Skeptic + Auditor | **RESOLVED by amendment**: Classified CVE table. POODLE's version-downgrade component is in scope; padding oracle is not. DROWN-specific (CVE-2016-0703) is in scope; general DROWN is borderline. 8 clean CVEs. |
| F5 | "At least one new vulnerability" promise is unfalsifiable | All three experts | **RESOLVED by amendment**: Drop new-vuln commitment. Certificates as primary contribution. |

**RESIDUAL RISKS (not fatal but require monitoring):**

| Risk | Severity | Mitigation |
|------|----------|------------|
| KLEE integration difficulty (C++ codebase, complex architecture) | MEDIUM | Budget 3-4 months for KLEE integration; have angr/Manticore as fallback |
| SMT solver timeout on large queries | MEDIUM | CEGAR refinement + incremental solving; fallback to query splitting |
| MIR/Go stretch goals may never materialize | LOW | These are explicitly future work; paper stands on C-only |
| ε may be unacceptably large in practice | MEDIUM | Measure early; if ε > 0.01, refine CEGAR loop or drop probabilistic qualifier |

**Verdict: 0 fatal flaws after amendments. 4 residual risks requiring monitoring.**

---

## Consensus Score Summary

| Pillar | Score | Status | Key Issue |
|--------|:-----:|--------|-----------|
| P1: Extreme & Obvious Value | **7** | ✅ PASS | Real problem, proven lethal by Terrapin, structural gap in tooling |
| P2: Genuine Difficulty | **6** | ⚠️ BELOW 7 | ~50K novel LoC, not 155K; honest framing needed |
| P3: Best-Paper Potential | **6** | ⚠️ BELOW 7 | T3+T4 novelty needs sharper articulation; certificate framing needed |
| P4: Laptop CPU + No Humans | **6** | ⚠️ BELOW 7 | Architecture sound but quantitative claims unvalidated |
| P5: Fatal Flaws | **0 fatal** | ✅ PASS | All 5 original flaws resolved by amendments |

**Composite: (7 + 6 + 6 + 6) / 4 = 6.25/10**

---

## Mandatory Amendments (all 5 required for CONTINUE)

1. **C-only scope via LLVM IR.** Target OpenSSL, BoringSSL, WolfSSL (TLS) + libssh2 (SSH). Build on KLEE's symbolic execution core. Rust support via LLVM backend as stretch goal. Go → future work. *Resolves F1.*

2. **Honest two-tier LoC narrative.** ~50K novel protocol-analysis code + ~40K integration/protocol modules, built on KLEE (~95K reused). Drop unqualified "155K" headline. *Resolves credibility concern.*

3. **Certificate-first empirical framing.** Drop "at least one new vulnerability" commitment. Primary contribution: first bounded-completeness certificates for production TLS/SSH libraries. New vulnerabilities are a bonus, not load-bearing. *Resolves F5.*

4. **Classified CVE table.** 8 clearly in-scope CVEs (FREAK, Logjam, POODLE version-downgrade, Terrapin, DROWN-specific/CVE-2016-0703, SSLv2 cipher override/CVE-2015-3197, CCS Injection/CVE-2014-0224, plus one additional clean negotiation CVE). Partial-scope CVEs explicitly labeled. *Resolves F4.*

5. **Empirical bounded-completeness validation.** Minimal k/n table for all historical CVEs, measured ε from CEGAR refinement, negotiation-state coverage at chosen bounds, structural argument re: protocol round-trip depth. *Resolves F3.*

---

## Recommendation

### CONDITIONAL CONTINUE

The proposal contains a genuine contribution to the protocol security literature: bounded-complete downgrade-attack synthesis from implementation source code via a protocol-aware merge operator. The core insight is novel, the problem is real and proven lethal by recent attacks, and the approach is technically sound.

However, the original proposal is over-scoped (3 languages → should be 1), over-promised (new vulnerabilities → should be certificates), and over-counted (155K LoC → ~50K novel). All five amendments above are mandatory. With amendments, projected scores rise to V7/D7/BP7/L7 (composite 7.0/10) — solidly in the "strong submission" range for IEEE S&P, USENIX Security, or ACM CCS.

**P(top-4 venue acceptance with amendments): 45-55%**
**P(best paper): 5-10%** (rises to 10-15% if a new vulnerability is discovered)
**P(any publication): 70-80%**
**P(ABANDON): 15-20%** (primarily from KLEE integration risk or ε being unacceptably large)

### Kill Gates

| Gate | Timing | Condition | Action |
|------|--------|-----------|--------|
| G0 | Week 2 | KLEE integration PoC: can symbolically execute OpenSSL's `ssl_ciph.c` | KILL if fails |
| G1 | Week 4 | Protocol-aware slicer extracts ≤10K lines from OpenSSL with ≥90% negotiation coverage | KILL if fails |
| G2 | Week 6 | Merge operator demonstrates measurable path reduction (≥10x) on OpenSSL negotiation | KILL if fails |
| G3 | Week 10 | End-to-end pipeline recovers ≥3 CVEs from historically vulnerable OpenSSL versions | KILL if fails |
| G4 | Week 14 | Full evaluation: ≥85% recall on 8 clean CVEs, certificates produced for all 4 current HEAD libraries | REASSESS if fails |

---

## Team Signoff

| Expert | Recommendation | Notes |
|--------|---------------|-------|
| Independent Auditor | CONDITIONAL CONTINUE | With all 5 amendments. Original score 6.5/10; amended projection 7.0/10. |
| Fail-Fast Skeptic | OVERRULED (voted KILL) | Both "independently fatal" flaws resolved by amendments. Skeptic's contribution was critical: every amendment traces to a Skeptic-identified flaw. |
| Scavenging Synthesizer | CONDITIONAL CONTINUE | Proposed the rescue strategy (C-only, KLEE-based, certificate framing) that all experts adopted. |

**Final disposition: CONDITIONAL CONTINUE (2-1, Skeptic dissents)**
