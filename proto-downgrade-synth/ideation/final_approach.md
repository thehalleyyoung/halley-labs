# Final Approach: Protocol-Aware Bounded-Complete Synthesis with Differential Extension

**Project:** Negotiation Under Fire: Bounded-Complete Synthesis of Protocol Downgrade Attacks from Library Source Code
**Slug:** `proto-downgrade-synth`
**Stage:** Ideation — final approach after 3-approach competition, adversarial debate, and independent verification
**Date:** 2026-03-08
**Verdict:** CONDITIONAL APPROVE (6 binding conditions addressed below)

---

## 1. One-Line Summary

A KLEE-based pipeline that exploits the algebraic structure of cipher-suite negotiation to synthesize concrete downgrade attack traces — or certify their absence — from C library source code, extended with cross-library differential analysis to catch interoperability vulnerabilities invisible to single-library reasoning.

---

## 2. Approach Overview

This approach takes Approach A's **protocol-aware symbolic execution engine** as the load-bearing core — the only architecture that delivers standalone bounded-completeness certificates — and grafts Approach C's **differential cross-library mining** as a modular extension that discovers interoperability attack surfaces no single-library analysis can reach.

**Core engine (from Approach A):** C source → LLVM IR → protocol-aware slice → KLEE with protocol-aware merge operator → bisimulation-quotient state machine → DY+SMT encoding with CEGAR → concrete byte-level attack traces OR bounded-completeness certificates.

**Differential extension (from Approach C):** Run the core engine on N ≥ 3 libraries in parallel, align outputs at the wire-protocol level (IANA cipher IDs), detect behavioral deviations via covering-design-generated scenarios, rank by security impact, and verify exploitability using the core engine's DY+SMT solver. This produces *cross-library consistency certificates* on top of per-library bounded-completeness certificates.

**What was discarded and why:**
- **Approach B entirely.** Building a custom LLVM IR abstract interpreter from scratch is a multi-year effort infeasible for a small team. The Difficulty Assessor rated it 9/10 with 24-36 month timeline. B3 (domain-relative completeness) was exposed as vacuous/ornamental by the Math Depth Assessor.
- **C's standalone completeness claim.** The Skeptic demonstrated that covering designs cover *input configurations*, not *attack sequences*. C cannot produce absence certificates without A's verification engine underneath.

---

## 3. Pipeline Architecture

```
                            ┌─────────────────────────────────────┐
                            │        PHASE 1: CORE ENGINE         │
                            │     (per-library, runs N times)     │
                            └─────────────────────────────────────┘

C Source (OpenSSL/   ──►  [wllvm/gclang]  ──►  LLVM IR Bitcode
BoringSSL/WolfSSL/        (REUSED)              │
libssh2)                                        ▼
                                    ┌──────────────────────┐
                                    │ Protocol-Aware Slicer│  ◄── NOVEL (~11K LoC)
                                    │ (taint + points-to   │      Linchpin: must deliver
                                    │  across SSL_METHOD   │      ≤2% of source as slice.
                                    │  vtables, callbacks)  │
                                    └──────────┬───────────┘
                                               ▼
                                    Negotiation Slice (~3-7K lines)
                                               │
                                    ┌──────────▼───────────┐
                                    │  KLEE + Merge Op     │  ◄── NOVEL merge op (~7K)
                                    │  (protocol-aware     │      + integration layer
                                    │   searcher exploits  │      (~8K); REUSED KLEE
                                    │   4 algebraic props) │      engine (~95K)
                                    └──────────┬───────────┘
                                               ▼
                                    Symbolic Execution Traces
                                    (empirical 10-100x path
                                     reduction vs. veritesting)
                                               │
                                    ┌──────────▼───────────┐
                                    │  State Machine       │  ◄── NOVEL (~8K LoC)
                                    │  Extractor           │      Bisimulation quotient
                                    └──────────┬───────────┘
                                               ▼
                                    ┌──────────▼───────────┐
                                    │  DY+SMT Encoder      │  ◄── NOVEL (~10K LoC)
                                    │  (BV+Arrays+UF+LIA   │      REUSED: tlspuffin DY
                                    │   + DY adversary)     │      term algebra
                                    └──────────┬───────────┘
                                               ▼
                                    ┌──────────▼───────────┐
                                    │  Z3 + CEGAR Loop     │  ◄── NOVEL CEGAR (~6K)
                                    │  SAT → Concretizer   │      REUSED: Z3
                                    │  UNSAT → Certificate  │      Fallback: incremental
                                    │  TIMEOUT → decompose  │      solving, CVC5
                                    └──────────┬───────────┘
                                               ▼
                              ┌─────────────────┴─────────────────┐
                              ▼                                   ▼
                   Byte-Level Attack Trace            Bounded-Completeness
                   (TLS record / SSH packet)          Certificate at (k, n)
                              │                       with coverage metric
                              ▼
                   [TLS-Attacker Replay]  ◄── REUSED
                   Validated Attack


                            ┌─────────────────────────────────────┐
                            │    PHASE 2: DIFFERENTIAL EXTENSION   │
                            │    (cross-library, runs once)        │
                            └─────────────────────────────────────┘

Per-Library State    ──►  [Wire-Protocol Alignment]  ◄── NOVEL (~5K LoC)
Machines (from              (IANA cipher IDs)              Output-based alignment
Phase 1, N libs)                    │
                                    ▼
                          ┌──────────────────┐
                          │ Covering-Design  │  ◄── NOVEL (~6K LoC)
                          │ Scenario Gen     │      C1 theorem: B(n,k,t) bound
                          └────────┬─────────┘
                                   ▼
                          ┌──────────────────┐
                          │ Deviation Detect │  ◄── NOVEL (~4K LoC)
                          │ + Security Rank  │
                          └────────┬─────────┘
                                   ▼
Ranked Candidates ──► [Core Engine DY+SMT] ──► Cross-Library Certificate
                                                OR Confirmed Interop Attack
```

### Novel vs. Reused Summary

| Component | Status | LoC | Notes |
|-----------|--------|-----|-------|
| Protocol-Aware Slicer | **NOVEL** | ~11K | Linchpin component |
| Protocol-Aware Merge Operator | **NOVEL** | ~7K | Crown algorithmic contribution |
| KLEE Integration Layer | **NOVEL** | ~8K | Custom Searcher, Rust↔C++ FFI |
| State Machine Extractor | **NOVEL** | ~8K | Bisimulation quotient |
| DY+SMT Encoder + CEGAR | **NOVEL** | ~16K | Adversary encoding + refinement + concretizer |
| Differential Extension | **NOVEL** | ~15K | Alignment + scenarios + ranker |
| **Subtotal novel** | | **~50K** | **(amended per verifier BC-1)** |
| Protocol Modules (TLS+SSH) | Integration | ~20K | RFC transcription, grammars |
| Eval Harness + Tests + CLI | Integration | ~20K | CVE oracles, SARIF, CI |
| **Subtotal integration** | | **~40K** | |
| KLEE engine | **REUSED** | ~95K | Mature symbolic execution |
| Z3, tlspuffin DY, TLS-Attacker | **REUSED** | — | SMT, adversary model, replay |

**Total: ~50K novel LoC + ~40K integration, built on KLEE's ~95K reused infrastructure.**

---

## 4. Value Proposition

| Stakeholder | Current Pain | What NegSynth Delivers |
|-------------|-------------|----------------------|
| **Library maintainers** (~50 engineers at OpenSSL, WolfSSL, etc.) | Manual negotiation review, weeks/release, scarce domain experts | Push-button analysis in hours; attack traces or certificates |
| **Protocol designers** (IETF TLS/QUIC WGs) | No tool checks implementations against intended properties | Formal counterexamples or confirmation for draft extensions |
| **Security auditors** | Black-box probing or painstaking manual source review | White-box, source-level analysis with formal guarantees |
| **Supply-chain teams** (heterogeneous TLS stacks) | No cross-library consistency tooling | Cross-library differential certificates (extension) |

### Why Transformative

1. **First bounded-completeness certificates for production TLS/SSH.** "Within bounds k=20, n=5, covering ≥99% of reachable negotiation states, OpenSSL 3.x contains no cipher-suite downgrade attack."
2. **Closes the specification-implementation gap end-to-end.** Analyzes what the code actually does, not what it's supposed to do.
3. **Cross-library differential analysis** discovers interoperability vulnerabilities invisible to single-library tools — the exact class that produced Terrapin.

### Scope Acknowledgment

TLS 1.3's anti-downgrade sentinel narrows TLS-specific surface for 1.3-only deployments. Primary value: (a) legacy TLS 1.0–1.2 (IoT/embedded/enterprise), (b) SSH (Terrapin 2023), (c) cross-version interaction paths.

---

## 5. Technical Difficulty — Honest Assessment

**Difficulty: 7/10** (per depth-check-agreed baseline)

~50K novel LoC with genuinely hard algorithmic core. Each component adapts existing techniques to a new domain rather than inventing from scratch, which is why this is 7, not 8-9. The Difficulty Assessor's detailed calibration: KLEE from scratch was 9/10; S2E was 8/10; TLS-Attacker was 6/10. NegSynth falls between S2E and TLS-Attacker as a significant systems research project.

**Key engineering challenges (from Difficulty Assessor):**
- KLEE FFI is the hidden complexity trap — S2E required ~20 person-months for integration
- OpenSSL's `STACK_OF(SSL_CIPHER)` macro-expanded containers complicate merge operator
- DY+SMT encoding fidelity — Tamarin/ProVerif had years of soundness fixes

---

## 6. New Mathematics — All Load-Bearing Theorems

### Core Theorems (T1–T5)

#### T3: Protocol-Aware Merge Correctness (Crown Theorem)

**Statement:** The merge operator ⊵ preserves protocol-bisimilarity: for any two symbolic states s₁, s₂ reachable during negotiation, merging via ⊵ produces exactly the same set of observable negotiation outcomes as exploring s₁ and s₂ independently.

**Complexity result:** On negotiation code with n cipher suites and m phases, generic veritesting explores O(2^n · m) paths; protocol-aware merge explores O(n · m) paths. This is the theoretical ceiling for ideal negotiation logic satisfying all four algebraic properties (finite outcomes, lattice preferences, monotonic progression, deterministic selection).

**Honest depth: 5/10.** Domain instantiation of existing techniques (Kuznetsov state merging + Milner bisimulation), not a fundamental breakthrough. Novelty is *identifying* that negotiation protocols have the right algebraic structure. Each ingredient is textbook; the combination is new.

**Empirical framing:** The O(n) bound is the theoretical ceiling. Real OpenSSL code includes callbacks, `#ifdef` forests, FIPS overrides. The empirical claim is **10-100x path reduction** on production code, validated by head-to-head comparison with generic veritesting ("money plot").

**Proof risk:** Low (~5%) for the theorem; medium (~20%) that real code limits speedup below 10x.
**Effort:** ~2 person-months.

#### T4: Bounded Completeness (Headline Result)

**Statement:** Within execution depth k and adversary budget n, NegSynth finds every downgrade attack or certifies absence, with concretization success rate ≥ 1−ε. Composes T1, T3, and T5 via three-level simulation chain.

**Critical amendment:** Bounds (k=20, n=5) externally validated by:
- **Coverage metric:** % of reachable negotiation states explored, independently validated by random testing. Target: ≥99%.
- **Empirical bound table:** Minimal (k,n) per historical CVE, demonstrating all known CVEs require k ≤ 15, n ≤ 5.
- **Structural argument:** TLS handshakes complete in ≤10 round trips; SSH in ≤8. k=20 covers 2× protocol depth.

**Honest depth: 4/10.** Composition theorem — transitivity of soundness, analogous to CompCert.
**Proof risk:** Medium (~15%) — risk that ε is too large, making guarantee vacuous. Kill gate: ε > 0.01 triggers investigation.
**Effort:** ~3 person-months (includes empirical validation).

#### T2: Attack Trace Concretizability

**Statement:** Every satisfying SMT assignment can be concretized into an executable byte-level attack trace with success rate ≥ 1−ε, where ε is empirically bounded per library. The CEGAR refinement loop refines failures, converging to a concrete trace or proving the abstract counterexample spurious.

**Disposition (per verifier BC-5):** T2 was present in the crystallized problem (5 person-months). It is logically distinct from T4: T4 assumes concretizability; T2 establishes it. The CEGAR loop's termination and precision (concretization failure → refinement → re-query cycle) is T2's content. T2 feeds into T4 as a dependency.

**Honest depth: 3/10.** Standard CEGAR soundness argument adapted for DY adversary domain. The protocol-specific content is the symbolic-to-concrete framing bridge for TLS/SSH wire formats.
**Proof risk:** Medium (~15%) — CEGAR may not converge tightly on complex cipher-suite interactions.
**Effort:** ~3 person-months.

#### T1: Extraction Soundness

**Statement:** Simulation relation between source-level execution states and state-machine states. Every trace of the extracted state machine corresponds to a feasible execution path.

**Depth: 3/10.** Standard simulation relation, adapted for LLVM IR semantics and the merge operator's state abstractions.
**Effort:** ~2 person-months.

#### T5: SMT Encoding Correctness

**Statement:** The SMT constraint system is equisatisfiable with the composed DY adversary and extracted state machine. Every satisfying assignment = valid attack; every UNSAT = no attack within bounds.

**Depth: 3/10.** Standard theory-combination argument with protocol-specific constructor encoding. Non-trivial part: faithful adversary-knowledge accumulation.
**Effort:** ~2 person-months.

### Extension Theorem

#### C1: Covering-Design Differential Completeness (Extension Only)

**Statement:** For N ≥ 3 libraries with n cipher suites, k versions, and interaction depth d, a covering design of strength t guarantees detection of all pairwise behavioral deviations within B(n,k,t) test configurations.

**Depth: 6/10 — mathematically deepest theorem.** Non-obvious connection between combinatorial design theory and protocol testing. The bound B(n,k,t) is meaningful and tight.

**Explicit limitation (per verifier BC-6):** This is a **testing completeness** guarantee over parameter space, NOT a verification completeness guarantee over execution paths. C1 complements T4 — it does not replace it. C1 is only proved if the differential extension (Phase 2) is built.

**Proof risk:** Medium (~15%) — 3-way interactions may escape pairwise coverage.
**Effort:** ~3 person-months.

### Theorem Summary

| ID | Role | Depth | Load-Bearing? | Effort | Status |
|----|------|:-----:|:---:|:---:|--------|
| T3 | Core tractability | 5/10 | ✅ Critical | 2 p-m | Core |
| T4 | End-to-end guarantee | 4/10 | ✅ Critical | 3 p-m | Core |
| T2 | Concretizability | 3/10 | ✅ Required | 3 p-m | Core |
| T1 | Extraction link | 3/10 | ✅ Required | 2 p-m | Core |
| T5 | Encoding link | 3/10 | ✅ Required | 2 p-m | Core |
| C1 | Differential coverage | 6/10 | ✅ Extension only | 3 p-m | Extension |

**Total proof effort: ~15 person-months** (parallelizable: T1+T5 concurrent, T2 concurrent with T3, T4 last, C1 independent).

---

## 7. Hardest Technical Challenges

### Challenge 1: KLEE + OpenSSL Bitcode Generation (Risk: HIGH, 30%)

**Problem:** OpenSSL's build uses Perl-generated Makefiles, assembly stubs, platform `#define` chains. #1 cause of death for KLEE-based research.

**Mitigation:**
1. Use `wllvm`/`gclang` — known to work (prior KLEE papers, S2E maintained recipes).
2. Budget **2 full months** (not hidden in timeline).
3. Stub assembly routines in `crypto/` with C equivalents — not negotiation-relevant.
4. Start with BoringSSL (cleaner build) as development target.

**Kill gate G0 (Week 4):** Can we symbolically execute `ssl3_choose_cipher` on OpenSSL bitcode? KILL if fails.

### Challenge 2: Slicer Precision (Risk: HIGH, 25%)

**Problem:** If slicer delivers 1-2% of source (~3-7K lines), pipeline works. If 5-10% (~25-50K lines), SMT solving timeouts.

**Mitigation:**
1. Two-phase slicing: coarse static slice → protocol-aware taint refinement.
2. Ground truth: negotiation decision logic empirically lives in `statem_clnt.c` + `statem_srvr.c` + `ssl_ciph.c` ≈ 5K lines out of 500K+.
3. Manual validation: enumerate CVE-reachable functions, verify slicer covers ≥90%.

**Kill gate G1 (Week 6):** Slice ≤10K lines from OpenSSL with ≥90% negotiation coverage. KILL if >15K.

### Challenge 3: Z3 Timeout on DY+SMT Encoding (Risk: MEDIUM-HIGH, 40%)

**Problem:** Combined BV+Arrays+UF+LIA theory at adversary budget n=5 may have thousands of variables. Z3 can spin for hours.

**Mitigation:**
1. Incremental solving via Z3 push/pop stack.
2. Query decomposition: per-round sub-queries connected by shared adversary-knowledge constraints.
3. CEGAR over-approximation: start with UF-only (fast), refine to BV for concretization.
4. Solver diversity: CVC5 fallback.

**Kill gate G2 (Week 10):** Full DY+SMT encoding for FREAK CVE solves in <30 minutes. If >2 hours, reduce adversary budget to n=3.

---

## 8. Risk Matrix

| # | Risk | Prob | Impact | Mitigation | Kill Gate |
|---|------|:----:|:------:|-----------|-----------|
| R1 | KLEE + OpenSSL bitcode fails | 30% | Critical | wllvm, BoringSSL fallback | G0 (W4) |
| R2 | Slicer delivers >5% of source | 25% | Critical | Two-phase slicing, manual validation | G1 (W6) |
| R3 | Z3 timeout on DY+SMT | 40% | High | Incremental, decomposition, CEGAR, CVC5 | G2 (W10) |
| R4 | Merge operator <10x speedup | 20% | High | Honest framing, acceptable at 5x | G2 (W10) |
| R5 | Rust↔C++ FFI soundness | 25% | Medium | Conservative binding design, extensive testing | Ongoing |
| R6 | No new vulnerability found | 50% | Low | Certificates are primary contribution | None |

**Compound probability (per verifier BC-2):**
- P(full vision succeeds) = 0.70 × 0.75 × 0.60 × 0.80 × 0.75 ≈ **0.19 ≈ 19%**
- P(minimal viable paper) ≈ **55%** (see Section 9)
- P(any publication) ≈ **70-80%** (includes minimal scope)
- P(ABANDON at kill gate) ≈ **20-25%**

---

## 9. Minimal Viable Paper (Fallback Scope)

Per verifier binding condition BC-3, the minimal viable paper is explicitly defined:

| Dimension | Full Vision | Minimal Viable Paper |
|-----------|------------|---------------------|
| **Libraries** | OpenSSL, BoringSSL, WolfSSL, libssh2 | OpenSSL only (+ BoringSSL if time) |
| **CVEs recovered** | 8 CVEs across 4 libraries | 3-4 CVEs (FREAK, Logjam, CCS Injection, POODLE) in OpenSSL |
| **Certificates** | 4 libraries, current HEAD | 1 library (OpenSSL 3.x HEAD) |
| **Differential extension** | Full cross-library analysis | Deferred to follow-up paper |
| **Theorems** | T1-T5 + C1 | T1-T5 only |
| **Target venue** | IEEE S&P / USENIX Security | CCS / USENIX Security |
| **P(success)** | ~19% | ~55% |
| **P(best paper)** | 5-10% | 3-5% |

The minimal paper is still strong: first bounded-completeness certificates for OpenSSL, first end-to-end source-to-attack pipeline, 3-4 CVE ground truth, and the merge operator "money plot."

---

## 10. Timeline (18 months, 2-3 person team)

### Phase 0: Foundation (Months 1-2)

| Month | Task | Deliverable | Kill Gate |
|-------|------|------------|-----------|
| M1 | OpenSSL/BoringSSL bitcode via wllvm/gclang; stub assembly; angr spike backup (1 week) | Working bitcode for ≥1 library | — |
| M2 | KLEE integration PoC: symbolically execute `ssl3_choose_cipher` | KLEE runs on negotiation function | **G0 (W4): KLEE executes ssl_ciph.c. KILL if fails.** |

### Phase 1: Core Engine (Months 3-7)

| Month | Task | Deliverable | Kill Gate |
|-------|------|------------|-----------|
| M3 | Protocol-aware slicer v1 | Slice of OpenSSL ≤10K lines | **G1 (W6): Slice ≤10K, ≥90% coverage. KILL if >15K.** |
| M4 | Protocol-aware merge operator v1 | Merge integrated into KLEE searcher | — |
| M5 | Merge validation on OpenSSL | ≥10x path reduction measured | **G2 (W10): ≥10x reduction. Investigate if <5x.** |
| M6 | State machine extractor + DY+SMT encoder v1 | End-to-end on FREAK CVE; Z3 <30min | **G2 (W10): SMT solves <30min.** |
| M7 | CEGAR concretizer + TLS-Attacker replay | First validated attack trace | **G3 (W14): ≥3 CVEs recovered. KILL if <2.** |

### Phase 2: Full Evaluation (Months 8-11)

| Month | Task | Deliverable |
|-------|------|------------|
| M8 | WolfSSL + libssh2 bitcode + slicing | 4 libraries in pipeline |
| M9 | Full CVE recovery (8 CVEs × multiple versions) | Recall measurement |
| M10 | Bounded-completeness certificates, 4 current HEAD libs | Primary empirical contribution |
| M11 | Bound validation table, coverage metric, merge "money plot" | Supporting experiments |

### Phase 3: Differential Extension (Months 12-14)

| Month | Task | Deliverable |
|-------|------|------------|
| M12 | Wire-protocol alignment + covering-design generator | Differential infrastructure |
| M13 | Deviation detection + security ranking + verification | Cross-library results |
| M14 | Cross-library certificates + C1 theorem proof | Extension contribution |

### Phase 4: Paper + Theorems (Months 15-18)

| Month | Task | Deliverable |
|-------|------|------------|
| M15 | T1, T3, T5 proofs finalized; property-based testing | Proof artifacts |
| M16 | T2, T4 composition; C1 proof; empirical ε measurement | Complete theorem chain |
| M17 | Paper draft; money plot Figure 1; artifact packaging | Submission-ready draft |
| M18 | Internal review, revisions, submission | IEEE S&P / USENIX Security |

---

## 11. Best-Paper Argument

### Why This Wins at IEEE S&P / USENIX Security / CCS

1. **Novel end-to-end pipeline with no prior analog.** No existing tool connects C source → protocol-aware slice → symbolic execution with algebraic merge → state machine → DY adversary encoding → concrete attack traces. The composition theorem (T4) guaranteeing end-to-end soundness is unprecedented.

2. **First bounded-completeness certificates for production libraries.** Novel artifacts with immediate practical value. "Within bounds k=20, n=5, covering ≥99% of reachable negotiation states, OpenSSL 3.x contains no cipher-suite downgrade attack."

3. **The "money plot" is Figure 1.** Generic veritesting: exponential path growth with cipher suites. Protocol-aware merge: linear. Visually striking, immediately communicates the contribution. Reproducible on artifact.

4. **Eight CVE recoveries including Terrapin (2023).** Ground-truth validation spanning FREAK (2015) to Terrapin (2023), across TLS + SSH, four libraries.

5. **Cross-library differential analysis (extension)** adds unique cross-library interoperability assurance with covering-design completeness — a dimension no other tool offers.

6. **Artifact as lasting resource.** ~90K lines (50K novel + 40K integration) with automated evaluation, SARIF output, TLS-Attacker-validated replay. Built on KLEE's maintained infrastructure.

### What Could Prevent Best Paper

- No new vulnerability discovered (mitigated: certificates are primary framing)
- T3 dismissed as "just veritesting for TLS" (mitigated: formal comparison "money plot")
- KLEE-based tool seen as incremental (mitigated: DY adversary model is fundamentally different analysis)

---

## 12. Scores (Reconciled per Verifier BC-4)

| Dimension | Score | Rationale |
|-----------|:-----:|-----------|
| **Value** | **7/10** | Proven-lethal problem (Terrapin 2023). Certificates useful to ~50 library maintainers. TLS 1.3 narrows TLS surface (prevents 8+). SSH + legacy TLS + cross-version keep it urgent. Narrow stakeholder pool prevents higher score. |
| **Difficulty** | **7/10** | ~50K novel LoC with hard algorithmic core. KLEE integration genuine. Each component adapts existing techniques to new domain rather than inventing from scratch. Calibrated: between S2E (8/10) and TLS-Attacker (6/10). |
| **Best-Paper Potential** | **7/10** | Novel pipeline + certificates + money plot + 8 CVEs + differential extension. T3 is domain instantiation (depth 5), T4 is composition (depth 4). C1 at depth 6 is deepest but extension-only. P(best-paper): **5-10%** (10-15% with new CVE). |
| **Feasibility** | **7/10** | Kill gates de-risk early. wllvm/gclang known to work. BoringSSL fallback exists. Minimal viable paper at ~55% probability. KLEE and Z3 are battle-tested foundations. Docked because three high-risk components must all succeed for full vision (~19% compound). |

**Composite: (7 + 7 + 7 + 7) / 4 = 7.0/10**

**Probability estimates:**
- P(top-4 venue acceptance): **45-55%** (capped per verifier BC-2)
- P(best paper): **5-10%** (rises to 10-15% with new CVE)
- P(any publication, including minimal scope): **70-80%**
- P(ABANDON at kill gate): **20-25%**

---

## 13. Binding Conditions Addressed

| BC# | Condition | Resolution |
|-----|-----------|-----------|
| BC-1 | Revert LoC to ~50K novel | ✅ Done. All tables show ~50K novel + ~40K integration. |
| BC-2 | Fix compound probability; cap P(acceptance) | ✅ Done. P(full) ≈ 19%. P(acceptance) capped at 45-55%. |
| BC-3 | Define minimal viable paper | ✅ Done. Section 9: OpenSSL only, 3-4 CVEs, T1-T5, ~55% probability. |
| BC-4 | Reconcile scores to depth-check baseline | ✅ Done. V7/D7/BP7/F7, composite 7.0. |
| BC-5 | Restore T2 or document disposition | ✅ Done. T2 restored in Section 6 with honest depth/effort. |
| BC-6 | Label C1 as extension-only | ✅ Done. C1 explicitly marked "Extension Only" throughout. |

---

## 14. Portfolio Differentiation

NegSynth is **clearly distinct** from all 28 existing portfolio projects. The closest potential overlap is `cross-lang-verifier` (cross-language verification), but that targets language interoperability, not protocol negotiation. No other project involves:
- Dolev-Yao adversary models
- TLS/SSH protocol negotiation analysis
- Bounded-completeness certificates for C library source code
- Protocol-aware symbolic execution merge operators
- Cipher-suite downgrade attack synthesis

The domain (cryptographic protocol implementation security), the technique (protocol-aware symbolic execution with adversary models), and the artifacts (bounded-completeness certificates, concrete attack traces) are all unique to this project.

---

## Appendix: Approach Selection Rationale

Three approaches were evaluated via independent expert competition:

| Approach | Composite (pre-debate) | Composite (post-debate) | Verdict |
|----------|:---:|:---:|---------|
| A: Algebraic Merge | 7.0 | 5.5 (Skeptic) | **Core engine selected** — only approach producing standalone certificates |
| B: Abstract Interpretation | 7.25 | 6.0 (Skeptic) | **Discarded** — infeasible (9/10 difficulty, 24-36 months, B3 ornamental) |
| C: Differential Mining | 7.75 | 6.0 (Skeptic) | **Extension adopted** — cross-library value unique, but cannot stand alone |

The winning approach synthesizes A's core engine with C's differential extension, addressing all fatal flaws raised in debate while preserving both unique contributions. Independent verification confirmed this synthesis with CONDITIONAL APPROVE, and all 6 binding conditions have been addressed in this document.
