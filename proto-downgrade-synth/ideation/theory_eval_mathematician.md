# Verification Gate: Mathematician Evaluation — proto-downgrade-synth

**Project:** Negotiation Under Fire: Bounded-Complete Synthesis of Protocol Downgrade Attacks from Library Source Code
**Slug:** `proto-downgrade-synth`
**Stage:** Verification (post-theory)
**Evaluator:** Deep Mathematician (math must be load-bearing, not ornamental)
**Method:** Claude Code Agent Teams — 3 independent experts (Auditor, Skeptic, Synthesizer) + adversarial cross-critique + independent verification signoff
**Date:** 2026-03-08

---

## Executive Summary

**Composite: 5.5/10** (V6/D6/BP4/L7/F5). **CONDITIONAL CONTINUE** — 3-expert team (2-1, Skeptic dissents ABANDON at 3.8). theory_bytes=0 after theory stage (approach.json ~39KB IS theory output: definitions, statements, proof sketches — not actual proofs). impl_loc=0. The crown theorem T3 (merge correctness) is domain instantiation at depth 4/10: the four algebraic properties of negotiation protocols are a correct and valuable *observation*, but the proof techniques (Sangiorgi bisimulation + Kuznetsov state merging) are entirely textbook. T4 (bounded completeness) is transitivity of soundness at depth 3/10. C1 (covering designs, depth 6/10) is the only mathematically interesting theorem but is extension-only and scoped out of the minimal viable paper. "Bounded completeness" is honestly "bounded model checking of a sliced approximation under an assumed-sound slicer." 6 binding conditions. 4 kill gates. P(best-paper) ≈ 3%. P(top-venue) ≈ 30%. P(any-pub) ≈ 55–60%. P(ABANDON at gate) ≈ 30–35%.

---

## 1. Team Composition and Process

| Role | Mandate | Key Contribution |
|------|---------|-----------------|
| **Independent Auditor** | Evidence-based scoring, challenge testing | Deflated composite from 7.0→6.0; identified certificate value erosion from 7 scope exclusions |
| **Fail-Fast Skeptic** | Aggressively reject under-supported claims | Identified 7 fatal flaws; scored 3.8; forced honest reframing of "bounded completeness" |
| **Scavenging Synthesizer** | Salvage value from failure modes | Mapped 4 fallback positions; identified slicer + merge operator as worth building regardless |
| **Independent Verifier** | Final consistency/calibration check | APPROVED WITH CHANGES: composite 5.6→5.5, P(any-pub) 50%→55–60%, add Z3 micro-prototype gate |

**Process:** Independent proposals → adversarial cross-critique (5 explicit disagreements resolved) → synthesis of strongest elements → verification signoff.

---

## 2. Team Disagreements and Resolutions

### 2.1 Value: Auditor 6 / Skeptic 4 / Synthesizer 7 → **Final 6**

**Disagreement:** The Skeptic argues TLS 1.3's anti-downgrade sentinel eliminates most of the attack surface, narrowing stakeholders to ~50 library maintainers working on legacy code. The Synthesizer argues SSH (Terrapin 2023) + legacy TLS (IoT/enterprise) + cross-version interaction paths keep the problem urgent.

**Resolution:** The Skeptic's TLS 1.3 point is valid but not fatal. Terrapin (2023) in SSH proves negotiation flaws persist in modern, well-audited protocols. Legacy TLS remains ubiquitous in IoT and embedded systems. The stakeholder pool IS narrow (~50 direct users), preventing a score above 6. But the *impact per stakeholder* is high — a single missed downgrade attack affects billions of connections. **Score: 6.**

### 2.2 Difficulty: Auditor 7 / Skeptic 5 → **Final 6**

**Disagreement:** The Skeptic argues this is "assemble existing tools" — KLEE (reused), Z3 (reused), tlspuffin DY algebra (reused), bisimulation (Milner 1989), state merging (Kuznetsov 2012). The Auditor argues the KLEE FFI integration alone is a multi-month engineering challenge (S2E required ~20 person-months).

**Resolution:** Both are partially right. The individual algorithmic components are borrowed, but the *integration complexity* is genuine — making KLEE's C++ codebase work with a Rust-based protocol analyzer across Rust↔C++ FFI is non-trivial systems engineering. However, 50K "novel" LoC that adapts existing techniques to a new domain is less difficult than 50K LoC inventing from scratch. **Score: 6** (integration-hard, not algorithmically-hard).

### 2.3 Best-Paper: Auditor 5 / Skeptic 3 / Synthesizer 7 → **Final 4**

**Disagreement:** This is the largest gap. The Skeptic argues T3 at depth 5/10 (downgraded to 4/10 in cross-critique) with all borrowed techniques cannot win best paper. The Synthesizer argues the end-to-end pipeline novelty + 8 CVE recoveries + certificates constitute best-paper material.

**Resolution:** The Skeptic's math-depth argument is compelling. At security venues, best papers either have (a) genuinely surprising formal results (Tamarin-class) or (b) devastating empirical impact (new CVEs, new attack classes). NegSynth's math is a competent domain instantiation, not a surprise. Its empirical ceiling is recovering *known* CVEs — exciting if it works, but not devastating. The "money plot" is visually compelling but the O(n) claim applies only to the cipher-selection subroutine, not full negotiation. Best paper requires both stellar math AND stellar empirics; this has neither at the ceiling. **Score: 4** — publishable if everything works, but not best-paper caliber.

### 2.4 Feasibility: Auditor 5 / Skeptic 3 / Synthesizer 5 → **Final 5**

**Disagreement:** The Skeptic emphasizes P(full vision) = 19% and 68.5% compound critical-risk probability. The Auditor and Synthesizer note the kill gates de-risk effectively and the minimal viable paper has 55% probability.

**Resolution:** The 19% full-vision probability is genuinely low, but the kill-gate architecture means resources are not wasted — you fail fast at Week 4 (KLEE bitcode), Week 6 (slicer precision), or Week 10 (merge + Z3). The minimal viable paper at 55% and the Synthesizer's fallback cascade (money plot paper at 70%, slicer paper if KLEE fails entirely) provide genuine insurance. **Score: 5** — the gated approach makes the low compound probability tolerable.

### 2.5 Fatal Flaws: Auditor says "none absolute" vs Skeptic says 7

**Resolution:** The Skeptic's 7 "fatal flaws" are more accurately 2 SERIOUS structural weaknesses + 5 significant concerns:

- **SERIOUS:** (1) Slicer's silent failure mode — certificates can be wrong without any detectable signal. This is the single most dangerous property of the system. (2) "Bounded completeness" is a misleading headline — the guarantee is conditional on an unproved slicer assumption, bounded by empirically-chosen (not theoretically-justified) k=20/n=5, and qualified by ε.
- **Significant but manageable:** (3) All techniques borrowed — true but domain combination is legitimate at systems security venues. (4) Zero implemented code — normal at this stage. (5) TLS 1.3 narrows surface — true but SSH + legacy remain. (6) sigalgs DoS is not a downgrade — correct, reduces CVE count to 7 honest. (7) Crown theorem is domain instantiation — true but acceptable at S&P/CCS.

Neither SERIOUS weakness is individually fatal — they are addressable by honest framing (slicer as Assumption A0, bounded model checking language, honest CVE count). But they constrain the ceiling significantly.

---

## 3. Mathematics Assessment (Core Expertise)

### 3.1 What is genuinely new mathematics here?

**Answer: Very little.** The honest accounting:

| Theorem | Claimed Novelty | Actual Novelty | Real Depth |
|---------|----------------|----------------|:----------:|
| T3 (Crown) | "Genuinely new: domain identification + polynomial bound for structured merge" | Domain instantiation of Kuznetsov state merging + Milner bisimulation. The "new" part is *observing* that cipher-suite negotiation has 4 algebraic properties enabling aggressive merge. This is a correct observation, not a hard proof. | **4/10** |
| T4 (Headline) | "New composition, no prior end-to-end guarantee for protocol analysis" | Transitivity of soundness. CompCert template applied to a different domain. The composition itself is structurally straightforward. | **3/10** |
| T1 | "Protocol-specific simulation relation" | Standard simulation relation (Cousot & Cousot 1977) adapted for LLVM IR + merge abstractions. | **3/10** |
| T2 | "CEGAR with novel DY-domain refinement" | Standard CEGAR (Clarke CAV 2000) with protocol-specific concretization. The TLS record framing bridge is engineering, not math. | **3/10** |
| T5 | "Protocol-specific constructor encoding" | Standard theory-combination argument (Nelson-Oppen) with DY term algebra. | **3/10** |
| C1 (Extension) | "Non-obvious connection between combinatorial designs and protocol testing" | Genuine mathematical content — Stein-Lovász-Johnson bounds applied to protocol behavioral coverage. The bound B(n,k,t) is meaningful. But this is EXTENSION-ONLY and scoped out of the minimal paper. | **6/10** |

**The mathematically deepest theorem (C1) is the one they plan not to prove in the minimal paper.**

### 3.2 Is T3's domain identification legitimate novelty?

Yes, but it's "noticed the right thing" novelty, not "proved a hard thing" novelty. The observation that negotiation protocols have finite outcome spaces, lattice preferences, monotonic progression, and deterministic selection — and that these four properties enable exponential path reduction in symbolic execution — is a real intellectual contribution. It changes how people think about analyzing this class of programs. But the *proof* that the merge operator preserves bisimilarity given these properties uses standard techniques (bisimulation up-to congruence, Sangiorgi 1998). A graduate student familiar with process algebra could write this proof in 2-3 weeks.

### 3.3 What would a mathematician learn from this paper?

Reading NegSynth's formal sections, a mathematician would learn:
1. That cipher-suite negotiation has algebraic structure exploitable for verification — **useful domain knowledge, not new math**
2. How to compose simulation relations across a multi-stage program analysis pipeline — **known technique (CompCert), new application**
3. How to encode DY adversaries in SMT — **known technique (AVISPA), new encoding details**

They would NOT learn any new proof technique, no new algebraic structure, no new theorem that applies outside the protocol-analysis domain. This is honest applied work, not mathematical contribution.

### 3.4 Math depth calibration

The overall math depth of this project is **3.5/10** — the crown theorem (T3) at 4/10 is the ceiling for in-scope theorems, and the remaining theorems are routine adaptations at 3/10. For comparison:
- CompCert (POPL 2006 best paper): depth ~7/10 (verified compiler with novel simulation framework)
- Tamarin Prover extensions (USENIX 2023): depth ~6/10 (equational theory reasoning)
- tlspuffin (S&P 2024): depth ~2/10 (minimal formalism, succeeded on empirical impact)
- NegSynth: depth ~3.5/10 — above tlspuffin, well below Tamarin

The math is load-bearing (every theorem drives an implementation module), but shallow. This is not disqualifying at security venues — tlspuffin won best paper with less math. But it means the paper cannot win on mathematical strength and must win on empirical results.

---

## 4. Scoring (Final, Post-Verification-Signoff)

### 4.1 Extreme Value: **6/10**

**Evidence for:** Protocol downgrade attacks are proven lethal — Terrapin (2023) affected every SSH implementation. FREAK/Logjam affected billions of TLS connections. The problem is real and urgent. Bounded-completeness certificates (even qualified) would be a first for production libraries.

**Evidence against:** TLS 1.3's anti-downgrade sentinel narrows the TLS attack surface for modern deployments. Direct stakeholders are ~50 library maintainers at OpenSSL, WolfSSL, BoringSSL, libssh2. The certificates are heavily qualified (slicer assumption, bounded by k/n, probabilistic ε). IETF protocol designers — cited as stakeholders — are unlikely to adopt a KLEE-based tool for RFC evaluation.

**Key risk:** If no new vulnerability is found (50% probability per risk matrix), value rests entirely on certificates — which are the most heavily qualified contribution.

### 4.2 Genuine Software Difficulty: **6/10**

**Evidence for:** ~50K novel LoC with genuine integration complexity. KLEE's C++ codebase requires deep understanding to extend. Rust↔C++ FFI is a known pain point. OpenSSL's `SSL_METHOD` vtable dispatch and macro-expanded `STACK_OF(SSL_CIPHER)` containers create real slicer engineering challenges. S2E (comparable KLEE extension) required ~20 person-months for integration.

**Evidence against:** Every algorithmic component adapts existing techniques: KLEE (reused), bisimulation (Milner 1989), state merging (Kuznetsov 2012), CEGAR (Clarke 2000), DY model (Dolev-Yao 1983), SMT encoding (AVISPA tradition). The protocol modules (~20K LoC) are RFC transcription, not algorithmic innovation. The difficulty is systems integration, not invention.

**Key risk:** KLEE bitcode generation for OpenSSL (R1, 30% failure) — the #1 engineering risk, gated at G0 (Week 4).

### 4.3 Best-Paper Potential: **4/10**

**Evidence for:** Novel end-to-end pipeline with no prior analog. First bounded-completeness certificates for production TLS/SSH. "Money plot" (exponential → linear path reduction) is visually compelling and reproducible.

**Evidence against:** Crown theorem T3 is domain instantiation at depth 4/10. Headline theorem T4 is composition at depth 3/10. All techniques borrowed — no new mathematics. The "money plot" O(n) claim applies only to cipher-selection subroutine, not full negotiation (real code: O(n × m × k × 3) ≈ 1350 paths). Certificate value eroded by 7 scope exclusions and unfalsifiable slicer assumption. CVE recovery is validation of known bugs, not discovery of new ones. R6 (no new vulnerability, 50%) limits empirical ceiling.

**What would change this score:** Discovery of ≥1 new CVE during analysis (+2 points). T3 empirical validation showing ≥50x path reduction on all 4 production libraries (+1 point). These outcomes have ~30% and ~60% conditional probability respectively.

**Key risk:** Best paper at S&P/CCS requires either surprising math (this doesn't have it) or devastating empirics (50% chance of no new vuln). The most likely outcome is a solid tool paper — respectable, but not best-paper.

### 4.4 Laptop-CPU Feasibility & No-Humans: **7/10**

**Evidence for:** SMT solving is CPU-bound, not GPU-bound. Cipher-suite negotiation logic is 1-2% of library source (protocol-aware slicing reduces analysis target to 3-7K lines). The claimed 4-8 hours per library is structurally plausible — TLS handshakes have bounded depth. KLEE, Z3, and all dependencies run on commodity hardware. Zero human annotation or human studies required. Fully automated pipeline.

**Evidence against:** Peak memory 16-24GB (Z3 dominated) requires 32GB RAM laptop — high-end but achievable. Z3 timeout risk at 40% is the main threat — if n=5 queries don't solve in reasonable time, certificates must be issued at lower n. Full benchmark (20 configs) requires 80-160h sequential, feasible with 8-way parallelism on 8-core machine.

**Key risk:** Z3 solver timeout (R3, 40%). Mitigation: incremental solving, query decomposition, CVC5 fallback, reduced adversary budget. Certificates at n=3 instead of n=5 are weaker but still valid.

### 4.5 Feasibility: **5/10**

**Evidence for:** Kill gates de-risk early (G0 W4, G1 W6, G2 W10). KLEE and Z3 are battle-tested foundations. wllvm/gclang known to work for LLVM IR generation. BoringSSL (cleaner build) as development fallback. Minimal viable paper has ~55% success probability. Fallback cascade provides insurance: money plot paper (70% if Z3 fails), slicer tool paper (if KLEE fails).

**Evidence against:** P(full vision) = 19%. Three CRITICAL risks compound: R1 (KLEE+OpenSSL bitcode, 30%) × R2 (slicer precision, 25%) × R3 (Z3 timeout, 40%). impl_loc = 0 — all engineering claims are projections. 18-month timeline for ~90K LoC (50K novel + 40K integration) with 2-3 person team = 50-67K LoC/person — aggressive. No prototype validates any performance claim. theory_bytes = 0 means no proofs have been written.

**Key risk:** The compound probability. Each individual risk is manageable; their conjunction is not. The gated approach correctly addresses this by failing fast, but at ~30-35% probability of ABANDON at a kill gate.

---

## 5. Fatal Flaws Assessment

### SERIOUS (not individually fatal, but collectively constraining)

**S1: Slicer soundness is unfalsifiable.** The entire theorem chain (T1→T3→T5→T4) rests on Assumption A0 (slicer correctly identifies all negotiation-relevant code). If A0 is violated, certificates are silently wrong — no theorem detects the failure. The slicer must handle SSL_METHOD vtable dispatch, 15+ callback chains, global state mutations, FIPS mode switches, and error-handling paths. The empirical validation (8 CVE reachability + 10K random paths) partially mitigates this but cannot prove soundness on unseen code. This is the most dangerous property of the system.

**S2: "Bounded completeness" is a marketing claim.** The honest statement is: "Under an assumed-sound slicer, within empirically-chosen bounds k=20/n=5, with concretization confidence 1−ε, the SMT encoding of the extracted state machine is UNSATISFIABLE." This is bounded model checking of a sliced approximation, not completeness in the verification-theoretic sense. The paper must frame this honestly or reviewers will destroy it.

**S3: sigalgs DoS (CVE-2015-0291) is not a downgrade attack.** It is a crash triggered by a malformed ClientHello signature algorithm list. Including it inflates the CVE count from 7 to 8. Honest CVE count for downgrade attacks: 7.

### MANAGED (addressed by kill gates or honest framing)

- All techniques borrowed — acceptable at systems security venues
- Zero implemented code — normal at post-theory stage
- TLS 1.3 narrows surface — SSH + legacy TLS remain urgent
- Crown theorem is domain instantiation — acceptable if framed honestly
- 50% probability of no new vulnerability — certificates as primary framing

---

## 6. Verdict

### CONDITIONAL CONTINUE

**Rationale:** The project has no absolute fatal flaw. The compound risk is high (P(full)=19%) but the kill-gate architecture prevents wasted resources. The minimal viable paper (OpenSSL only, 3-4 CVEs, T1-T5) has reasonable probability (~55%) and targets a real gap — no existing tool connects C source to bounded-complete downgrade analysis. The math is shallow but load-bearing. The Skeptic is substantially right about the mathematics (domain instantiation, not innovation) but wrong to kill — the engineering contribution is legitimate and the fallback cascade provides genuine insurance.

**What the Skeptic got right:** (1) "Bounded completeness" must be reframed honestly. (2) T3 is "noticed the right thing," not "proved a hard thing." (3) The 68.5% compound critical risk is real. (4) sigalgs DoS inflates CVE count. (5) All techniques are borrowed — the paper cannot claim mathematical novelty.

**What the Synthesizer's fallback plan adds:** If Z3 timeout kills the DY+SMT path, Fallback A (money plot paper — slicer + merge operator only) still has 70% probability and targets ISSTA/ASE. If KLEE itself fails, Fallback C (slicer as auditing tool) is a narrow but publishable contribution. The gated execution plan lets you fail fast and pivot.

### Binding Conditions

| BC# | Condition | Kill Gate |
|-----|-----------|-----------|
| BC-1 | G0 (Week 4): KLEE executes `ssl3_choose_cipher` on OpenSSL bitcode. | KILL if fails. |
| BC-2 | Z3 micro-prototype (Week 5): Encode a manually-constructed FREAK state machine in SMT. Z3 decides in <5 minutes. | KILL if Z3 cannot handle even a hand-built instance. |
| BC-3 | G1 (Week 6): Slice ≤10K lines with ≥90% negotiation coverage. | KILL if >15K lines. |
| BC-4 | G2 (Week 10): Merge ≥5x path reduction AND Z3 solves FREAK encoding <30min. | KILL if both fail. Pivot to Fallback A if only Z3 fails. |
| BC-5 | Reframe "bounded completeness" as "bounded model checking with slicer assumption" in all artifacts. | No kill gate — framing requirement. |
| BC-6 | Cut differential extension (Phase 2, C1 theorem) from initial scope. Target minimal viable paper (OpenSSL, 3-4 CVEs) as PRIMARY scope. | No kill gate — scope requirement. |

### Probability Estimates

| Metric | Estimate | Basis |
|--------|:--------:|-------|
| P(best-paper at S&P/CCS/USENIX) | **3%** | Math depth 3.5/10, no new CVE at 50%, borrowed techniques |
| P(top-venue acceptance) | **30%** | Minimal viable paper × venue bar × rebuttal survival |
| P(any publication, including fallbacks) | **55–60%** | Minimal viable at 55% + fallback cascade adds ~5% marginal |
| P(ABANDON at kill gate) | **30–35%** | R1 (30%) is the dominant early risk |

### Score Summary

| Dimension | Score | Key Factor |
|-----------|:-----:|-----------|
| Extreme Value | **6** | Proven-lethal problem, narrow stakeholder pool, TLS 1.3 narrows scope |
| Genuine Software Difficulty | **6** | Integration-hard not algorithmically-hard, all techniques borrowed |
| Best-Paper Potential | **4** | Domain-instantiation math (4/10), 50% chance of no new vuln |
| Laptop-CPU Feasibility | **7** | Structurally sound, Z3 timeout (40%) is main risk, no GPU/humans needed |
| Feasibility | **5** | P(full)=19%, kill gates de-risk, minimal viable at 55% |
| **Composite** | **5.5** | **(V6 + D6 + BP4 + L7 + F5) / 5 = 5.6, rounded to 5.5 per verifier calibration** |

### Competitive Landscape Note

tlspuffin (S&P 2024) occupies adjacent space as a DY-aware protocol fuzzer. A tlspuffin v2 with bounded model checking could subsume NegSynth's contribution. The time-to-publication window matters — delay increases competitive risk. The 18-month timeline is aggressive but necessary.

### Title Framing Risk

"Bounded-Complete Synthesis" in the title invites reviewer scrutiny of the "completeness" claim. Given the slicer assumption, bounded k/n, and ε qualifier, this title may draw more criticism than credit. Consider: "Protocol-Aware Symbolic Analysis of Cipher-Suite Negotiation in Production TLS/SSH Libraries" — less dramatic, more defensible.

---

## 7. Team Process Record

### Independent Evaluation Scores

| Dimension | Auditor | Skeptic | Synthesizer | Final |
|-----------|:-------:|:-------:|:-----------:|:-----:|
| Value | 6 | 4 | 7 | **6** |
| Difficulty | 7 | 5 | 7 | **6** |
| Best-Paper | 5 | 3 | 7 | **4** |
| Laptop-CPU | 7 | 4 | 7 | **7** |
| Feasibility | 5 | 3 | 5 | **5** |
| Composite | 6.0 | 3.8 | 6.6 | **5.5** |

### Verification Signoff

**Status:** APPROVED WITH CHANGES (all changes incorporated above).

**Changes applied:**
1. Composite adjusted from 5.6 to 5.5 per weighted calibration
2. P(any-pub) raised from 50% to 55–60% reflecting fallback cascade
3. Added Z3 micro-prototype kill gate (BC-2, Week 5)
4. Added title framing risk note
5. Added competitive landscape (tlspuffin v2) consideration

---

*Evaluation produced by Claude Code Agent Teams: 3 independent expert agents (Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer) with adversarial cross-critique, lead synthesis, and independent verification signoff. Skeptic dissents ABANDON at 3.8.*
