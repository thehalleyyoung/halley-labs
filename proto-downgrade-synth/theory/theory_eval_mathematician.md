# Final Mathematician's Evaluation: NegSynth

**Title:** "Negotiation Under Fire: Bounded-Complete Synthesis of Protocol Downgrade Attacks from Library Source Code"  
**Role:** Team Lead — Adversarial Cross-Critique & Final Synthesis  
**Evaluator Stance:** Deep mathematician evaluating quantity and quality of NEW, load-bearing mathematics  
**Date:** 2026-03-08  
**Inputs:** Independent Auditor (6.0), Fail-Fast Skeptic (3.8), Scavenging Synthesizer (6.0/5.5 minimal)  
**Status:** BINDING DECISION DOCUMENT

---

## 1. Adversarial Cross-Critique: Disagreement Resolution

### 1.1 Value: Auditor 6 vs. Skeptic 4 vs. Synthesizer 7

**The disagreement:** The Auditor sees genuine value eroded by 7 scope exclusions. The Skeptic argues TLS 1.3 eliminates most attack surface and the audience is ~50 library maintainers. The Synthesizer sees full-scope value at 7.

**Evidence assessment:**

The Skeptic's TLS 1.3 argument is factually correct but strategically incomplete. TLS 1.3 did not eliminate the problem — it displaced it. Terrapin (SSH, 2023) proves that negotiation flaws persist in modern, well-audited code *in other protocol families*. The real question is: does the scope-excluded certificate (no timing, no oracles, no multi-renegotiation, no dynamic providers, no cross-session, no n>5, no DoS) still say something useful?

After the 7 scope exclusions documented in the critique synthesis, the certificate says: "Within a single handshake of a statically-linked build, using the narrow observation function (cipher + version + extensions), with depth ≤20 and ≤5 adversary actions, under Assumption A0 (slicer soundness), no negotiation-outcome downgrade exists." This is a real statement — it covers FREAK, Logjam, Terrapin's core mechanism, CCS Injection — but it is not the headline "OpenSSL has no downgrade attacks." The Skeptic is right that the audience is narrow. The Synthesizer is right that narrow audiences can make high-impact papers (see: CompCert's audience of ~100 verified-compiler researchers).

**Resolution: Value = 6.** The Auditor's score is correct. The Skeptic under-weights the SSH/legacy value; the Synthesizer over-weights it by not pricing in how heavily the certificate must be qualified. A certificate with 7 caveats is still valuable — but a reviewer who reads the caveats will dock "immediate impact" accordingly.

---

### 1.2 Difficulty: Auditor 7 vs. Skeptic 5

**The disagreement:** The Auditor sees genuinely hard systems work (~50K novel LoC). The Skeptic sees "assemble existing tools" — KLEE + Z3 + some glue.

**Evidence assessment:**

The Skeptic's framing as "existing tools" is reductive but contains a core truth: every major algorithmic subsystem has a published template. The slicer follows Weiser (1984) instantiated with Andersen-style points-to analysis (SVF exists). The merge operator adapts Kuznetsov et al. (2012) veritesting with domain-specific mergeability predicates. The DY encoding follows AVISPA-style SMT encoding. The CEGAR loop is textbook Clarke et al. (2000).

However, the Skeptic underestimates *integration difficulty*. Building on KLEE's ~95K LoC C++ codebase, making it cooperate with a custom Rust merge operator via FFI, handling OpenSSL's macro-generated vtable dispatch in the slicer, and ensuring the entire composition chain maintains correctness guarantees across 5 subsystems — this is genuinely hard systems work. It's not "assemble," it's "adapt, integrate, and formally validate the composition." Most systems papers at top venues do exactly this.

**Resolution: Difficulty = 6.** Neither extreme is right. This is harder than "assemble" (5) but easier than "fundamental new algorithm" (8). The difficulty is real but is primarily *integration and domain instantiation difficulty*, not *algorithmic invention difficulty*. The deepest algorithmic idea — that negotiation protocols have four properties enabling efficient merging — is an observation (correct and valuable), not a technique. I dock one point from the Auditor's 7 because the 50K novel LoC claim, even after honest reframing, is generous: protocol modules (~20K LoC) are RFC transcription, not algorithmic work.

---

### 1.3 Best-Paper: Auditor 5 vs. Skeptic 3 vs. Synthesizer 7

**The disagreement:** This is the largest disagreement (4-point spread). The Synthesizer sees a novel end-to-end pipeline with certificates as first-of-kind artifacts. The Skeptic sees borrowed techniques at depth 5/10. The Auditor splits the difference.

**Evidence assessment:**

I must be precise about what "best paper" means at IEEE S&P / USENIX Security / ACM CCS. Best papers at these venues have historically been:
- **Finding real bugs** with a novel tool (KLEE @ OSDI 2008, tlspuffin @ S&P 2024)
- **Breaking deployed systems** (FREAK, POODLE, Terrapin disclosures)
- **Proving surprising theorems** about security properties (rarely)

NegSynth's best-paper case rests on certificates (novel artifacts) + CVE recovery (validation, not new bugs) + the merge operator (domain-specific optimization). The certificates are genuinely first-of-kind, but their value requires a reader to trust the entire pipeline — including Assumption A0. A reader who notices A0 will view the certificate as "conditional on an unproved assumption about a heuristic slicer," which substantially deflates the wow factor.

The Skeptic is right that T3's crown theorem at difficulty 5/10 is unlikely to generate best-paper excitement. The Synthesizer is right that the end-to-end composition is unprecedented. But "unprecedented composition of known techniques" is not how best papers are typically won — it's how solid accepts are won.

**Resolution: Best-Paper = 4.** I lean toward the Skeptic but not as far as 3. The 4 reflects: (a) no genuinely new vulnerability likely to be found, (b) the crown theorem is domain instantiation, (c) certificates are conditional on A0, but (d) the end-to-end pipeline IS novel as a composition and the certificates ARE first-of-kind artifacts, even if conditional. A 4 means "possible but unlikely" — roughly 3-5% probability.

---

### 1.4 Feasibility: Auditor 5 vs. Skeptic 3 vs. Synthesizer 5

**The disagreement:** The Skeptic computes 68.5% compound critical-risk probability (R1×R2×R3) and 19% success probability. The Auditor and Synthesizer both land at 5, arguing the gated execution plan de-risks.

**Evidence assessment:**

The Skeptic's compound probability calculation treats risks as independent, which the red-team's own critique (A4.3) notes they are NOT — slicer failure correlates with merge unsoundness and SMT difficulty. However, the Skeptic's mistake cuts *both ways*: correlated failure means either everything works or everything fails, which makes gated execution MORE effective (early gates catch correlated failures fast).

The real question: what's P(passing all gates)?

- G0 (KLEE on ssl_ciph.c, Week 2): ~85%. KLEE handles C well; ssl_ciph.c is ~2K lines.
- G1 (Slicer ≤15K lines, Week 4): ~70%. The red-team's 25% failure risk is credible.
- G2 (Merge ≥10× speedup, Week 6): ~75%. The domain structure IS favorable; the question is whether real code cooperates.
- G3 (≥3 CVEs recovered, Week 10): ~60%. FREAK and Logjam are straightforward; third CVE is the test.

Compound: 0.85 × 0.70 × 0.75 × 0.60 ≈ **27%** for the full pipeline. But the Synthesizer's fallback plans matter: if G2 fails partially (5× not 10×), the money-plot paper is still possible. If G3 fails at 3 CVEs but gets 2, the fallback slicer-as-auditing-tool paper is possible.

P(some publishable paper) ≈ 50%. P(full headline paper) ≈ 27%.

**Resolution: Feasibility = 5.** The Auditor and Synthesizer are right that 5 is appropriate when accounting for fallbacks. The Skeptic's 3 is too pessimistic because it ignores the fallback plan's genuine insurance value. But 5 is not comfortable — it means a coin flip for any publication.

---

### 1.5 Fatal Flaws: Auditor "none absolute" vs. Skeptic "7 fatal"

**Evidence assessment, flaw by flaw:**

| # | Skeptic's Flaw | Truly Fatal? | Resolution |
|---|----------------|:---:|------------|
| 1 | Slicer's silent failure mode makes certificates unfalsifiable | **No** — manageable | Honest framing as Assumption A0. Every verification tool has a trusted computing base. The slicer IS the TCB. State it. |
| 2 | 68.5% compound critical-risk probability | **No** — miscomputed | Risks are correlated, gated execution catches failures early. Real compound success ≈ 27%. |
| 3 | "Bounded completeness" is bounded model checking of a sliced approximation | **Correct characterization** — but not fatal | This IS what the paper does. "Bounded model checking of a protocol-aware slice with Dolev-Yao adversary" is still a contribution. The marketing must match the math. |
| 4 | T3 is domain instantiation (depth 5/10) | **Correct** — but not fatal | Domain instantiation is how most good systems papers work. The question is whether the domain insight is non-trivial. It is — four algebraic properties IS the right observation. |
| 5 | Zero implemented code | **Correct** — highest actual risk | All claims are projections. This is the standard risk of any pre-implementation evaluation. Gated execution is the mitigation. |
| 6 | TLS 1.3 eliminates most attack surface | **Partially correct** — not fatal | SSH + legacy TLS + cross-version paths remain. Surface is narrower than advertised but not zero. |
| 7 | sigalgs DoS is not a downgrade attack | **Correct** — but minor | Reduces CVE count to 7. The Auditor's scope table already reclassified this. |

**Resolution: 0 absolutely fatal, 3 correctly identified weaknesses (flaws 3, 4, 5) that constrain the recommendation.** The Skeptic's contribution is essential: flaw #3 should become the paper's honest framing, flaw #4 constrains the best-paper score, and flaw #5 is the dominant risk.

---

## 2. Mathematics Assessment (Core Expertise)

This is the section I weight most heavily. The question is: **what new mathematics does this paper require that a competent researcher couldn't derive in an afternoon from known results?**

### 2.1 T3 (Merge Correctness): Novel or Domain Instantiation?

**Honest assessment: Domain instantiation with a non-trivial observation.**

T3's proof uses bisimulation up-to congruence (Sangiorgi 1998), applied to a merge operator whose construction follows the template of Kuznetsov et al. (2012). The novelty is the *mergeability predicate* specialized to protocol structure, and the *complexity argument* that four algebraic properties (A1–A4) reduce the path count from O(2^n) to O(n·m).

Let me check each component for genuine novelty:

- **Bisimulation up-to congruence:** Textbook (Sangiorgi & Walker, "The Pi-Calculus," 2001). No new proof technique.
- **ITE merge construction:** Standard in symbolic execution (Kuznetsov et al., PLDI 2012). No new data structure.
- **Observation that negotiation protocols have finite outcome spaces, lattice preferences, monotonic progression, deterministic selection:** This IS the new content. But is it *mathematics* or is it a *domain observation*?

I judge it as a **domain observation that enables a clean mathematical argument**. The four properties are not mathematically deep — they are structural features of a well-understood application domain. A protocol researcher would recognize all four immediately. A verification researcher would recognize that these properties make merging trivial. The contribution is **connecting these two observations** — a bridging insight between the protocol and verification communities.

This is valuable but scores as **depth 4/10 on a pure math scale**. It's "noticed the right structural correspondence" — important for engineering, but the proof itself writes itself once you see the observation.

**The formal proposal's own self-assessment (5/10) is slightly generous.**

### 2.2 T4 (Bounded Completeness): Composition Theorem or Transitivity?

**Honest assessment: Transitivity of soundness with careful definition alignment.**

T4 composes T1 → T3 → T5 by transitivity:
1. Source → LTS (T1, simulation)
2. LTS → Merged LTS (T3, bisimulation)
3. Merged LTS → SMT formula (T5, equisatisfiability)

The composition yields: attack in source ↔ SAT formula. This is correct and important, but the proof is:

```
T4: By T1, source attacks ⊆ LTS attacks.
    By T3, LTS attacks = Merged LTS attacks.
    By T5, Merged LTS attacks ↔ SAT(Φ).
    Therefore: source attacks ⊆ SAT(Φ).
    For the other direction: by T1 part 2 + T3 + T5.
```

This is **exactly transitivity of soundness across three stages**. The Skeptic's characterization ("just composition") is accurate. The formal proposal's self-assessment (4/10) is also accurate — the hard work is in the components, not the composition.

The subtle part — ensuring definition alignment across three intermediate representations — is important engineering but not important mathematics. CompCert had the same structure, and CompCert's novelty was the components, not the composition lemma.

**Depth: 3/10.** The composition is load-bearing (the paper needs it) but not mathematically deep.

### 2.3 C1 (Covering Designs): The Only Deep Theorem — But Extension-Only

**Honest assessment: This IS the mathematically deepest theorem — and it's scoped as future work.**

C1 uses the Stein-Lovász-Johnson bound to guarantee that a covering design of strength t detects all t-way behavioral deviations between libraries. The connection between covering strength and protocol parameter interaction order is non-obvious and requires a genuine detection argument:

*Claim:* If libraries ℓ_i and ℓ_j differ on any configuration involving ≤t interacting parameters, a t-covering design contains a test configuration exposing the deviation.

This requires proving that behavioral deviations in protocol negotiation are captured by parameter interactions of bounded order — which is a non-trivial claim about the structure of negotiation functions. The proof needs to show that cipher-suite selection depends on a bounded number of parameters per decision point, and that the covering property catches all such interactions.

**Depth: 6/10.** This is genuine combinatorics applied to a protocol testing domain. The bound is tight, the detection argument is non-trivial, and the connection to protocol structure is original.

**But it's extension-only.** The critique synthesis correctly notes this is Phase 2. The core paper does not include C1. This means the paper's deepest mathematical result is excluded from scope.

### 2.4 The Real Math Depth of the Deepest In-Scope Theorem

Ranking in-scope theorems by genuine mathematical depth:

| Theorem | Honest Depth | Why |
|---------|:---:|-----|
| T3 (merge correctness) | 4/10 | Domain observation + standard bisimulation proof |
| T5 (SMT encoding) | 3/10 | Standard encoding-decoding bijection in a new domain |
| T4 (bounded completeness) | 3/10 | Transitivity of soundness |
| T2 (concretizability) | 3/10 | Standard CEGAR in a new domain |
| T1 (extraction soundness) | 3/10 | Standard simulation relation |
| L1 (merge congruence) | 2/10 | Follows from definitions |
| L3 (slicer soundness) | 1/10 (as stated) | Stated as assumption, not proved |

**Deepest in-scope theorem: T3 at depth 4/10.**

For reference, depth 4/10 means: "a graduate student who has read Kuznetsov et al. and Sangiorgi could write this proof in 2-3 weeks, given the domain observation." This is not a criticism — many excellent systems papers have their deepest theorem at depth 4-5/10. But it constrains the best-paper argument, which requires either surprising mathematics (depth 7+) or devastating empirical results (new CVE in BoringSSL).

### 2.5 What New Mathematics Exists Here?

**The honest answer:**

A mathematician reading this paper would learn:

1. **The four algebraic properties of negotiation protocols (A1–A4)** — a clean structural observation that has not been stated formally before, even though each property is individually obvious to protocol researchers.

2. **That these properties suffice for polynomial-time symbolic merging** — the connection between protocol structure and verification tractability is the paper's mathematical core.

3. **Nothing else that couldn't be derived from Kuznetsov + Milner + Clarke separately.** The proof techniques are entirely standard. The composition structure is standard (CompCert-style). The DY encoding follows AVISPA. The CEGAR loop follows Clarke et al.

**What's genuinely new is the BRIDGE between protocol structure and verification technique** — not a new proof method, not a new data structure, not a new algorithm, but a new domain observation that enables known techniques to work on a previously-intractable problem class.

This is the paper's real contribution, and it is a legitimate one. The question is whether it's enough for best-paper. My assessment: it is enough for a solid accept at a top-4 venue, but not for best paper.

---

## 3. Final Synthesized Scores

### 3.1 Extreme Value: **6/10**

**Evidence:**
- (+) Protocol downgrade attacks are real, lethal (Terrapin 2023), and structurally unsolved
- (+) No existing tool closes the specification-implementation gap for negotiation logic
- (+) Bounded-completeness certificates are first-of-kind artifacts
- (−) 7 scope exclusions heavily qualify the certificate
- (−) TLS 1.3 progressively eliminates the TLS-specific surface
- (−) Audience is ~50 library maintainers + security auditing firms — high impact per user, low total users
- (−) sigalgs DoS CVE (CVE-2015-0291) is not a downgrade attack, reducing clean CVE count to 7

**Calibration:** CompCert (audience: ~100 verified-compiler researchers) scored ~7 on value because its certificates were unconditional. NegSynth's certificates are conditional on Assumption A0 — which is the difference between 7 and 6.

### 3.2 Genuine Software Difficulty: **6/10**

**Evidence:**
- (+) ~50K novel protocol-analysis code is substantial
- (+) KLEE integration is genuinely hard (C++ FFI, custom Searcher, bitcode compatibility)
- (+) Five subsystems must maintain formal correctness guarantees across integration boundaries
- (+) Protocol-aware slicer handling OpenSSL's macro-generated vtables is non-trivial engineering
- (−) Every subsystem has a published algorithmic template (Weiser slicer, Kuznetsov merge, Clarke CEGAR, AVISPA DY encoding)
- (−) 20K LoC of protocol modules are RFC transcription, not algorithmic work
- (−) The deepest technical challenge is *integration* and *domain instantiation*, not *invention*

**Calibration:** A "7" would require at least one subsystem with no published template. Everything here has a template. The difficulty is making them work together on real code — which is real but is a "6" not a "7."

### 3.3 Best-Paper Potential: **4/10**

**Evidence:**
- (+) End-to-end pipeline (source → attack traces) has no prior analog
- (+) Bounded-completeness certificates are first-of-kind
- (+) Recovery of 7+ historical CVEs is strong validation
- (−) Crown theorem T3 is domain instantiation at depth 4/10
- (−) No genuinely new mathematical technique — all proofs follow known templates
- (−) Finding a new CVE is a bet on the external world (TLS 1.3 libraries are heavily audited)
- (−) Certificate value is conditional on Assumption A0 (unproved slicer)
- (−) The "money plot" (O(n) vs O(2^n)) applies to the cipher-selection subroutine only, not full negotiation

**Calibration:** Best papers at S&P/CCS require either a surprising theorem or a devastating empirical result. This paper has neither — it has a clean domain observation and solid engineering. That's a strong accept, not a best paper. P(best-paper) ≈ 3%.

### 3.4 Laptop-CPU Feasibility & No-Humans: **7/10**

**Evidence:**
- (+) SMT solving is inherently sequential — no GPU needed
- (+) Protocol-aware slicing genuinely reduces 200K → 3-7K lines (empirically grounded)
- (+) Z3 handles bounded BV+Arrays+UF problems efficiently at modest variable counts
- (+) Evaluation requires zero human annotation once protocol modules are authored
- (+) Gated execution enables fail-fast before full resource commitment
- (−) 16-32 GB RAM is tight; Z3 peak memory is unpredictable
- (−) 40% Z3 timeout risk at n=5 (red-team estimate) is concerning
- (−) Assembly stub authoring is a one-time manual cost

**Calibration:** The architecture is sound for laptop CPU. The risks are about *whether it works* (feasibility), not *whether it needs special hardware* (laptop-CPU). Score reflects that the hardware constraint is naturally satisfied.

### 3.5 Feasibility: **5/10**

**Evidence:**
- (+) Gated execution (G0-G4) enables fail-fast with clear kill criteria
- (+) Fallback plans (money-plot paper, slicer-as-auditing-tool) provide insurance
- (+) Each subsystem has a published template, reducing algorithmic risk
- (+) C-only scope (KLEE + LLVM) is a mature ecosystem
- (−) Zero implemented code — all estimates are projections
- (−) Compound probability through all gates ≈ 27% for full paper
- (−) P(some publication) ≈ 50% including fallbacks
- (−) Slicer soundness (the pipeline's trust anchor) depends on SVF's correctness on OpenSSL's codebase
- (−) Z3 timeout risk is high and non-mitigable by design
- (−) KLEE compatibility with OpenSSL 3.x (providers, C11 atomics) is uncertain

**Calibration:** A "5" means coin-flip for publication. This is accurate. The gated execution plan is well-designed, but the compound risk through multiple integration-heavy subsystems is high. The Synthesizer's fallback plans are the difference between 5 and 4.

---

## 4. Final Verdict

### CONDITIONAL CONTINUE

### Composite Score: **5.6/10**

Weighted: (Value × 0.25 + Difficulty × 0.15 + BestPaper × 0.20 + Laptop × 0.15 + Feasibility × 0.25)
= (6 × 0.25 + 6 × 0.15 + 4 × 0.20 + 7 × 0.15 + 5 × 0.25) = 1.50 + 0.90 + 0.80 + 1.05 + 1.25 = **5.50**

Simple average: (6 + 6 + 4 + 7 + 5) / 5 = **5.6**

### Probability Estimates

| Outcome | Probability | Rationale |
|---------|:---:|-----------|
| P(best-paper) | **3%** | T3 at depth 4/10 insufficient; no new CVE likely; certificates are conditional |
| P(top-4 venue accept) | **30%** | Need full pipeline working + 7+ CVEs recovered + certificates produced |
| P(any publication) | **50%** | Fallback plans raise floor: money-plot paper or slicer-as-tool paper at workshop/B-tier |
| P(abandon) | **30%** | Compound gate failure + KLEE/Z3 intractability |

### Binding Conditions for CONTINUE

1. **HONEST FRAMING IS NON-NEGOTIABLE.** The paper must state: "bounded model checking of a protocol-aware slice under Assumption A0, not full-library verification." The certificate wording must include A0 conditioning. Any marketing that says "OpenSSL has no downgrade attacks" without the qualifiers is unacceptable.

2. **T3 MUST DELIVER THE MONEY PLOT.** The merge operator must demonstrably achieve ≥10× path reduction on real OpenSSL negotiation code (not toy examples). If the real-world speedup is <5×, the paper's primary contribution evaporates. G2 at Week 6 is the correct kill gate.

3. **SCOPE EXCLUSIONS MUST BE FIRST-CLASS.** The attack-class scope table (from critique synthesis A3.3) must appear in the paper's Section 2, not buried in a limitations section. Readers must see what the certificate covers BEFORE they see the certificate.

4. **CUT THE DIFFERENTIAL EXTENSION ENTIRELY.** C1 (covering designs) is the mathematically deepest theorem but it's Phase 2. Do not mention it in the paper. It dilutes focus and creates a mismatch between the paper's claimed vs. proven contributions.

5. **PASS ALL GATES ON SCHEDULE.** G0 (Week 2), G1 (Week 4), G2 (Week 6), G3 (Week 10). Any gate failure triggers immediate reassessment. No extensions, no "partial passes."

### What the Skeptic Got Right

The Skeptic's most important contributions, which must constrain the final paper:

1. **"Bounded completeness is bounded model checking of a sliced approximation."** This is the correct description. The paper must use honest language. "Bounded-complete" should appear only with the full qualifier "(within bounds k, n, under Assumption A0)."

2. **T3 is domain instantiation, not fundamental new mathematics.** The paper should not oversell T3's novelty. Frame it as "we identify four structural properties of negotiation protocols that enable efficient symbolic analysis" — this is a contribution, but not a proof-technique contribution.

3. **Zero implemented code means all estimates are projections.** The 27% compound success probability is real. The paper should not be written until G3 passes. Theory documents are appropriate now; writing should not begin until empirical validation exists.

4. **The slicer is the pipeline's trust anchor and its weakest link.** Assumption A0 must be prominent. The paper's trustworthiness is exactly equal to the slicer's soundness on the target libraries.

5. **CVE-2015-0291 (sigalgs DoS) is not a downgrade attack.** Remove it from the in-scope CVE count. The honest count is 7 clean CVEs.

### What the Synthesizer's Fallback Plan Adds

The Synthesizer's gated fallback strategy is the reason this is CONTINUE rather than ABANDON:

1. **Minimal Viable Paper (55% success, composite ~6.0):** OpenSSL-only, 3-4 CVEs, certificates with honest scoping. This is a solid workshop or B-tier venue paper. It validates the approach without requiring the full four-library evaluation.

2. **Fallback A — Money Plot Paper (70% catchment if Z3 fails):** Slicer + merge operator demonstrated on negotiation code, showing path reduction, WITHOUT the DY+SMT encoding. This removes the Z3 timeout risk entirely. The paper becomes "protocol-aware symbolic execution" rather than "bounded-complete synthesis." Still publishable at ISSTA or ASE.

3. **Fallback C — Slicer as Auditing Tool (85% catchment):** If KLEE integration fails, the protocol-aware slicer alone — extracting negotiation-relevant code from production libraries — is a useful tool contribution. Publishable at a tools track.

4. **The fallback cascade means P(something publishable) ≈ 50%** rather than P(full headline paper) ≈ 27%. This is the insurance that justifies continued investment.

---

## 5. Summary Scorecard

| Dimension | Auditor | Skeptic | Synthesizer | **Final** | Delta from Ideation (7.0) |
|-----------|:---:|:---:|:---:|:---:|:---:|
| Value | 6 | 4 | 7 | **6** | −1.0 |
| Difficulty | 7 | 5 | 7 | **6** | −1.0 |
| Best-Paper | 5 | 3 | 7 | **4** | −3.0 |
| Laptop-CPU | 7 | 4 | 7 | **7** | 0.0 |
| Feasibility | 5 | 3 | 5 | **5** | — |
| **Composite** | **6.0** | **3.8** | **6.6** | **5.6** | **−1.4** |

**The −1.4 deflation from ideation's 7.0 reflects:**
- Best-Paper dropped 3 points (T3 math depth honestly assessed at 4/10)
- Difficulty dropped 1 point (every subsystem has a published template)
- Value dropped 1 point (7 scope exclusions erode the headline)
- Laptop-CPU held (architecture is genuinely sound for commodity hardware)
- Feasibility is a new dimension (coin-flip based on compound risk)

### Final Disposition

**CONDITIONAL CONTINUE** — proceed to implementation with strict gate enforcement. The project has genuine technical merit (the protocol-structure observation IS novel and useful) but the mathematical contribution alone cannot carry a best-paper argument. Success depends on engineering execution: making the pipeline actually work on real libraries, producing real certificates, and recovering real CVEs. The theory is adequate; the question is whether the system will work.

*The Skeptic dissents. The Skeptic is overruled but substantially right about the mathematics.*

---

*End of evaluation. All scores and conditions are binding for this stage.*
