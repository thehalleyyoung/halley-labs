# Verification Chair: Initial Theory Assessment

**Project:** NegSynth — "Negotiation Under Fire: Bounded-Complete Synthesis of Protocol Downgrade Attacks from Library Source Code"
**Phase:** Theory (quality gate)
**Gate Input:** Composite 6.25/10 (V7/D6/BP6/L6). CONDITIONAL CONTINUE (2-1, Skeptic dissents KILL). 0 fatal after 5 amendments.
**Date:** 2026-03-08
**Assessor Role:** Verification Chair

---

## 1. Theory Quality Criteria: What Makes Best-Paper-Caliber Theory

Before evaluating T1–T5 and C1, I want to be explicit about the bar I am applying. Best-paper theory at IEEE S&P, USENIX Security, or CCS is not judged like a pure-math publication — reviewers are looking for a different combination of virtues. The standard I use has five axes:

**Load-bearing.** Every theorem must drive an implementation decision or provide a guarantee that a user of the tool would rely on. A theorem is load-bearing if removing it would either (a) invalidate a claim in the evaluation, (b) remove a correctness guarantee from the pipeline, or (c) make an implementation module unjustified. Ornamental theorems — those that exist to pad a formalism section without constraining anything — are actively harmful at top venues because experienced reviewers detect them and it erodes trust. The B3 (domain-relative completeness) theorem from the discarded Approach B is the canonical example: "we find everything expressible in our domain" is vacuous. The current theorem set should contain zero instances of this pattern.

**Honest.** Proof difficulty ratings must be calibrated against the actual mathematical content, not inflated to sell novelty. If a theorem is a standard simulation-relation argument adapted to a new domain, it should say so. The Math Depth Assessor's ratings from ideation (T1: 3/10, T2: 3/10, T3: 5/10, T4: 4/10, T5: 3/10, C1: 6/10) set a useful baseline. Any significant upward revision in the theory stage must be justified by genuinely new mathematical content, not by repackaging. At S&P 2024, the best paper (tlspuffin, Dolev-Yao fuzzing) was transparent about its limited formal guarantees; honesty was rewarded.

**Complete where it matters.** The hard parts must be fully worked out — not deferred to "future work" or hidden behind proof sketches that wave at "standard techniques." The Skeptic's core concern — that the algebraic properties of T3 hold for an idealization but not for OpenSSL's actual C implementation — is exactly the kind of gap that peer reviewers will probe. The proof must either handle the gap or explicitly scope the guarantee. Papers like CompCert (Leroy, POPL 2006) and CertiKOS (Gu et al., OSDI 2016) set the standard: assumptions are stated, the gap between model and implementation is measured, and the proof structure is end-to-end.

**Skip ornamental.** No theorems should exist just to make the formalism section longer. Better to have four well-proved, load-bearing theorems than six where two are filler. The current set of five core theorems + one extension theorem is close to the right number, but I will examine each for ornamentality below.

**Implementation-mapped.** Each theorem should map to a specific code module. T3 → merge operator (~7K LoC). T1 → slicer + state machine extractor. T5 → DY+SMT encoder. T2 → concretizer + CEGAR loop. T4 → end-to-end pipeline composition. C1 → differential extension. This mapping exists in the current formulation and is a strength.

---

## 2. Assessment of T1–T5 and C1

### T1: Extraction Soundness — PASS (load-bearing, correctly rated)

**Statement:** Every trace of the extracted state machine corresponds to a feasible execution path in the original source code.

**Load-bearing?** Yes. Without T1, the entire pipeline is disconnected from the source code it claims to analyze. If the extracted state machine admits traces that the source code cannot produce, certificates are meaningless (certifying absence of attacks in a phantom program). If it misses traces the source can produce, attack synthesis is incomplete. T1 is the foundation on which T3 and T4 build.

**Claimed difficulty:** 3/10, ~2 person-months. **Assessment: accurate.** This is a simulation relation argument. The proof template is well-established from abstract interpretation (Cousot & Cousot 1977) and verified compilation (CompCert). The domain-specific adaptation — accounting for the merge operator's state abstractions over LLVM IR semantics — is non-trivial engineering but not mathematically deep.

**Proof sketch convincing?** Mostly yes, but the weakest link is the slicer's soundness. T1 implicitly assumes the protocol-aware slicer correctly identifies all negotiation-relevant code. If the slicer misses a relevant code path (e.g., an indirect callback through OpenSSL's `SSL_METHOD` vtable that affects cipher-suite selection), T1's simulation relation breaks silently — the extracted state machine simply doesn't contain the missing transition, and the proof still "goes through" vacuously for the extracted fragment. This is the most dangerous kind of gap: a theorem that is technically true but guarantees less than it appears to.

**Weakest link:** Slicer soundness is not itself a stated theorem. It is an implicit assumption of T1. This must be made explicit — either as a lemma or as a clearly stated assumption with empirical validation.

### T2: Attack Trace Concretizability — PASS (load-bearing, slight underrating)

**Statement:** Every satisfying SMT assignment can be concretized into an executable byte-level attack trace with success rate ≥ 1−ε.

**Load-bearing?** Yes. Without T2, the pipeline produces symbolic attack descriptions that may not correspond to real network traffic. The entire "concrete, byte-level attack traces" selling point collapses. More subtly, T2 is where the CEGAR refinement loop lives — it turns abstract counterexamples into concrete ones or proves them spurious, which is essential for the pipeline's precision.

**Claimed difficulty:** 3/10, ~3 person-months. **Assessment: slightly underrated, should be 3.5-4/10.** The CEGAR soundness argument itself is standard, but the protocol-specific content — mapping symbolic terms to correctly-framed TLS records and SSH packets with proper length fields, MAC tags, sequence numbers — is where the real complexity hides. TLS record framing alone has dozens of corner cases (fragmentation, multiple records per TCP segment, version-specific header formats). The symbolic-to-concrete bridge is where ProVerif and Tamarin have historically struggled to produce replay-ready outputs; it is not trivial.

**Proof sketch convincing?** Conditionally. The CEGAR termination argument (finite predicate vocabulary over bounded execution) is fine. The concern is the ε term. If ε is measured per-library and per-CVE, it becomes an empirical artifact rather than a formal guarantee. The theorem essentially says: "attacks concretize, except when they don't, and we'll tell you how often they don't." This is honest, but reviewers at S&P will probe whether ε is tightly controlled or a catch-all escape hatch. The kill-gate threshold of ε > 0.01 triggering investigation is appropriate.

**Weakest link:** The symbolic-to-concrete framing bridge for SSH is under-specified compared to TLS. Terrapin (CVE-2023-48795) exploits SSH extension negotiation with sequence-number manipulation — concretizing this requires byte-accurate SSH binary packet framing, which is a distinct engineering challenge from TLS record framing.

### T3: Protocol-Aware Merge Correctness — CONDITIONAL PASS (crown theorem, overrated in novelty, correctly rated in difficulty)

**Statement:** The merge operator ⊵ preserves protocol-bisimilarity: merged states produce exactly the same observable negotiation behaviors. Achieves O(n) path count where generic veritesting produces O(2^n) on negotiation code with n cipher suites.

**Load-bearing?** Absolutely. T3 is the reason the pipeline is tractable on production libraries. Without it, KLEE explores O(2^n) paths on negotiation code (n = number of cipher suites; OpenSSL supports ~100), which is infeasible. The merge operator is what makes the difference between "runs in hours" and "runs for centuries." Remove T3, and NegSynth is just "KLEE + DY encoding" — which is a reasonable research project but without the headline scalability claim.

**Claimed difficulty:** 5/10, ~2 person-months. **Assessment: difficulty rating is accurate. Novelty is moderately overstated.** The Math Depth Assessor's analysis from ideation is correct: each ingredient is individually well-known. Kuznetsov et al. (PLDI 2012) did state merging with cost models. Avgerinos et al. (ICSE 2014) did veritesting. Milner's bisimulation framework is 1980s-era process algebra. The genuinely new insight is *identifying* that cipher-suite negotiation has the four algebraic properties (finite outcomes, lattice preferences, monotonic progression, deterministic selection) that make aggressive merging sound. This is a real contribution — domain identification is legitimate novelty — but it is a "noticed the right thing" contribution, not a "proved a hard thing" contribution.

**Proof sketch convincing?** The formal structure is sound: define protocol LTS, define merge as congruence, show congruence closure under the four properties, conclude bisimilarity preservation. Standard process-algebraic argument. The O(n) vs O(2^n) complexity claim follows from the finite-outcome-space property and is straightforward to prove.

**The critical concern** is the gap between the idealized four properties and real implementation behavior. The Skeptic raised this in ideation and it was partially resolved: the resolution states that T3 holds for the *extracted model*, not the raw C code, and the gap is bridged by T1 + the slicer. This is the correct architecture, but it means T3's guarantee is conditional on T1's guarantee, which is conditional on the slicer's soundness, which is not formally guaranteed. The chain is: slicer correct → T1 holds → T3 meaningful → T4 follows. The slicer is the weakest link in the entire theorem chain.

**Additional concern:** The four algebraic properties are stated as axioms of the negotiation domain. But real code violates them in edge cases: (1) "deterministic selection" fails when `OPENSSL_CONF` or `SSL_CTX_set_cipher_list` introduce runtime-configurable preferences; (2) "monotonic progression" fails if error-handling code resets handshake state (CCS Injection exploits exactly this); (3) "finite outcome space" is violated by custom cipher suites or GREASE values. The proof must either (a) show these violations are handled by the slicer/extraction, or (b) explicitly scope the guarantee to exclude these cases. Currently, neither is done clearly.

**Weakest link:** The idealization gap. If a reviewer asks "what happens when `ssl3_choose_cipher` calls a user-registered callback that has side effects?" and the answer is "we assume the slicer handles it," the theorem's weight drops significantly.

### T4: Bounded Completeness — CONDITIONAL PASS (headline result, composition is correct in structure but fragile in practice)

**Statement:** Within execution depth k and adversary budget n, NegSynth finds every downgrade attack or certifies absence, with probability ≥ 1−ε.

**Load-bearing?** This is the headline theorem. The paper's title includes "bounded-complete." Without T4, NegSynth is a heuristic tool that finds some attacks — useful, but not qualitatively different from tlspuffin. T4 is what justifies "certificates" as a contribution. If T4 falls, the paper loses its primary differentiator.

**Claimed difficulty:** 4/10, ~3 person-months. **Assessment: difficulty rating is accurate as a composition theorem.** The structure is: T1 (extraction preserves traces) + T3 (merge preserves behaviors) + T5 (encoding preserves satisfiability) → end-to-end guarantee. This is a standard transitivity-of-soundness argument, analogous to CompCert's composition of compiler passes. The mathematical difficulty is low. The engineering difficulty of making the composition actually hold in implementation is high.

**Proof sketch convincing?** The composition structure is sound. My concerns are:

1. **The ε problem.** T4's guarantee is "with probability ≥ 1−ε," where ε comes from T2's concretization success rate. This means T4 is a *probabilistic* bounded-completeness guarantee, not a deterministic one. For certificates, this is a significant weakening. A "bounded-completeness certificate" that says "no attack exists, with probability 0.99" is a qualitatively different claim from "no attack exists within these bounds." The paper needs to be crystal clear about this distinction. If ε is small enough (< 0.001), the probabilistic qualifier is practically irrelevant. If ε is, say, 0.05, then 1-in-20 certificates might be wrong — which undercuts the entire certification story.

2. **The k, n bounds problem.** The Skeptic's "bounded-completeness theater" critique was addressed by the empirical bound validation table (all known CVEs require k ≤ 15, n ≤ 5; k=20, n=5 covers 2× protocol depth). This is a reasonable response, but it is an *empirical* argument, not a *formal* one. The formal theorem says "within bounds k, n." The practical question is whether k=20, n=5 is sufficient to capture *unknown* attack classes. The coverage metric (≥99% of reachable negotiation states) helps, but "99% of states explored" does not imply "99% of attacks captured" — attacks may concentrate in the unexplored 1%.

3. **Hidden dependency on slicer completeness.** T4 composes T1, T3, T5 — but the extraction step (T1) depends on the slicer having captured all negotiation-relevant code. If the slicer misses a code path, T4's "every downgrade attack" claim is silently scoped to "every downgrade attack reachable through slicer-identified code." This is a real gap.

**Weakest link:** The probabilistic qualifier and the slicer dependency are the two critical vulnerabilities. A reviewer who probes either can significantly weaken the headline claim.

### T5: SMT Encoding Correctness — PASS (load-bearing, correctly rated)

**Statement:** The SMT constraint system is equisatisfiable with the composed DY adversary and extracted state machine.

**Load-bearing?** Yes. Without T5, the Z3 queries might be over-constrained (missing real attacks) or under-constrained (producing spurious attacks). T5 ensures that satisfying SMT assignments correspond to actual attacks and that UNSAT means no attack exists within bounds.

**Claimed difficulty:** 3/10, ~2 person-months. **Assessment: accurate, but the "faithful adversary-knowledge accumulation" clause hides non-trivial work.** The standard theory-combination results (Nelson-Oppen for BV+Arrays+UF) apply. The protocol-specific content is encoding the Dolev-Yao adversary's knowledge-accumulation rule (adversary learns from messages it observes) as monotone set constraints in SMT. This is where Tamarin and ProVerif have rich prior art to draw on, but translating from their dedicated logics to SMT-LIB requires care.

**Proof sketch convincing?** The equisatisfiability argument follows from structural correspondence between the DY term algebra and its SMT encoding. The concern is scale: if the encoding blows up for adversary budget n=5 (each adversary action introduces new terms), Z3 may not decide the queries in reasonable time. This is an engineering risk, not a theoretical one — T5's correctness doesn't depend on solver performance.

**Weakest link:** The "uninterpreted functions" component for cryptographic operations (encryption, hashing) must enforce Dolev-Yao's perfect cryptography assumption. If the UF axioms are incomplete (missing necessary equalities) or inconsistent (admitting behaviors that violate DY), T5 fails subtly. The use of tlspuffin's peer-reviewed DY term algebra mitigates this risk but doesn't eliminate it.

### C1: Covering-Design Differential Completeness — PASS (extension-only, deepest math, correctly assessed)

**Statement:** For N ≥ 3 libraries, a covering design of strength t guarantees detection of all pairwise behavioral deviations within B(n,k,t) test configurations.

**Load-bearing?** For the differential extension only. C1 is explicitly scoped as a Phase 2 theorem and does not affect the core pipeline's guarantees. If Phase 2 is cut (as in the minimal viable paper), C1 is not needed. This scoping is correct and honest.

**Claimed difficulty:** 6/10, ~3 person-months. **Assessment: this is indeed the deepest theorem, correctly rated.** The connection between combinatorial covering designs and protocol behavioral coverage is non-obvious and mathematically interesting. The bound B(n,k,t) involves asymptotic results from combinatorial design theory (Rödl 1985, Gordon-Kuperberg-Patashnik 1995). Proving tightness requires constructive arguments that are not standard.

**Key limitation (correctly identified):** C1 guarantees *testing completeness* over parameter space, not *verification completeness* over execution paths. It complements T4 but does not replace it. The Skeptic's critique that "covering designs cover input configurations, not attack sequences" is valid and is properly acknowledged.

**Weakest link:** Pairwise coverage (strength t=2) may miss 3-way interactions. The risk is acknowledged but not quantified. For the extension, this is acceptable; if C1 were a core theorem, it would need to be addressed.

---

## 3. Theory Completeness Gaps

### Gap 1: Slicer Soundness Is the Untheoremed Foundation

The entire theorem chain rests on the protocol-aware slicer correctly identifying all negotiation-relevant code. T1 assumes it. T3 relies on the extracted model satisfying the four algebraic properties. T4 composes through it. Yet there is no theorem or even a formal statement of what "correct slicing" means in this context.

This is not merely an aesthetic gap. The slicer must handle: (a) indirect calls through `SSL_METHOD` vtable dispatch (macro-generated), (b) callback chains registered via `SSL_CTX_set_*` functions, (c) `#ifdef`-guarded code paths that are negotiation-relevant under some build configurations, (d) FIPS mode runtime switches that alter cipher availability. Each of these could cause the slicer to miss negotiation-relevant code, silently breaking T1.

**Recommendation:** Either (a) state slicer soundness as a formal assumption ("Assumption A0: the protocol-aware slicer identifies a superset of all negotiation-relevant code paths") with empirical validation (CVE reachability check), or (b) prove a slicer soundness lemma. Option (a) is honest and sufficient for a top-venue paper; option (b) is a separate research contribution.

### Gap 2: The Idealization Gap for T3's Four Properties

The four algebraic properties are stated as axioms of the "negotiation protocol domain." The proof of T3 proceeds *given* these axioms. But real implementations violate them:

- **Finite outcome space:** GREASE cipher suite values (RFC 8701) and custom OIDs violate finitude if not handled by the slicer/encoder.
- **Lattice-ordered preferences:** Client and server preferences are not always lattice-ordered; OpenSSL's `SSL_OP_CIPHER_SERVER_PREFERENCE` flag switches between two incompatible orderings at runtime.
- **Monotonic progression:** Error paths and renegotiation break monotonicity. CCS Injection (CVE-2014-0224) specifically exploits non-monotonic state transitions.
- **Deterministic selection:** `SSL_CTX_set_cert_cb` callbacks can introduce non-determinism into cipher selection based on certificate availability.

The current resolution — "the properties hold on the extracted model, not the raw C code" — is correct in principle but needs to be made rigorous. The extraction step must either (a) abstract away these violations (and prove the abstraction is sound) or (b) detect them and fall back to non-merged exploration. Case (b) is the practical answer and should be stated explicitly.

### Gap 3: T4's Composition Hides an Inductive Step

T4 composes T1, T3, and T5 "via three-level simulation chain." But the composition is not simply transitivity of three binary relations. The execution-depth bound k interacts with the merge operator: merging at depth d affects what states are reachable at depth d+1. The proof needs an inductive argument showing that the merge operator at each depth preserves the reachability set at subsequent depths. This is the kind of "obvious" lemma that turns out to require careful case analysis and could add 2-3 weeks to the proof timeline.

### Gap 4: Adversary Budget Semantics

The adversary budget n is described as "number of adversary actions" but not formally defined. Does n count message injections? Message drops? Message modifications? Reorderings? For Terrapin (CVE-2023-48795), the adversary performs a "delete-then-inject" sequence that requires both dropping a message and injecting a replacement. Does this cost n=1 or n=2? The formal semantics of n must be pinned down, because the bounded-completeness guarantee at n=5 means different things under different semantics.

### Gap 5: Certificate Format and Semantics

"Bounded-completeness certificates" are the paper's primary empirical contribution, but the theory section doesn't define what a certificate formally *is*. Is it an UNSAT proof from Z3? A witness of exhaustive state-space exploration? A combination? The certificate's semantic content — exactly what it guarantees and under what assumptions — needs a formal definition. Without this, "certificate" is a marketing term, not a technical one.

---

## 4. Comparison to Best Papers at Target Venues

**tlspuffin (S&P 2024):** Lightweight formalism (~2 pages of theory), succeeded on empirical impact. NegSynth's T1-T5 chain is substantially more ambitious — but more surface area for reviewer attacks. tlspuffin succeeded partly by promising less and delivering exactly that.

**Tamarin Prover extensions (USENIX Security 2023):** Dense equational-theory formalism, genuine 5-7/10 difficulty. NegSynth's core theorems (T1, T2, T5 at 3/10) fall below this bar; T3 at 5/10 is competitive. The aggregate theory package is adequate but not exceptional by Tamarin-paper standards.

**KLEE (OSDI 2008 best paper):** Minimal formalism, massive empirical impact. NegSynth's theory is deeper than KLEE's. If empirical results match KLEE's impact (new bugs), theory is more than sufficient. If results are incremental (known CVEs only), theory alone won't carry best paper.

**CompCert (Leroy, POPL 2006):** Gold standard for compositional correctness. T4's composition follows CompCert's template (simulation relations across pipeline stages) but is simpler. Framing the comparison explicitly ("our composition follows the CompCert template") builds credibility.

### Overall Calibration

NegSynth's theory is **adequate for acceptance** at S&P/CCS/USENIX Security but **not sufficient for best paper on theory alone.** Best paper requires either (a) a genuinely surprising theoretical result (T3 does not qualify — "finite things are finite" is the core insight) or (b) outstanding empirical results. The "money plot" (O(2^n) → O(n) path reduction) should be the centerpiece of the formalism section.

---

## 5. Risk Assessment for Theory Stage

### P(all proofs complete and correct): 70-75%

The individual theorem risks are low (T1, T5: ~5%), low-medium (T2, T3: ~10-15%), and medium (T4: ~15%). The compound risk is dominated by T4's composition, which depends on all components being correct simultaneously. The slicer-soundness gap (Gap 1) is the wild card: if it requires a formal treatment, add 2-3 months.

### P(theory survives peer review): 55-65%

Even if all proofs are correct, peer reviewers will attack: (a) the idealization gap in T3, (b) the probabilistic qualifier in T4, (c) the undefined certificate semantics, (d) the slicer-soundness assumption. A well-prepared author response can address (a)-(d), but all four must be anticipated. The 10-15% gap between "proofs correct" and "survives review" reflects the presentation risk — correct proofs can still be rejected if the gaps are poorly handled.

### Most likely failure mode: slicer-induced silent unsoundness

The most dangerous scenario is: proofs are all technically correct, the tool produces certificates, but the slicer misses a negotiation-relevant code path. The certificate then claims "no downgrade attack exists" for a library that actually has one. This would not be caught by any of the stated theorems (they are all conditional on slicer correctness). It would be caught by the empirical CVE recovery evaluation — but only for known CVEs. An unknown attack through a slicer-missed path would be invisible.

**Second most likely failure mode:** ε is too large. If CEGAR concretization fails > 5% of the time on a specific library, the bounded-completeness certificate for that library becomes "no attack, probably" — which reviewers may find insufficient.

---

## 6. Preliminary CONTINUE/ABANDON Signal

### CONDITIONAL CONTINUE

I lean toward continuing with the following mandatory conditions:

**Condition 1: Formalize the slicer-soundness assumption.** Either state it as Assumption A0 with empirical validation, or prove a slicer soundness lemma. This is the single most important gap. Without it, the entire theorem chain is built on an unstated premise.

**Condition 2: Define certificate semantics formally.** A certificate must be a formal artifact with a precise meaning: "Under Assumption A0 (slicer soundness), within execution depth k and adversary budget n (where budget counts [specific adversary actions]), with concretization confidence 1−ε, the SMT encoding of the extracted state machine composed with the DY adversary model is UNSATISFIABLE, implying no downgrade attack exists in the analyzed code paths." This is a mouthful but it is the honest statement. Shortening it loses precision.

**Condition 3: Address T3 idealization violations explicitly.** For each of the four algebraic properties, document which real-code patterns violate it and how the extraction/merge handles the violation (abstraction, fallback to non-merged exploration, or explicit scope exclusion). This should be a table in the paper.

**Condition 4: Pin down adversary budget semantics.** Define n precisely. Count the atomic actions and show that n=5 covers all known CVE attack sequences with headroom.

**Condition 5: Measure ε early.** If ε > 0.01 on any library, the concretization pipeline needs debugging before the theory is finalized. The kill gate (ε > 0.01 triggers investigation) from the final approach is appropriate.

**Rationale for CONTINUE (not ABANDON):** The theorem chain's structure is sound. The composition architecture (T1→T3→T5→T4) follows established patterns (CompCert). The individual theorems are provable — none requires inventing new mathematics. The gaps are all fixable: they are omissions of explicit statements, not fundamental impossibilities. The largest risk (slicer soundness) is addressable by either formalization or honest assumption-stating.

**Rationale against ABANDON:** No theorem is ornamental. No theorem has been identified as provably wrong. The difficulty ratings are honest (the Math Depth Assessor's independent assessment aligns with mine). The theory serves the implementation — it is not theory for theory's sake.

**What would flip me to ABANDON:**
- Discovery that T3's algebraic properties cannot be maintained for CCS Injection (CVE-2014-0224), because CCS Injection specifically exploits non-monotonic state transitions. If the merge operator cannot handle this CVE class without disabling itself, the tractability story collapses.
- ε > 0.05 in early measurements, indicating the symbolic-to-concrete bridge is fundamentally lossy.
- The slicer requires >15K lines from OpenSSL (violating kill gate G1), meaning the "negotiation is small" assumption is wrong.

---

## 7. Key Questions for Other Team Members

### Question 1 — For the Formal Methods Lead:

**Can T3's merge operator correctly handle CCS Injection (CVE-2014-0224)?** CCS Injection exploits a non-monotonic state transition — the `ChangeCipherSpec` message arrives at the wrong handshake phase, resetting state. This directly violates T3's "monotonic progression" axiom. If the merge operator must fall back to unmerged exploration for this CVE class, what fraction of real negotiation code requires the fallback? If it is >20%, the O(n) complexity claim is misleading. Provide a concrete worked example of the CCS Injection case through the merge operator.

### Question 2 — For the Algorithm Designer:

**What is the formal definition of "adversary budget" n, and what is the minimal n for each of the 8 target CVEs?** The current formulation says n=5 "suffices" but does not define what an adversary action is. For Terrapin (CVE-2023-48795), the attack requires: (1) intercepting a message, (2) stripping an SSH extension, (3) adjusting the sequence number, (4) forwarding the modified message. Is this n=1 (one compound action), n=3 (three atomic actions), or n=4? The bounded-completeness guarantee is only as meaningful as the budget semantics. Provide a table: [CVE, attack-action sequence, n under your definition].

### Question 3 — For the Empirical Scientist:

**What is your plan for measuring ε independently of the CVE oracle?** Measuring ε on known CVEs is circular — you know the concretization should succeed because you have the ground-truth attack trace. The real test is ε on the *certificate* side: when Z3 returns UNSAT, can you verify the certificate independently? And when Z3 returns SAT for a non-CVE query, what is the concretization success rate? Design an experiment that measures ε on synthetic attack scenarios where ground truth is known but not derived from historical CVEs.

### Question 4 — For the Red-Team Reviewer:

**Construct a concrete scenario where the slicer misses negotiation-relevant code and the pipeline produces a false certificate.** The current risk analysis acknowledges slicer imprecision but does not stress-test it with an adversarial example. I want a specific code pattern (e.g., a cipher-selection callback registered via `SSL_CTX_set_cert_cb` that disables a cipher suite under specific certificate conditions) that a plausible slicer would miss. If this scenario is constructible, it demonstrates the slicer-soundness gap is not merely theoretical. If it is not constructible (because the slicer's taint analysis covers all such paths), that is strong evidence the gap is manageable.

### Question 5 — For Everyone:

**If we could only prove three of the five core theorems, which three maximize the paper's value?** This is a contingency question. If proof effort runs over budget, we need a triage plan. My tentative answer: T1, T3, T5 (the three composition inputs to T4), with T4 stated as a corollary and T2 downgraded to "empirically validated." But I want the team's assessment of which theorem-deletion would hurt the paper most.

---

## Summary Scorecard

| Theorem | Load-Bearing | Difficulty Accurate | Proof Convincing | Weakest Link | Verdict |
|---------|:---:|:---:|:---:|---|---|
| T1 | ✅ | ✅ (3/10) | ⚠️ Slicer assumption | Unstated slicer soundness | PASS |
| T2 | ✅ | ⚠️ (slightly underrated) | ✅ | SSH concretization framing | PASS |
| T3 | ✅ | ✅ (5/10) | ⚠️ Idealization gap | Real-code property violations | CONDITIONAL |
| T4 | ✅ | ✅ (4/10) | ⚠️ Composition fragility | Probabilistic qualifier + slicer | CONDITIONAL |
| T5 | ✅ | ✅ (3/10) | ✅ | UF axiom completeness | PASS |
| C1 | ✅ (extension) | ✅ (6/10) | ✅ | Pairwise-only coverage | PASS |

**Overall theory health: 6.5/10.** Structurally sound, gaps are addressable, no fatal flaws. With the five conditions above resolved, this rises to 7.5/10 — adequate for top-venue submission and competitive for best-paper consideration if empirical results are strong.

**Signal: CONDITIONAL CONTINUE.** The theory is on track, but five specific gaps must be addressed before the theory stage is complete. None is a showstopper; all require explicit work.
