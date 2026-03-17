# Red-Team Review: NegSynth

**Title:** "Negotiation Under Fire: Bounded-Complete Synthesis of Protocol Downgrade Attacks from Library Source Code"
**Reviewer Role:** Adversarial Red-Team
**Date:** 2026-03-08
**Verdict:** 4 CRITICAL, 5 SERIOUS, 5 MINOR findings. The headline claim (T4) rests on a fragile composition chain whose weakest links — the unproved slicer and the idealized algebraic properties of T3 — are individually plausible but jointly under-specified enough that the bounded-completeness certificate could be a false negative factory.

---

## 1. Attack on T3: Protocol-Aware Merge Correctness (Crown Theorem)

### A3.1 — Real code violates the four algebraic properties [CRITICAL]

T3's O(n) path bound depends on four algebraic properties: (P1) finite enumerable outcome spaces, (P2) lattice-ordered preferences, (P3) monotonic state progression, (P4) deterministic selection given matching capability sets. The proof treats these as axioms. OpenSSL's negotiation logic refutes at least two of them in production configurations.

**P4 violation — non-deterministic selection.** OpenSSL exposes `SSL_CTX_set_cipher_list()` and the newer `SSL_CTX_set_ciphersuites()` APIs, which accept colon-separated priority strings from application code at runtime. The effective cipher ordering is therefore not a static lattice: it is an input-dependent permutation. Worse, `SSL_CTX_set_cert_cb()` allows an application-layer callback to run during handshake negotiation and *reject or modify* the selected cipher based on the peer's certificate chain. If this callback consults external state (a database of revoked keys, an OCSP response), the selection function is non-deterministic from the symbolic executor's perspective — it depends on environment state outside the slice. The merge operator's correctness proof assumes that two symbolic states with identical cipher capability sets will produce identical selection outcomes. A callback that behaves differently based on external state creates a forking condition the merge operator silently collapses.

**P2 violation — non-lattice preferences.** FIPS mode in OpenSSL (`OPENSSL_FIPS`) dynamically restricts the available cipher set at runtime, not at compile time. The preference structure is not a single lattice but a mode-dependent *family* of lattices. The `#ifdef` forests in `ssl_ciph.c` (e.g., `#ifndef OPENSSL_NO_EC`, `#ifdef SSL_FORBID_ENULL`) generate compile-time lattice variants, but FIPS mode filtering happens at runtime within `ssl_cipher_get_disabled()`. The merge operator must either: (a) treat FIPS mode as a symbolic variable, doubling the state space for every merge (destroying the O(n) claim), or (b) concretize FIPS mode early, losing coverage of FIPS/non-FIPS interaction paths — exactly where CVE-2015-3197 (disabled SSLv2 ciphers still selectable) lived.

**P3 violation — non-monotonic state via renegotiation and HRR.** TLS 1.2 renegotiation creates a state machine loop: after completing an initial handshake, the state machine re-enters negotiation with potentially different cipher constraints. TLS 1.3's HelloRetryRequest (HRR) creates a similar loop within the initial handshake. If P3 (monotonic/acyclic state progression) is enforced by bounding loop iterations, the merge operator is sound but the bounded-completeness claim (T4) inherits a coverage hole for multi-renegotiation scenarios.

**Concrete failure scenario:** Consider an OpenSSL server configured with `SSL_CTX_set_cert_cb` that selects ECDSA ciphers when the client presents an ECC certificate and RSA ciphers otherwise. Two symbolic states differing only in the client certificate will be merged by the operator (same capability set, same preference lattice). After the merge, the ITE-joined state must branch on the certificate type — but the merge was supposed to eliminate this branch. If the merged state resolves the ITE incorrectly or the bisimilarity proof's definition of "observable" excludes the certificate-check path, the merge is unsound for attacks that exploit certificate-type-dependent cipher selection.

**Severity: CRITICAL.** If even 5% of production OpenSSL configurations use cipher callbacks, the "bounded-completeness certificate" issued for that library silently excludes those configurations. The certificate's value proposition — "OpenSSL 3.x has no downgrade attack within these bounds" — is voided for any deployment using the affected APIs.

### A3.2 — The O(n) bound is for toy negotiation code [SERIOUS]

The proposal acknowledges this ("The O(n) bound is the theoretical ceiling... empirical claim is 10-100x path reduction"). But the acknowledged 10-100x range is itself optimistic. Real negotiation code has cross-cutting concerns that multiply the state space in ways the four algebraic properties do not address:

1. **Session resumption (TLS 1.2 session IDs, TLS 1.3 PSK).** The server's session cache is external mutable state. Resumption paths bypass cipher negotiation entirely or constrain it (resumed sessions inherit the original cipher). This creates a binary fork (resume vs. full handshake) that the merge operator cannot collapse because the fork's condition depends on cache lookup — an external operation.

2. **ALPN negotiation.** Application-Layer Protocol Negotiation runs concurrently with cipher selection and can influence which ciphers are acceptable (HTTP/2 requires specific cipher suites per RFC 7540 §9.2.2). ALPN adds a second enumeration dimension: the merge operator handles one cipher-suite dimension in O(n), but ALPN × cipher interaction is O(n × m) unless ALPN satisfies the same four algebraic properties (it arguably does, but this must be proved separately).

3. **SNI-based dispatch.** Server Name Indication selects a virtual host before cipher negotiation. Each virtual host may have a different cipher configuration. If the slicer includes SNI dispatch, the symbolic executor sees k virtual hosts × n cipher suites, and the merge operator can only collapse within each virtual host.

4. **Early data (0-RTT).** TLS 1.3 0-RTT introduces a path where application data is sent before the handshake completes. The negotiation state machine forks on whether 0-RTT is accepted, rejected, or absent. This three-way fork is not a cipher-suite enumeration — it's a mode switch — so the merge operator's algebraic properties do not apply.

**Net effect:** On a realistic OpenSSL configuration with session resumption, ALPN, SNI for 3 virtual hosts, and 0-RTT, the actual path count is closer to O(n × m × k × 3) rather than O(n). With n=30 ciphers, m=5 ALPN protocols, k=3 hosts, and a 3-way 0-RTT fork, that's 1350 base paths before considering renegotiation loops. The "money plot" showing O(n) vs O(2^n) is technically correct for the idealized model but misleading for production code.

**Severity: SERIOUS.** The "money plot" is the paper's primary visual communication tool. If reviewers test the artifact on realistic configurations and see 1000+ paths instead of ~30, the credibility of T3's complexity claim is destroyed. The mitigation is straightforward — frame the O(n) result as applying to the cipher-selection subroutine, not the entire negotiation — but this weakens the headline considerably.

### A3.3 — Bisimilarity's "observable" predicate is under-specified [SERIOUS]

The merge operator preserves "protocol-bisimilarity": merged states produce "exactly the same observable negotiation behaviors." But what counts as observable?

**Narrow definition (cipher suite selected + protocol version).** This is sufficient to detect attacks like FREAK and Logjam (wrong cipher selected). But it misses:
- **Alert-code-based oracles.** POODLE's padding oracle relies on the *error type* (bad_record_mac vs. decryption_failed). If alerts are not "observable," the merge can collapse states that differ only in error handling — exactly the states an attacker exploits.
- **Timing differences.** Lucky13 (CVE-2013-0169) exploits timing differences in MAC verification. If timing is not "observable," timing-based downgrade vectors are invisible.
- **Message ordering.** CCS Injection (CVE-2014-0224) exploits accepting ChangeCipherSpec at the wrong handshake state. The attack depends on *when* a message is accepted, not *what* cipher is selected. If "observable" is defined only over final negotiation outcomes, message-ordering attacks are outside the bisimilarity relation.

**Wide definition (all protocol outputs including alerts, timing, message acceptance points).** This is sound but likely destroys the merge operator's effectiveness: two states that select the same cipher but differ in their error-handling path are no longer bisimilar. The merge operator cannot collapse them, and the path count regresses toward generic veritesting.

The proposal does not specify which definition is used. This is not a minor oversight — it determines which attack classes NegSynth can and cannot detect, and therefore bounds the semantic content of the "bounded-completeness certificate."

**Severity: SERIOUS.** The paper must explicitly define the observation function and enumerate which attack classes fall inside and outside its scope. Without this, the certificate is uninterpretable.

---

## 2. Attack on T4: Bounded Completeness (Headline Result)

### A4.1 — The ε qualifier may be vacuously large [SERIOUS]

T4 states: "NegSynth finds every downgrade attack or certifies absence, with probability ≥ 1−ε." The proposal sets a kill gate at ε > 0.01. But ε is measured *per library*, and it aggregates three independent failure modes:

1. **Slicer incompleteness** (negotiation-relevant code excluded from slice).
2. **Merge unsoundness** (merged states that are not actually bisimilar — see A3.1).
3. **CEGAR non-convergence** (abstract counterexample cannot be concretized within the refinement budget).

These are not independent. If the slicer excludes a callback-driven cipher filter (mode 1), the merge operator will operate on an incomplete model (exacerbating mode 2), and CEGAR cannot refine what is missing (mode 3). The real ε is not the product of three small probabilities — it's the maximum of three correlated failure rates.

**Worst case:** If 5% of negotiation paths go through callback-based cipher filtering and the slicer misses the callback, ε ≥ 0.05 for that library. The certificate then says "with probability ≥ 0.95, no downgrade attack exists" — which is operationally useless. A security engineer cannot ship a library with a 5% chance of harboring a downgrade attack.

**How ε is measured matters.** The proposal says ε is "empirically bounded per target library." But how? If ε is measured as (1 − fraction of known CVEs recovered), it's circular: you can only measure recall on known attacks, not unknown ones. If ε is measured as (1 − fraction of negotiation states explored), it conflates state coverage with attack coverage — high state coverage does not imply high attack coverage if the uncovered states are precisely the callback-driven paths where attacks live.

**Severity: SERIOUS.** The paper must provide a rigorous, non-circular definition of ε and demonstrate that it is measurable. The current framing allows ε to be defined post-hoc in whatever way makes the numbers look good.

### A4.2 — Bounds k=20, n=5 are empirically validated, not theoretically justified [CRITICAL]

The structural argument offered is: "TLS handshakes complete in ≤10 round trips; SSH in ≤8. k=20 covers 2× protocol depth." This is a heuristic, not a proof. Several attack classes challenge it:

**Multi-renegotiation attacks.** TLS 1.2 permits unbounded renegotiation. An attack that requires the adversary to trigger renegotiation 3 times requires k ≥ 30 (3 × 10 round trips). The proposal's own CCS Injection CVE (CVE-2014-0224) requires injecting a CCS message at a precise handshake point — the minimal k for CCS Injection is reported as ≤15, but variants that exploit renegotiation would require higher k.

**Adversary budget n=5 is low for stateful attacks.** A Dolev-Yao adversary with budget n=5 can inject, drop, or modify 5 messages. But a sophisticated downgrade attack might require: (1) injecting a modified ClientHello, (2) dropping the server's ServerHello, (3) injecting a forged ServerHello with weaker ciphers, (4) modifying the client's response, (5) injecting a modified Finished message. That's 5 operations for a single handshake. A cross-handshake attack (e.g., exploiting session resumption after a downgraded initial handshake) would require n > 5.

**Unknown unknowns.** The deepest problem: k=20, n=5 is validated against 8 *known* CVEs. The entire point of NegSynth is to find *unknown* attacks. The validation provides no guarantee that unknown attacks with k > 20 or n > 5 don't exist. The certificate is not "this library has no downgrade attacks" — it's "this library has no downgrade attacks of depth ≤20 using ≤5 adversary operations." The practical value depends entirely on whether real attacks always fall within these bounds, which is an empirical bet, not a theorem.

**Severity: CRITICAL.** The headline claim is "bounded-complete synthesis." If the bounds are too small, the tool has a systematic blind spot for complex attack classes. The paper must either (a) provide a theoretical argument that all downgrade attacks of interest have bounded depth/budget (unlikely — TLS renegotiation is Turing-complete in the limit), or (b) explicitly acknowledge the bounds as a pragmatic assumption and drop "completeness" from the headline for attack classes involving renegotiation.

### A4.3 — The composition chain T1→T3→T5→T4 has assumption leakage [CRITICAL]

T4 is a composition: it threads T1 (extraction soundness), T3 (merge correctness), and T5 (SMT encoding correctness) into an end-to-end guarantee. Each theorem has assumptions:

| Theorem | Key Assumption | What Violates It |
|---------|---------------|------------------|
| T1 | Slicer is sound (includes all negotiation-relevant code) | Missed indirect calls, callbacks, macro-expanded dispatch |
| T3 | Four algebraic properties hold | Runtime cipher callbacks, FIPS mode, renegotiation loops |
| T5 | SMT encoding is faithful to the DY+state machine model | Quantifier encoding choices, bitvector width truncation, UF abstraction of crypto |
| T2 | CEGAR converges within refinement budget | Complex cipher-suite interactions, environment-dependent callbacks |

The composition theorem is only as strong as its weakest link. The proposal rates each individual assumption-violation probability as low (5-20%), but these are not independent. A slicer miss (T1 violation) that excludes a callback function also removes the code that violates P4, making T3's proof *vacuously* correct on the sliced code but *unsound* on the original. The composition then produces a certificate of absence that is valid for the slice but not for the library.

This is the deepest structural problem: **the composition chain is unfalsifiable on the artifacts it produces.** If the slicer misses code, the downstream analysis cannot detect the miss. The certificate says "no attack found within bounds on the sliced code." Whether this implies "no attack in the library" depends on slicer soundness, which is assumed, not proved.

**Severity: CRITICAL.** The certificate's value depends on an unproved assumption (slicer soundness) that is buried beneath two layers of formally proved theorems. A reviewer who checks T3, T4, and T5 will find correct proofs — but the certificate is still potentially wrong because the proofs operate on a subset of the code.

---

## 3. Attack on the Slicer (Pre-Theorem, Load-Bearing)

### A-S1 — Slicer soundness is assumed, not proved [CRITICAL]

T1 states: "Every trace of the extracted state machine corresponds to a feasible execution path in the original source code." This is a *forward* soundness guarantee (no false positives in the state machine). The critical *backward* guarantee — every negotiation-relevant execution path in the original source appears in the state machine — is provided by the slicer, and the slicer has no theorem.

The slicer must handle:

1. **`SSL_METHOD` vtable dispatch.** OpenSSL's `SSL_METHOD` struct is a vtable of function pointers populated by macro-generated initializers (e.g., `IMPLEMENT_ssl3_meth_func`). The method selected at `SSL_CTX_new()` time determines which negotiation functions are called. Andersen-style points-to analysis can resolve these in principle, but the macro expansion creates indirect call sites that many implementations miss.

2. **Callback chains.** OpenSSL has at least 15 user-registerable callbacks that execute during the handshake: `SSL_CTX_set_cert_cb`, `SSL_CTX_set_verify`, `SSL_CTX_set_info_callback`, `SSL_CTX_set_msg_callback`, `SSL_CTX_set_alpn_select_cb`, `SSL_CTX_set_tlsext_servername_callback`, etc. Some of these (cert_cb, alpn_select_cb, servername_callback) directly influence cipher selection. If the slicer treats callbacks as opaque (conservative: include all reachable code from callback type), the slice bloats. If it ignores callbacks (unsound), it misses negotiation-relevant paths.

3. **Global state mutations.** `OPENSSL_init_ssl()`, `SSL_CTX_set_options()`, and `FIPS_mode_set()` modify global or context-level state that constrains subsequent negotiation. These mutations happen before the handshake, potentially in different translation units. The slicer must perform inter-procedural, cross-TU taint analysis to include them.

4. **Error-handling paths.** Many downgrade-relevant bugs live in error handling: CCS Injection (CVE-2014-0224) occurs because the state machine *fails to reject* a ChangeCipherSpec message at the wrong point. The "error path" is the absence of an expected check. Slicing for negotiation-relevant *presence* of code is different from slicing for the *absence* of a check. It's unclear how the slicer handles this distinction.

**The meta-problem:** Slicer bugs produce *silent* failures. If the slicer misses a function, no downstream analysis flags the error. The certificate says "no attack found," and the missed function is invisible. Unlike a false positive (which produces an attack trace that can be validated by replay), a false negative produces *nothing* — it looks exactly like a correct certificate of absence.

**Severity: CRITICAL.** This is the single most dangerous component in the pipeline. The proposal allocates 11K LoC and a kill gate (G1: slice ≤10K lines with ≥90% coverage), but "90% coverage" is measured against *known* negotiation functions, not unknown ones. The slicer is exactly the component where unknown unknowns are most dangerous.

### A-S2 — The 1-2% slice ratio is fragile and load-bearing [SERIOUS]

The feasibility argument depends on slicing 200K-line libraries down to 3-7K lines. If the slice is larger, every downstream step suffers:

- **Symbolic execution**: KLEE's path explosion is polynomial in code size for the merge operator's idealized model, but the constant matters. 10K lines with complex control flow may produce 10,000+ paths.
- **SMT encoding**: More states → more variables → longer solving time. The 30-minute SMT target assumes a small state machine.
- **Memory**: The 16-24 GB peak memory estimate is based on the small-slice assumption.

The 1-2% ratio is measured against *total* library code. But negotiation code does not exist in isolation. Cipher selection in OpenSSL calls into `ssl_set_cert_masks()` which calls `X509_check_private_key()` which calls into ASN.1 parsing. If the slicer follows these calls (required for soundness — certificate-type checking feeds into cipher selection), the slice grows rapidly. The proposal's ground truth ("negotiation decision logic lives in `statem_clnt.c`, `statem_srvr.c`, `ssl_ciph.c`") counts only the top-level decision functions, not their transitive callees.

**Severity: SERIOUS.** The kill gate (G1) at 15K lines is a reasonable safety valve, but "KILL if >15K" means a significant probability (~25% by the proposal's own estimate) of project failure at the slicer stage.

---

## 4. Attack on T5: SMT Encoding Correctness

### A5.1 — Decidability does not imply tractability [SERIOUS]

BV+Arrays+UF+LIA is decidable but NEXPTIME-hard in the worst case. The proposal's claim of "<30 minutes solving time" is a bet on Z3's heuristic performance, not a complexity-theoretic guarantee.

At adversary budget n=5, each adversary action branches the search tree. With 30 cipher suites and 5 adversary actions, the state space has up to 30^5 ≈ 24 million configurations before SMT-level abstraction. Z3 must reason about message terms (arrays of bitvectors), adversary knowledge (sets of terms), and state-machine transitions (UF-encoded). The formula size grows quadratically with n (adversary knowledge accumulation requires checking all pairs of known terms for derivability).

The proposal acknowledges this risk (R3, 40% probability) and provides mitigations (incremental solving, query decomposition, CEGAR). But a 40% risk of timeout on the *primary solving step* is extraordinarily high for a system whose headline claim is "bounded-complete synthesis." If Z3 times out, NegSynth can produce neither an attack trace nor a certificate — the analysis is inconclusive, and the bounded-completeness claim does not apply.

**Severity: SERIOUS.** The paper must report timeout rates honestly. If 20% of library-version configurations hit the 30-minute wall and require fallback to n=3, the certificate is weaker than advertised.

### A5.2 — The Dolev-Yao model is too abstract for some in-scope attacks [MINOR]

Standard Dolev-Yao assumes perfect cryptography: the adversary cannot break encryption, forge MACs, or factor keys. But several in-scope CVEs exploit implementation-level properties:

- **DROWN (CVE-2016-0703):** Exploits an SSLv2-specific padding oracle — an implementation artifact, not a protocol flaw. The DY model represents encryption as a perfect function; the padding oracle requires modeling encryption as *imperfect*.
- **POODLE (CVE-2014-3566):** The cipher-suite downgrade *component* (forcing SSLv3) is DY-capturable, but the actual attack requires a CBC padding oracle — again, implementation-level.
- **Lucky13 (CVE-2013-0169):** Timing side-channel, completely invisible to DY.

The proposal scopes POODLE as "partial" (version-downgrade only) and doesn't claim Lucky13. This is honest. But the bounded-completeness certificate implicitly claims "no downgrade attack" — a reader may not realize that "downgrade attack" is defined narrowly to exclude oracle-based attacks that require DY model extensions.

**Severity: MINOR.** Addressable with clear scoping language. The DY model's limitations are well-understood in the formal methods community. The paper should explicitly define "downgrade attack" to exclude implementation-level cryptographic oracles.

---

## 5. Attack on the Evaluation Design

### A-E1 — CVE selection bias [SERIOUS]

The 8 CVEs were selected because they are "clearly-in-scope" negotiation attacks. But the total universe of downgrade-related CVEs is much larger. A non-exhaustive list of potentially-in-scope CVEs *not* in the benchmark:

- **CVE-2016-2107** (AES-NI padding oracle): cipher-suite selection feeds into a code path with a padding oracle. Is the cipher-selection component in scope?
- **CVE-2020-1967** (signature algorithm crash): crash during signature algorithm negotiation. Similar to CVE-2015-0291 (in scope) but more recent.
- **CVE-2022-4304** (timing oracle in RSA decryption): RSA cipher-suite selection leads to timing-vulnerable code. In scope for the cipher-selection phase?
- **Raccoon attack (CVE-2020-1968):** DH cipher-suite selection leads to timing-vulnerable DH key exchange.

If NegSynth *could* detect the cipher-selection component of these CVEs but the evaluation excludes them, the recall metric is inflated. If NegSynth *cannot* detect them, the evaluation should explain why — this bounds the tool's scope.

**Severity: SERIOUS.** The paper should include a complete enumeration of downgrade-adjacent CVEs in the target libraries and explicitly classify each as in-scope, partially-in-scope, or out-of-scope with justification.

### A-E2 — Baseline comparison is unfair [MINOR]

Comparing NegSynth (white-box source analysis) to TLS-Attacker (black-box network testing) conflates two dimensions: the analysis technique and the information available. A fairer baseline would be:

1. **KLEE + hand-written assertions.** This isolates the contribution of the protocol-aware analysis. If a skilled security engineer writes 20 assertions about negotiation correctness and runs KLEE on the same slice, how many CVEs does plain KLEE find? If plain KLEE finds 6/8 and NegSynth finds 8/8, the marginal contribution of the protocol-aware machinery is 2 CVEs — still valuable but less dramatic.

2. **KLEE + generic veritesting on the same slice.** This isolates the contribution of T3 (the merge operator). If generic veritesting times out on 200 cipher suites but succeeds on 15, the merge operator's practical value is "scales cipher-suite analysis from 15 to 200" — a meaningful but bounded claim.

**Severity: MINOR.** The proposed comparison against tlspuffin is fair (same problem, different technique). Adding the KLEE+assertions baseline would strengthen the paper.

### A-E3 — "First bounded-completeness certificates" is unfalsifiable [MINOR]

The bounded-completeness certificate is a novel artifact: "within bounds k=20, n=5, no downgrade attack exists." How does anyone verify this is correct? Options:

1. **Independent reimplementation.** Infeasible for reviewers.
2. **Checking the SMT proof.** If the UNSAT certificate includes a proof object (Z3 can produce these), an independent checker (e.g., DRAT) can verify it. But the UNSAT proof only certifies the *formula* is unsatisfiable — it does not certify that the formula faithfully represents the library.
3. **Finding a counterexample.** If a reviewer finds a downgrade attack within the claimed bounds, the certificate is falsified. But absence of a counterexample is not evidence of correctness.

The certificate's trustworthiness ultimately reduces to trust in the pipeline — which circles back to slicer soundness (A-S1), merge correctness (A3.1), and encoding fidelity (A5.1).

**Severity: MINOR.** This is a fundamental limitation of any verification tool, not specific to NegSynth. The paper should discuss it in the limitations section.

---

## 6. Hidden Contradictions

### HC1 — "Sound slicer" built on heuristic points-to analysis [SERIOUS]

The slicer claims to soundly identify all negotiation-relevant code. It uses Andersen's points-to analysis, which is sound in theory (it over-approximates the points-to set). But:

1. **Practical implementations are buggy.** SVF (the most widely-used LLVM points-to analysis) has had soundness bugs. A 2019 study (Sui et al.) found that SVF's handling of external function stubs (exactly the category OpenSSL's assembly routines fall into) was incomplete.

2. **Assembly stubs break the analysis.** The proposal plans to "stub assembly routines in `crypto/` with C equivalents." But the stubs must faithfully model the assembly's pointer behavior. If an assembly routine modifies a pointer that the C stub does not, the points-to analysis produces an unsound result. The proposal treats stubbing as routine engineering, but any incorrectly modeled stub is a potential slicer soundness violation.

3. **The contradiction:** T1's proof assumes slicer soundness. The slicer's soundness depends on the points-to analysis's soundness. The points-to analysis's soundness depends on correct assembly stubs. Correct assembly stubs depend on manual human effort. Therefore: the "fully automated, zero human involvement" pipeline depends on manually-authored assembly stubs being correct.

**Severity: SERIOUS.** The paper should acknowledge this dependency chain and describe validation measures for the assembly stubs (e.g., differential testing: run the same negotiation scenario with assembly and C stubs, compare outputs).

### HC2 — "No human involvement" vs. 20K LoC of hand-authored protocol modules [MINOR]

The evaluation claims "zero human involvement." But the protocol modules (TLS 1.0-1.3 message grammars, SSH v2 state predicates, downgrade-freedom property templates) total ~20K LoC of human-authored Rust code. This code encodes RFC knowledge: which TLS extensions affect negotiation, what constitutes a "downgrade," which state transitions are valid.

This is not a fatal flaw — every formal verification tool requires a specification. But calling the pipeline "fully automated" when it includes 20K LoC of human-authored protocol specifications is misleading. A more honest framing: "automated *per-analysis* — no human involvement beyond the one-time protocol module authoring."

The deeper concern: **if the protocol modules contain errors, the certificates are wrong.** A mistake in the TLS 1.3 message grammar (e.g., incorrectly modeling the `supported_versions` extension) could cause the DY encoder to miss attack vectors that exploit that extension. This is the same spec-implementation gap NegSynth claims to close — but now it exists within NegSynth's own protocol modules.

**Severity: MINOR.** Address by (a) open-sourcing protocol modules for community review, (b) validating modules against RFC test vectors, (c) using terminology like "push-button analysis" rather than "zero human involvement."

### HC3 — KLEE integration risk is under-weighted [MINOR]

KLEE's last major release was 2019 (KLEE 2.3). The project is maintained but not actively developed. OpenSSL 3.x uses C constructs and LLVM IR patterns that KLEE 2.x may not fully support:

- **OpenSSL 3.x's provider architecture** introduces a plugin system with runtime-loaded shared objects. KLEE cannot symbolically execute dynamically loaded code.
- **LLVM version drift.** KLEE 2.3 targets LLVM 9-11. OpenSSL 3.x may require LLVM 15+ for correct bitcode generation. The KLEE↔LLVM version mismatch could produce incorrect bitcode interpretation.
- **C11 atomics.** OpenSSL uses `_Atomic` types for lock-free reference counting. KLEE has limited support for atomic operations.

**Severity: MINOR.** Addressable by using KLEE's development branch (which tracks newer LLVM), by compiling OpenSSL with `-fno-provider` or equivalent, and by stubbing atomic operations. But each workaround narrows the analyzed code, potentially affecting slicer completeness.

---

## 7. Severity Classification Summary

| ID | Finding | Severity | Could Invalidate |
|----|---------|----------|------------------|
| A3.1 | Algebraic properties violated by callbacks/FIPS/renegotiation | **CRITICAL** | T3, T4, certificates |
| A4.2 | Bounds k=20, n=5 lack theoretical justification | **CRITICAL** | T4 headline claim |
| A4.3 | Composition chain has unproved slicer assumption | **CRITICAL** | T4, certificates |
| A-S1 | Slicer soundness is assumed, not proved | **CRITICAL** | T1, T4, certificates |
| A3.2 | O(n) bound is for idealized code, not production | SERIOUS | T3 framing |
| A3.3 | Observable predicate under-specified | SERIOUS | T3 scope, attack classes |
| A4.1 | ε may be vacuously large | SERIOUS | T4 practical value |
| A5.1 | SMT tractability is empirical, not guaranteed | SERIOUS | Pipeline reliability |
| A-S2 | 1-2% slice ratio fragile | SERIOUS | Feasibility |
| A-E1 | CVE selection bias | SERIOUS | Evaluation validity |
| HC1 | Sound slicer depends on heuristic analysis + manual stubs | SERIOUS | Pipeline soundness |
| A5.2 | DY model too abstract for some in-scope attacks | MINOR | Scope |
| A-E2 | Baseline comparison unfair | MINOR | Evaluation |
| A-E3 | Certificate unfalsifiable | MINOR | Limitations |
| HC2 | "No human involvement" vs. 20K protocol modules | MINOR | Framing |
| HC3 | KLEE integration risk | MINOR | Feasibility |

---

## 8. Constructive Recommendations

### For CRITICAL findings:

**R1 (addresses A3.1, A4.3, A-S1): Prove slicer soundness or weaken the certificate claim.**

Option A (preferred): Formalize the slicer as a separate theorem (T0). Define "negotiation-relevant" precisely. Prove that Andersen-style points-to analysis plus the protocol-specific taint rules produces a sound over-approximation. This is significant work (~3 person-months) but converts the weakest link from an assumption to a theorem.

Option B (fallback): Weaken the certificate to say "within bounds k, n, on the *analyzed slice*, no downgrade attack exists." This is still useful — it tells users exactly what was analyzed — but it no longer claims library-level completeness.

**R2 (addresses A3.1): Handle algebraic property violations explicitly.**

For each of the four algebraic properties, define a *property checker* that runs on the slice before the merge operator is applied. When a property violation is detected (e.g., a callback-driven cipher filter violates P4), the merge operator falls back to generic veritesting for that code region. Report which fraction of the code required fallback and how it affected path count. This makes the "money plot" honest: "protocol-aware merge achieves O(n) on 85% of negotiation code and falls back to veritesting on 15%."

**R3 (addresses A4.2): Provide a theoretical upper bound on attack depth.**

For TLS without renegotiation, the handshake has a fixed maximum depth (~15 messages). Prove that any downgrade attack on a single TLS handshake requires k ≤ K_max and n ≤ N_max, where K_max and N_max are derived from the protocol specification. Then separately acknowledge that renegotiation-based attacks are unbounded and out of scope. This splits the certificate into two claims: (a) a theorem for single-handshake attacks, (b) an empirical claim for multi-handshake attacks.

### For SERIOUS findings:

**R4 (addresses A3.2): Reframe the complexity claim.**

Replace "O(n) paths" with "O(n) paths for the cipher-selection subroutine." Report the total path count for the full negotiation (including session resumption, ALPN, SNI, 0-RTT) and show the merge operator's contribution to reducing that count. This is less dramatic but more honest.

**R5 (addresses A3.3): Define the observation function explicitly.**

Publish the observation function as a formal definition in the paper. Enumerate which protocol outputs are "observable" (cipher suite, protocol version, alert codes) and which are not (timing, message contents). Map each in-scope CVE to the observation predicates it requires. This bounds the certificate's scope precisely.

**R6 (addresses A4.1): Define and measure ε rigorously.**

Define ε as the probability that a randomly sampled negotiation-relevant execution path is excluded from the slice OR incorrectly merged OR not encodable in the SMT formula. Measure this by running a random path sampler on the full (un-sliced) library and checking what fraction of sampled paths appear in the extracted state machine. This provides an empirical lower bound on completeness that does not depend on known CVEs.

**R7 (addresses A-E1): Enumerate the full CVE universe.**

List all downgrade-adjacent CVEs in the four target libraries (a literature search will find 30-50). Classify each as in-scope, partially-in-scope, or out-of-scope with a one-line justification. Report recall against this complete set, not just the curated 8.

**R8 (addresses HC1): Validate assembly stubs.**

For each assembly routine stubbed with a C equivalent, run differential testing: execute the negotiation path with the real assembly and with the C stub, compare outputs on 10,000 random inputs. Report any discrepancies. This provides empirical evidence that stubs are faithful without requiring a formal proof.

---

## 9. Bottom Line

NegSynth is an ambitious and genuinely novel project. The protocol-aware merge operator is a real idea, the end-to-end pipeline has no prior analog, and bounded-completeness certificates for production libraries would be a significant contribution. The proposal is refreshingly honest about its risks (19% compound success probability) and has well-designed kill gates.

However, the headline claim — "bounded-complete synthesis from source code" — rests on an unproved slicer, idealized algebraic properties that real code violates, and empirically-validated bounds without theoretical justification. The composition chain T1→T3→T5→T4 is formally correct *given its assumptions*, but the assumptions are load-bearing and unverified. The bounded-completeness certificate is exactly as trustworthy as the slicer is sound — and the slicer is the one component without a theorem.

The most likely failure mode is not a dramatic blowup but a quiet erosion: the slicer misses 5% of negotiation code, the merge operator collapses states it shouldn't on another 3%, and the certificate confidently declares "no attack found" while a callback-driven downgrade vector sits in the un-analyzed code. This is worse than no certificate at all — a false certificate of absence creates false confidence.

**Recommendation:** Proceed, but invest heavily in slicer validation (R1), algebraic property checking (R2), and honest framing (R4, R5). The minimal viable paper — OpenSSL only, 3-4 CVEs, certificates with explicit scope limitations — is still a strong contribution if the limitations are stated clearly. Do not over-claim.
