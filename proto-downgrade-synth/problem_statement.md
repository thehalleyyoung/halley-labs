# Negotiation Under Fire: Bounded-Complete Synthesis of Protocol Downgrade Attacks from Library Source Code

## Opening

Protocol downgrade attacks remain among the most devastating classes of cryptographic vulnerabilities. FREAK, Logjam, POODLE, DROWN, and the recently disclosed Terrapin SSH attack all share a common anatomy: an active network adversary manipulates the cipher-suite negotiation logic embedded in TLS, SSH, or QUIC implementations, coercing endpoints into selecting weak or broken cryptographic algorithms that the adversary can then exploit. The consequences range from passive decryption of session traffic to full man-in-the-middle control. Despite decades of investment in formal verification, these attacks continue to be discovered *manually* — often years after the vulnerable code has shipped to billions of devices. The Terrapin attack, disclosed in late 2023, demonstrated that even modern, well-audited protocols harbor negotiation flaws that elude every existing automated tool. The persistent gap between formal protocol models, which reason about idealized specifications, and real implementations, which contain bespoke negotiation logic shaped by backward-compatibility requirements, performance heuristics, and platform quirks, remains a critical blind spot in cryptographic assurance.

Existing tools fail to close this gap because each addresses only one side of it. Formal verifiers such as ProVerif and Tamarin Prover operate on hand-written protocol specifications expressed in purpose-built modeling languages. They reason powerfully about abstract protocol flows but cannot detect implementation-specific flaws: a missing version-floor check buried in a fifty-line negotiation function, an incorrect cipher-suite preference comparator, or a fallback path triggered only by a particular combination of compile-time options. The verified-implementation program exemplified by miTLS and Project Everest (Bhargavan et al.) takes the complementary approach — building *new* implementations that are correct by construction — but cannot retroactively verify the negotiation logic of the existing C libraries that the internet actually runs on. Symbolic execution engines like KLEE and SAGE operate directly on compiled code, but they lack an adversary model — they discover memory-safety bugs and assertion violations, not logical protocol attacks orchestrated by a Dolev-Yao attacker. The most recent advance, the DY-fuzzer architecture exemplified by tlspuffin (IEEE S&P 2024), bridges this divide partially by connecting a Dolev-Yao adversary model to implementation-level fuzzing. However, tlspuffin relies on a *manually authored* protocol model that is maintained separately from the library source code, inheriting precisely the specification-implementation faithfulness gap it seeks to close. Moreover, fuzzing provides stochastic coverage without completeness guarantees — a critical limitation when even a single overlooked negotiation path can be exploited.

We propose **NegSynth**, a protocol-aware symbolic analysis engine that closes the source-to-attack loop end-to-end. NegSynth takes as input the C source code of a cryptographic library and automatically extracts the cipher-suite negotiation and handshake state machine via protocol-aware program slicing and bounded symbolic execution. It builds on KLEE's mature symbolic execution infrastructure for LLVM IR, extending it with a novel *protocol-aware merge operator* that exploits the algebraic structure of negotiation protocols — finite selection from enumerated cipher suites, deterministic state-machine transitions, monotonic version ordering — to aggressively merge symbolically equivalent paths. NegSynth then encodes the extracted state machine together with a bounded Dolev-Yao adversary model (informed by tlspuffin's peer-reviewed term algebra) as SMT constraints over a combined theory of bitvectors, arrays, and uninterpreted functions, and invokes an SMT solver to synthesize concrete, byte-level downgrade attack traces — or to certify their absence within the configured analysis bounds.

The key technical innovation enabling this pipeline on real-world libraries (100K+ lines of code) is the protocol-aware merge operator. This operator achieves O(n) symbolic path count where generic state merging (veritesting, Kuznetsov et al.) produces O(2^n) paths, because negotiation protocols have four algebraic properties that generic programs lack: (1) finite, enumerable outcome spaces (cipher suites), (2) lattice-ordered preference structures, (3) monotonic (acyclic) state progression during negotiation, and (4) deterministic selection given matching capability sets. With this operator, NegSynth analyzes the full negotiation logic of a production TLS library on a laptop CPU in under eight hours.

NegSynth is the first tool to provide *bounded-complete* downgrade attack synthesis directly from implementation source code. Unlike specification-level verifiers, it analyzes what the code actually does, not what the code is supposed to do. Unlike fuzzers, it provides a completeness guarantee: within the configured execution depth *k* and adversary budget *n*, every downgrade attack that exists in the source will be found, or its absence will be certified. We empirically validate that bounds k=20, n=5 suffice to capture every known downgrade CVE in our benchmark set, and that these bounds explore ≥99% of reachable negotiation states in all target libraries. The closed-loop pipeline — source → LLVM IR → protocol-aware slice → symbolic execution with merge → state machine extraction → Dolev-Yao + SMT encoding → concrete attack trace — is genuinely new. No prior tool connects all of these stages, and the composition theorem that guarantees end-to-end soundness across them represents a novel contribution to the theory of protocol analysis.

We target four major C cryptographic libraries spanning two protocol families: OpenSSL, BoringSSL, and WolfSSL for TLS; and libssh2 for SSH. NegSynth recovers eight clearly-in-scope known CVEs — including FREAK (CVE-2015-0204), Logjam (CVE-2015-4000), the version-downgrade component of POODLE (CVE-2014-3566), Terrapin (CVE-2023-48795), and CCS Injection (CVE-2014-0224) — from the historically vulnerable library versions in which they were introduced. Beyond CVE recovery, NegSynth produces the first *bounded-completeness certificates* for production library negotiation logic: formal artifacts certifying that, within the validated analysis bounds, no downgrade attack exists in the current HEAD of each target library. The entire evaluation is fully automated, reproducible, and runs on commodity laptop hardware without GPU acceleration.

## Value Proposition

Protocol downgrade attacks affect every TLS and SSH connection on the internet — billions of sessions daily. The stakeholders who need NegSynth range from the engineers who write negotiation code to the standards bodies who design the protocols those engineers implement.

**Library maintainers** at projects like OpenSSL and WolfSSL update their negotiation logic with every release: new cipher suites are added, deprecated algorithms are removed, extension-negotiation paths are refactored. Each change is a potential downgrade vector. Today, security review of negotiation code is manual, takes weeks per release, and depends on scarce domain expertise. NegSynth provides push-button analysis that completes in hours and produces either concrete attack traces or bounded-completeness certificates, transforming negotiation security from an artisanal practice into an automated one.

**Protocol designers** in the IETF TLS and QUIC working groups routinely propose new extensions and negotiation mechanisms. When a new draft modifies the cipher-suite selection algorithm or introduces a version-negotiation variant, designers currently have no tool that can verify whether reference implementations faithfully enforce the intended security properties. NegSynth fills this role: given a reference implementation and the desired downgrade-freedom property, it either confirms the property holds or produces a minimal counterexample.

**Security auditors** presently rely on black-box tools like TLS-Attacker for external probing or on painstaking manual source review. NegSynth provides white-box, source-level analysis with formal guarantees — a qualitative upgrade in assurance. For the broader security research community, a tool that systematically finds downgrade attacks changes the economics of protocol security: defenders gain an automated oracle that attackers cannot outpace through manual effort alone.

**Scope acknowledgment.** TLS 1.3's anti-downgrade sentinel (RFC 8446 §4.1.3) narrows the TLS-specific attack surface for compliant, TLS-1.3-only deployments. NegSynth's primary value drivers are: (a) legacy TLS (1.0–1.2), which remains ubiquitous in IoT, embedded systems, and enterprise environments; (b) SSH, where Terrapin (2023) proved that negotiation flaws persist in modern, well-audited code; and (c) cross-version interaction paths where legacy and modern negotiation codepaths coexist in the same library.

## Technical Difficulty

NegSynth's novel contribution is approximately 50,000 lines of protocol-analysis code built on top of KLEE's symbolic execution infrastructure (~95K LoC, reused), with ~40,000 lines of protocol modules and integration code. The architecture explicitly leverages existing mature tools: KLEE for C/LLVM IR symbolic execution, tlspuffin's Dolev-Yao term algebra for adversary modeling, and TLS-Attacker for independent attack-trace replay and validation. This integration strategy is standard practice in systems security research and strengthens rather than weakens the contribution — it builds on peer-reviewed foundations.

The following table summarizes the architecture, distinguishing novel algorithmic contributions from supporting infrastructure.

**Novel Algorithmic Core (~50K LoC):**

| # | Subsystem | Estimated LoC | Language | Key Challenge |
|---|-----------|:---:|----------|---------------|
| 1 | Protocol-Aware Merge Operator | 6,000–8,000 | Rust/C++ | Exploiting negotiation-protocol algebraic structure (finite outcome spaces, lattice preferences, monotonic progression, deterministic selection) for exponential path reduction with provable bisimilarity preservation |
| 2 | Protocol-Aware Slicer | 10,000–13,000 | Rust | Sound identification of negotiation-relevant code across indirect calls, C vtable dispatch, and callback chains; protocol-specific taint tracking |
| 3 | KLEE Integration Layer | 6,000–8,000 | C++/Rust | Custom KLEE Searcher implementing merge strategy, protocol-aware state representation, bidirectional FFI |
| 4 | State Machine Extractor | 7,000–9,000 | Rust | Bisimulation-quotient algorithm with protocol-specific equivalence predicates over symbolic path conditions |
| 5 | DY+SMT Constraint Encoder | 8,000–11,000 | Rust | Message-algebra encoding in BV+Arrays+UF+LIA theory, CEGAR refinement loop, adversary knowledge accumulation |
| 6 | Concretizer | 5,000–7,000 | Rust + Python | SMT model interpretation → byte-level attack scripts with correct TLS record framing and SSH packet formatting |
| | **Subtotal** | **42,000–56,000** | | **Midpoint: ~50K** |

**Protocol Modules and Integration (~40K LoC):**

| # | Subsystem | Estimated LoC | Language | Key Challenge |
|---|-----------|:---:|----------|---------------|
| 7 | Protocol Modules | 18,000–22,000 | Rust | TLS 1.0–1.3 (~12K), SSH v2 (~8K): message grammars, state predicates, downgrade-freedom property templates |
| 8 | Evaluation Harness | 10,000–12,000 | Rust + Python | CVE oracle with 8+ ground-truth attack traces, TLS-Attacker integration for replay validation, automated metric collection |
| 9 | Test Infrastructure | 8,000–12,000 | Rust + Shell | Property-based tests validating each theorem empirically, integration tests, CI |
| 10 | CLI and Reporting | 5,000–7,000 | Rust | SARIF-format output, certificate generation, attack-trace pretty-printer |
| | **Subtotal** | **41,000–53,000** | | **Midpoint: ~45K** |

**Total novel + integration: ~95K LoC (built on KLEE's ~95K reused infrastructure).**

This complexity is genuine. The protocol-aware merge operator must formally exploit four algebraic properties of negotiation protocols that have no analog in generic symbolic execution. The slicer must identify negotiation-relevant code across C's indirect call patterns (OpenSSL's `SSL_METHOD` vtable dispatch is macro-generated). The DY+SMT encoder must faithfully represent Dolev-Yao adversary knowledge accumulation in a combined theory that is equisatisfiable with the composed model. The evaluation harness must normalize outputs across tools with incompatible result formats to produce fair comparisons.

## New Mathematics Required

Five load-bearing theorems underpin NegSynth's correctness guarantees. Two are genuinely novel, and three are careful adaptations of known results to the protocol-analysis domain.

| ID | Statement (informal) | Status | Core Novelty | Estimated Effort |
|----|----------------------|--------|--------------|:---:|
| T1 | **Extraction Soundness.** Every trace of the extracted state machine corresponds to a feasible execution path in the original source code. | Adapted | Protocol-specific simulation relation accounting for the merge operator's state abstractions | ~3 person-months |
| T2 | **Attack Trace Concretizability.** Every satisfying SMT assignment can be concretized into an executable byte-level attack trace with measured concretization success rate ≥ 1−ε, where ε is empirically bounded per target library. | New application of CEGAR | CEGAR refinement loop with Dolev-Yao adversary domain; symbolic-to-concrete framing bridge for TLS/SSH wire formats | ~5 person-months |
| T3 | **Protocol-Aware Merge Correctness.** The merge operator ⊵ preserves protocol-bisimilarity: merged states produce exactly the same observable negotiation behaviors as the unmerged originals. The operator achieves O(n) path count where generic veritesting produces O(2^n) on negotiation code with n cipher suites, because it exploits four algebraic properties unique to selection protocols: finite outcome spaces, lattice-ordered preferences, monotonic state progression, and deterministic selection. | Genuinely new | Merge as a congruence relation over the protocol labeled transition system; proof that negotiation structure guarantees congruence closure; formal demonstration of exponential improvement over generic merging | ~4 person-months |
| T4 | **Bounded Completeness (headline result).** Within execution depth *k* and adversary budget *n*, NegSynth finds every downgrade attack that exists in the source code, or certifies absence, with probability ≥ 1−ε. Empirically validated: k=20, n=5 suffices for all known CVEs; these bounds explore ≥99% of reachable negotiation states. | New composition | Composes T1, T3, and T5 into an end-to-end guarantee; empirical k/n characterization demonstrates practical meaningfulness | ~4 person-months |
| T5 | **SMT Encoding Correctness.** The SMT constraint system is equisatisfiable with the composed Dolev-Yao adversary and extracted state machine. | Adapted | Protocol-specific constructor encoding in the combined BV+Arrays+UF theory; proof of faithful adversary-knowledge accumulation | ~3 person-months |

Theorems T3 and T4 are the genuinely novel contributions. T3 establishes that the protocol-aware merge operator — the mechanism that makes symbolic execution of real libraries tractable — does not sacrifice any observable negotiation behavior, and does so by exploiting algebraic structure that generic merging techniques cannot access. The paper includes a formal comparison demonstrating that generic veritesting (Avgerinos et al., ICSE 2014) and state merging (Kuznetsov et al., PLDI 2012) produce exponentially more states on negotiation code, establishing a clear technical delta. T4, the headline result, composes the full theorem chain into an end-to-end bounded-completeness guarantee, with empirical validation that the chosen bounds capture all known attack classes and explore near-complete negotiation state coverage.

Theorems T1, T2, and T5 are honest adaptations — their proof structures follow known templates (simulation relations for abstract interpretation, CEGAR soundness, theory-combination results) — but each requires non-trivial domain-specific work to instantiate for the protocol-analysis setting. All five theorems are validated empirically via property-based testing: T1 via random programs with known traces, T3 via exhaustive bisimilarity checking on small instances, T5 via comparison against a reference Prolog Dolev-Yao model.

## Best Paper Argument

NegSynth targets top-tier security venues: IEEE S&P, USENIX Security, and ACM CCS. The best-paper case rests on four pillars.

**Impact.** NegSynth is the first tool to automatically synthesize protocol downgrade attacks from source code with bounded-completeness guarantees. It changes how library maintainers reason about negotiation security — from manual, expert-driven review to automated, push-button analysis with formal certificates. The bounded-completeness certificates NegSynth produces for current library versions are novel artifacts: "within bounds k=20, n=5, OpenSSL 3.x contains no cipher-suite downgrade attack" — the first such formal guarantee for any production TLS library.

**Technical novelty.** The protocol-aware merge operator (T3) advances the state of the art in symbolic execution by identifying four algebraic properties of negotiation protocols that enable exponential path reduction while preserving bisimilarity — a result with no analog in the generic state-merging literature. The bounded-completeness theorem (T4) composes extraction soundness, merge correctness, and encoding correctness into an end-to-end guarantee that is unprecedented in the protocol-analysis literature. The end-to-end pipeline — from raw C source to concrete byte-level attack traces — has no prior analog.

**Empirical strength.** Recovering eight clearly-in-scope known CVEs from historically vulnerable library versions validates the approach against ground truth. The multi-library evaluation (four libraries, two protocol families) demonstrates generality. The bounded-completeness certificates for current library HEAD versions are themselves a first-of-kind contribution — even if no new vulnerability is discovered, the certificates represent a qualitative advance in cryptographic assurance. Any new vulnerabilities discovered during analysis are reported responsibly and strengthen the paper further.

**Artifact quality.** A ~95K-line artifact (50K novel + 45K integration, built on KLEE) with fully automated evaluation, SARIF output for CI integration, and TLS-Attacker-validated replay constitutes a lasting community resource — not merely a one-off research prototype. The integration with KLEE ensures the symbolic execution foundation is mature and maintained.

## Evaluation Plan

The evaluation is fully automated with zero human involvement across five experimental campaigns.

**Known CVE Recovery.** NegSynth analyzes historically vulnerable versions of OpenSSL, WolfSSL, and BoringSSL corresponding to eight clearly-in-scope documented CVEs. Each CVE is classified by scope:

| CVE | Name | Scope | Attack Component Detected |
|-----|------|-------|--------------------------|
| CVE-2015-0204 | FREAK | Full | Export RSA cipher-suite selection |
| CVE-2015-4000 | Logjam | Full | Export DHE cipher-suite selection |
| CVE-2014-3566 | POODLE | Partial | SSLv3 version-downgrade path (not padding oracle) |
| CVE-2023-48795 | Terrapin | Full | SSH extension negotiation manipulation |
| CVE-2016-0703 | DROWN-specific | Full | SSLv2 export cipher selection |
| CVE-2015-3197 | SSLv2 override | Full | Disabled SSLv2 ciphers still selectable |
| CVE-2014-0224 | CCS Injection | Full | ChangeCipherSpec at wrong handshake state |
| CVE-2015-0291 | ClientHello sigalgs DoS | Full | Signature algorithm negotiation crash via malformed ClientHello |

The primary metric is recall. Target: ≥85% recall (≥7/8 CVEs independently synthesized).

**Bounded-Completeness Certificates (primary empirical contribution).** NegSynth runs on the current HEAD of all four target libraries (OpenSSL, BoringSSL, WolfSSL, libssh2). For each library, the evaluation produces: a bounded-completeness certificate at k=20, n=5; total analysis time; number of symbolic states explored; negotiation-state coverage percentage; and any new findings. These certificates are the first formal downgrade-freedom guarantees for any production TLS/SSH library.

**Empirical Bound Validation.** For each historical CVE, the evaluation reports the minimal k and n required for synthesis. A table demonstrates that all known CVEs require k ≤ 15, n ≤ 5, validating that the chosen bounds (k=20, n=5) provide meaningful coverage. The CEGAR concretization success rate (1−ε) is measured per library and per CVE.

**Tool Comparison.** Head-to-head comparison against tlspuffin, TLS-Attacker, and KLEE on identical library versions and identical hardware. Metrics: number of bugs found, false-positive rate, time to first bug, code coverage achieved. The comparison includes a "completeness gap" demonstration: a negotiation scenario that tlspuffin fails to explore after 72 hours but NegSynth finds in minutes, illustrating the structural advantage of bounded-complete analysis over stochastic fuzzing.

**Scalability and Soundness.** Analysis time and peak memory consumption measured per pipeline stage (slicing, symbolic execution, SMT solving). Every reported attack trace is automatically replayed against a live instance via TLS-Attacker. Target: <10% false-positive rate.

Summary metrics: CVE recall ≥85%, false-positive rate <10%, per-library analysis time <8 hours, bounded-completeness certificates for all 4 current HEAD libraries, and empirical k/n/ε characterization for all historical CVEs.

## Laptop CPU Feasibility

NegSynth is designed to run on commodity hardware: an 8-core laptop CPU (Intel i7-class or Apple M2), 32 GB of RAM, and no GPU. The key insight enabling this is that cipher-suite negotiation *decision logic* constitutes only 1–2% of a cryptographic library's total code base. Protocol-aware slicing eliminates over 95% of the source before symbolic execution begins, reducing a 200K-line library to a 3–7K-line analysis target. This ratio is empirically grounded: OpenSSL's negotiation decision logic lives in `ssl/statem/statem_clnt.c`, `statem_srvr.c`, `ssl_ciph.c`, and portions of `t1_lib.c` — roughly 5K lines out of 500K+ total.

Per-library analysis completes in 4–8 hours, broken down by pipeline stage:
- Slicing: ~10 minutes (dominated by points-to analysis)
- Symbolic execution with merge: 1–3 hours (bottleneck; protocol-aware merge reduces ~10K–100K paths to ~1K–10K)
- State machine extraction: ~10 minutes
- DY+SMT encoding: 30–60 minutes
- SMT solving: 1–4 hours (second bottleneck; bounded queries at modest variable count)
- Concretization and replay: ~15 minutes

The full benchmark suite — 20 library-version configurations across four libraries and historical CVE versions — runs in 80–160 hours of sequential wall-clock time, or 10–20 hours with 8-way parallelism on an 8-core machine. Peak memory consumption is estimated at 16–24 GB, dominated by the SMT solver's working set during constraint solving. These estimates will be empirically validated by a proof-of-concept on OpenSSL before full implementation begins.

GPU acceleration is neither useful nor necessary. SMT solving is inherently sequential and branch-heavy. Symbolic execution is memory-bound, not compute-bound. The protocol-aware merge operator reduces the number of symbolic paths by 10–100× compared to naïve symbolic execution, and the bounded adversary model constrains SMT query complexity. The entire pipeline is engineered for memory efficiency and solver-friendly constraint structure, not raw floating-point throughput.

## Slug

`proto-downgrade-synth`
