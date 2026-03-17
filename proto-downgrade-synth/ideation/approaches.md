# NegSynth: Three Competing Technical Approaches

## Domain Vision Document — Ideation Phase

**Project:** Negotiation Under Fire: Bounded-Complete Synthesis of Protocol Downgrade Attacks from Library Source Code
**Prior Verification Scores:** V7 / D6 / BP6 / L6 — CONDITIONAL CONTINUE
**Key Amendments Incorporated:** C-only scope, honest LoC framing (~50K novel), certificate-first empirical strategy, formal veritesting comparison

---

## Approach A: Protocol-Aware Symbolic Execution with Merge Operator

*"The Algebraic Merge" — exploit the mathematical structure that makes negotiation different from arbitrary code*

### 1. Extreme Value Delivered

**Who desperately needs it:** OpenSSL/WolfSSL maintainers who ship negotiation changes quarterly with zero formal tooling; IETF working group chairs evaluating draft extensions against reference implementations; FIPS certification bodies needing machine-checkable downgrade-freedom evidence.

**What they get:** Push-button tool: C source → LLVM IR → protocol-aware slice → symbolic execution with algebraic merge → state machine → DY+SMT → concrete byte-level attack traces OR bounded-completeness certificates. The certificates are the headline deliverable — "within bounds k=20, n=5, OpenSSL 3.x HEAD contains no cipher-suite downgrade attack" — the first such guarantee for any production TLS/SSH library. Attack traces, when found, include concrete byte sequences with correct TLS record framing or SSH packet formatting, replayable via TLS-Attacker.

**Value magnitude:** Transforms negotiation security from artisanal expert review (weeks per release, scarce domain expertise) to automated analysis (hours, laptop CPU). A single missed negotiation path can produce a Terrapin-class vulnerability affecting billions of connections.

### 2. Why Genuinely Difficult as Software Artifact

**Hard Subproblem 1: Protocol-Aware Merge Operator (crown jewel).** Generic state merging (Kuznetsov et al. PLDI 2012, veritesting Avgerinos et al. ICSE 2014) produces O(2^n) paths on code that branches over n cipher suites. The merge operator must identify four algebraic properties — (1) finite enumerable outcome spaces, (2) lattice-ordered preferences, (3) monotonic state progression, (4) deterministic selection given matching capabilities — and exploit them to achieve O(n) paths. Requires defining a protocol-specific congruence relation over symbolic states that preserves all observable negotiation behaviors (bisimilarity).

**Hard Subproblem 2: Sound Slicing Across C Indirection.** OpenSSL's negotiation logic is dispatched through `SSL_METHOD` vtables (macro-generated), `SSL_CTX` callback chains, and `STACK_OF(SSL_CIPHER)` iterator patterns. Protocol-aware points-to analysis required.

**Hard Subproblem 3: DY+SMT Encoding Fidelity.** Dolev-Yao adversary knowledge accumulation encoded in combined theory of bitvectors, arrays, uninterpreted functions, and LIA. CEGAR loop refines concretization failures.

**Architecture:**
```
C Source → [LLVM IR via Clang] → [Protocol-Aware Slicer] → Negotiation Slice (~3-7K lines)
  → [KLEE + Merge Operator] → Symbolic Execution Traces
  → [State Machine Extractor] → Bisimulation-Quotient State Machine
  → [DY+SMT Encoder] → SMT Constraints (BV+Arrays+UF+LIA)
  → [Z3 Solver] → SAT: Attack Model | UNSAT: Certificate
  → [Concretizer/CEGAR] → Byte-Level Trace or Bounded-Completeness Certificate
  → [TLS-Attacker Replay] → Validated Attack / Confirmed Certificate
```

**Reused:** KLEE (~95K LoC), tlspuffin's DY term algebra, TLS-Attacker, Z3, Clang/LLVM
**Novel:** ~50K LoC — merge operator (~7K), slicer (~11K), KLEE integration (~7K), state machine extractor (~8K), DY+SMT encoder (~10K), concretizer (~6K)

### 3. New Math Required

**T3 (Protocol-Aware Merge Correctness) — genuinely new, ~4 person-months.** The merge operator ⊵ preserves protocol-bisimilarity: merged states produce exactly the same observable negotiation behaviors. Achieves O(n) vs O(2^n) by exploiting four algebraic properties. Proof by structural induction on negotiation phase.

**T4 (Bounded Completeness) — new composition, ~4 person-months.** Composes T1 (extraction soundness), T3, T5 (SMT encoding correctness) into end-to-end guarantee via three-level simulation chain.

**T1 (Extraction Soundness) — adapted, ~3 person-months.** Simulation relation between source-level execution states and state-machine states.

### 4. Best-Paper Potential

- *"The money plot"*: O(2^n) vs O(n) path reduction on OpenSSL negotiation
- *Certificates as artifacts:* First machine-checkable bounded-completeness certificates for production TLS/SSH
- *Eight CVEs recovered end-to-end* including Terrapin (2023)
- *End-to-end pipeline novelty:* No prior tool connects all stages

**Target:** IEEE S&P, USENIX Security, ACM CCS. Best-paper probability: 5-10%.

### 5. Hardest Technical Challenge

Making the merge operator work on real OpenSSL code — `ssl3_choose_cipher` is 400 lines with priority lists, disabled-cipher masks, `#ifdef` guards, FIPS overrides, and callback hooks. Algebraic properties hold semantically but are obscured syntactically. Addressed via: (1) slicer normalizes indirect calls, (2) merge on symbolic state predicates not syntax, (3) kill gate at Week 6 validates ≥10x path reduction.

### 6. Scores

| Dimension | Score | Rationale |
|-----------|:-----:|-----------|
| **Value** | **8** | Proven-lethal problem (Terrapin 2023). Certificates immediately useful. |
| **Difficulty** | **7** | ~50K novel LoC with genuinely hard algorithmic core. KLEE integration non-trivial. |
| **Potential** | **7** | Novel merge operator with formal complexity improvement. Certificate-first framing strong. |
| **Feasibility** | **6** | KLEE integration main risk. SMT scalability secondary risk. |

---

## Approach B: Abstract Interpretation with Negotiation Lattice Domain + CEGAR

*"The Over-Approximate Oracle" — trade precision for speed via a custom abstract domain, then refine*

### 1. Extreme Value Delivered

**Who desperately needs it:** CI/CD pipelines at organizations deploying TLS/SSH libraries who need *fast* per-commit security verdicts. Library maintainers wanting 15-minute smoke tests before every merge.

**What they get:** Two-phase tool: Phase 1 (abstract interpretation, ~10-15 min) over-approximates all reachable negotiation states and flags potential downgrade paths. Phase 2 (CEGAR-driven targeted symbolic execution) refines each candidate. 10-50x faster than Approach A for common case (no vulnerability present).

### 2. Why Genuinely Difficult as Software Artifact

**Hard Subproblem 1: Designing the Negotiation Abstract Domain.** Four sub-domains composed into reduced product: CipherReach (powerset lattice over cipher-suite identifiers), VersionBounds (interval domain over version partial order), Phase (finite-height lattice matching protocol state machine), TaintMap (adversary-controlled fields). The reduced product reduction operator maintaining relational cross-domain invariants is the crown jewel.

**Hard Subproblem 2: CEGAR Refinement.** Protocol-aware interpolation generating predicates in negotiation-domain vocabulary.

**Hard Subproblem 3: Sound Abstract Interpreter for LLVM IR.** Handling pointer arithmetic, indirect calls, and `memcpy` over cipher-suite arrays with custom non-numerical domain.

**Architecture:**
```
C Source → [LLVM IR] → [Slicer] → Negotiation Slice
  → [Abstract Interpreter + Negotiation Domain] → Over-Approximate State Graph
  → [Downgrade Property Checker] → Candidate Violations
  → [CEGAR Loop]:
       → [Targeted Symex (KLEE)] → Concrete Trace?
       → YES: Attack Trace → [TLS-Attacker Replay]
       → NO: [Protocol-Aware Interpolation] → Refined Domain → Re-analyze
  → No candidates: Bounded-Completeness Certificate
```

**Novel:** ~55K LoC — negotiation domain + reduced product (~12K), LLVM IR abstract interpreter (~15K), CEGAR engine (~10K), protocol-aware interpolation (~5K), slicer (~11K), concretizer (~6K)

### 3. New Math Required

**B1 (Negotiation Domain Soundness) — genuinely new, ~5 person-months.** Abstraction α and concretization γ for 4-sub-domain reduced product. 6 pairwise interaction soundness arguments.

**B2 (CEGAR Termination and Precision) — new application, ~4 person-months.** Terminates after ≤|Predicates_max| iterations via finite protocol vocabulary.

**B3 (Domain-Relative Bounded Completeness) — new, ~3 person-months.** Absence certificate sound for attacks expressible in negotiation domain vocabulary.

### 4. Best-Paper Potential

- First protocol-semantic abstract domain in the literature
- 10-50x speed advantage enables CI integration — compelling deployment story
- Cross-community appeal (PL + security)
- Protocol-aware CEGAR with provable termination

**Target:** PLDI, POPL, ACM CCS. Best-paper probability: 8-12% at PL venues, 4-8% at security venues.

### 5. Hardest Technical Challenge

Achieving sufficient precision to avoid false-positive explosion. Mitigated by: (1) aggressive reduced-product reduction, (2) predicate seeding from CVE patches, (3) fallback to full symex at 50 CEGAR iterations.

### 6. Scores

| Dimension | Score | Rationale |
|-----------|:-----:|-----------|
| **Value** | **8** | CI speed is transformative. Same certificate output as A but much faster. |
| **Difficulty** | **8** | Novel domain design intellectually harder. Custom LLVM AI from scratch. |
| **Potential** | **8** | Novel domain publishable at PLDI/POPL. Cross-community appeal. |
| **Feasibility** | **5** | Precision risk. No large reused engine. Custom C AI is multi-year project. |

---

## Approach C: Differential Specification Mining + Targeted Verification

*"The Consensus Oracle" — let N libraries vote on correct behavior, then verify deviants*

### 1. Extreme Value Delivered

**Who desperately needs it:** Security researchers hunting 0-days; maintainers of minority implementations (WolfSSL, mbedTLS) wanting consistency assurance against de facto standards; supply-chain security teams deploying multiple TLS stacks.

**What they get:** Tool taking N ≥ 3 library implementations as input, identifying behavioral deviations where libraries disagree on negotiation outcome, ranking by security impact, and verifying each candidate with targeted symbolic execution. Output: differential negotiation report with concrete attack traces for exploitable deviations, or cross-library consistency certificate.

**Value magnitude:** Discovers vulnerabilities *invisible to single-library analysis* — interoperability bugs where one library's interpretation differs from others. Terrapin-class attacks arise precisely from such interpretation differences.

### 2. Why Genuinely Difficult as Software Artifact

**Hard Subproblem 1: Semantic Alignment Across Libraries.** OpenSSL, BoringSSL, WolfSSL have completely different APIs, data structures, and naming conventions for the same concepts. Must establish semantic correspondence without manual annotation.

**Hard Subproblem 2: Complete Negotiation Input Space Generation.** ~300 cipher suites × ~10 versions × ~50 extensions × compression × signature algorithms. Covering-design theory to generate sufficient scenarios covering all behavioral equivalence classes.

**Hard Subproblem 3: Security Impact Ranking.** Distinguishing exploitable deviations from benign design choices. Requires security-strength partial order and adversary-influence analysis.

**Architecture:**
```
Libraries L1..Ln (C Source) → [Per-Library Slicing] → Negotiation Slices
  → [Semantic Alignment Engine] → Common Outcome Space
  → [Differential Scenario Generator] → Negotiation Scenarios via Covering Designs
  → [Per-Library Execution] → Outcome Maps
  → [Deviation Detector] → Behavioral Deviations
  → [Security Impact Ranker] → Ranked Candidates
  → [Targeted Verifier (KLEE)] → Confirmed Attacks / Benign Classifications
  → [Cross-Library Certificate] → Consistency Assurance
```

**Novel:** ~45K LoC — semantic alignment (~8K), scenario generator (~10K), deviation detector + ranker (~7K), targeted verification (~8K), slicer (~11K), concretizer (~6K)

### 3. New Math Required

**C1 (Differential Completeness) — genuinely new, ~5 person-months.** Covering-design bound B(n,k) guarantees all pairwise behavioral deviations within k interaction depth. Novel connection between combinatorial testing and protocol security.

**C2 (Deviation Exploitability) — genuinely new, ~4 person-months.** Game-theoretic characterization connecting deviations to Dolev-Yao adversary strategies.

**C3 (Cross-Library Consistency Certificate) — composition, ~3 person-months.** Combines C1 + C2.

### 4. Best-Paper Potential

- Fundamentally different approach — multiple libraries vote on correctness
- Discovers interoperability vulnerabilities invisible to single-library analysis
- Covering-design theory applied to protocol testing — novel connection
- Practical deployment for heterogeneous TLS stacks

**Target:** IEEE S&P, ISSTA, USENIX Security. Best-paper probability: 6-10% at security, 10-15% at testing venues.

### 5. Hardest Technical Challenge

Semantic alignment across libraries without manual annotation. Mitigated by: (1) output-based alignment at wire-protocol level (IANA cipher IDs standardized), (2) thin API harnesses (~500 LoC per library), (3) localized alignment for targeted verification.

### 6. Scores

| Dimension | Score | Rationale |
|-----------|:-----:|-----------|
| **Value** | **9** | Discovers vulnerability class invisible to single-library analysis. Cross-library certs uniquely valuable. |
| **Difficulty** | **7** | Semantic alignment hard. Covering-design instantiation non-trivial. |
| **Potential** | **8** | Fundamentally different approach. High novelty at security + testing venues. |
| **Feasibility** | **7** | Output-based alignment avoids hardest problem. KLEE reuse minimal and focused. |

---

## Comparative Summary

| Dimension | A: Algebraic Merge | B: Abstract Interpretation | C: Differential Mining |
|-----------|:---:|:---:|:---:|
| **Value** | 8 | 8 | 9 |
| **Difficulty** | 7 | 8 | 7 |
| **Potential** | 7 | 8 | 8 |
| **Feasibility** | 6 | 5 | 7 |
| **Composite** | **7.0** | **7.25** | **7.75** |
| **Primary Novelty** | Protocol-aware merge with O(2^n)→O(n) | Novel negotiation abstract domain | Covering-design differential completeness |
| **Crown Theorem** | T3: Merge bisimilarity preservation | B1: Domain soundness | C1: Differential completeness |
| **Analysis Time** | 4-8 hours/library | 15-30 min/library | 2-4 hours for N libraries |
| **Certificate Type** | Single-library bounded completeness | Domain-relative bounded completeness | Cross-library consistency |
| **Risk Profile** | KLEE integration, SMT scalability | Precision explosion, custom AI build | Semantic alignment, novelty gap |
| **Novel LoC** | ~50K | ~55K (likely 70-85K) | ~45K |
