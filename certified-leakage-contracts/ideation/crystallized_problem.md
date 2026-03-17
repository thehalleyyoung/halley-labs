# Certified Leakage Contracts: Compositional Quantitative Bounds for Speculative Side Channels in Cryptographic Binaries

## Problem Slug

`certified-leakage-contracts`

---

## Problem Statement

Side-channel security of deployed cryptographic software is assessed through methods that are monolithic, non-compositional, and speculation-unaware. When a crypto library maintainer audits a release for timing leaks, the analysis is bound to that specific binary, compiler version, optimization level, and CPU model. The result does not compose: if a TLS library calls into a separately-audited bignum implementation, there is no mechanism to derive a whole-program guarantee from the component analyses. And the result ignores speculative execution: transient instructions under Spectre-PHT, BTB, STL, and RSB misspeculation access secret-dependent cache lines before being squashed, leaving microarchitectural footprints invisible to every deployed source-level or architectural-semantics analysis. Existing tools occupy disjoint, insufficient points in the design space. CacheAudit (TISSEC 2015) computes quantitative cache-leakage bounds via abstract interpretation but is monolithic, x86-32-only, and ignores speculation. Spectector (S&P 2020) formalizes speculative non-interference but is qualitative (yes/no), relies on symbolic execution that does not scale, and produces no reusable contract. Binsec/Rel (S&P 2020) performs binary-level relational symbolic execution for constant-time verification—a boolean property that cannot express "leaks at most k bits." None of these tools can take compiled crypto code and, within minutes, produce a per-function quantitative leakage bound that accounts for speculative execution and composes across function boundaries.

We propose a compositional abstract-interpretation framework operating on x86-64 crypto binaries via an existing binary-analysis substrate (BAP, angr, or Ghidra). **We do not build a binary lifter from scratch**; instead, we build a 3–5K LoC adapter layer atop an established lifter, accepting its IR as input and restricting the supported instruction subset to ~150 crypto-critical instructions (general-purpose arithmetic, memory operations, core SIMD/AVX2, AES-NI; AVX-512 is deferred). The core technical construction is a reduced product abstract domain D_spec ⊗ D_cache ⊗ D_quant combining (1) a bounded speculative-window reachability domain tracking which program points are reachable under transient execution within a parameterized speculation window W, (2) a tainted abstract cache-state domain extending must/may analysis with per-line secret-dependence annotations, and (3) a quantitative channel-capacity domain accumulating bits of leakage via abstract counting over taint-restricted cache configurations. A reduction operator ρ propagates constraints downward: D_spec constrains which transient memory accesses reach D_cache (pruning infeasible speculative paths), and D_cache constrains D_quant (only tainted lines contribute to the count). This reduced product—the first abstract domain simultaneously modeling speculative reachability and cache channel capacity—is the primary theoretical novelty; the reduction operator ρ is genuinely new abstract-interpretation theory with no direct prior art, and it is the mathematical engine that makes the combination of speculation awareness, quantitative bounds, and compositionality tractable.

Per-function analysis produces leakage contracts of the form (τ_f : D_cache → D_cache, B_f : D_cache → ℝ≥0), where τ_f is an abstract cache-state transformer and B_f maps an initial abstract cache state to a leakage bound in bits. Sequential composition follows the rule B_{f;g}(s) = B_f(s) + B_g(τ_f(s))—the leakage of f;g is f's leakage plus g's leakage starting from f's output cache state—proved sound via a simulation argument over the non-Galois counting domain. The composition theorem is the intellectual crown jewel and is proved *before* implementation begins, tested against three representative crypto patterns (AES rounds, ChaCha20 quarter-rounds, Curve25519 scalar multiply) to validate that the required independence condition admits real code.

This framework is positioned as the software-side complement to LeaVe (CCS 2023 Distinguished Paper), which verifies that hardware RTL implementations of RISC-V processors satisfy ISA-level leakage contracts. LeaVe addresses the question "does the hardware honor the contract?" Our tool addresses the dual question: "given a leakage contract C specifying cache geometry, replacement policy, and speculation behavior, does the software leak at most ε bits?" Together they yield end-to-end guarantees. The analysis is parameterized by a microarchitectural contract C = (cache_geometry, replacement_policy, speculation_model, observation_granularity), following the contract framework of Guarnieri, Köpf, and Reineke. For v1, we hardcode 2–3 concrete configurations rather than implementing a full contract specification language: (1) ARM Cortex-A76 LRU with no speculation, (2) ARM Cortex-A76 LRU with bounded Spectre-PHT (W ≤ 50), and (3) a generic 8-way LRU configuration for sensitivity analysis. A full μarch contract language with parser, type checker, and contract comparison is deferred to a follow-up paper (see Scope Decisions and Deferrals). We operate under *assumed* contracts (since no formal hardware verification exists for proprietary Intel/AMD microarchitectures), which is strictly better than the status quo of implicit, unauditable assumptions about cache behavior.

**Scope-honest characterization of v1.** The v1 tool targets LRU replacement policy, not Intel tree-PLRU. The reason is principled: exact counting of reachable PLRU states is #P-hard (Reineke et al. 2007), and sound over-approximation under PLRU can introduce 10–50× over-estimation (13–22 spurious bits per 4 tainted sets, per Reineke et al. 2008). CacheAudit published successfully with LRU-only analysis. We target ARM Cortex-A series processors (which use LRU) as the primary evaluation platform, and we explicitly quantify the LRU→PLRU over-approximation gap as a known limitation. For Intel hardware, our bounds are sound but conservative; the tool remains useful for *regression detection* (see below) even when absolute bounds are loose.

What distinguishes this framework from all prior tools is the reduced product domain D_spec ⊗ D_cache ⊗ D_quant with its speculative reduction operator ρ—a genuinely new abstract-interpretation construction that enables four properties no existing system achieves jointly: (1) quantitative leakage bounds in bits (not boolean constant-time verdicts), (2) speculative-execution awareness (modeling transient cache pollution under bounded misspeculation), (3) binary-level analysis (catching compiler-introduced leaks invisible to source-level tools), and (4) compositional per-function contracts with a proved-sound composition rule. CacheAudit provides (1) and (3) (quantitative bounds on x86-32 binaries) but not (2) or (4). Spectector provides (2) on assembly-level code but only qualitatively, with no quantitative bounds or compositionality. Binsec/Rel provides (3) with boolean constant-time verdicts only. LeaVe provides compositionality and contracts on the hardware side only. No tool provides even three of these four simultaneously. Machine-checkable certificates—a fifth property from the original design—are deferred to a follow-up paper to keep scope focused on the novel domain construction and composition theory (see Scope Decisions and Deferrals).

**Regression detection as a first-class use case.** Even when absolute leakage bounds are conservative (due to LRU→PLRU over-approximation or widening imprecision), *changes* in bounds between binary versions reliably flag introduced leaks. This is the "real killer app" that is robust to modeling imprecision: a CI pipeline that runs the analyzer on every commit and alerts when any function's bound increases transforms the PLRU imprecision, widening imprecision, and composition imprecision from existential threats into tolerable over-approximations. The evaluation plan treats regression detection as a primary benchmark, not a side benefit.

The longer-term vision remains leakage contracts as a supply-chain primitive. Every function in a cryptographic library ships with a contract asserting its worst-case leakage under a specified microarchitectural model. Downstream consumers compose these contracts to derive whole-program bounds. CI pipelines flag leakage regressions on every commit. Hardware vendors express speculation behavior as formal contracts; software teams verify compliance. This is the same transformation that static type-checking brought to type safety: side-channel security becomes a continuous, automated, compositional property rather than a point-in-time, manual, monolithic assertion. The v1 tool delivers the mathematical and engineering foundation; the supply-chain vision drives v2+.

---

## Value Proposition

**Who needs this desperately:**

- **Crypto library maintainers** (OpenSSL, BoringSSL, libsodium, AWS-LC, WolfSSL). OpenSSL has shipped binaries with compiler-introduced timing leaks (CVE-2018-0734, CVE-2018-0735—ECDSA nonce leaks from optimizer transforms). BoringSSL's constant-time test suite cannot detect speculative leaks. libsodium's documentation warns that "constant-time guarantees apply to the source code, not the binary." These teams need a tool that runs on compiled output, accounts for speculation, and produces quantitative bounds they can threshold against release criteria. The direct audience is narrow (~50–100 crypto library maintainers globally), but the indirect beneficiaries—deployed systems relying on these libraries—number in the hundreds of millions.

- **FIPS 140-3 and Common Criteria auditors.** NIST's IG 2.4.A requires evidence of side-channel resistance; currently, labs accept vague prose. Quantitative bounds with formal soundness arguments would provide concrete, reproducible evidence. (Note: machine-checkable certificates that auditors could verify in seconds are a v2 deliverable; v1 provides the bounds and soundness proofs that certificates would attest to.)

- **Compiler developers** inserting speculation barriers (LFENCE, Speculative Load Hardening). They need quantitative feedback: does this barrier reduce leakage from 4.2 bits to 0? Current tools offer only boolean verdicts.

- **Hardware vendors** defining μarch contracts. Intel and ARM already publish informal speculation rules. Leakage contracts give these rules formal teeth—vendors specify the contract, software producers verify against it, cleanly separating hardware and software responsibility.

**What becomes possible:**

- CI regression detection: a GitHub Action runs the analyzer on every PR, flagging leakage regressions with per-function bit-level bounds. Even with conservative absolute bounds, *changes* between versions are precisely captured. This is the highest-impact near-term use case.
- Compositional guarantees: a TLS library derives whole-program bounds from per-function contracts of its crypto dependency—no re-analysis of the dependency required.
- Contract sensitivity analysis: the same binary analyzed under multiple hardcoded contracts (no-speculation, Spectre-PHT, full-speculative) reveals the marginal cost of each threat model.
- Quantitative barrier assessment: compiler teams measure the exact bit-level impact of inserting or removing speculation barriers.

**How this complements LLM-based side-channel triage.** In 2025, LLMs can triage many practical side-channel cases in seconds: identifying secret-dependent branches, variable-time lookups, and missing `cmov` instructions. This tool does not compete with LLM-based triage for these pattern-matching cases. Instead, it targets the residual cases that LLMs structurally cannot handle: subtle quantitative leakage from cache-line sharing patterns, PLRU-specific effects, speculative-window interactions, and compositional reasoning across function boundaries. More fundamentally, the tool provides *formal guarantees*—proved-sound bounds, compositional contracts, and (in v2) machine-checkable certificates—that LLMs cannot produce by construction. The intended workflow is complementary: LLMs for fast first-pass triage, formal analysis for the hard tail and for CI-integrated regression detection where correctness, not speed, is the constraint.

**Why existing tools fail them:** CacheAudit is x86-32, non-compositional, speculation-unaware, and produces no reusable contract. Spectector does not scale beyond ~100 instructions and produces boolean verdicts. Binsec/Rel checks constant-time only—useless for functions that intentionally leak bounded information (T-table AES, scatter/gather countermeasures). All tools are monolithic black boxes with no compositional reasoning or independent verification path.

---

## Technical Difficulty

### Hard Subproblems

1. **Binary lifting adapter for crypto code.** We reuse an existing lifter (BAP BIL, angr VEX, or Ghidra P-code) and build a 3–5K LoC adapter layer that maps the lifter's IR to our analysis IR with precise abstract transfer functions for ~150 crypto-critical instructions: general-purpose integer arithmetic, memory load/store, conditional moves (cmov), core AVX2 shuffles (vpshufb for T-table lookups), AES-NI intrinsics (aesenc, aesdec), carry-less multiplication (pclmulqdq for GCM), and 64-bit multiply-with-carry for big-integer arithmetic. AVX-512 lane-wise semantics are deferred to v2. The adapter must be differentially tested against the host lifter's concrete execution to catch semantic mismatches. *(Original scope: ~300 instructions with a from-scratch lifter. Reduced per Amendment 1 to eliminate the single highest-risk component.)*

2. **Speculative domain construction with formal soundness.** The concrete collecting semantics must be extended to include transient execution traces that are eventually squashed but leave residual cache state ("ghost footprints"). Formalizing a Galois connection (or γ-only sound abstraction) from this extended semantics to D_spec is genuine new abstract-interpretation theory with no direct prior art. We target Spectre-PHT misspeculation with bounded window (W ≤ 50 μops) for v1, deferring BTB, STL, and RSB to future work.

3. **Quantitative widening for counting domains.** Standard widening operators (intervals, octagons, polyhedra) guarantee convergence and soundness but provide no precision bound. Designing ∇_quant that converges finitely, never under-counts, and over-approximates by at most 2^k for k tainted cache sets is an open problem in abstract interpretation theory. The precision canary (see Phase Gates) validates this before the full framework is built.

4. **Reduced product precision engineering.** The direct product D_spec × D_cache × D_quant loses precision catastrophically without the reduction operator ρ. The reduction must propagate speculative infeasibility into cache taint (kill taint on unreachable paths) and cache untaintedness into capacity (zero capacity for untainted lines). ρ must be proved monotone and its fixpoint iteration must terminate. The interaction between speculative window pruning and cache taint removal is subtle and domain-specific. This reduction operator is the primary theoretical novelty of the work.

5. **Min-entropy composition soundness.** Min-entropy leakage does not satisfy the chain rule of Shannon entropy. The additive composition rule B_{f;g} = B_f + B_g(τ_f) requires a carefully crafted independence condition (no feedback from g's leakage into f's secrets). When feedback exists, a correction term d must be added. The composition theorem is proved first (Phase Gate 1) and tested on three crypto patterns before implementation begins. If the independence condition is too restrictive, we pivot to Rényi entropy (Boreale 2015, Fernandes et al. 2019), which provides meaningful bounds with composable semantics. *(Note: The algebraic identity instantiates Smith (2009) Theorem 4.3, but making it work soundly over the cache-state domain with taint-restricted counting, under speculative semantics, with convergence guarantees, is substantially more than an instantiation.)*

6. **LRU counting precision and PLRU gap characterization.** For v1, we target LRU replacement policy, where CacheAudit demonstrated tight counting bounds. We explicitly quantify the LRU→PLRU over-approximation gap: sound over-approximation of tree-PLRU via LRU abstraction introduces an estimated 10–50× over-estimation (Reineke et al. 2008), corresponding to ~13–22 spurious bits per 4 tainted cache sets. This gap is reported as a known limitation, not hidden. PLRU-specific counting heuristics are deferred to v2. *(Original scope included full PLRU/FIFO support. Reduced per Amendment 2 because exact PLRU counting is #P-hard.)*

7. **Scalability under product-domain state explosion.** An abstract cache domain for a 32KB L1D with 64 sets × 8 ways × 1 taint bit yields ~1K bits (~128 bytes) per abstract state. *(Note: the original statement erroneously computed 4096 sets; a 32KB / 8-way / 64B-line L1D has 64 sets, making per-point state 16× smaller than originally claimed.)* The speculative window multiplies this by the number of misspeculation points, but bounded speculation (W ≤ 50) keeps this tractable. Widening may still lose precision; crypto-specific widening strategies (full unrolling of fixed-iteration loops, precise summarization of table-lookup patterns) are essential, not optional.

### Estimated Subsystem Breakdown (Reduced-A Scope)

| Subsystem | Est. LoC | Complexity Source |
|-----------|----------|-------------------|
| **Lifter adapter** (BAP/angr/Ghidra → analysis IR) | ~4K | Adapter mapping existing lifter IR to analysis IR; abstract transfer functions for ~150 crypto-critical instructions (integer, memory, cmov, core AVX2, AES-NI, pclmulqdq). No decode tables, no instruction encoding—the host lifter handles that. Differentially tested against concrete execution. *(Reduced from 35K: we reuse an existing lifter rather than building from scratch. See Scope Decisions.)* |
| **Speculative reachability domain D_spec** | ~11K | Per-program-point tracking of speculation depth and arch/trans status; transfer functions for branches, fences (lfence, cpuid), and speculation window overflow; Spectre-PHT misspeculation modeling; speculative prefix closure for path merging. BTB/STL/RSB deferred to v2. |
| **Cache abstract domain D_cache** | ~11K | Set-associative cache model parameterized for LRU (PLRU/FIFO deferred); must/may analysis with taint annotations; transfer functions for LRU replacement with taint propagation through evictions; single-level L1D analysis. *(Reduced from 13K: LRU-only for v1, no multi-level hierarchy composition. See Scope Decisions.)* |
| **Quantitative channel capacity domain D_quant** | ~9K | Abstract counting over taint-restricted cache configurations; per-instruction leakage increment computation; per-cache-set leakage vectors for precision; widening operator with bounded precision loss; channel capacity derivation. |
| **Reduced product + fixpoint engine** | ~12K | Three-way product operations; reduction operator ρ with iterative refinement to fixpoint; widening strategy management (unroll-then-widen, delayed widening, crypto-pattern-aware thresholds); convergence detection; budget-bounded analysis with graceful degradation. |
| **Compositional contract system** | ~9K | Contract data structures; abstract cache-state transformer computation; composition engine (sequential and call-graph traversal); contract serialization/deserialization. *(Reduced from 12K: no contract parameterization by μarch model—hardcoded 2–3 configs instead. See Scope Decisions.)* |
| **CLI / configuration / CI integration** | ~4K | Command-line interface; per-function and whole-library analysis modes; CI integration (GitHub Action wrapper, exit codes, structured JSON output); regression-detection mode comparing two binary versions; configuration for analysis budgets, widening strategies. |
| **Test infrastructure** | ~13K | Per-domain unit tests; integration tests on crypto functions from BoringSSL, libsodium, and OpenSSL; precision benchmarks against exhaustive enumeration; regression-detection tests on pre/post-patch CVE binaries; property-based testing of domain operations. *(Reduced from 18K: no differential testing against hardware traces for lifter validation—the host lifter is trusted. See Scope Decisions.)* |
| **TOTAL** | **~73K** | |

**Genuinely novel LoC: ~52–58K.** The adapter layer (~4K) and a portion of the test infrastructure rely on established frameworks. The three abstract domains (~31K), reduced product + fixpoint engine (~12K), and compositional contract system (~9K) constitute the novel implementation. This is at the level of a strong PhD thesis—harder than a typical top-venue paper but achievable by a focused team within 1.5–2 years. For reference: CacheAudit (a simpler system without speculation, composition, or binary lifting) was ~15K LoC and took Boris Köpf's group ~3 years; Spectector (speculative non-interference only, no quantitative bounds) was a multi-year collaboration across three institutions. This tool subsumes and extends both.

---

## New Mathematics Required

### Summary

The mathematical specification for the Reduced-A scope comprises **~38 components**: ~20 definitions, ~7 lemmas, and ~11 theorems, organized across five categories (abstract domain theory, speculative execution semantics, cache hierarchy abstraction, quantitative information flow, and compositional reasoning). Certificate theory and soundness meta-theory for the certificate checker are deferred to v2 with the certificate system. Of the retained components, **~10 are fully established** results from prior work (Cousot & Cousot, CacheAudit, Spectector, Smith) that apply directly. **~12 are established with novel instantiation**—routine adaptations of known frameworks to the specific domain configuration. **~16 are novel results requiring new proofs**, including new domain constructions, new soundness arguments, and new composition theorems.

### Priority Sequencing

The mathematical work is sequenced by the Phase Gates (see below):

**First priority (Phase Gate 1, Month 1–2): Composition theorem.** Formally prove the min-entropy additive composition rule (Theorem E.2.1 + Lemma E.2.2) and characterize the independence condition. Demonstrate the condition holds for ≥3 representative crypto patterns. If it fails for all three, pivot to Rényi entropy before building any implementation. This is the highest-value de-risking action.

**Second priority (Phase Gate 2, Month 2–4): Quantitative domain soundness.** Prove the tainted counting bound (Theorem D.2), the capacity domain Galois connection (Theorem D.4), and the quantitative widening convergence guarantee. Validate against the precision canary (D_cache ⊗ D_quant on AES T-table).

**Third priority (Phase Gate 3, Month 4–8): Speculative domain construction.** Prove the speculative channel capacity bound (Theorem D.3.1), fence capacity reduction (Theorem D.3.2), and the three-way reduction operator soundness (Theorem A.3). These are the deepest novel results and the most uncertain.

**Deferred to v2:** Certificate theory (F.2.1, F.3.1), linked certificate soundness, end-to-end soundness meta-theorem (G.1.1), non-inclusive hierarchy composition (C.4.2), parallel composition with cross-interference (E.3.1).

### Novel Results Requiring New Proofs (~16)

Key novel results in scope include: the three-way product reduction operators r_{i←j,k} for speculative × cache × capacity (A.3); quantitative speculative non-interference qSNI as the formal contract target property (B.3); tainted cache-line tracking with taint soundness (C.2); taint-restricted state counting count_T (C.3); tainted variant of the counting bound (D.2); speculative channel capacity bound combining path counting with cache state counting (D.3.1); fence capacity reduction (D.3.2); capacity domain transfer functions and Galois connection (D.4); sequential composition of leakage contracts with min-entropy (E.2); additivity under independence (E.2.2); and feedback correction (E.2.3).

### The 3 Hardest Results

1. **Sequential composition with min-entropy (Theorem E.2.1 + Lemma E.2.2).** Min-entropy leakage is not sub-additive in general (Braun et al. 2009). The additive composition rule requires an independence condition that must be strong enough for soundness but weak enough to admit common code patterns. The proof uses a sub-multiplicative counting argument (|traces_{f;g}| ≤ |traces_f| × |traces_g|_{given τ_f(s)}) that holds under sequential single-threaded execution with no feedback, but the formal conditions delimiting "no feedback" through cache state are subtle. This sits at the frontier of quantitative information flow theory. *This is proved first (Phase Gate 1) because it is the intellectual crown jewel and the go/no-go gating result.* Fallback: Rényi entropy composition (Boreale 2015), which trades min-entropy's operational interpretation for composability.

2. **Speculative channel capacity bound (Theorem D.3.1).** Requires combining two fundamentally different analyses—speculative path enumeration (control-flow) and cache state counting (data-flow)—into a single tight bound. The interaction is non-trivial: speculative paths can share cache effects (two transient paths hitting the same set), making the naïve product k·m a gross overestimate. Tighter composition requires novel cache-aware speculative path merging that has no precedent in the literature. *If this bound proves vacuous (>10× true leakage), the Reduced-B fallback drops speculation from the first paper.*

3. **Three-way reduction operator soundness (Theorem A.3).** The reduction operators r_{i←j,k} must be proved monotone, their iterative refinement must terminate, and the fixpoint must be shown to dominate the best abstract approximation obtainable from the direct product alone. The interaction between speculative window pruning and cache taint removal is subtle and domain-specific. *This is the primary theoretical novelty—the mathematical engine that makes the four-property combination tractable.*

### Key Mathematical Barriers

- **Min-entropy non-composability.** Min-entropy does not satisfy the chain rule. If the required independence condition is too restrictive, the compositional framework degrades to whole-program analysis. Fallback strategies include Rényi divergence, g-leakage with composable gain functions, or accepting a constant-factor slack. The Phase Gate 1 composition proof explicitly tests this.

- **PLRU counting is #P-hard.** Exact counting of reachable PLRU states is intractable (Reineke et al. 2007). We defer PLRU to v2 and target LRU, where CacheAudit demonstrated tight bounds. The LRU→PLRU over-approximation gap is quantified as a known limitation.

- **Speculative path explosion.** The number of speculative continuations grows exponentially with the speculation window W. We bound W ≤ 50 μops (not the full 224-μop ROB depth of Skylake) as an engineering parameter that trades completeness for tractability, catching the vast majority of practical Spectre-PHT gadgets while keeping context count manageable. The "10–20 speculative contexts after prefix closure" claim from the original design is unsubstantiated; the actual count may be higher, and we treat this as an empirical question to be answered in Phase Gate 3.

- **Shared-cache concurrent composition.** Parallel composition (Theorem E.3.1) requires reasoning about all possible interleavings of cache accesses—exponential in the worst case. This is deferred to v2; v1 targets sequential single-threaded composition only.

All mathematics directly enables the artifact: definitions define the domain elements the implementation operates on; lemmas justify transfer functions and domain operations; theorems justify the contracts and composition rule that the tool produces.

---

## Best Paper Argument

The primary theoretical contribution is the **reduced product domain D_spec ⊗ D_cache ⊗ D_quant with its speculative reduction operator ρ**—the first abstract domain simultaneously modeling speculative reachability and cache channel capacity. The reduction operator is genuinely new abstract-interpretation theory with no direct prior art: it propagates speculative infeasibility into cache taint (pruning unreachable transient paths) and cache untaintedness into capacity bounds (zeroing capacity for untainted lines), requiring novel monotonicity and termination arguments over a non-standard counting domain. This is not a synthesis of existing components; it is a new domain construction that happens to enable a powerful combination of properties.

The work fills an **explicitly identified open problem**. Guarnieri and Reineke—LeaVe co-authors (Wang, Mohr, von Gleissenthall, Reineke, and Guarnieri; CCS 2023 Distinguished Paper)—along with Köpf, a key contributor to the broader hardware-software contracts program, have called out "software-side contract verification" as the next step in their leakage-contract research program, including in a 2025 SETSS tutorial. This paper answers that open question with a solution that is strictly harder than the hardware side: LeaVe verifies small, open-source RISC-V cores with known RTL; we analyze compiled binaries under speculative execution with quantitative reasoning and compositional contracts.

The framework contributes **3 independently publishable technical novelties**: (1) the reduced product D_spec ⊗ D_cache ⊗ D_quant with speculative reduction—a new abstract domain for speculative cache-channel analysis, requiring novel Galois-connection-like constructions over speculative collecting semantics; (2) cache-state-aware compositional leakage contracts with a non-trivially-sound additive composition rule for min-entropy, instantiating Smith (2009) for the cache domain with taint-restricted counting under speculative semantics (substantially more than an algebraic identity); and (3) a quantitative widening operator for counting domains with bounded precision loss—an open problem in abstract interpretation theory that would independently merit publication at SAS/VMCAI. Two additional novelties (machine-checkable certificates and a full μarch contract language) are deferred to v2 to keep the first paper focused on the core domain construction and composition theory.

The contribution is **agenda-setting**: it defines leakage contracts as a supply-chain primitive, analogous to type signatures for type safety. Best papers at S&P and CCS are increasingly agenda-setting (Guarnieri et al.'s hardware-software contracts, Cauligi et al.'s constant-time foundations). This paper concretizes that agenda with a working tool, formal proofs, and an evaluation on production code. The reduced scope makes this *more* credible, not less: the paper delivers on what it promises.

The work **bridges communities**: it appeals to PL/formal-methods reviewers (novel abstract domains, Galois connections, composition theorems) and systems-security reviewers (real CVE detection, CI integration, quantitative bounds on BoringSSL/libsodium). We target CCS as the primary venue, where theoretical depth is rewarded; S&P is the secondary target. This dual appeal is the profile that best-paper committees reward.

**Honest assessment of headwinds.** The paper faces structural challenges: the "synthesis" critique (combining known techniques, even with a novel domain) may attract skeptical reviewers; cache side-channel research had peak visibility in 2018–2020, though LeaVe's 2023 CCS Distinguished Paper demonstrates that formal treatments of the contract framework remain current. The evaluation must demonstrate non-vacuous precision (the precision canary addresses this) and clear improvement over CacheAudit and Spectector individually.

---

## Evaluation Plan

### Precision Canary (Phase Gate 2)

Before the full framework is built, we implement D_cache ⊗ D_quant (without speculation) on a single function: AES T-table lookup under LRU. We verify that bounds are within **3× of exhaustive enumeration** on small inputs (key ≤ 16 bits). If bounds exceed **5× of true leakage**, the quantitative widening is redesigned before anything else proceeds. If bounds exceed **10×**, the quantitative framework is considered fundamentally imprecise and the project is re-evaluated. This is the most cost-effective de-risking step: it validates the core quantitative machinery in ~2–4K LoC before committing to the full ~73K LoC framework.

### Benchmarks

- **Real crypto primitives from production libraries.** Functions extracted from BoringSSL, libsodium, and OpenSSL compiled with GCC-13 and Clang-17 at -O0, -O2, -O3: AES-128 (T-table and AES-NI), ChaCha20-Poly1305, Curve25519 (donna). RSA-2048 CRT is included as a scalability stress test. *(SHA-256 and ECDSA P-256 sign/verify are included if time permits but are not required for the core evaluation.)*

- **Known CVEs with regression detection.** Pre-patch and post-patch binaries for CVE-2018-0734 (ECDSA nonce leak), CVE-2018-0735 (ECDSA timing), CVE-2022-4304 (RSA timing oracle). The tool must produce non-zero bounds on vulnerable versions and 0 (or near-0) on patched versions. **Critically**, this benchmark also evaluates regression detection: the tool must detect the leakage *change* between versions even if absolute bounds are conservative. This is the primary validation that the tool is useful under modeling imprecision.

- **Spectre gadgets.** The Spectector benchmark suite and Paul Kocher's Spectre proof-of-concept examples. The tool must detect leakage under speculative contracts (LRU bounded-Spectre-PHT) and 0 leakage under non-speculative contracts. If speculative bounds are vacuous (>10× true leakage) on these benchmarks, the paper falls back to Reduced-B scope.

- **Compiler-broken constant-time.** libsodium functions that are CT at -O0, compiled with -O2/-O3 to demonstrate compiler-introduced leakage detection (e.g., GCC converting branchless CMOV to conditional branches, LLVM spilling secrets to stack).

- **Composition precision benchmark.** Three crypto patterns analyzed both monolithically and compositionally: AES rounds, ChaCha20 quarter-rounds, Curve25519 scalar-multiply steps. This directly validates the composition theorem's practical precision.

### Metrics (all fully automated, zero human involvement)

| Metric | Target | Measurement |
|--------|--------|-------------|
| Soundness | 0 false negatives | Exhaustive enumeration on small inputs (key ≤ 16 bits) as ground truth |
| Precision | Bounds within 3× of true leakage | Same exhaustive comparison; report tightness ratio per function |
| Precision canary | D_cache ⊗ D_quant within 3× on AES T-table | Implemented early (Phase Gate 2); 5× triggers redesign, 10× triggers kill |
| Regression sensitivity | Detect all 3 known CVE leakage changes | Bound on vulnerable version > bound on patched version, automated |
| Scalability | ≥90% of functions complete in <5 min | Wall-clock per function on laptop CPU |
| Composition precision | Composed bound within 2× of monolithic | Same program analyzed monolithically vs. compositionally |
| Contract sensitivity | Different contracts → different bounds | Automated comparison across 2–3 hardcoded contracts |
| CVE detection | Detect all 3 known CVEs | Bound on vulnerable version > bound on patched version, automated |
| CI integration | Full library in <30 min | Wall-clock for BoringSSL crypto/ directory *(revised from 15 min; realistic estimate is 30–90 min)* |

### Baselines

- **CacheAudit** — quantitative AI baseline (x86-32, non-speculative, monolithic): for direct precision comparison on shared benchmarks (AES T-table). The centerpiece evaluation demonstrates matching or exceeding CacheAudit's best-known bounds while additionally providing speculation awareness and composition.
- **Binsec/Rel** — binary-level CT checking (boolean baseline): to show quantitative bounds subsume boolean verdicts while handling non-CT code.
- **Spectector** — speculative analysis (qualitative baseline): to show quantitative bounds on Spectre gadgets provide strictly more information.
- **Naïve composition** — analyze per-function, sum bounds without cache-state awareness: to quantify the precision improvement from cache-state-aware composition.

---

## Laptop CPU Feasibility

The framework is designed to run comfortably on a single laptop CPU (e.g., Apple M2 or Intel i7 mobile) without GPU, cluster, or cloud resources. Five architectural properties ensure this:

1. **Abstract interpretation, not symbolic execution or SMT.** The entire analysis operates over abstract states—finite lattice elements—with no path enumeration, no constraint solving, and no SAT/SMT queries. Fixpoint computation is polynomial in the abstract domain size, which is fixed by cache geometry, not program size.

2. **Domain sizes bounded by cache geometry.** The abstract cache domain for a 32KB L1D with **64 sets** × 8 ways × 1 taint bit is ~1K bits ≈ **128 bytes** per abstract state. *(Corrected from the original "4096 sets" — a 32KB / 8-way / 64B-line L1D has 64 sets, not 4096. The correction makes per-point state 16× smaller than originally computed, substantially improving the feasibility argument.)* The speculative dimension adds a factor of the effective speculation depth under bounded speculation (W ≤ 50 μops); the actual context count after prefix closure is an empirical question, but even at 100 speculative contexts the per-program-point state is ~12.5KB—comfortably within L2 cache. Total per-program-point state across all three domains: ~30–100KB.

3. **Per-function analysis with compact summaries.** Each function is analyzed independently. The summary (τ_f, B_f) is a few KB. There is no whole-program fixpoint—only a linear traversal of the call graph bottom-up, applying the composition rule at each call site.

4. **Fixed-iteration crypto loops.** AES has 10/12/14 rounds. ChaCha20 has 20 rounds. Curve25519 scalar multiply has 255 iterations. These fixed-iteration loops can be precisely handled (full unrolling or exact widening). There are no data-dependent loop bounds to cause analysis divergence.

5. **Existing lifter handles the hard work.** By reusing BAP/angr/Ghidra for binary lifting and CFG recovery, we inherit their optimized implementations and avoid the memory/time overhead of a custom decoder. The adapter layer adds negligible overhead.

**Expected performance:** ~30 seconds per typical crypto function (AES round function, ChaCha quarter-round), ~5 minutes for complex functions (RSA modular exponentiation, Curve25519 scalar multiply). Full BoringSSL crypto/ directory: ~30–90 minutes on a modern laptop (most functions are trivial utilities analyzed in <1 second; ~30–50 crypto-critical functions at 30 seconds–5 minutes each). This is CI-compatible: build pipelines routinely take 30–120 minutes.

---

## Scope Decisions and Deferrals

The following scope reductions are applied based on the depth-check panel's binding amendments. Each is motivated by a specific risk identified during adversarial review. The core vision is unchanged; the execution plan is realistic.

### Deferred to v2 (Follow-Up Paper)

| Component | Original Scope | Deferred Because |
|-----------|---------------|-----------------|
| **Machine-checkable certificates** (certificate emission ~7K LoC, certificate checker ~3K LoC) | Per-function certificates with derivation trees, standalone OCaml checker | Certificates are a valuable but separable contribution. The first paper's novelty is the domain construction and composition theory; certificates add ~10K LoC and a separate soundness argument without strengthening the core contribution. Deferral keeps the paper focused and the implementation tractable. |
| **μarch contract language** (~6K LoC) | Full specification language with parser, AST, type checker, contract comparison | A formal contract language is premature before the core analysis is validated. For v1, we hardcode 2–3 configurations (ARM Cortex-A76 LRU no-spec, ARM Cortex-A76 LRU bounded-PHT, generic 8-way LRU). This covers the evaluation needs and avoids building language tooling that may need redesign after real-world contract iteration. |
| **From-scratch binary lifter** (~35K LoC) | Custom x86-64 → IR lifter with ~300 instruction semantics | Building a binary lifter is a multi-year effort orthogonal to the research contribution. Reusing BAP/angr/Ghidra eliminates the single highest-risk component and reduces total LoC by ~40%. The adapter layer (~4K LoC) restricts to ~150 crypto-critical instructions with AVX-512 deferred. |
| **PLRU/FIFO replacement policies** | Full parameterized replacement-policy support | Exact PLRU counting is #P-hard (Reineke et al. 2007). Sound over-approximation introduces 10–50× imprecision. CacheAudit published successfully with LRU-only analysis. We target ARM Cortex-A (LRU) and explicitly characterize the PLRU gap. |
| **Multi-level cache hierarchy** | Non-inclusive L1/L2/L3 composition | Adds substantial complexity (non-inclusive eviction semantics, cross-level taint propagation) with limited benefit for v1's single-function analysis focus. L1D analysis captures the dominant leakage vector for crypto code. |
| **Parallel composition** (Theorem E.3.1) | Cross-thread cache interference bounds | Requires reasoning about all possible interleavings—exponential worst case. v1 targets sequential single-threaded composition. |
| **End-to-end soundness meta-theorem** (G.1.1) | Full chain from fixpoint to certificate | Depends on the certificate system. Deferred with certificates. v1 proves soundness of the domain construction, composition rule, and per-function analysis separately. |

### What Remains in v1 (~55–65K LoC of novel work within ~73K total)

- **D_spec ⊗ D_cache ⊗ D_quant** — the complete three-way reduced product domain with speculative reduction operator ρ
- **Compositional leakage contracts** — per-function analysis producing (τ_f, B_f) with proved-sound additive composition
- **Speculative modeling** — bounded Spectre-PHT under parameterized window W ≤ 50
- **Quantitative bounds** — bits-of-leakage bounds with formal soundness guarantees
- **Regression detection** — first-class CI-integrated mode comparing binary versions
- **Evaluation** — precision canary, CVE detection, composition benchmarks, CacheAudit comparison

This scope preserves ~80% of the research value at ~48% of the original LoC estimate.

---

## Phase Gates

The project proceeds through four phase gates. Each gate has explicit success criteria and kill triggers. The gates enforce disciplined risk management: expensive implementation work does not begin until the theoretical foundations are validated.

### Phase Gate 1: Composition Theorem (Month 1–2)

**Objective:** Formally prove the min-entropy additive composition rule and characterize the independence condition.

**Success criteria:** The independence condition holds for ≥3 of the following crypto patterns: (a) AES rounds (sequential, non-overlapping byte positions), (b) ChaCha20 quarter-rounds (data-independent permutation structure), (c) Curve25519 scalar multiply (conditional swap + field operations).

**Pivot trigger:** If the condition fails for all three patterns, pivot to Rényi entropy composition (Boreale 2015, Fernandes et al. 2019) before proceeding.

**Kill trigger:** If the composition theorem provably requires conditions that exclude all standard crypto patterns *and* the Rényi fallback yields vacuous bounds, the project is abandoned.

### Phase Gate 2: Precision Canary (Month 2–4)

**Objective:** Implement D_cache ⊗ D_quant (without speculation) on AES T-table lookup under LRU and validate precision.

**Success criteria:** Bounds within 3× of exhaustive enumeration on small inputs (key ≤ 16 bits).

**Redesign trigger:** If bounds exceed 5× of true leakage, the quantitative widening operator must be redesigned before proceeding.

**Kill trigger:** If bounds exceed 10× of true leakage after redesign, the quantitative framework is fundamentally imprecise and the project is re-evaluated.

### Phase Gate 3: Speculative Prototype (Month 4–8)

**Objective:** Prototype the full D_spec ⊗ D_cache ⊗ D_quant on 3–5 crypto functions with bounded Spectre-PHT.

**Success criteria:** Speculative bounds within 10× of true leakage; non-speculative bounds remain within 3×.

**Fallback trigger:** If speculative bounds are vacuous (>10× true leakage), invoke the Reduced-B fallback: drop speculation from the first paper, publish D_cache ⊗ D_quant + composition + quantitative bounds as the contribution, and treat speculation as a separate follow-up.

### Phase Gate 4: Full Evaluation and Submission (Month 8–14)

**Objective:** Complete evaluation on AES, ChaCha20, Curve25519, and ≥1 known CVE. Prepare paper submission.

**Success criteria:** All evaluation metrics met (see Evaluation Plan). Paper submitted to CCS (preferred) or S&P.

**Target venue:** CCS, where theoretical depth is rewarded and PL/security cross-cutting work is well-received.

---

## Slug

`certified-leakage-contracts`
