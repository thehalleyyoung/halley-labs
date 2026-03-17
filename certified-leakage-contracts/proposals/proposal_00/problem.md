# Final Approach: Compositional Speculative Cache-Channel Analysis via Reduced Product Abstract Interpretation

## Approach Name and Summary

**Full Reduced Product Abstract Interpretation with Contract-Typed Composition and Differential Regression Detection.** We construct the first abstract domain simultaneously modeling speculative execution reachability and quantitative cache channel capacity—a three-way reduced product D_spec ⊗ D_cache ⊗ D_quant connected by a novel reduction operator ρ—and use it to produce per-function leakage contracts on x86-64 crypto binaries. These contracts compose additively via a cache-state-aware rule, enabling whole-library bounds from per-function analysis. The primary near-term use case is CI-integrated regression detection: flagging when a compiler upgrade or code change increases any function's leakage bound, a capability robust to the modeling imprecision that limits absolute bounds.

## Selection Rationale

**Approach A won the adversarial debate decisively.** It has the lowest kill probability (30%), highest publishable-paper probability (55%), strongest feasibility profile (polynomial scalability, CacheAudit existence proof, 16–22 month timeline), and most credible fallback paths (direct product if ρ fails, Reduced-B if speculation proves vacuous). The reduction operator ρ is the deepest genuine novelty across all three approaches after systematic deflation (60% genuinely new per MDA), and the work directly answers the software-side open question from LeaVe (CCS 2023 Distinguished Paper).

**Elements borrowed from Approach C (Type-Directed QIF):** C's central insight—that function types ARE contracts, making composition definitional rather than engineered—informs our contract interface design. We adopt a *contract-typed* representation where each function's leakage contract (τ_f, B_f) is treated as a type signature: callers consume contracts via a typed composition rule that mirrors type application. This does not import C's full type system (which faces fatal binary-level inference challenges per DA/Skeptic consensus), but borrows the clean compositional interface. Specifically, contracts are structured as:

```
f : CacheState →_τ CacheState  ×  CacheState →_B ℝ≥0
```

where composition follows the type-application pattern `(g ∘ f)(s) = (τ_g ∘ τ_f, B_f(s) + B_g(τ_f(s)))`. This simplifies the contract serialization format and enables IDE-friendly display of per-function leakage signatures.

**Elements borrowed from Approach B (Modular Relational Verification):** B's strength is differential analysis—comparing two binaries to quantify the marginal leakage change. While B's relational symbolic execution is fatally compromised by path explosion (65% kill probability, DA-rated FATAL), the *framing* of regression detection as a relational comparison is stronger than A's original treatment. Our regression detection mode incorporates a lightweight relational comparison: given two versions of a binary (v_old, v_new), we run the abstract interpretation independently on each and compute the *contract difference* Δ_f = B_f^{new}(s) − B_f^{old}(s) for each function f at each abstract input state s. This is cheaper than B's full relational SE (it reuses A's polynomial analysis twice) while capturing the relational insight that regression detection is fundamentally a *differential* property. Thresholded alerts (Δ_f > ε) drive the CI integration.

**How critiques were addressed:** Every major critique from the Skeptic and Math Depth Assessor is addressed below (see §Hard Subproblems and §Debate-Driven Improvements). The ρ-at-merge-points interaction is formally specified. The independence condition is stress-tested with a 4th benchmark. Regression detection is foregrounded as THE primary use case. LLM competition is explicitly positioned against. The Curve25519 memory estimate is addressed with state-sharing engineering. Speculative context counts are treated as empirical, not claimed.

## Technical Architecture

### Core Construction: The Reduced Product Domain

The analysis operates on a reduced product abstract domain D_spec ⊗ D_cache ⊗ D_quant:

1. **D_spec (Speculative Reachability Domain, ~8–12K LoC).** A tagged powerset domain tracking which program points are reachable under transient execution within a parameterized speculation window W ≤ 50 μops. Each abstract state element carries a tag (ARCHITECTURAL | TRANSIENT(depth)) indicating whether the instruction sequence is committed or speculative. Transfer functions model Spectre-PHT branch misspeculation (both directions taken up to depth W), fence instructions (lfence/cpuid collapse all TRANSIENT states), and speculation window overflow (states exceeding W are pruned). BTB, STL, and RSB misspeculation models are deferred to v2.

2. **D_cache (Tainted Abstract Cache-State Domain, ~9–13K LoC).** A set-associative cache model parameterized for LRU replacement, extending CacheAudit's must/may analysis with per-line secret-dependence taint annotations. For a 32KB L1D with 64 sets × 8 ways × 1 taint bit, each abstract state is ~128 bytes. Transfer functions model LRU replacement with taint propagation through evictions. The taint annotation layer tracks whether each cached block's presence depends on secret data—only tainted cache configurations contribute to the quantitative count.

3. **D_quant (Quantitative Channel-Capacity Domain, ~8–12K LoC).** Abstract counting over taint-restricted cache configurations, producing per-cache-set leakage vectors. The domain accumulates bits of leakage via counting the number of distinguishable cache states reachable from different secret inputs, restricted to tainted cache lines. This extends CacheAudit's counting abstraction (Köpf & Rybalchenko, CSF 2010) with taint-restricted counting—a straightforward but implementation-intensive adaptation.

4. **Reduction Operator ρ (~3–5K LoC of correctness-critical code within the ~10–14K reduced product + fixpoint engine).** The crown jewel. ρ propagates constraints downward through the product:
   - **ρ_{cache←spec}:** Speculative infeasibility prunes cache taint. If D_spec determines a speculative path is unreachable (window overflow, fence encountered), ρ removes the cache-line taint that path would have introduced.
   - **ρ_{quant←cache}:** Cache untaintedness zeros capacity. If D_cache shows a cache set has no tainted lines (after ρ_{cache←spec} pruning), ρ sets that set's contribution to D_quant to zero.
   - **ρ-at-merge-points (addressing Math critique §2):** At CFG merge points, ρ is applied *after* join, not before. When speculative contexts with different depths merge, the join conservatively re-includes cache accesses that ρ pruned in one branch but not the other. The inner ρ-fixpoint is then iterated at the merged state until convergence. This is potentially expensive when counting-domain intervals [0, 2^n] participate, but the bounded cache geometry (64 sets × 8 ways) ensures the inner fixpoint terminates in O(|sets| × |ways|) iterations. A restricted single-pass ρ (apply ρ_{cache←spec} then ρ_{quant←cache} once, without iterating to fixpoint) is the monotone-by-construction fallback if the iterative version fails termination or precision tests.

### Compositional Contract System (~7–10K LoC)

Per-function analysis produces leakage contracts of the form:

```
Contract_f = (τ_f : D_cache → D_cache,  B_f : D_cache → ℝ≥0)
```

where τ_f is the abstract cache-state transformer (how f changes the cache) and B_f maps an initial abstract cache state to a leakage bound in bits. Sequential composition follows:

```
B_{f;g}(s) = B_f(s) + B_g(τ_f(s))
```

This rule is sound under an independence condition: g's cache observations are independent of f's leaked information *given* the abstract cache state τ_f(s). The independence condition is validated for specific crypto patterns before implementation (Phase Gate 1).

**Contract-typed interface (from Approach C):** Contracts are serialized in a type-signature-like format for human readability and IDE integration:

```
aes_round : CacheState → CacheState {leaks ≤ 1.7 bits under LRU-PHT(W=50)}
```

This notation—borrowed from C's paradigm—makes contracts accessible as "leakage type signatures" without requiring the full type-theoretic machinery.

### Regression Detection Mode (from Approach B's relational framing)

The regression detection mode accepts two binary versions (v_old, v_new) and:

1. Runs the full analysis independently on each version.
2. Computes per-function contract differences: Δ_f(s) = B_f^{new}(s) − B_f^{old}(s).
3. Flags functions where Δ_f > 0 (leakage increased) or where new functions appear without contracts.
4. Produces structured JSON output suitable for CI integration (GitHub Action, exit codes).

This is the "killer app" identified by the depth check's Auditor: even when absolute bounds are 5–10× conservative (due to LRU→PLRU gap, widening imprecision, or composition imprecision), *changes* between versions are precisely captured. Regression detection transforms modeling imprecision from an existential threat into a tolerable over-approximation.

### Binary Analysis Substrate (~4–6K LoC adapter)

We reuse an existing lifter (BAP BIL, angr VEX, or Ghidra P-code) and build a 3–5K LoC adapter layer mapping the lifter's IR to our analysis IR. The adapter provides abstract transfer functions for ~150 crypto-critical instructions: general-purpose integer arithmetic, memory load/store, conditional moves (cmov), core AVX2 shuffles (vpshufb), AES-NI (aesenc, aesdec), carry-less multiply (pclmulqdq), and 64-bit multiply-with-carry. AVX-512 is deferred. The adapter is differentially tested against the host lifter's concrete execution.

### What Is Novel vs. Adapted from Prior Work

| Component | Status | Source |
|-----------|--------|--------|
| Reduced product framework | ADAPTED | Cousot & Cousot (1979), Granger (1992) |
| Speculative trace semantics | ADAPTED (~30% new) | Spectector (S&P 2020); new: quantitative observation function |
| Cache must/may analysis with taint | ADAPTED (~40% new) | CacheAudit (TISSEC 2015); new: per-line taint annotations |
| Taint-restricted counting | ADAPTED (~40% new) | Köpf & Rybalchenko (CSF 2010); new: restriction to tainted configs |
| **Reduction operator ρ** | **NOVEL (~60% new)** | No direct prior art for spec×cache×quant reduction |
| **Composition rule over cache-state domain** | **NOVEL (~40% new)** | Instantiates Smith (2009) Thm 4.3; new: cache-state parameterization under speculative semantics |
| Contract-typed interface | ADAPTED | Inspired by Approach C; engineering contribution |
| Regression detection via contract difference | ADAPTED | Inspired by Approach B's relational framing; engineering contribution |

## Mathematical Contributions

| # | Result | Classification | Load-Bearing? | Notes |
|---|--------|---------------|--------------|-------|
| A1 | Speculative collecting semantics with quantitative cache residues | NOVEL (~30% new) | YES | Extends Spectector's trace model with quantitative observation function. The trace structure is adapted from Guarnieri et al. (S&P 2020); the genuinely new ~30% is treating squashed-but-cache-polluting traces as quantitative channel contributors. |
| A2 | γ-only soundness from speculative collecting semantics to D_spec | NOVEL | YES | No Galois connection required; γ-only abstraction suffices |
| A3 | Tainted abstract cache-state domain D_cache | INSTANTIATION | YES | Extends CacheAudit with per-line taint; core cache model |
| A4 | Quantitative channel-capacity domain D_quant with taint-restricted counting | NOVEL | YES | ~40% new beyond CacheAudit's counting; produces bit-level bounds |
| **A5** | **Reduction operator ρ for D_spec ⊗ D_cache ⊗ D_quant** | **DEEP NOVEL** | **YES** | **Crown jewel.** Three-way reduction with domain-specific monotonicity/termination arguments. No direct prior art. ~60% genuinely new after deflation. |
| A6 | Quantitative widening ∇_quant with bounded precision loss | DEEP NOVEL (deferred) | ENABLING | Not required for v1 fixed-iteration targets; deferred to v2. Cannot be claimed as a v1 contribution unless exercised in evaluation. |
| A7 | Additive composition rule B_{f;g}(s) = B_f(s) + B_g(τ_f(s)) | INSTANTIATION | YES | Instantiates Smith (2009) Thm 4.3; domain-specific engineering to instantiate under speculative cache semantics with taint-restricted counting is non-trivial but follows established composition theory |
| A8 | Independence condition characterization for crypto patterns | NOVEL | YES | Applied verification; ~25% genuinely new |
| A9 | Fixpoint convergence of reduced product iteration | INSTANTIATION | YES | Standard; bounded domain ensures termination |

**Crown jewel identification: A5 (Reduction operator ρ).** This is the single most technically novel component—a three-way reduction propagating information across speculation, cache taint, and quantitative capacity domains. No prior reduced product in the abstract interpretation literature combines these three concerns. The MDA confirms this survives deflation best at ~60% genuinely new, requiring domain-specific arguments for monotonicity and termination that have no direct template. Everything else in the framework—the speculative semantics, the cache domain, the counting abstraction, the composition rule—is well-scaffolded by prior work. ρ is what makes the combination tractable rather than vacuously imprecise.

## Hard Subproblems and Mitigations

### 1. Reduction Operator ρ Correctness and Precision (Kill risk: 35%)

**The problem:** ρ must simultaneously be monotone (for fixpoint convergence), sound (never undercount leakage), and precise (actually tighten bounds over the direct product). These properties are in tension. Paper proofs establish soundness and monotonicity but say nothing about *precision*—you can prove ρ correct in a weekend and spend a year discovering it adds zero precision on real code.

**The ρ-at-merge-points problem (Math critique §2):** At CFG merge points, the join operation may re-introduce states that ρ pruned in one branch. When speculative contexts with different depths merge, the merged D_spec state must conservatively re-include cache accesses ρ pruned in one context but not the other. This "un-pruning" could silently undo most of ρ's precision gains.

**Mitigation:** (a) Prove ρ correct on paper *before* implementing—specifically addressing the merge-point interaction by proving that the ρ-after-join strategy is sound and that the inner ρ-fixpoint converges. (b) Design a precision test for ρ at merge points: construct a program with a diamond CFG where one branch creates a speculative path and the other doesn't, verify that ρ prunes the infeasible speculative taint after the join. (c) Build the precision canary (D_cache ⊗ D_quant without speculation) first to validate the counting machinery independently. (d) If iterative ρ fails, fall back to single-pass ρ (monotone by construction, weaker but functional). (e) If ρ fails entirely, the direct product still yields sound bounds—the paper degrades from "novel reduced product" to "CacheAudit extended with speculation + composition," which remains publishable. Estimated effort: 4–7 person-months.

### 2. Independence Condition for Min-Entropy Composition (Failure risk: 30%)

**The problem:** Min-entropy does not satisfy the chain rule. The additive composition requires that g's cache observations are independent of f's leaked information given the abstract cache state τ_f(s). The Math Assessor's analysis shows: AES independence approximately holds for distinct subkeys but fails for related subkeys (cache-set aliasing creates correlations); ChaCha20 trivially satisfies it (zero leakage—useless for validation); Curve25519 holds only if τ is sufficiently precise.

**The 4th benchmark requirement (Math critique §5):** The original three benchmarks do not genuinely stress-test composition. We add a 4th benchmark: **T-table AES with related subkeys** (AES-256 where the two 128-bit subkey halves share structure), which creates genuine cache-state correlation between composed rounds. Additionally, we include a **scatter/gather AES implementation** that uses secret-dependent gather instructions, creating feedback between rounds.

**Mitigation:** (a) Prove the composition theorem first (Phase Gate 1). (b) Characterize the independence condition on 4 crypto patterns (AES distinct subkeys, AES related subkeys, ChaCha20, Curve25519). (c) For patterns where independence fails, compute explicit correction terms d such that B_{f;g}(s) ≤ B_f(s) + B_g(τ_f(s)) + d. (d) If correction terms are too large for >50% of patterns, pivot to Rényi entropy.

### 3. Rényi Fallback Viability (Addressing Math critique §5)

**Honest assessment:** The Rényi fallback is a real Plan B but a costly one—more "Plan B that itself requires research" than a simple engineering pivot. Specifically:

- **Additional research effort:** Computing Rényi divergence abstractly requires a new domain operator, estimated at 1–2 additional person-months of research-level work.
- **Precision loss:** Rényi bounds are strictly looser (2–5× per Fernandes et al. 2019). Combined with AI over-approximation, the Rényi path may produce bounds where regression detection cannot distinguish signal from noise.
- **Budget implication:** If Phase Gate 1 triggers the Rényi pivot, add 1–2 months to the timeline and accept that resulting bounds will be 2–5× looser than the min-entropy path.

The Rényi fallback prevents *total* failure of the composition story but should not be counted on for strong results. It is insurance against abandonment, not a path to a best paper.

### 4. Quantitative Widening Convergence (Risk: 20% for v1)

**The problem:** Standard widening operators provide no precision bound. For v1, we target fixed-iteration crypto (AES 10 rounds, ChaCha20 20 rounds, Curve25519 255 iterations) where loops are fully unrolled, avoiding widening entirely. Widening is needed only for variable-iteration code (deferred).

**Mitigation:** (a) v1 avoids widening via full unrolling—this is CacheAudit's strategy and is validated. (b) For Curve25519's 255 iterations, unrolling is feasible but memory-intensive (see §5). (c) The widening operator ∇_quant remains a genuine open problem in abstract interpretation theory that would independently merit publication at SAS/VMCAI—but it is not on the critical path for v1.

### 5. Curve25519 Memory Scalability (Addressing Skeptic §5)

**The problem:** The Skeptic estimates 255 iterations × 50 instructions × 200 speculative contexts × 1.2 KB ≈ 3 GB of state for Curve25519, which may not fit in memory.

**Mitigation:** (a) **State sharing across iterations:** Curve25519's Montgomery ladder has a fixed access pattern per iteration—the cache state transformer τ is nearly identical across iterations. Sub-states sharing identical cache configurations (common when the iteration touches the same memory layout) are hash-consed, reducing the 3 GB estimate by an estimated 5–10×. (b) **Bounded speculation contexts:** The "200 contexts" estimate is worst-case. With bounded W ≤ 50 and prefix closure, the actual context count is an empirical question (see §6). If it exceeds memory, we fall back to non-speculative analysis for Curve25519, which still validates the composition theorem. (c) **Incremental analysis:** Analyze a single ladder step, extract contract, compose 255 times. This is the composition theorem's raison d'être—if composition works, we never hold 255 unrolled iterations in memory simultaneously.

### 6. Speculative Context Count Uncertainty (Addressing Skeptic §2)

**The problem:** The original "10–20 speculative contexts after prefix closure" claim is unsubstantiated. The Skeptic and Auditor agree this number has zero evidence behind it.

**Mitigation:** We do *not* claim a specific context count. The speculative context count is treated as an **empirical question** to be answered in Phase Gate 3. The analysis is parameterized by W (speculation window depth), and Phase Gate 3 measures the actual context count on each benchmark function. If the count exceeds tractable limits (>500 contexts, causing analysis time >5 minutes per function), we reduce W or fall back to non-speculative analysis for that function. The CI integration reports per-function analysis time alongside leakage bounds.

### 7. Bounded Speculation Soundness (Addressing debate Amendment 4)

**The problem:** If W ≤ 50 misses speculative paths that access additional secret-dependent cache lines beyond depth 50, the leakage upper bound is unsound (under-counts leakage).

**Mitigation:** We prove a **residual leakage bound** for paths of length > W. The argument: for speculation windows deeper than W, the number of additional cache lines reachable is bounded by the function's total memory footprint divided by the cache-line size. For crypto functions with footprint F bytes and 64-byte cache lines, the residual leakage is at most log₂(⌈F/64⌉) bits per misspeculation point beyond W. This term is added to the reported bound as a conservative residual. For typical crypto functions (F < 64KB), the residual is ≤ 10 bits—sound though potentially conservative. This obligation is proved as part of the speculative collecting semantics (A1/A2).

## Evaluation Plan

### Precision Canary (Phase Gate 2)

Before the full framework is built, implement D_cache ⊗ D_quant (without speculation) on AES T-table lookup under LRU. Verify bounds within **3× of exhaustive enumeration** on small inputs (key ≤ 16 bits). Thresholds: 5× triggers redesign, 10× triggers kill. This is ~2–4K LoC and validates the core quantitative machinery before committing to the full framework.

### Benchmarks

1. **Real crypto primitives from production libraries.** Functions from BoringSSL, libsodium, and OpenSSL compiled with GCC-13 and Clang-17 at -O0, -O2, -O3:
   - AES-128 T-table implementation
   - AES-128 with AES-NI
   - ChaCha20-Poly1305
   - Curve25519 (donna)
   - RSA-2048 CRT (scalability stress test)

2. **CVE regression detection (primary benchmark).** Pre-patch and post-patch binaries for:
   - CVE-2018-0734 (OpenSSL ECDSA nonce leak from optimizer transforms)
   - CVE-2018-0735 (OpenSSL ECDSA timing)
   - CVE-2022-4304 (RSA timing oracle)

   The tool must detect the leakage *change* (Δ_f > 0 on vulnerable, Δ_f ≈ 0 on patched) even if absolute bounds are conservative.

3. **Spectre gadgets.** Spectector benchmark suite + Paul Kocher's Spectre PoCs. Must detect leakage under speculative contracts and 0 leakage under non-speculative contracts.

4. **Composition precision benchmark.** AES rounds, ChaCha20 quarter-rounds, Curve25519 scalar-multiply steps analyzed both monolithically and compositionally. Additionally: **T-table AES with related subkeys** (the 4th benchmark demanded by Math critique §5) to stress-test the independence condition on code with genuine cache-state correlation.

5. **Compiler-broken constant-time.** libsodium functions that are CT at -O0 but leak under -O2/-O3 (GCC cmov→branch conversion, LLVM secret spills to stack).

### Metrics (fully automated, zero human involvement)

| Metric | Target | Measurement |
|--------|--------|-------------|
| Soundness | 0 false negatives | Exhaustive enumeration on small inputs (key ≤ 16 bits) |
| Precision | ≤ 3× true leakage (non-speculative) | Tightness ratio per function vs. exhaustive |
| Speculative precision | ≤ 10× true leakage | Same; 10× is "useful for regression, not for absolute bounds" |
| Regression sensitivity | Detect all 3 CVE leakage changes | Δ_f > 0 on vulnerable, Δ_f ≈ 0 on patched |
| Composition precision | ≤ 2× monolithic bound | Same program analyzed compositionally vs. monolithically |
| Scalability | ≥ 90% of functions < 5 min | Wall-clock per function on laptop CPU |
| CI integration | Full library < 90 min | Wall-clock for BoringSSL crypto/ directory |

### Baselines

- **CacheAudit** — quantitative AI baseline (x86-32, non-speculative, monolithic). Centerpiece: match or exceed CacheAudit bounds on AES T-table while adding speculation awareness and composition.
- **Binsec/Rel** — binary-level CT checking (boolean baseline). Show quantitative bounds subsume boolean verdicts.
- **Spectector** — speculative analysis (qualitative baseline). Show quantitative speculation bounds provide strictly more information.
- **Naïve composition** — sum per-function bounds without cache-state awareness. Quantify improvement from cache-state-aware composition.
- **cachegrind diff** — informal regression baseline. The formal regression detection mode catches cases that cachegrind diffing structurally misses: (a) speculative leaks — cachegrind traces architectural execution only, missing transient cache pollution under Spectre-PHT; (b) formal soundness guarantees — cachegrind diffing can miss subtle leakage changes lost in trace noise, while our analysis provides proved sound-upper-bound differences with zero false negatives under the assumed model; (c) quantitative delta precision — cachegrind reports raw cache-miss counts, not bits of secret-dependent leakage, conflating secret-independent and secret-dependent cache behavior. Our tool reports Δ_f in bits of secret-dependent leakage, isolating the security-relevant signal.

## Phase Gates

### Phase Gate 1: Composition Theorem (Month 1–2)

**Objective:** Formally prove the min-entropy additive composition rule and characterize the independence condition.

**Deliverable:** Paper proof of B_{f;g}(s) = B_f(s) + B_g(τ_f(s)) with independence condition; independence verified on 4 crypto patterns (AES distinct subkeys, **AES related subkeys**, ChaCha20, Curve25519).

**Success criteria:** Independence holds for ≥ 3 of 4 patterns (ChaCha20's trivial satisfaction counts but is noted as non-stress-testing).

**Pivot trigger:** Independence fails for all non-trivial patterns → pivot to Rényi entropy (budget +1–2 months).

**Kill trigger:** Composition provably requires conditions excluding all standard crypto patterns AND Rényi fallback yields vacuous bounds.

### Phase Gate 2: Precision Canary (Month 2–4)

**Objective:** Implement D_cache ⊗ D_quant (without speculation) on AES T-table under LRU.

**Deliverable:** ~2–4K LoC prototype producing leakage bounds on AES T-table with small keys.

**Success criteria:** Bounds within 3× of exhaustive enumeration.

**Redesign trigger:** Bounds > 5× → redesign quantitative counting.

**Kill trigger:** Bounds > 10× after redesign → quantitative framework fundamentally imprecise.

### Phase Gate 3: Speculative Prototype (Month 4–8)

**Objective:** Prototype full D_spec ⊗ D_cache ⊗ D_quant with bounded Spectre-PHT on 3–5 crypto functions.

**Deliverables:** (a) Empirical speculative context counts per benchmark function. (b) ρ precision test at merge points. (c) Speculative leakage bounds on Spectre PoCs.

**Success criteria:** Speculative bounds ≤ 10× true leakage; non-speculative bounds remain ≤ 3×; ρ measurably tightens bounds over direct product on ≥ 1 benchmark.

**Fallback trigger:** Speculative bounds > 10× → invoke Reduced-B: drop speculation from first paper, publish D_cache ⊗ D_quant + composition as core contribution.

### Phase Gate 4: Full Evaluation and Submission (Month 8–14)

**Objective:** Complete evaluation on all benchmarks. Prepare paper.

**Success criteria:** All metrics met. Regression detection catches all 3 CVEs. Composition precision within 2× of monolithic on ≥ 2 benchmarks.

**Target venue:** CCS (primary), where theoretical depth and PL/security cross-cutting is rewarded. S&P (secondary).

## Estimated Subsystem Breakdown

LoC estimates use the DA's audited numbers, not the original claims.

| Subsystem | Estimated LoC | Complexity Source |
|-----------|:------------:|-------------------|
| Lifter adapter (BAP/angr/Ghidra → analysis IR) | 4–6K | SIMD/AES-NI transfer functions; ~150 crypto-critical instructions |
| D_spec (speculative reachability) | 8–12K | Tagged powerset; speculative forking/merging; fence handling |
| D_cache (tainted cache domain) | 9–13K | CacheAudit core (~5–8K) + taint annotation layer |
| D_quant (quantitative capacity) | 8–12K | Taint-restricted counting; per-set leakage vectors; widening skeleton |
| Reduced product + fixpoint engine | 10–14K | ρ (~3–5K correctness-critical); widening strategy; convergence detection |
| Compositional contract system | 7–10K | Contract types; composition engine; serialization; call-graph traversal |
| CLI + CI integration + regression mode | 4–6K | JSON output; GitHub Action wrapper; contract-diff mode |
| Test infrastructure | 18–22K | Per-domain unit tests; integration; precision canary; CVE regression; property-based testing |
| **TOTAL** | **75–85K** | **50–60K genuinely novel** |

**Novelty vs. engineering split:** ~30% novel. Novel components: reduction operator ρ (~4K), quantitative widening skeleton (~3K), speculative domain design (~5K), cache-state-parameterized composition (~4K), independence condition proofs. Engineering: cache AI core (~9K), fixpoint engine (~4K), lifter adapter (~5K), CLI/tests (~25K+), contract serialization (~5K).

**Timeline:** 16–22 months for one experienced PL/security researcher.

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Phase Gate 1: Composition theorem + design | 2 months | Paper proofs; independence characterization |
| Phase Gate 2: Precision canary | 2–3 months | D_cache ⊗ D_quant on AES; ~4K LoC |
| Phase Gate 3: Speculative prototype | 4–6 months | Full D_spec ⊗ D_cache ⊗ D_quant; ρ; ~35K LoC |
| Phase Gate 4: Composition + CLI + eval | 2–3 months | Contracts, regression mode, CI; ~20K LoC |
| Evaluation + paper writing | 3–4 months | Benchmarks, baselines, paper |
| Buffer for redesign iterations | 3–4 months | ρ redesign, widening redesign, etc. |

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|:-----------:|--------|------------|
| ρ precision failure (provable but adds zero precision) | 25% | HIGH — degrades paper to "CacheAudit + speculation" | Phase-gated: precision canary validates counting; ρ tested on merge-point diamond CFGs; single-pass ρ fallback |
| ρ soundness/termination failure | 15% | HIGH — requires restricted single-pass ρ | Prove on paper first; single-pass ρ is monotone by construction |
| Independence condition fails for >50% of crypto patterns | 30% | MEDIUM-HIGH — composition degrades; Rényi pivot adds 1–2 months | Phase Gate 1 tests 4 patterns; Rényi fallback; explicit correction terms |
| Speculative bounds vacuous (>10×) | 30% | MEDIUM — Reduced-B fallback preserves composition + quantitative | Phase Gate 3 catches early; non-speculative analysis still novel |
| Curve25519 memory scalability | 20% | MEDIUM — lose one benchmark | State sharing; incremental composition; non-speculative fallback for Curve25519 |
| Lifter adapter SIMD/AES-NI gaps | 15% | LOW-MEDIUM — grows adapter LoC | Restrict instruction subset; test against concrete execution; accept gaps as known limitations |
| 2025 relevance concern (reviewer fatigue with cache side channels) | 15% | MEDIUM — reviewer assignment risk | LeaVe CCS 2023 disproves "passé" narrative; foreground contract paradigm, not cache analysis per se |

**Overall kill probability: ~30–35% at Reduced-A scope with all amendments.** This accounts for partial-success modes (a tool with 5× loose bounds is still publishable; composition working for 60% of patterns is still novel). The phase-gate structure ensures failure is detected early (months 2–8), not after 18 months.

## Scores

| Criterion | Score | Justification |
|-----------|:-----:|---------------|
| **Value** | **7/10** | Directly answers LeaVe's open question; CI regression detection is genuinely valuable; but direct audience is narrow (~50–100 crypto maintainers) and LLM-based triage covers the easy cases |
| **Difficulty** | **7/10** | ~50–60K genuinely novel LoC; CacheAudit existence proof validates core technique; novel ρ and composition are hard but contained; prior art (CacheAudit, Spectector) provides scaffolding |
| **Best-Paper Potential** | **6/10** | Deepest genuine novelty of any approach (ρ, DEEP NOVEL); answers Distinguished Paper's open question; bridges PL and security. However, "synthesis" headwind, ~8% best-paper probability from debate consensus, and precision uncertainty limit ceiling. Sufficient for a solid CCS paper, marginal for best-paper without a dramatic precision result. |
| **Feasibility** | **7/10** | Polynomial scalability; phase-gated risk management; CacheAudit existence proof; ~65% probability of publishable paper; clear fallback paths (direct product, Reduced-B) |

## Amendments Incorporated

### From Depth Check (all 8 binding amendments)

1. **Amendment 1: Mandatory Lifter Reuse.** No from-scratch binary lifter. 3–5K LoC adapter on BAP/angr/Ghidra. ~150 crypto-critical instructions. AVX-512 deferred. ✅
2. **Amendment 2: LRU-First, PLRU-Deferred.** LRU replacement for v1. PLRU gap quantified as 10–50× over-approximation (13–22 spurious bits per 4 tainted sets). ARM Cortex-A as primary platform. ✅
3. **Amendment 3: Prove Composition Theorem First.** Phase Gate 1. Independence tested on ≥3 crypto patterns before implementation. ✅
4. **Amendment 4: Build Precision Canary Before Full Framework.** Phase Gate 2. D_cache ⊗ D_quant on AES T-table. 3× target, 5× redesign, 10× kill. ✅
5. **Amendment 5: Scope to Reduced-A (~55–65K LoC novel).** Certificates, μarch contract language, PLRU/FIFO, multi-level cache, parallel composition all deferred. ✅
6. **Amendment 6: Foreground Regression Detection.** First-class use case. CVE regression benchmark. Contract-diff mode. ✅
7. **Amendment 7: Fix Cache Geometry Parameter.** 64 sets × 8 ways (not 4096 sets). ~128 bytes per abstract state. ✅
8. **Amendment 8: Address LLM Positioning.** Explicit paragraph: LLMs handle pattern-matching triage; tool handles quantitative residual cases and provides formal guarantees LLMs cannot. ✅

### From the Adversarial Debate (new amendments)

9. **Amendment D1: Explicit ρ-at-merge-points treatment.** ρ applied after join; inner ρ-fixpoint iterated at merged state; single-pass fallback if termination fails. Precision test on diamond CFG added to Phase Gate 3. (From Math critique §2.)

10. **Amendment D2: 4th benchmark with genuine cache-state correlation.** T-table AES with related subkeys added to composition benchmark suite. ChaCha20's trivial satisfaction of independence noted as a limitation. (From Math critique §5.)

11. **Amendment D3: Honest Rényi fallback characterization.** Rényi is insurance against abandonment, not a path to strong results. Budget +1–2 months if triggered. 2–5× additional precision loss acknowledged. (From Math critique §5.)

12. **Amendment D4: Regression detection as primary use case.** Foregrounded throughout. Contract-diff mode with thresholded alerts. cachegrind diff as baseline. "So what?" answered primarily by regression detection. (From Skeptic §6.)

13. **Amendment D5: Explicit LLM competition positioning.** LLMs handle ~90% of pattern-matching cases. Tool targets residual 10% requiring quantitative reasoning, plus formal guarantees (conditional on assumed μarch contracts) that LLMs cannot produce. Complementary workflow: LLMs for first-pass triage, formal analysis for hard tail and CI regression. (From Skeptic §7.)

14. **Amendment D6: Curve25519 memory mitigation.** State sharing via hash-consing; incremental composition via contract extraction per ladder step; non-speculative fallback if memory exceeds budget. 3 GB estimate reduced to ~300–600 MB with state sharing. (From Skeptic §5.)

15. **Amendment D7: Speculative context count as empirical.** No claim of "10–20 contexts." Context count measured empirically in Phase Gate 3. Analysis parameterized by W. Tractability threshold: >500 contexts triggers W reduction or non-speculative fallback for that function. (From Skeptic §2.)

16. **Amendment D8: Bounded speculation soundness proof.** Residual leakage term for paths of length > W: at most log₂(⌈F/64⌉) bits per misspeculation point, where F is function memory footprint. Added to speculative collecting semantics proof obligations. (From debate synthesis.)

17. **Amendment D9: Conditional formal bound framing.** Guarantees are "conditional formal bounds under assumed microarchitectural contracts," not unconditional formal guarantees. Honestly acknowledged throughout. Strictly better than the status quo of implicit, unauditable assumptions. (From Skeptic cross-cutting critique.)

18. **Amendment D10: CI timing target revised.** Full BoringSSL crypto/ directory: ~30–90 minutes (not 10–15 minutes). Most functions trivial (<1 sec); ~30–50 crypto-critical at 30 sec–5 min each. CI-compatible but not "fast." (From depth check reconciliation.)

## Debate-Driven Improvements

The adversarial process produced the following concrete changes to the approach:

1. **Foregrounded regression detection over absolute bounds.** The Skeptic's "so what?" critique (§6) showed that absolute bounds on Intel/AMD hardware are useless due to the LRU→PLRU gap (10–50× loose). Regression detection survives all precision concerns. The approach now positions regression detection as the *primary* use case, with absolute bounds on ARM (LRU) as the secondary precision benchmark.

2. **Eliminated the unsubstantiated context-count claim.** The original "10–20 speculative contexts" had zero evidence. We now treat context count as empirical, measured in Phase Gate 3, with explicit tractability thresholds.

3. **Added the 4th composition benchmark.** The Math Assessor showed ChaCha20 is useless for testing composition (zero leakage, trivial independence). T-table AES with related subkeys creates genuine cache-state correlation that stress-tests the independence condition.

4. **Specified ρ's interaction with join operations.** The Math Assessor identified a gap: join can "un-prune" states ρ eliminated. The approach now specifies ρ-after-join with inner fixpoint iteration and a single-pass fallback.

5. **Honestly characterized the Rényi fallback.** Originally presented as a simple Plan B, the debate revealed it requires 1–2 months of additional research and produces 2–5× looser bounds. The approach now budgets this explicitly and does not count on it for strong results.

6. **Explicitly positioned against LLM-based triage.** The Skeptic's novel argument that LLMs handle 90% of practical cases is addressed: the tool targets the 10% requiring quantitative reasoning and provides conditional formal guarantees LLMs cannot produce.

7. **Addressed Curve25519 memory with state sharing and incremental composition.** The Skeptic's 3 GB estimate is mitigated by hash-consed state sharing and per-step contract composition—exactly the scenario the composition theorem is designed for.

8. **Adopted contract-typed notation from Approach C.** C's insight that "function types ARE contracts" simplifies the contract interface. Contracts are displayed as type signatures, improving accessibility without importing C's infeasible binary-level type inference.

9. **Adopted differential framing from Approach B for regression mode.** B's relational analysis is fatally compromised by path explosion, but the *framing* of regression as a differential property is elegant. The regression mode runs A's polynomial analysis twice and computes contract differences—B's key insight implemented within A's tractable framework.

10. **Revised CI timing target to 30–90 minutes.** The original 10–15 minute claim was unrealistic. 30–90 minutes is CI-compatible (build pipelines routinely take 30–120 minutes) and honest.

11. **Added bounded-speculation residual term.** The debate identified that W ≤ 50 could produce unsound bounds if deeper speculative paths exist. A conservative residual leakage term based on function memory footprint ensures soundness.

12. **Adopted "conditional formal bound" framing.** Guarantees are conditional on assumed microarchitectural contracts that are unverified against proprietary hardware. This is strictly better than the status quo but honestly less than "end-to-end formal guarantees."
