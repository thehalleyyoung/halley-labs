# Certified Leakage Contracts: Unified Approach Analysis

Three architecturally distinct approaches to compositional, quantitative, speculation-aware side-channel analysis of x86-64 crypto binaries. Each shares a common lifter adapter (~4–6K LoC atop BAP/angr/Ghidra) and targets the same evaluation benchmarks (AES, ChaCha20, Curve25519, CVE regression detection), but differs fundamentally in analysis technique and mathematical foundation.

---

## Approach A: Full Reduced Product Abstract Interpretation

### 1. Extreme Value Delivered

**Who desperately needs this:** Crypto library maintainers at OpenSSL, BoringSSL, libsodium, and AWS-LC—roughly 50–100 engineers maintaining code that protects hundreds of millions of TLS endpoints. Today they have *no* binary-level quantitative side-channel guarantee. Their only options are source-level constant-time reasoning (invalidated by compiler optimizations—CVE-2018-0734, CVE-2018-0735) or monolithic, speculation-unaware tools like CacheAudit (x86-32 only, no composition, no Spectre). Compiler developers inserting speculation barriers (LFENCE, SLH) need quantitative feedback—"this barrier reduces leakage from 4.2 bits to 0"—not boolean pass/fail.

**What becomes possible:** A CI pipeline running on every commit to a crypto library that produces per-function quantitative leakage bounds under explicit microarchitectural contracts, including speculative execution. A TLS library derives whole-program bounds from per-function contracts of its bignum dependency *without re-analyzing the dependency*. Regression detection—flagging when a compiler upgrade increases any function's leakage—becomes automatic and robust even when absolute bounds are conservative. This directly answers the open question identified by LeaVe (CCS 2023 Distinguished Paper): LeaVe verifies hardware honors ISA-level leakage contracts; this tool verifies software leaks at most ε bits *given* a contract. Together they close the hardware-software gap.

The abstract interpretation architecture means analysis is polynomial in domain size (bounded by cache geometry, not program size), enabling laptop-feasible whole-library analysis—a scalability advantage both the Math Depth Assessor and Difficulty Assessor confirmed as critical for real-world adoption.

### 2. Genuine Software Artifact Difficulty

**Hard subproblems:**

1. **Reduction operator ρ correctness and precision (~3–5K LoC of correctness-critical code).** The entire framework's value hinges on ρ propagating information between D_spec, D_cache, and D_quant precisely enough to avoid vacuous bounds. A single sign error or missed case silently produces unsound or vacuous results. Testing is extraordinarily difficult—ground-truth bounds from exhaustive enumeration are needed for validation.

2. **Quantitative widening convergence.** Widening for *counting domains* over cache configurations has no off-the-shelf solution. The operator must guarantee convergence, never undercount, and overcount by at most a bounded factor. This will require multiple design iterations. Risk severity: HIGH—if widening overcounts by >10×, the entire quantitative framework is useless.

3. **State space management under speculation.** Even with bounded speculation (W ≤ 50 μops), each misspeculation point forks a new abstract state. For crypto code with table lookups inside loops, speculative contexts can grow to 50–200 before merging. Efficient management (shared sub-states, GC of infeasible contexts) is a systems engineering challenge. Risk severity: MEDIUM—fallback to non-speculative analysis preserves most value.

4. **Integration testing across three domains.** Testing the *interaction* through ρ requires constructing programs where the reduced product is measurably tighter than the direct product, then verifying against ground truth. The test matrix is combinatorially large.

5. **Adapter fidelity for SIMD/AES-NI.** Abstract transfer functions for vpshufb, aesenc, and pclmulqdq must soundly over-approximate cache effects of wide-register operations across multiple cache lines simultaneously.

**Realistic LoC estimates (DA audit-corrected):**

| Subsystem | DV Claimed | DA Realistic | Notes |
|-----------|-----------|-------------|-------|
| Lifter adapter | ~4K | 4–6K | SIMD/AES-NI transfer functions are painstaking |
| D_spec | ~11K | 8–12K | Conceptually simpler tagged powerset |
| D_cache | ~11K | 9–13K | CacheAudit's cache domain is ~5–8K; taint adds complexity |
| D_quant | ~9K | 8–12K | Widening iterations push toward upper bound |
| Reduced product + fixpoint | ~12K | 10–14K | ρ is ~3–5K of intricate correctness-critical code |
| Composition system | ~9K | 7–10K | |
| CLI | ~4K | 3–5K | |
| Tests | ~13K | 18–22K | **Underestimated.** Property-based + differential + integration testing required |
| **Total** | **~73K** | **75–85K** | **50–60K genuinely novel** |

**Novelty vs. engineering split:** ~25–30% novel. Novel components: reduction operator ρ (~4K), quantitative widening (~3K), speculative domain design (~5K), cache-state-parameterized composition (~4K). Engineering: cache AI core (~9K), fixpoint engine (~4K), lifter adapter (~5K), CLI/tests (~22K+), contract serialization (~5K).

**Timeline estimate:** 16–22 months for one experienced PL/security researcher. Phase-gated: composition theorem + design (2 mo) → precision canary D_cache ⊗ D_quant (2–3 mo) → speculative domain + reduced product (4–6 mo) → composition system + CLI (2–3 mo) → evaluation + paper (3–4 mo) → buffer for redesign (3–4 mo). Aligns with CacheAudit's ~3-year history for a simpler system, adjusted for reuse of CacheAudit's published techniques.

**Top engineering risks:**

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Quantitative widening produces vacuous bounds | HIGH | Phase Gate 2 catches this early; v1 targets fixed-iteration crypto only (no widening needed) |
| Reduction operator ρ too weak or unsound | HIGH | Prove ρ on paper before implementing; build precision canary first |
| Speculative context explosion | MEDIUM | Bounded W ≤ 50; fallback to non-speculative analysis preserves most value |

### 3. New Mathematics Required

**Math inventory:**

| # | Result | Classification | Load-Bearing? |
|---|--------|---------------|--------------|
| A1 | Speculative collecting semantics — transient execution traces with residual cache effects | NOVEL | YES — without it, degrades to CacheAudit |
| A2 | Galois connection (or γ-only soundness) from speculative collecting semantics to D_spec | NOVEL | YES — without it, speculative analysis has no formal guarantee |
| A3 | Tainted abstract cache-state domain D_cache — must/may LRU with taint annotations | INSTANTIATION | YES — core cache model |
| A4 | Quantitative channel-capacity domain D_quant — abstract counting over taint-restricted configs | NOVEL | YES — produces actual bit-level bounds |
| A5 | **Reduction operator ρ for D_spec ⊗ D_cache ⊗ D_quant** | **DEEP NOVEL** | YES — without it, direct product bounds are vacuous |
| A6 | Quantitative widening operator ∇_quant with bounded precision loss (≤ 2^k) | **DEEP NOVEL** | ENABLING — not required for v1 fixed-iteration targets |
| A7 | Additive composition rule B_{f;g}(s) = B_f(s) + B_g(τ_f(s)) | INSTANTIATION | YES — mechanism for per-function contracts |
| A8 | Independence condition characterization for standard crypto patterns | NOVEL | YES — if independence fails, composition is unsound |
| A9 | Fixpoint convergence of reduced product iteration | INSTANTIATION | YES — analysis must terminate |

**Crown jewel: The reduction operator ρ (A5).** This is the single most technically novel component—a three-way reduction propagating information across speculation, cache taint, and quantitative capacity domains. No prior reduced product in the abstract interpretation literature combines these three concerns. Reduced products exist (Cousot & Cousot, 1979; Granger, 1992), but the specific cross-domain propagation—where D_spec prunes transient cache accesses in D_cache, and D_cache's taint status zeros capacity contributions in D_quant—has no direct prior art. The MDA confirms this is "not a routine instantiation" and the monotonicity/termination proofs require new arguments specific to the interaction of speculative reachability, cache taint, and counting abstractions.

**Mathematical risk assessment (MDA): ~35% overall.**

| Risk | Probability | Impact |
|------|------------|--------|
| ρ monotonicity/termination failure (A5) | ~15% | Mitigation: restrict to single-pass reduction (weaker but monotone by construction) |
| ∇_quant vacuous bounds on variable-iteration code (A6) | ~20% | Mitigation: target fixed-iteration crypto for v1 (CacheAudit's approach) |
| Independence condition fails for standard crypto (A8) | ~30% | Mitigation: Rényi entropy fallback (Boreale, 2015); yields looser bounds |
| Speculative collecting semantics corner cases (A1) | ~10% | Conceptually clear; risk in nested speculation / store-to-load forwarding |

**Math-to-artifact coupling: Tight.** The abstract domains *are* the implementation. Every lattice operation, transfer function, and widening operator translates directly to code. The reduction operator ρ executes at every program point during analysis. The composition theorem directly drives the contract system's API. Fallback paths exist: without A5, you get a direct product with loose bounds (functional but less precise); without A6, you restrict to fixed-iteration loops (limiting but acceptable for v1); without A1–A2, you get CacheAudit-plus-composition (less novel but still a contribution).

### 4. Best-Paper Argument

**Narrative:** "We introduce the first abstract domain that simultaneously models speculative execution reachability and quantitative cache channel capacity, connected through a novel reduction operator. This enables the first tool producing compositional, quantitative, speculation-aware leakage contracts for compiled cryptographic code—directly answering the software-side open question from LeaVe (CCS 2023 Distinguished Paper)."

**Why best-paper potential is real:** The paper contributes: (1) a genuinely new abstract-interpretation construction (the reduced product with speculative reduction), (2) a composition theorem enabling per-function contracts as a supply-chain primitive, and (3) a tool that detects real CVEs and quantifies the exact bit-level impact of speculation barriers on production crypto binaries. It bridges the PL/formal-methods community (novel domains, soundness proofs) and the systems-security community (CVE detection, CI integration, real-world benchmarks). CCS reviewers reward exactly this dual appeal.

**Headwinds:** The "synthesis" objection—that this combines existing ingredients (CacheAudit's cache domains, Spectector's speculative semantics, Smith's composition)—is real but partially mitigated by the genuinely novel reduction operator ρ, which is not available from prior work. The MDA classifies ρ as DEEP NOVEL, which provides strong defense against the synthesis critique. Another risk: if speculative bounds are vacuous (>100× over-approximation), the tool's key differentiator over CacheAudit evaporates.

### 5. Hardest Technical Challenge

**Consensus across all assessors: The reduction operator ρ.** The DV, MDA, and DA all independently identify ρ as the critical risk. Everything depends on ρ being simultaneously sound (preserving over-approximation), precise (actually tightening bounds meaningfully), and efficient (computing to fixpoint without blowing up). The speculative-infeasibility-to-cache-taint propagation is especially delicate: if ρ is too aggressive (unsound), the tool misses real leaks; if ρ is too conservative (imprecise), speculative analysis adds nothing and the key selling point evaporates.

**Mitigation strategy:** Prove ρ correct on paper *before* implementing. Build the precision canary (D_cache ⊗ D_quant without speculation) first to validate the cache-counting machinery. Then add D_spec and ρ incrementally, testing on known Spectre gadgets (Kocher's proof-of-concepts) where ground-truth leakage is known. The phase-gate structure (composition theorem → precision canary → speculative prototype) is explicitly designed to de-risk ρ development.

### 6. Scores

| Criterion | DV | DA | MDA | Reconciled | Reasoning |
|-----------|----|----|-----|-----------|-----------|
| Value | 8 | — | — | **8** | Consensus: directly answers LeaVe's open question; CI regression detection; broad impact |
| Difficulty | 8 | 7 | — | **7** | DA's lower rating reflects that prior art (CacheAudit) validates core technique. DV's 8 reflects novel ρ. Reconcile at 7: hard but *tractable*, with contained novelty and clear fallbacks |
| Best-Paper Potential | 7 | — | Strong | **7** | MDA confirms deepest math novelty of the three. "Synthesis" headwind limits ceiling. Strong CCS fit |
| Feasibility | 6 | 7 | ~65% | **7** | DV's cautious 6 reflects speculative bound precision risk. DA's 7 reflects phase-gated structure and CacheAudit existence proof. MDA's 65% probability aligns with 7. Reconcile at 7: best risk/reward profile |

---

## Approach B: Modular Relational Verification with Bounded Speculation

### 1. Extreme Value Delivered

**Who desperately needs this:** Security auditors performing *differential* analysis—comparing two binaries (pre-patch vs. post-patch, -O0 vs. -O3, with-LFENCE vs. without) and needing to know exactly how many bits of distinguishing advantage an attacker gains. Binsec/Rel already serves this community with binary-level relational verification, but its boolean constant-time verdicts are too coarse: when a function intentionally leaks bounded information (T-table AES, scatter/gather, table-based S-boxes), Binsec/Rel can only say "not constant-time"—it cannot distinguish 0.5 bits from 128 bits.

**What becomes possible:** Path-sensitive relational analysis that produces not just "these two executions differ in cache behavior" but "across all secret-input pairs, the cache observation channel has at most k distinguishable classes, leaking at most log₂(k) bits." Applied to speculative paths, this yields the first tool that can quantify the *marginal leakage contribution* of individual Spectre gadgets: "removing this branch misprediction window reduces leakage from 3.7 bits to 0.2 bits." Relational summaries at function boundaries enable modular composition without re-exploring callee paths.

**Why this approach has adoption advantages:** Relational verification is the dominant paradigm in binary-level security analysis (Binsec/Rel, RelSE, Pitchfork). By *extending* this paradigm rather than replacing it, Approach B is adoptable by teams already using relational tools. The path-sensitive analysis naturally handles irregular crypto control flow (conditional swaps, branchless-but-not-quite-constant-time patterns) without the precision loss inherent in lattice abstraction.

### 2. Genuine Software Artifact Difficulty

**Hard subproblems:**

1. **Path explosion in relational symbolic execution (FATAL risk per DA).** Self-composition doubles the state space. Speculation forks both copies. A crypto function with a 10-iteration loop and 3 branches per iteration generates ~3^10 × 2 paths relationally. Even with aggressive merging, this is orders of magnitude harder than single-program SE. Binsec/Rel handles only small functions for boolean properties; quantitative relational analysis is strictly harder.

2. **SMT solver performance on relational bitvector queries (HIGH risk).** Relational queries assert properties about *pairs* of executions over bitvector arithmetic, memory loads, and cache states—among the hardest queries for modern SMT solvers. Timeouts are the expected primary bottleneck on any non-trivial function.

3. **#SAT model counting scalability (HIGH risk).** Approximate model counters (ApproxMC) struggle beyond ~10K variables. A relational encoding of cache states for a 64-set, 8-way cache produces formulas of this size *per path*.

4. **Contract extraction from path-sensitive results.** Converting per-path symbolic results to reusable abstract summaries requires either collecting all paths (infeasible for complex functions) or abstracting on-the-fly (hard to do soundly in a relational setting).

5. **Memory model for relational pairs.** Two copies of the program share no memory but their cache behaviors must be compared. Encoding this SMT-friendly without formula size blowup is a significant design challenge.

**Realistic LoC estimates (DA audit-corrected):**

| Subsystem | DV Claimed | DA Realistic | Notes |
|-----------|-----------|-------------|-------|
| Lifter adapter | ~4K | 4–6K | Same as A |
| Relational SE engine | ~15K | **20–30K** | **Severely underestimated.** Self-composition, path exploration, SMT integration, memory model. Binsec/Rel took multiple years for boolean-only. |
| Speculation modeling | ~8K | 10–14K | Speculative forking of both relational copies is combinatorially expensive |
| Model counting | ~6K | 8–12K | BV-to-boolean encoding for #SAT is non-trivial |
| Contract extraction | ~8K | 8–12K | Well-known hard problem in SE-based analysis |
| Composition engine | ~7K | 7–9K | Plausible |
| CLI | ~4K | 3–5K | |
| Tests | ~13K | 16–22K | Path-sensitive testing compounds the problem |
| **Total** | **~65K** | **85–110K** | **55–70K novel. DV estimate severely underestimated.** |

This is the single largest disagreement between assessors. The DA flags the DV's 65K estimate as unrealistic by 30–70%, driven primarily by the relational SE engine, which alone is a PhD-thesis-level system.

**Novelty vs. engineering split:** ~25% novel. Novel: speculation-aware relational verification (~8K), quantitative relational contracts (~6K), cache-state model counting encoding (~5K). Engineering: relational SE engine (~22K), SMT/model counter integration (~11K), lifter/CLI/tests (~25K+).

**Timeline estimate: 24–36 months.** Relational SE engine alone: 8–12 months. Speculation + model counting: 7–10 months. Contract extraction + evaluation: 6–8 months. Buffer: 3–6 months. The DA judges this infeasible for one researcher on a publication timeline—Binsec/Rel's development involved multiple researchers over multiple years for boolean-only verification.

**Top engineering risks:**

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Path explosion kills scalability | **FATAL** | No known general solution. Binsec/Rel struggles at ~100 instructions for qualitative analysis |
| Model counting produces vacuous bounds | HIGH | Approximate counting with PAC guarantees; decompose by cache set |
| SMT solver timeouts on relational queries | HIGH | Fundamental to the approach; limited mitigation available |

### 3. New Mathematics Required

**Math inventory:**

| # | Result | Classification | Load-Bearing? |
|---|--------|---------------|--------------|
| B1 | Self-composition / product program construction for quantitative cache analysis | INSTANTIATION | YES |
| B2 | Bounded speculative path semantics | INSTANTIATION | YES |
| B3 | #SAT / model counting over path constraints for quantitative leakage | ESTABLISHED | YES |
| B4 | Relational soundness of product-program transformation | INSTANTIATION | YES |
| B5 | **Modular relational summary composition** | **NOVEL** | YES — sole genuinely novel math |
| B6 | Complexity bounds on model counting over cache-path constraints | ESTABLISHED | ENABLING only |
| B7 | Soundness of bounded speculation approximation | INSTANTIATION | YES |

**Crown jewel: Modular relational summary composition (B5).** The only genuinely novel mathematical result. Everything else instantiates known frameworks. The challenge: composing path-based relational summaries where each summary is a set of (path-condition, cache-observation, count) triples without exponential blowup at composition boundaries. Prior work on compositional QIF (Yasuoka & Terauchi, 2010; Kawamoto et al., 2015) provides substantial scaffolding.

The MDA explicitly notes this crown jewel is *less impressive* than Approach A's reduction operator—it is a composition mechanism for a specific summary representation, not a new abstract domain construction.

**Mathematical risk assessment (MDA): ~25% overall.** Lower than Approach A because most components are well-established.

| Risk | Probability | Impact |
|------|------------|--------|
| Summary explosion at composition (B5) | ~35% | Exponential triple growth across function boundaries; mitigation: heuristic path merging (lossy but sound) |
| #SAT scalability (B3) | ~20% | Real crypto formulas may time out; mitigation: ApproxMC with (ε,δ) bounds |
| Bounded speculation soundness gap (B7) | ~15% | Engineering tradeoff on W, not mathematical barrier |

**Math-to-artifact coupling: Moderate.** Self-composition and model counting are well-understood enough that implementation proceeds with high confidence. B5 is the mathematical bottleneck driving contract system design. The tool could function *without* B5 as a monolithic analyzer—losing compositionality but retaining quantitative speculation-aware analysis. This decoupling is both a strength (lower risk) and a weakness (narrower unique contribution).

### 4. Best-Paper Argument

**Narrative:** "We show that relational verification—the dominant paradigm for binary-level security—can be extended to produce quantitative leakage bounds under speculative execution, with modular contracts enabling compositional analysis. By staying within the relational paradigm, our approach inherits the path-sensitivity and precision of tools like Binsec/Rel while adding the quantitative dimension that security auditors need."

**Why the argument has traction:** The paper contributes: (1) the first speculation-aware relational analysis with quantitative bounds, (2) modular contract extraction for relational summaries, and (3) tighter bounds than abstract interpretation on irregular crypto patterns (conditional swaps, scatter/gather). The narrative resonates with the security community, and the reviewer sees exactly which prior framework is extended and why the extension is non-trivial.

**Headwinds:** A cynical reviewer could frame this as "Binsec/Rel + ApproxMC + bounded Spectector + modular summaries" (the MDA's characterization). The single novel mathematical contribution (B5) is narrower than Approach A's. Scalability problems may prevent evaluation on meaningful benchmarks, which would be devastating for reviewer credibility. The DA's FATAL rating on path explosion means the tool may not produce results on full cipher functions—the difference between "analyzed one AES round" and "analyzed full AES-256" is the difference between acceptance and rejection.

### 5. Hardest Technical Challenge

**DV identifies:** Model counting scalability on real crypto formulas—the quantitative dimension depends on efficiently counting observation equivalence classes via #SMT.

**DA identifies:** Path explosion in relational SE—the fundamental scalability wall that squares the single-program path count.

**Reconciled view: Path explosion is the harder problem.** Model counting can be mitigated by decomposition (per-cache-set counting) and approximate methods (ApproxMC with PAC guarantees). Path explosion in relational SE has no known general solution and is the constraint that makes the DA rate this approach as infeasible for a single researcher. If path explosion prevents analysis of functions >500 instructions, the model counting question becomes moot because there are no results to count.

**Mitigation:** Taint-directed pruning (only fork on secret-dependent branches), speculation-bounded exploration (limit W), and aggressive path merging. For crypto code with fixed iteration counts, loop unrolling is feasible. But the DA's assessment is that these mitigations are insufficient for production crypto functions.

### 6. Scores

| Criterion | DV | DA | MDA | Reconciled | Reasoning |
|-----------|----|----|-----|-----------|-----------|
| Value | 7 | — | — | **7** | Strong for differential analysis; extends established paradigm; narrower audience than A |
| Difficulty | 7 | **9** | — | **9** | **Major disagreement.** DV's 7 underestimates the relational SE engine (DA flags LoC as 30–70% higher than claimed). DA's 9 reflects the fundamental scalability barrier. Reconcile at DA's 9: the path explosion is not an engineering problem with an engineering solution |
| Best-Paper Potential | 6 | — | Weak | **5** | MDA rates mathematical contribution as weakest. "Extending Binsec/Rel" risks appearing incremental. Scalability may prevent compelling evaluation |
| Feasibility | 7 | **3** | ~75% math | **4** | **Major disagreement.** DV's optimistic 7 assumes reuse of symbolic execution techniques. DA's 3 reflects the fatal path-explosion barrier and 24–36 month timeline. MDA's 75% math probability doesn't account for engineering infeasibility. Reconcile at 4: math works but engineering doesn't scale |

---

## Approach C: Type-Directed Quantitative Information Flow

### 1. Extreme Value Delivered

**Who desperately needs this:** Crypto library *developers* (not just auditors) who want leakage contracts as part of the development workflow, not a post-hoc analysis step. Today, writing side-channel-resistant code requires implicit mental tracking of what depends on secrets—analogous to manual memory management before ownership types. A type system for quantitative information flow makes leakage bounds *syntactically visible* at every function boundary: the function signature *is* the contract.

**What becomes possible:** Every function in a crypto library carries a type annotation: `f : Secret[128] × Public → Public {leaks ≤ 2.3 bits under LRU-PHT}`. Type checking is inherently compositional—the composition rule is type application, the most well-understood composition mechanism in computer science. No contract serialization, fixpoint re-computation, or summary formats needed. The type system naturally handles higher-order functions, callbacks, and function pointers—patterns that abstract interpretation and relational verification struggle with. IDE integration, incremental checking (only re-check changed functions), and compiler-enforced leakage budgets become possible.

**Why this is paradigm-defining:** Type systems have the strongest track record of any formal method for real-world adoption (Rust's ownership types, TypeScript's gradual types). A QIF type system for binary IR that works on real crypto code would establish a new paradigm for side-channel analysis—one that could eventually integrate into compilers as a type-checking pass on IR before code generation. This is the most "agenda-setting" approach: it doesn't just build a tool, it proposes a *language-theoretic framework*.

Both MDA and DA note, however, that this paradigm-defining vision faces a fundamental tension: type systems are inherently coarse, and the binary-level setting exacerbates this. CacheAudit-quality precision (within 3× of exhaustive enumeration) seems unlikely without heavy flow-sensitive extensions that would undermine the type-theoretic elegance.

### 2. Genuine Software Artifact Difficulty

**Hard subproblems:**

1. **Type inference on untyped binary IR (DA: single hardest engineering problem across all three approaches).** Binary code has no type annotations, no structured scoping, and uses registers/memory interchangeably. Inferring quantitative information-flow types requires solving a global constraint system over the entire function. The constraint rules must handle bitwise operations (AND, OR, XOR, shift) that mix type levels—`x XOR secret` is trivially tainted, but `x AND 0xFF` may or may not reduce leakage depending on x's type. No prior system has successfully inferred quantitative IFT types on binary code.

2. **Precision of quantitative types for binary operations.** Source-level QIF type systems work because source code has structure (variables, scopes, types). Binary code destroys this. A type system that taints everything conservatively is useless. Achieving CacheAudit-quality precision requires type-level reasoning about memory layout, register allocation, and compiler idioms—essentially undoing compilation.

3. **Speculation in a type-theoretic framework (genuinely uncharted territory).** The type system must account for instructions that execute speculatively and are later squashed, but whose cache effects persist. The MDA notes it's unclear whether standard effect-system techniques (monadic effects, graded monads) can capture speculative semantics without becoming vacuously imprecise.

4. **Compositionality vs. precision tradeoff.** Function types must conservatively bound *all possible* leakage for *all possible* calling contexts. This can easily produce bounds 10–100× looser than context-sensitive AI analysis—and unlike Approach A, there is no obvious fix within the type-theoretic framework.

5. **No prior art for validation.** Approaches A and B can validate against CacheAudit and Binsec/Rel. Approach C has no comparable existing tool. Every design decision must be validated from scratch against exhaustive enumeration.

**Realistic LoC estimates (DA audit-corrected):**

| Subsystem | DV Claimed | DA Realistic | Notes |
|-----------|-----------|-------------|-------|
| Lifter adapter | ~4K | 4–6K | Same |
| Type system core | ~12K | 12–16K | Binary IR with registers/memory/flags lacks natural type structure |
| Type inference engine | ~10K | **14–22K** | **Significantly underestimated.** Closer to decompilation-quality analysis than standard Hindley-Milner |
| Speculation type extension | ~8K | 8–14K | No prior work; 8K achievable *only if design works on first attempt* |
| Quantitative flow analysis | ~8K | 7–10K | Mechanical given well-typed code |
| Contract generation | ~5K | 4–6K | Mechanical from type derivations |
| CLI | ~4K | 3–5K | |
| Tests | ~12K | 15–20K | Exhaustive judgment-rule testing, soundness validation |
| **Total** | **~63K** | **72–95K** | **45–65K novel. Wide range reflects genuine feasibility uncertainty.** |

**Novelty vs. engineering split: ~45–50% novel (highest of all three approaches).** Novel: binary-level QIF type system (~12K), speculative type extension (~10K), quantitative type inference for binary IR (~12K). Engineering: lifter/CLI/tests (~27K), flow analysis infrastructure (~5K), contract generation (~5K). The high novelty ratio cuts both ways—high publication value but high feasibility risk.

**Timeline estimate: 20–30 months.** Type system design + formalization (3–5 mo) → type inference engine (5–8 mo) → speculation extension (3–5 mo) → quantitative flow + contracts (3–4 mo) → evaluation (3–4 mo) → buffer for inference redesign (3–4 mo). Wide range reflects genuine uncertainty: if the type inference design works early, 20 months; if binary-level typing proves fundamentally imprecise, the project could stall.

**Top engineering risks:**

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Type inference on binary IR infeasible in practice | HIGH | May require manual annotations at function boundaries (acceptable for crypto but limits automation) |
| Binary-level type system too imprecise (10–100× loose) | HIGH | No obvious fix within type-theoretic framework |
| Speculative type extension has no viable design | MEDIUM | Fallback to non-speculative QIF typing is still a novel contribution |

### 3. New Mathematics Required

**Math inventory:**

| # | Result | Classification | Load-Bearing? |
|---|--------|---------------|--------------|
| C1 | Quantitative IFT system — flow types expressing bits-of-leakage | INSTANTIATION | YES — the type system is the analysis |
| C2 | Type soundness theorem — well-typed programs satisfy leakage bounds | INSTANTIATION | YES — without soundness, types are meaningless |
| C3 | Quantitative subject reduction | INSTANTIATION | YES — required for C2 |
| C4 | **Speculative type extension** — type annotations for transient information flows | **NOVEL** | YES — without it, system ignores speculation |
| C5 | Subtyping for leakage refinement | INSTANTIATION | ENABLING — improves usability, not required for correctness |
| C6 | **Speculative type safety** — well-typed programs don't speculatively leak more than their type allows | **NOVEL** | YES — key differentiator |
| C7 | Compositionality from type structure | ESTABLISHED | YES — but falls out of standard type theory |

**Crown jewel: Speculative type safety (C6).** Combines the speculative type extension (C4) with quantitative soundness. No prior type system models speculative execution with quantitative bounds. The proof requires showing that the transient typing context's leakage accounting is sound across the speculation/resolution boundary—where type preservation must hold even as transient instructions "undo" their architectural effects but not their cache effects. This breaks standard subject-reduction technique and requires a step-indexed logical relation.

**Honest depth comparison (MDA):** The crown jewel is "less technically demanding" than Approach A's reduction operator. Type soundness proofs, even for novel type systems, follow well-worn proof patterns (Wright & Felleisen, 1994). The speculative extension adds new proof *content* within established *methodology*, whereas Approach A's ρ requires new domain-specific arguments with no direct template.

**Mathematical risk assessment (MDA): ~20% overall.** Lowest of the three approaches.

| Risk | Probability | Impact |
|------|------------|--------|
| Speculative type extension unwieldy for binary IR with complex control flow (C4) | ~20% | Type systems work best on structured programs |
| Type-based bounds too coarse (overall precision) | ~30% | Inherent to type systems; flow-sensitive extensions are expensive |
| Binary-level type inference undecidable/intractable in practice | ~25% | May require manual annotations |

**Math-to-artifact coupling: Strong with a caveat.** Type checking IS the analysis—the tightest possible coupling. However, the tighter the coupling, the harder it is to add engineering heuristics for precision. Abstract interpretation (Approach A) allows precision engineering through widening strategies and trace partitioning. A type system's precision is largely determined by the type structure. Without C4/C6 (speculative types), you get a non-speculative QIF type system—still useful but not novel beyond existing work (Hunt & Sands, 2006; Smith, 2007). Without type inference, you need annotations—impractical for binaries.

### 4. Best-Paper Argument

**Narrative:** "We develop the first quantitative information-flow type system for binary code that expresses side-channel leakage bounds as types, with a speculative type extension that tracks transient information flows. Type soundness directly implies that well-typed code satisfies its leakage contract under speculative execution—compositionality comes for free from the type discipline."

**Why the ceiling is highest:** This is the most *paradigm-defining* approach. Rather than building another analysis tool, it proposes a formal framework for reasoning about quantitative side-channel leakage. The type soundness theorem—connecting types to operational min-entropy under speculative cache semantics—is a single, clean, powerful result. It bridges three communities: PL theory (novel type system), QIF (new application), and hardware security (speculation-aware cache analysis). Best papers at CCS/S&P increasingly reward work that *reframes* a problem (LeaVe reframed hardware verification as contracts; this reframes side-channel analysis as typing). A mechanized proof (Coq/Lean, ~3K LoC proof script) would dramatically strengthen the case.

**Headwinds:** The DV notes this approach has the highest risk despite the highest ceiling. If the type system produces bounds 10–100× loose—a real possibility given the binary-level setting—the evaluation section will be weak regardless of the theoretical elegance. CCS reviewers will ask "does it actually find real bugs?" and imprecise bounds may not distinguish meaningful leaks from noise. Also, the "binary-level type inference" claim may seem overreaching if the system requires manual annotations in practice.

### 5. Hardest Technical Challenge

**DV identifies:** Type soundness under speculative semantics—a research-level proof requiring a logical-relations argument over non-standard evaluation (transient instructions modify cache but not architectural state).

**DA identifies:** Type inference on untyped binary IR—no prior system has done this successfully, and it may be fundamentally infeasible.

**Reconciled view: Both are critical, but type inference is the higher-priority risk.** The type soundness proof is difficult but tractable with a two-stage approach (non-speculative first, then speculative extension). Type inference feasibility is an existential question: if no workable inference algorithm exists for binary IR, the entire approach collapses. The MDA's low mathematical risk rating (~20%) applies to the theorems, but the DA's assessment that inference is "closer to decompilation-quality analysis than standard Hindley-Milner" is the binding constraint.

**Mitigation:** Two-stage development: (1) non-speculative type soundness first (independently publishable), then (2) speculative extension. For inference, accept manual annotations at function boundaries as an acceptable compromise for crypto libraries (few functions, well-documented). Use mechanized proofs (Coq/Lean) for the soundness theorem to catch subtle errors early and strengthen the best-paper case.

### 6. Scores

| Criterion | DV | DA | MDA | Reconciled | Reasoning |
|-----------|----|----|-----|-----------|-----------|
| Value | 7 | — | — | **7** | Paradigm-defining; inherently compositional; but binary-level type inference is a hard sell for practitioners |
| Difficulty | 9 | 8 | — | **8** | DV's 9 reflects novel type soundness proof; DA's 8 notes it's easier than B (no path explosion, polynomial inference). Reconcile at 8: very hard but not fundamentally intractable like B |
| Best-Paper Potential | 8 | — | Moderate | **8** | Highest ceiling. MDA rates math contribution as "moderate" (less deep than A's ρ) but paradigm-reframing compensates. Clean type soundness theorem + mechanized proof = strong CCS/POPL fit |
| Feasibility | 5 | 5 | ~80% math | **5** | Full agreement across assessors. Math is likely to work (80%), but binary-level type inference feasibility is genuinely uncertain. No prior system validates the fundamental approach |

---

## Cross-Approach Comparison

### Reconciled Comparison Table

| Dimension | A: Reduced Product AI | B: Modular Relational | C: Type-Directed QIF |
|-----------|----------------------|----------------------|---------------------|
| **Mathematical foundation** | Lattice theory, Galois connections, reduced products | Self-composition, symbolic execution, model counting | Type theory, graded monads, logical relations |
| **Novel load-bearing theorems** | 4 (+ 1 DEEP NOVEL enabling) | 1 | 2 |
| **Deepest single result** | DEEP NOVEL (ρ) | NOVEL (summary composition) | NOVEL (speculative type safety) |
| **Composition mechanism** | Explicit contract (τ_f, B_f) with proved additive rule | Relational summary substitution (explosion risk) | Type application (free from type discipline) |
| **Speculation handling** | Abstract domain D_spec with bounded reachability | Speculation tree with bounded symbolic exploration | Graded monad □_W indexed by window depth |
| **Quantitative counting** | Abstract counting over taint-restricted configs | #SMT model counting over observation formulas | Type-level leakage accumulation via effect system |
| **Scalability profile** | Polynomial in domain size | Exponential path exploration (FATAL risk) | Polynomial type inference |
| **Realistic LoC** | 75–85K (50–60K novel) | **85–110K** (55–70K novel) | 72–95K (45–65K novel) |
| **Novelty ratio** | ~25–30% | ~25% | ~45–50% |
| **Timeline (1 researcher)** | 16–22 months | **24–36 months** | 20–30 months |
| **Math risk (MDA)** | ~35% | ~25% | ~20% |
| **Precision ceiling** | Highest | Medium | Lowest |
| **Adoption path** | Standalone tool → CI integration | Extends Binsec/Rel ecosystem | Paradigm shift → compiler integration |
| **Best venue** | CCS | S&P or CCS | CCS or POPL |

### Reconciled Score Summary

| Criterion | A: Reduced Product | B: Relational | C: Type-Directed |
|-----------|:------------------:|:-------------:|:----------------:|
| Value | **8** | 7 | 7 |
| Difficulty | 7 | 9 | 8 |
| Best-Paper Potential | 7 | 5 | **8** |
| Feasibility | **7** | 4 | 5 |

### Key Disagreements and Resolutions

1. **Approach B LoC and feasibility.** The DV estimated 65K LoC and rated feasibility 7/10. The DA assessed 85–110K LoC and rated feasibility 3/10, citing the relational SE engine alone as a multi-year PhD-level effort and path explosion as a FATAL risk. **Resolution:** The DA's assessment is more credible here—it is grounded in concrete analysis of the relational SE engine's subsystems and calibrated against Binsec/Rel's actual development history. The DV's estimate underweights the engineering complexity of relational symbolic execution.

2. **Approach A difficulty.** The DV rated difficulty 8/10; the DA rated 7/10. **Resolution:** The DA's argument that CacheAudit provides existence proof for the core technique, and that the novel components are well-contained with clear fallbacks, justifies the lower rating. Reconciled at 7.

3. **Approach C crown jewel depth.** The DV describes the type soundness proof as a "research-level proof" requiring novel logical-relations arguments. The MDA rates it as "less technically demanding" than Approach A's ρ, noting it follows established proof patterns. **Resolution:** Both are correct at different levels. The *proof structure* follows Wright & Felleisen patterns (MDA is right), but the *proof content* for the speculative case is genuinely new (DV is right). The reconciled best-paper score of 8 reflects that paradigm-reframing matters more to reviewers than raw proof depth.

### Recommendation

**Approach A is the highest expected-value choice.** It balances genuine novelty (DEEP NOVEL reduction operator ρ), engineering tractability (polynomial analysis, CacheAudit existence proof, 16–22 month timeline), and strategic positioning (directly answers LeaVe's open question). Phase-gated risk management provides early detection of failures and clear fallback paths.

**Approach C has the highest ceiling** but should only be chosen by a team with strong PL-theory capabilities willing to accept feasibility uncertainty in exchange for potential CCS/POPL best-paper impact. The type-theoretic reframing is the most agenda-setting contribution if it works.

**Approach B is not recommended** in its current form. The DA's FATAL rating on path explosion and severe LoC underestimation make it infeasible for a single researcher on a publication timeline. The mathematical contribution is the weakest, and the engineering effort is the largest.
