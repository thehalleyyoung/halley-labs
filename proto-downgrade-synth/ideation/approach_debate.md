# NegSynth: Approach Debate

**Method:** Three independent expert assessments (Math Depth Assessor, Difficulty Assessor, Adversarial Skeptic) followed by cross-challenge synthesis.

---

## Part I: Math Depth Assessment

### Approach A: Protocol-Aware Symbolic Execution

#### T3 (Protocol-Aware Merge Correctness) — Depth: 5/10

**Load-bearing?** Partially. The merge operator *is* needed for tractability, but the O(n) vs O(2^n) claim relies on four "algebraic properties" that are essentially *definitional restrictions* on the domain. You're proving that a finite, deterministic, acyclic selection process has polynomial state space — close to saying "finite things are finite." The bisimilarity preservation proof is a standard congruence argument over an LTS with a well-founded ordering.

**Novel?** Moderately. The *combination* of veritesting + protocol domain restrictions is new, but each ingredient is textbook. Kuznetsov (PLDI 2012) did state merging. Avgerinos (ICSE 2014) did veritesting. Milner bisimulation is 1980s. The novelty is "we noticed negotiation protocols are easy for state merging."

**Proof failure risk:** Low (~5%). The four properties genuinely hold. Risk is that real implementations *violate* the idealized model (callback-driven cipher selection isn't purely deterministic), forcing ugly escape hatches.

**Honest effort:** ~1.5 person-months, not 4.

#### T4 (Bounded Completeness) — Depth: 4/10

**Load-bearing?** Yes, but it's a composition theorem — a transitivity-of-soundness argument. The ε term is *empirically* bounded, meaning the theorem is really saying "we believe this works because we tested it."

**Novel?** The composition across the specific pipeline is new as an artifact, but composition theorems of this shape are standard in verified compilation (CompCert).

**Proof failure risk:** Medium (~15%). Risk isn't that composition is wrong — it's that ε is unacceptably large, making bounded completeness near-vacuous.

#### T1 (Extraction Soundness) — Depth: 3/10

Standard simulation relation adapted from 40-year-old abstract interpretation theory. Necessary but expected.

### Approach B: Abstract Interpretation + CEGAR

#### B1 (Negotiation Domain Soundness) — Depth: 4/10

**Load-bearing?** Yes — standard cost of doing abstract interpretation. Every AI paper proves domain soundness. The 4-sub-domain reduced product is genuine engineering, but mathematically it's a direct application of Cousot's reduced product construction.

**Novel?** The specific domain is new, but the *technique* is 1977-era abstract interpretation.

**Proof failure risk:** Medium (~20%). Real risk is domain too coarse (false negatives) or too fine (timeouts).

#### B2 (CEGAR Termination) — Depth: 3/10

**Load-bearing?** Partially. Termination follows trivially from finiteness of predicate vocabulary. Says nothing about iteration count or practical precision.

#### B3 (Domain-Relative Completeness) — Depth: 3/10 — **ORNAMENTAL**

This theorem is vacuous: "we find all attacks expressible in our domain" is a tautology. Every sound abstract interpretation is trivially "domain-relative complete." **This is ornamental formalism.**

### Approach C: Differential Mining

#### C1 (Covering-Design Bound) — Depth: 6/10 — **DEEPEST THEOREM**

**Load-bearing?** Yes, and the most mathematically interesting theorem across all three approaches. Connects combinatorial design theory to protocol testing in a non-obvious way. The bound B(n,k) is meaningful and tight.

**Novel?** Moderately to genuinely novel. Covering designs are well-studied, but application to differential protocol analysis is new.

**Proof failure risk:** Medium (~15%). Risk that behavioral deviations aren't well-modeled as pairwise — 3-way interactions could escape.

#### C2 (Exploitability Game) — Depth: 5/10

**Load-bearing?** Partially. Game-theoretic bridge connecting deviations to DY adversary strategies is genuinely needed but follows standard security-game machinery.

### Math Depth Verdict

**Ranking by mathematical honesty × depth × necessity: C > A >> B**

| Theorem | Load-Bearing? | Novel? | Depth | Risk | Ornamental? |
|---------|:---:|:---:|:---:|:---:|:---:|
| A-T3 | ✅ Partially | Moderate | 5 | Low | No, but overstated |
| A-T4 | ✅ Yes | Low | 4 | Medium | No |
| A-T1 | ✅ Expected | No | 3 | Low | No |
| B-B1 | ✅ Expected | Low | 4 | Medium | No |
| B-B2 | ⚠️ Weakly | No | 3 | Low | Borderline |
| **B-B3** | ❌ | No | 3 | ~0% | **YES** |
| **C-C1** | ✅ Yes | **Genuine** | **6** | Medium | No |
| C-C2 | ✅ Partially | Moderate | 5 | Medium | No |
| C-C3 | ✅ Routine | No | 3 | Low | No |

---

## Part II: Difficulty Assessment

### Approach A — Difficulty: 7.5/10

**LoC estimate:** Fair (50K ± 5K). KLEE integration layer underestimated (7K → 10-12K). Concretizer slightly inflated (6K → 3-5K).

**Hidden complexity trap:** KLEE FFI — Rust calling into KLEE's C++ internals requires `cxx` bindings to `ExecutionState`, `MemoryObject`, `ObjectState`. S2E project required ~20 person-months for integration layer.

**Timeline:** 14-18 months with KLEE expertise on team. Solo PhD: 24-30 months.

**Most likely engineering failure:** (1) KLEE version lock-in (LLVM version mismatch), (2) OpenSSL's custom allocators confusing KLEE's memory model, (3) merge operator corner cases on `STACK_OF(SSL_CIPHER)` macro-expanded data structures, (4) Rust↔C++ FFI memory safety.

**Similar systems:** KLEE (engine reused), S2E (extension paradigm), veritesting (generic merge). Unique: source→DY model automatically has no direct prior art.

### Approach B — Difficulty: 9/10

**LoC estimate: SEVERELY UNDERESTIMATED.** 55K → 70-85K realistic. LLVM IR abstract interpreter alone is 25-35K (IKOS is ~45K). The proposal claims 15K for this component.

**Hidden complexity trap:** Building a correct abstract interpreter that handles `phi` nodes, `getelementptr`, memory model, calling conventions, SSA form is massive. Missing any LLVM IR instruction = soundness hole.

**Timeline:** 24-36 months minimum even with AI expertise. Solo PhD: 3-4 years — this IS the dissertation.

**Most likely engineering failure:** (1) Domain precision death spiral — too imprecise → CEGAR fires → adds disjunctions → analysis explodes → reinvented symbolic execution, (2) LLVM IR coverage gaps, (3) Widening operator tuning, (4) Team skill bottleneck — abstract interpretation expertise is rare.

**Verdict:** "Three PhD theses stapled together, each individually high-risk." Building a C abstract interpreter AND inventing a novel domain AND implementing CEGAR AND producing evaluation, all in one paper.

### Approach C — Difficulty: 5.5/10

**LoC estimate:** Most accurate (45K ± 5K), but difficulty per LoC varies enormously.

**Hidden complexity trap:** Semantic alignment engine — aligning APIs across four libraries with different representations, error handling, build configs is combinatorial. ~30 `SSL_CTX_set_*` functions per library.

**Timeline:** Most feasible — 10-14 months for 2-3 person team.

**Most likely engineering failure:** (1) Semantic alignment errors produce false signals, (2) covering design misses critical parameter combinations, (3) security ranker can't distinguish intentional design choices from vulnerabilities, (4) novelty rejection at review.

**CRITICAL LIMITATION:** Cannot produce absence certificates. Cannot deliver headline bounded-completeness theorem (T4).

**Scoop risk:** HIGH. Most prior art (Frankencerts, mucert, tlspuffin). Most incremental methodology.

### Difficulty Verdict

| Dimension | A | B | C |
|-----------|:---:|:---:|:---:|
| **Difficulty** | 7.5/10 | 9/10 | 5.5/10 |
| **Algorithmic novelty** | 7/10 | 8.5/10 | 5.5/10 |
| **Engineering risk** | HIGH | VERY HIGH | MODERATE |
| **Timeline (3-person)** | 14-18 mo | 24-36 mo | 10-14 mo |
| **Prior art density** | Moderate | Low | High |
| **Scoop risk** | Low | Very Low | High |

**Approach A is the Goldilocks choice** — hard enough for a real contribution, feasible enough to ship, uniquely positioned for absence certificates.

---

## Part III: Adversarial Skeptic Attack

### Approach A — Revised Composite: 5.5 (from 7.0)

**FATAL FLAWS:**

1. **F-A1: O(n) merge claim almost certainly false for real code.** The four algebraic properties describe an *idealized* protocol, not OpenSSL's actual implementation with side effects, `OPENSSL_CONF` runtime state, `#ifdef` forests, and callback hooks. Proving a variable is "negotiation-irrelevant" requires solving the analysis problem you claim to be tractable.

2. **F-A2: "Bounded-completeness certificates" are vacuous without externally validated bounds.** k=20, n=5 were chosen *after* seeing which CVEs you wanted to catch. Searching your apartment and certifying you've searched the whole house. "Bounded-completeness theater."

3. **F-A3: KLEE cannot handle OpenSSL's build system.** Perl-generated Makefiles, assembly stubs, platform-specific `#define` chains. Generating correct LLVM bitcode for OpenSSL's negotiation paths is a multi-month effort conspicuously absent from the timeline. #1 cause of death for KLEE-based research.

**SERIOUS RISKS:** Z3 timeout on DY+SMT encoding (60% probability). Slicer too aggressive/conservative (40%). 50K LoC undeliverable at research quality (50%).

**KILL SHOT:** "The paper's central theoretical contribution claims exponential speedup by exploiting four algebraic properties, but these properties hold only for a mathematical idealization — not the C implementations the tool analyzes. The gap between formalism and artifact is precisely the specification-implementation gap the paper claims to close."

### Approach B — Revised Composite: 6.0 (from 7.25)

**FATAL FLAWS:**

1. **F-B1: Custom abstract interpreter for C is multi-year project — not a paper.** Astrée took a team of 10+ researchers a decade. "Mostly from scratch" for a C abstract interpreter means mostly broken.

2. **F-B2: No widening operator preserves useful precision.** Product domain CipherReach × VersionBounds × Phase × TaintMap is non-distributive. Standard product widening loses precision catastrophically. Either fail to terminate or widen to ⊤ after two loop iterations.

3. **F-B3: "10-50x faster" is unfalsifiable marketing.** Compares two hypothetical tools on non-existent benchmarks.

**SERIOUS RISKS:** CEGAR non-convergence (70%). Abstract interpreter unsoundness from C corner cases (50%). CI integration requires sub-10-minute times, unlikely with sufficient precision (40%).

**KILL SHOT:** "Building a correct, sound abstract interpreter for production C is itself a top-venue paper — the authors propose to do this AND invent a novel domain AND implement CEGAR AND produce evaluation, all in one paper. This is three papers stapled together, each individually high-risk."

### Approach C — Revised Composite: 6.0 (from 7.75)

**FATAL FLAWS:**

1. **F-C1: Covering-design "completeness" is a category error.** Covering designs guarantee combinatorial coverage of *input configurations*, not *behavioral coverage*. Downgrade attacks exploit *sequential* adversary actions — temporal property that combinatorial coverage doesn't address. Complete over the wrong domain.

2. **F-C2: Deviation ≠ Vulnerability.** Most cross-library differences are intentional design choices. Deviation detector drowns in true-but-uninteresting differences. Security ranking is either heuristic (unsound) or requires formal adversary model — which is Approach A's hardest component, now demoted to sub-module.

3. **F-C3: Cross-library certificates require compositional verification semantics that don't exist.** Formalizing "OpenSSL client + WolfSSL server" composition is the entire research program of compositional verification.

**SERIOUS RISKS:** False positive flood from deviation detector (50%). "Targeted symex" is load-bearing but under-specified (40%). N² pairwise comparisons don't scale (35%).

**KILL SHOT:** "This approach outsources its hardest problem (formal verification of deviations) to 'targeted symex' — which is Approach A without the merge operator — while claiming novelty from the differential framing, a testing methodology, not a verification methodology."

### Skeptic's Comparative Verdict

| | A | B | C |
|---|:---:|:---:|:---:|
| **Revised Composite** | 5.5 | 6.0 | 6.0 |
| **Worst Fatal Flaw** | O(n) false on real C | Custom C AI multi-year | Completeness over wrong domain |
| **Cause of Death** | KLEE + OpenSSL bitcode | CEGAR non-convergence | False-positive flood |
| **Produces Concrete Attacks?** | Yes (if Z3 cooperates) | Only via CEGAR (fragile) | Only via targeted symex |
| **Handles Cross-Library?** | No | No | Yes (sole advantage) |
| **Standalone Certificates?** | Yes | Yes | No |

**"If forced to pick, A has the clearest path to partial success."**

---

## Part IV: Cross-Expert Challenge Resolution

### Challenge 1: Math Assessor vs Skeptic on T3 depth

**Math Assessor:** T3 is depth 5/10 — valid but overstated domain instantiation.
**Skeptic:** T3 is likely FALSE on real code because algebraic properties don't hold for OpenSSL's implementation.
**Resolution:** Both are partially right. T3 holds for the *abstraction* of negotiation logic, not the raw C code. The gap is bridged by T1 (extraction soundness) + the slicer. If the slicer correctly isolates negotiation-relevant logic, the algebraic properties hold on the extracted model. The risk is in the slicer, not T3.

### Challenge 2: Difficulty Assessor vs Skeptic on Approach B feasibility

**Difficulty:** 9/10 difficulty, 24-36 months minimum.
**Skeptic:** Three PhD theses in a trench coat. Feasibility 3/10.
**Resolution:** Agreement. Approach B is infeasible as a single paper.

### Challenge 3: Skeptic vs Visionary on Approach C novelty

**Visionary:** C has highest composite (7.75), strongest value proposition.
**Skeptic:** C's completeness theorem is over the wrong domain. Most incremental methodology.
**Math Assessor:** C has deepest math (C1 at depth 6/10).
**Difficulty Assessor:** C has highest scoop risk and cannot produce absence certificates.
**Resolution:** C's differential methodology is genuinely valuable but cannot stand alone — it needs A's verification engine for the "targeted verification" step. C without A is a testing tool, not a verification tool.

### Challenge 4: All experts on what should survive

**Consensus elements:**
1. A's protocol-aware merge operator + KLEE-based pipeline is the core engine (only approach that delivers standalone certificates)
2. A's slicer is shared infrastructure across all approaches
3. C's differential methodology adds unique cross-library value but as an *extension*, not replacement
4. B should be abandoned — infeasible for a small team
5. T3 should be honestly framed as "domain instantiation with sharp empirical demonstration" not "fundamental breakthrough"
6. Bounded-completeness bounds (k, n) need external validation, not just CVE-calibrated selection
