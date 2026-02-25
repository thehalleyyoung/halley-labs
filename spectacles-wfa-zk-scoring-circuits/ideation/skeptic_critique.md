# Adversarial Skeptic Critique: Spectacles Architectural Approaches

*Jana Kulkarni — PL Theory & Formal Methods*

**Posture:** I am here to find the problems that will cost you three months of wasted engineering. Every point below is a genuine technical objection that demands a concrete answer, not a hand-wave. I score each approach on my four dimensions at the end.

---

## Prelude: Concerns That Cut Across All Three Approaches

Before attacking each proposal individually, let me flag structural problems shared by all three.

**1. The WFA encoding of NLP metrics is not as clean as claimed.** The problem statement asserts that BLEU, ROUGE, token-F1, etc. "decompose into weighted finite automata over semirings." This is stated as if it's a settled fact, but it conceals serious complications:

- **BLEU's clipped precision** is not a WFA operation. Clipped n-gram counts require `min(count_candidate(ngram), count_reference(ngram))` — a comparison between two separate WFA evaluations. The `min` is the tropical semiring's addition, but you now need the *composition* of a counting-semiring WFA (to get counts) with a tropical-semiring operation (to clip), producing a cross-semiring dependence that the "parameterize over a semiring" story does not accommodate. The problem statement acknowledges this obliquely with "non-automata arithmetic post-processing compiled as separate circuit gadgets," but this means BLEU — the single most important metric in the system — is *not* a WFA. It's a WFA plus arithmetic glue. How much of the formalism actually applies to the glue?

- **Token-level F1** requires computing precision and recall (each a ratio) and then their harmonic mean. Division is a field operation, not a semiring operation. The harmonic mean is not even a rational function of WFA outputs in a natural way. Where exactly does F1 live in the algebraic hierarchy?

- **pass@k** is described as "repeated exact match with counting aggregation," but the actual pass@k formula is `1 - C(n-c, k) / C(n, k)` where `c` is the number of correct samples out of `n`. This involves binomial coefficients — combinatorial functions that are not WFA-computable. You can count correct samples with a WFA, but the pass@k formula is a non-trivial post-processing step.

All three approaches inherit this problem. The clean "everything is a WFA over a semiring" narrative is load-bearing for the entire paper, and it's partially false. The honest statement is: "the *inner loops* of these metrics are WFA-computable; the *aggregation* requires additional circuit gadgets that live outside the WFA formalism and have their own proof obligations."

**2. The 117K–142K LoC estimate is a confession, not a boast.** This is roughly the size of the Rust compiler's type checker. For a research project with (presumably) 1–3 full-time engineers over 6–12 months, this is a guaranteed scope disaster. The "novel LoC" count of 94K–114K means almost nothing can be borrowed. Even at an aggressive 100 lines of debugged, tested code per day, this is 940–1140 engineer-days. With one person, that's 4–5 years. With three, it's 1.5–2 years with zero time for paper writing, debugging conceptual issues, or responding to reviews.

**3. The Lean 4 formalization ambitions are disconnected from Lean 4 reality.** All three proposals assume substantial Lean 4 libraries can be built in research-project timescales. Let me note some facts:
- Mathlib's current automata theory coverage is minimal. There is no WFA library. There is no `KleeneSemiring`. The `Computability.Language` infrastructure handles *unweighted* languages.
- Building a `sorry`-free 12–15K LoC Lean 4 library touching `Matrix`, `Finsupp`, and `Computability.Language` means interfacing with three different corners of Mathlib, each with its own conventions and implicit API assumptions. Expect 2–4 weeks of "why won't this typecheck" per major Mathlib integration point.
- The problem statement's pilot estimate of ~800 LoC for `circuit_sound_algebraic` in the Boolean case is the easy case. The counting semiring case requires reasoning about overflow and field embedding of natural numbers — more subtle. The tropical case (bit decomposition) is a different proof entirely.

---

## Approach A: Coalgebraic WFA Semantics with Functorial Circuit Compilation

### 1. Fatal Flaws

**The coalgebraic framework does not actually buy you anything you can't get more cheaply.** The proposal claims that defining the compiler as a natural transformation between functors gives you a "universal proof template" where new metrics automatically inherit soundness. But what does this universality actually mean? You still need to:
- Define the specific WFA for each metric (metric-specific work).
- Verify that the WFA correctly implements the metric (metric-specific proof).
- Check that the semiring embeds into 𝔽_p (semiring-specific work for Tier 1) or build gadgets (semiring-specific work for Tier 2).

The only thing the coalgebraic machinery saves you is the proof that "if the semiring embeds correctly, then the trace simulation is correct." But that proof is equally achievable with a simple induction on input length over the matrix recurrence `v_{t+1} = v_t · M_{σ_t}`. You do not need Lambek's lemma, coalgebra homomorphisms, or the functor $F_S(X) = S \times X^\Sigma$ to prove this. A direct proof by induction is 50 lines of Lean 4. The coalgebraic version requires building an entire `Coalgebra` library infrastructure (hundreds of lines of typeclass hierarchy, instances, lemmas about final coalgebras) to achieve the same 50-line result.

**Theorem 3 (Syntactic Monoid Contamination Bound) is wrong as stated.** The claim: "the n-gram overlap between two corpora equals the number of elements in the syntactic monoid of their shared prefix automaton that are reachable from both initial states." This is not true. The syntactic monoid of a regular language has size at most $|Q|^{|Q|}$ (the full transformation monoid), and it captures the *algebraic structure* of the language, not the *cardinality* of finite subsets. N-gram overlap is about counting specific strings; syntactic monoid reachability is about state-transformation equivalence classes. These are different mathematical objects. I challenge the authors to write down a formal proof of this theorem; I believe it will fail.

Even if some repaired version of this theorem holds, running PSI on syntactic monoid elements rather than n-grams gains nothing: the monoid can be *larger* than the n-gram set for structured languages, and computing the syntactic monoid of a corpus's n-gram set is itself O(|Q|³) per monoid element — more expensive than just hashing the n-grams.

### 2. Hidden Complexity

**Coinductive proofs in Lean 4 are a research problem, not an engineering task.** The proposal acknowledges this but drastically underestimates it. The mitigation — "reduce coinductive proofs to inductive ones via bounded bisimulation" — is stated in one sentence and contains an entire PhD thesis. The claim that "behavioral equivalence up to depth $|Q|^2$ implies full bisimulation" for WFAs over Noetherian semirings requires a proof that (a) the WFA's state space generates a Noetherian semimodule (not automatic for arbitrary semirings), (b) any pair of states not separated by a word of length ≤ $|Q|^2$ are in the same Myhill-Nerode equivalence class (true for Boolean semirings via pumping, non-trivial for weighted), and (c) the Lean formalization of Noetherian descent terminates on all inputs (requires well-founded recursion on the descending chain condition). Each of (a), (b), (c) is 1–2 months of proof engineering.

**The `Coalgebra` typeclass hierarchy does not exist in Mathlib and designing it right is hard.** You need `Coalgebra F α` to interact correctly with `Functor`, `Monad`, `Category`, etc. Lean 4's typeclass resolution has known performance issues with deep hierarchies. The proposal needs `WFACoalgebra` to be an instance of `Coalgebra` which must compose with `AIRCoalgebra`, and the compilation morphism must be a `CoalgebraHom`. Designing this so it doesn't trigger synthesis timeouts or produce inscrutable error messages is 2–3 months of Lean metaprogramming expertise.

### 3. The "So What?" Test

The coalgebraic framework is *ornamental*, not load-bearing. Strip away the category theory and ask: what does a user of Spectacles actually experience differently because the compiler is a "natural transformation between functors" rather than a "function proven correct by induction"? The STARK proof is identical. The verification time is identical. The certificate is identical. The only difference is the *internal proof structure*, which matters to CAV reviewers and to approximately nobody else.

The extreme-value story — AI safety organizations verifying dangerous-capability benchmarks — is compelling, but it doesn't require coalgebras. It requires the *system to exist and work*. The coalgebraic framing makes the system harder to build, not more useful.

### 4. Comparison Attacks

- **vs. B (KAT):** Approach B achieves specification equivalence via KAT's decision procedure, which has a 30-year track record and a known complexity bound. Approach A achieves it via coinductive bisimulation, which has no efficient implementation in any proof assistant. B is strictly more feasible for the equivalence story.
- **vs. C (Extraction):** Approach C puts more code inside the Lean TCB, giving a stronger end-to-end guarantee. Approach A's coalgebraic elegance doesn't compensate for having a larger unverified Rust codebase.
- **What A sacrifices:** Feasibility. The feasibility score of 5 (the proposal's own score!) is a red flag. A research project with feasibility 5/10 has a coin-flip chance of producing a working artifact.

### 5. Scope Creep Risk

**The `Coalgebra` Mathlib contribution will become its own project.** Building a proper `Coalgebra` library in Lean 4 — with final coalgebras, bisimulation up-to techniques, coinduction principles, and integration with `CategoryTheory` — is a 6-month project *in isolation*. The team will be pulled into making this "Mathlib-ready," responding to Mathlib reviewers' API design feedback, and generalizing beyond what Spectacles needs. This is a classic research trap: the infrastructure becomes the project.

**Syntactic monoid PSI is a dead end that will consume months.** Once someone starts implementing the syntactic monoid computation and discovers the theorem is wrong (see Fatal Flaws), they'll spend weeks trying to repair it, then months trying to make the repaired version efficient, then eventually fall back to standard n-gram PSI anyway.

### 6. The Kill Shot

**The coalgebraic framing is a solution in search of a problem.** The proposal cannot name a single property of the compiled system that is *provable* with coalgebraic semantics but *not provable* with a direct inductive argument. If the coalgebra machinery is removed and replaced with straightforward matrix-induction proofs, the system has identical functionality, identical security guarantees, identical performance, and identical user-facing behavior — but is buildable in half the time. The category theory is there to impress reviewers, not to enable the artifact.

### 7. Salvageable Elements

- **The two-tier compilation architecture** (algebraic embedding for counting/Boolean, gadget-assisted for tropical) is the right design regardless of the proof framework. Keep this.
- **The contamination-integrated evaluation certificate** (single certificate attesting both score correctness and data separation) is a genuinely novel contribution. Keep this — but with standard PSI, not syntactic monoid PSI.
- **The idea of WFA behavioral equivalence as a decidable specification-comparison tool** is strong and should survive. But implement it via weighted Myhill-Nerode minimization + isomorphism checking, not coinductive bisimulation.

### Scores (Kulkarni Dimensions)

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Mathematical Soundness | 4 | Theorem 3 is likely false as stated; coinductive-to-inductive reduction is hand-waved; the coalgebra-to-AIR "natural transformation" degrades to a simulation relation for Tier 2, undermining the framework |
| Load-Bearing Math | 3 | The coalgebraic apparatus is ornamental — identical results achievable via direct induction; syntactic monoid PSI adds complexity without benefit |
| Novelty of Theoretical Contribution | 6 | A coalgebraic verified compiler would be novel, but the novelty is in the framing, not in the theorems (which are known results dressed in new notation) |
| Verification Feasibility | 3 | Coinductive Lean 4 proofs at scale are a research problem; `Coalgebra` typeclass design is months of unbudgeted work; feasibility 5/10 by the proposers' own admission |

---

## Approach B: Algebraic Program Analysis via Kleene Algebra with Tests

### 1. Fatal Flaws

**Weighted KAT (WKAT) completeness does not exist, and the proposal doesn't have a path to it.** The entire approach hinges on Theorem 3: "the WKAT equational theory is complete for evaluation-program equivalence." But WKAT completeness is an *open research problem*, not a known result. Standard KAT completeness (Kozen & Smith 1996) relies critically on the following: (1) tests form a Boolean algebra, (2) the Boolean algebra is freely generated, and (3) the reduction from KAT to language equivalence goes through a construction that exploits Boolean complementation. In WKAT:

- If you keep tests Boolean (Approach (a) in the proposal), you don't have WKAT — you have KAT with weighted transitions. Call it wKAT. But then Theorem 3 must be reproved for wKAT: does the completeness theorem survive when the "program" part carries weights? The answer is *not obvious*. The Kozen-Smith proof constructs a specific automaton from a KAT term and shows language equivalence implies KAT equivalence. With weights, language equivalence becomes formal-power-series equivalence, and the automaton construction must be weight-preserving. This is not the same proof with extra annotations — it's a different proof.

- If you go with Approach (b) (encoding weights into the alphabet), the alphabet $\Sigma \times S$ is infinite when $S$ is infinite (e.g., $\mathbb{N}$ for the counting semiring). KAT over infinite alphabets is undecidable in general (Kozen 2003). You'd need to restrict to bounded weights, which changes the semantics and introduces the question of whether the bounded version is complete for the metrics you actually care about.

**The proposal treats this as the "hardest technical challenge" but does not adequately grapple with the possibility that it's unsolvable within the project timeline.** If WKAT completeness fails, you lose Theorem 3 and the "metric equivalence oracle" story — the headline feature of Approach B — collapses entirely.

**The "symbolic PSI" via KAT Boolean tests is nonsensical.** The claim that "the PSI protocol operates on KAT normal forms rather than raw n-gram sets" and that "two normal forms have the same Boolean-test component if and only if they share the same n-gram vocabulary" is confused. KAT normal forms are algebraic expressions; they are *specifications*, not data. PSI operates on *data sets* (the actual n-grams in a corpus). You cannot run PSI on a KAT expression because the KAT expression denotes a function, not a set of strings. The Boolean test `b_w$ ("does the input contain n-gram $w$?") is a predicate, not a data element. The contamination detection problem is: given two *corpora* (finite sets of strings), compute their n-gram overlap. This is a data problem, not an algebra problem.

### 2. Hidden Complexity

**Matrix Kleene star over 𝔽_p is not straightforward.** The proposal says "over $\mathbb{F}_p$, this sum is finite (since $M^{p^n} = M$ for nilpotent matrices)." But WFA transition matrices are generally *not* nilpotent — they have cycles! For a non-nilpotent matrix over $\mathbb{F}_p$, $M^* = (I - M)^{-1}$ when $I - M$ is invertible, and is undefined otherwise. When is $I - M$ invertible? When $1$ is not an eigenvalue of $M$ over $\mathbb{F}_p$. For a random WFA over $\mathbb{F}_p$, this fails with probability $\approx 1/p$, which is negligible — but the *proof* that it doesn't fail for your specific WFAs requires showing that no WFA produced by the EvalSpec compiler has $1$ as an eigenvalue in $\mathbb{F}_p$. This is a non-trivial spectral constraint that the proposal doesn't mention.

More fundamentally, you don't actually *need* the matrix Kleene star for circuit compilation. The STARK trace is generated step-by-step: you process one input symbol at a time and extend the trace. The Kleene star is needed for *closed-form* evaluation of the WFA on all inputs simultaneously — which is the specification-equivalence story, not the circuit-compilation story. So this complexity falls entirely on the equivalence-checking side.

**The `KATAlgebra` typeclass extending Mathlib's `KleeneAlgebra`.** The proposal says Mathlib has `KleeneAlgebra` as `Finset`-based language operations in `Computability`. This is misleading. Mathlib has `Language` as a type alias for `Set (List α)` with concatenation and Kleene star, but it does not have a `KleeneAlgebra` typeclass with the Kleene algebra axioms (idempotent semiring + star axioms). Mathlib has `StarRing` but that's for C*-algebras — different axioms. Building `KATAlgebra` means building `KleeneAlgebra` first, which means building `IdempotentSemiring` first, and then proving all the axiom interactions. This is 2–3 months of Lean 4 formalization before you can even *state* your main theorems.

### 3. The "So What?" Test

The "metric equivalence oracle" is genuinely valuable — this is the strongest contribution across all three approaches. Being able to mechanically decide "are these two BLEU implementations equivalent?" solves a real problem that has caused actual harm (the SacreBLEU paper exists because of this exact issue).

However, the equivalence oracle *does not require KAT*. It requires WFA minimization + isomorphism checking, which is Schützenberger (1961). The KAT framework is overkill for this: you're building a general-purpose program equivalence framework to solve what is essentially a finite-state equivalence problem. The KAT machinery would buy you something if you needed to reason about *programs that construct WFAs* (e.g., "these two Python functions produce equivalent WFAs") — but that's not what the proposal does. The proposal converts EvalSpec terms to WFAs and then checks WFA equivalence. KAT is an unnecessary middle layer.

### 4. Comparison Attacks

- **vs. A (Coalgebraic):** Approach A's coalgebraic bisimulation achieves the same specification-equivalence result as KAT's decision procedure, but with a better-understood relationship to WFA semantics (bisimulation *is* the standard notion of WFA equivalence). KAT adds an algebraic indirection that doesn't simplify the problem.
- **vs. C (Extraction):** Approach C puts the compiler inside Lean 4, giving compile-time guarantees that B's "prove properties of external Rust code" approach cannot match. If the goal is maximum formal assurance, C dominates B.
- **What B sacrifices:** Conceptual clarity. The KAT framework is a general-purpose algebraic theory of programs. WFAs over semirings are a specific algebraic structure. Using KAT to reason about WFAs is like using a sledgehammer to drive a finishing nail — it works, but it obscures what's actually happening and introduces complexity that doesn't serve the application.

### 5. Scope Creep Risk

**WKAT completeness will become the project.** If the team pursues WKAT completeness seriously, they will spend 6+ months on a pure algebra/logic problem that may not have a solution within the project's constraints. The temptation will be to "just prove this one more lemma" while the actual system (DSL, compiler, STARK integration, PSI, benchmarks) sits unbuilt.

**The `kleene_dec` tactic is a project unto itself.** Building a Lean 4 tactic that decides KAT equivalence via reduction to automata, with `native_decide` for small instances and a verified Antimirov construction for larger ones, is a 2–3 month project requiring deep Lean 4 metaprogramming expertise. The proposal describes it as if it's a straightforward application of known techniques, but *implementing a decision procedure as a Lean tactic* requires generating *proof terms*, not just yes/no answers. The tactic must construct a bisimulation witness (for equivalence) or a distinguishing word (for non-equivalence) as a Lean term that typechecks. This is much harder than implementing the decision procedure in Rust.

### 6. The Kill Shot

**WKAT completeness is an open problem that may not be solvable in the project timeline, and without it, Approach B's headline contribution (the equivalence oracle with completeness guarantees) is no better than Approach A or a direct WFA-equivalence check.** The proposal bets the entire approach on a theorem (Theorem 3, WKAT completeness) that no one has proved. If this theorem turns out to require non-trivial new mathematics — or worse, turns out to be false for the semirings of interest — the KAT framing provides no advantage over direct WFA methods, while having consumed months of formalization effort on the KAT infrastructure.

### 7. Salvageable Elements

- **The metric equivalence oracle idea** is the single best contribution across all three approaches. "Given two metric implementations, decide equivalence or produce a distinguishing input" is genuinely useful, immediately deployable, and solves a real problem. Keep this — but implement it via weighted automata minimization, not KAT.
- **The `kleene_dec` tactic** for equational Kleene algebra goals is a valuable Mathlib contribution independent of Spectacles. If someone builds it, it will be useful for program verification, network verification, and other KAT applications. But it should be scoped as a standalone contribution, not as a critical-path dependency for Spectacles.
- **The observation that Boolean tests in KAT correspond naturally to token-matching predicates in NLP metrics** is a nice conceptual insight that could appear in a related-work or motivation section, even if the full KAT machinery isn't used.

### Scores (Kulkarni Dimensions)

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Mathematical Soundness | 5 | Theorems 1 and 2 are plausible but unproved; Theorem 3 (WKAT completeness) is an open problem; symbolic PSI is confused |
| Load-Bearing Math | 5 | KAT machinery is overkill — WFA equivalence is achievable without it; the KAT framework adds a layer of indirection that doesn't simplify the core problem |
| Novelty of Theoretical Contribution | 7 | WKAT decidability (if achieved) would be novel; KAT-ZK connection is new; but novelty depends on solving an open problem |
| Verification Feasibility | 5 | KAT has better existing theory than coalgebras, but `KATAlgebra` must be built from scratch in Lean 4; WKAT completeness proof is high-risk |

---

## Approach C: Stratified Verification with Extraction — Prove in Lean, Extract to Rust, Certify via STARK

### 1. Fatal Flaws

**There is no Lean 4-to-Rust extraction pipeline.** The proposal acknowledges this ("Lean 4 does not have a production-quality extraction pipeline to Rust") and then proceeds as if this is a minor inconvenience rather than a fundamental obstacle. Let me be precise:

- `lean4export` exports Lean's kernel terms as an interchange format. It does not produce Rust.
- There is no maintained, tested Lean 4-to-Rust code generator. The "community efforts" mentioned are experimental prototypes that handle a fraction of Lean 4's type system.
- Even Lean 4's own C code generation (the compiler backend) is not verified. The gap between "Lean term" and "executable binary" is the *entire* verified-extraction problem, which CakeML spent a decade on.

The proposal's fallback — "translation validation via differential testing" — means this approach degenerates into: "write the specification in Lean 4, write the implementation in Rust by hand, and test that they agree." This is *exactly what Approaches A and B do*, but with the additional cost of writing the entire WFA engine and circuit synthesizer in Lean 4 (for the specification side) on top of writing it in Rust (for the execution side). You've doubled the implementation effort for a "correct by construction" narrative that is, in fact, "correct by testing."

**40K+ LoC of algorithmic Lean 4 code is unprecedented and likely infeasible.** The largest verified algorithmic systems in Lean 4 are on the order of 5–10K LoC (e.g., parts of the Lean compiler itself, selected Mathlib algorithms). Quadrupling that while simultaneously building the system you're verifying is a project-management impossibility. For comparison:
- CakeML's verified compiler is ~30K LoC of HOL4, built over 8+ years by a team of 5–10 researchers.
- CompCert is ~100K LoC of Coq, built over 15+ years by a team at Inria.
- Spectacles-C proposes 40K LoC of verified Lean 4 in a research-project timeline.

**Theorem 1 (Extraction Preservation) is explicitly NOT proved.** The proposal states this openly: "This theorem is NOT proved in Lean 4 — it is the correctness claim about the extraction pipeline itself, which is part of the TCB." So the headline claim — "the compiler *is* the proof" and "extracted code is correct by construction" — is false. The code is correct *if the extraction is correct*, and the extraction is not verified. You've moved the trust assumption from "the Rust code matches the Lean spec" (testable) to "the extraction pipeline preserves semantics" (also testable, but you've added a complex intermediate step).

### 2. Hidden Complexity

**Performance of Lean 4 code is a non-starter for STARK trace generation.** The proposal's own analysis identifies this as the hardest challenge and proposes three mitigations. Let me evaluate each:

- **(a) `@[extern]` for field arithmetic:** This means the field arithmetic in Lean 4 is axiomatized, not proved. You've introduced axioms into your Lean 4 development that assert "this Rust function correctly implements Goldilocks field multiplication." If those axioms are wrong (e.g., the Rust function uses Montgomery form and the Lean axiom assumes standard representation), your entire soundness proof is vacuous. Every `@[extern]` is a hole in your verification. For performance-critical code like field arithmetic (which dominates STARK trace generation), you'll end up with `@[extern]` on 80% of the hot path, meaning 80% of the computation is trusted, not verified.

- **(b) Sparse matrix via `Finsupp`:** `Finsupp` in Mathlib is *not* designed for high-performance computation. It's a mathematical abstraction (functions with finite support) implemented as a sorted association list. Iterating over a `Finsupp` involves linked-list traversal with reference counting on each node. For a WFA with 500 states and 100 non-zero entries per row, each matrix-vector multiply involves 50,000 `Finsupp.sum` calls, each of which traverses a linked list. This will be 100–1000× slower than a flat array in Rust.

- **(c) Fallback to hand-written Rust with differential validation:** This is the honest answer, and it's what will actually happen. But if you fall back to this, you're doing the same thing as Approaches A and B, with the additional sunk cost of having written 40K LoC of Lean 4 that you're now not executing.

**Writing seven NLP metrics in Lean 4 requires string processing infrastructure that doesn't exist.** BLEU needs tokenization, n-gram extraction, and counting. ROUGE-L needs longest common subsequence. Regex match needs a regex engine. Token-level F1 needs set intersection cardinality. None of these have Lean 4 library support. You'll spend months building basic string/list processing utilities before you can even *state* the metric specifications, let alone prove properties about them.

### 3. The "So What?" Test

The "verified metric standard library" story is aspirational but disconnected from how standards bodies actually operate. NIST does not publish Lean 4 packages. ISO/IEC JTC 1/SC 42 does not use proof assistants. The standards-body audience is a fantasy. The actual audience for a verified metric library is the formal methods research community — which is real, but much smaller than the regulatory/safety audience targeted by Approaches A and B.

More fundamentally, the value proposition of Approach C over Approaches A and B reduces to: "the extraction gives you a tighter trust chain." But since the extraction is unverified (Theorem 1 is not proved), the actual trust chain is: Lean 4 kernel → Lean 4 compiler → unverified extraction → Rust compiler → binary. Compare with Approach B: Lean 4 kernel → Lean 4 proofs about Rust code → Rust compiler → binary. The TCB is *larger* in Approach C (it includes the extraction pipeline) than in B (which doesn't need extraction).

### 4. Comparison Attacks

- **vs. A (Coalgebraic):** Approach A at least uses an established verification strategy (prove properties of external code). Approach C's extraction strategy requires infrastructure that doesn't exist.
- **vs. B (KAT):** Approach B has a feasibility score of 7 vs. C's 6 (by the proposers' own scores). B also has a stronger theoretical contribution (KAT-ZK connection) and a cleaner implementation path (Rust codebase with Lean proofs about it).
- **What C sacrifices:** Everything it claims to provide. The "correct by construction" narrative fails because extraction is unverified. The "single source of truth" narrative fails because you'll need `@[extern]` for performance. The "CakeML for ZK" narrative fails because CakeML actually verified its extraction; C does not.

### 5. Scope Creep Risk

**The Lean 4-to-Rust extraction pipeline will become the project.** If someone starts building or improving the extraction pipeline (because the existing one doesn't work), they will spend 6+ months on a program-transformation problem that has nothing to do with WFAs, ZK proofs, or NLP evaluation. The extraction pipeline is a prerequisite, not a contribution.

**Lean 4 systems programming will produce a never-ending stream of performance issues.** Every algorithmic optimization (hash maps for state caching, bit-manipulation for Goldilocks arithmetic, SIMD for NTT) will require either an `@[extern]` escape hatch or a month of Lean 4 performance engineering. The team will oscillate between "make it correct in Lean" and "make it fast in Rust" without converging.

**The "verified WFA minimization" (Theorem 3) is a significant standalone project.** Weighted Hopcroft minimization requires partition refinement with weight-aware splitting. Formalizing this in Lean 4 requires proving termination of the refinement loop, correctness of the splitting criterion (two states are inequivalent iff they have different weights or different successors under some input), and minimality of the result (no further splits are possible). This is 2–3 months of proof engineering. It's a valuable contribution, but it's not on the critical path for the ZK pipeline — you can compile unminimized WFAs and just pay a performance cost.

### 6. The Kill Shot

**Approach C's entire value proposition rests on verified extraction from Lean 4 to Rust, which does not exist and cannot be built within a research-project timeline. Without it, Approach C is strictly worse than Approach B: same Lean 4 proof obligations, same Rust implementation, but with the additional cost of duplicating the entire system in Lean 4 as a "specification" that is never executed. You pay double the implementation cost for zero additional assurance.**

### 7. Salvageable Elements

- **The stratified trust model** (explicitly decomposing soundness into proved/validated/assumed layers) is methodologically excellent and should be adopted by whichever approach is chosen. Every verified system has trust assumptions; making them explicit is a mark of intellectual honesty.
- **Verified WFA minimization in Lean 4** is a genuine Mathlib contribution that serves the whole project (smaller WFAs → smaller circuits → faster proofs). It should be built regardless of the overall approach.
- **The `@[extern]` pattern for field arithmetic** is the right engineering choice for Lean 4 performance. Axiomatize the field operations with explicit trust assumptions, implement them in Rust, and test thoroughly. Just don't pretend this is "correct by construction."

### Scores (Kulkarni Dimensions)

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Mathematical Soundness | 6 | Theorems 2 and 3 are sound; Theorem 1 (extraction) is explicitly unproved, which is honest but fatal to the narrative |
| Load-Bearing Math | 4 | The extraction is the load-bearing component and it's unverified; the Lean 4 specifications are valuable but could equally serve Approaches A or B |
| Novelty of Theoretical Contribution | 5 | "CakeML for ZK" would be novel if extraction were verified; without that, it's "Lean 4 specifications + Rust implementation + differential testing," which is standard practice |
| Verification Feasibility | 3 | 40K LoC of algorithmic Lean 4 is likely infeasible; extraction pipeline doesn't exist; performance will force fallback to unverified Rust on critical paths |

---

## Cross-Cutting Verdict

### Which approach should die?

**Approach A (Coalgebraic) should be abandoned.** The coalgebraic framework is ornamental mathematics — it does not enable any property that isn't achievable with direct proofs. The coinductive Lean 4 obligations are high-risk with no fallback. The syntactic monoid PSI is likely unsound. The proposers themselves rate feasibility at 5/10. If I have to bet engineering months, I don't bet on a coin flip.

### Which approach should be restructured?

**Approach C (Extraction) should be restructured into a supporting role.** The Lean 4 specifications and the stratified trust model are valuable, but the extraction story should be dropped. Instead, use Lean 4 as a *specification and proof environment* (12–15K LoC as in the problem statement), implement in Rust (as in Approach B), and validate across the boundary with differential testing. The verified WFA minimization can still be built in Lean 4 and is a valuable contribution.

### Which approach should lead?

**Approach B (KAT), but heavily modified.** Drop the WKAT completeness ambition — it's an open problem and not needed for the system. Drop the "symbolic PSI" — it's confused. Keep the `KleeneSemiring` typeclass (which is the foundation all approaches need anyway). Keep the metric equivalence oracle, but implement it via weighted Myhill-Nerode minimization + isomorphism, not via KAT reduction. Use the two-tier compilation architecture from the problem statement. Adopt the stratified trust model from Approach C.

The result: a pragmatic system with a `KleeneSemiring` typeclass, verified WFA-to-AIR compilation (by induction, not by coinduction or KAT), a WFA equivalence decision procedure (by minimization, not by bisimulation or KAT reduction), standard PSI for contamination, and honest trust boundaries.

### Final Risk Assessment

The deepest risk across all approaches is not technical but *scopal*. This project tries to be:
1. A DSL compiler (EvalSpec → WFA)
2. A verified compiler (WFA → STARK AIR, with Lean 4 proofs)
3. A ZK proof system (STARK prover/verifier)
4. A cryptographic protocol (PSI for contamination)
5. A Lean 4 library contribution (KleeneSemiring, WFA library, tactics)
6. A TLA+ specification
7. An evaluation of seven NLP metrics

Any *two* of these is a strong paper. All *seven* is a 142K-LoC engineering marathon that will produce a system where everything works but nothing is polished enough for a best-paper award. The team must ruthlessly cut scope. My recommendation: items 1, 2, 5, and 7, with items 3 and 4 using existing libraries (Winterfell for STARK, a standard PSI library for contamination). Item 6 (TLA+) can be dropped entirely — the protocol is simple enough that TLA+ model checking is nice-to-have, not essential.

---

*— Jana Kulkarni, adversarial review complete. I expect concrete responses to the kill shots before any implementation begins.*
