# Spectacles: Three Competing Architectural Approaches

*Domain Visionary Analysis — Automata Theory × Zero-Knowledge Proofs × Lean 4 Formalization*

---

## Approach A: Coalgebraic WFA Semantics with Functorial Circuit Compilation

### The Architecture

This approach takes the category-theoretic view seriously: weighted finite automata are coalgebras for the functor $F_S(X) = S \times X^\Sigma$ on the category of $S$-semimodules, where $S$ is the weight semiring and $\Sigma$ the input alphabet. Rather than treating WFA-to-circuit compilation as ad hoc engineering, we define it as a *natural transformation* between two functors: the WFA-acceptance functor (mapping states to formal power series) and a circuit-semantics functor (mapping AIR column assignments to field elements). The compiler is correct if and only if this natural transformation commutes with the observation maps of both coalgebras.

The Lean 4 formalization builds a `Coalgebra` typeclass hierarchy: `Coalgebra F α` for general coalgebras, `WFACoalgebra S Σ α` for weighted automata coalgebras, and `AIRCoalgebra p α` for STARK AIR constraint coalgebras over a prime field $\mathbb{F}_p$. The compilation theorem states that the compiler morphism is a coalgebra homomorphism, which by Lambek's lemma preserves all observable behaviors — i.e., the circuit computes the same formal power series as the WFA.

The PSI contamination layer operates on the *syntactic monoid* of the WFA: since the syntactic monoid $M(L)$ of a regular language $L$ is the transition monoid of its minimal DFA, and since n-gram membership is a regular property, we can represent contamination thresholds as properties of the syntactic monoid quotient. This gives contamination detection algebraic structure that integrates cleanly with the WFA pipeline rather than being bolted on.

### 1. Extreme Value

**Who needs this:** Formal methods teams at AI safety organizations (e.g., METR, Apollo Research, ARC Evals) who evaluate frontier models on capability benchmarks where contamination directly affects safety-relevant conclusions. If GPT-5 scores 95% on a dangerous-capabilities benchmark but 30% of test items leaked into training, the safety conclusion is invalidated. Today, no one can cryptographically distinguish "genuinely capable" from "memorized the test." Spectacles-A would let a safety auditor verify both the score and the contamination bound from a single certificate, with the verification itself taking under 100ms.

**What becomes possible:** A *contamination-aware safety evaluation registry* — a public ledger where each capability evaluation carries a machine-checkable certificate. Regulators under the EU AI Act could mandate certificate submission. More importantly, the coalgebraic framework makes the system *extensible*: new metrics are new coalgebras, and the compilation theorem applies generically. A metric author writes a WFA specification, and the compiler produces a correct circuit without metric-specific proof work.

### 2. Genuine Difficulty

The coalgebraic framing introduces three genuinely hard subproblems:

**Coalgebra-to-AIR trace layout.** A WFA coalgebra has observation structure $(S \times (-)^\Sigma)$ but AIR traces are flat column vectors over $\mathbb{F}_p$. The compiler must *flatten* the coalgebraic structure into a linear trace while preserving the transition invariant. For the tropical semiring, this is especially painful: the $\min$ operation has no field counterpart, so the "natural transformation" degrades to a simulation relation proved via bit-decomposition gadgets. The two-tier structure (algebraic embedding for counting/Boolean, gadget-assisted for tropical) means the coalgebraic story is only fully clean for Tier 1 semirings.

**Coinductive proof obligations.** The behavioral equivalence of WFA coalgebras is defined coinductively (two states are bisimilar if their observations agree and their successors are bisimilar). Lean 4's `Cofix` is notoriously difficult to work with for non-trivial coinductive types. Gabriel Ebner's own work on Lean's kernel has documented the challenges of productive corecursion. We need coinductive proofs that the circuit coalgebra simulates the WFA coalgebra at every step, which requires careful management of guardedness conditions.

**Syntactic monoid PSI.** Computing the syntactic monoid of a WFA over a non-Boolean semiring requires weighted Myhill-Nerode theory (Berstel-Reutenauer), where the equivalence classes are defined by matrix rank conditions. The PSI protocol must operate on representations of these equivalence classes without revealing the underlying automaton structure — a non-trivial privacy requirement.

### 3. Load-Bearing Math

**Theorem 1 (Coalgebraic Compilation Soundness).** Let $\mathcal{A} = (Q, \Sigma, S, \delta, \lambda, \rho)$ be a WFA coalgebra and $\mathcal{C}$ its compiled AIR coalgebra. There exists a coalgebra homomorphism $h: \mathcal{A} \to \mathcal{C}$ such that for all inputs $w \in \Sigma^*$, $\text{weight}_{\mathcal{A}}(w) = \iota^{-1}(\text{trace}_{\mathcal{C}}(w))$ where $\iota: S \hookrightarrow \mathbb{F}_p$ is the semiring embedding (Tier 1) or the gadget-mediated encoding (Tier 2). This theorem is proved in Lean 4, parameterized over a `CompilableSemiring` typeclass that axiomatizes the embedding/encoding requirements.

**Theorem 2 (Coalgebraic Bisimulation Decidability).** For WFAs over a commutative Noetherian semiring $S$ with decidable equality, behavioral equivalence (coinductive bisimulation) is decidable and coincides with language equivalence (the induced formal power series are equal). Formalized in Lean 4 as a decision procedure that either constructs a bisimulation relation or produces a distinguishing word.

**Theorem 3 (Syntactic Monoid Contamination Bound).** The n-gram overlap between two corpora, expressed as the cardinality of the intersection of their n-gram sets, equals the number of elements in the syntactic monoid of their shared prefix automaton that are reachable from both initial states. This algebraic characterization enables PSI over monoid elements rather than raw n-grams, reducing communication when prefix sharing is high.

**Why these are load-bearing:** Theorem 1 is the soundness guarantee — without it, the STARK proof means nothing. Theorem 2 is the specification-equivalence guarantee — without it, two "BLEU" implementations cannot be proved identical. Theorem 3 connects contamination detection to the automata-theoretic framework — without it, PSI is just bolted-on crypto with no algebraic integration.

### 4. Best-Paper Potential

This approach would be the first *coalgebraic verified compiler* — CompCert and CakeML use operational/denotational semantics; Spectacles-A uses coalgebraic semantics. This is a genuine novelty for CAV. The coalgebraic framing provides a universal proof template: any new metric that can be expressed as a coalgebra for a supported functor automatically inherits the compilation soundness theorem. For Gabriel Ebner and the Lean/Mathlib community, this would be the first substantial `Coalgebra` library in Lean 4, filling a gap that Mathlib contributors have discussed but no one has built. The combination of coalgebraic semantics + ZK circuits + mechanized proofs is unprecedented and would generate significant cross-community interest (LICS × CAV × crypto).

### 5. Hardest Technical Challenge

**Coinductive proofs in Lean 4 at scale.** Lean 4's kernel handles coinductive types, but the proof engineering for coinductive bisimulation proofs over WFAs with 100–1000 states is uncharted territory. The guardedness checker may reject natural proof terms; the automation (`cofix` tactic) may produce terms that are slow to type-check.

**Mitigation:** Reduce coinductive proofs to inductive ones via *bounded bisimulation*: prove that for WFAs over a Noetherian semiring, behavioral equivalence up to depth $|Q|^2$ implies full bisimulation. This converts the coinductive obligation to a finite computation, amenable to `native_decide`. Gabriel Ebner's `omega` tactic for natural-number arithmetic provides a template for how to build Lean tactics that discharge obligations via certified decision procedures.

### 6. Scores

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Value | 9 | Addresses both trust crises with algebraically integrated architecture |
| Difficulty | 9 | Coalgebraic Lean 4 proofs are genuinely hard; coinductive reasoning at scale is unsolved |
| Potential | 9 | First coalgebraic verified compiler; novel for CAV, LICS, and Lean communities simultaneously |
| Feasibility | 5 | Coinductive Lean 4 proofs may hit kernel limitations; syntactic monoid PSI adds complexity |

---

## Approach B: Algebraic Program Analysis via Kleene Algebra with Tests

### The Architecture

This approach reframes the problem using *Kleene algebra with tests* (KAT), the algebraic framework developed by Kozen that unifies program reasoning and automata theory. Instead of treating WFAs and circuits as separate objects connected by a compiler, we embed both into a single KAT: the *evaluation KAT* $\mathcal{K}_{\text{eval}}$. An EvalSpec term denotes a KAT expression. A WFA is a matrix representation of a KAT element. An AIR circuit is another matrix representation over $\mathbb{F}_p$. The compiler correctness theorem reduces to: two matrix representations of the same KAT element produce identical formal power series.

The key insight is that KAT has a *complete equational theory* with respect to relational models (Kozen & Smith 1996), meaning KAT equivalence is decidable and captures exactly the input-output behavior of evaluation programs. This gives us specification equivalence "for free" — two EvalSpec terms compute the same metric if and only if their KAT denotations are equal, decidable via the KAT decision procedure.

The Lean 4 formalization targets a `KATAlgebra` typeclass extending Mathlib's existing `KleeneAlgebra` (which exists only as `Finset`-based language operations in `Computability`). The `KATAlgebra` adds Boolean tests, the while-program encoding, and the completeness theorem. The WFA library is built as `KATAlgebra (Matrix (Fin n) (Fin n) S)` — matrices over a semiring form a Kleene algebra, and the WFA transition structure is exactly matrix Kleene-star iteration.

For contamination detection, this approach uses *symbolic PSI*: the n-gram membership test is a KAT expression (a Boolean test in the algebra), and the PSI protocol operates on KAT normal forms rather than raw n-gram sets. Two normal forms have the same Boolean-test component if and only if they share the same n-gram vocabulary, enabling contamination checks at the algebraic level.

### 1. Extreme Value

**Who needs this:** Benchmark platform operators (Hugging Face, EleutherAI, Papers With Code) who maintain dozens of metrics and need to answer: "Is this new metric implementation equivalent to the old one?" Today, when a BLEU implementation changes (and this happens regularly — SacreBLEU exists precisely because BLEU implementations diverge), the only recourse is manual inspection or statistical comparison on test corpora. Neither provides a guarantee.

**What becomes possible:** A *metric equivalence oracle*. A benchmark operator submits two EvalSpec terms (e.g., "SacreBLEU v2.3.1" and "SacreBLEU v2.4.0") and receives either a machine-checked equivalence proof or a concrete distinguishing input. This is not a test — it is a decision procedure with completeness guarantees. Combined with the ZK scoring pipeline, this means benchmark results are *reproducible by definition*: any implementation that passes the equivalence check produces identical scores, and the STARK proof attests the specific implementation used.

### 2. Genuine Difficulty

**KAT-to-weighted-KAT extension.** Standard KAT operates over Boolean tests and relation-algebraic programs. We need *weighted KAT* (WKAT) where programs carry semiring weights. This is not a trivial extension: the completeness theorem for KAT relies on the Boolean algebra structure of tests, and replacing Boolean with semiring weights breaks the proof. We need a new completeness result for WKAT, or a careful reduction from WKAT equivalence to KAT equivalence via encoding weights as test sequences.

**Matrix Kleene star in $\mathbb{F}_p$.** The Kleene star of a matrix $M$ over a semiring is $M^* = \sum_{k=0}^{\infty} M^k$. Over $\mathbb{F}_p$, this sum is finite (since $M^{p^n} = M$ for nilpotent matrices), but computing it requires matrix inversion techniques over finite fields. For the tropical semiring, matrix Kleene star is the all-pairs shortest path (Floyd-Warshall), which again has no direct field representation.

**KAT decision procedure performance.** The KAT decision procedure (reducing KAT terms to automata and checking language equivalence) has worst-case exponential complexity. For the WFA sizes in our problem (≤1100 states), the decision procedure must complete in seconds on a laptop. This requires careful implementation of the Antimirov partial-derivative construction and on-the-fly determinization with early termination.

### 3. Load-Bearing Math

**Theorem 1 (Weighted KAT Soundness).** For semirings $S$ embeddable into $\mathbb{F}_p$, the WKAT denotation $\llbracket e \rrbracket_{\text{WFA}}$ of an EvalSpec term $e$ and its circuit denotation $\llbracket e \rrbracket_{\text{AIR}}$ satisfy $\llbracket e \rrbracket_{\text{WFA}} = \iota^{-1} \circ \llbracket e \rrbracket_{\text{AIR}}$. Proved by induction on the KAT term structure, with each case handled by the semiring homomorphism properties of $\iota$.

**Theorem 2 (WKAT Decidability).** Equivalence of WKAT expressions over a commutative semiring $S$ with decidable equality is decidable. The decision procedure: (1) convert WKAT expressions to WFAs via a weighted Antimirov construction, (2) minimize via weighted Myhill-Nerode, (3) check isomorphism of minimal realizations. Formalized in Lean 4 with a `Decidable` instance for `WKATEquiv`.

**Theorem 3 (KAT Completeness for Evaluation Programs).** Every scoring function expressible in EvalSpec has a WKAT denotation, and the WKAT equational theory is complete for evaluation-program equivalence: $e_1 \equiv e_2$ in WKAT if and only if $\llbracket e_1 \rrbracket = \llbracket e_2 \rrbracket$ as formal power series. This is the critical theorem that makes specification equivalence a *decision* rather than a *heuristic*.

**Why these are load-bearing:** Theorem 1 is the compilation bridge. Theorem 2 is the specification oracle. Theorem 3 closes the gap between syntactic KAT equivalence and semantic scoring-function equivalence — without it, the decision procedure might miss semantically equivalent but syntactically different metrics.

### 4. Best-Paper Potential

KAT has been a staple of the program-verification community since Kozen's 1997 paper, but it has never been connected to zero-knowledge proof systems. This approach shows that KAT's equational theory provides *exactly* the right abstraction for verified ZK compilation of structured computations: the equational theory gives specification equivalence, the matrix representation gives WFA compilation, and the relational semantics gives circuit soundness. For CAV, this is a natural extension of a well-understood framework (KAT) to a high-impact application (ZK-verified AI evaluation). The Lean 4 formalization of WKAT decidability would be a significant Mathlib contribution — Kozen himself has noted the lack of mechanized KAT formalizations.

For Gabriel Ebner specifically, this approach plays to his interests in *proof automation*: the `kleene_dec` tactic, which decides KAT equivalence via reduction to automata problems, is exactly the kind of certified decision procedure that Lean 4 excels at. The tactic would use `native_decide` for small instances and a verified Antimirov construction for larger ones, following the pattern of `omega` and `norm_num`.

### 5. Hardest Technical Challenge

**Extending KAT completeness to weighted settings.** The standard KAT completeness theorem (Kozen & Smith 1996) crucially uses the Boolean structure of tests to reduce KAT equivalence to language equivalence of automata. In WKAT, tests carry weights, and the reduction breaks. Two possible approaches:

**Approach (a):** Restrict to *Boolean-tested* WKAT where tests remain Boolean but program transitions carry weights. This preserves the KAT completeness machinery while adding semiring weights. The restriction is natural for evaluation: tests are "does this token match?" (Boolean), while weights are "how much does this match contribute?" (semiring).

**Approach (b):** Prove a new completeness theorem for full WKAT by encoding weights into the alphabet (each weighted transition $q \xrightarrow{a/s} q'$ becomes an unweighted transition $q \xrightarrow{(a,s)} q'$ over the product alphabet $\Sigma \times S$) and reducing WKAT equivalence to standard KAT equivalence over the enlarged alphabet. This is mathematically clean but may blow up the alphabet size for large semirings.

**Mitigation:** Start with approach (a) for the initial formalization, as it covers all seven target metrics (all tests are Boolean token comparisons). Pursue approach (b) as a theoretical extension with explicit `sorry` markers if needed.

### 6. Scores

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Value | 8 | Specification equivalence is the killer feature; contamination integration is less algebraically elegant |
| Difficulty | 7 | KAT is well-understood; the weighted extension is the main challenge |
| Potential | 8 | First KAT-ZK connection; strong CAV fit; but less novel than coalgebraic approach |
| Feasibility | 7 | KAT machinery is more mature than coalgebraic; weighted extension is tractable via restriction |

---

## Approach C: Stratified Verification with Extraction — Prove in Lean, Extract to Rust, Certify via STARK

### The Architecture

This approach inverts the verification strategy: instead of writing a Rust compiler and separately proving its properties in Lean 4, we write the *entire WFA engine and circuit synthesizer in Lean 4* and extract verified Rust code via a Lean-to-Rust extraction pipeline. The extracted code is correct by construction — no separate soundness proof is needed because the code *is* the proof.

The architecture has three strata:

**Stratum 1: Specification (Lean 4, pure).** A `WFA S Σ` type with formal-power-series semantics, a `KleeneSemiring` typeclass, and an `EvalSpec` inductive type whose denotational semantics maps to `WFA`. All seven metrics are defined as Lean 4 functions `EvalSpec → WFA S Σ`. This stratum is pure mathematics — no side effects, no IO.

**Stratum 2: Compilation (Lean 4, verified).** A function `compile : WFA S Σ → AIRConstraintSystem 𝔽_p` with a proof `compile_sound : ∀ w, wfa.eval w = (compile wfa).eval w` attached. The compilation function is total, structurally recursive, and extracts to efficient Rust. The two-tier structure (algebraic vs. gadget-assisted) is encoded as a typeclass dispatch: `EmbeddableSemiring S 𝔽_p` triggers the algebraic path; `GadgetSemiring S 𝔽_p` triggers the gadget path.

**Stratum 3: Execution (Extracted Rust + STARK prover).** The extracted Rust code handles WFA construction, AIR trace generation, and constraint evaluation. The STARK prover (Winterfell, used as an unverified library) produces the proof. The trusted computing base is: Lean 4 kernel + extraction pipeline + Winterfell + Rust compiler. The Lean proofs guarantee that *if* the extracted code runs and *if* Winterfell is sound, *then* the STARK proof attests the correct WFA weight.

For contamination detection, the PSI protocol is implemented in Rust (not extracted from Lean) but its interface is specified in Lean 4 as a pre/post-condition contract. The composition of the evaluation certificate and the contamination attestation is proved in Lean 4 as a theorem about sequential composition of simulation-secure protocols.

### 1. Extreme Value

**Who needs this:** Standards bodies (NIST, ISO/IEC JTC 1/SC 42) drafting AI evaluation standards who need *reference implementations* that are simultaneously (a) mathematically precise specifications, (b) executable code, and (c) machine-checkably correct. Today, a standard like ISO/IEC 24029 (AI robustness) specifies metrics in English prose, and every implementation is an interpretation. Spectacles-C provides reference implementations where the specification *is* the implementation — verified extraction ensures they compute exactly what the mathematics says.

**What becomes possible:** A *verified metric standard library*. NIST publishes a Lean 4 package; any evaluation framework that uses the extracted code is *provably* computing the standardized metric. Disputes about "which BLEU" become impossible because the metric is defined by its Lean 4 specification, and equivalence with any other implementation is decidable. The ZK layer then lets a model provider prove compliance with the standard without revealing proprietary data.

### 2. Genuine Difficulty

**Lean-to-Rust extraction fidelity.** Lean 4 does not have a production-quality extraction pipeline to Rust. The existing `lean4export` targets C, and community efforts for Rust extraction are experimental. We need either: (a) a verified extraction from Lean 4 to Rust (extremely ambitious — CakeML spent years on this for ML-to-assembly), or (b) a *translation validation* approach where the extracted Rust code is tested against the Lean specification via differential testing and the extraction itself is part of the trusted computing base.

**Performance of extracted code.** Lean 4's code generator produces C code optimized for Lean's reference-counted runtime. Direct extraction to Rust requires mapping Lean's memory model (reference counting, arena allocation) to Rust's ownership model. Naive extraction produces code that is 10-100× slower than hand-written Rust due to excessive cloning and indirection. For STARK trace generation (the performance-critical path), this may push proof times beyond the laptop-CPU budget.

**Lean 4 as a programming language at scale.** Writing 40K+ LoC of algorithmic code in Lean 4 (not just proofs) requires Lean 4 to function as a serious programming language. While this is a design goal of Lean 4, the ecosystem for string processing, I/O, and data structures is immature compared to Rust. The WFA engine needs efficient sparse matrix operations, hash maps for state caching, and bit-manipulation for Goldilocks field arithmetic — all of which have optimized Rust crates but limited Lean 4 support.

### 3. Load-Bearing Math

**Theorem 1 (Extraction Preservation).** If `compile` is a total Lean 4 function with proof `compile_sound`, and `extract : LeanExpr → RustAST` is the extraction function, then `extract(compile)` computes the same mathematical function as `compile` on all inputs. *This theorem is NOT proved in Lean 4* — it is the correctness claim about the extraction pipeline itself, which is part of the TCB. Instead, we provide a *translation validation* procedure: for each extracted function, generate 100K random inputs and verify input-output agreement between the Lean evaluator and the Rust binary.

**Theorem 2 (Stratified Soundness Composition).** The end-to-end soundness decomposes as: (a) EvalSpec-to-WFA semantic preservation (`eval_equiv_wfa`, proved in Lean 4), (b) WFA-to-AIR compilation soundness (`compile_sound`, proved in Lean 4), (c) AIR-to-STARK proof soundness (assumed from Winterfell's security analysis), (d) extraction fidelity (validated, not proved). The composition is a chain of implications, each with explicit trust assumptions.

**Theorem 3 (Verified WFA Minimization).** A Lean 4-verified implementation of weighted Hopcroft minimization: given a WFA $\mathcal{A}$ over a commutative semiring, produce a minimal WFA $\mathcal{A}_{\min}$ with $|\mathcal{A}_{\min}| \leq |\mathcal{A}|$ and $\text{eval}(\mathcal{A}) = \text{eval}(\mathcal{A}_{\min})$. This is load-bearing for circuit size: minimizing the WFA before compilation reduces AIR trace width from $2|Q|$ to $2|Q_{\min}|$, which can be a 5-10× reduction for metrics with redundant states (e.g., BLEU n-gram matching with shared prefixes).

**Why these are load-bearing:** Theorem 2 makes the trust assumptions explicit and compositional — reviewers can evaluate each stratum independently. Theorem 3 directly impacts performance feasibility: without minimization, BLEU-4 WFAs may exceed the laptop-CPU budget. The absence of a proved Theorem 1 is the honest cost of this approach — extraction is trusted, not verified.

### 4. Best-Paper Potential

This approach positions Spectacles as the *CakeML of ZK compilers*: a system where the compiler is extracted from its own correctness proof. While CakeML achieved verified extraction from HOL4 to machine code over a decade of work, Spectacles-C demonstrates the same philosophy for a narrower domain (WFA-to-AIR) in Lean 4, which is a much more practical proof assistant for this kind of work. The narrative is compelling: "We wrote the compiler in Lean 4. The compiler *is* the proof. The extracted Rust code runs on your laptop."

For Gabriel Ebner, this approach maximizes the Lean 4 contribution: 40K+ LoC of verified algorithmic code, not just proofs about external code. The `KleeneSemiring` typeclass, verified WFA library, and `kleene_dec`/`wfa_equiv` tactics would be the largest single contribution to Mathlib's automata-theory and algebra infrastructure. The extraction pipeline, even if imperfect, demonstrates Lean 4's viability as a verified programming language for real systems — a narrative that the Lean community actively wants to advance.

For CAV specifically, the stratified verification architecture — with explicit trust boundaries between proved, validated, and assumed components — is methodologically honest in a way that CAV reviewers reward. Rather than claiming end-to-end verification (which would require verifying Winterfell and the Rust compiler), we clearly delineate what is proved, what is tested, and what is assumed.

### 5. Hardest Technical Challenge

**Performance of Lean 4 algorithmic code and extracted Rust.** The STARK trace generation for a 512-token input with a 500-state WFA requires evaluating ~256,000 field multiplications and ~512,000 field additions. In hand-written Rust with Goldilocks field optimizations, this takes ~2 seconds. Lean 4's compiler may produce code that is 10-50× slower due to (a) boxed integers instead of native `u64`, (b) reference counting overhead on matrix elements, and (c) inability to exploit SIMD or cache-line alignment.

**Mitigation (three-pronged):**

**(a) Lean 4 `@[extern]` for field arithmetic.** Mark the Goldilocks field operations (`add`, `mul`, `inv`) as `@[extern]` with hand-written Rust implementations. The field operations are axiomatized in Lean 4 (their properties are proved) but executed via foreign function calls. This is the standard Lean 4 pattern for performance-critical numeric code (used in `Nat.decLe`, for example).

**(b) Sparse matrix representation.** WFA transition matrices are typically 70-90% zero. Use `Finsupp` (already in Mathlib) as the matrix representation, avoiding computation on zero entries. This reduces the effective operation count by 5-10×.

**(c) Fallback to hand-written Rust with differential validation.** If extracted performance remains insufficient, use hand-written Rust for the trace generator and validate against the Lean implementation via differential testing on 100K inputs. This increases the TCB but preserves the laptop-CPU constraint. Document this explicitly as a pragmatic compromise.

### 6. Scores

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Value | 7 | Standards-body use case is real but narrower than safety/regulatory scenarios |
| Difficulty | 8 | Large-scale Lean 4 programming + extraction is hard but tractable with `@[extern]` escape hatches |
| Potential | 7 | "CakeML for ZK" is a strong narrative but extraction fidelity gap weakens the story |
| Feasibility | 6 | Extraction pipeline immaturity is the main risk; `@[extern]` + differential testing is the realistic path |

---

## Comparative Summary

| Dimension | A: Coalgebraic | B: KAT | C: Extraction |
|-----------|---------------|--------|---------------|
| **Value** | 9 | 8 | 7 |
| **Difficulty** | 9 | 7 | 8 |
| **Potential** | 9 | 8 | 7 |
| **Feasibility** | 5 | 7 | 6 |
| **Risk Profile** | High risk, high reward | Moderate risk, strong reward | Moderate risk, moderate reward |
| **Lean 4 Novelty** | Coalgebra library (new) | WKAT decidability (new) | Large-scale verified extraction (new) |
| **WFA Role** | Coalgebra for a functor | KAT matrix representation | Lean 4 inductive type with extraction |
| **Contamination Integration** | Syntactic monoid PSI | KAT Boolean-test PSI | Lean-specified, Rust-implemented PSI |
| **Gabriel Ebner Interest** | Coinductive proofs, `Coalgebra` typeclass | `kleene_dec` tactic, decision procedures | Large Lean 4 systems programming, `@[extern]` patterns |

**Recommendation for portfolio strategy:** Lead with Approach B (KAT) as the primary proposal — it has the best feasibility-to-potential ratio and the strongest fit with existing Lean 4/Mathlib infrastructure. Develop Approach A (Coalgebraic) as the ambitious "moonshot" variant for a top-tier venue submission. Keep Approach C (Extraction) as a methodological backup — if the KAT or coalgebraic proofs stall, the extraction approach provides a different verification strategy that may be easier to complete.
