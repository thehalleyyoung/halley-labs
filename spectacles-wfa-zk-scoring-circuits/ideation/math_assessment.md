# Math Depth Assessment — Spectacles Proposals A, B, C

**Assessor role:** Formalization quality arbiter (Lean 4 / proof automation / Mathlib perspective)

**Date:** 2026-02-24

---

## Preliminary Note on Common Infrastructure

All three proposals share certain mathematical infrastructure: a `KleeneSemiring` typeclass, a WFA library over Mathlib's `Matrix`, and two tactics (`kleene_dec`, `wfa_equiv`). I assess these shared components once here, then focus each per-approach section on the approach-specific mathematical claims.

### Shared: `KleeneSemiring` Typeclass

**Current Mathlib state (as of early 2026):** Mathlib has `Semiring`, `OrderedSemiring`, `StarRing` (for C*-algebras — the star there is involution, *not* Kleene star). There is `Computability.Language` with concatenation and Kleene star on `Set (List α)`, but this is not a typeclass; it is an operation on a specific type. There is **no** `KleeneSemiring` typeclass in Mathlib.

**Would Mathlib accept one?** Possibly, but with significant design friction:
- The axioms of Kleene algebra (Kozen's equational axioms, or Conway's) need to be chosen carefully. Kozen's axiomatization is equational and has a completeness theorem; Conway's uses more axioms. Mathlib maintainers would want the "right" axiomatization.
- The relationship to `Computability.Language` must be spelled out: `Language α` with `∪`, `·`, `∗` should be a `KleeneSemiring` instance, and this must be proved.
- The relationship to `StarRing` must be disambiguated. Having two unrelated "star" operations in the hierarchy would be a red flag for Mathlib reviewers.
- The interaction with `Matrix n n S` (matrices over a Kleene semiring form a Kleene semiring via Floyd-Warshall / Gauss-Jordan) is known math but nontrivial to formalize. The proof that `Matrix (Fin n) (Fin n) S` is a `KleeneSemiring` when `S` is one requires induction on `n` with block-matrix decomposition.

**Estimate:** Getting a clean `KleeneSemiring` accepted into Mathlib is 2–3 person-months of design iteration and review alone, independent of the Spectacles project.

### Shared: WFA Library

**Current Mathlib state:** Mathlib has `NFA` and `DFA` in `Computability.` but these are *unweighted* over `Set`. There is no `WFA` type. `Matrix`, `Finsupp`, and the `Semiring` hierarchy are mature.

**Would Mathlib accept a WFA library?** Yes, if well-designed. The natural representation is:
```
structure WFA (S : Type*) [Semiring S] (Σ : Type*) (n : ℕ) where
  transition : Σ → Matrix (Fin n) (Fin n) S
  initial : Fin n → S   -- or: (Fin n) →₀ S
  final : Fin n → S
```
This is clean. The formal-power-series semantics `eval : WFA S Σ n → List Σ → S` defined as `initial ⬝ (∏ aᵢ, transition aᵢ) ⬝ final` is standard. This part is feasible.

**Estimate:** Core WFA library (type + eval + basic lemmas): 1–2 person-months.

### Shared: Tactics `kleene_dec` and `wfa_equiv`

**`kleene_dec`:** Decide equational Kleene algebra goals by reducing to NFA equivalence and calling `native_decide`. This is a certified decision procedure. The main work is:
1. A *reflection* step: parsing a Lean goal into a KA term AST.
2. A *compilation* step: KA term → NFA (Antimirov or Brzozowski derivatives).
3. An *equivalence check*: NFA language equivalence via Hopcroft-Karp.
4. A *certification* step: producing a Lean proof term from the decision.

This is similar in architecture to `omega` or `norm_num` extensions. It is well-understood how to do this in Lean 4. The hard part is the reflection and the NFA construction.

**`wfa_equiv`:** Decide WFA equivalence over commutative semirings. This is substantially harder because:
- WFA equivalence requires computing the rank of the Hankel matrix (or equivalently, testing equality of minimal linear representations).
- Over general commutative semirings, this requires linear algebra over semirings, which is messier than over fields.
- Over fields (or embeddable semirings), you can test equality of minimal representations via the Schützenberger algorithm.

The tactic's *completeness* is deferred in the problem statement, which is honest — completeness of WFA equivalence decision procedures over arbitrary commutative semirings is subtle.

**Estimate:** `kleene_dec` (basic version, unweighted, small state count): 2–3 person-months. `wfa_equiv` (basic version, over fields): 3–4 person-months. Full generality: add 2–3 months each.

---

## Approach A: Coalgebraic WFA Semantics with Functorial Circuit Compilation

### 1. Is the Math Load-Bearing?

**Theorem 1 (Coalgebraic Compilation Soundness):**
> *"There exists a coalgebra homomorphism h: A → C such that for all inputs w, weight_A(w) = ι⁻¹(trace_C(w))."*

**Verdict: The core claim IS load-bearing; the coalgebraic framing is NOT.**

The actual load-bearing statement is: "the compiled AIR circuit computes the same function as the WFA." This is a straightforward simulation argument. You can state and prove it as:

```
theorem compile_sound (A : WFA S Σ n) (w : List Σ) :
    A.eval w = decode (compile A |>.eval w)
```

You do NOT need coalgebra homomorphisms to state or prove this. The coalgebraic language is a *reformulation* of a standard induction-on-input-length simulation proof. The proof goes through by showing the circuit state after processing prefix `w[0..i]` corresponds to the WFA state vector after processing `w[0..i]`, for each `i`. This is a plain inductive argument.

**Would the artifact break without coalgebra theory?** No. The artifact needs `compile_sound`. It does not need `Coalgebra F α` or coalgebra homomorphisms. You could prove the identical theorem with a `simp` lemma chain that unfolds `compile` and `eval` step by step.

**The coalgebraic framing buys you exactly one thing:** if you want to add a new observation functor (say, for transducers instead of acceptors), you get the compilation soundness "for free" by showing the new functor fits into the framework. But the proposal only covers three specific semirings with two tiers. The generality is aspirational, not exercised.

**Theorem 2 (Coalgebraic Bisimulation Decidability):**
> *"For WFAs over a commutative Noetherian semiring, behavioral equivalence is decidable and coincides with language equivalence."*

**Verdict: Load-bearing for the equivalence oracle, but the coalgebraic formulation adds complexity.**

The standard result (Schützenberger 1961, Berstel-Reutenauer 2011) is: two WFAs over a field are equivalent iff their minimal linear representations are isomorphic. Over commutative Noetherian semirings, the decidability holds by a different argument (well-quasi-ordering on state vectors). The *coalgebraic bisimulation* formulation (Rutten's framework) provides an alternative proof route but is strictly more complex to formalize than the direct algebraic one. You need to define the coinductive bisimulation relation, prove it coincides with trace equivalence, and then show decidability. The direct approach — minimize and compare — is more straightforward.

**Would the artifact break without coalgebraic bisimulation?** No. You need WFA equivalence decidability. You do not need it stated in coalgebraic language.

**Theorem 3 (Syntactic Monoid Contamination Bound):**
> *"The n-gram overlap equals the number of elements in the syntactic monoid of the shared prefix automaton reachable from both initial states."*

**Verdict: Ornamental. And likely incorrect as stated.**

The syntactic monoid of a regular language is the transition monoid of its minimal DFA. For *unweighted* n-gram membership (is this n-gram in the corpus?), this is a Boolean/set question — fine. But the claim that n-gram *overlap cardinality* equals the number of *reachable monoid elements* is not standard. The syntactic monoid captures the language's algebraic structure, not the cardinality of intersections. You can compute n-gram overlap as set intersection cardinality directly — there is no need to route through syntactic monoids.

Moreover, for PSI the whole point is to compute intersection cardinality *without revealing the sets*. The PSI protocol operates on hashed n-gram representations. Routing through syntactic monoids adds algebraic machinery that the PSI protocol does not use and cannot use (the monoid structure would leak information about the automaton).

**Would the artifact break without this theorem?** No. PSI works on n-gram hashes directly. The "algebraic integration" is aspirational, not operational.

### 2. Is the Math Correct?

**Theorem 1:** The statement is well-formed but under-specified. What is the `CompilableSemiring` typeclass? It needs to axiomatize:
- For Tier 1: existence of an injective semiring homomorphism `ι : S →+* 𝔽_p`.
- For Tier 2: existence of a "gadget encoding" with specific correctness properties.

For Tier 1, the counting semiring `ℕ` embeds into `𝔽_p` only up to `p - 1`. For counting n-gram matches in a 512-token sequence, you need counts up to 512. With Goldilocks `p ≈ 2⁶⁴`, this is fine. But the *injectivity* claim needs a bound: `ι` is injective on `{0, ..., N}` where `N` bounds the maximum WFA weight during evaluation. This bound exists (it's exponential in `|Q|` and linear in input length) but must be stated explicitly. The proposal does not mention this.

For Tier 2 (tropical), the bit-decomposition gadget correctness is standard ZK engineering. The proof that `min(a, b)` via bit decomposition is correct in `𝔽_p` requires `a, b < 2^k` for some `k < log₂ p`. This is achievable. The proof is tedious but not hard — it's bit-level arithmetic reasoning that `omega` and `norm_num` can handle once the setup is done.

**Theorem 2:** The statement conflates two different things: *behavioral equivalence* (coinductive) and *language equivalence* (trace equivalence). For WFAs over a field, these coincide — this is the Schützenberger-Carlyle theorem. For WFAs over a general commutative Noetherian semiring, this is **not known to hold in general**. The proposal says "commutative Noetherian semiring with decidable equality" but the correct hypothesis is "commutative *ring* with decidable equality" or more precisely, the semiring must be *positive* (a + b = 0 implies a = b = 0) or a field for the minimization theory to work cleanly. Over `ℕ` (a commutative Noetherian semiring), WFA minimization is more subtle than over `ℚ`.

**Hidden assumption:** The decidability result over arbitrary Noetherian semirings may require the semiring to be a *principal ideal domain* or at least a *field* for the rank-based approach. The proposal does not acknowledge this.

**Theorem 3:** As discussed above, the statement is likely not well-formed. "Number of elements in the syntactic monoid reachable from both initial states" is not a standard quantity that equals n-gram overlap cardinality. The syntactic monoid is a quotient of `Σ*`; its elements correspond to equivalence classes of strings under the Myhill-Nerode relation, not to individual n-grams.

### 3. Is the Math Novel?

| Component | Novel? | Comment |
|-----------|--------|---------|
| Coalgebraic WFA = coalgebra for `S × (-)^Σ` | No | This is Rutten 2000, textbook |
| Coalgebra homomorphism = correctness of compilation | Somewhat | The *application* to ZK circuits is new; the coalgebra homomorphism concept is not |
| Coinductive bisimulation decidability for WFA | No | Known since Schützenberger; coalgebraic reformulation by Bonchi, Pous, et al. |
| Syntactic monoid for PSI | Novel claim, but likely incorrect | See above |
| Lean 4 `Coalgebra` library | New formalization | No substantial `Coalgebra` library exists in Lean 4 / Mathlib |
| Compilation of WFA to AIR | New engineering | No prior art for WFA → STARK circuits |

**Honest summary:** The novel contributions are the *engineering* (WFA → AIR compiler) and the *formalization* (Lean 4 coalgebra library). The *mathematics* is well-known. The coalgebraic framing is a presentation choice that adds formalization complexity without adding mathematical power for this specific application.

### 4. Mathlib Readiness

**`Coalgebra F α` typeclass:** Mathlib has no coalgebra library. Building one from scratch is a significant undertaking. Key design questions:
- Should `Coalgebra` be a typeclass or a structure? (Mathlib precedent: `Algebra R A` is a typeclass, so `Coalgebra F α` as a typeclass is natural.)
- What universe polymorphism issues arise? Functors `F : Type u → Type u` on types are fine, but the category-theoretic generality (functors on arbitrary categories) would pull in Mathlib's `CategoryTheory` library, which has its own universe issues.
- The connection to `Mathlib.CategoryTheory.Coalgebra` (which exists as morphisms in the Eilenberg-Moore category) must be disambiguated — that's coalgebras for a *monad*, not for an endofunctor in the automata-theory sense.

**Prediction:** A `Coalgebra` typeclass for endofunctors on `Type` would be acceptable in Mathlib but would require extensive review (3–6 months of design discussion). It would NOT be accepted as part of a rush project submission.

### 5. Math Difficulty Estimate

| Proof obligation | Person-months | Difficulty | Notes |
|-----------------|---------------|------------|-------|
| `Coalgebra F α` typeclass + basic instances | 2–3 | Medium | Design iteration with Mathlib reviewers |
| `WFACoalgebra S Σ α` + formal power series semantics | 2–3 | Medium | Connecting coalgebra structure to matrix semantics |
| Coinductive bisimulation = trace equivalence (over fields) | 3–5 | **Hard** | Coinductive reasoning in Lean 4; guardedness issues |
| Coalgebra homomorphism for Tier 1 compilation | 2–3 | Medium | Inductive argument dressed in coalgebraic language |
| Coalgebra homomorphism for Tier 2 (tropical gadgets) | 3–4 | Hard | Bit-decomposition gadget correctness + simulation relation |
| Syntactic monoid PSI theorem | 2–3 | **Unknown** | Theorem may not be provable as stated |
| `kleene_dec` tactic | 2–3 | Medium | Standard certified decision procedure pattern |
| `wfa_equiv` tactic (over fields) | 3–4 | Hard | Minimization + isomorphism checking |
| **Total** | **19–28** | | ~2 person-years |

### 6. Fatal Math Issues

1. **Theorem 3 (Syntactic Monoid Contamination)** is likely not well-formed as stated. The syntactic monoid quotient does not directly give you n-gram overlap cardinality. This theorem would need to be substantially reformulated or dropped.

2. **Coinductive proofs in Lean 4:** Lean 4's kernel *does* support coinductive types (via `CoInductive` / quotients of polynomial functors), but the proof engineering for coinductive bisimulation is genuinely difficult. The mitigation (bounded bisimulation → `native_decide`) is sound in principle but:
   - The bound `|Q|²` assumes the semiring is a field (over general Noetherian semirings, the bound may be different or not exist).
   - `native_decide` on WFAs with 1000 states means deciding equivalence of automata with `|Q|² = 10⁶` potential pairs — this may not terminate in reasonable time.

3. **Conflation of Noetherian semiring and field hypotheses** in Theorem 2. The decidability result as stated (over arbitrary commutative Noetherian semirings) is stronger than what's known. Over `ℕ` with the standard ordering, WFA equivalence IS decidable (by Hliněný's result or by embedding in `ℚ`), but the *minimization* approach requires passing through the fraction field. The proof would need to handle this embedding, which is doable but not acknowledged.

**Overall assessment for Approach A:**

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Formalization Quality | 5/10 | Coalgebraic framing adds complexity without proportional benefit; Theorem 3 is ill-formed |
| Automation Sophistication | 6/10 | `kleene_dec` is solid; coinductive bisimulation tactic is ambitious but risky |
| Mathematical Depth | 4/10 | All underlying math is known; coalgebraic reformulation is not novel |
| Tooling Contribution | 7/10 | A Lean 4 coalgebra library would be genuinely useful, if well-designed |

---

## Approach B: Algebraic Program Analysis via Kleene Algebra with Tests

### 1. Is the Math Load-Bearing?

**Theorem 1 (Weighted KAT Soundness):**
> *"The WKAT denotation of an EvalSpec term over WFA and its circuit denotation satisfy the embedding equation."*

**Verdict: Load-bearing, and the KAT framing is more natural than coalgebra here.**

The key insight is that KAT expressions give you a *compositional* proof: if each KAT combinator (sequential composition, union/choice, test, iteration) is correctly compiled, then any expression built from these combinators is correctly compiled. This is genuine compositionality — the proof is by structural induction on the KAT term, and each case is a local lemma.

However, this requires that EvalSpec terms actually *are* KAT expressions. For the seven target metrics:
- Exact match: sequence of tests → KAT expression ✓
- Token F1: counting matches → needs semiring weights, not just Boolean tests ⚠
- Regex match: regular expression → KAT expression ✓
- BLEU/ROUGE n-gram: counting → weighted ⚠
- ROUGE-L: longest common subsequence → tropical semiring ⚠

The problem is that standard KAT has Boolean tests. Weighted metrics need *semiring-valued* tests. The proposal acknowledges this ("Weighted KAT extension") and identifies it as the hardest challenge.

**Would the artifact break without KAT?** Partially. The compositional proof structure is valuable. But you could achieve the same compositionality with a straightforward denotational semantics (EvalSpec → WFA) and prove `compile` correct by induction on EvalSpec terms. KAT gives you the additional benefit of the equational theory for deciding equivalence, which is a genuine win.

**Theorem 2 (WKAT Decidability):**
> *"Equivalence of WKAT expressions over a commutative semiring with decidable equality is decidable."*

**Verdict: Load-bearing for the specification equivalence oracle, BUT the statement needs qualification.**

WFA equivalence over a field is decidable (Schützenberger). Over a commutative semiring (like `ℕ`), decidability holds but requires embedding into the fraction field (embed `ℕ` into `ℚ`, decide equivalence there). The proposal does not distinguish these cases.

More importantly, the *WKAT* decidability is not the same as *WFA* decidability. WKAT expressions are a specific syntax for defining WFAs. The decidability of WKAT equivalence follows from: (1) compile WKAT expressions to WFAs, (2) decide WFA equivalence. So "WKAT decidability" is a corollary of "WFA decidability" + "WKAT-to-WFA compilation correctness." This is fine, but the proposal presents it as if WKAT decidability is an independent result.

**Theorem 3 (KAT Completeness for Evaluation Programs):**
> *"Every scoring function in EvalSpec has a WKAT denotation, and WKAT is complete for evaluation-program equivalence."*

**Verdict: This is where things get mathematically problematic.**

**Part (a):** "Every EvalSpec term has a WKAT denotation." This is a *definition* — you define the semantics of EvalSpec as mapping to WKAT. It is not a theorem; it is a design choice. The interesting question is whether the WKAT denotation *captures the intended semantics* of the EvalSpec term. This is `eval_equiv_wfa` (EvalSpec → WFA semantic preservation), which is a standard compiler correctness theorem.

**Part (b):** "WKAT is complete for evaluation-program equivalence." This is the *completeness* claim: if two EvalSpec terms compute the same function, then they are provably equal in WKAT. This requires a completeness theorem for WKAT, which **does not exist in the literature**.

Standard KAT is complete for the *relational* interpretation (Kozen & Smith 1996) and for the *language* interpretation. But:
- WKAT (weighted KAT) is not standard. It is an extension proposed by this project.
- Completeness of KAT relies on the *Boolean algebra structure* of tests. If tests carry semiring weights instead of Boolean values, the completeness proof breaks.
- The proposal acknowledges this ("the reduction breaks") and proposes two mitigations: (a) restrict to Boolean-tested WKAT, or (b) encode weights into the alphabet.

**For mitigation (a):** If tests remain Boolean, then WKAT is just "KAT + semiring-valued programs." The completeness of this hybrid system is plausible (you can likely reduce it to standard KAT completeness by encoding weights as extra state) but has not been proved. This is a genuinely new result, if it works.

**For mitigation (b):** Encoding weights into the alphabet gives `Σ × S` as the new alphabet. If `S` is infinite (like `ℕ`), the alphabet is infinite, and KAT completeness over infinite alphabets is known (Kozen's proof works for any alphabet) — so this would work, but the resulting automata are over an infinite alphabet, which is incompatible with the Hopcroft minimization algorithm (which requires finite alphabet). This creates a tension with the decidability claim.

### 2. Is the Math Correct?

**Theorem 1:** Well-formed as stated, assuming the WKAT-to-WFA compilation is correct. The proof by induction on KAT term structure is standard. Each case:
- Test `b`: WFA with a single transition checking `b` → circuit with a constraint checking the embedded test → correct if the test embedding is correct.
- Sequence `e₁ · e₂`: WFA product → circuit concatenation → correct by semiring homomorphism on matrix multiplication.
- Union `e₁ + e₂`: WFA sum → circuit addition → correct.
- Star `e*`: WFA Kleene star → circuit iteration → **this is the hard case**. The circuit must simulate unbounded iteration, but STARK AIR traces have finite length. For the metrics in scope, `e*` is always *bounded* (you iterate over a finite input), so the star unfolds to a finite sum. This must be stated explicitly.

**Hidden assumption:** The star operation in the circuit is always *bounded* by the input length. This is true for the WFA application (you process a finite input) but is not a consequence of the KAT axioms. The proof must handle this.

**Theorem 2:** Mostly correct, but the statement conflates WKAT decidability with WFA decidability. The proof route (WKAT → WFA → minimize → compare) is sound. The subtlety is that the *weighted Antimirov construction* (step 1) produces a WFA whose size may be exponential in the WKAT expression size. The decidability is not in question; the *complexity* claim (completing in seconds on a laptop) is.

**Theorem 3:** As analyzed above, Part (b) is a **new conjecture**, not a known theorem. It may be true for the Boolean-tested WKAT restriction, but it has not been proved. Claiming it as a theorem is overreach — it should be stated as a conjecture or an axiom.

### 3. Is the Math Novel?

| Component | Novel? | Comment |
|-----------|--------|---------|
| KAT for program verification | No | Kozen 1997, textbook |
| WKAT (weighted KAT) | Somewhat novel | Weight extensions of KAT have been considered (e.g., by Grathwohl, Kozen, Henglein) but not in this exact form |
| KAT × ZK circuits | Novel application | No prior work connecting KAT to zero-knowledge proofs |
| WKAT completeness | Novel conjecture | No known completeness result for weighted KAT |
| Lean 4 KAT formalization | New formalization | Limited KAT formalization exists (Struth's Isabelle/AFP; some Coq work by Pous/Braibant) |
| `kleene_dec` via Antimirov + `native_decide` | New automation | Similar in spirit to Braibant-Pous's Coq KA tactic, but for Lean 4 |

**Honest summary:** The KAT-ZK connection and the WKAT extension are genuinely novel contributions. The Lean 4 formalization would be new. The risk is that the WKAT completeness claim is a *conjecture masquerading as a theorem*.

### 4. Mathlib Readiness

**`KATAlgebra` typeclass:** More feasible than the coalgebra approach. The natural hierarchy is:
```
class KleeneSemiring (S : Type*) extends Semiring S where
  kstar : S → S
  kstar_unfold : ∀ a, kstar a = 1 + a * kstar a
  kstar_least : ∀ a x, a * x ≤ x → kstar a * x ≤ x  -- or similar
```
Then `KATAlgebra` extends `KleeneSemiring` with a Boolean subalgebra of tests.

**Design issues:**
- What ordering to use? Kozen's KA uses the *natural* partial order `a ≤ b ↔ a + b = b` (idempotent semiring). But not all semirings are idempotent. Over `ℕ` with `+` and `·`, this ordering is trivial. The KA axioms require idempotent addition for the order to work.
- If `KleeneSemiring` requires idempotent addition (`a + a = a`), then `ℕ` is NOT a `KleeneSemiring`. But the problem statement says the counting semiring `(ℕ, +, ·)` is used. **This is a fundamental tension.**
- Resolution: You need a `StarSemiring` (with Kleene star but without the KA ordering axioms) and a separate `KleeneAlgebra` that adds idempotent addition. The WFA library would use `StarSemiring`; the equivalence decision procedure would require `KleeneAlgebra` or would work at the level of formal power series rather than semiring elements.

This design tension is significant and would take 1–2 months of Mathlib review discussion to resolve.

**Prediction:** A basic `KleeneSemiring` / `StarSemiring` typeclass is the most likely Mathlib-accepted component from any of the three proposals.

### 5. Math Difficulty Estimate

| Proof obligation | Person-months | Difficulty | Notes |
|-----------------|---------------|------------|-------|
| `KleeneSemiring` / `KATAlgebra` typeclass + instances | 2–3 | Medium | Design iteration; idempotent-semiring vs. star-semiring question |
| `Language α` is a `KleeneSemiring` instance | 1–2 | Medium | Need to prove KA axioms for language operations |
| `Matrix (Fin n) (Fin n) S` is a `KleeneSemiring` instance | 3–4 | **Hard** | Block decomposition; convergence of Kleene star for matrices |
| WKAT-to-WFA compilation (weighted Antimirov) | 2–3 | Medium | Standard construction, tedious to formalize |
| Compilation soundness (Theorem 1) by structural induction | 2–3 | Medium | Each KAT combinator case is a local proof |
| WKAT decidability (Theorem 2) | 3–4 | Hard | Depends on WFA minimization over the right semiring class |
| WKAT completeness (Theorem 3) | 4–6+ | **Very Hard / Open** | This is a research-level conjecture; may not be provable |
| `kleene_dec` tactic | 2–3 | Medium | Antimirov + NFA equivalence + `native_decide` |
| `wfa_equiv` tactic | 3–4 | Hard | Weighted minimization + isomorphism |
| **Total** | **22–32** | | ~2–2.5 person-years |

### 6. Fatal Math Issues

1. **WKAT completeness (Theorem 3b) is an open problem.** The proposal presents it as a theorem to be proved, but no proof exists in the literature. Over Boolean-tested WKAT (mitigation (a)), completeness might follow from standard KAT completeness by a product construction, but this has not been verified. **If this theorem cannot be proved, the "specification oracle" story weakens but does not collapse** — you can still decide WFA equivalence directly without going through WKAT completeness.

2. **The counting semiring `(ℕ, +, ·)` is NOT an idempotent semiring**, so it is not a Kleene algebra in Kozen's sense. The `KleeneSemiring` typeclass must be designed carefully to accommodate both idempotent semirings (for KA reasoning) and non-idempotent semirings (for weighted automata). This is a solvable design issue but it affects the entire typeclass hierarchy.

3. **Matrix Kleene star convergence.** Over `(ℕ, +, ·)`, the Kleene star `M* = I + M + M² + ...` diverges unless `M` is nilpotent. For WFA evaluation, you never need the infinite star — you process a finite input and unfold the star `|w|` times. But the `KleeneSemiring` typeclass would require `kstar` to be total, which it isn't for `Matrix (Fin n) (Fin n) ℕ` in general. **Resolution:** Use `StarSemiring` without a convergence axiom, and add a well-foundedness/nilpotency hypothesis where needed. This is doable but adds complexity to the typeclass design.

4. **Symbolic PSI via KAT normal forms** is vaguely specified. What is a "KAT normal form"? KAT has a normal-form theorem (via the Kozen-Smith completeness proof), but this gives automata, not algebraic expressions. The PSI protocol would operate on automata representations, which is essentially the same as the standard approach. The "KAT Boolean-test PSI" description does not add algebraic insight.

**Overall assessment for Approach B:**

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Formalization Quality | 7/10 | KAT is well-suited; but WKAT completeness is overreach; typeclass design is subtle |
| Automation Sophistication | 7/10 | `kleene_dec` is a genuine certified decision procedure; good fit for Lean 4 |
| Mathematical Depth | 5/10 | KAT is known; WKAT extension is interesting but completeness is open |
| Tooling Contribution | 8/10 | `KleeneSemiring` + `kleene_dec` tactic would be widely useful in Mathlib |

---

## Approach C: Stratified Verification with Extraction

### 1. Is the Math Load-Bearing?

**Theorem 1 (Extraction Preservation):**
> *"The extracted Rust code computes the same function as the Lean 4 implementation."*

**Verdict: Explicitly NOT proved. This is honest, and the theorem is vacuously load-bearing because it's the TCB boundary.**

The proposal is transparent: extraction fidelity is *validated* (differential testing on 100K inputs), not *proved*. This is the right call — proving extraction correctness is a multi-year project (CakeML spent ~7 years on it). The trust boundary is clearly delineated.

However, this means **the Lean proofs do not provide end-to-end soundness**. The chain is:
1. `eval_equiv_wfa`: EvalSpec → WFA (proved in Lean 4) ✓
2. `compile_sound`: WFA → AIR (proved in Lean 4) ✓
3. AIR → STARK: assumed from Winterfell ✓ (standard cryptographic assumption)
4. Lean → Rust extraction: **tested, not proved** ✗ (breaks the proof chain)

The net effect: the Lean proofs guarantee that *the Lean implementation is correct*, and differential testing provides *evidence* that the Rust extraction matches. This is weaker than Approaches A and B, which prove correctness about Rust code that is separately written and tested. Wait — actually, Approaches A and B also don't prove anything about the Rust code directly. They prove properties about mathematical objects (WFAs, circuits) and then separately implement those objects in Rust with differential testing. **All three approaches have the same trust gap**: mathematical proofs in Lean 4 + differential testing that the Rust implementation matches.

The difference: Approach C narrows the gap by generating the Rust code from Lean 4 (so the extraction is the only source of discrepancy), while A and B have a hand-written Rust implementation (where any line could diverge from the Lean specification).

**This is actually an advantage of Approach C**, even without proved extraction.

**Theorem 2 (Stratified Soundness Composition):**
> *"End-to-end soundness decomposes into four components with explicit trust assumptions."*

**Verdict: Load-bearing as a *structuring principle*, not as a single theorem.**

This is really a *proof architecture* claim: the verification can be decomposed into independent, composable pieces. This is good engineering and good paper structure. Each piece has a clear trust assumption. The composition is by modus ponens. This is not deep mathematics — it's a *proof plan* — but it is valuable.

**Theorem 3 (Verified WFA Minimization):**
> *"Weighted Hopcroft minimization produces a minimal equivalent WFA."*

**Verdict: Load-bearing for performance (not correctness).**

Without minimization, the circuit may have too many columns to fit in the laptop-CPU budget. The theorem guarantees that minimization preserves semantics: `eval(A_min) = eval(A)`. This is a standard result (Berstel-Reutenauer, Chapter 2) but has never been formalized in Lean 4.

**Key subtlety:** Weighted Hopcroft minimization works over *fields* (you need to compute the column space / row space of the Hankel matrix). Over `ℕ` (the counting semiring), there is no canonical minimal WFA — you need to embed in `ℚ`, minimize there, and show the minimal WFA has `ℕ`-valued entries. This embedding step is nontrivial and must be handled explicitly.

### 2. Is the Math Correct?

**Theorem 2 (Stratified Composition):** Correct. The composition is a chain of implications:
- `eval_equiv_wfa` gives: EvalSpec semantics = WFA semantics
- `compile_sound` gives: WFA semantics = AIR trace semantics (up to encoding)
- STARK soundness gives: if the verifier accepts, then the AIR trace satisfies the constraints
- Extraction fidelity gives: the Rust code computes the same as the Lean code

These compose by transitivity. No hidden assumptions beyond those stated.

**Theorem 3 (WFA Minimization):** The statement is correct for WFAs over fields. The proof in Lean 4 would require:
- A `Field` instance on the weight semiring (or at minimum, `DivisionRing`)
- Linear algebra over `Fin n → S` (row reduction, rank computation) — this exists in Mathlib
- The Myhill-Nerode quotient construction — this is the hard part, connecting the Hankel matrix rank to the minimal state count

**Hidden assumption:** "Weighted Hopcroft minimization" is a misnomer. Hopcroft's algorithm is for *unweighted* DFA minimization (partition refinement). Weighted minimization uses *algebraic* methods (basis reduction, not partition refinement). The proposal conflates two different algorithms. The correct reference is the *weighted Myhill-Nerode theorem* (Carlyle-Paz, Fliess), not Hopcroft.

### 3. Is the Math Novel?

| Component | Novel? | Comment |
|-----------|--------|---------|
| Lean 4 code extraction to Rust | New engineering | No production-quality Lean 4 → Rust pipeline exists |
| `compile_sound` proved in Lean 4 | New formalization | No prior mechanized WFA-to-circuit correctness proof |
| Stratified trust architecture | Known methodology | Standard in verified systems (CompCert, CakeML, seL4) |
| Verified WFA minimization in Lean 4 | New formalization | Not mechanized in any proof assistant (to my knowledge) |
| `@[extern]` FFI for field arithmetic | Known Lean 4 pattern | Standard technique |

**Honest summary:** The novelty here is in *engineering and formalization*, not in *mathematics*. The proof architecture is standard. The WFA minimization formalization is new. The extraction pipeline is new engineering. The mathematical depth is the shallowest of the three approaches.

### 4. Mathlib Readiness

**Best of the three approaches for Mathlib contributions.** The components are:
- `KleeneSemiring` typeclass (shared — see above)
- WFA library with `Matrix`-based semantics (shared — see above)
- Verified WFA minimization (would be a genuine Mathlib contribution if done for WFAs over fields)
- `compile_sound` theorem (too application-specific for Mathlib; belongs in the project repo)

The Approach C specific contributions (extraction pipeline, stratified verification) are *not Mathlib contributions* — they are project infrastructure.

**Key advantage:** By writing the WFA engine in Lean 4, the code and the specification are the same artifact. This avoids the specification-implementation gap that plagues Approaches A and B. For Mathlib, this means the WFA library is not just theorems but also *executable* — you can `#eval` a WFA and get results. This is highly valued in modern Mathlib.

### 5. Math Difficulty Estimate

| Proof obligation | Person-months | Difficulty | Notes |
|-----------------|---------------|------------|-------|
| `KleeneSemiring` typeclass + instances | 2–3 | Medium | Shared infrastructure |
| WFA library (type + eval + basic lemmas) | 1–2 | Medium | Shared infrastructure |
| `eval_equiv_wfa` (EvalSpec → WFA) | 2–3 | Medium | By induction on EvalSpec terms; each metric is a separate case |
| `compile_sound` for Tier 1 (algebraic) | 3–4 | Hard | Matrix arithmetic over 𝔽_p; semiring homomorphism properties |
| `compile_sound` for Tier 2 (tropical) | 3–5 | **Hard** | Bit-decomposition gadget correctness; comparison circuit verification |
| WFA minimization (over fields) | 3–5 | **Hard** | Linear algebra; Myhill-Nerode quotient; rank computation |
| `kleene_dec` tactic | 2–3 | Medium | Shared infrastructure |
| Lean-to-Rust extraction pipeline | 4–6 | **Very Hard** | Not a proof obligation but engineering prerequisite |
| `@[extern]` FFI for Goldilocks field | 1–2 | Easy | Standard Lean 4 pattern |
| Differential testing infrastructure | 1–2 | Easy | Engineering, not math |
| **Total (proof obligations only)** | **17–29** | | ~1.5–2.5 person-years |
| **Total (including extraction pipeline)** | **21–35** | | ~2–3 person-years |

### 6. Fatal Math Issues

1. **Lean-to-Rust extraction does not exist at production quality.** This is the Achilles heel. The proposal's mitigation (`@[extern]` for hot paths + differential testing) is pragmatic but means the actual Rust code running the STARK prover is a mix of extracted code, hand-written FFI, and unverified Winterfell calls. The "compiler is the proof" narrative partially collapses to "the compiler is the proof, but only for the parts we didn't optimize away."

2. **Performance of extracted code is a genuine risk.** The proposal estimates 10–50× slowdown for naive extraction. With a 2-second hand-written Rust target, a 50× slowdown means 100 seconds per proof — feasible but marginal. With a 30-second target (ROUGE-L), a 50× slowdown means 25 minutes per proof — infeasible for the 50-example, 7-metric, 60-minute end-to-end target.

3. **"Weighted Hopcroft" does not exist.** The proposal must use weighted Myhill-Nerode / basis reduction, not Hopcroft's partition refinement. This is a conceptual error, not a fatal one — the correct algorithm exists and is formalizable — but it suggests incomplete understanding of the weighted minimization literature.

4. **WFA minimization over `ℕ`** (the counting semiring) requires embedding into `ℚ`, minimizing the `ℚ`-WFA, and checking that the minimal WFA has `ℕ`-valued entries. If it doesn't (which can happen — the minimal `ℚ`-WFA may have rational entries not realizable over `ℕ`), then there is no unique minimal `ℕ`-WFA. The proposal does not address this.

**Overall assessment for Approach C:**

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Formalization Quality | 6/10 | Honest trust boundaries; but extraction gap weakens the "proof is the code" narrative |
| Automation Sophistication | 5/10 | Tactics are shared infrastructure; no approach-specific automation novelty |
| Mathematical Depth | 3/10 | Shallowest of the three; mostly engineering + standard formalization |
| Tooling Contribution | 6/10 | WFA library is good; extraction pipeline is high-risk and not Mathlib-relevant |

---

## Cross-Cutting Issues Affecting All Three Approaches

### Issue 1: The Semiring Embedding Injectivity Bound

All three approaches need the fact that `ι : ℕ → 𝔽_p` (embedding the counting semiring into the prime field) is injective *on the range of values that actually appear during WFA evaluation*. For a WFA with `n` states processing input of length `ℓ`, the maximum weight is at most `(max_weight)^ℓ · n^ℓ` (crude bound). For the Goldilocks field `p ≈ 2⁶⁴`, with `n ≤ 1100` and `ℓ ≤ 512`, the actual weights are at most `1100^512` which is **astronomically larger than `2⁶⁴`**.

**This means the counting semiring does NOT embed injectively into the Goldilocks field for all inputs.**

The fix: the WFA weights for NLP metrics are actually *bounded*. For exact match (Boolean), weights are 0 or 1. For token F1 (counting), weights count matches in a 512-token sequence, so they are at most 512. For BLEU unigram precision, the clipped count is at most the reference count. The maximum weight needs to be bounded *per metric* and checked against `p`. This is straightforward but must be stated explicitly as a hypothesis in the soundness theorem.

**None of the three proposals mention this issue.** This is a hidden assumption that must be surfaced.

### Issue 2: The Tropical Semiring's `+∞`

The tropical semiring `(ℤ ∪ {+∞}, min, +)` includes `+∞` as the additive identity (identity for `min`). Encoding `+∞` in `𝔽_p` requires a sentinel value, and the bit-decomposition gadgets must handle this case. Standard ZK practice uses a "large number" as infinity, with a flag bit. This is formalizable but adds edge cases to every tropical-semiring proof.

### Issue 3: `sorry`-Free Compilation

The problem statement claims `circuit_sound_algebraic` and `circuit_sound_tropical` will be `sorry`-free at submission. Based on my estimates:
- `circuit_sound_algebraic`: 3–4 person-months of hard Lean 4 work. Achievable.
- `circuit_sound_tropical`: 3–5 person-months. The bit-decomposition gadgets are the bottleneck. Achievable but tight.

Both require the WFA library and `KleeneSemiring` typeclass as prerequisites (2–4 months). Total: 8–13 person-months before you can even *start* on the main theorems. This is a 1.5-year minimum timeline for a single person. With 2–3 Lean experts, 6–8 months is plausible.

### Issue 4: Decidability of WFA Equivalence Over Non-Field Semirings

The tropical semiring `(ℤ ∪ {+∞}, min, +)` is NOT a field. WFA equivalence over the tropical semiring is decidable (it reduces to the equality of rational functions via Krob's theorem), but the algorithm is completely different from the field case (Schützenberger). The proposal does not acknowledge that WFA equivalence requires different algorithms for different semirings, and the `wfa_equiv` tactic cannot be uniform.

In fact, WFA equivalence over the tropical semiring is **significantly harder** than over fields — it relates to decidability of the equality of entries of products of tropical matrices, which connects to deep questions in tropical geometry. The decidability is known (Krob 1994) but the algorithm is complex.

---

## Final Comparative Summary

| Dimension | A: Coalgebraic | B: KAT | C: Extraction |
|-----------|---------------|--------|---------------|
| **Formalization Quality** | 5/10 | 7/10 | 6/10 |
| **Automation Sophistication** | 6/10 | 7/10 | 5/10 |
| **Mathematical Depth** | 4/10 | 5/10 | 3/10 |
| **Tooling Contribution** | 7/10 | 8/10 | 6/10 |
| **Estimated effort (person-months)** | 19–28 | 22–32 | 17–29 (21–35 w/ extraction) |
| **Fatal issues** | Theorem 3 ill-formed; coinductive proofs risky | WKAT completeness open; `ℕ` not a KA | Extraction pipeline doesn't exist; perf risk |

### Recommendation

**Approach B (KAT) is the most mathematically sound**, with the caveat that WKAT completeness (Theorem 3b) must be downgraded from "theorem" to "conjecture" or restricted to the Boolean-tested fragment where it is provable. The `KleeneSemiring` typeclass and `kleene_dec` tactic are the highest-value Mathlib contributions and are achievable.

**Approach A (Coalgebraic) is the most over-engineered.** The coalgebraic framing adds 5–8 person-months of proof engineering to say the same thing that a direct inductive proof says. The payoff (extensibility to new functors) is aspirational and not exercised. Theorem 3 (syntactic monoid PSI) should be dropped entirely.

**Approach C (Extraction) is the most honest about trust boundaries** but has the least mathematical depth. It is the right engineering approach but the wrong research contribution — a paper whose novelty is "we wrote it in Lean and extracted it" needs the extraction to be *verified*, which it isn't.

**All three approaches share a hidden issue:** the semiring embedding injectivity bound. This must be addressed explicitly in whichever approach is chosen. It is not hard to fix (add a hypothesis that weights are bounded by `p`) but its omission from all three proposals suggests insufficient engagement with the concrete details of finite-field arithmetic.
