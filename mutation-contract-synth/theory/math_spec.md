# Mathematical Specification: Mutation-Driven Contract Synthesis

## Preamble: Core Definitions

**Program model.** Fix a simple imperative language L with functions f: X → Y over
first-order types. A *program* P is a finite set of function definitions. A *test suite*
T = {t₁,…,tₙ} is a finite set of input/expected-output pairs for functions in P.

**Mutation operators.** A *mutation operator* μ: Prog → ℘(Prog) maps a program to a
finite set of syntactic variants. Write M(P) = ⋃_μ μ(P) for the set of all mutants.
Each mutant m ∈ M(P) differs from P in exactly one syntactic location (first-order
mutants; higher-order is discussed in §Framing B).

**Killing.** Mutant m is *killed by* test t (written t ⊢ m↯) iff running m on t's input
produces an output distinct from t's expected output. Mutant m *survives* T iff
∀t ∈ T. ¬(t ⊢ m↯).

**Specification lattice.** Fix a logical language Spec (e.g., quantifier-free LIA+arrays).
The set of all specifications forms a lattice (Spec, ⊑) under logical implication:
φ ⊑ ψ iff φ ⊨ ψ. Meet is conjunction, join is disjunction. Top ⊤ = true (weakest),
bottom ⊥ = false (strongest/inconsistent).

**Contract.** A *contract* for f is a pair (Pre, Post) where Pre ∈ Spec is a precondition
and Post ∈ Spec is a postcondition. f *satisfies* (Pre, Post) iff for all inputs x:
Pre(x) ⟹ Post(x, f(x)). Write f ⊨ (Pre, Post).

---

## FRAMING A: Theoretical — The Mutation–Specification Correspondence

### Core Question
*Does the mutation-survival boundary uniquely determine a minimal specification in
(Spec, ⊑), and under what conditions?*

### Required Theorems and Lemmas

#### A1. Mutation Killing as Specification Witness (LOAD-BEARING)

**Theorem (Mutation-Specification Duality).**
Let f be a function, T a test suite, M(f) the set of first-order mutants.
Define the *mutation-induced specification*:

  φ_M(f,T) = ⋀_{m killed by T} ¬ℰ(m)

where ℰ(m) is the *error predicate* characterizing the behavioral difference between
m and f (i.e., ℰ(m)(x) holds iff m(x) ≠ f(x)).

Then:
(a) f ⊨ φ_M(f,T) (soundness: the original program satisfies the induced spec).
(b) For every killed mutant m: m ⊭ φ_M(f,T) (completeness w.r.t. killed mutants).
(c) φ_M(f,T) is the *weakest* specification in Spec that separates f from all killed mutants.

**Status:** Part (a) is straightforward (definition-chasing). Part (b) is straightforward.
Part (c) requires showing maximality in the lattice — moderate difficulty; essentially
a Craig interpolation argument restricted to the mutation fragment.

**Novelty:** The formulation is new. The underlying technique (using distinguishing
predicates) appears in concept learning and Angluin-style learning, but the
connection to mutation testing is unexploited in the PL literature.

**Difficulty:** Moderate.

---

#### A2. Lattice Structure of Mutation-Reachable Specifications (LOAD-BEARING)

**Theorem (Mutation Lattice Embedding).**
Let MKill(f,T) ⊆ M(f) be the killed mutants and MSurv(f,T) ⊆ M(f) the survivors.
Define the *discrimination lattice* D(f,T) as the lattice of subsets of MKill(f,T)
ordered by set inclusion, mapped into Spec via the map:

  σ: ℘(MKill) → Spec,  σ(S) = ⋀_{m∈S} ¬ℰ(m)

Then:
(a) σ is a lattice homomorphism from (℘(MKill), ⊆) to (Spec, ⊑).
(b) The image σ(℘(MKill)) is a finite sub-lattice of Spec.
(c) The minimal element σ(MKill) = φ_M(f,T) is the strongest mutation-derivable spec.
(d) Every specification φ with f ⊨ φ and φ distinguishes f from all killed mutants
    satisfies σ(MKill) ⊑ φ (i.e., σ(MKill) is strongest among sound discriminators).

**Status:** Part (a)–(c) are straightforward lattice theory. Part (d) requires care about
the expressiveness of Spec — it holds when Spec is closed under the relevant Boolean
operations, which is true for QF-LIA, QF-BV, etc.

**Novelty:** Known lattice-theoretic machinery applied in a new setting. The specific
embedding and the characterization of its image are new.

**Difficulty:** Straightforward to moderate.

---

#### A3. Completeness of Mutation Operators (LOAD-BEARING)

**Theorem (Mutation Completeness ⇒ Specification Tightness).**
Say a set of mutation operators M is *ε-complete for specification class* Spec if:
for every specification φ ∈ Spec such that f ⊨ φ but φ is not a tautology, there
exists a mutant m ∈ M(f) such that m ⊭ φ.

If M is ε-complete for Spec, then the mutation-induced specification φ_M(f,T) from
Theorem A1, computed using a *complete* test suite (one that kills all killable
mutants), equals the strongest specification in Spec satisfied by f.

**Status:** This is the key *coupling* between mutation adequacy and specification
strength. The "competent programmer hypothesis" in mutation testing is a
soft version of this; our theorem gives the formal version.

**Novelty:** Genuinely new. This theorem provides the first formal justification for
why mutation-adequate test suites yield "good" specifications, rather than merely
good test suites. It bridges the mutation testing and specification mining literatures.

**Difficulty:** Hard. The ε-completeness condition is the crux: characterizing which
mutation operators are complete for which spec languages requires analysis of
expressive power. For QF-LIA over simple imperative programs, we conjecture
{AOR, ROR, LCR, UOI} is ε-complete — proving this is nontrivial.

---

#### A4. Surviving Mutants and Specification Gaps (LOAD-BEARING)

**Theorem (Survivor Characterization).**
Let m ∈ MSurv(f,T). Then exactly one of:
(a) m is *equivalent* to f (∀x. m(x) = f(x)) — the specification correctly ignores it.
(b) m is *distinguishable* from f but T fails to witness this — the specification has
    a *gap* (a behavior allowed by the inferred spec but not by the "true" spec).

Moreover, for every surviving non-equivalent mutant m, there exists a specification
strengthening φ' ⊏ φ_M(f,T) such that f ⊨ φ' and m ⊭ φ'.

**Corollary (Equivalent Mutant Barrier).**
The problem of deciding whether a surviving mutant is equivalent is undecidable in
general (reduces to program equivalence). Therefore: *no algorithm can compute the
strongest sound specification from mutation analysis alone.*

**Status:** The theorem is moderate (case analysis + construction). The corollary is a
known undecidability result applied to our setting.

**Novelty:** The framing in terms of specification gaps is new and actionable (it tells
the engineer exactly what surviving mutants mean for contract quality).

**Difficulty:** Moderate (theorem), straightforward (corollary).

---

#### A5. Minimality via Interpolation (NICE-TO-HAVE)

**Theorem (Craig Interpolation for Minimal Contracts).**
Given program f with semantics ⟦f⟧ and a set of killed mutants {m₁,…,mₖ} with
semantics ⟦m₁⟧,…,⟦mₖ⟧, if the formula ⟦f⟧(x,y) ∧ ⋁ᵢ ⟦mᵢ⟧(x,y') ∧ y≠y'
is unsatisfiable restricted to precondition Pre(x), then there exists a Craig
interpolant I(x,y) such that:
- Pre(x) ∧ ⟦f⟧(x,y) ⊨ I(x,y)
- Pre(x) ∧ I(x,y) ⊨ ⋀ᵢ ¬(⟦mᵢ⟧(x,y) ∧ y≠f(x))

and I is *minimal* in the interpolation lattice.

**Status:** Application of Craig interpolation (well-known) to the mutation setting.

**Novelty:** The application is new; the math is classical.

**Difficulty:** Straightforward (given interpolating SMT solvers).

---

#### A6. Galois Connection to Abstract Interpretation (NICE-TO-HAVE, but deep)

**Theorem (Mutation Abstraction as Galois Connection).**
Define the concrete domain C = ℘(Traces(f)) (sets of program traces) and
abstract domain A = ℘(M(f)) × {killed, survived}^|T| (mutation outcomes).
There exists a Galois connection (C, α, γ, A) where:
- α maps a set of traces to the mutation outcomes they induce
- γ maps mutation outcomes back to the most precise set of traces consistent
  with those outcomes

The induced specification φ_M(f,T) is exactly α ∘ γ applied to the abstract
state "all killed mutants killed, all survivors survived."

**Status:** This connects our framework to Cousot's abstract interpretation hierarchy.
The Galois connection structure is the standard template; showing it holds for
mutation outcomes requires verifying the adjunction properties.

**Novelty:** New connection. Abstract interpretation has been linked to testing
(Cousot 2019) but not to mutation-based specification inference.

**Difficulty:** Moderate.

---

### Summary Table: Framing A

| ID  | Statement | New? | Difficulty | Load-bearing? |
|-----|-----------|------|------------|---------------|
| A1  | Mutation-Spec Duality | Yes (formulation) | Moderate | YES |
| A2  | Lattice Embedding | Partially | Straightforward–Moderate | YES |
| A3  | Completeness ⇒ Tightness | Yes | Hard | YES |
| A4  | Survivor Characterization | Yes (framing) | Moderate | YES |
| A5  | Craig Interpolation for Minimality | Application | Straightforward | No |
| A6  | Galois Connection to AI | Yes (connection) | Moderate | No |

---

## FRAMING B: Practical — Mathematics for Scaling

### Core Question
*What mathematical machinery enables mutation-driven contract synthesis on
100K+ LOC codebases within reasonable time?*

### Required Theorems and Lemmas

#### B1. Abstract Mutation via Predicate Abstraction (LOAD-BEARING)

**Theorem (Sound Mutation Approximation).**
Let α: Stmt → Stmt# be an abstraction mapping concrete mutations to *abstract
mutation classes* (e.g., "any arithmetic operator replacement in expression e"
rather than each specific replacement). Define the abstract error predicate:

  ℰ#(μ#)(x) = ∃m ∈ γ(μ#). m(x) ≠ f(x)

Then ℰ#(μ#) ⊒ ⋁_{m ∈ γ(μ#)} ℰ(m) (over-approximation).

The specification computed with abstract mutations:

  φ#_M = ⋀_{μ# killed} ¬ℰ#(μ#)

satisfies: φ#_M ⊑ φ_M (it is weaker than the concrete spec — sound but less precise).

**Why it matters:** Instead of enumerating O(n·k) concrete mutants (n = program
points, k = operators), we work with O(n) abstract mutation classes.

**Status:** Standard abstract interpretation soundness argument.

**Novelty:** Application of known technique.

**Difficulty:** Straightforward.

---

#### B2. Mutant Subsumption and Redundancy Elimination (LOAD-BEARING)

**Theorem (Subsumption-Based Reduction).**
Mutant m₁ *subsumes* m₂ (written m₁ ≥ m₂) iff every test that kills m₁ also
kills m₂. If m₁ ≥ m₂, then ¬ℰ(m₁) ⊑ ¬ℰ(m₂), so m₂ is redundant for
specification inference: removing m₂ does not weaken φ_M.

Define the *dominator set* D ⊆ MKill as a minimal subset such that every m ∈ MKill
is subsumed by some d ∈ D. Then φ_M computed from D equals φ_M computed from MKill.

**Reduction bound:** For programs with n mutation sites and k operator types,
|D| = O(n) in practice (each site contributes ≤1 dominator), versus |MKill| = O(nk).

**Status:** Mutant subsumption is known (Ammann et al., Kurtz et al.). The connection
to specification strength via the lattice is new.

**Novelty:** The subsumption ↔ specification implication correspondence is new.

**Difficulty:** Straightforward.

---

#### B3. Compositional Mutation Analysis via Frame Rule (LOAD-BEARING)

**Theorem (Compositional Mutation Frame).**
Let P = f₁; f₂ be a sequential composition. If mutation m affects only f₁, and
we have contracts (Pre₁, Post₁) for f₁ and (Pre₂, Post₂) for f₂ such that
Post₁ ⊨ Pre₂, then:

(a) m kills test t on P iff m kills t' on f₁, where t' is t restricted to f₁'s interface.
(b) The specification inferred for f₁ via mutations local to f₁ is *compositionally sound*:
    combining it with f₂'s spec via sequential composition in Hoare logic yields a
    valid spec for P.

**Why it matters:** Enables per-function analysis. Without compositionality, mutation
analysis of a 100K LOC program requires whole-program execution for each mutant.

**Status:** This is essentially the Hoare logic frame rule applied to mutation analysis.
The proof obligation is showing that mutation-local reasoning is sound when combined
with contract-based assume/guarantee reasoning.

**Novelty:** Application of well-known frame rule; the mutation-specific formulation
and proof of compatibility are new.

**Difficulty:** Moderate. The subtlety is handling mutations that affect shared state
(heap, globals). For purely functional fragments: straightforward. For stateful
programs: requires framing conditions (separation logic style).

---

#### B4. Symbolic Mutation via Weakest Precondition Differencing (LOAD-BEARING)

**Theorem (WP Differencing).**
Let f be a function and m a mutant differing at statement s. Let s' be the
mutated statement. Then:

  ℰ(m)(x) ≡ wp(f[s↦s'], Post)(x) ⊕ wp(f, Post)(x)

where ⊕ is the *symmetric difference* of the two weakest precondition formulas
(the set of inputs where they disagree). In practice:

  ℰ(m)(x) ≡ wp(prefix, wp(s', Post_suffix) ∧ ¬wp(s, Post_suffix))(x)
            ∨ wp(prefix, ¬wp(s', Post_suffix) ∧ wp(s, Post_suffix))(x)

where prefix is the code before s and Post_suffix = wp(suffix, Post).

**Why it matters:** Avoids executing mutants. The error predicate is computed
*symbolically* via WP calculus, enabling batch SMT solving over all mutants at
a given site simultaneously.

**Status:** WP calculus is textbook. The differencing formulation and its use for
batch mutation analysis are new.

**Novelty:** New formulation enabling significant practical speedup.

**Difficulty:** Straightforward (the formula). Moderate (making the SMT encoding
efficient for batch solving across mutation sites).

---

#### B5. Bounded Model Checking Sufficiency (LOAD-BEARING)

**Theorem (Bounded Completeness for Finite-State Fragments).**
For programs over bounded integer types (e.g., bitvectors of width w) with
loop unrolling bound k, the bounded SMT encoding is *complete*: if a contract
violation exists within k iterations, the bounded checker finds it.

Moreover, for loop-free code, the bound k=1 suffices (no unrolling needed),
and the SMT query is in NP (QF-BV satisfiability).

**Completeness gap:** For programs with unbounded loops, bounded checking is
incomplete. The gap is characterized precisely: a contract verified to bound k
is a *k-inductive invariant* of the contract, which may not be a true invariant.

**Status:** Well-known (CBMC, etc.). Stated here for completeness.

**Novelty:** None (application).

**Difficulty:** Straightforward.

---

#### B6. Incremental SyGuS via Counterexample Recycling (NICE-TO-HAVE)

**Lemma (Counterexample Monotonicity).**
Let φ₁ ⊑ φ₂ be two specifications (φ₁ stronger). If candidate contract C
is rejected by SyGuS with counterexample cex for φ₁, then cex is also a valid
counterexample for φ₂ (or C already satisfies φ₂).

**Consequence:** When strengthening a specification (adding more killed mutants),
previous counterexamples remain valid. The SyGuS solver can warm-start from
the previous counterexample set, avoiding redundant work.

**Status:** Simple monotonicity observation.

**Novelty:** Minor.

**Difficulty:** Straightforward.

---

#### B7. Complexity Bounds (NICE-TO-HAVE)

**Theorem (Complexity of Mutation-Driven Synthesis).**
For a program with n statements, k mutation operators, and a SyGuS grammar
of size g:
- Naive: O(nk) mutant executions × O(|T|) tests × SyGuS(g, nk counterexamples)
- With B1 (abstraction): O(n) abstract classes × symbolic analysis
- With B2 (subsumption): O(n) dominator mutants
- With B3 (compositionality): O(Σᵢ nᵢkᵢ) where nᵢ is per-function size
- With B4 (symbolic): O(n) WP computations + O(n) SMT queries (batched)

**Net:** Per function of size n, the cost is O(n) SMT queries of size O(n) each,
giving O(n²) total formula size per function — tractable for functions up to ~1000 LOC.

**Status:** Complexity analysis of the algorithmic pipeline.

**Novelty:** The specific bounds are new; the techniques are known.

**Difficulty:** Straightforward.

---

### Summary Table: Framing B

| ID  | Statement | New? | Difficulty | Load-bearing? |
|-----|-----------|------|------------|---------------|
| B1  | Abstract Mutation Approximation | Application | Straightforward | YES |
| B2  | Subsumption ↔ Spec Implication | Partially | Straightforward | YES |
| B3  | Compositional Frame Rule | Partially | Moderate | YES |
| B4  | WP Differencing | Yes (formulation) | Straightforward–Moderate | YES |
| B5  | Bounded Completeness | Known | Straightforward | YES |
| B6  | CE Recycling | Minor | Straightforward | No |
| B7  | Complexity Bounds | New (analysis) | Straightforward | No |

---

## FRAMING C: Systems — SyGuS Grammar, SMT Encoding, Compositional Verification

### Core Question
*What math governs the design of the synthesis grammar, the structure of SMT
queries, and the compositional glue?*

### Required Theorems and Lemmas

#### C1. Grammar Completeness for Mutation-Induced Specs (LOAD-BEARING)

**Theorem (Grammar Sufficiency).**
Let Spec_M ⊆ Spec be the set of specifications expressible as Boolean
combinations of atomic mutation-error predicates ¬ℰ(m). Define a SyGuS grammar
G_M that includes:
- Arithmetic comparisons (x ≤ c, x ≥ c, x = c) for ROR/AOR mutations
- Nullity/type checks for UOI/SDL mutations
- Array bounds predicates (0 ≤ i < len(a)) for array-related mutations
- Boolean connectives (∧, ∨, ¬, ⟹)

Then G_M is *complete* for Spec_M: every specification in Spec_M is expressible
as a formula generated by G_M.

**Why it matters:** The SyGuS solver cannot synthesize contracts outside the grammar.
If the grammar misses a predicate needed to express a mutation-induced spec,
the system silently produces a weaker contract. This theorem guarantees no such gap.

**Status:** Requires analyzing each mutation operator class and showing its error
predicate decomposes into grammar atoms.

**Novelty:** The systematic mutation-operator → grammar-atom correspondence is new.

**Difficulty:** Moderate (case analysis over mutation operators; tedious but not deep).

---

#### C2. SMT Encoding of Contract Verification (LOAD-BEARING)

**Theorem (Encoding Correctness).**
The SMT encoding of "function f satisfies contract (Pre, Post)" is:

  ∃x. Pre(x) ∧ encode(f)(x, y) ∧ ¬Post(x, y)

where encode(f) is the SSA transformation of f's body into a quantifier-free
formula. This encoding is:
(a) *Equisatisfiable* with the negation of the Hoare triple {Pre} f {Post}.
(b) For loop-free code: exact (no approximation).
(c) For code with loops unrolled to depth k: a sound under-approximation of
    the negation (if SAT, the contract is truly violated; if UNSAT, the
    contract holds up to depth k).

**Status:** Standard BMC encoding. Well-known.

**Novelty:** None. Included because the system literally cannot work without it.

**Difficulty:** Straightforward.

---

#### C3. SyGuS Constraint Construction from Mutation Outcomes (LOAD-BEARING)

**Theorem (CEGIS Constraint Correctness).**
The SyGuS problem for synthesizing postcondition Post given precondition Pre,
function f, and killed mutants {m₁,…,mₖ} is:

  Find Post ∈ G_M such that:
  (1) ∀x,y. Pre(x) ∧ y = f(x) ⟹ Post(x,y)          [soundness]
  (2) ∀i. ∃x,y. Pre(x) ∧ y = mᵢ(x) ∧ ¬Post(x,y)    [mutation killing]
  (3) Post is ⊑-maximal subject to (1) and (2)         [minimality]

The CEGIS loop:
- Propose candidate Post from G_M
- Check (1) via SMT: encode(f) ∧ Pre ∧ ¬Post — if SAT, return CE
- Check (2) via SMT for each i: encode(mᵢ) ∧ Pre ∧ Post — if UNSAT, strengthen
- Repeat until fixed point

This loop terminates when G_M is finite (guaranteed for bounded grammars).

**Status:** Standard CEGIS framework applied to mutation constraints.

**Novelty:** The specific constraint formulation (especially condition (2) as a
mutation-discrimination constraint) is new.

**Difficulty:** Straightforward.

---

#### C4. Compositional Verification via Contract Chaining (LOAD-BEARING)

**Theorem (Modular Verification Soundness).**
Given functions f₁,…,fₙ with inferred contracts (Preᵢ, Postᵢ), and a caller
g that invokes f₁,…,fₙ, the *modular verification* of g proceeds as:

Replace each call fᵢ(eᵢ) in g's body with:
  assert Preᵢ(eᵢ); havoc result; assume Postᵢ(eᵢ, result);

If the Hoare triple {Pre_g} g_abstracted {Post_g} is valid, then the Hoare
triple {Pre_g ∧ ⋀ᵢ (Preᵢ ⟹ Postᵢ is valid for fᵢ)} g {Post_g} is valid
for the concrete program.

**Status:** This is standard modular verification (Boogie/Dafny-style).

**Novelty:** None. Essential infrastructure.

**Difficulty:** Straightforward.

---

#### C5. Precondition Inference via Abduction (LOAD-BEARING)

**Theorem (Precondition Abduction from Mutation Failures).**
Given function f, postcondition Post (already synthesized), and mutants
that are killed only *on some inputs*, the precondition Pre can be inferred via
abduction:

  Find Pre such that: Pre ∧ encode(f) ⊨ Post

This is the *abduction problem*. When the candidate comes from a SyGuS grammar
G_Pre, the problem reduces to:

  Find Pre ∈ G_Pre such that ∀x,y. Pre(x) ∧ y = f(x) ⟹ Post(x,y)

with the additional constraint that Pre is *maximal* (weakest precondition
within the grammar) to avoid over-restricting the domain.

**Status:** Abductive inference is well-studied (Dillig et al., PLDI 2012).
The mutation-guided version uses mutation survival on the boundary to identify
inputs where the precondition should exclude.

**Novelty:** The use of mutation boundaries to guide abduction is new.

**Difficulty:** Moderate.

---

#### C6. Fixpoint Characterization of Iterative Refinement (NICE-TO-HAVE)

**Theorem (Convergence of Specification Refinement).**
The iterative process:
1. Run mutation analysis → get killed/survived sets
2. Synthesize spec φₙ from killed set
3. Use φₙ to generate new tests (targeting surviving mutants)
4. Update killed/survived sets, goto 1

converges in finitely many steps to a fixpoint specification φ* when:
- The grammar G_M is finite
- The specification lattice restricted to G_M is finite
- Each iteration strictly strengthens the spec (kills at least one new mutant)

At the fixpoint, every surviving mutant is either equivalent or requires
predicates outside G_M to distinguish.

**Status:** Standard fixpoint argument over a finite lattice.

**Novelty:** The specific iterative scheme is new.

**Difficulty:** Straightforward.

---

#### C7. Differential Specification for Bug Detection (LOAD-BEARING)

**Theorem (Latent Bug Detection via Specification Comparison).**
Let φ_inferred be the mutation-inferred specification and φ_declared be a
user-declared specification (or a specification from documentation/types).

If φ_declared ⊑ φ_inferred is not valid (the declared spec is not implied by
the inferred one), then there exist inputs x such that φ_declared(x) holds
but ¬φ_inferred(x) — these are potential latent bugs (behaviors the developer
intended to specify but that tests do not enforce).

Conversely, if φ_inferred ⊑ φ_declared is not valid, the tests enforce
behaviors beyond the declared spec — these may indicate over-testing or
undocumented invariants.

**Status:** Specification comparison/differencing is a known technique. The mutation-
based instantiation and the bug-detection interpretation are new.

**Novelty:** New framing with practical implications.

**Difficulty:** Straightforward.

---

### Summary Table: Framing C

| ID  | Statement | New? | Difficulty | Load-bearing? |
|-----|-----------|------|------------|---------------|
| C1  | Grammar Sufficiency | Yes | Moderate | YES |
| C2  | SMT Encoding Correctness | Known | Straightforward | YES |
| C3  | CEGIS Constraint Construction | Partially | Straightforward | YES |
| C4  | Modular Verification | Known | Straightforward | YES |
| C5  | Precondition Abduction | Partially | Moderate | YES |
| C6  | Refinement Convergence | Application | Straightforward | No |
| C7  | Differential Bug Detection | Partially | Straightforward | YES |

---

## Cross-Cutting Analysis

### Most Novel Mathematical Contribution

**Theorem A3 (Mutation Completeness ⇒ Specification Tightness)** is the deepest
new result. It provides the foundational justification for the entire approach by
formally connecting mutation adequacy (a testing concept) to specification strength
(a verification concept). No prior work establishes this bridge.

The key open sub-problem: *characterize ε-completeness of standard mutation operator
sets for standard specification languages.* This is adjacent to results in
descriptive complexity (characterizing which properties are expressible in which
logics) but in a synthesis-flavored setting.

**Runner-up:** The Galois connection (A6) to abstract interpretation. If fully
developed, this could yield a new abstract domain for specification mining —
potentially unifying mutation testing, specification inference, and abstract
interpretation into a single framework.

### Deepest Connections to Existing PL Theory

1. **Abstract Interpretation (Cousot & Cousot).** Theorem A6 directly. The
   mutation outcome lattice is an abstract domain; specification inference is
   abstract interpretation over this domain. This could be developed into a
   full abstract interpretation of "test adequacy" — a significant theoretical
   contribution.

2. **Type Theory / Refinement Types.** The inferred specifications are essentially
   refinement types (e.g., {x:Int | x > 0} → {y:Int | y > x}). The SyGuS
   grammar G_M determines the refinement language. Connection to Liquid Types
   (Rondon et al., PLDI 2008): our mutation-driven approach can be seen as an
   alternative to the type-inference–based approach for discovering refinements,
   where mutations play the role of type-error witnesses.

3. **Program Logic (Hoare Logic, Separation Logic).** Theorems B3 and C4 directly.
   The compositional approach is standard Hoare-logic modularity. The mutation-
   specific contribution is showing that mutation-local reasoning composes
   correctly — essentially, that mutation analysis respects the frame rule.

4. **Learning Theory (Angluin, Dana).** Theorem A1's mutation-specification duality
   is structurally similar to learning from equivalence queries and membership
   queries. Killed mutants are negative examples; the original program on test
   inputs provides positive examples. The specification is the learned concept.
   Connection to exact learning: if we view the SyGuS solver as a learner and
   mutation outcomes as an oracle, we can derive PAC-style bounds on specification
   quality.

### Impossibility Results and Fundamental Limits

**Limit 1: The Equivalent Mutant Barrier (from A4).**
Deciding whether a surviving mutant is equivalent to the original program is
undecidable (Rice's theorem). Consequence: *no algorithm can guarantee that
the inferred specification is the strongest possible sound specification.*
The best we can achieve is the strongest specification that the test suite's
killed mutants support.

**Limit 2: Grammar Expressiveness Ceiling.**
If the SyGuS grammar G_M cannot express the "true" specification, the system
will produce a strictly weaker approximation. There is no way to detect this
from within the system (the grammar's limitations are invisible to the CEGIS
loop). This is a *fundamental design choice*, not a bug.

**Limit 3: Bounded Verification Incompleteness (from B5).**
For programs with unbounded loops/recursion, bounded model checking cannot
verify contracts in general. The gap between "verified to bound k" and "truly
valid" is irreducible without additional invariant inference.

**Limit 4: Mutation–Specification Expressiveness Mismatch.**
Even with ε-complete mutation operators, if the specification language Spec
is more expressive than the Boolean closure of mutation error predicates, there
exist valid specifications that no mutation analysis can discover. Formally:
the image of σ (Theorem A2) may be a strict sub-lattice of Spec.

**Limit 5: Compositional Precision Loss.**
Modular verification (C4) introduces imprecision at call boundaries (the
havoc/assume pattern over-approximates). For deeply nested call chains, this
imprecision compounds. Quantifying this loss requires interprocedural analysis
and is problem-specific.

---

## Dependency Graph of Load-Bearing Math

```
A1 (Mutation-Spec Duality) ──────────────────┐
  │                                           │
  ├── A2 (Lattice Embedding)                  │
  │     │                                     ▼
  │     ├── A3 (Completeness ⇒ Tightness)   C3 (CEGIS Construction)
  │     │                                     │
  │     └── B2 (Subsumption Reduction)        ├── C1 (Grammar Sufficiency)
  │                                           │
  ├── A4 (Survivor Characterization)          ├── C5 (Precondition Abduction)
  │     │                                     │
  │     └── C7 (Differential Bug Detection)   └── C2 (SMT Encoding)
  │                                                 │
  ├── B1 (Abstract Mutation) ◄────────────────      │
  │                                                 ▼
  ├── B3 (Compositional Frame) ──────────────► C4 (Modular Verification)
  │
  └── B4 (WP Differencing) ──────────────────► [SMT Solver Backend]

B5 (Bounded Completeness) ──────────────────► C2 (SMT Encoding)
```

Foundation: A1 → A2 → A3 is the theoretical spine.
Scaling: B1–B4 make it practical.
Systems: C1–C5 + C7 make it an actual tool.
A3 + C1 together ensure the system infers *meaningful* specifications.

---

## Verdict: What a POPL Reviewer Wants to See

A POPL-caliber paper from this work would center on:

1. **Theorems A1–A4** as the core theoretical contribution (the mutation–specification
   correspondence, formalized).
2. **Theorem A3** as the headline result, with a concrete proof for QF-LIA and
   standard mutation operators.
3. **Theorem A6** (Galois connection) as the "deep insight" connecting to the
   abstract interpretation tradition.
4. **The impossibility results** (equivalent mutant barrier, grammar ceiling) as
   evidence of intellectual honesty and understanding of limits.

The scaling math (Framing B) and systems math (Framing C) support an
implementation paper (PLDI/OOPSLA) but would be secondary in a POPL submission.

**Estimated effort to formalize:** ~6–9 months for a strong PhD student. A3 is the
hardest; the rest follows from standard techniques applied carefully.
