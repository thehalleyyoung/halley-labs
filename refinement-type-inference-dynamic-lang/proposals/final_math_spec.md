# New Mathematics Required: Final Specification

## Scope and Framing

This document enumerates the **load-bearing mathematics** for the crystallized problem statement: *automated refinement type inference for Python via guard-harvesting CEGAR with predicate-sensitive incremental re-analysis*. Each item below is something that must be formally developed and proved; removal of any item causes a clearly identified part of the system to lose its soundness or termination guarantee.

**What was cut and why:**
- QF_UFLIA+H heap interpolation (former Theorem 1) is dropped. The target bug classes (array bounds, null/None, division-by-zero, type-tag confusion) do not require separation-logic heap reasoning. The bounded-depth restriction (heap depth ≤ k) is too restrictive for real Python (trees, graphs, recursive structures), and the combination result is a standalone contribution to mathematical logic that does not connect to the system's actual predicate language. It belongs in a future LICS/CAV paper, not here.

---

## I. The Predicate Template Language P

### Definition

The predicate template language **P** is the set of quantifier-free formulas over the following signature:

| Sort | Constants | Interpretation |
|------|-----------|----------------|
| `Int` | integer literals from program text | mathematical integers |
| `Bool` | `True`, `False` | standard |
| `Tag` | a finite set of type-tag names extracted from the program (`int`, `str`, `float`, `list`, `NoneType`, user-defined class names) | uninterpreted, finite domain |
| `Str` | string literals from program text | equality only (no concatenation, no regex) |

**Function symbols:**

| Symbol | Sort | Interpretation |
|--------|------|----------------|
| `len(x)` | `Any → Int` | length of sequences/containers |
| `isinstance(x, T)` | `Any × Tag → Bool` | runtime type-tag test |
| `is_none(x)` | `Any → Bool` | `x is None` |
| `is_truthy(x)` | `Any → Bool` | Python truthiness (`bool(x)`) |
| `hasattr(x, k)` | `Any × Str → Bool` | attribute/key presence |
| standard arithmetic | `Int × Int → Int` | `+`, `-`, `*`, `//`, `%` |
| comparisons | `Int × Int → Bool` | `<`, `≤`, `=`, `≠`, `≥`, `>` |

**Atomic predicates** are of the form:
```
e₁ ⊲⊳ e₂     where ⊲⊳ ∈ {<, ≤, =, ≠, ≥, >}
               and e ::= x | c | len(x) | e + e | e - e | e * c | e // c | e % c
isinstance(x, T)
is_none(x) / ¬is_none(x)
is_truthy(x) / ¬is_truthy(x)
hasattr(x, k) / ¬hasattr(x, k)
```

**Refinement predicates** are Boolean combinations (∧, ∨, ¬) of atomic predicates.

### Why P Is Sufficient

**Claim.** P captures the guard conditions that Python programmers actually write in the top-100 PyPI packages.

**Argument (empirical, to be validated):** A survey of runtime guards in Python code reveals four dominant patterns:
1. **Comparison guards:** `if i < len(arr)`, `if n > 0`, `if x != 0` → captured by arithmetic comparisons over `Int` with `len`.
2. **Type-tag guards:** `if isinstance(x, int)`, `if type(x) is str` → captured by `isinstance(x, T)`.
3. **Nullity guards:** `if x is not None`, `if x` → captured by `is_none(x)` and `is_truthy(x)`.
4. **Structural guards:** `if hasattr(obj, 'field')`, `if 'key' in d` → captured by `hasattr(x, k)`.

**What P does NOT capture:** string operations beyond equality, floating-point arithmetic, set/dict comprehension predicates, predicates involving `eval` or dynamic attribute names, higher-order predicates ("for all elements in the list..."). These are explicitly out of scope. Functions whose correctness depends on such predicates will receive the trivial refinement `{x : τ | true}` — a sound but uninformative result.

### Why P matters

Without a fixed, finite predicate template language:
- The CEGAR loop has no finite lattice to ascend → no termination guarantee.
- The SMT encoding has no decidable theory to target → solver divergence.
- The incremental engine has no bounded predicate space to diff → invalidation is unsound.

---

## II. Theorem A: Decidability and Complexity of Refinement Subtyping (coNP-complete)

### Statement

Let **τ** range over structural object types with width subtyping:
```
τ ::= {k₁: τ₁, …, kₙ: τₙ, …}   (open row, with row variable)
    | Int | Str | Bool | None | List[τ] | τ₁ → τ₂
    | τ₁ ∪ τ₂                     (union types)
```

Let refinement types have the form `{x : τ | φ}` where `φ ∈ P`.

**Theorem.** The refinement subtyping judgment

> Γ ⊢ {x : τ₁ | φ₁} <: {x : τ₂ | φ₂}

is decidable when:
- (a) τ₁ <: τ₂ in the structural subtyping order (width subtyping: supertypes may have fewer fields),
- (b) predicates φ₁, φ₂ are drawn from P,
- (c) the set of possible keys K is finite (extracted from program text), and
- (d) `hasattr` predicates range over K.

The decision problem is **coNP-complete**.

### Why It's Load-Bearing

Without this result:
- The type checker cannot decide whether an inferred refinement type is a valid subtype of an expected type at a call site. Every subtyping check becomes a potentially non-terminating SMT query.
- The CEGAR loop cannot verify or refute candidate refinements — it degenerates to an oracle machine.
- The system cannot provide any soundness guarantee whatsoever: it might accept ill-typed programs or reject well-typed ones.

### Why Existing Math Is Insufficient

1. **Liquid Types (Rondon et al. 2008, Vazou et al. 2014):** Subtyping reduces to SMT validity in QF_UFLIA. This works because ML types have no width subtyping and no dynamic keys. Adding open rows with `hasattr` introduces an implicit universal quantifier over the key domain that QF_UFLIA cannot express.

2. **Occurrence typing (Tobin-Hochstadt & Felleisen 2008):** Handles type narrowing through runtime tests but does not produce refinement-level guarantees (no arithmetic predicates, no `len` reasoning). The subtyping relation is over type-tag unions, not predicate-decorated types.

3. **TypeScript's structural subtyping:** Decidable (in practice, modulo Turing-completeness of conditional types), but has no refinement predicates. Adding predicates to TypeScript's type system is a new problem.

4. **Set-theoretic types (Castagna, Frisch, etc.):** Handle unions, intersections, and negation of types, but the "types" are sets of values characterized by type tags, not by arithmetic predicates. Extending to refinements is non-trivial.

### Proof Strategy

The key insight is **Skolemization over the finite key domain K**.

1. Width subtyping introduces the judgment: "for all keys k in τ₂, if hasattr(x, k) then the field type at k in τ₁ is a subtype of the field type at k in τ₂."
2. Since K is finite (extracted from program text by a pre-pass), the universal quantifier becomes a finite conjunction: ∧_{k ∈ K} (hasattr(x, k) → ...).
3. This finite conjunction is expressible in P, reducing the subtyping check to a QF_UFLIA satisfiability query (is `φ₁ ∧ ¬φ₂` satisfiable?).
4. coNP-hardness: reduction from propositional tautology checking (straightforward — encode propositional variables as integer comparisons).
5. coNP membership: the complement problem (is `φ₁ ∧ ¬φ₂` satisfiable?) is in NP because P predicates live in QF_UFLIA, which has NP satisfiability.

### Difficulty Assessment

**Straightforward extension.** The Skolemization argument is clean and self-contained. The main technical obligation is the pre-pass that extracts K from the program text and the proof that K is indeed finite and complete (i.e., no dynamically-generated keys outside K affect the subtyping judgment). The latter requires defining a notion of "key-completeness" for the analyzable subset of Python. Estimated effort: 2–3 pages of proof, no novel proof techniques needed.

---

## III. Theorem B: Soundness and Completeness of Incremental Fixed-Point Maintenance under Stratified Negation

### Statement

Model the whole-program refinement type inference as a **stratified Datalog¬ program** P:
- **Base facts:** predicate valuations from intraprocedural analysis of each function body.
- **Rules:** interprocedural propagation (caller–callee contract matching), predicate implication, type-tag flow.
- **Negation:** appears in well-formedness checks ("predicate φ does NOT hold at call site c, so callee's precondition is violated") and in type-narrowing ("x is NOT an instance of T on the else-branch").

Let P′ = P[Δ] be the program obtained by replacing the base facts and rules for a set of changed function bodies.

**Theorem.** The incremental maintenance algorithm (semi-naïve delta propagation with stratum-respecting invalidation) satisfies:

1. **Soundness:** Every fact in the incrementally maintained model M′_inc is in the minimal model M′ of P′.
2. **Completeness:** Every fact in M′ is in M′_inc.
3. **Complexity:** The algorithm runs in time O(|Δ_out| · poly(|P|)), where |Δ_out| = |M ⊕ M′| is the symmetric difference between the old and new minimal models.

**Structural Lemma (prerequisite).** Function-level updates (replacing the clauses for a single function body) preserve the stratification of P. That is, if P is stratified with stratification σ, then P′ = P[Δ] is stratified with a stratification σ′ that agrees with σ on all unchanged strata.

### Why It's Load-Bearing

Without this result:
- The incremental engine might silently produce wrong results: it could claim a function is safe when a transitive dependency's contract changed in a way that invalidates the safety proof.
- Alternatively, without completeness, the engine might conservatively invalidate everything, collapsing to full re-analysis on every commit — destroying the CI-speed requirement.
- The structural lemma is critical: if a code change breaks the stratification, the incremental algorithm's correctness guarantee evaporates. We need to know this cannot happen for function-level edits.

### Why Existing Math Is Insufficient

1. **DRed (Gupta et al. 1993):** Handles semi-positive Datalog but not stratified negation. Over-deletes and then re-derives, which is sound but not output-sensitive (can be exponentially slower than |Δ_out|).

2. **Counting algorithms (Motik et al. 2019, Hu et al. 2023):** Handle stratified negation in general Datalog¬, but do not provide output-sensitive complexity bounds. Their algorithms are designed for RDF/knowledge-graph workloads and have not been analyzed for the specific clause structure arising from refinement type inference.

3. **The specific problem:** The clause structure of refinement type inference has a particular shape — negation arises from well-formedness checks and type-narrowing, which always reference facts in strictly lower strata (intraprocedural facts). This special structure is what enables the stratification-preservation lemma and the output-sensitive bound. No existing work exploits this structure.

### Proof Strategy

1. **Stratification preservation:** Show that the dependency graph of P has a specific shape: negation edges only go from interprocedural rules (stratum ≥ 1) to intraprocedural facts (stratum 0). Function-level updates only modify stratum-0 facts and stratum-1 rules for the changed function. Since no negation edge is added between strata ≥ 1, the stratification is preserved.

2. **Soundness:** By induction on strata. At stratum 0 (intraprocedural), the updated facts are recomputed from scratch for changed functions — trivially sound. At stratum i > 0, semi-naïve evaluation with invalidation produces exactly the consequences of the new stratum-(i-1) facts under the unchanged rules — sound by the standard semi-naïve correctness argument.

3. **Completeness:** Show that the invalidation phase does not over-delete. Key insight: a derived fact f at stratum i is invalidated only if some fact it depends on (positively or negatively) at stratum < i changed. Since stratum-0 changes are exact (recomputed from scratch) and higher-stratum negation only references stratum-0 facts, no spurious invalidation occurs.

4. **Complexity:** The algorithm touches each changed fact at most poly(|P|) times (bounded by the number of rules that can derive it). Total work is O(|Δ_out| · poly(|P|)).

### Difficulty Assessment

**Moderate — straightforward given the structural lemma, which requires careful argument.** The core incremental Datalog machinery is well-understood. The novel contribution is (a) identifying the stratification structure specific to refinement type inference, (b) proving it is preserved under function-level updates, and (c) deriving the output-sensitive bound from this structure. The structural lemma is the crux — it requires a clean formalization of "what clauses does a function body generate" and a proof that the negation pattern is always stratum-descending. Estimated effort: 3–4 pages of proof.

**Scope limitation:** The structural lemma may fail when a code change alters the call graph in a way that creates a new negation cycle (e.g., function A now calls function B, which has a precondition that negates a predicate derived from A). We conjecture this does not happen in practice for the clause structure we generate, but the formal statement must either (a) prove it for a restricted class of call-graph changes, or (b) detect the failure and fall back to full re-analysis. Option (b) is pragmatically acceptable.

---

## IV. Theorem C: Convergence of Guard-Harvesting CEGAR

### Statement

Define the **guard-harvesting CEGAR algorithm** as follows:

1. **Initialize:** Extract the set of runtime guards G from the program text (isinstance checks, None checks, comparisons, truthiness tests, hasattr tests). Map each guard to a predicate template in P. This yields an initial predicate set Q₀ ⊆ P.
2. **Abstract:** Compute the predicate abstraction of each function body with respect to Q_i (the current predicate set).
3. **Check:** For each function, verify the inferred refinement contract against the function's specification (or against the contracts of its callers/callees). If all checks pass, terminate.
4. **Refine:** For each failed check, extract a counterexample trace. Concretize the trace via SMT. If the trace is feasible (a real bug), report it. If spurious, extract a Craig interpolant from the infeasibility proof. Project the interpolant onto P to obtain new predicates. Set Q_{i+1} = Q_i ∪ {new predicates}.
5. **Repeat** from step 2.

**Theorem.** The guard-harvesting CEGAR algorithm terminates in at most |P_prog| iterations, where P_prog is the set of all atomic predicates in P constructible from the program's constants, variables, and function symbols.

**Corollary.** |P_prog| = O(n² · |K| · |T|) where n is the number of program variables in scope, |K| is the number of string keys, and |T| is the number of type tags. The algorithm terminates in polynomial-many CEGAR iterations (though each iteration involves an NP-hard SMT check).

### Why It's Load-Bearing

Without a convergence argument:
- The CEGAR loop might run forever on some programs, producing no output. This is not a theoretical concern — CEGAR non-convergence on real code is identified as the #1 existential risk by all three framings.
- Without bounding the iteration count, no wall-clock time budget can be meaningfully set: you don't know if the loop is "almost done" or "hopelessly stuck."
- The guard-harvesting initialization is what makes convergence practical: by starting with programmer-written guards, Q₀ already contains most of the predicates needed, so the CEGAR loop typically needs few additional iterations.

### Why Existing Math Is Insufficient

1. **Standard CEGAR convergence (Clarke et al. 2000, 2003):** Proves termination for finite-state model checking where the abstraction lattice is finite. In predicate abstraction for programs, the predicate space is infinite (every subexpression is a candidate). Convergence requires a *syntactic* bound on the predicate language, which standard CEGAR does not provide.

2. **Liquid Types predicate inference:** Assumes the programmer provides the predicate templates (qualifiers). There is no CEGAR loop — the system solves a constraint system over the given qualifiers. Our contribution is automating the qualifier discovery via guard harvesting + CEGAR.

3. **Interpolation-based refinement (McMillan 2003, Henzinger et al. 2004):** The interpolant is guaranteed to exist (by the interpolation theorem for the underlying theory), but there is no guarantee that the interpolant falls within the predicate language P. Our **projection step** — mapping an arbitrary QF_UFLIA interpolant onto the nearest predicate in P — is where the novelty lies. We must show that this projection does not prevent convergence.

### Proof Strategy

1. **Finite height:** The lattice of predicate sets over P_prog, ordered by inclusion, has finite height |P_prog|.
2. **Strict progress:** Each CEGAR iteration either (a) terminates (all checks pass, or a real bug is found) or (b) adds at least one new predicate to Q_i that was not in Q_i (otherwise the same counterexample would be spurious again, contradicting the interpolant's existence).
3. **Projection soundness:** Show that the interpolant projection onto P either yields a predicate in P (progress) or signals that the counterexample cannot be refuted within P (the function is marked "unresolvable within the predicate language" — a sound but incomplete outcome).
4. **Termination:** After at most |P_prog| iterations, either all functions are verified, all bugs are found, or the remaining functions are marked unresolvable.

### Difficulty Assessment

**Straightforward given P is well-defined, with one subtle point.** The finite-height argument is standard. The strict-progress argument requires showing that the interpolant projection onto P is not lossy enough to prevent progress — i.e., that projecting an interpolant I onto the nearest predicate in P yields a predicate that still separates the spurious counterexample. This is **not guaranteed in general**: the interpolant might lie outside P entirely, and its projection might be too coarse. The honest resolution is: (a) prove convergence when the interpolant is already in P (which it is for the common guards), and (b) for the remaining cases, bound the number of "failed projections" and fall back to marking the function as unresolvable. This makes the theorem a **conditional convergence result**, which is scientifically honest. Estimated effort: 2 pages of proof.

---

## V. Connecting Framework: The Predicate-Abstraction–CEGAR–Datalog Pipeline

### The Unifying Mathematical Object

The three theorems above are not independent results — they are layers of a single formal pipeline. The unifying object is the **predicate abstraction lattice** L = 2^{P_prog}, which appears in each theorem with a different role:

| Theorem | Role of L |
|---------|-----------|
| Theorem A (Decidability) | L determines the SMT theory for subtyping checks. Decidability of subtyping = decidability of entailment in L. |
| Theorem B (Incrementality) | L determines the granularity of the dependency graph. Predicate-sensitive invalidation tracks changes at the level of individual predicates in L, not just function-level changes. |
| Theorem C (CEGAR convergence) | L is the search space the CEGAR loop traverses. Convergence = the ascending chain in L stabilizes. |

### How the Pipeline Works

```
Program text
    │
    ▼
┌─────────────────────────┐
│  Guard Harvesting       │  Extract runtime guards → initial predicate set Q₀ ⊆ L
│  (Section IV)           │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Intraprocedural        │  Abstract interpretation of each function body
│  Predicate Abstraction  │  with respect to current Q_i
│  over L                 │  Produces base facts for Datalog encoding
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Interprocedural        │  Stratified Datalog¬ program P
│  Fixed Point            │  Rules encode caller–callee contract matching
│  (Theorem B)            │  Negation encodes well-formedness failures
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Subtyping Checks       │  For each call site: Γ ⊢ τ_actual <: τ_expected
│  (Theorem A)            │  Decidable in coNP via Skolemization over K
└────────────┬────────────┘
             │
        ┌────┴────┐
        │ pass?   │
        ▼         ▼
      Done    ┌─────────────────────────┐
              │  CEGAR Refinement       │  Extract spurious CE, compute interpolant,
              │  (Theorem C)            │  project onto L, add to Q_{i+1}
              └────────────┬────────────┘
                           │
                           └──► loop back to predicate abstraction
```

### On Subsequent Commits (Incremental Mode)

When function bodies change:
1. Recompute intraprocedural facts for changed functions only (new base facts).
2. Apply Theorem B's incremental maintenance to propagate deltas through the Datalog program.
3. For any newly-invalid subtyping checks, re-enter the CEGAR loop locally.
4. Theorem B guarantees the result equals from-scratch re-analysis.

---

## VI. Soundness Guarantees

### What We Guarantee

**Theorem (System Soundness).** If the analysis reports that function f satisfies refinement contract C = {x : τ | φ}, then for all inputs v satisfying f's precondition, f(v) satisfies φ — *with respect to the modeled semantics*.

**Formally:** Let ⟦·⟧ be the denotational semantics of the analyzable subset of Python (no eval, no metaclasses, no C extensions, no monkey-patching of builtins). If the analysis derives Γ ⊢ f : {x : τ | φ₁} → {y : τ₂ | φ₂}, then for all v ∈ ⟦τ₁⟧ with ⟦φ₁⟧(v) = true, we have ⟦φ₂⟧(⟦f⟧(v)) = true.

### What We Do NOT Guarantee

1. **Completeness of bug finding.** Functions marked "unresolvable" may contain bugs that the predicate language P cannot express. This is inherent to predicate abstraction.
2. **Soundness beyond the modeled subset.** Code using `eval`, `exec`, `__getattr__` with dynamic dispatch, ctypes, or metaclass `__new__` overrides is outside the modeled semantics. The analysis is sound only for the analyzable fragment.
3. **Termination of the target program.** Refinement types describe partial correctness, not total correctness.
4. **Floating-point precision.** Arithmetic predicates are interpreted over mathematical integers, not IEEE 754 floats. Programs that depend on float precision may have unsound refinements.

### Soundness Architecture

Soundness is compositional across the three theorems:
- **Theorem A** ensures that every subtyping check the system performs is a valid logical entailment (no false positives from the type checker itself).
- **Theorem B** ensures that the incrementally maintained solution is identical to the from-scratch solution (no false negatives from stale cached results).
- **Theorem C** ensures that the CEGAR loop either finds a proof, finds a bug, or explicitly reports inability — it never silently accepts an unsafe program by failing to terminate.

Together: the system is **sound** (no false negatives within the modeled subset) and **conditionally complete** (complete up to the expressiveness of P and the CEGAR projection step).

---

## VII. Summary: Theorem Inventory

| ID | Statement (informal) | Load-bearing for | Difficulty | Pages (est.) |
|----|----------------------|------------------|------------|--------------|
| **A** | Refinement subtyping with width subtyping + dynamic keys over P is coNP-complete | Every subtyping check in the system | Straightforward extension | 2–3 |
| **B** | Incremental Datalog¬ maintenance is sound, complete, and output-sensitive for the clause structure of refinement type inference | Incremental re-analysis correctness and performance | Moderate (structural lemma is the crux) | 3–4 |
| **C** | Guard-harvesting CEGAR terminates in |P_prog| iterations (conditional on interpolant projectability) | CEGAR loop termination; the #1 existential risk | Straightforward with one subtle projection argument | 2 |
| **Def** | Predicate template language P: definition + sufficiency argument | Everything (P is the foundation of A, B, and C) | Empirical validation + formal definition | 1–2 |
| **Sys** | System soundness (compositional, over modeled subset) | Overall correctness claim | Follows from A + B + C | 1 |

**Total novel proof obligation: ~10–12 pages.**

No theorem requires a genuine breakthrough. Theorem A is a clean Skolemization argument. Theorem B applies known incremental Datalog techniques to a specific clause structure, with a novel structural lemma. Theorem C is a standard finite-lattice convergence argument with an honest caveat about interpolant projection. The contribution is not any single deep theorem but the **coherent formal architecture** that connects predicate abstraction, CEGAR, and incremental maintenance through the shared lattice L = 2^{P_prog}, applied to a domain (Python refinement type inference) where no such architecture previously existed.
