# Guard-Harvesting CEGAR for Automated Refinement Type Inference in Python

## Problem and Approach

Python's type checkers—mypy, Pyright, Pytype—verify that a variable is an `int` or `str | None`, but they cannot verify that an index is within bounds, a divisor is nonzero, or a `None` check is sufficient to protect all downstream uses. The result is that `IndexError`, `ZeroDivisionError`, and `AttributeError: NoneType` survive type checking and reach production. These are not exotic bugs: they are the #1, #3, and #5 most common runtime exceptions in Python crash telemetry. Developers compensate with ad-hoc assertions and defensive coding, but these are neither machine-checked nor compositional. **The gap between what Python's type system can express and what its toolchain can verify is where silent data corruption and runtime crashes live.**

We propose a system that **automatically proves your array accesses are in-bounds, your divisions are non-zero, and your None checks are sufficient—requiring zero developer annotation.** The system outputs machine-checkable contracts like `{x: int | 0 ≤ x < len(arr)}` that go beyond what any existing Python type checker can express.

The key insight is counterintuitive: **dynamically-typed languages are paradoxically better suited to automated refinement type inference than statically-typed ones.** In Haskell or ML, invariants like "this value is positive" are implicit in the type system and must be discovered from scratch. In Python, the programmer *already wrote* the predicate: `if x > 0:`, `if isinstance(x, int):`, `if x is not None:`, `if i < len(arr):`. These runtime guards are explicit predicates that our system **harvests** as seed predicates for a counterexample-guided abstraction refinement (CEGAR) loop, dramatically shrinking the predicate search space compared to inferring refinements from scratch. The CEGAR loop operates over a formally defined predicate template language **P** encompassing integer arithmetic with `len`, type-tag tests (`isinstance`), and nullity checks. This language is expressive enough to capture the three dominant guard patterns in real Python code while remaining decidable for SMT solving.

The system combines three techniques not previously composed for dynamic languages: (1) abstract interpretation over a reduced product of numeric, type-tag, and nullity domains, producing candidate invariants that seed the CEGAR loop; (2) Liquid-style Horn clause solving over predicate variables, extended to handle dynamic dispatch and mutable state; and (3) **predicate-sensitive incremental re-analysis**, where the unit of invalidation is not a function but a *(function, inferred contract)* pair—callers are re-analyzed only when a callee's contract *changes*, not merely when its implementation does. This granularity, strictly finer than any existing incremental analyzer (Infer operates at procedure level; mypy and Pyright at file level), is what makes the system practical as a CI-integrated tool rather than a batch analysis that runs overnight.

The output is a set of per-function, machine-checkable refinement contracts—emitted as `.pyi` stubs with `Annotated[int, Gt(0)]` syntax, SARIF diagnostics for CI, and counterexample traces when verification fails. The system targets commodity laptop CPUs with fully automated evaluation: no GPU, no cloud cluster, no human annotation. The paper contribution is Python-only, evaluated head-to-head against Pyright, mypy, Mopsa, CrossHair, and Infer on real-world codebases.

**Modeled semantics and honest limitations.** The system analyzes a precisely characterized subset of Python: no `eval`/`exec`, no metaclasses, no monkey-patching of builtins, no C extensions. Functions using these features are marked "outside modeled semantics" and reported with no contract. We acknowledge one existential risk: CEGAR convergence. The loop is guaranteed to terminate (the predicate lattice over P has finite height bounded by O(n² · |T|) where n is the number of in-scope variables and |T| the number of type tags), but the bound is loose—real convergence depends on whether interpolant projection onto P preserves enough information to make progress. If the system resolves fewer than 60% of functions in popular Python repositories within a per-function time budget, the contribution reframes as "refinement type inference for a characterized subset of Python idioms." We measure and report this rate explicitly as a precision-recall curve.

---

## Value Proposition

**The problem is concrete:** Python's type checkers cannot express "this index is within bounds." `IndexError` is the most common runtime exception in Python. Our system catches it statically, with zero annotations, on codebases that haven't even adopted mypy—because it requires no type annotations at all.

**Five bug classes that survive type checking and that we target:**

1. **Array/list out-of-bounds** (`IndexError`): `arr[i]` where `i ≥ len(arr)`. mypy/Pyright see `int` and approve.
2. **None dereference** (`AttributeError`): `x.method()` where `x` may be `None` after an insufficient guard. Pyright handles simple cases; we handle cross-function and conditional paths.
3. **Division by zero** (`ZeroDivisionError`): `a / b` where `b` is not provably nonzero. No Python type checker addresses this.
4. **Type-tag confusion** (`TypeError`): Passing an `int` where a `str` is expected after a branch that doesn't narrow enough. Pyright handles single-function cases; we handle interprocedural narrowing.
5. **Off-by-one in range/slice** (silent data corruption): `range(1, n)` vs. `range(0, n)`. No type checker can distinguish these because the types are identical.

**Who benefits most:**

- **Any Python developer** who writes `if i < len(arr):` guards manually. Our system verifies these guards are correct and sufficient—or shows you a concrete counterexample where they fail.
- **Library maintainers** who field bug reports rooted in value-dependent API misuse. Inferred refinement signatures serve as executable documentation: not "this function takes an int" but "this function takes `{n: int | 0 < n ≤ len(self.columns)}`."
- **Teams under compliance pressure** (SOC 2, HIPAA) who need evidence of defect-class absence. Machine-checkable refinement contracts, emitted as CI artifacts, replace manual code audits.

**Why not LLMs?** Neural type predictors produce probabilistic guesses. Our inferred contracts are *sound*: if the system says `x > 0`, it holds on all execution paths reaching that point. Soundness—with respect to the precisely characterized modeled subset—is the differentiator that no LLM can provide.

**Why not runtime checking?** Tools like `beartype`, `jaxtyping`, and `typeguard` check at runtime, requiring test coverage to trigger. Our system checks all paths statically, finding bugs in untested code paths.

---

## Technical Difficulty

### Subsystem Breakdown

The system is a compiler-scale engineering effort. No individual subsystem is novel in isolation—SSA construction exists (Pytype), abstract domains exist (Astrée/Mopsa), SMT encoding is ubiquitous, CEGAR is textbook. What does not exist is a system that composes all four with predicate-sensitive incremental invalidation to produce value-level refinement contracts for unannotated Python. The primary difficulty is not in any single component but in the **interaction constraints** between them. The predicate language P must simultaneously: (a) be expressive enough that >85% of harvested guards project onto it (coverage), (b) have a decidable SMT theory (Theorem A), (c) induce a finite lattice with tractable height (Theorem C), and (d) support efficient entailment checking for incremental contract-diff (Theorem B). These four constraints are in tension: expressiveness pulls toward undecidability; decidability pulls toward triviality; finite height pulls toward coarse predicates; efficient entailment pulls toward simple theories. The design of P—and the proof that it satisfies all four constraints simultaneously on real Python code—is the central technical challenge.

We present LoC estimates at two scopes: **paper scope** (the artifact supporting all paper claims) and **vision scope** (the full long-term system). The paper scope is the commitment; the vision scope is future work.

| # | Subsystem | Paper LoC | Vision LoC | Notes |
|---|-----------|-----------|------------|-------|
| 1 | **Python SSA Compiler** | 10K | 22K | Uses Python `ast` module for parsing. Adds predicate-preserving IR pass: every `isinstance`, `None` check, comparison becomes a named predicate the CEGAR loop references. Paper scope covers Python 3.10–3.12 core: exception control flow, comprehension scoping, `*args`/`**kwargs` packing. Vision adds match statements (PEP 634), full type stub resolution. |
| 2 | **Refinement IR** | 6K | 8K | Preserves refinement-relevant semantics: predicate guards as first-class constructs, type-tag lattice annotations, source provenance. Includes well-formedness checking, serialization for incremental caching. |
| 3 | **Abstract Domain Library** | 14K | 26K | Paper scope: numeric intervals (5K), type-tag lattice (3K), nullity/optionality (2K), reduced product with 3 pairwise reductions (4K). Vision adds octagons, dictionary shape domain, and 6 pairwise reductions. **The reduced product is the single hardest component**—one incorrect reduction produces either unsoundness or precision collapse. |
| 4 | **Predicate Inference & CEGAR Engine** | 16K | 22K | Guard-harvesting template miner (3K), Liquid constraint generation with κ-variables (5K), CEGAR loop controller with time budgets and graceful degradation (4K), worklist-based fixed-point solver (4K). |
| 5 | **SMT Encoding & Z3 Interface** | 8K | 14K | Paper scope: IR→SMT encoding for QF_UFLIA (3K), Z3 session management (2K), Craig interpolation with fallback (2K), query caching (1K). Vision adds crash recovery, batching, alternative solver backends. |
| 6 | **Incremental Dependency Tracker** | 12K | 18K | Predicate-labeled dependency graph (4K), contract-diff via SMT entailment (2K), persistent content-addressed store (3K), VCS integration (1K), SCC-aware speculative re-analysis scheduler with rollback (2K). **This is the paper's genuinely novel subsystem.** |
| 7 | **Standard Library Models** | 3K | 12K | Paper scope: top-50 stdlib functions by call frequency (`len`, `range`, `list.__getitem__`, `dict.get`, `int`, `float`, etc.) with refinement signatures. Vision: 150 functions with higher-order models. |
| 8 | **Contract Output & Diagnostics** | 4K | 6K | Counterexample trace rendering (1.5K), `.pyi` stub emission with `Annotated` syntax (1.5K), SARIF output (1K). |
| 9 | **Test & Benchmark Infrastructure** | 5K | 5K | Lattice-law property tests, integration tests on curated Python snippets, regression benchmarks. |
| | **Paper scope total** | **~78K** | | |
| | **Vision scope total** | | **~133K** | |

**What we reuse (0 LoC):** Z3 solver, Python `ast` module, typeshed stubs as input, `tree-sitter` for VCS diff → function mapping.

### The Three Hardest Challenges

1. **The reduced product is where dreams die.** With 3 domains (paper scope), 3 pairwise reductions must propagate information bidirectionally without cycles. `type(x) = int` → numeric domain learns `x ∈ ℤ` → nullity domain learns `x ≠ None`. One imprecise step produces spurious CEGAR counterexamples that waste the time budget. One unsound step certifies buggy code as safe. Estimated: 30–40% of debugging time. (Vision scope adds dictionary shape for 6 pairwise reductions.)

2. **CEGAR convergence on real code.** The predicate space for a 100-line Python function can exceed 10⁶ atomic predicates. The loop must converge in 3–5 iterations to be practical, requiring aggressive relevance filtering and predicate generalization heuristics for which no known optimal algorithm exists.

3. **Predicate-sensitive incremental invalidation.** Determining whether a function's contract changed requires re-analyzing it first—but its analysis depends on its callees' contracts. This chicken-and-egg problem demands SCC-aware speculative execution with rollback, and in the worst case (global contract change) degrades to full re-analysis. The engineering challenge is ensuring the common case (local change, contract unchanged) costs O(1) lookups.

---

## New Mathematics Required

Three theorems form a coherent formal pipeline connected by a shared mathematical object: the **predicate abstraction lattice** L = 2^{P_prog}, where P_prog is the set of all atomic predicates in the template language P constructible from a program's constants, variables, and function symbols. Each theorem is load-bearing—removing any one causes a precisely identified part of the system to lose its soundness, completeness, or termination guarantee.

### Theorem A: Decidability of Refinement Subtyping (coNP-complete)

**Statement.** The refinement subtyping judgment Γ ⊢ {x : τ₁ | φ₁} <: {x : τ₂ | φ₂}—over structural object types with width subtyping, union types, and predicates drawn from P mentioning `hasattr(x, k)` for string-valued k—is decidable when the set of possible keys K is finite (extracted from program text). The decision problem is coNP-complete.

**Why it's load-bearing.** Without decidable subtyping, every subtype check in the system is potentially non-terminating; the CEGAR loop degenerates to an oracle machine; no soundness guarantee is possible.

**Why existing math fails.** Liquid Types (Rondon et al. 2008) reduce subtyping to QF_UFLIA validity, but ML types have no width subtyping and no dynamic keys. Adding open rows with `hasattr` introduces an implicit universal quantifier that QF_UFLIA cannot express. Occurrence typing (Tobin-Hochstadt & Felleisen 2008) handles type narrowing but not arithmetic refinement predicates.

**Proof strategy.** Skolemization over the finite key domain K: the universal quantifier "for all keys k in τ₂" becomes a finite conjunction ∧_{k ∈ K}(hasattr(x, k) → ...), expressible in P, reducing subtyping to QF_UFLIA satisfiability. coNP-hardness by reduction from propositional tautology; coNP membership because the complement (satisfiability of φ₁ ∧ ¬φ₂ in QF_UFLIA) is in NP. Estimated: 2–3 pages.

### Theorem B: Soundness and Completeness of Incremental Maintenance under Stratified Negation

**Statement.** Model whole-program refinement type inference as a stratified Datalog¬ program P where negation encodes well-formedness failures and type-narrowing else-branches. Let P′ = P[Δ] be the program after replacing clauses for changed function bodies. The incremental maintenance algorithm (semi-naïve delta propagation with stratum-respecting invalidation) is sound (every incrementally derived fact is in the minimal model of P′), complete (every fact in the minimal model is derived), and output-sensitive: O(|Δ_out| · poly(|P|)) where |Δ_out| is the symmetric difference between old and new minimal models.

**Structural Lemma (prerequisite).** Function-level updates preserve stratification: negation edges only descend from interprocedural rules (stratum ≥ 1) to intraprocedural facts (stratum 0), and function-level edits modify only stratum-0 facts and stratum-1 rules for the changed function.

**Why it's load-bearing.** Without this result, the incremental engine may silently produce wrong contracts (unsound) or conservatively invalidate everything (collapsing to full re-analysis, destroying CI-speed).

**Why existing math fails.** DRed (Gupta et al. 1993) handles semi-positive Datalog but not stratified negation. Counting algorithms (Motik et al. 2019) handle stratified negation in general but lack output-sensitive bounds and have not been analyzed for the specific clause structure of refinement type inference, where negation arises only from well-formedness checks referencing strictly lower strata.

**Proof strategy.** Induction on strata: stratum-0 (intraprocedural) facts are recomputed from scratch for changed functions; at stratum i > 0, semi-naïve evaluation with invalidation is correct by the standard argument, and no spurious invalidation occurs because higher-stratum negation only references stratum-0 facts. Estimated: 3–4 pages.

### Theorem C: Convergence of Guard-Harvesting CEGAR

**Statement.** The guard-harvesting CEGAR algorithm—initialize with runtime guards mapped to predicates in P, iterate predicate abstraction → subtyping check → counterexample extraction → interpolant projection onto P—terminates in at most |P_prog| iterations.

**Corollary.** |P_prog| = O(n² · |K| · |T|) where n = program variables in scope, |K| = string keys, |T| = type tags. Polynomial-many CEGAR iterations (each involving an NP-hard SMT check).

**Why it's load-bearing.** CEGAR non-convergence is the #1 existential risk identified across all framings. Without a convergence argument, the system may produce no output for arbitrary functions—you cannot distinguish "almost done" from "hopelessly stuck."

**Why existing math fails.** Standard CEGAR convergence (Clarke et al. 2000) targets finite-state model checking; the predicate space for programs is infinite without a syntactic bound. Liquid Types assume programmer-supplied qualifiers with no CEGAR loop. Interpolation-based refinement (McMillan 2003) guarantees interpolant existence but not that the interpolant falls within P.

**Proof strategy.** Finite height of the lattice 2^{P_prog} + strict progress (each iteration adds ≥1 new predicate or terminates) + projection soundness (interpolant projection onto P either yields a separating predicate or signals "unresolvable within P"—a sound incomplete outcome). This is a **conditional convergence result**: convergence is guaranteed when interpolants project onto P; functions with non-projectable interpolants are marked unresolvable and reported honestly. Estimated: 2 pages.

### Connecting Architecture

The three theorems are layers of a single pipeline through the shared lattice L:

| Theorem | Role of L |
|---------|-----------|
| A (Decidability) | L determines the SMT theory for subtyping; decidability of subtyping = decidability of entailment in L |
| B (Incrementality) | L determines dependency-graph granularity; predicate-sensitive invalidation tracks changes at individual predicates in L |
| C (Convergence) | L is the CEGAR search space; convergence = the ascending chain in L stabilizes |

**System Soundness (compositional).** Theorem A ensures every subtyping check is a valid entailment. Theorem B ensures incremental results equal from-scratch results. Theorem C ensures the CEGAR loop either proves safety, finds a bug, or explicitly reports inability—never silently accepts unsafe code. Together: the system is sound within the modeled semantics (no `eval`, no metaclasses, no C extensions, no monkey-patching of builtins) and conditionally complete up to P's expressiveness.

Total novel proof obligation: ~10–12 pages. No theorem requires a genuine breakthrough; the contribution is the **coherent formal architecture** connecting predicate abstraction, CEGAR, and incremental maintenance through L, applied to a domain where no such architecture existed.

---

## Best Paper Argument

This paper earns a best-paper award by demonstrating a counterintuitive structural insight about dynamic languages, backed by a coherent formal architecture and decisive empirical results.

**The intellectual surprise.** Dynamic languages are conventionally viewed as hostile to formal verification—duck typing, runtime dispatch, and mutable state seem to make static reasoning harder. This paper inverts the narrative: **runtime type tests, `None` guards, and comparison checks are explicit predicates that the programmer already wrote, and harvesting them as CEGAR seeds makes the predicate-discovery problem in refinement type inference fundamentally different from statically-typed languages where equivalent invariants must be discovered from scratch.** This is not merely an observation—it yields a formal consequence: the CEGAR loop's convergence guarantee (Theorem C) relies on the abundance of harvestable guards to make the initial predicate set Q₀ large enough that few additional iterations are needed.

**Two genuinely novel contributions.** First, predicate-sensitive incremental invalidation (Theorem B) refines the granularity of incremental program analysis from procedures to *(procedure, inferred specification)* pairs, with provable output-sensitive complexity bounds under stratified Datalog negation—a maintenance granularity not previously formalized or implemented. Second, the guard-harvesting CEGAR algorithm is the first automated refinement type inference system for a dynamically-typed language, requiring no programmer annotations and producing sound contracts.

**Theory-practice unity.** The paper presents three interlocking theorems (decidability, incrementality, convergence) that are each load-bearing for a working system evaluated on real codebases. The predicate abstraction lattice L = 2^{P_prog} is the shared mathematical object that makes the theorems cohere—a reviewer can trace from the convergence guarantee (Theorem C) through the subtyping decision procedure (Theorem A) to the incremental maintenance algorithm (Theorem B) and see a single formal architecture. This is not a theory paper with a toy implementation, nor a systems paper with no formal content.

**Empirical falsifiability.** The evaluation produces numbers that either validate or kill the thesis. There is no subjective evaluation, no user study, no cherry-picked examples. Results are reported as precision-recall curves across confidence tiers, with explicit CEGAR convergence rates and incremental speedup distributions. If the system finds real bugs in popular Python packages that all five comparators miss, with quantified precision-recall tradeoffs, the result speaks for itself.

**Positioning precision.** The paper precisely distinguishes itself from every relevant competitor:
- vs. **Liquid Haskell / RSC**: those systems require annotations and target statically-typed languages; we infer predicates automatically for dynamically-typed Python.
- vs. **Mopsa**: abstract interpretation for Python that does not produce machine-checkable refinement contracts; we bridge abstract interpretation and refinement typing via CEGAR.
- vs. **CrossHair**: symbolic execution that *checks* programmer-written contracts but does not *infer* them; our contribution is inference.
- vs. **Infer**: separation-logic analysis with procedure-level incrementality; our predicate-sensitive incrementality is strictly finer-grained.
- vs. **Pyright / mypy**: type-level narrowing (`int`, `str | None`) vs. value-level refinements (`{x: int | x > 0 ∧ x < len(arr)}`); a qualitative difference in guarantee strength.

---

## Evaluation Plan

All experiments are fully automated—no human annotation, no manual labeling, no subjective judgment. Every metric is computed by script and reproducible from a single command.

### Benchmarks

- **Primary corpus:** Top 50 most-downloaded PyPI packages by monthly download count, filtered to those with ≥10K LoC of pure Python (no C extensions in hot paths). Expected: ~30 qualifying packages covering NumPy-using scientific code, web frameworks (Flask/Django utilities), data processing (Pandas helpers), and CLI tools.
- **Secondary corpus:** 50 most-starred Python repositories on GitHub with CI passing, covering a broader range of idioms.
- **Bug oracle:** Known CVEs and closed bug-report issues tagged as crashes/data-corruption in the target packages, plus synthetically injected mutations (array-bounds violations, None dereferences, division-by-zero) for recall measurement.

### Metrics

Results are reported as **precision-recall curves across confidence tiers**, not single-point thresholds. Three numbers are always reported together: (a) false-positive rate on the analyzable subset (functions where CEGAR converges and all callees have resolved contracts), (b) fraction of functions analyzable, and (c) false-positive rate on the full corpus.

| Metric | Measurement | Success Threshold |
|--------|-------------|-------------------|
| **True bugs found** | Warnings on unmutated code confirmed by automated matching against CVE/issue databases and by automated differential testing (inject known fix → warning disappears; revert → warning reappears). A manually validated holdout of 20 cases calibrates the automated method. | ≥10 previously-unknown or independently-confirmed real bugs across corpus |
| **False-positive rate (analyzable)** | Warnings on analyzable functions not confirmed as bugs, divided by total warnings on analyzable functions | <10% on the primary corpus |
| **Analyzable fraction** | Fraction of functions where the CEGAR loop terminates within a 10-second budget and all callees have resolved contracts | >60% of functions across corpus |
| **Incremental speedup** | Ratio of full-analysis time to incremental re-analysis time, measured on the last 100 commits of each repo. Reported as full distribution (median, p90, p99). | Median ≥10× speedup over full re-analysis |
| **Predicate coverage** | Fraction of runtime guards in source code that are captured by the predicate language P. Measured by AST analysis on the full corpus as an independent empirical contribution. | >85% of `isinstance`, `None`, and comparison guards |
| **Head-to-head comparison** | Bugs found by our system that Pyright, mypy, Mopsa, and CrossHair each individually miss | ≥5 bugs missed by every comparator |

**Graceful degradation budget.** We report function coverage at 1s, 5s, 10s, and 30s per-function budgets to characterize the latency-coverage tradeoff.

### Comparators

| Tool | What It Represents |
|------|--------------------|
| **Pyright** | State-of-the-art industrial type checker with flow-sensitive narrowing and incremental analysis |
| **mypy** | Most widely-deployed Python type checker |
| **Mopsa** | Research abstract interpreter for Python (closest academic competitor) |
| **CrossHair** | SMT-backed contract checker for Python (checks but does not infer) |
| **Infer** | Industrial separation-logic analyzer with procedure-level incrementality (architecture competitor) |

### Ablation Studies

1. **Guard harvesting off:** Initialize CEGAR with empty predicate set instead of runtime guards. Measures the contribution of the key insight.
2. **Predicate-sensitive incrementality off:** Fall back to function-level invalidation (à la Infer). Measures the contribution of the novel subsystem.
3. **Domain ablation:** Remove each abstract domain (numeric, type-tag, nullity) one at a time. Measures per-domain contribution to precision.

### Minimum Viable Evaluation

If time constraints prevent the full 50-repo evaluation, the minimum viable evaluation comprises: (a) 10 popular PyPI packages with known bug histories, (b) FP rate as precision-recall curve across confidence thresholds, (c) incremental speedup on 100 real commits mined from those packages, (d) CEGAR convergence rate with per-function breakdown. This minimum is sufficient for all empirical claims.

---

## Laptop CPU Feasibility

The system is designed for commodity hardware. Every architectural decision enforces this constraint.

**Target hardware.** Apple M-series or Intel i7/i9 laptop, 16–32 GB RAM, no GPU, no network access during analysis. All evaluation numbers reported on a single specified machine.

**Why it fits.**

1. **SMT queries are small for most functions.** Each verification condition involves a single function's predicates over P—typically 5–20 atomic predicates, 2–8 program variables. Z3 solves these in milliseconds. The predicate language P is deliberately restricted (no quantifiers, no string operations beyond equality, no heap predicates) to stay within Z3's fast path (QF_UFLIA). For complex functions (100+ lines, 10+ branch points), individual queries may take seconds; the 10-second per-function budget provides headroom for 3–5 CEGAR iterations on such functions.

2. **Incrementality bounds memory.** The persistent store caches SSA IR, abstract states, and inferred contracts on disk. On each commit, only changed functions and their (predicate-sensitively) invalidated dependents are loaded into memory. For a typical commit touching 3–5 functions, working-set memory is <500 MB even on 100K+ LoC codebases.

3. **Parallelism is embarrassing.** Independent functions (no mutual recursion) can be analyzed on separate CPU cores. The SCC decomposition identifies the dependency structure; within each topological level, analysis is embarrassingly parallel. An 8-core laptop analyzes 8 independent functions simultaneously.

4. **CEGAR iterations are bounded.** Theorem C bounds iterations at |P_prog|, but in practice guard harvesting provides most predicates in Q₀, and the loop needs 0–3 additional iterations per function. The 10-second per-function budget is enforced by the CEGAR controller; functions exceeding it are marked "unresolvable" and reported.

5. **No training, no model loading.** Unlike LLM-based approaches, the system has zero startup cost beyond loading the cached analysis state from disk. Cold-start full analysis of a 50K-LoC codebase is the worst case; incremental re-analysis on subsequent commits is the steady state.

**Concrete latency targets.** Full analysis of 50K LoC: <30 minutes on laptop. Incremental re-analysis per commit (median case: 3–5 changed functions, contract unchanged): <30 seconds. These targets are achievable because the SMT queries are in a decidable fragment with small variable counts, and predicate-sensitive invalidation limits re-analysis scope.

---

## Slug

`guard-harvest-refinement-types-python`
