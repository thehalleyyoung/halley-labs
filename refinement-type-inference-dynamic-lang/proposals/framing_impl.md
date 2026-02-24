# Framing: Engineering Difficulty

## Title

**Liquid Loom: A 165K-LoC Incremental Refinement Type Engine for Untyped Languages — and Why Every Line Earns Its Keep**

## Problem and Approach

### The Core Engineering Problem

Building a refinement type inference system for *statically typed* languages is already a PhD-level effort (Liquid Haskell required years of expert development for a single well-typed source language). This project attempts something qualitatively harder: inferring refinement types for Python and TypeScript — languages whose semantics are defined by layers of runtime dispatch, prototype chains, duck typing, operator overloading, and eval-adjacent dynamism. The system cannot lean on an existing typed intermediate representation; it must *construct* one from scratch, faithfully modeling semantics that language specifications leave intentionally vague or implementation-defined. Every downstream component (abstract domains, predicate inference, SMT encoding) depends on the fidelity of this front-end, meaning a bug in SSA lowering doesn't just produce a wrong answer — it can make the SMT solver diverge, the CEGAR loop spin, or the incremental invalidation engine silently miss a dependency.

### Why This Is a Systems Problem, Not Just an Algorithms Problem

The algorithmic pieces — abstract interpretation, Liquid-style predicate inference, CEGAR — are individually well-understood in the literature. The engineering challenge is making them *compose* across two wildly different source languages, *scale* to real codebases (100K+ LoC targets), and *remain incremental* at CI speed. Each subsystem imposes interface contracts on its neighbors: the SSA compilers must produce a common IR that preserves enough source-level information for counterexample visualization but is normalized enough for efficient abstract interpretation; the abstract domain library must be modular (you need to swap numeric domains without touching pointer analysis) yet tightly coupled for reduced-product precision; the SMT encoding must map IR constructs to Z3 theories without blowing up quantifier instantiation; and the incremental engine must track fine-grained dependencies at the predicate level, not just the function level, to avoid catastrophic re-analysis. Getting any one of these interfaces wrong forces a redesign that propagates through 50K+ LoC of downstream code.

### Why 165K LoC Is the Honest Number

A common objection to large-LoC research artifacts is padding: test harnesses inflated with boilerplate, copy-pasted code across language front-ends, over-abstracted framework layers. This system resists that charge because the two source languages share almost nothing at the AST level (Python's grammar and TypeScript's grammar are structurally incompatible), the abstract domains have genuinely distinct implementations for distinct theories (interval arithmetic ≠ string prefix analysis ≠ array shape tracking), and the test/benchmark infrastructure must exercise cross-cutting interactions (e.g., "does CEGAR find the right predicate when a TypeScript function calls a Python module through a FFI boundary stub?"). Removing any subsystem doesn't just lose a feature — it breaks the compositional verification guarantee that is the system's central claim.

---

## Detailed Subsystem Breakdown

### 1. Python Type-Aware AST → SSA Compiler — 25K LoC

| Component | Est. LoC | Justification |
|---|---|---|
| Python parser integration + AST normalization | 5K | Must handle Python 3.8–3.12 grammar evolution, decorators, walrus operator, match statements, comprehension scoping. Cannot use `ast` module output directly — need augmented AST with source locations, inferred scope chains, and type-stub integration. |
| Type stub resolver + typeshed integration | 4K | Must load, parse, and resolve `.pyi` stubs for the standard library and popular third-party packages. Handles overload resolution, generic instantiation, and protocol matching. Without this, every call to `len()`, `dict.get()`, or `os.path.join()` is opaque. |
| SSA construction with Pythonic semantics | 8K | Python's scoping (LEGB), mutable default arguments, `*args`/`**kwargs` packing, exception control flow, generator/async generator suspension points, and `with`-statement desugaring all require bespoke SSA construction passes. Each is a source of subtle bugs. |
| Dynamic dispatch modeling | 4K | `__getattr__`, `__getitem__`, descriptor protocol, metaclasses, MRO linearization. Must produce conservative-but-useful SSA for calls that dispatch through dunder methods. Over-approximate and you lose all precision; under-approximate and you're unsound. |
| Type narrowing / isinstance lowering | 4K | `isinstance` / `issubclass` checks, `type()` comparisons, truthiness narrowing, `None` checks — all must be lowered to SSA phi-nodes with predicate guards that the downstream abstract interpreter can exploit. |

**Why it can't be smaller:** Python's surface syntax is enormous (the grammar has 90+ productions). Every production that can appear in a function body needs an SSA lowering rule. The type stub resolver alone requires a mini type-checker (generic substitution, protocol structural matching) because Python's type system is gradual and stubs use the full `typing` vocabulary.

### 2. TypeScript AST → SSA Compiler — 25K LoC

| Component | Est. LoC | Justification |
|---|---|---|
| TypeScript parser integration + AST extraction | 4K | Must interface with the TypeScript compiler API (`ts.createProgram`) to get fully resolved ASTs with type information. Handles project references, path mapping (`tsconfig` paths), and declaration merging. |
| Type system modeling | 7K | TypeScript's type system is Turing-complete (conditional types, mapped types, template literal types, recursive type aliases). Must model enough of it to resolve call targets and narrow union types. Cannot punt to "any" without losing all refinement precision. |
| SSA construction with JS semantics | 7K | Hoisting, `var`/`let`/`const` scoping, prototype chain reads, optional chaining (`?.`), nullish coalescing (`??`), destructuring assignment (nested, with defaults), `for...of` with iterators, `async`/`await` desugaring — each needs correct SSA representation. |
| Control flow graph for truthiness/narrowing | 4K | TypeScript's control flow analysis is sophisticated: type guards, `in` operator narrowing, discriminated unions, assertion functions. Must replicate these in the SSA CFG so the abstract interpreter sees the same narrowing the programmer expects. |
| Common IR emission + source-map threading | 3K | Both compilers target the same IR. Must carry source locations, original variable names, and type-annotation provenance through to IR level for counterexample rendering. |

**Why it can't share code with Python front-end:** The two languages differ in scoping rules, object models, module systems, type system expressiveness, and control flow semantics. The only shared component is the target IR format and the source-map threading logic (~3K shared). Attempting to unify the front-ends would produce a leaky abstraction that miscompiles both languages.

### 3. Abstract Domain Library — 30K LoC

| Component | Est. LoC | Justification |
|---|---|---|
| Numeric domains (intervals, octagons, polyhedra) | 10K | Intervals are simple but imprecise for relational properties (e.g., `i < len(arr)`). Octagons capture difference constraints. Polyhedra capture general linear constraints. Each needs: lattice operations (join, meet, widen), transfer functions for arithmetic/comparison, and serialization for incremental caching. |
| Pointer / reference domain | 6K | Must model aliasing, field sensitivity, and prototype chains (TS) / attribute dictionaries (Python). Points-to sets must be context-sensitive enough to distinguish `arr1` from `arr2` in the same scope but efficient enough to not explode on large heap graphs. |
| String domain (prefix/suffix, regex membership) | 4K | Needed for type-tag correctness (Python `type(x).__name__`) and TypeScript template literal types. At minimum: constant strings, prefix/suffix abstraction, and length bounds. |
| Array shape domain | 5K | Tracks array lengths, element types, and structural shape (e.g., "2D array of shape [N, 3]"). Essential for bounds checking. Must interact with the numeric domain (array length is a numeric value used in comparisons). |
| Reduced product construction | 5K | The power of abstract interpretation comes from combining domains. The reduced product must propagate information between domains at each program point: "pointer p is non-null" informs the numeric domain that `p.length` is well-defined; "integer i is in [0, len(arr))" informs the array domain that access is safe. Implementing reduction correctly and efficiently is the single hardest piece of the abstract domain library. |

**Why it can't be smaller:** Each domain is mathematically distinct. You cannot implement octagon analysis by parameterizing interval analysis — the data structures (difference-bound matrices vs. interval pairs) and algorithms (incremental closure vs. trivial join) are fundamentally different. The reduced product is additional code on top of the individual domains, not a replacement.

### 4. Predicate Inference and CEGAR Loop — 25K LoC

| Component | Est. LoC | Justification |
|---|---|---|
| Liquid-style predicate abstraction engine | 8K | Given a set of candidate predicates and an abstract state, computes the strongest conjunction of predicates that is implied by the state. Requires efficient implication checking (batched SMT queries), predicate indexing, and fixed-point computation over the CFG with predicate-abstracted states. |
| Predicate template mining | 4K | Automatically generates candidate predicates from the program: comparison expressions, array-index expressions, nullity checks, type-test expressions. Must normalize templates to avoid duplicates and prioritize likely-useful predicates (heuristics matter enormously for performance). |
| CEGAR loop controller | 6K | When verification fails, extracts a counterexample trace from the abstract interpreter, concretizes it via SMT, checks feasibility, and if spurious, extracts new predicates from the interpolant. Must handle: trace infeasibility checking, Craig interpolation via Z3, predicate generalization, and loop termination heuristics. |
| Fixed-point solver (Liquid constraint system) | 7K | Solves Horn-clause-like constraints over predicate variables (κ-variables in Liquid Types terminology). Iterative refinement with widening, qualifiers, and dependency-driven worklist. Must handle recursive functions (requires well-founded termination arguments) and mutually recursive function clusters. |

**Why it can't be smaller:** CEGAR is inherently a complex feedback loop. Each component (predicate mining, abstraction, counterexample analysis, interpolation, re-abstraction) is a distinct algorithm with its own data structures. The fixed-point solver is essentially a custom constraint solver — this is not code you can delegate to Z3, because the constraints are over predicate *variables*, not ground formulas.

### 5. SMT Encoding and Z3 Interface — 15K LoC

| Component | Est. LoC | Justification |
|---|---|---|
| IR → SMT-LIB theory encoding | 6K | Must encode: integer arithmetic, real arithmetic (for floating-point approximation), bitvector operations, array theory (for heap modeling), algebraic datatypes (for tagged unions / discriminated unions), uninterpreted functions (for abstract heap operations). Each IR construct maps to a specific SMT theory fragment; incorrect theory selection causes solver divergence. |
| Z3 binding layer + solver lifecycle management | 4K | Manages Z3 context creation/destruction, incremental assertion stacks (push/pop for CEGAR iterations), model extraction, unsat-core extraction, timeout enforcement, and memory monitoring. Must handle Z3 crashes gracefully (Z3 does crash on certain inputs). |
| Interpolation and abduction engine | 3K | Craig interpolation for CEGAR predicate extraction. Abductive inference for contract strengthening (given pre/post, find missing precondition). Neither is natively exposed by Z3's stable API — requires encoding tricks or use of experimental Z3 features with fallback. |
| Query caching and incremental SMT | 2K | SMT queries are the performance bottleneck. Must cache query results keyed on normalized formula structure, reuse solver states across related queries, and batch independent queries for parallel solving. Without caching, analysis of a 10K-LoC target takes hours instead of minutes. |

**Why it can't be smaller:** Z3's API is powerful but low-level. The encoding layer is where semantic mismatches between the IR and SMT theories surface: Python integers are arbitrary-precision but SMT integers are mathematical integers (no overflow) while SMT bitvectors have fixed width. Each such mismatch requires explicit encoding decisions with soundness implications.

### 6. Incremental Dependency Tracker and Invalidation Engine — 20K LoC

| Component | Est. LoC | Justification |
|---|---|---|
| Fine-grained dependency graph | 7K | Tracks dependencies at the granularity of (function, predicate-set) pairs, not just function-to-function call edges. A change to function `f` that doesn't alter `f`'s inferred contract should NOT trigger re-analysis of `f`'s callers. Requires contract-diff computation and transitive invalidation with fixed-point. |
| Persistent storage layer | 4K | Serializes/deserializes: SSA IR, abstract states, inferred contracts, predicate sets, dependency edges. Must handle schema evolution (IR format changes across tool versions) and corruption recovery. Uses a content-addressed store to share sub-structures across versions. |
| VCS integration (git diff → changed function set) | 3K | Parses git diffs, maps textual changes to affected functions (requires maintaining a persistent source-location-to-function index), handles file renames, and deals with merge commits that touch the same function from multiple branches. |
| Incremental re-analysis scheduler | 4K | Given the invalidated set, schedules re-analysis respecting the dependency order (SCCs first, then topological order). Must handle timeouts per-function, partial results (if one function in an SCC times out, what contracts can we still assert?), and parallel analysis of independent functions. |
| CI integration harness | 2K | Produces machine-readable output (SARIF, JSON), integrates with GitHub Actions / GitLab CI, supports baseline comparison (new warnings only), and provides exit codes compatible with CI gate workflows. |

**Why it can't be smaller:** Incrementality is the feature that makes the system practical rather than academic. Without fine-grained invalidation, every commit triggers full re-analysis (~minutes to hours on real codebases). The dependency graph must be *semantic* (predicate-sensitive), not just *syntactic* (call-graph), which requires its own fixed-point computation. The persistent storage layer is unavoidable — you can't recompute SSA IR from scratch on every CI run and still meet latency targets.

### 7. Counterexample Visualization and Contract Output — 10K LoC

| Component | Est. LoC | Justification |
|---|---|---|
| Counterexample concretization + trace rendering | 4K | When verification fails, must produce a concrete execution trace (variable values at each step) that demonstrates the violation. Requires: SMT model extraction, IR-to-source mapping, trace simplification (remove irrelevant steps), and output formatting. |
| Contract serialization (multiple formats) | 3K | Outputs inferred contracts as: Python type stubs (`.pyi` with `Annotated[int, Gt(0)]`), TypeScript `.d.ts` declarations, standalone JSON, and inline source comments. Each format has its own serialization logic and must round-trip correctly. |
| Diagnostic rendering + IDE protocol | 3K | Produces diagnostics compatible with LSP (Language Server Protocol) for IDE integration, terminal-friendly colored output, and SARIF for CI. Must map IR-level locations back to source locations accurately, even through macro-like expansions (Python decorators, TS decorators). |

**Why it can't be smaller:** Usability determines adoption. A system that says "verification failed" without showing *why* is useless for debugging. A system that infers contracts but can't export them in a format other tools consume provides no value beyond the analysis itself.

### 8. Test Suite and Benchmark Infrastructure — 15K LoC

| Component | Est. LoC | Justification |
|---|---|---|
| Unit tests for each subsystem | 6K | Each abstract domain, each SSA lowering rule, each SMT encoding pattern needs unit tests. Abstract domain tests are particularly important: lattice laws (join is commutative, widen terminates) are easy to get wrong and hard to debug at the integration level. |
| Integration test suite | 4K | End-to-end tests on real-world code snippets: "given this Python function, the system should infer {x: int \| x >= 0} for the return type." Covers cross-subsystem interactions that unit tests miss. |
| Benchmark harness + regression tracking | 3K | Automated benchmarking on target codebases (e.g., subsets of popular Python/TS libraries). Tracks: analysis time, SMT query count, predicate count, false-positive rate, incremental speedup ratio. Stores historical results for regression detection. |
| Fuzzing infrastructure | 2K | Grammar-based fuzzing of the SSA compilers (generate random Python/TS programs, compile to SSA, verify SSA well-formedness invariants). Essential for finding crash bugs in the front-ends, which handle adversarial-complexity input grammars. |

**Why it can't be smaller:** A 150K-LoC system without thorough testing is unpublishable. Reviewers will (rightly) question whether the system actually works. The benchmark infrastructure is also essential for the paper's evaluation section — without automated benchmarking, you cannot credibly claim CI-scale performance.

---

## The Hardest Engineering Challenges

### 1. The Reduced Product Is Where Dreams Die

Combining abstract domains sounds clean in theory (it's a lattice product!). In practice, information flows between domains through *reduction operators* that must be hand-written for each pair of domains. With 5 domains, that's up to 10 pairwise reductions. Each reduction must be sound, precise enough to be useful, and efficient enough to not dominate analysis time. Getting one wrong produces either unsoundness (accepting buggy programs) or imprecision cascades (the system infers trivially weak contracts like `{x: int | true}`). This is the component most likely to consume months of debugging time.

### 2. CEGAR Termination and Predicate Explosion

The CEGAR loop is guaranteed to terminate only if the predicate space is finite and the abstraction refinement is monotone. In practice, predicate spaces for real programs are unbounded (every subexpression is a potential predicate), and each CEGAR iteration can produce multiple new predicates, causing the Liquid constraint system to grow exponentially. The engineering challenge is designing heuristics — predicate generalization, subsumption checking, relevance filtering — that keep the predicate set manageable without losing the predicates needed for verification. There is no known algorithm for this; it's engineering judgment encoded as code.

### 3. Faithfully Modeling Two Dynamic Languages

Python and TypeScript have specification documents that are (respectively) ~1,800 and ~800 pages. No SSA compiler can model all of this. The engineering challenge is choosing *which subset* to model, documenting the semantic gap, and ensuring the abstract interpreter is sound *with respect to the modeled subset*. Every shortcut (e.g., "we don't model metaclasses") must be tracked as a limitation that could produce false negatives. The choice of what to model is itself a research contribution — there's no existing reference for "the right subset of Python semantics for refinement type inference."

### 4. Incremental Invalidation Correctness

The incremental engine must guarantee: if function `f` is not re-analyzed, then `f`'s inferred contract is still valid given the current state of all functions `f` depends on. This is a non-trivial invariant when dependencies are predicate-sensitive. Example: function `g` changes its implementation but its inferred contract `{return: int | return > 0}` remains the same. Callers of `g` should NOT be re-analyzed. But determining that the contract is unchanged requires *running the analysis on `g` first*, creating a chicken-and-egg problem for scheduling. The solution involves speculative re-analysis with rollback, which adds significant complexity.

### 5. Z3 as a Collaborator, Not a Black Box

SMT solvers are powerful but temperamental. The same logical formula can take 10ms or 10 minutes depending on the encoding. The system must learn (through engineering experience, not ML) which encodings work for which query patterns: when to use arrays vs. uninterpreted functions for heap modeling, when to use bitvectors vs. integers for arithmetic, when to split a large query into independent subqueries. This encoding expertise cannot be abstracted away — it permeates the SMT interface layer and affects every component that generates verification conditions.

---

## Best-Paper Argument (Engineering Contribution Angle)

The strongest best-paper argument is:

**"This is the first system that demonstrates refinement type inference is practical for real dynamic-language codebases at CI scale, and the paper honestly shows what it took to get there."**

The formal methods community has published extensively on refinement types, abstract interpretation, and CEGAR — individually. No one has built a system that composes all three, targets languages as complex as Python and TypeScript, and achieves incremental analysis fast enough for CI. The engineering contribution is the *composition itself*: demonstrating that these techniques can coexist in a single system without the interfaces between them becoming the dominant source of imprecision or performance degradation.

The paper would be strongest at a venue like **PLDI**, **OOPSLA**, or **ICSE** (systems track), where reviewers value working artifacts. The 165K LoC artifact, if well-documented and reproducible, is itself a contribution — it provides a platform for future research on refinement types for dynamic languages, something that currently doesn't exist.

A secondary argument: the incremental analysis algorithm, validated at scale, contributes to the broader goal of making formal methods compatible with modern development workflows. Most formal verification tools require batch analysis. Demonstrating that predicate-sensitive incrementality works in practice is a result that generalizes beyond this specific system.

---

## Fatal Flaws

### 1. Scope Risk Is Extreme
165K LoC is a multi-year, multi-person effort. If this is a single-PhD project, the likely outcome is a 40K-LoC system that handles a subset of one language, with the paper describing the architecture for the full system but evaluating only the partial implementation. Reviewers will note the gap between ambition and artifact.

**Mitigation:** Ruthlessly prioritize. Ship Python-only first (drop TypeScript to "future work"). Implement only intervals + nullity domains (drop octagons, polyhedra, string, and array-shape domains). This gets you to ~60K LoC, which is achievable and still publishable.

### 2. False Positive Rate May Be Unacceptable
Refinement type inference on dynamic languages will produce false positives (spurious warnings) because the abstract model necessarily over-approximates runtime semantics. If the false-positive rate exceeds ~10-15%, practitioners won't use the tool, and reviewers will question the practical contribution. There is no way to know the false-positive rate until the system is built and evaluated on real code.

**Mitigation:** Design the system to report confidence levels, not just pass/fail. Allow users to suppress warnings with explicit annotations. Measure precision on curated benchmarks before claiming CI-readiness.

### 3. Evaluation Requires Real-World Codebases That Cooperate
The system must be evaluated on non-trivial Python/TypeScript projects. But real-world code uses features the system doesn't model (eval, dynamic imports, C extensions, WASM interop). If the system can only analyze toy programs or heavily-curated subsets of real projects, the evaluation is weak.

**Mitigation:** Choose evaluation targets carefully — scientific Python (NumPy-using code), well-typed TypeScript libraries. Measure and report the "analyzable fraction" of each target honestly.

### 4. Incrementality Benefits Are Hard to Demonstrate
Claiming "incremental analysis is fast" requires showing that (a) full analysis is slow enough that incrementality matters, and (b) typical commits invalidate a small fraction of the dependency graph. If full analysis takes 30 seconds on the evaluation targets, incrementality is a non-contribution. If typical commits invalidate 80% of the dependency graph, incrementality provides negligible speedup.

**Mitigation:** Evaluate on codebases large enough (50K+ LoC targets) that full analysis takes minutes. Mine real commit histories to show the distribution of invalidation set sizes.

### 5. The "Two Languages" Claim May Backfire
Supporting both Python and TypeScript doubles the front-end work but doesn't double the scientific contribution. Reviewers may ask: "Why not go deeper on one language instead of wider on two?" The answer ("because it validates IR generality") is legitimate but may not satisfy reviewers who wanted deeper semantic modeling of one language.

**Mitigation:** Frame the two-language support as an engineering contribution (the common IR is reusable for future languages) and evaluate both languages, but make the primary scientific contributions (CEGAR for dynamic languages, predicate-sensitive incrementality) language-agnostic.
