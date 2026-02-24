# Technical Difficulty: Final Subsystem Breakdown

## Scope and Framing

**Paper scope:** Python-only refinement type inference with predicate-sensitive incremental analysis.
**Full artifact scope:** Python + TypeScript support (TypeScript is a stretch goal, not a paper claim).
**Central contribution:** The *composition* of known techniques (SSA construction, abstract domains, SMT encoding, CEGAR, incremental analysis) into a system that infers value-level refinement predicates for unannotated dynamic-language code. No individual subsystem is novel in isolation — SSA construction exists (Pytype), abstract domains exist (Astrée/Mopsa), SMT encoding is ubiquitous, CEGAR is textbook. What does not exist is a system that composes all four and adds **predicate-sensitive incremental invalidation** to make it CI-practical on real Python codebases.

Every subsystem description below states: (a) what it does, (b) why existing implementations cannot be reused, (c) what makes *this* implementation's engineering different from prior work, and (d) why the LoC estimate is irreducible.

---

## Full Subsystem Enumeration

### Total: ~155K LoC

| # | Subsystem | LoC | Paper-critical? |
|---|-----------|-----|-----------------|
| 1 | Python SSA Compiler | 22K | Yes |
| 2 | TypeScript SSA Compiler | 22K | No (stretch) |
| 3 | Common Refinement IR | 8K | Yes |
| 4 | Abstract Domain Library | 26K | Yes |
| 5 | Predicate Inference & CEGAR Engine | 22K | Yes |
| 6 | SMT Encoding & Z3 Interface | 14K | Yes |
| 7 | Incremental Dependency Tracker | 18K | Yes |
| 8 | Standard Library Models | 12K | Yes |
| 9 | Contract Output & Diagnostics | 6K | Yes |
| 10 | Test & Benchmark Infrastructure | 5K | Yes |
| | **Total** | **155K** | |
| | **Paper-critical subset (Python-only)** | **~133K** | |
| | **Core system without TS front-end** | **~133K** | |

---

## Subsystem Details

### 1. Python SSA Compiler — 22K LoC

**What it does.** Lowers Python 3.10–3.12 source into a typed SSA intermediate representation with explicit predicate guards at control-flow merge points.

**Why Pytype's SSA isn't reusable.** Pytype operates on *bytecode*, not source AST, and its SSA representation discards source-level predicate structure — `isinstance` checks become opaque boolean phi-nodes. Our system needs SSA where every branch condition is a *named predicate* in the refinement language (e.g., `isinstance(x, int)` becomes a guard `τ(x) = int` that the CEGAR loop can reference). This predicate-preserving lowering is the front-end's distinguishing feature and does not exist in any current Python compiler.

**What's hard.**
- *Exception control flow:* Python's try/except/finally creates implicit edges that break SSA dominance. Each exception handler is a merge point requiring phi-nodes for every live variable, with predicate guards encoding "exception of type T was raised." This interacts badly with `with`-statement desugaring (2 implicit exception paths per `with` block).
- *Comprehension scoping:* List/dict/set comprehensions and generator expressions have their own scope in Python 3, but capture enclosing variables by reference. SSA must model the capture precisely or aliasing analysis downstream is unsound.
- *`*args`/`**kwargs` packing:* The SSA must represent argument packing as explicit operations that the abstract domain can track — a call `f(1, 2, key=3)` must lower to a form where the abstract interpreter knows `args[0] = 1`, `args[1] = 2`, `kwargs["key"] = 3`.
- *Match statements (PEP 634):* Structural pattern matching introduces complex destructuring with guard clauses. Each arm is a predicate-guarded branch; the guards involve type tests, value comparisons, and structural shape checks simultaneously.

**Why it can't be smaller.** Python 3.10–3.12 has 93 grammar productions that can appear in a function body. Each needs an SSA lowering rule. The predicate-preserving aspect adds ~40% overhead vs. a standard SSA construction because every branch condition must be symbolically represented, not just evaluated. Type stub resolution (typeshed + third-party `.pyi` files) requires a mini type-checker for generic substitution and protocol matching (~4K LoC alone).

---

### 2. TypeScript SSA Compiler — 22K LoC (Stretch Goal)

**What it does.** Same target IR as the Python front-end, but from TypeScript 5.x source via the `ts.createProgram` API.

**Why tsc's internal CFG isn't reusable.** TypeScript's compiler performs control-flow narrowing internally but does not expose a predicate-labeled CFG. Its narrowing results are consumed and discarded during type-checking; there is no API to extract "at this program point, the type of `x` has been narrowed to `string` because of the `typeof x === 'string'` guard on line 42." We need exactly this information in a form the CEGAR loop can manipulate.

**What's hard.**
- *TypeScript's type system is Turing-complete:* Conditional types, mapped types, and template literal types require a bounded evaluator to resolve call targets. Over-approximating "any" loses all refinement precision; precise evaluation risks non-termination.
- *Structural typing with excess property checks:* TypeScript's subtyping has contextual rules (excess property checks apply at assignment but not at type assertion) that affect which predicates are valid at a given program point.
- *`this` typing and method extraction:* Methods can be extracted from objects and lose their `this` binding. The SSA must track `this`-binding status as a predicate.

**Why it can't share code with the Python front-end.** The only shared component is the target IR format and source-map threading (~3K LoC, accounted for in subsystem 3). The languages differ in scoping (LEGB vs. lexical with hoisting), object models (attribute dict + MRO vs. prototype chain), module systems (`import` semantics are different), and type system expressiveness. A unified parser would be a leaky abstraction.

---

### 3. Common Refinement IR — 8K LoC

**What it does.** Defines the intermediate representation that both front-ends target: SSA-form instructions, predicate-labeled control flow, a type lattice for dynamic type tags, and source-location provenance for counterexample rendering.

**What makes this different from LLVM IR or similar.** The IR must preserve *refinement-relevant* source semantics that compilation IRs discard:
- **Predicate guards as first-class IR constructs.** Each conditional branch carries an explicit predicate term (not just a boolean variable) that the abstract interpreter and CEGAR loop consume directly.
- **Type-tag annotations.** Every SSA variable carries a type-tag lattice element (the dynamic type at that program point), which is distinct from the refinement predicate. The abstract interpreter needs both.
- **Source provenance.** Every IR node maps back to source locations, original variable names, and annotation context. This is essential for counterexample rendering but also for predicate mining (the CEGAR loop generates predicates from source-level expressions).

**Why it can't be smaller.** The IR definition itself is compact (~2K LoC), but well-formedness checking (~2K), serialization/deserialization for incremental caching (~2K), and the provenance threading (~2K) are all necessary for downstream subsystems to function correctly. Skipping serialization means no incremental analysis; skipping provenance means no usable error messages.

---

### 4. Abstract Domain Library — 26K LoC

**What it does.** Implements the abstract domains used by the fixed-point analyzer to compute over-approximate invariants at each program point. These invariants seed the CEGAR loop's predicate search.

**Why Mopsa's/Astrée's domains aren't reusable.** Mopsa's Python analysis uses interval and pointer domains but does not produce predicates in a form compatible with Liquid-style Horn clause solving. Astrée's domains target C semantics (fixed-width integers, explicit pointers). Our domains must operate over Python's semantics (arbitrary-precision integers, duck-typed attribute access, truthiness coercion) and emit their results as *refinement predicates* that feed into the CEGAR loop. This predicate-emission interface is what distinguishes every domain implementation from its textbook counterpart.

| Domain | LoC | What makes it different here |
|--------|-----|------------------------------|
| Numeric (intervals + octagons) | 8K | Must handle Python's arbitrary-precision integers (no overflow) and truthiness (`0` is falsy, `""` is falsy). Octagon constraints must be expressible as refinement predicates for the CEGAR loop. Standard octagon implementations (APRON) target C/Java fixed-width semantics. |
| Type-tag (dynamic type lattice) | 4K | Lattice over Python's runtime type hierarchy including `type()`, `isinstance`, and structural protocol checks. No analog in C/Java analyzers. Interacts with every other domain via reduction (knowing `type(x) = int` constrains the numeric domain; knowing `type(x) = NoneType` constrains the nullity domain). |
| Nullity / optionality | 3K | Tracks `None`/`not-None` with path-sensitive precision. Must handle Python's `Optional` semantics, `None`-coalescing patterns (`x or default`), and truthiness-based null checks (`if x:`). |
| Dictionary shape | 5K | Tracks known keys and their associated type/value abstractions for dict-as-struct patterns. Python dicts used as records (`config["timeout"]`) require per-key tracking. No existing domain handles this for Python with refinement-predicate output. |
| Reduced product | 6K | Pairwise reduction operators between all domain pairs. With 4 domains, there are up to 6 non-trivial reductions. Each must be hand-written, sound, and efficient. **This is the hardest component in the domain library** — getting one reduction wrong produces either unsoundness or precision collapse, and bugs manifest as incorrect predicates fed to the CEGAR loop, which are extremely difficult to diagnose. |

**Why it can't be smaller.** Each domain is mathematically distinct (different lattice structures, different transfer functions, different widening operators). The reduced product is additional code on top of individual domains, not a replacement. Dropping any domain eliminates a class of inferrable predicates: drop octagons and you lose relational properties like `i < len(arr)`; drop the dictionary domain and you lose shape tracking for dict-as-struct patterns, which are pervasive in Python.

---

### 5. Predicate Inference & CEGAR Engine — 22K LoC

**What it does.** The core algorithmic loop: given abstract states from subsystem 4, infers refinement predicates via Liquid-style Horn clause solving, checks them via SMT, and refines via CEGAR when counterexamples are found.

**What makes this different from Liquid Haskell's CEGAR.** Liquid Haskell operates on a pure, statically-typed language where call targets are known, mutation doesn't exist, and predicate templates are programmer-supplied. Our CEGAR loop must:
1. **Mine predicates automatically** from source-level expressions (comparisons, `isinstance` checks, attribute access guards) — the "guard harvesting" insight. In Liquid Haskell, the programmer provides qualifiers; here, the system discovers them.
2. **Handle dynamic dispatch** — when the CEGAR loop encounters a call through duck typing or `__getattr__`, it cannot enumerate call targets statically. It must over-approximate the callee's behavior using the abstract state, then refine if the approximation is too coarse.
3. **Interact with multiple abstract domains** — counterexample traces pass through the reduced product, and interpolant extraction must produce predicates that belong to the appropriate domain. This cross-domain interpolation has no analog in Liquid Haskell.

| Component | LoC | What's hard |
|-----------|-----|-------------|
| Predicate template mining | 4K | Must normalize source expressions into predicate templates, deduplicate, and rank by likely usefulness. The ranking heuristic (which predicates to try first) has enormous impact on convergence speed — there is no known algorithm; it is engineering judgment encoded as code. |
| Liquid constraint generation | 6K | Generates Horn clauses with κ-variables (predicate unknowns) from the IR + abstract states. Must handle: recursive functions (requires well-founded termination arguments), mutual recursion (SCC-based solving), and dynamic dispatch (over-approximate κ for unknown callees). |
| CEGAR loop controller | 6K | Counterexample extraction → feasibility check → interpolant computation → predicate generalization → re-abstraction. Each step is a distinct algorithm. The loop must enforce time budgets per function and degrade gracefully (report "unknown" rather than diverge). |
| Fixed-point solver | 6K | Worklist-based iterative solver for the Horn clause system. Dependency-driven: when a κ-variable's solution changes, only clauses mentioning that variable are re-evaluated. This is NOT delegable to Z3 — the constraints are over predicate *variables*, not ground formulas. |

**Why it can't be smaller.** The CEGAR loop is inherently a multi-component feedback system. Each component has distinct data structures and algorithms. The predicate mining component alone requires AST traversal, normalization, deduplication, and ranking — these are not reducible to a library call.

---

### 6. SMT Encoding & Z3 Interface — 14K LoC

**What it does.** Translates verification conditions from the IR and predicate inference engine into SMT-LIB formulas, manages Z3 solver sessions, and extracts interpolants/models.

**What makes this different from standard Z3 usage.** The encoding must faithfully model Python's value semantics, which diverge from standard SMT theories in specific ways:
- Python integers are arbitrary-precision → must use SMT integer sort, not bitvectors, but then bitwise operations require explicit axioms.
- Python's `//` operator is floor division (rounds toward negative infinity), not truncation division (C semantics) → requires a non-standard encoding.
- Python's truthiness coercion (`bool(0) = False`, `bool("") = False`, `bool([]) = False`) must be encoded as theory axioms.
- String operations (`str.startswith`, `in` operator on strings) require the SMT string theory (CVC5's SLIA), but combining string theory with integer arithmetic and uninterpreted functions can cause solver divergence.

| Component | LoC | Why it's irreducible |
|-----------|-----|---------------------|
| IR → SMT encoding | 5K | One encoding rule per IR instruction type × relevant SMT theory. Python's operator overloading means the same IR instruction (`add`) maps to different theories depending on the type-tag domain's result. |
| Z3 session management | 3K | Incremental push/pop stacks for CEGAR iterations, timeout enforcement, crash recovery (Z3 does segfault on certain inputs), memory monitoring. |
| Interpolant extraction | 3K | Craig interpolation for predicate discovery. Z3's interpolation API is experimental and sometimes produces interpolants outside the desired theory fragment → requires post-processing and fallback strategies. |
| Query caching & batching | 3K | Normalized formula hashing, result caching across CEGAR iterations, parallel query dispatch for independent VCs. Without caching, analysis of a 10K-LoC target takes hours. |

**Why it can't be smaller.** Every encoding decision has soundness implications. Getting Python's `//` wrong (truncation vs. floor) produces unsound results on negative operands. Getting truthiness wrong produces false positives on idiomatic `if x:` checks. Each such decision is a code path with tests.

---

### 7. Incremental Dependency Tracker — 18K LoC

**What it does.** Tracks inter-function dependencies at the granularity of *(function, inferred-contract)* pairs. When a function's implementation changes, determines the minimal set of functions requiring re-analysis by computing whether the changed function's *contract* (not just its code) has changed.

**This is the paper's genuinely novel subsystem.** Existing incremental analyzers (Infer, mypy, Pyright) invalidate at function-level or file-level granularity: if `f` changes, all callers of `f` are re-analyzed. Our system is **predicate-sensitive**: if `f`'s implementation changes but its inferred contract `{return: int | return > 0}` remains the same, callers of `f` are NOT re-analyzed. This requires:

1. **Dependency edges labeled with predicate sets.** The dependency graph records not just "g depends on f" but "g depends on f's contract containing the predicate `return > 0`." If `f`'s contract changes to `{return: int | return ≥ 0}` (weakened), `g` is invalidated. If it changes to `{return: int | return > 0 ∧ return < 100}` (strengthened), `g` is NOT invalidated (strengthening a callee's postcondition cannot break a caller).
2. **Speculative re-analysis with rollback.** To determine if `f`'s contract changed, we must re-analyze `f` first — but `f`'s analysis depends on its callees' contracts. This creates a chicken-and-egg problem requiring SCC-aware scheduling with speculative execution and rollback on contract change.
3. **Stratified invalidation for negated dependencies.** Some dependencies arise from negation: "g's type-check passes because `f` does NOT raise `TypeError` on inputs of type `int`." If `f` now raises `TypeError` on some `int` inputs, `g` must be re-analyzed. Tracking negated dependencies requires stratification-aware propagation (the connection to Datalog with stratified negation identified in the math framing).

| Component | LoC | What's hard |
|-----------|-----|-------------|
| Predicate-labeled dependency graph | 6K | Data structure for edges labeled with predicate sets + monotonicity metadata (strengthening vs. weakening). Must support efficient transitive closure queries for invalidation. |
| Contract-diff computation | 3K | Given old and new contracts for a function, determines whether the change is a strengthening (no invalidation needed), weakening (invalidation needed), or incomparable (invalidation needed). Requires SMT entailment checks. |
| Persistent store | 4K | Serialization of dependency graph, contracts, and SSA IR to disk. Content-addressed for deduplication across commits. Schema evolution for tool version upgrades. |
| VCS integration | 2K | Maps git diffs to affected function sets. Handles file renames, moved functions, and merge commits. |
| Re-analysis scheduler | 3K | Topological scheduling over SCCs with speculative execution, timeout handling, and partial-result propagation. |

**Why it can't be smaller.** The predicate-sensitivity is the novel contribution, and it permeates every component. A function-level dependency tracker would be ~5K LoC; the additional 13K is entirely due to predicate labeling, contract-diff computation, and the speculative scheduling required to support it.

---

### 8. Standard Library Models — 12K LoC

**What it does.** Provides refinement-annotated models for Python's standard library functions that the analyzer encounters in real code. Without these, every call to `len()`, `dict.get()`, `range()`, or `list.append()` is opaque, and the inferred refinements are trivially weak.

**Why typeshed stubs aren't sufficient.** Typeshed provides *type* signatures (e.g., `len(s: Sized) -> int`). We need *refinement* signatures (e.g., `len(s: Sized) -> {r: int | r >= 0}`). These do not exist anywhere and must be written from scratch.

**What's hard.**
- *Coverage vs. precision tradeoff:* Python's standard library has ~300 commonly-used functions. Each needs a hand-written refinement model. Under-modeling produces "unknown" results; over-modeling risks unsoundness if the model doesn't match CPython's actual behavior.
- *Higher-order functions:* `map`, `filter`, `sorted` with key functions require models that are parametric in the callback's refinement contract. This is Liquid-style abstract refinement — the model must express "if the callback ensures `return > 0`, then `map(callback, xs)` produces a list where all elements are `> 0`."
- *Container semantics:* `list.append` must update the abstract state to include the new element's refinement. `dict.__getitem__` must produce a result refined by the key's value. `set.add` must ensure the membership predicate holds. Each container operation is a custom transfer function.

**Why it can't be smaller.** Empirical measurement: the top 50 Python packages by download count use ~150 distinct standard library functions in their hot paths. Each function needs a refinement model averaging ~80 LoC (signature + preconditions + postconditions + edge cases). 150 × 80 = 12K. Modeling fewer functions means the analyzer goes opaque on common code and produces trivial results.

---

### 9. Contract Output & Diagnostics — 6K LoC

**What it does.** Renders inferred contracts and verification failures in formats consumable by developers and CI systems.

| Component | LoC | Purpose |
|-----------|-----|---------|
| Counterexample trace renderer | 2K | Concretizes SMT models into step-by-step execution traces mapped to source locations. Simplifies traces by removing irrelevant steps. |
| Contract serialization | 2K | Emits inferred contracts as `.pyi` stubs with `Annotated[int, Gt(0)]` syntax, standalone JSON, and inline comments. |
| SARIF / CI output | 2K | Machine-readable output for GitHub Actions / GitLab CI. Baseline comparison (new warnings only). LSP-compatible diagnostic objects. |

**Why it can't be smaller.** A system that says "error at line 42" without explaining *why* or *what the inferred contract is* provides no value beyond a linter. The counterexample renderer is essential for debugging false positives. The contract serializer is essential for the system's output to be consumed by other tools (including the system's own incremental engine on the next run).

---

### 10. Test & Benchmark Infrastructure — 5K LoC

**What it does.** Unit tests for domain lattice laws, integration tests on curated Python snippets, regression benchmarks on real packages.

**Why only 5K.** Following the critique synthesis: tests are engineering necessity, not research contribution. We include only what's needed for correctness and evaluation. Fuzzing infrastructure is dropped; lattice-law property tests (~1K) replace it.

---

## The Three Hardest Engineering Challenges

### Challenge 1: The Reduced Product (Domain Library, §4)

**Why it's the hardest.** Combining N abstract domains requires up to N(N−1)/2 pairwise reduction operators, each hand-written, each with soundness obligations. With 4 domains (numeric, type-tag, nullity, dictionary shape), there are 6 reductions. Each reduction must propagate information bidirectionally without introducing cycles in the reduction cascade. Example: the type-tag domain learns `type(x) = int`, which tells the numeric domain that `x` is in ℤ, which tells the nullity domain that `x ≠ None`, which tells the dictionary domain that `x` is not a dict. If any step is imprecise, downstream CEGAR iterations produce spurious counterexamples that waste the time budget. If any step is unsound, the system certifies buggy code as safe.

**Why no existing solution helps.** APRON (the standard abstract domain library) provides reductions for numeric domains only and assumes C semantics. Our domains include Python-specific type-tag and dictionary-shape domains that have no APRON analog. The reductions involving these domains must be designed and implemented from scratch.

**Estimated debugging time:** 30–40% of total development time will be spent diagnosing precision failures in the reduced product.

### Challenge 2: CEGAR Convergence on Real Python Code (CEGAR Engine, §5)

**Why it's hard.** The CEGAR loop's predicate space for real Python functions is effectively unbounded (every subexpression is a candidate predicate). The loop is guaranteed to terminate only if the predicate lattice has finite height, but the lattice height is bounded by |program variables| × |atomic predicates| — which can be >10⁶ for a 100-line function. In practice, the loop must converge in 3–5 iterations or it's useless.

**What makes this different from CEGAR in model checking.** In software model checking (SLAM, BLAST), CEGAR targets single safety properties and the predicate space is constrained by the property. Here, we are inferring *all* refinement predicates for a function simultaneously, and the "property" is the entire contract — a conjunction of potentially dozens of predicates. The predicate discovery must be guided by *relevance filtering* (which predicates from a counterexample are actually needed?) and *generalization* (can we replace `x > 5` with `x > 0` without losing the proof?). There is no known algorithm for optimal relevance filtering; it is an engineering heuristic that determines whether the system works on real code or only on benchmarks.

**The existential risk.** If CEGAR resolves <70% of functions in the top-100 Python repos within a 10-second budget, the paper cannot credibly claim "refinement type inference for Python." It becomes "refinement type inference for a subset of Python idioms" — still publishable, but with a weaker narrative.

### Challenge 3: Predicate-Sensitive Incremental Invalidation (Dependency Tracker, §7)

**Why it's hard.** The dependency graph must answer queries of the form: "given that function `f`'s contract changed from C₁ to C₂, which other functions' contracts might change?" This requires:
1. Determining whether C₁ → C₂ is a strengthening (check: does C₂ ⊨ C₁? — an SMT query).
2. If it's a weakening or incomparable change, traversing the dependency graph to find all transitively affected functions.
3. Scheduling re-analysis in topological order over SCCs, with speculative execution: re-analyze `g` assuming `f`'s old contract, check if `g`'s new contract differs, and if so, propagate.

The chicken-and-egg problem: to know if re-analysis is needed, you must partially re-analyze. The speculative approach can cascade: if `g`'s contract changes, `g`'s callers must be speculatively re-analyzed too. In the worst case (global contract change), this degrades to full re-analysis. The engineering challenge is ensuring that the *common case* (local code change, contract unchanged) is fast (O(1) dependency lookups, O(0) re-analysis of callers).

**Why this is genuinely novel.** Infer's incrementality is procedure-level: any change to `f` re-analyzes all callers. Mypy/Pyright's incrementality is file-level. No existing system tracks dependencies at the predicate level. The data structures (predicate-labeled dependency edges with monotonicity annotations), the scheduling algorithm (SCC-aware speculative execution with rollback), and the correctness argument (stratification-preserving propagation) are all new.

---

## Why This Is Genuinely a 155K LoC Problem

**The "can't you just" test:** For each subsystem, we ask "can't you just reuse X?" and explain why not.

| "Can't you just..." | Why not |
|---------------------|---------|
| ...use Pytype's SSA? | Pytype operates on bytecode, discards predicate structure. We need source-level predicate guards in SSA. |
| ...use APRON for domains? | APRON assumes C semantics (fixed-width integers, explicit pointers). Python has arbitrary-precision integers, duck typing, truthiness coercion. |
| ...use Liquid Haskell's solver? | Liquid Haskell's solver assumes programmer-supplied qualifiers and pure, statically-typed code. We must mine predicates automatically from dynamically-typed code with mutation. |
| ...use Z3 directly? | We do use Z3, but the encoding layer (~5K LoC) handles the semantic gap between Python values and SMT theories. Python `//` ≠ SMT `div`. Python truthiness ≠ SMT boolean. Each mismatch is a code path. |
| ...use Infer's incrementality? | Infer invalidates at procedure level. Our predicate-sensitive invalidation is strictly finer-grained and requires fundamentally different data structures. |
| ...use typeshed for stdlib models? | Typeshed provides type signatures, not refinement signatures. `len() -> int` vs. `len() -> {r: int \| r ≥ 0}`. The refinement annotations must be written from scratch. |
| ...do Python only and save 22K? | The paper IS Python-only. The 22K TypeScript front-end is the stretch goal that makes this a full 155K system. Without it, the paper-critical artifact is ~133K. |

**The composition tax.** Beyond the individual subsystems, there is a *composition tax*: the interfaces between subsystems impose constraints that increase each subsystem's complexity. The SSA compiler must emit IR that the abstract domain library can consume, that the CEGAR loop can query, that the SMT encoder can translate, and that the incremental tracker can serialize. Each interface is a contract with invariants that must be maintained as any subsystem evolves. This cross-cutting complexity is not localized in any one subsystem — it is distributed across all of them and accounts for an estimated 15–20% of total LoC (implicit in the estimates above, not double-counted).

**Why the number isn't padded.** The prior art audit identified every subsystem's textbook analog. In each case, the textbook version cannot be reused because it targets a different source language semantics, produces output in a different format, or lacks the predicate-emission interface that the CEGAR composition requires. The delta between "textbook implementation" and "implementation that composes into this system" is typically 40–60% of the subsystem's LoC. That delta is the engineering contribution — not any one subsystem in isolation, but the fact that all of them work together to produce a result that no existing tool produces: automatically inferred, SMT-checked, value-level refinement contracts for unannotated Python programs, maintained incrementally across commits.
