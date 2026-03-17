# Implementation Scoping: Mutation-Driven Contract Synthesis Engine

## Lead: Implementation Scope Lead
## Phase: Crystallization
## Date: 2026-03-08

---

# 0. EXECUTIVE SUMMARY

The core intellectual insight — "mutation survival boundaries encode implicit contracts" —
is a ~25K LoC idea. Building a production-grade, multi-language, formally verified
pipeline that is genuinely novel at PLDI/POPL caliber pushes to ~95–110K LoC of real
complexity. Reaching 150K LoC *without padding* requires the research-platform layer
(benchmarking, visualization, comparison infrastructure). I recommend **Approach B+C
hybrid** targeting ~155K LoC, implemented in **Rust**, and I'll be brutally honest about
where each line comes from.

---

# 1. APPROACH A — Minimal Viable (Core Insight Only)

## What's the smallest system that demonstrates mutation→contract?

Single language (Java), single mutation framework (Major/PIT-style), single solver (Z3),
no build-system integration, no CI, no multi-language support. Hardcoded to work on
method-level Java functions with JUnit tests.

## Subsystem Breakdown

| # | Subsystem | LoC | Rationale |
|---|-----------|-----|-----------|
| A1 | Java source parser (tree-sitter or Eclipse JDT binding) | 2,500 | Parse Java into typed AST, extract method signatures, types |
| A2 | Mutation engine (operator library + AST rewriting) | 4,000 | ~15 mutation operators (AOR, ROR, LCR, UOI, SDL, etc.), first-order only |
| A3 | Test harness & mutation execution | 3,000 | Compile mutants, run JUnit, capture kill/survive verdicts |
| A4 | Mutation–survival boundary analysis | 3,500 | Cluster mutations by location, compute survival frontiers, extract "what the tests actually check" |
| A5 | Contract template library & SyGuS encoding | 4,500 | Pre/postcondition grammar, encode boundary data as SyGuS constraints |
| A6 | SyGuS solver integration (CVC5 binding) | 2,000 | Serialize SyGuS-IF problems, parse synthesis results |
| A7 | SMT verification of synthesized contracts | 2,500 | Encode source semantics + contract, bounded model check via Z3 |
| A8 | Contract output formatter (JML annotation) | 1,000 | Emit contracts as JML annotations |
| A9 | CLI driver & orchestration | 1,500 | Pipeline coordination, caching, error handling |
| A10 | Core data structures & utilities | 1,500 | IR types, graph structures, logging |
| | **TOTAL** | **~26,000** | |

## Verdict: Does this reach 150K?

**No. Not remotely.** This is a ~26K LoC system. It demonstrates the insight but is:
- Single-language, single-solver, no real engineering
- No soundness guarantees on the contract inference
- No handling of heap, aliasing, concurrency, exceptions
- No evaluation infrastructure
- Essentially a prototype / proof-of-concept paper artifact

---

# 2. APPROACH B — Full Pipeline (Production-Grade Tool)

## Complete system: source → verified contracts, multi-language, real build integration

## Subsystem Breakdown

| # | Subsystem | LoC | Description |
|---|-----------|-----|-------------|
| **LAYER 1: Language Front-Ends** | | | |
| B1 | Language-agnostic IR (MuIR) | 6,000 | Typed intermediate representation capturing control flow, data flow, heap model, type constraints. Must unify Java references, C pointers, Python dynamic types. This is the hardest design problem in the system. |
| B2 | Java front-end (Eclipse JDT → MuIR) | 5,500 | Full Java 17 support: generics, lambdas, method references, exceptions, try-with-resources. Not just "parse" — must lower to MuIR faithfully. |
| B3 | C front-end (Clang libTooling → MuIR) | 6,500 | C11 support: pointers, structs, unions, function pointers, preprocessor handling. Pointer analysis for heap model. Harder than Java due to undefined behavior. |
| B4 | Python front-end (AST module → MuIR) | 5,000 | Python 3.10+: dynamic typing requires type inference layer (gradual typing), decorators, generators, comprehensions. Must handle duck typing for contracts. |
| B5 | Type reconstruction & inference | 4,000 | Unification-based type inference for Python, generic instantiation for Java, typedef resolution for C. Feeds MuIR type annotations. |
| | | | |
| **LAYER 2: Mutation Infrastructure** | | | |
| B6 | Mutation operator library | 5,500 | 25+ operators across all three languages, operating on MuIR. First-order and higher-order (2nd order) mutations. Language-specific operators (e.g., null-injection for Java, use-after-free patterns for C, type-coercion for Python). |
| B7 | Mutation scheduling & reduction | 4,000 | Equivalent mutant detection (TCE), dominator-based reduction, mutant sampling strategies. Critical for performance — naive approach generates 100K+ mutants for real projects. |
| B8 | Mutant compilation & sandboxed execution | 7,000 | Per-language compilation (javac, clang, Python bytecode), sandboxed execution with timeout/memory limits, crash detection for C, exception capture for Java/Python. Process isolation. |
| B9 | Test framework adapters | 4,500 | JUnit 4/5, pytest, Google Test, CTest. Must parse test results, correlate with mutants, handle flaky tests, support test prioritization. |
| B10 | Build system integration | 5,000 | Maven/Gradle (Java), CMake/Make (C), setuptools/poetry (Python). Must inject mutants without breaking builds. Incremental compilation support. |
| | | | |
| **LAYER 3: Boundary Analysis (THE CORE NOVELTY)** | | | |
| B11 | Kill/survive matrix computation | 3,500 | Sparse matrix of (mutant × test) outcomes. Incremental updates. Compression for large projects. |
| B12 | Mutation boundary extraction | 6,000 | **The key algorithm.** For each program point, compute the "boundary" — the set of mutations that transition from killed→survived. This boundary encodes what the tests *implicitly* specify. Requires: (a) spatial clustering by program location, (b) semantic clustering by mutation effect, (c) boundary surface computation in the mutation-effect space. |
| B13 | Boundary → contract constraint translation | 5,500 | Translate "these mutations survive, those don't" into logical constraints on pre/postconditions. E.g., "ROR(>=, >) survives on `x >= 0`" → the test suite doesn't distinguish `x >= 0` from `x > 0`, so the precondition boundary is at `x == 0`. This is the core theoretical contribution. |
| B14 | Weakest-precondition / strongest-postcondition engine | 5,000 | WP/SP transformer operating on MuIR. Handles loops (with bounded unrolling), heap (separation logic fragments), exceptions. Needed to relate mutation effects to contract obligations. |
| | | | |
| **LAYER 4: Contract Synthesis** | | | |
| B15 | SyGuS grammar construction | 4,500 | Build synthesis grammars from: (a) program types, (b) boundary constraints, (c) common contract patterns (non-null, bounds, sortedness, etc.). Language-aware: Java uses `@NonNull`, C uses pointer validity, Python uses type guards. |
| B16 | SyGuS solver integration (CVC5 + custom CEGIS) | 5,000 | CVC5 for standard SyGuS. Custom CEGIS loop for mutation-specific counterexample generation: use surviving mutants as counterexamples. This is novel — standard SyGuS doesn't have a mutation oracle. |
| B17 | Contract minimization & simplification | 3,500 | Synthesized contracts are often redundant. Minimize via: (a) implication checking, (b) subsumption, (c) human-readability heuristics. E.g., `x >= 0 && x >= -1` → `x >= 0`. |
| B18 | Inductive invariant inference for loops | 4,000 | Mutation boundaries at loop heads encode loop invariants. Use Houdini-style iterative weakening seeded by boundary data. Novel: mutations guide the invariant candidate space. |
| | | | |
| **LAYER 5: Verification** | | | |
| B19 | Bounded SMT verification engine | 5,500 | Verify synthesized contracts against source. Encode MuIR + contract into SMT-LIB. Bounded loop unrolling (k=5,10,20). Theory support: integers, bitvectors, arrays, algebraic datatypes. |
| B20 | Counterexample-guided refinement | 3,500 | When verification fails: extract counterexample, feed back into synthesis as new constraint. CEGAR loop between B16 and B19. |
| B21 | Heap reasoning (separation logic fragment) | 5,000 | Java/C heap contracts require separation logic. Implement symbolic heap graphs with frame rule. This is genuinely hard — full separation logic is undecidable, must choose decidable fragment. |
| B22 | Soundness certification | 3,000 | For each emitted contract, produce a machine-checkable certificate (proof witness). Enables independent verification without re-running the engine. |
| | | | |
| **LAYER 6: Integration & Output** | | | |
| B23 | Contract annotation emitters | 3,000 | JML (Java), ACSL (C), Python type stubs + docstring contracts. Must integrate with existing annotation ecosystems. |
| B24 | CI/CD integration layer | 3,500 | GitHub Actions / Jenkins / GitLab CI plugins. Incremental mode: only re-synthesize for changed functions. Diff-aware mutation. |
| B25 | Latent bug reporter | 3,000 | When a synthesized contract reveals a surviving mutant that *should* be killed (contract says behavior is wrong, but no test catches it): report as latent bug with witness input. |
| B26 | Incremental analysis engine | 4,000 | Cache mutation results, invalidate on code changes, re-synthesize only affected contracts. Essential for usability on real projects. |
| | | | |
| **LAYER 7: Infrastructure** | | | |
| B27 | Parallel execution engine | 4,500 | Mutation testing is embarrassingly parallel but needs: work-stealing scheduler, memory-bounded mutant queue, result aggregation. Must run efficiently on 4-16 core laptops. |
| B28 | Configuration & project model | 2,500 | Project detection, language identification, test discovery, build configuration. |
| B29 | Logging, diagnostics, error handling | 2,000 | Structured logging, progress reporting, error recovery. |
| B30 | Core data structures & utilities | 3,000 | Graphs, union-find, bit matrices, interning, arena allocation. |

### Approach B Total

| Layer | LoC |
|-------|-----|
| Language Front-Ends (B1–B5) | 27,000 |
| Mutation Infrastructure (B6–B10) | 26,000 |
| Boundary Analysis (B11–B14) | 20,000 |
| Contract Synthesis (B15–B18) | 17,000 |
| Verification (B19–B22) | 17,000 |
| Integration & Output (B23–B26) | 13,500 |
| Infrastructure (B27–B30) | 12,000 |
| **TOTAL** | **~132,500** |

## Verdict

Close to 150K but not quite there. And this is genuinely necessary complexity — nothing
here is padding. Multi-language support alone is ~27K LoC because Java, C, and Python
have fundamentally different type systems, memory models, and execution semantics.

---

# 3. APPROACH C — Research Platform (Extensible Framework)

Adds to Approach B:

| # | Subsystem | LoC | Description |
|---|-----------|-----|-------------|
| C1 | Plugin architecture & extension API | 4,000 | Language plugins, mutation operator plugins, solver plugins, output format plugins. Trait-based in Rust. |
| C2 | Benchmarking infrastructure | 5,500 | Automated benchmark suite: Defects4J (Java), Coreutils/SIR (C), BugsInPy (Python). Metrics: contract precision, recall, mutation score correlation, synthesis time, verification time. Statistical analysis (Wilcoxon, Vargha-Delaney). |
| C3 | Comparison framework | 4,500 | Adapters for Daikon (dynamic invariant detection), EvoSpex (evolutionary spec inference), JDoctor (Javadoc-based), SpecFuzzer. Normalize output formats for fair comparison. |
| C4 | Visualization & debugging tools | 5,000 | Mutation boundary visualizer (which mutations cluster at which program points), contract evolution viewer (how contracts refine through CEGAR iterations), kill matrix heatmaps. Terminal-based TUI + SVG export. |
| C5 | Ablation study framework | 2,500 | Systematically disable components (e.g., no higher-order mutations, no heap reasoning, no boundary analysis — fall back to naive Daikon-style) and measure impact. |
| C6 | Reproducibility infrastructure | 2,000 | Deterministic execution, seed control, result serialization, artifact packaging. |
| C7 | Documentation & specification | 3,000 | Architecture docs, algorithm specifications, API reference. (This is NOT padding — a research platform is unusable without it.) |

### Approach C Total (B + C additions)

| Component | LoC |
|-----------|-----|
| Approach B (full pipeline) | 132,500 |
| Research platform additions | 26,500 |
| **TOTAL** | **~159,000** |

---

# 4. RECOMMENDED APPROACH: B+C Hybrid

## Argument

**Approach A** is a workshop paper. It proves the concept but won't survive peer review
at PLDI/POPL — reviewers will ask "does this work beyond toy examples?" and "how does
this compare to Daikon/EvoSpex?"

**Approach B alone** is a strong tool paper but lacks the research infrastructure to
produce compelling evaluations. At top PL venues, you need: (a) comparison against
baselines, (b) ablation studies, (c) statistical rigor, (d) benchmarks on real-world
projects.

**Approach B+C** is the sweet spot: a production-grade tool with the research
infrastructure to evaluate it rigorously. The ~159K LoC is genuinely necessary.

---

# 5. LANGUAGE RECOMMENDATION: Rust

## Why Rust over OCaml or Java?

### Against Java:
- We're building a tool that analyzes Java. Analyzing Java *in* Java creates a
  bootstrapping nightmare (classpath conflicts, version skew with analysis target).
- JVM startup overhead is brutal for the mutation execution loop where we spawn
  thousands of short-lived processes.
- No algebraic data types without Valhalla (still preview). Pattern matching via
  sealed classes is verbose. IR transformations become walls of visitor boilerplate.
- GC pauses are unacceptable when we're doing bounded SMT with tight timeouts.

### Against OCaml:
- OCaml is the traditional PL-research choice and would be excellent for layers 3–5
  (boundary analysis, synthesis, verification). The IR and solver integration code
  would be beautiful.
- BUT: OCaml's ecosystem for systems integration is weak. Build system integration
  (B10), parallel execution (B27), CI/CD plugins (B24), and sandboxed process
  management (B8) are painful. The `Unix` module is thin. Cross-platform support
  (we need macOS + Linux for "laptop CPU") is worse than Rust.
- OCaml's parallelism story (OCaml 5 domains) is immature. We need real work-stealing
  parallelism for mutation execution.
- Library ecosystem is small: no tree-sitter bindings as mature as Rust's, no good
  process sandboxing crates, limited CI integration libraries.

### For Rust:
- **tree-sitter-rust bindings** are first-class. We get Java, C, and Python parsers
  essentially for free (tree-sitter grammars exist).
- **Algebraic data types + pattern matching** make IR manipulation clean. `enum` +
  `match` is precisely what you want for MuIR transformations.
- **Zero-cost abstractions** matter for the mutation execution loop. We're running
  potentially 50K+ mutant compilations — overhead per-mutant must be minimal.
- **rayon** gives us trivial data-parallelism for mutation execution on laptop CPUs.
- **Process sandboxing** via `nix` crate (Unix) is mature.
- **z3-sys / z3 crate** provides direct Z3 bindings. CVC5 bindings exist.
- **serde** makes serialization (caching, incremental analysis) trivial.
- **No GC** means deterministic performance — critical for reproducible benchmarks.
- The type system catches the bugs that would otherwise plague a 150K LoC codebase.
  Lifetime tracking prevents the dangling-reference bugs that haunt long-lived
  analysis pipelines.

### Trade-off acknowledged:
Rust has a steeper learning curve and slower prototyping velocity than OCaml for
pure symbolic manipulation. The synthesis/verification layers (B15–B22) would be
~15% less code in OCaml. But the systems layers (B6–B10, B23–B28) would be ~40%
more code in OCaml due to ecosystem gaps. Net: Rust wins for this project.

---

# 6. DETAILED SUBSYSTEM ANALYSIS (Recommended Approach)

## 6.1 Complete Enumeration with Genuine Difficulty Assessment

### SUBSYSTEM B1: Language-Agnostic IR (MuIR) — 6,000 LoC

**What makes it genuinely hard:**
- Must faithfully represent three languages with fundamentally different memory models.
  Java has garbage-collected heap with references. C has manual memory with raw
  pointers, stack allocation, and undefined behavior. Python has reference-counted
  objects with dynamic typing.
- The IR must be expressive enough that mutations on MuIR correspond to meaningful
  source-level mutations, but abstract enough that the boundary analysis and synthesis
  layers don't need language-specific logic.
- Deciding the heap model is an open research question. Too concrete (like C's byte-
  addressable memory) → Java contracts are awkward. Too abstract (like Java's object
  references) → can't express C pointer arithmetic contracts.

**Novel engineering challenges:**
- No prior tool unifies mutation analysis across Java/C/Python in a single IR. PIT is
  Java-only. Mull is C/C++-only. MutPy is Python-only. We're the first to need this.
- The IR must support "mutation slots" — annotated points where mutations can be
  injected. This is not a standard IR feature.

**Off-the-shelf:** Nothing. Must be built from scratch. LLVM IR is too low-level
(loses type information needed for contracts). JVM bytecode is too Java-specific.

**Dependencies:** None (foundational).

---

### SUBSYSTEM B2: Java Front-End — 5,500 LoC

**What makes it genuinely hard:**
- Java generics with type erasure mean the source-level types and runtime types diverge.
  Contracts must be expressed in source-level types but verified against runtime behavior.
- Lambda expressions and method references create anonymous types that must be tracked.
- Exception handling (checked + unchecked) interacts with postconditions: does the
  contract specify behavior on exceptional paths?

**Novel challenges:**
- No prior mutation tool preserves full generic type information through to contract
  synthesis. PIT works at bytecode level (types are erased).

**Off-the-shelf:** tree-sitter-java for parsing. Eclipse JDT for type resolution
(via JNI or subprocess). ~30% of this subsystem is binding code.

**Dependencies:** B1 (MuIR).

---

### SUBSYSTEM B3: C Front-End — 6,500 LoC

**What makes it genuinely hard:**
- The C preprocessor. Mutations must operate on pre-processed code but contracts must
  be expressed in terms of pre-preprocessing source. Mapping between the two is
  notoriously difficult (source locations through macro expansions).
- Undefined behavior: a mutation might turn defined behavior into UB. The boundary
  analysis must distinguish "mutation killed by crash (UB)" from "mutation killed by
  assertion failure (semantic difference)." These carry very different contract
  implications.
- Pointer aliasing: determining whether two pointers alias is undecidable in general.
  Must choose a sound approximation (Steensgaard or Andersen-style) to enable
  heap contracts.

**Novel challenges:**
- Expressing C contracts requires pointer validity predicates (`\valid(p)` in ACSL).
  No prior mutation tool connects mutation survival to pointer validity requirements.

**Off-the-shelf:** tree-sitter-c for parsing. Clang's libTooling for type resolution
and preprocessor mapping. ~35% binding code.

**Dependencies:** B1 (MuIR).

---

### SUBSYSTEM B4: Python Front-End — 5,000 LoC

**What makes it genuinely hard:**
- Dynamic typing means we can't know types statically without inference. A contract
  like `requires isinstance(x, int) and x > 0` mixes type guards with value
  constraints. The synthesis engine must handle both simultaneously.
- Python's duck typing means contracts should be structural, not nominal. "This
  function requires an iterable" not "this function requires a list."
- Decorators, metaclasses, and descriptors can change method semantics in ways that
  are opaque to static analysis.

**Novel challenges:**
- No prior contract synthesis tool targets Python with formal verification. iContractor
  and PyContract are manual annotation frameworks, not synthesizers.

**Off-the-shelf:** tree-sitter-python for parsing. Python's `ast` module (via
subprocess) for type inference hints. MyPy's type inference as an optional oracle.

**Dependencies:** B1 (MuIR), B5 (type inference, heavily).

---

### SUBSYSTEM B5: Type Reconstruction — 4,000 LoC

**What makes it genuinely hard:**
- Unification-based inference for Python must handle gradual typing (some annotations
  present, some not). Must propagate type information from: (a) explicit annotations,
  (b) call sites, (c) attribute access patterns, (d) test code (tests are often the
  best type documentation).
- Java generic instantiation with wildcards (`? extends T`, `? super T`) requires
  constraint solving.
- C typedef chains and implicit conversions must be resolved.

**Off-the-shelf:** Can leverage MyPy's inference for Python (subprocess oracle).
Java and C type resolution from their respective front-end tools.

**Dependencies:** B2, B3, B4 (all front-ends).

---

### SUBSYSTEM B6: Mutation Operator Library — 5,500 LoC

**What makes it genuinely hard:**
- Designing operators that are *contract-relevant*. Standard mutation operators (AOR,
  ROR, etc.) were designed for mutation testing adequacy, not contract inference. We
  need additional operators specifically designed to probe contract boundaries:
  - **Null injection** (Java): replace non-null value with null → tests non-null preconditions
  - **Bounds perturbation**: `a[i]` → `a[i±1]` → tests array bounds preconditions
  - **Type widening** (Python): replace `int` argument with `float` → tests type preconditions
  - **Aliasing introduction** (C): make two pointers alias → tests separation preconditions
- Higher-order mutations (composing two first-order mutations) are needed to detect
  conjunctive contract clauses. The space explodes combinatorially.

**Novel challenges:**
- Contract-relevant mutation operators are a novel contribution. Prior mutation testing
  focuses on fault detection, not specification inference.

**Off-the-shelf:** None of the standard operator sets (PIT, Major, Mull) target
contract inference. Must be designed from scratch, though the implementation pattern
(AST rewriting on MuIR) is standard.

**Dependencies:** B1 (MuIR).

---

### SUBSYSTEM B7: Mutation Scheduling & Reduction — 4,000 LoC

**What makes it genuinely hard:**
- Equivalent mutant detection: some mutations don't change observable behavior (e.g.,
  `x * 1` → `x * 1` after optimization). These pollute the survival set and produce
  spurious contract constraints. Trivial compiler equivalence (TCE) catches ~15% of
  equivalents, but detecting the rest is undecidable in general.
- The number of mutants for a real project (10K+ LoC source) can exceed 100K. Running
  all of them is infeasible on a laptop. Must select a *representative* subset that
  preserves boundary information. This is a novel optimization problem: minimize
  mutants while preserving the boundary surface.

**Novel challenges:**
- Boundary-preserving mutant reduction. Standard reduction (random sampling, operator-
  based selection) doesn't guarantee that the boundary surface is preserved. We need
  a new algorithm that samples densely near boundaries and sparsely in interiors.

**Off-the-shelf:** TCE from literature. Dominator analysis from standard compiler
infrastructure.

**Dependencies:** B6 (operators), B1 (MuIR).

---

### SUBSYSTEM B8: Mutant Compilation & Sandboxed Execution — 7,000 LoC

**What makes it genuinely hard:**
- Each language has different compilation and execution characteristics:
  - Java: compile with `javac`, run on JVM. JVM startup is ~200ms — dominates for
    small tests. Must use persistent JVM with class reloading.
  - C: compile with `clang/gcc`, run native. Mutants can segfault, infinite loop,
    corrupt memory. Need seccomp/sandbox.
  - Python: no compilation, but import caching means module reload is tricky. Must
    isolate module state between mutant runs.
- Crash classification for C: distinguish segfault (likely null-deref precondition
  violation) from SIGFPE (division-by-zero precondition) from timeout (possible
  infinite loop postcondition) from assertion failure (explicit check).
- Memory limits: C mutants with memory leaks must be killed before they exhaust
  system memory.

**Novel challenges:**
- No prior tool needs crash-classified mutation results for contract inference. The
  crash type directly maps to contract clause type.

**Off-the-shelf:** `nix` crate for process isolation on Unix. `wait4` for resource
measurement. JVM class reloading via custom classloader (must be built).

**Dependencies:** B6, B7 (mutant generation), B9, B10 (test/build integration).

---

### SUBSYSTEM B9: Test Framework Adapters — 4,500 LoC

**What makes it genuinely hard:**
- JUnit 4 and JUnit 5 have completely different architectures (runners vs. extensions).
  Must support both since real-world Java projects use a mix.
- pytest's fixture system means test functions have implicit dependencies. Must resolve
  fixtures to understand test isolation.
- Google Test's parameterized tests generate multiple test instances from one source
  definition. Must track which parameter kills which mutant.
- Flaky tests: a test that fails intermittently will show as "mutation killed" when
  actually the mutation is irrelevant. Must detect and exclude flaky tests (run
  multiple times on unmodified source).

**Off-the-shelf:** Can invoke test frameworks as subprocesses and parse their output
formats (XML, JSON). ~50% of this is output parsing.

**Dependencies:** B8 (execution), B10 (build system).

---

### SUBSYSTEM B10: Build System Integration — 5,000 LoC

**What makes it genuinely hard:**
- Maven and Gradle have different project models. Must detect which is in use, parse
  dependencies, and inject mutated source without breaking the build.
- CMake's generated build files make it hard to surgically replace a single source file
  with a mutant. Must either: (a) modify CMakeLists.txt (fragile), or (b) intercept
  the compile command (requires compile_commands.json).
- Incremental compilation: after mutating one function, ideally only recompile that
  translation unit. Requires understanding the build system's dependency graph.

**Off-the-shelf:** Can shell out to build tools. compile_commands.json for C/C++ via
CMake. Maven/Gradle model parsing from existing libraries.

**Dependencies:** B2, B3, B4 (front-ends for project detection).

---

### SUBSYSTEM B11: Kill/Survive Matrix — 3,500 LoC

**What makes it genuinely hard:**
- For a project with 1,000 tests and 50,000 mutants, the full matrix has 50M entries.
  Must use sparse representation (most mutants are killed by very few tests).
- Incremental updates: when source changes, some mutants become invalid. Must
  efficiently invalidate and recompute affected rows/columns.

**Off-the-shelf:** Sparse matrix libraries exist but need custom serialization for
our specific mutation + test metadata.

**Dependencies:** B8, B9 (execution results).

---

### SUBSYSTEM B12: Mutation Boundary Extraction — 6,000 LoC

**What makes it genuinely hard (THIS IS THE CORE CONTRIBUTION):**
- The "boundary" is not a simple line in mutation space. For a single program point,
  there may be multiple mutation operators, each with multiple instantiations. The
  boundary is a *surface* in this high-dimensional space.
- Spatial clustering: mutations at the same program point interact. `x >= 0` with
  mutations `>=` → `>`, `>=` → `==`, `>=` → `!=` gives four data points. The boundary
  between killed and survived reveals the implicit precondition.
- Semantic clustering: mutations with the same *semantic effect* across different
  program points may reveal the same contract clause. E.g., all bounds checks being
  off-by-one suggests a systematic contract about array bounds.
- The boundary must be *stable* — small changes in the test suite should produce small
  changes in the boundary. This requires smoothing/regularization.

**Novel challenges:**
- This algorithm does not exist in any prior work. Mutation testing literature studies
  kill/survive as a binary metric. We're using the *geometry* of kill/survive in
  mutation space as a specification signal. This is the core novelty.

**Off-the-shelf:** Nothing. Entirely novel algorithm.

**Dependencies:** B11 (kill/survive matrix), B6 (operator semantics).

---

### SUBSYSTEM B13: Boundary → Contract Constraint Translation — 5,500 LoC

**What makes it genuinely hard:**
- The boundary gives us *observational* information: "the test suite behaves the same
  for these two program variants." Must translate this into *logical* constraints:
  "the precondition must/must not include this clause."
- Formal soundness: when is it valid to infer a precondition from mutation survival?
  Need a formal argument: if mutation M at point P survives all tests, and M changes
  the behavior on inputs in region R, then the tests don't exercise region R, so the
  precondition *may* exclude R. But this is only valid if the tests are deterministic
  and the mutation is semantically non-trivial.
- Must handle negation: a killed mutation means the test suite *does* check that
  behavior, so the contract *should* include the corresponding clause.

**Novel challenges:**
- The formal correspondence between mutation survival and contract clauses is the
  paper's main theorem. The implementation must faithfully realize this correspondence,
  including all corner cases (equivalent mutants, subsumed mutants, flaky tests).

**Off-the-shelf:** Nothing. Novel translation.

**Dependencies:** B12 (boundaries), B14 (WP/SP engine for semantic interpretation).

---

### SUBSYSTEM B14: WP/SP Engine — 5,000 LoC

**What makes it genuinely hard:**
- Weakest precondition computation through loops requires invariants — which is what
  we're trying to synthesize. Circular dependency. Solution: bounded unrolling for
  WP, then use boundary analysis to *suggest* invariants for the inductive step.
- Heap operations (allocation, field access, array access) require a heap model in
  the WP logic. Must choose between: (a) flat memory model (simple but imprecise),
  (b) points-to model (precise but complex), (c) separation logic (most precise but
  requires frame rule reasoning).
- Exception handling in WP: exceptional postconditions are distinct from normal
  postconditions. Must track both.

**Off-the-shelf:** Can adapt existing WP calculus implementations, but none handle
MuIR's three-language semantics. Must be built from scratch on well-known principles.

**Dependencies:** B1 (MuIR).

---

### SUBSYSTEM B15: SyGuS Grammar Construction — 4,500 LoC

**What makes it genuinely hard:**
- The grammar must be large enough to express useful contracts but small enough that
  synthesis terminates in reasonable time. This is the *grammar design problem* for
  SyGuS and it's notoriously sensitive — too large and CVC5 times out, too small
  and you miss important contracts.
- Must incorporate boundary data: if the boundary analysis says "the test suite checks
  that `x >= 0` but not `x > 0`," the grammar should include `x >= 0` and `x > 0`
  as candidate atoms. This "boundary-guided grammar" is novel.
- Language-specific idioms: Java contracts reference `null`, C contracts reference
  pointer validity, Python contracts reference type checks. The grammar must include
  these as first-class atoms.

**Novel challenges:**
- Boundary-guided grammar construction. Prior SyGuS work uses either fixed grammars
  or example-guided enumeration. Using mutation boundary data to focus the grammar
  is novel.

**Off-the-shelf:** SyGuS-IF standard format. Grammar structure from SyGuS competition
benchmarks.

**Dependencies:** B12, B13 (boundary data), B1 (types from MuIR).

---

### SUBSYSTEM B16: SyGuS Solver Integration + Custom CEGIS — 5,000 LoC

**What makes it genuinely hard:**
- Standard SyGuS is "synthesize a function from input/output examples." Our problem
  is different: "synthesize a predicate such that all killed mutants violate it and
  all surviving mutants satisfy it (or vice versa)." This requires a custom CEGIS
  loop where counterexamples are *mutants*, not input/output pairs.
- CVC5's SyGuS solver can timeout on large grammars. Must implement fallback
  strategies: (a) grammar splitting (synthesize clauses independently), (b)
  enumerative search for small grammars, (c) template-based synthesis for common
  patterns.
- Must handle the case where no contract exists in the grammar: detect timeout vs.
  unsatisfiability and report accordingly.

**Novel challenges:**
- Mutation-driven CEGIS: using surviving mutants as counterexamples for synthesis.
  This is a novel synthesis paradigm.

**Off-the-shelf:** CVC5 via SMT-LIB/SyGuS-IF serialization. The CEGIS loop and
mutant-as-counterexample logic is entirely novel.

**Dependencies:** B15 (grammar), B13 (constraints), B19 (SMT for CEGAR).

---

### SUBSYSTEM B17: Contract Minimization — 3,500 LoC

**What makes it genuinely hard:**
- Minimization requires checking logical implication between contract clauses, which
  is itself an SMT problem. For N clauses, there are O(N²) implication checks.
- Must balance minimality with readability. `x >= 0 && x <= 100 && x != 50` might
  be minimal but a developer would prefer `x in [0, 100] \ {50}` — but expressing
  this requires set notation the target annotation language may not support.
- Quantified contracts (e.g., `forall i. 0 <= i < n => a[i] >= 0`) require
  quantifier instantiation for minimization.

**Off-the-shelf:** Implication checking via Z3. Minimization heuristics from the
specification inference literature (Daikon's splitter-based simplification).

**Dependencies:** B16 (synthesized contracts), B19 (SMT for implication).

---

### SUBSYSTEM B18: Loop Invariant Inference — 4,000 LoC

**What makes it genuinely hard:**
- Loop invariants are harder than pre/postconditions because they must be inductive:
  true on entry, preserved by the loop body, and strong enough to prove the
  postcondition.
- Houdini-style iterative weakening starts with a large candidate set and removes
  non-inductive candidates. The key innovation: use mutation boundary data to
  generate the initial candidate set. Mutations at the loop head that survive
  indicate which invariant clauses are untested.
- Nested loops require nested invariants. The interaction between inner and outer
  invariants complicates the fixed-point computation.

**Novel challenges:**
- Mutation-seeded invariant candidates. Prior work (Houdini, ICE learning, CEGIS)
  generates candidates from templates or examples. Using mutation boundaries as
  the candidate source is novel.

**Off-the-shelf:** Houdini algorithm is well-known. ICE learning from literature.

**Dependencies:** B12 (boundary data), B14 (WP/SP), B19 (SMT verification).

---

### SUBSYSTEM B19: Bounded SMT Verification — 5,500 LoC

**What makes it genuinely hard:**
- Encoding MuIR semantics into SMT-LIB faithfully. Must handle: integer overflow
  (bitvector vs. mathematical integers — choice affects soundness), floating-point
  (IEEE 754 theory is expensive), arrays (theory of arrays with extensionality),
  algebraic datatypes (for variant types and option types).
- Bounded loop unrolling: must choose bound k. Too small → misses bugs. Too large →
  solver timeout. Adaptive strategy: start with k=5, increase if solver returns SAT
  quickly.
- Soundness: bounded verification is inherently incomplete (can prove contracts
  *wrong* but can't fully *verify* them). Must be transparent about this limitation.
  Can optionally connect to unbounded tools (Dafny, Frama-C) for full verification.

**Off-the-shelf:** Z3 (via z3 crate). SMT-LIB serialization is standard. The encoding
of MuIR → SMT is custom.

**Dependencies:** B1 (MuIR), B14 (WP/SP), B16 (synthesized contracts).

---

### SUBSYSTEM B20: CEGAR Loop — 3,500 LoC

**What makes it genuinely hard:**
- Counterexamples from the SMT solver are concrete inputs. Must generalize them into
  new synthesis constraints. E.g., if `x = -1` is a counterexample to `requires x >= 0`,
  the constraint is not just "fails on -1" but "fails on negative integers."
- Termination: the CEGAR loop may not converge if the contract is not expressible in
  the current grammar. Must detect divergence and report gracefully.

**Off-the-shelf:** CEGAR is a standard paradigm. Implementation is custom to our
mutation-specific setting.

**Dependencies:** B16 (synthesis), B19 (verification).

---

### SUBSYSTEM B21: Heap Reasoning — 5,000 LoC

**What makes it genuinely hard:**
- Separation logic (even decidable fragments) is genuinely complex to implement.
  Must handle: points-to assertions, separating conjunction, list/tree predicates
  (inductively defined), magic wand (for frame rule).
- The interaction between heap reasoning and SMT is non-trivial. Must either: (a)
  encode separation logic into first-order logic (lossy), or (b) implement a
  custom decision procedure for the heap fragment.
- Java, C, and Python have different heap models. Java: no pointer arithmetic, GC
  ensures no dangling references. C: arbitrary pointer arithmetic, dangling pointers,
  stack vs. heap. Python: reference counting, everything is heap-allocated.

**Off-the-shelf:** Can use ideas from Smallfoot, jStar, Infer for separation logic
fragments. Must be reimplemented for MuIR.

**Dependencies:** B1 (MuIR heap model), B14 (WP/SP), B19 (SMT).

---

### SUBSYSTEM B22: Soundness Certification — 3,000 LoC

**What makes it genuinely hard:**
- Must produce machine-checkable proof witnesses, not just "the solver said UNSAT."
  Z3's proof output is notoriously verbose and unstable across versions.
- Certificate format must be independent of the solver used, so users can verify
  with a different tool.

**Off-the-shelf:** Z3 proof logging. LFSC (Lean-friendly proof format from CVC5).

**Dependencies:** B19 (verification results).

---

### SUBSYSTEMS B23–B30, C1–C7: Integration & Research Platform

These are described above with LoC estimates. Their difficulty is primarily
engineering (not algorithmic) but they represent genuine, necessary complexity:

- **B23** (3,000 LoC): JML/ACSL/Python stub generation — format compliance is fiddly.
- **B24** (3,500 LoC): CI integration — must handle diverse CI environments.
- **B25** (3,000 LoC): Latent bug reporting — witness generation and formatting.
- **B26** (4,000 LoC): Incremental analysis — cache invalidation is "one of the two
  hard problems."
- **B27** (4,500 LoC): Parallel execution — work-stealing scheduler with memory bounds.
- **B28** (2,500 LoC): Configuration — project model detection heuristics.
- **B29** (2,000 LoC): Logging/diagnostics — structured logging, progress bars.
- **B30** (3,000 LoC): Core data structures — graphs, arena allocation, interning.
- **C1** (4,000 LoC): Plugin architecture — trait-based extension points.
- **C2** (5,500 LoC): Benchmarking — Defects4J/Coreutils/BugsInPy integration.
- **C3** (4,500 LoC): Comparison framework — Daikon/EvoSpex/JDoctor adapters.
- **C4** (5,000 LoC): Visualization — TUI + SVG mutation boundary viewer.
- **C5** (2,500 LoC): Ablation framework — systematic component disabling.
- **C6** (2,000 LoC): Reproducibility — deterministic execution, artifact packaging.
- **C7** (3,000 LoC): Documentation/specification — architecture, API, algorithms.

---

# 7. DEPENDENCY GRAPH

```
                    ┌──────────┐
                    │  B1: MuIR │
                    └─────┬────┘
              ┌───────────┼───────────┐
              ▼           ▼           ▼
        ┌─────────┐ ┌─────────┐ ┌─────────┐
        │B2: Java │ │B3: C FE │ │B4: Py FE│
        └────┬────┘ └────┬────┘ └────┬────┘
             └───────────┼───────────┘
                         ▼
                   ┌───────────┐
                   │B5: TypeInf│
                   └─────┬─────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
    ┌──────────┐  ┌───────────┐  ┌──────────┐
    │B6: MutOps│  │B10: Build │  │B9: Tests │
    └────┬─────┘  └─────┬─────┘  └────┬─────┘
         ▼              │              │
    ┌──────────┐        │              │
    │B7: Sched │        │              │
    └────┬─────┘        │              │
         └──────────────┼──────────────┘
                        ▼
                  ┌───────────┐
                  │B8: ExecSB │
                  └─────┬─────┘
                        ▼
                  ┌───────────┐
                  │B11: Matrix│
                  └─────┬─────┘
                        ▼
                  ┌───────────┐          ┌──────────┐
                  │B12: Bound.│◄─────────│B14: WP/SP│
                  └─────┬─────┘          └────┬─────┘
                        ▼                     │
                  ┌───────────┐               │
                  │B13: B→Cnst│◄──────────────┘
                  └─────┬─────┘
                        ▼
                  ┌───────────┐
                  │B15: Grammar│
                  └─────┬─────┘
                        ▼
         ┌────────────────────────────┐
         │B16: SyGuS + Custom CEGIS  │
         └──────┬─────────────────────┘
                │              ▲
                ▼              │ (CEGAR loop)
         ┌───────────┐  ┌───────────┐
         │B17: Minim.│  │B20: CEGAR │
         └─────┬─────┘  └─────┬─────┘
               │               │
               ▼               ▼
         ┌───────────────────────────┐
         │   B19: SMT Verification   │
         │   B21: Heap Reasoning     │
         └──────────┬────────────────┘
                    ▼
         ┌───────────────────────────┐
         │ B22: Soundness Cert       │
         │ B23: Annotation Emitters  │
         │ B25: Latent Bug Reporter  │
         └───────────────────────────┘
```

**Critical path:** B1 → B2/B3/B4 → B6 → B7 → B8 → B11 → B12 → B13 → B15 → B16 → B19

**Parallelizable early work:**
- B2, B3, B4 (front-ends) can be built in parallel after B1
- B9, B10 (test/build integration) can parallel with B6, B7
- B14 (WP/SP) can parallel with B6–B11 (mutation infrastructure)
- C1–C7 (research platform) can parallel with B19–B22 (verification)

---

# 8. HONEST LOC ASSESSMENT

| Component | LoC | Genuine? |
|-----------|-----|----------|
| Language front-ends (B1–B5) | 27,000 | **YES.** Three languages with different type systems. Anyone who has built multi-language analysis tools will confirm this. |
| Mutation infrastructure (B6–B10) | 26,000 | **YES.** Sandboxed execution across three languages is hard systems engineering. |
| Boundary analysis (B11–B14) | 20,000 | **YES.** This is novel algorithm development. The 6K for B12 is dense algorithmic code. |
| Contract synthesis (B15–B18) | 17,000 | **YES.** Custom CEGIS with mutation oracle is novel. Loop invariants add real complexity. |
| Verification (B19–B22) | 17,000 | **YES.** Heap reasoning alone justifies 5K. |
| Integration (B23–B26) | 13,500 | **MOSTLY.** B24 (CI integration) could be trimmed to 2K if we skip Jenkins/GitLab. |
| Infrastructure (B27–B30) | 12,000 | **YES.** Parallel execution on laptop CPUs needs careful work-stealing. |
| Research platform (C1–C7) | 26,500 | **MOSTLY.** C7 (docs, 3K) is borderline — but for a research platform, documentation IS the product. C4 (visualization, 5K) could be trimmed to 3K if we skip the TUI and do SVG-only. |

### Final honest number:

- **Rock-solid necessary:** ~130K LoC (everything except CI trimming and vis trimming)
- **Defensible as necessary:** ~155K LoC (full B+C as specified)
- **If challenged to cut:** Could reach 120K by dropping one language (Python) and
  the research platform. Below 120K requires cutting the core contribution.

### Is 150K genuinely necessary?

**Yes, IF you want multi-language support and a research platform.** The core insight
(boundary analysis + SyGuS + SMT, single language) is ~25K. Adding each language costs
~15K (front-end + build + test + execution). Adding verification with heap reasoning
costs ~17K. Adding the research platform costs ~25K. The arithmetic works:
25K + 3×15K + 17K + 25K + 12K infra = **~124K minimum for B+C scope.**

The remaining ~30K to reach 155K comes from: CI integration, incremental analysis,
contract minimization, soundness certification, and plugin architecture — all of which
a PLDI reviewer would expect in a mature tool.

**The 150K target is achievable without padding, but only with multi-language support.**
A single-language version tops out at ~65K with the research platform.

---

# 9. BUILD TIMELINE ESTIMATE (for resource planning)

| Phase | Duration | Subsystems | Team Size |
|-------|----------|------------|-----------|
| Foundation | Months 1–2 | B1, B14, B30 | 2 |
| Front-ends | Months 2–4 | B2, B3, B4, B5 | 3 (1 per language) |
| Mutation | Months 3–5 | B6, B7, B8, B9, B10, B27 | 2 |
| Core algorithm | Months 4–7 | B11, B12, B13 | 2 (senior) |
| Synthesis | Months 6–8 | B15, B16, B17, B18 | 2 |
| Verification | Months 7–9 | B19, B20, B21, B22 | 2 |
| Integration | Months 8–10 | B23, B24, B25, B26, B28, B29 | 2 |
| Research platform | Months 9–11 | C1–C7 | 2 |
| Evaluation | Months 11–12 | Run benchmarks, write paper | Full team |

**Estimated total:** 12 months, team of 3–4 experienced PL researchers/engineers.
