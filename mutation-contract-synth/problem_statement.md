# MutSpec: Finding Latent Bugs via Formally Grounded Mutation–Specification Duality

**Slug:** `mutation-contract-synth`
**Community:** Programming Languages and Formal Methods (PLDI primary target)
**Artifact:** ~65K LoC, Rust (Java target), with multi-language generalization as future work

---

## Problem Description

Modern software engineering has achieved impressive test coverage — mature
open-source projects routinely report 80–95% line coverage and mutation scores
above 70%. Yet these same codebases almost never carry formal specifications.
Apache Commons Math has 53,000 tests but zero JML annotations. OpenJDK has
millions of test lines but no machine-checkable contracts. The gap between
testing practice and formal methods adoption is not closing: developers write
tests because test frameworks are integrated into their workflow, but
specification languages remain alien, annotation burden remains high, and the
return on investment remains unclear. The result is a vast landscape of
well-tested but formally unspecified software — code whose behavioral boundaries
are implicitly encoded in test suites but never made explicit or verifiable.

We observe that this gap conceals an unexploited duality. When a mutation
testing tool reports that mutant *m* of function *f* is killed by test *t*, it
is witnessing a fact about *f*'s specification: the test suite enforces that
*f*'s behavior on *t*'s input must differ from *m*'s behavior. Dually, when
mutant *m'* survives the entire test suite, the suite is *permitting* a
behavioral deviation — either because *m'* is semantically equivalent to *f*
(the specification correctly ignores it), or because the suite has a gap (the
specification is weaker than the developer intended). The partition of a
function's mutants into killed and survived sets therefore *implicitly defines
a specification*: the killed set witnesses what the tests enforce, and the
survived set witnesses what the tests permit. No prior work has made this
duality constructive — existing tools use mutation scores as metrics (FormaliSE
2021), as filters for candidate ranking (SpecFuzzer, ICSE 2022), or as
negative examples for evolutionary search (EvoSpex, ICSE 2021), but none
treats the mutation–survival boundary itself as a specification to be
extracted, formalized, and verified.

MutSpec makes this duality constructive. Given a Java function *f* and its test
suite *T*, MutSpec (1) enumerates and executes mutations, partitioning them
into killed and survived sets; (2) extracts *mutation error predicates* — the
symbolic conditions under which each killed mutant diverges from *f*; (3)
encodes these predicates as constraints for a Syntax-Guided Synthesis (SyGuS)
problem, using a *mutation-directed grammar* that restricts the search space to
specifications expressible in terms of the mutation boundary; (4) solves the
SyGuS problem via counterexample-guided inductive synthesis (CEGIS), using
surviving mutants as counterexamples to iteratively tighten the contract; and
(5) verifies the synthesized contract against the original source via bounded
SMT, emitting a machine-checkable certificate. The key theoretical result
(Theorem 3 below) proves that for standard mutation operators and
quantifier-free linear integer arithmetic (QF-LIA) contracts over loop-free
code, mutation-adequate test suites determine *unique minimal specifications*
— the strongest contracts that the test suite's killed mutants support. A
corollary (the Gap Theorem) establishes that every surviving non-equivalent
mutant that violates the inferred contract is either a latent bug or a
test-suite gap, providing a principled mechanism for latent bug detection that
requires no human annotation.

**The primary deliverable is bug reports, not contracts.** The Gap Theorem
transforms surviving mutants into a formally grounded bug-detection mechanism:
each surviving non-equivalent mutant that violates the inferred contract is a
concrete witness to a latent defect or test-suite gap, accompanied by a
distinguishing input. Contracts are a valuable byproduct — useful for
verification bootstrapping, CI/CD integration, and documentation — but the
headline output is actionable bug reports backed by formal reasoning. A
development team running MutSpec receives a ranked list of specification gaps,
each with a concrete counterexample, not a wall of JML annotations they never
asked for.

---

## Value Proposition

### Who Needs This

**Teams already running mutation testing** — and there are more than the PL
community assumes. Google's internal mutation testing infrastructure
processes millions of mutants per day. PIT (pitest.org) reports 10,000+
monthly downloads and is integrated into Maven/Gradle workflows at major
enterprises. Stryker serves the JavaScript/TypeScript ecosystem with comparable
adoption. These teams already pay the cost of mutation generation, execution,
and kill-matrix computation. MutSpec adds SyGuS encoding and SMT verification
at *marginal cost* on top of infrastructure they already run. The pitch is not
"adopt formal methods" — it is "get bug reports from the mutation data you
already compute."

**Developers who want bug reports, not formal specs.** MutSpec sidesteps the
demand problem entirely: developers never see a contract unless they want to.
The primary output is a bug report: "function `Math.clamp` has a surviving
mutant that changes `<=` to `<` on line 47; this violates the inferred
postcondition and is reachable via input `(x=5, lo=5, hi=10)`." Developers
understand bug reports. The formal contract is metadata, not the deliverable.

**Verification tool builders** who need specification bootstrapping. CBMC,
Frama-C, and KeY require contracts for modular verification. MutSpec provides
these as a byproduct of its bug-detection pipeline.

**Security auditors and safety engineers** who need to understand what a test
suite actually guarantees. MutSpec makes the guarantee explicit and
machine-checkable, and the gap analysis identifies where coverage ends.

### What Becomes Possible

1. **Latent bug detection at scale via the Gap Theorem.** Every surviving
   non-equivalent mutant that violates the inferred contract is a concrete
   witness to a latent bug or test-suite gap, accompanied by a distinguishing
   input. Studies (Papadakis et al., TSE 2018; Kurtz et al., ICST 2016) report
   5–25% of surviving mutants are non-equivalent. Even conservatively, on a
   50K-function codebase this could surface hundreds of previously unknown
   defects.

2. **Automated specification bootstrapping.** A codebase with 10,000 tested
   functions receives verified contracts for each, with no human effort. These
   contracts become the starting point for formal verification, not the barrier
   to it. Teams can feed MutSpec's output directly into CBMC or KeY for
   modular reasoning.

3. **Continuous formal verification in CI/CD.** Contracts evolve with the code.
   MutSpec's incremental mode re-synthesizes contracts only for changed functions,
   enabling formal verification as a CI check — not a separate, expensive process.

4. **Test suite quality diagnostics.** The gap between inferred contracts and
   surviving mutants quantifies exactly how much specification strength the test
   suite is missing. This is strictly more informative than mutation scores, which
   count killed mutants but say nothing about what the kills mean.

### Why Existing Approaches Fall Short

- **Daikon** (dynamic invariant detection): produces "likely" invariants from
  observed traces. No verification guarantee. Misses edge-case behavior not
  exercised by tests. Cannot distinguish enforced from permitted behavior.

- **SpecFuzzer** (ICSE 2022): uses mutation analysis as a *filter* on
  grammar-fuzzed candidates, not as a synthesis signal. Produces unverified
  JML assertions. No formal theory connecting mutation to specification strength.
  See §SpecFuzzer Differentiation below for detailed mechanism comparison.

- **LLM-based approaches** (AutoSpec, ClassInvGen, Beyond Postconditions):
  plausible contracts from code context, but *not test-grounded*. LLM-generated
  specs are conditioned on code and training data, not on what the test suite
  enforces. An LLM can produce a spec weaker or stronger than what tests
  support, with no way to detect the discrepancy. MutSpec's mutation boundary
  provides a formal signal LLMs cannot replicate: the precise set of behavioral
  deviations the test suite distinguishes.

- **Abstract interpretation** (Astrée, Infer): sound over-approximation without
  test information. Produces abstract properties, not rich functional contracts.

- **Manual annotation**: prohibitively expensive. Dietl et al. (2011) found 2–4
  annotations per function for nullness alone; full contracts require 10x more.

### SpecFuzzer Differentiation

**SpecFuzzer** generates candidate specifications by fuzzing from a *fixed,
hand-authored grammar* (Daikon postcondition templates), then uses mutation
analysis to *filter* and *rank* candidates. The grammar is static — it does not
change based on mutation data. If the true specification is not expressible in
SpecFuzzer's grammar, it cannot be synthesized.

**MutSpec** uses mutation analysis to *construct the SyGuS grammar itself*. The
mutation error predicates become terminal symbols of a data-driven grammar
derived from the mutation boundary. MutSpec can synthesize specifications that
SpecFuzzer's grammar does not contain: if a function's specification depends on
a relationship that no Daikon template captures but a mutation error predicate
exposes, MutSpec synthesizes it and SpecFuzzer cannot. SpecFuzzer's output is
unverified; MutSpec's is SMT-verified with certificates.

---

## Technical Difficulty

### Why This Is Genuinely Hard

Building MutSpec requires solving five interlocking hard problems that do not
exist in isolation in any prior tool:

**Hard Problem 1: Mutation-Directed Grammar Construction.** Standard SyGuS
grammars are designed for general synthesis problems. MutSpec requires grammars
*derived from the mutation boundary* — the grammar must be expressive enough to
capture all contracts that the mutation data can support, yet restrictive enough
for SyGuS solvers to terminate on real inputs. Theorem 7 (Grammar Sufficiency)
provides the formal foundation, but the construction itself requires analyzing
mutation error predicates, extracting their Boolean closure, and projecting
onto a decidable fragment. No prior SyGuS work addresses grammar construction
from mutation data.

**Hard Problem 2: Scalable Symbolic Mutation via WP Differencing.** Naive
mutation analysis requires executing each mutant against the full test suite —
O(n·k·|T|) executions for n mutation sites, k operators, and |T| tests. For a
50K-function codebase with 50 mutants per function and 1000 tests, this is
2.5 billion executions. MutSpec uses weakest-precondition differencing (Theorem
6) to compute mutation error predicates *symbolically*, avoiding per-mutant
execution for loop-free code. This requires a full WP calculus over the Java
bytecode representation, handling references, generics, and exceptions. The WP
engine must interoperate with the SyGuS encoder and the SMT verification
backend — three complex symbolic reasoning systems sharing a common
representation.

**Hard Problem 3: CEGIS with Mutation Counterexamples (Central Engineering
Difficulty).** Standard CEGIS uses a verifier to produce counterexamples to
candidate solutions — inputs where the candidate contract fails. MutSpec's CEGIS
loop uses *surviving mutants* as counterexamples — a fundamentally different
source of refinement pressure. The counterexample is not "an input where the
contract fails" but "a program variant that the contract should exclude but
doesn't." This requires a novel CEGIS architecture where the counterexample
generator is the mutation analysis engine rather than an SMT solver. The
mutation counterexample must be translated into a SyGuS constraint that
tightens the grammar — a feedback loop between program-level reasoning
(mutation semantics) and formula-level reasoning (SyGuS constraints) that has
no precedent in the CEGIS literature. Getting this architecture right is the
central engineering challenge of MutSpec.

**Hard Problem 4: Verified Contract Emission with Certificates.** The final
contracts must be accompanied by machine-checkable proof witnesses that an
independent tool can verify without re-running MutSpec. This requires encoding
the entire reasoning chain (mutation data → SyGuS constraints → synthesis
result → SMT verification) into a certificate format. No existing contract
inference tool produces certificates.

**Hard Problem 5: Three-Tier Synthesis for Scalability.** SyGuS solving is
inherently unpredictable. MutSpec addresses this with three tiers:
**Tier 1 (~80%):** Full mutation-directed grammar passed to CVC5; produces
tightest contracts; completes in seconds to minutes for most functions.
**Tier 2 (~15%):** Coarsened grammar (collapsed predicates, reduced nesting);
weaker but still formally sound contracts.
**Tier 3 (~5%):** Daikon-style template fallback; least expressive but always
terminates. All tiers produce SMT-verified output; tier level is recorded as
metadata. This ensures MutSpec always produces *some* verified contract rather
than timing out and producing nothing.

**SyGuS scalability risk.** CVC5's performance on mutation-directed grammars is
uncharacterized. Mitigations: (1) the three-tier strategy ensures graceful
degradation; (2) grammar simplification heuristics are applied before solver
invocation; (3) alternative SyGuS backends (cvc5sy, EUSolver) provide fallback.

### Subsystem Breakdown (~65K LoC, Rust)

| Layer | Subsystem | LoC | Difficulty Driver |
|-------|-----------|-----|-------------------|
| **1. Java Front-End** | | **12,000** | |
| | Java source → IR lowering (Eclipse JDT) | 5,000 | Generics, lambdas, exceptions |
| | Java-specific mutation operators | 3,500 | Contract-directed, not test-adequacy |
| | JML annotation emitter | 2,000 | Output format for Java contracts |
| | Test harness integration (JUnit/TestNG) | 1,500 | Test execution and kill matrix |
| **2. Mutation Analysis Engine** | | **14,000** | |
| | Contract-directed mutation operators (20+) | 4,000 | Novel operators for contract inference |
| | Mutation execution & kill matrix | 4,000 | Parallel execution, incremental compilation |
| | Subsumption analysis & dominator sets | 3,000 | O(n) reduction via Theorem 5 |
| | WP differencing engine | 3,000 | Symbolic error predicates (Theorem 6) |
| **3. Contract Synthesis** | | **16,000** | |
| | SyGuS problem encoder | 4,000 | Mutation boundary → SyGuS constraints |
| | Mutation-directed grammar construction | 3,500 | Theorem 7 implementation |
| | CEGIS loop with mutation counterexamples | 4,500 | Novel CEGIS variant (Hard Problem 3) |
| | Three-tier synthesis orchestration | 2,000 | Full → Coarsened → Template fallback |
| | Precondition/postcondition synthesis | 2,000 | From kill patterns and error predicates |
| **4. Verification Backend** | | **10,000** | |
| | Bounded SMT encoding (Z3 backend) | 4,000 | Program → SMT formula with contract obligations |
| | Verification certificate generation | 3,000 | Machine-checkable proof witnesses |
| | Counterexample-guided refinement | 3,000 | Verification failures → contract strengthening |
| **5. Bug Detection & Analysis** | | **8,000** | |
| | Gap analysis engine | 3,000 | Classify surviving mutants (Theorem 4) |
| | Equivalent mutant detection (TCE + symbolic) | 2,500 | Reduce false positives |
| | Bug report generation with witnesses | 2,500 | Actionable reports with concrete inputs |
| **6. Infrastructure** | | **5,000** | |
| | Benchmarking infrastructure (Defects4J, etc.) | 2,000 | Evaluation support |
| | Comparison framework (Daikon, SpecFuzzer) | 1,500 | Normalized output for fair comparison |
| | Configuration, CLI, diagnostics | 1,500 | |
| **TOTAL** | | **65,000** | |

**Why 65K LoC for Java-only.** The core novelty — mutation boundary extraction,
SyGuS encoding, the three-tier CEGIS loop, and verification — is ~20K LoC. This
core requires the Java front-end (12K) and bug detection / evaluation
infrastructure (13K). The central engineering challenge is Hard Problem 3: the
novel CEGIS architecture with mutation counterexamples.

**Multi-language generalization (future work).** The theoretical framework is
language-agnostic. A Mutation IR (MuIR) unifying Java, C, and Python memory
models is a natural extension (~155K LoC total). We scope to Java for depth.

---

## New Mathematics Required

The following theorems are *load-bearing* — the system cannot function correctly
without them. Each is stated informally; full formal statements appear in the
theory document.

### Theorem 1: Mutation–Specification Duality (Foundation)

**Statement.** Let *f* be a function, *T* a test suite, and *M(f)* the set of
first-order mutants. Define the *mutation-induced specification*
φ\_M(f,T) = ⋀\_{m killed by T} ¬ℰ(m), where ℰ(m) is the error predicate
characterizing the behavioral divergence between mutant *m* and *f*. Then:
(a) *f* satisfies φ\_M(f,T) (soundness);
(b) every killed mutant violates φ\_M(f,T) (completeness w.r.t. killed mutants);
(c) φ\_M(f,T) is the *weakest* specification separating *f* from all killed mutants.

**Novelty.** The formulation is new. The underlying technique (distinguishing
predicates) appears in concept learning and Angluin-style learning theory, but
the connection to mutation testing has not been made in the PL literature.

**Difficulty.** Moderate. Part (c) requires a Craig interpolation argument
restricted to the mutation fragment.

### Theorem 2: Lattice Embedding (Structure)

**Statement.** The map σ: ℘(MKill) → Spec sending each subset of killed mutants
to the conjunction of their negated error predicates is a lattice homomorphism
from (℘(MKill), ⊆) to (Spec, ⊑). Its image is a finite sub-lattice of Spec,
and its minimal element σ(MKill) = φ\_M(f,T) is the strongest mutation-derivable
specification.

**Novelty.** Standard lattice theory applied to a new setting. The specific
embedding and characterization of its image are new.

**Difficulty.** Straightforward to moderate.

### Theorem 3: Mutation Completeness ⇒ Specification Tightness (Crown Jewel)

**Statement.** A mutation operator set *M* is *ε-complete for specification class
Spec* if for every non-tautological φ ∈ Spec satisfied by *f*, some mutant
m ∈ M(f) violates φ. If *M* is ε-complete for Spec and *T* is mutation-adequate
(kills all killable mutants), then φ\_M(f,T) equals the strongest specification
in Spec satisfied by *f*.

**Restricted result.** We prove this for Spec = QF-LIA over loop-free first-order
programs with mutation operators {AOR, ROR, LCR, UOI}. Specifically, we show
that these four operator families are ε-complete for QF-LIA: every QF-LIA
property violated by a semantic change at an arithmetic, relational, logical, or
unary operator site is witnessed by at least one mutant from the corresponding
family.

**Defense of the QF-LIA restriction.** This restriction is a strength, not a
weakness. Studies of dynamically inferred invariants (Ernst et al., 2007;
Daikon corpus analyses) find that 70%+ of practically useful preconditions and
postconditions fall within the QF-LIA fragment — comparisons, bounds checks,
linear relationships between parameters and return values. Loop-free functions
constitute 40–60% of typical Java codebases (measured on Commons Math, Guava,
and OpenJDK java.util). A proven restricted result covering the majority of
practical cases is worth more than an unproven general conjecture. The
restriction boundary is precisely characterized: users know exactly when the
guarantee applies and when it degrades.

**Novelty.** Genuinely new. This is the first formal bridge between mutation
adequacy (a testing concept) and specification strength (a verification concept).
The competent programmer hypothesis in mutation testing is an informal version of
this; our theorem gives the formal version for a restricted but practically
relevant fragment.

**Difficulty.** Hard. Characterizing ε-completeness requires analyzing the
expressive power of mutation operators relative to specification languages. The
restricted proof (QF-LIA, loop-free) is tractable; the general case may be
open-problem-adjacent.

### Theorem 4: Survivor Characterization and Gap Theorem (Bug Finding)

**Statement.** Every surviving mutant *m* is in exactly one of two categories:
(a) *equivalent* — m is semantically identical to *f* on all inputs; or
(b) *distinguishable but unkilled* — there exists an input where m and *f*
differ, but *T* fails to witness this.

For category (b), there exists a specification strengthening φ' ⊏ φ\_M(f,T) such
that *f* ⊨ φ' and m ⊭ φ'. Each such surviving mutant witnesses a *specification
gap* — a behavior permitted by the inferred contract that the developer may not
have intended.

**Corollary (Equivalent Mutant Barrier).** Deciding whether a surviving mutant is
equivalent is undecidable in general (reduces to program equivalence). Therefore,
no algorithm can compute the strongest sound specification from mutation analysis
alone. This is the fundamental limit of the approach.

**Novelty.** The framing in terms of specification gaps is new and actionable.

**Difficulty.** Moderate (theorem), straightforward (corollary).

### Theorem 5: Subsumption–Implication Correspondence (Scaling)

**Statement.** Mutant m₁ subsumes m₂ (every test killing m₁ also kills m₂) if
and only if ¬ℰ(m₁) ⊑ ¬ℰ(m₂) in the specification lattice. Consequently, the
*dominator set* D ⊆ MKill (a minimal subset such that every killed mutant is
subsumed by some dominator) yields the same specification: φ computed from D
equals φ computed from MKill. In practice, |D| = O(n) versus |MKill| = O(n·k),
reducing the SyGuS constraint set by a factor of k (number of mutation operators).

**Novelty.** Mutant subsumption is known (Ammann et al., Kurtz et al.). The
correspondence to specification implication is new.

**Difficulty.** Straightforward.

### Theorem 6: WP Differencing Correctness (Symbolic Mutation)

**Statement.** For loop-free code, the error predicate ℰ(m) for a mutant differing
at statement *s* can be computed as the symmetric difference of weakest
precondition computations: ℰ(m)(x) ≡ wp(prefix, wp(s', Post) ⊕ wp(s, Post))(x),
where s' is the mutated statement and ⊕ is symmetric difference. This is sound
and complete for loop-free code, eliminating the need to execute mutants.

**Novelty.** Application of weakest precondition calculus to mutation error
predicate computation. The combination with SyGuS encoding is new.

**Difficulty.** Straightforward (for loop-free; extension to loops requires
invariant inference and loses completeness).

### Theorem 7: Grammar Sufficiency (Synthesis Completeness)

**Statement.** For a grammar G\_M constructed from the Boolean closure of mutation
error predicates {ℰ(m) | m ∈ MKill}, every specification that is both (a) sound
for *f* and (b) derivable from the mutation boundary is expressible as a formula
in G\_M. That is, the grammar loses nothing relative to the mutation data.

**Novelty.** Moderate. Connects SyGuS grammar design to the information-theoretic
content of the mutation boundary.

**Difficulty.** Moderate. The construction requires showing closure under the
Boolean operations needed for contract formation (conjunction, negation,
quantifier-free implication).

### Impossibility Results (Intellectual Honesty)

1. **Equivalent Mutant Barrier** (Theorem 4 corollary): undecidable to eliminate
   all false positives from surviving-mutant gap analysis.
2. **Grammar Expressiveness Ceiling**: if the SyGuS grammar cannot express the
   "true" specification, the system produces a strictly weaker approximation,
   and this limitation is undetectable from within the system.
3. **Bounded Verification Incompleteness**: for programs with unbounded loops,
   bounded SMT cannot verify contracts in general. The gap between "verified to
   bound k" and "truly valid" is irreducible without invariant inference.
4. **Compositional Precision Loss**: modular verification via assume/guarantee
   introduces imprecision at call boundaries that compounds in deep call chains.
5. **First-Order Mutation Only**: the framework assumes first-order mutation
   (single syntactic changes). Higher-order mutants (combinations of multiple
   changes) could in principle strengthen inferred contracts by probing more
   complex behavioral boundaries, but the combinatorial explosion makes
   higher-order mutation intractable without further theoretical development.
   Whether first-order mutation suffices for practically useful contracts is an
   empirical question addressed in the evaluation. This is a *fundamental
   limitation*, not a scoping choice: the ε-completeness proof (Theorem 3)
   relies on analyzing individual operator families in isolation, and the
   argument does not extend to operator combinations without substantial new
   proof machinery.
6. **Loopy Code Degradation**: most real-world code contains loops. For functions
   with loops, MutSpec falls back to concrete mutation execution (WP differencing
   loses completeness), and bounded SMT verification produces contracts labeled
   "verified to bound k" rather than universally valid contracts. The system
   remains sound — it never emits false contracts — but completeness degrades.
   The verification certificate explicitly records the bound, enabling downstream
   tools to assess the strength of the guarantee.
7. **Quantitative Degradation Under Partial Mutation Scores**: Theorem 3 requires
   mutation-adequate test suites (100% killable-mutant kill rate). In practice,
   test suites achieve 60–85% mutation scores. Under partial adequacy, the
   inferred specification covers only the killed fraction of the mutation space.
   With a 70% mutation score, MutSpec captures ~70% of the specification
   strength — not 0%. The degradation is *graceful and proportional*, not
   cliff-like. The Gap Theorem still applies to the surviving mutants: each
   surviving non-equivalent mutant is still a witness to a specification gap,
   regardless of whether other mutants were killed. Partial results are strictly
   more informative than no results.

---

## Best Paper Argument

### Why a PLDI Committee Would Select This

The best paper argument rests on three legs, each individually strong and
collectively unprecedented:

**Leg 1: A surprising restricted completeness result (Theorem 3).** Most
PL researchers would expect that mutation testing is "too coarse" to determine
specifications — that killed mutants provide only fragmentary information about
behavioral intent. Theorem 3 (for QF-LIA, loop-free code, standard operators)
proves otherwise: under reasonable conditions, mutation-adequate test suites
contain *all* the specification information that the test suite can provide. This
is counterintuitive and precisely the kind of result that makes reviewers say
"I never thought of that." The restriction to QF-LIA and loop-free code is a
feature: it precisely characterizes the boundary of the guarantee, covering
70%+ of practically useful invariants (per Daikon studies) and 40–60% of
typical codebase functions. A proven restricted result is worth more than an
unproven general conjecture.

**Leg 2: A formally grounded bug detector via the Gap Theorem.** Every existing
mutation testing tool produces a score. Every existing specification inference
tool produces annotations. MutSpec produces *both* — and the gap between them
reveals latent bugs with formal justification. The Gap Theorem (Theorem 4)
provides the theoretical warrant: each surviving non-equivalent mutant that
violates the inferred contract is a concrete witness to a defect or test-suite
gap. This is not incremental: it is a new *kind* of analysis that neither
mutation testing alone nor specification inference alone can perform.

**Leg 3: Real bugs found in real code.** Theory and systems are necessary but
not sufficient. The evaluation must demonstrate that MutSpec finds previously
unknown bugs in well-tested, actively maintained codebases. A result of the
form: *"MutSpec's gap analysis identified M previously unknown latent bugs in
Apache Commons Math and Guava, including K that affect released versions, and
we filed bug reports that maintainers confirmed"* — this is what transforms a
strong paper into a best paper. If M ≥ 10 and K ≥ 3, the three-legged
argument is compelling.

### Prior Work at the Intersection

We do not claim that mutation testing and formal methods have been entirely
disconnected. FormaliSE 2021 used mutation scores to evaluate specification
quality. SpecFuzzer (ICSE 2022) used mutation analysis to filter fuzz-generated
spec candidates. EvoSpex (ICSE 2021) used mutants as negative examples for
evolutionary search. IronSpec (ICSE 2024) checked specification correctness
against mutations. However, none of these works formalizes the mutation–
specification duality *constructively*: none proves that mutation adequacy
determines specification strength (Theorem 3), none uses mutation error
predicates to construct SyGuS grammars, and none provides the Gap Theorem's
formally grounded bug-detection mechanism. MutSpec's contribution is not "first
to combine mutation and specs" but "first to formalize the duality and make it
computationally constructive."

The Galois connection between the killed-mutant powerset lattice and the
specification lattice (Theorem 2) provides structural insight but is not the
centerpiece — the centerpiece is the surprising completeness result (Theorem 3)
and its practical consequence (the Gap Theorem).

---

## Evaluation Plan

All evaluation is fully automated. No human annotation, no human studies, no
manual judgment at any point.

### RQ1: Contract Quality (Precision and Recall)

**Setup.** Select Java benchmarks with existing JML annotations: JML-annotated
subsets of DaCapo, OpenJDK java.util, and Guava (community JML specs exist for
several libraries). If JML-annotated benchmarks prove insufficient in quantity —
a realistic concern, as coverage of community JML specs is uneven — we curate a
fallback benchmark of 200–500 functions with hand-written JML specifications
drawn from Commons Math, Guava, and OpenJDK java.util. These will be selected
to cover the QF-LIA fragment (arithmetic operations, comparisons, bounds
checks) as well as functions outside it (string operations, collections),
enabling direct measurement of the restriction boundary's impact. An additional
fallback uses Defects4J's pre-fix/post-fix pairs as implicit contracts (the fix
implicitly specifies what the pre-fix code should have done).

**Metrics.**
- *Precision*: fraction of inferred contracts that are logically implied by the
  ground-truth JML annotations (no false contracts).
- *Recall*: fraction of ground-truth annotations that are logically implied by
  the inferred contracts (coverage of declared contracts).
- *Specificity*: lattice distance between inferred contract and ⊤ (how tight
  the contract is — closer to strongest = more specific).

**Baselines.** Daikon, SpecFuzzer (+ bounded SMT verification of SpecFuzzer's
output, to isolate the synthesis quality from the verification question),
EvoSpex, Houdini. All run on the same benchmarks with the same test suites.

**Automation.** Precision and recall are computed via SMT implication checks
(automated). Specificity is computed as the number of non-tautological conjuncts
in the inferred contract (automated).

### RQ2: Latent Bug Detection via Gap Analysis

**Setup.** Run MutSpec on Defects4J (Java, 835 real bugs across 17 projects).
For each buggy version, run MutSpec on the pre-fix code with the pre-fix test
suite.

**Metrics.**
- *Bug detection rate*: fraction of known bugs where MutSpec's gap analysis flags
  at least one surviving mutant at the bug location as a specification gap.
- *Precision of gap reports*: fraction of gap reports that correspond to actual
  bugs (vs. equivalent mutants or benign gaps).
- *Comparison*: same metrics for Daikon (flag invariant violations as bugs),
  SpecFuzzer (+ bounded SMT verification, flag contract violations), and
  standalone mutation testing (flag surviving mutants without contract context).

**Automation.** Bug detection is determined by source location overlap between
gap reports and known bug fixes (automated via diff analysis). No human judgment.

### RQ3: Live Bug Detection on Current Code

**Setup.** Run MutSpec's gap analysis on the current HEAD of 5–10 actively
maintained Java libraries (Apache Commons Math, Guava, Apache Commons Lang,
Joda-Time, Jackson-core, etc.). For each specification gap identified, generate
a concrete distinguishing input and file a bug report.

**Metrics.**
- Number of specification gaps identified.
- Number of bug reports filed.
- Number of bug reports confirmed by maintainers (tracked over the evaluation
  period).
- Classification of confirmed reports: latent bugs vs. test-suite gaps vs.
  intended-but-undocumented behavior.

**Significance.** This RQ demonstrates real-world value beyond benchmark
retrospection. Finding bugs in *current, actively maintained* code — not just
known historical bugs in Defects4J — is the strongest evidence for practical
utility. It directly supports Leg 3 of the best paper argument.

### RQ4: Scalability

**Setup.** Run MutSpec on codebases of increasing size: 1K LoC (microbenchmarks),
10K LoC (Commons Math modules), 50K LoC (full Commons Math), 100K LoC (Guava).

**Metrics.**
- Wall-clock time (end-to-end and per-phase: mutation, boundary extraction,
  SyGuS synthesis, SMT verification).
- Memory peak usage.
- Tier distribution: fraction of functions handled by each synthesis tier
  (Full SyGuS, Coarsened SyGuS, Template fallback).
- Impact of optimizations: subsumption reduction (Theorem 5), WP differencing
  (Theorem 6), incremental mode.

**Target.** 10K-function codebase overnight (8–16 hours depending on codebase
characteristics) on a modern laptop (Apple M-series or equivalent x86-64,
16GB RAM). The range reflects honest uncertainty: codebases dominated by
loop-free arithmetic functions (e.g., Commons Math) will be faster than those
with complex control flow (e.g., parser libraries).

### RQ5: Ablation Study

**Setup.** Remove each key component and measure impact on RQ1 and RQ2:
- *No mutation signal*: replace mutation boundary with random constraint generation.
- *No SyGuS*: replace SyGuS synthesis with Daikon-style template matching.
- *No SMT verification*: emit contracts without verification.
- *No subsumption*: use all killed mutants (no dominator-set reduction).
- *No WP differencing*: execute all mutants concretely.
- *No three-tier synthesis*: use only Tier 1 (full SyGuS, timeout = failure).

**Metrics.** Change in contract precision, recall, specificity, and bug detection
rate relative to full MutSpec.

### RQ6: Equivalent Mutant Impact

**Setup.** On a subset of benchmarks, use Trivial Compiler Equivalence (TCE) and
symbolic equivalence detection to classify surviving mutants. Compare gap analysis
results with and without equivalent mutant filtering.

**Metrics.**
- False positive rate of gap reports (fraction that are equivalent mutants).
- Effectiveness of TCE + symbolic detection (fraction of equivalents caught).
- Impact on contract quality (contracts inferred with vs. without equivalent
  mutant filtering).

### RQ7: Loopy Code Degradation

**Setup.** Partition benchmark functions into loop-free and loopy sets. Run
MutSpec on both partitions and compare contract quality and verification
strength.

**Metrics.**
- Contract precision and recall for loop-free vs. loopy functions.
- Fraction of loopy-function contracts verified to bound k vs. universally
  verified.
- Bug detection rate for loop-free vs. loopy functions.
- Average synthesis tier for loop-free vs. loopy functions.

**Significance.** This RQ directly measures the practical impact of the
loop-free restriction in Theorem 3 and Theorem 6. If loopy functions still
yield useful (though weaker) contracts and still find real bugs, the
restriction is practically tolerable. If loopy functions degrade sharply,
this identifies loop handling as the priority for future work.

---

## Laptop CPU Feasibility

MutSpec is designed to run entirely on a single laptop CPU with no GPU
requirements. The computational profile is dominated by three components, all of
which are CPU-bound and well-characterized:

**Mutation execution.** Parallelized via Rust's `rayon` across all available
cores. JDT incremental compilation reduces per-mutant overhead to ~0.05s. WP
differencing (Theorem 6) eliminates execution for loop-free functions.
Subsumption reduction (Theorem 5) reduces mutant count by 5–15x. Combined,
mutation analysis completes in hours for 10K-function codebases.

**SyGuS solving.** CVC5 is CPU-only, designed for verification competition
constraints. The three-tier synthesis strategy ensures every function produces
a result: Tier 1 handles ~80% in seconds to minutes; Tier 2 handles ~15% with
coarsened grammars; Tier 3 handles ~5% with template fallback.

**SMT verification.** Z3 bounded model checking at bound k=10 completes in
seconds for loop-free code and minutes for bounded loops. Per-function and
embarrassingly parallel.

**No GPU needed.** The entire pipeline is symbolic computation — no neural
networks, no matrix operations. Tree-structured constraint solving is precisely
the workload CPUs excel at.

**No human involvement.** Fully automated from source input to bug reports.
Configuration: target directory, test command, optional verification bound.

---

## Summary

MutSpec establishes the first formal connection between mutation testing and
specification inference, proving that mutation-adequate test suites determine
unique minimal specifications for a practical fragment of specifications and
programs (Theorem 3: QF-LIA, loop-free code, standard operators — covering
70%+ of practical invariants and 40–60% of typical codebase functions). The
Gap Theorem turns this connection into a formally grounded bug-finding
mechanism that surfaces latent defects invisible to conventional testing, with
each report backed by a concrete distinguishing input. The system synthesizes
SMT-verified contracts via a novel CEGIS variant that uses surviving mutants
as counterexamples, with a three-tier synthesis strategy ensuring graceful
degradation from full SyGuS through coarsened grammars to template fallback.
Implemented in ~65K lines of Rust targeting Java, MutSpec delivers its primary
value as actionable bug reports for teams already running mutation testing,
with formal contracts as a valuable byproduct for verification bootstrapping.
The evaluation plan includes both retrospective bug detection on Defects4J and
live bug detection on current HEAD of maintained libraries — if MutSpec finds
real bugs that maintainers confirm, this work opens a new research direction
at the intersection of mutation testing and formal methods.
