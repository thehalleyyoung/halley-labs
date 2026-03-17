# Competing Problem Framings: Mutation-Driven Contract Synthesis

> **Seed idea:** An engine that synthesizes formally verified function contracts by
> analyzing which code mutations kill existing tests, using the mutation–survival
> boundary to infer minimal pre/postconditions via SyGuS and verifying them against
> source via bounded SMT to catch latent bugs.

---

## FRAMING A — THE THEORETICAL INSIGHT

### Title: *The Mutation–Specification Galois Connection: A Lattice-Theoretic Foundation for Deriving Minimal Contracts from Test Adequacy*

### Problem & Approach

Mutation testing and specification inference have evolved as parallel research
threads with almost no formal bridge between them.  Mutation testing asks "does
this test suite detect injected faults?" while specification inference asks "what
logical property does this code satisfy?"  We establish that these two questions
are *duals* connected by a precise Galois connection between the lattice of
mutation-kill sets (ordered by inclusion) and the lattice of Hoare-style
contracts (ordered by logical strength).

Concretely, define the *mutation space* **M** of a function *f* as the set of
all syntactically reachable mutants under a fixed operator vocabulary (arithmetic
replacement, relational boundary, conditional negation, statement deletion—the
standard sufficient set).  A test suite *T* partitions **M** into killed set
*K(T)* and surviving set *S(T)*.  A contract φ = ⟨P, Q⟩ independently partitions
**M** into *Viol(φ)* (mutants that violate the contract under the precondition)
and *Sat(φ)* (mutants that still satisfy it).  We prove:

1. **Soundness:** For any contract φ validated by *T*, *K(T) ⊆ Viol(φ)*.
   Intuitively: every mutation the tests catch must also be rejected by a
   correct specification.
2. **Galois connection:** The abstraction function α: 2^**M** → Spec that maps a
   kill set to the *weakest* contract rejecting exactly those mutants, and the
   concretization γ: Spec → 2^**M** that maps a contract to its violation set,
   form a Galois connection (2^**M**, ⊆) ⇌ (Spec, ⊑).
3. **Minimality:** The fixpoint α(K(T)) yields the *weakest precondition and
   strongest postcondition* derivable from *T*'s mutation kills—the minimal
   contract that exactly captures what the tests enforce.
4. **Gap theorem:** Surviving mutants in *Viol(α(K(T)))* — i.e., mutants that
   the inferred contract rejects but tests miss — are *provably* latent bugs or
   test-suite gaps.  This gives a *completeness metric* for test suites that
   subsumes mutation score.

The constructive algorithm is counterexample-guided: we use killed mutants as
*negative examples* and surviving mutants as *positive examples* for SyGuS
synthesis over a predicate grammar, then verify candidate contracts against the
source via bounded SMT (bit-width 32, loop bound *k*).  The Galois connection
guarantees convergence to the unique fixpoint.

### Who Desperately Needs This

The formal methods community has lacked a principled theory connecting testing to
specification.  Testing researchers cannot explain *what* a test suite specifies,
only *how much* it covers.  Specification researchers cannot leverage existing
tests.  This duality collapses that wall.  Program analysis researchers get a new
abstraction for reasoning about test adequacy.  Verification researchers get a
mechanical specification generator grounded in lattice theory rather than
heuristics.

### Best-Paper Argument

This reframes two mature subfields—mutation testing (40+ years) and contract
synthesis (20+ years)—as manifestations of a single algebraic structure.  The
Galois connection is not a metaphor; it is a formal theorem with a constructive
proof-as-algorithm.  The gap theorem gives the first *mathematical* explanation
for why mutation testing finds bugs: it is a sound but incomplete approximation
of specification checking, and the incompleteness is precisely measurable.  This
is the kind of unifying result that POPL/PLDI best-paper committees select—it
changes how people *think* about the relationship between testing and
verification, not just how they build tools.

Comparable precedents: the abstract-interpretation Galois connection (Cousot &
Cousot, 1977) unified static analyses; the testing–verification spectrum
(Godefroid, 2005) unified model checking with testing.  This result fills the
missing cell: the formal connection between *mutation testing* and *specification
inference*.

### What Makes This More Than "Mutation Testing + Contract Synthesis"

The Galois connection is new.  Prior work (Daikon, SpecFuzzer, PIE) infers
invariants from *execution traces*—they observe *behaviors* and generalize.
This work infers contracts from *the structure of which mutations tests
kill*—it reasons about *perturbations* and derives the logical boundary.  The
mutation–specification duality means minimality is not a heuristic ("find a small
invariant that holds") but a theorem ("the fixpoint is the unique weakest
contract").  The gap theorem is also new: it gives a formal criterion for
detecting latent bugs that no prior specification-mining approach provides,
because it exploits the *asymmetry* between kill sets and violation sets.

---

## FRAMING B — THE PRACTICAL IMPACT

### Title: *Semantic Scaffolding: Automatically Extracting Verified Function Contracts from Legacy Test Suites to Enable Formal Reasoning at Scale*

### Problem & Approach

Formal methods have a deployment paradox: the tools exist, but the specifications
do not.  Bounded model checkers (CBMC, JBMC), deductive verifiers (Dafny,
Frama-C, KeY), and compositional analyzers (Infer, SLAM) all require function
contracts—preconditions and postconditions—as input.  Writing these contracts
manually costs 2–10× the effort of writing the code itself (verified estimates
from seL4, CompCert, and ironFleet projects).  The result: formal verification
is confined to safety-critical greenfield projects.  The other 99.9% of the
world's code—legacy C, Java, Python codebases with millions of lines and
thousands of test cases—remains unverifiable, not because the tools are
inadequate, but because nobody will write the specifications.

We solve the *specification bootstrapping problem* for legacy code.  The key
observation is that an existing test suite already encodes *implicit behavioral
expectations*—but in a form (concrete input/output pairs with oracles) that
verification tools cannot consume.  We *distill* those expectations into formal
contracts by analyzing the *boundary* between mutations that tests kill and
mutations that survive.  Killed mutants are negative examples ("this behavior is
wrong"); surviving mutants that pass all tests are positive examples ("this
behavior is acceptable, or at least untested").  We feed both into a SyGuS
synthesizer operating over a grammar of predicate expressions (linear arithmetic,
array bounds, nullness, algebraic relations) to produce candidate pre/postcondition
pairs.  We then verify each candidate against the source function using bounded
SMT (Z3, bit-width 32, configurable loop unrolling depth).

The verified contracts serve three purposes simultaneously:
1. **Bug detection:** A surviving mutant that violates the inferred contract
   exposes a *latent bug*—a behavioral deviation that tests fail to catch but
   that is provably inconsistent with the contract the tests *do* enforce.
   These are real bugs, not hypothetical: they survive the test suite.
2. **Specification documentation:** The contracts are human-readable JML/ACSL-
   style annotations that can be emitted directly into source, giving
   developers machine-checked documentation for free.
3. **Verification enablement:** The contracts serve as assume/guarantee
   specifications for downstream tools (CBMC, Frama-C, Dafny), enabling
   modular verification of codebases that previously had no entry point for
   formal reasoning.

### Who Desperately Needs This

**Every organization with legacy code and test suites but no formal specs**—which
is effectively every software organization on Earth.  Specific high-value targets:

- **Safety-critical legacy systems** (automotive, avionics, medical devices):
  regulatory standards (DO-178C, ISO 26262, IEC 62304) increasingly require
  formal evidence, but rewriting 20-year-old C codebases with contracts is
  economically infeasible.  Automated contract extraction from existing tests
  is the only viable path.
- **Open-source infrastructure** (OpenSSL, Linux kernel, Apache projects):
  these codebases have extensive test suites but zero formal contracts.
  Security audits would be dramatically more effective with machine-checked
  contracts identifying exactly what each function guarantees.
- **Platform teams at large tech companies** enforcing API contracts across
  thousands of internal libraries: automatically mined contracts replace
  tribal knowledge with verified assertions.

### Best-Paper Argument

The specification bottleneck is the single largest barrier to formal methods
adoption.  This is the first system that *automatically* bridges the gap between
"has tests" and "has verified contracts" for real-world code.  Prior
specification miners (Daikon, GAssert, EvoSpex, SpecFuzzer) produce *likely*
invariants from dynamic traces with no formal guarantee—they are statistical
guesses that frequently include spurious or overfitted invariants.  Our approach
produces *verified* contracts: every emitted contract is checked against the
source by bounded SMT.  This is not an incremental improvement over Daikon; it
is a category change from "likely" to "verified."

The evaluation story is compelling and fully automated: run the tool on Apache
Commons Math, Guava, and OpenJDK library modules (real code, real tests, real
bugs).  Measure: (a) number of functions that receive verified contracts, (b)
precision of contracts (verified by SMT), (c) latent bugs found (surviving
mutants that violate inferred contracts, confirmed as real defects), (d)
downstream usability (feed contracts into JBMC and measure additional properties
provable).  ICSE/OOPSLA best-paper committees reward tools that *change
developer workflows at scale*, and this does exactly that.

### What Makes This More Than "Mutation Testing + Contract Synthesis"

The distinction is *the specification bootstrapping thesis*: that existing test
suites contain enough implicit specification content to enable formal
verification, and that mutation analysis is the correct extraction mechanism.
Prior work on "tests as specifications" either stops at test amplification
(DSpot, EvoSuite—generating *more tests*, not contracts) or at dynamic invariant
detection (Daikon—guessing invariants from traces).  Neither produces *verified
formal contracts usable by verification tools*.  The mutation–survival boundary
is not a heuristic—it is the *information-theoretic limit* of what the test
suite specifies.  The system doesn't just find contracts; it resolves the
20-year-old open problem of how to get formal methods into legacy codebases
without manual annotation.

---

## FRAMING C — THE SYSTEMS CONTRIBUTION

### Title: *MutaContract: A Scalable Architecture for Compositional Mutation-Guided Specification Synthesis with Incremental Verification*

### Problem & Approach

Building a specification synthesizer that works on real code—not 50-line
benchmarks but functions with loops, aliasing, library calls, and complex
control flow—requires solving five hard systems problems simultaneously, each
of which has defeated prior tools in isolation:

**P1: Mutation-space explosion.**  A function with *n* mutable sites and *k*
mutation operators has O(*nk*) first-order mutants and O(*n²k²*) higher-order
mutants.  A 200-line function easily generates 5,000+ mutants.  A codebase with
10,000 functions produces 50 million mutants.  Naïve enumeration is infeasible.
We develop *subsumption-directed mutation pruning*: using static dominator
analysis to identify mutant equivalence classes, retaining only the *subsuming*
mutants (typically 5–15% of the total).  We further accelerate kill analysis via
*infection-propagation-output* (IPO) decomposition, using weakest-precondition
analysis to determine statically which mutants cannot affect outputs under the
test harness, avoiding execution entirely.

**P2: SyGuS scalability over real program states.**  Standard SyGuS solvers
(CVC5-SyGuS, EUSolver) operate over fixed grammars with small term sizes.
Real function contracts require predicates over heap shapes, array segments,
arithmetic relations among multiple parameters, and disjunctive conditions.  We
design a *mutation-directed grammar generator* that analyzes the syntactic
positions of killed vs. surviving mutants to restrict the SyGuS grammar to
*relevant* predicates.  If mutations at array-index positions are killed but
mutations at arithmetic-result positions survive, the grammar emphasizes
bounds predicates over arithmetic postconditions.  This typically reduces
grammar size by 10–100× compared to a generic predicate grammar, making
synthesis tractable.

**P3: Incremental bounded verification.**  Each candidate contract must be
verified against the source function via bounded SMT.  Naïve re-encoding per
candidate is prohibitive.  We implement *incremental SMT scaffolding*: the
function's SSA encoding and loop unrollings are computed once and cached as a
persistent Z3 context; each candidate contract is asserted as an additional
constraint, checked, and then retracted via push/pop.  For functions with
multiple candidate contracts (precondition variants × postcondition variants),
this amortizes encoding cost across all candidates.

**P4: Compositional contract propagation.**  Real codebases have call chains:
function *f* calls *g* calls *h*.  A contract for *g* should be usable as an
assumption when synthesizing *f*'s contract.  We implement *bottom-up
compositional synthesis*: process the call graph in reverse topological order,
using verified callee contracts as assume-clauses for caller synthesis.  This
makes the system modular—changing one function re-synthesizes only its
transitive callers—and avoids the combinatorial explosion of inlining.

**P5: Language-parametric front end.**  The mutation operators, test harness
integration, and SyGuS grammar templates must be instantiated per source
language.  We define a *mutation-synthesis IR* (MSIR)—a typed intermediate
representation that captures mutable sites, test observation points, and
predicate vocabularies—and implement front ends for Java (via Soot/WALA), C
(via Clang/LLVM), and Python (via ast module).  The core synthesis and
verification pipeline operates entirely on MSIR, making language addition a
front-end-only task (~5K LoC per language).

The resulting system, **MutaContract**, is approximately 160K LoC:
~35K for mutation infrastructure (operators, subsumption, IPO analysis),
~25K for the MSIR and three language front ends,
~30K for the SyGuS synthesizer (grammar generation, CEGIS loop, ranking),
~25K for the incremental bounded verifier (SSA encoding, push/pop, loop
unrolling), ~20K for compositional propagation (call graph, assume-guarantee,
incrementality), and ~25K for infrastructure (test harness integration, caching,
parallelism, serialization, CLI, reporting).

### Who Desperately Needs This

**Verification tool builders** who need reusable, scalable infrastructure for
combining mutation analysis with synthesis and verification.  Today, each tool
(PITest, Major, µJava for mutation; CVC5-SyGuS, Rosette for synthesis; CBMC,
JBMC for verification) exists in isolation with incompatible IRs, incompatible
assumptions, and no composition protocol.  MutaContract's MSIR and compositional
architecture provide the missing integration layer.

**Researchers** building the next generation of specification-inference tools
need a platform that handles the engineering drudgery (mutation, test execution,
SMT encoding) so they can focus on algorithmic innovation.  MutaContract's
modular architecture enables swapping synthesizers, verifiers, and mutation
strategies independently.

**Industrial static-analysis teams** (e.g., Meta Infer, Google ErrorProne,
Microsoft Infer.NET) who need function summaries for interprocedural analysis
but cannot require manual annotation.  Automatically synthesized contracts
serve as verified summaries.

### Best-Paper Argument

The verification-tools community has a *composability crisis*.  Mutation tools,
synthesis tools, and verification tools each work in isolation but cannot be
composed into end-to-end pipelines because they disagree on intermediate
representations, assume different program models, and scale at different rates.
MutaContract solves this with an architecture contribution: the MSIR that
decouples mutation/synthesis/verification, the incremental SMT scaffolding that
makes verification per-candidate cheap, and the subsumption-directed pruning
that makes mutation analysis tractable at scale.

The best-paper case rests on three pillars: (a) **novel architecture** — no
prior system composes mutation, SyGuS, and bounded SMT in a single pipeline
with a shared IR; (b) **scaling results** — demonstrating tractability on
codebases with 50K+ functions where prior tools fail (Daikon runs but produces
unverified invariants; SyGuS solvers alone time out on realistic grammars;
mutation tools alone produce kill matrices but no specifications); (c)
**reusability** — the MSIR and modular design enable other researchers to
build on top, which is the hallmark of a systems paper that earns lasting
impact.

The precedent: KLEE (OSDI '08 best paper) succeeded not because symbolic
execution was new, but because it *engineered* symbolic execution to work on
real code at scale.  MutaContract does the same for mutation-guided contract
synthesis.

### What Makes This More Than "Mutation Testing + Contract Synthesis"

Combining two existing ideas is incremental.  *Engineering them to work together
at scale on real code* is a systems contribution.  The five problems above (P1–
P5) are not "glue code"—each requires a novel technical solution.  Subsumption-
directed pruning (P1) extends prior mutation-reduction work (Ammann et al.,
2014) with static IPO analysis that avoids execution entirely for provably
non-observable mutants.  Mutation-directed grammar generation (P2) is new:
prior SyGuS work uses fixed or user-provided grammars.  Incremental SMT
scaffolding (P3) applies push/pop incrementality in a new context (contract
verification over a persistent function encoding).  Compositional contract
propagation (P4) brings assume-guarantee reasoning into the synthesis loop,
which no prior SyGuS-based tool attempts.  The MSIR (P5) is a new IR
designed for the mutation–synthesis–verification pipeline.  Each of these
alone would be a respectable workshop paper; together they constitute a
major systems contribution.

---

## COMPARATIVE SUMMARY

| Dimension | Framing A (Theory) | Framing B (Practical) | Framing C (Systems) |
|-----------|-------------------|-----------------------|---------------------|
| **Core claim** | Galois connection between mutation kills and contract strength | First system to bootstrap verified contracts from legacy tests | Scalable architecture composing mutation + SyGuS + SMT |
| **Novelty type** | Mathematical theorem + constructive algorithm | New capability (verified contracts from tests at scale) | Engineering breakthroughs (5 novel subsystems) |
| **Target venue** | POPL, PLDI (theory track) | ICSE, OOPSLA, PLDI (tools track) | OOPSLA, PLDI, OSDI (systems track) |
| **Hero result** | Galois connection theorem + gap theorem | N bugs found in Apache Commons / Guava / OpenJDK | 50K-function codebase processed in < 24 hrs on laptop |
| **Risk** | Galois connection must be provably tight (not just suggestive) | Real-world contracts must be non-trivial and bugs must be real | 160K LoC must actually be built and work end-to-end |
| **Reward** | Reshapes understanding of testing ↔ verification | Changes how industry adopts formal methods | Becomes the standard platform for spec-synthesis research |
| **Comparison to Daikon** | Daikon guesses from traces; we derive from mutation structure with guarantees | Daikon produces likely invariants; we produce verified contracts | Daikon is monolithic; we are modular and extensible |
| **150K LoC justification** | Same engine, but paper foregrounds the theory | Same engine, but paper foregrounds the evaluation | The engine IS the contribution |

---

## RECOMMENDATION

**Build one engine.  Write three papers.**  The technical artifact is the same
~160K LoC system in all three framings.  The framings determine which aspects
the paper foregrounds:

- **Framing A** foregrounds the Galois connection proof, with the tool as a
  constructive witness.  Highest ceiling (reshapes a subfield) but highest risk
  (the theorem must be tight and nontrivial).
- **Framing B** foregrounds the evaluation on real codebases, with the theory
  as justification.  Most likely to be accepted (concrete, measurable impact)
  and most likely to be cited (practitioners adopt the tool).
- **Framing C** foregrounds the architecture and scaling, with the evaluation
  as validation.  Best for establishing a *platform* that others build on,
  maximizing long-term influence.

For a best-paper push, **lead with Framing A** (the Galois connection is the
most publishable insight) and **validate with Framing B's evaluation** (real
bugs in real code).  The systems work (Framing C) is necessary regardless but
can be foregrounded in a companion paper or artifact evaluation.
