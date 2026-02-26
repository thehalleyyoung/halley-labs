# Witness-Certified Coalgebraic Compression for Finite-Domain TLA+ Model Checking

## Problem Description

The state explosion problem remains the central barrier to practical model checking of distributed protocols specified in TLA+. TLC, Lamport's explicit-state checker, exhaustively enumerates every reachable state; for Two-Phase Commit with 7 participants or Paxos with 3 nodes, this reaches hundreds of thousands to millions of states, consuming hours of CPU time even on modest configurations. Apalache's symbolic approach trades enumeration for SMT encoding but hits solver limits on the same protocols. Critically, neither tool produces any independently inspectable artifact — a user must trust the tool's implementation, its runtime, and the machine it ran on. For teams sharing verification results across organizational boundaries (e.g., infrastructure teams demonstrating protocol correctness to partner teams or auditors), the inability to independently verify a claimed result without re-running the entire computation is a practical limitation.

The core technical insight is that Lamport's stuttering equivalence — the semantic backbone of TLA+ refinement mappings — is a coalgebraic bisimulation for a specific endofunctor F on the category of fair transition systems. Beohar & Küpper (CALCO 2017) established the coalgebraic treatment of stuttering bisimulation via path-based functors. Our contribution extends this to incorporate TLA+-style weak and strong fairness acceptance conditions into the functor's polynomial structure. The interaction between the stutter-closure monad T (from Beohar-Küpper) and the fairness component Fair(X) requires a non-trivial coherence condition: stuttering-equivalent paths must agree on which fairness acceptance pairs they satisfy. This coherence is the primary new mathematical construction.

Once functor F is in hand, the Barlocco-Kupke-Rot (FoSSaCS 2019) categorical learning framework applies: we instantiate L*-style coalgebraic learning to compute the minimal bisimulation quotient of a TLA+ specification's state space by posing membership and equivalence queries against the concrete transition system. The quotient preserves all stuttering-invariant safety properties (the universal fragment of CTL*\X). We also prove liveness preservation under fairness by establishing that the T-Fair coherence condition ensures acceptance pairs are respected by the quotient — this is a core contribution, not a stretch goal.

CoaCert-TLA is a Rust-based pipeline operating on TLA-lite, a precisely defined finite-domain fragment of TLA+ sufficient for all standard distributed protocol benchmarks. The pipeline has five stages: (1) parse TLA-lite specifications into a typed AST, (2) evaluate specifications via a semantic engine constructing the concrete fair transition system, (3) explore the state space while running an L*-style coalgebraic learner that constructs the minimal bisimulation quotient on-the-fly, (4) emit a Merkle-hashed bisimulation witness binding every equivalence class to its representative state and every transition in the quotient to its witnesses in the original system, and (5) provide a standalone witness verifier that checks the hash chain integrity and validates that the encoded relation satisfies bisimulation closure conditions. The verifier is a small, auditable Rust binary (~3–5K LoC).

The contributions are: (a) the first construction of a coalgebraic endofunctor for TLA+ transition systems incorporating both stuttering closure and fairness acceptance conditions, with a proof of the T-Fair coherence condition enabling liveness preservation, (b) a non-trivial instantiation of L*-style learning to this functor, extending the Barlocco-Kupke-Rot (2019) framework to handle fairness-dependent structure with explicit convergence bounds, (c) a preservation theorem covering both safety (CTL*\X) and liveness under fairness, (d) a lightweight Merkle-hashed bisimulation witness scheme providing hash-based integrity and auditability for the bisimulation relation, and (e) an end-to-end implementation with empirically measured compression ratios on standard protocol benchmarks.

## Value Proposition

**Who needs this.** The formal methods research community working at the intersection of coalgebra, automata learning, and model checking — where the theoretical contribution (stuttering-fair functor + learning instantiation) advances the state of the art. Practitioners using TLA+ for protocol specification who want faster model checking and inspectable verification artifacts. Teams sharing verification results across organizational boundaries who need auditable evidence beyond "we ran TLC and it passed."

**Why valuable.** No existing tool produces a bisimulation witness that a third party can inspect without re-running the entire computation. Bisimulation quotienting can reduce state spaces by measured factors (empirically determined per benchmark, expected range 3–20× based on symmetry reduction literature), making previously slow model checking runs significantly faster. The coalgebraic framework provides the first algorithmic path to *automatically* computing stuttering bisimulation quotients with fairness — prior techniques (symmetry reduction, partial-order reduction) are manually configured and protocol-specific.

**What becomes possible.** A developer model-checks Two-Phase Commit with 7 participants or Paxos-3 on a laptop in minutes rather than hours. The verification produces an inspectable bisimulation witness (~KBs–MBs) that any auditor or collaborator can independently validate using a standalone verifier. The pipeline is fully automatic: no manual symmetry annotations, no protocol-specific tuning.

## Technical Difficulty

### Hard Subproblems

1. **Coalgebraic functor for stuttering-fair TLA+ transition systems.** Constructing F: Set → Set such that F-coalgebra morphisms are exactly the stuttering bisimulations respecting weak and strong fairness constraints. The critical challenge is the T-Fair coherence condition: proving that the stutter-closure monad T (from Beohar-Küpper 2017) distributes correctly over the fairness component Fair(X), ensuring stuttering-equivalent paths agree on acceptance pair satisfaction. No existing functor in the literature handles this combination.

2. **Liveness preservation under fairness.** Proving that F-bisimulation quotients preserve liveness properties under weak and strong fairness, extending Browne-Clarke-Grümberg (1988) to the coalgebraic setting. This requires the T-Fair coherence condition and an additional condition ensuring acceptance pairs are respected by equivalence classes. This is the mathematically deepest contribution.

3. **Adapting L\* to the stuttering-fair functor.** Angluin's L\* assumes deterministic, finite-alphabet, finitely-branching targets. TLA-lite transition systems are nondeterministic with structured actions. The learner handles nondeterministic successors via a powerset decomposition at the functor level. The key challenge: proving the fairness component Fair(X) preserves surjections, which is non-obvious for the dependent structure of acceptance pairs.

4. **Convergence bounds for L\* over F.** Instantiating the Barlocco-Kupke-Rot (2019) framework for functor F. Their framework assumes Set functors preserving surjections. The fairness component introduces structure requiring an extension of their base category. We prove convergence in O(n² · k) membership queries where n is the quotient size and k is the maximum counterexample length.

5. **Equivalence oracle via bounded conformance testing.** True equivalence queries are undecidable. We implement a bounded conformance tester exploring to depth k. This provides a practical (not absolute) guarantee: if the test passes, undiscovered distinguishing sequences must have length > k. We are explicit that this is a bounded-depth heuristic, not a formal proof of equivalence. Correctness of the final quotient is validated by differential testing against TLC.

6. **On-the-fly quotient construction interleaving exploration and learning.** The learner interleaves state-space exploration with quotient refinement, maintaining consistency and closedness invariants during incremental state discovery.

7. **Merkle-hashed bisimulation witness design.** The witness binds equivalence classes to canonical representatives and quotient transitions to original-system witnesses, with a Merkle root enabling O(log n) membership proofs and spot-checking. Witness size is O(|quotient| · log |original|).

### TLA-lite Fragment Definition

TLA-lite is a precisely defined subset of TLA+ sufficient for all standard distributed protocol benchmarks:
- **Types:** Finite-domain integers, booleans, finite sets, finite functions, bounded sequences
- **Operators:** Primed variables, UNCHANGED, ∈, ∪, ⊆, IF/THEN/ELSE, LET/IN, quantification over finite domains
- **Temporal:** □[Action]_vars, WF_vars(Action), SF_vars(Action)
- **Excluded:** CHOOSE, recursive operators, unbounded quantification, infinite sets, function sets, module instantiation with renaming

This fragment captures Paxos, Raft, Two-Phase Commit, PBFT, and all protocols in the TLA+ examples repository.

### Subsystem Breakdown

| Subsystem | LoC Estimate | Language |
|---|---|---|
| TLA-lite Parser & AST | ~5,000 | Rust |
| TLA-lite Semantic Engine | ~8,000 | Rust |
| Explicit-State Explorer | ~5,000 | Rust |
| Coalgebraic Functor Engine | ~8,000 | Rust |
| L\*-Style Learning for TLA-lite Coalgebras | ~8,000 | Rust |
| Bisimulation Witness Emitter (Merkle) | ~4,000 | Rust |
| Standalone Witness Verifier | ~3,000 | Rust |
| Benchmark Suite & Evaluation Harness | ~5,000 | Python/Rust |
| TLA-lite Specification Library | ~3,000 | TLA+ |
| CLI, Integration, Testing Infrastructure | ~6,000 | Rust |
| **Total** | **~55,000** | |
| **Novel algorithmic code** | **~24,000** | Rust |

## New Mathematics Required

All mathematics below is load-bearing: each result directly enables a system component that cannot function without it.

1. **Stuttering-Fair Transition Functor with T-Fair Coherence.** Define F: Set → Set by F(X) = P(AP) × (P(X))^Act × Fair(X), where P(AP) assigns atomic propositions, (P(X))^Act gives nondeterministic action-labeled successors, and Fair(X) encodes fairness as acceptance pairs (B_i, G_i) ⊆ X × X. The stutter-closure monad T from Beohar & Küpper (CALCO 2017) adjoins stutter-equivalent paths. Our primary new construction is the **T-Fair coherence condition**: we prove that T distributes over Fair(X) in the sense that stuttering-equivalent paths satisfy the same acceptance pairs, and that this coherence is necessary and sufficient for F-bisimulations to preserve fairness constraints. This extends Beohar-Küpper's stuttering coalgebra with the fairness dimension native to TLA+.

2. **Preservation Theorem: Safety and Liveness.** We prove F-bisimulation quotients preserve (a) the stuttering-invariant safety fragment CTL*\X (following from the Browne-Clarke-Grümberg 1988 classical result, verified to hold for our functor), and (b) liveness properties under weak and strong fairness, using the T-Fair coherence to show acceptance pairs are preserved by equivalence classes. The liveness result is a core contribution requiring careful treatment of the interaction between fairness quantification and quotient construction.

3. **Coalgebraic Myhill-Nerode for Bounded TLA-lite.** For TLA-lite specifications (finite variable domains, bounded sequences), the F-coalgebra is locally finite. We prove a Myhill-Nerode theorem: the minimal F-coalgebra quotient exists, is unique up to isomorphism, and has size bounded by the number of distinguishable states under stuttering-fair behavioral equivalence. This guarantees L\* termination.

4. **L\* Convergence over F.** We instantiate the Barlocco-Kupke-Rot (FoSSaCS 2019) categorical learning framework for functor F. The adaptation requires proving that Fair(X) preserves surjections despite its acceptance-pair structure — a non-trivial verification that the functor satisfies the framework's preconditions. We prove convergence in O(n² · k) membership queries.

## Best Paper Argument

This work closes the last gap between coalgebraic bisimulation theory and TLA+ semantics by incorporating fairness constraints — the component that distinguishes TLA+ from simpler process algebras where coalgebraic methods already apply. Beohar & Küpper (2017) handled stuttering; we handle stuttering + fairness and prove the T-Fair coherence condition that makes liveness preservation possible.

The theoretical contribution stands independently of the implementation: a CALCO/FoSSaCS paper presenting the functor construction, T-Fair coherence, and Myhill-Nerode theorem would advance the coalgebraic treatment of fair transition systems. The systems contribution stands independently of the theory: a CAV/TACAS paper presenting the tool, bisimulation witnesses, and empirical benchmarks would introduce auditable verification artifacts to a community that currently produces none.

A two-paper strategy targets each contribution at the venue where it is strongest, rather than diluting both in a single sprawling paper. The CALCO/FoSSaCS paper competes on mathematical depth; the CAV/TACAS paper competes on practical impact and artifact quality.

## Evaluation Plan

All evaluation is fully automated with zero human involvement.

**Metrics:**
- **State-space compression ratio:** |quotient states| / |original states|, *empirically measured* per benchmark. We report measured values without pre-committing to a range; preliminary estimates based on symmetry reduction literature suggest 3–20× is realistic.
- **Wall-clock time:** Total pipeline time vs. TLC explicit-state time on identical specifications.
- **Witness size:** Bytes of the emitted Merkle-hashed witness.
- **Witness verification time:** Wall-clock time for the standalone verifier.
- **Correctness validation:** Differential testing against TLC: for every safety property verified on the quotient, TLC independently verifies the same property on the original (up to TLC's tractability bound). Any disagreement is a bug.

**Baselines:**
- TLC explicit-state model checker (standard baseline)
- Apalache symbolic model checker (SMT-based baseline)
- TLC with manual symmetry reduction (best-known TLC optimization)

**Benchmarks:**
- Two-Phase Commit: 3, 5, 7 participants (primary benchmarks — state spaces 10³–10⁶)
- Simple leader election: 3, 5 nodes
- Paxos: 3 nodes (stretch benchmark — state space ~10⁵–10⁶)
- Peterson's mutual exclusion: 2, 3 processes (small but theoretically interesting)

**Automation:** A single `make eval` command runs all benchmarks, collects metrics into structured JSON, generates comparison tables, and performs differential correctness checks.

## Laptop CPU Feasibility

CoaCert-TLA targets TLA-lite specifications with finite-domain state spaces of 10³–10⁶ states. Feasibility analysis:

**Membership query cost.** Each query evaluates a single TLA-lite state and its F-successors. For TLA-lite (finite domains, no CHOOSE, no recursive operators), next-state evaluation is fast: bounded quantification over finite sets, deterministic expression evaluation. Estimated per-query cost: 1–100 μs in optimized Rust.

**Query count.** O(n² · k) where n is quotient size, k is max counterexample length. For Two-Phase Commit with 5 participants (original ~10⁴ states, estimated quotient ~10²–10³ states): n²·k ≈ 10⁶ · 50 = 5 × 10⁷ queries. At 10 μs/query: ~500 seconds ≈ 8 minutes. For Paxos-3 (original ~10⁵, estimated quotient ~10³–10⁴): n²·k ≈ 10⁸ · 100 = 10¹⁰ queries. At 1 μs/query: ~10⁴ seconds ≈ 2.8 hours. This is aggressive but potentially feasible with caching optimizations.

**Memory.** Observation table: O(|quotient|² · |alphabet|). For quotient = 10³, alphabet = 20: 2 × 10⁷ entries × 8 bytes = 160 MB. Well within laptop RAM.

**Target performance.** Full pipeline completes in under 30 minutes for Two-Phase Commit benchmarks, under 4 hours for Paxos-3 (if compression is sufficient). Witness verification completes in under 10 seconds for all benchmarks. These are estimates subject to empirical validation; actual measured times will be reported.

---

**Slug:** `coalg-cert-tlaplus-compress`
