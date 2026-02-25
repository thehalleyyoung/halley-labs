# Spectacles: Systems-Level Difficulty Assessment

*Tyler Sorensen — Systems & Concurrency Reviewer*

---

## Preamble: What This Document Assesses

I evaluate the three proposed approaches (A: Coalgebraic, B: KAT, C: Extraction) on
genuine systems engineering difficulty — what is hard to *build*, not just hard to
*prove*. I focus on architecture integration points, CPU feasibility with concrete
numbers, LoC reality, and catastrophic risk factors.

My overall assessment: the problem is genuinely hard, the CPU feasibility is
surprisingly good, but all three proposals significantly underestimate certain
engineering costs while overestimating the novelty of their categorical/algebraic
framing. The hard problems are in the circuit synthesizer and the Lean formalization,
not in the choice of algebraic framework.

---

## 0. Cross-Cutting: CPU Feasibility Analysis

Before assessing approaches individually, I resolve the question that constrains all
three: *will STARK proof generation for these WFAs run on an M2 in under 30 seconds?*

### Methodology

I model the STARK prover cost as four components: (1) trace generation
(matrix-vector multiply per step), (2) NTT for low-degree extension
(per-column, over the blowup domain), (3) constraint evaluation over the extended
domain, (4) Merkle-tree hashing for FRI commitments. I use the Goldilocks field
(p = 2⁶⁴ − 2³² + 1) with 64-bit native arithmetic at ~2 Gops/s throughput on M2,
BLAKE3 at ~8 GB/s, and a FRI blowup factor of 8.

### Results

| Metric | States | Trace Width | Aux Cols (bit-decomp) | Total Width | Domain | Est. Time | Memory |
|--------|--------|-------------|----------------------|-------------|--------|-----------|--------|
| Exact match | 514 | 1,033 | 0 | 1,033 | 4,096 | **0.07–0.1 s** | 58 MB |
| Token F1 | 200 | 410 | 0 | 410 | 4,096 | **0.03–0.04 s** | 23 MB |
| Regex match | 50 | 110 | 0 | 110 | 4,096 | **< 0.02 s** | 6 MB |
| ROUGE-1 | 100 | 210 | 0 | 210 | 4,096 | **0.02 s** | 12 MB |
| ROUGE-L (tropical) | 513 | 1,036 | ~6,200 | ~7,700 | 4,096 | **0.5–1.1 s** | 240 MB |
| BLEU-1 | 100 | 210 | 0 | 210 | 4,096 | **0.02 s** | 12 MB |
| BLEU-4 (4 proofs) | 4×100 | 4×210 | 0 | 210 each | 4,096 | **0.08 s total** | 12 MB peak |
| **Dense 500 (stress)** | 500 | 1,010 | 0 | 1,010 | 4,096 | **1.5 s** | 57 MB |
| **Dense 1100 (stress)** | 1,100 | 2,210 | 0 | 2,210 | 4,096 | **7.2 s** | 124 MB |
| **Tropical 1000 (stress)** | 1,000 | 2,005 | ~12,000 | ~15,000 | 4,096 | **1.2 s** | 469 MB |

### Key Findings

1. **The problem statement's timing estimates are pessimistic by 10–30× for Tier 1
   metrics.** Exact match at ~2s? It's more like 0.1s. The overestimate probably
   comes from confusing dense matrix costs with the actual sparse WFA structure of
   NLP metrics.

2. **Tropical semiring (ROUGE-L) is the real stress test.** Bit-decomposition for
   `min` operations adds ~6K auxiliary columns for a 513-state WFA (11 bits per
   comparison, 1 comparison per state). This pushes ROUGE-L to ~0.5–1.1s and
   240 MB — still feasible but an order of magnitude more expensive than Tier 1
   metrics.

3. **Memory is the binding constraint, not time.** A tropical WFA with 1,000 states
   needs ~469 MB for the extended trace. On a 16 GB M2, this leaves headroom, but
   combined with Merkle trees and FRI layers, the working set approaches 600–800 MB.
   Fine for a single proof, but parallel proving (multi-core) would need careful
   memory management.

4. **The 30-second single-proof budget is very generous.** No realistic NLP metric
   WFA (with the decomposition strategy described) comes close. The actual bottleneck
   is end-to-end throughput: 50 examples × 7 metrics × ~0.3s average = ~105s ≈ 1.75
   minutes single-threaded. Well within the 60-minute budget.

5. **Alphabet encoding is a hidden design decision the proposals don't discuss.**
   With token vocabularies of 30K+, one-hot encoding is infeasible (30K extra
   columns). The viable approach is to bake reference values into constraint
   polynomials as row-dependent constants. This works but means the AIR is
   *instance-specific* (different circuit per reference), which has implications for
   proof batching and circuit caching that none of the proposals address.

### Verdict

**CPU feasibility is confirmed.** The <30s single-proof target is achievable with
significant margin. The <60 minute end-to-end target is achievable single-threaded.
Memory is adequate on 16 GB. The tropical semiring is the only potential surprise, and
even that stays well within budget with bounded-bit-width comparisons.

---

## 1. Approach A: Coalgebraic WFA Semantics with Functorial Circuit Compilation

### 1.1 Genuine Software Artifact Difficulty

**Architecture.** The coalgebraic framing creates a deep typeclass hierarchy
(`Coalgebra F α` → `WFACoalgebra S Σ α` → `AIRCoalgebra p α`) in Lean 4. The hard
integration points are:

- **Typeclass diamond resolution.** `WFACoalgebra` extends both `Coalgebra` and
  `Semiring`; `AIRCoalgebra` extends `Coalgebra` and `Field`. Lean 4's typeclass
  resolution struggles with deep diamonds involving Mathlib's algebraic hierarchy.
  Expect 2–3 weeks of debugging instance-resolution loops.

- **Functor encoding.** The observation functor F_S(X) = S × X^Σ must be encoded as
  a Lean 4 type. For finite Σ, this is `S × (Σ → X)`, which is fine. But the
  natural transformation to AIR semantics requires relating `S × (Σ → X)` to `𝔽_p ×
  (Σ → 𝔽_p^n)`, which involves a product of the semiring embedding and a power map.
  This is *possible* but not *ergonomic* in Lean 4's type system.

- **Coinductive vs. inductive mismatch.** WFA acceptance is defined coinductively
  (behavioral equivalence) but AIR traces are finite (inductive — fixed-length
  vectors). The compilation correctness proof must bridge this gap. The proposal
  suggests "bounded bisimulation" as mitigation, which is correct but effectively
  abandons the coinductive framing for the critical theorem.

**Performance.** The coalgebraic framing has *zero performance impact*. The compiled
circuits are identical regardless of whether you arrive at them via coalgebra
homomorphisms or direct matrix encoding. Same trace widths, same constraint degrees,
same NTT sizes. The category theory is a proof organization strategy, not a
performance strategy.

**Engineering — novel vs. glue.** The actually novel engineering in Approach A is:

| Component | Novel? | Why/Why Not |
|-----------|--------|-------------|
| `Coalgebra` Lean typeclass | YES | First substantial coalgebra library in Lean 4 |
| Coinductive bisimulation prover | YES | Genuinely new Lean 4 proof engineering |
| WFA-to-AIR trace layout | NO | Same engineering as B and C; coalgebra framing is decoration |
| Syntactic monoid PSI | PARTIALLY | Novel algebraic integration, but no practical benefit over flat PSI |
| STARK prover integration | NO | Winterfell/Plonky3 API calls in all approaches |
| Constraint polynomial generation | NO | Same degree-2 polynomial constraints regardless of framing |

I estimate ~8K LoC of genuinely novel Lean 4 code (coalgebra library + coinductive
proofs) and ~5K LoC of unnecessary complexity (syntactic monoid PSI). The remaining
~100K LoC is shared with approaches B and C.

### 1.2 Hard Subproblems

**1. Coinductive proofs in Lean 4 at scale (Risk: HIGH)**

- *Why hard:* Lean 4's `Cofix` requires productive corecursion — every corecursive
  call must be guarded by a constructor. For bisimulation proofs over WFAs, the
  corecursive structure is: "states are bisimilar if their observation (weight) agrees
  and their successor states (under all inputs) are bisimilar." The guardedness
  checker must see through the matrix arithmetic to verify the constructor guard. For
  WFAs with hundreds of states, this involves type-checking terms with hundreds of
  nested constructor applications.
- *State of the art:* No Lean 4 project has attempted coinductive proofs at this
  scale. Coq's `paco` library handles parameterized coinduction but has no Lean 4
  analogue. The Lean 4 community has limited experience with non-trivial coinductive
  types (most uses are for streams and lazy lists).
- *Risk of failure:* **40%.** If the guardedness checker rejects natural proof terms,
  the fallback is bounded bisimulation (reduce to induction over depth |Q|²), which
  works but collapses the coalgebraic narrative.

**2. Syntactic monoid computation for weighted automata (Risk: MEDIUM)**

- *Why hard:* The syntactic monoid of a WFA over a non-Boolean semiring is defined
  via matrix rank conditions (Berstel-Reutenauer). Computing it requires finding the
  minimal quotient of the transition monoid where rank-equivalent matrices are
  identified. For 500-state WFAs, the transition monoid can have exponentially many
  elements.
- *State of the art:* Efficient syntactic monoid computation exists for DFAs
  (semigroup-based, O(|Q|²|Σ|)). For weighted automata over arbitrary semirings,
  there are no practical implementations.
- *Risk of failure:* **30%.** If syntactic monoid computation is too expensive, the
  PSI integration degrades to flat n-gram PSI (which works fine but breaks the
  algebraic-integration story).

**3. Natural transformation as compilation correctness (Risk: LOW-MEDIUM)**

- *Why hard:* The "natural transformation between functors" must commute with
  observation maps. For Tier 1 semirings, this follows from the semiring
  homomorphism. For Tier 2 (tropical), it doesn't — the "natural transformation"
  degrades to a simulation relation, and the functorial story is incomplete.
- *State of the art:* No prior work on coalgebra-to-circuit natural transformations.
- *Risk of failure:* **20%.** The Tier 1 proof will work. The Tier 2 gap is a
  narrative weakness, not a technical failure.

**4. Lean 4 typeclass hierarchy engineering (Risk: MEDIUM)**

- *Why hard:* Mathlib's algebraic hierarchy is already deep
  (`Semiring` → `CommSemiring` → `Ring` → ...). Adding `Coalgebra`,
  `WFACoalgebra`, and `AIRCoalgebra` on top creates resolution paths that may time
  out. Gabriel Ebner has documented issues with Lean 4 typeclass resolution for
  deeply nested hierarchies.
- *State of the art:* Mathlib manages deep hierarchies via careful use of `instance`
  priorities and `@[reducible]`. But no Mathlib hierarchy combines algebraic and
  coalgebraic structure.
- *Risk of failure:* **25%.** Solvable with engineering effort but could consume 2–4
  weeks.

**5. Tropical semiring bit-decomposition correctness proof (Risk: MEDIUM)**

- *Why hard:* The Lean 4 proof of `circuit_sound_tropical` must show that
  bit-decomposition comparison gadgets correctly simulate tropical `min`. This
  requires proving: (a) each decomposed bit is 0 or 1, (b) the reconstruction equals
  the original value, (c) the comparison result is correct, (d) the value is in the
  bounded range. Lean 4's `omega` tactic handles arithmetic, but the bit-level
  reasoning requires `Finset.range` and `Nat.testBit` lemmas that are sparse in
  Mathlib.
- *State of the art:* Bit-decomposition proofs exist in Coq (Fiat-Crypto project)
  but not in Lean 4 at this level of detail.
- *Risk of failure:* **25%.** Achievable but time-consuming (estimate 3–5 weeks for
  the tropical soundness proof alone).

### 1.3 LoC Reality Check

The problem statement claims 117K–142K total. For Approach A specifically:

| Component | Claimed LoC | Approach A Reality | Notes |
|-----------|-------------|-------------------|-------|
| EvalSpec DSL | 15–18K | 15K (same) | Approach-independent |
| WFA Engine | 18–22K | 20K (same) | Approach-independent |
| Circuit Synthesizer | 24–28K | 28–35K | **Underestimated.** Novel engineering, no prior art, expect iteration |
| Protocol Engine | 8–10K | 8K (same) | Approach-independent |
| TLA+ Spec | 6–8K | 5K | **Overestimated.** TLA+ specs are terse; 2K TLA+ + 3K harness |
| PSI Module | 10–12K | 15–18K | **Underestimated** for Approach A. Syntactic monoid PSI adds 5K+ |
| Lean 4 Library | 12–15K | 20–28K | **Severely underestimated.** Coalgebra typeclass, coinductive proofs, WFA library, tactics. Lean proof engineering always takes 1.5–2× estimates |
| Scoring Library | 10–12K | 12K (same) | Approach-independent |
| Test Infrastructure | 14–17K | 18–22K | **Underestimated.** Cross-representation differential testing across 9 subsystems |
| **Total** | **117–142K** | **141–173K** | **20–30K more than claimed** |

**What blows up:** The Lean 4 component. Always. Proof engineering estimates should
be multiplied by 1.5–2× for anything involving new typeclass hierarchies or
coinductive reasoning. For Approach A, the Lean 4 component alone could reach 25–30K
LoC, consuming 30–40% of the entire project's proof effort on the coalgebraic framing
that doesn't improve the circuit synthesizer.

**Novel LoC claim (94–114K):** Inflated. Much "novel" code is standard
infrastructure (error handling, serialization, CLI, argument parsing, logging). I
estimate 55–70K of genuinely novel algorithmic/verification code for Approach A, plus
~15K of novel-but-unnecessary complexity (syntactic monoid PSI, coalgebra hierarchy).

### 1.4 CPU Feasibility

See Section 0. The coalgebraic framing has no CPU impact. The circuits are identical.
The Lean 4 build is the only approach-specific concern: with the expanded typeclass
hierarchy, expect Lean build times of 20–40 minutes (incremental, with Mathlib
cached). The `native_decide` calls for bounded bisimulation are bounded to ≤20 states
per the problem statement, which is fine — but this means the coinductive story is
never actually tested on realistic WFAs in the proof assistant.

### 1.5 Risk Assessment

| Risk | Probability | Impact | P × I | Notes |
|------|------------|--------|-------|-------|
| Coinductive proofs hit Lean 4 kernel limits | 40% | 8/10 | **3.2** | Kills the coalgebraic narrative entirely |
| Typeclass resolution timeouts | 25% | 5/10 | 1.25 | Solvable with effort but delays |
| Syntactic monoid PSI infeasible at scale | 30% | 4/10 | 1.2 | Fallback to flat PSI; narrative loss only |
| Tropical soundness proof takes >6 weeks | 50% | 6/10 | **3.0** | Shared with all approaches |
| Circuit synthesizer requires 2× estimated LoC | 40% | 5/10 | 2.0 | Shared with all approaches |
| Lean 4 Mathlib breaking changes during dev | 30% | 4/10 | 1.2 | Shared with all approaches |

**Most likely to kill Approach A:** The coinductive proof obligation. If Lean 4's
`Cofix` doesn't cooperate with the WFA bisimulation proofs, the team either (a)
abandons the coalgebraic framing (collapsing to a simpler approach) or (b) uses
`sorry` on the key theorem, which defeats the purpose of the coalgebraic story.

### 1.6 Revised Difficulty Score

| Dimension | Score | Justification |
|-----------|-------|---------------|
| Systems Difficulty | 8/10 | Coalgebraic Lean 4 proofs are genuinely hard; but the circuit engineering is approach-independent |
| Architecture Novelty | 6/10 | Coalgebra framing is intellectually appealing but doesn't change the circuit architecture. The "natural transformation" is a way of *talking about* the compiler, not a way of *building* it |
| Performance | 4/10 | No performance engineering from the coalgebraic framing. Same circuits as B and C |
| Best-Paper Potential | 5/10 | "First coalgebraic verified compiler" is novel for LICS but CAV reviewers will ask "what does the category theory buy?" The honest answer — a universal proof template for Tier 1 semirings only — is underwhelming |

**Overall: 6/10.** The coalgebraic framing adds proof complexity without adding
system capability. It's a research contribution to Lean 4/Mathlib's coalgebra
library, wrapped in a systems project. The *hard systems engineering* (circuit
synthesizer, trace layout, tropical gadgets) is approach-independent.

---

## 2. Approach B: Algebraic Program Analysis via Kleene Algebra with Tests

### 2.1 Genuine Software Artifact Difficulty

**Architecture.** The KAT framing embeds both WFAs and circuits into a single
algebraic structure. The hard integration points are:

- **Weighted KAT extension.** Standard KAT completeness (Kozen & Smith 1996)
  crucially depends on the Boolean structure of tests. The proposal's Theorem 3
  claims completeness for *weighted* KAT (WKAT), which is an open mathematical
  question. The fallback — Boolean-tested WKAT with weighted transitions — is
  tractable but reduces KAT to a notational convenience over standard WFA theory.

- **KAT decision procedure implementation.** The Antimirov partial-derivative
  construction followed by on-the-fly DFA equivalence checking. Worst-case
  exponential, but for the structured WFAs arising from NLP metrics (mostly
  deterministic, sparse transitions), it should terminate quickly. The engineering
  challenge is early-termination heuristics and memory management for the state-space
  exploration.

- **Matrix Kleene star in 𝔽_p.** The star A* = Σ_{k≥0} A^k of a transition matrix
  over a finite field. For nilpotent matrices, this sum is finite (A^n = 0 for some
  n ≤ |Q|). For non-nilpotent matrices (common in WFAs with self-loops), the star
  doesn't converge in 𝔽_p. The proposal doesn't address this. In practice, the WFA
  trace is bounded by input length, so you never need A* — you need A^ℓ for a
  specific ℓ. But this means the Kleene star is a *specification tool*, not a
  *computation tool*, which somewhat undermines the KAT framing.

**Performance.** Same as Approach A — the KAT framing has no impact on circuit
structure or STARK performance. The only performance-relevant component is the KAT
decision procedure, which for specification equivalence (not proof generation) needs
to run in seconds on laptop. For WFAs with ≤100 states (practical NLP metrics), the
Antimirov construction produces automata with ≤10K states, and on-the-fly equivalence
checking is O(n log n) via Hopcroft. This is fast.

For WFAs with 500+ states (exact match, ROUGE-L), the decision procedure could
produce automata with 250K+ states. Hopcroft on 250K states: O(n log n) ≈ 4.5M
operations, ~2ms. Still fast. The exponential blowup from Antimirov is the risk, but
for deterministic WFAs (which NLP metrics mostly are), there's no blowup.

**Engineering — novel vs. glue.**

| Component | Novel? | Why/Why Not |
|-----------|--------|-------------|
| `KATAlgebra` Lean typeclass | YES | New Lean 4 KAT formalization |
| WKAT decidability proof | YES | New formalization (possibly new mathematics for full WKAT) |
| `kleene_dec` tactic | YES | Certified decision procedure via `native_decide` |
| WFA-to-AIR compilation | NO | Same engineering as A and C |
| Symbolic PSI via KAT | PARTIALLY | Algebraic framing of n-gram membership; marginal practical value |
| KAT-to-WFA translation | NO | Standard textbook construction (Antimirov) |

I estimate ~6K LoC of genuinely novel Lean 4 code (KAT typeclass + WKAT decidability
+ `kleene_dec`) and ~3K of marginal-value code (symbolic PSI integration). The bulk
of the system is shared infrastructure.

### 2.2 Hard Subproblems

**1. Weighted KAT completeness theorem (Risk: HIGH for full WKAT, LOW for restriction)**

- *Why hard:* Full WKAT completeness — proving that WKAT equivalence captures exactly
  scoring-function equivalence — requires extending Kozen's completeness theorem from
  Boolean tests to semiring-weighted tests. This is open mathematics, not just open
  formalization. The Boolean algebra of tests is used in ~15 places in the standard
  completeness proof (complementation, meet, join, distribution). Replacing Boolean
  with semiring breaks all 15.
- *State of the art:* Kozen & Mamouras (2014) extended KAT to *NetKAT* (network
  programming), but NetKAT retains Boolean tests. No published completeness result
  for WKAT with semiring-valued tests.
- *Risk of failure:* **60% for full WKAT.** The restriction to Boolean-tested WKAT
  (Approach B's "Approach (a)") has **10% failure risk** — it's a notational
  reorganization of standard WFA theory, not a new theorem.
- *Consequence of failure:* If full WKAT completeness fails, the KAT-specific
  contribution reduces to "a nicer way to talk about WFA equivalence." The system
  still works; the theorem narrative weakens.

**2. `kleene_dec` tactic engineering (Risk: LOW-MEDIUM)**

- *Why hard:* The tactic must: (a) parse a KAT goal into an Antimirov-style
  expression, (b) construct the derivative automaton, (c) check equivalence, (d)
  produce a Lean proof term witnessing equivalence or a counterexample. Step (d) is
  the hard part — the proof term must be type-checkable by the Lean kernel, which
  means either `native_decide` (fast but limited to small instances) or a reflected
  proof (slower but unbounded).
- *State of the art:* `omega` in Lean 4 follows this pattern for linear arithmetic.
  `norm_num` does it for numeric normalization. No KAT tactic exists in any proof
  assistant.
- *Risk of failure:* **15%.** The pattern is well-established; the engineering is
  substantial but tractable. Estimate 4–6 weeks for a basic version, 8–12 weeks for
  a polished version.

**3. Matrix Kleene star divergence in 𝔽_p (Risk: MEDIUM)**

- *Why hard:* If the WFA transition matrix has eigenvalue 1 (i.e., there's a cycle
  with weight 1), then A* = Σ A^k diverges over ℤ. Over 𝔽_p, it wraps around modulo
  p, giving wrong results. The problem statement's two-tier structure doesn't address
  this: even for Tier 1 (counting semiring), the Kleene star of a matrix with
  self-loops diverges.
- *Why it might be OK:* The WFA accepts a specific finite input w of length ℓ. The
  acceptance weight is computed via the trace v_0 · M_{w_1} · M_{w_2} · ... ·
  M_{w_ℓ} · ρ, which is a finite product — no Kleene star needed. The KAT
  completeness is for the *specification equivalence* decision procedure, not for
  circuit computation. So the Kleene star is used in the *tactic*, not in the
  *circuit*.
- *Risk of failure:* **20%.** The tactic can bound the star iteration by input length
  and use A^{≤ℓ} instead of A*. But this must be proved to be equivalent for the
  class of WFAs arising from EvalSpec, which requires a lemma about the acyclicity or
  nilpotency of the relevant transition matrices.

**4. Tropical semiring bit-decomposition (shared with A and C) (Risk: MEDIUM)**

- Same analysis as Approach A §1.2.5.

**5. Circuit synthesizer for novel AIR encoding (shared with A and C) (Risk: MEDIUM)**

- *Why hard:* The input symbol encoding in AIR is the deepest engineering challenge
  that *none of the three proposals discuss.* For a token vocabulary of 30K+, you
  cannot use one-hot encoding (30K extra columns). The viable approach — baking
  reference constants into constraint polynomials — makes the AIR instance-specific.
  This means: different reference strings → different constraint polynomials →
  different AIR programs. This is fine for single proofs but kills circuit caching
  and batched verification.
- *Risk of failure:* **15%.** The instance-specific approach works; the batching
  limitation is a feature trade-off, not a failure.

### 2.3 LoC Reality Check

| Component | Claimed LoC | Approach B Reality | Notes |
|-----------|-------------|-------------------|-------|
| EvalSpec DSL | 15–18K | 15K (same) | Approach-independent |
| WFA Engine | 18–22K | 18–20K | Slightly simpler: KAT provides cleaner internal API |
| Circuit Synthesizer | 24–28K | 28–33K | **Underestimated.** Same novel engineering as A |
| Protocol Engine | 8–10K | 8K (same) | Approach-independent |
| TLA+ Spec | 6–8K | 5K | Same overestimate as A |
| PSI Module | 10–12K | 10–12K | Standard flat PSI; symbolic PSI adds ~2K marginal code |
| Lean 4 Library | 12–15K | 15–20K | KAT typeclass + WKAT decidability + tactics. Less risky than A's coalgebra hierarchy but still underestimated |
| Scoring Library | 10–12K | 12K (same) | Approach-independent |
| Test Infrastructure | 14–17K | 17–20K | Slightly less than A (no syntactic monoid tests) |
| **Total** | **117–142K** | **128–155K** | **10–15K more than claimed** |

**What blows up:** The circuit synthesizer (approach-independent) and the Lean 4
KAT formalization. But the blowup is smaller than Approach A because KAT has more
mature theory and the typeclass hierarchy is shallower.

### 2.4 CPU Feasibility

Same as Section 0 for proof generation. The KAT decision procedure for specification
equivalence is the only approach-specific CPU concern:

- Antimirov construction for a 100-state WFA: produces ≤10K derivative states.
  On-the-fly equivalence checking via BFS with union-find: O(n α(n)) ≈ 10K
  operations. **< 1ms.**
- For a 500-state WFA: up to 250K derivative states (if nondeterministic). Hopcroft
  equivalence: O(n log n) ≈ 4.5M operations. **< 5ms.**
- Exponential blowup from nondeterminism: possible but unlikely for NLP-metric WFAs,
  which are mostly deterministic. **Risk: 10%** that some metric produces a WFA where
  the decision procedure takes >10 seconds. Mitigation: early-termination with
  timeout.

### 2.5 Risk Assessment

| Risk | Probability | Impact | P × I | Notes |
|------|------------|--------|-------|-------|
| Full WKAT completeness fails (open math) | 60% | 5/10 | **3.0** | Falls back to Boolean-tested WKAT; narrative weakens |
| Boolean-tested WKAT is "just WFA theory" | 80% | 3/10 | 2.4 | Reviewers see through the framing |
| Circuit synthesizer requires 2× LoC | 40% | 5/10 | 2.0 | Shared with all approaches |
| Tropical soundness proof >6 weeks | 50% | 6/10 | **3.0** | Shared with all approaches |
| `kleene_dec` tactic takes >12 weeks | 20% | 4/10 | 0.8 | Could ship with `sorry` on tactic soundness |
| Matrix Kleene star subtlety | 20% | 4/10 | 0.8 | Solvable via bounded iteration |

**Most likely to kill Approach B:** Not a single catastrophic risk but a *narrative
death by a thousand cuts.* If full WKAT completeness fails (likely), the KAT framing
becomes "a nice notation for standard WFA equivalence." If Boolean-tested WKAT is the
fallback (likely), reviewers will ask "what does KAT add beyond Hopcroft
minimization?" The honest answer is "cleaner proofs and a tactic," which is solid
Lean/Mathlib work but not a systems contribution.

### 2.6 Revised Difficulty Score

| Dimension | Score | Justification |
|-----------|-------|---------------|
| Systems Difficulty | 7/10 | KAT is well-established; the hard systems work (circuit synthesizer, tropical gadgets) is approach-independent |
| Architecture Novelty | 5/10 | KAT-ZK connection is new but KAT doesn't change the circuit architecture. The "evaluation KAT" is a naming convention, not a design decision |
| Performance | 5/10 | `kleene_dec` is a real performance-engineered component (Antimirov + on-the-fly equivalence). No circuit-level performance contribution |
| Best-Paper Potential | 6/10 | Good CAV fit; KAT is well-loved in the community. But "we used KAT as a notation for WFA equivalence" is incremental. The ZK application saves it |

**Overall: 6/10.** The most grounded of the three approaches. Lower risk than A,
higher feasibility than C. But the KAT framing is a notational convenience, not a
conceptual breakthrough. The real systems contribution — the WFA-to-STARK circuit
synthesizer — is approach-independent.

---

## 3. Approach C: Stratified Verification with Extraction

### 3.1 Genuine Software Artifact Difficulty

**Architecture.** This is by far the most ambitious approach architecturally. Writing
the entire WFA engine and circuit synthesizer in Lean 4 and extracting to Rust is a
fundamentally different engineering paradigm.

The hard integration points are:

- **Lean-to-Rust extraction pipeline.** This does not exist. The proposal
  acknowledges this and proposes three mitigations: (a) `@[extern]` for hot paths,
  (b) `Finsupp` for sparse matrices, (c) differential testing as fallback. All three
  are concessions that the extraction story is incomplete. In practice, Approach C
  will use `@[extern]` so extensively that it becomes "write Rust with Lean
  contracts" — which is what Approaches A and B do, but with more ceremony.

- **Lean 4 as a systems programming language.** The proposal wants 40K+ LoC of
  algorithmic Lean 4 code. Lean 4's ecosystem for systems programming (string
  processing, hash maps, bit manipulation, SIMD) is immature. `HashMap` exists but
  is not `O(1)` amortized (it's a persistent data structure). `ByteArray` exists but
  has no SIMD-accelerated operations. The Goldilocks field needs 64-bit modular
  arithmetic with Montgomery multiplication — Lean 4's `UInt64` doesn't expose the
  128-bit intermediate needed for modular reduction.

- **The extraction gap.** The proposal's Theorem 1 (Extraction Preservation) is
  explicitly NOT proved. It's validated by differential testing on 100K random inputs.
  This means the TCB includes the extraction pipeline (whatever form it takes) and
  the differential tests. This is honest but undermines the "compiler IS the proof"
  narrative.

**Performance.** This is where Approach C is weakest. Concrete concerns:

- **Lean 4 compiled code speed.** Lean 4 compiles to C via `leanc`. For numeric
  code, it produces boxed integers (heap-allocated `Nat` or `Int` objects) unless
  using `UInt64` directly. For WFA trace generation (inner loop: field multiply +
  accumulate over sparse matrix), Lean 4 compiled code is estimated 10–50×
  slower than hand-written Rust with SIMD-friendly data layout.

- **Impact on proof generation time.** If trace generation takes 10× longer, ROUGE-L
  goes from ~1s to ~10s. Still within the 30-second budget, but the end-to-end
  50-example-×-7-metric pipeline goes from ~2 minutes to ~20 minutes. Feasible but
  unimpressive.

- **The `@[extern]` escape hatch.** If performance-critical functions are implemented
  in Rust via `@[extern]`, then the extracted code isn't what runs — the hand-written
  Rust is. At that point, you need the same verification strategy as Approaches A and
  B (Lean spec + Rust implementation + differential testing or cross-verification).
  The extraction approach collapses into the standard approach with extra steps.

**Engineering — novel vs. glue.**

| Component | Novel? | Why/Why Not |
|-----------|--------|-------------|
| WFA engine in Lean 4 | YES | First substantial WFA implementation in Lean 4 |
| Circuit synthesizer in Lean 4 | YES | First circuit synthesizer in Lean 4 |
| Lean-to-Rust extraction/validation | YES | New extraction pipeline or validation framework |
| `@[extern]` FFI for Goldilocks | NO | Standard Lean 4 FFI pattern |
| STARK prover integration | NO | Winterfell API calls (same as A and B) |
| Differential test harness | PARTIALLY | Standard testing methodology, applied to novel extraction |

~15K LoC of genuinely novel Lean 4 algorithmic code (WFA engine + circuit
synthesizer in Lean). ~5K LoC of novel extraction/validation infrastructure. The
remaining ~95K LoC includes the Rust STARK integration, DSL, PSI, tests.

### 3.2 Hard Subproblems

**1. Lean 4 algorithmic code at scale (Risk: HIGH)**

- *Why hard:* Writing 40K+ LoC of *algorithmic* code (not proofs) in Lean 4 is
  unprecedented. The largest algorithmic Lean 4 codebases are ~10K LoC (Lean 4
  compiler itself is an exception at ~100K but is bootstrapped). Lean 4's
  `HashMap`, `Array`, and `IO` are functional-first, which means: (a) no in-place
  mutation without `IO.Ref` or `ST`, (b) array indexing returns `Option α` requiring
  pattern matching on every access, (c) nested loops require tail-recursive
  functions with explicit accumulators. This is doable but 3–5× slower to write than
  equivalent Rust.
- *State of the art:* Lean 4 is designed for this (it's a "programming language AND
  proof assistant"). But the community's focus has been on the proof assistant side.
  Large-scale algorithmic Lean 4 code is uncharted.
- *Risk of failure:* **35%.** Not failure to write the code, but failure to write it
  in time. The 3–5× development-speed penalty for Lean 4 vs. Rust means 40K LoC
  takes as long as 120–200K LoC in Rust.

**2. Performance of Lean 4-compiled code (Risk: HIGH)**

- *Why hard:* Lean 4's C backend produces code with: (a) reference-counted objects
  (RC increment/decrement on every assignment), (b) boxed scalars for `Nat`/`Int`
  (heap allocation per arithmetic operation), (c) no SIMD vectorization (the C code
  is scalar). For the STARK trace generation inner loop (multiply-accumulate over
  Goldilocks field), this means: 1 heap allocation per field multiply + 2 RC
  operations per assignment = ~10ns per field operation vs. ~0.5ns for hand-written
  Rust. That's a 20× slowdown.
- *State of the art:* Lean 4's `UInt64` avoids boxing, but Goldilocks modular
  reduction needs 128-bit intermediates. `UInt64.mul` in Lean 4 wraps at 2⁶⁴, which
  is wrong for Goldilocks (need (a × b) mod p, which requires 128-bit
  intermediate). The `@[extern]` escape is the only option.
- *Risk of failure:* **50%** for achieving <5× slowdown vs. hand-written Rust without
  `@[extern]`. With `@[extern]` on the hot path, the risk drops to **10%** but the
  extraction story collapses.

**3. Lean-to-Rust extraction fidelity (Risk: VERY HIGH)**

- *Why hard:* No production-quality Lean 4 → Rust extraction exists. Options: (a)
  Build one (multi-year project, far out of scope). (b) Use `lean4export` → C → Rust
  transpilation (fragile, untested). (c) Manually translate Lean 4 → Rust and
  validate (defeats the purpose). (d) Run the Lean 4 binary directly via `leanc`
  (performance penalty).
- *State of the art:* CakeML (HOL4 → ML → x86) took 10+ years and a team of 5–10
  researchers. Coq extraction to OCaml is mature but not to Rust. Lean 4 `lean4export`
  targets C with Lean's runtime, not standalone Rust.
- *Risk of failure:* **70%** for building a new extraction pipeline. **20%** for the
  fallback (use `leanc`-compiled binary + `@[extern]` for hot paths).

**4. Integration with Winterfell STARK library (Risk: MEDIUM)**

- *Why hard:* Winterfell expects Rust `trait` implementations for `Air`,
  `TraceTable`, and `ProofOptions`. If the trace generator is in Lean 4 (compiled to
  C), calling Winterfell requires: (a) Lean 4 → C → Rust FFI chain, or (b) implement
  the Winterfell traits in Rust that call into the Lean 4 compiled code. Both are
  fragile and have ABI-stability risks.
- *Risk of failure:* **30%.** Solvable with effort, but the FFI boundary is a
  maintenance nightmare and a source of subtle bugs.

**5. Verified WFA minimization performance (Risk: MEDIUM)**

- *Why hard:* Hopcroft minimization requires partition refinement with a priority
  queue. In Lean 4, a functional priority queue (leftist heap or similar) has O(log n)
  operations but with high constant factors (heap allocation per insert). For 500-state
  WFAs, this is fine (~500 operations). For 5,000-state WFAs (hypothetical future
  metrics), the constant factors might matter.
- *Risk of failure:* **10%.** Not a realistic concern for the current 7 metrics.

### 3.3 LoC Reality Check

| Component | Claimed LoC | Approach C Reality | Notes |
|-----------|-------------|-------------------|-------|
| EvalSpec DSL | 15–18K | 10–12K | **Some saved**: if DSL is also in Lean 4, share types |
| WFA Engine | 18–22K | 25–35K | **Blows up.** Lean 4 algorithmic code is 1.5–2× the Rust equivalent for the same functionality, plus proof obligations on every function |
| Circuit Synthesizer | 24–28K | 30–40K | **Blows up.** Same 1.5–2× factor, plus `@[extern]` wrappers for hot paths |
| Protocol Engine | 8–10K | 8K in Rust | Not extracted; Lean-specified interface |
| TLA+ Spec | 6–8K | 5K | Same as A and B |
| PSI Module | 10–12K | 10–12K in Rust | Not extracted; Lean-specified interface |
| Lean 4 Library | 12–15K | **Merged into WFA + Circuit** | Specifications are co-located with implementation |
| Scoring Library | 10–12K | 15–20K | Triple implementation PLUS Lean implementations of all 7 metrics |
| Test Infrastructure | 14–17K | 20–25K | **Blows up.** Extraction validation tests, FFI boundary tests, Lean vs. Rust differential tests |
| Extraction/FFI Glue | 0 | 5–10K | **New component.** `@[extern]` declarations, C-Rust bridges, build system integration |
| **Total** | **117–142K** | **128–177K** | **Could exceed 175K** |

**What blows up:** Everything that's in Lean 4 instead of Rust takes 1.5–2× more LoC
for the same functionality. The extraction/FFI glue is an entirely new component
(5–10K LoC) not in the original estimate. The test infrastructure explodes because
you're testing three representations (Lean spec, Lean compiled, Rust) instead of two.

### 3.4 CPU Feasibility

This is Approach C's Achilles' heel.

**With Lean 4 compiled code (no `@[extern]`):**

| Metric | Approach A/B Time | Approach C Time (est. 15× overhead) | Feasible? |
|--------|-------------------|-------------------------------------|-----------|
| Exact match | 0.1 s | 1.5 s | ✓ |
| Token F1 | 0.04 s | 0.6 s | ✓ |
| ROUGE-L | 1.0 s | 15.0 s | ⚠️ Tight |
| BLEU-4 | 0.08 s | 1.2 s | ✓ |
| 50 examples × 7 metrics | 1.75 min | 26.3 min | ✓ but slow |

**With `@[extern]` on Goldilocks field ops (the realistic path):**

If field arithmetic is in Rust via `@[extern]`, the overhead drops to ~3–5× (the
remaining overhead is from Lean's RC runtime managing the matrix structures). Then
ROUGE-L goes from 1.0s to 3–5s, and end-to-end is ~5–9 minutes. Feasible.

**Verdict:** CPU-feasible only with `@[extern]` on the hot path, which means the
"compiler IS the proof" narrative has a significant asterisk.

### 3.5 Risk Assessment

| Risk | Probability | Impact | P × I | Notes |
|------|------------|--------|-------|-------|
| Extraction pipeline doesn't materialize | 70% | 7/10 | **4.9** | Fallback to `leanc` + `@[extern]`; narrative damage |
| Performance too slow without `@[extern]` | 50% | 6/10 | **3.0** | Mitigated by `@[extern]`; but undermines extraction story |
| 40K+ LoC Lean 4 takes 2× expected time | 60% | 7/10 | **4.2** | Development-speed penalty is real |
| Lean 4 ↔ Winterfell FFI fragility | 30% | 5/10 | 1.5 | ABI stability issues |
| Tropical soundness proof >6 weeks | 50% | 6/10 | **3.0** | Shared with all approaches |
| Lean 4 `leanc` produces incorrect code | 5% | 10/10 | 0.5 | Low probability but catastrophic |

**Most likely to kill Approach C:** The combined effect of (1) no extraction pipeline
and (2) Lean 4 development speed. Together, these mean the team spends 2× the time
writing code in Lean 4 that is slower than Rust, doesn't extract to Rust, and uses
`@[extern]` for all performance-critical paths — arriving at the same architecture as
Approaches A/B but with more ceremony and a bigger TCB.

### 3.6 Revised Difficulty Score

| Dimension | Score | Justification |
|-----------|-------|---------------|
| Systems Difficulty | 9/10 | Writing a full WFA engine + circuit synthesizer in Lean 4 is genuinely hard. The extraction challenge is unsolved |
| Architecture Novelty | 7/10 | "Compiler IS the proof" is a genuinely different architecture. The stratified trust model is methodologically interesting |
| Performance | 2/10 | Worst performance of all three approaches. The `@[extern]` escape concedes the performance argument |
| Best-Paper Potential | 4/10 | "CakeML for ZK" is a great title for a paper that delivers verified extraction. Without verified extraction (likely), it's "we wrote slow code in Lean 4 and called fast code via FFI" — not a compelling narrative |

**Overall: 5.5/10.** The most ambitious approach, but ambition is not difficulty in
the productive sense. The hard problems (extraction, performance) are problems that
*should not be solved in this project* — they're multi-year research programs
(CakeML's extraction pipeline took a decade). Attempting them here risks the entire
project.

---

## 4. Comparative Summary

### 4.1 Systems Difficulty Matrix

| Dimension | A: Coalgebraic | B: KAT | C: Extraction |
|-----------|---------------|--------|---------------|
| Systems Difficulty | 8 | 7 | 9 |
| Architecture Novelty | 6 | 5 | 7 |
| Performance Engineering | 4 | 5 | 2 |
| Best-Paper Potential | 5 | 6 | 4 |
| **Revised Overall** | **6** | **6** | **5.5** |

### 4.2 The Elephant in the Room

**All three proposals overvalue their algebraic framing and undervalue the
approach-independent engineering.** The hardest systems work — the WFA-to-STARK
circuit synthesizer, the tropical bit-decomposition gadgets, the input encoding in
AIR, the Winterfell integration, the differential testing infrastructure — is
*identical across all three approaches.* This shared core represents ~75% of the
total engineering effort. The approach-specific framing (coalgebra vs. KAT vs.
extraction) accounts for ~15% of the effort but ~80% of the proposal narrative.

The choice between approaches is really a choice about how to organize the *Lean 4
proof component* (~15–25K LoC out of ~130–170K total). The Rust code, the circuit
synthesizer, the STARK integration, the PSI protocol, and the test infrastructure
are approach-independent.

### 4.3 What Would Actually Make This Hard

The proposals dress up the difficulty in categorical/algebraic clothing. Here's what's
*actually* hard from a systems perspective:

1. **The circuit synthesizer has no prior art.** No one has compiled weighted
   automata to STARK AIR constraints. The encoding decisions (trace layout, symbol
   representation, constraint structure) require significant design iteration. This
   is ~25–35K LoC of genuinely novel Rust code.

2. **The tropical semiring breaks the clean algebraic story.** `min` has no field
   homomorphism. Every approach must handle this as a special case. The
   bit-decomposition gadgets require careful range analysis and Lean 4 proofs that
   are not covered by the generic theorems.

3. **The input encoding problem.** With token vocabularies of 30K+, encoding the
   input symbol in each AIR row requires baking reference constants into constraint
   polynomials. This makes every proof instance-specific, with implications for
   caching, batching, and universal circuit design that none of the proposals
   address.

4. **Triple implementation with differential testing.** Maintaining three
   implementations of seven metrics (reference Python, WFA Rust, circuit Rust)
   with cross-representation agreement is a configuration-management nightmare.
   Any change to a metric definition requires synchronized changes in three
   codebases plus the Lean 4 specification.

5. **The Lean 4 proofs, regardless of framing.** Whether you use coalgebra, KAT, or
   extraction, the Lean 4 proof of `circuit_sound_algebraic` requires formalizing:
   matrix-vector multiplication over a semiring, the semiring-to-field embedding,
   the AIR trace semantics, and the polynomial constraint evaluation. This is 8–15K
   LoC of proof engineering that is hard in any algebraic framing.

### 4.4 Recommendations

1. **Choose Approach B (KAT) with the Boolean-tested WKAT restriction.** It has the
   best risk-adjusted return. KAT is well-understood, the typeclass hierarchy is
   shallow, and the `kleene_dec` tactic is a concrete, valuable artifact. Don't
   attempt full WKAT completeness — be honest that Boolean-tested WKAT is a notational
   reorganization of standard WFA equivalence, and sell the tactic and the ZK
   application.

2. **Budget 2× the estimated LoC for the circuit synthesizer and Lean 4 components.**
   These are the components with no prior art. The circuit synthesizer will hit
   design dead-ends requiring refactoring. The Lean 4 proofs will hit Mathlib
   dependency issues requiring workarounds.

3. **Front-load the tropical semiring.** If `circuit_sound_tropical` cannot be proved
   in Lean 4, the paper loses one of its three key theorems. Start with a pilot
   proof for a 2-state tropical WFA before committing to the full formalization.

4. **Don't attempt Lean-to-Rust extraction (Approach C) unless you have 3+ years and
   a team of 5+.** CakeML is the existence proof that verified extraction is possible.
   It is also the existence proof that it takes a decade.

5. **Address the input encoding problem explicitly.** The choice between
   instance-specific AIR (reference constants in polynomials) and generic AIR
   (symbol-encoding columns) is the single most important circuit-architecture
   decision, and none of the proposals discuss it.

---

## 5. Final Scores

| Approach | Systems Difficulty | Architecture Novelty | Performance | Best-Paper | Overall |
|----------|-------------------|---------------------|-------------|------------|---------|
| A: Coalgebraic | 8 | 6 | 4 | 5 | **6.0** |
| B: KAT | 7 | 5 | 5 | 6 | **6.0** |
| C: Extraction | 9 | 7 | 2 | 4 | **5.5** |

**Note on scores:** These reflect the difficulty of *building the system correctly and
efficiently*, not the intellectual difficulty of the mathematics. All three approaches
are built on top of the same genuinely hard systems core (circuit synthesizer, STARK
integration, differential testing). The approach-specific difficulty comes from the
Lean 4 proof engineering, where Approach A is hardest (coinductive), C is most
ambitious (extraction), and B is most grounded (decision procedures).

The best-paper scores reflect that *CAV reviewers will evaluate the system*, not just
the theorems. A system that's 2× slower because it's written in Lean 4 (Approach C)
or requires coinductive proofs that only work for 20-state WFAs via `native_decide`
(Approach A) will not impress systems-oriented reviewers. Approach B's `kleene_dec`
tactic, on the other hand, is a *tool* that other researchers can use — and tools are
what win best-paper awards at verification venues.
