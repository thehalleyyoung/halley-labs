# Mathematician Verification: IsoSpec (txn-isolation-verifier)

**Panel:** 3-expert team (Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer)  
**Lead:** Deep Mathematician  
**Method:** Independent proposals → adversarial cross-critique → consensus synthesis  
**Date:** 2026-03-08  
**Evidence base:** 356KB across 22 files (theory/, ideation/, proposals/), State.json, approach.json

---

## Executive Summary

IsoSpec proposes formal operational semantics (LTS models) for three production SQL engines (PostgreSQL 16.x SSI, MySQL 8.0 InnoDB gap locking, SQL Server 2022 dual-mode concurrency), connected to Adya's DSG isolation specs via refinement, with differential SMT analysis for cross-engine portability checking. After a complete theory stage, the project has produced **356KB of planning prose and zero executable artifacts** — no Lean proofs, no SMT encodings, no Rust code. State.json confirms: `theory_bytes=0`, `impl_loc=0`, `code_loc=0`.

The panel's own red team identified 4 FATAL mathematical flaws (NULL handling in M5, PG SSI model gaps, k=3 proof errors, unvalidated SMT performance). None have been fixed in executable form. The flagship mathematical contribution (M5 predicate conflict decidability) is **unsound for standard SQL** and trivial when restricted to the fragment where it works. The honest count of genuinely new mathematical results is **1 novel artifact (PG SSI model), 0 new theorems**.

**VERDICT: CONDITIONAL CONTINUE (2-1, Skeptic dissents ABANDON)**  
**Composite: 27/50 (V6/D6/BP4/L7/F4)**  
**Scope: PG-only deep dive (Path B), not full 3-engine proposal**

---

## Panel Scores

| Dimension | Auditor | Skeptic | Synthesizer | **Consensus** | Rationale |
|-----------|:-------:|:-------:|:-----------:|:-------------:|-----------|
| **Value (V)** | 6 | 5 | 7 | **6** | Real problem, real gap, but no cited production incidents; LLM+Hermitage covers 70-80% of one-off needs; formal guarantee value is narrow |
| **Difficulty (D)** | 6 | 5 | 7 | **6** | Engine formalization is PhD-level domain engineering, but the mathematical machinery (LTS, refinement, BMC) is entirely standard; ~45-50K genuinely novel LoC, not 60K |
| **Best-Paper (BP)** | 5 | 3 | 5 | **4** | 0 genuinely new theorems (M5 unsound for SQL, k=3 unproven for predicate G2); 1 novel artifact (PG SSI model); CLOTHO "incremental extension" risk is real |
| **Laptop-CPU (L)** | 7 | 6 | 8 | **7** | Z3 QF_LIA for k≤3 tractable; Docker sequential feasible; gap lock disjunctions bounded risk |
| **Feasibility (F)** | 5 | 3 | 5 | **4** | Zero artifacts after theory stage; 11-month timeline is 14-16 months realistically; PG-only path (4-5 months) is viable |
| **Composite** | 29 | 22 | 32 | **27/50** | |

---

## Pillar 1: Extreme & Obvious Value — 6/10

**What's real:** Database migration isolation bugs are a genuine, well-documented class of failure. Oracle→PostgreSQL migration is a massive industry trend driven by $47K/processor/year licensing costs. Cloud migration market exceeds $10B annually. AWS/Azure/Google DMS handle schema and data transfer but perform zero isolation semantics verification. No existing tool answers: "Will my transaction break when I migrate, and here is a runnable script proving it?" This capability gap is real and confirmed by prior art analysis.

**What weakens the case:**

- **Zero production incident citations.** The proposal asserts "silent data corruption" and "$2-10M testing budgets" but cites zero concrete incidents where isolation divergences caused migration failures. Hermitage documents *differences*, not *failures caused by those differences*. The gap between "engines behave differently" and "this routinely causes production outages" is asserted, not evidenced.

- **LLM threat.** By 2027 submission, LLMs can enumerate common isolation differences conversationally. For one-off migration consultations, LLM + Hermitage + targeted testing covers ~70-80% of practitioner needs. IsoSpec's unique value is CI/CD-integrated, completeness-guaranteed, regression-testable analysis — but this argument is not made sharply.

- **Adoption barriers.** DBAs don't use formal verification tools. The proposal oscillates between "tool for DBAs" and "framework for PL researchers" without committing to an audience.

**Panel disagreement:** Skeptic (5) argued consultants cover 70-75% of the value; Synthesizer (7) argued the gap is genuinely unserved. Auditor (6) was best-calibrated — real problem with narrow incremental value over existing approaches.

---

## Pillar 2: Genuine Difficulty — 6/10

### Honest LoC Assessment

| Category | Proposal Claims | Consensus Estimate | Notes |
|----------|:-:|:-:|---|
| Engine models (3) | 18K | 14-16K | PG best-documented; MySQL gap locking genuinely hard; SQL Server documentation gaps |
| M5 predicate theory | 5K | 3-4K | Theory is hard but code is small; complexity is in proof, not LoC |
| SMT encoding | 7K | 5-6K | Standard BMC patterns with engine-specific content |
| Refinement + portability | 9K | 6-7K | Known technique, novel framing |
| Parser + IR | 6K | 5-6K | sqlparser-rs handles base SQL; dialect extensions are bounded |
| Witness + mixed-iso | 6K | 4-5K | MUS standard; MaxSMT unreliable |
| Lean 4 | 5K | 3-5K | Proof engineering is unpredictable |
| Validation oracle | 4K | 3K | Docker infra |
| **Total genuinely novel** | **~60K** | **~45-50K** | **Still 20-25% inflated from 60K claim** |

### Difficulty Decomposition

- **Genuinely hard (requires deep expertise):** PG SSI formalization including dangerous structure detection and read-only optimization; MySQL InnoDB index-dependent gap locking over-approximation; M5 decidability boundary characterization
- **Tedious but tractable (known techniques):** SMT encoding via QF_LIA+UF+Arrays following CLOTHO patterns; refinement checking via standard process algebra; MUS-based witness extraction; Docker orchestration; SQL parser dialect extensions
- **The ratio:** ~20% novel math, ~40% novel domain engineering, ~40% known-technique application

**Panel disagreement:** Skeptic argued engine models are "reading source code in math notation" (D=5); Synthesizer called them "PhD-level" (D=7). Truth is between: formalizing `predicate.c` into an LTS requires dozens of non-trivial abstraction decisions, but the mathematical framework is entirely standard. The difficulty is in *domain content*, not *methodology*.

---

## Pillar 3: Best-Paper Potential — 4/10

### Math Novelty Audit

| # | Claimed Contribution | Status After Red Team | Honest Rating |
|---|---|---|---|
| **M5** | Predicate conflict decidability (FLAGSHIP) | **UNSOUND for SQL** — NULL 3VL breaks convex polytope argument (FATAL-1). Restricted to NOT NULL numerics, reduces to LP feasibility — textbook since Dantzig 1947. | Application of known technique to new domain; not a new theorem |
| **M1** | Engine operational semantics | PG SSI model missing memory pressure behaviors (FATAL-2). MySQL/SQL Server models untested. | Novel artifact, standard technique. Genuine contribution as a formalization, not as math. |
| **M2** | Isolation refinement relation | Standard trace inclusion from CSP/CCS (Hoare 1985, Milner 1989). Red team notes trace inclusion may miss timing-dependent SSI properties (SERIOUS-3). | Known technique applied to new domain. Not novel. |
| **k=3** | Bounded completeness | **PROOF ERRORS** — G1a needs k=2 not k=3; G2-item example uses 6 transactions (FATAL-3). Predicate-level G2 completely unanalyzed. | Unproven. Likely correct for item-level by simple graph theory, but the formal claim is not established. |
| **M3** | Portability coNP-completeness | Expected complexity classification. | Routine. |
| **M6** | Mixed-isolation MaxSMT | Z3 MaxSMT has 30-50% timeout rates on comparable sizes. | Engineering, not theory. |
| **M7** | Bounded soundness/completeness | Minor — follows from anomaly definition structure. | Almost definitional. |
| **M8** | Compositionality | Standard monotonicity of refinement. | Almost definitional. |

**Honest count: 1 genuine novel artifact (PG SSI operational semantics), 0 genuinely new mathematical theorems.** M5's decidability "result" is standard LP feasibility restricted to a fragment where the problem is trivial. The k=3 "proof" contains demonstrable errors. M2, M7, M8 are textbook applications.

The project's **own verification report** scored theory quality at 2/5 for theorems, noting "several 'theorem' statements are actually definitions."

### CLOTHO Differentiation

The prior art analysis honestly admits: "IsoSpec adopts CLOTHO's parse→encode→solve architecture and extends it with engine-specific constraint content." A hostile SIGMOD reviewer writes: "This is CLOTHO extended with engine-specific constraint libraries. The intellectual contribution is the constraint content, which is engineering, not science." This reviewer would not be wrong. The differentiation is real — CLOTHO models specs, IsoSpec models implementations — but the methodological contribution is incremental.

### Discovery-First Framing

The "N undocumented isolation behaviors" headline is the strongest path to best-paper, but:
- The red team identified that novelty is circularly defined (SERIOUS-8): "not in Hermitage or top SO results" is unfalsifiable
- Hermitage already found 47+ divergences with simple test scripts. How many more can formal models surface?
- If N < 10, the paper has no headline. If N ≥ 15 with migration-affecting discoveries, it's compelling. This is a bet on an unknown quantity.

### Venue Analysis

- **SIGMOD 2027:** Strong-accept if discovery results are compelling (15+ novel divergences + migration case study). Best-paper requires extraordinary findings. With PG-only scope, SIGMOD Demo is more realistic.
- **CAV 2027:** Tool paper with formal verification angle. Strong-accept for PG SSI formalization + SMT encoding.
- **VLDB:** Experimental paper with discovery results. Weaker formal contribution but broader systems audience.

---

## Pillar 4: Laptop-CPU Feasibility — 7/10

- **Z3 QF_LIA+UF+Arrays for k≤3:** Well within tractability envelope. Engine-specific constraints *prune* search space. 50K-100K constraints for k=3, n=50 is comfortable for Z3.
- **Docker for 3 engines:** PG (~200MB) + MySQL (~400MB) + SQL Server (~1.5GB) = ~2.1GB sequential. Fine on 16GB RAM.
- **Gap lock disjunctions:** For tables with 5 indexes, 3 range predicates: 5³ = 125× branching. Approaching sensitivity zone but manageable with index-stratified refinement.
- **Lean 4:** CPU-only, batch-mode, minutes for ~5K LoC. No concern.
- **Interleaving forcing:** 25-40% failure rate for Docker-based advisory lock forcing (red team's realistic estimate, vs. 10% claimed). Manageable with retry logic.

---

## Pillar 5: Feasibility — 4/10

### The Zero-Artifact Problem

The most damaging fact about this project: **after completing the theory stage, there are zero executable artifacts.** 356KB of markdown prose exists, but:
- No `.lean` files (0 bytes of mechanized proof)
- No `.smt2` files (0 bytes of SMT encoding)
- No `.rs` files (0 bytes of Rust implementation)
- State.json: `theory_bytes: 0`, `impl_loc: 0`

The theory stage produced *proposals about what to prove*, not *proofs*. The "formal methods proposal" is a narrative containing mathematical notation, not mechanized or validated mathematics. The k=3 "proof sketch" contains elementary errors (G1a k-value, G2-item transaction count) that would be caught instantly by any formalization attempt. This suggests the mathematical claims have never been tested against any rigorous framework.

### Timeline Reality

| Phase | Claimed | Realistic | Notes |
|-------|---------|-----------|-------|
| PG model + infra | 2 months | 3-4 months | Well-documented but still requires reading ~4K lines of `predicate.c` |
| MySQL + M5 | 2 months | 3-4 months | MySQL gap locking is "the single hardest problem" + M5 theory simultaneously — unrealistic |
| SQL Server + refinement | 2 months | 2-3 months | Documentation gaps; 12 refinement pairs |
| Witnesses + discovery | 2 months | 2 months | Engineering, realistic |
| Evaluation + Lean | 2 months | 3-4 months | Lean proof engineering is unpredictable; R4 at 35% delay |
| Paper writing | 1 month | 1.5 months | 14-page SIGMOD paper while polishing experiments |
| **Total** | **11 months** | **14-18 months** | **30-60% compressed** |

Engine models alone (PG + MySQL + SQL Server) require 8-11 months realistically. That consumes the entire timeline before M5, Lean, evaluation, or paper writing.

### The PG-Only Path (Path B)

The Synthesizer's staged approach is the only credible plan:
- **PG-IsoSpec:** 19K LoC, 4-5 months, 60-70% publication probability
- Validates SMT approach on one engine before committing to three
- Produces a standalone publication (SIGMOD Demo / CAV tool paper) regardless of full project outcome
- Every FATAL flaw either fixed (FATAL-1 scoped to NOT NULL PG) or manageable (FATAL-2 scoped to unbounded memory, FATAL-3 dropped, FATAL-4 benchmarkable at PG scale)

---

## Fatal Flaws

| # | Flaw | Severity | Fixable? | Fix |
|---|------|----------|----------|-----|
| **FF1** | M5 decidability unsound for SQL (NULL 3VL) | HIGH | YES | Restrict to NOT NULL; sound over-approximation for nullable. Weakens claim but preserves soundness. |
| **FF2** | PG SSI model missing memory pressure (lock promotion, SIREAD cleanup, summarization) | HIGH | YES | Scope to "unbounded lock memory" with explicit adequacy criterion. Standard practice. |
| **FF3** | k=3 proof mathematically wrong (G1a k-value, G2-item count, predicate G2 unanalyzed) | **CRITICAL** | PARTIALLY | Item-level: straightforward graph theory fix. Predicate-level G2: genuinely open question — drop the general claim. |
| **FF4** | SMT performance claims unsubstantiated | MEDIUM | YES | Benchmark on 10 workloads. Day's work once encoding exists. |
| **FF5** | Zero executable artifacts after theory stage | **CRITICAL** | — | Start building. The next output must be code, not markdown. |
| **FF6** | Consultant replicability at 65-70% | HIGH | NO | Structural — the irreplaceable formal-guarantee value is narrow. |
| **FF7** | Novel divergence count unknown (N may be <10) | HIGH | UNKNOWN | Empirical — can only be resolved by building and running the tool. |

### Are There Showstoppers?

**FF3 (k=3 proof) is the competence signal.** Getting G1a wrong (needs k=2, claimed k=3) and miscounting transactions in G2-item are elementary errors that any database formalization researcher would catch. This suggests the mathematical claims have not been subjected to peer review, formalization, or even careful self-review. Combined with FF5 (zero artifacts), the pattern is: ambitious claims, no verification.

However, the underlying *idea* is sound. The k=3 bound is likely correct for item-level anomalies by simple graph-theoretic argument (minimal directed cycles in DSG). The project's mathematical weakness is in *execution and rigor*, not in *conception*.

---

## Math Quality Assessment

### Is the Math Load-Bearing?

**M5 (Predicate Conflict):** Would be load-bearing if correct — the tool cannot handle WHERE clauses without it. Currently unsound. The NOT NULL restriction makes it a useful optimization (handle the easy cases in P, conservatively over-approximate the rest) rather than a fundamental result. A consultant could replace this with conservative over-approximation (treat all complex predicates as conflicting) at the cost of false positives. **Verdict: Enabling optimization, not load-bearing.**

**M1 (Engine Models):** Genuinely load-bearing — the system cannot exist without formal engine models. The PG SSI LTS is the core intellectual asset. However, the "math" here is formalization of existing behavior, not new mathematics. The difficulty is in *domain knowledge faithfully encoded*, not in *mathematical technique*. **Verdict: Load-bearing artifact, not load-bearing math.**

**M2 (Refinement):** Standard technique connecting M1 to Adya. A consultant could check "does this engine behavior violate this spec?" empirically without the formal refinement. The formalism enables systematic checking across 12 pairs, which is useful but not strictly necessary. **Verdict: Enabling formalism, partially replaceable by empirical comparison.**

**k=3 Bound:** Load-bearing for bounded model checking — without it, you enumerate over arbitrary k. But the bound is also "obvious" to anyone who understands Adya's anomaly definitions. It's important for implementation efficiency, not for theoretical depth. **Verdict: Load-bearing optimization with elementary proof (when done correctly).**

### Overall: The math enables automation but does not create new mathematical knowledge. The project's value is in *domain artifacts* (engine models) and *engineering integration* (SMT-based portability checking), not in *mathematical contributions*. A consultant with deep database knowledge could replicate 65-70% of the value through empirical testing.

---

## Salvage Analysis

### Crown Jewel: PostgreSQL 16.x SSI Operational Semantics

The single highest-value component. No one has produced an implementation-faithful LTS for PostgreSQL SSI. Cahill et al. (2008) describe SSI theory; PostgreSQL's `predicate.c` implements something subtly different. The formal gap between the two has never been characterized. This is:
- Standalone publishable (SIGMOD Demo / CAV tool paper)
- ~8K LoC of core model, 4-5 months
- Maximum ratio of novelty to effort
- Anchors every other path

### Salvage Paths (if full project fails)

| Path | Scope | Time | P(pub) | Venue |
|------|-------|------|--------|-------|
| **B: PG Deep Dive** | PG SSI model + refinement + anomaly detection + 5-8 novel divergences | 4-5 months | **60-70%** | SIGMOD Demo, CAV tool |
| **C: Empirical Discovery** | Docker testing infra + N novel divergences (no formal models) | 3-4 months | **65-75%** | VLDB experimental |
| **C+E: Discovery + Tool** | Empirical testing + practical migration checker | 3-4 months | **55-65%** | ICSE/FSE tool |
| **D: M5 Theory** | Pure decidability characterization (fix NULLs, prove results) | 4-6 months | **35-50%** | PODS, CONCUR |
| **A: Full Project** | 3 engines + M5 + refinement + Lean | 14-18 months | **40-55%** | SIGMOD research |

**Recommendation: Path B is the optimal first milestone.** It validates the SMT approach, benchmarks Z3 performance, and produces a standalone publication regardless of whether the full project continues. Then stage to Path A if B succeeds.

### Math to Keep vs. Drop

| **KEEP** | **DROP** |
|----------|----------|
| M1 PG SSI LTS (crown jewel) | k=3 general completeness claim (broken) |
| M5 NOT NULL fragment (as optimization) | Compositionality M8 (definitional) |
| Refinement Definition 4 (useful framing) | Mixed-isolation M6 (MaxSMT unreliable) |
| Portability coNP (expected but needed) | Full Lean mechanization (scope to 1-2 lemmas) |

---

## Panel Recommendation

### CONDITIONAL CONTINUE (2-1: Auditor + Synthesizer CONTINUE; Skeptic dissents ABANDON)

**Skeptic's dissent (strongest form):** "The project has completed a theory stage and produced zero formalized artifacts. The math is either wrong (k=3), trivially scoped (M5 without NULLs), standard (M2 refinement), or engineering formalization (M1). The 11-month timeline to produce ~55K LoC starting from zero is unrealistic. P(best-paper) ≈ 2%. ABANDON."

**Auditor's response:** "The 4 FATAL flaws are all fixable — FATAL-1 and FATAL-2 by scoping, FATAL-3 by correct graph-theory argument for item-level, FATAL-4 by benchmarking. The PG SSI formalization is a genuine intellectual contribution. P(any pub at top venue) ≈ 45-55% via Path B."

**Synthesizer's response:** "The intellectual assets are real — 356KB of deep domain analysis, well-articulated problem framing, clear architecture. Path B (PG-only, 4-5 months) creates value. Staged commitment with kill gates prevents wasted effort."

**Lead mathematician's judgment:** The Skeptic raises valid concerns about mathematical rigor — the k=3 errors are an alarming competence signal. But the underlying conception is sound: engine-specific operational semantics connected via refinement to isolation specifications is a real, unsolved problem. The PG SSI model has never been built. If it can be built correctly and used to discover novel engine behaviors, that is a publishable contribution at a good venue. The full 3-engine SIGMOD research paper is not credible at current state; the PG-only path is.

### Binding Conditions

**BC-1 (Week 2): Fix or honestly scope the math.** M5 restricted to NOT NULL with explicit limitations. k=3 claim either rigorously proved for item-level or dropped. If neither: ABANDON.

**BC-2 (Week 6): Working SMT prototype.** A `.smt2` file that encodes PG SSI write skew detection and returns correct SAT/UNSAT. If this doesn't exist: ABANDON.

**BC-3 (Week 10): One novel divergence.** At least one PG isolation behavior not in Hermitage, confirmed on real engine. If zero: PIVOT to pure formalization paper.

**BC-4 (Week 16): Draft paper with evaluation.** Venue-ready draft with at least Hermitage reproduction + 3 novel findings. If not: ABANDON.

### Kill Gates

| Gate | Deadline | Condition | If Failed |
|------|----------|-----------|-----------|
| G1 | Week 2 | Math fixed or scoped | ABANDON |
| G2 | Week 6 | Working SMT for PG SSI | ABANDON |
| G3 | Week 10 | ≥1 novel PG divergence | PIVOT to formalization-only paper |
| G4 | Week 16 | Draft paper | ABANDON |

---

## Probability Estimates (Consensus)

| Outcome | Probability | Notes |
|---------|:-----------:|-------|
| Best-paper at SIGMOD | **2-4%** | Two FATAL math flaws, zero artifacts, standard techniques |
| Strong accept at SIGMOD/CAV | **25-35%** | PG-only path with fixed math; SIGMOD Demo or CAV tool paper |
| Any publication at top venue | **45-55%** | Path B is viable; discovery results are the swing factor |
| Working tool (no pub) | **55-65%** | PG-only verifier is technically achievable |
| ABANDON at next gate | **35-45%** | High probability of hitting G1 or G2 |

---

## The Single Most Important Thing

**Build the SMT encoding. Stop writing markdown.**

The project has 356KB of planning documents and zero bytes of executable mathematics. The next artifact produced must be a `.smt2` file encoding a concrete PG SSI schedule with a known anomaly. If you cannot encode "write skew under Serializable on PostgreSQL 16" as an SMT instance within one week, the project's theoretical framework does not translate to implementation, and no amount of additional planning will fix that.

The PG SSI operational semantics are the crown jewel. Protect their quality and depth at the expense of breadth. Everything else — MySQL, SQL Server, Lean proofs, M5 decidability, mixed-isolation optimization — is secondary until the core engine model is validated as an SMT-encodable, anomaly-detecting, divergence-discovering formal artifact.

---

## Evaluator Calibration

| Evaluator | Overall Calibration | Strength | Weakness |
|-----------|:-------------------:|----------|----------|
| **Auditor** | Best-calibrated (hit or nearest on 4/5 dimensions) | Transparent evidence chains; reproducible methodology | Slightly generous on BP (5 vs consensus 4) |
| **Skeptic** | Best at risk identification | Correctly identified zero-artifact problem; accurate consultant replication estimate | Overstated on Feasibility (3 vs consensus 4); may conflate "currently broken" with "unfixable" |
| **Synthesizer** | Best constructive recommendations | Path B staged approach is the optimal plan; crown jewel identification accurate | Consistently 1-2 points optimistic; anchored by potential rather than current state |

---

## Verdict Summary

A project with a sound conception, a genuine unsolved problem (engine-specific isolation formalization), and one potential crown jewel (PG SSI operational semantics) — buried under 30% LoC inflation, flawed mathematical claims, and zero executable output after a complete theory stage. The full 3-engine SIGMOD research paper is not credible. The PG-only deep dive (Path B) is viable with 4-5 months of focused implementation work, producing a publishable result at SIGMOD Demo or CAV tool paper tier.

**Decision: CONDITIONAL CONTINUE — Path B only, with mandatory kill gates G1-G4. The next artifact must be code, not markdown.**
