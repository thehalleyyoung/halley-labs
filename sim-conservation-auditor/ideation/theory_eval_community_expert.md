# Community Expert Verification: sim-conservation-auditor (ConservationLint)

## Committee
- Independent Auditor: Evidence-based scoring and challenge testing
- Fail-Fast Skeptic: Aggressively reject under-supported claims
- Scavenging Synthesizer: Salvage value from abandoned proposals
- Independent Verifier: Final signoff with bias/consistency checks
- Cross-critique phase: adversarial challenge + synthesis

## Verdict: CONDITIONAL CONTINUE — 4-Week Prove-or-Kill Sprint

Composite: **4.1/10** (V4/D5/BP3/L5.5/F3.5)

Down from ideation gate 5.75/10. The core idea — bridging Noether's theorem and program analysis — remains genuinely novel and unanimously confirmed as unexplored territory. However, `theory_bytes=0` after the theory stage is a concrete failure to deliver, not a marginal miss. The project has zero proofs, zero code, zero user validation, and zero empirical data after completing crystallization, ideation, adversarial debate, verification signoff, and theory — five stages producing no tangible artifacts beyond structured assessment documents. The 4-week sprint is a bounded experiment to test whether the two gating unknowns (T2 proof viability, extraction feasibility) are solvable, not a project endorsement.

**Vote: 2-1 for ABANDON (Auditor, Skeptic) vs. CONDITIONAL CONTINUE (Synthesizer).** The lead resolves to CONDITIONAL CONTINUE because the sprint is cheap (4 weeks, ~2K LoC), the gating questions are testable, and the salvage floor is high (P(≥1 pub) ≈ 60-70% regardless of outcome). The override of majority ABANDON is procedurally legitimate but creates an enforcement obligation: the 4-week hard deadline is non-negotiable.

---

## Pillar Scores

### 1. EXTREME AND OBVIOUS VALUE — 4/10

**The pain is real but narrow, not the community's top priority, and faces a rising LLM threat.**

**What practitioners actually care about (SC, SIAM CSE, ICIAM):**
1. Performance/scalability — "Make my simulation run faster"
2. Portability — "Make my code run on the new exascale machine"
3. Reproducibility — "Make my results match across platforms"
4. General correctness — "Debug my segfault / wrong answer"
5. Numerical stability — "Fix my NaN / divergence"
6. **Conservation-specific violations — ConservationLint's target**

Conservation violations are real (the Wan et al. 2019 JAMES citation is compelling — a conservation-violating coupling scheme persisted three years in a major climate model, corrupted published conclusions). But they rank ~6th in practitioner pain hierarchy.

**Evidence for narrow value:**
- **TAM: ~300–500 users** (the proposal's own honest estimate). A talk at SIAM CSE would draw an interested niche (~50 people), not a packed plenary.
- **Existing monitors partially solve detection.** GROMACS `gmx energy`, LAMMPS `thermo_style`, OpenMM test suites detect *that* conservation is violated. They don't localize or prove obstructions, but detection covers the majority of practitioner needs.
- **Python-only scope excludes ~80% of production codes.** National lab teams running Fortran/C++ — who most need conservation auditing — are outside scope.
- **The liftable fragment excludes the most important bug pattern.** Data-dependent branching (`if r < r_cut`) — the most common MD conservation-violation source — is explicitly excluded from Tier 1. The tool cannot formally analyze the pattern causing the most conservation bugs in its primary domain.
- **Zero validated demand.** No user interviews, no letters of support. The "3–5 structured interviews" are planned for Phase 1 — still unexecuted after five pipeline stages.

**LLM competition is existential and worsening:**
- GPT-4/Claude already covers ~70% of diagnostic value for single-method integrators at zero cost.
- The proposal's own Skeptic estimated unique value at ~6% of total problem space.
- ConservationLint's moat (formal proofs, obstruction certificates) protects only the liftable fragment.
- If the liftable fragment covers 15-25% (Tree-sitter only) and LLMs cover 100% heuristically, most practitioners choose breadth.

**Why 4 and not lower:** The paradigm ("physics-aware program analysis") is genuinely novel and opens a research direction. The obstruction certificate concept — "this splitting *cannot* conserve angular momentum; restructure" — is qualitatively new. The Wan et al. scenario is real.

**Why 4 and not higher:** Zero demand validation after five pipeline stages. Narrow TAM. Python-only excludes primary audience. LLM competition growing. The Skeptic's demand score of 3/10 is harsh but evidence-grounded; the Auditor's 4/10 is already generous.

### 2. GENUINE SOFTWARE DIFFICULTY — 5/10

**~30% genuinely novel engineering. T2 is the real open question. Integration risk is the real difficulty.**

| Component | LoC Est. | Classification | Evidence |
|-----------|----------|---------------|----------|
| Hybrid extraction (jaxpr + dispatch + Tree-sitter) | 12K | **NOVEL** | Zero prototype; no precedent for jaxpr → conservation IR |
| Conservation-aware IR | 5K | INCREMENTAL | Hard-but-known compiler engineering; polyhedral model precedent |
| Symbolic algebra engine | 7K | ROUTINE | Rust `polynomial` crate ~3K; adding Lie brackets doubles it |
| Restricted Lie-symmetry solver | 5K | ROUTINE | "Solvable in milliseconds" per proposal's own words |
| BCH engine + provenance tags | 5K | INCREMENTAL | Textbook (Hairer/Lubich/Wanner 2006) + bitset labels |
| Obstruction checker (T2) | 3K | **POTENTIALLY NOVEL** | Self-graded C in approach.json; EXPSPACE in general |
| Causal localization | 6K | **NOVEL as PL concept** | "Differential symbolic slicing" is new; implementation is graph traversal |
| Dynamic tier | 7K | ROUTINE | Standard numerical differentiation + causal intervention |
| Benchmark + eval + CLI | 21K | ROUTINE | Infrastructure and test code |

**Genuinely novel: ~21K LoC (~30% of 71K).** The remaining 70% is known techniques, infrastructure, and test code.

**The integration challenge is real but not unprecedented.** Five pipeline stages with three ontologies (continuous ODEs, discrete flows, imperative code) is significant engineering. But the Difficulty Assessor noted: "Remove extraction (which may be impossible) and the remaining pipeline is a 4/10."

**Why 5 and not higher:** The genuinely hard part (code→math extraction) has zero feasibility evidence. The tractable parts (Lie-symmetry, BCH) are textbook. approach.json grades T2 as C. theory_bytes=0 means the one genuinely hard open math problem hasn't been touched.

### 3. BEST-PAPER POTENTIAL — 3/10

**theory_bytes=0 after the theory stage is devastating. The crown jewel has no proof.**

**The theory stage failed to produce output:**
- No formal proof of T2 exists — not even a sketch
- No formal construction of T1 exists
- No paper draft or skeleton exists
- The approach.json (34KB) is a structured assessment document, not mathematical content
- All theorem entries in approach.json have null proof_sketch and null status fields

**T2 (crown jewel) assessment:**
- Decidability follows trivially from Tarski-Seidenberg (known since 1951, no novelty)
- The efficient reduction (the claimed novel contribution) is unproven
- approach.json self-grades T2 as C and flags "exponential complexity for realistic problems" (HIGH severity)
- Probability of elegant T2 result: ~15%. Brute-force for small k,p (~50%). Intractable in general (~35%).

**T1 is self-described as "~20% new math, ~80% known BCH with engineering annotations."** This is engineering scaffolding, not a theorem.

**T3 (Liftable Fragment) is a definition, not a theorem.** Every static analysis paper defines its fragment.

**Novel theorem-equivalents: ~0.55** (T2 conditional at ~0.5, T1 at ~0.05, T3 at 0). Far below the ~1.5-2.0 range needed for best-paper consideration at OOPSLA/CAV.

**The "impossible bridge" narrative:** OOPSLA reviewers would find "bridging geometric numerical integration and program analysis" interesting but would immediately ask: "Show me the bridge works." With theory_bytes=0 and impl_loc=0, the bridge is a proposal. SLAM had a working tool; Herbie had experimental results; Compcert had a proof. ConservationLint has a plan.

**P(best paper) ≈ 1-3%.** Self-assessed at 5% *before* the theory stage failure. Revised downward.

### 4. LAPTOP-CPU FEASIBILITY & NO-HUMANS — 5.5/10

**The architecture is genuinely CPU-only with no training data, but phase-space annotations create human burden and runtimes are unvalidated.**

**Laptop-tractable components:**
- Lie-symmetry analysis with restricted ansatz: milliseconds (structured linear algebra, ≤50 state vars)
- BCH at k=5, p=4: O(625) brackets, seconds on any laptop
- Dynamic tier: numerical integration + ablation, CPU-bound
- No GPU at any stage, no training data, no neural networks

**Concerns:**
- BCH at k=10, p=6: ~1M brackets × ~1ms ≈ 17 minutes (busts 10-minute budget)
- Dynamic tier runtime unbounded for complex codes with many conservation laws × ablation experiments
- **Phase-space annotations require domain expertise.** Users must declare phase-space variables and expected conservation laws. This is non-trivial human labor not reflected in "no-humans" framing.
- Noether's Razor baseline is GPU-only; CPU fallback (SINDy) is untested

**Why 5.5:** The tool itself is CPU-only (genuine strength). But the qualified claim (k≤5, p≤4) has never been empirically validated, the dynamic tier runtime is unknown, and phase-space annotations create real human burden. The Skeptic's and independent verifier's concerns about human labor in annotations are legitimate.

### 5. FEASIBILITY — 3.5/10

**Zero deliverables after theory stage. Three stacked unknowns. Prior gate conditions unmet.**

| Expected After Theory Stage | Actual |
|---------------------------|--------|
| Formal proof of T2 on 2-3 examples | Nothing |
| Formal construction of T1 | Nothing |
| Paper draft or skeleton | Nothing |
| Prototype extraction on simple kernels | Nothing |
| Any code at all | 0 LoC |
| Theory document | 0 bytes |

**Prior ideation gate conditions (3/4 unmet):**
- BC1 (scope reduction): ⚠️ Partially addressed in text, not validated
- BC2 (honest coverage measurement): ❌ No empirical measurement performed; 40-60% claim persists unvalidated
- BC3 (venue decision): ❌ Unresolved — final_approach says OOPSLA; approach.json says PLDI
- BC4 (external benchmark fidelity): ❌ No domain-expert validation performed

**Kill gate pass-rate assessment:**

| Gate | P(pass) | Basis |
|------|---------|-------|
| G1: Extraction viability | 40-55% | Zero prototype; jaxpr interception unprecedented for conservation analysis |
| G2: Coverage ≥20% | 50-60% | Achievable with hybrid if G1 passes |
| G3: T2 validation | 35-45% | Self-graded C; EXPSPACE in general; efficient reduction unproven |
| G4: LLM differentiation ≥30% | 55-65% | Favorable threshold on benchmarks, but LLMs improving monthly |
| G5: End-to-end demo k≥3 | 25-35% | Requires all prior gates + integration |

**P(all 5 gates pass) ≈ 2-8%.** Combined with P(best paper | all pass) ≈ 10-15%, the expected best-paper probability is <1%.

**Revised probabilities (post-theory-failure):**

| Outcome | approach.json estimate | Revised |
|---------|----------------------|---------|
| P(top venue) | 25% | 12-15% |
| P(best paper) | 5% | 1-3% |
| P(abandonment) | 15% | 35-50% |
| P(any publication) | 70% | 50-60% (salvage paths viable) |

**Why 3.5 and not lower:** Kill gates genuinely bound investment. Salvage publications exist at every gate. Phase 1A (~8K LoC, 3 months) is a rational bounded experiment.

**Why 3.5 and not higher:** theory_bytes=0 is not "the theory is hard" — it's "the theory stage produced nothing." Three stacked critical unknowns (extraction, T2, coverage) multiply. The liftable fragment excludes the most important bug pattern.

---

## 6. Fatal Flaws

### Flaw 1: Theory Stage Produced Zero Output — CRITICAL

The theory stage completed (`theory_complete` in State.json) with `theory_bytes=0`. T2 (crown jewel) has no proof, no sketch, no concrete worked examples. The `proposals/proposal_00/theory/` directory is empty. This is a realized failure, not a risk. approach.json's theorem entries have null proof_sketch and null status for all three theorems.

**Why potentially fatal:** If T2 cannot be proven (or is trivially Tarski-Seidenberg), the paper has no mathematical contribution. T1 is bookkeeping. T3 is a definition. Without T2, this is a systems paper about a tool that doesn't exist.

**Mitigation:** 4-week sprint condition C1 tests this directly.

### Flaw 2: T2 May Be Trivial or Intractable — HIGH

T2 is either (a) trivially decidable via Tarski-Seidenberg (no novelty), (b) EXPSPACE-complete in general (no practical value), or (c) admits an efficient structured reduction (the hoped-for contribution). Option (c) is undemonstrated.

**Evidence:** approach.json self-grades T2 as C. Adversarial finding: "Obstruction checking has exponential complexity for realistic problems" (severity: HIGH, status: MITIGATED not RESOLVED). The Difficulty Assessor: "≤200 Lie bracket conditions — each a polynomial identity checkable by direct computation. This is a finite, brute-force calculation, not an elegant structural theorem."

**Mitigation:** Even brute-force T2 for k≤5, p≤4 provides a qualitatively new capability (obstruction certificates). The paper can succeed with this, but not as best-paper.

### Flaw 3: Hybrid Extraction Entirely Unvalidated — HIGH

No prototype of jaxpr interception for conservation analysis exists anywhere. The TorchDynamo analogy glosses over JAX's different tracing model. If extraction fails, the static tier is nonviable and the paper degrades from "impossible bridge" to "practical dynamic analyzer."

**Mitigation:** Kill gate G1 (Month 3) and sprint condition C2.

### Flaw 4: Liftable Fragment Excludes Most Important Bug Pattern — SERIOUS

Data-dependent branching (`if r < r_cut`) — the most common MD conservation-violation source — is explicitly excluded from Tier 1 analysis. The tool cannot formally analyze the pattern causing the most conservation bugs in molecular dynamics.

**Mitigation:** Tier 2 covers these dynamically but without formal guarantees — exactly the guarantees differentiating ConservationLint from simpler tools.

### Flaw 5: Zero Demand Validation After Five Stages — SERIOUS

No user interviews. No letters of support. No survey data. Demand is assumed from first principles (the Wan et al. anecdote + multiplication: 0.2 × 0.3 × 0.2 × 0.5 × 50,000 ≈ 300 users). The proposal plans interviews for Phase 1 — still unexecuted after crystallization, ideation, adversarial debate, verification signoff, and theory stages.

**Mitigation:** Sprint condition C3 (one user signal).

### Flaw 6: LLM Obsolescence — SERIOUS (Inherent)

GPT-4/Claude covers ~70% of diagnostic cases today. Model capabilities improve rapidly. By the time a 71K LoC tool ships (12+ months), LLMs may handle obstruction-like reasoning via chain-of-thought. The unique-value window (heterogeneous compositions with k>2, formal obstruction proofs) is narrow and shrinking.

**Mitigation:** Kill gate G4 tests LLM differentiation. The formal-proof moat is the only durable defense.

---

## 7. What Can Be Salvaged If Grand Vision Fails

| Priority | Option | Venue | Timeline | P(accept) | Risk |
|----------|--------|-------|----------|-----------|------|
| 1 | Dynamic-only auditor + benchmark suite | ICSE/FSE Tools | 4-5 months | 60-70% | Low |
| 2 | Conservation benchmark suite (standalone) | JOSS | 2-3 months | 85-90% | Very Low |
| 3 | Survey/vision paper: physics-aware program analysis | ICSE-NIER / Computing Surveys | 2-3 months | 50-65% | Low |
| 4 | T2 as standalone math contribution | Numerische Math / BIT | 3-6 months | 40-55% | High (conditional on proof) |
| 5 | IR design as infrastructure contribution | SLE / GPCE | 4-5 months | 35-45% | Medium |

The salvage floor is genuinely high: P(≥1 publication via salvage) ≈ 85-95%.

---

## 8. VERDICT: CONDITIONAL CONTINUE — 4-Week Prove-or-Kill Sprint

### Binding Conditions (ALL within 4 weeks, hard calendar deadline, no extensions)

| # | Condition | Success Criterion | Classification |
|---|-----------|-------------------|----------------|
| **C1** | T2 proof on one concrete example | Prove obstruction criterion for Strang splitting (k=2), SO(3) angular momentum, order p=2. Proof must reveal structure beyond direct Tarski-Seidenberg; if trivial, document honestly and reclassify. | **EXISTENTIAL** |
| **C2a** | NumPy extraction prototype (MUST) | Extract IR from ≥1 pure-NumPy Verlet integrator. Demonstrate: extraction → Lie-symmetry check → conservation verdict. ~200 lines of code. | **EXISTENTIAL** |
| **C2b** | JAX-MD extraction prototype (SHOULD) | Extract IR from ≥1 JAX-MD force kernel via jaxpr interception. Failure acceptable if documented with clear technical obstacle analysis. | **ENABLING** |
| **C3** | One user signal | Documented conversation with ≥1 non-author Python simulation developer, including explicit willingness-to-try assessment. | **ENABLING** |
| **C4** | Written theory deliverable | ≥5 pages of mathematical content (T2 proof attempt, extraction formalization, or both). Addresses theory_bytes=0. | **ENABLING** |

### Failure Cascade (Asymmetric)

- **C1 AND C2a both pass → CONTINUE** to Phase 1A (remaining 2 months, full kill-gate structure G1-G5)
- **Exactly one of C1/C2a fails → PIVOT** to salvage option matching surviving capability:
  - C1 passes, C2a fails → T2 standalone math paper (Numerische Math)
  - C1 fails, C2a passes → Dynamic-only auditor + benchmarks (ICSE/FSE)
- **C1 AND C2a both fail → ABANDON.** Execute salvage options 2+3 (benchmark suite + survey paper).
- **C3/C4 failure → 2-week extension for these conditions only**, not a verdict change

### If Phase 1A Proceeds (Months 2-4):

| Gate | Deadline | Trigger |
|------|----------|---------|
| G1: Extraction viability (≥3 JAX-MD + ≥3 NumPy kernels) | Month 3 | Fail → restrict to NumPy + Tree-sitter |
| G2: Coverage ≥15% on 3/5 codebases | Month 4 | Fail → dynamic-only pivot |
| G3: T2 efficient reduction (runtime <60s for k=3, p=3) | Month 4 | Fail → reclassify as Conjecture |
| G4: LLM differentiation ≥30% on benchmarks | Month 5 | Fail → pivot to benchmark/survey paper |
| G5: End-to-end demo k≥3 | Month 6 | Fail → evaluate salvage |

### Score Summary

| Dimension | Ideation Gate | Theory Gate | Delta | Trend |
|-----------|--------------|-------------|-------|-------|
| Extreme Value | 5 | **4** | -1 | ↓ No new evidence; LLM threat worsening |
| Genuine Difficulty | 6 | **5** | -1 | ↓ ~30% novel; integration risk unvalidated |
| Best-Paper Potential | 5 | **3** | -2 | ↓↓ theory_bytes=0; T2 unproven |
| Laptop-CPU / No-Humans | 7 | **5.5** | -1.5 | ↓ Unvalidated runtimes; annotation burden |
| Feasibility | 6 | **3.5** | -2.5 | ↓↓↓ Zero deliverables; prior conditions unmet |
| **Composite** | **5.75** | **4.1** | **-1.65** | ↓↓ Below comfortable continuation; sprint is bounded experiment |

### Process Notes

- **Vote: 2-1 for ABANDON** (Auditor at 3.8, Skeptic at 3.3) vs. CONDITIONAL CONTINUE (Synthesizer at 5.6).
- Lead resolves to CONDITIONAL CONTINUE because the 4-week sprint is cheap, the gating questions are testable, and the salvage floor is high.
- **Prior ideation gate conditions: 3/4 unmet** (BC2 coverage measurement, BC3 venue decision, BC4 external benchmark validation). This execution track record informs low confidence in sprint commitments.
- **Independent verifier: APPROVED WITH MODIFICATIONS.** All six modifications incorporated (CPU score correction, asymmetric cascade, prior-condition documentation, C1 criterion sharpening, C2 MUST/SHOULD split, hard deadline).
- **The 4-week deadline is non-negotiable.** Continuation drift is the primary governance risk at this composite level. If deliverables are not ready at deadline, the failure cascade triggers automatically.

### The Diamond

The genuine diamond is the **obstruction certificate**: no LLM, no SymPy script, no dynamic tool, no human heuristic can tell you "this code is *unfixably* wrong with this algorithmic structure — restructure." If T2 works, even as brute-force for small k and p, this is a qualitatively new capability in scientific computing.

The second diamond is the **paradigm itself**. "Physics-aware program analysis" as a named research direction, with ConservationLint as the first instance, could inspire follow-on work: momentum-aware, charge-aware, entropy-aware program analysis. The paper's lasting impact may be the direction it opens, not the tool it builds.

Whether these diamonds can be extracted from the rough — that's what the 4-week sprint will determine.

---

*Post-theory community expert verification. 3-expert team (Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer) + independent verifier signoff. Evidence-based scoring against theory_bytes=0, impl_loc=0, approach.json (34KB). All scores grounded in documentary evidence from State.json, approach.json, final_approach.md, verification_signoff.md, depth_check.md, crystallized_problem.md, and approach_debate.md.*
