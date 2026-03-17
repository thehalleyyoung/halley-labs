# Depth Check: Penumbra (fp-diagnosis-repair-engine)

**Evaluator:** Best-paper committee chair (3-expert team: Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer)  
**Method:** Independent proposals → adversarial critiques → synthesis of strongest elements  
**Date:** 2026-03-08

---

## Scores

| Axis | Score | Verdict |
|------|-------|---------|
| 1. Extreme & Obvious Value | **5/10** | Real but niche; "desperate need" overstated |
| 2. Genuine Difficulty | **7/10** | Engineering-hard at real breadth; not algorithm-hard |
| 3. Best-Paper Potential | **5/10** | Publishable at strong venue; needs reframing + showstopper result for awards |
| 4. Laptop CPU + No Humans | **8/10** | Feasible with honest scope restriction |
| **Composite** | **6.25/10** | **CONDITIONAL CONTINUE** |

**Decision: CONDITIONAL CONTINUE — 2-1 (Skeptic dissents at 4.3/10, recommends CONDITIONAL REJECT without proof of T2)**

---

## Axis 1: EXTREME AND OBVIOUS VALUE — 5/10

### The problem is real.
- SciPy's GitHub tracker confirms ~40+ precision-related issues (`expm` conditioning, `logsumexp` underflow/overflow, `betainc` accuracy). These are not hypothetical.
- No existing Python-native tool fills this niche. Verificarlo requires LLVM-level instrumentation (C/Fortran only). Herbie handles single expressions, not pipelines. There is genuinely no `pip install X` solution for FP pipeline diagnosis.
- Scientists debug FP issues by ad-hoc intuition: days per bug, unreliable, non-transferable. Library maintainers waste real human hours on diagnosis.

### But "desperate need" is overstated.
- ~40 documented SciPy issues represent <0.3% of SciPy's tracker. The "tip of the iceberg" claim is asserted without evidence.
- If 2–8 ulp pipeline errors "go unnoticed," by definition they aren't causing pain. The proposal must demonstrate that hidden errors actually change scientific conclusions — it never does.
- Most scientists use float64 and it's good enough. The population who (a) hits FP errors, (b) knows they hit them, (c) can't solve them with `mpmath`/`np.longdouble`, and (d) would adopt a new instrumentation tool is small.
- **The LLM counter:** In 2026, a scientist suspecting a precision problem can get a reasonable qualitative diagnosis from an LLM in seconds. Penumbra's unique value must rest on *quantified, automated tracing* — something LLMs cannot do — but the proposal buries this under the less differentiated diagnosis/repair narrative.
- **Honest audience:** Library developers fixing known precision bugs and researchers in extreme-precision domains. Not "all scientific Python users."

### What would raise this to 7:
Demonstrate with concrete data (not anecdote) that FP bugs cause wrong scientific conclusions in published papers. Find a real, previously-unknown SciPy bug and get a PR merged. Narrow the value claim to library developers and own it.

---

## Axis 2: GENUINE DIFFICULTY — 7/10

### LoC audit (realistic range: 51–87K):

| Component | Claimed | Realistic | Novel |
|-----------|---------|-----------|-------|
| Shadow Instrumentation | 12–18K | 8–14K | ~5K |
| Multi-Precision Replay | 6–10K | 5–8K | ~3K |
| EAG Builder | 8–12K | 4–7K | 4–7K |
| Diagnosis Engine | 8–12K | 3–6K | 3–6K |
| Repair Synthesizer | 10–15K | 8–12K | ~5K |
| Certification Engine | 4–7K | 3–5K | ~2K |
| Trace/Storage/Rewriting | 6–10K | 5–9K | ~3K |
| Bench/CLI/Tests | 21–35K | 15–26K | 0 |
| **TOTAL** | **75–119K** | **51–87K** | **~25–31K novel** |

### What's genuinely hard:
- **MPFR replay of 100+ ufuncs:** Faithfully reproducing NumPy's exact reduction order, type promotion rules, and special-value handling (inf/NaN/denormals) at higher precision. Each ufunc needs individual attention. This is tedious-hard at real depth.
- **System integration:** Getting shadow instrumentation, MPFR replay, streaming EAG construction, diagnosis, and repair to work together on real SciPy code. Verificarlo took a multi-year research team.
- **Rust↔Python (PyO3):** The hot path (per-element MPFR arithmetic) demands Rust for performance. Pure Python with gmpy2 would be 100×+ slower for large arrays. The Rust choice is defensible.

### What's less hard than claimed:
- **EAG Builder:** A streaming DAG with weighted edges from first-order finite differences. Well-understood data structure. The name "Error Amplification Graph" is marketing for a dependency graph with sensitivity weights — essentially automatic differentiation of error propagation.
- **Taxonomic Diagnosis:** Five classifiers operating on graph subgraphs. Each is essentially a pattern matcher with threshold logic on condition numbers and operand magnitudes. The formal definition adds rigor but the underlying logic is textbook (Higham 2002).
- **Repair patterns:** 30 algebraic rewrites are a lookup table, not a synthesizer. The mixed-precision "universal fallback" means the real repair strategy often reduces to "promote to higher precision."

### The difficulty is primarily engineering-breadth, not algorithmic-depth.
This is still legitimate difficulty — but the proposal should be honest about the nature of the challenge.

### What would raise this to 8+:
Prove T2 (establishing genuine mathematical depth) or replace it with a provably correct decomposition theorem. Add non-trivial repair synthesis (e.g., e-graph equality saturation to *discover* rewrites rather than selecting from a fixed library).

---

## Axis 3: BEST-PAPER POTENTIAL — 5/10

### Venue analysis:
- **SC:** Best fit. Values practical impact on real scientific codes. If the evaluation demonstrates 10× error reduction on real libraries, SC is most receptive.
- **OOPSLA:** Possible if EAG is developed as a program analysis contribution.
- **ASPLOS:** Possible but must demonstrate comparable scale leap to Satire.
- **PLDI:** Unlikely. PLDI values deep formal language-theoretic contributions. The taxonomy is not deep enough.

### The EAG reframing is the key strategic insight.
The Scavenging Synthesizer identified the buried crown jewel: **the EAG is a novel program representation** — the first reified causal graph of error flow that supports graph algorithms. If reframed as the primary contribution (analogous to how PDGs enabled slicing, SSA enabled optimization, e-graphs enabled equality saturation), the paper becomes a *foundations contribution*, not a tool paper. The diagnosis taxonomy and repair synthesizer become *applications* demonstrating the representation's utility.

**Recommended title:** *Error Amplification Graphs: Causal Diagnosis of Floating-Point Error in Scientific Pipelines*

### What holds best-paper potential back:
1. **No provable core theorem.** T1 is a standard first-order error bound. T2 is a conjecture. T3 is exhaustive case analysis. T4 is empirical. Best papers need a crisp theorem with surprising implications.
2. **Fluctuat comparison is misleading.** Fluctuat already performs error decomposition via zonotopes/affine arithmetic with *formally sound* bounds. Marking it "partial" for diagnosis is indefensible. Reviewers familiar with Fluctuat will catch this immediately.
3. **Framing diffusion.** Too many contributions compete for the headline: EAG, diagnosis, repair, decomposition, certification. A best paper needs one crystalline idea.

### Comparison to caliber papers:
- **Herbie (PLDI'15 Distinguished):** Genuinely new synthesis technique (equality saturation over FP). Novel algorithm + surprising effectiveness.
- **Satire (ASPLOS'23):** Shadow-value analysis at unprecedented scale. Novel systems contribution at scale never before achieved.
- **Penumbra's pitch:** "We connect detection to repair via diagnosis." This is a workflow contribution — valuable, but incremental once Fluctuat's capabilities are honestly compared. The EAG reframing elevates it to "a new program representation for FP error" — significantly stronger.

### What would raise this to 7+:
(a) Reframe around the EAG as primary contribution. (b) Find and fix a real, previously-unknown bug in a major library (the "showstopper result"). (c) Prove T2 for a restricted class of EAGs (series-parallel graphs). (d) Cut repair from the headline — make the paper "EAG + Diagnosis" with repair as a demonstration.

---

## Axis 4: LAPTOP CPU + NO HUMANS — 8/10

### Feasible with honest scope:
- **CPU-bound by nature:** MPFR is sequential per-operation, EAGs are sparse irregular graphs, diagnosis is rule-based, repair is constraint solving. No GPU benefit.
- **Memory:** Shadow memory ~24N bytes per N-element array. 32GB handles ~1B elements for shadow arrays. However, EAG node metadata for dense matrix operations (e.g., 1000×1000 matrix multiply → ~10⁹ nodes × 64 bytes = 64GB) can exceed 32GB without aggressive summarization/streaming.
- **Runtime:** 8–24 hours realistic for scaled-down evaluation (small matrices, FPBench, curated SciPy functions). Full suite with large-matrix targets may push to 48 hours.
- **Ground truth:** Automated mining from GitHub issues is feasible for well-documented bugs. Semi-synthetic perturbation benchmarks supplement real bugs. FPBench adds standardized cases.

### The one concern:
**Ground-truth methodology for diagnosis accuracy.** The 85% target has no clear ground-truth basis for pipeline-level bugs. ~20–40 GitHub issues is insufficient for statistically meaningful accuracy claims. Confidence intervals must be reported.

### What would raise this to 9:
Docker/Nix environment with all dependencies pre-built. Drop FEniCS/Firedrake (C under the hood anyway). Restrict to pure-Python targets where instrumentation is honest.

---

## Axis 5: Fatal Flaws

### SERIOUS FLAW #1: `__array_ufunc__` Cannot Intercept LAPACK/BLAS ⚠️

**Severity: SERIOUS (project-threatening if unaddressed)**

`__array_ufunc__` intercepts only NumPy ufuncs. `__array_function__` intercepts some top-level functions but NOT SciPy's compiled routines. The proposal's marquee targets — `scipy.linalg.expm`, covariance inversion, stiffness-matrix assembly, PCA — ALL dispatch to compiled LAPACK/BLAS via F2PY/Cython wrappers. Coverage estimate: ~20–30% of operations in a typical scientific pipeline.

**This undermines the "pipeline-level" claim.** The tool can trace element-wise operations but is blind to the linear algebra core where the most consequential FP errors occur.

**Required amendment:** (a) Explicitly restrict scope to operations reachable through Python dispatch. (b) Treat LAPACK calls as black boxes with monkey-patched input/output error comparison. (c) Drop FEniCS/Firedrake from evaluation targets — their stiffness-matrix assembly happens in C/Fortran (PETSc) and is unreachable. (d) Add honest "instrumentation coverage" metric to evaluation.

### SERIOUS FLAW #2: T2 (EAG Decomposition Conjecture) as Central Contribution

**Severity: SERIOUS (submission-killing if central)**

A paper whose "central theoretical contribution" is a conjecture that might be false is vulnerable to immediate rejection. The additive/multiplicative mismatch (graphical model decomposition assumes additive potentials; FP error propagation is multiplicative) is a genuine mathematical obstacle. Furthermore, forward error propagation through a DAG is already polynomial — T2 addresses a problem that may not exist.

**Required amendment:** Demote T2 from "central contribution" (⭐) to "interesting empirical observation and open problem." The treewidth measurement across real codebases is genuinely novel data — present it as an empirical finding motivating future theory. Lead with EAG + Diagnosis as primary contributions. If T2 can be proved for a restricted class (e.g., series-parallel EAGs), promote it — but a conjecture cannot be the centerpiece.

### SURVIVABLE FLAW #3: Fluctuat Comparison Is Misleading

**Severity: SURVIVABLE with honest correction**

Fluctuat already performs error decomposition via zonotopes/affine arithmetic with formally sound bounds, attributing error to individual operations. Marking Fluctuat as "partial" for diagnosis is indefensible. Fluctuat's approach is *static* (all inputs) with *stronger soundness guarantees* than Penumbra's *dynamic* analysis (specific traces).

**Required amendment:** Correct the comparison table. Honest differentiation: "Fluctuat diagnoses but doesn't repair, targets C not Python, and doesn't handle pipelines at the dynamic execution level." The EAG-centered reframing sidesteps the direct comparison by positioning Penumbra as a different *kind* of analysis (graph-algorithmic framework vs. abstract-interpretation framework).

### SURVIVABLE FLAW #4: Evaluation Methodology Gaps

**Severity: SURVIVABLE with fixes**

- ~20–40 ground-truth labels is insufficient for statistically meaningful 85% accuracy claims.
- Semi-synthetic benchmarks are necessary but create circularity risk (tuned to find injected bugs).
- FPBench is single-expression, contradicting the pipeline-level pitch.
- No held-out evaluation methodology specified.

**Required amendment:** Report sample sizes and confidence intervals on all metrics. Find ≥5 real pipeline-level bugs that existing tools miss. Separate training/validation from held-out evaluation. Frame semi-synthetic benchmarks honestly as "fault injection" — standard methodology in reliability engineering.

### NON-FATAL CONCERNS:

- **T1 soundness fails for ill-conditioned targets:** First-order assumption ε·n·max(Lᵢ) ≪ 1 breaks down when condition numbers are 10⁸–10¹⁶. Acknowledge as scope limitation.
- **Taxonomy restricted to first-order patterns:** Multi-hop error amplification, emergent pipeline patterns, and interaction effects are outside scope. Acknowledge explicitly as v1 limitation.
- **Repair correctness:** Algebraic rewrites change numerical behavior. Certification validates error reduction but not functional correctness. Add a note about preservation of algorithmic intent.
- **LoC inflation:** 75–119K headline is 2–3× padded over realistic 51–87K. State honest estimates.

---

## Amendments Applied

Based on the three-expert evaluation and cross-critique, the following amendments are REQUIRED before continuing:

1. **A1: Demote T2 from central contribution** to open problem with empirical treewidth data. Lead with EAG + Diagnosis.
2. **A2: Address LAPACK interception gap** with black-box wrapping strategy. Drop FEniCS/Firedrake. Add instrumentation coverage metric.
3. **A3: Correct Fluctuat comparison** to honestly represent its error decomposition capabilities.
4. **A4: Reframe around EAG** as novel program representation. Title: *Error Amplification Graphs: Causal Diagnosis of Floating-Point Error in Scientific Pipelines*.
5. **A5: Fix evaluation methodology** — confidence intervals, held-out evaluation, ≥5 real pipeline-level bugs, honest framing of fault-injection benchmarks.
6. **A6: Honest LoC estimates** — 51–87K realistic range; ~25–31K novel.
7. **A7: Acknowledge first-order limitations** of T1 soundness and T3 completeness explicitly as scope boundaries.
8. **A8: Narrow value claim** to library developers and precision-sensitive researchers. Drop "no numerical-analysis PhD required" aspiration.

---

## Binding Conditions for CONTINUE

| ID | Condition | Gate |
|----|-----------|------|
| BC1 | T2 demoted from central contribution; EAG + Diagnosis centered | Before theory stage |
| BC2 | LAPACK black-box strategy designed; FEniCS/Firedrake dropped | Before implementation |
| BC3 | Fluctuat comparison corrected in related work | Before theory stage |
| BC4 | ≥5 real pipeline-level bugs identified as evaluation targets | Week 4 kill gate |
| BC5 | Instrumentation coverage metric defined and measured | Before evaluation |
| BC6 | Ground-truth methodology with sample sizes and confidence intervals | Before evaluation |

---

## Probability Estimates

| Outcome | Probability |
|---------|-------------|
| Publication at strong venue (SC, OOPSLA, ISSTA, FSE) | 45–55% |
| Best-paper at SC/OOPSLA | 5–10% |
| Best-paper at PLDI/ASPLOS | 2–4% |
| Project abandoned (T2 false + insufficient real bugs) | 15–25% |
| Any publication (including workshop/shorter venues) | 65–75% |

---

## Expert Votes

| Expert | Vote | Score | Rationale |
|--------|------|-------|-----------|
| Independent Auditor | CONDITIONAL CONTINUE | 5.3/10 | Real problem, real difficulty, but LAPACK gap and T2 status are serious. Publishable at SC with amendments. |
| Fail-Fast Skeptic | CONDITIONAL CONTINUE (dissenting at 4.3/10) | 4.3/10 | T2 is near-fatal, evaluation is circular, diagnosis may be shallow. Survives only with radical scope reduction and honest framing. |
| Scavenging Synthesizer | CONDITIONAL CONTINUE | 7.3/10 | EAG is a genuine crown jewel. Reframing + showstopper result = strong paper. No fatal flaws with amendments. |
| **Chair (consensus)** | **CONDITIONAL CONTINUE** | **6.25/10** | Amendments required. Composite reflects that the idea space is strong (Synthesizer is right about the EAG) but execution risks are real (Skeptic and Auditor are right about T2, LAPACK, Fluctuat). |

---

## The Three Things That Would Change Everything

1. **Prove T2 for series-parallel EAGs.** Even a restricted proof transforms the theoretical contribution. Raises Difficulty to 8, Best-Paper to 6–7.

2. **Find a real, previously-unknown SciPy bug and get a PR merged.** A single concrete result — "Penumbra discovered bug X in `scipy.Y`, silent since 20XX, affecting Z downstream packages" — raises Value to 7 and Best-Paper to 6–7. This is the correct north star for the evaluation.

3. **Reframe around the EAG and honestly restrict scope.** Drop FEniCS/Firedrake, acknowledge LAPACK as black-box, lead with EAG as a program representation, demote T2 to an open question. Costs nothing on any axis and raises Best-Paper by 1–2 points through clarity alone.
