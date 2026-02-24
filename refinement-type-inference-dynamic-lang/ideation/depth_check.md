# Depth Check: Guard-Harvesting CEGAR for Automated Refinement Type Inference in Python

**Slug:** `guard-harvest-refinement-types-python`
**Verification method:** Three-reviewer adversarial team (Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer) with independent proposals, adversarial critiques, and synthesis.

---

## Scoring Summary

| Axis | Score | Verdict |
|------|-------|---------|
| 1. Extreme & Obvious Value | **5/10** | BELOW THRESHOLD — needs amendment |
| 2. Genuine Difficulty | **7/10** | MEETS THRESHOLD (barely) |
| 3. Best-Paper Potential | **5/10** | BELOW THRESHOLD — needs amendment |
| 4. Laptop CPU + No Humans | **7/10** | MEETS THRESHOLD (barely) |
| **Total** | **24/40** | **AMENDMENTS REQUIRED** |

---

## Axis 1: EXTREME AND OBVIOUS VALUE — 5/10

### What the team agrees on

The value proposition has a real kernel: Python developers hit array-out-of-bounds, division-by-zero, and None-dereference bugs that mypy/Pyright cannot catch. Automated refinement type inference with zero annotations is a genuine contribution.

### Why the score is low

**Demand signal is weak.** The target audience is the intersection of three small sets: (1) Python developers, (2) who use static analysis at all, (3) who need value-level properties beyond what Pyright already provides. The Auditor's challenge is devastating: mypy adoption is still incomplete (<30% of top PyPI packages ship `py.typed`). Asking developers to care about *refinement* types when they haven't adopted *basic* types is asking them to skip a grade they haven't passed.

**Pyright already covers ~95% of type narrowing from runtime guards.** The proposal's own prior-art critique admits this. The incremental value is confined to arithmetic refinements (bounds, division-by-zero) — a much narrower wedge than the proposal presents.

**ML shape-checking is handled by runtime solutions.** The ecosystem has overwhelmingly chosen runtime shape checking (`jaxtyping`, `beartype`, `einops`, `torchtyping`). The predicate language P (`QF_UFLIA` with `len`, comparisons) cannot express tensor shape constraints like `[batch, seq_len, d_model]` or broadcasting rules. The ML value proposition is therefore speculative.

**Regulated-software argument is unsupported.** No evidence is provided that any fintech/healthcare Python shop has requested refinement-type certificates. The SOC 2 / HIPAA argument is plausible but entirely speculative.

### What would raise the score

- Reframe away from "refinement types" in the value pitch. Lead with: *"Automatically proves your array accesses are in-bounds, your divisions are non-zero, and your None checks are sufficient — zero annotations required."*
- Anchor to a single killer scenario with concrete numbers from the evaluation.
- Position against mypy adoption gap as a feature: the tool requires no type annotations, so it works on the 60% of codebases that haven't adopted mypy.
- Drop the ML tensor shape narrative unless P is extended to handle it.

---

## Axis 2: GENUINE DIFFICULTY AS A SOFTWARE ARTIFACT — 7/10

### What the team agrees on

The composition challenge is real. Combining abstract interpretation, CEGAR, Liquid-style Horn clause solving, and predicate-sensitive incrementality into a working system for Python is genuinely hard engineering.

### The LoC estimate is inflated

The Skeptic's analysis is persuasive: the 133K LoC figure is inflated 1.5–2× over what's genuinely irreducible.

- **Python SSA (22K):** Python's `ast` module handles parsing. SSA construction + predicate-preserving IR is ~8-12K, not 22K.
- **Abstract Domains (26K):** Individual domains could leverage APRON wrappers; the genuinely novel part is the reduced product (~12-18K realistic).
- **Stdlib Models (12K):** 150 functions at 80 LoC each is data entry, not research. Paper needs ~30 core functions (~3K).
- **SMT Encoding (14K):** Z3 Python bindings handle most of the heavy lifting (~8K realistic).

**Realistic paper-scope estimate: 55-70K LoC, not 133K.** This is still a major engineering effort — solidly in "senior PhD thesis" territory.

### The reduced product is the credible crux

All three reviewers agree: 6 pairwise reductions with bidirectional information flow, where one incorrect reduction causes unsoundness or precision collapse, is genuinely hard. The "30-40% of debugging time" estimate is credible. This is the engineering difficulty that justifies the score.

### Why not higher

Each individual component (SSA, domains, CEGAR, SMT, incrementality) has textbook treatments. The difficulty is in composition, not in any single novel algorithm. This is "compose known techniques carefully" difficulty, not "invent new paradigm" difficulty.

---

## Axis 3: BEST-PAPER POTENTIAL — 5/10

### What the team agrees on

The proposal is a solid PLDI/OOPSLA *acceptance* candidate if built and evaluated well, but the case for *best paper* is weak.

### The theorems are routine extensions

The proposal's own mathematical specification rates them honestly:
- **Theorem A (decidability):** "Straightforward Skolemization argument. 2-3 pages." The predicate language P is deliberately designed to be decidable; proving decidability of a language you engineered to be decidable is near-tautological.
- **Theorem B (incrementality):** "Moderate difficulty, 3-4 pages." Extension of semi-naïve Datalog with stratified negation. This is the strongest theorem — genuinely novel output-sensitive bounds for refinement type inference — but not a breakthrough.
- **Theorem C (convergence):** "Straightforward + 1 subtle point. 2 pages." Finite-height lattice + strict progress = termination. Trivially true for any finite predicate language. The conditional nature (convergence only when interpolants project onto P) further weakens it.

No theorem individually merits publication at a theory venue. Their value is architectural.

### "Composing known techniques for a new domain" is not best-paper material

Best papers require at least one of: (a) a surprising theoretical result, (b) a paradigm shift, or (c) overwhelming empirical results. The proposal has (b.5) — the guard-harvesting insight is genuinely interesting but incremental over occurrence typing — and aspires to (c), but has zero implementation results (`code_loc: 0`).

### What would raise the score

- **Elevate Theorem B** as the primary theoretical contribution with expanded treatment (5-6 pages, worked invalidation cascade examples).
- **Lead the abstract with the counterintuitive insight:** "Dynamic languages are paradoxically better suited to automated refinement type inference than static ones, because programmers write explicit guards."
- **Add an empirical impossibility result:** Show that no predicate language strictly smaller than P achieves >70% coverage, giving P a sense of inevitability.
- **Achieve overwhelming empirical results.** If the system finds 15+ real bugs in top-50 PyPI packages that all 5 comparators miss, with <5% FP, the paper writes itself regardless of theoretical novelty.

---

## Axis 4: LAPTOP CPU + NO HUMANS — 7/10

### Architecture is fundamentally CPU-friendly

The design is correct: SMT queries over QF_UFLIA with 5-20 predicates are fast, incremental analysis limits working set, embarrassing parallelism across independent functions works on 8-core laptops. No GPU dependency, no model loading, no training.

### Performance targets are optimistic

- **Z3 query latency:** Plausible for 80% of functions, but tail latency on complex functions (100+ lines, 10+ branch points, large key sets K) could exceed 10 seconds per CEGAR iteration. The Skeptic's math: 5 iterations × 1-10s per SMT query = 5-50s per function, before abstract interpretation.
- **50K LoC in 30 minutes:** Mopsa analyzes ~10K LoC in minutes for simpler abstract interpretation (no CEGAR). Refinement analysis is strictly harder. This target needs empirical validation.
- **90th-percentile incremental <60s:** 10% of commits taking >60s is a CI-gating dealbreaker.

### Bug confirmation methodology has a flaw

The Skeptic identifies a critical issue: differential testing ("warning disappears after fix") conflates three cases: (1) fix eliminated the real bug, (2) fix changed code enough to alter analysis precision, (3) fix moved code outside analyzable subset. Cases 2 and 3 inflate bug counts and deflate FP rate. Full automation of bug confirmation may require a more sophisticated methodology (e.g., comparing against a manually validated subset).

### What would solidify the score

- Add a "graceful degradation budget" table showing expected function coverage at 5s, 10s, 30s, and 60s budgets.
- Acknowledge that bug confirmation requires a small manually validated holdout set (10-20 cases) to calibrate the automated method.
- Report incremental latency as a distribution, not just median and 90th percentile.

---

## Axis 5: Fatal Flaws

### Fatal Flaw 1: CEGAR Convergence Is Unresolved (HIGH RISK)

All three reviewers flag this. Theorem C guarantees termination but not *useful* termination — a system that marks 50% of functions "unresolvable" satisfies the theorem but is useless. The convergence theorem is conditional on interpolant projection onto P, and no evidence exists that this condition holds frequently in practice. The 70% coverage target is a guess.

**Mitigation:** The proposal's honest acknowledgment and explicit measurement plan partially addresses this. But this remains the #1 existential risk.

### Fatal Flaw 2: False-Positive Rate Is Almost Certainly Underestimated (HIGH RISK)

The proposal claims <5% FP. The Skeptic argues 15-30% is more realistic, citing: Liquid Haskell with programmer annotations on a pure typed language still has precision issues; Infer achieves 15-25% FP on Java/C++ with more type information; every `*args`, unmodeled stdlib call, and `try/except` is an over-approximation. The proposal's own internal documents show inconsistency (<2% in one framing, <5% in another, <10-15% in a third).

**Mitigation:** Report as precision-recall curve across confidence tiers. Commit to a single threshold and measure honestly.

### Fatal Flaw 3: The Guard Harvesting Insight May Not Generalize (MEDIUM RISK)

The insight depends on Python developers writing explicit guards. But:
- Many codebases use `try/except` (EAFP idiom) instead of guards
- `assert` statements are stripped in production
- Decorator-based validation (Pydantic, attrs) produces guards outside function bodies
- `*args`/`**kwargs`, closures, generators, async/await, metaclasses, `eval` — all outside the "modeled semantics"

The Skeptic estimates 30-40% real coverage, not 70%. The 85% guard-coverage claim for predicate language P needs empirical validation.

### Fatal Flaw 4: Scope-to-Feasibility Mismatch (MEDIUM RISK)

133K LoC is a 2-3 year team effort. Realistic paper scope is ~60K LoC (as the Synthesizer independently concludes). The proposal should present 60K as the paper contribution and 133K as the long-term vision.

---

## Amendments Required

Since Axes 1 (Value: 5/10) and 3 (Best-Paper: 5/10) score below 7, **amendments to the problem statement are required.** An amended version has been written to `ideation/crystallized_problem.md`.

### Key amendments made:

1. **Value reframing:** Lead with concrete bug classes, not "refinement types." Drop ML tensor shape narrative. Position against mypy adoption gap.
2. **Scope reduction:** 60K LoC paper scope, 133K long-term vision. Three domains (intervals, nullity, type-tag), not six.
3. **Theorem elevation:** Theorem B (incrementality) promoted to primary contribution. Theorem C reframed with honest conditional nature.
4. **Honest metrics:** FP rate reported as precision-recall curve. Coverage reported on analyzable subset. Three-tier reporting (analyzable FP, analyzable fraction, whole-corpus FP).
5. **Best-paper argument sharpened:** Lead with counterintuitive insight. Emphasize theory-practice unity through the shared lattice L. Add empirical coverage characterization of P as independent contribution.
6. **LoC estimate corrected:** Novel vs. infrastructure LoC separated. Reuse opportunities enumerated.

---

*Verification team: Independent Auditor ✓ | Fail-Fast Skeptic ✓ | Scavenging Synthesizer ✓*
*Review signoff: All three reviewers concur on scores and amendments. Auditor-Skeptic disagreement on Difficulty (Auditor: 6, Skeptic: implicit 5, Synthesizer: 7) resolved at 7 — the composition challenge is real even if individual components are textbook. Auditor-Skeptic agreement on Value (4-5) and Best-Paper (3-5) drove the amendment requirement.*
