# Math Specification Verification Report

**Reviewer**: Math Specification Lead
**Artifact**: crystallized_problem.md — M1–M5 math contributions
**Reference**: math_spec.md (original analysis)

---

## 1. Grade Fidelity Check

| Contribution | Crystallized Grade | Original Grade | Match? |
|---|---|---|---|
| M1: Spatial EC Formalization | C+ | C+ (§1.4) | ✓ Faithful |
| M2: Geometric Consistency Pruning | B− | B− (§3.4, Thm 3.2) | ✓ Faithful |
| M3: Spatial Tractability Theorem | A−conditional | Not graded — was a *recommendation* | ⚠ See §3 |
| M4: Spatial Type System Soundness | B | Not analyzed in original | ⚠ See §4 |
| M5: End-to-End Compiler Correctness | B− | B− (§5.5) | ✓ Faithful |

M1, M2, M5 are accurately described and honestly graded. No inflation.

## 2. Inflation Analysis

The crystallized problem restructured the original 5-component breakdown (which
included C-grade R-tree Automata and C-grade Incremental Compilation) into 5
*math* contributions by dropping the two C-grade engineering items and adding M3
and M4. This is legitimate curation — the dropped items were correctly identified
as routine — but it raises the apparent mathematical profile. The original
concluded "C-grade mathematical content"; the crystallized version now reads
closer to B/B+ average. This is partially earned (M3/M4 add real content) but
the shift should be acknowledged.

The EC Axiom Engine at 70% Novel in the subsystem table is in tension with the
C+ math grade for M1. If "novel" means "no off-the-shelf library" rather than
"mathematically new," the number is defensible but the distinction should be
clearer.

## 3. M3 (A−conditional): Assessment

The original math_spec.md mentioned this only as **Recommendation #3**: "Identify
a decidability boundary... if the spatial structure yields a tractability result,
that would be genuinely interesting." It was a *suggested future direction*, not
an existing contribution.

The crystallized problem promotes this to a full contribution (M3) with grade
A−conditional. Concerns:

- **The theorem is openly conjectured**, not proved. Grading a conjecture is
  inherently speculative.
- **The "conditional" qualifier is honest** — risks are clearly stated (treewidth
  may not be bounded, 6–12 months, fallback to M2 if it fails).
- **A− is generous.** The original called it "genuinely interesting" — that maps
  to B+ if proved, not A−. The claim that it "would surprise experts" rests on
  the untested assertion that XR interference graphs have bounded treewidth.
  Treewidth-based model checking is well-studied (Berwanger et al., Obdržálek);
  the novelty is the *domain identification*, not the technique.
- **Verdict**: Downgrade to **B+conditional**. If the treewidth bound is
  empirically validated AND the compositional algorithm handles spatial separators
  correctly, B+ is earned. A− requires a deeper structural insight.

## 4. M4 (Spatial Type System Soundness, Grade B): Assessment

The original math_spec.md did not analyze type system soundness as a separate
contribution. Its absence suggests the original reviewer did not consider it
load-bearing math.

The crystallized problem's description is substantive: soundness for the convex-
polytope fragment reduces to LP feasibility (routine), but characterizing the
decidability boundary for bounded-depth CSG is NP-complete and requires a real
PL × computational-geometry argument.

**Verdict**: B is **marginally earned** — the LP-feasibility core is standard
(B−), but the CSG decidability boundary characterization adds genuine interest.
The grade is defensible but at the upper end of what the content supports.

## 5. Missing Dependencies and Gaps

- **M4 → M5 dependency**: M5 (end-to-end correctness) should compose M4 (type
  soundness) to guarantee compiled guards are satisfiable. This dependency is
  implicit but not explicitly stated in the crystallized problem's M5 description.
  Minor gap.
- **M3 → M2 dependency**: Correctly implied (M3's compositional algorithm uses
  geometric consistency at tree-decomposition separators).
- **No gap in M1 → M5 chain**: The Lipschitz bound threading is well-described.

## 6. Novel% vs. Mathematical Novelty

| Subsystem | Novel% | Math Grade | Alignment |
|---|---|---|---|
| Spatial-Temporal Type System | 60% | B (M4) | ✓ Reasonable |
| Event Calculus Axiom Engine | 70% | C+ (M1) | ⚠ Inflated — 70% implies high novelty but math is elementary |
| Reachability Checker | 65% | B−/A−cond (M2/M3) | ✓ If M3 succeeds |
| Deadlock Detector | 55% | B− (M2 applied) | ✓ Reasonable |

The EC Axiom Engine Novel% should be ~50% to align with C+ math content, or
the table should clarify that Novel% measures *implementation* novelty (no
reusable library exists) rather than *mathematical* novelty.

---

## Summary of Issues

1. M3 graded A−conditional; should be **B+conditional** (technique is known,
   novelty is domain identification only).
2. EC Axiom Engine Novel% (70%) overstates mathematical novelty vs. C+ grade.
3. M4→M5 dependency not explicitly stated.
4. Overall math profile reads higher than original assessment's "C-grade
   mathematical content" conclusion — partially earned, partially structural.

---

**SIGNOFF: APPROVE WITH CONDITIONS**

Conditions:
1. Downgrade M3 from A−conditional to B+conditional (or provide a specific
   structural argument for why XR treewidth is bounded that goes beyond analogy).
2. Reduce EC Axiom Engine Novel% from 70% to ~50%, or add a footnote
   distinguishing implementation novelty from mathematical novelty.
3. Add explicit M4→M5 dependency in the M5 description (one sentence).

With these three changes, the mathematical claims are honest and well-calibrated.
