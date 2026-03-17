# Prior Art Verification Report

**Auditor**: Prior Art Auditor (independent verification)
**Date**: 2025-07-18
**Documents reviewed**: `crystallized_problem.md`, `prior_art_audit.md`

---

## 1. Novelty Claims vs. Prior Art Landscape

**Consistent.** The crystallized problem claims novelty in the *integration*—DSL + EC semantics + R-tree-backed automata + formal verification for XR—which aligns with the audit's "architectural synthesis" characterization. Individual component novelty is not overclaimed. The EC→automata compilation direction (reversing prior art) and R-tree-backed automata transitions are correctly identified as unprecedented.

**Minor tension**: The crystallized problem frames this as a "new programming model" paper (Halide/Cg/TVM analogy), while the audit characterizes novelty as "architectural innovation." These are compatible but the best-paper framing is more aggressive than the audit warrants. Acceptable if M3 delivers.

## 2. "First System To..." Claim

**Defensible.** The claim: "first system to give interaction choreographies a formal syntax, static semantics, and a compilation target that is both executable and verifiable." The audit confirms no existing system combines all four properties. The qualification "interaction choreographies" (not "spatial DSL" or "XR testing") is precise enough to withstand challenge. A reviewer could only attack this by showing a system that does *all four*—none identified in the audit.

## 3. iv4XR Comparison

**Fair and accurate.** The crystallized problem states iv4XR uses "agent-based exploration with runtime assertions—categorically different from compile-time guarantee." The audit confirms: agents discover states empirically, not exhaustively. The word "categorically" is justified—agent exploration vs. model checking is a genuine methodological divide.

**One nuance missing**: iv4XR is an EU Horizon 2020 project with follow-on work. The crystallized problem should be prepared for reviewers citing post-2022 iv4XR extensions. Not a blocking issue.

## 4. UPPAAL Comparison

**Fair but incomplete.** The crystallized problem correctly identifies: no spatial predicates, no XR integration, manual model construction. However, the audit flags active research on spatial model checking extensions to UPPAAL. The crystallized problem does not acknowledge this risk. A reviewer working on TOPAAL or spatial UPPAAL extensions could challenge the "no spatial predicates" claim as outdated.

**Recommendation**: Add a sentence acknowledging that UPPAAL spatial extensions are emerging but remain research prototypes without XR-domain integration.

## 5. SpatiaLang Differentiation

**INSUFFICIENTLY ADDRESSED.** This is the most significant gap. The audit explicitly identified SpatiaLang as the closest prior art for the DSL aspect and the weakest novelty claim, recommending clear differentiation. The crystallized problem **never mentions SpatiaLang by name**. A reviewer familiar with SpatiaLang could argue the DSL contribution is incremental without seeing how Choreo differs.

**Recommendation**: The paper must explicitly compare against SpatiaLang on four axes: (1) no temporal reasoning in SpatiaLang, (2) no formal verification, (3) generates imperative code vs. verifiable automata, (4) no headless execution. This is a mandatory related-work item.

## 6. Prior Art Gaps / Reviewer Attack Vectors

Three omissions from the crystallized problem that could become attack vectors:

**(a) Multiparty Session Types (MPST).** The audit identifies MPST as "the strongest theoretical precedent" for choreography verification. A PL reviewer will immediately ask: "How is this not session types with spatial predicates?" The crystallized problem must preempt this by explaining that MPST are message-based (channel communication) while Choreo choreographies are spatially-grounded (geometric predicate evaluation). The type-theoretic machinery differs fundamentally.

**(b) Scenic (UC Berkeley, PLDI 2019).** A compiler for spatial-temporal scenario specifications with formal verification—in the autonomous driving domain. A reviewer could argue Choreo is "Scenic for XR." The differentiation (scene generation vs. interaction verification, no interactive multi-party choreographies in Scenic) must be stated explicitly.

**(c) Complex Event Processing (CEP).** Systems like Apache Flink + GeoT-Rex already do spatial-temporal event pattern matching. A systems reviewer could ask why Choreo isn't just a CEP engine with a type system. The answer—CEP lacks formal verification, is unidirectional stream processing, and has no interactive/bidirectional interaction semantics—needs to appear in related work.

## 7. Best-Paper Argument

**Conditionally holds.** The argument has two legs:

- **Theoretical leg (M3)**: The spatial tractability theorem is graded A−conditional with 6–12 months of risk. If M3 fails, the paper drops from "surprising theoretical result + working system" to "well-engineered system paper." Still strong, but the Halide/TVM analogy weakens—those papers had clean theoretical contributions.

- **Empirical leg (bug-finding)**: If Choreo finds ≥5 confirmed bugs in MRTK/Meta Interaction SDK, the practical impact argument is strong regardless of M3. This is the more reliable leg.

The best-paper argument is **viable but front-loads risk on M3 and evaluation delivery**. Without either, it becomes a solid systems paper but not best-paper caliber.

---

## Summary of Required Actions

| # | Issue | Severity |
|---|---|---|
| 1 | SpatiaLang not mentioned; must be explicitly compared | **High** |
| 2 | MPST not addressed; PL reviewers will attack | **High** |
| 3 | Scenic not addressed; PLDI reviewers will notice | **Medium** |
| 4 | CEP systems not addressed; systems reviewers will ask | **Medium** |
| 5 | UPPAAL spatial extensions not acknowledged | **Low** |
| 6 | iv4XR post-2022 follow-on work not addressed | **Low** |

---

## SIGNOFF: APPROVE WITH CONDITIONS

**Conditions:**

1. The paper's related work section **must** explicitly compare against SpatiaLang, MPST (Scribble/Honda et al.), and Scenic. These are the three most likely reviewer attack vectors identified in the audit but absent from the crystallized problem.
2. The "first system to..." claim is defensible as stated but should be accompanied by a clear differentiation table (Choreo vs. SpatiaLang vs. iv4XR vs. UPPAAL vs. Scenic) in the paper itself.
3. The best-paper argument should not rely solely on M3; the evaluation bug-finding narrative is the more defensible primary argument.

The core novelty claims are sound and consistent with the prior art landscape. The integration is genuinely unprecedented. The conditions above concern **presentation completeness**, not **substantive novelty defects**.
