# Approach Debate: Perceptual Sonification Compiler

**Stage**: Ideation → Debate  
**Date**: 2026-03-08  
**Participants**: Math Depth Assessor, Difficulty Assessor, Adversarial Skeptic, Debate Coordinator  
**Input**: 3 approaches, 3 independent evaluations

---

## Part 1: Per-Approach Debate

---

### Approach A: Graded Comonadic Sonification Calculus

**Self-assessed scores**: V5 / D8 / P8 / F5 (composite 26)

#### Skeptic Attack

1. **The resource semiring almost certainly does not exist (70% failure).** The composition operator ⊕ includes an indicator function $\mathbb{1}[r_{2,b} > 0]$ that makes it depend on the *support* of the second operand, not just its magnitude. This creates a structural asymmetry that breaks associativity—not an epsilon issue, but a fundamental incompatibility between context-dependent masking and context-free algebraic operations. The conservative over-approximation fallback "is effectively an admission that this will fail" and produces a semiring so pessimistic it "rejects everything with more than 3 streams."

2. **The coeffect formalization is a career-scale problem crammed into 4 weeks.** Novel type theory *always* produces unexpected metathory problems. The Granule team, who invented this framework, takes months to formalize new resource algebras. Without a genuine graded type theory expert on the team, "these problems become project-killing surprises at week 3." The four-week kill gate leaves zero room for the iterative discovery that PL theory demands.

3. **The evaluation cannot distinguish A from B (85% probability).** Both type systems check identical constraints and would produce the same accept/reject decisions on the same inputs. "The graded type theory adds zero value over Approach B for LLM competition resilience. The value is in constraint checking, which is approach-independent." If an OOPSLA reviewer asks "Why did you need graded comonadic types?" the only answer is aesthetic preference.

#### Mathematician's Challenge

1. **Result A1 (resource semiring) downgraded A → B+.** The novelty is "known technique, novel domain"—instantiating existing graded type theory with psychoacoustic quantities. The framework (graded comonads, coeffect tracking) is established; only the instantiation is new. Probability of exact associativity proof: **25%**. Even the ε-approximate version (60% success) requires extending graded type theory metatheory for approximate semirings, which "has not been done" and is itself an open research problem.

2. **Result A2 (coeffect typing) downgraded A → B+.** "This is not a co-crown jewel—it's the predictable consequence of A1." The typing rule is standard graded comonadic typing from Orchard et al.; the novelty is entirely parasitic on A1. "Claiming two Grade A results where the second is mechanically derived from the first is inflation."

3. **Result A4 (monotonicity) downgraded B → C: tautology.** "If 'uses fewer resources' is defined via a pre-order where less energy → less masking, and discriminability is inversely related to masking, then monotonicity follows by definition." This "theorem" restates the definitions. The Math Assessor flags it as **ornamental**.

#### Difficulty Assessment

- **Score: 7/10** (down from self-assessed 8/10).
- The resource semiring design is "genuinely research-level"—the only frontier-pushing component across all three approaches. But the 40% fallback probability and "routine audio/compiler engineering" surrounding the crown jewel pull the score down.
- **Novel LoC: ~28K** (vs. claimed 48–60K, a ~50% inflation).
- **Person-months: 8–12** for a strong PL PhD student with audio background. The semiring alone could take 2–3 months of false starts.
- **Hidden hard problems**: (1) The soundness proof via logical relations over continuous domains is "easily the most labor-intensive formal task and it's barely discussed." (2) DSL design is "mentioned as routine but is notoriously fiddly." (3) Calibrating the resource budget ceiling r_max is a domain-knowledge problem hiding inside the PL artifact.

#### Defense

The critics are largely right about the risks—but they undervalue the ceiling. Three points in A's defense:

1. **The semiring question is a genuinely interesting open problem.** The Math Assessor acknowledges this: "the psychoacoustic resource semiring is a real open problem that could yield a genuinely interesting structure." Even at B+ grade, it's the deepest mathematical contribution across all three approaches. If a tight ε-approximate semiring works with non-vacuous ε, the paper introduces "approximate resource semirings for graded types"—a contribution no one in the PL community has attempted. The Skeptic's claim that "an ε-approximate semiring is not a semiring" is technically correct but ignores that *creating the theory of approximate graded types* would itself be novel.

2. **The "perception as resource" insight is not just marketing.** The Skeptic calls it a "leaky analogy," but analogies that are only approximately correct still have enormous PL value. Linear types model memory ownership imperfectly (aliasing, borrowing rules are approximations). Session types model protocol compliance imperfectly (real protocols have timeouts, failures). The question is whether the approximation is useful, and for simultaneous steady-state masking (which covers the alarm palette use case), the resource model is physically faithful.

3. **The best-paper ceiling is real.** Even the Skeptic concedes A's ceiling is "Distinguished paper" and B's is "Strong OOPSLA paper." If the team's risk tolerance favors moonshots and the week-4 empirical test is promising, the expected value calculation shifts: a 30% chance of a distinguished paper may outweigh a 75% chance of a solid paper, depending on career context.

#### Cross-Examiner Questions

1. **Can you demonstrate, within 2 weeks, that the associativity gap for realistic 4–8 stream configurations is <10% relative?** If the gap is >20%, the conservative over-approximation will reject most valid configurations, and no amount of theoretical ingenuity will save the practical story.

2. **Who on the team has experience with graded type theory metatheory?** The Skeptic's flaw A-5 is pointed: "approximately 10-20 people worldwide deeply understand the metatheory." If no team member is among them, the formalization risk is terminal.

3. **What empirical result from your evaluation would be *different* under Approach A vs. Approach B?** If no evaluation metric distinguishes the two, the graded comonadic machinery is pure overhead and you should merge into B.

#### Verdict: **KILL**

The convergence of evidence is overwhelming: the Math Assessor gives 25% probability of exact semiring proof, the Skeptic gives 70% failure probability for the entire approach, and the Difficulty Assessor notes the 40% fallback probability. The Skeptic's most devastating observation—that the evaluation cannot distinguish A from B—means even *success* doesn't clearly outperform the safer alternative. The best elements of A (the resource-tracking insight, the compositionality ambition) can be carried into B as a potential upgrade if the week-4 empirical semiring test is promising.

**Adjusted scores**: V5 / D7 / P6 / F4 (composite 22, down from 26)

---

### Approach B: Liquid Sonification — SMT-Backed Refinement Types

**Self-assessed scores**: V6 / D7 / P7 / F8 (composite 28)

#### Skeptic Attack

1. **The custom SMT theory solver is an entire research project (40% failure).** Building a correct, efficient, complete-enough theory solver for non-linear real arithmetic with transcendental functions is "the kind of task that takes SMT researchers years, not weeks." The piecewise-linear fallback derisks this substantially—but at the cost of the paper's second crown jewel (B2). "If you fall back to QF_LRA encoding, you lose a Grade A result and B's difficulty score drops from 7 to 5."

2. **Refinement types for non-local constraints is a known hard problem, not a novel solution (55%).** If non-locality is handled by brute-force conjunction of O(k²) SMT constraints, "a PL reviewer will say: 'This isn't a type system. You're generating an SMT query and checking satisfiability.'" The distinction between "refinement type system" and "SMT verification tool" hinges on whether the system provides compositional reasoning—and brute-force re-checking is not compositional.

3. **The "linter for AI-generated code" framing is wishful thinking (70%).** The ESLint analogy fails on every dimension: ESLint serves millions of developers, runs in milliseconds, and integrates with existing workflows. SoniType-B serves dozens, takes seconds, and integrates with nothing. "This is a supply-side fantasy until proven otherwise."

#### Mathematician's Challenge

1. **Result B1 (refinement type system) adjusted A → A-/B+.** The environment-dependent refinement predicates—where $\phi_{\text{mask}}(v, \Gamma)$ depends on the typing environment—are "genuinely novel but incremental." It's "the strongest 'new PL result' across all three approaches, which says more about the other approaches than about this one." The proof requires a novel lemma about "environment extension preserving satisfiability under masking monotonicity"—doable but non-trivial. **Probability of successful proof: 75%.**

2. **Result B2 (SMT solver soundness) downgraded A → B+.** "Every custom SMT theory solver does this. Calling it a 'co-crown jewel' is inflation." The δ-soundness framing is standard numerical analysis. "Presenting verified engineering as a Grade A theorem damages credibility with PL reviewers."

3. **Result B4 (OMT approximation) downgraded B → C+.** The authors themselves acknowledge this is "a standard branch-and-bound convergence result applied to the psychoacoustic domain." The Math Assessor recommends moving it to an appendix.

#### Difficulty Assessment

- **Score: 5/10** (down from self-assessed 7/10).
- "The refinement type framework is well-understood machinery applied to a new domain." The custom SMT theory is genuine engineering difficulty but "likely unnecessary" given the piecewise-linear fallback.
- **Novel LoC: ~29K** (vs. the approaches document's implicit ~48K+ claim).
- **Person-months: 5–8** for a strong PL PhD student familiar with SMT solvers.
- **Key insight**: "Once you accept the piecewise-linear approximation, most claimed hard problems dissolve." Non-local constraint checking is "computationally trivial at the relevant scale (k ≤ 16)."
- **Hidden hard problem**: The refinement type soundness proof chains through δ-soundness of the custom theory → propagation of δ into type soundness → showing δ is perceptually negligible. "This chain is non-trivial and not discussed."

#### Defense

B has the best risk-adjusted profile, and the critics' attacks are largely mitigable:

1. **The custom SMT theory is a red herring—and that's fine.** Both the Skeptic and Difficulty Assessor agree: commit to piecewise-linear encoding from day one. This doesn't gut the paper—it sharpens it. The contribution becomes "refinement types for psychoacoustic verification with environment-dependent predicates," which the Math Assessor calls "the strongest new PL result across all three approaches." Dropping the custom theory frees 6+ weeks for the actual type system and evaluation.

2. **The compositionality challenge can be turned into the contribution.** The Skeptic flags that brute-force conjunction isn't compositional. But the incremental checking result (B3)—showing that adding a stream triggers only O(k·B) re-checks, not O(k²·B)—IS a compositionality result. The Skeptic even suggests this reframe: "We show that psychoacoustic constraints have bounded propagation, enabling incremental composition." If elevated from a Grade B afterthought to the primary algorithmic contribution, this directly addresses the "is it really a type system?" objection.

3. **The evaluation is the strongest of all three approaches.** The Difficulty Assessor and the approaches document agree that B's evaluation plan—adversarial bug-finding, LLM baseline, cross-model validation, type-checking performance, linting mode, diagnostic quality—is "the most comprehensive." Excellent evaluation has won OOPSLA distinguished papers before (the paper doesn't need to be the deepest type theory; it needs to be the most convincing).

4. **B is the only approach with a credible fallback at every stage.** SMT solver too slow → piecewise-linear. Full type system too hard → verification tool. Incremental checking fails → brute-force conjunction (still works for k≤16). No single failure point is catastrophic.

#### Cross-Examiner Questions

1. **Can you demonstrate, in week 1, that Z3 type-checks an 8-stream configuration in <2 seconds with piecewise-linear encoding?** The Skeptic's narrative predicts 0.5s for k=8, 3s for k=12, 15s for k=16. If the 8-stream baseline exceeds 5s, the interactive-use story collapses.

2. **What specific result distinguishes your type system from "a conjunction of SMT queries"?** The Math Assessor's B1 assessment gives this a 75% probability of success. If the environment-extension lemma and bounded-propagation proof both go through, you have a genuine type-theoretic contribution. If only one goes through, you still have a publishable paper. If neither, you're "constraint checking with PL vocabulary"—the exact failure mode the depth check flagged.

3. **Who will actually use the perceptual linting mode?** The Skeptic says "no user research conducted." Can you identify 5 concrete alarm-design teams who would beta-test the tool? Without evidence of demand, the value proposition rests on the regulatory angle alone.

#### Verdict: **CONDITIONAL CONTINUE**

B survives with conditions that all three evaluators converge on:

1. Drop the custom SMT theory solver as a crown jewel. Commit to piecewise-linear encoding.
2. Elevate incremental composition (B3) to the primary algorithmic contribution with formal proof.
3. Budget 25–30% of engineering time for diagnostic extraction.
4. Run SMT performance benchmark in week 1 as a go/no-go gate.
5. Be honest about the contribution level: this is "refinement types applied to a novel domain with non-local challenges," not "novel type theory."

**Adjusted scores**: V5 / D5 / P6 / F8 (composite 24, down from 28)

---

### Approach C: SoniSynth — Psychoacoustic Program Synthesis

**Self-assessed scores**: V7 / D6 / P6 / F9 (composite 28)

#### Skeptic Attack

1. **This is not program synthesis—it's constraint solving with a nice API (80%).** "Program synthesis at OOPSLA means SyGuS, sketching, type-inhabitation, proof-search—techniques that generate code with control flow, recursion, higher-order functions. SoniSynth generates flat parameter vectors." Calling this "program synthesis" invites comparison with real synthesis work and comes up short. Any OOPSLA reviewer familiar with Rosette, Sketch, or Leon will see through the framing.

2. **The NP-completeness result is trivial and the approximation is weak (75%).** "Graph coloring reductions are the default 'this is NP-complete' proof for any constraint-satisfaction problem over discrete domains with pairwise constraints. This is textbook material." The greedy approximation guarantee has an unspecified α parameter—if α=10, the guarantee is "useless (1/10 of optimal)."

3. **Non-expert users won't touch this tool (85%).** "Data journalists use Observable and Highcharts. They're not going to install a Rust compiler." The specification language "requires understanding d', cognitive load budgets, and discriminability thresholds. Non-experts don't know these concepts. The tool shifts expertise from 'audio design' to 'constraint specification'—a lateral move, not a simplification."

#### Mathematician's Challenge

1. **Result C1 (realizability NP-completeness) downgraded A → C+.** "This is the most egregious grade inflation across all three approaches. An NP-completeness result via trivial reduction from graph coloring is a homework exercise, not a crown jewel." Any CSP with pairwise constraints over a finite domain is NP-complete by this same argument. The Math Assessor calls this "the most ornamental result in the entire document."

2. **Result C2 (greedy packing) downgraded A → B-.** The core algorithm (farthest-point insertion in a metric space) is "well-studied in computational geometry." The approximation guarantee from max-min distance maximization is textbook. Additionally, the proof relies on a hidden assumption: that perceptual distance satisfies the triangle inequality. "If perceptual distance doesn't satisfy the triangle inequality, the greedy guarantee breaks."

3. **Result C4 (information-theoretic optimality) flagged as likely false.** "Mutual information is not submodular in general" when sources interact via masking. "Adding stream C can reduce the information contributed by stream A (if C masks A)"—a masking-release effect that breaks diminishing returns. **Probability of successful submodularity proof: 30%.** If it fails, the paper loses its theoretical optimization contribution entirely.

#### Difficulty Assessment

- **Score: 4/10** (down from self-assessed 6/10).
- "The NP-completeness of realizability is clean but the reduction from graph coloring is straightforward." The greedy algorithm is "textbook." Multi-objective optimization is "an afternoon of coding with an existing library."
- **Novel LoC: ~21K** (lowest of all three approaches).
- **Person-months: 3–5.** "A student with constraint programming experience could move very fast."
- **Damning assessment**: "This is a well-motivated engineering project with a nice framing, not a research-difficulty challenge." And: "OOPSLA reviewers will likely see through the 'synthesis' framing and recognize it as constraint solving with a nice API."

#### Defense

C takes the heaviest beating of the three approaches, but two genuine strengths survive:

1. **The user-facing value proposition is the strongest.** Despite the critics' valid point that the "non-expert" audience is imagined, C's *idea*—"declare what you want to perceive, not how to produce it"—is the most compelling framing for the project. The SQL analogy is imperfect, but the direction is right. If absorbed into B as a specification frontend, this framing broadens B's appeal significantly.

2. **The specification language and realizability checking are useful engineering.** The Skeptic's recommendation to "merge with B" is telling—they're not saying C's work is worthless, they're saying it's the wrong *venue* as a standalone. The specification lattice (C3), stripped of its "theorem" pretension, provides genuinely useful UX: when a spec fails, tell the user what to relax. The greedy packing algorithm, honestly presented as engineering rather than theory, gives users a "generate a starting point" capability that neither A nor B offers.

3. **The LLM-backend positioning is defensible.** The Skeptic concedes that "GPT-5 calls SoniSynth API" is the "only defensible story." But that IS a good story. In a world where LLMs generate code, verified synthesis backends have real value. C-as-an-API-for-LLMs, backed by B's verification, is a product that could actually gain adoption.

#### Cross-Examiner Questions

1. **Can you name one result in your paper that an OOPSLA PL theorist would consider novel?** The Math Assessor's honest count: zero genuinely novel theoretical contributions. If the answer is "no," this is an ICAD paper, not an OOPSLA paper, unless merged with B.

2. **Does $d'_{\text{model}}$ satisfy the triangle inequality in your perceptual space?** If not, the greedy approximation guarantee (C2) collapses, and your only remaining theoretical contribution (however modest) is gone.

3. **How does a user who doesn't know what d' means write a perceptual specification?** If the answer requires a tutorial on psychoacoustic fundamentals, you haven't eliminated the expertise barrier—you've just moved it.

#### Verdict: **KILL as standalone / ABSORB into B**

The Math Assessor's 3/10 depth score and the Skeptic's "not program synthesis" attack are independently fatal for OOPSLA submission. However, C's specification language, greedy packing algorithm, and user-facing framing are valuable *components*. They should be absorbed into B as a high-level frontend: users write perceptual specifications (C's contribution), which desugar into typed DSL programs verified by B's refinement type system (B's contribution).

**Adjusted scores**: V6 / D4 / P4 / F8 (composite 22, down from 28)

---

## Part 2: Head-to-Head Comparisons

### A vs B: Theoretical Ambition vs. Reliable Execution

**Where A's theoretical ambition outweighs B's reliability:**
- A's resource semiring, if it works, unifies the paper's contribution into a single elegant abstraction. B's refinement types are a technically correct but arguably prosaic application of known machinery. The Math Assessor concedes A has "the deepest genuine mathematical ambition."
- A's best-paper ceiling is strictly higher: a "genuinely beautiful correspondence" between semiring laws and masking physics, vs. B's "Liquid Haskell, but for audio." Distinguished papers need an insight that makes reviewers say "of course"—A has that potential, B probably doesn't.

**Where B's practicality outweighs A's elegance:**
- B has a 75% probability of a successful soundness proof; A has 25% (exact) to 60% (approximate). The risk-adjusted expected contribution favors B by a wide margin.
- B's failure mode is graceful (falls back to a solid verification tool); A's failure mode is total (collapses to B, having wasted 4 weeks).
- B's evaluation story is concrete and achievable from day one; A's evaluation "cannot distinguish A from B" on any empirical metric (Skeptic's devastating 85% probability).
- The Difficulty Assessor's verdict: "B has the best difficulty-to-contribution ratio. It's hard enough to be a genuine OOPSLA paper but not so hard that it risks catastrophic failure."

**Resolution:** B dominates unless the team has a genuine graded type theory expert AND the week-2 empirical semiring test shows <10% associativity violations. Both conditions must hold.

### B vs C: What Does C Add That B Lacks?

**What C adds:**
- A user-facing specification language that lets non-experts declare perceptual goals rather than write DSL programs. B requires users to write typed specifications; C lets users declare "5 streams, each pairwise discriminable with d' ≥ 2.0" and generates the specification automatically.
- A synthesis/generation capability: B verifies user-written designs; C creates designs from scratch. For users who don't know where to start, C's "generate then verify" workflow is strictly more helpful than B's "write then verify."
- The "LLM backend" positioning, which the Skeptic calls "the only defensible story" for C, adds a concrete adoption pathway.

**Why B's type system still matters:**
- C needs a verifier. Without B's type system, C's "correct-by-construction" claim rests on the constraint solver's correctness—but this isn't a PL contribution. With B's type system, synthesized programs are *verified* by an independent mechanism, giving a stronger guarantee.
- C alone doesn't pass OOPSLA review (Math Assessor: 3/10 depth; Skeptic: "not program synthesis"). B alone does (Math Assessor: 5/10 depth; Skeptic: "CONDITIONAL CONTINUE"). C is a force multiplier for B, not a substitute.

**Is C's specification language worth absorbing into B?** Yes—all three evaluators independently recommend this. The specification language as a frontend to B's verification backend gives the combined system the broadest value proposition (C's strength) with the deepest PL contribution (B's strength).

### A vs C: Highest Ceiling vs. Lowest Risk

**The moonshot vs. the engineering project:**
- A aims for "genuinely novel PL theory" with a 25–40% success probability. C aims for "standard algorithms in a novel domain" with ~85% success probability.
- A's Math Depth score is 6/10 (the highest); C's is 3/10 (the lowest). A's Feasibility is 4/10 (the lowest); C's is 8/10 (the highest).
- If everything works, A produces a paper that changes how the PL community thinks about perception; C produces a paper that "OOPSLA reviewers will likely see through" (Difficulty Assessor).

**The uncomfortable truth:** A and C represent the two failure modes—over-ambition and under-ambition. Neither is the right calibration for the venue. B occupies the productive middle ground.

---

## Part 3: Consensus Points

All three evaluators agree on the following—these form the foundation for the final approach:

1. **Approach B has the best risk-adjusted profile.** Math Assessor: "best ratio of math that matters to math that impresses." Difficulty Assessor: "best difficulty-to-contribution ratio." Skeptic: "least fatal flaws" and "every flaw is mitigable."

2. **The custom SMT theory solver should be dropped as a crown jewel.** All three evaluators independently recommend committing to piecewise-linear encoding from day one. The contribution is the refinement type system, not the solver.

3. **C's specification language should be absorbed as a frontend to B.** Skeptic: "Use C's specification language as a frontend to B's verification backend." Difficulty Assessor implicitly agrees by noting C's value is in "domain application." Math Assessor: "The greedy packing algorithm is useful as engineering."

4. **The novel LoC claims are inflated by ~50% across all approaches.** Realistic novel LoC: A ~28K, B ~29K, C ~21K. Total with glue/infrastructure: 60–75K. The "48–60K novel" claim doesn't survive scrutiny.

5. **Diagnostics are critical and under-budgeted.** Skeptic: "diagnostics are the #1 priority from a user's perspective." Elevate to a primary engineering contribution with 25–30% of engineering time.

6. **Only one genuinely research-hard problem exists across all three approaches:** whether psychoacoustic masking interactions admit compositional algebraic structure (A's semiring question). Everything else is known techniques in a new domain.

7. **The project's claimed Grade A results are inflated.** Honest assessment across all approaches: A1 is B+, A2 is B+, B1 is A-/B+, B2 is B+, C1 is C+, C2 is B-. The depth check's "Grade A theorems are B/B+ formalizations" assessment is validated.

8. **The psychoacoustic models' accuracy is the meta-risk no one can mitigate.** All evaluators note that models from the 1970s–1990s, validated on simple stimuli with trained listeners, may show 30–50% prediction errors for complex multi-stream sonification. The response must be: "guarantees relative to the model, improvable independently."

9. **B1 (environment-dependent refinement predicates) is the single strongest genuine PL contribution.** Math Assessor: "the strongest 'new PL result' across all three approaches." This should be the centerpiece theorem.

---

## Part 4: Unresolved Disagreements

### Disagreement 1: Is Approach A worth pursuing in parallel?

- **Approaches document**: "Lead with B, keep A as a stretch goal. In parallel (weeks 1–4), explore whether the resource semiring can be made to work."
- **Skeptic**: "KILL. Redirect all effort to B." The 4-week parallel exploration is wasted effort that diverts attention from B's type system.
- **Math Assessor**: Partially open—"if the week-4 empirical test shows associativity violations <5%, invest in the ε-approximate semiring theory. This would be genuinely novel."
- **Difficulty Assessor**: Sympathetic to the parallel strategy but warns "a student without strong algebra intuition could waste 4+ months."

**Both sides fairly stated:** The Skeptic's position is risk-minimization: every hour spent on A's semiring is an hour not spent on B's soundness proof and diagnostics. The parallel-exploration position is option-value: a small time investment could unlock a significantly better paper. The disagreement is about risk tolerance, not about the assessment of A's probability of success.

### Disagreement 2: How much difficulty is enough for OOPSLA?

- **Difficulty Assessor (5/10 for B)**: "The novelty is in the application, not in the technique. This is a solid integration project with moderate novelty." Scores B lower than the approaches document's 7/10 because "once you accept the piecewise-linear approximation, most claimed hard problems dissolve."
- **Skeptic**: Implicitly accepts B's difficulty as sufficient: "CONDITIONAL CONTINUE" with concrete, achievable conditions. Frames B as "a strong OOPSLA paper, competitive for distinguished with excellent evaluation."
- **Math Assessor (5/10 for B)**: "Liquid Haskell, but for audio—a perfectly valid OOPSLA paper but not mathematically deep."

**Both sides fairly stated:** The Difficulty Assessor worries B is too easy to be distinguished at OOPSLA. The Skeptic and Math Assessor consider B adequate for acceptance, with evaluation excellence providing the path to distinguished. The disagreement is whether OOPSLA demands frontier difficulty (Difficulty Assessor) or rewards well-executed domain application (Skeptic's "solid DSL paper" framing).

### Disagreement 3: Can C's "non-expert users" claim be rescued?

- **Skeptic (85% failure)**: "Data journalists use Observable and Highcharts. They're not going to install a Rust compiler." The user base is imagined, not interviewed.
- **Approaches document (V7 for C)**: "Broadest user base of the three. Non-experts can use it."
- **Difficulty Assessor**: Notes the "specification language expressiveness vs. decidability tradeoff" is under-examined—users who can't specify d' thresholds won't benefit.

**Both sides fairly stated:** C's *idea* of serving non-experts is compelling in the abstract; the *execution* (Rust CLI, psychoacoustic terminology in the spec language, no user studies) doesn't match the vision. The question is whether the non-expert value proposition can be realized through a web API or LLM integration, or whether it's fundamentally unreachable within this project's scope.

### Disagreement 4: How damaging is the "type system vs. constraint checker" critique?

- **Skeptic**: If B's composition degenerates to "regenerate and re-check the entire conjunction," there's no compositionality and hence no type-system contribution. This is potentially fatal.
- **Math Assessor**: The environment-dependent refinement predicates (B1) are a genuine, if incremental, PL contribution. The environment-extension lemma is real work, not vocabulary dressing.
- **Difficulty Assessor**: The incremental checking (B3) could provide compositional reasoning, but it's "a nice optimization, not a hard problem."

**Both sides fairly stated:** Whether B is "really a type system" or "a constraint checker dressed in type-system vocabulary" may depend on whether the bounded-propagation result for incremental composition (B3) can be elevated from an engineering observation to a formal theorem about the type system's compositional structure. If yes, B has a genuine compositionality story. If no, the Skeptic's critique stands.

---

## Part 5: Recommendation for Synthesis

### The Winning Approach: B + C's Frontend + A's Ambition (Conditional)

Based on the debate, the synthesized approach should:

**Preserve from Approach B (the core):**
- The refinement type system with environment-dependent psychoacoustic predicates (B1). This is the paper's centerpiece theorem and the strongest PL contribution available.
- The piecewise-linear SMT encoding with δ-soundness. Frame as a correctness guarantee, not a research contribution.
- Incremental composition checking (B3), elevated to a first-class contribution with formal proof of bounded propagation. This is the compositionality story that defends against "it's just a constraint checker."
- The comprehensive evaluation plan: adversarial bug-finding, LLM baseline, cross-model validation, performance benchmarks, and human-data anchoring.

**Preserve from Approach C (the frontend):**
- The perceptual specification language, stripped of the "program synthesis" framing. Call it "declarative perceptual specification" and present it as a high-level input mode that desugars into typed DSL terms.
- The greedy packing algorithm, presented as an engineering contribution (not a theorem) that generates initial designs from specifications.
- The realizability checker, presented as a utility (not a crown jewel) with the NP-completeness noted in a single sentence, not a theorem statement.

**Preserve from Approach A (conditional upgrade path):**
- The resource-tracking insight: "psychoacoustic perception as a trackable resource." Even if the graded comonadic formalization fails, this conceptual framing can inform how B's refinement predicates are structured and presented.
- The week-2 empirical semiring test. Budget 1–2 person-weeks (not 4) for a quick empirical probe. If associativity violations are <5% for realistic configurations, the ε-approximate semiring becomes a stretch goal for paper revision. If >20%, close the door permanently.

**Kill decisively:**
- The graded comonadic type system as the primary architecture (A). The risk is too high, the timeline too short, and the evaluation gains too uncertain.
- The "program synthesis" framing (C standalone). OOPSLA reviewers will not accept CSP solving as program synthesis. The specification language survives; the framing does not.
- All ornamental theorems: A4 (tautological monotonicity), A5 (trivial decidability), C1 (trivial NP-completeness via graph coloring), C3 (definition dressed as theorem), C5 (minor corollary).
- The custom SMT theory solver as a claimed contribution. Commit to piecewise-linear from day one. If time permits later, build the custom theory as an optimization—never as a selling point.
- B2 as a "co-crown jewel." Rename to "Implementation Correctness" and present as engineering verification.

### Adjusted Composite Scores

| Approach | Original V/D/P/F | Adjusted V/D/P/F | Original Composite | Adjusted Composite |
|----------|-------------------|-------------------|--------------------|--------------------|
| A | 5/8/8/5 | 5/7/6/4 | 26 | 22 |
| B | 6/7/7/8 | 5/5/6/8 | 28 | 24 |
| C | 7/6/6/9 | 6/4/4/8 | 28 | 22 |
| **B+C frontend** | — | **6/5/7/8** | — | **26** |

The synthesized B+C approach scores highest after adjustment, combining B's PL depth with C's value breadth. The Potential score rises to 7 because the combined system has a more compelling story: "Declare what you want to perceive; the type system guarantees your specification is physically realizable."

### Paper Framing for the Synthesized Approach

**Title direction**: "SoniType: Refinement Types for Psychoacoustic Verification of Auditory Displays"

**One-sentence pitch**: "We present a refinement type system with environment-dependent psychoacoustic predicates that provides compositional, SMT-backed verification of multi-stream sonification designs, catching perceptual defects—masking, indiscriminability, cognitive overload—that published alarm palettes and LLM-generated sonification code routinely exhibit."

**Contribution hierarchy**:
1. (Theory) Refinement type system with non-local environment-dependent predicates for psychoacoustic constraint verification, with soundness proof.
2. (Algorithm) Incremental composition checking with bounded propagation, enabling compositional verification.
3. (Tool) Perceptual linting mode + declarative specification frontend with greedy design generation.
4. (Evaluation) Adversarial bug-finding benchmark on published failures, LLM baseline comparison, cross-model validation, and human-data-anchored accuracy assessment.

### Timeline Gates

- **Week 1**: SMT performance benchmark (piecewise-linear, 8 streams, <2s target). Go/no-go.
- **Week 2**: Optional semiring empirical probe (associativity test, 10K configs). Informational only.
- **Week 4**: Type system formalization + soundness proof sketch complete. Kill gate.
- **Week 6**: LLM baseline comparison complete. Kill gate.
- **Week 8**: Human-data anchor comparison complete. Kill gate.
- **Week 10**: Full evaluation + paper draft.

---

*This debate document synthesizes three independent evaluations into a consensus recommendation. The surviving approach (B + C frontend) is not the most exciting option—it's the one most likely to produce a publishable OOPSLA paper. The team should be comfortable with this tradeoff: a guaranteed solid paper is worth more than a coin-flip at a great one.*
