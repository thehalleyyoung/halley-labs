# IsoSpec Approach Debate: Adversarial Skeptic vs. Mathematician

**Panel:** Adversarial Skeptic (AS) × Math Depth Assessor (MDA)  
**Subject:** Three competing approaches for verified cross-engine transaction isolation analysis  
**Date:** 2026-03-08

---

## Approach A: Engine Model Maximalist

**Self-scores:** V8 / D9 / P8 / F5

### Skeptic's Critique

**1. Fatal Flaws**

Four independently risky subsystems (five engine semantics, Lean 4 proofs, M5 predicate theory, SMT encoding), and they're *sequentially dependent* — the SMT encoding depends on the engine models, the Lean proofs depend on the encoding, M5 threads through everything. One subsystem slipping 3 months cascades through the entire chain.

The Lean 4 formalization of Cahill's dangerous structure theorem alone is PhD-sized work. The Difficulty Assessor gives a 40% probability of >6 month delay, and the Lean 4 ecosystem for database formalization is nonexistent.

**2. Unrealistic Assumptions**

The approach assumes five production engines can be modeled at comparable fidelity. They cannot. PostgreSQL's SSI has published academic descriptions (Cahill et al., TODS'09). Oracle's MVRC is proprietary — the proposal labels this model "approximate." MySQL's gap locking depends on optimizer internals that change across minor versions. You are claiming comparable formalization quality across engines with wildly different documentation quality.

The 55K novel LoC estimate (already corrected from 78K) is still 15–30% optimistic per the Difficulty Assessor.

**3. Prior Art Overlap**

Strip away the engine models and what remains is CLOTHO's architecture: parse SQL → encode as SMT → solve → extract counterexamples. The refinement theory (M2) is standard process-algebraic refinement. The compositionality theorem (M8) is nearly definitional. The bounded soundness result (M7) follows from anomaly definition structure. Four of eight claimed math contributions are standard or ornamental — the depth check confirmed genuine contributions number 2–3, not 8.

**4. Dealbreakers for Reviewers**

A SIGMOD reviewer will ask: "Why should I believe your PostgreSQL SSI model is faithful when PostgreSQL's behavior changes across minor versions?" Version pinning is not satisfying — the tool's guarantees expire with every `apt upgrade`. The validation oracle uses Docker-based interleaving forcing, which is "inherently approximate" by the proposal's own admission.

A CAV reviewer will ask: "You mechanized proofs for one engine out of five. How is this different from any paper that claims 'we prove X' in English?"

**5. The Question That Has No Good Answer**

*"Your evaluation claims >95% witness confirmation rate, but your validation oracle has acknowledged false negatives, your Oracle model is approximate, and your MySQL model over-approximates gap locking. What is your actual false positive rate, and how would a practitioner distinguish a model artifact from a real bug?"*

There is no clean answer. The over-approximation strategy is formally sound (no false negatives) but the resulting false positive rate is unknown a priori and could be high enough to make the tool useless in practice. You cannot measure this without the very empirical infrastructure that Approach B builds.

### Mathematician's Response

**Valid and damaging:** The Lean 4 schedule risk is real — the 40% delay probability is well-calibrated against comparable proof engineering efforts. The cascading subsystem dependencies are the central delivery risk. The Oracle model critique is also valid.

**Addressable with scope changes:** The strongest attacks dissolve if Lean 4 proofs are descoped to "key lemmas mechanized" (saving ~4 months) and engines drop from 5 to 3 (PG, MySQL, SQL Server). This reduces novel LoC to ~40K and eliminates the Oracle embarrassment. The core contribution survives intact.

**Overstated risk:** The CLOTHO overlap criticism is technically accurate but strategically misleading. CLOTHO checks abstract consistency models for distributed stores; IsoSpec checks engine-specific implementation models for SQL databases. Sharing a pipeline architecture does not make contributions duplicative. The engine models are the load-bearing novelty.

**Realistic best case:** 3 engines, paper proofs with mechanized key lemmas, full M5, discovery of 10+ novel divergences. A strong SIGMOD accept — but it is a significantly descoped version of the proposal.

---

## Approach B: Empirical-First Differential Testing

**Self-scores:** V9 / D6 / P7 / F8

### Skeptic's Critique

**1. Fatal Flaws**

The approach has a deceptive simplicity. "Just fuzz both engines and compare" sounds straightforward until you confront interleaving control. Differential testing of concurrent programs requires running *the same schedule* on two engines with different internal schedulers. The three-tier mitigation (advisory locks → retry 1000× → engine-specific hooks) is a euphemism for "fight non-determinism with brute force."

The fuzzer evaluation requires 140–1400 CPU-hours for meaningful coverage statistics — contradicting the "laptop CPU" constraint.

**2. Unrealistic Assumptions**

The approach assumes the symbolic post-hoc analyzer can explain divergences without engine models. But the Difficulty Assessor caught this: the symbolic analyzer "secretly depends on engine models." You cannot explain *why* PostgreSQL aborts a transaction that MySQL commits without modeling what each engine does. The "lightweight engine constraint sketches" are engine models by another name — just less rigorous ones.

The coverage theory promises detection probability bounds, but these are only meaningful if the "interesting dimensions" of the program space are correctly characterized. Miss a dimension (e.g., index-dependent gap locking), and the guarantee is vacuous.

**3. Prior Art Overlap**

This is Hermitage automated with a fuzzer. Hermitage manually wrote test cases and ran them on multiple engines; this generates them programmatically. The delta is automation, not insight. Jepsen and Elle do related black-box concurrent testing. The novelty rests on grammar-based SQL transaction fuzzing (variants exist, e.g., SQLsmith) and the symbolic explanation layer — which secretly requires engine models.

**4. Dealbreakers for Reviewers**

A SIGMOD reviewer will ask: "What is your soundness guarantee?" Answer: none. The tool discovers divergences but can never prove their absence. "We fuzzed for 100 hours and found nothing" is not a safety certificate.

A formal-methods reviewer will see this as "just testing" elevated by the word "differential." The paper needs spectacular results (50+ novel divergences) to overcome this, and flakiness risk suggests the confirmed count may disappoint.

**5. The Question That Has No Good Answer**

*"Your tool reports my migration is safe after 1000 hours of fuzzing. What is the probability that a divergence exists but was not found? Can you quantify the residual risk?"*

The honest answer is: no. The coverage theory gives asymptotic bounds under assumptions about the program space structure, but these assumptions are unverifiable for real workloads. A practitioner making a migration decision needs either a formal guarantee or a calibrated probability estimate. This approach provides neither.

### Mathematician's Response

**Valid and damaging:** The soundness gap is real and unfixable within this paradigm. The symbolic analyzer's secret dependence on engine models is a genuine architectural dishonesty. The CPU-hour requirement is a real constraint.

**Addressable with scope changes:** Reframe from "migration safety tool" to "divergence discovery tool." Drop the symbolic explanation layer, invest in better interleaving control, and the paper becomes "Hermitage 2.0: Automated Discovery of Transaction Isolation Divergences at Scale."

**Overstated risk:** The Skeptic's dismissal of empirical work as "just testing" applies a formal-methods bias inappropriate for SIGMOD/VLDB. Empirical systems papers regularly win best paper. If the divergence count is high enough (30+ novel, confirmed), no reviewer will call it "just testing." Imperfect interleaving reproduction is acceptable when the alternative is no testing at all.

**Realistic best case:** 30+ confirmed novel divergences, a reusable fuzzing framework, and a searchable engine behavior database. A solid SIGMOD accept — though likely not best-paper, because the method is thinner than the results.

---

## Approach C: Migration-Focused Portability Checker

**Self-scores:** V9 / D5 / P6 / F9

### Skeptic's Critique

**1. Fatal Flaws**

Delta completeness is an existential risk the proposal acknowledges but underestimates. A missing delta entry produces a false negative — declaring a migration safe when it is not. This is *worse* than no tool, because practitioners who trust it will skip manual review. The mitigation (docs + Hermitage + conservatism) is exactly the manual, documentation-dependent process the tool should replace.

How do you know your Oracle↔PG delta is complete when Oracle's concurrency control is proprietary?

**2. Unrealistic Assumptions**

The approach assumes the behavioral delta is substantially smaller than full engine models. The Difficulty Assessor challenges this: the "restricted" predicate theory saves ~30% effort, not 60%. For Oracle→PG, you need to model Oracle's MVRC visibility rules and PG's SSI dependency tracking well enough to compute their difference — requiring nearly the same depth as Approach A.

Compositionality of the delta is claimed as "minor" math, but real workloads interact through foreign keys, shared sequences, and implicit locking. The independence assumption rarely holds for non-trivial applications.

**3. Prior Art Overlap**

Differential analysis is well-studied in software engineering (differential symbolic execution, differential testing). Applying it to SQL isolation is a domain contribution, not a methodological one. The "differential isolation semantics" is essentially computing the symmetric difference of two schedule sets — the obvious formulation, novel mainly in branding.

**4. Dealbreakers for Reviewers**

A SIGMOD reviewer will ask: "How is this not just a migration linter? Where's the research contribution?" The 38K novel LoC and restricted predicate theory may not clear the novelty bar. The paper lives or dies on whether the tool catches real migration bugs practitioners missed.

A theory-oriented reviewer will note the "restricted M5" avoids the hard parts of predicate-level conflict theory. It's a scope reduction, not a theoretical contribution.

**5. The Question That Has No Good Answer**

*"You claim delta completeness via conservative over-approximation. What is your measured false positive rate on real workloads, and at what false positive rate does the tool become more annoying than useful?"*

If the over-approximated delta flags 40% of transactions, practitioners will ignore all warnings. The approach has no mechanism to distinguish high-confidence differences from speculative ones included for conservatism.

### Mathematician's Response

**Valid and damaging:** Delta completeness is genuinely existential. The mitigation doesn't solve the problem — it makes it someone else's problem. The restricted M5 saves only 30%, and the resulting theory is narrower and less publishable.

**Addressable with scope changes:** Focus on 2–3 highest-value pairs (Oracle→PG, SQL Server→MySQL, MySQL→PG) where documentation is best. Add explicit confidence tiers: "validated delta" vs. "approximate delta."

**Overstated risk:** The Skeptic judges against a theoretical ideal no practical tool achieves. A tool catching 90% of migration issues and flagging 10% as "uncertain" is enormously valuable. The false positive rate is measurable — run on known-safe workloads and calibrate. A focused, shippable tool solving the #1 practitioner problem is what SIGMOD tool papers reward.

**Realistic best case:** A clean tool paper with 3 well-validated migration pairs, compelling case studies, and sub-second analysis. Not a best-paper candidate, but a useful, publishable contribution deliverable in 6–9 months with ~80% confidence.

---

## Cross-Approach Debate

### Score Corrections

**Skeptic:** All three approaches have inflated self-scores. Let me correct them.

**Approach A** claims D9. I'll grant D8 — the raw difficulty is high, but 40% of the claimed math is ornamental (M3, M7, M8) and the Lean proofs are difficulty-without-purpose if they don't finish. You don't get difficulty credit for work that might not ship. P8 drops to P6 — the CLOTHO positioning problem is severe, and partial Lean mechanization is an awkward middle ground that impresses neither the formal methods community (not enough) nor the systems community (who cares?). F5 is actually *generous* — with four independently risky subsystems and a 40% Lean delay probability, F4 is more honest. **Corrected: V7 / D8 / P6 / F4.**

**Approach B** claims V9. I'll grant it — practitioners do trust empirical evidence. But D6 is inflated: the interleaving control problem is harder than acknowledged, and the symbolic analyzer secretly requires engine models, adding ~2 months of undisclosed work. Actual difficulty, accounting for hidden complexity, is D7. P7 is optimistic — without formal guarantees, the paper needs spectacular empirical results, and the flakiness risk means the confirmed-novel count may disappoint. P6 is fairer. F8 drops to F7 once you account for the 140–1400 CPU-hours and the Docker infrastructure complexity. **Corrected: V9 / D7 / P6 / F7.**

**Approach C** claims D5. This undersells the delta completeness challenge and the restricted M5 work. The Difficulty Assessor says the restricted predicate theory saves only 30%, not 60%. D6 is more honest. V9 holds, but only if the tool *works* — a migration checker that misses bugs is worse than none. V8 with a risk-adjusted discount. P6 is actually fair, maybe even generous given the thin theoretical contribution. F9 is optimistic — delta completeness validation is harder than assumed, and covering even 3 migration pairs thoroughly is ~4 months of engine-specific work each. F8. **Corrected: V8 / D6 / P6 / F8.**

**Mathematician:** The Skeptic's corrections are broadly reasonable but overly punitive on potential.

Approach A's P6 assumes the CLOTHO positioning problem is unsolvable. It isn't — "CLOTHO models what the spec says, we model what the engine does" is a one-sentence differentiator that any competent author can make clear. P7 is fairer: it's a strong accept at SIGMOD if scoped to 3 engines, and the discovery angle (novel divergences beyond Hermitage) is genuinely exciting.

Approach B's P6 applies a formal-methods bias to a systems venue. SIGMOD regularly awards best paper to empirical systems work. If the divergence count is high (30+) and the fuzzing framework is reusable, P7 holds. The D7 upgrade is fair — the Skeptic correctly identifies hidden complexity.

Approach C's scores are approximately right. The thin theory is a real liability, but SIGMOD tool papers exist and are valued. P6 is fair.

| Dimension | A (Self→Corrected) | B (Self→Corrected) | C (Self→Corrected) |
|-----------|:---:|:---:|:---:|
| **Value** | 8→7 | 9→9 | 9→8 |
| **Difficulty** | 9→8 | 6→7 | 5→6 |
| **Potential** | 8→7 | 7→6.5 | 6→6 |
| **Feasibility** | 5→4 | 8→7 | 9→8 |
| **Composite** | 30→26 | 30→29.5 | 29→28 |

### Which Approach Has the Best Risk-Adjusted Potential?

**Skeptic:** Approach B has the highest risk-adjusted composite (29.5), but its ceiling is capped by the absence of formal guarantees. The best version of B is a great empirical paper — but it's not a *framework* that future work builds on. Approach C at 28 is the safest bet with the highest delivery probability. Approach A at 26 is the most intellectually ambitious but the most likely to deliver a half-finished artifact that impresses no one.

My recommendation: **None of these approaches, as stated, should proceed.** The optimal project is a hybrid — C's differential framing, A's engine models for 3 engines (PG, MySQL, SQL Server), A's M5 predicate theory at C's restricted scope, and B's empirical validation as the ground-truth oracle. This hybrid was already suggested in the approaches document. The question is not "which approach" but "which hybrid."

If forced to choose one: **Approach C**, because it ships, solves the highest-value problem, and the theoretical thinness can be compensated with strong evaluation. But C-as-stated needs A's engine modeling discipline — the deltas must be derived from real engine understanding, not documentation scraping.

**Mathematician:** I agree with the hybrid recommendation but disagree on the forcing function. If forced to choose one: **Approach A, descoped to 3 engines with paper proofs instead of full Lean mechanization.** The engine models are the load-bearing novelty of the entire project. They enable Approach C (you need models to compute deltas), they enable Approach B (you need models to explain divergences), and they are the reusable research artifacts that future work cites. Approach C without rigorous engine understanding is a migration linter built on documentation summaries. Approach A, properly scoped, is the foundation everything else builds on.

The Skeptic is right that A-as-proposed is undeliverable. But A-descoped (3 engines, no full Lean, restricted M5) is essentially the strongest version of C — with the crucial addition that you actually understand the engines well enough to trust the deltas.

---

## Verdict

### Corrected Scores (Consensus)

| Dimension | A: Engine Maximalist | B: Empirical-First | C: Migration-Focused |
|-----------|:---:|:---:|:---:|
| **Value** | 7 | 9 | 8 |
| **Difficulty** | 8 | 7 | 6 |
| **Potential** | 7 | 6.5 | 6 |
| **Feasibility** | 4 | 7 | 8 |
| **Risk-Adjusted Composite** | 26 | 29.5 | 28 |

### Skeptic's Recommendation

**Approach C with mandatory upgrades.** C has the best risk-to-reward ratio and solves the problem practitioners actually pay money to solve. But it must be upgraded with: (a) real engine modeling discipline for the top 3 migration pairs (not documentation scraping), (b) the restricted M5 predicate theory (properly acknowledging it saves 30% not 60%), and (c) empirical delta validation from Approach B's infrastructure. Without these, C is a documentation-summarization tool with an SMT veneer.

If Approach C cannot incorporate engine modeling rigor, I recommend **Approach B** as a fallback — it at least produces truthful empirical results without pretending to formal guarantees it cannot deliver.

### Mathematician's Recommendation

**Approach A, aggressively descoped.** Three engines (PG, MySQL, SQL Server). Paper proofs with 3–5 key lemmas mechanized in Lean 4 (not full mechanization). Full M5 predicate theory for the conjunctive inequality fragment. Differential encoding (from C) for the portability analyzer. Empirical validation (from B) for model adequacy. This is ~40K novel LoC, deliverable in 10–12 months, and produces the reusable formal artifacts that give the project lasting research value.

The descoped A is the only approach that creates *new knowledge* (engine-faithful operational semantics that have never existed) rather than *new tools* built on existing knowledge. Tools are useful; knowledge is cited.

### Points of Agreement

1. **No approach should proceed as-stated.** All three need scope adjustments.
2. **M5 (predicate-level conflict theory) is essential** in some form. Without it, the tool cannot handle WHERE clauses, which means it cannot handle real SQL.
3. **Engine models for PG and MySQL are non-negotiable.** These are the two highest-value engines with the best documentation. Any serious approach must understand their concurrency control at depth.
4. **The Oracle model should be either explicitly approximate or dropped.** Claiming formal verification against proprietary internals is not credible.
5. **Empirical validation is required regardless of approach.** Models without empirical ground truth are academic exercises. B's validation infrastructure is needed even if B's overall approach is not chosen.
6. **Full Lean 4 mechanization is not worth the schedule risk** at this stage. Paper proofs with selected mechanized lemmas are sufficient for a first publication.
7. **The optimal project is a hybrid** drawing engine modeling rigor from A, differential framing from C, and empirical validation from B.

### Points of Disagreement

1. **Value of formal artifacts vs. practical tools.** The Mathematician prizes the engine models as lasting research contributions; the Skeptic values shipping a working tool that solves practitioner problems. This is a genuine values disagreement, not a factual one.
2. **Severity of delta completeness risk (C).** The Skeptic considers it existential; the Mathematician considers it manageable with empirical calibration. The truth depends on how many undocumented engine behaviors exist — which we don't know until someone does the work.
3. **Publication potential of B.** The Skeptic sees P6 (thin intellectual contribution); the Mathematician sees P7 if divergence count is high. This depends on whether reviewers at SIGMOD 2027 value formal novelty or empirical impact more heavily — an unknowable variable.
4. **Whether A-descoped is "really" Approach A.** The Mathematician's recommended version (3 engines, paper proofs, restricted scope) shares more DNA with C than with A-as-proposed. The Skeptic considers this a concession; the Mathematician considers it strategic scoping.
