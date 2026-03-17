# RegSynth: Approach Debate

**Slug:** `pareto-reg-trajectory-synth`
**Stage:** Ideation — Adversarial Debate
**Date:** 2026-03-08
**Participants:** Adversarial Skeptic (S), Math Depth Assessor (M), with responses from Domain Visionary (V) and Difficulty Assessor (D)

---

## Round 1: Critiques of Each Approach

### Approach A: "ConflictProver"

#### Skeptic Attack

**S1: The output is useless to practitioners.**
"ConflictProver's headline capability — formal infeasibility certificates — answers a question nobody is asking in a format nobody can use. Compliance officers already know which articles are in tension. What they need is *resolution guidance*: 'Given these conflict, here's what to do.' ConflictProver explicitly strips this out."

"A formal proof that EU AI Act Art. 12 and GDPR Art. 5(1)(c) conflict would be met by regulators with: 'Yes, we know. That's why we published guidance.' The system formalizes the *text* of regulations, not the *practice* of regulation — and compliance lives in the practice."

**S2: Novelty is integration, not innovation.**
"XACML policy conflict detection has been doing this for access control policies since the mid-2000s. SPINdle detects defeasible deontic logic conflicts. The jump from 'access control policies' to 'regulatory obligations' is a domain shift, not an algorithmic one. The 'first formal conflict prover for AI regulation' claim is technically true only because the specific combination of {AI regulation + SMT + MUS + certificates} hasn't been packaged before."

**S3: Circular trust.**
"The certificates are 'machine-checkable' but the machine is checking its own homework. The system's guarantees depend on encoding correctness, which is unvalidatable without the very legal expertise the system claims to supplement."

#### Mathematician Critique

**M1: Math is shallow and honest about it.**
"Score 3/10. M1 (algebra soundness) is a standard compiler-correctness homomorphism — textbook structural induction. M3 (certificates) is definitional. M2 (MUS coverage) and M4 (conflict density) are padding — the system works without them. There's nothing here a PL researcher would call 'new math.'"

**M2: The load-bearing math is right.**
"But that's fine for what this is. M1 and M3 are genuinely load-bearing — without them, the certificates are meaningless. ConflictProver doesn't pretend to be a math paper. The problem is: at ICSE Tools Track, is a tool with standard formal foundations enough for best-paper? Historically, no — ICSE best papers have deeper technical novelty."

**M3: M2 (MUS Coverage) is problematic.**
"The O(n log n) coverage claim requires structural assumptions about regulatory independence that probably don't hold. Drop it or state it as a proposition under explicit assumptions — calling it a 'theorem' invites attack."

#### Visionary Response

**V1: Resolution guidance can be added as output postprocessing.**
"The Skeptic is right that raw MUS output is insufficient. But the fix is simple: for each MUS (conflict core), generate a set of 'resolution options' — enumerate which obligations can be relaxed to resolve the conflict, ranked by regulatory penalty severity. This is a straightforward extension of MUS analysis (each element of the MUS, when removed, resolves the conflict). The formal infrastructure is the hard part; formatting resolution options is an afternoon's work."

**V2: The XACML comparison is misleading.**
"XACML policies are Boolean access control rules in a fixed schema. Regulatory obligations have temporal phasing, jurisdictional hierarchy, conditional triggers, exemptions, and cross-references. The encoding is qualitatively harder. The MUS extraction and provenance mapping at regulatory scale is novel engineering. Integration novelty *is* novelty at a tools venue."

#### Difficulty Assessor Response

**D1: The 'hard middle' problem is real.**
"Skeptic and Mathematician both miss the actual bottleneck: regulatory encoding (H2). Each of 300 articles requires 1–4 hours of legal-to-formal translation. This is 300–1200 hours of domain expert work. No team member with a CS background wants to do this. It will be done last, done hastily, and contain silent errors. This is the risk that kills ConflictProver — not the math, not the solver, but the inputs."

---

### Approach B: "RegSynth-Complete"

#### Skeptic Attack

**S4: The ILP baseline destroys the entire narrative.**
"If ILP delivers 80–90% of practical value (the depth check's assessment), then 70% of Approach B's effort — everything beyond the ILP baseline — is a marginal improvement at massive complexity cost. A reviewer will ask 'Why not just use ILP?' and the answer 'MaxSMT handles deeply nested logical implications' is weak when those implications are the least formalizable obligations."

**S5: 11 components, 6 math contributions, 0 depth.**
"The scope is the enemy, not the selling point. No single component will be done well enough to withstand scrutiny. The paper tries to do six things adequately instead of one thing brilliantly. A best-paper committee will see through this."

**S6: The formalizability completeness theorem is sensitivity analysis.**
"Unpack M1: 'If you correctly encode the easy parts, the result is correct for the easy parts, and we bound the error from the hard parts.' That's what every approximate model does. Operations researchers have been doing sensitivity analysis since the 1960s. Calling it a 'completeness theorem' is grade inflation."

**S7: The 4D Pareto frontier is an academic fantasy.**
"No compliance officer has ever looked at a Pareto frontier. Real decisions are made by: lawyers identifying minimum-viable compliance, business teams negotiating budget, executives choosing 'good enough.' The system answers a question nobody is asking."

#### Mathematician Critique

**M4: Most grade-inflated of all three approaches.**
"Six contributions graded B+ down to C+ that honestly range from C to B-. M4 (temporal dominance) is Bellman 1957 applied to regulations — an afternoon's work. M5 (confidence algebra) is standard abstract interpretation. M6 (incremental Pareto) is an algorithmic recipe, not a theorem. Only M1 (formalizability completeness) has potential, and it risks collapsing to sensitivity analysis."

**M5: The tightness claim for M1 is the swing factor.**
"If the bound Δ_X can be shown tight under realistic (non-independent) assumptions, M1 is a genuine B/B+ contribution. If tightness requires unrealistic independence assumptions, it collapses to C+. The proponents must demonstrate non-trivial tightness *on real regulatory instances* — not synthetic. This is the make-or-break proof."

**M6: The ε-constraint Pareto computation is not novel.**
"Haimes et al., 1971. Adapting to MaxSMT with blocking clauses adds a wrinkle (non-convex feasible regions) but doesn't constitute a new algorithm. The O(1/ε^{d-1}) bound with d=4 gives ~8000 iterations at ε=0.05 — each a MaxSMT call. This may be theoretically correct but practically useless."

#### Visionary Response

**V3: The ILP-vs-MaxSMT comparison IS the contribution.**
"The Skeptic frames ILP sufficiency as a fatal flaw. I frame it as a feature. The paper's contribution includes the *empirical characterization* of where ILP suffices and where MaxSMT is necessary. If ILP handles 80% of cases, the paper says: 'here is the formal tractability result; ILP works for the common case; MaxSMT is needed for deeply nested obligations.' This is an honest, publishable outcome that advances the field's understanding."

**V4: The formalizability theorem is NOT sensitivity analysis.**
"Standard sensitivity analysis asks: 'how does the optimal solution change if parameters vary?' M1 asks a fundamentally different question: 'what guarantees does a solver provide when a significant fraction of the model is *missing* (opaque obligations), and we can characterize the structure of the missing fragment?' The novelty is the interaction between the obligation algebra's compositionality and the formalizability grades. Naïve worst-case bounds are too conservative; the algebra's structure allows tighter bounds."

**V5: I concede on scope.**
"The Skeptic is right that 11 components is too many for one paper. The realistic paper covers: DSL with formalizability grading + ILP baseline + MUS detection + formalizability completeness theorem + planted benchmarks. This is Approach A + the formalizability theorem + Pareto via ILP. The temporal extension and remediation planner are stretch goals or follow-on work."

#### Difficulty Assessor Response

**D2: The realistic build IS 'A + ILP Pareto.'**
"The proponent's V5 concession confirms what the difficulty review found: Approach B descopes to Approach A + ILP Pareto baseline in practice. The 66-week realistic timeline for the full system is incompatible with any paper deadline. The practical version is achievable in 10–14 months."

**D3: The Pareto engine is the real engineering challenge.**
"The ε-coverage guarantee with 4D blocking clauses, incremental MaxSMT, and termination detection requires 6–10 weeks of focused work. This is the component that separates B from A. If it works well, the ILP-vs-MaxSMT comparison is a genuine contribution. If it doesn't, you have Approach A with wasted time."

---

### Approach C: "RegVerify"

#### Skeptic Attack

**S8: Timeline is incompatible with the regulatory moment.**
"The motivation is EU AI Act enforcement August 2026. The realistic timeline for a verified compiler is 2–4 years. By the time RegVerify ships, enforcement will have evolved, practices matured, and urgency passed. This is a solution optimized for today's problem that arrives after the problem has changed."

**S9: Verification of the wrong layer.**
"The verified compilation proves τ is correct. But the real correctness question is: 'Does obligation O, as expressed in the DSL, faithfully represent Article X of the EU AI Act?' This is human judgment no Lean 4 proof addresses. Approach C provides a machine-checked guarantee for the *uncontroversial* part (compilation) while leaving the *controversial* part (legal interpretation) to humans."

**S10: The fallback IS Approach A.**
"Kill gate at week 12: 'If core proof isn't complete, fall back to tested-but-unverified encoding.' In other words, the 50%+ downside scenario produces Approach A three months late. The expected value is negative relative to just building A."

**S11: Who is the user?**
"Financial services compliance teams use Excel, SharePoint, and GRC platforms. They've never heard of Lean 4. The 'high-assurance compliance environment' persona is aspirational. This is a solution for users who will exist in 10 years if everything else goes right."

#### Mathematician Critique

**M7: The only approach with genuinely hard math.**
"I disagree with the Skeptic on one critical dimension: Approach C's type theory metatheory (M1) is real PL theory. The dependent type system indexed by jurisdiction and temporal interval, with decidability and completeness characterization, has no precedent. The techniques are known (logical relations, normalization) but the application creates genuinely new verification challenges."

**M8: Grade A- is defensible IF dependent types are genuine.**
"The critical question: do the dependent types involve genuine type-level computation, or are they just finite-set labels? If jurisdiction indices are simple labels with no computation at the type level, the metatheory simplifies dramatically and the grade drops to B/B+. If conditional obligations create path-dependent types, decidability and normalization are genuinely hard. The proposal is ambiguous on this point."

**M9: The Lean 4 effort is catastrophically underestimated.**
"5,000–8,000 lines of Lean 4 for a verified regulatory compiler is absurd. CompCert: ~100K lines of Coq for C. Even simple lambda-calculus-to-stack-machine compilers take 3–5K lines. The obligation DSL with 4 custom operators, jurisdiction indexing, temporal sorts, and multi-backend lowering: 12–22K lines is realistic. This is a multi-thesis effort disguised as one component."

**M10: The most mathematically novel but least likely to deliver.**
"Risk-adjusted math quality: 7 × 0.45 = 3.15. Compare A: 3 × 0.95 = 2.85, B: 4 × 0.75 = 3.00. All three are surprisingly close on expected math value. C has the highest ceiling and the lowest floor."

#### Visionary Response

**V6: Partial verification is publishable.**
"The Skeptic's timeline argument assumes full verification. The realistic plan: verify the 'linear fragment' (no disjunction, no exception) which covers ~60% of obligations. This is achievable in 4–6 months. The paper's contribution: 'We verify compilation for a significant regulatory fragment, characterize the boundary, and provide tested-but-unverified compilation for the rest.' This is honest and publishable."

**V7: The trust argument survives the Skeptic's attack.**
"Yes, encoding correctness still depends on human judgment. But Approach C eliminates an entire *class* of errors (compiler bugs) that Approaches A/B can't. The trust argument is layered: 'Human experts validate the DSL encoding. The compiler is machine-verified. The solver is independently validated.' Each layer catches different errors. Approach C closes one layer completely."

#### Difficulty Assessor Response

**D4: The Rust-Lean synchronization problem is under-discussed.**
"The Skeptic and Mathematician both miss the integration nightmare: maintaining two implementations (production Rust, verification Lean 4) in sync. Every DSL change requires dual updates. No automated check ensures they match. This is the verified-compiler maintenance problem that has killed real projects."

**D5: A descoped C converges toward A.**
"If Lean 4 verification is abandoned (>50% probability), and dependent types are simplified (if the metatheory is too hard), Approach C becomes: Approach A with a slightly richer type system and wasted months. The unique value proposition evaporates."

---

## Round 2: Direct Challenges Between Experts

### Skeptic → Mathematician: "Your math scores don't matter."

**S12:** "You scored C at 7/10 math depth. But math depth is irrelevant if the paper never ships. The expected delivered math is 7 × 0.45 = 3.15 for C vs. 3 × 0.95 = 2.85 for A. For an 0.3-point difference in expected value, C requires 2× the timeline and has 4× the kill probability. The rational choice is A with honestly shallow math."

**Mathematician Response:**
**M11:** "I agree on expected value arithmetic. But I disagree on the framing. The question isn't 'which approach maximizes expected delivered math' — it's 'which approach maximizes P(best paper).' Best papers require surprise. A can never surprise: its math is routine, its system is competent, its contribution is integration novelty. C could surprise: a verified regulatory compiler is genuinely new. The tail outcome matters. If we're playing for publication, A is safe. If we're playing for impact, C is the only bet."

### Mathematician → Skeptic: "Your 'nobody wants this' argument is unfalsifiable."

**M12:** "You claim no practitioner wants formal conflict proofs. But practitioners didn't want formal program verification either — until they did (CompCert, seL4, HACL*). The regulatory compliance market is pre-formal: practitioners don't know what they want because the tool doesn't exist. Your argument proves too much — applied to 2003, it would have killed CompCert."

**Skeptic Response:**
**S13:** "CompCert took 12 years, a team of 5, and required aerospace/defense customers with explicit verification mandates. RegVerify has no analogous customer demanding verified regulatory compilation. The EU AI Act doesn't require formal methods. Financial regulators don't mandate Lean 4 proofs. CompCert succeeded because it found customers who *needed* verified compilation by law. Show me the customer who *needs* verified regulatory encoding by law, and I'll reconsider."

### Skeptic → Visionary: "Your concession on B's scope admits it's A with overhead."

**S14:** "In V5, you conceded that the realistic Approach B is: DSL + ILP + MUS + formalizability theorem + planted benchmarks. That's Approach A plus the formalizability theorem and ILP Pareto. The temporal extension, MaxSMT research delta, remediation planner, and 4 of the 6 math contributions are cut. What remains is not 'Approach B' — it's 'Approach A with a theorem about partial formalization.' Is one theorem worth 6–8 months of additional work?"

**Visionary Response:**
**V8:** "Yes. The formalizability completeness theorem transforms the intellectual contribution from 'we built a tool' to 'we proved that partial formalization is formally valid.' This is the difference between a tools paper and a research paper. It also protects against the strongest attack on A ('why should I trust your incomplete formalization?'). The theorem costs 1–2 months of proof work and adds 6–8 months total timeline. The additional time is mostly for the Pareto engine, which provides the ILP-vs-MaxSMT comparison that makes the evaluation independently interesting."

### Difficulty Assessor → All: "You're all ignoring the real bottleneck."

**D6:** "Every approach budgets regulatory encoding as 'domain data, not algorithmic novelty' — 15–18K LoC of DSL articles. But this is 300–1200 hours of legal-to-formal translation requiring domain expertise the team likely lacks. The depth check recommended external encoding validation (Amendment A5). None of the approaches has a plan for *producing* the encodings — only for *validating* them after the fact. Who does the encoding? At what cost? With what quality assurance? This is the bottleneck that determines whether any approach ships, and every approach hand-waves it."

---

## Round 3: Synthesis Points of Agreement

Despite fierce disagreement, the experts converge on several points:

1. **Infeasibility detection with MUS-mapped regulatory diagnoses is the highest-value capability.** All four experts agree this is commercially irreplaceable and academically defensible (Skeptic: "comes closest to viable"; Math: "M1 is load-bearing"; Difficulty: "MUS + provenance is medium-hard"; Visionary: "the headline").

2. **The regulatory encoding bottleneck is under-weighted in all approaches.** The 300-article corpus is the project's foundation and its greatest vulnerability. External validation (Amendment A5) is necessary but insufficient — the creation process needs a plan.

3. **The formalizability completeness theorem (B-M1) is the most intellectually distinctive contribution.** Even the Skeptic concedes this is a novel *framing* (though challenges the math depth). The Mathematician agrees it's moderate novelty in framing with low novelty in technique. All agree: if the tightness argument is non-trivial, this elevates the work.

4. **Approach C's verification is too risky for the regulatory timeline.** Even the Mathematician (who scores C highest on math depth) concedes the risk-adjusted value is comparable to A. The Skeptic's timeline argument is accepted by all.

5. **ILP baseline-first is non-negotiable.** The depth check mandated this (Amendment A4), and all experts agree. The ILP vs. MaxSMT comparison is an independently valuable evaluation axis.

6. **Resolution guidance must be added to the output.** The Skeptic's demand (S1) that the tool provide *resolution options* (not just conflict identification) was accepted by the Visionary (V1) as a straightforward extension.

7. **Scope must be ruthlessly controlled.** Both the Skeptic and Difficulty Assessor agree: the winning approach should do fewer things at higher quality. 11 components is too many. The viable paper covers 4–5 core components done well.
