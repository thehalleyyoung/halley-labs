# Theory Verification Signoff: SafeStep

**Project:** SafeStep — Verified Deployment Plans with Rollback Safety Envelopes for Multi-Service Clusters  
**Stage:** Theory  
**Reviewer:** Verification Chair (independent review)  
**Artifacts Reviewed:**  
- `theory/approach.json` (structured theory document, 536 lines JSON)  
- `theory/paper.tex` (full publication, 2030 lines LaTeX)  
- `ideation/depth_check.md` (ideation verification gate)  
- `ideation/final_approach.md` (synthesized approach)  

**Date:** 2026-03-08

---

## 1. Theorem-by-Theorem Verification

### Theorem 1: Monotone Sufficiency

**Statement:** Under bilateral downward closure on compatibility, any safe deployment plan can be transformed into a monotone (no-downgrade) plan of equal or shorter length.

**Precision and correctness:** The statement is correct *after* the mid-proof correction to bilateral DC. The original Definition (unilateral DC on the second argument) is insufficient — the proof discovers this in real time and strengthens the assumption. This is honest and well-handled in the paper (Remark following Theorem 1 acknowledges the strengthening). However, the fact that the proof had to be corrected mid-stream in a publication artifact is a **yellow flag** — it suggests the theorem was not fully pressure-tested before writing. The final statement under bilateral DC is sound.

**Proof sketch quality:** The exchange argument is the right technique. The proof in paper.tex is unusually detailed for a systems paper (~50 lines) and works through the subtle cases carefully. The key step — showing that the modified plan states are componentwise ≤ some original plan state, then applying bilateral DC — is correct. The termination argument (each elimination removes ≥2 steps) is clean.

**Gaps:** The bilateral DC strengthening changes the empirical coverage claim. The original ">92% of pairs satisfy DC" was measured under unilateral DC. Bilateral DC is strictly stronger. The paper does not re-quantify the empirical prevalence of *bilateral* DC. This is a **gap that must be closed** — if bilateral DC prevalence is significantly lower (say 75%), the monotone reduction applies to fewer pairs and the search space expands.

**Load-bearing?** ESSENTIAL. Without this theorem, the search is PSPACE-hard (arbitrary paths in exponential graph). With it, the problem is NP-complete with a tight completeness bound. Removing it would make the system computationally infeasible.

**Verdict: PASS with caveat** — bilateral DC prevalence must be empirically quantified.

---

### Corollary 1: BMC Completeness Bound

**Statement:** Under Theorem 1's conditions with atomic upgrades, k* = Σ(goal[i] − start[i]) is the BMC completeness threshold.

**Precision and correctness:** Correct and straightforward. A monotone plan advances each service at most goal[i] − start[i] times, so total length ≤ k*. The proof is one paragraph and airtight.

**Load-bearing?** ESSENTIAL. This converts BMC from semi-decision to complete decision procedure. Without it, SafeStep cannot certify plan *non-existence* — it could only say "no plan found within horizon k" without knowing if a longer plan exists. The concrete bound k* ≈ 200 is modest enough for SAT.

**Verdict: PASS** — clean, correct, no issues.

---

### Theorem 2: Interval Encoding Compression

**Statement:** Under interval structure, BMC constraints encode in O(log|V_i| · log|V_j|) clauses per pair per step, versus O(|V_i| · |V_j|) naively.

**Precision and correctness:** The statement is correct. The proof via binary encoding → multiplexer tree → comparator circuit → Tseitin transformation follows standard SAT encoding techniques. Step 3 (multiplexed comparison with monotone lo(·)/hi(·)) is the key insight — monotonicity of bound functions simplifies the mux tree. The clause count analysis is sound: O(b²) = O(log²L) per pair per step.

**Proof sketch quality:** Good. The five-step decomposition is clear and each step is standard. The concrete estimate (14.4M clauses at n=50, L=20, k=200, f=0.08) is carefully computed and consistent with the formula.

**Gaps:** The proof assumes lo(·) and hi(·) are monotone (stated as following from bilateral DC + interval structure). This is plausible but not proven — it should be stated as an additional condition or formally derived. If lo(·) is non-monotone, the mux tree doesn't simplify, and the clause count becomes O(b · 2^b) = O(L log L) rather than O(b²) — still better than O(L²) but weaker than claimed.

**Load-bearing?** ESSENTIAL. Without interval compression, encoding size at target scale is ~73.5M clauses. With it, ~14.4M. Both are technically feasible for CaDiCaL, but solver performance degrades significantly at 70M+. The compression provides a comfortable margin.

**Theorem on Encoding Sensitivity to Non-Interval Fraction:** Correct and useful. The crossover point f* ≈ 0.047 for L=20 means that even a small fraction of non-interval pairs contributes significantly. At empirical f ≈ 0.08, the non-interval tail accounts for ~41% of total clauses. This is honest accounting.

**Verdict: PASS** — minor concern about monotonicity of bound functions being implicit.

---

### Theorem 3: Treewidth FPT Tractability

**Statement:** Safe deployment plan existence is decidable in O(n · L^{2(w+1)} · k*) time and O(n · L^{2(w+1)}) space via DP on tree decomposition.

**Precision and correctness:** The statement is correct. The DP follows the standard template for treewidth-based algorithms (nice tree decomposition, four node types). The exponent 2(w+1) comes from tracking both current and target versions for w+1 services per bag — this is correct and tight (the target assignment is needed for the completeness bound).

**Proof sketch quality:** Adequate. The four-case analysis (leaf, introduce, forget, join) is standard. The k* factor for path length tracking is noted but not fully elaborated — it amounts to adding a "current step count" dimension to the DP table, which multiplies the state space by k*. The proof is sufficient for a systems paper.

**Feasibility table:** The approach.json correctly marks tw≥5 as infeasible and restricts the DP to tw≤3, L≤15. The paper's Table 1 is consistent: 15^8 ≈ 2.6×10⁹ at tw=3, L=15 (seconds), vs 20^12 ≈ 4.1×10¹⁵ at tw=5, L=20 (infeasible). This is an honest presentation — the DP is positioned as a narrow fast path, not a general solution.

**Concern:** The claim that "treewidth structure still guides SAT variable ordering for tw≥4, providing 1.5–3× speedup" is asserted without evidence. This is a soft claim and not load-bearing, but it should be evaluated empirically.

**Load-bearing?** ENABLING, not essential. The system works without the DP (SAT/BMC handles all cases). The DP is a performance optimization for a specific regime. Removing it would slow some cases from seconds to minutes but would not break correctness.

**Verdict: PASS** — well-scoped, honestly presented.

---

### Theorem 4: CEGAR Soundness and Termination

**Statement:** The CEGAR loop between CaDiCaL and Z3 is sound, refutation-complete, and terminates in ≤ min(2^R, |V_BMC|) iterations.

**Precision and correctness:** The soundness and refutation-completeness proofs are straightforward and correct — they follow directly from the CEGAR template. The termination bound min(2^R, |V_BMC|) is correct but vacuously large: 2^R for typical resource encodings is astronomical (R could be hundreds of bits). The practical claim of 5–20 iterations relies entirely on blocking clause generalization, which is not formally analyzed.

**Proof sketch quality:** Standard CEGAR correctness argument. No novel contribution beyond the domain-specific instantiation. The paper acknowledges this: "CEGAR is the computational mechanism, not the contribution."

**Gaps:** The GeneralizeBlocking procedure (Algorithm 3, line 13) is critical for practical convergence but only informally described. If generalization fails to eliminate large regions, the iteration count could be much higher than 5–20. The paper should include at minimum a concrete example of generalization.

**Load-bearing?** ESSENTIAL for systems with resource constraints. Without CEGAR, SafeStep cannot handle CPU/memory limits alongside API compatibility. In clusters where resource constraints are non-binding (many microservice deployments), CEGAR is unnecessary.

**Verdict: PASS** — standard result correctly applied; practical convergence claim needs empirical validation.

---

### Theorem 5: Adversary Budget Bound

**Statement:** Under independent oracle errors with confidence-bounded probabilities, k_α = ⌈μ + z_α · σ⌉ gives a probabilistic safety guarantee.

**Precision and correctness:** The mathematical statement is correct under the independence assumption. The one-sided Chebyshev/CLT application is standard. However, there is a subtle logical issue: **the theorem as stated verifies safety under all size-k_α subsets of Red-tagged constraints, but the budget k_α is computed over ALL constraints (including Green and Yellow).** The paper addresses this in the proof (step 5: "restricting enumeration to red-tagged constraints... accounts for the green/yellow tail") but the accounting is hand-wavy. A rigorous treatment would condition on the number of green/yellow errors and show the residual probability is small.

**More seriously:** The independence assumption is acknowledged as unrealistic (systematic schema analyzer failures violate it). The fallback "set k = |red| and check the single all-red subset" is computationally trivial but destroys the probabilistic guarantee — it becomes a worst-case check with no probabilistic interpretation. The theorem's practical value is therefore limited to the regime where independence approximately holds.

**Load-bearing?** LOAD-BEARING for the k-robustness certification claim. Without it, the choice of k is unprincipled. However, the practical implementation (k=2 by default, ≤45 SAT calls) works regardless of whether the theoretical bound is tight. The theorem provides justification rather than driving the algorithm.

**Verdict: PARTIAL PASS** — mathematically correct under stated assumptions; practical value limited by unrealistic independence assumption; hand-wavy restriction to Red constraints.

---

### Proposition A: Problem Characterization

**Statement:** Safe plan existence ↔ connectivity in G[Safe]; general case is PSPACE-hard; monotone case is NP-complete.

**Precision and correctness:** Part (1) is definitional. Part (2) reduces from Aeolus PSPACE-completeness — the reduction is plausible but the paper.tex proof sketch is loose: it claims Aeolus' configuration-to-configuration reachability is "a special case of version-product graph reachability," but the mapping of Aeolus' require/provide constraints to SafeStep's pairwise compatibility is non-trivial and not fully detailed. Part (3) NP membership via k* bound is correct; NP-hardness via 3-SAT reduction is well-sketched — each service maps to a variable, versions {0,1} to truth values, clauses to compatibility constraints. The claim that DC "holds vacuously because |V_i| = 2" is correct.

**Important caveat in paper.tex (§7.2):** The Aeolus/Zephyrus comparison explicitly notes that the PSPACE vs NP comparison is apples-to-oranges ("the comparison is *not* 'we reduced PSPACE to NP'"). This directly addresses depth-check amendment #7. Well done.

**Load-bearing?** STRUCTURAL — establishes the formal landscape but doesn't drive any algorithm directly.

**Verdict: PASS** — good framing with honest caveats.

---

### Proposition B: Replica Symmetry Reduction

**Statement:** Per-service state space collapses from L^r to O(L²) via symmetric representation.

**Precision and correctness:** Correct. The interchangeability argument for pods running the same container image is sound. The counting is straightforward: O(r+1) states per version pair × O(L²) version pairs = O(r·L²) ≈ O(L²) for constant r.

**Concern:** The proposition assumes a service has exactly two active versions during rolling update (old and new). Multi-step rolling updates (v1 → v2 → v3 with mixed pod populations at v1, v2, and v3 simultaneously) are not modeled. This is reasonable for a single upgrade step but limits composability with multi-version histories.

**Load-bearing?** ENCODING OPTIMIZATION — useful for scalability but not essential for correctness.

**Verdict: PASS** — clean, modest, correctly scoped.

---

## 2. Depth Check Amendment Compliance

### Amendment 1: Oracle validation experiment designed
**Verdict: PASS**

The theory stage delivers a fully designed oracle validation experiment (paper.tex §6.1, approach.json phase_0_oracle_validation). The design includes:
- 15 postmortems from named sources (Google, AWS, Cloudflare, Meta, Uber, danluu.com)
- Four-category taxonomy (Structural-Detectable, Structural-Ambiguous, Behavioral, Operational)
- Inter-rater reliability with Cohen's κ target
- Clear decision criteria (≥60% proceed, 40-60% proceed cautiously, <40% pivot)
- Sample size analysis acknowledging wide CI (±12%)
- Additional deliverable: bilateral DC quantification

This is a thorough experimental design. Execution is deferred to implementation, which is appropriate.

### Amendment 2: Clause count corrected to O(n²·log²L·k)
**Verdict: PASS**

The paper consistently uses O(n² · log²L · k) throughout (Theorem 2, Remark on Concrete Encoding Size, Algorithm 1 complexity). The approach.json states the correct formula. The concrete estimate (14.4M clauses at target parameters) uses the corrected formula with the non-interval fraction accounting (Theorem on Encoding Sensitivity). The body text no longer contains the old O(n² · log L) formula. The 4.3× discrepancy identified in the depth check is resolved.

### Amendment 3: Treewidth DP feasibility boundary explicit
**Verdict: PASS**

The feasibility boundary is explicit in both artifacts:
- approach.json: feasibility_table with tw≤3/L≤15 (seconds), tw=4/L≤8 (minutes), tw≥5 (infeasible)
- paper.tex: Table 1 with identical boundaries
- paper.tex Remark: "At w=5, L=20: L^{2(w+1)} = 20^{12} ≈ 4.1×10^{15} — completely infeasible"
- approach.json: "The DP is a fast path for a specific regime, not a general solution"
- SAT/BMC is presented as the primary method, DP as narrow optimization

The depth check's specific concern (claiming tw=3-5 was feasible) is fully addressed.

### Amendment 4: helm template subprocess (not reimplementation)
**Verdict: PASS**

Both artifacts specify `helm template` subprocess:
- paper.tex §5.6: "Manifest parsing delegates to the helm template CLI subprocess for correctness—reimplementing Go's template engine in Rust would create an enormous surface for semantic divergence."
- approach.json implementation_strategy: uses helm template subprocess
- final_approach.md explicitly lists Helm reimplementation as "REJECTED from A"

### Amendment 5: "Structurally verified" language used
**Verdict: PASS**

The paper uses "structurally verified relative to modeled API contracts" consistently:
- Abstract: "SafeStep's guarantees are structurally verified relative to modeled API contracts"
- §1.4 (Honest Framing): "We do not claim 'formally verified' or 'provably safe' in an unqualified sense."
- §8 (Conclusion): "structurally verified relative to modeled API contracts"
- §7 (Limitations): first subsection is "Oracle Coverage Gaps"
- The confidence coloring system (GREEN/YELLOW/RED) makes uncertainty visible throughout

I found zero instances of unqualified "formally verified" in the paper. This amendment is fully satisfied.

### Amendment 6: Methodology for empirical claims provided
**Verdict: PARTIAL PASS**

The oracle validation experiment (Phase 0) provides methodology for the 15-postmortem oracle coverage claim. The bilateral DC quantification is included as an additional Phase 0 deliverable. However:

- The ">92% interval structure" claim: The approach.json states it is "an empirical finding requiring methodology disclosure" and the paper's Phase 0 additional deliverable covers DC quantification, but **the interval structure methodology (from 847 projects) is not explicitly described in the paper.** The paper references it as empirical (§1.3: "empirically satisfied by >92% of pairwise constraints in a corpus of 847 open-source microservice projects") but does not provide the methodology for how these 847 projects were selected, how compatibility predicates were extracted, or how interval structure was verified. This is an important gap.

- The "18-23% of outages involve version incompatibility" and "median treewidth 3-5": These are referenced in approach.json and final_approach.md but I don't see explicit methodology in paper.tex. The treewidth claim appears in the related work section (§7.5: "median 3-5 in our 847-project dataset") but without methodology.

The methodology is partially designed (Phase 0) but not fully specified for all three empirical claims.

### Amendment 7: Aeolus/Zephyrus comparison caveated
**Verdict: PASS**

Paper.tex §7.2 contains an explicit, detailed caveat in bold:

> "Critical distinction. Aeolus and Zephyrus solve configuration synthesis: finding a valid target state. SafeStep solves plan synthesis: finding a safe path..."

> "Complexity comparison (important caveat). Aeolus' PSPACE-completeness applies to a different problem... SafeStep's NP-completeness under monotonicity applies to a restricted problem... The comparison is not 'we reduced PSPACE to NP' — it is 'our restricted problem, which captures the common case for version upgrades, admits an NP decision procedure.' The restriction is the key enabler."

This is a model caveat — clear, honest, and preemptive against the "apples-to-oranges" objection.

---

## 3. Scoring

| Dimension | Score | Justification |
|-----------|-------|---------------|
| **V (Value)** | **7** | The rollback safety envelope is a genuinely novel operational concept with clear use cases (SRE mid-incident rollback decisions, compliance auditing). The oracle validation experiment design is rigorous. The honest framing of limitations (confidence coloring, "structurally verified" language) actually increases value — operators trust tools that are honest about what they can't do. The paper's positioning (complement to canaries, not replacement) is smart. Elevation from ideation's 6 reflects the concrete evaluation design and the paper's compelling narrative (the 3:47 AM scenario). |
| **D (Difficulty)** | **7** | The theory stage demonstrates genuine mathematical substance: the exchange argument for Theorem 1 (including the mid-proof correction to bilateral DC) is non-trivial; the interval encoding proof is detailed and correct; the CEGAR integration is standard but carefully applied; the end-to-end soundness chain is explicit. The paper contains 5 theorems + 2 propositions + 1 corollary, all with full proofs. The algorithm section maps each theorem to pseudocode with complexity analysis. This is substantially above a "just engineering" contribution. Elevation from ideation's 6 reflects the detailed proofs and the bilateral DC discovery. |
| **BP (Best Paper %)** | **6** | The rollback safety envelope concept has "permanent vocabulary" potential. The paper is well-written with a compelling opening (the 3:47 AM scenario) and honest throughout. However: (1) all techniques are borrowed (BMC, CEGAR, treewidth DP), (2) the evaluation is not yet executed, (3) the strongest validation would be finding a real bug in an existing benchmark — this is designed but not demonstrated. The paper would be a strong accept at EuroSys/NSDI/ICSE; best paper requires either a killer evaluation result or a deeper theoretical contribution. The probability of best paper at a top venue is 10-15%. |
| **L (Laptop feasibility)** | **7.5** | SAT solving at 14.4M clauses is well within CaDiCaL's laptop capacity (routinely handles 50M+). The treewidth DP is honestly scoped to tw≤3. The envelope computation requires ~8 incremental SAT calls via binary search. The CEGAR loop typically converges in 5-20 iterations. k-robustness at k=2 adds 45 lightweight checks. Total pipeline for n=50, L=20: estimated 3-5 minutes on a modern laptop. The paper provides a clear performance tier table. Elevation from ideation's 6.5 reflects the corrected clause count (14.4M vs the originally inconsistent figures) and explicit feasibility analysis. |
| **F (Feasibility)** | **7** | The theory-to-implementation mapping is explicit: each theorem maps to a named Rust module with LoC estimates. The technology choices are sound (CaDiCaL via FFI, Z3 for LIA, helm template subprocess). The evaluation plan has four concrete phases with named benchmarks (DeathStarBench), baselines (Fast Downward, topological sort), and statistical methodology. The primary risk (oracle coverage <40%) has a designed experiment and a pivot plan. The honest LoC estimate (~45-65K core) is achievable. |

---

## 4. Critical Issues Found

### SERIOUS: Bilateral DC Prevalence Unquantified

The proof of Theorem 1 (the most important theorem) requires *bilateral* downward closure, not the originally stated unilateral DC. The empirical prevalence of >92% was measured under the weaker unilateral condition. Bilateral DC is strictly stronger — it requires that the compatible set {(v,w) : C(i,j,v,w)} is a downward-closed rectangle in V_i × V_j. The paper does not re-quantify prevalence under bilateral DC.

If bilateral DC prevalence is 85% rather than 92%, the monotone reduction applies to fewer pairs. If it drops below 70%, the practical impact on search space could be significant. The Oracle Validation Phase 0 should include bilateral DC quantification (approach.json mentions this), but this must be made an explicit, non-optional deliverable.

**Rating: SERIOUS** — directly affects the load-bearing theorem.

### MODERATE: Envelope Prefix Property Assumed Without Proof

Algorithm 4 uses binary search to find the first PNR state, relying on the claim that the envelope is a prefix of the plan (if s_t is backward-reachable to s_0, then s_{t-1} is also backward-reachable). The paper states this follows from monotone plans (s_{t-1} ≤ s_t componentwise).

But this argument is incomplete: backward reachability from s_{t-1} to s_0 requires a *backward* path through safe states. The fact that s_{t-1} ≤ s_t componentwise doesn't directly imply that a backward path from s_{t-1} exists whenever a backward path from s_t exists — the backward paths might use different intermediate states. Under bilateral DC, any backward path from s_t passes through states componentwise ≤ s_t, and s_{t-1} ≤ s_t, so states reachable backward from s_{t-1} form a subset of those reachable from s_t... actually, this is the *wrong direction*. The reachable-from set from s_{t-1} is potentially smaller than from s_t.

The correct argument needs: "s_{t-1} ≤ s_t, and the start state s_0 is reachable backward from s_t, therefore s_0 is reachable backward from s_{t-1}." This requires showing that any backward path from s_t to s_0 can be adapted to start from s_{t-1} instead. Under bilateral DC, this holds because all states on the backward path are componentwise ≤ s_t, and since s_{t-1} ≤ s_t, we can begin the path at s_{t-1} instead — but we need to verify that the first step (from s_{t-1} to the next state on the backward path) is still valid.

This is likely provable but the argument is not trivially obvious and should be formalized as a lemma.

**Rating: MODERATE** — the binary search optimization is important for performance but not for correctness; linear scan is a valid fallback.

### MODERATE: 847-Project Dataset Methodology Missing

The paper repeatedly references empirical findings from "847 open-source microservice projects" (interval structure prevalence, treewidth distribution) but provides no methodology for how this dataset was constructed, how compatibility predicates were extracted, or how interval structure was verified. This is a core supporting claim for two theorems (Theorem 1's DC prevalence, Theorem 2's interval structure prevalence).

**Rating: MODERATE** — methodology must be provided before submission.

### LOW: Non-Interval Encoding Fallback Under-Specified

The paper describes BDD encoding as the fallback for non-interval pairs (Algorithm 1, line 10) but does not provide the BDD encoding algorithm or complexity analysis. For the ~8% of non-interval pairs, the paper uses "naive O(L²)" as the cost, but a BDD encoding could be more efficient depending on the structure of the non-interval predicate. This is not a correctness issue — the naive encoding is correct — but it leaves performance uncertain for edge cases.

**Rating: LOW** — implementation detail, not a theoretical gap.

### LOW: Theorem 5 Probabilistic Guarantee is Weak in Practice

The Remark on Practical Parameters (§4.5) reveals a disconnect between theory and practice: the theoretical budget k_α = 10 requires enumerating C(60,10) subsets (enormous), so the system defaults to k=2 with 45 subsets. The gap between the theoretical guarantee (need k=10 for 95% confidence) and the practical implementation (k=2) is large. The paper acknowledges this but the probabilistic safety guarantee is effectively decorative for the practical system.

**Rating: LOW** — the honest acknowledgment mitigates this, and the practical k=2 check has standalone value.

---

## 5. Comparison to Ideation Gate

| Dimension | Ideation | Theory | Delta | Commentary |
|-----------|----------|--------|-------|------------|
| V (Value) | 6 | 7 | +1 | Concrete evaluation design with Phase 0 oracle gate; compelling narrative; honest framing increases operator trust |
| D (Difficulty) | 6 | 7 | +1 | Full proofs delivered; bilateral DC discovery shows genuine mathematical engagement; exchange argument is non-trivial |
| BP (Best Paper %) | 5.5 | 6 | +0.5 | Paper is well-structured and honest; "permanent vocabulary" potential of envelope concept is articulated; techniques still borrowed |
| L (Laptop) | 6.5 | 7.5 | +1 | Corrected clause count (14.4M) with explicit feasibility analysis; binary search envelope optimization; honest treewidth boundaries |
| F (Feasibility) | — | 7 | new | Explicit theorem-to-module mapping; sound technology choices; pivot plan for oracle failure |

**Overall trajectory: Positive.** The theory stage addresses the ideation gate's major concerns (clause count, treewidth, Helm reimplementation, honest language) and adds genuine substance through detailed proofs and evaluation design. The bilateral DC discovery is a sign of honest mathematical engagement — the proof found a real issue and corrected it. The paper quality is substantially higher than typical ideation-to-theory transitions.

---

## 6. Final Verdict

### **CONDITIONAL CONTINUE**

**Conditions (must be satisfied before implementation begins):**

1. **Bilateral DC prevalence quantification.** The 847-project dataset must be re-analyzed under bilateral DC (not just unilateral). Report the prevalence explicitly. If bilateral DC prevalence <80%, provide a graceful degradation analysis showing the search space expansion is manageable.

2. **Envelope prefix property formal argument.** Provide a 3-5 sentence proof that the rollback safety envelope under monotone plans is a prefix (enabling binary search). This can be a lemma in the appendix.

3. **847-project dataset methodology.** Provide at minimum: (a) how projects were selected, (b) how compatibility predicates were extracted, (c) how interval structure was verified, (d) the bilateral DC satisfaction rate. This can be a supplementary document or appendix section.

**Conditions (must be satisfied before submission):**

4. **Empirical claims backed by reproducible methodology.** The three empirical claims (>92% interval structure, 18-23% outage rate, median treewidth 3-5) must all have documented, reproducible methodologies.

5. **Phase 0 oracle validation executed.** Results determine paper positioning (deployment tool vs. theory contribution).

**Justification:** The theory stage delivers on its core promise: a well-defined formal framework with 5 theorems + 2 propositions, detailed proofs, a concrete evaluation design, and honest accounting of limitations. The rollback safety envelope is a genuinely novel concept with lasting value. The paper quality is high — it reads as a real submission, not a prototype. The issues found (bilateral DC prevalence, envelope prefix property, dataset methodology) are addressable within the normal development cycle and do not require fundamental rethinking. The risk profile is acceptable: 25% kill probability (primarily from oracle coverage), 60-65% publication probability at a strong venue. 

The project is on track for a solid contribution to EuroSys/NSDI/ICSE, with outside-shot potential at SOSP/OSDI if the evaluation delivers a compelling result (finding a real unsafe rollback state in DeathStarBench).

---

## 7. Red-Team Attack Vectors

### Attack 1: "This is just McClurg et al. (PLDI 2015) for Kubernetes"

**Objection:** The techniques (BMC, CEGAR, incremental SAT) are standard model-checking techniques applied to a new domain. The version-product graph is structurally identical to SDN's switch-configuration graph. This is a domain translation, not a research contribution.

**Defense:** The rollback safety envelope is genuinely absent from SDN work — no prior system computes bidirectional reachability for backward safety analysis. The interval encoding compression exploits domain-specific structure (version-ordered compatibility has contiguous compatible ranges) that has no analogue in SDN forwarding rules. The oracle confidence model with k-robustness is a qualitatively different trust model than SDN's assumption of perfect rule knowledge. The paper explicitly acknowledges the SDN parallel (§7.4) and argues the contributions are extensions, not translations. Strength: 7/10 — this is the most dangerous attack. Mitigated by the novelty of the envelope concept and honest SDN comparison.

### Attack 2: "The guarantees are as weak as the oracle, which you admit catches <100% of failures"

**Objection:** All formal guarantees are conditional on oracle correctness. The oracle is schema-based and misses behavioral incompatibilities. What good is a "structurally verified" plan if the structural model covers only 60-70% of real failure modes? This is formal methods theater — giving operators false confidence through mathematical notation applied to an incomplete model.

**Defense:** (1) Every static analysis tool in production (type checkers, linters, race detectors) has this property — incomplete coverage does not imply zero value. (2) The confidence coloring system makes uncertainty *visible*, not hidden. (3) The paper does not claim unqualified safety — "structurally verified relative to modeled API contracts" is the consistent framing. (4) The oracle validation experiment quantifies the gap rather than hiding it. (5) Even at 60% coverage, catching 60% of rollback failures preemptively is operationally valuable — the alternative is catching 0%. Strength: 6/10 — valid but addressed by honest framing.

### Attack 3: "15 postmortems is not a statistically meaningful sample"

**Objection:** The oracle validation experiment uses n=15 postmortems, giving a 95% CI of ±12%. The incident reconstruction also uses n=15. These sample sizes are too small for any confidence in the coverage claims. Furthermore, public postmortems are biased toward spectacular failures, not representative of the population of deployment incidents.

**Defense:** (1) The paper acknowledges the wide CI and frames Phase 0 as a pilot study. (2) The primary evaluation is Phase 1 (prospective, controlled, on DeathStarBench), not Phase 0. (3) Expanding to n=30 narrows the CI to ±9% — feasible if more postmortems are available. (4) The selection bias concern is valid but cuts both ways — spectacular failures may be *more* likely to involve structural incompatibilities (precisely SafeStep's strength). Strength: 5/10 — valid methodological concern but not fatal given the prospective evaluation.

### Attack 4: "The bilateral DC strengthening undermines confidence in the formalization"

**Objection:** The proof of Theorem 1 — the most important theorem — required a mid-proof correction from unilateral to bilateral downward closure. If the authors discovered this gap during proof writing, what other gaps exist in the formalization? The bilateral DC requirement is strictly stronger and may significantly reduce empirical coverage. This suggests the theoretical foundations were not fully verified before the paper was written.

**Defense:** (1) Discovering and correcting the assumption mid-proof is a sign of honest mathematical engagement, not sloppiness. (2) The correction is explicitly documented (Remark following Theorem 1). (3) Bilateral DC is a natural condition for version-ordered compatibility ("older versions are universally more compatible") and is likely satisfied by most real constraints. (4) The paper's Phase 0 includes bilateral DC quantification as a deliverable. (5) Graceful degradation for non-DC pairs is designed (non-monotone BMC for violating pairs). Strength: 6/10 — legitimate concern; mitigated by honest disclosure and planned quantification.

### Attack 5: "The evaluation baseline comparison is unfair — Fast Downward solves a different problem"

**Objection:** The paper compares SafeStep against Fast Downward (a PDDL planner), but Fast Downward doesn't compute rollback safety envelopes — it only finds forward plans. Comparing plan synthesis time is meaningful, but the envelope computation (SafeStep's primary contribution) has no baseline comparison. Additionally, topological sort and random plans are straw-man baselines that no serious deployment tool uses.

**Defense:** (1) There is no existing tool that computes rollback safety envelopes — the concept is novel, so baseline comparison for the envelope is impossible by definition. (2) Fast Downward is the closest automated planning tool and provides a meaningful comparison for the plan synthesis component. (3) Topological sort *is* what teams actually use in practice — manual runbooks ordered by dependency depth. Calling it a straw man misunderstands the operational reality. (4) The strongest evaluation is not comparative but demonstrative: finding a real unsafe rollback state in DeathStarBench that no baseline detects. Strength: 4/10 — pedantic objection; the novelty of the envelope concept makes baseline comparison inherently impossible.

---

*Assessment produced by independent Verification Chair review. All theorems verified against proof sketches in both approach.json and paper.tex. Scoring reflects honest assessment of current state, not aspirational potential.*
