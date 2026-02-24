# Review by Joseph S. Chang (automated_reasoning_and_logic_expert)

## Project: LITMUS∞: Cross-Architecture Memory Model Portability Checker

**Reviewer Expertise:** Automated reasoning, proof theory, decision procedures, SMT solving, theorem proving. Focuses on logical foundations, proof correctness, and reasoning completeness.

**Overall Score:** weak accept

---

## Summary

LITMUS∞ reduces cross-architecture portability checking to a finite inclusion problem over RF×CO outcome spaces, solved by exhaustive enumeration and cross-validated by Z3. The three paper-proof theorems provide conditional guarantees. The 55 UNSAT certificates are the project's strongest artifacts, but the theorems themselves have logical gaps that a careful proof-theoretic reading reveals. This review focuses on whether the formal claims are sound, whether the proof methodology is rigorous, and what logical improvements would elevate the contribution.

## Strengths

1. **The decision problem formulation is correct.** Definition 2 (Portability Safety) correctly formalizes the problem: test T is safe from M_A to M_B iff every M_B-consistent execution producing the forbidden outcome has an M_A-consistent counterpart. The contrapositive formulation is clean. For finite litmus tests, this reduces to a finite model checking problem, making exhaustive enumeration sound and complete. This is the right theoretical foundation.

2. **The 55 UNSAT certificates are genuine proof artifacts.** Each UNSAT result from Z3 constitutes a refutation proof: Z3's DPLL(T) procedure has explored the complete Boolean structure of the SMT formula and found no satisfying assignment. Modulo Z3's own correctness (which is extensively validated), these are logically irrefutable. The 40 SAT witnesses similarly provide concrete counterexample executions. Together, these 95 certificates are the project's only claims that are truly machine-checked.

3. **Honest scoping of theorem conditions.** Theorem 1's "provided the tool's model is at least as permissive" caveat, Theorem 2's "within the fixed fence menu" scope, and Remark 4's mechanization disclosure all represent appropriate intellectual honesty about the boundaries of the formal claims.

## Weaknesses

1. **Theorem 1's proof is incomplete.** The proof assumes "the enumeration covers all elements of RF × CO" without proving it. This requires: (a) that every execution uniquely determines an (rf, co) pair (standard in Alglave et al. 2014, but neither cited nor proven), and (b) that portcheck.py actually generates all such pairs (a code-level claim unprovable by paper proof). The 228/228 SMT agreement partially addresses (b), but the proof as written is a sketch.

2. **Theorem 2 is trivially true.** "The algorithm selects argmin_f cost(f) s.t. V_t ⊆ covers(f)" is a restatement of the algorithm's specification, not a theorem. The interesting question — whether V_t is correctly computed — is unaddressed. If V_t misses a violated pair, the fence is insufficient. The Remark acknowledges per-thread minimality ≠ whole-program minimality, so the theorem does not provide the guarantee developers actually need.

3. **Theorem 3's biconditional is incorrect.** The "if and only if" claims conditions (1)+(2) are *necessary* for fence effectiveness. But in real GPU models, ordering can also be restored via dependency chains, release-acquire pairs, or system-scope atomics. The "only if" is false outside the simplified 2-scope model — a scoping that is buried in setup rather than stated as a condition.

4. **SMT encoding correctness is unverified.** The integer-timestamp acyclicity encoding (edge (u,v) implies ts(u) < ts(v)) could produce spurious UNSAT if timestamp constraints become unsatisfiable for reasons other than ghb cyclicity (e.g., tight bounds). The 228/228 agreement reduces but does not eliminate this risk.

5. **Litmus synthesis is overstated.** "Rediscovering MP and LB" from 2-thread, 2-op skeletons means searching at most 4^4 = 256 candidates. This is brute-force over a trivial space, not "independent rediscovery." The claim would be interesting for novel patterns or 3+ thread skeletons where the space is non-trivial.

## Path to Best Paper

(1) Rewrite Theorem 1's proof to cite the RF×CO decomposition lemma from Alglave et al. 2014 and clearly separate the mathematical argument from the implementation correctness claim. (2) Strengthen Theorem 2 to include a correctness argument for V_t, or honestly relabel it as an "algorithm specification" rather than a theorem. (3) Fix Theorem 3's biconditional: either weaken to "if" (sufficient condition) or add the simplified-model scoping as an explicit condition. (4) Mechanize at least Theorem 1 in Lean or Coq — the finiteness of RF×CO makes this particularly tractable. (5) Provide a compositionality result connecting pattern-level and program-level safety. The UNSAT certificates are strong; the paper proofs need to match their quality.
