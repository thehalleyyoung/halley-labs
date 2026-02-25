# Review: CABER — Coalgebraic Behavioral Auditing of Foundation Models

**Reviewer:** Lila Zhang
**Persona:** Symbolic Reasoning and AI Expert
**Expertise:** Formal semantics, automata-based reasoning, coalgebraic methods, symbolic-neural integration, behavioral verification

---

## Summary

CABER applies coalgebraic semantics to the problem of black-box LLM behavioral verification. The framework treats LLMs as coalgebras over a behavioral endofunctor, learns finite approximations via active queries, and verifies temporal properties using a custom model checker. The coalgebraic framing is the most intellectually rigorous approach to LLM behavioral analysis I have seen, providing a mathematical foundation for concepts (behavioral equivalence, abstraction, distance) that are typically handled informally. However, the system currently operates only on toy mock models, and the gap between the theoretical machinery and practical applicability is substantial.

## Strengths

1. **Coalgebraic semantics provides the right abstraction level.** Treating LLMs as state-based systems with observable behavior via a functor F: Set → Set captures the essential structure while abstracting away implementation details. This is fundamentally more principled than ad-hoc behavioral testing frameworks.

2. **Functor bandwidth is an elegant complexity measure.** Defining behavioral complexity via ε-covering numbers of the functor image in Kantorovich metric space is a beautiful connection between metric geometry and automata theory. The sublinear growth (β ≪ |Σ|) demonstrates that behavioral diversity is much lower than lexical diversity — a key insight for tractable verification.

3. **QCTL_F with graded satisfaction is expressive and appropriate.** Graded satisfaction degrees in [0,1] naturally handle the probabilistic nature of LLM behavior. The specification templates translate real safety concerns into formal properties in a natural way.

4. **Kantorovich bisimulation distance is the canonical metric.** The choice of Kantorovich lifting over alternatives (total variation, Wasserstein) is correct for the coalgebraic setting and provides compositional distance computation.

5. **The theoretical framework is internally consistent.** Theorems 1-3 build on each other coherently: convergence guarantees → sample complexity → error composition. The proofs use standard techniques (Hoeffding, union bound, sequential conditioning) appropriately.

## Weaknesses

1. **The theory-practice gap is enormous.** The theoretical framework handles infinite-state systems in principle, but the implementation learns 19-40 states from 3-6 state mock models. Real LLMs exhibit context-dependent behavior over sequences of arbitrary length — the finite-state approximation may require an impractically large number of states to capture this.

2. **Non-functorial alphabet abstraction breaks the coalgebraic foundation.** The embedding-based clustering that maps LLM outputs to a finite alphabet is acknowledged as non-functorial. This means the learned automaton's relationship to the original LLM is not captured by the coalgebraic framework's homomorphism theorems. The entire theoretical edifice rests on an approximation whose quality is unknown.

3. **No comparison to simpler approaches on the same mock models.** Would a standard PDFA + PRISM pipeline achieve comparable results on the same mock models? The paper claims structural advantages but demonstrates no scenario where the coalgebraic machinery provides empirically different outcomes than simpler alternatives.

4. **The specification templates are not validated against real safety failures.** The six templates (RefusalPersistence, SycophancyResistance, etc.) are plausible but not empirically grounded in actual LLM safety incidents. Do real refusal failures actually correspond to violations of the RefusalPersistence temporal formula?

5. **The certificate format is not standardized.** The AuditCertificate struct includes a hash chain and timestamp, but there is no discussion of certificate interoperability, third-party verification, or integration with existing audit frameworks.

## Novelty Assessment

The coalgebraic framing of LLM behavioral analysis is highly original. The PCL* algorithm and functor bandwidth concept are genuine theoretical contributions. However, the application of coalgebraic model checking to verification is well-established in the programming languages community — the novelty lies in the LLM application domain, not in the coalgebraic methods themselves. **High application novelty, moderate methodological novelty.**

## Suggestions

1. Demonstrate a scenario where the coalgebraic machinery detects a behavioral property that PDFA + PRISM misses, to justify the additional complexity.
2. Investigate continuity-based approaches to characterize the alphabet abstraction error.
3. Ground the specification templates in actual documented LLM safety failures.
4. Consider a minimal real-LLM experiment (e.g., a fine-tuned GPT-2 with known behavioral properties) to bridge the theory-practice gap.

## Overall Assessment

CABER represents the most theoretically rigorous approach to LLM behavioral verification I have reviewed. The coalgebraic framing, functor bandwidth, and error composition framework are genuine intellectual contributions. However, the gap between the mathematical beauty of the theory and the toy-scale validation is concerning. The non-functorial alphabet abstraction is a fundamental weakness that undermines the theoretical guarantees. The work needs real-LLM validation and a concrete demonstration of coalgebraic advantages over simpler approaches.

**Score:** 7/10
**Confidence:** 4/5
