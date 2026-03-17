# CASCADE CONFIG VERIFIER — DECISION MATRICES & QUICK REFERENCE

---

## TABLE 1: FRAMING COMPARISON MATRIX

| Dimension | CascadeSMT (Monolithic) | MeshCheck (Compositional) | RetryStorm (Focused) | Math-A (TA) | Math-B (Tropical) | Math-C (MaxSAT) |
|-----------|------------------------|--------------------------|-------------------|-------------|-------------------|-----------------|
| **Scalability** | 15-30 services | 100+ services | 30-50 services | 15-30 | 30-50 | 20-40 |
| **Formalization Depth** | Medium | Medium | Medium | High | Very High | Medium |
| **Novelty Ratio** | 30% | 25% | 50% | 35% | 35% | 30% |
| **Implementation Complexity** | High | Very High | Medium | High | Very High | Medium-High |
| **Venue Fit** | OSDI/SOSP | OSDI/SOSP | OSDI/SOSP | Systems | Math-heavy | Systems |
| **Evaluation Data Needed** | Medium | High | Medium | Medium | Very High | Medium |
| **Risk Level** | Medium | Medium-High | Low-Medium | Medium | High | Low |
| **Differentiation from LDFI** | Weak | Medium | Strong | Medium | Weak | Medium |

---

## TABLE 2: PROBLEM SCOPE OPTIONS

| Scope Level | Services | Retry Policies | Timeouts | Circuit Breakers | Rate Limits | Resource Limits | Complexity |
|---|---|---|---|---|---|---|---|
| **Ultra-Narrow** | ≤10 | Yes | Yes | Basic | No | No | Low |
| **Narrow (Recommended)** | ≤50 | Yes | Yes | Full | Basic | No | Medium |
| **Medium** | ≤100 | Yes | Yes | Full | Full | Basic | High |
| **Broad** | ≤200 | Yes | Yes | Full | Full | Full | Very High |
| **Enterprise** | Any | Yes | Yes | Full | Full | Full | Extremely High |

**Recommendation:** Narrow scope (≤50 services, focus on retry×timeout×CB) gives best risk/novelty/effort ratio.

---

## TABLE 3: TECHNICAL CONTRIBUTION BREAKDOWN

| Area | What's Novel | What's Reused | Novelty % | Priority |
|---|---|---|---|---|
| **Problem formulation** | Retry amplification as decidable fragment | Satisfiability, BMC framework | 60% | 🔴 Critical |
| **Config extraction** | Automated K8s/Istio/Envoy parsing | YAML parsing libraries | 40% | 🟠 Important |
| **Cascade modeling** | Timed automata encoding | Alur-Dill theory | 20% | 🟠 Important |
| **Minimal failure discovery** | Antichain-based MinUnsat search | Monotone Boolean functions | 50% | 🔴 Critical |
| **Repair synthesis** | MaxSAT optimization (not CEGIS) | MaxSAT solvers | 30% | 🟠 Important |
| **Compositional decomposition** | Per-service contract lattice | Standard A-G reasoning | 15% | 🟢 Nice-to-have |
| **Evaluation framework** | Postmortem incident reproduction | Benchmark suite construction | 70% | 🔴 Critical |
| **Overall** | | | ~35% | |

---

## TABLE 4: NOVELTY RISK MATRIX

| Risk | If Novelty Weak | If Novelty Medium | If Novelty Strong |
|---|---|---|---|
| **LDFI Overlap** | ⚠️ Paper rejected as "LDFI++" | ✓ Paper acceptable if differentiation clear | ✓ Paper strong |
| **Citation Weight** | "Limited new techniques" | "Novel application + solid engineering" | "Fundamental new insight" |
| **Reviewer Perception** | Incremental | Respectable | High-impact |
| **Mitigation** | Emphasize implementation quality | Emphasize new problem characterization | Emphasize theoretical contribution |

**Our situation:** Medium novelty (35%) + Strong evaluation can succeed at OSDI/SOSP if story is well-told.

---

## TABLE 5: PRIOR ART ACKNOWLEDGMENT REQUIREMENTS

| Tool/Paper | How to Handle | Why It Matters |
|---|---|---|
| **LDFI (SIGMOD 2015)** | Lead with it; clearly differentiate | Core prior work; must show you advance beyond it |
| **P (Microsoft)** | Cite; explain why domain-specific analysis beats DSL | Existing tool for similar problem |
| **TLA+** | Mention; explain why not sufficient | Reviewer will ask "why not just write spec?" |
| **kube-score / OPA** | Benchmark against; show gaps | Baseline for practical impact |
| **Timeout/retry literature** | Comprehensive review (Leners et al., SRE book) | Establish problem is real |
| **Model checking for systems** | Cite (MODIST, SAMC, etc.); position as complementary | Show you're in conversation with established area |

---

## TABLE 6: EVALUATION PLAN (DETAILED)

| Component | Requirement | Effort | Timeline |
|---|---|---|---|
| **Real configs (20+)** | Open-source projects + sanitized production | 2-3 weeks | Early |
| **Postmortem incidents (5+)** | Extract from AWS/Google/Cloudflare/Twitter reports; reconstruct configs | 3-4 weeks | Mid-project |
| **Bug detection benchmark** | Tool must find all injected bugs; measure false positives | 1-2 weeks | Late |
| **Comparison harness** | Against kube-score, istio-analyze, OPA | 1 week | Late |
| **Scalability benchmarks** | Synthetic topologies 5-200 services; measure runtime | 2 weeks | Late |
| **Soundness validation** | Tool proves cascades it finds; no false positives | 2-3 weeks | Final |
| **Total evaluation effort** | | 10-15 weeks | 20-25% of project |

**Critical path item:** Do NOT skimp on evaluation. This is where papers win or lose at review time.

---

## TABLE 7: DECISION TREE: WHICH FRAMING TO USE?

```
START: Choose primary framing for synthesis

├─ Question 1: "Can I spare time for tropical algebra?"
│  ├─ NO  → Eliminate Math-B (overkill for systems venue)
│  └─ YES → Keep if truly interested (but risky)
│
├─ Question 2: "Do I have 3-4 full-time engineers for 6-7 months?"
│  ├─ NO  → Use RetryStorm (narrower, ~4 months possible)
│  └─ YES → Can do full 6-layer CascadeSMT + compositional
│
├─ Question 3: "Can I demonstrate 5+ postmortem incident reproductions?"
│  ├─ NO  → Use RetryStorm (narrow focus → easier to validate)
│  └─ YES → Can use broader scope (CascadeSMT + composition)
│
├─ Question 4: "Do I want to handle circuit breakers fully?"
│  ├─ NO  → Use RetryStorm, defer CB to future work (cleaner)
│  └─ YES → Use CascadeSMT + Math-A, must solve non-monotonicity
│
└─ Question 5: "Is this OSDI/SOSP bound (systems) or PL-adjacent?"
   ├─ OSDI/SOSP → Use RetryStorm + Math-A (systems focus)
   └─ PLDI/POPL → Can consider Math-B (more theoretical)

RECOMMENDATION OUTPUT:
- If answered NO,YES,YES,NO,OSDI → **RetryStorm (STRONG)**
- If answered YES,YES,YES,YES,OSDI → **CascadeSMT + composition (MEDIUM-STRONG)**
- If answered YES,YES,NO,YES,PLDI → **Math-B (RISKY but novel)**
- If answered NO,NO,NO,NO,OSDI → Reconsider scope (too ambitious)
```

---

## TABLE 8: RISK MITIGATION STRATEGIES

| Risk | Probability | Impact | Mitigation Strategy | Effort |
|---|---|---|---|---|
| **Scalability wall @ 30 services** | High | High | Compositional decomposition + honest framing ("critical subgraph") | 3-4 weeks |
| **LDFI feels too close** | Medium | High | Strong differentiation story (static config vs dynamic traces) + evaluation | 2-3 weeks |
| **Circuit breaker non-monotonicity** | Medium | Medium | Either restrict configs or develop quasi-monotone theory | 2-3 weeks |
| **Evaluation takes too long** | Medium | Very High | Start evaluation early; parallelizable | 10+ weeks |
| **No "obvious" bugs found in real configs** | Low-Medium | Very High | Have fallback: synthetic benchmark with injected bugs | 2 weeks |
| **MaxSAT solver doesn't scale** | Low | Medium | Have CEGIS fallback (though risky wrt portfolio) | 2-3 weeks |
| **Helm template expansion breaks parser** | High | Low | Document limitation or defer to `helm template` preprocessing | 1 week |

**Total mitigation effort:** ~25% of project budget. Worth it.

---

## TABLE 9: FRAMING DECISION SCORECARD

**Scoring: 1=Weak, 5=Strong**

| Criterion | CascadeSMT | MeshCheck | RetryStorm |
|-----------|-----------|-----------|-----------|
| Formalization clarity | 4 | 3 | 5 |
| Scalability to real deployments | 2 | 5 | 3 |
| Novelty (genuine intellectual contribution) | 3 | 2 | 5 |
| Differentiation from prior art | 2 | 3 | 5 |
| Implementation feasibility | 3 | 2 | 4 |
| Evaluation tractability | 3 | 2 | 4 |
| Venue fit (OSDI/SOSP) | 4 | 4 | 5 |
| Risk level (lower = better) | 3 | 2 | 4 |
| **TOTAL** | **24/40** | **21/40** | **35/40** |

**Recommendation:** RetryStorm (35/40) is strongest for OSDI/SOSP, especially if evaluation is strong.

---

## TABLE 10: CONFIGURATION PARSING COMPLEXITY BREAKDOWN

| Config Type | Complexity | Parsing Approach | Time Est. |
|---|---|---|---|
| **Basic Kubernetes** (Pod, Service, Deployment) | Low | kubectl-style parsing | 1 week |
| **Istio DestinationRule + VirtualService** | Medium | CRD parsing + cross-reference | 2 weeks |
| **Envoy bootstrap config** | Medium-High | Complex JSON schema | 1-2 weeks |
| **Helm templates** | High | Go template engine + value substitution | 2-3 weeks ← **HARD** |
| **Kustomize overlays** | High | Patch merging + conditional application | 1-2 weeks |
| **SMI standard (if supported)** | Low-Medium | Standard spec parsing | 1 week |
| **Multi-namespace resolution** | Medium | Graph linkage across namespaces | 1 week |
| **Total** | | | 8-11 weeks (≈20% of project) |

**Mitigations:**
- Start with basic K8s + Istio (drop Envoy first release)
- Defer Helm/Kustomize to MVP+1
- Document that `helm template` preprocessing is expected

---

## TABLE 11: PUBLICATION READINESS CHECKLIST

| Phase | Criteria | Status |
|---|---|---|
| **Problem Definition** | ✓ Real-world gap identified | ✅ Done (retry amplification) |
| | ✓ Scope clearly bounded | 🟡 In progress (retry focus recommended) |
| | ✓ Prior art reviewed | ✅ Done (LDFI, P, Jepsen, linters) |
| **Technical Approach** | ✓ Formalization precise | 🟡 In progress (Timed Automata clear, MaxSAT vs CEGIS TBD) |
| | ✓ Novelty clearly articulated | 🟡 In progress (35% genuine novelty identified) |
| | ✓ Feasibility demonstrated | 🟡 In progress (6-layer arch justified) |
| **Implementation** | ✓ Architecture detailed | ✅ Done (150K LoC breakdown) |
| | ✓ Technology choices justified | 🟡 In progress (Z3 vs other SMT TBD) |
| | ✓ Code complexity managed | 🟠 At risk (150K LoC is large) |
| **Evaluation** | ✓ Real-world datasets | 🔴 NOT STARTED (critical path) |
| | ✓ Baseline comparisons | 🟡 Planned (vs kube-score, OPA) |
| | ✓ Scalability analysis | 🟡 Planned (benchmarks up to 200 services) |
| | ✓ Soundness proof | 🔴 NOT STARTED |
| **Publication** | ✓ Venue selected | 🟡 Recommended: OSDI/SOSP |
| | ✓ Contribution statement clear | 🟡 Drafted (see above tables) |

**Critical path items to complete NOW:**
1. Finalize problem statement (retry-focused)
2. Start evaluation infrastructure (postmortem incident extraction)
3. Justify 150K LoC architecture (make sure each layer is necessary)

---

## TABLE 12: RECOMMENDED 9-MONTH TIMELINE

| Month | Milestone | Owner | Confidence |
|---|---|---|---|
| **M1: Jan** | Problem crystallization + literature review complete | Problem Architect | High |
| **M1-M2: Jan-Feb** | Math formalization (Timed Automata + MaxSAT) | Math Lead | High |
| **M1-M3: Jan-Mar** | Implementation planning; config parsing design | Impl Lead | Medium |
| **M3-M5: Mar-May** | Core verifier implementation (Layer 2) + repair synthesis (Layer 3) | Engineers | Medium |
| **M4-M6: Apr-Jun** | Config parsing (Layer 1) + compositional decomposition (Layer 4) | Engineers | Medium |
| **M5-M7: May-Jul** | Interface (Layer 5) + evaluation infrastructure (Layer 6) | Engineers | Medium |
| **M6-M8: Jun-Aug** | Evaluation (postmortems, real configs, benchmarks) | QA Lead | Low ⚠️ |
| **M8-M9: Aug-Sep** | Paper writing + artifact preparation | Authors | Low ⚠️ |
| **M9+: Oct+** | Submission + reviews | Everyone | N/A |

**Critical dependencies:**
- Math formalization must be done by M2 (blocks implementation decisions)
- Evaluation infrastructure must start M5 (needs time for configs/incidents)
- Paper writing cannot start before M7 (needs evaluation results)

---

## KEY QUESTIONS TO ANSWER BEFORE WRITING SYNTHESIS

1. **Framing selection:** Will you focus on RetryStorm (recommended) or broader CascadeSMT?
2. **Scope confirmation:** Retry amplification only, or include timeouts/circuit breakers?
3. **Scalability positioning:** Will you honestly say "15-30 services fully verified" or promise 100+?
4. **Repair method:** MaxSAT optimization or CEGIS loop? (MaxSAT recommended)
5. **Evaluation commitment:** Can you guarantee 5+ real postmortem incident reproductions?
6. **Venue target:** OSDI/SOSP (systems focus) or POPL (if theory-heavy)?
7. **Novelty story:** How will you differentiate from LDFI to reviewers?
8. **Implementation capacity:** Do you have 3-4 engineers for 6-7 months?

**Write these decisions explicitly into your synthesis document.** Clarity here = clarity in final paper.

---

END OF DECISION MATRICES
