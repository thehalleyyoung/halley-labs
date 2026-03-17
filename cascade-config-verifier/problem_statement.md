# CascadeVerify: Static Detection of Retry Amplification and Timeout Chain Violations in Microservice Configurations via Bounded Model Checking and MaxSAT Repair

## Problem Statement

Modern microservice architectures configure retry policies and timeout budgets across dozens of independently managed services. These two primitives—retries and timeouts—are the most frequently cited cascade mechanisms in production post-mortems, yet their cross-service interactions create an emergent failure surface that no human and no existing tool can reason about statically. When service A retries 3× against B, which retries 3× against C, the 9× load amplification on C is invisible in any single manifest yet is the root cause of documented cascading outages (AWS S3 2017, Google and Cloudflare post-mortems). Timeout chains that exceed upstream deadlines silently convert partial degradation into total unavailability. The fundamental issue is compositional: each retry count and timeout budget is locally reasonable, but the multiplicative and additive interactions across service boundaries produce failure modes invisible to per-service inspection.

Today's defenses are either too local, too manual, or too late. Linting tools (kube-score, Istio Analyze, OPA/Rego, Polaris) inspect individual resources without cross-service reasoning—they can flag a missing timeout but cannot detect that a chain of well-configured timeouts violates an end-to-end deadline. Formal specification approaches (TLA+, P language) require operators to hand-encode domain knowledge for every topology change, creating a specification burden incompatible with platform engineering's goal of self-service infrastructure. Runtime fault-injection frameworks (LDFI, Jepsen, Chaos Mesh) discover failures from execution traces after deployment, not from configuration manifests before deployment—they are invaluable for validation but cannot serve as a pre-merge gate. CascadeVerify works pre-deployment on configuration manifests; LDFI (Alvaro et al., SIGMOD 2015) requires execution traces post-deployment. There is no tool that ingests actual Kubernetes, Istio, and Envoy configuration files, builds a formal model of retry-timeout interactions across service boundaries, and answers the question: under what minimal set of component failures does this configuration cascade?

CascadeVerify fills this gap by operating entirely on static configuration artifacts—the YAML and JSON manifests operators actually commit to version control. The tool automatically extracts a Retry-Timeout Interaction Graph (RTIG): a directed dependency graph annotated with per-edge retry counts and timeout budgets, resolved through Helm and Kustomize template merging. This RTIG is encoded as a bounded model checking (BMC) problem in the quantifier-free linear integer arithmetic fragment (QF\_LIA), where each service is modeled with retry counters and timeout countdowns. Cascade reachability reduces to a satisfiability query: does there exist a set of ≤k component failures such that an entry-point service becomes unavailable? The key structural insight is that retry-timeout networks without circuit breakers are *monotone*—adding failures can only increase cascade likelihood—which makes antichain-based pruning provably sound and the completeness bound trivially computable. Repairs are synthesized via weighted partial MaxSAT: hard clauses enforce cascade-freedom under all failure scenarios up to the bound, soft clauses minimize parameter deviation from the operator's original configuration, and the solver produces Pareto-optimal patches in a single optimization pass.

Unlike LDFI, CascadeVerify operates pre-deployment on config manifests rather than post-deployment on execution traces. Unlike TLA+ and P, it requires no manual specification—built-in domain semantics for Kubernetes/Istio/Envoy retry and timeout primitives eliminate the specification burden. Unlike existing linters, it performs cross-service reasoning over retry-timeout interactions. The repair synthesis uses MaxSAT rather than counterexample-guided inductive synthesis (CEGIS), naturally expressing the optimization objective "change as little as possible while guaranteeing cascade-freedom"—a qualitative differentiation from verify-repair tools that use iterative counterexample-guided loops with no optimality guarantees on repair minimality.

## Value Proposition

CascadeVerify addresses an acute, widely recognized need across the cloud-native ecosystem. Site reliability engineers at any organization running more than 20 microservices face this problem daily: every major cloud outage post-mortem identifies retry storms and timeout misconfigurations as root causes or amplifiers of cascading failure. Platform engineering teams building Internal Developer Platforms need automated guardrails that catch dangerous retry-timeout configurations before they reach production, without requiring every application team to become an expert in distributed systems failure modes. With CascadeVerify, what becomes possible is: pre-deployment detection of retry amplification and timeout chain violations as a CI/CD gate, automated minimal-disruption repair synthesis that respects operational constraints, continuous configuration verification integrated into GitOps workflows, and a formal foundation for retry-timeout best practices grounded in proofs rather than heuristics.

## Technical Difficulty

The system requires approximately 105,000 lines of Rust across 10 subsystems, each addressing inherent (not accidental) complexity. Novel core: ~50,000 LoC. Engineering/infrastructure: ~30,000 LoC. Testing/evaluation: ~25,000 LoC.

**S1: Multi-Format Config Ingestion (15,000 LoC).** Parses Kubernetes, Istio, Envoy, and gRPC configuration with full fidelity. Istio's configuration precedence rules span 40+ pages of specification, with version-specific semantics across 4+ schema versions. Cross-resource reference resolution must handle Helm template rendering, Kustomize overlays, and default value inference where manifests omit fields that the control plane fills implicitly.

**S2: Service Topology Constructor (9,000 LoC).** Builds the service dependency graph from parsed configs, handling multi-port services, subset routing via DestinationRules, VirtualService delegation chains, weighted traffic splitting, and cycle detection in service dependencies.

**S3: Retry-Timeout Semantics Model (12,000 LoC).** Formal models of retry policies (with exponential backoff, jitter, retry budgets) and timeout budgets (per-attempt, per-request, chained across hops). The SMT encoding of retry-timeout interactions—how retry amplification compounds multiplicatively across hops and timeout chains compose additively under retry expansion—is the intellectual core of the system.

**S4: SMT Encoding and Constraint Generation (13,000 LoC).** BMC unrolling of the RTIG model, failure injection as Boolean variables, symmetry breaking to prune equivalent failure sets, and cone-of-influence reduction to eliminate variables unreachable from the cascade target. The monotone structure of retry-timeout networks enables sound antichain pruning without the complications of non-monotone circuit breaker state.

**S5: Cascade Analyzer and Bounded Model Checker (11,000 LoC).** Discovers minimal failure sets via the MARCO algorithm for MinUnsat enumeration. Monotonicity of the CB-free model guarantees that antichain pruning is sound: supersets of known cascading sets and subsets of known safe sets are safely skipped. Portfolio solving dispatches to multiple Z3 configurations. Incremental solving reuses learned clauses across related queries. Cascade classification categorizes failures by mechanism (retry amplification vs. timeout chain violation).

**S6: MaxSAT Repair Synthesizer (11,000 LoC).** Encodes repair as weighted partial MaxSAT: hard clauses assert cascade-freedom under the verified failure scenarios, soft clauses penalize deviation from current parameter values weighted by operational impact. Enumerates the Pareto frontier of repair options trading off minimality against robustness margin. Translates abstract repairs back to concrete configuration diffs with human-readable explanations.

**S7: Z3 Integration Layer (6,000 LoC).** Safe Rust FFI wrappers around Z3's C API, optimization modulo theories (OMT) and MaxSAT interfaces, thread-safe portfolio solving with shared clause databases.

**S8: Incremental Analysis and CI/CD Integration (8,000 LoC).** YAML-aware configuration diffing, affected-cone computation to identify which verification results are invalidated by a change, analysis result caching, and output in SARIF and JUnit formats for integration with GitHub Actions, GitLab CI, and ArgoCD.

**S9: Counterexample Visualization (8,000 LoC).** Projects SMT models back to concrete failure traces showing load propagation step-by-step. Extracts causal chains identifying the critical path from initial failure to cascade. Generates counterfactual explanations: "if retry count on edge A→B were reduced from 3 to 2, this cascade would not occur."

**S10: Evaluation and Testing (12,000 LoC).** Synthetic topology generator with configurable graph structure, retry-timeout parameter distributions, and injected bug patterns. Semi-synthetic configuration corpus management. Unit, integration, property-based (proptest), and snapshot tests.

**Total: ~105,000 LoC Rust. Novel core: ~50,000 (48%). Engineering/infra: ~30,000 (28%). Testing/eval: ~25,000 (24%).**

The four hardest engineering challenges are: (1) faithful implementation of Istio's 40+ pages of configuration precedence semantics across schema versions; (2) SMT encoding scalability—a 30-service topology generates tens of thousands of variables, requiring domain-specific optimizations (cone-of-influence, symmetry breaking, incremental solving) to remain tractable; (3) MaxSAT repair with operational sensibility constraints ensuring repairs stay within operator-acceptable parameter ranges; (4) compositional reasoning for topologies beyond the direct-verification threshold.

## New Mathematics Required

### Formal Model Definitions (4 Formalizations)

**A1.** Retry-timeout annotated service topology G = (V, E, κ, ρ), where V is the service set, E ⊆ V × V is the dependency relation, κ: V → ℕ assigns capacity descriptors, and ρ: E → R maps each edge to a resilience policy tuple R = (retries, timeout). This is the RTIG formalization restricted to the two primitives most cited in cascade post-mortems.

**A2.** Cascade reachability defined as discrete-step load propagation. The effective load on service v at step t is given by a recursive equation incorporating retry amplification from all predecessors. A cascade occurs when effective load exceeds capacity for a designated entry-point service. This formalization captures the retry-timeout interaction space that produces cascade failures in practice.

**A3.** Retry amplification formalized as a multiplicative load factor along dependency chains. The worst-case amplification along a path is the product of per-edge retry counts; the formalization encodes this as a linear constraint in QF\_LIA via logarithmic decomposition for bounded integer domains.

**A4.** Timeout chain feasibility as an algebraic constraint: the sum of per-hop timeouts (expanded by retry counts and backoff schedules) along any dependency path must not exceed the upstream caller's deadline. This constraint captures the interaction between retries and timeouts that causes silent deadline violations.

### Theorems (6 Required)

**B1 (Proposition).** Bounded cascade reachability is NP-complete. Membership in NP is immediate (guess a failure set, verify cascade in polynomial time via load propagation). NP-hardness by reduction from SUBSET-SUM. Routine but required to establish baseline complexity.

**B2 (Novel).** BMC encoding soundness: the QF\_LIA formula Φ\_BMC(G, k, d) is satisfiable if and only if there exists a failure set F ⊆ V with |F| ≤ k such that the RTIG model reaches a cascade state within d steps. Proof by structural induction on the unrolling depth, with case analysis for retry and timeout primitive encodings.

**B3 (Novel).** Completeness bound: d\* = diameter(G) × max\_retries suffices for completeness in the monotone retry-timeout model. This is trivially provable: in the absence of circuit breakers, load propagation is monotonically increasing and reaches a fixed point within one full traversal of the graph scaled by the maximum retry fan-out. This bound is tight and eliminates the open problem present in the CB-inclusive model.

**B4 (Adaptation).** MinUnsat cores of the negated cascade-freedom formula correspond exactly to minimal failure sets inducing cascades. Adapts known MUS theory (Liffiton & Sakallah) to the cascade domain, enabling use of off-the-shelf MUS enumeration algorithms.

**B5 (Adaptation).** MaxSAT repair optimality: the weighted partial MaxSAT formulation guarantees that returned repairs minimize total parameter deviation subject to cascade-freedom. Follows from standard MaxSAT optimality guarantees applied to the specific encoding.

**B6 (Novel — central contribution).** Retry-timeout networks without circuit breakers are monotone: for any failure sets F ⊆ F', if F induces a cascade then F' induces a cascade. Proof is a structural lemma showing that every operator in the retry-timeout load propagation equation is monotonically non-decreasing in the number of failed services. This is the key theoretical result: it guarantees soundness of antichain pruning for minimal failure set enumeration. The converse—that circuit breakers break monotonicity via protective tripping—is stated as the central open problem motivating future work on the CB-inclusive model.

### Complexity Results (4 Results)

**D1.** Cascade reachability is NP-complete (coincides with B1; the SUBSET-SUM reduction is the primary hardness contribution).

**D2.** Enumerating all minimal failure sets is #P-hard; the decision problem "is there a minimal failure set not in a given collection?" is Σ₂ᴾ-complete.

**D3 (Novel).** Optimal repair synthesis—finding a minimum-cost parameter modification guaranteeing cascade-freedom—is Σ₂ᴾ-complete. The ∀∃ quantifier alternation arises from universally quantifying over failure sets while existentially quantifying over load propagation witnesses.

**D4 (Novel application).** For service topologies of bounded treewidth w, cascade verification is fixed-parameter tractable in O(n · 2^O(w) · d · r\_max), where d is the unrolling depth and r\_max is the maximum retry count. Real microservice topologies typically have treewidth ≤ 5–8, making this result practically relevant for the monotone retry-timeout model.

**Central theoretical contribution:** B6 (monotonicity of CB-free networks) + B3 (tight completeness bound for the monotone case) together show that the retry-timeout verification problem, while NP-complete in general, is efficiently solvable via BMC with sound pruning. The non-monotonicity introduced by circuit breakers is the key open challenge for v2.

## Best Paper Argument

NSDI would select this work for five reasons.

**First, it formalizes a problem everyone recognizes but nobody has solved.** Every SRE knows about retry storms. Every post-mortem identifies timeout misconfigurations. Yet there is no formal definition of when a microservice retry-timeout configuration is cascade-safe, no complexity characterization of the verification problem, and no tool that answers the question from configuration manifests alone. CascadeVerify provides all three.

**Second, it delivers a clean theoretical contribution with immediate practical payoff.** The monotonicity lemma (B6) and tight completeness bound (B3) show that the retry-timeout verification problem—the common case responsible for the majority of documented cascade failures—is efficiently solvable via BMC with sound antichain pruning. The MinUnsat correspondence (B4) connects cascade analysis to mature SAT/SMT infrastructure. MaxSAT-optimal repair synthesis (B5, D3) is the first application of MaxSAT optimization to resilience configuration repair. This is novel domain formalization and engineering built on established BMC and MaxSAT techniques—honest about its foundations while demonstrating genuine novelty in their application.

**Third, it bridges formal methods and systems in the way NSDI rewards.** The tool operates on real Kubernetes, Istio, and Envoy configurations with built-in domain semantics for retry and timeout primitives—no manual specification required. It finds real bugs in real deployments, produces real fixes as configuration diffs, and integrates into CI/CD pipelines. NSDI values practical tools with solid theoretical grounding, and CascadeVerify delivers exactly that.

**Fourth, the evaluation is rigorous and honest.** Semi-synthetic topologies with injected bugs provide full ground-truth control. Real open-source configs from Artifact Hub Helm charts and standard demos demonstrate generality. Direct comparison against both a graph-analysis baseline and LDFI isolates CascadeVerify's specific contribution. Post-mortem case studies are presented as plausible reconstructions, not claimed ground truth.

**Fifth, it introduces a clean, memorable abstraction with a clear roadmap.** The Retry-Timeout Interaction Graph is a precise, composable representation of how retry and timeout primitives interact across service boundaries. The explicit scoping to the monotone case, with circuit breaker non-monotonicity as the stated open challenge, demonstrates intellectual honesty and invites follow-up work.

## Evaluation Plan

All evaluation is fully automated with zero human involvement.

**Axis 1 — Semi-Synthetic Bug Detection (Primary).** Generate synthetic topologies (chain, tree, mesh, hub-and-spoke) at scales of 5–50 services with injected retry amplification bombs and timeout chain violations. Full ground-truth control: every injected bug has a known minimal failure set and known optimal repair. Measure detection rate, false positive rate, and repair quality against ground truth.

**Axis 2 — Real Open-Source Configs (Secondary).** Extract retry and timeout configurations from open-source Kubernetes deployments: Google Online Boutique, Weaveworks Sock Shop, Istio BookInfo, microservices-demo variants, and Helm charts from Artifact Hub. Report previously unknown cascade risks with severity triage and proposed repairs. Direct comparison against LDFI on the semi-synthetic configurations where both tools can operate.

**Axis 3 — Graph-Analysis Baseline Comparison.** Compare CascadeVerify against a simple graph-based retry amplification detector that computes product-of-retry-counts along paths and sum-of-timeouts along paths. Demonstrate that CascadeVerify's SMT analysis catches what simple path analysis misses: shared-dependency fan-in effects, timeout-retry interactions under multi-failure scenarios, and optimal repair synthesis.

**Axis 4 — Scalability Benchmarks.** Measure end-to-end verification and repair synthesis time as a function of service count (5, 10, 20, 30, 50 services) and topology structure. Compare detection coverage against kube-score, Istio Analyze, OPA/Rego, and Polaris on the same configuration corpus. Target: 30–50 services for direct verification.

**Axis 5 — Repair Quality.** Measure repair minimality (number of parameters changed and total deviation magnitude), repair soundness (all repaired configurations pass re-verification), and operational sensibility (all repairs fall within operator-specified parameter bounds). Compare against naive repair baselines (reset to defaults, uniform reduction).

**Appendix — Post-Mortem Case Studies.** Reconstruct plausible configurations consistent with documented cascading-failure post-mortems (AWS S3 2017, Google 2019, Cloudflare 2019). Methodology explicitly documented: these are "plausible configurations consistent with described failure," not extracted ground truth. Demonstrate that CascadeVerify identifies the described root-cause patterns and proposes repairs consistent with the fixes described in each post-mortem.

All configurations, analysis scripts, and evaluation results are released as a public artifact for reproducibility.

## Laptop CPU Feasibility

CascadeVerify is designed to run on a single laptop CPU without GPU acceleration, exploiting the structural constraints of real service mesh configurations that make the theoretical worst cases irrelevant in practice.

Real configurations are dramatically more constrained than general timed-automaton networks. Retry counts are small integers (1–5). Timeouts are finite operator-chosen constants. Load propagation depth is bounded by graph diameter, which rarely exceeds 8–12 in production topologies. BMC unrolling depth is therefore bounded: d ≤ diameter(G) × max\_retries ≈ 60 steps for a deep topology with aggressive retries. Crucially, the monotonicity of the CB-free model (Theorem B6) guarantees that antichain pruning is sound in all cases—there is no non-monotone edge case requiring expensive re-checks.

The resulting QF\_LIA formula has O(n · d · c) variables, where n is the service count (≤50 for direct analysis), d is the unrolling depth (≤60), and c is the number of constraint variables per service per step (≤3 for retry-timeout only). This yields approximately 9,000 SMT variables for a 50-service topology—well within Z3's comfort zone for sub-minute solving. The absence of circuit breaker FSM encoding substantially reduces the per-service variable count compared to the full five-primitive model.

MaxSAT repair operates over at most 150 soft clauses (one per tunable retry count or timeout budget), not over the state space. Target performance: 30-service verification in under 30 seconds, 50-service verification in under 90 seconds, repair synthesis in under 120 seconds, on a single laptop core. All computations are symbolic (SMT/MaxSAT), requiring no GPU.

## Future Work: Circuit Breaker Non-Monotonicity

The deliberate scoping of v1 to retry-timeout networks is motivated by a precise theoretical obstacle: circuit breakers introduce non-monotonicity into cascade reachability. Adding a failure can cause a circuit breaker to trip, thereby *preventing* a downstream cascade—which means that a superset of a cascading failure set may be safe, invalidating antichain pruning. This non-monotonicity problem is the key open challenge for v2. Rate limiters and bulkheads, which introduce similar capacity-threshold discontinuities, are also deferred. Extending the model to circuit breakers requires either: (a) a stratified analysis that identifies monotone subregions and applies targeted re-checks only where CB interactions are relevant, or (b) a reformulation of the BMC encoding that tracks CB state transitions explicitly at the cost of a substantially larger variable space and a completeness bound that accounts for CB open→halfOpen→closed cycles. Establishing a tight completeness bound for the CB-inclusive non-monotone case remains the hardest open theoretical problem in this research program.

## Slug

cascade-config-verifier
