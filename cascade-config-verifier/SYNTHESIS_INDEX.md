# CASCADE CONFIG VERIFIER PROJECT — FULL CONTEXT SYNTHESIS
**Created:** 2026-03-08  
**Stage:** Crystallization (ideation → proposals phase)  
**Scope:** Complete project exploration for synthesis document writing

---

## QUICK REFERENCE

### What This Project Is About
An SMT-based static analyzer that discovers minimal microservice failure scenarios triggering cascading outages by analyzing Kubernetes/Istio resilience configurations, then synthesizes parameter repairs.

**Real-world problem:** A retry policy of 3 retries on service A calling service B (also 3 retries) calling service C creates 9× amplified load on C. No existing tool catches this interaction statically across service boundaries.

### Where Everything Lives

| Item | Location | Size | Content |
|------|----------|------|---------|
| **Seed Idea** | `/cascade-config-verifier/ideation/seed_idea.md` | 441 bytes | Original problem statement |
| **State Tracker** | `/cascade-config-verifier/State.json` | 15 lines | Current phase: crystallization |
| **Architect Brainstorm** | `/copilot/files/architect_brainstorm.md` | 164 lines | 3 architectural framings (CascadeSMT, MeshCheck, RetryStorm) |
| **Math Brainstorm** | `/copilot/files/math_brainstorm.md` | 197 lines | 3 mathematical approaches (Timed Automata, Tropical Algebra, Petri Nets) |
| **Implementation Brainstorm** | `/copilot/files/impl_brainstorm.md` | 201 lines | 6-layer architecture, 150K LoC breakdown |
| **Adversarial Critique** | `/copilot/files/adversarial_critique.md` | 357 lines | 9-framing cross-challenge, scalability reality check |
| **Prior Art Audit** | `/copilot/files/prior_art_critique.md` | 446 lines | Comparison with LDFI, P, Jepsen, TLA+, existing linters |

**Total Brainstorm Content:** 1,365 lines of analysis across 5 documents

---

## KEY FINDINGS SUMMARY

### 1. PROBLEM DEFINITION (SOLID)
✅ **Clear gap:** No tool models retry amplification across service boundaries
✅ **Real impact:** Multiple documented production incidents (AWS S3 2017, Twitter, Facebook)
✅ **Domain:** Kubernetes/Istio/Envoy configuration static analysis

### 2. TECHNICAL APPROACH OPTIONS (3 MAJOR PATHS)

**A. Monolithic SMT (CascadeSMT)** — Strongest formalization
- Single Z3 solver, BMC for cascade reachability, CEGIS for repair
- **Pros:** Tight formalization, produces executable traces of cascades
- **Cons:** Scalability ceiling ~15-30 services; close to LDFI (novelty ~30%)
- **Risk:** Must differentiate from prior work (LDFI, MODIST, Yang et al.)

**B. Compositional A-G (MeshCheck)** — Best scalability
- Per-service contracts, L* learning, assume-guarantee composition
- **Pros:** Scales to 100+ services; elegant theory
- **Cons:** Automatically inferred contracts may be coarse; novelty ~25% (standard AG technique)
- **Risk:** Feels incremental unless A-G for *configs* requires fundamentally new ideas

**C. Retry Amplification Focused (RetryStorm)** — Highest novelty
- Retry Amplification Graph (RAG), per-edge amplification factors, OMT optimization
- **Pros:** Laser-focused, ~50% novel contribution, clearest motivation
- **Cons:** Narrower scope; doesn't address all cascade mechanisms (rate limits, resource starvation)
- **Risk:** Might be seen as "too narrow for top venue" without strong evaluation

### 3. MATHEMATICAL LANDSCAPE

**Framing A: Timed Automata**
- Each service = TA with clocks encoding timeouts/retries
- Cascade = reachability in Network of Timed Automata
- **Complexity:** NP for bounded depth, PSPACE without bounding
- **Novelty:** Structural monotonicity theorem for cascade propagation (IF you handle circuit breaker non-monotonicity)
- **Critical Issue:** Circuit breakers break monotonicity — failure set {A,B} cascades but {A,B,D} doesn't (CB trips first)

**Framing B: Tropical Algebra & Polynomial Fixpoints**
- Service interactions = cascade polynomials in ℝ₊[x]
- Cascade = fixpoint of polynomial system exceeding threshold
- **Complexity:** Fixpoint computation O(n·log(1/ε)/log(1/λ)), phase transition detection O(n³ log(1/ε))
- **Novel:** Tropical-spectral duality, critical slowing-down phenomena
- **Problem:** Beautiful mathematics but feels disconnected from systems engineering; may be "wrong venue" (too theoretical for OSDI, too applied for POPL)

**Framing C: Petri Nets + MaxSAT**
- Service interactions = Petri net with colored tokens
- Repair = MinUnsat/MaxSAT problem
- **Novelty:** Medium; combines known techniques
- **Advantage:** MaxSAT solvers handle optimization better than CEGIS

### 4. SCALABILITY REALITY (CRITICAL)

**Verdict:** ~15-30 services max with full formal verification, 50+ with compositional decomposition

**Evidence:**
- Amazon's P: ~10-20 components before wall
- Jepsen: 3-5 nodes standard
- State explosion in 200-service topology: 200 × 3^5 = 48,600 states per time step (exceeds Z3 comfort zone)

**Honest framing required:** "We verify critical subgraph (latency-critical path) of large deployment, not entire mesh"

### 5. NOVELTY AUDIT (25-30% GENUINE NOVELTY)

**Already Exists:**
- ❌ SAT/SMT for finding minimal failure sets (LDFI, 2015)
- ❌ Model checking distributed systems (MODIST, SAMC, etc.)
- ❌ Assume-guarantee reasoning (40 years of prior art)
- ❌ CEGIS for repair (Solar-Lezama, 2008)
- ❌ Timeout/retry analysis (Leners et al., 2013; industry SRE knowledge)

**Genuinely Novel (if done right):**
- ✅ Formal characterization of retry-amplification interactions as decidable fragment
- ✅ Automated extraction from Kubernetes/Istio/Envoy configs to formal model
- ✅ Cross-manifest cascade reachability reasoning (no tool does this)
- ✅ Proof that minimal failure set discovery is NP-complete for this specific domain

### 6. PRIOR ART CHALLENGES

**LDFI (Alvaro et al., SIGMOD 2015) — CRITICAL BASELINE**
- Already finds minimal fault sets triggering safety violations
- **Your differentiation:** LDFI on dynamic traces; you on static configs
- **Quality of differentiation:** If weak, paper feels incremental; if strong, paper is defensible

**Existing Configuration Linters**
- kube-score, istio-analyze, OPA all miss retry amplification across boundaries
- **Gap is real** — no tool catches 9× load amplification from nested retries
- **BUT:** Must demonstrate non-obvious bugs; "catches things linters miss" not enough alone

### 7. IMPLEMENTATION SCOPE (150K LoC, 6 LAYERS)

1. **Config Parsing** (~20K): YAML/JSON + template expansion (Helm, Kustomize) — hardest part is Go template semantics
2. **Core Verifier** (~40K): SMT encoding, Z3 integration, cascade reachability, minimal failure discovery
3. **Repair Synthesis** (~35K): CEGIS loop, MaxSAT formulation, domain-specific heuristics
4. **Compositional Decomposition** (~20K): Service graph partitioning, contract inference, A-G composition
5. **Interface** (~20K): CLI, visualization, CI/CD integration, VS Code extension
6. **Testing & Validation** (~15K): Property-based tests, postmortem regression suite, benchmarks

**Development timeline:** 6-7 months, 3-4 full-time engineers

### 8. CRITICAL EVALUATION REQUIREMENTS (30-40% of effort)

**Must-haves for publication:**
- ✅ ≥20 real configurations from diverse open-source projects
- ✅ ≥5 documented postmortem incidents where tool would catch root cause
- ✅ Direct comparison showing bugs invisible to kube-score/istio-analyze/OPA
- ✅ Scalability benchmarks (runtime vs. topology size)
- ✅ Soundness validation (tool finds all injected bugs)

**Without this:** Even best framing feels like prototype, not publication

### 9. RECOMMENDATIONS (STRONGEST SYNTHESIS)

**Core framing: Retry Amplification Analysis (RetryStorm) + Timed Automata formalization (Math-A)**

**What to keep:**
- ✅ Retry amplification as primary focus (highest novelty)
- ✅ Timed automata formalization (tight, known to work)
- ✅ MaxSAT repair (not CEGIS — avoids portfolio overlap)
- ✅ Compositional decomposition (engineering contribution, not theory)
- ✅ Automated Kubernetes/Istio/Envoy extraction

**What to drop:**
- ❌ Tropical algebra framing (wrong venue, beautiful but disconnected)
- ❌ General "verify all resilience mechanisms" scope (too broad)
- ❌ Petri nets (adds complexity without novelty gain)
- ❌ CEGIS repair loop (portfolio overlap with dp-verify-repair)
- ❌ Rate limits, resource starvation (defer to future work)

**How to position:**
> "**RetryGuard: Formal Verification of Retry Amplification in Microservice Architectures**
>
> We formalize retry amplification in service mesh topologies as a cascade reachability problem over timed automata, discover minimal component-failure sets via MinUnsat-based search, and synthesize optimal configuration repairs via MaxSAT. Evaluation on 20+ real configurations validates our tool catches retry-amplification bugs invisible to existing linters and matches documented production incidents."

---

## ARTIFACTS FOR YOUR SYNTHESIS DOCUMENT

### Use These Directly:
1. **Seed idea** (clarity, real-world motivation)
2. **Problem statement from recommendations** (focused, honest scope)
3. **Three mathematical framings** (shows intellectual rigor)
4. **Novelty assessment** (25-30% genuine + 70% engineering is respectable)
5. **Scalability reality check** (honest limitations build credibility)
6. **6-layer implementation** (detailed, justified)
7. **Evaluation requirements** (high bar = paper is rigorous)

### Structure Your Synthesis:
```
1. Problem Statement (1-2 pages)
   - Real-world gap: retry amplification not modeled
   - Existing tools insufficient
   - Clear scope (retry focus, ~50 services, Kubernetes/Istio)

2. Technical Approach (2-3 pages)
   - Timed automata formalization
   - Cascade reachability as satisfiability
   - MinUnsat for discovery + MaxSAT for repair
   - Why this beats alternatives

3. Novelty (0.5-1 page)
   - 25-30% genuinely novel (formal characterization + automated extraction)
   - 70% engineering excellence
   - Clear differentiation from LDFI (static config vs dynamic traces)

4. Implementation Plan (1-2 pages)
   - 6-layer architecture
   - 150K LoC justified
   - 6-7 month timeline with 3-4 people

5. Evaluation Plan (1-2 pages)
   - 20+ real configs + 5+ postmortems
   - Comparison with existing tools
   - Scalability + soundness validation

6. Risks & Mitigation (0.5-1 page)
   - Scalability ceiling (~30 services)
   - Circuit breaker non-monotonicity handling
   - Evaluation effort critical path
```

---

## REMAINING UNKNOWNS / DECISION POINTS

**For you to resolve in synthesis:**
1. **Primary venue:** OSDI, SOSP, or NSDI? (Affects framing slightly)
2. **Circuit breaker handling:** Will you restrict to monotone configs, or develop quasi-monotone theory?
3. **OMT vs. MaxSAT:** OMT might be cleaner than MaxSAT for optimization; research if available?
4. **Evaluation strategy:** Will you extract configs from real incidents, or generate synthetic ones?
5. **Compositional scope:** Is A-G composition engineering detail, or part of core contribution?

**Document what you decide** — this makes your synthesis crisp and justified.

---

## FILES TO REFERENCE

All brainstorm files are available at:
- `/Users/halleyyoung/.copilot/session-state/54256d3d-8d58-453c-8c60-306434cd5e0b/files/`

Key sections in `math_brainstorm.md`:
- **Lines 1-150:** Timed Automata framing (Framing A) — READ THIS for formalization depth
- **Lines 150-450:** Tropical Algebra framing (Framing B) — beautiful but may be overkill

Key sections in `impl_brainstorm.md`:
- **Section 1-2:** Layer architecture overview
- **Section 5:** Technical claims (what to claim, what NOT to claim)
- **Section 5.7:** Biggest remaining risk = evaluation

Key sections in `adversarial_critique.md`:
- **Section 1.1:** Scalability challenge (reality check)
- **Section 1.3:** LDFI overlap (must differentiate clearly)
- **Section 4.2-4.4:** Recommended composite framing

Key sections in `prior_art_critique.md`:
- **Section 1.1, Table 1:** Direct competing prior art
- **Section 1.2:** Assessment: 25-30% novelty ratio

---

## SUMMARY IN 3 SENTENCES

1. **The problem is real:** Retry amplification across service boundaries causes production cascades; no tool catches this statically in configuration manifests.

2. **The approach is sound:** Timed automata formalization + MinUnsat discovery + MaxSAT repair is a solid technical contribution with ~25-30% genuine novelty.

3. **Success depends entirely on evaluation:** Without 20+ real configs, 5+ postmortem incident reproductions, and direct comparisons with existing tools, reviewers will see it as prototype, not publication.

---

END OF SYNTHESIS INDEX
