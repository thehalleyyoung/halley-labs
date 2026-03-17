# SoniType: A Perceptual Type System and Optimizing Compiler for Information-Preserving Data Sonification

**Slug:** `perceptual-sonification-compiler`

**Verification Panel Scores:** V4 / D6 / BP5 / L7.5 — Verdict: CONDITIONAL CONTINUE  
**Amendment Status:** This document incorporates all 7 binding amendments from the verification panel plus accepted non-binding recommendations. Amended sections are marked with parenthetical annotations.

---

## Problem and Approach

Data sonification—the systematic mapping of data variables to auditory parameters—remains a craft practice despite five decades of research. Practitioners hand-tune pitch ranges, timbral mappings, and temporal envelopes by trial-and-error, with no formal guarantee that the resulting audio stream actually conveys the intended information. The core dysfunction is that sonification design today lacks the two properties that made visualization mature: a declarative specification language with composable semantics, and an optimizing backend that automatically negotiates the constraints of the output channel. We observe that the human auditory system is, formally, a noisy channel with well-characterized capacity limits—critical-band masking, Weber-fraction just-noticeable differences (JNDs), and Bregman-style stream segregation constraints—and that sonification is therefore a *lossy coding problem* over a psychoacoustically-constrained perceptual channel. No existing system treats it as such.

(Amended per verification panel — safety-critical motivating example per non-binding recommendation) The consequences of this gap are not merely academic. Medical alarm fatigue is a documented, quantified patient-safety crisis: the ECRI Institute has listed alarm hazards as the **#1 health technology hazard** in multiple years, the Joint Commission issued a National Patient Safety Goal for clinical alarm management, and published data documents patient deaths directly attributable to alarm fatigue—clinicians ignoring alarms because too many acoustically indistinguishable alarms have desensitized them. A typical ICU has 30+ distinct alarm types. The IEC 60601-1-8 standard prescribes that "alarms should be distinguishable," but provides no computational tool to *verify* distinguishability across all alarm combinations in a given acoustic environment. This is the gap SoniType addresses: formal, automated verification and optimization of auditory discriminability.

We propose **SoniType**, a domain-specific language and optimizing compiler that accepts declarative data-to-sound mapping specifications and compiles them into bounded-latency audio renderers while maximizing the perceptual mutual information between input data and perceived audio. (Amended per verification panel — venue commitment: OOPSLA; PL contribution framing) The key intellectual contribution is a *perceptual type system* in which psychoacoustic constraints—masking thresholds, JND bounds, stream segregation predicates, and cognitive load budgets—are first-class type qualifiers. This is a new class of refinement type where the refinement predicates come from psychophysics rather than program logic, with domain-specific challenges absent from standard refinement type frameworks: non-local cross-stream interactions (adding a stream can invalidate previously satisfied constraints for other stream pairs) and non-convex feasibility regions arising from critical-band masking. A sonification specification that type-checks is guaranteed, relative to the psychoacoustic model, to produce perceptually discriminable, non-masked, cognitively tractable audio. The compiler's optimizer then searches the space of admissible mappings to maximize a novel objective, *psychoacoustically-constrained mutual information* $I_\psi(D; A)$, which quantifies how much data information survives the auditory channel after accounting for masking, fusion, and attentional limits. (Amended per verification panel — Halide comparison qualified per non-binding recommendation) This is to sonification what Halide's scheduling language is to image processing in *architectural approach*—where Halide separates algorithm from schedule over a hardware target, SoniType separates data semantics from perceptual rendering over a human auditory target. We note the structural parallel honestly: the impact scale differs (image processing is a billion-dollar industry; sonification is a niche field), but the separation-of-concerns design principle and the domain-specific compiler architecture are closely analogous.

The compiler pipeline operates in three phases. First, the *front-end* parses the declarative specification, performs perceptual type-checking (verifying stream segregation, masking clearance, JND satisfaction, and cognitive load feasibility), and lowers the specification into an intermediate representation of data-dependent audio graphs. Second, the *optimizer* applies psychoacoustic cost models to select concrete parameter bindings (pitch ranges, timbral indices, panning angles, temporal rates) that maximize $I_\psi(D; A)$ subject to the verified constraints. Third, the *back-end* compiles the optimized audio graph into a bounded-latency renderer with worst-case execution time (WCET) guarantees suitable for real-time audio callbacks at 256–1024 sample buffer sizes (5.8–23.2 ms at 44.1 kHz). Critically, all optimization occurs at compile time; the renderer merely executes a pre-optimized audio graph, making real-time performance trivial on commodity hardware.

(Amended per verification panel — perceptual linting positioning per non-binding recommendation) Beyond compilation, SoniType also functions as a **perceptual linter**: it can analyze *existing* sonification designs for masking violations, JND failures, and segregation problems without requiring users to rewrite their work in the SoniType DSL. This dramatically lowers the adoption barrier—users can check existing designs incrementally, analogous to how ESLint gained adoption as a checker before whole-program analysis tools could. An alarm designer can feed an existing alarm palette specification to SoniType and receive a formal discriminability report identifying all masking conflicts and below-JND parameter pairs.

This framing resolves a tension that has stalled the sonification field: researchers have long known that psychoacoustic constraints matter, but lacked a formal framework to *mechanize* those constraints into automated tooling. PAMPAS and related evaluation frameworks can assess sonification quality post hoc, but cannot guide design. Faust, SuperCollider, ChucK, and Csound provide powerful audio synthesis but are agnostic to data semantics and perceptual constraints—they are audio compilers, not sonification compilers. Dedicated sonification tools—Sonification Sandbox (Georgia Tech), xSonify (NASA), the Highcharts Sonification Module, and systematic mapping frameworks such as Weber et al.—provide accessible interfaces but lack type systems, psychoacoustic constraint verification, or information-theoretic optimization. SoniType is the first system to close this loop: declarative specification → psychoacoustic constraint verification → information-theoretic optimization → bounded-latency rendering, with formal guarantees at each stage.

The relationship to perceptual coding in MP3/AAC is instructive but critically different. Those codecs use psychoacoustic models to *discard* information below perceptual thresholds, minimizing bit-rate. SoniType inverts this: we use the same class of models to *maximize* information transmission through the auditory channel, ensuring that every discriminable perceptual dimension carries maximal data signal. The psychoacoustic model is not a compression tool but an optimization objective.

---

## Value Proposition

(Amended per verification panel — safety-critical alarm design as primary motivating example per non-binding recommendation)

**Who needs this.** Safety-critical alarm designers working on medical monitoring systems face a formally unsolved problem: verifying that an alarm palette's acoustic signatures are mutually discriminable under realistic conditions. Hospital ICUs with 30+ alarm types and documented alarm fatigue deaths represent the most urgent use case—not because SoniType replaces clinical validation, but because it provides **design-stage verification** that screens out acoustically flawed alarm configurations *before* expensive human factors testing. Accessibility researchers building auditory displays for blind and low-vision users currently spend weeks hand-tuning sonification parameters with no formal assurance of perceptual effectiveness. Climate scientists, epidemiologists, and financial analysts exploring multivariate time series through sound have no tool that automatically negotiates interference between simultaneous auditory streams.

**Why desperately.** The EU Accessibility Act (2025) and WCAG 2.2 are driving demand for non-visual data representations, though we note this demand is primarily served by screen readers, ARIA markup, and structured data formats—sonification addresses a complementary niche where temporal and multivariate data benefit from auditory display. The specific urgency is in safety-critical monitoring: medical alarm design today relies on prescriptive rules (IEC 60601-1-8) without formal psychoacoustic verification. Sonification remains niche precisely because the design cost is high and the quality assurance is nonexistent. A compiler that automates perceptual optimization and provides formal quality guarantees removes the primary barrier to adoption in domains where auditory discriminability is a safety requirement.

**What becomes possible.** (1) Automatic generation of perceptually optimized sonifications from data schema annotations, reducing design time from weeks to minutes. (2) Formal certificates—relative to the psychoacoustic model—that a sonification meets discriminability thresholds, enabling rapid iteration and CI/CD-style perceptual quality evaluation. (3) Composable multi-stream sonifications where adding a new data variable is guaranteed not to mask or fuse with existing streams, verified by type-checking. (4) Bounded-latency rendering suitable for safety-critical real-time monitoring, backed by static WCET cost estimation. (5) **Perceptual linting** of existing sonification and alarm designs—checking IEC 60601-1-8 alarm palettes, ICAD-published sonifications, or any audio configuration for masking conflicts and discriminability failures without requiring adoption of the full SoniType DSL.

---

## Technical Difficulty

(Amended per verification panel — venue commitment: OOPSLA) The system's primary intellectual contribution is to the programming languages community: a perceptual type system that brings psychoacoustic constraints into the domain of formal PL machinery. The psychoacoustic models and information-theoretic optimization serve as the application domain, analogous to how Halide's contribution is PL architecture applied to image processing, not image processing per se.

### Hard Subproblems

1. **Formalizing psychoacoustic constraints as decidable predicates.** Bregman's auditory scene analysis principles (onset synchrony, harmonicity, spectral proximity, common fate) are stated qualitatively in the literature. We must formalize them as Boolean predicates over audio stream configurations with decidable satisfiability—a novel formal-methods contribution to auditory science.

2. **Optimizing over a non-convex psychoacoustic landscape.** The psychoacoustically-constrained mutual information $I_\psi(D; A)$ is non-convex due to critical-band masking interactions. The optimizer must search a combinatorial space of parameter bindings (pitch ranges × timbral indices × panning × temporal rates) subject to masking, JND, segregation, and cognitive load constraints. We use constraint propagation to prune infeasible regions, then branch-and-bound with psychoacoustic cost model evaluations.

3. **Perceptual type inference with non-local constraints.** Inferring whether a multi-stream sonification satisfies all psychoacoustic constraints requires reasoning about joint spectral occupancy, temporal overlap, and attentional load. This is more complex than standard type inference because constraint interactions are non-local: adding a stream can violate segregation predicates for previously valid stream pairs. This non-local re-checking under composition is the core PL novelty.

4. **Bounded-latency code generation.** The renderer must execute within a hard real-time deadline (the audio buffer callback period). We adapt static WCET cost estimation from real-time systems to audio graph execution, bounding the cost of each node and verifying schedulability at compile time. (We note this is static cost estimation with conservative safety margins, not formal WCET analysis in the abstract-interpretation sense à la aiT/Chronos; the 50–100× headroom makes formal analysis unnecessary.)

5. **Principled automated evaluation.** Evaluating sonification quality without human subjects requires computing model-predicted discriminability ($d'_\text{model}$) from signal detection theory over psychoacoustic model predictions—validating that model-predicted discriminability tracks published human discriminability data across conditions. This is a methodological contribution independent of the compiler.

### Subsystem Breakdown

(Amended per verification panel — binding amendments #1 and #2: scope reduction to ≤100K LoC and honest LoC reporting with Novel Core / Supporting Infrastructure split)

The full system requires code spanning the compiler and application domains. We implement in Rust (core compiler and renderer) and Python (evaluation framework). The table below separates the **novel core**—the intellectually novel subsystems that define the contribution—from **supporting infrastructure** that is necessary but routine engineering.

**Novel Core (~48K LoC)**

| Subsystem | Est. LoC | Language | Justification |
|---|---|---|---|
| **DSL parser, AST, type-checker** | 12,000 | Rust | Declarative grammar with perceptual type qualifiers; constraint-propagation type-checker over psychoacoustic predicates; formal typing rules (§Type System Formalization) |
| **Psychoacoustic cost models** | 8,000 | Rust | Bark-scale critical-band masking (Schroeder/Zwicker), Weber-fraction JND models, Bregman segregation predicates, cognitive load scoring—all as compiler-integrated cost functions |
| **Information-theoretic optimizer** | 10,000 | Rust | $I_\psi(D; A)$ computation, constraint propagation, branch-and-bound search over mapping parameter space, Pareto front for multi-objective (MI vs. latency vs. cognitive load) |
| **IR and optimization passes** | 8,000 | Rust | Audio graph IR with data-dependency annotations; dead-stream elimination, masking-aware stream merging, temporal scheduling, spectral bin packing |
| **Code generator + static WCET analyzer** | 7,000 | Rust | Lowers optimized audio graph to bounded-latency callback code; static WCET cost estimation per node; schedulability verification |
| **Evaluation framework** | 3,000 | Python | Core $d'_\text{model}$ computation, $I_\psi$ estimation, cross-model evaluation protocol, published human-data comparison (§Evaluation) |
| **Novel core subtotal** | **~48,000** | | |

**Supporting Infrastructure (~47K LoC)**

| Subsystem | Est. LoC | Language | Justification |
|---|---|---|---|
| **Real-time audio renderer** | 10,000 | Rust | Lock-free audio graph executor, sample-accurate parameter interpolation, multi-stream mixer with per-stream gain/pan, output to system audio and WAV |
| **Standard library + built-in mappings** | 12,000 | Rust | Curated pitch scales (linear, log, Bark, mel), 3–4 timbral palettes (additive, FM, noise-band), temporal patterns, spatial mappings; preset sonification recipes for common data types |
| **Test infrastructure** | 10,000 | Rust | Property-based tests (quickcheck) for type-system soundness, compiler correctness, WCET bound validity; integration tests; audio regression tests via spectral fingerprinting |
| **Evaluation harness + baselines** | 7,000 | Python | Baseline comparison harness (SonifyR, TwoTone, hand-tuned ICAD, LLM-generated), statistical analysis and figure generation |
| **CLI, error reporting, tooling** | 8,000 | Rust/Python | Compiler CLI with structured diagnostics, perceptual lint mode, basic preview, documentation generation |
| **Supporting infrastructure subtotal** | **~47,000** | | |

| | **TOTAL** | **~95,000 LoC** | |

**Minimum Viable Compiler (MVC): ~50K LoC.** The MVC contains all novel intellectual contributions: DSL parser, perceptual type-checker, psychoacoustic cost models, information-theoretic optimizer, code generator, and minimal evaluation framework. The standard library ships with curated defaults sufficient for evaluation; full library expansion, CLI polish, preview server, and debug visualization are Phase 2.

---

## New Mathematics Required

### Crown Jewels (Grade A — define the contribution)

**Theorem 1: Psychoacoustically-Constrained Mutual Information.** Define $I_\psi(D; A) = I(D; \psi(A))$, where $\psi: \mathcal{A} \to \mathcal{P}$ is the perceptual front-end function mapping acoustic signal $A$ through the psychoacoustic channel (masking, JND quantization, segregation filtering) to perceived representation $\psi(A)$, and the mutual information is evaluated over the space of admissible mappings. The constraints $\mathcal{C}_\psi$ require that (i) no data-carrying spectral component falls below the masked threshold in its critical band, (ii) parameter differences between data-distinguished categories exceed their respective JNDs, and (iii) simultaneous streams satisfy Bregman segregation predicates. We prove this quantity is well-defined, computable in polynomial time for bounded stream counts, and that maximizing it over admissible mappings is NP-hard in general but admits efficient approximation when psychoacoustic constraints are decomposable across critical bands: the problem decomposes into $B=24$ independent Bark-band subproblems, and greedy submodular maximization under matroid constraints yields a $(1-1/e)$-approximation factor. **Novelty: No prior formulation exists.** Existing information-theoretic analyses of auditory perception (e.g., Durlach & Braida's intensity resolution model) do not formulate an optimizable objective over parameterized mappings.

**Theorem 2: Auditory Stream Segregation as Decidable Predicates.** We formalize Bregman's grouping principles—onset synchrony ($|\Delta t_\text{onset}| > \tau_\text{sync}$), harmonicity (spectral components lie on a common $f_0$ harmonic series within tolerance $\epsilon_h$), spectral proximity ($|\Delta f_\text{centroid}| > \delta_\text{Bark}$ on the Bark scale), and common fate ($|\Delta \text{AM}_\text{rate}| > \rho$ or $|\Delta \text{FM}_\text{rate}| > \rho$)—as Boolean predicates over pairs of audio stream descriptors. We prove that satisfiability of a conjunction of segregation predicates for $k$ streams is decidable in $O(k^2)$ pairwise checks when individual predicates are monotone in their stream parameters. **Novelty: First computationally tractable approximation of Bregman's grouping cues as decidable predicates over audio stream descriptors.** We acknowledge that the Boolean threshold model captures necessary conditions for stream segregation, not a complete model of auditory scene analysis; subthreshold grouping cues and context-dependent streaming effects are outside the model's scope.

**Theorem 3: Information Loss Bound.** For any sonification mapping $\sigma: \mathcal{D} \to \mathcal{A}$ and psychoacoustic channel model $\psi$, the perceptual information loss $L(\sigma) = I(D; A) - I_\psi(D; A)$ satisfies $L(\sigma) \leq \epsilon$ if and only if $\sigma$ satisfies the psychoacoustic feasibility constraints $\mathcal{C}_\psi$ with margin $\geq g(\epsilon)$ where $g$ is a monotone function derived from the masking and JND models. This biconditional holds relative to the psychoacoustic model $\psi$; its validity as a proxy for ground-truth human perception is established empirically via cross-model evaluation and published human-data anchoring (§Evaluation). **Significance:** This enables *automated evaluation*—a mapping's perceptual quality can be bounded without human listening studies, by checking constraint satisfaction margins, *subject to the psychoacoustic model's validity*. **Novelty: First result connecting psychoacoustic constraint margins to information-theoretic loss bounds.**

**Theorem 4: Perceptual Soundness.** If a DSL specification $S$ type-checks under the perceptual type system, and the compiler produces renderer $R$, then for all valid input data $d$: $I_\psi(d; R(d)) \geq I_\psi(d; \llbracket S \rrbracket(d)) - \delta$, where $\delta$ bounds rounding and temporal quantization losses in code generation. This is a standard compiler correctness theorem, but novel in this domain: it guarantees the compiled renderer preserves the information-theoretic properties verified by the type system. **Novelty: First compiler correctness result for a psychoacoustically-constrained audio system.**

### Essential Enablers (Grade B)

**Theorem 5: Cognitive Load Budget Algebra.** We model working-memory constraints using a resource algebra $(\mathcal{L}, \oplus, \leq B)$ where each auditory stream $s_i$ has a cognitive load cost $\ell(s_i) \in \mathcal{L}$, streams compose via $\oplus$, and the total must satisfy $\bigoplus_i \ell(s_i) \leq B$ for a capacity bound $B$ derived from Cowan's $4 \pm 1$ object limit. The type system enforces this as a linear resource constraint. **Adapted from** cognitive architecture theory (Cowan, Baddeley) but novel as a formal type-system resource qualifier.

**Theorem 6: Approximate Compositional Discriminability.** If streams $\{s_1, \ldots, s_k\}$ pairwise satisfy segregation predicates, then the joint model-predicted discriminability $d'_{\text{model},\text{joint}} \geq \alpha \cdot \min_i d'_{\text{model},i}$ for a factor $\alpha \in (0.7, 1.0]$ that depends on spectral overlap in non-critical regions. **Honest limitation:** This is an *approximate sufficient condition*, not an exact decomposition. Full compositionality fails when streams have correlated temporal modulations that induce perceptual grouping below the segregation threshold. We characterize the approximation gap empirically and prove the bound is tight for spectrally disjoint streams.

**Theorem 7: WCET Schedulability for Audio Graphs.** For an audio graph $G$ with nodes $\{n_1, \ldots, n_m\}$, each with WCET bound $w_i$ derived from operation-level cost models, the graph executes within a single audio buffer period $T_\text{buf}$ iff $\sum_j w_{n_j} \leq T_\text{buf}$ along every path in the topologically sorted execution order. We provide WCET bounds for all primitive operations (oscillators, filters, envelopes, mixers) calibrated to commodity x86-64 hardware. **Adapted from** real-time systems WCET analysis, novel in application to psychoacoustically-optimized audio graphs.

**Theorem 8: Masking and JND Cost Models as Compiler Intrinsics.** Bark-scale critical-band masking thresholds (Schroeder spreading function) and Weber-fraction JND models (Δf/f for pitch, ΔL/L for loudness) are formalized as cost functions in the compiler's intermediate representation. We prove that these cost functions are Lipschitz-continuous in stream parameters, enabling gradient-based local refinement within the branch-and-bound optimizer. **Adapted from** psychoacoustic modeling (Zwicker & Fastl), novel as compiler-integrated cost functions.

---

## Perceptual Type System Formalization

(Amended per verification panel — binding amendment #4: type system substantiation)

The perceptual type system is the primary intellectual contribution and must be substantiated as genuine PL machinery, not constraint checking with type-system vocabulary. This section commits to the formal structure required for OOPSLA submission.

### Typing Judgments

The type system extends a standard simply-typed lambda calculus with perceptual refinement qualifiers. The core judgment has the form:

$$\Gamma \vdash e : \tau \langle \phi \rangle$$

where $\Gamma$ is a typing context mapping variables to types, $e$ is a DSL expression, $\tau$ is a base type (Stream, Mapping, SonificationSpec), and $\phi$ is a perceptual qualifier—a conjunction of psychoacoustic predicates drawn from Theorems 2, 5, and 8. Key typing rules include:

- **T-Stream**: $\Gamma \vdash \text{stream}(f_0, \text{timbre}, \text{pan}) : \text{Stream}\langle \text{band}(f_0), \ell(\text{timbre}) \rangle$ — a stream literal is typed with its Bark-band occupancy and cognitive load cost.

- **T-Compose**: $\frac{\Gamma \vdash s_1 : \text{Stream}\langle \phi_1 \rangle \quad \Gamma \vdash s_2 : \text{Stream}\langle \phi_2 \rangle \quad \text{seg}(s_1, s_2) \quad \text{mask}(s_1, s_2) \quad \ell(\phi_1) \oplus \ell(\phi_2) \leq B}{\Gamma \vdash s_1 \| s_2 : \text{MultiStream}\langle \phi_1 \wedge \phi_2 \wedge \text{sep}(s_1, s_2) \rangle}$
  — composing two streams requires segregation predicates, masking clearance, and cognitive load budget satisfaction. This is where non-local re-checking occurs: composing $s_3$ with an existing $s_1 \| s_2$ re-checks all pairwise constraints.

- **T-Map**: $\frac{\Gamma \vdash d : \text{Data}[\sigma] \quad \Gamma \vdash s : \text{Stream}\langle \phi \rangle \quad \text{jnd}(\sigma, s) \geq \delta}{\Gamma \vdash \text{map}(d, s) : \text{Mapping}\langle \phi, \text{disc}(\sigma, s) \rangle}$
  — mapping a data variable to a stream requires that the mapping's parameter range exceeds the JND threshold for the data's value distribution.

- **T-Sub**: Perceptual qualifiers form a partial order where $\phi_1 \leq \phi_2$ iff $\phi_1$ implies stronger constraint satisfaction margins. Subtyping follows: $\frac{\Gamma \vdash e : \tau\langle \phi_1 \rangle \quad \phi_1 \leq \phi_2}{\Gamma \vdash e : \tau\langle \phi_2 \rangle}$

### Operational Semantics

The DSL has a small-step operational semantics where evaluation reduces a specification to an audio graph configuration. The key reduction rules are:

- **E-Compile**: A well-typed SonificationSpec reduces to an AudioGraph via the optimizer (which preserves all type-level invariants by construction).
- **E-Render**: An AudioGraph reduces to a sample buffer sequence via the renderer, with WCET bounds attached by Theorem 7.

### Soundness Proof Sketch

**Type Safety (Theorem 4, formalized):** If $\vdash S : \text{SonificationSpec}\langle \phi \rangle$, then compilation produces a renderer $R$ such that for all valid inputs $d$, $R(d)$ satisfies all psychoacoustic predicates in $\phi$ with margin $\geq g(\epsilon)$ minus the code-generation quantization bound $\delta$. The proof proceeds by:

1. **Constraint preservation through compilation**: Each compiler pass (IR lowering, optimization, code generation) preserves constraint satisfaction margins. The optimizer only selects parameter bindings that satisfy all type-level predicates with margin. The code generator introduces bounded quantization error $\delta$ (Theorem 4).

2. **Compositionality via pairwise re-checking**: The T-Compose rule checks all pairwise constraints at each composition step. Theorem 2 guarantees this is decidable. The resource algebra (Theorem 5) guarantees cognitive load composition is well-defined.

3. **Progress**: Every well-typed specification can be compiled—the optimizer is guaranteed to find at least one feasible mapping if the type checker accepts (the type checker verifies feasibility before the optimizer runs).

**Risk acknowledgment**: If the formal typing rules and soundness proof cannot be carried to the depth required for OOPSLA (e.g., if non-local constraint interactions resist clean formalization), the fallback framing is a **"constraint verification framework for perceptual audio"** — still a genuine PL-adjacent contribution, but targeting ICAD or SPLASH workshop venues rather than OOPSLA. This risk is assessed at ~40% probability and is the single highest-risk item in the project.

---

## Strong Publication Argument

(Amended per verification panel — binding amendment #3: venue commitment OOPSLA; non-binding recommendation: downgrade from "best paper" to "strong publication" framing)

An OOPSLA program committee would recognize SoniType as a strong contribution for two reasons.

**A novel class of domain-specific refinement types.** The perceptual type system introduces refinement predicates drawn from psychophysics rather than program logic. This connects to established PL work on refinement types (Liquid Haskell, Flux) and graded modal types (Granule), but introduces unique domain-specific challenges: non-local cross-stream constraint interactions (where adding a stream can invalidate previously satisfied constraints for other stream pairs) and non-convex feasibility regions arising from critical-band masking. OOPSLA has a strong tradition of domain-specific language papers (Flix, Granule, and numerous DSLs for databases, networking, and hardware description). A perceptual type system for sonification fits this tradition naturally. The psychoacoustic application domain provides the motivation and evaluation; the PL contribution—novel typing rules, operational semantics, soundness guarantees—stands on its own. (Amended per verification panel — Halide comparison qualified) The Halide comparison is instructive as a *structural* analogue—both systems separate domain semantics from backend optimization in a compiler—but we do not claim impact equivalence; Halide transformed a billion-dollar industry, while sonification serves a smaller community.

**Methodological paradigm shift for a research community.** The sonification community has been stuck in an empirical loop: design a mapping, run a listening study, iterate. SoniType provides the first *formal, automated* alternative: specify, type-check, optimize, and evaluate—with theorems guaranteeing (relative to the psychoacoustic model) that type-checked specifications meet discriminability thresholds. This does not replace listening studies (as we honestly acknowledge) but provides the formal backbone that enables systematic engineering of sonifications, analogous to how type systems do not replace testing but make whole classes of errors impossible.

The information-theoretic optimization (Theorem 1) and real-time systems (Theorem 7) aspects are supporting material for the OOPSLA submission. These yield 2–3 additional standalone publications: Theorem 1 as an information theory contribution to JASA or IEEE TASLP, the cross-model evaluation methodology as a methods paper at ICAD, and the formalized Bregman predicates (Theorem 2) as a formal-methods/auditory-science contribution.

(Amended per verification panel — adversarial bug-finding benchmark and listenable audio per non-binding recommendations) The evaluation includes an **adversarial bug-finding benchmark**: a curated set of known-bad sonification designs from the ICAD literature (designs that failed in human studies due to masking, poor discriminability, or cognitive overload). SoniType's type-checker must reject every known-bad design with a specific, actionable error message while accepting known-good designs. This is the "type systems catch real bugs" argument that PL reviewers find compelling. Additionally, **listenable audio supplementary materials** will be included—letting reviewers hear the difference between a SoniType-optimized sonification and an un-optimized baseline, which is profoundly more persuasive than $d'_\text{model}$ tables alone.

---

## Evaluation Plan

(Amended per verification panel — binding amendments #5, #6, #7: qualify all model-based metrics, add LLM baseline, add published human-data anchor)

All evaluation is fully automated and model-based with no new human subjects required. External human listening studies are outside the scope of this work.

### Primary Metrics

(Amended per verification panel — binding amendment #5: qualify all model-based metrics; use $d'_\text{model}$ throughout)

1. **Model-predicted discriminability ($d'_\text{model}$) via psychoacoustic simulation.** For each sonification, we compute the model-predicted discriminability index $d'_\text{model} = z(\text{hit rate}) - z(\text{false alarm rate})$ using signal detection theory, where hit/false alarm rates are derived from the psychoacoustic model's predicted response distributions. A sonification is "perceptually effective according to the model" if $d'_\text{model} \geq 1.0$ for all data-distinguished categories. We report $d'_\text{model}$ distributions across data conditions. **Qualification**: $d'_\text{model}$ is model-predicted discriminability, not measured human discriminability. Its validity as a proxy for human perception is bounded by the psychoacoustic model's fidelity, which we assess via published human-data anchoring (see below) and cross-model evaluation.

2. **Psychoacoustically-constrained mutual information $I_\psi(D; A)$.** Computed from the compiler's psychoacoustic model. Reported in bits. We compare SoniType-optimized mappings against baselines to demonstrate information gain. All $I_\psi$ values are conditioned on the psychoacoustic model $\psi$'s validity.

3. **Information loss bound $L(\sigma)$.** Computed from Theorem 3. Reported as bits lost due to masking, sub-JND differences, and segregation failures, as predicted by the model. We verify that SoniType-compiled mappings satisfy $L(\sigma) \leq \epsilon$ for user-specified $\epsilon$.

4. **Cross-model evaluation.** To guard against overfitting to a single psychoacoustic model, we compile with one model (e.g., Schroeder masking + Weber JND) and evaluate with an independent model (e.g., Glasberg-Moore excitation patterns + Zwicker loudness). Consistent $d'_\text{model}$ across models provides evidence of robustness. **Explicit qualification**: cross-model evaluation validates within the excitation-pattern model family. Both models share foundational psychoacoustic assumptions (critical bands, masking, Weber's law). Cross-model consistency demonstrates that results are not artifacts of a single model's idiosyncrasies, but does *not* validate against ground-truth human perception. The models represent two implementations of the same theoretical tradition, not independent epistemic sources.

5. **(Amended per verification panel — binding amendment #7: published human-data anchor) Published human-data anchoring.** We validate the psychoacoustic model's predictions against published human experimental data from classic auditory experiments. Specifically: (a) auditory streaming thresholds from Bregman's experiments and subsequent replications, (b) frequency and loudness JND data from Moore's psychoacoustic measurements, (c) sonification discriminability data from Walker & Nees (2011) guidelines and related ICAD studies. For each configuration where published human $d'$ or threshold data exists, we compute the model-predicted $d'_\text{model}$ for the identical stimulus configuration and report the correspondence. If SoniType's model predicts $d'_\text{model} = 2.1$ for a configuration where humans measured $d' = 1.9$, that validates the model. If the model predicts $d'_\text{model} = 2.1$ where humans measured $d' = 0.5$, the model is broken and we report this honestly. **Zero human subjects needed**—all comparison data is drawn from the published literature. This provides critical grounding that the model-based evaluation pipeline is not operating in a vacuum.

**Circularity disclosure.** Computing $d'_\text{model}$ using the same psychoacoustic model $M$ that the compiler optimized against creates a circularity risk: high model-predicted discriminability may reflect overfitting to model $M$ rather than genuine perceptual quality. The cross-model evaluation protocol is the primary guard against within-model-family circularity. The published human-data anchoring (metric 5) provides the critical external grounding. We frame the automated evaluation as "establishing that the compiler works correctly relative to the psychoacoustic model" (analogous to testing a compiler against a language spec), and acknowledge that "the model accurately predicts human perception" is a separate claim validated by the anchoring data and ultimately by the psychoacoustics community.

### Model-Independent Signal Metrics

(Amended per verification panel — non-binding recommendation: model-independent signal metrics)

To complement model-dependent metrics, we report signal-level measures that are independent of any psychoacoustic model:

6. **Spectral overlap ratio.** The fraction of energy in each stream that falls within the critical bands of other streams. Lower overlap implies better discriminability regardless of which psychoacoustic model one trusts.

7. **Temporal distinctiveness.** Autocorrelation and cross-correlation between stream amplitude envelopes. Streams with distinct temporal profiles are easier to segregate independently of the psychoacoustic model.

8. **Parameter space coverage.** The fraction of the available perceptual parameter space (pitch range × timbre space × panning × temporal rate) utilized by the mapping. Efficient use of the perceptual channel is measurable without a model.

### Compiler Performance Metrics

9. **Compile time.** Wall-clock time from DSL specification to renderer binary. Target: < 60 s for typical 4–8 stream sonifications.
10. **Render latency.** Measured buffer-completion time vs. WCET bound. Target: < 5 ms per buffer at 256 samples/44.1 kHz. Report WCET bound tightness (ratio of measured to predicted).
11. **Throughput.** Sustained real-time factor (ratio of audio duration produced to wall-clock time). Target: > 50× real-time for 8-stream sonifications.

### Baselines

- **SonifyR** (R package): Default parameter mappings for equivalent data types. No psychoacoustic optimization.
- **TwoTone** (web tool): Default sonification outputs for identical datasets. No customization or optimization.
- **Hand-tuned ICAD benchmarks**: Reproduce published sonification designs from ICAD proceedings (Walker & Nees 2011 guidelines, Flowers 2005 earcon study) as fixed baselines.
- **Faust/SuperCollider manual implementations**: Expert-coded sonifications for the same data, representing the current best practice with full manual control but no automated optimization.
- **(Amended per verification panel — binding amendment #6: LLM baseline) LLM-generated sonification code**: GPT-4 and Claude generating SuperCollider and Sonic Pi code from natural language specifications describing the same data and desired sonification goals. This tests whether the formal compiler adds value over AI-assisted sonification—if SoniType cannot demonstrably outperform "describe what you want to an LLM" on its own model-predicted metrics ($d'_\text{model}$, $I_\psi$, constraint satisfaction margins), the compiler's value proposition is fatally undermined. We provide the LLMs with the same data descriptions, target number of streams, and qualitative discriminability goals that SoniType receives as formal specifications.
- **Ablated SoniType variants**: (a) without psychoacoustic optimization (random feasible mappings), (b) without segregation constraints, (c) without cognitive load budgeting—isolating each contribution's effect on $d'_\text{model}$ and $I_\psi$.

### Adversarial Bug-Finding Benchmark

(Amended per verification panel — non-binding recommendation: adversarial bug-finding benchmark)

A curated suite of **known-bad sonification designs** drawn from the ICAD literature: designs that failed in published human listening studies due to masking, poor discriminability, or cognitive overload. SoniType's type-checker must reject every known-bad design with a specific, actionable error diagnostic, while accepting known-good designs from the same studies. We report precision and recall of the type-checker as a "perceptual bug detector." If 8/10 published sonification failures are caught by type-checking, that is a compelling "type systems catch real bugs" result for PL reviewers.

### Datasets

Standardized benchmark suite: (1) univariate time series (stock prices, temperature), (2) bivariate with correlation structure, (3) multivariate 4–8 channel (weather station data, physiological monitoring), (4) categorical event streams (network intrusion logs, medical alarms—including IEC 60601-1-8 alarm palette configurations). Selected for diversity in data dimensionality and sonification complexity.

### Supplementary Materials

(Amended per verification panel — non-binding recommendation: listenable audio)

All evaluation configurations will include rendered audio files as supplementary materials. Reviewers and readers can *listen* to the difference between SoniType-optimized sonifications, unoptimized baselines, LLM-generated outputs, and ablated variants. Audio is profoundly more persuasive than $d'_\text{model}$ tables for conveying perceptual quality differences.

---

## Laptop CPU Feasibility

### Compile-Time vs. Render-Time Separation

The fundamental architectural insight is that **all psychoacoustic optimization is a compile-time concern**. The optimizer searches the mapping parameter space, evaluates $I_\psi$ over candidate bindings, and selects the Pareto-optimal configuration—potentially taking seconds to minutes depending on stream count and parameter space size. This is analogous to an optimizing C compiler spending minutes on a build: the cost is amortized and paid once.

The resulting renderer is a static audio graph with pre-computed parameter bindings. At render time, it executes fixed-topology signal processing: oscillator evaluation, filter application, envelope shaping, mixing. These are the same operations that Faust and SuperCollider execute comfortably in real-time on decade-old hardware.

### Cost Estimates

**Psychoacoustic model evaluation** (single candidate mapping): Bark-scale masking is a closed-form spreading function evaluated over 24 critical bands (~2 μs). JND comparison is a scalar division (~10 ns). Segregation predicate evaluation for $k$ streams is $O(k^2)$ pairwise checks (~50 μs for 8 streams). **Total per candidate: < 100 μs.** The optimizer evaluates ~10,000–100,000 candidates during branch-and-bound, yielding 1–10 seconds of compile time for typical configurations. In worst-case scenarios with dense spectral configurations where branch-and-bound pruning degrades, compile time could reach 60 s or more; the optimizer employs a time-bounded fallback strategy, returning the best solution found so far after a configurable timeout (default 120 s).

**$I_\psi$ computation** (single mapping evaluation): Requires Monte Carlo estimation over data distribution. For low-dimensional cases (1–3 streams), ~1,000 MC samples suffice (~100 ms per mapping); for 8-stream sonifications, 5K–10K samples or a $k$-nearest-neighbor MI estimator with known convergence properties may be required (~500 ms per mapping). Evaluated ~100 times during optimization. **Total: 10–60 s depending on stream count**, trivially parallelizable across cores.

**Render-time budget** for 8 simultaneous streams at 256-sample buffers (5.8 ms deadline at 44.1 kHz):
- 8 oscillators: ~20 μs (wavetable lookup, 256 samples each)
- 8 filters (2nd-order IIR): ~15 μs
- 8 envelope generators: ~8 μs
- Mixing + panning: ~5 μs
- **Total: ~48 μs**, well under the 5.8 ms deadline (~0.8% CPU utilization)

Even scaling to 16 streams with more expensive synthesis (FM, granular) stays under 200 μs—3.4% of the buffer deadline. **No GPU required.** The render-time computation is embarrassingly cheap; the intellectual contribution is entirely in the compile-time optimization that determines *what* to render.

### Hardware Baseline

All performance targets assume a 2020-era laptop CPU (e.g., Apple M1, Intel i5-1135G7). The static WCET cost estimation is parameterized by a hardware cost model that can be recalibrated for specific targets, but the margins are so large (50–100× headroom) that even conservative models confirm feasibility on any machine capable of running a web browser. (This is static cost estimation with large safety margins, not formal WCET analysis in the abstract-interpretation sense; the substantial headroom makes such formal analysis unnecessary for this application.)

---

*This document represents the amended crystallized problem statement for the SoniType project, incorporating all 7 binding amendments from the verification panel (V4/D6/BP5/L7.5). The novel core is ~48K LoC of genuine intellectual contribution; total scope is capped at ~95K LoC with a minimum viable compiler of ~50K LoC. The system targets OOPSLA with the perceptual type system as the lead PL contribution. All model-based evaluation claims are explicitly qualified and anchored against published human data. The primary risk is type system substantiation—if the formal typing rules and soundness proof cannot be carried to OOPSLA depth, the fallback is a constraint verification framework targeting ICAD/SPLASH workshops.*
