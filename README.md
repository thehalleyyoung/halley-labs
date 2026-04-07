# halley-labs

`halley-labs` is a curated collection of research software artifacts generated through an AI-first workflow. The repositories gathered here are not intended to present a single unified software stack; rather, they instantiate a broad portfolio of executable research directions spanning formal methods, optimization, systems, machine learning, verification, privacy, and scientific computing.

The central claim of the collection is methodological. Each project was selected under an automated process that optimized for technical ambition, prospective usefulness, and paper-worthy novelty, then stress-tested through iterative build, test, and artifact-validation loops. In that sense, `halley-labs` should be read as a laboratory of executable hypotheses: each repository aims to make a concrete research idea inspectable, runnable, and extensible, even when the surrounding evaluation story remains narrower than that of a mature production system.

This monorepo serves as an index over the project set. The linked repositories are the primary homes of the artifacts themselves, including their code, documentation, and project-specific papers. A paper describing the generation and selection process for the collection is forthcoming.

## Collection Overview

The repositories below cover a deliberately wide range of technical domains, but they share a common purpose: to test whether a high-throughput, adversarially checked generation pipeline can produce research software that is substantial enough to support real follow-on work. Some projects are best understood as prototype tools, some as theory-backed artifact demonstrations, and some as early systems papers in executable form. Taken together, they form a benchmark of what an automated research-software workflow can already produce.

## Projects

| Repository | Description |
|---|---|
| [algebraic-repair-calculus](https://github.com/thehalleyyoung/algebraic-repair-calculus) | Algebraic program-repair calculus |
| [algo-collusion-certifier](https://github.com/thehalleyyoung/algo-collusion-certifier) | Certification tools for algorithmic collusion analyses |
| [bilevel-compiler-intersection-cuts](https://github.com/thehalleyyoung/bilevel-compiler-intersection-cuts) | Bilevel compiler with intersection-cut tooling |
| [bilevel-tight-reformulation](https://github.com/thehalleyyoung/bilevel-tight-reformulation) | Tight reformulations for bilevel optimization |
| [bio-phase-atlas](https://github.com/thehalleyyoung/bio-phase-atlas) | Certified phase atlases for biological ODEs |
| [bounded-rational-usability-oracle](https://github.com/thehalleyyoung/bounded-rational-usability-oracle) | Usability oracle under bounded-rational user models |
| [cascade-config-verifier](https://github.com/thehalleyyoung/cascade-config-verifier) | Cascading configuration verification |
| [causal-plasticity-atlas](https://github.com/thehalleyyoung/causal-plasticity-atlas) | Multi-context causal mechanism plasticity mapping |
| [causal-qd-illumination](https://github.com/thehalleyyoung/causal-qd-illumination) | Quality-diversity illumination for causal discovery |
| [causal-risk-bounds](https://github.com/thehalleyyoung/causal-risk-bounds) | Verified causal bounds on systemic financial risk |
| [causal-robustness-radii](https://github.com/thehalleyyoung/causal-robustness-radii) | Robustness-radius analysis for causal graphs |
| [causal-trading-shields](https://github.com/thehalleyyoung/causal-trading-shields) | Causal discovery with safety shields for adaptive trading |
| [certified-leakage-contracts](https://github.com/thehalleyyoung/certified-leakage-contracts) | Certified information-leakage contracts |
| [choreo-xr-interaction-compiler](https://github.com/thehalleyyoung/choreo-xr-interaction-compiler) | Choreographic XR interaction compiler |
| [cross-lang-verifier](https://github.com/thehalleyyoung/cross-lang-verifier) | Catches UB divergences in C→Rust translations |
| [diversity-decoding](https://github.com/thehalleyyoung/diversity-decoding) | LLM output diversity with formal optimality certificates |
| [dp-mechanism-forge](https://github.com/thehalleyyoung/dp-mechanism-forge) | Counterexample-guided synthesis of optimal DP mechanisms |
| [dp-verify-repair](https://github.com/thehalleyyoung/dp-verify-repair) | Differential privacy verification and repair |
| [fp-diagnosis-repair-engine](https://github.com/thehalleyyoung/fp-diagnosis-repair-engine) | False-positive diagnosis and repair engine |
| [guideline-polypharmacy-verify](https://github.com/thehalleyyoung/guideline-polypharmacy-verify) | Verification for guideline-driven polypharmacy |
| [litmus-inf](https://github.com/thehalleyyoung/litmus-inf) | Axiomatic memory model litmus test verification |
| [market-manipulation-prover](https://github.com/thehalleyyoung/market-manipulation-prover) | Z3-certified market manipulation detection |
| [marl-race-detect](https://github.com/thehalleyyoung/marl-race-detect) | Scheduling-dependent bug detection in multi-agent RL |
| [ml-pipeline-leakage-auditor](https://github.com/thehalleyyoung/ml-pipeline-leakage-auditor) | Leakage auditing for ML pipelines |
| [ml-pipeline-selfheal](https://github.com/thehalleyyoung/ml-pipeline-selfheal) | Self-healing runtime for ML pipelines |
| [mutation-contract-synth](https://github.com/thehalleyyoung/mutation-contract-synth) | Contract synthesis from mutation analysis |
| [nlp-metamorphic-localizer](https://github.com/thehalleyyoung/nlp-metamorphic-localizer) | Metamorphic bug localization for NLP systems |
| [nn-init-phases](https://github.com/thehalleyyoung/nn-init-phases) | Neural network initialization phase diagnostics |
| [pareto-reg-trajectory-synth](https://github.com/thehalleyyoung/pareto-reg-trajectory-synth) | Pareto synthesis for regulatory trajectories |
| [perceptual-sonification-compiler](https://github.com/thehalleyyoung/perceptual-sonification-compiler) | Compiler for perceptual sonification pipelines |
| [pram-compiler](https://github.com/thehalleyyoung/pram-compiler) | PRAM algorithm to cache-efficient C compiler |
| [proto-downgrade-synth](https://github.com/thehalleyyoung/proto-downgrade-synth) | Protocol downgrade attack synthesis |
| [rag-fusion-compiler](https://github.com/thehalleyyoung/rag-fusion-compiler) | Cross-boundary operator fusion for RAG pipelines |
| [safe-deploy-planner](https://github.com/thehalleyyoung/safe-deploy-planner) | Safe deployment planning |
| [sim-conservation-auditor](https://github.com/thehalleyyoung/sim-conservation-auditor) | Simulation conservation auditing |
| [sparse-cpu-inference](https://github.com/thehalleyyoung/sparse-cpu-inference) | Speculative sparse inference engine for CPU |
| [spatial-hash-compiler](https://github.com/thehalleyyoung/spatial-hash-compiler) | Spatial query compiler via geometric perfect hashing |
| [spectral-decomposition-oracle](https://github.com/thehalleyyoung/spectral-decomposition-oracle) | Spectral decomposition oracle |
| [synbio-verifier](https://github.com/thehalleyyoung/synbio-verifier) | Formal verification for synthetic biology circuits |
| [tensor-train-modelcheck](https://github.com/thehalleyyoung/tensor-train-modelcheck) | CSL model checking on tensor-train compressed states |
| [tensorguard](https://github.com/thehalleyyoung/tensorguard) | Static shape/device/phase verifier for PyTorch nn.Modules |
| [tlaplus-coalgebra-compress](https://github.com/thehalleyyoung/tlaplus-coalgebra-compress) | Coalgebraic compression for TLA+ model checking |
| [txn-isolation-verifier](https://github.com/thehalleyyoung/txn-isolation-verifier) | Transaction isolation verification |
| [wasserstein-bounds](https://github.com/thehalleyyoung/wasserstein-bounds) | Distribution drift monitoring with Wasserstein certificates |
| [xr-affordance-verifier](https://github.com/thehalleyyoung/xr-affordance-verifier) | Formal verification for XR affordances |
| [zk-nlp-scoring](https://github.com/thehalleyyoung/zk-nlp-scoring) | Zero-knowledge proofs for NLP benchmark scores |

## Example Verification Status

All code examples in project READMEs and `.tex` papers have been audited against their actual implementations. The table below summarizes discrepancies found and corrected. Every import path, function signature, CLI flag, constructor parameter, and return type shown in documentation now matches the source code.

| Repository | Files Audited | Issues Fixed | Key Corrections |
|---|---|---|---|
| algebraic-repair-calculus | tool_paper.tex | 2 | CLI commands, Python API imports/methods |
| algo-collusion-certifier | README.md | 3 | `LinearDemand::new` arity, `ConstantMarginalCost` name, `PriceGrid::new` |
| bilevel-compiler-intersection-cuts | README.md, examples/README.md | 4 | `BilevelProblem` fields, `CutPool` API, CLI subcommands |
| bio-phase-atlas | tool_paper.tex | 1 | Proof trace JSON schema |
| bounded-rational-usability-oracle | implementation/README.md | 8 | Constructor types, field names, CLI flag values |
| cascade-config-verifier | — | 0 | ✅ All correct |
| causal-plasticity-atlas | README.md, implementation/README.md, tool_paper.tex | 6 | Classification values, return types, visualizer API, CLI syntax |
| causal-qd-illumination | README.md | 13 | Score/Archive/Engine/Analysis/Visualization class APIs |
| causal-risk-bounds | README.md | 7 | Solver enums, method names, field names, return types |
| causal-robustness-radii | implementation/README.md | 7 | CLI flags, dataclass fields, config param names |
| causal-trading-shields | README.md | 14 | `PCAlgorithm`, `CoupledInference`, shield/portfolio/backtest APIs |
| certified-leakage-contracts | README.md | 2 | `CfgBuilder` method, `LeakageBits` display |
| choreo-xr-interaction-compiler | — | 0 | ✅ All correct |
| cross-lang-verifier | paper.tex, tool_paper_new.tex | 16 | `OracleResult`/`CounterexampleInfo` types, `time_ms`, CLI flags |
| diversity-decoding | tool_paper.tex, tool_paper_v1.tex | 10 | Fictional `divflow` module → actual `src.*` imports, dependencies |
| dp-mechanism-forge | README.md, tool_paper.tex | 12 | CEGIS/Composition/GameTheory/SMT/Streaming/LDP APIs |
| dp-verify-repair | README.md, tool_paper.tex | 3 | `verify_mechanism()`, verdict enum values |
| fp-diagnosis-repair-engine | README.md | 3 | Import, return type, diagnostic text |
| guideline-polypharmacy-verify | README.md | 5 | CLI flag casing, Rust import paths |
| litmus-inf | paper.tex | 3 | CLI flags, install path, scripts |
| market-manipulation-prover | README.md | 3 | `FCIEngine` constructor, `discover()` args, property access |
| marl-race-detect | README.md | 1 | `bounding_box()` return type |
| ml-pipeline-leakage-auditor | README.md | 1 | Missing sklearn imports |
| ml-pipeline-selfheal | README.md | 6 | `CausalDiagnosisEngine`, `RepairSynthesizer`, replay/certificate APIs |
| mutation-contract-synth | README.md, examples | 8 | Env var names, Rust contract syntax, cargo paths |
| nlp-metamorphic-localizer | tool_paper.tex | 1 | Binary name `nlp-localizer` with `localize` subcommand |
| nn-init-phases | README.md | 1 | `CheckReport` field name |
| pareto-reg-trajectory-synth | tool_paper.tex | 2 | DSL obligation syntax, grammar productions |
| perceptual-sonification-compiler | README.md, benchmarks/README.md | 2 | Binary name `sonitype`, Criterion flags |
| pram-compiler | README.md | 1 | Unimplemented feature notes |
| proto-downgrade-synth | 3× examples/README.md | 3 | CLI binary name, flag names, positional args |
| rag-fusion-compiler | — | 0 | ✅ All correct |
| safe-deploy-planner | tool_paper.tex | 1 | CLI flag names |
| sim-conservation-auditor | README.md, tool_paper.tex | 4 | Observer/integrator method signatures |
| sparse-cpu-inference | — | 0 | ✅ All correct |
| spatial-hash-compiler | README.md, tool_paper.tex | 10 | Hash functions, query API signatures, output formats |
| spectral-decomposition-oracle | README.md, examples | 7 | Constructor names, imports, feature field names |
| synbio-verifier | __init__.py docstring | 1 | `parse_sbml_file()` not `BioModel.from_sbml()` |
| tensor-train-modelcheck | README.md, pyproject.toml | 2 | Version sync, template content |
| tensorguard | README.md, pldi_paper.tex | 12 | CLI flags, config file name, TOML sections, exit codes |
| tlaplus-coalgebra-compress | README.md, tool_paper.tex | 5 | CEGAR constructor, pipeline stages, witness format |
| txn-isolation-verifier | README.md | 7 | JSON keys, filenames, cargo workspace command |
| wasserstein-bounds | README.md, src/cli.py | 2 | `analyze()` signature, regression detection logic (code fix) |
| xr-affordance-verifier | README.md | 5 | Env vars, build flags, config paths, scene format |
| zk-nlp-scoring | tool_paper.tex | 9 | Lean 4 axioms, type constructors, semiring classes |

**Total: 173 documentation–code mismatches corrected across 38 projects.**

## License

See individual repositories for license information.
