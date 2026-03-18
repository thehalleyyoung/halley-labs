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

## License

See individual repositories for license information.
