# halley-labs

`halley-labs` is a monorepo index for a large collection of research-software projects. Each top-level project now lives in its own Git repository and is linked here as a submodule-style gitlink, so the monorepo acts as a catalog and coordination layer rather than the primary home for each codebase.

## Cloning

Clone with submodules so every project checkout is present:

```bash
git clone --recurse-submodules https://github.com/thehalleyyoung/halley-labs.git
```

If you already cloned the monorepo, fetch the linked project repos with:

```bash
git submodule update --init --recursive
```

## Layout

- The repository root tracks project links in `.gitmodules`.
- Each project has its own GitHub repository under `thehalleyyoung`.
- Build instructions, papers, and validation details live in the individual project repositories.

## Projects

| Project | GitHub |
|---|---|
| `algebraic-repair-calculus` | <https://github.com/thehalleyyoung/algebraic-repair-calculus> |
| `algo-collusion-certifier` | <https://github.com/thehalleyyoung/algo-collusion-certifier> |
| `bilevel-compiler-intersection-cuts` | <https://github.com/thehalleyyoung/bilevel-compiler-intersection-cuts> |
| `bilevel-tight-reformulation` | <https://github.com/thehalleyyoung/bilevel-tight-reformulation> |
| `bio-phase-atlas` | <https://github.com/thehalleyyoung/bio-phase-atlas> |
| `bounded-rational-usability-oracle` | <https://github.com/thehalleyyoung/bounded-rational-usability-oracle> |
| `cascade-config-verifier` | <https://github.com/thehalleyyoung/cascade-config-verifier> |
| `causal-plasticity-atlas` | <https://github.com/thehalleyyoung/causal-plasticity-atlas> |
| `causal-qd-illumination` | <https://github.com/thehalleyyoung/causal-qd-illumination> |
| `causal-risk-bounds` | <https://github.com/thehalleyyoung/causal-risk-bounds> |
| `causal-robustness-radii` | <https://github.com/thehalleyyoung/causal-robustness-radii> |
| `causal-trading-shields` | <https://github.com/thehalleyyoung/causal-trading-shields> |
| `certified-leakage-contracts` | <https://github.com/thehalleyyoung/certified-leakage-contracts> |
| `choreo-xr-interaction-compiler` | <https://github.com/thehalleyyoung/choreo-xr-interaction-compiler> |
| `cross-lang-verifier` | <https://github.com/thehalleyyoung/cross-lang-verifier> |
| `diversity-decoding` | <https://github.com/thehalleyyoung/diversity-decoding> |
| `dp-mechanism-forge` | <https://github.com/thehalleyyoung/dp-mechanism-forge> |
| `dp-verify-repair` | <https://github.com/thehalleyyoung/dp-verify-repair> |
| `fp-diagnosis-repair-engine` | <https://github.com/thehalleyyoung/fp-diagnosis-repair-engine> |
| `guideline-polypharmacy-verify` | <https://github.com/thehalleyyoung/guideline-polypharmacy-verify> |
| `litmus-inf` | <https://github.com/thehalleyyoung/litmus-inf> |
| `market-manipulation-prover` | <https://github.com/thehalleyyoung/market-manipulation-prover> |
| `marl-race-detect` | <https://github.com/thehalleyyoung/marl-race-detect> |
| `ml-pipeline-leakage-auditor` | <https://github.com/thehalleyyoung/ml-pipeline-leakage-auditor> |
| `ml-pipeline-selfheal` | <https://github.com/thehalleyyoung/ml-pipeline-selfheal> |
| `mutation-contract-synth` | <https://github.com/thehalleyyoung/mutation-contract-synth> |
| `nlp-metamorphic-localizer` | <https://github.com/thehalleyyoung/nlp-metamorphic-localizer> |
| `nn-init-phases` | <https://github.com/thehalleyyoung/nn-init-phases> |
| `pareto-reg-trajectory-synth` | <https://github.com/thehalleyyoung/pareto-reg-trajectory-synth> |
| `perceptual-sonification-compiler` | <https://github.com/thehalleyyoung/perceptual-sonification-compiler> |
| `pram-compiler` | <https://github.com/thehalleyyoung/pram-compiler> |
| `proto-downgrade-synth` | <https://github.com/thehalleyyoung/proto-downgrade-synth> |
| `rag-fusion-compiler` | <https://github.com/thehalleyyoung/rag-fusion-compiler> |
| `safe-deploy-planner` | <https://github.com/thehalleyyoung/safe-deploy-planner> |
| `sim-conservation-auditor` | <https://github.com/thehalleyyoung/sim-conservation-auditor> |
| `sparse-cpu-inference` | <https://github.com/thehalleyyoung/sparse-cpu-inference> |
| `spatial-hash-compiler` | <https://github.com/thehalleyyoung/spatial-hash-compiler> |
| `spectral-decomposition-oracle` | <https://github.com/thehalleyyoung/spectral-decomposition-oracle> |
| `synbio-verifier` | <https://github.com/thehalleyyoung/synbio-verifier> |
| `tensor-train-modelcheck` | <https://github.com/thehalleyyoung/tensor-train-modelcheck> |
| `tensorguard` | <https://github.com/thehalleyyoung/tensorguard> |
| `tlaplus-coalgebra-compress` | <https://github.com/thehalleyyoung/tlaplus-coalgebra-compress> |
| `txn-isolation-verifier` | <https://github.com/thehalleyyoung/txn-isolation-verifier> |
| `wasserstein-bounds` | <https://github.com/thehalleyyoung/wasserstein-bounds> |
| `xr-affordance-verifier` | <https://github.com/thehalleyyoung/xr-affordance-verifier> |
| `zk-nlp-scoring` | <https://github.com/thehalleyyoung/zk-nlp-scoring> |

## License

See each linked project repository for its own license and usage terms.
