# CABER: Coalgebraic Behavioral Auditing of Foundation Models

Approximate behavioral testing of black-box LLMs, inspired by coalgebraic automata theory. Given only API access, CABER learns finite-state behavioral automata, model-checks temporal safety properties, and quantifies behavioral drift via Kantorovich bisimulation distance.

**This is approximate behavioral testing, not formal verification.** The coalgebraic framework provides structural advantages, but guarantees are approximate (see [Limitations](#important-limitations)).

## Quickstart (30 seconds)

```bash
cd implementation

# Mock LLM validation — no API key needed, ~10s
python3 phase0_experiments.py

# Build Rust core + run 38 property-based tests
cargo build --release && cargo test -p caber-integration --test property_tests

# Full experiments with embedding classifier (requires OPENAI_API_KEY, ~$5)
source ~/.bashrc && python3 pathb_deep_experiments.py

# Or the v2 analysis on existing data (no new API calls, uses cached embeddings)
python3 pathb_deep_experiments_v2.py
```

## Key Results

### 1. Semantic Embedding Classifier (135% improvement over keyword baseline)

Replacing keyword-based classification with `text-embedding-3-small` nearest-centroid achieves **0.63 cross-configuration accuracy** vs 0.27 for keyword (135% relative improvement):

| Train → Test | Embedding | Keyword |
|---|---|---|
| safety → creative | **0.80** | 0.27 |
| safety → rigid | **0.58** | 0.27 |
| creative → safety | **0.73** | 0.27 |
| Mean (6 pairs) | **0.63** | 0.27 |

### 2. Structural Advantage over Chi-Squared

In **7/12 multi-turn configuration pairs**, CABER's transition distance detects temporal behavioral patterns that chi-squared marginal tests miss entirely. Examples: escalation timing differences, behavioral drift under pushback, entropy rate divergence.

### 3. Calibration via Platt Scaling (115× improvement)

Platt scaling reduces ECE from 0.475 to 0.004 (~115×), bringing calibration well below 0.10. Graded satisfaction scores are now reliable probability estimates.

### 4. Compositional Specifications

CABER evaluates conjunctive properties (safety ∧ helpfulness) and temporal properties (consistency under pushback) inexpressible as single frequency tests.

## Important Limitations

- **Not formal verification.** CABER is approximate behavioral testing. Do not rely on it for safety-critical certification.
- **Single model only**: all experiments use `gpt-4.1-nano` (non-frontier). Cross-model validation needed.
- **PAC bounds vacuous** at operating sample sizes (54–90 vs ~143K required for ε=0.05). Use Bayesian posteriors.
- **Classifier generalization improved but not solved**: embedding classifier achieves 0.63 cross-config (vs 0.27 keyword), but per-prompt-type LOPO remains low. More training data needed.
- **Non-functorial abstraction**: 0–27% inconsistency rate. Approximate preservation bound: |φ̂ - φ*| ≤ 0.077 (instruction-rigid).
- **Paper proofs only**: no Lean 4/Coq mechanization. 38 property-based tests validate invariants.
- **Calibration**: Now 0.004 ECE after Platt scaling (improved from 0.28–0.73).

## Paper

Technical paper: `implementation/tool_paper.tex` (compile with `pdflatex`).

Claim-to-evidence mapping: `grounding.json`.
